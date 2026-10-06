"""SSTV encoder: image in, audio out.

The encoder walks a mode's scan plan and turns each timed segment into FM audio.
Pixel values become tones directly (1500 Hz black to 2300 Hz white), the VIS
header announces which mode follows, and the phase is carried from segment to
segment so the waveform never has a discontinuity.

Everything is done with :mod:`numpy` on whole rows at a time; a 4-minute PD 290
transmission encodes in a couple of seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import colorspace
from . import dsp
from .modes import (
    BLACK_HZ,
    MID_HZ,
    Channel,
    ModeSpec,
    Segment,
    vis_plan,
)

__all__ = ["EncodeResult", "encode", "encode_to_wav", "make_test_image"]

ProgressCallback = Callable[[float], None]


@dataclass
class EncodeResult:
    """The audio for one transmission plus the facts a caller wants to show."""

    samples: np.ndarray
    sample_rate: int
    mode: ModeSpec
    duration: float

    def __len__(self) -> int:
        return int(self.samples.size)

    @property
    def peak(self) -> float:
        return float(np.max(np.abs(self.samples))) if self.samples.size else 0.0


class _Builder:
    """Accumulates ``(frequency, sample_count)`` runs for one transmission."""

    __slots__ = ("freqs", "counts", "sample_rate")

    def __init__(self, sample_rate: int) -> None:
        self.freqs: list[np.ndarray] = []
        self.counts: list[np.ndarray] = []
        self.sample_rate = sample_rate

    def add_tone(self, hz: float, milliseconds: float,
                 samples: int | None = None) -> None:
        n = samples if samples is not None else int(round(self.sample_rate * milliseconds / 1000.0))
        if n <= 0:
            return
        self.freqs.append(np.array([hz], dtype=np.float64))
        self.counts.append(np.array([int(n)], dtype=np.int64))

    def add_values(self, values: np.ndarray, milliseconds: float, segment: Segment,
                   zero_span: float = 255.0, samples: int | None = None) -> None:
        """Emit *values* (levels) as one scan segment.

        *samples* overrides the count derived from *milliseconds*.  The caller
        passes the line's exact per-segment sample allocation so that the segments
        sum to the nominal line length rather than each rounding upward.
        """
        arr = np.asarray(values, dtype=np.float64).ravel()
        if arr.size == 0 or milliseconds <= 0:
            return
        total = samples if samples is not None else int(round(self.sample_rate * milliseconds / 1000.0))
        if total <= 0:
            return
        if arr.size != total:
            # The plan and the mode agree on pixel counts; when a caller hands us
            # a differently sized array we resample rather than silently shifting
            # every following segment.
            arr = np.interp(
                np.linspace(0.0, arr.size - 1, total),
                np.arange(arr.size, dtype=np.float64),
                arr,
            )
        hz = segment.value_to_hz(np.clip(arr, 0.0, zero_span), zero_span)
        self.freqs.append(hz)
        self.counts.append(np.ones(total, dtype=np.int64))

    def render(self) -> np.ndarray:
        if not self.freqs:
            return np.zeros(0, dtype=np.float64)
        freqs = np.concatenate(self.freqs)
        counts = np.concatenate(self.counts)
        wave, _ = dsp.render_frequencies(freqs, counts, self.sample_rate)
        return wave


def _image_to_planes(image, mode: ModeSpec) -> dict[str, np.ndarray]:
    """Resize *image* to the mode's resolution and expose its colour planes."""
    from PIL import Image

    width, height = mode.resolution
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.LANCZOS)
    rgb = np.asarray(image, dtype=np.float64)
    if mode.color_space == "RGB":
        return {
            "luminance": None,
            "red": rgb[..., 0],
            "green": rgb[..., 1],
            "blue": rgb[..., 2],
        }
    y, cr, cb = colorspace.rgb_to_ycrcb(rgb)
    # Robot 36 halves chroma horizontally; every other chroma mode we support
    # keeps it at full width and therefore needs no subsampling here.
    if mode.chroma_subsample > 1 and _robot_style_subsample(mode):
        cr = colorspace.downsample(cr, mode.chroma_subsample)
        cb = colorspace.downsample(cb, mode.chroma_subsample)
    return {"luminance": y, "cr": cr, "cb": cb, "red": None, "green": None, "blue": None}


def _robot_style_subsample(mode: ModeSpec) -> bool:
    """True for modes whose chroma is genuinely below luminance resolution.

    Robot 36 sends 160 distinct chroma pixels per line spread over a 320-sample
    scan; Robot 72 and the PD family send full-width chroma.
    """
    return mode.name == "Robot 36"


def _plane_for(planes: dict[str, np.ndarray], channel: Channel):
    if channel is Channel.LUMINANCE:
        return planes["luminance"]
    if channel is Channel.RED:
        return planes["red"]
    if channel is Channel.GREEN:
        return planes["green"]
    if channel is Channel.BLUE:
        return planes["blue"]
    if channel is Channel.CR:
        return planes["cr"]
    if channel is Channel.CB:
        return planes["cb"]
    return None


def _segment_values(seg: Segment, row_index: int, planes: dict[str, np.ndarray],
                    mode: ModeSpec, pair_row: int | None) -> np.ndarray | None:
    """Pick the image data a segment carries on the given line."""
    plane = _plane_for(planes, seg.channel)
    if plane is None:
        return None
    if pair_row is not None:
        # PD modes put two luminance scans and one shared pair of chroma scans on
        # a single radio line.  Every segment of such a line is therefore keyed to
        # the first row of the pair -- using the caller's row index for chroma
        # would make one line send the colour of an unrelated row further down
        # the picture, and the top half's colour then appears throughout.
        plane = plane[pair_row]
    else:
        plane = plane[row_index]
    values = np.asarray(plane, dtype=np.float64)
    if values.ndim > 1:
        values = values.reshape(-1)
    if seg.pixels and values.size != seg.pixels:
        if values.size * 2 == seg.pixels and mode.chroma_subsample > 1:
            # Halved chroma expanded back to the transmitted sample count.
            values = colorspace.upsample(values, 2, seg.pixels)
        else:
            values = np.interp(
                np.linspace(0.0, values.size - 1, seg.pixels),
                np.arange(values.size, dtype=np.float64),
                values,
            )
    return values


def _line_segments(mode: ModeSpec, line_index: int) -> list[Segment]:
    """The scan plan for one radio line, with Robot 36 parity resolved.

    Robot 36 is the only mode whose separator tone depends on the line number:
    even lines carry R-Y and are flagged with 1500 Hz, odd lines carry B-Y and are
    flagged with 2300 Hz.
    """
    plan = list(mode.line_plan())
    if mode.chroma_on_even_only:
        is_even = (line_index % 2) == 0
        chroma = Channel.CR if is_even else Channel.CB
        resolved: list[Segment] = []
        seen_chroma = False
        for seg in plan:
            if seg.channel in (Channel.CR, Channel.CB):
                if not seen_chroma:
                    seg = Segment(chroma, seg.milliseconds, seg.pixels,
                                  seg.black_hz, seg.white_hz)
                    seen_chroma = True
            elif seg.channel is Channel.SEPARATOR and seg.tone_hz == MID_HZ:
                # The parity marker itself: 1500 Hz means R-Y follows, 2300 Hz
                # means B-Y follows.
                seg = Segment(Channel.SEPARATOR, seg.milliseconds, 0,
                              tone_hz=BLACK_HZ if is_even else 2300.0)
            resolved.append(seg)
        plan = resolved
    return plan


def encode(
    image,
    mode: ModeSpec | str,
    sample_rate: int = 48000,
    *,
    include_header: bool = True,
    amplitude: float = 0.8,
    progress: ProgressCallback | None = None,
) -> EncodeResult:
    """Turn *image* into the audio for one SSTV transmission.

    Parameters
    ----------
    image:
        Anything PIL can open, or a ``(H, W, 3)`` uint8 array.  It is resized to
        the mode's exact resolution first.
    mode:
        A :class:`~sstv.modes.ModeSpec` or its name.
    sample_rate:
        Output rate in Hz.  48000 matches QSSTV and PySSTV.
    include_header:
        Emit the VIS calibration header.  Only turn this off when producing a
        fragment for testing.
    amplitude:
        Peak level, 0-1.  SSTV is conventionally transmitted well below full
        scale; 0.8 leaves headroom and avoids clipping on cheap sound cards.
    progress:
        Called with a 0-1 fraction as lines are completed.
    """
    from .modes import get_mode

    spec = get_mode(mode) if isinstance(mode, str) else mode
    planes = _image_to_planes(image, spec)
    builder = _Builder(sample_rate)

    if include_header:
        # The header is a fixed sequence of tones; round its boundaries
        # cumulatively for the same reason the line segments are.
        plan = vis_plan(spec.vis_code)
        elapsed = 0.0
        previous = 0
        for hz, ms in plan:
            elapsed += ms
            boundary = int(round(sample_rate * elapsed / 1000.0))
            builder.add_tone(hz, ms, boundary - previous)
            previous = boundary

    two_rows = _is_pd(spec)
    # Sample counts come from the mode's cumulative line allocation, so the
    # segments of every line sum to the nominal line length exactly instead of
    # each rounding up and slowly stretching the transmission.
    allocations = spec.segment_samples(sample_rate)
    for line_index in range(spec.line_count):
        pair_row = None
        if two_rows:
            pair_row = line_index * 2
        for index, seg in enumerate(_line_segments(spec, line_index)):
            samples = allocations[index] if index < len(allocations) else None
            if not seg.is_image:
                builder.add_tone(seg.frequency, seg.milliseconds, samples)
                continue
            values = _segment_values(seg, line_index, planes, spec, pair_row)
            if values is None:
                builder.add_tone(seg.frequency, seg.milliseconds, samples)
            else:
                builder.add_values(values, seg.milliseconds, seg, samples=samples)
        if progress is not None and (line_index % 8 == 0 or line_index == spec.line_count - 1):
            progress((line_index + 1) / spec.line_count)

    wave = builder.render()
    if amplitude != 1.0:
        wave = wave * float(amplitude)
    return EncodeResult(
        samples=wave,
        sample_rate=sample_rate,
        mode=spec,
        duration=wave.size / float(sample_rate),
    )


def _is_pd(mode: ModeSpec) -> bool:
    """True when one radio line carries two image rows (the PD family)."""
    return sum(1 for seg in mode.segments if seg.channel is Channel.LUMINANCE) == 2


def encode_to_wav(
    image,
    mode: ModeSpec | str,
    path: str,
    sample_rate: int = 48000,
    **kwargs,
) -> EncodeResult:
    """Encode *image* and write it to a 16-bit WAV file at *path*."""
    from . import audio

    result = encode(image, mode, sample_rate, **kwargs)
    audio.write_wav(path, result.samples, sample_rate)
    return result


# --------------------------------------------------------------------------
# Test material
# --------------------------------------------------------------------------

def make_test_image(width: int = 320, height: int = 256,
                    label: str | None = None):
    """A colour-bar test card with a greyscale ramp and registration marks.

    Useful both for eyeballing a transmission and for round-trip tests: the ramp
    shows level errors, the bars show hue errors, and the border makes timing
    offsets obvious.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    bar_colors = [
        (255, 255, 255), (255, 255, 0), (0, 255, 255), (0, 255, 0),
        (255, 0, 255), (255, 0, 0), (0, 0, 255), (0, 0, 0),
    ]
    bar_h = max(1, height // 3)
    bar_w = width / len(bar_colors)
    for i, color in enumerate(bar_colors):
        draw.rectangle(
            [int(i * bar_w), 0, int((i + 1) * bar_w) - 1, bar_h - 1],
            fill=color,
        )

    # Greyscale ramp over the middle third.
    for x in range(width):
        level = int(255 * x / max(1, width - 1))
        draw.line([(x, bar_h), (x, 2 * bar_h - 1)], fill=(level, level, level))

    # Saturated primaries plus a mid-grey over the bottom third.
    bottom = 2 * bar_h
    third = width / 3
    draw.rectangle([0, bottom, int(third) - 1, height - 1], fill=(200, 60, 40))
    draw.rectangle([int(third), bottom, int(2 * third) - 1, height - 1], fill=(40, 160, 90))
    draw.rectangle([int(2 * third), bottom, width - 1, height - 1], fill=(128, 128, 128))

    # Registration ticks in the corners.
    tick = max(2, min(width, height) // 40)
    for x0, y0 in ((0, 0), (width - tick, 0), (0, height - tick), (width - tick, height - tick)):
        draw.rectangle([x0, y0, x0 + tick - 1, y0 + tick - 1], fill=(255, 255, 255))

    if label:
        try:
            draw.text((4, height - 12), label, fill=(255, 255, 255))
        except Exception:
            pass
    return img
