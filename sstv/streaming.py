"""Decode a transmission while it is still arriving.

Waiting for a recording to finish before showing anything means staring at an
empty panel for up to five minutes on the slower modes.  The picture is built
line by line from the start, so it can be shown line by line too.

How it works
------------
The frequency track is extended block by block: each new piece of audio is
demodulated on its own and appended.  Demodulating separately is safe here
because the down-mixing discriminator in :mod:`sstv.dsp` carries no state between
samples -- the phase is measured within the block -- so a stitched track differs
from one pass over the whole recording by about half a hertz on average, which is
a quarter of a pixel level.  Measured on a Robot 36 transmission, the pictures are
indistinguishable.

Running cumulative sums are kept alongside the track, so reading a scan is a
subtraction rather than a fresh pass over the accumulated audio: the cost of an
update depends on how much audio arrived, not on how long the transmission has
been running.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import dsp
from . import vis as vis_module
from .modes import (
    BLACK_HZ,
    WHITE_HZ,
    Channel,
    ModeSpec,
    mode_by_vis,
)

__all__ = ["StreamingDecoder", "StreamingProgress"]

ProgressCallback = Callable[["StreamingProgress"], None]
ImageCallback = Callable[[np.ndarray], None]

# A scan is only read once its end is at least this far behind the newest sample,
# so the tail of a block never truncates the last scan of a line.
SAFETY_S = 0.05


@dataclass
class StreamingProgress:
    """What the decoder knows after an update."""

    seconds: float = 0.0
    mode: ModeSpec | None = None
    vis_code: int | None = None
    vis_confidence: float = 0.0
    header_end: float = 0.0
    lines_decoded: int = 0
    total_lines: int = 0
    frequency_offset_hz: float = 0.0
    finished: bool = False

    @property
    def coverage(self) -> float:
        if not self.total_lines:
            return 0.0
        return self.lines_decoded / self.total_lines

    @property
    def complete(self) -> bool:
        return self.total_lines > 0 and self.lines_decoded >= self.total_lines


class StreamingDecoder:
    """Build a picture from audio as it arrives.

    Feed it each new block with :meth:`feed`; it reports progress and hands back
    a partial image whenever new lines have been completed.  :meth:`result`
    returns a :class:`~sstv.decoder.DecodeResult` shaped like the offline
    decoder's, so callers can treat the two interchangeably.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        mode: ModeSpec | str | None = None,
        *,
        progress: ProgressCallback | None = None,
        on_image: ImageCallback | None = None,
        image_every: int = 4,
    ) -> None:
        from .modes import get_mode

        self.sample_rate = int(sample_rate)
        self.requested_mode = get_mode(mode) if isinstance(mode, str) else mode
        self.progress = progress
        self.on_image = on_image
        self.image_every = max(1, int(image_every))

        self._mode: ModeSpec | None = None
        self._header = None
        self._offset = 0.0
        self._track = np.zeros(0, dtype=np.float64)
        self._cumsum = np.zeros(1, dtype=np.float64)
        self._samples = 0
        self._next_line = 0
        self._lines = 0
        self._planes = None
        self._notes: list[str] = []
        # For a signal with no VIS header the mode has to be worked out from the
        # line rhythm; that is expensive, so it is attempted once.
        self._gave_up = False
        self._guess_tried = False
        self._header_end = 0.0

    # -- state ------------------------------------------------------------
    @property
    def mode(self) -> ModeSpec | None:
        return self._mode

    @property
    def seconds(self) -> float:
        """How much audio has been fed, in seconds."""
        return self._samples / float(self.sample_rate)

    @property
    def lines_decoded(self) -> int:
        return self._lines

    @property
    def notes(self) -> list[str]:
        return list(self._notes)

    @property
    def frequency_offset_hz(self) -> float:
        return self._offset

    # -- input ------------------------------------------------------------
    def feed(self, samples) -> StreamingProgress:
        """Add the next block of audio and decode whatever is now complete."""
        data = np.asarray(samples, dtype=np.float64)
        if data.size == 0:
            return self._progress()
        if data.ndim > 1:
            data = data.reshape(-1)

        # Demodulate this block on its own and append it to the track.
        block = dsp.instantaneous_frequency(data, self.sample_rate)
        self._append(block)
        self._samples += data.size

        self._identify()
        if self._mode is not None:
            self._decode_ready_lines()
        result = self._progress()
        # Reported on every update, including before the mode is known, so a
        # caller can show "listening..." and a running duration from the start.
        if self.progress is not None:
            try:
                self.progress(result)
            except Exception:
                pass
        return result

    def finalize(self) -> StreamingProgress:
        """Decode the tail after the audio has stopped.

        The last line of a transmission ends on the final sample, so the safety
        margin that normally holds a line back would leave it undecoded forever.
        Once no more audio is coming, a line is accepted if nearly all of it has
        arrived; what is missing is at most a few pixels of one line.
        """
        mode = self._mode
        if mode is None or self._planes is None:
            return self._progress()

        header_end = (self._header.end if self._header is not None
                      else self._header_end)
        available = self.seconds
        while self._next_line < mode.line_count:
            base = header_end + self._next_line * mode.line_seconds
            if base + mode.line_seconds > available:
                # Incomplete, but accept it when nearly all of the line is here.
                if available - base < mode.line_seconds * 0.9:
                    break
            self._read_line(self._planes, mode, base, self._next_line)
            self._next_line += 1
            self._lines += 1

        if self.on_image is not None:
            self.on_image(self.to_image())
        result = self._progress()
        result.finished = True
        if self.progress is not None:
            try:
                self.progress(result)
            except Exception:
                pass
        return result

    def _append(self, block: np.ndarray) -> None:
        if block.size == 0:
            return
        self._track = np.concatenate((self._track, block))
        # Running total, so a scan can be averaged by subtracting two entries.
        self._cumsum = np.concatenate(
            (self._cumsum, self._cumsum[-1] + np.cumsum(block))
        )

    # -- mode identification ---------------------------------------------
    def _identify(self) -> None:
        if self._mode is not None or self._header is not None or self._gave_up:
            return
        # The header is 910 ms; nothing can be read before it has all arrived.
        if self.seconds < 0.95:
            return
        from .decoder import SstvDecoder

        window = self._track[: int(self.sample_rate * 1.2)]
        header = vis_module.decode_vis(window, self.sample_rate, track=window)
        if header is not None and not SstvDecoder._header_is_plausible(header):
            self._notes.append(
                f"ignored a doubtful VIS header (code {header.code}, confidence "
                f"{header.confidence * 100:.0f}%)")
            header = None
        if header is not None:
            mode = self.requested_mode or mode_by_vis(header.code)
            if mode is not None:
                self._adopt(mode, header)
                return
            self._notes.append(
                f"VIS code {header.code} is not a mode this program implements")

        # No usable header.  Once there is enough audio for the line rhythm to be
        # recognisable, work the mode out from the sync pulses instead of giving up:
        # a transmission already in progress when listening started never sends its
        # header at all, and refusing to decode it would be no use.  Tried once, at
        # a fixed point, rather than on every update -- it is the expensive part of
        # live decoding and the answer does not improve with more audio.
        if self.requested_mode is not None:
            self._adopt(self.requested_mode, None)
            return
        if self.seconds < 8.0:
            return
        if self._guess_tried:
            self._gave_up = True
            return
        self._guess_tried = True
        mode, start, confidence, candidates = SstvDecoder._infer_mode(
            self._track, self.sample_rate, self._notes)
        if mode is None:
            self._gave_up = True
            self._notes.append(
                "no VIS header and no recognisable line rhythm; nothing to decode")
            return
        self._adopt(mode, None)
        self.guessed = True
        self.guess_confidence = confidence
        self.guess_candidates = candidates
        # The inferred grid describes where lines begin in the audio already
        # collected, so it becomes the origin that line decoding counts from.
        self._header_end = start
        self._notes.append(
            f"no VIS header used; inferred {mode.name} from the sync pulse rhythm "
            f"(confidence {confidence * 100:.0f}%)")

    def _adopt(self, mode: ModeSpec, header) -> None:
        self._mode = mode
        self._header = header
        self.guessed = header is None and self.requested_mode is None
        # Measure the tuning error from the header's leader tone, a known 1900 Hz, so
        # brightness errors can be corrected from the start.  Without a header there
        # is nothing trustworthy to measure against, so no correction is made: the
        # sync pulse reads tens of hertz high depending on where in it one looks, and
        # "correcting" by that moves every scan sideways.
        if header is None:
            return
        leader = self._average_hz(0.15, 0.28)
        if abs(leader - 1900.0) < 400.0:
            self._offset = leader - 1900.0
            if abs(self._offset) >= 1.0:
                self._notes.append(f"corrected a {self._offset:+.0f} Hz tuning offset")

    def _average_hz(self, start: float, end: float) -> float:
        a = max(0, int(start * self.sample_rate))
        b = min(self._track.size, max(a + 1, int(end * self.sample_rate)))
        if b <= a:
            return 1900.0
        return float(np.mean(self._track[a:b]))

    # -- line decoding ----------------------------------------------------
    def _decode_ready_lines(self) -> None:
        from .decoder import _Planes

        mode = self._mode
        assert mode is not None
        if self._planes is None:
            self._planes = _Planes(mode)
        planes = self._planes
        header_end = (self._header.end if self._header is not None
                      else self._header_end)
        available = self.seconds - SAFETY_S
        line_seconds = mode.line_seconds
        changed = 0

        while self._next_line < mode.line_count:
            base = header_end + self._next_line * line_seconds
            if base + line_seconds > available:
                break
            self._read_line(planes, mode, base, self._next_line)
            self._next_line += 1
            self._lines += 1
            changed += 1

        if changed and self.on_image is not None:
            if (self._lines % self.image_every == 0
                    or self._lines >= mode.line_count):
                self.on_image(self.to_image())

    def _read_line(self, planes, mode: ModeSpec, base: float, line_index: int) -> None:
        from .decoder import _SCAN_MARGIN_S, SstvDecoder

        segments = self._resolved_segments(mode, line_index)
        luminance_segments = [s for s in segments if s.channel is Channel.LUMINANCE]
        rows = ([line_index * 2, line_index * 2 + 1]
                if len(luminance_segments) == 2 else [line_index])
        luminance_index = 0
        offset = 0.0
        for seg in segments:
            if seg.is_image:
                count = SstvDecoder._decoded_pixels(mode, seg)
                margin = min(_SCAN_MARGIN_S, seg.seconds * 0.2)
                values = (self._slice(base + offset + margin,
                                      base + offset + seg.seconds - margin, count)
                          if count else np.zeros(0))
                if seg.channel is Channel.LUMINANCE:
                    row = rows[min(luminance_index, len(rows) - 1)]
                    luminance_index += 1
                    planes.scatter(Channel.LUMINANCE, [row], values)
                else:
                    planes.scatter(seg.channel, rows, values)
            offset += seg.seconds

    def _resolved_segments(self, mode: ModeSpec, line_index: int):
        """Resolve Robot 36's alternating chroma, as the offline decoder does."""
        plan = list(mode.line_plan())
        if not mode.chroma_on_even_only:
            return plan
        is_even = (line_index % 2) == 0
        out = []
        seen = False
        for seg in plan:
            if seg.channel in (Channel.CR, Channel.CB) and not seen:
                seg = type(seg)(Channel.CR if is_even else Channel.CB,
                                seg.milliseconds, seg.pixels, seg.black_hz, seg.white_hz)
                seen = True
            out.append(seg)
        return out

    def _slice(self, start: float, end: float, count: int) -> np.ndarray:
        """Average the track over each pixel interval, in one subtraction."""
        if count <= 0:
            return np.zeros(0, dtype=np.float64)
        size = self._track.size
        if size == 0:
            return np.full(count, 0.0)
        span = end - start
        if span <= 0:
            return np.full(count, self._track[min(size - 1, max(0, int(start * self.sample_rate)))])
        edges = start + np.arange(count + 1, dtype=np.float64) * (span / count)
        index = np.clip(np.round(edges * self.sample_rate).astype(np.int64), 0, size - 1)
        totals = self._cumsum[index[1:]] - self._cumsum[index[:-1]]
        widths = np.maximum(index[1:] - index[:-1], 1)
        hz = totals / widths
        if self._offset:
            hz = hz - self._offset
        scale = 255.0 / (WHITE_HZ - BLACK_HZ)
        return (hz - BLACK_HZ) * scale

    # -- output -----------------------------------------------------------
    def to_image(self):
        """The picture so far, with the lines not yet received left black."""
        if self._planes is None:
            return np.zeros((0, 0, 3), dtype=np.uint8)
        return self._planes.to_rgb()

    def _progress(self) -> StreamingProgress:
        mode = self._mode
        return StreamingProgress(
            seconds=self.seconds,
            mode=mode,
            vis_code=(self._header.code if self._header else None),
            vis_confidence=(self._header.confidence if self._header else 0.0),
            header_end=(self._header.end if self._header else 0.0),
            lines_decoded=self._lines,
            total_lines=(mode.line_count if mode else 0),
            frequency_offset_hz=self._offset,
            finished=bool(mode and self._lines >= mode.line_count),
        )

    def result(self):
        """A finished result, shaped like the offline decoder's."""
        from .decoder import DecodeResult

        image = self.to_image()
        mode = self._mode
        return DecodeResult(
            image=image,
            mode=mode,
            sample_rate=self.sample_rate,
            vis_code=(self._header.code if self._header else None),
            vis_confidence=(self._header.confidence if self._header else 0.0),
            header_end=(self._header.end if self._header else 0.0),
            lines_decoded=self._lines,
            total_lines=(mode.line_count if mode else 0),
            frequency_offset_hz=self._offset,
            notes=list(self._notes),
        )
