"""The waterfall display: a scrolling spectrum of the received signal.

A waterfall shows frequency across and time downwards, with brightness standing
for signal strength.  For SSTV it is the single most useful piece of feedback a
receiver can give, because the mode is literally drawn in frequency:

* a transmission appears as a **bright band** roughly 1200-2300 Hz wide, so you
  can see that a signal is present before any picture exists;
* each line shows up as a **repeating pattern of sync pulses**, and a clean,
  evenly spaced comb of them means the signal is good enough to decode;
* the VIS header at the start is a **short, unmistakable pattern** of tones, so
  the mode can usually be recognised by eye;
* noise, hum, over-deviation and clipping all look distinctly different from a
  healthy signal -- clipping, for instance, sprays energy across the whole width
  in broad bands rather than confining it to the SSTV tones.

The display is built from a rolling buffer of spectrum rows so that a live
recording can be shown as it arrives; :meth:`Waterfall.feed` is given only the
audio that has not been seen yet, so the cost of a repaint does not grow with the
length of the recording.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import dsp

__all__ = [
    "SCALE_MARKS",
    "Waterfall",
    "add_frequency_scale",
    "build_colormap",
    "render_spectrum",
]

# The SSTV band, with a little margin either side so that a signal which is off
# frequency is still visible rather than clipped at the edge.
DEFAULT_LOW_HZ = 1000.0
DEFAULT_HIGH_HZ = 2500.0

# Displayed dynamic range, in dB relative to full scale, used when auto-scaling
# is switched off.
FLOOR_DB = -85.0
CEIL_DB = 0.0

# Auto-scaling: the bottom of the scale is placed this far below the signal's
# own noise floor, and the top this far above it.  45 dB of range shows a clean
# signal with its sync pulses clearly separated from the background without
# washing the picture out.
AUTO_DB_RANGE = 45.0
AUTO_MIN_RANGE = 18.0


def build_colormap(reverse: bool = False) -> np.ndarray:
    """A 256-entry colour table, dark blue through cyan and green to red.

    Chosen because it is perceptually ordered for brightness -- the eye can rank
    two shades by how strong the signal is -- and because it is the convention
    radio operators already read waterfalls in.
    """
    stops = [
        (0.00, (4, 6, 24)),
        (0.16, (16, 40, 110)),
        (0.34, (14, 106, 168)),
        (0.50, (24, 176, 162)),
        (0.66, (120, 214, 78)),
        (0.80, (238, 206, 56)),
        (0.92, (240, 128, 36)),
        (1.00, (208, 40, 36)),
    ]
    positions = np.array([p for p, _ in stops], dtype=np.float64)
    values = np.array([c for _, c in stops], dtype=np.float64)
    ramp = np.linspace(0.0, 1.0, 256)
    table = np.empty((256, 3), dtype=np.uint8)
    for channel in range(3):
        table[:, channel] = np.interp(ramp, positions, values[:, channel]).astype(np.uint8)
    if reverse:
        table = table[::-1].copy()
    return table


def render_spectrum(
    rows: np.ndarray,
    *,
    low_hz: float,
    high_hz: float,
    bin_hz: np.ndarray | None = None,
    height: int | None = None,
    width: int | None = None,
    floor_db: float = FLOOR_DB,
    ceil_db: float = CEIL_DB,
    gain_db: float = 0.0,
    auto_range: bool = True,
    colormap: np.ndarray | None = None,
):
    """Turn spectrum rows into a finished waterfall image.

    Parameters
    ----------
    rows:
        ``(n_frames, n_bins)`` magnitudes in dB, oldest row first.
    low_hz, high_hz:
        Frequency range to show.  Bins outside are dropped, so the display can be
        zoomed without recomputing the transform.
    bin_hz:
        Centre frequency of each column of *rows*.  Defaults to evenly spaced.
    height, width:
        Output size in pixels.  Defaults to one pixel per frame per bin.
    auto_range:
        Choose the brightness window from the data itself.  Recordings arrive at
        wildly different levels -- a signal straight off a sound card can sit at
        -5 dBFS while one pulled off a tape sits at -45 -- and a fixed window
        either washes out the loud ones or renders the quiet ones as a flat
        smear.  Scaling to each signal's own noise floor makes both readable.
    """
    from PIL import Image

    if rows.size == 0:
        rows = np.full((1, 2), floor_db, dtype=np.float32)

    frame_rows = np.asarray(rows, dtype=np.float32)
    n_frames, n_bins = frame_rows.shape

    if bin_hz is None:
        bin_hz = np.linspace(low_hz, high_hz, n_bins)
    bin_hz = np.asarray(bin_hz, dtype=np.float64)

    keep = (bin_hz >= low_hz) & (bin_hz <= high_hz)
    if not keep.any():
        keep = np.ones_like(bin_hz, dtype=bool)
    selected = frame_rows[:, keep]
    selected_hz = bin_hz[keep]

    if auto_range:
        flat = selected.reshape(-1)
        floor, ceiling = np.percentile(flat, (5.0, 99.0))
        floor = float(floor)
        ceiling = max(float(ceiling), floor + AUTO_MIN_RANGE)
        # Keep the whole visible range inside the palette rather than pushing the
        # top past it, which would flatten every strong signal to one colour.
        if ceiling - floor > AUTO_DB_RANGE:
            ceiling = floor + AUTO_DB_RANGE
    else:
        floor, ceiling = float(floor_db), float(ceil_db)

    out_w = int(width) if width else max(1, selected.shape[1])
    out_h = int(height) if height else max(1, n_frames)

    # Resample the frequency axis to the requested width, then the time axis to
    # the requested height.  Doing frequency first keeps the zoom interpolation
    # one-dimensional.
    if out_w != selected.shape[1]:
        target = np.linspace(selected_hz[0], selected_hz[-1], out_w)
        source_index = np.interp(target, selected_hz, np.arange(selected_hz.size))
        lo = np.floor(source_index).astype(np.int64)
        hi = np.minimum(lo + 1, selected.shape[1] - 1)
        weight = (source_index - lo).astype(np.float32)
        selected = selected[:, lo] * (1.0 - weight) + selected[:, hi] * weight

    if out_h != selected.shape[0]:
        # Vectorised time-axis resample.  A loop over columns (the obvious
        # implementation with np.interp) costs tens of milliseconds per frame for
        # a full-width display, which is most of a 20 fps budget on its own.
        column = np.linspace(0.0, selected.shape[0] - 1, out_h)
        lower = np.floor(column).astype(np.int64)
        upper = np.minimum(lower + 1, selected.shape[0] - 1)
        blend = (column - lower).astype(np.float32)[:, None]
        selected = selected[lower] * (1.0 - blend) + selected[upper] * blend

    table = colormap if colormap is not None else build_colormap()
    span = max(1e-6, (ceiling + gain_db) - floor)
    scaled = (selected - floor) / span * 255.0
    indices = np.clip(scaled, 0.0, 255.0).astype(np.uint8)

    # Newest row on top, which is how every radio waterfall is drawn.
    rgb = table[indices[::-1]]
    return Image.fromarray(rgb, "RGB")


@dataclass
class Waterfall:
    """A rolling spectrum of a signal, fed incrementally.

    Typical use: call :meth:`feed` with whatever audio has arrived since the last
    call, then :meth:`to_image` to get something to blit.  Both are cheap: the
    transform only ever sees new samples, and the history is a fixed-size block of
    memory however long the recording runs.
    """

    sample_rate: int = 48000
    seconds: float = 12.0
    low_hz: float = DEFAULT_LOW_HZ
    high_hz: float = DEFAULT_HIGH_HZ
    frame: int = dsp.SPECTRUM_FRAME
    hop_fraction: float = dsp.SPECTRUM_HOP

    _rows: np.ndarray | None = field(default=None, repr=False)
    _bins: np.ndarray | None = field(default=None, repr=False)
    _capacity: int = field(default=0, repr=False)
    _count: int = field(default=0, repr=False)
    _leftover: np.ndarray | None = field(default=None, repr=False)
    _consumed: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        self.hop = max(1, int(round(self.frame * float(self.hop_fraction))))
        self._bins = dsp.bin_frequencies(self.frame, self.sample_rate)
        self._capacity = max(8, int(round(self.seconds * self.sample_rate / self.hop)))
        self._rows = np.full((self._capacity, self._bins.size), FLOOR_DB, dtype=np.float32)

    # -- geometry ---------------------------------------------------------
    @property
    def hop_samples(self) -> int:
        return self.hop

    @property
    def bins(self) -> np.ndarray:
        return self._bins

    @property
    def frames(self) -> int:
        """How many spectrum rows have been produced."""
        return self._count

    @property
    def history_seconds(self) -> float:
        return self._count * self.hop / float(self.sample_rate)

    def configure(self, *, seconds: float | None = None,
                  low_hz: float | None = None, high_hz: float | None = None) -> None:
        """Change the display window or span, keeping the audio already seen.

        Called from :meth:`clear` only when the history length actually changes:
        the rows are resized to fit the new window, and re-feeding nothing would
        otherwise blank a display the operator is in the middle of reading.
        """
        if low_hz is not None:
            self.low_hz = float(low_hz)
        if high_hz is not None:
            self.high_hz = float(high_hz)
        if seconds is not None and abs(seconds - self.seconds) > 1e-6:
            self.seconds = float(seconds)
            capacity = max(8, int(round(self.seconds * self.sample_rate / self.hop)))
            fresh = np.full((capacity, self._rows.shape[1]), FLOOR_DB, dtype=np.float32)
            carry = min(self._count, capacity)
            if carry:
                fresh[capacity - carry :] = self._view()[-carry:]
            self._rows = fresh
            self._capacity = capacity
            self._count = carry

    # -- input ------------------------------------------------------------
    def clear(self, *, reset_audio: bool = True) -> None:
        """Blank the display.

        *reset_audio* also forgets the audio already consumed, so the next
        :meth:`feed` starts a fresh stream; leave it False when only the
        frequency window changed and the history should be kept.
        """
        self._rows[:] = FLOOR_DB
        self._count = 0
        if reset_audio:
            self._leftover = None
            self._consumed = 0

    def feed(self, samples, sample_rate: int | None = None) -> int:
        """Add newly arrived audio.  Returns the number of rows produced.

        *samples* must be floats in ``[-1, 1]`` covering audio that has not been
        passed to :meth:`feed` before.
        """
        if sample_rate is not None and sample_rate != self.sample_rate:
            self.sample_rate = int(sample_rate)
            self._bins = dsp.bin_frequencies(self.frame, self.sample_rate)
            self._rows = np.full((self._capacity, self._bins.size), FLOOR_DB, dtype=np.float32)
            self._count = 0
            self._leftover = None

        data = np.asarray(samples, dtype=np.float64).reshape(-1)
        if data.size == 0:
            return 0

        # Prepend whatever was left over from last time so frames stay aligned to
        # the true sample grid instead of restarting at every call.
        if self._leftover is not None and self._leftover.size:
            data = np.concatenate((self._leftover, data))

        rows, hop = dsp.spectrogram(
            data, self.sample_rate, frame=self.frame, hop_fraction=self.hop_fraction
        )
        consumed = rows.shape[0] * hop
        self._leftover = data[consumed:].copy()
        if rows.size == 0:
            return 0

        produced = rows.shape[0]
        self._push(rows)
        return produced

    def feed_seconds(self, samples, sample_rate: int | None = None) -> int:
        return self.feed(samples, sample_rate)

    def _push(self, rows: np.ndarray) -> None:
        capacity = self._capacity
        count = min(rows.shape[0], capacity)
        if count < rows.shape[0]:
            rows = rows[-capacity:]
        # Roll the buffer: oldest rows leave at the top.
        if count < capacity:
            self._rows[:-count or None] = self._rows[count:]
            self._rows[-count:] = rows
        else:
            self._rows[:] = rows
        self._count = min(capacity, self._count + rows.shape[0])

    def _view(self) -> np.ndarray:
        if self._count == 0:
            return self._rows[:0]
        return self._rows[self._capacity - self._count :]

    # -- output -----------------------------------------------------------
    def to_image(self, *, width: int | None = None, height: int | None = None,
                 low_hz: float | None = None, high_hz: float | None = None,
                 gain_db: float = 0.0, auto_range: bool = True,
                 colormap: np.ndarray | None = None):
        """Render the current history as a PIL image."""
        rows = self._view()
        if rows.shape[0] == 0:
            rows = np.full((1, self._bins.size), FLOOR_DB, dtype=np.float32)
        return render_spectrum(
            rows,
            low_hz=self.low_hz if low_hz is None else low_hz,
            high_hz=self.high_hz if high_hz is None else high_hz,
            bin_hz=self._bins,
            width=width,
            height=height,
            gain_db=gain_db,
            auto_range=auto_range,
            colormap=colormap,
        )

    def peak_hz(self) -> float:
        """Strongest frequency in the most recent row, or 0 when there is none."""
        rows = self._view()
        if rows.shape[0] == 0:
            return 0.0
        column = int(np.argmax(rows[-1]))
        return float(self._bins[column])

    def band_level_db(self) -> float:
        """Strongest level inside the SSTV band over the last few rows.

        Used to tell whether anything is being received at all, independently of
        the audio level meter (which cannot distinguish signal from a hum).
        """
        rows = self._view()
        if rows.shape[0] == 0:
            return FLOOR_DB
        inside = (self._bins >= 1100.0) & (self._bins <= 2400.0)
        if not inside.any():
            return FLOOR_DB
        return float(np.max(rows[-min(4, rows.shape[0]):][:, inside]))


# The tones worth labelling: the sync pulse, black, the chroma mid-point and
# white.  A glance at where the bright band sits relative to these tells an
# operator whether the receiver is on frequency before any picture appears.
SCALE_MARKS = (
    (1200.0, "sync"),
    (1500.0, "black"),
    (1900.0, "mid"),
    (2300.0, "white"),
)


def add_frequency_scale(image, low_hz: float, high_hz: float, *,
                        strip: int = 30, minor_hz: float = 100.0):
    """Return *image* with a labelled frequency scale drawn along the top.

    Without a scale a waterfall is decorative; with one it becomes an instrument,
    because the operator can see at a glance whether the sync pulses are sitting
    where 1200 Hz should be.

    The top strip is laid out in two rows: the four named tones (sync, black, mid,
    white) on the first line and the numeric grid on the second.  Trying to fit
    both on one line collides as soon as the labels get close together.
    """
    from PIL import Image, ImageDraw

    if high_hz <= low_hz:
        return image
    width, height = image.size
    out = Image.new("RGB", (width, height + strip), (10, 11, 14))
    out.paste(image, (0, strip))
    draw = ImageDraw.Draw(out)
    span = high_hz - low_hz

    def x_of(hz: float) -> int:
        return int(round((hz - low_hz) / span * (width - 1)))

    # Numeric labels at a spacing that leaves room for the text, placed below the
    # named tones so the two rows do not sit on top of each other.
    step = minor_hz
    while span / step > 13:
        step *= 2
    first = np.ceil(low_hz / step) * step
    for hz in np.arange(first, high_hz + 1e-6, step):
        hz = float(hz)
        x = x_of(hz)
        draw.line([(x, strip), (x, height + strip - 1)], fill=(34, 37, 44))
        draw.text((x + 3, strip - 14), "%d" % round(hz), fill=(122, 128, 140))

    for hz, label in SCALE_MARKS:
        if not (low_hz <= hz <= high_hz):
            continue
        x = x_of(hz)
        draw.line([(x, strip), (x, height + strip - 1)], fill=(74, 80, 92))
        draw.line([(x, strip - 8), (x, strip)], fill=(150, 156, 168))
        draw.text((x + 3, 2), label, fill=(198, 204, 216))
    return out
