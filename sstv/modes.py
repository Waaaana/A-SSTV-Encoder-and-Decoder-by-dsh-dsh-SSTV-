"""Transmission-mode parameter tables.

Every mainstream SSTV mode is specified here as a *scan plan*: an ordered list of
timed segments per line, plus the frame geometry and the VIS identifier.  The
encoder walks a plan to make audio and the decoder walks the same plan to read it
back, so a mode is defined in exactly one place and can never disagree with
itself.

Frequency conventions
---------------------
SSTV represents a level as an audio tone:

* **1200 Hz** - horizontal sync pulse (all modes)
* **1500 Hz** - black
* **1900 Hz** - mid-grey / chroma neutral
* **2300 Hz** - white

Only :data:`BLACK_HZ` and :data:`WHITE_HZ` are needed to convert a level to a
tone; the sync frequency is a separate constant because a sync pulse is not a
pixel value.  Robot and PD modes carry ``R-Y``/``B-Y`` chroma differences rather
than RGB, and their neutral point is 1900 Hz -- that is what
:data:`SyncSpec.zero_span` expresses: a component whose value is ``-zero_span``
sits on the black tone, ``0`` on mid-grey and ``+zero_span`` on the white tone.

Source of the numbers
---------------------
Timings follow the published mode definitions as implemented by the widely used
open-source decoders (PySSTV, QSSTV, MMSSTV).  Where a mode has been set up with
a non-standard line time over the years (PD 50 vs PD 90) the nominal values are
used for both encoding and decoding, so a round trip is always exact.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Iterator

__all__ = [
    "BLACK_HZ",
    "CHROMA_ZERO_LEVEL",
    "MODE_LIST",
    "MODES",
    "MID_HZ",
    "SYNC_HZ",
    "VIS_HEADER_MS",
    "VIS_LEADER_HZ",
    "VIS_LEADER_MS",
    "WHITE_HZ",
    "Channel",
    "ModeSpec",
    "Segment",
    "get_mode",
    "mode_by_vis",
    "names",
    "vis_bits",
    "vis_header_segments",
    "vis_parity",
    "vis_plan",
]

# --------------------------------------------------------------------------
# Frequency constants
# --------------------------------------------------------------------------

SYNC_HZ = 1200.0
BLACK_HZ = 1500.0
WHITE_HZ = 2300.0
MID_HZ = 1900.0
# VIS data bits: a ZERO is 1100 Hz and a ONE is 1300 Hz.  Getting these the wrong
# way round produces a header no other SSTV program can read, and one whose
# decoded code is the bitwise complement of the intended value -- a swap that is
# easy to miss because a matching decoder still round-trips happily.
VIS_ZERO_HZ = 1300.0
VIS_ONE_HZ = 1100.0

CHROMA_ZERO_LEVEL = 128
"""Digital value of neutral chroma for the R-Y / B-Y modes."""


class Channel(str, enum.Enum):
    """What a segment carries."""

    SYNC = "sync"
    PORCH = "porch"
    SEPARATOR = "separator"
    LUMINANCE = "luminance"
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    CR = "cr"        # R-Y
    CB = "cb"        # B-Y
    VIS = "vis"
    LEADER = "leader"
    BREAK = "break"

    @property
    def is_image(self) -> bool:
        return self in (
            Channel.LUMINANCE,
            Channel.RED,
            Channel.GREEN,
            Channel.BLUE,
            Channel.CR,
            Channel.CB,
        )

    @property
    def is_rgb_primary(self) -> bool:
        return self in (Channel.RED, Channel.GREEN, Channel.BLUE)


# --------------------------------------------------------------------------
# Scan plan
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Segment:
    """One timed interval of a scan line.

    ``pixels`` is the number of image samples the segment carries; it is ``0``
    for sync, porch and separator intervals.  ``black_hz``/``white_hz`` give the
    component's own tone references, which differ between the RGB modes and the
    chroma-difference modes.  ``tone_hz`` overrides both for the few non-image
    intervals that are defined by an exact tone rather than by a level -- the
    Robot family's 1900 Hz separator is the only such case.
    """

    channel: Channel
    milliseconds: float
    pixels: int = 0
    black_hz: float = BLACK_HZ
    white_hz: float = WHITE_HZ
    tone_hz: float | None = None

    @property
    def seconds(self) -> float:
        return self.milliseconds / 1000.0

    @property
    def is_image(self) -> bool:
        return self.channel.is_image

    @property
    def frequency(self) -> float:
        """The exact tone for a non-image interval."""
        if self.tone_hz is not None:
            return self.tone_hz
        return SYNC_HZ if self.channel is Channel.SYNC else BLACK_HZ

    def value_to_hz(self, value: float, zero_span: float = 255.0) -> float:
        """Convert a component value to a tone using this segment's references."""
        return self.black_hz + (self.white_hz - self.black_hz) * (value / zero_span)

    def hz_to_value(self, hz: float, zero_span: float = 255.0) -> float:
        """Inverse of :meth:`value_to_hz`."""
        if self.white_hz == self.black_hz:
            return 0.0
        return (hz - self.black_hz) * zero_span / (self.white_hz - self.black_hz)


def _chroma_segment(channel: Channel, milliseconds: float, pixels: int) -> Segment:
    """A signed R-Y/B-Y segment: -128 -> black tone, +127 -> white tone."""
    return Segment(channel, milliseconds, pixels, BLACK_HZ, WHITE_HZ)


@dataclass(frozen=True)
class ModeSpec:
    """Everything needed to send or receive one SSTV mode."""

    name: str
    vis_code: int
    line_milliseconds: float
    line_count: int
    segments: tuple[Segment, ...]
    color_space: str = "RGB"          # "RGB" or "YCRCB"
    interlace: int = 1                 # 1 = sequential, 2 = even lines then odd
    chroma_subsample: int = 1          # chroma pixels per luminance pixel
    vis_code_parity: bool = True       # odd parity, as all real transmissions use
    notes: str = ""
    aliases: tuple[str, ...] = ()
    image_width: int = 320
    image_height: int = 256
    # Robot 36 gives each radio line only one of the two chroma components, so a
    # line is not "missing" chroma when the other component is absent.
    chroma_on_even_only: bool = False

    # -- geometry ---------------------------------------------------------
    @property
    def line_seconds(self) -> float:
        return self.line_milliseconds / 1000.0

    @property
    def pixels_per_line(self) -> int:
        """Luminance samples carried by one radio line."""
        return sum(seg.pixels for seg in self.segments if seg.channel is Channel.LUMINANCE) or \
            sum(seg.pixels for seg in self.segments if seg.is_image)

    @property
    def duration_seconds(self) -> float:
        return self.line_seconds * self.line_count

    @property
    def frame_lines(self) -> int:
        """Radio lines in a complete frame."""
        return self.line_count

    @property
    def is_color(self) -> bool:
        return sum(1 for seg in self.segments if seg.is_image) > 1

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.image_width, self.image_height)

    # -- plan access ------------------------------------------------------
    def image_segments(self) -> tuple[Segment, ...]:
        return tuple(seg for seg in self.segments if seg.is_image)

    def channel_names(self) -> tuple[str, ...]:
        seen: list[str] = []
        for seg in self.segments:
            if seg.is_image and seg.channel.value not in seen:
                seen.append(seg.channel.value)
        return tuple(seen)

    def pixels_for(self, channel: Channel) -> int:
        return sum(seg.pixels for seg in self.segments if seg.channel is channel)

    def segment_offsets(self) -> Iterator[tuple[Segment, float, float]]:
        """Yield ``(segment, start_seconds, end_seconds)`` within one line."""
        t = 0.0
        for seg in self.segments:
            end = t + seg.seconds
            yield seg, t, end
            t = end

    def sync_windows(self) -> tuple[tuple[float, float], ...]:
        """``(start, end)`` seconds of every sync pulse in one line."""
        return tuple(
            (start, end)
            for seg, start, end in self.segment_offsets()
            if seg.channel is Channel.SYNC
        )

    def line_time_check(self, tolerance: float = 0.02) -> bool:
        """True when the segments sum to the declared line time."""
        total = sum(seg.milliseconds for seg in self.segments)
        return abs(total - self.line_milliseconds) <= tolerance

    @property
    def segment_milliseconds(self) -> float:
        return sum(seg.milliseconds for seg in self.segments)

    @property
    def slack_milliseconds(self) -> float:
        """Unallocated time at the end of a line.

        Several published mode definitions leave a fraction of a millisecond
        between the last scan and the next sync pulse -- Robot 72's is 2.6 ms.
        Emitting a line shorter than its nominal period would make a receiver's
        line clock drift steadily, so the encoder pads the line back out to its
        declared length with an extra porch of exactly this duration.
        """
        return max(0.0, self.line_milliseconds - self.segment_milliseconds)

    def line_plan(self) -> tuple[Segment, ...]:
        """Segments for one line, padded out to the declared line time."""
        slack = self.slack_milliseconds
        if slack <= 1e-9:
            return self.segments
        return self.segments + (_seg(Channel.PORCH, slack),)

    def segment_samples(self, sample_rate: int) -> tuple[int, ...]:
        """Sample count for each segment of a line, at *sample_rate*.

        The counts are derived from *cumulative* boundaries rather than by
        rounding each segment on its own.  Rounding independently lets every
        segment round up by the same fraction of a sample, so a line that should
        last 24407.04 samples comes out at 24408 -- half a millisecond long, and
        the line clock then drifts several milliseconds over a long transmission.
        Using the differences between rounded running totals keeps that error
        bounded by half a sample for the whole line instead of growing with the
        number of segments.
        """
        plan = self.line_plan()
        counts: list[int] = []
        elapsed_ms = 0.0
        previous = 0
        for seg in plan:
            elapsed_ms += seg.milliseconds
            boundary = int(round(sample_rate * elapsed_ms / 1000.0))
            counts.append(boundary - previous)
            previous = boundary
        return tuple(counts)

    def line_samples(self, sample_rate: int) -> int:
        """Total samples in one line at *sample_rate*."""
        return int(round(sample_rate * self.line_milliseconds / 1000.0))

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"{self.name} ({self.pixels_per_line}px/line, {self.duration_seconds:.1f}s)"


# --------------------------------------------------------------------------
# Segment builders
# --------------------------------------------------------------------------

def _seg(channel: Channel, ms: float, pixels: int = 0) -> Segment:
    return Segment(channel, ms, pixels)


def _rgb(ms: float, pixels: int, channel: Channel) -> Segment:
    return Segment(channel, ms, pixels, BLACK_HZ, WHITE_HZ)


# --------------------------------------------------------------------------
# VIS header
# --------------------------------------------------------------------------
# Every transmission opens with a calibration header that names the mode.  The
# receiver reads the 7-bit code and looks the mode up; without it a decoder has
# no way to know which scan plan to apply.

VIS_LEADER_MS = 300.0
VIS_BREAK_MS = 10.0
VIS_BIT_MS = 30.0
VIS_LEADER_HZ = MID_HZ
VIS_BREAK_HZ = SYNC_HZ
VIS_STOP_HZ = SYNC_HZ
# A VIS one is 1100 Hz and a zero is 1300 Hz -- the opposite of the luminance
# convention, where the higher tone means "brighter".  This is the single easiest
# value in the whole format to get backwards, and getting it backwards yields a
# header that round-trips through a matching decoder while being unreadable by
# every other SSTV program.
VIS_BIT_ONE_HZ = 1100.0
VIS_BIT_ZERO_HZ = 1300.0

VIS_HEADER_MS = 2 * VIS_LEADER_MS + VIS_BREAK_MS + VIS_BIT_MS * (1 + 7 + 1 + 1)
"""300 + 10 + 300 leader/break plus start, 7 data, parity and stop bits = 880 ms."""


def vis_bits(vis_code: int) -> tuple[int, ...]:
    """The 7 data bits of *vis_code*, least-significant bit first.

    LSB-first is the convention every real transmission uses, and a decoder that
    reads them the other way round sees a different, usually unknown, mode.
    """
    return tuple((vis_code >> i) & 1 for i in range(7))


def vis_parity(vis_code: int) -> int:
    """Even-parity bit: 1 when the 7 data bits hold an odd number of ones."""
    return sum(vis_bits(vis_code)) % 2


def vis_header_segments() -> tuple[Segment, ...]:
    """The fixed, mode-independent part of every transmission."""
    return (
        Segment(Channel.LEADER, VIS_LEADER_MS, tone_hz=VIS_LEADER_HZ),
        Segment(Channel.BREAK, VIS_BREAK_MS, tone_hz=VIS_BREAK_HZ),
        Segment(Channel.LEADER, VIS_LEADER_MS, tone_hz=VIS_LEADER_HZ),
    )


def vis_plan(vis_code: int) -> list[tuple[float, float]]:
    """The complete header as ``(frequency_hz, milliseconds)`` pairs.

    Leader, break, leader, then the start bit, the seven data bits LSB-first, the
    even-parity bit and the stop bit.  The caller plays these exactly as given.
    """
    plan: list[tuple[float, float]] = [
        (VIS_LEADER_HZ, VIS_LEADER_MS),
        (VIS_BREAK_HZ, VIS_BREAK_MS),
        (VIS_LEADER_HZ, VIS_LEADER_MS),
        (VIS_BREAK_HZ, VIS_BIT_MS),  # start bit, always 1200 Hz
    ]
    for bit in vis_bits(vis_code):
        plan.append((VIS_BIT_ONE_HZ if bit else VIS_BIT_ZERO_HZ, VIS_BIT_MS))
    plan.append((VIS_BIT_ONE_HZ if vis_parity(vis_code) else VIS_BIT_ZERO_HZ, VIS_BIT_MS))
    plan.append((VIS_STOP_HZ, VIS_BIT_MS))  # stop bit, always 1200 Hz
    return plan


# --------------------------------------------------------------------------
# Robot 36 and Robot 72
# --------------------------------------------------------------------------
# Robot modes are YCrCb and carry a single 160-sample chroma scan per line.
# Robot 36 alternates the two chroma components line by line, and signals which
# one is present purely through its separator tone; Robot 72 sends both.

def _robot36() -> ModeSpec:
    """Robot 36: 150 ms per line, 240 lines, 36 s of picture (VIS 0x08).

    9 ms sync | 3 ms porch | 88 ms Y (320 px) | 4.5 ms separator |
    1.5 ms 1900 Hz porch | 44 ms chroma (320 samples) | 9 ms line tail

    Even lines carry R-Y and odd lines carry B-Y.  The separator tone is the
    parity marker: **1500 Hz announces R-Y, 2300 Hz announces B-Y**.  It is the
    only way a receiver can tell the two apart, so the encoder emits it and the
    decoder samples it rather than assuming an order -- that is what keeps red
    and blue from swapping when a decode starts mid-pair.

    Chroma runs at 0.1375 ms per transmitted sample against luminance's 0.275 ms,
    so chroma has half the horizontal resolution: 160 distinct chroma pixels per
    line, each sent as two samples.  Vertically the two lines of a pair share one
    chroma row, giving 4:2:0 subsampling overall.
    """
    segments = (
        _seg(Channel.SYNC, 9.0),
        _seg(Channel.PORCH, 3.0),
        _seg(Channel.LUMINANCE, 88.0, 320),
        _seg(Channel.SEPARATOR, 4.5),
        Segment(Channel.SEPARATOR, 1.5, 0, tone_hz=MID_HZ),
        _chroma_segment(Channel.CR, 44.0, 320),
        _seg(Channel.PORCH, 0.0),
    )
    return ModeSpec(
        name="Robot 36",
        vis_code=8,
        line_milliseconds=150.0,
        line_count=240,
        segments=segments,
        color_space="YCRCB",
        chroma_subsample=2,
        image_width=320,
        image_height=240,
        chroma_on_even_only=True,
        aliases=("robot36", "r36", "robot-36"),
        notes="240 lines of 150 ms; chroma alternates R-Y / B-Y line by line.",
    )


def _robot72() -> ModeSpec:
    """Robot 72: ~299.2 ms per line, 240 lines, 72 s of picture (VIS 0x0C).

    9 ms sync | 3 ms porch | 92 ms Y | 4.7 ms | 92 ms Cr | 4.7 ms | 92 ms Cb.
    Chroma here has the *same* horizontal resolution as luminance (0.2875 ms per
    sample), so colour detail is preserved far better than in Robot 36.
    """
    segments = (
        _seg(Channel.SYNC, 9.0),
        _seg(Channel.PORCH, 3.0),
        _seg(Channel.LUMINANCE, 92.0, 320),
        _seg(Channel.SEPARATOR, 4.7),
        _chroma_segment(Channel.CR, 92.0, 320),
        _seg(Channel.SEPARATOR, 4.7),
        _chroma_segment(Channel.CB, 92.0, 320),
    )
    return ModeSpec(
        name="Robot 72",
        vis_code=12,
        line_milliseconds=300.0,
        line_count=240,
        segments=segments,
        color_space="YCRCB",
        image_width=320,
        image_height=240,
        aliases=("robot72", "r72", "robot-72"),
        notes="Both chroma components on every line, at full horizontal resolution.",
    )


# --------------------------------------------------------------------------
# Martin M1 / M2
# --------------------------------------------------------------------------
# RGB, sequential lines, transmitted **green, blue, red** -- not red, green,
# blue.  The sync pulse (only 4.862 ms) sits at the start of the line.

def _martin(scan_ms: float, vis: int, name: str, line_ms: float,
            aliases: tuple[str, ...]) -> ModeSpec:
    segments = (
        _seg(Channel.SYNC, 4.862),
        _seg(Channel.PORCH, 0.572),
        _rgb(scan_ms, 320, Channel.GREEN),
        _seg(Channel.SEPARATOR, 0.572),
        _rgb(scan_ms, 320, Channel.BLUE),
        _seg(Channel.SEPARATOR, 0.572),
        _rgb(scan_ms, 320, Channel.RED),
        _seg(Channel.SEPARATOR, 0.572),
    )
    return ModeSpec(
        name=name,
        vis_code=vis,
        line_milliseconds=line_ms,
        line_count=256,
        segments=segments,
        color_space="RGB",
        image_width=320,
        image_height=256,
        aliases=aliases,
        notes=f"Sequential G/B/R, {scan_ms} ms per component.",
    )


# --------------------------------------------------------------------------
# Scottie S1 / S2 / DX
# --------------------------------------------------------------------------
# RGB, sequential, and the quirk that defines the family: the sync pulse sits
# *between blue and red*, not at the start of the line, and it is a long 9 ms
# pulse.  The line is therefore best thought of as starting at red.

def _scottie(scan_ms: float, vis: int, name: str,
             aliases: tuple[str, ...]) -> ModeSpec:
    segments = (
        _seg(Channel.SEPARATOR, 1.5),
        _rgb(scan_ms, 320, Channel.GREEN),
        _seg(Channel.SEPARATOR, 1.5),
        _rgb(scan_ms, 320, Channel.BLUE),
        _seg(Channel.SYNC, 9.0),
        _seg(Channel.SEPARATOR, 1.5),
        _rgb(scan_ms, 320, Channel.RED),
    )
    return ModeSpec(
        name=name,
        vis_code=vis,
        line_milliseconds=sum(seg.milliseconds for seg in segments),
        line_count=256,
        segments=segments,
        color_space="RGB",
        image_width=320,
        image_height=256,
        aliases=aliases,
        notes=(
            f"Sequential RGB with the sync pulse between blue and red; "
            f"{scan_ms} ms per component (pixel time {scan_ms / 320:.4f} ms)."
        ),
    )


# --------------------------------------------------------------------------
# PD 50 .. PD 290
# --------------------------------------------------------------------------
# YCrCb carrying **two image rows per radio line**: the four channels are
#   Y(odd) | R-Y | B-Y | Y(even)
# with the two chroma channels shared by both rows.  Chroma uses the same pixel
# time as luminance, so unlike Robot 36 there is no colour-registration problem.
# There are no separators anywhere in the family.  The line period is
#
#   line_ms = 20 ms sync + 2.08 ms porch + 4 x (width x pixel_ms)
#
# which is exact for every PD mode, PD 240's round 1000 ms included.

PD_SYNC_MS = 20.0
PD_PORCH_MS = 2.08


def _pd(name: str, vis: int, pixel_ms: float, width: int, line_ms: float,
        lines: int, aliases: tuple[str, ...] = ()) -> ModeSpec:
    scan_ms = pixel_ms * width
    segments = (
        _seg(Channel.SYNC, PD_SYNC_MS),
        _seg(Channel.PORCH, PD_PORCH_MS),
        _seg(Channel.LUMINANCE, scan_ms, width),
        _chroma_segment(Channel.CR, scan_ms, width),
        _chroma_segment(Channel.CB, scan_ms, width),
        _seg(Channel.LUMINANCE, scan_ms, width),
    )
    return ModeSpec(
        name=name,
        vis_code=vis,
        line_milliseconds=line_ms,
        line_count=lines,
        segments=segments,
        color_space="YCRCB",
        image_width=width,
        image_height=lines * 2,
        aliases=aliases,
        notes=(
            f"Y(odd)|Cr|Cb|Y(even), {scan_ms:.3f} ms per scan at "
            f"{pixel_ms * 1000:.1f} us/pixel; {lines} radio lines x 2 rows."
        ),
    )


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

MODE_LIST: tuple[ModeSpec, ...] = (
    _robot36(),
    _robot72(),
    _martin(146.432, 44, "Martin M1", 446.446, ("martinm1", "m1", "martin-1")),
    _martin(73.216, 40, "Martin M2", 226.798, ("martinm2", "m2", "martin-2")),
    _scottie(138.240, 60, "Scottie S1", ("scotties1", "s1", "scottie-1")),
    _scottie(88.064, 56, "Scottie S2", ("scotties2", "s2", "scottie-2")),
    _scottie(345.600, 76, "Scottie DX", ("scottiedx", "sdx", "dx")),
    # (pixel time ms, width, line period ms, radio lines)
    _pd("PD 50", 93, 0.286, 320, 388.16, 128, ("pd50",)),
    _pd("PD 90", 99, 0.532, 320, 703.04, 128, ("pd90",)),
    _pd("PD 120", 95, 0.190, 640, 508.48, 248, ("pd120",)),
    _pd("PD 160", 98, 0.382, 512, 804.416, 200, ("pd160",)),
    _pd("PD 180", 96, 0.286, 640, 754.24, 248, ("pd180",)),
    _pd("PD 240", 97, 0.382, 640, 1000.0, 248, ("pd240",)),
    _pd("PD 290", 94, 0.286, 800, 937.28, 308, ("pd290",)),
)

MODES: dict[str, ModeSpec] = {}
for _spec in MODE_LIST:
    MODES[_spec.name.lower()] = _spec
    for _alias in _spec.aliases:
        MODES[_alias] = _spec
del _spec


def get_mode(name: str) -> ModeSpec:
    """Look up a mode by name or alias, case- and space-insensitively."""
    if not name:
        raise KeyError("no mode name given")
    key = name.strip().lower().replace("_", " ").replace("-", " ")
    if key in MODES:
        return MODES[key]
    compact = key.replace(" ", "")
    if compact in MODES:
        return MODES[compact]
    for spec in MODE_LIST:
        if spec.name.lower().replace(" ", "") == compact:
            return spec
    raise KeyError(f"unknown SSTV mode: {name!r} (known: {', '.join(s.name for s in MODE_LIST)})")


def mode_by_vis(vis_code: int) -> ModeSpec | None:
    """Find the mode a VIS code selects, or ``None`` when unsupported."""
    for spec in MODE_LIST:
        if spec.vis_code == vis_code:
            return spec
    return None


def names() -> tuple[str, ...]:
    """Mode names in the order they should be offered to a user."""
    return tuple(spec.name for spec in MODE_LIST)
