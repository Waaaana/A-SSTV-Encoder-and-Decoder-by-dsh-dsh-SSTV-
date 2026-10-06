"""VIS header decoding: reading a transmission's mode identifier off the air.

Every SSTV transmission starts with the same calibration header, which names the
mode so the receiver knows which scan plan to apply.  Getting this right is what
lets the program lock onto an unknown recording with no user input at all.

Header layout (see :mod:`sstv.modes` for the encoder side)::

    1900 Hz 300 ms | 1200 Hz 10 ms | 1900 Hz 300 ms
    1200 Hz  30 ms | 7 data bits LSB-first, 30 ms each | parity 30 ms | 1200 Hz 30 ms

The seven data bits are 1100 Hz for a one and 1300 Hz for a zero -- the reverse of
the luminance convention, where a higher tone means a brighter pixel.  The header
is therefore found by locating that long 1200 Hz start bit rather than by assuming
where the file begins: recordings are routinely trimmed, and a receiver may open
mid-transmission.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import dsp
from .modes import (
    SYNC_HZ,
    VIS_BIT_MS,
    VIS_BIT_ONE_HZ,
    VIS_BIT_ZERO_HZ,
    VIS_BREAK_MS,
    VIS_LEADER_MS,
    mode_by_vis,
    vis_parity,
)

__all__ = ["VisHeader", "decode_vis", "find_signal_start", "find_vis_start"]

_BIT_S = VIS_BIT_MS / 1000.0
_START_S = VIS_BIT_MS / 1000.0
_LEADER_S = VIS_LEADER_MS / 1000.0
_BREAK_S = VIS_BREAK_MS / 1000.0
_HEADER_S = 2 * _LEADER_S + _BREAK_S + 9 * _BIT_S

# Tolerances.  Frequencies are far enough apart that +-150 Hz is generous; the
# timing tolerance absorbs the sample-rate error of a sound card over 30 ms.
_FREQ_TOL = 200.0
_SYNC_TOL = 260.0


@dataclass
class VisHeader:
    """The result of reading a calibration header."""

    code: int
    bits: tuple[int, ...]
    parity: int
    start: float
    """Timestamp of the first data bit, in seconds from the start of the audio."""
    confidence: float
    """0-1; low values mean the header was noisy and the code may be wrong."""
    measured_hz: tuple[float, ...] = ()
    parity_ok: bool = True
    stop_ok: bool = True

    @property
    def end(self) -> float:
        """Timestamp just past the stop bit, i.e. where the picture begins.

        The bit field is the start bit, seven data bits, the parity bit and the
        stop bit -- nine intervals in all.  Counting only the eight that follow
        the start bit leaves the caller reading audio 30 ms too early, which
        shifts every subsequent line by a whole bit.
        """
        return self.start + 9 * _BIT_S

    @property
    def length(self) -> float:
        return _HEADER_S


def _runs(mask: np.ndarray):
    """Yield ``(value, start_index, end_index)`` for each constant run in *mask*."""
    if mask.size == 0:
        return
    edges = np.flatnonzero(np.diff(mask.astype(np.int16)) != 0) + 1
    bounds = np.concatenate(([0], edges, [mask.size]))
    for i in range(len(bounds) - 1):
        a, b = int(bounds[i]), int(bounds[i + 1])
        yield bool(mask[a]), a, b


# The start bit sits at 610-640 ms in a well-formed header, so anything beyond
# roughly 0.68 s cannot be it.  Searching a wide window would happily latch onto
# the first horizontal sync pulse of the image instead.
_START_SEARCH_S = 0.61 + 0.045


def find_signal_start(track: np.ndarray, sample_rate: int,
                      leader_s: float = 0.05) -> float:
    """Where a VIS header appears to begin, in seconds.

    A capture very often starts before the station does: opening a sound device
    takes a moment, and the recorder runs while the operator waits.  One real
    capture had 380 ms of silence in front of the header, which put the header
    outside a search window that assumed it began at t=0 -- and the mode then went
    unidentified, with the picture lost.

    The anchor is the header's leader tone, which is the one thing in a VIS header
    that cannot be confused with anything else: it sits at 1900 Hz, a frequency no
    part of the header or of the silent carrier uses, and it lasts 300 ms.  The
    header therefore begins 300 ms before that tone starts.  Returns 0.0 when no
    leader is found, which leaves the caller searching from the beginning exactly
    as before.
    """
    if track.size == 0:
        return 0.0
    span = max(1, int(leader_s * sample_rate))
    if track.size < span:
        return 0.0
    at_leader = np.abs(track - 1900.0) < 150.0
    running = np.convolve(at_leader.astype(np.int32), np.ones(span, dtype=np.int32),
                          mode="valid")
    # The leader is a solid plateau, so require nearly the whole probe to be on it.
    found = np.flatnonzero(running >= span * 0.95)
    if found.size == 0:
        return 0.0
    leader_start = float(found[0] / sample_rate)
    return max(0.0, leader_start - _LEADER_S)


def find_vis_start(track: np.ndarray, sample_rate: int,
                   search_seconds: float = _START_SEARCH_S,
                   search_from: float = 0.0) -> float | None:
    """Find the timestamp of the start bit, in seconds.

    The start bit is the only 1200 Hz interval in the header longer than the 10 ms
    break pulse, so it is identified by run length.  Preferring the *longest*
    qualifying run makes the search robust against a break pulse that the
    demodulator reports as slightly wide.

    *search_from* shifts the window to start later, for a recording whose beginning
    is silence rather than signal; the header's own layout is unchanged, it just
    begins later in the file.
    """
    begin = max(0, int(search_from * sample_rate))
    limit = min(track.size, begin + int(search_seconds * sample_rate))
    if limit - begin < int(0.2 * sample_rate):
        return None
    near_sync = np.abs(track[begin:limit] - SYNC_HZ) < _SYNC_TOL
    best: tuple[int, int] | None = None
    for value, a, b in _runs(near_sync):
        if not value:
            continue
        length = b - a
        if length < int(0.016 * sample_rate) or length > int(0.048 * sample_rate):
            continue
        if best is None or length > best[0]:
            best = (length, a)
    if best is None:
        return None
    return (begin + best[1]) / float(sample_rate)


def decode_vis(
    samples,
    sample_rate: int,
    *,
    track: np.ndarray | None = None,
    search_seconds: float = _START_SEARCH_S,
) -> VisHeader | None:
    """Read the VIS header from *samples*, or return ``None`` if there is none.

    Pass a pre-computed *track* to avoid demodulating twice when the caller
    already has the frequency track.

    A capture usually begins before the station does -- a sound device takes a
    moment to open, and the recorder runs while the operator waits -- so the header
    is not at t=0.  It is also not reliably found by looking for the leader tone,
    because a receiver's filters and the sound card's own shaping move the tones
    around: one real capture opened with the header's *zero* tone rather than its
    leader, and no single rule about where to start found the header in it.

    So every plausible alignment is tried and the best-evidenced header wins.  A
    real header is a fixed 300 + 10 + 300 ms shape, so the candidate starts are the
    places the header could begin; scoring them by how well their tones match the
    two legal VIS tones and by whether the trailing stop bit is really a sync pulse
    picks the true one out, and lets a candidate that merely looks like a header
    lose.
    """
    if track is None:
        track = dsp.instantaneous_frequency(samples, sample_rate)
    if track.size == 0:
        return None

    starts: list[float] = []
    # The two supported readings first, so the old behaviour is preferred whenever
    # it works and the extra search only decides the cases it cannot.
    onset = find_signal_start(track, sample_rate)
    run = find_vis_start(track, sample_rate, search_seconds, search_from=onset)
    if run is not None:
        starts.append(run)
    fallback = find_vis_start(track, sample_rate, search_seconds)
    if fallback is not None and fallback not in starts:
        starts.append(fallback)
    # Then a sweep over the plausible region for the header to start in.
    sweep_end = min(track.size / sample_rate - _HEADER_S, 3.0)
    at = 0.0
    while at <= sweep_end:
        if all(abs(at - existing) > 0.02 for existing in starts):
            starts.append(at)
        at += 0.02
    if not starts:
        return None

    best: VisHeader | None = None
    best_rank = -1.0
    for candidate in starts:
        header = _read_header_at(track, sample_rate, candidate)
        if header is None:
            continue
        rank = _header_rank(header)
        if rank > best_rank:
            best_rank, best = rank, header
    return best


def _header_rank(header: "VisHeader") -> float:
    """How much the evidence says this really is the header.

    Two things count, and both are checks against known facts rather than against
    guesses: the data tones must sit near the two legal tones, and the header must
    end with a 1200 Hz sync pulse.  The last bit of the code is weighted up when it
    names a mode this program implements, because a header that decodes to a real
    mode is far less likely to be a coincidence.
    """
    rank = header.confidence
    if header.stop_ok:
        rank += 1.0
    if header.parity_ok:
        rank += 0.5
    if mode_by_vis(header.code) is not None:
        rank += 2.0
    return rank


def _read_header_at(track: np.ndarray, sample_rate: int,
                    start: float) -> "VisHeader | None":
    """Read a header that begins with its start bit at *start*."""
    disc = dsp.Discriminator(track, sample_rate)
    data_start = start + _START_S

    def read_bit(index: int) -> tuple[int, float]:
        # Sample the middle of the bit: the first and last few milliseconds of a
        # 30 ms tone can still carry filter ringing from the neighbouring bit.
        a = data_start + index * _BIT_S + 0.25 * _BIT_S
        b = data_start + index * _BIT_S + 0.75 * _BIT_S
        if a < 0 or b > disc.duration:
            return 0, 0.0
        hz = disc.average(a, b)
        # Compare against the encoder's own constants so the two can never
        # disagree about which tone means one.
        if abs(hz - VIS_BIT_ONE_HZ) <= abs(hz - VIS_BIT_ZERO_HZ):
            return 1, hz
        return 0, hz

    bits: list[int] = []
    measured: list[float] = []
    confidence = 1.0
    for i in range(7):
        bit, hz = read_bit(i)
        bits.append(bit)
        measured.append(hz)
        # Distance from the ideal tone, normalised by half the 200 Hz spacing.
        error = min(abs(hz - VIS_BIT_ONE_HZ), abs(hz - VIS_BIT_ZERO_HZ)) / 100.0
        confidence = min(confidence, max(0.0, 1.0 - error))

    parity, parity_hz = read_bit(7)
    measured.append(parity_hz)
    stop = disc.average(data_start + 8 * _BIT_S + 0.25 * _BIT_S,
                        data_start + 8 * _BIT_S + 0.75 * _BIT_S)

    code = sum(bit << i for i, bit in enumerate(bits))
    return VisHeader(
        code=code,
        bits=tuple(bits),
        parity=parity,
        start=data_start,
        confidence=confidence,
        measured_hz=tuple(measured),
        parity_ok=(parity == vis_parity(code)),
        stop_ok=abs(stop - SYNC_HZ) < _SYNC_TOL,
    )
