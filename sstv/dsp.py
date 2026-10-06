"""Signal-processing primitives shared by the SSTV encoder and decoder.

SSTV is frequency modulation: a pixel value is represented by an audio tone, and
the receiver recovers it by measuring the instantaneous frequency.  These
helpers implement that conversion in both directions, plus the resampling and
smoothing the decoder needs.  They deliberately use :mod:`numpy` where it makes
the difference between instant and unusable.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = [
    "Discriminator",
    "apply_frequency_offsets",
    "bandpass",
    "best_periodic_offset",
    "clip",
    "frequency_to_value",
    "instantaneous_frequency",
    "median_filter",
    "moving_average",
    "normalise",
    "resample",
    "smooth",
    "spectrogram",
    "stddev",
    "sync_envelope",
    "value_to_frequency",
]

TWO_PI = 2.0 * math.pi

# Window length for the waterfall's short-time Fourier transform, and the
# fraction of it that advances between frames.  2048 samples at 48 kHz is 43 ms,
# which resolves the 200 Hz spacing between SSTV's tones (the bins come out
# 23.4 Hz apart) while still refreshing 46 times a second at half-overlap.
SPECTRUM_FRAME = 2048
SPECTRUM_HOP = 0.5


# --------------------------------------------------------------------------
# Value <-> frequency mapping
# --------------------------------------------------------------------------

def value_to_frequency(value: float, black: float = 1500.0, white: float = 2300.0,
                       span: float = 255.0) -> float:
    """Map a pixel value in ``[0, span]`` to a tone in Hz."""
    return black + (white - black) * (value / span)


def frequency_to_value(freq: float, black: float = 1500.0, white: float = 2300.0,
                       span: float = 255.0) -> float:
    """Inverse of :func:`value_to_frequency`."""
    if white == black:
        return 0.0
    return (freq - black) * span / (white - black)


def clip(values, low: float = 0.0, high: float = 255.0):
    """Clamp an array (or scalar) into ``[low, high]``."""
    arr = np.asarray(values, dtype=np.float64)
    return np.clip(arr, low, high)


# --------------------------------------------------------------------------
# Waveform generation
# --------------------------------------------------------------------------

def render_frequencies(
    frequencies: Sequence[float] | np.ndarray,
    samples_per_segment: int | Sequence[int],
    sample_rate: int,
    *,
    phase: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Synthesise a constant-amplitude FM waveform from instantaneous frequencies.

    Parameters
    ----------
    frequencies:
        Desired tone for each segment, in Hz.  For pixel data this is the value
        sampled from the image; for calibration pulses it is an exact constant.
        ``nan`` marks a "silent" segment (used for the decoder's convenience
        only -- transmissions are continuous).
    samples_per_segment:
        Either one integer used for every entry, or one integer per entry.
    sample_rate:
        Output sample rate in Hz.
    phase:
        Starting phase in radians, so successive calls join seamlessly.

    Returns
    -------
    ``(waveform, end_phase)`` where *waveform* is ``float64`` in ``[-1, 1]`` and
    *end_phase* is the phase to hand to the next call.
    """
    freqs = np.asarray(frequencies, dtype=np.float64)
    if freqs.ndim == 0:
        freqs = freqs.reshape(1)
    if np.isscalar(samples_per_segment):
        counts = np.full(freqs.shape, int(samples_per_segment), dtype=np.int64)
    else:
        counts = np.asarray(samples_per_segment, dtype=np.int64)
    if counts.shape != freqs.shape:
        raise ValueError("samples_per_segment must match frequencies in length")

    counts = np.maximum(counts, 0)
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.float64), phase

    # Per-sample frequency, then cumulative phase.  Phase accumulates in a
    # running sum rather than per-segment so that segment joins stay continuous
    # -- a phase discontinuity would spray broadband noise across the spectrum
    # and corrupt the sync detectors at the far end.
    per_sample = np.repeat(freqs, counts)
    silent = np.isnan(per_sample)
    if silent.any():
        per_sample = np.where(silent, 0.0, per_sample)
    step = per_sample * (TWO_PI / float(sample_rate))
    cumulative = np.cumsum(step)
    waveform = np.sin(cumulative + phase)
    end_phase = float((phase + cumulative[-1]) % TWO_PI) if total else phase
    return waveform, end_phase


# --------------------------------------------------------------------------
# Demodulation
# --------------------------------------------------------------------------

def _hann(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones(max(length, 1), dtype=np.float64)
    n = np.arange(length, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(TWO_PI * n / (length - 1))


_BASEBAND_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _baseband_kernel(sample_rate: int, cutoff: float, taps: int = 257) -> np.ndarray:
    """A short low-pass FIR for the complex baseband, cached per (rate, cutoff).

    A frequency-domain (FFT) low-pass is *not* used here, even though it has a
    perfectly flat passband: its impulse response spans the whole recording, so
    the transient at every frequency change is spread across the entire signal
    and the first and last pixels of every scan come out wrong.  A short FIR
    keeps the transient local, which confines the damage to the handful of pixels
    either side of a transition -- and those are trimmed away when the scan is
    read.
    """
    key = (sample_rate, int(cutoff), taps)
    cached = _BASEBAND_CACHE.get(key)
    if cached is not None:
        return cached
    if taps % 2 == 0:
        taps += 1
    half = taps // 2
    n = np.arange(-half, half + 1, dtype=np.float64)
    # Windowed sinc, normalised so the passband gain is exactly one.
    x = 2.0 * cutoff / sample_rate
    ideal = np.sinc(x * n) * x
    window = np.hamming(taps)
    kernel = ideal * window
    kernel /= kernel.sum()
    _BASEBAND_CACHE[key] = kernel
    return kernel


def _baseband_lowpass(signal: np.ndarray, sample_rate: int,
                      cutoff: float) -> np.ndarray:
    """Low-pass the complex baseband signal with a short FIR."""
    if signal.size < 64:
        return signal
    kernel = _baseband_kernel(sample_rate, cutoff)
    if signal.size < kernel.size:
        return signal
    # 'same' keeps the output aligned with the input; the group delay of a
    # symmetric FIR is (taps-1)/2 samples, which is applied to both real and
    # imaginary parts identically and therefore cancels in the phase difference.
    real = np.convolve(signal.real, kernel, mode="same")
    imag = np.convolve(signal.imag, kernel, mode="same")
    return real + 1j * imag


def instantaneous_frequency(
    samples: Sequence[float] | np.ndarray,
    sample_rate: int,
    *,
    window: int = 1,
    if_filter: bool = True,
    centre_hz: float = 1750.0,
    baseband_cutoff: float = 900.0,
) -> np.ndarray:
    """Recover the instantaneous frequency of an SSTV signal, sample by sample.

    The signal is mixed down to baseband by a complex local oscillator and
    low-pass filtered, which yields a *true analytic* representation: a tone at
    ``centre_hz + df`` becomes a phasor rotating at exactly ``df`` with constant
    magnitude.  The instantaneous frequency is then the phase advance between
    consecutive samples.

    The common alternative -- a Hilbert FIR followed by a conjugate product -- is
    unreliable here: a windowed Hilbert transformer has a frequency-dependent gain
    that is well under unity near the low end of the 1200-2300 Hz band (0.65 at
    1500 Hz for a 127-tap design), so the reconstructed signal is not circular,
    the phase advance varies within a cycle, and the recovered frequency is wrong
    by an amount that depends on the tone.  Down-mixing has no such weakness.

    Parameters
    ----------
    window:
        Extra moving-average width applied to the result, in samples.  Defaults
        to 1 (off): the decoder integrates each pixel interval anyway, and any
        smoothing here would bleed one pixel into the next.
    if_filter:
        Apply :func:`bandpass` first.  The down-mix low-pass already rejects
        out-of-band noise, so this is belt and braces for badly contaminated
        recordings.
    centre_hz:
        Local-oscillator frequency.  Any value inside the SSTV band works; 1750 Hz
        is its centre, which minimises the required baseband width.
    baseband_cutoff:
        Baseband half-width in Hz.  SSTV's own deviation is about +-400 Hz, so
        900 Hz passes all of it while still rejecting the sum-frequency image.
    """
    data = np.asarray(samples, dtype=np.float64)
    if data.size == 0:
        return np.zeros(0, dtype=np.float64)
    if data.size < 64:
        return np.full(data.size, 1500.0, dtype=np.float64)
    if if_filter:
        data = bandpass(data, sample_rate)

    # Mix to baseband.  The sum-frequency image lands near 2*centre_hz and is
    # removed by the low-pass below; the difference term is the signal.
    n = np.arange(data.size, dtype=np.float64)
    baseband = data * np.exp(-2j * np.pi * centre_hz * n / sample_rate)
    filtered = _baseband_lowpass(baseband, sample_rate, baseband_cutoff)
    real = filtered.real
    imag = filtered.imag

    # Instantaneous frequency from the phase advance between consecutive samples.
    # The exact angle between two phasors is arctan2 of their cross and dot
    # products; the small-angle shortcut (cross/dot) is *not* accurate enough --
    # at 2300 Hz a sample advances 0.30 rad, where the shortcut is already a few
    # percent out, and that error varies with the tone and shows up as heavy
    # jitter on the recovered scan.
    x0, y0 = real[:-1], imag[:-1]
    x1, y1 = real[1:], imag[1:]
    cross = x0 * y1 - y0 * x1
    dot = x0 * x1 + y0 * y1
    magnitude = x0 * x0 + y0 * y0
    advance = np.zeros(cross.size, dtype=np.float64)
    np.arctan2(cross, dot, out=advance)
    advance[magnitude <= 1e-18] = 0.0
    if window and window > 1:
        advance = moving_average(advance, window)

    freq = np.empty(data.size, dtype=np.float64)
    # freq[i] is anchored to describe the signal *at* sample i, i.e. the advance
    # between samples i and i+1.  The last sample has no successor, so it repeats
    # its neighbour's value.  Anchoring the array this way is what makes
    # processing a recording in blocks give the same track as processing it in one
    # pass: any other convention puts every block boundary half a sample out, and
    # the decoder then reads each scan a fraction of a pixel early.
    freq[1:] = centre_hz + advance * (float(sample_rate) / TWO_PI)
    freq[0] = freq[1] if data.size > 1 else centre_hz
    return freq


_HILBERT_CACHE: dict[tuple[int, int], tuple[np.ndarray, int]] = {}


SYNC_SUSTAIN_S = 0.006
"""How long the sync tone must persist before it counts as a sync pulse.

Every mode's sync pulse lasts at least 9 ms (Robot 36's is the shortest, at 9 ms;
the PD family uses 20 ms).  A *single* sample passing through the sync frequency is
not a sync pulse at all -- it is a picture edge crossing that band on its way
between black and white, and in a picture with hard vertical edges one occurs at
nearly every line boundary.  Scoring those as syncs made a Scottie S1 recording
appear to fit 1200 Hz perfectly while pointing 286 ms away from the real sync, so
the decoder started part way down a scan and produced an unreadable picture.
Requiring the tone to last several milliseconds is what separates a pulse from a
crossing.
"""


def sync_envelope(track: np.ndarray, sample_rate: int, sync_hz: float,
                  tolerance_hz: float = 220.0,
                  sustain_s: float = SYNC_SUSTAIN_S) -> np.ndarray:
    """A per-sample signal showing where a *sustained* sync tone is present.

    Used to find the line rhythm of a transmission whose mode is *not* known.
    Every SSTV mode repeats its sync pulse once per line, and the line period is
    what tells the modes apart, so a plain "is the sync tone present here" signal
    is enough -- provided it responds to pulses rather than to momentary crossings.

    A softened threshold rather than a hard cut, so a signal that is slightly off
    frequency, or slightly weak, still scores, just lower.
    """
    if track.size == 0:
        return np.zeros(0, dtype=np.float64)
    distance = np.abs(track - sync_hz)
    envelope = np.clip(1.0 - distance / tolerance_hz, 0.0, 1.0)
    if sustain_s > 0:
        span = max(1, int(round(sustain_s * sample_rate)))
        if span > 1 and envelope.size >= span:
            # Running mean over one pulse length: a sustained tone keeps its full
            # value, while a single-sample crossing is diluted to about 1/span.
            # Shifted by half the window so the answer stays centred on the pulse --
            # unshifted, this lags the envelope by half a pulse length, and since
            # the offset comes from where the envelope peaks, every mode would be
            # reported several milliseconds late.
            kernel = np.ones(span, dtype=np.float64) / span
            envelope = np.convolve(envelope, kernel, mode="same")
            envelope = np.roll(envelope, -(span // 2))
    return envelope


def best_periodic_offset(envelope: np.ndarray, sample_rate: int, period_s: float,
                         window_s: float = 0.001) -> tuple[float, float]:
    """Find the offset whose periodic samples best line up with *envelope*.

    Returns ``(offset_seconds, score)``.  The score is normalised into 0-1: it is
    the mean envelope value at ``offset + k * period`` for every pulse that lies
    inside the signal, divided by the largest envelope value present.  Both halves
    of that matter.

    Dividing by the peak stops a *long* period scoring higher than the truth just
    because it averages over fewer pulses and happens to land on the tall ones: on
    a Robot 36 recording a 1050 ms comb scored 1.08 against its own 0.999, so every
    long-period mode would have beaten the real one.

    Summing each offset exactly, rather than correlating by FFT, is what keeps a
    mismatched period from scoring *above* a matched one: a circular correlation
    wraps the tail of the probe back onto its head, and the wrapped terms are not
    counted in the divisor, which inflated those same long periods past 1.0.
    """
    if envelope.size == 0 or period_s <= 0 or sample_rate <= 0:
        return 0.0, 0.0
    peak = float(np.max(envelope))
    if peak <= 1e-9:
        return 0.0, 0.0
    period = int(round(period_s * sample_rate))
    if period < 2 or period >= envelope.size:
        return 0.0, float(np.mean(envelope)) / peak

    steps = int(np.ceil(envelope.size / period))
    offsets = np.arange(period, dtype=np.int64)
    # One row per pulse, one column per candidate offset; column c holds the sum
    # of the envelope at c, c + period, c + 2 * period, ...
    index = offsets[None, :] + period * np.arange(steps, dtype=np.int64)[:, None]
    totals = np.cumsum(envelope, dtype=np.float64)
    totals = np.concatenate(([0.0], totals))
    inside = index < envelope.size
    last = envelope.size - 1
    clipped = np.where(inside, index, last)
    sums = (totals[clipped + 1] - totals[clipped]).sum(axis=0)
    counts = inside.sum(axis=0)
    counts[counts == 0] = 1
    score = sums / counts / peak
    best = int(np.argmax(score))
    return float(best / sample_rate), float(score[best])


def _hilbert_fir(sample_rate: int, taps: int = 127) -> tuple[np.ndarray, int]:
    """A windowed FIR Hilbert transformer, cached per (rate, length).

    A *global* FFT Hilbert transform is unusable for this application.  Its
    impulse response spans the whole recording, so the transient created by every
    frequency change is smeared across seconds of signal -- a 30 ms transition
    corrupts the demodulated track far beyond its own duration, which is exactly
    the failure that makes a whole scan line read as a ramp.  A short FIR version
    keeps the estimate local, which is what a real receiver's IF filter does.

    Returns ``(kernel, delay)``.  The kernel is real; convolving the signal with
    it yields the quadrature component, delayed by ``delay`` samples, which
    callers account for when slicing.
    """
    key = (sample_rate, taps)
    cached = _HILBERT_CACHE.get(key)
    if cached is not None:
        return cached
    if taps % 2 == 0:
        taps += 1
    half = taps // 2
    n = np.arange(-half, half + 1, dtype=np.float64)
    ideal = np.zeros(taps, dtype=np.float64)
    odd = (n % 2) != 0
    # Ideal Hilbert impulse response: 2/(pi*n) on odd taps, zero on even ones.
    ideal[odd] = 2.0 / (np.pi * n[odd])
    window = 0.54 + 0.46 * np.cos(np.pi * n / half)
    kernel = ideal * window
    _HILBERT_CACHE[key] = (kernel, half)
    return kernel, half


def _quadrature(data: np.ndarray, sample_rate: int, taps: int = 127) -> np.ndarray:
    """The 90-degree-shifted component of *data*, delayed by the FIR's half-length."""
    kernel, _ = _hilbert_fir(sample_rate, taps)
    if data.size < kernel.size:
        return np.zeros_like(data)
    return np.convolve(data, kernel, mode="same")


def bandpass(
    samples: Sequence[float] | np.ndarray,
    sample_rate: int,
    low: float = 1100.0,
    high: float = 2500.0,
    *,
    rolloff: float = 200.0,
) -> np.ndarray:
    """Zero-phase FFT band-pass -- the decoder's IF filter.

    SSTV occupies roughly 1200-2300 Hz and nothing else, so discarding
    everything outside a slightly wider window removes precisely the noise the
    FM discriminator would otherwise convert into frequency error.  The filter is
    applied in the frequency domain with raised-cosine skirts, which keeps the
    phase response exactly linear (zero phase) -- important, because a
    non-linear phase response near a sync pulse would smear the very edge the
    decoder's clock recovery depends on.
    """
    data = np.asarray(samples, dtype=np.float64)
    if data.size < 64:
        return data
    spectrum = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(data.size, 1.0 / float(sample_rate))
    lower = 0.5 - 0.5 * np.cos(np.pi * np.clip((freqs - (low - rolloff)) / rolloff, 0.0, 1.0))
    upper = 0.5 - 0.5 * np.cos(np.pi * np.clip(((high + rolloff) - freqs) / rolloff, 0.0, 1.0))
    gain = np.minimum(lower, upper)
    return np.fft.irfft(spectrum * gain, n=data.size)


def apply_frequency_offsets(freq: np.ndarray, offset: float,
                            span: float = 255.0, black: float = 1500.0,
                            white: float = 2300.0) -> np.ndarray:
    """Shift a whole transmission's frequency axis.

    Radios, sound cards and recordings routinely hand us a signal that is a few
    tens of Hz off.  Because every level in a mode is defined by its frequency,
    such an offset shows up as a uniform brightness/contrast error; correcting it
    in the frequency domain is exact.
    """
    scale = span / (white - black) if white != black else 1.0
    return (np.asarray(freq, dtype=np.float64) - black - offset) * scale


# --------------------------------------------------------------------------
# Filtering and statistics
# --------------------------------------------------------------------------

def moving_average(data: np.ndarray, width: int) -> np.ndarray:
    """Centred moving average; edge samples use a shrinking window."""
    arr = np.asarray(data, dtype=np.float64)
    if width <= 1 or arr.size == 0:
        return arr
    width = min(int(width), arr.size)
    kernel = np.ones(width, dtype=np.float64) / width
    pad = width // 2
    padded = np.concatenate((np.full(pad, arr[0]), arr, np.full(pad, arr[-1])))
    out = np.convolve(padded, kernel, mode="same")
    return out[pad : pad + arr.size]


def smooth(data: np.ndarray, width: int) -> np.ndarray:
    """Alias of :func:`moving_average`, for readability at call sites."""
    return moving_average(data, width)


def median_filter(data: np.ndarray, width: int) -> np.ndarray:
    """Centred median filter; removes impulsive noise without blurring edges."""
    arr = np.asarray(data, dtype=np.float64)
    if width <= 1 or arr.size == 0:
        return arr
    width = min(int(width) | 1, arr.size if arr.size % 2 else arr.size - 1)
    if width <= 1:
        return arr
    pad = width // 2
    padded = np.concatenate((np.full(pad, arr[0]), arr, np.full(pad, arr[-1])))
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    return np.median(windows, axis=-1)


def resample(data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resampler (adequate: SSTV audio is 3 kHz wide)."""
    arr = np.asarray(data, dtype=np.float64)
    if src_rate == dst_rate or arr.size == 0:
        return arr
    n_out = int(round(arr.size * dst_rate / src_rate))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float64)
    x_out = np.arange(n_out, dtype=np.float64) * (src_rate / dst_rate)
    x_in = np.arange(arr.size, dtype=np.float64)
    return np.interp(x_out, x_in, arr)


def normalise(data: np.ndarray) -> np.ndarray:
    """Scale to unit peak, leaving silence alone."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        return arr
    peak = float(np.max(np.abs(arr)))
    if peak < 1e-12:
        return arr
    return arr / peak


def stddev(data: np.ndarray) -> float:
    """Standard deviation, guarded for empty input."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.std(arr))


# --------------------------------------------------------------------------
# Spectrum analysis (for the waterfall display)
# --------------------------------------------------------------------------

_SPECTRUM_WINDOW_CACHE: dict[int, np.ndarray] = {}


def _spectrum_window(frame: int) -> np.ndarray:
    cached = _SPECTRUM_WINDOW_CACHE.get(frame)
    if cached is None:
        cached = _hann(frame)
        _SPECTRUM_WINDOW_CACHE[frame] = cached
    return cached


def spectrogram(
    samples,
    sample_rate: int,
    *,
    frame: int = SPECTRUM_FRAME,
    hop_fraction: float = SPECTRUM_HOP,
    unit: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Short-time Fourier transform, as one row of magnitudes per time step.

    Returns ``(rows, hop)`` where *rows* has shape ``(n_frames, frame // 2 + 1)``
    and is in decibels relative to full scale, and *hop* is the number of input
    samples between consecutive rows.  The hop is returned rather than derived
    because the caller needs it to know which samples a later call should start
    from -- that is what lets the waterfall show only *new* audio instead of
    recomputing the whole recording on every repaint.

    A Hann window is used, which keeps a steady tone from smearing into
    neighbouring bins; that matters here because the whole point of the display is
    to let the eye pick out SSTV's discrete tones and its sync pulses.
    """
    data = np.asarray(samples, dtype=np.float64)
    frame = int(frame)
    hop = max(1, int(round(frame * float(hop_fraction))))
    if frame < 8 or data.size < frame:
        return np.zeros((0, frame // 2 + 1), dtype=np.float32), hop

    # Scale so that a full-scale sine gives 0 dB.
    window = _spectrum_window(frame)
    window_gain = float(np.sum(window)) / 2.0
    normaliser = 1.0 / max(window_gain, 1e-12) if unit else 1.0

    starts = range(0, data.size - frame + 1, hop)
    rows = np.empty((len(range(0, data.size - frame + 1, hop)), frame // 2 + 1),
                    dtype=np.float32)
    for index, start in enumerate(starts):
        segment = data[start : start + frame]
        spectrum = np.fft.rfft(segment * window)
        magnitude = np.abs(spectrum) * normaliser
        rows[index] = 20.0 * np.log10(np.maximum(magnitude, 1e-7))
    return rows, hop


def bin_frequencies(frame: int, sample_rate: int) -> np.ndarray:
    """Centre frequency of each column produced by :func:`spectrogram`."""
    return np.fft.rfftfreq(int(frame), 1.0 / float(sample_rate))


class Discriminator:
    """Read pixel values out of a demodulated frequency track.

    The decoder works by advancing a clock across the recovered track and reading
    (or integrating) the value at each pixel.  Frequency-to-level conversion uses
    the mode's own black/white reference, so switching modes -- or correcting a
    receiver offset -- never needs a second conversion pass.
    """

    __slots__ = ("_freq", "_rate", "_last", "_black", "_white", "_span")

    def __init__(
        self,
        freq: np.ndarray,
        sample_rate: int,
        black: float = 1500.0,
        white: float = 2300.0,
        span: float = 255.0,
    ) -> None:
        self._freq = np.asarray(freq, dtype=np.float64)
        self._rate = float(sample_rate)
        self._last = float(self._freq[-1]) if self._freq.size else black
        self._black = float(black)
        self._white = float(white)
        self._span = float(span)

    def __len__(self) -> int:
        return self._freq.size

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def black(self) -> float:
        return self._black

    @property
    def white(self) -> float:
        return self._white

    @property
    def duration(self) -> float:
        """Length of the track in seconds."""
        return self._freq.size / self._rate if self._rate else 0.0

    @property
    def frequencies(self) -> np.ndarray:
        """The raw frequency track, in Hz."""
        return self._freq

    def to_value(self, freq) -> np.ndarray:
        """Convert one or more frequencies in Hz to pixel levels."""
        scale = self._span / (self._white - self._black) if self._white != self._black else 0.0
        return (np.asarray(freq, dtype=np.float64) - self._black) * scale

    def to_frequency(self, value: float) -> float:
        """Convert a pixel level back to Hz."""
        return self._black + (self._white - self._black) * (value / self._span)

    def frequency_at(self, instant: float) -> float:
        """Frequency in Hz at *instant* seconds (outside the track it clamps)."""
        if self._freq.size == 0:
            return self._black
        pos = instant * self._rate
        if pos <= 0.0:
            return float(self._freq[0])
        if pos >= self._freq.size - 1:
            return self._last
        lo = int(pos)
        frac = pos - lo
        return float(self._freq[lo] * (1.0 - frac) + self._freq[lo + 1] * frac)

    def at(self, instant: float) -> float:
        """Pixel level at *instant* seconds."""
        return float(self.to_value(self.frequency_at(instant)))

    def average(self, start: float, end: float) -> float:
        """Mean frequency in Hz over ``[start, end)`` -- for sync pulses."""
        if self._freq.size == 0:
            return self._black
        a = max(0, int(start * self._rate))
        b = min(self._freq.size, max(a + 1, int(end * self._rate)))
        if b <= a:
            return self.frequency_at(start)
        return float(np.mean(self._freq[a:b]))

    def average_value(self, start: float, end: float) -> float:
        """Mean pixel level over ``[start, end)``."""
        return float(self.to_value(self.average(start, end)))

    def slice_values(self, start: float, end: float, count: int,
                     margin: float = 0.0) -> np.ndarray:
        """Integrate a scan segment into *count* pixel values.

        Each output pixel averages the frequency over its own pixel interval,
        which is exactly how a real receiver's low-pass filter behaves and is
        markedly more noise-tolerant than point sampling.
        """
        if count <= 0:
            return np.zeros(0, dtype=np.float64)
        span = end - start
        if span <= 0 or self._freq.size == 0:
            return np.full(count, self.at(start), dtype=np.float64)
        inner = span - 2.0 * margin
        if inner <= 0:
            inner = span
            margin = 0.0
        edges = start + margin + np.arange(count + 1, dtype=np.float64) * (inner / count)
        idx = np.clip(np.round(edges * self._rate).astype(np.int64), 0, self._freq.size - 1)
        cs = np.concatenate(([0.0], np.cumsum(self._freq)))
        totals = cs[idx[1:]] - cs[idx[:-1]]
        widths = np.maximum(idx[1:] - idx[:-1], 1)
        return self.to_value(totals / widths)
