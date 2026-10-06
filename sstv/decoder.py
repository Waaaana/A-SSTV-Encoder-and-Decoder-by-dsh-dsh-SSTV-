"""SSTV decoder: audio in, image out.

The decoder reads the VIS header to learn which mode is arriving, then walks the
same scan plan the encoder used.  Because a real signal is never exactly on
schedule -- sound cards drift, a radio's tuning error shifts every tone, and a
recording may start mid-transmission -- it re-locks on every horizontal sync pulse
rather than trusting a single start offset, and it measures the received
sync/black/white tones so that a frequency error shows up as a diagnostic instead
of as a washed-out picture.

Typical use::

    result = decode_file("capture.wav")
    print(result.mode.name, result.image.shape)
    Image.fromarray(result.image).save("decoded.png")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from . import colorspace
from . import dsp
from . import vis as vis_module
from .modes import (
    BLACK_HZ,
    MODE_LIST,
    SYNC_HZ,
    WHITE_HZ,
    Channel,
    ModeSpec,
    get_mode,
    mode_by_vis,
)

__all__ = ["DecodeResult", "SstvDecoder", "decode", "decode_file", "identify"]

ProgressCallback = Callable[[float], None]
ImageCallback = Callable[[np.ndarray], None]

_SYNC_TOLERANCE_HZ = 200.0

# Each scan is read slightly inside its nominal window.  Two effects live at the
# ends of a scan: the baseband filter's finite impulse response has not settled,
# and the neighbouring interval's tone is still arriving.  In practice the worst
# case is a chroma scan whose tail runs into the next line's 1200 Hz sync pulse,
# which drags the last few chroma pixels far off colour and paints a ragged strip
# down the edge of the picture.  A 2 ms trim measured best across every mode
# family; it costs about 1.5% of a scan's width and removes the artefact entirely.
_SCAN_MARGIN_S = 0.002


@dataclass
class DecodeResult:
    """Everything the decoder learned, including how trustworthy it is."""

    image: np.ndarray
    mode: ModeSpec | None
    sample_rate: int
    vis_code: int | None = None
    vis_confidence: float = 0.0
    header_end: float = 0.0
    lines_decoded: int = 0
    total_lines: int = 0
    measured_sync_hz: float = 0.0
    measured_black_hz: float = 0.0
    measured_white_hz: float = 0.0
    frequency_offset_hz: float = 0.0
    sync_errors: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    mode_source: str = "none"
    """How the mode was decided: ``vis``, ``guessed``, ``requested`` or ``none``.

    A picture decoded from a *guessed* mode is worth showing -- the alternative is
    a blank panel -- but the operator should know the mode was inferred from the
    line rhythm rather than read off the air, because a wrong guess mis-frames the
    picture even when the signal itself is perfectly good.
    """

    mode_confidence: float = 0.0
    """For a guessed mode, how clearly it beat the alternatives (0-1)."""

    mode_candidates: list[tuple[str, float]] = field(default_factory=list)
    """The modes considered, best first, with their scores.  A runner-up that is
    close to the winner is the sign of an uncertain guess."""

    @property
    def image_size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return (w, h)

    @property
    def complete(self) -> bool:
        return self.total_lines > 0 and self.lines_decoded >= self.total_lines

    @property
    def coverage(self) -> float:
        if not self.total_lines:
            return 0.0
        return self.lines_decoded / self.total_lines

    @property
    def sync_error_hz(self) -> float:
        """Root-mean-square deviation of the detected sync pulses from 1200 Hz."""
        if not self.sync_errors:
            return 0.0
        return float(np.sqrt(np.mean(np.square(self.sync_errors))))


class _Planes:
    """Accumulators for the decoded image, in whatever colour space the mode uses."""

    def __init__(self, mode: ModeSpec) -> None:
        width, height = mode.resolution
        self.mode = mode
        self.width = width
        self.height = height
        self.luminance = np.zeros((height, width), dtype=np.float64)
        self.cr = np.zeros((height, width), dtype=np.float64)
        self.cb = np.zeros((height, width), dtype=np.float64)
        self.red = np.zeros((height, width), dtype=np.float64)
        self.green = np.zeros((height, width), dtype=np.float64)
        self.blue = np.zeros((height, width), dtype=np.float64)
        # Chroma-difference modes that send one component per line (Robot 36)
        # leave the other component unknown for that line, so each is tracked
        # separately and merged once every line has been seen.
        self.cr_seen = np.zeros(height, dtype=bool)
        self.cb_seen = np.zeros(height, dtype=bool)

    def scatter(self, channel: Channel, rows: Sequence[int], values: np.ndarray) -> None:
        """Write one decoded scan into the right plane."""
        for row in rows:
            if row < 0 or row >= self.height:
                continue
            target = {
                Channel.LUMINANCE: self.luminance,
                Channel.CR: self.cr,
                Channel.CB: self.cb,
                Channel.RED: self.red,
                Channel.GREEN: self.green,
                Channel.BLUE: self.blue,
            }.get(channel)
            if target is None:
                continue
            data = values
            if data.size != self.width:
                data = np.interp(
                    np.linspace(0.0, data.size - 1, self.width),
                    np.arange(data.size, dtype=np.float64),
                    data,
                )
            target[row] = data
            if self.mode.chroma_on_even_only:
                if channel is Channel.CR:
                    self.cr_seen[row] = True
                elif channel is Channel.CB:
                    self.cb_seen[row] = True

    def to_rgb(self) -> np.ndarray:
        if self.mode.color_space == "RGB":
            rgb = np.stack((self.red, self.green, self.blue), axis=-1)
        else:
            cr, cb = self.cr, self.cb
            if self.mode.chroma_on_even_only:
                cr, cb = self._fill_partial_chroma()
            rgb = colorspace.ycrcb_to_rgb(self.luminance, cr, cb)
        return np.clip(rgb, 0.0, 255.0).astype(np.uint8)

    def _fill_partial_chroma(self) -> tuple[np.ndarray, np.ndarray]:
        """Give every row both chroma components.

        Robot 36 alternates R-Y and B-Y line by line, so on any given row one of
        the two arrived on the line above or below.  Copying the nearest known
        value reproduces the 4:2:0 vertical sharing the mode is defined with.
        """
        cr = self.cr.copy()
        cb = self.cb.copy()
        for plane, seen in ((cr, self.cr_seen), (cb, self.cb_seen)):
            known = np.flatnonzero(seen)
            if known.size == 0:
                plane[:, :] = 128.0
                continue
            if known.size == self.height:
                continue
            # Nearest known row for every row, then copy it across.
            for row in range(self.height):
                nearest = known[np.argmin(np.abs(known - row))]
                plane[row] = self.cr[nearest] if plane is cr else self.cb[nearest]
        return cr, cb


def identify(samples, sample_rate: int = 48000) -> ModeSpec | None:
    """Return the mode a recording announces, or ``None`` if no header is found."""
    header = vis_module.decode_vis(samples, sample_rate)
    if header is None:
        return None
    return mode_by_vis(header.code)


class SstvDecoder:
    """Decodes one transmission at a time.

    The decoder is deliberately explicit about what it does at each stage so a
    failure can be attributed: header, then line sync, then pixel extraction.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 48000,
        mode: ModeSpec | str | None = None,
        skew_correction: bool = True,
        offset_correction: bool = True,
        progress: ProgressCallback | None = None,
        on_image: ImageCallback | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.requested_mode = get_mode(mode) if isinstance(mode, str) else mode
        self.skew_correction = skew_correction
        self.offset_correction = offset_correction
        self.progress = progress
        self.on_image = on_image
        self._reset()

    # -- state ------------------------------------------------------------
    def _reset(self) -> None:
        self._planes: _Planes | None = None
        self._mode: ModeSpec | None = None
        self._header: vis_module.VisHeader | None = None
        self._lines = 0
        self._sync_errors: list[float] = []
        self._notes: list[str] = []
        self._measured_sync = 0.0
        self._measured_black = 0.0
        self._measured_white = 0.0
        self._offset = 0.0
        self._last_image_line = -1
        # How the mode was decided.  Reset here because a decoder instance can be
        # reused, and a stale "vis" would mislabel a later guess.
        self.mode_source = "requested" if self.requested_mode is not None else "none"
        self.mode_confidence = 0.0
        self.mode_candidates: list[tuple[str, float]] = []
        self._measured_line_period: float | None = None

    # -- entry points -----------------------------------------------------
    def decode(self, samples: Sequence[float] | np.ndarray) -> DecodeResult:
        """Decode a complete recording."""
        self._reset()
        audio = np.asarray(samples, dtype=np.float64)
        if audio.size == 0:
            return self._result()

        track = dsp.instantaneous_frequency(audio, self.sample_rate)
        header = vis_module.decode_vis(audio, self.sample_rate, track=track)
        # A VIS header is only believed if it looks like one.  The search window
        # lands on any 300 ms of 1900 Hz followed by tone pairs, and a picture's own
        # brightness transitions can look just like that: two PD recordings yielded
        # a "code 0" header with no confidence and no stop bit, and because code 0 is
        # not a mode, decoding gave up instead of carrying on.
        if header is not None and not self._header_is_plausible(header):
            self._notes.append(
                f"ignored a doubtful VIS header (code {header.code}, confidence "
                f"{header.confidence * 100:.0f}%, stop bit "
                f"{'present' if header.stop_ok else 'missing'})")
            header = None
        self._header = header
        mode = self.requested_mode
        start = 0.0
        if mode is not None:
            self.mode_source = "requested"
            start = self._header.end if self._header is not None else 0.0
            if self._header is not None:
                from_header = mode_by_vis(self._header.code)
                if from_header is not None and from_header.name == self.requested_mode.name:
                    self.mode_source = "vis"
        elif self._header is not None and mode_by_vis(self._header.code) is not None:
            mode = mode_by_vis(self._header.code)
            self.mode_source = "vis"
            start = self._header.end
        else:
            if self._header is not None:
                self._notes.append(
                    f"VIS code {self._header.code} is not a mode this program "
                    f"implements; identifying the mode from the signal instead")
            # Either there was no header, or it named a mode this program does not
            # implement.  Neither is a reason to show nothing: every mode repeats a
            # sync pulse once per line, and the line period is what distinguishes the
            # modes, so the mode can be inferred from the rhythm of the sync pulses
            # alone.  A possibly-mis-framed picture beats a blank panel, as long as it
            # is labelled as a guess.
            mode, start, self.mode_confidence, self.mode_candidates = self._guess_mode(track)
            if mode is None:
                self._notes.append(
                    "no usable VIS header and no recognisable line rhythm; "
                    "nothing could be decoded")
                return self._result()
            self.mode_source = "guessed"
            best_score = self.mode_candidates[0][1] if self.mode_candidates else 0.0
            runner_up = self.mode_candidates[1][1] if len(self.mode_candidates) > 1 else 0.0
            self._notes.append(
                f"no VIS header used; inferred {mode.name} from the sync pulse rhythm "
                f"(confidence {self.mode_confidence * 100:.0f}%, "
                f"next best scored {runner_up / best_score * 100:.0f}% as well)"
                if best_score else
                f"no VIS header used; inferred {mode.name} from the sync pulse rhythm")
        self._mode = mode

        disc = dsp.Discriminator(track, self.sample_rate)
        if self.mode_source == "guessed":
            # The inferred grid is coarse; solve for the real one before decoding.
            start = self._refine_grid(disc, self._mode, start)
            self._measure_line_rate(disc, self._mode, start)
        self._calibrate(disc, start)
        if self.offset_correction and abs(self._offset) > 1.0:
            disc = self._shifted(disc, self._offset)
            self._notes.append(f"corrected a {self._offset:+.0f} Hz tuning offset")

        self._planes = _Planes(self._mode)
        self._decode_lines(disc, start)
        return self._result()

    # -- mode inference ---------------------------------------------------
    @staticmethod
    def _header_is_plausible(header) -> bool:
        """Whether a decoded VIS header is credible enough to act on.

        Two things have to hold, and both are facts about the header rather than
        guesses about the signal: the stop bit must be a 1200 Hz sync pulse, and the
        data tones must actually have been close to the two legal tones.  Without
        the first, a run of picture tone pairs gets read as a header; without the
        second, a header whose tones were nowhere near legal values is believed
        anyway.  The parity bit is *not* required, because a single wrong bit is
        common on a weak signal and rejecting the header outright would cost the
        whole picture -- the mode it names is verified against the mode table
        instead.
        """
        if not header.stop_ok:
            return False
        return header.confidence > 0.0

    def _infer_mode(track: np.ndarray, sample_rate: int,
                    notes: list[str] | None = None,
                    max_seconds: float = 12.0,
                    ) -> tuple[ModeSpec | None, float, float, list[tuple[str, float]]]:
        """Infer the mode, and where its line grid starts, from the sync rhythm.

        A transmission that began before listening did cannot be identified from a
        VIS header, but it can still be decoded: every mode repeats a sync pulse once
        per line, and the line period is what distinguishes the modes.  Scoring a
        "sustained sync tone present" signal against each candidate period therefore
        says which mode is on the air and where its lines begin.

        Shared by the offline decoder and the live one so both reach the same answer
        for the same audio.  *notes* collects an explanation if one is wanted.
        """
        probe = SstvDecoder(sample_rate=sample_rate)
        probe._notes = notes if notes is not None else []
        return probe._guess_mode(track, max_seconds=max_seconds)

    def _guess_mode(self, track: np.ndarray, max_seconds: float = 12.0,
                    ) -> tuple[ModeSpec | None, float, float,
                               list[tuple[str, float]]]:
        """Infer the mode from the line rhythm, for a signal with no VIS header.

        Every candidate mode has a different line period, so scoring a "sync tone
        present" signal against each period says which mode is arriving and where
        its lines start.  The winner is taken only if it clearly beats the field; a
        mode that fits no better than its rivals would produce a picture that is
        mis-framed in a way the operator cannot see from the image alone.

        The sync rhythm alone cannot separate every pair.  Robot 72 sends a sync
        every 150 ms within its 300 ms line, so a Robot 36 transmission fits Robot
        72's period exactly as well as its own -- the two are indistinguishable from
        timing.  Where the fit scores are close, the tie is broken by decoding a few
        lines with each candidate and preferring whichever wastes less of the
        brightness range, which is what distinguishes a correctly framed picture
        from one read at the wrong line rate.

        Returns ``(mode, start_seconds, confidence, ranked_candidates)``.
        """
        envelope = dsp.sync_envelope(track, self.sample_rate, SYNC_HZ)
        duration = track.size / self.sample_rate
        # A few lines are enough to recognise the rhythm; using the whole recording
        # would make this the slow part of decoding a long one.
        if duration > max_seconds:
            envelope = envelope[: int(max_seconds * self.sample_rate)]

        scored: list[tuple[float, ModeSpec, float]] = []
        for spec in MODE_LIST:
            offset, score = dsp.best_periodic_offset(
                envelope, self.sample_rate, spec.line_seconds)
            if score > 0.0:
                scored.append((score, spec, offset))
        if not scored:
            return None, 0.0, 0.0, []

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score = scored[0][0]
        if best_score < 0.25:
            # Too little sync structure to be worth guessing from.
            return None, 0.0, 0.0, [(s.name, sc) for sc, s, _ in scored]

        # Is the timing decisive?  If the leader's period fits far better than
        # anything else, that settles it.  If a rival fits just as well, the rhythm
        # cannot tell them apart and only then is it worth costing a few lines to
        # look at the picture.
        runner_up_fit = scored[1][0] if len(scored) > 1 else 0.0
        ambiguous = runner_up_fit >= best_score * 0.98

        if ambiguous:
            shortlist = scored[:3]
            disc = dsp.Discriminator(track, self.sample_rate)
            ranked: list[tuple[float, ModeSpec, float]] = []
            for fit, spec, offset in shortlist:
                start = self._grid_start(offset, spec)
                quality = self._quick_quality(disc, spec, start)
                # Fit still leads -- it is what identifies the mode family -- with
                # picture quality settling the cases timing genuinely cannot.
                ranked.append((0.6 * fit + 0.4 * quality, spec, start))
        else:
            ranked = [(fit, spec, self._grid_start(offset, spec))
                      for fit, spec, offset in scored]
        ranked.sort(key=lambda item: item[0], reverse=True)

        combined_best = ranked[0][0]
        runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
        confidence = max(0.0, 1.0 - (runner_up / combined_best if combined_best else 1.0))
        if ambiguous:
            # The answer rests on picture quality rather than a clean rhythm match:
            # real evidence, but weaker, so it is not reported as certainty.
            confidence *= 0.75
        candidates = [(spec.name, score) for score, spec, _ in ranked]
        return ranked[0][1], ranked[0][2], confidence, candidates

    @staticmethod
    def _grid_start(offset: float, mode: ModeSpec) -> float:
        """The first sync pulse at or after t=0 for a mode's line grid."""
        period = mode.line_seconds
        start = offset % period
        while start - period >= 0:
            start -= period
        return max(0.0, start)

    def _quick_quality(self, disc: dsp.Discriminator, mode: ModeSpec,
                       start: float) -> float:
        """How much of the brightness range a candidate mode actually uses.

        Decodes a handful of lines at the candidate's line rate and measures the
        proportion of scan samples that land inside the usable range rather than
        pinned at black or white.  A mode read at the wrong line rate integrates
        over the wrong spans, so its samples pile up against the limits; a correctly
        framed picture spreads across the range.  This is the evidence that
        separates modes whose sync timing is identical.
        """
        if mode.line_count == 0 or mode.line_seconds <= 0:
            return 0.0
        # Compare like with like: the same span of time for every candidate, so a
        # mode with a short line is not favoured merely for having more lines.
        budget_seconds = 1.6
        lines = int(min(mode.line_count, max(4, budget_seconds / mode.line_seconds)))
        inside = 0
        total = 0
        for line_index in range(lines):
            base = start + line_index * mode.line_seconds
            if base + mode.line_seconds > disc.duration:
                break
            offset = 0.0
            for seg in self._resolved_segments(mode, line_index):
                if seg.is_image:
                    count = self._decoded_pixels(mode, seg)
                    if count:
                        margin = min(_SCAN_MARGIN_S, seg.seconds * 0.2)
                        values = disc.slice_values(base + offset + margin,
                                                   base + offset + seg.seconds - margin,
                                                   count)
                        inside += int(np.count_nonzero((values > 1.0) & (values < 254.0)))
                        total += values.size
                offset += seg.seconds
        if total == 0:
            return 0.0
        return inside / total


    def decode_file(self, path: str, **kwargs) -> DecodeResult:
        """Load a WAV file and decode it."""
        from . import audio as audio_module

        samples, rate = audio_module.read_wav(path)
        return SstvDecoder(sample_rate=rate, **{**self._options(), **kwargs}).decode(samples)

    def _options(self) -> dict:
        return {
            "mode": self.requested_mode,
            "skew_correction": self.skew_correction,
            "offset_correction": self.offset_correction,
            "progress": self.progress,
            "on_image": self.on_image,
        }

    # -- calibration ------------------------------------------------------
    def _calibrate(self, disc: dsp.Discriminator, start: float) -> None:
        """Measure the tones actually received, so errors can be reported."""
        mode = self._mode
        if mode is None:
            return
        # The header's leader tone is a known 1900 Hz reference, so a deviation
        # there is a genuine tuning error and can safely be corrected out.
        if self._header is not None:
            leader = disc.average(0.15, 0.28)
            if abs(leader - 1900.0) < 400.0:
                self._offset = leader - 1900.0
        sync_hz = disc.average(start, start + min(0.004, mode.line_seconds * 0.4))
        self._measured_sync = sync_hz
        if mode.segments and mode.segments[0].channel is Channel.SYNC:
            self._notes.append(f"received sync at {sync_hz:.0f} Hz (nominal 1200)")
        # With no header there is no known tone to calibrate against, and the sync
        # pulse is not a substitute.  Measured a little inside the pulse it reads up
        # to tens of hertz high -- on a Scottie S1 recording it reported 1216 Hz and
        # "correcting" by those 16 Hz turned a 20.8 dB picture into a 6.9 dB one,
        # because a frequency shift moves every scan horizontally.  So nothing is
        # corrected here: a genuine mistuning is better reported than acted upon when
        # there is no reference to confirm it, and the mode's own black and white
        # limits still map the tones to levels correctly.
        if self._header is None and abs(sync_hz - SYNC_HZ) > 40.0:
            self._notes.append(
                f"no VIS header to calibrate against; the pulse at the line start "
                f"measured {sync_hz:.0f} Hz rather than {SYNC_HZ:.0f} Hz, which is "
                f"not reliable enough to correct for, so no tuning correction "
                f"was applied")

    def _shifted(self, disc: dsp.Discriminator, offset: float) -> dsp.Discriminator:
        return dsp.Discriminator(
            disc.frequencies - offset, self.sample_rate,
            black=BLACK_HZ, white=WHITE_HZ,
        )

    # -- line decoding ----------------------------------------------------
    def _decode_lines(self, disc: dsp.Discriminator, start: float) -> None:
        mode = self._mode
        assert mode is not None and self._planes is not None
        line_seconds = mode.line_seconds
        if self._measured_line_period is not None:
            # A rate already measured while locating the grid.  Using it here keeps
            # the framing consistent with the start that was chosen for it; leaving
            # it out would decode from a start aimed at one rate using another.
            line_seconds = self._measured_line_period
        elif self.skew_correction:
            line_seconds = self._estimate_line_period(disc, start, mode)
        position = start
        end_of_audio = disc.duration

        for line_index in range(mode.line_count):
            if position >= end_of_audio:
                self._notes.append("audio ended before the picture was complete")
                break
            segments = self._resolved_segments(mode, line_index)
            self._read_line(disc, position, segments, line_index)
            position += line_seconds
            self._lines += 1
            if self.progress is not None:
                self.progress((line_index + 1) / mode.line_count)
            if self.on_image is not None and (line_index % 8 == 0 or
                                              line_index == mode.line_count - 1):
                self.on_image(self._planes.to_rgb())

    def _resolved_segments(self, mode: ModeSpec, line_index: int):
        """Resolve per-line variation in the scan plan.

        Robot 36's separator tone is a parity marker: 1500 Hz announces R-Y and
        2300 Hz announces B-Y.  The decoder decides which chroma a line carries
        from that tone rather than from the line number, so a recording that was
        cut mid-pair cannot end up with red and blue swapped.
        """
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

    def _find_sync_edge(self, disc: dsp.Discriminator, expected: float,
                        mode: ModeSpec, window: float = 0.006) -> float | None:
        """Locate where a sync pulse *begins*, searching around *expected*.

        This is what the inferred-grid path needs, and it is not what
        :meth:`_find_sync` provides.  That one looks for the frequency nearest the
        sync tone inside a narrow window, which is right when the pulse position is
        already known to a fraction of a millisecond -- but inside a 20 ms pulse,
        every position is equally "nearest the sync tone", so the answer lands
        somewhere in the middle and varies from line to line.  Measuring a period
        between two such answers gave errors of hundreds of parts per million, which
        drift several milliseconds over a picture and blur every scan.

        A pulse has an edge, though, and the edge is a fixed fact about the signal:
        it is the first place the frequency settles on the sync tone.  Finding that
        gives the same answer every time, whatever the pulse's length.

        The probe is deliberately shorter than any sync pulse -- Robot's is the
        shortest at 9 ms -- because a probe as long as the pulse stops being able to
        see where the pulse starts.  At 1 ms, Robot 36 and Scottie decoded badly
        while the PD family, whose sync is 20 ms, decoded well: the probe was
        straddling the edge and reporting a position well inside the pulse.
        """
        lo = max(0.0, expected - window)
        probe = 0.0004
        hi = min(disc.duration - probe, expected + window)
        if hi <= lo:
            return None
        steps = max(8, int((hi - lo) * self.sample_rate / 4))
        tolerance = min(_SYNC_TOLERANCE_HZ, 150.0)
        for offset in np.linspace(lo, hi, steps):
            if abs(disc.average(offset, offset + probe) - SYNC_HZ) <= tolerance:
                return float(offset)
        return None

    def _refine_grid(self, disc: dsp.Discriminator, mode: ModeSpec,
                     coarse: float) -> float:
        """Aim the coarse line grid at a real sync pulse.

        The periodic search that identifies the mode reports the grid's phase to
        roughly half a millisecond -- under a pixel of the picture -- which is
        already close.  What it does not do is guarantee that the phase it reports
        lands *on* a pulse rather than in a gap beside one, and starting a line in a
        gap shifts every scan.  So the coarse value is used only to aim, and the
        decoder's own sync search then puts the start on the pulse itself.

        Locating a pulse edge is good to a fraction of a millisecond, but that is
        not enough for the *line rate* over a whole picture, so the rate is measured
        separately over a long baseline; see :meth:`_measure_line_rate`.
        """
        anchor = self._find_sync_edge(disc, coarse, mode)
        if anchor is None:
            anchor = self._find_sync(disc, coarse, mode, window=0.006)
        return anchor if anchor is not None else coarse

    def _measure_line_rate(self, disc: dsp.Discriminator, mode: ModeSpec,
                           start: float) -> None:
        """Measure the true line rate over as long a baseline as the audio allows.

        The per-pulse error is a fraction of a millisecond, which over a few dozen
        lines is a few hundred parts per million -- and that drifts several
        milliseconds by the end of a picture, enough to read every scan at the wrong
        place.  Averaging over half the picture divides the error away, so a rate
        that agrees with nominal to within a tight tolerance is treated as nominal,
        which is what the encoder that produced the signal actually used.
        """
        span = min(max(4, mode.line_count // 2),
                   int((disc.duration - start - 0.1) / mode.line_seconds))
        if span < 2:
            return
        expected = start + span * mode.line_seconds
        far = self._find_sync_edge(disc, expected, mode, window=0.012)
        if far is None:
            far = self._find_sync(disc, expected, mode, window=0.012)
        if far is None:
            return
        measured = (far - start) / span
        if not (0.9 * mode.line_seconds < measured < 1.1 * mode.line_seconds):
            return
        drift = abs(measured - mode.line_seconds)
        if drift > max(2e-4, 0.0002 * mode.line_seconds):
            self._notes.append(
                f"line period measured at {measured * 1000:.3f} ms "
                f"({(measured - mode.line_seconds) * 1e6 / mode.line_seconds:+.0f} ppm "
                f"against nominal); keeping nominal timing")
        else:
            self._measured_line_period = measured

    def _estimate_line_period(self, disc: dsp.Discriminator, start: float,
                              mode: ModeSpec) -> float:
        """Measure the real sync-to-sync interval and use it as the line clock.

        Re-locking on every individual sync seems like the robust choice but is
        actively harmful in the PD family, where the sync pulse is 20 ms long:
        searching a window around the expected position just finds the nearest
        edge *inside* that pulse, so each line starts a millisecond or two off and
        the scan windows drift randomly.  Measuring one period and then running
        open-loop is both simpler and far more accurate.

        The baseline matters more than the per-pulse precision.  A single sync
        edge can only be located to a fraction of a millisecond, so estimating the
        period from two or three lines leaves an error of a few tenths of a
        millisecond per line -- which accumulates into a whole line of drift by
        the end of a long transmission.  Spreading the measurement over many
        lines divides that error by the baseline.
        """
        nominal = mode.line_seconds
        span = max(4, min(40, mode.line_count // 8 or 4))
        # Measure the interval between two sync pulses directly, rather than
        # averaging their positions against the nominal timeline: that way any
        # residual error in locating the first pulse cancels out of the
        # difference, instead of being divided by the baseline and then applied
        # to every line of the picture.
        first = self._find_sync(disc, start, mode)
        last = self._find_sync(disc, start + span * nominal, mode)
        if first is not None and last is not None:
            return self._accept_period((last - first) / span, nominal)
        for index in (span // 2, 2, 1):
            anchor = self._find_sync(disc, start, mode)
            found = self._find_sync(disc, start + index * nominal, mode)
            if anchor is not None and found is not None:
                return self._accept_period((found - anchor) / index, nominal)
        self._notes.append("sync period could not be measured; using nominal timing")
        return nominal

    def _accept_period(self, measured: float, nominal: float) -> float:
        """Adopt a measured line period only when it is clearly a real drift.

        Locating a sync pulse edge is good to roughly a twentieth of a
        millisecond, which for a 150 ms line is already a few hundredths of a
        percent.  Adopting an estimate that small and then trusting it for
        hundreds of lines turns a measurement artefact into a visible skew, so
        anything inside the tolerance is treated as "on time" and the nominal
        period is kept.  Drifts that matter -- a sound card running slow, a tape
        transfer -- are far larger than this and still get corrected.
        """
        tolerance = max(2e-4, 0.0008 * nominal)
        drift = measured - nominal
        if abs(drift) < tolerance:
            return nominal
        if abs(drift) > 0.05 * nominal:
            self._notes.append(
                f"measured line period {measured * 1000:.1f} ms differs from nominal "
                f"{nominal * 1000:.1f} ms by more than 5%; keeping nominal timing"
            )
            return nominal
        self._notes.append(
            f"line period measured at {measured * 1000:.3f} ms "
            f"({drift * 1000:+.3f} ms against nominal); correcting for skew"
        )
        return measured

    def _find_sync(self, disc: dsp.Discriminator, expected: float,
                   mode: ModeSpec, window: float | None = None) -> float | None:
        """Locate the sync pulse nearest *expected*.

        The search window must stay well inside the sync pulse.  PD's sync is
        20 ms long, so a window anywhere near that wide contains nothing but sync
        and the "best" position becomes arbitrary -- which shows up as a line
        clock that runs consistently fast.  A window of about a millisecond lands
        on the leading edge, where the pulse actually begins.

        *window* widens the search, which is only safe when the caller knows it is
        aiming at a pulse roughly rather than precisely: it is used while picking up
        an inferred line grid, where the guess may be several milliseconds out.  The
        wider search then finds the *middle* of the pulse rather than its edge, so
        the result is only ever a starting point for a narrower search.
        """
        if window is None:
            window = min(0.0012, mode.line_seconds * 0.1)
        lo = max(0.0, expected - window)
        hi = min(disc.duration, expected + window)
        if hi - lo < 1e-4:
            return None
        steps = max(16, int((hi - lo) * self.sample_rate / 16))
        best_at, best_error = None, None
        for offset in np.linspace(lo, hi, steps):
            error = abs(disc.average(offset, offset + 0.001) - SYNC_HZ)
            if best_error is None or error < best_error:
                best_error, best_at = error, offset
        if best_error is None or best_error > 400.0:
            return None
        self._sync_errors.append(float(best_error))
        return best_at

    def _read_line(self, disc: dsp.Discriminator, base: float, segments,
                   line_index: int) -> None:
        planes = self._planes
        mode = self._mode
        assert planes is not None and mode is not None
        luminance_segments = [s for s in segments if s.channel is Channel.LUMINANCE]
        rows = ([line_index * 2, line_index * 2 + 1]
                if len(luminance_segments) == 2 else [line_index])
        luminance_index = 0
        offset = 0.0
        for seg in segments:
            if seg.is_image:
                count = self._decoded_pixels(mode, seg)
                # Keep the window inside the segment so the settled part of the
                # filter is what gets integrated.
                margin = min(_SCAN_MARGIN_S, seg.seconds * 0.2)
                values = (disc.slice_values(base + offset + margin,
                                            base + offset + seg.seconds - margin,
                                            count)
                          if count else np.zeros(0))
                if seg.channel is Channel.LUMINANCE:
                    row = rows[min(luminance_index, len(rows) - 1)]
                    luminance_index += 1
                    planes.scatter(Channel.LUMINANCE, [row], values)
                else:
                    planes.scatter(seg.channel, rows, values)
            offset += seg.seconds

    @staticmethod
    def _decoded_pixels(mode: ModeSpec, seg) -> int:
        """How many distinct pixels a scan carries.

        Robot 36 spends 320 transmitted samples on 160 chroma pixels -- each pixel
        is sent twice -- so reading all 320 as separate pixels would stretch the
        colour sideways by a factor of two and leave a band of garbage down the
        edge of the picture.  Every other mode sends one sample per pixel.
        """
        if (mode.chroma_on_even_only and seg.channel in (Channel.CR, Channel.CB)
                and mode.chroma_subsample > 1):
            return seg.pixels // mode.chroma_subsample
        return seg.pixels

    # -- result -----------------------------------------------------------
    def _result(self) -> DecodeResult:
        mode = self._mode
        image = (self._planes.to_rgb() if self._planes is not None
                 else np.zeros((0, 0, 3), dtype=np.uint8))
        return DecodeResult(
            image=image,
            mode=mode,
            sample_rate=self.sample_rate,
            vis_code=(self._header.code if self._header else None),
            vis_confidence=(self._header.confidence if self._header else 0.0),
            header_end=(self._header.end if self._header else 0.0),
            lines_decoded=self._lines,
            total_lines=(mode.line_count if mode else 0),
            measured_sync_hz=self._measured_sync,
            measured_black_hz=self._measured_black,
            measured_white_hz=self._measured_white,
            frequency_offset_hz=self._offset,
            sync_errors=list(self._sync_errors),
            notes=list(self._notes),
            mode_source=self.mode_source,
            mode_confidence=self.mode_confidence,
            mode_candidates=list(self.mode_candidates),
        )


# --------------------------------------------------------------------------
# Convenience wrappers
# --------------------------------------------------------------------------

def decode(samples, sample_rate: int = 48000,
           mode: ModeSpec | str | None = None, **kwargs) -> DecodeResult:
    """Decode audio into an image, identifying the mode from the header if needed."""
    return SstvDecoder(sample_rate=sample_rate, mode=mode, **kwargs).decode(samples)


def decode_file(path: str, mode: ModeSpec | str | None = None,
                **kwargs) -> DecodeResult:
    """Decode a WAV file into an image."""
    from . import audio as audio_module

    samples, rate = audio_module.read_wav(path)
    return decode(samples, rate, mode, **kwargs)
