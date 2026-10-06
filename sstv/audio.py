"""Audio input/output for SSTV Studio.

Everything here is built on the standard library plus Windows' own multimedia
API (``winmm.dll``, reached through :mod:`ctypes`).  No PortAudio, no
``sounddevice``, no PyAudio -- installed copies of those are unreliable in this
environment and the native API covers everything SSTV needs (16-bit PCM mono
capture and playback plus ordinary RIFF/WAVE files).

Public surface
--------------
* :func:`read_wav` / :func:`write_wav` -- RIFF/WAVE loading and saving
* :func:`list_input_devices` / :func:`list_output_devices`
* :class:`AudioPlayer` -- asynchronous speaker playback with stop/pause
* :class:`AudioRecorder` -- asynchronous microphone capture to a buffer
"""

from __future__ import annotations

import array
import ctypes
import os
import sys
import threading
import time
import wave
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "AudioError",
    "AudioPlayer",
    "AudioRecorder",
    "DEFAULT_SAMPLE_RATE",
    "DeviceInfo",
    "list_input_devices",
    "list_loopback_devices",
    "list_output_devices",
    "read_wav",
    "write_wav",
]

DEFAULT_SAMPLE_RATE = 48000
"""Sample rate used for all internally generated SSTV audio (Hz)."""

SAMPLES_PER_BLOCK = 32768
"""Samples per waveOut/waveIn buffer; ~0.68 s at 48 kHz."""

WAVE_MAPPER = ctypes.c_uint(-1).value


class AudioError(RuntimeError):
    """Raised when a Windows multimedia call fails."""


# --------------------------------------------------------------------------
# waveOut / waveIn error text
# --------------------------------------------------------------------------

_MMSYSERR = {
    0: "no error",
    1: "unspecified error",
    2: "device ID out of range",
    3: "driver failed",
    4: "not enough memory",
    5: "no driver",
    6: "invalid handle",
    7: "no device",
    8: "bad format",
    11: "not enabled",
    20: "bad parameter",
    32: "no driver",
    33: "bad interface",
    34: "unsupported function",
}


def _mm_text(rc: int) -> str:
    return _MMSYSERR.get(rc, f"MMSYSERR {rc}")


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise AudioError(f"{what} failed: {_mm_text(rc)} (code {rc})")


# --------------------------------------------------------------------------
# ctypes glue
# --------------------------------------------------------------------------

if sys.platform == "win32":
    _winmm = ctypes.WinDLL("winmm")
else:  # pragma: no cover - the application targets Windows
    _winmm = None


class _WAVEFORMATEX(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("wFormatTag", wintypes.WORD),
        ("nChannels", wintypes.WORD),
        ("nSamplesPerSec", wintypes.DWORD),
        ("nAvgBytesPerSec", wintypes.DWORD),
        ("nBlockAlign", wintypes.WORD),
        ("wBitsPerSample", wintypes.WORD),
        ("cbSize", wintypes.WORD),
    ]


class _WAVEHDR(ctypes.Structure):
    _fields_ = [
        ("lpData", ctypes.c_void_p),
        ("dwBufferLength", wintypes.DWORD),
        ("dwBytesRecorded", wintypes.DWORD),
        ("dwUser", ctypes.c_void_p),
        ("dwFlags", wintypes.DWORD),
        ("dwLoops", wintypes.DWORD),
        ("lpNext", ctypes.c_void_p),
        ("reserved", ctypes.c_void_p),
    ]


class _WAVEINCAPSW(ctypes.Structure):
    _fields_ = [
        ("wMid", wintypes.WORD),
        ("wPid", wintypes.WORD),
        ("vDriverVersion", wintypes.DWORD),
        ("szPname", wintypes.WCHAR * 32),
    ]


class _WAVEOUTCAPSW(ctypes.Structure):
    _fields_ = [
        ("wMid", wintypes.WORD),
        ("wPid", wintypes.WORD),
        ("vDriverVersion", wintypes.DWORD),
        ("szPname", wintypes.WCHAR * 32),
        ("dwFormats", wintypes.DWORD),
        ("wChannels", wintypes.WORD),
        ("wReserved1", wintypes.WORD),
        ("dwSupport", wintypes.DWORD),
    ]


WHDR_DONE = 0x0000_0001
WAVE_FORMAT_PCM = 1

if _winmm is not None:
    _winmm.waveOutGetNumDevs.restype = wintypes.UINT
    _winmm.waveInGetNumDevs.restype = wintypes.UINT
    _winmm.waveOutGetDevCapsW.argtypes = [
        wintypes.UINT, ctypes.POINTER(_WAVEOUTCAPSW), wintypes.UINT
    ]
    _winmm.waveInGetDevCapsW.argtypes = [
        wintypes.UINT, ctypes.POINTER(_WAVEINCAPSW), wintypes.UINT
    ]
    for _name, _args in (
        ("waveOutOpen", [ctypes.POINTER(wintypes.HANDLE), wintypes.UINT,
                         ctypes.POINTER(_WAVEFORMATEX), ctypes.c_void_p,
                         ctypes.c_void_p, wintypes.DWORD]),
        ("waveOutPrepareHeader", [wintypes.HANDLE, ctypes.POINTER(_WAVEHDR), wintypes.UINT]),
        ("waveOutWrite", [wintypes.HANDLE, ctypes.POINTER(_WAVEHDR), wintypes.UINT]),
        ("waveOutUnprepareHeader", [wintypes.HANDLE, ctypes.POINTER(_WAVEHDR), wintypes.UINT]),
        ("waveOutClose", [wintypes.HANDLE]),
        ("waveOutReset", [wintypes.HANDLE]),
        ("waveInOpen", [ctypes.POINTER(wintypes.HANDLE), wintypes.UINT,
                        ctypes.POINTER(_WAVEFORMATEX), ctypes.c_void_p,
                        ctypes.c_void_p, wintypes.DWORD]),
        ("waveInPrepareHeader", [wintypes.HANDLE, ctypes.POINTER(_WAVEHDR), wintypes.UINT]),
        ("waveInAddBuffer", [wintypes.HANDLE, ctypes.POINTER(_WAVEHDR), wintypes.UINT]),
        ("waveInStart", [wintypes.HANDLE]),
        ("waveInStop", [wintypes.HANDLE]),
        ("waveInReset", [wintypes.HANDLE]),
        ("waveInUnprepareHeader", [wintypes.HANDLE, ctypes.POINTER(_WAVEHDR), wintypes.UINT]),
        ("waveInClose", [wintypes.HANDLE]),
    ):
        getattr(_winmm, _name).argtypes = _args


def _make_format(rate: int, channels: int = 1, bits: int = 16) -> _WAVEFORMATEX:
    fmt = _WAVEFORMATEX()
    fmt.wFormatTag = WAVE_FORMAT_PCM
    fmt.nChannels = channels
    fmt.nSamplesPerSec = rate
    fmt.wBitsPerSample = bits
    fmt.nBlockAlign = channels * bits // 8
    fmt.nAvgBytesPerSec = rate * fmt.nBlockAlign
    fmt.cbSize = 0
    return fmt


# --------------------------------------------------------------------------
# Device enumeration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DeviceInfo:
    """A single waveIn, waveOut or loopback endpoint."""

    index: int
    name: str
    kind: str  # "input" | "output" | "loopback"
    available: bool = True
    channels: int = 1
    device_id: str = ""   # set for loopback endpoints, which need the WASAPI id

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.name


def _require_windows() -> None:
    if _winmm is None:
        raise AudioError("audio device access requires Windows (winmm.dll)")


def list_output_devices() -> list[DeviceInfo]:
    """Return every waveOut (speaker) endpoint, mapper first."""
    _require_windows()
    devices = [DeviceInfo(-1, "System default output", "output")]
    for i in range(_winmm.waveOutGetNumDevs()):
        caps = _WAVEOUTCAPSW()
        rc = _winmm.waveOutGetDevCapsW(i, ctypes.byref(caps), ctypes.sizeof(caps))
        name = caps.szPname if rc == 0 else f"Output device {i}"
        devices.append(DeviceInfo(i, name or f"Output device {i}", "output",
                                  available=rc == 0, channels=max(1, caps.wChannels)))
    return devices


def list_input_devices() -> list[DeviceInfo]:
    """Return every waveIn (microphone) endpoint, mapper first."""
    _require_windows()
    devices = [DeviceInfo(-1, "System default input", "input")]
    for i in range(_winmm.waveInGetNumDevs()):
        caps = _WAVEINCAPSW()
        rc = _winmm.waveInGetDevCapsW(i, ctypes.byref(caps), ctypes.sizeof(caps))
        name = caps.szPname if rc == 0 else f"Input device {i}"
        devices.append(DeviceInfo(i, name or f"Input device {i}", "input",
                                  available=rc == 0))
    return devices


def list_loopback_devices() -> list[DeviceInfo]:
    """Output devices that can be listened to, for decoding what is playing.

    These are render endpoints rather than capture endpoints: choosing one means
    "listen to whatever this computer is sending to that device", which is how a
    receiver plugged into the line input, or audio playing in another program, is
    decoded without a cable.
    """
    try:
        from . import wasapi
    except Exception:
        return []
    out: list[DeviceInfo] = []
    for device in wasapi.list_loopback_devices():
        out.append(DeviceInfo(device.index, device.name, "loopback",
                              available=True, device_id=device.device_id))
    return out


# --------------------------------------------------------------------------
# WAV file I/O
# --------------------------------------------------------------------------

def read_wav(path: str) -> tuple["list[float]", int]:
    """Load a RIFF/WAVE file.

    Returns ``(samples, sample_rate)`` where *samples* are mono floats in
    ``[-1.0, 1.0]``; multi-channel files are averaged down to mono.  Uncompressed
    8/16/24/32-bit PCM and 32-bit float are supported.
    """
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.getnframes()
        comptype = wf.getcomptype()
        raw = wf.readframes(frames)
    if comptype != "NONE":
        raise AudioError(f"unsupported WAVE compression: {comptype}")
    if frames == 0 or not raw:
        return [], rate

    if width == 1:
        data = array.array("B")
        data.frombytes(raw[: len(raw)])
        step = 128.0
        values = [(b - 128) / step for b in data]
    elif width == 2:
        data = array.array("h")
        data.frombytes(raw[: len(raw) // 2 * 2])
        values = [v / 32768.0 for v in data]
    elif width == 3:
        values = []
        for i in range(0, len(raw) - 2, 3):
            v = raw[i] | (raw[i + 1] << 8) | (raw[i + 2] << 16)
            if v & 0x800000:
                v -= 1 << 24
            values.append(v / 8388608.0)
    elif width == 4:
        # Could be 32-bit int PCM or IEEE float; both appear in the wild.
        ints = array.array("i")
        ints.frombytes(raw[: len(raw) // 4 * 4])
        floats = array.array("f")
        floats.frombytes(raw[: len(raw) // 4 * 4])
        looks_like_float = any(
            (f != f) or abs(f) > 1e-30 for f in floats[: min(len(floats), 64)]
        ) and all(abs(f) <= 1.5 or f != f for f in floats[: min(len(floats), 256)])
        if looks_like_float:
            values = [float(f) for f in floats]
        else:
            values = [v / 2147483648.0 for v in ints]
    else:
        raise AudioError(f"unsupported sample width: {width * 8} bits")

    if channels > 1:
        n = len(values) // channels
        mono = [0.0] * n
        for c in range(channels):
            base = c
            for i in range(n):
                mono[i] += values[base + i * channels]
        inv = 1.0 / channels
        values = [v * inv for v in mono]
    return values, rate


def write_wav(
    path: str,
    samples: Sequence[float],
    rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = 1,
    ) -> str:
    """Write mono float samples to a 16-bit PCM RIFF/WAVE file.

    Samples are clipped to ``[-1, 1]``.  Returns *path*.
    """
    buf = array.array("h", [0] * len(samples))
    for i, s in enumerate(samples):
        if s > 1.0:
            s = 1.0
        elif s < -1.0:
            s = -1.0
        buf[i] = int(s * 32767.0)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(buf.tobytes())
    return path


# --------------------------------------------------------------------------
# Playback
# --------------------------------------------------------------------------

class AudioPlayer:
    """Play 16-bit PCM through a waveOut device without blocking the caller.

    The signal is streamed in blocks, so multi-minute SSTV transmissions start
    instantly and can be stopped mid-transmission.
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, device: int = -1) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._lock = threading.Lock()
        self._progress = 0.0
        self._state = "idle"
        self._error: str | None = None
        self._on_progress: Callable[[float], None] | None = None
        self._on_finish: Callable[[str], None] | None = None

    # -- state ------------------------------------------------------------
    @property
    def state(self) -> str:
        return self._state

    @property
    def is_playing(self) -> bool:
        return self._state in ("playing", "paused")

    @property
    def progress(self) -> float:
        """Fraction of the transmission already sent, 0.0 - 1.0."""
        return self._progress

    @property
    def error(self) -> str | None:
        return self._error

    def set_callbacks(
        self,
        on_progress: Callable[[float], None] | None = None,
        on_finish: Callable[[str], None] | None = None,
    ) -> None:
        """Register GUI callbacks.  *on_finish* receives the final state."""
        self._on_progress = on_progress
        self._on_finish = on_finish

    # -- control ----------------------------------------------------------
    def play(self, samples: Sequence[float]) -> None:
        """Start playback of *samples* (floats in ``[-1, 1]``)."""
        _require_windows()
        self.stop(wait=True)
        # Test the length rather than truthiness: the encoder hands us a NumPy
        # array, and `if not array` raises instead of being falsey.
        if samples is None or len(samples) == 0:
            raise AudioError("nothing to play")
        pcm = array.array("h", [0] * len(samples))
        for i, s in enumerate(samples):
            v = int((1.0 if s > 1.0 else -1.0 if s < -1.0 else s) * 32767.0)
            pcm[i] = v
        self._stop.clear()
        self._pause.clear()
        self._progress = 0.0
        self._error = None
        self._state = "playing"
        self._thread = threading.Thread(
            target=self._run, args=(pcm.tobytes(),), daemon=True, name="sstv-playback"
        )
        self._thread.start()

    def pause(self) -> None:
        """Pause playback; the device is reset and resumed from the same point."""
        if self._state == "playing":
            self._pause.set()
            self._state = "paused"

    def resume(self) -> None:
        """Resume a paused transmission."""
        if self._state == "paused":
            self._pause.clear()
            self._state = "playing"

    def stop(self, wait: bool = False) -> None:
        """Abort playback."""
        self._stop.set()
        self._pause.clear()
        thread = self._thread
        if thread is not None and thread.is_alive():
            if wait:
                thread.join(timeout=5.0)
        elif self._state != "error":
            self._state = "idle"

    def wait(self, timeout: float | None = None) -> bool:
        """Block until playback finishes.  Returns True if it completed."""
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    # -- worker -----------------------------------------------------------
    def _run(self, pcm: bytes) -> None:
        handle = wintypes.HANDLE()
        fmt = _make_format(self.sample_rate)
        rc = _winmm.waveOutOpen(
            ctypes.byref(handle), self.device if self.device >= 0 else WAVE_MAPPER,
            ctypes.byref(fmt), None, None, 0,
        )
        if rc != 0:
            self._error = f"cannot open playback device: {_mm_text(rc)}"
            self._state = "error"
            self._notify_finish()
            return

        block_bytes = SAMPLES_PER_BLOCK * 2
        headers: list[tuple[_WAVEHDR, ctypes.Array]] = []
        finished_state = "done"
        try:
            offset = 0
            total = len(pcm)
            sent = 0
            in_flight: list[tuple[_WAVEHDR, ctypes.Array]] = []
            while offset < total or in_flight:
                if self._stop.is_set():
                    finished_state = "stopped"
                    break

                while len(in_flight) < 3 and offset < total:
                    while self._pause.is_set() and not self._stop.is_set():
                        time.sleep(0.03)
                    if self._stop.is_set():
                        break
                    chunk = pcm[offset : offset + block_bytes]
                    offset += len(chunk)
                    buf = ctypes.create_string_buffer(chunk, len(chunk))
                    hdr = _WAVEHDR()
                    hdr.lpData = ctypes.cast(buf, ctypes.c_void_p)
                    hdr.dwBufferLength = len(chunk)
                    headers.append((hdr, buf))
                    _check(
                        _winmm.waveOutPrepareHeader(handle, ctypes.byref(hdr), ctypes.sizeof(hdr)),
                        "waveOutPrepareHeader",
                    )
                    _check(
                        _winmm.waveOutWrite(handle, ctypes.byref(hdr), ctypes.sizeof(hdr)),
                        "waveOutWrite",
                    )
                    in_flight.append((hdr, buf))

                if not in_flight:
                    break

                head = in_flight[0]
                t0 = time.time()
                while not (head[0].dwFlags & WHDR_DONE):
                    if self._stop.is_set():
                        break
                    if time.time() - t0 > 30.0:
                        break
                    time.sleep(0.01)
                if self._stop.is_set():
                    finished_state = "stopped"
                    break
                in_flight.pop(0)
                sent = offset - sum(h[0].dwBufferLength for h in in_flight)
                self._progress = min(1.0, sent / total) if total else 1.0
                if self._on_progress is not None:
                    try:
                        self._on_progress(self._progress)
                    except Exception:
                        pass
            else:
                self._progress = 1.0
        finally:
            if finished_state == "stopped":
                _winmm.waveOutReset(handle)
            for hdr, _buf in headers:
                _winmm.waveOutUnprepareHeader(handle, ctypes.byref(hdr), ctypes.sizeof(hdr))
            _winmm.waveOutClose(handle)

        self._state = "idle" if finished_state == "done" else "stopped"
        if finished_state == "done":
            self._progress = 1.0
        self._notify_finish()

    def _notify_finish(self) -> None:
        if self._on_finish is not None:
            try:
                self._on_finish(self._state)
            except Exception:
                pass


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------

class AudioRecorder:
    """Capture mono audio in the background, from a microphone or the speakers.

    Two sources are supported:

    * a ``waveIn`` input device, when *loopback* is False -- the microphone or a
      line input;
    * WASAPI loopback on a ``waveOut`` device, when *loopback* is True -- whatever
      the computer is playing, so a receiver feeding the speakers can be decoded
      with no cable between the radio and this program.

    Either way the captured audio is normalised to 16-bit mono at
    :attr:`sample_rate`, so the waterfall and the decoder see one representation.
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, device: int = -1, *,
                 loopback: bool = False, device_id: str | None = None) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self.loopback = bool(loopback)
        self.device_id = device_id
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._chunks: list[bytes] = []
        self._lock = threading.Lock()
        self._state = "idle"
        self._error: str | None = None
        self._seconds = 0.0
        self._peak = 0
        self._on_level: Callable[[float, float], None] | None = None
        # Set once the capture device is actually running.  Opening a device
        # takes a moment, and anything played before it is ready is simply not
        # captured -- which for SSTV means losing the VIS header at the very
        # start of the transmission, the one part that cannot be recovered.
        self._ready = threading.Event()

    # -- state ------------------------------------------------------------
    @property
    def state(self) -> str:
        return self._state

    @property
    def is_recording(self) -> bool:
        return self._state == "recording"

    @property
    def seconds(self) -> float:
        """Duration captured so far."""
        return self._seconds

    @property
    def peak(self) -> int:
        """Highest absolute sample value seen so far (0-32768)."""
        return self._peak

    @property
    def error(self) -> str | None:
        return self._error

    def wait_until_ready(self, timeout: float = 2.0) -> bool:
        """Block until the capture device is actually running.

        Opening a device takes a moment, and audio that plays before it is ready
        is not captured at all.  For SSTV that means losing the VIS header at the
        very start of a transmission, which is the one part that cannot be
        recovered from the rest of the signal.  Returns False on timeout or if the
        capture failed.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._ready.is_set():
                return True
            if self._state == "error":
                return False
            time.sleep(0.01)
        return self._ready.is_set()

    def set_level_callback(self, callback: Callable[[float, float], None] | None) -> None:
        """*callback(seconds, normalised_peak)* is called for every block."""
        self._on_level = callback

    # -- control ----------------------------------------------------------
    def start(self) -> None:
        """Begin capturing."""
        _require_windows()
        self.stop(wait=True)
        self._chunks = []
        self._stop.clear()
        self._error = None
        self._seconds = 0.0
        self._peak = 0
        self._state = "recording"
        self._ready.clear()
        target = self._run_loopback if self.loopback else self._run
        name = "sstv-loopback" if self.loopback else "sstv-capture"
        self._thread = threading.Thread(target=target, daemon=True, name=name)
        self._thread.start()

    # -- WASAPI loopback source -------------------------------------------
    def _run_loopback(self) -> None:
        """Capture what the computer is playing, block by block."""
        from . import wasapi

        stream = None
        try:
            stream = wasapi.LoopbackStream(
                sample_rate=self.sample_rate, device_id=self.device_id,
                target_rate=self.sample_rate)
            stream.open()
        except Exception as exc:
            self._error = f"cannot listen to the system sound: {exc}"
            self._state = "error"
            if stream is not None:
                stream.close()
            return
        self._ready.set()      # the device is running; audio is now being captured

        total_bytes = 0
        try:
            while not self._stop.is_set():
                block = stream.read(timeout=0.2)
                if block.size == 0:
                    continue
                # The rest of the program works in 16-bit mono, so convert here
                # rather than teaching the waterfall and the decoder about floats.
                clipped = np.clip(block, -1.0, 1.0)
                pcm = (clipped * 32767.0).astype("<i2").tobytes()
                with self._lock:
                    self._chunks.append(pcm)
                total_bytes += len(pcm)
                peak = int(np.max(np.abs(clipped)) * 32768.0) if clipped.size else 0
                self._peak = max(self._peak, min(peak, 32768))
                self._seconds = total_bytes / 2.0 / self.sample_rate
                if self._on_level is not None:
                    try:
                        self._on_level(self._seconds, self._peak / 32768.0)
                    except Exception:
                        pass
        except Exception as exc:
            self._error = f"listening to the system sound failed: {exc}"
            self._state = "error"
        finally:
            if stream is not None:
                stream.close()
        if self._state == "recording":
            self._state = "idle"

    def stop(self, wait: bool = False) -> None:
        """Stop capturing; captured audio stays available via :meth:`samples`."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and wait:
            thread.join(timeout=5.0)
        if self._state == "recording":
            self._state = "idle"

    def wait(self, timeout: float | None = None) -> bool:
        """Block until capture stops."""
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def samples(self) -> list[float]:
        """Return everything captured so far as mono floats."""
        with self._lock:
            raw = b"".join(self._chunks)
        data = array.array("h")
        data.frombytes(raw[: len(raw) // 2 * 2])
        return [v / 32768.0 for v in data]

    def captured_bytes(self) -> int:
        """Total bytes captured so far.

        Handy as the cursor for :meth:`bytes_since`: a live display wants the
        audio that arrived since its last repaint, not the whole recording.
        """
        with self._lock:
            return sum(len(chunk) for chunk in self._chunks)

    def bytes_since(self, offset: int) -> tuple[bytes, int]:
        """Return ``(new_bytes, new_offset)`` for everything after *offset*.

        Only the chunks past the offset are joined, so the cost of a repaint
        depends on how much audio arrived since the last one rather than on how
        long the recording has been running.
        """
        with self._lock:
            chunks = list(self._chunks)
        total = 0
        pieces: list[bytes] = []
        for chunk in chunks:
            end = total + len(chunk)
            if end > offset:
                if total >= offset:
                    pieces.append(chunk)
                else:
                    pieces.append(chunk[offset - total:])
            total = end
        # An even length keeps the 16-bit sample alignment intact.
        raw = b"".join(pieces)
        raw = raw[: len(raw) // 2 * 2]
        return raw, offset + len(raw)

    # -- worker -----------------------------------------------------------
    def _run(self) -> None:
        handle = wintypes.HANDLE()
        fmt = _make_format(self.sample_rate)
        rc = _winmm.waveInOpen(
            ctypes.byref(handle), self.device if self.device >= 0 else WAVE_MAPPER,
            ctypes.byref(fmt), None, None, 0,
        )
        if rc != 0:
            self._error = f"cannot open recording device: {_mm_text(rc)}"
            self._state = "error"
            return

        block_bytes = SAMPLES_PER_BLOCK * 2

        class _Slot:
            __slots__ = ("hdr", "buf", "prepared")

            def __init__(self) -> None:
                self.buf = ctypes.create_string_buffer(block_bytes)
                self.hdr = _WAVEHDR()
                self.hdr.lpData = ctypes.cast(self.buf, ctypes.c_void_p)
                self.hdr.dwBufferLength = block_bytes
                self.hdr.dwFlags = 0
                self.hdr.dwBytesRecorded = 0
                self.prepared = True
                _check(
                    _winmm.waveInPrepareHeader(
                        handle, ctypes.byref(self.hdr), ctypes.sizeof(self.hdr)
                    ),
                    "waveInPrepareHeader",
                )

            def recycle(self) -> None:
                """Re-arm a completed buffer.

                A header whose ``WHDR_DONE`` flag is set must be unprepared
                before it can be queued again -- that call is what clears the
                flag, and a header that is merely re-added silently empties the
                capture queue.
                """
                if self.prepared:
                    _check(
                        _winmm.waveInUnprepareHeader(
                            handle, ctypes.byref(self.hdr), ctypes.sizeof(self.hdr)
                        ),
                        "waveInUnprepareHeader",
                    )
                self.hdr.dwFlags = 0
                self.hdr.dwBytesRecorded = 0
                _check(
                    _winmm.waveInPrepareHeader(
                        handle, ctypes.byref(self.hdr), ctypes.sizeof(self.hdr)
                    ),
                    "waveInPrepareHeader",
                )
                self.prepared = True

            def release(self) -> None:
                if self.prepared:
                    _winmm.waveInUnprepareHeader(
                        handle, ctypes.byref(self.hdr), ctypes.sizeof(self.hdr)
                    )
                    self.prepared = False

        slots: list[_Slot] = []
        try:
            for _ in range(3):
                slot = _Slot()
                _check(
                    _winmm.waveInAddBuffer(
                        handle, ctypes.byref(slot.hdr), ctypes.sizeof(slot.hdr)
                    ),
                    "waveInAddBuffer",
                )
                slots.append(slot)
            _check(_winmm.waveInStart(handle), "waveInStart")
            self._ready.set()   # the device is running; audio is now captured

            total_bytes = 0
            while not self._stop.is_set():
                progressed = False
                for slot in slots:
                    if not (slot.hdr.dwFlags & WHDR_DONE):
                        continue
                    n = slot.hdr.dwBytesRecorded
                    if n:
                        with self._lock:
                            self._chunks.append(slot.buf.raw[:n])
                        total_bytes += n
                        block = array.array("h")
                        block.frombytes(slot.buf.raw[: n // 2 * 2])
                        if block:
                            self._peak = max(self._peak, max(abs(v) for v in block))
                    slot.recycle()
                    _check(
                        _winmm.waveInAddBuffer(
                            handle, ctypes.byref(slot.hdr), ctypes.sizeof(slot.hdr)
                        ),
                        "waveInAddBuffer",
                    )
                    progressed = True
                self._seconds = total_bytes / 2.0 / self.sample_rate
                if self._on_level is not None and progressed:
                    try:
                        self._on_level(self._seconds, self._peak / 32768.0)
                    except Exception:
                        pass
                if not progressed:
                    time.sleep(0.02)
        except AudioError as exc:
            self._error = str(exc)
            self._state = "error"
        finally:
            _winmm.waveInStop(handle)
            _winmm.waveInReset(handle)
            for slot in slots:
                slot.release()
            _winmm.waveInClose(handle)
        if self._state == "recording":
            self._state = "idle"
