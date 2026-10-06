"""Capture whatever the computer is playing, via WASAPI loopback.

This is what makes "listen to the system sound" work.  A radio connected to the
line input, a receiver feeding the speakers, a WebSDR or a recording playing in a
browser -- all of them are audible on the sound card's output, and this module
taps that output, so SSTV can be decoded without any cable between two programs.

Why loopback rather than "Stereo Mix"
-------------------------------------
Windows has long offered a *Stereo Mix* input, but it is hidden and disabled by
default on most modern drivers, so a program that relies on it often has no
device to open at all.  WASAPI loopback asks the audio engine for a copy of what
is being rendered to a chosen output device, which works on any machine with
working sound and needs nothing enabled by hand.

The COM plumbing is done with :mod:`ctypes`, in keeping with the rest of this
program: no PyAudio, no sounddevice, nothing to install.

A note on the interface pointers below: each COM object is addressed through its
vtable, which is an array of function pointers.  The indices used here come from
the order of the methods in the Windows headers (``mmdeviceapi.h``,
``audioclient.h``) starting at 3, because the first three entries are
``QueryInterface``, ``AddRef`` and ``Release``.  Getting one of these numbers
wrong does not fail cleanly -- it calls an unrelated function -- so they are
called out by name at every use.
"""

from __future__ import annotations

import ctypes
import sys
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass

import numpy as np

__all__ = [
    "LoopbackDevice",
    "LoopbackError",
    "LoopbackStream",
    "list_loopback_devices",
    "loopback_available",
]


class LoopbackError(RuntimeError):
    """Raised when loopback capture cannot be set up."""


# --------------------------------------------------------------------------
# COM constants
# --------------------------------------------------------------------------

_HRESULT = ctypes.c_long
_CLSCTX_ALL = 0x17
_COINIT_MULTITHREADED = 0x0
_RPC_E_CHANGED_MODE = -2147417850

AUDCLNT_SHAREMODE_SHARED = 0
AUDCLNT_STREAMFLAGS_LOOPBACK = 0x00020000
AUDCLNT_BUFFERFLAGS_SILENT = 0x2

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE

_SUBTYPE_IEEE_FLOAT = "{00000003-0000-0010-8000-00aa00389b71}"
_SUBTYPE_PCM = "{00000001-0000-0010-8000-00aa00389b71}"

# 200 ms of buffer: long enough that a busy moment in the interface cannot lose
# audio, short enough that the level meter and waterfall still feel live.
BUFFER_HNS = 2_000_000

_CLSID_MMDeviceEnumerator = "{BCDE0395-E52F-467C-8E3D-C4579291692E}"
_IID_IMMDeviceEnumerator = "{A95664D2-9614-4F35-A746-DE8DB63617E6}"
_IID_IAudioClient = "{1CB9AD4C-DBFA-4c32-B178-C2F568A703B2}"
_IID_IAudioCaptureClient = "{C8ADBD64-E71E-48a0-A4DE-185C395CD317}"

# {A45C254E-DF1C-4EFD-8020-67D146A850E0}, 14 == PKEY_Device_FriendlyName
_PKEY_FRIENDLY_NAME = "{A45C254E-DF1C-4EFD-8020-67D146A850E0}"
VT_LPWSTR = 31

# IMMDeviceEnumerator vtable indices
_ENUM_AUDIO_ENDPOINTS = 3
_GET_DEFAULT_AUDIO_ENDPOINT = 4
_GET_DEVICE = 5

# IMMDevice vtable indices
_DEVICE_ACTIVATE = 3
_DEVICE_OPEN_PROPERTY_STORE = 4
_DEVICE_GET_ID = 5

# IPropertyStore vtable indices
_STORE_GET_AT = 4
_STORE_GET_VALUE = 5

# IAudioClient vtable indices
_CLIENT_INITIALIZE = 3
_CLIENT_GET_MIX_FORMAT = 8
_CLIENT_START = 10
_CLIENT_STOP = 11
_CLIENT_GET_SERVICE = 14

# IAudioCaptureClient vtable indices
_CAPTURE_GET_BUFFER = 3
_CAPTURE_RELEASE_BUFFER = 4
_CAPTURE_NEXT_PACKET_SIZE = 5

# IEnumAudioEndpoints vtable indices.  IUnknown takes three entries, then the
# interface declares Clone, Next, Reset and Skip *in that order* -- so Next is 4,
# not 3.  Using 3 calls Clone, which takes a different parameter list entirely.
_ENUM_ENDPOINTS_NEXT = 4

# eRender, eConsole
_EDATAFLOW_RENDER = 0
_EROLE_CONSOLE = 0


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    def __init__(self, text: str | None = None) -> None:
        super().__init__()
        if text is None:
            return
        cleaned = text.strip().strip("{}")
        parts = cleaned.split("-")
        if len(parts) != 5:
            raise ValueError(f"not a GUID: {text!r}")
        self.Data1 = int(parts[0], 16)
        self.Data2 = int(parts[1], 16)
        self.Data3 = int(parts[2], 16)
        tail = parts[3] + parts[4]
        for index in range(8):
            self.Data4[index] = int(tail[index * 2 : index * 2 + 2], 16)

    def __str__(self) -> str:  # pragma: no cover - display helper
        tail = "".join(f"{b:02x}" for b in self.Data4)
        return (f"{{{self.Data1:08x}-{self.Data2:04x}-{self.Data3:04x}-"
                f"{tail[:4]}-{tail[4:]}}}")


class WAVEFORMATEX(ctypes.Structure):
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


class WAVEFORMATEXTENSIBLE(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("Format", WAVEFORMATEX),
        ("wValidBitsPerSample", wintypes.WORD),
        ("dwChannelMask", wintypes.DWORD),
        ("SubFormat", GUID),
    ]


class PROPERTYKEY(ctypes.Structure):
    _fields_ = [("fmtid", GUID), ("pid", wintypes.DWORD)]


def _check(result: int, what: str) -> int:
    if result < 0:
        raise LoopbackError(f"{what} failed (HRESULT 0x{result & 0xFFFFFFFF:08X})")
    return result


def _method(pointer, index: int, restype, *argtypes):
    """Bind one vtable entry of the COM object at *pointer*."""
    vtable = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    prototype = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    return prototype(vtable[index])


def _release(pointer) -> None:
    """Call IUnknown::Release (vtable index 2) and swallow any failure."""
    if not pointer:
        return
    try:
        _method(pointer, 2, _HRESULT)(pointer)
    except Exception:
        pass


_OLE32 = None


def _ole32():
    """The ole32 entry points used here, loaded once.

    Loaded by explicit file name.  Going through ``ctypes.OleDLL("ole32")``
    relies on library-name munging, and on this Python build the attribute
    lookup on the module object is what actually fails.
    """
    global _OLE32
    if _OLE32 is None:
        library = ctypes.WinDLL("ole32.dll")
        library.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
        library.CoInitializeEx.restype = _HRESULT
        library.CoCreateInstance.argtypes = [
            ctypes.POINTER(GUID), ctypes.c_void_p, wintypes.DWORD,
            ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p),
        ]
        library.CoCreateInstance.restype = _HRESULT
        library.CoTaskMemFree.argtypes = [ctypes.c_void_p]
        library.PropVariantClear.argtypes = [ctypes.c_void_p]
        library.PropVariantClear.restype = _HRESULT
        _OLE32 = library
    return _OLE32


def _co_initialize() -> None:
    result = _ole32().CoInitializeEx(None, _COINIT_MULTITHREADED)
    # S_FALSE means "already initialised on this thread", which is fine.
    if result < 0 and result != _RPC_E_CHANGED_MODE:
        raise LoopbackError(f"CoInitializeEx failed (0x{result & 0xFFFFFFFF:08X})")


def loopback_available() -> bool:
    """True when this build can capture the system output at all."""
    return sys.platform == "win32"


def _create_enumerator():
    enumerator = ctypes.c_void_p()
    _check(
        _ole32().CoCreateInstance(
            ctypes.byref(GUID(_CLSID_MMDeviceEnumerator)), None, _CLSCTX_ALL,
            ctypes.byref(GUID(_IID_IMMDeviceEnumerator)), ctypes.byref(enumerator),
        ),
        "CoCreateInstance(MMDeviceEnumerator)",
    )
    return enumerator


# --------------------------------------------------------------------------
# Device enumeration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LoopbackDevice:
    """An output device whose sound can be captured."""

    index: int
    name: str
    device_id: str
    is_default: bool = False

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.name


def _device_id(device) -> str:
    string = ctypes.c_wchar_p()
    result = _method(device, _DEVICE_GET_ID, _HRESULT,
                     ctypes.POINTER(ctypes.c_wchar_p))(device, ctypes.byref(string))
    if result < 0 or not string.value:
        return ""
    value = string.value
    _ole32().CoTaskMemFree(string)
    return value


def _device_name(device) -> str:
    store = ctypes.c_void_p()
    result = _method(device, _DEVICE_OPEN_PROPERTY_STORE, _HRESULT, wintypes.DWORD,
                     ctypes.POINTER(ctypes.c_void_p))(device, 0, ctypes.byref(store))
    if result < 0 or not store:
        return ""
    try:
        key = PROPERTYKEY(GUID(_PKEY_FRIENDLY_NAME), 14)
        # PROPVARIANT for a string: 8-byte header (vt at offset 0) then pointer.
        variant = ctypes.create_string_buffer(64)
        result = _method(store, _STORE_GET_VALUE, _HRESULT,
                         ctypes.POINTER(PROPERTYKEY), ctypes.c_void_p)(
            store, ctypes.byref(key), ctypes.byref(variant))
        if result < 0:
            return ""
        vt = ctypes.cast(variant, ctypes.POINTER(wintypes.USHORT))[0]
        if vt != VT_LPWSTR:
            return ""
        pointer = ctypes.cast(ctypes.byref(variant, 8),
                              ctypes.POINTER(ctypes.c_wchar_p))[0]
        name = pointer or ""
        _ole32().PropVariantClear(ctypes.byref(variant))
        return name
    except Exception:
        return ""
    finally:
        _release(store)


def list_loopback_devices() -> list[LoopbackDevice]:
    """Every output device that can be listened to, default first.

    Built from the three *roles* rather than by walking the endpoint collection.
    ``IEnumAudioEndpoints`` needs the vtable slot for ``Next`` to be right, and
    getting that wrong either crashes or silently reports an empty collection --
    whereas ``GetDefaultAudioEndpoint`` accepts a role and reliably returns the
    device Windows would use for console, multimedia or communications audio.
    Those are exactly the endpoints a person would pick from, and asking for all
    three gives the same practical list without depending on the enumerator.

    An empty list means there is no usable output device, and the caller should
    offer the microphone instead.
    """
    if not loopback_available():
        return []

    # (dataFlow, role, label) -- all render roles, so HDMI and a second sound
    # card show up as well when Windows has been told to use them.
    candidates = (
        (_EDATAFLOW_RENDER, 0, "Console"),
        (_EDATAFLOW_RENDER, 1, "Multimedia"),
        (_EDATAFLOW_RENDER, 2, "Communications"),
    )

    enumerator = None
    seen: dict[str, str] = {}
    try:
        _co_initialize()
        enumerator = _create_enumerator()
        for flow, role, label in candidates:
            device = ctypes.c_void_p()
            result = _method(enumerator, _GET_DEFAULT_AUDIO_ENDPOINT, _HRESULT,
                             ctypes.c_int, ctypes.c_int,
                             ctypes.POINTER(ctypes.c_void_p))(
                enumerator, flow, role, ctypes.byref(device))
            if result < 0 or not device:
                continue
            try:
                ident = _device_id(device)
                if not ident or ident in seen:
                    continue
                name = _device_name(device) or f"Output device {len(seen) + 1}"
                seen[ident] = name
            finally:
                _release(device)
    except Exception:
        return []
    finally:
        _release(enumerator)

    return [
        LoopbackDevice(index, f"{name} (system sound)", ident, index == 0)
        for index, (ident, name) in enumerate(seen.items())
    ]


# --------------------------------------------------------------------------
# Capture stream
# --------------------------------------------------------------------------

class LoopbackStream:
    """A reader of the sound being played to one output device.

    :meth:`read` returns float samples in ``[-1, 1]``, mixed down to mono and
    resampled to the rate the caller asked for, so nothing else in the program has
    to care what the sound card's mix format happens to be.
    """

    def __init__(self, sample_rate: int = 48000, device_id: str | None = None,
                 target_rate: int | None = None) -> None:
        if not loopback_available():
            raise LoopbackError("capturing the system sound requires Windows")
        self.sample_rate = int(sample_rate)
        self.target_rate = int(target_rate or sample_rate)
        self.device_id = device_id
        self._closed = False
        self._client = ctypes.c_void_p()
        self._capture = ctypes.c_void_p()
        self._format_pointer = None
        self._mix_rate = self.target_rate
        self._channels = 1
        self._is_float = True
        self._bits = 32
        self._device_name = ""
        self._resample_pos = 0.0
        self._opened = False

    # -- properties -------------------------------------------------------
    @property
    def mix_rate(self) -> int:
        """The sound card's own rate, which may differ from the target."""
        return self._mix_rate

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def is_open(self) -> bool:
        return self._opened

    # -- lifecycle --------------------------------------------------------
    def open(self) -> None:
        """Set up the capture stream.  Raises :class:`LoopbackError` on failure."""
        if self._opened:
            return
        _co_initialize()
        enumerator = None
        device = None
        try:
            enumerator = _create_enumerator()
            device = ctypes.c_void_p()
            if self.device_id:
                _check(
                    _method(enumerator, _GET_DEVICE, _HRESULT, ctypes.c_wchar_p,
                            ctypes.POINTER(ctypes.c_void_p))(
                        enumerator, self.device_id, ctypes.byref(device)),
                    "IMMDeviceEnumerator.GetDevice",
                )
            else:
                _check(
                    _method(enumerator, _GET_DEFAULT_AUDIO_ENDPOINT, _HRESULT,
                            ctypes.c_int, ctypes.c_int,
                            ctypes.POINTER(ctypes.c_void_p))(
                        enumerator, _EDATAFLOW_RENDER, _EROLE_CONSOLE,
                        ctypes.byref(device)),
                    "IMMDeviceEnumerator.GetDefaultAudioEndpoint",
                )
            self._device_name = _device_name(device) or "System output"

            _check(
                _method(device, _DEVICE_ACTIVATE, _HRESULT, ctypes.POINTER(GUID),
                        wintypes.DWORD, ctypes.c_void_p,
                        ctypes.POINTER(ctypes.c_void_p))(
                    device, ctypes.byref(GUID(_IID_IAudioClient)), _CLSCTX_ALL,
                    None, ctypes.byref(self._client)),
                "IMMDevice.Activate(IAudioClient)",
            )
        finally:
            _release(device)
            _release(enumerator)

        # GetMixFormat hands back memory owned by the client; keep the pointer.
        format_pointer = ctypes.POINTER(WAVEFORMATEX)()
        _check(
            _method(self._client, _CLIENT_GET_MIX_FORMAT, _HRESULT,
                    ctypes.POINTER(ctypes.POINTER(WAVEFORMATEX)))(
                self._client, ctypes.byref(format_pointer)),
            "IAudioClient.GetMixFormat",
        )
        self._format_pointer = format_pointer
        self._describe_format(format_pointer)

        _check(
            _method(self._client, _CLIENT_INITIALIZE, _HRESULT, ctypes.c_int,
                    wintypes.DWORD, ctypes.c_longlong, ctypes.c_longlong,
                    ctypes.POINTER(WAVEFORMATEX), ctypes.c_void_p)(
                self._client, AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_LOOPBACK, BUFFER_HNS, 0,
                format_pointer, None),
            "IAudioClient.Initialize(loopback)",
        )

        _check(
            _method(self._client, _CLIENT_GET_SERVICE, _HRESULT,
                    ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p))(
                self._client, ctypes.byref(GUID(_IID_IAudioCaptureClient)),
                ctypes.byref(self._capture)),
            "IAudioClient.GetService(IAudioCaptureClient)",
        )

        _check(_method(self._client, _CLIENT_START, _HRESULT)(self._client),
               "IAudioClient.Start")
        self._opened = True

    def _describe_format(self, format_pointer) -> None:
        fmt = format_pointer.contents
        self._channels = max(1, int(fmt.nChannels))
        self._mix_rate = int(fmt.nSamplesPerSec)
        self._bits = int(fmt.wBitsPerSample) or 32
        tag = int(fmt.wFormatTag)
        if tag == WAVE_FORMAT_EXTENSIBLE:
            extended = ctypes.cast(
                format_pointer, ctypes.POINTER(WAVEFORMATEXTENSIBLE)).contents
            self._is_float = str(extended.SubFormat) == _SUBTYPE_IEEE_FLOAT
            self._bits = int(extended.wValidBitsPerSample) or self._bits
        else:
            self._is_float = tag == WAVE_FORMAT_IEEE_FLOAT
        if self._is_float and self._bits != 32:
            raise LoopbackError(f"unsupported mix format: {self._bits}-bit float")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._opened and self._client:
                _method(self._client, _CLIENT_STOP, _HRESULT)(self._client)
        except Exception:
            pass
        _release(self._capture)
        _release(self._client)
        self._capture = ctypes.c_void_p()
        self._client = ctypes.c_void_p()
        self._opened = False

    def __enter__(self) -> "LoopbackStream":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- reading ----------------------------------------------------------
    def read(self, timeout: float = 0.4) -> np.ndarray:
        """Collect the audio played since the last call, as mono floats.

        An empty array is a normal result: loopback produces no packets at all
        while the output device is silent, so the caller should treat "nothing"
        as silence rather than as an error.
        """
        if not self._opened:
            raise LoopbackError("the capture stream is not open")

        deadline = time.monotonic() + timeout
        pieces: list[np.ndarray] = []
        while True:
            packet = self._next_packet()
            if packet is not None:
                if packet.size:
                    pieces.append(packet)
                # Drain everything queued, then return: waiting for more would
                # add latency to the level meter and the waterfall.
                continue
            if pieces or time.monotonic() >= deadline:
                break
            time.sleep(0.002)

        if not pieces:
            return np.zeros(0, dtype=np.float64)
        return self._convert(np.concatenate(pieces))

    def _next_packet(self) -> np.ndarray | None:
        """One packet of raw samples, or ``None`` when the queue is empty."""
        size = wintypes.UINT()
        result = _method(self._capture, _CAPTURE_NEXT_PACKET_SIZE, _HRESULT,
                         ctypes.POINTER(wintypes.UINT))(
            self._capture, ctypes.byref(size))
        if result < 0 or not size.value:
            return None

        data = ctypes.POINTER(ctypes.c_ubyte)()
        frames = wintypes.UINT()
        flags = wintypes.DWORD()
        result = _method(
            self._capture, _CAPTURE_GET_BUFFER, _HRESULT,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
            ctypes.POINTER(wintypes.UINT), ctypes.POINTER(wintypes.DWORD),
            ctypes.c_void_p, ctypes.c_void_p,
        )(self._capture, ctypes.byref(data), ctypes.byref(frames),
          ctypes.byref(flags), None, None)
        if result < 0:
            return None

        try:
            if flags.value & AUDCLNT_BUFFERFLAGS_SILENT:
                # The engine says "silence here"; the buffer contents are
                # undefined, so do not read them.
                count = int(frames.value) * self._channels
                return np.zeros(max(count, 0), dtype=np.float32)
            count = int(frames.value) * self._channels * (self._bits // 8)
            if not data or count <= 0:
                return np.zeros(0, dtype=np.float32)
            block = ctypes.string_at(data, count)
            if self._is_float:
                return np.frombuffer(block, dtype=np.float32).copy()
            if self._bits == 16:
                raw = np.frombuffer(block, dtype=np.int16)
                return (raw.astype(np.float32) / 32768.0)
            raw = np.frombuffer(block, dtype=np.int32)
            return (raw.astype(np.float32) / 2147483648.0)
        finally:
            _method(self._capture, _CAPTURE_RELEASE_BUFFER, _HRESULT, wintypes.UINT)(
                self._capture, frames.value)

    def _convert(self, samples: np.ndarray) -> np.ndarray:
        """Mix to mono, then resample to the caller's rate."""
        channels = self._channels
        if channels > 1:
            usable = (samples.size // channels) * channels
            if usable:
                samples = samples[:usable].reshape(-1, channels).mean(axis=1)
        if self._mix_rate != self.target_rate and samples.size:
            samples = self._resample(samples)
        return samples.astype(np.float64)

    def _resample(self, samples: np.ndarray) -> np.ndarray:
        """Linear resampling that carries its phase across calls.

        Restarting the interpolation for every packet would put a step at each
        boundary, and the decoder would pick that up as periodic noise on the
        picture -- a comb of artefacts spaced by the packet interval.
        """
        ratio = self._mix_rate / float(self.target_rate)
        positions = np.arange(samples.size, dtype=np.float64) + self._resample_pos
        last_out = np.floor((samples.size - 1 - self._resample_pos) / ratio)
        out_count = int(last_out) + 1
        if out_count <= 0:
            self._resample_pos -= samples.size
            return np.zeros(0, dtype=np.float32)
        source = np.arange(samples.size, dtype=np.float64)
        wanted = np.arange(out_count, dtype=np.float64) * ratio + self._resample_pos
        wanted = np.clip(wanted, 0.0, samples.size - 1.0)
        result = np.interp(wanted, source, samples)
        # Consumed input samples, keeping the fractional remainder for next time.
        self._resample_pos = wanted[-1] + ratio - samples.size
        return result.astype(np.float32)
