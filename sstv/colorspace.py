"""Colour-space helpers shared by the encoder and decoder.

SSTV modes come in two flavours.  Martin, Scottie and Wraase send the three
primaries directly; Robot and PD send a luminance signal plus two colour
differences.  Both representations live in the same 0-255 numeric range, and the
difference signals are centred on 128 so that a neutral grey has a well-defined
tone instead of an arbitrary one.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "rgb_to_ycrcb",
    "ycrcb_to_rgb",
    "downsample",
    "upsample",
]

# ITU-R BT.601 luma coefficients, scaled so that RGB 0-255 maps to Y 0-255 and
# the chroma differences to roughly -128..+127 around a centre of 128.  These are
# the same coefficients every SSTV implementation uses, which matters: a
# different matrix would still round-trip through our own decoder but would tint
# every image exchanged with other software.
_KR = 0.299
_KB = 0.114
_KG = 1.0 - _KR - _KB


def rgb_to_ycrcb(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split ``(..., 3)`` RGB data into ``(Y, Cr, Cb)`` float arrays.

    All three outputs land in the 0-255 range that SSTV transmits, so a neutral
    grey is ``(Y, 128, 128)``.  With 0-255 inputs the colour-difference
    coefficients are the usual 0.713 and 0.564, i.e. the difference is divided by
    ``2 * (1 - K)`` and *not* scaled by 255 again -- that extra factor is already
    folded into the coefficients and applying it twice throws the chroma
    thousands of levels out of range.
    """
    arr = np.asarray(rgb, dtype=np.float64)
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    y = _KR * r + _KG * g + _KB * b
    # Pure primaries land a fraction above white and a fraction below black, so
    # clamp into the transmittable range rather than letting a colour wrap.
    cr = np.clip((r - y) / (2.0 * (1.0 - _KR)) + 128.0, 0.0, 255.0)
    cb = np.clip((b - y) / (2.0 * (1.0 - _KB)) + 128.0, 0.0, 255.0)
    return np.clip(y, 0.0, 255.0), cr, cb


def ycrcb_to_rgb(y: np.ndarray, cr: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rgb_to_ycrcb`; returns ``(..., 3)`` RGB clipped 0-255."""
    yy = np.asarray(y, dtype=np.float64)
    r = yy + (np.asarray(cr, dtype=np.float64) - 128.0) * (2.0 * (1.0 - _KR))
    b = yy + (np.asarray(cb, dtype=np.float64) - 128.0) * (2.0 * (1.0 - _KB))
    g = (yy - _KR * r - _KB * b) / _KG
    return np.clip(np.stack((r, g, b), axis=-1), 0.0, 255.0)


def downsample(data: np.ndarray, factor: int) -> np.ndarray:
    """Average *factor* neighbouring columns together (horizontal subsampling)."""
    arr = np.asarray(data, dtype=np.float64)
    if factor <= 1:
        return arr
    width = arr.shape[-1]
    usable = (width // factor) * factor
    trimmed = arr[..., :usable]
    shape = trimmed.shape[:-1] + (usable // factor, factor)
    return trimmed.reshape(shape).mean(axis=-1)


def upsample(data: np.ndarray, factor: int, width: int | None = None) -> np.ndarray:
    """Repeat every column *factor* times, truncating to *width* columns."""
    arr = np.asarray(data, dtype=np.float64)
    if factor <= 1:
        return arr
    out = np.repeat(arr, factor, axis=-1)
    if width is not None:
        out = out[..., :width]
        if out.shape[-1] < width:
            pad = width - out.shape[-1]
            out = np.concatenate((out, np.repeat(out[..., -1:], pad, axis=-1)), axis=-1)
    return out
