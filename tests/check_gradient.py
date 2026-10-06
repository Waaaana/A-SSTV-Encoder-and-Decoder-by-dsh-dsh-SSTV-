"""Round-trip check using a linear-gradient test image.

A test card hides errors: a bar that is one pixel out still looks like a bar.  A
gradient makes every row and column unique, so a vertical shift shows up as a
uniform red error, a horizontal one as a green error, and a colour-space mistake
as blue error.  Channel errors are reported separately for exactly that reason.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sstv_bootstrap
import numpy as np
from PIL import Image
from sstv import modes, encoder, decoder

RATE = 48000
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)


def gradient_image(w, h):
    """Red ramps down the rows, green ramps across the columns, blue is constant."""
    rows = np.linspace(0, 255, h)[:, None] * np.ones((1, w))
    cols = np.ones((h, 1)) * np.linspace(0, 255, w)[None, :]
    blue = np.full((h, w), 128.0)
    return np.stack((rows, cols, blue), axis=-1).astype(np.uint8)


def channel_errors(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    return [float(np.sqrt(np.mean(d[..., i] ** 2))) for i in range(3)]


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


print("%-12s %-7s %-22s %s" % ("mode", "PSNR", "RMSE R / G / B", "notes"))
for spec in modes.MODE_LIST:
    W, H = spec.resolution
    img = Image.fromarray(gradient_image(W, H), "RGB")
    res = encoder.encode(img, spec, RATE)
    dec = decoder.decode(res.samples, RATE)
    if dec.mode is None:
        print("%-12s NO MODE" % spec.name)
        continue
    ref = np.asarray(img)
    errs = channel_errors(ref, dec.image)
    Image.fromarray(dec.image).save(os.path.join(OUT, "grad-%s.png" % spec.name.replace(" ", "")))
    print("%-12s %-7.2f R=%6.1f G=%6.1f B=%6.1f  %s" % (
        spec.name, psnr(ref, dec.image), errs[0], errs[1], errs[2],
        "; ".join(dec.notes) or "-"))
