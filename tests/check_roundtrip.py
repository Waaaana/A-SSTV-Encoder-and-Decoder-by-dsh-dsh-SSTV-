"""Round-trip verification: encode each mode, decode it, compare with the source.

Reports PSNR and writes both images so a failure can be looked at rather than
guessed at.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sstv_bootstrap
import numpy as np
from PIL import Image
from sstv import modes, encoder, decoder

RATE = 48000
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)


def psnr(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


import math

print("%-12s %-7s %-6s %-6s %-8s %-9s %s" % (
    "mode", "PSNR", "lines", "cov", "syncRMS", "offset", "notes"))
worst = (99.0, None)
for spec in modes.MODE_LIST:
    W, H = spec.resolution
    img = encoder.make_test_image(W, H)
    res = encoder.encode(img, spec, RATE)
    t0 = time.time()
    dec = decoder.decode(res.samples, RATE)
    el = time.time() - t0
    ref = np.asarray(img.resize((W, H), Image.LANCZOS), dtype=np.uint8)
    if dec.mode is None:
        print("%-12s NO MODE DECODED" % spec.name)
        continue
    p = psnr(ref, dec.image)
    if p < worst[0]:
        worst = (p, spec.name)
    Image.fromarray(dec.image).save(os.path.join(OUT, "rt-%s.png" % spec.name.replace(" ", "")))
    print("%-12s %-7.2f %-6d %-6.2f %-8.1f %-9.1f %s (%.1fs)" % (
        spec.name, p, dec.lines_decoded, dec.coverage, dec.sync_error_hz,
        dec.frequency_offset_hz, "; ".join(dec.notes) or "-", el))

print()
print("worst mode:", worst)
