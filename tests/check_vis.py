import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sstv_bootstrap
import numpy as np
from sstv import modes, encoder, dsp, vis

RATE = 48000

import inspect
print("sstv.vis loaded from:", vis.__file__)
print("default search window:", vis._START_SEARCH_S)
print("find_vis_start signature:", inspect.signature(vis.find_vis_start))
print("_runs uses int16:", "int16" in inspect.getsource(vis._runs))
print()

print("=== VIS header round-trip, every mode ===")
fails = 0
for spec in modes.MODE_LIST:
    img = encoder.make_test_image(*spec.resolution)
    res = encoder.encode(img, spec, RATE)
    hdr = vis.decode_vis(res.samples, RATE)
    want_par = modes.vis_parity(spec.vis_code)
    ok = (hdr is not None and hdr.code == spec.vis_code
          and hdr.parity == want_par and hdr.stop_ok)
    if not ok:
        fails += 1
    print("%-12s code=%-3d want=%-3d  par=%d/%d  stop_ok=%-5s conf=%.2f  start=%.3fs %s" % (
        spec.name, hdr.code if hdr else -1, spec.vis_code,
        hdr.parity if hdr else -1, want_par,
        hdr.stop_ok if hdr else "-", hdr.confidence if hdr else 0.0,
        hdr.start if hdr else 0.0, "OK" if ok else "FAIL"))

print()
print("VIS failures:", fails)

print()
print("=== VIS survives a trimmed file and added noise ===")
spec = modes.get_mode("Robot 36")
res = encoder.encode(encoder.make_test_image(320, 240), spec, RATE)
rng = np.random.default_rng(3)
for label, wave in [
    ("clean", res.samples),
    ("10% noise", res.samples + 0.10 * rng.standard_normal(res.samples.size)),
    ("25% noise", res.samples + 0.25 * rng.standard_normal(res.samples.size)),
    ("lowercase 0.3", res.samples * 0.3),
    ("+40 Hz offset", None),
]:
    if wave is None:
        tr = dsp.instantaneous_frequency(res.samples, RATE) + 40.0
        hdr = vis.decode_vis(res.samples, RATE, track=tr)
    else:
        hdr = vis.decode_vis(wave, RATE)
    print("  %-16s -> code=%s conf=%s" % (
        label, hdr.code if hdr else None, "%.2f" % hdr.confidence if hdr else "-"))

# Trimming: header no longer at the file start
trimmed = res.samples[int(0.5 * RATE):]
hdr = vis.decode_vis(trimmed, RATE)
print("  %-16s -> code=%s (want %d)" % ("trimmed 0.5s", hdr.code if hdr else None, spec.vis_code))
