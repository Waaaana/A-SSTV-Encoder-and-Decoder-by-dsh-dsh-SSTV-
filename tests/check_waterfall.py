"""Check that the waterfall shows what it is supposed to show."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sstv_bootstrap
import numpy as np
from sstv import modes, encoder, dsp, waterfall as wf

RATE = 48000
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)

print("=== 1. a known tone lands in the right column ===")
for freq in (1200.0, 1500.0, 1900.0, 2300.0):
    tone = np.sin(2 * np.pi * freq * np.arange(RATE) / RATE) * 0.6
    w = wf.Waterfall(RATE)
    w.feed(tone)
    peak = w.peak_hz()
    level = w.band_level_db()
    error = abs(peak - freq)
    print("   %6.0f Hz -> peak %7.1f Hz (error %5.1f)  level %6.1f dBFS  frames %d  %s" % (
        freq, peak, error, level, w.frames, "OK" if error < 40 else "FAIL"))

print()
print("=== 2. silence reads as empty, not as a signal ===")
w = wf.Waterfall(RATE)
w.feed(np.zeros(RATE, dtype=np.float64))
print("   silence: peak %.0f Hz, band level %.1f dBFS" % (w.peak_hz(), w.band_level_db()))

print()
print("=== 3. the incremental feed equals a single pass ===")
spec = modes.get_mode("Robot 36")
signal = encoder.encode(encoder.make_test_image(*spec.resolution), spec, RATE).samples
whole = wf.Waterfall(RATE)
whole.feed(signal)
piecewise = wf.Waterfall(RATE)
step = 7777          # deliberately not a multiple of the hop
for start in range(0, signal.size, step):
    piecewise.feed(signal[start:start + step])
print("   frames: whole=%d piecewise=%d  %s" % (
    whole.frames, piecewise.frames, "OK" if whole.frames == piecewise.frames else "FAIL"))
a = whole._view(); b = piecewise._view()
same = a.shape == b.shape and np.allclose(a, b, atol=1e-4)
print("   spectra identical: %s" % ("OK" if same else "FAIL"))

print()
print("=== 4. cost of a live update (must be fast enough for ~20 fps) ===")
w = wf.Waterfall(RATE)
chunk = np.zeros(int(RATE * 0.05), dtype=np.float64)      # 50 ms of audio
warm = signal[: int(RATE * 0.5)]
w.feed(warm)
t0 = time.time()
N = 40
for i in range(N):
    w.feed(signal[int(RATE * (0.5 + 0.05 * i)) : int(RATE * (0.5 + 0.05 * (i + 1)))])
elapsed = (time.time() - t0) / N
print("   feed(50 ms of audio) : %6.2f ms   -> %.0f updates/s" % (elapsed * 1000, 1.0 / elapsed))
t0 = time.time()
img = w.to_image(width=760, height=230)
print("   to_image(760x230)    : %6.2f ms   -> %s" % ((time.time() - t0) * 1000, img.size))

print()
print("=== 5. does the display actually show the SSTV band? ===")
w = wf.Waterfall(RATE, seconds=4.0)
w.feed(signal)
img = w.to_image(width=640, height=200)
path = os.path.join(OUT, "瀑布图-自检.png")
img.save(path)
arr = np.asarray(img, dtype=np.float64)
brightness = arr.mean(axis=2)
print("   image %s  mean brightness %.1f" % (img.size, brightness.mean()))
# The signal occupies roughly 1200-2300 Hz of the 1000-2500 Hz span.
hz = np.linspace(w.low_hz, w.high_hz, img.size[0])
inside = (hz >= 1200) & (hz <= 2300)
outside = ~inside
print("   mean brightness inside the SSTV band  : %.1f" % brightness[:, inside].mean())
print("   mean brightness outside the SSTV band : %.1f" % brightness[:, outside].mean())
contrast = brightness[:, inside].mean() - brightness[:, outside].mean()
print("   contrast: %.1f  %s" % (contrast, "OK" if contrast > 15 else "FAIL"))
print("   wrote", path)

print()
print("=== 6. a short VIS header is visible as a distinct pattern ===")
short = encoder.encode(encoder.make_test_image(320, 240), spec, RATE,
                       include_header=True).samples[: int(RATE * 1.2)]
w = wf.Waterfall(RATE, seconds=2.0)
w.feed(short)
img = w.to_image(width=640, height=180)
print("   header image mean brightness %.1f (silence would be near the floor)" % (
    np.asarray(img, dtype=np.float64).mean()))
