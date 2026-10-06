"""Does streaming decode produce the same picture as decoding after the fact?"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sstv_bootstrap
import numpy as np
from PIL import Image
from sstv import decoder, encoder, modes, streaming

RATE = 48000
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)
failures = []


def check(ok, what, detail=""):
    print("  [%s] %-50s %s" % ("PASS" if ok else "FAIL", what, detail))
    if not ok:
        failures.append(what)


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


print("1. streaming and offline decode agree")
for name in ("Robot 36", "PD 120", "Scottie S1"):
    spec = modes.get_mode(name)
    original = encoder.make_test_image(*spec.resolution)
    signal = encoder.encode(original, spec, RATE).samples

    offline = decoder.decode(signal, RATE)

    live = streaming.StreamingDecoder(RATE)
    block = int(RATE * 0.25)
    for start in range(0, signal.size, block):
        live.feed(signal[start:start + block])
    live.finalize()          # the tail arrives only once the audio has stopped
    online = live.result()

    same_mode = (online.mode is not None and offline.mode is not None
                 and online.mode.name == offline.mode.name)
    check(same_mode, f"{name}: same mode identified",
          f"{online.mode.name if online.mode else None}")
    check(online.lines_decoded == offline.lines_decoded,
          f"{name}: same number of lines",
          f"{online.lines_decoded} vs {offline.lines_decoded}")

    # The meaningful comparison is against the original picture: two decoders can
    # differ slightly from each other while both being equally faithful, and a
    # test that only compares them would flag that as a fault.
    reference = np.asarray(original.resize(spec.resolution), dtype=np.uint8)
    if online.image.size and offline.image.size:
        online_score = psnr(reference, online.image)
        offline_score = psnr(reference, offline.image)
        check(abs(online_score - offline_score) < 2.0,
              f"{name}: streaming is as faithful as offline decoding",
              f"streaming {online_score:.1f} dB vs offline {offline_score:.1f} dB")
        agree = psnr(offline.image, online.image)
        check(agree > 28.0, f"{name}: the two agree pixel for pixel",
              f"{agree:.1f} dB between them")

print()
print("2. the picture really does appear before the transmission ends")
spec = modes.get_mode("Scottie S1")
original = encoder.make_test_image(*spec.resolution)
signal = encoder.encode(original, spec, RATE).samples
stages = []
live = streaming.StreamingDecoder(
    RATE, progress=lambda p: stages.append((round(p.seconds, 2), p.lines_decoded)),
    on_image=lambda img: None)
block = int(RATE * 0.25)
snapshots = []
for start in range(0, signal.size, block):
    live.feed(signal[start:start + block])
    if live.lines_decoded and len(snapshots) < 4 and live.lines_decoded >= (len(snapshots) + 1) * (spec.line_count // 5):
        snapshots.append((round(live.seconds, 1), live.lines_decoded, live.to_image().copy()))
check(len(stages) > 10, "progress is reported during the transmission",
      f"{len(stages)} updates")
check(len(snapshots) >= 3, "partial pictures were produced before the end",
      "; ".join(f"t={s[0]}s {s[1]} lines" for s in snapshots))
if snapshots:
    check(snapshots[-1][1] < spec.line_count,
          "the last snapshot is genuinely incomplete",
          f"{snapshots[-1][1]}/{spec.line_count} lines")
    strip = np.concatenate([np.asarray(s[2]) for s in snapshots], axis=1)
    path = os.path.join(OUT, "实时解码过程.png")
    Image.fromarray(strip).save(path)
    print(f"      saved the progression to {path}")

print()
print("3. cost per update stays low")
spec = modes.get_mode("PD 180")
signal = encoder.encode(encoder.make_test_image(*spec.resolution), spec, RATE).samples
live = streaming.StreamingDecoder(RATE)
block = int(RATE * 0.25)
times = []
for start in range(0, min(signal.size, RATE * 60), block):
    t0 = time.perf_counter()
    live.feed(signal[start:start + block])
    times.append((time.perf_counter() - t0) * 1000)
times = np.asarray(times)
check(times.mean() < 25.0, "an update stays well under a frame's budget",
      f"mean {times.mean():.2f} ms, max {times.max():.2f} ms per 0.25 s of audio")

print()
if failures:
    print("FAILURES:", ", ".join(failures))
    sys.exit(1)
print("STREAMING DECODE CHECKS PASSED")
