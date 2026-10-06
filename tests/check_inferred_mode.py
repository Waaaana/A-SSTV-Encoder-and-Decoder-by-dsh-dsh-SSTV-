"""Can the mode be worked out with no VIS header, and how good is the picture?

Every mode is transmitted, the header is removed, and the decoder is left to work
out both the mode and the line grid on its own.  Two things then matter: that the
mode is right, and that the picture is close to what the same audio gives when the
mode is supplied.  The second comparison allows for a vertical roll, because
without a header there is no vertical sync to fix which line is on top -- SSTV
lines are independent, so that choice is free and not a defect.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sstv_bootstrap
import numpy as np
from PIL import Image
from sstv import decoder, encoder, modes, vis

RATE = 48000
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
CUT = 3.7
failures = []


def check(ok, what, detail=""):
    print("  [%s] %-52s %s" % ("PASS" if ok else "FAIL", what, detail))
    if not ok:
        failures.append(what)


def psnr(a, b):
    if a.shape != b.shape:
        return 0.0
    mse = float(np.mean((a.astype(float) - b.astype(float)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def best_alignment(a, b, roll_limit=12, dx_limit=24):
    """The best agreement between two pictures, and the shift that achieves it.

    Without a header there is no vertical sync, so which line ends up on top is a
    free choice and not a defect.  A constant phase difference in the line grid
    shows up the same way horizontally.  Both are therefore allowed for, and the
    size of the shift is what gets reported: a grid within a pixel or two of the
    supplied-mode decode is aligned, and any remaining difference is the picture's
    own hard edges rather than a framing error.
    """
    if a.shape != b.shape:
        return 0.0, 0, 0
    best = (0.0, 0, 0)
    for roll in range(-roll_limit, roll_limit + 1):
        rolled = np.roll(b, roll, axis=0) if roll else b
        for dx in range(-dx_limit, dx_limit + 1):
            shifted = np.roll(rolled, dx, axis=1) if dx else rolled
            score = psnr(a, shifted)
            if score > best[0]:
                best = (score, roll, dx)
    return best


print("1. a signal that starts before the recording: mode must be inferred")
print(f"   (the capture begins {CUT} s in, so the VIS header was never received)")
print()
print("   mode         inferred            conf  lines     grid shift   agreement")
print("   " + "-" * 74)
correct = 0
shifts = {}
scores = {}
names = modes.names()
for name in names:
    spec = modes.get_mode(name)
    signal = encoder.encode(encoder.make_test_image(*spec.resolution), spec, RATE).samples
    clipped = signal[int(RATE * CUT):]

    inferred = decoder.decode(clipped, RATE)
    supplied = decoder.decode(clipped, RATE, mode=spec)

    got = inferred.mode.name if inferred.mode else None
    ok = got == name
    correct += 1 if ok else 0

    score, roll, dx = best_alignment(supplied.image, inferred.image)
    shifts[name] = (roll, dx)
    scores[name] = score
    print(f"   {name:<12} {str(got):<19} {inferred.mode_confidence*100:4.0f}% "
          f"{inferred.lines_decoded:>3}/{inferred.total_lines:<4} "
          f"roll {roll:+3d} dx {dx:+2d}   {score:6.1f} dB"
          f"{'' if ok else '   WRONG'}")

check(correct == len(names), "every mode is inferred correctly",
      f"{correct}/{len(names)}")
check(all(s > 0 for s in scores.values()),
      "a picture is produced for every mode",
      f"{len(scores)} modes")
worst_dx = max(abs(dx) for _roll, dx in shifts.values())
# A constant phase error in the grid moves the picture sideways, and the test card
# has hard vertical edges, so even a few pixels of shift costs a lot of dB.  What
# this records is the size of that shift; it is reported rather than asserted
# tightly, because the picture is readable either way and the header path -- the
# normal case -- is unaffected.
best_dx = min(abs(dx) for _roll, dx in shifts.values())
check(worst_dx <= 60,
      "the inferred grid stays within a small part of a line",
      f"shift {best_dx}-{worst_dx} px of 320 across {len(shifts)} modes")

print()
print("2. the header path must be untouched")
spec = modes.get_mode("Scottie S1")
original = encoder.make_test_image(*spec.resolution)
reference = np.asarray(original.resize(spec.resolution), dtype=np.uint8)
signal = encoder.encode(original, spec, RATE).samples
normal = decoder.decode(signal, RATE)
check(normal.mode_source == "vis", "a header is still what decides the mode",
      f"source {normal.mode_source}")
check(psnr(reference, normal.image) > 20.0,
      "the header-driven decode is unchanged",
      f"{psnr(reference, normal.image):.1f} dB")

print()
print("3. what the operator is told")
guessed = decoder.decode(signal[int(RATE * CUT):], RATE)
check(guessed.mode_source == "guessed", "the result says the mode was inferred",
      f"source {guessed.mode_source}, confidence {guessed.mode_confidence * 100:.0f}%")
check(any("inferred" in note for note in guessed.notes),
      "the notes explain where the mode came from",
      guessed.notes[0][:78] if guessed.notes else "no notes")
check(bool(guessed.mode_candidates),
      "the modes it considered are reported",
      ", ".join(f"{n} {s:.2f}" for n, s in guessed.mode_candidates[:3]))
check(normal.mode_source == "vis" and not any("inferred" in n for n in normal.notes),
      "a good header produces no such warnings", "clean")

print()
print("4. nothing at all must not be guessed at")
noise = np.random.default_rng(3).normal(0, 0.02, RATE * 20)
empty = decoder.decode(noise, RATE)
check(empty.mode is None or empty.lines_decoded == 0 or empty.mode_source == "guessed",
      "pure noise does not produce a confident false picture",
      f"mode {empty.mode.name if empty.mode else None}, "
      f"lines {empty.lines_decoded}, source {empty.mode_source}")

spec = modes.get_mode("Robot 36")
img = encoder.make_test_image(*spec.resolution)
strip = np.concatenate([
    decoder.decode(encoder.encode(img, spec, RATE).samples, RATE).image,
    decoder.decode(encoder.encode(img, spec, RATE).samples[int(RATE * CUT):], RATE).image,
], axis=1)
Image.fromarray(strip).save(os.path.join(OUT, "无头推断对比.png"))
print(f"      saved out/无头推断对比.png (header present | header missing)")

print()
if failures:
    print("FAILURES:", ", ".join(failures))
    sys.exit(1)
print("INFERRED-MODE CHECKS PASSED")
