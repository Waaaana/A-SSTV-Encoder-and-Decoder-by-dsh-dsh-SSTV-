"""End-to-end: send SSTV out of the speakers and decode it from system sound.

This exercises the whole chain the feature exists for -- encode, play, capture the
output device, decode -- and compares the recovered picture against the original,
so it proves the feature works rather than merely that it opens.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sstv_bootstrap
import numpy as np
from sstv import audio, decoder, encoder, modes

RATE = 48000
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)
failures = []


def check(ok, what, detail=""):
    print("  [%s] %-48s %s" % ("PASS" if ok else "FAIL", what, detail))
    if not ok:
        failures.append(what)


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


devices = audio.list_loopback_devices()
check(bool(devices), "a system-sound device is offered by the audio layer",
      devices[0].name if devices else "none")
if not devices:
    print("\nFAILURES:", failures); sys.exit(1)

spec = modes.get_mode("Robot 36")
original = encoder.make_test_image(*spec.resolution)
transmission = encoder.encode(original, spec, RATE)
print(f"      transmission: {spec.name}, {transmission.duration:.1f} s, "
      f"{len(transmission.samples)} samples")

print()
print("1. capture the transmission through the output device")
recorder = audio.AudioRecorder(RATE, loopback=True, device_id=devices[0].device_id)
recorder.start()
recorder.wait_until_ready(2.0)        # do not play into a device that is not listening yet
if recorder.error:
    check(False, "the loopback recorder starts", recorder.error)
    print("\nFAILURES:", failures); sys.exit(1)

player = audio.AudioPlayer(RATE)
player.play(transmission.samples * 0.5)   # a moderate volume, as a person would
player.wait(timeout=transmission.duration + 5.0)
time.sleep(0.4)                       # let the tail arrive
recorder.stop(wait=True)

captured = recorder.samples()
check(len(captured) > RATE * 5, "audio was captured from the output device",
      f"{len(captured)} samples ({len(captured) / RATE:.1f} s), peak {recorder.peak}")
if len(captured) < RATE * 5:
    print("\nFAILURES:", failures); sys.exit(1)

print()
print("2. decode the captured audio")
captured_array = np.asarray(captured, dtype=np.float64)
result = decoder.decode(captured_array, RATE)
check(result.mode is not None, "the transmission was identified",
      f"{result.mode.name if result.mode else 'nothing'} "
      f"(VIS {result.vis_code}, confidence {result.vis_confidence * 100:.0f}%)")
if result.mode is not None:
    check(result.mode.name == spec.name, "the right mode was identified", result.mode.name)
    check(result.complete, "the whole picture was decoded",
          f"{result.lines_decoded}/{result.total_lines} lines")

    reference = np.asarray(original.resize(spec.resolution), dtype=np.uint8)
    if result.image.size:
        score = psnr(reference, result.image)
        # The bar is not "perfect": the same picture decoded straight from the
        # encoder scores about the same, because the test card's hard edges cost
        # a few dB on their own.  What matters is that going out of the speakers
        # and back in through the sound card adds nothing measurable.
        check(score > 15.0, "the decoded picture is a good likeness",
              f"{score:.1f} dB PSNR")

        direct = decoder.decode(transmission.samples * 0.5, RATE)
        direct_score = psnr(reference, direct.image) if direct.image.size else 0.0
        check(score >= direct_score - 1.0,
              "the audio path costs nothing measurable",
              f"through the speakers {score:.1f} dB vs direct {direct_score:.1f} dB")

        from PIL import Image
        path = os.path.join(OUT, "系统声音解码.png")
        Image.fromarray(result.image).save(path)
        print(f"      saved {path}")

print()
if failures:
    print("FAILURES:", ", ".join(failures))
    sys.exit(1)
print("SYSTEM-SOUND DECODE PASSED")
