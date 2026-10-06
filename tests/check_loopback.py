"""Verify that system-sound capture really captures what the speakers play.

Plays a known tone out of the speakers and reads it back through WASAPI loopback,
comparing frequency and level.  This is the only way to prove the feature works:
no amount of successfully opening a stream shows that audio actually arrives.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sstv_bootstrap
import numpy as np
from sstv import audio, wasapi, dsp

RATE = 48000
failures = []


def check(ok, what, detail=""):
    print("  [%s] %-46s %s" % ("PASS" if ok else "FAIL", what, detail))
    if not ok:
        failures.append(what)


print("1. opening the loopback stream")
devices = wasapi.list_loopback_devices()
check(bool(devices), "an output device to listen to was found",
      devices[0].name if devices else "none")

stream = wasapi.LoopbackStream(sample_rate=RATE, target_rate=RATE,
                               device_id=devices[0].device_id if devices else None)
try:
    stream.open()
    opened = True
except Exception as exc:
    opened = False
    print("      open failed:", exc)
check(opened, "the loopback stream opens", f"device {stream.device_name!r}")
if not opened:
    print("\nFAILURES:", failures)
    sys.exit(1)
print("      mix format: %d Hz, %d channel(s), %s"
      % (stream.mix_rate, stream.channels, "float" if stream._is_float else "int"))

print()
print("2. silence reads as silence rather than as a signal")
t0 = time.monotonic()
quiet = stream.read(timeout=0.3)
elapsed = time.monotonic() - t0
# Loopback may hand back packets of digital silence rather than nothing at all,
# so the requirement is that it is *quiet* and that the read returns promptly.
quiet_peak = float(np.max(np.abs(quiet))) if quiet.size else 0.0
check(quiet_peak < 0.01, "reading while nothing plays yields no signal",
      f"peak {quiet_peak:.4f} over {quiet.size} samples in {elapsed:.2f} s")
check(elapsed < 1.5, "a silent read returns promptly", f"{elapsed:.2f} s")

print()
print("3. a played tone comes back through the capture")
tone_hz = 1500.0
seconds = 2.0
tone = (np.sin(2 * np.pi * tone_hz * np.arange(int(RATE * seconds)) / RATE) * 0.25)
player = audio.AudioPlayer(RATE)
player.play(tone)
collected = []
deadline = time.monotonic() + seconds + 3.0
while time.monotonic() < deadline:
    chunk = stream.read(timeout=0.2)
    if chunk.size:
        collected.append(chunk)
    if not player.is_playing and collected and time.monotonic() > deadline - 2.5:
        break
player.stop(wait=True)

captured = np.concatenate(collected) if collected else np.zeros(0)
check(captured.size > RATE * 0.3, "audio was captured while the tone played",
      f"{captured.size} samples ({captured.size / RATE:.2f} s)")

if captured.size > RATE * 0.3:
    peak = float(np.max(np.abs(captured)))
    check(peak > 0.02, "the captured audio is not silence", f"peak amplitude {peak:.3f}")
    # Measure the tone in the middle, away from the start/stop transitions.
    middle = captured[int(captured.size * 0.25): int(captured.size * 0.75)]
    if middle.size > 4096:
        track = dsp.instantaneous_frequency(middle, RATE)
        median_hz = float(np.median(track[1000:-1000])) if track.size > 2000 else 0.0
        check(abs(median_hz - tone_hz) < 30.0,
              "the captured tone is the tone that was played",
              f"played {tone_hz:.0f} Hz, measured {median_hz:.1f} Hz")

print()
print("4. cleanup")
stream.close()
check(not stream.is_open, "the stream closes cleanly")
stream.close()
check(True, "closing twice is harmless")

print()
if failures:
    print("FAILURES:", ", ".join(failures))
    sys.exit(1)
print("SYSTEM-SOUND CAPTURE CHECKS PASSED")
