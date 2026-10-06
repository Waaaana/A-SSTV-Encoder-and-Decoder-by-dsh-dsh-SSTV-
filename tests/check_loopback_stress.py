"""Repeat the speaker -> loopback -> decode path several times over.

Each round plays a whole transmission and decodes it back off the output device.
One pass can be unlucky -- the sound card is shared with the rest of the desktop
-- so repeating the run is what shows whether system-sound decoding is dependable
or merely occasionally works.  Any round that fails leaves its audio in ``out/``
so the cause can be examined rather than guessed at.

Usage: ``python tests/check_loopback_stress.py [rounds]``
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sstv_bootstrap
import numpy as np
from sstv import audio, decoder, dsp, encoder, modes, vis

RATE = 48000
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 6

spec = modes.get_mode("Robot 36")
original = encoder.make_test_image(*spec.resolution)
transmission = encoder.encode(original, spec, RATE)
device = audio.list_loopback_devices()[0]
print(f"{ROUNDS} rounds, {spec.name}, {transmission.duration:.1f} s each")

failures = 0
for round_index in range(1, ROUNDS + 1):
    recorder = audio.AudioRecorder(RATE, loopback=True, device_id=device.device_id)
    recorder.start()
    ready = recorder.wait_until_ready(2.0)
    player = audio.AudioPlayer(RATE)
    player.play(transmission.samples * 0.5)
    player.wait(timeout=transmission.duration + 5.0)
    time.sleep(0.3)
    recorder.stop(wait=True)

    samples = np.asarray(recorder.samples(), dtype=np.float64)
    result = decoder.decode(samples, RATE)
    ok = result.mode is not None and result.complete
    note = (f"mode={result.mode.name if result.mode else None} "
            f"lines={result.lines_decoded}/{result.total_lines} "
            f"len={samples.size / RATE:.1f}s peak={np.max(np.abs(samples)) if samples.size else 0:.4f}")
    print(f"  round {round_index}: {'PASS' if ok else 'FAIL'}  ready={ready}  {note}")
    if not ok:
        failures += 1
        path = os.path.join(OUT, f"捕获失败-{round_index}.wav")
        audio.write_wav(path, samples, RATE)
        header = vis.decode_vis(samples, RATE)
        window = int(RATE * 0.1)
        blocks = np.abs(samples[: samples.size // window * window]).reshape(-1, window).max(axis=1)
        active = np.flatnonzero(blocks > 0.05)
        print(f"      saved {path}")
        print(f"      VIS={header.code if header else None}  "
              f"signal {active[0] * 0.1 if active.size else -1:.2f}s.."
              f"{active[-1] * 0.1 if active.size else -1:.2f}s  "
              f"first 0.7 s peak {np.max(np.abs(samples[: int(RATE * 0.7)])):.4f}")
        if samples.size > RATE:
            track = dsp.instantaneous_frequency(samples[: RATE], RATE)
            print(f"      first second median freq {np.median(track[1000:-1000]):.0f} Hz")

print()
print(f"{ROUNDS - failures}/{ROUNDS} rounds decoded")
sys.exit(1 if failures else 0)
