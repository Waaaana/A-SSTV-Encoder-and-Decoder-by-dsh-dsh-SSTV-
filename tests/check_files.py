"""End-to-end file checks: the paths the GUI buttons actually call."""
import sys, os, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sstv_bootstrap
import numpy as np
from PIL import Image

from sstv import audio, decoder, encoder, modes, vis

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)
RATE = 48000

ok = True
print("1. encode -> WAV file -> read back -> decode")
for name in ("Robot 36", "Martin M1", "Scottie S1", "PD 120"):
    spec = modes.get_mode(name)
    W, H = spec.resolution
    img = encoder.make_test_image(W, H)
    path = os.path.join(OUT, "file-%s.wav" % name.replace(" ", ""))
    encoder.encode_to_wav(img, spec, path, RATE)
    size = os.path.getsize(path)
    samples, rate = audio.read_wav(path)
    result = decoder.decode(samples, rate)
    same = result.mode is not None and result.mode.name == name and result.complete
    ok = ok and same
    print("   %-11s %7.1f kB  %8.2f s  rate %d  -> %s %d/%d lines  %s" % (
        name, size / 1024, len(samples) / rate, rate,
        result.mode.name if result.mode else "?", result.lines_decoded,
        result.total_lines, "OK" if same else "FAIL"))

print()
print("2. VIS identification alone (what the receiver does first)")
for name in ("Robot 36", "PD 180", "Scottie DX"):
    spec = modes.get_mode(name)
    res = encoder.encode(encoder.make_test_image(*spec.resolution), spec, RATE)
    found = vis.decode_vis(res.samples, RATE)
    identified = decoder.identify(res.samples, RATE)
    good = found is not None and identified is not None and identified.name == name
    ok = ok and good
    print("   %-11s code %-3d confidence %.2f -> %s  %s" % (
        name, found.code, found.confidence,
        identified.name if identified else "?", "OK" if good else "FAIL"))

print()
print("3. decode survives added noise (a real radio recording is never clean)")
rng = np.random.default_rng(11)
for level in (0.05, 0.15, 0.30):
    spec = modes.get_mode("Robot 36")
    res = encoder.encode(encoder.make_test_image(*spec.resolution), spec, RATE)
    noisy = res.samples + level * rng.standard_normal(res.samples.size)
    result = decoder.decode(noisy, RATE)
    good = result.mode is not None and result.complete
    ok = ok and good
    print("   noise %2.0f%%  -> %s  %d/%d lines  %s" % (
        level * 100, result.mode.name if result.mode else "none",
        result.lines_decoded, result.total_lines, "OK" if good else "FAIL"))

print()
print("4. decode survives amplitude loss (weak signal)")
spec = modes.get_mode("Scottie S1")
res = encoder.encode(encoder.make_test_image(*spec.resolution), spec, RATE)
for level in (0.5, 0.2, 0.08):
    result = decoder.decode(res.samples * level, RATE)
    good = result.mode is not None and result.complete
    ok = ok and good
    print("   amplitude %.2f -> %s  %d/%d lines  %s" % (
        level, result.mode.name if result.mode else "none",
        result.lines_decoded, result.total_lines, "OK" if good else "FAIL"))

print()
print("5. decode survives a tuning offset (receiver not on frequency)")
base_track = None
from sstv import dsp
for offset in (20.0, 40.0, -35.0):
    res = encoder.encode(encoder.make_test_image(*spec.resolution), spec, RATE)
    track = dsp.instantaneous_frequency(res.samples, RATE) + offset
    result = decoder.decode(res.samples, RATE, track_override=None) if False else None
    # Feed the offset through the only public route: shift the audio's pitch by
    # resampling is wrong, so instead re-derive the track and hand it to a decoder.
    d = decoder.SstvDecoder(sample_rate=RATE)
    d._reset()
    d._header = vis.decode_vis(res.samples, RATE, track=track)
    mode = modes.mode_by_vis(d._header.code) if d._header else None
    d._mode = mode
    disc = dsp.Discriminator(track, RATE)
    d._calibrate(disc, d._header.end)
    shifted = d._shifted(disc, d._offset)
    d._planes = decoder._Planes(mode)
    d._decode_lines(shifted, d._header.end)
    out = d._result()
    good = out.complete
    ok = ok and good
    print("   offset %+6.1f Hz -> measured %+6.1f Hz  %d/%d lines  %s" % (
        offset, out.frequency_offset_hz, out.lines_decoded, out.total_lines,
        "OK" if good else "FAIL"))

print()
print("6. command line interface")
cli = os.path.join(ROOT, "sstv_app.py")
img_path = os.path.join(OUT, "cli-source.png")
Image.fromarray(np.asarray(encoder.make_test_image(320, 240))).save(img_path)
enc_out = os.path.join(OUT, "cli-encoded.wav")
r1 = subprocess.run([sys.executable, "-X", "utf8", cli, "--encode", img_path,
                     "--mode", "Robot 36", "--out", enc_out],
                    capture_output=True, text=True, encoding="utf-8", errors="replace")
good1 = r1.returncode == 0 and os.path.exists(enc_out)
r2 = subprocess.run([sys.executable, "-X", "utf8", cli, "--decode", enc_out],
                    capture_output=True, text=True, encoding="utf-8", errors="replace")
out_png = os.path.join(OUT, "cli-encoded-decoded.png")
good2 = r2.returncode == 0 and os.path.exists(out_png)
ok = ok and good1 and good2
print("   --encode exit %d, wav written %s" % (r1.returncode, good1))
print("     ", (r1.stdout or "").strip().replace("\n", " | "))
print("   --decode exit %d, png written %s" % (r2.returncode, good2))
print("     ", (r2.stdout or "").strip().replace("\n", " | "))
if r1.stderr.strip():
    print("   encode stderr:", r1.stderr.strip()[:400])
if r2.stderr.strip():
    print("   decode stderr:", r2.stderr.strip()[:400])

print()
print("=" * 60)
print("FILE AND CLI CHECKS:", "ALL PASSED" if ok else "FAILURES PRESENT")
