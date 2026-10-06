import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sstv_bootstrap
import numpy as np
from sstv import modes, encoder, dsp

RATE = 48000


def find_header_start(track, rate):
    """Locate the end of the second leader tone (start of the start bit).

    The header is leader(1900) -> break(1200) -> leader(1900) -> bits.  The last
    1900->1200 transition before the bit field is exactly the start bit, so we
    scan for the final falling edge inside the leader window instead of trusting
    a hard-coded offset.
    """
    window = track[: int(1.2 * rate)]
    is_leader = np.abs(window - 1900.0) < 250.0
    # indices where leader turns off after 0.6 s
    start = int(0.6 * rate)
    for i in range(start, len(is_leader) - 1):
        if is_leader[i] and not is_leader[i + 1]:
            return i / rate
    return None


def decode_vis(wave, rate):
    track = dsp.instantaneous_frequency(wave, rate)
    t0 = find_header_start(track, rate)
    if t0 is None:
        return None
    d = dsp.Discriminator(track, rate)
    t = t0 + 0.030  # skip the start bit
    bits, hz = [], []
    for _ in range(7):
        h = d.average(t + 0.006, t + 0.024)
        hz.append(round(h))
        bits.append(1 if abs(h - 1100.0) < abs(h - 1300.0) else 0)
        t += 0.030
    ph = d.average(t + 0.006, t + 0.024)
    parity = 1 if abs(ph - 1100.0) < abs(ph - 1300.0) else 0
    t += 0.030
    stop = d.average(t + 0.006, t + 0.024)
    code = sum(b << i for i, b in enumerate(bits))
    return {
        "t0": t0, "code": code, "bits": bits, "hz": hz,
        "parity": parity, "stop_hz": round(stop),
    }


img = encoder.make_test_image(320, 256)
print("=== VIS header decode for every mode ===")
fails = 0
for spec in modes.MODE_LIST:
    res = encoder.encode(img, spec, RATE)
    got = decode_vis(res.samples, RATE)
    want_par = modes.vis_parity(spec.vis_code)
    ok = (got and got["code"] == spec.vis_code and got["parity"] == want_par
          and abs(got["stop_hz"] - 1200) < 60)
    if not ok:
        fails += 1
    print("%-12s code=%-3d want=%-3d %s  par=%d/%d stop=%dHs  bits=%s hz=%s" % (
        spec.name, got["code"], spec.vis_code, "OK" if ok else "FAIL",
        got["parity"], want_par, got["stop_hz"], "".join(map(str, got["bits"])), got["hz"]))

print()
print("VIS failures:", fails)

print()
print("=== line structure check (no header) ===")
for name in ("Robot 36", "Martin M1", "Scottie S1", "PD 120"):
    spec = modes.get_mode(name)
    res = encoder.encode(img, spec, RATE, include_header=False)
    track = dsp.instantaneous_frequency(res.samples, RATE)
    d = dsp.Discriminator(track, RATE)
    print("%s: %d lines of %.3f ms" % (name, spec.line_count, spec.line_milliseconds))
    for seg, st, en in spec.segment_offsets():
        if seg.seconds <= 0:
            continue
        if seg.is_image:
            vals = d.slice_values(st + seg.seconds * 0.1, st + seg.seconds * 0.3, 6)
            print("   %-10s %7.3f-%7.3f ms  px=%-4d first levels=%s" % (
                seg.channel.value, st * 1000, en * 1000, seg.pixels,
                np.round(vals).astype(int)))
        else:
            # Stay strictly inside the segment: a window that overlaps the next
            # segment averages two different tones and reads low.
            mid = (st + en) / 2
            half = max(0.0002, seg.seconds * 0.25)
            f = d.average(mid - half, mid + half)
            print("   %-10s %7.3f-%7.3f ms  tone=%7.1f Hz (want %7.1f) %s" % (
                seg.channel.value, st * 1000, en * 1000, f, seg.frequency,
                "OK" if abs(f - seg.frequency) < 30 else "BAD"))
