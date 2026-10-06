import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sstv_bootstrap
from sstv import modes

EXPECTED_TOTAL = {
    "Robot 36": 36.0, "Robot 72": 72.0, "Martin M1": 114.3, "Martin M2": 58.0,
    "Scottie S1": 109.7, "Scottie S2": 71.1, "Scottie DX": 268.9,
    "PD 50": 49.7, "PD 90": 90.0, "PD 120": 126.1, "PD 160": 160.9,
    "PD 180": 187.1, "PD 240": 248.0, "PD 290": 288.7,
}

print("%-12s %-4s %-5s %-8s %-6s %-7s %-7s %-7s %s" % (
    "mode", "VIS", "hex", "line_ms", "lines", "imgWxH", "pic_s", "tot_s", "ok"))
bad = 0
for s in modes.MODE_LIST:
    tot = s.duration_seconds + modes.VIS_HEADER_MS / 1000.0
    want = EXPECTED_TOTAL.get(s.name)
    plan_ms = sum(seg.milliseconds for seg in s.line_plan())
    ok = (abs(plan_ms - s.line_milliseconds) < 1e-6
          and (want is None or abs(tot - want) < 1.2))
    if not ok:
        bad += 1
    print("%-12s %-4d 0x%02X  %-8.3f %-6d %-7s %-7.1f %-7.2f %s" % (
        s.name, s.vis_code, s.vis_code, s.line_milliseconds, s.line_count,
        "%dx%d" % s.resolution, s.duration_seconds, tot,
        "OK" if ok else "FAIL(want %.1f)" % (want or -1)))

print()
print("inconsistent modes:", bad)
print()
print("VIS plan for Robot 36 (code 8):")
for hz, ms in modes.vis_plan(8):
    print("   %7.1f Hz  %6.1f ms" % (hz, ms))
print("total header ms:", sum(ms for _, ms in modes.vis_plan(8)), "expected", modes.VIS_HEADER_MS)
print()
print("parity checks (even parity => total ones incl. parity bit is even):")
for s in modes.MODE_LIST:
    bits = modes.vis_bits(s.vis_code)
    p = modes.vis_parity(s.vis_code)
    byte = s.vis_code | (p << 7)
    assert (sum(bits) + p) % 2 == 0, s.name
    print("   %-12s bits=%s parity=%d full_byte=0x%02X" % (s.name, "".join(map(str, bits)), p, byte))
