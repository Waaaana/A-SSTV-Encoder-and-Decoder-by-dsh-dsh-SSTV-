"""Exercise the interface's system-sound path, not just the audio layer.

Runs the window, selects the system-sound source, starts the capture through the
real button handler, plays a transmission, stops, and checks that the app decodes
it and that the waterfall filled in along the way.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sstv_bootstrap
import numpy as np
from sstv import audio, encoder, modes, gui as gui_module

RATE = 48000
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)
failures = []


def check(ok, what, detail=""):
    print("  [%s] %-48s %s" % ("PASS" if ok else "FAIL", what, detail))
    if not ok:
        failures.append(what)


import tkinter as tk
root = tk.Tk()
root.geometry("1220x880")
app = gui_module.SstvApp(root)
app.notebook.select(app.rx_tab)
for _ in range(8):
    root.update_idletasks(); root.update(); time.sleep(0.03)

check(len(app._loopbacks) > 0, "the window found a system-sound device",
      app._loopbacks[0].name if app._loopbacks else "none")
if not app._loopbacks:
    root.withdraw(); app.close()
    print("\nFAILURES:", failures); sys.exit(1)

print()
print("1. selecting the system-sound source")
app.rx_source_var.set("loopback")
app._on_source_changed()
for _ in range(4):
    root.update_idletasks(); root.update(); time.sleep(0.03)
check(app._active_inputs == list(app._loopbacks), "the device list switched to output devices",
      f"{len(app._active_inputs)} device(s)")
check("system sound" in app.rx_record_btn["text"], "the button says what it will listen to",
      app.rx_record_btn["text"])

print()
print("2. recording a transmission through the interface")
spec = modes.get_mode("Robot 36")
original = encoder.make_test_image(*spec.resolution)
transmission = encoder.encode(original, spec, RATE)

app.rx_toggle_record()                      # press the button, as a user would
for _ in range(10):
    root.update_idletasks(); root.update(); time.sleep(0.03)
check(app.recorder.loopback, "the recorder is in system-sound mode", "loopback=True")

player = audio.AudioPlayer(RATE)
player.play(transmission.samples * 0.5)
# The point of progressive decoding: lines must appear well before the
# transmission finishes, rather than only once the operator stops recording.
marks = []
started = time.monotonic()
deadline = started + transmission.duration + 4.0
while time.monotonic() < deadline:
    root.update_idletasks(); root.update()
    if app._live_progress is not None:
        lines = app._live_progress.lines_decoded
        if not marks or lines != marks[-1][1]:
            marks.append((round(time.monotonic() - started, 1), lines,
                          app.rx_image is not None))
        if app._live_progress.complete:
            break
    time.sleep(0.03)
elapsed_to_complete = time.monotonic() - started
player.stop(wait=True)
time.sleep(0.3)
for _ in range(6):
    root.update_idletasks(); root.update(); time.sleep(0.03)

mid = [m for m in marks if 0 < m[1] < spec.line_count]
check(len(mid) >= 3, "lines appear while the transmission is still arriving",
      "; ".join(f"t={m[0]}s {m[1]} lines" for m in mid[:4])
      + (" ..." if len(mid) > 4 else ""))
check(any(m[2] for m in mid), "a partial picture is drawn before the end",
      f"{sum(1 for m in mid if m[2])} of {len(mid)} updates drew an image")
# The last line cannot be decoded before its own audio has arrived, so the
# requirement is that the picture is essentially complete well before the end and
# that the tail costs no measurable extra time.
near_end = [m for m in marks if m[1] >= spec.line_count - 5]
check(bool(near_end), "the picture is essentially complete before the end",
      f"{spec.line_count - 5} lines by t={near_end[0][0]} s of {transmission.duration:.1f} s"
      if near_end else "never reached")
check(elapsed_to_complete <= transmission.duration + 1.0,
      "the final line costs no measurable extra time",
      f"complete at {elapsed_to_complete:.1f} s of {transmission.duration:.1f} s")
first_lines = next((m for m in marks if m[1] > 0), None)
check(first_lines is not None and first_lines[0] < 6.0,
      "the first lines appear within a few seconds",
      f"first line at t={first_lines[0]} s" if first_lines else "never")

captured = app.recorder.samples()
check(len(captured) > RATE * 5, "the interface captured the audio",
      f"{len(captured)} samples ({len(captured) / RATE:.1f} s)")
check(app.waterfall.frames > 50, "the waterfall filled in while listening",
      f"{app.waterfall.frames} spectrum rows")

print()
print("3. stopping finishes the picture immediately")
app.rx_toggle_record()                      # press again to stop
stop_started = time.monotonic()
deadline = stop_started + 30.0
while time.monotonic() < deadline:
    root.update_idletasks(); root.update()
    text = app.rx_info["text"] or ""
    if app.rx_image is not None and text and "Receiving" not in text:
        break
    time.sleep(0.03)
stop_seconds = time.monotonic() - stop_started
check(stop_seconds < 3.0, "stopping does not start a long decode",
      f"finished {stop_seconds:.2f} s after pressing stop")

if app.rx_image is None:
    # Report what state it got stuck in, rather than just "no picture".
    print("      diagnostics:")
    print("        status line :", app.status["text"])
    print("        tasks queued:", app._tasks.qsize(), " worker busy:", app._worker_busy)
    print("        recorder    :", app.recorder.state, "error:", app.recorder.error)
    print("        rx_info     :", repr((app.rx_info["text"] or "")[:80]))
    # Keep the audio, so the reason can be found rather than guessed at.
    import wave, array, sstv_bootstrap as _bootstrap  # noqa: F401
    from sstv import audio as _audio
    debug_wav = os.path.join(OUT, "界面环回捕获.wav")
    _audio.write_wav(debug_wav, app.recorder.samples(), RATE)
    print("        saved capture:", debug_wav,
          f"({os.path.getsize(debug_wav) // 1024} kB)")
    kept = np.asarray(app.recorder.samples(), dtype=np.float64)
    print("        captured peak:", float(np.max(np.abs(kept))) if kept.size else 0.0)
    from sstv import vis as _vis
    header = _vis.decode_vis(kept, RATE) if kept.size else None
    print("        VIS on capture:", header.code if header else None)
    if kept.size:
        head = kept[: int(RATE * 0.1)]
        print("        first 0.1 s rms:", float(np.sqrt(np.mean(head ** 2))))

check(app.rx_image is not None, "a picture was produced",
      None if app.rx_image is None else str(app.rx_image.shape))
if app.rx_image is not None:
    summary = app.rx_info["text"] or ""
    check("Robot 36" in summary, "the report names the decoded mode",
          summary.splitlines()[1] if "\n" in summary else summary[:60])
    check("240 / 240" in summary, "every line of the picture was recovered",
          [ln for ln in summary.splitlines() if "Lines" in ln][0] if "Lines" in summary else "")
    from PIL import Image
    path = os.path.join(OUT, "系统声音-界面解码.png")
    Image.fromarray(app.rx_image).save(path)
    print(f"      saved {path}")

print()
print("4. falling back when there is no system-sound device")
app.rx_source_var.set("microphone")
app._on_source_changed()
check(app._active_inputs == list(app._inputs), "the microphone list comes back",
      f"{len(app._active_inputs)} input(s)")

root.withdraw()
app.close()
print()
if failures:
    print("FAILURES:", ", ".join(failures))
    sys.exit(1)
print("SYSTEM-SOUND INTERFACE CHECKS PASSED")
