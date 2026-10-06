"""Exercise the waterfall as the window uses it: build, render, feed, control."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sstv_bootstrap
import numpy as np
from sstv import encoder, modes, waterfall as wf, gui as gui_module

RATE = 48000
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)

failures = []


def check(ok, what, detail=""):
    print("  [%s] %-46s %s" % ("PASS" if ok else "FAIL", what, detail))
    if not ok:
        failures.append(what)


import tkinter as tk
root = tk.Tk()
root.geometry("1180x820")
app = gui_module.SstvApp(root)
root.update()
print("window built; waterfall canvas %dx%d" % (
    app.rx_waterfall.winfo_width(), app.rx_waterfall.winfo_height()))

spec = modes.get_mode("Robot 36")
signal = encoder.encode(encoder.make_test_image(*spec.resolution), spec, RATE).samples

print()
print("1. rendering into the canvas")
app._render_waterfall()
root.update()
check(app._waterfall_photo is not None, "canvas received an image",
      "photo %s" % ("present" if app._waterfall_photo else "missing"))

print()
print("2. building the display from a whole recording (file path)")
app._build_waterfall_from_audio(signal, RATE)
root.update()
check(app.waterfall.frames > 100, "spectrum built from file audio",
      "%d frames, %.1f s of history" % (app.waterfall.frames, app.waterfall.history_seconds))
peak = app.waterfall.peak_hz()
check(1000 < peak < 2500, "peak frequency inside the SSTV band", "%.0f Hz" % peak)
check(app.rx_peak_label["text"] != "", "strength readout shown", app.rx_peak_label["text"])

print()
print("3. live feeding through the recorder cursor (microphone path)")
rec = app.recorder
rec._chunks = []
off = 0
pcm = (np.clip(signal[: RATE * 2], -1, 1) * 32767).astype("<i2").tobytes()
app.waterfall.clear()
app._waterfall_cursor = 0
step = 4800      # 100 ms of 16-bit mono
for start in range(0, len(pcm), step):
    rec._chunks.append(pcm[start : start + step])
raw, app._waterfall_cursor = rec.bytes_since(app._waterfall_cursor)
if raw:
    app.waterfall.feed(np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0, RATE)
app._render_waterfall()
root.update()
check(app.waterfall.frames > 20, "live incremental feed produced frames",
      "%d frames" % app.waterfall.frames)
check(app._waterfall_photo is not None, "canvas updated from live feed")

print()
print("4. settings actually change the view")
app.wf_span_var.set("1400 - 2000 Hz")
app.wf_seconds_var.set("3 seconds")
app._apply_waterfall_settings()
root.update()
low, high = app._waterfall_span()
check((low, high) == (1400.0, 2000.0), "span selection parsed", "%s - %s Hz" % (low, high))
check(abs(app.waterfall.seconds - 3.0) < 1e-6, "history selection applied",
      "%.1f s" % app.waterfall.seconds)
img = app.waterfall.to_image(width=400, height=120)
check(img.size == (400, 120), "rendered at the requested span", "%s" % (img.size,))

print()
print("5. every span and history option renders")
bad = []
for span in gui_module.WATERFALL_SPANS:
    for window in gui_module.WATERFALL_WINDOWS:
        app.wf_span_var.set(span)
        app.wf_seconds_var.set(window)
        app._apply_waterfall_settings()
        try:
            app._build_waterfall_from_audio(signal[: RATE * 2], RATE)
            root.update()
        except Exception as exc:
            bad.append("%s / %s: %s" % (span, window, exc))
check(not bad, "all %d span/history combinations work" % (
    len(gui_module.WATERFALL_SPANS) * len(gui_module.WATERFALL_WINDOWS)),
    "; ".join(bad[:2]) if bad else "no errors")

print()
print("6. 'suggested history for mode' matches the mode")
app.rx_mode_var.set("PD 180")
app.auto_mode_var.set(False)
app._suggest_waterfall_window()
root.update()
check(app.wf_seconds_var.get() == "6 seconds", "PD 180 suggests a suitable history",
      app.wf_seconds_var.get())
app.rx_mode_var.set("Robot 36")
app._suggest_waterfall_window()
check(app.wf_seconds_var.get() == "3 seconds", "Robot 36 suggests a shorter history",
      app.wf_seconds_var.get())
app.auto_mode_var.set(True)

print()
print("7. the scale can be turned off, and clearing works")
app.wf_scale_var.set(False)
app._mark_waterfall_dirty()
app._render_waterfall()
root.update()
app.wf_scale_var.set(True)
app.waterfall.feed(signal[: RATE])
before = app.waterfall.frames
app._clear_waterfall()
root.update()
check(app.waterfall.frames == 0 and before > 0, "clearing empties the history",
      "%d frames before, %d after" % (before, app.waterfall.frames))

print()
print("8. save a picture of the live display for inspection")
app.wf_span_var.set("1000 - 2500 Hz")
app.wf_seconds_var.set("6 seconds")
app._apply_waterfall_settings()
app._build_waterfall_from_audio(signal, RATE)
root.update()
canvas_img = app.waterfall.to_image(width=760, height=190)
canvas_img = wf.add_frequency_scale(canvas_img, app.waterfall.low_hz, app.waterfall.high_hz)
path = os.path.join(OUT, "瀑布图-界面.png")
canvas_img.save(path)
check(os.path.exists(path), "live display saved for inspection", path)

print()
print("9. shutdown cancels the update timer")
app._start_waterfall()
app.close()
check(app._waterfall_job is None, "waterfall timer cancelled on close")

print()
if failures:
    print("FAILURES: %s" % ", ".join(failures))
    sys.exit(1)
print("WATERFALL GUI CHECKS PASSED")
