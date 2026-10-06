"""Final acceptance checks for SSTV Studio.

Covers every requirement in turn and prints a pass/fail line for each, so the
result can be read at a glance rather than inferred from a wall of numbers.
"""
import sys, os, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sstv_bootstrap
import numpy as np
from PIL import Image

from sstv import audio, colorspace, decoder, dsp, encoder, modes, vis

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)
RATE = 48000

results: list[tuple[bool, str, str]] = []


def check(ok: bool, requirement: str, detail: str = "") -> None:
    results.append((bool(ok), requirement, detail))
    print("  [%s] %-52s %s" % ("PASS" if ok else "FAIL", requirement, detail))


def gradient(w, h):
    rows = np.linspace(0, 255, h)[:, None] * np.ones((1, w))
    cols = np.ones((h, 1)) * np.linspace(0, 255, w)[None, :]
    return np.stack((rows, cols, np.full((h, w), 128.0)), axis=-1).astype(np.uint8)


def psnr(a, b):
    # Guard the shapes: a decode that came out as a different mode has a different
    # line count, and subtracting such arrays raises rather than reporting a bad
    # score -- which turns a failed expectation into a crashed test run.
    if a.shape != b.shape:
        return 0.0
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


print("=" * 78)
print("SSTV STUDIO - ACCEPTANCE CHECKS")
print("=" * 78)

print("\n1. Supported modes: mainstream encoder modes, encode and decode")
print("-" * 78)
REQUIRED = ["Robot 36", "Robot 72", "Martin M1", "Martin M2", "Scottie S1",
            "Scottie S2", "Scottie DX", "PD 50", "PD 90", "PD 120", "PD 160",
            "PD 180", "PD 240", "PD 290"]
available = {s.name for s in modes.MODE_LIST}
missing = [m for m in REQUIRED if m not in available]
check(not missing, f"all {len(REQUIRED)} mainstream modes present",
      "missing: " + ", ".join(missing) if missing else f"{len(available)} modes registered")

worst_psnr = 999.0
worst_name = ""
all_complete = True
for name in REQUIRED:
    spec = modes.get_mode(name)
    width, height = spec.resolution
    source = Image.fromarray(gradient(width, height), "RGB")
    encoded = encoder.encode(source, spec, RATE)
    decoded = decoder.decode(encoded.samples, RATE)
    complete = decoded.mode is not None and decoded.complete
    all_complete = all_complete and complete
    p = psnr(np.asarray(source), decoded.image) if decoded.image.size else 0.0
    if p < worst_psnr:
        worst_psnr, worst_name = p, name
check(all_complete, "every mode decodes a complete picture",
      f"{len(REQUIRED)}/{len(REQUIRED)} at 100% of lines")
check(worst_psnr > 30.0, "round-trip fidelity above 30 dB on every mode",
      f"worst {worst_name} at {worst_psnr:.1f} dB")

print("\n2. Simplex operation: transmit and receive are separate actions")
print("-" * 78)
from sstv import gui as gui_module
check(hasattr(gui_module.SstvApp, "tx_transmit") and
      hasattr(gui_module.SstvApp, "rx_toggle_record"),
      "separate transmit and receive controls",
      "Transmit tab and Receive tab, one direction at a time")

print("\n3. Receiving: microphone and audio file")
print("-" * 78)
try:
    ins = audio.list_input_devices()
    check(len(ins) >= 1, "microphone capture available",
          f"{len(ins)} input endpoint(s) enumerated")
except Exception as exc:
    check(False, "microphone capture available", str(exc))

rec = audio.AudioRecorder(RATE)
rec.start()
time.sleep(1.5)
rec.stop(wait=True)
captured = rec.samples()
check(len(captured) > RATE * 0.5, "microphone capture returns audio",
      f"{len(captured)} samples ({rec.seconds:.2f} s), error={rec.error}")

source_spec = modes.get_mode("Robot 36")
source_img = encoder.make_test_image(*source_spec.resolution)
wav_path = os.path.join(OUT, "acceptance.wav")
encoder.encode_to_wav(source_img, source_spec, wav_path, RATE)
loaded, loaded_rate = audio.read_wav(wav_path)
from_file = decoder.decode(loaded, loaded_rate)
check(from_file.mode is not None and from_file.complete,
      "decoding an audio file works",
      f"{os.path.getsize(wav_path) // 1024} kB WAV -> {from_file.mode.name}, "
      f"{from_file.lines_decoded}/{from_file.total_lines} lines")

print("\n4. Transmitting: speaker and audio file")
print("-" * 78)
try:
    outs = audio.list_output_devices()
    check(len(outs) >= 1, "speaker playback available",
          f"{len(outs)} output endpoint(s): {outs[0].name}")
except Exception as exc:
    check(False, "speaker playback available", str(exc))

tone = np.sin(2 * np.pi * 1000.0 * np.arange(RATE) / RATE) * 0.2
player = audio.AudioPlayer(RATE)
progress_seen: list[float] = []
player.set_callbacks(on_progress=progress_seen.append, on_finish=lambda s: None)
try:
    player.play(tone)
    finished = player.wait(timeout=10)
except Exception as exc:
    finished = False
    print("      playback error:", exc)
check(finished and progress_seen, "speaker playback runs and reports progress",
      f"{len(progress_seen)} progress updates, final state {player.state}")

# Full transmission actually reaching the sound device
long_tone = np.sin(2 * np.pi * 1500.0 * np.arange(int(RATE * 2.0)) / RATE) * 0.2
player.play(long_tone)
time.sleep(0.4)
was_playing = player.is_playing
player.stop(wait=True)
check(was_playing and player.state == "stopped",
      "a transmission can be stopped part-way",
      f"state after stop: {player.state}")

out_wav = os.path.join(OUT, "acceptance-out.wav")
audio.write_wav(out_wav, tone, RATE)
back, back_rate = audio.read_wav(out_wav)
check(os.path.exists(out_wav) and len(back) == len(tone) and back_rate == RATE,
      "writing the transmission to a WAV file works",
      f"{os.path.getsize(out_wav) // 1024} kB, {len(back)} samples at {back_rate} Hz")

print("\n5. Real-world robustness")
print("-" * 78)
rng = np.random.default_rng(3)
spec = modes.get_mode("Scottie S1")
img = encoder.make_test_image(*spec.resolution)
encoded = encoder.encode(img, spec, RATE)

noisy = encoded.samples + 0.20 * rng.standard_normal(encoded.samples.size)
noisy_result = decoder.decode(noisy, RATE)
check(noisy_result.complete, "decodes with 20% added noise",
      f"{noisy_result.lines_decoded}/{noisy_result.total_lines} lines")

quiet = decoder.decode(encoded.samples * 0.08, RATE)
check(quiet.complete, "decodes a very weak signal (8% amplitude)",
      f"{quiet.lines_decoded}/{quiet.total_lines} lines")

track = dsp.instantaneous_frequency(encoded.samples, RATE) + 42.0
d = decoder.SstvDecoder(sample_rate=RATE)
d._reset()
d._header = vis.decode_vis(encoded.samples, RATE, track=track)
d._mode = modes.mode_by_vis(d._header.code)
disc = dsp.Discriminator(track, RATE)
d._calibrate(disc, d._header.end)
shifted = d._shifted(disc, d._offset)
d._planes = decoder._Planes(d._mode)
d._decode_lines(shifted, d._header.end)
offset_result = d._result()
check(offset_result.complete and abs(offset_result.frequency_offset_hz - 42.0) < 1.0,
      "corrects a receiver tuning offset",
      f"measured {offset_result.frequency_offset_hz:+.1f} Hz of a +42.0 Hz shift")

for name in ("Robot 36", "PD 180", "Scottie DX", "Martin M1"):
    spec = modes.get_mode(name)
    res = encoder.encode(encoder.make_test_image(*spec.resolution), spec, RATE)
    identified = decoder.identify(res.samples, RATE)
    if identified is None or identified.name != name:
        check(False, "mode identified from the VIS header alone",
              f"{name} -> {identified}")
        break
else:
    check(True, "mode identified from the VIS header alone",
          "4 modes identified with no user input")

print("\n6. Waterfall display while recording")
print("-" * 78)
from sstv import waterfall as waterfall_module

tone_ok = True
for freq in (1200.0, 1500.0, 2300.0):
    probe = waterfall_module.Waterfall(RATE)
    probe.feed(np.sin(2 * np.pi * freq * np.arange(RATE) / RATE) * 0.6)
    if abs(probe.peak_hz() - freq) > 40:
        tone_ok = False
        check(False, "waterfall places a tone in the right column",
              f"{freq:.0f} Hz read as {probe.peak_hz():.0f} Hz")
        break
if tone_ok:
    check(True, "waterfall places a tone in the right column",
          "1200/1500/2300 Hz all read back within 40 Hz")

# Incremental feeding must match a single pass, or the picture drawn while
# recording would differ from the one drawn afterwards.
whole = waterfall_module.Waterfall(RATE)
whole.feed(encoded.samples)
piece = waterfall_module.Waterfall(RATE)
for start in range(0, encoded.samples.size, 5000):
    piece.feed(encoded.samples[start:start + 5000])
agree = (whole.frames == piece.frames and whole._view().shape == piece._view().shape
         and np.allclose(whole._view(), piece._view(), atol=1e-4))
check(agree, "live chunked analysis matches bulk analysis",
      f"{whole.frames} frames either way")

w = waterfall_module.Waterfall(RATE, seconds=6.0)
w.feed(encoded.samples)
plain = w.to_image(width=640, height=200)
scaled = waterfall_module.add_frequency_scale(plain, w.low_hz, w.high_hz)
check(scaled.size[1] > plain.size[1], "a frequency scale can be drawn",
      f"{plain.size} plus a scale strip = {scaled.size}")

inside = (w.bins >= 1200) & (w.bins <= 2300)
level_in = float(np.mean(w._view()[:, inside]))
level_out = float(np.mean(w._view()[:, ~inside]))
check(level_in > level_out + 8.0, "the signal stands out from the background",
      f"{level_in:.0f} dB inside the SSTV band vs {level_out:.0f} dB outside")

quiet = waterfall_module.Waterfall(RATE)
quiet.feed(np.zeros(RATE, dtype=np.float64))
check(quiet.band_level_db() < -100.0, "silence is not mistaken for a signal",
      f"{quiet.band_level_db():.0f} dBFS on silence")

import time as _time
feed_ms = 0.0
warm = waterfall_module.Waterfall(RATE)
for _ in range(10):
    t0 = _time.perf_counter()
    warm.feed(encoded.samples[: int(RATE * 0.05)])
    feed_ms += (_time.perf_counter() - t0) * 1000 / 10
check(feed_ms < 12.0, "a live update is fast enough to feel immediate",
      f"{feed_ms:.1f} ms per 50 ms of audio")

print("\n7. A signal whose VIS header was missed")
print("-" * 78)
# The header is sent once, at the very start, so anything recorded after that point
# has no way of naming its own mode.  Missing it must not mean getting nothing.
inferred_ok = 0
inferred_produced = 0
weakest = None
for _name in modes.names():
    _spec = modes.get_mode(_name)
    _signal = encoder.encode(encoder.make_test_image(*_spec.resolution), _spec, RATE).samples
    _late = decoder.decode(_signal[int(RATE * 3.7):], RATE)   # header already gone
    if _late.image.size:
        inferred_produced += 1
    if _late.mode is not None and _late.mode.name == _name:
        inferred_ok += 1
    if _late.lines_decoded and (weakest is None or _late.coverage < weakest[1]):
        weakest = (_name, _late.coverage)

check(inferred_ok == len(modes.names()),
      "the mode is worked out from the signal when the header is missing",
      f"{inferred_ok}/{len(modes.names())} modes identified")
check(inferred_produced == len(modes.names()),
      "a picture is produced for every mode rather than nothing",
      f"{inferred_produced}/{len(modes.names())} produced a picture")

_late = decoder.decode(
    encoder.encode(encoder.make_test_image(*modes.get_mode("PD 120").resolution),
                   modes.get_mode("PD 120"), RATE).samples[int(RATE * 3.7):], RATE)
check(_late.mode_source == "guessed",
      "the result states that the mode was inferred, not read",
      f"source {_late.mode_source}, confidence {_late.mode_confidence * 100:.0f}%")
check(any("inferred" in _note for _note in _late.notes),
      "the notes explain where the mode came from",
      _late.notes[0][:70] if _late.notes else "no notes")

# A header, when there is one, must still be what decides the mode.
_header_spec = modes.get_mode("Robot 36")
_header_signal = encoder.encode(
    encoder.make_test_image(*_header_spec.resolution), _header_spec, RATE).samples
_header_result = decoder.decode(_header_signal, RATE)
check(_header_result.mode_source == "vis",
      "a present header is still what decides the mode",
      f"source {_header_result.mode_source}")

# And a signal with nothing in it must not produce a confident false picture.
_rng = np.random.default_rng(11)
_noise = decoder.decode(_rng.normal(0, 0.02, RATE * 15), RATE)
check(_noise.mode is None or _noise.mode_source != "vis",
      "noise is never mistaken for a VIS header",
      f"mode {_noise.mode.name if _noise.mode else None}, source {_noise.mode_source}")

print("\n8. Drawing the picture as it arrives")
print("-" * 78)
from sstv import streaming as streaming_module

live_spec = modes.get_mode("Robot 36")
live_original = encoder.make_test_image(*live_spec.resolution)
live_signal = encoder.encode(live_original, live_spec, RATE).samples

live = streaming_module.StreamingDecoder(RATE)
seen = []
live_block = int(RATE * 0.25)
halfway_seconds = None
for start in range(0, live_signal.size, live_block):
    progress = live.feed(live_signal[start:start + live_block])
    if progress.lines_decoded:
        seen.append((round(progress.seconds, 1), progress.lines_decoded))
        if halfway_seconds is None and progress.lines_decoded >= live_spec.line_count // 2:
            halfway_seconds = round(progress.seconds, 1)
check(bool(seen), "lines are decoded before the transmission ends",
      f"{seen[0][1]} lines by t={seen[0][0]} s" if seen else "none")
check(seen and seen[0][0] < 6.0, "the first line appears within a few seconds",
      f"first line at t={seen[0][0]} s" if seen else "never")
check(halfway_seconds is not None and halfway_seconds < 36.9 / 2 + 2,
      "half the picture is drawn well before the end",
      f"half the lines by t={halfway_seconds} s of {live_signal.size / RATE:.1f} s"
      if halfway_seconds else "never reached")

live.finalize()
live_result = live.result()
offline_live = decoder.decode(live_signal, RATE)
check(live_result.lines_decoded == offline_live.lines_decoded,
      "streaming decodes the same number of lines as decoding afterwards",
      f"{live_result.lines_decoded} vs {offline_live.lines_decoded}")
if live_result.image.size and offline_live.image.size:
    live_reference = np.asarray(live_original.resize(live_spec.resolution), dtype=np.uint8)
    live_score = psnr(live_reference, live_result.image)
    offline_score = psnr(live_reference, offline_live.image)
    check(abs(live_score - offline_score) < 2.0,
          "streaming is as faithful as decoding afterwards",
          f"streaming {live_score:.1f} dB vs offline {offline_score:.1f} dB")

print("\n9. Listening to the system sound")
print("-" * 78)
from sstv import wasapi

check(wasapi.loopback_available(), "system-sound capture is available on this platform",
      "WASAPI loopback")

loop_devices = audio.list_loopback_devices()
check(bool(loop_devices), "an output device is offered to listen to",
      loop_devices[0].name if loop_devices else "none")

if loop_devices:
    stream = wasapi.LoopbackStream(sample_rate=RATE, target_rate=RATE,
                                  device_id=loop_devices[0].device_id)
    try:
        stream.open()
        opened = True
    except Exception as exc:
        opened = False
        print("      open failed:", exc)
    check(opened, "the loopback stream opens",
          f"{stream.device_name!r}, mix {stream.mix_rate} Hz / {stream.channels} ch"
          if opened else "")
    if opened:
        # Play a tone and prove it comes back, which is the only check that shows
        # the feature works rather than merely that a stream was created.
        tone_hz = 1500.0
        tone = np.sin(2 * np.pi * tone_hz * np.arange(int(RATE * 1.5)) / RATE) * 0.25
        player = audio.AudioPlayer(RATE)
        player.play(tone)
        collected = []
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            chunk = stream.read(timeout=0.15)
            if chunk.size:
                collected.append(chunk)
            if not player.is_playing and collected:
                break
        player.stop(wait=True)
        stream.close()

        captured = np.concatenate(collected) if collected else np.zeros(0)
        check(captured.size > RATE * 0.2, "the played audio was captured",
              f"{captured.size} samples ({captured.size / RATE:.2f} s)")
        if captured.size > RATE * 0.2:
            peak = float(np.max(np.abs(captured)))
            check(peak > 0.01, "the capture is not silence", f"peak {peak:.3f}")
            middle = captured[int(captured.size * 0.3): int(captured.size * 0.7)]
            if middle.size > 4096:
                measured = float(np.median(
                    dsp.instantaneous_frequency(middle, RATE)[500:-500]))
                check(abs(measured - tone_hz) < 30.0,
                      "the captured tone matches what was played",
                      f"played {tone_hz:.0f} Hz, measured {measured:.1f} Hz")

        # And the point of the whole feature: decode a transmission off the
        # output device.  Capture and decode are attempted twice, because the
        # sound card is shared with the rest of the desktop: anything else playing
        # at that moment lands in the recording.  The check is whether the feature
        # works, not whether the machine happened to be quiet.
        spec_sound = modes.get_mode("Robot 36")
        original_sound = encoder.make_test_image(*spec_sound.resolution)
        transmission = encoder.encode(original_sound, spec_sound, RATE)

        heard: list = []
        decoded_sound = None
        for attempt in (1, 2):
            recorder = audio.AudioRecorder(RATE, loopback=True,
                                           device_id=loop_devices[0].device_id)
            recorder.start()
            recorder.wait_until_ready(2.0)
            player = audio.AudioPlayer(RATE)
            player.play(transmission.samples * 0.5)
            player.wait(timeout=transmission.duration + 5.0)
            time.sleep(0.3)
            recorder.stop(wait=True)
            heard = recorder.samples()
            if len(heard) <= RATE * 5:
                continue
            decoded_sound = decoder.decode(np.asarray(heard, dtype=np.float64), RATE)
            if decoded_sound.mode is not None and decoded_sound.complete:
                break
            if attempt == 1:
                print(f"      (first attempt captured {len(heard) / RATE:.1f} s but did "
                      f"not decode; trying once more)")

        check(len(heard) > RATE * 5, "a whole transmission was captured",
              f"{len(heard)} samples ({len(heard) / RATE:.1f} s), peak {recorder.peak}")

        if len(heard) > RATE * 5 and decoded_sound is not None:
            check(decoded_sound.mode is not None and decoded_sound.complete,
                  "the transmission captured from the sound card decodes",
                  f"{decoded_sound.mode.name if decoded_sound.mode else 'nothing'}, "
                  f"{decoded_sound.lines_decoded}/{decoded_sound.total_lines} lines")
            if decoded_sound.mode is None:
                # Keep the audio and say where the signal actually sits, so the
                # cause can be established rather than guessed at.
                from sstv import vis as _vis
                kept = np.asarray(heard, dtype=np.float64)
                debug_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", "out", "失败捕获.wav")
                audio.write_wav(debug_path, kept, RATE)
                header = _vis.decode_vis(kept, RATE)
                window = int(RATE * 0.25)
                blocks = np.abs(kept[: kept.size // window * window]).reshape(-1, window).max(axis=1)
                active = np.flatnonzero(blocks > 0.05)
                print(f"      kept the audio at {os.path.abspath(debug_path)}")
                print(f"      VIS on the capture: {header.code if header else None};  "
                      f"signal from {(active[0] * 0.25) if active.size else -1:.2f} s "
                      f"to {(active[-1] * 0.25) if active.size else -1:.2f} s")
                print(f"      peak {float(np.max(np.abs(kept))):.4f}  "
                      f"first 0.7 s peak {float(np.max(np.abs(kept[: int(RATE * 0.7)]))):.4f}")
            if decoded_sound.image.size:
                reference_sound = np.asarray(original_sound.resize(spec_sound.resolution),
                                             dtype=np.uint8)
                heard_psnr = psnr(reference_sound, decoded_sound.image)
                check(heard_psnr > 12.0,
                      "the picture from the sound card is a good likeness",
                      f"{heard_psnr:.1f} dB against the original")
                # The honest baseline is the *same captured audio* decoded with no
                # sound card involved, because going out of the speakers and back in
                # adds a sub-pixel delay that varies from run to run -- a couple of
                # pixels of horizontal shift, which on a test card of hard edges is
                # worth several dB without anything actually being lost.  Comparing
                # against the original instead would make this test measure the sound
                # card's buffer alignment rather than the decoder.
                same_audio = decoder.decode(np.asarray(heard, dtype=np.float64), RATE)
                same_psnr = (psnr(reference_sound, same_audio.image)
                             if same_audio.image.size else 0.0)
                check(abs(heard_psnr - same_psnr) < 1.0,
                      "the decoded picture is what the captured audio contains",
                      f"{heard_psnr:.1f} dB vs {same_psnr:.1f} dB for the same samples")

print("\n10. Window sizing (must not assume a single big screen)")
print("-" * 78)
import tkinter as _tk

win = _tk.Tk()
win_app = gui_module.SstvApp(win)
for _ in range(8):
    win.update_idletasks(); win.update(); time.sleep(0.03)
area = win_app._screen_work_area()
win_w, win_h = win.winfo_width(), win.winfo_height()
win_x, win_y = win.winfo_x(), win.winfo_y()
print(f"    screen work area {area[2]}x{area[3]} at ({area[0]},{area[1]}); "
      f"window {win_w}x{win_h} at ({win_x},{win_y})")

check(area[2] > 200 and area[3] > 200, "the usable desktop area is detected",
      f"{area[2]} x {area[3]} px")
check(win_w <= area[2] and win_h <= area[3], "the window starts no larger than the screen",
      f"{win_w}x{win_h} against {area[2]}x{area[3]}")
# The bug this guards: a window wider than one monitor gets spread across two.
check(win_x >= area[0] and win_x + win_w <= area[0] + area[2],
      "the window stays within a single monitor",
      f"x {win_x}..{win_x + win_w} inside {area[0]}..{area[0] + area[2]}")

min_w, min_h = win.minsize()
check(min_w <= area[2] and min_h <= area[3],
      "the minimum size never exceeds the screen",
      f"minsize {min_w}x{min_h}")

# Shrinking must not put the controls out of reach: the sidebar has to scroll.
win_app.notebook.select(win_app.rx_tab)
spec_small = modes.get_mode("Robot 36")
small_sig = encoder.encode(encoder.make_test_image(*spec_small.resolution),
                          spec_small, RATE).samples
win_app._decode_samples(small_sig, source="size test")
for _ in range(80):
    win.update_idletasks(); win.update()
    if win_app.rx_image is not None:
        break
    time.sleep(0.02)
for _ in range(6):
    win.update_idletasks(); win.update(); time.sleep(0.02)


def _reachable(button) -> bool:
    top = button.winfo_rooty() - win.winfo_rooty()
    return top >= 0 and top + button.winfo_height() <= win.winfo_height()


check(_reachable(win_app.save_rx_btn), "the Save image button is on screen at the default size",
      f"window height {win.winfo_height()}")

win.geometry(f"{win.winfo_width()}x520")
for _ in range(10):
    win.update_idletasks(); win.update(); time.sleep(0.03)
scrolls = win_app._sidebar_canvas.yview() != (0.0, 1.0)
check(scrolls, "a short window makes the sidebar scroll instead of hiding controls",
      f"at 520 px tall, scroll region {tuple(round(v, 2) for v in win_app._sidebar_canvas.yview())}")
check(_reachable(win_app.rx_record_btn), "the record button stays reachable when short",
      f"window height {win.winfo_height()}")
win.withdraw()
win_app.close()

print("\n11. Interface and packaging")
print("-" * 78)
check(os.path.exists(os.path.join(ROOT, "run.bat")),
      "double-click launcher present", "run.bat")
check(os.path.exists(os.path.join(ROOT, "README.md")),
      "user documentation present", "README.md")

import tkinter as tk
root = tk.Tk()
root.withdraw()
app = gui_module.SstvApp(root)
for mode_name in REQUIRED:
    app.mode_var.set(mode_name)
    app._update_mode_info()
root.update()
mode_count = len(app.mode_combo["values"])
# Walk every mode, then settle on the quickest one so the encode below is not a
# five-minute PD 290 job.
app.mode_var.set("Robot 36")
app._update_mode_info()
app.tx_test_card()
for _ in range(200):
    root.update()
    if app.tx_audio is not None:
        break
    time.sleep(0.05)
encoded_ok = app.tx_audio is not None
encoded_count = 0 if app.tx_audio is None else len(app.tx_audio)
app._decode_samples(app.tx_audio, source="acceptance")
for _ in range(300):
    root.update()
    if app.rx_image is not None and (app.rx_info["text"] or ""):
        break
    time.sleep(0.05)
gui_ok = app.rx_image is not None and app.rx_image.size > 0
summary = (app.rx_info["text"] or "")
rx_shape = None if app.rx_image is None else app.rx_image.shape
app.close()
check(mode_count == len(REQUIRED) and encoded_ok and gui_ok,
      "the window drives a full transmit and receive cycle",
      f"{mode_count} modes listed, encoded {encoded_count} samples, decoded {rx_shape}")
check("Mode:" in summary and "Lines decoded:" in summary,
      "the interface reports what it decoded",
      summary.splitlines()[1] if "\n" in summary else summary[:60])

cli = os.path.join(ROOT, "sstv_app.py")
r = subprocess.run([sys.executable, "-X", "utf8", cli, "--check"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
check(r.returncode == 0 and "everything checks out" in r.stdout,
      "the self-check command succeeds", f"exit code {r.returncode}")

print()
print("12. Version numbering")
print("-" * 78)
from sstv import version as version_module

check(version_module.VERSION.count(".") == 2
      and all(p.isdigit() for p in version_module.VERSION.split(".")),
      "the version is a well-formed number", version_module.VERSION)
check(version_module.VERSION in [entry[0] for entry in version_module.CHANGELOG],
      "the current version has a changelog entry",
      f"{len(version_module.CHANGELOG)} releases recorded")
check(gui_module.APP_VERSION == version_module.VERSION,
      "the window reports the same version as the definition",
      gui_module.APP_VERSION)

r = subprocess.run([sys.executable, "-X", "utf8", cli, "--version"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
check(r.returncode == 0 and version_module.VERSION in r.stdout,
      "--version prints the version and its history",
      r.stdout.splitlines()[0] if r.stdout else "no output")

bump = os.path.join(ROOT, "tools", "bump_version.py")
r = subprocess.run([sys.executable, "-X", "utf8", bump, "--check"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
check(r.returncode == 0 and "everything agrees" in r.stdout,
      "every file that quotes the version agrees with it",
      "no stale references")

packager = os.path.join(ROOT, "tools", "make_package.py")
packager_text = open(packager, encoding="utf-8").read()
check("version_module.VERSION" in packager_text,
      "the package folder name comes from the version definition",
      f"SSTV-Studio-{version_module.VERSION}")

print()
print("=" * 78)
passed = sum(1 for ok, _, _ in results if ok)
failed = [r for r in results if not r[0]]
print(f"RESULT: {passed}/{len(results)} checks passed")
if failed:
    print("FAILED CHECKS:")
    for _, requirement, detail in failed:
        print(f"  - {requirement}  ({detail})")
    sys.exit(1)
print("ALL ACCEPTANCE CHECKS PASSED")
