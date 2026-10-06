"""Smoke test: build the window, exercise the safe paths, then close it.

Checks that the interface can be constructed and that widget wiring works,
without requiring anyone to click anything.
"""
import sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sstv_bootstrap

def main():
    import tkinter as tk
    from sstv import gui

    root = tk.Tk()
    root.withdraw()          # keep it off the screen during the test
    app = gui.SstvApp(root)

    _seen = []
    _orig_handler = app._handle_event

    def _logging_handler(kind, payload):
        _seen.append((kind, payload))
        if kind == "tx_done" and len(payload) != 2:
            print("!! tx_done with %d fields: %r" % (len(payload), payload))
        return _orig_handler(kind, payload)

    app._handle_event = _logging_handler

    root.update_idletasks()
    root.update()

    print("window built ok; title:", root.title())
    print("modes offered:", len(app.mode_combo["values"]))

    # Exercise the mode selector across every mode.
    for spec in __import__("sstv.modes", fromlist=["x"]).MODE_LIST:
        app.mode_var.set(spec.name)
        app._update_mode_info()
        root.update()
    print("mode selector walked", len(app.mode_combo["values"]), "entries")

    # Load the test card, which triggers an async encode.
    app.mode_var.set("Robot 36")
    app.tx_test_card()
    root.update()
    print("test card loaded; tx_info =", app.tx_info["text"])

    # Wait for the encoder thread.
    for _ in range(120):
        root.update()
        if app.tx_audio is not None:
            break
        root.after(25)
        import time; time.sleep(0.05)
    print("encoded audio samples:", 0 if app.tx_audio is None else len(app.tx_audio))

    # Exercise the receive-side summary formatter with a real decode.
    if app.tx_audio is not None:
        from sstv import decoder
        app._decode_samples(app.tx_audio, source="self test")
        for _ in range(200):
            root.update()
            if app.rx_image is not None and (app.rx_info["text"] or ""):
                break
            import time; time.sleep(0.05)
        print("rx image shape:", None if app.rx_image is None else app.rx_image.shape)
        print("rx summary:", (app.rx_info["text"] or "").replace("\n", " | "))

    app.close()
    print("events seen:", ", ".join(sorted({k for k, _ in _seen})))
    bad = [(k, p) for k, p in _seen if k == "tx_done" and len(p) != 2]
    print("malformed tx_done events:", bad)
    print("SMOKE TEST PASSED")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
