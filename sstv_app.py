"""Launch SSTV Studio.

Usage:
    python sstv_app.py                 # open the window
    python sstv_app.py --version       # print the version and what changed
    python sstv_app.py --check         # verify the install without a window
    python sstv_app.py --decode FILE   # decode an audio file from the command line
    python sstv_app.py --encode IMG --mode "Robot 36" --out out.wav
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sstv_bootstrap  # noqa: F401  (puts vendor/ on sys.path)

from sstv import audio as audio_module
from sstv import decoder as decoder_module
from sstv import encoder as encoder_module
from sstv import modes as modes_module
from sstv import version as version_module


def cmd_version() -> int:
    """Print the version and the history behind it."""
    print(f"SSTV Studio {version_module.version_string(with_date=True)}")
    print()
    for number, released, changes in version_module.CHANGELOG:
        marker = "  <- this build" if number == version_module.VERSION else ""
        print(f"{number}  ({released}){marker}")
        for change in changes:
            print(f"    - {change}")
        print()
    return 0


def cmd_check() -> int:
    """Report what is and is not working, without opening a window."""
    print(f"SSTV Studio {version_module.version_string(with_date=True)} - self check")
    print("=" * 60)
    problems = 0

    try:
        import numpy
        print(f"  numpy      {numpy.__version__}")
    except Exception as exc:
        print(f"  numpy      MISSING ({exc})")
        problems += 1
    try:
        import PIL
        print(f"  Pillow     {PIL.__version__}")
    except Exception as exc:
        print(f"  Pillow     MISSING ({exc})")
        problems += 1
    try:
        import tkinter
        print(f"  tkinter    Tk {tkinter.TkVersion}")
    except Exception as exc:
        print(f"  tkinter    MISSING ({exc})")
        problems += 1

    print()
    print(f"  modes      {len(modes_module.MODE_LIST)} supported")
    for spec in modes_module.MODE_LIST:
        minutes, seconds = divmod(spec.duration_seconds + modes_module.VIS_HEADER_MS / 1000.0, 60)
        print(f"     {spec.name:<12} {spec.image_width:>3} x {spec.image_height:<3}"
              f"  {spec.line_milliseconds:>8.3f} ms/line  "
              f"{int(minutes)}m{seconds:04.1f}s  VIS {spec.vis_code}")

    print()
    print("  audio output devices:")
    try:
        for device in audio_module.list_output_devices():
            mark = " " if device.available else "!"
            print(f"    {mark} [{device.index:>2}] {device.name}")
    except Exception as exc:
        print(f"    unavailable: {exc}")
        problems += 1
    print("  audio input devices:")
    try:
        for device in audio_module.list_input_devices():
            mark = " " if device.available else "!"
            print(f"    {mark} [{device.index:>2}] {device.name}")
    except Exception as exc:
        print(f"    unavailable: {exc}")
        problems += 1

    print()
    print("  round-trip test (encode then decode every mode):")
    import numpy as np
    from PIL import Image

    failures = 0
    for spec in modes_module.MODE_LIST:
        width, height = spec.resolution
        image = encoder_module.make_test_image(width, height)
        encoded = encoder_module.encode(image, spec, 48000)
        decoded = decoder_module.decode(encoded.samples, 48000)
        wanted = np.asarray(image, dtype=np.float64)
        if decoded.mode is None or decoded.image.size == 0:
            print(f"    {spec.name:<12} FAILED to decode")
            failures += 1
            continue
        got = decoded.image.astype(np.float64)
        mse = float(np.mean((got - wanted) ** 2))
        psnr = 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
        flag = "ok" if decoded.complete else "partial"
        print(f"    {spec.name:<12} {decoded.mode.name:<12} "
              f"{decoded.lines_decoded:>3}/{decoded.total_lines:<3} lines  "
              f"PSNR {psnr:5.1f} dB  {flag}")
        if not decoded.complete:
            failures += 1
    problems += failures

    print()
    print("  waterfall display:")
    try:
        from sstv import waterfall as waterfall_module

        probe = waterfall_module.Waterfall(48000)
        probe.feed(np.sin(2 * np.pi * 1500.0 * np.arange(48000) / 48000) * 0.6)
        silence = waterfall_module.Waterfall(48000)
        silence.feed(np.zeros(48000, dtype=np.float64))
        placed = abs(probe.peak_hz() - 1500.0) < 40
        quiet = silence.band_level_db() < -100.0
        if placed and quiet:
            print(f"    ok        1500 Hz tone placed at {probe.peak_hz():.0f} Hz, "
                  f"silence reads {silence.band_level_db():.0f} dBFS")
            print(f"    ok        {probe.frames} spectrum rows from 1 s of audio")
        else:
            print("    FAILED    tone placement or silence detection is wrong")
            problems += 1
    except Exception as exc:
        print(f"    FAILED    {exc}")
        problems += 1

    print()
    print("  system sound (listening to what is playing):")
    try:
        from sstv import wasapi

        if not wasapi.loopback_available():
            print("    n/a       not available on this platform")
        else:
            listenable = audio_module.list_loopback_devices()
            if not listenable:
                print("    none      no output device could be opened for capture")
            else:
                for device in listenable:
                    print(f"    ok        {device.name}")
                stream = wasapi.LoopbackStream(48000, target_rate=48000,
                                              device_id=listenable[0].device_id)
                try:
                    stream.open()
                    print(f"    ok        capture stream opens "
                          f"({stream.mix_rate} Hz, {stream.channels} ch)")
                finally:
                    stream.close()
    except Exception as exc:
        print(f"    FAILED    {exc}")
        problems += 1

    print()
    if problems:
        print(f"RESULT: {problems} problem(s) found")
        return 1
    print("RESULT: everything checks out")
    return 0


def cmd_decode(path: str) -> int:
    from PIL import Image

    result = decoder_module.decode_file(path)
    if result.mode is None:
        print("No SSTV signal was identified in that file.")
        return 2
    print(f"mode      {result.mode.name} (VIS {result.vis_code}, "
          f"confidence {result.vis_confidence * 100:.0f}%)")
    print(f"lines     {result.lines_decoded}/{result.total_lines}")
    if result.frequency_offset_hz:
        print(f"offset    {result.frequency_offset_hz:+.1f} Hz")
    for note in result.notes:
        print(f"note      {note}")
    out = os.path.splitext(path)[0] + "-decoded.png"
    Image.fromarray(result.image).save(out)
    print(f"wrote     {out}")
    return 0


def cmd_encode(path: str, mode: str, out: str) -> int:
    from PIL import Image

    image = Image.open(path)
    image.load()
    result = encoder_module.encode_to_wav(image, mode, out)
    print(f"mode      {result.mode.name}")
    print(f"duration  {result.duration:.2f} s")
    print(f"wrote     {out}")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sstv_app",
        description="SSTV Studio - encode and decode slow-scan television.",
    )
    parser.add_argument("--version", action="store_true",
                        help="print the version and what changed in each release")
    parser.add_argument("--check", action="store_true",
                        help="verify the installation and run a round-trip test")
    parser.add_argument("--decode", metavar="AUDIO",
                        help="decode a WAV file and write a PNG next to it")
    parser.add_argument("--encode", metavar="IMAGE",
                        help="encode an image to a WAV file")
    parser.add_argument("--mode", default="Robot 36",
                        help="mode to use with --encode (default: Robot 36)")
    parser.add_argument("--out", default=None,
                        help="output WAV path for --encode")
    parser.add_argument("--sample-rate", type=int,
                        default=audio_module.DEFAULT_SAMPLE_RATE,
                        help="audio sample rate in Hz (default 48000)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.version:
        return cmd_version()
    if args.check:
        return cmd_check()
    if args.decode:
        return cmd_decode(args.decode)
    if args.encode:
        out = args.out or os.path.splitext(args.encode)[0] + ".wav"
        return cmd_encode(args.encode, args.mode, out)
    return run_gui(args.sample_rate)


def run_gui(sample_rate: int) -> int:
    try:
        from sstv.gui import run
    except Exception as exc:
        print(f"Could not load the graphical interface: {exc}", file=sys.stderr)
        print("Run 'python sstv_app.py --check' to see what is missing.", file=sys.stderr)
        return 1
    try:
        return run(sample_rate)
    except Exception as exc:
        print(f"The window could not be opened: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
