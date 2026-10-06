"""SSTV Studio - encode and decode slow-scan television on Windows.

A pure-Python, dependency-light SSTV package:

* :mod:`sstv.modes`    - the transmission-mode parameter tables
* :mod:`sstv.encoder`  - image -> audio
* :mod:`sstv.decoder`  - audio -> image
* :mod:`sstv.audio`    - WAV files, speaker playback and microphone capture
* :mod:`sstv.gui`      - the single-window desktop application
"""

__version__ = "1.0.0"
__all__ = ["audio", "decoder", "encoder", "modes"]
