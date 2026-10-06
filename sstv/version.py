"""The program's version, and what changed in each release.

One definition, used everywhere: the window title, ``--version``, ``--check``,
both READMEs and the package folder name all read it from here, so a release
cannot end up half-labelled.

Bumping the version
-------------------
Run ``python tools/bump_version.py`` to move to the next version.  It updates the
numbers and the release date below, then rewrites the places that quote the
version: both READMEs, the packaging script and the launcher banner.  Add the new
entry to :data:`CHANGELOG` at the same time -- the command prints the lines to
paste in.
"""

from __future__ import annotations

__all__ = ["VERSION", "VERSION_INFO", "RELEASED", "CHANGELOG", "version_string"]

VERSION = "1.2.0"
"""The version, as ``MAJOR.MINOR.PATCH``.

* MAJOR -- something that changes how existing recordings or settings behave.
* MINOR -- a new capability, such as a new way to receive or a new mode.
* PATCH -- a fix or an internal change with no new capability.
"""

VERSION_INFO = tuple(int(part) for part in VERSION.split("."))
"""``VERSION`` as a comparable tuple, for code that needs to sort or compare."""

RELEASED = "2026-02-14"
"""Date of this release, ISO format."""

CHANGELOG: list[tuple[str, str, list[str]]] = [
    (
        "1.2.0",
        "2026-02-14",
        [
            "Work out the mode from the signal when the VIS header is missing, so a "
            "transmission that was already in progress when listening started is "
            "still decoded instead of producing nothing. The result says the mode "
            "was inferred and how confident the inference is.",
            "Ignore a VIS header that does not look like one. A picture's own "
            "brightness changes could be read as a header, and a header naming a "
            "mode that does not exist made decoding give up entirely.",
            "Stop 'correcting' the tuning from a headerless signal's sync pulse. The "
            "pulse reads tens of hertz high depending on where in it one looks, and "
            "the correction shifted every scan sideways until the picture was lost.",
        ],
    ),
    (
        "1.1.0",
        "2026-02-13",
        [
            "Receive from the system sound, so a receiver feeding the speakers "
            "or a signal playing in another program can be decoded without any cable.",
            "Draw the picture line by line as the transmission arrives instead of "
            "only after recording stops: the first lines now appear about a second in, "
            "and stopping no longer starts a long decode.",
            "Keep the waterfall history when a decode completes, instead of clearing it.",
            "Use each monitor's own work area for sizing, so the window no longer "
            "spreads across two screens, and let the controls scroll when the window "
            "is short so the buttons at the bottom stay reachable.",
        ],
    ),
    (
        "1.0.0",
        "2026-02-12",
        [
            "Encode and decode all 14 mainstream SSTV modes.",
            "Simplex operation: transmit and receive are separate actions.",
            "Receive from a microphone or an audio file; transmit to the speakers "
            "or an audio file.",
            "Real-time waterfall while recording, with a frequency scale and per-mode "
            "history lengths.",
            "Decode without being told the mode, by reading the VIS header.",
        ],
    ),
]


def version_string(with_date: bool = False) -> str:
    """``1.1.0``, or ``1.1.0 (2026-02-13)`` when *with_date* is set."""
    return f"{VERSION} ({RELEASED})" if with_date else VERSION
