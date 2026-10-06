"""Bootstrap: make the vendored dependencies importable.

The build/run environment for this project cannot use ``pip`` (its temporary
directory provisioning is blocked), so third-party wheels are downloaded and
unpacked into ``vendor/`` next to this file.  Importing this module first makes
that directory visible on ``sys.path``.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(_ROOT, "vendor")


def ensure_vendor_on_path() -> str | None:
    """Prepend ``vendor/`` to ``sys.path`` if it exists."""
    if os.path.isdir(_VENDOR) and _VENDOR not in sys.path:
        sys.path.insert(0, _VENDOR)
    return _VENDOR if os.path.isdir(_VENDOR) else None


ensure_vendor_on_path()
