"""Move the program to its next version, in one command.

The version lives in exactly one place -- ``sstv/version.py`` -- and this rewrites
the handful of files that quote it, so a release cannot end up half-labelled.  It
also prints the changelog skeleton to paste in, because only a person knows what
actually changed.

Usage:
    python tools/bump_version.py patch        # 1.1.0 -> 1.1.1
    python tools/bump_version.py minor        # 1.1.0 -> 1.2.0
    python tools/bump_version.py major        # 1.1.0 -> 2.0.0
    python tools/bump_version.py 1.4.2        # set an exact version
    python tools/bump_version.py --check      # report any file that disagrees
"""

from __future__ import annotations

import datetime
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sstv import version as version_module  # noqa: E402

VERSION_FILE = os.path.join(ROOT, "sstv", "version.py")

# Files that quote the version, and how to recognise the quote.  Each pattern
# captures the version in group 1 and, where the file also shows a release date,
# that date in group 2 -- a README that says "1.1.1 (released 2026-02-13)" would
# otherwise go out pairing a new version with an old date.
READS_DEFINITION = "reads-definition"
REFERENCES = [
    ("README.md", r"SSTV Studio (\d+\.\d+\.\d+)\*\* \(released (\d{4}-\d{2}-\d{2})\)"),
    ("README-zh.md", r"SSTV Studio (\d+\.\d+\.\d+)\*\*（(\d{4}-\d{2}-\d{2}) 发布）"),
    ("run.bat", r"SSTV Studio (\d+\.\d+\.\d+)"),
    ("run-zh.bat", r"SSTV Studio (\d+\.\d+\.\d+)"),
    ("tools/make_package.py", READS_DEFINITION),
]


def _bump_text(text: str, pattern: str, new: str, today: str) -> tuple[str, int]:
    """Replace the version (and date, if present) in *text*.  Returns the count."""
    if pattern == READS_DEFINITION:
        return text, 0

    def replace(match: re.Match) -> str:
        out = match.group(0)
        # Longest first, so replacing the old version cannot corrupt the old date.
        out = out.replace(match.group(1), new)
        if match.lastindex and match.lastindex >= 2 and match.group(2):
            out = out.replace(match.group(2), today)
        return out

    return re.subn(pattern, replace, text)


def next_version(current: str, instruction: str) -> str:
    """Work out the version that *instruction* asks for."""
    major, minor, patch = (int(p) for p in current.split("."))
    if instruction == "major":
        return f"{major + 1}.0.0"
    if instruction == "minor":
        return f"{major}.{minor + 1}.0"
    if instruction == "patch":
        return f"{major}.{minor}.{patch + 1}"
    if re.fullmatch(r"\d+\.\d+\.\d+", instruction):
        return instruction
    raise SystemExit(
        f"not a version or an increment: {instruction!r}\n"
        "use patch, minor, major, or an exact version such as 1.4.2")


def rewrite_version_file(new: str, today: str) -> None:
    text = open(VERSION_FILE, encoding="utf-8").read()
    text = re.sub(r'^VERSION = "[^"]+"', f'VERSION = "{new}"', text, count=1, flags=re.M)
    text = re.sub(r'^RELEASED = "[^"]+"', f'RELEASED = "{today}"', text, count=1, flags=re.M)
    open(VERSION_FILE, "w", encoding="utf-8").write(text)


def rewrite_references(new: str, today: str) -> list[tuple[str, int]]:
    """Update every file that quotes the version.  Returns what changed."""
    changed: list[tuple[str, int]] = []
    for relative, pattern in REFERENCES:
        path = os.path.join(ROOT, relative.replace("/", os.sep))
        if not os.path.exists(path):
            continue
        text = open(path, encoding="utf-8", errors="replace").read()
        updated, count = _bump_text(text, pattern, new, today)
        if count and updated != text:
            open(path, "w", encoding="utf-8").write(updated)
            changed.append((relative, count))
    return changed


def check() -> int:
    """Report every place whose version disagrees with the definition."""
    current = version_module.VERSION
    released = version_module.RELEASED
    print(f"defined version: {current}  (released {released})")
    problems = 0
    for relative, pattern in REFERENCES:
        path = os.path.join(ROOT, relative.replace("/", os.sep))
        if not os.path.exists(path):
            continue
        text = open(path, encoding="utf-8", errors="replace").read()
        if pattern == READS_DEFINITION:
            ok = "version_module" in text
            print(f"  {'ok  ' if ok else 'STALE'} {relative}: reads the definition")
            problems += 0 if ok else 1
            continue
        found = re.findall(pattern, text)
        if not found:
            print(f"  --   {relative}: does not quote a version")
            continue
        stale = 0
        for match in found:
            version = match[0] if isinstance(match, tuple) else match
            date = match[1] if isinstance(match, tuple) and len(match) > 1 else None
            if version != current:
                stale += 1
            if date and date != released:
                stale += 1
        shown = ", ".join(
            f"{m[0]}" + (f" ({m[1]})" if isinstance(m, tuple) and len(m) > 1 else "")
            for m in found)
        print(f"  {'ok  ' if not stale else 'STALE'} {relative}: {shown}")
        problems += stale
    # The changelog must know about the version being built.
    known = [entry[0] for entry in version_module.CHANGELOG]
    if current not in known:
        print(f"  STALE sstv/version.py: {current} is not in CHANGELOG")
        problems += 1
    else:
        print(f"  ok   sstv/version.py: {current} has a changelog entry")
    print()
    print("everything agrees" if not problems else f"{problems} stale reference(s)")
    return 1 if problems else 0
    # The changelog must know about the version being built.
    known = [entry[0] for entry in version_module.CHANGELOG]
    if current not in known:
        print(f"  STALE sstv/version.py: {current} is not in CHANGELOG")
        problems += 1
    else:
        print(f"  ok   sstv/version.py: {current} has a changelog entry")
    print()
    print("everything agrees" if not problems else f"{problems} stale reference(s)")
    return 1 if problems else 0


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--check":
        return check()

    current = version_module.VERSION
    new = next_version(current, argv[0])
    if new == current:
        print(f"already at {current}; nothing to do")
        return 0

    today = datetime.date.today().isoformat()
    rewrite_version_file(new, today)
    changed = rewrite_references(new, today)

    print(f"{current} -> {new}  (released {today})")
    for relative, count in changed:
        print(f"  updated {relative} ({count} reference(s))")
    if not changed:
        print("  no file quoted the version directly")
    print()
    print("Now add the changelog entry at the top of CHANGELOG in sstv/version.py:")
    print()
    print(f'    (')
    print(f'        "{new}",')
    print(f'        "{today}",')
    print(f'        [')
    print(f'            "what changed, in one sentence",')
    print(f'        ],')
    print(f'    ),')
    print()
    print("Then run:  python tools/bump_version.py --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
