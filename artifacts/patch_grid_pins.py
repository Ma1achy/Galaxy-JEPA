"""Check the code of record against the SHA-1 table in docs/patch_grid/prereg.md §3 (the launcher calls this before a real run)."""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROW = re.compile(r"^\s*\| `([^`]+)`[^|]*\| `([0-9a-f]{40})` \|", re.M)


def table(text: str | None = None) -> dict[str, str]:
    t = text if text is not None else (REPO / "docs/patch_grid/prereg.md").read_text()
    return dict(ROW.findall(t))


def mismatches(text: str | None = None, root: Path = REPO) -> list[str]:
    pins = table(text)
    if not pins:
        return ["no pinned files found in docs/patch_grid/prereg.md"]
    out = []
    for f, want in pins.items():
        p = root / f
        got = hashlib.sha1(p.read_bytes()).hexdigest() if p.exists() else "missing"
        if got != want:
            out.append(f"{f}: {got[:12]} is not the pinned {want[:12]}")
    return out


def verify() -> None:
    bad = mismatches()
    if bad:
        raise SystemExit("patch grid: the code of record differs from the pre-registration:\n  " + "\n  ".join(bad))


if __name__ == "__main__":
    if sys.argv[1:] == ["rows"]:  # print the table rows for the files listed in PINNED
        for f in (l.strip() for l in sys.stdin if l.strip()):
            print(f"  | `{f}` | `{hashlib.sha1((REPO / f).read_bytes()).hexdigest()}` |")
    else:
        verify()
        print(f"{len(table())} files match")
