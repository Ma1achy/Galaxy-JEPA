"""Check the code of record against the SHA-1 table in docs/patch_grid/prereg.md §3, and the shipped cache against its verified manifest.
The launcher, the embedding program and the scoring programs call these before real work.

    python artifacts/patch_grid_pins.py            # check the pre-registration's own hash and every pinned file
    python artifacts/patch_grid_pins.py hash       # the pre-registration section's SHA-1
    ls files | python artifacts/patch_grid_pins.py rows   # table rows for the files on stdin
"""
from __future__ import annotations

import hashlib
import json
import os
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


def section_sha1(text: str | None = None) -> str:
    """SHA-1 of the pre-registration: "## Pre-registration" to the next level-2 heading, trailing newline (as docs/axes/prereg_a_bis.md)."""
    L = (text if text is not None else (REPO / "docs/patch_grid/prereg.md").read_text()).splitlines()
    a = L.index("## Pre-registration")
    b = next((k for k in range(a + 1, len(L)) if L[k].startswith("## ")), len(L))
    return hashlib.sha1(("\n".join(L[a:b]).rstrip() + "\n").encode()).hexdigest()


def verify() -> None:
    f = REPO / "docs/patch_grid/prereg.sha1"
    if not f.exists() or f.read_text().strip() != section_sha1():
        raise SystemExit("patch grid: the pre-registration is not hashed, or has changed since (docs/patch_grid/prereg.sha1)")
    bad = mismatches()
    if bad:
        raise SystemExit("patch grid: the code of record differs from the pre-registration:\n  " + "\n  ".join(bad))


def verify_cache(cache_dir: Path, pipeline_hash: str, manifest_sha1: str | None = None) -> None:
    """The shipping plan's gate: `cache_verified.json` (written by the pull, after every raw SHA-1 passed) must name the manifest whose SHA-1 the
    user passed by hand (GJ_CACHE_MANIFEST_SHA1, the trust anchor printed on the Mac) and the pipeline hash the run resolves to."""
    want = manifest_sha1 or os.environ.get("GJ_CACHE_MANIFEST_SHA1")
    if not want:
        raise SystemExit("patch grid: set GJ_CACHE_MANIFEST_SHA1 (printed on the Mac at upload) before a real run")
    f = Path(cache_dir) / "cache_verified.json"
    if not f.exists():
        raise SystemExit(f"patch grid: {f} does not exist: the cache has not passed the manifest check")
    v = json.loads(f.read_text())
    if v.get("manifest_sha1") != want or v.get("pipeline_hash") != pipeline_hash:
        raise SystemExit(f"patch grid: the verified cache is {str(v.get('manifest_sha1'))[:12]} / {str(v.get('pipeline_hash'))[:12]}, expected {want[:12]} / {pipeline_hash[:12]}")


if __name__ == "__main__":
    if sys.argv[1:] == ["rows"]:  # print the table rows for the files listed in PINNED
        for f in (l.strip() for l in sys.stdin if l.strip()):
            print(f"  | `{f}` | `{hashlib.sha1((REPO / f).read_bytes()).hexdigest()}` |")
    elif sys.argv[1:] == ["hash"]:
        print(section_sha1())
    else:
        verify()
        print(f"{len(table())} files match")
