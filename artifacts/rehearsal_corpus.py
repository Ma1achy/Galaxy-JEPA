"""v2 rehearsal — a 20,000-stamp scratch corpus cut from the aligned re-pull, for a 500-step run.

The rehearsal exists to walk the whole v2 chain (E5 freeze -> bake -> scalars -> F0 -> train ->
stop -> resume) at a size that costs minutes, before the 827k corpus commits days. So the corpus
is a *view*, not a copy: ``metadata.csv`` holds the v1 rows (byte-for-byte lines — the v2 pull
writes no metadata of its own; ``petroRad_r`` and the photometry are per-object, not per-cut) for
the first N ids the v2 cutter logged, and each ``<id>.fits`` is a symlink into ``pretrain_v2``.
Nothing is written inside ``pretrain_v2`` — a live download writes there.

``cut_log.csv`` is read in order and only its head is used; its tail may be mid-append.

Investigation code: terse, excluded from lint/CI.

    uv run python artifacts/rehearsal_corpus.py [N]    # default 20,000; refuses if it exists
"""

from __future__ import annotations

import csv
import io
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NAME = os.environ.get("REHEARSAL_NAME", "_v2_rehearsal")  # a second view (e.g. the E5 fit's) sits beside it
if not NAME.startswith("_v2_rehearsal"):
    raise SystemExit(f"refusing: REHEARSAL_NAME must start with _v2_rehearsal, got {NAME!r}")
LINK = REPO / "data" / NAME
TARGET = Path("/Volumes/X10 Pro/galaxy-jepa/raw") / NAME
V2 = REPO / "data" / "pretrain_v2"
V1_META = REPO / "data" / "pretrain" / "metadata.csv"
FITS_BLOCK = 2880


def head_ids(n: int) -> list[str]:
    """The first n distinct object_ids in cut order, as strings (never through a float)."""
    out: list[str] = []
    seen: set[str] = set()
    with (V2 / "cut_log.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            oid = r["object_id"]
            if oid not in seen:
                seen.add(oid)
                out.append(oid)
                if len(out) == n:
                    return out
    raise SystemExit(f"cut_log.csv has only {len(out)} distinct ids, asked for {n}")


def v1_lines(want: set[str]) -> tuple[bytes, dict[str, bytes]]:
    """The header and each wanted row, as the exact bytes on disk. Parsed with csv only to
    read the id, and refused if a line is not one whole record (a quoted newline would split it)."""
    rows: dict[str, bytes] = {}
    with V1_META.open("rb") as fh:
        header = fh.readline()
        cols = next(csv.reader(io.StringIO(header.decode())))
        k = cols.index("object_id")
        for line in fh:
            rec = next(csv.reader(io.StringIO(line.decode())))
            if len(rec) != len(cols):
                raise SystemExit(f"metadata line is not one record ({len(rec)} fields): {line[:80]!r}")
            if rec[k] in want:
                if rec[k] in rows:
                    raise SystemExit(f"object_id {rec[k]} appears twice in {V1_META}")
                rows[rec[k]] = line
    return header, rows


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
    if os.path.lexists(LINK) or os.path.lexists(TARGET):
        raise SystemExit(f"refusing: {LINK} or {TARGET} already exists — delete both to rebuild")

    ids = head_ids(n)
    for oid in ids:
        f = V2 / f"{oid}.fits"
        size = f.stat().st_size if f.exists() else -1
        if size <= 0 or size % FITS_BLOCK:
            raise SystemExit(f"{f} is missing or not whole FITS blocks ({size} bytes)")
    header, rows = v1_lines(set(ids))
    missing = [o for o in ids if o not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} cut ids have no v1 metadata row, e.g. {missing[:3]}")

    TARGET.mkdir(parents=True)
    with (TARGET / "metadata.csv").open("wb") as fh:
        fh.write(header)
        for oid in ids:
            fh.write(rows[oid])
    for oid in ids:
        # absolute, resolved: the link must not depend on data/pretrain_v2 staying a symlink
        (TARGET / f"{oid}.fits").symlink_to((V2 / f"{oid}.fits").resolve())
    LINK.symlink_to(TARGET)

    with (LINK / "metadata.csv").open(newline="") as fh:
        back = [r["object_id"] for r in csv.DictReader(fh)]
    links = sum(1 for p in TARGET.iterdir() if p.is_symlink() and p.suffix == ".fits")
    assert back == ids, "metadata.csv does not read back in cut order"
    assert links == len(ids), (links, len(ids))
    print(f"{LINK} -> {TARGET}\n  metadata.csv: {len(back):,} rows (+ header), cut-log order\n"
          f"  symlinks    : {links:,} .fits -> {V2.resolve()}\n"
          f"  first/last  : {ids[0]} / {ids[-1]}")


if __name__ == "__main__":
    main()
