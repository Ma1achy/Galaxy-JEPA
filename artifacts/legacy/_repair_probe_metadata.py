"""Rebuild data/probe/metadata.csv from the authoritative target list.

Why this exists: `_accumulate` appended every fetched chunk's rows in the chunk's own
column order beneath the corpus's *alphabetised* header (see sciserver_pull._accumulate).
The 40,000-row prefix written by `pull.write_metadata` is correct; the 190,358 rows the
full pull appended are the right values under the wrong names -- silently, because
`object_id` occupies slot 0 in both orderings.

No image is touched. `.sciserver_work/probe_all_targets.csv` holds all 230,358 rows exactly
as SkyServer returned them, so the corpus metadata is fully recoverable offline.

Streams: the naive "read every row into a list" version of this OOM'd an 18 GiB machine at
~120k rows of ~130 columns, which is why only the salvage map and the on-disk id set are
ever held whole.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

CORPUS = Path("data/probe")
TARGETS = Path(".sciserver_work/probe_all_targets.csv")
SALVAGE = ("expAB_r", "deVAB_r")  # only ever present in the good 40k prefix (D13 top-up)

csv.field_size_limit(10_000_000)


def main() -> None:
    meta = CORPUS / "metadata.csv"
    if not TARGETS.exists():
        sys.exit(f"missing {TARGETS} -- the rebuild has no source of truth")

    # -- the stamps that actually exist; the rebuilt csv must describe exactly these ------
    on_disk = {int(p.stem) for p in CORPUS.iterdir() if p.suffix == ".fits"}
    print(f"on-disk stamps: {len(on_disk):,}")

    # -- salvage the axis ratios from full-width (uncorrupted) rows only -----------------
    salvaged: dict[int, tuple[str, ...]] = {}
    if meta.exists():
        with meta.open(newline="") as fh:
            r = csv.reader(fh)
            hdr = next(r)
            idx = {c: i for i, c in enumerate(hdr)}
            if all(c in idx for c in SALVAGE):
                for row in r:
                    if len(row) != len(hdr):  # a corrupt row: its values are misaligned
                        continue
                    vals = tuple(row[idx[c]] for c in SALVAGE)
                    if any(v not in ("", "NULL") for v in vals):
                        salvaged[int(row[idx["object_id"]])] = vals
    print(f"salvaged {SALVAGE} for {len(salvaged):,} objects from the intact prefix")

    # -- canonical column order: exactly pull.write_metadata's ---------------------------
    with TARGETS.open(newline="") as fh:
        tgt_hdr = next(csv.reader(fh))
    rest = sorted((set(tgt_hdr) - {"objID", "object_id"}) | set(SALVAGE))
    fieldnames = ["object_id", *rest]
    print(f"rebuilt header: {len(fieldnames)} columns (object_id first, rest sorted)")

    tmp = meta.with_suffix(".csv.rebuild")
    seen: set[int] = set()
    written = skipped = 0
    with TARGETS.open(newline="") as src, tmp.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=fieldnames, extrasaction="ignore", restval="")
        writer.writeheader()
        for row in reader:
            oid = int(row["objID"])
            if row.get("object_id") not in (None, "") and int(row["object_id"]) != oid:
                sys.exit(f"target row {oid}: derived object_id disagrees with objID")
            if oid not in on_disk:
                skipped += 1
                continue
            if oid in seen:
                sys.exit(f"duplicate objID {oid} in the target list")
            seen.add(oid)
            out = {k: v for k, v in row.items() if k != "objID"}
            out["object_id"] = oid
            ax = salvaged.get(oid)
            out["expAB_r"], out["deVAB_r"] = ax if ax else ("", "")
            writer.writerow(out)
            written += 1
            if written % 50_000 == 0:
                print(f"  {written:,} rows ...", flush=True)

    print(f"wrote {written:,} rows ({skipped:,} targets had no stamp on disk)")
    missing = on_disk - seen
    if missing:
        sys.exit(f"{len(missing):,} on-disk stamps absent from the target list, e.g. {sorted(missing)[:3]}")
    print(f"staged at {tmp} -- verify before swapping")


if __name__ == "__main__":
    main()
