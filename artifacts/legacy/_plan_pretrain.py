"""Pass 2 of the pretrain corpus: cut the target set out of the scanned pool and chunk it.

Pass 1 (`_scan_pretrain_pool`) writes *every* unlabelled galaxy in the magnitude window, run
by run. This pass narrows that pool to the corpus and plans the jobs:

1. **Exclude Galaxy Zoo.** D6 wants the encoder trained on galaxies that were never labelled
   -- the union of all five GZ2 tables (`_pull_gz_ids`), not just the probe corpus.
2. **Apply the resolution floor.** GZ2 labelled the bright, well-resolved galaxies, so
   excluding it leaves a pool whose median galaxy spans ~18 px at 0.396"/px -- about one
   16x16 ViT patch, against ~33 px for the probe corpus. Pretraining on one-token galaxies
   would spend I-JEPA's budget predicting sky and make a weak probe unreadable: it would
   measure the tokeniser, not the science. `petroRad_r > 5"` restores parity at ~34 px, and
   `<= 25"` (the probe's own p99) drops the deblending failures the raw pool runs to -- some
   with a Petrosian radius of 258", larger than the whole stamp.
3. **Take everything that survives.** No sub-sampling: the two cuts already bring the corpus
   under the disk ceiling, so the population is exactly what the query says it is and the
   provenance needs no selection rule on top.
4. **Prove the leak guard.** `resolve_corpora` runs on the real manifests, not just in a unit
   test -- a probe galaxy cannot survive into pretraining.

Streams throughout: the pool is millions of rows and only the id set is ever held whole.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
csv.field_size_limit(1 << 24)

from galaxy_jepa.data.metadata import pretrain_sql  # noqa: E402
from galaxy_jepa.data.orchestrate import resolve_corpora  # noqa: E402
from galaxy_jepa.data.pull import with_derived_columns  # noqa: E402

PETRO_MIN = 5.0
PETRO_MAX = 25.0  # the probe corpus's own p99: above this are deblending failures
MAX_PER_JOB = 1_000  # the Small domain kills a job at 60 min; 4,000 targets overran it
#                      (wave 1 was cancelled at exactly 60:00). 1,000 is what the probe
#                      pull actually completed on: ~6.4 min fixed + ~10.4 min/1,000.
BATCH = 20_000
OUT = Path("data/pretrain")
PROBE = Path("data/probe")
WORK = Path(".sciserver_work")
POOL = WORK / "pretrain_pool.csv"
GZ = WORK / "gz_all_ids.csv"
TARGETS = WORK / "pretrain_all_targets.csv"
STATE = WORK / "pretrain.job.json"


def _eligible(row: dict, gz: set[int]) -> bool:
    if int(row["objID"]) in gz:
        return False
    try:
        rad = float(row["petroRad_r"])
    except (ValueError, KeyError):
        return False  # a galaxy with no measured size cannot clear a size floor
    return PETRO_MIN < rad <= PETRO_MAX


def main() -> None:
    if not POOL.exists():
        sys.exit(f"{POOL} missing -- run artifacts/_scan_pretrain_pool.py first")
    with GZ.open(newline="") as fh:
        gz = {int(r["dr8objid"]) for r in csv.DictReader(fh)}
    print(f"Galaxy Zoo exclusion set: {len(gz):,} ids")

    keep: set[int] = set()
    n_pool = n_gz = n_small = 0
    with POOL.open(newline="") as fh:
        for row in csv.DictReader(fh):
            n_pool += 1
            if int(row["objID"]) in gz:
                n_gz += 1
                continue
            if not _eligible(row, gz):
                n_small += 1
                continue
            keep.add(int(row["objID"]))
    print(f"pool {n_pool:,}; {n_gz:,} in Galaxy Zoo; {n_small:,} outside the "
          f"({PETRO_MIN}\", {PETRO_MAX}\"] size window; {len(keep):,} selected")

    with (PROBE / "metadata.csv").open(newline="") as fh:
        probe_ids = [int(r["object_id"]) for r in csv.DictReader(fh)]
    deduped = resolve_corpora(keep, probe_ids)
    if len(deduped) != len(keep):
        print(f"resolve_corpora removed {len(keep) - len(deduped):,} probe galaxies")
        keep = set(deduped)
    print(f"leak guard passed: {len(keep):,} pretraining ids, disjoint from "
          f"{len(set(probe_ids)):,} probe ids")

    written, prev = 0, None
    with POOL.open(newline="") as src, TARGETS.open("w", newline="") as dst:
        writer, batch = None, []
        for row in csv.DictReader(src):
            oid = int(row["objID"])
            if oid not in keep:
                continue
            if prev is not None and oid <= prev:
                sys.exit(f"pool is not in ascending objID order at {oid}; refusing to chunk")
            prev = oid
            batch.append(row)
            if len(batch) >= BATCH:
                written, writer = _flush(batch, dst, writer, written)
                batch = []
        if batch:
            written, writer = _flush(batch, dst, writer, written)
    if written != len(keep):
        sys.exit(f"wrote {written:,} rows but selected {len(keep):,}")
    print(f"wrote {written:,} targets -> {TARGETS}")

    chunks, off, k = [], 0, 0
    while off < written:
        n = min(MAX_PER_JOB, written - off)
        chunks.append({"k": k, "offset": off, "n_targets": n,
                       "status": "pending", "jid": None, "rel": None})
        off += n
        k += 1
    STATE.write_text(json.dumps({
        "corpus": "pretrain", "out_dir": str(OUT), "stamp_px": 256,
        "query": pretrain_sql(written, petro_min=PETRO_MIN, petro_max=PETRO_MAX),
        "limit": written,
        "max_per_job": MAX_PER_JOB, "chunks": chunks,
    }, indent=2))
    print(f"planned {len(chunks):,} chunks of <={MAX_PER_JOB} -> {STATE}")
    print(f"estimated: {written * 792_000 / 2**40:.2f} TiB raw, "
          f"{written * 393_216 / 2**40:.2f} TiB fp16 cache")


def _flush(batch, dst, writer, written):
    derived = with_derived_columns(batch)
    if writer is None:
        writer = csv.DictWriter(dst, fieldnames=list(derived[0].keys()))
        writer.writeheader()
    writer.writerows(derived)
    return written + len(derived), writer


if __name__ == "__main__":
    main()
