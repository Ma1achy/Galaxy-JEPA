"""Pass 2 of the pretrain corpus: select the target set from the scanned pool and chunk it.

Pass 1 (`_scan_pretrain_pool`) writes *every* unlabelled galaxy in the magnitude window,
run by run. This pass turns that pool into the corpus target list:

1. **Exclude Galaxy Zoo.** D6 wants the encoder trained on galaxies that were never labelled.
   That is the union of all five GZ2 tables (`_pull_gz_ids`), not just the probe corpus.
2. **Select uniformly, exactly.** `assignment_unit` is a stable SHA-256 map to [0, 1), so
   taking the N smallest values is a reproducible uniform sample of the *whole* pool. It has
   to be the whole pool: truncating a sky-ordered scan early would hand back a contiguous
   patch of sky, which is the defect the whole exercise exists to avoid.
3. **Prove the leak guard.** `resolve_corpora` runs on the real manifests, not just in a unit
   test -- a probe galaxy cannot survive into pretraining.
4. **Chunk** into job-sized pieces for the SciServer driver.

Streams throughout: the pool is millions of rows and only two flat arrays (one hash, one id)
are ever held whole.
"""

from __future__ import annotations

import csv
import json
import sys
from array import array
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
csv.field_size_limit(1 << 24)

from galaxy_jepa.data.metadata import pretrain_sql  # noqa: E402
from galaxy_jepa.data.orchestrate import resolve_corpora  # noqa: E402
from galaxy_jepa.data.pull import with_derived_columns  # noqa: E402
from galaxy_jepa.data.splits import assignment_unit  # noqa: E402

TARGET = 2_000_000
SEED = 0
SALT = "pretrain-select"
MAX_PER_JOB = 4_000
BATCH = 20_000
OUT = Path("data/pretrain")
WORK = Path(".sciserver_work")
POOL = WORK / "pretrain_pool.csv"
GZ = WORK / "gz_all_ids.csv"
TARGETS = WORK / "pretrain_all_targets.csv"
STATE = WORK / "pretrain.job.json"


def _gz_ids() -> set[int]:
    with GZ.open(newline="") as fh:
        return {int(r["dr8objid"]) for r in csv.DictReader(fh)}


def main() -> None:
    if not POOL.exists():
        sys.exit(f"{POOL} missing -- run artifacts/_scan_pretrain_pool.py first")
    gz = _gz_ids()
    print(f"Galaxy Zoo exclusion set: {len(gz):,} ids")

    # -- pass A: hash every eligible galaxy -------------------------------------------------
    units, oids = array("d"), array("q")
    n_pool = n_gz = 0
    with POOL.open(newline="") as fh:
        for row in csv.DictReader(fh):
            n_pool += 1
            oid = int(row["objID"])
            if oid in gz:
                n_gz += 1
                continue
            units.append(assignment_unit(oid, SEED, salt=SALT))
            oids.append(oid)
    n_elig = len(oids)
    print(f"pool {n_pool:,} galaxies; {n_gz:,} were in Galaxy Zoo; {n_elig:,} eligible")
    if n_elig < TARGET:
        sys.exit(f"only {n_elig:,} eligible galaxies, need {TARGET:,} -- widen the magnitude window")

    u = np.frombuffer(units, dtype=np.float64)
    ids = np.frombuffer(oids, dtype=np.int64)
    keep = set(ids[np.argpartition(u, TARGET)[:TARGET]].tolist())
    print(f"selected {len(keep):,} of {n_elig:,} ({len(keep) / n_elig:.1%}) by uniform hash")
    del u, ids, units, oids

    # -- the structural leak guard, on the real id sets --------------------------------------
    probe_ids = [int(r["object_id"]) for r in csv.DictReader((OUT.parent / "probe" / "metadata.csv").open(newline=""))]
    deduped = resolve_corpora(keep, probe_ids)
    if len(deduped) != len(keep):
        print(f"resolve_corpora removed {len(keep) - len(deduped):,} probe galaxies")
        keep = set(deduped)
    print(f"leak guard passed: {len(keep):,} pretraining ids, disjoint from {len(set(probe_ids)):,} probe ids")

    # -- pass B: write the target list, in scan (= objID) order ------------------------------
    written, prev = 0, None
    with POOL.open(newline="") as src, TARGETS.open("w", newline="") as dst:
        reader, writer, batch = csv.DictReader(src), None, []
        for row in reader:
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

    # -- chunk plan ---------------------------------------------------------------------------
    chunks, off, k = [], 0, 0
    while off < written:
        n = min(MAX_PER_JOB, written - off)
        chunks.append({"k": k, "offset": off, "n_targets": n, "status": "pending", "jid": None, "rel": None})
        off += n
        k += 1
    STATE.write_text(json.dumps({
        "corpus": "pretrain", "out_dir": str(OUT), "stamp_px": 256,
        "query": pretrain_sql(TARGET), "limit": written,
        "max_per_job": MAX_PER_JOB, "chunks": chunks,
    }, indent=2))
    print(f"planned {len(chunks):,} chunks of <={MAX_PER_JOB} -> {STATE}")


def _flush(batch, dst, writer, written):
    derived = with_derived_columns(batch)
    if writer is None:
        writer = csv.DictWriter(dst, fieldnames=list(derived[0].keys()))
        writer.writeheader()
    writer.writerows(derived)
    return written + len(derived), writer


if __name__ == "__main__":
    main()
