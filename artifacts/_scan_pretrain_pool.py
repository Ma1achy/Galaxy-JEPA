"""Pass 1 of the pretrain target list: scan the unlabelled galaxy pool run by run.

Why run-partitioned rather than paged:

    `PhotoPrimary` is ~1e9 rows and `type=3 AND clean=1 AND modelMag_r BETWEEN ...` is not an
    indexed combination, so any query that has to *find* its matches scans the table. A single
    50,000-row page ran past 15 minutes on the SkyServer endpoint and had to be killed -- with
    or without a `objID % k` stride (the modulus was never the problem; the scan was). But
    `run` is the high-order part of the clustered objID key, so bounding on it turns the same
    query into a range scan: measured 8,028 rows in 19s and 21,462 rows in 66s, ~370 rows/s.

    765 runs at that rate is a handful of hours, unattended and resumable, and it needs no
    SciServer token -- this is the public SkyServer endpoint, not the Jobs API.

This pass writes *every* matching galaxy (~11M expected). Sub-sampling to the corpus size is
pass 2's job, and it is done by a uniform hash over the whole pool rather than by stopping
early: stopping early would hand back a contiguous patch of sky, which is exactly the defect
the stride was introduced to remove.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
csv.field_size_limit(1 << 24)

from sciserver_pull import _net_retry  # noqa: E402

from galaxy_jepa.data.metadata import run_sql  # noqa: E402

MAG_MIN, MAG_MAX = 14.0, 19.0
WORK = Path(".sciserver_work")
POOL = WORK / "pretrain_pool.csv"
DONE = WORK / "pretrain_pool.runs_done"
LOCK = WORK / "pretrain_pool.lock"
COLS = (
    "objID", "ra", "dec", "petroRad_r", "petroRadErr_r",
    "modelMag_r", "modelMagErr_r", "run", "camcol", "field", "rerun",
)


def _run_sql_for(run: int) -> str:
    cols = ", ".join(f"p.{c}" for c in COLS)
    return (
        f"SELECT {cols}\nFROM PhotoPrimary AS p\n"
        f"WHERE p.run = {run} AND p.type = 3 AND p.clean = 1\n"
        f"  AND p.modelMag_r BETWEEN {MAG_MIN} AND {MAG_MAX}\nORDER BY p.objID"
    )


def _claim_lock() -> None:
    """Refuse to run twice. Two scanners append to one CSV and one runs-done list, which
    interleaves rows and marks runs complete whose rows never landed -- observed: 24,439
    duplicated objIDs and a runs-done list claiming 8 runs more than the file contained."""
    if LOCK.exists():
        pid = LOCK.read_text().strip()
        alive = pid.isdigit() and Path(f"/proc/{pid}").exists()
        if not alive and pid.isdigit():  # macOS has no /proc; ask the kernel directly
            try:
                os.kill(int(pid), 0)
                alive = True
            except (ProcessLookupError, ValueError):
                alive = False
            except PermissionError:
                alive = True
        if alive:
            sys.exit(f"another scan is already running (pid {pid}); refusing to double-write {POOL}")
        print(f"clearing stale lock from dead pid {pid}", flush=True)
    LOCK.write_text(str(os.getpid()))


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    _claim_lock()
    runs = [int(r["run"]) for r in _net_retry(run_sql, "SELECT DISTINCT run FROM Field ORDER BY run", timeout=600)]
    done = set()
    if DONE.exists():
        done = {int(x) for x in DONE.read_text().split() if x.strip()}
    todo = [r for r in runs if r not in done]
    print(f"{len(runs)} runs in DR17; {len(done)} already scanned, {len(todo)} to go", flush=True)

    total = 0
    if POOL.exists():
        with POOL.open(newline="") as fh:
            total = sum(1 for _ in fh) - 1
        print(f"pool file already holds {total:,} galaxies", flush=True)

    for i, run in enumerate(todo, start=1):
        try:
            rows = _net_retry(run_sql, _run_sql_for(run), timeout=1200, _tries=4)
        except Exception as exc:  # noqa: BLE001 -- one bad run must not end the scan
            print(f"  run {run}: FAILED ({type(exc).__name__}: {str(exc)[:90]}); leaving for a later pass", flush=True)
            continue
        write_header = not POOL.exists()
        with POOL.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(COLS), extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerows(rows)
        with DONE.open("a") as fh:
            fh.write(f"{run}\n")
        total += len(rows)
        if i % 10 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] run {run}: +{len(rows):,}  pool now {total:,}", flush=True)

    LOCK.unlink(missing_ok=True)
    print(f"\nscan complete: {total:,} galaxies across {len(runs)} runs -> {POOL}")


if __name__ == "__main__":
    main()
