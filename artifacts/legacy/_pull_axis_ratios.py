"""Pull expAB_r / deVAB_r for the full probe corpus and merge them in (D13 inclination proxy).

Two things the package's `pull_axis_ratios` + `merge_columns` cannot do at 230k:

*   **Pages.** `axis_ratio_sql` is one `TOP {limit}` REST call; the probe planner already saw
    SkyServer 500 on a single large response. Keyset-page on `dr8objid > last` exactly as
    `_plan_full_probe` does. Paging is transport only -- `AXIS_RATIO_SQL` mirrors
    `PROBE_SQL_TEMPLATE`'s FROM/JOIN/ORDER BY, so it walks the same objID set in the same order.
    Only three columns, so pages are ~50x the probe's and still a fraction of the size.
*   **Streams the merge.** `merge_columns` reads the whole corpus into dicts and rewrites it;
    230k rows x 131 columns is a couple of GB, on the machine that already OOM'd at ~120k during
    the pull. `expAB_r`/`deVAB_r` are already declared in the header (the 40k top-up put them
    there), so the merge only has to fill two cells per row -- one row in memory at a time.

Resumable: pages append to a partial CSV as they land, so a flaky page costs one page.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
csv.field_size_limit(1 << 24)

from sciserver_pull import _net_retry  # noqa: E402

from galaxy_jepa.data.metadata import AXIS_RATIO_SQL, run_sql  # noqa: E402
from galaxy_jepa.data.pull import AXIS_RATIO_COLS  # noqa: E402

LIMIT = 230_358
PAGE = 50_000
CORPUS = Path("data/probe")
WORK = Path(".sciserver_work")
PARTIAL = WORK / "axis_ratios.partial.csv"
FINAL = WORK / "axis_ratios.csv"


def _page_sql(page: int, after: int | None) -> str:
    sql = AXIS_RATIO_SQL.format(limit=page)
    if after is not None:  # transport only; the population is still AXIS_RATIO_SQL's
        sql = sql.replace("ORDER BY g.dr8objid", f"WHERE g.dr8objid > {after}\nORDER BY g.dr8objid")
    return sql


def _resume() -> tuple[int, int | None]:
    if not PARTIAL.exists():
        return 0, None
    n, last = 0, None
    with PARTIAL.open(newline="") as fh:
        for row in csv.DictReader(fh):
            n += 1
            last = row["objID"]
    return n, (int(last) if last is not None else None)


def fetch() -> None:
    if FINAL.exists():
        print(f"{FINAL} already complete; skipping fetch")
        return
    WORK.mkdir(parents=True, exist_ok=True)
    n, after = _resume()
    if n:
        print(f"resuming: {n:,} rows already fetched (last objID {after})")
    while n < LIMIT:
        got = _net_retry(run_sql, _page_sql(min(PAGE, LIMIT - n), after), timeout=900)
        if not got:
            break
        write_header = not PARTIAL.exists()
        with PARTIAL.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["objID", *AXIS_RATIO_COLS])
            if write_header:
                w.writeheader()
            for row in got:
                w.writerow({k: row[k] for k in ("objID", *AXIS_RATIO_COLS)})
        n += len(got)
        after = int(got[-1]["objID"])
        print(f"  axis ratios {n:,} / {LIMIT:,}  (last objID {after})", flush=True)
    os.replace(PARTIAL, FINAL)
    print(f"fetched {n:,} rows -> {FINAL}")


def merge() -> None:
    lookup: dict[int, tuple[str, str]] = {}
    with FINAL.open(newline="") as fh:
        for row in csv.DictReader(fh):
            lookup[int(row["objID"])] = tuple(row[c] for c in AXIS_RATIO_COLS)  # type: ignore[assignment]
    print(f"lookup holds {len(lookup):,} objects")

    meta = CORPUS / "metadata.csv"
    tmp = meta.with_suffix(".csv.axisratio")
    with meta.open(newline="") as src, tmp.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        fieldnames = reader.fieldnames or []
        missing_cols = [c for c in AXIS_RATIO_COLS if c not in fieldnames]
        if missing_cols:
            sys.exit(f"corpus header lacks {missing_cols}; refusing to widen it here")
        writer = csv.DictWriter(dst, fieldnames=fieldnames, restval="")
        writer.writeheader()
        matched = total = 0
        for row in reader:
            total += 1
            vals = lookup.get(int(row["object_id"]))
            if vals is not None:
                matched += 1
                for col, v in zip(AXIS_RATIO_COLS, vals, strict=True):
                    row[col] = v
            writer.writerow(row)
    print(f"merged: {matched:,} matched, {total - matched:,} unmatched, {total:,} rows")
    if matched != total:
        sys.exit(f"refusing to swap: {total - matched:,} corpus rows had no axis ratio")
    os.replace(tmp, meta)
    print(f"swapped {meta}")


if __name__ == "__main__":
    fetch()
    merge()
