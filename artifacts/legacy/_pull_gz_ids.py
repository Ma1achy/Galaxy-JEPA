"""Pull every Galaxy-Zoo-2 object ID, so the pretrain corpus can exclude them (D6).

"Never in Galaxy Zoo" is the union of all five GZ2 tables, not just the probe corpus's
`zoo2MainSpecz`. Pulling the ~1M ids once and filtering locally beats carrying five
`NOT EXISTS` clauses through a scan of millions of PhotoPrimary rows -- the exclusion then
also becomes an offline, auditable set rather than a property of a query nobody can re-check.

`resolve_corpora` still runs on the result: this pull is an optimisation that avoids cutting
stamps we would only throw away, not the guarantee. The guarantee stays structural.
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

TABLES = (
    "zoo2MainSpecz",
    "zoo2MainPhotoz",
    "zoo2Stripe82Normal",
    "zoo2Stripe82Coadd1",
    "zoo2Stripe82Coadd2",
)
PAGE = 100_000
WORK = Path(".sciserver_work")
OUT = WORK / "gz_all_ids.csv"


def _page(table: str, after: int | None) -> str:
    where = f"WHERE dr8objid > {after}\n" if after is not None else ""
    return f"SELECT TOP {PAGE} dr8objid\nFROM {table}\n{where}ORDER BY dr8objid"


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    ids: set[int] = set()
    for table in TABLES:
        after, got_total = None, 0
        while True:
            rows = _net_retry(run_sql, _page(table, after), timeout=900)
            if not rows:
                break
            for r in rows:
                ids.add(int(r["dr8objid"]))
            got_total += len(rows)
            after = int(rows[-1]["dr8objid"])
            if len(rows) < PAGE:
                break
        print(f"  {table:<22s} {got_total:>8,} ids   (union now {len(ids):,})", flush=True)

    tmp = OUT.with_suffix(".csv.partial")
    with tmp.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dr8objid"])
        for i in sorted(ids):
            w.writerow([i])
    os.replace(tmp, OUT)
    print(f"\nwrote {len(ids):,} unique Galaxy Zoo object ids -> {OUT}")


if __name__ == "__main__":
    main()
