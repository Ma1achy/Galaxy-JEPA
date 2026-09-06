"""Plan the full-GZ2 probe pull (230,358) reusing the 40,000 already on disk.

Three things this does that the driver's own `_load_or_plan` cannot:

*   **Pages the target query.** `_targets` issues one `TOP {limit}` REST call; at 230k rows x
    ~130 vote columns that response is ~350 MB and the endpoint 500s. Paging on `dr8objid >
    last` is a transport detail only — the `query` recorded in the state, and so in the manifest
    hash, stays the canonical `probe_sql(limit)` that *defines* the population.
*   **Streams.** 230k rows x 130 columns held as dicts is several GB; accumulating them OOM'd an
    18 GiB machine at ~120k. Nothing here holds more than one page: pages append to a partial
    CSV as they land (which also makes a flaky page cost one page, not the whole fetch), and the
    derive/verify passes read that file back a row at a time.
*   **Marks the prefix done.** `PROBE_SQL` is `ORDER BY g.dr8objid`, so the existing 40,000 are
    exactly the first 40,000 of the 230,358 (verified against the live catalogue, not assumed —
    and re-verified below objID-by-objID before anything is marked). With `max_per_job=4000`
    that is chunks 0-9 exactly, so the driver's own resume path skips them.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
csv.field_size_limit(1 << 24)

from _sciserver_auth import authenticate  # noqa: E402
from sciserver_pull import _net_retry  # noqa: E402

from galaxy_jepa.data.metadata import PROBE_SQL_TEMPLATE, gz2_vote_columns, probe_sql, run_sql
from galaxy_jepa.data.pull import with_derived_columns

LIMIT = 230_358
PAGE = 10_000
MAX_PER_JOB = 4_000
BATCH = 5_000  # rows held at once in the derive pass
OUT = Path("data/probe")
WORK = Path(".sciserver_work")
PARTIAL = WORK / "probe_targets_full.partial.csv"
TARGETS = WORK / "probe_all_targets.csv"


def _page_sql(page: int, after: int | None) -> str:
    votes = ",\n".join(f"    g.{c}" for c in gz2_vote_columns())
    sql = PROBE_SQL_TEMPLATE.format(limit=page, votes=votes)
    if after is not None:  # paging is transport; the recorded query stays the canonical one
        sql = sql.replace("ORDER BY g.dr8objid", f"WHERE g.dr8objid > {after}\nORDER BY g.dr8objid")
    return sql


def _resume() -> tuple[int, int | None]:
    """(rows already fetched, last objID) — read a row at a time, never held."""
    if not PARTIAL.exists():
        return 0, None
    n, last = 0, None
    with PARTIAL.open(newline="") as fh:
        for row in csv.DictReader(fh):
            n += 1
            last = row["objID"]
    return n, (int(last) if last is not None else None)


def _fetch() -> None:
    n, after = _resume()
    if n:
        print(f"resuming: {n:,} targets already on disk (last objID {after})", flush=True)
    while n < LIMIT:
        got = _net_retry(run_sql, _page_sql(min(PAGE, LIMIT - n), after), timeout=900, _tries=6)
        if not got:
            break
        with PARTIAL.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(got[0].keys()))
            if not n:
                w.writeheader()
            w.writerows(got)
        n += len(got)
        after = int(got[-1]["objID"])
        del got
        print(f"  targets {n:>7,} / {LIMIT:,}  (last objID {after})", flush=True)
    if n != LIMIT:
        sys.exit(f"expected {LIMIT} targets, got {n} — refusing to plan a partial pull")


def _derive_and_verify() -> int:
    """Stream partial -> targets CSV applying the single derivation site; verify order + prefix."""
    have = sorted(int(r["object_id"]) for r in csv.DictReader((OUT / "metadata.csv").open()))
    n_have = len(have)

    n, prev, writer, fh_out = 0, None, None, None
    with PARTIAL.open(newline="") as fh_in, TARGETS.open("w", newline="") as fh_out:
        batch: list[dict] = []
        for row in csv.DictReader(fh_in):
            batch.append(row)
            if len(batch) < BATCH:
                continue
            n, prev, writer = _flush(batch, fh_out, writer, n, prev, have, n_have)
            batch = []
        if batch:
            n, prev, writer = _flush(batch, fh_out, writer, n, prev, have, n_have)
    if n != LIMIT:
        sys.exit(f"derived {n} rows, expected {LIMIT}")
    return n_have


def _flush(batch, fh_out, writer, n, prev, have, n_have):
    derived = with_derived_columns(batch)
    if writer is None:
        writer = csv.DictWriter(fh_out, fieldnames=list(derived[0].keys()))
        writer.writeheader()
    for r in derived:
        oid = int(r["object_id"])
        if prev is not None and oid <= prev:
            sys.exit(f"targets not strictly increasing at objID {oid} — refusing")
        if n < n_have and oid != have[n]:
            sys.exit(f"row {n}: target {oid} != corpus {have[n]} — the {n_have} stamps on disk "
                     "are NOT the ordered prefix; refusing to mark them done")
        prev = oid
        n += 1
    writer.writerows(derived)
    return n, prev, writer


def main() -> None:
    authenticate(verbose=True)
    WORK.mkdir(exist_ok=True)
    _fetch()
    n_have = _derive_and_verify()
    if n_have % MAX_PER_JOB:
        sys.exit(f"{n_have} on disk is not a whole number of {MAX_PER_JOB}-target chunks")

    chunks, off, k = [], 0, 0
    while off < LIMIT:
        m = min(MAX_PER_JOB, LIMIT - off)
        chunks.append({"k": k, "offset": off, "n_targets": m,
                       "status": "done" if off + m <= n_have else "pending",
                       "jid": None, "rel": None})
        off += m
        k += 1
    (WORK / "probe.job.json").write_text(json.dumps(
        {"corpus": "probe", "out_dir": str(OUT), "stamp_px": 256, "query": probe_sql(LIMIT),
         "limit": LIMIT, "max_per_job": MAX_PER_JOB, "chunks": chunks}, indent=2))
    done = sum(c["status"] == "done" for c in chunks)
    print(f"planned {len(chunks)} chunks of <={MAX_PER_JOB}: {done} already on disk, "
          f"{len(chunks) - done} to pull ({LIMIT - n_have:,} stamps)")


if __name__ == "__main__":
    main()
