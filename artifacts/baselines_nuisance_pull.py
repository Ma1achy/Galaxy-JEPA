"""§N's nuisance columns for the identical split (baselines_comparison.md §N; user, 2026-10-23).

`sky_r` is not in probe_v2's metadata, and the earlier pull (`distractor_plants.pull_sky`) covers only
the 40,000-capped train + test. This pulls `PhotoObjAll.sky_r` for **exactly the identical split's
probe-train and probe-test galaxies** from public SkyServer (DR17, `metadata.run_sql`, no token), in
`distractor_plants`' 100-ID batches with its backoff, keeping each batch as it lands so a rerun
resumes. The same query also returns `psfWidth_r` and `modelMag_r`, which probe_v2's metadata already
carries: the record states how well the two sources agree, as the confirmation of those columns.

No encoder, bank or probe output is opened. The ids come from `prepare(None, 0)` under
LADDER_SPLIT=identical and must hash to the answer set's `split_ids_sha1`.

    .venv/bin/python artifacts/baselines_nuisance_pull.py   # -> artifacts/out/baselines_nuisance.{csv,json}

Investigation code: terse, excluded from lint/CI.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_run  # noqa: E402,F401  — A1's v2 environment, before F0 reads it

os.environ.setdefault("LADDER_SPLIT", "identical")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from j4_spread_controls import OUT, prepare  # noqa: E402

from galaxy_jepa.data.metadata import run_sql  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SKY_BATCH = 100  # distractor_plants': 400-ID IN-lists failed to connect on 2026-10-01, 100-ID ones went through


def _sql_retry(sql: str, tries: int = 12):
    """distractor_plants._sql_retry, copied (importing it pulls torch and the c13 machinery): the
    public endpoint drops connections, so back off and retry."""
    for k in range(tries):
        try:
            return run_sql(sql, timeout=120)
        except Exception as exc:  # noqa: BLE001 — timeouts / 5xx / refused connections are transient
            if k == tries - 1:
                raise
            print(f"  SkyServer call failed ({type(exc).__name__}); retry {k + 1} in {min(5 * 2 ** k, 120)} s",
                  file=sys.stderr, flush=True)
            time.sleep(min(5 * 2 ** k, 120))


# psfWidth_r lives in Field, not PhotoObjAll (metadata.PROBE_SQL_TEMPLATE joins it the same way)
SQL = ("SELECT CAST(p.objID AS varchar(20)) AS objID, p.sky_r, f.psfWidth_r, p.modelMag_r "
       "FROM PhotoObjAll AS p JOIN Field AS f ON f.fieldID = p.fieldID WHERE p.objID IN ({})")
IDS, PART = OUT / "baselines_split_ids.json", OUT / "baselines_nuisance.partial.json"
CSV, RECORD = OUT / "baselines_nuisance.csv", OUT / "baselines_nuisance.json"
TOL = 1e-4  # CSV round-trip of a float; the two sources are the same PhotoObjAll row


def _ids() -> dict:
    if IDS.exists():
        return json.loads(IDS.read_text())
    s = prepare(None, 0, label="baselines-nuisance", sources=1)
    tr, te = sorted(map(int, s.train_ids)), sorted(map(int, s.test_ids))
    sha = hashlib.sha1(json.dumps([tr, te]).encode()).hexdigest()
    want = json.loads((OUT / "baselines_answer_set.json").read_text())["split_ids_sha1"]
    if sha != want:
        raise SystemExit(f"nuisance: split SHA-1 {sha[:12]} is not the answer set's {want[:12]}")
    rec = {"split": "identical", "split_ids_sha1": sha, "train": tr, "test": te}
    IDS.write_text(json.dumps(rec))
    return rec


def main() -> None:
    if RECORD.exists():
        raise SystemExit(f"nuisance: {RECORD} exists; the pull is made once and pinned")
    t0 = time.time()
    sp = _ids()
    ids = sorted({str(i) for i in (*sp["train"], *sp["test"])}, key=int)
    got = json.loads(PART.read_text()) if PART.exists() else {}
    for k in range(0, len(ids), SKY_BATCH):
        if str(k) in got:
            continue
        got[str(k)] = _sql_retry(SQL.format(",".join(ids[k:k + SKY_BATCH])))
        PART.write_text(json.dumps(got))
        if k // SKY_BATCH % 100 == 0:
            print(f"  {k + SKY_BATCH:,} / {len(ids):,} ids  {time.time() - t0:5.0f}s", file=sys.stderr, flush=True)
    df = pd.DataFrame([r for k in sorted(got, key=int) for r in got[k]]).rename(columns={"objID": "object_id"})
    dup = int(df.object_id.duplicated().sum())
    df = df.drop_duplicates("object_id")
    df = df.assign(_k=df.object_id.astype(int)).sort_values("_k").drop(columns="_k").reset_index(drop=True)
    df.to_csv(CSV, index=False)
    sky = pd.to_numeric(df.sky_r, errors="coerce")

    # confirmation: the two columns probe_v2 already carries, against the same PhotoObjAll rows
    meta = pd.read_csv(REPO / "data/probe_v2/metadata.csv", usecols=["object_id", "psfWidth_r", "modelMag_r"],
                       dtype={"object_id": str})
    m = df.merge(meta, on="object_id", how="left", suffixes=("_sky", "_meta"))
    confirm = {}
    for c in ("psfWidth_r", "modelMag_r"):
        a, b = pd.to_numeric(m[f"{c}_sky"], errors="coerce"), pd.to_numeric(m[f"{c}_meta"], errors="coerce")
        ok = a.notna() & b.notna() & (a > -9000) & (b > -9000)
        d = (a[ok] - b[ok]).abs()
        confirm[c] = {"compared": int(ok.sum()), "agree_within_tol": int((d <= TOL).sum()),
                      "max_abs_diff": float(d.max()) if len(d) else None,
                      "missing_in_metadata": int(b.isna().sum()), "missing_in_skyserver": int(a.isna().sum()),
                      "sentinel_meta": int((b <= -9000).sum()), "sentinel_skyserver": int((a <= -9000).sum())}
    missing = set(ids) - set(df.object_id)
    rec = {"when": time.strftime("%Y-%m-%d %H:%M:%S"),
           "service": "public SkyServer SQL (galaxy_jepa.data.metadata.run_sql, DR17), no token",
           "query_template": SQL, "batch": SKY_BATCH,
           "galaxies": "the identical split's probe-train + probe-test (ladder v2 decision 1)",
           "split_ids_sha1": sp["split_ids_sha1"], "n_train": len(sp["train"]), "n_test": len(sp["test"]),
           "ids_sha1": hashlib.sha1(",".join(ids).encode()).hexdigest(), "ids_requested": len(ids),
           "rows_returned": len(df), "duplicate_rows_dropped": dup, "ids_missing": len(missing),
           "ids_missing_train": len(missing & {str(i) for i in sp["train"]}),
           "ids_missing_test": len(missing & {str(i) for i in sp["test"]}),
           "sky_r_sentinel_le_minus_9000": int((sky <= -9000).sum()), "sky_r_non_positive": int((sky <= 0).sum()),
           "sky_r_unparseable": int(sky.isna().sum()), "confirm_against_probe_v2_metadata": confirm,
           "tolerance": TOL, "output": "artifacts/out/" + CSV.name,
           "output_sha1": hashlib.sha1(CSV.read_bytes()).hexdigest(), "seconds": round(time.time() - t0)}
    RECORD.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "query_template"}, indent=1))


if __name__ == "__main__":
    main()
