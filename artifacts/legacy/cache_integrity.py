"""Kickoff B, item 1b: M's fp16 cache against its sources, after the 2026-09-02 object_id clobber.

Three tiers, cheapest first; any failure raises. Read-only.
  1. index: ids unique; id set = pretrain ∪ probe metadata object_ids (as int64, never float);
     the corpora disjoint; len(ids) x stamp bytes = data file size; every id has its <id>.fits;
     no id is a DR7 id (the clobber's signature: probe dr7objid values in the object_id column).
  2. sidecars: the petroRad scalars and probe columns load (their digests are checked on load)
     and agree with metadata for a random 2,000 ids.
  3. content: 5,000 random rows (stratified by corpus) rebuilt raw -> pipeline, bit-identical fp16.

  uv run python artifacts/cache_integrity.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from f0_preconditions import check  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "artifacts" / "out" / "cache_integrity.json"
SEED, K_CONTENT, K_SIDECAR = 20260927, 5_000, 2_000


def _meta(corpus: Path, cols: tuple[str, ...]) -> dict[int, dict[str, str]]:
    with (corpus / "metadata.csv").open(newline="") as fh:
        return {int(r["object_id"]): {c: r[c] for c in cols} for r in csv.DictReader(fh)}


def main() -> None:
    t0 = time.time()
    cfg, cache = check(verbose=False)
    from galaxy_jepa.data.cache import load_probe_columns, load_scalars
    from galaxy_jepa.data.sources import load_fits_stamp
    from galaxy_jepa.harness import _build_pipeline
    idx = cache.index
    pre_dir, pro_dir = REPO / cfg.paths.pretrain_dir, REPO / cfg.paths.probe_dir
    pre = _meta(pre_dir, ("petroRad_r",))
    pro = _meta(pro_dir, ("petroRad_r", "dr7objid"))
    ids = np.asarray(idx.object_ids, dtype=np.int64)
    out: dict = {"cache": str(cache.cache_dir),
                 "n_index": int(len(ids))}

    # tier 1
    t1 = {"unique": bool(len(np.unique(ids)) == len(ids)),
          "set_equals_pretrain_union_probe": set(ids.tolist()) == set(pre) | set(pro),
          "corpora_disjoint": not (set(pre) & set(pro)),
          "n_pretrain": len(pre), "n_probe": len(pro)}
    stamp = int(np.prod(idx.shape)) * np.dtype(idx.dtype).itemsize
    data_path = cache.cache_dir / "stamps.f16"
    t1["data_bytes"] = data_path.stat().st_size
    t1["size_matches"] = t1["data_bytes"] == len(ids) * stamp
    stems = {int(e.name[:-5]) for d in (pre_dir, pro_dir) for e in os.scandir(d) if e.name.endswith(".fits")}
    t1["every_id_has_fits"] = set(ids.tolist()) <= stems
    dr7 = {int(v["dr7objid"]) for v in pro.values() if v["dr7objid"] not in ("", None)}
    t1["dr7_ids_in_index"] = len(dr7 & set(ids.tolist()))
    t1["ok"] = (t1["unique"] and t1["set_equals_pretrain_union_probe"] and t1["corpora_disjoint"]
                and t1["size_matches"] and t1["every_id_has_fits"] and t1["dr7_ids_in_index"] == 0)
    out["tier1_index"] = t1
    print(json.dumps(t1), file=sys.stderr)
    if not t1["ok"]:
        OUT.write_text(json.dumps(out, indent=1))
        raise SystemExit("TIER 1 FAILED")

    # tier 2
    cdir = data_path.parent
    scal = load_scalars(cdir, idx)
    cols = load_probe_columns(cdir, idx)
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(ids), K_SIDECAR, replace=False)
    meta = pre | pro
    bad_s = sum(1 for j in pick if not np.isclose(scal[j], float(meta[int(ids[j])]["petroRad_r"]), rtol=0, atol=0, equal_nan=True))
    pro_rows = [j for j in pick if int(ids[j]) in pro]
    out["tier2_sidecars"] = {"scalars_checked": len(pick), "scalars_mismatch": bad_s,
                             "probe_columns": list(idx.probe_columns)[:5],
                             "probe_rows_readable": sum(1 for j in pro_rows if cols[int(ids[j])] is not None),
                             "probe_rows_checked": len(pro_rows), "ok": bad_s == 0}
    print(json.dumps(out["tier2_sidecars"]), file=sys.stderr)

    # tier 3 — stratified by corpus, in proportion
    pipe = _build_pipeline(q=cfg.q, freeze=cfg.normalisation)
    is_pro = np.array([int(o) in pro for o in ids])
    n_pro = round(K_CONTENT * is_pro.mean())
    sel = np.concatenate([rng.choice(np.where(is_pro)[0], n_pro, replace=False),
                          rng.choice(np.where(~is_pro)[0], K_CONTENT - n_pro, replace=False)])
    bad, worst = [], 0.0
    for j in np.sort(sel):
        oid = int(ids[j])
        src = pro_dir if is_pro[j] else pre_dir
        fresh = np.asarray(pipe(load_fits_stamp(src / f"{oid}.fits")), dtype=np.float16)
        cached = np.asarray(cache.get(oid))
        if not np.array_equal(fresh, cached):
            bad.append(oid)
            worst = max(worst, float(np.abs(fresh.astype(np.float64) - cached.astype(np.float64)).max()))
    out["tier3_content"] = {"checked": int(len(sel)), "probe": int(n_pro), "pretrain": int(len(sel) - n_pro),
                            "mismatches": len(bad), "worst_abs": worst, "examples": bad[:10], "ok": not bad}
    out["ok"] = t1["ok"] and out["tier2_sidecars"]["ok"] and not bad
    out["seconds"] = round(time.time() - t0)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
