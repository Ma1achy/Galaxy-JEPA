"""Data check of the leakage audit's loaders on the real corpora, before the hash (user, 2026-10-03,
the condition on item 5). No audit statistic, no predictor fitted, no state computed.

For 2,000 galaxies of each corpus's audit sample (the first 2,000 by the audit's own leakage order):
every input `audit()` reads, through `audit()`'s own loaders — completeness, the cut logs, the
identity inputs, the sample (against the pull's recorded sample SHA-1s), the pinned physics file
(against its recorded SHA-1), the metadata join, the targets with the s_b ≡ frac(origin) assertion,
the folds, the stamps (fp16, as the audit stores them) and their re-injected copy, and the variable
family. Shapes, dtypes, finite values and coverage are reported.

    uv run python artifacts/leakage_datacheck.py
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import leakage_audit as la  # noqa: E402
from galaxy_jepa.data.splits import assignment_unit  # noqa: E402

N_CHECK = 2_000


def _arr(a: np.ndarray) -> dict:
    a = np.asarray(a)
    fin = np.isfinite(a) if a.dtype.kind == "f" else np.ones(a.shape, bool)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "finite_frac": float(fin.mean()),
            **({"min": float(np.nanmin(a)), "max": float(np.nanmax(a))} if fin.any() and a.dtype.kind in "fiu" else {})}


def main() -> None:
    t0 = time.time()
    out: dict = {"n_check_per_corpus": N_CHECK, "no_statistic": "no predictor fitted, no state computed"}
    out["completeness"] = {c: la._complete(c) for c in la.CORPORA}
    logs = {c: pd.read_csv(la.REPO / "data" / c / "cut_log.csv", dtype={"object_id": str}) for c in la.CORPORA}
    out["cut_log"] = {c: {"rows": len(lg), "columns": len(lg.columns),
                          "ra": _arr(lg.ra.to_numpy(float)), "dec": _arr(lg.dec.to_numpy(float)),
                          "object_id_unique": bool(lg.object_id.is_unique)} for c, lg in logs.items()}
    samples = la._samples(logs)
    rec = json.loads(la.PHYS_RECORD.read_text())
    out["sample"] = {c: {"n": len(samples[c]),
                         "sha1": (s := hashlib.sha1(",".join(samples[c].object_id).encode()).hexdigest()),
                         "matches_pull_record": s == rec["sample_sha1"][c]} for c in la.CORPORA}
    got = hashlib.sha1((la.OUT / "audit_physics.csv").read_bytes()).hexdigest()
    out["physics_file"] = {"sha1": got, "recorded": rec["output_sha1"], "matches": got == rec["output_sha1"]}
    phys = la._physics([o for c in la.CORPORA for o in samples[c].object_id])
    out["family"] = {"stated": list(la.STATED), "n_stated": len(la.STATED), "family": la.FAMILY,
                     "family_is_42": la.FAMILY == 42}
    la.assert_resolution()
    out["family"]["resolution_ok"] = True
    out["corpora"] = {}
    for c in la.CORPORA:
        md = pd.read_csv(la.REPO / "data" / c / "metadata.csv", usecols=["object_id", "petroRad_r"], dtype={"object_id": str})
        md["petroRad_r"] = md.petroRad_r.where(md.petroRad_r > 0)
        full = samples[c].merge(phys, on="object_id", how="left").merge(md, on="object_id", how="left")
        lg = full.iloc[:N_CHECK].reset_index(drop=True)
        r: dict = {"join_rows_equal_sample": len(full) == len(samples[c]),
                   "physics_rows_found": float(samples[c].object_id.isin(phys.object_id).mean()),
                   "metadata_rows_found": float(samples[c].object_id.isin(md.object_id).mean()),
                   "missing_any_baseline_input_full_sample": float(full[la.AUDIT_PHYS].isna().any(axis=1).mean()),
                   "missing_per_input_full_sample": {k: float(full[k].isna().mean()) for k in la.AUDIT_PHYS},
                   "within_2pct_gate": bool(full[la.AUDIT_PHYS].isna().any(axis=1).mean() <= la.MAX_MISSING)}
        tb = la._targets(lg)  # raises if s_b ≠ 159.5 − frac(origin)
        r["s_b_equals_frac_origin"] = True
        tb.update({k: lg[k].to_numpy(float) for k in la.AUDIT_PHYS + la.AUDIT_COND})
        fold = np.array([int(assignment_unit(int(o), la.SEED, salt="leakage-fold") * la.N_FOLDS) for o in lg.object_id])
        r["folds"] = {int(k): int(v) for k, v in zip(*np.unique(fold, return_counts=True))}
        r["targets"] = {k: _arr(v) for k, v in tb.items() if not k.startswith("_")}
        r["v1_offsets"] = _arr(tb["_v1_off"])
        r["stated_finite_rows_min"] = int(min(np.isfinite(tb[t]).sum() for t in la.STATED))
        ids = lg.object_id.tolist()
        st = la._stamps(c, ids, "_datacheck")
        r["stamps"] = _arr(st)
        r["stamps_overflow_fp16"] = int((np.abs(np.asarray(st, np.float32)) >= 65504).sum())
        rj = la._stamps(c, ids[:200], "_datacheck_reinjected", tb["_v1_off"][:200])
        r["reinjected_first_200"] = _arr(rj)
        for tag in ("_datacheck", "_datacheck_reinjected"):
            (la.OUT / f"audit_{c}{tag}.f16").unlink(missing_ok=True)
        out["corpora"][c] = r
        print(f"  {c}: checked", flush=True)
    out["seconds"] = round(time.time() - t0)
    (la.OUT / "datacheck.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
