"""Does v2's ~1.7% higher per-band std come from the trimmed sets or from the stamps? (user, 2026-10-01)

Each freeze trims its own 827 heaviest stamps from the fit (D16), not the same 827. Recompute both
trims from the per-stamp moments (asserting each against its freeze's recorded trim_ids_sha256),
then the valid-pixel mean and std per band on the stamps NEITHER fit trimmed, for v1 and v2. If the
gap survives on the common set, it is in the stamps; if it closes, it was the trims.

    E5_CORPUS=pretrain uv run python artifacts/e5_corpus_moments.py   # v1's moments (v2's exist)
    uv run python artifacts/v1_v2_std_common.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e5_fit_normalisation import stats, trim_keep  # noqa: E402

SCRATCH = Path("/private/tmp/claude-501/-Users-malachy-Documents-Galaxy-JEPA/519f0c5e-3b4c-4159-bf72-4f179da196ae/scratchpad")
REPO = Path(__file__).resolve().parents[1]
RUNS = {"v1": ("corpus_moments.npz", "configs/pretrain.yaml"), "v2": ("corpus_moments_pretrain_v2.npz", "configs/pretrain_v2.yaml")}


def main() -> None:
    z, trimmed, freeze = {}, {}, {}
    for v, (npz, cfg) in RUNS.items():
        z[v] = np.load(SCRATCH / npz)
        freeze[v] = yaml.safe_load((REPO / cfg).read_text())["normalisation"]
        keep, _ = trim_keep(z[v])
        ex = np.sort(z[v]["object_id"][~keep])
        sha = hashlib.sha256(b"".join(int(o).to_bytes(8, "big") for o in ex)).hexdigest()
        assert sha == freeze[v]["trim_ids_sha256"], f"{v}: recomputed trim {sha[:12]} != freeze's"
        trimmed[v] = set(ex.tolist())
    common = set(z["v1"]["object_id"].tolist()) & set(z["v2"]["object_id"].tolist())
    common -= trimmed["v1"] | trimmed["v2"]
    out = {"trimmed_v1": len(trimmed["v1"]), "trimmed_v2": len(trimmed["v2"]),
           "trimmed_both": len(trimmed["v1"] & trimmed["v2"]), "common_untrimmed": len(common), "bands": "g, r, i"}
    for v in RUNS:
        ids = z[v]["object_id"]
        rows = np.flatnonzero(np.isin(ids, np.fromiter(common, np.int64)))
        s, _, _ = stats(z[v], rows)
        out[v] = {"common_mean": [float(x) for x in s.mean], "common_std": [float(x) for x in s.std],
                  "freeze_mean": freeze[v]["mean"], "freeze_std": freeze[v]["std"]}
    out["std_v2_over_v1_common"] = [b / a for a, b in zip(out["v1"]["common_std"], out["v2"]["common_std"])]
    out["std_v2_over_v1_freeze"] = [b / a for a, b in zip(out["v1"]["freeze_std"], out["v2"]["freeze_std"])]
    (REPO / "artifacts" / "out" / "v1_v2_std_common.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
