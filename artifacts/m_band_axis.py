"""Layer 3 (re-pull), now on M: when during training does the band-misregistration axis appear?

At every saved checkpoint of M (and the final encoder), embed a fixed 5,000-galaxy PCA sample (a
deterministic stride of M's probe-train ids) and AA3a's 2,000 galaxies. Fit PCA on the sample and
project the 2,000. Report the CV R² of each of the top-10 PCs on AA3a's measured INTER band
offsets (`aa3a_offsets.npz`; AA3a's own `cv_r2`, rank-normal, 5-fold). The axis need not be
PC1/PC2 early on, so the state reads the best of the ten.

States and thresholds: `repull_findings.md` §Layer 3a.

  --planted   D28 on the statistic: a planted axis is recovered; a shuffled one is not
  --scan      the scan (checkpoints in step order, then the final encoder)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import aa3a_pose as A  # noqa: E402
import r_nonlinear as R  # noqa: E402
import w2_name_pcs as W  # noqa: E402
from i3_loss_usability import frozen_from  # noqa: E402
from j4_spread_controls import _release, prepare  # noqa: E402

from galaxy_jepa.models.vit import load_frozen_encoder  # noqa: E402
from galaxy_jepa.probing.extract import extract_matrix  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "runs" / "m"
OUT = R.OUT / "m_band_axis.json"
PLANTED = R.OUT / "m_band_axis_planted.json"
N_PCA = 5_000
K = 10
BAR = 0.3  # CV R² at which the axis is "present"
EARLY_STEP = 12_664  # the first two checkpoints (≈ 2 h of M's training each for a smoke rerun)
AA3A_FINAL = (0.83, 0.84)  # AA3a: INTER → PC1, PC2 at the final encoder
REPRO_TOL = 0.10


def _inter() -> tuple[np.ndarray, np.ndarray]:
    z = np.load(A.MEAS)
    return z["ids"], A.candidates(z["cen"], z["core"])["inter"]


def _pcs(frozen, setup, pca_idx, aa_idx) -> np.ndarray:
    ds = setup.ds
    xp = extract_matrix(frozen, torch.utils.data.Subset(ds, pca_idx), device=setup.device).x.astype(np.float64)
    xa = extract_matrix(frozen, torch.utils.data.Subset(ds, aa_idx), device=setup.device).x.astype(np.float64)
    _release(setup.device)
    mu = xp.mean(0)
    _, v = np.linalg.eigh(np.cov(xp, rowvar=False))
    v = v[:, ::-1][:, :K]
    sd = ((xp - mu) @ v).std(0)
    return ((xa - mu) @ v) / sd


def score(pc: np.ndarray, inter: np.ndarray) -> dict:
    r2 = [A.cv_r2(pc[:, k], inter) for k in range(pc.shape[1])]
    best = int(np.argmax(r2))
    return {"r2_by_pc": r2, "pc1": r2[0], "pc2": r2[1], "best": r2[best], "best_pc": best + 1}


def state(rows: list[dict]) -> str:
    """EARLY: present (best ≥ 0.3) at a checkpoint ≤ step 12,664. LATE: first present later.
    NEVER: never present, the final encoder included (it would contradict AA3a — read as a defect
    of this scan, not a finding)."""
    first = next((r["step"] for r in rows if r["best"] >= BAR), None)
    if first is None:
        return "NEVER"
    return "EARLY" if first <= EARLY_STEP else "LATE"


def planted() -> dict:
    ids, inter = _inter()
    rng = np.random.default_rng(3)
    z = np.column_stack([W.rank_normal(np.nan_to_num(c, nan=np.nanmedian(c))) for c in inter.T])
    axis = z[:, 0::2].sum(1)
    axis /= axis.std()
    out = {}
    for noise in (0.0, 1.0, 1.5, 3.0):
        pc = np.column_stack([axis + noise * rng.normal(size=ids.size)] + [rng.normal(size=ids.size) for _ in range(K - 1)])
        out[f"axis_on_pc1_noise{noise}"] = score(pc, inter)["best"]
    pc = np.column_stack([rng.normal(size=ids.size) for _ in range(K)])
    out["no_axis"] = score(pc, inter)["best"]
    buried = np.column_stack([rng.normal(size=ids.size) for _ in range(K)])
    buried[:, 6] = axis + 1.0 * rng.normal(size=ids.size)
    s = score(buried, inter)
    out["axis_on_pc7"] = {"best": s["best"], "best_pc": s["best_pc"]}
    rows = [{"step": 6331, "best": 0.05}, {"step": 12662, "best": 0.35}]
    out["states"] = {"early": state(rows), "late": state([{"step": 6331, "best": 0.1}, {"step": 50648, "best": 0.6}]),
                     "never": state([{"step": 6331, "best": 0.1}])}
    return out


def scan() -> dict:
    setup = prepare(None, R.MAX_TRAIN, label="L3a", sources=1)
    ids, inter = _inter()
    pos = {int(o): i for i, o in enumerate(setup.ds.object_ids)}
    stride = len(setup.train_ids) / N_PCA
    pca_idx = [pos[setup.train_ids[int(i * stride)]] for i in range(N_PCA)]
    aa_idx = [pos[int(o)] for o in ids]
    entries = json.loads((RUN / "checkpoints" / "checkpoints.json").read_text())["entries"]
    rows = []
    scratch = R.OUT / "l3a_scratch_encoder.pt"
    for e in sorted(entries, key=lambda e: e["step"]):
        frozen, step = frozen_from(RUN / "checkpoints" / e["file"], scratch)
        rows.append({"step": step, **score(_pcs(frozen, setup, pca_idx, aa_idx), inter)})
        print(f"  step {step:>7,}: PC1 {rows[-1]['pc1']:.3f}  PC2 {rows[-1]['pc2']:.3f}  "
              f"best {rows[-1]['best']:.3f} (PC{rows[-1]['best_pc']})", flush=True)
        del frozen
    final = score(_pcs(load_frozen_encoder(RUN / "encoder.pt"), setup, pca_idx, aa_idx), inter)
    repro = abs(final["pc1"] - AA3A_FINAL[0]) <= REPRO_TOL and abs(final["pc2"] - AA3A_FINAL[1]) <= REPRO_TOL
    scratch.unlink(missing_ok=True)
    out = {"n_pca": N_PCA, "n_aa3a": int(ids.size), "checkpoints": rows, "final": final,
           "reproduces_aa3a": repro, "state": state(rows + [{"step": 10**9, **final}])}
    if not repro:
        out["state"] = f"{out['state']} — SCAN DEFECT (final encoder does not reproduce AA3a)"
    return out


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--scan"
    out, path = (planted(), PLANTED) if mode == "--planted" else (scan(), OUT)
    path.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "checkpoints"}, indent=1, default=float))


if __name__ == "__main__":
    main()
