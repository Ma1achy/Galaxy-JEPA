"""Brief DD Part 4, the unit test at scale: ablate latent 328 (i−r x offset) through the hooks.

Pre-registered expectation (artifacts/interp_tooling.md, "Part 4 pre-registration"): ablating 328 at
block 11 (its decoder contribution removed from every token, the SAE error kept) moves the i−r x
offset probe towards chance, with a small change to the 37 morphology probes.

  uv run python artifacts/dd_part4.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import dd_core as D  # noqa: E402
import dd_sae as S  # noqa: E402
from dd_circuits import latent_patch, pooled_with  # noqa: E402
from dd_part2 import offsets  # noqa: E402

LATENT, LAYER, MULT = 328, 11, 8
E1_MIN_TOWARD, E2_MEAN, E2_MAX = 0.05, 0.01, 0.03
OUT = D.LOCAL / "sae" / "part4.json"


def _pooled_from_tokens(sample: str) -> np.ndarray:
    tok = np.load(S.TokenStore.path("M", LAYER, sample), mmap_mode="r")
    return np.stack([np.asarray(tok[g * 256:(g + 1) * 256], np.float32).mean(0) for g in range(len(tok) // 256)])


def offset_probe():
    """sign(i−r x) from the block-11 pooled embedding: standardised L2 logistic (C = 1), trained on the
    SAE's training galaxies (probe-train split) with recorded offsets."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ids = np.load(D.LOCAL / "sae_ids.npy")
    off = offsets(ids)
    rows = [k for k, o in enumerate(ids) if int(o) in off]
    x = _pooled_from_tokens("sae")[rows]
    y = np.array([off[int(ids[k])][2] > 0 for k in rows], int)
    sc = StandardScaler().fit(x)
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(x), y)
    w = clf.coef_[0] / sc.scale_
    b = float(clf.intercept_[0] - (clf.coef_[0] * sc.mean_ / sc.scale_).sum())
    return D.Readout(["i-r x > 0"], w[:, None], np.array([b])), len(rows)


def verdict(d_off_toward: float, dmorph: np.ndarray) -> str:
    """E1: |AUC − 0.5| falls by ≥ 0.05. E2: mean |ΔAUC| over the 37 ≤ 0.01 and max ≤ 0.03."""
    e1 = d_off_toward >= E1_MIN_TOWARD
    e2 = float(np.mean(np.abs(dmorph))) <= E2_MEAN and float(np.max(np.abs(dmorph))) <= E2_MAX
    return {(True, True): "AS EXPECTED", (False, True): "NO EFFECT", (True, False): "NOT SELECTIVE",
            (False, False): "NEITHER"}[(e1, e2)]


@torch.no_grad()
def main() -> dict:
    import r_nonlinear as R
    from j4_spread_controls import prepare
    from sklearn.metrics import roc_auc_score
    setup = prepare(None, R.MAX_TRAIN, label="DD4", sources=1)
    probes = D.probe_readout(setup, "real")
    band = D.band_offset_readout(setup)
    off_ro, n_train = offset_probe()
    m = D.m_encoder()
    sae = S.load("M", LAYER, MULT).to(D.DEVICE)
    ids, st = D.stamps("sae_eval")
    base, abl, abl_rand, abl_all = [], [], [], []
    tok = np.load(S.TokenStore.path("M", LAYER, "sae_eval"), mmap_mode="r")
    dens = np.zeros(sae.W_enc.shape[0])
    for a in range(0, min(len(tok), 256 * 1000), 65536):
        dens += (sae.encode(torch.from_numpy(np.asarray(tok[a:a + 65536], np.float32)).to(D.DEVICE)) > 0).float().sum(0).cpu().numpy()
    dens /= min(len(tok), 256 * 1000)
    # control: a random live latent within ±25% of 328's density
    near = [j for j in np.where(np.abs(dens - dens[LATENT]) <= 0.25 * dens[LATENT])[0] if j != LATENT]
    ctrl = int(np.random.default_rng(20260926).choice(near))
    score = json.loads((D.LOCAL / "sae" / "score.json").read_text())
    offset_set = sorted({t["latent"] for t in score["S1"]["b11"]["top"]})

    def drop_all(tokens):
        z = sae.encode(tokens)
        idx = torch.tensor(offset_set, device=D.DEVICE)
        return tokens - (z[..., idx] @ sae.W_dec[:, idx].T) / sae.scale

    for a in range(0, len(st), 64):
        x = torch.from_numpy(np.asarray(st[a:a + 64], np.float32)).to(D.DEVICE)
        base.append(pooled_with(m, x).cpu().numpy())
        abl.append(pooled_with(m, x, {LAYER: latent_patch(sae, LATENT)}).cpu().numpy())
        abl_rand.append(pooled_with(m, x, {LAYER: latent_patch(sae, ctrl)}).cpu().numpy())
        abl_all.append(pooled_with(m, x, {LAYER: drop_all}).cpu().numpy())
    base, abl, abl_rand, abl_all = (np.concatenate(v).astype(np.float64) for v in (base, abl, abl_rand, abl_all))
    # consistency: the hook path at block 11 equals the token arithmetic on the stored tokens
    tb = _pooled_from_tokens("sae_eval")
    consistency = float(np.abs(tb - base).max())

    off = offsets(ids)
    rows = [k for k, o in enumerate(ids) if int(o) in off]
    y_off = np.array([off[int(ids[k])][2] > 0 for k in rows], int)

    def off_auc(p):
        return float(roc_auc_score(y_off, off_ro.scores(p[rows])[:, 0]))

    def morph_aucs(p):
        out = {}
        s = probes.scores(p)
        for j, f in enumerate(probes.names):
            el = setup.labels.eligible(f, [int(i) for i in ids])
            yy = np.asarray(setup.labels.binary_label(f, el))
            if len(np.unique(yy)) < 2:
                continue
            r = [k for k, o in enumerate(ids) if int(o) in set(el)]
            out[f] = float(roc_auc_score(yy, s[r, j]))
        return out

    res: dict = {"latent": LATENT, "control_latent": ctrl, "density_328": float(dens[LATENT]),
                 "density_control": float(dens[ctrl]), "offset_probe_train_n": n_train, "offset_test_n": len(rows),
                 "hook_vs_token_arithmetic_max_abs": consistency}
    m0 = morph_aucs(base)
    for tag, p in (("ablate 328", abl), ("ablate control", abl_rand), (f"ablate all {len(offset_set)} offset latents", abl_all)):
        a0, a1 = off_auc(base), off_auc(p)
        toward = abs(a0 - 0.5) - abs(a1 - 0.5)
        mm = morph_aucs(p)
        d = np.array([mm[f] - m0[f] for f in m0])
        pc1 = (band.scores(p) - band.scores(base))[:, 0]
        res[tag] = {"offset_auc_base": a0, "offset_auc_ablated": a1, "toward_chance": toward,
                    "morph_mean_abs_dauc": float(np.abs(d).mean()), "morph_max_abs_dauc": float(np.abs(d).max()),
                    "morph_worst": max(m0, key=lambda f: abs(mm[f] - m0[f])),
                    "pc1_shift_median_abs": float(np.median(np.abs(pc1))), "verdict": verdict(toward, d)}
    res["state"] = res["ablate 328"]["verdict"]
    OUT.write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
