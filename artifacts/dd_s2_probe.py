"""Brief DD Stop-3 item 3: is the injected trail represented in M's block-11 tokens at all?

A token-level linear probe for "trail present" on the same injected-trail set as S2 (identical
seeds and draw order: `dd_sae_score.s2_eval`). The train/test split is by galaxy. Criterion hashed in
artifacts/interp_tooling.md: DETECTED if the held-out AUC ≥ 0.75.

  uv run python artifacts/dd_s2_probe.py plants    # before the hash
  uv run python artifacts/dd_s2_probe.py run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import dd_core as D  # noqa: E402
from dd_sae_score import S2_FRAC, SEED, inject_trail  # noqa: E402

OUT = D.LOCAL / "sae"
AUC_BAR, TEST_FRAC = 0.75, 0.3


def injected_set():
    """The S2 set exactly: rng(SEED + 2) picks 2% of sae_eval, then trails are drawn in row order."""
    _, st = D.stamps("sae_eval")
    rng = np.random.default_rng(SEED + 2)
    rows = sorted(int(k) for k in rng.choice(len(st), int(round(S2_FRAC * len(st))), replace=False))
    imgs, cores = zip(*[inject_trail(st[k], rng) for k in rows], strict=True)
    return rows, np.stack(imgs), np.stack(cores).reshape(len(rows), 256)


@torch.no_grad()
def tokens(imgs: np.ndarray, layer: int) -> np.ndarray:
    m = D.m_encoder()
    out = []
    for a in range(0, len(imgs), 32):
        t = D.block_tokens(m, torch.from_numpy(imgs[a:a + 32]).to(D.DEVICE), (layer,))[layer]
        out.append(t.cpu().numpy())
    return np.concatenate(out)  # (G, 256, D)


def pixel_features(imgs: np.ndarray) -> np.ndarray:
    """Context baseline: per patch, each band's mean and standard deviation (6 features)."""
    p = imgs.reshape(len(imgs), 3, 16, 16, 16, 16).transpose(0, 2, 4, 1, 3, 5).reshape(len(imgs), 256, 3, 256)
    return np.concatenate([p.mean(-1), p.std(-1)], axis=-1)


def probe_auc(feat: np.ndarray, lab: np.ndarray, seed: int = SEED) -> dict:
    """Standardised L2 logistic regression (C = 1); galaxies split 70/30 (seeded); token-level AUC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    g = len(feat)
    test = np.zeros(g, bool)
    test[np.random.default_rng(seed).choice(g, int(round(TEST_FRAC * g)), replace=False)] = True
    xtr, ytr = feat[~test].reshape(-1, feat.shape[-1]), lab[~test].ravel()
    xte, yte = feat[test].reshape(-1, feat.shape[-1]), lab[test].ravel()
    sc = StandardScaler().fit(xtr)
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(xtr), ytr)
    auc = float(roc_auc_score(yte, clf.decision_function(sc.transform(xte))))
    return {"auc": auc, "n_train_gal": int((~test).sum()), "n_test_gal": int(test.sum()),
            "test_pos": int(yte.sum()), "test_tokens": len(yte), "state": "DETECTED" if auc >= AUC_BAR else "NOT DETECTED"}


def plants() -> dict:
    rows, imgs, cores = injected_set()
    tok = tokens(imgs, 11)
    rng = np.random.default_rng(SEED + 3)
    planted = np.concatenate([tok, (cores + rng.normal(0, 0.5, cores.shape))[..., None]], axis=-1)
    shuffled = np.stack([rng.permutation(c) for c in cores])  # same count per galaxy, positions scrambled
    out = {"plant (tokens + noisy label column)": probe_auc(planted, cores),
           "null (labels scrambled within galaxy)": probe_auc(tok, shuffled),
           "trail patches": int(cores.sum()), "galaxies": len(rows)}
    out["fire_as_expected"] = (out["plant (tokens + noisy label column)"]["state"] == "DETECTED"
                               and out["null (labels scrambled within galaxy)"]["state"] == "NOT DETECTED")
    (OUT / "s2_probe_plants.json").write_text(json.dumps(out, indent=1))
    return out


def run() -> dict:
    rows, imgs, cores = injected_set()
    out = {"block 11 (criterion)": probe_auc(tokens(imgs, 11), cores),
           "block 6 (reported)": probe_auc(tokens(imgs, 6), cores),
           "pixel baseline (reported)": probe_auc(pixel_features(imgs), cores)}
    out["state"] = out["block 11 (criterion)"]["state"]
    (OUT / "s2_probe.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    print(json.dumps(plants() if sys.argv[1] == "plants" else run(), indent=1))
