"""Aligned comparison, criterion 2 (morphology, balanced 2×2): machinery and D28 plants.

Encoders: M1 = M (runs/m, train seed 0), M2 = O2 (runs/o2, train seed 1; M's config_hash and splits).
A1/A2 (aligned, seeds 0/1) do not exist yet. Probes follow the headline protocol's training size:
the full probe-train split, `_fit` with C from probe.yaml; AUC on the whole probe-test split.

Statistic (artifacts/aligned_comparison.md): over the powered answers, D̄ = mean_j [mean(A1,A2) −
mean(M1,M2)], noise S̄ = mean_j [(|M1−M2| + |A1−A2|)/2] (pooled seed spread, both conditions);
95% CI of D̄ from a Poisson galaxy bootstrap, weights shared across answers and encoders.

  uv run python artifacts/aligned_c2.py embed <tag> <checkpoint>   # e.g. m1 runs/m/encoder.pt
  uv run python artifacts/aligned_c2.py plants
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from j4_spread_controls import OUT, prepare  # noqa: E402
from probe_bank import load_bank, write_bank  # noqa: E402

N_BOOT, SEED, Q = 2_000, 20260926, 0.05
POWER_MIN_CLASS, POWER_MIN_AUC = 100, 0.6


def _bank(tag: str) -> Path:
    return OUT / f"c2_{tag}_embeddings.npz"


def embed(tag: str, checkpoint: str) -> dict:
    from galaxy_jepa.models.vit import load_frozen_encoder
    from galaxy_jepa.probing.extract import extract_matrix
    s = prepare(checkpoint, 0, label=f"C2-{tag}", sources=1)
    m = extract_matrix(load_frozen_encoder(s.ckpt), s.ds, device=s.device)
    write_bank(_bank(tag), ids=m.object_ids, x=m.x.astype(np.float32), checkpoint=str(s.ckpt))
    return {"tag": tag, "n": len(m.object_ids), "train": len(s.train_ids), "test": len(s.test_ids)}


def fit_scores(tags: list[str]) -> tuple[dict, dict, list[int]]:
    """Per encoder tag and answer: test-split probe scores. Returns (scores[tag][f], y[f], test_ids)."""
    from galaxy_jepa.probing.extract import EmbeddingMatrix, feature_embeddings, feature_ids
    from galaxy_jepa.probing.logistic import _fit
    s = prepare(None, 0, label="C2-fit", sources=1)
    scores: dict = {t: {} for t in tags}
    ys: dict = {}
    rows: dict = {}
    for t in tags:
        b = load_bank(_bank(t))  # the recorded checkpoint must still hash to the bank's SHA-1
        mat = EmbeddingMatrix(b["ids"].astype(np.int64), b["x"].astype(np.float64), t)
        for f in s.labels.features:
            tr = feature_embeddings(mat, s.labels, f, s.train_ids)
            te = feature_embeddings(mat, s.labels, f, s.test_ids)
            if len(np.unique(tr.y)) < 2 or len(np.unique(te.y)) < 2:
                continue
            sc, clf = _fit(tr, c=s.pc.c)
            scores[t][f] = clf.decision_function(sc.transform(te.x))
            ys.setdefault(f, te.y.astype(int))
            rows.setdefault(f, feature_ids(mat, s.labels, f, s.test_ids))
    return scores, {"y": ys, "rows": rows}, list(s.test_ids)


def auc(s: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def boot_aucs(s: np.ndarray, y: np.ndarray, w: torch.Tensor) -> np.ndarray:
    """AUC under each row of galaxy weights w (B × n): Σ_pos w·(neg weight below) / (W_pos W_neg)."""
    o = np.argsort(s, kind="stable")
    yo = torch.from_numpy(y[o].astype(np.float32))
    wo = w[:, torch.from_numpy(o)]
    neg_cum = torch.cumsum(wo * (1 - yo), dim=1) - wo * (1 - yo)
    num = (wo * yo * neg_cum).double().sum(1)
    return (num / ((wo * yo).sum(1).double() * (wo * (1 - yo)).sum(1).double())).numpy()


def weights(test_ids: list[int], seed: int = SEED, b: int = N_BOOT) -> tuple[torch.Tensor, dict]:
    g = torch.Generator().manual_seed(seed)
    w = torch.poisson(torch.ones((b, len(test_ids)), dtype=torch.float64), generator=g).float()
    return w, {int(o): k for k, o in enumerate(test_ids)}


def powered(sc: dict, ys: dict) -> list[str]:
    """Smaller test class ≥ 100 and mean(M1, M2) AUC ≥ 0.6 — fixed from M alone."""
    out = []
    for f, y in ys.items():
        if f not in sc["m1"] or f not in sc["m2"]:
            continue
        if min(int(y.sum()), int((1 - y).sum())) >= POWER_MIN_CLASS and \
                (auc(sc["m1"][f], y) + auc(sc["m2"][f], y)) / 2 >= POWER_MIN_AUC:
            out.append(f)
    return out


def _by(p: np.ndarray, q: float = Q) -> np.ndarray:
    """Benjamini–Yekutieli rejections."""
    m = len(p)
    o = np.argsort(p)
    c = np.sum(1 / np.arange(1, m + 1))
    ok = p[o] <= q * np.arange(1, m + 1) / (m * c)
    k = np.max(np.where(ok)[0]) + 1 if ok.any() else 0
    rej = np.zeros(m, bool)
    rej[o[:k]] = True
    return rej


#: M1's and M2's bootstrap AUCs do not change across plant realisations; the plants fill this once.
_boot_cache: dict[str, np.ndarray] = {}


def statistic(sc: dict, meta: dict, answers: list[str], w: torch.Tensor, pos: dict) -> dict:
    """The criterion-2 read on encoders sc['a1'], sc['a2'], sc['m1'], sc['m2']."""
    enc = ("a1", "a2", "m1", "m2")
    pt = {e: np.array([auc(sc[e][f], meta["y"][f]) for f in answers]) for e in enc}
    bt = {e: _boot_cache[e] if e in _boot_cache else
          np.stack([boot_aucs(sc[e][f], meta["y"][f], w[:, [pos[int(o)] for o in meta["rows"][f]]])
                    for f in answers], 1) for e in enc}  # B × answers
    d = (pt["a1"] + pt["a2"]) / 2 - (pt["m1"] + pt["m2"]) / 2
    s = (np.abs(pt["m1"] - pt["m2"]) + np.abs(pt["a1"] - pt["a2"])) / 2
    db = (bt["a1"] + bt["a2"]) / 2 - (bt["m1"] + bt["m2"]) / 2
    dbar, sbar = float(d.mean()), float(s.mean())
    lo, hi = np.percentile(db.mean(1), [2.5, 97.5])

    def state(lo_, hi_, s_):
        if lo_ > s_:
            return "BETTER"
        if hi_ < -s_:
            return "WORSE"
        if lo_ >= -s_ and hi_ <= s_:
            return "SAME"
        return "UNRESOLVED"

    def per_answer(bar: np.ndarray) -> list[str]:
        p_b = np.array([(db[:, j] <= bar[j]).mean() for j in range(len(answers))])
        p_w = np.array([(db[:, j] >= -bar[j]).mean() for j in range(len(answers))])
        rb, rw = _by(p_b), _by(p_w)
        out_ = []
        for j in range(len(answers)):
            l_, h_ = np.percentile(db[:, j], [2.5, 97.5])
            out_.append("BETTER" if rb[j] else "WORSE" if rw[j] else
                        ("SAME" if l_ >= -bar[j] and h_ <= bar[j] else "UNRESOLVED"))
        return out_

    # bar S_j, the pooled S̄, or the shrunk B_j = (S_j + S̄)/2 (the fallback under the 6% rule)
    own, pooled, shrunk = per_answer(s), per_answer(np.full(len(answers), sbar)), per_answer((s + sbar) / 2)
    per = {}
    for j, f in enumerate(answers):
        l_, h_ = np.percentile(db[:, j], [2.5, 97.5])
        per[f] = {"D": float(d[j]), "S": float(s[j]), "ci": [float(l_), float(h_)], "state": own[j],
                  "state_pooled_bar": pooled[j], "state_shrunk_bar": shrunk[j]}
    return {"state": state(lo, hi, sbar), "D_bar": dbar, "S_bar": sbar, "ci": [float(lo), float(hi)],
            "per_answer": per}


def shifted(s: np.ndarray, y: np.ndarray, target: float) -> np.ndarray:
    """Scores with positives shifted by δ·sd so that AUC = target (bisection on δ)."""
    sd = s.std()
    lo, hi = -5.0, 5.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if auc(s + mid * sd * y, y) < target:
            lo = mid
        else:
            hi = mid
    return s + (lo + hi) / 2 * sd * y


def plants(reps: int = 20) -> dict:
    """D28 for criterion 2, through the identical statistic path. 'Aligned' encoders are M1 and M2
    with each powered answer's AUC moved to M_e + Δ + ε (ε ~ N(0, σ_seed) independently per seed and
    answer; σ_seed from M1–M2), Δ ∈ {+0.02, −0.02, 0}. Detection and false-call rates over `reps`
    realisations. Separately, the realistic-width power check from M1 vs M2 as independent draws.

    A fourth plant, Δ = 0 heteroscedastic, draws ε_j ~ N(0, σ_j) with σ_j = |M1_j − M2_j|/√2 floored
    at 0.25 σ_seed: the per-answer bar's false-call rate when seed noise varies by answer. It runs
    last, so the three homoscedastic plants draw the identical RNG stream they drew before it."""
    sc, meta, test_ids = fit_scores(["m1", "m2"])
    ans = powered(sc, meta["y"])
    w, pos = weights(test_ids)
    m = {e: np.array([auc(sc[e][f], meta["y"][f]) for f in ans]) for e in ("m1", "m2")}
    sigma = float(np.sqrt(np.mean((m["m1"] - m["m2"]) ** 2) / 2))
    rng = np.random.default_rng(SEED)
    for e in ("m1", "m2"):
        _boot_cache[e] = np.stack([boot_aucs(sc[e][f], meta["y"][f], w[:, [pos[int(o)] for o in meta["rows"][f]]])
                                   for f in ans], 1)
    out: dict = {"powered": ans, "n_powered": len(ans), "sigma_seed": sigma,
                 "M_seed_spread_mean_abs": float(np.abs(m["m1"] - m["m2"]).mean()), "plants": {}}
    sig_j = np.maximum(np.abs(m["m1"] - m["m2"]) / np.sqrt(2), 0.25 * sigma)
    noisiest = np.argsort(-sig_j)[:int(np.ceil(len(ans) / 3))]  # the third with the largest σ_j
    top = np.zeros(len(ans), bool)
    top[noisiest] = True
    out["hetero_sigma_j"] = dict(zip(ans, map(float, sig_j), strict=True))
    out["hetero_noisiest_third"] = [ans[j] for j in noisiest]
    for name, delta, want, sd in (("+0.02", 0.02, "BETTER", sigma), ("-0.02", -0.02, "WORSE", sigma),
                                  ("0", 0.0, "SAME", sigma), ("0 heteroscedastic", 0.0, "SAME", sig_j)):
        fam, calls = [], {"state": [], "state_pooled_bar": [], "state_shrunk_bar": []}
        for _ in range(reps):
            planted = dict(sc)
            for e, a in (("m1", "a1"), ("m2", "a2")):
                eps = rng.normal(0, sd, len(ans))
                planted[a] = {f: shifted(sc[e][f], meta["y"][f], float(np.clip(m[e][j] + delta + eps[j], 0.5, 0.999)))
                              for j, f in enumerate(ans)}
            r = statistic(planted, meta, ans, w, pos)
            fam.append(r["state"])
            for k in calls:
                calls[k].append([v[k] for v in r["per_answer"].values()])
        rec = {"must_read": want, "family_states": {k: fam.count(k) for k in set(fam)},
               "family_rate_as_required": fam.count(want) / reps}
        for k, prefix in (("state", "per_answer_rate"), ("state_pooled_bar", "pooled_bar_rate"),
                          ("state_shrunk_bar", "shrunk_bar_rate")):
            c = np.array(calls[k])  # reps × answers
            for st in ("BETTER", "WORSE", "SAME"):
                rec[f"{prefix}_{st}"] = float((c == st).mean())
            if delta == 0.0:
                false = np.isin(c, ("BETTER", "WORSE"))
                rec[f"{prefix}_false_call"] = float(false.mean())
                rec[f"{prefix}_false_call_noisiest_third"] = float(false[:, top].mean())
                rec[f"{prefix}_false_call_rest"] = float(false[:, ~top].mean())
        out["plants"][name] = rec
    # the 6% rule (aligned_comparison.md, criterion 2): above it, the per-answer bar becomes B_j
    fc = out["plants"]["0 heteroscedastic"]["pooled_bar_rate_false_call"]
    out["per_answer_bar_decision"] = {"pooled_false_call_hetero": fc,
                                      "bar": "pooled S̄" if fc <= 0.06 else "shrunk B_j = (S_j + S̄)/2"}
    # Realistic width: two genuinely independent draws. sd of mean_j(M1_j − M2_j) under the bootstrap;
    # with four draws and independent errors, sd(D̄) ≈ that / √2. Minimum detectable D̄ ≈ S̄ + 1.96 sd.
    b1, b2 = _boot_cache["m1"], _boot_cache["m2"]
    sd_pair = float((b1 - b2).mean(1).std())
    s_bar = out["M_seed_spread_mean_abs"]  # the aligned spread is unknown; assume equal to M's
    out["power"] = {"sd_mean_diff_independent_draws": sd_pair, "sd_Dbar_approx": sd_pair / np.sqrt(2),
                    "S_bar_assumed": s_bar,
                    "min_detectable_Dbar": s_bar + 1.96 * sd_pair / np.sqrt(2)}
    out["fire_as_expected"] = all(v["family_rate_as_required"] >= 0.95 for v in out["plants"].values())
    (OUT / "c2_plants.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    r = embed(sys.argv[2], sys.argv[3]) if cmd == "embed" else plants()
    print(json.dumps(r, indent=1))
