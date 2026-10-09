"""Brief DD, Part 3 criteria S1–S3: scoring functions, plants (D28), and the verdicts. TOOL VALIDATION.

The pre-registration is in artifacts/interp_tooling.md ("Part 3 pre-registration"). Every plant goes
through the same function as the real test.

  uv run python artifacts/dd_sae_score.py plants     # before the hash
  uv run python artifacts/dd_sae_score.py score      # after training + evaluation
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

OUT = D.RUN / "sae"
SEED = 20260925
S1_RHO, S1_MATCH_MAX, S1_MIN_N = 0.5, 0.2, 500
S2_FRAC, S2_PREC, S2_RECALL = 0.02, 0.8, 0.2
S3_MEAN, S3_MAX = 0.02, 0.05
POWER_MIN_CLASS, POWER_MIN_AUC = 100, 0.6
OFFSETS = ("g-r x", "g-r y", "i-r x", "i-r y")


def _rank(a: np.ndarray) -> np.ndarray:
    """Column-wise average ranks (for vectorised Spearman)."""
    from scipy.stats import rankdata
    return rankdata(a, axis=0)


def spearman_cols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(Lx, Ly) Spearman between every column of x and of y (rows = galaxies). Constant columns → 0."""
    rx, ry = _rank(x), _rank(y)
    rx = rx - rx.mean(0)
    ry = ry - ry.mean(0)
    nx, ny = np.linalg.norm(rx, axis=0), np.linalg.norm(ry, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (rx.T @ ry) / np.outer(nx, ny)
    return np.nan_to_num(r)


# ── S1 ───────────────────────────────────────────────────────────────────────────────────────────

def s1_stat(acts_m: np.ndarray, acts_u: np.ndarray, match: dict[int, int], y: np.ndarray) -> dict:
    """acts: galaxy-level mean activation (G, L); y: per-galaxy offsets (G, 4); match: M latent → its
    nearest untrained latent (by token-level activation correlation). States, in precedence:
    INSUFFICIENT (G < 500) > FAIL (no M latent at |ρ| ≥ 0.5 with any offset) > NOT-SPECIFIC (every such
    latent's match reaches |ρ| ≥ 0.2 with that offset) > PASS."""
    g = len(y)
    if g < S1_MIN_N:
        return {"n": g, "state": "INSUFFICIENT"}
    r = spearman_cols(acts_m, y)  # (L, 4)
    best = np.abs(r).max(1)
    cand = [int(c) for c in np.argsort(-best, kind="stable") if best[c] >= S1_RHO]  # ties: lower index
    out: dict = {"n": g, "max_abs_rho": float(best.max()), "n_candidates": len(cand), "top": []}
    ru_all = np.abs(spearman_cols(acts_u, y)).max(0)  # the untrained SAE's best latent per offset
    out["untrained_best_abs_rho"] = dict(zip(OFFSETS, ru_all.round(3).tolist(), strict=True))
    specific = []
    for c in cand[:50]:
        j = int(np.argmax(np.abs(r[c])))
        if c not in match:
            continue
        ru = float(spearman_cols(acts_u[:, [match[c]]], y[:, [j]])[0, 0])
        rec = {"latent": c, "offset": OFFSETS[j], "rho": float(r[c, j]), "match": match[c], "match_rho": ru}
        out["top"].append(rec)
        if abs(ru) < S1_MATCH_MAX:
            specific.append(rec)
    out["specific"] = specific[:10]
    out["state"] = "FAIL" if not cand else ("NOT-SPECIFIC" if not specific else "PASS")
    return out


@torch.no_grad()
def match_latents(sae_m, sae_u, tok_m: np.ndarray, tok_u: np.ndarray, cands: list[int], chunk: int = 65_536) -> dict:
    """For each candidate M latent, the untrained latent with the highest Pearson correlation of
    token-level activations over the same tokens (the two SAEs' decoders live in different
    networks' spaces, so decoder cosine is undefined)."""
    if not cands:
        return {}
    sae_m, sae_u = sae_m.to(D.DEVICE), sae_u.to(D.DEVICE)
    n = len(tok_m)
    sx = sxx = None
    sy = syy = sxy = None
    idx = torch.tensor(cands, device=D.DEVICE)
    for a in range(0, n, chunk):
        xm = sae_m.encode(torch.from_numpy(np.asarray(tok_m[a:a + chunk], np.float32)).to(D.DEVICE))[:, idx]
        xu = sae_u.encode(torch.from_numpy(np.asarray(tok_u[a:a + chunk], np.float32)).to(D.DEVICE))
        if sx is None:
            sx, sxx = xm.sum(0), (xm * xm).sum(0)
            sy, syy, sxy = xu.sum(0), (xu * xu).sum(0), xm.T @ xu
        else:
            sx += xm.sum(0)
            sxx += (xm * xm).sum(0)
            sy += xu.sum(0)
            syy += (xu * xu).sum(0)
            sxy += xm.T @ xu
    cov = sxy / n - torch.outer(sx / n, sy / n)
    sdx = (sxx / n - (sx / n) ** 2).clamp_min(1e-12).sqrt()
    sdy = (syy / n - (sy / n) ** 2).clamp_min(1e-12).sqrt()
    corr = (cov / torch.outer(sdx, sdy)).cpu().numpy()
    return {c: int(np.argmax(np.abs(corr[i]))) for i, c in enumerate(cands)}


# ── S2 ───────────────────────────────────────────────────────────────────────────────────────────

def inject_trail(img: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """A straight satellite trail in all three bands: Gaussian cross-section, FWHM 3.5 px; peak 3–10×
    each band's own sky σ (normalised units, `dd_core.sky_stats`); random angle; passing within 64 px
    of the centre. Returns (stamp, trail-patch mask (16, 16)): a patch is a trail patch if ≥ 16 of its
    pixels lie within 2 px of the trail's centre line."""
    _, sig = D.sky_stats(img)
    th, d, amp = rng.uniform(0, np.pi), rng.uniform(-64, 64), rng.uniform(3, 10)
    yy, xx = np.mgrid[:256, :256] - 127.5
    dist = np.abs(xx * np.sin(th) - yy * np.cos(th) - d)
    prof = np.exp(-0.5 * (dist / (3.5 / 2.355)) ** 2)
    out = np.asarray(img, np.float32) + (amp * sig[:, None, None] * prof[None]).astype(np.float32)
    core = (dist <= 2).reshape(16, 16, 16, 16).sum(axis=(1, 3)) >= 16
    return out, core


def s2_stat(fires_trail: np.ndarray, fires_total: np.ndarray, n_trail: int) -> dict:
    """Per latent: precision = fires on trail patches / all fires (over the whole held-out set, 2% of
    it trailed); recall = fires on trail patches / trail patches. DETECTED if any latent reaches
    precision ≥ 0.8 with recall ≥ 0.2 (a floor, so a latent firing once cannot count), else NOT DETECTED.
    Supporting, not decisive."""
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(fires_total > 0, fires_trail / fires_total, 0.0)
    rec = fires_trail / max(n_trail, 1)
    ok = (prec >= S2_PREC) & (rec >= S2_RECALL)
    best = int(np.argmax(np.where(rec >= S2_RECALL, prec, -1)))
    return {"n_trail_patches": int(n_trail), "n_detecting": int(ok.sum()),
            "best": {"latent": best, "precision": float(prec[best]), "recall": float(rec[best])},
            "state": "DETECTED" if ok.any() else "NOT DETECTED"}


# ── S3 ───────────────────────────────────────────────────────────────────────────────────────────

def s3_state(faith: dict, powered_only: bool = True) -> dict:
    """PASS (mean AUC drop ≤ 0.02 and no single answer > 0.05) > UNEVEN (mean ≤ 0.02, some answer
    > 0.05) > FAIL (mean > 0.02). The chosen dictionary is the smaller one (8×) at block 11 unless 8×
    fails and 16× does not.

    Only powered answers (smaller class ≥ 100 and baseline AUC ≥ 0.6) can trip UNEVEN: the aligned
    rerun's amendment (interp_tooling.md, "S3 — pre-rerun amendment"), stated as the rule for all four
    encoders (aligned_comparison.md, declared 2026-10-09; M's eight rows keep their states under it).
    The mean, and so FAIL, still counts every answer. `powered_only=False` is the pre-amendment rule,
    kept to show the identity."""
    per = faith["per_answer"]
    drops = np.array([v["drop"] for v in per.values()])
    mean, mx = float(drops.mean()), float(drops.max())
    out = {"mean_drop": mean, "max_drop": mx, "n_answers": len(drops)}
    if powered_only:
        if any("n_pos" not in v for v in per.values()):
            raise SystemExit("s3_state: the powered-UNEVEN amendment needs n_pos; re-evaluate the SAE")
        pw = [v["drop"] for v in per.values() if min(v["n_pos"], v["n"] - v["n_pos"]) >= POWER_MIN_CLASS
              and v["auc"] >= POWER_MIN_AUC]
        mx = float(max(pw)) if pw else 0.0
        out |= {"max_drop_powered": mx, "n_powered": len(pw)}
    out["state"] = "FAIL" if mean > S3_MEAN else ("UNEVEN" if mx > S3_MAX else "PASS")
    return out


# ── plants ───────────────────────────────────────────────────────────────────────────────────────

def plants() -> dict:
    rng = np.random.default_rng(SEED)
    g, L = 2000, 400
    y = rng.standard_normal((g, 4))
    base_m, base_u = rng.exponential(1, (g, L)), rng.exponential(1, (g, L))
    out: dict = {}
    # S1: a planted latent tracks g−r x at ρ ≈ 0.7; its untrained match does not → PASS
    m = base_m.copy()
    m[:, 7] = np.exp(y[:, 0] + 0.7 * rng.standard_normal(g))
    out["S1 plant"] = s1_stat(m, base_u, {7: 3}, y)["state"]
    out["S1 null"] = s1_stat(base_m, base_u, {}, y)["state"]
    u = base_u.copy()
    u[:, 3] = np.exp(y[:, 0] + 0.7 * rng.standard_normal(g))
    out["S1 untrained also tracks"] = s1_stat(m, u, {7: 3}, y)["state"]
    out["S1 thin"] = s1_stat(m[:300], base_u[:300], {7: 3}, y[:300])["state"]
    # S2: an oracle latent firing exactly on trail patches → DETECTED; random latents → NOT
    n_tok, n_trail = 5000 * 256, 900
    tot = rng.binomial(n_tok, 0.005, 300).astype(float)
    tr = rng.binomial(n_trail, 0.005, 300).astype(float)
    t2, f2 = tr.copy(), tot.copy()
    t2[11], f2[11] = 0.6 * n_trail, 0.6 * n_trail / 0.9
    out["S2 plant"] = s2_stat(t2, f2, n_trail)["state"]
    out["S2 null"] = s2_stat(tr, tot, n_trail)["state"]
    # S3: identity reconstruction → PASS; a lossy one → FAIL; one powered answer collapsing → UNEVEN;
    # one unpowered answer collapsing → PASS (only powered answers trip UNEVEN)
    pw = {"auc": 0.9, "n": 1000, "n_pos": 500}
    per = {f"a{i}": {"drop": 0.0, **pw} for i in range(37)}
    out["S3 identity"] = s3_state({"per_answer": per})["state"]
    out["S3 lossy"] = s3_state({"per_answer": {k: {**v, "drop": 0.05} for k, v in per.items()}})["state"]
    out["S3 one collapses"] = s3_state({"per_answer": {**per, "a0": {**pw, "drop": 0.2}}})["state"]
    out["S3 one unpowered collapses"] = s3_state({"per_answer": {**per, "a0": {**pw, "n_pos": 40, "drop": 0.2}}})["state"]
    # S2 injection geometry: the trail patches of a real stamp are a line of patches
    _, st = D.stamps("sae_eval")
    img, core = inject_trail(st[0], np.random.default_rng(1))
    out["S2 geometry: trail patches in one stamp"] = int(core.sum())
    expect = {"S1 plant": "PASS", "S1 null": "FAIL", "S1 untrained also tracks": "NOT-SPECIFIC",
              "S1 thin": "INSUFFICIENT", "S2 plant": "DETECTED", "S2 null": "NOT DETECTED",
              "S3 identity": "PASS", "S3 lossy": "FAIL", "S3 one collapses": "UNEVEN",
              "S3 one unpowered collapses": "PASS"}
    out["all_fire_as_expected"] = all(out[k] == v for k, v in expect.items())
    (OUT / "plants.json").write_text(json.dumps(out, indent=1))
    return out


# ── verdicts ─────────────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def galaxy_acts(sae, tok: np.ndarray, chunk_gal: int = 256) -> np.ndarray:
    sae = sae.to(D.DEVICE)
    n_gal = len(tok) // 256
    out = []
    for g in range(0, n_gal, chunk_gal):
        x = torch.from_numpy(np.asarray(tok[g * 256:(g + chunk_gal) * 256], np.float32)).to(D.DEVICE)
        out.append(sae.encode(x).reshape(-1, 256, sae.W_enc.shape[0]).mean(1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def s2_eval(layer: int, mult: int) -> dict:
    """Trails injected into 2% (100) of the held-out sae_eval galaxies (seeded); the other 98% are the
    stored clean tokens. Fires counted per latent over all 5,000 galaxies' tokens."""
    sae = S.load(D.TAG, layer, mult).to(D.DEVICE)
    model = D.m_encoder()
    _, st = D.stamps("sae_eval")
    n_gal = len(st)
    rng = np.random.default_rng(SEED + 2)
    picked = set(int(k) for k in rng.choice(n_gal, int(round(S2_FRAC * n_gal)), replace=False))
    tok = np.load(S.TokenStore.path(D.TAG, layer, "sae_eval"), mmap_mode="r")
    L = sae.W_enc.shape[0]
    total, trail, n_trail = np.zeros(L), np.zeros(L), 0
    for g in range(0, n_gal, 64):
        x = torch.from_numpy(np.asarray(tok[g * 256:(g + 64) * 256], np.float32)).to(D.DEVICE)
        f = (sae.encode(x) > 0).reshape(-1, 256, L)
        keep = [i for i in range(f.shape[0]) if g + i not in picked]
        total += f[keep].sum((0, 1)).cpu().numpy()
    rows = sorted(picked)
    for a in range(0, len(rows), 32):
        imgs, cores = zip(*[inject_trail(st[k], rng) for k in rows[a:a + 32]], strict=True)
        t = D.block_tokens(model, torch.from_numpy(np.stack(imgs)).to(D.DEVICE), (layer,))[layer]
        f = (sae.encode(t) > 0).cpu().numpy()  # (b, 256, L)
        c = np.stack(cores).reshape(len(imgs), 256)
        total += f.sum((0, 1))
        trail += f[c].sum(0)
        n_trail += int(c.sum())
    return s2_stat(trail, total, n_trail)


def chosen_size(layer: int = 11) -> int:
    ev = {m: json.loads((OUT / f"{D.TAG}_b{layer}_x{m}.eval.json").read_text()) for m in S.MULTS}
    st = {m: s3_state(ev[m]["faithfulness"])["state"] for m in S.MULTS}
    return 8 if st[8] != "FAIL" or st[16] == "FAIL" else 16


def score() -> dict:
    from dd_part2 import offsets
    ids = np.load(D.LOCAL / "sae_eval_ids.npy")
    res: dict = {"S3": {}}
    for layer in S.LAYERS:
        for mult in S.MULTS:
            for enc in (D.TAG, "untrained"):
                ev = json.loads((OUT / f"{enc}_b{layer}_x{mult}.eval.json").read_text())
                res["S3"][f"{enc}_b{layer}_x{mult}"] = {**s3_state(ev["faithfulness"]),
                                                       "variance_explained": ev["variance_explained"],
                                                       "dead_frac": ev["dead_frac"]}
    mult = chosen_size(11)
    res["chosen"] = mult
    res["S3_state"] = res["S3"][f"{D.TAG}_b11_x{mult}"]["state"]
    off = offsets(ids)
    rows = [k for k, o in enumerate(ids) if int(o) in off]
    y = np.stack([off[int(ids[k])] for k in rows])
    res["S1"] = {}
    for layer in S.LAYERS:
        sm, su = S.load(D.TAG, layer, mult), S.load("untrained", layer, mult)
        tm = np.load(S.TokenStore.path(D.TAG, layer, "sae_eval"), mmap_mode="r")
        tu = np.load(S.TokenStore.path("untrained", layer, "sae_eval"), mmap_mode="r")
        am, au = galaxy_acts(sm, tm)[rows], galaxy_acts(su, tu)[rows]
        r = np.abs(spearman_cols(am, y)).max(1)
        cands = [int(c) for c in np.argsort(-r) if r[c] >= S1_RHO][:50]
        res["S1"][f"b{layer}"] = s1_stat(am, au, match_latents(sm, su, tm, tu, cands), y)
    res["S1_state"] = res["S1"]["b11"]["state"]
    res["S2"] = {f"b{layer}": s2_eval(layer, mult) for layer in S.LAYERS}
    res["S2_state"] = res["S2"]["b11"]["state"]
    (OUT / "score.json").write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    cmd = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "plants":
        print(json.dumps(plants(), indent=1))
    elif cmd == "score":
        print(json.dumps(score(), indent=1, default=float))
