"""Brief DD Stop-4 item 1b: is the offset set's morphology collateral SPECIFIC or GENERIC?

Ablate Part 4's set (the 40 S1 offset latents) and 50 random sets of 40 latents that do not carry the
offset, matched to it on removed energy, at block 11 of M's 8× SAE; record every answer's probe-AUC
change. Criterion hashed in artifacts/interp_tooling.md ("Part 4b pre-registration").

Block 11 is M's last block and the probe reads its mean-pooled tokens, so ablating a set S is exact
token arithmetic: pooled − (Ā_S W_dec,Sᵀ)/scale, Ā the galaxy-mean activations (Part 4 checked the
hook path against this to 0.0024, the fp16 token storage). Removed energy is exact too:
E(S) = 1_Sᵀ C 1_S with C = (ZᵀZ/n) ⊙ (W_decᵀW_dec/scale²), one pass over the held-out tokens.

  uv run python artifacts/dd_part4b.py flags     # the full card-flag vector (cached)
  uv run python artifacts/dd_part4b.py stats     # Ā, C, band-offset ρ (cached)
  uv run python artifacts/dd_part4b.py pools     # pool sizes and sampler acceptance (no outcomes)
  uv run python artifacts/dd_part4b.py plants    # before the hash
  uv run python artifacts/dd_part4b.py run
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

LAYER, MULT, N_SETS, SEED = 11, 8, 50, 20260926
LIVE, OFF_RHO_MAX, E_TOL, MAX_PROPOSALS, MIN_SETS, ALPHA = 1e-4, 0.3, 0.15, 200_000, 20, 0.5
POWER_MIN_CLASS, POWER_MIN_AUC, PLANT_LAMBDA = 100, 0.6, 0.75
OUT = D.RUN / "sae"
FLAGS, STATS = OUT / "flags.npz", OUT / "part4b_stats.npz"


@torch.no_grad()
def flags() -> dict:
    """The cards' flag, for every latent (the cards kept only their top-20 lists): flagged iff the
    strongest galaxy-level |ρ| is with a nuisance or brightness variable rather than a vote fraction."""
    import r_nonlinear as R
    from dd_sae_cards import _corr_masked, _panel, _token_stats
    from dd_sae_score import galaxy_acts
    from j4_spread_controls import prepare

    sae = S.load(D.TAG, LAYER, MULT)
    ids, st = D.stamps("sae_eval")
    tok = np.load(S.TokenStore.path(D.TAG, LAYER, "sae_eval"), mmap_mode="r")
    rng = np.random.default_rng(20260925)  # the cards' subsample, so token flux ρ is theirs exactly
    sub = np.sort(rng.choice(len(tok), 200_000, replace=False))
    flux_tok = np.array([float(np.asarray(st[t // 256][1], np.float64)[(t % 256) // 16 * 16:(t % 256) // 16 * 16 + 16,
                                                                         (t % 256) % 16 * 16:(t % 256) % 16 * 16 + 16].sum())
                         for t in sub])
    dens, _, _, _, tok_flux = _token_stats(sae, tok, flux_tok, sub)
    acts = galaxy_acts(sae, tok)
    lab = prepare(None, R.MAX_TRAIN, label="DD4b", sources=1).labels
    rv = []
    for f in lab.features:
        el = set(lab.eligible(f, [int(i) for i in ids]))
        v = lab.vote_fraction(f, [int(i) for i in ids]).astype(float)
        v[[k for k, i in enumerate(ids) if int(i) not in el]] = np.nan
        rv.append(_corr_masked(acts, v))
    nuis = _panel(ids)
    nuis["brightness (galaxy r flux)"] = np.array([float(np.asarray(s[1], np.float64).sum()) for s in st])
    rn = [_corr_masked(acts, v) for v in nuis.values()] + [tok_flux]
    best_v, best_n = np.abs(np.stack(rv)).max(0), np.abs(np.stack(rn)).max(0)
    np.savez(FLAGS, density=dens, best_vote=best_v, best_nuisance=best_n, flagged=best_n > best_v)
    return {"flagged_live": int((best_n > best_v)[dens >= LIVE].sum()), "live": int((dens >= LIVE).sum())}


@torch.no_grad()
def stats() -> dict:
    """Galaxy-mean activations Ā, the energy matrix C, and each latent's max |ρ| with the four band offsets."""
    from dd_sae_cards import _corr_masked, _panel
    sae = S.load(D.TAG, LAYER, MULT).to(D.DEVICE)
    ids, _ = D.stamps("sae_eval")
    tok = np.load(S.TokenStore.path(D.TAG, LAYER, "sae_eval"), mmap_mode="r")
    L = sae.W_enc.shape[0]
    ztz = torch.zeros((L, L), device=D.DEVICE)
    abar = []
    for g in range(0, len(tok) // 256, 256):
        z = sae.encode(torch.from_numpy(np.asarray(tok[g * 256:(g + 256) * 256], np.float32)).to(D.DEVICE))
        ztz += z.T @ z
        abar.append(z.reshape(-1, 256, L).mean(1).cpu().numpy())
    w = sae.W_dec.detach()
    gram = (w.T @ w) / sae.scale ** 2
    c = (ztz / len(tok) * gram).cpu().double().numpy()
    abar = np.concatenate(abar)
    pan = _panel(ids)
    off_rho = np.abs(np.stack([_corr_masked(abar, pan[n]) for n in ("g-r x", "g-r y", "i-r x", "i-r y")])).max(0)
    np.savez(STATS, abar=abar, C=c, offset_rho=off_rho)
    return {"n_tokens": len(tok), "galaxies": len(abar)}


def _offset_set() -> list[int]:
    score = json.loads((OUT / "score.json").read_text())
    return sorted({t["latent"] for t in score["S1"]["b11"]["top"]})


def _energy(c: np.ndarray, s) -> float:
    s = np.asarray(s)
    return float(c[np.ix_(s, s)].sum())


def pools() -> tuple[np.ndarray, np.ndarray]:
    """Primary: live, not in the offset set, max |ρ| with every band offset < 0.3 (flags ignored).
    Secondary (exploratory): the cards' non-flagged live latents, not in the offset set."""
    f, st = np.load(FLAGS), np.load(STATS)
    dens, off = f["density"], _offset_set()
    live = (dens >= LIVE) & ~np.isin(np.arange(len(dens)), off)
    return np.where(live & (st["offset_rho"] < OFF_RHO_MAX))[0], np.where(live & ~f["flagged"])[0]


def energy_sets(seed: int = SEED) -> dict:
    """Rejection sampling for sets of 40 within ±15% of the offset set's removed energy. Proposal: 40
    distinct pool latents drawn with probability ∝ C_jj^0.5 (α set from the energies alone so the median
    proposal sits near the target; per-latent pairing cannot work, as the offset set's four densest
    latents each carry more energy than any pool latent). Acceptance is on the set's total energy."""
    c = np.load(STATS)["C"]
    primary, _ = pools()
    off = _offset_set()
    target = _energy(c, off)
    w = np.diag(c)[primary] ** ALPHA
    w /= w.sum()
    rng = np.random.default_rng(seed)
    sets, tried = [], 0
    while len(sets) < N_SETS and tried < MAX_PROPOSALS:
        tried += 1
        s = sorted(int(j) for j in rng.choice(primary, len(off), replace=False, p=w))
        if abs(_energy(c, s) / target - 1) <= E_TOL:
            sets.append(s)
    return {"target_energy": target, "sets": sets, "proposals": tried,
            "energy_ratio": [_energy(c, s) / target for s in sets]}


def density_sets(seed: int = SEED) -> dict:
    """Secondary: the non-flagged pool, each offset latent matched within ×3 in density (the loosest band
    at which the draw is feasible); a draw that runs out of candidates is redrawn."""
    dens, c = np.load(FLAGS)["density"], np.load(STATS)["C"]
    _, pool = pools()
    off = _offset_set()
    cand = [pool[(dens[pool] >= dens[j] / 3) & (dens[pool] <= dens[j] * 3)] for j in off]
    rng = np.random.default_rng(seed + 1)
    sets, tried = [], 0
    while len(sets) < N_SETS and tried < 10_000:
        tried += 1
        used: list[int] = []
        for j in rng.permutation(len(off)):
            free = [int(p) for p in cand[j] if p not in used]
            if not free:
                break
            used.append(int(rng.choice(free)))
        if len(used) == len(off):
            sets.append(sorted(used))
    t = _energy(c, off)
    return {"sets": sets, "proposals": tried, "energy_ratio": [_energy(c, s) / t for s in sets]}


def jaccard(sets: list[list[int]]) -> float:
    ss = [set(s) for s in sets]
    return float(np.mean([len(a & b) / len(a | b) for i, a in enumerate(ss) for b in ss[i + 1:]]))


def pool_report() -> dict:
    primary, secondary = pools()
    es, ds = energy_sets(), density_sets()
    return {"primary_pool": len(primary), "secondary_pool": len(secondary), "offset_set": len(_offset_set()),
            "energy_sets": len(es["sets"]), "energy_proposals": es["proposals"],
            "energy_ratio_range": [min(es["energy_ratio"], default=np.nan), max(es["energy_ratio"], default=np.nan)],
            "energy_jaccard": jaccard(es["sets"]) if len(es["sets"]) > 1 else None,
            "density_sets": len(ds["sets"]), "density_jaccard": jaccard(ds["sets"]) if len(ds["sets"]) > 1 else None,
            "density_energy_ratio_median": float(np.median(ds["energy_ratio"])) if ds["sets"] else None}


# ── ablation and read-outs ─────────────────────────────────────────────────────────────────────────

class Bench:
    """Everything an ablation is read against: pooled block-11 embeddings, the 37 probes, the offset probe."""

    def __init__(self) -> None:
        import r_nonlinear as R
        from dd_part4 import _pooled_from_tokens, offset_probe
        from dd_part2 import offsets
        from j4_spread_controls import prepare
        self.setup = prepare(None, R.MAX_TRAIN, label="DD4b", sources=1)
        self.probes = D.probe_readout(self.setup, "real")
        self.off_ro, _ = offset_probe()
        self.ids, _ = D.stamps("sae_eval")
        self.base = _pooled_from_tokens("sae_eval").astype(np.float64)
        sae = S.load(D.TAG, LAYER, MULT)
        self.w_dec = sae.W_dec.detach().double().numpy()
        self.scale = float(sae.scale)
        self.abar = np.load(STATS)["abar"].astype(np.float64)
        off = offsets(self.ids)
        self.off_rows = [k for k, o in enumerate(self.ids) if int(o) in off]
        self.y_off = np.array([off[int(self.ids[k])][2] > 0 for k in self.off_rows], int)
        self.__rows, self.__y = {}, {}
        for f in self.probes.names:
            el = self.setup.labels.eligible(f, [int(i) for i in self.ids])  # as dd_part4.morph_aucs
            yy = np.asarray(self.setup.labels.binary_label(f, el))
            if len(np.unique(yy)) == 2:
                s = set(el)
                self.__rows[f] = [k for k, o in enumerate(self.ids) if int(o) in s]
                self.__y[f] = yy
        self.answers = list(self.__rows)
        base = self.aucs(self.base)
        # Standing rule (interp_tooling.md): only powered answers can trip a collateral bound.
        self.powered = [f for f in self.answers
                        if min(int(self.__y[f].sum()), int((1 - self.__y[f]).sum())) >= POWER_MIN_CLASS
                        and base[f] >= POWER_MIN_AUC]

    def delta(self, s) -> np.ndarray:
        s = np.asarray(s)
        return self.abar[:, s] @ self.w_dec[:, s].T / self.scale

    def aucs(self, pooled: np.ndarray) -> dict[str, float]:
        from sklearn.metrics import roc_auc_score
        sc = self.probes.scores(pooled)
        out = {f: float(roc_auc_score(self.__y[f], sc[self.__rows[f], self.probes.names.index(f)])) for f in self.answers}
        out["__offset__"] = float(roc_auc_score(self.y_off, self.off_ro.scores(pooled[self.off_rows])[:, 0]))
        return out


def read(bench: Bench, base_auc: dict, pooled: np.ndarray) -> dict:
    a = bench.aucs(pooled)
    d = {f: a[f] - base_auc[f] for f in bench.answers}
    pw = np.abs([d[f] for f in bench.powered])
    return {"dauc": d, "max_abs": float(pw.max()), "mean_abs": float(pw.mean()),
            "worst": bench.powered[int(pw.argmax())], "offset_auc": a["__offset__"]}


def verdict(test: dict, nulls: list[dict]) -> dict:
    """SPECIFIC iff the tested set's max |ΔAUC| and its mean |ΔAUC| over the powered answers each exceed
    the random sets' 95th percentiles of the same statistics. Otherwise GENERIC. INSUFFICIENT under MIN_SETS random sets. Precedence in that
    order: INSUFFICIENT, then SPECIFIC, then GENERIC."""
    if len(nulls) < MIN_SETS:
        return {"state": "INSUFFICIENT", "n_sets": len(nulls)}
    p_max = float(np.percentile([n["max_abs"] for n in nulls], 95))
    p_mean = float(np.percentile([n["mean_abs"] for n in nulls], 95))
    a, m = test["max_abs"] > p_max, test["mean_abs"] > p_mean
    return {"state": "SPECIFIC" if a and m else "GENERIC", "max_above": a, "mean_above": m,
            "max_p95": p_max, "mean_p95": p_mean, "n_sets": len(nulls)}


def answer_bands(test: dict, nulls: list[dict], powered: list[str]) -> dict:
    """Exploratory, per answer: the tested set's signed ΔAUC against the random sets' 2.5–97.5% band."""
    out = {}
    for f in test["dauc"]:
        v = np.array([n["dauc"][f] for n in nulls])
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[f] = {"test": test["dauc"][f], "lo": float(lo), "hi": float(hi), "median": float(np.median(v)),
                  "outside": bool(test["dauc"][f] < lo or test["dauc"][f] > hi), "powered": f in powered}
    return out


def plants() -> dict:
    """D28 through the identical path. (1) SPECIFIC plant: the offset set's removal plus 0.75 of the
    morphology-probe subspace replaced by another galaxy's (a permutation), which must beat the random
    band on both statistics. (2) GENERIC plant: each random set in turn tested against the other 49;
    the SPECIFIC rate must stay ≤ 10% (nominal ≤ 5% per criterion; joint lower). (3) Saturation: the
    random sets' statistics must not sit at 0, and the offset probe must stay above chance at baseline."""
    b = Bench()
    base_auc = b.aucs(b.base)
    es = energy_sets()
    nulls = [read(b, base_auc, b.base - b.delta(s)) for s in es["sets"]]
    w = b.probes.w  # (D, 37) in pooled space
    q, _ = np.linalg.qr(w)
    perm = np.random.default_rng(SEED + 7).permutation(len(b.base))
    mix = (b.base - b.base[perm]) @ q @ q.T
    off = _offset_set()
    planted = read(b, base_auc, b.base - b.delta(off) - PLANT_LAMBDA * mix)
    loo = [verdict(nulls[i], nulls[:i] + nulls[i + 1:])["state"] for i in range(len(nulls))]
    rate = loo.count("SPECIFIC") / len(loo)
    # Blind: the tested set's statistics are already known from Part 4, so the null's quantiles would
    # give the verdict away before the hash. Only states and booleans leave this function.
    sat = (min(n["mean_abs"] for n in nulls) > 0) and (0.55 < base_auc["__offset__"] < 1.0)
    out = {"n_sets": len(nulls), "energy_proposals": es["proposals"],
           "SPECIFIC plant": verdict(planted, nulls)["state"],
           "GENERIC plant (leave-one-out SPECIFIC rate)": rate,
           "not saturated (random mean |dAUC| > 0; offset probe off chance and off ceiling)": sat}
    out["fire_as_expected"] = out["SPECIFIC plant"] == "SPECIFIC" and rate <= 0.10 and sat
    (OUT / "part4b_plants.json").write_text(json.dumps(out, indent=1))
    return out


def run() -> dict:
    b = Bench()
    base_auc = b.aucs(b.base)
    off = _offset_set()
    test = read(b, base_auc, b.base - b.delta(off))
    es, ds = energy_sets(), density_sets()
    nulls = [read(b, base_auc, b.base - b.delta(s)) for s in es["sets"]]
    sec = [read(b, base_auc, b.base - b.delta(s)) for s in ds["sets"]]
    out = {"tested": {k: v for k, v in test.items() if k != "dauc"},
           "primary": {**verdict(test, nulls), "energy_proposals": es["proposals"],
                       "energy_ratio_range": [min(es["energy_ratio"]), max(es["energy_ratio"])],
                       "jaccard": jaccard(es["sets"]),
                       "random offset-probe AUC range": [min(n["offset_auc"] for n in nulls), max(n["offset_auc"] for n in nulls)]},
           "primary_answers": answer_bands(test, nulls, b.powered),
           "secondary (exploratory, no verdict)": {
               **{k: v for k, v in verdict(test, sec).items() if k != "state"},
               "would_read": verdict(test, sec)["state"], "jaccard": jaccard(ds["sets"]),
               "energy_ratio_median": float(np.median(ds["energy_ratio"]))},
           "secondary_answers": answer_bands(test, sec, b.powered),
           "sets": {"primary": es["sets"], "secondary": ds["sets"]}, "powered": b.powered}
    # consistency with Part 4's hook path (all answers, as Part 4 read them)
    allabs = np.abs(list(test["dauc"].values()))
    out["part4_consistency"] = {"mean_abs_all": float(allabs.mean()), "max_abs_all": float(allabs.max()),
                                "part4_hook": {"mean_abs_all": 0.010174503924306117, "max_abs_all": 0.057005249477600606}}
    # Part 4 note, post hoc (D27): latent 328 alone under the powered-answer rule
    l328 = read(b, base_auc, b.base - b.delta([328]))
    out["latent_328_powered (post hoc)"] = {k: l328[k] for k in ("max_abs", "mean_abs", "worst", "offset_auc")}
    out["state"] = out["primary"]["state"]
    (OUT / "part4b.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    fn = {"flags": flags, "stats": stats, "pools": pool_report, "plants": plants, "run": run}[cmd]
    r = fn()
    if cmd == "run":
        r = {k: v for k, v in r.items() if not k.endswith("answers") and k != "sets"}
    print(json.dumps(r, indent=1, default=float))
