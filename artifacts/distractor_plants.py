"""Distractor sensitivity (Kickoff B task 3): machinery and D28 plants.

How much of each encoder's representation is spent on instrumental nuisance rather than morphology?
Pre-registration: `artifacts/distractor_sensitivity.md`. The JEPA pair is M (c2_m1 bank) and O2
(c2_m2 bank); MAE and MoCo do not exist yet, so every "baseline" below is planted from M and O2. The
untrained floor is O1's bank (M's architecture, seed 0; capped train + test, 74,829 galaxies), with Brief
R's seeds 1 and 2 on the same galaxies (a range, no state). Every readability state is read twice: the
primary index over the seven, and the secondary over the six without PSF, each with its own fixed MDD.
Variable 7 is `sky_r` (PhotoObjAll, pulled once by `pull-sky`, pinned by its record); `snr_r` is reported.

Measures (i) and (ii) read the DD cap (40,000 train, H5's stride) and the whole probe-test split, the
galaxies N1's nuisance panel and criteria 1 and 3 read. (iii) is the headline protocol (full train,
`aligned_c2.fit_scores`) and needs full-split banks, so it is not in the plants.

  uv run python artifacts/distractor_plants.py pull-sky          # sky_r for the capped train + test, once
  uv run python artifacts/distractor_plants.py plants [--limit N] [--reps R] [--out DIR]
  uv run python artifacts/distractor_plants.py embed-untrained   # full-split untrained bank, for (iii)
  uv run python artifacts/distractor_plants.py morph [tag ...]    # (iii), context, no state
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import aligned_c2  # noqa: E402
from aligned_c2 import _bank, auc, boot_aucs, fit_scores, shifted, statistic, weights  # noqa: E402
from aligned_c13 import (OFFSETS, SEED, _load, _r_flux, c1_arrays, families,  # noqa: E402
                         representation, setup)
from j4_spread_controls import OUT, _capped_train  # noqa: E402
from probe_bank import load_bank, untrained_descriptor, write_bank  # noqa: E402

N_BOOT = 2_000  # criterion 2's paired Poisson bootstrap; the comparison needs draws shared across encoders
MIN_TEST, MIN_CLASS = 500, 100  # criterion 1's INSUFFICIENT floor; the powered-answer class floor
CHANCE_HALF, CI_BAND = 0.05, (0.40, 0.60)  # 1a's bounds, for the per-encoder readability label only
CONTINUOUS = ("psf", "magnitude", "sky")  # median split, as N1's panel (`nuisance_label`)
NUISANCE = OFFSETS + CONTINUOUS  # the seven the index averages
SECONDARY = tuple(n for n in NUISANCE if n != "psf")  # the secondary index: PSF carries JEPA's seed noise
REPORTED = ("snr", "redshift", "size")  # snr: declared secondary (slot 7 before sky_r); N1's physical pair
SKY_BATCH = 100  # IDs per SkyServer call: 400-ID IN-lists failed to connect on 2026-10-01, 100-ID ones went through
SKY_SQL = "SELECT CAST(p.objID AS varchar(20)) AS objID, p.sky_r FROM PhotoObjAll p WHERE p.objID IN ({})"
SKY_CSV, SKY_RECORD = OUT / "distractor_sky_r.csv", OUT / "distractor_sky_r.json"
INJECT_SHARE, MORPH_SHARE = 0.50, 0.20  # embedding plants' variance shares (gross, by design: logic, not power)
DELTA, DELTA_POWER = 0.03, 0.02  # score plants: required shift; the shift reported for power only
DISAGREE_MARGIN = 2  # SEEDS DISAGREE: each seed beyond the JEPA mean by 2 s_m (1 s_m false-called the null)
MATERIAL = 0.10  # (ii): |D| on a top-10 share must reach 10 points (the seed-range rule alone false-calls 1/3)
Z95 = 1.96  # the power check's multiplier: minimum detectable D = S_J + 1.96 sd(D)
STATE = {"BETTER": "MORE", "WORSE": "LESS", "SAME": "SAME", "UNRESOLVED": "UNRESOLVED"}


# ── measure (i): readability ────────────────────────────────────────────────────────────────────────

def _value(d: dict, name: str, ids) -> np.ndarray:
    """A nuisance's raw value: `sky` from the pinned pull (NaN where missing or not positive), the rest
    from the corpus's `LabelProvider`."""
    if name == "sky":
        return np.array([d["sky"].get(int(i), np.nan) for i in ids])
    return d["s"].labels.nuisance_value(name, ids)


def _valid(d: dict, name: str, ids) -> np.ndarray:
    return np.ones(len(ids), bool) if name == "sky" else d["s"].labels.nuisance_valid(name, ids)


def _median_arrays(d: dict, x_tr: np.ndarray, x_te: np.ndarray, name: str):
    """N1's median split (`LabelProvider.nuisance_label`): each split at its own median, over usable
    rows. Same return shape as `aligned_c13.c1_arrays`."""
    out = []
    for ids, x in ((d["train"], x_tr), (d["test"], x_te)):
        v = _value(d, name, ids)
        ok = _valid(d, name, ids) & ~np.isnan(v)
        out.append((x[ok], (v[ok] >= np.median(v[ok])).astype(int), v[ok], ok))
    (a, ya, _, _), (b, yb, vb, okb) = out
    return a, ya, b, yb, vb, okb


def _arrays(d, x_tr, x_te, name):
    return c1_arrays(d, x_tr, x_te, name) if name in OFFSETS else _median_arrays(d, x_tr, x_te, name)


def probe(d: dict, x_tr: np.ndarray, x_te: np.ndarray, names=NUISANCE) -> dict:
    """Per variable: test scores of the one fit path (`_fit`: standardised L2 logistic, C = 1 — 1a's
    probe and N1's), the binary target and the test-row mask."""
    from galaxy_jepa.probing.logistic import Embeddings, _fit
    out = {}
    for name in names:
        xtr, ytr, xte, yte, _, keep = _arrays(d, x_tr, x_te, name)
        sc, clf = _fit(Embeddings(xtr, ytr, np.zeros(len(ytr))), c=1.0)
        out[name] = {"s": clf.decision_function(sc.transform(xte)), "y": yte, "keep": keep}
    return out


def _boot(s: np.ndarray, y: np.ndarray, w: torch.Tensor, cols: np.ndarray) -> np.ndarray:
    """`boot_aucs` in blocks of 250 draws: the full B × n temporaries do not fit beside the rehearsal."""
    c = torch.from_numpy(cols)
    return np.concatenate([boot_aucs(s, y, w[a:a + 250][:, c]) for a in range(0, w.shape[0], 250)])


def readout(sc: dict, w: torch.Tensor, rows: np.ndarray | None = None) -> dict:
    """AUC, paired-bootstrap draws and the index NI = mean_v (AUC_v − 0.5). `rows` restricts the test
    split (the thin-n plants); None is the whole split."""
    per = {}
    for name, r in sc.items():
        cols = np.flatnonzero(r["keep"])  # test-split positions of this variable's rows
        sel = np.ones(len(cols), bool) if rows is None else np.isin(cols, rows)
        s, y = r["s"][sel], r["y"][sel]
        a = auc(s, y) if len(np.unique(y)) == 2 else 0.5
        bs = _boot(s, y, w, cols[sel]) if len(np.unique(y)) == 2 else np.full(w.shape[0], 0.5)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        lab = ("NEAR CHANCE" if abs(a - 0.5) <= CHANCE_HALF and CI_BAND[0] <= lo and hi <= CI_BAND[1]
               else "READABLE" if lo > CI_BAND[1] or hi < CI_BAND[0] else "UNRESOLVED")
        per[name] = {"auc": a, "ci": [float(lo), float(hi)], "n_test": int(len(y)),
                     "min_class": int(min(y.sum(), len(y) - y.sum())), "label_1a": lab, "_b": bs, "_s": s, "_y": y}
    if not any(n in per for n in NUISANCE):  # the reported-only pair carries no index
        return {"per": per}
    ni = float(np.mean([per[n]["auc"] - 0.5 for n in NUISANCE if n in per]))
    nib = np.mean([per[n]["_b"] - 0.5 for n in NUISANCE if n in per], 0)
    return {"per": per, "NI": ni, "_NIb": nib, "NI_ci": [float(v) for v in np.percentile(nib, [2.5, 97.5])]}


def _ni(r: dict, names=NUISANCE) -> float:
    return float(np.mean([r["per"][n]["auc"] - 0.5 for n in names]))


def _nib(r: dict, names=NUISANCE) -> np.ndarray:
    return np.mean([r["per"][n]["_b"] - 0.5 for n in names], 0)


def min_detectable(rm: tuple[dict, dict], names=NUISANCE) -> dict:
    """The power check (criterion 2's realistic width), from the JEPA pair alone: J1 and J2 are two
    genuinely independent draws, so sd(D) ≈ bootstrap sd(NI_J1 − NI_J2)/√2, and the minimum detectable
    D = S_J + 1.96 sd(D), with S_J = mean_v |J1_v − J2_v|. Nothing from a baseline enters, so it is fixed
    before any baseline is read; a CI that straddles ±it reads UNRESOLVED (`_straddles`)."""
    sd_pair = float((_nib(rm[0], names) - _nib(rm[1], names)).std())
    s_bar = float(np.mean([abs(rm[0]["per"][n]["auc"] - rm[1]["per"][n]["auc"]) for n in names]))
    return {"sd_NI_diff_independent_draws": sd_pair, "sd_D_approx": sd_pair / np.sqrt(2),
            "S_bar_assumed_JEPA": s_bar, "min_detectable_D": s_bar + Z95 * sd_pair / np.sqrt(2)}


def _straddles(lo: float, hi: float, mdd: float) -> bool:
    """The CI straddles the detectable threshold: lo < MDD < hi, or lo < −MDD < hi (user, 2026-09-28)."""
    return lo < mdd < hi or lo < -mdd < hi


def compare_readability(rb: list[dict], rm: tuple[dict, dict], mdd: float, names=NUISANCE) -> dict:
    """States of measure (i) for one baseline B (one or two seeds) against the JEPA pair.

    Two seeds: criterion 2's statistic unchanged (`aligned_c2.statistic`, the variables in place of the
    answers): D̄ = mean_v D_v = NI_B − NI_JEPA, pooled bar S̄ = mean_v S_v, 95% CI from the shared Poisson
    bootstrap; per variable, the shrunk bar B_v = (S_v + S̄)/2 with BY at q = 0.05. Precedence, index and
    per variable alike: INSUFFICIENT, then UNRESOLVED when the CI straddles ±`mdd` (the fixed minimum
    detectable D; `_straddles`), whatever the bar, then SEEDS DISAGREE, then the statistic's MORE / LESS / SAME, then
    LEANS MORE / LEANS LESS where the CI excludes 0 but is not beyond the bar, then UNRESOLVED.
    `names` is the index's variable set; per-variable states are the primary index's only."""
    thin = [n for n in names if any(r["per"][n]["n_test"] < MIN_TEST or r["per"][n]["min_class"] < MIN_CLASS
                                    for r in (*rb, *rm))]
    ni_m = (_ni(rm[0], names), _ni(rm[1], names))
    m_mean, s_m = sum(ni_m) / 2, abs(ni_m[0] - ni_m[1])
    d_e = [_ni(r, names) - m_mean for r in rb]
    auc_ = {n: [r["per"][n]["auc"] for r in (*rb, *rm)] for n in names}
    primary = tuple(names) == NUISANCE
    straddle_v: list[str] = []
    if len(rb) == 1:  # JEPA's pooled spread alone is the bar; the baseline's own seed noise is unknown
        d = d_e[0]
        s = float(np.mean([abs(a[1] - a[2]) for a in auc_.values()]))
        db = _nib(rb[0], names) - (_nib(rm[0], names) + _nib(rm[1], names)) / 2
        lo, hi = (float(v) for v in np.percentile(db, [2.5, 97.5]))
        fam = "MORE" if lo > s else "LESS" if hi < -s else "SAME" if lo >= -s and hi <= s else "UNRESOLVED"
        per = {n: "NO STATE (one seed)" for n in names} if primary else {}
    else:
        sc4, meta = {}, {"y": {n: rm[0]["per"][n]["_y"] for n in names}, "rows": {}}
        for e, r in zip(("a1", "a2", "m1", "m2"), (*rb, *rm), strict=True):
            sc4[e] = {n: r["per"][n]["_s"] for n in names}
            aligned_c2._boot_cache[e] = np.stack([r["per"][n]["_b"] for n in names], 1)
        try:
            st = statistic(sc4, meta, list(names), None, None)
        finally:
            aligned_c2._boot_cache.clear()
        d, s, (lo, hi) = st["D_bar"], st["S_bar"], st["ci"]
        fam = STATE[st["state"]]
        per = {}
        for n in names if primary else ():
            a = auc_[n]
            dv = [a[0] - (a[2] + a[3]) / 2, a[1] - (a[2] + a[3]) / 2]
            lv, hv = st["per_answer"][n]["ci"]
            if _straddles(lv, hv, mdd):
                per[n] = "UNRESOLVED"
                straddle_v.append(n)
            elif dv[0] * dv[1] < 0 and min(map(abs, dv)) > DISAGREE_MARGIN * abs(a[2] - a[3]):
                per[n] = "SEEDS DISAGREE"
            else:
                per[n] = STATE[st["per_answer"][n]["state_shrunk_bar"]]
    if fam == "UNRESOLVED" and lo > 0:
        fam = "LEANS MORE"
    elif fam == "UNRESOLVED" and hi < 0:
        fam = "LEANS LESS"
    straddle = _straddles(lo, hi, mdd)
    if thin:
        fam = "INSUFFICIENT"
        per |= dict.fromkeys(thin if primary else [], "INSUFFICIENT")
    elif straddle:
        fam = "UNRESOLVED"
    elif len(rb) == 2 and d_e[0] * d_e[1] < 0 and min(map(abs, d_e)) > DISAGREE_MARGIN * s_m:
        fam = "SEEDS DISAGREE"
    if len(rb) == 1 and fam != "INSUFFICIENT":
        fam += " (PROVISIONAL, one seed)"
    counter = [n for n, v in per.items() if (v == "LESS" and fam.startswith("MORE")) or (v == "MORE" and fam.startswith("LESS"))]
    return {"index": fam, "D": float(d), "S": float(s), "ci": [float(lo), float(hi)], "ci_half_width": (hi - lo) / 2,
            "min_detectable_D": mdd, "unresolved_ci_straddles_mdd": bool(straddle and not thin),
            "per_variable_unresolved_ci_straddles_mdd": straddle_v, "NI_B": [_ni(r, names) for r in rb],
            "per_variable": per, "counter_direction": counter, "insufficient": thin}


# ── measure (ii): top-10 PC shares ──────────────────────────────────────────────────────────────────

def nuisance_families(d: dict) -> dict:
    """Criterion 3's families plus 'observing' (PSF, sky): the nuisance family is offsets ∪ brightness ∪
    observing, the seven of (i) plus criterion 3's total r flux."""
    return families(d) | {"observing": [_value(d, "psf", d["test"]), _value(d, "sky", d["test"])]}


def shares(x: np.ndarray, fam: dict) -> dict:
    r = representation(x, fam)
    n = sum(p["share"] for p in r["pcs"] if p["tracks"] in ("offsets", "brightness", "observing"))
    return {"N": n, "Mo": r["Mo"], "PR": r["PR"],
            "pcs": [(p["pc"], round(p["share"], 3), p["top_family"], round(p["rho"][p["top_family"]], 2), p["tracks"])
                    for p in r["pcs"]]}


def share_state(b: list[float], m: tuple[float, float]) -> str:
    """Criterion 3's seed-range rule with a materiality floor. Under an exchangeable null of four seeds
    'both B beyond both JEPA' happens 1 time in 3 (1/6 each way), so direction alone is not a state:
    MORE needs both B seeds above both JEPA seeds **and** D = mean(B) − mean(JEPA) ≥ 0.10; SPLIT needs
    one seed above both and one below both, each ≥ 0.10 from the JEPA mean; UNRESOLVED is a
    seed-range call short of 0.10 in the other; SAME otherwise. One seed: B beyond both and |D| ≥ 0.10."""
    mean = (m[0] + m[1]) / 2
    if len(b) == 2 and max(b) > max(m) and min(b) < min(m) and min(abs(v - mean) for v in b) >= MATERIAL:
        return "SPLIT"
    d = float(np.mean(b)) - mean
    if min(b) > max(m):
        return "MORE" if d >= MATERIAL else "UNRESOLVED"
    if max(b) < min(m):
        return "LESS" if d <= -MATERIAL else "UNRESOLVED"
    return "SAME"


def compare_shares(rb: list[dict], rm: tuple[dict, dict]) -> dict:
    tag = " (PROVISIONAL, one seed)" if len(rb) == 1 else ""
    return {"nuisance_share": share_state([r["N"] for r in rb], (rm[0]["N"], rm[1]["N"])) + tag,
            "morphology_share": share_state([r["Mo"] for r in rb], (rm[0]["Mo"], rm[1]["Mo"])) + tag}


def _r_flux_pread(d: dict) -> np.ndarray:
    """`aligned_c13._r_flux`, read as one `os.pread` of the r plane per stamp in cache-row order: the
    memmap's page faults count every touched stamp into RSS (3.7 GB at 10,000 galaxies) and run slowly
    off the X10. Checked equal to `_r_flux` on the first 512 test galaxies in every run."""
    import os
    from f0_preconditions import check
    from galaxy_jepa.data.cache import _DATA_FILE
    _, cache = check(verbose=False)
    dt = np.dtype(cache.index.dtype)
    c, h, w = cache.index.shape
    plane = h * w * dt.itemsize
    rows = np.array([cache._row_of[i] for i in d["test"]])
    out = np.empty(len(rows))
    fd = os.open(cache.cache_dir / _DATA_FILE, os.O_RDONLY)
    try:
        for k in np.argsort(rows):
            buf = os.pread(fd, plane, int(rows[k]) * c * plane + plane)  # channel 1 = r
            out[k] = np.frombuffer(buf, dt).reshape(h, w).astype(np.float64).sum((0, 1))
    finally:
        os.close(fd)
    ref = _r_flux({"test": d["test"][:512]})
    if not np.allclose(out[:512], ref, rtol=1e-12, atol=0):
        raise SystemExit(f"pread r flux differs from _r_flux (max |Δ| {np.abs(out[:512] - ref).max()})")
    return out


# ── plants ──────────────────────────────────────────────────────────────────────────────────────────

def _z(v: np.ndarray) -> np.ndarray:
    v = np.where(np.isnan(v), np.nanmean(v), v)
    return (v - v.mean()) / v.std()


def _targets(d: dict, ids: list[int], pan: dict) -> np.ndarray:
    return np.stack([_z(pan[n]) if n in OFFSETS else _z(_value(d, n, ids)) for n in NUISANCE], 1)


def inject(x_tr, x_te, z_tr, z_te, share: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """One random orthonormal direction per column of z, each carrying that standardised variable; together
    they hold `share` of the new total variance (scale fixed on train, applied to test)."""
    q, _ = np.linalg.qr(rng.standard_normal((x_tr.shape[1], z_tr.shape[1])))
    tot = np.var(x_tr, axis=0).sum()
    a = np.sqrt(share / (1 - share) * tot / z_tr.shape[1])
    return x_tr + a * z_tr @ q.T, x_te + a * z_te @ q.T


def erase(x_tr, x_te, y_tr) -> tuple[np.ndarray, np.ndarray]:
    """LEACE (Belrose et al. 2023), fitted on train: no linear predictor keeps covariance with any column
    of y_tr (the seven binary probe targets; missing → the column mean)."""
    mu = x_tr.mean(0)
    xc = x_tr - mu
    lam, v = np.linalg.eigh(np.cov(xc, rowvar=False))
    lam = np.maximum(lam, lam.max() * 1e-10)
    wh, unwh = (v / np.sqrt(lam)) @ v.T, (v * np.sqrt(lam)) @ v.T
    yc = y_tr - np.nanmean(y_tr, 0)
    yc = np.where(np.isnan(yc), 0.0, yc)
    q, _ = np.linalg.qr(wh @ (xc.T @ yc / len(xc)))
    p = wh @ q @ q.T @ unwh
    return x_tr - xc @ p, x_te - (x_te - mu) @ p


def _binary_targets(d: dict, ids: list[int], pan: dict) -> np.ndarray:
    cols = []
    for n in NUISANCE:
        v = pan[n] if n in OFFSETS else _value(d, n, ids)
        b = v > 0 if n in OFFSETS else v >= np.nanmedian(v)  # PSF and magnitude have no NaN: nanmedian = median
        cols.append(np.where(np.isnan(v), np.nan, b.astype(float)))
    return np.stack(cols, 1)


def _public(r: dict) -> dict:
    return {"NI": r["NI"], "NI_ci": r["NI_ci"],
            "per": {n: {k: v for k, v in p.items() if not k.startswith("_")} for n, p in r["per"].items()}}


def _score_plant(sc_m: dict, delta: float, eps_sd: dict, rng, noise: float = 0.0, rows=None) -> dict:
    """Criterion 2's score plant: each variable's AUC moved to M_e + Δ + ε via `shifted`. `noise` adds
    independent per-galaxy score noise first (in units of the score sd), decorrelating B's errors from M's."""
    out = {}
    for n, r in sc_m.items():
        s, y = r["s"], r["y"]
        if rows is not None:  # the target AUC is M's on the rows the plant is read on
            cols = np.flatnonzero(r["keep"])
            sel = np.isin(cols, rows)
            base = auc(s[sel], y[sel])
        else:
            base = auc(s, y)
        s0 = s + noise * s.std() * rng.standard_normal(len(s)) if noise else s
        tgt = float(np.clip(base + delta + rng.normal(0, eps_sd[n]), 0.5, 0.999)) if eps_sd[n] else \
            float(np.clip(base + delta, 0.5, 0.999))
        out[n] = r | {"s": shifted(s0, y, tgt) if rows is None else _shift_rows(s0, y, tgt, r["keep"], rows)}
    return out


def _shift_rows(s, y, tgt, keep, rows):
    """`shifted`, with the target hit on the plant's rows (the thin-n plants)."""
    cols = np.flatnonzero(keep)
    sel = np.isin(cols, rows)
    sd = s[sel].std()
    lo, hi = -5.0, 5.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if auc(s[sel] + mid * sd * y[sel], y[sel]) < tgt:
            lo = mid
        else:
            hi = mid
    return s + (lo + hi) / 2 * sd * y


def plants(limit: int, reps: int, out_dir: Path) -> dict:
    t0 = time.perf_counter()
    d = setup()
    if limit:  # dev: H5's deterministic stride on both splits
        from dd_sae_cards import _panel
        d["train"], d["test"] = _capped_train(d["train"], limit), _capped_train(d["test"], limit)
        d["pan_train"], d["pan_test"] = _panel(np.array(d["train"])), _panel(np.array(d["test"]))
    d["r_flux"] = _r_flux_pread(d)
    d["sky"], sky_rec = _sky()
    fam = nuisance_families(d)
    rng = np.random.default_rng(SEED)
    w, _ = weights(d["test"], b=N_BOOT)
    x = {t: (_load(t, d["train"]), _load(t, d["test"])) for t in ("m1", "m2")}
    o1 = load_bank(OUT / "o1_embeddings.npz")
    pos = {int(o): k for k, o in enumerate(o1["ids"])}
    x["untrained"] = tuple(o1["untrained"][[pos[i] for i in ids]].astype(np.float64) for ids in (d["train"], d["test"]))
    # untrained seeds 1 and 2 (Brief R's bank): with O1's seed 0, three draws, reported as a range
    us = load_bank(OUT / "r_untrained_seeds.npz")
    ids_ok = {"seed_bank_ids_equal_o1": bool(np.array_equal(us["ids"], o1["ids"])),
              "o1_ids_equal_capped_train_plus_test": sorted(map(int, o1["ids"])) == sorted(d["train"] + d["test"])}
    if not limit and not all(ids_ok.values()):
        raise SystemExit(f"untrained seed bank is not over the capped train + test: {ids_ok}")
    for k in (1, 2):
        x[f"untrained_s{k}"] = tuple(us[f"seed{k}"][[pos[i] for i in ids]].astype(np.float64)
                                     for ids in (d["train"], d["test"]))
    del o1, us
    out: dict = {"mode": f"DEV (subsample: {len(d['train']):,} train, {len(d['test']):,} test)" if limit else "FULL",
                 "n_train": len(d["train"]), "n_test": len(d["test"]), "n_boot": N_BOOT, "untrained_seed_ids": ids_ok,
                 "sky_r": {"output_sha1": sky_rec["output_sha1"],
                           "missing_train_test": [int(np.isnan(_value(d, "sky", d[k])).sum()) for k in ("train", "test")]},
                 "encoders": {}, "plants": {}}

    # the three real encoders: M1, M2 (the JEPA pair) and the untrained floor; N1's reported pair on each
    sc = {t: probe(d, *x[t], NUISANCE + REPORTED) for t in x}
    rd = {t: readout({n: v for n, v in sc[t].items() if n in NUISANCE}, w) for t in x}
    rep = {t: readout({n: v for n, v in sc[t].items() if n in REPORTED}, w)["per"] for t in x}
    sh = {t: shares(x[t][1], fam) for t in x}
    for t in x:
        out["encoders"][t] = _public(rd[t]) | {"reported_no_state": {n: {k: v for k, v in p.items() if not k.startswith("_")}
                                                                   for n, p in rep[t].items()},
                                              "shares": sh[t], "NI_secondary": _ni(rd[t], SECONDARY)}
    rm = (rd["m1"], rd["m2"])
    shm = (sh["m1"], sh["m2"])
    out["jepa_noise"] = {"NI_M1_minus_M2": rd["m1"]["NI"] - rd["m2"]["NI"],
                         "per_variable_M1_minus_M2": {n: rd["m1"]["per"][n]["auc"] - rd["m2"]["per"][n]["auc"] for n in NUISANCE},
                         "N_M1_M2": [shm[0]["N"], shm[1]["N"]], "Mo_M1_M2": [shm[0]["Mo"], shm[1]["Mo"]]}
    out["headroom"] = {n: 1 - max(rm[0]["per"][n]["auc"], rm[1]["per"][n]["auc"]) for n in NUISANCE} | \
                      {"NI": 0.5 - max(rm[0]["NI"], rm[1]["NI"])}
    u = ("untrained", "untrained_s1", "untrained_s2")
    out["untrained_floor"] = {
        ix: {"NI_seeds_0_1_2": [_ni(rd[t], names) for t in u],
             "range": [min(_ni(rd[t], names) for t in u), max(_ni(rd[t], names) for t in u)],
             "excess_NI_M1_M2_over_seed_range": [[_ni(rd[m], names) - max(_ni(rd[t], names) for t in u),
                                                  _ni(rd[m], names) - min(_ni(rd[t], names) for t in u)] for m in ("m1", "m2")]}
        for ix, names in (("primary", NUISANCE), ("secondary", SECONDARY))}
    out["untrained_floor"]["per_variable_auc_seeds_0_1_2"] = {n: [rd[t]["per"][n]["auc"] for t in u] for n in NUISANCE}
    out["untrained_floor"]["N_seeds_0_1_2"] = [sh[t]["N"] for t in u]
    # the power check fixes each index's minimum detectable D from the JEPA pair alone
    out["power"] = min_detectable(rm)
    out["power_secondary"] = min_detectable(rm, SECONDARY)
    mdd, mdd2 = out["power"]["min_detectable_D"], out["power_secondary"]["min_detectable_D"]
    s_bar, s_bar2 = out["power"]["S_bar_assumed_JEPA"], out["power_secondary"]["S_bar_assumed_JEPA"]
    # N1's panel (t01's rows = every capped-train and test galaxy): the same probe on the same galaxies
    n1 = json.loads((OUT / "n1_spread_controls.json").read_text())["features"][0]["nuisance_aucs"]
    ours = {n: rd["m1"]["per"][n]["auc"] for n in ("psf", "magnitude")} | {n: rep["m1"][n]["auc"] for n in REPORTED}
    out["reconcile_n1"] = {n: {"n1": n1[n], "here": ours[n], "diff": ours[n] - n1[n]} for n in ours}
    print(f"  base encoders {time.perf_counter() - t0:6.0f}s", file=sys.stderr)

    # embedding plants — the fit path end to end
    z_tr, z_te = _targets(d, d["train"], d["pan_train"]), _targets(d, d["test"], d["pan_test"])
    y_tr = _binary_targets(d, d["train"], d["pan_train"])
    dirty, clean = {}, {}
    for t in ("m1", "m2"):
        xi = inject(*x[t], z_tr, z_te, INJECT_SHARE, rng)
        dirty[t] = (readout(probe(d, *xi), w), shares(xi[1], fam))
        xe = erase(*x[t], y_tr)
        clean[t] = (readout(probe(d, *xe), w), shares(xe[1], fam))
    print(f"  inject/erase  {time.perf_counter() - t0:6.0f}s", file=sys.stderr)
    P = out["plants"]

    def rec(name, rb, sb, want_i, want_ii=None, **extra):
        r = {"readability": compare_readability(rb, rm, mdd) if rb else None,
             "readability_secondary": compare_readability(rb, rm, mdd2, SECONDARY) if rb else None,
             "shares": compare_shares(sb, shm) if sb else None, "must_read": {"index": want_i, "shares": want_ii}} | extra
        if rb:
            r["NI_B"] = [b["NI"] for b in rb]
        if sb:
            r["N_B"], r["Mo_B"] = [b["N"] for b in sb], [b["Mo"] for b in sb]
        got_i = r["readability"]["index"] if rb else None
        got_ii = r["shares"] if sb else None
        r["fires"] = (want_i is None or got_i == want_i) and (want_ii is None or all(got_ii[k] == v for k, v in want_ii.items()))
        if rb:  # the secondary index is read on the same plant and must read the same state
            r["fires_secondary"] = r["readability_secondary"]["index"] == want_i
        P[name] = r

    rec("inject: seven nuisance directions, 50% of variance", [dirty["m1"][0], dirty["m2"][0]],
        [dirty["m1"][1], dirty["m2"][1]], "MORE", {"nuisance_share": "MORE"},
        per_variable_auc={n: [dirty[t][0]["per"][n]["auc"] for t in ("m1", "m2")] for n in NUISANCE})
    rec("erase: LEACE on the seven binary targets", [clean["m1"][0], clean["m2"][0]],
        [clean["m1"][1], clean["m2"][1]], "LESS", {"nuisance_share": "LESS"},
        per_variable_auc={n: [clean[t][0]["per"][n]["auc"] for t in ("m1", "m2")] for n in NUISANCE})
    rec("swap: B = (M2, M1)", [rd["m2"], rd["m1"]], [sh["m2"], sh["m1"]], "SAME",
        {"nuisance_share": "SAME", "morphology_share": "SAME"})
    rec("split: B = (inject M1, erase M2)", [dirty["m1"][0], clean["m2"][0]], [dirty["m1"][1], clean["m2"][1]],
        "SEEDS DISAGREE", {"nuisance_share": "SPLIT"})
    rec("one seed: B = (inject M1)", [dirty["m1"][0]], [dirty["m1"][1]], "MORE (PROVISIONAL, one seed)",
        {"nuisance_share": "MORE (PROVISIONAL, one seed)"})
    # morphology-share plant: a component carrying the featured-or-disk fraction, 20% of variance
    feat = "t01_smooth_or_features_a02_features_or_disk"
    lab = d["s"].labels
    fz_tr, fz_te = _z(lab.vote_fraction(feat, d["train"]))[:, None], _z(lab.vote_fraction(feat, d["test"]))[:, None]
    mor = [shares(inject(*x[t], fz_tr, fz_te, MORPH_SHARE, rng)[1], fam) for t in ("m1", "m2")]
    rec("morphology: featured-or-disk direction, 20% of variance", None, mor, None, {"morphology_share": "MORE"})

    # the null does not saturate: every target permuted across galaxies (train and test alike)
    perm_tr, perm_te = rng.permutation(len(d["train"])), rng.permutation(len(d["test"]))
    xm = (x["m1"][0][perm_tr], x["m1"][1][perm_te])  # embeddings permuted against the fixed targets
    null = readout(probe(d, *xm), w)
    P["null: M1 embeddings permuted against the targets"] = _public(null) | {
        "must_read": "every variable NEAR CHANCE; NI CI contains 0",
        "fires": all(p["label_1a"] == "NEAR CHANCE" for p in null["per"].values()) and null["NI_ci"][0] <= 0 <= null["NI_ci"][1]}
    print(f"  embed plants  {time.perf_counter() - t0:6.0f}s", file=sys.stderr)

    # score plants — criterion 2's construction, through the identical comparison
    scm = {t: {n: v for n, v in sc[t].items() if n in NUISANCE} for t in ("m1", "m2")}
    sig = {n: float(abs(rd["m1"]["per"][n]["auc"] - rd["m2"]["per"][n]["auc"]) / np.sqrt(2)) for n in NUISANCE}
    out["score_plant_sigma"] = sig
    for name, delta, want in ((f"+{DELTA}", DELTA, "MORE"), (f"-{DELTA}", -DELTA, "LESS"), ("0", 0.0, "SAME"),
                              (f"+{DELTA_POWER} (power, no requirement)", DELTA_POWER, None)):
        got, got2, pv = [], [], []
        for _ in range(reps):
            rb = [readout(_score_plant(scm[t], delta, sig, rng), w) for t in ("m1", "m2")]
            c = compare_readability(rb, rm, mdd)
            got.append(c["index"])
            got2.append(compare_readability(rb, rm, mdd2, SECONDARY)["index"])
            pv.append(list(c["per_variable"].values()))
        pv = np.array(pv)  # reps × variables
        tgt = want or "MORE"
        P[f"score {name}, seed noise ε ~ N(0, σ_v), {reps} realisations"] = {
            "must_read": want, "states": {k: got.count(k) for k in set(got)}, "rate": got.count(tgt) / reps,
            "per_variable_rates": {st: float((pv == st).mean()) for st in
                                   ("MORE", "LESS", "SAME", "UNRESOLVED", "SEEDS DISAGREE")},
            "per_variable_rate": {n: float((pv[:, j] == tgt).mean()) for j, n in enumerate(NUISANCE)},
            "fires": want is None or got.count(want) / reps >= 0.95,
            "states_secondary": {k: got2.count(k) for k in set(got2)},
            "fires_secondary": want is None or got2.count(want) / reps >= 0.95}
    # (ii) under seed noise: N_B = N_Me + ε, ε ~ N(0, τ), τ = |N_M1 − N_M2|/√2 — the rule's own false-call
    # rate at Δ = 0 and its detection at Δ = +0.20, through `share_state`
    tau = abs(shm[0]["N"] - shm[1]["N"]) / np.sqrt(2)
    for name, delta, want in (("0", 0.0, "SAME"), ("+0.20", 0.20, "MORE")):
        got = [share_state([shm[0]["N"] + delta + rng.normal(0, tau), shm[1]["N"] + delta + rng.normal(0, tau)],
                           (shm[0]["N"], shm[1]["N"])) for _ in range(2_000)]
        false = sum(got.count(k) for k in ("MORE", "LESS", "SPLIT")) / 2_000 if delta == 0 else None
        # a null must not call: UNRESOLVED is a named non-call, so the Δ = 0 test is false calls ≤ 5%
        P[f"shares {name}, seed noise ε ~ N(0, τ), 2,000 realisations"] = {
            "must_read": want if delta else "false calls (MORE, LESS, SPLIT) ≤ 5%", "tau": tau,
            "states": {k: got.count(k) for k in set(got)}, "rate": got.count(want) / 2_000, "false_call_rate": false,
            "fires": false <= 0.05 if delta == 0 else got.count(want) / 2_000 >= 0.95}
    no_eps = dict.fromkeys(NUISANCE, 0.0)
    for name, delta, want in (("lean up: Δ = S̄ (the JEPA pooled spread)", s_bar, "LEANS MORE"),
                              ("lean down: Δ = −S̄", -s_bar, "LEANS LESS")):
        rb = [readout(_score_plant(scm[t], delta, no_eps, rng), w) for t in ("m1", "m2")]
        c = compare_readability(rb, rm, mdd)
        c2 = compare_readability(rb, rm, mdd2, SECONDARY)  # Δ is the primary's bar: reported, no requirement
        P[f"score {name}"] = {"must_read": want, "delta": delta, "got": c["index"], "D": c["D"], "S": c["S"],
                              "ci": c["ci"], "ci_half_width": c["ci_half_width"], "fires": c["index"] == want,
                              "secondary_no_requirement": {k: c2[k] for k in ("index", "D", "S", "ci")}}
    # thin n: 600 test galaxies (above the 500 floor), Δ = half the bar there, independent per-galaxy score
    # noise (2 sd) so B's errors are not M's → a CI wider than the bar, straddling 0 and +S
    rows600 = np.sort(np.random.default_rng(3).choice(len(d["test"]), 600, replace=False))
    rm600 = tuple(readout(scm[t], w, rows600) for t in ("m1", "m2"))
    half = float(np.mean([abs(rm600[0]["per"][n]["auc"] - rm600[1]["per"][n]["auc"]) for n in NUISANCE])) / 2
    rb = [readout(_score_plant(scm[t], half, no_eps, rng, noise=2.0, rows=rows600), w, rows600) for t in ("m1", "m2")]
    c = compare_readability(rb, rm600, mdd)
    c2 = compare_readability(rb, rm600, mdd2, SECONDARY)
    P["score thin: 600 test galaxies, Δ = S̄/2, independent score noise"] = {
        "must_read": "UNRESOLVED", "delta": half, "got": c["index"], "D": c["D"], "S": c["S"], "ci": c["ci"],
        "ci_half_width": c["ci_half_width"], "min_detectable_D": mdd, "fires": c["index"] == "UNRESOLVED",
        "secondary_no_requirement": {k: c2[k] for k in ("index", "D", "S", "ci", "ci_half_width")}}
    rows400 = rows600[:400]
    rm400 = tuple(readout(scm[t], w, rows400) for t in ("m1", "m2"))
    rb = [readout(_score_plant(scm[t], DELTA, no_eps, rng, rows=rows400), w, rows400) for t in ("m1", "m2")]
    c = compare_readability(rb, rm400, mdd)
    c2 = compare_readability(rb, rm400, mdd2, SECONDARY)
    P[f"score insufficient: 400 test galaxies, Δ = +{DELTA}"] = {
        "must_read": "INSUFFICIENT", "got": c["index"], "fires": c["index"] == "INSUFFICIENT",
        "per_variable": c["per_variable"], "got_secondary": c2["index"], "fires_secondary": c2["index"] == "INSUFFICIENT"}
    # secondary-only plants, last so every plant above draws the RNG stream it drew before: the lean pair
    # and the thin plant at the secondary index's own bar
    for name, delta, want in (("lean up: Δ = S̄' (the JEPA pooled spread without PSF)", s_bar2, "LEANS MORE"),
                              ("lean down: Δ = −S̄'", -s_bar2, "LEANS LESS")):
        rb = [readout(_score_plant(scm[t], delta, no_eps, rng), w) for t in ("m1", "m2")]
        c2 = compare_readability(rb, rm, mdd2, SECONDARY)
        P[f"secondary score {name}"] = {"must_read": want, "delta": delta, "got_secondary": c2["index"], "D": c2["D"],
                                        "S": c2["S"], "ci": c2["ci"], "fires_secondary": c2["index"] == want}
    half2 = float(np.mean([abs(rm600[0]["per"][n]["auc"] - rm600[1]["per"][n]["auc"]) for n in SECONDARY])) / 2
    rb = [readout(_score_plant(scm[t], half2, no_eps, rng, noise=2.0, rows=rows600), w, rows600) for t in ("m1", "m2")]
    c2 = compare_readability(rb, rm600, mdd2, SECONDARY)
    P["secondary score thin: 600 test galaxies, Δ = S̄'/2, independent score noise"] = {
        "must_read": "UNRESOLVED", "delta": half2, "got_secondary": c2["index"], "D": c2["D"], "S": c2["S"],
        "ci": c2["ci"], "ci_half_width": c2["ci_half_width"], "min_detectable_D": mdd2,
        "fires_secondary": c2["index"] == "UNRESOLVED"}
    out["fire_as_expected"] = all(v["fires"] for v in P.values() if "fires" in v)
    out["fire_as_expected_secondary"] = all(v["fires_secondary"] for v in P.values() if "fires_secondary" in v)
    out["seconds"] = time.perf_counter() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("distractor_plants_dev.json" if limit else "distractor_plants.json")
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"  wrote {path}", file=sys.stderr)
    return out


# ── sky_r: the one pull, pinned ─────────────────────────────────────────────────────────────────────

def _sql_retry(sql: str, tries: int = 12):
    """`metadata.run_sql` (public SkyServer, no token) with backoff: the endpoint drops connections."""
    from galaxy_jepa.data.metadata import run_sql
    for k in range(tries):
        try:
            return run_sql(sql, timeout=120)
        except Exception as exc:  # noqa: BLE001 — timeouts / 5xx / refused connections are transient
            if k == tries - 1:
                raise
            print(f"  SkyServer call failed ({type(exc).__name__}); retry {k + 1} in {min(5 * 2 ** k, 120)} s",
                  file=sys.stderr)
            time.sleep(min(5 * 2 ** k, 120))


def pull_sky() -> dict:
    """`PhotoObjAll.sky_r` for variable 7's galaxies, the capped train + test (O1's bank ids, which every
    plant run checks equal to them), by objID (= the probe's dr8objid) in 100-ID batches. Each batch is kept
    as it lands, so a rerun resumes; made once, and `_sky` refuses a file that no longer matches its record."""
    import pandas as pd
    if SKY_RECORD.exists():
        raise SystemExit(f"pull-sky: {SKY_RECORD} exists; the pull is made once and pinned")
    t0 = time.time()
    ids = sorted(str(int(i)) for i in load_bank(OUT / "o1_embeddings.npz")["ids"])
    part = OUT / "distractor_sky_r.partial.json"
    got = json.loads(part.read_text()) if part.exists() else {}
    for k in range(0, len(ids), SKY_BATCH):
        if str(k) in got:
            continue
        got[str(k)] = _sql_retry(SKY_SQL.format(",".join(ids[k:k + SKY_BATCH])))
        part.write_text(json.dumps(got))
        if k // SKY_BATCH % 50 == 0:
            print(f"  {k + SKY_BATCH:,} / {len(ids):,} ids  {time.time() - t0:5.0f}s", file=sys.stderr)
    df = pd.DataFrame([r for k in sorted(got, key=int) for r in got[k]]).rename(columns={"objID": "object_id"})
    dup = int(df.object_id.duplicated().sum())
    df = df.drop_duplicates("object_id").sort_values("object_id").reset_index(drop=True)
    df.to_csv(SKY_CSV, index=False)
    v = pd.to_numeric(df.sky_r, errors="coerce")
    rec = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "service": "public SkyServer SQL (metadata.run_sql, DR17), no token",
           "query_template": SKY_SQL, "batch": SKY_BATCH, "galaxies": "capped train (40,000, H5's stride) + whole test",
           "ids_sha1": hashlib.sha1(",".join(ids).encode()).hexdigest(), "ids_requested": len(ids),
           "rows_returned": len(df), "duplicate_rows_dropped": dup, "ids_missing": len(set(ids) - set(df.object_id)),
           "sentinel_le_minus_9000": int((v <= -9000).sum()), "non_positive": int((v <= 0).sum()),
           "unparseable": int(v.isna().sum()),
           "quantiles_valid": {q: float(np.quantile(v[v > 0], q)) for q in (0, 0.01, 0.5, 0.99, 1)},
           "output": "artifacts/out/" + SKY_CSV.name, "output_sha1": hashlib.sha1(SKY_CSV.read_bytes()).hexdigest(),
           "seconds": round(time.time() - t0)}
    SKY_RECORD.write_text(json.dumps(rec, indent=1))
    return rec


def _sky() -> tuple[dict, dict]:
    """objID → sky_r from the pinned pull; a sentinel or a non-positive value is missing (NaN)."""
    import pandas as pd
    rec = json.loads(SKY_RECORD.read_text())
    if hashlib.sha1(SKY_CSV.read_bytes()).hexdigest() != rec["output_sha1"]:
        raise SystemExit(f"{SKY_CSV} does not match its record {SKY_RECORD}")
    df = pd.read_csv(SKY_CSV, dtype={"object_id": str})
    v = pd.to_numeric(df.sky_r, errors="coerce").to_numpy(float)
    return dict(zip(df.object_id.astype(int), np.where(v > 0, v, np.nan), strict=True)), rec


# ── (iii) and the full-split untrained bank, for scoring ────────────────────────────────────────────

def embed_untrained() -> dict:
    """The untrained encoder (M's architecture, seed 0) on the full train + test splits, as a c2 bank.
    Must equal O1's untrained bank on the shared galaxies, as M1's c2 bank equalled O1's M."""
    from galaxy_jepa.models.vit import load_frozen_encoder
    from galaxy_jepa.probing.controls import untrained_encoder_matrix
    from j4_spread_controls import prepare
    s = prepare(None, 0, label="DS-untrained", sources=1)
    config = load_frozen_encoder(s.ckpt).config
    m = untrained_encoder_matrix(config, s.ds, device=s.device, seed=0)
    write_bank(_bank("untrained"), ids=m.object_ids, x=m.x.astype(np.float32),
               checkpoint=untrained_descriptor(config, 0))
    o1 = load_bank(OUT / "o1_embeddings.npz")
    pos = {int(o): k for k, o in enumerate(m.object_ids)}
    k = o1["ids"][:5000]
    diff = float(np.abs(m.x[[pos[int(i)] for i in k]] - o1["untrained"][:5000]).max())
    return {"n": len(m.object_ids), "max_abs_vs_o1_untrained_5000": diff}


def morph(tags: list[str]) -> dict:
    """(iii), context, no state: mean AUC over criterion 2's 33 powered answers, headline protocol."""
    ans = json.loads((OUT / "c2_plants.json").read_text())["powered"]
    sc, meta, _ = fit_scores(tags)
    out = {t: {"mean_auc_powered": float(np.mean([auc(sc[t][f], meta["y"][f]) for f in ans])),
               "per_answer": {f: auc(sc[t][f], meta["y"][f]) for f in ans}} for t in tags}
    (OUT / "distractor_morph.json").write_text(json.dumps(out, indent=1))
    return {t: v["mean_auc_powered"] for t, v in out.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plants", "pull-sky", "embed-untrained", "morph"))
    ap.add_argument("tags", nargs="*", default=["m1", "m2"])
    ap.add_argument("--limit", type=int, default=0, help="dev: stride-subsample train and test to N each")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    r = (plants(a.limit, a.reps, a.out) if a.cmd == "plants" else pull_sky() if a.cmd == "pull-sky"
         else embed_untrained() if a.cmd == "embed-untrained" else morph(a.tags))
    print(json.dumps(r, indent=1, default=float))
