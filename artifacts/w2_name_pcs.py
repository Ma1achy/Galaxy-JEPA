"""Brief W2 — name PC1 and PC2: padding, sky, noise, crowding, colour (+ exploratory stamp axes).

Pre-registered in artifacts/w_findings.md §W2 (hashed before --run; the --planted results are in
the pre-registration, per D28). PCA exactly as V3: covariance PCA on P2's train (A), scores for
every union galaxy.

    uv run python artifacts/w2_name_pcs.py --planted   # D28 checks, before hashing
    uv run python artifacts/w2_name_pcs.py --measure   # stamp pass -> out/w2_stamp_axes.npz
    uv run python artifacts/w2_name_pcs.py --run       # -> out/w2_name_pcs.json (+ grids)
"""

from __future__ import annotations

import csv
import json
import sys
import time

import numpy as np
from scipy import ndimage
from scipy.stats import norm, rankdata

sys.path.insert(0, "artifacts")
import r_nonlinear as R  # noqa: E402

from galaxy_jepa.data.validity import invalid_planes  # noqa: E402
from galaxy_jepa.models.vit import load_frozen_encoder  # noqa: E402
from galaxy_jepa.probing import nulls as nulls_mod  # noqa: E402

N_PERM, BATCH, ALPHA = 10_000, 250, 0.05
DETECT_SIGMA, MIN_AREA = 5.0, 5
PIX = 0.396
CONFIRMATORY = ("padded_fraction", "padded_status", "interior_invalid",
                "sky_g", "sky_r", "sky_i", "noise_g", "noise_r", "noise_i",
                "crowd_count", "contam_flux", "g_r", "u_r")
EXPLORATORY = ("stamp_colour_gr", "centroid_offset", "stamp_flux", "target_area")
BANK = R.OUT / "w2_stamp_axes.npz"
OUT = R.OUT / "w2_name_pcs.json"


# ------------------------------------------------------------------ one stamp


def stamp_axes(img: np.ndarray, petro_px: float) -> dict[str, float]:
    """Label-free measurements of one (3, H, W) normalised stamp, as the encoder saw it."""
    edge, interior = invalid_planes(img)
    valid = ~(edge | interior)
    h, w = img.shape[1:]
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2)
    rp = float(np.clip(petro_px, 3.0, 100.0))
    outer = valid & (r > 2 * rp)
    if outer.sum() < 1000:
        outer = valid & (r > rp)
    out = {"padded_fraction": float(edge.mean()), "interior_invalid": float(interior.mean())}
    sky, noise = [], []
    for c in range(3):
        v = img[c][outer].astype(np.float64)
        for _ in range(3):  # sigma-clip the sources away
            med, mad = np.median(v), 1.4826 * np.median(np.abs(v - np.median(v)))
            v = v[np.abs(v - med) < 3 * mad] if mad > 0 else v
        sky.append(float(np.median(v)))
        noise.append(float(1.4826 * np.median(np.abs(v - np.median(v)))))
    for c, b in enumerate("gri"):
        out[f"sky_{b}"], out[f"noise_{b}"] = sky[c], noise[c]
    band = img[1].astype(np.float64) - sky[1]
    det = valid & (band > DETECT_SIGMA * max(noise[1], 1e-12))
    lab, n = ndimage.label(det, structure=np.ones((3, 3)))
    area = ndimage.sum(det, lab, index=np.arange(1, n + 1))
    flux = ndimage.sum(band, lab, index=np.arange(1, n + 1))
    near = ndimage.minimum(r, lab, index=np.arange(1, n + 1)) if n else np.array([])
    big = area >= MIN_AREA
    target = big & (near <= rp)
    others = big & (near > rp)
    out["crowd_count"] = float(others.sum())
    tf = float(flux[target].sum()) if target.any() else 0.0
    out["contam_flux"] = float(flux[others].sum() / tf) if tf > 0 else np.nan
    out["target_area"] = float(area[target].sum()) if target.any() else 0.0
    ap = valid & (r <= rp)
    g, rr = img[0][ap].astype(np.float64) - sky[0], img[1][ap].astype(np.float64) - sky[1]
    out["stamp_colour_gr"] = float(g.mean() - rr.mean()) if ap.any() else np.nan
    out["stamp_flux"] = float(np.clip(band, 0, None)[valid].sum())
    wgt = np.clip(np.where(ap, band, 0), 0, None)
    s = wgt.sum()
    out["centroid_offset"] = (float(np.hypot((wgt * yy).sum() / s - (h - 1) / 2,
                                             (wgt * xx).sum() / s - (w - 1) / 2))
                              if s > 0 else np.nan)
    return out


# ------------------------------------------------------------------ statistics


def spearman_perm(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    rx, ry = rankdata(x[ok]), rankdata(y[ok])
    rx, ry = rx - rx.mean(), ry - ry.mean()
    rx, ry = rx / np.linalg.norm(rx), ry / np.linalg.norm(ry)
    obs = float(rx @ ry)
    rng, hits = np.random.default_rng(seed), 0
    for s in range(0, N_PERM, BATCH):
        b = min(BATCH, N_PERM - s)
        p = rng.permuted(np.broadcast_to(ry, (b, ry.size)).copy(), axis=1)
        hits += int(np.sum(np.abs(p @ rx) >= abs(obs) - 1e-12))
    return obs, (1 + hits) / (1 + N_PERM)


def state(rho: float, sig: bool) -> str:
    if not sig:
        return "NONE"
    a = abs(rho)
    return ("NAMES" if a >= 0.5 else "CONTRIBUTES" if a >= 0.3 else
            "WEAK" if a >= 0.1 else "NEGLIGIBLE")


def rank_normal(v: np.ndarray) -> np.ndarray:
    return norm.ppf((rankdata(v) - 0.5) / v.size)


def joint_r2(target: np.ndarray, covs: np.ndarray, seed: int) -> dict[str, float]:
    """5-fold out-of-sample R² of the PC score on the candidates: linear and boosted trees."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_predict

    ok = np.isfinite(target) & np.all(np.isfinite(covs), axis=1)
    y = rank_normal(target[ok])
    x = np.column_stack([rank_normal(c) for c in covs[ok].T])
    kf = KFold(5, shuffle=True, random_state=seed)
    out = {"n": int(ok.sum())}
    for name, model in (("linear", LinearRegression()),
                        ("trees", HistGradientBoostingRegressor(random_state=seed))):
        pred = cross_val_predict(model, x, y, cv=kf)
        out[name] = float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
    return out


def joint_state(r2: float) -> str:
    return "EXPLAINED" if r2 >= 0.5 else "PARTLY EXPLAINED" if r2 >= 0.2 else "UNEXPLAINED"


# ------------------------------------------------------------------ inputs


def pcs(ctx, setup):
    """V3's covariance PCA on A, for M and each untrained draw; scores over the whole union."""
    pos = {int(o): i for i, o in enumerate(ctx.real.object_ids)}
    A = np.array([pos[int(o)] for o in setup.train_ids])
    out = []
    for m in [ctx.real, *ctx.untrained]:
        xa = m.x[A].astype(np.float64)
        mu = xa.mean(0)
        lam, v = np.linalg.eigh(np.cov(xa, rowvar=False))
        v = v[:, ::-1][:, :10]
        out.append({"name": m.encoder_name, "scores": (m.x.astype(np.float64) - mu) @ v,
                    "share": (lam[::-1][:10] / lam.sum()).tolist()})
    return out, A


def colours(union: np.ndarray) -> dict[str, np.ndarray]:
    best: dict[int, tuple[float, dict]] = {}
    with open(R.OUT / "ext" / "union_sdss7_xmatch.csv") as fh:
        for r in csv.DictReader(fh):
            o, d = int(r["object_id"]), float(r["angDist"])
            if o not in best or d < best[o][0]:
                best[o] = (d, r)
    pos = {int(o): i for i, o in enumerate(union)}
    out = {k: np.full(union.size, np.nan) for k in ("g_r", "u_r")}
    for o, (_, r) in best.items():
        i = pos.get(o)
        if i is None or not (r["umag"] and r["gmag"] and r["rmag"]):
            continue
        out["g_r"][i] = float(r["gmag"]) - float(r["rmag"])
        out["u_r"][i] = float(r["umag"]) - float(r["rmag"])
    return out


# ------------------------------------------------------------------ modes


def measure(setup) -> None:
    ds, union = setup.ds, np.asarray(setup.union)
    keys = (*CONFIRMATORY, *EXPLORATORY)
    cols = {k: np.full(union.size, np.nan) for k in keys if k not in ("padded_status", "g_r",
                                                                          "u_r")}
    t0 = time.perf_counter()
    for i in range(len(ds)):
        item = ds[i]
        if int(item["object_id"]) != int(union[i]):
            raise SystemExit("W2: dataset order is not the union order")
        rp = float(item["petro_rad_arcsec"]) / float(item["pixel_scale"])
        a = stamp_axes(item["image"].float().numpy(), rp if np.isfinite(rp) else 10.0)
        for k in cols:
            cols[k][i] = a[k]
        if i % 10_000 == 0:
            print(f"  W2 stamps {i:>6d}/{len(ds)} {time.perf_counter() - t0:5.0f}s",
                  file=sys.stderr)
    np.savez(BANK, ids=union, **cols)


def planted(setup, ctx) -> dict:
    """D28: each test is shown able to reach its positive state, and not to fire on noise."""
    rng = np.random.default_rng(7)
    rec: dict = {}
    # P1 crowding counter: inject k point sources outside 2·Rp into real stamps
    ds = setup.ds
    idx = rng.choice(len(ds), 300, replace=False)
    deltas = {1: [], 3: []}
    base_counts = []
    for i in idx:
        item = ds[int(i)]
        img = item["image"].float().numpy().copy()
        rp = float(np.clip(float(item["petro_rad_arcsec"]) / float(item["pixel_scale"]), 3, 100))
        a0 = stamp_axes(img, rp)
        base_counts.append(a0["crowd_count"])
        h, w = img.shape[1:]
        yy, xx = np.mgrid[:h, :w]
        for k in deltas:
            im = img.copy()
            placed = 0
            while placed < k:
                cy, cx = rng.uniform(12, h - 12), rng.uniform(12, w - 12)
                if np.hypot(cy - (h - 1) / 2, cx - (w - 1) / 2) <= 2 * rp + 10:
                    if 2 * rp + 10 > min(h, w) / 2 - 12:
                        break
                    continue
                psf = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.5 ** 2))
                im += (10 * max(a0["noise_r"], 1e-6) * psf)[None]
                placed += 1
            if placed == k:
                deltas[k].append((stamp_axes(im, rp)["crowd_count"] - a0["crowd_count"]) / k)
    noise = rng.normal(0, 1, (3, 256, 256)).astype(np.float32)
    rec["crowding"] = {"recovery_per_source": {k: float(np.mean(v)) for k, v in deltas.items()},
                       "n_injected_stamps": {k: len(v) for k, v in deltas.items()},
                       "pure_noise_count": stamp_axes(noise, 10.0)["crowd_count"],
                       "real_count_quantiles": np.percentile(base_counts, [5, 25, 50, 75, 95])
                       .tolist(),
                       "real_frac_zero": float(np.mean(np.array(base_counts) == 0))}
    # P2 sky / noise: constant sky + Gaussian noise + a Gaussian galaxy
    yy, xx = np.mgrid[:256, :256]
    gal = 50 * np.exp(-((yy - 127.5) ** 2 + (xx - 127.5) ** 2) / (2 * 8 ** 2))
    syn = (0.7 + rng.normal(0, 0.2, (3, 256, 256)) + gal[None]).astype(np.float32)
    a = stamp_axes(syn, 10.0)
    rec["sky_noise"] = {"sky_r": a["sky_r"], "true_sky": 0.7, "noise_r": a["noise_r"],
                        "true_noise": 0.2}
    # P3 padding: a constant frame strip of 10% of the stamp
    pad = syn.copy()
    pad[:, :, : int(0.1 * 256)] = 0.0
    rec["padding"] = {"padded_fraction": stamp_axes(pad, 10.0)["padded_fraction"], "true": 0.1}
    # P4 the correlation test and its family, on M's real PC1 scores
    specs, _ = pcs(ctx, setup)
    pc1 = specs[0]["scores"][:, 0]
    fam = len(CONFIRMATORY) * 2
    nulls_mod.assert_null_resolution(N_PERM, alpha=ALPHA, method="benjamini_yekutieli",
                                     n_tests=fam)
    z = rank_normal(pc1)
    planted_pos = 0.32 * z + np.sqrt(1 - 0.32 ** 2) * rng.normal(size=z.size)
    planted_neg = rng.permutation(z)
    rp_, pp = spearman_perm(pc1, planted_pos, seed=1)
    rn_, pn = spearman_perm(pc1, planted_neg, seed=2)
    thr = ALPHA / (fam * sum(1 / k for k in range(1, fam + 1)))
    rec["correlation"] = {"planted_rho": rp_, "planted_p": pp,
                          "planted_state": state(rp_, pp <= thr),
                          "negative_rho": rn_, "negative_p": pn,
                          "negative_state": state(rn_, pn <= thr), "by_rank1_threshold": thr,
                          "family": fam}
    # P5 joint R²: a noisy copy of PC1 among noise candidates; and noise alone
    noise_c = rng.normal(size=(z.size, 5))
    pos_c = np.column_stack([0.8 * z + 0.6 * rng.normal(size=z.size), noise_c])
    rec["joint"] = {"planted": joint_r2(pc1, pos_c, 0), "noise_only": joint_r2(pc1, noise_c, 0)}
    rec["joint"]["planted_state"] = joint_state(rec["joint"]["planted"]["trees"])
    rec["joint"]["noise_state"] = joint_state(rec["joint"]["noise_only"]["trees"])
    return rec


def run(setup, ctx) -> dict:
    specs, A = pcs(ctx, setup)
    union = ctx.real.object_ids
    bank = np.load(BANK, allow_pickle=False)
    if not np.array_equal(bank["ids"], union):
        raise SystemExit("W2: the stamp bank is not over the union")
    cand = {k: bank[k] for k in bank.files if k != "ids"}
    cand["padded_status"] = (cand["padded_fraction"] > 0.01).astype(float)
    cand.update(colours(union))
    rec: dict = {"n_union": int(union.size), "share_M": specs[0]["share"],
                 "padded_rate": float(np.mean(cand["padded_status"]))}
    fam = len(CONFIRMATORY) * 2
    nulls_mod.assert_null_resolution(N_PERM, alpha=ALPHA, method="benjamini_yekutieli",
                                     n_tests=fam)
    p, rho = {}, {}
    for j in (0, 1):
        s = specs[0]["scores"][:, j]
        for k in (*CONFIRMATORY, *EXPLORATORY):
            key = f"PC{j + 1}:{k}"
            rho[key], p[key] = spearman_perm(s, cand[k], seed=100 * j + len(key))
    sig = nulls_mod.family_significant({k: v for k, v in p.items()
                                        if k.split(":")[1] in CONFIRMATORY},
                                       alpha=ALPHA, method="benjamini_yekutieli", n_tests=fam)
    rec["correlations"] = {}
    for key in rho:
        conf = key.split(":")[1] in CONFIRMATORY
        s_ = sig[key] if conf else p[key] < ALPHA
        untr = []
        for u in specs[1:]:
            ok = np.isfinite(cand[key.split(":")[1]])
            untr.append(max(abs(float(np.corrcoef(rankdata(u["scores"][ok, jj]),
                                                  rankdata(cand[key.split(":")[1]][ok]))[0, 1]))
                            for jj in (0, 1)))
        rec["correlations"][key] = {"rho": rho[key], "p": p[key], "confirmatory": conf,
                                    "state": state(rho[key], s_) + ("" if conf else
                                                                    " (exploratory)"),
                                    "untrained_top2_max_abs": untr}
    # joint: confirmatory set, then confirmatory + exploratory
    conf_m = np.column_stack([cand[k] for k in CONFIRMATORY if k != "padded_status"])
    all_m = np.column_stack([conf_m] + [cand[k] for k in EXPLORATORY])
    rec["joint"] = {}
    for j in (0, 1):
        s = specs[0]["scores"][:, j]
        c = joint_r2(s, conf_m, 0)
        e = joint_r2(s, all_m, 0)
        rec["joint"][f"PC{j + 1}"] = {"confirmatory": c, "state": joint_state(c["trees"]),
                                      "with_exploratory": e,
                                      "state_with_exploratory": joint_state(e["trees"])}
    # learned or architectural: M's PC scores against the untrained draws' top-10
    rec["vs_untrained"] = {
        f"PC{j + 1}": [max(abs(float(np.corrcoef(rankdata(specs[0]["scores"][:, j]),
                                                 rankdata(u["scores"][:, jj]))[0, 1]))
                           for jj in range(10)) for u in specs[1:]] for j in (0, 1)}
    return rec


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--run"
    setup = R.prepare("runs/m/encoder.pt", R.MAX_TRAIN, label="W2", sources=1)
    if mode == "--measure":
        measure(setup)
        return
    ctx = R.Ctx(setup, load_frozen_encoder(setup.ckpt), 0, dry=True)
    if mode == "--planted":
        out = planted(setup, ctx)
        (R.OUT / "w2_planted.json").write_text(json.dumps(out, indent=1, default=float))
        print(json.dumps(out, indent=1, default=float), file=sys.stderr)
        return
    rec = run(setup, ctx)
    OUT.write_text(json.dumps(rec, indent=1, default=float))
    for k, v in rec["correlations"].items():
        print(f"W2 {k:<24s} rho {v['rho']:+.3f} p {v['p']:.1e} {v['state']:<28s} "
              f"untrained top-2 max|rho| {[round(x, 2) for x in v['untrained_top2_max_abs']]}",
              file=sys.stderr)
    print(f"W2 joint: {json.dumps(rec['joint'], default=float)}", file=sys.stderr)


if __name__ == "__main__":
    main()
