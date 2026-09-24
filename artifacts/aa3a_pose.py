"""Brief AA3a — is the pose code (PC1, PC2) instrumental or structural?

X1c: (PC1, PC2) transform as an image-plane polar vector. X's post hoc tie to the core's centroid
about the stamp centre (R² 0.40) survived no whole-stamp shift, so it is not position. A shift
BETWEEN bands is translation-invariant and was never tested; this is that test.

On X1c's 2,000 galaxies, per band (g, r, i), on valid pixels within the Petrosian aperture,
sky-subtracted:
  c_b     light centroid
  core_b  centroid within 3 px of the band's (σ=1-smoothed) peak
Candidates (all translation-invariant):
  INTER   [c_g − c_r, c_r − c_i, core_g − core_r, core_r − core_i]   band misregistration / DCR
  COMMON  median_b(core_b − c_b)                                      lopsidedness
          (the band median: one misregistered band cannot move it — AA3a's fidelity plant found
          the band mean leaked a g-only shift into COMMON, +0.17 px per +0.5 px)
States, family and thresholds: `aa_findings.md` §AA3a.

  --planted   D28 — synthetic targets through the identical path; an injected g-band shift is
              recovered by the measurement; rot180 moves the PC readout (positive control)
  --aa3a      the test, and the causal arm (a g-only sub-pixel shift, re-embedded with M)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
import r_nonlinear as R  # noqa: E402
import w2_name_pcs as W  # noqa: E402
import x1_handedness as X  # noqa: E402

from galaxy_jepa.data.validity import invalid_planes  # noqa: E402
from galaxy_jepa.models.vit import load_frozen_encoder  # noqa: E402
from galaxy_jepa.probing import nulls as nulls_mod  # noqa: E402
from galaxy_jepa.probing.extract import extract_matrix  # noqa: E402

OUT = R.OUT / "aa3a_pose.json"
PLANTED = R.OUT / "aa3a_planted.json"
MEAS = R.OUT / "aa3a_offsets.npz"
N_PERM = 1000
HOLD_R2 = 0.05
ADDS_R2 = 0.02
N_CAUSAL = 200
C = 127.5


def offsets(img: np.ndarray, petro_px: float) -> dict[str, np.ndarray]:
    """Per-band aperture centroid and core (x right, y down, about the stamp centre)."""
    edge, interior = invalid_planes(img)
    valid = ~(edge | interior)
    h, w = img.shape[1:]
    yy, xx = np.mgrid[:h, :w]
    dx, dy = xx - C, yy - C
    r = np.hypot(dx, dy)
    rp = float(np.clip(petro_px, 3.0, 100.0))
    outer = valid & (r > 2 * rp)
    if outer.sum() < 1000:
        outer = valid & (r > rp)
    cen, core = np.full((3, 2), np.nan), np.full((3, 2), np.nan)
    for b in range(3):
        v = img[b][outer].astype(np.float64)
        for _ in range(3):
            med, mad = np.median(v), 1.4826 * np.median(np.abs(v - np.median(v)))
            v = v[np.abs(v - med) < 3 * mad] if mad > 0 else v
        band = img[b].astype(np.float64) - np.median(v)
        ap = np.clip(np.where(valid & (r <= rp), band, 0), 0, None)
        s = ap.sum()
        if s <= 0:
            continue
        cen[b] = (ap * dx).sum() / s, (ap * dy).sum() / s
        sm = ndimage.gaussian_filter(np.where(valid, band, 0), 1.0)
        sm = np.where(valid & (r <= rp), sm, -np.inf)
        py, px = np.unravel_index(np.argmax(sm), sm.shape)
        near = valid & (np.hypot(xx - px, yy - py) <= 3)
        cw = np.clip(np.where(near, band, 0), 0, None)
        if cw.sum() > 0:
            core[b] = (cw * dx).sum() / cw.sum(), (cw * dy).sum() / cw.sum()
    return {"cen": cen, "core": core}


def candidates(cen: np.ndarray, core: np.ndarray) -> dict[str, np.ndarray]:
    """cen, core: (n, 3 bands, 2)."""
    inter = np.concatenate([cen[:, 0] - cen[:, 1], cen[:, 1] - cen[:, 2],
                            core[:, 0] - core[:, 1], core[:, 1] - core[:, 2]], 1)
    common = np.median(core - cen, axis=1)
    return {"inter": inter, "common": common}


def _prep(y, x):
    ok = np.isfinite(y) & np.all(np.isfinite(x), 1)
    return W.rank_normal(y[ok]), np.column_stack([W.rank_normal(c) for c in x[ok].T])


def cv_r2(y: np.ndarray, x: np.ndarray, perm: np.ndarray | None = None) -> float:
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_predict

    y, x = _prep(y, x)
    if perm is not None:
        y = y[perm]
    pred = cross_val_predict(LinearRegression(), x, y, cv=KFold(5, shuffle=True, random_state=0))
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def perm_p(y: np.ndarray, x: np.ndarray, obs: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    n = int((np.isfinite(y) & np.all(np.isfinite(x), 1)).sum())
    null = [cv_r2(y, x, rng.permutation(n)) for _ in range(N_PERM)]
    return float((1 + np.sum(np.array(null) >= obs)) / (1 + N_PERM))


def pairing(pc: np.ndarray, vec: np.ndarray) -> dict:
    """Spearman of PC1 with each x component and PC2 with each y component (even columns x)."""
    from scipy.stats import spearmanr

    out = {}
    for j in range(0, vec.shape[1], 2):
        ok = np.isfinite(vec[:, j]) & np.isfinite(pc[:, 0])
        out[f"v{j // 2}"] = {"PC1~x": float(spearmanr(pc[ok, 0], vec[ok, j])[0]),
                             "PC2~y": float(spearmanr(pc[ok, 1], vec[ok, j + 1])[0]),
                             "PC1~y": float(spearmanr(pc[ok, 0], vec[ok, j + 1])[0]),
                             "PC2~x": float(spearmanr(pc[ok, 1], vec[ok, j])[0])}
    return out


def test(pc: np.ndarray, cand: dict[str, np.ndarray], seed: int, perm: bool = True) -> dict:
    """CV R² of each PC on each candidate and on both; permutation p; state."""
    both = np.concatenate([cand["inter"], cand["common"]], 1)
    rec: dict = {}
    p = {}
    for k in ("inter", "common"):
        for j, name in enumerate(("PC1", "PC2")):
            r2 = cv_r2(pc[:, j], cand[k])
            rec[f"{k}:{name}"] = r2
            if perm:
                p[f"{k}:{name}"] = perm_p(pc[:, j], cand[k], r2, seed + j + 10 * len(k))
    for j, name in enumerate(("PC1", "PC2")):
        rec[f"joint:{name}"] = cv_r2(pc[:, j], both)
    rec["p"] = p
    return rec


def state(rec: dict, sig: dict) -> str:
    """First match applies (aa_findings.md §AA3a)."""
    def holds(k):
        return all(rec[f"{k}:{n}"] >= HOLD_R2 and sig.get(f"{k}:{n}", False) for n in ("PC1", "PC2"))

    def adds(k, other):
        return any(rec[f"joint:{n}"] - rec[f"{other}:{n}"] >= ADDS_R2 for n in ("PC1", "PC2"))

    hi, hc = holds("inter"), holds("common")
    if not hi and not hc:
        return "NEITHER"
    if hi and hc:
        ai, ac = adds("inter", "common"), adds("common", "inter")
        if ai and ac:
            return "BOTH"
        if ai:
            return "INSTRUMENTAL"
        if ac:
            return "STRUCTURAL"
        return "INSEPARABLE"
    return "INSTRUMENTAL" if hi else "STRUCTURAL"


def fam_sig(p: dict) -> dict:
    nulls_mod.assert_null_resolution(N_PERM, alpha=W.ALPHA, method="benjamini_yekutieli",
                                     n_tests=len(p))
    return nulls_mod.family_significant(p, alpha=W.ALPHA, method="benjamini_yekutieli",
                                        n_tests=len(p))


# ------------------------------------------------------------------ images and embeddings


def fshift(band: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Fourier sub-pixel shift of one band (x right, y down)."""
    h, w = band.shape
    ky = torch.fft.fftfreq(h).reshape(-1, 1)
    kx = torch.fft.fftfreq(w).reshape(1, -1)
    ph = torch.exp(-2j * torch.pi * (kx * dx + ky * dy))
    return torch.fft.ifft2(torch.fft.fft2(band.double()) * ph).real.to(band.dtype)


class Shifted(torch.utils.data.Dataset):
    """The g band alone shifted by (dx, dy); r and i untouched. ``rot180`` is the control."""

    def __init__(self, base, index, dx: float = 0.0, dy: float = 0.0, rot180: bool = False):
        self.base, self.index, self.dx, self.dy, self.rot = base, index, dx, dy, rot180

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        item = dict(self.base[int(self.index[i])])
        img = item["image"].clone()
        if self.rot:
            img = X.transform(img, "rot180")
        elif self.dx or self.dy:
            img[0] = fshift(img[0], self.dx, self.dy)
        item["image"] = img
        return item


def context(setup):
    frozen = load_frozen_encoder(setup.ckpt)
    ctx = R.Ctx(setup, frozen, 0, dry=True)
    B = X.basis(ctx, setup)
    blob = np.load(X.BANK)
    ids = blob["ids"]
    pcs = {}
    for b, key in zip(B, ("M", "untrained-s0", "untrained-s1", "untrained-s2"), strict=True):
        pcs[key] = ((blob[f"{key}:orig"].astype(np.float64) - b["mu"]) @ b["v"][:, :2]) / b["sd"][:2]
    return frozen, B, ids, pcs


def measure(setup, ids) -> tuple[np.ndarray, np.ndarray]:
    if MEAS.exists():
        z = np.load(MEAS)
        if np.array_equal(z["ids"], ids):
            return z["cen"], z["core"]
    idx = X.ds_index(setup, ids)
    cen, core = np.full((ids.size, 3, 2), np.nan), np.full((ids.size, 3, 2), np.nan)
    for i, j in enumerate(idx):
        item = setup.ds[int(j)]
        o = offsets(item["image"].float().numpy(), item["petro_rad_arcsec"] / item["pixel_scale"])
        cen[i], core[i] = o["cen"], o["core"]
    np.savez(MEAS, ids=ids, cen=cen, core=core)
    return cen, core


def pc_of(frozen, B0, ds) -> np.ndarray:
    x = extract_matrix(frozen, ds, device=ds_device[0]).x.astype(np.float64)
    return ((x - B0["mu"]) @ B0["v"][:, :2]) / B0["sd"][:2]


ds_device = ["cpu"]  # set from setup.device by the entry points


def planted(setup) -> dict:
    frozen, B, ids, pcs = context(setup)
    ds_device[0] = setup.device
    cen, core = measure(setup, ids)
    cand = candidates(cen, core)
    rng = np.random.default_rng(12)
    out: dict = {"n": int(ids.size)}

    # (1) synthetic PC targets from the measured candidates, identical test + state
    def target(v):
        z = np.column_stack([W.rank_normal(np.nan_to_num(c, nan=np.nanmedian(c))) for c in v.T])
        a, b = z[:, 0::2].sum(1), z[:, 1::2].sum(1)  # x components → PC1, y → PC2
        return np.column_stack([a / a.std(), b / b.std()])

    noise = rng.normal(size=(ids.size, 2))
    plants = {"instrumental": target(cand["inter"]) + 1.5 * noise,
              "structural": target(cand["common"]) + 1.5 * noise,
              "both": target(cand["inter"]) + target(cand["common"]) + 2.0 * noise,
              "neither": noise}
    for k, pc in plants.items():
        rec = test(pc, cand, seed=50)
        rec["state"] = state(rec, fam_sig(rec["p"]))
        out[f"synthetic_{k}"] = rec

    # (2) measurement fidelity: g shifted +0.5 px in x on 200 stamps → Δ(c_g − c_r) ≈ (+0.5, 0)
    sub = np.sort(rng.choice(ids.size, N_CAUSAL, replace=False))
    idx = X.ds_index(setup, ids[sub])
    d = []
    for j in idx[:100]:
        item = setup.ds[int(j)]
        img = item["image"].float()
        pr = item["petro_rad_arcsec"] / item["pixel_scale"]
        o0 = offsets(img.numpy(), pr)
        img2 = img.clone()
        img2[0] = fshift(img2[0], 0.5, 0.0)
        o1 = offsets(img2.numpy(), pr)
        d.append(np.r_[(o1["cen"][0] - o1["cen"][1]) - (o0["cen"][0] - o0["cen"][1]),
                       np.median(o1["core"] - o1["cen"], 0) - np.median(o0["core"] - o0["cen"], 0)])
    d = np.array(d)
    out["fidelity_g_plus_half_px_x"] = {"d_gr_cen_median": np.nanmedian(d[:, :2], 0).tolist(),
                                        "common_median": np.nanmedian(d[:, 2:], 0).tolist()}

    # (3) positive control for the causal readout: rot180 should send (PC1, PC2) → −(PC1, PC2)
    base = pc_of(frozen, B[0], Shifted(setup.ds, idx))
    rot = pc_of(frozen, B[0], Shifted(setup.ds, idx, rot180=True))
    out["control_rot180"] = {"mean_delta_sd": (rot - base).mean(0).tolist(),
                             "corr_rot_vs_base": [float(np.corrcoef(rot[:, k], base[:, k])[0, 1])
                                                  for k in (0, 1)],
                             "base_matches_bank": [float(np.corrcoef(base[:, k], pcs["M"][sub, k])[0, 1])
                                                   for k in (0, 1)]}
    return out


def aa3a(setup) -> dict:
    frozen, B, ids, pcs = context(setup)
    ds_device[0] = setup.device
    cen, core = measure(setup, ids)
    cand = candidates(cen, core)
    rec: dict = {"n": int(ids.size),
                 "offset_scale_px": {"inter_mad": np.nanmedian(np.abs(cand["inter"]), 0).tolist(),
                                     "common_mad": np.nanmedian(np.abs(cand["common"]), 0).tolist()}}
    m = test(pcs["M"], cand, seed=100)
    sig = fam_sig(m["p"])
    m["significant"] = {k: bool(v) for k, v in sig.items()}
    m["state"] = state(m, sig)
    m["pairing_inter"] = pairing(pcs["M"], cand["inter"])
    m["pairing_common"] = pairing(pcs["M"], cand["common"])
    rec["M"] = m
    rec["untrained"] = {k: test(pcs[k], cand, seed=0, perm=False)
                        for k in ("untrained-s0", "untrained-s1", "untrained-s2")}

    # Causal arm: shift g alone, re-embed with M, read (PC1, PC2)
    rng = np.random.default_rng(12)
    sub = np.sort(rng.choice(ids.size, N_CAUSAL, replace=False))
    idx = X.ds_index(setup, ids[sub])
    base = pc_of(frozen, B[0], Shifted(setup.ds, idx))
    causal = {}
    for name, (dx, dy) in {"g+0.5x": (0.5, 0), "g+1x": (1.0, 0), "g+0.5y": (0, 0.5),
                           "g+1y": (0, 1.0)}.items():
        pc = pc_of(frozen, B[0], Shifted(setup.ds, idx, dx, dy))
        dlt = pc - base
        se = dlt.std(0, ddof=1) / np.sqrt(len(dlt))
        causal[name] = {"mean_delta_sd": dlt.mean(0).tolist(), "se": se.tolist()}
    comp = {"x": ("g+1x", 0), "y": ("g+1y", 1)}
    resp = {}
    for axis, (name, k) in comp.items():
        mu, se = causal[name]["mean_delta_sd"][k], causal[name]["se"][k]
        resp[axis] = bool(abs(mu) >= 0.10 and abs(mu) > 1.96 * se)
    causal["state"] = ("RESPONDS" if all(resp.values()) else
                       "RESPONDS ON ONE AXIS" if any(resp.values()) else "DOES NOT RESPOND")
    rec["causal"] = causal
    return rec


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--aa3a"
    setup = R.prepare("runs/m/encoder.pt", R.MAX_TRAIN, label="AA3a", sources=1)
    out, path = (planted(setup), PLANTED) if mode == "--planted" else (aa3a(setup), OUT)
    path.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:8000])


if __name__ == "__main__":
    main()
