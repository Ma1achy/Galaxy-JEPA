"""Brief X1 — is PC1/PC2 spiral handedness?

  --planted  D28 checks through the identical code path: pixel-feature "encoders" (a log-polar
             chirality pseudoscalar, left-right asymmetry, aperture flux) on the X1c stamps and
             their transforms, the ±-sign planted magnitude variable for X1a, and synthetic log
             spirals of both handedness (out of distribution) through chi and through M.
  --bank     embed the X1c sample and its rot90 / rot180 / mirror through M and 3 untrained draws
  --run      X1a (magnitude vs spirality), X1b (GZ1 handedness votes), X1c (symmetry test)

Parity (established before the pre-registration): our stamps are the SDSS frame pixel array,
cut with Cutout2D and written without WCS. det(CD) < 0 in all ten DR17 frames sampled, and in
seven galaxies from three near-axis-aligned frames the dihedral transform best matching the
SkyServer JPEG onto our stamp was a transpose each time. **Our stamps are mirror images of the
GZ1 JPEGs** (up to a rotation): a galaxy GZ1 calls clockwise winds anticlockwise in our arrays.
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import r_nonlinear as R  # noqa: E402
import w2_name_pcs as W  # noqa: E402

from galaxy_jepa.models.vit import VisionTransformer, load_frozen_encoder  # noqa: E402
from galaxy_jepa.probing import nulls as nulls_mod  # noqa: E402
from galaxy_jepa.probing.extract import extract_matrix  # noqa: E402

N_GROUP = 1000  # per group in X1c
MIN_GROUP = 300
N_BOOT = 1000
SEED = 0
TRANSFORMS = ("orig", "rot90", "rot180", "mirror")
BANK = R.OUT / "x1_transform_bank.npz"
OUT = R.OUT / "x1_handedness.json"
PLANTED = R.OUT / "x1_planted.json"
GZ1 = R.OUT / "ext" / "gz1_table2.csv.gz"
COL = {"feat": "t01_smooth_or_features_a02_features_or_disk_fraction",
       "smooth": "t01_smooth_or_features_a01_smooth_fraction",
       "edge_no": "t02_edgeon_a05_no_fraction",
       "spiral": "t04_spiral_a08_spiral_fraction",
       "sp_n": "t04_spiral_a08_spiral_count", "nosp_n": "t04_spiral_a09_no_spiral_count"}


def transform(img: torch.Tensor, name: str) -> torch.Tensor:
    """The dihedral transforms, on the last two axes. The stamp centre (127.5) and the 16-px
    patch grid are both fixed by all of them, so nothing but the image content moves."""
    if name == "orig":
        return img
    if name == "rot90":
        return torch.rot90(img, 1, dims=(-2, -1))
    if name == "rot180":
        return torch.rot90(img, 2, dims=(-2, -1))
    if name == "mirror":
        return torch.flip(img, dims=(-1,))
    raise ValueError(name)


class Transformed(Dataset):
    def __init__(self, base, index: np.ndarray, name: str) -> None:
        self.base, self.index, self.name = base, index, name

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        item = dict(self.base[int(self.index[i])])
        item["image"] = transform(item["image"], self.name)
        return item


class Tensors(Dataset):
    """In-memory images (the synthetic spirals) behind the same ``image``/``object_id`` item."""

    def __init__(self, images: torch.Tensor, name: str) -> None:
        self.images, self.name = images, name

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> dict:
        return {"image": transform(self.images[i], self.name), "object_id": i}


# ------------------------------------------------------------------ pixel-feature encoders


class PixelFeatures(torch.nn.Module):
    """Three features with known symmetry, read from the r band, centred on pixel 127.5.

    chi   log-polar Fourier chirality: sum over m=1..4, k>0 of P(k,m) − P(−k,m), over total
          power. Rotation shifts theta (phase only), reflection sends P(k,m) to P(−k,m) — so
          chi is rotation-invariant and reflection-odd *exactly* on this grid: 128 theta bins
          at half-bin offsets map onto themselves under rot90 and under x → −x.
    lr    (right − left) / total absolute flux inside r < 40 px: orientation-like.
    flux  total flux inside r < 40 px: invariant to both.
    """

    def __init__(self, n_r: int = 64, n_t: int = 128, r_min: float = 2.0, r_max: float = 80.0):
        super().__init__()
        r = torch.exp(torch.linspace(np.log(r_min), np.log(r_max), n_r))
        t = 2 * np.pi * (torch.arange(n_t) + 0.5) / n_t
        x = (r[:, None] * torch.cos(t)[None, :]) / 127.5  # align_corners: 0 ↔ pixel 127.5
        y = (r[:, None] * torch.sin(t)[None, :]) / 127.5
        self.register_buffer("grid", torch.stack([x, y], -1)[None], persistent=False)
        self.register_buffer("win", torch.hann_window(n_r, periodic=False)[:, None],
                             persistent=False)
        yy, xx = torch.meshgrid(torch.arange(256) - 127.5, torch.arange(256) - 127.5,
                                indexing="ij")
        self.register_buffer("disc", ((xx**2 + yy**2) < 40**2).float(), persistent=False)
        self.register_buffer("right", (xx > 0).float() * 2 - 1, persistent=False)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        im = images[:, 1:2].float()
        lp = torch.nn.functional.grid_sample(im, self.grid.expand(len(im), -1, -1, -1),
                                             mode="bilinear", align_corners=True)[:, 0]
        lp = lp - lp.mean(-1, keepdim=True)  # m = 0 removed
        f = torch.fft.fft2(lp * self.win)
        p = f.real**2 + f.imag**2  # (B, k, m)
        n_r = p.shape[1]
        kpos, kneg = slice(1, n_r // 2), [(-k) % n_r for k in range(1, n_r // 2)]
        ms = slice(1, 5)
        chi = (p[:, kpos, ms].sum((1, 2)) - p[:, kneg, ms].sum((1, 2)))
        chi = chi / (p[:, :, 0:5].sum((1, 2)) + 1e-12)
        d = im[:, 0] * self.disc
        flux = d.sum((1, 2))
        lr = (d * self.right).sum((1, 2)) / (d.abs().sum((1, 2)) + 1e-12)
        return torch.stack([chi, lr, flux], 1)


PIXEL_NAMES = ("chi", "lr", "flux")


def pixel_encoder() -> PixelFeatures:
    enc = PixelFeatures().eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


# ------------------------------------------------------------------ synthetic spirals


def synthetic_spirals(n: int, seed: int, sky: float, peak: float) -> tuple[torch.Tensor,
                                                                             np.ndarray]:
    """Two-armed logarithmic spirals, half each handedness; exponential disc + bulge + noise.

    In array coordinates (row index down, column right) the arm phase is
    theta − s·ln(r)/tan(pitch): s = +1 winds anticlockwise outward as displayed row-down.
    Out of distribution by construction — a check that the path can see handedness, not a
    claim about real galaxies.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(256) - 127.5, np.arange(256) - 127.5, indexing="ij")
    r = np.hypot(xx, yy) + 1e-3
    th = np.arctan2(-yy, xx)  # displayed row-down: angle measured anticlockwise on screen
    s = np.where(np.arange(n) < n // 2, 1, -1)
    rng.shuffle(s)
    out = np.empty((n, 3, 256, 256), np.float32)
    for i in range(n):
        h = rng.uniform(8, 18)
        pitch = np.deg2rad(rng.uniform(10, 30))
        phase = rng.uniform(0, 2 * np.pi)
        arms = 1 + 0.8 * np.cos(2 * (th - s[i] * np.log(r) / np.tan(pitch)) - phase)
        disc = np.exp(-r / h) * arms * (1 - np.exp(-(r / 4) ** 2))
        bulge = 1.5 * np.exp(-((r / rng.uniform(2, 4)) ** 2))
        img = (disc + bulge) / 2.5
        for b, c in enumerate((0.7, 1.0, 1.2)):
            out[i, b] = peak * c * img + rng.normal(0, sky, img.shape)
    return torch.from_numpy(out), s


# ------------------------------------------------------------------ the signature (shared)


def _r(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else float("nan")


def signature(s: dict[str, np.ndarray], spir: np.ndarray, smooth: np.ndarray,
              seed: int = SEED) -> dict:
    """Per component: r under rot90 / rot180 / mirror on spirals (and smooth), the odd-variance
    fraction, and the spiral/smooth odd-variance ratio — with bootstrap CIs over galaxies."""
    rng = np.random.default_rng(seed)
    o, r90, r180, m = s["orig"], s["rot90"], s["rot180"], s["mirror"]
    odd = (o - m) / 2
    out = []
    for j in range(o.shape[1]):
        def stats(ix_sp, ix_sm, j=j):
            rs = {"r90": _r(o[ix_sp, j], r90[ix_sp, j]), "r180": _r(o[ix_sp, j], r180[ix_sp, j]),
                  "r_mir": _r(o[ix_sp, j], m[ix_sp, j])}
            rs["r_rot"] = min(rs["r90"], rs["r180"])
            ev = (o[ix_sp, j] + m[ix_sp, j]) / 2
            rs["odd_frac"] = float(odd[ix_sp, j].var() / (odd[ix_sp, j].var() + ev.var()))
            rs["conc"] = float(odd[ix_sp, j].var() / max(odd[ix_sm, j].var(), 1e-30))
            return rs
        sp, sm = np.flatnonzero(spir), np.flatnonzero(smooth)
        point = stats(sp, sm)
        boots = [stats(rng.choice(sp, sp.size), rng.choice(sm, sm.size)) for _ in range(N_BOOT)]
        ci = {k: [float(np.percentile([b[k] for b in boots], q)) for q in (2.5, 97.5)]
              for k in point}
        smooth_rs = {"r_rot": min(_r(o[sm, j], r90[sm, j]), _r(o[sm, j], r180[sm, j])),
                     "r_mir": _r(o[sm, j], m[sm, j])}
        out.append({**point, "ci": ci, "smooth": smooth_rs, "state": symmetry_state(point)})
    return {"components": out, "n_spiral": int(spir.sum()), "n_smooth": int(smooth.sum())}


def symmetry_state(p: dict) -> str:
    """X1c's per-component states — complete over (r_rot, r_mir) by construction (D27)."""
    if p["r_rot"] <= 0.5:
        return "ORIENTATION-LIKE"
    if p["r_rot"] < 0.8:
        return "ROTATION-SENSITIVE (WEAK)"
    if p["r_mir"] <= -0.5:
        return "HANDEDNESS" if p["conc"] >= 2 else "REFLECTION-ODD, NOT SPIRAL-CONCENTRATED"
    if p["r_mir"] >= 0.8:
        return "INVARIANT TO BOTH"
    return "PARTLY REFLECTION-SENSITIVE"


def odd_direction(s: dict[str, np.ndarray], fit: np.ndarray) -> np.ndarray:
    """The most reflection-odd direction of the given subspace, fitted on ``fit`` rows only:
    max u'C_odd u / u'C u, a generalised eigenproblem."""
    from scipy.linalg import eigh

    o, m = s["orig"][fit], s["mirror"][fit]
    c_odd = np.cov(((o - m) / 2).T).reshape(o.shape[1], -1)
    c = np.cov(o.T).reshape(o.shape[1], -1)
    _, v = eigh(c_odd, c + 1e-9 * np.trace(c) * np.eye(len(c)))
    u = v[:, -1]
    return u / np.linalg.norm(u)


def whole_odd_share(s: dict[str, np.ndarray], rows: np.ndarray) -> float:
    o, m = s["orig"][rows], s["mirror"][rows]
    odd, tot = (o - m) / 2, o
    return float(np.trace(np.cov(odd.T)) / np.trace(np.cov(tot.T)))


# ------------------------------------------------------------------ inputs


def f(row: dict, k: str) -> float:
    v = row.get(COL[k], "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def groups(setup, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reach = np.array([f(setup.rows[int(o)], "sp_n") + f(setup.rows[int(o)], "nosp_n")
                      for o in ids])
    get = {k: np.array([f(setup.rows[int(o)], k) for o in ids]) for k in COL}
    spir = ((get["feat"] >= 0.8) & (get["spiral"] >= 0.8) & (get["edge_no"] >= 0.8)
            & (reach >= 21))
    smooth = get["smooth"] >= 0.8
    return spir, smooth


def sample(setup) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X1c's sample: up to N_GROUP confident spirals and smooth from P2's test partition B."""
    test = np.asarray(sorted(setup.test_ids))
    spir, smooth = groups(setup, test)
    rng = np.random.default_rng(SEED)
    a = rng.choice(np.flatnonzero(spir), min(N_GROUP, int(spir.sum())), replace=False)
    b = rng.choice(np.flatnonzero(smooth), min(N_GROUP, int(smooth.sum())), replace=False)
    ids = test[np.concatenate([a, b])]
    is_sp = np.r_[np.ones(a.size, bool), np.zeros(b.size, bool)]
    print(f"X1c sample: {a.size} spirals (of {int(spir.sum())}), {b.size} smooth "
          f"(of {int(smooth.sum())}) from B", file=sys.stderr)
    return ids, is_sp, ~is_sp


def ds_index(setup, ids: np.ndarray) -> np.ndarray:
    pos = {int(o): i for i, o in enumerate(setup.ds.object_ids)}
    return np.array([pos[int(o)] for o in ids])


def basis(ctx, setup) -> list[dict]:
    """V3's covariance PCA on A — mean, top-10 axes, A's sd per axis — per encoder."""
    pos = {int(o): i for i, o in enumerate(ctx.real.object_ids)}
    A = np.array([pos[int(o)] for o in setup.train_ids])
    out = []
    for m in [ctx.real, *ctx.untrained]:
        xa = m.x[A].astype(np.float64)
        mu = xa.mean(0)
        _, v = np.linalg.eigh(np.cov(xa, rowvar=False))
        v = v[:, ::-1][:, :10]
        sd = ((xa - mu) @ v).std(0)
        out.append({"name": m.encoder_name, "mu": mu, "v": v, "sd": sd, "x": m.x})
    return out


def embed_all(encoder, base, index, device) -> dict[str, np.ndarray]:
    return {t: extract_matrix(encoder, Transformed(base, index, t), device=device).x
            for t in TRANSFORMS}


# ------------------------------------------------------------------ modes


def bank(setup, frozen) -> None:
    ids, _, _ = sample(setup)
    idx = ds_index(setup, ids)
    blob: dict = {"ids": ids}
    encs = [("M", frozen)]
    for s in range(3):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(s)
            u = VisionTransformer(**dict(frozen.config))
        u.eval().requires_grad_(False)
        encs.append((f"untrained-s{s}", u))
    for name, enc in encs:
        t0 = time.perf_counter()
        for t, x in embed_all(enc, setup.ds, idx, setup.device).items():
            blob[f"{name}:{t}"] = x
        print(f"  bank {name}: {time.perf_counter() - t0:.0f}s", file=sys.stderr)
        R._release(setup.device)
    np.savez(BANK, **blob)


def planted(setup, frozen) -> dict:
    ids, sp, sm = sample(setup)
    idx = ds_index(setup, ids)
    enc = pixel_encoder()
    feats = embed_all(enc, setup.ds, idx, "cpu")
    sig = signature(feats, sp, sm)
    out: dict = {"pixel": {n: sig["components"][j] for j, n in enumerate(PIXEL_NAMES)}}
    expect = {"chi": "HANDEDNESS", "lr": "ORIENTATION-LIKE", "flux": "INVARIANT TO BOTH"}
    out["pixel_pass"] = {n: out["pixel"][n]["state"] == e for n, e in expect.items()}

    # X1a's planted magnitude variables: a known spirality score, shifted so the confident-smooth
    # median is zero, times a random sign per galaxy. The logistic score is the positive (lobes
    # 2.5 sd apart among spirals); the CAV projection (1.3 sd) is kept as the case below the
    # bimodality check's resolution (it resolves ±1.5 sd mixtures, not ±1.0).
    ctx = R.Ctx(setup, frozen, 0, dry=True)
    union = ctx.real.object_ids
    spir_u, smooth_u = groups(setup, union)
    vote = {k: np.array([f(setup.rows[int(o)], k) for o in union]) for k in ("spiral", "feat")}
    reach = np.array([f(setup.rows[int(o)], "sp_n") + f(setup.rows[int(o)], "nosp_n")
                      for o in union])
    x = ctx.real.x.astype(np.float64)
    c = x[spir_u].mean(0) - x[smooth_u].mean(0)
    proj = x @ c
    mag = proj - np.median(proj[smooth_u])
    eps = np.random.default_rng(7).choice([-1.0, 1.0], union.size)
    from sklearn.linear_model import LogisticRegression

    zx = (x - x.mean(0)) / x.std(0)
    rows = spir_u | smooth_u
    w = LogisticRegression(max_iter=3000).fit(zx[rows], spir_u[rows]).coef_[0]
    lg = zx @ w
    lmag = lg - np.median(lg[smooth_u])
    out["x1a_planted"] = x1a_tests({"planted_logit": eps * lmag, "planted_cav": eps * mag},
                                   vote, reach, spir_u, smooth_u)
    out["x1a_planted"]["lobe_separation_sd"] = {
        "planted_logit": float(lmag[spir_u].mean() / lmag[spir_u].std()),
        "planted_cav": float(mag[spir_u].mean() / mag[spir_u].std())}

    # synthetic spirals (OOD): chi must read the handedness; M descriptive.
    sm_img = torch.stack([setup.ds[int(i)]["image"].float() for i in idx[sp][:200]])
    sky = float(sm_img[:, 1, :20, :20].std())
    peak = float(sm_img[:, 1, 120:136, 120:136].mean())
    imgs, hand = synthetic_spirals(400, SEED, sky, peak)
    syn = {t: extract_matrix(enc, Tensors(imgs, t), device="cpu").x for t in TRANSFORMS}
    chi = syn["orig"][:, 0]
    out["synthetic"] = {"sky": sky, "peak": peak,
                        "chi_auc_vs_s": _auc(hand > 0, chi),
                        "chi_signature": signature(syn, np.ones(400, bool), np.ones(400, bool)
                                                   )["components"][0]}
    msyn = {t: extract_matrix(frozen, Tensors(imgs, t), device=setup.device).x
            for t in TRANSFORMS}
    R._release(setup.device)
    out["synthetic"]["M"] = synthetic_m(msyn, hand, basis(ctx, setup)[0])
    return out


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def synthetic_m(msyn: dict[str, np.ndarray], hand: np.ndarray, b: dict) -> dict:
    """Does M separate synthetic handedness at all, and does PC1/PC2 carry it? Descriptive."""
    from sklearn.linear_model import LogisticRegression

    n = hand.size
    half = np.arange(n) < n // 2
    x = msyn["orig"]
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(x[half], hand[half])
    z = {t: ((v - b["mu"]) @ b["v"]) / b["sd"] for t, v in msyn.items()}
    sig = signature({t: v[:, :2] for t, v in z.items()}, np.ones(n, bool), np.ones(n, bool))
    return {"heldout_auc_full_embedding": _auc(hand[~half] > 0,
                                                clf.decision_function(x[~half])),
            "pc1_auc": _auc(hand > 0, z["orig"][:, 0]), "pc2_auc": _auc(hand > 0, z["orig"][:, 1]),
            "pc12_signature": [{k: c[k] for k in ("r_rot", "r_mir", "odd_frac", "state")}
                               for c in sig["components"]],
            "whole_odd_share": whole_odd_share(msyn, np.arange(n))}


# ---------------------------------------------------------------- X1a


def bimodality(v: np.ndarray, seed: int) -> dict:
    """KDE mode structure at Silverman's bandwidth, and its stability over 200 bootstraps.
    BIMODAL: two modes on opposite sides of 0 with a dip ≤ 0.8 x the lower mode, in ≥ 90%.
    UNIMODAL CENTRED: one mode with |mode| ≤ 0.5 sd, in ≥ 90%. Otherwise NEITHER."""
    from scipy.stats import gaussian_kde

    rng = np.random.default_rng(seed)
    v = v[np.isfinite(v)]
    sd = v.std()
    grid = np.linspace(-3 * sd, 3 * sd, 301)

    def one(x):
        d = gaussian_kde(x, bw_method="silverman")(grid)
        peaks = [i for i in range(1, 300) if d[i] > d[i - 1] and d[i] >= d[i + 1]
                 and d[i] > 0.05 * d.max()]
        top = sorted(peaks, key=lambda i: -d[i])[:2]
        if len(top) == 2:
            lo, hi = sorted(top)
            dip = d[lo:hi + 1].min()
            if grid[lo] < 0 < grid[hi] and dip <= 0.8 * min(d[lo], d[hi]):
                return "BIMODAL"
        if len(peaks) == 1 and abs(grid[peaks[0]]) <= 0.5 * sd:
            return "UNIMODAL CENTRED"
        return "NEITHER"

    calls = [one(rng.choice(v, v.size)) for _ in range(200)]
    frac = {k: calls.count(k) / 200 for k in ("BIMODAL", "UNIMODAL CENTRED", "NEITHER")}
    call = ("BIMODAL" if frac["BIMODAL"] >= 0.9 else
            "UNIMODAL CENTRED" if frac["UNIMODAL CENTRED"] >= 0.9 else "NEITHER")
    return {"call": call, "frac": frac, "point": one(v)}


def x1a_tests(variables: dict[str, np.ndarray], vote: dict, reach: np.ndarray,
              spir: np.ndarray, smooth: np.ndarray) -> dict:
    """Magnitude and signed Spearman against t04 spiral (reach ≥ 21) and t01 featured, BY over
    the family; bimodality among confident spirals, centring among confident smooth."""
    elig = {"spiral": reach >= 21, "feat": np.isfinite(vote["feat"])}
    p, rho = {}, {}
    for name, v in variables.items():
        for form, val in (("magnitude", np.abs(v)), ("signed", v)):
            for k in ("spiral", "feat"):
                key = f"{name}|{form}|{k}"
                rho[key], p[key] = W.spearman_perm(val[elig[k]], vote[k][elig[k]],
                                                   seed=len(key) * 31 + 3)
    fam = len(p)
    nulls_mod.assert_null_resolution(W.N_PERM, alpha=W.ALPHA, method="benjamini_yekutieli",
                                     n_tests=fam)
    sig = nulls_mod.family_significant(p, alpha=W.ALPHA, method="benjamini_yekutieli",
                                       n_tests=fam)
    out: dict = {"tests": {k: {"rho": rho[k], "p": p[k], "sig": sig[k]} for k in p}}
    for name, v in variables.items():
        sd = np.nanstd(v)
        bim_sp = bimodality(v[spir] / sd, 11)
        bim_sm = bimodality(v[smooth] / sd, 12)
        mag = [out["tests"][f"{name}|magnitude|{k}"] for k in ("spiral", "feat")]
        sgn = [out["tests"][f"{name}|signed|{k}"] for k in ("spiral", "feat")]
        out[name] = {"bimodal_spirals": bim_sp, "centred_smooth": bim_sm,
                     "state": x1a_state(mag, sgn, bim_sp["call"], bim_sm["call"],
                                        int(spir.sum()), int(smooth.sum()))}
    return out


def x1a_state(mag, sgn, bim_sp, bim_sm, n_sp, n_sm) -> str:
    """First match applies (D27: complete)."""
    tracks = any(t["sig"] and t["rho"] >= 0.1 for t in mag)
    inverse = any(t["sig"] and t["rho"] <= -0.1 for t in mag)
    signed = any(t["sig"] and abs(t["rho"]) >= 0.1 for t in sgn)
    if min(n_sp, n_sm) < MIN_GROUP:
        return "INSUFFICIENT"
    if signed:
        return "SIGNED TRACKS"
    if tracks and bim_sp == "BIMODAL" and bim_sm == "UNIMODAL CENTRED":
        return "CONSISTENT WITH HANDEDNESS"
    if tracks:
        return "MAGNITUDE ONLY"
    if inverse:
        return "INVERSE MAGNITUDE"
    return "NO ASSOCIATION"


# ---------------------------------------------------------------- X1b


def gz1(setup, union: np.ndarray) -> dict[str, np.ndarray]:
    # read as strings: int(float(id)) rounds an 18-digit objID and matches nothing
    want = {int(o) for o in union}
    dr7 = {}
    with open(__import__("pathlib").Path(setup.cfg.paths.probe_dir) / "metadata.csv") as fh:
        for r in csv.DictReader(fh):
            if int(r["object_id"]) in want and r["dr7objid"].strip():
                dr7[int(r["dr7objid"])] = int(r["object_id"])
    pos = {int(o): i for i, o in enumerate(union)}
    out = {k: np.full(union.size, np.nan) for k in ("cw", "acw", "nvote")}
    with gzip.open(GZ1, "rt") as fh:
        for r in csv.DictReader(fh):
            o = dr7.get(int(r["OBJID"]))
            if o is None:
                continue
            i = pos[o]
            out["cw"][i], out["acw"][i] = float(r["P_CW"]), float(r["P_ACW"])
            out["nvote"][i] = float(r["NVOTE"])
    return out


def x1b(setup, ctx, z: np.ndarray, spir: np.ndarray) -> dict:
    """GZ1 handedness h = P_CW − P_ACW (GZ1's JPEG frame) against PC1, PC2 and the plane
    direction fitted on A, tested on B. Our frame is mirrored, so h_ours = −h."""
    union = ctx.real.object_ids
    g = gz1(setup, union)
    h = g["cw"] - g["acw"]
    ok = spir & np.isfinite(h) & (g["nvote"] >= 10)
    test = np.isin(union, np.asarray(setup.test_ids))
    A, B = ok & ~test, ok & test
    rec: dict = {"n_matched_union": int(np.isfinite(h).sum()), "n_A": int(A.sum()),
                 "n_B": int(B.sum()), "mean_h_B": float(np.mean(h[B])) if B.any() else None}
    if B.sum() < MIN_GROUP:
        rec["state"] = "INSUFFICIENT"
        return rec
    angles = np.linspace(0, np.pi, 180, endpoint=False)
    rhos = [abs(_spearman(np.cos(a) * z[A, 0] + np.sin(a) * z[A, 1], h[A])) for a in angles]
    phi = float(angles[int(np.argmax(rhos))])
    plane = np.cos(phi) * z[:, 0] + np.sin(phi) * z[:, 1]
    p, rho = {}, {}
    for name, v in (("PC1", z[:, 0]), ("PC2", z[:, 1]), ("plane", plane)):
        rho[name], p[name] = W.spearman_perm(v[B], h[B], seed=len(name) + 41)
    nulls_mod.assert_null_resolution(W.N_PERM, alpha=W.ALPHA, method="benjamini_yekutieli",
                                     n_tests=3)
    sig = nulls_mod.family_significant(p, alpha=W.ALPHA, method="benjamini_yekutieli",
                                       n_tests=3)
    rec["plane_angle_deg_fitted_on_A"] = float(np.degrees(phi))
    rec["tests"] = {k: {"rho": rho[k], "p": p[k], "sig": sig[k]} for k in p}
    rec["magnitude_desc"] = {k: _spearman(np.abs(v[B]), np.abs(h[B])) for k, v in
                             (("PC1", z[:, 0]), ("PC2", z[:, 1]), ("plane", plane))}
    rec["state"] = ("SIGNED ASSOCIATION" if any(sig[k] and abs(rho[k]) >= 0.1 for k in p)
                    else "WEAK SIGNED ASSOCIATION" if any(sig.values()) else "NO ASSOCIATION")

    # chi on our stamps against GZ1: the parity check through the classifications. chi is
    # higher for s = +1 (anticlockwise outward in our row-down array); mirrored, that is
    # clockwise (Z-wise) in the JPEG, so the mirror parity predicts rho(chi, h) > 0.
    rows = np.flatnonzero(B)
    chi = extract_matrix(pixel_encoder(), Transformed(setup.ds, ds_index(setup, union[rows]),
                                                      "orig"), device="cpu").x[:, 0]
    rc, pc = W.spearman_perm(chi, h[rows], seed=97)
    rec["chi_vs_h"] = {"rho": rc, "p": pc, "n": int(rows.size),
                       "parity": ("MIRROR CONFIRMED" if pc < W.ALPHA and rc > 0 else
                                  "MIRROR CONTRADICTED" if pc < W.ALPHA else "UNRESOLVED")}
    p2, r2 = {}, {}
    for name, v in (("PC1", z[rows, 0]), ("PC2", z[rows, 1]), ("plane", plane[rows])):
        r2[name], p2[name] = W.spearman_perm(v, chi, seed=len(name) + 53)
    s2 = nulls_mod.family_significant(p2, alpha=W.ALPHA, method="benjamini_yekutieli",
                                      n_tests=3)
    rec["vs_chi"] = {k: {"rho": r2[k], "p": p2[k], "sig": s2[k]} for k in p2}
    return rec


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


# ---------------------------------------------------------------- X1c


def x1c(setup) -> dict:
    blob = np.load(BANK)
    ids = blob["ids"]
    spir, smooth = groups(setup, ids)
    ctx = R.Ctx(setup, load_frozen_encoder(setup.ckpt), 0, dry=True)
    bases = basis(ctx, setup)
    names = ["M", "untrained-s0", "untrained-s1", "untrained-s2"]
    rec: dict = {"n_spiral": int(spir.sum()), "n_smooth": int(smooth.sum())}
    half = np.random.default_rng(SEED).random(ids.size) < 0.5
    for name, b in zip(names, bases, strict=True):
        raw = {t: blob[f"{name}:{t}"].astype(np.float64) for t in TRANSFORMS}
        z = {t: ((v - b["mu"]) @ b["v"]) / b["sd"] for t, v in raw.items()}
        top = signature(z, spir, smooth)
        u = odd_direction({t: v[:, :2] for t, v in z.items()}, spir & half)
        pl = {t: (v[:, :2] @ u)[:, None] for t, v in z.items()}
        plane = signature(pl, spir & ~half, smooth)
        u10 = odd_direction(z, spir & half)
        p10 = signature({t: (v @ u10)[:, None] for t, v in z.items()}, spir & ~half, smooth)
        uall = odd_direction(raw, spir & half)
        pall = signature({t: (v @ uall)[:, None] for t, v in raw.items()}, spir & ~half, smooth)
        rec[name] = {"top10": top["components"],
                     "plane_odd": {"angle_deg": float(np.degrees(np.arctan2(u[1], u[0]))),
                                   **plane["components"][0]},
                     "top10_odd": p10["components"][0], "whole_odd": pall["components"][0],
                     "whole_odd_share": {"spirals": whole_odd_share(raw, np.flatnonzero(spir)),
                                         "smooth": whole_odd_share(raw, np.flatnonzero(smooth))}}
        print(f"X1c {name}: PC1 {top['components'][0]['state']} "
              f"(r_rot {top['components'][0]['r_rot']:+.2f} r_mir "
              f"{top['components'][0]['r_mir']:+.2f}); PC2 {top['components'][1]['state']} "
              f"(r_rot {top['components'][1]['r_rot']:+.2f} r_mir "
              f"{top['components'][1]['r_mir']:+.2f}); plane-odd {plane['components'][0]['state']}"
              f" r_mir {plane['components'][0]['r_mir']:+.2f}", file=sys.stderr)
    unt_min = [min(c["r_mir"] for c in rec[n]["top10"]) for n in names[1:]]
    for key, comp in (("PC1", rec["M"]["top10"][0]), ("PC2", rec["M"]["top10"][1]),
                      ("plane", rec["M"]["plane_odd"])):
        comp["qualifier"] = "LEARNED" if comp["r_mir"] < min(unt_min) else "ARCHITECTURAL-LEVEL"
    rec["untrained_min_rmir_top10"] = unt_min
    return rec


def overall(c: dict) -> str:
    """X1's verdict from X1c (first match applies)."""
    if min(c["n_spiral"], c["n_smooth"]) < MIN_GROUP:
        return "INSUFFICIENT"
    s1, s2 = c["M"]["top10"][0]["state"], c["M"]["top10"][1]["state"]
    sp = c["M"]["plane_odd"]["state"]
    if "HANDEDNESS" in (s1, s2, sp):
        return "HANDEDNESS"
    if s1 == s2 == "ORIENTATION-LIKE":
        return "ORIENTATION-LIKE"
    if s1 == s2 == "INVARIANT TO BOTH":
        return "INVARIANT TO BOTH"
    return "MIXED"


def run(setup) -> dict:
    frozen = load_frozen_encoder(setup.ckpt)
    ctx = R.Ctx(setup, frozen, 0, dry=True)
    union = ctx.real.object_ids
    b = basis(ctx, setup)[0]
    z = ((ctx.real.x.astype(np.float64) - b["mu"]) @ b["v"][:, :2]) / b["sd"][:2]
    spir, smooth = groups(setup, union)
    vote = {k: np.array([f(setup.rows[int(o)], k) for o in union]) for k in ("spiral", "feat")}
    reach = np.array([f(setup.rows[int(o)], "sp_n") + f(setup.rows[int(o)], "nosp_n")
                      for o in union])
    rec: dict = {"n_confident": {"spiral": int(spir.sum()), "smooth": int(smooth.sum())}}
    rec["x1a"] = x1a_tests({"PC1": z[:, 0], "PC2": z[:, 1], "radius": np.hypot(z[:, 0], z[:, 1])},
                           vote, reach, spir, smooth)
    unt = []
    for bu in basis(ctx, setup)[1:]:
        zu = ((bu["x"].astype(np.float64) - bu["mu"]) @ bu["v"][:, :2]) / bu["sd"][:2]
        unt.append({k: _spearman(np.abs(v)[reach >= 21], vote["spiral"][reach >= 21])
                    for k, v in (("PC1", zu[:, 0]), ("PC2", zu[:, 1]),
                                 ("radius", np.hypot(zu[:, 0], zu[:, 1])))})
    rec["x1a"]["untrained_magnitude_rho_spiral"] = unt
    rec["x1b"] = x1b(setup, ctx, z, spir)
    rec["x1c"] = x1c(setup)
    rec["verdict"] = overall(rec["x1c"])
    return rec


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--run"
    setup = R.prepare("runs/m/encoder.pt", R.MAX_TRAIN, label="X1", sources=1)
    if mode == "--bank":
        bank(setup, load_frozen_encoder(setup.ckpt))
        return
    if mode == "--planted":
        out = planted(setup, load_frozen_encoder(setup.ckpt))
        PLANTED.write_text(json.dumps(out, indent=1, default=float))
        print(json.dumps(out, indent=1, default=float), file=sys.stderr)
        return
    rec = run(setup)
    OUT.write_text(json.dumps(rec, indent=1, default=float))
    print(f"X1 verdict: {rec['verdict']}", file=sys.stderr)


if __name__ == "__main__":
    main()
