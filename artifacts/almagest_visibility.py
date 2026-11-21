"""Almagest visibility tests on A1: exploratory, embeddings plus inference only, no training. Not in the science record;
feeds the 8×8 pre-registration.

  b1   resolution curves: per STRONG/SEPARATE answer, cleaned and raw probe AUC in deciles of petroRad_r / psfWidth_r
       (test galaxies eligible for the answer), bootstrap CIs, untrained-seed range per decile, knee
  b2   artificial redshifting (FERENGI-style, simplified) of low-z, large test galaxies to z = 0.07, 0.10, 0.13 plus a
       planted no-change run at each galaxy's own z; re-embed with A1; score movement, nuisance movement, NN-specz check

B2 RESAMPLES IMAGES. It is an intervention test, separate from the training pipeline's no-rebin rule
(docs/spec/data.md): nothing it makes is cached, trained on or probed as data; it only asks how A1's frozen readout
moves when a galaxy is made to look more distant.

Stamps are a fixed 256 px at the native 0.396″/px (not scaled to galaxy size); petroRad_r enters pretraining only as
the JEPA target-sampling box. Run: .venv/bin/python artifacts/almagest_visibility.py b1|b1b|b2
(b1b: B1 on galaxies with question reach ≥ median and vote agreement ≥ 0.8, label agreement per decile; 21 Nov)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import almagest_axes as AXS  # noqa: E402
import almagest_spiral as S  # noqa: E402

DOCS, AX, ALM = AXS.DOCS, AXS.AX, AXS.ALM
OUT = ALM / "visibility"
SEED = 20261121
N_BOOT = 1000
KNEE_MARGIN = 0.05
PIX = 0.396  # arcsec / px, native
FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
Z_TARGETS = (0.07, 0.10, 0.13)
Z_MAX_SOURCE = 0.05
Z_MIN_SOURCE = 0.01  # declared deviation (21 Nov): z ≤ 0 has no D_A; below 0.01 peculiar velocity makes it ≥10% off
Z_WIN = 0.005  # real galaxies "at z": |specz − z| < Z_WIN
BORDER = 20  # px: the stamp's outer frame, for sky level and noise
NN_K = 10


def log(*a) -> None:
    print(*a, f"[load {os.getloadavg()[0]:.1f}]", file=sys.stderr, flush=True)


def ci(x) -> list[float]:
    return [float(v) for v in np.nanpercentile(x, [2.5, 97.5])]


def good_answers(out) -> list[str]:
    return [f for f in out["order"] if out["labels"][f].split(" of")[0] in AXS.GOOD]


# --- B1 ------------------------------------------------------------------------------------------------
def step_b1() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out = json.loads((DOCS / "a1_axes.json").read_text())
    assert "floored" in out, "run almagest_axes.py relabel2 first"
    feats = good_answers(out)
    d = AXS.Data(untrained=True)
    names = [k for k in d.mats if k != "A1"]
    size = d.lab.nuisance_value("size", d.union).astype(float)
    psf = d.lab.nuisance_value("psf", d.union).astype(float)
    res = size / psf
    recs = {}
    for f in feats:
        t0 = time.perf_counter()
        tr, te = d.rows(f)
        y_tr, _ = d.yv(f, tr); y_te, _ = d.yv(f, te)
        ok = np.isfinite(res[te])
        te, y_te = te[ok], y_te[ok]
        scores = {}
        for enc in ["A1"] + names:
            for kind, M in (("cleaned", d.clean[enc]), ("raw", d.mats[enc])):
                P = AXS.Probe(M[tr], y_tr, d.c)
                scores[enc, kind] = P.d(M[te])
        r_te = res[te]
        edges = np.quantile(r_te, np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(edges, r_te, side="right") - 1, 0, 9)
        rng = np.random.default_rng(SEED + out["order"].index(f))
        dec = []
        for k in range(10):
            m = np.flatnonzero(b == k)
            e = {"decile": k + 1, "ratio_range": [float(edges[k]), float(edges[k + 1])], "n": int(m.size),
                 "n_pos": int(y_te[m].sum())}
            for kind in ("cleaned", "raw"):
                a = AXS.fast_auc(y_te[m], scores["A1", kind][m])
                bs = []
                for _ in range(N_BOOT):
                    i = rng.choice(m, m.size)
                    bs.append(AXS.fast_auc(y_te[i], scores["A1", kind][i]))
                un = [AXS.fast_auc(y_te[m], scores[n, kind][m]) for n in names]
                e[kind] = {"auc": a, "ci95": ci(bs), "untrained": un, "untrained_max": float(np.nanmax(un)),
                           "clears": bool(a > np.nanmax(un) + KNEE_MARGIN)}
            dec.append(e)
        knee = {kind: next((e["decile"] for e in dec if e[kind]["clears"]), None) for kind in ("cleaned", "raw")}
        recs[f] = {"label": out["labels"][f], "n_test": int(len(te)), "deciles": dec, "knee": knee,
                   "auc_all": {kind: AXS.fast_auc(y_te, scores["A1", kind]) for kind in ("cleaned", "raw")}}
        log(f"b1 {f}: knee {knee} ({time.perf_counter() - t0:.0f}s)")
    res_out = {"note": "exploratory; not in the science record. Deciles of petroRad_r / psfWidth_r over each answer's "
                       "eligible test galaxies; probes fitted on its train rows (A1 and the 3 untrained seeds, cleaned and "
                       f"raw); bootstrap {N_BOOT} galaxy resamples within the decile. Knee: first decile (ascending "
                       f"resolution) where A1's AUC exceeds the untrained max by {KNEE_MARGIN}.",
               "answers": recs}
    (DOCS / "a1_visibility_b1.json").write_text(json.dumps(res_out, indent=1, default=float))
    fig_b1(recs)
    log("→ docs/almagest/a1_visibility_b1.json, a1_visibility_b1.png")


def fig_b1(recs) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = list(recs)
    cols = 5
    rows = -(-len(fs) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.5 * rows), squeeze=False)
    for ax, f in zip(axs.flat, fs):
        dec = recs[f]["deciles"]
        x = np.arange(1, 11)
        for kind, c in (("cleaned", "#1f5fa8"), ("raw", "#b5651d")):
            a = [e[kind]["auc"] for e in dec]
            lo = [e[kind]["ci95"][0] for e in dec]; hi = [e[kind]["ci95"][1] for e in dec]
            ax.fill_between(x, lo, hi, color=c, alpha=0.18, lw=0)
            ax.plot(x, a, color=c, lw=1.4, label=f"A1 {kind}")
            ax.plot(x, [e[kind]["untrained_max"] for e in dec], color=c, lw=0.8, ls=":")
        k = recs[f]["knee"]["cleaned"]
        if k:
            ax.axvline(k, color="#444", lw=0.7, ls="--")
        ax.set_title(f"{f.split('_', 2)[2] if f.count('_') >= 2 else f} (knee {k or '—'})", fontsize=7)
        ax.tick_params(labelsize=6); ax.set_ylim(0.4, 1.0)
    for ax in list(axs.flat)[len(fs):]:
        ax.axis("off")
    axs.flat[0].legend(fontsize=6)
    fig.supxlabel("decile of petroRad_r / psfWidth_r (test, per answer)", fontsize=8)
    fig.supylabel("AUC (dotted: untrained max)", fontsize=8)
    fig.tight_layout(); fig.savefig(DOCS / "a1_visibility_b1.png", dpi=130); plt.close(fig)


# --- B1b: B1 on well-voted, high-agreement galaxies ---------------------------------------------------------
AGREE_MIN = 0.8  # max answer fraction within the question (user brief, 21 Nov)
THIN = 10        # a decile with fewer than THIN of either class is flagged thin


def step_b1b() -> None:
    """B1 repeated on test galaxies whose question reach ≥ the answer's test median and whose vote agreement
    (the question's max answer fraction) ≥ 0.8. Declared choices: the probes are B1's (fitted on the same train
    rows, unfiltered), so only the evaluated population changes; the decile edges are B1's (all eligible test
    galaxies), so the curves are comparable bin for bin; the reach median is over the answer's eligible test rows.
    Beside the curves: label agreement per decile over all eligible test galaxies."""
    OUT.mkdir(parents=True, exist_ok=True)
    out = json.loads((DOCS / "a1_axes.json").read_text())
    b1 = json.loads((DOCS / "a1_visibility_b1.json").read_text())["answers"]
    feats = good_answers(out)
    assert set(feats) == set(b1), "B1b must cover B1's answers"
    d = AXS.Data(untrained=True)
    names = [k for k in d.mats if k != "A1"]
    res = d.lab.nuisance_value("size", d.union).astype(float) / d.lab.nuisance_value("psf", d.union).astype(float)
    spec = d.lab.scheme.by_name
    recs = {}
    for f in feats:
        t0 = time.perf_counter()
        tr, te = d.rows(f)
        y_tr, _ = d.yv(f, tr); y_te, _ = d.yv(f, te)
        ok = np.isfinite(res[te])
        te, y_te = te[ok], y_te[ok]
        sib = [g for g, sp in spec.items() if sp.question == spec[f].question]
        fr = np.column_stack([np.asarray(d.lab.vote_fraction(g, d.union[te]), dtype=np.float64) for g in sib])
        agree = np.nanmax(fr, 1)
        reach = d.reach(f, te)
        keep = (reach >= np.nanmedian(reach)) & (agree >= AGREE_MIN)
        scores = {}
        for enc in ["A1"] + names:
            for kind, M in (("cleaned", d.clean[enc]), ("raw", d.mats[enc])):
                scores[enc, kind] = AXS.Probe(M[tr], y_tr, d.c).d(M[te])
        edges = np.array([b1[f]["deciles"][0]["ratio_range"][0]] + [e["ratio_range"][1] for e in b1[f]["deciles"]])
        b = np.clip(np.searchsorted(edges, res[te], side="right") - 1, 0, 9)
        rng = np.random.default_rng(SEED + 500 + out["order"].index(f))
        dec = []
        for k in range(10):
            m_all = np.flatnonzero(b == k)
            m = m_all[keep[m_all]]
            e = {"decile": k + 1, "n_all": int(m_all.size), "n": int(m.size), "n_pos": int(y_te[m].sum()),
                 "agreement_median_all": float(np.median(agree[m_all])) if m_all.size else None,
                 "frac_agreement_ge_0.8_all": float(np.mean(agree[m_all] >= AGREE_MIN)) if m_all.size else None,
                 "frac_kept": float(m.size / m_all.size) if m_all.size else None}
            e["thin"] = bool(min(e["n_pos"], e["n"] - e["n_pos"]) < THIN)
            for kind in ("cleaned", "raw"):
                if len(np.unique(y_te[m])) < 2:
                    e[kind] = None
                    continue
                a = AXS.fast_auc(y_te[m], scores["A1", kind][m])
                bs = [AXS.fast_auc(y_te[i], scores["A1", kind][i]) for i in (rng.choice(m, m.size) for _ in range(N_BOOT))]
                un = [AXS.fast_auc(y_te[m], scores[n, kind][m]) for n in names]
                e[kind] = {"auc": a, "ci95": ci(bs), "untrained_max": float(np.nanmax(un)),
                           "clears": bool(a > np.nanmax(un) + KNEE_MARGIN), "b1_auc": b1[f]["deciles"][k][kind]["auc"]}
            dec.append(e)
        knee = {kind: next((e["decile"] for e in dec if e[kind] and e[kind]["clears"]), None) for kind in ("cleaned", "raw")}
        clears = {kind: int(sum(bool(e[kind] and e[kind]["clears"]) for e in dec)) for kind in ("cleaned", "raw")}
        recs[f] = {"label": out["labels"][f], "n_test_all": int(te.size), "n_test_kept": int(keep.sum()),
                   "reach_median": float(np.nanmedian(reach)), "siblings": sib, "deciles": dec, "knee": knee,
                   "deciles_clearing": clears, "b1_knee": b1[f]["knee"], "b1_deciles_clearing": {
                       kind: int(sum(e[kind]["clears"] for e in b1[f]["deciles"])) for kind in ("cleaned", "raw")}}
        log(f"b1b {f}: kept {keep.sum()}/{te.size}, knee {knee} (B1 {b1[f]['knee']}) ({time.perf_counter() - t0:.0f}s)")
    res_out = {"note": "exploratory; not in the science record. " + (step_b1b.__doc__ or "").replace("\n", " "),
               "agreement_min": AGREE_MIN, "thin_below": THIN, "answers": recs}
    (DOCS / "a1_visibility_b1b.json").write_text(json.dumps(res_out, indent=1, default=float))
    fig_b1b(recs)
    log("→ docs/almagest/a1_visibility_b1b.json, a1_visibility_b1b.png")


def fig_b1b(recs) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = list(recs)
    cols = 5
    rows = -(-len(fs) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.5 * rows), squeeze=False)
    x = np.arange(1, 11)
    for ax, f in zip(axs.flat, fs):
        dec = recs[f]["deciles"]
        ax.plot(x, [e["cleaned"]["b1_auc"] if e["cleaned"] else np.nan for e in dec], color="#9aa7b8", lw=1.2, label="B1, all")
        a = np.array([e["cleaned"]["auc"] if e["cleaned"] else np.nan for e in dec])
        lo = np.array([e["cleaned"]["ci95"][0] if e["cleaned"] else np.nan for e in dec])
        hi = np.array([e["cleaned"]["ci95"][1] if e["cleaned"] else np.nan for e in dec])
        ax.fill_between(x, lo, hi, color="#1f5fa8", alpha=0.18, lw=0)
        ax.plot(x, a, color="#1f5fa8", lw=1.4, label="B1b, kept")
        thin = [e["thin"] for e in dec]
        ax.scatter(x[thin], a[thin], s=10, facecolors="none", edgecolors="#1f5fa8", lw=0.8)
        ax.plot(x, [e["cleaned"]["untrained_max"] if e["cleaned"] else np.nan for e in dec], color="#1f5fa8", lw=0.8, ls=":")
        ax2 = ax.twinx()
        ax2.bar(x, [e["frac_agreement_ge_0.8_all"] or 0 for e in dec], color="#b5651d", alpha=0.18, width=0.8)
        ax2.set_ylim(0, 1); ax2.tick_params(labelsize=5, colors="#b5651d")
        ax.set_title(f"{f.split('_', 2)[2] if f.count('_') >= 2 else f} (kept {recs[f]['n_test_kept']}/{recs[f]['n_test_all']})", fontsize=7)
        ax.tick_params(labelsize=6); ax.set_ylim(0.4, 1.0); ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    for ax in list(axs.flat)[len(fs):]:
        ax.axis("off")
    axs.flat[0].legend(fontsize=6, loc="lower right")
    fig.supxlabel("decile of petroRad_r / psfWidth_r (B1's edges); bars: share of all eligible test galaxies with agreement ≥ 0.8", fontsize=8)
    fig.supylabel("cleaned AUC (dotted: untrained max; open: thin decile)", fontsize=8)
    fig.tight_layout(); fig.savefig(DOCS / "a1_visibility_b1b.png", dpi=130); plt.close(fig)


# --- B2 ------------------------------------------------------------------------------------------------
class Stretch:
    """The parity-locked pipeline (AsinhStretch(q) → frozen Normalise), with its exact inverse to linear frame flux."""

    def __init__(self) -> None:
        import f0_preconditions as F0
        from galaxy_jepa.data.transforms import Normalise
        from galaxy_jepa.harness import _build_pipeline
        self.cfg, self.cache = F0.check(verbose=False)
        pipe = _build_pipeline(q=self.cfg.q, freeze=self.cfg.normalisation)
        self.pipe = pipe
        norm = next(t for t in pipe.transforms if isinstance(t, Normalise))
        self.q = float(self.cfg.q)
        self.mu = np.asarray(norm.mean, float).reshape(-1, 1, 1)
        self.sd = np.asarray(norm.std, float).reshape(-1, 1, 1)
        self.aq = np.arcsinh(self.q)

    def to_linear(self, y):
        return np.sinh(self.aq * (np.asarray(y, float) * self.sd + self.mu)) / self.q

    def to_model(self, x):
        out = (np.arcsinh(self.q * x) / self.aq - self.mu) / self.sd
        return out.astype(np.float16)  # the cache's own dtype

    def check(self, y) -> float:
        """Round trip error of the inverse on a real stamp, and parity with the pipeline's forward."""
        x = self.to_linear(y)
        fwd = np.asarray(self.pipe(x), float)
        return float(np.max(np.abs(fwd - np.asarray(y, float))))


def border(a):
    m = np.ones(a.shape[-2:], bool)
    m[BORDER:-BORDER, BORDER:-BORDER] = False
    return a[..., m]


def sky(a):
    """Per-band sky level (median) and noise (1.4826 MAD) on the stamp's outer frame."""
    b = border(a)
    med = np.median(b, axis=-1)
    return med, 1.4826 * np.median(np.abs(b - med[:, None]), axis=-1)


def redshift_stamp(x, z0, z1, psf0, psf1, sig1, cosmo, rng, planted=False):
    """Linear (3, 256, 256) stamp at z0 → the same galaxy seen at z1. Angular size × D_A(z0)/D_A(z1); flux ×
    (D_L(z0)/D_L(z1))²; Gaussian PSF up to psf1 (r-band FWHM, all bands); Gaussian noise up to sig1 per band; centred in a
    256-px stamp whose outside is sky noise only. No k-correction. planted: z1 = z0, psf1 = psf0, sig1 = own noise."""
    from scipy.ndimage import gaussian_filter, zoom
    n = x.shape[-1]
    bg, _ = sky(x)
    if planted:
        return x.copy(), {"f": 1.0, "g": 1.0, "psf_add_px": 0.0, "sig_add": [0.0] * 3}
    f = float(cosmo.angular_diameter_distance(z0) / cosmo.angular_diameter_distance(z1))
    g = float((cosmo.luminosity_distance(z0) / cosmo.luminosity_distance(z1)) ** 2)
    s_aa = max((1.0 / f - 1.0) / 2.0, 0.0)  # anti-alias before shrinking, in input px
    m = int(round(n * f))
    assert m >= 16, f"shrink to {m} px (z0={z0}, z1={z1}): below the source floor?"
    eff2 = (f * psf0) ** 2 + (FWHM * s_aa * f * PIX) ** 2  # arcsec², FWHM after shrinking
    add = np.sqrt(max(psf1 ** 2 - eff2, 0.0)) / FWHM / PIX  # px
    out = np.empty_like(x)
    sig_add = []
    for c in range(x.shape[0]):
        im = x[c] - bg[c]
        if s_aa > 0:
            im = gaussian_filter(im, s_aa)
        sm = zoom(im, m / n, order=1) * (g * (n / m) ** 2)  # flux per new pixel
        if add > 0:
            sm = gaussian_filter(sm, add)
        canvas = np.zeros((n, n))
        o = (n - m) // 2
        canvas[o:o + m, o:o + m] = sm
        fb = np.ones((m, m), bool); w = max(2, int(BORDER * f)); fb[w:-w, w:-w] = False
        sres = 1.4826 * np.median(np.abs(sm[fb] - np.median(sm[fb])))
        sa_in = np.sqrt(max(sig1[c] ** 2 - sres ** 2, 0.0))
        noise = rng.normal(0.0, sig1[c], (n, n))
        noise[o:o + m, o:o + m] *= sa_in / sig1[c]
        out[c] = canvas + noise + bg[c]
        sig_add.append(float(sa_in))
    return out, {"f": f, "g": g, "psf_add_px": float(add), "sig_add": sig_add, "m": m}


def psf_check(cosmo, cases=((0.03, 0.07, 1.3, 1.4), (0.03, 0.10, 1.3, 1.5), (0.02, 0.13, 1.2, 1.5),
                             (0.045, 0.07, 1.6, 1.4))) -> list[dict]:
    """A Gaussian point source of FWHM psf0, shrunk to z1, must come out at max(psf1, s·psf0) within ±5%: the added blur
    is σ_add² = σ_target² − (s·σ_orig)² − (anti-alias σ, carried through the shrink)²."""
    n = 256
    yy, xx = np.mgrid[:n, :n] - n / 2 + 0.5
    rng = np.random.default_rng(0)

    def fwhm_of(im):
        w = np.clip(im, 0, None); w = w / w.sum()
        cy, cx = (w * yy).sum(), (w * xx).sum()
        return FWHM * np.sqrt((w * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum() / 2) * PIX

    out = []
    for z0, z1, p0, p1 in cases:
        s = p0 / FWHM / PIX
        x = np.stack([np.exp(-(yy ** 2 + xx ** 2) / (2 * s * s))] * 3)
        xs, info = redshift_stamp(x, z0, z1, p0, p1, [1e-12] * 3, cosmo, rng)
        tgt = max(p1, info["f"] * p0)
        got = fwhm_of(xs[1])
        out.append({"z0": z0, "z1": z1, "psf0": p0, "psf1": p1, "shrunk_only": info["f"] * p0, "out": got,
                    "rel_err": got / tgt - 1})
        assert abs(got / tgt - 1) < 0.05, f"PSF check failed: {out[-1]}"
    return out


def step_b2() -> None:
    import torch
    from astropy.cosmology import Planck18 as cosmo
    import almagest_diffusion as DF
    OUT.mkdir(parents=True, exist_ok=True)
    out = json.loads((DOCS / "a1_axes.json").read_text())
    order = out["order"]
    d = AXS.Data(untrained=False)
    st = Stretch()
    N = {n: d.lab.nuisance_value(n, d.union).astype(float) for n in S.NUIS}
    test_rows = np.flatnonzero(np.isin(d.union, np.asarray(d.setup.test_ids)) & d.ok)
    z_te = N["redshift"][test_rows]
    p95 = float(np.nanpercentile(z_te, 95))
    targets_ok = {z: bool(z < p95) for z in Z_TARGETS}
    assert all(targets_ok.values()), f"a target z is above the test sample's 95th percentile specz ({p95:.3f})"
    low = test_rows[z_te < Z_MAX_SOURCE]
    big = low[N["size"][low] > np.nanmedian(N["size"][low])]
    sample = big[N["redshift"][big] >= Z_MIN_SOURCE]  # floor applied after the brief's size-median cut
    log(f"b2 sample: test {len(test_rows)}, specz < {Z_MAX_SOURCE}: {len(low)}, petroRad_r above that median: {len(big)}, "
        f"of which specz ≥ {Z_MIN_SOURCE}: {len(sample)}; specz p95 {p95:.3f}")
    # targets: median PSF and per-band sky noise of real test galaxies at each z
    rng = np.random.default_rng(SEED)
    tgt = {}
    for z in Z_TARGETS:
        near = test_rows[np.abs(z_te - z) < Z_WIN]
        pick = rng.choice(near, min(400, len(near)), replace=False)
        sig = np.array([sky(st.to_linear(st.cache.get(int(d.union[i]))))[1] for i in pick])
        tgt[z] = {"n_real": int(len(near)), "psf_fwhm_arcsec": float(np.nanmedian(N["psf"][near])),
                  "sky_sigma_linear": np.median(sig, axis=0).tolist(), "n_for_sky": int(len(pick))}
    log("b2 targets", json.dumps(tgt))
    # inverse check on the sample's first stamps
    inv_err = max(st.check(st.cache.get(int(d.union[i]))) for i in sample[:20])
    psf_ok = psf_check(cosmo)
    log("b2 psf check", json.dumps(psf_ok))
    dev = DF.device()
    enc = DF.a1_encoder(dev)
    # build and embed: own (re-embedded original), planted, each target
    runs = ["own", "planted"] + [f"z={z:.2f}" for z in Z_TARGETS]
    E = {k: np.zeros((len(sample), 384), np.float32) for k in runs}
    meta = {k: [] for k in runs[2:]}
    B = 64
    for s0 in range(0, len(sample), B):
        rows = sample[s0:s0 + B]
        ys = {k: [] for k in runs}
        for i in rows:
            y = st.cache.get(int(d.union[i])).astype(np.float32)
            x = st.to_linear(y)
            ys["own"].append(y)
            ys["planted"].append(st.to_model(redshift_stamp(x, 0, 0, 0, 0, None, cosmo, rng, planted=True)[0]))
            z0, p0 = N["redshift"][i], N["psf"][i]
            for z in Z_TARGETS:
                t = tgt[z]
                xs, info = redshift_stamp(x, z0, z, p0, t["psf_fwhm_arcsec"], t["sky_sigma_linear"], cosmo, rng)
                ys[f"z={z:.2f}"].append(st.to_model(xs))
                meta[f"z={z:.2f}"].append(info)
        for k in runs:
            E[k][s0:s0 + len(rows)] = DF.embed(enc, torch.from_numpy(np.stack(ys[k]).astype(np.float32)), dev)
        if s0 // B % 10 == 0:
            log(f"b2 embedded {s0 + len(rows)}/{len(sample)}")
    np.savez(OUT / "b2_embeddings.npz", rows=sample, **{k.replace("=", "_"): v for k, v in E.items()})
    bank = d.mats["A1"][sample]
    cos = lambda a, b: np.sum(a * b, 1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)  # noqa: E731
    checks = {"inverse_stretch_max_abs_err": inv_err, "psf_point_source": psf_ok,
              "cos_own_vs_bank": {"median": float(np.median(cos(E["own"], bank))), "min": float(np.min(cos(E["own"], bank)))},
              "cos_planted_vs_own": {"median": float(np.median(cos(E["planted"], E["own"]))),
                                     "min": float(np.min(cos(E["planted"], E["own"])))}}
    log("b2 checks", json.dumps(checks))
    # simulated nuisances: size × f, magnitude − 2.5 log10 g, psf and z of the target; snr from real galaxies near (z, mag)
    def nuis_at(z, f_, g_):
        size = N["size"][sample] * f_
        mag = N["magnitude"][sample] - 2.5 * np.log10(g_)
        snr = np.empty(len(sample))
        for k, mg in enumerate(mag):
            for w in (0.25, 0.5, 1.0):
                m = (np.abs(N["redshift"][test_rows] - z) < 0.01) & (np.abs(N["magnitude"][test_rows] - mg) < w)
                if m.sum() >= 20:
                    break
            snr[k] = np.nanmedian(N["snr"][test_rows][m])
        cols = {"size": size, "magnitude": mag, "snr": snr, "psf": np.full(len(sample), tgt[z]["psf_fwhm_arcsec"]),
                "redshift": np.full(len(sample), z)}
        return np.column_stack([cols[n] for n in S.NUIS])
    Nown = np.column_stack([N[n][sample] for n in S.NUIS])
    cl = d.cleaner["A1"]
    C = {"planted": cl(E["planted"].astype(np.float64), Nown)}
    Nsim = {}
    for z in Z_TARGETS:
        k = f"z={z:.2f}"
        fz = np.array([m_["f"] for m_ in meta[k]]); gz = np.array([m_["g"] for m_ in meta[k]])
        Nsim[k] = nuis_at(z, fz, gz)
        C[k] = cl(E[k].astype(np.float64), Nsim[k])
    # probe scores: cleaned (stored filters), raw (refit on raw train rows)
    arrs = {f: np.load(AX / f"{f}.npz") for f in order}
    per = {}
    for f in order:
        tr, te = d.rows(f)
        y_tr, _ = d.yv(f, tr)
        Pr = AXS.Probe(d.mats["A1"][tr], y_tr, d.c)
        sd_c = float(np.nanstd(d.clean["A1"][te] @ arrs[f]["we"]))
        sd_r = float(np.nanstd(Pr.d(d.mats["A1"][te])))
        e = {"label": out["labels"][f], "sd_test": {"cleaned": sd_c, "raw": sd_r}}
        base_c = C["planted"] @ arrs[f]["we"] + float(arrs[f]["be"])
        base_r = Pr.d(E["planted"].astype(np.float64))
        for z in Z_TARGETS:
            k = f"z={z:.2f}"
            for kind, dv in (("cleaned", C[k] @ arrs[f]["we"] + float(arrs[f]["be"]) - base_c),
                             ("raw", Pr.d(E[k].astype(np.float64)) - base_r)):
                bs = [np.mean(dv[rng.integers(0, len(dv), len(dv))]) for _ in range(N_BOOT)]
                e.setdefault(kind, {})[k] = {"mean_delta": float(np.mean(dv)), "ci95": ci(bs),
                                             "mean_delta_in_test_sd": float(np.mean(dv) / e["sd_test"][kind])}
        per[f] = e
    # movement along nuisance directions: OLS of the raw embedding on standardised nuisances (union train rows)
    trm = np.isin(d.union, np.asarray(d.setup.train_ids)) & d.ok
    Nt = np.column_stack([N[n][trm] for n in S.NUIS])
    mu_n, sd_n = Nt.mean(0), Nt.std(0)
    A = np.column_stack([np.ones(trm.sum()), (Nt - mu_n) / sd_n])
    Bc = np.linalg.lstsq(A, d.mats["A1"][trm], rcond=None)[0][1:]  # (5, 384)
    move = {}
    for z in Z_TARGETS:
        k = f"z={z:.2f}"
        dx = (E[k] - E["planted"]).astype(np.float64)
        implied = np.linalg.lstsq(Bc.T, dx.T, rcond=None)[0].T  # (n, 5) in nuisance SDs
        actual = (Nsim[k] - Nown) / sd_n
        move[k] = {n: {"implied_mean_sd": float(np.mean(implied[:, j])), "implied_ci95": ci(
            [np.mean(implied[rng.integers(0, len(dx), len(dx)), j]) for _ in range(N_BOOT)]),
            "actual_mean_sd": float(np.nanmean(actual[:, j])),
            "projection_mean": float(np.mean(dx @ (Bc[j] / np.linalg.norm(Bc[j]))))} for j, n in enumerate(S.NUIS)}
    # nearest neighbours of each simulated embedding among real union galaxies (cosine, raw), their specz
    R = d.mats["A1"][d.ok]; zR = N["redshift"][d.ok]; idR = np.flatnonzero(d.ok)
    Rn = R / np.linalg.norm(R, axis=1, keepdims=True)
    nn = {}
    for k in runs[1:]:
        Q_ = E[k].astype(np.float64); Q_ /= np.linalg.norm(Q_, axis=1, keepdims=True)
        med = np.empty(len(sample))
        for s0 in range(0, len(sample), 512):
            sim = Q_[s0:s0 + 512] @ Rn.T
            for r_, i in enumerate(range(s0, min(s0 + 512, len(sample)))):
                sim[r_, idR == sample[i]] = -2  # not itself
            top = np.argpartition(-sim, NN_K, axis=1)[:, :NN_K]
            med[s0:s0 + len(top)] = np.nanmedian(zR[top], axis=1)
        nn[k] = {"nn_median_specz": {"median": float(np.median(med)), "iqr": [float(np.percentile(med, 25)),
                                                                               float(np.percentile(med, 75))]}}
    nn["real_specz_of_sample"] = {"median": float(np.median(N["redshift"][sample]))}
    res = {"note": "exploratory; not in the science record. B2 resamples images as an intervention test, separate from "
                   "the training pipeline's no-rebin rule. Simplified FERENGI: angular size × D_A(z0)/D_A(z1), flux × "
                   "(D_L(z0)/D_L(z1))², Gaussian PSF up to the median r-band psfWidth of real test galaxies at z1 (applied "
                   "to all bands), Gaussian sky noise up to their median per-band sky σ (outer 20-px frame, MAD), "
                   "centred in a 256-px stamp whose outside is noise only; Planck18. The D_A shrink with the D_L² dimming gives the "
                   "(1+z)⁴ surface-brightness dimming exactly (D_L = (1+z)² D_A); only the k-correction is missing. The added "
                   "blur is σ_add² = σ_target² − (s·σ_orig)² − (anti-alias σ after the shrink)², s = D_A(z0)/D_A(z1). Score changes are "
                   "against the planted (no-change) run; cleaned scores use the simulated nuisances (size × f, magnitude − "
                   "2.5 log10 g, target PSF and z, snr = median of real test galaxies near (z, magnitude)). Declared deviation: "
                   "source specz ≥ 0.01, applied after the size-median cut (z ≤ 0 has no D_A; below 0.01 peculiar velocities "
                   "make D_A from redshift ≥ 10% off, and the shrink to z = 0.13 leaves < 23 px).",
           "sample": {"n_test": int(len(test_rows)), "n_specz_lt_0.05": int(len(low)), "n_above_size_median": int(len(big)),
                      "z_min_source": Z_MIN_SOURCE, "n": int(len(sample)),
                      "specz_p95_test": p95, "targets_below_p95": {str(k): v for k, v in targets_ok.items()}},
           "targets": {str(k): v for k, v in tgt.items()},
           "geometry": {k: {"f_median": float(np.median([m_["f"] for m_ in meta[k]])),
                            "g_median": float(np.median([m_["g"] for m_ in meta[k]])),
                            "psf_add_px_median": float(np.median([m_["psf_add_px"] for m_ in meta[k]]))} for k in meta},
           "checks": checks, "scores": per, "nuisance_movement": move, "nearest_neighbours": nn}
    (DOCS / "a1_visibility_b2.json").write_text(json.dumps(res, indent=1, default=float))
    sheet_b2(st, d, sample, tgt, cosmo, N)
    log("→ docs/almagest/a1_visibility_b2.json, a1_visibility_b2_examples.png")


def sheet_b2(st, d, sample, tgt, cosmo, N) -> None:
    """Eight galaxies: original | planted | z = 0.07 | 0.10 | 0.13, display stretch as the contact sheets."""
    from PIL import Image
    rng = np.random.default_rng(SEED + 7)
    pick = rng.choice(sample, 8, replace=False)
    T = 128
    im = Image.new("RGB", (5 * T, 8 * T), (250, 249, 246))
    for r_, i in enumerate(pick):
        y = st.cache.get(int(d.union[i])).astype(np.float32)
        x = st.to_linear(y)
        cols = [y, st.to_model(x)]
        for z in Z_TARGETS:
            t = tgt[z]
            cols.append(st.to_model(redshift_stamp(x, N["redshift"][i], z, N["psf"][i], t["psf_fwhm_arcsec"],
                                                   t["sky_sigma_linear"], cosmo, rng)[0]))
        for c, a in enumerate(cols):
            a = np.asarray(a, np.float32)
            rgb = np.stack([a[2], a[1], a[0]], -1)
            lo, hi = np.percentile(rgb, [1, 99.7])
            u8 = (np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)[64:192, 64:192] * 255).astype(np.uint8)
            im.paste(Image.fromarray(u8).resize((T, T), Image.LANCZOS), (c * T, r_ * T))
    im.save(DOCS / "a1_visibility_b2_examples.png", optimize=True)


def main() -> None:
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    {"b1": step_b1, "b1b": step_b1b, "b2": step_b2}.get(step, lambda: (_ for _ in ()).throw(SystemExit(__doc__)))()


if __name__ == "__main__":
    main()
