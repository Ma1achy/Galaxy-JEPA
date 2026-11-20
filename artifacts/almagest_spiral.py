"""Almagest geometry — A1's spiral axis: linearity re-check, nuisance cleaning, a straightened coordinate.

DESCRIPTIVE, for the Plate (user brief, 20 Nov 2026, branch `almagest-geometry`): no claim, nothing in the
science record, no rung change. Answer `t04_spiral_a08_spiral`, full population (vote_count_min 1), on the
ladder's own R footing: A1's real embeddings and the three banked untrained seeds over P2's union (39,997
train + 34,828 test, identical split), the ladder's penultimate-block mean-pooled 384-d vector. Everything
is fitted on train and measured on test.

Reads (never writes) `runs/ladder_v2/a1/` through symlinks in `runs/almagest/a1/`; writes only
`runs/almagest/a1/` (large arrays) and `docs/almagest/` (JSON, figures).

    .venv/bin/python artifacts/almagest_spiral.py r1      # step 1: linear probe vs MLP, R1's controls
    .venv/bin/python artifacts/almagest_spiral.py clean   # step 2: ridge-cleaned embeddings + checks
    .venv/bin/python artifacts/almagest_spiral.py curve   # step 3: γ(d) along the cleaned probe score, s, checks
    .venv/bin/python artifacts/almagest_spiral.py plate   # step 4: Plate format v3 (additive) in runs/almagest/a1
    .venv/bin/python artifacts/almagest_spiral.py figs    # step 5: path, sideways and strip figures
    ALM_BASIS=log … clean | curve                         # the log-value sensitivity reading

Investigation code: terse, excluded from lint/CI.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ALM = REPO / "runs/almagest/a1"
DOCS = REPO / "docs/almagest"
LADDER_A1 = REPO / "runs/ladder_v2/a1"
LINKS = ("o1_embeddings.npz", "r_untrained_seeds.npz", "r1_clusters.npz")
# A1's ladder environment (docs/ladder_v2/manifest.yaml common_env + A1's entry), output redirected here
os.environ.update({
    "GJ_CONFIG": "configs/pretrain_v2.yaml",
    "F0_CACHE_BASE": "/Volumes/X10 Pro/galaxy-jepa/runs/m_v2/cache",
    "F0_NORM_PREFIX": "246d8de6ba9a",
    "LADDER_SPLIT": "identical",
    "LADDER_ENCODER": "runs/m_v2/encoder.pt",
    "LADDER_OUT": str(ALM),
    "LADDER_TAG": "a1",
    "LADDER_UNTRAINED_BANK": "runs/ladder_v2/shared/p1_untrained_bank.json",
    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4"),
})
ALM.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)
for name in LINKS:  # read-only views of the ladder's banks; nothing is written into runs/ladder_v2
    link = ALM / name
    if not link.exists():
        link.symlink_to(LADDER_A1 / name)

import numpy as np  # noqa: E402

sys.path.insert(0, str(REPO / "artifacts"))
import r_nonlinear as R  # noqa: E402
from galaxy_jepa.models.vit import load_frozen_encoder  # noqa: E402
from galaxy_jepa.probing import controls as ctl  # noqa: E402
from galaxy_jepa.probing import geometry as geo  # noqa: E402
from galaxy_jepa.probing import mlp as mlp_mod  # noqa: E402
from galaxy_jepa.probing.extract import feature_ids  # noqa: E402
from galaxy_jepa.probing.logistic import Embeddings, _fit, paired_auc_bootstrap, probe_scores  # noqa: E402

F = "t04_spiral_a08_spiral"
NUIS = ("size", "magnitude", "snr", "psf", "redshift")  # petroRad_r, modelMag_r, snr_r, psfWidth_r, specz
RIDGE_ALPHA = 1.0
EDGES = R.R2_EDGES  # 10 equal-width vote-fraction bins
MIN_BIN = geo.MIN_OCCUPANCY  # 100
VAR_KEEP = 0.99
N_DENSE = 1_000


def sha1(p: Path) -> str:
    import hashlib
    h = hashlib.sha1()
    with open(Path(p).resolve(), "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def save(name: str, obj: dict) -> None:
    (DOCS / name).write_text(json.dumps(obj, indent=1, default=float))
    print(f"→ docs/almagest/{name}", file=sys.stderr)


class Data:
    """The ladder's R context for A1, the spiral's eligible train/test rows, nuisances and labels."""

    def __init__(self) -> None:
        self.setup = R.prepare("runs/m_v2/encoder.pt", R.MAX_TRAIN, label="ALM", sources=1)
        self.ctx = R.Ctx(self.setup, load_frozen_encoder(self.setup.ckpt), 0, dry=True, only=[F])
        self.lab = self.ctx.labels["full"]
        real = self.ctx.real
        self.ids_tr = np.array(feature_ids(real, self.lab, F, self.setup.train_ids))
        self.ids_te = np.array(feature_ids(real, self.lab, F, self.setup.test_ids))
        self.union = np.asarray(real.object_ids)
        self.pc = self.setup.pc

    def rows(self, m, ids):
        return m.x[m.rows_for(ids)].astype(np.float64)

    def y(self, ids):
        v = self.lab.vote_fraction(F, ids)
        return (v >= 0.5).astype(int), v

    def nuisances(self, ids) -> np.ndarray:
        return np.column_stack([self.lab.nuisance_value(n, ids) for n in NUIS])

    def reach(self, ids) -> np.ndarray:
        return sum(self.lab._column(ids, c) for c in self.lab.scheme.by_name[F].reach_count_cols())


def metadata(d: Data, ids, col: str) -> np.ndarray:
    """A probe-corpus metadata column (as plate_export reads expAB_r), NaN where absent or ≤ 0 for b/a."""
    import pandas as pd
    md = pd.read_csv(REPO / d.setup.cfg.paths.probe_dir / "metadata.csv", usecols=["object_id", col]).set_index("object_id")
    v = md[col].reindex(np.asarray(ids, dtype=np.int64)).to_numpy(dtype=np.float64)
    return np.where(v > 0, v, np.nan) if col == "expAB_r" else v


def provenance(d: Data) -> dict:
    return {"answer": F, "population": "full", "encoder": "A1", "checkpoint": "runs/m_v2/encoder.pt",
            "checkpoint_sha1": sha1(REPO / "runs/m_v2/encoder.pt"),
            "o1_bank_sha1": sha1(ALM / "o1_embeddings.npz"),
            "c2_bank_sha1": sha1(LADDER_A1 / "c2_a1_embeddings.npz"),
            "seed_bank_sha1": sha1(ALM / "r_untrained_seeds.npz"),
            "n_train": int(d.ids_tr.size), "n_test": int(d.ids_te.size), "max_train": R.MAX_TRAIN}


# --- step 1 -----------------------------------------------------------------------------------------
def step_r1(d: Data) -> dict:
    """R1 for spiral only, as `r_nonlinear.stage_r1` builds it (full population)."""
    pc, kw = d.pc, R._mlp_kw(d.pc)
    real = d.ctx.real
    tr = Embeddings(d.rows(real, d.ids_tr), *d.y(d.ids_tr))
    te = Embeddings(d.rows(real, d.ids_te), *d.y(d.ids_te))
    lin = probe_scores(tr, te, c=pc.c)
    width, val = mlp_mod.select_width(tr, widths=tuple(pc.mlp_widths), **kw)
    mlp = mlp_mod.mlp_fit(tr, te, width=width, **kw)
    head = paired_auc_bootstrap([te.y, te.y], [mlp.test_scores, lin], [1.0, -1.0], n_boot=R.N_BOOT, seed=pc.seed)
    rng = np.random.default_rng(pc.seed)
    perm = mlp_mod.mlp_fit(Embeddings(tr.x, rng.permutation(tr.y), tr.fraction), te, width=width, **kw)
    blob = np.load(ALM / "r1_clusters.npz", allow_pickle=False)
    assert np.array_equal(blob["ids"], real.object_ids), "cluster bank not co-indexed"
    tr_rows, te_rows = real.rows_for(d.ids_tr), real.rows_for(d.ids_te)
    sel, ctrl = {}, {}
    for c in R.CLUSTER_COUNTS:
        yc = ctl.cluster_control_labels(blob[f"c{c}"][np.concatenate([tr_rows, te_rows])],
                                        base_rate=float(tr.y.mean()), seed=pc.seed + c)
        yc_tr, yc_te = yc[: tr_rows.size], yc[tr_rows.size:]
        ctr, cte = Embeddings(tr.x, yc_tr, tr.fraction), Embeddings(te.x, yc_te, te.fraction)
        lin_c = probe_scores(ctr, cte, c=pc.c)
        mlp_c = mlp_mod.mlp_fit(ctr, cte, width=width, **kw)
        v = paired_auc_bootstrap([te.y, te.y, yc_te, yc_te], [mlp.test_scores, lin, mlp_c.test_scores, lin_c],
                                 [1.0, -1.0, -1.0, 1.0], n_boot=R.N_BOOT, seed=pc.seed)
        sel[str(c)] = {"point": v[0], "lo": v[1], "hi": v[2]}
        ctrl[str(c)] = {"mlp": auc(yc_te, mlp_c.test_scores), "linear": auc(yc_te, lin_c)}
    verdict = R._three_state(head[1], {int(k): (v["point"], v["lo"], v["hi"]) for k, v in sel.items()})
    ladder = json.loads((LADDER_A1 / "r_r1.json").read_text())[f"full:{F}"]
    out = {"width": width, "val_auc": val, "linear_auc": auc(te.y, lin), "mlp_auc": auc(te.y, mlp.test_scores),
           "headroom": head[0], "headroom_ci95": [head[1], head[2]], "perm_test_auc": auc(te.y, perm.test_scores),
           "perm_train_auc": perm.train_auc, "real_train_auc": mlp.train_auc, "cluster_selective": sel,
           "cluster_control_aucs": ctrl, "verdict_as_R1": verdict,
           "mlp_beats_linear_beyond_ci": bool(head[1] > 0),
           "ladder_record_r_r1": {k: ladder[k] for k in ("width", "linear_auc", "mlp_auc", "headroom",
                                                           "headroom_lo", "headroom_hi", "verdict")}}
    return out


# --- step 2 -----------------------------------------------------------------------------------------
# Primary basis (20 columns): rank − 0.5, its square, the normal score, its square. The brief's basis was the
# first two alone; a bounded S-curve of the rank cannot follow an effect that grows into a nuisance's tails, and
# the planted check caught it (docs/almagest/a1_spiral_report.md, "Deviations"). ALM_BASIS=log is the reported
# sensitivity: the same 20 plus each nuisance's log value, all columns standardised on train.
BASIS = os.environ.get("ALM_BASIS", "primary")
if BASIS not in ("primary", "log"):
    raise SystemExit(f"ALM_BASIS={BASIS!r}: primary | log")
SUFFIX = "" if BASIS == "primary" else "_logsens"
LOG_TF = {"size": np.log, "magnitude": lambda v: v, "snr": np.log, "psf": np.log,
          "redshift": np.log1p}  # magnitude is already logarithmic; log(1+z): 14 union specz are below zero


def rank_fit(train_col: np.ndarray):
    srt = np.sort(train_col)
    n = srt.size  # train empirical CDF, held inside [0.5/n, 1 − 0.5/n] so the normal score stays finite
    return lambda v: np.clip(np.searchsorted(srt, v, side="right") / n, 0.5 / n, 1 - 0.5 / n)


def covariates(N: np.ndarray, fits) -> np.ndarray:
    from scipy.stats import norm
    r = np.column_stack([f(N[:, j]) for j, f in enumerate(fits)])
    q, z = r - 0.5, norm.ppf(r)
    return np.column_stack([q, q ** 2, z, z ** 2])


class Cleaner:
    """Ridge of the 384-d embedding on rank-transformed nuisances and their squares (train); subtract."""

    def __init__(self, X_tr: np.ndarray, N_tr: np.ndarray, alpha: float = RIDGE_ALPHA, extra_tr=None) -> None:
        from sklearn.linear_model import Ridge
        cols = N_tr if extra_tr is None else np.column_stack([N_tr, extra_tr])
        self.fits = [rank_fit(cols[:, j]) for j in range(cols.shape[1])]
        # log basis: an extra (planted) column gets a log value only where one exists (all positive)
        self.tf = [LOG_TF[n] for n in NUIS] + [np.log if (cols[:, j] > 0).all() else None
                                               for j in range(len(NUIS), cols.shape[1])]
        self.mu = self.sd = None
        self.model = Ridge(alpha=alpha).fit(self._cov(cols, fit=True), X_tr)

    def _cov(self, cols, fit=False):
        C = covariates(cols, self.fits)
        if BASIS == "log":
            C = np.column_stack([C] + [f(cols[:, j]) for j, f in enumerate(self.tf) if f is not None])
        if fit:  # centred on train in both bases, so subtracting C·coef keeps the embedding's train mean
            self.mu, self.sd = C.mean(0), (C.std(0) if BASIS == "log" else np.ones(C.shape[1]))
        return (C - self.mu) / self.sd

    def __call__(self, X: np.ndarray, N: np.ndarray, extra=None) -> np.ndarray:
        cols = N if extra is None else np.column_stack([N, extra])
        return X - (self.model.predict(self._cov(cols)) - self.model.intercept_)  # keep the embedding's own mean


def decode_r2(X_tr, t_tr, X_te, t_te) -> float:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X_tr)
    m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(X_tr), t_tr)
    pred = m.predict(sc.transform(X_te))
    return float(1 - np.sum((t_te - pred) ** 2) / np.sum((t_te - t_te.mean()) ** 2))


def probe_dir(X_tr, y_tr, c) -> np.ndarray:
    scaler, clf = _fit(Embeddings(X_tr, y_tr, np.zeros(len(y_tr))), c=c)
    w = clf.coef_[0] / scaler.scale_  # direction in raw embedding units
    return w / np.linalg.norm(w)


def complete(d: Data, ids) -> np.ndarray:
    N = d.nuisances(ids)
    return np.isfinite(N).all(1)


def fit_cleaner(d: Data, m):
    """Fit on all union TRAIN rows with complete nuisances."""
    N = d.nuisances(d.union)
    ok = np.isfinite(N).all(1)
    is_tr = np.isin(d.union, np.asarray(d.setup.train_ids))
    X = m.x.astype(np.float64)
    return Cleaner(X[ok & is_tr], N[ok & is_tr]), X, N, ok


def clean_matrix(d: Data, m):
    """The fitted cleaner applied to every union row with complete nuisances (NaN elsewhere)."""
    cl, X, N, ok = fit_cleaner(d, m)
    Xc = np.full_like(X, np.nan)
    Xc[ok] = cl(X[ok], N[ok])
    return Xc, ok, N


def step_clean(d: Data) -> dict:
    real = d.ctx.real
    Xc, ok, N = clean_matrix(d, real)
    np.save(ALM / f"a1_cleaned{SUFFIX}.npy", Xc.astype(np.float32))
    np.save(ALM / "a1_union_ids.npy", d.union)
    row = {int(o): i for i, o in enumerate(d.union)}
    tr = np.array([row[int(i)] for i in d.ids_tr]); te = np.array([row[int(i)] for i in d.ids_te])
    tr_ok, te_ok = tr[ok[tr]], te[ok[te]]
    y_tr, _ = d.y(d.union[tr_ok]); y_te, _ = d.y(d.union[te_ok])
    X = real.x.astype(np.float64)
    out: dict = {"nuisance_columns": dict(zip(NUIS, ("petroRad_r", "modelMag_r", "snr_r", "psfWidth_r", "specz"))),
                 "basis": BASIS,
                 "covariates": "per nuisance: train-ECDF rank − 0.5, its square, the normal score Φ⁻¹(rank), its square "
                               "(20 columns)" + ("; plus each nuisance's log value (size, snr, psf: log; redshift: "
                               "log(1+z); magnitude as is), all 25 standardised on train" if BASIS == "log" else ""),
                 "ridge_alpha": RIDGE_ALPHA, "fit_rows": "union train rows with all five nuisances",
                 "dropped_missing": {"union": int((~ok).sum()), "spiral_train": int((~ok[tr]).sum()),
                                     "spiral_test": int((~ok[te]).sum()),
                                     "by_nuisance_union": {n: int((~np.isfinite(N[:, j])).sum()) for j, n in enumerate(NUIS)}},
                 "size_flagged_kept_union": int((~d.lab.nuisance_valid("size", list(d.union))).sum()),
                 "n_train": int(tr_ok.size), "n_test": int(te_ok.size)}
    # a. spiral AUC raw vs cleaned (same rows)
    c = d.pc.c
    s_raw = probe_scores(Embeddings(X[tr_ok], y_tr, np.zeros(len(y_tr))), Embeddings(X[te_ok], y_te, np.zeros(len(y_te))), c=c)
    s_cl = probe_scores(Embeddings(Xc[tr_ok], y_tr, np.zeros(len(y_tr))), Embeddings(Xc[te_ok], y_te, np.zeros(len(y_te))), c=c)
    out["auc_raw"], out["auc_cleaned"] = auc(y_te, s_raw), auc(y_te, s_cl)
    # b. cosine of the probe directions
    out["cosine_probe_dirs"] = float(probe_dir(X[tr_ok], y_tr, c) @ probe_dir(Xc[tr_ok], y_tr, c))
    # c. nuisance decodability, raw vs cleaned (test R², target = rank and raw value)
    trU = np.flatnonzero(ok & np.isin(d.union, d.setup.train_ids)); teU = np.flatnonzero(ok & np.isin(d.union, d.setup.test_ids))
    fits = [rank_fit(N[trU, j]) for j in range(len(NUIS))]
    dec = {}
    for j, n in enumerate(NUIS):
        tr_t, te_t = fits[j](N[trU, j]), fits[j](N[teU, j])
        dec[n] = {"r2_raw_rank": decode_r2(X[trU], tr_t, X[teU], te_t),
                  "r2_cleaned_rank": decode_r2(Xc[trU], tr_t, Xc[teU], te_t),
                  "r2_cleaned_value": decode_r2(Xc[trU], N[trU, j], Xc[teU], N[teU, j])}
    out["decode"] = dec
    out["decode_over_0.05"] = [n for n, v in dec.items() if max(v["r2_cleaned_rank"], v["r2_cleaned_value"]) > 0.05]
    # d. planted nuisance: a known direction scaled by a synthetic covariate, then cleaned with it. The gate is the
    # Gaussian plant; the lognormal one (heavy right tail, like size) is a stress reading beside it, not a gate.
    rng = np.random.default_rng(20261120)
    u = rng.normal(size=X.shape[1]); u /= np.linalg.norm(u)
    g = rng.normal(size=len(d.union))
    amp = 3.0 * float(np.linalg.norm(X.std(0)))
    out["planted"] = {"amplitude": amp, "direction_seed": 20261120,
                      "injection": "X + amp · (t − mean_train)/sd_train · u; t cleaned as an extra nuisance column"}
    for name, t in (("gaussian", g), ("lognormal", np.exp(g))):
        Xp = X + amp * np.outer((t - t[trU].mean()) / t[trU].std(), u)
        cl_p = Cleaner(Xp[trU], N[trU], extra_tr=t[trU, None])
        Xpc, tr_pc = cl_p(Xp[teU], N[teU], t[teU, None]), cl_p(Xp[trU], N[trU], t[trU, None])
        out["planted"][name] = {"corr_t_with_u_before": float(np.corrcoef(Xp[teU] @ u, t[teU])[0, 1]),
                                "corr_t_with_u_after": float(np.corrcoef(Xpc @ u, t[teU])[0, 1]),
                                "r2_decode_t_before": decode_r2(Xp[trU], t[trU], Xp[teU], t[teU]),
                                "r2_decode_t_after": decode_r2(tr_pc, t[trU], Xpc, t[teU])}
    gp = out["planted"]["gaussian"]
    out["planted"]["removed"] = bool(abs(gp["corr_t_with_u_after"]) < 0.02 and gp["r2_decode_t_after"] < 0.01)
    out["planted"]["gate"] = "gaussian: |corr after| < 0.02 and decode R² after < 0.01; lognormal reported, not gated"
    return out


# --- step 3 -----------------------------------------------------------------------------------------
# γ(d): the spiral axis as the cleaned linear probe's score d, with the mean embedding's path along it.
# The first design read s off the nearest point of a vote-fraction curve; within-bin spread (3–4 SD per PC)
# dwarfs the path, so that coordinate was noise (AUC 0.672 against 0.746; kept in a1_spiral_curve_nearest*).
# Here s is a function of d alone, so it orders galaxies exactly as the probe does.
N_DBINS = 20
D_GRID = 4_000  # γ sampled on this many d values across the train range of d
S_SAMPLES = 1_001  # γ sampled at equal arc length, s = 0, 0.001, …, 1
TAIL_S = 0.8
TAIL_RATIO = 2.0  # set before the run: the tail is flagged if its disagreement is over twice the overall


def bin_means(Z, v):
    b = geo._bin_of(v, EDGES)
    keep = [k for k in range(len(EDGES) - 1) if (b == k).sum() >= MIN_BIN]
    M = np.array([Z[b == k].mean(0) for k in keep])
    return M, keep, np.array([(b == k).sum() for k in keep])


def top_k(M):
    centre = M.mean(0)
    _, S, Vt = np.linalg.svd(M - centre, full_matrices=False)
    var = S ** 2 / np.sum(S ** 2)
    k = int(np.searchsorted(np.cumsum(var), VAR_KEEP) + 1)
    return centre, Vt[:k], var


class Curve:
    """The vote-fraction path (10 equal-width bins), by arc length: kept only for the comparison with γ."""

    def __init__(self, M: np.ndarray) -> None:
        from scipy.interpolate import make_smoothing_spline
        self.centre, self.P, self.var = top_k(M)
        self.k = self.P.shape[0]
        Y = (M - self.centre) @ self.P.T
        seg = np.linalg.norm(np.diff(Y, axis=0), axis=1)
        t = np.concatenate([[0], np.cumsum(seg)]); t /= t[-1]
        spl = [make_smoothing_spline(t, Y[:, j]) for j in range(self.k)]
        tt = np.linspace(0, 1, 20 * S_SAMPLES)
        pts = np.column_stack([sp(tt) for sp in spl])
        arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        self.length = float(arc[-1])
        t_at = np.interp(np.linspace(0, 1, S_SAMPLES) * arc[-1], arc, tt)
        self.G = self.centre + np.column_stack([sp(t_at) for sp in spl]) @ self.P  # S_SAMPLES × D, equal arc


class Gamma:
    """γ(d): mean z in N_DBINS equal-count bins of d (train), a smoothing cubic spline through the bin means
    in their top-k PCs with the bin-mean d as parameter; beyond the end bin means, straight on along the
    secant of the last two (the extrapolation rule). s(d) = arc length from the first bin mean's d, over the
    arc length to the last's: s = 0 and 1 are γ's last data-supported points, and the straight continuation
    carries s below 0 and above 1. (The train range of d was the first spec: 71% of γ's length then lay
    in the extrapolated ends, out to a few outliers.)"""

    def __init__(self, Z: np.ndarray, dsc: np.ndarray) -> None:
        from scipy.interpolate import make_smoothing_spline
        q = np.quantile(dsc, np.linspace(0, 1, N_DBINS + 1))
        b = np.clip(np.searchsorted(q, dsc, side="right") - 1, 0, N_DBINS - 1)
        self.counts = np.bincount(b, minlength=N_DBINS)
        assert (self.counts > 0).all()
        self.d_ranges = np.array([[dsc[b == k].min(), dsc[b == k].max()] for k in range(N_DBINS)])
        M = np.array([Z[b == k].mean(0) for k in range(N_DBINS)])
        self.dbar = np.array([dsc[b == k].mean() for k in range(N_DBINS)])
        self.centre, self.P, self.var = top_k(M)
        self.k = self.P.shape[0]
        self.Y = (M - self.centre) @ self.P.T
        self.spl = [make_smoothing_spline(self.dbar, self.Y[:, j]) for j in range(self.k)]
        self.g0 = (self.Y[1] - self.Y[0]) / (self.dbar[1] - self.dbar[0])  # k-D change per unit d, ends
        self.g1 = (self.Y[-1] - self.Y[-2]) / (self.dbar[-1] - self.dbar[-2])
        self.lo, self.hi = float(self.dbar[0]), float(self.dbar[-1])
        self.train_d_range = (float(dsc.min()), float(dsc.max()))
        self.grid = np.linspace(self.lo, self.hi, D_GRID)
        pts = self.at(self.grid)
        step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        assert (step > 0).all(), "γ stalls: s would not be strictly monotone in d"
        self.arc = np.concatenate([[0], np.cumsum(step)])
        self.length = float(self.arc[-1])
        self.speed0, self.speed1 = float(np.linalg.norm(self.g0)), float(np.linalg.norm(self.g1))

    def at(self, dv) -> np.ndarray:
        """γ in the k-D PC coordinates."""
        dv = np.asarray(dv, dtype=np.float64)
        a, z = self.dbar[0], self.dbar[-1]
        out = np.column_stack([sp(np.clip(dv, a, z)) for sp in self.spl])
        lo, hi = dv < a, dv > z
        if lo.any():
            out[lo] = np.array([sp(a) for sp in self.spl]) + (dv[lo] - a)[:, None] * self.g0
        if hi.any():
            out[hi] = np.array([sp(z) for sp in self.spl]) + (dv[hi] - z)[:, None] * self.g1
        return out

    def full(self, dv) -> np.ndarray:
        return self.centre + self.at(dv) @ self.P

    def s(self, dv) -> np.ndarray:
        dv = np.asarray(dv, dtype=np.float64)
        a = np.interp(dv, self.grid, self.arc)
        a = np.where(dv < self.lo, (dv - self.lo) * self.speed0, a)  # γ is the straight secant out there
        a = np.where(dv > self.hi, self.length + (dv - self.hi) * self.speed1, a)
        return a / self.length

    def d_of_s(self, sv) -> np.ndarray:  # inside [0, 1]
        return np.interp(np.asarray(sv) * self.length, self.arc, self.grid)

    def sampled(self, n: int = S_SAMPLES) -> np.ndarray:
        return self.full(self.d_of_s(np.linspace(0, 1, n)))

    def describe(self) -> dict:
        G = self.sampled()
        chord = float(np.linalg.norm(G[-1] - G[0]))
        return {"n_bins": N_DBINS, "counts": self.counts.tolist(), "d_bin_ranges": self.d_ranges.round(4).tolist(),
                "end_bins": {"first": {"count": int(self.counts[0]), "d_range": self.d_ranges[0].tolist()},
                             "last": {"count": int(self.counts[-1]), "d_range": self.d_ranges[-1].tolist()}},
                "s_span_d": [self.lo, self.hi], "d_train_range": list(self.train_d_range), "bin_mean_d": self.dbar.tolist(),
                "k": self.k, "var_explained": self.var[: self.k].tolist(), "off_pc1_share": float(1 - self.var[0]),
                "arc_length": self.length, "chord": chord, "bend_1_minus_chord_over_arc": 1 - chord / self.length}


def hausdorff(A, B) -> float:
    from scipy.spatial.distance import directed_hausdorff
    return max(directed_hausdorff(A, B)[0], directed_hausdorff(B, A)[0])


def bend(G: np.ndarray, w: np.ndarray, sw: np.ndarray, L: float) -> dict:
    """G sampled at equal arc length. Bend = the displacement from γ(0) orthogonal to γ's own chord, in
    path lengths. A straight mean path E[z|d] runs along Σw (the pattern), not w (the filter): so the chord
    is read against both, and 'sideways' is measured from the Σw line (displacement orthogonal to Σw)."""
    disp = G - G[0]
    ch = (G[-1] - G[0]) / np.linalg.norm(G[-1] - G[0])
    wh, sh = w / np.linalg.norm(w), sw / np.linalg.norm(sw)
    dev = disp - np.outer(disp @ ch, ch)
    side = disp - np.outer(disp @ sh, sh)
    dn, sn = np.linalg.norm(dev, axis=1) / L, np.linalg.norm(side, axis=1) / L
    step = (len(dn) - 1) // 100
    return {"chord_dev_max": float(dn.max()), "chord_dev_at_max_s": float(np.argmax(dn) / (len(dn) - 1)),
            "bend_1_minus_chord_over_arc": float(1 - np.linalg.norm(G[-1] - G[0]) / L),
            "cos_chord_sigma_w": float(ch @ sh), "cos_chord_w": float(ch @ wh), "cos_w_sigma_w": float(wh @ sh),
            "off_sigma_w_end": float(sn[-1]), "off_sigma_w_max": float(sn.max()),
            "_dev": dev, "_dev_profile": dn[::step].tolist(), "_side_profile": sn[::step].tolist()}


class Fit:
    """Clean → probe (train) → z in the probe's own standardised space → d = w·z + b → γ(d)."""

    def __init__(self, d: Data, m, cleaned: bool = True) -> None:
        cl, X, N, ok = fit_cleaner(d, m)
        self.cl = cl
        Xc = np.full_like(X, np.nan)
        Xc[ok] = cl(X[ok], N[ok]) if cleaned else X[ok]
        row = {int(o): i for i, o in enumerate(d.union)}
        tr = np.array([row[int(i)] for i in d.ids_tr]); te = np.array([row[int(i)] for i in d.ids_te])
        self.tr, self.te = tr[ok[tr]], te[ok[te]]
        self.ids_tr, self.ids_te = d.union[self.tr], d.union[self.te]
        self.y_tr, self.v_tr = d.y(self.ids_tr); self.y_te, self.v_te = d.y(self.ids_te)
        self.scaler, clf = _fit(Embeddings(Xc[self.tr], self.y_tr, np.zeros(len(self.y_tr))), c=d.pc.c)
        self.w, self.b = clf.coef_[0].astype(np.float64), float(clf.intercept_[0])
        self.Z_tr, self.Z_te = self.scaler.transform(Xc[self.tr]), self.scaler.transform(Xc[self.te])
        self.d_tr, self.d_te = self.Z_tr @ self.w + self.b, self.Z_te @ self.w + self.b
        assert np.allclose(self.d_te, clf.decision_function(self.Z_te))
        self.g = Gamma(self.Z_tr, self.d_tr)
        self.s_te = self.g.s(self.d_te)

    def z(self, X_raw, N):
        """Any galaxy's z (cleaned, then the probe's standardisation)."""
        return self.scaler.transform(self.cl(X_raw, N))


def checks_monotone(f: Fit) -> dict:
    o = np.argsort(f.d_te, kind="stable")
    ds, ss = np.diff(f.d_te[o]), np.diff(f.s_te[o])
    assert (ss >= 0).all() and (ss[ds > 0] > 0).all(), "s is not strictly monotone in d"
    a_d, a_s = auc(f.y_te, f.d_te), auc(f.y_te, f.s_te)
    assert a_d == a_s, (a_d, a_s)
    return {"s_strictly_monotone_in_d": True, "auc_d": a_d, "auc_s": a_s, "auc_equal": True,
            "test_s_range": [float(f.s_te.min()), float(f.s_te.max())],
            "test_outside_0_1": int(((f.s_te < 0) | (f.s_te > 1)).sum())}


def compare(fa: Gamma, fb: Gamma, d_grid_owner: Gamma, L: float) -> dict:
    """Two γ fits in the same z space (same probe d), compared on the reference fit's d grid."""
    A, B = fa.full(d_grid_owner.grid), fb.full(d_grid_owner.grid)
    tail = d_grid_owner.s(d_grid_owner.grid) >= TAIL_S
    gap = np.linalg.norm(A - B, axis=1) / L  # at matched d: a mean, so the tail can exceed the whole (a max can hardly)
    return {"hausdorff_over_length": hausdorff(A, B) / L, f"hausdorff_over_length_s>={TAIL_S}": hausdorff(A[tail], B[tail]) / L,
            "mean_gap_over_length": float(gap.mean()), f"mean_gap_over_length_s>={TAIL_S}": float(gap[tail].mean())}


def gamma_record(d: Data, m, *, full: bool) -> tuple[dict, Fit]:
    f = Fit(d, m)
    g = f.g
    res = {**g.describe(), **checks_monotone(f)}
    f.sigma_w = np.cov(f.Z_tr.T) @ f.w
    G = g.sampled()
    sw = bend(G, f.w, f.sigma_w, g.length)
    res["bend"] = {k: v for k, v in sw.items() if k != "_dev"}
    if not full:
        return res, f
    # off-axis residual r = z − γ(d), test (descriptive)
    from scipy.stats import spearmanr
    rn = np.linalg.norm(f.Z_te - g.full(f.d_te), axis=1)
    cov = {n: d.lab.nuisance_value(n, f.ids_te) for n in NUIS}
    cov["question_reach"] = d.reach(f.ids_te)
    cov["expAB_r"] = metadata(d, f.ids_te, "expAB_r")
    res["off_axis"] = {"r": "z − γ(d) in the 384-d standardised cleaned space; |r| its norm",
                       "median_r_over_length": float(np.median(rn) / g.length),
                       "spearman_abs_r": {k: {"rho": float(spearmanr(rn[np.isfinite(v)], v[np.isfinite(v)])[0]),
                                              "n": int(np.isfinite(v).sum())} for k, v in cov.items()}}
    # cross-fits: the well-voted half (reach ≥ median), the other half, and a random-halves reference
    reach = d.reach(f.ids_tr); hi = reach >= np.nanmedian(reach)
    rng = np.random.default_rng(20261120); rnd = rng.permutation(len(hi)) < len(hi) // 2
    fits = {"well_voted": Gamma(f.Z_tr[hi], f.d_tr[hi]), "other_half": Gamma(f.Z_tr[~hi], f.d_tr[~hi]),
            "random_a": Gamma(f.Z_tr[rnd], f.d_tr[rnd]), "random_b": Gamma(f.Z_tr[~rnd], f.d_tr[~rnd])}
    cf = {"median_reach": float(np.nanmedian(reach)), "n": {"well_voted": int(hi.sum()), "other_half": int((~hi).sum()),
                                                           "random_a": int(rnd.sum()), "random_b": int((~rnd).sum())}}
    for name, h in fits.items():
        ds = np.abs(h.s(f.d_te) - f.s_te)
        cf[f"{name}_vs_full"] = {**compare(g, h, g, g.length), "k": h.k, "s_span_d": [h.lo, h.hi],
                                 "bend": h.describe()["bend_1_minus_chord_over_arc"],
                                 "abs_ds_median": float(np.median(ds)), "abs_ds_max": float(ds.max())}
    cf["well_voted_vs_other_half"] = compare(fits["well_voted"], fits["other_half"], g, g.length)
    cf["random_a_vs_random_b"] = compare(fits["random_a"], fits["random_b"], g, g.length)
    for key in ("well_voted_vs_other_half", "random_a_vs_random_b"):
        c = cf[key]
        c["tail_over_overall_hausdorff"] = c[f"hausdorff_over_length_s>={TAIL_S}"] / c["hausdorff_over_length"]
        c["tail_over_overall_mean_gap"] = c[f"mean_gap_over_length_s>={TAIL_S}"] / c["mean_gap_over_length"]
    cf["tail_flag"] = {"rule": f"tail mean gap at matched d > {TAIL_RATIO}× overall, any fit vs the full fit or the two "
                               "halves (Hausdorff is a maximum, so its tail value can hardly exceed the whole; reported only)",
                       "flagged": [k for k, c in cf.items() if isinstance(c, dict) and "mean_gap_over_length" in c
                                   and c[f"mean_gap_over_length_s>={TAIL_S}"] > TAIL_RATIO * c["mean_gap_over_length"]]}
    res["cross_fit"] = cf
    # the vote-fraction path (the first design's curve) against γ, same z space, both by their own arc length
    M, keep, occ = bin_means(f.Z_tr, f.v_tr)
    vf = Curve(M)
    sv = bend(vf.G, f.w, f.sigma_w, vf.length)
    a_dev, b_dev = sw["_dev"], sv["_dev"]
    cosf = lambda a, b: float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b)))  # noqa: E731
    Pg = g.P
    in_span = lambda D: float(np.linalg.norm(D @ Pg.T @ Pg) ** 2 / np.linalg.norm(D) ** 2)  # noqa: E731
    res["vs_vote_fraction_path"] = {
        "vote_path": {"n_bins": len(keep), "k": vf.k, "arc_length": vf.length,
                      "bend_1_minus_chord_over_arc": 1 - float(np.linalg.norm(vf.G[-1] - vf.G[0])) / vf.length,
                      "chord_dev_max": sv["chord_dev_max"], "cos_chord_sigma_w": sv["cos_chord_sigma_w"],
                      "cos_chord_w": sv["cos_chord_w"]},
        "cos_sideways_profiles": cosf(a_dev, b_dev),
        "cos_sideways_at_max": cosf(a_dev[int(np.argmax(np.linalg.norm(a_dev, axis=1)))],
                                    b_dev[int(np.argmax(np.linalg.norm(b_dev, axis=1)))]),
        "sideways": "each path's displacement orthogonal to its own chord, at 1,001 equal arc-length points of its own; "
                    "profiles = all flattened; at_max = the two displacement vectors where each is largest",
        "cos_chords": float((vf.G[-1] - vf.G[0]) @ (G[-1] - G[0]) / (np.linalg.norm(vf.G[-1] - vf.G[0]) * np.linalg.norm(G[-1] - G[0]))),
        "vote_path_displacement_in_gamma_pcs": in_span(vf.G - vf.G[0])}
    # extrapolation: unit end tangents (secant of the last two bin means), in arc-length units (× L)
    t0 = -g.g0 @ g.P / g.speed0; t1 = g.g1 @ g.P / g.speed1  # outward
    res["extrapolation"] = {"rule": "below s = 0 / above s = 1: γ(s) = γ(end) + (s − end)·L·t_end, t_end the outward unit "
                                    "secant of the last two bin means, L the arc length; s carries on at the same rate "
                                    "in d (ds/dd = |γ'|/L at that end). Defined, not rendered.",
                            "ds_dd_below": g.speed0 / g.length, "ds_dd_above": g.speed1 / g.length}
    return res, f, (t0, t1, sw, sv, vf)


def save_gamma(f: Fit, t0, t1, extra: dict) -> None:
    g = f.g
    np.savez(ALM / f"a1_spiral_gamma{SUFFIX}.npz", G=g.sampled(), grid=g.grid, arc=g.arc, length=g.length,
             lo=g.lo, hi=g.hi, dbar=g.dbar, centre=g.centre, P=g.P, Y=g.Y, g0=g.g0, g1=g.g1, t0=t0, t1=t1,
             w=f.w, b=f.b, std_mu=f.scaler.mean_, std_sd=f.scaler.scale_, test_ids=f.ids_te, d_test=f.d_te,
             s_test=f.s_te, y_test=f.y_te, v_test=f.v_te, **extra)


def step_curve(d: Data) -> dict:
    out = {"construction": {"d": "the cleaned linear probe's decision score (L2 logistic, train), d = w·z + b, z the "
                                 "cleaned embedding in the probe's own standardisation",
                            "gamma": f"mean z in {N_DBINS} equal-count bins of d (train); smoothing cubic spline "
                                     "(scipy make_smoothing_spline, GCV) through the bin means in their top-k PCs "
                                     f"(k reaching {VAR_KEEP:.0%}), parameter = the bin-mean d; straight on along "
                                     "the end secants beyond the end bin means",
                            "s": "arc length of γ from the train minimum of d, over the arc length to the train "
                                 "maximum; monotone in d by construction (asserted), so AUC(s) = AUC(d) (asserted)"}}
    t0 = time.perf_counter()
    rec, f, (ta, tb, sw, sv, vf) = gamma_record(d, d.ctx.real, full=True)
    out["a1"] = rec
    print(f"  γ A1 {time.perf_counter() - t0:.0f}s", file=sys.stderr)
    save_gamma(f, ta, tb, {"sigma_w": f.sigma_w, "vote_G": vf.G})
    if BASIS == "log":  # sensitivity: A1 only, against the primary γ in cleaned-embedding units
        out["vs_primary"] = compare_bases(ALM / "a1_spiral_gamma.npz", ALM / "a1_spiral_gamma_logsens.npz")
        return out
    out["untrained"] = {u.encoder_name: gamma_record(d, u, full=False)[0] for u in d.ctx.untrained}
    keys = ("bend_1_minus_chord_over_arc", "k", "auc_d", "off_pc1_share")
    rng = {k: [min(v[k] for v in out["untrained"].values()), max(v[k] for v in out["untrained"].values())] for k in keys}
    for k in ("chord_dev_max", "cos_chord_w", "cos_chord_sigma_w", "off_sigma_w_max"):
        vals = [v["bend"][k] for v in out["untrained"].values()]
        rng[k] = [min(vals), max(vals)]
    out["untrained_range"] = rng
    return out


def compare_bases(path_a: Path, path_b: Path) -> dict:
    """Two γ fits from different cleaning bases: in cleaned-embedding units, at equal s; s on test."""
    from scipy.stats import pearsonr, spearmanr
    a, b = np.load(path_a), np.load(path_b)
    assert np.array_equal(a["test_ids"], b["test_ids"])
    A, B = a["G"] * a["std_sd"] + a["std_mu"], b["G"] * b["std_sd"] + b["std_mu"]
    L = float(np.sum(np.linalg.norm(np.diff(A, axis=0), axis=1)))
    sg = np.linspace(0, 1, len(A)); tail = sg >= TAIL_S
    sa, sb = a["s_test"], b["s_test"]
    out = {"space": "cleaned 384-d embedding units, over the primary γ's length there", "primary_length": L}
    for name, m, mt in (("overall", np.ones(len(A), bool), np.ones(len(sa), bool)), (f"s>={TAIL_S}", tail, sa >= TAIL_S)):
        ds = np.abs(sa[mt] - sb[mt])
        out[name] = {"hausdorff_over_length": hausdorff(A[m], B[m]) / L, "n_test": int(mt.sum()),
                     "mean_gap_over_length": float((np.linalg.norm(A[m] - B[m], axis=1) / L).mean()),
                     "s_pearson": float(pearsonr(sa[mt], sb[mt])[0]), "s_spearman": float(spearmanr(sa[mt], sb[mt])[0]),
                     "abs_ds_median": float(np.median(ds)), "abs_ds_p90": float(np.percentile(ds, 90))}
    o, t = out["overall"], out[f"s>={TAIL_S}"]
    out["tail_flag"] = {"rule": f"tail mean gap (matched s) or tail median |Δs| > {TAIL_RATIO}× overall",
                        "hausdorff_ratio": t["hausdorff_over_length"] / o["hausdorff_over_length"],
                        "mean_gap_ratio": t["mean_gap_over_length"] / o["mean_gap_over_length"],
                        "abs_ds_ratio": t["abs_ds_median"] / max(o["abs_ds_median"], 1e-12)}
    out["tail_flag"]["flagged"] = bool(out["tail_flag"]["mean_gap_ratio"] > TAIL_RATIO or out["tail_flag"]["abs_ds_ratio"] > TAIL_RATIO)
    return out


# --- step 4 -----------------------------------------------------------------------------------------
PLATE_SRC = LADDER_A1 / "plate/plate.bin"
PLATE_OUT = ALM / "plate.bin"
CURVE_POINTS = 101  # header copy of γ and of the d → s map: s = 0, 0.01, …, 1
J_SPIRAL = 7


def u8(v, lo, hi):
    """0–254 over [lo, hi] (clamped), 255 = missing."""
    q = np.round(np.clip((v - lo) / (hi - lo), 0, 1) * 254)
    return np.where(np.isfinite(v), q, 255).astype(np.uint8)


def step_plate(d: Data) -> dict:
    """Format v3 = A1's ladder Plate (read only, every v2 section byte-identical) + the spiral axis fix + γ."""
    import plate_export as PE
    h, A = PE.read_plate(PLATE_SRC)
    assert h["answers"][J_SPIRAL]["id"] == F
    f = Fit(d, d.ctx.real)
    g = f.g
    gid = A["objid"].astype(np.int64)
    test = (A["flags"] & 1).astype(bool)
    assert not np.isin(gid[test], f.ids_tr).any(), "a Plate test galaxy is in the spiral probe's train rows"
    bank = np.load(LADDER_A1 / "c2_a1_embeddings.npz", allow_pickle=False)
    pos = {int(o): i for i, o in enumerate(bank["ids"])}
    Xp = bank["x"][[pos[int(o)] for o in gid]].astype(np.float64)
    real = d.ctx.real
    sh = np.flatnonzero(np.isin(gid, real.object_ids))[:500]  # the c2 and o1 banks agree where both hold a galaxy
    assert np.allclose(Xp[sh], real.x[real.rows_for(list(gid[sh]))], atol=1e-5)
    Np = d.nuisances(list(gid))
    ok = np.isfinite(Np).all(1)
    Zp = np.full_like(Xp, np.nan)
    Zp[ok] = f.z(Xp[ok], Np[ok])
    dp = Zp @ f.w + f.b
    sp = g.s(dp)
    rp = np.linalg.norm(Zp - g.full(np.where(ok, dp, 0.0)), axis=1)
    rp[~ok] = np.nan
    d_lo, d_hi = (float(x) for x in np.nanpercentile(dp, [0.1, 99.9]))
    r_hi = float(np.nanmax(rp))
    d8, s8, r8 = u8(dp, d_lo, d_hi), u8(sp, 0.0, 1.0), u8(rp, 0.0, r_hi)
    # into the file's 80-D coordinates: z_γ (probe-standardised, cleaned) → x → Plate standardised → basis − centre
    def to80(zk, vector=False):
        x = zk * f.scaler.scale_ + (0 if vector else f.scaler.mean_)
        st = (x - (0 if vector else A["std_mean"])) / A["std_scale"]
        return st @ A["basis"].T - (0 if vector else A["centre"])
    s_knots = np.linspace(0, 1, CURVE_POINTS)
    d_knots = g.d_of_s(s_knots)
    gam80 = to80(g.full(d_knots))
    t0 = (-g.g0 @ g.P) / g.speed0 * g.length; t1 = (g.g1 @ g.P) / g.speed1 * g.length
    tan0, tan1 = to80(t0[None], True)[0], to80(t1[None], True)[0]
    # w is a filter (a covector: d = w·z), Σw a pattern (a vector: where the mean moves as d grows); they map
    # into the 80-D coordinates differently, and neither need lie in the basis span, so the kept share is reported
    sig_w = np.cov(f.Z_tr.T) @ f.w
    w_st = (f.w / f.scaler.scale_) * A["std_scale"]  # filter in the Plate's standardised space
    w80_raw = A["basis"] @ w_st
    p80_raw = to80(sig_w[None], True)[0]
    p_st = sig_w * f.scaler.scale_ / A["std_scale"]
    kept = {"filter_w": float(np.linalg.norm(w80_raw) / np.linalg.norm(w_st)),
            "pattern_sigma_w": float(np.linalg.norm(p80_raw) / np.linalg.norm(p_st))}
    w80, p80 = w80_raw / np.linalg.norm(w80_raw), p80_raw / np.linalg.norm(p80_raw)
    # the old axis (QR coordinate 7, on the stored int8 z), M's raw probe (the stored read-out), the new one
    elig = A["counts"][h["answers"][J_SPIRAL]["q"]] >= 1
    y = A["votes"][J_SPIRAL] >= 128
    m = test & elig & ok
    Zq = A["coords"].astype(np.float64) * np.asarray(h["coords"]["scale"])[:, None]  # [80, n] dequantised
    qr7 = Zq[J_SPIRAL]
    pb = h["probe_basis"]
    wj = np.asarray(pb["w"][J_SPIRAL]) @ Zq + pb["b"][J_SPIRAL]
    aucs = {"n_plate_test_eligible": int(m.sum()), "n_pos": int(y[m].sum()),
            "old_qr_coordinate_7": auc(y[m], qr7[m]),
            "probe_basis_w7_on_coords": auc(y[m], wj[m]),
            "filter_w80_on_coords": auc(y[m], (w80 @ Zq)[m]),
            "raw_probe_readout_byte": auc(y[m], A["readout"][J_SPIRAL][m]),
            "raw_probe_384": h["reference_auc"][J_SPIRAL].get("auc_384") if isinstance(h.get("reference_auc"), list) else None,
            "new_cleaned_d_float": auc(y[m], dp[m]), "new_cleaned_d_byte": auc(y[m], d8[m]),
            "s_byte": auc(y[m], s8[m])}
    h2 = {k: v for k, v in h.items() if k != "sections"}
    h2["version"] = 3
    h2["spiral"] = {
        "answer": F, "answer_index": J_SPIRAL,
        "axis": "the spiral axis reads the CLEANED probe's decision d (spiral at fixed size and brightness), not QR "
                "coordinate 7 (answer 7's direction with answers 0–6 removed) and not readout[7] (the raw probe)",
        "spiral_d": {"decode": "d = lo + v*(hi − lo)/254; 255 missing", "lo": d_lo, "hi": d_hi,
                     "range": "0.1 and 99.9 percentiles of d over the file's galaxies; clamped outside"},
        "spiral_s": {"decode": "s = v/254, clamped to [0, 1]; 255 missing", "map": "d → s below"},
        "spiral_r": {"decode": "|r| = v * r_scale; 255 missing", "r_scale": r_hi / 254,
                     "meaning": "off-axis distance |z − γ(d)| in the 384-d probe-standardised cleaned space"},
        "cleaning": "ridge of the 384-d embedding on size, magnitude, snr, psf, redshift (each: train rank − 0.5, its "
                    "square, normal score, its square; centred), fitted on the ladder union's train rows; subtracted",
        "gamma": {"k": g.k, "arc_length_z": g.length, "s": s_knots.round(4).tolist(), "d": d_knots.round(5).tolist(),
                  "coords": gam80.round(5).tolist(),
                  "note": "γ at s = 0, 0.01, …, 1 in this file's 80-D coordinate space (float, not int8). γ lives in the "
                          "cleaned space; the 80-D basis is the raw embedding's, so a galaxy's own coords are not "
                          "cleaned and need not lie near γ"},
        "d_to_s": {"d": d_knots.round(6).tolist(), "s": s_knots.tolist(), "interpolate": "piecewise linear in between",
                   "below": {"ds_dd": g.speed0 / g.length}, "above": {"ds_dd": g.speed1 / g.length},
                   "note": "outside the knots s continues linearly (straight continuation of γ); monotone throughout"},
        "directions": {
            "filter_w": w80.round(6).tolist(), "pattern_sigma_w": p80.round(6).tolist(),
            "share_kept_in_basis": kept,
            "note": "unit vectors in this file's 80-D coordinates. filter_w is the cleaned probe's weight vector: use "
                    "it to READ a position (d grows along w·z). pattern_sigma_w is Σw, the direction the mean galaxy "
                    "moves as d grows: use it to WALK. They differ (cos in 384-d: see the report) because the probe "
                    "reads a low-variance direction; walking along w would leave the galaxies. Both are projected onto "
                    "the 80-D basis (share kept above) and the coords are the RAW embedding's, so w·z on coords is "
                    "not the cleaned d: per-galaxy d is spiral_d"},
        "extrapolation": {"rule": "γ(s) = γ(end) + (s − end) * tangent_end for s < 0 or s > 1; tangents carry the arc "
                                  "length (one unit of s = the whole of γ); defined, not rendered",
                          "tangent_s0": tan0.round(5).tolist(), "tangent_s1": tan1.round(5).tolist()},
        "construction": "docs/almagest/a1_spiral_report.md", "auc": aucs}
    h2["axes"] = {F: {"reads": "spiral_d", "header": "spiral", "superseded": "QR coordinate 7 / probe_basis.w[7] (kept)"}}
    secs = [(sec["name"], A[sec["name"]], sec["meaning"]) for sec in h["sections"]]
    secs += [("spiral_d", d8, "[n] cleaned spiral probe decision, header.spiral.spiral_d"),
             ("spiral_s", s8, "[n] γ arc-length coordinate s, header.spiral.spiral_s"),
             ("spiral_r", r8, "[n] off-axis |r|, header.spiral.spiral_r")]
    PE.write_plate(PLATE_OUT, h2, secs)
    h3, A3 = PE.read_plate(PLATE_OUT)
    same = all(np.array_equal(A3[k], A[k]) for k in A)
    # the header's d → s map against s computed in full
    s_map = np.interp(dp[ok], d_knots, s_knots)
    inside = (dp[ok] >= d_knots[0]) & (dp[ok] <= d_knots[-1])
    return {"source": str(PLATE_SRC.relative_to(REPO)), "source_sha1": sha1(PLATE_SRC),
            "out": str(PLATE_OUT.relative_to(REPO)), "out_sha1": sha1(PLATE_OUT), "bytes": PLATE_OUT.stat().st_size,
            "source_bytes": PLATE_SRC.stat().st_size, "n": int(len(gid)), "missing_nuisances": int((~ok).sum()),
            "v2_sections_identical": bool(same), "auc": aucs, "share_kept_in_basis": kept,
            "cos_w_sigma_w_384": float(f.w @ sig_w / (np.linalg.norm(f.w) * np.linalg.norm(sig_w))),
            "cos_w80_p80": float(w80 @ p80),
            "s_outside_0_1": int(((sp < 0) | (sp > 1))[ok].sum()), "d_clamped": int(((dp < d_lo) | (dp > d_hi))[ok].sum()),
            "d_to_s_map_max_abs_err_inside": float(np.max(np.abs(s_map[inside] - sp[ok][inside])))}


# --- step 5 -----------------------------------------------------------------------------------------
def step_figs(d: Data) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import j4_spread_controls as J
    gz = np.load(ALM / "a1_spiral_gamma.npz")
    rec = json.loads((DOCS / "a1_spiral_curve.json").read_text())["curve"]
    INK, RED, GREY = "#1d1b19", "#c0392b", "#8a8580"
    # 1. γ on its top-2 PCs, raw vs cleaned
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    for a, (name, cleaned) in zip(ax, (("raw", False), ("cleaned", True))):
        f = Fit(d, d.ctx.real, cleaned=cleaned)
        g = f.g
        D = g.at(g.d_of_s(np.linspace(0, 1, 400)))
        a.plot(D[:, 0], D[:, 1], "-", color=RED, lw=2, label="γ(d), smoothing spline")
        a.plot(g.Y[:, 0], g.Y[:, 1], "o", color=INK, ms=4, label=f"{N_DBINS} equal-count bin means of d (train)")
        for sv in np.linspace(0, 1, 11):
            p = g.at(g.d_of_s([sv]))[0]
            a.annotate(f"{sv:.1f}", p[:2], fontsize=7, color=RED, xytext=(3, 3), textcoords="offset points")
        a.set_title(f"A1 spiral γ, {name} embedding: top 2 of {g.k} PCs, s marked (bend {g.describe()['bend_1_minus_chord_over_arc']:.3f})", fontsize=8.5)
        a.set_xlabel("PC1"); a.set_ylabel("PC2"); a.set_aspect("equal", adjustable="datalim"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(DOCS / "a1_spiral_path.png", dpi=130); plt.close(fig)
    # 2. γ's sideways displacement against s, A1 and the untrained range
    s101 = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    un = rec["untrained"]
    for a, key, title in ((ax[0], "_dev_profile", "off its own chord (bend)"),
                          (ax[1], "_side_profile", "off the Σw line (the pattern direction)")):
        U = np.array([v["bend"][key] for v in un.values()])
        a.fill_between(s101, U.min(0), U.max(0), color=GREY, alpha=0.35, lw=0, label="untrained seeds (range of 3, own spans)")
        a.plot(s101, rec["a1"]["bend"][key], color=RED, lw=2, label="A1")
        a.set_xlabel("s"); a.set_ylabel("displacement / path length"); a.set_title(f"γ's displacement {title}", fontsize=9)
        a.set_xlim(0, 1); a.set_ylim(bottom=0); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(DOCS / "a1_spiral_sideways.png", dpi=130); plt.close(fig)
    # 3. strip: test galaxies with s in [t − 0.05, t + 0.05], the 5 nearest γ(t)
    f = Fit(d, d.ctx.real)
    g = f.g
    _, cache = J.check(verbose=False)
    fig, axes = plt.subplots(5, 11, figsize=(11 * 1.1, 5 * 1.15))
    chosen = {}
    for c, t in enumerate(np.linspace(0, 1, 11)):
        cand = np.flatnonzero(np.abs(f.s_te - t) <= 0.05)
        dist = np.linalg.norm(f.Z_te[cand] - g.full(g.d_of_s([t]))[0], axis=1)
        pick = cand[np.argsort(dist)[:5]]
        chosen[f"{t:.1f}"] = {"objid": [int(i) for i in f.ids_te[pick]], "s": f.s_te[pick].round(3).tolist(),
                              "vote_fraction": f.v_te[pick].round(3).tolist(), "n_in_slice": int(cand.size),
                              "dist_over_length": (np.sort(dist)[:5] / g.length).round(3).tolist()}
        for r, i in enumerate(pick):
            im = np.asarray(cache.get(int(f.ids_te[i])), dtype=np.float32)
            rgb = np.stack([im[2], im[1], im[0]], -1)
            lo, hi = np.percentile(rgb, [1, 99.7])
            axes[r, c].imshow(np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)[64:192, 64:192])
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(f"s = {t:.1f}", fontsize=9)
    fig.suptitle("A1 spiral s (cleaned): test galaxies in each s slice (±0.05) nearest γ(s); central 128 px of the 256 px stamp, i r g", fontsize=9)
    fig.tight_layout(); fig.savefig(DOCS / "a1_spiral_strip.png", dpi=110); plt.close(fig)
    return {"path_figure": "docs/almagest/a1_spiral_path.png", "sideways_figure": "docs/almagest/a1_spiral_sideways.png",
            "strip": "docs/almagest/a1_spiral_strip.png", "strip_galaxies": chosen,
            "gamma_npz_length_matches": bool(np.isclose(float(gz["length"]), g.length))}


def main() -> None:
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step not in ("r1", "clean", "curve", "plate", "figs"):
        raise SystemExit(__doc__)
    if BASIS != "primary" and step not in ("clean", "curve"):
        raise SystemExit("ALM_BASIS=log is the sensitivity reading: clean and curve only")
    t0 = time.perf_counter()
    d = Data()
    rec = {"provenance": provenance(d), "basis": BASIS, "note": "descriptive, for Almagest; not in the science record"}
    if step == "r1":
        rec["r1"] = step_r1(d); save("a1_spiral_r1.json", rec)
    elif step == "clean":
        rec["clean"] = step_clean(d); save(f"a1_spiral_clean{SUFFIX}.json", rec)
    elif step == "curve":
        rec["curve"] = step_curve(d); save(f"a1_spiral_curve{SUFFIX}.json", rec)
    elif step == "plate":
        rec["plate"] = step_plate(d); save("a1_spiral_plate.json", rec)
    elif step == "figs":
        rec["figs"] = step_figs(d); save("a1_spiral_figs.json", rec)
    print(f"{step}: {time.perf_counter() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
