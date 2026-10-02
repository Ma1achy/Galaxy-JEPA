"""Layer 2 leakage audit (plan A7): can the pixels alone say how a stamp was processed?

Pre-registration: `artifacts/leakage_audit.md`. Data only: no encoder is involved. Per processing
variable, a ridge on image statistics and a small CNN are fitted to recover it from the pixels. A
physics-only baseline (gradient-boosted trees on catalogue physics) is fitted beside them, and a
second stage on physics ⊕ the pixel prediction measures what the pixels add. The leak is that
held-out excess (ΔR², or ΔAUC for corpus membership), with a family-corrected bootstrap bound.

The identity criterion (shared objIDs, near-duplicates, stamp-footprint overlap across the two
corpora) reads coordinates only.

  uv run python artifacts/leakage_audit.py plants [--only S0,S1,...] [--n N]
  uv run python artifacts/leakage_audit.py calibrate --alphas 0.1,0.2   # the threshold plant's dose
  uv run python artifacts/leakage_audit.py pull                          # the baseline's physics, once, pinned
  uv run python artifacts/leakage_audit.py audit                         # refuses until hashed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sciserver_cut_v2 import bilinear_shift, fourier_shift  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "out" / "leakage"
PREREG = Path(__file__).with_name("leakage_audit.md")
PREREG_SHA1: str | None = None  # filled at hash time; `audit` refuses until the section matches it

SEED, N_FOLDS, N_PER_CORPUS = 0, 5, 20_000
N_BOOT, ALPHA, FLOOR = 20_000, 0.05, 0.01  # FLOOR: ΔR² 0.01 ≈ |ρ| 0.1, the pilot checks' bound
MIN_ROWS = 15_000  # a stated variable with fewer usable rows in the audit is INSUFFICIENT
MIN_TAIL = 10  # bootstrap draws beyond the family-corrected bound; fewer and the bar is unreachable
BANDS = ("g", "r", "i")
SHIFTS = tuple(f"{b}_s{a}" for b in BANDS for a in "xy")  # the applied shift s_b (≡ frac(origin))
V1_REL = ("gr_v1x", "gr_v1y", "ir_v1x", "ir_v1y")  # v1's in-stamp offsets, g−r and i−r (M's code)
STATED = SHIFTS + V1_REL  # per corpus; plus corpus membership on the pooled pair
FAMILY = 2 * (2 * len(STATED) + 1)  # (2 corpora × 10 + corpus) × 2 predictors = 42
CAMCOL_FAMILY = 6
CALIB_ALPHAS = (0.01, 0.02)  # the threshold plant's sweep; α* read off it by the rule in `calibrate`
ORDER = ("LEAK", "UNRESOLVED", "TRACE", "CLEAN")  # worst first; INSUFFICIENT precedes all


# ── the pre-registration lock ───────────────────────────────────────────────────────────────────────

def prereg_section() -> str:
    """From '## Pre-registration' to the next level-2 heading, trailing newline (as aligned_comparison)."""
    lines = PREREG.read_text().splitlines()
    a = lines.index("## Pre-registration")
    b = next((k for k in range(a + 1, len(lines)) if lines[k].startswith("## ")), len(lines))
    return "\n".join(lines[a:b]).rstrip() + "\n"


def assert_hashed() -> None:
    if PREREG_SHA1 is None:
        raise SystemExit("leakage_audit: the pre-registration is not hashed yet; the audit does not run")
    got = hashlib.sha1(prereg_section().encode()).hexdigest()
    if got != PREREG_SHA1:
        raise SystemExit(f"leakage_audit: pre-registration changed since the hash ({got} != {PREREG_SHA1})")


def assert_resolution(n_boot: int = N_BOOT, family: int = FAMILY) -> None:
    """A bar the bootstrap cannot resolve is an error, not a CLEAN (CLAUDE.md)."""
    tail = n_boot * ALPHA / family
    if tail < MIN_TAIL:
        raise SystemExit(f"leakage_audit: {n_boot} draws leave {tail:.1f} beyond α/{family}; need ≥ {MIN_TAIL}")


# ── pixel predictors ────────────────────────────────────────────────────────────────────────────────

def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean((-2, -1), keepdims=True)
    b = b - b.mean((-2, -1), keepdims=True)
    return (a * b).sum((-2, -1)) / np.sqrt((a * a).sum((-2, -1)) * (b * b).sum((-2, -1)) + 1e-30)


def _moments(a: np.ndarray) -> list[np.ndarray]:
    m = a.mean((-2, -1))
    d = a - m[..., None, None]
    v = (d * d).mean((-2, -1)) + 1e-30
    return [m, np.sqrt(v), (d ** 3).mean((-2, -1)) / v ** 1.5, (d ** 4).mean((-2, -1)) / v ** 2 - 3]


def image_stats(x: np.ndarray) -> np.ndarray:
    """(n, 3, S, S) → (n, 65): the plan's list — per-band moments, windowed centroids and second
    moments, cross-band correlation (with its ±1 px asymmetry, the shift-sensitive part), lag-1
    autocorrelation on the border strips, border statistics."""
    x = x.astype(np.float64)
    n, _, s, _ = x.shape
    h, e = min(16, s // 4), 8
    ctr = x[:, :, s // 2 - h:s // 2 + h, s // 2 - h:s // 2 + h]
    ring = np.ones((s, s), bool)
    ring[e:-e, e:-e] = False
    sky = np.median(x[:, :, ring], -1)
    f = []
    for b in range(3):
        f += _moments(x[:, b]) + _moments(ctr[:, b])
    # windowed centroid (σ = 3 px, 6 iterations) and second moments, background-subtracted
    yy, xx = np.mgrid[-h:h, -h:h] + 0.5
    img = np.clip(ctr - sky[:, :, None, None], 0, None)
    cx, cy = np.zeros((n, 3)), np.zeros((n, 3))
    for _ in range(6):
        w = img * np.exp(-((xx - cx[..., None, None]) ** 2 + (yy - cy[..., None, None]) ** 2) / 18.0)
        sw = w.sum((-2, -1)) + 1e-30
        cx, cy = (w * xx).sum((-2, -1)) / sw, (w * yy).sum((-2, -1)) / sw
    dx, dy = xx - cx[..., None, None], yy - cy[..., None, None]
    f += [cx[:, b] for b in range(3)] + [cy[:, b] for b in range(3)]
    f += [(w * q).sum((-2, -1))[:, b] / sw[:, b] for q in (dx * dx, dy * dy, dx * dy) for b in range(3)]
    f += [cx[:, 0] - cx[:, 1], cy[:, 0] - cy[:, 1], cx[:, 2] - cx[:, 1], cy[:, 2] - cy[:, 1]]
    g, r, i = ctr[:, 0], ctr[:, 1], ctr[:, 2]
    f += [_corr(g, r), _corr(r, i), _corr(g, i)]
    for a in (g, i):
        f += [_corr(a[:, :, 1:], r[:, :, :-1]) - _corr(a[:, :, :-1], r[:, :, 1:]),
              _corr(a[:, 1:, :], r[:, :-1, :]) - _corr(a[:, :-1, :], r[:, 1:, :])]
    for b in range(3):
        top, left = x[:, b, :e, :], x[:, b, :, :e]
        f += [_corr(top[:, :, 1:], top[:, :, :-1]), _corr(left[:, 1:, :], left[:, :-1, :])]
        f += [sky[:, b], x[:, b, ring].std(-1), (x[:, b, ring] == 0).mean(-1)]
    return np.stack(f, 1)


def features(st: np.ndarray) -> np.ndarray:
    chunk = int(min(500, max(50, 2e7 // np.prod(st.shape[1:]))))  # ~160 MB of float64 per chunk
    return np.concatenate([image_stats(np.asarray(st[k:k + chunk])) for k in range(0, len(st), chunk)])


def _ridge(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(xtr)
    a, b = np.nan_to_num(sc.transform(xtr)), np.nan_to_num(sc.transform(xte))
    out = np.full((len(xte), ytr.shape[1]), np.nan)
    for t in range(ytr.shape[1]):
        ok = np.isfinite(ytr[:, t])
        if ok.sum() > 10:
            out[:, t] = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(a[ok], ytr[ok, t]).predict(b)
    return out


CNN = {"widths": (32, 64, 64, 128), "epochs": 8, "batch": 128, "lr": 2e-3, "wd": 1e-4, "val": 0.1}


def _net(n_out: int):
    import torch.nn as nn
    ch, layers = (5,) + CNN["widths"], []
    for k, (a, b) in enumerate(zip(ch[:-1], ch[1:])):
        layers += [nn.Conv2d(a, b, 5 if k == 0 else 3, stride=2, padding=2 if k == 0 else 1),
                   nn.BatchNorm2d(b), nn.GELU()]
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(4), nn.Flatten(),
                         nn.Linear(ch[-1] * 16, 128), nn.GELU(), nn.Linear(128, n_out))


def _device():
    import torch
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _batch(st, idx: np.ndarray, scale, dev):
    """asinh on a fixed per-band scale (the sample's median border σ, no per-stamp normalisation, so
    noise level stays readable) plus two coordinate channels (absolute position must be visible)."""
    import torch
    x = torch.from_numpy(np.asarray(st[np.sort(idx)], dtype=np.float32)).to(dev)
    x = torch.asinh(x / scale)
    s = x.shape[-1]
    g = torch.linspace(-1, 1, s, device=dev)
    return torch.cat([x, g[None, None, None, :].expand(len(x), 1, s, s),
                      g[None, None, :, None].expand(len(x), 1, s, s)], 1), np.sort(idx)


def _cnn(st, tr: np.ndarray, ytr: np.ndarray, te: np.ndarray, kind: str, scale: np.ndarray, seed: int) -> np.ndarray:
    """One fold: multi-output regression (masked MSE on standardised targets) or corpus BCE; the best
    epoch on a 10% inner split is kept. Returns predictions for `te` in `te`'s order."""
    import torch
    torch.manual_seed(seed)
    dev = _device()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tr))
    nv = int(len(tr) * CNN["val"])
    va, fit = perm[:nv], perm[nv:]
    mu, sd = np.nanmean(ytr, 0), np.nanstd(ytr, 0) + 1e-12
    yz = torch.from_numpy(((ytr - mu) / sd) if kind == "reg" else ytr).float()
    sc = torch.from_numpy(scale.astype(np.float32)).to(dev)[None, :, None, None]
    net = _net(ytr.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=CNN["lr"], weight_decay=CNN["wd"])
    steps = CNN["epochs"] * int(np.ceil(len(fit) / CNN["batch"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CNN["lr"], total_steps=steps)
    pos = {int(r): k for k, r in enumerate(tr)}

    def loss_of(out, y):
        if kind == "bin":
            return torch.nn.functional.binary_cross_entropy_with_logits(out, y)
        ok = torch.isfinite(y)
        return ((out - torch.nan_to_num(y)) ** 2)[ok].mean()

    def run(rows, train: bool):
        net.train(train)
        tot, cnt = 0.0, 0
        order = rng.permutation(len(rows)) if train else np.arange(len(rows))
        for k in range(0, len(rows), CNN["batch"]):
            x, idx = _batch(st, tr[rows[order[k:k + CNN["batch"]]]], sc, dev)
            y = yz[[pos[int(i)] for i in idx]].to(dev)
            with torch.set_grad_enabled(train):
                ls = loss_of(net(x), y)
            if train:
                opt.zero_grad()
                ls.backward()
                opt.step()
                sched.step()
            tot, cnt = tot + float(ls.detach()) * len(idx), cnt + len(idx)
        return tot / cnt

    best, state = np.inf, None
    for _ in range(CNN["epochs"]):
        run(fit, True)
        v = run(va, False)
        if v < best:
            best, state = v, {k: t.detach().clone() for k, t in net.state_dict().items()}
    net.load_state_dict(state)
    net.eval()
    out = np.zeros((len(te), ytr.shape[1]))
    where = {int(r): k for k, r in enumerate(te)}
    with torch.no_grad():
        for k in range(0, len(te), 256):
            x, idx = _batch(st, te[k:k + 256], sc, dev)
            p = net(x).cpu().numpy().astype(np.float64)
            out[[where[int(i)] for i in idx]] = p * sd + mu if kind == "reg" else p
    return out


def _free_mps() -> None:
    """The MPS allocator keeps its cache after a fit; on the 18 GB box that pushes the next stage to swap."""
    import torch
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ── the physics baseline and the second stage ───────────────────────────────────────────────────────

def _hgb(kind: str):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    kw = {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 100,
          "early_stopping": False, "random_state": SEED}
    return HistGradientBoostingRegressor(**kw) if kind == "reg" else HistGradientBoostingClassifier(**kw)


def _oof_hgb(x: np.ndarray, y: np.ndarray, folds: np.ndarray, kind: str) -> np.ndarray:
    from threadpoolctl import threadpool_limits
    out = np.full(len(y), np.nan)
    ok = np.isfinite(y)
    # One OpenMP thread: torch's and sklearn's libomp both load, and their barriers thrash (6× slower at 4)
    with threadpool_limits(limits=1, user_api="openmp"):
        for k in range(N_FOLDS):
            tr, te = ok & (folds != k), folds == k
            m = _hgb(kind).fit(x[tr], y[tr])
            out[te] = m.predict(x[te]) if kind == "reg" else m.predict_proba(x[te])[:, 1]
    return out


# ── metrics under shared Poisson weights ────────────────────────────────────────────────────────────

def _r2(w: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """w (c, n), p (k, n) → (c, k) weighted R²."""
    sw = w.sum(1)
    my = w @ y / sw
    sst = w @ (y * y) - sw * my * my
    return 1 - (w @ ((y[None] - p) ** 2).T) / sst[:, None]


def _auc(w: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Weighted AUC with ties at half credit, (c, k)."""
    from scipy import sparse
    out = np.zeros((len(w), len(p)))
    for j, s in enumerate(p):
        u, inv = np.unique(s, return_inverse=True)
        g = sparse.csr_matrix((np.ones(len(s)), (np.arange(len(s)), inv)), shape=(len(s), len(u)))
        wp, wn = np.asarray((g.T @ (w * y).T).T), np.asarray((g.T @ (w * (1 - y)).T).T)
        below = np.cumsum(wn, 1) - wn
        out[:, j] = ((wp * (below + 0.5 * wn)).sum(1)) / (wp.sum(1) * wn.sum(1))
    return out


def boot(y: np.ndarray, p: np.ndarray, kind: str, seed: int, mask: np.ndarray | None = None,
         n_boot: int = N_BOOT, chunk: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Point metrics (k,) and bootstrap metrics (n_boot, k) of each row of p, paired by galaxy."""
    ok = np.isfinite(y) & np.isfinite(p).all(0) & (True if mask is None else mask)
    y, p = y[ok], p[:, ok]
    f = _r2 if kind == "reg" else _auc
    rng = np.random.default_rng(seed)
    bs = [f(rng.poisson(1.0, (min(chunk, n_boot - k), len(y))).astype(np.float64), y, p)
          for k in range(0, n_boot, chunk)]
    return f(np.ones((1, len(y))), y, p)[0], np.concatenate(bs)


def _state(delta: float, lo: float) -> str:
    if lo > 0:
        return "LEAK" if delta >= FLOOR else "TRACE"
    return "UNRESOLVED" if delta >= FLOOR else "CLEAN"


def _worst(states) -> str:
    return min(states, key=ORDER.index)


# ── the core: one path for plants and audit ─────────────────────────────────────────────────────────

def score_block(st, rows: np.ndarray, feats: np.ndarray, phys: np.ndarray, targets: dict, kind: str,
                folds: np.ndarray, scale: np.ndarray, camcol: np.ndarray, stated: tuple,
                cond: np.ndarray | None = None, family: int = FAMILY, log=print,
                alt: tuple[np.ndarray, np.ndarray | None] | None = None) -> dict:
    """Out-of-fold pixel predictions (ridge, CNN), physics and physics ⊕ pixel second stages, bootstrap
    bounds, per-variable states. `targets` maps name → (n,) array; `stated` names those with a state.
    `alt` = (physics, conditions) of the reported second baseline, scored into r["alt"]."""
    names = list(targets)
    y = np.stack([targets[t] for t in names], 1).astype(np.float64)
    pix = {"ridge": np.full(y.shape, np.nan), "cnn": np.full(y.shape, np.nan)}
    t0 = time.time()
    for k in range(N_FOLDS):
        tr, te = folds != k, folds == k
        pix["ridge"][te] = _ridge(feats[tr], y[tr], feats[te])
        pix["cnn"][te] = _cnn(st, rows[tr], y[tr], rows[te], kind, scale, SEED + k)
        _free_mps()
        log(f"    fold {k}: {time.time() - t0:.0f}s")
    q = ALPHA / family
    res = {}
    for j, t in enumerate(names):
        yt = y[:, j]
        r = {"n": int(np.isfinite(yt).sum())}
        seed = SEED + 1 + j
        if t not in stated:  # readable, reported only: the pixels' own metric (camcol one-hots by AUC)
            pt, bt = boot(yt, np.stack([pix["ridge"][:, j], pix["cnn"][:, j]]),
                          "bin" if t.startswith("camcol_") else kind, seed)
            r["pixel"] = {m: {"metric": float(pt[a]), "lo": float(np.quantile(bt[:, a], q))}
                          for a, m in enumerate(("ridge", "cnn"))}
            res[t] = r
            continue
        sec, base, both = _second_stage(phys, cond, pix, j, yt, folds, kind, seed, q)
        r.update(sec)
        if alt is not None:  # the same pixel predictions over the other baseline (user, 2026-09-28: sky)
            r["alt"] = _second_stage(alt[0], alt[1], pix, j, yt, folds, kind, seed, q)[0]
        if kind == "reg":  # kept for the corpus-level camcol test; stripped before writing
            m = max(("ridge", "cnn"), key=lambda k: r[k]["excess"])
            r["_pred"] = (yt, base, both[m])
        res[t] = r
    return res


def _second_stage(phys, cond, pix, j, yt, folds, kind, seed, q) -> tuple[dict, np.ndarray, dict]:
    """Trees on physics alone and on physics ⊕ each pixel prediction (⊕ conditions if given), paired
    bootstrap, per-predictor states. The bootstrap seed is the variable's, so two baselines scored
    over the same pixel predictions share their draws."""
    base = _oof_hgb(phys, yt, folds, kind)
    both = {m: _oof_hgb(np.column_stack([phys, pix[m][:, j]]), yt, folds, kind) for m in pix}
    rows_p = [base, both["ridge"], both["cnn"], pix["ridge"][:, j], pix["cnn"][:, j]]
    if cond is not None:
        cb = _oof_hgb(np.column_stack([phys, cond]), yt, folds, kind)
        rows_p += [cb] + [_oof_hgb(np.column_stack([phys, cond, pix[m][:, j]]), yt, folds, kind) for m in pix]
    pt, bt = boot(yt, np.stack(rows_p), kind, seed)
    chance = 0.0 if kind == "reg" else 0.5
    r: dict = {"physics": float(pt[0])}
    for a, m in enumerate(("ridge", "cnn")):
        d, db = pt[1 + a] - pt[0], bt[:, 1 + a] - bt[:, 0]
        r[m] = {"pixel": float(pt[3 + a]), "pixel_lo": float(np.quantile(bt[:, 3 + a], q)),
                "readable": bool(np.quantile(bt[:, 3 + a], q) > chance),
                "both": float(pt[1 + a]), "excess": float(d), "lo": float(np.quantile(db, q)),
                "state": _state(float(d), float(np.quantile(db, q)))}
        if cond is not None:
            dc, dcb = pt[6 + a] - pt[5], bt[:, 6 + a] - bt[:, 5]
            r[m]["cond_excess"], r[m]["cond_lo"] = float(dc), float(np.quantile(dcb, q))
            r[m]["cond_state"] = _state(float(dc), float(np.quantile(dcb, q)))
    r["state"] = _worst([r["ridge"]["state"], r["cnn"]["state"]])
    r["label"] = []
    if r["state"] == "CLEAN" and (r["ridge"]["readable"] or r["cnn"]["readable"]):
        r["label"].append("PHYSICS-EXPLAINED")
    if cond is not None and r["state"] in ("LEAK", "TRACE", "UNRESOLVED"):
        r["label"].append(_conditions_label(r))
    return r, base, both


def _conditions_label(r: dict) -> str:
    """Explained iff, for every predictor that is not CLEAN, the excess over physics + conditions is
    below the floor (CLEAN or TRACE). Requiring CLEAN failed S5: at AUC ≈ 1 a 1.4e-5 residue is
    significant, which is the magnitude-versus-significance case the floor exists for."""
    cs = [r[m]["cond_state"] for m in ("ridge", "cnn") if r[m]["state"] != "CLEAN"]
    return "CONDITIONS-EXPLAINED" if all(c in ("CLEAN", "TRACE") for c in cs) else "NOT CONDITIONS-EXPLAINED"


def camcol_structure(res: dict, camcol: np.ndarray, shifts: tuple, seed: int) -> dict:
    """One test per corpus, over its non-CLEAN shift variables (each at its larger-excess predictor):
    d̄_c = mean over variables of (excess inside camcol c − excess outside c), shared Poisson weights.
    Structured iff some d̄_c's one-sided bound at α / (6 camcols × 2 corpora) exceeds 0."""
    hit = [t for t in shifts if t in res and res[t]["state"] != "CLEAN"]
    if not hit:
        return {"tested": [], "structured": False, "where": []}
    n = len(camcol)
    rng = np.random.default_rng(seed)
    w = np.vstack([np.ones((1, n)), rng.poisson(1.0, (4_000, n))]).astype(np.float32)
    g = np.stack([camcol == c for c in range(1, 7)], 1).astype(np.float32)  # n × 6
    d = np.zeros((len(w), 6))
    for t in hit:
        y, base, both = res[t]["_pred"]
        ok = (np.isfinite(y) & np.isfinite(base) & np.isfinite(both)).astype(np.float32)
        y0, e0, e1 = (np.nan_to_num(v).astype(np.float32) for v in (y, (y - base) ** 2, (y - both) ** 2))
        cols = [ok, ok * y0, ok * y0 * y0, ok * e0, ok * e1]
        sums_in = np.stack([w @ (c[:, None] * g) for c in cols])  # 5 × B × 6
        sums_all = np.stack([w @ c for c in cols])[:, :, None]  # 5 × B × 1

        def excess(s_):
            sw, sy, syy, s0, s1 = s_
            sst = syy - sy * sy / sw
            return (s0 - s1) / sst  # R²(both) − R²(base)
        d += excess(sums_in) - excess(sums_all - sums_in)
    d /= len(hit)
    q = ALPHA / (6 * 2)
    lo = np.quantile(d[1:], q, axis=0)
    where = [f"camcol{c + 1}" for c in range(6) if lo[c] > 0]
    return {"tested": hit, "d": [float(v) for v in d[0]], "lo": [float(v) for v in lo],
            "structured": bool(where), "where": where}


BLOCKING = ("INSUFFICIENT", "INVALID", "LEAK", "UNRESOLVED")  # TRACE is reported, not blocking (user, 2026-09-28)


def verdict(blocks: dict) -> dict:
    """INSUFFICIENT > LEAK > UNRESOLVED > TRACE > CLEAN over every stated variable. TRACE is named and
    proceeds: the audit resolves ~0.003 px rms, so significance below ε is expected at this n."""
    st = {f"{b}:{t}": r["state"] for b, res in blocks.items() for t, r in res.items()
          if not t.startswith("_") and "state" in r}
    if any(s == "INSUFFICIENT" for s in st.values()):
        return {"state": "INSUFFICIENT", "variables": st, "blocks_training": True}
    s = _worst(st.values())
    return {"state": s, "variables": st, "blocks_training": s in BLOCKING}


def view(res: dict, which: str = "alt") -> dict:
    """The run as read under the reported baseline: each scored variable's r["alt"] in its place.
    `_camcol` stays the primary's (the camcol test reads the primary's predictions)."""
    blocks = {b: {t: (r["alt"] if not t.startswith("_") and "alt" in r else r) for t, r in res_.items()}
              for b, res_ in res["blocks"].items()}
    return {"blocks": blocks, "verdict": verdict(blocks)}


def run(corpora: dict, phys_cols: list[str], cond_cols: list[str], stated: tuple = STATED,
        reported: tuple = (), shifts: tuple = STATED, log=print,
        alt: tuple[list[str], list[str]] | None = None) -> dict:
    """`corpora`: name → {"st": stamps (n,3,S,S), "table": dict of arrays incl. phys/cond cols, targets,
    "camcol", "fold"}. Per corpus: every stated and reported target; then corpus membership pooled.
    `alt` = (physics cols, condition cols) of the reported second baseline (without sky): scored from
    the same pixel predictions, read with `view`."""
    assert_resolution()
    blocks, feats, scale = {}, {}, {}
    for name, c in corpora.items():
        t0 = time.time()
        feats[name] = features(c["st"])
        ring = np.ones(c["st"].shape[-2:], bool)
        ring[8:-8, 8:-8] = False
        sub = np.asarray(c["st"][:: max(1, len(c["st"]) // 2000)], dtype=np.float64)[:, :, ring]
        scale[name] = np.median(1.4826 * np.median(np.abs(sub - np.median(sub, -1, keepdims=True)), -1), 0)
        log(f"  {name}: features {feats[name].shape} in {time.time() - t0:.0f}s; scale {np.round(scale[name], 4)}")
    common = np.mean(list(scale.values()), 0)
    for name, c in corpora.items():
        tb = c["table"]
        tg = {t: tb[t] for t in stated + reported if t in tb and np.nanstd(tb[t]) > 0}
        phys = np.column_stack([tb[k] for k in phys_cols])
        alt_c = None if alt is None else (np.column_stack([tb[k] for k in alt[0]]), None)
        log(f"  {name}: {len(tg)} targets")
        blocks[name] = score_block(c["st"], np.arange(len(c["st"])), feats[name], phys, tg, "reg",
                                   tb["fold"], common, tb["camcol"], stated, log=log, alt=alt_c)
        cs = camcol_structure(blocks[name], tb["camcol"], shifts, SEED + 7)
        blocks[name]["_camcol"] = cs
    if len(corpora) == 2:
        (a, ca), (b, cb) = corpora.items()
        st = _Stack(ca["st"], cb["st"])
        cols = set(phys_cols + cond_cols + ([] if alt is None else alt[0] + alt[1]))
        cat = {k: np.concatenate([ca["table"][k], cb["table"][k]]) for k in sorted(cols) + ["fold", "camcol"]}
        y = np.r_[np.zeros(len(ca["st"])), np.ones(len(cb["st"]))]

        def _pc(p_cols, c_cols):
            return (np.column_stack([cat[k] for k in p_cols]),
                    np.column_stack([cat[k] for k in c_cols] + [cat["camcol"] == c for c in range(1, 7)]).astype(float))
        phys, cond = _pc(phys_cols, cond_cols)
        log("  corpus membership")
        blocks["pooled"] = score_block(st, np.arange(len(st)), np.concatenate([feats[a], feats[b]]), phys,
                                       {"corpus": y}, "bin", cat["fold"], common, cat["camcol"],
                                       ("corpus",), cond=cond, log=log, alt=None if alt is None else _pc(*alt))
    for res in blocks.values():
        for r in res.values():
            r.pop("_pred", None)
    out = {"blocks": blocks, "verdict": verdict(blocks)}
    if alt is not None:
        out["verdict_alt"] = view(out)["verdict"]
    return out


class _Stack:
    """Two stamp arrays indexed as one, without a copy."""

    def __init__(self, a, b):
        self.a, self.b, self.shape = a, b, (len(a) + len(b),) + tuple(a.shape[1:])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        idx = np.atleast_1d(idx)
        na = len(self.a)
        out = np.empty((len(idx),) + self.shape[1:], np.float32)
        ia, ib = idx < na, idx >= na
        if ia.any():
            out[ia] = self.a[idx[ia]]
        if ib.any():
            out[ib] = self.b[idx[ib] - na]
        return out


# ── re-injection (the plan's plant) ─────────────────────────────────────────────────────────────────

def _margin(s: int) -> int:
    """The smallest margin ≥ s/4 whose padded length s + 2m + 1 is odd and 7-smooth (a prime length
    such as 97 sends the FFT down Bluestein's path, ~10× slower)."""
    m = s // 4
    while True:
        p = s + 2 * m + 1
        for f in (3, 5, 7):
            while p % f == 0:
                p //= f
        if p == 1:
            return m
        m += 1


def reinject(x: np.ndarray, off: np.ndarray, alpha: float = 1.0, bands=(0, 1, 2)) -> np.ndarray:
    """Move band b's content by α·off[b] = (dy, dx) px: reflect-padded to odd P, the cutter's Fourier
    shift, cropped back. `x` (3,S,S), `off` (3,2)."""
    s = x.shape[-1]
    m = _margin(s)
    out = x.astype(np.float64).copy()
    for b in bands:
        p = np.pad(out[b], ((m, m + 1), (m, m + 1)), mode="reflect")
        out[b] = fourier_shift(p, alpha * off[b, 0], alpha * off[b, 1])[m:m + s, m:m + s]
    return out


# ── synthetic corpora ───────────────────────────────────────────────────────────────────────────────

def _gauss(yy, xx, cy, cx, cov, flux):
    det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] ** 2
    inv = np.stack([cov[:, 1, 1], -cov[:, 0, 1], cov[:, 0, 0]], 1) / det[:, None]
    dx, dy = xx[None] - cx[:, None, None], yy[None] - cy[:, None, None]
    q = inv[:, 0, None, None] * dx * dx + 2 * inv[:, 1, None, None] * dx * dy + inv[:, 2, None, None] * dy * dy
    return flux[:, None, None] / (2 * np.pi * np.sqrt(det))[:, None, None] * np.exp(-0.5 * q)


def synth(n: int, seed: int, corpus: str, size: int = 64, inject: float = 0.0, inject_camcol: int | None = None,
          shift: str = "fourier", pedestal: float = 0.0, chunk: int = 500) -> tuple[np.ndarray, dict]:
    """Two-component (bulge + disc) galaxies with colour gradients, a per-band PSF and white sky
    noise, rendered on the frame grid at a random sub-pixel phase per band and cut as the v2 cutter
    does (padded to odd P, shifted by s_b, cropped). Corpus 'B' is probe-like: brighter, larger,
    redder. `inject` re-injects α × v1's offsets (−s_b); `shift='bilinear'` is the pilot's plant."""
    rng = np.random.default_rng(seed)
    b_like = corpus == "B"
    mag = rng.uniform(14.0, 17.0, n) if b_like else rng.uniform(14.5, 17.8, n)
    rad = np.exp(rng.normal(np.log(2.8 if b_like else 2.2), 0.3, n))
    gr = rng.normal(0.75 if b_like else 0.6, 0.2, n)
    ri = rng.normal(0.38 if b_like else 0.33, 0.1, n)
    z = np.clip(0.02 + 0.03 * (mag - 14) / rad * 2.5 * np.exp(rng.normal(0, 0.15, n)), 0.005, None)
    bf, q, th = rng.uniform(0.1, 0.7, n), rng.uniform(0.4, 1.0, n), rng.uniform(0, np.pi, n)
    psf = rng.uniform(1.0, 1.8, n)
    camcol = rng.integers(1, 7, n)
    ped = np.full(n, pedestal)
    s_all = rng.uniform(-0.5, 0.5, (n, 3, 2))  # (dy, dx) per band: the applied shift s_b
    fr = 50 * 10 ** (-0.4 * (mag - 20))
    m = _margin(size)
    p = size + 2 * m + 1
    c = (size - 1) / 2 + m
    yy, xx = np.mgrid[0:p, 0:p].astype(np.float64)
    st = np.empty((n, 3, size, size), np.float16)
    band_col = {0: (0.2, -0.1), 1: (0.0, 0.0), 2: (-0.1, 0.05)}  # (bulge, disc) colour offsets vs r
    band_psf = (1.08, 1.0, 0.95)
    shifter = bilinear_shift if shift == "bilinear" else fourier_shift
    for k0 in range(0, n, chunk):
        sl = slice(k0, min(n, k0 + chunk))
        nn_ = sl.stop - sl.start
        rot = np.stack([np.stack([np.cos(th[sl]), -np.sin(th[sl])], 1),
                        np.stack([np.sin(th[sl]), np.cos(th[sl])], 1)], 1)

        def cov(r_, q_, sp):
            d = np.zeros((nn_, 2, 2))
            d[:, 0, 0], d[:, 1, 1] = r_ ** 2, (q_ * r_) ** 2
            return rot @ d @ rot.transpose(0, 2, 1) + (sp ** 2 + 1 / 12)[:, None, None] * np.eye(2)

        for b in range(3):
            col = (gr[sl] if b == 0 else 0) - (ri[sl] if b == 2 else 0)  # m_b − m_r
            fb = fr[sl] * 10 ** (-0.4 * col)
            sp = psf[sl] * band_psf[b]
            sy, sx = s_all[sl, b, 0], s_all[sl, b, 1]
            img = (_gauss(yy, xx, c - sy, c - sx, cov(0.35 * rad[sl], np.sqrt(q[sl]), sp),
                          fb * bf[sl] * 10 ** (-0.4 * band_col[b][0]))
                   + _gauss(yy, xx, c - sy, c - sx, cov(rad[sl], q[sl], sp),
                            fb * (1 - bf[sl]) * 10 ** (-0.4 * band_col[b][1])))
            img += rng.normal(0, 1, img.shape) + ped[sl, None, None]
            if shift == "bilinear":
                for j in range(nn_):
                    st[k0 + j, b] = shifter(img[j], sy[j], sx[j])[m:m + size, m:m + size]
            else:  # fourier_shift, batched: the same phase ramp per image
                ky = np.fft.fftfreq(p)[None, :, None]
                kx = np.fft.fftfreq(p)[None, None, :]
                ramp = np.exp(-2j * np.pi * (ky * sy[:, None, None] + kx * sx[:, None, None]))
                st[sl, b] = np.fft.ifft2(np.fft.fft2(img) * ramp).real[:, m:m + size, m:m + size]
        if inject:
            for j in range(nn_):
                if inject_camcol is None or camcol[k0 + j] == inject_camcol:
                    st[k0 + j] = reinject(st[k0 + j].astype(np.float64), -s_all[k0 + j], inject)
    sig = np.sqrt(4 * np.pi * (rad ** 2 * (1 - bf) + (0.35 * rad) ** 2 * bf + psf ** 2))
    tb = {"mag": mag + rng.normal(0, 0.02, n), "size": 1.5 * rad * np.exp(rng.normal(0, 0.05, n)),
          "snr": fr / sig * (1 + rng.normal(0, 0.03, n)), "gr": gr + rng.normal(0, 0.03, n),
          "ri": ri + rng.normal(0, 0.03, n), "zp": z + rng.normal(0, 0.01, n),
          "psf": psf + rng.normal(0, 0.02, n), "sky": ped + rng.normal(0, 0.05, n), "camcol": camcol,
          "fold": rng.permutation(n) % N_FOLDS}
    tb["sb"] = tb["mag"] + 5 * np.log10(tb["size"])
    for c in range(1, 7):
        tb[f"camcol_{c}"] = (camcol == c).astype(float)
    for b, band in enumerate(BANDS):
        tb[f"{band}_sy"], tb[f"{band}_sx"] = s_all[:, b, 0], s_all[:, b, 1]
    o = -s_all  # v1's in-stamp offset from the centre: the grid snap v2 undoes
    tb["gr_v1x"], tb["gr_v1y"] = o[:, 0, 1] - o[:, 1, 1], o[:, 0, 0] - o[:, 1, 0]
    tb["ir_v1x"], tb["ir_v1y"] = o[:, 2, 1] - o[:, 1, 1], o[:, 2, 0] - o[:, 1, 0]
    return st, tb


# Sky is in the physics baseline, on principle: an observing condition, like PSF width (user,
# 2026-09-28). The plan's literal baseline (no sky; sky a condition) is scored beside it and reported.
SYN_PHYS_NOSKY = ["mag", "size", "snr", "gr", "ri", "sb", "zp", "psf"]
SYN_PHYS, SYN_COND = SYN_PHYS_NOSKY + ["sky"], []
SYN_ALT = (SYN_PHYS_NOSKY, ["sky"])
CAMCOL_1HOT = tuple(f"camcol_{c}" for c in range(1, 7))
SYN_REPORTED = ("psf", "sky") + CAMCOL_1HOT


def _syn_pair(n, seed, **kw) -> dict:
    kb = {k[2:]: v for k, v in kw.items() if k.startswith("B_")}
    ka = {k[2:]: v for k, v in kw.items() if k.startswith("A_")}
    both = {k: v for k, v in kw.items() if not k[:2] in ("A_", "B_")}
    out = {}
    for name, extra, sd in (("A", ka, seed), ("B", kb, seed + 1_000)):
        st, tb = synth(n, sd, name, **{**both, **extra})
        out[name] = {"st": st, "table": tb}
    return out


# ── real held-aside plant: v1 probe stamps (not probe_v2, not pretrain_v2) ───────────────────────────

REAL_PHYS = ["modelMag_r", "snr_r", "petroRad_r", "sb_r", "psfWidth_r", "specz"]


def real_plant(n: int = 4_000, seed: int = SEED) -> dict:
    """v1 probe stamps, the first n by assignment_unit(objID, 0, 'leakage-plant') with finite physics.
    g is Fourier-shifted by t_g (must read LEAK); t_r and t_i are drawn and not injected (must read
    CLEAN); phys_mix = z(mag) + z(log size) + N(0, 1) is readable from pixels and absorbed by the
    baseline (must read CLEAN, PHYSICS-EXPLAINED)."""
    import pandas as pd
    from astropy.io import fits

    from galaxy_jepa.data.splits import assignment_unit
    md = pd.read_csv(REPO / "data" / "probe" / "metadata.csv", usecols=["object_id", "camcol"] + [
        c for c in REAL_PHYS if c != "sb_r"], dtype={"object_id": str})
    md = md[np.isfinite(md[["modelMag_r", "snr_r", "petroRad_r", "psfWidth_r"]]).all(1)]
    md["u"] = [assignment_unit(int(o), SEED, salt="leakage-plant") for o in md.object_id]
    md = md.nsmallest(n, "u").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    t = rng.uniform(-0.5, 0.5, (n, 3, 2))
    path = OUT / f"real_plant_{n}.f16"
    st = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(n, 3, 256, 256))
    for k, o in enumerate(md.object_id):
        with fits.open(REPO / "data" / "probe" / f"{o}.fits") as h:
            x = np.asarray(h[0].data, dtype=np.float64)
        st[k] = reinject(x, t[k], 1.0, bands=(0,))
    st.flush()
    tb = {c: md[c].to_numpy(np.float64) for c in REAL_PHYS if c != "sb_r"}
    tb["sb_r"] = tb["modelMag_r"] + 5 * np.log10(tb["petroRad_r"])
    tb["camcol"] = md.camcol.to_numpy()
    tb["fold"] = np.array([int(assignment_unit(int(o), SEED, salt="leakage-fold") * N_FOLDS) for o in md.object_id])
    for b, band in enumerate(BANDS):
        tb[f"t{band}_y"], tb[f"t{band}_x"] = t[:, b, 0], t[:, b, 1]
    z = lambda v: (v - v.mean()) / v.std()  # noqa: E731
    tb["phys_mix"] = z(tb["modelMag_r"]) + z(np.log(tb["petroRad_r"])) + rng.normal(0, 1, n)
    return {"R": {"st": np.load(path, mmap_mode="r"), "table": tb}}


REAL_STATED = ("tg_x", "tg_y", "tr_x", "tr_y", "ti_x", "ti_y", "phys_mix")


# ── identity criterion (coordinates only) ───────────────────────────────────────────────────────────

NEAR_DUP_ARCSEC, NEAR_REPORT_ARCSEC = 1.0, 3.0
FOOTPRINT_ARCSEC = 128 * 0.396  # half the 256-px stamp


def _unit(ra, dec):
    r, d = np.radians(ra), np.radians(dec)
    return np.column_stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])


def identity(pre_ids, pre_ra, pre_dec, pro_ids, pro_ra, pro_dec, pro_split) -> dict:
    """resolve_corpora (raises LeakError if the dedup failed); near-duplicates within 1″ (state) and
    1–3″ (reported); probe galaxies inside a pretrain stamp's footprint (reported, no state)."""
    from scipy.spatial import cKDTree

    from galaxy_jepa.data.orchestrate import resolve_corpora
    pre_ids, pro_ids = np.asarray(pre_ids, np.int64), np.asarray(pro_ids, np.int64)
    raw = len(np.intersect1d(pre_ids, pro_ids))
    keep = np.isin(pre_ids, np.fromiter(resolve_corpora(pre_ids.tolist(), pro_ids.tolist()), np.int64))
    tree = cKDTree(_unit(pre_ra[keep], pre_dec[keep]))
    pu = _unit(pro_ra, pro_dec)
    chord = lambda a: 2 * np.sin(np.radians(a / 3600) / 2)  # noqa: E731
    d, _ = tree.query(pu, k=1)
    sep = np.degrees(2 * np.arcsin(d / 2)) * 3600
    near = sep < NEAR_DUP_ARCSEC
    close = (sep >= NEAR_DUP_ARCSEC) & (sep < NEAR_REPORT_ARCSEC)
    cand = tree.query_ball_point(pu, chord(FOOTPRINT_ARCSEC * np.sqrt(2)))
    pra, pdec = pre_ra[keep], pre_dec[keep]
    inside = np.zeros(len(pro_ids), bool)
    for k, js in enumerate(cand):
        if js:
            js = np.asarray(js)
            dx = (pra[js] - pro_ra[k] + 180) % 360 - 180
            inside[k] = bool(np.any((np.abs(dx * np.cos(np.radians(pro_dec[k]))) * 3600 < FOOTPRINT_ARCSEC)
                                    & (np.abs(pdec[js] - pro_dec[k]) * 3600 < FOOTPRINT_ARCSEC)))
    by = {s: {"n": int((pro_split == s).sum()), "near_dup": int((near & (pro_split == s)).sum()),
              "close_1_3": int((close & (pro_split == s)).sum()),
              "in_footprint": int((inside & (pro_split == s)).sum())} for s in np.unique(pro_split)}
    return {"raw_shared_ids": raw, "removed_by_dedup": int((~keep).sum()), "per_split": by,
            "state": "NEAR-DUPLICATES" if near.any() else "CLEAN", "near_dup_total": int(near.sum())}


def identity_plants(seed: int = SEED) -> dict:
    """Synthetic catalogues in a 4 deg² patch: clean → CLEAN; 7 pretrain rows at 0.5″ from probe
    galaxies under new IDs → NEAR-DUPLICATES (7); 10 shared IDs → the raw guard raises LeakError and
    resolve_corpora removes exactly those 10."""
    from galaxy_jepa.data.splits import LeakError, assert_no_cross_corpus_leak
    rng = np.random.default_rng(seed)
    npro, npre = 5_000, 20_000
    pro_ids = np.arange(1, npro + 1, dtype=np.int64) + 10 ** 18
    pre_ids = np.arange(1, npre + 1, dtype=np.int64) + 2 * 10 ** 18
    pro_ra, pro_dec = rng.uniform(150, 152, npro), rng.uniform(0, 2, npro)
    pre_ra, pre_dec = rng.uniform(150, 152, npre), rng.uniform(0, 2, npre)
    # clean: push pretrain rows off any probe galaxy by ≥ 3″ so the clean catalogue is clean by construction
    from scipy.spatial import cKDTree
    d, _ = cKDTree(_unit(pro_ra, pro_dec)).query(_unit(pre_ra, pre_dec))
    ok = np.degrees(2 * np.arcsin(d / 2)) * 3600 >= 3.0
    pre_ids, pre_ra, pre_dec = pre_ids[ok], pre_ra[ok], pre_dec[ok]
    split = np.array(["train", "val", "test"])[rng.integers(0, 3, npro)]
    out = {"clean": identity(pre_ids, pre_ra, pre_dec, pro_ids, pro_ra, pro_dec, split)}
    k = rng.choice(npro, 7, replace=False)
    nd = (np.r_[pre_ids, 3 * 10 ** 18 + np.arange(7)], np.r_[pre_ra, pro_ra[k] + 0.5 / 3600 / np.cos(np.radians(pro_dec[k]))],
          np.r_[pre_dec, pro_dec[k]])
    out["near_dup"] = identity(*nd, pro_ids, pro_ra, pro_dec, split)
    j = rng.choice(npro, 10, replace=False)
    shared = (np.r_[pre_ids, pro_ids[j]], np.r_[pre_ra, pro_ra[j] + 0.1], np.r_[pre_dec, pro_dec[j]])
    try:
        assert_no_cross_corpus_leak(shared[0].tolist(), pro_ids.tolist())
        raised = False
    except LeakError:
        raised = True
    r = identity(*shared, pro_ids, pro_ra, pro_dec, split)
    out["shared_ids"] = {"raw_guard_raised": raised, "raw_shared_ids": r["raw_shared_ids"],
                         "removed_by_dedup": r["removed_by_dedup"], "state": r["state"]}
    out["fires"] = (out["clean"]["state"] == "CLEAN" and out["near_dup"]["state"] == "NEAR-DUPLICATES"
                    and out["near_dup"]["near_dup_total"] == 7 and raised and r["removed_by_dedup"] == 10)
    return out


# ── plants ──────────────────────────────────────────────────────────────────────────────────────────

def _shift_states(res: dict, corpora=("A", "B")) -> dict:
    return {f"{c}:{t}": res["blocks"][c][t]["state"] for c in corpora for t in STATED}


def _required(tag: str, res: dict, which: str = "primary") -> tuple[bool, str]:
    """`which`: "primary" (sky in the baseline; gates) or "alt" (the plan's baseline, read via `view`).
    Only S5 differs: the sky pedestal is physics under the primary, a condition under the plan's."""
    if which == "alt":
        res = view(res)
    b = res["blocks"]
    corpus = b["pooled"]["corpus"] if "pooled" in b else None
    sh = _shift_states(res) if tag.startswith("S") else {}
    if tag == "S0":
        ok = res["verdict"]["state"] == "CLEAN" and "PHYSICS-EXPLAINED" in corpus["label"]
        return ok, "every variable CLEAN; corpus PHYSICS-EXPLAINED"
    if tag == "S1":
        ok = all(s == "LEAK" for s in sh.values()) and corpus["state"] == "CLEAN" and not any(
            b[c]["_camcol"]["structured"] for c in "AB")
        return ok, "all 20 shift variables LEAK; each corpus UNSTRUCTURED; corpus membership CLEAN"
    if tag == "S2":
        ok = corpus["state"] == "LEAK" and "NOT CONDITIONS-EXPLAINED" in corpus["label"] and all(
            b["A"][t]["state"] == "CLEAN" for t in STATED)
        return ok, "corpus LEAK, NOT CONDITIONS-EXPLAINED; A's shift variables CLEAN"
    if tag == "S3":
        ok = all(s in ("LEAK", "TRACE") for s in sh.values())
        return ok, "every shift variable LEAK or TRACE (never CLEAN); the split is reported"
    if tag == "S3b":
        near = [f"{c}:{t}" for c in "AB" for t in STATED
                if 0.5 * FLOOR <= max(b[c][t][m]["excess"] for m in ("ridge", "cnn")) <= 2 * FLOOR]
        ok = all(s in ("LEAK", "TRACE") for s in sh.values()) and bool(near) and all(sh[v] != "CLEAN" for v in near)
        return ok, ("no shift variable CLEAN; at least one variable at the threshold (larger excess in "
                    "[ε/2, 2ε]), and every such variable LEAK or TRACE")
    if tag == "S4":
        ok = all(s in ("LEAK", "TRACE") for s in sh.values()) and all(
            b[c]["_camcol"]["where"] == ["camcol3"] for c in "AB")
        return ok, "every shift variable LEAK or TRACE; each corpus CAMCOL-STRUCTURED at camcol 3 only"
    if tag == "S5":
        shifts_clean = all(b[c][t]["state"] == "CLEAN" for c in "AB" for t in STATED)
        if which == "primary":  # v3 first required CLEAN; at AUC ≈ 1 a 1.3e-5 residue reads TRACE
            ok = corpus["state"] in ("CLEAN", "TRACE") and shifts_clean
            return ok, ("corpus not blocking, CLEAN or TRACE (the baseline's sky absorbs the pedestal); "
                        "shift variables CLEAN")
        ok = corpus["state"] == "LEAK" and "CONDITIONS-EXPLAINED" in corpus["label"] and shifts_clean
        return ok, "corpus LEAK, CONDITIONS-EXPLAINED; shift variables CLEAN"
    if tag == "R":
        r = b["R"]
        ok = (all(r[t]["state"] == "LEAK" for t in ("tg_x", "tg_y"))
              and all(r[t]["state"] == "CLEAN" for t in ("tr_x", "tr_y", "ti_x", "ti_y"))
              and r["phys_mix"]["state"] == "CLEAN" and "PHYSICS-EXPLAINED" in r["phys_mix"]["label"])
        return ok, "t_g LEAK; t_r, t_i CLEAN; phys_mix CLEAN, PHYSICS-EXPLAINED"
    raise KeyError(tag)


SYN_PLANTS = {
    "S0": {},                                          # clean control
    "S1": {"inject": 1.0},                             # the plan's plant: v1 offsets re-injected
    "S2": {"B_shift": "bilinear"},                     # corpus B resampled bilinearly (the pilot's fingerprint)
    "S3": {"inject": "threshold"},                     # at the floor (α* from `calibrate`: FAILED, see the doc)
    "S3b": {"inject": 0.01},                           # revised after S3 failed: the sweep point where the
                                                       # weakest variables sat at the floor
    "S4": {"inject": 1.0, "inject_camcol": 3},         # the leak lives in camcol 3 only
    "S5": {"B_pedestal": 0.3},                         # corpora differ by a sky pedestal (a condition)
}


def _syn_run(tag: str, n: int, seed: int, log=print) -> dict:
    kw = dict(SYN_PLANTS[tag])
    if kw.get("inject") == "threshold":
        cal = OUT / "calibrate.json"
        if not cal.exists():
            raise SystemExit("S3 needs the calibrated dose: run `calibrate` first")
        kw["inject"] = json.loads(cal.read_text())["alpha_star"]
    t0 = time.time()
    data = _syn_pair(n, seed, **kw)
    log(f"  synthesised 2 × {n} in {time.time() - t0:.0f}s")
    return run(data, SYN_PHYS, SYN_COND, reported=SYN_REPORTED, log=log, alt=SYN_ALT)


def _slim(res: dict) -> dict:
    return {"verdict": res["verdict"], "blocks": res["blocks"]}  # each scored r carries its "alt"


def plants(only: list[str], n: int, n_real: int) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "plants.json"
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    for tag in only:
        t0 = time.time()
        print(f"[{tag}]", flush=True)
        if tag == "I":
            r = identity_plants()
            done[tag] = {"required": "clean CLEAN; 7 near-dups NEAR-DUPLICATES; shared IDs raise LeakError, "
                                     "dedup removes 10", "fires": r["fires"], "result": r}
        elif tag == "R":
            res = run(real_plant(n_real), REAL_PHYS, [], stated=REAL_STATED, shifts=REAL_STATED[:6],
                      log=lambda s: print(s, flush=True))
            ok, req = _required(tag, res)
            done[tag] = {"required": req, "fires": ok, "n": n_real, **_slim(res)}
        else:
            seed = SEED + 100 * int(tag[1]) + (50 if tag == "S3b" else 0)
            res = _syn_run(tag, n, seed, log=lambda s: print(s, flush=True))
            ok, req = _required(tag, res)
            ok_alt, req_alt = _required(tag, res, "alt")
            done[tag] = {"required": req, "fires": ok, "n": n, **_slim(res),
                         "without_sky": {"required": req_alt, "fires": ok_alt, "verdict": res["verdict_alt"]}}
            if tag == "S3":
                done[tag]["alpha"] = json.loads((OUT / "calibrate.json").read_text())["alpha_star"]
        done[tag]["seconds"] = round(time.time() - t0)
        print(f"[{tag}] fires={done[tag]['fires']} ({done[tag]['seconds']}s)", flush=True)
        _summary(tag, done[tag])
        out_path.write_text(json.dumps(done, indent=1, default=float))
    return done


def _summary(tag: str, d: dict) -> None:
    if tag == "I":
        print(json.dumps(d["result"], indent=1, default=int)[:1500])
        return
    for blk, res in d["blocks"].items():
        for t, r in res.items():
            if t == "_camcol":
                if r["tested"]:
                    print(f"  {blk}: camcol {'STRUCTURED ' + ','.join(r['where']) if r['structured'] else 'UNSTRUCTURED'}"
                          f"  d̄ {np.round(r['d'], 4).tolist()}  lo {np.round(r['lo'], 4).tolist()}")
                continue
            if "state" not in r:
                print(f"  {blk}:{t} reported  " + "  ".join(f"{m} {v['metric']:.3f} (lo {v['lo']:.3f})"
                                                           for m, v in r["pixel"].items()))
                continue
            print(f"  {blk}:{t} {r['state']:<10} phys {r['physics']:.3f}  " + "  ".join(
                f"{m} pix {r[m]['pixel']:.3f} Δ {r[m]['excess']:+.4f} (lo {r[m]['lo']:+.4f}) {r[m]['state']}"
                for m in ("ridge", "cnn")) + ("  " + ",".join(r["label"]) if r["label"] else ""))
    print(f"  verdict: {d['verdict']['state']}")
    if "without_sky" in d:
        w = d["without_sky"]
        pooled = d["blocks"].get("pooled", {}).get("corpus", {}).get("alt")
        print(f"  without sky: verdict {w['verdict']['state']}, fires={w['fires']}"
              + (f"; corpus {pooled['state']} {','.join(pooled['label'])}" if pooled else ""))


def calibrate(alphas: list[float], n: int) -> dict:
    """The threshold plant's dose, on its own seed (never S3's). Per α, the median over the 20 shift
    variables of the larger predictor's excess; α* puts that median at FLOOR by a log–log line
    through the sweep (R² ∝ α² for a weak shift)."""
    out: dict = {"sweep": {}}
    for a in alphas:
        data = _syn_pair(n, SEED + 900, inject=a)
        res = run(data, SYN_PHYS, SYN_COND, reported=SYN_REPORTED, log=lambda s: print(s, flush=True), alt=SYN_ALT)
        ex = [max(res["blocks"][c][t][m]["excess"] for m in ("ridge", "cnn")) for c in "AB" for t in STATED]
        out["sweep"][str(a)] = {"median_max_excess": float(np.median(ex)), "min": float(np.min(ex)),
                                "max": float(np.max(ex)), "states": _shift_states(res),
                                "excess": {f"{c}:{t}": float(e) for (c, t), e in
                                           zip([(c, t) for c in "AB" for t in STATED], ex)}}
        print(f"alpha {a}: median max-predictor excess {np.median(ex):.4f} (range {np.min(ex):.4f}–{np.max(ex):.4f})",
              flush=True)
    xs = np.log([float(k) for k in out["sweep"]])
    ys = np.log([max(v["median_max_excess"], 1e-6) for v in out["sweep"].values()])
    slope, icpt = np.polyfit(xs, ys, 1)
    out["fit"] = {"slope": float(slope), "intercept": float(icpt)}
    out["alpha_star"] = float(np.exp((np.log(FLOOR) - icpt) / slope))
    print(f"alpha* = {out['alpha_star']:.4f} (log-log slope {slope:.2f})", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibrate.json").write_text(json.dumps(out, indent=1))
    return out


# ── the audit (locked) ──────────────────────────────────────────────────────────────────────────────

CORPORA = ("pretrain_v2", "probe_v2")
WORK = REPO / ".sciserver_work"
AUDIT_PHYS_NOSKY = ["modelMag_r", "snr_r", "petroR50_r", "gr", "ri", "sb50_r", "photoz", "psfWidth_r"]
AUDIT_PHYS, AUDIT_COND = AUDIT_PHYS_NOSKY + ["sky_r"], []  # sky in the baseline (user, 2026-09-28)
AUDIT_ALT = (AUDIT_PHYS_NOSKY, ["sky_r"])  # the plan's literal baseline, reported
AUDIT_REPORTED = ("psfWidth_r", "sky_r", "valid_frac", "edge_dist") + CAMCOL_1HOT
MAX_MISSING = 0.02  # the physics pull may miss at most 2% of the sample, else INSUFFICIENT
PHYS_BATCH = 100  # IDs per SkyServer call (see `_physics`)
PHYS_SQL = ("SELECT CAST(p.objID AS varchar(20)) AS objID, p.modelMag_g, p.modelMag_r, p.modelMag_i, "
            "p.modelMagErr_r, p.petroR50_r, p.sky_r, f.psfWidth_r, z.z AS photoz FROM PhotoObjAll p "
            "JOIN Field f ON f.fieldID = p.fieldID LEFT JOIN Photoz z ON z.objID = p.objID WHERE p.objID IN ({})")


def _complete(corpus: str) -> dict:
    """The pull is complete iff every target is either cut (one cut_log row) or in failed.csv."""
    import pandas as pd
    d = REPO / "data" / corpus
    n_t = len(pd.read_csv(WORK / f"{corpus}_all_targets.csv", usecols=[0]))
    n_c = len(pd.read_csv(d / "cut_log.csv", usecols=["object_id"])) if (d / "cut_log.csv").exists() else 0
    n_f = len(pd.read_csv(d / "failed.csv", usecols=[0])) if (d / "failed.csv").exists() else 0
    return {"targets": n_t, "cut": n_c, "failed": n_f, "complete": bool(n_c + n_f == n_t and (d / "manifest.json").exists())}


def _samples(logs: dict) -> dict:
    """The audit's sample from the cut logs alone (no pixels): pretrain_v2 deduplicated against probe_v2,
    then the N_PER_CORPUS smallest assignment_unit(objID, 0, 'leakage') per corpus."""
    from galaxy_jepa.data.orchestrate import resolve_corpora
    from galaxy_jepa.data.splits import assignment_unit
    pre, pro = logs["pretrain_v2"], logs["probe_v2"]
    keep = {str(i) for i in resolve_corpora(pre.object_id.astype(int).tolist(), pro.object_id.astype(int).tolist())}
    logs = {**logs, "pretrain_v2": pre[pre.object_id.isin(keep)]}
    samples = {}
    for c in CORPORA:
        lg = logs[c].copy()
        lg["u"] = [assignment_unit(int(o), SEED, salt="leakage") for o in lg.object_id]
        samples[c] = lg.nsmallest(N_PER_CORPUS, "u").reset_index(drop=True)
    return samples


PHYS_RECORD = OUT / "audit_physics.json"


def pull() -> dict:
    """Pull the baseline's catalogue physics for the audit's 40,000 sampled objects before the hash
    (user, 2026-09-28): public SkyServer, no token, no pixels read. Records the query and the output's
    SHA-1; `audit` refuses a physics file that no longer matches it."""
    import pandas as pd
    for c in CORPORA:
        if not _complete(c)["complete"]:
            raise SystemExit(f"leakage_audit pull: {c} is not complete")
    logs = {c: pd.read_csv(REPO / "data" / c / "cut_log.csv", dtype={"object_id": str},
                           usecols=["object_id"]) for c in CORPORA}
    samples = _samples(logs)
    ids = [o for c in CORPORA for o in samples[c].object_id]
    path = OUT / "audit_physics.csv"
    if PHYS_RECORD.exists():
        raise SystemExit(f"leakage_audit pull: {PHYS_RECORD} exists; the pull is made once and pinned")
    t0 = time.time()
    path.unlink(missing_ok=True)  # derived from the kept batches; rebuilt, never re-queried
    df = _physics(ids)
    df = df.sort_values("object_id").reset_index(drop=True)
    df.to_csv(path, index=False)
    df = pd.read_csv(path, dtype={"object_id": str})  # as `audit` reads it: objIDs as strings
    rec = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "service": "public SkyServer SQL (run_sql), no token",
           "query_template": PHYS_SQL, "batch": PHYS_BATCH, "ids_requested": len(ids), "rows_returned": len(df),
           "ids_missing": len(set(ids) - set(df.object_id)),
           "sample_sha1": {c: hashlib.sha1(",".join(samples[c].object_id).encode()).hexdigest() for c in CORPORA},
           "output": str(path.relative_to(REPO)), "output_sha1": hashlib.sha1(path.read_bytes()).hexdigest(),
           "missing_any_baseline_input": {c: float(samples[c][["object_id"]].merge(df, on="object_id", how="left")
                                                   [AUDIT_PHYS].isna().any(axis=1).mean()) for c in CORPORA},
           "seconds": round(time.time() - t0)}
    PHYS_RECORD.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
    return rec


def _physics(ids: list[str]):
    """Public SkyServer SQL (no token), cached. snr_r through the package's one derivation site."""
    import pandas as pd
    from sciserver_pull import _net_retry

    from galaxy_jepa.data.metadata import run_sql
    from galaxy_jepa.data.pull import with_derived_columns
    path = OUT / "audit_physics.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"object_id": str})
    # SkyServer drops connections intermittently (2026-10-01: 400-ID calls failed to connect while
    # 100-ID calls went through), so batches are small and each is kept as it lands: a rerun resumes.
    part = OUT / "audit_physics.partial.json"
    got = json.loads(part.read_text()) if part.exists() else {}
    for k in range(0, len(ids), PHYS_BATCH):
        if str(k) in got:
            continue
        got[str(k)] = _net_retry(run_sql, PHYS_SQL.format(",".join(ids[k:k + PHYS_BATCH])), timeout=300, _tries=12)
        part.write_text(json.dumps(got))
        time.sleep(1)
    rows = [r for k in sorted(got, key=int) for r in got[k]]
    rows = with_derived_columns(rows)
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c != "object_id" and c != "objID":
            df[c] = pd.to_numeric(df[c], errors="coerce").where(lambda v: v > -9000)  # SDSS −9999 = missing
    df["gr"], df["ri"] = df.modelMag_g - df.modelMag_r, df.modelMag_r - df.modelMag_i
    df["sb50_r"] = df.modelMag_r + 5 * np.log10(df.petroR50_r)
    df.to_csv(path, index=False)
    return df


def _targets(log) -> dict:
    """Stated and reported targets from cut_log. frac(origin) ≡ s_b by the cutter's construction
    (s_b = 159.5 − (x_b − o_b)); asserted, so a cutter change cannot silently split them."""
    tb = {}
    for b in BANDS:
        for a, pos, org in (("x", f"{b}_x", f"{b}_ox"), ("y", f"{b}_y", f"{b}_oy")):
            s_ = log[f"{b}_s{a}"].to_numpy(float)
            if not np.allclose(s_, 159.5 - (log[pos].to_numpy(float) - log[org].to_numpy(float)), atol=1e-6):
                raise SystemExit(f"leakage_audit: s_b ≠ 159.5 − frac(origin) for {b}{a}; the cutter changed")
            tb[f"{b}_s{a}"] = s_
    rel = {b: (log[f"{b}_v1_relx"].to_numpy(float), log[f"{b}_v1_rely"].to_numpy(float)) for b in BANDS}
    tb["gr_v1x"], tb["gr_v1y"] = rel["g"][0] - rel["r"][0], rel["g"][1] - rel["r"][1]
    tb["ir_v1x"], tb["ir_v1y"] = rel["i"][0] - rel["r"][0], rel["i"][1] - rel["r"][1]
    tb["valid_frac"] = log.valid_frac.to_numpy(float)
    tb["edge_dist"] = np.min([log[f"{b}_edge_dist"].to_numpy(float) for b in BANDS], 0)
    tb["camcol"] = log.camcol.to_numpy(int)
    for c in range(1, 7):
        tb[f"camcol_{c}"] = (tb["camcol"] == c).astype(float)
    tb["_v1_off"] = np.stack([np.stack([rel[b][1] - 127.5, rel[b][0] - 127.5], 1) for b in BANDS], 1)
    return tb


def _stamps(corpus: str, ids: list[str], tag: str, off: np.ndarray | None = None) -> np.ndarray:
    from astropy.io import fits
    path = OUT / f"audit_{corpus}{tag}.f16"
    st = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(ids), 3, 256, 256))
    for k, o in enumerate(ids):
        with fits.open(REPO / "data" / corpus / f"{o}.fits") as h:
            x = np.asarray(h[0].data, dtype=np.float64)
        st[k] = x if off is None else reinject(x, off[k])
    st.flush()
    return np.load(path, mmap_mode="r")


def audit() -> None:
    """Refuses until the pre-registration is hashed. Then, in order: completeness (INSUFFICIENT),
    identity, sample, physics pull (INSUFFICIENT), the audit, the re-injected plant (INVALID if it
    does not read LEAK on every shift variable)."""
    assert_hashed()
    import pandas as pd

    from galaxy_jepa.data.orchestrate import assign_three_way
    from galaxy_jepa.data.splits import assignment_unit
    OUT.mkdir(parents=True, exist_ok=True)
    rep: dict = {"prereg_sha1": PREREG_SHA1, "code_sha1": hashlib.sha1(Path(__file__).read_bytes()).hexdigest()}

    def done(state: str) -> None:
        rep["state"] = state
        (OUT / "audit.json").write_text(json.dumps(rep, indent=1, default=float))
        print(json.dumps({"state": state}, indent=1))
        raise SystemExit(0)

    rep["completeness"] = {c: _complete(c) for c in CORPORA}
    if not all(v["complete"] for v in rep["completeness"].values()):
        done("INSUFFICIENT")
    logs = {c: pd.read_csv(REPO / "data" / c / "cut_log.csv", dtype={"object_id": str}) for c in CORPORA}
    pre, pro = logs["pretrain_v2"], logs["probe_v2"]
    sp = assign_three_way(pro.object_id.astype(int).tolist(), seed=SEED)
    split = np.where(pro.object_id.astype(int).isin(sp.train), "train",
                     np.where(pro.object_id.astype(int).isin(sp.val), "val", "test"))
    rep["identity"] = identity(pre.object_id.astype(np.int64), pre.ra.to_numpy(float), pre.dec.to_numpy(float),
                               pro.object_id.astype(np.int64), pro.ra.to_numpy(float), pro.dec.to_numpy(float), split)
    samples = _samples(logs)
    corpora = {}
    pinned = json.loads(PHYS_RECORD.read_text()) if PHYS_RECORD.exists() else None
    got = hashlib.sha1((OUT / "audit_physics.csv").read_bytes()).hexdigest() if pinned else None
    if pinned is None or got != pinned["output_sha1"]:
        raise SystemExit("leakage_audit: the physics pull is missing or differs from its pinned record "
                         f"({got} vs {pinned and pinned['output_sha1']}); run `pull` once, before the hash")
    phys = _physics([o for c in CORPORA for o in samples[c].object_id])
    for c in CORPORA:
        lg = samples[c].merge(phys, on="object_id", how="left")
        miss = float(lg[AUDIT_PHYS].isna().any(axis=1).mean())
        rep.setdefault("physics_missing", {})[c] = miss
        if miss > MAX_MISSING:
            done("INSUFFICIENT")
        tb = _targets(lg)
        tb.update({k: lg[k].to_numpy(float) for k in AUDIT_PHYS + AUDIT_COND})
        tb["fold"] = np.array([int(assignment_unit(int(o), SEED, salt="leakage-fold") * N_FOLDS) for o in lg.object_id])
        for t in STATED:
            if np.isfinite(tb[t]).sum() < MIN_ROWS:
                done("INSUFFICIENT")
        corpora[c] = {"ids": lg.object_id.tolist(), "table": tb}
        rep.setdefault("sample_sha1", {})[c] = hashlib.sha1(",".join(corpora[c]["ids"]).encode()).hexdigest()
    for c in CORPORA:
        corpora[c]["st"] = _stamps(c, corpora[c]["ids"], "")
    res = run(corpora, AUDIT_PHYS, AUDIT_COND, reported=AUDIT_REPORTED, log=lambda s: print(s, flush=True),
              alt=AUDIT_ALT)
    rep["audit"] = res
    for c in CORPORA:
        corpora[c]["st"] = _stamps(c, corpora[c]["ids"], "_reinjected", corpora[c]["table"]["_v1_off"])
    plant = run(corpora, AUDIT_PHYS, AUDIT_COND, reported=AUDIT_REPORTED, log=lambda s: print(s, flush=True),
              alt=AUDIT_ALT)
    rep["plant"] = plant
    fired = all(plant["blocks"][c][t]["state"] == "LEAK" for c in CORPORA for t in STATED)
    rep["plant_fired"] = fired
    done(res["verdict"]["state"] if fired else "INVALID")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plants")
    p.add_argument("--only", default="I,R,S0,S1,S2,S3,S4,S5")
    p.add_argument("--n", type=int, default=N_PER_CORPUS)
    p.add_argument("--n-real", type=int, default=4_000)
    c = sub.add_parser("calibrate")
    c.add_argument("--alphas", default=",".join(map(str, CALIB_ALPHAS)))
    c.add_argument("--n", type=int, default=N_PER_CORPUS)
    sub.add_parser("pull")
    sub.add_parser("audit")
    a = ap.parse_args()
    if a.cmd == "plants":
        plants(a.only.split(","), a.n, a.n_real)
    elif a.cmd == "calibrate":
        calibrate([float(x) for x in a.alphas.split(",")], a.n)
    elif a.cmd == "pull":
        pull()
    else:
        audit()


if __name__ == "__main__":
    main()
