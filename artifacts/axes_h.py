"""Part H1 + H2a (kickoff, 24 Nov 2026): the mean against the tokens, and where features form across blocks. A1 and the three untrained seeds.

    .venv/bin/python artifacts/axes_h.py plan                  # disk and memory, printed before anything runs
    .venv/bin/python artifacts/axes_h.py selftest              # planted token-only feature, null, mean-only feature through the same trainer
    .venv/bin/python artifacts/axes_h.py targets               # rows and labels for every target → runs/almagest/axes_phys/h/targets.npz
    .venv/bin/python artifacts/axes_h.py extract --enc A1      # one forward pass: 12 block means, block-11 4×4 grids and tokens, 6k-subset token grids
    .venv/bin/python artifacts/axes_h.py h1 --enc A1           # readouts (a′) mean, (b) 4×4 grid, (c) attention pooling
    .venv/bin/python artifacts/axes_h.py h2a --enc A1          # mean probes at every block
    .venv/bin/python artifacts/axes_h.py report                # docs/axes/h1.json, h2a.json

Fixed before any H number was computed (descriptive, not hashed):
- The readout block. The current readout is the mean of block 11's tokens (the encoder's pinned penultimate layer, un-normed). H1's three readouts all use block 11, so
  (a) IS the current readout; block 12 (the last) appears in H2a. The kickoff says "the final block's tokens": a reading of it, stated here.
- Rows: the union (39,997 train / 34,828 test) and the 1,245 EFIGI galaxies outside it. Raw embeddings (the nuisance cleaner is defined on the pooled vector, not on tokens).
  A result is read against the untrained seeds, which carry the same nuisances. Binary rows are d.rows (the 37-axis run's); ordinal rows are Part A's; EFIGI rows Part G's.
- Validation = 15% of the train galaxies, one fixed draw, never test; it picks epoch and L2 for every readout, encoder and target by the same trainer (Adam, minibatch, joint over targets
  but with independent parameters and losses; L2 grid on the standardised features; best epoch kept per output).
- (a) the ladder's fixed-C logistic probe (binary targets; its own code); (a′) a linear probe on the block-11 mean through this trainer, the paired baseline for (b) and (c): same selection, same
  data; (b) linear on 16 regions × 384 with strong L2; (c) attention pooling: one query per target, softmax over 256 tokens, pooled vector → linear; 769 parameters; Q = 0 at the start (exactly mean
  pooling). Ordinal targets: cumulative-logit loss, per-vote weights; score η.
- Shuffled control: every target once more with its labels permuted among its eligible train rows, for A1 and untrained-s0, all readouts.
- Gain = (metric of b or c) − (metric of a′); metric = test AUC (binary) or Kendall τ and the LL gap (ordinal). A1's gain counts only beside the untrained gains: paired bootstrap (200) of
  gain_A1 − mean(gain_untrained), its 95% CI, and the range over seeds. Deciles of resolution as before (union test edges).
- H2a: per block, per encoder, the logistic probe by Newton (the ladder's C) on the standardised mean; ordinal and EFIGI targets by the same cumulative logit, λ picked on the same 15% split. Emergence = first
  block whose A1 metric clears the best untrained by 0.05 on the bootstrap CI (lower end of (A1 − max untrained) > 0.05). Peak = argmax.

Investigation code: terse, excluded from lint/CI. Large outputs in runs/almagest/axes_phys/h/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import axes_a_ordinal as A

H = A.OUT / "h"
ENC = ("A1", "untrained-s0", "untrained-s1", "untrained-s2")
SEED = A.SEED + 120
VAL_FRAC = 0.15
BLOCK = 11  # 1-based: the readout block
SUB_BLOCKS = (4, 8, 10, 12)
N_SUB = 6000
WD = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
EFIGI = ("armC", "armS", "B/T", "floc", "inc", "Pert", "vDust")


# --- plan -------------------------------------------------------------------------------------------------------------------
def step_plan():
    n = 74_825 + 1_245
    b = {"block means 12×384 fp32 per encoder": n * 12 * 384 * 4, "block-11 4×4 grid fp16 per encoder": n * 16 * 384 * 2,
         "block-11 tokens fp16 (scratch, one encoder at a time, overwritten)": n * 256 * 384 * 2, "6k subset tokens, blocks 4/8/10/12 fp16 per encoder": N_SUB * 4 * 256 * 384 * 2}
    for k, v in b.items():
        print(f"{k:72s} {v / 1e9:6.2f} GB")
    tot = 4 * (b["block means 12×384 fp32 per encoder"] + b["block-11 4×4 grid fp16 per encoder"] + b["6k subset tokens, blocks 4/8/10/12 fp16 per encoder"]) + b["block-11 tokens fp16 (scratch, one encoder at a time, overwritten)"]
    print(f"{'total on disk, all four encoders':72s} {tot / 1e9:6.2f} GB (kept after the run: {(tot - 0) / 1e9:.1f}; the scratch file is the 14.7 GB one)")
    print("memory: forward ≈ 1.0 GB (encoder + 64-image batch + 12 kept blocks); trainer ≈ 1.5 GB; one encoder per process; no process holds two token sets")
    os.system("df -h '/Volumes/X10 Pro' | tail -1; sysctl -n vm.swapusage; uptime; memory_pressure 2>/dev/null | tail -1")


# --- targets ----------------------------------------------------------------------------------------------------------------
def build_targets():
    """Every target, as rows into ids (= union ∪ EFIGI extras). Saved to targets.npz (arrays only)."""
    AXS, d = A.load()
    import axes_g as G
    order = json.loads((AXS.DOCS / "a1_axes.json").read_text())
    recs = order["answers"]
    keep = [f for f in order["order"] if recs[f]["label"] in ("STRONG", "SEPARATE")]
    cat = G.vizier("efigi.tsv")
    eids, esrc, etr, ete, edec = G.efigi_rows(d)
    extra = np.array(sorted(int(i) for i in eids if int(i) not in d.row), dtype=np.int64)
    ids = np.concatenate([d.union.astype(np.int64), extra])
    pos = {int(o): k for k, o in enumerate(ids)}
    rng = np.random.default_rng(SEED)
    size = np.asarray(d.lab.nuisance_value("size", ids), float)
    psf = np.asarray(d.lab.nuisance_value("psf", ids), float)
    res = size / psf
    ed = G.edges(d)
    dec = np.clip(np.searchsorted(ed, res, side="right") - 1, 0, 9)
    part = {int(i): p for i, p in zip(d.union, np.where(np.isin(d.union, np.asarray(d.setup.train_ids)), "train", "test"))}
    out = {"ids": ids, "dec": dec, "res": res, "is_test": np.array([part.get(int(i), None) == "test" if int(i) in part else False for i in ids]),
           "is_train": np.array([part.get(int(i), None) == "train" if int(i) in part else False for i in ids])}
    # EFIGI rows outside the union carry their own partition
    for i, tr, te in zip(eids, etr, ete):
        k = pos[int(i)]
        out["is_train"][k], out["is_test"][k] = bool(tr), bool(te)
    val = np.zeros(len(ids), bool)
    tr_rows = np.flatnonzero(out["is_train"])
    val[rng.choice(tr_rows, int(round(VAL_FRAC * len(tr_rows))), replace=False)] = True
    out["is_val"] = val
    meta = []
    for f in keep:
        tr, te = d.rows(f)
        y, v = d.yv(f, np.concatenate([tr, te]))
        meta.append({"name": f, "kind": "binary", "label": recs[f]["label"]})
        out[f"{f}|rows"] = np.concatenate([tr, te])
        out[f"{f}|y"] = y
        out[f"{f}|tr"] = np.concatenate([np.ones(len(tr), bool), np.zeros(len(te), bool)])
    for q, Q in A.QUESTIONS.items():
        r = A.question_rows(AXS, d, q)
        meta.append({"name": q, "kind": "ordinal", "label": "Part A"})
        out[f"{q}|rows"] = r["rows"]
        out[f"{q}|p"], out[f"{q}|n"] = r["p"], r["n"]
        out[f"{q}|tr"] = r["tr"]
    for k in EFIGI:
        v = G.vals(cat, k)[esrc]
        ok = np.isfinite(v) & np.isfinite(np.column_stack([d.lab.nuisance_value(n, eids) for n in AXS.S.NUIS])).all(1)
        P, cats, merged = G.onehot(v[ok], etr[ok])
        meta.append({"name": f"efigi:{k}", "kind": "efigi", "label": f"categories {cats}"})
        out[f"efigi:{k}|rows"] = np.array([pos[int(i)] for i in eids[ok]])
        out[f"efigi:{k}|p"], out[f"efigi:{k}|n"] = P, np.ones(len(P))
        out[f"efigi:{k}|raw_y"] = v[ok].astype(np.float64)
        out[f"efigi:{k}|tr"] = etr[ok]
    sub_pool = np.flatnonzero(np.isin(d.union, np.asarray(d.setup.test_ids)) & d.ok)
    sub = np.concatenate([rng.choice(sub_pool[dec[sub_pool] == k], N_SUB // 10, replace=False) for k in range(10)])
    out["subset"] = np.sort(sub)
    H.mkdir(parents=True, exist_ok=True)
    (H / "probe_c.json").write_text(json.dumps({"c": float(d.c)}))
    # Cross-check damped Newton against the ladder's Probe on well-balanced targets AND the rarest
    # binary target (imbalanced classes previously diverged under undamped Newton).
    rarest = min(keep, key=lambda f: min(out[f"{f}|y"][out[f"{f}|tr"]].mean(), 1.0 - out[f"{f}|y"][out[f"{f}|tr"]].mean()))
    for f in dict.fromkeys([*keep[:2], rarest]):
        rows, tr = out[f"{f}|rows"], out[f"{f}|tr"]
        X, y = d.mats["A1"][rows], out[f"{f}|y"]
        ref = AXS.Probe(X[tr], y[tr], d.c).d(X[~tr])
        mu, sd = X[tr].mean(0), np.maximum(X[tr].std(0), 1e-9)
        w, b = newton_logistic((X[tr] - mu) / sd, y[tr].astype(float), float(d.c))
        new = (X[~tr] - mu) / sd @ w + b
        print(f, "Newton vs Probe AUC", AXS.fast_auc(y[~tr], new), AXS.fast_auc(y[~tr], ref), "r", np.corrcoef(new, ref)[0, 1])
        assert abs(AXS.fast_auc(y[~tr], new) - AXS.fast_auc(y[~tr], ref)) < 1e-3
    np.savez(H / "targets.npz", **out, meta=json.dumps(meta))
    print(json.dumps({"n_ids": len(ids), "extras": len(extra), "val": int(val.sum()), "targets": {m["name"]: [int(out[m["name"] + "|tr"].sum()), int((~out[m["name"] + "|tr"]).sum())] for m in meta}}, indent=1))


# --- extraction ---------------------------------------------------------------------------------------------------------------
def step_extract(enc):
    import torch
    z = np.load(H / "targets.npz", allow_pickle=False)
    ids = z["ids"]
    AXS, d = A.load()
    import j4_spread_controls as J
    from galaxy_jepa.models.vit import VisionTransformer
    st = d.setup
    fz = AXS.load_frozen_encoder(st.ckpt)
    if enc == "A1":
        model = fz
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(enc[-1]))
            model = VisionTransformer(**dict(fz.config))
    model.to(st.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ds = type(st.ds)(J.check(verbose=False)[1], st.rows, [int(i) for i in ids])
    assert np.array_equal(np.asarray(ds.object_ids, dtype=np.int64), ids), "dataset order"
    n = len(ids)
    layers = np.lib.format.open_memmap(H / f"layers_{enc}.npy", mode="w+", dtype=np.float32, shape=(n, 12, 384))
    grid = np.lib.format.open_memmap(H / f"grid_{enc}.npy", mode="w+", dtype=np.float16, shape=(n, 16, 384))
    tok = np.lib.format.open_memmap(H / "tokens_scratch.npy", mode="w+", dtype=np.float16, shape=(n, 256, 384))
    (H / "tokens_scratch.owner").write_text(enc)
    subset = z["subset"]
    spos = {int(k): j for j, k in enumerate(subset)}
    sub = np.lib.format.open_memmap(H / f"subset_{enc}.npy", mode="w+", dtype=np.float16, shape=(len(subset), 4, 256, 384))
    loader = torch.utils.data.DataLoader(ds, batch_size=64)
    t0, i0 = time.perf_counter(), 0
    with torch.inference_mode():
        for b in loader:
            x = b["image"].float().to(st.device)
            outs = model.layer_tokens(x)
            k = len(x)
            layers[i0:i0 + k] = torch.stack([o.mean(1) for o in outs], 1).float().cpu().numpy()
            t = outs[BLOCK - 1]
            g = t.reshape(k, 4, 4, 4, 4, 384).mean((2, 4)).reshape(k, 16, 384)
            grid[i0:i0 + k] = g.half().cpu().numpy()
            tok[i0:i0 + k] = t.half().cpu().numpy()
            hit = [(j, spos[i0 + j]) for j in range(k) if i0 + j in spos]
            for j, s in hit:
                sub[s] = torch.stack([outs[L - 1][j] for L in SUB_BLOCKS]).half().cpu().numpy()
            i0 += k
            if i0 % 6400 < 64:
                el = time.perf_counter() - t0
                A.log(f"extract {enc} {i0}/{n} {i0 / el:.0f} img/s")
    for m in (layers, grid, tok, sub):
        m.flush()
    own = layers[: len(d.union), BLOCK - 1]
    chk = float(np.abs(own - d.mats[enc]).max())
    print(f"{enc}: block-{BLOCK} mean vs the banked readout, max|Δ| {chk:.2e}")
    (H / f"extract_{enc}.json").write_text(json.dumps({"max_abs_diff_vs_bank": chk, "n": n, "seconds": time.perf_counter() - t0}))
    assert chk < 2e-2, "block 11 does not reproduce the readout"


# --- trainer ------------------------------------------------------------------------------------------------------------------
class Target:
    """One target: row indices (into the feature array), train/val/test masks and labels."""
    def __init__(self, z, name, kind):
        self.name, self.kind = name, kind
        rows, tr = z[f"{name}|rows"], z[f"{name}|tr"]
        self.rows, self.te = rows, ~tr
        val = z["is_val"][rows] & tr
        self.fit, self.val = tr & ~val, val
        if kind == "binary":
            self.y = z[f"{name}|y"].astype(np.float32)
            self.w = np.ones(len(rows))
        else:
            self.p, self.n = z[f"{name}|p"], z[f"{name}|n"]
            self.K = self.p.shape[1]
            self.raw_y = z[f"{name}|raw_y"] if f"{name}|raw_y" in z.files else None


def shuffled(t: Target, rng):
    """A copy with labels permuted among its eligible train rows (fit + val), test untouched by construction (scored with no shuffle)."""
    import copy
    s = copy.copy(t)
    m = np.flatnonzero(t.fit | t.val)
    perm = rng.permutation(m)
    if t.kind == "binary":
        s.y = t.y.copy()
        s.y[m] = t.y[perm]
    else:
        s.p, s.n = t.p.copy(), t.n.copy()
        s.p[m], s.n[m] = t.p[perm], t.n[perm]
    return s


def train(readout, feats, targets, seed, epochs, batch, lr, n_rows, device, tok=None):
    """Joint training of every target × WD grid: independent parameters and losses, best epoch per output on val. feats: (N,F) float32 array, or tokens (N,256,384) memmap for
    `attn`. Returns per target: eta on test (best grid), chosen wd, val loss, thresholds."""
    import torch
    torch.manual_seed(seed)
    T, G = len(targets), len(WD)
    M = T * G
    F = feats.shape[-1]
    attn = readout == "attn"
    W = torch.zeros(M, F, device=device, requires_grad=True)
    b = torch.zeros(M, device=device, requires_grad=True)
    Q = torch.zeros(M, F, device=device, requires_grad=True) if attn else None
    ths, kof = {}, {}
    params = [W, b] + ([Q] if attn else [])
    wdv = torch.tensor(np.tile(np.array(WD), T), dtype=torch.float32, device=device)
    for ti, t in enumerate(targets):
        if t.kind != "binary":
            m = t.fit | t.val
            q = np.clip((t.n[m][:, None] * t.p[m]).sum(0), 1e-9, None)
            c = np.cumsum(q)[:-1] / q.sum()
            th0 = np.log(c / (1 - c))
            raw = np.concatenate([[th0[0]], np.log(np.maximum(np.diff(th0), 1e-3))])
            ths[ti] = torch.tensor(np.tile(raw, (G, 1)), dtype=torch.float32, device=device, requires_grad=True)
            params.append(ths[ti])
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=lr * 0.01)
    # standardise features on fit rows (tokens: per feature over a sample of fit rows' tokens)
    used = np.unique(np.concatenate([t.rows[t.fit] for t in targets]))
    rng = np.random.default_rng(seed)
    if attn:
        smp = np.sort(rng.choice(used, min(1500, len(used)), replace=False))
        s = np.asarray(feats[smp], dtype=np.float32)
        mu, sd = s.reshape(-1, F).mean(0), s.reshape(-1, F).std(0)
    else:
        mu, sd = feats[used].astype(np.float64).mean(0), feats[used].astype(np.float64).std(0)
    sd = np.maximum(sd, 1e-6)
    mut, sdt = torch.tensor(mu, dtype=torch.float32, device=device), torch.tensor(sd, dtype=torch.float32, device=device)
    lu = {}
    L = len(feats)
    # label tables on the full row index: per target dense arrays (−1 = not eligible)
    tab = []
    for t in targets:
        idx = np.full(L, -1, np.int64)
        idx[t.rows] = np.arange(len(t.rows))
        tab.append(idx)
    tot_fit = [float(t.fit.sum()) if t.kind == "binary" else float(t.n[t.fit].sum()) for t in targets]

    def forward(Xb):
        Xb = (Xb - mut) / sdt
        if attn:
            s = torch.einsum("bnf,mf->bnm", Xb, Q)
            a = torch.softmax(s, 1)
            return torch.einsum("bnm,bnf,mf->bm", a, Xb, W) + b
        return Xb @ W.T + b

    def losses(eta, rows, which):
        """Σ over outputs of the weighted data loss on the rows in `which` ∈ {'fit','val'}; per-output values too (for val)."""
        per = []
        for ti, t in enumerate(targets):
            i = tab[ti][rows]
            mask = t.fit if which == "fit" else t.val
            sel = np.flatnonzero((i >= 0) & np.where(i >= 0, mask[np.maximum(i, 0)], False))
            if not len(sel):
                per.append(torch.zeros(G, device=device)); continue
            k = i[sel]
            e = eta[torch.as_tensor(sel, device=device)][:, ti * G:(ti + 1) * G]
            if t.kind == "binary":
                y = torch.as_tensor(t.y[k], device=device)[:, None].expand_as(e)
                l = torch.nn.functional.binary_cross_entropy_with_logits(e, y, reduction="none").sum(0)
            else:
                w = torch.as_tensor(t.n[k].astype(np.float32), device=device)
                P = torch.as_tensor(t.p[k].astype(np.float32), device=device)
                th = torch.cat([ths[ti][:, :1], ths[ti][:, :1] + torch.cumsum(torch.exp(ths[ti][:, 1:]), 1)], 1)  # (G, K−1)
                a_ = th[None] - e[:, :, None]  # (B, G, K−1)
                lo = torch.nn.functional.logsigmoid
                lp = [lo(a_[..., 0])] + [lo(a_[..., kk]) + lo(-a_[..., kk - 1]) + torch.log(-torch.expm1(th[:, kk - 1] - th[:, kk])) for kk in range(1, t.K - 1)] + [lo(-a_[..., -1])]
                lp = torch.stack(lp, -1)
                l = -(w[:, None] * (P[:, None, :] * lp).sum(-1)).sum(0)
            per.append(l)
        return per

    best = {k: None for k in ("loss", "W", "b", "Q")}
    bl = torch.full((M,), float("inf"), device=device)
    best_W = torch.zeros_like(W); best_b = torch.zeros_like(b); best_Q = torch.zeros_like(Q) if attn else None
    best_th = {ti: ths[ti].detach().clone() for ti in ths}
    order_fit = used
    eval_rows = np.unique(np.concatenate([t.rows[t.val] for t in targets]))
    Nf = len(order_fit)
    for ep in range(epochs):
        perm = rng.permutation(order_fit)
        for k0 in range(0, Nf, batch):
            rows = np.sort(perm[k0:k0 + batch])
            X = torch.as_tensor(np.asarray(feats[rows], dtype=np.float32), device=device)
            eta = forward(X)
            per = losses(eta, rows, "fit")
            total = 0.0
            for ti, l in enumerate(per):
                total = total + (l * (Nf / len(rows)) / tot_fit[ti]).sum()
            reg = 0.5 * (wdv[:, None] * (W ** 2)).sum() + (0.5 * (wdv[:, None] * (Q ** 2)).sum() if attn else 0.0)
            loss = total + reg
            opt.zero_grad(); loss.backward(); opt.step()
        # val
        with torch.no_grad():
            acc = [torch.zeros(G, device=device) for _ in targets]
            for k0 in range(0, len(eval_rows), batch):
                rows = eval_rows[k0:k0 + batch]
                X = torch.as_tensor(np.asarray(feats[rows], dtype=np.float32), device=device)
                for ti, l in enumerate(losses(forward(X), rows, "val")):
                    acc[ti] = acc[ti] + l
            nv = [float(t.val.sum()) if t.kind == "binary" else float(t.n[t.val].sum()) for t in targets]
            v = torch.cat([acc[ti] / nv[ti] for ti in range(T)])
            better = v < bl
            bl = torch.where(better, v, bl)
            best_W[better] = W[better]; best_b[better] = b[better]
            if attn:
                best_Q[better] = Q[better]
            for ti in ths:
                sel = better[ti * G:(ti + 1) * G]
                best_th[ti][sel] = ths[ti][sel]
        sched.step()
    with torch.no_grad():
        W.copy_(best_W); b.copy_(best_b)
        if attn:
            Q.copy_(best_Q)
    vb = bl.reshape(T, G).cpu().numpy()
    g = vb.argmin(1)
    out = {}
    te_all = np.unique(np.concatenate([t.rows[t.te] for t in targets]))
    eta_te = np.zeros((len(te_all), T), np.float32)
    idx_te = {int(r): i for i, r in enumerate(te_all)}
    with torch.no_grad():
        for k0 in range(0, len(te_all), batch):
            rows = te_all[k0:k0 + batch]
            X = torch.as_tensor(np.asarray(feats[rows], dtype=np.float32), device=device)
            e = forward(X).cpu().numpy()
            eta_te[k0:k0 + len(rows)] = e[:, np.arange(T) * G + g]
    for ti, t in enumerate(targets):
        te_rows = t.rows[t.te]
        out[t.name] = {"eta": eta_te[np.array([idx_te[int(r)] for r in te_rows]), ti], "wd": float(WD[g[ti]]), "val_loss": float(vb[ti].min()), "at_edge": bool(g[ti] in (0, G - 1)),
                       "th": (lambda r: np.concatenate([[r[0]], r[0] + np.cumsum(np.exp(r[1:]))]))(best_th[ti][g[ti]].cpu().numpy()) if ti in ths else None}
        if attn:
            out[t.name]["Q"] = Q[ti * G + g[ti]].detach().cpu().numpy()
        out[t.name]["W"] = W[ti * G + g[ti]].detach().cpu().numpy()
    return out


# --- H1 -----------------------------------------------------------------------------------------------------------------------
def load_targets(z):
    metas = json.loads(str(z["meta"]))
    return [Target(z, m["name"], m["kind"]) for m in metas], metas


def dev():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def step_h1(enc):
    z = np.load(H / "targets.npz", allow_pickle=False)
    T, metas = load_targets(z)
    gz = [t for t in T if t.kind in ("binary", "ordinal")]
    ef = [t for t in T if t.kind == "efigi"]
    d_ = dev()
    L = np.load(H / f"layers_{enc}.npy", mmap_mode="r")
    mean = np.asarray(L[:, BLOCK - 1], dtype=np.float32)
    grid = np.asarray(np.load(H / f"grid_{enc}.npy", mmap_mode="r"), dtype=np.float32).reshape(len(mean), -1)
    assert (H / "tokens_scratch.owner").read_text().strip() == enc, "the scratch token file holds another encoder's tokens; run extract for this one first"
    tok = np.load(H / "tokens_scratch.npy", mmap_mode="r")
    res = {}
    for ro, feats, grp in (("mean", mean, None), ("grid", grid, None), ("attn", tok, None)):
        # Equal batch size and epoch budget across all three readouts (mean/grid/attn) so every
        # readout receives identical Adam step counts and `mean` converges to match `fixed_C_mean`.
        for group, ts, ep, bs, lr in (("GZ", gz, 60, 256, 5e-3), ("EFIGI", ef, 150, 256, 5e-3)):
            vs = [("real", ts)] + ([("shuffled", [shuffled(t, np.random.default_rng(SEED + 2 + k)) for k, t in enumerate(ts)])] if enc in ("A1", "untrained-s0") else [])
            for tag, tt in vs:
                t0 = time.perf_counter()
                o = train(ro, feats, tt, SEED + 3, ep, bs, lr, len(mean), d_)
                for name, v in o.items():
                    res[f"{ro}|{group}|{tag}|{name}"] = v
                A.log(f"H1 {enc} {ro} {group} {tag} {len(tt)} targets {time.perf_counter() - t0:.0f}s")
        np.savez(H / f"h1_{enc}.npz", **{f"{k}|{kk}": vv for k, v in res.items() for kk, vv in v.items() if vv is not None})
    np.savez(H / f"h1_{enc}.npz", **{f"{k}|{kk}": vv for k, v in res.items() for kk, vv in v.items() if vv is not None})


# --- H2a ----------------------------------------------------------------------------------------------------------------------
def newton_logistic(X, y, C, iters=50):
    """L2 logistic in sklearn's convention (min ½‖w‖² + C Σ log loss, intercept free), damped
    Newton with Armijo backtracking and clipped logits; X standardised."""
    n, p = X.shape
    p0 = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))
    th = np.r_[np.zeros(p, dtype=np.float64), np.log(p0 / (1.0 - p0))]
    Xb = np.c_[X.astype(np.float64), np.ones(n, dtype=np.float64)]
    reg = np.r_[np.ones(p, dtype=np.float64), 0.0]

    def obj(t):
        eta = Xb @ t
        # Numerically stable softplus: log(1 + exp(eta)) - y * eta
        ll = np.where(eta >= 0, eta + np.log1p(np.exp(-eta)), np.log1p(np.exp(eta))) - y * eta
        return float(0.5 * np.dot(reg * t, t) + C * ll.sum())

    f_cur = obj(th)
    converged = False
    for _ in range(iters):
        eta = np.clip(Xb @ th, -40.0, 40.0)
        s = 1.0 / (1.0 + np.exp(-eta))
        g = C * (Xb.T @ (s - y)) + reg * th
        if float(np.max(np.abs(g))) < 1e-6:
            converged = True
            break
        w_diag = np.maximum(s * (1.0 - s), 1e-12)
        Hs = C * (Xb * w_diag[:, None]).T @ Xb + np.diag(reg) + 1e-9 * np.eye(p + 1)
        step = np.linalg.solve(Hs, g)
        slope = float(np.dot(g, step))
        alpha = 1.0
        while alpha > 1e-8:
            cand = th - alpha * step
            f_cand = obj(cand)
            if np.isfinite(f_cand) and f_cand <= f_cur - 1e-4 * alpha * slope:
                th, f_cur = cand, f_cand
                break
            alpha *= 0.5
        else:
            break
        if float(np.max(np.abs(alpha * step))) < 1e-7:
            converged = True
            break
    if not np.isfinite(th).all():
        raise RuntimeError("newton_logistic produced non-finite parameters")
    if not converged and float(np.max(np.abs(g))) > 1e-2:
        raise RuntimeError(f"newton_logistic did not converge (max|g|={float(np.max(np.abs(g))):.2e})")
    return th[:-1], float(th[-1])


def step_h2a(enc):
    z = np.load(H / "targets.npz", allow_pickle=False)
    T, metas = load_targets(z)
    L = np.load(H / f"layers_{enc}.npy", mmap_mode="r")
    C = float(json.loads((H / "probe_c.json").read_text())["c"])
    out = {}
    for L_i in range(12):
        Xall = np.asarray(L[:, L_i], dtype=np.float64)
        for t in T:
            rows, tr, te = t.rows, t.fit | t.val, t.te
            X, mu, sd = Xall[rows], None, None
            mu, sd = X[tr].mean(0), X[tr].std(0)
            X = (X - mu) / np.maximum(sd, 1e-9)
            if t.kind == "binary":
                w, b = newton_logistic(X[tr], t.y[tr].astype(float), C)
                s = X[te] @ w + b
            else:
                fitrows, val = t.fit, t.val
                best = None
                mdl = A.Model("ord", X[fitrows], np.zeros(fitrows.sum()), t.p[fitrows], t.n[fitrows])
                for lam in (10.0 ** np.arange(1, -5.25, -0.5)):
                    m = mdl.fit(lam)
                    ll = float(t.n[val] @ (t.p[val] * A.logp("ord", m, X[val], np.zeros(val.sum()))).sum(1)) / t.n[val].sum()
                    if best is None or ll > best[0]:
                        best = (ll, lam)
                mdl = A.Model("ord", X[tr], np.zeros(tr.sum()), t.p[tr], t.n[tr])
                for lam in (10.0 ** np.arange(1, -5.25, -0.5)):
                    m = mdl.fit(lam)
                    if lam == best[1]:
                        break
                s = X[te] @ m["w"]
            out[f"{L_i + 1}|{t.name}"] = s.astype(np.float32)
        A.log(f"H2a {enc} block {L_i + 1} done")
        np.savez(H / f"h2a_{enc}.npz", **out)


# --- report -------------------------------------------------------------------------------------------------------------------
def _ll_gap(t, eta, th):
    """LL per vote of the fitted cumulative logit on test, minus the train marginal's (ordinal targets with a fitted η and thresholds)."""
    te = t.te
    lp = A.ord_logp(eta.astype(float), th.astype(float))
    m = (t.n[t.fit | t.val][:, None] * t.p[t.fit | t.val]).sum(0)
    base = np.log(np.clip(m / m.sum(), 1e-12, None))
    return float((t.n[te] * ((t.p[te] * lp).sum(1) - t.p[te] @ base)).sum() / t.n[te].sum())


def step_report():
    """docs/axes/h1.json, h2a.json. Rules fixed in the module docstring."""
    from scipy.stats import kendalltau
    import almagest_axes as AXS
    z = np.load(H / "targets.npz", allow_pickle=False)
    T, metas = load_targets(z)
    label = {m["name"]: m["label"] for m in metas}
    h1 = {e: np.load(H / f"h1_{e}.npz", allow_pickle=False) for e in ENC if (H / f"h1_{e}.npz").exists()}
    h2 = {e: np.load(H / f"h2a_{e}.npz", allow_pickle=False) for e in ENC if (H / f"h2a_{e}.npz").exists()}
    grp = lambda t: "EFIGI" if t.kind == "efigi" else "GZ"  # noqa: E731
    out1, out2 = {"targets": {}, "note": __doc__}, {"targets": {}, "note": __doc__}
    for t in T:
        te_ = t.te
        dec = z["dec"][t.rows[te_]]
        n = int(te_.sum())
        rng = np.random.default_rng(SEED + 7)
        idx = [rng.integers(0, n, n) for _ in range(200)]
        yt = (
            t.y[te_]
            if t.kind == "binary"
            else (t.raw_y[te_] if getattr(t, "raw_y", None) is not None else t.p[te_] @ np.arange(1, t.K + 1))
        )

        def met(eta, i=None):
            e, y = (eta, yt) if i is None else (eta[i], yt[i])
            return AXS.fast_auc(y, e) if t.kind == "binary" else float(kendalltau(e, y)[0])
        rec = {"kind": t.kind, "label": label[t.name], "n_test": n, "n_train": int((t.fit | t.val).sum()), "readouts": {}}
        g = grp(t)
        S = {(e, ro): h1[e][f"{ro}|{g}|real|{t.name}|eta"] for e in h1 for ro in ("mean", "grid", "attn")}
        for (e, ro), v in S.items():
            rec["readouts"].setdefault(ro, {})[e] = {"metric": met(v), "wd": float(h1[e][f"{ro}|{g}|real|{t.name}|wd"]), "val_loss": float(h1[e][f"{ro}|{g}|real|{t.name}|val_loss"]), "wd_at_grid_edge": bool(h1[e][f"{ro}|{g}|real|{t.name}|at_edge"])}
            if t.kind != "binary":
                rec["readouts"][ro][e]["ll_gap"] = _ll_gap(t, v, h1[e][f"{ro}|{g}|real|{t.name}|th"])
        if h2 and t.kind == "binary":  # (a): the ladder's fixed-C logistic probe on block 11
            rec["readouts"]["fixed_C_mean"] = {e: {"metric": met(h2[e][f"{BLOCK}|{t.name}"])} for e in h2}
        if h1:
            for ro in ("grid", "attn"):
                a = {e: {"gain": rec["readouts"][ro][e]["metric"] - rec["readouts"]["mean"][e]["metric"]} for e in h1}
                if h2 and t.kind == "binary":
                    for e in h1:
                        if e in rec["readouts"]["fixed_C_mean"]:
                            a[e]["gain_over_fixed_C_mean"] = rec["readouts"][ro][e]["metric"] - rec["readouts"]["fixed_C_mean"][e]["metric"]
                bs = {}
                for e in h1:
                    bs[e] = np.array([met(S[e, ro], i) - met(S[e, "mean"], i) for i in idx])
                    a[e]["gain_ci"] = np.percentile(bs[e], [2.5, 97.5]).tolist()
                un = [e for e in h1 if e != "A1"]
                if "A1" in h1 and un:
                    dm = bs["A1"] - np.column_stack([bs[e] for e in un]).mean(1)
                    a["A1_gain_minus_untrained_mean"] = {"value": a["A1"]["gain"] - float(np.mean([a[e]["gain"] for e in un])), "ci": np.percentile(dm, [2.5, 97.5]).tolist(),
                                                         "untrained_gain_range": [float(min(a[e]["gain"] for e in un)), float(max(a[e]["gain"] for e in un))]}
                # decile breakdown of the gain (A1 and the untrained mean)
                dd = []
                for k in range(10):
                    m = np.flatnonzero(dec == k)
                    if len(m) < 30 or (t.kind == "binary" and min(yt[m].sum(), len(m) - yt[m].sum()) < 10):
                        dd.append({"decile": k + 1, "n": int(len(m))})
                        continue
                    f = lambda e: met(S[e, ro], m) - met(S[e, "mean"], m)  # noqa: E731
                    dd.append({"decile": k + 1, "n": int(len(m)), "A1": f("A1") if "A1" in h1 else None, "untrained_mean": float(np.mean([f(e) for e in un])) if un else None})
                a["deciles"] = dd
                rec["readouts"][ro]["gain_over_mean"] = a
            # shuffled control (report both metric and val_loss: single-draw AUC has high variance under strong WD)
            rec["shuffled"] = {
                f"{e}|{ro}": {
                    "metric": met(h1[e][f"{ro}|{g}|shuffled|{t.name}|eta"]),
                    "val_loss": float(h1[e][f"{ro}|{g}|shuffled|{t.name}|val_loss"]),
                }
                for e in ("A1", "untrained-s0")
                if e in h1
                for ro in ("mean", "grid", "attn")
                if f"{ro}|{g}|shuffled|{t.name}|eta" in h1[e].files
            }
        # H2a
        if h2:
            b = {}
            for e in h2:
                b[e] = np.array([met(h2[e][f"{k + 1}|{t.name}"]) for k in range(12)])
            r2 = {"metric_by_block": {e: b[e].tolist() for e in b}, "label": label[t.name], "kind": t.kind}
            if "A1" in b and len(b) > 1:
                un = [e for e in b if e != "A1"]
                emer = None
                cis = []
                for k in range(12):
                    u = np.column_stack([[met(h2[e][f"{k + 1}|{t.name}"], i) for i in idx] for e in un]).max(1)
                    a1 = np.array([met(h2["A1"][f"{k + 1}|{t.name}"], i) for i in idx])
                    lo, hi = np.percentile(a1 - u, [2.5, 97.5])
                    cis.append([float(lo), float(hi)])
                    if emer is None and lo > 0.05:
                        emer = k + 1
                r2["A1_minus_max_untrained_ci_by_block"] = cis
                r2["emergence_block"] = emer
                r2["A1_peak_block"] = int(b["A1"].argmax()) + 1
                r2["untrained_max_by_block"] = np.column_stack([b[e] for e in un]).max(1).tolist()
            out2["targets"][t.name] = r2
        out1["targets"][t.name] = rec
        (A.DOC / "h1.json").write_text(json.dumps(out1, indent=1, default=float))
        (A.DOC / "h2a.json").write_text(json.dumps(out2, indent=1, default=float))
        A.log(f"report {t.name}")


# --- selftest -----------------------------------------------------------------------------------------------------------------
def step_selftest():
    rng = np.random.default_rng(0)
    N, K, F = 3000, 64, 24
    X = rng.normal(size=(N, K, F)).astype(np.float32)
    marker = rng.integers(0, K, N)  # each image has one token carrying the marker feature (index 0) and a content bit (index 1): a token-only feature the mean cannot see well
    X[np.arange(N), marker, 0] += 6.0
    bit = rng.choice([-1.0, 1.0], N)
    X[np.arange(N), marker, 1] += 4.0 * bit
    mean_bit = (X[:, :, 2].mean(1) * 8 + rng.normal(size=N) * 0.3 > 0)
    y_tok = (bit > 0)
    z = {}
    rows, tr = np.arange(N), np.r_[np.ones(2000, bool), np.zeros(1000, bool)]
    val = np.zeros(N, bool); val[rng.choice(2000, 300, replace=False)] = True
    z["is_val"] = val
    for nm, y in (("token", y_tok), ("meanonly", mean_bit), ("noise", rng.random(N) < 0.5)):
        z[f"{nm}|rows"], z[f"{nm}|tr"], z[f"{nm}|y"] = rows, tr, y.astype(np.float32)
    ts = [Target(z, nm, "binary") for nm in ("token", "meanonly", "noise")]
    import almagest_axes as AXS
    out = {}
    for ro, feats, bs in (("mean", X.mean(1), 256), ("grid", X.reshape(N, 8, 8, F).mean(1).reshape(N, -1), 256), ("attn", X, 128)):
        r = train(ro, feats, ts, 0, 60, bs, 5e-3, N, dev())
        out[ro] = {t.name: float(AXS.fast_auc(t.y[t.te], r[t.name]["eta"])) for t in ts}
    print(json.dumps(out, indent=1))
    assert out["attn"]["token"] > 0.95 and out["mean"]["token"] < 0.75, "attention must read the token-only feature where the mean cannot"
    assert out["attn"]["noise"] < 0.60 and out["mean"]["meanonly"] > 0.9
    # ordinal head: planted one-axis through the same trainer
    p = np.zeros((N, 4)); lat = X[:, :, 5].mean(1) * 8 + 0.3 * rng.normal(size=N)
    for k, c in enumerate(np.digitize(lat, [-1, 0, 1])):
        p[k, c] = 1
    z["ord|rows"], z["ord|tr"], z["ord|p"], z["ord|n"] = rows, tr, p, np.ones(N)
    to = Target(z, "ord", "ordinal")
    r = train("mean", X.mean(1), [to], 0, 60, 256, 5e-3, N, dev())
    from scipy.stats import kendalltau
    tau = float(kendalltau(r["ord"]["eta"], lat[2000:])[0])
    print("ordinal head τ with the planted latent", round(tau, 3))
    assert tau > 0.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["plan", "selftest", "targets", "extract", "h1", "h2a", "report"])
    ap.add_argument("--enc", choices=ENC)
    a = ap.parse_args()
    if a.step == "plan":
        step_plan()
    elif a.step == "selftest":
        step_selftest()
    elif a.step == "targets":
        build_targets()
    elif a.step == "extract":
        step_extract(a.enc)
    elif a.step == "report":
        step_report()
    elif a.step == "h1":
        step_h1(a.enc)
    else:
        step_h2a(a.enc)


if __name__ == "__main__":
    main()
