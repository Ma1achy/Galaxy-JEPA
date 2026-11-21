"""Almagest diffusion — feasibility pilot: render a galaxy from a point in A1's embedding space (RCDM-style).

VISUALISATION ONLY (user brief, 20 Nov 2026, branch `almagest-geometry`): no claim, nothing in the science record.
The conditioning vector is RAW A1 — frozen `runs/m_v2/encoder.pt`, penultimate block, mean-pooled, 384-d, the
probes' own vector. Cleaning and γ are for walks only, never for conditioning.

The 64×64 images here are DISPLAY-ONLY: 4×4 area-averaged from the 256-px native stamps for this model alone.
The science pipeline's no-rebin rule (docs/spec/data.md) is untouched: nothing here feeds a probe, and A1's
embeddings are always taken at native 256 px.

    .venv/bin/python artifacts/almagest_diffusion.py data    # split, 64-px arrays, A1 embedding bank
    .venv/bin/python artifacts/almagest_diffusion.py bench   # steps/s, memory (load recorded)
    .venv/bin/python artifacts/almagest_diffusion.py train --hours 2.5
    .venv/bin/python artifacts/almagest_diffusion.py eval    # reconstruction sheet, re-embedding cosines

Investigation code: terse, excluded from lint/CI. Writes runs/almagest/diffusion/ and docs/almagest/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs/almagest/diffusion"
DOCS = REPO / "docs/almagest"
CKPT_A1 = REPO / "runs/m_v2/encoder.pt"
CACHE = "/Volumes/X10 Pro/galaxy-jepa/runs/m_v2/cache/eab81d179579f2f083f11b2301a662db781bd395e664f5ba5d71041163d5abfd"
SEED = 20261120
N_TRAIN, N_TEST = 100_000, 2_048
RES, POOL = 64, 4
SQUASH = 0.8  # model pixels y = asinh(x) / SQUASH (pixel SD ≈ 0.42; 2.0 left SD ≈ 0.17 against unit noise); x = sinh(SQUASH·y)
T_STEPS = 1000
P_UNCOND = 0.10


def sha1(p: Path) -> str:
    h = hashlib.sha1()
    with open(Path(p).resolve(), "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def machine() -> dict:
    load = os.getloadavg()
    mp = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout.strip().splitlines()
    return {"load_avg_1_5_15": [round(x, 2) for x in load], "under_load": load[0] > 16.0,
            "memory_free": mp[-1] if mp else None, "time": time.strftime("%Y-%m-%d %H:%M:%S")}


def rss_gb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # macOS: bytes


# --- data --------------------------------------------------------------------------------------------
def pool64(x: np.ndarray) -> np.ndarray:
    """(B, 3, 256, 256) → (B, 3, 64, 64) area average: display-only."""
    b = x.shape[0]
    return x.reshape(b, 3, RES, POOL, RES, POOL).mean((3, 5))


def up256(x: torch.Tensor) -> torch.Tensor:
    """64 → 256 for re-embedding with A1 (bilinear)."""
    return F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)


def to_model(x):
    return np.arcsinh(x) / SQUASH


def from_model(y):
    return torch.sinh(SQUASH * y) if isinstance(y, torch.Tensor) else np.sinh(SQUASH * y)


def a1_encoder(dev):
    from galaxy_jepa.models.vit import load_frozen_encoder
    enc = load_frozen_encoder(CKPT_A1).to(dev).eval()
    return enc


@torch.no_grad()
def embed(enc, x: np.ndarray | torch.Tensor, dev, bs: int = 128) -> np.ndarray:
    out = []
    for i in range(0, len(x), bs):
        b = x[i:i + bs]
        b = b if isinstance(b, torch.Tensor) else torch.from_numpy(np.asarray(b, dtype=np.float32))
        out.append(enc.encode(b.to(dev).float()).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def step_data() -> None:
    from galaxy_jepa.data.cache import TensorCache
    from galaxy_jepa.data.orchestrate import resolve_corpora, split_pretrain
    from galaxy_jepa.data.sources import DirectorySource
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"machine_start": machine()}
    t0 = time.perf_counter()
    pre = [int(r["object_id"]) for r in DirectorySource(REPO / "data/pretrain_v2").rows]
    probe = [int(r["object_id"]) for r in DirectorySource(REPO / "data/probe_v2").rows]
    sp = split_pretrain(resolve_corpora(pre, probe), seed=0, monitor_frac=0.02)  # A1's own split (configs/pretrain_v2.yaml)
    tc = TensorCache(CACHE)
    tr_all = np.array(sorted(o for o in sp.train if o in tc), dtype=np.int64)
    mo_all = np.array(sorted(o for o in sp.monitor if o in tc), dtype=np.int64)
    rng = np.random.default_rng(SEED)
    ids = {"train": np.sort(rng.choice(tr_all, N_TRAIN, replace=False)),
           "test": np.sort(rng.choice(mo_all, N_TEST, replace=False))}
    rec["split"] = {"source": "split_pretrain(resolve_corpora(pretrain_v2, probe_v2), seed=0, monitor_frac=0.02) — A1's "
                              "own pretraining split; diffusion train ⊂ A1 train, diffusion test ⊂ A1's monitor slice "
                              "(seen by neither A1's training nor the diffusion model)",
                    "pretrain_v2": len(pre), "a1_train_in_cache": int(tr_all.size), "a1_monitor_in_cache": int(mo_all.size),
                    "n_train": N_TRAIN, "n_test": N_TEST, "draw_seed": SEED}
    assert not np.isin(ids["test"], ids["train"]).any()
    dev = device()
    enc = a1_encoder(dev)
    for part, oid in ids.items():
        rows = np.array([tc.row_of(int(o)) for o in oid])
        o = np.argsort(rows)  # read the memmap in row order (the SSD prefers it)
        oid, rows = oid[o], rows[o]
        img = np.lib.format.open_memmap(OUT / f"{part}_x64.npy", mode="w+", dtype=np.float16, shape=(len(oid), 3, RES, RES))
        emb = np.zeros((len(oid), 384), np.float32)
        emb_rt = np.zeros((len(oid), 384), np.float32) if part == "test" else None
        bs = 256
        for i in range(0, len(oid), bs):
            x = np.asarray(tc.data[rows[i:i + bs]], dtype=np.float32)
            emb[i:i + bs] = embed(enc, x, dev)
            p = pool64(x)
            img[i:i + bs] = p.astype(np.float16)
            if emb_rt is not None:  # the 64-px round trip of a REAL galaxy: the ceiling a 64-px sample can reach
                emb_rt[i:i + bs] = embed(enc, up256(torch.from_numpy(p)), dev)
            if i % (bs * 40) == 0:
                print(f"  {part} {i}/{len(oid)} {time.perf_counter() - t0:.0f}s", file=sys.stderr, flush=True)
        img.flush()
        np.savez(OUT / f"{part}_bank.npz", ids=oid, emb=emb, **({"emb_roundtrip": emb_rt} if emb_rt is not None else {}),
                 checkpoint=str(CKPT_A1.relative_to(REPO)), checkpoint_sha1=sha1(CKPT_A1),
                 vector="penultimate block (DEFAULT_LAYER = -2), mean-pooled, 384-d, native 256 px")
    rec["checkpoint_sha1"] = sha1(CKPT_A1)
    rec["seconds"] = time.perf_counter() - t0
    rec["machine_end"] = machine()
    (OUT / "data.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


# --- model: a small ADM-style conditional U-Net ---------------------------------------------------------
def t_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    f = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class ResBlock(nn.Module):
    """GroupNorm → SiLU → conv; the second norm is FiLM-modulated (scale, shift) by the time+condition embedding,
    as ADM's use_scale_shift_norm. RCDM puts the representation into every residual block's normalisation."""

    def __init__(self, cin, cout, edim) -> None:
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(32, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.e = nn.Linear(edim, 2 * cout)
        self.n2, self.c2 = nn.GroupNorm(32, cout), nn.Conv2d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        s, b = self.e(F.silu(e))[:, :, None, None].chunk(2, 1)
        h = self.c2(F.silu(self.n2(h) * (1 + s) + b))
        return self.skip(x) + h


class Attn(nn.Module):
    def __init__(self, c, heads=4) -> None:
        super().__init__()
        self.n, self.qkv, self.o, self.h = nn.GroupNorm(32, c), nn.Conv2d(c, 3 * c, 1), nn.Conv2d(c, c, 1), heads
        nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)

    def forward(self, x):
        b, c, hh, ww = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(b, 3, self.h, c // self.h, hh * ww).unbind(1)
        a = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.o(a.transpose(-1, -2).reshape(b, c, hh, ww))


class UNet(nn.Module):
    def __init__(self, ch=96, mult=(1, 2, 2, 3), nres=2, attn_res=(16, 8), cdim=384, edim=512) -> None:
        super().__init__()
        self.ch, self.edim = ch, edim
        self.temb = nn.Sequential(nn.Linear(ch, edim), nn.SiLU(), nn.Linear(edim, edim))
        # the condition: raw A1 (standardised by the train bank's mean/sd) → 512 → added to the time embedding,
        # so it reaches every residual block's FiLM; a learned null embedding stands in when dropped (CFG)
        self.cemb = nn.Sequential(nn.Linear(cdim, edim), nn.SiLU(), nn.Linear(edim, edim))
        self.null = nn.Parameter(torch.zeros(edim))
        self.inp = nn.Conv2d(3, ch, 3, padding=1)
        self.down, chans, c, res = nn.ModuleList(), [ch], ch, RES
        for i, m in enumerate(mult):
            for _ in range(nres):
                blk = nn.ModuleList([ResBlock(c, ch * m, edim)]); c = ch * m
                if res in attn_res:
                    blk.append(Attn(c))
                self.down.append(blk); chans.append(c)
            if i < len(mult) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(c, c, 3, stride=2, padding=1)])); chans.append(c); res //= 2
        self.mid = nn.ModuleList([ResBlock(c, c, edim), Attn(c), ResBlock(c, c, edim)])
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mult))):
            for _ in range(nres + 1):
                blk = nn.ModuleList([ResBlock(c + chans.pop(), ch * m, edim)]); c = ch * m
                if res in attn_res:
                    blk.append(Attn(c))
                self.up.append(blk)
            if i > 0:
                self.up.append(nn.ModuleList([nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, c, 3, padding=1)]))
                res *= 2
        self.out = nn.Sequential(nn.GroupNorm(32, c), nn.SiLU(), nn.Conv2d(c, 3, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t, cond, drop=None):
        e = self.temb(t_embed(t, self.ch))
        ce = self.cemb(cond)
        if drop is not None:
            ce = torch.where(drop[:, None], self.null[None].expand_as(ce), ce)
        e = e + ce
        h, hs = self.inp(x), []
        hs.append(h)
        for blk in self.down:
            if isinstance(blk[0], ResBlock):
                h = blk[0](h, e)
                for a in blk[1:]:
                    h = a(h)
            else:
                h = blk[0](h)
            hs.append(h)
        h = self.mid[2](self.mid[1](self.mid[0](h, e)), e)
        for blk in self.up:
            if isinstance(blk[0], ResBlock):
                h = blk[0](torch.cat([h, hs.pop()], 1), e)
                for a in blk[1:]:
                    h = a(h)
            else:
                h = blk[1](blk[0](h))
        return self.out(h)


def schedule(dev):
    """Cosine ᾱ (Nichol & Dhariwal 2021), T = 1000; ε-prediction, fixed variance."""
    s = 0.008
    tt = torch.linspace(0, T_STEPS, T_STEPS + 1, dtype=torch.float64)
    ab = torch.cos(((tt / T_STEPS) + s) / (1 + s) * math.pi / 2) ** 2
    ab = (ab / ab[0]).clamp(1e-5, 1)[1:]
    return ab.float().to(dev)


class Bank:
    def __init__(self, part: str) -> None:
        z = np.load(OUT / f"{part}_bank.npz")
        self.ids, self.emb = z["ids"], z["emb"]
        self.emb_rt = z["emb_roundtrip"] if "emb_roundtrip" in z else None
        self.x = np.load(OUT / f"{part}_x64.npy", mmap_mode="r")
        tr = np.load(OUT / "train_bank.npz")["emb"]
        self.mu, self.sd = tr.mean(0), tr.std(0) + 1e-6

    def cond(self, e):
        return ((e - self.mu) / self.sd).astype(np.float32)


def batch(bank: Bank, rng, bs, dev):
    i = np.sort(rng.choice(len(bank.ids), bs, replace=False))
    x = torch.from_numpy(to_model(np.asarray(bank.x[i], dtype=np.float32)))
    return x.to(dev), torch.from_numpy(bank.cond(bank.emb[i])).to(dev)


def loss_fn(net, x0, c, ab, gen):
    b = x0.shape[0]
    t = torch.randint(0, T_STEPS, (b,), device=x0.device, generator=gen)
    n = torch.randn(x0.shape, device=x0.device, generator=gen)
    a = ab[t][:, None, None, None]
    xt = a.sqrt() * x0 + (1 - a).sqrt() * n
    drop = torch.rand(b, device=x0.device, generator=gen) < P_UNCOND
    return F.mse_loss(net(xt, t, c, drop), n)


def make(dev):
    torch.manual_seed(SEED)
    net = UNet().to(dev)
    return net, sum(p.numel() for p in net.parameters())


def mps_mem() -> dict:
    if torch.backends.mps.is_available():
        return {"mps_current_gb": torch.mps.current_allocated_memory() / 1e9,
                "mps_driver_gb": torch.mps.driver_allocated_memory() / 1e9}
    return {}


def step_bench(bs=64, n=60, warm=10) -> dict:
    dev = device()
    net, npar = make(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4)
    bank, rng = Bank("train"), np.random.default_rng(SEED)
    ab = schedule(dev); gen = torch.Generator(device=dev).manual_seed(SEED)
    m0 = machine()
    for k in range(warm + n):
        if k == warm:
            if dev.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
        x0, c = batch(bank, rng, bs, dev)
        loss = loss_fn(net, x0, c, ab, gen)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if dev.type == "mps":
        torch.mps.synchronize()
    dt = time.perf_counter() - t0
    out = {"device": dev.type, "params": npar, "batch": bs, "steps_per_s": n / dt, "img_per_s": n * bs / dt,
           "machine_before": m0, "machine_after": machine(), "rss_peak_gb": rss_gb(), **mps_mem()}
    print(json.dumps(out, indent=1))
    return out


def flops_per_step(bs=64) -> float:
    """Forward+backward FLOPs of one training step (CPU flop counter, ×3 of forward)."""
    from torch.utils.flop_counter import FlopCounterMode
    net = UNet()
    x, t, c = torch.zeros(bs, 3, RES, RES), torch.zeros(bs, dtype=torch.long), torch.zeros(bs, 384)
    with FlopCounterMode(display=False) as fc:
        net(x, t, c)
    return 3.0 * fc.get_total_flops()


def step_train(hours: float, bs=64, lr=2e-4) -> None:
    dev = device()
    net, npar = make(dev)
    ema = UNet().to(dev); ema.load_state_dict(net.state_dict()); ema.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    bank, rng = Bank("train"), np.random.default_rng(SEED)
    ab = schedule(dev); gen = torch.Generator(device=dev).manual_seed(SEED)
    log = open(OUT / "train_log.jsonl", "a")
    t_start, step, last, sm = time.perf_counter(), 0, time.perf_counter(), None
    log.write(json.dumps({"event": "start", "params": npar, "batch": bs, "lr": lr, "hours": hours, **machine()}) + "\n")
    while time.perf_counter() - t_start < hours * 3600:
        x0, c = batch(bank, rng, bs, dev)
        loss = loss_fn(net, x0, c, ab, gen)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); step += 1
        with torch.no_grad():
            d = min(0.999, (1 + step) / (10 + step))
            for pe, p in zip(ema.parameters(), net.parameters()):
                pe.mul_(d).add_(p.detach(), alpha=1 - d)
        lv = float(loss) if step % 50 == 0 else None
        if lv is not None:
            sm = lv if sm is None else 0.9 * sm + 0.1 * lv
        if step % 500 == 0:
            now = time.perf_counter()
            rec = {"step": step, "loss_ema": sm, "steps_per_s_window": 500 / (now - last),
                   "elapsed_h": (now - t_start) / 3600, "rss_peak_gb": rss_gb(), **mps_mem(), **machine()}
            log.write(json.dumps(rec) + "\n"); log.flush(); last = now
            print(f"step {step} loss {sm:.4f} {rec['steps_per_s_window']:.2f} it/s load {rec['load_avg_1_5_15'][0]}",
                  file=sys.stderr, flush=True)
        if step % 5000 == 0:
            torch.save({"ema": ema.state_dict(), "net": net.state_dict(), "step": step}, OUT / "pilot.pt")
    torch.save({"ema": ema.state_dict(), "net": net.state_dict(), "step": step}, OUT / "pilot.pt")
    log.write(json.dumps({"event": "end", "step": step, "hours": (time.perf_counter() - t_start) / 3600, **machine()}) + "\n")
    log.close()


# --- eval ----------------------------------------------------------------------------------------------
@torch.no_grad()
def sample(net, cond, ab, gen, w=1.0, steps=100):
    """DDIM (η = 0) over `steps` of the 1000; classifier-free guidance ε = ε_u + w (ε_c − ε_u) (w = 1: conditional)."""
    dev = cond.device
    b = cond.shape[0]
    x = torch.randn(b, 3, RES, RES, device=dev, generator=gen)
    ts = torch.linspace(T_STEPS - 1, 0, steps, device=dev).long()
    yes, no = torch.zeros(b, dtype=torch.bool, device=dev), torch.ones(b, dtype=torch.bool, device=dev)
    for i, t in enumerate(ts):
        tb = t.repeat(b)
        ec = net(x, tb, cond, yes)
        e = ec if w == 1.0 else net(x, tb, cond, no) + w * (ec - net(x, tb, cond, no))
        a = ab[t]
        a_prev = ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
        x0 = ((x - (1 - a).sqrt() * e) / a.sqrt()).clamp(-2.0, 6.0)
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * e
    return x


def cos_rows(a, b):
    return np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def show(x):  # (3, H, W) normalised stamp → display RGB (i r g), per-image percentile stretch as the strips
    rgb = np.stack([x[2], x[1], x[0]], -1)
    lo, hi = np.percentile(rgb, [1, 99.7])
    return (np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)


def step_eval(n_sheet=64, n_stat=256, k=4, guid=(1.0, 3.0)) -> None:
    from PIL import Image, ImageDraw, ImageFont
    dev = device()
    ck = torch.load(OUT / "pilot.pt", map_location="cpu")
    net = UNet().to(dev); net.load_state_dict(ck["ema"]); net.eval()
    enc = a1_encoder(dev)
    te, tr = Bank("test"), np.load(OUT / "train_bank.npz")
    ab = schedule(dev)
    rng = np.random.default_rng(SEED + 1)
    pick = np.sort(rng.choice(len(te.ids), n_stat, replace=False))
    e_cond = te.emb[pick]
    mu = tr["emb"].mean(0)
    res = {"step": int(ck["step"]), "n_galaxies": n_stat, "samples_each": k, "sampler": "DDIM η=0, 100 steps",
           "machine": machine()}
    sheets = {}
    for w in guid:
        gen = torch.Generator(device=dev).manual_seed(SEED)
        cond = torch.from_numpy(te.cond(np.repeat(e_cond, k, 0))).to(dev)
        t0 = time.perf_counter()
        xs = torch.cat([sample(net, cond[i:i + 128], ab, gen, w) for i in range(0, len(cond), 128)])
        if dev.type == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - t0
        xr = from_model(xs.float())  # back to normalised-stamp units
        e_s = embed(enc, up256(xr), dev)
        ec = np.repeat(e_cond, k, 0)
        # RCDM-style rank: the conditioning galaxy's rank among the n_stat by cosine to the re-embedded sample
        S = (e_s / np.linalg.norm(e_s, axis=1, keepdims=True)) @ (e_cond / np.linalg.norm(e_cond, axis=1, keepdims=True)).T
        own = np.repeat(np.arange(n_stat), k)
        rank = 1 + (S > S[np.arange(len(own)), own][:, None]).sum(1)
        Sc = ((e_s - mu) / np.linalg.norm(e_s - mu, axis=1, keepdims=True)) @ ((e_cond - mu) / np.linalg.norm(e_cond - mu, axis=1, keepdims=True)).T
        rank_c = 1 + (Sc > Sc[np.arange(len(own)), own][:, None]).sum(1)
        res[f"w={w}"] = {"sample_seconds_per_image": dt / len(cond),
                         "cos_sample_to_condition": {"raw": float(np.median(cos_rows(e_s, ec))),
                                                     "centred": float(np.median(cos_rows(e_s - mu, ec - mu)))},
                         "rank_of_condition_among_n": {"median": float(np.median(rank)), "top1": float((rank == 1).mean()),
                                                       "median_centred": float(np.median(rank_c)), "top1_centred": float((rank_c == 1).mean())},
                         "cos_between_samples_same_condition_centred": float(np.median([
                             cos_rows(np.repeat(e_s[i * k:i * k + 1] - mu, k - 1, 0), e_s[i * k + 1:(i + 1) * k] - mu).mean()
                             for i in range(n_stat)]))}
        sheets[w] = xr[: n_sheet * k].cpu().numpy()
    # reference cosines among REAL galaxies (raw, and centred on the train bank's mean)
    rt = te.emb_rt[pick]
    pr = rng.permutation(n_stat)
    Etr = tr["emb"]
    En = (e_cond - mu) / np.linalg.norm(e_cond - mu, axis=1, keepdims=True)
    Tn = (Etr - mu) / np.linalg.norm(Etr - mu, axis=1, keepdims=True)
    nn_c = np.max(En @ Tn.T, 1)
    rnd = rng.choice(len(Etr), (2000, 2))
    res["reference"] = {
        "real_64px_roundtrip_to_own": {"raw": float(np.median(cos_rows(rt, e_cond))), "centred": float(np.median(cos_rows(rt - mu, e_cond - mu)))},
        "random_real_pairs": {"raw": float(np.median(cos_rows(Etr[rnd[:, 0]], Etr[rnd[:, 1]]))),
                              "centred": float(np.median(cos_rows(Etr[rnd[:, 0]] - mu, Etr[rnd[:, 1]] - mu)))},
        "random_real_pairs_roundtrip": {"centred": float(np.median(cos_rows(rt - mu, rt[pr] - mu)))},
        "nearest_real_neighbour_in_train_100k": {"centred": float(np.median(nn_c)),
                                                 "raw": float(np.median(np.max((e_cond / np.linalg.norm(e_cond, axis=1, keepdims=True)) @ (Etr / np.linalg.norm(Etr, axis=1, keepdims=True)).T, 1)))},
        "note": "medians over the galaxies; 'centred' subtracts the train bank's mean embedding (raw cosines share a "
                "large common mean); round trip = the real galaxy pooled to 64 px, upsampled bilinear to 256, re-embedded "
                "(the ceiling any 64-px sample can reach)"}
    rt_S = ((rt - mu) / np.linalg.norm(rt - mu, axis=1, keepdims=True)) @ En.T
    res["reference"]["real_roundtrip_rank_centred"] = {"median": float(np.median(1 + (rt_S > np.diag(rt_S)[:, None]).sum(1))),
                                                       "top1": float(np.mean(np.argmax(rt_S, 1) == np.arange(n_stat)))}
    (DOCS / "diffusion_pilot_eval.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    # sheets: original | 4 samples, 64 galaxies in 4 column groups of 16
    T, G = 64, 12
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except OSError:
        font = ImageFont.load_default()
    for w, xs in sheets.items():
        sheet = Image.new("RGB", (4 * (5 * T + G) + G, 16 * T + 40), (250, 249, 246))
        dr = ImageDraw.Draw(sheet)
        dr.text((G, 10), f"A1 RCDM pilot, step {ck['step']}, guidance w={w}: held-out galaxy (64 px, display-only) | 4 samples "
                         "from its own A1 embedding. i r g, per-image stretch.", fill=(29, 27, 25), font=font)
        orig = np.asarray(te.x[pick[:n_sheet]], dtype=np.float32)
        for g in range(n_sheet):
            x0, y0 = G + (g // 16) * (5 * T + G), 34 + (g % 16) * T
            sheet.paste(Image.fromarray(show(orig[g])), (x0, y0))
            for s in range(k):
                sheet.paste(Image.fromarray(show(xs[g * k + s])), (x0 + (s + 1) * T, y0))
        sheet.save(DOCS / f"diffusion_pilot_recon_w{w:g}.png", optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["data", "bench", "train", "eval", "flops"])
    ap.add_argument("--hours", type=float, default=2.5)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    if a.step == "data":
        step_data()
    elif a.step == "bench":
        out = step_bench(bs=a.batch)
        out["flops_per_step"] = flops_per_step(a.batch)
        (OUT / f"bench_b{a.batch}.json").write_text(json.dumps(out, indent=1))
    elif a.step == "train":
        step_train(a.hours, bs=a.batch)
    elif a.step == "eval":
        step_eval()
    elif a.step == "flops":
        print(flops_per_step(a.batch) / 1e9, "GFLOP/step")


if __name__ == "__main__":
    main()
