"""Almagest diffusion — 64-px smoke run of three switchable input fixes (user brief, 21 Nov 2026).

VISUALISATION ONLY, branch `almagest-geometry`. Stats only; no re-embedding judgement at 64 px. Reads the
pilot's 64-px arrays and A1 banks (never writes them); the U-Net, schedule and loss are the pilot's, imported.

    .venv/bin/python artifacts/almagest_diffusion_smoke.py sky              # sky levels of all 100k (fix 2's store)
    .venv/bin/python artifacts/almagest_diffusion_smoke.py run full  --steps 2000 --batch 32
    .venv/bin/python artifacts/almagest_diffusion_smoke.py run nosky --steps 2000 --batch 32   # ablation: fix 2 off
    .venv/bin/python artifacts/almagest_diffusion_smoke.py eval full|nosky   # fresh process: 64 w = 0 samples, stats

Pipeline per image, x = the normalised stamp pooled to 64 px:
  fix 2 (sky):   x ← x − s, s = per-image, per-band 3σ-clipped median of the outer 8-px ring (the crop's own edge,
                 as in the realistic padded-crop row); s is stored; renders add back the median s per band.
  (kept)         y = asinh(x) / 0.8, the pilot's squash.
  fix 1 (scale): z = (y − m_c) / sd_c per channel; m, sd from a frozen train draw, stored in the checkpoint.
  fix 3 (clamp): the sampler's x0 clamp = the training z's per-channel 0.1 / 99.9 percentiles (after 1 and 2).

Investigation code: terse, excluded from lint/CI. Writes runs/almagest/diffusion/smoke_*/ and docs/almagest/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import almagest_diffusion as D  # noqa: E402
import almagest_diffusion_checks as C  # noqa: E402

N_FIT = 8192  # frozen train draw for the scale and clamp constants
RING = C.ring(D.RES, 8)
SKY = D.OUT / "smoke_sky_levels.npz"


def ring_sky(x: np.ndarray) -> np.ndarray:
    """(B, 3, 64, 64) → (B, 3): 3σ-clipped (1.4826·MAD) median of the outer 8-px ring, 3 passes."""
    p = x[:, :, RING].astype(np.float64)
    for _ in range(3):
        med = np.nanmedian(p, 2, keepdims=True)
        mad = 1.4826 * np.nanmedian(np.abs(p - med), 2, keepdims=True)
        p = np.where(np.abs(p - med) > 3 * mad, np.nan, p)
    return np.nanmedian(p, 2)


def step_sky() -> None:
    x64 = np.load(D.OUT / "train_x64.npy", mmap_mode="r")
    s = np.zeros((len(x64), 3), np.float32)
    for i in range(0, len(x64), 2048):
        s[i:i + 2048] = ring_sky(np.asarray(x64[i:i + 2048], dtype=np.float32))
    np.savez(SKY, ids=np.load(D.OUT / "train_bank.npz")["ids"], sky=s,
             note="per-image, per-band 3σ-clipped median of the 64-px stamp's outer 8-px ring, x units")
    print({"median": np.median(s, 0).tolist(), "iqr": np.subtract(*np.percentile(s, [75, 25], 0)).tolist()})


class Prep:
    def __init__(self, sky_on: bool, x_fit: np.ndarray, s_fit: np.ndarray) -> None:
        self.sky_on = sky_on
        self.sky_median = np.median(s_fit, 0).astype(np.float32) if sky_on else np.zeros(3, np.float32)
        y = self._y(x_fit, s_fit)
        self.m = y.mean((0, 2, 3)).astype(np.float32)
        self.sd = y.std((0, 2, 3)).astype(np.float32)
        z = (y - self.m[None, :, None, None]) / self.sd[None, :, None, None]
        zc = np.moveaxis(z, 1, 0).reshape(3, -1)
        self.lo, self.hi = np.percentile(zc, 0.1, 1).astype(np.float32), np.percentile(zc, 99.9, 1).astype(np.float32)

    def _y(self, x, s):
        if self.sky_on:
            x = x - s[:, :, None, None]
        return np.arcsinh(x) / D.SQUASH

    def fwd(self, x: np.ndarray, s: np.ndarray) -> np.ndarray:
        return ((self._y(x, s) - self.m[None, :, None, None]) / self.sd[None, :, None, None]).astype(np.float32)

    def inv(self, z: np.ndarray) -> np.ndarray:
        """Model z → x, adding back the median sky (renders)."""
        y = z * self.sd[None, :, None, None] + self.m[None, :, None, None]
        return np.sinh(D.SQUASH * y) + self.sky_median[None, :, None, None]

    def state(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("m", "sd", "lo", "hi", "sky_median")} | {"sky_on": self.sky_on}


def loss_v(net, x0, c, ab, gen):
    """The pilot's loss with the target swapped: v = √ᾱ ε − √(1−ᾱ) x0 (Salimans & Ho 2022); same t, noise, dropout."""
    b = x0.shape[0]
    t = torch.randint(0, D.T_STEPS, (b,), device=x0.device, generator=gen)
    n = torch.randn(x0.shape, device=x0.device, generator=gen)
    a = ab[t][:, None, None, None]
    xt = a.sqrt() * x0 + (1 - a).sqrt() * n
    drop = torch.rand(b, device=x0.device, generator=gen) < D.P_UNCOND
    return torch.nn.functional.mse_loss(net(xt, t, c, drop), a.sqrt() * n - (1 - a).sqrt() * x0)


def ddim_v(vfn, x, ab, steps=100):
    """DDIM η = 0 driven by a v-predictor `vfn(x, t)`, no clamp: x0 = √ᾱ x − √(1−ᾱ) v, ε = √(1−ᾱ) x + √ᾱ v."""
    ts = torch.linspace(D.T_STEPS - 1, 0, steps, device=x.device).long()
    for i, t in enumerate(ts):
        v = vfn(x, t)
        a = ab[t]
        a_prev = ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=x.device)
        x0 = a.sqrt() * x - (1 - a).sqrt() * v
        e = (1 - a).sqrt() * x + a.sqrt() * v
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * e
    return x


@torch.no_grad()
def sample_v(net, n, ab, gen, dev, cond=None):
    """w = 0 (cond None: the null branch) or w = 1 (cond given), v-prediction, no clamp."""
    x = torch.randn(n, 3, D.RES, D.RES, device=dev, generator=gen)
    c = torch.zeros(n, 384, device=dev) if cond is None else cond
    drop = torch.full((n,), cond is None, dtype=torch.bool, device=dev)
    return ddim_v(lambda x_, t: net(x_, t.repeat(n), c, drop), x, ab)


def step_oracle() -> dict:
    """Planted check of the v sampler: exact E[v | x_t] for Gaussian data x0 ~ N(μ, s²) must return (μ, s)."""
    ab = D.schedule(torch.device("cpu"))
    out = {}
    for mu, s in ((0.0, 1.0), (0.0, 0.44), (0.5, 0.3), (-0.2, 1.5)):
        def vfn(x, t, mu=mu, s=s):
            a = ab[t]
            den = a * s * s + 1 - a
            e = (1 - a).sqrt() * (x - a.sqrt() * mu) / den
            x0 = mu + a.sqrt() * s * s * (x - a.sqrt() * mu) / den
            return a.sqrt() * e - (1 - a).sqrt() * x0
        x = ddim_v(vfn, torch.randn(64, 3, 64, 64, generator=torch.Generator().manual_seed(0)), ab)
        lvl = x.mean((2, 3))
        out[f"N({mu},{s}^2)"] = {"mean": float(x.mean()), "sd": float(x.std()), "per_image_level_sd": float(lvl.std()),
                                 "expected_level_sd": s / 64}
        assert abs(float(x.mean()) - mu) < 0.02 and abs(float(x.std()) / s - 1) < 0.03, out
    print(json.dumps(out, indent=1))
    return out


@torch.no_grad()
def sample_uncond(net, n, ab, gen, dev, lo, hi, steps=100):
    """DDIM η = 0, unconditional branch only (w = 0), x0 clamped per channel to [lo, hi]."""
    x = torch.randn(n, 3, D.RES, D.RES, device=dev, generator=gen)
    ts = torch.linspace(D.T_STEPS - 1, 0, steps, device=dev).long()
    c = torch.zeros(n, 384, device=dev)
    no = torch.ones(n, dtype=torch.bool, device=dev)
    lo_, hi_ = (torch.tensor(v, device=dev)[None, :, None, None] for v in (lo, hi))
    for i, t in enumerate(ts):
        e = net(x, t.repeat(n), c, no)
        a = ab[t]
        a_prev = ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
        x0 = torch.maximum(torch.minimum((x - (1 - a).sqrt() * e) / a.sqrt(), hi_), lo_)
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * e
    return x


@torch.no_grad()
def sample_cond(net, cond, ab, gen, dev, lo, hi, steps=100):
    """DDIM η = 0, conditional branch only (w = 1), the same per-channel clamp: does the null branch lag?"""
    n = cond.shape[0]
    x = torch.randn(n, 3, D.RES, D.RES, device=dev, generator=gen)
    ts = torch.linspace(D.T_STEPS - 1, 0, steps, device=dev).long()
    yes = torch.zeros(n, dtype=torch.bool, device=dev)
    lo_, hi_ = (torch.tensor(v, device=dev)[None, :, None, None] for v in (lo, hi))
    for i, t in enumerate(ts):
        e = net(x, t.repeat(n), cond, yes)
        a = ab[t]
        a_prev = ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
        x0 = torch.maximum(torch.minimum((x - (1 - a).sqrt() * e) / a.sqrt(), hi_), lo_)
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * e
    return x


def stats_table(z_in, z_s, x_in, x_s) -> dict:
    rm_in, rm_s = [RING] * len(z_in), [RING] * len(z_s)
    out = {}
    for name, a, b in (("model_units", z_in, z_s), ("x_units_rendered", x_in, x_s)):
        ref = np.median(a[:, :, RING], (0, 2))
        out[name] = {"inputs": {"whole": C.chan_stats(a), "ring": C.chan_stats(a, rm_in),
                                "offset_gt_1sd": C.offsets_over_noise(a, rm_in, ref)},
                     "samples_w0": {"whole": C.chan_stats(b), "ring": C.chan_stats(b, rm_s),
                                    "offset_gt_1sd": C.offsets_over_noise(b, rm_s, ref)}}
    return out


def step_run(arm: str, steps: int, bs: int, lr: float = 2e-4) -> None:
    sky_on = arm in ("full", "full_v")
    vpred = arm == "full_v"  # v-prediction, no clamp (user, 21 Nov); fixes 1 and 2 kept
    out = D.OUT / f"smoke_{arm}"
    out.mkdir(parents=True, exist_ok=True)
    x64 = np.load(D.OUT / "train_x64.npy", mmap_mode="r")
    sky = np.load(SKY)["sky"]
    bank = D.Bank("train")
    rng = np.random.default_rng(D.SEED + 21)
    fit = np.sort(rng.choice(len(x64), N_FIT, replace=False))
    prep = Prep(sky_on, np.asarray(x64[fit], dtype=np.float32), sky[fit])
    dev = D.device()
    net, npar = D.make(dev)
    ema = D.UNet().to(dev)
    ema.load_state_dict(net.state_dict())
    ema.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    ab = D.schedule(dev)
    gen = torch.Generator(device=dev).manual_seed(D.SEED)
    log = open(out / "train_log.jsonl", "a")
    log.write(json.dumps({"event": "start", "arm": arm, "pred": "v" if vpred else "eps", "params": npar, "batch": bs, "steps": steps, "prep": prep.state(),
                          **D.machine()}) + "\n")
    t0, last, sm, start = time.perf_counter(), time.perf_counter(), None, 1
    part = out / "partial.pt"  # resume after a swap-guard kill: weights, optimiser, both RNGs, the loss EMA
    if part.exists():
        pk = torch.load(part, map_location="cpu", weights_only=False)
        net.load_state_dict(pk["net"]); ema.load_state_dict(pk["ema"]); opt.load_state_dict(pk["opt"])
        rng.bit_generator.state = pk["rng"]; gen.set_state(pk["gen"])
        start, sm, t0 = pk["step"] + 1, pk["sm"], t0 - pk["elapsed"]
        log.write(json.dumps({"event": "resume", "from_step": pk["step"], **D.machine()}) + "\n")
    for step in range(start, steps + 1):
        i = np.sort(rng.choice(len(x64), bs, replace=False))
        x0 = torch.from_numpy(prep.fwd(np.asarray(x64[i], dtype=np.float32), sky[i])).to(dev)
        c = torch.from_numpy(bank.cond(bank.emb[i])).to(dev)
        loss = loss_v(net, x0, c, ab, gen) if vpred else D.loss_fn(net, x0, c, ab, gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            d = min(0.999, (1 + step) / (10 + step))
            for pe, p in zip(ema.parameters(), net.parameters()):
                pe.mul_(d).add_(p.detach(), alpha=1 - d)
        if step % 50 == 0:
            lv = float(loss.detach())
            sm = lv if sm is None else 0.9 * sm + 0.1 * lv
        if step % 250 == 0 or step == steps:
            now = time.perf_counter()
            rec = {"step": step, "loss_ema": sm, "steps_per_s_window": (250 if step % 250 == 0 else step % 250) / (now - last),
                   "rss_peak_gb": D.rss_gb(), **D.mps_mem(), **D.machine()}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            last = now
            if step % 500 == 0 and step < steps:
                torch.save({"net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                            "rng": rng.bit_generator.state, "gen": gen.get_state(), "sm": sm,
                            "elapsed": time.perf_counter() - t0}, part)
            print(f"{arm} step {step} loss {sm if sm is None else round(sm, 4)} {rec['steps_per_s_window']:.2f} it/s", file=sys.stderr, flush=True)
    train_s = time.perf_counter() - t0
    torch.save({"ema": ema.state_dict(), "step": steps, "prep": prep.state(), "pred": "v" if vpred else "eps", "batch": bs, "train_seconds": train_s,
                "final_loss_ema": sm}, out / "smoke.pt")
    log.write(json.dumps({"event": "end", "train_seconds": train_s, **D.machine()}) + "\n")
    log.close()
    part.unlink(missing_ok=True)


def step_eval(arm: str, w1: bool = False) -> None:
    """A fresh process (the Mac is at its swap edge): EMA weights only, 64 w = 0 samples in batches of 16, stats."""
    out = D.OUT / f"smoke_{arm}"
    ck = torch.load(out / "smoke.pt", map_location="cpu")
    st = ck["prep"]
    prep = Prep.__new__(Prep)
    for k in ("m", "sd", "lo", "hi", "sky_median"):
        setattr(prep, k, np.asarray(st[k], np.float32))
    prep.sky_on = st["sky_on"]
    dev = D.device()
    ema = D.UNet().to(dev)
    ema.load_state_dict(ck["ema"])
    ema.eval()
    ab = D.schedule(dev)
    g2 = torch.Generator(device=dev).manual_seed(D.SEED + 11)
    x64 = np.load(D.OUT / "train_x64.npy", mmap_mode="r")
    sky = np.load(SKY)["sky"]
    ev = np.sort(np.random.default_rng(D.SEED + 31).choice(len(x64), 64, replace=False))
    vpred = ck.get("pred") == "v"
    if vpred:
        bank = D.Bank("train")
        cond = torch.from_numpy(bank.cond(bank.emb[ev])).to(dev) if w1 else None
        zs = torch.cat([sample_v(ema, 16, ab, g2, dev, None if cond is None else cond[k:k + 16]).float()
                        for k in range(0, 64, 16)]).cpu().numpy()
    elif w1:  # conditioned on the 64 evaluation galaxies' own A1 embeddings (train galaxies: a fit check, not a test)
        bank = D.Bank("train")
        cond = torch.from_numpy(bank.cond(bank.emb[ev])).to(dev)
        zs = torch.cat([sample_cond(ema, cond[k:k + 16], ab, g2, dev, prep.lo, prep.hi).float()
                        for k in range(0, 64, 16)]).cpu().numpy()
    else:
        zs = torch.cat([sample_uncond(ema, 16, ab, g2, dev, prep.lo, prep.hi).float() for _ in range(4)]).cpu().numpy()
    x_in = np.asarray(x64[ev], dtype=np.float32)
    z_in = prep.fwd(x_in, sky[ev])
    res = {"arm": arm, "guidance": "w = 1, own embeddings" if w1 else "w = 0", "pred": "v" if vpred else "eps",
           "fixes": {"1_unit_variance": True, "2_sky_subtraction": prep.sky_on, "3_percentile_clamp": not vpred},
           "steps": ck["step"], "batch": ck["batch"], "train_seconds": ck["train_seconds"],
           "final_loss_ema": ck["final_loss_ema"], "prep": st,
           "stats": stats_table(z_in, zs, x_in, prep.inv(zs)), "machine_end": D.machine()}
    np.save(out / ("samples_w1_64_z.npy" if w1 else "samples_w0_64_z.npy"), zs)
    p = D.DOCS / "diffusion_smoke.json"
    allr = json.loads(p.read_text()) if p.exists() else {}
    allr[arm + ("_w1" if w1 else "")] = res
    p.write_text(json.dumps(allr, indent=1, default=float))
    print(json.dumps(res["stats"]["model_units"], default=float)[:1500])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sky", "run", "eval", "oracle"])
    ap.add_argument("arm", nargs="?", choices=["full", "nosky", "full_v"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--w1", action="store_true", help="eval: conditional samples on the eval galaxies' own embeddings")
    a = ap.parse_args()
    if a.step == "sky":
        step_sky()
    elif a.step == "oracle":
        step_oracle()
    elif a.step == "run":
        step_run(a.arm, a.steps, a.batch)
    else:
        step_eval(a.arm, a.w1)


if __name__ == "__main__":
    main()
