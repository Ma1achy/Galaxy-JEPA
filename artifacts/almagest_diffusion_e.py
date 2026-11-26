"""Almagest diffusion, Part E (brief, 24 Nov 2026): rectified flow matching + 80-D Plate conditioning, 64-px Mac smoke.

VISUALISATION ONLY, branch `almagest-geometry`. The pilot's U-Net (cdim switched to 80), fixes 1 and 2 kept (smoke's Prep), no clamp.
Flow: x_t = t x0 + (1 − t) ε, t ~ U[0, 1], target x0 − ε, time embedding 1000·t; Euler from t = 0 (noise) to 1.

    .venv/bin/python artifacts/almagest_diffusion_e.py selftest   # guided sampler (2 passes, same output) and the Euler sampler on Gaussians
    .venv/bin/python artifacts/almagest_diffusion_e.py recon      # 384-D embedding vs its 80-D reconstruction
    .venv/bin/python artifacts/almagest_diffusion_e.py run --steps 2000 --batch 32
    .venv/bin/python artifacts/almagest_diffusion_e.py eval       # fresh process: FM at 8/16/32/100 Euler, v at DDIM 8/16/32/100
    .venv/bin/python artifacts/almagest_diffusion_e.py sheet

Investigation code: terse, excluded from lint/CI.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import almagest_diffusion as D
import almagest_diffusion_smoke as SM

OUT = D.OUT / "smoke_fm80"
JS = D.DOCS / "diffusion_e.json"
PLATE = D.REPO / "runs/almagest/a1/plate_axes.bin"
EULER = (8, 16, 32, 100)


class Coord80:
    """A1's 384-D vector → the browser's 80-D coordinates: z = B((x − μ)/σ) − c, int8-rounded with the Plate's own scale (clipped at ±127)."""

    def __init__(self):
        import plate_export as PE
        h, A = PE.read_plate(str(PLATE))
        self.mu, self.sd, self.B, self.c = (A[k].astype(np.float64) for k in ("std_mean", "std_scale", "basis", "centre"))
        self.scale = np.asarray(h["coords"]["scale"])
        self.encoder = h["provenance"]["checkpoint"]

    def q(self, x):
        z = ((x - self.mu) / self.sd) @ self.B.T - self.c
        return np.clip(np.round(z / self.scale), -127, 127), z

    def __call__(self, x):
        return self.q(np.asarray(x, np.float64))[0] * self.scale

    def back(self, z):
        return self.mu + self.sd * ((z + self.c) @ self.B)


class Cond80:
    def __init__(self, emb_train):
        self.C = Coord80()
        z = self.C(emb_train)
        self.m, self.s = z.mean(0), z.std(0) + 1e-6

    def __call__(self, emb):
        return ((self.C(emb) - self.m) / self.s).astype(np.float32)


def loss_fm(net, x0, c, gen):
    b = x0.shape[0]
    t = torch.rand(b, device=x0.device, generator=gen)
    n = torch.randn(x0.shape, device=x0.device, generator=gen)
    xt = t[:, None, None, None] * x0 + (1 - t[:, None, None, None]) * n
    drop = torch.rand(b, device=x0.device, generator=gen) < D.P_UNCOND
    return torch.nn.functional.mse_loss(net(xt, 1000 * t, c, drop), x0 - n)


def euler(vfn, x, steps):
    for i in range(steps):
        x = x + vfn(x, i / steps) / steps
    return x


@torch.no_grad()
def sample_fm(net, n, gen, dev, steps, cond=None, w=1.0, cdim=80):
    """w = 0 if cond is None; else w = 1 (conditional) or guided: v_u + w (v_c − v_u), one unconditional pass per step."""
    x = torch.randn(n, 3, D.RES, D.RES, device=dev, generator=gen)
    c = torch.zeros(n, cdim, device=dev) if cond is None else cond
    yes, no = torch.zeros(n, dtype=torch.bool, device=dev), torch.ones(n, dtype=torch.bool, device=dev)

    def vfn(x_, t):
        tb = torch.full((n,), 1000.0 * t, device=dev)
        if cond is None:
            return net(x_, tb, c, no)
        vc = net(x_, tb, c, yes)
        if w == 1.0:
            return vc
        vu = net(x_, tb, c, no)
        return vu + w * (vc - vu)
    return euler(vfn, x, steps)


@torch.no_grad()
def sample_v_steps(net, n, ab, gen, dev, steps):
    """The v smoke's unconditional sampler with the DDIM step count exposed."""
    x = torch.randn(n, 3, D.RES, D.RES, device=dev, generator=gen)
    c, no = torch.zeros(n, 384, device=dev), torch.ones(n, dtype=torch.bool, device=dev)
    return SM.ddim_v(lambda x_, t: net(x_, t.repeat(n), c, no), x, ab, steps)


def step_selftest():
    """(a) D.sample at w = 3 costs two passes per step and equals the old three-pass expression; (b) Euler with the exact velocity
    of Gaussian data converges to the target and the step-count error is recorded; (c) the loss's own minimiser is the exact velocity."""
    out = {}
    calls = [0]

    class Stub(nn.Module):  # deterministic stand-in for the U-Net; counts passes
        def forward(self, x, t, c, drop):
            calls[0] += 1
            return torch.tanh(x + c.mean(1)[:, None, None, None] * (~drop).float()[:, None, None, None] + t[:, None, None, None] / 1000)
    net, ab = Stub(), D.schedule(torch.device("cpu"))
    cond = torch.randn(4, 384, generator=torch.Generator().manual_seed(1))
    g = lambda: torch.Generator().manual_seed(2)  # noqa: E731
    xs = D.sample(net, cond, ab, g(), w=3.0, steps=10)
    out["guided_passes_per_step"] = calls[0] / 10
    x = torch.randn(4, 3, D.RES, D.RES, generator=g())  # the old three-pass loop, verbatim
    ts = torch.linspace(D.T_STEPS - 1, 0, 10).long()
    yes, no = torch.zeros(4, dtype=torch.bool), torch.ones(4, dtype=torch.bool)
    for i, t in enumerate(ts):
        tb = t.repeat(4); ec = net(x, tb, cond, yes)
        e = net(x, tb, cond, no) + 3.0 * (ec - net(x, tb, cond, no))
        a = ab[t]; ap = ab[ts[i + 1]] if i + 1 < 10 else torch.tensor(1.0)
        x0 = ((x - (1 - a).sqrt() * e) / a.sqrt()).clamp(-2.0, 6.0)
        x = ap.sqrt() * x0 + (1 - ap).sqrt() * e
    out["max_abs_vs_old_loop"] = float((x - xs).abs().max())
    assert out["guided_passes_per_step"] == 2 and out["max_abs_vs_old_loop"] == 0
    for mu, s in ((0.0, 1.0), (0.0, 0.44), (0.5, 0.3), (-0.2, 1.5)):
        def v(x_, t, mu=mu, s=s):  # E[x0 − ε | x_t] for x0 ~ N(μ, s²)
            var = t * t * s * s + (1 - t) ** 2
            r = x_ - t * mu
            return mu + (t * s * s - (1 - t)) / var * r
        x0 = torch.randn(64, 3, 64, 64, generator=torch.Generator().manual_seed(0))
        out[f"N({mu},{s}^2)"] = {k: {"mean": float(euler(v, x0, k).mean()), "sd": float(euler(v, x0, k).std())} for k in (8, 16, 32, 100, 1000)}
        r = out[f"N({mu},{s}^2)"][1000]
        assert abs(r["mean"] - mu) < 0.01 and abs(r["sd"] / s - 1) < 0.01, out
    # (c) at fixed t the least-squares predictor of x0 − ε from x_t is that velocity (Monte Carlo slope and intercept)
    gen = torch.Generator().manual_seed(3)
    x0 = 0.5 + 0.3 * torch.randn(2_000_000, generator=gen); n = torch.randn(2_000_000, generator=gen)
    for t in (0.1, 0.5, 0.9):
        xt, y = t * x0 + (1 - t) * n, x0 - n
        b = float(((xt - xt.mean()) * (y - y.mean())).mean() / xt.var()); want = (t * .09 - (1 - t)) / (t * t * .09 + (1 - t) ** 2)
        out[f"slope_t{t}"] = [b, want]
        assert abs(b - want) < 0.02, out
    print(json.dumps(out, indent=1)); put("selftest", out)


def put(k, v):
    j = json.loads(JS.read_text()) if JS.exists() else {}
    j[k] = v
    JS.write_text(json.dumps(j, indent=1, default=float))


def step_recon():
    C = Coord80()
    cos = lambda a, b: (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)  # noqa: E731
    out = {"encoder": C.encoder}
    m = np.load(D.OUT / "train_bank.npz")["emb"].astype(np.float64).mean(0)
    for part in ("train", "test"):
        x = np.load(D.OUT / f"{part}_bank.npz")["emb"].astype(np.float64)
        q, z = C.q(x)
        xh = C.back(q * C.scale)
        s = (x - C.mu) / C.sd
        f = lambda v: {k: float(p) for k, p in zip(("p5", "median", "p95", "mean"), [*np.percentile(v, [5, 50, 95]), v.mean()])}  # noqa: E731
        out[part] = {"n": len(x), "cos_raw": f(cos(x, xh)), "cos_centred_on_train_mean": f(cos(x - m, xh - m)),
                     "cos_standardised": f(cos(s, (xh - C.mu) / C.sd)), "cos_float_vs_int8_centred": f(cos(C.back(z) - m, xh - m)),
                     "variance_kept_standardised": float(1 - ((s - (xh - C.mu) / C.sd) ** 2).sum() / ((s - s.mean(0)) ** 2).sum()),
                     "galaxies_with_a_clipped_coordinate": float((np.abs(z / C.scale) > 127.5).any(1).mean())}
    print(json.dumps(out, indent=1)); put("recon80", out)


def step_run(steps, bs, lr=2e-4):
    OUT.mkdir(parents=True, exist_ok=True)
    x64 = np.load(D.OUT / "train_x64.npy", mmap_mode="r")
    sky = np.load(SM.SKY)["sky"]
    emb = np.load(D.OUT / "train_bank.npz")["emb"]
    C80 = Cond80(emb)
    cond_all = C80(emb)
    del emb
    rng = np.random.default_rng(D.SEED + 21)
    fit = np.sort(rng.choice(len(x64), SM.N_FIT, replace=False))
    prep = SM.Prep(True, np.asarray(x64[fit], dtype=np.float32), sky[fit])  # fixes 1 and 2, identical draw to the v smoke
    dev = D.device()
    torch.manual_seed(D.SEED)
    net = D.UNet(cdim=80).to(dev)
    ema = D.UNet(cdim=80).to(dev)
    ema.load_state_dict(net.state_dict()); ema.requires_grad_(False)
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    gen = torch.Generator(device=dev).manual_seed(D.SEED)
    log = open(OUT / "train_log.jsonl", "a")
    log.write(json.dumps({"event": "start", "arm": "fm80", "params": npar, "batch": bs, "steps": steps, "prep": prep.state(), **D.machine()}) + "\n")
    t0, last, last_step, sm, start = time.perf_counter(), time.perf_counter(), 0, None, 1
    part = OUT / "partial.pt"
    if part.exists():
        pk = torch.load(part, map_location="cpu", weights_only=False)
        net.load_state_dict(pk["net"]); ema.load_state_dict(pk["ema"]); opt.load_state_dict(pk["opt"])
        rng.bit_generator.state = pk["rng"]; gen.set_state(pk["gen"])
        start, last_step, sm, t0 = pk["step"] + 1, pk["step"], pk["sm"], t0 - pk["elapsed"]
        log.write(json.dumps({"event": "resume", "from_step": pk["step"], **D.machine()}) + "\n")
    for step in range(start, steps + 1):
        i = np.sort(rng.choice(len(x64), bs, replace=False))
        x0 = torch.from_numpy(prep.fwd(np.asarray(x64[i], dtype=np.float32), sky[i])).to(dev)
        loss = loss_fm(net, x0, torch.from_numpy(cond_all[i]).to(dev), gen)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            d = min(0.999, (1 + step) / (10 + step))
            for pe, p in zip(ema.parameters(), net.parameters()):
                pe.mul_(d).add_(p.detach(), alpha=1 - d)
        if step % 50 == 0:
            lv = float(loss.detach()); sm = lv if sm is None else 0.9 * sm + 0.1 * lv
        if step % 250 == 0 or step == steps:
            now = time.perf_counter()
            rec = {"step": step, "loss_ema": sm, "steps_per_s_window": (step - last_step) / max(now - last, 1e-6), "rss_peak_gb": D.rss_gb(), **D.mps_mem(), **D.machine()}
            log.write(json.dumps(rec) + "\n"); log.flush(); last, last_step = now, step
            if step % 500 == 0 and step < steps:
                torch.save({"net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step, "rng": rng.bit_generator.state,
                            "gen": gen.get_state(), "sm": sm, "elapsed": time.perf_counter() - t0}, part)
            print(f"fm80 step {step} loss {sm and round(sm, 4)} {rec['steps_per_s_window']:.2f} it/s load {rec['load_avg_1_5_15'][0]}", file=sys.stderr, flush=True)
    torch.save({"ema": ema.state_dict(), "step": steps, "prep": prep.state(), "pred": "fm", "cond": {"m": C80.m, "s": C80.s}, "batch": bs,
                "train_seconds": time.perf_counter() - t0, "final_loss_ema": sm}, OUT / "smoke.pt")
    log.write(json.dumps({"event": "end", "train_seconds": time.perf_counter() - t0, **D.machine()}) + "\n"); log.close()


def load_prep(st):
    prep = SM.Prep.__new__(SM.Prep)
    for k in ("m", "sd", "lo", "hi", "sky_median"):
        setattr(prep, k, np.asarray(st[k], np.float32))
    prep.sky_on = st["sky_on"]
    return prep


def radial_power(z):
    """Mean log10 radial power per band (window: none), bins of |k| 1…31, over the 64 images."""
    f = np.abs(np.fft.fft2(z - z.mean((2, 3), keepdims=True))) ** 2
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(64) * 64, np.fft.fftfreq(64) * 64, indexing="ij"))
    return np.stack([np.log10(f[:, :, (k >= r - .5) & (k < r + .5)].mean((0, 2))) for r in range(1, 32)], 1)


def stop_rule(t):
    """The written stop rule in sky-subtracted model units: IQR ≤ 3× real in every band,
    whole-image SD ratio in [0.75, 1.35] (guards variance collapse), ≤ 25% offset > 1 SD, |mean| ≤ 0.2."""
    a, b = t["inputs"], t["samples_w0"]
    bands = list(a["ring"])
    iqr = [b["ring"][k]["per_image_level_iqr"] / a["ring"][k]["per_image_level_iqr"] for k in bands]
    sd_ratio = [b["whole"][k]["sd"] / a["whole"][k]["sd"] for k in bands]
    mean = [b["whole"][k]["mean"] for k in bands]
    return {"iqr_ratio": iqr, "sd_ratio": sd_ratio, "mean": mean, "offset": b["offset_gt_1sd"],
            "passes": bool(max(iqr) <= 3 and 0.75 <= min(sd_ratio) and max(sd_ratio) <= 1.35 and max(map(abs, mean)) <= 0.2 and b["offset_gt_1sd"] <= 0.25)}


def step_eval():
    """One process, EMA weights only, one model at a time; evaluates on held-out test_x64.npy."""
    ck = torch.load(OUT / "smoke.pt", map_location="cpu", weights_only=False)
    dev = D.device()
    test_x = D.OUT / "test_x64.npy"
    test_bank = D.OUT / "test_bank.npz"
    if not test_x.exists() or not test_bank.exists():
        raise SystemExit(f"eval requires held-out {test_x} and {test_bank} (refusing to evaluate on memorised train_x64.npy)")
    x64 = np.load(test_x, mmap_mode="r")
    ev = np.sort(np.random.default_rng(D.SEED + 31).choice(len(x64), 64, replace=False))
    x_in = np.asarray(x64[ev], dtype=np.float32)
    # Compute outer-ring sky per band directly on the held-out images (matches SM.SKY definition).
    sky_ev = np.median(x_in[:, :, SM.RING], axis=-1)
    res = {"steps": ck["step"], "train_seconds": ck["train_seconds"], "final_loss_ema": ck["final_loss_ema"], "eval_split": "test", "arms": {}}
    emb = np.load(test_bank)["emb"][ev]
    runs = []
    net = D.UNet(cdim=80).to(dev); net.load_state_dict(ck["ema"]); net.eval()
    prep = load_prep(ck["prep"])
    C80 = Coord80()
    cond = torch.from_numpy((((C80(emb)) - ck["cond"]["m"]) / ck["cond"]["s"]).astype(np.float32)).to(dev)
    z = {}
    for k in EULER:
        g = torch.Generator(device=dev).manual_seed(D.SEED + 11)
        z[f"fm80_euler{k}"] = torch.cat([sample_fm(net, 16, g, dev, k) for _ in range(4)]).float().cpu().numpy()
    g = torch.Generator(device=dev).manual_seed(D.SEED + 11)
    z["fm80_euler16_w1"] = torch.cat([sample_fm(net, 16, g, dev, 16, cond[i:i + 16]) for i in range(0, 64, 16)]).float().cpu().numpy()
    del net
    vk = torch.load(D.OUT / "smoke_full_v/smoke.pt", map_location="cpu", weights_only=False)
    vprep = load_prep(vk["prep"])
    net = D.UNet().to(dev); net.load_state_dict(vk["ema"]); net.eval()
    ab = D.schedule(dev)
    for k in EULER:
        g = torch.Generator(device=dev).manual_seed(D.SEED + 11)
        z[f"v384_ddim{k}"] = torch.cat([sample_v_steps(net, 16, ab, g, dev, k) for _ in range(4)]).float().cpu().numpy()
    np.savez(OUT / "eval_samples.npz", **z)
    same = {k: float(np.abs(getattr(prep, k) - getattr(vprep, k)).max()) for k in ("m", "sd", "sky_median")}
    assert max(same.values()) == 0, same  # both arms read the identical input transform
    z_in = prep.fwd(x_in, sky_ev)
    ref = radial_power(z_in)
    for name, a in z.items():
        st = SM.stats_table(z_in, a, x_in, prep.inv(a))
        pw = radial_power(a)
        res["arms"][name] = {"model_units": st["model_units"], "x_units_rendered": st["x_units_rendered"],
                             "log10_power_rms_vs_inputs_per_band": np.sqrt(((pw - ref) ** 2).mean(1)).tolist(),
                             "log10_power_bias_per_band": (pw - ref).mean(1).tolist(), "stop": stop_rule(st["model_units"])}
    res["machine_end"] = D.machine()
    put("eval", res)
    for n, r in res["arms"].items():
        s = r["stop"]
        print(n, "mean", np.round(s["mean"], 3), "sd", [round(v["sd"], 2) for v in r["model_units"]["samples_w0"]["whole"].values()], "iqr ratio", np.round(s["iqr_ratio"], 2), "offset", s["offset"], "PASS" if s["passes"] else "fail")


def step_sheet():
    from PIL import Image, ImageDraw, ImageFont
    f = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    ck = torch.load(OUT / "smoke.pt", map_location="cpu", weights_only=False)
    prep = load_prep(ck["prep"])
    z = np.load(OUT / "eval_samples.npz")
    ev = np.sort(np.random.default_rng(D.SEED + 31).choice(100_000, 64, replace=False))
    x = np.asarray(np.load(D.OUT / "train_x64.npy", mmap_mode="r")[ev], dtype=np.float32)
    rows = [("real", x[:8])] + [(n, prep.inv(z[k])[:8]) for k, n in (("fm80_euler8", "FM, 8 Euler"), ("fm80_euler16", "FM, 16 Euler"), ("fm80_euler32", "FM, 32 Euler"),
                                                                    ("v384_ddim8", "v, 8 DDIM"), ("v384_ddim16", "v, 16 DDIM"), ("v384_ddim100", "v, 100 DDIM"), ("fm80_euler16_w1", "FM 16, conditional"))]
    W, T = 130, 112
    im = Image.new("RGB", (W + 8 * T, len(rows) * T), (250, 249, 246)); dr = ImageDraw.Draw(im)
    for r, (n, a) in enumerate(rows):
        dr.text((8, r * T + T // 2 - 14), n if n == "real" else f"generated\n{n}", fill=(29, 27, 25) if n == "real" else (160, 60, 20), font=f)
        for c in range(8):
            im.paste(Image.fromarray(D.show(a[c])).resize((T, T), Image.NEAREST), (W + c * T, r * T))
    im.save(D.DOCS / "diffusion_e_samples.png", optimize=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["selftest", "recon", "run", "eval", "sheet"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    step_run(a.steps, a.batch) if a.step == "run" else globals()["step_" + a.step]()
