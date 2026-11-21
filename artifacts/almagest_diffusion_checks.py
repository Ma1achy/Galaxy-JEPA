"""Almagest diffusion — two cheap checks before any new training (user brief, 21 Nov 2026).

VISUALISATION ONLY, branch `almagest-geometry`. Reads the pilot's banks and checkpoint (never writes them):

    .venv/bin/python artifacts/almagest_diffusion_checks.py norm   # 1. normalisation: inputs as fed vs w = 0 samples
    .venv/bin/python artifacts/almagest_diffusion_checks.py crop   # 2. ceiling of a native 128-px centre crop, padded

1. Per-channel mean / SD of the pilot's training data as fed (model units y = asinh(x)/0.8, 64 px) and in
   normalised-stamp units x, over the whole image and over the background (outside the 2.5 × R_petro box);
   the same for 64 unconditional samples (w = 0), whose background is the outer ring (no R_petro exists).
2. Fraction of the 100k whose 2.5 × R_petro box fits a 128 × 128 native crop; A1 re-embedding of the crop
   padded back to 256 px with sky noise matched to that stamp's measured background; cosine and rank against
   the native embedding, beside the pilot's 64-px round trip on the same 256 galaxies; a pure-background plant.

Investigation code: terse, excluded from lint/CI. Writes runs/almagest/diffusion/checks/ and docs/almagest/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import almagest_diffusion as D  # noqa: E402  (constants and helpers only; nothing runs at import)

EMB_BS = 32  # A1 embedding batch on MPS, small for memory (the pilot used 128)

CK = D.OUT / "checks"
PIX = 0.396
K_BOX, GLOBAL_FRAC = 2.5, 0.4  # configs/pretrain_v2.yaml masking: petro_k, global_box_frac (fallback box)
CROP = 128
BANDS = ("g", "r", "i")


def petro_of(ids) -> np.ndarray:
    from galaxy_jepa.data.cache import TensorCache, load_scalars
    tc = TensorCache(D.CACHE)
    sc = load_scalars(D.CACHE, tc.index)
    return np.array([sc[tc.row_of(int(o))] for o in ids], dtype=np.float64)


def half_px(petro: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """bbox.petrosian_box, vectorised: k·R/0.396 clamped to 128; non-finite/≤0 → global 0.4 × 256."""
    bad = ~np.isfinite(petro) | (petro <= 0)
    h = np.where(bad, GLOBAL_FRAC * 256, K_BOX * np.where(bad, 1.0, petro) / PIX)
    return np.minimum(h, 128.0), bad


def outside_box(n: int, half: float) -> np.ndarray:
    """Pixel centres of an n-px image (covering the 256-px stamp) beyond the centred box of half-width `half`."""
    c = (np.arange(n) + 0.5) * (256 / n) - 128
    return np.maximum(np.abs(c)[:, None], np.abs(c)[None, :]) > half


def ring(n: int, w: int) -> np.ndarray:
    m = np.zeros((n, n), bool)
    m[:w], m[-w:], m[:, :w], m[:, -w:] = True, True, True, True
    return m


def chan_stats(x: np.ndarray, masks: list[np.ndarray] | None = None) -> dict:
    """x (B, 3, H, W). Pooled mean/SD per channel; with masks, the background's per-image level and noise."""
    out = {}
    for c, b in enumerate(BANDS):
        v = x[:, c]
        if masks is None:
            out[b] = {"mean": float(v.mean()), "sd": float(v.std())}
            continue
        px = [v[i][m] for i, m in enumerate(masks) if m.any()]
        lvl = np.array([np.median(p) for p in px])
        noise = np.array([1.4826 * np.median(np.abs(p - np.median(p))) for p in px])
        allp = np.concatenate(px)
        out[b] = {"mean": float(allp.mean()), "sd": float(allp.std()),
                  "per_image_level_median": float(np.median(lvl)),
                  "per_image_level_sd": float(lvl.std()),
                  "per_image_level_iqr": float(np.subtract(*np.percentile(lvl, [75, 25]))),
                  "per_image_noise_median": float(np.median(noise)), "n_images": len(px)}
    return out


def offsets_over_noise(x: np.ndarray, masks: list[np.ndarray], ref_level: np.ndarray) -> float:
    """Fraction of images with any channel's background level > 1 own-noise SD from the reference level."""
    hit = []
    for i, m in enumerate(masks):
        if not m.any():
            continue
        p = x[i][:, m]
        lv = np.median(p, 1)
        nz = 1.4826 * np.median(np.abs(p - lv[:, None]), 1)
        hit.append(bool((np.abs(lv - ref_level) > nz).any()))
    return float(np.mean(hit))


@torch.no_grad()
def sample_uncond(net, n, ab, gen, dev, steps=100):
    """DDIM η = 0, unconditional branch only (w = 0: ε = ε_u, one forward per step), the pilot's x0 clamp."""
    x = torch.randn(n, 3, D.RES, D.RES, device=dev, generator=gen)
    ts = torch.linspace(D.T_STEPS - 1, 0, steps, device=dev).long()
    c = torch.zeros(n, 384, device=dev)
    no = torch.ones(n, dtype=torch.bool, device=dev)
    for i, t in enumerate(ts):
        e = net(x, t.repeat(n), c, no)
        a = ab[t]
        a_prev = ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
        x0 = ((x - (1 - a).sqrt() * e) / a.sqrt()).clamp(-2.0, 6.0)
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * e
    return x


def step_norm() -> dict:
    t0 = time.perf_counter()
    tr = np.load(D.OUT / "train_bank.npz")
    x64 = np.load(D.OUT / "train_x64.npy", mmap_mode="r")
    rng = np.random.default_rng(D.SEED + 7)
    res = {"units": "y = asinh(x)/0.8 is what the U-Net is fed; x is the normalised stamp (A1's input) 4×4-pooled"}
    pet_all = petro_of(tr["ids"])
    for name, n in (("batch_32", 32), ("sample_4096", 4096)):
        i = np.sort(rng.choice(len(tr["ids"]), n, replace=False))
        x = np.asarray(x64[i], dtype=np.float32)
        y = D.to_model(x)
        h, bad = half_px(pet_all[i])
        masks = [outside_box(D.RES, hh) for hh in h]
        res[f"inputs_{name}"] = {
            "y_whole": chan_stats(y), "x_whole": chan_stats(x),
            "y_background_outside_2.5R": chan_stats(y, masks), "x_background_outside_2.5R": chan_stats(x, masks),
            "y_border_ring_8px": chan_stats(y, [ring(D.RES, 8)] * n),
            "n_no_background_pixels": int(sum(not m.any() for m in masks)), "n_petro_fallback": int(bad.sum())}
    # what fraction of the 64-px border ring lies outside the box: is the ring a fair background for the samples?
    h_all, _ = half_px(pet_all)
    res["border_ring_outside_box_fraction"] = float(np.mean(h_all < 128 - 32))
    dev = D.device()
    ck = torch.load(D.OUT / "pilot.pt", map_location="cpu")
    net = D.UNet().to(dev)
    net.load_state_dict(ck["ema"])
    net.eval()
    ab = D.schedule(dev)
    gen = torch.Generator(device=dev).manual_seed(D.SEED + 11)
    ts = time.perf_counter()
    ys = torch.cat([sample_uncond(net, 16, ab, gen, dev).float() for _ in range(4)])  # batches of 16: the Mac is near its swap edge
    if dev.type == "mps":
        torch.mps.synchronize()
    res["samples_seconds"] = time.perf_counter() - ts
    ys = ys.cpu().numpy()
    xs = D.from_model(ys)
    rmask = [ring(D.RES, 8)] * len(ys)
    res["samples_w0_64"] = {"checkpoint_step": int(ck["step"]), "y_whole": chan_stats(ys), "x_whole": chan_stats(xs),
                           "y_border_ring_8px": chan_stats(ys, rmask), "x_border_ring_8px": chan_stats(xs, rmask),
                           "x0_clamp": "(-2, 6) in y, as the pilot"}
    xin = np.asarray(x64[np.sort(rng.choice(len(tr["ids"]), 64, replace=False))], dtype=np.float32)
    lvl_ref = np.median(xin[:, :, ring(D.RES, 8)], (0, 2))
    res["fraction_with_channel_offset_gt_1_noise_sd"] = {
        "inputs_64": offsets_over_noise(xin, [ring(D.RES, 8)] * 64, lvl_ref),
        "samples_w0_64": offsets_over_noise(xs, rmask, lvl_ref),
        "note": "border ring; level compared with the inputs' median ring level, in units of each image's own ring noise"}
    np.save(CK / "samples_w0_64_y.npy", ys.astype(np.float32))
    res["seconds"] = time.perf_counter() - t0
    return res


def bg_stats(st: np.ndarray, half: float) -> tuple[np.ndarray, np.ndarray, str, int]:
    """Per-band sky mean/SD of one 256-px stamp: outside the 2.5R box, 3σ-clipped (stars), else the outer 16 px."""
    m, src = outside_box(256, half), "outside_box"
    if m.sum() < 2000:
        m, src = ring(256, 16), "ring16_fallback"
    mu, sd = np.zeros(3), np.zeros(3)
    for c in range(3):
        p = st[c][m].astype(np.float64)
        for _ in range(5):
            mm, ss = p.mean(), p.std()
            p = p[np.abs(p - mm) < 3 * ss]
        mu[c], sd[c] = p.mean(), p.std()
    return mu, sd, src, int(m.sum())


def pad_crop(st: np.ndarray, mu: np.ndarray, sd: np.ndarray, rng) -> np.ndarray:
    out = (mu[:, None, None] + sd[:, None, None] * rng.standard_normal((3, 256, 256))).astype(np.float32)
    a = (256 - CROP) // 2
    out[:, a:a + CROP, a:a + CROP] = st[:, a:a + CROP, a:a + CROP]
    return out


def ring_stats_of_crop(st: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The same clipped sky estimate from the crop's own outer 8 px: what a generated crop would have to go on."""
    a = (256 - CROP) // 2
    c = st[:, a:a + CROP, a:a + CROP]
    m = ring(CROP, 8)
    mu, sd = np.zeros(3), np.zeros(3)
    for k in range(3):
        p = c[k][m].astype(np.float64)
        for _ in range(5):
            mm, ss = p.mean(), p.std()
            p = p[np.abs(p - mm) < 3 * ss]
        mu[k], sd[k] = p.mean(), p.std()
    return mu, sd


def unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def compare(e_new: np.ndarray, e_nat: np.ndarray, mu: np.ndarray) -> dict:
    """Cosine to the own native embedding and the pilot's rank: own native galaxy among all n by centred cosine."""
    S = unit(e_new - mu) @ unit(e_nat - mu).T
    rank = 1 + (S > np.diag(S)[:, None]).sum(1)
    pr = np.random.default_rng(3).permutation(len(e_new))
    return {"cos_to_own_raw_median": float(np.median(np.sum(unit(e_new) * unit(e_nat), 1))),
            "cos_to_own_centred_median": float(np.median(np.diag(S))),
            "rank_median": float(np.median(rank)), "rank_mean": float(rank.mean()),
            "top1": float(np.mean(rank == 1)), "top5": float(np.mean(rank <= 5)),
            "random_pairs_among_these_centred_median": float(np.median(np.sum(unit(e_new - mu) * unit(e_new[pr] - mu), 1)))}


def step_crop() -> dict:
    from galaxy_jepa.data.cache import TensorCache
    t0 = time.perf_counter()
    tr, te = np.load(D.OUT / "train_bank.npz"), np.load(D.OUT / "test_bank.npz")
    mu = tr["emb"].mean(0)
    res = {"crop": f"{CROP} × {CROP} centre crop at native 0.396″/px; padded to 256 with Gaussian sky per band"}
    h_tr, bad_tr = half_px(petro_of(tr["ids"]))
    res["fits_in_crop_train_100k"] = {"fraction_box_half_le_64": float(np.mean(h_tr <= CROP / 2)),
                                      "fraction_excluding_petro_fallback": float(np.mean(h_tr[~bad_tr] <= CROP / 2)),
                                      "n": int(h_tr.size), "n_petro_fallback": int(bad_tr.sum()),
                                      "box_half_px_percentiles_5_25_50_75_95": [float(v) for v in np.percentile(h_tr, [5, 25, 50, 75, 95])]}
    pick = np.sort(np.random.default_rng(D.SEED + 1).choice(len(te["ids"]), 256, replace=False))  # the pilot eval's 256
    ids = te["ids"][pick]
    h_te, bad_te = half_px(petro_of(ids))
    tc = TensorCache(D.CACHE)
    rows = np.array([tc.row_of(int(o)) for o in ids])
    o = np.argsort(rows)
    st = np.empty((256, 3, 256, 256), np.float32)
    st[o] = np.asarray(tc.data[rows[o]], dtype=np.float32)
    dev = D.device()
    enc = D.a1_encoder(dev)
    e_nat = D.embed(enc, st, dev, bs=EMB_BS)
    res["native_reembed_max_abs_diff_vs_bank"] = float(np.abs(e_nat - te["emb"][pick]).max())
    rng = np.random.default_rng(D.SEED + 13)
    bg = [bg_stats(st[i], h_te[i]) for i in range(256)]
    pad_full = np.stack([pad_crop(st[i], bg[i][0], bg[i][1], rng) for i in range(256)])
    own = [ring_stats_of_crop(st[i]) for i in range(256)]
    pad_own = np.stack([pad_crop(st[i], own[i][0], own[i][1], rng) for i in range(256)])
    e_pf, e_po = D.embed(enc, pad_full, dev, bs=EMB_BS), D.embed(enc, pad_own, dev, bs=EMB_BS)
    e_rt = te["emb_roundtrip"][pick]
    fits = h_te <= CROP / 2
    res["test_256"] = {
        "n_fits_in_crop": int(fits.sum()), "n_petro_fallback": int(bad_te.sum()),
        "bg_source_counts": {s: int(sum(b[2] == s for b in bg)) for s in ("outside_box", "ring16_fallback")},
        "padded_crop_bg_from_full_stamp": compare(e_pf, e_nat, mu),
        "padded_crop_bg_from_crop_ring": compare(e_po, e_nat, mu),
        "roundtrip_64px_pilot": compare(e_rt, e_nat, mu),
        "native_random_pairs_centred_median": compare(e_nat, e_nat, mu)["random_pairs_among_these_centred_median"],
        "by_fit": {k: {"n": int(m.sum()),
                       "padded_bg_full_cos_centred_median": float(np.median(np.sum(unit(e_pf[m] - mu) * unit(e_nat[m] - mu), 1))),
                       "roundtrip_64_cos_centred_median": float(np.median(np.sum(unit(e_rt[m] - mu) * unit(e_nat[m] - mu), 1)))}
                   for k, m in (("box_fits", fits), ("box_exceeds", ~fits))}}
    # planted: pure Gaussian sky with each stamp's own measured sky; crop + pad must land where a fresh full draw lands
    P = 64
    sky = lambda i: (bg[i][0][:, None, None] + bg[i][1][:, None, None] * rng.standard_normal((3, 256, 256))).astype(np.float32)  # noqa: E731
    S0 = np.stack([sky(i) for i in range(P)])
    S1 = np.stack([sky(i) for i in range(P)])
    Sp = np.stack([pad_crop(S0[i], bg[i][0], bg[i][1], rng) for i in range(P)])
    e0, e1, ep = (D.embed(enc, s, dev, bs=EMB_BS) for s in (S0, S1, Sp))
    cc = lambda a, b: np.sum(unit(a - mu) * unit(b - mu), 1)  # noqa: E731
    res["planted_pure_background"] = {
        "construction": "S0 = Gaussian sky with stamp i's clipped mean/SD per band (i < 64); padded = S0's centre 128 "
                        "inside fresh sky of the same parameters; reference = S1, an independent full draw",
        "pass_rule": "declared before the run: median centred cos(S0, padded) ≥ median cos(S0, S1) − 0.02",
        "cos_S0_padded_centred_median": float(np.median(cc(e0, ep))),
        "cos_S0_S1_centred_median": float(np.median(cc(e0, e1))),
        "cos_S0_padded_raw_median": float(np.median(np.sum(unit(e0) * unit(ep), 1))),
        "cos_S0_own_galaxy_centred_median": float(np.median(cc(e0, e_nat[:P]))),
    }
    pl = res["planted_pure_background"]
    pl["pass"] = bool(pl["cos_S0_padded_centred_median"] >= pl["cos_S0_S1_centred_median"] - 0.02)
    # the plant can fail: pad with the WRONG sky (level + 1 SD, SD × 2); this must land below the fresh-draw reference
    Sw = np.stack([pad_crop(S0[i], bg[i][0] + bg[i][1], 2 * bg[i][1], rng) for i in range(P)])
    ew = D.embed(enc, Sw, dev, bs=EMB_BS)
    pl["mismatch_arm"] = {"construction": "S0's centre 128 inside sky with level + 1 SD and SD × 2",
                          "cos_S0_mismatched_pad_centred_median": float(np.median(cc(e0, ew))),
                          "fires": bool(np.median(cc(e0, ew)) < pl["cos_S0_S1_centred_median"] - 0.02)}
    np.savez(CK / "crop_embeddings.npz", ids=ids, native=e_nat, padded_bg_full=e_pf, padded_bg_crop_ring=e_po,
             roundtrip64=e_rt, box_half_px=h_te, sky_mu=np.array([b[0] for b in bg]), sky_sd=np.array([b[1] for b in bg]))
    sheet(st[:8], pad_full[:8])
    res["seconds"] = time.perf_counter() - t0
    return res


def sheet(st, pad) -> None:
    from PIL import Image
    T, G = 256, 8
    im = Image.new("RGB", (2 * T + 3 * G, len(st) * (T + G) + G), (250, 249, 246))
    for k in range(len(st)):
        for j, x in enumerate((st[k], pad[k])):
            im.paste(Image.fromarray(D.show(x)), (G + j * (T + G), G + k * (T + G)))
    im.save(D.DOCS / "diffusion_checks_padded_crops.png", optimize=True)


def main() -> None:
    CK.mkdir(parents=True, exist_ok=True)
    step = sys.argv[1]
    res = {"norm": step_norm, "crop": step_crop}[step]()
    res["machine"] = D.machine()
    p = D.DOCS / "diffusion_checks.json"
    allr = json.loads(p.read_text()) if p.exists() else {}
    allr[step] = res
    p.write_text(json.dumps(allr, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
