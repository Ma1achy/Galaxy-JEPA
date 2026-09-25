"""P2DFFT exploratory rerun on BB2's Stage-1 images — EXPLORATORY (user, 2026-09-25).

The pre-registered Stage-1 verdicts stand; nothing here changes them. Question: is the pile-up of
P2DFFT answers at 90° / 75.96° / 63.43° / 82.87° / 45° (arctan m/p at the lowest frequencies, p in
steps of 0.25) the method or the configuration? Four conditions on the 619 Stage-1 images
(`s1_q5_armless12` stays dropped), each in auto (-m) and oracle-m (-a arms) modes:

  base        the grid as run: noisy images, inner radius 1 (reproduces grid_s1_rows.csv p2_auto/p2_oracle)
  bulge       noisy images, starting inner radius beyond the bulge
  clean       noise-free copies (same spec, same h), inner radius 1
  clean_bulge noise-free copies, starting inner radius beyond the bulge
  arm, clean_arm  supplementary: starting radius at the arms' own start, ceil(R_IN·h) — the bulge
              crossing (3–7 px) lies inside it, so this separates "bulge too small to matter" from
              "the method"

p2pa sums each mode's complex spectrum over the annuli whose inner radius runs start…end (end = 90%
of the outer radius, Davis et al. 2012). `start` is the BAR FITS keyword when present — P2DFFT's own
starting-radius mechanism (Manual: "the BAR keyword value will control the starting inner radius").
The bulge conditions write BAR into a copy of each FITS and link the same .h5, so p2dfft is not rerun.

The starting radius is truth-derived (a best case, not a blind recipe): the first radius, moving
out, where the PSF-convolved disc's azimuthal profile reaches the PSF-convolved bulge's, rounded up, capped at
end − 5 (capped images are counted).

  uv run python artifacts/bb2_p2_explore.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bb2  # noqa: E402

SRC = bb2.OUT / "grid_s1"
WORK = bb2.OUT / "p2_explore"
PILE = (90.0, 75.96, 63.43, 82.87, 45.0)  # arctan(m/p): (1,0) (1,.25) (1,.5) (2,.25) (1,1)
PILE_TOL = 0.05
CONDS = ("base", "bulge", "clean", "clean_bulge", "arm", "clean_arm")  # arm*: supplementary
MODES = ("auto", "oracle")


def specs() -> list[bb2.Spec]:
    out = []
    for r in csv.DictReader((SRC / "truth.csv").open()):
        kw = {}
        for k, f in bb2.Spec.__dataclass_fields__.items():
            v = r[k]
            kw[k] = (int(float(v)) if f.type == "int" else float(v) if f.type == "float" else v)
        s = bb2.Spec(**kw)
        if np.isfinite(s.h_pix):
            out.append(s)
    return out


def _profile(img: np.ndarray) -> np.ndarray:
    y, x = np.indices(img.shape)
    r = np.hypot(x + 0.5 - bb2.N / 2, y + 0.5 - bb2.N / 2).astype(int)
    return np.bincount(r.ravel(), img.ravel()) / np.maximum(np.bincount(r.ravel()), 1)


def bulge_radius(s: bb2.Spec) -> int:
    """First radius (px), moving out, where the convolved disc outshines the convolved bulge.
    (The de Vaucouleurs wings overtake the exponential again far out — not the bulge a start excludes.)"""
    face = bb2.Spec(**{**asdict(s), "kind": "armless"})
    disc = _profile(bb2.observe(bb2.model(bb2.Spec(**{**asdict(face), "bt": 0.0}), s.h_pix, n=bb2.N * 2),
                                s.psf_fwhm, os=2)) * (1 - s.bt)
    bulge = _profile(bb2.observe(bb2.model(bb2.Spec(**{**asdict(face), "bt": 1.0}), s.h_pix, n=bb2.N * 2),
                                 s.psf_fwhm, os=2)) * s.bt
    below = np.nonzero(disc[: bb2.N // 2] >= bulge[: bb2.N // 2])[0]
    return int(below[0]) + 1 if below.size else bb2.N // 2


def clean_image(s: bb2.Spec) -> np.ndarray:
    """render() without the noise draw: same spec, same solved h."""
    face = bb2.Spec(**{**asdict(s), "kind": "armless" if s.kind == "noise" else s.kind})
    img = bb2.observe(bb2.model(face, s.h_pix), s.psf_fwhm) * 10 ** (-0.4 * (s.mag - 22.5))
    return img.astype(np.float32)


def with_bar(src: Path, dst: Path, sp: list[bb2.Spec], start: dict[str, int]) -> None:
    """FITS copies carrying BAR = starting radius; the .h5 linked, not recomputed."""
    from astropy.io import fits

    dst.mkdir(parents=True, exist_ok=True)
    for s in sp:
        data, hd = fits.getdata(src / f"{s.gid}.fits", header=True)
        hd["BAR"] = start[s.gid]
        fits.writeto(dst / f"{s.gid}.fits", data, hd, overwrite=True)
        link = dst / f"{s.gid}.h5"
        if not link.exists():
            os.symlink(src / f"{s.gid}.h5", link)


def linked(src: Path, dst: Path, sp: list[bb2.Spec]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for s in sp:
        for ext in (".fits", ".h5"):
            if not (dst / f"{s.gid}{ext}").exists():
                os.symlink(src / f"{s.gid}{ext}", dst / f"{s.gid}{ext}")


def p2pa_both(d: Path, sp: list[bb2.Spec]) -> dict[str, dict[str, float]]:
    stems = [s.gid for s in sp]
    auto = bb2.B.p2pa(d, stems)
    orc = bb2.B.p2pa(d, stems, arms={s.gid: max(1, s.arms) for s in sp})
    return {m: {g: abs(res.get(g, {}).get("pa", np.nan)) for g in stems} for m, res in (("auto", auto), ("oracle", orc))}


def summarise(sp: list[bb2.Spec], ans: dict[str, float]) -> dict:
    spi = [s for s in sp if s.kind == "spiral"]
    t = np.array([s.pitch for s in spi])
    m = np.array([ans[s.gid] for s in spi])
    q = np.array([s.quintile for s in spi])
    fin = np.isfinite(m)
    out = {"n_spiral": len(spi), "n_answer": int(fin.sum()),
           "pile": {f"{v:.2f}": float(np.mean(np.abs(m[fin] - v) <= PILE_TOL)) for v in PILE}}
    out["pile_total"] = float(sum(out["pile"].values()))
    out["quantiles"] = {str(k): float(np.percentile(m[fin], k)) for k in (5, 25, 50, 75, 95)}
    e = (m - t)[fin]
    out["rms"] = float(np.sqrt(np.mean(e ** 2)))
    out["median_err"] = float(np.median(e))
    out["median_by_truth"] = {str(int(p)): float(np.nanmedian(m[t == p])) for p in bb2.PITCHES}
    sub = fin & (t >= bb2.RANK_PITCH[0]) & (t <= bb2.RANK_PITCH[1])
    out["rho_pooled"] = bb2._rho(t, m, sub)
    out["rho_by_quintile"] = {str(k): bb2._rho(t, m, sub & (q == k)) for k in bb2.QUINTILES}
    nul = [ans[s.gid] for s in sp if s.kind == "armless" and np.isfinite(ans[s.gid])]
    out["armless_null"] = {"n": len(nul), "median": float(np.median(nul)) if nul else np.nan,
                           "pile_total": float(np.mean([min(abs(v - p) for p in PILE) <= PILE_TOL for v in nul]))
                           if nul else np.nan}
    return out


def figure(sp: list[bb2.Spec], ans: dict, res: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spi = [s for s in sp if s.kind == "spiral"]
    t = np.array([s.pitch for s in spi])
    fig, ax = plt.subplots(3, len(CONDS), figsize=(4.2 * len(CONDS), 11.5))
    bins = np.arange(0, 90.5, 0.5)
    for j, c in enumerate(CONDS):
        for i, mo in enumerate(MODES):
            m = np.array([ans[c][mo][s.gid] for s in spi])
            a = ax[i, j]
            a.hist(m[np.isfinite(m)], bins=bins, color="#4f81bd" if mo == "auto" else "#c0504d")
            for v in PILE:
                a.axvline(v, color="k", lw=0.5, ls=":")
            r = res[c][mo]
            a.set_title(f"{c} · {mo}: pile-up {100 * r['pile_total']:.0f}%, ρ(10–30) {r['rho_pooled']:.2f}",
                        fontsize=9)
            a.set_xlabel("|pitch| answered (°)", fontsize=8)
        a = ax[2, j]
        rng = np.random.default_rng(0)
        for mo, col, dx in (("auto", "#4f81bd", -0.8), ("oracle", "#c0504d", 0.8)):
            m = np.array([ans[c][mo][s.gid] for s in spi])
            a.scatter(t + dx + rng.uniform(-0.5, 0.5, t.size), m, s=4, alpha=0.35, color=col, label=mo)
            med = [np.nanmedian(m[t == p]) for p in bb2.PITCHES]
            a.plot(np.array(bb2.PITCHES) + dx, med, "o-", color=col, ms=4, lw=1.2)
        a.plot([0, 45], [0, 45], "k--", lw=0.7)
        a.set_xlim(0, 45)
        a.set_ylim(0, 92)
        a.set_xlabel("true pitch (°)", fontsize=8)
        a.set_ylabel("measured |pitch| (°)", fontsize=8)
        a.set_title(f"{c}: measured vs truth (median per pitch)", fontsize=9)
        a.legend(fontsize=7, loc="upper left")
    fig.suptitle("BB2 Stage 1 — P2DFFT exploratory rerun (EXPLORATORY; pre-registered verdicts stand). "
                 "Dotted: 90°, 82.87°, 75.96°, 63.43°, 45° = arctan(m/p) at the lowest frequencies.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = bb2.OUT / "p2_explore.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    sp = specs()
    print(f"[explore] {len(sp)} Stage-1 images", flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    start, arm_start, capped, raw = {}, {}, [], {}
    for s in sp:
        end = int(bb2.p2_outer(s) * 0.9)
        raw[s.gid] = bulge_radius(s) if s.kind != "noise" else 1
        start[s.gid] = min(raw[s.gid], end - 5)
        arm_start[s.gid] = min(int(np.ceil(bb2.R_IN * s.h_pix)), end - 5)
        if start[s.gid] < raw[s.gid]:
            capped.append(s.gid)
    ans: dict = {}
    linked(SRC / "fits", WORK / "base", sp)
    ans["base"] = p2pa_both(WORK / "base", sp)
    print("[explore] base done", flush=True)
    with_bar(SRC / "fits", WORK / "bulge", sp, start)
    ans["bulge"] = p2pa_both(WORK / "bulge", sp)
    print("[explore] bulge done", flush=True)
    # noise-free: the noise-only nulls would be all-zero images; they are dropped from both clean runs
    cl = [s for s in sp if s.kind != "noise"]
    (WORK / "clean").mkdir(exist_ok=True)
    for s in cl:
        p = WORK / "clean" / f"{s.gid}.fits"
        if not p.exists():
            bb2.write_fits(p, clean_image(s))
    bb2._p2(WORK, cl, "clean")
    ans["clean"] = p2pa_both(WORK / "clean", cl)
    print("[explore] clean done", flush=True)
    with_bar(WORK / "clean", WORK / "clean_bulge", cl, start)
    ans["clean_bulge"] = p2pa_both(WORK / "clean_bulge", cl)
    print("[explore] clean_bulge done", flush=True)
    with_bar(SRC / "fits", WORK / "arm", sp, arm_start)
    ans["arm"] = p2pa_both(WORK / "arm", sp)
    with_bar(WORK / "clean", WORK / "clean_arm", cl, arm_start)
    ans["clean_arm"] = p2pa_both(WORK / "clean_arm", cl)
    print("[explore] arm-start runs done", flush=True)
    for c in ("clean", "clean_bulge", "clean_arm"):
        for mo in MODES:
            ans[c][mo].update({s.gid: np.nan for s in sp if s.kind == "noise"})

    rows = {r["gid"]: r for r in csv.DictReader((bb2.OUT / "grid_s1_rows.csv").open())}
    check = {}
    for mo in MODES:
        a = np.array([ans["base"][mo][s.gid] for s in sp])
        b = np.array([float(rows[s.gid][f"p2_{mo}"]) for s in sp])
        both = np.isfinite(a) & np.isfinite(b)
        check[mo] = {"n": len(sp), "max_abs_diff": float(np.max(np.abs(a[both] - b[both]))),
                     "nan_mismatch": int((np.isfinite(a) != np.isfinite(b)).sum())}
    res = {c: {mo: summarise(sp, ans[c][mo]) for mo in MODES} for c in CONDS}
    st = np.array([start[s.gid] for s in sp if s.kind == "spiral"])
    arm0 = np.array([np.ceil(bb2.R_IN * s.h_pix) for s in sp if s.kind == "spiral"])
    rec = {"label": "EXPLORATORY — pre-registered Stage-1 verdicts stand",
           "reproduction_check": check,
           "start_radius": {"rule": "first r (outward) where PSF-convolved disc >= bulge (azimuthal), +1; cap end-5",
                            "spiral_quantiles_px": {str(k): float(np.percentile(st, k)) for k in (5, 25, 50, 75, 95)},
                            "arm_start_R_IN_h_quantiles_px": {str(k): float(np.percentile(arm0, k)) for k in (5, 25, 50, 75, 95)},
                            "n_capped": len(capped), "capped": capped},
           "results": res,
           "answers": {c: {mo: {g: (None if not np.isfinite(v) else v) for g, v in ans[c][mo].items()}
                           for mo in MODES} for c in CONDS}}
    rec["figure"] = str(figure(sp, ans, res))
    (bb2.OUT / "p2_explore.json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps({"check": check, "start": {k: v for k, v in rec["start_radius"].items() if k != "capped"}},
                     indent=1))
    for c in CONDS:
        for mo in MODES:
            r = res[c][mo]
            print(f"{c:12s} {mo:6s} pile {r['pile_total']:.3f} {r['pile']} rms {r['rms']:.1f} "
                  f"rho {r['rho_pooled']:.3f} q {[round(v, 2) for v in r['rho_by_quintile'].values()]} "
                  f"quant {[round(v, 1) for v in r['quantiles'].values()]}")
            print(f"{'':19s} median by truth {({k: round(v, 1) for k, v in r['median_by_truth'].items()})} "
                  f"armless {r['armless_null']}")


if __name__ == "__main__":
    main()
