"""Brief BB2 — synthetic SDSS-matched spirals of known pitch, both methods, one harness.

Stages (bb_findings.md §BB2): 1 clean face-on; 2 inclined, true ellipse supplied; 3 inclined and
flocculent / broken / barred. Nulls (noise-only, armless disc) in every stage. Each synthetic borrows
its degradation from a real "donor" spiral and inherits that donor's V1 visibility score.

  --plant        D28: generator validity, ellipse plumbing, null-range plumbing, state logic
  --pilot        ~300 Stage 1 images: timing and detection rates only (no accuracy is computed)
  --grid ...     full run (after the go-ahead)
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import brentq
from scipy.signal import fftconvolve
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import bb_pitch as B  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = B.OUT / "bb2"
SF_OUT = B.TOOLS / "sparcfire-docker" / "out"
RUNNER = REPO / "tools" / "sparcfire-docker" / "run-in-image.sh"
PIX = 0.396  # arcsec / pixel, native SDSS (no rebin)
N = 256  # stamp size, the corpus's frozen spec
OS = 4  # oversampling for the model before PSF + pixel integration
VIS_COLS = ("modelMag_r", "snr_r", "petroRad_r", "specz")  # V1's composite; magnitude first

# ── the grid's factor levels (pre-registered) ──────────────────────────────────────────────────
PITCHES = (5, 10, 15, 20, 25, 30, 40)
ARMS = (1, 2, 3, 4)
CONTRASTS = (0.3, 0.6, 1.2)  # arm/inter-arm intensity ratio 1 + A: 1.3, 1.6, 2.2
QUINTILES = (1, 2, 3, 4, 5)  # V1 quintiles over the donor pool, 1 = most visible
VARIANTS3 = ("flocculent", "broken", "barred")
FEATHER = 4  # arm profile ((1 + cos mψ)/2)^FEATHER
R_IN, R_OUT = 0.75, 4.0  # arm extent in disc scale lengths (barred: r_in = bar half-length)
BAR_L = 1.0  # bar half-length in scale lengths
RE_BULGE = 0.2  # bulge effective radius in scale lengths
Q_MIN = 0.3  # inclined stages: q = max(expAB_r, 0.3)


@dataclass
class Spec:
    gid: str
    stage: int
    kind: str  # "spiral" | "noise" | "armless"
    pitch: float
    arms: int
    contrast: float
    variant: str  # "clean" | flocculent | broken | barred
    chirality: int  # +1 / −1
    q: float
    pa_deg: float  # major-axis angle, counter-clockwise from +column, array frame
    bt: float
    donor: str
    v1: float
    quintile: int
    psf_fwhm: float  # arcsec
    mag: float
    snr: float
    petro_rad: float  # arcsec
    seed: int
    sky: float = np.nan  # donor's r-band sky σ, nanomaggies (stamp border, robust)
    h_pix: float = np.nan  # disc scale length, pixels (solved)
    snr_synth: float = np.nan  # matched-filter SNR of the rendered galaxy under its noise model


# ── donors and V1 ──────────────────────────────────────────────────────────────────────────────


def donor_pool() -> tuple[list[dict], dict]:
    """Y's spiral population with all four V1 covariates valid; V1 = composite() over the pool."""
    import pandas as pd

    import v1_loose_ends as V1

    t = pd.read_csv(REPO / "data" / "probe" / "metadata.csv", dtype={"object_id": str, "dr7objid": str},
                    low_memory=False)
    f = lambda c: pd.to_numeric(t[c], errors="coerce").to_numpy(float)  # noqa: E731
    spir = ((f("t01_smooth_or_features_a02_features_or_disk_fraction") >= 0.5)
            & (f("t02_edgeon_a05_no_fraction") >= 0.5) & (f("t04_spiral_a08_spiral_fraction") >= 0.5))
    vis = np.column_stack([f(c) for c in VIS_COLS])
    pool = spir & np.isfinite(vis).all(axis=1)
    v1, info = V1.composite(vis[pool])
    p = t[pool].assign(v1=v1)
    edges = np.quantile(v1, [0.2, 0.4, 0.6, 0.8])
    p["quintile"] = 1 + np.searchsorted(edges, p.v1.to_numpy(), side="right")
    ok = (np.isfinite(pd.to_numeric(p.psfWidth_r, errors="coerce")) & np.isfinite(pd.to_numeric(p.expAB_r, errors="coerce"))
          & (pd.to_numeric(p.petroRad_r, errors="coerce") > 0)
          & ~p.petrorad_suspect.astype(str).str.lower().isin(("true", "1")))
    info.update({"n_pool": int(pool.sum()), "n_donor_eligible": int(ok.sum()), "quintile_edges": edges.tolist()})
    cols = ["object_id", "v1", "quintile", "psfWidth_r", "modelMag_r", "snr_r", "petroRad_r", "expAB_r", "specz"]
    return p.loc[ok, cols].to_dict("records"), info


def v1_place(pool_vis: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Place galaxies outside the pool on the pool's V1 axis (ECDF ranks, pool loadings)."""
    import v1_loose_ends as V1

    _, info = V1.composite(pool_vis)
    load = np.array(list(info["loadings"].values()))
    z = []
    for j in range(pool_vis.shape[1]):
        r = rankdata(pool_vis[:, j])
        s = np.sort(pool_vis[:, j])
        ro = (np.searchsorted(s, other[:, j], "left") + np.searchsorted(s, other[:, j], "right") + 1) / 2
        z.append((ro - r.mean()) / r.std())
    return np.column_stack(z) @ load


# ── the model ──────────────────────────────────────────────────────────────────────────────────


def _face_coords(n: int, scale: float, q: float, pa: float) -> tuple[np.ndarray, np.ndarray]:
    """Face-on (u, v) in pixels for an n×n array whose pixel is `scale` output pixels."""
    c = (np.arange(n) + 0.5) * scale - N / 2
    x, y = np.meshgrid(c, c)  # x = column, y = row
    ca, sa = np.cos(np.radians(pa)), np.sin(np.radians(pa))
    xp, yp = x * ca + y * sa, -x * sa + y * ca
    return xp, yp / q


def model(s: Spec, h: float, n: int = N * OS, with_arms: bool = True) -> np.ndarray:
    """Unconvolved surface brightness (arbitrary units), sampled on an n×n grid."""
    scale = N / n
    rng = np.random.default_rng(s.seed)
    q = s.q if s.stage > 1 else 1.0
    u, v = _face_coords(n, scale, q, s.pa_deg)
    r = np.hypot(u, v) + 1e-9
    th = np.arctan2(v, u)
    disc = np.exp(-r / h)
    arm = np.zeros_like(r)
    barred = s.variant == "barred"
    theta0 = rng.uniform(0, 2 * np.pi)
    r_in = (BAR_L if barred else R_IN) * h
    if with_arms and s.kind == "spiral":
        tanp = np.tan(np.radians(s.pitch))
        psi = th - s.chirality * np.log(r / r_in) / tanp - theta0
        f = ((1 + np.cos(s.arms * psi)) / 2) ** FEATHER
        win = 0.5 * (1 + np.tanh((r - r_in) / (0.2 * h))) * 0.5 * (1 - np.tanh((r - R_OUT * h) / (0.5 * h)))
        if s.variant == "flocculent":
            g = gaussian_filter(rng.standard_normal(r.shape), 0.3 * h / scale)
            g /= g.std()
            f = f * np.exp(0.7 * g - 0.245)
        elif s.variant == "broken":
            k = np.floor((s.arms * psi / (2 * np.pi)) + 0.5) % s.arms  # which arm
            lr = np.log(np.maximum(r, r_in) / r_in)
            mask = np.ones_like(r)
            span = np.log(R_OUT / (BAR_L if barred else R_IN))
            for a in range(s.arms):
                for _ in range(rng.integers(2, 4)):
                    c0 = rng.uniform(0, span)
                    gap = 0.5 * (np.tanh((np.abs(lr - c0) - 0.125) / 0.03) + 1)
                    mask = np.where(k == a, mask * gap, mask)
            f = f * mask
        arm = s.contrast * win * f
    img = disc * (1 + arm)
    if barred:
        ba = theta0
        cb, sb = np.cos(ba), np.sin(ba)
        bu, bv = u * cb + v * sb, -u * sb + v * cb
        bar = np.exp(-0.5 * ((bu / (0.5 * BAR_L * h)) ** 2 + (bv / (0.125 * BAR_L * h)) ** 2))
        img = img + 0.1 * img.sum() / bar.sum() * bar
    if s.kind == "noise":
        return np.zeros_like(r)
    xb = np.hypot(*_sky(n, scale))
    bulge = np.exp(-7.669 * ((xb + 1e-9) / (RE_BULGE * h)) ** 0.25)
    img = img / img.sum() * (1 - s.bt) + bulge / bulge.sum() * s.bt
    return img


def _sky(n: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    c = (np.arange(n) + 0.5) * scale - N / 2
    return np.meshgrid(c, c)


def observe(img_os: np.ndarray, psf_fwhm: float, os: int = OS) -> np.ndarray:
    """PSF (Gaussian, FWHM = psfWidth_r) on the oversampled grid, then integrate to 0.396″ pixels."""
    sig = psf_fwhm / 2.3548 / PIX * os
    k = int(np.ceil(5 * sig))
    g = np.exp(-0.5 * (np.arange(-k, k + 1) / sig) ** 2)
    g /= g.sum()
    conv = fftconvolve(fftconvolve(img_os, g[None, :], mode="same"), g[:, None], mode="same")
    return conv.reshape(N, os, N, os).sum(axis=(1, 3))


def petrosian_radius(img: np.ndarray) -> float:
    """SDSS Petrosian radius (η = 0.2; ratio over 0.8–1.25 R), pixels, on a circular profile."""
    y, x = np.indices(img.shape)
    r = np.hypot(x + 0.5 - N / 2, y + 0.5 - N / 2)
    edges = np.arange(0, N / 2, 0.5)
    ann = np.bincount(np.digitize(r.ravel(), edges), weights=img.ravel(), minlength=edges.size + 1)
    cum = np.cumsum(ann)[1:edges.size]  # flux inside edges[1:]
    area = np.pi * edges[1:] ** 2
    radii = edges[1:]

    def eta(R):
        f = lambda rr: np.interp(rr, radii, cum)  # noqa: E731
        sb = (f(1.25 * R) - f(0.8 * R)) / (np.pi * ((1.25 * R) ** 2 - (0.8 * R) ** 2))
        return sb / (f(R) / (np.pi * R ** 2)) - 0.2

    lo, hi = 1.0, N / 2 / 1.25 - 1
    if not eta(hi) < 0:
        return float("inf")  # the ratio never falls to 0.2 inside the stamp
    return brentq(eta, lo, hi)


def solve_h(s: Spec) -> float:
    """Disc scale length (pixels) so the PSF-convolved face-on model's Petrosian radius = donor's."""
    face = Spec(**{**asdict(s), "stage": 1, "kind": "spiral" if s.kind == "noise" else s.kind})
    target = s.petro_rad / PIX

    cap = N / 2 / 1.25 - 1

    def err(logh):
        rp = petrosian_radius(observe(model(face, np.exp(logh), n=N * 2), s.psf_fwhm, os=2))
        return min(rp, cap) - target

    if target >= cap:
        return float("nan")  # larger than a 256-px stamp can measure: dropped and counted

    try:
        return float(np.exp(brentq(err, np.log(0.3), np.log(60.0), xtol=1e-3)))
    except ValueError:
        return float("nan")


E_PER_NMGY = 4.7 / 0.005  # nominal SDSS r: gain ~4.7 e⁻/DN, ~0.005 nMgy/DN (varies by frame)


def donor_sky(object_id: str) -> float:
    """Robust σ of the donor stamp's r-band border (16 px frame; zero-filled pixels excluded)."""
    from astropy.io import fits

    st = fits.getdata(REPO / "data" / "probe" / f"{object_id}.fits")[1]
    b = np.r_[st[:16].ravel(), st[-16:].ravel(), st[16:-16, :16].ravel(), st[16:-16, -16:].ravel()]
    b = b[b != 0]
    return float(1.4826 * np.median(np.abs(b - np.median(b)))) if b.size > 1000 else float("nan")


def render(s: Spec) -> tuple[np.ndarray, Spec]:
    """Noisy image in nanomaggies: the donor's own sky σ plus Poisson noise from the galaxy."""
    h = solve_h(s)
    s.h_pix = h
    face = Spec(**{**asdict(s), "kind": "armless" if s.kind == "noise" else s.kind})
    clean = observe(model(face, h), s.psf_fwhm) * 10 ** (-0.4 * (s.mag - 22.5))
    var = s.sky ** 2 + np.clip(clean, 0, None) / E_PER_NMGY
    s.snr_synth = float(np.sqrt(np.sum(clean ** 2 / var)))
    if s.kind == "noise":
        clean, var = np.zeros_like(clean), np.full_like(clean, s.sky ** 2)
    rng = np.random.default_rng(s.seed + 7)
    return (clean + rng.normal(0, 1, clean.shape) * np.sqrt(var)).astype(np.float32), s


def write_fits(path: Path, img: np.ndarray) -> None:
    """Blind: no truth in the header (p2pa silently reads an ARMS keyword)."""
    from astropy.io import fits

    hd = fits.Header()
    hd["BUNIT"] = "nanomaggy"
    hd["PIXSCALE"] = PIX
    fits.writeto(path, img, hd, overwrite=True)


def deproject(img: np.ndarray, q: float, pa_deg: float) -> np.ndarray:
    """Face-on view from the true ellipse: rotate the major axis to +x, stretch the minor by 1/q."""
    c = (np.arange(N) + 0.5) - N / 2
    u, v = np.meshgrid(c, c)
    ca, sa = np.cos(np.radians(pa_deg)), np.sin(np.radians(pa_deg))
    xp, yp = u, v * q  # face-on (u, v) → projected frame (x', y')
    x, y = xp * ca - yp * sa, xp * sa + yp * ca
    return map_coordinates(img, [y + N / 2 - 0.5, x + N / 2 - 0.5], order=3, mode="constant").astype(np.float32)


# ── SpArcFiRe's ellipse file (convention calibrated by the D28 echo check) ─────────────────────

ELPS_CAL = OUT / "elps_calibration.json"


def elps_text(s: Spec) -> str:
    """SpArcFiRe's '#key=value' ellipse string for the true (q, PA), in its own convention."""
    cal = json.loads(ELPS_CAL.read_text())
    ang = np.radians(cal["angle_sign"] * s.pa_deg + cal["angle_offset_deg"])
    ang = (ang + np.pi / 2) % np.pi - np.pi / 2  # atan range, as findCovarElpsAxes returns it
    sig_maj = np.polyval(cal["sigma_major_per_h_fit"], s.q) * s.h_pix  # linear in q (D28 calibration)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    cov = R @ np.diag([sig_maj ** 2, (sig_maj * s.q) ** 2]) @ R.T
    maj = np.polyval(cal["maj_len_per_h_fit"], s.q) * s.h_pix
    ctr = N / 2 + 0.5
    kv = {"diskMajAxsLen": maj, "diskMajAxsAngle": ang, "diskAxisRatio": s.q, "muFit": f"[{ctr} {ctr}]",
          "covarFit": f"[{cov[0, 0]:.10g} {cov[0, 1]:.10g};{cov[1, 0]:.10g} {cov[1, 1]:.10g}]",
          "gaussLogLik": 0, "wtdLik": 1, "likOfCtr": 1, "brtUnifScore": 0, "contourBrtRatio": 1,
          "diskMinAxsLen": maj * s.q, "diskMajAxsAngleRadians": ang}
    return "#".join(f"{k}={v:.16g}" if isinstance(v, float) else f"{k}={v}" for k, v in kv.items())


# ── running the methods ────────────────────────────────────────────────────────────────────────

SF_SETTINGS = {  # name -> flags after the three directories
    "default": [],
    "runsh": ["-stopThres", "0.1", "-useImageStandardization", "0", "-allowArcBeyond2pi", "0",
              "-errRatioThres", "5", "-unsharpMaskSigma", "10"],
    "runsh_amt15": ["-stopThres", "0.1", "-useImageStandardization", "0", "-allowArcBeyond2pi", "0",
                    "-errRatioThres", "5", "-unsharpMaskSigma", "10", "-unsharpMaskAmt", "15"],
}


def run_sparcfire(fits_dir: Path, tag: str, setting: str, elps: Path | None = None) -> dict:
    """Official SpArcFiRe (BB0c image), FITS path. One container at a time."""
    t0 = time.time()
    cmd = [str(RUNNER), "fits", tag, str(fits_dir), str(elps) if elps else "NONE", *SF_SETTINGS[setting]]
    log = SF_OUT / f"{tag}.log"
    with log.open("w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"SpArcFiRe run {tag} exited {rc}; see {log}")
    return {"wall_s": time.time() - t0, "n": len(list(fits_dir.glob("*.fits")))}


def read_sparcfire(tag: str) -> dict[str, dict]:
    """Per galaxy: |DCO|, all-arcs |pitch|, arc count, echoed ellipse. Ragged rows = no arcs."""
    path = SF_OUT / tag / "output" / "galaxy.tsv"
    pa, why = B.sf_read(path)
    lines = path.read_text().splitlines()
    head = lines[0].split("\t")
    out = {}
    for ln in lines[1:]:
        f = ln.split("\t")
        rec = {"dco": abs(pa[f[0]]) if np.isfinite(pa[f[0]]) else np.nan, "why": why.get(f[0])}
        if len(f) == len(head):
            g = dict(zip(head, f, strict=True))
            # "[]": run.sh does no disk/bulge fit, so those columns are empty matrices
            num = lambda k: float(g[k]) if g.get(k, "").strip() not in ("", "NaN", "[]") else np.nan  # noqa: E731
            rec["top2"] = g.get("top2_chirality_agreement", "").strip("'")  # exploratory detection (Y4)
            rec.update(all_arcs=abs(num("pa_alenWtd_avg")), n_arcs=num("totalNumArcs"),
                       chir=g.get("chirality_alenWtd"), axis_ratio=num("diskAxisRatio"),
                       maj_angle=num("diskMajAxsAngleRadians"), cov=g.get("covarFit"),
                       maj_len=num("diskMajAxsLen"))
        else:
            rec.update(all_arcs=np.nan, n_arcs=0.0)
        out[f[0]] = rec
    return out


P2_OUTER = 1.5  # P2DFFT outer radius = 1.5 × the donor's catalogued Petrosian radius (capped at the stamp)


def run_p2dfft(fits_dir: Path, stems: list[str], arms: dict[str, int], outer: dict[str, int]) -> dict[str, dict]:
    """p2dfft with a per-image outer radius from the catalogue (not the full stamp), then p2pa."""
    todo = [st for st in stems if not (fits_dir / f"{st}.h5").exists()]
    for i in range(0, len(todo), 50):
        par = fits_dir / f"p2dfft_radii_{i}.txt"
        # the 4th field is required: p2dfft 6.2 otherwise names the output after the radius (a stale
        # getline token in astro_class.cpp's parser)
        par.write_text("".join(f"{st}.fits,1,{outer[st]},{st}\n" for st in todo[i:i + 50]))
        subprocess.run([str(B.P2 / "build" / "p2dfft"), "-i", par.name], cwd=fits_dir, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    missing = [st for st in stems if not (fits_dir / f"{st}.h5").exists()]
    if missing:
        raise RuntimeError(f"p2dfft wrote no output for {len(missing)}: {missing[:3]}")
    return {"auto": B.p2pa(fits_dir, stems), "oracle": B.p2pa(fits_dir, stems, arms=arms)}


def p2_outer(s: Spec) -> int:
    return int(min(round(P2_OUTER * s.petro_rad / PIX), N // 2 - 1))


# ── building a set ─────────────────────────────────────────────────────────────────────────────


def build(specs: list[Spec], work: Path, elps: bool = True) -> dict:
    """Write each synthetic (and, for stages 2–3, its deprojected copy and ellipse file)."""
    (work / "fits").mkdir(parents=True, exist_ok=True)
    (work / "deproj").mkdir(exist_ok=True)
    (work / "elps").mkdir(exist_ok=True)
    t0, rows = time.time(), []
    bad = [s.gid for s in specs if "." in s.gid]
    if bad:
        raise ValueError(f"SpArcFiRe rejects names with a second '.': {bad[:3]}")
    for s in specs:
        img, s = render(s)
        write_fits(work / "fits" / f"{s.gid}.fits", img)
        if s.stage > 1:
            write_fits(work / "deproj" / f"{s.gid}.fits", deproject(img, s.q, s.pa_deg))
            if elps:
                (work / "elps" / f"{s.gid}_elps.txt").write_text(elps_text(s) + "\n")
        rows.append(asdict(s))
    with (work / "truth.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return {"n": len(rows), "render_s_per_image": (time.time() - t0) / max(1, len(rows))}


def pick_donors(pool: list[dict], quintile: int, k: int, rng) -> list[dict]:
    cand = [d for d in pool if d["quintile"] == quintile]
    return [cand[i] for i in rng.choice(len(cand), size=k, replace=False)]


def spec_from(d: dict, gid: str, stage: int, kind: str, pitch: float, arms: int, contrast: float,
              variant: str, rng) -> Spec:
    return Spec(gid=gid, stage=stage, kind=kind, pitch=pitch, arms=arms, contrast=contrast, variant=variant,
                chirality=int(rng.choice([-1, 1])), q=max(float(d["expAB_r"]), Q_MIN),
                pa_deg=float(rng.uniform(0, 180)), bt=float(rng.uniform(0.05, 0.3)), donor=str(d["object_id"]),
                v1=float(d["v1"]), quintile=int(d["quintile"]), psf_fwhm=float(d["psfWidth_r"]),
                mag=float(d["modelMag_r"]), snr=float(d["snr_r"]), petro_rad=float(d["petroRad_r"]),
                seed=int(rng.integers(2**31)), sky=donor_sky(str(d["object_id"])))


# ── analysis (identical code for plants and real runs) ─────────────────────────────────────────

THRESH_DEG = 7.0  # the ~7° spread between real galaxies (z_findings.md §Z1)
DET_MIN = 0.5
MARGIN_DET, MARGIN_RMS = 0.10, 1.0  # a difference smaller than this is a TIE whatever its CI
MIN_CELL, MIN_DET = 40, 20  # spirals per stage-quintile, detections per method-cell
N_BOOT = 2000


RHO_MIN = 0.70  # rank-faithful: Spearman with truth; ρ ≈ σ/√(σ²+e²) = 0.707 at e = σ = 7° (§BB2)
RANK_PITCH = (10, 30)  # true pitches the rank bar is read on (sd 7.1°, the real ~7° spread)
MARGIN_RHO = 0.05  # a Spearman difference smaller than this is a TIE whatever its CI
HARD_MARGIN = 1.0  # degrees: common support (b) — P2DFFT's extra error where SpArcFiRe abstains


def p2dfft_detections(meas: np.ndarray) -> np.ndarray:
    """Option 1 (user, 2026-09-25): P2DFFT never abstains, so it detects by definition — every
    finite answer is a detection. On nulls that is a 100% false-answer rate, reported as such."""
    return np.isfinite(meas)


def null_range(vals: np.ndarray) -> tuple[float, float]:
    """Central 95% of a method's |pitch| on the stage's null images."""
    v = vals[np.isfinite(vals)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v.size else (np.nan, np.nan)


def detections(meas: np.ndarray, rng_lohi: tuple[float, float] | None) -> np.ndarray:
    """SpArcFiRe: any arc (rng_lohi None; meas NaN = no arc). P2DFFT: |pitch| outside the null range."""
    ok = np.isfinite(meas)
    if rng_lohi is None:
        return ok
    lo, hi = rng_lohi
    return ok & ((meas < lo) | (meas > hi))


def cell_stats(truth: np.ndarray, meas: np.ndarray, det: np.ndarray) -> dict:
    e = (meas - truth)[det]
    n = int(truth.size)
    out = {"n": n, "n_det": int(det.sum()), "det": float(det.mean()) if n else np.nan}
    if e.size:
        out.update(rms=float(np.sqrt(np.mean(e ** 2))), bias=float(np.median(e)),
                   scatter=float(1.4826 * np.median(np.abs(e - np.median(e)))))
    return out


def _boot_idx(n: int, rng) -> np.ndarray:
    return rng.integers(0, n, size=(N_BOOT, n))


def _rms(e: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Row-wise RMS over detected entries (bootstrap matrices)."""
    k = d.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(np.where(d, e ** 2, 0).sum(axis=1) / k)


def works(truth, meas, det, rng) -> dict:
    """WORKS iff detection ≥ 0.5 and conditional RMS ≤ 7°; borderline if a CI straddles a bar."""
    st = cell_stats(truth, meas, det)
    if st["n"] < MIN_CELL:
        return {**st, "state": "INSUFFICIENT", "borderline": False}
    if st["n_det"] < MIN_DET:  # detection is measured on all n; too few detections to read accuracy
        return {**st, "state": "FAILS" if st["det"] < DET_MIN else "INSUFFICIENT", "borderline": False}
    ix = _boot_idx(st["n"], rng)
    e = np.where(det, meas - truth, 0.0)
    dets, rmss = det[ix].mean(axis=1), _rms(e[ix], det[ix])
    d_ci = np.percentile(dets, [2.5, 97.5])
    r_ci = np.nanpercentile(rmss, [2.5, 97.5])
    ok = st["det"] >= DET_MIN and st["rms"] <= THRESH_DEG
    border = bool((d_ci[0] < DET_MIN <= d_ci[1]) or (r_ci[0] <= THRESH_DEG < r_ci[1]))
    return {**st, "state": "WORKS" if ok else "FAILS", "borderline": border,
            "det_ci": d_ci.tolist(), "rms_ci": r_ci.tolist()}


def _rank_sub(truth, meas, det) -> np.ndarray:
    return det & np.isfinite(meas) & (truth >= RANK_PITCH[0]) & (truth <= RANK_PITCH[1])


def _rho(truth, meas, sub) -> float:
    if sub.sum() < 3 or np.ptp(truth[sub]) == 0:
        return np.nan
    return float(spearmanr(meas[sub], truth[sub])[0])


def rank_works(truth, meas, det, rng) -> dict:
    """RANK-FAITHFUL iff detection ≥ 0.5 and Spearman(measured, truth) ≥ 0.70 on detected spirals
    with true pitch 10–30°; borderline if a CI straddles either bar (feeds FRAGILE)."""
    st = cell_stats(truth, meas, det)
    sub = _rank_sub(truth, meas, det)
    st["n_rank"] = int(sub.sum())
    if st["n"] < MIN_CELL:
        return {**st, "state": "INSUFFICIENT", "borderline": False}
    if st["n_rank"] < MIN_DET:
        return {**st, "state": "FAILS" if st["det"] < DET_MIN else "INSUFFICIENT", "borderline": False}
    st["rho"] = _rho(truth, meas, sub)
    ix = _boot_idx(st["n"], rng)
    dets = det[ix].mean(axis=1)
    rhos = np.array([_rho(truth[i], meas[i], sub[i]) for i in ix])
    d_ci, r_ci = np.percentile(dets, [2.5, 97.5]), np.nanpercentile(rhos, [2.5, 97.5])
    ok = st["det"] >= DET_MIN and st["rho"] >= RHO_MIN
    border = bool((d_ci[0] < DET_MIN <= d_ci[1]) or (r_ci[0] < RHO_MIN <= r_ci[1]))
    return {**st, "state": "WORKS" if ok else "FAILS", "borderline": border,
            "det_ci": d_ci.tolist(), "rho_ci": r_ci.tolist()}


def compare_rank(truth, mS, dS, mF, dF, rng) -> dict:
    """Paired, where both are rank-faithful: better via detection (as `compare`) or via Spearman
    (Δρ ≥ 0.05 with its CI above 0, detection not worse beyond its margin)."""
    ix = _boot_idx(truth.size, rng)
    sS, sF = _rank_sub(truth, mS, dS), _rank_sub(truth, mF, dF)
    dd = dS[ix].mean(axis=1) - dF[ix].mean(axis=1)
    dr = np.array([_rho(truth[i], mS[i], sS[i]) - _rho(truth[i], mF[i], sF[i]) for i in ix])
    pd_, pr = float(dS.mean() - dF.mean()), _rho(truth, mS, sS) - _rho(truth, mF, sF)
    ddc, drc = np.percentile(dd, [2.5, 97.5]), np.nanpercentile(dr, [2.5, 97.5])

    def better(sign):
        dl = ddc[0] if sign > 0 else -ddc[1]
        rl = drc[0] if sign > 0 else -drc[1]
        ru = drc[1] if sign > 0 else -drc[0]
        via_det = dl > 0 and sign * pd_ >= MARGIN_DET and ru > -MARGIN_RHO
        via_rank = rl > 0 and sign * pr >= MARGIN_RHO and dl > -MARGIN_DET
        return via_det or via_rank, (dl > 0 or rl > 0)

    s_b, s_sig = better(+1)
    f_b, f_sig = better(-1)
    c = "S" if s_b else "F" if f_b else "TIE"
    note = "significant, below margin" if c == "TIE" and (s_sig or f_sig) else ""
    return {"call": c, "note": note, "d_det": pd_, "d_det_ci": ddc.tolist(), "d_rho": pr, "d_rho_ci": drc.tolist()}


def compare(truth, mS, dS, mF, dF, rng) -> dict:
    """Paired comparison where both methods WORK. S/F better only beyond the margins (D27)."""
    n = truth.size
    ix = _boot_idx(n, rng)
    eS, eF = np.where(dS, mS - truth, 0.0), np.where(dF, mF - truth, 0.0)
    dd = dS[ix].mean(axis=1) - dF[ix].mean(axis=1)
    dr = _rms(eS[ix], dS[ix]) - _rms(eF[ix], dF[ix])
    pd_, pr = float(dS.mean() - dF.mean()), float(_rms(eS[None], dS[None])[0] - _rms(eF[None], dF[None])[0])
    ddc, drc = np.percentile(dd, [2.5, 97.5]), np.nanpercentile(dr, [2.5, 97.5])

    def better(sign):  # sign +1: S better; −1: F better
        dl, du = (ddc[0], ddc[1]) if sign > 0 else (-ddc[1], -ddc[0])
        rl, ru = (drc[0], drc[1]) if sign > 0 else (-drc[1], -drc[0])
        pdet, prms = sign * pd_, sign * pr
        via_det = dl > 0 and pdet >= MARGIN_DET and ru < MARGIN_RMS
        via_acc = ru < 0 and -prms >= MARGIN_RMS and dl > -MARGIN_DET
        return via_det or via_acc, (dl > 0 or ru < 0)

    s_b, s_sig = better(+1)
    f_b, f_sig = better(-1)
    c = "S" if s_b else "F" if f_b else "TIE"
    note = "significant, below margin" if c == "TIE" and (s_sig or f_sig) else ""
    return {"call": c, "note": note, "d_det": pd_, "d_det_ci": ddc.tolist(), "d_rms": pr, "d_rms_ci": drc.tolist()}


def quintile_call(wS: dict, wF: dict, cmp: dict | None) -> str:
    if "INSUFFICIENT" in (wS["state"], wF["state"]):
        return "INSUFFICIENT"
    if wS["state"] == "FAILS" and wF["state"] == "FAILS":
        return "NEITHER"
    if wS["state"] == "WORKS" and wF["state"] == "FAILS":
        return "S"
    if wF["state"] == "WORKS" and wS["state"] == "FAILS":
        return "F"
    return cmp["call"] if cmp else "TIE"


def verdict(calls: list[str], wS_states: list[str], wF_states: list[str]) -> str:
    """Stage verdict from the five quintile calls; first match applies (D27)."""
    if calls.count("INSUFFICIENT") >= 2:
        return "INSUFFICIENT"
    W_S, W_F = wS_states.count("WORKS"), wF_states.count("WORKS")
    nS, nF = calls.count("S"), calls.count("F")
    if W_S <= 2 and W_F <= 2:
        return "BOTH FAIL AT SDSS QUALITY"
    if nS >= 1 and nF >= 1:
        return "DEPENDS ON VISIBILITY"
    if nF == 0 and (nS >= 2 or (W_S >= 3 and W_F <= 2)):
        return "SPARCFIRE BETTER"
    if nS == 0 and (nF >= 2 or (W_F >= 3 and W_S <= 2)):
        return "FOURIER BETTER"
    if W_S >= 3 and W_F >= 3:
        return "BOTH ADEQUATE"
    raise AssertionError(f"verdict ladder incomplete: {calls} {wS_states} {wF_states}")


def stop_visibility(states: list[str], edges: list[float]) -> dict:
    """Most → least visible: where the method first stops working (quintile upper edges on V1)."""
    first = next((i for i, s in enumerate(states) if s != "WORKS"), None)
    if first is None:
        return {"state": "WORKS THROUGHOUT"}
    later = any(s == "WORKS" for s in states[first + 1:])
    if first == 0:
        return {"state": "NEVER WORKS", "non_monotone": later}
    return {"state": "STOPS", "last_working_quintile": first, "v1_stop": edges[first - 1], "non_monotone": later}


def stage_verdict(truth, mS, dS, mF, dF, quint, edges, seed: int, ladder: str = "accuracy") -> dict:
    """Per-quintile WORKS for each method, paired calls, the stage verdict, FRAGILE, stop points.
    ladder "accuracy": WORKS = detection ≥ 0.5 and RMS ≤ 7°; "rank": detection ≥ 0.5 and ρ ≥ 0.70."""
    rng = np.random.default_rng(seed)
    wfn, cfn = (works, compare) if ladder == "accuracy" else (rank_works, compare_rank)
    rows = []
    for q in QUINTILES:
        k = quint == q
        wS, wF = wfn(truth[k], mS[k], dS[k], rng), wfn(truth[k], mF[k], dF[k], rng)
        both = wS["state"] == "WORKS" and wF["state"] == "WORKS"
        cmp = cfn(truth[k], mS[k], dS[k], mF[k], dF[k], rng) if both else None
        rows.append({"quintile": q, "sparcfire": wS, "fourier": wF, "compare": cmp,
                     "call": quintile_call(wS, wF, cmp)})
    sS, sF = [r["sparcfire"]["state"] for r in rows], [r["fourier"]["state"] for r in rows]
    v = verdict([r["call"] for r in rows], sS, sF)
    alts = set()
    for flip in ("WORKS", "FAILS"):  # every borderline cell pushed one way, then the other
        fS = [flip if r["sparcfire"]["borderline"] and s != "INSUFFICIENT" else s for r, s in zip(rows, sS, strict=True)]
        fF = [flip if r["fourier"]["borderline"] and s != "INSUFFICIENT" else s for r, s in zip(rows, sF, strict=True)]
        calls = [quintile_call({"state": a}, {"state": b}, r["compare"] if (a, b) == ("WORKS", "WORKS") else None)
                 for a, b, r in zip(fS, fF, rows, strict=True)]  # both WORKS, never compared → TIE
        alts.add(verdict(calls, fS, fF))
    alts.discard(v)
    cross = [(a["quintile"], b["quintile"]) for a, b in zip(rows, rows[1:], strict=False)
             if {a["call"], b["call"]} == {"S", "F"}]
    return {"verdict": v, "fragile": sorted(alts), "quintiles": rows, "crossover": cross,
            "stop": {"sparcfire": stop_visibility(sS, edges), "fourier": stop_visibility(sF, edges)}}


def stage_verdicts(truth, mS, dS, mF, dF, quint, edges, seed: int) -> dict:
    """Both co-primary ladders, side by side (§BB2): accurate and rank-faithful."""
    return {"accuracy": stage_verdict(truth, mS, dS, mF, dF, quint, edges, seed, "accuracy"),
            "rank": stage_verdict(truth, mS, dS, mF, dF, quint, edges, seed, "rank")}


def bb3_reference(rank_stage3: dict) -> dict:
    """BB3 may use a method as reference only if it is rank-faithful in ≥ 3 of 5 Stage-3 quintiles."""
    out = {}
    for m in ("sparcfire", "fourier"):
        n = sum(r[m]["state"] == "WORKS" for r in rank_stage3["quintiles"])
        out[m] = {"rank_faithful_quintiles": n, "usable_in_bb3": n >= 3}
    return out


def common_support(truth, mS, dS, mF, seed: int) -> dict:
    """(a) Both methods on the spirals SpArcFiRe detected; (b) is P2DFFT worse where SpArcFiRe
    abstains? ABSTAINS WHERE HARD iff median |err_F| on abstentions − on detections has CI > 0 and
    ≥ 1°; NOT WHERE HARD iff its CI lies below 1°; else UNRESOLVED; INSUFFICIENT under 20 in a group."""
    rng = np.random.default_rng(seed)
    fin = np.isfinite(mF)
    k = dS & np.isfinite(mS) & fin
    a = {"n": int(k.sum())}
    if k.sum() >= MIN_DET:
        eS, eF = mS[k] - truth[k], mF[k] - truth[k]
        sub = (truth[k] >= RANK_PITCH[0]) & (truth[k] <= RANK_PITCH[1])
        a.update(sparcfire=cell_stats(truth[k], mS[k], np.ones(k.sum(), bool)),
                 fourier=cell_stats(truth[k], mF[k], np.ones(k.sum(), bool)),
                 rho_sparcfire=_rho(truth[k], mS[k], sub), rho_fourier=_rho(truth[k], mF[k], sub))
        ix = _boot_idx(int(k.sum()), rng)
        drms = np.sqrt((eS[ix] ** 2).mean(1)) - np.sqrt((eF[ix] ** 2).mean(1))
        drho = [_rho(truth[k][i], mS[k][i], sub[i]) - _rho(truth[k][i], mF[k][i], sub[i]) for i in ix]
        a.update(d_rms=float(np.sqrt((eS ** 2).mean()) - np.sqrt((eF ** 2).mean())),
                 d_rms_ci=np.percentile(drms, [2.5, 97.5]).tolist(),
                 d_rho=float(a["rho_sparcfire"] - a["rho_fourier"]), d_rho_ci=np.nanpercentile(drho, [2.5, 97.5]).tolist())
    ab, de = ~dS & fin, dS & fin
    b = {"n_abstained": int(ab.sum()), "n_detected": int(de.sum())}
    if ab.sum() < MIN_DET or de.sum() < MIN_DET:
        b["state"] = "INSUFFICIENT"
    else:
        eA, eD = np.abs(mF[ab] - truth[ab]), np.abs(mF[de] - truth[de])
        diff = float(np.median(eA) - np.median(eD))
        bs = [np.median(rng.choice(eA, eA.size)) - np.median(rng.choice(eD, eD.size)) for _ in range(N_BOOT)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        b.update(median_err_abstained=float(np.median(eA)), median_err_detected=float(np.median(eD)),
                 diff=diff, ci=[float(lo), float(hi)],
                 state="ABSTAINS WHERE HARD" if lo > 0 and diff >= HARD_MARGIN
                 else "NOT WHERE HARD" if hi < HARD_MARGIN else "UNRESOLVED")
    return {"a_both_on_sparcfire_detections": a, "b_fourier_where_sparcfire_abstains": b}


def null_answers(mS: np.ndarray, mF: np.ndarray) -> dict:
    """Nulls: SpArcFiRe's any-arc rate (a false detection) and P2DFFT's answers (option 1: every
    finite answer is a false answer), with P2DFFT's |pitch| distribution."""
    f = mF[np.isfinite(mF)]
    return {"n": int(mS.size), "sparcfire_false_detection": float(np.isfinite(mS).mean()) if mS.size else np.nan,
            "fourier_false_answer": float(np.isfinite(mF).mean()) if mF.size else np.nan,
            "fourier_pitch_quantiles": np.percentile(f, [0, 5, 25, 50, 75, 95, 100]).tolist() if f.size else []}


def fisher_ci(r: float, n: int) -> tuple[float, float]:
    se = np.sqrt((1 + r ** 2 / 2) / (n - 3))  # Bonett & Wright
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def diagnostic(a: np.ndarray, b: np.ndarray, seed: int) -> dict:
    """Do the two methods agree with each other? Z1's bar: ρ = 0.5, both intervals on one side."""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 30:
        return {"state": "INSUFFICIENT", "n": int(a.size)}
    r = float(spearmanr(a, b)[0])
    rng = np.random.default_rng(seed)
    bs = [spearmanr(a[i], b[i])[0] for i in rng.integers(0, a.size, (N_BOOT, a.size))]
    f, p = fisher_ci(r, a.size), np.nanpercentile(bs, [2.5, 97.5])
    lo, hi = min(f[0], p[0]), max(f[1], p[1])
    s = "AGREE" if lo >= 0.5 else "DISAGREE" if hi < 0.5 else "UNRESOLVED"
    return {"state": s, "n": int(a.size), "rho": r, "ci": [lo, hi], "fisher": list(f), "boot": p.tolist()}


def peng(truth, dco, allarc, det, quint, seed: int) -> dict:
    """Peng et al. 2018's SpArcFiRe trends, tested on synthetics with known truth (SpArcFiRe primary).

    P1 convergence (slope of measured on true falls from Q1 to Q5); P2 error grows (RMS Q5 > Q1);
    P3 (a) loose arms err more in degrees AND (b) tight arms more as a fraction, Q4–5 — Peng's pair;
    (b) alone holds for any constant error, so P3 matches only when both do; P4 tight arms lost first
    (detection, Q4–5);
    P5 DCO less affected than all arcs (RMS, Q4–5). Each: MATCHES / OPPOSITE / NOT RESOLVED /
    NOT TESTABLE (a group under 20)."""
    rng = np.random.default_rng(seed)

    def state(fn, groups):
        if any(g.sum() < 20 for g in groups):
            return {"state": "NOT TESTABLE"}
        pt = fn(*[np.flatnonzero(g) for g in groups])
        bs = [fn(*[rng.choice(np.flatnonzero(g), g.sum()) for g in groups]) for _ in range(N_BOOT)]
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        s = "MATCHES" if lo > 0 else "OPPOSITE" if hi < 0 else "NOT RESOLVED"
        return {"state": s, "value": float(pt), "ci": [float(lo), float(hi)]}

    def slope(i):
        return np.polyfit(truth[i], dco[i], 1)[0] if np.ptp(truth[i]) > 0 else np.nan

    def rms(i, m=dco):
        return np.sqrt(np.mean((m[i] - truth[i]) ** 2))

    d = det & np.isfinite(dco)
    q1, q5, q45 = d & (quint == 1), d & (quint == 5), d & (quint >= 4)
    tight, loose = truth <= 10, truth >= 30
    both = q45 & np.isfinite(allarc)
    res = {
        "P1_convergence": state(lambda a, b: slope(a) - slope(b), [q1, q5]),
        "P2_error_grows": state(lambda a, b: rms(b) - rms(a), [q1, q5]),
        "P3a_loose_larger_absolute": state(
            lambda a, b: np.median(np.abs(dco[b] - truth[b])) - np.median(np.abs(dco[a] - truth[a])),
            [q45 & tight, q45 & loose]),
        "P3b_tight_larger_fraction": state(
            lambda a, b: np.median(np.abs(dco[a] - truth[a]) / truth[a]) - np.median(np.abs(dco[b] - truth[b]) / truth[b]),
            [q45 & tight, q45 & loose]),
        "P4_tight_lost_first": state(lambda a, b: det[b].mean() - det[a].mean(),
                                     [(quint >= 4) & tight, (quint >= 4) & loose]),
        "P5_dco_less_affected": state(lambda a: rms(a, allarc) - rms(a), [both]),
    }
    pair = [res["P3a_loose_larger_absolute"]["state"], res["P3b_tight_larger_fraction"]["state"]]
    res["P3_peng_pair"] = {"state": "NOT TESTABLE" if "NOT TESTABLE" in pair else "OPPOSITE" if "OPPOSITE" in pair
                           else "MATCHES" if pair == ["MATCHES", "MATCHES"] else "NOT RESOLVED"}
    return res


def plant_logic() -> dict:
    """D28 on the analysis: fake methods through the identical code, at the planned cell size."""
    grid = [(p, a, c, q) for q in QUINTILES for p in PITCHES for a in ARMS for c in CONTRASTS]
    truth = np.array([g[0] for g in grid], float)
    quint = np.array([g[3] for g in grid])
    edges = [-1.67, -0.43, 0.64, 1.67]
    rng = np.random.default_rng(11)

    def fake(sd, det_rate, bad_q=(), shuffle=False, bias=0.0):
        m = truth + bias + rng.normal(0, sd, truth.size)
        if shuffle:
            m = rng.permutation(truth) + rng.normal(0, sd, truth.size)
        bad = np.isin(quint, bad_q)
        m = np.where(bad, rng.uniform(0, 45, truth.size), m)
        d = rng.random(truth.size) < det_rate
        return np.abs(m), d

    good, junk = (2.0, 0.9), dict(shuffle=True, sd=2.0, det_rate=0.9)
    cases = {
        "both good -> BOTH ADEQUATE": (fake(*good), fake(*good)),
        "S good, F shuffled -> SPARCFIRE BETTER": (fake(*good), fake(**junk)),
        "S shuffled, F good -> FOURIER BETTER": (fake(**junk), fake(*good)),
        "both shuffled -> BOTH FAIL": (fake(**junk), fake(**junk)),
        "S good Q1-2 only, F good Q3-5 only -> DEPENDS": (fake(2.0, 0.9, bad_q=(3, 4, 5)), fake(2.0, 0.9, bad_q=(1, 2))),
        "S detects 20% -> FOURIER BETTER": (fake(2.0, 0.2), fake(*good)),
        "S RMS 2 vs F RMS 5, both work -> SPARCFIRE BETTER (accuracy)": (fake(2.0, 0.9), fake(5.0, 0.9)),
        "S RMS 2 vs F 2.5 -> BOTH ADEQUATE (below margin)": (fake(2.0, 0.9), fake(2.5, 0.9)),
        "S RMS ~7 on the bar, F shuffled -> FRAGILE": (fake(7.0, 0.9), fake(**junk)),
    }
    out = {}
    for name, ((mS, dS), (mF, dF)) in cases.items():
        r = stage_verdict(truth, mS, dS, mF, dF, quint, edges, seed=1)
        out[name] = {"verdict": r["verdict"], "fragile": r["fragile"], "crossover": r["crossover"],
                     "stop_S": r["stop"]["sparcfire"]["state"], "stop_F": r["stop"]["fourier"]["state"]}
    # the rank ladder (co-primary) and option 1 (P2DFFT detects every finite answer). A single
    # draw can land a chance margin call in one quintile, so these report verdict frequencies
    # over 10 independent draws; a label is reached if it is the modal verdict.
    rank_cases = {
        "both compressed (0.4·truth+7) -> accuracy BOTH FAIL, rank BOTH ADEQUATE": ("comp", "comp"),
        "S shuffled, F good (always answers) -> rank FOURIER BETTER": ("junk", "fF"),
        "both good, S detects 90%, F always answers -> FOURIER BETTER via detection (option 1)": ("good", "fF"),
        "S rho ~0.70 on the bar (sd 7.1), F shuffled -> rank FRAGILE": ("bar", "junk"),
    }
    for name, (a_, b_) in rank_cases.items():
        freq: dict = {"accuracy": {}, "rank": {}, "rank_fragile_any": 0}
        for seed in range(10):
            g = np.random.default_rng(1000 + seed)
            mk = {"comp": lambda: (np.abs(0.4 * truth + 7 + g.normal(0, 1.5, truth.size)), g.random(truth.size) < 0.9),
                  "fF": lambda: (np.abs(truth + g.normal(0, 2.0, truth.size)), np.ones(truth.size, bool)),
                  "good": lambda: (np.abs(truth + g.normal(0, 2.0, truth.size)), g.random(truth.size) < 0.9),
                  "bar": lambda: (np.abs(truth + g.normal(0, 7.1, truth.size)), g.random(truth.size) < 0.9),
                  "junk": lambda: (np.abs(g.permutation(truth) + g.normal(0, 2.0, truth.size)),
                                   g.random(truth.size) < 0.9)}
            (mS, dS), (mF, dF) = mk[a_](), mk[b_]()
            if b_ == "fF":
                dF = p2dfft_detections(mF)
            r = stage_verdicts(truth, mS, dS, mF, dF, quint, edges, seed=seed)
            for k in ("accuracy", "rank"):
                freq[k][r[k]["verdict"]] = freq[k].get(r[k]["verdict"], 0) + 1
            freq["rank_fragile_any"] += bool(r["rank"]["fragile"])
        out[name] = freq
    # common support (b): S abstains exactly where F errs (hard) vs at random
    hard = rng.random(truth.size) < 0.4
    mF = np.abs(truth + np.where(hard, rng.normal(0, 8, truth.size), rng.normal(0, 1, truth.size)))
    mS = np.abs(truth + rng.normal(0, 2, truth.size))
    out["common support: S abstains where F errs -> ABSTAINS WHERE HARD"] = common_support(
        truth, mS, ~hard, mF, seed=4)["b_fourier_where_sparcfire_abstains"]["state"]
    out["common support: S abstains at random -> NOT WHERE HARD"] = common_support(
        truth, mS, rng.random(truth.size) > 0.4, mF, seed=4)["b_fourier_where_sparcfire_abstains"]["state"]
    out["common support: 10 abstentions -> INSUFFICIENT"] = common_support(
        truth, mS, np.arange(truth.size) >= 10, mF, seed=4)["b_fourier_where_sparcfire_abstains"]["state"]
    out["bb3_reference (rank WORKS in 3/5) -> usable"] = bb3_reference(
        {"quintiles": [{"sparcfire": {"state": s}, "fourier": {"state": "FAILS"}}
                       for s in ("WORKS", "WORKS", "WORKS", "FAILS", "FAILS")]})
    thin = np.isin(np.arange(truth.size), rng.choice(truth.size, 150, replace=False))
    (mS, dS), (mF, dF) = fake(*good), fake(*good)
    r = stage_verdict(truth[thin], mS[thin], dS[thin], mF[thin], dF[thin], quint[thin], edges, seed=1)
    out["30 spirals per quintile -> INSUFFICIENT"] = {"verdict": r["verdict"]}
    # the diagnostic, at the pitch range and a plausible co-detected n
    sub = (truth >= 10) & (truth <= 30)
    t = truth[sub][:60]
    d = {}
    for name, (a, b) in {"agree (both truth+2°)": (t + rng.normal(0, 2, t.size), t + rng.normal(0, 2, t.size)),
                         "disagree (independent)": (rng.permutation(t), t + rng.normal(0, 2, t.size)),
                         "moderate (truth+6°)": (t + rng.normal(0, 6, t.size), t + rng.normal(0, 6, t.size)),
                         "n=20": (t[:20], t[:20])}.items():
        d[name] = diagnostic(a, b, seed=2)["state"]
    out["diagnostic"] = d
    # Peng: a planted convergence (slope 1 in Q1 → 0.2 in Q5) must MATCH; no trend must not
    m = np.where(quint == 5, 0.2 * truth + 0.8 * truth.mean(), truth) + rng.normal(0, 1, truth.size)
    m = np.where(np.isin(quint, (2, 3, 4)), truth, m)
    det = ~((quint >= 4) & (truth <= 10) & (rng.random(truth.size) < 0.35)) | (quint < 4)
    m = np.where((quint >= 4) & (truth >= 30), truth - 8, np.where((quint >= 4) & (truth <= 10), truth + 3, m))
    out["peng_planted"] = {k: v["state"] for k, v in peng(truth, m, m + rng.normal(0, 4, truth.size), det, quint, 3).items()}
    flat = truth + rng.normal(0, 1, truth.size)
    out["peng_null"] = {k: v["state"] for k, v in peng(truth, flat, flat, np.ones(truth.size, bool), quint, 3).items()}
    return out


# ── D28 plumbing: the generator, the ellipse, the null range (real methods) ────────────────────

CLEAN_DONOR = {"object_id": "plant", "v1": 0.0, "quintile": 1, "psfWidth_r": 1.2, "modelMag_r": 13.0,
               "snr_r": 1e4, "petroRad_r": 20.0, "expAB_r": 1.0, "specz": 0.02}


def _clean(gid, stage, kind, pitch, arms, q, pa, contrast=1.2, chir=1, seed=0) -> Spec:
    s = Spec(gid=gid, stage=stage, kind=kind, pitch=pitch, arms=arms, contrast=contrast, variant="clean",
             chirality=chir, q=q, pa_deg=pa, bt=0.1, donor="plant", v1=0.0, quintile=1, psf_fwhm=1.2,
             mag=13.0, snr=1e4, petro_rad=20.0, seed=seed)
    s.sky = 1e-4
    return s


def _p2(work: Path, specs: list[Spec], sub: str = "fits") -> dict:
    d = work / sub
    stems = [s.gid for s in specs]
    return run_p2dfft(d, stems, {s.gid: max(1, s.arms) for s in specs}, {s.gid: p2_outer(s) for s in specs})


def plant_generator() -> dict:
    """(a) generator validity and (b) SpArcFiRe's ellipse convention, one SpArcFiRe run."""
    work = OUT / "plant_gen"
    a = [_clean(f"gen_p{p}_m2", 1, "spiral", p, 2, 1.0, 0, seed=i) for i, p in enumerate(PITCHES)]
    a += [_clean(f"gen_p25_m{m}", 1, "spiral", 25, m, 1.0, 0, seed=10 + m) for m in (1, 3, 4)]
    a += [_clean("gen_p25_m2_Z", 1, "spiral", 25, 2, 1.0, 0, chir=-1, seed=20)]
    b = [_clean(f"cal_q{round(q * 10):02d}_pa{pa}", 2, "armless", 25, 2, q, pa, seed=30 + i)
         for i, (q, pa) in enumerate((q, pa) for q in (0.4, 0.6, 0.8) for pa in (20, 70, 130))]
    info = build(a + b, work, elps=False)  # these discs define the ellipse convention
    done = (SF_OUT / "bb2_plant_gen" / "output" / "galaxy.tsv").exists() and "--reuse" in sys.argv
    t = {"reused": True} if done else run_sparcfire(work / "fits", "bb2_plant_gen", "default")
    sf = read_sparcfire("bb2_plant_gen")
    p2 = _p2(work, a)
    gen = []
    for s in a:
        gen.append({"gid": s.gid, "truth": s.pitch, "arms": s.arms, "chir": s.chirality,
                    "sparcfire": sf.get(s.gid, {}).get("dco"), "sf_chir": sf.get(s.gid, {}).get("chir"),
                    "p2_oracle": abs(p2["oracle"].get(s.gid, {}).get("pa", np.nan)),
                    "p2_oracle_signed": p2["oracle"].get(s.gid, {}).get("pa", np.nan),
                    "p2_auto": abs(p2["auto"].get(s.gid, {}).get("pa", np.nan)),
                    "p2_auto_mode": p2["auto"].get(s.gid, {}).get("mode")})
    tr = np.array([g["truth"] for g in gen])

    def gate(key):
        m = np.array([np.nan if g[key] is None else g[key] for g in gen], float)
        ok = np.isfinite(m)
        rho = float(spearmanr(m[ok], tr[ok])[0]) if ok.sum() > 3 else np.nan
        med = float(np.median(np.abs(m[ok] - tr[ok]))) if ok.any() else np.nan
        return {"n_measured": int(ok.sum()), "spearman": rho, "median_abs_err": med,
                "valid": bool(ok.sum() >= 9 and rho >= 0.9 and med <= 3.0)}

    cal_rows = []
    for s in b:
        r = fitted_ellipse("bb2_plant_gen", s.gid)
        cal_rows.append({"gid": s.gid, "q": s.q, "pa": s.pa_deg, "h": s.h_pix, "fit_q": r.get("diskAxisRatio"),
                         "fit_angle": r.get("diskMajAxsAngle"), "cov": r.get("covarFit"), "maj_len": r.get("diskMajAxsLen")})
    cal = calibrate_ellipse(cal_rows)
    ELPS_CAL.write_text(json.dumps(cal, indent=1))
    return {"build": info, "sparcfire_run": t, "generator": gen,
            "gates": {"sparcfire": gate("sparcfire"), "p2dfft_oracle": gate("p2_oracle"), "p2dfft_auto": gate("p2_auto")},
            "calibration_rows": cal_rows, "calibration": cal}


def fitted_ellipse(tag: str, gid: str) -> dict:
    """SpArcFiRe's own ellipse for one galaxy, from its per-galaxy '-elps-fit-params.txt'."""
    f = SF_OUT / tag / "output" / gid / f"{gid}-elps-fit-params.txt"
    if not f.exists():
        return {}
    kv = dict(x.split("=", 1) for x in f.read_text().strip().split("#"))
    out = {}
    for k, v in kv.items():
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def calibrate_ellipse(rows: list[dict]) -> dict:
    """SpArcFiRe's angle = sign·PA + offset (mod 180°); its Gaussian σ_major and contour length ∝ h."""
    ok = [r for r in rows if r["fit_angle"] is not None and np.isfinite(r["fit_angle"])]
    best = None
    for sign in (1, -1):
        for off in np.arange(0, 180, 0.5):
            d = [((np.degrees(r["fit_angle"]) - (sign * r["pa"] + off) + 90) % 180) - 90 for r in ok]
            cost = float(np.median(np.abs(d)))
            if best is None or cost < best[0]:
                best = (cost, sign, float(off), d)
    sig, lens = [], []
    for r in ok:
        c = np.array([[float(x) for x in row.split()] for row in r["cov"].strip("[]").split(";")])
        sig.append(np.sqrt(np.linalg.eigvalsh(c).max()) / r["h"])
        lens.append(r["maj_len"] / r["h"])
    qs = [r["q"] for r in ok]
    return {"angle_sign": best[1], "angle_offset_deg": best[2], "angle_resid_deg": best[3],
            "sigma_major_per_h_fit": np.polyfit(qs, sig, 1).tolist(), "sigma_major_per_h_all": sig,
            "maj_len_per_h_fit": np.polyfit(qs, lens, 1).tolist(), "maj_len_per_h_all": lens,
            "fit_q_vs_true": [(r["q"], r["fit_q"]) for r in ok], "n": len(ok)}


def plant_ellipse() -> dict:
    """(c) the true ellipse reaches SpArcFiRe unchanged, and deprojection preserves pitch (both)."""
    work = OUT / "plant_elps"
    sp = [_clean(f"elps_p{p}_pa{pa}", 2, "spiral", p, 2, 0.5, pa, seed=40 + i)
          for i, (p, pa) in enumerate((p, pa) for p in (15, 30) for pa in (30, 100, 160))]
    info = build(sp, work)
    reuse = "--reuse" in sys.argv and (SF_OUT / "bb2_plant_elps_own" / "output" / "galaxy.tsv").exists()
    t = {"reused": True} if reuse else run_sparcfire(work / "fits", "bb2_plant_elps", "default", elps=work / "elps")
    sf = read_sparcfire("bb2_plant_elps")
    t2 = {"reused": True} if reuse else run_sparcfire(work / "fits", "bb2_plant_elps_own", "default")
    own = read_sparcfire("bb2_plant_elps_own")
    p2 = _p2(work, sp, "deproj")
    rows = []
    for s in sp:
        sup = dict(kv.split("=", 1) for kv in (work / "elps" / f"{s.gid}_elps.txt").read_text().strip().split("#"))
        r = sf.get(s.gid, {})
        rows.append({"gid": s.gid, "truth": s.pitch, "q": s.q, "pa": s.pa_deg,
                     "echo_q": r.get("axis_ratio"), "supplied_q": float(sup["diskAxisRatio"]),
                     "echo_angle": r.get("maj_angle"), "supplied_angle": float(sup["diskMajAxsAngle"]),
                     "sparcfire": r.get("dco"), "sparcfire_own_ellipse": own.get(s.gid, {}).get("dco"),
                     "own_fit_q": own.get(s.gid, {}).get("axis_ratio"),
                     "p2_oracle_deproj": abs(p2["oracle"].get(s.gid, {}).get("pa", np.nan))})
    echo = all(r["echo_q"] is not None and abs(r["echo_q"] - r["supplied_q"]) < 0.01
               and abs(((r["echo_angle"] - r["supplied_angle"]) + np.pi / 2) % np.pi - np.pi / 2) < 0.01 for r in rows)
    e_sf = [abs(r["sparcfire"] - r["truth"]) for r in rows if r["sparcfire"] is not None and np.isfinite(r["sparcfire"])]
    same = [abs(r["sparcfire"] - r["sparcfire_own_ellipse"]) for r in rows
            if r["sparcfire"] is not None and r["sparcfire_own_ellipse"] is not None
            and np.isfinite(r["sparcfire"]) and np.isfinite(r["sparcfire_own_ellipse"])]
    e_p2 = [abs(r["p2_oracle_deproj"] - r["truth"]) for r in rows if np.isfinite(r["p2_oracle_deproj"])]
    # plumbing only: the supplied ellipse arrives unchanged and does what SpArcFiRe's own ellipse does,
    # and P2DFFT recovers pitch from our deprojection. SpArcFiRe's accuracy is BB2's question, recorded.
    return {"build": info, "runs": [t, t2], "rows": rows, "echo_exact": echo,
            "sparcfire_median_abs_err": float(np.median(e_sf)) if e_sf else None, "sparcfire_n": len(e_sf),
            "sparcfire_supplied_vs_own_median_abs": float(np.median(same)) if same else None,
            "p2_median_abs_err": float(np.median(e_p2)) if e_p2 else None, "p2_n": len(e_p2),
            "valid": bool(echo and len(same) >= 5 and np.median(same) <= 1.5 and len(e_p2) >= 5 and np.median(e_p2) <= 3)}


def plant_nulls(pool: list[dict]) -> dict:
    """(d) P2DFFT's null range from real-donor nulls; planted spirals must fall outside, held-out nulls inside."""
    work = OUT / "plant_nulls"
    rng = np.random.default_rng(50)
    specs = []
    for i, q in enumerate([1, 2, 3, 4, 5] * 12):
        d = pick_donors(pool, q, 1, rng)[0]
        kind = "noise" if i % 2 else "armless"
        specs.append(spec_from(d, f"null{i:02d}_{kind}", 1, kind, 25, 2, 0.0, "clean", rng))
    for i in range(10):
        d = pick_donors(pool, 1, 1, rng)[0]
        specs.append(spec_from(d, f"strong{i}", 1, "spiral", float(rng.choice([15, 25, 30])), 2, 1.2, "clean", rng))
    info = build(specs, work)
    p2 = _p2(work, specs)
    auto = {k: abs(v["pa"]) for k, v in p2["auto"].items()}
    fit = [s.gid for s in specs if s.kind != "spiral"][:40]
    held = [s.gid for s in specs if s.kind != "spiral"][40:]
    strong = [s.gid for s in specs if s.kind == "spiral"]
    lohi = null_range(np.array([auto.get(g, np.nan) for g in fit]))
    det = lambda gs: detections(np.array([auto.get(g, np.nan) for g in gs]), lohi)  # noqa: E731
    out = {"build": info, "null_range": lohi, "null_values": sorted(auto.get(g, np.nan) for g in fit),
           "strong_detected": int(det(strong).sum()), "n_strong": len(strong),
           "heldout_false": int(det(held).sum()), "n_heldout": len(held),
           "strong_values": {g: auto.get(g) for g in strong}}
    # the null-range detection rule FAILED (the |pitch| null range [2.2°, 90°] saturates): option 1.
    out["valid_null_range_rule"] = bool(out["strong_detected"] >= 8 and out["heldout_false"] <= 3)
    nulls = [s.gid for s in specs if s.kind != "spiral"]
    truth = {s.gid: s.pitch for s in specs if s.kind == "spiral"}
    na = null_answers(np.array([np.nan] * len(nulls)), np.array([auto.get(g, np.nan) for g in nulls]))
    na["sparcfire_false_detection"] = "not run in this plant"
    err = [abs(auto[g] - truth[g]) for g in strong if g in auto and np.isfinite(auto[g])]
    out["option1"] = {"null_answers": na, "strong_median_abs_err": float(np.median(err)) if err else None,
                      "n_strong_answered": len(err)}
    # option 1's plumbing: P2DFFT answers on (nearly) every null, and is right on the strong spirals
    out["valid"] = bool(na["fourier_false_answer"] >= 0.95 and len(err) >= 8 and np.median(err) <= 3.0)
    return out


# ── the Stage 1 pilot (timing and detection only; no accuracy is computed) ─────────────────────

PILOT_PER_Q, PILOT_NULLS_PER_Q = 52, 8


def pilot(pool: list[dict]) -> dict:
    """~300 Stage 1 images (52 spirals + 8 nulls per V1 quintile): timing, detection rates, null
    answers. No accuracy is computed here — that is the grid's test, after the go-ahead."""
    work = OUT / "pilot"
    rng = np.random.default_rng(2026_09_25)
    specs = []
    for q in QUINTILES:
        for j, d in enumerate(pick_donors(pool, q, PILOT_PER_Q + PILOT_NULLS_PER_Q, rng)):
            if j < PILOT_PER_Q:
                specs.append(spec_from(d, f"pilot_q{q}_s{j:02d}", 1, "spiral", float(rng.choice(PITCHES)),
                                       int(rng.choice(ARMS)), float(rng.choice(CONTRASTS)), "clean", rng))
            else:
                kind = "noise" if j % 2 else "armless"
                specs.append(spec_from(d, f"pilot_q{q}_{kind}{j:02d}", 1, kind, 25, 2, 0.0, "clean", rng))
    t0 = time.time()
    info = build(specs, work, elps=False)
    dropped = [s.gid for s in specs if not np.isfinite(s.h_pix)]
    t_build = time.time() - t0
    reuse = "--reuse" in sys.argv and (SF_OUT / "bb2_pilot" / "output" / "galaxy.tsv").exists()
    tsf = {"reused": True} if reuse else run_sparcfire(work / "fits", "bb2_pilot", "default")
    sf = read_sparcfire("bb2_pilot")
    t1 = time.time()
    p2 = _p2(work, specs)
    t_p2 = time.time() - t1
    rows = []
    for s in specs:
        r = sf.get(s.gid, {})
        rows.append({"gid": s.gid, "kind": s.kind, "quintile": s.quintile, "arms": s.arms, "pitch": s.pitch,
                     "contrast": s.contrast, "h_pix": s.h_pix, "snr_synth": s.snr_synth,
                     "sf_detect": bool(np.isfinite(r.get("dco", np.nan))), "sf_n_arcs": r.get("n_arcs"),
                     "p2_auto_answer": bool(np.isfinite(p2["auto"].get(s.gid, {}).get("pa", np.nan))),
                     "p2_auto_abs": abs(p2["auto"].get(s.gid, {}).get("pa", np.nan))})
    det = {}
    for q in QUINTILES:
        sp = [r for r in rows if r["quintile"] == q and r["kind"] == "spiral"]
        nu = [r for r in rows if r["quintile"] == q and r["kind"] != "spiral"]
        det[q] = {"n_spiral": len(sp), "sparcfire_detect": float(np.mean([r["sf_detect"] for r in sp])),
                  "fourier_answer": float(np.mean([r["p2_auto_answer"] for r in sp])),
                  "n_null": len(nu), "sparcfire_false_detect": float(np.mean([r["sf_detect"] for r in nu])),
                  "fourier_false_answer": float(np.mean([r["p2_auto_answer"] for r in nu]))}
    by_arms = {m: float(np.mean([r["sf_detect"] for r in rows if r["kind"] == "spiral" and r["arms"] == m]))
               for m in ARMS}
    by_contrast = {c: float(np.mean([r["sf_detect"] for r in rows if r["kind"] == "spiral" and r["contrast"] == c]))
                   for c in CONTRASTS}
    nulls = [r for r in rows if r["kind"] != "spiral"]
    n = len(specs)
    out = {"n": n, "dropped_h_nan": dropped, "build": info,
           "timing": {"build_s_per_image": t_build / n, "sparcfire": tsf,
                      "sparcfire_s_per_image": tsf.get("wall_s", np.nan) / n, "p2dfft_s_per_image": t_p2 / n},
           "detection_by_quintile": det, "sparcfire_detect_by_arms": by_arms,
           "sparcfire_detect_by_contrast": by_contrast,
           "null_answers": null_answers(np.array([np.nan if not r["sf_detect"] else 1.0 for r in nulls]),
                                        np.array([r["p2_auto_abs"] for r in nulls])),
           "rows": rows}
    (OUT / "pilot.json").write_text(json.dumps(out, indent=1, default=float))
    return out


# ── the full grid (user go-ahead 2026-09-25; bb_findings.md §BB2 and "Recorded before the grid") ──

GRID_NULLS = 20  # per kind per quintile


def grid_specs(pool: list[dict], stage: int) -> list[Spec]:
    rng = np.random.default_rng(2026_09_25 + stage)
    combos = [(p, a, ci) for p in PITCHES for a in ARMS for ci in range(len(CONTRASTS))]
    specs = []
    for q in QUINTILES:
        donors = pick_donors(pool, q, len(combos) + 2 * GRID_NULLS, rng)
        var = rng.permutation(np.repeat(np.arange(3), len(combos) // 3)) if stage == 3 else None
        for i, (p, a, ci) in enumerate(combos):
            v = VARIANTS3[var[i]] if stage == 3 else "clean"
            specs.append(spec_from(donors[i], f"s{stage}_q{q}_i{i:02d}_p{p}_m{a}_c{ci}", stage, "spiral",
                                   float(p), a, CONTRASTS[ci], v, rng))
        for j in range(2 * GRID_NULLS):
            kind = "noise" if j % 2 else "armless"
            specs.append(spec_from(donors[len(combos) + j], f"s{stage}_q{q}_{kind}{j:02d}", stage, kind,
                                   25.0, 2, 0.0, "clean", rng))
    return specs


def _half(s: Spec) -> bool:
    """The fixed half that gets the run.sh sensitivities: every other grid point / null."""
    return int(s.gid.split("_")[2][1:]) % 2 == 0 if s.kind == "spiral" else int(s.gid[-2:]) % 2 == 0


def _sf_done(tag: str, n: int) -> bool:
    f = SF_OUT / tag / "output" / "galaxy.tsv"
    return f.exists() and len(f.read_text().splitlines()) - 1 >= n


def grid_stage(pool: list[dict], stage: int) -> dict:
    """Build, run both methods (primary + sensitivities), and return the per-image records."""
    import shutil
    work = OUT / f"grid_s{stage}"
    specs = grid_specs(pool, stage)
    if not (work / "truth.csv").exists():
        info = build(specs, work)
    else:  # re-render is deterministic; reuse the files, re-derive h/snr from truth.csv
        info = {"reused": True}
        tr = {r["gid"]: r for r in csv.DictReader((work / "truth.csv").open())}
        for s in specs:
            s.h_pix, s.snr_synth, s.sky = (float(tr[s.gid][k]) for k in ("h_pix", "snr_synth", "sky"))
    # dropped and counted, as in the pilot (`dropped_h_nan`): a donor no disc can match (Petrosian
    # radius under the PSF's, or over the stamp) renders all-NaN. Kept out of both methods' inputs —
    # left in, SpArcFiRe's missing row would read as an abstention.
    info["dropped_h_nan"] = [s.gid for s in specs if not np.isfinite(s.h_pix)]
    for g in info["dropped_h_nan"]:
        for sub, suffix in (("fits", ".fits"), ("fits_half", ".fits"), ("deproj", ".fits"), ("elps", "_elps.txt")):
            (work / sub / f"{g}{suffix}").unlink(missing_ok=True)
    if info["dropped_h_nan"]:
        print(f"[grid] stage {stage}: dropped (h undefined): {info['dropped_h_nan']}", flush=True)
    specs = [s for s in specs if np.isfinite(s.h_pix)]
    half = [s for s in specs if _half(s)]
    (work / "fits_half").mkdir(exist_ok=True)
    for s in half:
        dst = work / "fits_half" / f"{s.gid}.fits"
        if not dst.exists():
            shutil.copy2(work / "fits" / f"{s.gid}.fits", dst)
    runs = {}
    elps = work / "elps" if stage > 1 else None
    for tag, d, setting, e, n in ((f"bb2_grid_s{stage}_default", work / "fits", "default", elps, len(specs)),
                                  (f"bb2_grid_s{stage}_runsh", work / "fits_half", "runsh", None, len(half)),
                                  (f"bb2_grid_s{stage}_runsh_amt15", work / "fits_half", "runsh_amt15", None, len(half))):
        runs[setting] = {"reused": True} if _sf_done(tag, n) else run_sparcfire(d, tag, setting, e)
        print(f"[grid] stage {stage} {setting}: {runs[setting]}", flush=True)
    sf = {k: read_sparcfire(f"bb2_grid_s{stage}_{k}") for k in ("default", "runsh", "runsh_amt15")}
    t0 = time.time()
    p2 = _p2(work, specs, "fits" if stage == 1 else "deproj")
    runs["p2dfft_s"] = time.time() - t0
    rows = []
    for s in specs:
        r = {"gid": s.gid, "stage": stage, "kind": s.kind, "quintile": s.quintile, "v1": s.v1, "pitch": s.pitch,
             "arms": s.arms, "contrast": s.contrast, "variant": s.variant, "q": s.q, "half": _half(s),
             "h_pix": s.h_pix, "snr_synth": s.snr_synth}
        for k in ("default", "runsh", "runsh_amt15"):
            g = sf[k].get(s.gid, {})
            r[f"sf_{k}"] = g.get("dco", np.nan) if g else np.nan
            if k == "default":
                r["sf_all_arcs"], r["sf_top2"] = g.get("all_arcs", np.nan), g.get("top2", "")
        r["p2_auto"] = abs(p2["auto"].get(s.gid, {}).get("pa", np.nan))
        r["p2_oracle"] = abs(p2["oracle"].get(s.gid, {}).get("pa", np.nan))
        rows.append(r)
    return {"build": info, "runs": runs, "rows": rows}


def _breakdown(rows: list[dict], key: str, meas: str, det_fn) -> dict:
    out = {}
    for v in sorted({r[key] for r in rows}, key=str):
        g = [r for r in rows if r[key] == v]
        t = np.array([r["pitch"] for r in g])
        m = np.array([r[meas] for r in g], float)
        out[str(v)] = cell_stats(t, m, det_fn(m))
    return out


def grid_analysis(stage: int, rows: list[dict], edges: list[float]) -> dict:
    sp = [r for r in rows if r["kind"] == "spiral"]
    nu = [r for r in rows if r["kind"] != "spiral"]
    col = lambda rs, k: np.array([r[k] for r in rs], float)  # noqa: E731
    truth, quint = col(sp, "pitch"), np.array([r["quintile"] for r in sp])
    mS, mF, mFo = col(sp, "sf_default"), col(sp, "p2_auto"), col(sp, "p2_oracle")
    dS = np.isfinite(mS)
    seed = 100 + stage
    out: dict = {"n_spiral": len(sp), "n_null": len(nu)}
    out["primary"] = stage_verdicts(truth, mS, dS, mF, p2dfft_detections(mF), quint, edges, seed)
    out["sensitivity_p2dfft_oracle"] = stage_verdicts(truth, mS, dS, mFo, p2dfft_detections(mFo), quint, edges, seed)
    h = np.array([r["half"] for r in sp])
    for k in ("runsh", "runsh_amt15"):
        m = col(sp, f"sf_{k}")[h]
        out[f"sensitivity_sparcfire_{k}"] = stage_verdicts(truth[h], m, np.isfinite(m), mF[h],
                                                           p2dfft_detections(mF[h]), quint[h], edges, seed)
    out["common_support"] = {"pooled": common_support(truth, mS, dS, mF, seed),
                             "by_quintile": {int(q): common_support(truth[quint == q], mS[quint == q], dS[quint == q],
                                                                    mF[quint == q], seed)
                                             for q in QUINTILES}}
    co = dS & np.isfinite(mF)
    out["diagnostic"] = diagnostic(mS[co], mF[co], seed)
    out["peng"] = {k: v["state"] for k, v in peng(truth, mS, col(sp, "sf_all_arcs"), dS, quint, seed).items()}
    out["nulls"] = {"all": null_answers(col(nu, "sf_default"), col(nu, "p2_auto")),
                    **{kind: null_answers(col([r for r in nu if r["kind"] == kind], "sf_default"),
                                          col([r for r in nu if r["kind"] == kind], "p2_auto"))
                       for kind in ("armless", "noise")},
                    "by_quintile": {int(q): null_answers(col([r for r in nu if r["quintile"] == q], "sf_default"),
                                                         col([r for r in nu if r["quintile"] == q], "p2_auto"))
                                    for q in QUINTILES}}
    fin = np.isfinite
    out["breakdown"] = {m: {k: _breakdown(sp, k, m, fin if m.startswith("sf") else p2dfft_detections)
                            for k in ("pitch", "arms", "contrast", "variant")}
                        for m in ("sf_default", "p2_auto")}
    out["exploratory_top2_agree"] = {  # EXPLORATORY (user, 2026-09-25): enters no verdict
        int(q): {"spiral": float(np.mean([r["sf_top2"] == "agree" for r in sp if r["quintile"] == q])),
                 "armless": float(np.mean([r["sf_top2"] == "agree" for r in nu
                                           if r["quintile"] == q and r["kind"] == "armless"])),
                 "noise": float(np.mean([r["sf_top2"] == "agree" for r in nu
                                         if r["quintile"] == q and r["kind"] == "noise"]))}
        for q in QUINTILES}
    if stage == 3:
        out["bb3_reference"] = bb3_reference(out["primary"]["rank"])
    return out


def grid(pool: list[dict], info: dict) -> dict:
    edges = info["quintile_edges"]
    res: dict = {"quintile_edges": edges}
    for stage in (1, 2, 3):
        st = grid_stage(pool, stage)
        with (OUT / f"grid_s{stage}_rows.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(st["rows"][0]))
            w.writeheader()
            w.writerows(st["rows"])
        res[f"stage{stage}"] = {"runs": st["runs"], "build": st["build"], **grid_analysis(stage, st["rows"], edges)}
        (OUT / "grid.json").write_text(json.dumps(res, indent=1, default=float))
        print(f"[grid] stage {stage}: accuracy {res[f'stage{stage}']['primary']['accuracy']['verdict']}, "
              f"rank {res[f'stage{stage}']['primary']['rank']['verdict']}", flush=True)
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arg = sys.argv[1]
    if arg == "--plant-logic":
        r = plant_logic()
        (OUT / "plant_logic.json").write_text(json.dumps(r, indent=1, default=float))
    elif arg == "--plant-nulls":
        pool, _ = donor_pool()
        r = plant_nulls(pool)
        (OUT / "plant_nulls.json").write_text(json.dumps(r, indent=1, default=float))
    elif arg == "--grid":
        pool, info = donor_pool()
        r = grid(pool, info)
        r = {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ("primary", "runs")})
             for k, v in r.items()}
    elif arg == "--pilot":
        pool, _ = donor_pool()
        r = pilot(pool)
        r = {k: v for k, v in r.items() if k != "rows"}
    else:
        raise SystemExit(f"unknown {arg}")
    print(json.dumps(r, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
