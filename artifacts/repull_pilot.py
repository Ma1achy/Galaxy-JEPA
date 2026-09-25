"""The aligned re-pull pilot (repull_findings.md §Pilot): sample, then the pre-registered checks.

  sample   ~776 objects → .sciserver_work/repull_pilot_all_targets.csv (the cutter's input):
           250 per corpus stratified over the 6 camcols, 30 per corpus within 100 px of a frame
           edge (so part-off-frame), 200 of AA3a's galaxies (their v1 offsets are measured), and
           the 16 pretrain targets v1 failed to cut. Public SkyServer SQL only (no token).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sciserver_pull import _net_retry  # noqa: E402

from galaxy_jepa.data.metadata import run_sql

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / ".sciserver_work"
PILOT = WORK / "repull_pilot_all_targets.csv"
AA3A = REPO / "artifacts" / "out" / "aa3a_offsets.npz"
FRAME_ROWS, FRAME_COLS = 1489, 2048
EDGE_PX = 100
CUT_COLS = ("objID", "ra", "dec", "run", "camcol", "field", "rerun", "v1_ra", "v1_dec", "corpus", "group")


def _edge_dist(rowc: float, colc: float) -> float:
    return min(rowc, FRAME_ROWS - 1 - rowc, colc, FRAME_COLS - 1 - colc)


def _rowcol(ids: list[str]) -> dict[str, tuple[float, float]]:
    out = {}
    for i in range(0, len(ids), 250):
        q = ("SELECT CAST(objID AS varchar(20)) AS objID, rowc_r, colc_r FROM PhotoObjAll WHERE objID IN ("
             + ",".join(ids[i:i + 250]) + ")")
        for r in _net_retry(run_sql, q, timeout=300, _tries=6):
            out[r["objID"]] = (float(r["rowc_r"]), float(r["colc_r"]))
    return out


def sample() -> None:
    rng = np.random.default_rng(20260925)
    probe = list(csv.DictReader((WORK / "probe_v2_targets.csv").open(newline="")))
    pretrain = list(csv.DictReader((WORK / "pretrain_v2_targets.csv").open(newline="")))
    failed = list(csv.DictReader((WORK / "pretrain_v1_failed.csv").open(newline="")))
    aa3a = {str(i) for i in np.load(AA3A)["ids"]}

    def cut_row(r: dict, corpus: str, group: str) -> dict:
        v1 = (r["gz2_ra"], r["gz2_dec"]) if corpus == "probe" else (r["ra"], r["dec"])
        return {"objID": r["objID"], "ra": r["ra"], "dec": r["dec"], "run": r["run"], "camcol": r["camcol"],
                "field": r["field"], "rerun": r["rerun"], "v1_ra": v1[0], "v1_dec": v1[1],
                "corpus": corpus, "group": group}

    chosen: dict[str, dict] = {}
    aa = [r for r in probe if r["objID"] in aa3a]
    for r in (aa[i] for i in rng.choice(len(aa), 200, replace=False)):
        chosen[r["objID"]] = cut_row(r, "probe", "aa3a")
    for corpus, rows in (("probe", probe), ("pretrain", pretrain)):
        by_cc: dict[str, list[dict]] = {}
        for r in rows:
            if r["objID"] not in chosen:
                by_cc.setdefault(r["camcol"], []).append(r)
        assert sorted(by_cc) == list("123456"), sorted(by_cc)
        for k, cc in enumerate(sorted(by_cc)):
            pool = by_cc[cc]
            n = 42 if k < 4 else 41  # 250 over 6 camcols
            for i in rng.choice(len(pool), n, replace=False):
                chosen[pool[i]["objID"]] = cut_row(pool[i], corpus, "random")
    # edge objects: probe carries rowc/colc; pretrain's come from one small public query
    cand = [r for r in probe if r["objID"] not in chosen
            and _edge_dist(float(r["rowc_r"]), float(r["colc_r"])) < EDGE_PX]
    for i in rng.choice(len(cand), 30, replace=False):
        chosen[cand[i]["objID"]] = cut_row(cand[i], "probe", "edge")
    sub = [pretrain[i] for i in rng.choice(len(pretrain), 1500, replace=False) if pretrain[i]["objID"] not in chosen]
    rc = _rowcol([r["objID"] for r in sub])
    cand = [r for r in sub if r["objID"] in rc and _edge_dist(*rc[r["objID"]]) < EDGE_PX]
    for i in rng.choice(len(cand), 30, replace=False):
        chosen[cand[i]["objID"]] = cut_row(cand[i], "pretrain", "edge")
    for r in failed:
        chosen[r["objID"]] = cut_row(r, "pretrain", "v1_failed")
    rows = sorted(chosen.values(), key=lambda r: int(r["objID"]))  # objID order keeps fields adjacent
    with PILOT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CUT_COLS)
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(f"pilot: {len(rows)} targets → {PILOT}")
    print(Counter((r["corpus"], r["group"]) for r in rows))
    print("camcols:", Counter((r["corpus"], r["camcol"]) for r in rows if r["group"] == "random"))



# ── the pre-registered checks (repull_findings.md §Pilot). Identical code for plants and data. ──

PAD = 32  # the cutter's MARGIN
C = 127.5
SMOOTH_SIG = 1.3  # px; core centroid smoothing (≈ r seeing 1.2″), the same for both corpora
STAR_SNR = 20.0
STAR_EXCL = 40  # px about the target excluded from the star search
MIN_STARS, MIN_EDGE, MIN_PAIRED = 200, 30, 300
TOL_MED = 0.03  # px, median band-to-band star offset
TOL_SLOPE = 0.1  # residual share of v1's misregistration
TOL_RHO = 0.1  # |Spearman| magnitude bar for "depends on"
PLANT_RHO = 0.2
TOL_BORDER = 0.1  # σ, rms(v2 − margin64) in the outer 4 px
TOL_PADRING = (0.9, 1.1)  # variance ratio, 1-4 px inside the pad vs interior sky
TOL_OLDNEW = 0.1  # σ, median rms(v2 − shifted v1) in the centre
TOL_VAR = (0.98, 1.02)
TOL_HF = (0.95, 1.05)
TOL_LAG = 0.02
KS_P = 0.01
N_BOOT = 1000
BY_Q = 0.05


def _sigma(img: np.ndarray) -> float:
    v = img[np.isfinite(img) & (img != 0)]
    med = np.median(v)
    return float(1.4826 * np.median(np.abs(v - med)))


def _smooth(img: np.ndarray, sig: float) -> np.ndarray:
    from scipy import ndimage
    return ndimage.gaussian_filter(img, sig)


def sky_mask(img: np.ndarray, pad: np.ndarray) -> np.ndarray:
    """True on source-free sky: smoothed image under 2σ, dilated 3 px, pad excluded."""
    from scipy import ndimage
    s = _sigma(img)
    src = _smooth(img, 1.0) > 2 * s / 2.0  # a σ=1 smooth cuts the noise ~2×
    src = ndimage.binary_dilation(src, iterations=3)
    return ~src & ~pad


def find_stars(stamp: np.ndarray, pad: np.ndarray) -> list[tuple[int, int]]:
    """Compact, isolated, unsaturated point sources in r, away from the target and the pad."""
    from scipy import ndimage
    r = stamp[1]
    s = _sigma(r)
    sm = _smooth(r, SMOOTH_SIG)
    peaks = (sm == ndimage.maximum_filter(sm, 9)) & (sm > STAR_SNR * s / (2 * np.sqrt(np.pi) * SMOOTH_SIG))
    near_pad = ndimage.binary_dilation(pad, iterations=10)
    yy, xx = np.nonzero(peaks)
    out = []
    for y, x in zip(yy, xx, strict=True):
        if np.hypot(x - C, y - C) < STAR_EXCL or not (15 <= x < 241 and 15 <= y < 241) or near_pad[y, x]:
            continue
        w = r[y - 4:y + 5, x - 4:x + 5]
        if w.max() <= 0:
            continue
        f = w - np.median(r[y - 8:y + 9, x - 8:x + 9])
        tot = f.sum()
        if tot <= 0:
            continue
        g_y, g_x = np.mgrid[-4:5, -4:5]
        size = np.sqrt(max((f * (g_x ** 2 + g_y ** 2)).sum() / tot / 2, 0))
        if 0.6 < size < 2.2:  # a PSF (σ ≈ 1-1.8 px), not a galaxy
            out.append((int(y), int(x)))
    far = [p for p in out if all(np.hypot(p[0] - q[0], p[1] - q[1]) > 12 for q in out if q != p)]
    return far


def win_centroid(img: np.ndarray, y: int, x: int, sig: float = 1.5, it: int = 20) -> tuple[float, float]:
    """Gaussian-windowed iterative centroid (SExtractor XWIN-style): unbiased for a symmetric source."""
    yy, xx = np.mgrid[y - 7:y + 8, x - 7:x + 8]
    f = img[y - 7:y + 8, x - 7:x + 8] - np.median(img[y - 10:y + 11, x - 10:x + 11])
    cy, cx = float(y), float(x)
    for _ in range(it):
        w = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig ** 2)) * f
        n = w.sum()
        if n <= 0:
            return np.nan, np.nan
        ny_, nx_ = (w * yy).sum() / n, (w * xx).sum() / n
        cy, cx = cy + 2 * (ny_ - cy), cx + 2 * (nx_ - cx)
        if abs(ny_ - cy) > 7 or abs(nx_ - cx) > 7:
            return np.nan, np.nan
    return cy, cx


def star_offsets(stamp: np.ndarray, stars) -> np.ndarray:
    """(n, 4): per star, g−r and i−r offsets (dx, dy) in px."""
    rows = []
    for y, x in stars:
        c = [win_centroid(stamp[b], y, x) for b in range(3)]
        rows.append([c[0][1] - c[1][1], c[0][0] - c[1][0], c[2][1] - c[1][1], c[2][0] - c[1][0]])
    a = np.array(rows, float).reshape(-1, 4)
    return a[np.all(np.isfinite(a), axis=1) & np.all(np.abs(a) < 2, axis=1)]


def lag1(sky: np.ndarray, m: np.ndarray) -> float:
    p = np.where(m, sky - sky[m].mean(), 0.0)
    v = (p[m] ** 2).mean()
    h = (p[:, :-1] * p[:, 1:])[m[:, :-1] & m[:, 1:]].mean()
    u = (p[:-1, :] * p[1:, :])[m[:-1, :] & m[1:, :]].mean()
    return float((h + u) / 2 / v)


def hf_frac(img: np.ndarray, m: np.ndarray) -> float:
    """Share of sky noise power above 0.25 cycles/px (the top octave)."""
    a = np.where(m, img - img[m].mean(), 0.0)
    pw = np.abs(np.fft.fft2(a)) ** 2
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(256), np.fft.fftfreq(256)))
    return float(pw[k > 0.25].sum() / pw[k > 0].sum())


RING_R = 8  # radial bins [0,1) … [7,8) px


def ring_profile(img: np.ndarray, y: int, x: int) -> np.ndarray:
    """A star's azimuthal profile in 1-px bins about its windowed centroid, as a share of its flux
    within 8 px — the object 3b compares, paired, between the shifted stamp and unshifted v1."""
    yy, xx = np.mgrid[y - 9:y + 10, x - 9:x + 10]
    cy, cx = win_centroid(img, y, x)
    if not np.isfinite(cy):
        return np.full(RING_R, np.nan)
    r = np.hypot(yy - cy, xx - cx)
    f = img[y - 9:y + 10, x - 9:x + 10] - np.median(img[max(y - 14, 0):y + 15, max(x - 14, 0):x + 15])
    tot = f[r < RING_R].sum()
    if tot <= 0:
        return np.full(RING_R, np.nan)
    return np.array([f[(r >= a) & (r < a + 1)].mean() / tot for a in range(RING_R)])


def core_centroid(r: np.ndarray) -> tuple[float, float]:
    """SDSS-like: peak of the PSF-smoothed r image within 5 px of centre, 3×3 quadratic refinement."""
    sm = _smooth(r, SMOOTH_SIG)
    y0, x0 = 123, 123
    win = sm[y0:y0 + 10, x0:x0 + 10]
    iy, ix = np.unravel_index(np.argmax(win), win.shape)
    y, x = y0 + iy, x0 + ix
    z = sm[y - 1:y + 2, x - 1:x + 2]
    dy = 0.5 * (z[0, 1] - z[2, 1]) / (z[0, 1] - 2 * z[1, 1] + z[2, 1])
    dx = 0.5 * (z[1, 0] - z[1, 2]) / (z[1, 0] - 2 * z[1, 1] + z[1, 2])
    return y + float(np.clip(dy, -1, 1)), x + float(np.clip(dx, -1, 1))


def fshift(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    ky = np.fft.fftfreq(img.shape[0])[:, None]
    kx = np.fft.fftfreq(img.shape[1])[None, :]
    return np.fft.ifft2(np.fft.fft2(img) * np.exp(-2j * np.pi * (ky * dy + kx * dx))).real


def _spear(a, b, seed=0) -> dict:
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 20:
        return {"n": int(a.size), "rho": np.nan, "p": np.nan, "ci": [np.nan, np.nan]}
    r, p = spearmanr(a, b)
    rng = np.random.default_rng(seed)
    bs = [spearmanr(a[i], b[i])[0] for i in rng.integers(0, a.size, (N_BOOT, a.size))]
    return {"n": int(a.size), "rho": float(r), "p": float(p), "ci": np.nanpercentile(bs, [2.5, 97.5]).tolist()}


def _by(ps: list[float]) -> list[float]:
    """Benjamini–Yekutieli adjusted p."""
    ps = np.asarray(ps, float)
    m = ps.size
    c = np.sum(1.0 / np.arange(1, m + 1))
    o = np.argsort(ps)
    adj = np.empty(m)
    run = 1.0
    for rank, i in reversed(list(enumerate(o, 1))):
        run = min(run, ps[i] * m * c / rank)
        adj[i] = run
    return adj.tolist()


def _depends(tests: dict[str, dict], bar: float = TOL_RHO) -> dict:
    """D27: DEPENDS iff BY-significant AND |ρ| ≥ bar; significant-but-small is named, not a defect."""
    keys = [k for k in tests if np.isfinite(tests[k]["p"])]
    adj = dict(zip(keys, _by([tests[k]["p"] for k in keys]), strict=True)) if keys else {}
    out = {}
    for k, t in tests.items():
        sig = adj.get(k, 1.0) < BY_Q
        big = np.isfinite(t["rho"]) and abs(t["rho"]) >= bar
        out[k] = {**t, "p_by": adj.get(k), "state": "DEPENDS" if sig and big else
                  "SIGNIFICANT BELOW BAR" if sig else "INDEPENDENT"}
    return out


def measure(obj: dict) -> dict:
    """Every per-object quantity the checks use. obj: v2, v1 (or None), variants, log, corpus, group."""
    v2, log = obj["v2"], obj["log"]
    pad = (v2 == 0).all(axis=0)
    m: dict = {"id": log["object_id"], "corpus": obj["corpus"], "group": obj["group"],
               "valid_frac": float(log["valid_frac"]), "camcol": int(log["camcol"]), "run": int(log["run"])}
    for b in "gri":
        m[f"sx_{b}"], m[f"sy_{b}"] = float(log[f"{b}_sx"]), float(log[f"{b}_sy"])
        m[f"x_{b}"], m[f"y_{b}"] = float(log[f"{b}_x"]), float(log[f"{b}_y"])
        m[f"edge_{b}"] = float(log[f"{b}_edge_dist"])
        # 1a: where the log says the target now sits
        m[f"pos_{b}"] = (float(log[f"{b}_x"]) - int(log[f"{b}_ox"]) + m[f"sx_{b}"] - PAD,
                         float(log[f"{b}_y"]) - int(log[f"{b}_oy"]) + m[f"sy_{b}"] - PAD)
        # v1: where the target sat inside the v1 stamp (x_b of THIS coord, v1's own origin)
        m[f"v1rel_{b}"] = (float(log[f"{b}_x"]) - int(log[f"{b}_v1_ox"]), float(log[f"{b}_y"]) - int(log[f"{b}_v1_oy"]))
    m["v1_pred"] = [m["v1rel_g"][0] - m["v1rel_r"][0], m["v1rel_g"][1] - m["v1rel_r"][1],
                    m["v1rel_i"][0] - m["v1rel_r"][0], m["v1rel_i"][1] - m["v1rel_r"][1]]
    stars = find_stars(v2, pad)
    m["stars_v2"] = star_offsets(v2, stars).tolist()
    if obj.get("v1") is not None:  # 3b: per star, profile(shifted) − profile(same star in v1), per band
        v1s = obj["v1"]
        for name, src in (("ring_d", v2), ("ring_d_bilinear", obj["variants"].get("bilinear"))):
            if src is None:
                continue
            m[name] = {}
            for k, b in enumerate("gri"):
                rel = m["v1rel_" + b]
                rows = []
                for y, x in stars:  # the same star in v1 sits (C − v1rel) earlier
                    y1, x1 = int(round(y - (C - rel[1]))), int(round(x - (C - rel[0])))
                    if not (15 <= x1 < 241 and 15 <= y1 < 241):
                        continue
                    rows.append((ring_profile(src[k], y, x) - ring_profile(v1s[k], y1, x1)).tolist())
                m[name][b] = rows
    sm = [sky_mask(v2[k], pad) for k in range(3)]
    m["sigma"] = [_sigma(v2[k]) for k in range(3)]
    m["lag1"] = [lag1(v2[k], sm[k]) for k in range(3)]
    m["hf"] = [hf_frac(v2[k], sm[k]) for k in range(3)]
    if "bilinear" in obj["variants"]:
        bl = obj["variants"]["bilinear"]
        m["lag1_bilinear"] = [lag1(bl[k], sm[k]) for k in range(3)]
        m["hf_bilinear"] = [hf_frac(bl[k], sm[k]) for k in range(3)]
    ring4 = np.zeros((256, 256), bool)
    ring4[:4, :] = ring4[-4:, :] = ring4[:, :4] = ring4[:, -4:] = True
    for name in ("margin64", "nomargin"):
        ref = obj["variants"].get("margin64")
        cmp_ = v2 if name == "margin64" else obj["variants"].get("nomargin")
        if ref is not None and cmp_ is not None:
            k = ring4 & ~pad
            m[f"border_{'v2' if name == 'margin64' else 'nomargin'}"] = [
                float(np.sqrt(np.mean((cmp_[b][k] - ref[b][k]) ** 2)) / m["sigma"][b]) for b in range(3)]
    if pad.any() and not pad.all():
        from scipy import ndimage
        d = ndimage.distance_transform_edt(~pad)
        near = (d >= 1) & (d <= 4)
        far = (d > 20)
        m["padring"] = [float(np.var(v2[b][near & sm[b]]) / np.var(v2[b][far & sm[b]]))
                        if (near & sm[b]).sum() > 50 and (far & sm[b]).sum() > 500 else np.nan for b in range(3)]
        if "padring_plant" in obj:
            pl = obj["padring_plant"]
            m["padring_plant"] = [float(np.var(pl[b][near & sm[b]]) / np.var(pl[b][far & sm[b]]))
                                  if (near & sm[b]).sum() > 50 and (far & sm[b]).sum() > 500 else np.nan
                                  for b in range(3)]
    cy, cx = core_centroid(v2[1])
    m["centre"] = [cx - C, cy - C]
    if "centre_plant" in obj:
        py, px = core_centroid(obj["centre_plant"])
        m["centre_plant"] = [px - C, py - C]
    v1 = obj.get("v1")
    if v1 is not None:
        pad1 = (v1 == 0).all(axis=0)
        stars1 = find_stars(v1, pad1)
        m["stars_v1"] = star_offsets(v1, stars1).tolist()
        cen = np.zeros((256, 256), bool)
        cen[40:216, 40:216] = True
        keep = cen & ~pad & ~pad1
        on, pl = [], []
        for b in range(3):
            rel = m["v1rel_" + "gri"[b]]
            dx, dy = C - rel[0], C - rel[1]
            sh = fshift(v1[b].astype(float), dy, dx)
            on.append(float(np.sqrt(np.mean((v2[b][keep] - sh[keep]) ** 2)) / m["sigma"][b]))
            if b == 0:  # plant: g moved a further 0.5 px
                sh2 = fshift(v1[b].astype(float), dy, dx + 0.5)
                pl.append(float(np.sqrt(np.mean((v2[b][keep] - sh2[keep]) ** 2)) / m["sigma"][b]))
        m["oldnew"], m["oldnew_plant_g"] = on, pl[0]
        sm1 = [sky_mask(v1[k], pad1) for k in range(3)]
        m["var_ratio"] = [float(np.var(v2[k][sm[k] & sm1[k]]) / np.var(v1[k][sm[k] & sm1[k]])) for k in range(3)]
        m["lag1_v1"] = [lag1(v1[k], sm1[k]) for k in range(3)]
        m["hf_v1"] = [hf_frac(v1[k], sm1[k]) for k in range(3)]
    return m


def _hsub(ms, b):
    return [abs(x[f"sx_{b}"]) + abs(x[f"sy_{b}"]) for x in ms]


def checks(ms: list[dict]) -> dict:
    """The six checks + 3b, their in-run plants, and the pilot state (precedence in repull_findings.md)."""
    res: dict = {}
    rnd = [m for m in ms if m["group"] in ("random", "aa3a", "edge")]
    paired = [m for m in rnd if "oldnew" in m]
    # 1a recorded registration (+ plant: r's logged shift perturbed by (0.3, −0.2))
    dev = max(max(abs(m[f"pos_{b}"][0] - C), abs(m[f"pos_{b}"][1] - C)) for m in rnd for b in "gri")
    pm = rnd[0]
    plant = (pm["pos_r"][0] + 0.3 - C, pm["pos_r"][1] - 0.2 - C)
    res["1a"] = {"max_dev_px": dev, "pass": dev <= 0.01,
                 "plant_recovered": plant, "plant_fired": abs(plant[0] - 0.3) <= 0.01 and abs(plant[1] + 0.2) <= 0.01}
    # 1b empirical registration: stars
    s2 = [(np.array(m["stars_v2"]).reshape(-1, 4), m["v1_pred"]) for m in rnd]
    st = np.concatenate([a for a, _ in s2]) if s2 else np.zeros((0, 4))
    pred2 = np.concatenate([np.tile(p, (len(a), 1)) for a, p in s2]) if s2 else np.zeros((0, 4))
    s1 = [(np.array(m["stars_v1"]).reshape(-1, 4), m["v1_pred"]) for m in paired]
    st1 = np.concatenate([a for a, _ in s1]) if s1 else np.zeros((0, 4))
    pred1 = np.concatenate([np.tile(p, (len(a), 1)) for a, p in s1]) if s1 else np.zeros((0, 4))

    def slope(y, x, seed=1):
        if len(y) < 20:
            return np.nan, [np.nan, np.nan]
        b = np.polyfit(x, y, 1)[0]
        rng = np.random.default_rng(seed)
        bs = [np.polyfit(x[i], y[i], 1)[0] for i in rng.integers(0, len(y), (N_BOOT, len(y)))]
        return float(b), np.percentile(bs, [2.5, 97.5]).tolist()

    comp = ["gr_dx", "gr_dy", "ir_dx", "ir_dy"]
    med = {c: float(np.median(st[:, j])) if len(st) else np.nan for j, c in enumerate(comp)}
    sl2 = {c: slope(st[:, j], pred2[:, j]) for j, c in enumerate(comp)}
    sl1 = {c: slope(st1[:, j], pred1[:, j]) for j, c in enumerate(comp)}
    res["1b"] = {"n_stars": int(len(st)), "n_stars_v1": int(len(st1)), "median_offset_px": med,
                 "spread_px": {c: float(1.4826 * np.median(np.abs(st[:, j] - np.median(st[:, j])))) if len(st) else np.nan
                               for j, c in enumerate(comp)},
                 "slope_v2_on_v1_pred": sl2, "slope_v1_on_v1_pred": sl1,
                 "insufficient": len(st) < MIN_STARS,
                 "pass": all(abs(med[c]) < TOL_MED and abs(sl2[c][0]) < TOL_SLOPE for c in comp),
                 "plant_fired": all(np.isfinite(sl1[c][0]) and sl1[c][0] > 1 - 2 * TOL_SLOPE * 1.5 for c in comp)}
    # 1c residual independence (per-stamp median star offset, |g−r|+|i−r| magnitude by component)
    per = [(m, np.median(np.array(m["stars_v2"]).reshape(-1, 4), axis=0)) for m in rnd if len(m["stars_v2"]) >= 1]
    tests = {}
    for j, c in enumerate(comp):
        y = [v[j] for _, v in per]
        tests[f"{c}~x_r"] = _spear(y, [m["x_r"] for m, _ in per])
        tests[f"{c}~y_r"] = _spear(y, [m["y_r"] for m, _ in per])
        tests[f"{c}~edge_r"] = _spear(y, [m["edge_r"] for m, _ in per])
        tests[f"{c}~camcol"] = _spear(y, [m["camcol"] for m, _ in per])
        tests[f"{c}~run"] = _spear(y, [m["run"] for m, _ in per])
        b = "g" if c.startswith("g") else "i"
        ax = 0 if c.endswith("dx") else 1
        tests[f"{c}~s_{b}"] = _spear(y, [m[f"s{'xy'[ax]}_{b}"] for m, _ in per])
    dep = _depends(tests)
    per1 = [(m, np.median(np.array(m["stars_v1"]).reshape(-1, 4), axis=0)) for m in paired if len(m["stars_v1"]) >= 1]
    pl = _depends({c: _spear([v[j] for _, v in per1], [m["v1_pred"][j] for m, _ in per1]) for j, c in enumerate(comp)})
    res["1c"] = {"tests": dep, "pass": all(t["state"] != "DEPENDS" for t in dep.values()),
                 "plant": pl, "plant_fired": all(t["state"] == "DEPENDS" for t in pl.values())}
    # 2 noise + resampling fingerprint
    two: dict = {}
    for k, b in enumerate("gri"):
        h = _hsub(rnd, b)
        two[f"lag1~h_{b}"] = _spear([m["lag1"][k] for m in rnd], h)
        two[f"hf~h_{b}"] = _spear([m["hf"][k] for m in rnd], h)
    dep2 = _depends(two)
    plant2 = {}
    for k, b in enumerate("gri"):
        h = _hsub(rnd, b)
        plant2[f"lag1~h_{b}"] = _spear([m.get("lag1_bilinear", [np.nan] * 3)[k] for m in rnd], h)
    plant2 = _depends(plant2, PLANT_RHO)
    vr = [float(np.median([m["var_ratio"][k] for m in paired])) if paired else np.nan for k in range(3)]
    hfr = [float(np.median([m["hf"][k] / m["hf_v1"][k] for m in paired])) if paired else np.nan for k in range(3)]
    dlag = [float(np.median([m["lag1"][k] - m["lag1_v1"][k] for m in paired])) if paired else np.nan for k in range(3)]
    res["2"] = {"tests": dep2, "var_ratio_median": vr, "hf_ratio_median": hfr, "lag1_delta_median": dlag,
                "insufficient": len(paired) < MIN_PAIRED,
                "pass": all(t["state"] != "DEPENDS" for t in dep2.values())
                and all(TOL_VAR[0] <= x <= TOL_VAR[1] for x in vr) and all(TOL_HF[0] <= x <= TOL_HF[1] for x in hfr)
                and all(abs(x) < TOL_LAG for x in dlag),
                "plant": plant2, "plant_fired": all(t["state"] == "DEPENDS" for t in plant2.values())}
    # 3 edges: border convergence (vs margin64) + pad-boundary variance
    bv2 = [float(np.median([m["border_v2"][k] for m in rnd if "border_v2" in m])) for k in range(3)]
    bnm = [float(np.median([m["border_nomargin"][k] for m in rnd if "border_nomargin" in m])) for k in range(3)]
    edge = [m for m in rnd if "padring" in m]
    prv = [float(np.nanmedian([m["padring"][k] for m in edge])) if edge else np.nan for k in range(3)]
    prp = [float(np.nanmedian([m["padring_plant"][k] for m in edge if "padring_plant" in m])) if edge else np.nan
           for k in range(3)]
    res["3"] = {"border_rms_v2_sigma": bv2, "border_rms_nomargin_sigma": bnm, "padring_var_ratio": prv,
                "padring_plant": prp, "n_edge": len(edge), "insufficient": len(edge) < MIN_EDGE,
                "pass": all(x <= TOL_BORDER for x in bv2) and all(TOL_PADRING[0] <= x <= TOL_PADRING[1] for x in prv),
                "plant_fired": all(x > TOL_BORDER for x in bnm) and all(x > TOL_PADRING[1] for x in prp)}
    # 3b star ringing: the paired radial residual profile (v2 − v1) at each radius vs |s_b|
    t3b, p3b = {}, {}
    for name, dst in (("ring_d", t3b), ("ring_d_bilinear", p3b)):
        for b in "gri":
            rows = [(r, abs(m[f"sx_{b}"]) + abs(m[f"sy_{b}"])) for m in paired if name in m for r in m[name][b]]
            if not rows:
                continue
            prof = np.array([r for r, _ in rows])
            h = [hh for _, hh in rows]
            for a in range(RING_R):
                dst[f"{b}_r{a}"] = _spear(prof[:, a], h)
    d3b, pl3b = _depends(t3b), _depends(p3b, PLANT_RHO)
    n3b = min((t["n"] for t in t3b.values()), default=0)
    res["3b"] = {"tests": d3b, "n_star_pairs": n3b, "insufficient": n3b < MIN_STARS,
                 "pass": all(t["state"] != "DEPENDS" for t in d3b.values()),
                 "plant": pl3b,  # fires iff, in every band, the bilinear residual depends on |s| at some radius
                 "plant_fired": bool(pl3b) and all(any(t["state"] == "DEPENDS" for k, t in pl3b.items() if k[0] == b)
                                                  for b in "gri")}
    # 4 old vs new
    on = [float(np.median([m["oldnew"][k] for m in paired])) if paired else np.nan for k in range(3)]
    onp = float(np.median([m["oldnew_plant_g"] for m in paired])) if paired else np.nan
    res["4"] = {"rms_sigma_median": on, "plant_g_rms_sigma": onp, "insufficient": len(paired) < MIN_PAIRED,
                "pass": all(x <= TOL_OLDNEW for x in on), "plant_fired": onp > 2 * TOL_OLDNEW}
    # 5 centring: KS corpus equality decisive; median |Δ| report-only
    from scipy.stats import ks_2samp
    ran = [m for m in rnd if m["group"] == "random"]
    rad = {c: np.array([np.hypot(*m["centre"]) for m in ran if m["corpus"] == c]) for c in ("probe", "pretrain")}
    ks = ks_2samp(rad["probe"], rad["pretrain"])
    radp = np.array([np.hypot(*m["centre_plant"]) for m in ran if m["corpus"] == "probe" and "centre_plant" in m])
    ksp = ks_2samp(radp, rad["pretrain"]) if radp.size else None
    res["5"] = {"ks_p": float(ks.pvalue), "median_abs_px": {c: float(np.median(v)) for c, v in rad.items()},
                "pass": ks.pvalue > KS_P, "plant_ks_p": float(ksp.pvalue) if ksp else np.nan,
                "plant_fired": bool(ksp and ksp.pvalue < KS_P)}
    return res


def verdict(res: dict, per_corpus: dict[str, dict] | None = None) -> dict:
    """Precedence (D27): INCONCLUSIVE (a plant did not fire) > DEFECT > INSUFFICIENT > READY."""
    keys = ("1a", "1b", "1c", "2", "3", "3b", "4", "5")
    unfired = [k for k in keys if not res[k]["plant_fired"]]
    failed = [k for k in keys if not res[k]["pass"]]
    thin = [k for k in keys if res[k].get("insufficient")]
    six = []
    if per_corpus:  # 6: the same state for each check in each corpus
        six = [k for k in keys if k != "5" and len({per_corpus[c][k]["pass"] for c in per_corpus}) > 1]
    if unfired:
        state = f"INCONCLUSIVE (plant did not fire: {', '.join(unfired)})"
    elif failed or six:
        state = f"DEFECT ({', '.join(failed + [f'6:{k}' for k in six])})"
    elif thin:
        state = f"INSUFFICIENT ({', '.join(thin)})"
    else:
        state = "READY"
    return {"state": state, "failed": failed, "check6_differs": six, "unfired": unfired, "insufficient": thin}


# ── plants built onto an object (identical for synthetic D28 and the pilot data) ───────────────

def with_plants(obj: dict, rng) -> dict:
    """Pad-ring plant: a damped ±0.6σ oscillation along the pad boundary (ringing, planted).
    Centring plant: the r band moved 0.5 px in a random direction (probe only; v1-sized)."""
    v2 = obj["v2"]
    pad = (v2 == 0).all(axis=0)
    if pad.any() and not pad.all():
        from scipy import ndimage
        d = ndimage.distance_transform_edt(~pad)
        wave = np.where(pad, 0.0, np.cos(np.pi * d) * np.exp(-(d - 1) / 3.0))
        obj["padring_plant"] = np.stack([v2[b] + 0.6 * _sigma(v2[b]) * wave for b in range(3)])
    if obj["corpus"] == "probe":
        t = rng.uniform(0, 2 * np.pi)
        obj["centre_plant"] = fshift(v2[1].astype(float), 0.5 * np.sin(t), 0.5 * np.cos(t))
    return obj


# ── D28: synthetic frames through the real cutter and the identical checks ─────────────────────

def _synthetic_objects(n_rand: int = 150, n_edge: int = 30, seed: int = 7, misregister_v2: float = 0.0) -> list[dict]:
    from astropy.coordinates import SkyCoord
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS
    from astropy.wcs.utils import skycoord_to_pixel
    from scipy.special import erf

    import sciserver_cut_v2 as CUT

    CUT.VARIANTS = ("v2", "nomargin", "bilinear", "margin64")
    rng = np.random.default_rng(seed)
    H = Wd = 800
    xe = np.arange(Wd + 1) - 0.5

    def blob(x0, y0, sig, flux):
        fx = np.diff(erf((xe - x0) / (sig * np.sqrt(2)))) / 2
        fy = np.diff(erf((xe - y0) / (sig * np.sqrt(2)))) / 2
        return flux * fy[:, None] * fx[None, :]

    objs = []
    for k in range(2 * (n_rand + n_edge)):
        corpus = "probe" if k % 2 == 0 else "pretrain"
        group = "edge" if k >= 2 * n_rand else "random"
        ra0, dec0 = 150.0 + k * 0.5, 2.0
        wcs = {}
        for b in "gri":
            w = WCS(naxis=2)
            w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            w.wcs.crval = [ra0, dec0]
            w.wcs.crpix = [400.5 + rng.uniform(-4, 4), 400.5 + rng.uniform(-4, 4)]
            s = 0.396 / 3600
            w.wcs.cd = [[-s, 0], [0, s]]
            wcs[b] = w
        if group == "edge":
            tx, ty = Wd - rng.uniform(20, 90), rng.uniform(200, 600)
        else:
            tx, ty = rng.uniform(250, 550), rng.uniform(250, 550)
        tcoord = wcs["r"].pixel_to_world(tx, ty)
        star_sky = [wcs["r"].pixel_to_world(rng.uniform(5, Wd - 5), rng.uniform(5, H - 5)) for _ in range(60)]
        fluxes = 10 ** rng.uniform(2.3, 3.6, len(star_sky))
        frames = {}
        for bi, b in enumerate("gri"):
            sig = (1.35, 1.25, 1.2)[bi] * rng.uniform(0.85, 1.15)
            img = rng.normal(0, 1.0, (H, Wd))
            gx, gy = (float(t) for t in skycoord_to_pixel(tcoord, wcs[b], origin=0, mode="all"))
            yy, xx = np.mgrid[:H, :Wd]
            img += 30 * np.exp(-np.hypot(xx - gx, yy - gy) / 4.0)
            for c, f in zip(star_sky, fluxes, strict=True):
                sx_, sy_ = (float(t) for t in skycoord_to_pixel(c, wcs[b], origin=0, mode="all"))
                if b != "r":
                    sx_ += misregister_v2 + rng.normal(0, 0.02)
                    sy_ += rng.normal(0, 0.02)
                img += blob(sx_, sy_, sig, f)
            frames[b] = (f"frame-{b}-{k}", img, wcs[b], f"{k:064d}")
        CUT._frame = lambda band, row, fr=frames: fr[band]
        ra, dec = repr(float(tcoord.ra.deg)), repr(float(tcoord.dec.deg))
        row = {"objID": str(10_000 + k), "ra": ra, "dec": dec, "v1_ra": ra, "v1_dec": dec,
               "run": str(int(rng.integers(100, 8000))), "camcol": str(1 + k % 6), "field": "11", "rerun": "301"}
        oid, arrs, log, _, err = CUT.cut_one(row)
        assert not err, err
        v1 = np.stack([Cutout2D(frames[b][1], tcoord, size=256, wcs=wcs[b], mode="partial", fill_value=0.0).data
                       for b in "gri"]).astype(np.float32)
        obj = {"v2": arrs["v2"], "v1": v1, "variants": {v: arrs[v] for v in CUT.VARIANTS[1:]},
               "log": log, "corpus": corpus, "group": group}
        objs.append(with_plants(obj, rng))
    return objs


def plant() -> dict:
    """D28 before the hash: every check and every in-run plant reaches its state on synthetics,
    and a synthetic misregistered 'v2' (g moved 0.3 px in the frame) reads DEFECT."""
    import json
    out = {}
    for name, mis in (("aligned", 0.0), ("misregistered_g_0.3px", 0.3)):
        objs = _synthetic_objects(misregister_v2=mis)
        ms = [measure(o) for o in objs]
        res = checks(ms)
        per = {c: checks([m for m in ms if m["corpus"] == c]) for c in ("probe", "pretrain")}
        out[name] = {"verdict": verdict(res, per), "checks": res}
        print(name, "→", out[name]["verdict"]["state"], flush=True)
    (REPO / "artifacts" / "out" / "repull_pilot_plant.json").write_text(json.dumps(out, indent=1, default=float))
    return out



# ── the pilot data ──────────────────────────────────────────────────────────────────────────────

PILOT_OUT = REPO / "data" / "repull_pilot"


def _fits(path: Path) -> np.ndarray:
    from astropy.io import fits
    with fits.open(path) as h:
        return np.asarray(h[0].data, dtype=np.float32)


def analyse() -> dict:
    """The pre-registered pilot analysis (repull_findings.md §Pilot), on the cut pilot stamps."""
    import json
    tgt = {r["objID"]: r for r in csv.DictReader(PILOT.open(newline=""))}
    logs = list(csv.DictReader((PILOT_OUT / "cut_log.csv").open(newline="")))
    failed = list(csv.DictReader((PILOT_OUT / "failed.csv").open(newline="")))
    rng = np.random.default_rng(20260926)
    ms = []
    for i, log in enumerate(logs):
        t = tgt[log["object_id"]]
        v1p = REPO / "data" / t["corpus"] / f"{log['object_id']}.fits"
        obj = {"v2": _fits(PILOT_OUT / f"{log['object_id']}.fits"),
               "v1": _fits(v1p) if t["group"] != "v1_failed" and v1p.exists() else None,
               "variants": {v: _fits(PILOT_OUT / "variants" / v / f"{log['object_id']}.fits")
                            for v in ("nomargin", "bilinear", "margin64")},
               "log": log, "corpus": t["corpus"], "group": t["group"]}
        ms.append(measure(with_plants(obj, rng)))
        if (i + 1) % 100 == 0:
            print(f"  measured {i + 1}/{len(logs)}", flush=True)
    res = checks(ms)
    per = {c: checks([m for m in ms if m["corpus"] == c]) for c in ("probe", "pretrain")}
    v16 = {r["objID"]: "cut" if any(lg["object_id"] == r["objID"] for lg in logs) else
           next((f["error"] for f in failed if f["object_id"] == r["objID"]), "absent")
           for r in tgt.values() if r["group"] == "v1_failed"}
    out = {"verdict": verdict(res, per), "checks": res, "per_corpus": per, "n_cut": len(logs),
           "n_failed": len(failed), "failed": failed, "v1_failed_16": v16}
    (REPO / "artifacts" / "out" / "repull_pilot.json").write_text(json.dumps(out, indent=1, default=float))
    print(out["verdict"])
    return out


# ── Amendment A (post hoc, D27; repull_findings.md §Pilot amendment A) ──────────────────────────
# 1b/1c: v2's star offsets must equal v1's minus v1's snapping (the SDSS band-to-band WCS residual
# both cutters inherit), per star, within TOL_PAIRED. 2: nanmedian, the dropped stamps named.
# 5: a KS on fresh probe vs pretrain drawn to match it on magnitude, size and SNR.

TOL_PAIRED = 0.02  # px (user, Stop 1)
PLANT_PAIRED = 0.05  # px: g moved in v2 — about the camcol residual's size
PLANT_CAMCOL = 3
SMD_MAX = 0.1  # matched-sample balance, standardised mean difference per covariate
SUPPORT_RAD, SUPPORT_MAG = 5.0, (14.0, 19.0)  # pretrain's selection box
CHECK5 = WORK / "repull_check5_all_targets.csv"
CHECK5_OUT = REPO / "data" / "repull_check5"
COMP = ("gr_dx", "gr_dy", "ir_dx", "ir_dy")


def _covariates(corpus: str, ids: set[str] | None = None):
    import pandas as pd
    t = pd.read_csv(REPO / "data" / corpus / "metadata.csv", dtype={"object_id": str},
                    usecols=["object_id", "modelMag_r", "petroRad_r", "snr_r"])
    t = t[(t.petroRad_r > 0) & (t.snr_r > 0) & t.modelMag_r.between(5, 30)]
    if ids is not None:
        t = t[t.object_id.isin(ids)]
    return t.assign(logsize=np.log(t.petroRad_r), logsnr=np.log(t.snr_r)).set_index("object_id")


def _match(probe, pool, rng) -> tuple[list[str], dict]:
    """Nearest pretrain neighbour per probe object, without replacement, on standardised
    (modelMag_r, log petroRad_r, log snr_r); the balance (SMD per covariate) is returned."""
    from scipy.spatial import cKDTree
    cov = ["modelMag_r", "logsize", "logsnr"]
    mu, sd = probe[cov].mean(), probe[cov].std()
    tree = cKDTree(((pool[cov] - mu) / sd).to_numpy())
    taken: set[int] = set()
    out = []
    for oid in rng.permutation(probe.index.to_numpy()):
        q, k, j = ((probe.loc[oid, cov] - mu) / sd).to_numpy(), 64, None
        while j is None and len(taken) < len(pool):
            _, idx = tree.query(q, k=min(k, len(pool)))
            j = next((int(i) for i in np.atleast_1d(idx) if int(i) not in taken), None)
            k *= 4
        if j is None:
            break  # pool exhausted (only in a small dry-run pool)
        taken.add(j)
        out.append(pool.index[j])
    m = pool.loc[out]
    smd = {c: float((probe[c].mean() - m[c].mean()) / np.sqrt((probe[c].var() + m[c].var()) / 2)) for c in cov}
    return out, smd


def sample_check5() -> None:
    """Fresh data: 250 random probe objects not in the pilot, 250 pretrain matched to them, and
    the 55 probe objects whose GZ2 position is 1″–3″ from PhotoObj (the finalisation flag)."""
    rng = np.random.default_rng(20260927)
    used = {r["objID"] for r in csv.DictReader(PILOT.open(newline=""))}
    probe = {r["objID"]: r for r in csv.DictReader((WORK / "probe_v2_all_targets.csv").open(newline=""))}
    flagged = [o for o, r in probe.items() if r["gz2_sep_arcsec"] and float(r["gz2_sep_arcsec"]) > 1.0]
    assert len(flagged) == 55, len(flagged)
    cand = sorted(o for o, r in probe.items() if o not in used and o not in set(flagged)
                  and float(r["gz2_sep_arcsec"] or 0) <= 1.0)
    pcov = _covariates("probe", set(cand))
    # common support: pretrain's own selection box (petroRad_r ≥ 5″, 14 ≤ r ≤ 19; 25% of the
    # probe lies below 5″, where no pretrain counterpart exists to match)
    pcov = pcov[(pcov.petroRad_r >= SUPPORT_RAD) & pcov.modelMag_r.between(*SUPPORT_MAG)]
    pick = list(rng.choice(pcov.index.to_numpy(), 250, replace=False))
    pool = _covariates("pretrain")
    pool = pool[~pool.index.isin(used)]
    matched, smd = _match(pcov.loc[pick], pool, rng)
    pre = {}
    want = set(matched)
    with (WORK / "pretrain_v2_all_targets.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["objID"] in want:
                pre[r["objID"]] = r
    rows = ([{**probe[o], "corpus": "probe", "group": "random"} for o in pick]
            + [{**probe[o], "corpus": "probe", "group": "sep_flag"} for o in flagged]
            + [{**pre[o], "corpus": "pretrain", "group": "matched"} for o in matched])
    rows.sort(key=lambda r: int(r["objID"]))
    cols = [*CUT_COLS[:-2], "gz2_sep_arcsec", "corpus", "group"]
    with CHECK5.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (REPO / "artifacts" / "out" / "repull_check5_sample.json").write_text(json.dumps({"n": len(rows), "smd": smd}, indent=1))
    print(f"check5 sample: {len(rows)} → {CHECK5}; SMD {smd}")


def paired_stars(v2: np.ndarray, v1: np.ndarray, m: dict) -> dict:
    """Per v2 star, the same star in v1 (r's in-stamp offset), both measured by `star_offsets`:
    d = off_v2 − (off_v1 − v1_pred). Also d for the plant: v2's g moved +PLANT_PAIRED px in x."""
    pad = (v2 == 0).all(axis=0)
    stars = find_stars(v2, pad)
    rel = m["v1rel_r"]
    g_moved = v2.copy()
    g_moved[0] = fshift(v2[0].astype(float), 0.0, PLANT_PAIRED)
    d, dp = [], []
    for y, x in stars:
        y1, x1 = int(round(y - C + rel[1])), int(round(x - C + rel[0]))
        if not (15 <= x1 < 241 and 15 <= y1 < 241):
            continue
        a, b, p = star_offsets(v2, [(y, x)]), star_offsets(v1, [(y1, x1)]), star_offsets(g_moved, [(y, x)])
        if len(a) and len(b) and len(p):
            base = b[0] - np.asarray(m["v1_pred"])
            d.append((a[0] - base).tolist())
            dp.append((p[0] - base).tolist())
    return {"paired_d": d, "paired_d_plant": dp}


def _med_ci(x: np.ndarray, seed: int = 2) -> tuple[float, list[float]]:
    rng = np.random.default_rng(seed)
    bs = [np.median(x[i]) for i in rng.integers(0, len(x), (N_BOOT, len(x)))]
    return float(np.median(x)), np.percentile(bs, [2.5, 97.5]).tolist()


def _paired_block(ms: list[dict], key: str) -> dict:
    """Median d per component, pooled and per camcol; a camcol-3-only plant rides on `key`."""
    def stack(sel):
        a = [np.array(m[key]).reshape(-1, 4) for m in sel if m.get(key)]
        return np.concatenate(a) if a else np.zeros((0, 4))
    pooled = stack(ms)
    out = {"n_pairs": int(len(pooled)),
           "pooled": {c: _med_ci(pooled[:, j]) if len(pooled) else (np.nan, [np.nan, np.nan]) for j, c in enumerate(COMP)},
           "camcol": {}}
    for cc in range(1, 7):
        s = stack([m for m in ms if m["camcol"] == cc])
        out["camcol"][cc] = {"n_pairs": int(len(s)),
                             **{c: float(np.median(s[:, j])) if len(s) else np.nan for j, c in enumerate(COMP)}}
    return out


def _camcol_plant(ms: list[dict]) -> list[dict]:
    """camcol PLANT_CAMCOL's stamps take the g-moved d; every other camcol keeps its own."""
    return [{**m, "paired_d": m["paired_d_plant"]} if m["camcol"] == PLANT_CAMCOL else m for m in ms]


def _within(block: dict) -> tuple[bool, list[str]]:
    bad = [f"pooled:{c}" for c in COMP if not abs(block["pooled"][c][0]) <= TOL_PAIRED]
    bad += [f"camcol{cc}:{c}" for cc, v in block["camcol"].items() for c in COMP if not abs(v[c]) <= TOL_PAIRED]
    return not bad, bad


def checks_amended(ms: list[dict]) -> dict:
    """The hashed checks, with 1b, 1c and 2 as amended. Check 5 is set by `check5_matched`."""
    res = checks(ms)
    rnd = [m for m in ms if m["group"] in ("random", "aa3a", "edge")]
    pr = [m for m in rnd if "paired_d" in m]
    real = _paired_block(pr, "paired_d")
    plant_all = _paired_block(pr, "paired_d_plant")
    plant_cc = _paired_block(_camcol_plant(pr), "paired_d")
    ok, bad = _within(real)
    # 1b′: the hashed slope bar (the snapping is gone) + the pooled paired median within TOL_PAIRED
    slope_ok = all(abs(res["1b"]["slope_v2_on_v1_pred"][c][0]) < TOL_SLOPE for c in COMP)
    pooled_ok = not [b for b in bad if b.startswith("pooled")]
    res["1b"] = {**res["1b"], "paired": real, "pass": slope_ok and pooled_ok,
                 "insufficient": real["n_pairs"] < MIN_STARS,
                 "plant_paired_g": plant_all["pooled"]["gr_dx"],
                 "plant_fired": res["1b"]["plant_fired"] and abs(plant_all["pooled"]["gr_dx"][0]) > TOL_PAIRED}
    # 1c′: every camcol's paired median within TOL_PAIRED; the hashed tests on x, y, edge, run, s kept
    kept = {k: t for k, t in res["1c"]["tests"].items() if not k.endswith("~camcol")}
    cc_ok = not [b for b in bad if b.startswith("camcol")]
    _, bad_pl = _within(plant_cc)
    res["1c"] = {**res["1c"], "tests": kept, "paired_camcol": real["camcol"], "paired_fail": bad,
                 "pass": cc_ok and all(t["state"] != "DEPENDS" for t in kept.values()),
                 "plant_camcol": {"camcol": PLANT_CAMCOL, "flags": bad_pl},
                 "plant_fired": res["1c"]["plant_fired"]
                 and any(b.startswith(f"camcol{PLANT_CAMCOL}:gr_dx") for b in bad_pl)
                 and not any(b.startswith("camcol") and not b.startswith(f"camcol{PLANT_CAMCOL}") for b in bad_pl)}
    # 2′: nanmedian; the stamps a NaN drops, and which quantity
    paired = [m for m in rnd if "oldnew" in m]
    keys = ("var_ratio", "hf", "hf_v1", "lag1", "lag1_v1")
    drop = [{"id": m["id"], "corpus": m["corpus"], "group": m["group"],
             "nan_in": [k for k in keys if not np.all(np.isfinite(m[k]))],
             "bands": [b for i, b in enumerate("gri") if not np.isfinite(m["var_ratio"][i])]}
            for m in paired if not all(np.all(np.isfinite(m[k])) for k in keys)]
    vr = [float(np.nanmedian([m["var_ratio"][k] for m in paired])) for k in range(3)]
    hfr = [float(np.nanmedian([m["hf"][k] / m["hf_v1"][k] for m in paired])) for k in range(3)]
    dlag = [float(np.nanmedian([m["lag1"][k] - m["lag1_v1"][k] for m in paired])) for k in range(3)]
    two = res["2"]
    res["2"] = {**two, "var_ratio_median": vr, "hf_ratio_median": hfr, "lag1_delta_median": dlag, "dropped": drop,
                "pass": all(t["state"] != "DEPENDS" for t in two["tests"].values())
                and all(TOL_VAR[0] <= x <= TOL_VAR[1] for x in vr) and all(TOL_HF[0] <= x <= TOL_HF[1] for x in hfr)
                and all(abs(x) < TOL_LAG for x in dlag)}
    return res


def check5_matched(ms5: list[dict], smd: dict[str, float]) -> dict:
    """5′: KS on the centring residual, fresh random probe vs its matched pretrain; the hashed
    0.5 px probe plant through the same KS. Balance fails → INSUFFICIENT, never a pass."""
    from scipy.stats import ks_2samp
    a = np.array([np.hypot(*m["centre"]) for m in ms5 if m["group"] == "random"])
    b = np.array([np.hypot(*m["centre"]) for m in ms5 if m["group"] == "matched"])
    ap = np.array([np.hypot(*m["centre_plant"]) for m in ms5 if m["group"] == "random" and "centre_plant" in m])
    ks, ksp = ks_2samp(a, b), ks_2samp(ap, b)
    thin = min(a.size, b.size) < 200 or any(abs(v) >= SMD_MAX for v in smd.values())
    # unbalanced or thin → the KS is not read at all: INSUFFICIENT, neither a pass nor a DEFECT
    return {"ks_p": float(ks.pvalue), "n": [int(a.size), int(b.size)], "smd": smd,
            "median_abs_px": {"probe": float(np.median(a)), "pretrain_matched": float(np.median(b))},
            "insufficient": thin, "ks_read": not thin,
            "pass": thin or ks.pvalue > KS_P, "plant_ks_p": float(ksp.pvalue), "plant_fired": ksp.pvalue < KS_P}


def _measure_all(logs, tgt, root: Path, rng, pair: bool) -> list[dict]:
    ms = []
    for i, log in enumerate(logs):
        t = tgt[log["object_id"]]
        v1p = REPO / "data" / t["corpus"] / f"{log['object_id']}.fits"
        vdir = root / "variants"
        obj = {"v2": _fits(root / f"{log['object_id']}.fits"),
               "v1": _fits(v1p) if t["group"] != "v1_failed" and v1p.exists() else None,
               "variants": {v: _fits(vdir / v / f"{log['object_id']}.fits")
                            for v in ("nomargin", "bilinear", "margin64") if (vdir / v / f"{log['object_id']}.fits").exists()},
               "log": log, "corpus": t["corpus"], "group": t["group"]}
        m = measure(with_plants(obj, rng))
        if pair and obj["v1"] is not None and t["group"] in ("random", "aa3a", "edge"):
            m.update(paired_stars(obj["v2"], obj["v1"], m))
        ms.append(m)
        if (i + 1) % 100 == 0:
            print(f"  measured {i + 1}/{len(logs)}", flush=True)
    return ms


def analyse_amended(dry_run: bool = False) -> dict:
    """Amendment A on the pilot (checks 1–4, 6; corrected log) and on fresh data (5′).
    `dry_run`: before the hash — 5′'s code path on the pilot's own random groups (D28)."""
    tgt = {r["objID"]: r for r in csv.DictReader(PILOT.open(newline=""))}
    logs = list(csv.DictReader((PILOT_OUT / "cut_log.csv").open(newline="")))
    assert all(127 < float(lg["r_v1_relx"]) <= 128 for lg in logs), "cut_log not corrected (repull_fix_v1log.py)"
    ms = _measure_all(logs, tgt, PILOT_OUT, np.random.default_rng(20260926), pair=True)
    res = checks_amended(ms)
    per = {c: checks_amended([m for m in ms if m["corpus"] == c]) for c in ("probe", "pretrain")}
    if dry_run:
        ran = [m for m in ms if m["group"] == "random"]
        pc = _covariates("probe", {m["id"] for m in ran if m["corpus"] == "probe"})
        tc = _covariates("pretrain", {m["id"] for m in ran if m["corpus"] == "pretrain"})
        got, smd = _match(pc, tc, np.random.default_rng(1))
        sel = [{**m, "group": "matched"} for m in ran if m["id"] in set(got)] + [m for m in ran if m["corpus"] == "probe"]
        res["5"] = check5_matched(sel, smd)
        res["5"]["DRY_RUN"] = "pilot random groups matched within themselves — plumbing only, not a result"
    else:
        t5 = {r["objID"]: r for r in csv.DictReader(CHECK5.open(newline=""))}
        l5 = list(csv.DictReader((CHECK5_OUT / "cut_log.csv").open(newline="")))
        ms5 = _measure_all(l5, t5, CHECK5_OUT, np.random.default_rng(20260928), pair=False)
        smd = json.loads((REPO / "artifacts" / "out" / "repull_check5_sample.json").read_text())["smd"]
        res["5"] = check5_matched(ms5, smd)
        sep = {lg["object_id"]: lg.get("gz2_sep_flag", None) for lg in l5}
        flagged = {o for o, r in t5.items() if r["group"] == "sep_flag"}
        res["sep_flag"] = {"n_flagged_targets": len(flagged),
                           "flagged_cut": sum(o in sep for o in flagged),
                           "flag_set_on_all_flagged": all(sep.get(o) == "1-3arcsec" for o in flagged if o in sep),
                           "flag_set_elsewhere": sum(1 for o, f in sep.items() if f and o not in flagged),
                           "v1_log_in_range": all(127 < float(lg[f"{b}_v1_rel{a}"]) <= 128
                                                  for lg in l5 for b in "gri" for a in "xy")}
    out = {"amendment": "A (post hoc, D27)", "dry_run": dry_run, "verdict": verdict(res, per),
           "checks": res, "per_corpus": {c: {k: v.get("pass") for k, v in p.items()} for c, p in per.items()}}
    name = "repull_pilot_amendA_dry.json" if dry_run else "repull_pilot_amendA.json"
    (REPO / "artifacts" / "out" / name).write_text(json.dumps(out, indent=1, default=float))
    print(out["verdict"])
    return out


if __name__ == "__main__":
    {"sample": sample, "plant": plant, "analyse": analyse, "sample_check5": sample_check5,
     "analyse_amended": analyse_amended, "dry_run": lambda: analyse_amended(dry_run=True)}[sys.argv[1]]()
