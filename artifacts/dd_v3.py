"""Brief DD, V3 ground truth: Galaxy Zoo 3D volunteer masks reprojected into M's stamps.

TOOL VALIDATION ONLY (artifacts/interp_tooling.md).

GZ3D (DR17 VAC, galaxyzoo3d/v4_0_0) gives per-pixel volunteer counts for bar and spiral-arm masks on
its own 525² image and TAN WCS. M's v1 stamps are a 256² window of the SDSS r frame starting at the
virtual origin ceil(x − 128), x = the frame pixel of the GZ2 coordinate (sciserver_cut.py: Cutout2D on
the GZ2 ra/dec, mode='all'). So stamp pixel (row, col) is frame pixel (y0 + row, x0 + col), and the
chain stamp → frame → sky → GZ3D is exact up to v1's per-band snapping (≤ 0.5 px ≪ a 16-px patch).

  uv run python artifacts/dd_v3.py fetch      # test ∩ GZ3D files + r-frame headers (network, resumable)
  uv run python artifacts/dd_v3.py overlay    # 20-galaxy overlay figure — eyeball before selecting
  uv run python artifacts/dd_v3.py select     # the V3 list (≤ 500 bar, ≤ 500 spiral), hashed
"""

from __future__ import annotations

import bz2
import csv
import gzip
import hashlib
import io
import json
import math
import sys
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO / "runs" / "dd"
GZ3D_DIR = LOCAL / "gz3d"
HDR_DIR = LOCAL / "frame_hdr"
FIG = REPO / "artifacts" / "out" / "dd"
GZ3D_URL = "https://data.sdss.org/sas/dr17/manga/morphology/galaxyzoo3d/v4_0_0/"
FRAME_URL = ("https://data.sdss.org/sas/dr17/eboss/photoObj/frames/{rerun}/{run}/{camcol}/"
             "frame-r-{run:06d}-{camcol}-{field:04d}.fits.bz2")
HDR_BYTES = 700_000  # the bz2 stream's first block(s) hold the primary header
STAMP, PATCH, GRID = 256, 16, 16
SUPER = 4  # sub-samples per stamp pixel per axis in the reprojection
VOTES = (3, 2, 5)  # primary, then sensitivities
PATCH_FRAC = 0.5
PATCH_FRAC_SENS = 0.25  # declared, reported sensitivity: bars are narrower than a 6.3″ patch
N_MAX, SEED = 500, 20260925
HDU = {"spiral": 3, "bar": 4}


# ── fetch ────────────────────────────────────────────────────────────────────────────────────────

def _targets() -> list[dict]:
    """Test-split galaxies with a GZ3D file, with the metadata the v1 cut used."""
    import r_nonlinear as R
    from j4_spread_controls import prepare
    setup = prepare(None, R.MAX_TRAIN, label="DDV3", sources=1)
    test = {int(i) for i in setup.test_ids}
    by_mid = {}
    for name in (LOCAL / "gz3d_files.txt").read_text().split():
        by_mid.setdefault(name.split("_")[1], name)
    pos = {r["mangaid"]: r for r in csv.DictReader(open(LOCAL / "gz3d_positions.csv"))}
    want = {int(r["probe_id"]): mid for mid, r in pos.items() if r["probe_id"] and int(r["probe_id"]) in test}
    out = []
    with open(REPO / "data" / "probe" / "metadata.csv") as fh:
        for r in csv.DictReader(fh):
            oid = int(r["object_id"])
            if oid in want and want[oid] in by_mid:
                out.append({"id": oid, "mangaid": want[oid], "file": by_mid[want[oid]],
                            "ra": r["ra"], "dec": r["dec"], "run": int(r["run"]),
                            "camcol": int(r["camcol"]), "field": int(r["field"]), "rerun": int(r["rerun"])})
    return sorted(out, key=lambda t: t["id"])


def _get(url: str, rng: int | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{rng - 1}"} if rng else {})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:  # noqa: BLE001 — retried, then loud
            if attempt == 3:
                raise
    raise AssertionError


def _fetch_one(t: dict) -> str:
    g = GZ3D_DIR / t["file"]
    if not g.exists():
        g.write_bytes(_get(GZ3D_URL + t["file"]))
    key = f"{t['run']}-{t['camcol']}-{t['field']}"
    h = HDR_DIR / f"{key}.hdr"
    if not h.exists():
        url = FRAME_URL.format(**t)
        raw = bz2.BZ2Decompressor().decompress(_get(url, HDR_BYTES))
        end = raw.find(b"END" + b" " * 77)
        if end < 0 or end % 80:
            raise RuntimeError(f"no END card in the first {HDR_BYTES} B of {url}")
        h.write_bytes(raw[:end + 80])
    return t["file"]


def fetch() -> None:
    GZ3D_DIR.mkdir(parents=True, exist_ok=True)
    HDR_DIR.mkdir(parents=True, exist_ok=True)
    ts = _targets()
    (LOCAL / "v3_targets.json").write_text(json.dumps(ts))
    print(f"test ∩ GZ3D: {len(ts):,} galaxies, {len({(t['run'], t['camcol'], t['field']) for t in ts}):,} frames",
          flush=True)
    bad = []
    with ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(_fetch_one, t): t for t in ts}
        for i, f in enumerate(futs, 1):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001 — counted and reported, never silent
                bad.append((futs[f]["id"], repr(e)))
            if i % 250 == 0:
                print(f"  {i:,}/{len(ts):,}  failed {len(bad)}", flush=True)
    (LOCAL / "v3_fetch_failed.json").write_text(json.dumps(bad))
    print(f"fetched; {len(bad)} failed")


# ── reprojection ─────────────────────────────────────────────────────────────────────────────────

def _frame_wcs(t: dict):
    from astropy.io import fits
    from astropy.wcs import WCS
    hdr = fits.Header.fromstring((HDR_DIR / f"{t['run']}-{t['camcol']}-{t['field']}.hdr").read_bytes().decode("ascii"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WCS(hdr)


def stamp_origin(t: dict, wcs) -> tuple[int, int, float, float]:
    """v1's virtual origin (x0, y0) and the float frame position (x, y) of the GZ2 coordinate."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.wcs.utils import skycoord_to_pixel
    x, y = (float(v) for v in skycoord_to_pixel(SkyCoord(float(t["ra"]) * u.deg, float(t["dec"]) * u.deg),
                                                 wcs, mode="all"))
    return math.ceil(x - STAMP / 2), math.ceil(y - STAMP / 2), x, y


def load_gz3d(t: dict):
    from astropy.io import fits
    from astropy.wcs import WCS
    with fits.open(io.BytesIO(gzip.decompress((GZ3D_DIR / t["file"]).read_bytes()))) as h:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = WCS(h[0].header, naxis=2)
        rgb = np.asarray(h[0].data)
        masks = {k: np.asarray(h[i].data, np.float32) for k, i in HDU.items()}
        meta = h[5].data
        votes = {c: int(meta[c][0]) for c in meta.columns.names if c.startswith("GZ_")}
    return w, rgb, masks, votes


def reproject(t: dict) -> dict:
    """Per-pixel volunteer counts on the stamp grid (256², via SUPER² sub-samples, nearest GZ3D
    pixel, then the sub-sample mean of the thresholded mask per stamp pixel) and the footprint:
    stamp pixels whose sub-samples all land inside the GZ3D image."""
    fw = _frame_wcs(t)
    x0, y0, x, y = stamp_origin(t, fw)
    gw, rgb, masks, votes = load_gz3d(t)
    s = (np.arange(STAMP * SUPER) + 0.5) / SUPER - 0.5  # sub-sample centres in stamp pixel coords
    cc, rr = np.meshgrid(s, s)
    sky = fw.pixel_to_world(x0 + cc, y0 + rr)
    gx, gy = gw.world_to_pixel(sky)
    ix, iy = np.rint(gx).astype(int), np.rint(gy).astype(int)
    n = masks["bar"].shape
    inside = (ix >= 0) & (iy >= 0) & (ix < n[1]) & (iy < n[0])
    ixc, iyc = np.clip(ix, 0, n[1] - 1), np.clip(iy, 0, n[0] - 1)
    out = {"x0": x0, "y0": y0, "x": x, "y": y, "votes": votes, "rgb_shape": rgb.shape,
           "footprint": _pool(inside.astype(np.float32)) == 1.0}
    for k, m in masks.items():
        v = np.where(inside, m[iyc, ixc], 0.0)
        out[k] = {th: _pool((v >= th).astype(np.float32)) for th in VOTES}
        out[f"{k}_max"] = float(m.max())
    out["gz_xy"] = (gx, gy)
    return out


def _pool(a: np.ndarray, f: int = SUPER) -> np.ndarray:
    return a.reshape(a.shape[0] // f, f, a.shape[1] // f, f).mean(axis=(1, 3))


def patch_labels(frac_pix: np.ndarray, footprint: np.ndarray,
                 min_frac: float = PATCH_FRAC) -> tuple[np.ndarray, np.ndarray]:
    """(label, domain) on the 16×16 patch grid: a patch is positive if ≥ 50% of its pixels are inside
    the mask; the domain is the patches wholly inside the GZ3D footprint."""
    frac = frac_pix.reshape(GRID, PATCH, GRID, PATCH).mean(axis=(1, 3))
    dom = footprint.reshape(GRID, PATCH, GRID, PATCH).all(axis=(1, 3))
    return frac >= min_frac, dom


# ── overlay figure ───────────────────────────────────────────────────────────────────────────────

def overlay(n: int = 20) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dd_step0 import _rgb

    ts = json.loads((LOCAL / "v3_targets.json").read_text())
    ids = np.load(LOCAL / "v3_overlay_ids.npy") if (LOCAL / "v3_overlay_ids.npy").exists() else None
    rng = np.random.default_rng(SEED)
    if ids is None:
        # masks with bars and arms both represented; drawn from the galaxies with any ≥3 mask pixel
        cand = []
        for t in rng.permutation(ts):
            gw, rgb, masks, votes = load_gz3d(t)
            if masks["bar"].max() >= VOTES[0] or masks["spiral"].max() >= VOTES[0]:
                cand.append(int(t["id"]))
            if len(cand) == n:
                break
        ids = np.asarray(cand)
        np.save(LOCAL / "v3_overlay_ids.npy", ids)
    by = {t["id"]: t for t in ts}
    st = _stamps_for([int(i) for i in ids])
    fig, axes = plt.subplots(4, 10, figsize=(25, 12.5))
    for k, oid in enumerate(ids):
        t = by[int(oid)]
        rp = reproject(t)
        gw, rgb, masks, votes = load_gz3d(t)
        a1, a2 = axes[2 * (k // 10), k % 10], axes[2 * (k // 10) + 1, k % 10]
        a1.imshow(rgb, origin="upper")  # GZ3D's own RGB, as volunteers saw it
        for key, col in (("bar", "cyan"), ("spiral", "magenta")):
            if masks[key].max() >= VOTES[0]:
                a1.contour(masks[key] >= VOTES[0], levels=[0.5], colors=col, linewidths=0.8)
        a1.set_title(f"{oid}: GZ3D image + masks", fontsize=7)
        a2.imshow(_rgb(st[k]), origin="lower")
        for key, col in (("bar", "cyan"), ("spiral", "magenta")):
            if rp[f"{key}_max"] >= VOTES[0]:
                a2.contour(rp[key][VOTES[0]], levels=[0.5], colors=col, linewidths=0.8)
        a2.contour(rp["footprint"], levels=[0.5], colors="w", linewidths=0.5, linestyles=":")
        a2.set_title("M's stamp + reprojected", fontsize=7)
        for a in (a1, a2):
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle("DD V3 — GZ3D volunteer masks (≥3 votes; cyan bar, magenta arms) reprojected through GZ3D WCS → "
                 "sky → SDSS r-frame WCS → v1 stamp origin. White dotted: GZ3D footprint. Features should sit "
                 "on the same structure in both rows.", fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, hspace=0.25, wspace=0.05)
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / "dd_v3_overlays.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _stamps_for(ids: list[int]) -> np.ndarray:
    """v1 stamps from M's cache (normalised fp16), in the order given."""
    from f0_preconditions import check
    _, cache = check(verbose=False)
    return np.stack([np.asarray(cache.data[cache._row_of[i]], np.float32) for i in ids])


# ── selection ────────────────────────────────────────────────────────────────────────────────────

def select() -> dict:
    """V3's galaxies. Per structure and patch-coverage rule (primary ≥ 50%; declared sensitivity
    ≥ 25%), those whose ≥3-vote mask gives at least one positive and one negative patch inside the
    footprint; up to 500 per list at random (seed); every list hashed."""
    ts = json.loads((LOCAL / "v3_targets.json").read_text())
    bad = {int(i) for i, _ in json.loads((LOCAL / "v3_fetch_failed.json").read_text())}
    keys = {"bar": "bar_3", "spiral": "spiral_3", "bar_f25": "bar_3_f25", "spiral_f25": "spiral_3_f25"}
    elig: dict[str, list[int]] = {k: [] for k in keys}
    labels: dict[str, dict] = {}
    for t in ts:
        if t["id"] in bad:
            continue
        rp = reproject(t)
        dom = rp["footprint"].reshape(GRID, PATCH, GRID, PATCH).all(axis=(1, 3))
        rec = {"footprint": dom}
        for k in ("bar", "spiral"):
            for th in VOTES:
                rec[f"{k}_{th}"] = patch_labels(rp[k][th], rp["footprint"])[0]
            rec[f"{k}_{VOTES[0]}_f25"] = patch_labels(rp[k][VOTES[0]], rp["footprint"], PATCH_FRAC_SENS)[0]
        for name, key in keys.items():
            if rec[key][dom].any() and (~rec[key][dom]).any():
                elig[name].append(t["id"])
        labels[str(t["id"])] = rec
    rng = np.random.default_rng(SEED)
    chosen = {k: sorted(int(i) for i in (rng.choice(v, N_MAX, replace=False) if len(v) > N_MAX else v))
              for k, v in elig.items()}
    union = set().union(*chosen.values())
    np.savez_compressed(LOCAL / "v3_patch_labels.npz",
                        **{f"{oid}__{k}": v for oid, rec in labels.items() for k, v in rec.items()
                           if int(oid) in union})
    rec = {k: {"eligible": len(elig[k]), "n": len(v),
               "sha256": hashlib.sha256("\n".join(map(str, v)).encode()).hexdigest()} for k, v in chosen.items()}
    (LOCAL / "v3_list.json").write_text(json.dumps({"lists": chosen, "record": rec, "n_union": len(union)}))
    print(json.dumps(rec, indent=1), "union", len(union))
    return rec


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fetch":
        fetch()
    elif cmd == "overlay":
        print(overlay())
    elif cmd == "select":
        select()
