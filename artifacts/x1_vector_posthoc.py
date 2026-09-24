"""X1 post hoc (EXPLORATORY): which image-plane vector are PC1 and PC2 the components of?

X1c found (PC1, PC2) transforming as a polar vector: rot90 carries PC1 → PC2 and PC2 → −PC1;
the left-right mirror flips PC1 and keeps PC2; the same in smooth galaxies as in spirals. W2
tested only magnitudes, and a vector's components average out against any magnitude. Here each
candidate is measured as a 2-vector (x = column rightwards, y = row downwards, about pixel
127.5) on the X1c sample, and PC1/PC2 are regressed on it (5-fold R²). Named after seeing X1c,
so nothing here is confirmatory.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import r_nonlinear as R  # noqa: E402
import w2_name_pcs as W  # noqa: E402
import x1_handedness as X  # noqa: E402

from galaxy_jepa.data.validity import invalid_planes  # noqa: E402
from galaxy_jepa.models.vit import load_frozen_encoder  # noqa: E402


def vectors(img: np.ndarray, petro_px: float) -> dict[str, tuple[float, float]]:
    edge, interior = invalid_planes(img)
    valid = ~(edge | interior)
    h, w = img.shape[1:]
    yy, xx = np.mgrid[:h, :w]
    dx, dy = xx - (w - 1) / 2, yy - (h - 1) / 2
    r = np.hypot(dx, dy) + 1e-9
    rp = float(np.clip(petro_px, 3.0, 100.0))
    outer = valid & (r > 2 * rp)
    if outer.sum() < 1000:
        outer = valid & (r > rp)
    v = img[1][outer].astype(np.float64)
    for _ in range(3):
        med, mad = np.median(v), 1.4826 * np.median(np.abs(v - np.median(v)))
        v = v[np.abs(v - med) < 3 * mad] if mad > 0 else v
    sky, noise = float(np.median(v)), float(1.4826 * np.median(np.abs(v - np.median(v))))
    band = img[1].astype(np.float64) - sky
    out = {}
    ap = np.clip(np.where(valid & (r <= rp), band, 0), 0, None)
    s = ap.sum()
    out["centroid"] = ((ap * dx).sum() / s, (ap * dy).sum() / s) if s > 0 else (np.nan, np.nan)
    inner = np.clip(np.where(valid & (r <= 2 * rp), band, 0), 0, None)
    s = inner.sum()
    out["lopsided"] = (((inner * dx / r).sum() / s, (inner * dy / r).sum() / s) if s > 0
                       else (np.nan, np.nan))
    det = valid & (band > 3 * max(noise, 1e-12))
    lab, n = ndimage.label(det, structure=np.ones((3, 3)))
    ids = np.arange(1, n + 1)
    near = ndimage.minimum(r, lab, index=ids) if n else np.array([])
    area = ndimage.sum(det, lab, index=ids) if n else np.array([])
    keep = ids[(area >= 10) & (near > rp)] if n else ids
    m = np.isin(lab, keep) & det
    wgt = np.where(m, band, 0)
    s = wgt.sum()
    out["neighbours"] = ((wgt * dx / r).sum() / s, (wgt * dy / r).sum() / s) if s > 0 else (0, 0)
    s = edge.sum()
    out["padding"] = ((dx[edge] / r[edge]).mean(), (dy[edge] / r[edge]).mean()) if s else (0, 0)
    sk = outer & (np.abs(band) < 3 * max(noise, 1e-12))
    a = np.c_[dx[sk], dy[sk], np.ones(sk.sum())]
    coef = np.linalg.lstsq(a, band[sk], rcond=None)[0]
    out["sky_gradient"] = (float(coef[0]), float(coef[1]))
    return out


def cv_r2(y: np.ndarray, x: np.ndarray) -> float:
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_predict

    ok = np.isfinite(y) & np.all(np.isfinite(x), 1)
    y, x = W.rank_normal(y[ok]), np.column_stack([W.rank_normal(c) for c in x[ok].T])
    pred = cross_val_predict(LinearRegression(), x, y, cv=KFold(5, shuffle=True, random_state=0))
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main() -> None:
    setup = R.prepare("runs/m/encoder.pt", R.MAX_TRAIN, label="X1-post", sources=1)
    ctx = R.Ctx(setup, load_frozen_encoder(setup.ckpt), 0, dry=True)
    b = X.basis(ctx, setup)[0]
    blob = np.load(X.BANK)
    ids = blob["ids"]
    z = ((blob["M:orig"].astype(np.float64) - b["mu"]) @ b["v"][:, :2]) / b["sd"][:2]
    idx = X.ds_index(setup, ids)
    names = ("centroid", "lopsided", "neighbours", "padding", "sky_gradient")
    vec = {k: np.full((ids.size, 2), np.nan) for k in names}
    for i, j in enumerate(idx):
        item = setup.ds[int(j)]
        img = item["image"].float().numpy()
        for k, v in vectors(img, item["petro_rad_arcsec"] / item["pixel_scale"]).items():
            vec[k][i] = v
    rec: dict = {"n": int(ids.size)}
    for k in names:
        rec[k] = {"PC1_on_xy": cv_r2(z[:, 0], vec[k]), "PC2_on_xy": cv_r2(z[:, 1], vec[k]),
                  "PC1~x": X._spearman(z[:, 0], vec[k][:, 0]),
                  "PC2~y": X._spearman(z[:, 1], vec[k][:, 1]),
                  "PC1~y": X._spearman(z[:, 0], vec[k][:, 1]),
                  "PC2~x": X._spearman(z[:, 1], vec[k][:, 0])}
        print(k, {q: round(v, 3) for q, v in rec[k].items()}, file=sys.stderr)
    allx = np.column_stack([vec[k] for k in names])
    rec["joint"] = {"PC1": cv_r2(z[:, 0], allx), "PC2": cv_r2(z[:, 1], allx)}
    print("joint", rec["joint"], file=sys.stderr)
    (R.OUT / "x1_vector_posthoc.json").write_text(json.dumps(rec, indent=1, default=float))


if __name__ == "__main__":
    main()
