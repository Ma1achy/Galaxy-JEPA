"""Brief T2 — `t08 odd: other`, the new conditional R1: look before claiming it.

Its direction is the ladder's own canonical probe (conditional population, M's 4-epoch checkpoint,
fitted on train), scored on the conditional test galaxies. Rendered, then checked against the
known artefact axes:

  * padded frame (E1)       — invalid fraction of the stamp (`data.validity.validity_mask`)
  * petrorad_suspect        — the deblending / over-stamp flag (D6)
  * star_or_artifact votes  — t01 a03's vote fraction
  * bright neighbour        — brightest background-subtracted pixel beyond 2 R_petro over the
                              brightest within R_petro (declared here; > 0.5 flagged)

and the >100% retention under magnitude matching is taken apart into its four numbers.

    uv run python artifacts/t2_other.py  -> artifacts/out/t2_other.json, t2_other_{top,bottom,random}.png
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "artifacts")
import r_nonlinear as R  # noqa: E402

from galaxy_jepa.data.validity import validity_mask  # noqa: E402
from galaxy_jepa.models.vit import load_frozen_encoder  # noqa: E402
from galaxy_jepa.probing.extract import feature_embeddings, feature_ids  # noqa: E402
from galaxy_jepa.probing.logistic import probe_direction  # noqa: E402

FEATURE = "t08_odd_feature_a23_other"
STAR = "t01_smooth_or_features_a03_star_or_artifact"
N_SHOW = 30
NEIGHBOUR_FLAG = 0.5
OUT = R.OUT / "t2_other.json"


def auc(score: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(score)
    s, t = score[ok], y[ok].astype(bool)
    if t.all() or not t.any():
        return float("nan")
    ranks = np.argsort(np.argsort(s)) + 1.0
    n1 = t.sum()
    return float((ranks[t].sum() - n1 * (n1 + 1) / 2) / (n1 * (~t).sum()))


def _within(score: np.ndarray, y: np.ndarray, v: np.ndarray) -> float:
    """Mean AUC within quintiles of ``v`` — the direction with that axis held roughly fixed."""
    ok = np.isfinite(v)
    edges = np.quantile(v[ok], np.linspace(0, 1, 6))
    bins = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, 4)
    vals = [auc(score[ok & (bins == b)], y[ok & (bins == b)]) for b in range(5)]
    return float(np.nanmean(vals))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    ra = np.argsort(np.argsort(a[ok]))
    rb = np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1])


def stamp_axes(img: np.ndarray, petro_px: float) -> tuple[float, float]:
    """(invalid fraction, bright-neighbour ratio) for one (C, H, W) stamp."""
    valid = validity_mask(img)
    padded = float(1.0 - valid.mean())
    flat = np.where(valid[None], img, np.nan)
    sub = flat - np.nanmedian(flat.reshape(img.shape[0], -1), axis=1)[:, None, None]
    lum = np.nansum(sub, axis=0)
    h, w = lum.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2)
    rp = float(np.clip(petro_px, 3.0, 100.0))
    inner = np.nanmax(np.where(r <= rp, lum, np.nan))
    outer = np.where((r > 2 * rp) & valid, lum, np.nan)
    ratio = float(np.nanmax(outer) / inner) if inner > 0 and np.isfinite(outer).any() else np.nan
    return padded, ratio


def to_rgb(img: np.ndarray) -> np.ndarray:
    """(g, r, i) -> RGB (i, r, g), background-subtracted, one asinh stretch per stamp."""
    x = img[::-1].astype(np.float64)
    x = x - np.median(x.reshape(3, -1), axis=1)[:, None, None]
    peak = np.percentile(x, 99.7)
    soft = max(peak / 30.0, 1e-6)
    x = np.arcsinh(np.clip(x, 0, None) / soft) / np.arcsinh(max(peak, soft) / soft)
    return np.clip(np.moveaxis(x, 0, -1), 0, 1)


def grid(path, rows: list[dict], images: dict[int, np.ndarray], title: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 6, figsize=(13, 11.6))
    for ax, r in zip(axes.flat, rows, strict=False):
        ax.imshow(to_rgb(images[r["id"]]), origin="lower", interpolation="nearest")
        flags = "".join(k for k, on in (("P", r["padded"] > 0.01), ("S", r["suspect"]),
                                        ("*", (r["star_frac"] or 0) >= 0.2),
                                        ("N", (r["neighbour"] or 0) > NEIGHBOUR_FLAG)) if on)
        ax.set_title(f"s={r['score']:+.1f} f={r['frac']:.2f} {flags}", fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in list(axes.flat)[len(rows):]:
        ax.axis("off")
    fig.suptitle(title + "\ns = score along the direction, f = 'other' vote fraction; "
                 "P padded, S petrorad_suspect, * star/artifact >= 0.2, N bright neighbour",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    setup = R.prepare("runs/m/encoder.pt", R.MAX_TRAIN, label="T2", sources=1)
    frozen = load_frozen_encoder(setup.ckpt)
    real, untrained = R.load_matrices(setup, frozen)
    lab = setup.labels.with_population("conditional")
    pc = setup.pc

    tr = feature_embeddings(real, lab, FEATURE, setup.train_ids)
    te = feature_embeddings(real, lab, FEATURE, setup.test_ids)
    ids = np.asarray(feature_ids(real, lab, FEATURE, setup.test_ids))
    d = probe_direction(tr, name=FEATURE, c=pc.c)
    score = te.x @ d.w_raw + d.bias
    y = te.y.astype(int)
    print(f"T2 {FEATURE}: {y.size} conditional test galaxies, {y.sum()} 'other'; "
          f"AUC of the direction {auc(score, y):.4f}", file=sys.stderr)

    pos = {int(o): i for i, o in enumerate(np.asarray(setup.union))}
    rows_meta = setup.rows
    padded = np.empty(ids.size)
    neighbour = np.empty(ids.size)
    images: dict[int, np.ndarray] = {}
    for k, o in enumerate(ids):
        item = setup.ds[pos[int(o)]]
        if int(item["object_id"]) != int(o):
            raise SystemExit("T2: dataset order is not the union order")
        img = item["image"].float().numpy()
        petro_px = float(item["petro_rad_arcsec"]) / float(item["pixel_scale"])
        padded[k], neighbour[k] = stamp_axes(img, petro_px)
        images[int(o)] = img
    col = lambda c: np.array([float(rows_meta[int(o)].get(c, np.nan)) for o in ids])  # noqa: E731
    suspect = col("petrorad_suspect") > 0.5
    star = np.asarray(lab.vote_fraction(STAR, ids))
    frac = np.asarray(lab.vote_fraction(FEATURE, ids))
    nuis = {n: np.asarray(lab.nuisance_value(n, ids)) for n in lab.nuisances}
    z = col("z") if any("z" in rows_meta[int(o)] for o in ids[:50]) else None

    axes = {"padded_fraction": padded, "petrorad_suspect": suspect.astype(float),
            "star_or_artifact_frac": star, "bright_neighbour_ratio": neighbour, **nuis}
    if z is not None:
        axes["redshift"] = z
    table = {}
    for name, v in axes.items():
        table[name] = {"spearman_with_score": spearman(v, score),
                       "auc_for_other": auc(v, y),
                       "mean_other": float(np.nanmean(v[y == 1])),
                       "mean_not": float(np.nanmean(v[y == 0]))}
    flagged3 = (padded > 0.01) | suspect | (star >= 0.2)
    flagged = flagged3 | (neighbour > NEIGHBOUR_FLAG)
    robust = {
        "all": auc(score, y),
        "no padded": auc(score[padded <= 0.01], y[padded <= 0.01]),
        "no suspect": auc(score[~suspect], y[~suspect]),
        "no star-voted": auc(score[star < 0.2], y[star < 0.2]),
        "no bright neighbour": auc(score[~(neighbour > NEIGHBOUR_FLAG)],
                                   y[~(neighbour > NEIGHBOUR_FLAG)]),
        "none of the four": auc(score[~flagged], y[~flagged]),
        # the neighbour ratio as defined flags ~88% of stamps (outer max > half the core max is
        # normal on a normalised, crowded 256-px stamp), so it separates nothing; the three
        # defined flags are also reported without it
        "none of padded/suspect/star": auc(score[~flagged3], y[~flagged3]),
        "n_flagged_3": int(flagged3.sum()),
        **{f"within {n} quintiles": _within(score, y, v) for n, v in
           (("redshift", axes.get("redshift")), ("size", nuis.get("size")),
            ("magnitude", nuis.get("magnitude"))) if v is not None},
        "n_flagged": int(flagged.sum()),
        "share_flagged_other": float(flagged[y == 1].mean()),
        "share_flagged_not": float(flagged[y == 0].mean()),
    }

    # the >100% retention, taken apart: which of the four numbers moved, and why the bar fell
    s2 = json.loads((R.OUT / "s2_ladder.json").read_text())
    rec = next(r for r in s2["conditional"] if r["feature"] == FEATURE)
    u_scores = []
    for u in untrained:
        utr = feature_embeddings(u, lab, FEATURE, setup.train_ids)
        ute = feature_embeddings(u, lab, FEATURE, setup.test_ids)
        ud = probe_direction(utr, name=FEATURE, c=pc.c)
        u_scores.append(ute.x @ ud.w_raw + ud.bias)
    mag = nuis.get("magnitude")
    retention = {
        "A": rec["auc"], "C": rec["C"], "M": rec["matched_auc"], "C_m": rec["C_m"],
        "real_drop": rec["auc"] - rec["matched_auc"], "bar_drop": rec["C"] - rec["C_m"],
        "margin_unmatched": rec["auc"] - rec["C"], "margin_matched": rec["matched_auc"] - rec["C_m"],
        "magnitude_auc_for_other": auc(mag, y) if mag is not None else None,
        "spearman_magnitude_real_score": spearman(mag, score) if mag is not None else None,
        "spearman_magnitude_untrained_score": [spearman(mag, s) for s in u_scores]
        if mag is not None else None,
        "untrained_auc_check": [auc(s, y) for s in u_scores],
    }

    order = np.argsort(score)
    rng = np.random.default_rng(pc.seed)
    groups = {
        "top": order[::-1][:N_SHOW],
        "bottom": order[:N_SHOW],
        "random": rng.choice(np.flatnonzero(y == 1), N_SHOW, replace=False),
    }
    shown = {}
    for g, idx in groups.items():
        rows = [{"id": int(ids[i]), "score": float(score[i]), "frac": float(frac[i]),
                 "label": int(y[i]), "padded": float(padded[i]), "suspect": bool(suspect[i]),
                 "star_frac": float(star[i]), "neighbour": float(neighbour[i]),
                 **{n: float(v[i]) for n, v in nuis.items()}} for i in idx]
        shown[g] = rows
        title = {"top": f"Top {N_SHOW} along the 'other' direction (conditional test)",
                 "bottom": f"Bottom {N_SHOW} along the 'other' direction (conditional test)",
                 "random": f"{N_SHOW} random 'other'-labelled galaxies (conditional test)"}[g]
        grid(R.OUT / f"t2_other_{g}.png", rows, images, title)
        summary = {k: float(np.nanmean([r[k] for r in rows])) for k in
                   ("frac", "label", "padded", "star_frac", "neighbour")}
        summary["suspect"] = float(np.mean([r["suspect"] for r in rows]))
        print(f"T2 {g:<7s} " + "  ".join(f"{k} {v:.3f}" for k, v in summary.items()),
              file=sys.stderr)

    OUT.write_text(json.dumps({"feature": FEATURE, "n_test": int(y.size),
                               "positives": int(y.sum()), "axes": table, "robustness": robust,
                               "retention": retention, "shown": shown}, indent=1, default=float))
    print("\nT2 artefact axes (conditional test):", file=sys.stderr)
    for name, t in table.items():
        print(f"  {name:<24s} rho(score) {t['spearman_with_score']:+.3f}  AUC(other) "
              f"{t['auc_for_other']:.3f}  mean other {t['mean_other']:.3f} vs {t['mean_not']:.3f}",
              file=sys.stderr)
    print("T2 robustness: " + json.dumps(robust, default=lambda v: round(v, 4)), file=sys.stderr)
    print("T2 retention: " + json.dumps(retention, default=float), file=sys.stderr)


if __name__ == "__main__":
    main()
