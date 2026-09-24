"""Brief Z1 — do the two PAnDa pitch-angle references agree with each other?

Hart et al. 2017 (SpArcFiRe) and Yu & Ho 2020 (2DFFT), matched to each other within 3″. The
threshold (ρ = 0.5) and the states are fixed in `z_findings.md` §Z1 before this runs; the Y2 look
at the same 34 pairs is disclosed there. No embeddings, no encoder: light.

  --planted   D28 — plants at the observed n through the identical `z1_state`
  --z1        the test
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import r_nonlinear as R  # noqa: E402
import u2_uncertainty as U2  # noqa: E402
import v2_independent as V2  # noqa: E402
import y_pitch as Y  # noqa: E402

THRESHOLD = 0.5
MIN_PAIRS = 10
N_PLANT = 500
OUT = R.OUT / "z1_agreement.json"
PLANTED = R.OUT / "z1_planted.json"
META = Path(__file__).parents[1] / "data" / "probe" / "metadata.csv"


def fisher_ci(rho: float, n: int) -> list[float]:
    """Spearman CI on the Fisher scale with Bonett & Wright's (2000) variance."""
    if n <= 3:
        return [-1.0, 1.0]
    se = np.sqrt((1 + rho**2 / 2) / (n - 3))
    z = np.arctanh(np.clip(rho, -0.9999, 0.9999))
    return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]


def compare(a: np.ndarray, b: np.ndarray, vis: np.ndarray | None, seed: int) -> dict:
    out: dict = {"n": int(a.size)}
    if a.size < MIN_PAIRS:
        return out
    out["rho"] = V2.spearman(a, b)
    out["ci_fisher"] = fisher_ci(out["rho"], a.size)
    out["ci_boot"] = V2.boot_ci(a, b, seed)
    if vis is not None:
        out["partial_vis"] = U2.partial_perm(a, b, vis[:, None], seed=0, perm=False)[0]
        out["ci_fisher_partial"] = fisher_ci(out["partial_vis"], a.size - 1)
    out["median_abs_diff_deg"] = float(np.median(np.abs(a - b)))
    return out


def z1_state(c: dict) -> str:
    """First match applies (z_findings.md §Z1). A verdict needs BOTH intervals on the same side."""
    if c["n"] < MIN_PAIRS:
        return "INSUFFICIENT OVERLAP"
    hi = max(c["ci_fisher"][1], c["ci_boot"][1])
    lo = min(c["ci_fisher"][0], c["ci_boot"][0])
    if hi < THRESHOLD:
        return "DISAGREE"
    if lo >= THRESHOLD:
        return "AGREE"
    return "INSUFFICIENT OVERLAP"


def n_needed(rho: float, power: float = 0.8) -> int:
    """Smallest n at which a true ρ clears the threshold's side with the stated power (Fisher)."""
    if abs(rho - THRESHOLD) < 1e-9:
        return -1
    for n in range(MIN_PAIRS, 100_000):
        z, zt = np.arctanh(rho), np.arctanh(THRESHOLD)
        se = np.sqrt((1 + rho**2 / 2) / (n - 3))
        # P(lower bound ≥ t) for ρ above, P(upper bound < t) for ρ below; normal on the z scale
        from scipy.stats import norm

        pw = norm.sf((zt + 1.96 * se - z) / se) if rho > THRESHOLD else norm.cdf(
            (zt - 1.96 * se - z) / se)
        if pw >= power:
            return n
    return -1


def pairs() -> pd.DataFrame:
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    h, y = Y.panda("hart"), Y.panda("yuho")
    ch = SkyCoord(h.RA_deg.values * u.deg, h.Dec_deg.values * u.deg)
    cy = SkyCoord(y.RA_deg.values * u.deg, y.Dec_deg.values * u.deg)
    i, d2, _ = ch.match_to_catalog_sky(cy)
    ok = d2.arcsec < 3
    out = pd.DataFrame({"hart": h.PA_degrees.values[ok], "yuho": y.PA_degrees.values[i[ok]],
                        "ra": h.RA_deg.values[ok], "dec": h.Dec_deg.values[ok],
                        "sep_arcsec": d2.arcsec[ok]})
    return out[np.isfinite(out.hart) & np.isfinite(out.yuho)].reset_index(drop=True)


def with_visibility(p: pd.DataFrame) -> tuple[np.ndarray, int]:
    """V1 visibility for the pairs found in our corpus (3″ to our metadata)."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    t = pd.read_csv(META, usecols=["ra", "dec", *Y.VIS_COLS], low_memory=False)
    c = SkyCoord(p.ra.values * u.deg, p.dec.values * u.deg)
    i, d2, _ = c.match_to_catalog_sky(SkyCoord(t.ra.values * u.deg, t.dec.values * u.deg))
    ok = d2.arcsec < 3
    vis = np.full(len(p), np.nan)
    rows = t.iloc[i[ok]]
    fin = np.isfinite(rows[list(Y.VIS_COLS)].to_numpy(float)).all(1)
    idx = np.flatnonzero(ok)[fin]
    vis[idx] = Y.visibility(rows[fin])
    return vis, int(ok.sum())


def planted(n: int) -> dict:
    """True ρ 0.9 / 0.65 / 0 / −0.25 at the observed n, N_PLANT draws each, identical path."""
    rng = np.random.default_rng(5)
    out = {}
    for rho in (0.9, 0.65, 0.0, -0.25):
        states = []
        for k in range(N_PLANT):
            # Gaussian copula at the Pearson ρ that gives the target Spearman ρ
            r = 2 * np.sin(np.pi * rho / 6)
            x = rng.normal(size=n)
            y = r * x + np.sqrt(1 - r**2) * rng.normal(size=n)
            states.append(z1_state(compare(x, y, None, seed=k)))
        u, c = np.unique(states, return_counts=True)
        out[f"rho={rho}"] = {"n": n, **{s: int(k) for s, k in zip(u, c, strict=True)},
                             "n_needed_80pct": n_needed(rho)}
    return out


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--z1"
    p = pairs()
    if mode == "--planted":
        rec = planted(len(p))
        PLANTED.write_text(json.dumps(rec, indent=1))
        print(json.dumps(rec, indent=1))
        return
    vis, in_corpus = with_visibility(p)
    fin = np.isfinite(vis)
    rec = {"n_pairs": len(p), "in_corpus": in_corpus, "n_with_visibility": int(fin.sum()),
           "max_sep_arcsec": float(p.sep_arcsec.max()),
           "primary": compare(p.hart.to_numpy(), p.yuho.to_numpy(), None, seed=1),
           "sensitivity_vis": compare(p.hart.to_numpy()[fin], p.yuho.to_numpy()[fin], vis[fin],
                                      seed=2),
           "sd_deg": {"hart": float(p.hart.std()), "yuho": float(p.yuho.std())},
           "median_deg": {"hart": float(p.hart.median()), "yuho": float(p.yuho.median())}}
    rec["state"] = z1_state(rec["primary"])
    s = rec["sensitivity_vis"]
    if "partial_vis" in s:
        s["state_partial"] = z1_state({"n": s["n"], "ci_fisher": s["ci_fisher_partial"],
                                       "ci_boot": s["ci_fisher_partial"]})
    if rec["state"] == "INSUFFICIENT OVERLAP" and "rho" in rec["primary"]:
        rec["n_needed"] = {f"true_rho={r}": n_needed(r) for r in (0.65, 0.4, 0.3)}
    OUT.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
