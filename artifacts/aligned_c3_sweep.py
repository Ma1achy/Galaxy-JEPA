"""aligned_comparison.md v5, criterion 3: the DIRTIER and CLEANER plants as power curves (user, 2026-09-28).

v4's single plant (an i−r x component at 20% of variance) read SAME under F_N = 0.048. F stays; the
plant becomes a sweep, and the smallest strength that reads correctly in ≥ 80% of realisations is
declared as criterion 3's detectable effect.

  DIRTIER at p: "M" = M1, M2 as they are; "aligned" = each plus a fresh i−r x-correlated component
      carrying a share p of the new total variance (v4's `inject`, at p instead of 0.20).
  CLEANER at p: the base is M1, M2 with their offset-tracking top-10 PCs projected out (v4's `clean`),
      so CLEANER's second clause (no offset-tracking PC in either aligned seed) can hold. "aligned" =
      the base; "M" = the base plus a fresh component at p. The pairs differ only by the planted
      component, mirroring DIRTIER.

Both arms score through aligned_c13's own `representation` and `seed_range`, with the floors fixed at
real M's (`c3_floor` on M1, M2): F_N = 0.048, as the user kept it. Strengths are the user's: 10–40% in
steps of 5, 20 realisations each; a realisation draws a fresh direction for each seed.

    uv run python artifacts/aligned_c3_sweep.py      # -> artifacts/out/c3_sweep.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aligned_c13 import OUT, _load, _r_flux, c3_floor, families, representation, seed_range, setup  # noqa: E402

SEED, N_REAL, POWER = 20261001, 20, 0.80
STRENGTHS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)


def inject(x: np.ndarray, z: np.ndarray, p: float, rng) -> np.ndarray:
    """Add a z-correlated component on a random direction carrying share p of the new total variance."""
    v = rng.standard_normal(x.shape[1])
    v /= np.linalg.norm(v)
    tot = np.var(x - x.mean(0), axis=0).sum()
    return x + np.sqrt(p / (1 - p) * tot) * np.outer(z, v)


def clean(x: np.ndarray, r: dict) -> np.ndarray:
    xc = x - x.mean(0)
    _, vec = np.linalg.eigh(np.cov(xc, rowvar=False))
    ks = [p["pc"] - 1 for p in r["pcs"] if p["tracks"] == "offsets"]
    v = vec[:, ::-1][:, ks]
    return x - xc @ v @ v.T


def states(ra, rm, floors) -> dict:
    def rule(q, hi, same, lo):
        return seed_range((ra[0][q], ra[1][q]), (rm[0][q], rm[1][q]), hi, same, lo, floors[q])
    n = rule("N", "DIRTIER", "SAME", "LOWER")
    nuis = "CLEANER" if n == "LOWER" and not (ra[0]["offset_pc"] or ra[1]["offset_pc"]) else ("SAME" if n == "LOWER" else n)
    return {"nuisance": nuis, "morphology_share": rule("Mo", "GAINED", "SAME", "LOST"),
            "dimensionality": rule("PR", "HIGHER", "SAME", "LOWER")}


def brief(r: dict) -> dict:
    return {"N": r["N"], "Mo": r["Mo"], "PR": r["PR"], "offset_pc": r["offset_pc"]}


def main() -> None:
    t0 = time.time()
    d = setup()
    d["r_flux"] = _r_flux(d)
    fam = families(d)
    m = (_load("m1", d["test"]), _load("m2", d["test"]))
    rm = tuple(representation(x, fam) for x in m)
    floors = {q: c3_floor((rm[0][q], rm[1][q])) for q in ("N", "Mo", "PR")}
    off = d["pan_test"]["i-r x"]
    off = np.where(np.isnan(off), np.nanmean(off), off)
    z = (off - off.mean()) / off.std()
    base = tuple(clean(x, r) for x, r in zip(m, rm))
    rb = tuple(representation(x, fam) for x in base)
    out = {"seed": SEED, "realisations": N_REAL, "power_bar": POWER, "floors": floors,
           "M": [brief(r) for r in rm], "clean_base": [brief(r) for r in rb], "dirtier": {}, "cleaner": {}}
    print(f"setup {time.time() - t0:.0f}s | F_N {floors['N']:.4f} | M N {rm[0]['N']:.3f} {rm[1]['N']:.3f} "
          f"| base N {rb[0]['N']:.3f} {rb[1]['N']:.3f} offset_pc {rb[0]['offset_pc']} {rb[1]['offset_pc']}", flush=True)
    rng = np.random.default_rng(SEED)
    for p in STRENGTHS:
        for arm in ("dirtier", "cleaner"):
            reals = []
            for _ in range(N_REAL):
                if arm == "dirtier":
                    ra, rr = tuple(representation(inject(x, z, p, rng), fam) for x in m), rm
                else:
                    ra, rr = rb, tuple(representation(inject(x, z, p, rng), fam) for x in base)
                reals.append({"states": states(ra, rr, floors), "aligned": [brief(r) for r in ra],
                              "reference": [brief(r) for r in rr]})
            want = arm.upper()
            hit = sum(r["states"]["nuisance"] == want for r in reals)
            out[arm][f"{p:.2f}"] = {"rate": hit / N_REAL, "hits": hit, "realisations": reals}
            print(f"  {arm:8s} p={p:.2f}  {want} {hit}/{N_REAL}  ({time.time() - t0:.0f}s)", flush=True)
            (OUT / "c3_sweep.json").write_text(json.dumps(out, indent=1, default=float))
    for arm in ("dirtier", "cleaner"):
        ok = [p for p in STRENGTHS if out[arm][f"{p:.2f}"]["rate"] >= POWER]
        out[f"{arm}_detectable"] = min(ok) if ok else None
    (OUT / "c3_sweep.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({a: {p: out[a][p]["rate"] for p in out[a]} for a in ("dirtier", "cleaner")}),
          "| detectable:", out["dirtier_detectable"], out["cleaner_detectable"])


if __name__ == "__main__":
    main()
