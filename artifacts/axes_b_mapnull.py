"""R² of each GZ axis score from 18 directions: the physical set (map's own), 18 shuffled-target w's,
and 18 random directions drawn from the cleaned space's covariance N(0, Σ) via Cholesky factor S.

Usage:
    .venv/bin/python artifacts/axes_b_mapnull.py --out docs/axes/b_map_null_erratum.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "artifacts") not in sys.path:
    sys.path.insert(0, str(REPO / "artifacts"))


def sample_covariance_directions(
    cov: np.ndarray, k: int, rng: np.random.Generator, *, jitter: float = 1e-6
) -> np.ndarray:
    """Draw ``k`` random directions with covariance ``cov`` via Cholesky factor ``S @ z``."""
    c = np.asarray(cov, dtype=np.float64)
    d = c.shape[0]
    chol = np.linalg.cholesky(c + jitter * np.eye(d, dtype=np.float64))
    z = rng.normal(size=(d, k))
    return chol @ z  # (d, k) with E[w w^T] = cov + jitter * I


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Part B Gaussian covariance map-null R² check.")
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "docs" / "axes" / "b_map_null_erratum.json",
        help="Output JSON path (defaults to a versioned erratum file so historical output is kept).",
    )
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    import axes_a_ordinal as A

    AXS, d = A.load()
    B = Path("runs/almagest/axes_phys/b")
    m = json.loads((REPO / "docs/axes/b_map.json").read_text())
    names = m["names"]
    gz = names[: m["n_gz"]]
    strong = m["strong_physical"]
    X = d.clean["A1"]
    te = np.isin(d.union, np.asarray(d.setup.test_ids)) & d.ok
    rng = np.random.default_rng(args.seed)

    real = np.column_stack(
        [X @ np.load(B / "fits" / (n.replace("/", "_").replace(" ", "_") + ".npz"))["cleaned_we"] for n in strong]
    )
    nulls = np.column_stack(
        [
            X @ np.load(B / "fits" / (n.replace("/", "_").replace(" ", "_") + ".npz"))["null_we"][0]
            for n in strong
        ]
    )
    w_rand = sample_covariance_directions(np.cov(X[te].T), k=len(strong), rng=rng)
    rd = X @ w_rand

    out = {}
    for f in gz:
        a = np.load(AXS.AX / f"{f}.npz")
        s = X @ a["we"] + float(a["be"])
        e = np.zeros(len(X), bool)
        e[d.rows(f)[1]] = True
        o = np.flatnonzero(e & te)
        h = rng.permutation(len(o)) < len(o) // 2
        r = []
        for M0 in (real, nulls, rd):
            M = np.column_stack([np.ones(len(o)), M0[o]])
            M[:, 1:] = (M[:, 1:] - M[h, 1:].mean(0)) / M[h, 1:].std(0)
            c = np.linalg.solve(M[h].T @ M[h] + 1e-3 * np.eye(M.shape[1]), M[h].T @ s[o][h])
            y = s[o][~h]
            r.append(1 - ((y - M[~h] @ c) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        out[f] = r

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": __doc__,
        "r2": {
            f: {
                "physical": float(r[0]),
                "shuffled_target_directions": float(r[1]),
                "random_directions": float(r[2]),
            }
            for f, r in out.items()
        },
    }
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n")
    tmp.replace(args.out)


if __name__ == "__main__":
    main()
