"""The normalised sky σ in M's v1 cache, per band (user, 2026-10-13; baselines.md §3.4 item 4).

MoCo's additive noise is σ_aug = u × σ_sky (per band), u ~ U(0, 1), so σ_sky must be measured on the
exact tensors training reads: M's fp16 cache (asinh stretch q = 4, then the frozen Normalise
`75100066…`), through F0, so the cache is checked against the live freeze first.

Method, fixed before running:
- 2,000 stamps drawn uniformly (numpy seed 20261013) from M's pretrain **train** split.
- Sky pixels: each stamp's outer 16-px frame (256-px stamps, galaxy centred; the frame is
  ≥ 112 px = 44″ from the centre).
- Per stamp and band: robust σ = 1.4826 × MAD after iterative 3σ clipping about the median (to
  convergence, ≤ 10 rounds), so neighbouring sources and the galaxy's own wings do not inflate it.
  Also the frame's median (the sky level in normalised units).
- Reported per band: the median over stamps, the 16th and 84th percentiles, and the fraction of the
  frame clipped. The asinh stretch is near-linear at sky level, so a single σ per band is meaningful.

    uv run python artifacts/sky_sigma.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f0_preconditions import check  # noqa: E402
from i2_sigreg_run import _split_ids  # noqa: E402
from j4_spread_controls import OUT  # noqa: E402

N, SEED, FRAME, BANDS = 2_000, 20261013, 16, ("g", "r", "i")


def clipped_sigma(x: np.ndarray) -> tuple[float, float, float]:
    keep = np.ones(x.shape, bool)
    for _ in range(10):
        med = np.median(x[keep])
        s = 1.4826 * np.median(np.abs(x[keep] - med))
        new = np.abs(x - med) <= 3 * s
        if (new == keep).all():
            break
        keep = new
    return float(s), float(med), float(1 - keep.mean())


def main() -> None:
    cfg, cache = check(verbose=False)
    train, _, _ = _split_ids(cfg)
    ids = np.random.default_rng(SEED).choice(np.array(train), N, replace=False)
    rows = sorted(cache.row_of(int(i)) for i in ids)  # row order: sequential reads
    h = cache.index.height
    frame = np.ones((h, h), bool)
    frame[FRAME:-FRAME, FRAME:-FRAME] = False
    sig, med, clip = (np.empty((N, 3)) for _ in range(3))
    for k, r in enumerate(rows):
        x = np.asarray(cache.data[r], dtype=np.float32)
        for b in range(3):
            sig[k, b], med[k, b], clip[k, b] = clipped_sigma(x[b][frame])
    out = {"n_stamps": N, "seed": SEED, "frame_px": FRAME, "normalisation": cfg.normalisation.content_hash,
           "cache": str(cache.cache_dir), "per_band": {}}
    for b, name in enumerate(BANDS):
        out["per_band"][name] = {
            "sigma_median": float(np.median(sig[:, b])),
            "sigma_p16": float(np.percentile(sig[:, b], 16)),
            "sigma_p84": float(np.percentile(sig[:, b], 84)),
            "sky_level_median": float(np.median(med[:, b])),
            "clipped_fraction_median": float(np.median(clip[:, b])),
        }
    (OUT / "sky_sigma_v1.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
