"""Brief E2 — how much of what the masker samples is padding, with the real sampler.

Drives ``MultiBlockMasker.sample`` and the same weight-map construction ``Jepa.weight_maps``
uses (``petrosian_box`` -> ``box_to_token_mask`` -> ``token_weight_map``) rather than
reimplementing any of it, so the number measured is the number training would see.

Investigation code: terse, excluded from lint/CI, reads only.

    uv run python artifacts/e2_mask_exposure.py [n_stamps] [n_draws]
"""

from __future__ import annotations

import sys
import time

import numpy as np

from galaxy_jepa.data.bbox import petrosian_box
from galaxy_jepa.data.sources import DirectorySource, load_fits_stamp
from galaxy_jepa.data.validity import invalid_planes, token_invalid_fraction
from galaxy_jepa.masking.blocks import (
    MaskConfig,
    MultiBlockMasker,
    _block_dims,
    box_to_token_mask,
    token_weight_map,
)

PIXEL_SCALE, PETRO_K, GLOBAL_BOX_FRAC, STAMP_PX, GRID = 0.396, 2.5, 0.40, 256, 16
BETAS = (0.0, 0.5, 1.0)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
DRAWS = int(sys.argv[2]) if len(sys.argv) > 2 else 20


def load(corpus: str, n: int):
    """Stride across the objID-sorted corpus — uniform over the footprint, not one patch."""
    rows = sorted(DirectorySource(f"data/{corpus}").rows, key=lambda r: int(r["object_id"]))
    step = max(1, len(rows) // n)
    picked = rows[::step][:n]
    out = []
    for r in picked:
        img = load_fits_stamp(f"data/{corpus}/{r['object_id']}.fits")
        edge, interior = invalid_planes(img)
        inv = token_invalid_fraction(edge | interior, GRID)
        half = petrosian_box(
            float(r.get("petroRad_r", float("nan")) or "nan"),
            PIXEL_SCALE,
            k=PETRO_K,
            stamp_px=STAMP_PX,
            global_half_width_px=GLOBAL_BOX_FRAC * STAMP_PX,
        ).half_width_px
        out.append((inv, box_to_token_mask(half, STAMP_PX, GRID), float((edge | interior).mean())))
    return out


def target_block_sizes(cfg: MaskConfig, seed: int) -> list[int]:
    """The per-block token counts for this draw, from the sampler's own size draw.

    ``MultiBlockMasker.sample`` draws the four target block dims first, from a fresh
    ``default_rng(seed)``, and shares them across the batch — so replaying that first draw
    reproduces them exactly. The caller asserts the total against the returned index width, so
    any change to the sampler's draw order fails loudly instead of mismeasuring quietly.
    """
    rng = np.random.default_rng(seed)
    n = GRID * GRID
    return [
        h * w
        for (h, w) in (
            _block_dims(cfg.target_scale, cfg.target_aspect, n, GRID, rng)
            for _ in range(cfg.n_target_blocks)
        )
    ]


def run(stamps, beta: float, draws: int):
    cfg = MaskConfig(beta=beta)
    masker = MultiBlockMasker(GRID, cfg)
    inv_flat = np.stack([s[0].reshape(-1) for s in stamps])
    weights = np.stack([token_weight_map(s[1], beta) for s in stamps])

    blocks = {"target": [], "context": []}
    tokens = {"target": [], "context": []}
    per_stamp = np.zeros(len(stamps))
    for d in range(draws):
        seed = 1000 + d
        ctx, tgt = masker.sample(weights, seed=seed)
        sizes = target_block_sizes(cfg, seed)
        assert sum(sizes) == tgt.shape[1], "sampler draw order changed; block sizes are wrong"

        tvals = np.take_along_axis(inv_flat, tgt.numpy(), axis=1)  # (B, sum(sizes))
        tokens["target"].append(tvals.reshape(-1))
        means = np.column_stack([part.mean(axis=1) for part in np.split(tvals, np.cumsum(sizes)[:-1], axis=1)])
        blocks["target"].append(means.reshape(-1))  # (B * n_target_blocks,)
        per_stamp += (means >= 0.5).sum(axis=1)

        cvals = np.take_along_axis(inv_flat, ctx.numpy(), axis=1)
        tokens["context"].append(cvals.reshape(-1))
        blocks["context"].append(cvals.mean(axis=1))

    out = {}
    for name in ("target", "context"):
        per_block = np.concatenate(blocks[name])
        out[name] = {
            "100%": 100 * float((per_block >= 0.999).mean()),
            ">=50%": 100 * float((per_block >= 0.5).mean()),
            ">=10%": 100 * float((per_block >= 0.1).mean()),
            "aggregate": 100 * float(np.concatenate(tokens[name]).mean()),
            "n_blocks": per_block.size,
        }
    return out, per_stamp


if __name__ == "__main__":
    t0 = time.time()
    stamps = load("pretrain", N)
    contaminated = sum(1 for s in stamps if s[2] > 0)
    print(f"loaded {len(stamps)} pretrain stamps in {time.time()-t0:.0f}s; "
          f"{contaminated} ({100*contaminated/len(stamps):.1f}%) carry padding")

    for beta in BETAS:
        t = time.time()
        out, per_stamp = run(stamps, beta, DRAWS)
        print(f"\n--- beta = {beta}  ({DRAWS} draws x {len(stamps)} stamps, {time.time()-t:.0f}s) ---")
        print(f"{'':<9}{'100% pad':>10}{'>=50%':>9}{'>=10%':>9}{'aggregate':>11}")
        for name in ("target", "context"):
            r = out[name]
            print(f"{name:<9}{r['100%']:>9.2f}%{r['>=50%']:>8.2f}%{r['>=10%']:>8.2f}%{r['aggregate']:>10.2f}%")
        hit = per_stamp[per_stamp > 0]
        share = np.sort(per_stamp)[::-1]
        top10 = 100 * share[: max(1, len(share) // 10)].sum() / max(share.sum(), 1)
        print(f"concentration: {len(hit)} of {len(stamps)} stamps ever hit; "
              f"top decile carries {top10:.1f}% of >=50%-pad target blocks")
