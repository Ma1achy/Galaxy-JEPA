"""Bounding-box-biased multi-block masking (docs/masking.md).

The one genuinely novel architectural piece. It is a **strict generalisation** of I-JEPA's
multi-block masking: block sizes/counts are unchanged; only *where* target blocks are
sampled from changes, via a per-token weight map derived from the per-galaxy box and a
single bias strength ``β``:

* ``w = 1`` inside the galaxy box, ``w = 1 − β`` on sky;
* target-block top-left positions are sampled ∝ the mean weight under the block.

``β = 0`` ⇒ uniform weights ⇒ **uniform multi-block target sampling** (our unbiased I-JEPA
masking control). A full-frame box ⇒ uniform for any ``β``. As ``β`` rises, target blocks
concentrate on the galaxy and the "sky waste" (target tokens that are sky) falls — the direct
evidence the scheme works.

Everything operates on the **token grid** (``G×G``); pixel→token projection of the box is
:func:`box_to_token_mask`. The block-size distribution, counts, and context-exclusion rule follow
I-JEPA; this module owns the weighting and sampling. The masker returns, per batch, rectangular
``context``/``target`` index tensors (per-sample positions, sizes shared across the batch, context
subsampled uniformly without replacement to the batch minimum rather than Meta's index-order
truncation) so the objective can gather tokens without ragged batches.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

__all__ = [
    "MaskConfig",
    "MultiBlockMasker",
    "box_to_token_mask",
    "draw_blocks",
    "rebuild_masks",
    "token_weight_map",
    "sky_waste",
]


def box_to_token_mask(half_width_px: float, stamp_px: int, grid_size: int) -> np.ndarray:
    """Project a centred pixel box onto the token grid → boolean ``(G, G)`` in-box mask.

    A token is in-box if its centre pixel lies within ``±half_width_px`` of the stamp
    centre (per axis), so the box is the symmetric central region the galaxy occupies.
    """
    patch = stamp_px / grid_size
    centres = (np.arange(grid_size) + 0.5) * patch
    centre = stamp_px / 2.0
    in_axis = np.abs(centres - centre) <= half_width_px  # (G,)
    return np.outer(in_axis, in_axis)


def token_weight_map(in_box: np.ndarray, beta: float) -> np.ndarray:
    """Sampling weight per token: ``1`` in-box, ``1 − β`` on sky (docs/masking.md §4.1)."""
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1] (T1.beta-in-range), got {beta}")
    return np.where(in_box, 1.0, 1.0 - beta)


@dataclass(frozen=True)
class MaskConfig:
    """I-JEPA block geometry + the bbox-bias knobs (docs/masking.md §5). β is the lever."""

    n_target_blocks: int = 4
    target_scale: tuple[float, float] = (0.15, 0.20)
    target_aspect: tuple[float, float] = (0.75, 1.50)
    context_scale: tuple[float, float] = (0.85, 1.00)
    beta: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1] (T1.beta-in-range), got {self.beta}")


def _block_dims(
    scale: tuple[float, float], aspect: tuple[float, float], n: int, g: int, rng
) -> tuple[int, int]:
    area = rng.uniform(*scale) * n
    a = rng.uniform(*aspect)
    h = int(round(float(np.sqrt(area * a))))
    w = int(round(float(np.sqrt(area / a))))
    return max(1, min(h, g)), max(1, min(w, g))


def _sample_topleft(weight: np.ndarray, h: int, w: int, rng) -> tuple[int, int]:
    """Sample a block top-left ∝ mean token weight under the block (∝ score)."""
    g = weight.shape[0]
    rows, cols, scores = [], [], []
    for r in range(g - h + 1):
        for c in range(g - w + 1):
            rows.append(r)
            cols.append(c)
            scores.append(weight[r : r + h, c : c + w].mean())
    s = np.asarray(scores, dtype=np.float64)
    total = s.sum()
    p = s / total if total > 0 else None  # all-zero (β=1, all-sky block) → uniform fallback
    i = rng.choice(len(rows), p=p)
    return rows[i], cols[i]


def _block_tokens(r: int, c: int, h: int, w: int, g: int) -> np.ndarray:
    rr, cc = np.meshgrid(np.arange(r, r + h), np.arange(c, c + w), indexing="ij")
    return (rr * g + cc).reshape(-1)


def _sub_rng(seed: int | None) -> np.random.Generator:
    """Deterministic sub-generator for context min-keep subsampling, decoupled from block draws."""
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng([int(seed), 0x517CC1B7])


def draw_blocks(
    grid_size: int,
    config: MaskConfig,
    weight_maps: np.ndarray,
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw block dimensions ``(n_target + 1, 2)`` and per-sample top-left corners
    ``(B, n_target + 1, 2)`` (target blocks first, context block last).

    Target positions are sampled ∝ ``weight_maps`` (galaxy-biased); the context block has
    square aspect ``(1.0, 1.0)`` and uniform top-left sampling (``docs/masking.md`` §4.3).
    """
    rng = np.random.default_rng(seed)
    g, n = grid_size, grid_size * grid_size
    dims = [
        _block_dims(config.target_scale, config.target_aspect, n, g, rng)
        for _ in range(config.n_target_blocks)
    ]
    ctx_h, ctx_w = _block_dims(config.context_scale, (1.0, 1.0), n, g, rng)
    dims.append((ctx_h, ctx_w))
    tl = []
    for w_map in weight_maps:
        # On tiny test grids (e.g. 4x4), four 2x2 target blocks can occasionally tile every token;
        # resample target corners until at least one non-target token remains for context (never
        # fires on real 16x16 / 32x32 / 64x64 grids, and guarantees zero target-in-context leak).
        row: list[tuple[int, int]] = []
        for _attempt in range(64):
            row = [_sample_topleft(w_map, h, w, rng) for (h, w) in dims[:-1]]
            target_union = np.unique(
                np.concatenate(
                    [
                        _block_tokens(r, c, h, w, g)
                        for (r, c), (h, w) in zip(row, dims[:-1], strict=True)
                    ]
                )
            )
            if target_union.size < n:
                break
        else:
            raise ValueError(
                "target blocks cover the entire token grid; no non-overlapping context exists"
            )
        cr = int(rng.integers(0, g - ctx_h + 1))
        cc = int(rng.integers(0, g - ctx_w + 1))
        row.append((cr, cc))
        tl.append(row)
    return np.asarray(dims, dtype=np.int64), np.asarray(tl, dtype=np.int64)


def rebuild_masks(
    dims: np.ndarray,
    tl: np.ndarray,
    grid_size: int,
    *,
    seed: int | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Build ``(context_idx, target_blocks)`` from stored block dims and top-left corners.

    Each target block is kept separate (``list[Tensor]`` of length ``n_target_blocks``, each
    ``(B, h_k * w_k)``) so the predictor runs independently per target block (standard I-JEPA).
    Context is the context block minus the union of all target tokens; if a sample's targets
    cover its entire context block, non-target tokens from the rest of the grid are used rather
    than leaking a target token into context. Min-keep batch truncation subsamples uniformly at
    random rather than dropping bottom raster rows.
    """
    n_t = len(dims) - 1
    n_tokens = grid_size * grid_size
    all_tokens = np.arange(n_tokens, dtype=np.int64)
    per_block: list[list[np.ndarray]] = [[] for _ in range(n_t)]
    contexts: list[np.ndarray] = []

    for corners in tl:
        sample_blocks = []
        for k in range(n_t):
            r, c = int(corners[k][0]), int(corners[k][1])
            h, w = int(dims[k][0]), int(dims[k][1])
            blk = _block_tokens(r, c, h, w, grid_size)
            sample_blocks.append(blk)
            per_block[k].append(blk)
        target_union = np.unique(np.concatenate(sample_blocks))
        cr, cc = int(corners[n_t][0]), int(corners[n_t][1])
        ch, cw = int(dims[n_t][0]), int(dims[n_t][1])
        ctx_tokens = _block_tokens(cr, cc, ch, cw, grid_size)
        context = np.setdiff1d(ctx_tokens, target_union, assume_unique=True)
        if context.size == 0:
            context = np.setdiff1d(all_tokens, target_union, assume_unique=True)
            if context.size == 0:
                raise ValueError(
                    "target blocks cover the entire token grid; no non-overlapping context exists"
                )
        contexts.append(context)

    keep = min(c.size for c in contexts)
    sub_rng = _sub_rng(seed)
    context_idx = np.stack(
        [
            c if c.size == keep else np.sort(sub_rng.choice(c, size=keep, replace=False))
            for c in contexts
        ]
    )
    target_blocks = [torch.from_numpy(np.stack(blks)).long() for blks in per_block]
    return torch.from_numpy(context_idx).long(), target_blocks


class MultiBlockMasker:
    """Samples context/target token indices for a batch (docs/masking.md §4)."""

    def __init__(self, grid_size: int, config: MaskConfig | None = None):
        self.grid_size = grid_size
        self.n_tokens = grid_size * grid_size
        self.config = config or MaskConfig()

    def sample(
        self, weight_maps: np.ndarray, *, seed: int | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return ``(context_idx (B, Lc), target_blocks [(B, L_k), ...])`` long tensors.

        ``weight_maps`` is ``(B, G, G)`` (from :func:`token_weight_map` per galaxy). Block
        *sizes* are drawn once per batch (so token counts match and the batch is rectangular);
        target *positions* are sampled per galaxy ∝ its weight map; context *position* is
        uniform with square aspect. Context is the context block minus the union of target
        tokens, randomly subsampled to the batch-minimum length (the I-JEPA ``min_keep`` trick).
        """
        dims, tl = draw_blocks(self.grid_size, self.config, weight_maps, seed=seed)
        return rebuild_masks(dims, tl, self.grid_size, seed=seed)


def sky_waste(in_box_maps: np.ndarray, target_idx: torch.Tensor | Sequence[torch.Tensor]) -> float:
    """Fraction of sampled target tokens that are sky (the docs/masking.md §7 diagnostic)."""
    flat = in_box_maps.reshape(in_box_maps.shape[0], -1)
    if isinstance(target_idx, torch.Tensor):
        idx = target_idx.cpu().numpy()
    else:
        idx = torch.cat(list(target_idx), dim=1).cpu().numpy()
    in_box = np.take_along_axis(flat, idx, axis=1)
    return float(1.0 - in_box.mean())
