"""Bounding-box-biased multi-block masking (docs/masking.md).

The one genuinely novel architectural piece. It is a **strict generalisation** of I-JEPA's
multi-block masking: block sizes/counts are unchanged; only *where* target blocks are
sampled from changes, via a per-token weight map derived from the per-galaxy box and a
single bias strength ``β``:

* ``w = 1`` inside the galaxy box, ``w = 1 − β`` on sky;
* target-block top-left positions are sampled ∝ the mean weight under the block.

``β = 0`` ⇒ uniform weights ⇒ **standard I-JEPA** (the published control). A full-frame box
⇒ uniform for any ``β``. As ``β`` rises, target blocks concentrate on the galaxy and the
"sky waste" (target tokens that are sky) falls — the direct evidence the scheme works.

Everything operates on the **token grid** (``G×G``); pixel→token projection of the box is
:func:`box_to_token_mask`. The block-size distribution, counts, and context handling are
I-JEPA's; this module owns only the weighting and the sampling. The masker returns, per
batch, rectangular ``context``/``target`` index tensors (per-sample positions, sizes shared
across the batch, context truncated to the batch minimum — the I-JEPA collator trick) so
the objective can gather tokens without ragged batches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["MaskConfig", "MultiBlockMasker", "box_to_token_mask", "token_weight_map", "sky_waste"]


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


class MultiBlockMasker:
    """Samples context/target token indices for a batch (docs/masking.md §4)."""

    def __init__(self, grid_size: int, config: MaskConfig | None = None):
        self.grid_size = grid_size
        self.n_tokens = grid_size * grid_size
        self.config = config or MaskConfig()

    def sample(
        self, weight_maps: np.ndarray, *, seed: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(context_idx (B, Lc), target_idx (B, T))`` long tensors for the batch.

        ``weight_maps`` is ``(B, G, G)`` (from :func:`token_weight_map` per galaxy). Block
        *sizes* are drawn once per batch (so token counts match and the batch is
        rectangular); *positions* are sampled per galaxy ∝ its weight map. Context is the
        context block minus the union of target tokens, truncated to the batch-minimum
        length (the I-JEPA ``min_keep`` collator trick).
        """
        rng = np.random.default_rng(seed)
        cfg = self.config
        g, n = self.grid_size, self.n_tokens

        # sizes shared across the batch → equal token counts → rectangular gather
        tgt_dims = [
            _block_dims(cfg.target_scale, cfg.target_aspect, n, g, rng)
            for _ in range(cfg.n_target_blocks)
        ]
        ctx_h, ctx_w = _block_dims(cfg.context_scale, cfg.context_scale, n, g, rng)

        batch_targets: list[np.ndarray] = []
        batch_contexts: list[np.ndarray] = []
        for w_map in weight_maps:
            target_tokens = np.concatenate(
                [_block_tokens(*_sample_topleft(w_map, h, w, rng), h, w, g) for (h, w) in tgt_dims]
            )
            cr, cc = _sample_topleft(w_map, ctx_h, ctx_w, rng)
            ctx_tokens = _block_tokens(cr, cc, ctx_h, ctx_w, g)
            context = np.setdiff1d(ctx_tokens, target_tokens, assume_unique=False)
            if context.size == 0:  # degenerate: targets ate the context — keep one token
                context = ctx_tokens[:1]
            batch_targets.append(target_tokens)
            batch_contexts.append(context)

        keep = min(c.size for c in batch_contexts)
        context_idx = np.stack([np.sort(c[:keep]) for c in batch_contexts])
        target_idx = np.stack(batch_targets)
        return (
            torch.from_numpy(context_idx).long(),
            torch.from_numpy(target_idx).long(),
        )


def sky_waste(in_box_maps: np.ndarray, target_idx: torch.Tensor) -> float:
    """Fraction of sampled target tokens that are sky (the docs/masking.md §7 diagnostic)."""
    flat = in_box_maps.reshape(in_box_maps.shape[0], -1)
    idx = target_idx.cpu().numpy()
    in_box = np.take_along_axis(flat, idx, axis=1)
    return float(1.0 - in_box.mean())
