"""JEPA's masks, precomputed (patch-grid pre-registration, user 24 Nov 2026).

``MultiBlockMasker.sample`` costs about 0.5 s a batch at a 64-token grid, so at patch 4 it would
dominate the training step. The masks are a pure function of ``(seed, step, batch)``: each step's
generator is ``default_rng(train_seed + step)``, and the batch's weight maps follow from the data
order. So they are drawn once on CPU, by the same code, and replayed.

Only the draws are stored: per step the four target block sizes and the context size, and per
galaxy the five top-left corners (about 330 bytes a step at batch 32). Index tensors are rebuilt
exactly as ``sample`` builds them. Every stored step carries a digest of the weight maps it was
drawn from, and the lookup refuses a batch whose maps differ: a data-order or box change cannot be
read past.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from galaxy_jepa.masking.blocks import (
    MaskConfig,
    MultiBlockMasker,
    _block_dims,
    _block_tokens,
    _sample_topleft,
)

SCHEME = "precomputed-masks-v1"


def maps_digest(weight_maps: np.ndarray) -> np.uint64:
    """First 8 bytes of SHA-1 over the float64 weight maps (shape included)."""
    w = np.ascontiguousarray(weight_maps, dtype=np.float64)
    h = hashlib.sha1(str(w.shape).encode() + w.tobytes()).digest()
    return np.frombuffer(h[:8], dtype=np.uint64)[0]


def draw(
    masker: MultiBlockMasker, weight_maps: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """``sample``'s generator calls, in its order → block dims ``(5, 2)`` (4 targets, context) and
    top-left corners ``(B, 5, 2)``. The exactness test replays these against ``sample`` itself."""
    rng = np.random.default_rng(seed)
    cfg, g, n = masker.config, masker.grid_size, masker.n_tokens
    dims = [
        _block_dims(cfg.target_scale, cfg.target_aspect, n, g, rng)
        for _ in range(cfg.n_target_blocks)
    ]
    dims.append(_block_dims(cfg.context_scale, cfg.context_scale, n, g, rng))
    tl = []
    for w_map in weight_maps:
        tl.append([_sample_topleft(w_map, h, w, rng) for (h, w) in dims])
    return np.asarray(dims, dtype=np.uint8), np.asarray(tl, dtype=np.uint8)


def rebuild(dims: np.ndarray, tl: np.ndarray, grid: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The index tensors ``sample`` returns, from the stored draws (its own construction)."""
    n_t = len(dims) - 1
    contexts, targets = [], []
    for corners in tl:
        target_tokens = np.concatenate(
            [
                _block_tokens(int(r), int(c), int(h), int(w), grid)
                for (r, c), (h, w) in zip(corners[:n_t], dims[:n_t], strict=True)
            ]
        )
        (cr, cc), (ch, cw) = corners[n_t], dims[n_t]
        ctx_tokens = _block_tokens(int(cr), int(cc), int(ch), int(cw), grid)
        context = np.setdiff1d(ctx_tokens, target_tokens, assume_unique=False)
        if context.size == 0:
            context = ctx_tokens[:1]
        targets.append(target_tokens)
        contexts.append(context)
    keep = min(c.size for c in contexts)
    context_idx = np.stack([np.sort(c[:keep]) for c in contexts])
    return torch.from_numpy(context_idx).long(), torch.from_numpy(np.stack(targets)).long()


class PrecomputedMasker:
    """Drop-in for :class:`MultiBlockMasker` in a :class:`Jepa`: same ``sample``, stored draws.

    ``path`` holds ``dims.npy (S, 5, 2)``, ``tl.npy (S, B, 5, 2)``, ``digest.npy (S,)`` and
    ``meta.json``. ``seed`` passed to ``sample`` is ``train_seed + step``.
    """

    def __init__(self, path: str | Path, *, grid_size: int, config: MaskConfig, train_seed: int):
        p = Path(path)
        self.meta = json.loads((p / "meta.json").read_text())
        want = {
            "scheme": SCHEME,
            "grid_size": int(grid_size),
            "train_seed": int(train_seed),
            "mask_config": _cfg_json(config),
        }
        for k, v in want.items():
            if self.meta[k] != v:
                raise ValueError(
                    f"precomputed masks at {p}: {k} is {self.meta[k]!r}, the run needs {v!r}"
                )
        self.grid_size, self.config = int(grid_size), config
        self.train_seed = int(train_seed)
        self.dims = np.load(p / "dims.npy", mmap_mode="r")
        self.tl = np.load(p / "tl.npy", mmap_mode="r")
        self.digest = np.load(p / "digest.npy", mmap_mode="r")

    def sample(
        self, weight_maps: np.ndarray, *, seed: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seed is None:
            raise ValueError("precomputed masks need the step's seed (train_seed + step)")
        step = int(seed) - self.train_seed
        if not 0 <= step < len(self.digest):
            raise IndexError(f"step {step} is outside the {len(self.digest):,} precomputed steps")
        if maps_digest(weight_maps) != self.digest[step]:
            raise ValueError(
                f"precomputed masks, step {step}: this batch's weight maps differ from those the "
                "masks were drawn from (data order, boxes or batch size changed)"
            )
        return rebuild(np.asarray(self.dims[step]), np.asarray(self.tl[step]), self.grid_size)


def _cfg_json(c: MaskConfig) -> dict:
    return json.loads(
        json.dumps(
            {
                "n_target_blocks": c.n_target_blocks,
                "target_scale": c.target_scale,
                "target_aspect": c.target_aspect,
                "context_scale": c.context_scale,
                "beta": c.beta,
            }
        )
    )
