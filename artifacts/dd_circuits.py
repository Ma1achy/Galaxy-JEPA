"""Brief DD, Part 4: groundwork for circuits — capture and patch activations at every block, and SAE
feature ablation. BUILD ONLY; no analysis. TOOL VALIDATION (artifacts/interp_tooling.md).

`Hooks` captures any block's output tokens and can replace them on the way through (a forward hook
returning a new output). `ablate_latent` zeroes one SAE latent at a layer while keeping the SAE's
reconstruction error (x − x̂ is added back), so only that latent's contribution moves:
    x' = x − W_dec[:, j] · z_j / scale
then runs forward to block 11, pools, and reports the change in each read-out.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import dd_core as D  # noqa: E402


class Hooks:
    """Forward hooks on `model.blocks` (1-based names). `capture` records outputs; `patch` maps
    {block: fn(tokens) -> tokens} replace them. Use as a context manager so hooks never leak."""

    def __init__(self, model, capture: tuple[int, ...] = (), patch: dict[int, Callable] | None = None) -> None:
        self.model = model
        self.capture = set(capture)
        self.patch = patch or {}
        self.acts: dict[int, torch.Tensor] = {}
        self.__handles: list = []

    def __enter__(self) -> Hooks:
        for i, blk in enumerate(self.model.blocks, 1):
            if i in self.capture or i in self.patch:
                self.__handles.append(blk.register_forward_hook(self.__hook(i)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self.__handles:
            h.remove()
        self.__handles.clear()

    def __hook(self, i: int):
        def fn(_mod, _inp, out):
            if i in self.patch:
                out = self.patch[i](out)
            if i in self.capture:
                self.acts[i] = out.detach()
            return out
        return fn


@torch.no_grad()
def pooled_with(model, images: torch.Tensor, patch: dict[int, Callable] | None = None) -> torch.Tensor:
    """Block-11 mean-pooled embedding under optional patches (the probe read-out's input)."""
    with Hooks(model, patch=patch):
        return D.pooled(model, model.patch_embed_tokens(images))


def latent_patch(sae, j: int) -> Callable:
    """A patch fn removing latent j's contribution from every token (error term kept)."""
    def fn(tokens: torch.Tensor) -> torch.Tensor:
        z = sae.encode(tokens)
        return tokens - (z[..., j:j + 1] * sae.W_dec[:, j]) / sae.scale
    return fn


@torch.no_grad()
def ablate_latent(model, sae, layer: int, j: int, images: torch.Tensor, readouts: list) -> dict[str, np.ndarray]:
    """Per read-out name: score change (ablated − original) per image, in read-out SD units when the
    read-out is calibrated."""
    base = pooled_with(model, images).double().cpu().numpy()
    abl = pooled_with(model, images, {layer: latent_patch(sae, j)}).double().cpu().numpy()
    out = {}
    for ro in readouts:
        d = ro.scores(abl) - ro.scores(base)
        if ro.sd is not None:
            d = d / ro.sd
        for k, n in enumerate(ro.names):
            out[n] = d[:, k]
    return out
