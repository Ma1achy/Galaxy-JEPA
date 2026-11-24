"""MAE with pixel-fixed masking: whole 16-px groups are masked, and the normalised target is
computed over the group (patch-grid pre-registration §3, user 21 and 24 Nov 2026).

At the encoder's patch 16 a group is one patch, so this is :class:`~galaxy_jepa.objectives.mae.Mae`
exactly: same masks, same targets, same loss. At patch 8 and 4 a group holds 4 or 16 tokens.

* **Masking.** The same ``uniform_mask`` draw is made over *groups*, so the sky hidden at every
  patch size is identical (75% of groups, same seed, same groups as at patch 16). Token order is
  the encoder's row-major order.
* **Target.** Each group's pixels are normalised by that group's mean and variance (norm-pix, F5),
  then cut into its tokens. A token therefore regresses the same numbers as the same pixels would
  at patch 16, and the loss averages over the masked tokens' pixels, which weights every masked
  pixel equally at every patch size.

``mae.py`` keeps its fp32 arithmetic and its config hashes: its one change (24 Nov) is a dtype cast
in the decoder that lets it run under bf16 autocast, and does nothing in fp32.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch

from galaxy_jepa.models.vit import VisionTransformer
from galaxy_jepa.objectives.mae import Mae, MaeConfig, patchify, uniform_mask


@dataclasses.dataclass
class GroupedMaeConfig(MaeConfig):
    #: Side of the masked group in pixels. Must be a whole multiple of the encoder's patch.
    mask_group_px: int = 16


def group_members(grid_size: int, patch: int, group_px: int) -> np.ndarray:
    """``(n_groups, k²)`` token indices, row-major groups and row-major tokens inside a group."""
    if group_px % patch:
        raise ValueError(f"mask_group_px {group_px} is not a multiple of the patch {patch}")
    k = group_px // patch
    if grid_size % k:
        raise ValueError(f"a {grid_size}-token grid does not hold whole {k}-token groups")
    g = grid_size // k
    t = np.arange(grid_size * grid_size).reshape(grid_size, grid_size)
    return t.reshape(g, k, g, k).transpose(0, 2, 1, 3).reshape(g * g, k * k)


class GroupedMae(Mae):
    """:class:`Mae` that masks and normalises whole ``mask_group_px`` groups."""

    def __init__(self, encoder: VisionTransformer, config: GroupedMaeConfig):
        super().__init__(encoder, config)
        self.members = group_members(encoder.grid_size, encoder.patch_size, config.mask_group_px)
        self.n_groups = self.members.shape[0]
        self.n_visible_groups = int(round(self.n_groups * (1.0 - config.mask_ratio)))
        self.n_visible = self.n_visible_groups * self.members.shape[1]

    def target(self, images: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        assert isinstance(cfg, GroupedMaeConfig)
        t = patchify(images, self.encoder.patch_size)  # (B, N, p²C)
        if not cfg.norm_pix:
            return t
        g = patchify(images, cfg.mask_group_px)  # (B, n_groups, g²C): the whole group's pixels
        mean = g.mean(dim=-1, keepdim=True)
        var = g.var(dim=-1, keepdim=True)
        scale = (var + cfg.norm_pix_eps) ** 0.5
        owner = np.empty(self.members.size, dtype=np.int64)
        for i, row in enumerate(self.members):
            owner[row] = i
        own = torch.from_numpy(owner).to(images.device)
        return (t - mean[:, own]) / scale[:, own]

    def token_losses(
        self, batch: dict[str, Any], *, seed: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = batch["image"].float()
        device = images.device
        b = images.shape[0]
        vis_g, msk_g = uniform_mask(
            b, self.n_groups, self.n_visible_groups, 0 if seed is None else seed
        )
        vis_idx = torch.from_numpy(np.sort(self.members[vis_g].reshape(b, -1), axis=1)).to(device)
        msk_idx = torch.from_numpy(np.sort(self.members[msk_g].reshape(b, -1), axis=1)).to(device)
        tokens = self.encoder.patch_embed_tokens(images)
        visible = torch.gather(tokens, 1, vis_idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        pred = self.decoder(self.encoder.run_tokens(visible), vis_idx)
        per_patch = (pred - self.target(images)).pow(2).mean(dim=-1)
        masked = torch.zeros_like(per_patch).scatter(1, msk_idx, 1.0)
        return per_patch, masked
