"""MAE objective — He et al. 2022 on M's backbone (``artifacts/baselines.md`` §4; D12 amended).

The reconstructive arm of D12: the data and the encoder are M's, only the objective differs. A
uniform 75% of each stamp's 256 patch tokens is masked; the encoder runs on the 64 visible tokens
only (``patch_embed_tokens`` → gather → ``run_tokens``, the hooks JEPA's context path uses); an
8-block, 512-wide decoder (F2) takes the encoded visible tokens plus a learned mask token at every
masked position, adds 2-D sin-cos positions, and predicts each patch's pixels; the loss is MSE on
the masked patches only, against the per-patch normalised target (norm-pix, F5).

Everything objective-specific lives here. The mask token and the decoder never touch the encoder,
so the saved encoder is a plain :class:`VisionTransformer`, and ``load_frozen_encoder`` and
``encode()`` read it over all 256 tokens, unmasked, exactly as they read M.

**Sources, as settled (§4.2, F2, F5, F7).** Mask ratio 0.75 and the loss on masked patches only
(He §3, §4); decoder 8 × 512, 16 heads (He's default); norm-pix with ``var`` unbiased and eps 1e-6
inside the root, as He's code computes it; AdamW betas (0.9, 0.95), weight decay 0.05 with none on
biases or 1-D parameters, as the MAE code's ``add_weight_decay`` does (the mask token, 3-D, decays
there too). No augmentation (§2).
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from galaxy_jepa.callbacks.collapse import CollapseFloorFreeze
from galaxy_jepa.models.vit import VisionTransformer, _Block, _sincos_2d
from galaxy_jepa.objectives.loop import LossParts


@dataclasses.dataclass
class MaeConfig:
    steps: int = 253_270
    batch_size: int = 32
    lr: float = 1.875e-5  # F7: blr 1.5e-4 × 32 / 256
    lr_final: float | None = 1.875e-8  # M's cosine shape: the floor at 1e-3 of the peak
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 1250
    mask_ratio: float = 0.75
    dec_dim: int = 512
    dec_depth: int = 8
    dec_heads: int = 16
    norm_pix: bool = True
    norm_pix_eps: float = 1e-6
    monitor_every: int = 100
    checkpoint_every: int = 6331
    seed: int = 0


def patchify(images: torch.Tensor, patch: int) -> torch.Tensor:
    """``(B, C, H, W)`` → ``(B, N, patch·patch·C)``, tokens in the encoder's row-major order."""
    b, c, h, w = images.shape
    g = h // patch
    x = images.reshape(b, c, g, patch, g, patch)
    return x.permute(0, 2, 4, 3, 5, 1).reshape(b, g * g, patch * patch * c)


def uniform_mask(
    batch: int, n_tokens: int, n_visible: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per sample, a uniform random ``n_visible`` of ``n_tokens`` kept; the rest masked. Sorted
    index arrays ``(B, n_visible)`` and ``(B, n_tokens − n_visible)``. A pure function of ``seed``,
    which the loop sets to ``cfg.seed + step``, so a resume reproduces the masks with no saved RNG
    (as JEPA's masker)."""
    rng = np.random.default_rng(seed)
    perm = np.argsort(rng.random((batch, n_tokens)), axis=1)
    return np.sort(perm[:, :n_visible], axis=1), np.sort(perm[:, n_visible:], axis=1)


class MaeDecoder(nn.Module):
    """Encoded visible tokens + mask tokens at the masked positions → per-patch pixels."""

    def __init__(
        self,
        *,
        embed_dim: int,
        grid_size: int,
        patch_px: int,
        in_chans: int,
        dim: int = 512,
        depth: int = 8,
        heads: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.embed = nn.Linear(embed_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.register_buffer("pos_embed", _sincos_2d(dim, grid_size).unsqueeze(0))
        self.blocks = nn.ModuleList([_Block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, patch_px * patch_px * in_chans)
        self.n_tokens = grid_size * grid_size

    def forward(self, visible: torch.Tensor, vis_idx: torch.Tensor) -> torch.Tensor:
        b, _, _ = visible.shape
        x = self.mask_token.expand(b, self.n_tokens, -1).clone()
        x = x.scatter(1, vis_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]), self.embed(visible))
        x = x + cast(torch.Tensor, self.pos_embed)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))  # (B, N, patch²·C)


class Mae(nn.Module):
    """Encoder (visible tokens only) + decoder, with the masked-patch MSE loss step."""

    name = "mae"

    def __init__(self, encoder: VisionTransformer, config: MaeConfig):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.decoder = MaeDecoder(
            embed_dim=encoder.embed_dim,
            grid_size=encoder.grid_size,
            patch_px=encoder.patch_size,
            in_chans=int(cast(int, encoder.config["in_chans"])),
            dim=config.dec_dim,
            depth=config.dec_depth,
            heads=config.dec_heads,
        )
        self.n_visible = int(round(encoder.num_tokens * (1.0 - config.mask_ratio)))

    def target(self, images: torch.Tensor) -> torch.Tensor:
        t = patchify(images, self.encoder.patch_size)
        if self.config.norm_pix:
            mean = t.mean(dim=-1, keepdim=True)
            var = t.var(dim=-1, keepdim=True)
            t = (t - mean) / (var + self.config.norm_pix_eps) ** 0.5
        return t

    def token_losses(
        self, batch: dict[str, Any], *, seed: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-patch MSE ``(B, N)`` and the masked indicator ``(B, N)`` — the loss before it is
        averaged, so a diagnostic (the rehearsal's sky/galaxy split, F5) reads the same numbers."""
        images = batch["image"].float()
        device = images.device
        b = images.shape[0]
        vis, msk = uniform_mask(
            b, self.encoder.num_tokens, self.n_visible, 0 if seed is None else seed
        )
        vis_idx = torch.from_numpy(vis).to(device)
        msk_idx = torch.from_numpy(msk).to(device)
        tokens = self.encoder.patch_embed_tokens(images)
        visible = torch.gather(tokens, 1, vis_idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        pred = self.decoder(self.encoder.run_tokens(visible), vis_idx)
        per_patch = (pred - self.target(images)).pow(2).mean(dim=-1)
        masked = torch.zeros_like(per_patch).scatter(1, msk_idx, 1.0)
        return per_patch, masked

    def loss_parts(self, batch: dict[str, Any], *, seed: int | None = None) -> LossParts:
        per_patch, masked = self.token_losses(batch, seed=seed)
        loss = (per_patch * masked).sum() / masked.sum()
        return LossParts(total=loss, prediction=loss, sigreg=None)

    # ── the objective interface (objectives/loop.py) ──────────────────────────────────────────

    def make_optimiser(self) -> torch.optim.Optimizer:
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for name, p in [*self.encoder.named_parameters(), *self.decoder.named_parameters()]:
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
        return torch.optim.AdamW(
            [
                {"params": no_decay, "weight_decay": 0.0},
                {"params": decay, "weight_decay": self.config.weight_decay},
            ],
            lr=self.config.lr,
            betas=self.config.betas,
        )

    def momentum_update(self, step: int) -> None:
        """MAE has no momentum encoder."""

    @property
    def soft_rank_floor(self) -> bool:
        return False  # F8: M's soft floor was fitted on JEPA traces; reported, never gating here

    def soft_floor_note(self, floor: CollapseFloorFreeze) -> str:
        return (
            f"MAE: M's soft rank floor ({floor.min_effective_rank:.2f}) is reported, not gating "
            f"(F8: it was fitted on JEPA traces). The hard floor ({floor.hard_floor:.2f}) and the "
            "std floor still apply."
        )

    def checkpoint_modules(self) -> dict[str, nn.Module]:
        return {"encoder": self.encoder, "decoder": self.decoder}
