"""I-JEPA objective — context/target encoders, predictor, latent-MSE (docs/spec/objectives.md).

A config-driven training procedure that **produces a frozen encoder**. The pieces live here,
not on the encoder (``docs/spec/objectives.md`` §2): the bbox-biased masking
(``masking/blocks.py``), the EMA target encoder, the predictor, and the latent-MSE loss. On
export it writes a provenance-stamped checkpoint that ``probing/`` reloads frozen — the freeze
boundary runs through disk (§3).

The recipe is standard I-JEPA: the context encoder sees only the context tokens; the EMA
target encoder embeds the full image; the predictor reconstructs the target-block embeddings
from the context (+ mask tokens at the target positions); the loss is the L2 distance in
latent space. The *only* departure is **where** the target blocks are sampled — the bbox bias
— and that is isolated in the masker, so β=0 reproduces the published method exactly.

Device-agnostic (CPU / MPS / CUDA). On MPS the masking math is numpy (host) and only the
tensor ops run on device; bf16 autocast and SDPA are the throughput levers (the slice plan).
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from galaxy_jepa.callbacks.collapse import CollapseMonitor
from galaxy_jepa.data.bbox import petrosian_box
from galaxy_jepa.masking.blocks import (
    MaskConfig,
    MultiBlockMasker,
    box_to_token_mask,
    token_weight_map,
)
from galaxy_jepa.models.vit import VisionTransformer, _Block, _sincos_2d, save_encoder

logger = logging.getLogger(__name__)


def _gather_tokens(tokens: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather ``(B, L, D)`` from ``(B, N, D)`` tokens at per-sample indices ``(B, L)``."""
    d = tokens.shape[-1]
    return torch.gather(tokens, 1, idx.unsqueeze(-1).expand(-1, -1, d))


class Predictor(nn.Module):
    """Narrow ViT predictor: context tokens + mask tokens → target-block embeddings."""

    def __init__(
        self,
        *,
        embed_dim: int,
        grid_size: int,
        pred_dim: int = 192,
        depth: int = 6,
        heads: int = 6,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.embed = nn.Linear(embed_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.register_buffer("pos_embed", _sincos_2d(pred_dim, grid_size).unsqueeze(0))
        self.blocks = nn.ModuleList([_Block(pred_dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(pred_dim)
        self.head = nn.Linear(pred_dim, embed_dim)

    def _pos(self, idx: torch.Tensor) -> torch.Tensor:
        pos_embed = cast(torch.Tensor, self.pos_embed)[0]  # (N, pred_dim)
        return pos_embed[idx]  # (N, D) indexed by (B, L) -> (B, L, D)

    def forward(
        self, context: torch.Tensor, context_idx: torch.Tensor, target_idx: torch.Tensor
    ) -> torch.Tensor:
        b, lc, _ = context.shape
        x = self.embed(context) + self._pos(context_idx)  # (B, Lc, pred_dim)
        masks = self.mask_token.expand(b, target_idx.shape[1], -1) + self._pos(target_idx)
        h = torch.cat([x, masks], dim=1)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.head(h[:, lc:, :])  # predictions at the target positions


@dataclasses.dataclass
class JepaConfig:
    """Training hyperparameters (stamped via the run config; the slice plan's defaults)."""

    steps: int = 1000
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.04
    warmup_steps: int = 100
    ema_start: float = 0.996
    ema_end: float = 1.0
    pred_dim: int = 192
    pred_depth: int = 6
    pred_heads: int = 6
    mask: MaskConfig = dataclasses.field(default_factory=MaskConfig)
    petro_k: float = 2.5
    global_box_frac: float = 0.40  # global-fallback box half-width as a fraction of stamp_px
    monitor_every: int = 100
    seed: int = 0


class Jepa(nn.Module):
    """Context encoder + EMA target encoder + predictor, with the latent-MSE loss step."""

    def __init__(self, encoder: VisionTransformer, config: JepaConfig):
        super().__init__()
        self.encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(
            embed_dim=encoder.embed_dim,
            grid_size=encoder.grid_size,
            pred_dim=config.pred_dim,
            depth=config.pred_depth,
            heads=config.pred_heads,
        )
        self.config = config
        self.masker = MultiBlockMasker(encoder.grid_size, config.mask)
        self.stamp_px = encoder.grid_size * encoder.patch_size
        self._global_half = config.global_box_frac * self.stamp_px

    def weight_maps(self, petro_arcsec: np.ndarray, pixel_scale: np.ndarray) -> np.ndarray:
        """Build the per-sample token weight maps from the per-galaxy Petrosian boxes."""
        maps = []
        for pr, ps in zip(petro_arcsec, pixel_scale, strict=True):
            box = petrosian_box(
                float(pr) if math.isfinite(pr) else None,
                float(ps),
                k=self.config.petro_k,
                stamp_px=self.stamp_px,
                global_half_width_px=self._global_half,
            )
            in_box = box_to_token_mask(box.half_width_px, self.stamp_px, self.encoder.grid_size)
            maps.append(token_weight_map(in_box, self.config.mask.beta))
        return np.stack(maps)

    def loss_step(self, batch: dict[str, Any], *, seed: int | None = None) -> torch.Tensor:
        """One latent-MSE step on a batch dict (image + petro/pixel-scale)."""
        images = batch["image"].float()
        device = images.device
        petro = batch["petro_rad_arcsec"].cpu().numpy()
        pscale = batch["pixel_scale"].cpu().numpy()
        weight_maps = self.weight_maps(petro, pscale)
        context_idx, target_idx = self.masker.sample(weight_maps, seed=seed)
        context_idx, target_idx = context_idx.to(device), target_idx.to(device)

        tokens = self.encoder.patch_embed_tokens(images)
        context = self.encoder.run_tokens(_gather_tokens(tokens, context_idx))

        with torch.no_grad():
            full = self.target_encoder.run_tokens(self.target_encoder.patch_embed_tokens(images))
            targets = _gather_tokens(full, target_idx)

        pred = self.predictor(context, context_idx, target_idx)
        return torch.nn.functional.mse_loss(pred, targets)

    @torch.no_grad()
    def ema_update(self, momentum: float) -> None:
        for online, target in zip(
            self.encoder.parameters(), self.target_encoder.parameters(), strict=True
        ):
            target.mul_(momentum).add_(online.detach(), alpha=1.0 - momentum)


def ema_momentum(step: int, total: int, start: float, end: float) -> float:
    """Cosine ramp from ``start`` to ``end`` over ``total`` steps (I-JEPA schedule)."""
    if total <= 1:
        return end
    t = min(step, total) / total
    return end - (end - start) * (math.cos(math.pi * t) + 1.0) / 2.0


@dataclasses.dataclass
class TrainResult:
    """What the pretrain returns: where the checkpoint landed + the traces to read."""

    checkpoint: Path | None
    losses: list[float]
    collapse_trace: dict[str, list[float]]
    halted: bool


def train_jepa(
    jepa: Jepa,
    loader: Any,
    *,
    device: str = "cpu",
    monitor_batch: dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> TrainResult:
    """Run the JEPA pretrain loop with the collapse monitor live; export a frozen checkpoint.

    ``loader`` yields batch dicts (from ``StampDataset``); it is cycled until ``config.steps``.
    ``monitor_batch`` is a fixed held-out batch (the ``pretrain-monitor`` slice) the collapse
    monitor reads. Halts on NaN/Inf loss or representation collapse, recording why.
    """
    cfg = jepa.config
    jepa.to(device)
    opt = torch.optim.AdamW(
        [*jepa.encoder.parameters(), *jepa.predictor.parameters()],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    monitor = CollapseMonitor()
    losses: list[float] = []
    halted = False

    data = _cycle(loader)
    bar = _progress(cfg.steps)
    for step in bar:
        batch = _to_device(next(data), device)
        lr = cfg.lr * min(1.0, (step + 1) / max(cfg.warmup_steps, 1))
        for group in opt.param_groups:
            group["lr"] = lr

        opt.zero_grad(set_to_none=True)
        if autocast_dtype is not None:
            with torch.autocast(device_type=device.split(":")[0], dtype=autocast_dtype):
                loss = jepa.loss_step(batch, seed=cfg.seed + step)
        else:
            loss = jepa.loss_step(batch, seed=cfg.seed + step)

        if not torch.isfinite(loss):  # T3.no-nan-loss
            logger.error("step %d: non-finite loss (%s) — halting", step, loss.item())
            halted = True
            break
        loss.backward()
        opt.step()
        jepa.ema_update(ema_momentum(step, cfg.steps, cfg.ema_start, cfg.ema_end))
        loss_val = float(loss.item())
        losses.append(loss_val)
        _set_postfix(bar, loss=loss_val)

        if monitor_batch is not None and step % cfg.monitor_every == 0:
            with torch.no_grad():
                emb = jepa.encoder.encode(_to_device(monitor_batch, device)["image"].float())
            signals = monitor.update(step, emb)
            _set_postfix(
                bar,
                loss=loss_val,
                std=signals.std,
                erank=signals.effective_rank,
                cos=signals.mean_cosine,
            )
            logger.info(
                "step %d: loss=%.4f std=%.4f erank=%.1f cos=%.3f",
                step,
                loss_val,
                signals.std,
                signals.effective_rank,
                signals.mean_cosine,
            )
            if monitor.should_halt(signals):  # T3.collapse-monitor
                logger.error("step %d: representation collapse detected — halting", step)
                halted = True
                break

    checkpoint = None
    if checkpoint_path is not None:
        checkpoint = save_encoder(
            jepa.encoder, checkpoint_path, extra={"steps": len(losses), "halted": halted}
        )
    return TrainResult(checkpoint, losses, monitor.trace(), halted)


def _progress(steps: int) -> Any:
    """A tqdm progress bar over the step range (loss + collapse metrics + ETA in the postfix).

    tqdm is a core dependency, but fall back to a bare range if it is ever absent so the
    training loop never depends on a display library being importable.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - tqdm is a declared core dep
        return range(steps)
    return tqdm(range(steps), desc="jepa", unit="step", dynamic_ncols=True)


def _set_postfix(bar: Any, **fields: float) -> None:
    """Update the bar's live readout, tolerant of the bare-range fallback."""
    set_postfix = getattr(bar, "set_postfix", None)
    if set_postfix is not None:
        set_postfix(fields, refresh=False)


def _cycle(loader: Any) -> Any:
    while True:
        yield from loader


def _to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out
