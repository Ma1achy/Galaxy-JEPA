"""MoCo objective — v2's queue on M's backbone, MoCo v3's ViT optimiser (``baselines.md`` §3).

The contrastive arm of D12. Settled (F3, F4, F7 (b)): queue K = 65,536 of normalised 128-d keys,
τ = 0.2, a 2-layer MLP projector (384 → 2048 → 128, no BatchNorm, no predictor), a key encoder and
key projector updated as a constant 0.999 moving average of the query side, and v2's **asymmetric**
InfoNCE: the query is view 1, the key view 2, the positive at index 0 and the queue's K keys as
negatives. AdamW, torch's default betas, weight decay 0.1 on every trained parameter, at
1.5e-4 × 32 / 256 (MoCo v3's ViT rule).

**Views (§3.4): minimal, objective-intrinsic, no flips, no rotations, no resampling.** Per step one
window side s ∈ {10, …, 14} tokens, shared by the batch; per image and view an independent window
position. Each view is the image after a gain, a Gaussian blur at p = 0.5 and additive
Gaussian noise, then ``patch_embed_tokens`` and a gather of the window: the hook JEPA's context path
uses, so positions are an exact subset of the fixed sin-cos grid and the full-grid read-out stays in
distribution. Every draw comes from generators seeded with the loop's per-step seed, so a resume
reproduces the views with no saved augmentation state.

**Gain and noise, settled (user, 2026-10-13; §3.4).** A global gain ×U(0.9, 1.1), the same for all
bands (brightness, not colour); a per-band gain ×U(0.98, 1.02), independent per band, about SDSS's
photometric calibration and small enough to keep colour; additive noise σ_aug = u × σ_sky per band,
u ~ U(0, 1) per view, with σ_sky the normalised sky σ measured in M's v1 cache
(``artifacts/sky_sigma.py``). Augmentation defines MoCo's invariances: a ±10% per-band gain would
make it invariant to colour, a strong morphology signal, and handicap it on what is probed.

The projector reads the mean of the window's final-block tokens after the final norm (there is no
CLS token). The saved encoder is the query encoder, a plain :class:`VisionTransformer`, read by
``encode()`` over all 256 tokens; the projector and the key branch are discarded.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from galaxy_jepa.callbacks.collapse import CollapseFloorFreeze
from galaxy_jepa.models.vit import VisionTransformer
from galaxy_jepa.objectives.loop import LossParts


@dataclasses.dataclass
class MocoConfig:
    steps: int = 253_270
    batch_size: int = 32
    lr: float = 1.875e-5  # F7 (b): MoCo v3's ViT rule, 1.5e-4 × 32 / 256
    lr_final: float | None = 1.875e-8  # M's cosine shape: the floor at 1e-3 of the peak
    weight_decay: float = 0.1
    warmup_steps: int = 1250
    queue: int = 65_536
    momentum: float = 0.999
    tau: float = 0.2
    proj_hidden: int = 2048
    proj_dim: int = 128
    window_min: int = 10  # tokens per side; 16 is the full grid
    window_max: int = 14
    gain_global: float = 0.1  # ×U(1 − g, 1 + g), one draw per image and view, all bands alike
    gain_band: float = 0.02  # ×U(1 − g, 1 + g), independent per band, image and view
    blur_sigma: tuple[float, float] = (0.1, 2.0)  # px, uniform; v2's aug+
    blur_p: float = 0.5
    #: The normalised sky σ per band (g, r, i): median over 2,000 train stamps' 16-px frames of a
    #: 3σ-clipped MAD σ in M's v1 cache (``artifacts/sky_sigma.py``, 2026-10-13).
    sky_sigma: tuple[float, float, float] = (0.414, 0.501, 0.612)
    noise_u_max: float = 1.0  # σ_aug = u × sky_sigma, u ~ U(0, noise_u_max) per image and view
    monitor_every: int = 100
    checkpoint_every: int = 6331
    seed: int = 0


class Projector(nn.Sequential):
    def __init__(self, dim: int, hidden: int, out: int):
        super().__init__(nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out))


def _gauss_kernels(sigma: torch.Tensor, radius: int) -> torch.Tensor:
    """1-D normalised Gaussian kernels, one per σ ``(M,)`` → ``(M, 2·radius + 1)``; σ = 0 is the
    identity (a unit impulse), so an unblurred view is exactly unblurred."""
    x = torch.arange(-radius, radius + 1, device=sigma.device, dtype=torch.float32)
    s = sigma.clamp_min(1e-6).unsqueeze(1)
    k = torch.exp(-0.5 * (x / s) ** 2)
    k = torch.where(sigma.unsqueeze(1) > 0, k, (x == 0).float().expand_as(k))
    return k / k.sum(dim=1, keepdim=True)


def blur(images: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Separable Gaussian blur, σ per image ``(B,)`` (shared across bands), reflect-padded so the
    stamp's edge is not darkened."""
    b, c, h, w = images.shape
    radius = max(1, math.ceil(3.0 * float(sigma.max())))
    k = _gauss_kernels(sigma, radius).repeat_interleave(c, dim=0)  # (B·C, L)
    x = images.reshape(1, b * c, h, w)
    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, k.view(b * c, 1, 1, -1), groups=b * c)
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    x = F.conv2d(x, k.view(b * c, 1, -1, 1), groups=b * c)
    return x.reshape(b, c, h, w)


class Moco(nn.Module):
    """Query encoder + projector, momentum key branch, the negative queue, InfoNCE."""

    name = "moco"

    def __init__(self, encoder: VisionTransformer, config: MocoConfig):
        super().__init__()
        if config.queue % config.batch_size:
            raise ValueError(
                f"queue {config.queue} must be a multiple of batch {config.batch_size}"
            )
        self.encoder = encoder
        self.config = config
        self.projector = Projector(encoder.embed_dim, config.proj_hidden, config.proj_dim)
        self.key_encoder = copy.deepcopy(encoder)
        self.key_projector = copy.deepcopy(self.projector)
        for p in [*self.key_encoder.parameters(), *self.key_projector.parameters()]:
            p.requires_grad_(False)
        # MoCo v1's initialisation: random unit keys, replaced as real keys arrive
        g = torch.Generator().manual_seed(config.seed)
        self.register_buffer(
            "queue", F.normalize(torch.randn(config.queue, config.proj_dim, generator=g), dim=1)
        )
        self.register_buffer("queue_ptr", torch.zeros((), dtype=torch.long))
        #: The last step's contrastive top-1 (positive ranked first), for the rehearsal's shortcut
        #: gate (§6). A diagnostic; not training state.
        self.last_top1: float = math.nan

    # ── views ─────────────────────────────────────────────────────────────────────────────────

    def views(self, images: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Two augmented views as token windows ``(B, s², D)`` each (query, key), and the side s.
        Every draw is from generators seeded by ``seed``: the side and positions on the CPU (small),
        the per-pixel noise on the images' device."""
        cfg = self.config
        b, c = images.shape[:2]
        if len(cfg.sky_sigma) != c:
            raise ValueError(f"sky_sigma has {len(cfg.sky_sigma)} bands, the images {c}")
        sky = torch.tensor(cfg.sky_sigma, dtype=images.dtype).view(1, c, 1, 1)
        grid = self.encoder.grid_size
        g = torch.Generator().manual_seed(int(seed))
        side = int(torch.randint(cfg.window_min, cfg.window_max + 1, (1,), generator=g))
        gd = torch.Generator(device=images.device).manual_seed(int(seed) + 1)
        out = []
        for _view in range(2):
            gain = (1.0 + cfg.gain_global * (2.0 * torch.rand(b, 1, 1, 1, generator=g) - 1.0)) * (
                1.0 + cfg.gain_band * (2.0 * torch.rand(b, c, 1, 1, generator=g) - 1.0)
            )
            sd = cfg.noise_u_max * torch.rand(b, 1, 1, 1, generator=g) * sky
            on = torch.rand(b, generator=g) < cfg.blur_p
            lo, hi = cfg.blur_sigma
            sig = torch.where(on, lo + (hi - lo) * torch.rand(b, generator=g), torch.zeros(b))
            x = images * gain.to(images.device)
            if bool(on.any()):
                x = blur(x, sig.to(images.device))
            x = x + sd.to(x.device) * torch.randn(x.shape, generator=gd, device=x.device)
            y0 = torch.randint(0, grid - side + 1, (b,), generator=g)
            x0 = torch.randint(0, grid - side + 1, (b,), generator=g)
            r = torch.arange(side)
            idx = (
                (y0[:, None, None] + r[None, :, None]) * grid
                + (x0[:, None, None] + r[None, None, :])
            ).reshape(b, side * side)
            tokens = (
                self.encoder.patch_embed_tokens(x)
                if _view == 0
                else self.key_encoder.patch_embed_tokens(x)
            )
            idx = idx.to(images.device)
            out.append(torch.gather(tokens, 1, idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])))
        return out[0], out[1], side

    # ── loss ──────────────────────────────────────────────────────────────────────────────────

    def loss_parts(self, batch: dict[str, Any], *, seed: int | None = None) -> LossParts:
        images = batch["image"].float()
        q_tok, k_tok, _ = self.views(images, 0 if seed is None else seed)
        q = F.normalize(self.projector(self.encoder.run_tokens(q_tok).mean(dim=1)), dim=1)
        with torch.no_grad():
            k = F.normalize(
                self.key_projector(self.key_encoder.run_tokens(k_tok).mean(dim=1)), dim=1
            )
        queue = cast(torch.Tensor, self.queue)
        l_pos = (q * k).sum(dim=1, keepdim=True)
        l_neg = q @ queue.clone().detach().T
        logits = torch.cat([l_pos, l_neg], dim=1) / self.config.tau
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            self.last_top1 = float((logits.argmax(dim=1) == 0).float().mean())
            self._enqueue(k)
        return LossParts(total=loss, prediction=loss, sigreg=None)

    @torch.no_grad()
    def _enqueue(self, keys: torch.Tensor) -> None:
        queue, ptr = cast(torch.Tensor, self.queue), int(cast(torch.Tensor, self.queue_ptr))
        queue[ptr : ptr + keys.shape[0]] = keys
        self.queue_ptr.fill_((ptr + keys.shape[0]) % queue.shape[0])  # type: ignore[operator]

    # ── the objective interface (objectives/loop.py) ──────────────────────────────────────────

    def make_optimiser(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            [*self.encoder.parameters(), *self.projector.parameters()],
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    @torch.no_grad()
    def momentum_update(self, step: int) -> None:
        m = self.config.momentum  # constant (F3 residue, v2)
        for q, k in (
            (self.encoder, self.key_encoder),
            (self.projector, self.key_projector),
        ):
            for pq, pk in zip(q.parameters(), k.parameters(), strict=True):
                pk.mul_(m).add_(pq.detach(), alpha=1.0 - m)

    @property
    def soft_rank_floor(self) -> bool:
        return False  # F8: M's soft floor was fitted on JEPA traces; reported, never gating here

    def soft_floor_note(self, floor: CollapseFloorFreeze) -> str:
        return (
            f"MoCo: M's soft rank floor ({floor.min_effective_rank:.2f}) is reported, not gating "
            f"(F8: it was fitted on JEPA traces). The hard floor ({floor.hard_floor:.2f}) and the "
            "std floor still apply."
        )

    def checkpoint_modules(self) -> dict[str, nn.Module]:
        return {
            "encoder": self.encoder,
            "projector": self.projector,
            "key_encoder": self.key_encoder,
            "key_projector": self.key_projector,
        }

    def extra_state(self) -> dict[str, torch.Tensor]:
        """The queue is training state: without it a resume trains on different negatives."""
        return {
            "queue": cast(torch.Tensor, self.queue).clone(),
            "queue_ptr": cast(torch.Tensor, self.queue_ptr).clone(),
        }

    def load_extra_state(self, state: dict[str, torch.Tensor]) -> None:
        cast(torch.Tensor, self.queue).copy_(state["queue"])
        cast(torch.Tensor, self.queue_ptr).copy_(state["queue_ptr"])
