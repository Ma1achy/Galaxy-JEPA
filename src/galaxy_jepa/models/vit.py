"""ViT-S/16 @256² — the from-scratch encoder for the JEPA vertical slice.

Implements ``docs/spec/encoder.md``: a standard ViT-Small backbone (``embed_dim=384``,
depth 12, 6 heads, 16-px patches → a 16×16 = 256-token grid for a 256² stamp), satisfying
the :class:`~galaxy_jepa.core.encoder.Encoder` Protocol. The headline probe reads the
**penultimate** block (``DEFAULT_LAYER = -2``), mean-pooled over tokens — no CLS token, so
the pooling is objective-agnostic and identical across JEPA/MAE/contrastive later.

The JEPA-specific machinery (predictor, EMA target, masking, loss) is **not** here — it
lives in ``objectives/jepa.py`` (``docs/spec/objectives.md``). What this module exposes for
the objective is the minimum it needs: patch-embedding with positional encoding
(:meth:`patch_embed_tokens`) and running the transformer blocks over an arbitrary set of
tokens (:meth:`run_tokens`), so the context encoder can process a *subset* of tokens and
the target encoder the full set. Attention uses ``F.scaled_dot_product_attention`` (the
memory-efficient kernel, supported on MPS).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from galaxy_jepa.core.encoder import DEFAULT_LAYER


def _sincos_2d(dim: int, grid: int) -> torch.Tensor:
    """Fixed 2D sin-cos positional embedding, ``(grid*grid, dim)`` (I-JEPA convention)."""
    if dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4 for 2D sin-cos, got {dim}")
    coords = np.arange(grid, dtype=np.float64)
    gy, gx = np.meshgrid(coords, coords, indexing="ij")

    def _axis(pos: np.ndarray) -> np.ndarray:
        omega = np.arange(dim // 4, dtype=np.float64) / (dim / 4.0)
        omega = 1.0 / (10000.0**omega)
        out = pos.reshape(-1)[:, None] * omega[None, :]
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    emb = np.concatenate([_axis(gy), _axis(gx)], axis=1)  # (grid*grid, dim)
    return torch.from_numpy(emb).float()


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (b, heads, n, head_dim)
        out = F.scaled_dot_product_attention(q, k, v)  # memory-efficient / flash kernel
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """ViT backbone satisfying the :class:`~galaxy_jepa.core.encoder.Encoder` Protocol.

    ``encode``/``encode_tokens``/``encode_at`` operate on full ``(B, C, H, W)`` images and
    read the pinned penultimate layer; ``patch_embed_tokens``/``run_tokens`` are the lower-
    level hooks the JEPA objective drives directly.
    """

    def __init__(
        self,
        *,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        name: str = "vit_s16_jepa",
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} not divisible by patch_size {patch_size}")
        self.embed_dim = embed_dim
        self.name = name
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_tokens = self.grid_size**2
        # explicit constructor record so a frozen checkpoint reloads to the same architecture
        self.config: dict[str, object] = {
            "img_size": img_size,
            "patch_size": patch_size,
            "in_chans": in_chans,
            "embed_dim": embed_dim,
            "depth": depth,
            "heads": heads,
            "mlp_ratio": mlp_ratio,
            "name": name,
        }

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.register_buffer("pos_embed", _sincos_2d(embed_dim, self.grid_size).unsqueeze(0))
        self.blocks = nn.ModuleList([_Block(embed_dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    # --- low-level hooks for the JEPA objective ------------------------------------

    def patch_embed_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """``(B, C, H, W)`` → ``(B, N, embed_dim)`` patch tokens with positional encoding."""
        x = self.patch_embed(images)  # (B, dim, grid, grid)
        x = x.flatten(2).transpose(1, 2)  # (B, N, dim)
        return x + self.pos_embed

    def run_tokens(self, tokens: torch.Tensor, *, norm: bool = True) -> torch.Tensor:
        """Run the transformer blocks over arbitrary tokens; final-block (norm) output.

        The context encoder passes a *subset* of tokens (positions already encoded by
        :meth:`patch_embed_tokens` then gathered); the target encoder passes all tokens.
        """
        x = tokens
        for block in self.blocks:
            x = block(x)
        return self.norm(x) if norm else x

    def layer_tokens(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Per-block token outputs for the full image (no final norm), one per block."""
        x = self.patch_embed_tokens(images)
        outs: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            outs.append(x)
        return outs

    # --- Encoder Protocol ----------------------------------------------------------

    def encode_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Per-token embeddings at the pinned :data:`DEFAULT_LAYER` → ``(B, N, embed_dim)``."""
        return self.layer_tokens(images)[DEFAULT_LAYER]

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Pooled embedding at :data:`DEFAULT_LAYER` → ``(B, embed_dim)`` (mean over tokens)."""
        return self.encode_tokens(images).mean(dim=1)

    def encode_at(self, images: torch.Tensor, layer: int) -> torch.Tensor:
        """Pooled embedding at an explicit block (the supplementary depth profile only)."""
        return self.layer_tokens(images)[layer].mean(dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)


def vit_s16(**kwargs: object) -> VisionTransformer:
    """ViT-S/16 @256² with the spec defaults (``docs/spec/encoder.md``)."""
    return VisionTransformer(**kwargs)  # type: ignore[arg-type]


# --- checkpoint I/O (the freeze boundary; docs/spec/objectives.md §3) ----------------
#
# Both writer (objectives/) and reader (probing/) go through here, so probing never has to
# import objectives/ — it consumes a checkpoint + this module. The checkpoint carries the
# constructor record so the architecture reloads exactly.


def save_encoder(model: VisionTransformer, path: str | Path, *, extra: dict | None = None) -> Path:
    """Write the encoder weights + its constructor config to a checkpoint."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": model.config, "state_dict": model.state_dict(), "extra": extra or {}}
    torch.save(payload, out)
    return out


def load_frozen_encoder(path: str | Path, *, map_location: str = "cpu") -> VisionTransformer:
    """Rebuild the encoder from a checkpoint and return it **frozen** (eval, no grad).

    The probing layer's entry point: the object returned passes ``assert_frozen`` — there
    is no unfreeze path, so labels can never bend the representation (``docs/spec/encoder.md``).
    """
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    model = VisionTransformer(**payload["config"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
