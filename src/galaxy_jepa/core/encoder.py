"""The ``Encoder`` Protocol — the keystone seam.

Implements ``docs/architecture.md`` "The keystone — the ``Encoder`` Protocol" and
``docs/spec/encoder.md``.

The cross-objective control (``DECISIONS.md`` D12) requires the *same* probe ladder
to run identically over JEPA, MAE, and contrastive encoders, so the seam between
"how an encoder was trained" and "how it is probed" is a structural
:class:`typing.Protocol` — satisfied by duck typing, no shared base class. All
training machinery (masking, EMA target, predictor, losses) lives in ``objectives/``
and is *not* part of this contract: the object you pretrain is the object you freeze
and probe.

Two decisions are pinned here (see ``docs/spec/encoder.md`` for the reasoning):

* **Pooling** — :meth:`Encoder.encode` returns the *mean* over the chosen layer's
  token embeddings. Objective-agnostic: JEPA and MAE do not train a CLS token the way
  some supervised ViTs do, so a mean keeps the D12 comparison free of a
  pooling artefact.
* **Layer policy** — the headline comparison reads the **penultimate** transformer
  block for all three objectives (``DEFAULT_LAYER``). MAE's final block specialises
  for its decoder and under-probes; a fixed, matched depth keeps a rung difference
  attributable to the objective rather than a per-result layer choice. Intermediate
  layers remain reachable via :meth:`Encoder.encode_at` for the supplementary
  layer-profile analysis, but the layer is never optimised per result.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

#: Layer selector for the headline probe: the penultimate transformer block,
#: matched across JEPA / MAE / contrastive. ``encode()`` reads this layer.
DEFAULT_LAYER: int = -2


@runtime_checkable
class Encoder(Protocol):
    """Images -> embeddings. The only contract the probing/eval layer knows about.

    JEPA, MAE, and contrastive encoders all satisfy this without a shared base class.
    ``encode``/``encode_tokens`` return the **pre-projection backbone** representation
    (projection and decoder heads excluded), pooled identically across objectives.
    """

    #: Width of the pooled embedding returned by :meth:`encode`.
    embed_dim: int

    #: Stable, human-readable slug identifying the encoder (e.g. ``"vit_s16_jepa"``).
    #: Used in artefact stamps and the cross-objective comparison table.
    name: str

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Pooled embedding at the pinned :data:`DEFAULT_LAYER`.

        Args:
            images: ``(B, C, H, W)`` batch.

        Returns:
            ``(B, embed_dim)`` — mean over the penultimate layer's tokens.
        """
        ...

    def encode_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Per-token embeddings at the pinned :data:`DEFAULT_LAYER`.

        Returns:
            ``(B, N, embed_dim)`` for ``N`` tokens.
        """
        ...

    def encode_at(self, images: torch.Tensor, layer: int) -> torch.Tensor:
        """Pooled embedding at an explicit transformer block.

        For the supplementary AUC-vs-depth layer profile only. The headline ladder
        uses :meth:`encode` (the pinned penultimate layer); ``layer`` is never chosen
        to maximise a per-result metric (see ``docs/spec/encoder.md``).

        Returns:
            ``(B, embed_dim)`` — mean over ``layer``'s tokens.
        """
        ...


def is_frozen(encoder: object) -> bool:
    """Return ``True`` iff no parameter of ``encoder`` requires gradients.

    Structural enforcement of the "encoder frozen during probing" hard invariant
    (``docs/architecture.md`` "Hard invariants"). Duck-typed on ``parameters()`` so it
    works for any ``nn.Module``-shaped object; an object exposing no parameters is
    vacuously frozen.
    """
    parameters = getattr(encoder, "parameters", None)
    if parameters is None:
        raise TypeError(
            f"{type(encoder).__name__} exposes no parameters(); cannot check frozen state"
        )
    return all(not p.requires_grad for p in parameters())


def assert_frozen(encoder: object) -> None:
    """Raise ``RuntimeError`` unless every parameter of ``encoder`` is frozen.

    The probing layer calls this on entry: a still-trainable encoder fails loudly
    rather than letting labels silently bend the representation.
    """
    if not is_frozen(encoder):
        trainable = [
            name
            for name, p in getattr(encoder, "named_parameters", lambda: [])()
            if p.requires_grad
        ]
        raise RuntimeError(
            "Encoder must be frozen during probing, but parameters still require "
            f"gradients: {trainable or '<unnamed>'}. Freeze the encoder before probing; "
            "there is no unfreeze path on the probing API."
        )
