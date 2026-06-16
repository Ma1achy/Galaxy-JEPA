"""Tests for the ViT-S/16 encoder (docs/spec/encoder.md).

The encoder is the frozen object the whole measurement layer is built against, so these
pin the contract: it satisfies the ``Encoder`` Protocol, produces the declared shapes at
the pinned penultimate layer, the JEPA hooks (patchify / run a token subset) behave, and
``is_frozen``/``assert_frozen`` work on it (the probing freeze boundary).
"""

from __future__ import annotations

import pytest
import torch

from galaxy_jepa.core.encoder import DEFAULT_LAYER, Encoder, assert_frozen, is_frozen
from galaxy_jepa.models.vit import VisionTransformer, vit_s16


def _tiny() -> VisionTransformer:
    # small + shallow so the test is fast; the contract is identical to the @256² default
    return VisionTransformer(img_size=32, patch_size=16, embed_dim=16, depth=3, heads=2)


def test_satisfies_encoder_protocol():
    model = _tiny()
    assert isinstance(model, Encoder)
    assert model.embed_dim == 16
    assert model.name == "vit_s16_jepa"


def test_encode_shapes():
    model = _tiny()
    x = torch.randn(4, 3, 32, 32)
    assert model.encode(x).shape == (4, 16)  # pooled (B, embed_dim)
    assert model.encode_tokens(x).shape == (4, 4, 16)  # (B, N=2x2 grid, embed_dim)
    assert model.encode_at(x, 0).shape == (4, 16)


def test_default_layer_is_penultimate():
    model = _tiny()
    x = torch.randn(2, 3, 32, 32)
    layers = model.layer_tokens(x)
    assert len(layers) == 3  # one per block
    # encode_tokens reads DEFAULT_LAYER (-2): the penultimate block output
    assert torch.equal(model.encode_tokens(x), layers[DEFAULT_LAYER])


def test_jepa_hooks_run_token_subset():
    model = _tiny()
    x = torch.randn(2, 3, 32, 32)
    tokens = model.patch_embed_tokens(x)
    assert tokens.shape == (2, 4, 16)  # (B, N, dim) with positional encoding added
    # the context encoder runs over a *subset* of tokens (here the first two positions)
    subset = tokens[:, :2, :]
    out = model.run_tokens(subset)
    assert out.shape == (2, 2, 16)


def test_default_config_is_vit_s16_at_256():
    model = vit_s16()
    assert model.embed_dim == 384
    assert model.num_tokens == 256  # 16x16 token grid at 256² / 16-px patches
    assert len(model.blocks) == 12


def test_frozen_state_helpers():
    model = _tiny()
    assert not is_frozen(model)  # fresh module is trainable
    for p in model.parameters():
        p.requires_grad_(False)
    assert is_frozen(model)
    assert_frozen(model)  # must not raise once frozen


def test_assert_frozen_raises_on_trainable():
    model = _tiny()
    with pytest.raises(RuntimeError):
        assert_frozen(model)
