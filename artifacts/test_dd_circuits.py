"""Unit tests for Brief DD Part 4 (hooks, latent ablation) — synthetic, CPU, tiny ViT.

The S1 test proper (ablating S1's latent moves AA3a's PC1 score) runs at Stop 3 on the real SAE;
here the same code path is exercised on a planted latent whose decoder vector *is* a read-out
direction, so ablation must move that read-out and leave an orthogonal one still.

  uv run pytest artifacts/test_dd_circuits.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import dd_core as D  # noqa: E402
from dd_circuits import Hooks, ablate_latent, pooled_with  # noqa: E402
from dd_sae import TopKSAE  # noqa: E402

D.DEVICE = "cpu"


def _tiny():
    from galaxy_jepa.models.vit import VisionTransformer
    torch.manual_seed(0)
    m = VisionTransformer(img_size=64, patch_size=16, in_chans=3, embed_dim=32, depth=12, heads=2)
    return m.eval()


def test_capture_matches_block_tokens():
    m = _tiny()
    x = torch.randn(2, 3, 64, 64)
    with Hooks(m, capture=(6, 11)) as h:
        D.pooled(m, m.patch_embed_tokens(x))
    ref = D.block_tokens(m, x, (6, 11))
    for b in (6, 11):
        assert torch.allclose(h.acts[b], ref[b])


def test_identity_patch_is_a_no_op_and_hooks_detach():
    m = _tiny()
    x = torch.randn(2, 3, 64, 64)
    a = pooled_with(m, x)
    b = pooled_with(m, x, {6: lambda t: t})
    assert torch.allclose(a, b)
    assert all(len(blk._forward_hooks) == 0 for blk in m.blocks)


def test_ablating_a_planted_latent_moves_its_readout_only():
    m = _tiny()
    x = torch.randn(4, 3, 64, 64)
    with Hooks(m, capture=(11,)) as h:
        D.pooled(m, m.patch_embed_tokens(x))
    d = 32
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((d, 2)))
    u, v = q[:, 0], q[:, 1]  # orthonormal: the planted latent's direction, and an orthogonal read-out
    sae = TopKSAE(d, 8, k=8)
    with torch.no_grad():
        sae.W_dec.data[:, 0] = torch.tensor(u, dtype=torch.float32)
        sae.W_enc.data[0] = torch.tensor(u, dtype=torch.float32)
        sae.b_enc.data[0] = 100.0  # always active, with a large code
    ro_u = D.Readout(["u"], u[:, None], np.zeros(1))
    ro_v = D.Readout(["v"], v[:, None], np.zeros(1))
    out = ablate_latent(m, sae, 11, 0, x, [ro_u, ro_v])
    assert np.all(np.abs(out["u"]) > 1.0)
    assert np.all(np.abs(out["v"]) < 1e-4)
    # an ablation at block 11 is exactly the token shift, pooled: Δu = −mean(z0)
    with torch.no_grad():
        z0 = sae.encode(h.acts[11])[..., 0].mean(1).numpy()
    assert np.allclose(out["u"], -z0, rtol=1e-4, atol=1e-3)


def test_ablating_a_dead_latent_changes_nothing():
    m = _tiny()
    x = torch.randn(4, 3, 64, 64)
    sae = TopKSAE(32, 8, k=2)
    with torch.no_grad():
        sae.b_enc.data[5] = -1e6  # never in the top-k: dead
    rng = np.random.default_rng(1)
    ro = D.Readout(["a", "b"], rng.standard_normal((32, 2)), np.zeros(2))
    out = ablate_latent(m, sae, 11, 5, x, [ro])
    assert np.all(out["a"] == 0) and np.all(out["b"] == 0)
