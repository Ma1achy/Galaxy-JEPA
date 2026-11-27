"""Independent differential reference & regression suite for I-JEPA and MoCo (§2, REG-001/015/019).

Compares :class:`galaxy_jepa.objectives.jepa.Jepa` and :class:`galaxy_jepa.objectives.moco.Moco`
against an independently written CPU float64 scalar-loop reference oracle (no shared code path
with ``Jepa.loss_parts``). Explicitly documents Galaxy-JEPA's equal-block MSE choice vs Meta
I-JEPA's released Smooth-L1 reference training script (``facebookresearch/ijepa``).
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from galaxy_jepa.masking.blocks import MaskConfig
from galaxy_jepa.models.vit import VisionTransformer
from galaxy_jepa.objectives.jepa import Jepa, JepaConfig, Predictor
from galaxy_jepa.objectives.moco import Moco, MocoConfig


def _tiny_jepa(seed: int = 0, *, dtype: torch.dtype = torch.float64) -> Jepa:
    torch.manual_seed(seed)
    enc = VisionTransformer(img_size=32, patch_size=8, embed_dim=16, depth=2, heads=2)
    cfg = JepaConfig(
        steps=10,
        batch_size=4,
        pred_dim=12,
        pred_depth=2,
        pred_heads=2,
        mask=MaskConfig(n_target_blocks=3, beta=0.0),
    )
    model = Jepa(enc, cfg).to(dtype=dtype)
    return model


def _independent_scalar_gather(tokens: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Explicit scalar-loop gather oracle (B, N, D) at (B, L) -> (B, L, D)."""
    b, l_len = idx.shape
    d = tokens.shape[-1]
    out = torch.empty((b, l_len, d), dtype=tokens.dtype, device=tokens.device)
    for bi in range(b):
        for li in range(l_len):
            out[bi, li] = tokens[bi, int(idx[bi, li])]
    return out


def _independent_predictor_forward(
    pred: Predictor,
    z_c: torch.Tensor,
    ctx_idx: torch.Tensor,
    target_idx: torch.Tensor,
) -> torch.Tensor:
    """Manual Transformer predictor oracle reading raw tensors only (no Predictor.forward).

    Catches dropped target/context positional embeddings, wrong attention scaling, or wrong slice.
    """
    b, lc, _ = z_c.shape
    pos = pred.pos_embed[0]  # (N, pred_dim)
    x_ctx = F.linear(z_c, pred.embed.weight, pred.embed.bias) + _independent_scalar_gather(
        pos.unsqueeze(0).expand(b, -1, -1), ctx_idx
    )
    x_tgt = pred.mask_token.expand(b, target_idx.shape[1], -1) + _independent_scalar_gather(
        pos.unsqueeze(0).expand(b, -1, -1), target_idx
    )
    h = torch.cat([x_ctx, x_tgt], dim=1)
    for blk in pred.blocks:
        # Pre-norm multi-head self-attention via explicit Q K^T / sqrt(d_h)
        h1 = F.layer_norm(h, (h.shape[-1],), blk.norm1.weight, blk.norm1.bias, blk.norm1.eps)
        qkv = F.linear(h1, blk.attn.qkv.weight, blk.attn.qkv.bias)
        nh, hd = blk.attn.heads, blk.attn.head_dim
        qkv = qkv.reshape(b, h.shape[1], 3, nh, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.softmax((q @ k.transpose(-2, -1)) / (hd**0.5), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, h.shape[1], nh * hd)
        h = h + F.linear(out, blk.attn.proj.weight, blk.attn.proj.bias)
        # Pre-norm MLP
        h2 = F.layer_norm(h, (h.shape[-1],), blk.norm2.weight, blk.norm2.bias, blk.norm2.eps)
        m = F.gelu(F.linear(h2, blk.mlp[0].weight, blk.mlp[0].bias))
        h = h + F.linear(m, blk.mlp[2].weight, blk.mlp[2].bias)
    h = F.layer_norm(h, (h.shape[-1],), pred.norm.weight, pred.norm.bias, pred.norm.eps)
    return F.linear(h[:, lc:, :], pred.head.weight, pred.head.bias)


def _independent_ijepa_oracle(
    jepa: Jepa,
    images: torch.Tensor,
    ctx_idx: torch.Tensor,
    target_blocks: list[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Independent per-block gather, manual predictor forward & equal-block MSE oracle."""
    all_tokens = jepa.encoder.patch_embed_tokens(images)
    ctx_tokens = _independent_scalar_gather(all_tokens, ctx_idx)
    z_c = jepa.encoder.run_tokens(ctx_tokens, norm=True)

    with torch.no_grad():
        teacher_full = jepa.target_encoder.run_tokens(
            jepa.target_encoder.patch_embed_tokens(images), norm=True
        )

    preds: list[torch.Tensor] = []
    block_losses: list[torch.Tensor] = []
    for t_idx in target_blocks:
        y_hat = _independent_predictor_forward(jepa.predictor, z_c, ctx_idx, t_idx)
        y_true = _independent_scalar_gather(teacher_full, t_idx)
        # Explicit hand-computed per-block MSE: mean over (B, n_i, D)
        sq = (y_hat - y_true) ** 2
        b_loss = sq.sum() / float(sq.numel())
        preds.append(y_hat)
        block_losses.append(b_loss)

    agg_loss = sum(block_losses) / float(len(block_losses))
    return z_c, preds, block_losses, agg_loss


@pytest.mark.regression
@pytest.mark.differential
def test_reg001_independent_target_block_prediction_and_gradient_isolation() -> None:
    """REG-001: Predictor matches manual oracle, is position-sensitive, and isolates blocks."""
    jepa = _tiny_jepa(seed=1, dtype=torch.float64).eval()
    torch.manual_seed(2)
    z_c = torch.randn(2, 5, 16, dtype=torch.float64)
    ctx_idx = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    t_a = torch.tensor([[10, 11], [0, 1]])
    t_b1 = torch.tensor([[12, 13], [2, 3]])  # equal size to t_b2 (tests location, not length!)
    t_b2 = torch.tensor([[14, 15], [10, 11]])

    # 1. Exact parity against independent manual Transformer predictor forward (raw weights only)
    pred_a = jepa.predictor(z_c, ctx_idx, t_a)
    pred_a_ref = _independent_predictor_forward(jepa.predictor, z_c, ctx_idx, t_a)
    assert torch.allclose(pred_a, pred_a_ref, rtol=1e-11, atol=1e-11)

    # 2. Fixed-length target-location AND context-location sensitivity (catches dropped _pos!)
    pred_b1 = jepa.predictor(z_c, ctx_idx, t_b1)
    pred_b2 = jepa.predictor(z_c, ctx_idx, t_b2)
    assert not torch.allclose(pred_b1, pred_b2, atol=1e-4), "target pos_embed dropped"
    ctx_shifted = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    assert not torch.allclose(pred_a, jepa.predictor(z_c, ctx_shifted, t_a), atol=1e-4), (
        "context pos_embed dropped"
    )

    # 3. Historical buggy joint-concatenation MUST couple block A to equal-size block B changes:
    joint_1 = jepa.predictor(z_c, ctx_idx, torch.cat([t_a, t_b1], dim=1))[:, : t_a.shape[1]]
    joint_2 = jepa.predictor(z_c, ctx_idx, torch.cat([t_a, t_b2], dim=1))[:, : t_a.shape[1]]
    assert not torch.allclose(joint_1, joint_2, atol=1e-4), (
        "distinguishing fixture failed: joint concatenation did not couple block A to block B"
    )

    # 4. Production Jepa.loss_parts check: loss(t_a, t_b1) == 0.5 * (loss(t_a) + loss(t_b1)).
    #    Under joint concatenation, blocks cross-attend so loss([t_a, t_b1]) != average of singles!
    jepa32 = _tiny_jepa(seed=1, dtype=torch.float32).eval()
    imgs = torch.randn(2, 3, 32, 32, dtype=torch.float32)
    batch = {
        "image": imgs,
        "petro_rad_arcsec": torch.tensor([2.0, 3.0], dtype=torch.float32),
        "pixel_scale": torch.tensor([0.396, 0.396], dtype=torch.float32),
    }

    def _run_loss(blocks: list[torch.Tensor]) -> float:
        class _StubMasker:
            def sample(self, _w: object, *, seed: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
                return ctx_idx, blocks

        jepa32.masker = _StubMasker()  # type: ignore[assignment]
        with torch.no_grad():
            return float(jepa32.loss_parts(batch, seed=0).total)

    l_a = _run_loss([t_a])
    l_b1 = _run_loss([t_b1])
    l_ab = _run_loss([t_a, t_b1])
    l_ba = _run_loss([t_b1, t_a])
    assert l_ab == pytest.approx(0.5 * (l_a + l_b1), rel=1e-5, abs=1e-6)
    assert l_ba == pytest.approx(l_ab, rel=1e-6, abs=1e-6)


@pytest.mark.regression
@pytest.mark.differential
def test_reg019_equal_block_vs_token_weighted_mse_on_unequal_blocks() -> None:
    """REG-019: Unequal target blocks distinguish equal-block MSE vs legacy token-weighted MSE."""
    # Block 1: n_1 = 2 tokens, residual = 1.0 everywhere -> block MSE = 1.0
    # Block 2: n_2 = 8 tokens, residual = 3.0 everywhere -> block MSE = 9.0
    # Equal-block MSE = (1.0 + 9.0) / 2 = 5.0
    # Legacy token-weighted MSE = (2 * 1.0 + 8 * 9.0) / 10 = 7.4
    d = 4
    p1 = torch.zeros(1, 2, d, dtype=torch.float64)
    y1 = torch.ones(1, 2, d, dtype=torch.float64)
    p2 = torch.zeros(1, 8, d, dtype=torch.float64)
    y2 = torch.full((1, 8, d), 3.0, dtype=torch.float64)

    equal_block = 0.5 * (F.mse_loss(p1, y1) + F.mse_loss(p2, y2))
    token_weighted = F.mse_loss(torch.cat([p1, p2], dim=1), torch.cat([y1, y2], dim=1))
    assert float(equal_block) == pytest.approx(5.0, abs=1e-12)
    assert float(token_weighted) == pytest.approx(7.4, abs=1e-12)
    assert abs(float(equal_block) - float(token_weighted)) > 2.0

    # Verify Jepa.loss_parts uses equal-block MSE against our independent scalar-loop oracle
    jepa = _tiny_jepa(seed=3, dtype=torch.float32)
    torch.manual_seed(4)
    images = torch.randn(2, 3, 32, 32, dtype=torch.float32)
    ctx_idx = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    t_blocks = [
        torch.tensor([[10, 11], [0, 1]]),
        torch.tensor([[12, 13, 14, 15, 6, 7], [2, 3, 4, 10, 11, 12]]),  # unequal size + overlap
    ]

    class _FixedMasker:
        def sample(self, _w: object, *, seed: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
            return ctx_idx, t_blocks

    jepa.masker = _FixedMasker()  # type: ignore[assignment]
    batch = {
        "image": images,
        "petro_rad_arcsec": torch.tensor([2.0, 3.0], dtype=torch.float32),
        "pixel_scale": torch.tensor([0.396, 0.396], dtype=torch.float32),
    }
    # Compare loss AND parameter gradients against the independent oracle
    jepa_ref = copy.deepcopy(jepa)
    parts = jepa.loss_parts(batch, seed=0)
    parts.total.backward()

    _, _, _, oracle_loss = _independent_ijepa_oracle(jepa_ref, images, ctx_idx, t_blocks)
    oracle_loss.backward()

    assert float(parts.total.detach()) == pytest.approx(
        float(oracle_loss.detach()), rel=1e-5, abs=1e-6
    )
    for (n1, p_live), (n2, p_ref) in zip(
        jepa.named_parameters(), jepa_ref.named_parameters(), strict=True
    ):
        assert n1 == n2
        if n1.startswith("target_encoder."):
            assert p_live.grad is None and not p_live.requires_grad
        else:
            assert p_live.grad is not None and p_ref.grad is not None
            assert torch.allclose(p_live.grad, p_ref.grad, rtol=1e-4, atol=1e-5), n1


@pytest.mark.differential
def test_ijepa_ema_exactness_and_finite_difference_gradients() -> None:
    """Verify exact EMA arithmetic (m=0, m=1, interior) and fp64 finite-difference gradients."""
    jepa = _tiny_jepa(seed=10, dtype=torch.float64)
    # Perturb student so student != teacher
    with torch.no_grad():
        for p in jepa.encoder.parameters():
            p.add_(0.25)

    before_t = [p.detach().clone() for p in jepa.target_encoder.parameters()]
    student_s = [p.detach().clone() for p in jepa.encoder.parameters()]

    # m = 1.0 -> teacher unchanged
    jepa.ema_update(momentum=1.0)
    for pt, bt in zip(jepa.target_encoder.parameters(), before_t, strict=True):
        assert torch.equal(pt, bt)

    # m = 0.7 -> exact convex combination
    jepa.ema_update(momentum=0.7)
    for pt, bt, ps in zip(jepa.target_encoder.parameters(), before_t, student_s, strict=True):
        assert torch.allclose(pt, 0.7 * bt + 0.3 * ps, rtol=1e-13, atol=1e-14)

    # m = 0.0 -> teacher equals student exactly
    jepa.ema_update(momentum=0.0)
    for pt, ps in zip(jepa.target_encoder.parameters(), student_s, strict=True):
        assert torch.allclose(pt, ps, rtol=1e-14, atol=1e-14)

    # Teacher parameters must NOT appear in optimiser param groups
    opt = jepa.make_optimiser()
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    for p in jepa.target_encoder.parameters():
        assert id(p) not in opt_ids

    # Central finite-difference check in float64 on predictor head & encoder patch_embed weights
    torch.manual_seed(11)
    images = torch.randn(2, 3, 32, 32, dtype=torch.float64)
    ctx_idx = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    t_blocks = [torch.tensor([[8, 9], [10, 11]]), torch.tensor([[12, 13, 14], [0, 1, 2]])]

    def _eval_loss() -> torch.Tensor:
        _, _, _, loss = _independent_ijepa_oracle(jepa, images, ctx_idx, t_blocks)
        return loss

    jepa.zero_grad(set_to_none=True)
    _eval_loss().backward()
    eps = 1e-6
    for param in (jepa.predictor.head.weight, jepa.encoder.patch_embed.weight):
        assert param.grad is not None
        g_auto = float(param.grad.view(-1)[0])
        with torch.no_grad():
            orig = float(param.view(-1)[0])
            param.view(-1)[0] = orig + eps
            lp = float(_eval_loss())
            param.view(-1)[0] = orig - eps
            lm = float(_eval_loss())
            param.view(-1)[0] = orig
        g_fd = (lp - lm) / (2.0 * eps)
        assert g_auto == pytest.approx(g_fd, rel=1e-5, abs=1e-7)


@pytest.mark.regression
def test_reg015_moco_nonfinite_loss_skips_queue_update_and_fp32_infonce() -> None:
    """REG-015: MoCo keeps InfoNCE/queue in fp32 under bf16 and skips enqueue on non-finite loss."""
    torch.manual_seed(0)
    enc = VisionTransformer(img_size=32, patch_size=8, embed_dim=16, depth=2, heads=2)
    cfg = MocoConfig(
        steps=5,
        batch_size=4,
        queue=16,
        proj_hidden=32,
        proj_dim=8,
        window_min=2,
        window_max=3,
    )
    moco = Moco(enc, cfg)
    ptr_before = int(moco.queue_ptr.item())
    q_before = moco.queue.clone()

    # Non-finite image -> non-finite loss -> queue must NOT be corrupted or advanced
    bad_batch = {"image": torch.full((4, 3, 32, 32), float("nan"))}
    parts = moco.loss_parts(bad_batch, seed=0)
    assert not torch.isfinite(parts.total)
    assert int(moco.queue_ptr.item()) == ptr_before
    assert torch.equal(moco.queue, q_before)

    # Clean batch under CPU bf16 autocast -> finite fp32 loss and unit-norm fp32 queue
    clean_batch = {"image": torch.randn(4, 3, 32, 32)}
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        parts_bf16 = moco.loss_parts(clean_batch, seed=1)
    assert parts_bf16.total.dtype == torch.float32
    assert torch.isfinite(parts_bf16.total)
    assert moco.queue.dtype == torch.float32
    assert torch.isfinite(moco.queue).all()
    assert int(moco.queue_ptr.item()) == (ptr_before + 4) % 16


def test_predictor_rejects_empty_target_or_out_of_range_indices() -> None:
    pred = Predictor(embed_dim=16, grid_size=4, pred_dim=8, depth=1, heads=2)
    ctx = torch.randn(2, 4, 16)
    ctx_idx = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    with pytest.raises(IndexError):
        pred(ctx, ctx_idx, torch.tensor([[99], [99]]))
