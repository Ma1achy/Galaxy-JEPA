"""SIGReg — the Epps–Pulley statistic and the properties the ablation leans on (Brief I).

Unit-level and CPU-only. What matters here is that the reformulation is the paper's statistic
and not merely something shaped like it, and that the penalty actually *identifies* the thing
it claims to: near its floor for an isotropic Gaussian batch, large for anything else.
"""

from __future__ import annotations

import math

import pytest
import torch

from galaxy_jepa.objectives.sigreg import (
    DEFAULT_DOMAIN,
    DEFAULT_QUAD_POINTS,
    sigreg,
    sketch_directions,
)


def _algorithm_1(x: torch.Tensor, *, seed: int, slices: int) -> torch.Tensor:
    """The paper's algorithm 1, transcribed in complex arithmetic — the reference.

    Only the sketch draw is shared with the implementation; the statistic is computed the
    paper's way, so agreement is evidence about the reformulation rather than a tautology.
    """
    a = sketch_directions(x.shape[1], slices, seed=seed)
    t = torch.linspace(-DEFAULT_DOMAIN, DEFAULT_DOMAIN, DEFAULT_QUAD_POINTS)
    exp_f = torch.exp(-0.5 * t**2)
    x_t = (x @ a).unsqueeze(2) * t
    ecf = (1j * x_t).exp().mean(0)
    err = (ecf - exp_f).abs().square().mul(exp_f)
    return (torch.trapezoid(err, t, dim=1) * x.shape[0]).mean()


def _batch(seed: int, n: int = 32, d: int = 64, scale: float = 1.0) -> torch.Tensor:
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed)) * scale


class TestItIsThePapersStatistic:
    @pytest.mark.invariant
    @pytest.mark.parametrize("scale", [0.5, 1.0, 2.5, 4.0])
    def test_real_arithmetic_matches_the_complex_form(self, scale):
        """The modulus expands exactly, so this is an identity — not an approximation.

        The complex form needs kernels MPS supports only patchily; the real form is what runs.
        If these ever diverge, the runs are measuring something the paper did not define.
        """
        x = _batch(0, scale=scale)
        assert sigreg(x, seed=11, slices=256) == pytest.approx(
            float(_algorithm_1(x, seed=11, slices=256)), rel=1e-6, abs=1e-9
        )

    @pytest.mark.invariant
    def test_the_sketch_is_unit_norm_directions_and_is_reproducible(self):
        a = sketch_directions(64, 128, seed=5)
        assert a.shape == (64, 128)
        assert torch.allclose(a.norm(p=2, dim=0), torch.ones(128), atol=1e-6)
        assert torch.equal(a, sketch_directions(64, 128, seed=5))
        assert not torch.equal(a, sketch_directions(64, 128, seed=6))

    def test_the_sketch_is_drawn_on_the_host_so_it_cannot_vary_by_backend(self):
        """Deliberate departure from algorithm 1, which seeds a device generator.

        ``runtime.device`` is hashed into ``config_hash`` because backends differ numerically.
        The *sketch* has no reason to be one of the places they differ, so it is drawn on the
        CPU and moved — which is only true while this stays a plain CPU tensor.
        """
        assert sketch_directions(8, 4, seed=0).device.type == "cpu"


class TestItIdentifiesIsotropy:
    @pytest.mark.invariant
    def test_an_isotropic_gaussian_batch_sits_far_below_anything_else(self):
        """Necessary direction of the whole ablation: the minimum is N(0, 1).

        The floor is not zero — the finite-batch estimate is positively biased (the paper's
        thm. 6), measured at ≈ 1.0 for n=32 — so the assertion is about separation, not size.
        """
        iso = _batch(1)
        floor = sigreg(iso, seed=2)
        assert floor < 2.0
        assert sigreg(iso * 4.0, seed=2) > 5.0 * floor  # wrong scale
        assert sigreg(iso + 3.0, seed=2) > 5.0 * floor  # wrong mean
        rank_one = _batch(3, d=1) @ _batch(4, n=1)
        assert sigreg(rank_one, seed=2) > 2.0 * floor  # degenerate covariance

    def test_gradient_descent_on_it_moves_a_batch_towards_isotropy(self):
        """Not a training test — evidence the penalty is optimisable, not merely computable.

        Its gradients are bounded by design (the paper's thm. 4), which is what makes it stable
        and also what makes it *slow* far from the target; that trade-off is why the arms record
        whether it converged within their step budget rather than assuming it did.

        Note what this does **not** show. The statistic reaches its floor here while the batch's
        std is still ≈ 1.7, because at n=32 an empirical CF over random 1-D projections pins the
        scale only loosely — satisfaction accumulates across resampled directions over steps,
        not within one. That looseness is the measured form of the isotropic-but-empty risk.
        """
        z = (_batch(7, d=32, scale=4.0)).requires_grad_(True)
        opt = torch.optim.AdamW([z], lr=1e-2, weight_decay=0.0)
        first = None
        for step in range(800):
            opt.zero_grad()
            loss = sigreg(z, seed=step, slices=256)
            loss.backward()
            opt.step()
            first = float(loss.detach()) if step == 0 else first
        assert float(loss.detach()) < 0.1 * first  # 25.5 -> 0.9 as measured
        assert 0.8 < float(z.detach().std()) < 2.5  # from 4.0, and not overshooting to zero


class TestItRefusesWhatItCannotMeasure:
    @pytest.mark.invariant
    @pytest.mark.parametrize("shape", [(32,), (4, 8, 16)])
    def test_a_non_matrix_batch_is_an_error_not_a_reshape(self, shape):
        with pytest.raises(ValueError, match="expected .N, D. embeddings"):
            sigreg(torch.zeros(*shape), seed=0)

    def test_it_is_computed_in_float32_under_a_bf16_autocast(self):
        """Under bf16 the products ``t * z`` reach ~60 with ~3 significant digits, and cos/sin
        of that is noise — a penalty that still descends while measuring nothing.
        """
        x = _batch(9, scale=3.0)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            under = sigreg(x, seed=1, slices=256)
        assert under.dtype == torch.float32
        assert float(under) == pytest.approx(float(sigreg(x, seed=1, slices=256)), rel=1e-6)

    def test_it_is_finite_on_a_degenerate_batch(self):
        assert math.isfinite(float(sigreg(torch.zeros(16, 32), seed=0, slices=64)))
