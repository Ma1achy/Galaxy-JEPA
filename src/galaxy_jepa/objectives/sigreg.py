"""SIGReg — Sketched Isotropic Gaussian Regularisation (LeJEPA, arXiv:2511.08544 §4).

A distribution-matching penalty on the embedding batch: push the embeddings towards an
isotropic Gaussian by testing *random one-dimensional projections* of them against N(0, 1).
Definition 2 of the paper,

    SIGReg(A, {f(x_n)}) = (1/|A|) Σ_{a ∈ A} T({aᵀ f(x_n)}_{n=1..N}),

with ``T`` the Epps–Pulley statistic — the weighted L2 distance between the empirical
characteristic function of the projected samples and the CF of a standard Gaussian,

    EP = N ∫ |φ̂(t) − φ(t)|² w(t) dt,   φ(t) = w(t) = e^{−t²/2}.

Linear in N in both time and memory. The paper chooses Epps–Pulley over moment- and
CDF-based tests because it has *bounded* loss, gradient and curvature for any input
distribution (their thm. 4), where moment tests explode and CDF tests need a sort.

**Why this project has it at all.** H5 measured the training loss running *against* the
objective: the arm 19× better on latent MSE lost frozen-probe AUC decisively, because latent
MSE against a moving EMA target rewards co-adaptation onto a shared mean component. SIGReg is
the published claim to fix exactly that, so it is ablated rather than adopted — see
``artifacts/i_decision_rule.md``.

Two deliberate departures from the paper's algorithm 1, both stated in the write-up:

* **Real arithmetic.** Their empirical CF is ``(1j * x_t).exp().mean(0)``, which needs complex
  kernels that MPS supports only patchily. Because the target CF is real, the modulus expands
  exactly — ``|φ̂ − φ|² = (C − φ)² + S²`` for ``C = mean cos(tz)``, ``S = mean sin(tz)`` — so
  the same number is computed in real arithmetic. This is an identity, not an approximation,
  and ``tests/test_objectives_sigreg.py`` pins it against the complex form.
* **The projections are drawn on the host.** Their generator is seeded per step on the compute
  device; ours draws on the CPU and moves the matrix over, so the sketch directions are
  bit-identical across CPU / MPS / CUDA. ``runtime.device`` is hashed into ``config_hash``
  because backends differ numerically; there is no reason for the *sketch* to be one of the
  places they differ.
"""

from __future__ import annotations

import torch

__all__ = ["DEFAULT_DOMAIN", "DEFAULT_QUAD_POINTS", "DEFAULT_SLICES", "sigreg", "sketch_directions"]

#: §6.1: "we thus recommend to use 17 integration points, an integration domain of [−5, 5],
#: and 1024 slices as starting points". Their algorithm 1 listing defaults to 256 slices; the
#: recommendation is the later, ablated number (their table 1a) and is what we take.
DEFAULT_SLICES = 1024
DEFAULT_QUAD_POINTS = 17
DEFAULT_DOMAIN = 5.0


def sketch_directions(dim: int, slices: int, *, seed: int) -> torch.Tensor:
    """``(dim, slices)`` unit-norm columns — directions drawn uniformly on the sphere.

    Resampled every step by the caller, which is the paper's design rather than an accident:
    their figure 7 shows a resampled set of ``M`` directions beating a *fixed* set by roughly
    an order of magnitude on the end-of-training statistic, because the directions accumulate
    coverage over steps and only have to cover locally at each one.
    """
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(dim, slices, generator=generator)
    return a / a.norm(p=2, dim=0, keepdim=True)


def sigreg(
    embeddings: torch.Tensor,
    *,
    seed: int,
    slices: int = DEFAULT_SLICES,
    quad_points: int = DEFAULT_QUAD_POINTS,
    domain: float = DEFAULT_DOMAIN,
) -> torch.Tensor:
    """The SIGReg penalty on an ``(N, D)`` embedding batch — a differentiable scalar.

    Zero is unreachable: the finite-batch estimate carries a positive bias (the paper's
    thm. 6), which *measured here* puts the floor at ≈ 1.0 for a true N(0, 1) batch — 1.065 at
    n=32, 0.934 at n=512, so it is a near-constant floor rather than one that shrinks away.
    Read differences between arms, never the absolute value. For scale: the same batch scaled
    ×4 — roughly where this project's embeddings sit — scores ≈ 26.
    """
    if embeddings.dim() != 2:
        raise ValueError(f"expected (N, D) embeddings, got shape {tuple(embeddings.shape)}")
    # float32 regardless of any enclosing autocast, and autocast *disabled* across the body
    # rather than only cast at the door: `.float()` alone does not hold, because autocast
    # intercepts the projection matmul and casts its inputs back down. Under bf16 the products
    # `t * z` reach ~60 with only ~3 significant decimal digits, and cos/sin of that is noise —
    # the penalty would still be finite and still descend, which is the dangerous kind of wrong.
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        z = embeddings.float()
        n, d = z.shape

        a = sketch_directions(d, slices, seed=seed).to(z.device)
        t = torch.linspace(-domain, domain, quad_points, device=z.device)
        phi = torch.exp(-0.5 * t.square())  # the N(0,1) CF, and the window w(t), at once

        zt = (z @ a).unsqueeze(2) * t  # (N, slices, quad_points)
        real = zt.cos().mean(0)  # ℜ φ̂(t) per slice
        imag = zt.sin().mean(0)  # ℑ φ̂(t); the target CF is real, so this is pure error
        err = ((real - phi).square() + imag.square()) * phi
        per_slice = torch.trapezoid(err, t, dim=1) * n
        return per_slice.mean()
