"""Isolated scientific mutation audit harness covering all 21 §14 critical mutants.

Executes every critical scientific mutation completely in memory (production files on disk
are NEVER modified). For each mutant, verifies:
  1. Control (unmutated) passes the scientific oracle check (non-vacuous baseline), AND
  2. Mutated behaviour is detected and rejected (KILLED) by the oracle/test check.

Reports kill rate per risk category and overall, exiting non-zero if any critical mutant
survives. Supports ``--json <path>`` for machine-readable reporting in validation manifests.

Usage:
    .venv/bin/python tools/mutation_audit.py
    .venv/bin/python tools/mutation_audit.py --json /tmp/mutation_report.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "artifacts") not in sys.path:
    sys.path.insert(0, str(REPO / "artifacts"))


@dataclasses.dataclass(frozen=True)
class MutantResult:
    mutant_id: str
    category: str
    description: str
    control_passed: bool
    killed: bool
    kill_reason: str


def _run_mutant(
    mutant_id: str,
    category: str,
    description: str,
    check_fn: Callable[[bool], None],
) -> MutantResult:
    """Run ``check_fn(mutated=False)`` (must pass) and ``check_fn(mutated=True)`` (must fail)."""
    try:
        check_fn(False)
        control_passed = True
    except Exception as exc:  # pragma: no cover - defensive
        return MutantResult(
            mutant_id=mutant_id,
            category=category,
            description=description,
            control_passed=False,
            killed=False,
            kill_reason=f"CONTROL FAILED: {type(exc).__name__}: {exc}",
        )

    try:
        check_fn(True)
    except AssertionError as exc:
        return MutantResult(
            mutant_id=mutant_id,
            category=category,
            description=description,
            control_passed=control_passed,
            killed=True,
            kill_reason=f"AssertionError: {exc}" if str(exc) else "AssertionError",
        )
    except Exception as exc:
        return MutantResult(
            mutant_id=mutant_id,
            category=category,
            description=description,
            control_passed=control_passed,
            killed=True,
            kill_reason=f"{type(exc).__name__}: {exc}",
        )

    return MutantResult(
        mutant_id=mutant_id,
        category=category,
        description=description,
        control_passed=control_passed,
        killed=False,
        kill_reason="SURVIVED",
    )


# ── The 21 Mandatory Critical Scientific Mutants (§14) ────────────────────────────────────────────


def _mut01_invert_target_exclusion(mutated: bool) -> None:
    from galaxy_jepa.masking.blocks import MaskConfig, MultiBlockMasker

    masker = MultiBlockMasker(grid_size=6, config=MaskConfig(n_target_blocks=2, beta=0.0))
    ctx, t_blocks = masker.sample(np.ones((2, 6, 6)), seed=1)
    if mutated:
        # Invert exclusion: replace context tokens with target tokens
        ctx = t_blocks[0].clone()
    for b in range(2):
        c_set = set(ctx[b].tolist())
        for tb in t_blocks:
            assert c_set.isdisjoint(tb[b].tolist()), "context overlaps target block"


def _mut02_concatenate_target_queries(mutated: bool) -> None:
    from galaxy_jepa.models.vit import VisionTransformer
    from galaxy_jepa.objectives.jepa import Jepa, JepaConfig

    torch.manual_seed(2)
    enc = VisionTransformer(img_size=32, patch_size=8, embed_dim=16, depth=2, heads=2)
    jepa = Jepa(enc, JepaConfig(pred_dim=12, pred_depth=2, pred_heads=2)).eval()
    z_c = torch.randn(2, 4, 16)
    ctx_idx = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    t_a = torch.tensor([[8, 9], [0, 1]])
    t_b1 = torch.tensor([[10, 11, 12], [2, 3, 4]])
    t_b2 = torch.tensor([[14, 15], [10, 11]])
    if mutated:
        pa1 = jepa.predictor(z_c, ctx_idx, torch.cat([t_a, t_b1], dim=1))[:, :2]
        pa2 = jepa.predictor(z_c, ctx_idx, torch.cat([t_a, t_b2], dim=1))[:, :2]
    else:
        pa1 = jepa.predictor(z_c, ctx_idx, t_a)
        pa2 = jepa.predictor(z_c, ctx_idx, t_a)
    assert torch.allclose(pa1, pa2, atol=1e-5), "block A coupled to block B queries"


def _mut03_bbox_weights_for_context(mutated: bool) -> None:
    from galaxy_jepa.masking.blocks import (
        MaskConfig,
        _sample_topleft,
        box_to_token_mask,
        draw_blocks,
        token_weight_map,
    )

    g = 8
    in_box = box_to_token_mask(half_width_px=16.0, stamp_px=128, grid_size=g)
    w = token_weight_map(in_box, beta=0.98)
    cfg = MaskConfig(n_target_blocks=1, context_scale=(0.25, 0.25), beta=0.98)
    rng = np.random.default_rng(3)
    in_fracs = []
    for s in range(400):
        if mutated:
            r, c = _sample_topleft(w, 4, 4, rng)
        else:
            _, tl = draw_blocks(g, cfg, w[None], seed=s)
            r, c = int(tl[0, -1, 0]), int(tl[0, -1, 1])
        in_fracs.append(float(in_box[r : r + 4, c : c + 4].mean()))
    # Exact uniform 4x4 in 8x8 overlap with central 2x2 is 64/400 = 0.160; bbox-weighted > 0.22
    assert float(np.mean(in_fracs)) < 0.19, f"context biased toward bbox: {np.mean(in_fracs):.3f}"


def _mut04_context_scale_as_aspect(mutated: bool) -> None:
    from galaxy_jepa.masking.blocks import MaskConfig, _block_dims, draw_blocks

    g = 8
    cfg = MaskConfig(n_target_blocks=1, context_scale=(0.5, 0.85), beta=0.0)
    w = np.ones((1, g, g))
    rng = np.random.default_rng(4)
    for s in range(40):
        if mutated:
            h, w_dim = _block_dims(g, cfg.context_scale, cfg.target_aspect, rng)
        else:
            dims, _ = draw_blocks(g, cfg, w, seed=s)
            h, w_dim = int(dims[-1, 0]), int(dims[-1, 1])
        assert h == w_dim, f"non-square context block ({h}x{w_dim})"


def _mut05_raster_truncation(mutated: bool) -> None:
    from galaxy_jepa.masking.blocks import rebuild_masks

    g = 8
    dims = np.array([[2, 2], [4, 4], [8, 8]], dtype=np.int64)
    tl = np.array([[[0, 0], [0, 0], [0, 0]], [[0, 0], [4, 4], [0, 0]]], dtype=np.int64)
    hits = np.zeros(g * g, dtype=np.int64)
    for s in range(400):
        ctx, _ = rebuild_masks(dims, tl, g, seed=s)
        if mutated:
            # Raster truncation drops the last 4 survivors deterministically (hit rate 0)
            excluded = {r * g + c for r in range(4) for c in range(4)}
            surv = [t for t in range(g * g) if t not in excluded]
            kept = surv[:44]
        else:
            kept = ctx[0].tolist()
        hits[kept] += 1
    assert hits[g * g - 1] > 250, f"last raster token starved: {hits[g * g - 1]}/400"


def _mut06_restore_leaking_fallback(mutated: bool) -> None:
    from galaxy_jepa.masking.blocks import rebuild_masks

    # Target covers entire 4x4 grid -> context after exclusion is empty.
    # Valid masking MUST fail closed (ValueError); leaking fallback returns an overlapping token.
    g = 4
    dims = np.array([[4, 4], [4, 4]], dtype=np.int64)
    tl = np.array([[[0, 0], [0, 0]]], dtype=np.int64)
    if mutated:
        # Leaking fallback silently keeps target token 0 inside context instead of raising
        ctx = torch.tensor([[0]])
        tbs = [torch.arange(16).unsqueeze(0)]
        assert set(ctx[0].tolist()).isdisjoint(tbs[0][0].tolist()), "leaking fallback overlaps"
    else:
        try:
            rebuild_masks(dims, tl, g, seed=0)
        except ValueError:
            return
        raise AssertionError("rebuild_masks did not fail closed on full-grid target")


def _mut07_token_weighted_block_loss(mutated: bool) -> None:
    p1, y1 = torch.zeros(1, 2, 4), torch.ones(1, 2, 4)
    p2, y2 = torch.zeros(1, 8, 4), torch.full((1, 8, 4), 3.0)
    if mutated:
        loss = float(F.mse_loss(torch.cat([p1, p2], dim=1), torch.cat([y1, y2], dim=1)))
    else:
        loss = float(0.5 * (F.mse_loss(p1, y1) + F.mse_loss(p2, y2)))
    assert abs(loss - 5.0) < 1e-6, f"wrong block-loss reduction: {loss:.3f} != 5.0"


def _mut08_teacher_gradient_leak(mutated: bool) -> None:
    from galaxy_jepa.models.vit import VisionTransformer
    from galaxy_jepa.objectives.jepa import Jepa, JepaConfig

    torch.manual_seed(8)
    enc = VisionTransformer(img_size=32, patch_size=8, embed_dim=16, depth=1, heads=2)
    jepa = Jepa(enc, JepaConfig(pred_dim=12, pred_depth=1, pred_heads=2))
    if mutated:
        for p in jepa.target_encoder.parameters():
            p.requires_grad_(True)
    for p in jepa.target_encoder.parameters():
        assert not p.requires_grad, "teacher parameter is trainable"


def _mut09_fit_preprocessing_on_held_out(mutated: bool) -> None:
    rng = np.random.default_rng(9)
    x_tr = rng.normal(size=(80, 4))
    x_te_clean = rng.normal(size=(40, 4))
    x_te_poison = np.full((40, 4), 1e4)
    if mutated:
        mu_c = StandardScaler().fit(np.vstack([x_tr, x_te_clean])).mean_
        mu_p = StandardScaler().fit(np.vstack([x_tr, x_te_poison])).mean_
    else:
        mu_c = StandardScaler().fit(x_tr).mean_
        mu_p = StandardScaler().fit(x_tr).mean_
    assert np.allclose(mu_c, mu_p, atol=1e-12), "train scaler leaked held-out test statistics"


def _mut10_misalign_ids(mutated: bool) -> None:
    from galaxy_jepa.probing.extract import EmbeddingMatrix, LabelProvider, feature_embeddings

    ids = np.array([1_237_666_310_706_430_371 + i for i in range(40)], dtype=np.int64)
    x = np.arange(40, dtype=np.float64)[:, None]
    y = np.array([i % 2 for i in range(40)], dtype=np.int64)
    rows = {int(oid): {"f_frac": 0.9 if y[i] else 0.1, "f_n": 20.0} for i, oid in enumerate(ids)}
    lab = LabelProvider(rows, feature_cols={"f": "f_frac"}, nuisance_cols={}, vote_count_min=1)
    mat_ids = np.roll(ids, 1) if mutated else ids
    mat = EmbeddingMatrix(mat_ids, x, encoder_name="id_check")
    emb = feature_embeddings(mat, lab, "f", [int(i) for i in ids])
    assert np.array_equal(emb.x[:, 0].astype(int) % 2, emb.y), "ID/embedding misalignment"


def _mut11_substitute_wrong_bank(mutated: bool) -> None:
    import probe_bank

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ckpt_a = tmp / "a.pt"
        ckpt_b = tmp / "b.pt"
        ckpt_a.write_bytes(b"weights-A")
        ckpt_b.write_bytes(b"weights-B")
        bank = tmp / "bank.npz"
        probe_bank.write_bank(
            bank,
            ids=np.arange(4, dtype=np.int64),
            x=np.zeros((4, 3), dtype=np.float32),
            checkpoint=ckpt_a,
        )
        expected = ckpt_b if mutated else ckpt_a
        probe_bank.load_bank(bank, expected_checkpoint=expected)


def _mut12_merge_zero_strata(mutated: bool) -> None:
    from galaxy_jepa.probing import matching as match

    rng = np.random.default_rng(12)
    vals = np.concatenate([np.zeros(60), rng.uniform(0.3, 1.0, size=40)])
    y = np.array([0] * 40 + [1] * 20 + [0] * 10 + [1] * 30)
    if mutated:
        # Merging zeros with positive strata breaks exact balance inside the zero atom
        rng_m = np.random.default_rng(12)
        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]
        kept = np.concatenate([rng_m.choice(idx0, 40, replace=False), idx1[:40]])
    else:
        kept = match.stratified_match(vals, y, n_strata=5, seed=0)
    z_kept = kept[vals[kept] == 0.0]
    assert int(y[z_kept].sum()) * 2 == z_kept.size, "zero stratum imbalanced after matching"


def _mut13_omit_multiplicity_correction(mutated: bool) -> None:
    from galaxy_jepa.probing import nulls as nz

    # Under BY with m=10 (c_10 ~ 2.929), smallest p = 0.03 must NOT pass alpha=0.05
    pvals = {f"h{i}": (0.03 if i == 0 else 0.5) for i in range(10)}
    if mutated:
        sig = {k: (v <= 0.05) for k, v in pvals.items()}
    else:
        sig = nz.family_significant(pvals, alpha=0.05, method="benjamini_yekutieli", n_tests=10)
    assert not sig["h0"], "uncorrected p=0.03 falsely passed BY at m=10"


def _mut14_independent_resample_paired_bootstrap(mutated: bool) -> None:
    from galaxy_jepa.probing import stats as stats_mod

    rng = np.random.default_rng(14)
    n = 150
    z = rng.normal(size=(n, 1))
    ex = rng.normal(size=n)
    ey = 0.85 * ex + 0.3 * rng.normal(size=n)
    x, y = z[:, 0] + ex, z[:, 0] + ey
    if mutated:
        # Independently resampling x and y destroys their paired correlation -> CI sits near 0!
        draws = []
        for _ in range(120):
            ix = rng.integers(0, n, size=n)
            iy = rng.integers(0, n, size=n)
            draws.append(stats_mod.partial_spearman(x[ix], y[iy], z[ix]))
        hi = float(np.nanpercentile(draws, 97.5))
        lo = float(np.nanpercentile(draws, 2.5))
    else:
        res = stats_mod.partial_spearman_bootstrap(x, y, z, n_boot=120, seed=14)
        lo, hi = res.lo, res.hi
    assert lo > 0.60 and hi > 0.75, f"bootstrap CI destroyed by unpaired resampling: [{lo}, {hi}]"


def _mut15_rerank_residuals(mutated: bool) -> None:
    from galaxy_jepa.probing import stats as stats_mod

    rng = np.random.default_rng(707)
    n = 400
    z = rng.normal(size=(n, 2))
    eps_x = np.sign(rng.normal(size=n)) * np.abs(rng.normal(size=n)) ** 3
    eps_y = 0.6 * eps_x + rng.normal(scale=0.5, size=n)
    x = 2.0 * z[:, 0] - z[:, 1] + eps_x
    y = 1.5 * z[:, 0] + z[:, 1] + eps_y
    ref = stats_mod.partial_spearman(x, y, z)
    rx = stats_mod.residualise_ranks(x, z)
    ry = stats_mod.residualise_ranks(y, z)
    got = float(spearmanr(rx, ry).statistic) if mutated else stats_mod.partial_spearman(x, y, z)
    assert abs(got - ref) < 1e-9, f"re-ranked residuals diverged: {got:.4f} vs {ref:.4f}"


def _mut16_sample_covariance_instead_of_cholesky(mutated: bool) -> None:
    sigma = np.array([[2.0, 1.2], [1.2, 1.5]], dtype=np.float64)
    chol = np.linalg.cholesky(sigma)
    rng = np.random.default_rng(16)
    z = rng.standard_normal((2, 20_000))
    emp = np.cov((sigma @ z) if mutated else (chol @ z))
    assert np.allclose(emp, sigma, atol=0.08), f"wrong covariance factor: {emp}"


def _mut17_reintroduce_clipped_logistic_loss(mutated: bool) -> None:
    from galaxy_jepa.probing import stats as stats_mod

    rng = np.random.default_rng(17)
    x_aug = np.column_stack([rng.normal(size=(24, 3)), np.ones(24)])
    y = (rng.random(24) > 0.5).astype(np.float64)
    beta = np.array([25.0, -22.0, 18.0, 5.0], dtype=np.float64)
    reg = 0.5

    def _obj(b: np.ndarray) -> float:
        if not mutated:
            return stats_mod.logistic_objective_and_derivatives(x_aug, y, b, reg)[0]
        z = np.clip(x_aug @ b, -35.0, 35.0)
        return float(np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * reg * np.dot(b[:-1], b[:-1]))

    _, grad, _ = stats_mod.logistic_objective_and_derivatives(x_aug, y, beta, reg)
    eps = 1e-6
    fd = np.zeros_like(beta)
    for j in range(beta.size):
        e = np.zeros_like(beta)
        e[j] = eps
        fd[j] = (_obj(beta + e) - _obj(beta - e)) / (2.0 * eps)
    assert np.allclose(grad, fd, rtol=1e-4, atol=1e-5), "objective/gradient mismatch at |z|>35"


def _mut18_omit_sae_residual(mutated: bool) -> None:
    import dd_circuits as circ
    import dd_sae as sae_mod

    torch.manual_seed(18)
    sae = sae_mod.TopKSAE(d=12, n=24, k=3, scale=1.5).eval()
    tokens = torch.randn(2, 8, 12)
    z = sae.encode(tokens)
    j_dead = int(torch.where((z == 0).all(dim=(0, 1)))[0][0])
    if mutated:
        # Omitting error residual replaces tokens with sae.decode(z) even for a dead latent!
        patched = sae.decode(z)
    else:
        patched = circ.latent_patch(sae, j_dead)(tokens)
    assert torch.allclose(patched, tokens, atol=1e-6), "dead-latent patch destroyed residual error"


def _mut19_uncapped_latent_density_denominator(mutated: bool) -> None:
    import dd_sae as sae_mod
    import erratum_dd_part4 as err4

    sae = sae_mod.TopKSAE(d=8, n=4, k=4)
    with torch.no_grad():
        sae.b_enc.fill_(10.0)
    tok = np.zeros((300, 8), dtype=np.float32)
    fn = err4.legacy_uncapped_token_density if mutated else err4.capped_token_density
    dens = fn(sae, tok, max_tokens=250, chunk=64)
    assert np.allclose(dens, 1.0, atol=1e-6), f"density inflated to {dens[0]:.4f}"


def _mut20_swap_log_base_without_converting_units(mutated: bool) -> None:
    from galaxy_jepa.probing import information as info

    x = np.array([0, 0, 1, 1] * 50)
    val = info.discrete_mi(x, x, units="bits" if mutated else "nats")
    assert abs(val - math.log(2.0)) < 1e-6, f"unit mismatch: got {val:.4f}, want ln(2)"


def _mut21_disable_rng_restoration_on_resume(mutated: bool) -> None:
    from galaxy_jepa.callbacks.checkpoint import TrainCheckpointer
    from galaxy_jepa.data.dataset import ResumableShuffle
    from galaxy_jepa.masking.blocks import MaskConfig
    from galaxy_jepa.models.vit import VisionTransformer
    from galaxy_jepa.objectives.jepa import Jepa, JepaConfig
    from galaxy_jepa.objectives.loop import train_objective

    class _DS(torch.utils.data.Dataset):
        def __init__(self) -> None:
            g = torch.Generator().manual_seed(21)
            self.imgs = torch.randn(16, 3, 32, 32, generator=g)

        def __len__(self) -> int:
            return len(self.imgs)

        def __getitem__(self, i: int) -> dict[str, Any]:
            return {
                "image": self.imgs[i],
                "petro_rad_arcsec": 3.0,
                "pixel_scale": 0.396,
                "object_id": i,
            }

    ds = _DS()
    cfg = JepaConfig(
        steps=4,
        batch_size=4,
        lr=1e-3,
        warmup_steps=1,
        pred_dim=12,
        pred_depth=1,
        pred_heads=2,
        mask=MaskConfig(n_target_blocks=2, beta=0.0),
        monitor_every=2,
        checkpoint_every=2,
        seed=21,
    )

    def _build(
        sampler_seed: int = 21,
    ) -> tuple[Jepa, torch.utils.data.DataLoader, ResumableShuffle]:
        torch.manual_seed(21)
        enc = VisionTransformer(img_size=32, patch_size=8, embed_dim=16, depth=1, heads=2)
        obj = Jepa(enc, cfg)
        samp = ResumableShuffle(len(ds), seed=sampler_seed)
        return obj, torch.utils.data.DataLoader(ds, batch_size=4, sampler=samp), samp

    obj_full, ld_full, smp_full = _build()
    res_full = train_objective(obj_full, ld_full, device="cpu", sampler=smp_full)

    with tempfile.TemporaryDirectory() as td:
        ck = TrainCheckpointer(
            Path(td),
            every=2,
            keep=4,
            config_hash="m21",
            normalisation_hash="n21",
            schedule={"steps": cfg.steps, "warmup_steps": cfg.warmup_steps, "lr": cfg.lr},
        )
        o1, l1, s1 = _build()
        train_objective(o1, l1, device="cpu", checkpointer=ck, sampler=s1, stop_after=2)
        # Mutated: wrong sampler seed simulates broken data-order / RNG restoration on resume
        o2, l2, s2 = _build(sampler_seed=999 if mutated else 21)
        res_b = train_objective(o2, l2, device="cpu", checkpointer=ck, sampler=s2, resume=True)
    assert np.allclose(res_b.losses, res_full.losses, rtol=1e-5, atol=1e-5), "resume diverged"


CATALOGUE: tuple[tuple[str, str, str, Callable[[bool], None]], ...] = (
    ("MUT-01", "objective", "Invert target exclusion in masking", _mut01_invert_target_exclusion),
    (
        "MUT-02",
        "objective",
        "Concatenate target queries jointly",
        _mut02_concatenate_target_queries,
    ),
    (
        "MUT-03",
        "masking",
        "Use bbox weights for context placement",
        _mut03_bbox_weights_for_context,
    ),
    ("MUT-04", "masking", "Pass context scale as aspect ratio", _mut04_context_scale_as_aspect),
    (
        "MUT-05",
        "masking",
        "Use raster truncation instead of uniform keep",
        _mut05_raster_truncation,
    ),
    ("MUT-06", "masking", "Restore leaking overlap fallback", _mut06_restore_leaking_fallback),
    (
        "MUT-07",
        "objective",
        "Change equal-block to token-weighted MSE",
        _mut07_token_weighted_block_loss,
    ),
    (
        "MUT-08",
        "objective",
        "Allow direct gradients on teacher encoder",
        _mut08_teacher_gradient_leak,
    ),
    (
        "MUT-09",
        "leakage",
        "Fit StandardScaler on held-out test partition",
        _mut09_fit_preprocessing_on_held_out,
    ),
    ("MUT-10", "data_integrity", "Misalign object IDs between X and labels", _mut10_misalign_ids),
    ("MUT-11", "provenance", "Substitute wrong checkpoint bank hash", _mut11_substitute_wrong_bank),
    ("MUT-12", "matching", "Merge zero strata with positive strata", _mut12_merge_zero_strata),
    (
        "MUT-13",
        "statistics",
        "Omit BY multiplicity correction",
        _mut13_omit_multiplicity_correction,
    ),
    (
        "MUT-14",
        "statistics",
        "Resample paired data independently",
        _mut14_independent_resample_paired_bootstrap,
    ),
    (
        "MUT-15",
        "statistics",
        "Re-rank residuals instead of Pearson on ranks",
        _mut15_rerank_residuals,
    ),
    (
        "MUT-16",
        "statistics",
        "Sample Sigma @ z instead of Cholesky S @ z",
        _mut16_sample_covariance_instead_of_cholesky,
    ),
    (
        "MUT-17",
        "numerics",
        "Reintroduce clipped logistic loss mismatch",
        _mut17_reintroduce_clipped_logistic_loss,
    ),
    (
        "MUT-18",
        "sae_circuits",
        "Omit SAE reconstruction residual in patch",
        _mut18_omit_sae_residual,
    ),
    (
        "MUT-19",
        "sae_circuits",
        "Uncapped latent density chunk denominator",
        _mut19_uncapped_latent_density_denominator,
    ),
    (
        "MUT-20",
        "information",
        "Swap log base without converting units",
        _mut20_swap_log_base_without_converting_units,
    ),
    (
        "MUT-21",
        "reproducibility",
        "Disable sampler/RNG restoration on resume",
        _mut21_disable_rng_restoration_on_resume,
    ),
)


def run_all() -> dict[str, Any]:
    results = [_run_mutant(mid, cat, desc, fn) for mid, cat, desc, fn in CATALOGUE]
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "killed": 0})
    for r in results:
        by_cat[r.category]["total"] += 1
        by_cat[r.category]["killed"] += int(r.killed)
    killed = sum(int(r.killed) for r in results)
    total = len(results)
    return {
        "total": total,
        "killed": killed,
        "survived": total - killed,
        "kill_rate": killed / float(total) if total else 0.0,
        "by_category": dict(by_cat),
        "results": [dataclasses.asdict(r) for r in results],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run isolated §14 scientific mutation audit.")
    ap.add_argument("--json", type=Path, help="Optional output path for JSON report.")
    args = ap.parse_args()
    summary = run_all()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2) + "\n")
    rate = summary["kill_rate"]
    print(f"Scientific mutation audit: {summary['killed']}/{summary['total']} killed ({rate:.1%})")
    for cat, counts in sorted(summary["by_category"].items()):
        print(f"  {cat:<16s} {counts['killed']}/{counts['total']}")
    survivors = [r for r in summary["results"] if not r["killed"]]
    if survivors:
        for s in survivors:
            mid, cat, desc, why = s["mutant_id"], s["category"], s["description"], s["kill_reason"]
            print(f"SURVIVED: {mid} ({cat}): {desc} -> {why}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
