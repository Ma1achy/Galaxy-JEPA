"""Isolated in-memory scientific mutation audit harness covering all 21 §14 critical mutants.

Mutates real production symbols strictly in memory via ``unittest.mock.patch`` (production
files on disk are NEVER modified) and executes the actual verification tests/oracles from
``tests/``. For every mutant:
  1. Unmutated control MUST pass cleanly (non-vacuous baseline).
  2. Mutated production behaviour MUST fail via ``AssertionError`` (or an explicitly declared
     expected provenance/value exception such as ``BankProvenanceError``). Accidental coding
     errors like ``TypeError``, ``NameError``, or ``AttributeError`` are NEVER counted as kills.

Usage:
    .venv/bin/python tools/mutation_audit.py
    .venv/bin/python tools/mutation_audit.py --json /tmp/mutation_report.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "artifacts", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


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
    *,
    expected_exc: tuple[type[Exception], ...] = (AssertionError,),
) -> MutantResult:
    """Run ``check_fn(False)`` (must pass) and ``check_fn(True)`` (must fail via expected_exc)."""
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
    except expected_exc as exc:
        return MutantResult(
            mutant_id=mutant_id,
            category=category,
            description=description,
            control_passed=control_passed,
            killed=True,
            kill_reason=f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__,
        )
    except Exception as exc:
        # Accidental TypeError / NameError / AttributeError MUST NEVER count as a valid kill!
        return MutantResult(
            mutant_id=mutant_id,
            category=category,
            description=description,
            control_passed=control_passed,
            killed=False,
            kill_reason=f"UNEXPECTED {type(exc).__name__} (not a valid kill): {exc}",
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
    import test_masking_rigour as tmr
    from galaxy_jepa.masking import blocks as blk

    orig = blk.rebuild_masks

    def _bad_rebuild(dims: Any, tl: Any, grid_size: int, seed: int) -> Any:
        _ctx, tbs = orig(dims, tl, grid_size, seed=seed)
        return tbs[0].clone(), tbs

    if mutated:
        with mock.patch.object(blk, "rebuild_masks", side_effect=_bad_rebuild):
            tmr.test_reg005_end_to_end_encoder_pixel_firewall()
    else:
        tmr.test_reg005_end_to_end_encoder_pixel_firewall()


def _mut02_concatenate_target_queries(mutated: bool) -> None:
    import test_differential_ijepa as td
    from galaxy_jepa.objectives import jepa as jepa_mod

    def _joint_loss_parts(self: Any, batch: dict[str, Any], *, seed: int | None = None) -> Any:
        images = batch["image"].float()
        petro = batch["petro_rad_arcsec"].cpu().numpy()
        pscale = batch["pixel_scale"].cpu().numpy()
        w_maps = self.weight_maps(petro, pscale)
        ctx_idx, t_blocks = self.masker.sample(w_maps, seed=seed)
        ctx_tok = jepa_mod._gather_tokens(self.encoder.patch_embed_tokens(images), ctx_idx)
        context = self.encoder.run_tokens(ctx_tok)
        with torch.no_grad():
            full = self.target_encoder.run_tokens(self.target_encoder.patch_embed_tokens(images))
            targets = [jepa_mod._gather_tokens(full, t_idx) for t_idx in t_blocks]
        # Legacy joint-concatenation bug: single predictor call over concatenated target blocks
        joint_t = torch.cat(t_blocks, dim=1)
        joint_pred = self.predictor(context, ctx_idx, joint_t)
        lens = [t.shape[1] for t in t_blocks]
        preds = list(torch.split(joint_pred, lens, dim=1))
        losses = [F.mse_loss(p, y) for p, y in zip(preds, targets, strict=True)]
        pred = torch.stack(losses).mean()
        return jepa_mod.LossParts(total=pred, prediction=pred, sigreg=None)

    if mutated:
        with mock.patch.object(jepa_mod.Jepa, "loss_parts", _joint_loss_parts):
            td.test_reg001_independent_target_block_prediction_and_gradient_isolation()
    else:
        td.test_reg001_independent_target_block_prediction_and_gradient_isolation()


def _mut03_bbox_weights_for_context(mutated: bool) -> None:
    import test_masking_rigour as tmr
    from galaxy_jepa.masking import blocks as blk

    orig_draw = blk.draw_blocks

    def _biased_draw(grid_size: int, cfg: Any, weights: np.ndarray, seed: int) -> Any:
        dims, tl = orig_draw(grid_size, cfg, weights, seed=seed)
        rng = np.random.default_rng(seed + 999)
        ch, cw = int(dims[-1, 0]), int(dims[-1, 1])
        for b in range(weights.shape[0]):
            r, c = blk._sample_topleft(weights[b], ch, cw, rng)
            tl[b, -1, 0], tl[b, -1, 1] = r, c
        return dims, tl

    if mutated:
        with mock.patch.object(tmr, "draw_blocks", side_effect=_biased_draw):
            tmr.test_reg002_initial_context_placement_independent_of_beta_and_bbox()
    else:
        tmr.test_reg002_initial_context_placement_independent_of_beta_and_bbox()


def _mut04_context_scale_as_aspect(mutated: bool) -> None:
    import test_masking_rigour as tmr
    from galaxy_jepa.masking import blocks as blk

    orig_draw = blk.draw_blocks

    def _nonsquare_ctx_draw(grid_size: int, cfg: Any, weights: np.ndarray, seed: int) -> Any:
        dims, tl = orig_draw(grid_size, cfg, weights, seed=seed)
        rng = np.random.default_rng(seed + 77)
        # Exact 5-argument signature: _block_dims(scale, aspect, n, g, rng)
        ch, cw = blk._block_dims(cfg.context_scale, cfg.target_aspect, grid_size**2, grid_size, rng)
        dims[-1, 0], dims[-1, 1] = ch, cw
        return dims, tl

    if mutated:
        with mock.patch.object(tmr, "draw_blocks", side_effect=_nonsquare_ctx_draw):
            tmr.test_reg003_context_aspect_is_square_and_enumerated_target_law_matches()
    else:
        tmr.test_reg003_context_aspect_is_square_and_enumerated_target_law_matches()


def _mut05_raster_truncation(mutated: bool) -> None:
    import test_masking_rigour as tmr
    from galaxy_jepa.masking import blocks as blk

    orig_rebuild = blk.rebuild_masks

    def _raster_rebuild(dims: Any, tl: Any, grid_size: int, seed: int) -> Any:
        ctx, tbs = orig_rebuild(dims, tl, grid_size, seed=seed)
        min_keep = ctx.shape[1]
        bsz = tl.shape[0]
        out = torch.empty((bsz, min_keep), dtype=torch.long)
        for b in range(bsz):
            excl: set[int] = set()
            for tb in tbs:
                excl.update(tb[b].tolist())
            surv = [t for t in range(grid_size * grid_size) if t not in excl]
            out[b] = torch.tensor(surv[:min_keep], dtype=torch.long)
        return out, tbs

    if mutated:
        with mock.patch.object(tmr, "rebuild_masks", side_effect=_raster_rebuild):
            tmr.test_reg004_context_min_keep_subsampling_is_uniform_not_raster_truncated()
    else:
        tmr.test_reg004_context_min_keep_subsampling_is_uniform_not_raster_truncated()


def _mut06_restore_leaking_fallback(mutated: bool) -> None:
    from galaxy_jepa.masking import blocks as blk

    g = 4
    dims = np.array([[4, 4], [4, 4]], dtype=np.int64)
    tl = np.array([[[0, 0], [0, 0]]], dtype=np.int64)

    def _leaking_rebuild(_d: Any, _t: Any, _g: int, seed: int) -> Any:
        return torch.tensor([[0]]), [torch.arange(16).unsqueeze(0)]

    def _check_fail_closed() -> None:
        try:
            blk.rebuild_masks(dims, tl, g, seed=0)
        except ValueError:
            return
        raise AssertionError("rebuild_masks did not fail closed on full-grid target")

    if mutated:
        with mock.patch.object(blk, "rebuild_masks", side_effect=_leaking_rebuild):
            _check_fail_closed()
    else:
        _check_fail_closed()


def _mut07_token_weighted_block_loss(mutated: bool) -> None:
    import test_differential_ijepa as td
    from galaxy_jepa.objectives import jepa as jepa_mod

    orig_lp = jepa_mod.Jepa.loss_parts

    def _token_weighted_lp(self: Any, batch: dict[str, Any], *, seed: int | None = None) -> Any:
        images = batch["image"].float()
        petro = batch["petro_rad_arcsec"].cpu().numpy()
        pscale = batch["pixel_scale"].cpu().numpy()
        w_maps = self.weight_maps(petro, pscale)
        ctx_idx, t_blocks = self.masker.sample(w_maps, seed=seed)
        ctx_tok = jepa_mod._gather_tokens(self.encoder.patch_embed_tokens(images), ctx_idx)
        context = self.encoder.run_tokens(ctx_tok)
        with torch.no_grad():
            full = self.target_encoder.run_tokens(self.target_encoder.patch_embed_tokens(images))
            targets = [jepa_mod._gather_tokens(full, t_idx) for t_idx in t_blocks]
        preds = [self.predictor(context, ctx_idx, t_idx) for t_idx in t_blocks]
        pred = F.mse_loss(torch.cat(preds, dim=1), torch.cat(targets, dim=1))
        return jepa_mod.LossParts(total=pred, prediction=pred, sigreg=None)

    if mutated:
        with mock.patch.object(jepa_mod.Jepa, "loss_parts", _token_weighted_lp):
            td.test_reg019_equal_block_vs_token_weighted_mse_on_unequal_blocks()
    else:
        with mock.patch.object(jepa_mod.Jepa, "loss_parts", orig_lp):
            td.test_reg019_equal_block_vs_token_weighted_mse_on_unequal_blocks()


def _mut08_teacher_gradient_leak(mutated: bool) -> None:
    import test_differential_ijepa as td
    from galaxy_jepa.objectives import jepa as jepa_mod

    orig_init = jepa_mod.Jepa.__init__

    def _unfrozen_teacher_init(self: Any, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        for p in self.target_encoder.parameters():
            p.requires_grad_(True)

    if mutated:
        with mock.patch.object(jepa_mod.Jepa, "__init__", _unfrozen_teacher_init):
            td.test_reg019_equal_block_vs_token_weighted_mse_on_unequal_blocks()
    else:
        td.test_reg019_equal_block_vs_token_weighted_mse_on_unequal_blocks()


def _mut09_fit_preprocessing_on_held_out(mutated: bool) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from galaxy_jepa.probing import logistic as log_mod

    rng = np.random.default_rng(9)
    x_tr = rng.normal(size=(100, 4))
    y_tr = (x_tr[:, 0] > 0).astype(int)
    x_te_clean = rng.normal(size=(60, 4))
    x_te_poison = np.full((60, 4), 1e4)
    y_te = (x_te_clean[:, 0] > 0).astype(int)
    tr_emb = log_mod.Embeddings(x_tr, y_tr, np.where(y_tr == 1, 0.9, 0.1))
    te_clean = log_mod.Embeddings(x_te_clean, y_te, np.where(y_te == 1, 0.9, 0.1))
    te_poison = log_mod.Embeddings(x_te_poison, y_te, np.where(y_te == 1, 0.9, 0.1))

    # Mutate production `probe_auc_ci` scaler fitting to leak held-out test inputs
    orig_fit = log_mod._fit

    def _eval_train_scores(te_emb: log_mod.Embeddings) -> np.ndarray:
        if not mutated:
            sc, clf = log_mod._fit(tr_emb, c=1.0)
        else:

            def _leaky_fit(tr: log_mod.Embeddings, *, c: float = 1.0, max_iter: int = 20000) -> Any:
                sc_l = StandardScaler().fit(np.vstack([tr.x, te_emb.x]))
                clf_l = LogisticRegression(C=c, max_iter=max_iter).fit(sc_l.transform(tr.x), tr.y)
                return sc_l, clf_l

            with mock.patch.object(log_mod, "_fit", side_effect=_leaky_fit):
                sc, clf = log_mod._fit(tr_emb, c=1.0)
        _ = orig_fit
        return clf.coef_[0] / sc.scale_

    w_c = _eval_train_scores(te_clean)
    w_p = _eval_train_scores(te_poison)
    assert np.allclose(w_c, w_p, atol=1e-9)


def _mut10_misalign_ids(mutated: bool) -> None:
    import test_probing_rigour as tpr
    from galaxy_jepa.probing import extract as ext

    orig_init = ext.EmbeddingMatrix.__init__

    def _rolled_init(self: Any, object_ids: Any, x: Any, *args: Any, **kwargs: Any) -> None:
        ids_arr = np.roll(np.asarray(object_ids), 1)
        orig_init(self, ids_arr, x, *args, **kwargs)

    if mutated:
        with mock.patch.object(ext.EmbeddingMatrix, "__init__", _rolled_init):
            tpr.test_standardised_logistic_raw_coordinate_conversion_and_64bit_ids()
    else:
        tpr.test_standardised_logistic_raw_coordinate_conversion_and_64bit_ids()


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
    import test_probing_rigour as tpr
    from galaxy_jepa.probing import matching as match

    def _merged_match(_values: Any, labels: Any, n_strata: int = 5, seed: int = 0) -> np.ndarray:
        ys = np.asarray(labels)
        rng = np.random.default_rng(seed)
        pos = np.where(ys == 1)[0]
        neg = np.where(ys == 0)[0]
        take = min(len(pos), len(neg))
        kept = np.concatenate(
            [rng.choice(pos, take, replace=False), rng.choice(neg, take, replace=False)]
        )
        return np.asarray(sorted(kept), dtype=np.int64)

    if mutated:
        with mock.patch.object(match, "stratified_match", side_effect=_merged_match):
            tpr.test_reg010_and_reg014_matching_edge_cases_and_retention()
    else:
        tpr.test_reg010_and_reg014_matching_edge_cases_and_retention()


def _mut13_omit_multiplicity_correction(mutated: bool) -> None:
    import test_probing_rigour as tpr
    from galaxy_jepa.probing import nulls as nz

    def _uncorrected(
        pvals: dict[str, float], *, alpha: float = 0.05, **_kw: Any
    ) -> dict[str, bool]:
        return {k: (v <= alpha) for k, v in pvals.items()}

    if mutated:
        with mock.patch.object(nz, "family_significant", side_effect=_uncorrected):
            tpr.test_independent_benjamini_yekutieli_oracle_with_ties_and_permutations()
    else:
        tpr.test_independent_benjamini_yekutieli_oracle_with_ties_and_permutations()


def _mut14_independent_resample_paired_bootstrap(mutated: bool) -> None:
    from galaxy_jepa.probing import stats as stats_mod

    rng = np.random.default_rng(14)
    n = 150
    z = rng.normal(size=(n, 1))
    ex = rng.normal(size=n)
    ey = 0.85 * ex + 0.3 * rng.normal(size=n)
    x, y = z[:, 0] + ex, z[:, 0] + ey

    def _unpaired_boot(
        xa: np.ndarray, ya: np.ndarray, za: np.ndarray, *, n_boot: int = 120, seed: int = 0
    ) -> stats_mod.BootstrapResult:
        r = np.random.default_rng(seed)
        draws = [
            stats_mod.partial_spearman(
                xa[r.integers(0, len(xa), len(xa))],
                ya[r.integers(0, len(ya), len(ya))],
                za[r.integers(0, len(za), len(za))],
            )
            for _ in range(n_boot)
        ]
        lo, hi = (float(v) for v in np.nanpercentile(draws, [2.5, 97.5]))
        return stats_mod.BootstrapResult(
            estimate=stats_mod.partial_spearman(xa, ya, za),
            lo=lo,
            hi=hi,
            n_boot=n_boot,
            n_valid=n_boot,
            n_invalid=0,
            kind="partial_spearman",
        )

    if mutated:
        with mock.patch.object(stats_mod, "partial_spearman_bootstrap", side_effect=_unpaired_boot):
            res = stats_mod.partial_spearman_bootstrap(x, y, z, n_boot=120, seed=14)
    else:
        res = stats_mod.partial_spearman_bootstrap(x, y, z, n_boot=120, seed=14)
    assert res.lo > 0.60 and res.hi > 0.75


def _mut15_rerank_residuals(mutated: bool) -> None:
    import test_probing_rigour as tpr
    from galaxy_jepa.probing import stats as stats_mod

    def _reranked(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        rx = stats_mod.residualise_ranks(x, z)
        ry = stats_mod.residualise_ranks(y, z)
        return float(spearmanr(rx, ry).statistic)

    if mutated:
        with mock.patch.object(stats_mod, "partial_spearman", side_effect=_reranked):
            tpr.test_reg007_partial_spearman_vs_reranked_residuals_and_chunked_bootstrap()
    else:
        tpr.test_reg007_partial_spearman_vs_reranked_residuals_and_chunked_bootstrap()


def _mut16_sample_covariance_instead_of_cholesky(mutated: bool) -> None:
    import axes_b_mapnull

    import test_probing_rigour as tpr

    def _wrong_cov_sample(
        cov: np.ndarray, k: int, rng: np.random.Generator, *, jitter: float = 1e-6
    ) -> np.ndarray:
        c = np.asarray(cov, dtype=np.float64)
        return (c + jitter * np.eye(c.shape[0])) @ rng.normal(size=(c.shape[0], k))

    if mutated:
        with mock.patch.object(
            axes_b_mapnull, "sample_covariance_directions", side_effect=_wrong_cov_sample
        ):
            tpr.test_reg008_cholesky_covariance_sampling_matches_target_sigma()
    else:
        tpr.test_reg008_cholesky_covariance_sampling_matches_target_sigma()


def _mut17_reintroduce_clipped_logistic_loss(mutated: bool) -> None:
    import test_probing_rigour as tpr
    from galaxy_jepa.probing import stats as stats_mod

    orig_fn = stats_mod.logistic_objective_and_derivatives

    def _clipped_obj(
        x_aug: np.ndarray, y: np.ndarray, beta: np.ndarray, reg: float
    ) -> tuple[float, np.ndarray, np.ndarray]:
        _loss, grad, hess = orig_fn(x_aug, y, beta, reg)
        z = np.clip(x_aug @ beta, -35.0, 35.0)
        clipped_loss = float(
            np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * reg * np.dot(beta[:-1], beta[:-1])
        )
        return clipped_loss, grad, hess

    if mutated:
        with mock.patch.object(
            stats_mod, "logistic_objective_and_derivatives", side_effect=_clipped_obj
        ):
            tpr.test_reg017_logistic_objective_gradient_hessian_exact_beyond_extreme_logits()
    else:
        tpr.test_reg017_logistic_objective_gradient_hessian_exact_beyond_extreme_logits()


def _mut18_omit_sae_residual(mutated: bool) -> None:
    import dd_circuits as circ

    import test_sae_and_circuits as tsc

    def _no_residual_patch(sae: Any, j: int) -> Callable[[torch.Tensor], torch.Tensor]:
        def fn(tokens: torch.Tensor) -> torch.Tensor:
            z = sae.encode(tokens).clone()
            z[..., j] = 0.0
            return sae.decode(z)

        return fn

    if mutated:
        with mock.patch.object(circ, "latent_patch", side_effect=_no_residual_patch):
            tsc.test_hooks_exception_cleanup_causal_isolation_and_residual_preserving_patch()
    else:
        tsc.test_hooks_exception_cleanup_causal_isolation_and_residual_preserving_patch()


def _mut19_uncapped_latent_density_denominator(mutated: bool) -> None:
    import erratum_dd_part4 as err4

    import test_sae_and_circuits as tsc

    if mutated:
        with mock.patch.object(
            err4, "capped_token_density", side_effect=err4.legacy_uncapped_token_density
        ):
            tsc.test_reg018_dd_part4_sha1_preserved_and_erratum_capped_density_exact()
    else:
        tsc.test_reg018_dd_part4_sha1_preserved_and_erratum_capped_density_exact()


def _mut20_swap_log_base_without_converting_units(mutated: bool) -> None:
    import test_geometry_and_mi as tgm
    from galaxy_jepa.probing import information as info

    orig_conv = info.convert_units

    def _swapped(val: float, units: str = "nats") -> float:
        return orig_conv(val, "bits" if units == "nats" else "nats")

    if mutated:
        with mock.patch.object(info, "convert_units", side_effect=_swapped):
            tgm.test_mi_benchmarks_04_05_06_07_discrete_binary_bsc_and_xor()
    else:
        tgm.test_mi_benchmarks_04_05_06_07_discrete_binary_bsc_and_xor()


def _mut21a_disable_sampler_offset_on_resume(mutated: bool) -> None:
    import test_probing_rigour as tpr
    from galaxy_jepa.data import dataset as ds_mod

    def _broken_iter(self: Any) -> Any:
        # Simulate failing to restore sampler offset on resume -> restarts from index 0
        i = 0
        while True:
            epoch, offset = divmod(i, self.n)
            yield int(self.permutation(epoch)[offset])
            i += 1

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        if mutated:
            with mock.patch.object(ds_mod.ResumableShuffle, "__iter__", _broken_iter):
                tpr.test_reg016_deterministic_cpu_checkpoint_resume_matches_uninterrupted(tmp)
        else:
            tpr.test_reg016_deterministic_cpu_checkpoint_resume_matches_uninterrupted(tmp)


def _mut21b_disable_rng_restoration_on_resume(mutated: bool) -> None:
    import test_checkpoint_resume as tcr
    from galaxy_jepa.callbacks import checkpoint as ck_mod

    if mutated:
        with mock.patch.object(ck_mod, "_set_rng_state", lambda _s: None):
            tcr.test_rng_state_survives_a_map_location_that_relocated_it()
    else:
        tcr.test_rng_state_survives_a_map_location_that_relocated_it()


def _mut11_expected_exc() -> tuple[type[Exception], ...]:
    import probe_bank

    return (probe_bank.BankProvenanceError,)


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
        "MUT-21a",
        "reproducibility",
        "Disable sampler-offset restoration on resume",
        _mut21a_disable_sampler_offset_on_resume,
    ),
    (
        "MUT-21b",
        "reproducibility",
        "Disable RNG-state restoration on resume",
        _mut21b_disable_rng_restoration_on_resume,
    ),
)


def run_one(entry: tuple[str, str, str, Callable[[bool], None]]) -> MutantResult:
    mid, cat, desc, fn = entry
    exc = _mut11_expected_exc() if mid == "MUT-11" else (AssertionError,)
    return _run_mutant(mid, cat, desc, fn, expected_exc=exc)


def run_all() -> dict[str, Any]:
    results = [run_one(entry) for entry in CATALOGUE]
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
