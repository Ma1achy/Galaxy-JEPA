# Scientific Validation Audit Inventory

**Branch:** `audit-fixes` (PR #4 targeting `almagest-geometry`)  
**Pinned pre-rental tag preserved:** `patch-grid-v1` at `f36ce88` (all 101 `docs/patch_grid/prereg.md` §3 SHA-1 pins verified unchanged at tag)  
**Pinned historical artefact preserved:** `artifacts/dd_part4.py` SHA-1 `e7a769c4df7b5bfe9c38d3d226237f9d7e27539c` (`docs/ladder_v2/manifest.yaml`)

This inventory maps every requirement in the scientific remediation and validation brief (§§1–17) to its status, concrete code/test/doc path, symbol, commit hash, and verification evidence.

---

## 1. Core Model, Objective & Differential Reference (§2)

| Requirement | Status | Path & Symbol | Commit | Evidence |
|---|---|---|---|---|
| Independent target-block prediction (no cross-block query leakage) | Verified | `src/galaxy_jepa/objectives/jepa.py::Jepa.loss_parts`, `Predictor.forward` | `f36ce88`, `a0c947c` | `tests/test_differential_ijepa.py::test_reg001_independent_target_block_prediction_and_gradient_isolation` |
| Equal-block MSE vs legacy token-weighted MSE explicitly tested & documented vs Meta Smooth-L1 | Verified | `src/galaxy_jepa/objectives/jepa.py::Jepa.loss_parts` | `f36ce88`, `a0c947c` | `tests/test_differential_ijepa.py::test_reg019_equal_block_vs_token_weighted_mse_on_unequal_blocks` |
| Independent scalar-loop gather & equal-block MSE oracle (shares `VisionTransformer`/`Predictor` modules; Meta `facebookresearch/ijepa` not imported directly) | Verified | `tests/test_differential_ijepa.py::_independent_ijepa_oracle` | `a0c947c` | Exact parity on loss + all student/predictor gradients; zero grad on teacher |
| EMA exactness ($m \in \{0, 0.7, 1\}$), teacher optimiser exclusion & fp64 finite-diff gradients | Verified | `src/galaxy_jepa/objectives/jepa.py::Jepa.ema_update` | `a0c947c` | `tests/test_differential_ijepa.py::test_ijepa_ema_exactness_and_finite_difference_gradients` |
| MoCo fp32 InfoNCE/queue under bf16 autocast & non-finite queue guard | Verified | `src/galaxy_jepa/objectives/moco.py::Moco.loss_parts` | `f36ce88`, `a0c947c` | `tests/test_differential_ijepa.py::test_reg015_moco_nonfinite_loss_skips_queue_update_and_fp32_infonce` |
| Methodological deviations vs Meta `facebookresearch/ijepa` (`52c1ae95`) | Documented | `docs/patch_grid/prereg.md` Amendment 1, `src/galaxy_jepa/objectives/jepa.py`, `tests/test_differential_ijepa.py` | `f36ce88`, `a0c947c`, PR-review fix | Three intentional/explicit variant differences documented: (1) equal-block MSE (paper Eq. 2) vs Meta's Smooth-L1 script, (2) learned-affine ViT final LayerNorm used directly vs Meta's extra functional `F.layer_norm` on teacher tokens before gathering, (3) uniform without-replacement context subsampling vs Meta's index-order truncation |

## 2. Masking Distribution, Pixel Firewall & Precomputed Cache (§3)

| Requirement | Status | Path & Symbol | Commit | Evidence |
|---|---|---|---|---|
| Initial context top-left placement uniform and independent of $\beta$ / bbox | Verified | `src/galaxy_jepa/masking/blocks.py::draw_blocks` | `f36ce88`, `a0c947c` | `tests/test_masking_rigour.py::test_reg002_initial_context_placement_independent_of_beta_and_bbox` |
| Context block square aspect ($h = w$) & exact enumerated target PMF | Verified | `src/galaxy_jepa/masking/blocks.py::draw_blocks`, `_sample_topleft` | `f36ce88`, `a0c947c` | `tests/test_masking_rigour.py::test_reg003_context_aspect_is_square_and_enumerated_target_law_matches` |
| Uniform without-replacement batch context subsampling (no raster truncation) | Verified | `src/galaxy_jepa/masking/blocks.py::rebuild_masks` | `f36ce88`, `a0c947c` | `tests/test_masking_rigour.py::test_reg004_context_min_keep_subsampling_is_uniform_not_raster_truncated` |
| End-to-end ViT pixel firewall (withheld target pixels cannot change student context) | Verified | `src/galaxy_jepa/models/vit.py::VisionTransformer`, `MultiBlockMasker` | `f36ce88`, `a0c947c` | `tests/test_masking_rigour.py::test_reg005_end_to_end_encoder_pixel_firewall` |
| Property-based firewall across grids, boxes, and $\beta \in [0, 1]$ | Verified | `tests/test_masking_rigour.py::test_masking_firewall_property_across_grids_boxes_and_betas` | `a0c947c` | Hypothesis 30 examples across grids `{4, 6, 8, 16}` |
| `precomputed-masks-v2` parity & loud refusal of tampered/stale caches | Verified | `src/galaxy_jepa/masking/precomputed.py::PrecomputedMasker` | `f36ce88`, `a0c947c` | `tests/test_masking_rigour.py::test_precomputed_masks_v2_roundtrip_and_negative_refusals` |

## 3. Probing, Preprocessing, Statistics & Multiplicity (§§4–6)

| Requirement | Status | Path & Symbol | Commit | Evidence |
|---|---|---|---|---|
| Unclipped logistic objective/gradient/Hessian exact beyond $\|z\| > 40$ + strict `\|\|grad\|\|_inf <= tol` convergence (`REG-017`, `REG-006`) | Fixed & Verified | `src/galaxy_jepa/probing/stats.py::logistic_objective_and_derivatives`, `damped_logistic`; `artifacts/axes_h.py::newton_logistic` | `ce18da4`, `a0c947c` | `tests/test_probing_rigour.py::test_reg017_*`, `test_reg006_*` |
| Standardised logistic coordinate conversion + 64-bit SDSS ID integrity ($> 2^{53}$) | Verified | `src/galaxy_jepa/probing/logistic.py`, `extract.py` | `2b01f73`, `a0c947c` | `tests/test_probing_rigour.py::test_standardised_logistic_raw_coordinate_conversion_and_64bit_ids` |
| Poisoned held-out partition leakage firewall (`REG-012`) | Verified | `src/galaxy_jepa/probing/extract.py`, `logistic.py` | `a0c947c` | `tests/test_probing_rigour.py::test_reg012_poisoned_test_partition_does_not_affect_train_fitted_artefacts` |
| Full-ID/population cache keys & bank SHA-1 provenance (`REG-009`) | Verified | `src/galaxy_jepa/probing/ladder.py::untrained_split_key`; `artifacts/probe_bank.py::load_bank` | `2b01f73`, `a0c947c` | `tests/test_probing_rigour.py::test_reg009_cache_keys_and_bank_provenance_bind_full_ids_and_population` |
| Partial Spearman vs re-ranked residuals + chunked full paired-row bootstrap (`REG-007`) | Fixed & Verified | `src/galaxy_jepa/probing/stats.py::partial_spearman`, `partial_spearman_bootstrap` | `c7c1046`, `ce18da4`, `a0c947c` | `tests/test_probing_rigour.py::test_reg007_*`; `tests/test_calibration_suite.py::test_bootstrap_ci_coverage_*` |
| Cholesky covariance factor sampling ($S z \sim \mathcal{N}(0, \Sigma)$) (`REG-008`) | Fixed & Verified | `artifacts/axes_b_mapnull.py` | `7465d50`, `a0c947c` | `tests/test_probing_rigour.py::test_reg008_cholesky_covariance_sampling_matches_target_sigma` |
| Zero-inflated stratified matching & non-positive retention verdict (`REG-010`, `REG-014`) | Fixed & Verified | `src/galaxy_jepa/probing/matching.py::stratified_match`, `retention_verdict` | `2b01f73`, `a0c947c` | `tests/test_probing_rigour.py::test_reg010_and_reg014_*`; `tests/test_calibration_suite.py::test_conditional_matching_*` |
| Independent Benjamini–Yekutieli oracle & correlated FDR calibration | Verified | `src/galaxy_jepa/probing/nulls.py::family_significant` | `a0c947c`, `3974e9b` | `tests/test_probing_rigour.py::test_independent_benjamini_yekutieli_*`; `tests/test_calibration_suite.py::test_benjamini_yekutieli_*` |
| C3 1-df plug-in out-of-sample calibration (`REG-011`) | Verified | `artifacts/erratum_aligned_c3.py::calibrate_plugin` | `e0274a3`, `a0c947c`, `3974e9b` | Independent seeds confirm Type-I error $\approx 0.05$ vs legacy $\approx 0.168$ |
| Symmetric joint-row entangled decorrelation (`REG-013`) | Fixed & Verified | `src/galaxy_jepa/probing/ladder.py::_entangled_map` | `2b01f73`, `a0c947c` | `tests/test_probing_rigour.py::test_reg013_entangled_map_symmetric_under_feature_order_swap` |
| Deterministic CPU checkpoint/resume trajectory parity (`REG-016`) | Verified | `src/galaxy_jepa/callbacks/checkpoint.py`, `objectives/loop.py` | `2b01f73`, `a0c947c` | `tests/test_probing_rigour.py::test_reg016_deterministic_cpu_checkpoint_resume_matches_uninterrupted` |

## 4. Geometry, Entanglement & Mutual Information (§§7–8)

| Requirement | Status | Path & Symbol | Commit | Evidence |
|---|---|---|---|---|
| Covariance-aware alignment $c_\Sigma(w_i, w_j)$ GL-invariance & Pearson parity | Helper implemented & unit-tested (production geometry integration outstanding) | `src/galaxy_jepa/probing/entanglement.py::covariance_alignment` | `ce18da4`, `9e1d252` | `tests/test_geometry_and_mi.py::test_covariance_alignment_matches_score_pearson_and_is_gl_invariant` (`entanglement_geometry` / `run_ladder` keep pre-registered raw cosine & MP until formally amended) |
| Empirical shuffled-label Gram null (`mode="joint_rows"` preserving label co-occurrence, plus `"independent_cols"`) | Helper implemented & unit-tested (production geometry integration outstanding) | `src/galaxy_jepa/probing/entanglement.py::empirical_gram_null` | `ce18da4`, `9e1d252`, PR-review fix | `tests/test_geometry_and_mi.py::test_gram_spectrum_svd_parity_and_empirical_gram_null_calibration` (verified on identical-label counterexample under `joint_rows` vs `independent_cols`) |
| Complete `adjudicate_pair` truth table; missing/NaN CAV returns `INCONCLUSIVE`; `"unassailable"` removed | Fixed & Verified | `src/galaxy_jepa/probing/entanglement.py::adjudicate_pair` | `ce18da4`, `9e1d252` | `tests/test_geometry_and_mi.py::test_adjudicate_pair_complete_truth_table_and_missing_cav_inconclusive` (16 states) |
| Absence of historical MI estimator documented; transfer ratio guarded against MI conflation | Documented & Verified | `src/galaxy_jepa/probing/information.py`, `docs/validation/historical_impact_matrix.md` | `ce18da4`, `9e1d252` | `tests/test_geometry_and_mi.py::test_transfer_ratio_sign_alignment_bias_denominator_floor_and_non_mi` |
| Exploratory MI module (`discrete_mi`, `discrete_cmi`, `ksg_mi`, `variational_mi_lower_bound`) validated on all 10 §8.2 analytical benchmarks | Implemented & Verified | `src/galaxy_jepa/probing/information.py` | `ce18da4`, `9e1d252` | `tests/test_geometry_and_mi.py::test_mi_benchmark_*` (all 10 benchmarks + units + unclipped null bound) |

## 5. SAE, Ablations, Circuits & `REG-018` (§9)

| Requirement | Status | Path & Symbol | Commit | Evidence |
|---|---|---|---|---|
| `TopKSAE` structural invariants, AuxK dead-latent gradients, permutation invariance, planted sparse dictionary recovery | Verified | `artifacts/dd_sae.py::TopKSAE`, `losses` | `9e1d252` | `tests/test_sae_and_circuits.py::test_topk_sae_invariants_auxk_permutation_and_planted_dictionary` |
| `Hooks` & `latent_patch`: identity/dead no-op, exception-safe cleanup, layer isolation, residual-preserving subtraction | Verified | `artifacts/dd_circuits.py::Hooks`, `latent_patch` | `9e1d252` | `tests/test_sae_and_circuits.py::test_hooks_exception_cleanup_causal_isolation_and_residual_preserving_patch` |
| `REG-018` token density over-count (`262144/256000 = +2.4%`): pinned `dd_part4.py` preserved byte-identical + corrected erratum helper | Fixed & Verified | `artifacts/erratum_dd_part4.py::capped_token_density` (pinned `artifacts/dd_part4.py` untouched) | `ce18da4`, `9e1d252` | `tests/test_sae_and_circuits.py::test_reg018_dd_part4_sha1_preserved_and_erratum_capped_density_exact` |
| Removed energy $E(S) = \mathbf{1}_S^\top C \mathbf{1}_S$ cross-term parity & last-layer readout oracle | Verified | `artifacts/dd_part4b.py::_energy`, `artifacts/dd_circuits.py` | `9e1d252` | `tests/test_sae_and_circuits.py::test_removed_energy_cross_terms_and_last_layer_ablation_score_oracle` |
| Standalone 8-motif structural-model sanity fixture (distractor, redundancy, synergy, cancellation, bypass, negative control) | Sanity fixture verified (does not validate production circuit discovery) | `tests/test_sae_and_circuits.py::_PlantedCircuit` | `9e1d252` | Demonstrates causal pitfalls on a synthetic SCM; historical block-11 ablations remain exploratory final-readout diagnostics |

## 6. Synthetic Fixtures, Calibration, Miniature E2E, Mutation Audit & CI Gate (§§10–15)

| Requirement | Status | Path & Symbol | Commit | Evidence |
|---|---|---|---|---|
| Synthetic embedding & deterministic 3-band procedural galaxy fixtures | Implemented & Verified | `tests/fixtures_scientific.py` | `3974e9b` | Used across calibration and miniature E2E suites |
| Statistical calibration suite (Type-I error, CI coverage, BY FDR, matching balance/power) | Implemented & Verified | `tests/test_calibration_suite.py` | `3974e9b` | 4 calibration tests with binomial/MC bounds |
| Self-contained CPU miniature E2E pipeline (JEPA $\beta \in \{0, 0.5\}$ + resume, MoCo, MAE, freeze, probes, matching, $c_\Sigma$, MI, SAE, JSON report) | Implemented & Verified | `tests/test_e2e_mini_pipeline.py` | `3974e9b` | Passes locally on CPU in ~3s |
| Isolated §14 scientific mutation audit (all 21 mandatory critical mutants) | Implemented & Verified | `tools/mutation_audit.py`, `tests/test_mutation_catalogue.py` | `185269a` | **21/21 critical mutants killed (100.0%)** across 11 risk categories |
| Provenance-bound validation manifest, `make research-validate`, & staged CI | Implemented & Verified | `tools/research_validate.py`, `Makefile`, `.github/workflows/ci.yml`, `pyproject.toml` | `185269a` | Verifies clean HEAD, pins, tests, mutation audit, and credentials; discloses blocked CUDA/catalogue checks |
| CUDA GPU parity & external SDSS/GZ2 full catalogue reruns | Explicitly Blocked | `docs/validation/rerun_runbook.md` | — | No CUDA GPU on local Mac; no paid rentals or heavy catalogue reruns without user approval |
