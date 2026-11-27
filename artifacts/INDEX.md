# `artifacts/` Directory Index

`artifacts/` holds experiment drivers, pre-registration documents, cloud-shipping scripts, and
historical investigation scripts. Files pinned by SHA-1 in any pre-registration or runner manifest
remain at their original paths so every historical hash and import continues to resolve verbatim.

## 1. Active experiment drivers

### Patch-size grid (Nebius 8×H200 — `docs/patch_grid/prereg.md`, tag `patch-grid-v1`)
- `patch_grid_masks.py` — precomputes float32 `petro`/`pscale` multi-block masks (`precomputed-masks-v2`).
- `patch_grid_run.py` — single-arm training driver (JEPA, MoCo, grouped MAE) with committed-step resume.
- `patch_grid_stage1.py` — Stage 1 throughput smokes + `judge` gate (`--smokes-only` supported).
- `patch_grid_embed.py`, `patch_grid_auc.py`, `patch_grid_score.py`, `patch_grid_spread.py`, `patch_grid_frozen.py`, `patch_grid_killtest.py` — embedding extraction, scoring, and pre-run kill tests.
- `patch_grid_pins.py` — verifies all 101 pinned SHA-1s listed in `docs/patch_grid/prereg.md`.
- `run.sh`, `nebius_watch.sh`, `ship.py`, `results.py` — Nebius launch/watch, R2 shard shipping, and result pull/verify.

### Axes programme (Parts A–H — `docs/axes/`)
- `axes_a_bis.py`, `axes_a_ordinal.py` — Part A (ordinal/continuous probes).
- `axes_b.py`, `axes_b_ids.py`, `axes_b_mapnull.py`, `axes_b_partial.py`, `axes_b_pull.py`, `axes_b_statmorph.py` — Part B (non-parametric morphology / statmorph partial correlations and map null).
- `axes_c_matched.py`, `axes_c_star.py` — Part C (matched / star-forming controls).
- `axes_f.py`, `axes_f1_thin.py`, `axes_f345.py`, `axes_f5_topup.py` — Part F.
- `axes_g.py`, `axes_g3.py` — Part G (EFIGI independent morphology test).
- `axes_h.py` — Part H (spatial pooling & multi-layer readout probes; damped Newton L2 logistic).

### Almagest geometry & diffusion (`docs/almagest/`)
- `almagest_axes.py`, `almagest_spiral.py`, `almagest_visibility.py`, `almagest_visibility_b2b.py`, `b2b_kcorrect.py` — visibility, spiral, and axis geometry.
- `almagest_diffusion.py`, `almagest_diffusion_checks.py`, `almagest_diffusion_e.py`, `almagest_diffusion_smoke.py` — conditional diffusion reconstruction checks.

### Audit errata & verification scripts (Nov 2026)
- `erratum_aligned_c3.py` — 1-df plug-in Monte Carlo calibration ($k_{\text{plug-in}} \approx 5.84$) and rescoring for `aligned_comparison.md` Criterion 3.
- `erratum_ladder_v2_e5e6.py` — builds a conditional-population $K = 30$ untrained bank and re-runs Ladder v2 E5 + E6 into `docs/ladder_v2_erratum/` without modifying `runs/ladder_v2/`.

## 2. Pinned historical studies (do not move or alter)

- **Aligned comparison (`aligned_comparison.md`):** `aligned_a3_report.py`, `aligned_c13.py`, `aligned_c2.py`, `aligned_c3_floor.py`, `aligned_c3_sweep.py`, `aligned_launch.sh`, `aligned_score.py`, `aligned_train.py`, `pretrain_v2_validate.py`, `v1_v2_std_common.py`, `v3_relabel_check.py`.
- **Baselines comparison (`baselines.md`, `baselines_comparison.md`):** `baseline_run.py`, `baseline_f7.py`, `baseline_rehearsal.py`, `baseline_trajectory_probe.py`, `baselines_answer_set.py`, `baselines_comparison.py`, `baselines_false_same.py`, `baselines_nuisance_pull.py`, `baselines_p3.sha1`, `baselines_p3_s2.sha1`, `baselines_outputs.sha1`, `sky_sigma.py`.
- **Ladder v2 & M comparison (`docs/ladder_v2/`, `docs/m_comparison/`):** `ladder_env.py`, `ladder_v2_run.py`, `ladder_v2_manifest.py`, `ladder_v2_baselines_manifest.py`, `ladder_v2_baselines_export.py`, `ladder_v2_m_manifest.py`, `ladder_v2_c2.py`, `ladder_v2_ctl.py`, `ladder_v2_e2.py`, `ladder_disk_guard.sh`, `ladder_v2_watchdog.sh`, `m_comparison.py`, `m_comparison_explore.py`, `p1_untrained_bank.py`, `p2_ladder.py`, `s1_spiral_inclination.py`, `s2_verify.py`, `t1_ladder.py`, `t2_other.py`, `u1_bar_winding.py`, `u2_uncertainty.py`, `u3_graded.py`, `v1_loose_ends.py`, `v2_independent.py`, `v3_own_directions.py`, `w2_name_pcs.py`, `x1_handedness.py`, `x1_vector_posthoc.py`, `y3_science.py`, `y_pitch.py`, `z3_bt.py`, `z_checks.py`, `aa1_label_efficiency.py`, `aa2_decals.py`, `aa3a_pose.py`, `aa3b_pose_average.py`, `bb2.py`, `bb2_p2_explore.py`, `bb_pitch.py`, `r_nonlinear.py`, `r2_reach_split.py`, `plate_export.py`, `probe_bank.py`, `o1_matched.py`, `o2_second_seed.py`, `o3_overfit_gate.py`, `n1_controls.py`, `n2_floor_evidence.py`, `m1_preflight.py`, `m2_long_run.py`, `m_band_axis.py`, `j1_preflight.py`, `j4_spread_controls.py`, `j5_floor_evidence.py`, `k2_trajectory_probe.py`, `l1a_cache_embeddings.py`, `l1b_mlp_ladder.py`, `h5_probe_lean.py`, `f0_preconditions.py`.
- **Interpretability / DD (`interp_tooling.md`, `leakage_audit.md`):** `dd_core.py`, `dd_circuits.py`, `dd_l2336_scan.py`, `dd_part1.py`, `dd_part1b.py`, `dd_part2.py`, `dd_part4.py`, `dd_part4b.py`, `dd_s2_probe.py`, `dd_sae.py`, `dd_sae_cards.py`, `dd_sae_score.py`, `dd_step0.py`, `dd_v3.py`, `dd_v4ref_4999_chain.sh`, `leakage_audit.py`, `leakage_datacheck.py`, `leakage_diag_invalid.py`, `sae_auxk_identity.py`, `sae_mem_diag.py`, `test_dd_circuits.py`.
- **Data pull & pre-processing:** `sciserver_cut.py`, `sciserver_cut_b2b.py`, `sciserver_cut_v2.py`, `sciserver_pull.py`, `_sciserver_auth.py`, `e5_corpus_moments.py`, `e5_fit_normalisation.py`, `i4_probe_columns.py`, `repull_fix_v1log.py`, `repull_pilot.py`, `repull_targets.py`, `fidelity_test.py`.
- **Process helpers & findings notes:** `_heavy.sh`, `_detach.py`, `train_watchdog.sh`, `f_watchdog.sh`, `f_proc_watchdog.sh`, `*_findings.md`, `readme_figures.py`.

## 3. Archived legacy scripts (`artifacts/legacy/`)

Unpinned, unimported one-off exploration and scaffolding scripts from early data-pulling and
preliminary runs (`_domain_detail.py`, `_fetch.py`, `_plan_*.py`, `e1_validity_survey.py`,
`e6_bake_cache.py`, `g2_build_scalars.py`, `g3_remeasure.py`, `j3_medium_run.py`, `k2_report.py`,
`rehearsal_corpus.py`, `sciserver_driver.py`, etc.) have been archived under `artifacts/legacy/`
to keep the top-level directory navigable without deleting project history.
