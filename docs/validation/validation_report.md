# Scientific Validation Report

**Date:** 2026-11-27  
**Branch:** `audit-fixes` (extending PR #4 targeting `almagest-geometry`)  
**Host Environment:** macOS 27.0.1 (`arm64`, 11 CPU cores, Apple MPS available, CUDA **unavailable**)  
**Software Versions:** Python `3.11.8`, PyTorch `2.12.0`, NumPy `2.4.6`, SciPy `1.17.1`, scikit-learn `1.9.0`, Hypothesis `6.155.2`, pytest `9.1.0`

---

## 1. Executive Summary of Verification Runs

| Gate / Tool | Exact Command | Result | Exact Counts |
|---|---|---|---|
| Lint | `.venv/bin/ruff check .` | **PASS** | 0 errors across all checked files |
| Formatting | `.venv/bin/ruff format --check .` | **PASS** | 133 files formatted cleanly |
| Static Typing | `.venv/bin/mypy --cache-dir=/tmp/mypy_cache src` | **PASS** | 0 issues in 61 source files |
| Fast CI Gate (`unit`, `invariant`, `regression`, `property`, `differential`, `scientific`, `calibration`, `mutation`) | `PYTHONDONTWRITEBYTECODE=1 HYPOTHESIS_STORAGE_DIRECTORY=/tmp/hyp_cache .venv/bin/pytest -q -p no:cacheprovider --basetemp=/tmp/pytest_tmp -m "not integration and not e2e and not gpu and not slow and not external_data" --durations=10` | **PASS** | **638 passed, 0 failed, 0 skipped** |
| Integration + CPU Miniature E2E Gate | `PYTHONDONTWRITEBYTECODE=1 HYPOTHESIS_STORAGE_DIRECTORY=/tmp/hyp_cache .venv/bin/pytest -q -p no:cacheprovider --basetemp=/tmp/pytest_tmp -m "integration or (e2e and not gpu)" --durations=10` | **PASS** | **114 passed, 0 failed, 0 skipped** |
| Combined Repository Suite | `fast` + `integration + e2e` | **PASS** | **752 passed, 0 failed, 0 skipped** (+78 new validation tests over prior 674) |
| Isolated §14 Scientific Mutation Audit | `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/mutation_audit.py` | **PASS** | **21 / 21 critical mutants killed (100.0%)** across 11 risk categories |
| Credential & Secret Scan | `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/credential_check.py` | **PASS** | 0 env-value hits, `[0, 0, 0, 0, 0]` pattern hits |
| Historical Pin Integrity | `shasum artifacts/dd_part4.py artifacts/dd_part4b.py` | **PASS** | `dd_part4.py` = `e7a769c4df7b5bfe9c38d3d226237f9d7e27539c` (exact match); `dd_part4b.py` = `c4fec662cc3af7426e554850f9045785ed15c25b` |

---

## 2. Independent Reference Oracles & Numerical Tolerances

| Suite | Reference Authority | Seeds | Tolerances & Key Verified Properties |
|---|---|---|---|
| **I-JEPA & MoCo Differential** (`tests/test_differential_ijepa.py`) | Hand-written CPU scalar-loop gather + per-block MSE oracle (`_independent_ijepa_oracle`), fp64 central finite differences ($\varepsilon = 10^{-6}$) | `0, 1, 2, 3, 4, 10, 11` | Loss parity `rel=1e-5, abs=1e-6` (fp32 `loss_parts`), gradient parity `rtol=1e-4, atol=1e-5`; EMA exact `rtol=1e-13, atol=1e-14`; fp64 FD gradient parity `rel=1e-5, abs=1e-7`; MoCo fp32 InfoNCE under bf16 + NaN queue guard |
| **Masking & Pixel Firewall** (`tests/test_masking_rigour.py`) | Exact enumerated 4×4 PMF (`9,000` draws), Hypothesis property generator (30 examples across grids `{4, 6, 8, 16}`) | `0..1199, 7, 42, 123` | Target PMF `atol=0.02`; uniform context subsampling `atol=0.05`; zero pixel leakage (`torch.equal`) across ViT context path |
| **Probing, Logistic & Bootstrap** (`tests/test_probing_rigour.py`) | `sklearn.linear_model.LogisticRegression`, fp64 central finite differences ($\varepsilon = 10^{-6}$), textbook Benjamini–Yekutieli step-up | `6, 10, 11, 12, 13, 17, 41, 101, 202, 707, 808` | Unclipped logistic gradient/Hessian FD match `rtol=1e-5, atol=1e-6` at $\|z\| > 40$; `sklearn` logit parity `rtol=1e-3, atol=1e-3`; chunk-invariant bootstrap `abs=1e-12`; 64-bit IDs $> 2^{53}$ exact |
| **Geometry & 10 Analytical MI Benchmarks** (`tests/test_geometry_and_mi.py`) | Analytic Gaussian MI ($-\frac{1}{2}\ln(1-\rho^2)$, $\frac{1}{2}\ln\frac{|\Sigma_X||\Sigma_Y|}{|\Sigma_{XY}|}$), exact BSC($p$) ($\ln 2 - H_b(p)$), exact XOR truth table, SVD parity | `71, 72, 81, 8201, 8203, 8206, 8208` | $c_\Sigma$ GL-invariance `rel=1e-11`; discrete MI/CMI `abs=1e-12` (BSC $N=20\text{k}$ `abs=0.015`); KSG-1 Gaussian MI within `0.04–0.09` nats ($N=4\text{k}$–$6\text{k}$); unclipped Barber–Agakov bound negative under overfitted null |
| **SAE, Ablations & Planted Circuits** (`tests/test_sae_and_circuits.py`) | Exact $E(S) = \mathbf{1}_S^\top C \mathbf{1}_S$, exact last-layer readout $\Delta s_k$, 8-motif structural causal model (`_PlantedCircuit`) | `91, 92, 94, 95, 910, 911, 941` | Planted sparse dictionary variance explained $> 0.85$ & median atom cosine $> 0.85$; energy parity `rel=1e-12`; last-layer score delta `rtol=1e-5`; 100% precision/recall on causal motifs + redundancy/distractor discrimination |

---

## 3. Statistical Calibration Summary (`tests/test_calibration_suite.py`)

| Calibration Check | Design & Seeds | Nominal Target | Accepted Band (Monte Carlo Uncertainty) | Status |
|---|---|---|---|---|
| **C3 1-df Plug-in Type-I Error** | Calibrated on `40,000` draws (`seed=1101`); evaluated out-of-sample on `40,000` independent draws (`seed=1102`) | $\alpha = 0.050$ | $[0.044, 0.056]$ ($\text{SE} = 0.0011$, $> 5\,\text{SE}$) | **PASS** (legacy `K_ORIGINAL` confirmed $\approx 0.168$ in `test_probing_rigour.py`) |
| **Permutation Null Type-I Error** | `200` independent null datasets (`n=80`, `99` permutations each, `seed=1103`) | $\alpha = 0.050$ | $[0.010, 0.100]$ (99.7% binomial band) | **PASS** |
| **`partial_spearman_bootstrap` 95% CI Coverage** | `100` replications (`n=160`, `250` paired resamples) against `80,000`-sample ground truth (`seeds 1110, 1111`) | $0.950$ | $[0.870, 0.990]$ | **PASS** |
| **`probe_auc_ci` 95% CI Coverage** | `90` replications (`n=160`, `250` test resamples) against `40,000`-sample fixed-probe population AUC (`seed=1112`) | $0.950$ | $[0.860, 0.990]$ | **PASS** |
| **Benjamini–Yekutieli FDR & Power** | `200` strongly correlated mixed families (`m=12`: 8 nulls + 4 alternatives at $+4\sigma$, positive & negative off-diagonals, `seed=1120`) | $\text{FDR} \le 0.050$, Power $\ge 0.70$ | Mean $\text{FDP} \le 0.060$, Power $\ge 0.70$ | **PASS** |
| **Conditional Matching Balance & Power** | World-confounded (`seed=1130`) vs representational (`seed=1131`) synthetic pairs (`n=900`, 10 strata) | $|\text{SMD}| < 0.12$; confound AUC $< 0.58$; signal AUC $> 0.78$ | All three criteria simultaneously met | **PASS** |

---

## 4. Miniature CPU End-to-End Pipeline (`tests/test_e2e_mini_pipeline.py`)

Verified locally on CPU with zero external credentials or astronomical catalogue downloads:
- **Procedural catalogue (`tests/fixtures_scientific.py::generate_procedural_galaxies`):** 120 deterministic 3-band `(g, r, i)` stamps (`32×32`) with 64-bit IDs $> 2^{53}$, split strictly disjointly (`40` pretrain / `40` probe-train / `40` probe-test).
- **Pretraining & resume:** Tiny ViT (`patch_size=8, embed_dim=16, depth=2, heads=2`) trained for I-JEPA $\beta=0.0$ (4 steps), I-JEPA $\beta=0.5$ (interrupted at step 2, resumed to step 4 via `TrainCheckpointer`), MoCo (3 steps), and MAE (3 steps). All losses finite.
- **Freeze boundary:** Reloaded via `load_frozen_encoder`; verified `not p.requires_grad` and `p.grad is None` for every parameter before extraction.
- **Downstream analysis:** Linear probes + bootstrap CIs + untrained controls, stratified matching on brightness nuisance, covariance-aware alignment $c_\Sigma$, exploratory KSG & Barber–Agakov MI bounds, tiny `TopKSAE` + `Hooks`/`latent_patch` block-2 intervention, and machine-readable JSON report cross-checked against live values.

---

## 5. Explicitly Disclosed Blocked Checks & Known Limitations

1. **CUDA GPU Parity (`blocked`):** No CUDA GPU is attached to the local Apple Silicon workstation (`cuda_available: false` in `tools/research_validate.py`). GPU execution parity and bf16 FlashAttention/CUDA throughput checks are explicitly marked **blocked** (never passed) until an authorised GPU run.
2. **External SDSS / GZ2 / EFIGI Full Catalogue Reruns (`blocked`):** No external-SSD heavy reruns (`axes_h.py`, `axes_b.py`, `erratum_ladder_v2_e5e6.py`, `almagest_diffusion_e.py`) or paid cloud GPU rentals were executed, per explicit user authorisation scope. Exact commands and resource estimates are documented in [`rerun_runbook.md`](rerun_runbook.md).
3. **Scope of Synthetic Benchmarks:** Procedural galaxy stamps and planted causal circuits validate algorithmic, statistical, and software correctness only; passing tests is **not** proof of astrophysical validity or that standard I-JEPA outperforms the historical joint-query variant on real SDSS galaxies.
4. **Slowest Fast-Gate Tests:** `test_bootstrap_ci_coverage_partial_spearman_and_probe_auc` (36.8s) and `test_type1_error_c3_plugin_and_permutation_null` (29.9s) keep the full `fast` gate around ~95s locally; GitHub Actions runs `fast` cleanly within the ~2 min target.
5. **Pre-existing Hypothesis Deadline Fix:** Added `deadline=None` to `tests/test_data_orchestrate.py::test_three_way_hits_ratios_at_scale` after observing a CPU-load `DeadlineExceeded` flake during full-suite execution.
