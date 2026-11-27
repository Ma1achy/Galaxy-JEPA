# Authorised Rerun Runbook & Factorial Ablation Proposal

This runbook documents exact commands, input/output paths, pinned hashes, prerequisites, and estimated resource requirements for:
1. Local/external-disk errata reruns whose code fixes are already merged on `audit-fixes` (do **not** launch automatically without user confirmation and mounted external storage),
2. Pre-rental validation gate before any paid GPU run, and
3. The pre-registered §16 factorial ablation proposal (proposal only — no GPU rental authorised).

> **Important:** Nothing in this runbook launches paid compute or mutates historical artefacts (`runs/m_v2/`, `runs/ladder_v2/`, `artifacts/out/`, `artifacts/dd_part4.py`, `artifacts/dd_part4b.py`). All reruns write to versioned or dedicated erratum output directories.

---

## 0. Mandatory Pre-Flight Validation Gate

Before any expensive local rerun or GPU rental, run the clean-tree validation manifest generator and verifier:

```bash
make research-validate
.venv/bin/python tools/research_validate.py --verify runs/validation_manifest.json
```

If an intentional development experiment must bypass a stale/dirty manifest, use the explicit override flag (recorded with `"accepted": false`, never masquerading as an accepted scientific run):

```bash
.venv/bin/python tools/research_validate.py \
  --verify runs/validation_manifest.json \
  --dev-override "<reason>"
```

**Pre-rental R2 credential prerequisite (`docs/patch_grid/runbook.md:46`):**  
Recreate the Cloudflare R2 token stored in local `.env` as `R2_RO_*` with strictly **Object Read only** permissions on the cache bucket before copying `.env` onto any remote GPU box.

**Note on `patch-grid-v1` SHA-1 pins (`docs/patch_grid/prereg.md` §3):**  
The pre-registered patch-grid code of record is tagged `patch-grid-v1` at commit `f36ce88` (where `python artifacts/patch_grid_pins.py` verifies `101 files match`). Subsequent remediation commits on `audit-fixes` (Phases 2–5 and validation suites) intentionally modify `src/galaxy_jepa/` and `pyproject.toml` after `f36ce88`. Therefore:
- To run the pre-registered patch-grid box verbatim under Amendment 1 (`f849acbc`), check out tag `patch-grid-v1` (`f36ce88`).
- To run patch-grid from `audit-fixes` HEAD, issue a formal Amendment 2 updating the SHA-1 table in `docs/patch_grid/prereg.md` and `docs/patch_grid/prereg.sha1` before booking compute.

---

## 1. Local / External-Disk Errata Reruns (Authorised Resources Only)

**Common prerequisites:**
- External SSD mounted at `/Volumes/X10 Pro/galaxy-jepa/` (check `F0_CACHE_BASE=/Volumes/X10 Pro/galaxy-jepa/runs/m_v2/cache`, `F0_NORM_PREFIX=246d8de6ba9a`).
- Local virtualenv `.venv` synced (`uv sync --extra dev --extra data --extra eval`).

### 1.1 Axes Part H (`artifacts/axes_h.py` — H1 & H2a)
- **Fixes verified:** unclipped Armijo damped Newton (`REG-006`, `REG-017`), equalised H1 tuning budget, validation-loss shuffled control, `raw_y` alignment for EFIGI $\tau$. Corrupted `docs/axes/h1.json` and `docs/axes/h2a.json` were removed in `7465d50`.
- **Inputs:** A1 / untrained embedding banks + EFIGI catalogue table referenced by `artifacts/axes_h.py`.
- **Command:**
  ```bash
  .venv/bin/python artifacts/axes_h.py --help   # verify subcommands before running
  .venv/bin/python artifacts/axes_h.py h1
  .venv/bin/python artifacts/axes_h.py h2a
  ```
- **Outputs:** Versioned replacement JSONs in `docs/axes/` (verify convergence diagnostics and non-NaN logits before committing).
- **Estimated resources:** Apple Silicon CPU/MPS, ~10–30 min (estimate; depends on CV grid size).

### 1.2 Axes Part B & Gaussian Map-Null (`artifacts/axes_b.py`, `artifacts/axes_b_mapnull.py`)
- **Fixes verified:** Pearson-on-rank-residuals + chunked full paired-row bootstrap (`REG-007`), Cholesky covariance factor $S z$ (`REG-008`).
- **Command:**
  ```bash
  .venv/bin/python artifacts/axes_b.py --help
  .venv/bin/python artifacts/axes_b_mapnull.py --help
  ```
- **Outputs:** Updated Part B and map-null summary JSONs alongside preserved historical originals.
- **Estimated resources:** CPU, ~5–15 min (estimate).

### 1.3 Aligned Comparison C3 Erratum (`artifacts/erratum_aligned_c3.py`)
- **Fixes verified:** Out-of-sample calibrated 1-df plug-in threshold `k_plugin_95` ($\approx 5.85$, empirical Type-I error $0.050 \pm 0.005$ vs legacy $0.168$; `REG-011`).
- **Command (calibration + re-scoring saved ladder_v2 outputs):**
  ```bash
  .venv/bin/python artifacts/erratum_aligned_c3.py --draws 200000 --seed 20261127
  ```
- **Outputs:** Dedicated erratum JSON report (does not overwrite `runs/ladder_v2/`).
- **Estimated resources:** CPU, < 10 s (measured: 100k draws take < 1 s).

### 1.4 Ladder v2 E5/E6 Dedicated Bank & Split-Key Erratum (`artifacts/erratum_ladder_v2_e5e6.py`)
- **Fixes verified:** Dedicated checkpoint-verified bank (`REG-009`), full sorted-ID + population cache key, symmetric joint-row decorrelation (`REG-013`).
- **Inputs:** `runs/m_v2/encoder.pt` (SHA-1 `34bb30a74d38b1fc4622861f43959a92fc01520c`), external cache at `F0_CACHE_BASE`.
- **Command (two-stage contract):**
  ```bash
  .venv/bin/python artifacts/erratum_ladder_v2_e5e6.py --stage bank
  .venv/bin/python artifacts/erratum_ladder_v2_e5e6.py --stage ladder
  ```
- **Outputs:** Dedicated erratum directory (preserves `runs/ladder_v2/` untouched).
- **Estimated resources:** M3 Pro MPS, ~0.5–2 h for bank extraction + ladder probing (estimate based on `docs/ladder_v2/manifest.yaml` `est_h`).

### 1.5 Brief DD Part 4 Density Erratum Check (`artifacts/erratum_dd_part4.py`)
- **Pinned file preserved:** `artifacts/dd_part4.py` SHA-1 `e7a769c4df7b5bfe9c38d3d226237f9d7e27539c`.
- **Synthetic check (no external disk required; verified < 1 s):**
  ```bash
  .venv/bin/python artifacts/erratum_dd_part4.py --synthetic-check
  ```
- **Optional real-store quantification (requires `runs/dd/sae_tokens/`):** Compare `capped_token_density` vs `legacy_uncapped_token_density` on `sae_eval` tokens to check whether any latent crosses the `LIVE = 1e-4` threshold after removing the +2.4% (`262144/256000`) over-count.
- **Estimated resources:** MPS/CPU, ~1–3 min (estimate).

### 1.6 Almagest Part E Diffusion Evaluation (`artifacts/almagest_diffusion_e.py`)
- **Fixes verified:** Held-out evaluation split, exact step-rate calculation, and SD-ratio stop guard (`7465d50`).
- **Estimated resources:** Optional; MPS, ~15–45 min (estimate) if Part E results are promoted beyond exploratory notes.

---

## 2. Pre-Registered Factorial Ablation Proposal (§16 — Proposal Only)

**Not authorised for execution; no GPU rental booked.**

To separate the effects of the historical joint-query predictor, masking bias/truncation fixes, and target weighting $\beta$ cleanly without confounding factors, we propose a pre-registered $2 \times 2 \times 2$ factorial design (plus quarantined legacy-leakage diagnostic if needed):

| Factor | Levels | Notes |
|---|---|---|
| **Predictor** | `joint_legacy` vs `independent_block` | Isolates `REG-001` (cross-block query coupling) |
| **Masking** | `legacy_no_leak` (bbox-context + raster truncation, leak fallback **quarantined off**) vs `corrected_ijepa` (uniform square context + uniform keep) | Isolates `REG-002`, `REG-003`, `REG-004` without re-enabling pixel leakage (`REG-005`) |
| **Target weighting $\beta$** | `beta = 0.0` vs `beta = 0.5` | Tests whether galaxy-weighted target sampling helps once context placement is decoupled |
| **Optional loss-reduction factor** | `equal_block_mse` vs `token_weighted_mse` | Only needed if target block areas vary across arms (`REG-019`) |

**Fixed experimental controls across all arms:**
- Identical dataset split, normalisation hash (`246d8de6ba9a`), ViT backbone, step budget, optimiser/LR/EMA schedule, and `>= 3` seeds per cell.
- **Tracked metrics (with bootstrap/seed uncertainty):** compute cost (GPU-hours), effective context/target token counts, training loss curves, representation effective rank & collapse monitor, held-out linear & nonlinear GZ2 probe AUCs, conditional matching retention, covariance-aware alignment $c_\Sigma$, rotation/handedness robustness, and sky-waste fraction.
- **No prior claim of improvement:** the factorial ablation is designed to measure, not assume, whether standard independent-block I-JEPA outperforms the historical joint-query variant on galaxy morphology.
