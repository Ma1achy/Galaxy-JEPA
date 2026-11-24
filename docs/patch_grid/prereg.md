# Patch-size grid 16 → 8 → 4 (D11): pre-registration (DRAFT v0, browser draft; not hashed)

**Status.** [CC] items completed 24 Nov by Claude Code; mismatches and ambiguities are in
`docs/patch_grid/flags.md` and were not reconciled here. Still a draft, still unhashed.
Design draft written 24 Nov 2026 from B1, B1b, Part A and the cost estimate
(`docs/patch_ablation_cost.md`). Nothing here is hashed. Claude Code completes the operational
detail marked **[CC]**, writes and runs the planted tests (D28), and reports them. Then the user
approves the hash. The hash covers the section from "## Pre-registration" to the next level-2
heading (SHA-1, trailing newline). It comes **before any trained 8×8 or 4×4 checkpoint exists**:
the rental starts only after the hash is committed and pushed.

**What this draft was written from.** `docs/almagest/a1_visibility_report.md` (B1, B1b),
`docs/axes/a_report.md` (Part A), `docs/patch_ablation_cost.md`, and the chat record of the decisions
of 21 Nov. No 8×8 or 4×4 result exists. B2b's numbers are not yet available: §8 says how they enter.

---

## Pre-registration

### 1. Question

SDSS seeing is about 1.4″, so the PSF FWHM is about 3.5 px at 0.396″/px. A1 reads first-order
morphology at every resolution. But structure (bar, spiral, ring) appears only above R_petro ≈ 4–5 × PSF
(B1 knee at decile 3–4), and fine detail (dust lane, disturbed, merger, dominant bulge) only above
≈ 6–10 × PSF. Arm count and winding order only in the best-resolved deciles (Part A).

**Q1.** Is that limit set by the 16-px patch (architecture) or by the PSF (data)?
**Q2.** Does the answer differ by objective (JEPA, MoCo, MAE)?

**Hypotheses, stated in advance:**
- **H_patch:** at 16×16 the patch limits resolution, so halving it to 8×8 lets structure and fine
  detail be read in smaller galaxies.
- **H_psf:** a 4-px patch is about one PSF width, so 8 → 4 adds little beyond 16 → 8.
- **Objective ordering (weakly held, from design):** MAE, which reconstructs pixels, gains most from
  smaller patches; JEPA, which predicts in latent space, gains less; MoCo, whose augmentations blur
  fine detail, gains least.

### 2. Arms

| objective | patch | seeds (train_seed) | role |
|---|---|---|---|
| JEPA | 16 (GPU, bf16) | 0, 1 | GPU reference; device/precision control against Mac A1 (seed 0), A2 (seed 1) |
| JEPA | 8 | 0, 1 | |
| JEPA | 4 | 0, 1 | |
| MoCo | 16 (GPU, bf16) | 0, 1 | GPU reference; control against the Mac MoCo runs |
| MoCo | 8 | 0, 1 | |
| MoCo | 4 | 0, 1 | |
| MAE | 16 (GPU, bf16) | 0, 1 | GPU reference; control against the Mac MAE runs |
| MAE | 8 | 0, 1 | |
| MAE | 4 | 0, 1 | |

18 trained encoders. Untrained banks: 3 seeds per patch size per objective architecture, as for patch 16.
**[CC, completed 24 Nov]** All three objectives have a paired device control. By train_seed, the Mac
runs (`runs/`, hashed in the baselines comparison, `artifacts/baselines_comparison.md`) are:
- seed 0: A1 (`m_v2`), `moco_s0`, `mae_s0`;
- seed 1: A2 (`m_v2_s1`), `moco_s1`, `mae_s1`;
- seed 2 (no GPU partner in this grid): A3 (`m_v2_s2`), `moco_s2`, `mae_s2`.

The Mac runs trained on MPS without autocast (fp32), so device and precision change together.

### 3. Recipe held fixed

- A1's recipe: 810,491 train stamps, batch 32, 101,308 steps (4 epochs), cosine over 253,270 steps stopped
  at 101,308. The same cache and images for every arm.
- bf16 with flash attention on every GPU arm, including the patch-16 references.
- **Masking fixed in pixels, not tokens:**
  - JEPA: target blocks and the 2.5 × R_petro box cover the same sky at every patch size (cost doc table).
  - MoCo: windows scale with pixels (10–14 tokens at 16; 20–28 at 8; 40–56 at 4).
  - MAE: masking in 16-px groups. The normalised pixel target is computed over the 16-px group, identical
    to patch 16, so only the encoder token size changes.
- Positional embeddings sin-cos from each grid; a new patch-embedding layer at each size.
- The 2.5 × R_petro masking box depends on galaxy size. It is held fixed, and this is recorded as a known
  way size enters training (B0).
- **[CC, completed 24 Nov as far as it can be]** SHA-1 of the code of record as it stands:

  | file | SHA-1 |
  |---|---|
  | `artifacts/aligned_train.py` (A1–A3 launcher) | `c25432afc00e1ebb107a3f714d970f44842ccb4a` |
  | `artifacts/aligned_launch.sh` | `e7566a33ef4ed4d2e545ccd2e02ca815d0a2d67d` |
  | `artifacts/m2_long_run.py` | `5c8c70155cdc90043838c1adb2c40a06dc38fb2e` |
  | `artifacts/baseline_run.py` (MoCo, MAE) | `fafc5583deacb2385ce5e9e23c17f9edc72635a5` |
  | `configs/pretrain_v2.yaml` | `c232b054f42114c9ad6e5e76dbc8c8c342ebf838` |
  | `configs/moco.yaml` | `8aa9d3ddaed785e34751477eb925d163bb45d89e` |
  | `configs/mae.yaml` | `4ccbd4be31b55d993558780a27b8854b8e9e9018` |
  | `configs/probe.yaml` | `c0ddb4e322b70e90adfaf6b2d7760ba42003d28b` |
  | `src/galaxy_jepa/objectives/jepa.py` | `992653bcc3164fefe9530625f1bf909a0ea7ed51` |
  | `src/galaxy_jepa/objectives/moco.py` | `00e52331967d062c706207c85b2bde8868505d5b` |
  | `src/galaxy_jepa/objectives/mae.py` | `3bb939a8b2cb24f863489b63ad5986b656ba0d0c` |
  | `src/galaxy_jepa/objectives/loop.py` | `4853f30a971c4eef8a488f69a491807e5c1ba18b` |
  | `src/galaxy_jepa/masking/blocks.py` | `2120144c2ba8fc83dee47b659013e0dd9efc84db` |
  | `src/galaxy_jepa/models/vit.py` | `c5f3d35daad4b4986f063e3657c96eda6415f592` |
  | `src/galaxy_jepa/callbacks/collapse.py` | `88133c65de5b29935008ff569c578237e31bfa68` |
  | `src/galaxy_jepa/harness.py` | `11d457d5586b7ca71d6efff142fc32ba4f21e866` |
  | `artifacts/almagest_visibility.py` (B1) | `9e1c0e112c272af91b246278ca6ecb38bec27018` |
  | `artifacts/almagest_axes.py` | `92235371250cdee52d9d29295ebce3a3e5f411c6` |
  | `artifacts/patch_grid_score.py` (scoring, plants) | re-pinned at the hash |

  **Not pinnable yet:** the patch-8 and patch-4 configs, the GPU bf16 launcher, MAE's 16-px grouping and
  the patch-scaled MoCo windows do not exist (`flags.md` 1–3). They must be written and tested before the
  hash, and their SHA-1s added to this table.

### 4. Evaluation set, frozen before any training

- Probes: the Scheme-1 L2-logistic probes, the ladder's recipe, fitted per encoder on the frozen train
  split; evaluated on the frozen test split.
- **Resolution deciles:** B1's decile edges in R_petro_r / psfWidth_r, per answer, taken from A1's test
  galaxies and frozen. The same galaxies sit in the same decile for every arm.
- **Answers:** the 30 STRONG or SEPARATE answers from relabel2, grouped by B1:
  - **G1, first-order** (clear at every resolution): smooth, features, edge-on yes, edge-on no,
    completely round, cigar, odd yes, odd no, no bulge.
  - **G2, structure** (knee at decile 2–4): bar, no bar, spiral, no spiral, ring, lens/arc, obvious bulge,
    in between, loose winding, 2 arms, bulge-shape no bulge.
  - **G3, fine** (knee ≥ 5, or none): dominant bulge, dust lane, disturbed, irregular, other, merger, bulge
    rounded, tight winding, just noticeable, can't tell (arms).
  - The groups are fixed now from A1's B1 results and do not change after any new result.
- **Ordinal axes (Part A):** arm count and winding, scored as Kendall τ of the ordinal score with
  E[category] within the same deciles.
- **[CC, completed 24 Nov]** As implemented, the Scheme-1 ladder probes use **raw** embeddings: cleaning exists
  only in the Almagest code (`artifacts/almagest_spiral.py`), which B1 and Part A used. Either choice runs, and the
  scoring code reports both. The choice of primary is the user's; the draft's choice:
  **raw primary** (the decile already controls apparent size, and B1 showed the cleaner can distort
  within-decile comparisons, e.g. no bulge), cleaned reported beside it.

### 5. Primary statistic

For encoder e (objective o, patch p, seed s) and answer a:

- **Low-resolution margin** L(e, a) = the mean over deciles 1–5 of [AUC(e, a, d) − max over untrained
  seeds at the same o and p of AUC(u, a, d)].
  - Deciles 1–5 are where B1 placed the knees for G2 and G3, so a resolution gain shows up here first.
  - Margins are taken against the untrained encoders **of the same patch size**, so a change in what an
    untrained 4×4 ViT already reads is not credited to training.
- **Step change** for objective o, from patch p to patch q: Δ(o, a, p→q) = mean over seeds of L at q
  minus mean over seeds of L at p. Seeds pair by train_seed.
- **Group statistic:** the median of Δ over the answers in the group.
- **Seed spread:** for each (o, p), the |L(seed 0) − L(seed 1)| per answer; the group's spread s_g is
  the median of those over the group, pooled across the two patch sizes in the step.
- **Secondary, descriptive:** the full-sample existence AUC per answer and arm; the knee per answer,
  defined on the CI as the first decile d where deciles d and d+1 both have their bootstrap 95% CI lower
  bound above the untrained maximum (1,000 galaxy resamples within decile), or NONE; and the ordinal τ
  per decile for arm count and winding.

### 6. States (D27: every outcome has one; first match applies)

Constants, consistent with the baselines comparison: **SESOI = 0.02** and **NEGLIGIBLE = 0.005** AUC.
Bar for a step: B_g = max(SESOI/2, 2 · s_g) = max(0.01, 2 · s_g).

**Per group, per objective, per step (16 → 8 and 8 → 4):**

| # | condition | step state |
|---|---|---|
| 1 | fewer than 5 answers in the group have a defined L at both sizes | INSUFFICIENT |
| 2 | median Δ > B_g, and at least 2/3 of the group's answers have Δ > 0 | GAIN |
| 3 | median Δ < −B_g, and at least 2/3 have Δ < 0 | LOSS |
| 4 | \|median Δ\| ≤ NEGLIGIBLE and s_g ≤ SESOI/2 | NO CHANGE |
| 5 | otherwise | UNRESOLVED |

**Per group, per objective, combining the two steps:**

| # | 16 → 8 | 8 → 4 | group reading |
|---|---|---|---|
| 1 | INSUFFICIENT in either | | INSUFFICIENT |
| 2 | LOSS in either | | WORSE WITH SMALLER PATCHES (say which step) |
| 3 | GAIN | NO CHANGE | **PATCH-LIMITED TO A PSF FLOOR** (H_patch and H_psf both supported) |
| 4 | GAIN | GAIN | **PATCH-LIMITED BELOW 8 px** (H_psf not supported) |
| 5 | GAIN | UNRESOLVED | PATCH-LIMITED; FLOOR UNRESOLVED |
| 6 | NO CHANGE | NO CHANGE | **DATA-LIMITED AT 16 px** (H_patch not supported) |
| 7 | NO CHANGE | GAIN | LIMITED BELOW 8 px ONLY (unexpected; report as such) |
| 8 | otherwise | | UNRESOLVED |

**Predicted readings, stated so they can fail:**
- G1: DATA-LIMITED AT 16 px, or a small GAIN (it is already resolved at decile 1).
- G2 and G3: PATCH-LIMITED TO A PSF FLOOR for JEPA.
- Arm count and winding (descriptive, §5): τ in deciles 4–7 rises at 8 × 8 and changes little at 4 × 4.

**Objective comparison (Q2), per group and step:** D(o1, o2) = Δ(o1) − Δ(o2), using the same answers.
Bar: max(SESOI/2, 2 · √(s_o1² + s_o2²)).

| # | condition | state |
|---|---|---|
| 1 | either objective INSUFFICIENT | INSUFFICIENT |
| 2 | \|D\| > bar | DIFFERS (sign reported) |
| 3 | \|D\| ≤ NEGLIGIBLE and both spreads ≤ SESOI/2 | SAME |
| 4 | otherwise | UNRESOLVED |

Pairs: JEPA–MAE, JEPA–MoCo, MAE–MoCo. Predicted (weakly): MAE > JEPA > MoCo in Δ for G2 and G3 at 16 → 8.

**Device/precision control (not a reading on Q1 or Q2):**
- For each objective with a Mac counterpart, the per-answer full-sample AUC of the GPU patch-16 run against
  the Mac run with the same train_seed.
- **DEVICE EFFECT** if the median absolute paired difference exceeds the Mac A1–A3 seed spread for the
  same answers; otherwise **NO DEVICE EFFECT**.
- Either way, every Q1 and Q2 comparison uses the GPU patch-16 runs as the reference. The Mac runs are
  never mixed into a step change.

### 7. Validity, exclusions and what stops the run

- **Collapse.** The effective-rank collapse monitor runs on every arm at the project's existing bar
  (**[CC, completed 24 Nov]** `collapse_floor` in `configs/pretrain_v2.yaml`, and in `moco.yaml` and
  `mae.yaml`: min_effective_rank 2.5, grace fraction 0.10, hard floor 2.0 after step 100, three consecutive
  readings. It was re-derived at D17, made active at D21, and lives in `src/galaxy_jepa/callbacks/collapse.py`.
  **This bar is not the same for every objective in the code of record:** F8 settled that for MoCo and MAE
  only the hard floor gates, and the soft floor is reported. See `flags.md` 7). An arm that collapses is VOID. It is rerun once with the same seed and recipe. A
  second collapse leaves it VOID, and the groups are scored without it, marked. Seeds are never swapped
  or chosen.
- **Divergence.** NaN loss or a loss above the run's step-1,000 value after step 20,000 counts as
  divergence and is handled as collapse.
- **Timing smoke.** A 200-step timed smoke per arm on the rented box comes first. It may change the GPU
  type or the order of runs, never the recipe.
- **Cache parity.** The launcher refuses to train unless the shipped cache passes the manifest SHA-1 check
  and the pipeline_hash matches (shipping plan).
- **Blindness.**
  - Decile edges, group membership, probes' recipe and untrained banks are frozen before any trained
    checkpoint exists.
  - The scoring code is pinned by SHA-1 before any trained 8 × 8 or 4 × 4 record is opened.
  - The outputs are listed in an outputs.sha1 manifest before scoring, as for the M comparison.

### 8. Secondary analyses (descriptive; no states)

- **Labels per arm:** the 37-axis relabel2 pipeline (STRONG / SEPARATE / PROXY / WEAK / NOISE) on every
  trained arm. In particular, do the NOISE answers (boxy bulge, 3 arms, more than 4 arms) become stable
  at 8 × 8 or 4 × 4?
- **Part A on every arm:** the ordinal-versus-free shares and gaps for arm count, winding, bulge and
  roundness.
- **B2b on every arm, if B2b has run on A1 by then:** the per-answer score shift with simulated redshift.
  Prediction: smaller shifts at 8 × 8 for G2 and G3, if the patch was the limit. If B2b has not run by the
  hash, this item is dropped from the pre-registration and run later as exploratory.
- **Patch-16 GPU references** also give a third and fourth seed's worth of JEPA at patch 16, which is
  added to the run-noise estimate for future comparisons.

### 9. D28: planted tests, before the hash **[CC: code and plants written 24 Nov; results in `docs/patch_grid/flags.md`]**

- Synthetic L curves for all 18 encoders and 30 answers at the observed seed spread, run through the
  identical scoring path. Plants must reach every step state, every group reading (including WORSE and
  LIMITED BELOW 8 px ONLY) and every objective-comparison state.
- **Power, stated plainly:** with 2 seeds per arm, report the smallest group-median Δ that reads GAIN
  with 80% probability at the observed s_g, and the probability that a true Δ of 0 reads NO CHANGE. If
  80% power needs Δ > 0.03 for G2, report it before the hash; the user then decides whether to add a
  third seed per arm.

---

## Notes outside the hashed section

- The renderer run shares the rental but is not part of this pre-registration.
- Cost from `docs/patch_ablation_cost.md`: full grid plus all three patch-16 controls is roughly 30–57
  H100-hours.
- Open for the user before the hash: (1) raw or cleaned as primary (draft: raw); (2) a third seed if the
  power check in §9 says 2 seeds cannot resolve the expected effects; (3) whether B2b stays in §8.
