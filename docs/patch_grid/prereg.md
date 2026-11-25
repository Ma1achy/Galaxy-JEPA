# Patch-size grid 16 → 8 → 4 (D11): pre-registration (HASHED 25 Nov 2026)

**Status.** Approved by the user and hashed on 25 Nov 2026, before any trained 8×8 or 4×4 checkpoint exists. The hash is the SHA-1 of the section from "## Pre-registration" to the next level-2 heading (trailing
newline): **`0e16af84ba079cfbd2d352194d4b193c14831bd2`**, also in `docs/patch_grid/prereg.sha1`, which the launcher, embedding and scoring programs check.
Design draft 24 Nov 2026 from B1, B1b, Part A and the cost estimate (`docs/patch_ablation_cost.md`); [CC] items completed 24–25 Nov by Claude Code; the user's decisions of 24 and 25 Nov applied (paired design, three seeds, raw
primary, B2b removed, |G| + b ≤ E, F7's divergence rule, no ρ gate, SAME in the E form, raw-only groups, the pinned code of record, the expected-UNRESOLVED statement). Ambiguities and mismatches are in `docs/patch_grid/flags.md`.
A change to anything in the section or to a pinned file after this is an amendment: the user's decision, dated and labelled, never an edit in place.

**What this was written from.** `docs/almagest/a1_visibility_report.md` (B1, B1b),
`docs/axes/a_report.md` (Part A), `docs/patch_ablation_cost.md`, and the chat record of the decisions
of 21 Nov. No 8×8 or 4×4 result exists. B2b has not run and is not part of this pre-registration (user, 24 Nov); it will be a later
exploratory analysis.

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
| JEPA | 16 (GPU, bf16) | 0, 1, 2 | GPU reference; device/precision control against Mac A1 (seed 0), A2 (seed 1), A3 (seed 2) |
| JEPA | 8 | 0, 1, 2 | |
| JEPA | 4 | 0, 1, 2 | |
| MoCo | 16 (GPU, bf16) | 0, 1, 2 | GPU reference; control against the Mac MoCo runs |
| MoCo | 8 | 0, 1, 2 | |
| MoCo | 4 | 0, 1, 2 | |
| MAE | 16 (GPU, bf16) | 0, 1, 2 | GPU reference; control against the Mac MAE runs |
| MAE | 8 | 0, 1, 2 | |
| MAE | 4 | 0, 1, 2 | |

27 trained encoders (3 objectives × 3 patch sizes × 3 seeds), the whole grid, all of it run. Untrained banks: 3 seeds (0, 1, 2) per patch size. The encoder architecture is the same for all
three objectives, so the nine banks are shared by them.
**[CC, completed 24 Nov]** All three objectives have a paired device control. By train_seed, the Mac
runs (`runs/`, hashed in the baselines comparison, `artifacts/baselines_comparison.md`) are:
- seed 0: A1 (`m_v2`), `moco_s0`, `mae_s0`;
- seed 1: A2 (`m_v2_s1`), `moco_s1`, `mae_s1`;
- seed 2: A3 (`m_v2_s2`), `moco_s2`, `mae_s2`.

The Mac runs trained on MPS without autocast (fp32), so device and precision change together. Seed 1 is kept although it is known to run high for
JEPA and MAE (`findings_summary.md`, the seed-1 puzzle). It was not removed after being seen.

### 2a. Order of the runs

Stage 1 (patch 16 and 8, all objectives, seeds 0–2) runs first, **as an engineering check only: no scientific decision is attached to it** (user, 25 Nov; the blind ρ gate of the 24 Nov draft is removed).
A fixed engineering checklist, outside this section, stops the run before stage 2 if any item fails. It reads throughput, memory, losses and utilisation, and never a probe, an AUC or a score. Otherwise stage 2 (the 4×4 arms) runs
whatever stage 1 looks like; §7's rules apply throughout.
- The scoring stays sealed until the whole grid is done: `artifacts/patch_grid_auc.py final` is the only program that forms a difference between patch sizes, and it refuses a partial grid.
- **ρ̂, descriptive.** After the final scoring, ρ̂ is reported per objective for 16 → 8 and 8 → 4: each seed's L minus that patch size's per-answer mean over seeds, correlated over all (answer, seed)
  pairs, raw. It carries no decision. It is printed beside its planted bias and SD (`plants_rho.json`): it is independent of the mean Δ (exactly) but pulled towards 0 and noisy on three seeds.

### 3. Recipe held fixed

- A1's recipe: 810,491 train stamps, batch 32, 101,308 steps (4 epochs), cosine over 253,270 steps stopped
  at 101,308. The same cache and images for every arm.
- bf16 with flash attention on every GPU arm, including the patch-16 references.
- **Masking fixed in pixels, not tokens:**
  - JEPA: target blocks and the 2.5 × R_petro box cover the same sky **in distribution** at every patch size, which is the intended meaning (user, 25 Nov). The block scales are fractions of the
    grid and the draws differ with it, so a galaxy does not get the same boxes at each size. The fraction of sky masked is within about 3% at patch 8; at patch 4 the context is 18% smaller than 16 × the patch-16 count.
  - MoCo: windows scale with pixels (10–14 tokens at 16; 20–28 at 8; 40–56 at 4).
  - MAE: masking in 16-px groups. The normalised pixel target is computed over the 16-px group, identical
    to patch 16, so only the encoder token size changes.
- Positional embeddings sin-cos from each grid; a new patch-embedding layer at each size.
- The 2.5 × R_petro masking box depends on galaxy size. It is held fixed, and this is recorded as a known
  way size enters training (B0).
- **Code of record for the GPU arms (written 24–25 Nov, before the hash):** `configs/patch_grid/<objective>_p<size>.yaml` are overlays that name a base
  config and only what differs; `artifacts/patch_grid_run.py` merges them, refuses any determining difference outside a fixed list, and runs one arm.
  MAE in 16-px groups is `objectives/mae_grouped.py` (at patch 16 it equals the Mac MAE; tested). The MoCo windows are set per size.
- **JEPA's patch-4 masks are precomputed on the rental's CPUs**, deterministically per train_seed, by `artifacts/patch_grid_masks.py` (the reference masker's own
  draws, seeded train_seed + step; independent of the number of worker processes). `check` compares 1,024 galaxies (32 batches of 32) with `MultiBlockMasker.sample` exactly
  and any difference refuses the arm. On the Mac's CPU, 1,024 galaxies at grid 64: 0 batches differ. The launcher won't start `jepa_p4` without them, and the precomputed
  masker also compares each batch's own weight-map digest, so a batch in a different order is refused.
- **Embedding code:** `artifacts/patch_grid_embed.py` embeds the 74,825-galaxy union (`docs/patch_grid/union_ids.npy`, SHA-1 pinned) with a trained arm, or with an untrained encoder
  (seeds 0, 1 and 2, patch 16, 8 and 4), by the ladder's recipe: fp32, no autocast, the frozen encoder's penultimate block mean-pooled. Patch 16 is re-embedded on the same box, so all arms and banks share one path.
  It reproduces the ladder's banked vectors on the Mac (max difference 1e-5; A2, MoCo s0 and untrained seed 1) and refuses a checkpoint whose patch size isn't the arm's.
- **From embeddings to readings:** `artifacts/patch_grid_auc.py` (`auc`, `valid`, `final`).
- **[CC, completed 25 Nov]** SHA-1 of the code of record (every repo file the training, mask, embedding and scoring programs import, the configs, the frozen evaluation set and the environment lock):

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
  | `src/galaxy_jepa/objectives/mae.py` | `bcb9b0f1ffbb16571a0c28f467a5e284ff65ff80` |
  | `src/galaxy_jepa/objectives/loop.py` | `4853f30a971c4eef8a488f69a491807e5c1ba18b` |
  | `src/galaxy_jepa/masking/blocks.py` | `2120144c2ba8fc83dee47b659013e0dd9efc84db` |
  | `src/galaxy_jepa/models/vit.py` | `c5f3d35daad4b4986f063e3657c96eda6415f592` |
  | `src/galaxy_jepa/callbacks/collapse.py` | `88133c65de5b29935008ff569c578237e31bfa68` |
  | `src/galaxy_jepa/harness.py` | `ad9d6a028dd2b7a00ac93697b57ced348a309cf3` |
  | `artifacts/almagest_visibility.py` (B1) | `9e1c0e112c272af91b246278ca6ecb38bec27018` |
  | `artifacts/almagest_axes.py` | `92235371250cdee52d9d29295ebce3a3e5f411c6` |
  | `src/galaxy_jepa/objectives/mae_grouped.py` | `e813951f399dab7256f1bcd3025f11b039cf1816` |
  | `src/galaxy_jepa/masking/precomputed.py` | `8129556fdc7ed890642a11e3ab368a85c2ad6319` |
  | `configs/patch_grid/jepa_p16.yaml` | `22582fe7ed0a8fff7d5f850908e730c29f900125` |
  | `configs/patch_grid/jepa_p4.yaml` | `2236ca1aee92032dc93e77dd2891ec170ef78601` |
  | `configs/patch_grid/jepa_p8.yaml` | `012f417584ae98d785b630b1b73543b67e54843d` |
  | `configs/patch_grid/mae_p16.yaml` | `163b793d635b23d1234aa5521267ecd516356007` |
  | `configs/patch_grid/mae_p4.yaml` | `926c2f16dee2639a606ada117f1352fd770ee748` |
  | `configs/patch_grid/mae_p8.yaml` | `908477f438cd2bdd1252df1ac22eaeda15342f55` |
  | `configs/patch_grid/moco_p16.yaml` | `515e47d12bede2b3d01d89d30d5c204981154f12` |
  | `configs/patch_grid/moco_p4.yaml` | `d3eb9535dd6138eaf121c9bb20e60aa1b83ea024` |
  | `configs/patch_grid/moco_p8.yaml` | `0cf54a066286f78a330ca8c825964f1c521e2f9b` |
  | `artifacts/patch_grid_run.py` | `05ae8252e3c28acbea80782ad12ac1eb461ba3b7` |
  | `artifacts/patch_grid_masks.py` | `7eb2669df188fcce758dc73ff289f86cc268aa11` |
  | `docs/patch_grid/union_ids.npy` | `d5cba4b7debce7693a8181b69688fc1d3957260b` |
  | `artifacts/patch_grid_embed.py` | `b0afb12ccbb9615674d3cda459824270aca22ec6` |
  | `artifacts/patch_grid_auc.py` | `ccd6cdf4d9cfb021e71f3691b714f85366295446` |
  | `artifacts/patch_grid_pins.py` | `79da170ab5358e422bc1962438f3627eccdf1814` |
  | `artifacts/patch_grid_score.py` | `91d4bb44672ef3c7540d1ed108368baa13b420f7` |
  | `artifacts/f0_preconditions.py` | `b5491b4f50f4630258928fdf0a32ec1b36a0cdf5` |
  | `artifacts/i2_sigreg_run.py` | `668858d2e1aabb455fdb1cd74e4328fc8e4fdf2b` |
  | `artifacts/patch_grid_frozen.py` | `7a603bcaeb6ae18cf860c89b7db16412e5ece451` |
  | `docs/patch_grid/frozen_deciles.json` | `f245f70a19d435a6e0b619e194f0a423c3df48d4` |
  | `docs/patch_grid/calibration.json` | `f0d8295ad7a325e20c39c6cbc91bbb94d0c7fc11` |
  | `artifacts/aligned_c2.py` | `029adeb522a3026603e36f7bf9187e3e9ae39c8a` |
  | `artifacts/almagest_spiral.py` | `1504990bdccc773a5205fa96928fa76f97ff36c6` |
  | `artifacts/f1_loader_bench.py` | `75376602ff0b50093ed2af4edaf59053541f3cbb` |
  | `artifacts/j1_preflight.py` | `b040e9ef8f004e90a1f49028a47b33eac81ef85d` |
  | `artifacts/j4_spread_controls.py` | `de65785469401de2a480dfdea227a5f3f3ddd14a` |
  | `artifacts/ladder_env.py` | `7ffe0f430ef55c67a744bd19aa873cd71872eb6c` |
  | `artifacts/m1_preflight.py` | `6cfece5853cdb5cceb22191388c92d901a379acc` |
  | `artifacts/o3_overfit_gate.py` | `805e9e7e9c0b63ffbe03619f8697d9788937356b` |
  | `artifacts/plate_export.py` | `c3853acb22b770927e5d0f04c2c0bfca9e24560a` |
  | `artifacts/probe_bank.py` | `b5b854f9011b0b9965e562ef8adeb3c4bddbbf57` |
  | `artifacts/r_nonlinear.py` | `9e7aeaa276a43bf219997e9d4a474036c4df57ca` |
  | `configs/effect_floor.json` | `e6b8b8ab7f0f0a9fea733a5d7d68135098b43cd1` |
  | `docs/patch_grid/power.json` | `e3c8fec9b37dbf5d07199f432a37629ff2ed7d22` |
  | `pyproject.toml` | `84010fd16b4d4b77f01c5a545891a7580df70877` |
  | `src/galaxy_jepa/__init__.py` | `8064d9dc4adb851e41666a791b334f2e096f1ba6` |
  | `src/galaxy_jepa/callbacks/__init__.py` | `fa0c2581e385898530ea53b3fd9ecb90410a2b3a` |
  | `src/galaxy_jepa/callbacks/checkpoint.py` | `2fe9da69d0e15b6e388266d22971856a6b9fbe14` |
  | `src/galaxy_jepa/core/__init__.py` | `e7676bfb7d8e0d034c5d62d439f8f234e42c2fc2` |
  | `src/galaxy_jepa/core/config.py` | `6122acb6d83c190b0dabf4f1502b90939cc36292` |
  | `src/galaxy_jepa/core/encoder.py` | `1b8c3d9db7aa2239b5c2dbf91b1e27924f20c2b9` |
  | `src/galaxy_jepa/core/gates.py` | `143f3486bf90f82b66c0bbd5dde8b331d3b10e5d` |
  | `src/galaxy_jepa/core/stopping.py` | `7f9de336332612637c554027172784b39b73e738` |
  | `src/galaxy_jepa/data/__init__.py` | `3ce9cc864f10bc6b1c97999f645303bf57bad841` |
  | `src/galaxy_jepa/data/bbox.py` | `70ba86cde9b7ec98262d9eeff7e1ed0599126e82` |
  | `src/galaxy_jepa/data/cache.py` | `be226049571c382c8f751f4b56045d5600ac76f7` |
  | `src/galaxy_jepa/data/dataset.py` | `4ec9db8e1e786fa7b65c31907dcda0a936086938` |
  | `src/galaxy_jepa/data/manifest.py` | `9ed250e094f424ce2776f455252574f5dfa8d8d5` |
  | `src/galaxy_jepa/data/metadata.py` | `8e74e5549379050920461b5df7adc56fd5cc521e` |
  | `src/galaxy_jepa/data/orchestrate.py` | `20dd45b5f34a47e08b7eee47af3a41661b2fcef5` |
  | `src/galaxy_jepa/data/sanity.py` | `8aec763bd6a5d94b8a28b372e636209c8f2c8719` |
  | `src/galaxy_jepa/data/sources.py` | `d5b72f8ace947ee669dc9cbde0064584de473257` |
  | `src/galaxy_jepa/data/splits.py` | `99a551bc6b82fbc41c64a913bf2ff7700fd3d143` |
  | `src/galaxy_jepa/data/transforms.py` | `a01467e344765bd61c7a5531962ea5a2b8e8ebc2` |
  | `src/galaxy_jepa/data/validity.py` | `508865975b2e53963c64a734d663f86b05fabe0b` |
  | `src/galaxy_jepa/eval/__init__.py` | `269c0f264eafc0ab6af96bc6447edbdad15b955c` |
  | `src/galaxy_jepa/eval/embed.py` | `ace06c8374ceaab74f3212a19a0f173a77140220` |
  | `src/galaxy_jepa/eval/figures.py` | `85609430d18e89a08b21044aa1d881b014d93a6b` |
  | `src/galaxy_jepa/masking/__init__.py` | `1aa49f94404e01deab0c6359d78ff774756a2e81` |
  | `src/galaxy_jepa/models/__init__.py` | `0e5e28470ded6de1d2aa8a3b0126ffc9996e9114` |
  | `src/galaxy_jepa/objectives/__init__.py` | `8b91f1929b038ae13c2b5f858a312f1d2fc9e8e0` |
  | `src/galaxy_jepa/objectives/sigreg.py` | `4e5247c680538a88938db1bb5e4659fd7d3f39a3` |
  | `src/galaxy_jepa/probing/__init__.py` | `bf5be5f11d46b76ba6c81ed4bbcebc2a07d86238` |
  | `src/galaxy_jepa/probing/config.py` | `dcf13b6a2b517412c9dfacefcf9d0d23387abafa` |
  | `src/galaxy_jepa/probing/controls.py` | `65ea392e90eab49a3a589f8783b6341e07083a42` |
  | `src/galaxy_jepa/probing/entanglement.py` | `1cc5f7e36ba08b4c5fb1d0f151db4e421a623e56` |
  | `src/galaxy_jepa/probing/extract.py` | `f17ea69f14371d55ccc773c5b8baf0921547ae87` |
  | `src/galaxy_jepa/probing/floor.py` | `9a0c264481787927a025539c8e37d086b71d322e` |
  | `src/galaxy_jepa/probing/gates.py` | `742464ff05a0a200bebec62a7793e22f0f6533cc` |
  | `src/galaxy_jepa/probing/geometry.py` | `ea0d3da48684935179832bff24ec5ad0c4e77f04` |
  | `src/galaxy_jepa/probing/ladder.py` | `94e633490633d04c63f250448ed98d1426cf51db` |
  | `src/galaxy_jepa/probing/logistic.py` | `a632dc90383400fdd6dacc291117c2932762b5a9` |
  | `src/galaxy_jepa/probing/matching.py` | `a5ccfe0bf8d35e6146e66fb92a1dd79e89cf9b52` |
  | `src/galaxy_jepa/probing/mlp.py` | `bf2317526babbf6880c3cf1994d8c7b8f1362088` |
  | `src/galaxy_jepa/probing/nulls.py` | `10c7f06d43bb3a34bdb569125786afc2780e885d` |
  | `src/galaxy_jepa/probing/orientation.py` | `8fd11e71a6a68853b38a0d94cbb66f87bbdfdf30` |
  | `src/galaxy_jepa/probing/run.py` | `4ac720c4c1799a8cd69f0fc4c878c5eb57e3c483` |
  | `src/galaxy_jepa/probing/schemes.py` | `eccc60a1ac51074391a457100c251a6284ec89e1` |
  | `src/galaxy_jepa/probing/uncertainty.py` | `d383b845a4c8fa8c364fe9132e877ad00df64705` |
  | `uv.lock` | `2f3ee623460be463c725f0d576aa24bd6468ab6e` |
  | `docs/almagest/a1_visibility_b1.json` | `a9900b3159943a824412512c88c78041d7fa46d9` |

  The launcher, the embedding program and the scoring programs check this section's own SHA-1 (`docs/patch_grid/prereg.sha1`) and every file in this table before real work, and refuse on a mismatch.

### 4. Evaluation set, frozen before any training

- Probes: the Scheme-1 L2-logistic probes, the ladder's recipe, fitted per encoder on the frozen train split of
  39,997 galaxies (the ladder's identical split; user, 25 Nov); evaluated on the frozen test split.
- **Resolution deciles:** B1's decile edges in R_petro_r / psfWidth_r, per answer, taken from A1's test
  galaxies and frozen. The same galaxies sit in the same decile for every arm.
- **Answers:** the 30 STRONG or SEPARATE answers from relabel2, in three groups.
- **Groups (user, 25 Nov): defined from A1's raw B1 record alone** (`a1_visibility_b1.json`), by this rule, applied once and fixed. A decile *clears* if A1's raw AUC exceeds the untrained maximum by more than
  0.05; the *knee* is the first decile that clears (both as B1). The first match applies:

  | # | raw knee and raw deciles clearing | group |
  |---|---|---|
  | 1 | knee at decile 1, and at least 7 deciles clear | **G1** |
  | 2 | knee at decile 4 or earlier, and at least 4 deciles clear | **G2** |
  | 3 | otherwise (knee at 5 or later, no knee, or too few clearing) | **G3** |

  7 is the fewest raw clearing deciles among B1's own first-order answers (odd no), and 4 the fewest among its "knee at 3–4" answers (obvious bulge). On A1's record the counts never decide a case: every
  knee-1 answer clears at least 7 deciles and every knee-2–4 answer at least 4. The scoring code reproduces the lists from the JSON by the rule and asserts it.
  - **G1, first-order (12):** smooth, features, edge-on yes, edge-on no, completely round, cigar, odd yes, odd no, no bulge, no bar, no spiral, ring.
  - **G2, structure (9):** bar, spiral, lens/arc, obvious bulge, dominant bulge, in between, loose winding, 2 arms, bulge-shape no bulge.
  - **G3, fine (9):** dust lane, disturbed, irregular, other, merger, bulge rounded, tight winding, just noticeable, can't tell (arms).
  - Against the draft, "no bar", "no spiral" and "ring" move from G2 to G1, and "dominant bulge" from G3 to G2. The groups don't change after any new result.
- **Ordinal axes (Part A):** arm count and winding, scored as Kendall τ of the ordinal score with
  E[category] within the same deciles.
- **[CC, completed 24 Nov]** As implemented, the Scheme-1 ladder probes use **raw** embeddings: cleaning exists
  only in the Almagest code (`artifacts/almagest_spiral.py`), which B1 and Part A used. Either choice runs, and the
  scoring code reports both. **Decided (user, 24 Nov): raw is primary** (the decile already controls apparent size, and B1 showed the cleaner can distort
  within-decile comparisons, e.g. no bulge), cleaned reported beside it.

### 5. Primary statistic

For encoder e (objective o, patch p, seed s) and answer a:

- **Low-resolution margin** L(e, a) = the mean over deciles 1–5 of [AUC(e, a, d) − max over untrained
  seeds at the same o and p of AUC(u, a, d)].
  - Deciles 1–5 are where B1 placed the knees for G2 and G3, so a resolution gain shows up here first.
  - Margins are taken against the untrained encoders **of the same patch size**, so a change in what an
    untrained 4×4 ViT already reads is not credited to training.
- **Paired step change** for objective o, from patch p to patch q, seed s, answer a: Δ_s(a) = L(o, q, s, a) − L(o, p, s, a)
  (user, 24 Nov). Seed s at q is compared only with seed s at p.
- **Group statistic:** G_s = the median of Δ_s(a) over the group's answers; G = the mean of G_s over the seeds valid at both sizes
  (n of them; three unless an arm is VOID).
- **Paired noise:** S = the larger of the two steps' SDs of G_s over seeds, for that objective and group (the larger
  noise sets the bar, as max(S_A, S_X) does in the baselines comparison).
- **Bar:** b = max(SESOI/2, K_n · S / √n). K_n is calibrated by Monte Carlo, not chosen: the smallest value on a 0.05 grid whose
  false GAIN-or-LOSS rate and false DIFFERS rate are at most 2.5% in every noise regime (the observed patch-16 noise of every
  objective and group, raw and cleaned, at half, one and twice its size; with one seed in three running far off; with the
  run-level seed effect shared between patch sizes fully, half or not at all): **K₃ = 3.2, K₂ = 5.6** (recalibrated 25 Nov on the new groups, unchanged; n = 2 only if a seed is VOID;
  `docs/patch_grid/calibration.json`). Baselines: K = 3.0.
- **Secondary, descriptive:** the full-sample existence AUC per answer and arm; the knee per answer,
  defined on the CI as the first decile d where deciles d and d+1 both have their bootstrap 95% CI lower
  bound above the untrained maximum (1,000 galaxy resamples within decile), or NONE; and the ordinal τ
  per decile for arm count and winding.

### 6. States (D27: every outcome has one; first match applies)

Constants: **SESOI = 0.02** AUC and **E = SESOI = 0.02** (the equivalence bound). Bars: §5.

**Per group, per objective, per step (16 → 8 and 8 → 4):**

| # | condition | step state |
|---|---|---|
| 1 | fewer than 5 answers have a defined L at both sizes in every valid seed, or fewer than 2 seeds are valid at both sizes | INSUFFICIENT |
| 2 | G > b, and at least 2/3 of the group's answers have mean Δ(a) > 0 | GAIN |
| 3 | G < −b, and at least 2/3 have mean Δ(a) < 0 | LOSS |
| 4 | \|G\| + b ≤ E, with E = SESOI = 0.02 (user, 24 Nov; replaces \|G\| ≤ 0.005 and b ≤ SESOI) | NO CHANGE |
| 5 | otherwise (with the label "seeds disagree in sign" if some G_s ≥ +SESOI/2 and some ≤ −SESOI/2) | UNRESOLVED |

E is not to be widened; a group (G1 in particular) may read UNRESOLVED for that reason. GAIN and LOSS are tested before NO CHANGE, so a step with |G| > b is never NO CHANGE.

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

**Expected before any run (user, 25 Nov), so that a wall of UNRESOLVED is not read as news.** ρ is the share of a seed's run-level effect common to two patch sizes; it is unknown. Δ80 is the group-median Δ that reads GAIN
with 80% probability on three paired seeds at the observed patch-16 noise (`power.json`); P(NC) is the probability that a true Δ of 0 reads NO CHANGE. Cells are ρ = 0 / 0.5 / 1, raw.

| | Δ80 | P(NC) |
|---|---|---|
| JEPA G1 | 0.124 / 0.090 / 0.024 | 0.00 / 0.00 / 0.48 |
| JEPA G2 | 0.068 / 0.050 / 0.021 | 0.01 / 0.04 / 0.65 |
| JEPA G3 | 0.037 / 0.027 / 0.013 | 0.14 / 0.36 / 1.00 |
| MAE G1 | 0.164 / 0.117 / 0.035 | 0.00 / 0.00 / 0.16 |
| MAE G2 | 0.076 / 0.055 / 0.014 | 0.01 / 0.03 / 0.98 |
| MAE G3 | 0.047 / 0.034 / 0.014 | 0.06 / 0.18 / 0.99 |
| MoCo G1 | 0.015 / 0.014 / 0.013 | 0.92 / 0.99 / 1.00 |
| MoCo G2 | 0.014 / 0.014 / 0.013 | 0.98 / 1.00 / 1.00 |
| MoCo G3 | 0.013 / 0.013 / 0.013 | 1.00 / 1.00 / 1.00 |

- **At low ρ, JEPA's and MAE's Q1 readings are expected to be UNRESOLVED** (or, for a large gain, PATCH-LIMITED; FLOOR UNRESOLVED): Δ80 is 0.037–0.124 for JEPA and 0.047–0.164 for MAE against E = 0.02, and a true null reads NO CHANGE at most
  14% and 6% of the time (all at ρ = 0). **MoCo's Q1 is the reading expected to resolve** (Δ80 ≤ 0.015, P(NC) ≥ 0.92, at any ρ).
- **Most Q2 readings are expected to be UNRESOLVED at low ρ:** SAME is reached by a true null at most 5% of the time at ρ = 0 (14% at ρ = 0.5), and DIFFERS needs a difference of 0.051–0.251 in group-median Δ at ρ = 0.

  | | Δ80 for DIFFERS | P(SAME) at a true D of 0 |
  |---|---|---|
  | JEPA–MAE G1 | 0.251 / 0.175 / 0.049 | 0.00 / 0.00 / 0.01 |
  | JEPA–MAE G2 | 0.116 / 0.085 / 0.029 | 0.00 / 0.00 / 0.25 |
  | JEPA–MAE G3 | 0.069 / 0.050 / 0.018 | 0.00 / 0.01 / 0.84 |
  | JEPA–MoCo G1 | 0.174 / 0.122 / 0.033 | 0.00 / 0.00 / 0.21 |
  | JEPA–MoCo G2 | 0.094 / 0.068 / 0.029 | 0.00 / 0.01 / 0.32 |
  | JEPA–MoCo G3 | 0.051 / 0.037 / 0.015 | 0.05 / 0.14 / 0.96 |
  | MAE–MoCo G1 | 0.233 / 0.163 / 0.048 | 0.00 / 0.00 / 0.05 |
  | MAE–MoCo G2 | 0.104 / 0.075 / 0.019 | 0.00 / 0.01 / 0.78 |
  | MAE–MoCo G3 | 0.064 / 0.047 / 0.018 | 0.02 / 0.06 / 0.86 |

- An UNRESOLVED says the design couldn't tell. It is not evidence for NO CHANGE, SAME, H_patch or H_psf.

**Objective comparison (Q2), per group and step:** D = the mean over seeds of G_s(o1) − G_s(o2), paired by seed.
Bar: max(SESOI/2, K_n · max(S_o1, S_o2) · √(2/n)), as the baselines comparison's family bar.

| # | condition | state |
|---|---|---|
| 1 | either objective INSUFFICIENT | INSUFFICIENT |
| 2 | \|D\| > bar | DIFFERS (sign reported) |
| 3 | \|D\| + b ≤ E, with b the bar above (user, 25 Nov) | SAME |
| 4 | otherwise | UNRESOLVED |

Pairs: JEPA–MAE, JEPA–MoCo, MAE–MoCo. Predicted (weakly): MAE > JEPA > MoCo in Δ for G2 and G3 at 16 → 8.

**Device/precision control (not a reading on Q1 or Q2):**
- For each objective with a Mac counterpart, the per-answer full-sample AUC of the GPU patch-16 run against
  the Mac run with the same train_seed (the Mac record is `runs/almagest/patch_grid/spread.json`, SHA-1 `d4d05a2b072f91d2188524d9527a0235f53f1c15`, checked by the scoring code).
- **DEVICE EFFECT** if the median absolute paired difference exceeds the Mac A1–A3 seed spread for the
  same answers; otherwise **NO DEVICE EFFECT**.
- Either way, every Q1 and Q2 comparison uses the GPU patch-16 runs as the reference. The Mac runs are
  never mixed into a step change.

### 7. Validity, exclusions and what stops the run

- **Collapse.** The effective-rank collapse monitor runs on every arm at the project's existing bar
  (**[CC, completed 24 Nov; user: the one used in A1–A3's training runs]**, `collapse_floor` in `configs/pretrain_v2.yaml`, passed to the trainer by
  `artifacts/m2_long_run.py` line 361: min_effective_rank 2.5 with a grace fraction of 0.10 of the schedule, hard floor 2.0 after step 100, three
  consecutive readings. It was re-derived at D17, made active at D21, unchanged since `11b9067` (the first A1–A3 launch) and lives in
  `src/galaxy_jepa/callbacks/collapse.py`. **JEPA arms use it whole (A1–A3's bar). The hard floor (2.0, F8) gates every objective; for MoCo and MAE the soft floor is reported and does not gate** (user, 25 Nov, as coded)). An arm that collapses is VOID. It is rerun once with the same seed and recipe (a diverged arm too). A
  second collapse or divergence leaves it VOID, and the groups are scored without it, marked. Seeds are never swapped
  or chosen.
- **Divergence (F7's rule; user, 25 Nov).** An arm has diverged if its loss is non-finite at any step, or its final 500-step mean loss is more than 50% above its lowest 500-step rolling mean
  (`artifacts/baselines.md` F7; `patch_grid_auc.py valid`, from the training traces). It is handled as collapse: VOID, rerun once with the same seed and recipe. The 50% line stays as it is (user, 25 Nov).
  - **Why the drafted rule was wrong.** It called a loss above the run's step-1,000 value after step 20,000 a divergence. JEPA predicts in latent space against an EMA target: its loss falls fast in the first
    thousand steps and then rises as the target moves (A1's 500-step mean peaks at 0.26 and ends at 0.114), so the step-1,000 value is a dip and not a reference. Applied to A1, A2 and A3 the drafted rule would have
    voided all three: they are above their step-1,000 loss for 92.6%, 99.5% and 100% of the steps after 20,000. For MoCo, step 1,000 sits on the start-up hump, so the rule could hardly have fired.
  - F7's rule fires on none of the nine Mac runs (largest rise 36%, A2) and on planted rises of 51% and 60% and on a non-finite loss; not on a planted 49% rise or a JEPA-like early dip (`plants_divergence.json`).
- **Timing smoke.** A 200-step timed smoke per arm on the rented box comes first. It may change the GPU
  type or the order of runs, never the recipe.
- **Cache parity.** The launcher and the embedding program refuse a real run unless `cache_verified.json` (written by the pull after every raw SHA-1 passed, `docs/shipping_plan.md`) names the manifest whose SHA-1
  the user passes by hand and the pipeline hash the run resolves to.
- **Blindness.**
  - Decile edges, group membership, probes' recipe and untrained banks are frozen before any trained
    checkpoint exists.
  - The scoring code is pinned by SHA-1 before any trained 8 × 8 or 4 × 4 record is opened.
  - No L, Δ, G or state is computed before all 27 arms (VOID ones named) and the nine untrained banks are scored; `patch_grid_auc.py final` refuses a partial grid. Training traces and the timed smoke are read freely: they hold no score.
  - The AUC records are listed in an `outputs.sha1` manifest before scoring (`patch_grid_auc.py manifest`, written once); `final` refuses without it or if a record has changed.

### 8. Secondary analyses (descriptive; no states)

- **Labels per arm:** the 37-axis relabel2 pipeline (STRONG / SEPARATE / PROXY / WEAK / NOISE) on every
  trained arm. In particular, do the NOISE answers (boxy bulge, 3 arms, more than 4 arms) become stable
  at 8 × 8 or 4 × 4?
- **Part A on every arm:** the ordinal-versus-free shares and gaps for arm count, winding, bulge and
  roundness.
- **Code for these, and for §5's CI knee and ordinal τ, is written and pinned by SHA-1 before the final scoring opens** (user, 25 Nov); it is not part of this hash.
- **Patch-16 GPU references** also give three more patch-16 runs per objective (device and precision differ from the Mac's, and the
  train_seeds repeat A1–A3's, so these are not new independent seeds: `flags.md` 8).

### 9. D28: planted tests, before the hash **[CC: written and run 24–25 Nov; results in `docs/patch_grid/flags.md`]**

- Synthetic L curves for all 27 encoders and 30 answers at the observed seed spread, run through the
  identical scoring path. Plants reach every step state, every group reading (including WORSE at both steps and LIMITED
  BELOW 8 px ONLY), every objective-comparison state (SAME in its E form) and both device states (`plants.json`).
- The group lists are reproduced from B1's record by §4's rule, and asserted (`patch_grid_score.py check`).
- The divergence rule's plants (`plants_divergence.json`), and ρ̂'s bias and SD (`plants_rho.json`, descriptive).
- `final` on a planted grid built from the Mac's own patch-16 records returns the planted reading for every cell, refuses a missing arm, and scores a named VOID arm as marked.
- **Power, stated plainly:** three seeds, the paired bar: the smallest group-median Δ that reads GAIN with 80% probability, the probability
  that a true Δ of 0 reads NO CHANGE, and the same for DIFFERS and SAME between objectives, per objective (pair) and group, at ρ = 0 and 1 (`docs/patch_grid/power.json`, summarised in §6 and `flags.md` D).

---

## Notes outside the hashed section

- The renderer run shares the rental but is not part of this pre-registration.
- Cost from `docs/patch_ablation_cost.md`: the full grid (27 encoders), 45–87 H100-hours.
- Open items, all reported and none reconciled: `docs/patch_grid/flags.md`.

**Stage 1 engineering checklist (user, 25 Nov; outside the hash). Any failure stops the run before stage 2.** Code: `artifacts/patch_grid_stage1.py`. It is run per arm type (nine, seed 0, 2,000 steps at the real batch, one at a time on an idle GPU); items 1, 2 and 4 are also read from stage 1's real arms.

1. **Throughput and peak memory** per arm against the cost doc, ± 30%.
2. **Loss** finite, and falling through the first 2,000 steps: mean(steps 1,501–2,000) < mean(steps 1–500). A step-by-step or window-by-window fall would fail every JEPA and MoCo run on the Mac
   (JEPA dips by step 1,000 and rises; MoCo has a start-up hump); all nine Mac runs pass this form.
3. **Flash attention active:** the encoder, predictor and decoder run forward and backward in bf16 under a kernel context that allows flash only, so it raises if flash can't serve them.
4. **Loader:** GPU utilisation above 85% (nvidia-smi, 1-s samples, after step 200).
5. **Checkpoint round trip:** save, reload, identical embedding on 64 galaxies.
