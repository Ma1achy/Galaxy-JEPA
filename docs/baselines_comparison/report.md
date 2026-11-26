# D12 baselines comparison: write-up

Scored 15 Nov 2026 under the pre-registration hashed at `b021e868` (`artifacts/baselines_comparison.md`,
section "Pre-registration"). Code of record `baselines_comparison.py` `2e87a800` (commit `d30ac14`). Score
record `docs/baselines_comparison/baselines_score.json`, SHA-1 `45c3d74f…`, byte-identical to
`artifacts/out/baselines_score.json`. Every number below is read from that record, except the per-seed
arm means in §1.1, which are recomputed from the trajectory files (they reproduce the record's S values
exactly). **Blind of M:** no M or O2 record was read, and nothing here compares with M.

**Status: exploratory.** Every encoder in this comparison, A1–A3 and the six baselines, is a
`smoke: true` run (`baselines.md` F1: "The D12 comparison on A1 is exploratory until a headline run
exists"). The states below are the hashed rules applied as written; none is a headline result.

**Preconditions.** P1 passes for A1, A2 and A3 (identity and end-to-end). P3: 72 files checked against
their recorded SHA-1s, 0 mismatches; every refitted held-out AUC equalled its trajectory file's `auc`.
Both ladders complete (36/36 and A1–A3's), every ladder file matching the runner's recorded SHA-1.
**VOIDs: 0 of 156 states.**

## 1. Primary: two family states at 4 epochs (step 101,308)

D is baseline minus JEPA, mean over the 35 answers and three seed pairs. CI is the shared galaxy
bootstrap, 97.5%. Bar b = K × max(S_A, S_X) × √(2/3), K 3.0.

| objective | **state** | D̄ | 97.5% CI | bar b | S_A | S_X | pair means (s0, s1, s2) | flags |
|---|---|---|---|---|---|---|---|---|
| MoCo vs JEPA | **UNRESOLVED** (underpowered: b > SESOI; also large but not resolved) | +0.0419 | [+0.0402, +0.0434] | 0.0404 | 0.0165 | 0.0009 | +0.052, +0.023, +0.050 | SPREADS DIFFER (run 18.3×, answer-level 4.4×) |
| MAE vs JEPA | **UNRESOLVED** (underpowered: b > SESOI) | +0.0044 | [+0.0031, +0.0059] | 0.0491 | 0.0165 | 0.0200 | +0.006, +0.008, −0.001 | — |

- **Neither primary state carries a claim.** By the Reading section, "the JEPA reads morphology
  better than X" is claimed only on WORSE, and "no different" only on SAME. Neither was reached.
- **Why both are underpowered: the arms' own run-level seed spread.** SESOI is 0.02, and the bar
  exceeds it in both comparisons, so SAME is unreachable whatever the CI (§C's UNRESOLVED,
  "underpowered"). For MoCo the bar is set by the JEPA's spread (S_A 0.0165, against MoCo's 0.0009);
  for MAE by MAE's own (0.0200). Both lie above every run-level spread in §F's calibration grid
  (τ ≤ 0.015). §F's false-call, power and SPLIT rates are therefore extrapolations here, not
  measurements, and are quoted as such below.
- **MoCo is a near miss, and it is reported as UNRESOLVED.** BETTER needs the CI wholly above +b. The
  CI's lower end is 0.0402 against a bar of 0.0404: short by 0.0002. Two of §C's UNRESOLVED reasons
  apply (underpowered; large but not resolved, as |D̄| ≥ SESOI). The record names the first, which
  the code checks first. All three MoCo pair means are positive at every epoch (§2).
- **Chance counts beside the states.** Under no effect, a directional family call is expected at
  most 2.5% per objective (§F, equal spreads; 3.6% with a 3× noisier arm); no directional call was
  made. Neither state is SPLIT, so no SPLIT null rate attaches (for reference: 0–8% per objective at
  equal spreads, 22% at 3×).

### 1.1 Where the seed spread comes from (reported, no state; **exploratory**, noticed after scoring)

Per-seed arm means m_s = mean over the 35 answers of AUC, from the trajectory files:

| epoch | JEPA A1, A2, A3 | S_A | MoCo s0, s1, s2 | S_MoCo | MAE s0, s1, s2 | S_MAE |
|---|---|---|---|---|---|---|
| 0.5 | 0.715, 0.734, 0.714 | 0.011 | 0.732, 0.735, 0.733 | 0.002 | 0.700, 0.737, 0.692 | 0.024 |
| 1 | 0.716, 0.738, 0.719 | 0.012 | 0.744, 0.744, 0.742 | 0.001 | 0.692, 0.733, 0.687 | 0.025 |
| 2 | 0.715, 0.745, 0.725 | 0.015 | 0.751, 0.751, 0.749 | 0.001 | 0.702, 0.736, 0.703 | 0.019 |
| 4 | 0.704, 0.733, 0.705 | 0.017 | 0.757, 0.757, 0.755 | 0.001 | 0.710, 0.742, 0.705 | 0.020 |

Training seed 1 (A2, MAE s1) sits about 0.03 above seeds 0 and 2 for both the JEPA and MAE at every
epoch; MoCo's three seeds agree to within 0.002. The seed fixes both the encoder's initial weights
(byte-identical across objectives for the same seed) and the data order, so a seed-level effect
shared by two objectives is possible; it is not tested here. This pattern was seen only after
scoring and has no state. It is not a reason to drop a seed: the rule takes the seeds as they are.

## 2. Trajectory (secondary)

The family state at 0.5, 1, 2 and 4 epochs, read into one trajectory state per objective (§C):

| objective | 0.5 ep | 1 ep | 2 ep | 4 ep | **trajectory** |
|---|---|---|---|---|---|
| MoCo vs JEPA | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | **CONSISTENT** |
| MAE vs JEPA | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | **CONSISTENT** |

Here CONSISTENT means the same state at all four epochs: consistently unresolved (underpowered), not a
consistent direction. No trajectory is MIXED, so the 11 Nov reporting rule for SAME/UNRESOLVED mixes
does not arise. D̄ at each epoch (reported, no state; 97.5% CI):

| objective | 0.5 ep | 1 ep | 2 ep | 4 ep |
|---|---|---|---|---|
| MoCo vs JEPA: D̄ [CI], b | +0.012 [+0.011, +0.014], 0.028 | +0.019 [+0.017, +0.020], 0.029 | +0.022 [+0.021, +0.023], 0.037 | +0.042 [+0.040, +0.043], 0.040 |
| MoCo pair means | +0.017, +0.001, +0.019 | +0.027, +0.006, +0.024 | +0.036, +0.006, +0.024 | +0.052, +0.023, +0.050 |
| MAE vs JEPA: D̄ [CI], b | −0.012 [−0.013, −0.010], 0.058 | −0.021 [−0.022, −0.019], 0.062 | −0.015 [−0.016, −0.013], 0.047 | +0.004 [+0.003, +0.006], 0.049 |
| MAE pair means | −0.015, +0.003, −0.022 | −0.025, −0.006, −0.032 | −0.013, −0.009, −0.022 | +0.006, +0.008, −0.001 |
| MoCo vs MAE (MAE − MoCo; reported, no state) | −0.024 [−0.025, −0.023] | −0.040 [−0.041, −0.038] | −0.037 [−0.038, −0.035] | −0.038 [−0.039, −0.036] |

## 3. Per-answer states at 4 epochs (secondary)

The family table on each answer's D_j, with b_j and a 95% CI. No state is pooled across answers.

| objective | BETTER | WORSE | SPLIT | NEGLIGIBLE | SAME | UNRESOLVED | expected by chance (§C) |
|---|---|---|---|---|---|---|---|
| MoCo vs JEPA | **9** | 0 | 0 | 0 | 0 | 26 (underpowered) | ≤ 0.9 directional; SPLIT 0.4–6 |
| MAE vs JEPA | 0 | 0 | 0 | 0 | 0 | 35 (underpowered) | ≤ 0.9 directional; SPLIT 0.4–6 |

- **MoCo: 9 BETTER against ≤ 0.9 expected by chance** (the calibrated per-answer rate, ≤ 2.5% of 35,
  measured at equal spreads; the spreads here differ 18×, outside the grid). The nine:
  edge-on yes and no (`t02` a04, a05: complementary answers of one question, so not independent),
  cigar-shaped, completely round and in-between (`t07`), no bulge (`t05` a10), loose winding (`t10`
  a30), and odd yes and no (`t06` a14, a15, again complementary).
- Every per-answer bar exceeds SESOI (MoCo 0.042–0.068; MAE 0.052–0.106), so SAME is unreachable on
  every answer, and every non-directional answer reads UNRESOLVED (underpowered).
- Reported, no state: MoCo's D_j is positive on all 35 answers, with the 95% CI above 0 on 34 (`t10`
  a29, medium winding, crosses 0). MAE's D_j ranges from −0.009 to +0.022.
- No answer is SPLIT, so no per-answer SPLIT null rate attaches.

Full per-answer readings (D_j, 95% CI, bar, the three pair differences):

| answer | MoCo D_j [95% CI] | bar b_j | pair means | MoCo state | MAE D_j [95% CI] | bar b_j | pair means | MAE state |
|---|---|---|---|---|---|---|---|---|
| `t01_smooth_or_features_a01_smooth` | +0.023 [+0.021, +0.025] | 0.042 | +0.030, +0.010, +0.028 | UNRESOLVED | +0.007 [+0.005, +0.009] | 0.052 | +0.011, +0.003, +0.006 | UNRESOLVED |
| `t01_smooth_or_features_a02_features_or_disk` | +0.022 [+0.020, +0.024] | 0.042 | +0.029, +0.010, +0.028 | UNRESOLVED | +0.007 [+0.005, +0.009] | 0.054 | +0.011, +0.003, +0.007 | UNRESOLVED |
| `t02_edgeon_a04_yes` | +0.115 [+0.111, +0.120] | 0.043 | +0.144, +0.074, +0.128 | **BETTER** | +0.018 [+0.014, +0.022] | 0.056 | +0.016, +0.046, -0.009 | UNRESOLVED |
| `t02_edgeon_a05_no` | +0.115 [+0.111, +0.120] | 0.042 | +0.144, +0.070, +0.132 | **BETTER** | +0.018 [+0.014, +0.022] | 0.054 | +0.014, +0.043, -0.004 | UNRESOLVED |
| `t03_bar_a06_bar` | +0.022 [+0.017, +0.028] | 0.044 | +0.028, +0.005, +0.034 | UNRESOLVED | -0.004 [-0.008, +0.000] | 0.057 | -0.010, -0.005, +0.002 | UNRESOLVED |
| `t03_bar_a07_no_bar` | +0.021 [+0.016, +0.027] | 0.043 | +0.029, +0.004, +0.030 | UNRESOLVED | -0.003 [-0.008, +0.001] | 0.055 | -0.005, -0.008, +0.002 | UNRESOLVED |
| `t04_spiral_a08_spiral` | +0.012 [+0.010, +0.014] | 0.042 | +0.019, +0.003, +0.013 | UNRESOLVED | +0.001 [-0.001, +0.003] | 0.053 | +0.004, +0.000, -0.002 | UNRESOLVED |
| `t04_spiral_a09_no_spiral` | +0.015 [+0.013, +0.017] | 0.042 | +0.022, +0.005, +0.017 | UNRESOLVED | +0.001 [-0.000, +0.003] | 0.052 | +0.006, +0.001, -0.002 | UNRESOLVED |
| `t05_bulge_prominence_a10_no_bulge` | +0.066 [+0.062, +0.070] | 0.046 | +0.073, +0.044, +0.082 | **BETTER** | +0.022 [+0.019, +0.026] | 0.061 | +0.006, +0.049, +0.012 | UNRESOLVED |
| `t05_bulge_prominence_a11_just_noticeable` | +0.036 [+0.033, +0.039] | 0.042 | +0.042, +0.026, +0.040 | UNRESOLVED | +0.013 [+0.011, +0.016] | 0.053 | +0.012, +0.019, +0.008 | UNRESOLVED |
| `t05_bulge_prominence_a12_obvious` | +0.037 [+0.035, +0.040] | 0.042 | +0.041, +0.028, +0.042 | UNRESOLVED | +0.011 [+0.009, +0.013] | 0.053 | +0.006, +0.028, -0.000 | UNRESOLVED |
| `t05_bulge_prominence_a13_dominant` | +0.044 [+0.038, +0.051] | 0.047 | +0.048, +0.034, +0.050 | UNRESOLVED | +0.012 [+0.007, +0.017] | 0.064 | +0.010, +0.018, +0.007 | UNRESOLVED |
| `t06_odd_a14_yes` | +0.055 [+0.049, +0.062] | 0.045 | +0.072, +0.032, +0.062 | **BETTER** | -0.003 [-0.008, +0.003] | 0.059 | +0.003, +0.004, -0.015 | UNRESOLVED |
| `t06_odd_a15_no` | +0.056 [+0.050, +0.062] | 0.043 | +0.073, +0.034, +0.062 | **BETTER** | -0.004 [-0.009, +0.002] | 0.055 | +0.002, +0.003, -0.016 | UNRESOLVED |
| `t07_rounded_a16_completely_round` | +0.103 [+0.099, +0.107] | 0.042 | +0.130, +0.065, +0.114 | **BETTER** | -0.008 [-0.011, -0.005] | 0.053 | -0.007, +0.004, -0.021 | UNRESOLVED |
| `t07_rounded_a17_in_between` | +0.051 [+0.047, +0.056] | 0.042 | +0.067, +0.019, +0.068 | **BETTER** | -0.009 [-0.013, -0.005] | 0.052 | -0.000, -0.020, -0.006 | UNRESOLVED |
| `t07_rounded_a18_cigar_shaped` | +0.117 [+0.113, +0.121] | 0.043 | +0.148, +0.070, +0.132 | **BETTER** | +0.016 [+0.013, +0.019] | 0.056 | +0.013, +0.036, -0.001 | UNRESOLVED |
| `t08_odd_feature_a19_ring` | +0.028 [+0.023, +0.034] | 0.044 | +0.036, +0.013, +0.036 | UNRESOLVED | -0.006 [-0.010, -0.001] | 0.057 | -0.013, +0.006, -0.010 | UNRESOLVED |
| `t08_odd_feature_a20_lens_or_arc` | +0.010 [+0.004, +0.015] | 0.048 | +0.011, +0.002, +0.015 | UNRESOLVED | -0.002 [-0.006, +0.003] | 0.065 | -0.003, +0.003, -0.005 | UNRESOLVED |
| `t08_odd_feature_a21_disturbed` | +0.007 [+0.001, +0.013] | 0.044 | +0.011, -0.004, +0.014 | UNRESOLVED | -0.003 [-0.007, +0.001] | 0.057 | -0.001, -0.008, -0.002 | UNRESOLVED |
| `t08_odd_feature_a22_irregular` | +0.022 [+0.019, +0.026] | 0.044 | +0.024, +0.014, +0.029 | UNRESOLVED | -0.001 [-0.004, +0.002] | 0.057 | -0.005, +0.001, +0.002 | UNRESOLVED |
| `t08_odd_feature_a23_other` | +0.011 [+0.007, +0.014] | 0.042 | +0.013, +0.007, +0.012 | UNRESOLVED | +0.002 [-0.001, +0.004] | 0.054 | +0.001, +0.003, +0.000 | UNRESOLVED |
| `t08_odd_feature_a24_merger` | +0.036 [+0.028, +0.043] | 0.047 | +0.044, +0.028, +0.036 | UNRESOLVED | +0.004 [-0.001, +0.009] | 0.064 | +0.010, +0.008, -0.007 | UNRESOLVED |
| `t08_odd_feature_a38_dust_lane` | +0.054 [+0.037, +0.072] | 0.066 | +0.058, +0.026, +0.079 | UNRESOLVED | +0.018 [+0.002, +0.033] | 0.101 | +0.012, +0.018, +0.023 | UNRESOLVED |
| `t09_bulge_shape_a25_rounded` | +0.041 [+0.036, +0.047] | 0.043 | +0.046, +0.026, +0.052 | UNRESOLVED | +0.004 [-0.000, +0.009] | 0.056 | +0.003, +0.015, -0.005 | UNRESOLVED |
| `t09_bulge_shape_a26_boxy` | +0.054 [+0.042, +0.066] | 0.052 | +0.072, +0.034, +0.055 | UNRESOLVED | +0.007 [-0.004, +0.018] | 0.074 | +0.007, +0.007, +0.006 | UNRESOLVED |
| `t09_bulge_shape_a27_no_bulge` | +0.040 [+0.035, +0.045] | 0.045 | +0.045, +0.027, +0.050 | UNRESOLVED | +0.005 [+0.001, +0.008] | 0.059 | +0.001, +0.014, -0.001 | UNRESOLVED |
| `t10_arms_winding_a28_tight` | +0.032 [+0.026, +0.038] | 0.043 | +0.041, +0.015, +0.040 | UNRESOLVED | +0.013 [+0.008, +0.018] | 0.054 | +0.015, +0.014, +0.010 | UNRESOLVED |
| `t10_arms_winding_a29_medium` | +0.003 [-0.003, +0.010] | 0.043 | +0.007, +0.004, -0.001 | UNRESOLVED | -0.007 [-0.013, -0.001] | 0.055 | -0.004, -0.003, -0.014 | UNRESOLVED |
| `t10_arms_winding_a30_loose` | +0.061 [+0.054, +0.068] | 0.045 | +0.078, +0.033, +0.071 | **BETTER** | +0.019 [+0.012, +0.024] | 0.060 | +0.025, +0.019, +0.012 | UNRESOLVED |
| `t11_arms_number_a31_1` | +0.021 [+0.011, +0.031] | 0.051 | +0.033, +0.014, +0.016 | UNRESOLVED | +0.002 [-0.006, +0.011] | 0.073 | +0.012, -0.004, -0.001 | UNRESOLVED |
| `t11_arms_number_a32_2` | +0.019 [+0.013, +0.025] | 0.043 | +0.026, -0.005, +0.036 | UNRESOLVED | -0.006 [-0.011, -0.000] | 0.055 | +0.003, -0.018, -0.001 | UNRESOLVED |
| `t11_arms_number_a33_3` | +0.063 [+0.043, +0.082] | 0.067 | +0.083, +0.044, +0.062 | UNRESOLVED | +0.006 [-0.013, +0.023] | 0.103 | +0.026, -0.001, -0.007 | UNRESOLVED |
| `t11_arms_number_a36_more_than_4` | +0.039 [+0.023, +0.055] | 0.068 | +0.048, +0.011, +0.059 | UNRESOLVED | +0.004 [-0.009, +0.018] | 0.106 | +0.009, -0.002, +0.005 | UNRESOLVED |
| `t11_arms_number_a37_cant_tell` | +0.011 [+0.006, +0.015] | 0.043 | +0.017, -0.003, +0.018 | UNRESOLVED | +0.002 [-0.002, +0.006] | 0.054 | +0.009, -0.003, -0.001 | UNRESOLVED |

## 4. §N: nuisance readout at 4 epochs (secondary)

Per variable v: the probe AUC for v split at its own median; D_v = baseline minus JEPA, mean over the
three seed pairs. Positive means the baseline carries **more** of v. Bar b_v = 3.0 × max(S_A,v, S_X,v) ×
√(2/3); 97.5% CI. Coverage is complete for all six variables: 160,849 train and 34,828 test galaxies.

**The six readings with states:**

| objective | variable | **state** | D_v | 97.5% CI | b_v | S_A,v | S_X,v | pair means |
|---|---|---|---|---|---|---|---|---|
| MoCo | PSF (`psfWidth_r`) | **LESS** | −0.178 | [−0.183, −0.173] | 0.144 | 0.059 | 0.005 | −0.217, −0.105, −0.212 |
| MoCo | brightness (`modelMag_r`) | UNRESOLVED (underpowered) | +0.058 | [+0.056, +0.060] | 0.059 | 0.024 | 0.001 | +0.072, +0.031, +0.071 |
| MoCo | sky (`sky_r`) | **LESS** | −0.084 | [−0.087, −0.081] | 0.011 | 0.004 | 0.003 | −0.083, −0.081, −0.088 |
| MAE | PSF | **SPLIT** (no direction claimed) | −0.039 | [−0.042, −0.035] | 0.144 | 0.059 | 0.011 | −0.064, +0.025, −0.078 |
| MAE | brightness | UNRESOLVED (underpowered) | +0.007 | [+0.005, +0.009] | 0.087 | 0.024 | 0.036 | +0.006, +0.020, −0.004 |
| MAE | sky | **SAME** (CI excludes 0: "detectable but below seed noise") | +0.006 | [+0.004, +0.007] | 0.011 | 0.004 | 0.001 | +0.002, +0.010, +0.005 |

- **Chance count: 0.15 MORE/LESS calls expected across the six readings; two were made**, both MoCo,
  both LESS. MoCo carries less PSF and less sky information than the JEPA. On sky the spreads are
  small and equal (0.004, 0.003), inside the calibrated range (§N plants: τ 0.004, null SAME 23 /
  UNRESOLVED 7 of 30, no directional call). On PSF the JEPA's spread is 0.059, far outside it; the
  bar takes that spread and the CI still clears it by 0.029.
- **MAE PSF is SPLIT.** Its null rate, beside it: §N has §F's pure run-level structure (σ = 0), whose
  SPLIT null rate is 7.5% at τ 0.006 with equal spreads, and 22% with a 3× noisier arm. The spreads
  here (0.059 and 0.011) are far beyond that grid, and a SPLIT's null rate rises with the spread, so
  this SPLIT is the expected reading of noise this size as much as of a seed-dependent effect. As
  hashed, it claims no direction.
- **MAE sky is SAME:** |D_v| + b_v = 0.016 ≤ 0.02. The CI excludes 0, so §C's label "detectable but
  below seed noise" applies (the record's label reads "detectable-but-tiny").
- The record's per-variable state and the hashed table agree on every reading (checked by hand
  against §C's precedence).

**Reported, no state** (`snr_r`, `specz`, `petroRad_r`):

| objective | variable | D_v | 97.5% CI | b_v | pair means |
|---|---|---|---|---|---|
| MoCo | S/N | +0.081 | [+0.078, +0.083] | 0.089 | +0.099, +0.039, +0.104 |
| MoCo | redshift | +0.048 | [+0.045, +0.050] | 0.050 | +0.055, +0.024, +0.064 |
| MoCo | size | +0.058 | [+0.056, +0.060] | 0.059 | +0.069, +0.030, +0.075 |
| MAE | S/N | +0.009 | [+0.006, +0.011] | 0.135 | +0.006, +0.029, −0.009 |
| MAE | redshift | +0.008 | [+0.006, +0.010] | 0.072 | −0.002, +0.019, +0.008 |
| MAE | size | +0.016 | [+0.014, +0.018] | 0.059 | +0.022, +0.006, +0.020 |

**Criterion 2 (draft v1's), reported, no state:** of the top-10 principal components of each
encoder's 4-epoch test embeddings, the variance share tracking a nuisance family (brightness or
observing), N; the share tracking morphology, Mo; and the participation ratio, PR.

| encoder | N | Mo | PR |
|---|---|---|---|
| A1 / A2 / A3 | 0.39 / 0.36 / 0.45 | 0.00 / 0.09 / 0.00 | 16.5 / 11.5 / 16.0 |
| MoCo s0 / s1 / s2 | 0.23 / 0.30 / 0.22 | 0.20 / 0.18 / 0.20 | 10.0 / 11.4 / 10.6 |
| MAE s0 / s1 / s2 | 0.70 / 0.80 / 0.73 | 0.02 / 0.00 / 0.00 | 5.0 / 5.1 / 5.3 |

## 5. Ladder (a): each encoder's own existence ladder (§D)

**How it is read.** §D: each baseline's own ladder states, "beside A1–A3's with an agreement column,
with no comparative state". The agreement column counts **seed agreement within each objective**
only: the three JEPA seeds (A1–A3), the three MoCo seeds, the three MAE seeds. It reads 3/3, or 2/1
naming the seed that differs, on the t1 rung. Agreement between a baseline and the JEPA is not
computed: it would be a comparative state ("existence survives in JEPA and not MoCo"), which §D says
would need its own D27 table and plants before the hash, and the hash has none. **This definition of
the agreement column is a reading recorded after the hash** (user, 11 Nov), not a hashed rule.

**Cells:** the t1 ladder's rung · retention under matching (S survives, P partial, C collapses, U
unresolved). Rungs: R1 clean linear direction; R2 present linearly but entangled, confounded, below
the effect floor or with nuisance clearance unresolved (the mechanism is in the record); R4 not
recoverable linearly, MLP decode unadjudicated (no untrained-MLP bar, D25). R3 is never emitted. Every
encoder's three ladder files (p2, s2, t1) agree on the rung for every answer and population.

**Per encoder, rung counts over the 37 answers:**

| encoder | full: R1 / R2 / R4 | full: S / P / C / U | conditional: R1 / R2 / R4 | conditional: S / P / C / U |
|---|---|---|---|---|
| A1 | 0 / 33 / 4 | 30 / 0 / 1 / 6 | 2 / 25 / 10 | 24 / 0 / 1 / 12 |
| A2 | 1 / 34 / 2 | 31 / 0 / 0 / 6 | 2 / 27 / 8 | 24 / 0 / 1 / 12 |
| A3 | 1 / 32 / 4 | 30 / 1 / 0 / 6 | 0 / 27 / 10 | 23 / 1 / 1 / 12 |
| MoCo s0 | 1 / 35 / 1 | 32 / 0 / 0 / 5 | 4 / 27 / 6 | 25 / 0 / 0 / 12 |
| MoCo s1 | 2 / 35 / 0 | 32 / 0 / 0 / 5 | 5 / 28 / 4 | 25 / 0 / 0 / 12 |
| MoCo s2 | 2 / 34 / 1 | 32 / 0 / 0 / 5 | 4 / 28 / 5 | 25 / 0 / 0 / 12 |
| MAE s0 | 0 / 33 / 4 | 30 / 1 / 0 / 6 | 2 / 26 / 9 | 24 / 0 / 1 / 12 |
| MAE s1 | 1 / 33 / 3 | 31 / 0 / 0 / 6 | 5 / 24 / 8 | 24 / 0 / 1 / 12 |
| MAE s2 | 1 / 32 / 4 | 31 / 0 / 0 / 6 | 2 / 25 / 10 | 23 / 0 / 1 / 13 |

**Seed agreement, answers at 3/3 out of 37:** full population JEPA 34, MoCo 35, MAE 35; conditional
JEPA 32, MoCo 34, MAE 32. No answer has all three seeds differing.

**Full population:**

| answer | A1 | A2 | A3 | JEPA agree | MoCo s0 | s1 | s2 | MoCo agree | MAE s0 | s1 | s2 | MAE agree |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `t01_smooth_or_features_a01_smooth` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t01_smooth_or_features_a02_features_or_disk` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t01_smooth_or_features_a03_star_or_artifact` | R4·U | R4·U | R4·U | 3/3 | R4·U | R2·U | R4·U | 2/1 (moco_s1 differs) | R4·U | R4·U | R4·U | 3/3 |
| `t02_edgeon_a04_yes` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t02_edgeon_a05_no` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t03_bar_a06_bar` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t03_bar_a07_no_bar` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t04_spiral_a08_spiral` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t04_spiral_a09_no_spiral` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t05_bulge_prominence_a10_no_bulge` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t05_bulge_prominence_a11_just_noticeable` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t05_bulge_prominence_a12_obvious` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t05_bulge_prominence_a13_dominant` | R2·S | R2·S | R2·S | 3/3 | R2·S | R1·S | R1·S | 2/1 (moco_s0 differs) | R2·S | R2·S | R2·S | 3/3 |
| `t06_odd_a14_yes` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t06_odd_a15_no` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t07_rounded_a16_completely_round` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t07_rounded_a17_in_between` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t07_rounded_a18_cigar_shaped` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t08_odd_feature_a19_ring` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t08_odd_feature_a20_lens_or_arc` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t08_odd_feature_a21_disturbed` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t08_odd_feature_a22_irregular` | R2·S | R1·S | R1·S | 2/1 (a1 differs) | R1·S | R1·S | R1·S | 3/3 | R2·S | R1·S | R1·S | 2/1 (mae_s0 differs) |
| `t08_odd_feature_a23_other` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t08_odd_feature_a24_merger` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t08_odd_feature_a38_dust_lane` | R2·U | R2·U | R2·U | 3/3 | R2·U | R2·U | R2·U | 3/3 | R2·U | R2·U | R2·U | 3/3 |
| `t09_bulge_shape_a25_rounded` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t09_bulge_shape_a26_boxy` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t09_bulge_shape_a27_no_bulge` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t10_arms_winding_a28_tight` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t10_arms_winding_a29_medium` | R4·U | R4·U | R4·U | 3/3 | R2·S | R2·S | R2·S | 3/3 | R4·U | R4·U | R4·U | 3/3 |
| `t10_arms_winding_a30_loose` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t11_arms_number_a31_1` | R2·C | R2·S | R2·P | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·P | R2·S | R2·S | 3/3 |
| `t11_arms_number_a32_2` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t11_arms_number_a33_3` | R4·U | R2·U | R4·U | 2/1 (a2 differs) | R2·U | R2·U | R2·U | 3/3 | R4·U | R4·U | R4·U | 3/3 |
| `t11_arms_number_a34_4` | R4·U | R2·U | R4·U | 2/1 (a2 differs) | R2·U | R2·U | R2·U | 3/3 | R4·U | R2·U | R4·U | 2/1 (mae_s1 differs) |
| `t11_arms_number_a36_more_than_4` | R2·U | R2·U | R2·U | 3/3 | R2·U | R2·U | R2·U | 3/3 | R2·U | R2·U | R2·U | 3/3 |
| `t11_arms_number_a37_cant_tell` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |

**Conditional population:**

| answer | A1 | A2 | A3 | JEPA agree | MoCo s0 | s1 | s2 | MoCo agree | MAE s0 | s1 | s2 | MAE agree |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `t01_smooth_or_features_a01_smooth` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t01_smooth_or_features_a02_features_or_disk` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t01_smooth_or_features_a03_star_or_artifact` | R4·U | R4·U | R4·U | 3/3 | R4·U | R2·U | R4·U | 2/1 (moco_s1 differs) | R4·U | R4·U | R4·U | 3/3 |
| `t02_edgeon_a04_yes` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t02_edgeon_a05_no` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t03_bar_a06_bar` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t03_bar_a07_no_bar` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t04_spiral_a08_spiral` | R4·U | R4·U | R4·U | 3/3 | R4·U | R4·U | R4·U | 3/3 | R4·U | R4·U | R4·U | 3/3 |
| `t04_spiral_a09_no_spiral` | R4·U | R4·U | R4·U | 3/3 | R4·U | R4·U | R4·U | 3/3 | R4·U | R4·U | R4·U | 3/3 |
| `t05_bulge_prominence_a10_no_bulge` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t05_bulge_prominence_a11_just_noticeable` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t05_bulge_prominence_a12_obvious` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t05_bulge_prominence_a13_dominant` | R4·U | R4·U | R4·U | 3/3 | R2·U | R2·U | R2·U | 3/3 | R4·U | R4·U | R4·U | 3/3 |
| `t06_odd_a14_yes` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t06_odd_a15_no` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t07_rounded_a16_completely_round` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t07_rounded_a17_in_between` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t07_rounded_a18_cigar_shaped` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t08_odd_feature_a19_ring` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t08_odd_feature_a20_lens_or_arc` | R4·U | R4·U | R4·U | 3/3 | R4·U | R4·U | R4·U | 3/3 | R4·U | R4·U | R4·U | 3/3 |
| `t08_odd_feature_a21_disturbed` | R4·U | R4·U | R4·U | 3/3 | R4·U | R4·U | R4·U | 3/3 | R4·U | R4·U | R4·U | 3/3 |
| `t08_odd_feature_a22_irregular` | R2·S | R2·S | R2·P | 3/3 | R1·S | R1·S | R1·S | 3/3 | R2·S | R1·S | R2·S | 2/1 (mae_s1 differs) |
| `t08_odd_feature_a23_other` | R2·S | R2·S | R2·S | 3/3 | R1·S | R1·S | R1·S | 3/3 | R2·S | R1·S | R2·S | 2/1 (mae_s1 differs) |
| `t08_odd_feature_a24_merger` | R2·S | R2·S | R2·S | 3/3 | R2·S | R1·S | R2·S | 2/1 (moco_s1 differs) | R2·S | R1·S | R2·S | 2/1 (mae_s1 differs) |
| `t08_odd_feature_a38_dust_lane` | R4·U | R2·U | R4·U | 2/1 (a2 differs) | R2·U | R2·U | R2·U | 3/3 | R4·U | R2·U | R4·U | 2/1 (mae_s1 differs) |
| `t09_bulge_shape_a25_rounded` | R1·S | R1·S | R2·S | 2/1 (a3 differs) | R1·S | R1·S | R1·S | 3/3 | R1·S | R1·S | R1·S | 3/3 |
| `t09_bulge_shape_a26_boxy` | R4·U | R4·U | R4·U | 3/3 | R4·U | R2·U | R2·U | 2/1 (moco_s0 differs) | R4·U | R4·U | R4·U | 3/3 |
| `t09_bulge_shape_a27_no_bulge` | R1·S | R1·S | R2·S | 2/1 (a3 differs) | R1·S | R1·S | R1·S | 3/3 | R1·S | R1·S | R1·S | 3/3 |
| `t10_arms_winding_a28_tight` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t10_arms_winding_a29_medium` | R2·C | R2·C | R2·C | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·C | R2·C | R2·C | 3/3 |
| `t10_arms_winding_a30_loose` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t11_arms_number_a31_1` | R4·U | R2·U | R2·U | 2/1 (a1 differs) | R2·U | R2·U | R2·U | 3/3 | R2·U | R2·U | R2·U | 3/3 |
| `t11_arms_number_a32_2` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 |
| `t11_arms_number_a33_3` | R4·U | R4·U | R4·U | 3/3 | R2·U | R2·U | R2·U | 3/3 | R2·U | R2·U | R2·U | 3/3 |
| `t11_arms_number_a34_4` | R2·U | R2·U | R4·U | 2/1 (a3 differs) | R2·U | R2·U | R2·U | 3/3 | R4·U | R4·U | R4·U | 3/3 |
| `t11_arms_number_a36_more_than_4` | R2·U | R2·U | R2·U | 3/3 | R2·U | R2·U | R2·U | 3/3 | R2·U | R2·U | R2·U | 3/3 |
| `t11_arms_number_a37_cant_tell` | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R2·S | 3/3 | R2·S | R2·S | R4·U | 2/1 (mae_s2 differs) |

## 6. What this does and does not say

- **Primary:** neither comparison reaches a claim. MoCo against the JEPA and MAE against the JEPA are
  both UNRESOLVED (underpowered) at 4 epochs and at every earlier epoch, because the bar exceeds SESOI.
  For MoCo the bar is set by the JEPA seeds' spread; for MAE by MAE's. MoCo misses BETTER by 0.0002 in
  the CI's lower end. Under the hashed rules that is UNRESOLVED and is reported as such.
- **Secondary:** MoCo is BETTER on 9 answers against ≤ 0.9 expected by chance (two complementary pairs
  among them). §N: MoCo carries less PSF and less sky than the JEPA (two LESS against 0.15 expected);
  MAE is SPLIT on PSF and SAME on sky.
- **Ladder (a):** each objective's own existence ladder, with seed agreement within objective; no
  comparative state.
- **Calibration:** the run-level seed spreads observed (JEPA 0.0165, MAE 0.0200 at 4 epochs; up to
  0.059 on §N's PSF) lie outside §F's grid. Every chance rate quoted beside a state is from the grid
  and is an extrapolation at these spreads.
- **Exploratory throughout:** every encoder is a smoke run (`baselines.md` F1). No comparison with M is
  made here; that step is held.

## Provenance

| item | value |
|---|---|
| pre-registration | `artifacts/baselines_comparison.md`, hashed section SHA-1 `b021e8688970f684ad429babfa44a2fcebe88a9c` (commit `bb80b42`) |
| scoring code | `artifacts/baselines_comparison.py` SHA-1 `2e87a80034e454c9d798ba76ebfdba4197f38dea` (commit `d30ac14`) |
| score record | `docs/baselines_comparison/baselines_score.json` SHA-1 `45c3d74f757499664409b2efb4ec4f69af4db496` (commit `905e4c4`) |
| split | identical split, ids SHA-1 `17296e0f7616bcccf3f968bb9e31167d86d03724` |
| P3 lists | `artifacts/baselines_p3.sha1`, `artifacts/baselines_p3_s2.sha1`, `baselines.md` §6.1 |
| ladder (a) run record | `docs/ladder_v2_baselines/` (state, outputs.sha1, run log; commit `53aee9e`) |
| §1.1 seed means | recomputed from `artifacts/out/{a1..a3}_traj{1..4}_trajectory.json`, `{moco,mae}_s{0..2}{1..4}_trajectory.json`, over the 35-answer set |
