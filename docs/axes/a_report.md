# Part A: are ordered Galaxy Zoo questions one coordinate in A1? (24 Nov 2026)

Branch `almagest-geometry`. Frozen A1 embeddings only. Pre-registration: `prereg_part_a.md` (SHA-1 `058e4dd1…`;
brief verbatim `cfaed329…`; hashed and committed before any real fit, 065e30d). Numbers: `a_ordinal.json`. Plants:
`a_plants.json`. Strips: `a_strips.png`. Code: `artifacts/axes_a_ordinal.py`.

## Readings (hashed)

| question | cleaned | raw |
|---|---|---|
| arm count | SEVERAL | SEVERAL |
| winding | MOSTLY ONE AXIS | MOSTLY ONE AXIS |
| bulge prominence | SEVERAL | MOSTLY ONE AXIS |
| roundness | MOSTLY ONE AXIS | MOSTLY ONE AXIS |

**Read the arm-count line with care: the covariate drives it.**
- The brief keeps the "can't tell" fraction as a covariate and takes the marginal as the baseline. I put the covariate in
  both models (written into the prereg before any fit), so the gap and the share include what it explains.
- Adding it to cleaned A1 raises the free model by 0.029 nats per vote and the ordinal model by 0.005. The free model has
  one slope per category for it; the ordinal model has one slope in all. The share falls because of that difference.
- Without it (the sensitivity declared in the prereg, which carries no reading), the same rule would give **0.953
  [0.878, 1.042], MOSTLY ONE AXIS, cleaned** and **0.997 [0.951, 1.048], ONE AXIS, raw**.
- **The hashed reading stays SEVERAL.** As evidence about *A1's* directions, the sensitivity is the more direct number.
  A later run could hash it with a marginal-plus-covariate baseline.

## What the fits say

| question | emb. | encoder | LL base / ord / free | gap free − base [CI] | captured share [CI] | reading |
|---|---|---|---|---|---|---|
| arm count | cleaned | A1 | -1.0514 / -1.0150 / -0.9893 | 0.0621 [0.0562, 0.0684] | 0.587 [0.532, 0.637] | **SEVERAL** |
| arm count | cleaned | A1, no covariate | -1.0514 / -1.0200 / -1.0185 | 0.0329 [0.0270, 0.0389] | 0.953 [0.878, 1.042] | *MOSTLY ONE AXIS* (sensitivity, not a reading) |
| arm count | cleaned | untrained-s0 | -1.0514 / -1.0353 / -1.0114 | 0.0400 [0.0353, 0.0449] | 0.403 [0.319, 0.482] | SEVERAL |
| arm count | cleaned | untrained-s1 | -1.0514 / -1.0337 / -1.0110 | 0.0405 [0.0365, 0.0446] | 0.439 [0.354, 0.519] | SEVERAL |
| arm count | cleaned | untrained-s2 | -1.0514 / -1.0359 / -1.0119 | 0.0395 [0.0348, 0.0443] | 0.394 [0.301, 0.470] | SEVERAL |
| arm count | raw | A1 | -1.0514 / -0.9911 / -0.9672 | 0.0842 [0.0775, 0.0903] | 0.717 [0.674, 0.759] | **SEVERAL** |
| arm count | raw | A1, no covariate | -1.0514 / -0.9947 / -0.9946 | 0.0569 [0.0497, 0.0629] | 0.997 [0.951, 1.048] | *ONE AXIS* (sensitivity, not a reading) |
| arm count | raw | untrained-s0 | -1.0514 / -1.0111 / -0.9873 | 0.0641 [0.0584, 0.0696] | 0.630 [0.577, 0.678] | SEVERAL |
| arm count | raw | untrained-s1 | -1.0514 / -1.0092 / -0.9862 | 0.0653 [0.0596, 0.0708] | 0.647 [0.591, 0.698] | SEVERAL |
| arm count | raw | untrained-s2 | -1.0514 / -1.0117 / -0.9887 | 0.0627 [0.0573, 0.0679] | 0.633 [0.578, 0.681] | SEVERAL |
| winding | cleaned | A1 | -1.0594 / -1.0474 / -1.0461 | 0.0133 [0.0110, 0.0155] | 0.902 [0.859, 0.949] | **MOSTLY ONE AXIS** |
| winding | cleaned | untrained-s0 | -1.0594 / -1.0558 / -1.0561 | 0.0033 [0.0025, 0.0042] | 1.075 [0.967, 1.180] | ONE AXIS |
| winding | cleaned | untrained-s1 | -1.0594 / -1.0562 / -1.0566 | 0.0028 [0.0017, 0.0040] | 1.130 [1.029, 1.308] | ONE AXIS |
| winding | cleaned | untrained-s2 | -1.0594 / -1.0566 / -1.0567 | 0.0028 [0.0017, 0.0037] | 1.014 [0.831, 1.159] | MOSTLY ONE AXIS |
| winding | raw | A1 | -1.0594 / -1.0435 / -1.0422 | 0.0172 [0.0149, 0.0196] | 0.922 [0.889, 0.957] | **MOSTLY ONE AXIS** |
| winding | raw | untrained-s0 | -1.0594 / -1.0517 / -1.0522 | 0.0072 [0.0057, 0.0086] | 1.080 [0.977, 1.189] | ONE AXIS |
| winding | raw | untrained-s1 | -1.0594 / -1.0520 / -1.0526 | 0.0068 [0.0053, 0.0083] | 1.094 [1.031, 1.167] | ONE AXIS |
| winding | raw | untrained-s2 | -1.0594 / -1.0525 / -1.0529 | 0.0065 [0.0049, 0.0080] | 1.058 [0.989, 1.136] | ONE AXIS |
| bulge prominence | cleaned | A1 | -1.1655 / -1.1412 / -1.1291 | 0.0364 [0.0342, 0.0386] | 0.668 [0.644, 0.693] | **SEVERAL** |
| bulge prominence | cleaned | untrained-s0 | -1.1655 / -1.1583 / -1.1559 | 0.0095 [0.0081, 0.0109] | 0.753 [0.675, 0.824] | MOSTLY ONE AXIS |
| bulge prominence | cleaned | untrained-s1 | -1.1655 / -1.1583 / -1.1557 | 0.0097 [0.0084, 0.0111] | 0.739 [0.664, 0.809] | SEVERAL |
| bulge prominence | cleaned | untrained-s2 | -1.1655 / -1.1581 / -1.1556 | 0.0099 [0.0085, 0.0113] | 0.743 [0.680, 0.799] | SEVERAL |
| bulge prominence | raw | A1 | -1.1655 / -1.0770 / -1.0584 | 0.1070 [0.1038, 0.1101] | 0.826 [0.815, 0.837] | **MOSTLY ONE AXIS** |
| bulge prominence | raw | untrained-s0 | -1.1655 / -1.1198 / -1.1087 | 0.0568 [0.0532, 0.0602] | 0.804 [0.783, 0.827] | MOSTLY ONE AXIS |
| bulge prominence | raw | untrained-s1 | -1.1655 / -1.1174 / -1.1068 | 0.0587 [0.0550, 0.0620] | 0.819 [0.798, 0.841] | MOSTLY ONE AXIS |
| bulge prominence | raw | untrained-s2 | -1.1655 / -1.1196 / -1.1093 | 0.0562 [0.0530, 0.0594] | 0.817 [0.797, 0.838] | MOSTLY ONE AXIS |
| roundness | cleaned | A1 | -0.9979 / -0.9046 / -0.8860 | 0.1119 [0.1080, 0.1157] | 0.834 [0.821, 0.848] | **MOSTLY ONE AXIS** |
| roundness | cleaned | untrained-s0 | -0.9979 / -0.9781 / -0.9760 | 0.0219 [0.0198, 0.0242] | 0.903 [0.867, 0.941] | MOSTLY ONE AXIS |
| roundness | cleaned | untrained-s1 | -0.9979 / -0.9814 / -0.9792 | 0.0187 [0.0168, 0.0208] | 0.885 [0.846, 0.929] | MOSTLY ONE AXIS |
| roundness | cleaned | untrained-s2 | -0.9979 / -0.9820 / -0.9802 | 0.0177 [0.0159, 0.0195] | 0.897 [0.861, 0.934] | MOSTLY ONE AXIS |
| roundness | raw | A1 | -0.9979 / -0.8958 / -0.8855 | 0.1124 [0.1084, 0.1162] | 0.908 [0.897, 0.920] | **MOSTLY ONE AXIS** |
| roundness | raw | untrained-s0 | -0.9979 / -0.9703 / -0.9692 | 0.0288 [0.0263, 0.0310] | 0.960 [0.931, 0.989] | ONE AXIS |
| roundness | raw | untrained-s1 | -0.9979 / -0.9717 / -0.9711 | 0.0268 [0.0246, 0.0292] | 0.977 [0.949, 1.009] | ONE AXIS |
| roundness | raw | untrained-s2 | -0.9979 / -0.9734 / -0.9723 | 0.0257 [0.0235, 0.0277] | 0.958 [0.930, 0.983] | ONE AXIS |

- **All gaps exclude zero** (no NO SIGNAL), and all share CIs are defined: 0 undefined resamples everywhere.
  The free model's gap over the marginal is small for winding (0.013 cleaned, 0.017 raw nats per vote) and larger for
  roundness (0.112 cleaned) and raw bulge (0.107).
- **Winding: MOSTLY ONE AXIS** (0.902 and 0.922; the CI lower bounds, 0.859 and 0.889, fall short of 0.9). The free model
  gains only 0.001 over the ordinal one.
- **Bulge prominence: SEVERAL cleaned (0.668), MOSTLY ONE AXIS raw (0.826).** Cleaning
  removes two thirds of the free model's gap (0.107 → 0.036), so most of the raw bulge signal is shared with the five
  nuisances.
- **Roundness: MOSTLY ONE AXIS** (0.834 cleaned, 0.908 raw; CI lower bound 0.897 on raw).
- **The share does not measure what A1 learned.** Untrained encoders often read *closer* to one axis (winding and raw
  roundness: ONE AXIS) because they carry 2–5 times less signal, and weak signal is easily captured by one direction.
  Shares above 1 mean the ordinal model beats the free one on test, which overfits. A1's lower share goes with a larger
  gap. Compare encoders on the gap and on τ.

## Visibility control (Kendall τ of the ordinal score with E[category], within deciles of R_petro / PSF)

| question | emb. | control | τ all, A1 (untrained) | deciles A1 clears | A1 τ, decile 1 → 10 | untrained max, decile 1 → 10 |
|---|---|---|---|---|---|---|
| arm count | cleaned | PARTIAL | 0.132 (0.071, 0.074, 0.067) | 3 (8, 9, 10) | -0.05 -0.01 0.00 0.04 0.05 0.07 0.13 0.15 0.22 0.30 | -0.04 -0.01 -0.01 0.04 0.04 0.05 0.10 0.10 0.13 0.22 |
| arm count | raw | PARTIAL | 0.187 (0.127, 0.136, 0.129) | 5 (6, 7, 8, 9, 10) | -0.06 0.01 0.01 0.04 0.07 0.11 0.14 0.15 0.24 0.33 | -0.03 -0.02 -0.03 0.03 0.04 0.06 0.08 0.07 0.11 0.23 |
| winding | cleaned | PARTIAL | 0.081 (0.038, 0.038, 0.034) | 4 (6, 8, 9, 10) | 0.04 0.01 0.03 0.03 0.06 0.10 0.06 0.15 0.16 0.22 | 0.02 0.01 0.04 0.03 0.05 0.02 0.03 0.05 0.08 0.15 |
| winding | raw | PARTIAL | 0.073 (0.037, 0.030, 0.029) | 4 (6, 8, 9, 10) | 0.01 -0.00 0.01 0.03 0.05 0.07 0.06 0.16 0.18 0.28 | -0.01 0.00 0.02 0.04 0.02 -0.01 0.03 0.05 0.10 0.24 |
| bulge prominence | cleaned | HOLDS | 0.146 (0.105, 0.105, 0.099) | 9 (1, 2, 3, 4, 5, 6, 7, 8, 10) | 0.06 0.11 0.14 0.14 0.18 0.17 0.18 0.18 0.17 0.16 | 0.03 0.05 0.07 0.10 0.12 0.13 0.14 0.16 0.15 0.14 |
| bulge prominence | raw | HOLDS | 0.400 (0.306, 0.310, 0.302) | 10 (1, 2, 3, 4, 5, 6, 7, 8, 9, 10) | 0.15 0.24 0.31 0.35 0.41 0.41 0.45 0.48 0.50 0.51 | 0.10 0.15 0.23 0.26 0.30 0.33 0.36 0.39 0.42 0.44 |
| roundness | cleaned | HOLDS | 0.333 (0.165, 0.156, 0.150) | 10 (1, 2, 3, 4, 5, 6, 7, 8, 9, 10) | 0.27 0.31 0.33 0.33 0.35 0.35 0.33 0.34 0.36 0.38 | 0.11 0.12 0.12 0.12 0.12 0.15 0.16 0.18 0.25 0.30 |
| roundness | raw | HOLDS | 0.352 (0.187, 0.183, 0.177) | 10 (1, 2, 3, 4, 5, 6, 7, 8, 9, 10) | 0.27 0.32 0.34 0.36 0.37 0.37 0.36 0.37 0.38 0.37 | 0.10 0.11 0.15 0.17 0.17 0.19 0.21 0.23 0.26 0.26 |

- **Bulge and roundness HOLD**, clearing every untrained seed in 9–10 of 10 deciles (both kinds). Roundness is the cleanest:
  τ 0.27–0.38 at every decile against ≤ 0.30 untrained.
- **Arm count and winding are PARTIAL** and grow with resolution. Arm-count |τ| is ≤ 0.07 in deciles 1–5 and reaches
  0.30–0.33 in decile 10. Winding clears the untrained seeds only in deciles 6 and 8–10. Both orderings are present only
  in well-resolved galaxies, which fits V1's finding (`artifacts/v_findings.md`) that visibility
  falls monotonically with arm count and 4+ needs a resolved galaxy.
- "Holds" here is my bar (CI lower bound above the largest untrained τ, 200 resamples, ≥ 6 of 10 deciles), not yours.

## Geometry

| question | emb. | r(ordinal, adjacent FREE contrast) | r between FREE contrasts | singular-value shares | r(ordinal, own-question probes) |
|---|---|---|---|---|---|
| arm count | cleaned | a32_2 − a31_1 +0.10; a33_3 − a32_2 +0.85; a34_4 − a33_3 +0.37; a36_more_than_4 − a34_4 +0.09 | -0.20; +0.02; -0.02; +0.03; -0.16; +0.14 | 0.64, 0.18, 0.12, 0.06 | a31_1 -0.35; a32_2 -0.14; a33_3 +0.46; a34_4 +0.07; a36_more_than_4 -0.07; a37_cant_tell -0.56 |
| arm count | raw | a32_2 − a31_1 +0.56; a33_3 − a32_2 +0.90; a34_4 − a33_3 +0.65; a36_more_than_4 − a34_4 -0.14 | +0.31; +0.35; -0.22; +0.44; -0.30; -0.10 | 0.61, 0.19, 0.14, 0.06 | a31_1 -0.61; a32_2 -0.14; a33_3 +0.41; a34_4 +0.02; a36_more_than_4 -0.30; a37_cant_tell -0.69 |
| winding | cleaned | a29_medium − a28_tight +0.75; a30_loose − a29_medium +0.88 | +0.35 | 0.86, 0.14 | a28_tight -0.52; a29_medium +0.06; a30_loose +0.73 |
| winding | raw | a29_medium − a28_tight +0.78; a30_loose − a29_medium +0.86 | +0.37 | 0.88, 0.12 | a28_tight -0.46; a29_medium +0.30; a30_loose +0.74 |
| bulge prominence | cleaned | a11_just_noticeable − a10_no_bulge +0.53; a12_obvious − a11_just_noticeable +0.89; a13_dominant − a12_obvious -0.19 | +0.12; -0.62; -0.03 | 0.69, 0.23, 0.08 | a10_no_bulge -0.52; a11_just_noticeable -0.46; a12_obvious +0.65; a13_dominant +0.28 |
| bulge prominence | raw | a11_just_noticeable − a10_no_bulge +0.77; a12_obvious − a11_just_noticeable +0.98; a13_dominant − a12_obvious +0.02 | +0.68; -0.52; +0.07 | 0.74, 0.21, 0.05 | a10_no_bulge -0.85; a11_just_noticeable -0.81; a12_obvious +0.96; a13_dominant +0.62 |
| roundness | cleaned | a17_in_between − a16_completely_round +0.90; a18_cigar_shaped − a17_in_between +0.92 | +0.66 | 0.92, 0.08 | a16_completely_round -0.95; a17_in_between -0.06; a18_cigar_shaped +0.92 |
| roundness | raw | a17_in_between − a16_completely_round +0.94; a18_cigar_shaped − a17_in_between +0.96 | +0.83 | 0.93, 0.07 | a16_completely_round -0.96; a17_in_between +0.29; a18_cigar_shaped +0.94 |

| question | strongest correlations with other questions' probes (cleaned) |
|---|---|
| arm count | t01_smooth_or_features_a02_features_or_disk +0.70; t01_smooth_or_features_a01_smooth -0.70; t04_spiral_a09_no_spiral -0.69; t04_spiral_a08_spiral +0.68 |
| winding | t05_bulge_prominence_a11_just_noticeable -0.62; t01_smooth_or_features_a02_features_or_disk -0.53; t01_smooth_or_features_a01_smooth +0.53; t04_spiral_a08_spiral -0.51 |
| bulge prominence | t08_odd_feature_a22_irregular -0.46; t08_odd_feature_a19_ring +0.45; t03_bar_a06_bar +0.40; t11_arms_number_a32_2 +0.38 |
| roundness | t02_edgeon_a05_no -0.86; t02_edgeon_a04_yes +0.84; t09_bulge_shape_a26_boxy -0.51; t08_odd_feature_a38_dust_lane +0.49 |

- **Bulge prominence looks like more than one direction, not just uneven spacing.** "Dominant − obvious" is
  uncorrelated with the ordinal score (−0.19 cleaned, +0.02 raw) and runs against the "just noticeable − none"
  contrast (−0.62, −0.52). Uneven spacing alone would keep the contrasts parallel. "Dominant" is 8% of votes.
- **Roundness and winding look like one axis with a smaller second part:** top singular shares 0.92 and 0.86 (cleaned),
  every contrast +0.75 to +0.96 with the ordinal score.
- **Arm count's contrasts are nearly independent cleaned** (|r| ≤ 0.20), and the ordinal score mostly follows the
  2 → 3 step (+0.85 cleaned, +0.90 raw). The fits these come from include the covariate. Votes for 4 and 4+ are 10% of the total.
- **The ordinal scores overlap other questions' axes.** Arm count goes with features-or-disc (+0.70) and spiral (+0.68),
  roundness with edge-on (+0.84), winding with a weaker bulge (−0.62 with "just noticeable").

## What else to know

- **Population.** A median of 4 votes per galaxy on the target answers (arm count, winding), 6 (bulge) and 32 (roundness). Arm count drops
  10,076 eligible galaxies whose arm votes were all "can't tell" (17,899 train and 15,628 test are left), so its
  population is selected on the label's non-exit, as the brief's target requires.
- **Baseline skew.** Arm-count votes are 68% "2"; the expected category varies little across the strip (E between 1.6 and 3.0).
- **Strips (`a_strips.png`).** Roundness orders clearly (round → edge-on). Bulge orders only loosely. Winding and arm
  count show little visible ordering, consistent with their small LL gaps. Slices with few galaxies at the extremes are
  individual galaxies, not averages.

**A-bis (24 Nov, hashed separately):** the arm-count question with the covariate in the baseline. Cleaned MOSTLY ONE AXIS, raw ONE AXIS; see
`a_bis_report.md`. The SEVERAL reading above stays on the record.
