# Comparison with M: write-up

Scored 19 Nov 2026 under the pre-registration hashed at `a39f724e` (`docs/m_comparison/prereg.md`, section
"Pre-registration"). Scoring code of record `artifacts/m_comparison.py` `47635acc`. Score record
`docs/m_comparison/m_comparison_score.json`, SHA-1 `8c4ad114…` (commit `f9eceeb`), byte-identical to
`artifacts/out/m_comparison_score.json`. Inputs: the re-laddered M and O2 (`docs/ladder_v2_m/outputs.sha1`
`5e9f29df`; runner 13/13 done) and A1–A3's ladder v2 records (`docs/ladder_v2/outputs.sha1`), all on the
identical split (160,849 train + 34,828 test; split ids `17296e0f`).

**The design in one line.** Two seed-matched pairs, (M, A1) at train seed 0 and (O2, A2) at train seed 1,
each a v1-corpus run against a v2-corpus run with identical initial weights; A3 (v2, seed 2) is unpaired
and can veto a direction. A per-answer directional state needs the same-direction existence change in
both pairs, A3 agreeing, and both changes beyond the guard.

**What is a state and what is not.** §§1, 2 and 5 report the hashed states. §§3 and 4 are exploratory
readings made after the hash, labelled as such; they add no state and change no state (the user,
19 Nov: "add no new states"). Their numbers come from `artifacts/m_comparison_explore.py` →
`docs/m_comparison/explore.json`, which reuses the scoring code's own z, BY family and noise functions and
asserts that its z and existence flags equal the record's for every run and cell.

## 1. Headline: full population (confirmatory)

**AGREES on 32 of 32 scoreable answers. THIN 5. No directional state. Overall state INSUFFICIENT (n = 0).**

| state | count | expected by chance (§D, the observed-like calibration row) |
|---|---|---|
| AGREES | **32** | 22.6 |
| THIN | 5 | — |
| A-LOWER / A-HIGHER | 0 / 0 | 0.04 / 0.02 |
| A-SPLIT (A3 veto) | 0 | 0.21 of the 32 scoreable (0.24 of 37; 0.64% per answer) |
| NEAR-BAR ↓ / ↑ | 0 / 0 | 0.10 / 0.05 |
| SHIFTED ↓ / ↑ | 0 / 0 | 0.45 / 0.44 |
| ONE-PAIR ↓ / ↑ | 0 / 0 | 2.8 / 2.8 |
| PAIRS-OPPOSE | 0 | 0.43 |
| SEED-SPLIT | 0 | 2.0 |
| VOID | 0 | — |

- **THIN (5; underpowered in all five runs):** `t01_smooth_or_features_a03_star_or_artifact`,
  `t08_odd_feature_a38_dust_lane`, `t11_arms_number_a33_3`, `t11_arms_number_a34_4`,
  `t11_arms_number_a36_more_than_4`.
- **Overall state INSUFFICIENT (n = 0).** The aggregate counts answers where both pairs change rung. There
  were none, so the binomial test of "M's existence results carried by the misregistration" (CARRIED) or
  "masked by it" (MASKED) had nothing to count. INSUFFICIENT carries no false call; the noise block prints
  the CARRIED/MASKED tipping points (0.42× / 0.44× at 2.5%, 0.77× / 0.76× at 5%) for reference only.
- **Plain meaning.** No existence result of M's was changed by training on band-aligned data. On every
  answer the ladder could score, each v1 run and its seed-matched v2 partner give the same existence
  verdict (present or absent), and none of M, O2, A1, A2 or A3 differs from the others. M's ladder
  results stand.

**The limit.** This is the existence rung: present against absent at D23's bar. It is not a statement
about magnitude. The guard-based state that would register a magnitude move without a rung change,
SHIFTED, was also 0, but its guard was wide here (§4: median 6.8 z, about 0.041 AUC, in the full
population), so SHIFTED = 0 rules out only large same-direction moves in both pairs. Magnitude
differences between the corpora are visible in §4 and carry no state.

## 2. Reproduction check (§I; pre-registered not to gate): PASS

The re-laddered M gives the same rung as M's original `artifacts/out/t1_ladder.json` (23 Sep, SHA-1
`d71ce4ee…`) in every cell, both populations. What this rules out: the ladder code changes since 23 Sep
(three commits on the ladder path, including `floor.py`, `nulls.py` and `probe.yaml`) and the removal of
the nine galaxies probe_v2 lacks did not move any of M's rungs. M's original ladder and the one compared
here are, rung for rung, the same result. (The count would mix the two causes had it failed; it did not
fail, so nothing needs separating.)

## 3. Fewer departures than chance (exploratory, post-hash)

None of the "something differs" states occurred: 0 departures against **9.4 expected** under the
observed-like calibration row (σ 2.0, τ/σ 1.0, β 0.1; chosen by the hashed rule before any record was
read). The reason is where the answers sit relative to the existence bar.

**Distance from the bar.** For each run, the BY step-up over 37 answers (α 0.05, t with 29 df against the
K = 30 untrained bank) sets one critical z: 2.41–2.44 in the full population, 2.49–2.52 in the
conditional. The distance is z − z_crit; in AUC terms, the same distance times the cell's z denominator
(√(bank sd² + real_se²)), i.e. how far the run's AUC sits above (or below) the AUC that would just clear
the bar.

- **Full population:** across the 32 scoreable answers, the nearest run is a median 8.9 z (0.050 AUC)
  from the bar. One answer is near it: medium winding (`t10_arms_winding_a29_medium`) sits below the bar
  in all five runs (−0.4 to −1.8 z; within 0.010 AUC) and is the only answer absent everywhere. The next
  nearest are boxy bulge (nearest run +2.1 z, A3) and 1 arm (+2.4 z, A3); 29 of 32 have every run at least
  4.5 z above.
- **Conditional population:** median nearest distance 8.7 z (0.077 AUC). Five answers have their nearest
  run below +2 z: three just above the bar, can't tell (arms number; A1 +0.3, A3 +0.4), no bar (A3 +0.6)
  and medium winding (O2 +0.7, all five present), and the two spiral answers, which are absent in all
  five runs by 4–13 z.

**Reading.** The calibration modelled half the non-borderline answers as sitting below the bar
(z of −2 to +1, against a critical z near 2.4) plus a borderline fraction β at it; with σ 2 noise those cells flip often, which is
where the ~9 expected departures (mostly ONE-PAIR and SEED-SPLIT) come from. These data are not like
that: nearly every answer is far above the bar in all five runs, and the one far-below answer in the full
population is below in all five. **The calibration's borderline mass was pessimistic for these data.**
This does not weaken AGREES. AGREES is read from the existence flags themselves, and those flags are
far from flipping on nearly every answer; a pessimistic chance count makes 0 departures less surprising
to explain, not less true.

**Per answer: z − z_crit, with the AUC distance in brackets** (sorted by the nearest run; positive =
above the bar; from `explore.json`).

#### Full population (32 scoreable)

| answer | M | A1 | O2 | A2 | A3 | nearest |
|---|---|---|---|---|---|---|
| `t10_arms_winding_a29_medium` | -1.8 (-0.010) | -0.4 (-0.002) | -1.2 (-0.007) | -1.6 (-0.009) | -1.0 (-0.005) | -1.8 |
| `t09_bulge_shape_a26_boxy` | +3.5 (+0.037) | +2.6 (+0.029) | +2.9 (+0.032) | +5.5 (+0.060) | +2.1 (+0.023) | +2.1 |
| `t11_arms_number_a31_1` | +5.1 (+0.053) | +2.5 (+0.027) | +4.2 (+0.043) | +4.8 (+0.051) | +2.4 (+0.026) | +2.4 |
| `t11_arms_number_a37_cant_tell` | +6.1 (+0.033) | +4.7 (+0.024) | +5.3 (+0.028) | +8.9 (+0.045) | +4.5 (+0.023) | +4.5 |
| `t10_arms_winding_a28_tight` | +6.3 (+0.033) | +5.3 (+0.025) | +6.2 (+0.034) | +10.5 (+0.051) | +4.7 (+0.023) | +4.7 |
| `t08_odd_feature_a24_merger` | +7.1 (+0.056) | +5.2 (+0.039) | +6.6 (+0.052) | +8.5 (+0.064) | +6.6 (+0.050) | +5.2 |
| `t08_odd_feature_a22_irregular` | +7.0 (+0.034) | +6.7 (+0.035) | +6.8 (+0.033) | +9.5 (+0.049) | +5.4 (+0.028) | +5.4 |
| `t08_odd_feature_a20_lens_or_arc` | +6.6 (+0.051) | +6.4 (+0.052) | +6.7 (+0.052) | +8.2 (+0.067) | +5.9 (+0.049) | +5.9 |
| `t09_bulge_shape_a27_no_bulge` | +7.1 (+0.051) | +7.2 (+0.049) | +6.2 (+0.045) | +11.4 (+0.076) | +7.2 (+0.049) | +6.2 |
| `t09_bulge_shape_a25_rounded` | +7.7 (+0.048) | +7.1 (+0.044) | +6.5 (+0.041) | +11.3 (+0.070) | +6.8 (+0.042) | +6.5 |
| `t08_odd_feature_a21_disturbed` | +8.3 (+0.043) | +7.2 (+0.036) | +6.9 (+0.036) | +10.4 (+0.051) | +6.7 (+0.034) | +6.7 |
| `t10_arms_winding_a30_loose` | +10.5 (+0.069) | +7.1 (+0.048) | +11.5 (+0.075) | +15.6 (+0.102) | +8.0 (+0.053) | +7.1 |
| `t11_arms_number_a32_2` | +10.5 (+0.056) | +8.9 (+0.048) | +8.8 (+0.047) | +15.5 (+0.082) | +7.5 (+0.041) | +7.5 |
| `t06_odd_a15_no` | +10.2 (+0.068) | +8.9 (+0.055) | +8.2 (+0.054) | +16.0 (+0.096) | +9.3 (+0.057) | +8.2 |
| `t05_bulge_prominence_a13_dominant` | +11.8 (+0.076) | +8.8 (+0.070) | +10.2 (+0.067) | +11.6 (+0.090) | +8.3 (+0.066) | +8.3 |
| `t05_bulge_prominence_a11_just_noticeable` | +11.6 (+0.051) | +8.9 (+0.039) | +11.2 (+0.049) | +12.0 (+0.052) | +8.9 (+0.038) | +8.9 |
| `t06_odd_a14_yes` | +11.3 (+0.069) | +8.9 (+0.055) | +9.1 (+0.056) | +15.9 (+0.095) | +9.1 (+0.056) | +8.9 |
| `t05_bulge_prominence_a10_no_bulge` | +12.9 (+0.078) | +10.2 (+0.065) | +11.8 (+0.072) | +15.8 (+0.098) | +9.2 (+0.059) | +9.2 |
| `t07_rounded_a17_in_between` | +11.7 (+0.048) | +9.7 (+0.041) | +13.6 (+0.055) | +25.1 (+0.105) | +11.1 (+0.048) | +9.7 |
| `t08_odd_feature_a23_other` | +10.2 (+0.046) | +11.3 (+0.047) | +10.0 (+0.045) | +12.3 (+0.051) | +11.1 (+0.046) | +10.0 |
| `t03_bar_a06_bar` | +12.4 (+0.070) | +11.8 (+0.070) | +11.8 (+0.066) | +15.5 (+0.091) | +10.4 (+0.062) | +10.4 |
| `t05_bulge_prominence_a12_obvious` | +14.0 (+0.059) | +11.3 (+0.051) | +13.4 (+0.057) | +14.8 (+0.066) | +11.7 (+0.053) | +11.3 |
| `t03_bar_a07_no_bar` | +13.5 (+0.082) | +12.7 (+0.077) | +11.4 (+0.070) | +17.3 (+0.102) | +12.2 (+0.074) | +11.4 |
| `t04_spiral_a08_spiral` | +16.1 (+0.073) | +16.1 (+0.069) | +14.2 (+0.064) | +19.6 (+0.084) | +16.8 (+0.072) | +14.2 |
| `t08_odd_feature_a19_ring` | +16.3 (+0.092) | +15.8 (+0.085) | +14.3 (+0.081) | +20.0 (+0.105) | +14.9 (+0.081) | +14.3 |
| `t04_spiral_a09_no_spiral` | +17.3 (+0.078) | +16.5 (+0.073) | +15.2 (+0.070) | +20.4 (+0.089) | +17.2 (+0.076) | +15.2 |
| `t01_smooth_or_features_a01_smooth` | +17.0 (+0.084) | +15.6 (+0.078) | +15.6 (+0.077) | +19.5 (+0.096) | +16.0 (+0.080) | +15.6 |
| `t01_smooth_or_features_a02_features_or_disk` | +17.7 (+0.083) | +16.8 (+0.078) | +16.1 (+0.076) | +20.9 (+0.096) | +17.0 (+0.080) | +16.1 |
| `t02_edgeon_a04_yes` | +20.7 (+0.148) | +16.2 (+0.120) | +21.1 (+0.152) | +27.0 (+0.193) | +17.9 (+0.131) | +16.2 |
| `t02_edgeon_a05_no` | +21.3 (+0.161) | +17.4 (+0.134) | +21.9 (+0.166) | +28.1 (+0.209) | +18.2 (+0.140) | +17.4 |
| `t07_rounded_a16_completely_round` | +25.6 (+0.108) | +20.1 (+0.090) | +25.1 (+0.106) | +39.7 (+0.164) | +22.6 (+0.100) | +20.1 |
| `t07_rounded_a18_cigar_shaped` | +24.9 (+0.169) | +20.4 (+0.142) | +25.9 (+0.175) | +33.4 (+0.221) | +22.4 (+0.155) | +20.4 |

#### Conditional population (27 scoreable; secondary)

| answer | M | A1 | O2 | A2 | A3 | nearest |
|---|---|---|---|---|---|---|
| `t04_spiral_a09_no_spiral` | -9.2 (-0.081) | -11.1 (-0.098) | -11.3 (-0.103) | -7.7 (-0.067) | -12.7 (-0.115) | -12.7 |
| `t04_spiral_a08_spiral` | -5.9 (-0.053) | -7.9 (-0.073) | -8.3 (-0.077) | -4.4 (-0.039) | -9.8 (-0.091) | -9.8 |
| `t11_arms_number_a37_cant_tell` | +4.7 (+0.044) | +0.3 (+0.003) | +1.9 (+0.019) | +6.0 (+0.057) | +0.4 (+0.004) | +0.3 |
| `t03_bar_a07_no_bar` | +2.5 (+0.020) | +2.3 (+0.019) | +3.2 (+0.026) | +5.1 (+0.041) | +0.6 (+0.005) | +0.6 |
| `t10_arms_winding_a29_medium` | +1.5 (+0.012) | +1.7 (+0.014) | +0.7 (+0.006) | +1.1 (+0.009) | +1.3 (+0.011) | +0.7 |
| `t03_bar_a06_bar` | +7.0 (+0.054) | +6.4 (+0.052) | +7.2 (+0.057) | +9.2 (+0.073) | +4.7 (+0.038) | +4.7 |
| `t08_odd_feature_a23_other` | +7.5 (+0.096) | +6.1 (+0.080) | +6.4 (+0.084) | +6.8 (+0.091) | +6.2 (+0.082) | +6.1 |
| `t05_bulge_prominence_a10_no_bulge` | +12.9 (+0.120) | +8.5 (+0.090) | +11.0 (+0.102) | +12.2 (+0.118) | +6.7 (+0.075) | +6.7 |
| `t09_bulge_shape_a27_no_bulge` | +8.6 (+0.121) | +7.3 (+0.108) | +8.7 (+0.122) | +11.8 (+0.154) | +9.4 (+0.127) | +7.3 |
| `t07_rounded_a17_in_between` | +8.5 (+0.038) | +7.7 (+0.035) | +9.8 (+0.044) | +25.0 (+0.113) | +8.9 (+0.041) | +7.7 |
| `t09_bulge_shape_a25_rounded` | +10.7 (+0.140) | +8.0 (+0.113) | +9.7 (+0.129) | +13.9 (+0.172) | +10.1 (+0.133) | +8.0 |
| `t06_odd_a15_no` | +10.2 (+0.067) | +8.8 (+0.055) | +8.1 (+0.054) | +15.9 (+0.096) | +9.2 (+0.057) | +8.1 |
| `t05_bulge_prominence_a12_obvious` | +14.9 (+0.090) | +9.2 (+0.062) | +14.3 (+0.085) | +16.2 (+0.099) | +8.4 (+0.057) | +8.4 |
| `t05_bulge_prominence_a11_just_noticeable` | +15.2 (+0.097) | +8.7 (+0.059) | +13.7 (+0.090) | +15.2 (+0.097) | +8.7 (+0.059) | +8.7 |
| `t08_odd_feature_a24_merger` | +10.1 (+0.108) | +9.8 (+0.104) | +10.9 (+0.118) | +12.2 (+0.129) | +8.8 (+0.095) | +8.8 |
| `t06_odd_a14_yes` | +11.2 (+0.068) | +8.9 (+0.054) | +9.0 (+0.055) | +15.8 (+0.094) | +9.0 (+0.055) | +8.9 |
| `t10_arms_winding_a30_loose` | +12.7 (+0.131) | +9.5 (+0.104) | +13.0 (+0.135) | +14.9 (+0.156) | +10.4 (+0.112) | +9.5 |
| `t10_arms_winding_a28_tight` | +15.2 (+0.120) | +13.0 (+0.101) | +15.8 (+0.126) | +17.8 (+0.131) | +11.8 (+0.093) | +11.8 |
| `t01_smooth_or_features_a01_smooth` | +17.0 (+0.084) | +15.5 (+0.077) | +15.4 (+0.077) | +19.4 (+0.096) | +15.9 (+0.080) | +15.4 |
| `t01_smooth_or_features_a02_features_or_disk` | +17.6 (+0.083) | +16.7 (+0.078) | +16.0 (+0.075) | +20.8 (+0.096) | +16.9 (+0.079) | +16.0 |
| `t08_odd_feature_a19_ring` | +25.0 (+0.229) | +18.9 (+0.195) | +17.1 (+0.185) | +31.3 (+0.257) | +19.5 (+0.203) | +17.1 |
| `t11_arms_number_a32_2` | +21.8 (+0.160) | +17.6 (+0.131) | +20.5 (+0.152) | +21.4 (+0.157) | +17.3 (+0.130) | +17.3 |
| `t08_odd_feature_a22_irregular` | +21.0 (+0.170) | +19.6 (+0.166) | +21.9 (+0.169) | +21.9 (+0.179) | +18.0 (+0.156) | +18.0 |
| `t07_rounded_a18_cigar_shaped` | +25.1 (+0.176) | +21.2 (+0.153) | +26.3 (+0.183) | +35.3 (+0.236) | +22.7 (+0.162) | +21.2 |
| `t07_rounded_a16_completely_round` | +26.0 (+0.115) | +21.9 (+0.102) | +25.2 (+0.113) | +41.3 (+0.177) | +23.4 (+0.109) | +21.9 |
| `t02_edgeon_a05_no` | +28.0 (+0.230) | +25.1 (+0.213) | +27.0 (+0.225) | +32.6 (+0.260) | +24.9 (+0.213) | +24.9 |
| `t02_edgeon_a04_yes` | +30.2 (+0.239) | +26.9 (+0.221) | +29.2 (+0.234) | +34.6 (+0.268) | +26.9 (+0.222) | +26.9 |

## 4. Run-level noise (exploratory; no state rests on it)

The record prints two run-level numbers beside the overall state. Both lie beyond the calibrated grid
(which runs to 1.5× per-answer run noise).

| | drift: mean (Δ1 − Δ2), z | SE | in AUC: mean [(A1 − M) − (A2 − O2)] | A-bound (× per-answer run noise σ) | A-bound in z | A-bound in AUC |
|---|---|---|---|---|---|---|
| full | −5.70 | 0.65 | −0.040 (SE 0.006) | 5.76× | 12.7 | 0.078 |
| conditional | −5.68 | 0.83 | −0.051 (SE 0.010) | 4.43× | 12.4 | 0.113 |

The drift is labelled, as hashed, "one draw of the run-level noise; not an estimate of its size". The
A-bound is a one-sided 95% upper bound with 2 df, from A1–A3's run-level spread.

**Translating z into AUC.** z = (AUC − bank mean) / √(bank sd² + real_se²), and real_se shrinks as
1/√n_test, so the same AUC offset is a larger z on a well-sampled answer. The AUC column of the drift is
computed directly from the AUCs, cell by cell; the A-bound is converted at the median z denominator
(0.0061 AUC full, 0.0091 conditional). The per-answer run noise σ̂_c on the centred footing is 2.2 z
(full) and 2.8 z (conditional), about 0.013 and 0.026 AUC.

**What the drift is made of.** Mean AUC over the 37 full-population answers, per run:

| M (v1, s0) | A1 (v2, s0) | O2 (v1, s1) | A2 (v2, s1) | A3 (v2, s2) |
|---|---|---|---|---|
| 0.706 | 0.687 | 0.702 | **0.724** | 0.690 |

The seed-0 pair moves down from v1 to v2 (A1 − M ≈ −0.019); the seed-1 pair moves up (A2 − O2 ≈
+0.022). The difference, about −0.04 AUC, is the drift. It comes from A2: A2 is the furthest of the five
runs above the bar on 29 of the 32 scoreable full-population answers (22 of 27 conditional), while O2, A2's v1 partner
with identical initial weights, sits with M. A1 and A3 agree closely and sit below M and O2.

**Relation to earlier readings.** The same pattern has appeared twice before:

- **The aligned comparison (A3 report, 13 Oct):** "A2 sits above all on every answer", with A1 and A3
  close together (|A1 − A3| 0.0066, about M's own seed spread) and below M; it left open whether A2 is an
  outlying basin or part of a wide spread.
- **The baselines comparison (§1.1, 15 Nov):** training seed 1 sits about 0.03 above seeds 0 and 2 for
  both the JEPA (A2) and MAE (MAE s1), at every epoch, on v2; MoCo's seeds agree to within 0.002.

What this comparison adds is the v1 side of seed 1: **O2, the same seed on the v1 corpus, does not show
it.** The seed fixes both the initial weights and the data order; O2 shares A2's initial weights but sees
a different corpus, so a different sequence. A seed-1 effect carried by the initial weights alone would
appear in O2 as well. It does not, which points (exploratory, untested) to the seed-1 data order on v2,
or its interaction with those weights, rather than the weights alone. MAE s1 shares both weights and
v2 data order with A2, and shows the effect.

**The guard was inflated by this drift.** The pair noise σ̂² behind the guard is computed uncentred, by
design (so that a pair-level drift widens it), and here the drift dominates it: σ̂² 33.9, of which the
centred per-answer variance accounts for about 9.5–9.8 (2 × 2.21²) and the −5.7 drift contributes the
remaining ~24. The guard's median is 6.8 z (0.041 AUC) in the full population, against about 3.6 z had
it been built on the centred per-answer noise. This is why SHIFTED = 0 (§1) is a weak
statement about magnitude. It does not touch AGREES, which rests on the existence flags.

**Had CARRIED or MASKED fired, it would have been uninterpretable at this noise level.** The A-bound
(5.76×) is beyond the grid's end, where the CARRIED/MASKED false-call rate is already 8.5% / 8.7% and
rising; the run-specific noise the pre-registration warned two pairs cannot fix is, on A1–A3's evidence,
not excluded at many times the size the grid covers. Neither fired; the overall state is INSUFFICIENT.

## 5. Secondary

**Conditional population (secondary, as hashed).** AGREES 27,
THIN 10, VOID 0, no directional state, overall INSUFFICIENT (n = 0). THIN:
`t01_smooth_or_features_a03_star_or_artifact`, `t05_bulge_prominence_a13_dominant`,
`t08_odd_feature_a20_lens_or_arc`, `t08_odd_feature_a21_disturbed`, `t08_odd_feature_a38_dust_lane`,
`t09_bulge_shape_a26_boxy`, `t11_arms_number_a31_1`, `t11_arms_number_a33_3`, `t11_arms_number_a34_4`,
`t11_arms_number_a36_more_than_4`. Expected departures by chance 7.9; observed 0.

**Grade tier (R1 against R2; unguarded and exploratory, as hashed).**

| population | G-AGREES | G-NA | G-ONE-PAIR | G-SPLIT |
|---|---|---|---|---|
| full | 30 | 6 | 1: `t08_odd_feature_a22_irregular` | 0 |
| conditional | 23 | 12 | 1: `t08_odd_feature_a24_merger` | 1: `t09_bulge_shape_a25_rounded` |

One pair changing grade on one answer, and a split on another, are single unguarded readings with no
direction claimed.

## Provenance

- Pre-registration `docs/m_comparison/prereg.md`, hashed section SHA-1 `a39f724e2bf9f512cbc802e52caf12247fabb491`
  (commit `1791d56`); drafted blind of M, O2 and every A1–A3 ladder value, slips declared there.
- Scoring code `artifacts/m_comparison.py` `47635accdaa7a9870f7f5fd2468ec5b8b81ff590` (commit `58445ed`);
  plants `docs/m_comparison/plants.json` `95416bac…`, calibration `calibration.json` `9c598007…`.
- Score record `docs/m_comparison/m_comparison_score.json` `8c4ad114ae16943d9469ef627ddd652c2a52df8c` (commit `f9eceeb`).
- M/O2 ladder: `docs/ladder_v2_m/` (manifest at `17945d9`, runner record at `58445ed`); A1–A3: `docs/ladder_v2/`.
- Exploratory readings (§§3–4): `artifacts/m_comparison_explore.py` → `docs/m_comparison/explore.json`.
