# Part F — vote noise, label sources and unsupervised axes (24 Nov 2026)

Branch `almagest-geometry`. **Descriptive throughout: nothing here is hashed and no state is assigned beyond Part A's existing reading labels.**
Frozen A1 (`runs/m_v2/encoder.pt`), raw and cleaned embeddings, three untrained seeds as the floor, Part A's ordinal machinery
(one-axis ordinal fit against the free multinomial; τ = Kendall τ of the ordinal score with E[category]; gap = free gap over the base;
share = ordinal share of that gap). Code: `artifacts/axes_f.py` (F1, F2), `artifacts/axes_f1_thin.py`, `artifacts/axes_f345.py`. Numbers: `f1.json`,
`f1_thin.json`, `f2.json`, `f3.json`, `f4.json`, `f5.json`.

## Summary (read this first)

- **F1. Signal rises with votes, but not only because of the votes.** τ for arm count goes from about 0 at 1–4 votes to 0.25 (cleaned) / 0.34 (raw) at 17+.
  Votes and resolution rise together, so the bins can't separate them. Thinning the best-voted galaxies' labels to 1–2 votes can: their τ stays far above
  the 1–4 bin's (arms, cleaned: 0.16 at one vote against −0.00). At one vote, thinning costs the top bin 30–37% of its τ for arms and 51–70% for winding; for roundness almost nothing (cleaned 0.35 to 0.33). What is left of the
  low bins' deficit is something else, and resolution is the obvious candidate, not a tested one.
- **F2. DECaLS labels change little.** Scored on the same galaxies, A1's τ against the deeper DECaLS votes is within 0.02 of its τ against GZ2's for arms, pooled and in GZD-5. In GZD-1/2 (about 2,000 test galaxies) the DECaLS τ is higher, raw: 0.259 against 0.177 for arms,
  0.152 against 0.021 for winding; DECaLS has more votes there (12–15 against 6–8). The readings mostly agree. The two catalogues agree with each other only at ρ = 0.40–0.56.
- **F3. Training on clean galaxies only does not help, with one exception.** The exception is the cleaned bulge axis on clean test galaxies (0.340 against 0.285). It is worse on
  everyone (0.031 against 0.146). Clean fits use 4–9 times fewer galaxies, so cleanness and sample size are confounded.
- **F4. Weighted against raw vote fractions: no difference.** Every share and τ agrees to 0.001.
- **F5. PCA of the cleaned space has no component that is one thing; ICA has one that stands out (ICA16: winding score 0.54, concentration 0.36).** No names are assigned.

## F1. Vote-count bins

Bins of votes on the answer's own question, 1–4, 5–8, 9–16 and 17+. No bin fell below the sample floor, so none was merged.

- **Arm count:** cleaned τ −0.00 / 0.01 / 0.05 / 0.25; raw 0.00 / 0.10 / 0.17 / 0.34. The untrained maximum at 17+ is 0.13 cleaned and 0.24 raw.
  The 17+ bin against the matched baseline has gap 0.0235, share 1.22 (ONE AXIS) cleaned, and 0.0507, share 1.06 raw.
- **Winding:** cleaned 0.04 / 0.10 / 0.13 / 0.20; raw 0.04 / 0.07 / 0.15 / 0.25.
- **Bulge:** cleaned 0.21 / 0.21 / 0.23 / 0.35; raw 0.28 / 0.38 / 0.47 / 0.47.
- **Roundness:** cleaned 0.25 / 0.37 / 0.42 / 0.35; raw 0.20 / 0.29 / 0.35 / 0.37. It is the flattest.

**The confound.** Vote count is tied to resolution. Median size / PSF rises from 4.9 to 7.3 over the arm bins (winding 4.4 to 7.4, bulge 4.0 to 7.0). It falls from 8.2 to 4.4
for roundness. Median votes in the 1–4 bin are 1 for arms and winding, 3 for bulge and 2 for roundness.

**The thinning supplement (post hoc, labelled as such).** In the 17+ bin, each test galaxy's votes are thinned to k = 1 or 2 by hypergeometric draws, and the axes refitted or scored
on the thinned labels. This holds the galaxies fixed and changes only the label noise.

| A1 τ in the 17+ bin, thinned labels / full labels (untrained max, thinned) | k = 1 | k = 2 |
|---|---|---|
| arm count, cleaned | 0.158 / 0.205 (0.115) | 0.183 / 0.219 (0.129) |
| arm count, raw | 0.236 / 0.318 (0.194) | 0.261 / 0.307 (0.215) |
| winding, cleaned | 0.060 / 0.097 (0.061) | 0.125 / 0.178 (0.080) |
| winding, raw | 0.122 / 0.213 (0.090) | 0.164 / 0.230 (0.114) |
| bulge, cleaned | 0.204 / 0.299 (0.070) | 0.249 / 0.323 (0.088) |
| bulge, raw | 0.278 / 0.439 (0.188) | 0.343 / 0.458 (0.250) |
| roundness, cleaned | 0.329 / 0.341 | 0.341 / 0.344 |
| roundness, raw | 0.350 / 0.366 | 0.364 / 0.369 |

"Full" is the same thinned-fit axis scored against the full labels in its own row, so it differs from F1's table, where the fit also uses the full labels.

**Reading.** At the vote counts that the 1–4 bin has, the best-resolved galaxies still give τ = 0.16–0.24 for arms against ≈ 0. So label noise does not explain the low bins.
For winding the thinned τ at k = 1 is 0.06 cleaned (the 1–4 bin has 0.04, and the untrained maximum thinned is 0.06), so there noise alone can account for the low bin; raw it can't (0.12 against 0.04).
For bulge the 1–4 bin has a median of 3 votes, so it should lie between k = 2 and k = 4 (cleaned 0.25–0.29, raw 0.34–0.41). It sits at 0.21 and 0.28, below that: vote noise explains part of the deficit, not all.
The supplement can't isolate resolution from other differences in the 17+ galaxies, since nobody thinned the low bins.

Full table, τ (gap free-over-base) / max untrained τ:

**arm count**: test n 7821 / 1924 / 2041 / 3842; median votes 1 / 6 / 12 / 28; median size/PSF 4.9 / 5.8 / 6.4 / 7.3

| | 1–4 | 5–8 | 9–16 | 17+ | 17+ thinned to 4 | thinned to 2 | thinned to 1 |
|---|---|---|---|---|---|---|---|
| cleaned | -0.00 (0.013) / -0.00 | +0.01 (0.061) / +0.06 | +0.05 (0.079) / +0.03 | +0.25 (0.075) / +0.13 | +0.22 / +0.13 | +0.18 / +0.13 | +0.16 / +0.11 |
| raw | +0.00 (0.012) / +0.00 | +0.10 (0.061) / +0.07 | +0.17 (0.082) / +0.11 | +0.34 (0.102) / +0.24 | +0.31 / +0.23 | +0.26 / +0.21 | +0.24 / +0.19 |

**winding**: test n 10156 / 2318 / 2415 / 5355; median votes 1 / 6 / 12 / 30; median size/PSF 4.4 / 5.5 / 6.0 / 7.4

| | 1–4 | 5–8 | 9–16 | 17+ | 17+ thinned to 4 | thinned to 2 | thinned to 1 |
|---|---|---|---|---|---|---|---|
| cleaned | +0.04 (0.001) / +0.03 | +0.10 (0.004) / +0.05 | +0.13 (0.006) / +0.10 | +0.20 (0.015) / +0.11 | +0.17 / +0.09 | +0.12 / +0.08 | +0.06 / +0.06 |
| raw | +0.04 (0.002) / +0.04 | +0.07 (0.003) / +0.05 | +0.15 (0.005) / +0.11 | +0.25 (0.026) / +0.17 | +0.21 / +0.16 | +0.16 / +0.11 | +0.12 / +0.09 |

**bulge prominence**: test n 13543 / 7021 / 4408 / 8108; median votes 3 / 6 / 12 / 29; median size/PSF 4.0 / 4.5 / 5.1 / 7.0

| | 1–4 | 5–8 | 9–16 | 17+ | 17+ thinned to 4 | thinned to 2 | thinned to 1 |
|---|---|---|---|---|---|---|---|
| cleaned | +0.21 (0.034) / +0.08 | +0.21 (0.025) / +0.09 | +0.23 (0.027) / +0.08 | +0.35 (0.051) / +0.13 | +0.29 / +0.11 | +0.25 / +0.09 | +0.20 / +0.07 |
| raw | +0.28 (0.089) / +0.20 | +0.38 (0.095) / +0.27 | +0.47 (0.102) / +0.34 | +0.47 (0.088) / +0.35 | +0.41 / +0.30 | +0.34 / +0.25 | +0.28 / +0.19 |

**roundness**: test n 2258 / 1737 / 3305 / 26755; median votes 2 / 6 / 13 / 35; median size/PSF 8.2 / 7.4 / 6.4 / 4.4

| | 1–4 | 5–8 | 9–16 | 17+ | 17+ thinned to 4 | thinned to 2 | thinned to 1 |
|---|---|---|---|---|---|---|---|
| cleaned | +0.25 (0.084) / +0.07 | +0.37 (0.125) / +0.17 | +0.42 (0.160) / +0.18 | +0.35 (0.108) / +0.16 | +0.35 / +0.16 | +0.34 / +0.16 | +0.33 / +0.16 |
| raw | +0.20 (0.037) / +0.06 | +0.29 (0.061) / +0.12 | +0.35 (0.103) / +0.14 | +0.37 (0.111) / +0.20 | +0.37 / +0.20 | +0.36 / +0.19 | +0.35 / +0.19 |

## F2. DECaLS labels on the same galaxies

For every galaxy with both catalogues, the axes are fitted and scored against DECaLS E[category] and against GZ2 E[category], same galaxies and split. Pooled DECaLS uses GZD-5 where it has a
vote and GZD-1/2 otherwise; GZD-5 alone and GZD-1/2 alone are also given. GZD-5's bulge question has five answers, GZD-1/2's three; GZ2 has four. Only arm count and winding have matching answers (`aa_findings.md`),
so bulge uses each survey's native answers.

| question, catalogue | n train / test | median votes DECaLS / GZ2 | label ρ(E) | embedding | target | A1 gap | share, reading | A1 τ | max untrained τ |
|---|---|---|---|---|---|---|---|---|---|
| arm count: DECaLS pooled (GZD-5 where it has a vote) | 6267 / 5463 | 4 / 9 | 0.45 | cleaned | decals | 0.012 | 1.72 ONE AXIS | 0.179 | 0.068 |
| arm count: DECaLS pooled (GZD-5 where it has a vote) | 6267 / 5463 | 4 / 9 | 0.45 | cleaned | gz2 | 0.029 | 1.19 ONE AXIS | 0.173 | 0.099 |
| arm count: DECaLS pooled (GZD-5 where it has a vote) | 6267 / 5463 | 4 / 9 | 0.45 | raw | decals | 0.038 | 1.19 ONE AXIS | 0.261 | 0.170 |
| arm count: DECaLS pooled (GZD-5 where it has a vote) | 6267 / 5463 | 4 / 9 | 0.45 | raw | gz2 | 0.058 | 1.07 ONE AXIS | 0.244 | 0.180 |
| arm count: GZD-5 only | 4800 / 4194 | 4 / 10 | 0.47 | cleaned | decals | 0.014 | 1.30 MOSTLY ONE AXIS | 0.173 | 0.076 |
| arm count: GZD-5 only | 4800 / 4194 | 4 / 10 | 0.47 | cleaned | gz2 | 0.027 | 1.10 ONE AXIS | 0.174 | 0.109 |
| arm count: GZD-5 only | 4800 / 4194 | 4 / 10 | 0.47 | raw | decals | 0.039 | 1.12 ONE AXIS | 0.251 | 0.165 |
| arm count: GZD-5 only | 4800 / 4194 | 4 / 10 | 0.47 | raw | gz2 | 0.058 | 1.06 ONE AXIS | 0.257 | 0.189 |
| arm count: GZD-1/2 only | 2118 / 1870 | 15 / 6 | 0.46 | cleaned | decals | 0.002 | 1.47 NO SIGNAL | 0.090 | 0.013 |
| arm count: GZD-1/2 only | 2118 / 1870 | 15 / 6 | 0.46 | cleaned | gz2 | 0.003 | 1.79 MOSTLY ONE AXIS | 0.088 | 0.061 |
| arm count: GZD-1/2 only | 2118 / 1870 | 15 / 6 | 0.46 | raw | decals | 0.025 | 1.33 ONE AXIS | 0.259 | 0.174 |
| arm count: GZD-1/2 only | 2118 / 1870 | 15 / 6 | 0.46 | raw | gz2 | 0.034 | 1.04 MOSTLY ONE AXIS | 0.177 | 0.139 |
| winding: DECaLS pooled (GZD-5 where it has a vote) | 7569 / 6549 | 4 / 11 | 0.40 | cleaned | decals | 0.008 | 1.15 ONE AXIS | 0.113 | 0.041 |
| winding: DECaLS pooled (GZD-5 where it has a vote) | 7569 / 6549 | 4 / 11 | 0.40 | cleaned | gz2 | 0.011 | 1.07 ONE AXIS | 0.109 | 0.064 |
| winding: DECaLS pooled (GZD-5 where it has a vote) | 7569 / 6549 | 4 / 11 | 0.40 | raw | decals | 0.015 | 1.14 ONE AXIS | 0.150 | 0.092 |
| winding: DECaLS pooled (GZD-5 where it has a vote) | 7569 / 6549 | 4 / 11 | 0.40 | raw | gz2 | 0.017 | 1.02 ONE AXIS | 0.117 | 0.076 |
| winding: GZD-5 only | 5866 / 5086 | 4 / 12 | 0.41 | cleaned | decals | 0.004 | 1.44 ONE AXIS | 0.097 | 0.035 |
| winding: GZD-5 only | 5866 / 5086 | 4 / 12 | 0.41 | cleaned | gz2 | 0.009 | 1.21 ONE AXIS | 0.112 | 0.066 |
| winding: GZD-5 only | 5866 / 5086 | 4 / 12 | 0.41 | raw | decals | 0.011 | 1.19 ONE AXIS | 0.126 | 0.089 |
| winding: GZD-5 only | 5866 / 5086 | 4 / 12 | 0.41 | raw | gz2 | 0.017 | 1.04 ONE AXIS | 0.123 | 0.078 |
| winding: GZD-1/2 only | 2479 / 2170 | 12 / 8 | 0.41 | cleaned | decals | 0.003 | 0.81 MOSTLY ONE AXIS | 0.107 | 0.032 |
| winding: GZD-1/2 only | 2479 / 2170 | 12 / 8 | 0.41 | cleaned | gz2 | 0.001 | 0.97 MOSTLY ONE AXIS | 0.063 | 0.040 |
| winding: GZD-1/2 only | 2479 / 2170 | 12 / 8 | 0.41 | raw | decals | 0.013 | 1.20 ONE AXIS | 0.152 | 0.099 |
| winding: GZD-1/2 only | 2479 / 2170 | 12 / 8 | 0.41 | raw | gz2 | 0.002 | 1.05 MOSTLY ONE AXIS | 0.021 | 0.034 |
| bulge prominence: gzd5 native answers | 11049 / 9557 | 3 / 9 | 0.54 | cleaned | decals | 0.020 | 0.66 SEVERAL | 0.152 | 0.084 |
| bulge prominence: gzd5 native answers | 11049 / 9557 | 3 / 9 | 0.54 | cleaned | gz2 | 0.034 | 0.70 SEVERAL | 0.164 | 0.087 |
| bulge prominence: gzd5 native answers | 11049 / 9557 | 3 / 9 | 0.54 | raw | decals | 0.078 | 0.92 MOSTLY ONE AXIS | 0.366 | 0.282 |
| bulge prominence: gzd5 native answers | 11049 / 9557 | 3 / 9 | 0.54 | raw | gz2 | 0.096 | 0.84 MOSTLY ONE AXIS | 0.421 | 0.316 |
| bulge prominence: gzd12 native answers | 4955 / 4359 | 7 / 6 | 0.56 | cleaned | decals | 0.013 | 0.53 SEVERAL | 0.155 | 0.067 |
| bulge prominence: gzd12 native answers | 4955 / 4359 | 7 / 6 | 0.56 | cleaned | gz2 | 0.015 | 0.93 MOSTLY ONE AXIS | 0.123 | 0.038 |
| bulge prominence: gzd12 native answers | 4955 / 4359 | 7 / 6 | 0.56 | raw | decals | 0.049 | 0.95 MOSTLY ONE AXIS | 0.359 | 0.235 |
| bulge prominence: gzd12 native answers | 4955 / 4359 | 7 / 6 | 0.56 | raw | gz2 | 0.079 | 0.89 MOSTLY ONE AXIS | 0.378 | 0.251 |

- **Arms and winding.** A1's τ is 0.17–0.18 (cleaned) and 0.24–0.26 (raw) for arms against either catalogue; winding 0.11 and 0.12–0.15. Pooled and in GZD-5 the two catalogues differ by at most 0.03. The exception is GZD-1/2 raw, where DECaLS gives 0.259 against 0.177 for arms and 0.152 against 0.021 for winding (the GZ2 τ there is below its untrained maximum, 0.034). DECaLS has more votes in that subset (medians 15 and 12 against 6 and 8), so the gap may be vote count and not depth.
- **The GZD-1/2 arm reading is the weak one:** cleaned DECaLS is NO SIGNAL (gap 0.002), and raw τ is 0.259 against an untrained maximum of 0.174.
- **The agreement between catalogues is low** (Spearman of E[category] 0.40–0.47 for arms and winding, 0.54–0.56 for bulge), while the median votes are 4 (DECaLS pooled) against 9–11 (GZ2). A large part of
  that is vote noise on a four-vote estimate; F1's thinning costs 30–70% of τ for arms and winding at one vote.
- **Bulge** stays SEVERAL in the cleaned embedding against GZD-5 and GZ2 both (0.66 and 0.70) and is MOSTLY ONE AXIS raw.

## F3. Clean galaxies only

A galaxy is clean if its votes on the question are at or above the 75th percentile and its top fraction is ≥ 0.6. Train-clean counts: arms 3,467 of 17,899; winding 2,683 of 23,360; bulge 5,454 of 37,997; roundness 9,465 of 39,113.
The test set is scored both on clean galaxies and on all.

| question | n train clean / all | embedding | cos(clean, all) | τ on clean test: clean-fit / all-fit (max untrained clean-fit) | τ on all test: clean-fit / all-fit (max untrained clean-fit) |
|---|---|---|---|---|---|
| arm count | 3467 / 17899 | cleaned | 0.69 | 0.159 / 0.197 (0.099) | 0.105 / 0.132 (0.062) |
| arm count | 3467 / 17899 | raw | 0.80 | 0.242 / 0.265 (0.172) | 0.171 / 0.187 (0.119) |
| winding | 2683 / 23360 | cleaned | 0.84 | 0.224 / 0.251 (0.122) | 0.055 / 0.081 (0.034) |
| winding | 2683 / 23360 | raw | 0.84 | 0.290 / 0.306 (0.202) | 0.031 / 0.073 (0.008) |
| bulge prominence | 5454 / 37997 | cleaned | 0.79 | 0.340 / 0.285 (0.126) | 0.031 / 0.146 (0.025) |
| bulge prominence | 5454 / 37997 | raw | 0.89 | 0.462 / 0.468 (0.346) | 0.381 / 0.400 (0.272) |
| roundness | 9465 / 39113 | cleaned | 0.79 | 0.284 / 0.275 (0.082) | 0.299 / 0.332 (0.083) |
| roundness | 9465 / 39113 | raw | 0.77 | 0.300 / 0.299 (0.137) | 0.308 / 0.352 (0.139) |

- All-fit is the better axis almost everywhere, on clean and on all test galaxies. Exceptions on the clean test set: **bulge cleaned** (0.340 against 0.285) and roundness, level (0.284 / 0.275 cleaned, 0.300 / 0.299 raw).
- The bulge exception reverses on everyone (0.031 against 0.146 cleaned): the clean-fit bulge axis does not generalise.
- cos(clean-fit, all-fit) is 0.69–0.89, so the two are related but not the same direction.
- **The confound:** clean training sets are 4–9 times smaller, and their top category is dominated by one class (2 arms 0.87; obvious and just-noticeable bulge 0.68).

## F4. Weighted against raw vote fractions

The GZ2 user-weighted fractions replace raw fractions, over the rows where both are finite (91–285 rows dropped per question).

| question | n train / test | ρ(E raw, E weighted), mean fraction difference | embedding | share raw / weighted | τ raw / weighted (max untrained, raw votes) |
|---|---|---|---|---|---|
| arm count | 17797 / 15532 | 0.9993, 0.0004 | cleaned | 0.586 / 0.586 | 0.135 / 0.135 (0.075) |
| arm count | 17797 / 15532 | 0.9993, 0.0004 | raw | 0.716 / 0.715 | 0.190 / 0.189 (0.136) |
| winding | 23213 / 20106 | 0.9997, 0.0007 | cleaned | 0.902 / 0.902 | 0.082 / 0.082 (0.038) |
| winding | 23213 / 20106 | 0.9997, 0.0007 | raw | 0.922 / 0.921 | 0.073 / 0.073 (0.031) |
| bulge prominence | 37943 / 33043 | 0.9991, 0.0009 | cleaned | 0.668 / 0.668 | 0.147 / 0.146 (0.106) |
| bulge prominence | 37943 / 33043 | 0.9991, 0.0009 | raw | 0.826 / 0.826 | 0.400 / 0.400 (0.310) |
| roundness | 39015 / 33981 | 0.9992, 0.0022 | cleaned | 0.834 / 0.835 | 0.334 / 0.334 (0.166) |
| roundness | 39015 / 33981 | 0.9992, 0.0022 | raw | 0.908 / 0.909 | 0.354 / 0.355 (0.188) |

Every share and τ agrees to 0.001, and the E[category] ranks agree at ρ = 0.9991–0.9997. The published weighting does not change any reading. (A first run read a label sidecar without the weighted
columns and gave all NaN; it was discarded, and the code now asserts that the columns exist.)

## F5. PCA and ICA of the cleaned space

Twenty components of each, fitted on the 39,997 training galaxies' cleaned embedding, scored on 34,828 test galaxies. PCA explains 10.8%, 6.7%, 6.0%, 5.4%, 4.7% … of the variance; ICA converged in 54 iterations.
Each component is compared with the four ordinal **scores** (A1's own cleaned axes, not the votes, so those ρ are alignment inside the embedding) and the photometric and MPA and Simard table-1 measures available now. **To top up after Part B** adds the rest (Simard's remaining tables, statmorph).
The components are not named. Extremes: `f5_pca_strips.png`, `f5_ica_strips.png`.

**PCA** (component: top target ρ): 1: winding +0.17; 2: g_minus_r_observed_model -0.14; 3: bulge prominence -0.28; 4: bulge prominence +0.12; 5: winding -0.16; 6: winding -0.34; 7: roundness +0.23; 8: log_ssfr_mpa +0.23; 9: winding -0.26; 10: arm count +0.12; 11: bulge prominence -0.15; 12: winding +0.20; 13: roundness -0.15; 14: winding -0.28; 15: roundness +0.30; 16: roundness -0.12; 17: winding +0.15; 18: arm count +0.18; 19: bulge prominence +0.09; 20: bulge prominence +0.15

**ICA** (component: top target ρ): 1: arm count -0.14; 2: roundness +0.12; 3: log_ssfr_mpa -0.10; 4: roundness +0.25; 5: bulge prominence -0.30; 6: bulge prominence +0.21; 7: g_minus_r_observed_model +0.10; 8: log_ssfr_mpa +0.07; 9: bulge prominence -0.19; 10: arm count -0.09; 11: bulge prominence -0.26; 12: roundness -0.16; 13: arm count +0.20; 14: bulge prominence -0.13; 15: bulge prominence -0.21; 16: winding +0.54; 17: bulge prominence -0.14; 18: roundness -0.19; 19: bulge prominence +0.18; 20: winding +0.08

- **PCA:** the largest |ρ| of any component with any target is 0.34 (component 6, the winding score); the rest sit between 0.09 and 0.30. Against a measurement alone the largest is 0.24 (component 14, log sSFR). No principal component is one human or physical quantity.
- **ICA16** is the exception: winding score 0.54, concentration 0.36, arm count −0.35, roundness −0.34, log sSFR −0.34, fracDeV 0.30, Simard B/T 0.30, g−r 0.29. It correlates with several at once, and is the only component above 0.4; I name nothing.
- With 20 components and about 17 targets, a largest |ρ| of 0.1–0.2 is not evidence of anything at this n (34,828, so significance is trivially reached); read the sizes only.

## What I would not conclude

- That vote noise is "the" reason arms and winding are weak. It accounts for part of the fall; it does not explain arms' zero.
- That DECaLS depth helps. Pooled and GZD-5 show nothing; GZD-1/2 shows something, but it can be vote count.
- That a cleaner-labelled subset yields better axes. F3's sample sizes forbid saying so.

### F5 top-up after Part B (25 Nov)

The saved component scores against every Part B measurement, statmorph included (`f5_topup.json`; `axes_f5_topup.py`; `f5.json` unchanged). No component is one physical quantity:
- the largest |ρ| of any **PCA** component with any measurement is 0.26 (component 8, g−r); the rest sit between 0.03 and 0.25. Components 1, 2, 6–8, 14 and 18 lean the same way: colour, sSFR, concentration.
- **ICA16** stays the outlier: concentration 0.36, Gini 0.34, sSFR −0.34 (and, from F5, the winding score 0.54). The rest of ICA tops out at 0.17.
- What Part B finds for directions trained on a target (cleaned ρ 0.2–0.55) is not found along any axis of maximal variance or independence: the physics is in directions that need the target to be found.
