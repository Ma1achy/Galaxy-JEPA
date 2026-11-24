# Part A pre-registration: ordinal axes (24 Nov 2026)

The brief's own section, verbatim, is `prereg_part_a_brief.md` (SHA-1 `cfaed32917e90f11ef3bebdf4cdcc1c56bc82ed6`,
hashed first). The section below adds the operational detail and is hashed before any real fit.

## Pre-registration

### The brief, verbatim

PART A — ordinal axes: one axis or many? (embeddings only, Mac)
Question: does A1 store ordered Galaxy Zoo questions as one coordinate, or as separate per-category directions?
- Questions:
  - arm count: 1 / 2 / 3 / 4 / 4+, with "can't tell" excluded from the target and its fraction kept as a covariate;
  - winding: tight / medium / loose;
  - bulge prominence: none / just noticeable / obvious / dominant;
  - roundness: completely round / in between / cigar.
- Population: each question's eligible galaxies, frozen train/test split. Run on both cleaned (20-column) and raw A1 embeddings.
- Targets: soft vote fractions. Loss weighted by that question's own vote count: L = −Σ_i n_i Σ_k p_ik log p̂_ik.
- Two models per question, same loss, L2 chosen by 5-fold CV on train:
  - ORDINAL: cumulative logit with one shared direction w and K−1 ordered thresholds;
  - FREE: multinomial with K−1 independent directions.
  - Baseline: the train marginal distribution.
- Statistic (test): mean per-vote log-likelihood for each model. Captured share = (LL_ord − LL_base) / (LL_free − LL_base), with a 1,000-resample galaxy bootstrap CI. Also give the same numbers for the three untrained seeds.
- Readings, written before any number is computed (first match applies):
  - NO SIGNAL: the CI of LL_free − LL_base includes 0;
  - ONE AXIS: the captured share's CI lower bound ≥ 0.9;
  - MOSTLY ONE AXIS: point estimate ≥ 0.75;
  - SEVERAL: otherwise.
  Report all four for both cleaned and raw.
- Visibility control: Kendall τ between the ordinal score and each galaxy's expected category (Σ k·p_k), within deciles of petroRad_r / psfWidth_r, with the untrained range per decile. An axis that holds within deciles is structure; one that disappears is a visibility gradient (cf. V1's finding for arm count).
- Geometry: the score correlation between ordinal w and each FREE contrast direction; and ordinal w against the existing per-answer probes from the 37-axis run.
- Strip: galaxies along each ordinal score, 5 per slice, as in the contact sheet.
- Hash this Part A section before computing anything, as with earlier pre-registrations.

### Operational detail (this repo; added before any vote fraction was read by the fits)

**Data.** Rows: the 37-axis run's eligible train/test rows for the question (`almagest_axes.Data.rows`, vote-count
floor 1, complete nuisances), so cleaned and raw use the same galaxies. Soft targets: the ladder's raw
`_fraction` columns (never debiased), renormalised over the ordered categories. n_i: the sum of those categories'
raw `_count` columns. Galaxies with no vote on the ordered categories are dropped and counted. Features: train-standardised.
Cleaned and raw A1 embeddings and the three banked untrained encoders (each cleaned with its own cleaner).

**Covariate (arm count).** "can't tell" fraction is added to both models as one covariate (standardised on train,
unpenalised): a single slope in the ordinal model, one per category in the free model. The baseline stays the
marginal, as the brief says, so LL_free − LL_base includes what the covariate explains; a no-covariate fit is
reported beside it as a sensitivity and does not enter a reading. The ordinal score used for τ, geometry and strips
is w·x, without the covariate term.

**Fit.** Per-vote-mean loss + (λ/2)‖W‖², intercepts and thresholds unpenalised; 13 λ from 1e1 to 1e-5; 5-fold
CV on train by held-out per-vote log-likelihood, the folds identical for the two models (and identical for every
encoder). The free model uses the symmetric softmax (K weight rows, rank K−1). L-BFGS to gtol 1e-6; a
non-converged fit or a CV choice at the smallest λ raises.

**Completing the reading table (D27).** The brief's NO SIGNAL fires when the CI of LL_free − LL_base
*includes* 0. The case where the whole CI is below 0 (free worse than the marginal) is not named, so it is assigned to
NO SIGNAL as well: the rule is "CI lower bound ≤ 0". The share CI is computed on resamples with a positive gap; the
number of undefined resamples is reported. A point estimate ≥ 0.75 with lower bound ≥ 0.9 is ONE AXIS (first match). An
ordinal model that beats the free one (share > 1) is ONE AXIS if the CI lower bound ≥ 0.9.

**Visibility control (my operational bar, since "holds" was not defined).** Per decile: Kendall τ of w·x with
the expected category, 200 bootstrap resamples of the galaxies in the decile. A decile holds if A1's CI lower bound
exceeds the largest untrained τ. HOLDS: ≥ 6 of 10 deciles; DISAPPEARS: ≤ 2; PARTIAL: otherwise. Reported per question
and embedding kind; it does not change a reading.

**Geometry.** Pearson r over the question's test rows between the ordinal score and each adjacent FREE contrast
score (W_{k+1} − W_k)·x, and between the ordinal score and the 37-axis probe scores (stored cleaned probes; for raw,
the same probe refitted on raw rows, own question only). Also the singular-value shares of the centred FREE weights.

**Plants (D28; `a_plants.json`).** Synthetic soft votes on A1's cleaned arm-count rows with the real vote counts, through
the identical code; no vote fraction enters. All four readings are reached.

| plant | reading | captured share | LL gap (free − base) |
|---|---|---|---|
| ordinal (true one axis) | ONE AXIS | 1.016 | 0.2965 |
| independent directions (true several) | SEVERAL | 0.280 | 0.3267 |
| one axis + independent directions, b=0.1 | ONE AXIS | 0.962 | 0.3044 |
| one axis + independent directions, b=0.2 | ONE AXIS | 0.943 | 0.3124 |
| one axis + independent directions, b=0.3 | ONE AXIS | 0.914 | 0.3187 |
| one axis + independent directions, b=0.4 | MOSTLY ONE AXIS | 0.877 | 0.3293 |
| one axis + independent directions, b=0.6 | MOSTLY ONE AXIS | 0.793 | 0.3601 |
| one axis + independent directions, b=0.8 | SEVERAL | 0.708 | 0.3964 |
| stereotype: one axis, uneven spacing φ=(0,.1,1,1.1,1.2) | MOSTLY ONE AXIS | 0.766 | 0.2833 |
| weak ordinal, amplitude 0.15 | ONE AXIS | 1.151 | 0.0030 |
| weak ordinal, amplitude 0.3 | ONE AXIS | 1.046 | 0.0127 |
| weak ordinal, amplitude 0.6 | ONE AXIS | 1.020 | 0.0485 |
| ordinal, rare top categories | ONE AXIS | 1.015 | 0.1112 |
| null (no signal in X) | NO SIGNAL | -0.823 | 0.0000 |

**Known limit, declared:** the ordinal model assumes the categories sit evenly in the sense of the cumulative logit. A
*true* single axis with very uneven category spacing (the stereotype plant) reads MOSTLY ONE AXIS, so
a reading below ONE AXIS means "not a proportional-odds axis", not necessarily "more than one direction". The free
contrasts' mutual score correlations and the singular-value shares are reported to separate the two cases.
