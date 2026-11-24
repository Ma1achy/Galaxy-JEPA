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
