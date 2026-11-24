# A-bis — arm count with the covariate in the baseline (24 Nov 2026)

Branch `almagest-geometry`. Descriptive. The section below is hashed; `artifacts/axes_a_bis.py run` refuses on a mismatch.
**Part A's hashed arm-count reading (SEVERAL, `058e4dd1`) stays on the record whatever this says.**

## Pre-registration

### The request, verbatim (user, 24 Nov)

2. A-bis (small, hashed separately before running): arm count only. The baseline becomes marginal + the "can't tell" covariate (the same covariate model fitted without embeddings), so the gap and share measure only what A1 adds. Same readings, same plants check. Report it beside the original; the original's hashed SEVERAL stays on the record.

### Operational detail

**Status.** Written after Part A's result was seen, including its no-covariate sensitivity (0.953 cleaned, 0.997 raw) and the
covariate's effect on the two models. A-bis is therefore not blind to the direction the answer may go. It is hashed before any A-bis
number on real votes exists.

**Unchanged from Part A** (`axes_a_ordinal`, imported, not copied): rows, soft targets, vote weights, standardisation, seeds,
folds, λ grid, both models, the 1,000-galaxy bootstrap and its seed, and the three untrained seeds. The ordinal and free fits must
reproduce Part A's test log-likelihoods; the differences are reported, and any above 1e-6 voids the run.

**Baselines.** The covariate enters the two models differently: one slope in the ordinal model, one per category in the free
model. Each model's baseline is *its own* covariate model with no embeddings: ORD-COV (thresholds + one slope) and FREE-COV
(intercepts + K slopes), unpenalised, fitted on train.
- **gap** = LL_free − LL_freecov (A1's contribution in the free model);
- **captured share** = (LL_ord − LL_ordcov) / (LL_free − LL_freecov).

**Readings, first match applies, same as Part A:** NO SIGNAL (gap CI lower bound ≤ 0); ONE AXIS (share CI lower bound ≥ 0.9);
MOSTLY ONE AXIS (share point ≥ 0.75); SEVERAL otherwise. Resamples with gap ≤ 0 leave the share undefined and are counted. A share
above 1 is possible (the ordinal model's single slope leaves covariate structure that w·x can pick up) and reads by the same table.

**Reported without a reading.** (i) The share with FREE-COV as the common baseline of both models, which charges the ordinal model for
the covariate structure its single slope cannot carry; (ii) Part A's marginal baseline, recomputed, which must match; (iii) each
covariate's own gain over the marginal; (iv) the three untrained seeds. Nothing else changes: the ordinal w is Part A's, so τ,
visibility, geometry and the strip are not repeated.

**Declared limits.** The baselines differ between numerator and denominator by construction, so this is "what A1 adds within each
model", not a nested likelihood ratio. Part A's limit on uneven category spacing still applies.

**Plants (D28, `a_bis_plants.json`).** Synthetic soft votes with the real vote counts on A1's cleaned arm-count rows, through the
identical code; the covariate is the real "can't tell" fraction and no arm-count fraction enters. Readings reached: MOSTLY ONE AXIS, NO SIGNAL, ONE AXIS, SEVERAL.

| plant | reading | share [95%] | gap | covariate gain free / ordinal | Part A baseline's share |
|---|---|---|---|---|---|
| one axis, no covariate effect | ONE AXIS | 1.015 [1.013, 1.016] | 0.3141 | -0.0001 / -0.0002 | 1.015 |
| one axis + covariate, strong | ONE AXIS | 1.011 [1.007, 1.016] | 0.3047 | +0.0253 / +0.0002 | 0.935 |
| one axis + covariate, weak (real-size gap) | ONE AXIS | 1.065 [1.047, 1.084] | 0.0108 | +0.0283 / +0.0020 | 0.346 |
| independent directions + covariate (several) | SEVERAL | 0.202 [0.191, 0.213] | 0.3113 | +0.0198 / +0.0002 | 0.190 |
| one axis + independent b=0.3 + covariate | ONE AXIS | 0.918 [0.912, 0.923] | 0.3325 | +0.0239 / -0.0002 | 0.855 |
| one axis + independent b=0.5 + covariate | MOSTLY ONE AXIS | 0.846 [0.839, 0.852] | 0.3565 | +0.0217 / -0.0002 | 0.797 |
| one axis + independent b=0.7 + covariate | MOSTLY ONE AXIS | 0.767 [0.758, 0.776] | 0.3913 | +0.0201 / -0.0002 | 0.729 |
| weak mixture + covariate (real-size gap) | MOSTLY ONE AXIS | 0.862 [0.842, 0.881] | 0.0217 | +0.0268 / +0.0005 | 0.395 |
| null in X, covariate only | NO SIGNAL | 0.634 [nan, nan] | -0.0000 | +0.0289 / +0.0003 | 0.011 |
| null in X, covariate only, A1-correlated covariate kept | NO SIGNAL | 0.446 [-0.569, 3.584] | 0.0000 | +0.1001 / +0.0017 | 0.017 |
