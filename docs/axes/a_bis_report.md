# A-bis — arm count with the covariate in the baseline (24 Nov 2026)

Branch `almagest-geometry`. Pre-registration `prereg_a_bis.md`, SHA-1 `659d2540…`, hashed and committed (`8e744fa`) before the run.
Numbers `a_bis.json`, plants `a_bis_plants.json`, code `artifacts/axes_a_bis.py`.
**Part A's hashed SEVERAL (`058e4dd1`) stays on the record.** A-bis was written after Part A's result was seen.

**Result: cleaned MOSTLY ONE AXIS, raw ONE AXIS, on the matched baselines.** The reading depends on which baseline is used: on a
common free-covariate baseline the share is 0.25 cleaned and 0.58 raw.

| | Part A (marginal baseline) | **A-bis (matched baselines), reading** | common free-cov baseline (no reading) |
|---|---|---|---|
| cleaned A1: share | 0.587 [0.532, 0.637], SEVERAL | **0.909 [0.841, 0.987], MOSTLY ONE AXIS** | 0.253 [0.067, 0.379] |
| raw A1: share | 0.717 [0.674, 0.759], SEVERAL | **0.976 [0.933, 1.025], ONE AXIS** | 0.577 [0.483, 0.651] |
| cleaned A1: gap (nats/vote) | 0.062 (free − marginal) | 0.0343 [0.0283, 0.0403] | same |
| raw A1: gap | 0.084 | 0.0565 [0.0493, 0.0629] | same |
| untrained, cleaned: gap | | 0.0122, 0.0127, 0.0117 | |
| untrained, raw: gap | | 0.0363, 0.0375, 0.0349 | |

- The fits reproduce Part A exactly: the test log-likelihoods of all models, all encoders, differ by 0.0.
- The gap is what A1 adds over the covariate alone. A1's is 2.8 times the untrained seeds' cleaned and 1.5 times raw.
- **What the covariate explains on its own:** 0.028 nats per vote in the free model, 0.005 in the ordinal. The free model's slopes
  are −0.57 for two arms and +0.40 for more than four; one slope across all categories cannot carry that.
- **Which baseline?** The request says "the same covariate model fitted without embeddings". I read that as each model's own
  covariate model (declared in the pre-registration). With one shared free-covariate baseline the ordinal model is charged for the
  covariate structure its single slope cannot carry, which is the covariate's fault, not the embedding's. So A-bis
  measures how well one axis carries *A1's addition*, not whether one ordinal model with a single covariate slope
  predicts the votes as well as the free one.
- **Plants:** all four readings reached, including ONE AXIS on a one-axis truth with a strong covariate effect (the Part A
  baseline reads the weak version of it 0.35, i.e. SEVERAL), and SEVERAL on independent directions with covariate (0.20).
  A one-axis truth mixed with weak independent directions at about A1's gap reads MOSTLY (0.86).
- **Untrained also read MOSTLY (cleaned) and ONE AXIS (raw)**, with wider intervals. As in Part A, the share measures one
  axis versus several, not learning; compare the gaps.

**Standing:** the *sensitivity* that Part A declared (no covariate: 0.953 cleaned, 0.997 raw) and A-bis agree in direction. Part A's
SEVERAL is, on this evidence, the covariate misfit of the ordinal model, not several A1 directions for arm count. It remains the
hashed reading; it is not withdrawn.
