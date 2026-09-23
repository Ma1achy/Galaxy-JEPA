# Brief S — close-out findings

Cheap and descriptive. No new pretraining, and no rung changes intended. Encoder: `runs/m/encoder.pt`
(M, 4-epoch). Split: the frozen 40,000 / 34,829.

---

## S1 — spiral × inclination

### Pre-registration (written 2026-09-23, before any S1 number was computed)

**Question.** R2 found spiral's conditional-mean path is an arc that survives the reach split.
PC2 is an inverted U peaking at the undecided middle (f ≈ 0.55), so uncertain-vote galaxies are
displaced in one common direction off the line. Hypothesis: that direction is **inclination**.
Spiral arms become ambiguous at intermediate tilt, which is D13's projection confound.

**Construction.**
- *Galaxies:* spiral-eligible (`t04_spiral_a08_spiral`) galaxies in P2's union. Embeddings are
  z-scored per dimension over the union, as in R2.
- *Bend direction:* the **second eigenvector of the cross-fitted path covariance S**, R2's own
  construction (10 equal-width bins, ≥100 per bin, 2 split-halves). It is estimated on **train
  galaxies only**, and every statistic below is measured on **test galaxies only**, so the
  direction is not fitted to the galaxies it is scored on. The eigenvectors of S are orthogonal, so
  this is also the leading **off-line residual direction**: the path's PC2, orthogonal to its
  best-fit line.
- *Sign:* oriented so that the undecided-middle bins (centres 0.35–0.65) project above the end bins
  (centres ≤ 0.25 and ≥ 0.75). Positive displacement means displaced the way uncertain-vote
  galaxies are.
- *Inclination proxy:* `expAB_r`, the exponential-profile axis ratio (b/a). Spirals are discs
  (D13). It is streamed from `data/probe/metadata.csv`, as in R3.
- *Partial correlation, controlling for vote fraction:* both the displacement s = z·d and `expAB_r`
  are residualised on **20 equal-width vote-fraction bins (bin-mean removal)**. That removes any
  shape of dependence on f, including the inverted U, which a linear control would leave in.
  Pearson r of the residuals. 95% CI from 2,000 galaxy bootstraps.
- *Bend share explained:* the U-amplitude is a = mean s over the middle bins minus mean s over the
  end bins. Refit it after removing inclination's within-f linear effect,
  s′ = s − β·(b/a − mean), with β the pooled within-bin slope. **Share explained = 1 − a′/a.**
  This asks the attribution question at the level of the bend itself: per-galaxy r is diluted by
  within-bin scatter, which dwarfs the bin-mean displacement.
- *Null:* `expAB_r` shuffled across the same galaxies, residualised again each draw, with
  **B = 10,000**. p is two-sided and add-one.
- *Populations:* the **full** spiral-eligible population, and the **well-voted half** (question
  reach ≥ its median, reach = sum over t04's answer counts, as in `r2_reach_split.py`). The
  direction is re-estimated inside each population.
- *Untrained reference:* the same pipeline on each of the three banked untrained seeds, each with
  its own bend direction.

**Physically sensible sign: NEGATIVE.** Displacement toward the undecided middle should go with
more inclination, i.e. *lower* b/a.

**Pre-registered read** (both populations must agree for a verdict):

| verdict | condition, on BOTH full and well-voted half |
|---|---|
| **the bend direction is inclination** | partial r ≤ −0.10, permutation p ≤ 0.001, bootstrap CI excludes 0, **and** share explained ≥ 0.5 |
| **inclination contributes** | partial r < 0 with p ≤ 0.001 and CI excluding 0, and share explained ≥ 0.2, but not the above |
| **not inclination** | anything else, including a significant *positive* r, which is the wrong sign |

Where the two populations disagree, the finding is reported as population-dependent, not given a
verdict. **Learned or architectural** (descriptive, K = 3, a range): *learned* iff M's partial r
lies beyond all three untrained seeds' values, on the negative side.

**If not inclination:** the same partial r is computed against size (`petroRad_r`), magnitude
(`modelMag_r`), SNR (`snr_r`) and PSF (`psfWidth_r`), **labelled exploratory, with no verdict and no
attribution forced.**

**Truncation note.** The spiral question is reached mainly by featured, not-edge-on galaxies, so
the well-voted half spans face-on to moderate tilt, the regime where arm visibility degrades. The
full population also holds low-reach galaxies that reached t04 with only a few votes, including
some that are more inclined.

*The pre-registration above was hashed (`git hash-object`: `8574452c`) before the analysis ran.
Script: `artifacts/s1_spiral_inclination.py`. Record: `artifacts/out/s1_spiral_inclination.json`.*

### Result — **not inclination**, by the pre-registered read

Spiral-eligible galaxies: 38,003 train, 33,082 test. Median question reach is 6. `expAB_r` is
finite on 100%.

| encoder | population | partial r (95% CI) | perm. p | share of bend explained | U-amplitude → after |
|---|---|---|---|---|---|
| **M** | full | **−0.162** (−0.173, −0.152) | 1e-4 | **0.087** | 1.024 → 0.935 |
| **M** | well-voted | **−0.052** (−0.068, −0.038) | 1e-4 | **0.027** | 1.143 → 1.111 |
| untrained s0–s2 | full | −0.089 to −0.091 | 1e-4 | 0.15–0.19 | 0.34–0.42 → 0.28–0.35 |
| untrained s0–s2 | well-voted | −0.047 to −0.051 | 1e-4 | 0.12–0.16 | 0.21–0.26 → 0.17–0.23 |

- **The sign is physically right, and the correlation is real.** Negative in every row, with the
  permutation p at its floor. Displacement toward the undecided middle does go with more
  inclination.
- **But inclination does not explain the bend.** The attribution needs a share explained of at
  least 0.2 on both populations for *contributes*, and at least 0.5 for *is*. Removing inclination's
  within-f effect removes **9%** of the U-amplitude on the full population and **3%** on the
  well-voted half. Partial r clears −0.10 only on the full population (−0.162), and not on the
  well-voted half (−0.052), where the low-reach mixture is removed. **Verdict: not inclination.**
- **Learned or architectural (descriptive, K = 3):** on the full population M's r (−0.162) lies well
  beyond the untrained range (−0.089 to −0.091), so the inclination component there is **learned**.
  On the well-voted half M (−0.052) sits 0.002 beyond the untrained range, far inside the CI width,
  so there is no meaningful difference: **architectural**. Training grew M's bend about threefold
  (U-amplitude 1.0 against 0.3–0.4) without adding inclination to it.

### Exploratory — labelled as such, no attribution forced

The pre-registration asks for this because the verdict is *not inclination*. It uses the same
partial correlation (20-bin f removal, test galaxies, M's own bend direction), with no null, no
share test and no untrained reference:

| nuisance | partial r, full | partial r, well-voted |
|---|---|---|
| magnitude (`modelMag_r`; larger = fainter) | **+0.652** | **+0.671** |
| SNR (`snr_r`) | **−0.616** | **−0.622** |
| size (`petroRad_r`, flagged rows excluded) | **−0.534** | **−0.584** |
| PSF width (`psfWidth_r`) | +0.019 | −0.021 |

Uncertain-vote spirals are displaced toward **fainter, smaller, lower-SNR** images. The effect is
equally strong in both populations, so it is not the reach mixture. It reads like an *image-
information* axis: the galaxies volunteers could not decide on are the ones whose images carry the
least. **That is a reading, not an attribution.** Magnitude, SNR and size are strongly
intercorrelated and were not separated, and none was pre-registered as the hypothesis.

**For the uncertainty-geometry brief:** the bend it would project onto a line is mostly not
inclination (D13's confound). A direction that tracks faintness, smallness and SNR carries it. If
that brief conditions on anything, it should be those. It should also pre-register the separation
(e.g. partial on each, controlling the other two) rather than inherit this exploratory look.

---

## S2 — one D-entry for the ladder's survival test (**D24**)

- **Decision** (`DECISIONS.md` D24): the ladder's nuisance clearance applies O1's retention rule in
  place of *matched AUC ≥ effect floor*. The recorded principle: the effect floor is an absolute
  clean-vs-marginal threshold and never tests anything relative (second instance, after the power
  rule).
- **Margin floor:** retention is judged only where the unmatched margin passed D23 existence;
  otherwise UNRESOLVED. That covers the noise-over-noise case for the denominator. One residual is
  declared: the retention bar is a K = 3 mean, not existence's K = 30, which is worth up to ~18% on
  the ratio for a feature sitting exactly at the BY bar.
- **K = 3** for both C and C_m (R0's construction). The ladder refuses fewer.
- **Verification** (`artifacts/s2_verify.py`, a full production rerun, both populations):
  - **Reproduces R0 exactly:** 33/33 and 29/29 verdicts; max |Δ| = 0 on A, C, M, M_lo and C_m.
  - **Expected flips confirmed:** `t10 winding: medium` (full) COLLAPSES → UNRESOLVED, as the brief
    expected. Two more follow from the same rule: `t04 spiral` and `t04 no spiral`
    (conditional; both fail existence there) SURVIVES → UNRESOLVED. I predicted these in D24 before
    the rerun.
  - **Rung counts: full unchanged; conditional R1 1 → 2.** `t08 odd: other` becomes a clean linear
    direction. It had been "confounded by magnitude" only by the floor arithmetic; it retains 110% of
    its margin, is not entangled, and is not underpowered. I named it as a candidate before the rerun
    and report it as D24's consequence. It is a rung change, and this brief intended none.
  - P's 22 "confounded" (full) are now **12 entangled, 8 below the effect floor, 2 unresolved**.
- **Found and flagged, not fixed:** the effect floor still tests something relative in the
  entanglement leg (`_entangled_map`, a third instance) and in the MLP decode threshold. Fixing
  either can change rungs, so each needs its own brief (`TODO.md`).
- **Record gap closed:** failing verdicts now carry their clearance (`_failing_rung(..., matched)`).

## S3 — D8 scoped

Recorded against D8 in `DECISIONS.md`. Shallow votes attenuate toward chance for **readout**, but
**conditioning** on the fraction (binning, projecting, stratifying) mixes vote-reach groups and
manufactures structure: edge-on, where both halves are straight and only the mixture bends. The
unfiltered population is right for verdicts. A fraction-conditioning measurement needs a local,
pre-registered reach floor. Cross-referenced to the uncertainty-geometry TODO.

## S4 — D10 revised

D10 now reads: **no rotation/reflection augmentation across the encoder family**, divergence
recorded, orientation measured. The four reasons are in the entry: retraining M invalidates the
record; orientation is mostly architectural (R3); the angle is not a morphology confound while
elongation is D13's; and D12 needs one symmetry policy. **Forward constraint**, flagged in the
entry and in `TODO.md` against the MoCo baseline: remove horizontal flips, since they are
reflections. Check MAE's pipeline too. Decided in the baselines brief.

## S5 — R3 criterion check: it changed, and it matters for M

The first Brief R plan required effective dimension **≈ 2**. The amended plan, written before any
R3 data, required **≥ 1.5**, and the code implements ≥ 1.5. No reason was recorded at the time; one
is now in `r_findings.md` §R3: a non-planar closed loop is still a loop, and non-1-D is what the
instrument had to show. **Under the original criterion M's circle (3.54) would not have been
recovered**, so R2's validation on M rests on the amended criterion. That is stated there
rather than left as an unqualified "recovered".
