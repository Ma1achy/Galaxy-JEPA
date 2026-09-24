# Brief W — findings

This brief corrects the record (W1), names PC1 and PC2 (W2), and unblocks pitch angle (W3).
Before hashing, every test here has passed both standing rules:
- **D27:** every outcome has a state.
- **D28:** a planted positive fires, and the null does not saturate the statistic's range.

## W1 — the record, corrected (no computation)

- **The energy test is void.** Recorded in `v_findings.md` §V3 ("Corrected (Brief W1)"). The
  salience result is the held-out recovery test: no human concept is one of the encoder's own
  components (≤ 0.63 of the probe margin; 15/33 no better than untrained).
- **D28**, a sibling to D27: before hashing, a planted positive fires and the null does not
  saturate. Recorded in `DECISIONS.md`, `CLAUDE.md` and memory.
- **U3 is restated** in `u_findings.md`, D26 and the README:

  | axis | standing |
  |---|---|
  | roundness | ordered, and tracks measured axis ratio |
  | bulge prominence | mostly ordered (the top end is underpowered), and tracks measured B/T |
  | winding | ordered, but untested against a physical measurement |
  | arm count | ordered, but substantially a visibility gradient; not established as morphology |

## W2 — naming PC1 and PC2

### Pre-registration (written 2026-09-24, before any PC–candidate correlation was computed)

**Object.**
- V3's covariance PCA of M's pooled embedding (`DEFAULT_LAYER` = −2), fitted on P2's train (A).
  PC1 carries 24.9% of the variance and PC2 11.9%.
- Scores are computed for every galaxy in P2's union (74,829).
- **Declared deviation from the brief's "full corpus":** embeddings are banked for the union only,
  a random third of the probe corpus. Extracting the other 155k would cost hours for no change in
  what is being asked.

**Candidates, confirmatory.** All are label-free and fixed now.

| candidate | definition |
|---|---|
| **padding** | `padded_fraction`: the edge-connected invalid fraction of the stamp (`data.validity.invalid_planes`, E1's measure). `padded_status`: fraction > 0.01. `interior_invalid`: saturated cores and other interior invalid pixels |
| **sky** | `sky_g/r/i`: per-band median of valid pixels at r > 2·R_p from the centre (r > R_p if fewer than 1,000 pixels remain), after 3 rounds of 3σ clipping, on the **normalised stamp the encoder saw** |
| **noise** | `noise_g/r/i`: 1.4826 × MAD of those pixels |
| **crowding** | see below |
| **colour** | `g_r`, `u_r`: SDSS DR7 model magnitudes (VizieR II/294, CDS XMatch within 1″, nearest). All 74,829 match; DR7 r − our `modelMag_r` has a median of 0.001 |

- **Crowding** (`crowd_count`, `contam_flux`), in the r band:
  1. Subtract the sky and threshold at 5σ of the noise.
  2. Take 8-connected components of ≥ 5 pixels.
  3. The target is every component reaching within R_p of the centre. `crowd_count` is the number
     of the other components.
  4. `contam_flux` is their summed above-sky flux divided by the target's.
- R_p = `petroRad_r` / 0.396″, clipped to [3, 100] px.

**Candidates, exploratory** (suggested by V3's grids, labelled as such):
- `stamp_colour_gr`: the mean g − r channel difference within R_p, in normalised units;
- `centroid_offset`: the light-weighted centre's distance from the stamp centre;
- `stamp_flux`: total above-sky r-band flux in the stamp;
- `target_area`: the pixel area of the target component, i.e. apparent size as the stamp shows it.

**Statistic.**
- Spearman(PC score, candidate) over the union.
- p from a 10,000-draw permutation of the candidate, two-sided, add-one.
- **BY across the confirmatory family: 13 candidates × 2 PCs = 26**, rank-1 threshold 5.0e-4.
  `assert_null_resolution` passes.
- Exploratory p's are reported raw, never in the family.
- **States**, per PC × candidate:
  - **NAMES**: |ρ| ≥ 0.5;
  - **CONTRIBUTES**: 0.3 ≤ |ρ| < 0.5;
  - **WEAK**: 0.1 ≤ |ρ| < 0.3;
  - **NEGLIGIBLE**: significant but |ρ| < 0.1, where significance and magnitude disagree;
  - **NONE**: not significant.

  NAMES, CONTRIBUTES and WEAK all require significance.
- **Sign** is reported but carries no state of its own. PCA signs are arbitrary, so a sign flip is
  not an outcome here.

**Jointly.**
- The out-of-sample (5-fold) R² of the rank-normal PC score on the rank-normal confirmatory
  candidates. Two models:
  - linear;
  - gradient-boosted trees (`HistGradientBoostingRegressor`), which catch non-monotone relations
    such as a U-shape in centring.
- **Headline = trees.** EXPLAINED ≥ 0.5 > PARTLY EXPLAINED ≥ 0.2 > UNEXPLAINED.
- Repeated with the exploratory candidates added, and reported separately.
- **Insufficient sample:** not applicable; n ≈ 74,800, and rows with any missing candidate are
  dropped and counted.

**Learned or architectural.**
- For every candidate: the max |ρ| over the untrained draws' own PC1 and PC2 (each draw's own
  covariance PCA on A).
- For M's PC1 and PC2: the max |Spearman| with any of each untrained draw's top 10 component
  scores.
- **ARCHITECTURAL** if M's PC has |ρ| ≥ 0.5 with an untrained top-10 component, or if the
  candidate that names it also reaches |ρ| ≥ 0.3 on an untrained top-2 component.
- **LEARNED** otherwise.

**The padding reading.** If padding NAMES PC1, that is recorded **prominently**: an artefact
dominating a quarter of an SSL representation's variance is a data-preparation finding about
pretraining on survey cutouts. If nothing reaches CONTRIBUTES and the trees' R² < 0.2, the axis is
recorded as **UNEXPLAINED**, with the grids.

**D28 planted checks (run 2026-09-24, before this hash; `out/w2_planted.json`).**

| check | planted | result | pass |
|---|---|---|---|
| crowding counter: point sources (σ 1.5 px, peak 10σ) injected outside 2R_p into 299 real stamps | k = 1 and k = 3 sources | recovered **0.98 and 0.98** per source | ✓ |
| crowding counter, planted negative | pure Gaussian noise stamp | count **0** | ✓ |
| crowding, not saturated (T2's bright-neighbour flag fired on 88%) | 300 real stamps | counts 1 / 4 / 6 / 8 / 12 at the 5/25/50/75/95th percentiles; 0.3% zero | ✓ |
| sky and noise estimator | sky 0.7, noise 0.2, a Gaussian galaxy | 0.701, 0.200 | ✓ |
| padding measure | a constant 10% frame strip | 0.098 | ✓ |
| correlation test, positive, at the real family size (26) | a covariate at ρ ≈ 0.3 with PC1 | ρ 0.307, p 1e-4 → **CONTRIBUTES** (BY rank-1 5.0e-4) | ✓ |
| correlation test, negative | a shuffled copy | ρ −0.003, p 0.46 → NONE | ✓ |
| null range | at n = 74,829 the permutation null of ρ has sd ≈ 0.004, so an extreme ρ reaches the 1e-4 floor | the null does not saturate [−1, 1] | ✓ |
| joint R², positive | a noisy copy of PC1 (ρ ≈ 0.8) among 5 noise columns | trees 0.638 → **EXPLAINED** | ✓ |
| joint R², negative | 5 noise columns | trees −0.001 → UNEXPLAINED | ✓ |

**Expectation**, from V3's grids:
- **PC2 ↔ noise level.** The high end showed visibly noisy backgrounds. I expect `noise_r`
  CONTRIBUTES or NAMES.
- **PC1 ↔ colour and crowding.** The low end is redder and more crowded; I expect `u_r` or
  `crowd_count` CONTRIBUTES.
- Padding WEAK at most: one padded stamp appeared in 48.
- Joint: PARTLY EXPLAINED for both.
- I expect both axes to be LEARNED. The untrained spectrum is collapsed onto one direction (PR 1.6).

*W2 pre-registration ends here: first 124 lines, SHA-1 `2143b69978cb7be8b6c400a8047f5e6a41c57532`.*

### W2 — result (`out/w2_name_pcs.json`, `out/w2_stamp_axes.npz`; the planted checks in `out/w2_planted.json`)

**Both axes: UNEXPLAINED.** No candidate reaches WEAK, let alone CONTRIBUTES.

| candidate | PC1 ρ | PC2 ρ | untrained top-2, max \|ρ\| |
|---|---|---|---|
| padded fraction | +0.010 (NEGLIGIBLE) | −0.001 (NONE) | 0.08 |
| padded status | +0.010 (NEGLIGIBLE) | −0.001 (NONE) | 0.08 |
| interior invalid | −0.000 (NONE) | −0.001 (NONE) | 0.01 |
| sky g / r / i | −0.024 / −0.031 / −0.026 | −0.049 / −0.049 / −0.054 | **0.50 / 0.59 / 0.68–0.71** |
| noise g / r / i | +0.002 / −0.018 / −0.002 | −0.014 / −0.023 / −0.028 | 0.10 / 0.19 / 0.20 |
| crowd count | −0.033 | +0.006 (NONE) | **0.32–0.33** |
| contaminating flux | −0.053 | +0.041 | 0.25 |
| g − r | −0.081 | +0.034 | 0.26–0.31 |
| u − r | −0.077 | +0.025 | 0.29–0.33 |
| *stamp colour (expl.)* | +0.084 | −0.039 | 0.23–0.28 |
| *centroid offset (expl.)* | +0.041 | −0.020 | 0.09–0.10 |
| *stamp flux (expl.)* | +0.014 | −0.088 | **0.79–0.82** |
| *target area (expl.)* | +0.061 | −0.079 | **0.40–0.42** |

- Every confirmatory significant result is **NEGLIGIBLE**: significant at n = 74,829 but |ρ| < 0.1.
  This is significance and magnitude disagreeing, and D27's state reports it as such.
- **Jointly, out of sample** (5-fold):

  | | trees R², confirmatory | linear R² | with exploratory | state |
  |---|---|---|---|---|
  | PC1 | 0.011 | 0.010 | 0.015 | **UNEXPLAINED** |
  | PC2 | 0.012 | 0.011 | 0.019 | **UNEXPLAINED** |

  The same models reached 0.64 on the planted positive, so the test could see an explanation.
- **Padding does not name PC1** (ρ = +0.010). 12.9% of union stamps are padded; the axis ignores
  them. The data-preparation alarm is not raised.
- **Learned, not architectural.**
  - M's PC1 and PC2 correlate ≤ 0.04 and ≤ 0.10 with every untrained draw's top-10 components.
  - The untrained encoders' own top components *are* named by these candidates: stamp flux at
    0.8, sky at 0.5–0.7, target area at 0.4 and crowding at 0.33.
  - The architecture organises its variance around stamp-level photometry. Pretraining replaced
    that with two axes that no stamp-level quantity measured here describes.
- **Against my stated expectation.** I expected noise to name PC2 and colour or crowding to name
  PC1; all are negligible. I expected padding WEAK at most: it is negligible. I expected LEARNED:
  it is. The extremes in V3's grids looked different in size, colour, crowding and noise. The bulk
  is not organised by any of them.

**Exploratory, post hoc** (`out/w2_exploratory_posthoc.json`, `out/w2_ring_posthoc.json`,
`out/w2_pc12_plane.png`). These were added after the result, so no verdict.
- **Not outlier channels.** Each component's loadings spread over about 90 of the 384 dimensions
  (loading participation ratio 90 and 96; the top 5 dimensions hold 14%). No raw dimension carries
  more than 1.3% of the variance.
- **No observation systematic.**
  - Camcol η² is 0.003 and 0.001.
  - Run η² (238 runs) is 0.004 and 0.005, against 0.003 for shuffled scores.
  - |ρ| with RA, Dec and field is ≤ 0.02.
- **Bounded and flat-topped.** The excess kurtosis is −0.80 (PC1) and −0.56 (PC2); PC3–PC6 are
  +0.3 to +1.6.
  - The (PC1, PC2) plane is a square-edged block, each axis bounded near ±2.3 sd, and the two look
    roughly independent.
  - **Not a ring:** the radius CV is 0.44 against 0.52 for a Gaussian, and the angle is uniform.
  - Not position angle: every circular correlation with θ or 2θ is ≤ 0.03.
- **Proposed next measurement, not run.** A mirror flip changes none of the scalars measured here,
  but pretraining had no flip augmentation (D10). Embed about 2,000 stamps and their mirror
  images, and ask whether PC1 or PC2 changes sign or shifts under reflection. That decides whether
  the axes encode image **parity or handedness**: frame-level information that nothing in a
  catalogue describes, and that a rotation- and flip-free JEPA is free to spend variance on.
- **The axes stay named "unexplained".** Grids: `out/v3_pc1_{low,high}.png`,
  `out/v3_pc2_{low,high}.png`.

## W3 — pitch angle: unblocking by substitution

### Verification of Yu & Ho (2020, ApJ 900, 150; doi:10.3847/1538-4357/abac5b), before any join

| check | result | pass |
|---|---|---|
| **A machine-readable table exists** | IOP links only `apjabac5bt1_ascii.txt`, and it holds the **10 rows** printed in the paper. No `_mrt` file exists (404). Not on VizieR (`J/ApJ/900/150` not found). Two follow-ups on the same sample release none either: Yu, Ho & Wang 2022 (AJ, doi:10.3847/1538-3881/ac88c5) and the 2024 grand-design/multi-armed paper (AJ, doi:10.3847/1538-3881/ad46fb) | **✗** |
| Pitch angles measured algorithmically, with no reference to Galaxy Zoo votes | Yes. 1-D and 2-D Fourier decomposition of deprojected SDSS r-band images. The paper contrasts its method with Galaxy Zoo's vote-calibrated estimates | ✓ |
| Method | 2-D Fourier transform pitch angle, arm strength and f₃ on NSA v0.1.2 galaxies | ✓ (recorded) |
| Overlap with our 230k | **cannot be computed without the table.** The selection (NSA, 0.005 ≤ z < 0.03, r < 14.5, ellipticity < 0.5, 4,378 galaxies) is bright and nearby, well inside GZ2's r < 17, so the overlap is plausibly most of the sample | — |

**Verdict: FAILS on the first condition. As the brief directs, stop and report.**
- 2b and Experiment D are **not run**.
- The design pre-registered in V2 still stands and runs unchanged on whichever table arrives first.
- For Yu & Ho, the visibility-control and substitution statements would be added before its hash.
- Yu & Ho would **substitute** for Hart, not replicate it: a different method (2-D Fourier against
  SpArcFiRe arc fitting) and a different sample (bright nearby NSA galaxies against GZ2's
  stellar-mass-complete spirals).

### Data requests, drafted for you to send

Both authors' contact details are in the papers. I have not looked up or guessed addresses.

**1. To the corresponding author of Hart et al. (2017), MNRAS 472, 2263:**

> Subject: Request for SpArcFiRe pitch angles from Hart et al. (2017)
>
> Dear Dr Hart,
>
> I am working on a label-free study of Galaxy Zoo 2 morphology using a self-supervised image
> encoder, and I would like to test whether its representation of spiral-arm winding tracks a
> measurement made independently of the volunteers' votes.
>
> Would you be willing to share the machine-measured SpArcFiRe pitch angles (ψ_galaxy) for the
> galaxies in your sample with reliable arcs? I need only that subset, together with the SDSS DR7
> objIDs and the reliability flag. I will not use the GZ2-calibrated pitch angles (your Eq. 8),
> because my test depends on the measurement being independent of the votes.
>
> I will cite the paper, and I'm happy to share the results with you before anything is
> circulated.
>
> With thanks,
> [your name, affiliation]

**2. To the corresponding author of Yu & Ho (2020), ApJ 900, 150:**

> Subject: Request for the full Table 1 of Yu & Ho (2020)
>
> Dear Dr Yu,
>
> I am studying how a self-supervised image encoder trained on SDSS galaxies represents
> spiral-arm winding, and I would like to compare it with your Fourier-based pitch angles. They are
> exactly the kind of algorithmic, vote-independent measurement the test needs.
>
> The journal's machine-readable file for Table 1 contains only the ten rows printed in the paper.
> Would you be willing to share the full table for the 4,378 galaxies: the NSA IDs, coordinates,
> pitch angle with its uncertainty, and arm strength?
>
> I will cite the paper, and I'm happy to share the results with you.
>
> With thanks,
> [your name, affiliation]
