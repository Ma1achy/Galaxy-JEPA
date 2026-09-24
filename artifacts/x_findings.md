# Brief X — findings

## X1 — is PC1/PC2 spiral handedness?

### Pre-registration (written 2026-09-24, after the D28 planted checks; before any X1 statistic on M or the untrained draws)

**The hypothesis.** W2 left PC1 (24.9%) and PC2 (11.9%) learned but UNEXPLAINED. Spiral
handedness would fit: a large, learned, label-orthogonal distinction that no vote asks about and
no nuisance tracks. The axis may be PC1, PC2 or a rotated direction in their plane.

**Parity (established before this pre-registration).** Our stamps are the SDSS frame pixel array,
cut by `Cutout2D` and written without WCS, so parity is the frame's det(CD).
- det(CD) < 0 in all 10 DR17 frames sampled (dec −3.6° to +66°, six camcols).
- Empirical check: 7 galaxies from three near-axis-aligned frames (runs 1140, 1231, 2141). The
  dihedral transform best mapping the SkyServer JPEG onto our r-band stamp was a transpose
  (`mirror+rot270`) every time.
- **So our stamps are mirror images of the GZ1 JPEGs, up to a rotation.** GZ1 "clockwise"
  winds anticlockwise in our arrays. X1b's sign is therefore interpretable, not undetermined.

**Groups** (raw `_fraction` columns):
- **Confident spirals:** t01 features ≥ 0.8, t04 spiral ≥ 0.8, t02 edge-on "no" ≥ 0.8, and t04
  reach (spiral + no-spiral counts) ≥ 21.
- **Confident smooth:** t01 smooth ≥ 0.8.

**Coordinates.** V3's covariance PCA on A (40,000), not rescaled, per encoder. Scores are divided
by A's sd per axis, so the radius is in sd units. The untrained draws are R's three (seeds 0–2),
each in its own basis.

#### X1a — magnitude against spirality (union, 74,829; no fitting)

- **Variables:** |PC1|, |PC2|, radius √(PC1² + PC2²), and signed PC1 and PC2.
- **Targets:** t04 spiral fraction (galaxies with t04 reach ≥ 21) and t01 featured fraction (all).
- **Statistic:** Spearman, 10,000-permutation, two-sided, add-one. BY over the 12 tests
  (3 variables × magnitude/signed × 2 targets), with `assert_null_resolution` run first.
- **Bimodality:** KDE at Silverman's bandwidth, 200 bootstraps.
  - BIMODAL: two modes on opposite sides of 0, dip ≤ 0.8 × the lower mode, in ≥ 90% of
    bootstraps.
  - UNIMODAL CENTRED: one mode within 0.5 sd of 0, in ≥ 90%.
  - Otherwise NEITHER.
  - Read among confident spirals (BIMODAL expected) and confident smooth (UNIMODAL CENTRED
    expected).
- **Resolution of the bimodality check (calibrated, D28):**

  | test distribution | BIMODAL calls |
  |---|---|
  | ±1.5 sd Gaussian mixture | 100% |
  | ±1.26 sd | 4% |
  | ±1.0 sd | 0% |
  | normal | 0% |
  | uniform | 0.5% |
  | generalised normal, β = 4 (excess kurtosis −0.82, PC1's shape) | 0% |

  It does not fire on a flat-topped unimodal block, and **cannot see lobes closer than about
  ±1.5 sd**. A NEITHER or UNIMODAL call among spirals is read in that light.
- **State per variable**, the first match applies. A "hit" is BY-significant with |ρ| ≥ 0.1.

  | # | condition | state |
  |---|---|---|
  | 1 | either confident group < 300 | **INSUFFICIENT** |
  | 2 | a signed hit | **SIGNED TRACKS** (a conventional axis, not handedness) |
  | 3 | a magnitude hit with ρ > 0, spirals BIMODAL, smooth UNIMODAL CENTRED | **CONSISTENT WITH HANDEDNESS** |
  | 4 | a magnitude hit with ρ > 0 | **MAGNITUDE ONLY** |
  | 5 | a magnitude hit with ρ < 0 | **INVERSE MAGNITUDE** |
  | 6 | otherwise | **NO ASSOCIATION** |

  For the radius, which has no signed form, row 2 applies to PC1/PC2's signed tests.
- **Untrained reference:** Spearman of |PC1|, |PC2| and the radius with t04 spiral on each draw,
  as a range.

#### X1b — GZ1 handedness votes

- **Table:** GZ1 table 2 (`GalaxyZoo1_DR_table2.csv.gz`, 667,944 rows). Columns verified:
  `OBJID, …, P_CW, P_ACW, …`. OBJID is the DR7 objID; joined on `dr7objid`.
- **Rows:** confident spirals (as above) with a GZ1 match and NVOTE ≥ 10. h = P_CW − P_ACW, in
  GZ1's JPEG frame.
- **Plane direction:** φ = argmax over 180 angles in [0°, 180°) of |Spearman(cos φ·PC1 + sin φ·PC2, h)|
  on **A's** rows. Frozen, then tested on **B**. One angle is chosen; no weights are fitted.
- **Tests on B:** PC1, PC2 and the plane direction against h. Spearman, 10,000-permutation,
  two-sided, add-one, BY over 3.
- **State:**
  - INSUFFICIENT: B < 300.
  - SIGNED ASSOCIATION: any BY hit with |ρ| ≥ 0.1.
  - WEAK SIGNED ASSOCIATION: BY hit, |ρ| < 0.1.
  - NO ASSOCIATION: otherwise.
  - Descriptive: Spearman of |score| with |h|.
- **GZ1's anticlockwise excess** is a perception bias (Land et al. 2008). It shifts the mean of h;
  the per-galaxy ranking stays informative, and ranks are what is tested.
- **Parity through the classifications (secondary; it rests on one convention).** χ is the
  planted pixel chirality below, computed on our stamps for B's rows.
  - χ is higher for s = +1: arms turning anticlockwise outward in our row-down array (planted:
    AUC 1.00).
  - Mirrored, that is clockwise in the JPEG. GZ1's clockwise is Z-wise (Land et al. 2008): an arm
    traced outward turns clockwise.
  - So the mirror parity predicts ρ(χ, h) > 0.
  - States: MIRROR CONFIRMED (p < 0.05, ρ > 0); MIRROR CONTRADICTED (p < 0.05, ρ < 0);
    UNRESOLVED.
  - Then PC1, PC2 and the plane direction against χ, as an algorithmic handedness label. BY over 3,
    reported beside the GZ1 tests. **Exploratory.**

#### X1c — the symmetry test (decisive)

- **Sample:** 1,000 confident spirals (of 3,137) and 1,000 confident smooth (of 14,988), drawn
  from **B**, seed 0.
- **Embedding:** each stamp, plus `torch.rot90` by 90° and 180° and a left-right flip, all through
  the ladder's extraction path (`extract_matrix`, pooled layer −2). Encoders: M and the three
  untrained draws.
  - All three transforms fix the stamp centre (127.5) and the 16-px patch grid. Only the image
    content moves.
- **Per component** (PC1…PC10 in each encoder's own basis), on spirals:
  - r90, r180 = Pearson of the original score with the rotated score; **r_rot = min(r90, r180)**;
  - r_mir = Pearson with the mirrored score;
  - the odd part a = (s − s_mirror)/2 and the odd-variance fraction var(a) / (var(a) + var(even));
  - **conc** = var(a) on spirals / var(a) on smooth;
  - 1,000 bootstrap CIs over galaxies. The same r's on smooth are reported beside them.
- **Plane direction:** the most reflection-odd direction within (PC1, PC2), from
  max u′C_odd u / u′C u. It is fitted on a random half of the spirals and every number is read
  on the other half. The same is done within the top 10 and in the full 384-d space (descriptive).
- **Whole-embedding odd share:** tr(C_odd) / tr(C) on spirals and smooth, for every encoder.
- **States per component** (complete over (r_rot, r_mir), D27; the first match applies):

  | # | condition | state |
  |---|---|---|
  | 1 | r_rot ≤ 0.5 | **ORIENTATION-LIKE** |
  | 2 | 0.5 < r_rot < 0.8 | **ROTATION-SENSITIVE (WEAK)** |
  | 3 | r_rot ≥ 0.8, r_mir ≤ −0.5, conc ≥ 2 | **HANDEDNESS** |
  | 4 | r_rot ≥ 0.8, r_mir ≤ −0.5, conc < 2 | **REFLECTION-ODD, NOT SPIRAL-CONCENTRATED** |
  | 5 | r_rot ≥ 0.8, r_mir ≥ 0.8 | **INVARIANT TO BOTH** |
  | 6 | r_rot ≥ 0.8, −0.5 < r_mir < 0.8 | **PARTLY REFLECTION-SENSITIVE** |

- **Qualifier:** LEARNED if M's r_mir lies below the most reflection-odd top-10 component of every
  untrained draw; otherwise ARCHITECTURAL-LEVEL.

#### X1's verdict (from X1c; the first match applies)

| # | condition | verdict |
|---|---|---|
| 1 | either X1c group < 300, or any D28 check below failed | **INSUFFICIENT**, naming what would resolve it |
| 2 | PC1, PC2 or the plane direction reads HANDEDNESS | **HANDEDNESS**: recorded prominently, cross-referenced to D10 |
| 3 | PC1 and PC2 both ORIENTATION-LIKE | **ORIENTATION-LIKE** |
| 4 | PC1 and PC2 both INVARIANT TO BOTH | **INVARIANT TO BOTH**: the axes stay "unexplained" |
| 5 | otherwise | **MIXED**: each component's own state reported |

- X1a and X1b are reported beside the verdict as corroboration or contradiction. They do not
  override it.
- **Expectation:** none held strongly. W2 found PC1/PC2 blind to every stamp axis. Against
  handedness: M separates synthetic handedness only weakly (below), and its PC1/PC2 do not
  separate it at all.

#### D28 — planted checks, run before this was hashed (`out/x1_planted.json`)

**X1c path** (the identical `signature` / `symmetry_state` code). Pixel-feature "encoders" on the
same 2,000 stamps and transforms, through `extract_matrix`:

| planted feature | expected | r_rot | r_mir | conc | state | fires |
|---|---|---|---|---|---|---|
| χ: log-polar Fourier chirality | HANDEDNESS | +1.00 | −1.00 | 6.84 | HANDEDNESS | ✓ |
| left − right flux asymmetry | ORIENTATION-LIKE | −1.00 | −1.00 | 0.86 | ORIENTATION-LIKE | ✓ |
| aperture flux | INVARIANT TO BOTH | +1.00 | +1.00 | 3.55 | INVARIANT TO BOTH | ✓ |

- Three of the six states are shown reachable. The other three (weak rotation-sensitive,
  reflection-odd not concentrated, partly sensitive) are intermediate bands of the same two
  statistics.
- **The null does not saturate:** r_rot and r_mir run over their full [−1, +1] range on this path.

**X1a path.** A known spirality score, shifted so the confident-smooth median is 0, times a random
± sign per galaxy (see the results for the planted row):

| plant | lobes (sd) | magnitude ρ (spiral / featured) | signed ρ | spirals | smooth | state | fires |
|---|---|---|---|---|---|---|---|
| logistic score × ±1 | 2.46 | +0.30 / +0.55 (p 1e-4) | −0.006 / +0.001 (n.s.) | BIMODAL | UNIMODAL CENTRED | CONSISTENT WITH HANDEDNESS | ✓ |
| CAV projection × ±1 | 1.26 | +0.15 / +0.35 (p 1e-4) | −0.011 / 0.000 (n.s.) | UNIMODAL CENTRED | UNIMODAL CENTRED | MAGNITUDE ONLY | below resolution, as calibrated |

- The magnitude test fires and the signed test does not, which is what the brief asks of the plant.
- An extreme effect reaches p = 1e-4, which clears BY at m = 12, so the null does not saturate.
- SIGNED TRACKS was not separately planted. It is the same `spearman_perm` statistic on the
  unsigned score.

**Synthetic log spirals** (out of distribution; 400, half each handedness, disc + bulge + noise
matched to the real stamps' sky and peak):
- χ separates them perfectly (AUC 1.00).
- **M separates handedness only weakly:** a held-out logistic probe on the full embedding reaches
  AUC 0.68. PC1 and PC2 reach 0.52 and 0.56.
- On synthetics, M's PC1 and PC2 are ORIENTATION-LIKE (r_rot −0.64 and −0.63). OOD, descriptive.


*X1 pre-registration ends here: first 184 lines, SHA-1 `f587b22ace2eafaa9403d23f7a92f1914ebb30d7`.*

### X1 — result (`out/x1_handedness.json`; planted checks in `out/x1_planted.json`; bank `out/x1_transform_bank.npz`)

**Verdict (row 3): ORIENTATION-LIKE. PC1 and PC2 are not handedness.** Every D28 check fired, and
each X1c group holds 1,000 galaxies.

**X1c — the symmetry test.** 1,000 spirals; bootstrap 95% CIs in brackets.

| component | r90 | r180 | r_mir | conc | state | on smooth (r_rot, r_mir) |
|---|---|---|---|---|---|---|
| **M PC1** | +0.05 | **−0.94** [−0.94, −0.93] | **−0.94** [−0.94, −0.93] | 1.27 | ORIENTATION-LIKE | −0.97, −0.96 |
| **M PC2** | +0.02 | **−0.91** [−0.92, −0.90] | **+0.90** | 8.1 | ORIENTATION-LIKE | −0.95, +0.99 |
| M plane direction (held-out half, 178°) | +0.09 | −0.93 | −0.94 | 1.28 | ORIENTATION-LIKE | −0.97, −0.97 |
| untrained PC1, PC2 (all three draws) | +1.00 | +1.00 | +1.00 | 1.0–1.8 | INVARIANT TO BOTH | — |

- **Qualifier:** PC1 LEARNED (its r_mir of −0.94 lies below every untrained draw's most
  reflection-odd top-10 component: −0.14, +0.31, −0.17).
  - PC2 reads ARCHITECTURAL-LEVEL by the rule as written, because the rule tests only
    reflection-oddness and PC2 is reflection-even. Its rotation sensitivity is plainly learned:
    untrained r_rot = +1.00.
  - Noted as a limit of the qualifier's wording, not re-stated.
- **Descriptive:**
  - M's PC3–PC10 read weakly rotation-sensitive to orientation-like. None reads HANDEDNESS.
  - The most reflection-odd direction within the top 10 is ORIENTATION-LIKE (r_rot −0.97).
  - The full-384-d odd direction, fitted on ~500 spirals, overfits (held-out r_mir +0.82) and is
    not read.
- **Whole-embedding odd share:** M 0.41 (spirals) and 0.38 (smooth); untrained 0.015 and
  0.06–0.07.
  - **Two fifths of M's variance changes sign under a mirror, against about 2–7% for random
    weights, and no more so in spirals than in smooth galaxies.**

**What the transforms say (read off the table; not a pre-registered state).**
- rot90 carries PC1 → PC2 (r +0.91) and PC2 → −PC1 (r −0.96).
- The left-right mirror flips PC1 and keeps PC2.
- rot180 flips both.
- **(PC1, PC2) transform as the x and y components of a polar vector in the image plane.** It is a
  360°-periodic direction, which is why W2's axial (180°) position angle and its magnitude-only
  candidates could not see it.
- It is the same in smooth galaxies as in spirals, so it is not morphology.
- It fits the D10 divergence: with no rotation or flip augmentation, a frame-fixed direction is
  free to be learned. **Cross-reference D10**, though as ORIENTATION-LIKE, not handedness.

**X1a — magnitude against spirality: NO ASSOCIATION on all three.**
- Every |ρ| ≤ 0.08; the largest is signed PC1 × featured, +0.075.
- Several are BY-significant at n ≈ 74,829, but none reaches the 0.1 hit threshold.
- Bimodality among spirals: PC1 NEITHER, PC2 UNIMODAL CENTRED, radius NEITHER.
- Untrained |PC| × spiral: +0.01 to +0.05.

**X1b — GZ1: NO ASSOCIATION.** 73,750 of 74,829 matched; 3,509 confident spirals in A, 3,093 in B.
- On B: PC1 ρ +0.017 (p 0.35), PC2 +0.031 (p 0.084), and the plane direction fitted on A at 25°
  gives +0.031 (p 0.084).
- Mean h on B is −0.041: GZ1's anticlockwise excess.
- Against χ: PC1 −0.029, PC2 −0.011, plane −0.034, none significant.

**Parity through the classifications: MIRROR CONTRADICTED, by the pre-registered state.**
- ρ(χ, h) = **−0.644** (n 3,093, p 1e-4).
- The magnitude says χ is a strong algorithmic handedness label. The sign is the opposite of the
  prediction.
- **Diagnosis (post hoc).** The chain had three links:
  1. **The array-level mirror.** Direct evidence: the dihedral match of JPEG to stamp, 7 of 7,
     plus det(CD) < 0 in 10 of 10 frames.
  2. **χ's sign on synthetic spirals.** Planted, AUC 1.00.
  3. **"GZ1 clockwise = Z-wise traced outward".** A convention I stated from memory and flagged
     as the weak link.
- Link 3 is the one that fails. Empirically, a galaxy GZ1 calls clockwise has arms that turn
  **clockwise outward in our arrays**, hence **anticlockwise outward (S-wise) in the JPEG**. GZ1's
  "clockwise" reads as the rotation sense of trailing arms, not the outward winding of the arm.
- **For Brief Y:** PyArcFiRe run on our stamps reads our array frame. Its spin sign must be
  compared against **−h**, the mirror, **and** this convention, not either alone. χ (ρ = −0.64
  against GZ1) is a ready-made cross-check.

**Exploratory, post hoc (named after seeing X1c; nothing here is confirmatory).** What vector?
Measured on the X1c sample (2,000), 5-fold R², rank-normal.

| candidate 2-vector | R² PC1 | R² PC2 | PC1 ~ x | PC2 ~ y | record |
|---|---|---|---|---|---|
| neighbours' flux direction | 0.00 | 0.00 | +0.02 | +0.02 | `out/x1_vector_posthoc.json` |
| padding direction | 0.00 | 0.00 | 0.00 | −0.01 | same |
| sky-plane gradient | 0.00 | 0.00 | +0.03 | −0.04 | same |
| centroid in the Petrosian aperture | 0.11 | 0.10 | −0.39 | +0.35 | same |
| light-weighted lopsidedness (≤ 2 R_p) | 0.12 | 0.10 | −0.40 | +0.35 | same |
| core centroid, r ≤ 5 px about 127.5 | 0.37 | 0.28 | −0.64 | +0.54 | `out/x1_vector_posthoc_fine.json` |
| core centroid, r ≤ 3 px about 127.5 | **0.40** | **0.30** | **−0.67** | **+0.55** | same |
| core − envelope offset (translation-invariant) | 0.00 | 0.00 | −0.09 | +0.05 | `out/x1_vector_posthoc_intrinsic.json` |

- The pairing is exactly what the transforms predict: PC1 with x, PC2 with y, cross terms ≈ 0. It
  sharpens as the aperture shrinks onto the core.
- **But interventions refute each position reading I tried.** On 500 stamps:
  - integer `torch.roll` by 1 or 16 px in x or y leaves PC1 and PC2 unchanged (r = 1.00), although
    the embedding moves 4–5% (`out/x1_shift_posthoc.json`). So it is not the core's position on
    the 16-px patch grid;
  - Fourier sub-pixel shifts of 0.25 and 0.5 px leave them unchanged too (r = 1.00;
    `out/x1_subpixel_posthoc.json`). So it is not sub-pixel phase.
- Frame-level grouping (camcol, run, run × camcol; group means on A, R² on B) explains nothing:
  R² ≤ 0.003 (`out/x1_frame_posthoc.json`). So it is not a per-frame PSF direction.
- **Standing:** (PC1, PC2) is a learned, translation-invariant, per-galaxy direction in the image
  frame. It is equal in smooth and spiral galaxies. It correlates with the core's centroid offset
  in the stamps as cut, but that offset is not its cause. **Still unexplained, now with its
  symmetry known.**
- What would resolve it:
  - the per-galaxy PSF asymmetry vector at the object's position (psField), not frame means;
  - an attribution map (gradient of PC1 w.r.t. pixels) on a handful of galaxies;
  - a D10-augmented retrain (B4), which should remove it.
