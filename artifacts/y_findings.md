# Brief Y (revised) — findings

The PyArcFiRe port is parked. Public machine-measured pitch angles are used instead. Driver:
`artifacts/y_pitch.py`.

## Y1 — acquire and verify (`out/y1_manifest.json`)

| file | MD5 | rows | columns | claim |
|---|---|---|---|---|
| Hayes `SF5-CS.5+axisRatio.5.tsv` (Dropbox link on ics.uci.edu/~wayne/research/students/) | `2f8ccae37d32b5c75faea798c68ad360` | 94,322 | 195 | ✓ MD5, rows and columns as claimed; `fit_state` = OK on all 94,322; P_CS ≥ 0.501 and diskAxisRatio ≥ 0.500, consistent with the stated selection |
| Galaxy PAnDa v1.0.1 `galaxies_all_v1.0.1.csv` (Zenodo 10.5281/zenodo.19704211, CC BY 4.0) | `d7a66794d1432d482a464bd9779e0333` | 10,373 | 14 | ✓ MD5 |
| Shamir SpArcFiRe chirality, non-mirrored (people.cs.ksu.edu/~lshamir/data/sparcfire/) | `2ba16cd61ae05f260579012a545f8e3f` | 666,415 | 5 | McAdam & Shamir 2023 |
| Shamir SpArcFiRe chirality, mirrored (same page) | `c11c101709f3d965290c1560f071150b` | 666,415 | 5 | same |
| GZ1 table 2 (X1b) | `766cb56d64f936e00d26699a55f0669b` | 667,944 | 16 | — |

**Joins.** All IDs are read as strings.
- **Hayes.** `name` is the **DR8+ objID** (`1237…`), not DR7, so it joins on our `object_id`. On
  `dr7objid` it matches 0; on `object_id`, **37,381** of our 230,358.
- **PAnDa** has no objIDs, so it is cross-matched within 3″ (median separation 0.47″).
  - **Hart et al. 2017:** 3,028 rows, Method `sparcfire`, r band, SDSS. That is the
    **machine-measured half** (~3,000), not the 6,222-row vote-calibrated ψ_GZ2. No duplicates.
    **2,941** match.
  - **Yu & Ho 2020:** 2,438 rows, 2DFFT, R band, SDSS. NGC1423 and PGC016368 are each listed
    twice with conflicting values (14.6° vs 19.5°; 14.5° vs 27.2°), so both galaxies are dropped,
    leaving 2,434. **1,194** match.
- **Shamir** matches 227,052 on `dr7objid`, 187,152 of them with a chirality label.

| catalogue | in 230k | embedded union | A (train) | B (test) | train, not embedded | val (untouched) |
|---|---|---|---|---|---|---|
| Hayes | 37,381 | 12,092 | 6,377 | 5,715 | 19,666 | 5,623 |
| Hart (machine) | 2,941 | 971 | 507 | 464 | 1,549 | 421 |
| Yu & Ho 2020 | 1,194 | 367 | 203 | 164 | 673 | 154 |

- Pairwise overlaps in the 230k: Hayes ∩ Hart 2,090; Hayes ∩ Yu & Ho 376; Hart ∩ Yu & Ho 34.
- **Citation (CC BY 4.0):** *Galaxy PAnDa: Galaxy Pitch Angle Database*, v1.0.1, Zenodo,
  doi:10.5281/zenodo.19704211.

## Y2 — the Hayes table's provenance, measured

### Pre-registration (written 2026-09-24, after the D28 planted checks; before any Y2 comparison)

**Primary estimator: `pa_alenWtd_avg__abs`** (arc-length-weighted, unsigned).
- `pa_avg__abs`, `pa_alenWtd_median`, |`pa_longest`| and |`pa_alenWtd_avg_domChiralityOnly`| are
  reported as **sensitivity only**, as Spearman against each reference. None is chosen by
  agreement.

**Pitch comparisons** (on the whole 230k overlap; no embedding is involved):
- **same algorithm:** Hayes against Hart's machine values (~2,090);
- **different algorithm:** Hayes against Yu & Ho 2020 (~376).
- **Statistics:**
  - Spearman, 10,000-permutation, two-sided, add-one;
  - the **partial Spearman controlling V1's visibility index** (PC1 of rank-z magnitude, SNR,
    size and redshift, recomputed on each comparison's rows), Freedman–Lane;
  - 2,000-bootstrap CI;
  - median |Δ| and the median signed offset, in degrees.
- **States** (first match):

  | # | condition | state |
  |---|---|---|
  | 1 | n < 100 | INSUFFICIENT |
  | 2 | not BY-significant, or ρ ≤ 0 | BROKEN |
  | 3 | ρ ≥ 0.5 and partial ρ ≥ 0.5 | CONSISTENT |
  | 4 | otherwise | DIVERGENT |

  - The 0.5 bar is a declared choice, not a literature constant.
- Hart vs Yu & Ho (34 in the overlap) is reference-only.
- **Localisation (descriptive).** Hart applied an SVM arc-reliability filter this table lacks. So
  |Hayes − Hart| is reported against the table's own `totalNumArcs`, `alenAt50pct` and
  `top2_chirality_agreement`.

**Handedness comparisons.** Labels are coded +1 S-wise and −1 Z-wise; EQ and missing are
dropped.
- Label against label, as the agreement fraction with a permutation p against 0.5:
  - Hayes vs Shamir non-mirrored;
  - Hayes vs Shamir mirrored;
  - Shamir non-mirrored vs mirrored (SpArcFiRe's own reflection consistency);
  - Hayes vs GZ1. GZ1 is converted through X1's empirical mapping: CW ↔ clockwise-outward in our
    arrays, coded −sign(h); NVOTE ≥ 10 and h ≠ 0.
- χ against a label, as the AUC of χ for S vs Z with a permutation p: Hayes, Shamir non-mirrored,
  Shamir mirrored. χ is computed on 3,000 random Hayes-labelled galaxies' stamps (seed 0).
- **States (up to a parity flip).** d = |agreement − 0.5| or |AUC − 0.5|:

  | # | condition | state |
  |---|---|---|
  | 1 | n < 100 | INSUFFICIENT |
  | 2 | not BY-significant, or d < 0.1 | BROKEN |
  | 3 | d ≥ 0.3 | CONSISTENT |
  | 4 | otherwise | DIVERGENT |

  - The sign is reported, not graded.
- **BY family:** the 2 pitch + 4 label + 3 χ tests = 9. 10,000 draws resolve rank 1 at m = 9.
- **Parity of the Hayes run.** Shamir's non-mirrored run used the SDSS JPEG cutouts, and our
  arrays mirror those (X1).
  - JPEG PARITY: Hayes agrees more with the non-mirrored run, and one of the two comparisons is
    CONSISTENT.
  - ARRAY PARITY: Hayes agrees more with the mirrored run, likewise.
  - UNDETERMINED: neither comparison is CONSISTENT.
  - **Triangle check:** χ's sign against Hayes must equal χ's sign against the Shamir run Hayes
    agrees with. CLOSES or OPEN.
- **Selection (descriptive).**
  - Matched Hayes galaxies against all our spirals (t01 features ≥ 0.5, t02 not edge-on ≥ 0.5,
    t04 spiral ≥ 0.5).
  - Compared on visibility index, redshift, Petrosian size, `expAB_r` and t04 spiral fraction:
    medians and Cliff's δ.
  - Hayes is selected on **P_CS > 0.5**, volunteer spirality, which Peng et al. showed biases
    pitch with redshift because tight arms fade with distance. It is also selected on **disc axis
    ratio > 0.5** (face-on to moderately inclined). Both are stated as limits whatever the numbers
    say.

**D28 planted checks** (`out/y2_planted.json`), through the identical compare/state code and BY at
m = 9:

| plant | statistic | state | |
|---|---|---|---|
| Hart + 0.3 sd noise | ρ 0.95, partial 0.95 | CONSISTENT | ✓ |
| Hart + 2.5 sd noise | ρ 0.30 | DIVERGENT | ✓ |
| Hart, permuted | ρ −0.02, p 0.32 | BROKEN | ✓ |
| labels, 10% flipped | agree 0.90 | CONSISTENT | ✓ |
| labels, 30% flipped | agree 0.69 | DIVERGENT | ✓ |
| labels, random | agree 0.50, p 0.66 | BROKEN | ✓ |
| χ, gap 3 sd | AUC 1.00 | CONSISTENT | ✓ |
| χ, gap 0.5 sd | AUC 0.77 | DIVERGENT | ✓ |
| χ, no gap | AUC 0.50, p 0.85 | BROKEN | ✓ |

- Extreme effects reach p = 1e-4, which clears BY at m = 9, so the null does not saturate.

*Y2 pre-registration ends here: the Y2 section above (from "## Y2", 125 lines of the file so far), SHA-1 `c603433247c14356d279b54ef1d9d9d23e0970a5`.*

### Y2 — result (`out/y2_provenance.json`)

**Pitch: DIVERGENT with the same algorithm, BROKEN against a different one.**

| comparison | n | ρ [95% CI] | partial (visibility) | median \|Δ\| | median offset | state |
|---|---|---|---|---|---|---|
| Hayes vs Hart (SpArcFiRe, SVM-filtered) | 2,090 | **+0.38** [0.34, 0.42] | +0.38 | 4.0° | +0.0° | **DIVERGENT** |
| Hayes vs Yu & Ho 2020 (2DFFT) | 376 | **−0.03** [−0.13, +0.08] (p 0.63) | −0.02 | 7.2° | −4.5° | **BROKEN** |
| Hart vs Yu & Ho (reference) | 34 | −0.25 (p 0.15) | — | 7.1° | — | reference only; 34 is also the whole PAnDa overlap |

- **Sensitivity** (Spearman against Hart; not chosen among):
  - `pa_avg__abs` +0.13;
  - `pa_alenWtd_median` +0.34;
  - |`pa_longest`| +0.42;
  - |`pa_alenWtd_avg_domChiralityOnly`| +0.43.
  - Against Yu & Ho every estimator lies between −0.04 and +0.01, so the break does not depend on
    the estimator.
- **Visibility changes nothing:** each partial equals its raw ρ.
- **Localisation (descriptive):** disagreement with Hart grows where the table's two longest arcs
  disagree on chirality (median |Δ| 5.2°, against 3.8° where they agree, and 2.7° with one long
  arc). It shrinks as the median arc length rises (ρ −0.18), and grows with the arc count
  (+0.13). That is what an SVM arc filter would remove.
- **Reading.**
  - The Hayes table carries a real, weak, same-algorithm signal. It is SpArcFiRe agreeing with
    SpArcFiRe, with no calibration offset.
  - **No second algorithm confirms it.** There is no evidence here that the measured quantity is
    pitch angle rather than a SpArcFiRe-specific arc statistic.
  - The strongest evidence the brief asked for, two independent methods agreeing, is **absent**.
    The same-algorithm ρ of 0.38 caps any correlation with this table at roughly that
    reliability.

**Handedness.** Up to a parity flip; sign reported. Coding: +1 S-wise.

| comparison | n | agreement / AUC | state |
|---|---|---|---|
| Shamir non-mirrored vs mirrored | 154,582 | agree 0.073: **92.7% flip** | CONSISTENT (as a mirror should be; 7.3% of labels do not flip) |
| Hayes vs Shamir non-mirrored | 32,120 | agree 0.244 | DIVERGENT |
| Hayes vs Shamir mirrored | 32,118 | agree **0.756** | DIVERGENT (d 0.256, below 0.3) |
| Hayes vs GZ1 (via X1's mapping) | 36,278 | agree 0.192 | CONSISTENT (d 0.31) |
| χ vs Hayes | 3,000 | AUC 0.254 | DIVERGENT (d 0.246) |
| χ vs Shamir non-mirrored | 2,581 | AUC 0.751 | DIVERGENT |
| χ vs Shamir mirrored | 2,579 | AUC 0.250 | DIVERGENT |

- **Parity: UNDETERMINED, by the pre-registered rule.** Neither Hayes–Shamir comparison clears the
  0.3 bar; 0.756 is 0.044 short.
  - The direction is not in doubt: Hayes agrees with Shamir's **mirrored** run, 76 against 24.
  - **The triangle closes:** χ against Hayes (0.254) has the sign of χ against the mirrored run
    (0.250).
  - So the Hayes run most likely used **array (FITS-frame) parity**, mirroring the JPEGs as our
    stamps do. It is stated as the direction the evidence points, not as the verdict.
- **What caps the label agreement is SpArcFiRe's own label noise.**
  - The algorithm's labels do not flip under an exact mirror 7.3% of the time.
  - χ agrees with each SpArcFiRe run at AUC ≈ 0.75, against 0.644 (|ρ|) with GZ1 votes in X1b.
- **S/Z naming (for any later use).**
  - In Shamir's non-mirrored (JPEG) run, S-wise goes with χ *high*. χ high means
    anticlockwise-outward in our row-down arrays, i.e. clockwise-outward in the row-down JPEG.
  - So SpArcFiRe's "S-wise" is named in a vertically flipped (row-up) frame relative to a
    row-down display.
  - With that naming, Hayes vs GZ1 (0.19) is consistent with array parity. Recorded so nobody
    re-derives it from the letter shapes.

**Selection** (37,381 Hayes matches against 42,564 of our spirals; 22,172 in both; Cliff's δ,
Hayes minus spirals):

| axis | Hayes median | spirals median | δ |
|---|---|---|---|
| visibility index (+ = fainter) | +0.42 | −0.25 | **+0.17** (Hayes less visible) |
| Petrosian size | 8.1″ | 9.3″ | **−0.22** (smaller) |
| redshift | 0.070 | 0.065 | +0.07 |
| `expAB_r` | 0.665 | 0.665 | +0.03 |
| t04 spiral fraction | 0.77 | 0.91 | **−0.32** |

- **Limits, whatever the numbers:**
  - P_CS > 0.5 selects on volunteer spirality. That is Peng et al.'s bias: tight arms fade with
    distance.
  - The disc axis ratio > 0.5 cut keeps face-on to moderately inclined discs only.
- The matched sample is **fainter, smaller and less confidently spiral** than our spirals.
  - Arc detection degrades exactly there, and the P_CS selection compounds it.
  - Every Y3 comparison controls visibility for this reason.

## Y3 — the science

### Pre-registration (written 2026-09-24, after Y2's result and the D28 planted checks; before any Y3 statistic on real targets)

**Standing from Y2, which bounds everything below.**
- The primary pitch, Hayes `pa_alenWtd_avg__abs`, agrees with Hart's filtered SpArcFiRe at only
  ρ 0.38. No second algorithm confirms it (Yu & Ho: −0.03).
- Any association with it is capped near that reliability. Y3 reads it as "a SpArcFiRe arc
  statistic, independent of the votes", **never** as ground truth.
- The Yu & Ho replication is the one cross-algorithm check.
- Visibility (V1's composite, recomputed on each test's rows) is controlled in every comparison.

**Samples.**
- **Hayes:** 37,381 in the 230k; the embedding tests use A (fit) → B (test) within P2's union.
- **Replications:** Hart (machine) and Yu & Ho 2020. Their galaxies in P2's train partition
  outside the union were embedded with the identical path (2,209; M + untrained seeds 0–2 via
  `untrained_encoder_matrix`), z-scored with the union's statistics. **Val is untouched.**
- **Winding:** GZ2 t10, reach ≥ 21 (U3's floor).
  - w_avg = 0.5·f_medium + 1.0·f_tight (Masters et al. 2019).
  - Plurality category tight / medium / loose, with ties dropped.

**T1 — agreement of w_avg with measured pitch** (every matched row; no embeddings).
- Spearman, 10,000-permutation, plus the visibility partial and a CI.
- Expected **negative**: tighter votes go with smaller pitch.
- BY over Hayes, Hart and Yu & Ho (m = 3).
- **States:** INSUFFICIENT (n < 100) · NO AGREEMENT (n.s.) · CONTRARY (ρ > 0) · AGREE
  (ρ ≤ −0.3 and partial ≤ −0.3) · WEAK AGREEMENT (otherwise).

**T2 — the 2×2** (Hayes, A → B). Ridge (RidgeCV, V2's alphas) decodes pitch and w_avg on M and each
untrained draw.
- **A_m** = partial(pred_pitch, pitch | **pred_w_avg**, visibility).
- **A_v** = partial(pred_w_avg, w_avg | **pred_pitch**, visibility).
- **Deviation from V2, forced by D28.** V2 controlled the *noisy other target*. On a planted pair
  of noisy copies of one quantity, that form reads A_m 0.63, A_v 0.65, i.e. BOTH. Here each leg
  controls the other **decoder's prediction**, and must also beat a **matched shared-quantity
  null**:
  - the latent is M's own pitch decoder direction;
  - each copy's noise is tuned to reproduce the observed decode ρ;
  - 50 draws; the leg must exceed the null's 95th percentile.
- **Leg true iff:** BY-significant (m = 4: two decodes, two partials), ρ ≥ 0.10, above all three
  untrained draws, and above the shared null's 95th percentile. INVERTED if significant and
  negative.
- **States (V2's names):** BOTH · MEASUREMENT BEYOND VOTES · VOTES BEYOND MEASUREMENT · SHARED ONLY
  (both decoded, neither leg) · NEITHER · INVERTED.
- Margins over untrained are stated for every decode. V2's form is reported beside each partial,
  labelled descriptive.
- **Replication** (Hart, Yu & Ho; rows not in the Hayes fit):
  - (i) the Hayes-fitted M pitch decoder transferred, as a visibility partial against the
    reference;
  - (ii) a 5-fold ridge within the reference.
  - BY over 4. States: REPLICATED (sig, ρ > 0, above every untrained draw) · REPLICATED, NOT ABOVE
    UNTRAINED · INVERTED · NOT REPLICATED.

**T3 — U3-C: does "medium" partly mean "couldn't tell"?** (every matched row).
- Pitch is residualised on visibility (rank-linear).
- D = MAD(medium) − max(MAD(tight), MAD(loose)).
- Null: categories permuted **within visibility quintiles**, 10,000 draws; p two-sided (twice the
  smaller tail). BY m = 3.
- **States:** INSUFFICIENT (a group < 100) · NOT WIDER (n.s.) · WIDER (D > 0) · NARROWER (D < 0).
- Group medians are reported: an ordered winding should run tight < medium < loose in pitch.

**T4 — winding's ordering against a visibility gradient.**
- **(a)** Kendall τ of category (tight 0 … loose 2) against pitch: raw, and the mean within
  visibility quintiles (stratified permutation, 2,000 draws).
  - States: INSUFFICIENT · NOT ORDERED · REVERSED (τ_within < 0) · ORDERED BEYOND VISIBILITY
    (τ_within ≥ ½ τ_raw) · ORDERED, MOSTLY VISIBILITY.
- **(b)** U3's winding axis (logistic on A's tight vs loose, + = tight, per encoder) projected on
  B's Hayes galaxies, against pitch. Existence from the raw Spearman (10,000). Retention =
  partial(visibility) / raw.
  - States: INSUFFICIENT · NONE · REVERSED UNDER CONTROL · MOSTLY VISIBILITY (retention < 0.5) ·
    NOT ABOVE UNTRAINED (|partial| ≤ an untrained draw's) · WINDING BEYOND VISIBILITY
    (|partial| ≥ 0.1) · BEYOND VISIBILITY, NEGLIGIBLE.
  - The expected sign is negative (tight-positive axis ↔ smaller pitch). It is read by magnitude,
    with the sign stated.
- One BY family over (a) × 3 and (b) (m = 4; 2,000 draws resolve rank 1).

**T5 — exploratory, labelled; no states.**
- Hayes S/Z handedness: logistic (C = 0.1) on A → AUC on B, for M and each untrained draw.
  PC1/PC2 are not retested (X1).
- `numDcoArcsGE000`, `GE040` and `GE100` against visibility, and against the GZ2 mean arm count,
  raw and visibility-partial.

**D28 planted checks** (`out/y3_planted.json`), through the identical functions and BY at each real
family size:

| test | plant | read | state | |
|---|---|---|---|---|
| T1 | −w_avg + 0.3 sd noise | ρ −0.95 | AGREE | ✓ |
| T1 | permuted | ρ 0.00 | NO AGREEMENT | ✓ |
| T1 | +w_avg + 1 sd noise | ρ +0.70 | CONTRARY | ✓ |
| T2 | two independent directions of M's embedding + noise | A_m 0.83, A_v 0.83 (null 95%: 0.13, 0.15) | BOTH | ✓ |
| T2 | **one direction, two noisy copies** | A_m 0.10, A_v 0.10 (null 95%: 0.13, 0.14); V2's form 0.63, 0.65 | **SHARED ONLY** | ✓ (V2's form would read BOTH) |
| T2 | noise | 0.02, 0.00 | NEITHER | ✓ |
| T3 | medium drawn from the tight/loose mixture | D +0.18, p 0.005 | WIDER | ✓ |
| T3 | medium shrunk to the median | D −2.8 | NARROWER | ✓ |
| T3 | permuted | p 0.40 | NOT WIDER | ✓ |
| T4b | the axis's own score + noise | retention 0.92 | WINDING BEYOND VISIBILITY | ✓ |
| T4b | score and target both visibility | retention 0.04 | MOSTLY VISIBILITY | ✓ |
| T4b | noise | — | NONE | ✓ |

- Extreme effects reach p = 1e-4 against each BY bar, so the null does not saturate.

*Y3 pre-registration ends here: the Y3 section above (101 lines from "## Y3"), SHA-1 `b7bd772688d5c0a2aec837586cd111345a5b086a`.*

### Y3 — result (`out/y3_pitch.json`; planted checks in `out/y3_planted.json`)

**T1 — w_avg against measured pitch: SpArcFiRe agrees with the votes; the Fourier method does not.**

| sample | n | ρ [95% CI] | visibility partial | state |
|---|---|---|---|---|
| Hayes | 16,274 | **−0.35** [−0.36, −0.33] | −0.31 | **AGREE** |
| Hart (machine) | 2,762 | **−0.32** [−0.35, −0.28] | −0.32 | **AGREE** |
| Yu & Ho 2020 (2DFFT) | 550 | −0.03 [−0.12, +0.06] (p 0.47) | −0.04 | **NO AGREEMENT** |

- The sign is as expected: tighter votes go with smaller pitch, and visibility barely moves it.
- The two SpArcFiRe tables agree with the votes about as well as they agree with each other (Y2,
  0.38). Yu & Ho agrees with neither the votes nor SpArcFiRe.
- This design cannot say which side is right. The votes and SpArcFiRe may share an image-level
  response to arc visibility, or 2DFFT may be noisy on these SDSS galaxies.
- **"Independent of the votes" holds only within one algorithm family.**

**T2 — the 2×2 (Hayes, 2,721 train → 2,491 test): VOTES BEYOND MEASUREMENT.**

| | M | untrained (3) | margin | shared null 95% | V2's form |
|---|---|---|---|---|---|
| decode pitch | ρ 0.229 | 0.17 / 0.20 / 0.21 | **+0.02 … +0.06** | — | — |
| decode w_avg | ρ 0.372 | 0.23 / 0.25 / 0.27 | **+0.11 … +0.14** | — | — |
| A_m: pitch beyond votes | **0.097** | 0.03 / 0.03 / 0.07 | — | 0.070 | 0.100 |
| A_v: votes beyond pitch | **0.273** | 0.08 / 0.10 / 0.12 | — | 0.159 | 0.268 |

- **A_v clears every bar.** The encoder carries vote-winding content that measured pitch does not.
- **A_m clears the untrained draws and the shared null, but sits 0.003 under the 0.10
  negligibility bar**, so its leg is false.
  - A measurement with same-algorithm reliability 0.38 can barely show content beyond anything.
  - A_m is small, not absent.
- **Most of what the encoder reads from measured pitch is architectural:** the learned margin is
  only +0.02 to +0.06. The learned part of winding is the vote side (+0.11 to +0.14).
- **Replication.**
  - Hart, Hayes decoder transferred: partial +0.03 (p 0.11), **NOT REPLICATED**.
  - Hart, 5-fold within: +0.07 (p 0.002), **REPLICATED**, but tiny.
  - Yu & Ho: +0.02 transferred and −0.08 within, **NOT REPLICATED** both ways.

**T3 — U3-C, is "medium" wider?: NARROWER in both SpArcFiRe samples. The "couldn't tell"
reading is not supported.**

| sample | n (tight / medium / loose) | median pitch (°) | MAD, visibility-residualised (°) | D | state |
|---|---|---|---|---|---|
| Hayes | 5,636 / 8,001 / 2,110 | 16.4 < 19.1 < 21.0 | 3.09 / **3.06** / 3.60 | −0.54 | **NARROWER** |
| Hart | 1,020 / 1,367 / 284 | 16.0 < 18.8 < 23.5 | 3.76 / **4.37** / 5.81 | −1.44 | **NARROWER** |
| Yu & Ho | 337 / 178 / 26 | — | — | — | INSUFFICIENT |

- Medium sits between tight and loose in median pitch, and its spread is no wider than the widest
  end. In Hayes it is as tight as "tight".
- By pitch, medium looks like an intermediate winding, not a mixture of misjudged tight and loose.
- The pre-registered state names the comparison with the wider end. Medium is not narrower than
  both: in Hart it is wider than tight.

**T4 — winding against a visibility gradient.**

| test | read | state |
|---|---|---|
| (a) votes' category vs pitch, Hayes | τ 0.25 raw → **0.22** within visibility quintiles | **ORDERED BEYOND VISIBILITY** |
| (a) Hart | τ 0.23 → **0.23** | **ORDERED BEYOND VISIBILITY** |
| (a) Yu & Ho | τ 0.00 (p 0.87) | NOT ORDERED |
| (b) U3's winding axis vs Hayes pitch (B, 2,491) | raw **−0.18**, partial −0.08 (untrained −0.02 … −0.05), retention **0.42** | **MOSTLY VISIBILITY** |

- **The votes' winding ordering is not a visibility gradient** by the SpArcFiRe measurement: it
  survives stratification almost whole. This does for winding what V2 did for roundness and
  bulge, within one algorithm family.
- **The encoder's winding axis is weaker.**
  - Its alignment with measured pitch has the right sign and exceeds untrained.
  - But 58% of it goes with visibility: fainter galaxies measure looser (pitch vs visibility
    ρ +0.24) and sit off the tight end.
  - Stated plainly: the encoder axis tracks pitch partly, and mostly through visibility.
- The ridge-decoded pitch keeps a visibility partial of 0.11.

**T5 — exploratory.**
- **Handedness is not linearly decodable from the embedding.**
  - Hayes S/Z, logistic A → B: M AUC 0.498; untrained 0.49–0.52 (n 2,721 / 2,491).
  - With X1, the encoder carries image-frame pose strongly and chirality not at all. A mirror
    changes 41% of the variance without encoding which way the arms wind.
  - Caveat: the label is noisy. SpArcFiRe's own mirror consistency is 92.7%; its agreement with χ
    is AUC ≈ 0.75.
- **Machine arc counts track the voted arm count beyond visibility.**

  | arc count | vs GZ2 mean arm count | visibility partial | vs visibility |
  |---|---|---|---|
  | `numDcoArcsGE000` | +0.23 | +0.23 | −0.03 |
  | `numDcoArcsGE040` | +0.31 | +0.29 | −0.12 |
  | `numDcoArcsGE100` | +0.27 | +0.24 | −0.20 |

  - The voted arm count falls with visibility (ρ −0.24).
  - But its association with machine-counted arcs survives visibility nearly intact.
  - So V1's "arm count is a visibility gradient" is not the whole story: there is arc-count
    content beyond it. Exploratory; not a verdict change.

### What Y3 changes in the record

- **Winding's physical grounding, A19: from PENDING to SUPPORTED WITH CAVEAT, within one algorithm
  family.**
  - The votes' ordering tracks SpArcFiRe pitch beyond visibility (T4a), and w_avg agrees with it
    (T1).
  - Neither holds against 2DFFT (Yu & Ho). That is the one cross-algorithm check, and it confirms
    nothing, including SpArcFiRe itself (Y2).
- **The encoder's winding is more vote-like than measurement-like.**
  - The learned margin sits on w_avg, not on pitch. VOTES BEYOND MEASUREMENT.
  - Its winding axis meets measured pitch mostly through visibility.
- **V2's Experiment D needs re-running.**
  - V2 read B/T as BOTH (A_m 0.36, A_v 0.47) by controlling each partial on the other *noisy
    target*.
  - Y3's planted check shows that form reads **BOTH on two noisy copies of one quantity** (0.63,
    0.65).
  - V2's B/T verdict is therefore unverified, not refuted. It should be re-run with Y3's
    cross-decoder controls and matched shared null. It is flagged in TODO, not re-run here.
