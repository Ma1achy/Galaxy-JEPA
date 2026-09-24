# Brief V — findings

U's loose ends (V1), measurements independent of the votes (V2), and the encoder's own directions
(V3). Each stage is pre-registered and hashed before it runs. From this brief on, every
pre-registration enumerates **all** its outcomes, including sign flips, insufficient samples, and
significance disagreeing with magnitude (V1.4, D27).

## V1 — U's loose ends

### Pre-registration (written 2026-09-24, before any V1 number was computed)

V1.1, V1.2 and V1.4 restate or record results that already exist. Only V1.3 and V1.5 compute
anything new.

**V1.1 — Fig 2 as a margin over untrained.**
- Every U2 number is restated as **trained − untrained**. The margin interval runs from
  trained − (max over the three untrained draws) to trained − (min over them).
- It is the same framing existence uses against the untrained bar. The draws are a range (K = 3),
  not a spread estimate.

**V1.2 — effect size beside every off-axis verdict.**
- Every off-axis verdict carries its **retention**: partial / raw, both on the visibility-complete
  rows. It is the share of the raw ambiguity association left once visibility is controlled.
- The magnitude reading is fixed now:
  - **mostly visibility**: retention < 0.5;
  - **mostly beyond visibility**: retention ≥ 0.5;
  - **reversed**: retention < 0, whatever the significance says.
- Where the verdict and the reading disagree (spiral: BEYOND VISIBILITY at retention 0.27), the
  sentence follows the magnitude and names the disagreement.

**V1.3 — suppression check, merger and bulge "obvious". Exploratory.** The U2 rule had no reversal
state, and this check is added after seeing the result, so its verdict is post hoc.
- **Rows:** exactly U2's headline rows for `full:t08_odd_feature_a24_merger:21` and
  `full:t05_bulge_prominence_a12_obvious:21`. Test galaxies with all four visibility covariates
  valid; the same bend coordinate; the same ambiguity.
- **Composite visibility index:**
  1. Each of `modelMag_r`, `snr_r`, `petroRad_r` and `specz` is rank-transformed over those rows
     (as U2's partial does) and z-scored.
  2. PC1 of their 4 × 4 correlation matrix, i.e. the Spearman matrix. Sign fixed so that + is
     fainter (positive loading on `modelMag_r`).
  3. Reported: loadings, PC1's share of the variance, and the four covariates' pairwise Spearman
     correlations.
- **Statistic:** partial Spearman of bend against ambiguity, controlling the PC1 score alone.
  Freedman–Lane, 10,000 draws, two-sided, add-one. BY across the two answers (m = 2; rank-1
  threshold 0.017, so 1e-4 resolves).
- **Descriptive:** Spearman of bend with each covariate and of ambiguity with each covariate. This
  is the classical suppression pattern. The same partial is computed on the three untrained draws
  (range, no permutation).
- **Outcomes**, with ρ_raw the raw association on the same rows (positive for both answers) and
  ρ_c the composite partial. The first matching row applies:

  | # | condition | verdict |
  |---|---|---|
  | 1 | ρ_c < 0 and BY-significant | **REVERSAL SURVIVES** the composite. The flip is not a joint-fit artefact. Tagged *small* if \|ρ_c\| ≤ ½ ρ_raw. |
  | 2 | ρ_c < 0, not significant, \|ρ_c\| > ½ ρ_raw | **REVERSAL, UNRESOLVED** |
  | 3 | \|ρ_c\| ≤ ½ ρ_raw (not significantly negative) | **SUPPRESSION; VISIBILITY ACCOUNTS.** The four-way flip was suppression, and a single visibility index removes most of the raw effect. |
  | 4 | ρ_c > ½ ρ_raw and BY-significant | **SUPPRESSION; ASSOCIATION SURVIVES.** The flip was suppression, and a positive association beyond one visibility index remains. |
  | 5 | ρ_c > ½ ρ_raw, not significant | **SUPPRESSION, UNRESOLVED** |

- **Qualifier on every row:** if PC1 carries < 50% of the covariates' variance, one index is not a
  fair summary of visibility. The verdict is then reported as **COMPOSITE INADEQUATE** beside its
  row.
- **Expectation:** U2's separation showed size alone reversing merger (−0.093), so I expect merger
  to be row 1. For "obvious", single-variable controls pulled opposite ways (faintness +0.186,
  redshift −0.084). I expect row 3 or 4, i.e. suppression.

**V1.4 — standing rule.**
- Recorded as D27, in `CLAUDE.md`, and in memory.
- A reversal state goes into U2's verdict function for future runs, behind a flag. The recorded U2
  verdicts are not recomputed.

**V1.5 — the escape-hatch test.** The hypothesis: uncertain volunteers take whatever exit the answer
set offers. Arm count has an explicit "can't tell"; bulge has a natural "none" endpoint; winding has
neither, so its uncertain votes pile into "medium".

This is a test of votes against photometry. No embedding enters it.

*Primary — arm count's "can't tell" (confirmatory).*
- **Population:** full, every galaxy in P2's union (train ∪ test) whose t11 reach (all answers'
  counts summed) is ≥ 21.
- **Groups:** plurality answer is "can't tell" (a37), against plurality in the ordered set
  {1, 2, 3, 4, 4+}. Exact ties are dropped.
- **Visibility index V:** U3's `visibility_index`. It is the mean rank-normal score of fainter,
  lower-SNR, smaller and higher-z; + = less visible; ranks are over the analysed galaxies. Galaxies
  missing a covariate are dropped and counted.
- **Statistic:** AUC = P(V_can't-tell > V_ordered). **Predicted > 0.5**: can't-tell galaxies are
  less visible. The p is one-sided, from a 10,000-draw label permutation, add-one.
- **Outcomes:**

  | condition | verdict |
  |---|---|
  | < 100 can't-tell galaxies at reach ≥ 21 | rerun at reach ≥ 10, labelled *fallback*. If still < 100: **INSUFFICIENT**, not tested |
  | p < 0.05 and AUC ≥ 0.556 (≈ Cohen's d 0.2) | **SUPPORTED** |
  | p < 0.05 and 0.5 < AUC < 0.556 | **SIGNIFICANT BUT NEGLIGIBLE** (magnitude and significance disagree) |
  | the reverse one-sided p < 0.05 (AUC < 0.5) | **CONTRARY**, tagged *negligible* if AUC > 0.444 |
  | otherwise | **NOT SUPPORTED** |

- **Inclination qualifier (pre-registered).** Arms are uncountable on inclined discs for geometric
  reasons, not visibility.
  - Also reported: AUC of (1 − `expAB_r`) for can't-tell against ordered.
  - The primary AUC is recomputed within `expAB_r` quintiles: a stratified AUC, weighted by
    n₁n₂ per stratum.
  - If the inclination AUC ≥ 0.556 and the stratified visibility AUC falls below 0.556 while the
    raw one was ≥ 0.556, the verdict is qualified **INCLINATION, NOT VISIBILITY**.
- **Descriptive:**
  - mean V per ordered category and for can't-tell;
  - the AUC of can't-tell against each ordered category separately;
  - galaxy-level Spearman(V, can't-tell vote fraction) over all reach ≥ 21 galaxies;
  - the same AUC with seeing (`psfWidth_r`, worse = less visible). Seeing is exogenous: the
    atmosphere, not the galaxy.

*Secondary — bulge "none" at comparable size and brightness (exploratory).*
- **Population:** full union, t05 reach ≥ 21, plurality answers only, ties dropped.
- **Groups:** plurality "no bulge" (a10) against plurality in {just noticeable, obvious,
  dominant}.
- **Strata:** quintiles of `petroRad_r` × quintiles of `modelMag_r` over the analysed galaxies,
  25 cells. Cells with fewer than 10 in either group are dropped, and the count is reported.
- **Within-stratum visibility:** size and brightness are held, so what remains is:
  - **seeing** (`psfWidth_r`; **predicted: none has worse seeing**, AUC > 0.5). Exogenous, so the
    cleaner of the two.
  - **SNR** (`snr_r`; **predicted: none has lower SNR**, i.e. AUC of −SNR > 0.5).
- **Statistic:** the stratified AUC, a weighted mean of within-cell AUCs with weights n₁n₂. The p
  is one-sided, from permuting labels within cells (10,000, add-one).
- **Outcomes:** the same five states as the primary, each tagged *exploratory*. With two
  measures, a split between seeing and SNR is reported as **SPLIT**, not averaged.

**Expectation.**
- Primary: SUPPORTED, AUC around 0.6. I also expect inclination to carry a real part of it, which
  would make the qualifier a live possibility.
- Secondary: weak. Seeing AUC 0.50–0.53 (negligible), because at fixed size and brightness seeing
  varies little across SDSS.

*V1 pre-registration ends here: first 131 lines, SHA-1 `439ecbc8313b3490699dd8e843b9c57f2a064158`.*

### V1 — result (`artifacts/out/v1_loose_ends.json`)

**V1.1 / V1.2** are restated in `u_findings.md` §U2 ("Restated (Brief V1)") and in the README.
- The learned on-axis margin is **+0.04 to +0.23** on the 27 answers.
- Spiral reads **mostly visibility** (retention 0.29 and 0.22), whatever U2's significance-driven
  BEYOND said.
- Merger and "obvious" read **reversed**.

**V1.3 — suppression check (exploratory, post hoc).**

| | merger | bulge "obvious" |
|---|---|---|
| n (visibility-complete test rows) | 3,089 | 6,465 |
| raw ρ(bend, ambiguity) | +0.063 | +0.053 |
| four-way partial (reproduces U2) | −0.089 | −0.039 |
| **composite (PC1) partial** | **−0.096**, p = 1e-4 | **+0.120**, p = 1e-4 |
| PC1 share of the covariates' variance | 0.58 | 0.69 |
| PC1 loadings (mag, snr, size, z) | +0.62, −0.51, −0.42, +0.43 | +0.57, −0.50, −0.45, +0.46 |
| ρ(bend, PC1) / ρ(ambiguity, PC1) | +0.45 / +0.32 | +0.52 / −0.09 |
| untrained composite partial | +0.20…+0.22 | −0.03…−0.01 |
| **verdict** | **REVERSAL SURVIVES** | **SUPPRESSION; ASSOCIATION SURVIVES** |

- **Merger.** The reversal is not a joint-fit artefact: one visibility index gives the same
  negative partial as the four together.
  - Bend and ambiguity both rise with faintness (+0.45, +0.32). That shared dependence is the whole
    raw positive.
  - At fixed visibility, more-ambiguous merger votes sit *less* far along M's bend.
  - Untrained encoders show the opposite, and larger (+0.20). M's learned geometry runs against the
    architecture's here.
  - Consistent with T2: merger's direction is entangled with "other", partly distance.
- **Obvious.** Classical suppression.
  - Bend rises with PC1 (+0.52); ambiguity falls slightly with it (−0.09). Controlling the single
    index therefore *raises* the association, from +0.053 to +0.120.
  - U2's four-way negative came from entering four correlated covariates (mag–snr −0.90) together.
  - Untrained is ~0, so the +0.12 is learned.
- **Against expectation:** both as predicted, merger row 1 and obvious row 3/4 (it landed on
  row 4). Post hoc, and labelled so.

**V1.4 — standing rule.**
- Recorded as D27, in `CLAUDE.md`, and in memory.
- `u2_uncertainty.REVERSAL_STATE` adds **REVERSES** (a significant partial of the opposite sign)
  for future runs. It stays `False` for the U2 record, which is not recomputed.

**V1.5 — the escape-hatch test.**

*Primary (confirmatory): arm count's "can't tell" is **CONTRARY**.*
- n = 1,819 can't-tell against 7,156 ordered, at reach ≥ 21 (301 dropped for a missing
  covariate).
- AUC(V) = **0.403**, reverse one-sided p = 1e-4. Can't-tell galaxies are **more** visible than
  the ordered ones, not less.
- **Inclination qualifier: not triggered.**
  - Can't-tell galaxies are, if anything, *rounder* (AUC of 1 − b/a = 0.476).
  - Within b/a quintiles the visibility AUC is unchanged (0.401).
- **Seeing** agrees in sign (AUC 0.456, worse seeing *less* common among can't-tell).
- **Descriptive: visibility falls monotonically with arm count.**

  | plurality | 1 | 2 | 3 | 4 | 4+ | can't tell |
  |---|---|---|---|---|---|---|
  | n | 256 | 5,753 | 772 | 233 | 142 | 1,819 |
  | mean V (+ = less visible) | +0.30 | +0.16 | −0.10 | −0.43 | −0.95 | −0.17 |
  | AUC(can't tell vs this) | 0.33 | 0.38 | 0.47 | 0.60 | 0.77 | — |

  The galaxy-level Spearman of V against the can't-tell vote fraction is −0.15.
- **Reading.** The exit uncertain volunteers take on arm count is not "can't tell". It is *fewer
  arms*: the least visible galaxies are voted 1 or 2 arms. Counting 4+ arms needs a
  well-resolved galaxy.
  - "Can't tell" sits between 2 and 3 in visibility. It is chosen on reasonably visible galaxies,
    plausibly for a different reason: flocculent or ambiguous structure, not faintness.
- **Consequence for U3-C.** Arm count's ordered axis is itself a visibility gradient: 4+ is the
  most visible endpoint. So "middles more visible than their position predicts" is measured against
  endpoints that differ in visibility, and does not need an exit.

*Secondary (exploratory): bulge "none" is **SPLIT**.*
- n = 13,463 at reach ≥ 21, 731 of them "none". 18 of 25 size × magnitude cells are usable,
  covering 707 and 10,373 galaxies.
- **Seeing:** stratified AUC 0.542, p = 0.001. SIGNIFICANT BUT NEGLIGIBLE.
- **Low SNR:** stratified AUC 0.692, p = 1e-4. SUPPORTED.
- **The two disagree, and the cleaner one is weak.** Seeing is exogenous; SNR is not. SNR
  correlates −0.90 with magnitude, and quintile cells leave residual magnitude inside each cell.
  So the SNR result is partly faintness the strata did not remove. It is also partly surface
  brightness, which is a property of the galaxy, not only of the observation.
- *Exploratory*: at comparable size and brightness, "no bulge" galaxies have lower SNR, and at most
  marginally worse seeing.

**Against my stated expectation.**
- *Wrong on the primary.* I expected SUPPORTED at ~0.6. It is CONTRARY at 0.40. The escape-hatch
  hypothesis fails for arm count as stated.
- *Wrong on inclination.* I expected it to carry part of the effect. It carries none.
- *Right on seeing* (negligible).
- *Unpredicted:* the monotone visibility gradient across arm counts, and "fewer arms" as the
  low-visibility exit.
- **Where this leaves U3-C's contrast.**
  - Winding has no exit, and "medium" is less visible: this still stands from U3.
  - For arm count, the ordered axis is itself confounded with visibility. That is now the better
    explanation of its middles' displacement.
  - Bulge is open: "none" is enriched in low-SNR galaxies at comparable size and brightness, but
    the exogenous test (seeing) is negligible.

## V2 — measurements independent of the votes

### Pre-registration (written 2026-09-24, before any V2 correlation was computed)

**Framing.** Every measurement here is **independent of the votes**, never "ground truth".
Algorithmic errors that do not depend on the votes attenuate agreement towards chance; they cannot
manufacture it. The same argument made label noise conservative (D8).

**What was checked before writing this (counts and column definitions only).**
- **Axis ratio:** `expAB_r` and `deVAB_r` are populated for all 230,358 probe galaxies. There is
  **no `fracDeV_r` on disk**, so the brief's "expAB for discs, deVAB for ellipticals" split cannot
  be made photometrically. Making it from the smooth/featured votes would put the votes back into
  the measurement. **Declared deviation:**
  - `expAB_r` for every galaxy is primary: one vote-free rule, applied uniformly.
  - `deVAB_r` is the secondary.
- **Pitch angle (2b): blocked.**
  - Hart et al. (2017) publish no machine-readable table: it is not on VizieR, the paper has no
    data statement, and the Zenodo DOI they cite is their matching code.
  - The paper does fix the columns. ψ_galaxy is the SpArcFiRe measurement from reliable arcs.
    ψ_GZ2 = 6.37·w_avg + 1.30·m_avg + 4.34 (their Eq. 8) is **vote-derived** and must never be
    used.
  - 2b's design is fixed below so that it is hashed before any pitch-angle data exists. It runs
    when the catalogue is obtained.
- **B/T (2c): covered.**
  - Simard et al. (2011) table 1 (n_b = 4 bulge + exponential disc; VizieR J/ApJS/196/11) has
    1,123,718 rows. It covers 98.3% of the probe corpus, 73,541 of 74,829 in P2's union, and
    13,645 of the 14,191 union galaxies with t05 reach ≥ 21.
  - Coverage is not thin, so 2c runs.
  - Fit-quality columns: `e_(B/T)r` (median 0.01; 0.8% > 0.1) and `PpS`, the F-test probability
    that a bulge + disc is *not* required over a single Sérsic (77% ≤ 0.32).

**Common protocol.**
- M's 4-epoch checkpoint and P2's split (train fits, test scores).
- The embedding is z-scored per dimension over the union.
- **Decoding a measurement:** ridge regression, with α chosen by generalised cross-validation on
  **train** over logspace(−2, 4, 13).
  - Metric: Spearman(prediction, measurement) on test, with a 2,000-resample bootstrap 95% CI.
  - p from a 10,000-draw permutation of the measurement on test, two-sided, add-one.
- **Untrained reference:** the same fit on R's three untrained draws.
  - **Margin** = trained − (max untrained) … trained − (min untrained), as in V1.1.
- **Multiplicity:** BY across the confirmatory tests actually run in this stage: 2a decode, 2c
  decode B/T, 2c decode B_avg, 2c A_m, 2c A_v. So m = 5 (rank-1 threshold 4.4e-3; 1e-4 resolves).
- **Magnitude floor:** |ρ| < 0.10 is **negligible** whatever its p (significance and magnitude can
  disagree; D27).

**2a — axis ratio.**
- **Decoding.** Predict `expAB_r` (primary) and `deVAB_r` (secondary) over every union galaxy
  with a valid value.
  - Outcomes: **DECODED** (BY-significant, ρ ≥ 0.10, margin lower end > 0); **NOT ABOVE
    UNTRAINED** (significant, but the margin lower end ≤ 0); **NEGLIGIBLE** (significant,
    ρ < 0.10); **NOT DECODED**.
  - Expectation: DECODED with ρ ≈ 0.8. Axis ratio is a gross shape property, so I expect untrained
    to reach ≈ 0.6 as well: a real but modest margin.
- **Direction comparison — do the votes and the measurement define the same direction?**
  - The comparison sets **everything but the label source** equal: the ladder's canonical probe
    (`probe_direction`, logistic, the ladder's C), the same galaxies (the vote feature's
    eligible train set), and the same base rate.
  - The measurement-defined label is `expAB_r` < τ, with τ chosen so the label's base rate equals
    the vote label's on those galaxies. For "completely round" it is `expAB_r` > τ.
  - Pairs, with the expected sign of the aligned reading:

    | vote label | measurement label | expected |
    |---|---|---|
    | edge-on yes (t02 a04) | low b/a | + |
    | cigar (t07 a18) | low b/a | + |
    | completely round (t07 a16) | high b/a | + |

  - **Split-half disattenuation.** Train is split into two random halves, A and B, and both
    directions are fitted on each.
    - r_raw = mean of cos(vote_A, meas_B) and cos(vote_B, meas_A).
    - The reliabilities are cos(vote_A, vote_B) and cos(meas_A, meas_B).
    - r_d = r_raw / √(product of the reliabilities).
  - The same is computed for the **CAV** (mean difference, z-scored) beside the logistic
    direction. The two can disagree, and both are reported.
  - Random band: ±2.576/√384 = ±0.131.
  - Outcomes, logistic headline; the first match applies:

    | # | condition | verdict |
    |---|---|---|
    | 1 | either reliability < 0.3 | **UNRELIABLE** (the direction is not reproducible) |
    | 2 | r_raw < −0.131 | **OPPOSED** (the sign flip) |
    | 3 | r_raw within ±0.131 | **UNRELATED** |
    | 4 | r_d ≥ 0.8 | **SAME DIRECTION** |
    | 5 | 0.3 ≤ r_d < 0.8 | **RELATED** |
    | 6 | otherwise | **DISTINCT** (outside the band, but r_d < 0.3) |

  - Expectation: edge-on SAME DIRECTION or RELATED (r_d ≈ 0.7); cigar RELATED; round RELATED but
    weaker. Roundness votes on smooth galaxies mix axis ratio with concentration.

**2c — bulge-to-total (Simard 2011, `(B/T)r`), against bulge prominence (t05).**
- **Population (primary):** union galaxies in Simard with t05 reach ≥ 21, train/test by P2.
  - This uses the votes to *select* who was asked, not to measure; declared.
  - Secondary: every union galaxy with a B/T, for decoding only.
- **Vote score:** Masters et al. (2019) B_avg = 0.2·p(just noticeable) + 0.8·p(obvious) +
  1.0·p(dominant). The coefficients are arbitrary by the paper's own account.
- **Agreement first:** Spearman(B_avg, B/T) over the primary population, bootstrap CI. Reported
  before any encoder number.
  - Expectation: positive, ρ ≈ 0.4–0.5. The two agree in direction but disagree on many galaxies.
- **Decoding:**
  - ridge → B/T, and ridge → B_avg, both on test;
  - DECODED / NOT ABOVE UNTRAINED / NEGLIGIBLE / NOT DECODED, as in 2a.
- **Experiment D, the 2×2 of what the encoder carries beyond the other.** On test:
  - **A_m** = partial Spearman(ridge→B/T prediction, B/T | B_avg): measurement information the
    votes do not carry.
  - **A_v** = partial Spearman(ridge→B_avg prediction, B_avg | B/T): vote information the
    measurement does not carry.
  - Freedman–Lane, 10,000 draws, two-sided. Untrained margins are reported beside each.
  - "Carries" means BY-significant, positive, ≥ 0.10, and above every untrained draw.

  | A_m carries | A_v carries | verdict |
  |---|---|---|
  | yes | yes | **BOTH**: the encoder holds each beyond the other |
  | yes | no | **MEASUREMENT BEYOND VOTES**: sees structure people's votes miss |
  | no | yes | **VOTES BEYOND MEASUREMENT**: learned what people see |
  | no | no, and both decodes DECODED | **SHARED ONLY**: what the encoder carries is common to both |
  | no | no, and a decode fails | **NEITHER** |

  - A significant but negative A_m or A_v is reported as **INVERTED** on that leg (D27), never
    as "no".
- **Expectation:** VOTES BEYOND MEASUREMENT or BOTH. B/T from a two-component fit is noisy for
  small bulges, where the volunteers' "just noticeable" is still informative.
- **Degeneracy flags, not drops.** A galaxy is flagged if any of these holds: `petroRad_r` < 3″,
  `modelMag_r` > 17, `PpS` > 0.32, or `e_(B/T)r` > 0.1.
  - The primary includes flagged galaxies. The whole of 2c is repeated excluding them, as a
    sensitivity analysis.
  - A verdict that changes between the two is reported as **FLAG-SENSITIVE**.
- **Descriptive:** the median and IQR of B/T per plurality category (none, just noticeable,
  obvious, dominant). Is the ordering monotone, and are the middles wider than the endpoints?

**2b — pitch angle (design fixed now; runs only when Hart et al.'s table is obtained).**
- **Column rule:**
  - only galaxies with a **machine-measured** ψ_galaxy (reliable SpArcFiRe arcs, about half of
    6,222);
  - ψ_GZ2 is never read;
  - arm number from that catalogue is vote-derived and not used.
- The column's definition is verified against the paper before the join.
- **Selection is stated:** reliable-arc galaxies skew to clearly visible spirals.
- **Agreement:** Spearman(w_avg, ψ_galaxy). Expected negative: tight arms have a small pitch and
  a high w_avg.
- **Decoding and the 2×2:** exactly as 2c, with ψ_galaxy for B/T and w_avg for B_avg.
- **The sharp test of U3-C.**
  - Measured-ψ dispersion (IQR) of medium-plurality galaxies against the tight and loose groups,
    with a 2,000-resample bootstrap CI on each difference.
  - Outcomes:
    - **WIDER**: medium's IQR exceeds both, CIs excluding 0; supports "medium partly means
      couldn't tell".
    - **CONCENTRATED**: below both.
    - **INTERMEDIATE**: otherwise.
    - **INSUFFICIENT**: < 100 in any group.

*V2 pre-registration ends here: first 381 lines, SHA-1 `fbf23fc39087ddc57c4ffd0e7b85dfde7c8e6db8`.*

## V3 — the encoder's own directions

### Pre-registration (written 2026-09-24, before any V3 number was computed)

**The hypothesis**, licensed by R0: morphology survives size matching, so it is **decodable but not
salient**. It lives in low-variance directions beneath the nuisance axes. A clean negative for
plain PCA *is* that finding, not a failure.

**Tensor and partitions.**
- The banked M embedding (O1's bank), extracted by the ladder's own path at `DEFAULT_LAYER` = −2:
  the pooled tensor every probe reads.
- **A** = P2's train (40,000) is for discovery and alignment. **B** = P2's test (34,829) is for
  the held-out test.
- Nothing fitted in V3 ever sees a B embedding before the alignment is frozen. The driver writes
  the frozen alignment to disk before it loads B's rows.

**Discovery (label-free, on A).**
- **Primary: covariance PCA** of A's embeddings, centred on A's mean and **not** rescaled.
  Salience is about the encoder's own variance, and z-scoring would equalise it away.
- **Secondary: correlation PCA** (A-standardised), reported beside it.
- **The spectrum.**
  - Eigenvalues and the participation ratio PR.
  - **k\* = round(PR)** is the primary "dominant subspace", fixed by the spectrum and not by
    labels. k ∈ {1, 2, 5, 10, 20, 50} is swept descriptively.
- **Well separated.** Refit PCA on two random halves of A. Component j is **stable** iff
  max_i |cos(v_j^(1), v_i^(2))| ≥ 0.9.
  - Unstable components are interpreted only as members of their near-degenerate subspace, never
    singly.
- **Naming the top components (descriptive).** Spearman of each top-20 PC score on A with:
  - the five nuisances (magnitude, SNR, size, redshift, seeing);
  - `expAB_r`;
  - orientation (cos 2θ, sin 2θ from R3's bank, elongated galaxies only, R3's gates);
  - the 37 vote fractions, each on its eligible A galaxies.
- **Expectation:** PC1 and PC2 each correlate |ρ| ≥ 0.5 with magnitude or size.

**Concept directions.**
- The **CAV** is μ₁ − μ₀ on A, in the same centred raw space as the PCA, with the ladder's binary
  labels (`feature_embeddings` on train).
- The logistic weight (≈ Σ⁻¹(μ₁ − μ₀)) is computed alongside **only to show the artefact**: it is
  pushed away from high-variance axes by construction. It never enters a verdict.
- **Nuisance CAVs as positive controls.** Median splits on A of magnitude, size, SNR, redshift and
  seeing. They should be salient if the hypothesis's premise holds.

**Salience test, per feature.**
- E_k(c) = ‖P_k c‖² / ‖c‖², the projection energy of the CAV onto the top-k PC subspace.
- **Two baselines, both reported:**
  - the isotropic k/384;
  - the **shuffled-label null**. The CAV of randomly permuted labels, with the same class sizes, is
    pure noise contrast ∝ Σ^½·g, and its energy concentrates in the top PCs by the variance share
    alone. It is the fair baseline for "more or less in the dominant axes than a random contrast".
- **Test:** E_k\*(c) against 10,000 shuffled-label CAVs, two-sided, add-one. BY within the family
  of the answers with a linear direction (R1/R2, full population in T1; m = 33).
  - `assert_null_resolution` runs first: 10,000 resolves BY rank-1 at m = 33.
- **Outcomes**, the first match applies:

  | # | condition | verdict |
  |---|---|---|
  | 1 | either class < 50 on A | **INSUFFICIENT** |
  | 2 | BY-significant, E above the null's median | **SALIENT**: over-represented in the dominant axes |
  | 3 | BY-significant, E below the null's median | **SUBMERGED**: under-represented there, i.e. decodable but not salient |
  | 4 | otherwise | **UNREMARKABLE**: indistinguishable from a random contrast's energy |

  - Magnitude beside each (D27): the salience ratio E_k\* / (null median). A significant ratio
    within [0.9, 1.1] is tagged *negligible*.
- **Untrained:** the same energies on R's three draws, as a range, without verdicts.
- **Expectation:**
  - The nuisance controls are SALIENT.
  - Smooth/featured is SALIENT: it is the one clean direction, and it co-varies with colour and
    concentration.
  - Most other morphology answers are SUBMERGED.

**Subspaces (descriptive).**
- The concept group is the span of the gated answers' CAVs, orthonormalised; the nuisance group is
  the span of the five nuisance CAVs.
- Principal-angle cosines between each group and the top-k\* PC subspace.
- The overlap O = tr(P_group·P_k\*) / min(dims), against its isotropic expectation max(dims)/384
  and against the same overlap for groups of shuffled-label CAVs (200 draws).
- **RSA is not re-run.** P's same-corpus Spearman of +0.643, between the embedding cosine matrix
  and the human vote-correlation matrix, is the representational-similarity result, cited.

**Held-out protocol: can a label-free axis stand in for the probe?**
1. **Discover:** covariance-PCA components 1…K on A, with **K = 50** a declared pool.
2. **Align on A:** for each feature, choose **one** component,
   j\* = argmax_{j ≤ 50} |Spearman(PC_j score, vote fraction)| on the feature's eligible A
   galaxies, and its sign.
   - **Degrees of freedom:** one choice among 50 components, plus a sign, about 6.6 bits. **No
     weights are fitted;** the labels only name.
3. **Freeze:** (j\*, sign) for every feature is written to `out/v3_alignment.json` before B's
   rows are read.
4. **Test on B:**
   - the AUC of the frozen signed PC score for the feature's ladder label on B;
   - beside it, the logistic probe fitted on A and scored on B, as the decodable reference;
   - **recovery** = (AUC_PC − 0.5) / (AUC_probe − 0.5);
   - p from a 10,000-draw permutation of B's labels, two-sided, add-one, BY across the 33;
   - the same protocol on the three untrained draws (their own PCA, alignment and B test).
- **Outcomes**, the first match applies:

  | # | condition | verdict |
  |---|---|---|
  | 1 | either class < 50 on B | **INSUFFICIENT** |
  | 2 | BY-significant with AUC_PC < 0.5 | **INVERTED**: the sign chosen on A fails on B |
  | 3 | not BY-significant | **NO AXIS** |
  | 4 | recovery ≤ the max untrained recovery | **NOT ABOVE UNTRAINED** |
  | 5 | recovery ≥ 0.8 | **ENCODER AXIS**: the concept *is* one of the encoder's own components |
  | 6 | 0.5 ≤ recovery < 0.8 | **PARTIAL AXIS** |
  | 7 | recovery < 0.5 | **WEAK AXIS** (significant, but mostly the probe's doing) |

  - **Qualifier:** if j\*'s |Spearman| with any nuisance on A is ≥ 0.5, the verdict is tagged
    **NUISANCE PROXY**, meaning the concept is being read off a nuisance axis.
- **Expectation:**
  - ENCODER AXIS or PARTIAL for smooth/featured only, and probably tagged NUISANCE PROXY.
  - WEAK AXIS or NO AXIS for most others.
  - That would be the salience finding, stated as held-out evidence.

**Discovery channel.**
- Among the top 20 components, the **stable** ones matching **no human label** (max |ρ| with any
  vote fraction < 0.3) are unnamed.
- For each unnamed component: its nuisance, axis-ratio and orientation correlations, plus grids of
  the 24 highest and 24 lowest B galaxies. The grids are **looked at**, and what they show is
  described.
- An unstable unnamed component is shown only as its near-degenerate pair's plane, labelled
  basis-arbitrary.
- **Expectation:** orientation (|ρ| ≥ 0.3 with cos 2θ or sin 2θ) is among the unnamed components.
  There is no rotation augmentation (D10), and R3 measured orientation.
- **Outcomes, per unnamed component:**
  - **ORIENTATION**: |ρ| ≥ 0.3 with cos 2θ or sin 2θ;
  - **NUISANCE**: |ρ| ≥ 0.3 with a nuisance or `expAB_r`;
  - **UNEXPLAINED**: neither, described from the grids;
  - **NO UNNAMED COMPONENT**: every stable top-20 component matches a label.

**Not run** (the brief's follow-ups): gradient normals (R1 found no nonlinearity), layer-by-layer
PCA, ICA, sparse PCA, SAEs, CCA and diffusion maps.

*V3 pre-registration ends here: first 516 lines, SHA-1 `5f0c90e1c1fc7a645c4d6569a79a3590c8211347`. V2 and V3 results follow.*

### V2 — result (`artifacts/out/v2_independent.json`)

All five confirmatory tests are BY-significant at the permutation floor (p = 1e-4, m = 5).

**2a — axis ratio.**

| measurement | test ρ (95% CI) | untrained | margin | state |
|---|---|---|---|---|
| `expAB_r` (primary) | **+0.535** (0.528, 0.543) | 0.263–0.274 | **+0.26…+0.27** | **DECODED** |
| `deVAB_r` | +0.556 (0.548, 0.563) | 0.282–0.295 | +0.26…+0.27 | DECODED |

- The encoder carries photometric axis ratio on galaxies it never trained a probe on, at twice the
  untrained level.
- **Lower than I expected** (ρ ≈ 0.8). b/a is a noisy measurement for small, round and
  PSF-dominated galaxies, and that noise attenuates. Because it is independent of the votes, it
  lowers ρ; it does not fake it.

*Direction comparison: the same probe, galaxies and base rate; only the label source differs.*

| vote label | n | label agreement | logistic: reliabilities (v, m) | r_raw | **r_d** | verdict | CAV: r_d | verdict |
|---|---|---|---|---|---|---|---|---|
| edge-on yes | 39,500 | 0.906 | 0.82, 0.85 | +0.82 | **0.98** | **SAME DIRECTION** | 0.73 | RELATED |
| cigar | 39,116 | 0.944 | 0.86, 0.85 | +0.84 | **0.98** | **SAME DIRECTION** | 0.98 | SAME DIRECTION |
| completely round | 39,116 | 0.866 | 0.73, 0.71 | +0.71 | **0.98** | **SAME DIRECTION** | 0.99 | SAME DIRECTION |

- **The direction the volunteers' votes define is, within split-half noise, the direction the
  measurement defines**, for all three.
- **Caveat.** The two labels are built to the same base rate on the same galaxies, and they agree on
  87–94% of them. Most of the r_d ≈ 1 is therefore built in. The comparison can detect a
  *different* direction only through the 6–13% of galaxies where the labels disagree. It says the
  encoder does not organise those disagreements along a separate axis. It is not an independent
  replication of the whole direction.
- **Edge-on's CAV is only RELATED** (0.73) while its logistic direction is SAME (0.98).
  - The mean-difference contrast for edge-on votes carries something the axis-ratio contrast does
    not, in the high-variance subspace that the logistic weighting discounts. Thin-disc and
    dust-lane structure is the plausible candidate.
  - This is the one place the two estimators disagree, and it is the V3 trap seen from the other
    side.

**2b — pitch angle: NOT RUN (blocked).** Hart et al. (2017) publish no machine-readable table; see
the pre-registration. The design is hashed above and waits for the catalogue.

**2c — B/T (Simard 2011) against bulge prominence.**
- Population: 13,516 union galaxies with t05 reach ≥ 21 and a finite B/T. 3,396 are flagged as
  degenerate-risk.
- **Agreement first:** Spearman(B_avg, B/T) = **+0.717** (0.708, 0.726); unflagged +0.666.
  - Far higher than the ~0.45 I expected.
  - In this sample, the volunteers' bulge prominence and a photometric decomposition largely agree.

| | ρ (95% CI) | untrained | margin | state |
|---|---|---|---|---|
| decode B/T | +0.648 (0.633, 0.663) | 0.413–0.431 | +0.22…+0.23 | DECODED |
| decode B_avg | +0.701 (0.688, 0.715) | 0.498–0.522 | +0.18…+0.20 | DECODED |
| decode B/T, every union galaxy (n = 34,132 test) | +0.765 (0.761, 0.770) | 0.563–0.580 | +0.19…+0.20 | (secondary) |

*Experiment D — what the encoder carries of each beyond the other (test, partial Spearman):*

| leg | M | untrained | margin | carries? |
|---|---|---|---|---|
| **A_m**: B/T beyond B_avg | **+0.358** | 0.151–0.152 | +0.21 | yes |
| **A_v**: B_avg beyond B/T | **+0.470** | 0.317–0.335 | +0.13…+0.15 | yes |

- **Verdict: BOTH**, and the same excluding flagged galaxies (A_m +0.349, A_v +0.460).
  **Not FLAG-SENSITIVE.**
- The encoder holds photometric bulge fraction that the votes do not carry, *and* the volunteers'
  judgement beyond what the decomposition measures.
- The learned margin is larger on the measurement leg (+0.21) than on the vote leg (+0.14). The
  untrained encoder already carries much of the vote-only part (0.32), plausibly the
  size/brightness cues volunteers also use.
- **B/T by plurality answer** (median [IQR]):

  | plurality | n | B/T median [IQR] |
  |---|---|---|
  | none | 820 | 0.05 [0.01–0.13] |
  | just noticeable | 8,677 | 0.13 [0.06–0.23] |
  | obvious | 3,938 | 0.37 [0.25–0.48] |
  | dominant | 81 | 0.59 [0.39–0.69] |

  - The ordering is monotone.
  - The IQR widens with B/T. That is heteroscedasticity, not middles wider than endpoints: the
    middles are **not** anomalously spread.
  - Obvious → dominant is ordered on the measurement, even though U3's A found it inverted in the
    embedding on 38 test galaxies.

**Against my stated expectations.**
- *Right:* 2a DECODED; 2c BOTH.
- *Wrong:* 2a's ρ (0.54, not 0.8). The B_avg–B/T agreement (0.72, not 0.45). Round's direction
  (SAME, not merely RELATED).
- *Unpredicted:* edge-on's CAV/logistic disagreement.

### V3 — result (`artifacts/out/v3_own_directions.json`, `out/v3_alignment.json`, grids `out/v3_pc*_{low,high}.png`)

**The spectrum.** The covariance-PCA participation ratio on A is **9.7**, so **k\* = 10**.

| top k | 1 | 2 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| share of variance | 0.249 | 0.367 | 0.596 | 0.772 | 0.900 | 0.988 |

- The untrained draws are far more collapsed (PR 1.5–1.6); correlation PCA gives PR 13.1.
- Every top-20 component is split-half stable (≥ 0.97), except the near-degenerate pair PC13/PC14
  (0.72, 0.71).

**The top components are not brightness and size. That premise of the hypothesis is wrong.**
- **PC1 (25%) and PC2 (12%)** correlate with nothing measured: |ρ| ≤ 0.08 with each of
  - magnitude, SNR, size, redshift and seeing;
  - `expAB_r` and orientation;
  - all 37 vote fractions.
- The measured nuisances first appear at **PC3–PC5**:
  - PC4: size −0.40, magnitude +0.38, redshift +0.33;
  - PC3: SNR −0.26 and bulge "none" +0.31;
  - PC5: about ±0.23 on all four.
- Beyond that, they are spread thinly through PC5–PC12.
- **No top-20 component carries orientation**: max |ρ| with cos 2θ or sin 2θ is 0.06. My
  expectation was wrong. R3's orientation signal (a ridge fit) lives in low-variance directions.

**Salience: every gated answer reads UNREMARKABLE. That is an instrument failure, not a finding.**
- The embedding is so anisotropic that a shuffled-label CAV, i.e. pure noise contrast ∝ Σ^½g,
  already puts a **median 0.755** of its energy in the top 10.
- The null's 1st–99th percentile band is 0.39–0.93. Concept energies are 0.70–0.98, and the
  smallest p is 0.066 (irregular).
- **SALIENT could not fire by arithmetic.** Even a CAV lying wholly in the top 10 (energy 1.0) sits
  just beyond the null's 99th percentile, and could not reach BY's rank-1 threshold at m = 33.
  Only SUBMERGED was reachable.
- I did not tabulate the reachable p before hashing. That was exactly the check memory records
  ("check a gate couldn't fire by arithmetic"), and I missed it.
- The four photometric nuisance controls read the same (ratio 1.27–1.29, p ≈ 0.07). They are
  indistinguishable from concepts under this null.
- **Seeing is the one significant result: SUBMERGED** (energy 0.09 against 0.75; p = 1e-4). The
  atmosphere lives almost entirely outside the dominant subspace. It is the only control the test
  could see, because it is the only one on the reachable side.

**What the numbers do say (descriptive, against the isotropic 10/384 = 0.026).**
- Every concept CAV, and every nuisance CAV except seeing, puts **70–98% of its energy in the
  top 10**, 27–38 times isotropic.
- Almost none of that energy is in PC1–PC2, which hold 37% of the variance: the k = 2 energy is
  0.01–0.18 for concepts and 0.02–0.11 for nuisances. The bulk arrives by k = 5.
- **Morphology and the photometric nuisances share one subspace, PC3–PC10.** The concept span's
  overlap with the top 10 is 0.93; the nuisance span has three principal cosines ≥ 0.97.
- So the picture is neither "morphology beneath the nuisance axes" nor "morphology is the dominant
  axes". The two most dominant directions are something **unmeasured**, and morphology and
  visibility sit together in the next eight.
- **The logistic-weight trap, demonstrated.** Every logistic direction has top-10 energy
  < 0.001, against 0.70–0.98 for the same answer's CAV. Comparing PCs with probe weights would
  have "shown" that no concept lives anywhere near the dominant variance.

**Subspaces.**
- The concept group (33 CAVs) has overlap 0.93 with the top 10, against an isotropic 0.086 and
  **0.91–0.95 for groups of shuffled-label CAVs**. The group test is powerless for the same
  reason.
- The nuisance group (5 CAVs) has principal cosines 0.99, 0.98, 0.97, 0.55 and 0.14. The last is
  seeing.
- RSA is P's +0.643, cited, not re-run.

**Held-out protocol.**
- One PC out of 50, plus a sign, chosen per answer on A; about 6.6 bits; frozen to disk before B.

| verdict | n | answers |
|---|---|---|
| ENCODER AXIS | **0** | — |
| PARTIAL AXIS | 3 | bulge obvious (0.57), irregular (0.63), other (0.56); all on PC3 |
| WEAK AXIS | 8 | smooth (0.44, PC4), spiral / no spiral (0.48 / 0.45), no bulge (0.49), just noticeable (0.50), rounded bulge (0.43), ring (0.26), 2 arms (0.26) |
| NOT ABOVE UNTRAINED | 15 | features, edge-on ×2, dominant, anything odd ×2, round, in-between, cigar, lens, disturbed, merger, t09 no bulge, tight, can't tell |
| NO AXIS | 5 | bar ×2, dust lane, boxy, 4+ arms |
| INVERTED | 2 | loose, 1 arm |

(recovery = (AUC_PC − 0.5) / (AUC_probe − 0.5), on B)

- **No human concept is one of the encoder's own components.** The best frozen single component
  recovers 0.56–0.63 of the probe's margin above chance, and all three come from **PC3**, which
  itself carries SNR (−0.26).
- The labels pick PC3 or PC4 for most answers. These are the first components the measured
  nuisances load on.
- **15 of 33 do no better than the same protocol on an untrained encoder**, whose PC1–PC2 carry
  gross structure well enough. Anything-odd untrained recovers 0.85 against M's 0.63.
- **No NUISANCE PROXY tag fired.** The largest nuisance |ρ| on a chosen component is 0.40 (PC4,
  size), below the 0.5 bar. PC4 is nonetheless the size/magnitude/redshift component in all but
  name.
- **The two INVERTED answers** (loose, 1 arm) chose PC4 on A with |ρ_A| ≤ 0.11 and failed on B.
  That is weak selection on the visibility component, not a stable axis.

**Discovery channel.**
- 17 stable top-20 components match no vote at |ρ| ≥ 0.3. PC3 is the exception (0.31), and
  PC13/PC14 are basis-arbitrary and not interpreted.
- PC4 is NUISANCE (size). Every other one is **UNEXPLAINED** at the pre-registered 0.3 bar, with
  its best nuisance |ρ| ≤ 0.24.
- **What the PC1 and PC2 grids show:**
  - **PC1.** High end: large, centred, bluish discs and spirals in clean fields. Low end: smaller,
    redder, compact galaxies in crowded fields, with bright stars and edge artefacts.
  - **PC2.** High end: small galaxies on visibly noisy backgrounds. Low end: large, bright spirals
    and edge-on discs.
- The extremes differ visibly in apparent size, colour and field crowding, yet over all of A the
  components correlate ≤ 0.08 with size and magnitude. What separates the tails does not organise
  the bulk.
- **Candidates, none of them measured here:** colour (g − r is not in the corpus), sky noise
  level, and companion or star count in the stamp. Named as the next measurement, not claimed.
- **ORIENTATION: none**, against my expectation.

**Verdict on the hypothesis ("decodable but not salient").**
- **Held-out: supported.** Every answer the ladder decodes is poorly recovered, or not recovered
  at all, by any single label-free component. The best is 0.63 of the probe's margin, and 15 of 33
  are no better than untrained.
- **The salience test itself: uninformative.** SALIENT was arithmetically unreachable.
- **The mechanism: not as stated.** Morphology is not *beneath* brightness and size. Both sit
  together in PC3–PC10, and the two largest directions are something neither the votes nor the
  photometry measures.
- **A plain-PCA negative, as the brief anticipated.** The route to rung 4 is ICA or sparse
  dictionaries, which the brief lists as follow-ups. The first job is to name PC1 and PC2.

**Against my stated expectations.**
- *Right:* no ENCODER AXIS; most answers WEAK or worse.
- *Wrong:*
  - PC1/PC2 are not brightness or size.
  - The nuisances are not SALIENT; they are untestable, as are the concepts.
  - Smooth/featured is not ENCODER or PARTIAL.
  - Orientation is absent from the top 20.
- *Instrument:* the salience gate could not fire (D27's lesson, and memory's arithmetic check),
  reported rather than re-run with a changed null.

### Corrected (Brief W1): the energy test is void, and the salience result is the held-out test

- **The energy test could not fire, and is recorded as VOID, not "UNREMARKABLE".** Its design
  paired two incompatible things:
  - the **mean-difference direction (CAV)**;
  - a **k/384 random-direction baseline**.
- A CAV's own sampling noise is covariance-shaped (∝ Σ^½·g). A shuffled CAV therefore concentrates
  in the top components by the variance share alone: here a median 0.755 of its energy, with the
  99th percentile at 0.93.
- Against that null, no concept could exceed the bar. The null saturated the statistic's range, so
  no row of the verdict table said anything about the encoder.
- The design came from the label-free note (projection energy against k/384, with the CAV rather
  than the logistic weight), not from Brief V. V adopted it without running a planted positive
  through it. D28 records the rule that would have caught it.
- **Seeing's SUBMERGED stands as a descriptive observation only.** The low side of the range was
  reachable, but a test that cannot reach one of its states is not read on the other.
- **Salience was answered by the held-out recovery test, which could fail and did not.**
  - No human concept is one of the encoder's own components.
  - The best single frozen component reaches ≤ 0.63 of the probe's margin above chance.
  - 15 of 33 answers are no better than the same protocol on an untrained encoder.
  - **That is the salience result: decodable, not salient.**
