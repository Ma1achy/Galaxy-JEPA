# Brief U — findings

M's 4-epoch checkpoint, P2's union and split (40,000 train / 34,829 test).

## U1 — bar × winding and arm count, against the same-corpus votes

### Pre-registration (written 2026-09-24, before any U1 number was computed)

**Hypothesis under test (not assumed).** Volunteers tie bar strongly to spiral features (v1:
+0.42, +0.51). If the human correlation is strong and the encoder's weak, the human correlation
looks like labelling bleed, not image structure.

**Measurements** (`artifacts/u1_bar_winding.py`; full population; directions are the ladder's
canonical probe fitted on train):
- **Encoder cosine**: bar's unit direction · partner's unit direction. Partners are tight,
  medium and loose winding, and arm counts 1, 2, 3, 4 and 4+ ("can't tell" excluded).
  - This construction reproduces Brief P's matrix (bar × tight −0.239 is a check).
  - Medium winding and arms 3 and 4 fail existence (R4), so their directions are weakly
    determined. They are reported and flagged.
- **Human correlation (the test)**: P's same-corpus matrix. That is the Pearson correlation of
  vote fractions, pairwise-complete over the galaxies eligible for both answers
  (`human_vote_correlation`, n ≥ 200).
- **Human correlation, tree-zero (diagnostic only)**: the same over all 74,829 galaxies, keeping
  GZ2's stored 0.0 for a question never reached.
  - In this construction the decision tree itself ties bar to every spiral answer: a galaxy that
    is not a featured, face-on disc has zero for both.
  - I expect it to be far larger than the pairwise-complete one, and v1's +0.42/+0.51 are likely
    of this kind. It is the mechanical form of "confident featured votes spilling downstream".
    It is not the test, because it compares the tree, not what volunteers said about the same
    galaxies.
- **Null.**
  - Random directions: cosine sd = 1/√384 = 0.051 for isotropic unit vectors.
  - The embedding is not isotropic, so as a check I also measure bar's direction against partner
    directions fitted on shuffled partner labels (30 per partner).
  - The null sd is the **wider** of the isotropic value and the shuffled-label median.
  - The band is ±2.576·sd (99%, two-sided). With 8 pairs, a 95% band would expect 0.4 false
    "outside".
- **Untrained reference**: the same cosines on R's three untrained draws, reported as a range.

**Per-pair read**, with "human-strong" meaning |r| ≥ 0.30, the same cut the encoder's pair list
uses:

| class | condition |
|---|---|
| HUMAN-ONLY | human-strong, encoder cosine inside the band |
| SHARED | human-strong, encoder outside the band, same sign |
| CONTRARY | human-strong, encoder outside the band, opposite sign |
| ENCODER-ONLY | human weak, encoder outside the band |
| NEITHER | human weak, encoder inside the band |

**Verdict on the hypothesis:**
- **SUPPORTED (labelling bleed)**: every human-strong pair is HUMAN-ONLY.
- **REJECTED (the encoder carries the association)**: every human-strong pair is SHARED.
- **MIXED**: anything else, reported pair by pair.
- **UNTESTABLE**: no pair is human-strong on the pairwise-complete matrix.
  - Then the premise itself fails on the right construction: volunteers do *not* tie bar strongly
    to winding or arm count on the same galaxies.
  - If the tree-zero correlations are strong, v1's numbers are the decision tree, not an
    association volunteers expressed. That also answers the bleed question, mechanically, and is
    reported that way.

**My expectation, stated so it can be wrong.**
- On the pairwise-complete matrix, bar × winding is weak (|r| < 0.3). The tree zeros are what
  make v1's numbers large.
- Bar × 2 arms may be the exception (bars and grand-design two-arm spirals co-occur physically).
- The encoder cosines are within or near the band except bar × tight (−0.24), which is outside
  it.
- So my expected verdict is UNTESTABLE on winding, with bar × 2 arms read individually.

*U1 pre-registration ends here: first 68 lines, SHA-1 `73a9391c82e25e7d29c4b1d57fc9e177dbc3d8e7`,
taken before the run.*

### Result (`artifacts/out/u1_bar_winding.json`)

**Null.** Isotropic sd 0.051; shuffled-label median 0.058. The wider is used, giving a 99% band of
**±0.150**. Bar × tight reproduces P's −0.239.

| bar × | encoder cosine | human r (pairwise) | n | human, tree-zero | untrained (3 draws) | class |
|---|---|---|---|---|---|---|
| tight | **−0.239** | −0.190 | 43,605 | +0.063 | −0.12 / −0.06 / +0.04 | ENCODER-ONLY |
| medium | +0.103 | +0.094 | 43,605 | +0.238 | +0.08 / +0.13 / −0.02 | NEITHER |
| loose | +0.042 | +0.132 | 43,605 | +0.214 | −0.24 / −0.21 / −0.36 | NEITHER |
| 1 arm | −0.146 | −0.086 | 43,603 | +0.010 | −0.08 / −0.12 / −0.30 | NEITHER |
| **2 arms** | **+0.419** | **+0.382** | 43,603 | +0.417 | +0.23 / +0.29 / +0.10 | **SHARED** |
| 3 arms | −0.083 | +0.005 | 43,603 | +0.073 | +0.09 / +0.09 / +0.00 | NEITHER |
| 4 arms | −0.028 | −0.003 | 43,603 | +0.044 | +0.07 / +0.09 / +0.00 | NEITHER |
| 4+ arms | −0.141 | −0.065 | 43,603 | −0.008 | −0.22 / +0.04 / −0.19 | NEITHER |

**Verdict, by the pre-registered rule: REJECTED.** The labelling-bleed hypothesis is not supported.
- The only human-strong pair is bar × 2 arms, and the encoder carries it (+0.42 against the
  humans' +0.38).
- **The verdict rests on that single pair.**
- For winding the premise fails. On the same galaxies, volunteers do not tie bar strongly to any
  winding answer (|r| ≤ 0.19), so there is no human association for the encoder to lack.

**Against my stated expectation.**
- *Right:* bar × winding is weak on the pairwise-complete matrix; bar × 2 arms is the exception;
  bar × tight falls outside the band.
- *Wrong:* the tree-zero construction does **not** inflate the correlations (+0.06 to +0.24 for
  winding; 2 arms is +0.42 either way). So the source of v1's +0.42/+0.51 is not shown here, and I
  withdraw the guess that it was the tree.
- *Wrong:* the verdict. I expected UNTESTABLE on winding. The rule reads REJECTED because one
  pair qualifies.

**Exploratory, labelled as such.** Across the eight pairs the encoder cosine tracks the human
correlation closely:
- Pearson **0.98**, same sign in 7 of 8. The eighth is 3 arms, −0.08 against +0.005.
- The mean of the three untrained draws manages 0.47.
- The untrained draws put bar × loose at −0.21 to −0.36, outside the band; M does not (+0.04).
  So the architecture has a bar/loose relation of its own, which training removed.

**For D13.** On this evidence the hard case's worry, a bar direction that is really "confidently
classified", does not describe M. M reproduces the specific human pattern (2 arms positive, tight
negative, the rest near zero) rather than a uniform lean towards every spiral answer. That fits
physical co-occurrence (bars with grand-design two-arm spirals), but it is not proof of it. The
Hart winding claim remains unresolved: the only winding signal on either side is a weak lean away
from tight.

## U2 — uncertainty geometry (Fig 2)

### Pre-registration (written 2026-09-24, after counts only; no U2 correlation had been computed)

**Protocol, unchanged (design 4, `uncertainty.py`).**
- The concept axis is the ladder's logistic probe, fitted on **train** consensus extremes
  (f ≥ 0.8 vs f ≤ 0.2). `assert_uncertainty_firewall` runs on the fit set.
- The **test** set's ambiguous middle (0.2 < f < 0.8) is held out entirely and projected onto the
  axis.
- Answers are gated on having a linear direction: R1 or R2 in T1's ladder. That is 33 in the full
  population and 29 in the conditional one.
- The **full population is the headline**, as in P's catalogue. The conditional population is
  reported beside it.
- Driver: `artifacts/u2_uncertainty.py`.

**Vote floor, local to this test (D8's scoping).**
- The floor applies to the question's reach, all answers' counts summed, for **both** the fitting
  extremes and the tested galaxies.
- Headline **21**, justified by precision: SE ≈ √(p(1−p)/n) gives ±0.11 at n = 21, and a galaxy
  truly at 0.8 is read at or below 0.5 about 0.1% of the time.
- **Fallback decided on counts** (`out/u2_counts.json`, produced before any correlation):
  - `N_MIN = 483` middle-band test galaxies, which detects Spearman ρ = 0.20 (the smallest effect
    of interest, declared) at BY's rank-1 threshold for m = 37 with 80% power (Fisher z).
  - Each train extreme class also needs ≥ 50 galaxies.
  - An answer that fails at 21 gets its 10-vote result, labelled as such. One that fails at 10 is
    reported *cannot resolve* and is not tested.
- Resulting headline floors:

  | population | at 21 | fallback to 10 | cannot resolve |
  |---|---|---|---|
  | full | 22 | 5: ring, disturbed, other, no bulge (t09), can't tell | 6: dominant, lens, dust lane, boxy, 1 arm, 4+ arms |
  | conditional | 20 | 2: no bulge (t09), can't tell | 7: ring, other, medium winding, 1, 3, 4 and 4+ arms |

  *(Corrected before any run: my first tally here read 23/5/5 and 18/2/8. The counts file is the
  authority, and the table now matches it.)*
- The floor is never lowered because a result came out weak.
- **Sensitivity:** everything is rerun at 10 and 37 and reported as robustness. The headline stays
  at each answer's pre-set floor.

**ON-AXIS.** Spearman(projection, f) over the test middle. Expected positive (the axis points to
the positive class).

**OFF-AXIS, against AMBIGUITY, not f.**
- **Ambiguity** is a = 1 − 2|f − 0.5|. It is a monotone transform of the answer's binary vote
  entropy, so Spearman against it equals Spearman against that entropy.
- It is measured over **every** test galaxy at the floor, so ambiguity spans 0–1.
- **Primary displacement:** S1's bend coordinate, the second eigenvector of R2's cross-fitted path
  covariance.
  - It is fitted on **train** galaxies at all f, in the per-dimension z-scored embedding, oriented
    towards the middle bins, and scored on test galaxies.
  - *Declared:* the bend direction is estimated from train galaxies' graded fractions. This is
    galaxy-held-out, but it is not the firewall's extremes-only construction.
- **Secondary displacement (firewalled):** distance from the line through the two train-extreme
  class centroids, in the same z-scored space. It uses extremes only.
- Expected sign positive: more ambiguous galaxies sit further off the axis.

**VISIBILITY CONTROL.**
- Four variables: faintness (`modelMag_r`), SNR (`snr_r`), size (`petroRad_r`, with
  `petrorad_suspect` rows excluded through `nuisance_valid`) and redshift (`z`). Rows missing any of
  them are dropped for the partial; the count is reported.
- **Method: partial Spearman.** Rank-transform the displacement, the ambiguity and all four
  variables; residualise the two ranks on the four (plus intercept); correlate the residuals.
  - p comes from a Freedman–Lane permutation, permuting the residualised ambiguity.
- **Separation, descriptive.** For each variable:
  - the displacement's partial correlation with it, controlling the other three (which variable
    the displacement carries);
  - the displacement-vs-ambiguity partial controlling it alone.

  Distance is expected to be the common cause of the first three.

**Statistics.**
- Permutation nulls shuffle the vote-side quantity: f on-axis, ambiguity off-axis.
- 10,000 draws, two-tailed, add-one. The minimum p is 1.0e-4, which resolves BY rank-1 at the
  actual family sizes (full: m = 27, 4.8e-4; conditional: m = 22, 6.2e-4). `assert_null_resolution`
  checks this before any draw.
- BY at α = 0.05 within each family: (population × measurement) at the headline floor. The five
  measurements are on-axis, bend raw, bend partial, line raw and line partial.
- **Untrained reference throughout:** every ρ is recomputed on R's three untrained draws and
  reported as a range, without permutation.
- **R2's linear readout ceiling** (`readout`, full population only) sits beside every on-axis
  result. Merger, spiral and "anything odd" are flagged (R2's robust curved paths).

**Verdicts per answer.**
- **On-axis.**
  - TRACKS: BY-significant, ρ > 0.
  - CONTRARY: BY-significant, ρ < 0.
  - NO EVIDENCE: otherwise.
- **Off-axis** (bend is primary, line secondary; the same rule for each).
  - NONE: raw not BY-significant positive.
  - **BEYOND VISIBILITY:** the partial is BY-significant positive.
  - **RESOLUTION-LIMITED:** raw significant, partial not significant, and |partial| ≤ ½|raw| on the
    same rows. Visibility accounts for it, which is a D13 verdict, measured.
  - UNRESOLVED: raw significant, partial not significant, but |partial| > ½|raw|.

**What I expect** (stated so it can be wrong):
- On-axis TRACKS for most answers with a strong direction (smooth/features, edge-on, rounded).
  Weaker or no evidence where the readout ceiling is low or the middle is thin.
- Off-axis: spiral shows raw ambiguity tracking (S1), and I expect it to be mostly
  **RESOLUTION-LIMITED**: S1's exploratory partials with visibility were 0.5–0.7.
- Across answers I expect visibility to account for most of the off-axis signal, with a minority
  BEYOND VISIBILITY.
- **Reconciliation to state in the write-up, whatever comes out:** R0 says the *concept
  direction* is not carried by size. U2 measures the *uncertainty offset from it*. Those are
  different objects, so a resolution-limited offset does not contradict R0.

## U3 — Scheme 2: graded existence for the ordered questions

### Pre-registration (written 2026-09-24, before any U3 number was computed)

**What this decides.** The graded existence test `schemes.py` left "deliberately not implemented"
is **A** below. A decides the Scheme 2 verdict; that decision is recorded as D26.

**Axes and categories.**
- Four ordered questions:
  - winding: tight, medium, loose;
  - bulge prominence: none, just noticeable, obvious, dominant;
  - roundness: round, in between, cigar;
  - arm count: 1, 2, 3, 4, 4+ ("can't tell" excluded).
- A galaxy's category is its **plurality** answer. The question's reach must be ≥ 21 votes, the
  same floor as U2, because membership conditions on the fraction. Exact ties are dropped.
- Full population (reach defines who was asked). Driver: `artifacts/u3_graded.py`.

**A — existence (the verdict).**
- Fit the logistic axis on **train** galaxies of the two **endpoint** categories only. The axis
  never sees a middle category.
- Project **test** galaxies of every category.
- **Jonckheere–Terpstra** ordering in the stated direction, computed as Kendall's τ-b between
  category index and projection. For a fixed tie structure τ-b is a rescaling of JT's S, so this
  assumes order only, not spacing.
- p is one-sided, by permuting category labels: 10,000 draws, add-one. BY across the four axes.
- **Middle landing in the middle** is the out-of-sample prediction: every adjacent-pair AUC
  (category j vs j+1, test) > 0.5.
- **Nulls:** shuffled labels (the permutation) and the untrained encoder (the same τ-b on R's three
  draws, K = 3, a range).
- **Verdict:**
  - **ORDERED**: BY-significant, **and** every adjacent AUC > 0.5, **and** M's τ-b above all
    three untrained draws.
  - PARTLY ORDERED: significant and above untrained, but some adjacent pair inverted.
  - NOT ABOVE UNTRAINED: significant, but not above untrained.
  - NOT ORDERED: otherwise.

**B — continuous score (winding and bulge only).**
- The coefficients are taken from Masters et al. (2019), verified in the paper's text:
  - Eq. 1: `w_avg = 0.5 p_medium + 1.0 p_tight` (1 = tightest).
  - Eq. 3: `B_avg = 0.2 p_just noticeable + 0.8 p_obvious + 1.0 p_dominant`.
- By the paper's own account both sets are arbitrary.
- **Correction to the brief:** w_avg is equally spaced (0, 0.5, 1). B_avg is **not**: its steps are
  0, 0.2, 0.8, 1.0.
- Spearman against A's projection on test galaxies, with a two-sided permutation p.
- Expected sign: **negative** for winding (the axis runs tight → loose while w_avg rises with
  tightness); **positive** for bulge.

**C — where the middle sits** (z-scored embedding, train and test galaxies at the floor).
- For each middle category, cross-fitted over random halves:
  - its position t along the endpoint-centroid line, against the equal-spacing expectation j/(K−1);
  - its **off-line distance**, √(r_A·r_B), where r is each half's residual off the line. This is
    unbiased by centroid noise.
- **"Off the line"** means an off-line share (distance / endpoint separation) ≥ 0.10 with
  r_A·r_B > 0.
- **Alignment with visibility.** The visibility direction is the label-free *pattern* cov(z, V).
  V is the mean rank-normal score of fainter, lower-SNR, smaller and higher-z, fitted on train
  galaxies.
  - The middle's off-line residual is compared with that direction's component perpendicular to
    the line, as a cosine.
  - Null: 200 shuffles of category labels. **Aligned** iff the cosine exceeds the null's 99th
    percentile.
- **Central-tendency check, directly on the covariates:** the middle category's mean V minus the
  linear interpolation of the endpoints at its t, with a bootstrap 95% CI. Positive means "less
  visible than its position predicts", the reading in which "medium" partly means "couldn't tell".

**Alongside:**
- The endpoint AUC (test) and Scheme 1's per-answer ladder AUCs (T1, full), for comparison.
- **A and U2 are distinct:** A asks whether the *groups* are ordered; U2 asks whether galaxies
  *within* the ambiguous middle are ranked.
- **D** (votes against measured winding) is deferred to the label-free note with the Hart
  pitch-angle join.

**What I expect.**
- Winding: ORDERED or PARTLY ORDERED, with medium landing between tight and loose. That would
  turn Scheme 1's medium-winding R4 into an ordinal success.
- Bulge prominence and roundness: ORDERED, with a strong axis.
- Arm count: at best PARTLY ORDERED, since 3, 4 and 4+ are thin.
- For C, I expect medium winding to sit **off** the line and aligned with visibility (the
  couldn't-tell reading).

*U2/U3 pre-registration ends here: first 302 lines, SHA-1 prefix `bf6959e6`, checked unchanged
after both runs.*

---

## U2 — result (`artifacts/out/u2_uncertainty.json`)

Full population, headline floors as pre-set (22 at 21 votes, 5 at 10). ρ is Spearman; p is add-one
over 10,000 draws (floor 1.0e-4); every verdict is BY within its (population × measurement)
family. "untr" is the range over R's three untrained draws. The partial drops the 0.3–4.1% of test
rows missing a visibility covariate.

| answer | floor | n mid | on-axis ρ | untr | ceiling | bend raw | bend partial | line raw / partial | on / bend / line |
|---|---|---|---|---|---|---|---|---|---|
| smooth | 21 | 15,319 | +0.476 | +0.32 | 0.99 | +0.209 | +0.149 | +0.01 / −0.01 | TRACKS / **BEYOND** / none |
| features | 21 | 12,852 | +0.399 | +0.28 | 0.98 | +0.247 | +0.220 | +0.01 / −0.03 | TRACKS / **BEYOND** / none |
| edge-on yes | 21 | 679 | +0.242 | +0.14…+0.20 | 0.99 | +0.018 | −0.001 | +0.02 / +0.01 | TRACKS / none / none |
| edge-on no | 21 | 679 | +0.242 | +0.14…+0.20 | 0.99 | +0.029 | +0.011 | +0.02 / +0.01 | TRACKS / none / none |
| bar | 21 | 2,882 | +0.200 | +0.07…+0.11 | 0.89 | +0.074 | +0.044 | −0.00 / +0.01 | TRACKS / **BEYOND** / none |
| no bar | 21 | 2,882 | +0.200 | +0.07…+0.11 | 0.88 | +0.072 | +0.043 | −0.00 / +0.01 | TRACKS / **BEYOND** / none |
| ⚑ spiral | 21 | 2,073 | +0.139 | +0.04…+0.05 | 0.98 | +0.142 | +0.038 | −0.02 / +0.01 | TRACKS / **BEYOND** / none |
| ⚑ no spiral | 21 | 2,073 | +0.139 | +0.04…+0.05 | 0.98 | +0.124 | +0.025 | −0.02 / +0.01 | TRACKS / RES-LIMITED / none |
| no bulge (t05) | 21 | 1,102 | +0.272 | +0.10…+0.12 | 0.90 | −0.015 | −0.013 | +0.05 / +0.02 | TRACKS / none / RES-LIMITED |
| just noticeable | 21 | 5,448 | +0.391 | +0.19…+0.20 | 0.80 | +0.065 | +0.038 | −0.00 / +0.00 | TRACKS / **BEYOND** / none |
| obvious | 21 | 4,018 | +0.357 | +0.17…+0.22 | 0.94 | +0.059 | **−0.039** | −0.05 / −0.01 | TRACKS / UNRESOLVED / none |
| ⚑ odd yes | 21 | 10,036 | +0.147 | +0.07…+0.08 | 0.98 | +0.126 | +0.097 | +0.13 / +0.09 | TRACKS / **BEYOND** / **BEYOND** |
| ⚑ odd no | 21 | 10,036 | +0.147 | +0.07…+0.08 | 0.98 | +0.131 | +0.100 | +0.13 / +0.09 | TRACKS / **BEYOND** / **BEYOND** |
| completely round | 21 | 7,519 | +0.128 | +0.06 | 0.88 | +0.012 | +0.018 | +0.01 / +0.03 | TRACKS / none / none |
| in between | 21 | 10,807 | +0.093 | +0.03…+0.04 | 0.96 | +0.005 | +0.003 | +0.00 / +0.01 | TRACKS / none / none |
| cigar | 21 | 2,976 | +0.174 | +0.08…+0.10 | 0.96 | +0.002 | −0.001 | −0.01 / −0.02 | TRACKS / none / none |
| ring | *10* | 1,193 | +0.171 | +0.06…+0.10 | 0.94 | +0.051 | +0.037 | −0.01 / −0.04 | TRACKS / **BEYOND** / none |
| disturbed | *10* | 1,953 | +0.236 | +0.18…+0.20 | 0.80 | +0.263 | +0.192 | +0.12 / +0.08 | TRACKS / **BEYOND** / **BEYOND** |
| irregular | 21 | 956 | +0.399 | +0.27…+0.31 | 0.81 | −0.038 | −0.018 | +0.07 / +0.02 | TRACKS / none / RES-LIMITED |
| other | *10* | 4,684 | +0.283 | +0.12…+0.16 | 0.89 | −0.003 | +0.017 | −0.07 / −0.03 | TRACKS / none / none |
| ⚑ merger | 21 | 1,551 | +0.122 | +0.02…+0.06 | 0.83 | +0.077 | **−0.089** | −0.09 / −0.01 | TRACKS / UNRESOLVED / none |
| rounded bulge | 21 | 708 | +0.363 | +0.13…+0.19 | 0.84 | +0.024 | +0.016 | −0.01 / −0.01 | TRACKS / none / none |
| no bulge (t09) | *10* | 1,282 | +0.332 | +0.20…+0.23 | 0.86 | +0.006 | +0.025 | +0.02 / −0.01 | TRACKS / none / none |
| tight | 21 | 2,861 | +0.238 | +0.15…+0.18 | 0.92 | +0.116 | +0.125 | +0.03 / +0.00 | TRACKS / **BEYOND** / none |
| loose | 21 | 1,387 | +0.196 | +0.09…+0.12 | 0.91 | −0.010 | −0.061 | −0.03 / +0.02 | TRACKS / none / none |
| 2 arms | 21 | 2,050 | +0.151 | +0.08…+0.10 | 0.95 | +0.051 | +0.055 | +0.01 / +0.01 | TRACKS / **BEYOND** / none |
| can't tell | *10* | 3,673 | +0.100 | +0.01…+0.04 | 0.60 | +0.104 | +0.068 | +0.03 / +0.00 | TRACKS / **BEYOND** / none |

*Italic floor* = the pre-set 10-vote fallback, labelled as such. ⚑ = R2's curved paths (merger,
spiral, "anything odd"). The ceiling is R2's linear readout ceiling (full population only).

**Tallies.**

| population | on-axis | bend: beyond / res-limited / unresolved / none | line: beyond / res-limited / none |
|---|---|---|---|
| full (m = 27) | TRACKS 27 | 13 / 1 / 2 / 11 | 3 / 2 / 22 |
| conditional (m = 22) | TRACKS 22 | 9 / 1 / 3 / 9 | 2 / 2 / 18 |

### On-axis

- **Every gated answer TRACKS in both populations.** Within the held-out ambiguous middle, a
  galaxy's projection on the extremes-only axis ranks its vote fraction (ρ +0.09 to +0.48).
- **Every real ρ is above all three untrained draws.** The untrained encoder is far from zero,
  though. It reaches +0.32 on smooth and +0.31 on irregular, so a large share of the on-axis
  ranking is architectural (plausibly visibility). The learned increment is ρ ≈ +0.03 (disturbed)
  to +0.16 (smooth).
- **The readout ceiling does not predict the on-axis ρ.** Ceilings are 0.80–0.99 for all but
  can't-tell (0.60), yet ρ spans 0.09–0.48. The ceiling reads bin *centroids*; U2 ranks individual
  galaxies, whose scatter about the centroid path is what caps ρ. can't-tell, the lowest ceiling,
  has the second-lowest ρ; beyond that the two do not co-vary.
- **Sensitivity (floors 10 and 37).**
  - On-axis ρ keeps its sign at every floor for every answer whose middle has more than ~70
    galaxies.
  - The one sign change is ring at 37, with n = 7: uninterpretable.
  - Magnitudes move by ≤ 0.05 between 10 and 21 for all but t09 no-bulge (0.332 → 0.194 at 21,
    where it is not headline).
  - At 37 the thin middles shrink ρ (edge-on 0.075 at n = 93) without reversing it.

### Off-axis (displacement vs ambiguity a = 1 − 2|f − 0.5|)

- **Bend (primary): 13 of 27 BEYOND VISIBILITY.** More ambiguous galaxies sit further along the
  bend coordinate even after ranks are residualised on faintness, SNR, size and redshift.
- The strongest are:

  | answer | raw | partial | untrained partial |
  |---|---|---|---|
  | features | +0.25 | +0.22 | +0.11…+0.13 |
  | smooth | +0.21 | +0.15 | +0.06…+0.07 |
  | disturbed | +0.26 | +0.19 | −0.07 |
  | tight | +0.12 | +0.13 | +0.06…+0.07 |
  | anything odd | +0.13 | +0.10 | +0.00…+0.02 |

  Tight's partial does not shrink at all under visibility control.
- **Only one answer is RESOLUTION-LIMITED on the bend, and it is spiral's.** For spiral, visibility
  removes 70–80% of the raw effect (0.142 → 0.038; 0.124 → 0.025).
  - "No spiral" falls under the half-raw bar and is RESOLUTION-LIMITED.
  - "Spiral" keeps a residual +0.038 that is still BY-significant, so the rule calls it BEYOND.
  - Read as a pair: spiral's uncertainty offset is *mostly* visibility with a small residual.
  - The separation names the carrier. Controlling size alone takes spiral's raw 0.13 to 0.048, and
    faintness to 0.055. Controlling redshift alone leaves 0.150. Apparent size and brightness carry
    it; distance per se does not.
- **The line (secondary, firewalled) is almost silent.**
  - It is BEYOND on three answers only: anything-odd ×2 (+0.09) and disturbed (+0.08).
  - It is RESOLUTION-LIMITED on two: t05 no-bulge and irregular.
  - The bend sees what the line does not because the bend direction is *fitted to* the graded
    path, and it is declared non-firewalled. The line only asks whether ambiguous galaxies fall off
    the extreme-to-extreme chord, and for most answers they do not.
- **Untrained reference.** The untrained bend partials are small (|ρ| ≤ 0.13) on most answers.
  They exceed M on merger and loose, below.
- **Sensitivity.** The bend partial is stable across floors on smooth, features, anything odd,
  tight and disturbed. It is floor-sensitive on:
  - t05 no-bulge: +0.134 at 10, −0.013 at 21;
  - completely round: +0.078 at 10, +0.018 at 21.

  Below 21 votes, part of the off-axis "ambiguity" is vote-count noise, the reason the floor exists.

### Flagged and odd

- **Merger (⚑): the partial reverses sign.** Raw +0.077 (BY-significant), partial −0.089 (p = 1e-4).
  - The rule calls it UNRESOLVED, because it has no state for a reversal. That is a gap in the
    pre-registered verdicts, not a finding of weakness.
  - Controlling size alone gives −0.093, so size drives the raw positive.
  - At fixed visibility, *more* ambiguous merger votes sit *less* far along the bend.
  - The untrained draws give +0.15…+0.16 on the same measurement, above M.
  - Merger's off-axis structure is therefore a size effect the architecture already carries.
    M's learned bend runs the other way.
- **Bulge "obvious": the same reversal.** Raw +0.059, partial −0.039 (UNRESOLVED).
  - The separation is non-monotone. Controlling faintness alone *raises* the partial to +0.186;
    controlling redshift alone makes it −0.084. This is a suppressor structure among the
    covariates.
  - Reported as measured; not interpreted further here.
- **Loose.** The bend raw is −0.010 (NONE), but the partial is −0.061 at p ≤ 1e-4. The rule tests
  the positive direction only, so this goes unscored. Loose-winding ambiguity sits *nearer* the
  bend's origin once visibility is held fixed. The untrained draws sit at +0.09…+0.10.
- **Anything odd (⚑): the cleanest off-axis result.** It is BEYOND on both displacements, the
  partial holds 75% of the raw, and it is near zero untrained.

### Against my stated expectation

- *Right:* on-axis TRACKS for the strong directions. Spiral's offset is mostly visibility (70–80%
  removed), as S1's 0.5–0.7 partials suggested.
- *Wrong:* "visibility accounts for most of the off-axis signal, with a minority BEYOND".
  - Visibility accounts for spiral's and little else.
  - Of the 16 answers with a significant raw bend, 13 survive the partial, and most keep more than
    half their raw.
  - The pre-registered D13 reading, *resolution-limited*, applies to one answer.
- *Wrong in the letter:* spiral itself splits BEYOND / RESOLUTION-LIMITED on a residual of 0.038.
  The verdict is significance-driven; the magnitudes say "mostly visibility".
- *Unpredicted:* the on-axis ranking is on every gated answer, including the weak ones
  (in-between 0.093, can't-tell 0.100). Also unpredicted: the merger and obvious-bulge sign
  reversals, and loose's negative partial.

### Reconciliation with R0

R0 found the *concept direction* is not carried by size: matched re-probes retained the effect.
U2 measures the *uncertainty offset* from that direction, a different object. The two agree where
they overlap:

- The offset is visibility-driven only for spiral.
- Merger's raw offset is size-driven, and its direction reverses under control.

Neither contradicts R0: a concept can sit on a size-robust axis while the galaxies humans argue
about are displaced from it by how visible they are. Where visibility does *not* account for the
offset (13 answers), the ambiguity is carried in the representation beyond the four covariates.
That is the positive D13 reading: disagreement has a geometry of its own.

### Restated (Brief V1): margins over untrained, and magnitude beside every verdict

- **Margin:** trained − untrained. The interval runs from trained − (the max of the three draws) to
  trained − (the min), as existence is stated against the untrained bar.
- **Retention:** bend partial / raw on the visibility-complete rows.
- **Reading:** *mostly visibility* < 0.5 ≤ *mostly beyond*; *reversed* < 0, whatever the
  significance.
- The verdict column is U2's, unchanged; it was not recomputed with the reversal state (V1.4).

| answer | on-axis ρ | **on-axis margin** | bend verdict (U2) | retention | reading | bend-partial margin |
|---|---|---|---|---|---|---|
| t01 smooth | +0.476 | +0.15…+0.16 | BEYOND VISIBILITY | +0.73 | mostly beyond | +0.08…+0.09 |
| t01 features or disk | +0.399 | +0.12…+0.12 | BEYOND VISIBILITY | +0.90 | mostly beyond | +0.09…+0.11 |
| t02 yes | +0.242 | +0.04…+0.10 | NONE | — | — | -0.01…+0.02 |
| t02 no | +0.242 | +0.04…+0.10 | NONE | — | — | -0.04…+0.07 |
| t03 bar | +0.200 | +0.09…+0.13 | BEYOND VISIBILITY | +0.61 | mostly beyond | +0.02…+0.03 |
| t03 no bar | +0.200 | +0.09…+0.13 | BEYOND VISIBILITY | +0.61 | mostly beyond | +0.02…+0.07 |
| t04 spiral | +0.139 | +0.09…+0.10 | BEYOND VISIBILITY | +0.29 | **mostly visibility** | +0.13…+0.13 |
| t04 no spiral | +0.139 | +0.09…+0.10 | RESOLUTION-LIMITED | +0.22 | **mostly visibility** | +0.12…+0.12 |
| t05 no bulge | +0.272 | +0.16…+0.18 | NONE | — | — | +0.04…+0.06 |
| t05 just noticeable | +0.391 | +0.19…+0.20 | BEYOND VISIBILITY | +0.56 | mostly beyond | +0.02…+0.06 |
| t05 obvious | +0.357 | +0.14…+0.18 | UNRESOLVED | -0.74 | **reversed** | -0.11…-0.10 |
| t06 yes | +0.147 | +0.07…+0.07 | BEYOND VISIBILITY | +0.75 | mostly beyond | +0.07…+0.09 |
| t06 no | +0.147 | +0.07…+0.07 | BEYOND VISIBILITY | +0.75 | mostly beyond | +0.08…+0.10 |
| t07 completely round | +0.128 | +0.07…+0.07 | NONE | — | — | +0.02…+0.02 |
| t07 in between | +0.093 | +0.05…+0.06 | NONE | — | — | +0.00…+0.01 |
| t07 cigar shaped | +0.174 | +0.08…+0.10 | NONE | — | — | -0.01…+0.01 |
| t08 ring | +0.171 | +0.07…+0.11 | BEYOND VISIBILITY | +0.74 | mostly beyond | -0.05…+0.12 |
| t08 disturbed | +0.236 | +0.04…+0.06 | BEYOND VISIBILITY | +0.74 | mostly beyond | +0.26…+0.27 |
| t08 irregular | +0.399 | +0.09…+0.13 | NONE | — | — | +0.11…+0.12 |
| t08 other | +0.283 | +0.13…+0.16 | NONE | — | — | -0.08…+0.01 |
| t08 merger | +0.122 | +0.06…+0.10 | UNRESOLVED | -1.40 | **reversed** | -0.25…-0.23 |
| t09 rounded | +0.363 | +0.17…+0.23 | NONE | — | — | +0.04…+0.06 |
| t09 no bulge | +0.332 | +0.11…+0.13 | NONE | — | — | -0.02…-0.01 |
| t10 winding a28 tight | +0.238 | +0.06…+0.09 | BEYOND VISIBILITY | +0.97 | mostly beyond | +0.05…+0.06 |
| t10 winding a30 loose | +0.196 | +0.08…+0.11 | NONE | — | — | -0.16…-0.15 |
| t11 number a32 2 | +0.151 | +0.05…+0.07 | BEYOND VISIBILITY | +1.05 | mostly beyond | +0.04…+0.04 |
| t11 number a37 cant tell | +0.100 | +0.06…+0.09 | BEYOND VISIBILITY | +0.67 | mostly beyond | +0.04…+0.10 |

- **The learned on-axis margin is +0.04 to +0.23.** It is real on every answer, but it is
  typically half the headline ρ or less. "Reproduces human uncertainty without seeing a vote" is
  mostly the architecture's for smooth/features (untrained +0.32 against +0.48).
- **Spiral is mostly visibility.** Retention is 0.29 and 0.22; U2's BEYOND on "spiral" was
  significance on a 0.038 residual. Its bend-partial margin over untrained is still +0.12,
  because the untrained draws bend the other way (−0.09).
- **Merger reverses**, with a margin of −0.24: the untrained encoders carry the positive
  association (+0.15), and M's is opposite. See V1.3 for the suppression check.
- **Disturbed is the largest learned off-axis margin** (+0.26). Anything-odd, smooth/features and
  tight follow, at +0.05 to +0.11.

## U3 — result (`artifacts/out/u3_graded.json`)

Full population, reach ≥ 21, plurality categories. τ is Kendall's τ-b (the JT statistic); p is
one-sided over 10,000 label permutations, add-one; BY across the four axes. Untrained values are
the three draws.

### A — existence (decides the verdict)

| axis | n test (per category) | τ-b | p | untrained τ-b | adjacent AUCs | endpoint AUC (untr) | **verdict** |
|---|---|---|---|---|---|---|---|
| winding (tight→loose) | 1,550 / 2,098 / 699 | **+0.289** | 1e-4 | 0.179–0.187 | 0.652, 0.672 | 0.796 (0.67–0.69) | **ORDERED** |
| bulge (none→dominant) | 374 / 4,126 / 1,945 / 38 | **+0.322** | 1e-4 | 0.132–0.158 | 0.719, 0.718, **0.495** | 0.878 (0.66–0.72) | **PARTLY ORDERED** |
| roundness (round→cigar) | 8,633 / 12,653 / 3,418 | **+0.401** | 1e-4 | 0.201–0.210 | 0.699, 0.805 | 0.919 (0.73–0.74) | **ORDERED** |
| arm count (1→4+) | 127 / 2,766 / 372 / 105 / 91 | **+0.210** | 1e-4 | 0.121–0.131 | 0.541, 0.640, 0.624, 0.729 | 0.926 (0.83–0.86) | **ORDERED** |

- All four are BY-significant at the permutation floor, and each τ-b sits well above all three
  untrained draws.
- The middle categories land between the endpoints on an axis that never saw them, the
  out-of-sample prediction, everywhere except obvious → dominant (AUC 0.495).
- Dominant is 38 test galaxies and trained its endpoint on 47. The inversion rests on the thinnest
  category in the test, but the rule reads it as it stands: PARTLY ORDERED.
- **Winding's medium sits between tight and loose.** Scheme 1 had medium winding at AUC 0.523 as a
  one-vs-rest answer. Graded, it is ordered.

### B — continuous score (Masters et al. 2019; coefficients arbitrary by the paper's account)

| axis | score | Spearman vs projection | p | expected sign |
|---|---|---|---|---|
| winding | w_avg | **−0.418** | 1e-4 | negative ✓ |
| bulge | B_avg | **+0.484** | 1e-4 | positive ✓ |

### C — where the middle sits

| axis | middle | t (equal spacing) | off-line share | cos with visibility (shuffled 1–99%) | ΔV (95% CI) |
|---|---|---|---|---|---|
| winding | medium | 0.60 (0.50) | **0.26 off** | **+0.66** (−0.41…+0.42) **aligned** | **+0.065** (+0.031, +0.102) |
| bulge | just noticeable | 0.54 (0.33) | 0.33 off | −0.48 (−0.62…+0.62) | −0.306 (−0.389, −0.213) |
| bulge | obvious | 0.94 (0.67) | 0.38 off | **−0.80** (−0.58…+0.54) anti-aligned | −0.580 (−0.729, −0.430) |
| roundness | in between | 0.47 (0.50) | 0.08 **on** | −0.64 (−0.41…+0.45) | +0.022 (+0.009, +0.034) |
| arms | 2 | −0.00 (0.25) | 0.14 off | −0.24 (−0.41…+0.40) | −0.141 (−0.240, −0.046) |
| arms | 3 | 0.18 (0.50) | 0.23 off | **−0.46** (−0.40…+0.43) anti-aligned | −0.172 (−0.272, −0.079) |
| arms | 4 | 0.45 (0.75) | 0.33 off | **−0.61** (−0.53…+0.48) anti-aligned | −0.168 (−0.277, −0.053) |

- **Winding: the couldn't-tell reading is supported.** Medium sits off the tight–loose line
  (26% of the endpoint separation). Its off-line residual aligns with the visibility pattern
  (cosine +0.66, beyond the shuffled 99th percentile). Its galaxies are less visible than their
  position predicts (ΔV +0.065, CI excludes 0). Part of "medium" is "harder to see".
- **Roundness: in-between sits on the line, at the halfway point** (t 0.47, off-line 8%). A clean
  ordinal axis. Its perpendicular cosine is moot at that distance; ΔV is small.
- **Bulge and arm count: the opposite of couldn't-tell.**
  - Their middles sit off the line in the direction *against* the visibility pattern, significantly
    so for obvious, 3 arms and 4 arms. Their galaxies are *more* visible than their position
    predicts (ΔV < 0, every CI excludes 0).
  - A middle bulge or arm count is what humans answer on the better-resolved galaxies.
  - Arm count's middles are also compressed towards "1 arm": 2 arms at t ≈ 0, and 4 arms at 0.45
    against 0.75. The line is ordered but unevenly spaced.
- Obvious bulge at t 0.94 sits nearly on top of dominant, the geometry behind the 0.495 adjacent
  AUC.

### Alongside

- **Scheme 1 per-answer AUCs (T1, full).** The endpoints are well separated when fitted as a
  contrast. As one-vs-rest answers they are weak.

  | axis | Scheme 1 AUCs (in axis order) | endpoint AUC (A) |
  |---|---|---|
  | winding | 0.589 / 0.523 / 0.654 | 0.796 |
  | bulge | 0.840 / 0.701 / 0.792 / 0.752 | 0.878 |
  | roundness | 0.746 / 0.602 / 0.846 | 0.919 |
  | arm count | 0.640 / 0.581 / 0.557 / 0.581 / 0.694 | 0.926 |

- **The untrained endpoint AUC for arm count is 0.83–0.86.** Most of 1-vs-4+ is architectural,
  plausibly apparent size and brightness. M's ordering margin over untrained (τ 0.21 vs ≤ 0.13) is
  what A credits.
- **A and U2 are distinct.**
  - A asks whether the *groups* (plurality categories) are ordered along an endpoint axis.
  - U2 asks whether galaxies *within* one answer's ambiguous middle are ranked by their vote
    fraction.
  - A group can be ordered while its members are unranked, and the reverse.
- **D** (votes against measured winding, Hart join) is deferred to the label-free note.
- **The untrained-MLP bar** is deferred to the rental runs, logged in TODO.

### Against my stated expectation

- *Right:*
  - Winding ORDERED, with medium between.
  - Roundness ORDERED.
  - Medium winding off the line and aligned with visibility.
- *Wrong:*
  - Bulge: PARTLY ORDERED, not ORDERED. The inversion is at obvious → dominant, on 38 galaxies.
  - Arm count: ORDERED, not "at best PARTLY". Every adjacent pair clears 0.5, the weakest being
    1 → 2 at 0.541.
- *Unpredicted:* the bulge and arm-count middles are displaced *against* visibility. The
  couldn't-tell mechanism is specific to winding.
