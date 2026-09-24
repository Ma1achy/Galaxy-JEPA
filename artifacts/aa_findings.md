# Brief AA — label efficiency, a deeper referee, and the pose code

Status: complete. AA2 MATCH VALID; AA3a INSTRUMENTAL; AA1 CONTRADICTED; AA3b MIXED (pose HARMLESS once ensembling is separated, exploratory). Each stage's section is hashed (SHA-1 over the section text up to its footer)
before its run. Order: AA2, AA3a, AA1, AA3b.

## AA2 — Galaxy Zoo DECaLS volunteer coverage

### Pre-registration

**Source: volunteer votes only.** These were verified and read before any match was made.

| campaign | file | record | MD5 (verified against Zenodo's API) | rows | tree |
|---|---|---|---|---|---|
| GZD-1/2 | `gz_decals_volunteers_1_and_2.parquet` | Zenodo 4573248 v0.0.2 (Walmsley et al. 2022), CC BY 4.0 | `e9de511c…` | 92,960 | bar yes/no; bulge none/obvious/dominant; no "can't tell" in arm count |
| GZD-5 | `gz_decals_volunteers_5.parquet` | same | `364d0b59…` | 253,286 | bar strong/weak/no; bulge 5-way; arm count adds "can't tell" |
| GZD-8 | `gz_desi_gzd8_volunteer_core_catalog.parquet` | Zenodo 8360385 v1.0.1 (Walmsley et al. 2023), CC BY 4.0 | `ba79d9d9…` | 54,716 | GZD-5's tree, plus "anything odd" |

- **Not used:** `gz_decals_auto_posteriors` and the GZ DESI deep-learning catalogues. Both are
  Zoobot predictions trained on these votes, which would make the referee circular. The GZD-8
  *extended* table (galaxies with ≥ 5 artifact votes) is also left out.
- **The schema is wrong about one column.** `schema.md` names the totals `{question}_total`, but
  the tables use `{question}_total-votes`. The GZD-8 tables have no total column at all. Totals
  here are always the sum of the mapped answer counts.
- **GZD-5 is thin.** It used active learning, so its median votes on the first question is 5
  (GZD-1/2: 38; GZD-8: 39). Most GZD-5 galaxies cannot be "confident" under the reach rule below.
  That is a property of the campaign, stated here, not a defect of the match.

**Match.** Each DECaLS row is matched to the nearest of our 230k within **3″** (the PAnDa
radius). One DECaLS row is kept per galaxy of ours, the nearest. DECaLS uses NSA `iauname`, not
SDSS objIDs, so the match is on coordinates.

**Referee choice.** For each galaxy and each question, the referee is the campaign with the most
votes on that question.

**Question map.** Each side reduces to fractions over the same categories.

| question | GZ2 | DECaLS | equivalence |
|---|---|---|---|
| smooth / featured | t01 smooth / features-or-disk / star-or-artifact | smooth / featured-or-disk / artifact | near: GZ2's "star or artifact" includes stars, but DECaLS pre-excludes stars |
| edge-on | t02 yes / no | disk-edge-on yes / no | equivalent |
| bar | t03 bar / no bar | GZD-1/2: yes / no. GZD-5/8: strong + weak / no | **not equivalent in GZD-5/8**: "weak" was added to catch bars that GZ2 volunteers call "no" |
| spiral | t04 spiral / no | has-spiral-arms yes / no | equivalent |
| bulge | t05 no + just noticeable / obvious / dominant | GZD-1/2: none / obvious / dominant. GZD-5/8: none + small / moderate + large / dominant | **not equivalent**: 4 vs 3 vs 5 options, collapsed to three. "Just noticeable" → none-or-small is a choice |
| winding | t10 tight / medium / loose | spiral-winding, same three | equivalent options; the image differs |
| arm count | t11 1 / 2 / 3 / 4 / > 4 | spiral-arm-count, same five | GZD-1/2 has no "can't tell", so it is dropped everywhere and compared on the five |

**Definitions (fixed now).**
- *Reached:* GZ2 total over the mapped categories ≥ 21 (U3's floor). DECaLS total ≥ 10.
- *SDSS uncertain:* reached, and plurality fraction < 0.6.
- *DECaLS confident:* reached, and plurality fraction ≥ 0.8.
- **Sharp set** = SDSS uncertain ∧ DECaLS confident. It is counted per question, per split
  (A / B / train / val / test / absent) and per referee campaign.

**Match states (D27).**
- **MATCH VALID** iff *both* of these hold:
  - the 60″-shifted chance-match count is < 1% of the true match count, in every campaign;
  - GZ2 against DECaLS smooth fraction (GZD-1/2, both reached) has Spearman ρ ≥ 0.5.
- **MATCH SUSPECT** otherwise. In that case every count below is reported, but no power state is
  claimed.

**Power (rough, fixed now), per test:**
- **(a) Can probes beat the SDSS voters?** On the sharp set, a probe score is compared with the
  SDSS vote fraction for predicting DECaLS's plurality answer (one-vs-rest on the commonest
  plurality). The statistic is a paired AUC difference.
  - Minimum detectable difference (MDD) = 2.8 · SE_HM(0.75) · √(2(1 − 0.5)), at 80% power,
    two-sided α 0.05.
  - Counted on B (embedded test, scorable now) and on B ∪ test (scorable after embedding).
  - **POWERED** if MDD ≤ 0.05. **UNDERPOWERED** if ≤ 0.15. **NOT POSSIBLE** otherwise, or if
    either class has < 10 galaxies.
- **(b) Winding referee.** A partial Spearman between the encoder's winding readout and DECaLS
  winding, visibility controlled, on B galaxies where both sides reached winding.
  - Minimum detectable effect (MDE) = 2.8/√(n − 3).
  - **POWERED** if MDE ≤ 0.10. **UNDERPOWERED** if ≤ 0.25. **NOT POSSIBLE** otherwise (also if
    n < 30).
- **(c) Imaging depth (D13).** A vote-fraction trend, SDSS against DECaLS, per question, on every
  galaxy where both sides reached. MDE as in (b), same states.

**D28 (`aa2_decals.py --planted`, run before hashing):**

| check | result |
|---|---|
| chance matches at 3″ after a 60″ shift | GZD-1/2 18, GZD-5 55, GZD-8 1 (of 93k / 253k / 55k rows) |
| MDD formula: power at a true ΔAUC = MDD (DeLong, 300 sims) | 100/100: 0.90 · 300/700: 0.86 · 1000/3000: 0.84 |
| MDE formula: power at a true ρ = MDE (300 sims) | n 100: 0.79 · 400: 0.81 · 1600: 0.77 |

- The chance rate is low enough that MATCH VALID can fire. A true 3″ match rate near zero would
  still read SUSPECT through the ratio rule.
- The power formulas deliver about 80% as claimed. The AUC formula is slightly conservative.
- AA2 reports no test outcome, so there is no hypothesis state to plant. The states above are
  coverage and power states, and each is reachable by arithmetic from n.

*AA2 pre-registration ends here: the AA2 section above (85 lines from "## AA2"), SHA-1 `24f57ac88cfff0a0c5985daf26b8a7562b42cb47`.*

### Results (`out/aa2_decals.json`)

**MATCH VALID.**
- Chance matches after the 60″ shift are 0.06% (GZD-1/2, 18 / 30,892), 0.06% (GZD-5,
  55 / 88,926) and 0.4% (GZD-8, 1 / 232) of the true matches.
- GZ2 against DECaLS smooth fraction: ρ = **0.82**, n 30,887.
- Median separation is 0.12″, and 95% are within 0.33″.

**Overlap: 108,113 of our 230,359 galaxies (47%).**

| split | A (probe train) | B (probe test = the whole test split) | train, not embedded | val |
|---|---|---|---|---|
| overlap | 18,801 | 16,219 | 56,854 | 16,239 |

| campaign | matched | median votes, first question |
|---|---|---|
| GZD-1/2 | 30,892 | 37 |
| GZD-5 | 88,926 | 5 |
| GZD-8 | **232** | 39 |

- GZD-1/2 and GZD-5 share 11,937 of our galaxies.
- GZD-8 barely overlaps: DR8 added galaxies mostly outside the SDSS spectroscopic sample.

**Per question, on the overlap.** "Reached" means GZ2 ≥ 21 and DECaLS ≥ 10 over the mapped
categories. "Agree" is the plurality agreement where both are confident.

| question | DECaLS answered | both reached | SDSS uncertain | DECaLS confident | **sharp** (B) | plurality agree | referee votes (median, GZD-1/2 / GZD-5) |
|---|---|---|---|---|---|---|---|
| smooth / featured | 108,113 | 55,165 | 30,296 | 17,201 | **1,189** (184) | 0.99 | 37 / 5 |
| edge-on | 94,297 | 13,886 | 1,181 | 24,918 | **46** (5) | 1.00 | 12 / 4 |
| bar | 86,868 | 10,754 | 5,300 | 10,889 | **233** (36) | 1.00 | 7 / 3 |
| spiral | 86,868 | 10,754 | 4,015 | 14,692 | **364** (58) | 1.00 | 7 / 3 |
| bulge | 86,868 | 10,751 | 12,888 | 8,564 | **900** (127) | **0.47** | 7 / 3 |
| winding | 51,157 | 7,264 | 16,095 | 1,808 | **276** (47) | 1.00 | 8 / 4 |
| arm count | 46,737 | 5,056 | 4,732 | 5,776 | **64** (13) | 1.00 | 8 / 3 |

- **The downstream DECaLS questions are thin.** Even in GZD-1/2 the median is 7–8 votes, so only
  12–40% of answered galaxies reach 10.
- The sharp set's referee is GZD-5 for 55–70% of each question's galaxies. Those are the GZD-5
  galaxies that were retired late with ≥ 10 votes.

**The bulge map fails.**
- Where both SDSS and DECaLS are confident, their bulge plurality answers agree only **47%** of the
  time. Every other question agrees 99–100%.
- The collapse (GZ2 "just noticeable" → none-or-small; DECaLS moderate + large → obvious) doesn't
  line up the answer boundaries.
- **DECaLS cannot referee bulge prominence under this map.** Any bulge use needs an ordinal
  treatment that is calibrated per campaign. That is not attempted here.

**Power.**

| test | n | MDD / MDE | state |
|---|---|---|---|
| (a) probes vs SDSS voters, one-vs-rest AUC on the sharp set, B | smooth 184 (178 : 6) · edge-on 5 · bar 36 (21 : 15) · spiral 58 (53 : 5) · bulge 127 (121 : 6) · winding 47 (46 : 1) · arm count 13 | bar 0.23; all others ∞ (a class < 10) | **NOT POSSIBLE**, every question |
| (b) winding referee: encoder readout vs DECaLS winding, B, both reached | 1,105 | ρ 0.084 | **POWERED** |
| (c) imaging depth, per question, both reached | smooth 55,165 · edge-on 13,886 · bar / spiral / bulge ≈ 10,750 · arm count 5,056 | ρ 0.012–0.039 | **POWERED**, every question (bulge subject to the map above) |

**Why (a) fails, and what would work.** This is a note for the next brief. It is not run and
not pre-registered here.
- The sharp set is almost one-sided by construction. Galaxies SDSS voters were unsure about, but
  DECaLS voters were sure about, are overwhelmingly the ones the deeper image resolves as
  *featured* or *has arms* (smooth 178 : 6; spiral 53 : 5; winding 46 : 1).
- A one-vs-rest AUC has almost no negatives. The brief's test needs a different statistic. For
  example, a paired sign test: on each sharp galaxy, is the probe's calibrated probability or the
  SDSS vote fraction nearer the DECaLS answer?
- That needs no negative class, and it can use every sharp galaxy outside A: smooth 1,003, bar
  199, spiral 298, winding 229.
  - At smooth's n, a sign test detects a 55 / 45 split at about 88% power.
  - That needs embedding the 56,854 overlap galaxies in "train, not embedded". Cheap locally
    (Y3's bank took the same route).

**Consequences for the three uses.**
- **(a)** Not possible as specified. It is possible with a paired sign test on about 1,000 smooth
  / featured galaxies, after embedding the non-union overlap.
- **(b)** DECaLS can referee winding: 1,105 B galaxies where both reached, plurality agreement 1.00
  where both are confident. The sharp set is 276 galaxies (47 in B).
- **(c)** The depth comparison is powered on every question except bulge, whose map fails. It
  needs no rental: the votes, not new images, carry it.

## AA3a — the pose code: instrumental or structural?

### Pre-registration

**What is already known** (`x_findings.md` §X1, exploratory):
- (PC1, PC2) transform as an image-plane polar vector.
- They correlate with the core's centroid about the stamp centre (R² 0.40 with a 3 px core
  aperture; the median offset is 0.47 px, not 3 px).
- But whole-stamp integer and sub-pixel shifts leave them unchanged (r = 1.00). So they are not
  position.
- An offset **between bands** survives any whole-stamp shift, and it was never tested. This test
  is that.

**Sample.** X1c's 2,000 galaxies (`x1_transform_bank.npz`). PCs are V3's covariance PCA on A,
z-scored by A's sd, for M and the three untrained draws.

**Measurement** (`aa3a_pose.offsets`). This is per band (g, r, i), on valid pixels only (not
edge- or interior-invalid), sky-subtracted (sigma-clipped outer annulus), positive-clipped,
within the Petrosian aperture:
- c_b = the light centroid.
- core_b = the centroid within 3 px of the band's σ = 1-smoothed peak.
- x runs right and y runs down, about pixel 127.5.

**Candidates.** Both are translation-invariant.
- **INTER** = [c_g − c_r, c_r − c_i, core_g − core_r, core_r − core_i], an 8-vector. This is
  band misregistration or differential refraction.
- **COMMON** = the band median of (core_b − c_b), a 2-vector. This is lopsidedness shared by all
  bands.
- The median was chosen after the fidelity plant below. The band mean had leaked a g-only shift
  into COMMON.
- Typical sizes: |INTER| is 0.20–0.22 px per component; |COMMON| is 0.22 px.

**Statistic.**
- 5-fold CV R² (rank-normal, linear) of PC1 and of PC2 on each candidate, and on both jointly.
- Permutation p from 1,000 shuffles of the PC.
- **BY family m = 4**: {INTER, COMMON} × {PC1, PC2}. The resolution is 1/1001, well under BY's
  rank-1 bar of 0.006.

**A candidate holds** iff CV R² ≥ 0.05 for **both** PC1 and PC2, and both are BY-significant. A
pose code is a 2-vector, so one component isn't enough.

**States (D27). The first match applies:**

| # | condition | state |
|---|---|---|
| 1 | neither holds | **NEITHER** |
| 2 | both hold, and each adds ≥ 0.02 joint R² beyond the other on PC1 or PC2 | **BOTH** |
| 3 | both hold, only INTER adds | **INSTRUMENTAL** |
| 4 | both hold, only COMMON adds | **STRUCTURAL** |
| 5 | both hold, neither adds (collinear) | **INSEPARABLE** |
| 6 | only INTER holds | **INSTRUMENTAL** |
| 7 | only COMMON holds | **STRUCTURAL** |

- *Sign and pairing* are reported, not gated: PC1 with the x components and PC2 with the y
  components, with the cross terms beside them.
- *Magnitude against significance:* a BY-significant R² below 0.05 does not hold.
- *Untrained reference:* the same regressions onto each untrained draw's own PC1/PC2. This is
  descriptive: whether a random ViT's top axes also track the candidate.

**Causal arm.** This is secondary, with its own state.
- 200 of the 2,000 galaxies are re-embedded with M after shifting **the g band alone** by +0.5
  and +1 px, in x and separately in y (Fourier). r and i are untouched.
- The readout is the mean change in PC1 and PC2, in A-sd units, with a paired SE.
- **RESPONDS** iff at 1 px both of these hold:
  - the paired component (x → PC1, y → PC2) moves by |Δ| ≥ 0.10 sd;
  - it moves by > 1.96 SE.
- **RESPONDS ON ONE AXIS** if only one axis does. **DOES NOT RESPOND** otherwise.
- A response is a causal instrumental reading, whatever the regression says. A non-response
  alongside a holding INTER means the encoder carries something correlated with band offsets,
  not the offsets themselves.

**D28 (`aa3a_pose.py --planted`, run before hashing, on the median definition):**

| check | result |
|---|---|
| synthetic PC from INTER + noise (identical `test` → `state`) | **INSTRUMENTAL** (R² INTER 0.30 / 0.28, COMMON ≈ 0) |
| synthetic PC from COMMON + noise | **STRUCTURAL** (COMMON 0.29 / 0.29, INTER ≈ 0) |
| synthetic from both | **BOTH** (each ≈ 0.13–0.15, joint 0.30 / 0.29) |
| pure noise | **NEITHER** (all ≤ 0) |
| fidelity: the g band shifted +0.5 px in x on 100 stamps | Δ(c_g − c_r) = (+0.35, −0.01) px: attenuated to 0.71 by aperture truncation, on the right axis. ΔCOMMON = (0.00, 0.00) |
| fidelity at +1 px (scratch check) | Δ(c_g − c_r).x +0.71; Δ(core_g − core_r).x +1.0; ΔCOMMON 0.00 |
| causal readout control: rot180 on 200 stamps | PC1, PC2 flip (r −0.93, −0.92). The re-embedded baseline equals the bank (r 1.00) |

- **The first planted run found a leak, fixed before hashing.** With COMMON as the band *mean*,
  a g-only +0.5 px shift moved COMMON by +0.17 px, so a purely instrumental cause would have
  partly read as structural. The band median cannot be moved by one band, and now it isn't.
- The measurement attenuates a true band shift to about 0.7. Direction and axis are preserved,
  and a regression on rank-normal candidates is insensitive to scale.

*AA3a pre-registration ends here: the AA3a section above (88 lines from "## AA3a"), SHA-1 `3e349d9dea9bd2aec864d5ed67baf292e354c75e`.*

### Result (`out/aa3a_pose.json`)

**INSTRUMENTAL.** It is also causal: the causal arm reads **RESPONDS**.

| candidate | R² PC1 | R² PC2 | p (1,000 perms) | holds |
|---|---|---|---|---|
| **INTER** (offsets between bands) | **0.833** | **0.840** | 0.001, 0.001 | yes |
| COMMON (band-median core − centroid) | −0.004 | −0.003 | 0.85, 0.64 | no |
| joint | 0.834 | 0.840 | | COMMON adds nothing |
| untrained draws, their own PC1/PC2 on INTER | ≤ −0.01 | ≤ −0.01 | | |

**Pairing** (Spearman).
- PC1 ~ x of (c_r − c_i): **−0.86**. PC1 ~ x of (c_g − c_r): +0.57.
- PC2 ~ y of (c_r − c_i): +0.85. PC2 ~ y of (c_g − c_r): −0.59.
- The core-based offsets are the same.
- The cross terms are |ρ| ≤ 0.13.
- The signs say that PC1/PC2 read **where the r band sits relative to g and i**.

**Causal arm.** 200 galaxies were re-embedded with M after shifting the g band alone.

| shift (g only) | ΔPC1 (sd) | ΔPC2 (sd) |
|---|---|---|
| +0.5 px x | **+0.343** ± 0.007 | −0.023 |
| +1 px x | **+0.668** ± 0.014 | −0.041 |
| +0.5 px y | −0.029 | **−0.344** ± 0.007 |
| +1 px y | −0.051 | **−0.667** ± 0.013 |

The response is linear in the shift, on the paired component only, with the sign the regression
predicts.

**What PC1/PC2 are.** The encoder's two largest components, 37% of M's variance, encode the
sub-pixel registration of the three bands against each other.
- The typical offset between bands is 0.20–0.22 px per component (0.29 px before the measurement's
  0.7 attenuation).
- That is the size expected from each band being placed independently to the nearest whole pixel.

**The likely cause is in our own cutout, not the sky.**
- `artifacts/sciserver_cut.py:81` cuts each band with `Cutout2D` against **that band's own WCS**.
  `Cutout2D` does not resample: it snaps the stamp to the nearest whole pixel on that band's grid.
- SDSS's g, r and i frames are separate CCDs in the drift scan, on different pixel grids. So each
  band's stamp puts the galaxy at its own sub-pixel position, off by up to ±0.5 px.
- This is a by-product of the **no-rebin** invariant (`docs/spec/data.md:80`): no resampling
  means no sub-pixel alignment either.
- It explains everything X found:
  - **Invariant to whole-stamp shifts:** those move all bands together.
  - **Frame means explain nothing:** the residual is per object, not per frame.
  - **The same in smooth and spiral galaxies**, and **absent in untrained encoders**.
  - **Correlated with the r-band core's position** about the stamp centre, which is the r band's
    own rounding residual.
- **Confirming step, not run:** one CasJobs query for `colc`/`rowc` in g, r and i on these 2,000
  galaxies. The prediction is that frac(colc_g) − frac(colc_r) predicts c_g − c_r at |ρ| ≫ 0.5.
  It needs the SciServer token, which lives only in `artifacts/`.

**Consequences.**
- **A22 is resolved.** The image-plane vector is band misregistration introduced by the cutout.
  It is not morphology and not handedness.
- **For the rental: a retrain on registered bands**, not only an augmented one.
  - Dihedral augmentation (B4) would keep the code, because a rotated misregistration is still a
    misregistration.
  - Registering the bands would remove it at the source. The options are a Fourier sub-pixel shift
    of g and i onto r's grid, which preserves power below Nyquist (the fidelity test's 0.11 was
    for rebinning, not a sub-pixel shift), or recording the per-band offsets and correcting them
    at load.
  - That is a change to a frozen data invariant, so it needs a D-entry. **Flagged, not made.**
- AA3b now asks whether this instrumental code costs the morphology readout anything.

## AA1 — the label-efficiency curve

### Pre-registration

**Protocol** (`aa1_label_efficiency.py`).
- The ladder's linear probe: `probe_auc`, a standardised L2 logistic at the pipeline's C, on the
  raw embedding matrices.
- Encoders: M and the three untrained draws.
- Features: all 37 answers in the **full** population (readout: no vote floor, D8 superseded).
  Labels are the pipeline's binarisation.
- Training pool: the eligible part of A (40,000).
- Training sizes: n ∈ {100, 300, 1k, 3k, 10k, 30k, full}, sizes at or above the eligible pool
  dropped.
- Subsampling: stratified on the label, with at least one of each class. Five subsets per n,
  seeds 0–4. **The same subsets serve every encoder.** Full is fitted once.
- Test: every eligible B galaxy (34,829), fixed across n, subsets and encoders.
- An n that cannot hold both classes is recorded as *below class floor*.

**Reported per answer:**
- AUC(n) for M, and for untrained (the mean of the three draws);
- the paired margin M − untrained per subset, its mean and SE;
- **n90 and n95**, the smallest n at which M's mean AUC reaches 90% or 95% of its own full-data
  AUC.
  - *Literal*, as the brief words it: AUC(n) ≥ q · AUC_full.
  - *Above chance*: AUC(n) − 0.5 ≥ q · (AUC_full − 0.5).
  - The literal form is loose, because chance alone is 0.5 / AUC_full of the full score (59% at
    0.85). **The above-chance form is the one read.** Both are reported.

**The prediction (pre-registered): M's margin over untrained is largest at small n.**
Per answer, the first match applies (D27). "Small" is n ≤ 1k; "large" is n > 1k, including full.

| # | condition | state |
|---|---|---|
| 1 | fewer than 3 evaluable n, or none on one side of 1k | **INSUFFICIENT** |
| 2 | no n where the margin's lower bound (mean − 1.96 SE) > 0 | **NOT ABOVE UNTRAINED** |
| 3 | max small-n margin − max large-n margin > 2·√(SE_s² + SE_l²) | **LARGEST AT SMALL N** |
| 4 | the reverse, by the same amount | **LARGEST AT LARGE N** |
| 5 | otherwise | **FLAT** |

- Full is a single fit, so its SE is 0. It enters only through the small-n side's SE. That
  makes rules 3 and 4 slightly easier to meet when the large-n peak is at full. This is declared.
- **Catalogue:**
  - **SUPPORTED** if more than half of the answers in states 3–5 read LARGEST AT SMALL N;
  - **CONTRADICTED** if more than half read LARGEST AT LARGE N;
  - **MIXED** otherwise;
  - **NOT EVALUABLE** if no answer reaches states 3–5.
- There is no multiplicity correction. The states are descriptive magnitudes with noise bars, and
  no p enters.

**D28 (`aa1_label_efficiency.py --planted`, run before hashing):**

| check | result |
|---|---|
| state logic, planted margin curves | small-peaked → **LARGEST AT SMALL N**; rising → **LARGEST AT LARGE N**; constant → **FLAT**; ≈ 0 → **NOT ABOVE UNTRAINED**; two points → **INSUFFICIENT** |
| catalogue logic | 3 small + 1 flat → SUPPORTED; 3 large + 1 flat → CONTRADICTED; one of each → MIXED; none evaluable → NOT EVALUABLE |
| a planted label (linear in M's embedding + noise, 30% positive) through the identical fits | M 0.72 → 0.82 and untrained 0.55 → 0.67 over n 100 → 40k. Margin 0.165, 0.183, 0.197, 0.191, 0.173, 0.160, 0.158. n95 above chance 10k (literal 3k). State FLAT: a peak at 1k beats 3k by 0.006, inside 2 SE (0.007) |

- Every state is reachable.
- The planted label shows the fits and the paired margin behave as intended.
- It also shows how strict the noise rule is: a real but small peak (0.006) reads FLAT, not SMALL.

*AA1 pre-registration ends here: the AA1 section above (60 lines from "## AA1"), SHA-1 `d458ab1e7d8079cc4b540f5b594c50bdb8855cb0`.*

### Result (`out/aa1_label_efficiency.json`)

**The prediction is CONTRADICTED.** The states are:

| state | answers |
|---|---|
| LARGEST AT LARGE N | 24 |
| FLAT | 9 |
| LARGEST AT SMALL N | 4 |

The four small-n answers are smooth, featured, spiral and no-spiral, the concepts M reads best.

**The shape, described post hoc (not pre-registered).**
- M's margin over untrained is a **hump, not a decay**. It rises from n = 100, peaks in the low
  thousands, and eases slightly towards full.
- Where the margin peaks:

  | peak n | 300 | 1k | 3k | 10k | 30k | full |
  |---|---|---|---|---|---|---|
  | answers | 2 | 8 | 9 | 9 | 2 | 7 |

- At n = 100, M's probe barely beats the untrained one on most answers: the median margin is
  +0.03. **M needs a few hundred labels before its structure shows.** After that, the untrained
  encoder closes the gap only slowly as labels accumulate.
- The prediction assumed a strong representation is readable from very few labels. That holds
  only for smooth/featured and spiral:
  - smooth/featured has a margin of +0.10 at n = 100, peaking at +0.15 at 1k;
  - spiral has +0.09 at 100 and +0.14 at 300.

**Label efficiency: how many labels reach full-data performance.** Figures are above chance,
the read form.

| | n = 3k | 10k | 30k | only at full (the pool is < 30k) |
|---|---|---|---|---|
| **90% of M's full-data AUC** | 5 | 16 | 9 | 7 |
| **95%** | — | 7 | 18 | 12 |

- **About 10k labels reach 90% for most answers, and about 30k reach 95%.**
- The literal form (AUC ≥ q · AUC_full) reads 3–10× earlier, because chance already supplies most
  of it. It is in the record, not read.

| answer | n = 100 | 1k | 10k | full | untrained full |
|---|---|---|---|---|---|
| featured / disk | 0.709 | 0.829 | 0.878 | **0.884** | 0.792 |
| spiral | 0.668 | 0.762 | 0.808 | **0.816** | 0.732 |
| edge-on | 0.565 | 0.689 | 0.781 | **0.800** | 0.631 |
| cigar-shaped | 0.605 | 0.736 | 0.829 | **0.846** | 0.656 |
| bar | 0.519 | 0.584 | 0.645 | **0.658** | 0.574 |
| winding: medium | 0.502 | 0.515 | 0.518 | **0.523** | — |

**Reading.**
- As a *few-label* classifier, M is modest. 1k labels give about 0.83 on smooth/featured and
  0.76 on spiral, and most answers sit under 0.70 at n ≤ 1k.
- Its advantage over the architecture alone is biggest at **1k–10k labels**, the regime of a
  small labelling campaign. It is not biggest at one-shot scale.
- The fine-scale answers (winding, arm count) never get far from chance at any n. That matches
  A19/A20 and B3.
- **For the rental.** The comparison that decides the practical claim is a supervised-from-scratch
  ViT and a fine-tuned M at the same n. Without them, "label-efficient" is a claim against an
  untrained encoder only.

## AA3b — does the pose code cost anything?

### Pre-registration

**Construction** (`aa3b_pose_average.py`).
- Every union galaxy (74,829) is embedded by M under the 8 elements of the dihedral group D4:
  rotations of 0°, 90°, 180° and 270°, each with and without a left–right mirror.
- The eight are averaged per galaxy.
- A 180° rotation sends AA3a's band-offset vector to its negative, so the average should cancel
  it.
- **Mirror-averaging also removes handedness.** That is accepted: Y found handedness is not
  linearly decodable (AUC 0.50).
- Only M is averaged. The untrained references are the recorded, un-averaged ones. This is
  declared: the comparison asks what the pose code costs *M*.

**Collapse check (a precondition, not a result).** On A, take the variance of the averaged
embedding along M's own PC1 and PC2 (V3's basis), as a share of M's.
- **Collapsed** iff both shares are ≤ 0.10.
- If not, every readout below is **NOT INTERPRETABLE**: the average failed to remove the pose
  code.

**Readouts, on the averaged M against M as trained:**
1. **Existence.**
   - The ladder's probe (standardised L2 logistic, pipeline C), all 37 answers, full population,
     A → B.
   - ΔAUC = AUC(avg) − AUC(orig), with a paired bootstrap 95% CI (2,000 draws, same test
     galaxies).
   - **IMPROVED**: Δ ≥ +0.01 and the CI excludes 0. **WORSENED**: Δ ≤ −0.01 and the CI excludes 0.
     **UNCHANGED** otherwise.
   - The 0.01 floor sits about 5 paired SE from zero at these n. By-chance calls are not
     expected, so no further multiplicity correction is applied.
2. **Salience.**
   - V3's held-out recovery test, re-run on the averaged M: one of 50 PCs aligned per answer on A
     by |Spearman| with the vote fraction, frozen, AUC on B against the probe's.
   - Permutation p (10,000 draws), BY over the 33 gated answers.
   - The verdicts follow V3's rule against the **recorded** untrained recoveries.
   - A verdict moves **up** or **down** by rank: NO AXIS / INVERTED < NOT ABOVE UNTRAINED < WEAK
     < PARTIAL < ENCODER. The nuisance-proxy suffix is ignored.
3. **Winding.**
   - Y3's T4b: U3's winding axis against Hayes pitch, visibility controlled, on B (the Y1 join).
   - It is recomputed on the original M and on the averaged M under the same rule (existence at a
     raw p < 0.05). The two states are compared. This one is descriptive; it doesn't enter the
     catalogue state.

**Catalogue state.** *Better* counts readout moves in the averaged M's favour: existence IMPROVED
answers plus salience verdicts that move up. *Worse* counts WORSENED plus moves down. An answer can
count once per readout. Winding does not count. The first match applies (D27):

| # | condition | state |
|---|---|---|
| 1 | collapse check fails | **NOT INTERPRETABLE** |
| 2 | better ≥ 2 **and** worse ≥ 2 | **MIXED** |
| 3 | ≥ 2 better | **POSE WAS COSTING** |
| 4 | ≥ 2 worse | **POSE WAS HELPING** (the surprise: pose carried morphology-relevant signal) |
| 5 | otherwise | **HARMLESS** |

**D28 (`aa3b_pose_average.py --planted`, run before hashing):**

`out/aa3b_planted.json`. Every state is reachable and the plants fire:

| check | read | required | |
|---|---|---|---|
| identity element r0 vs M's recorded embedding | r = 0.99999999 | ≈ 1 (the bank is M) | fires |
| collapse: averaged variance on M's PC1, PC2 (share of M's) | **0.022, 0.030** | both ≤ 0.10 | fires (PC3–10 keep 0.41–0.83) |
| planted **pose** label (M's PC1 + noise), existence path | AUC 0.948 → **0.541**, Δ −0.407 [−0.413, −0.401] | WORSENED | fires |
| planted **invariant** label (averaged M's PC3 + noise) | 0.934 → 0.937, Δ +0.002 [+0.002, +0.003] | not WORSENED | UNCHANGED |
| state logic on planted change lists | COSTING, HELPING, MIXED, HARMLESS; salience-up alone → COSTING; no collapse → NOT INTERPRETABLE | all five states reachable | fires |

- **Resolution.** The salience permutation null has 10,000 draws, so its floor is 1 × 10⁻⁴.
  BY's rank-1 threshold over 33 answers is 0.05 / (33 · 4.09) ≈ 3.7 × 10⁻⁴, which clears the floor.
- The invariant plant's Δ is inside the 0.01 materiality floor even though its CI excludes 0. That
  is what the floor is for.

*AA3b pre-registration ends here: the AA3b section above (72 lines from "## AA3b"), SHA-1 `19d1ebb34434346e81c89c1e4de8260cf8571334`.*

### Result (`out/aa3b_pose_average.json`)

**Pre-registered state: MIXED.** Better 39 (33 existence IMPROVED + 6 salience up), worse 6 (0
WORSENED + 6 salience down). The collapse check passed (PC1 0.022, PC2 0.030).

**Existence.** 33 of 37 IMPROVED, 4 UNCHANGED (star/artifact, spiral, winding medium, 4 arms), none
WORSENED. ΔAUC +0.009 to +0.051, median +0.021. Largest: edge-on +0.05, cigar-shaped +0.05, odd
+0.04, dust lane +0.04. Smooth/featured +0.013.

**Salience** (V3's held-out protocol, against the recorded untrained recoveries).

| moved | answers | recovery, orig → averaged |
|---|---|---|
| **up** | smooth (WEAK → PARTIAL), featured (NOT ABOVE UNTRAINED → PARTIAL), spiral and no-spiral (WEAK → PARTIAL), bar and no-bar (NO AXIS → NOT ABOVE UNTRAINED) | smooth 0.44 → 0.71, featured 0.32 → 0.71, bar 0.04 → 0.37 |
| **down** | bulge none / just noticeable (WEAK → NOT ABOVE), bulge obvious (PARTIAL → NOT ABOVE), odd "other" (PARTIAL → NOT ABOVE), bulge rounded (WEAK → NOT ABOVE), 2 arms (WEAK → INVERTED) | bulge obvious 0.57 → 0.49, 2 arms 0.26 → −0.12 |

- The aligned component moves up the ranking: index 3 on M, 1–2 on the averaged embedding
  (`pc` in the record). The smooth/featured/spiral component aligns more cleanly after averaging.
  The bulge answers shared that component on M and lose alignment on the averaged one.
- No human concept becomes ENCODER. A21 ("decodable, not salient") stands.

**Winding** (Y3 T4b, recomputed both ways): orig partial −0.077, retention 0.42; averaged −0.081,
retention 0.45. **MOSTLY VISIBILITY** both ways. The pose code was not what hid winding.

### Exploratory, post hoc: pose removal or view ensembling? (`out/aa3b_ensemble.json`)

Not in the hashed pre-registration. Run because averaging 8 views is also test-time augmentation:
it ensembles away view-dependent noise whether or not a pose code exists, and the design above
cannot tell the two apart. Four arms, each removing the pose code, with 0, 2, 4 and 8 views:

| arm | PC1, PC2 share left | median ΔAUC (range) | IMPROVED / WORSENED |
|---|---|---|---|
| M with PC1/PC2 projected out (0 views) | 0.00, 0.00 | **−0.0007** (−0.011, +0.016) | **0** / 0 |
| mean of identity and 180° (2 views) | 0.026, 0.035 | +0.013 (+0.001, +0.046) | 20 / 0 |
| mean of the four rotations (4 views) | 0.023, 0.031 | +0.019 (+0.008, +0.046) | 29 / 0 |
| D4 (8 views) | 0.022, 0.030 | +0.021 (+0.009, +0.051) | 33 / 0 |

- **Removing the pose code alone costs and buys nothing.** The linear probe already ignores it.
- The gain grows with the number of views while the pose removal is the same in every arm. **It is
  test-time ensembling, not pose.** Edge-on goes +0.001 → +0.027 → +0.044 → +0.049 across the arms.
- **Read with that: the pose code is harmless to the linear readout.** The pre-registered MIXED
  stands as the record. Its "better" side is the ensemble's doing, and HARMLESS is the reading of
  the pose code itself. It is labelled exploratory beside the record, per D27, and does not replace it.
- Salience was not rerun on the projected arm, so the salience moves are not separated in the same
  way. Projecting out two components cannot raise another component's alignment by itself, so
  the salience moves are most plausibly ensembling too. That is not measured.
- **For the rental.** Pose is not a reason to retrain. Band registration (AA3a) is still right as
  data hygiene: 37% of the variance spent on a cutout artefact is capacity not spent on galaxies.
  But this says it will not move the linear catalogue. A free +0.02 is available from
  test-time D4 averaging on any encoder, including the baselines. If it is used, it must be applied
  to every encoder alike.
