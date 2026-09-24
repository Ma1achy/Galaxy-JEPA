# Brief Z — three small checks

Status: complete. Z1 DISAGREE; Z2 no verdict changes; Z3 BOTH. Each section is hashed (SHA-1 over the section text up to
its footer) before its test runs.

## Z1 — does Hart (SpArcFiRe) agree with Yu & Ho (2DFFT)?

### Pre-registration

**Question.** Of the two PAnDa pitch-angle references, is either a usable independent check on the
other? If they agree, Y's divergence points to the Hayes table as the weak link. If they don't,
pitch angle has no reference that two methods support.

**Disclosure, made before anything is hashed.** This is not blind. Y2 already computed this
comparison (`y_findings.md` §Y2, Yu & Ho vs Hart): **n = 34 pairs, ρ = −0.252, p = 0.15, median
|Δ| 7.1°**. Those 34 are the whole PAnDa Hart ∩ Yu & Ho overlap within 3″, and all 34 are in our
230k. The threshold and the states below were chosen from the literature, not from that number.
The D28 plants below show the states can be reached. But a reader should know the likely outcome
was already visible.

**Pairs.** PAnDa entries for "Hart et al 2017" and "Si-Yue Yu and Luis C. Ho 2020"
(`Y.panda`: duplicated names are dropped). Each Hart entry is matched to its nearest Yu & Ho
entry, and the pair is kept if the separation is under 3″. Statistic: Spearman ρ on `PA_degrees`.

**Threshold: ρ = 0.5, set from the published scatter of the methods.**
- Hewitt & Treuthardt 2020 (MNRAS 493, 3854) measured the error of each method against toy galaxies
  with known pitch. Mean |Δ| from the truth: SpArcFiRe 7.3–10.6°, 2DFFT (p2dfft:auto) 1.4–1.8°.
- Hart et al. 2017 compared SpArcFiRe with GZ2-based pitch and found rs = 0.30 with ±7° rms.
- Take the true pitch sd among these spirals as ≈ 7°, the width of the published distributions.
  Two methods whose errors are independent then correlate at ρ ≈ σ² / √((σ² + e₁²)(σ² + e₂²)).
  With e₁ = 7–10.6° and e₂ = 1.5–2.5°, that gives **ρ ≈ 0.52–0.68**.
- So 0.5 is the bottom of what two methods working as published should reach. Below it, at least
  one method is worse than its published error on these galaxies, or they measure different things.
- This is the same bar as Y2's CONSISTENT (ρ ≥ 0.5).

**Uncertainty.** Two 95% intervals for ρ:
- Fisher-z with Bonett & Wright's variance, (1 + ρ²/2)/(n − 3).
- A 2,000-draw percentile bootstrap (`V2.boot_ci`).

A verdict needs both intervals on the same side of 0.5. The outer envelope of the two is used.

**States — the first match applies (D27):**

| # | condition | state |
|---|---|---|
| 1 | n < 10 pairs | **INSUFFICIENT OVERLAP** |
| 2 | upper bound of both CIs < 0.5 | **DISAGREE** |
| 3 | lower bound of both CIs ≥ 0.5 | **AGREE** |
| 4 | otherwise (the interval straddles 0.5) | **INSUFFICIENT OVERLAP**, reported with the n that would resolve it |

- **Sign flip:** a negative ρ falls into DISAGREE (state 2), and its sign is reported beside it.
  For this question a negative and a null agreement have the same consequence: the references
  don't support each other. So no separate state is needed.
- **Significance against magnitude:** there is no significance test here. Only the interval's
  position relative to 0.5 counts. So a ρ that is "significantly positive" but whose CI straddles
  0.5 is INSUFFICIENT, not AGREE.

**Sensitivity: V1 visibility controlled.**
- The pairs are matched to our metadata within 3″. V1's composite is computed on the pairs that
  have it, via `Y.visibility`.
- The partial Spearman given visibility gets its own Fisher interval (n − 1) and its own state
  under the same table.
- The primary verdict is the raw one. If the partial's state differs, the verdict is reported as
  **SENSITIVE TO VISIBILITY** beside it.

**What each outcome does:**
- **AGREE** → the Hayes table is the weak link. Propose (do not start) our own SpArcFiRe run on
  our stamps, with a reliability filter.
- **DISAGREE** → record pitch angle as not a usable independent reference. State this as a
  limitation on A19 and A20 in `framing_a_claims.md`.
- **INSUFFICIENT OVERLAP** → state the n that would resolve it: the smallest n at which a true ρ
  of 0.65 (the middle of the expected range) clears 0.5 with 80% power.

**D28: planted cases** (`z_checks.py --planted`, run before this is hashed):
- 500 draws at the observed n = 34 through the identical `compare` → `z1_state` path.
- Gaussian copula at the Pearson r that gives the target Spearman ρ.

| true ρ | AGREE | DISAGREE | INSUFFICIENT | n for 80% power |
|---|---|---|---|---|
| 0.90 | 496 | 0 | 4 | 16 |
| 0.65 | 85 | 0 | 415 | 190 |
| 0.00 | 0 | 421 | 79 | 30 |
| −0.25 | 0 | 497 | 3 | 16 |

- AGREE and DISAGREE both fire, and INSUFFICIENT is reachable.
- There is no permutation null to saturate: the intervals are analytic or bootstrap, and a true
  0.9 clears the bar at n = 34.
- The power is stated plainly. At n = 34, a pair of methods working as published (ρ ≈ 0.65) is
  called AGREE only 17% of the time. **A DISAGREE here is informative; an INSUFFICIENT is the
  expected reading for two methods that do agree.**
- Resolving the typical case takes about 190 pairs.

*Z1 pre-registration ends here: the Z1 section above (87 lines from "## Z1"), SHA-1 `daf50d1a0eb14af3df2b106577dd5735ee1c5e38`.*

### Result (`out/z1_agreement.json`)

**DISAGREE.** Visibility does not change it.

| | n | ρ | Fisher 95% | bootstrap 95% | median \|Δ\| |
|---|---|---|---|---|---|
| Hart vs Yu & Ho, raw (primary) | 34 | **−0.252** | [−0.548, +0.099] | [−0.552, +0.099] | 7.1° |
| partial on V1 visibility | 34 | −0.262 | [−0.560, +0.095] | — | |

- **Pairs.**
  - All 34 are in our corpus, and all have visibility.
  - Every pair matches at 0.0″: PAnDa gives both entries the same coordinates.
  - The two samples have similar spreads (sd 6.6° and 6.0°) and medians (16.6° and 16.1°).
    They agree on where pitch sits and not on which galaxy has more of it.
- **Upper bounds.** Both intervals' upper bounds (≈ +0.10) are far below 0.5, the bottom of the
  range two methods working as published should reach. The disagreement is not a matter of low
  power: at this n, a true ρ of 0 reads DISAGREE 84% of the time, and a true ρ of 0.65 never does.
- **Sign.** Negative, but the interval includes 0. Read it as no agreement, not as an
  anti-correlation.
- **What it means.** Pitch angle has **no independent reference that two methods support** on
  these galaxies. Earlier findings in context:
  - Y2 found Hayes vs Hart only DIVERGENT (0.38): same algorithm, different runs.
  - Y2 found Hayes vs Yu & Ho BROKEN.
  - Z1 now finds the two PAnDa references don't support each other either.
  - So SpArcFiRe's pitch agrees only with itself, and weakly.
  - And 2DFFT agrees with nothing we can check it against, at n ≤ 550.
- **Caveat.** 34 galaxies is the whole overlap. It is one small sample, not a population
  estimate. But what it can resolve, it resolves.
- **For the tool.** The pre-registered consequence is a limitation, not a proposal:
  - The AGREE branch (our own SpArcFiRe run, with a reliability filter) is **not** triggered.
  - Running SpArcFiRe on our stamps would give a third reading from the same algorithm family as
    Hart and Hayes. Nothing here says that family is the right one to trust.
  - Before any pitch tool is built, the missing thing is a **cross-method check**, not more of
    one method: 2DFFT or a hand-measured sample on galaxies we also measure.
- **Recorded as a limitation on A19 and A20** in `framing_a_claims.md`.

## Z2 — reconciling the Hayes match count

### Counts (measured before any rerun)

Every Hayes ID column was joined as a string against the matching ID in our 230k metadata.

| Hayes column | our column | matches |
|---|---|---|
| `name` (DR8+ objID) | `object_id` | **37,381** (Y1's join) |
| `OBJID` (DR7 objID) | `dr7objid` | **46,882** |
| `OBJID2`, `objId`, `objID` | `dr7objid` | 46,882: identical to `OBJID` row by row |
| `nnObjID`, `nnFarObjID` | — | 34,206 / 34,816: **not valid joins** (see below) |

- **`nnObjID` and `nnFarObjID` are neighbours.** They are the IDs of other galaxies near each
  Hayes target, so matching on them lands on the wrong galaxy.
- **Overlap.** `name` ∩ `OBJID` = 37,381. Nothing matches on `name` alone, and **9,501** match on
  `OBJID` alone. All 37,381 shared matches point to the same galaxy both ways.
- **Why the 9,501 were missed.** Decoding the bits of their `name` gives a different run, field
  and object number from our `object_id`. The same galaxy was detected in an overlapping SDSS
  run: Hayes' DR8+ `name` is that run's detection, while our `object_id` is the primary. The DR7
  `OBJID` names the same primary we do.
- **ChatGPT's 49,182** is Hayes `OBJID` ∩ the GZ2 Hart16 table: original 41,361, extra 6,032,
  stripe82 1,789. Of these:
  - 46,882 are in our corpus;
  - 2,300 are not (original 1,693, stripe82 323, extra 284). They are among the 12,643 GZ2
    galaxies our corpus lacks.
- Both numbers are right. They count different intersections.

### Pre-registration: the rerun on the larger set

**What reruns.**
- Y3's primary tests go through the identical code (`y3_science.py --y3 --z2`). The only change
  is that the Hayes columns are joined `OBJID` → `dr7objid` (`Y.HAYES_KEY`).
- Unchanged:
  - the states, thresholds, BY families and seeds;
  - the V1 visibility control;
  - the shared null;
  - the extra embedding bank (it serves the PAnDa replication legs, which this join doesn't touch).
- Tests that use no Hayes column (the Hart and Yu & Ho legs of T1, T3 and T4a, and T2's
  replication) read the same rows as before. They move only through the shared BY family.

**What counts as a change (D27).**
- A verdict **changes** iff its state name differs from `y_findings.md` §Y3.
- Movement in ρ, n or margin with the same state name is reported as **SAME STATE**, with the
  numbers beside it.
- A state that becomes reachable only on the larger n (e.g. INSUFFICIENT → a verdict) counts as
  a change and is labelled **RESOLVED BY N**.
- The verdicts compared:
  - T1 (hayes, hart, yuho);
  - T2 (Hayes 2×2);
  - T2 replication (hart and yuho × transfer and CV);
  - T3 (hayes, hart, yuho);
  - T4a (hayes, hart, yuho);
  - T4b (axis).
- T5 is exploratory, and is reported only if it moves by more than its own CI.

**D28 on the new rows** (`y3_science.py --planted3 --z2`, run before this is hashed):
Y3's planted states were rerun on the 46,882-row Hayes table:

| test | planted | state (Z rule) | state (Y3 rule) |
|---|---|---|---|
| T1 | agree / none / contrary | AGREE / NO AGREEMENT / CONTRARY | same |
| T2 | both | BOTH (A_m 0.85, A_v 0.90) | BOTH |
| T2 | shared | **MEASUREMENT BEYOND VOTES** (A_m 0.154 vs bar 0.150) | MEASUREMENT BEYOND VOTES |
| T2 | neither | NEITHER | NEITHER |
| T3 | wider / narrower / null | WIDER / NARROWER / NOT WIDER | same |
| T4b | beyond / visibility / none | WINDING BEYOND VISIBILITY / MOSTLY VISIBILITY / NONE | — |
| T4b | reversed (new) | REVERSED UNDER CONTROL (partial −0.84) | — |

**The first planted run failed twice. Both failures were fixed before this section was hashed.**
The fixes apply to Z and to future runs. They are not retroactive to Y3 (D27). Y3's recorded
verdicts are recomputed under Y3's own rule beside the new one.

1. **T4b: REVERSED UNDER CONTROL fired on noise.**
   - What happened: the pure-visibility plant has a partial of about 0 (−0.025 on these rows). A
     sign test read that as a reversal, because REVERSED came before MOSTLY VISIBILITY in the
     precedence. On Y1's rows the same plant happened to land at +0.03.
   - Amended rule: REVERSED now requires an opposite sign **and** a permutation p < 0.05 on the
     partial **and** |partial| ≥ 0.10. Otherwise the test falls through to retention.
   - A REVERSED plant was added (partial −0.84). It fires.
   - Y3's T4b (MOSTLY VISIBILITY, retention 0.42, same sign) reads the same under both rules.

2. **T2: the shared-quantity null was biased low.**
   - Measurement (`shared_rate*.py`, scratchpad): 100 plants per direction, with fresh noise,
     each against its own matched null. Under Y3's null, a planted shared quantity fired a leg
     often:

     | Y3's null (ridge latent) | direction 1 | direction 2 | direction 3 |
     |---|---|---|---|
     | 50 draws, 95th, pooled over directions | 17% | | |
     | 200 draws, 97.5th | 4% | 22% | 10% |
     | 200 draws, 95th | 8% | 32% | 20% |

   - Cause: Y3's null puts the shared latent on M's ridge fit to the target, which sits in the
     easily decoded subspace. Its copies are decoded more cleanly than a real shared quantity's,
     so they leave less for the partial.
   - Amended null: each of **200 draws** puts the shared latent on a **fresh random direction**
     of M's embedding. This is the shared hypothesis with the direction unknown. Each leg must
     clear the null's **97.5th** (Bonferroni over the two legs).
   - Calibration: 100 fresh random-direction shared plants against this null.

     | null | either-leg rate | per leg |
     |---|---|---|
     | random-direction, 97.5th | **6%** | 2%, 4% |
     | Y3's ridge null, same 200 draws, 97.5th | 9% | |
     | Y3's ridge null, 95th | 14% | |

   - **The D28 evidence for T2 is the rate, not the single draw.** A shared quantity reads as a
     one-sided verdict about 6% of the time. The named shared plant above is one of those draws
     (A_m 0.154 against 0.150). Its seed is Y3's, unchanged. Re-seeding it to get a pass would be
     choosing the outcome.
   - Consequence: a leg within about 0.02 of the bar is weak evidence and is reported as such.
   - Every T2 verdict is reported under both rules: the Z rule (primary) and Y3's (ridge null,
     95th).

*Z2 pre-registration ends here: the Z2 section above (114 lines from "## Z2"), SHA-1 `91bb1542375d876ba6b10cfc448af6ef7e3d21fd`.*

### Results (`out/y3_pitch_z2.json`)

**No verdict changes.** Every Y3 state holds on the 46,882-row join, under the Z rule and under
Y3's rule.

| test | Y3 (37,381 join) | Z2 (46,882 join) | change |
|---|---|---|---|
| T1 hayes | AGREE: n 16,274, ρ −0.347, partial −0.307 | AGREE: n 20,278, ρ −0.352, partial −0.311 | SAME STATE |
| T1 hart / yuho | AGREE / NO AGREEMENT | identical rows | SAME STATE |
| T2 Hayes 2×2 | VOTES BEYOND MEASUREMENT: 2,721 → 2,491 | VOTES BEYOND MEASUREMENT (Y3 rule: same): 3,404 → 3,065 | SAME STATE |
| — decode ρ, pitch / w_avg | 0.229 / 0.372 | 0.228 / 0.399 | |
| — A_m (bar, Z rule) | 0.097 (Y3 null 95th 0.070) | 0.094 (random-direction 97.5th 0.068); **below the 0.10 floor** | |
| — A_v (bar, Z rule) | 0.273 (0.159) | 0.308 (0.172); untrained ≤ 0.12 | |
| T2 replication (hart, yuho) | transfer NOT / CV REPLICATED; NOT / NOT | same (yuho n 999 → 985) | SAME STATE |
| T3 hayes | NARROWER: D −0.543 | NARROWER: D −0.530, n 6,955 / 9,993 / 2,684 | SAME STATE |
| T3 hart / yuho | NARROWER / INSUFFICIENT | identical | SAME STATE |
| T4a hayes | ORDERED BEYOND VISIBILITY: τ 0.247 → 0.217 | ORDERED BEYOND VISIBILITY: τ 0.251 → 0.219 | SAME STATE |
| T4a hart / yuho | ORDERED BEYOND VISIBILITY / NOT ORDERED | identical | SAME STATE |
| T4b axis | MOSTLY VISIBILITY: retention 0.42 | MOSTLY VISIBILITY: retention 0.46 (raw −0.188, partial −0.086) | SAME STATE |

**Reading.**
- The 9,501 recovered galaxies are the same kind of galaxy as the 37,381. They were missed only
  because they were detected in a different run.
- Adding them moves every statistic by less than its own interval.
- **Y3's findings stand on the full matched set.** The pitch decoder carries nothing beyond the
  votes (A_m under the floor on both joins), and the votes carry something beyond measured pitch.
- **T5 (exploratory) is checked and unchanged.**
  - Handedness AUC: 0.498 → 0.502; untrained 0.49–0.52.
  - Arcs vs voted arm count, partial on visibility: 0.23–0.29 → 0.23–0.29. n rises from 33,992 to
    42,571.

## Z3 — V2's B/T 2×2, rerun with the corrected design

### Pre-registration

**Why rerun.**
- V2's Experiment D (`v_findings.md` §V2) read **BOTH** (A_m 0.36, A_v 0.47) on Simard B/T
  against the GZ2 bulge-prominence average. Each leg controlled the *other target*, which is a
  noisy proxy.
- Y3's D28 check showed that this form reads BOTH on two noisy copies of **one** quantity.
- So V2's verdict cannot tell "each carries something the other lacks" from "both are noisy
  readings of one thing".

**Rows: V2's, unchanged** (`z3_bt.py`, reusing `V2.metadata`, `V2.simard`, `V2.B_AVG`).
- `base` = finite B/T & finite bulge average & t05 reach ≥ 21.
- Train = A (40,000), test = B (34,829), restricted to `base`.
- V2's flag (petroRad_r < 3″ | modelMag_r > 17 | pps > 0.32 | e_bt > 0.1) defines the
  **unflagged** sensitivity block.

**Design: Y3's `two_by_two`, as amended in Z2.**
- A_m = partial(pred_BT, B/T | pred_bavg, visibility). A_v = partial(pred_bavg, bavg | pred_BT,
  visibility).
- Visibility is V1's composite on the analysed rows, as in Y3. V2 had no visibility control, so
  the no-visibility form is reported descriptively, beside V2's original noisy-control form.
- A leg **holds** iff all of the following:
  - it is BY-significant (family below);
  - ρ ≥ 0.10;
  - it is above every untrained draw;
  - it is above the **random-direction shared null's 97.5th** (200 draws, noise matched to the
    observed decode ρ's).
- Y3's rule (ridge-latent null, 95th) is reported beside it, as in Z2.

**States (D27), as `state_2x2`, first match applies:**

| # | condition | state |
|---|---|---|
| 1 | either leg is BY-significant with ρ < 0 | **INVERTED** |
| 2 | both legs hold | **BOTH** |
| 3 | only A_m holds | **MEASUREMENT BEYOND VOTES** |
| 4 | only A_v holds | **VOTES BEYOND MEASUREMENT** |
| 5 | neither leg holds, and both targets are DECODED | **SHARED ONLY** |
| 6 | otherwise | **NEITHER** |

- **Family.** BY at m = 4 (decode_bt, decode_bavg, a_m, a_v), 10,000 permutations. V2 had m = 5:
  its fifth member (2a) isn't rerun here.
- **Unflagged block.** Unadjusted, as V2 read it. If its state differs from the primary's, the
  result is **FLAG-SENSITIVE**.
- **Near-bar leg.** A leg within 0.02 of its shared-null bar is reported as **MARGINAL**, beside
  the state. Z2's calibration puts about 6% of shared quantities over the bar, clustered just
  above it.

**What each outcome does to A18** (`framing_a_claims.md`):
- **BOTH** → A18's "B/T tracked both ways" is restored under the corrected design.
- **MEASUREMENT / VOTES BEYOND** → A18 is restated one way.
- **SHARED ONLY** → A18 becomes "the encoder decodes B/T and bulge prominence, but only as one
  shared quantity".
- **NEITHER / INVERTED** → A18 is withdrawn as stated.

In every case `v_findings.md` §V2 is marked superseded, with the reason.

**D28 on V2's rows** (`z3_bt.py --planted`, run before this section is hashed):

| planted | decode ρ (bt, bavg) | A_m, A_v | bar (97.5th) | state (Z rule) | state (Y3 rule) | V2's form |
|---|---|---|---|---|---|---|
| both | 0.89, 0.87 | 0.88, 0.86 | 0.118, 0.098 | **BOTH** | BOTH | 0.88, 0.86 |
| shared | 0.89, 0.89 | 0.079, 0.064 | 0.109, 0.109 | **SHARED ONLY** | SHARED ONLY | **0.64, 0.65 (reads BOTH)** |
| neither | ≈ 0 | ≈ 0 | 0.029, 0.022 | **NEITHER** | NEITHER | ≈ 0 |

**Calibration.** 100 fresh random-direction shared plants were run on these rows.

| null | either-leg rate | per leg (A_m, A_v) |
|---|---|---|
| random-direction, 97.5th | **12%** | 4%, 8% |
| ridge (Y3's), 97.5th | 10% | |
| random-direction, 95th | 21% | |
| ridge (Y3's), 95th | 24% | |

- The planted shared quantity reads correctly. This is the case V2's form got wrong.
- As a rate, though, the corrected design's **size on these rows is about 12%**: a shared quantity
  reads one-sided about one time in eight. That is above nominal, and it is stated here, not
  tuned away.
- Consequence: a one-sided or BOTH verdict whose legs clear the bar by a wide margin is not a
  size artefact. A leg within 0.02 of the bar is reported as **MARGINAL** and carries no claim on
  its own.

*Z3 pre-registration ends here: the Z3 section above (83 lines from "## Z3"), SHA-1 `23cfc3d67e02190aa589e08ca7d3711d017e3f16`.*

### Result (`out/z3_bt.json`)

**BOTH.** The verdict holds under the corrected design, with wide margins. It is not
flag-sensitive, no leg is marginal, and it is the same under Y3's rule.

| leg | M | untrained | shared-null bar (97.5th; median) | Y3 ridge 95th | V2's form | carries? |
|---|---|---|---|---|---|---|
| **A_m**: B/T beyond the bulge-average *decoder*, visibility controlled | **+0.269** | 0.098–0.122 | 0.082 (0.050) | 0.136 | 0.363 | yes: +0.15 over untrained, 3.3× the bar |
| **A_v**: bulge average beyond the B/T *decoder*, visibility controlled | **+0.388** | 0.249–0.277 | 0.112 (0.076) | 0.199 | 0.472 | yes: +0.11 over untrained, 3.5× the bar |

- **Rows.** V2's primary rows reproduce: n 13,516 base (3,396 flagged), 7,193 → 6,323. The
  no-visibility, noisy-control form gives 0.358 / 0.470, which is V2's 0.358 / 0.470 exactly.
- **Decode.** B/T ρ 0.648 and bulge average 0.701, both DECODED. Untrained reaches ≤ 0.43 and ≤ 0.52.
- **Agreement** between the two targets: 0.717 (0.716 with visibility controlled).
- **Unflagged block** (5,405 → 4,715): BOTH. A_m 0.260 (bar 0.091), A_v 0.349 (bar 0.126).
- **Visibility.** Leaving it out moves nothing: A_m 0.265, A_v 0.388. Neither target is a
  visibility proxy here.
- **What the correction did.**
  - Controlling the other decoder's prediction instead of the other noisy target lowered both
    partials: A_m 0.36 → 0.27, A_v 0.47 → 0.39. That drop is the share V2's form had credited to
    a shared quantity only half removed.
  - What remains clears a shared quantity's reach three times over.
  - The corrected design's measured size on these rows is 12%, but that concerns legs near the
    bar, and these are nowhere near it.
- **The caveat V2 already had now sharpens.** The vote leg's *learned* margin is small.
  - An untrained encoder carries 0.25–0.28 of "votes beyond B/T", against M's 0.39.
  - Most of what the volunteers see beyond the decomposition, a random ViT sees too. Plausibly
    it is the size and brightness cues, since visibility doesn't absorb it.
  - The measurement leg's learned margin (+0.15) is the cleaner result: **M holds photometric
    bulge fraction that the votes do not carry, beyond anything a shared quantity or an untrained
    encoder produces.**
- **A18:** "tracks measured B/T both ways" is restored as SUPPORTED WITH CAVEAT. The caveat is
  that the vote leg is mostly architectural.
