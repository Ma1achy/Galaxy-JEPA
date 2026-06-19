# Galaxy-JEPA — Probing Harness Design (Paper 1)

*Status: architecture in progress. This is the experimental design of Paper 1's probing
stage — the science, not infrastructure. Decisions here are being made adversarially:
every fork is weighted toward "what survives a hostile reviewer trying to explain the
result away as an artefact," not toward convenience or even cleanliness. British English.*

*Sub-systems are designed in dependency order, not list order. Controls (3) first, because
how tightly the controls gate everything constrains how the ladder (2) and uncertainty
geometry (4) are built.*

**Progress:** Sub-system 3 (controls) — LOCKED. Sub-system 2 (ladder) — 2A locked, 2B
structure-locked / formula-flagged, 2C locked; bottom-of-ladder (MLP gating + resolution
ablation) still to architect. Sub-systems 4 (uncertainty geometry), 1 (full run), 5
(headline figures) — not yet architected.

---

## Two kinds of "locked"

As we architect, decisions fall into two categories — important to keep distinct:

- **Structurally locked & understood** — the design logic (gates, triangulation, ladder
  cascade, feature scope). These are settled.
- **Structurally locked, formula-flagged** — the *structure* is settled, but a specific
  statistical tool/threshold is tagged **"Malachy to own before finalising"** rather than
  rubber-stamped. The architecture is owned; the statistical machinery is specified but
  explicitly pending genuine understanding.

> **OPEN ITEM — stats grounding (do before finalising any formula-flagged decision).**
> The controls/uncertainty-geometry layer is statistically dense and is the part most
> likely to be hammered by a methodologically-sharp reviewer. Before the flagged formulas
> are finalised, work through a focused, *paper-specific* grounding in the ~dozen concepts
> this work actually uses: what a null distribution / p-value means *for the existence
> test*, effect size vs significance *for the AUC*, multiple comparisons, Spearman/rank
> correlation *for the uncertainty geometry*, AUC confidence intervals. Not generic stats —
> tied to the exact use-cases here, enough to *read the methods section and know it's
> right, and defend each choice*. The goal: the most rigorous part of the paper should not
> be the part least understood. Claude to produce the tailored explainer.

---

## Sub-system 3 — The Controls Battery  **[LOCKED]**

**What it's for.** The controls are the entire difference between "I trained a probe and it
predicted bars" (worthless — a probe can predict anything) and "I demonstrated bar-ness is
a recoverable direction in a label-free representation, and ruled out that the probe is
exploiting capacity, nuisance correlates, or label priors" (the paper). Without the
controls, every positive result is uninterpretable. They are load-bearing on the central
claim, not a robustness appendix.

Three families, each answering a distinct "how do I know it's real" attack:
- **Selectivity (Hewitt–Liang)** — real-label probe performance minus control-label
  performance. Answers *"is this just probe capacity?"*
- **Negative controls** — concepts that *should* fail. Answers *"does the pipeline cry
  wolf?"*
- **Nuisance battery** — can the same machinery read off z / size / brightness / SNR / PSF?
  Answers *"is the 'morphology' direction secretly a nuisance?"*

### 3A — Hard gate (rung verdict is a deterministic function of controls)

The keystone decision. The rung verdict is **not** a human reading the numbers — it is a
**pre-registered deterministic function of the controls**, emitted in code, applied
identically to every feature, thresholds fixed *before* looking and stamped. Built on the
existing `Gate` machinery (verdict-independent-of-run-status). The MLP specifically cannot
return a Rung-3 verdict unless selectivity AND nuisance-clearance pass — enforced
structurally, the same way a probe galaxy cannot enter pretraining.

**Why:** controls only protect you if they *gate the conclusion*, not if they merely
*accompany* it. If a human assigns the rung after seeing the numbers, "Rung 1" encodes a
researcher degree of freedom and the whole apparatus is decorative. The defensible answer
to "how did you decide what's a clean direction?" is "a pre-registered threshold function,
applied uniformly, stamped — here it is."

**Cost (accepted):** pre-registered thresholds mean a feature can land just below the bar
and be forced to a rung you "feel" is wrong. That is the point — it's what makes the result
honest.

**The boundary protocol (sub-threshold-but-feels-present features).** When the gate says
Rung 4 for a feature intuition says is present, the response is *not* to argue the verdict
down. It is:
1. The **pre-registered alternative-hypothesis tests fire automatically**, each testing a
   *named* alternative and each gated on its *own* controls:
   - **Controlled MLP** → is it nonlinearly encoded? (Rung 3 — but only if the MLP clears
     its own selectivity + nuisance; you cannot rescue a feature by throwing capacity at
     it.)
   - **Resolution / patch-size ablation** → is it under-resolved by the 16×16 tokeniser
     rather than absent? (The named Rung-4 control.)
   - **Cross-objective** → is it Rung-4 in JEPA but recoverable in MAE/MoCo (an *objective*
     effect, not an image fact)?
2. If a pre-registered test recovers it (clearing its own controls), the feature moves rung
   **with a named mechanism** ("recoverable nonlinearly," "recoverable at higher
   resolution").
3. If none recover it → **report honestly as Rung 4**, and interpret that as a *possible
   finding*: human-label confidence exceeding image information — the v1-confusion-
   explained-by-pixels story, which is the paper's reason for existing. A confidently-
   labelled feature the representation can't recover may be exactly the v2 result.

**The discipline that keeps this honest:** "prove it another way" is legitimate *only* when
the other way is a pre-registered control testing a *named* alternative hypothesis, each
itself gated. It is *never* "try seventeen probe architectures until one clears the bar."
The available rescue-routes are fixed in advance; no new probe may be invented mid-analysis.

### 3B — Null-calibrated thresholds + effect-size floor

The gate is only as good as its thresholds. Thresholds are **not arbitrary constants**:

- **Existence verdict = exceeds the negative-control null at p < 0.05.** The bar for "real"
  is set *relative to* what the negative controls achieve, not hand-picked. This converts
  every threshold from "a number I chose" to "exceeds the null distribution, which I can
  show you" — and it adapts to sample size and probe capacity automatically.
- **PLUS a pre-registered effect-size floor** for "clean vs marginal" among the real
  effects. Guards the opposite failure from false positives: a *statistically significant
  but scientifically trivial* result (with enough test galaxies, AUC 0.54 can beat the null
  while being useless). Significance scales with sample size; effect size doesn't — so the
  floor is what keeps "clean direction" meaningful at full scale.
- The effect floor is a pre-registered constant, which is acceptable *because it no longer
  does the existence work* — significance-vs-null decides real/not-real (principled); the
  floor only decides clean-vs-marginal among confirmed-real effects (lower stakes). Set
  with a defensible rationale (e.g. tied to a meaningful AUC), once, before the run,
  applied uniformly.

**Structural dependency for the build:** the negative-control null is an *input to the gate
function*. The gate **cannot fire until the negative-control null is computed.** Build order
must be: negatives → null distribution → real features gated against it.

> **FORMULA-FLAGGED:** the specific significance machinery (and its multiplicity-corrected
> form, see 2B) is tagged for Malachy to own before finalising. Structure (null-calibrated
> existence + effect floor) is locked.

### 3C — The negative-control battery (five nulls)

The existence verdict is calibrated against the **max (most conservative) across all five**
— a real feature must beat the *strongest* null, so there is no "but you didn't control for
X" left. All five are cheap (computed anyway) and each closes a *distinct* attack:

1. **Shuffled vote fractions** — same label distribution, destroyed image-label
   correspondence. Kills: "the probe exploits label marginals / class imbalance, not image
   content." *Primary selectivity null.*
2. **Random embeddings** (Gaussian, matched to real embedding covariance) — kills: "any
   high-D vector predicts this." *Probe-capacity null.*
3. **Noise images through the real frozen encoder** (noise stamps matched to the data's
   *pixel-level* statistics — post the same asinh+normalise pipeline, NOT white noise, so
   the comparison is structured-galaxy vs unstructured-same-marginals, not in-vs-out-of-
   distribution) — kills: "the encoder imposes structure on *anything*, including noise."
   *Encoder-artefact null — the strong one; weight it in the writeup.* "Noise images through
   our encoder yield chance-level morphology AUC" is a clean, convincing statement that the
   encoder learned galaxy structure specifically.
4. **Untrained-encoder embeddings** (frozen randomly-initialised ViT) — kills: "the probe,
   not the pretraining, does the work." *The headline "pretraining mattered" control* — "a
   random ViT gets ~0.5 on featured-ness, ours gets ~0.9" is the single most convincing
   one-line demonstration that label-free pretraining did the work.
5. **Sky-background / noise-level labels** — kills: "the probe reads image depth/quality,
   not morphology." *Image-quality nuisance negative.*

### 3D — The nuisance battery (five nuisances + triggered matching)

Answers the single most likely *real* confound: morphology genuinely correlates with
physical quantities (bars more visible in bigger/closer/brighter galaxies; faint features
need depth), so a "morphology" direction secretly tracking size is the default thing that
goes wrong, not a hypothetical.

**3D-i — the five nuisances** (all from the existing metadata join; all included — leaving
one out is a "did you control for seeing?" gap):
- redshift (z), apparent magnitude (brightness), Petrosian radius (apparent size), SNR_r
  (image depth), PSF width (seeing).
- (Survey identity is *not* a nuisance for single-survey Paper 1 — deferred to Paper 2.)

**3D-ii — adjudication (parallel-probe standard + triggered matched-evaluation):**
- **Parallel-probe (standard, every feature):** run the same probe machinery to predict
  each nuisance from the embeddings; report morphology-AUC alongside nuisance-AUC. This is
  *diagnostic* — shows *whether* there's a concern.
- **Triggered matched-evaluation (the gate component):** when a nuisance-AUC comes out
  comparable-to-or-above the morphology-AUC, that feature **must** be re-tested on galaxies
  *matched* on that nuisance (so within the matched set the nuisance is held constant and
  can't be the signal). The feature either **survives** (signal is real, not the nuisance)
  or is **marked confounded**. A feature cannot claim a clean rung while a nuisance is
  competitive AND it hasn't survived matching.
- This makes matching *targeted* (fires only for flagged features — bounded cost), not
  "always" (too expensive) or "never/Paper-2" (leaves confounds unresolved). It promotes
  matched-evaluation from the scratchpad's "Paper-2/if-feasible" to "Paper-1, targeted."

**Why every branch is defensible:** "no nuisance competitive → clean" (strong); "nuisance
competitive → matched → survived" (arguably *more* convincing — found a plausible confound
and killed it); "nuisance competitive → didn't survive → confounded" (still a real finding:
"this apparent morphology direction is actually tracking size"). There is no outcome you'd
want to hide — the mark of a well-designed control.

---

## Sub-system 2 — The Ladder

The per-feature Rung 1/2/3/4 classification — the core scientific output ("which Galaxy Zoo
concepts are nameable directions, which entangled, nonlinear, or absent"). The controls (3)
make each rung credible; the ladder is the structure that assigns them.

Rungs: **R1** clean linear direction · **R2** entangled linear (present but not orthogonal
to other concepts) · **R3** nonlinear (linear fails, MLP+controls succeed) · **R4** not
recoverable by anything. Per 3A, escalation is a *gated cascade* where each rung-up tests a
*named* alternative hypothesis and is itself gated.

**The cascade (per feature):**
1. **Linear probe** (canonical L2-logistic) → beats the 5-null max (3C) at the effect floor
   (3B)? **No** → not linearly present, go to step 4. **Yes** → at least R2, continue.
2. **Entanglement test** (2A) → clean (R1) or entangled (R2)?
3. **Nuisance gate** (3D) → competitive nuisance → matched eval → survive (stays R1/R2) or
   confounded.
4. *(only if linear failed)* **MLP probe**, gated on its own selectivity + nuisance →
   beats null? **Yes, controlled** → R3. **No** → resolution ablation (named "under-
   resolved?" test) → R3-if-recovered / else R4.

### 2A — Entanglement (R1 vs R2): eigen-led triangulation  **[LOCKED — strong]**

The strongest part of the controls design. *Not* raw-cosine-vs-arbitrary-threshold (which
is ambiguous between representational entanglement and world-correlation, probe-dependent,
and trivially-passed because random high-D vectors are near-orthogonal by default).
Instead: **several independent measures with different failure modes, triangulated** — the
power is that their weaknesses don't overlap, so where they agree, agreement can't be an
artefact of any single method's blind spot. And where they *disagree*, the disagreement is
*designed to localise the cause*, so every outcome is interpretable.

**Quantification (the eigen spine):**
- **Effective rank of the concept-direction Gram matrix** (WWᵀ, k features × directions) →
  global "I named k concepts but they span only k_eff effective dimensions." Reuses the
  collapse-monitor effective-rank code. *Blind spot:* can't tell representation-entanglement
  from world-correlation.
- **Marchenko–Pastur null** on that spectrum → "significantly more entangled than random
  directions, by random-matrix theory" — a principled, citable null, not a hand-picked
  threshold. *Blind spot:* tests *that* it's non-random, not *why*.
- **Eigenvectors of WWᵀ** → *which* features collapse onto shared axes (an eigenvector
  loading on both "bar" and "bulge" = they entangle along it). Localises entanglement.
- **Embedding-covariance spectrum** (separate eigen-analysis) → the encoder's intrinsic
  effective dimensionality, as *context* ("k concepts occupy 7 of the encoder's ~40
  effective dims — what's in the other 33?" → connects to the SAE/discovery parking-lot
  angle). The ratio concept-span : total-dimensionality is itself interesting.

**Cross-checks (each covers an eigen blind spot, from a different origin):**
- **Cosine matrix** (the WWᵀ off-diagonals, human-readable form) → the **direct visual
  bridge to v1**: the cosine matrix *recovered from the embeddings* laid against v1's
  Fig 18/19 *human-confusion* matrices. Does the representation entangle what humans
  confused? (Spectacular if it holds — the model entangles exactly what humans confused,
  without training on the confusion.) Headline figure 3.
- **Logistic-vs-CAV direction disagreement** → *genuinely independent* of the Gram analysis
  (compares two *different definitions* of the direction — discriminative vs marginal). If
  they disagree, other features are interfering. Convergence with eigen = two unrelated
  methods agreeing.
- **Conditional recoverability (matched)** → the *causal attribution*, the **only** measure
  that resolves representation-vs-world (the eigen blind spot). Reuses 3D's matching
  machinery. Fired **surgically** — only on the feature-pairs the eigenstructure flags as
  most entangled (so it's cheap). Matching on B removes the *world correlation*; what
  survives is *representational*.

**Verdict logic (pre-registered — critical):** convergence across measures = strong claim;
**disagreements are interpreted, not averaged** — and the disagreement-interpretation rules
are fixed *before* seeing results (else triangulation collapses back into researcher-
degrees-of-freedom):
- eigen says entangled + cosine shows it + logistic-CAV disagree + conditional confirms
  survives world-control → representational entanglement, four distinct methods agree —
  unassailable.
- eigen says entangled BUT conditional shows it vanishes under matching → the entanglement
  was *world-correlation (astrophysics)*, not representational — itself a clean finding,
  distinguishable *because* of the causal cross-check.
- eigen says clean + logistic-CAV agree + cosine low → converging evidence for R1.
- The R1/R2 gate verdict requires the eigen-quantification to clear the MP null AND, for
  flagged pairs, the conditional test to attribute it representationally.

**Discipline:** many measures = NOT a license to cherry-pick per feature. *One
pre-registered function* maps (dis)agreement to verdicts, fixed before results.

> **FORMULA-FLAGGED:** the Marchenko–Pastur null application and the significance machinery
> on the spectrum are tagged for Malachy to own before finalising. Structure (eigen
> triangulation, verdict logic) is locked and understood.

### 2B — Multiplicity correction  **[STRUCTURE LOCKED / FORMULA FLAGGED]**

The problem (plain): every "is this real?" test against noise has a small chance of a fluke
pass. Run ~150–300 tests across the full ladder and a handful of flukes are guaranteed
(like flipping enough coins to get a meaningful-looking run of heads). Reporting everything
that "passed" means some "discoveries" are flukes. The correction raises the bar so flukes
don't sneak through — more tests, higher bar. **This is a *design* decision** because the
correction changes the gate threshold in 3B (structural), so it's baked into the gate, not
a post-hoc footnote.

**Structure (LOCKED, understood) — two-level, mirroring the science's structure:**
1. **Primary family = the ~12 "is this feature real?" existence tests.** These carry the
   strict multiplicity bar — they are the headline claims, so the correction goes where it
   matters.
2. **Secondary tests** (nuisance, entanglement, rung, uncertainty-geometry) are
   *characterisation of confirmed-real features only*, run via **hierarchical gatekeeping**:
   tests on features that *failed* existence never happen, so they don't inflate the count.
   The gate structure (3A) *is* the multiplicity control for the secondary tests — you don't
   pay the multiplicity price for nuisance tests on null features. This is *not really a
   stats idea* — it's the structural "confirm first, characterise second" logic, which
   happens to also shrink the multiplicity burden.

**Formula (FLAGGED — Malachy to own before finalising):**
- Tentatively **Bonferroni** on the ~12 primary existence tests — chosen *for
  defensibility*: simple to explain and stand behind ("12 tests, so I divide my threshold
  by 12; a feature clears the stricter bar"). With only ~12 primary tests, Bonferroni is
  not crippling.
- **Benjamini–Yekutieli (dependence-robust FDR)** noted as the higher-power alternative —
  more powerful, and honest about the fact that the tests are *correlated* (shared
  embeddings, correlated features), which vanilla Benjamini–Hochberg assumes away. To be
  chosen between *knowingly* once the stats grounding is in place.
- **Principle driving the flag:** a method Malachy can defend beats a slightly-better method
  he can't. The choice waits until it can be made as *Malachy's* decision, not a
  rubber-stamp. (Claude's recommendation reversed toward Bonferroni *because* of this — the
  honesty about the stats gap was what surfaced the better call.)

### 2C — Feature scope  **[LOCKED]**

- **Full ladder on all features of the dissertation-fixed GZ2 decision tree** — the version
  Malachy "fixed" in v1 to better represent actual features (vs canonical GZ2). Chosen for
  direct comparability with v1 (same features → the "v2 explains what v1 found" story is
  clean).
- **Headline analysis focused on the v1-confused features** (bulge size, winding, arm-count
  — where v1's models inherited human confusion), because that's where the nameability
  ladder has the most to say.
- **Required documentation (reviewer-facing):** the paper must (a) state precisely how the
  fixed tree differs from canonical GZ2, (b) justify per-feature *why* each change better
  represents the underlying feature, (c) note features aren't 1:1 comparable to other GZ2
  papers using the canonical tree (but ARE comparable to v1, the comparison that matters).
- **Build flag:** does the fixed tree change the *conditional structure* (which questions
  depend on which)? This affects per-feature sample sizes → affects the null-calibration in
  3B. Parked for the build; matters for implementation, not the design decision.

### Bottom-of-ladder (MLP gating + resolution ablation)  **[LOCKED]**

How the linear→MLP→R4 cascade fires when the linear probe fails. The MLP inherits the gate
machinery (3A/3C/3D); the resolution ablation is the named Rung-4 control.

**2D — the MLP: a bounded capacity ladder (the capacity-trap-proof R3/R4 boundary).**
The MLP's job is to detect nonlinear-but-present structure — but capacity is a trap: too
weak misses real nonlinearity (false R4); too strong decodes *anything* including the nulls
(false R3). And selectivity alone doesn't fully resolve it — a high-capacity MLP can overfit
*both* the real label and the control, collapsing selectivity to ~0 in *either* direction.
Resolution: a **bounded capacity ladder** — sweep capacity, but only within the range where
the probe still fails the negative controls.
- **Sweep knob:** hidden-layer **width**, with depth / regularisation / training-time
  **fixed** — a clean one-dimensional, interpretable capacity axis.
- **Selectivity ceiling:** the width at which negative-control AUC **first exceeds its own
  null**. Below the ceiling, the probe genuinely can't decode noise (selectivity is
  meaningful); above it, the probe overfits and any decodability is untrustworthy.
- **R3/R4 verdict (principled, non-arbitrary):** a feature is **R3 iff it becomes decodable
  at some width *below* the ceiling**; **R4 if it only decodes at/above the ceiling** (where
  the controls also get decoded). This kills the capacity trap *by construction* — you
  cannot declare R3 by cranking capacity, because past the ceiling the verdict is invalid.
- **Bonus measurement:** *where* in the valid range a feature becomes decodable = "how much
  nonlinearity it needs," reported for the headline confused features (a richer result than
  a single MLP verdict).
- **Scope:** fires **only on linear-failures** (cleanly-R1/R2 features never need it),
  bounding the expensive/complex test to exactly where R3-vs-R4 is live.
- **Justification for the complexity:** this is the most complex single piece of the ladder
  — complex *in exact proportion to* it being the part most prone to manufacturing false
  positives, which is where complexity is justified.

> **FORMULA-FLAGGED:** the precise "exceeds its own null" ceiling mechanics (the null-
> distribution machinery for the selectivity breakdown point) is tagged for Malachy to own
> before finalising. Structure (bounded ladder, width-axis, ceiling-defined R3/R4 boundary,
> linear-failures-only) is locked and understood.

**2E — resolution ablation: full 8×8 encoder + full ladder re-run (an *experiment*, not
just a control).** Distinguishes "absent from the pixels" from "under-resolved by the 16×16
tokeniser": a feature R4 at 16×16 but recoverable at 8×8 was *under-resolved*, not absent —
a genuinely different finding. **Crucially, patch size affects the *entire* representation,
not only the R4 features** — so the full ladder is re-run at 8×8, not just the R4 features
re-probed. This captures effects across the board: a feature that's R2 (entangled) at 16×16
may become R1 (clean) at 8×8; the *entanglement structure itself* (the eigen-analysis) may
shift with resolution. Re-probing only R4 features would miss all of that. Patch size is a
genuine *axis of the experiment* (per the scratchpad's Rung-4 ablation), not merely an R4
control — running the full ladder at both resolutions gives the complete "how does
tokenisation resolution shape nameability" picture (a figure: the ladder at 16×16 vs 8×8).
- **Cost (accepted):** a second full pretraining run + full re-probe. Real, but buys a
  second axis of results, not just a control.
- **Discipline:** identical pretraining recipe, *only* patch size varies, so any difference
  is attributable to resolution (parity, again).

**2F — failure-cascade order: MLP first.** linear fails → **bounded-capacity MLP ladder**
(gated) → if that also fails → **8×8 resolution re-run** (2E) → genuine **R4**.
Cheapest-and-most-specific first: the MLP is probe-time (no retraining) and "nonlinearly
present at this resolution" is a more specific hypothesis than "recoverable at finer
resolution."

---

## Remaining sub-systems (not yet architected)

## Sub-system 4 — The Uncertainty Geometry (the headline)  **[LOCKED]**

The result to chase — if it lands, it flips v1's conclusion. Structurally, the hardest part
is *already built*: the non-circular firewall lives in `data/splits.py`
(`assert_uncertainty_firewall`, `partition_uncertainty`), so the discipline that makes this
non-circular (the axis never sees the gradient it's later asked to reproduce) is already a
structural guarantee. Sub-system 4 is the *measurement* layer on top of that firewall.

**The claim, sharply:** not "does the axis classify bars?" (that's the ladder) but **"does
distance along the unsupervised concept axis reproduce the human vote *fraction* — the
graded uncertainty — for galaxies the axis never saw?"** Fit the axis on consensus extremes
only (v>0.8 vs v<0.2, binary); project the held-out *ambiguous middle* (0.2<v<0.8); test
whether projection distance *ranks* their human vote fractions. If an axis trained only on
confident examples orders the ambiguous ones by a human uncertainty it never saw, the
geometry genuinely aligns with the human ambiguity gradient — so the ambiguity is a *real
property of the images*, not a labelling artefact the model absorbed (the v1 reading). Same
observation, opposite conclusion.

### 4A — Test statistic + significance

- **Spearman (rank) correlation primary** — projection distance vs vote fraction on the
  held-out middle. Rank, because you care about *ordering* and don't want to assume a linear
  projection-to-fraction relationship; robust to the projection's arbitrary scale. **Pearson
  secondary.**
- **Significance via permutation test** — shuffle the vote fractions, recompute Spearman
  many times, see where the real value falls in that null distribution. Same null-
  calibration philosophy as the whole controls battery.

> **FORMULA-FLAGGED:** the permutation-test mechanics (and what the resulting p actually
> means *here*) are tagged for Malachy to own in the stats grounding. Structure (Spearman
> primary, permutation null) locked.

### 4B — Scope: gated on R1/R2, run on all

Uncertainty geometry only makes sense for features that are **recoverable in the first
place** — you can't ask "does the axis reproduce the vote fraction" for an R4 feature (no
meaningful axis exists). So it is **gated on the ladder: only R1/R2 features** (clean or
entangled linear directions — there's an axis to project onto) are candidates. Run on **all
R1/R2 features** (cheap once the axis exists — a projection + a rank correlation), with the
**headline on the confused features** (where "the geometry reproduces the human uncertainty
the supervised v1 model absorbed" is the money result).

### 4C — The v1-vs-v2 comparison (first-class deliverable, not an aside)

The force of this result is the *flip* against v1. v1's signature finding was the model's
confusion matrices *mirroring* the volunteers' — *because it was trained on the votes*. The
uncertainty-geometry result is only a flip if you explicitly show: v1 reproduced human
uncertainty *because trained on it*; v2 reproduces it *without ever seeing it*. So
sub-system 4 **must produce a direct, quantified v1-vs-v2 comparison** — same features, v1's
supervised-confusion-mirroring laid against v2's label-free-geometry-recovery — as a
**headline figure/result**, not a discussion point. Without it: "an axis correlates with
vote fractions" (mildly interesting). With it: "the ambiguity v1 proved was a labelling
artefact is actually recoverable from the images alone" (the paper's hook). **The comparison
*is* the result.**

### Honest framing (designed-in, per the scratchpad)

Uncertainty geometry is the **highest-upside AND most-likely-to-come-out-null** result — an
axis fit on extremes might simply not order the ambiguous middle (the geometry might not
encode the gradient). Designed for accordingly:
- The paper is **backboned on the ladder + controls + v1-comparison** (solid); uncertainty
  geometry is the **high-beta headline**, not the foundation. Don't bet the paper on it
  landing positive.
- **A null result is itself a finding** (same discipline as ladder R4): "the representation
  separates confident cases but does not encode the *graded* human uncertainty — the
  geometry is categorical, not graded." Reported honestly either way.
## Sub-system 1 — The Full Run  **[LOCKED]**

Mostly settled — the slice proved the recipe (ViT-S/16 @256², β=0.5, the EMA/masking config,
green at pilot scale). The full run is "the same, at the scale that makes a *null*
trustworthy" (100k pretrain / 50k steps — the pilot is explicitly too undertrained for an
honest null).

### Compute path (the gate that shapes 1A/1B)  **[LOCKED]**

Principle: **never rent compute to debug — renting is for throughput, not correctness.** The
full programme (headline run + sweep + replicates) is infeasible on the Mac in reasonable
time, but the *correctness/worth-it* questions don't need the full local run. So:

1. **Medium local run (~30–40k galaxies, ~10–15k steps)** — the code-correctness + worth-it
   gate. Exercises the *entire* pipeline end-to-end (real dataloader at a working-set size
   that exposes thrash/memory issues, full train loop, freeze, probe, all figures, explorer
   blobs) in hours not days. If it runs clean *and* shows signs of life on the science
   (ladder starting to work on a couple of features), both questions are answered.
2. **100k-working-set dataloader smoke** — bake the full 100k cache and iterate the
   dataloader for a few hundred steps to confirm it feeds without thrashing (the thrash
   cliff is working-set-size dependent; 100k is bigger than 40k), then stop. The
   100k-specific de-risk *without* a full multi-day training run. An hour, not days.
3. **Then rent** (budget available — Colab Pro / rented A100 etc.) for the *actual* full
   100k/50k headline run, the β-sweep, and the replicates — on **proven code**, buying speed
   not debugging.

This is faster overall than a full local run first (rent-decision in hours, not days) and
doesn't burn days on a local run that the rented run supersedes. Same prove-small-before-you-
spend logic as the original pilot, one level up. The harness consolidation already made the
SciServer pull chunkable and the code device-agnostic for exactly this.

### 1A — β stays 0.5 for the headline; β-sweep is a downstream ablation

The headline full run is **β=0.5** (the proven pilot config) — *one* clean encoder for the
whole ladder/controls/uncertainty-geometry, not three encoders splitting attention before a
single complete result exists. The **β-sweep (β=0 / 0.5 / 1.0)** is a *separate downstream
ablation* run **after** the headline result — it answers "does galaxy-biased masking matter,
or would standard I-JEPA (β=0) do as well?", which is a genuine *result* (publishable:
"biasing masks toward the galaxy changes nameability in this way"; β=0-stable-but-0.5-
better/worse localises what the bias does), but a *secondary* one. With rent budget it's an
easy, affordable add once the primary result is in — deferred to *after*, not indefinitely.

### 1B — replication: free probe-replication always + paid encoder-replication on headline features

- **Free probe-replication (always):** the probe is logistic regression on frozen
  embeddings (seconds), so re-run it with different seeds / probe-set splits *for free*.
  Tests whether the *rung verdicts* are stable given the encoder — the part most likely to
  be flaky (split sensitivity). Costs nothing, real rigour.
- **Paid encoder-replication (budget available) on the headline confused features:** re-run
  the full pretraining with different seeds to assess *encoder-init variance* — only on the
  headline features, not the whole ladder. "Rungs stable across *both* probe-resampling
  *and* encoder seeds" is a much stronger claim than a single run.
- Honest either way: if budget runs short, "rungs stable across probe resampling;
  encoder-init variance from N seeds on headline features" is a legitimate, reviewer-
  acceptable position.

### 1C — checkpoint choice: label-blind (NOT best-AUC)  **[the subtle one]**

**Do NOT pick the checkpoint that maximises probe AUC.** Best-by-AUC lets the *labels* (via
the probe) choose which encoder you keep — which quietly **breaks the "encoder never saw
labels" firewall** (you'd be label-selecting the encoder), a subtle circularity a sharp
reviewer catches. The checkpoint must be chosen by a **label-blind** rule, pre-registered
*before* looking at any probe AUC:
- **Final checkpoint** (most-trained), gated on the collapse trace being stable at the end
  (which the pilot showed) — **unless** the collapse trace shows late-stage degradation, in
  which case **last-stable checkpoint**.
- (Alternative label-free criteria — lowest pretraining loss, healthiest effective rank —
  also acceptable; the discipline is label-blindness, not the specific criterion.)
- This is the one place the intuitive choice ("best performing") is subtly *wrong* —
  best-performing-by-AUC is exactly the label-leak the frozen-encoder design exists to
  prevent.
## Sub-system 5 — The Headline Figures (deliverables)  **[LOCKED]**

Not new methodology — *specifying the deliverables* sub-systems 2–4 produce, so the harness
is built to **emit** them rather than reverse-engineering figures from raw outputs afterward.
"The harness must produce figure X" is a build requirement, so the figures fall out of the
run.

**Figure 1 — The nameability ladder** (sub-system 2's output). Per-feature: rung (R1/R2/R3/
R4) + AUC + selectivity + calibration + whether it cleared the controls. Ordered to tell the
story ("smooth/featured and edge-on clean R1; bar R2-entangled; winding R3-nonlinear;
arm-count R4-absent" — the ladder *as narrative*), confused features highlighted.

**Figure 2 — Uncertainty geometry** (sub-system 4's output). Projection-distance vs human
vote-fraction on the held-out ambiguous middle, per feature, with Spearman. Money panel
pairs a *clean* feature (axis recovers the gradient) against a *confused* one; the **v1-vs-v2
comparison (4C)** lives here or as its own panel — v1's supervised-confusion-mirroring beside
v2's label-free-geometry-recovery.

**Figure 3 — The entanglement geometry** (sub-system 2A's output). The cosine matrix
recovered *from the embeddings* laid against v1's Fig 18/19 *human-confusion* matrices (does
the representation entangle what humans confused?), plus the eigenspectrum (effective rank,
Marchenko–Pastur comparison). The visually striking one — "the model's entanglement geometry
matches the humans' confusion geometry, without being trained on the confusion."

**Controls — also get figures** (generate generously, triage editorially later). At the
architecture stage, produce figures for the controls too (the five-null comparisons, the
nuisance-AUC panels) rather than prematurely consigning them to supplementary — you can't
sensibly decide headline-vs-supplementary before you have the results. Cut later if too much.
- **Note for the eventual triage:** the two control results most likely to earn *headline*
  placement are **"untrained encoder ~0.5 vs ours ~0.9"** (proves the pretraining did the
  work — arguably the single most important control) and **"noise-images-through-encoder give
  chance AUC"** (kills the encoder-artefact worry). If it comes to a cut, fight for those
  two; the rest triages on the evidence.

---

## ARCHITECTURE COMPLETE

All five sub-systems are architected — every fork resolved or explicitly flagged. The full
experimental design of Paper 1's probing stage is specified: the controls battery (3), the
ladder (2, including the bottom-of-ladder MLP/resolution cascade), the uncertainty geometry
(4), the full run + compute path (1), and the headline figures (5). The statistical formulas
are honestly flagged for the grounding session; the structure is owned throughout.

**Next:** (a) the stats-grounding session (the boxed OPEN ITEM — before any formula-flagged
decision is finalised), then (b) hand this design to Claude Code to produce a *build* plan
for the harness (the way the slice was build-planned), then (c) the medium local run →
dataloader smoke → rent → full run → the science.

---

## Open items (carry forward)

1. **Stats grounding** (before any formula-flagged decision is finalised) — the tailored,
   paper-specific explainer. See the boxed OPEN ITEM near the top.
2. **Formula-flagged decisions awaiting (1):** 2B multiplicity correction (Bonferroni vs
   BY); 2A Marchenko–Pastur null application; 3B significance machinery; (4) Spearman
   significance when architected.
3. **Build flags:** 2C conditional-tree-structure → per-feature sample sizes → 3B null
   calibration.
4. **Still to architect:** bottom-of-ladder (MLP gating + resolution ablation); sub-systems
   4, 1, 5.
