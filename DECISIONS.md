# Decisions

Open forks needing your call, each with a **recommendation + reasoning**. Status
is one of: **needs your call**, **decided (scratchpad)**, or **proposed (this
doc)**. Tick a box when you sign off.

> The scratchpad (`galaxy-jepa-scratchpad.md`) is the source of truth. Where a
> fork is already settled there, it is repeated here for completeness and marked
> *decided (scratchpad)*. If a decision below changes the design, the change is
> proposed as an edit to the scratchpad, not made silently.

---

## D1 — Framework — *decided (signed off): PyTorch*

- [x] **PyTorch** ☑  ·  Keep TF/Keras (v1) ☐

**Recommendation: PyTorch.** The I-JEPA reference implementations and the SSL /
interpretability ecosystem (linear probes, CAV/TCAV, SAEs) live in PyTorch, and a
from-scratch JEPA with custom masking, EMA, and probing wants that flexibility.
v1 was 100% TF/Keras but **none of its model code is reusable** (different
framework), so there is no porting cost on the model side — only logic to
re-express (label schemes, crop, vote handling). This matches your stated lean.

---

## D2 — Backbone — *an ablation axis, not a one-time pick (default: clean ViT-S/16)*

- [x] **Paper-1 default: clean ViT-S/16 @ 256²** (→ 16×16 = 256 tokens; matches v1 patchification).
- [ ] **Backbone sweep (rung confound control):** clean ViT → conv-stem hybrid (CCT/CvT) → E(2)-equivariant ViT ☐

**Reframe — the backbone is a controlled variable, not a perf pick to defer to
"ViT-B later".** "Architecture" is the same term in the
*(feature × SSL-objective × architecture × probe)* decomposition, so the backbone
sweep is a **rung confound control** sitting alongside the patch-size (Rung-4,
**D11**) and cross-objective (Rung-3, **D12**) controls — a feature can be Rung 3
under a clean ViT but Rung 1 under a conv stem or an equivariant prior.

**Why clean ViT is the Paper-1 default (I-JEPA-specific reason):** a conv stem's
receptive field **bleeds target-region pixels into context tokens *before*
masking**, leaking the exact masking mechanism this project is built to study and
muddying the β sky-fraction diagnostic (`docs/masking.md`). Clean ViT keeps the
novel masking semantics unambiguous — **principled, not inherited from v1**.

**Dependency on D6:** from-scratch ViT is viable *conditional on* the larger
unlabelled corpus (D6). If the corpus stays thin, v1's conv-stem **CCT is the
data-efficient fallback**.

**Caveat (no over-claiming):** Zoobot settled on ConvNeXt/MaxViT, so make **no
"ViT is best for galaxy morphology" claim** anywhere — clean ViT is chosen for
masking cleanliness, not assumed-optimal accuracy.

---

## D3 — Environment tooling — *decided (signed off): uv + devcontainer*

- [x] **uv + devcontainer + pytest + pre-commit (ruff), Python 3.11** ☑

**Recommendation:** match your other repos — **uv** for env, **devcontainer** for
reproducibility, **pytest** for tests, **pre-commit + ruff** for lint/format.
Propose **Python 3.11** (v1 used 3.10; 3.11 is a safe, faster default and well
supported by PyTorch). Flag if you'd rather pin 3.10 for parity with v1.

---

## D4 — From-scratch vs ImageNet warm-start — *decided (signed off): from-scratch*

- [x] **From-scratch** ☑  ·  ImageNet warm-start ☐

**Recommendation: from-scratch.** The central claim is that morphological
directions are *present before any label*; an ImageNet-initialised encoder imports
natural-image priors that **muddy attribution** ("is this direction from galaxy
images or from ImageNet?"). From-scratch is the clean, defensible canonical run.
Keep **warm-start as a Paper 2 ablation** (label-efficiency / compute trade-off),
eyes open.

---

## D5 — Masking strategy — *decided (signed off): bounding-box-biased; see `docs/masking.md`*

- [x] **Bounding-box-biased multi-block** (β-sweep, β=0 = I-JEPA control) ☑

**Recommendation:** adopt the scheme in `docs/masking.md`. It is a **strict
generalisation** of I-JEPA (β=0 reproduces it), adds three knobs (β, τ, φ),
biases the prediction budget onto the galaxy, and ships its own sky-waste
diagnostic. **This is the masking sign-off the kickoff asks for before any model
code.**

---

## D6 — Pretraining vs probing corpus — *decided (signed off): decouple; both single-survey*

- [x] **Decouple corpora** — pretrain on a **large unlabelled SDSS** sample
  (≫250k), probe on the **GZ2-labelled ~250k** ☐
- [x] **Single-survey for Paper 1** — no multi-survey (that is Paper 2).

**Recommendation: decouple the pretraining corpus from the probing corpus.**
Pretraining needs **no labels**, so there is no reason to cap it at the
GZ2-labelled set. From-scratch I-JEPA on only ~250k is **thin** — I-JEPA was
trained on ImageNet-1k/22k (1.3M–14M). Galaxy images are lower-entropy (centred,
black background, a small morphology vocabulary) so it may cope, but this is the
**single biggest risk to the from-scratch call (D4)**: an undertrained encoder
gives a muted probing story and you cannot distinguish *"information absent from
the pixels"* (a real Rung-4 result) from *"the encoder never learned it"* (an
artefact).

The fix **stays single-survey**: pretrain on a large unlabelled SDSS galaxy sample
(SDSS imaging has far more galaxies than the GZ2-labelled subset), probe on the
GZ2-labelled ~250k. It is also **more on-thesis** — the entire LSST framing is
"oceans of unlabelled data", so pretraining on *exactly and only* the labelled
subset would quietly undercut the "we don't need labels" claim. Cost: **one extra
SkyServer pull** of galaxy cutouts beyond GZ2, which slots straight into the P2
data layer.

**Data-layer consequence:** the unlabelled pretraining pull must also fetch
**petroRad + the cutout's arcsec/pixel scale** per galaxy — needed for the
per-galaxy masking box (`docs/masking.md` §3.1), since masking runs on this
*pretraining* corpus, **not** the GZ2 probing set the nuisance battery covers.
`petroRad` is in SDSS `PhotoObjAll` for every photometrically-detected galaxy
(not just spectroscopic), so it is available — but it is a **distinct pull**, not
covered by the nuisance-battery join. Going fainter than the GZ2 spectroscopic
limit (r < 17.77) to reach ≫250k makes petroRad noisier on the faint end (the
global-box fallback + the *k* slack absorb this), implying a mild
pretraining-vs-probing distribution shift (fainter, smaller apparent size).

Multi-survey scaling + the survey-leakage merge experiment remain **Paper 2**.

---

## D7 — Canonical probe — *decided (scratchpad): L2 logistic*

- [x] **L2-regularised logistic** is canonical; **mean-difference (CAV)** is the
  robustness check (and their disagreement is itself an entanglement signal).

---

## D8 — "Reliable" label filter — *decided (signed off): reuse v1's mean+2σ method*

- [x] **Reuse v1 vote-agreement filter (mean + 2σ)** ☑

**Recommendation:** reuse v1's agreement filter for **general probe label
quality**, but note it is **separate** from the uncertainty-geometry protocol —
which deliberately uses the **consensus-extremes** split (train on v>0.8 vs v<0.2,
test on the held-out 0.2–0.8 middle) and must *not* be pre-filtered in a way that
removes the ambiguous middle it needs to test on. *(v1 only applied a 0.5
threshold; the mean+2σ filter is net-new and must be implemented.)*

---

## D9 — Confidence usage — *decided (scratchpad): both, kept separate*

- [x] **Both axes, strictly separate.** Confidence may be a **probe target**
  (regress / weight by vote fraction) *and* the **uncertainty-geometry test**,
  but never the same axis — recovering a vote fraction along an axis trained on it
  is a tautology. The two uses inform the **probe and the evaluation, never the
  encoder**.

---

## D10 — Symmetry — *decided (scratchpad): augmentation first*

- [x] **Rotation/reflection augmentation first** (simplest, derisks the minimal
  run); **E(2)-equivariant ViT as a later ablation** (it reshapes encoder
  geometry, so establish the vanilla-ViT ladder before baking in symmetry).

---

## D11 — Resolution / patch-size ablation — *decided (scratchpad): 8×8 as Rung-4 control*

- [x] **8×8-patch (or higher-res) ablation** is the **Rung-4 control** — it
  distinguishes *absent from the pixels* from *under-resolved by the tokeniser*
  (thin arms / tight winding below the token floor). Not the default backbone (D2).

---

## D12 — Cross-objective ladder — *decided (scratchpad): JEPA vs MAE vs contrastive*

- [x] **Same probe ladder across JEPA, MAE, contrastive** — the **Rung-3
  control**, attributing a rung to the *objective* vs the *images*.
- [x] **Every cross-objective baseline (MAE and contrastive) must be trained on the
  *same SDSS pretraining corpus* as the JEPA (D6).** The Rung-3 control holds the
  **dataset** fixed and varies only the **objective**; an off-the-shelf baseline
  trained on a different instrument varies objective *and* instrument, confounding
  exactly what the control isolates. This **resolves the earlier "train ours vs
  adapt published" sub-decision → train ours on SDSS.**

**MAE:** reproduce the **Wu & Walmsley (arXiv 2510.23749)** recipe — a ViT, ~30M
params, 3-layer decoder, **8×8 patches** (same patch size as the D11 Rung-4
control) — on our SDSS corpus. Their **released Euclid MAE is a reference / a way
to validate the reimplementation, *not* the controlled baseline** (it is
Euclid-trained). Byline verified and **unchanged**: John F. Wu & Michael Walmsley,
two co-first authors (see `docs/related-work.md`).

**Contrastive: MoCo — *sub-decision resolved (signed off)*, trained on the SDSS corpus.** BYOL is negative-free +
EMA-target — *too architecturally close to JEPA* for a clean Rung-3 contrast; MoCo's
explicit negatives make it a genuinely different objective, and it is the established
galaxy-SSL baseline (Hayat et al. 2021, on SDSS — see `docs/related-work.md`).

---

## D13 — Confound taxonomy + inclination conditioning — *decided (signed off; Framing-B mechanism)*

- [x] **Human confusion has distinct *physical* causes, diagnosed with the label-free encoder.**
- [x] **Inclination is a first-class conditioning axis; proxy = axis ratio (b/a).**

The label-free encoder never sees votes, so per confused feature we can ask whether the confusion
is in the **data** (encoder also confused → genuine information limit) or the **humans** (info in
the pixels; encoder separates what people cannot). Grounded in v1's own correlation analysis,
confusion splits three ways, each with a distinct fingerprint across the
**(inclination × imaging-depth)** plane:

1. **Projection** (viewing-angle information loss) — e.g. edge-on disk ↔ cigar elliptical
   (v1: Edge-on × Cigar = +0.83). Angle-dependent, imaging-depth-**invariant**.
2. **Resolution *or* semantic** — arm-count, winding, bulge-shape (near-zero v1 off-diagonals).
   The imaging-depth axis distinguishes them: resolution **improves** with deeper imaging;
   semantic does not.
3. **Genuine co-occurrence vs artefactual correlation** *(HYPOTHESIS — unconfirmed)* — bar +
   spiral structure (v1: Bar × 2-arms = +0.56). The method's hard case: entanglement here may be
   **correct physics**, not a representation limit. Adjudicated by the eigen-triangulation's
   causal cross-check (conditional-recoverability under matching).

**Inclination proxy = axis ratio (b/a)** — an *independent photometric* measurement (SDSS
pipeline, from the pixels), so conditioning on it to study *vote*-confusion is **not circular**
(using the T01/T07 votes as the proxy *would* be). This is a **new capability on the existing
probe** (probe within inclination bins / with b/a as covariate) — it does **not** revise the
locked probing sub-systems.

**Status of the taxonomy for the paper:** it is the *mechanism for Framing-B's earned payoff*,
held as interpretive lens **pending results** — NOT the paper's spine (which stays Framing-A:
method + ladder + controls). See the design spec (`docs/galaxy-jepa-spec.pdf`, §Framing,
§Confound).

**Data-layer consequence:** b/a (`expAB_r`, `deVAB_r`) is SDSS photometry, **not** a GZ2 vote
column — a **new pull requirement** for the probe corpus, distinct from both the masking pull
(petroRad + arcsec/pixel) and the nuisance join (z/mag/radius/SNR/PSF). Cheap: a `PhotoObj` join
on `objID`, no image re-cut.

> **Landed.** `metadata.AXIS_RATIO_SQL` / `pull.pull_axis_ratios` + `pull.merge_columns`; the two
> columns are in `data/probe-40k/metadata.csv` (40,000/40,000 matched). Deliberately **not** in
> `probing.extract.DEFAULT_NUISANCE_COLS` — inclination is a conditioning axis, and regressing it
> out as a nuisance would remove the very thing the taxonomy studies (invariant-tested).
> **Open:** which axis ratio per population (`expAB` for disks vs `deVAB` for ellipticals) — both
> are pulled, so the choice stays downstream of the data.

---

## D14 — Feature-set = a two-scheme experiment, conditional-population probing — *decided (signed off)*

- [x] **The feature set is an *experiment over schemes*, not a fixed choice.**
- [x] **Each feature probed within its conditional population — as a *comparison*, not a hard mask.**

**Conditional-population probing.** The GZ2 tree is conditional: a feature is only well-defined
within the population that reaches its question (boxy-bulge is meaningless for a no-bulge galaxy —
v1's "Q4 can't be yes and Q7 can't be no for the same galaxy"). **But do not hard-mask the
"incoherent" galaxies away** — a no-bulge galaxy carrying boxy-bulge votes is a *measurement of
human disagreement*, and masking it pre-imposes the tree's logic before testing whether it holds
(circular). Instead: probe each feature across **different** population definitions (full vs
consensus-conditional) and **compare**; study the off-population galaxies as their own object
(concentrated = systematic confusion = finding; scattered = noise). Reuse `data/splits.py`
firewall machinery. The consensus gate threshold is a **per-run knob**.

**The two schemes (the experiment).**
- **Scheme 1 — full tree (37 answers, per-bucket)**, each in its conditional population. Honest
  baseline; expected weak on the v1-confused features (echoes v1 = a finding). BY family = 37.
  **Power confound:** per-bucket deep-feature weakness is confounded between genuine-absence and
  split-sample (~9,870 spirals ÷ 6 arm-buckets ≈ 1,600 each) — Scheme 1 alone can't distinguish;
  do **not** read per-bucket weakness as "absent."
- **Scheme 2 — reduced/smart**: graded questions → one graded axis each; binary well-posed → one
  binary feature; odd-subtypes exploratory. BY family ≈ 10–13. Also a **power diagnostic**.
- **The comparison is a result.** Same ladder both ways ⇒ reduction cosmetic; differ ⇒ reducing
  changes what's expressible ⇒ a real taxonomy result. **Order: full first** (transparent).

Implementation: **schemes are configs**, one harness (reconfigure, don't rebuild); BY family count
is per-config. See `docs/galaxy-jepa-spec.pdf`, §Feature-scheme experiment.

> **Landed.** `probing/schemes.py` (`full_tree_scheme` = 37, `reduced_scheme`), the conditional
> chains as `metadata.GZ2_CONDITIONS`, per-feature eligibility on `LabelProvider`, and the
> full-vs-conditional comparison in `run_probing` (`ProbingReport.population_comparison`).
> **Open sub-question — graded-axis existence test (AUC vs correlation).** A graded axis may get a
> *correlation* test (Spearman/permutation) rather than AUC — but that is the *same measurement*
> as the uncertainty geometry for that feature, so they may collapse. Binary features keep AUC.
> **Not resolved:** `FeatureSpec.require_testable()` raises `GradedExistenceTestUndecided` rather
> than defaulting. Scheme 1 has no graded features, so it runs first and this blocks nothing.
> **t09 bulge shape — resolved (signed off): one binary feature, boxy versus rounded**,
> conditioned on edge-on **and** bulge-present. Scheme 2 goes to **10 primaries**, inside the
> spec's stated band.
>
> *Why binary and not a fifth graded axis.* Rounded / boxy / no-bulge is not ordered. The four
> axes already named are all genuinely ordinal (1→2→3→4→5+; tight→medium→loose;
> none→just-noticeable→obvious→dominant; round→in-between→cigar). Bulge shape is a categorical
> contrast with an absence bolted on, and collapsing it to an axis would impose an order that
> does not exist — the exact failure the graded framing exists to prevent. The two-way contrast
> also *is* D13's confound-2 deliverable: whether the encoder separates boxy from rounded where
> humans cannot lives entirely there. The double condition exercises the conditional-population
> machinery harder than a flat three-way split would.
>
> *Rejected-but-**deferred**, not discarded: three per-answer binaries at family 12.* Its one
> real argument is that t05 and t09 are asked of **disjoint** populations (featured non-edge-on
> versus featured edge-on), so t09's no-bulge is **not** redundant with t05's low end — it is the
> same concept measured on the other branch, structurally v1's Q4/Q7 situation. If a cross-branch
> consistency check is wanted later, family 12 is where it lives.
>
> Implementation note: the gate is expressed as the **summed** rounded+boxy share clearing the
> consensus threshold, not as a negated no-bulge gate, so every condition in the scheme keeps
> pointing the same way (`schemes.FeatureSpec.condition_groups`).

---

## Summary — decisions and their state

All of D1–D14 are now resolved. The table records what was chosen.

| # | Fork | Decision |
|---|---|---|
| D1 | Framework | **PyTorch** |
| D2 | Backbone | **Clean ViT-S/16 default** (masking-clean); backbone sweep (ViT→CCT/CvT→E(2)) is a rung confound control; CCT fallback if corpus thin |
| D3 | Env / Python version | **uv + devcontainer; Python 3.11** |
| D4 | From-scratch vs warm-start | **From-scratch** |
| D5 | Masking | **Bounding-box-biased** (`docs/masking.md`) |
| D6 | Pretraining vs probing corpus | **Decouple** — pretrain on large unlabelled SDSS, probe on GZ2 ~250k (both single-survey) |
| D8 | Reliable-label filter | **Reuse v1 mean+2σ** (separate from uncertainty protocol) |
| D12 | Cross-objective baselines | **All trained on the same SDSS corpus** (MAE = reproduce Wu & Walmsley recipe on SDSS; Euclid MAE is reference only) |
| D12 (sub) | Contrastive choice | **MoCo** (SDSS-trained) — explicit negatives = clean contrast vs JEPA; established galaxy baseline (Hayat) |
| D13 | Confound taxonomy + inclination | **Axis ratio (b/a) as the non-circular inclination proxy**; taxonomy is the Framing-B interpretive layer, held pending results |
| D14 | Feature scope | **Two schemes as configs on one harness** (full-37 first, then reduced); conditional population as a **comparison**, never a mask; BY family per-scheme |

**Still open** (tracked in `docs/galaxy-jepa-spec.pdf` §Open questions register, not re-litigated
here): the graded-axis existence test (D14); the effect-floor *value*; tie-handling in the
existence p and the permutation test; the consensus-gate and vote-count thresholds; which axis
ratio per population (D13).
