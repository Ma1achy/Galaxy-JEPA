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

### D6 final — what was actually built

Both corpora are pulled and verified; these are measured, not planned, numbers.

| | probe | pretrain |
|---|---|---|
| galaxies | **230,358** | **826,968** |
| source | full `zoo2MainSpecz` | SDSS `PhotoPrimary`, never in any GZ2 table |
| selection | GZ2's own | `type=3`, `clean=1`, `modelMag_r` 14–19, **`petroRad_r` ∈ (5″, 25″]** |
| labels | raw vote fractions, t01–t11 (no debiased column) | none |
| footprint | 171 GB | 612 GB |
| median angular size | 2.05 ViT patches | 1.94 ViT patches |

The pretrain pull targeted 826,984 and landed **826,968** — sixteen galaxies lost to chunks that
died at the SciServer end and whose retries also failed. Recorded rather than papered over; at
this scale it changes nothing, and `data_snapshot` hashes what exists, not what was intended.

**The reasoning, recorded so it is never relitigated:**

- **The 5″ floor is a *resolution* requirement, not distribution matching.** Below it the median
  galaxy spans about one 16×16 patch — one token, no resolved morphology to learn, and nothing
  for the bbox-biased masking (D5) to bias toward. After the cut the pretrain median is **1.94
  patches against the probe's 2.05**: parity, and it is the floor that buys it.
- **The 25″ ceiling is "wider than the stamp", and it is *not* purely a quality cut.** Measured on
  the 2,143 probe galaxies above it: 25–100″ is 98.7% of them and they are **real** — redshift
  falling monotonically with radius (0.023 → 0.007) at r ≈ 13.5–14.5, featured fraction steady at
  ~0.66 — big nearby disks, correctly measured, simply too large for a 256 px (101″) cutout. Only
  past 100″ (27 objects, to 258″ = 651 px) does the signature invert into deblending failure:
  fainter, more distant, featured collapsing to 0.41, star-or-artifact vote quadrupling to 0.20.
  So the ceiling **also declines a genuine population** — bright nearby spirals the encoder will
  not see in pretraining. Accepted, because an uncontained galaxy teaches a truncated shape, but
  it is a selection consequence, not a free win. The probe corpus keeps them, flagged
  (`petrorad_suspect`), and only the Petrosian-radius nuisance control excludes them.
- **The magnitude/SNR shift is structural and accepted** (KS ≈ 0.78 / 0.76). GZ2 labelled
  essentially every bright, well-resolved SDSS galaxy, so *unlabelled* nearly means *fainter*.
  Verified: even discarding the rarest 5% of the probe distribution, a strictly matched corpus
  caps at **146,552** — smaller than the probe corpus itself. It is *impossible*, not merely
  inconvenient.
- **Why that is acceptable:** the shift runs in the **favourable transfer direction** (train
  faint/noisy → probe bright/clean); normalisation absorbs much of it (relative structure
  survives, absolute flux does not); the pilot cleared AUC 0.905 under a *worse* mismatch (no size
  floor at all); and the brightness/SNR **nuisance probes exist precisely to test** whether the
  representation encodes these rather than morphology.
- **Write-up stance:** a **known limitation leaning on the existing controls** — *not* a matched
  design. Do not overclaim parity.

---

## D7 — Canonical probe — *decided (scratchpad): L2 logistic*

- [x] **L2-regularised logistic** is canonical; **mean-difference (CAV)** is the
  robustness check (and their disagreement is itself an entanglement signal).

---

## D8 — "Reliable" label filter — *SUPERSEDED: run unfiltered, frozen at the defined minimum*

> The original decision and its correction are kept below **as the record of what was
> reversed**, not as current guidance. The live decision is "D8 superseded" at the end of
> this section.

- [x] **Reuse v1 vote-agreement filter (mean + 2σ)** ☑

**Recommendation:** reuse v1's agreement filter for **general probe label
quality**, but note it is **separate** from the uncertainty-geometry protocol —
which deliberately uses the **consensus-extremes** split (train on v>0.8 vs v<0.2,
test on the held-out 0.2–0.8 middle) and must *not* be pre-filtered in a way that
removes the ambiguous middle it needs to test on.

That separation survives scrutiny, and it is worth saying why: this filter removes poorly
**sampled** galaxies, not ambiguous ones. A 50/50 split on 60 votes is thoroughly ambiguous but
well measured, so it passes the filter and remains available to the uncertainty test — which is
exactly the galaxy that test exists to use.

### D8 correction — the method transfers, the value does not

An earlier version of this record said *"v1 only applied a 0.5 threshold; the mean+2σ filter is
net-new"*. **That was wrong**, and it is corrected here rather than quietly edited: v1's
dissertation §5.2.1 specifies mean+2σ ≈ 21 votes, and the 0.5 was its *binarisation* threshold
in `__to_binary` — a separate mechanism. v1 used both, and `schemes.derive_vote_count_min`'s
citation was right all along.

**But the number does not carry over.** v1 computed 21 on the PyPI `galaxy-datasets` release;
this corpus is a direct SciServer pull with different vote counts. Re-deriving the same method
here gives **≈36.6 per question**, or ≈300 taken over `total_votes` — an order of magnitude
apart depending on which distribution it is applied to, which is itself a sign the heuristic is
doing less work than it appears to.

So `DEFAULT_VOTE_COUNT_MIN` is **removed**. `vote_count_min` is now a **required** field with no
default anywhere in the code, and `headline=True` is refused until a `VoteCountFreeze` pins it —
the same posture as the effect floor, because it is the same kind of call. A known-wrong default
sitting in the path of every result is worse than no default.

**The value itself is still open**, and two things should shape it:

- A **standard-error framing** is more defensible than either heuristic. The filter exists
  because a vote fraction from few votes is a noisy estimate, and `SE ≈ √(p(1−p)/n)` makes that
  explicit: ±0.11 at n=21, ±0.08 at n=37. "Include galaxies whose vote fraction is known to ±X"
  justifies itself on its own terms and converts cleanly to a count, with no appeal to v1.
- **Per-question beats one global number.** The tree funnels — t11 is reached only by spirals —
  so a single floor either over-filters the deep questions or under-filters the shallow ones.

Measured reach on the 230,358-galaxy corpus, for the questions this decides
(`artifacts/` recount, vote-count floor → galaxies reaching the question):

| question | ≥5 | ≥21 | ≥37 |
|---|---|---|---|
| t09 bulge shape | 33,956 | 7,108 | **829** |
| t10 arms winding | 66,999 | 29,738 | 8,844 |
| t11 arms number | 66,998 | 29,721 | 8,834 |

At ≥21 the per-**bucket** positives are what bite: t09 boxy 100, t11 4-arms 229, t11 >4-arms
271. Scheme 1's deep per-bucket tests are underpowered at any threshold in this range, and
raising the floor to 36.6 makes that strictly worse — which is the trade the choice has to
weigh, not a reason to keep 21.

### D8 superseded — the filter is withdrawn, and that is the decision

**This is a reversal with a reason, not a value being filled in.** D8 above said to reuse v1's
mean+2σ agreement filter and left only the *number* open. The filter itself is now withdrawn, and
the floor runs **unfiltered**. A bare `vote_count_min = 1` would read as an oversight, so the
reasoning is recorded here and, verbatim, in the freeze artefact that stamps every result.

> v1 needed the mean+2σ filter because v1 **trained on the labels** — vote noise flowed through
> the loss and bent the encoder weights, so noisy galaxies had to be excluded up front. v2 breaks
> that coupling: the encoder never sees a label. The filter's original purpose does not transfer.
>
> Label noise in a **probe target** is conservative: it attenuates measured association toward
> chance and cannot manufacture a direction. A feature clearing the gate despite unfiltered
> labels is therefore a **stronger** result, not a weaker one. The existence null is computed on
> the same labels, so the comparison stays like-for-like.
>
> Filtering costs power precisely on the features the paper is about.

**The value is 1, not 0 — the minimum at which the fraction is *defined*.** A question nobody
answered has a 0/0 fraction and no measurement to probe. On this corpus that is load-bearing
rather than pedantic: **GZ2 stores an unreached question's fraction as a literal `0.0`, not as a
blank.** 118,962 of the 230,358 probe galaxies (51.6%) carry
`t09_bulge_shape_a26_boxy_fraction = 0.0` meaning *never asked*, byte-identical by value to
*asked, nobody said boxy*. Nothing downstream can tell them apart: `binary_label` computes
`fraction >= 0.5`, so they would enter as negatives silently — not as NaN, which at least would
be visible. The vote floor is the **only** thing standing between that and the probe, which is
why "unfiltered" means 1 and not 0.

**A defect found while closing this, and fixed.** `eligible_ids` summed `spec.count_col` — the
*single answer's own* count — for binary specs, while graded specs already summed every answer. A
fraction's denominator is the question total, so the per-answer version filtered on the numerator
and discarded the **well-defined zeros**: at a floor of 1 it would have dropped 72.6% of t09
boxy's eligible galaxies, 85.6% of t11 4-arms and 87.2% of t11 >4-arms — almost all the
negatives, leaving probe sets of nearly nothing but positives. It also disagreed with D8's own
published reach table above, which the fixed version now reproduces exactly at ≥5 / ≥21 / ≥37.
`FeatureSpec.reach_count_cols()` is the one denominator, for every kind.

**Reach and per-bucket positives at the frozen floor** (full population, binarised at 0.5):

| bucket | ≥1 (frozen) | ≥5 | ≥11 | ≥21 | ≥37 |
|---|---|---|---|---|---|
| *reach* — t09 bulge shape | **111,396** | 33,956 | 17,837 | 7,108 | 829 |
| *reach* — t10 arms winding | **134,688** | 66,999 | 46,565 | 29,738 | 8,844 |
| *reach* — t11 arms number | **134,684** | 66,998 | 46,574 | 29,721 | 8,834 |
| t09 boxy | **7,894** | 302 | 125 | 100 | 33 |
| t10 tight | **60,809** | 24,436 | 15,239 | 9,136 | 2,649 |
| t10 medium | **49,519** | 23,210 | 16,808 | 11,533 | 3,727 |
| t10 loose | **22,541** | 8,534 | 5,874 | 3,902 | 1,162 |
| t11 1-arm | **8,265** | 2,233 | 1,264 | 572 | 91 |
| t11 4-arms | **1,075** | 258 | 246 | 229 | 113 |
| t11 >4-arms | **2,801** | 286 | 274 | 271 | 163 |

The brief's "t09 boxy: 302 at ≥5, 100 at ≥21, 829 at ≥37" mixed one reach figure into a positives
series; **33** is the positives count at ≥37, and the corrected series runs the same way and
further — 302 → 100 → 33 against 7,894 unfiltered.

**Stated honestly: most of the extra reach is shallow.** 89.8% of t09 boxy's positives at this
floor rest on ≤2 votes, median question total 1; t11 >4-arms is 88.5%, t11 4-arms 73.9%. Those
are *unreplicated human judgements*, not fabrications — a fraction of 1/1 is one person's real
answer with n=1 — so they attenuate toward chance rather than inventing a direction, exactly as
the reasoning above requires. What turns that from an assertion into a measurement is the sweep.

**The sensitivity sweep is pre-registered, and it is a robustness check, not a selection step.**
Recorded here and in `VoteCountFreeze.sweep` **before any results exist**: the ladder is re-run at
**{1, 5, 11, 21, 37}** and the agreement across them reported as a stability claim. The headline
threshold is **fixed in advance at 1** and **must not be revised on the basis of which threshold
produces better results** — the same discipline the pre-registered hard gate carries.
`ProbingConfig` refuses a headline value that is not one of its own registered sweep points, so
"chosen in advance" is checkable rather than asserted.

**Frozen like the others.** `VoteCountFreeze` carries value, sweep, `derived_from`, `frozen_at`,
`frozen_by` and the rationale; it is a `RunConfig`, so it is hashed into `config_hash` and
stamped on every artefact, and a record disagreeing with the live value is refused at load.
`headline=True` was already gated on it. With this frozen, the **effect floor is the only one of
the five still open**.

**Still separate, and deliberately not acted on here.** Vote count is *not* merely noise for the
uncertainty geometry — see the open item in `TODO.md`.

> **Scope — recorded 2026-09-23 (Brief S3, from Brief R2's measurement).** The reasoning above,
> that shallow votes *attenuate toward chance rather than inventing a direction*, holds for
> **readout**: a probe's AUC or direction fitted to noisy labels. It **fails for any measurement
> that bins, conditions on, or projects against the vote fraction.**
>
> Two facts break it. First, low-reach galaxies pile onto quantised fractions (1/4, 1/3, 1/2):
> conditional questions reach a median of only 5–8 volunteers. Second, reach tracks the parent
> answer. So equal-width fraction bins hold *different mixes of populations*, and E[z | f] bends
> from the mixture. Edge-on shows it: both vote-reach halves are straight (bend 0.02 and −0.05), and
> only their mixture is curved (0.30). 11 of R2's 20 curved paths are this. It is a **conditioning
> artefact, not attenuation**: noise that moves which galaxies share a bin does not wash out, it
> manufactures structure.
>
> **Scope of D8, therefore:**
> - The unfiltered population, frozen at 1, remains right for **verdicts**: existence, rungs,
>   readout.
> - A measurement that conditions on the fraction needs a **reach floor, chosen locally** for that
>   measurement and pre-registered with it.
> - The right population depends on the measurement, not on a global setting. Cross-reference: the
>   uncertainty-geometry vote-count item in `TODO.md` ("decide a LOCAL floor"), and
>   `artifacts/r_findings.md` §R2.
---

## D9 — Confidence usage — *decided (scratchpad): both, kept separate*

- [x] **Both axes, strictly separate.** Confidence may be a **probe target**
  (regress / weight by vote fraction) *and* the **uncertainty-geometry test**,
  but never the same axis — recovering a vote fraction along an axis trained on it
  is a tautology. The two uses inform the **probe and the evaluation, never the
  encoder**.

---

## D10 — Symmetry — *REVISED (Brief S4): no rotation/reflection augmentation across the encoder family; divergence recorded; orientation measured*

> The original decision and the divergence note are kept below **as the record of what was
> revised**. The live decision is "D10 revised" at the end of this section.

- [x] ~~**Rotation/reflection augmentation first** (simplest, derisks the minimal
  run); **E(2)-equivariant ViT as a later ablation** (it reshapes encoder
  geometry, so establish the vanilla-ViT ladder before baking in symmetry).~~

> **Spec/code divergence — recorded 2026-09-22 (Brief R).** This decision was never implemented.
> Pretraining applies **no rotation or reflection augmentation**: every run on the probe ladder —
> J, the D18/D21 λ arms, M — learned from stamps in their native orientation. Nothing here reverses
> D10; the gap is recorded so it is not mistaken for a choice.

**D10 revised (2026-09-23, Brief S4).** **No rotation or reflection augmentation, across the whole
encoder family** (M and every D12 baseline). The divergence stays recorded, and orientation is
**measured** rather than removed. The E(2)-equivariant ViT stays as the later ablation D2
already lists. Four reasons:

1. **Retraining M would invalidate every result on record:** Briefs J–S, the frozen effect floor
   (D22), and the 30-seed untrained bank's pairing with this architecture and split (D23). None of
   them could be carried across.
2. **Orientation is measured, and mostly architectural.** Brief R3: a linear readout recovers the
   doubled position angle at ridge R² **0.42–0.44 from random weights** and **0.505 from M**.
   Training adds about 0.07. Augmentation would have to remove what the patch embedding sees for
   free, not just what M learned.
3. **The orientation angle is not a morphology confound.** Nothing about a spiral's arms, a bar or
   a bulge depends on which way the stamp was cut. **Elongation** is a confound, and that is the
   axis ratio, already handled by D13 (`expAB_r` / `deVAB_r`). Brief S1 found that even
   inclination explains only 3–9% of spiral's bend.
4. **D12's cross-objective comparison needs ONE symmetry policy across all encoders.** A
   difference between JEPA, MAE and MoCo must be a difference of objective, not of which images
   each saw.

**Forward constraint, for the baselines brief (flagged in `TODO.md` against D12). Decided there,
not here.** Standard MoCo uses **horizontal flips**, which *are* reflections. MoCo needs
augmentation to function at all, so its other augmentations (crops, colour, blur) are
objective-intrinsic and acceptable. **Flips must be removed**, or the MoCo arm becomes
reflection-invariant where M is not. That would be an asymmetry in exactly the comparison D12
exists to make. The MAE recipe is to be checked for the same (flip in its default pipeline).

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
   **correct physics**, not a representation limit.

   **Adjudicated in TWO SEQUENTIAL STAGES.** `DECISIONS.md` and `galaxy-jepa-spec.tex` previously
   named different adjudicators for this confound, which read as a contradiction. They are not
   competing; they are consecutive, and the order matters because stage two is meaningless
   without stage one:

   * **Stage 1 — is the association representational, or real in the data?** The
     eigen-triangulation's causal cross-check: conditional recoverability under matching
     (`entanglement.adjudicate_pair`). If the bar direction vanishes when winding is held
     constant, the association lives in the world, not in the representation.
   * **Stage 2 — given a real correlation, is it astrophysics or a labelling artefact?**
     Invariance to angle and depth, plus the literature. This stage only runs on a pair stage 1
     returned `world_correlation` for.

   **Stage 2 now has a directional anchor rather than a judgement call.** Hart et al. measured
   spiral arms in strongly barred galaxies as roughly **4–6° looser** than in unbarred ones. That
   converts "check the literature" into a prediction the encoder can be held to
   (`entanglement.bar_winding_alignment`):

   * the bar direction leaning towards **loose** winding specifically, in the order
     loose > medium > tight → **tracking physics**;
   * the bar direction sitting equally close to **every** spiral answer → **confident-
     classification bleed**, which is precisely what the hard case warns about: a galaxy
     confidently called barred is a galaxy confidently called everything.
   * separated but in the contrary order → reported as measured, not explained.

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

## D15 — What a run's identity covers — *decided (signed off)*

**Fork.** `config_hash` hashed the whole `HarnessConfig`, `pretrain_dir` / `probe_dir` /
`out_dir` included. Moving the probe corpus onto the external SSD would therefore have
restamped every run — a different hash for identical science.

**Decision.** Split the config into *where* it ran and *what it was*, and hash only the
latter.

- `PathsConfig` (`paths:`) holds the three directories and is named in
  `RunConfig.NON_DETERMINING`, so `determining_dump()` drops it before hashing. A location is
  neither necessary nor sufficient for data identity; `RunStamp.data_snapshot` already hashes
  the object-id set, which is.
- `RuntimeConfig` (`runtime:`) holds `device` and **is** hashed. A backend is not a location:
  MPS, CPU and CUDA differ numerically, so they must hash apart. `device: null` resolves to
  the concrete backend *before* hashing, or two backends would collide on one hash.
- A **deny**-list, not an allow-list: a new field is hashed by default, so the failure mode is
  a spurious "different run", never a false "same run".
- The stamped hash carries a scheme marker (`STAMP_SCHEME = "v2:"`) so a v1 hex can never be
  quietly compared against a v2 one. It is applied in `RunStamp.create` and **not** inside
  `config_hash`, because `data.cache.pipeline_hash` reuses `config_hash` as the fp16 cache
  *directory name* — prefixing there would force a full re-bake through the parity lock.

**Landed.** `core/config.py` (`NON_DETERMINING`, `determining_dump`, `STAMP_SCHEME`,
`RunStamp.device`), `harness.py` (`PathsConfig`, `RuntimeConfig`, `with_resolved_device`),
`configs/pretrain.yaml` renested — `extra='forbid'` makes an un-renested config a loud
load-time error, which is the intended crossing of the hash-scheme boundary. Pinned by
`tests/test_provenance_identity.py`, including the re-bake guard on `pipeline_hash`.

---

## D16 — The normalisation statistic is an artefact, not a per-run computation — *decided (signed off)*

**Fork.** `harness._build_pipeline` fitted the statistic on every run and persisted nothing. It
was seeded, so it read as reproducible — but the subsample is `rng.choice(len(source), n_sample)`
and `len(source)` went from 10,000 to 826,968, so the same seed began drawing an entirely
different sample. The stamped `config.json` recorded `norm_sample: 8000` — the *instruction* to
fit — not the constants fitting produced. `runs/slice/config.json` still does: replayed today it
fits different numbers, lands in a different `pipeline_hash`, silently re-bakes, and stamps the
same `config_hash`. The statistic is the parity lock across the pretraining corpus, the probing
corpus and every baseline; a lock that re-derives itself per run is not one.

**Decision.** Fit it **once**, on **valid pixels only**, over the **whole pretraining corpus less
a 0.1% heaviest-stamp trim**, and pin it as a `FrozenChoice`.

- **`NormalisationFreeze`** carries mean/std, corpus, `n_sample`, `stretch_q`,
  `valid_pixels_only`, the detector rule, the full trim rule, a content hash over exactly those,
  and the usual `frozen_at` / `frozen_by` / `rationale`. Being a `RunConfig` it is hashed into
  `config_hash` and written to every artefact, so a result now says where its normalisation came
  from.
- **The harness cannot fit.** `_build_pipeline` takes the record and has no fitting path;
  `harness.py` no longer imports `fit_normalise`. **No escape hatch**, unlike `effect_floor`: a
  run that fitted its own statistic and stamped the forfeit would still have broken parity with
  every other run, so the forfeit would be unpayable. A missing record, a `Q` mismatch, or a
  hand-edited value each raise at load — including a value merely *rounded*, which is how the
  guard first proved itself.
- **Valid pixels only.** Cutout padding is exactly-constant and sits at 0, dragging every mean
  down 1.71% and deflating every σ by 0.83–0.84% (`docs/spec/data.md` §1.2–1.3).
- **The whole corpus, not a subsample.** `n_sample=8000` could not pin the number — two disjoint
  halves disagreed by 4.54%, and the curve is a clean 1/√n that nothing reachable clears. Every
  stamp is counted, `n_sample` is removed from the config, and no run can ask for a draw at all.
- **The trim is a degree of freedom, so it is pinned like one.** The census statistic still failed
  the gate: 11.0% of 200 disjoint halves disagreed past 1%, because the variance is carried by a
  minority (the heaviest stamp alone holds 0.090% of the corpus's ch0 sum-of-squares, 745× its
  uniform share). Stamps are ranked by **one** scalar — total valid-pixel sum-of-squares across
  all channels — and those above the 99.9th percentile are dropped **from the fit only**: 827
  stamps, 0.100%, pinned by a sha256 over the excluded IDs. One scalar and not a per-channel cut,
  or different stamps would leave different channel statistics and the channels would stop being
  comparable. Every stamp stays in the corpus, the cache and training. With the trim the gate
  passes cleanly: median 0.314%, worst 0.861%, **0.0% breaching**.
  It is **not** gate-passing: constants are a preprocessing transform, not a population parameter
  needing an unbiased estimate, and a σ inflated by outliers compresses the typical galaxy's
  post-normalisation range. Bright stamps should extend past ±1; that is what ±1 is for.
- **The gate had to be fixed before it could be trusted.** The first version took the worst of 20
  half-splits — a noisy estimator of a tail, which read 0.95% on one run and 1.70% on the next and
  so decided by luck. Same 1% tolerance, stable estimator: 200 splits, gated on the *fraction*
  that breach (≤5%).

**Landed.** `data/validity.py` (new), `data/cache.py` (`fit_normalise` over valid pixels,
`NormaliseFit` carrying the naive statistic alongside as the evidence), `data/transforms.py`
(`NormalisationFreeze`), `core/config.py` (`FrozenChoice` lifted here), `harness.py`
(`normalisation:` replaces `norm_sample:`), `configs/pretrain.yaml`. The fit has one honest
window, `artifacts/e5_fit_normalisation.py`, which refuses to emit a record unless the corpus
halves agree. Pinned by `tests/test_normalisation_freeze.py` and by a shipped-config check in
`tests/test_configs_load.py`.

**Flagged, not acted on.** The 827 trimmed stamps are not bright galaxies or saturated stars but
**low-SNR stamps concentrated in a few bad SDSS imaging runs** — 67.4% from run 1000 alone, which
is trimmed at 111× the corpus rate. No GZ2 probe galaxy comes from those runs, so the probing
corpus is untouched. Recorded in `docs/spec/data.md` §1.3 as a data-quality finding.
---

## Summary — decisions and their state

All of D1–D15 are now resolved. The table records what was chosen.

| # | Fork | Decision |
|---|---|---|
| D1 | Framework | **PyTorch** |
| D2 | Backbone | **Clean ViT-S/16 default** (masking-clean); backbone sweep (ViT→CCT/CvT→E(2)) is a rung confound control; CCT fallback if corpus thin |
| D3 | Env / Python version | **uv + devcontainer; Python 3.11** |
| D4 | From-scratch vs warm-start | **From-scratch** |
| D5 | Masking | **Bounding-box-biased** (`docs/masking.md`) |
| D6 | Pretraining vs probing corpus | **Decouple** — pretrain on large unlabelled SDSS, probe on GZ2 ~250k (both single-survey) |
| D8 | Reliable-label filter | **SUPERSEDED — run unfiltered.** v1 needed it because v1 trained on the labels; v2's encoder never sees one, and probe-target noise is conservative. Frozen at **1**, the minimum where a fraction is defined (GZ2 stores an unreached question as a literal 0.0). Sweep {1,5,11,21,37} pre-registered as robustness, not selection |
| D12 | Cross-objective baselines | **All trained on the same SDSS corpus** (MAE = reproduce Wu & Walmsley recipe on SDSS; Euclid MAE is reference only) |
| D12 (sub) | Contrastive choice | **MoCo** (SDSS-trained) — explicit negatives = clean contrast vs JEPA; established galaxy baseline (Hayat) |
| D13 | Confound taxonomy + inclination | **Axis ratio (b/a) as the non-circular inclination proxy**; taxonomy is the Framing-B interpretive layer, held pending results |
| D14 | Feature scope | **Two schemes as configs on one harness** (full-37 first, then reduced); conditional population as a **comparison**, never a mask; BY family per-scheme |
| D15 | Run identity | **`paths` excluded from `config_hash`, `runtime` kept in**; deny-list; stamped hash carries a `v2:` scheme marker (never the cache key) |
| D16 | Normalisation statistic | **Fitted once as an artefact, never per run** — valid pixels only, whole pretraining corpus less a 0.1% heaviest-stamp trim (fit only); `NormalisationFreeze` hashed into `config_hash`; refitting refused, no escape hatch |

**Still open** (tracked in `docs/galaxy-jepa-spec.pdf` §Open questions register, not re-litigated
here): the graded-axis existence test (D14); the effect-floor *value*; tie-handling in the
existence p and the permutation test; the consensus-gate and vote-count thresholds; which axis
ratio per population (D13).

## D17 — The learning-rate schedule is the reference recipe scaled to this batch, not I-JEPA's literal numbers — *decided (measured)*

**Fork.** `train_jepa` applied **warmup only**: `lr * min(1, (step+1)/warmup)`, clamped at the peak
for the remaining 49,900 steps. No cosine decay, no weight-decay ramp. The reference (I-JEPA,
Assran et al. 2023, arXiv:2301.08243) uses both, and — the part that matters — its published peak
of **1e-3 is a batch-2048 learning rate**. This project runs batch 32. Borrowing the number without
the batch borrows nothing but the digits.

**The scaling argument, written out.** For batch *B* against reference *B*<sub>ref</sub>:

| rule | equivalent peak at batch 32 | configured 1e-3 is above it by |
|---|---|---|
| linear, `lr x B/B_ref` | 1.563e-5 | 64.0x |
| square-root, `lr x sqrt(B/B_ref)` | **1.250e-4** | 8.00x |

**Square-root, not linear.** Linear scaling is derived for SGD with momentum (Goyal et al. 2017),
where the parameter update is proportional to the gradient, so shrinking the batch by *k* shrinks
the gradient noise and the step together. AdamW normalises each coordinate by a running second
moment, so its step size does not track gradient magnitude the same way and the linear rule
over-corrects. Square-root is the conventional choice for Adam-family optimisers. This is the
reason 1.25e-4 was chosen — **not** that it produced the nicest trace.

Two further numbers follow from the same source by the same logic:

* **Warmup 1,250 steps.** The reference warms over 15 of 600 epochs = **2.50%** of its schedule.
  Ours was 100 of 50,000 = 0.20%, twelve and a half times shorter *in relative terms*. 2.50% of
  50,000 is 1,250.
* **Cosine floor 1.25e-7.** The reference decays peak -> peak/1000 (1e-3 -> 1e-6). Scaling **both**
  endpoints by the same square-root factor preserves that 1000x range; scaling only the peak would
  silently compress it to 125x and change the recipe's shape while appearing to follow it.

**Not adopted: the 0.04 -> 0.4 weight-decay ramp.** The reference has it. H2's `wd_ramp` arm was
indistinguishable from baseline (erank 4.10 vs 3.75; both crossed the old floor at the identical
step 125), and 500 steps cannot speak to a regularisation schedule anyway. Bundling an unmeasured
change with two measured ones would make the outcome unattributable. Weight decay stays constant at
0.04. This is the one place the recipe deliberately departs from the reference.

**Decision.** `objective.lr: 1.25e-4`, `warmup_steps: 1250`, `lr_final: 1.25e-7`, weight decay
unchanged. `JepaConfig.lr_final = None` means warmup-only, so the field lands **inert** and an
unchanged config keeps its `config_hash`.

**Evidence — a controlled comparison ending in the objective, not a diagnostic.** H5
(`artifacts/h5_findings.md`, `runs/h5/`): two arms, 3,000 steps, one seed, identical initial
weights, identical data order, identical per-step mask seeds, `steps=50000` fixed so the EMA ramp
could not move. Both frozen encoders probed on the same 34,829 held-out galaxies.

| | old recipe | D17 recipe |
|---|---|---|
| frozen-probe AUC (consensus) | 0.9043 `[0.8988, 0.9097]` | **0.9358** `[0.9315, 0.9402]` |
| AUC, all held-out | 0.8084 | **0.8420** |
| AUC, ambiguous middle | 0.6358 | **0.6624** |
| effective rank, final / min | 7.91 / 3.50 | **11.77 / 7.59** |
| mean pairwise cosine | +0.984 | **+0.286** |
| latent MSE at step 3,000 | **0.0164** | 0.3168 |

No confidence interval overlaps on any of the three AUCs.

**The loss inverted the answer, and that is the finding to carry forward.** The old recipe was
**19x better on loss** — on both framings, which agree at 3,000 steps — and lost the objective
decisively. Latent MSE is measured against a moving EMA target, so a predictor and target that
co-adapt onto a large shared mean component score beautifully while encoding little; the old
recipe ends with mean pairwise cosine +0.984, embeddings 98% aligned. **Low latent MSE is a
collapse signature, not a score.** Anyone selecting on loss alone would have kept the worse recipe.

**What this does not establish.** The cosine decay is **untested**. Because the schedule is the
real 50,000-step one rather than a compressed proxy, the LR is still 99.7% of peak at step 3,000 —
so H5 tested the **peak and the warmup**, and the decay is carried on the reference's authority
alone. It is the weakest-supported third of this decision and should be revisited from the first
full-length run.

**Consequences, recorded.**

* **`CollapseFloorFreeze` is re-derived** (same commit). Not merely because the recipe moved, but
  because H5 **falsified** the old value: the baseline arm sat below the 5.0 soft floor for 53
  consecutive readings from step 125 and still reached AUC 0.9043. Had the 5,000-step grace
  elapsed, the frozen criterion would have killed a working run. New soft floor 2.5, now bounded
  *from above* by evidence (below 3.50, the lowest rank yet seen in a run that probed
  successfully) rather than derived as a fraction of a working level — weaker grounding than what
  it replaces, and flagged as such in the freeze's own `rationale`.
* **The pilot's status changes.** Its AUC 0.905 was measured under the pre-D17 recipe on 10,000
  stamps seen ~19x each. It is an **existence proof** that the premise works, and is no longer a
  like-for-like baseline for anything. H5's own baseline arm (0.9043 on 827k at 3,000 steps)
  reproduced it closely, which is what makes the comparison a fair fight rather than a straw man.
* **The beta sweep gets stronger.** beta = 0 is the published-I-JEPA control. Under the old recipe it
  differed from the reference in beta *and* in three schedule respects (peak, relative warmup,
  decay). Under D17 it differs in beta and the weight-decay ramp alone.
* **Every H-series artefact predating this decision was produced under the old recipe** and is
  stamped with the old `config_hash` (`157903bd5180788b...`; D17 moves it to `538bf997880a8767...`).
* **The cosine decay is adopted but UNTESTED.** H5's arms ran 3,000 steps, where the LR is still
  **99.7% of peak** — so what was measured is the peak and the warmup, and the decay rides on the
  reference recipe's authority alone. It is the weakest third of this decision. Deliberately *not*
  tested with a compressed proxy: a proxy over 3,000 steps answers a different question (how a
  steep decay behaves early), and the real schedule will be exercised by the full run. Carry the
  limitation into the write-up rather than manufacturing a number for it.

---

## D18 — SIGReg is on, at the paper's λ — *decided (measured)*

**Fork.** H5 established that this project's training loss runs *against* the objective: the arm
19× better on latent MSE lost frozen-probe AUC decisively, because latent MSE against a moving EMA
target rewards a predictor and target co-adapting onto a shared mean component. LeJEPA
(Balestriero & LeCun, arXiv:2511.08544) claims SIGReg — a distribution-matching penalty pushing
embeddings towards an isotropic Gaussian, via the Epps–Pulley statistic on random 1-D projections —
fixes exactly that. Brief I ablated it **on I-JEPA**, everything else held. Full measurements in
`artifacts/i_findings.md`; the rule was committed before any arm ran (`artifacts/i_decision_rule.md`).

**Decision: adopt, at λ = 0.05.** `config_hash` moves `538bf997880a8767` → `b5acc6779df49070`.

| | d17 (λ=0) | sigreg_050 | sigreg_006 |
|---|---|---|---|
| consensus AUC | 0.9358 [0.9315, 0.9402] | **0.9470** [0.9435, 0.9506] | **0.9471** [0.9433, 0.9508] |
| all held-out | 0.8420 [0.8376, 0.8467] | **0.8558** [0.8516, 0.8603] | **0.8573** [0.8529, 0.8616] |
| ambiguous middle | 0.6624 [0.6524, 0.6720] | 0.6734 [0.6637, 0.6829] | 0.6760 [0.6661, 0.6854] |
| erank / std / cosine | 11.77 / 4.03 / +0.286 | 34.23 / 0.996 / +0.114 | 29.87 / 1.255 / +0.116 |
| loss @3000 | **0.3573** | 0.4285 | 0.4028 |
| steps/s | 1.4790 | 1.4765 | 1.4782 |

Both SIGReg arms separate from the comparator on consensus (+0.011) and all-held-out
(+0.014 / +0.015); the ambiguous middle overlaps and reads as indistinguishable. Cost: **0.17%**.

### λ = 0.05 is a **provenance** tie-break, not a performance one

0.9470 against 0.9471 is not a difference; the two arms are statistically indistinguishable on
every read-out. The tie is broken on where the number comes from. **0.05 is stated verbatim** in
§6.1 — *"we thus recommend to use λ = 0.05, V_g = 2, V_l = 8, and batch size ≥ 128 as starting
points"*. **0.00625 was extrapolated** from figure 8's view-count trend, below its plotted range,
by this project. Between two equal measurements, take the one whose provenance is a statement
rather than a reading-off.

**The λ-robustness finding is itself a result worth reporting.** An 8× difference in weight
produced indistinguishable AUC, and near-identical geometry (rank 34.2 vs 29.9, std 1.00 vs 1.26).
That **corroborates the paper's stability claim** on a corpus, architecture and objective it did
not test, and it is the strongest evidence that the AUC gain is SIGReg rather than a tuned
coefficient. It is a positive replication and should be reported as one.

### Where the penalty attaches — the most consequential judgement in the implementation

LeJEPA regularises the **encoder output**, and for LeJEPA that is the same tensor a linear probe
reads: encoder output, prediction input and probed representation all coincide. **Here they do
not.** `DEFAULT_LAYER = -2`, so `encode()` — what `extract_embeddings`, the frozen probe and
`CollapseMonitor` all read — is the **penultimate block, pre-norm, mean-pooled**, while the context
handed to the predictor is the **final block, post-`norm`**. Two different tensors, two blocks and
a normalisation apart.

When an identity in the source breaks, the right move is to follow the **intent**, not the literal
line. The intent of SIGReg is to shape the distribution that downstream probing consumes — the
paper's whole derivation (§3) is about which embedding distribution minimises worst-case
downstream risk. So the penalty goes on the probed tensor: pooled `DEFAULT_LAYER`, pre-norm.

Two consequences follow, both wanted. The collapse monitor's rank, std and cosine now describe
*the constrained object*, so the diagnostics and the constraint agree about what they are talking
about. And the final block and `norm` stay **unconstrained**, free to specialise for the pretext
task — the penalty shapes the representation without dictating the prediction head.

Attaching at the encoder output instead would have regularised a tensor two blocks from the one
measured, and a null result would then have been uninterpretable: unclear whether SIGReg does not
help, or simply does not reach. **This was chosen, not tested** — final-block post-norm, and
per-token rather than pooled, are separate arms that were not run.

### The seven deviations, and what they bound

| # | Here | The paper | Bounds the conclusion how |
|---|---|---|---|
| 1 | **Predictor + EMA target retained** | LeJEPA removes both | **Substantive.** This is not LeJEPA; it is I-JEPA plus SIGReg. Says nothing about LeJEPA's own claim that the predictor and teacher–student machinery become unnecessary. |
| 2 | **One context view**, bbox-biased multi-block masking | DINO multi-crop, V=8 | **Substantive.** λ's recommended value is tied to view count (§6.1), so the transfer of 0.05 to V=1 is an assumption the ablation could not test — and the λ-robustness result is the only reason to think it is safe. |
| 3 | Latent MSE against EMA targets | views-predict-global-mean | **Substantive.** The prediction term is a different objective, so the *balance* λ strikes is not the paper's balance. |
| 4 | **Attached at the penultimate block, pre-norm** | encoder output | **Substantive.** Argued above; not ablated. A different attachment could give a different number. |
| 5 | batch 32 | recommends ≥ 128 | Their Epps–Pulley bias is O(1/N) and stated fine "as small as 16"; measured floor here ≈ 1.0 at n=32. |
| 6 | Real-arithmetic ECF | complex ECF | None. The target CF is real, so the modulus expands exactly; verified against a transcription of algorithm 1 at 0.0e+00 across four scales. |
| 7 | Host-drawn sketch, 1024 slices | device generator, listing defaults 256 | None. 1024 is §6.1's own recommendation; host drawing makes the directions backend-independent. |

Taken together: **this decision is about SIGReg as a regulariser on I-JEPA at one attachment point,
on this corpus.** It is not a reproduction of LeJEPA and does not claim to be.

### Question 2 — the paper's loss-selection claim **did not transfer**

The second reason for running this was checkpoint selection: LeJEPA reports its training loss
correlating ρ_s ≈ 0.99 with downstream accuracy, which would make label-free selection possible.
**It did not transfer to this configuration.** Six checkpoints per arm, probed on a fixed reduced
subset; exact two-sided permutation p over all 720 orderings, where the 5% threshold at n=6 is
|ρ| = 0.886:

| arm | ρ(total loss, AUC) | p | ρ(prediction) | ρ(penalty) |
|---|---|---|---|---|
| `d17` | **+0.657** | 0.175 | +0.657 | — |
| `sigreg_050` | +0.086 | 0.919 | −0.143 | −0.314 |
| `sigreg_006` | −0.086 | 0.919 | −0.086 | **−0.714** (p 0.136) |

`d17` reproduces H5's anti-correlation *within a single run*; both SIGReg arms sit at ≈ 0. So the
misleading signal weakens — but **no correlation here is significant**, and removing a misleading
signal is not supplying a good one. The practical read is unambiguous: **selecting the lowest-loss
checkpoint costs AUC in every arm** — **−0.0110** (`d17`), **−0.0072** (`sigreg_050`),
**−0.0181** (`sigreg_006`). SIGReg roughly halves the cost at λ = 0.05 and makes it worse at
0.00625.

**This is not evidence against the paper.** Different regime in every respect that matters: their
ρ is measured *across runs over a hyperparameter sweep* (learning rate, weight decay, epochs, λ)
on ImageNet with eight views; ours is *within one 3,000-step run* at one view and one λ, over six
points. Those are different quantities, and ours is underpowered by construction. What can be said
is only that the claim did not reproduce here, in the form the 1C checkpoint rule would need.

**Therefore the 1C label-blind checkpoint rule STANDS, unchanged** — final checkpoint, gated on
collapse stability. Best-AUC selection would let labels choose the encoder and breach the firewall;
lowest-loss selection is measurably worse than taking the last checkpoint. Nothing here licenses
changing it.

### The collapse floor is **scoped**, not removed

Under SIGReg effective rank is constraint-satisfied rather than diagnostic — 34.23 against 11.77,
monotone, and still climbing at 3,000 steps. `CollapseFloorFreeze.min_effective_rank` (2.5, itself
re-derived only last brief on grounding H5 had already undercut) could therefore never fire, and a
criterion that cannot fire is decoration rather than a tripwire.

So the **soft** rank floor does not apply when `sigreg_lambda > 0`, and applies normally when it is
zero. `CollapseMonitor` takes `soft_rank_floor`, `train_jepa` derives it from `sigreg_lambda` —
which is hashed into `config_hash`, so a run cannot quietly forfeit the tripwire without its
identity changing — and logs the fact. The **hard floor** (erank < 2) and the **std floor** still
apply under SIGReg, and the whole mechanism is intact for **D12's non-SIGReg arms** (MAE, MoCo,
plain I-JEPA), which still need a collapse gate and do not get isotropy for free.

### The eigen-triangulation: one leg weakened, measured rather than argued

The logistic-vs-CAV cross-check compares a *discriminative* concept direction against a *marginal*
one; SIGReg constrains the **total** covariance, not the within-class covariance, and since
Σ_total = Σ_within + Σ_between, Σ_within = σ²I − Σ_between is **not** isotropic. The algebra
predicts the check should weaken without vanishing. Measured over each arm's six checkpoints:

| arm | disagreement, step 500 → 3000 |
|---|---|
| `d17` | 0.9804 → 0.9479 |
| `sigreg_050` | 0.9738 → **0.9118** |
| `sigreg_006` | 0.9691 → 0.9142 |

Enforced isotropy does bring the two definitions closer, and does **not** collapse them: at 0.912
they still disagree on 91% of the available angle. **The leg survives, weakened, as predicted.**
The **Marchenko–Pastur null becomes better justified** — its isotropy assumption is now enforced
rather than hoped for.

### D12 — flagged, not settled

Adopting SIGReg gives the JEPA arm a distributional constraint MAE and MoCo lack. Both sides are
real, and neither is decided here:

* **Against.** "Of course its geometry differs — you constrained it" becomes a fair objection to any
  cross-objective *consistency* argument built on embedding geometry. The arms are no longer
  like-for-like in the respect the comparison reads.
* **For.** Entanglement **surviving enforced isotropy** is *stronger* evidence it is in the data
  than entanglement in an unconstrained representation, where it could be an artefact of an
  arbitrary covariance. A constraint that fails to remove a structure is informative about the
  structure.

This is a framing decision for the write-up, not a config one. Options when it is faced: run the
JEPA arm both ways, restrict the cross-objective claim to non-geometric read-outs, or state the
asymmetry and argue the second point. Record both; decide later.

### What is not known

3,000 steps against a 50,000-step run — all three arms were still moving, and `sigreg_050`'s rank
climbed 29.9 → 34.2 over its last 500 steps. One seed, one feature, everything stamped
`smoke: true`, so "separated" means separated on bootstrap intervals over the test set, not across
training runs. The attachment point was chosen, not ablated. And **D17's cosine decay remains
untested** and is inherited unchanged by this decision.

---

## D19 — The sky/noise control (3C-5) is a diagnostic, not a null — *decided (measured; corrects a pre-registered gate)*

**This changes a pre-registered gate.** It is written down at length because of that: the bar
moved *after* a measurement existed, which is exactly the situation where a reader is entitled to
ask whether it moved because it was wrong or because it was inconvenient. The reasoning is below,
and it does not depend on which features pass.

**The defect.** `nulls.five_null_samples` took the elementwise maximum over all five 3C controls.
Four of them break something and are therefore *chance-calibrated* — they answer "what AUC does
this machinery reach when the thing being measured is absent?":

| control | what it breaks | what it kills |
|---|---|---|
| 3C-1 shuffled vote fractions | the image–label correspondence | "the probe exploits label marginals" |
| 3C-2 random embeddings | the representation | "any high-D vector predicts this" |
| 3C-3 noise through the real encoder | the images | "the encoder imposes structure on anything" |
| 3C-4 untrained encoder | the **pretraining** | "the probe, not the pretraining, did the work" |

3C-5 breaks nothing. Real images, real frozen encoder, real probe — and a *different real label*.
Its AUC is not "what chance looks like"; it is **how much image-quality content the representation
holds**. Folding it into a maximum asks a morphology probe to beat a nuisance probe before the
morphology feature is allowed to exist. That is a category error, not a bug: no value of it is
evidence about whether morphology is a direction in the representation.

**The proof that it is the same measurement twice.** 3C-5 and the `snr` nuisance probe came out
**bit-identical** on all six features probed at J4 — 0.8373, 0.8381, 0.8416, 0.8416, 0.8416,
0.8355. One measurement, entered once as a bar and once as a diagnostic. Both read `snr_r` through
a median split on the same eligible ids; they are the same code path with two names.

**What it did.** Measured on the 50,000-step encoder (`artifacts/out/j4_spread_controls.json`):

| feature | real AUC | 3C-5 | strongest chance-calibrated null | verdict, old bar | verdict, corrected bar |
|---|---|---|---|---|---|
| t01 featured-or-disk | 0.8365 | 0.8373 | 0.7908 (untrained) | **fails** | clears by +0.0457 |
| t02 edge-on yes | 0.7320 | 0.8381 | 0.6368 (untrained) | **fails** | clears by +0.0952 |
| t10 arms tight | 0.5740 | 0.8416 | 0.5474 (untrained) | **fails** | clears by +0.0266 |
| t10 arms medium | 0.5161 | 0.8416 | 0.5160 (untrained) | **fails** | +0.0001 — a tie |
| t10 arms loose | 0.6098 | 0.8416 | 0.5645 (untrained) | **fails** | clears by +0.0453 |
| t09 bulge boxy | 0.5534 | 0.8355 | 0.5359 (untrained) | **fails** | clears by +0.0175 |

**State it plainly: every feature failed existence under the broken bar, featured-ness included.**
The designed ladder would have returned an all-R3/R4 catalogue — a catalogue that reads like a
scientific null and is nothing of the kind, on an encoder whose headline feature probes at 0.9278
consensus. **That near-miss is why this is documented rather than quietly applied.** A gate that
cannot be passed by a real effect is not conservative; it is broken, and its output is
indistinguishable from an honest negative result.

**The decision.** The existence null is the max over the **four chance-calibrated** controls.
3C-5 stays computed, stays on `FeatureControls`, stays in the reported record, and is adjudicated
where it belongs: **3D-ii's triggered matched evaluation**, the machinery built precisely to ask
whether a morphology axis is really a nuisance axis. In the artefact its key is renamed
`sky_noise_diagnostic`, so `ladder_summary.json` cannot read a diagnostic back as a bar.

**What this does not do.** It does not make the nuisance problem go away — it relocates it to the
machinery that can answer it. The J4 panel is severe: magnitude 0.8733, size 0.8501, SNR 0.8373,
redshift 0.7918 against featured-ness at 0.8365 and every other morphology feature below 0.74. On
this evidence matched evaluation fires for **every** feature, not for a flagged few, which is a
scope change for 3D-ii recorded in `TODO.md`. Nor does it close the register's degeneracy item
(item 8): the untrained-encoder singleton still exceeds every resampled draw, so the combined null
still has zero variance and the existence *p* can still only be 1/(n+1) or 1.

**Naming.** `five_null_samples` → `existence_null_samples`. A function that combines four controls
must not be named for five; the count belonged in the name only while the count was the claim.

- Code: `probing/nulls.py`, `probing/controls.py`, `probing/ladder.py`, `probing/run.py`.
- Spec: `docs/galaxy-jepa-spec.tex` §3C + open-questions item 8, `docs/probing-harness-design.md`
  §3C, `docs/spec/gates.md`. **The PDF is stale** — no LaTeX toolchain in this environment; the
  `.tex` is the corrected source and the PDF needs a rebuild.
- Test: `tests/test_probing_nulls.py::TestGroundedStatistics
  ::test_the_sky_noise_control_is_a_diagnostic_and_never_sets_the_bar`.

## D20 — An ablation's horizon must reach the regime the decision will run in — *decided (measured; the lesson, not the recipe)*

This entry is about **how a decision was made**, not about which λ is right; D21 carries that. It
is written separately because the two are independent: the process lesson holds whatever the next
measurement says, and burying it inside a recipe change would let it be forgotten the moment the
recipe changes again.

**The unconditional form.** D18 was decided on a **3,000-step** ablation and applied to a
**50,000-step** run. At 3,000 steps neither arm had passed its peak. A 3,000-step ablation could
not see the regime the decision would run in, and no amount of care *within* that horizon could
have fixed it — the evidence was sound and the extrapolation was the gap. **An ablation's horizon
must reach the regime the decision will run in.**

**The D18-specific verdict, which Brief L's branch 1 supports.** L2 ran the λ=0 arm to 10,500
steps — the comparison D18 never had, because λ=0 had never been run past 3,000. On
`t01_consensus`, held-out, same split and protocol:

| step | λ=0 | λ=0.05 | leader |
|---|---|---|---|
| 3,000 | 0.9358 | **0.9470** | λ=0.05, by 0.0112 — **D18's evidence** |
| 6,000 | 0.9458 | 0.9448 | level, +0.0010, intervals overlapping |
| 10,500 | **0.9554** | 0.9412 | λ=0, by 0.0142, intervals cleanly apart |

**The crossover lies between 3,000 and 6,000 — immediately past the horizon D18 measured to.**
D18 did not misread its data. Its data were right and stopped one regime short. That is the
failure mode worth naming: not carelessness, but a horizon chosen for what it cost rather than for
where the decision would land.

**What follows in practice.** A recipe ablation is quoted with its horizon, and a recipe adopted
for an N-step run is not licensed by evidence from a run much shorter than N. Where the full
horizon is unaffordable, the ablation says so and the adoption is provisional **in writing**,
rather than the shortfall living in the reader's head.

**The same trap is live right now, and D21 says so rather than repeating it.** The λ=0 evidence
reaches 10,500 steps; the headline run is 50,000. Adopting λ=0 for a 50,000-step run on
10,500-step evidence is D18's error at a longer lever arm. D21 is therefore written as a
provisional adoption with the missing measurement named.

**Source.** `artifacts/l_findings.md` (L2); `artifacts/out/l1_mlp_ladder_l2.json`,
`artifacts/out/l1_mlp_ladder_full.json`; the λ=0 arm at `runs/l2/d17`.

## D21 — SIGReg is the cause of the probe decline; λ goes back to 0, provisionally — *decided (measured; reverses D18 at a longer horizon)*

**The finding.** Brief L's L2 ran λ=0 to 10,500 steps against the λ=0.05 trajectory on the same
split, probe config and seed, with only `sigreg_lambda` differing. **λ=0 rises monotonically,
+0.0361; λ=0.05 falls monotonically, −0.0065.** By 10,500 λ=0 reaches 0.9554, higher than λ=0.05
reaches anywhere on its own 50,000-step trajectory (best 0.9477). The second feature agrees:
+0.0962 against +0.0426, with λ=0 ahead by 0.0355 at 10,500 and intervals apart.

The decay, batch size, EMA schedule and recipe are held identical across the arms. **SIGReg is the
cause of the decline**, not the cosine decay — which K2 had already falsified on timing — and not
the recipe at large.

**What L1 established alongside it**, and why it rules out the gentler reading: the information is
not merely becoming less linearly accessible. An MLP probe declines *more* than the linear probe
across the λ=0.05 trajectory (−0.0238 against −0.0199), and the nonlinear headroom collapses
+0.0081 → +0.0042. Effective rank rising 24.3 → 57.6 was **not** the same information spread
thinner; information is leaving.

**The decision: `sigreg_lambda: 0.05 → 0.0`.** This reverts D18. Of the three candidates —
accumulate embeddings across steps to raise the statistic's effective sample; reduce λ; disable
SIGReg — only the third is **supported by a measurement at the required horizon**. The first two
are hypotheses about *why* SIGReg hurts, and adopting either would repeat the D18 pattern of
choosing a recipe from reasoning rather than from a run long enough to test it.

**Provisional, and what would settle it (D20).** The λ=0 evidence reaches 10,500 steps; the
headline is 50,000. λ=0.05's decline was invisible at 3,000, so nothing here proves λ=0 has no turn
of its own later, and its effective rank is still climbing at the last reading (8.1 → 18.6). **The
measurement that would settle it is a λ=0 arm at the headline horizon**, and until it exists this
adoption is provisional in writing rather than in someone's memory.

**What this does not claim.** Not that SIGReg is wrong in general — LeJEPA's result stands in its
own regime, at batch 2048 and eight views, against this project's batch 32 and one view. The
live hypothesis remains that the Epps–Pulley statistic is badly estimated from 32 samples in 384
dimensions, which is the risk I1 pre-registered. That hypothesis is now **worth testing with the
accumulated-embedding estimator**, and it is a next experiment, not a next adoption.

**A second measured difference, recorded because it affects how the ladder reads.** On 3 of 4 λ=0
`t01` checkpoints the MLP's shuffled-label control clears the selectivity threshold at *every*
width, so no admissible nonlinear reading exists; across all eight λ=0.05 checkpoints it never
fired. λ=0 embeddings carry std ≈ 3.1–4.6 against λ=0.05's ≈ 0.9–1.0. The effect fades as λ=0
trains — by 10,500 the ceiling no longer fires. **MLP readings are not comparable across the two
arms**, and the ladder's R3 rung will behave differently under the reverted recipe.

**Source.** `artifacts/l_findings.md` (L1, L2); `runs/l2/d17`;
`artifacts/out/l1_mlp_ladder_{full,l2}.json`.

## D22 — The effect floor is frozen at 0.7267, absolute — *decided (measured; closes the last of the five statistical gates)*

**What was open.** The floor separates *clean* from *marginal* among effects that have already
passed existence. D-series and spec 3B fixed its mechanism; the **number** was a scientific call,
and J5 declined to make it on four objections. Two are now closed: the bar was wrong (3C-5 was in
it; D19 removed it) and the encoder was wrong (J's was measurably worse than step 3,000 of its own
trajectory). Brief M produced a settled encoder and Brief N re-ran the battery on it.

**The decision: `effect_floor = 0.7267`, form (a), absolute.** Frozen in `configs/probe.yaml` with
an `EffectFloorFreeze` carrying its provenance, exactly as the normalisation statistic, the vote
floor and the collapse floor are frozen. `headline=True` is now loadable; `effect_floor_open` no
longer stamps onto artefacts.

**Chosen by structure, not by which features it admits.** Every value in **(0.6538, 0.7995]** gives
an *identical* partition of the six probed features — a band 0.146 wide, bounded below by
arms-loose and above by edge-on. 0.7267 is its centre, the point furthest from any feature's flip
(0.0729 either way). The old placeholder 0.6500 sat 0.0038 from a flip and the pooled null ceiling
0.7908 sat 0.0087 from one: both are coincidences rather than thresholds. It admits featured-ness
(0.8845) and edge-on (0.7995); it excludes arms-loose (0.6538), arms-tight (0.5889), bulge-boxy
(0.5847) and arms-medium (0.5226). **All six clear their own binding null**, so all six still pass
*existence* — the floor is doing the job 3B gives it and no other.

**The margin form was measured and rejected.** J5's third objection pointed at it: an absolute AUC
cannot encode a null that varies per feature, and on J's encoder the binding null spanned 0.275.
Four grounds against, the first decisive:

1. A feature's null ceiling is **exactly** the supremum of `existence_null_samples` — verified
   against the production function over 200 random cases. So a margin floor is a **tightening of
   existence**, not a second gate. That is the one property 3B disclaims when it grounds the floor
   as acceptable *"because it no longer does the existence work"*.
2. It reorders the catalogue by a random network's luck. Bulge-boxy has a *lower* real AUC than
   arms-tight (0.5847 vs 0.5889) but a *higher* margin (+0.0489 vs +0.0415); at margin 0.10 it
   excludes **featured-ness**, the strongest feature in the catalogue, while admitting edge-on.
3. Its verdicts move with the untrained seed (the admitted set flips at margin 0.05 and 0.09). An
   absolute floor cannot move with the seed, because it never reads the null.
4. `effect_floor` has five consumers; `ladder.py:121`, `:161` and `:248` have no defined "the
   feature's own binding null" to take a margin over.

**A definition this makes explicit rather than leaving implicit: the existence bar is
architecture-determined.** `controls.untrained_encoder_matrix` never sees the trained checkpoint, so
the bar is a property of *(architecture, seed, galaxies)* alone — M's untrained nulls came back
identical to J's to four decimals on all six features. The untrained singleton dominates every
resampled draw (0.5160 against 0.5143 at the narrowest), so `existence_null_samples` has zero
variance and **the existence test reduces exactly to `real_auc > untrained_encoder_auc`**.

**Two limitations travel with the value and are to be reported, not hidden.**

* **J5's fourth objection is unresolved.** n = 6 cannot locate a threshold for a catalogue of 37,
  and the widest gap is a fact about which six features were chosen. The band argument softens it —
  any point inside (0.6538, 0.7995] costs nothing *here* — but does not remove it.
* **t10 arms-medium's verdict is seed-dependent**: its margin over the binding null is +0.0065 at
  untrained seed 0 and +0.0006 at seed 1. No floor fixes that. It is honest evidence that some
  features sit below what this apparatus resolves.

**What this does not license.** No rung verdicts. The nuisance panel on M's encoder has every
nuisance beating five of the six morphology features, and magnitude (0.9033) and size (0.9061)
beating even featured-ness — so matched evaluation must be read before any ladder is published. A
floor is not a licence to run the ladder; it is one of the preconditions for doing so.

**Also recorded:** the floor cannot serve as the matched-evaluation survival bar. Production passes
`survive_threshold=config.effect_floor`, and four of the six features sit below 0.7267 *unmatched* —
they would fail a floor-based gate by arithmetic whatever matching did. Brief O1 therefore reads
matched evaluation against each feature's own unmatched margin instead. If matched evaluation is
ever wired into the ladder, that interaction needs deciding, not inheriting.

**Source.** `artifacts/n_findings.md`; `artifacts/out/n1_spread_controls.json` (M's 4-epoch
encoder, three untrained seeds), `artifacts/out/n2_floor_evidence.txt`; drivers
`artifacts/n1_controls.py`, `artifacts/n2_floor_evidence.py`.

---

## D23 — Existence is tested against the untrained bar across K seeds, not a point-mass null — *decided (Brief P; changes how a pre-registered gate computes existence)*

**What was wrong.** `nulls.existence_null_samples` combines four chance-calibrated controls by a
per-draw elementwise maximum, and two of the four are per-feature *constants* that floor every
draw. Brief N measured the consequence on all six probed features: the untrained-encoder singleton
exceeds the shuffled and random-embedding maxima every time, narrowest gap 0.5160 against 0.5143.
So the null is a **point mass**, `existence_pvalue` can only return `1/(n+1)` or `1.0`, and
existence reduces exactly to `real_auc > untrained_encoder_auc`.

That is survivable for a six-feature spread and **not** survivable for Scheme 1. Benjamini–Yekutieli
over a 37-test family needs a p-value it can order and threshold; a two-valued one gives it nothing
to act on. A 37-feature catalogue without working multiplicity control is precisely the failure BY
was chosen (D-series decision 3) to prevent: near-bar features clearing by chance.

**Why an empirical seed null does not fix it.** The obvious repair — build the null from untrained
AUCs across many seeds — hits the *same wall*. BY's rank-1 threshold at family 37 is
`0.05 / (37 · H₃₇)` = **3.216 × 10⁻⁴**, so an add-one estimator needs **≥3,109 draws**
(`nulls.required_null_draws`, pinned in the tests). At ~576 s per untrained extraction that is
≈41,000 GPU-hours. The 3,109 figure is a property of the **add-one estimator**, not of existence.

**The decision.** Existence under `existence_method: untrained_z` is

```
H0_f :  AUC_f <= C_f
z = (AUC_real − mean(C)) / sqrt( sd(C)² + se_real² ) ,   p = t.sf(z, df = K−1)
```

where `C` is the untrained-encoder bar measured across **K = 30** seeds and `se_real` is the
bootstrap SE of the real AUC. Continuous in `AUC_real`, so BY applies normally and no resolution
floor arises.

**Both uncertainties enter, and that is the point.** The bar is *not* a constant: Brief N2 measured
per-feature ranges of 0.0026–0.0209 across three seeds (median 0.0073). Treating it as known would
call features significant on a margin narrower than the bar's own seed-to-seed movement.

**Student t with df = K−1, not the normal.** The denominator's `sd` is estimated from K samples, and
BY's rank-1 bar is a ~3.4σ statement. At df = 29 the t quantile there is 3.70 against the normal's
3.41 — the normal would be optimistic in exactly the tail the correction cares about, and the
conservative direction is the right one for a gate.

**What is given up.** A distributional assumption, where the add-one estimator had none. This is the
weakest joint in the construction and is not hidden: normality is **tested and reported per feature**
(Shapiro–Wilk and QQ data at K = 30), and where the untrained tail is non-normal that is stated as a
limitation on those features' p-values.

**The resolution requirement moved; it did not disappear.**
`nulls.assert_untrained_bank_resolution` refuses a bank below `K_MIN = 20`, because K is what every
z-denominator's `sd` is estimated from (the relative SE of an sd estimate is `1/√(2(K−1))` — 16% at
K=20, 13% at K=30). `assert_null_resolution` is untouched and still raises under its own method.
A gate that stops biting once its neighbour is satisfied is not a gate; both are pinned by tests.

**Scope.** The library default stays `empirical`, so the switch must be declared by a config a run
actually loads — the same posture as the effect floor. The untrained bar never touches a trained
checkpoint (`controls.untrained_encoder_matrix` builds from the model constructor record and a
seed), so the bank is a property of (architecture, seed, galaxies) and is **reusable across
encoders**: M's untrained nulls matched J's to four decimals.

**Checked against real margins before adoption** (bank sd 0.010, K = 30): t01 featured-or-disk
p = 1×10⁻²⁷; t09 bulge-boxy p = 8.7×10⁻⁵, clearing BY's rank-1 bar by a factor of 3.7; t10
arms-medium p = 0.90 — failing independently, in agreement with its seed-dependent existence and its
collapse under all six of O1's matched evaluations.

---

## D24 — Matched survival is retention of the margin, and the effect floor never tests anything relative — *decided (Brief S; changes how a pre-registered gate judges nuisance clearance)*

**What was wrong.** The ladder's nuisance gate judged "survived matching" as *matched AUC ≥ the
effect floor* (`ladder.py`, `survive_threshold=config.effect_floor`). An answer already below 0.7267
unmatched fails that whatever matching does. In Brief P that was **21 of the 22** answers labelled
"confounded", and matching had moved their AUC by a median of 0.021. Brief R0 re-judged all 37
under O1's pre-registered retention rule: **none** of the 22 loses its effect. P's "size dominates"
headline was the gate's arithmetic.

**The principle, recorded because this is the second time.** The effect floor (D22) is an
**absolute clean-vs-marginal threshold** for a real effect. Its one legitimate consumer is R1's
`existence.clean`. It must **never** test anything *relative*: a margin, a retention or a change.
The power rule was the first instance, and was corrected to a margin over the bar (`65b904c`,
`nulls.resolvable_margin`). The matched-survival gate is the second.

**The decision.** The ladder's nuisance clearance applies **O1's retention rule, unchanged**
(`matching.retention_verdict`):

```
SURVIVES  : (M − C_m) ≥ 0.5·(A − C)  AND  M's CI lower bound > C_m
COLLAPSES : (M − C_m) ≤ 0            OR  M's CI contains C_m
PARTIAL   : between the two
UNRESOLVED: < 500 matched test galaxies, or < 10% of the unmatched test set
```

- A is the unmatched AUC. M is the matched AUC on the ladder's own matched rows (the worst
  nuisance, `nuisance_valid` rows, 5 quantile strata), with a 2,000-resample CI.
- **C** is the untrained bar on the full rows, the mean of **K = 3** untrained draws (seeds s,
  s+1, s+2). **C_m** is the same **K = 3** draws re-measured on the **same matched rows**.
- This is Brief R0's construction, kept so the ladder reproduces R0 exactly.
  `ControlEmbeddings.untrained_extra` carries the further draws, and the ladder **refuses** fewer
  than `matching.RETENTION_SEEDS` rather than silently computing a one-draw bar. One untrained draw
  decided a verdict under D23.
- A feature clears the nuisance gate iff no nuisance is competitive **or** retention is SURVIVES.
  PARTIAL, COLLAPSES and UNRESOLVED do not clear.

**The margin floor reuses an existing gate: D23 existence.** Retention is a ratio of margins,
(M − C_m)/(A − C). Where the unmatched margin is not itself established, the ratio divides noise by
noise. Brief R hit this on `t10 winding: medium`: margin 0.003 against a 0.006 spread of the bar
across seeds, which read COLLAPSES linearly and SURVIVES with an MLP. So **retention is judged
only where the unmatched margin has passed D23's existence test; otherwise the verdict is
UNRESOLVED**. That is a statement about the evidence, not the nuisance, and it adds no parameter.

*Does D23 cover the noise-over-noise case?* For the denominator, yes. Passing existence means
A − mean(C₃₀) exceeds √(sd(C)² + se²) at the family-corrected bar: the margin is established
against both the seed spread and the real AUC's sampling noise. **One residual, declared:**
retention's C is a K = 3 mean, not the K = 30 bar existence uses. Its own error, about
sd/√3 ≈ 0.006 at the measured sd ≈ 0.010, can move the ratio by up to ~18% for a feature sitting
exactly at the BY rank-1 bar. It is far less for R0's full-population survivors, every one of which has a margin at
least 2× its seed spread. The CI condition on M guards the numerator's sampling noise but not
C_m's seed noise. If a later brief needs tighter, C can take the K = 30 bank mean. C_m cannot,
since the bank holds full-row AUCs only, so it would need its own bank on matched rows. The
alternative (a multiple of the 30-seed sd as a floor) was not adopted: it duplicates what D23
already decides, with a new free constant.

**Mechanism labels name the gate that failed.** The label used to call every cleared-but-not-clean
answer "entangled linear". Once matching stopped failing below-floor answers by arithmetic, that
would have relabelled twenty of them with a second wrong reason. `_passing_rung` now says, in order:
- *confounded by X (collapsed / partial retention under matching)*
- *nuisance clearance unresolved (X)*
- *entangled linear*
- *present, below the effect floor*
- *present, not selective*

**Not changed here, flagged.** The effect floor still tests something relative in two places:
- **The entanglement leg** (`_entangled_map`) judges the matched cross-check (A's probe matched on
  B's vote fraction) as *matched AUC ≥ floor*. That is a third instance of the same defect, and it
  feeds `adjudicate_pair`, so fixing it can change rungs. It needs its own before/after.
- **The MLP decode threshold** (`rung_from_sweep(..., decode_threshold=effect_floor)`) asks a
  feature that failed existence to clear 0.7267. That is why R3 = 0 is unreadable (Brief P §a).

Both are logged in `TODO.md`. **→ Both fixed in D25 (Brief T1).**

**Verification** (`artifacts/s2_verify.py` → `artifacts/out/s2_ladder.json`). The production ladder
is re-run on P2's union and split, both populations, and compared row by row with R0 and P2.

*Predicted before the rerun, from R0 and P2 alone:*
- Retention verdicts equal R0's everywhere except where the feature failed existence in P. Those
  become UNRESOLVED: **full** `t10 winding: medium` (COLLAPSES →); **conditional** `t04 spiral` and
  `t04 no spiral` (both SURVIVES →). The last two are the same rule, not a surprise: both are R4
  in the conditional population.
- A, C, M, M_lo and C_m reproduce R0 exactly.
- Rung counts are unchanged unless an answer that is now cleared *and* above the floor turns out
  unentangled and selective. That status was hidden by the old label order. Two candidates:
  **full** `t05 bulge: dominant` (AUC 0.7517) and **conditional** `t08 odd: other` (0.7346). If
  either becomes R1, that is reported as the consequence of D24, not suppressed.

*Result (2026-09-23).*
- **Reproduction is exact.** Every judged row matches R0's retention verdict: 33/33 full, 29/29
  conditional. The maximum difference across A, C, M, M_lo and C_m is **0**. K = 3 for both C and
  C_m.
- **The three predicted flips hold** (`s2_verify.py --flips` → `artifacts/out/s2_flips.json`).
  Through the production clearance with the margin floor lifted, each gives R0's verdict (COLLAPSES,
  SURVIVES, SURVIVES). With its real existence status (failed), each gives **UNRESOLVED**. They had
  to be checked directly: P2-era failing verdicts never carried their clearance onto the record, so
  `_failing_rung` now attaches it.
- **Rung counts: full unchanged** (R1 1, R2 32, R4 4). **Conditional: one change, R1 1 → 2.**
  `t08 odd: other` (AUC 0.7346, 467 of 3,483 test positives, resolvable margin 0.041, not
  underpowered) became R1 here. **Superseded by D25:** it is R2, entangled with `merger`
  (Brief T1), and its direction is not one coherent category (Brief T2). It was one of the two candidates named above.
  It retains 110% of its margin under magnitude matching, is not entangled, and is selective.
  Its old "confounded by magnitude" label was the floor arithmetic. The other candidate, full
  `bulge: dominant`, is **entangled**, so it stays R2.
- **Mechanisms, full population.** P's 22 "confounded" are now **12 entangled**, **8 present but
  below the effect floor**, and **2 nuisance clearance unresolved** (thin matched sets). The old
  label order had been hiding the first group. The conditional population keeps one genuine
  confound: `t10 winding: medium`, which passes existence there and COLLAPSES under magnitude
  matching.

## D25 — Every consumer of the effect floor, classified; the two relative ones fixed — *decided (Brief T1; changes how 2A attributes a pair and how the MLP rung is assigned)*

**The rule (D24's principle, now enforced everywhere).** The effect floor (D22, 0.7267) is an
**absolute clean-vs-marginal threshold** for an effect already shown to be real. It must never test
anything **relative**: a margin, a retention, a change, or whether a second probe recovers what the
first did not.

**The audit.** Every consumer, `file:line` at this commit. This is the whole list: N2 counted five
comparisons in 2026-09; these are the same five, and there is no sixth.

*Consumers that test something:*

| # | site | what it compares | class | action |
|---|---|---|---|---|
| 1 | `nulls.py:506` `existence_verdicts` | `clean = significant ∧ AUC ≥ floor` | **ABSOLUTE**, the legitimate one (R1's clean) | kept |
| 2 | `gates.py:62` `build_gates` | the same `AUC ≥ floor`, in the rendered gate tree | **ABSOLUTE** (see note) | kept; name flagged |
| 3 | `ladder.py` `_nuisance_clearance` | was `matched AUC ≥ floor` | **RELATIVE** | fixed by D24 |
| 4 | `ladder.py:202` `_entangled_map` (2A conditional leg) | was `A's matched AUC ≥ floor` | **RELATIVE** | **fixed here** |
| 5 | `ladder.py` `_failing_rung` (MLP decode) | was `MLP AUC ≥ floor` for an existence-failing answer | **RELATIVE** | **fixed here** |

Note on #2: the gate is named `existence` in the tree, but it holds the clean bar. The name misleads
a reader of the tree; the comparison is legitimate.

*Bookkeeping, not tests:*
- `config.py:163–164`: the value and its freeze.
- `config.py:238`: freeze consistency.
- `config.py:278/283`: a headline run requires the freeze.
- `run.py:203`: stamps `effect_floor_open`.
- `ladder.py:684`: passes the floor to #1.

*Not a consumer:* `matching.nuisance_competitive` compares against its own flagged margin
(`nuisance_competitive_margin`, 0.0), never the floor.

*Artefacts (outside lint and CI):*
- `r_nonlinear.py:206/247`: *A < floor*, descriptive, absolute.
- `readme_figures.py:495–534`: draws the floor and buckets P's labels by the *unmatched* AUC,
  absolute. Its "matched < floor" bucket names P's defect and goes when the figure is regenerated.
- `p2_ladder.py:119/158`: print and record.
- `f0_preconditions.py:84–98`: freeze checks.
- Docstrings and prints only: `j4`, `j5`, `n2`, `m2`, `o1`.

`matching.matched_evaluation` and its `survive_threshold` are **removed**: after this entry nothing
judges survival by a threshold. Invariant tests pin it: moving the floor from 0.51 to 0.99 changes no
retention, no pair verdict and no failing rung (`tests/test_probing_ladder.py`, D25 block).

**Fix #4: 2A's conditional leg judges retention.** A is matched on B's vote fraction, and D24's rule
applies unchanged: K = 3 untrained draws for C and C_m, with C shared with nuisance clearance through
one helper (`ladder._retained`).
- SURVIVES → survived.
- COLLAPSES → vanished (world correlation).
- PARTIAL or UNRESOLVED → the leg did not attribute (`survived_matching=None`, so *inconclusive*,
  which marks nothing).
- The margin floor holds by construction, because only existence-passing directions reach a pair;
  the ladder asserts it rather than assuming it.
- The pair verdict now carries its retention state, the retained fraction, M, C_m and the matched
  count.

**Fix #5: the MLP decode is not adjudicated.** "The MLP decodes it" should mean the MLP clears an
existence-style test against an untrained-MLP bar. That bar needs K ≥ 20 untrained MLP fits per
answer (`nulls.K_MIN`); R has K = 3. So the rung is not assigned on a stand-in:
- Every existence-failing answer reads R4, *not recoverable linearly; MLP decode unadjudicated*.
- The sweep and ceiling stay on the record.

**The construction for #5 (TODO, P2; not run).**
- D23's `z = (MLP − mean_K MLP_untrained)/√(sd_K² + se²)`, Student-t with df K − 1, K ≥ 20.
- Width chosen on an inner split, as in R1.
- BY across the answers that reach the rung.
- Nuisance clearance by the MLP's own retention.
- ≈ 7–8 h.
- **It cannot change any current rung.** All 12 existence-failing answers have every nuisance
  competitive, so their clearance is UNRESOLVED (D24) and R3 is impossible whatever the decode says.
  The full-population K = 3 bound agrees: the largest MLP z is ≈ 2.3.

**Verification, pre-registered** (`artifacts/t_findings.md` §T1, hash `aff39c7d`, before the
rerun):
- **Every predicted rung change held.**
  - Full: unchanged, {R1 1, R2 32, R4 4}.
  - Conditional: {R1 2, R2 27, R4 8} → **{R1 1, R2 28, R4 8}**. `t08 odd: other` goes R1 → R2,
    because `other × merger` SURVIVES (1.09) and so the pair is representational.
  - `no bulge` stays R1 (its pair is UNRESOLVED at 272 matched).
  - Nuisance clearance reproduces S2 exactly.
- **Unpredicted:** world correlation collapses from 18 → 0 pairs (full) and 18 → 1 (conditional).
  Every old world-correlation verdict was a matched AUC under the floor. P's "`edge-on ×
  cigar-shaped` is a geometric necessity, correctly labelled world correlation" is withdrawn.
  Separately, `smooth × features` and `odd yes × no` SURVIVE although A is matched on a
  near-complement. I had predicted COLLAPSES or UNRESOLVED for those.

**Standing consequence.** With retention on both legs, entanglement is the common state:
48 of 53 full pairs are representational. Entanglement is now the finding it was designed to be,
not the absence of a floor crossing. That makes the uncertainty-geometry and D13 stage-2 briefs
the place where "entangled with what, and why" gets answered.
