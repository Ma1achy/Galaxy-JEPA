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

## D8 — "Reliable" label filter — *decided (signed off): reuse v1's mean+2σ method*

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
| D8 | Reliable-label filter | **Reuse v1 mean+2σ** (separate from uncertainty protocol) |
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
