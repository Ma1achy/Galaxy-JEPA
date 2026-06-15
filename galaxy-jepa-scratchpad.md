# Galaxy-JEPA — Scratchpad

*Working title. v2 of Galaxy-Zoo-Classifier. Status: scoping — living document.*
*Rev 4 — incorporates second external review: ladder confounds named, uncertainty-geometry circularity fixed, SSL baselines promoted to controls, nuisance-metadata dependency resolved.*

---

## TL;DR

Build a representation of galaxy images **unsupervised** with a JEPA, then use the Galaxy Zoo labels **only as a read-out key** — to name and test directions the representation already learned, never to train it. The labels cannot reshape the encoder's geometry; they can only *misname or blur a read-out direction*. So the experiment **moves label noise out of representation-learning and into a measurement stage where it's inspectable and controllable** — it doesn't pretend the probe is noise-free. The scientific question, per feature: is the human morphological concept a *linearly nameable direction*, an *entangled* or *nonlinear* one, or *absent* from the image information at this resolution?

---

## Why this, why now

**v1's finding, restated.** Galaxy-Zoo-Classifier showed the labels are the bottleneck: supervised models fit the volunteer votes, so they inherited volunteer *confusion* on the hard questions (bulge shape, winding, arm count). The model confusion matrices mirrored the volunteers'.

**The structural problem.** Supervised training couples two things that should be separate — learning what a galaxy looks like, and fitting the noisy votes. JEPA breaks the coupling: the encoder sees only images; labels enter later, through a probe that can't touch the encoder.

**The arc.** v1's intro motivated everything with LSST — ~150 PB, far more than crowdsourcing can label. SSL on the unlabelled deluge *is* the field's answer. v1 asked "can we match the volunteers?"; v2 asks "can we stop needing them — and use them only to read off what the images already taught the model?"

---

## The core idea — labels as a read-out key, not a training signal

The methodological heart, stated precisely.

The encoder is built unsupervised and **frozen**. It carves the latent space along whatever axes of variation actually exist in galaxy images. **The labels never get to bend those axes** — they're consulted only afterwards, to ask *"which of these pre-existing directions lines up with what humans called a bar / bulge / edge-on disk?"*

The clean separation of failure modes:

- A mislabelled galaxy can make you **misname or blur a read-out direction** — a measurement error: local, inspectable, controllable.
- It cannot **reshape the encoder's geometry** the way it deformed v1's weights — a representation error: global, baked in.

So noise isn't *eliminated* — it's **relocated** from representation-learning to measurement, where it can be quantified and bounded with controls. That relocation is the win: probing here isn't just a cleaner engineering choice, it's a sharper, *auditable* scientific instrument.

> ⚠ "Feature = direction" is a **working hypothesis**, not an assumption. The science is *which* features are nameable directions and which aren't. See the ladder — and the controls that keep it honest.

---

## The nameability ladder

For any feature, "can the labels name it?" has four possible answers — each a *different result*:

1. **Clean linear direction.** A single vector **w** whose projection tracks the feature. The feature is one coordinate of the latent space. *(Bet: holds for the easy features — smooth/featured, edge-on.)*
2. **Entangled linear direction.** The axis exists but isn't orthogonal to others — moving along "bar" drags bulge-size or inclination. Still nameable, but not independent. Measurable via the cosine between concept directions.
3. **Nonlinear manifold.** No single vector captures it (e.g. winding tightness on a curved submanifold). The linear probe fails; a small MLP / k-NN decodes it. *Candidate* for "present but tangled" — **but see the guardrail.**
4. **Not recoverable.** No probe — linear or not — finds it.

> **Guardrail on rung 3.** A nonlinear probe succeeding does **not** by itself mean "feature present but tangled." It may mean the probe found a *spurious correlate* — survey artefact, redshift, apparent size, brightness, SNR, or label prior. Rung 3 is only credible with the controls below. Without them, "an MLP decodes winding" is too weak to claim.

**The diagnosis v1 couldn't give.** Map the confused features onto the ladder:

- Come back as **clean directions** → this would support the strongest version of the v1 thesis: the information was present in the pixels, but the *supervised objective and noisy vote targets* failed to extract it cleanly. (Note: *failed to extract cleanly* — not "purely the labels." Architecture, splits, imbalance, thresholding, augmentation and resolution could also contribute.)
- Need a **(controlled) nonlinear probe** → present but tangled; supervised training never had a chance through noisy labels.
- **Nothing decodes them** → genuinely not in the pixels at this resolution; never a labels problem.

Which rung each confused feature lands on **is the paper**.

### What sets the rung (besides the feature)

The rung is **not an intrinsic property of the feature** — it's a joint property of *(feature × SSL objective × architecture/resolution × probe)*. Four confounds, each with its control:

- **Probe capacity / spurious correlates** → the controls battery (selectivity + nuisance probes). A nonlinear probe can win on size / brightness / survey, not morphology.
- **Probe-target circularity** → train the axis on high-consensus *extremes* only, test on the held-out ambiguous middle (see uncertainty geometry). Regressing the vote fraction and then "recovering" it is a tautology.
- **SSL-objective geometry** → run the *same ladder* on JEPA vs contrastive vs MAE. JEPA's latent-MSE objective may not flatten semantic manifolds the way contrastive's push-pull does, so a Rung-3 result in JEPA could be linear-in-contrastive. Cross-objective comparison separates *intrinsic to the images* from *artefact of this objective*.
- **Patch size / resolution** → 8×8-patch (or higher-res) ablation. The token grid sets the spatial-resolution floor; thin arms / tight winding living below it can read as Rung 4 (absent) when they're really just under-resolved. This is the Rung-4 control, as the baselines are the Rung-3 control.

Attribution is the whole game: a rung only means something about *the images* once these are held down.

---

## Method

### Stage 1 — representation (label-free)

I-JEPA (Assran et al. 2023): context-encoder ViT, EMA target encoder (anti-collapse), predictor; predict masked-block *embeddings* from context, MSE in latent space, **no pixel reconstruction** (this is the axis that distinguishes it from MAE — see prior art). Reuse v1's 256×256 → 256 × (16×16) patchification as the tokeniser. (Galaxy-specific design below.)

### Stage 2 — readout = **measurement** (the heart)

Encoder **frozen throughout** — labels never reach it. Stage 2 is a measurement protocol, not a training stage.

**(a) Define the concept direction — canonically, an L2-regularised logistic probe.**
> *A feature is linearly nameable if an L2-regularised logistic probe on frozen embeddings reaches statistically significant held-out AUC **and** selectivity over control labels. The unit-normalised probe weight vector is the concept direction.*

Logistic is the canonical choice over mean-difference because it gives a proper decision axis, AUC, regularisation, calibration, and class-imbalance handling. **Keep the mean-difference (CAV) vector as a robustness check** — and note they answer slightly different questions: logistic gives the *discriminative* direction (best separating axis given everything else), mean-difference gives the *marginal* one. Their *disagreement* is itself an entanglement signal.

**(b) Walk the ladder.** Linear (logistic) probe first (defines "clean direction"); then MLP / k-NN *only* to distinguish *entangled-but-present* (rung 3) from *absent* (rung 4) — and only with the controls below.

**(c) Confidence does double duty** — informing the **probe and the evaluation, never the encoder** — but the two uses must stay *separate axes* to avoid circularity:
- *As a probe target (one axis):* regress the vote fraction, or weight the probe loss by confidence. Useful, but **this axis cannot be used to test uncertainty geometry** — recovering the vote fraction along an axis you trained on it is a tautology.
- *As uncertainty geometry — the headline (a different axis):* train the concept axis on **high-consensus extremes only** (v > 0.8 vs v < 0.2, binary), then project the **held-out ambiguous middle** (0.2 < v < 0.8) and test whether margin distance ranks their vote fractions. The axis never saw the graded values it's asked to reproduce (see below).

**(d) Fine-tuning** — subordinate comparison only. Unfreezing buys a few points but lets vote-noise bend the encoder again, re-introducing v1's coupling. A contrast, eyes open — **not** the main line.

---

## Concept directions + uncertainty geometry

The uncertainty-geometry result is the one to chase, because of what it would *mean* — and because it's a stronger diagnostic than AUC alone. AUC can be high on a brittle decision boundary; the real question is whether the representation has a **graded morphology coordinate**.

Not just *"does the axis classify bars?"* but *"does distance along the unsupervised axis reproduce the volunteer vote fraction?"* — ambiguous galaxies (votes ≈ 0.5) near the boundary, high-consensus galaxies far out.

**The non-circular protocol (critical).** Fit the axis on **high-consensus extremes only** — a *binary* logistic probe on v > 0.8 vs v < 0.2. The ambiguous middle (0.2 < v < 0.8) is held out of estimation entirely. Then project those unseen middle galaxies and test (Spearman) whether their margin distance ranks their human vote fraction. If an axis trained only on confident examples orders the ambiguous ones by a human uncertainty it never saw, the unsupervised geometry genuinely aligns with the human ambiguity gradient — not by construction.

v1's signature finding was the model's confusion matrices **mirroring** the volunteers' — inherited *because it was trained on the votes*. If the latent geometry reproduces that same uncertainty structure **without ever being trained on the votes**, the meaning flips:

- **v1 reading:** the ambiguity is a labelling artefact the model absorbed.
- **v2 reading:** the ambiguity is a *real property of the images* — Galaxy Zoo uncertainty is partly recoverable as geometry in the image distribution.

Same observation, opposite conclusion — the clean hook. The headline figures:
- **Concept-direction AUC + calibration per feature** — the ladder, quantified.
- **Projection vs vote-fraction, per feature** (Spearman / Pearson) — *uncertainty geometry*.
- **Cosine-similarity matrix between concept directions** — entanglement structure; compare to v1's Fig 18/19 confusion geometry, recovered *from the embeddings*.

> Risk note: uncertainty geometry is the highest-upside result **and** the most likely to come out null. Structure Paper 1 so the ladder + controls + v1 comparison is the solid backbone and uncertainty geometry is the high-beta headline — don't bet the paper on it landing positive.

---

## Controls & guardrails (making the ladder credible)

These are what separate "we measured something real" from "the probe found a shortcut." **Tiered** to protect a small Paper 1.

**Must-have (Paper 1):**
- **Selectivity (Hewitt–Liang).** Real-label probe performance *minus* control-label performance. A direction only counts if it beats a control task.
- **Negative controls.** Concepts that should *not* be clean directions: shuffled vote fractions, random labels, random embeddings, image-metadata bins, sky-background / noise level. The ladder is credible when these *fail*: "bars linear; winding nonlinear; random labels fail; survey identity low after homogenisation."
- **Core nuisance probes.** Can the same embedding-axis machinery read off redshift, apparent magnitude, angular size (and survey, if multi-survey)? If a "morphology" direction is really tracking size or brightness, this catches it.

**Deferred (Paper 2 / if feasible):**
- **Matched evaluation.** Re-test on galaxies matched in redshift / size / brightness, where sample size allows.
- **Cross-split robustness.** Learn the concept direction on one split, test on another; ideally train on SDSS, test on DESI/DECaLS where labels later overlap. (Needs multi-survey → Paper 2.)

The point: a feature is "linearly nameable" only if the logistic direction clears **AUC + selectivity + the nuisance battery**, not AUC alone.

---

## Galaxy-specific design

- **Masking must target the galaxy.** Block-masking a mostly-black SDSS cutout wastes budget on empty sky. Bias mask sampling toward the galaxy — reuse v1's average-image bounding box. **Upgrade:** a **per-galaxy box scaled by the Petrosian radius** (already pulled for the nuisance battery) handles apparent-size variation that a single global box cannot — loose on small/distant galaxies, clipping large/near ones. Full scheme + the β-generalisation (β=0 ⇒ standard I-JEPA) in `docs/masking.md`.
- **Rotational symmetry is a free inductive bias.** No canonical orientation → start with rotation/reflection *augmentation* (simplest, derisks the minimal run). The principled extension is an **E(2)-equivariant ViT** (bakes in continuous rotation/reflection, freeing capacity otherwise spent learning that a rotated galaxy is the same object) — but it reshapes the encoder geometry, so establish the vanilla-ViT ladder first, then ablate. Precedent for physics-baked JEPA: Lens-JEPA beats supervised + vanilla I-JEPA.
- **Data scale / multi-survey** — its own section below. More (and more diverse) data could mean richer axes — *if* survey identity doesn't become a shortcut.
- **Resolution / patch size.** v1 cropped to 224/256. The token grid (256 ÷ 16 = 16×16 tokens) sets the spatial-resolution floor — the model can't attend below a patch, and the patch projection compresses sub-patch structure (not aliasing per se; the patch embed is a linear map, not a resampling filter). Thin arms / tight winding may live below that floor, so an **8×8-patch (or higher-res) ablation** is the Rung-4 control — it distinguishes *absent from the pixels* from *under-resolved by the tokeniser*.

---

## Data / multi-survey

> **Scoping:** this is **Paper 2 / a major ablation, not Paper 1.** Cross-survey transfer is its own deep topic (DeepAstroUDA, AstroCLIP) — keep it out of the first clean result.

**Is "more data → richer representations" even true here?** Mostly yes, but *conditionally*. SSL's premise is that scale and diversity buy generality, and galaxy imaging sits on hundreds of millions of unlabelled cutouts — the regime JEPA wants, and one 200k can't reach. But "richer" isn't automatic: it's gated on the *new* variation being morphological, not instrumental. Add a second survey and if the dominant new axis the model discovers is "SDSS-ness vs DESI-ness," you've spent capacity learning the cameras apart. **That gate is the whole game.**

### What breaks — the homogenisation checklist

Worst-first. Each is a channel through which *which telescope took the picture* leaks into the representation — and JEPA's objective will exploit any leak that makes the masked target easier to predict from context (survey identity always does, since context and target share a survey):

- **Pixel scale (arcsec/pixel).** SDSS ~0.396″, DESI Legacy ~0.262″, HST ~0.05″, JWST ~0.03″. Resizing arrays *looks* like it fixed it but doesn't — each crop then covers a different patch of sky. Hold **arcsec/pixel** constant (or kpc/pixel, which needs a redshift), not pixel count.
- **PSF / sharpness.** Ground-based is seeing-limited and blurry; space-based is diffraction-limited and crisp. A space galaxy is intrinsically sharper than any resampled ground-based one — an almost-perfect survey fingerprint.
- **Bandpass + morphological k-correction.** ugriz vs grz vs near-IR — colour means different physical things across filter sets. At high z the observed bands sample bluer rest-frame light, so a high-z JWST galaxy isn't even the same physical object as a low-z SDSS one in nominally "the same" colours.
- **Depth / SNR / noise texture.** Deeper surveys reveal faint tidal features shallow ones miss; noise statistics differ by instrument.

### The design fork — degrade-down vs preserve-and-correct

- **Degrade-down.** Match everyone to the worst common denominator: coarsest PSF, shallowest depth, largest pixel scale. Conservative, kills most leakage, yields a genuinely comparable corpus. Cost: throws away the resolution of the good data.
- **Preserve-and-correct.** Keep native quality and make the model invariant to the gap — augmentation that spans it (blur / noise / PSF jitter so sharpness can't be a cue), explicit domain adaptation, or trust scale to wash it out. Higher ceiling, far more to go wrong.

**Recommendation: degrade-down first** — because the whole thesis rests on probing being a *trustworthy instrument*, and you can't trust a direction that might mean "barred" vs "the survey where bars were easier to see." Get the comparable corpus and honest diagnostics working; *then*, as a deliberate experiment, relax toward preserve-and-correct and watch leakage and probe quality move.

### The merge as a controlled experiment

The survey-leakage probe turns "did the merge help?" into a measurement, not a hope:

- Pretrain **single-survey vs multi-survey**; probe morphology on the *same* held-out labelled set both ways.
- Read **two** numbers off each: **morphology probe AUC** (representation quality) and **survey-leakage AUC** (instrument contamination).
- **Clean win:** morphology-AUC ↑ *and* leakage-AUC flat/↓. **The trap (and the interesting failure):** morphology-AUC flat/↓ while leakage-AUC ↑ — the model spent the extra data on the cameras.
- Scale to an ablation: plot both AUCs against corpus heterogeneity (single → two close surveys → add a space-based one). DeepAstroUDA's warning applies — cross-survey models tend to extract dataset-specific, non-robust features unless you force otherwise.

### Sequencing

Don't reach for JWST to prove the point. SDSS + DESI Legacy are both ground-based optical with overlapping bands — gap mostly pixel-scale + band-mapping, homogenisation tractable, and if leakage shows up *even there*, cheap lesson. The space-vs-ground jump is experiment two.

---

## Evaluation

- **v1 is the baseline** — same schemes (Hubble / Reduced Set / All Features), same metrics, direct comparability with the existing results table.
- **The ladder, per feature** — logistic concept-direction AUC + calibration → MLP/k-NN selectivity. The core deliverable.
- **Uncertainty geometry** — projection-vs-vote-fraction (Spearman / Pearson) per feature.
- **Embedding correlation geometry** — cosine matrix of concept directions vs v1's Fig 18/19.
- **Controls** — selectivity, negative controls, nuisance probes (see Controls section). Non-negotiable for Paper 1.
- **Label-efficiency curve** — probe accuracy vs #labels, SSL-pretrained vs supervised-from-scratch. Quantifies the value of pretraining (and answers the LSST framing).
- **SSL baselines = a control, not box-ticking.** Run the *same probe ladder* on JEPA vs **MAE** (latent prediction vs pixel reconstruction — Wu et al.'s is public) vs **contrastive (MoCo / SimCLR / BYOL)**. Cross-objective comparison is what separates *intrinsic to the images* from *artefact of how JEPA organises the manifold* — e.g. a feature that's Rung 3 in JEPA but Rung 1 in contrastive is an objective effect, not an image fact.
- **Survey-leakage probe** *(Paper 2 / multi-survey)* — predict source survey from embeddings; drives the merge experiment in **Data / multi-survey**.
- **GalaxyMNIST** (Walmsley, 4-class) — external comparability.

---

## Prior art / landscape

> ⚠ Verify with a proper arXiv sweep — some IDs/details from search snippets. **Do not claim firstness.** Safest framing: *JEPA-style (latent-prediction) SSL appears underexplored for galaxy morphology relative to contrastive and masked-reconstruction approaches.*

**Galaxy SSL — contrastive, and (newer) masked-reconstruction:**
- Hayat et al. 2021 — contrastive (MoCo) SSL on SDSS; downstream morphology + photo-z.
- Stein et al. 2022 — MoCo for lens-finding.
- Walmsley et al. 2022 — *Towards Galaxy Foundation Models with Hybrid Contrastive Learning* (BYOL-style + GZ info; introduces **GZ-Evo**, ~552k labelled images + ~1.34M comparable unlabelled). (arXiv 2206.11927)
- Walmsley et al. 2023 — *Galaxy Zoo DESI*, 8.7M galaxies (arXiv 2309.11425); Zoobot.
- AstroCLIP (arXiv 2310.03024) — cross-modal FM, DESI spectra + Legacy images.
- **Masked autoencoders (MAE)** for galaxies — see Wu et al. below; pixel-reconstruction SSL, the natural foil for JEPA's latent prediction.

**Closest neighbour — interpreting galaxy representations against Galaxy Zoo:**
- **Wu et al. 2025 — *Re-envisioning Euclid Galaxy Morphology: Identifying and Interpreting Features with Sparse Autoencoders* (arXiv 2510.23749).** SAEs on supervised (Zoobot) *and* self-supervised (MAE) embeddings, Euclid Q1; finds interpretable features aligned with **and beyond** the GZ decision tree, stronger GZ-alignment than PCA. Public code + MAE.
- Wu 2025 — *Insights into Galaxy Evolution from Interpretable Sparse Feature Networks* (ApJ 980; arXiv 2501.00089).
- **Our distinction:** (1) **JEPA latent-prediction vs their MAE pixel-reconstruction** — their MAE is the ideal head-to-head baseline; (2) **top-down hypothesis-testing** (probe a *named* human concept, walk the ladder) vs **bottom-up discovery** (read features off an SAE dictionary); (3) **uncertainty geometry** — we test vote-fraction *recovery*, not just label alignment. Complementary, not competing (could run an SAE on the JEPA encoder — see parking lot).

**JEPA — proven adjacent, not (apparently) galaxy morphology:**
- Assran et al. 2023 — I-JEPA. · Lens-JEPA — physics-informed, gravitational lensing (NeurIPS ML4PS 2025); beats supervised + vanilla I-JEPA. · HEP-JEPA — collider physics (arXiv 2502.03933).

**Cross-survey domain gap:**
- DeepAstroUDA (Ćiprijanović et al., ML:S&T 2023) — frames cross-survey morphology as domain adaptation; warns models extract dataset-specific, non-robust features.
- Vision-FM-for-astronomy survey (arXiv 2409.11175) — FM features helped optical galaxy classification, poor on radio. Transfer isn't free.

**Probing / concept-direction lineage (the method's ancestry):**
- Alain & Bengio 2016 — linear classifier probes. · Kim et al. 2018 — TCAV / Concept Activation Vectors. · Hewitt & Liang 2019 — control tasks + selectivity.

---

## Risks / unknowns

- **Spurious-correlate probes (rung 3).** A high-capacity probe can decode a feature via nuisance correlates (z, size, brightness, survey, SNR, PSF), not morphology. *The* reason the controls battery exists; never let an MLP result stand alone.
- **Probe-capacity confound.** A powerful enough probe decodes *anything* (even random labels). Claim strength is tied to probe simplicity: linear+selective = strongest; MLP-only = weaker and must be controlled. Don't let probe capacity smuggle in structure.
- **Representation collapse** — JEPA's standard failure. Budget tuning on EMA schedule + masking ratio.
- **Survey shortcut** *(multi-survey)* — instrument identity as the dominant axis. Caught by the survey-leakage probe; mitigated by homogenisation + close surveys first.
- **Compute + orchestration** — pretraining a ViT from scratch ≫ v1's transfer-learning runs. EMA-schedule and masking-ratio tuning is a *modest* grid for Paper 1, so a lightweight sweep harness suffices — don't build MLOps before the science is derisked. The heavy declarative-sweep machinery (data-degradation × hyperparameter cross-product) earns its keep at Paper 2's survey-leakage ablation, which is exactly the kind of workload Prism is built to orchestrate. Within FMX hardware, but plan it.
- **Null uncertainty geometry** — see the risk note; backbone the paper on the ladder + controls + v1, not on this landing positive.
- **Scope creep** — Stage 1 + frozen logistic probing + must-have controls + the three headline figures is the minimum that answers the question.

---

## Paper shape (two papers, à la Principia)

1. **The clean science — single-survey (SDSS / GZ2).** Unsupervised JEPA; logistic concept directions; the nameability ladder per feature *with must-have controls*; uncertainty geometry; cosine matrix vs v1; MAE + contrastive baselines. Thesis: *which Galaxy Zoo concepts are nameable directions present before any label, which are entangled / nonlinear / absent, and does the geometry reproduce human uncertainty?* **No multi-survey.**
2. **The performance / scaling story** — fine-tuning, confidence-aware readouts, multi-survey scaling + survey-leakage experiment, deferred controls (matched eval, cross-split), GalaxyMNIST.

---

## Open questions / decisions

- [x] **Canonical probe — decided:** L2-regularised logistic; mean-difference (CAV) as robustness + entanglement signal.
- [ ] Probe-capacity control — selectivity / control tasks as standard (lean yes).
- [x] **Nuisance metadata — tractable.** GZ2 → SDSS join via **CasJobs / SkyServer**; z, apparent magnitude, Petrosian radius complete for the GZ2 spectroscopic main sample (it's built on SDSS DR7 main-sample galaxies); SNR / PSF width derivable from the photometric + field tables. *(Confirm exact completeness fractions when pulling.)*
- [ ] Confidence — probe target, validation axis, or both? (Lean both; different questions.)
- [ ] Backbone size — ViT-S/16 to start, scale later?
- [x] **Pretrain corpus — decided: decouple from the probing corpus.** Pretrain on a *large unlabelled SDSS* sample (≫250k); probe on the GZ2-labelled ~250k. Pretraining needs no labels, so there's no reason to cap it at the labelled set — and from-scratch I-JEPA on only ~250k is thin (I-JEPA used 1.3M–14M), the biggest risk to the from-scratch call: an undertrained encoder can't tell "info absent" from "encoder never learned it". Stays **single-survey** (Paper 1); also more on-thesis (the LSST "unlabelled deluge" framing). Multi-survey scaling = Paper 2. (See DECISIONS.md D6.)
- [ ] From-scratch vs warm-start the encoders from an ImageNet ViT? (From-scratch is cleaner.)
- [~] Masking specifics — **proposed in `docs/masking.md`**: bounding-box-biased multi-block with a single bias strength β, a strict generalisation of I-JEPA (β=0 ⇒ standard I-JEPA), so β ∈ {0, 0.5, 1.0} is a free ablation. Open sub-points: per-galaxy **Petrosian-radius-scaled** box as the default vs the global average-image box; and re-tuning the EMA schedule / masking ratio per β, since biasing all targets onto the galaxy shifts the prediction-difficulty distribution (removing the easy, collapse-adjacent sky-prediction task).
- [ ] Symmetry — augmentation first (decided for the minimal run); E(2)-equivariant ViT as a later ablation.
- [ ] "Reliable" labels — reuse v1's vote-agreement filter (mean + 2σ)?
- [ ] Resolution / patch-size ablation — does 8×8 (or higher res) recover winding / arm count? (Rung-4 control.)
- [ ] Cross-objective ladder — JEPA vs MAE vs contrastive, same probes, to attribute rungs to the objective vs the images. (Rung-3 control.)

---

## Parking lot

- **Naming.** Working title only — yours. (The method is interpretability-of-representations.)
- **SAE as a disentanglement instrument (not just a bridge).** Run a sparse autoencoder on the frozen JEPA embeddings: if a linear probe fails but the SAE finds a clean *monosemantic* feature for the concept, that's evidence for *present-but-entangled* (Rung 2/3) — and a **less-confounded discriminator than an MLP**, since a monosemantic dictionary feature is far harder to explain away as a spurious correlate. Also unifies top-down probing with bottom-up discovery and bridges to Wu et al. (cross-check which SAE features align with GZ concepts *and* reproduce vote fractions).
- **Spectra cross-link — central, not a stretch.** Polar decomposition / Jacobian analysis of the JEPA encoder (the Spectra toolkit) is a third, *mechanistic* angle on "what did the representation learn" — beyond probes, the operator structure of the layers.
- **Attention routing (exploratory, with a caveat).** Do specific ViT heads reliably attend arm-patches → bulge-patches — does the network structurally "see" morphology before the probe aggregates tokens? Worth a look, but attention weights are unreliable as explanations ("attention is not explanation"), so treat it as a hypothesis generator, with the Jacobian/Spectra angle as the principled version.
- **Unsupervised clustering gut-check** — UMAP of embeddings coloured by morphology vs by survey. Cheap; doubles as the leakage visual.
- **Stretch: multimodal** — pair images with SDSS spectra/photometry (cf. AstroCLIP). Different project, adjacent.

---

## Next actions

1. **arXiv sweep** — confirm the JEPA-for-galaxy-morphology gap; read Wu et al. 2510.23749 properly (and grab their MAE as a baseline); pin the contrastive baselines + probing/CAV refs.
2. **Pull the data + metadata** — galaxy-datasets / SDSS imaging; and the nuisance battery (z, Petrosian mag/radius, SNR, PSF) via **CasJobs / SkyServer**. Single-survey (SDSS / GZ2) for Paper 1.
3. **Minimal Stage-1 run** — ViT-S/16, 256-patch pipeline, bounding-box-biased masking, pretrain on a *large unlabelled SDSS* sample (probe later on the GZ2-labelled set — corpora decoupled, see Open questions).
4. **Logistic concept-direction probing on the Reduced Set** — AUC + calibration per feature, the cosine matrix, **plus selectivity + negative controls + core nuisance probes** from the start.
5. **First headline figure** — projection vs vote-fraction scatter for a clean feature (bar) and a confused one (winding), with a shuffled-label control overlaid. If the clean one shows the confidence gradient, the confused one doesn't, and the control is flat — the ladder *and* uncertainty geometry are already speaking.
