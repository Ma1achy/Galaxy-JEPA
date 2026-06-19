<p align="center">
  <img src="assets/banner.png" alt="Galaxy-JEPA" />
</p>

# **About**

---

Learning galaxy morphology *without labels*, then using the Galaxy Zoo votes only to read off what was learned.

This is **v2 of my undergraduate dissertation** &mdash; a direct follow-on from [Galaxy-Zoo-Classifier](https://github.com/Ma1achy/Galaxy-Zoo-Classifier), which trained supervised CNNs and transformers to reproduce human morphological classifications and found that the models inherited the *confusion* in the crowdsourced labels. Galaxy-JEPA asks what happens if you take the labels out of the learning entirely.

A [JEPA](https://arxiv.org/abs/2301.08243) (Joint-Embedding Predictive Architecture) is trained **self-supervised** on hundreds of thousands of galaxy images: mask out patches, and have the model predict the *representation* of the hidden region from the visible context &mdash; never the pixels, and never a human label. The idea is that to predict a masked galaxy region well, the model has to build an internal representation of what galaxies actually look like &mdash; their shapes, structures and features. The encoder is then **frozen**, and the Galaxy Zoo labels are brought in only as a *read-out key*: to test which human morphological concepts correspond to directions the representation already learned on its own.

> v1's finding was that the bottleneck is the labels, not the architecture &mdash; the models were as good as the volunteers, confusion and all. v2 relocates the label noise *out of representation-learning and into a measurement stage*, where it can be quantified and controlled rather than baked into the weights.

> **Status &mdash; research in progress.** The core premise is proven at pilot scale (see [The First Result](#the-first-result)); the full-scale run and the probing harness are designed and being built. This README is a tour of the project as it stands.

# **The Problem**

---

Sky surveys have already catalogued hundreds of millions of galaxies, and the Galaxy Zoo successfully used crowdsourcing to classify their morphologies. But upcoming surveys make that approach untenable: the Legacy Survey of Space and Time (LSST) alone is expected to produce ~30 TB of imagery *per night*, for a total of ~150 PB &mdash; far more than volunteers can ever label by hand.

Self-supervised learning is the field's answer to that unlabelled deluge: learn from the images themselves, with no labels at all. v1 asked *"can a model match the volunteers?"*. v2 asks *"can we stop needing them &mdash; and use them only to read off what the images already taught the model?"*.

The deeper motivation is v1's central finding. Supervised training couples two things that should be separate: **learning what a galaxy looks like**, and **fitting the noisy votes**. Because the votes are confused on the hard questions (bulge shape, spiral winding, arm count), the model inherits that confusion &mdash; its confusion matrices mirror the volunteers'. JEPA breaks the coupling: the encoder sees only images; labels enter later, through a probe that cannot touch the encoder.

# **The Catch, Revisited**

---

v1 analysed the correlation structure of the votes before training, and found that the questions volunteers *disagreed* on were precisely the ones the models failed on.

<p align="center">
  <img src="assets/v1_correlation_matrices.png" width="900" alt="Per-question vote correlation matrices from v1" />
</p>

<p align="center">
  <em>From v1: per-question vote correlations. "Edge on Disk?", "Bar?" and "Has Spiral Arms?" show clean structure &mdash; volunteers agreed. But "Bulge Shape?", "Spiral Winding?" and "Spiral Arm Count?" show almost none: the votes themselves don't separate these features. A supervised model trained on these labels can only ever reproduce the ambiguity.</em>
</p>

This is the problem v2 is built around. If the ambiguity lives in the *labels*, a label-free encoder shouldn't inherit it. So the question becomes: for each morphological feature, **is the human concept actually present in the image information &mdash; recoverable as a direction in a representation learned without labels &mdash; or not?**

# **The Idea: Labels as a Read-Out Key**

---

The method has two stages, and the separation between them is the whole point.

**Stage 1 &mdash; representation (label-free).** An [I-JEPA](https://arxiv.org/abs/2301.08243)-style encoder (a Vision Transformer context-encoder, an EMA target encoder, and a predictor) is trained to predict the *embeddings* of masked galaxy regions from the visible context. No pixel reconstruction, no labels, no reward. The encoder carves the latent space along whatever axes of variation actually exist in galaxy images.

**Stage 2 &mdash; measurement.** The encoder is **frozen**. The labels are consulted only now, to ask: *which of these pre-existing directions line up with what humans called a bar, a bulge, an edge-on disk?* A mislabelled galaxy can blur a read-out direction (a local, inspectable measurement error) &mdash; it **cannot** reshape the encoder's geometry the way it deformed v1's weights (a global, baked-in representation error). The noise isn't eliminated; it's *relocated* to where it can be measured and bounded.

<p align="center">
  <img src="assets/method_diagram.png" width="760" alt="Two-stage method: label-free JEPA pretraining, then frozen probing" />
</p>

<p align="center">
  <em>The two stages. Left: masked-region embedding prediction builds the representation, labels nowhere in sight. Right: the frozen encoder is probed with Galaxy Zoo labels used only as a read-out key.</em>
</p>

# **The Nameability Ladder**

---

For any feature, *"can the labels name it?"* has four possible answers &mdash; and each is a **different scientific result**:

| Rung | Meaning |
| --- | --- |
| **1 &mdash; Clean linear direction** | A single vector whose projection tracks the feature. The concept is one coordinate of the latent space. |
| **2 &mdash; Entangled linear direction** | The axis exists but isn't independent &mdash; moving along "bar" drags "bulge". Still nameable, but tangled. |
| **3 &mdash; Nonlinear** | No single vector captures it; a controlled nonlinear probe decodes it. *Present, but not a simple direction.* |
| **4 &mdash; Not recoverable** | No probe &mdash; linear or not &mdash; finds it. The information isn't in the pixels at this resolution. |

Mapping v1's confused features onto this ladder *is the experiment*. A confused feature that comes back as a **clean direction** means the information was in the pixels all along and v1's supervised objective simply couldn't extract it cleanly. One that's **genuinely absent** means it was never a labels problem &mdash; the image doesn't contain it. Either way, the diagnosis is one v1 could not give.

The headline result chases something stronger than classification: **uncertainty geometry**. Fit a concept axis on high-consensus galaxies only (the ones volunteers overwhelmingly agreed on), then project the *ambiguous* ones the axis never saw, and ask whether their distance along the axis reproduces the human *vote fraction*. If an unsupervised geometry reproduces the volunteers' uncertainty **without ever being trained on it**, then that ambiguity is a real property of the images &mdash; flipping v1's reading that it was a labelling artefact.

# **The First Result**

---

A from-scratch JEPA was trained at **pilot scale** (10k galaxies, 6k steps, on a laptop) to answer one question before committing to anything larger: *does a label-free encoder learn useful galaxy structure at all, without collapsing?*

<p align="center">
  <img src="assets/pilot_collapse_trace.png" width="760" alt="Collapse-monitor trace over pilot training" />
</p>

<p align="center">
  <em>The collapse monitor over 6k steps. Representation collapse (the standard JEPA failure) would drive embedding standard deviation to zero and mean cosine to one; instead the representation spreads out, effective rank stabilises, and the early norm transient self-arrests as the EMA target freezes. The encoder learned &mdash; it did not collapse.</em>
</p>

A linear probe on the **frozen** embeddings, trained only as a read-out key, separated smooth from featured galaxies at:

> **AUC = 0.905** (95% CI 0.873&ndash;0.933) &mdash; on a label the encoder never saw during pretraining.

<p align="center">
  <img src="assets/pilot_umap.png" width="620" alt="UMAP of frozen pilot embeddings coloured by morphology" />
</p>

<p align="center">
  <em>A UMAP of the frozen pilot embeddings, coloured by morphology. The structure is visible even at this tiny scale &mdash; a 2D shadow of a separation that lives, more cleanly, in the full representation.</em>
</p>

This is a *signs-of-life* result, not the final science &mdash; the pilot is deliberately undertrained. But it clears the gate the whole project was staked on: the premise works. The full-scale run and the probing harness follow.

# **The Data Layer**

---

A self-supervised model trained to predict masked regions will happily learn *any* structure in its inputs &mdash; including artefacts. So the fidelity of the imagery matters more here than it did for a supervised classifier, and a substantial part of this project is a data pipeline built to not lie to the encoder.

**Native FITS, not display JPGs.** v1 used 8-bit display-stretched cutouts; those irreversibly compress the low-surface-brightness range where the confused features (winding, arm count, tidal structure) live. Probing on them would confound *"absent from the pixels"* with *"destroyed by the 8-bit quantisation"*. v2 pulls raw FITS frames and applies a single, stamped `asinh` stretch.

**No rebinning &mdash; an empirically proven choice.** A fast cutout service was tested as a shortcut and **rejected** by a fidelity test: it preserved calibrated flux and bright signal, but attenuated high-frequency power to ~11% of native and correlated the pixel noise &mdash; injecting learnable fake structure exactly in the faint regime the science depends on. Native-resolution frames, never resampled, are therefore a *measured* protection, not a preference.

**Server-side cutouts at native fidelity.** Direct frame download is throttled to the point of infeasibility at corpus scale, so cutouts are made *next to the data* on [SciServer Compute](https://www.sciserver.org/) &mdash; only ~50 KB stamps cross the link, byte-identical to the native frame, at ~17&times; the throughput.

**Leak-impossible splits.** The pretraining and probing corpora are deduplicated by object ID so the frozen encoder can never have seen a probe-test galaxy during pretraining; the uncertainty-geometry firewall (consensus galaxies fit the axis, ambiguous galaxies test it) is enforced *in code*, not by discipline. These guarantees are merge-blocking invariants &mdash; a split that could leak cannot be committed.

# **Architecture**

---

The codebase is built around a few structural commitments, several inherited as design DNA from a separate orchestration project:

- **The encoder is just an `nn.Module` behind a Protocol.** JEPA, and the planned MAE / contrastive baselines, all satisfy the same interface, so the probing ladder stays identical across objectives &mdash; the baseline comparison can't become mush.
- **Gates as first-class acceptance criteria.** "The run succeeded" and "the scientific claim passed its controls" are separated: a failed nuisance-control is a *finding*, not a broken run.
- **Provenance everywhere.** Every run is config-hash + git-SHA stamped; the preprocessing stretch and split assignment are part of the experiment record, not notebook constants.
- **Fail loud, never silently corrupt.** Schema assumptions are tested at small scale before any large pull; metrics are anchored so they can't quietly return a meaningless number.

# **What's Next**

---

The full experimental design of the probing stage is architected: the nameability ladder, a controls battery that gates every rung verdict (selectivity, negative controls, a nuisance battery), the uncertainty-geometry measurement, and MAE / contrastive baselines run through the same ladder to separate *intrinsic to the images* from *artefact of the objective*. The immediate path is: full-scale pretraining run &rarr; the frozen probing harness &rarr; the per-feature ladder, the uncertainty geometry, and the comparison back to v1.

# **Dependencies**

---

The quickest way in is the included **dev container** (`.devcontainer/`) &mdash; open the repo in VS Code or a Codespace and *Reopen in Container*, and it builds a Python environment with [uv](https://docs.astral.sh/uv/) and installs the project.

To set it up locally instead:

```
uv sync --extra dev --extra data --extra eval
```

The stack is PyTorch for the model (with [Apple MPS](https://developer.apple.com/metal/pytorch/) support on Apple Silicon, CUDA elsewhere), `astropy` / `astroquery` for the FITS imagery and the SDSS / Galaxy Zoo catalogue joins, and the usual scientific Python tooling (NumPy, scikit-learn, UMAP, Matplotlib) for the probing and figures. The optional extras gate the heavier dependencies: `data` for the imagery pull, `eval` for the probing stack.

> The Galaxy Zoo morphology labels are the [Galaxy Zoo 2](https://data.galaxyzoo.org/) vote fractions; the imagery is pulled as native SDSS frames via SciServer.

# **Usage**

---

**Obtaining the data.** Imagery is pulled as native FITS frames from the SDSS Science Archive Server via [SciServer Compute](https://www.sciserver.org/) (which requires a free token), with the Galaxy Zoo 2 vote fractions joined from the SDSS catalogue. The pull is staged &mdash; a small slice for development, scaling to the full corpus.

**Pretraining** trains the label-free JEPA encoder and writes a provenance-stamped, frozen checkpoint, with the collapse monitor logging the health of the representation throughout.

**Probing** reloads the frozen encoder and fits the read-out probes &mdash; the per-feature ladder, the controls, and the uncertainty geometry &mdash; emitting the headline figures.

# **Repository**

---

| Path | Contents |
| --- | --- |
| `src/galaxy_jepa/core/` | The encoder Protocol, config + provenance, and the Gate machinery |
| `src/galaxy_jepa/data/` | The FITS pull, `asinh` pipeline, masking boxes, and the leak-impossible split guards |
| `src/galaxy_jepa/models/` | The Vision Transformer encoder |
| `src/galaxy_jepa/objectives/` | The JEPA objective and training loop |
| `src/galaxy_jepa/probing/` | The frozen read-out probes |
| `src/galaxy_jepa/harness.py` | The reusable train &rarr; freeze &rarr; probe &rarr; figures entrypoint |
| `docs/spec/` | Specifications for the encoder, config, gates, data, splits and validation |
| `artifacts/` | Networked, credential-touching pull glue (kept out of the importable package) |
| `.devcontainer/` | Dev container (uv-based) |

# **Help**

---

**Why train without labels at all &mdash; isn't that throwing away information?**

The labels aren't thrown away; they're *moved*. v1 showed that training on the noisy votes bakes their confusion into the model. Here the labels are used only to *measure* the label-free representation, so their noise becomes a measurement error you can quantify and control &mdash; not a representation error baked into the weights. See [The Idea](#the-idea-labels-as-a-read-out-key).

**What does the JEPA actually predict, if not pixels?**

The *representation* of the masked region &mdash; an abstract feature vector &mdash; not the pixels themselves. This is the key difference from a masked autoencoder, and it's why the encoder learns semantic structure rather than wasting capacity reconstructing noise. It's what the "Predictive" in Joint-Embedding *Predictive* Architecture refers to.

**Why is the imagery pulled as FITS instead of the JPGs v1 used?**

Display JPGs compress exactly the faint-structure range the hard morphological features live in. Probing on them couldn't distinguish *"the feature isn't in the pixels"* from *"the JPG destroyed it"*. See [The Data Layer](#the-data-layer).

**Is this finished?**

No &mdash; it's research in progress. The premise is proven at pilot scale; the full-scale run and the probing harness are designed and being built. See [What's Next](#whats-next).