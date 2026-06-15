# Related work — arXiv sweep

*Status: first pass (June 2026). Living document. Cited from a web sweep; details
marked "verify" should be checked against the primary PDFs before any are quoted
in a paper.*

> **No firstness claims.** Nothing here asserts we are first to anything. The
> safe, defensible framing is: *JEPA-style (latent-prediction) self-supervised
> learning appears **underexplored for galaxy morphology** relative to
> contrastive and masked-reconstruction approaches.* This sweep is the evidence
> for "underexplored", not a guarantee of novelty — treat absence of a hit as
> absence of evidence, not evidence of absence.

---

## What this sweep was for

Three jobs from the kickoff / scratchpad:

1. Confirm the **JEPA-for-galaxy-morphology gap** (or find the paper that closes
   it).
2. Read **Wu & Walmsley 2025 (arXiv 2510.23749)** properly — the closest
   neighbour — and pin its **public MAE** as our pixel-reconstruction baseline.
3. Pin the **contrastive baselines** and the **probing / concept-direction
   lineage** that the method descends from.

---

## 1. The gap: JEPA vs galaxy morphology

I-JEPA (latent-prediction SSL) is well established for natural images and has
proven astrophysics-adjacent variants, but a direct application to **galaxy
morphology** did not surface in this sweep.

- **Assran et al. 2023 — I-JEPA.** *Self-Supervised Learning from Images with a
  Joint-Embedding Predictive Architecture*, CVPR 2023. The method we build on:
  context-encoder ViT, EMA target encoder (anti-collapse), predictor; predict
  **masked-block embeddings** from context, MSE **in latent space**, no pixel
  reconstruction. Multi-block masking: 4 target blocks (scale 0.15–0.20, aspect
  0.75–1.5), one context block (scale 0.85–1.0) with target overlaps removed.
  arXiv: <https://arxiv.org/abs/2301.08243>.
- **Lens-JEPA — physics-informed JEPA for gravitational lensing** (NeurIPS ML4PS
  2025). Astrophysics application of JEPA, but **lensing, not morphology**;
  reported to beat supervised + vanilla I-JEPA on its task — precedent that a
  physics-baked JEPA can pay off (relevant to our E(2)-equivariant ablation).
  <https://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_340.pdf>.
- **HEP-JEPA — collider physics** (arXiv 2502.03933, *verify*). Another
  domain-specific JEPA; cited for breadth.

**Read of the gap.** Galaxy SSL to date is dominated by **contrastive** (MoCo /
BYOL) and, more recently, **masked-reconstruction (MAE)** objectives. A
*latent-prediction* (JEPA) encoder probed for named morphological concepts is the
white space this project sits in. Framing to use: *underexplored relative to
contrastive and masked-reconstruction*, not *first*.

---

## 2. Closest neighbour — Wu & Walmsley 2025 (arXiv 2510.23749)

**Re-envisioning Euclid Galaxy Morphology: Identifying and Interpreting Features
with Sparse Autoencoders.** John F. Wu & Michael Walmsley. *(Author list verified
against the arXiv abstract, June 2026 — exactly these two authors, in this order;
the MAE is released as part of this paper, so "Wu & Walmsley MAE" is the correct
attribution.)* Accepted to the NeurIPS *Machine Learning and the Physical
Sciences* workshop 2025 (submitted 2025-10-27, revised 2025-11-12).
- arXiv: <https://arxiv.org/abs/2510.23749> · HTML: <https://arxiv.org/html/2510.23749v2>

**What they do.** Train **sparse autoencoders (SAEs)** on the embeddings of two
pretrained models — **Zoobot (supervised)** and a **new MAE (self-supervised)** —
on **Euclid Q1** images. Findings: a PCA on the supervised model mostly recovers
features **already aligned** with the Galaxy Zoo decision tree, whereas the SAE
surfaces **interpretable features beyond** that tree, with **stronger
GZ-label alignment than PCA**. Framed as an engine for discovery beyond
human-defined classes.

**Why it matters to us.**
- **The MAE is our head-to-head baseline.** Both **code and trained MAE weights
  are public** (HuggingFace, with an interactive demo). This is the
  pixel-reconstruction foil for JEPA's latent prediction — same probe ladder,
  both objectives. *Action: pull the MAE; confirm licence, input resolution,
  patch size, and whether Euclid-trained weights transfer to SDSS/GZ2 or whether
  we retrain the MAE on our corpus for a fair single-survey comparison.*
- **Our distinction is clean and complementary, not competing:**
  1. **Objective** — our **JEPA latent-prediction** vs their **MAE
     pixel-reconstruction** (and vs Zoobot supervised).
  2. **Direction of inference** — our **top-down hypothesis-testing** (probe a
     *named* human concept, walk the nameability ladder) vs their **bottom-up
     discovery** (read features off an SAE dictionary).
  3. **Uncertainty geometry** — we test **vote-fraction recovery** (does distance
     along an unsupervised axis reproduce the volunteer vote fraction,
     non-circularly), not just label alignment.
- **Convergence, parked:** an SAE on our frozen JEPA encoder would bridge
  top-down and bottom-up and let us cross-check which dictionary features align
  with GZ concepts *and* reproduce vote fractions (scratchpad parking lot).

*Related Wu work:* Wu 2025, *Insights into Galaxy Evolution from Interpretable
Sparse Feature Networks*, ApJ 980 (arXiv 2501.00089) — *verify*.

---

## 3. Galaxy SSL baselines (the controls, not box-ticking)

The scratchpad promotes SSL baselines to **controls**: run the *same* probe
ladder on JEPA vs MAE vs contrastive to attribute a rung to the **objective** vs
the **images**.

**Masked-reconstruction (MAE):**
- **Wu & Walmsley 2025** (above) — public MAE = the canonical pixel-reconstruction
  baseline.

**Contrastive:**
- **Hayat et al. 2021 — *Self-Supervised Representation Learning for Astronomical
  Images*** (arXiv 2012.13083). First to apply **MoCo** to SDSS imagery;
  downstream **GZ2 morphology classification + photometric redshift**; reaches
  supervised-level accuracy with **~2–4× fewer labels**; uses optical-specific
  augmentations (galactic-extinction reddening, varying PSF). The label-efficiency
  precedent for our LSST framing. <https://arxiv.org/abs/2012.13083>
- **Walmsley et al. 2022 — *Towards Galaxy Foundation Models with Hybrid
  Contrastive Learning*** (arXiv 2206.11927; ICML ML4Astro). **BYOL-style** +
  supervised Dirichlet head; introduces **GZ-Evo** (~96.5M responses over 552k
  labelled images + ~1.34M comparable unlabelled). +6% downstream accuracy at 750
  labels (low-label regime). <https://arxiv.org/abs/2206.11927>
- **AstroCLIP** (arXiv 2310.03024, *verify*) — cross-modal foundation model, DESI
  spectra + Legacy images; the multimodal contrastive reference (mostly Paper 2 /
  parking lot).

**Suggested baseline set for the cross-objective ladder (Paper 1):** JEPA (ours)
vs Wu & Walmsley **MAE** vs a **contrastive** encoder (MoCo per Hayat, or
BYOL per Walmsley) — all probed identically. *Decision to confirm: train the
contrastive baseline ourselves on the GZ2 corpus for fairness, or adapt a
published encoder.*

**Adjacent / scaling context (not baselines):**
- Walmsley et al. 2023 — *Galaxy Zoo DESI*, 8.7M galaxies (arXiv 2309.11425);
  Zoobot. · *Scaling Laws for Galaxy Images* (arXiv 2404.02973, *verify*). ·
  *Galaxy Zoo Evo: 1M human-annotated images* (arXiv 2512.23691, *verify*) —
  newer, larger label set; relevant to Paper 2 scaling.

---

## 4. Cross-survey domain gap (Paper 2 context)

- **DeepAstroUDA — Ćiprijanović et al.** (ML:S&T 2023). Frames cross-survey
  morphology as **domain adaptation**; warns cross-survey models extract
  **dataset-specific, non-robust** features unless forced otherwise — the direct
  justification for the survey-leakage probe and degrade-down strategy in Paper 2.
- *From Galaxy Zoo DECaLS to BASS/MzLS* (arXiv 2412.15533, *verify*) —
  unsupervised domain adaptation for morphology across surveys; another Paper 2
  pointer.
- Vision-FM-for-astronomy survey (arXiv 2409.11175, *verify*) — FM features help
  optical galaxy classification, poor on radio; transfer isn't free.

---

## 5. Probing / concept-direction lineage (the method's ancestry)

- **Alain & Bengio 2016** — *Understanding intermediate layers using linear
  classifier probes*. The linear-probe foundation. *(verify arXiv id)*
- **Kim et al. 2018** — **TCAV / Concept Activation Vectors** (ICML 2018). The
  concept-direction idea; our mean-difference (CAV) robustness check descends
  from here.
- **Hewitt & Liang 2019** — *Designing and Interpreting Probes with Control
  Tasks* (EMNLP 2019). **Selectivity** = real-label minus control-label probe
  performance; our must-have control.

---

## Open follow-ups before paper-writing

- [ ] Fetch and read the Wu & Walmsley **MAE** card on HuggingFace: licence,
      resolution, patch size, training corpus; decide transfer vs retrain.
- [ ] Verify all `*verify*`-tagged arXiv IDs and dates against primary PDFs.
- [ ] Decide the contrastive baseline (MoCo vs BYOL) and whether we train it.
- [ ] Confirm no direct **JEPA-for-galaxy-morphology** paper exists via a
      dedicated ADS/arXiv full-text search (this sweep used web search only).

## Sources

- I-JEPA — <https://arxiv.org/abs/2301.08243>
- Wu & Walmsley 2025 (SAEs / MAE) — <https://arxiv.org/abs/2510.23749>
- Lens-JEPA — <https://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_340.pdf>
- Hayat et al. 2021 (MoCo / SDSS) — <https://arxiv.org/abs/2012.13083>
- Walmsley et al. 2022 (hybrid contrastive / GZ-Evo) — <https://arxiv.org/abs/2206.11927>
- Galaxy Zoo DESI — <https://arxiv.org/abs/2309.11425>
- Galaxy Zoo Evo (1M) — <https://arxiv.org/abs/2512.23691>
