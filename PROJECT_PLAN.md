# Project plan — Galaxy-JEPA

*v2 of Galaxy-Zoo-Classifier. Planning artefact; the design source of truth is
`galaxy-jepa-scratchpad.md`. British English throughout.*

---

## 1. Thesis

Pretrain a galaxy-image representation **unsupervised** with an I-JEPA, **freeze**
it, then use Galaxy Zoo labels **only as a read-out key** — to *name and test*
directions the representation already learned, never to train it. The science is
the **nameability ladder** and **uncertainty geometry**, not classification
accuracy. The win is **relocating label noise** out of representation-learning and
into an inspectable, controllable measurement stage.

### Load-bearing guardrails (do not quietly drop)
- **Paper 1 is single-survey (SDSS / GZ2).** Multi-survey is Paper 2.
- **Encoder frozen for all probing.** Labels never train the encoder.
- **Controls are mandatory and first-class** — selectivity (Hewitt–Liang),
  negative controls (shuffled labels / random embeddings / metadata bins), core
  nuisance probes (z, magnitude, Petrosian size, SNR, PSF). They ship *with* the
  probing code, not after.
- **Uncertainty geometry is non-circular** — train the axis on consensus extremes
  (v>0.8 vs v<0.2), test on the **held-out** ambiguous middle (0.2–0.8). The test
  axis must never see the graded vote fractions.
- **Canonical probe = L2 logistic** (mean-difference CAV as robustness check).
- **SSL baselines (MAE, contrastive) are a control, not optional** — same probe
  ladder across objectives.
- **No firstness / novelty claims** in code or docs (arXiv sweep is evidence, not
  a guarantee — see `docs/related-work.md`).

---

## 2. Two-paper shape

| | Paper 1 — the clean science | Paper 2 — performance & scaling |
|---|---|---|
| Corpus | Single-survey SDSS. **Pretrain** on a large *unlabelled* SDSS sample (≫250k); **probe** on the GZ2-labelled ~250k (corpora decoupled — see `DECISIONS.md` D6) | Multi-survey (SDSS + DESI Legacy → space-based) |
| Core | Unsupervised JEPA; logistic concept directions; nameability ladder **with must-have controls**; uncertainty geometry; cosine matrix vs v1; **MAE + contrastive baselines** | Fine-tuning; confidence-aware read-outs; multi-survey scaling + **survey-leakage** experiment; deferred controls (matched eval, cross-split); GalaxyMNIST |
| Question | Which GZ concepts are nameable directions present *before any label*; which are entangled / nonlinear / absent; does the geometry reproduce human uncertainty? | Does more / more-diverse data buy richer morphology axes without the survey becoming a shortcut? |

This plan is **Paper 1**; Paper 2 items appear only as deferred markers.

---

## 3. Phases, milestones, dependencies

| Phase | Goal | Milestone (done = ) | Depends on |
|---|---|---|---|
| **P0** Planning & scaffolding | This round | Plan docs + masking note signed off; repo skeleton green (`uv sync`, `pytest`, `pre-commit`) | — |
| **P1** Env & repo skeleton | Reproducible env | devcontainer builds; `import galaxy_jepa`; CI-lint clean | P0 |
| **P2** Data layer | GZ2 probing set + **large unlabelled SDSS pretraining set** + nuisance metadata, **small sample end-to-end first** | A small labelled+metadata sample loads as tensors; CasJobs/SkyServer join verified (z, Petrosian mag/radius, SNR, PSF); reliable-label (mean+2σ) filter; **separate unlabelled SDSS pretraining pull** (≫250k) wired in, **including petroRad + arcsec/pixel for the per-galaxy masking box** (distinct from the nuisance join) (D6) | P1 |
| **P3** Masking module | Bounding-box-biased masking + bbox computation | Mean-image bbox computed; masking matches `docs/masking.md`; β=0 reproduces I-JEPA; sky-waste metric falls with β | P2 |
| **P4** JEPA model | ViT encoder + predictor + EMA target + latent-MSE | **Overfit-one-batch passes**; **collapse monitor** live (rep variance / rank); shapes correct | P3 |
| **P5** Pretraining loop | Config-driven pretraining, small scale first | A small pretrain run completes without collapse; EMA + masking-ratio sweep harness (lightweight) | P4 |
| **P6** Probing harness | Logistic concept directions + ladder + **controls** + uncertainty geometry | Per-feature AUC + calibration; selectivity; negative controls; nuisance battery; non-circular uncertainty Spearman | P5 |
| **P7** Figures | The three headline figures | All three render from real probe outputs | P6 |
| **Parallel** arXiv sweep | `docs/related-work.md` | First pass done (this round); follow-ups closed before write-up | — (runs from day 1) |
| **Baselines** (control) | MAE + contrastive, **same probe ladder**, **all trained on the same SDSS pretraining corpus** | Each baseline encoder probed identically; cross-objective comparison table. MAE = Wu & Walmsley recipe reproduced on SDSS (released Euclid MAE = reference/validation only); contrastive trained on SDSS (D12) | P5 (SDSS-trained baselines) → P6 |

### Sanity gates (non-negotiable, from the scratchpad)
- **Before any real pretrain run:** overfit-one-batch **and** the collapse monitor
  must be in place (P4).
- **Before any ladder claim:** the controls battery (P6) must be wired in — a rung
  means nothing about the images until selectivity + nuisance probes hold it down.

---

## 4. Paper 1 critical path → the three headline figures

```
P1 env ─► P2 data (large unlabelled SDSS pretrain set + GZ2 probe set + metadata join, small sample first)
            │
            ▼
        P3 masking (bbox + bounding-box-biased, β=0 == I-JEPA)
            │
            ▼
        P4 JEPA model ──[overfit-one-batch + collapse monitor]──► P5 pretrain (small first)
                                                                      │
                                                                      ▼
                                                   P6 probing harness (frozen encoder)
                                                   ├─ logistic concept directions + calibration
                                                   ├─ CONTROLS: selectivity, negatives, nuisance battery
                                                   └─ non-circular uncertainty protocol
                                                                      │
                            ┌─────────────────────────┼─────────────────────────┐
                            ▼                          ▼                          ▼
                  FIG 1  ladder / AUC      FIG 2  concept-direction     FIG 3  uncertainty-geometry
                  bar chart (per feature,  cosine-similarity matrix     scatter (projection vs
                  AUC + selectivity)       (entanglement; vs v1 18/19)  vote-fraction, held-out middle)
```

The three figures **are** the Paper 1 deliverable:
1. **Ladder / AUC bar chart** — per-feature concept-direction AUC + selectivity;
   the nameability ladder, quantified.
2. **Concept-direction cosine-similarity matrix** — entanglement structure,
   recovered from the embeddings; compared to v1's Fig 18/19 confusion geometry.
3. **Uncertainty-geometry scatter** — projection vs vote-fraction (Spearman), axis
   trained on consensus extremes, tested on the held-out ambiguous middle. The
   high-upside / high-beta headline — backbone the paper on figs 1–2 + controls +
   v1 comparison so the paper stands even if fig 3 comes out null.

---

## 5. Reuse from v1 (`/Users/malachy/Documents/Galaxy-Zoo-Classifier`)

v1 is 100% TF/Keras → **port logic, not code**, into PyTorch.

| Asset | v1 location | Use in v2 |
|---|---|---|
| Label schemes (Hubble / Reduced / All, Q0–Q10) | `main/dataframe/keys.py` | Re-express as v2 label config |
| `galaxy-datasets` pipeline (424² JPEG → centre-crop) | `main/dataframe/dataframe.py` | Data layer (P2) |
| Centre-crop logic | `main/dataframe/dataframe.py:284` | Crop in P2 |
| Q10 "bulge present" construction | `image_preprocessing/cleandataset.py:94` | Label prep (P2) |
| uv + devcontainer + pinned deps | `.devcontainer/`, `requirements.txt` | Env conventions (P1) |

**Net-new (must build):** average-image **bounding box** (README documents it;
code absent — rebuild, see `docs/masking.md` §3); **mean+2σ reliable-label
filter** (v1 only thresholds at 0.5); **all CasJobs/SkyServer metadata**; a
**large unlabelled SDSS pretraining pull** beyond the GZ2-labelled set, with
**petroRad + arcsec/pixel** for the per-galaxy masking box (D6); **SDSS-trained MAE
+ contrastive baselines** (reproduce the Wu & Walmsley MAE recipe on SDSS, not the
off-the-shelf Euclid model — D12); the **entire PyTorch JEPA stack**.

---

## 6. Proposed repo structure

```
Galaxy-JEPA/
  galaxy-jepa-scratchpad.md   # design source of truth (name kept)
  KICKOFF.md
  PROJECT_PLAN.md  TODO.md  DECISIONS.md  README.md
  pyproject.toml              # uv
  .pre-commit-config.yaml     # ruff + basic hooks
  .devcontainer/              # devcontainer.json + Dockerfile (uv, py3.11)
  docs/
    masking.md                # the design note
    related-work.md           # arXiv sweep
  configs/                    # pretrain.yaml, probe.yaml (placeholders)
  src/galaxy_jepa/
    data/        # GZ2 cutouts (galaxy-datasets) + CasJobs metadata join + label prep
    masking/     # bounding-box computation + bounding-box-biased multi-block masking
    models/      # ViT encoder, predictor, EMA target, JEPA wrapper
    training/    # pretrain loop, EMA schedule, collapse monitor
    probing/     # logistic concept directions, ladder, controls, uncertainty geometry
    eval/        # metrics + the three figures
  scripts/       # pull_data, run_pretrain, run_probe (placeholders)
  tests/
```

---

## 7. Risk register (pointer)

Full detail in the scratchpad; the live ones for Paper 1:

- **Representation collapse** — JEPA's standard failure → collapse monitor + EMA /
  masking-ratio tuning *before* any real run (P4 gate).
- **Spurious-correlate probes (Rung 3)** — a high-capacity probe decodes via
  nuisance correlates, not morphology → the controls battery exists for this;
  never let an MLP result stand alone.
- **Probe-capacity confound** — a powerful enough probe decodes anything → claim
  strength tied to probe simplicity (linear + selective = strongest).
- **Null uncertainty geometry** — highest upside, most likely null → paper
  backboned on figs 1–2 + controls + v1, not on fig 3 landing positive.
- **Scope creep** — Stage 1 + frozen logistic probing + must-have controls + the
  three figures is the minimum that answers the question. Multi-survey stays in
  Paper 2.

---

## 8. Status

- **P0** in progress: planning docs + `docs/masking.md` + `docs/related-work.md`
  delivered for review; repo skeleton to be scaffolded on sign-off (no model /
  training code). See `DECISIONS.md` for the forks awaiting your call — model code
  starts only after the plan **and** the masking approach are signed off.
