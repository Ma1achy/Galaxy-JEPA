# Galaxy-JEPA — Claude Code kick-off

> Drop this in the repo next to the scratchpad. Then either paste it into Claude Code, or just say *"read `KICKOFF.md` and begin."*
> (Adjust the scratchpad filename below to whatever you committed — assumed `SCRATCHPAD.md`.)

## What this is

v2 of Galaxy-Zoo-Classifier. **The full design lives in `SCRATCHPAD.md` — read it first and treat it as the source of truth.** In one line: pretrain a label-free galaxy representation with a JEPA, freeze it, then use Galaxy Zoo labels only to *name and test* directions in that representation. The science is the **nameability ladder** + **uncertainty geometry**, not classification accuracy. **Paper 1 is single-survey (SDSS / GZ2).**

## Current phase: planning & scaffolding — NOT training

Do **not** start a pretraining run or write the full training loop yet. We're converting the design doc into a plan, a repo skeleton, and a prioritised backlog, and resolving a couple of open decisions first.

## First response I want (proposals for review)

Read `SCRATCHPAD.md`, then produce these four planning docs:

1. **`PROJECT_PLAN.md`** — a phased plan mapped to Paper 1 / Paper 2, with milestones and dependencies. Make the Paper 1 critical path explicit; it should bottom out in the three headline figures (ladder/AUC bar chart, concept-direction cosine matrix, uncertainty-geometry scatter).
2. **`TODO.md`** — an actionable, prioritised backlog (epics → tasks), tagged by phase, with the **must-have controls treated as first-class tasks**, not afterthoughts.
3. **`DECISIONS.md`** — the open forks needing my call, each with *your recommendation + reasoning*. At least: framework (v1 was TF/Keras; I lean **PyTorch** — the I-JEPA reference impls and the SSL/interpretability ecosystem live there, and a from-scratch JEPA with custom masking/EMA/probing wants that flexibility), backbone (ViT-S/16 to start?), env tooling (**uv + devcontainer**, to match my other repos), from-scratch vs ImageNet warm-start, and the masking strategy (below).
4. A **proposed repo structure** (refine the starting point below).

Create those four. **Hold off on model/training code until I've signed off on the plan and the masking approach.**

## The one design decision to nail before any model code: masking

This is the only genuinely novel architectural piece, and two independent review rounds both converged on it as *the* open item. Problem: I-JEPA's multi-block masking on a mostly-black SDSS cutout burns its budget predicting empty sky.

Research how I-JEPA samples context/target blocks, then propose a concrete **bounding-box-biased** scheme (reuse the average-image bounding-box idea from v1) as a short design note at `docs/masking.md`: the sampling procedure, the parameters, and how it cleanly degrades back to standard I-JEPA. **Propose, don't implement.**

## Load-bearing discipline (from the scratchpad — do not quietly drop)

- **Paper 1 is single-survey.** Multi-survey is Paper 2. No scope-creep.
- **Encoder frozen for all probing.** Labels never train the encoder.
- **Controls are mandatory**: selectivity (Hewitt–Liang), negative controls (shuffled labels / random embeddings / metadata bins), core nuisance probes (z, magnitude, Petrosian size, SNR, PSF). Probing code ships with them, not after.
- **Uncertainty geometry uses the non-circular protocol**: train the axis on consensus extremes (v > 0.8 vs v < 0.2), test on the **held-out** ambiguous middle (0.2–0.8). Flag any probing code that would let the test axis see the graded vote fractions — that's the tautology to avoid.
- **Canonical probe = L2 logistic** (mean-difference CAV as a robustness check). **SSL baselines (MAE, contrastive) are a control, not optional** — same probe ladder across objectives, to attribute a rung to the *objective* vs the *images*.
- **No firstness/novelty claims** in code or docs — the arXiv sweep is still pending.

## Suggested repo structure (propose / refine)

```
galaxy-jepa/
  SCRATCHPAD.md              # design source of truth (already here)
  PROJECT_PLAN.md  TODO.md  DECISIONS.md
  README.md
  pyproject.toml             # uv
  .devcontainer/
  docs/
    masking.md               # the design note
    related-work.md          # arXiv sweep (see below)
  configs/                   # yaml: pretrain.yaml, probe.yaml
  src/galaxy_jepa/
    data/                    # GZ2 cutouts (galaxy-datasets) + CasJobs metadata join
    masking/                 # bounding-box-biased multi-block masking
    models/                  # ViT encoder, predictor, EMA target, JEPA wrapper
    training/                # pretrain loop, EMA schedule, collapse monitor
    probing/                 # logistic concept directions, ladder, controls, uncertainty geometry
    eval/                    # metrics + figures
  scripts/                   # pull_data, run_pretrain, run_probe
  tests/
```

## Work ordering (once plan + masking are agreed)

1. Repo skeleton + env (uv, devcontainer, pyproject, pytest, pre-commit).
2. **Data layer**: GZ2 image cutouts via `galaxy-datasets`; nuisance-metadata join via **CasJobs / SkyServer** (z, Petrosian mag/radius, SNR, PSF). Get a *small* sample end-to-end before scaling anything.
3. **Masking module** (per the agreed note) + bounding-box computation.
4. **JEPA model**: ViT encoder + predictor + EMA target + latent-MSE loss; an overfit-one-batch sanity check + a collapse monitor *before* any real run.
5. **Pretraining loop** (config-driven), small scale first.
6. **Probing harness**: logistic concept directions + the ladder + the controls + the uncertainty-geometry protocol.
7. **Figures**: cosine matrix, ladder bar chart, uncertainty-geometry scatter.

Can run in parallel from day one: the **arXiv sweep** — confirm the JEPA-for-galaxy-morphology gap, read Wu et al. 2510.23749 properly and pull their public MAE as the baseline, pin the contrastive baselines — written up as `docs/related-work.md`.

## Conventions

British English in all prose, docs, and comments. Match my other repos (uv, devcontainer, clean module layout, pytest). If the plan reveals something that changes the design, **propose an edit to `SCRATCHPAD.md`** rather than silently diverging from it.

## Start

Read `SCRATCHPAD.md`, then give me `PROJECT_PLAN.md`, `TODO.md`, `DECISIONS.md`, and the proposed structure — as proposals. Then the `docs/masking.md` design note. No model or training code until I sign off on the plan and the masking approach.
