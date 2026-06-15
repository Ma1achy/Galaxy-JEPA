# TODO — Galaxy-JEPA backlog

Prioritised, epics → tasks. **Paper 1 only** (Paper 2 items are marked and
parked). Controls are **first-class tasks**, interleaved into probing — not a
trailing afterthought.

**Tags:** `[P1]`…`[P7]` phase · `[control]` · `[baseline]` · `[parallel]` ·
`[P2-paper]` deferred to Paper 2.
**Priority:** `(P0)` blocking critical path · `(P1)` needed for Paper 1 · `(P2)`
nice-to-have.
Port targets reference v1 at `/Users/malachy/Documents/Galaxy-Zoo-Classifier`.

---

## Epic A — Environment & repo skeleton `[P1]`
- [ ] (P0) `uv` `pyproject.toml`, Python 3.11, core deps declared. *(D3)*
- [ ] (P0) `.devcontainer/` (devcontainer.json + Dockerfile, uv) — port conventions from v1 `.devcontainer/`.
- [ ] (P0) `.pre-commit-config.yaml` (ruff lint+format) + `.gitignore`.
- [ ] (P0) `src/galaxy_jepa/{data,masking,models,training,probing,eval}/` packages + `tests/` import test.
- [ ] (P1) `README.md` stub (thesis, scope guardrails, how to run).
- [ ] (P1) Verify: `uv sync`, `uv run pytest`, `pre-commit run --all-files` all green.

## Epic B — Data layer `[P2]` — *small sample end-to-end first*
- [ ] (P0) Pull a **small** GZ2 sample via `galaxy-datasets`; load images + vote-fraction catalogue. Port loading logic from v1 `main/dataframe/dataframe.py`. *(this is the **probing** corpus)*
- [ ] (P0) **Large unlabelled SDSS pretraining pull** (≫250k, beyond the GZ2-labelled subset) via SkyServer — the **pretraining** corpus, decoupled from probing. *(D6; net-new; biggest de-risk for from-scratch D4)*
- [ ] (P0) Centre-crop to 256² (256-token grid). Port v1 `dataframe.py:284`.
- [ ] (P0) Label schemes (Hubble / Reduced / All, Q0–Q10) as v2 config. Port v1 `main/dataframe/keys.py`.
- [ ] (P1) Q10 "bulge present" construction. Port v1 `image_preprocessing/cleandataset.py:94`.
- [ ] (P1) **Reliable-label filter (mean + 2σ)** — net-new (v1 only thresholds 0.5). *(D8)*
- [ ] (P0) **CasJobs / SkyServer metadata join** — z, Petrosian mag/radius, SNR, PSF for the GZ2 spectroscopic main sample; confirm completeness fractions. *(net-new)*
- [ ] (P1) Rotation/reflection augmentation pipeline (symmetry, augmentation-first). *(D10)*
- [ ] (P1) Scale data layer from small sample to the full corpora once E2E verified — GZ2 ~250k (probe) + large unlabelled SDSS (pretrain).

## Epic C — Masking & bounding box `[P3]` — per `docs/masking.md`
- [ ] (P0) **Average-image bbox**: mean of ~2,000 cutouts, threshold τ=0.085, centred box → fractional box → `G×G` token mask. *(net-new; v1 code absent — the cheap fallback / β-degradation reference)*
- [ ] (P1) **Per-galaxy Petrosian-scaled bbox** (half-width k·R_petro, k~2–3, clamped) — recommended default once metadata join lands; handle missing R_petro. *(docs/masking.md §3.1; reuses CasJobs pull)*
- [ ] (P1) Re-tune EMA/masking-ratio per β — β=0 tuning does **not** transfer to β=1 (β removes the easy sky-prediction task). *(docs/masking.md §7)*
- [ ] (P0) Token weight map `w` from box + bias strength β.
- [ ] (P0) Bounding-box-biased multi-block sampler (M=4 targets, I-JEPA scale/aspect; position ∝ mean weight; optional φ floor). *(D5)*
- [ ] (P0) **Degradation test**: β=0 (or full box) reproduces standard I-JEPA exactly.
- [ ] (P1) Diagnostics: sky-waste metric vs β; coverage; mask-overlay visualisations.

## Epic D — JEPA model `[P4]`
- [ ] (P0) ViT-S/16 context encoder (256² → 256 tokens), from-scratch. *(D2, D4)*
- [ ] (P0) Predictor + EMA target encoder + latent-MSE loss (I-JEPA recipe).
- [ ] (P0) **Overfit-one-batch sanity check** — gate before any real run.
- [ ] (P0) **Collapse monitor** — representation variance / rank / std tracking — gate before any real run.

## Epic E — Pretraining loop `[P5]`
- [ ] (P0) Config-driven loop (`configs/pretrain.yaml`); small scale first.
- [ ] (P1) EMA schedule + masking-ratio **lightweight sweep harness** (modest grid; no heavy MLOps for Paper 1).
- [ ] (P1) Checkpointing + frozen-encoder export for probing.

## Epic F — Probing harness `[P6]` (frozen encoder) — controls interleaved
- [ ] (P0) **L2 logistic concept-direction probe** → per-feature held-out AUC + calibration; unit-normalised weight = concept direction. *(D7)*
- [ ] (P1) Mean-difference (CAV) direction as robustness + entanglement signal (logistic-vs-CAV disagreement). *(D7)*
- [ ] (P0) `[control]` **Selectivity (Hewitt–Liang)** — real-label minus control-label probe performance.
- [ ] (P0) `[control]` **Negative controls** — shuffled vote fractions, random labels, random embeddings, image-metadata bins, sky-background/noise level (these must *fail*).
- [ ] (P0) `[control]` **Core nuisance probes** — z, apparent magnitude, Petrosian radius, SNR, PSF off the same axis machinery.
- [ ] (P0) **Non-circular uncertainty geometry** — binary axis on consensus extremes (v>0.8 vs v<0.2); project **held-out** middle (0.2–0.8); Spearman of margin distance vs vote fraction. Guard against the test axis seeing graded votes.
- [ ] (P1) Confidence as a probe target (separate axis from the uncertainty test). *(D9)*
- [ ] (P1) Ladder rungs 3/4: MLP / k-NN **only** with controls; never standalone.

## Epic G — Baselines as controls `[baseline]` — *same probe ladder*
- [ ] (P1) `[baseline]` **MAE** — pull Wu & Walmsley public model (`docs/related-work.md`); decide transfer vs retrain on GZ2; probe identically. *(D12, Rung-3 control)*
- [ ] (P1) `[baseline]` **Contrastive** (MoCo or BYOL) — train ours vs adapt published *(D12 sub — needs your call)*; probe identically.
- [ ] (P1) Cross-objective comparison table (rung per feature × objective).

## Epic H — Figures & eval `[P7]`
- [ ] (P0) **FIG 1** — ladder / AUC bar chart (per-feature AUC + selectivity).
- [ ] (P0) **FIG 2** — concept-direction cosine-similarity matrix (vs v1 Fig 18/19).
- [ ] (P0) **FIG 3** — uncertainty-geometry scatter (projection vs vote-fraction, held-out middle).
- [ ] (P1) Label-efficiency curve (SSL-pretrained vs supervised-from-scratch).
- [ ] (P1) v1-comparable evaluation (Hubble / Reduced / All schemes, same metrics).

## Epic I — arXiv sweep `[parallel]`
- [x] First pass → `docs/related-work.md` (gap confirmed cautiously; Wu & Walmsley MAE pinned; contrastive + CAV lineage pinned).
- [ ] (P1) Fetch Wu & Walmsley MAE card (licence, resolution, patch size, corpus).
- [ ] (P1) Verify all `*verify*`-tagged arXiv IDs; dedicated ADS full-text search for any JEPA-morphology paper.

## Epic R — Rung controls / ablations `[control]`
- [ ] (P1) **Masking β-sweep** {0, 0.5, 1.0} as the masking ablation (β=0 control).
- [ ] (P1) **8×8-patch (higher-res) ablation** — Rung-4 control (under-resolved vs absent). *(D11)*

## Deferred — Paper 2 `[P2-paper]`
- [ ] Multi-survey corpus (SDSS + DESI Legacy → space-based); homogenisation (degrade-down first).
- [ ] Survey-leakage probe + single-vs-multi merge experiment.
- [ ] Fine-tuning comparison; confidence-aware read-outs.
- [ ] Deferred controls: matched evaluation, cross-split robustness.
- [ ] E(2)-equivariant ViT ablation; GalaxyMNIST external comparison.
- [ ] Parking lot: SAE on frozen JEPA embeddings; Spectra/Jacobian mechanistic angle; UMAP gut-check.
