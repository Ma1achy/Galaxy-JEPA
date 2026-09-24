# TODO — Galaxy-JEPA backlog

Prioritised, epics → tasks. **Paper 1 only** (Paper 2 items are marked and
parked). Controls are **first-class tasks**, interleaved into probing — not a
trailing afterthought.

**Tags:** `[P1]`…`[P7]` phase · `[control]` · `[baseline]` · `[parallel]` ·
`[P2-paper]` deferred to Paper 2.
**Priority:** `(P0)` blocking critical path · `(P1)` needed for Paper 1 · `(P2)`
nice-to-have.
Port targets reference v1 at `/Users/malachy/Documents/Galaxy-Zoo-Classifier`.

> **Design source of truth:** `docs/galaxy-jepa-spec.pdf` (consolidated spec, incl. D13/D14 and
> the five grounded statistical decisions). `docs/architecture.md` + `docs/spec/` remain the
> engineering contracts. This file tracks *state*, and is current as of the probing-load-path pass.

---

## Epic A — Environment & repo skeleton `[P1]` — **done**
- [x] (P0) `uv` `pyproject.toml`, Python 3.11, core deps declared. *(D3)*
- [x] (P0) `.devcontainer/` (devcontainer.json + Dockerfile, uv).
- [x] (P0) `.pre-commit-config.yaml` (ruff lint+format) + `.gitignore`.
- [x] (P0) `src/galaxy_jepa/{core,data,masking,models,objectives,probing,callbacks,eval}/` + tests.
- [x] (P1) `README.md` (thesis, scope guardrails, how to run).
- [x] (P1) Verify: `uv sync`, `uv run pytest`, `pre-commit run --all-files` all green.

## Epic B — Data layer `[P2]`
- [x] (P0) Small GZ2 sample pull; images + vote-fraction catalogue (`data/pull.py`, `data/sources.py`).
- [x] (P0) **Large unlabelled SDSS pretraining pull** — **826,968 stamps, 612 GB, landed.**
  Selection is a resolution window, not a magnitude prefix: `type=3`, `clean=1`,
  `modelMag_r` 14–19, `petroRad_r` ∈ (5″, 25″], uniform modulus stride, never in any of the five
  GZ2 tables. Pulled in 827 chunked waves by `artifacts/sciserver_pull.py` over ~6 days; audited
  clean on every check (12 columns, zero duplicate objIDs across nine restarts, one distinct stamp
  size, exact objID↔FITS set equality, zero probe overlap, `resolve_corpora()` PASS). *(D6)*
  - `data_snapshot` = `manifest:cb69cea28f3ba37856ee011ca6bf15bab1990dd5f7c32deca08cf53a2680de61`.
  - Corpora live on the external SSD (`data/{probe,pretrain}` are symlinks): 808 GB of 3.6 TiB
    used, **2.8 TiB free** — the internal-disk arithmetic that used to sit here is obsolete.
- [x] (P0) Centre-crop to 256² native (no rebin — `artifacts/fidelity_test.py` proved resampling
  attenuates high-frequency power to ~0.11 of native).
- [x] (P0) Label schemes as config — superseded by the **two-scheme experiment**
  (`probing/schemes.py`), which is the D14 form of this task.
- [ ] (P1) Q10 "bulge present" construction. Port v1 `image_preprocessing/cleandataset.py:94`.
- [x] (P0) **Reliable-label filter — SUPERSEDED, and frozen as such.** The mean+2σ filter is
  withdrawn: v1 applied it in its dataset analysis (§5.2.1), and the decision does not rest on v1.
  Probe-target noise is conservative — it attenuates toward chance and cannot manufacture a
  direction — so a feature clearing the gate unfiltered is a *stronger* result. Frozen at **1**,
  the minimum where a vote fraction is defined, via `VoteCountFreeze` (hashed, stamped, refit
  refused). Sweep **{1, 5, 11, 21, 37}** pre-registered as a robustness claim, not a selection
  step. *(D8 superseded)*
- [ ] (P1) **Vote count is not merely noise for the uncertainty geometry — decide a LOCAL floor.**
  **Measured in Brief R2 (2026-09-22):** 11 of R2's 20 curved paths are a mixture of vote-reach
  groups; each group is straight, and only the mixture bends (edge-on 0.30 → 0.02 on the well-voted
  half). Conditional questions reach a median of 5–8 volunteers. `artifacts/r_findings.md` §R2.
  Separate sub-system, deliberately untouched by the corpus-wide decision above. A galaxy at
  50/50 on 60 votes is *genuinely ambiguous* — people looked and disagreed — while 2/2 on 4 votes
  may be obvious and merely undersampled; likewise `v = 1.0` on 3 votes is a weak
  consensus-extreme, not a strong one. The uncertainty-geometry test and the consensus-extreme
  split (`extreme_low`/`extreme_high`) both read the fraction as if it carried the same meaning
  at every depth, and it does not. Decide a vote-count floor **or** a weighting **local to those
  two**; do **not** impose it corpus-wide, which would delete the ambiguous middle that test
  exists to use. Bites hard at the frozen floor: 89.8% of t09 boxy's positives rest on ≤2 votes.
- [ ] (P1) **Spiral's bend is a VISIBILITY axis, not inclination — the uncertainty-geometry brief
  must separate it.** *(Brief S1, 2026-09-23; `artifacts/s_findings.md` §S1, pre-registration
  hash `8574452c`.)* Inclination explains **0.087** of spiral's bend (full) and **0.027**
  (well-voted), against a pre-registered attribution bar of 0.5. What carries it, exploratory and
  unseparated: magnitude partial r **+0.65 / +0.67** (fainter), SNR **−0.62 / −0.62**, size
  **−0.53 / −0.58** (full / well-voted). That is **D13's *resolution* confound, not its projection
  one**: uncertain-vote spirals are the images that carry the least information. The brief must:
  - **Pre-register the separation of faintness, SNR, size AND redshift** — partial on each,
    controlling the other three. Redshift is the likely common cause (distance makes a galaxy
    fainter, smaller and noisier at once), and `z` is already joined.
  - **Answer one question:** does off-axis displacement predict vote uncertainty **beyond
    visibility**? If visibility accounts for all of it, the uncertainty is resolution-limited —
    a D13 verdict (volunteers disagreed because the image could not settle it), not evidence that
    the representation encodes human ambiguity.
  - **Test off-axis displacement as well as on-axis distance.** The bend lives off the concept
    line; projecting onto the line alone discards exactly the component S1 found.
  - **Reconcile with R0, which is not a contradiction:** the *concept direction* is not carried by
    size or brightness (R0: spiral SURVIVES, retaining 0.86 of its margin under size matching in
    the full population and 1.18 under magnitude matching in the conditional one); the
    *uncertainty offset* tracks visibility (S1). One
    axis says what a spiral is, the other how sure anyone could be — they are different
    directions and a result on one says nothing about the other.
- [x] (P0) **CasJobs / SkyServer metadata join** — z, Petrosian mag/radius, SNR, PSF, verified by
  a 10-row ra/dec guard + at-scale range summary. SNR is derived image-domain (`snr_r`) at the
  single derivation site and backfilled across the pulled corpora.
- [x] (P1) **Axis-ratio pull for inclination conditioning** — `expAB_r` + `deVAB_r` from SDSS
  `PhotoObj`, joined on `objID` for the GZ2 **probe** corpus. Independent photometric
  inclination proxy (non-circular). Distinct from both the masking pull (petroRad + arcsec/pixel)
  and the nuisance join. Catalogue-only, no image re-cut. **Landed: 230,358/230,358 matched, zero
  null.** *(D13)*
- [x] (P1) **Deblending / over-stamp flag** — `petrorad_suspect` (`petroRad_r` > 25″) derived at
  the single site (`data.pull.with_derived_columns`); **2,143 of 230,358 flagged (0.93%),
  none dropped**. Excluded from the Petrosian-radius nuisance control only
  (`probing.extract.NUISANCE_FLAG_COLS`); a corpus missing the column refuses that control rather
  than running it uncorrected. Most flagged objects are real large nearby disks, not failures —
  see D6.
- [x] (P0) **Pixel-validity detector** (`data/validity.py`) — bit-identity to a 4-neighbour in
  every channel, components below 256 px dropped (the floor is measured: 14,486 regions over 800
  stamps are *disjoint* in size, nothing between 33 and 255 px). 13.8% of pretrain stamps carry
  cutout padding; every qualifying region in both corpora is edge padding, interior count zero.
  *(Brief E1)*
- [x] (P0) **Mask/padding exposure, measured with the real sampler** — 0.22% of target blocks are
  ≥50% invalid in the worst case (β=0, the pure-I-JEPA control). Inside the pre-registered
  "< 1% ⇒ negligible" branch, so **the sampler is unchanged** and β=0 keeps its meaning as the
  published control. *(Brief E2/E3)*
- [x] (P0) **Normalisation fitted once and frozen** — valid pixels only, whole pretraining corpus
  less a 0.1% heaviest-stamp trim (fit only; 827 stamps, sha-pinned). Closes the per-run refit
  whose seeded subsample moved when the corpus grew 10k → 827k. Refitting refused, no escape
  hatch. Stability: 0.0% of 200 disjoint halves breach the 1% tolerance. *(D16)*
- [x] ~~(P1) Rotation/reflection augmentation pipeline (symmetry, augmentation-first).~~ *(D10)*
  **Superseded by D10's revision (Brief S4, 2026-09-23):** no rotation/reflection augmentation across
  the encoder family; orientation measured (R3), not removed.
  **Divergence, recorded under D10 (2026-09-22):** decided, never built — every checkpoint on the
  probe ladder, M included, trained without it. Brief R3 sizes the orientation nuisance this leaves.
- [x] (P0) **Scale the data layer to the full corpora** — the fp16 parity cache is baked over
  **both** corpora under the one frozen pipeline (`pipeline_hash 9f88ddefe946`, index recording
  `normalisation_hash 75100066b3e0`): **1,057,326 stamps, 415.8 GB, 3.88 h at 76 stamps/s**,
  drive-bound not compute-bound. Full coverage (826,968 + 230,358, no duplicates, corpora
  disjoint), index and data file agree exactly, and raw→pipeline reproduces the cache
  **bit-identically on 5,000 sampled stamps, 2,500 per corpus**. Sharing one hash-keyed directory
  across both corpora *is* the parity rule made physical. *(Brief E6)*
- [x] (P0) **Loader ceiling measured, drive and pipeline separately** — the number the model can
  never exceed. *Device layer*, `F_NOCACHE` set so the page cache is bypassed and the figure is
  the drive's rather than RAM's: external USB SSD **1,383 stamps/s shuffled (544 MB/s)** vs 1,440
  sequential — **shuffled access costs nothing on this SSD**, so the training access pattern does
  not collapse; internal SSD 3,837 shuffled (1,509 MB/s) as the ceiling reference, 2.8× the
  external. *Pipeline layer*, the real `TensorCache`→`StampDataset`→`DataLoader` at batch 32,
  `num_workers=0` (what `_prepare` builds): **479 stamps/s shuffled, 551 sequential** — 35% of
  what the drive can deliver, so the Python per-item path, not the USB link, is the data-side
  constraint. No warm-cache spike; twelve 10-second windows flat at 458–503. *(Brief F1)*
- [x] (P1) **The 4.07 GB metadata table is gone — one aligned float64 array instead.** Was:
  `harness._prepare` handed `StampDataset` both corpora's full tables (1,057,326 rows, 4.07 GB
  peak) so `__getitem__` could read one key, `petroRad_r`. Now `petro_rad_arcsec.f64` sits beside
  the cache — one float64 per stamp **in the index's own row order**, **8.46 MB**, sha vouched for
  by `index.json` (the cache's commit point) and written last. Misalignment is refused, not
  tolerated: `write_scalars` rejects any indexed object absent from the map ("a missing galaxy is a
  misalignment, not a missing value"), `load_scalars` rejects an unvouched sidecar, a missing file,
  a length mismatch or a digest mismatch. Deliberately **not** in `pipeline_hash`, so the cache key
  is unchanged and nothing re-baked. Parity on the real 415.8 GB cache: **0/20,000 value
  disagreements under exact float comparison** (scrambled order) and **0/2,000 `box_to_token_mask`
  outputs differ** — float64 because at float32, 19,996/20,000 differed in the last bits and a
  parity claim needing a tolerance is not one. Whole dataset process now **0.43 GB** incl. torch.
  *(Brief G2)*
- [ ] (P1) **`num_workers > 0` still refuses, and the reason is not the dataset.** Re-measured
  after G2 with an 8.46 MB dataset: `num_workers` 2 (twice), 4 and 8 all drive the machine into
  swap and are killed, while **each worker process holds 0.01 GB** and the tree's total RSS stays
  flat near 1.5 GB — so the memory leaves the machine outside anybody's RSS, with anonymous pages
  system-wide falling 7.7 → 3.9 GB as the kernel evicts other processes. The control that pins it:
  **two *independent* single-process readers** of the same 415.8 GB cache, concurrently, cost
  **−0.23 GB of swap** and aggregate to **681.8 stamps/s (1.39×)**. So the refusal belongs to
  DataLoader worker IPC, not to the dataset and not to multiple mappers — and
  `torch.multiprocessing.get_all_sharing_strategies()` is `{'file_system'}` on macOS, so the usual
  `file_descriptor` remedy **does not exist here**. It buys nothing locally either: the model
  consumes 34–41 of the 489.6 stamps/s the single-process loader delivers. If a faster machine ever
  needs it, the shape that measurably works is *independent sharded reader processes*, not workers.
  *(Brief G3)*
- [ ] (P2) **Flagged, not acted on: SDSS run 1000 and friends.** The 827 stamps the normalisation
  fit trims are low-SNR, not bright — 67.4% from run 1000, trimmed at 111× the corpus rate, with
  4× the corpus rate of failed Petrosian fits. An imaging-quality problem, not astrophysics. No
  GZ2 probe galaxy comes from those runs, so the probing corpus is untouched; decide separately
  whether the *pretraining* corpus should drop them. *(docs/spec/data.md §1.3)*

## Epic C — Masking & bounding box `[P3]` — per `docs/masking.md`
- [x] (P0) **Average-image bbox** (mean cutout, threshold τ, centred → fractional → `G×G` mask).
- [x] (P1) **Per-galaxy Petrosian-scaled bbox** (`data/bbox.py`), with the global-box fallback and
  a `fallback_rate` to watch at the gate.
- [ ] (P1) Re-tune EMA/masking-ratio per β — β=0 tuning does **not** transfer to β=1.
- [x] (P0) Token weight map `w` from box + bias strength β.
- [x] (P0) Bounding-box-biased multi-block sampler (M=4 targets, I-JEPA scale/aspect). *(D5)*
- [x] (P0) **Degradation test**: β=0 reproduces standard I-JEPA (property-tested).
- [~] (P1) Diagnostics: `sky_waste` exists; the β-sweep curve and mask-overlay visualisations do not.

## Epic D — JEPA model `[P4]` — **done**
- [x] (P0) Clean ViT-S/16 context encoder (256² → 256 tokens), from-scratch. *(D2, D4)*
- [x] (P0) Predictor + EMA target encoder + latent-MSE loss.
- [ ] (P0) **Overfit-one-batch sanity check** — still not written as a gate.
- [x] (P0) **Collapse monitor** (std / effective rank / mean cosine; halts on collapse).

## Epic E — Pretraining loop `[P5]`
- [x] (P0) Config-driven loop — `harness.run_harness`, one `HarnessConfig` determines a run.
- [~] (P1) Sweep harness — `harness.calibrate` measures compute- vs data-bound and batch scaling;
  no EMA/masking-ratio grid yet.
- [x] (P1) Checkpointing + frozen-encoder export (`load_frozen_encoder`, freeze boundary on disk).
- [x] (P0) **Mid-run checkpointing, with the resume proven identical.** `callbacks/checkpoint.py`:
  online encoder + **EMA target encoder as separate state** (re-deriving it from the online encoder
  would restart the EMA and look like a clean resume) + predictor + AdamW moments + `step` (which
  *is* the position in both the LR warmup and the EMA cosine ramp — neither is a stateful scheduler)
  + the schedule + RNG states (torch/MPS/numpy/python) + `config_hash` **and**
  `normalisation_hash`. Written `.tmp` → fsync → `os.replace` → **fsync the directory** → sha256 →
  read back via `mmap` → **then** the manifest, which is the commit point exactly as `index.json` is
  for the cache; `keep=3`. Eight refusals tested, including a resume that would cross a freeze or a
  changed schedule, plus a **refusal to resume without a `ResumableShuffle` sampler** — the weights
  would restore and the data order would not. **Resume identity is an equality, not a tolerance**:
  20 steps straight vs 10-stop-resume-10 from a *different weight init*, `second.losses ==
  whole.losses` on all 20, because the trajectory is a function of `(seed, step)` alone (per-step
  mask seeding, no dropout, seed-pure data order). Measured cost: **380.6 MB, 2.34 s/write**, so at
  `checkpoint_every: 1500` → **0.167% of wall-clock, 23.4 min at risk**. `0` is stamped as the
  `no_mid_run_checkpoint` forfeit. *(Brief G1)*
- [x] (P0) **`RunStamp.seed` now actually determines the encoder.** Found off the brief, while two
  G3 throughput runs at the same seed produced different collapse traces: **nothing in the package
  ever called `torch.manual_seed`**. `seed` reached the masker (`loss_step(seed=cfg.seed + step)`)
  and, since G1, the data order (`ResumableShuffle`) — but the ViT's parameters came from whatever
  ambient RNG state the process held, so two runs with byte-identical stamps produced different
  encoders. `harness.seed_init` is now the one construction site and seeds before the encoder, which
  covers its deepcopy into the EMA target and the predictor too. The resume-identity proof was never
  affected: a restore overwrites the init entirely. Three tests. *(found during Brief G3)*
- [x] (P0) **The collapse kill criterion is pre-registered, not judged at hour thirty.**
  `CollapseFloorFreeze` is a `FrozenChoice`: erank < **5.0** for **3 consecutive** readings after a
  grace of **10% of `steps`**, plus a **hard floor of 2.0** from step 100. 5.0 is half the pilot's
  ~10.3 — the only erank in this project tied to a working probe (AUC 0.905) — and deliberately
  **not** the smoke's 4.1, which would be reading the threshold off the curve it judges. Hashed into
  `config_hash` (`157903bd5180788b…`), enforced in `CollapseMonitor`, stamps `collapse_floor_open`
  when unset. Confirmed on the real trace: erank 22.14 → 3.23 over 300 steps with `would_halt=False`
  throughout, because 300 ≪ the 5,000-step grace. Explicitly **provisional** — the pilot is one
  trace on a different corpus; re-derive from the first real run as a new freeze. *(Brief G5)*
- [x] (P0) **Training smoke on the M3 Pro, measured not estimated** — the real path end to end
  (bbox-biased `MultiBlockMasker` → ViT context + EMA target → predictor → latent MSE → EMA
  update → collapse monitor) over the real 415.8 GB cache; 300 steps after a 20-step warmup;
  `smoke=True` so the artefact can never be read as a result. **1.297 steps/s = 41.5 stamps/s at
  batch 32**, **compute-bound: 83.8% compute / 16.2% data-wait**, cross-checking against F1's
  479 stamps/s loader ceiling (the model draws 8.7% of it). MPS driver peak **8.07 GB**, host RSS
  4.38 GB. Largest batch that **fits 128** (OOM at 192 against the 14.3 GB recommended cap);
  **fastest is 64 at 52.8 stamps/s**, and 128 is slower than 64 — so batch 64, not the configured
  32, with accumulation if a larger effective batch is wanted. Host-side mask work is 5.8% of
  compute, effectively all of it `MultiBlockMasker.sample`. Loss finite throughout. *(Brief F2)*
- [x] (P0) **A training run is now markable as a smoke** — `HarnessConfig.smoke`, mirroring
  `ProbingConfig.smoke`: a determining field, so it moves `config_hash` and a smoke's artefacts
  cannot collide with a run's, *and* written into `escape_hatches_used` so the stamp says so in
  words. The probing path has carried this since the D-series; the training path had nothing.
- [x] (P0) **Exactly one op in the whole training path has no MPS kernel** — `aten::_linalg_svd.U`,
  i.e. `torch.linalg.svdvals` in the collapse monitor, the one thing the pilot is said to read.
  Now relocated to the CPU **explicitly** (`callbacks/collapse.py`), on a matrix at most
  (batch, embed_dim), so a 300-step run completes with `PYTORCH_ENABLE_MPS_FALLBACK` **unset**.
  That is the point: with the blanket variable set — as the pilot must have had it — any future
  unimplemented op would quietly move to the CPU instead of raising, which is precisely how a
  silent fallback hides. `probing/entanglement.py`'s two `svdvals` calls force float64 and were
  therefore always on the CPU. *(Brief F2.3)*
- [ ] (P1) **The step budget is not decided, and `steps: 50000` is 1.97 epochs.** At batch 32 one
  epoch over the **810,491** training stamps is 25,328 steps; the configured 50,000 is 1.6M samples,
  8.3× the pilot's 192,000. Re-measured at **1.067–1.165 steps/s** (G3; F2's 1.297 was on a cold
  machine and the 10% gap is compute-side but unisolated — thermal state or variance), so 50,000
  steps is **11.9–13.0 h**, 10 epochs 2.5–2.7 d, pilot-epoch parity (19.2) 4.8–5.3 d, 50 epochs
  12.6–13.7 d. **I-JEPA's schedule cannot anchor this** — checked against the paper (G4,
  `artifacts/g4_ijepa_schedule.md`): 600 epochs for ViT-B/L and 300 for ViT-H/14 at batch 2048, and
  the paper **never pretrains a ViT-S with I-JEPA at all**. Matching its 769M samples would be 24.0M
  steps = 214 days. Our batch is 64× smaller at the same 1e-3 peak LR, and `train_jepa` applies
  warmup only — no cosine decay to 1e-6, no 0.04 → 0.4 WD ramp — so borrowing the epoch count
  borrows the cost without the mechanism. The masking geometry *is* identical to the paper's, which
  is what the β = 0 control's integrity rests on. Argue the budget on samples-seen and the collapse
  trace. *(Brief F4/G4)*
- [x] (P1) **Collapse trace at 300 steps: effective rank falls to ~4 and flattens.** 22.6 → 10.5
  → 4.8 → 4.1 by step 175, with std rising 0.28 → 5.17 and mean cosine falling +0.948 → +0.697.
  It does *not* flatline immediately and the loss stays finite, so nothing is degenerate at this
  length — but the pilot held erank ≈ 10.2–10.6 out to 6,000 steps, so ~4 is lower than the one
  reference trace that ended in AUC 0.905. **Cause now identified — see the schedule item below.**
- [x] (P0) **The LR schedule is a measured cause of the rank fall.** *(resolved by H5 — see below)*
  Six 500-step arms, one seed, identical data order / masks / EMA / `steps`, varying only
  `(lr, wd)` per step (`artifacts/h2_schedule_arms.py`, read by `h3_read_arms.py`; findings in
  `artifacts/h3_schedule_verdict.md`). The causal question is settled four ways: erank@175 is
  **monotone in mean early LR across a 64× range** (1.08e-5 → 17.14, 7.08e-5 → 10.00, 8.66e-5 →
  9.46, 6.76e-4 → 4.96, 6.93e-4 → 4.84); **replicated by two different mechanisms** at matched early
  LR (`sqrt` lowers the peak, `warmup1250` only reaches it slowly — 9.46 vs 10.00); the comparison is
  **controlled to bit-identity** (`baseline` and `cosine` share a schedule through step 100 and their
  eranks agree to four decimals there, diverging only from step 125); and **monitor-batch std inverts
  the erank ordering exactly** across all six arms (baseline inflates **20.4×**, 0.304 → 6.201, at
  erank 3.75; `linear` only 2.0× at 12.50). So the failure mode is *not* embeddings shrinking to a
  point — they **grow in magnitude while concentrating into fewer directions**. **Three-quarters of
  the fall is inside the 100-step warmup**, while the LR is still ramping, not from sitting at the
  peak. Decay **mitigates rather than prevents**: `cosine` tracks baseline through the fall then
  recovers 4.26 → 5.87 as its LR anneals, so rank is LR-responsive in both directions. The **WD ramp
  is not the lever** (4.10 vs 3.75; floor crossed at the identical step 125). **Mean-cosine does not
  discriminate at all** — baseline's +0.715 sits between `wd_ramp`'s +0.661 and `cosine`'s +0.757 —
  so std, not mean-cosine, is the number to watch, and is a candidate addition to the G5 floor.
  `would_halt` was `False` everywhere, which is **structural, not a pass**: the soft floor's grace is
  5,000 steps, so inside 500 only the hard floor of 2.0 could fire.
  **Not resolved at 500 steps:** *which* schedule is best. On deepest loss ever the baseline wins
  (0.0476 vs `sqrt`'s 0.0713); at equal step 500 `sqrt` wins on both axes (loss 0.0757 vs 0.1878,
  erank 7.19 vs 3.75) because the baseline turned over at step 178 and is **rising** +0.063 over the
  final 100. And the pilot held erank 10.2–10.6 at the **same** 1e-3 on a 10k corpus, so the absolute
  level is not set by LR alone. `linear` has the best rank (12.50) and the worst loss (0.1945, 4.1×
  baseline, still descending) — it is the arm **rejected**, which is the trap this experiment was
  built to avoid.
  **Next:** the resolving run — `baseline` vs the H4 proposal, **3,000 steps, ~1.6 h for the pair** —
  then probe both frozen checkpoints, because the objective is AUC and 500 steps of erank cannot
  stand in for it. *(Brief H1–H3)*
- [x] (P0) **Proposed schedule — adopted as D17 after the resolving run.** The **reference recipe adapted**, each number by a
  stated rule and **none read off an H2 trace**: peak **1.25e-4** (√-scaling of I-JEPA's batch-2048
  1e-3 — √ not linear because AdamW normalises by the gradient's second moment, so linear scaling's
  SGD derivation does not apply); warmup **1,250 steps** (the reference's *relative* 2.50%, 15 of 600
  epochs, against our 0.20%); **cosine decay with both endpoints scaled** (peak → peak/1000, keeping
  the reference's 1000× range rather than compressing it to 125×). The **WD ramp is declined**: 500
  steps cannot speak to a regularisation schedule, and bundling an unmeasured change with two measured
  ones would make the result unattributable. Code cost is **small** — two `JepaConfig` fields and ~8
  lines replacing `jepa.py:269-271`, one site; `lr_final = 0.0` lands it inert so existing hashes
  don't move, and resume is unaffected because the schedule stays a pure function of
  `(step, cfg.steps)`. **The real cost is the G5 floor:** `CollapseFloorFreeze.derived_from` cites the
  pilot and the F smoke, **both taken at lr = 1e-3 with no decay**, so a recipe change leaves the
  floor mechanically consistent (H1 measured that every schedule edit moves `config_hash`) and
  **empirically ungrounded** — re-derive it from the resolving run as a fresh freeze. The **β sweep
  gets stronger** (β = 0 would then differ from published I-JEPA in β and the WD ramp alone, not in β
  plus three schedule respects); the **pilot comparison degrades** and should be recorded as an
  existence proof, not a like-for-like baseline. Needs a **D-series entry** with the scaling argument
  written out — drafted on 3,000-step evidence, not on 500. *(Brief H4)*

- [x] (P0) **H5 resolving run: the schedule is settled, and the loss was lying.** Two arms, 3,000
  steps, one seed, identical data order / masks / EMA / `steps`; both frozen encoders probed on the
  same 34,829 held-out galaxies. Decision rule pre-registered **before** the arms ran
  (`artifacts/h5_decision_rule.md`; the commit precedes the launch). **The D17 recipe wins on AUC
  with no interval overlap** — consensus **0.9358** `[0.9315, 0.9402]` vs **0.9043**
  `[0.8988, 0.9097]`; all-held-out 0.8420 vs 0.8084; ambiguous middle 0.6624 vs 0.6358. It also
  holds erank 11.77 vs 7.91 (minimum **7.59 vs 3.50**) and never let std exceed **4.07** where the
  baseline's peaked at **11.68**.
  **The finding to carry forward: the loss inverted the answer.** The baseline was **19× better on
  loss** (0.0164 vs 0.3168), on *both* framings — which agree at 3,000 steps where they disagreed
  at 500 — and lost the objective decisively. Its mean pairwise cosine ends at **+0.984**:
  embeddings 98% aligned, predictor and target co-adapted onto a shared mean component. **Low
  latent MSE against a moving EMA target is a collapse signature, not a score.** Selecting on loss
  would have kept the worse recipe.
  Three corrections to H2's 500-step reading: **`std_final` is the wrong summary** (both arms
  finish within 10% by opposite routes — the peak separates them 2.87×); **mean-cosine is not
  uninformative** (+0.984 vs +0.286 is the widest separation of any diagnostic, and it tracks the
  AUC — H2's finding was an artefact of stopping early); and **the baseline's 500-step picture was
  a transient** (bottomed at 3.50, recovered to 7.91, reached 0.9043 — a working recipe, not a
  straw man). The extension clause was **not** triggered and that was stated before acting: nothing
  was descending and the arms had separated. *(Brief H5)*
- [x] (P0) **`CollapseFloorFreeze` re-derived — the old value was falsified, not just outdated.**
  H5's baseline sat **below the 5.0 soft floor for 53 consecutive readings** from step 125 and went
  on to score AUC 0.9043. Had the 5,000-step grace elapsed, the frozen criterion would have killed
  a working run. New soft floor **2.5**, bounded *from above* by evidence (below 3.50, the lowest
  rank yet seen in a run that probed successfully) rather than derived as a fraction of a working
  level. **Weaker grounding than what it replaces**, and flagged as such in the freeze's own
  `rationale`: every trace this project holds that dipped low still worked, so the evidence cannot
  yet say where a genuinely dead run sits. Re-derive again from the first full-length D17 run.
  Neither floor has ever fired, on any run.
- [ ] (P1) **The cosine decay is adopted but untested.** Because D17's schedule is the real
  50,000-step one rather than a compressed proxy, the LR is still **99.7% of peak at step 3,000** —
  so H5 tested the *peak and the warmup*. The decay rides on the reference recipe's authority
  alone and is the weakest-supported third of D17. **Not to be tested with a compressed proxy** —
  a proxy answers a different question; the real schedule is exercised by the full run. Carried
  into the write-up under "limitations, not tasks".
- [x] (P1) **The probing path carried the metadata table the training path escaped — fixed.**
  `harness.evaluate_probe` built `rows_by_id(DirectorySource(probe_dir).rows)` over 230,358 rows
  × 132 columns and `LabelProvider` copied it: measured **1.49 GB → 2.99 GB**, to read columns the
  probing layer could name up front. It was killed for memory before embedding a single stamp.
  Now `data.cache.write_probe_columns` bakes the 81 columns `probing.extract.required_columns`
  declares into one index-aligned float64 block, with the digest in the index and a refusal on
  mismatch or length disagreement — G2's discipline, applied where it still hurt. Columns load
  lazily, so a single-feature probe never materialises the other seventy-nine. Parity is exact
  over 2,000,000 sampled values, and `evaluate_probe` now runs to completion on the 230k corpus at
  **2.43 GB peak RSS** for the whole process. *(found during H5, fixed in Brief I)*

## Epic F — Probing harness `[P6]` (frozen encoder) — controls interleaved
- [x] (P0) **L2 logistic concept-direction probe** → held-out AUC + bootstrap CI; unit-normalised
  weight = concept direction. *(D7)*
- [x] (P1) Mean-difference (CAV) direction + logistic-vs-CAV disagreement as an entanglement signal.
- [x] (P0) `[control]` **Selectivity (Hewitt–Liang)**.
- [x] (P0) `[control]` **Negative controls** — the five nulls (shuffled votes, random embeddings,
  noise-through-encoder, untrained encoder, sky/noise labels).
- [x] (P0) `[control]` **Core nuisance probes** — z, magnitude, Petrosian radius, SNR, PSF.
- [x] (P0) **Non-circular uncertainty geometry** — axis on consensus extremes, held-out middle
  projected, Spearman vs vote fraction; firewall enforced in `data/splits.py`.
- [ ] (P1) Confidence as a probe target (separate axis from the uncertainty test). *(D9)*
- [x] (P1) Ladder rungs 3/4: MLP capacity sweep under a selectivity ceiling; never standalone.
- [x] (P1) **Load path** — `harness.probe_frozen_checkpoint` hands `run_probing` a real frozen
  checkpoint + the baked cache + a `LabelProvider`. Smoked against the pilot encoder.
- [x] (P1) **Conditional-population probing (as a comparison)** — each feature probed across the
  full and consensus-conditional populations and compared; off-population galaxies are **not**
  masked away. Gate threshold is a per-run knob. *(D14)*
- [x] (P1) **Two-scheme feature experiment** — Scheme 1 (full-37 per-bucket) and Scheme 2
  (reduced graded-axis) as configs on one harness; **per-config BY family**; full-first order.
  Watch the Scheme-1 power confound. *(D14)*
- [ ] (P1) **Graded-axis existence test** — AUC vs Spearman/permutation (may collapse into the
  uncertainty geometry for that feature). **Open; blocks Scheme 2 only.** `require_testable()`
  raises rather than defaulting. *(D14)*
- [x] (P1) **t09 bulge shape** — one binary feature (boxy vs rounded), conditioned on edge-on
  **and** bulge-present; Scheme 2 family = 10. Three-per-answer at family 12 is *deferred*, not
  discarded — it is where a t05/t09 cross-branch consistency check would live. *(D14)*
- [x] (P0) **Effect-floor freeze gate** — the floor is set from the medium local run and frozen
  before the headline; `headline=True` is refused while `effect_floor_freeze` is unset, and an
  open floor is stamped into the artefact's ledger.

### Statistics — the five decisions are **grounded**, not placeholders
Transcribed into code (`probing/config.py`, `nulls.py`, `entanglement.py`, `uncertainty.py`):
existence = the real value against a chance null, with a **separate** effect-size floor on top
(two gates, both must pass); **multiplicity = Benjamini–Yekutieli** (correlated tests; FDR is the
right target for a discovery catalogue), family count **per-scheme**; permutation ≥10,000
two-tailed on the shuffled vote fractions; **MP edge for the actual matrix shape**.
- [ ] (P1) **Effect-floor value** — the mechanism is settled; the number is a scientific call,
  pre-registered before any verdict is read.
- [x] (P1) **Tie-handling, permutation test** — DECIDED: add-one estimator with ties counted
  into the tail, so the attainable floor is `1/(n+1)` and a permutation p is never zero
  (Phipson & Smyth); ties count against the observed value, the conservative direction.
- [ ] (P1) **Tie-handling, existence p at AUC ≈ 1** — still open (the ceiling case only).
- [ ] (P1) **`mlp.selectivity_ceiling` predicate** — not one of the five and not mapped onto any
  of them; still `# FLAGGED`. *Candidate recorded, not chosen:* give the MLP rung its own null
  from the untrained-encoder control rather than a fixed constant, so selectivity stays relative
  the way every other rung is.
- [ ] (P0) **The combined null is a point mass — measured, and it blocks the budget question.**
  Three of the five controls (noise-through-encoder, untrained-encoder, sky/noise) are *single
  AUCs*, not distributions; only shuffled-labels and random-embeddings resample. Because the
  combination rule takes the per-draw max, and the largest singleton exceeds every resampled
  draw, the combined null has **zero variance** (measured: `unique=1`, std 2e-16 at both n=50 and
  n=200). So the existence p can take only two values — `1/(n+1)` or `1.0` — and raising the
  draw count buys **no resolution**, only a lower floor. Fix the degeneracy before sizing the
  budget; a draw budget over a constant null is compute spent on arithmetic.
- [ ] (P0) **Null-draw budget** (blocked on the above). `nulls.required_null_draws` derives it:
  the BY rank-1 bar at family 37 is 3.216e-4, so the floor `1/(n+1)` needs **n ≥ 3,109**; family
  10 needs 585. Sized from **Scheme 1 and applied to both** (`BUDGET_FAMILY_SIZE`), so "different
  null resolution" can never explain a Scheme-1-vs-Scheme-2 disagreement. `assert_null_resolution`
  stays live regardless — it is what stops a future config change from silently producing an
  all-R3 catalogue that reads like a scientific null.
- [ ] (P0) **Cost reality.** Measured on this machine at N=3,000: **471 ms per draw per feature**
  (two logistic fits at (2136, 384); `multivariate_normal` is only 34% of one of them, so caching
  the covariance factor saves ~13%, not orders). Scheme 1 × 2 ladders: **30 h at n=3,109**,
  **290 h at n=30,000**. 30k is not affordable locally; the trade needs to be made explicitly.

- [x] (P1) **The effect floor still tests something relative in two places — closed by D25
  (Brief T1).** Every consumer is audited and classified in D25; the two relative ones are fixed.
  (i) The entanglement leg now judges D24 retention (A matched on B's vote fraction); (ii) the MLP
  decode no longer reads the floor, and R3 is not assigned until (iii) below exists.
- [ ] (P2) **Untrained-MLP bar for the MLP rung (D25).** R3 ⇔ the MLP clears D23's existence test
  against an untrained-MLP bar at K ≥ 20 (`nulls.K_MIN`), width chosen on an inner split, BY across
  the answers reaching the rung; nuisance clearance by the MLP's own retention. Costed at ≈ 7–8 h
  (`artifacts/t_findings.md` §T1). Until then every existence-failing answer reads R4 *MLP decode
  unadjudicated*.
  **Deferred to the rental runs (Brief U):** no current rung can reach R3 (every failing answer has
  all nuisances competitive), and the bar is reusable across encoders that share the architecture.
- [ ] (P2) **Wire D26's graded existence test into the package** when a Scheme 2 run through
  `probing/` is its second consumer: replace `schemes.GradedExistenceTestUndecided` with the
  endpoint-fit JT test (`artifacts/u3_graded.py`), and rewrite the schemes docstring's open-question
  paragraph to cite D26.
- [x] (P2) **U2's off-axis verdict has no state for a sign reversal** — *closed by D27 (Brief V1.4): `REVERSAL_STATE` for future runs; U2's record kept* (Brief U2): merger and bulge
  "obvious" read UNRESOLVED with raw +, partial − (both significant), and loose's significant
  negative partial goes unscored because the rule tests the positive direction only. Add a
  REVERSES state before the uncertainty geometry is re-run on another encoder.
- [x] (P1) **Pitch angle: unblocked and run (Brief Y, revised).** Public machine-measured tables
  replaced the data requests: the Hayes SpArcFiRe table (37,381 matched), PAnDa's Hart machine
  (2,941) and Yu & Ho 2020 (1,194) rows. Hayes is DIVERGENT with Hart (ρ 0.38) and BROKEN with
  Yu & Ho (−0.03). Winding's ordering tracks SpArcFiRe pitch beyond visibility, but no 2DFFT
  check confirms it. The encoder: VOTES BEYOND MEASUREMENT. `artifacts/y_findings.md`.
- [ ] (P1) **Re-run V2's Experiment D (B/T 2×2) with Y3's corrected design.** Y3's planted check
  showed V2's form (each partial controls the other *noisy target*) reads BOTH on two noisy copies
  of one quantity (0.63/0.65). V2's B/T "BOTH" (A_m 0.36, A_v 0.47) is unverified until re-run
  with cross-decoder controls and the matched shared null (`y3_science.shared_null`,
  `state_2x2`). A claim in `framing_a_claims.md` (A18) rests on it.
- [ ] (P3) **PyArcFiRe port — parked (Brief Y revised).** Only needed for galaxies outside the Hayes
  selection, or to measure on our own stamps. Licence: BSD-3-Clause (PyArcFiRe 0.1.1). Also parked:
  per-band centroid offsets against PC1/PC2, and test-time dihedral symmetrisation.
- [ ] (P1) **V3's PC1 and PC2 are UNEXPLAINED** (Brief W2). They hold 37% of the variance, and
  every tested candidate is negligible: padding, sky, noise, crowding, g − r, u − r and four stamp
  axes (joint out-of-sample R² ≤ 0.02). They are learned (≤ 0.10 against the untrained top 10),
  bounded and flat-topped. *Brief X1:* **not handedness, ORIENTATION-LIKE.** (PC1, PC2) transform
  as the x/y components of an image-plane vector (rot90 maps PC1 → PC2; the mirror flips PC1
  only), equally in smooth and spiral galaxies; 41% of M's variance is mirror-odd against 2–7%
  untrained (D10). The vector is invariant to integer and sub-pixel shifts and to frame, so its
  identity is still open. **Next:** the per-object PSF asymmetry (psField), and attribution maps of
  PC1 on a handful of galaxies. `x_findings.md` §X1.
- [ ] (P2) **Label-free discovery beyond PCA** (ICA, sparse dictionaries, SAEs). V3's plain-PCA
  held-out test found no concept that is an encoder component. The salience test needs a null
  with reachable power first: its shuffled-label null was too wide to fire (V3).

## Epic G — Baselines as controls `[baseline]` — *same probe ladder, all SDSS-trained*
- [ ] (P1) `[baseline]` **MAE** — reproduce the Wu & Walmsley recipe on our SDSS corpus. *(D12)*
- [ ] (P1) `[baseline]` **Contrastive (MoCo)** — same SDSS corpus; probe identically. *(D12 sub)*
  **Forward constraint (D10 revised, Brief S4): remove horizontal flips.** They are reflections; M has
  none, so a flipped MoCo arm would be reflection-invariant where M is not — an asymmetry in exactly
  the comparison D12 makes. Its other augmentations are objective-intrinsic and stay. Check the MAE
  recipe for flips too. Decide in the baselines brief.
- [ ] (P1) Cross-objective comparison table (rung per feature × objective).

## Epic H — Figures & eval `[P7]`
- [x] (P0) **FIG 1** — ladder / AUC bar chart (`eval.figures.figure_ladder`).
- [x] (P0) **FIG 2** — concept-direction cosine matrix (`figure_entanglement`).
- [x] (P0) **FIG 3** — uncertainty-geometry scatter (`figure_uncertainty`).
- [x] (P1) **FIG 4** — controls panel (`figure_controls`).
- [ ] (P1) Label-efficiency curve (SSL-pretrained vs supervised-from-scratch).
- [ ] (P1) v1-comparable evaluation, and the v1-vs-v2 comparison as a first-class deliverable.

## Brief I — SIGReg `[D18 adopted]`
- [x] SIGReg on at lambda=0.05 in `configs/pretrain.yaml`; `config_hash` b5acc6779df49070.
- [x] Soft rank floor scoped to non-SIGReg runs (not deleted — D12's arms still need the gate).
- [x] Probing path's metadata table replaced by an index-aligned column sidecar.
- [ ] (P1) **Keep a lambda=0 arm in the beta sweep.** beta=0 with SIGReg on is no longer the
  published-I-JEPA control; the control now needs both knobs off.
- [ ] (P1) **D12 framing, unsettled.** SIGReg gives the JEPA arm a distributional constraint MAE
  and MoCo lack, so "of course its geometry differs" is a fair objection to a cross-objective
  consistency argument — but entanglement *surviving* enforced isotropy is stronger evidence it
  is in the data. Decide when D12 is written: run the JEPA arm both ways, restrict the claim to
  non-geometric read-outs, or state the asymmetry and argue the second point.
- [ ] (P2) Question 2 is underpowered — n=6 per arm cannot resolve |rho| < 0.886. A real answer
  needs many more checkpoints, or the paper's across-run design over a hyperparameter sweep.
  The *penalty term alone* was the most informative component (-0.31, -0.71); worth a powered test.
- [ ] (P2) The attachment point was chosen, not tested: final-block post-norm, or per-token rather
  than pooled, are separate arms.

## Brief J — the medium run `[smoke, 50,000 steps]`
- [x] 50,000 steps end to end, 11.09 h, no halt, zero resume gaps, 34 checkpoints retained.
  Stamped `v2:bb9945b617b53e3b`, `escape_hatches_used: ["smoke"]`. See `artifacts/j_findings.md`.
- [x] `run_harness`'s post-train probe read the 4 GB metadata table — the third call site Brief I
  missed. Now the sidecar. `traces.json` now persists the loss decomposition.
- [x] (P0) **The sky-noise control made the existence gate unreachable — corrected as D19.** 3C-5
  measured 0.8355-0.8416 on every feature and entered the per-draw maximum, so **every feature
  failed existence on this encoder, featured-ness included** (0.8365 under its own control's
  0.8373) — an all-R3/R4 catalogue that reads like a scientific null and is nothing of the kind.
  Four of the five controls break something and are chance-calibrated by construction; 3C-5 keeps
  images, encoder and probe real and swaps in a different real label, so it measures nuisance
  content, and came out **bit-identical** to the `snr` nuisance probe on all six features. It is
  now a diagnostic: `nulls.existence_null_samples` (renamed) takes the max over the four
  chance-calibrated controls. Spec corrected in `.tex` §3C + register item 8,
  `probing-harness-design.md` §3C, `spec/gates.md`. **The spec PDF is stale** — no LaTeX toolchain
  here; rebuild it from the corrected `.tex`.
- [ ] (P0) **The representation encodes observing conditions more strongly than morphology, and
  that promotes matched evaluation into Paper 1.** Nuisance panel on the frozen embedding:
  magnitude 0.8733, size 0.8501, SNR 0.8373, redshift 0.7918, PSF 0.5813 — against featured-ness
  at 0.8365 and every other morphology feature below 0.74. **Scope change:** 3D-ii specifies
  matched evaluation as *targeted* — "fires only for flagged features", which is what bounded its
  cost and let it be promoted from "Paper-2/if-feasible" to "Paper-1, targeted". On this evidence
  it fires for **every** feature on **three or four** nuisances each, so it is not a targeted
  contingency but a load-bearing component of Paper 1, and its cost is a headline-run cost rather
  than a tail. `matching.py` moves from formality to critical path. Record it in the spec (3D-ii's
  "targeted, bounded cost" claim is now measured to be false on this encoder) before the headline
  run is budgeted. **RE-TESTED on M's encoder (Brief N1): still false, and worse — every nuisance
  rose, PSF 0.5813 -> 0.8186. Featured-ness now beats SNR/redshift/PSF but not magnitude or size.**
- [ ] (P1) **Training longer made the representation worse, and the budget is unexplained.** This
  run is Brief I's `sigreg_050` arm continued (verified: all three loss traces agree to 5.0e-7
  over 3,000 steps; every collapse reading identical at every shared step, step 0 included). One
  trajectory, two stopping points: 0.9470 at step 3,000, 0.9278 at 50,000, intervals separated,
  all three framings agreeing. The prediction term bottomed at ~step 2,000 and rose 28% while the
  SIGReg penalty fell to 1.086 against a measured isotropic floor of ~1.0. Candidates to separate:
  over-training under the EMA/SIGReg trade-off, the cosine decay's long tail at a tiny LR, SIGReg
  dominating once the constraint saturates, or 1.93 epochs vs 0.12. **This does not license
  AUC-based checkpoint selection** — 1C stands; Brief I measured that question and found no usable
  signal.
- [ ] (P1) **The untrained encoder is the binding null on every feature.** Margins: edge-on
  +0.0952, featured +0.0457, t10 loose +0.0453, tight +0.0267, t09 boxy +0.0176, t10 medium
  +0.0000. A random-init ViT-S reaches 0.7908 on featured-ness. The measured structure argues for
  a **margin over the untrained control** rather than an absolute `effect_floor`; that is a change
  to a mechanism the spec records as settled, so it is raised, not made. **ANSWERED by Brief N2:
  recommended AGAINST.** The bar is architecture-determined (M's untrained nulls are identical to
  J's, to four decimals), a margin is a tightening of *existence* rather than a second gate, and it
  reorders the catalogue by a random network's luck. See `artifacts/n_findings.md`.
- [ ] (P1) **Effect floor: proposed NOT to freeze yet.** Its input population is empty while
  existence is blocked; the encoder is not the best on its own trajectory; an absolute AUC cannot
  encode a per-feature untrained baseline spanning 0.5160-0.7908; and n=6 cannot locate a
  threshold. Candidates recorded in `artifacts/j_findings.md` — 0.7908 (the untrained ceiling) is
  the one whose meaning survives questioning. `effect_floor_freeze` stays `None`.
  **SUPERSEDED by Brief N2** — objections 1 and 2 are resolved (bar fixed, encoder settled), 3 is
  answered against the margin form, 4 stands. Proposal: **form (a) absolute, 0.7267**, band
  (0.6538, 0.7995]. Still not frozen.
- [ ] (P1) **`EmbeddingMatrix.index` is quadratic at every call site — a one-line fix worth
  ~2.8 h of a 6 h ladder run.** It is a plain `@property` rebuilding a `{object_id: row}` dict on
  every read (5.6 ms over 74,829 entries), and `probing.extract.feature_ids` reads it *inside a
  comprehension's condition*, so it is rebuilt once per element. Measured: filtering 40,000 ids
  costs 8.6 ms hoisted, 225 s not — a factor of 26,000. `build_feature_controls` does eight such
  filters per feature (six `feature_embeddings` + two `feature_ids`), four over 40,000 ids and
  four over 34,829, so ~1,680 s per feature; over J4's six features ~10,080 s of its 21,598 s
  total. I had attributed that time to memory-compressor thrashing — the thrashing was real and
  additional, this was underneath it. Fix: bind `index` once in `feature_ids`, or make it a
  cached attribute on the frozen dataclass. Production code on the critical path of every ladder
  run. **Recorded, not acted on** (Brief K3).
- [ ] (P1) **`code_dirty` on the run stamp can be wrong about the code it describes.**
  `RunStamp.create` shells out to `git status --porcelain` at `_make_stamp` time, which
  `run_harness` reaches *after* `_prepare` — 72 minutes into the J3 run. An `artifacts/` file
  edited at 21:30, three minutes after the run started, was recorded as dirtiness of the code the
  run executed (stamped at 22:39). Two distinct failings: **timing** (capture at
  `_make_stamp`, not at process start) and **scope** (`--porcelain` covers the whole tree,
  including `artifacts/`, which is excluded from lint/CI and never imported by the package, so it
  cannot change a number the run produces). A stamp that can be wrong about its own code is the
  same class of defect as the three reproducibility holes already closed. Fix: capture
  `(sha, dirty)` once at process start and pass it down, and narrow the status check to the
  package + configs. **Recorded, not acted on** (Brief K3).
- [ ] (P2) The 72-minute `_prepare` setup is unbudgeted and unmeasured elsewhere: two
  `DirectorySource` passes over 1.06 M rows, `resolve_corpora`, two cache scans, the sidecar write.
  Worth knowing before costing any multi-arm sweep. **Related and confirmed, not a defect:** the
  full-table build audit asked for in K3 found **no fourth site**. Every probe-path consumer
  (`run_harness`'s post-train probe, `evaluate_probe`, `run_probing`) now goes through
  `_probe_rows`. Two full-table builds remain, both bounded and deliberate: `_prepare`
  (`harness.py:463-464`), which is the *producer* of the sidecars and so cannot read them, and is
  explicitly `del`'d before training begins; and `_probe_rows`' own fallback, which fires only
  when no sidecar exists and logs a warning when it does. `artifacts/f1_loader_bench.py:223` is a
  deliberate benchmark of the old path.

## Brief K — the correction and the trajectory `[no training run]`
- [x] **K1 — 3C-5 removed from the existence bar (D19).** `nulls.five_null_samples` →
  `existence_null_samples`, max over the four chance-calibrated controls. 3C-5 stays on
  `FeatureControls`, stays in the nuisance panel, and is renamed `sky_noise_diagnostic` in
  `ladder_summary.json` so the artefact cannot read it back as a bar. Spec corrected in three
  places with the reasoning written out; D19 records that **every feature failed under the broken
  bar, featured-ness included**. `artifacts/j5_floor_evidence.py` corrected too — it pooled 3C-5
  into the floor's null ceiling, which moves from 0.8416 to **0.7908**.
- [ ] **Rebuild `docs/galaxy-jepa-spec.pdf` from the corrected `.tex`.** No LaTeX toolchain in
  this environment; the PDF is canonical and is now stale against its own source.
- [x] **K2 — the trajectory probed, both replication checks exact.** Eight checkpoints on J4(A)'s
  split and probe config (step 3,000 -> 0.9470 = I2; step 50,000 -> 0.9278 = J4(A), point and both
  CI ends to four decimals). Full table in `artifacts/k_findings.md`.
- [ ] (P1) **The degradation is monotonic from the earliest checkpoint, and the cosine decay's
  temporal signature is falsified.** 0.9477 (step 1,500) -> 0.9278 (50,000), every step down, all
  three framings agreeing. **91% of the loss happens by step 27,000, while the LR is still above
  half of peak; the last 23,000 steps — the whole window in which the cosine collapses — cost
  -0.0017.** The decay looks like what arrests the decline, not what causes it. Consistent with
  SIGReg saturating (penalty 1.988 -> 1.080 against a measured isotropic floor ~1.065, erank
  24.3 -> 57.6, cosine +0.226 -> +0.025, all saturating on the same schedule), but **not a
  controlled separation**: every quantity is monotone in step, so rank correlations among them are
  +/-1 by construction; n=8, one trajectory, one seed; and epochs (0.058 -> 1.94) are confounded
  with both suspects. The AUC peak is at or before the earliest checkpoint and cannot be bounded
  below. **Does not license checkpoint selection — 1C stands.** Candidates for an experiment that
  would separate them are recorded in `k_findings.md`; none proposed for launch.
- [ ] (P1) **Nuisance content falls FASTER than morphology — the re-allocation story is not what
  happened.** Nothing climbs while morphology falls. Excess-over-chance retained, step 1,500 ->
  50,000: morphology **-4.4%**, PSF -4.6% (near-noise base), redshift -8.7%, magnitude -10.9%,
  SNR -11.7%, size -13.7%. Morphology is the best-preserved informative axis on the trajectory.
  Mechanism candidate, not a finding: isotropisation strips the dominant low-dimensional
  high-variance directions first, and observing conditions are exactly that kind of structure, so
  SIGReg may be removing nuisance axes preferentially with morphology as collateral damage. One
  trajectory, no control arm.

## Brief L — is it lost, is it SIGReg, what fixes it `[parked mid-brief]`
- [x] **L0 — `EmbeddingMatrix.index` is cached (`functools.cached_property`).** Closes K3-iv: the
  property rebuilt a 74,829-entry dict per read *inside a comprehension's condition*, so filtering
  40,000 ids cost 225 s against 8.6 ms — a factor of **26,000**, about 2.8 h of one six-hour
  ladder run. Pinned by three invariant tests in `tests/test_probing_extract.py`, one of them a
  wall-clock bound the quadratic form cannot meet. `sigreg-ablation` pushed; the corrected 3C spec
  and D19 are on origin.
- [x] **L1a — eight checkpoints' embeddings banked.** 74,829 x 384 fp32 each, 882 MB, 1.71 h, on
  the SSD at `runs/l1_embeddings/full`; step 1,500 re-probes to 0.9477, reproducing K2 exactly.
  **Bank the embeddings.** K2 threw these away and this brief paid 1.7 h to re-make them; an
  extraction pass costing ten minutes per checkpoint should write its output to disk.
  `l1a_cache_embeddings.py` is idempotent, so a re-run is a no-op.
- [x] **L1 — the information is not merely less linearly accessible; it is being destroyed, and
  the nonlinear part goes faster.** MLP capacity sweep on all 8 banked checkpoints, both features,
  1,121 s. **Branch three fired** — the pre-registered "MLP declines MORE, say so rather than force
  a story". t01 linear **-0.0199**, MLP **-0.0238**. The nonlinear headroom collapses monotonically,
  +0.0081 -> +0.0042 on t01 and +0.0254 -> +0.0112 on t02, and the best width falls 512 -> 128 ->
  64: late representations have less structure left to exploit. **This kills the
  isotropisation-hides-it hypothesis** that motivated L1 — erank 24.3 -> 57.6 was not the same
  information spread thinner, or the MLP would have held. Both guards clean: probe-adequacy passes
  (+0.0081 at the peak), and the **selectivity ceiling never fires at any checkpoint** (controls
  0.4646-0.5333 against thresholds 0.5158-0.5496), so the FLAGGED predicate is not load-bearing
  here. The two earliest checkpoints are capacity-limited, which makes the measured decline a
  **lower bound** — conservative in the right direction.
- [ ] (P1) **A single-probe reading of this trajectory can carry the wrong sign.** t02's linear
  probe ends *higher* than it started (0.7261 -> 0.7320) while its MLP ends *lower* (-0.0083):
  linear-only, t02 looks mildly improved; with the MLP it lost a third of its nonlinear headroom.
  t02 also peaks at 6,000 rather than falling from the first checkpoint, so K2's "monotonic from
  the earliest checkpoint" is a **t01 fact, not a property of the run**. Carry both into the
  write-up.
- [x] **L2 — BRANCH 1: SIGReg is the cause.** lambda=0 to 10,500 steps, no halt, 1.515 steps/s.
  **lambda=0 rises monotonically (+0.0361); lambda=0.05 falls monotonically (-0.0065).** At 10,500
  lambda=0 reaches **0.9554** against 0.9412 — higher than lambda=0.05 reaches anywhere on its own
  50,000-step trajectory. `t02` agrees: +0.0962 against +0.0426. Only `sigreg_lambda` differed.
  Continuity proved by measurement rather than by hash: step 3,000 probes to **0.9358**, exactly
  D18's recorded lambda=0 figure to four decimals and both interval ends.
- [x] **The crossover sits between 3,000 and 6,000 — immediately past D18's horizon.** At 3,000
  lambda=0.05 leads 0.9470/0.9358 (D18's evidence, correct); at 6,000 they are level; at 10,500
  lambda=0 leads with intervals apart. **D18 did not misread its data — it read data that stopped
  one regime short of the one the decision would run in.** Recorded as **D20** in the
  branch-independent form, with the D18-specific verdict added because branch 1 supports it.
- [x] **L3 — `sigreg_lambda: 0.05 -> 0.0` (D21), reversing D18.** Of the three candidates, only
  disabling SIGReg is supported by a measurement at the required horizon; accumulating embeddings
  across steps and reducing lambda are hypotheses about *why* it hurts, and adopting either would
  repeat D18's pattern. The soft rank floor is **active again** and the config says so. 338 tests,
  ruff, mypy green; the two pinned config hashes moved to `7ecf5dce5a1f60ba` / `de87b8f9704b7e2d`
  with the reasoning in the tests, and D17's stripped anchor `538bf997880a8767` is **unchanged**,
  which is what proves nothing outside the SIGReg block moved.
- [ ] (P0) **The adoption is PROVISIONAL and the missing measurement is named.** D21's evidence
  reaches 10,500 steps; a headline run is 50,000. lambda=0.05's decline was invisible at 3,000, so
  nothing proves lambda=0 has no turn of its own later, and its effective rank was still climbing
  at the last reading (8.1 -> 18.6). **Run a lambda=0 arm at the headline horizon before the
  headline.** Adopting for 50,000 on 10,500-step evidence is D18's error at a longer lever arm.
- [ ] (P1) **Test the batch-32 estimator hypothesis, as an experiment and not an adoption.** I1
  pre-registered the risk that the Epps-Pulley statistic is badly estimated from 32 samples in 384
  dimensions. The principled fix is to accumulate embeddings across steps so the statistic sees an
  effective sample far larger than the batch, without touching the batch. **Validate at >=10,000
  steps against the lambda=0 arm before any adoption.** LeJEPA's own result stands in its regime —
  batch 2048, eight views — and this is not evidence against it.
- [ ] (P1) **lambda=0 embeddings are memorisable at every MLP width, early.** On 3 of 4 lambda=0
  `t01` checkpoints the shuffled-label control clears the selectivity threshold at every width down
  to 16, so **no admissible nonlinear reading exists**; across all eight lambda=0.05 checkpoints it
  never fired once. std is 3.1-4.6 against 0.9-1.0. The effect fades as lambda=0 trains — by 10,500
  the ceiling no longer fires. Consequences: **MLP readings are not comparable across the two
  arms**, and the ladder's R3 rung will behave differently under the reverted recipe.
- [ ] (P1) **A driver must not die on its own guardrail.** `l1b` crashed when
  `mlp_best_sub_ceiling` came back `None` — the ceiling firing at the smallest width — because the
  print assumed a number. A guardrail firing is an OUTCOME, not an error, and the run that dies on
  it cannot report the thing it exists to catch. Fixed to report "no admissible MLP" as a result.
- [ ] (P1) **Two contention facts, measured, worth keeping.** (i) A CPU sweep beside a *training*
  loop costs **2.8x** (1.9 s/step against 0.68) because the loop's CPU dataloader feeds the GPU
  every step; beside an *extraction* pass it costs 2%. The two profiles do not generalise to each
  other, and the plan's claim that the MLP sweep could ride inside the training run was wrong.
  **L1b runs before or after L2, never alongside.** (ii) Repeated extraction over the same 29.4 GB
  of stamps speeds up as the page cache warms, 881 s -> 659 s; cost a multi-checkpoint probe on
  the later figure.
- [ ] (P1) **The machine is oversubscribed before any run starts.** Measured with every one of my
  own processes exited: 0.08 GB free of 19.33 GB, **39.72 GB of logical memory in the
  compressor**, 12.8 GB of swap. The arm degraded to 0.045 steps/s against 1.479 measured, with
  RSS collapsed to 11 MB — paged out and stalled, not working. Killing it moved swap by 190 MB,
  which is the proof it was never the consumer. L2 needs a quiet machine, and freeing it is not
  something the run can do for itself.
- [ ] **L3 — the fix, plus D20 (unconditional: an ablation's horizon must reach the regime the
  decision will run in) and D21 (the fix, with the measurement that chose it).** Blocked on the
  branch L2 names. Validate any fix at **>=10,000 steps before adoption** — D18's failure mode was
  adoption on 3,000-step evidence, and the point of this brief is not to repeat it.

## Brief M — the long lambda=0 baseline `[10 epochs, stopped early at 4]`
- [x] **M — the returns flatten at 4 epochs, and the rule caught it.** 253,270-step budget
  (810,491 train // 32 = 25,327/epoch), stopped at **101,308 (4.00 ep)** by the pre-registered
  rule: dAUC **+0.0010 then +0.0005**, both under 0.002, not rising. **~28 h of budget returned.**
  Consensus 0.9593 (0.5 ep) -> 0.9631 -> 0.9642 -> **0.9646** (4 ep). No halt; 19.3 h wall.
- [ ] (P0) **A real lambda=0 comparator now exists**, at a horizon that means something. Everything
  downstream is measured against it: the MAE/MoCo arms, the beta sweep, and the SIGReg re-test.
  M at HALF an epoch (0.9593) already beats anything lambda=0.05 reached anywhere on its
  50,000-step trajectory (best 0.9477, end 0.9278).
- [ ] (P1) **Effective rank rose 22.2 -> 37.3 while AUC ROSE.** Under lambda=0.05 rank rose
  24.3 -> 57.6 while AUC fell. Rank growth on its own is therefore not what costs morphology,
  which is consistent with L1 (information leaving, not spread thinner) and removes a suspect.
- [ ] (P1) **The 10,500-step provisional adoption in D21 is now covered to 4 epochs.** D21 named
  the missing measurement: evidence reached 10,500 steps while a headline run is 50,000. M reaches
  101,308 and lambda=0 rises throughout with no turn. The D20 caveat is answered for lambda=0 --
  though still on one seed, one trajectory.
- [ ] (P1) **No thermal drift over 19.3 h.** Throughput stepped down 5.5% between segment 1 and 2
  (1.529 -> 1.459 steps/s) then held flat for 14.6 h -- a step, not an accumulation, so memory
  pressure rather than heat. The machine ran ~34 GB of logical demand in 19 GB physical throughout.
- [ ] (P1) **The rental case is WEAK on this evidence.** Final slope **+0.0002 AUC/epoch**; another
  10x of compute buys ~+0.002 if the slope held, about one interval width, and a flattening curve's
  slope does not hold. Where compute would plausibly pay instead, none acted on: **batch size** (32
  against the reference's 2048, the largest unexamined divergence and the SIGReg hypothesis), a
  second seed to put an interval on the plateau, then the SIGReg re-test.
- [ ] (P1) **Three driver defects, all found by running rather than reading.** A 60-step plumbing
  test caught two in ninety seconds -- resume broken on MPS/CUDA (`map_location` relocates the RNG
  state and `set_rng_state` refuses it; the CPU-only resume test could never see it), and
  `probe_points` wrongly placed in the checkpointer's `schedule`, which made a run refuse to
  continue itself because we had changed our mind about when to LOOK. The third cost a crash but no
  data: the probe's output filename was inverted. **Observing a run is not part of its recipe** --
  the same distinction `monitor_every` needed, got wrong in the other direction.

## Brief N — the bar on a good encoder, and the floor's evidence `[no training run]`
- [x] **N1 — the controls battery re-run on M's 4-epoch encoder.** Same split as J4 (40,000 /
  34,829), same six features, declared before the numbers and unedited. All six improved:
  featured-ness 0.8365 -> **0.8845**, edge-on 0.7320 -> **0.7995**, arms-loose 0.6098 -> **0.6538**,
  arms-tight 0.5740 -> **0.5889**, bulge-boxy 0.5534 -> **0.5847**, arms-medium 0.5161 -> **0.5226**.
  Gains are largest at the easy end. The graded axis still dips in the middle (0.5889 / 0.5226 /
  0.6538) on a second independent encoder.
- [x] **The existence bar is a property of the ARCHITECTURE, not the run.** The untrained-encoder
  null came back identical to J's on all six features to four decimals, because
  `untrained_encoder_matrix` never sees the trained checkpoint. J's bar and M's bar are the same
  bar, so every margin gain is the encoder. Margins roughly doubled throughout.
- [x] **D19's fix works on a good encoder.** The sky-noise diagnostic sits at 0.8662-0.8692 and
  would, under the pre-K1 bar, still have failed featured-ness (0.8845) by a hair and everything
  else comfortably. It is out of the bar; the bar is the untrained encoder. Still bit-identical to
  `nuisance_aucs["snr"]`.
- [ ] (P0) **The nuisance panel did NOT go away — it got worse.** magnitude 0.8733 -> **0.9033**,
  size 0.8501 -> **0.9061**, SNR 0.8373 -> **0.8687**, redshift 0.7918 -> **0.8371**, PSF
  **0.5813 -> 0.8186**. Training longer at lambda=0 made the representation encode observing
  conditions *more* strongly, and faster than it improved morphology. Featured-ness (0.8845) now
  beats SNR, redshift and PSF -- it beat none of them on J -- but **magnitude and size still beat
  it**, and all five nuisances still beat all five other morphology features. **3D-ii's "targeted,
  fires only for flagged features" remains measured FALSE**; matched evaluation stays load-bearing
  for Paper 1 and `matching.py` stays on the critical path. Update the spec's scope note.
- [x] **N2 — the effect floor: evidence and a proposal, not a freeze.** `effect_floor_freeze`
  stays `None`; `configs/probe.yaml` untouched; `headline=True` still refused.
- [ ] (P1) **The binding null's variability, measured for the first time — verdict MARGINAL.**
  Three untrained seeds (0 primary, 1, 2). Per-feature range 0.0026 (featured) to **0.0209**
  (boxy), median 0.0073 -- inside the pre-registered 0.010-0.030 band, so the rule fixed before the
  numbers says **recommend the ABSOLUTE form**, which does not inherit the instability. No feature
  is strictly seed-determined, but **t10-medium clears its own bar by 0.0006**: under seed 1 its
  margin is +0.0006 rather than +0.0065, so its existence verdict is decided by which random
  network was drawn. No floor fixes that feature.
- [ ] (P1) **The margin form is recommended AGAINST, on four measured grounds.** (i) The per-feature
  ceiling is exactly `sup(existence_null_samples)` -- verified against the production function over
  200 cases -- so a margin is a *tightening of existence*, the one property 3B disclaims. (ii) It
  reorders the catalogue by a random network's luck: boxy has a lower real AUC than tight (0.5847
  vs 0.5889) but a higher margin (+0.0489 vs +0.0415), and at margin 0.10 it excludes
  **featured-ness**, the strongest feature, while admitting edge-on. (iii) Its verdicts move with
  the seed (flips at 0.05 and 0.09); form (a)'s cannot. (iv) `effect_floor` has five consumers and
  a margin has no definition at `ladder.py:121`, `:161` or `:248`. **No D22 was drafted**, because
  one is needed only if (b) is recommended.
- [ ] (P1) **PROPOSED effect floor: form (a) absolute, value 0.7267** -- the midpoint of the widest
  gap in the real spread and the point furthest from any flip (0.0729 either way). **The band
  matters more than the point:** every value in **(0.6538, 0.7995]** gives the identical partition.
  **Admits** featured-ness (0.8845) and edge-on (0.7995); **excludes** arms-loose (0.6538),
  arms-tight (0.5889), bulge-boxy (0.5847), arms-medium (0.5226). All six would still pass
  *existence* -- the floor separates clean from marginal among the real, which is 3B's job for it.
  **J5's fourth objection is unresolved and travels with the value: n = 6 cannot locate a threshold
  for a catalogue of 37.** The value is Malachy's.

### Recorded by Brief N, not acted on
- [ ] (P1) **The targeted compute case is BATCH SIZE, not a longer run.** M measured +0.0002
  AUC/epoch at the plateau, so another 10x of compute buys about one interval width. **32 against
  the reference's 2048 is the largest unexamined divergence in the recipe AND the standing
  hypothesis for why SIGReg failed here** -- those are one experiment, not two. Log it as the
  targeted compute case; a lambda=0 baseline at a real horizon now exists to test it against.
- [ ] (P1) **M's limits, carried forward.** One seed, one trajectory, four probe points. The
  plateau is *located* between 2 and 4 epochs, **not bounded**. A second seed at 4 epochs is the
  cheapest thing that would put an interval on it rather than a point.

## Brief O — the floor frozen, the confound answered `[no training run for O0/O1]`
- [x] (P0) **D22 — the effect floor is frozen at 0.7267**, form (a), absolute, chosen by structure:
  every value in (0.6538, 0.7995] gives an identical partition of the six probed features, and
  0.7267 is that band's centre. Spec register item 4 is closed. See `configs/probe.yaml`.
- [x] (P0) **O1 — matched evaluation run on M's 4-epoch encoder.** Six features x five single
  nuisances x one joint magnitude-x-size match. 30 of 36 SURVIVES, 6 COLLAPSES (all of them
  t10-medium), no PARTIAL, no UNRESOLVED. The joint match survives wherever the singles do.
- [ ] (P0) **DEFECT — `ladder._nuisance_clearance` does not apply `labels.nuisance_valid`.**
  `ladder.py:157-159` passes `labels.nuisance_value(worst, present_*)` straight into
  `match.matched_evaluation`, while `controls.build_feature_controls` (`controls.py:304-313`)
  filters the same vectors through `nuisance_valid` first. So the production matched re-probe
  builds `size` strata over `petrorad_suspect` rows — rows the flag exists to exclude, and which
  are systematically bright, nearby and featured, i.e. exactly the confound direction being
  matched away. **Measured consequence:** O1's driver applies the filter and drops 213-336 flagged
  test rows per feature from each size-matched set (336 on t01, 334 on t02, 242 on each t10 arm,
  213 on t09); production keeps them. Not fixed here, because `ladder.py` is on the verdict path
  and a change there belongs in its own brief with its own before/after.
- [x] (P1) **The effect floor cannot be the matched-survival bar.** **Closed by D24 (Brief S2):**
  the ladder applies O1's retention rule, K = 3 untrained draws for C and C_m. `ladder.py:121` passes
  `survive_threshold=config.effect_floor`; four of O1's six features sit below 0.7267 *unmatched*
  and would fail by arithmetic whatever matching did. O1 therefore used a margin-over-own-null
  read instead. Whichever a later brief adopts, this must be decided rather than inherited.
  **Measured consequence in Brief P (found in Brief R, 2026-09-22):** 18 of P's 19 "confounded by
  size" features, and 21 of 22 "confounded" overall, were already below 0.7267 *before* matching;
  matching moved their AUC by a median 0.021. P's "size dominates" headline was this arithmetic. Brief R0
  re-adjudicates all 37 under O1's retention rule (`matching.retention_verdict`), but production is
  unchanged — **owed a D-entry** before the next catalogue run.
- [x] (P1) **O1's retention rule has no floor under the unmatched margin.** **Closed by D24:**
  retention is judged only where the unmatched margin passed D23 existence, else UNRESOLVED. It is a ratio,
  (M − C_m)/(A − C). When A − C sits inside the untrained seed range, the ratio divides noise by
  noise. Brief R hit it on `t10 winding: medium`: A − C = 0.003 against a 0.006 bar spread, so
  linear read COLLAPSES and the MLP read SURVIVES, both on noise. It affects none of R0's 31
  survivors (every margin ≥ 2× its seed spread). Before the rule is reused, it needs a floor, e.g.
  A − C above the resolvable margin, else UNRESOLVED. That is a pre-registration change, so it
  belongs in its own D-entry.
- [ ] (P1) **Split variance is owed, and O2 deliberately excludes it.** An honest interval on a
  published AUC should cover which galaxies landed in the test set, not only which training draw
  was made. O2 holds the splits fixed so it can answer its own narrow question; the split-varying
  arm is a separate measurement, logged now so it is not rediscovered at write-up.
- [ ] (P1) **O2's driver deviates from the production path.** `run_harness` moves every seed
  together; O2's driver splits `config_seed` from `train_seed` on purpose. The deviation is
  confined to which seed reaches `seed_init` / `ResumableShuffle` / the masker, but it is a real
  caveat on transferring O2's interval to a production run.
- [x] (P0) **FIXED — the stopping rule labelled a DECLINING curve `FLAT`.**
  `m2_long_run._stopping_rule` uses `flat = d1 < FLAT_DELTA and d2 < FLAT_DELTA`, a **signed**
  comparison, so `-0.0069 < 0.002` passes. O2's curve fell from 0.9678 at 2 epochs to 0.9609 at 4
  and was reported as `FLAT`. Stopping was the right action; the label was the opposite of what
  happened. Needs `abs(d) < FLAT_DELTA` plus a separate DECLINING branch, since "stop, it has
  plateaued" and "stop, it is getting worse" mean different things to a rental case. M's own
  verdict is unaffected (+0.0010 then +0.0005, genuinely flat).
  **DONE (P1):** rule moved to `src/galaxy_jepa/core/stopping.py` with three branches
  (RISING / FLAT / DECLINING) plus UNSETTLED and INSUFFICIENT; symmetric band; DECLINING checked
  first; `stop` carried separately from `label`. `m2_long_run._read_rule` is a thin wrapper.
  O2's record relabelled in place with the original kept in `stop_reason_original`.
  `tests/test_core_stopping.py`, 10 invariants, both directions.
- [ ] (P1) **M's 4-epoch AUC needs its range attached, and "plateau" needs qualifying.** Across two
  training draws with splits held fixed the 4-epoch consensus AUC spans **[0.9609, 0.9646]**,
  range 0.0037. The stopping *location* reproduced (both stopped at 4 epochs); the *shape* did not
  (M flattens, O2 turns over, and O2's 2 -> 4 decline has disjoint CIs). Report M's 0.9646 with the
  range, and do not use "plateau" unqualified for a recipe whose second draw declined.
- [ ] (P1) **Prediction loss does not track probe quality — measured within one recipe.** O2's
  prediction loss is higher than M's at all four probe points while its AUC is higher at three and
  lower at the fourth; splits fixed, recipe identical, only the training draw differs. Sharper than
  the D18 Q2 non-reproduction because it is within-recipe. **Loss curves must not be presented as
  representation-quality curves**, in either direction.
- [ ] **"Settled" was overclaimed.** Past ~2 epochs the recipe's behaviour is draw-dependent:
  the stopping *location* reproduces, the *direction* does not. Say that, rather than describing
  the recipe as settled.

## Brief P4 — the schedule tension `[logged and proposed, NOT run]`
- [ ] (P1) **Early stopping and a full-budget cosine are structurally in tension, and it now
  blocks a clean reading of the plateau.** The cosine is defined over 10 epochs; both runs stopped
  at 4, where LR is ~65% of peak (~90% at 2 epochs, where O2 peaked). **The annealing tail — the
  reason cosine decay exists at all — never ran**, and O2's decline happens at high LR. So
  "flattens at 4 epochs" is really **"flattens before annealing"**. This is D17's untested third
  (see the write-up note below), no longer merely untested but actively confounding.
- [ ] (P1) **Candidate fixes — PROPOSED, NOT ADOPTED.**
  - **(a) Budget-matched cosine.** Set `steps` to the intended stopping horizon so the full decay
    runs inside it. Simple, and it makes the schedule honest about the budget. Cost: it re-couples
    budget and schedule, so changing the stopping horizon changes the recipe — the same rigidity
    `j1_preflight` already warns about ("budget is fixed").
  - **(b) Warmup-stable-decay (WSD).** Constant LR through a long stable phase, then a short
    cooldown branched from ANY stable-phase checkpoint. **Literature checked before proposing, as
    instructed:** WSD originates with MiniCPM and is now standard in LLM pretraining; the
    checkpoint-branching property is the documented reason it exists — one stable-phase checkpoint
    can be branched into multiple decay experiments without restarting, and a stable checkpoint
    plus a fixed-length decay is reported to match full-length cosine baselines. Typical shape:
    0.5-2% warmup, 80-90% stable, 10-20% decay. This decouples the stop decision from the
    schedule, which is exactly the tension above. NOT adopted — it is a change to D17, which
    needs its own D-series entry and its own argument.
- [ ] (P2) **Cheapest diagnostic, when the time comes — NOT NOW, and NOT before P2's matched
  work is finished.** Branch a short LR cooldown off M's and O2's 4-epoch checkpoints and probe
  both. If annealing lifts O2 back, the decline was a high-LR artefact and the tail matters.
  **Frame it as schedule diagnosis, never as headline-encoder selection** — picking whichever
  branch scores best is 1C by another route.

## Brief P — the full ladder `[in progress]`
- [x] (P0) **Q0 — the +0.984 cosine attribution corrected.** It was H5's baseline arm, not J under
  SIGReg (J ended at +0.0250; SIGReg forces isotropy). Recalibration recorded: high cosine is the
  UNTRAINED DEFAULT (+0.988) and learning means moving off it.
- [x] (P0) **Q1 / D23 — existence moves to the untrained-z construction.** The point-mass null gave
  BY nothing to act on. K=30 untrained seeds, Student t at df=K-1, both uncertainties in the
  denominator. `assert_untrained_bank_resolution` refuses below K_MIN=20; the empirical gate still
  bites under its own method. Declared in `configs/probe.yaml`.
- [x] (P0) **Power travels with every rung.** `resolvable_margin` per feature, stated as a margin
  over the feature's own bar. An underpowered R4 means "cannot resolve at this N", never "absent".
- [x] (P0) **Q3 — 2A's three gaps closed**: eigenvectors (localisation), the logistic-vs-CAV
  cross-check wired into the ladder, and the pre-registered `adjudicate_pair` verdict function.
  The MP null is now a gate input; the geometry is serialised rather than surviving as a PNG.
- [ ] (P0) **Q2 — run the 37-feature ladder.** Blocked on the untrained bank (~5 h, resumable).
- [ ] (P1) **Report the normality check.** D23 buys BY a usable p-value at the price of a
  distributional assumption; Shapiro-Wilk + QQ per feature at K=30, reported whatever it says.
  This is the weakest joint in the construction and must not be quietly omitted.
- [ ] (P1) **v1's Figs 18-19 are not vendored.** Only three prose constants exist in-repo (edge-on
  x cigar +0.83, bar x 2-arms +0.56, 3<->4 arms 0.16). A real overlay needs data pulled from
  `/Users/malachy/Documents/Galaxy-Zoo-Classifier` — **user action**.
- [ ] (P1) **D13's adjudicator is named differently in two places.** `DECISIONS.md:391-394` says the
  2A conditional cross-check; `spec.tex:477-487` says angle/depth invariance plus the literature.
  Flagged rather than silently resolved.
- [ ] (P2) **R3 is nearly unreachable for deep buckets.** `ladder.py` uses `effect_floor` as the MLP
  decode threshold, and features reach that branch BECAUSE they failed existence. A 0.7267 bar on a
  bucket near 0.55 is unreachable by arithmetic. Reported as a limitation; changing it is a
  mechanism change needing its own D-entry.

## Carried into the write-up — limitations, not tasks `[write-up]`
- [ ] **D17's cosine decay is adopted but untested.** At 3,000 steps the LR is 99.7% of peak, so
  H tested the peak and the warmup; the decay rides on the reference recipe's authority and is the
  weakest third of D17. **Do not test it with a compressed proxy** — a proxy answers a different
  question (how a steep decay behaves early). The real schedule is exercised by the full run.
- [ ] **The SIGReg loss-selection claim did not transfer** (D18 Q2). Report it as a
  non-reproduction in this regime, not as evidence against the paper: their rho is across runs
  over a hyperparameter sweep on ImageNet at eight views; ours is within one 3,000-step run at one
  view. The 1C label-blind checkpoint rule stands unchanged.
- [ ] **RECALIBRATE every mean-cosine figure on record — Brief O3 gives the origin.** An *untrained*
  encoder on this architecture already sits at mean-cosine **+0.988**. **High cosine is the untrained
  DEFAULT, and learning means moving off it** — so every cosine figure must be read against +0.988,
  not against 0. A run at +0.6 has moved a long way; a run at +0.95 has barely moved.
  **Applies to H5's baseline, NOT to J.** H5's baseline (pre-D17: lambda=0, LR 1e-3, warmup 100, no
  SIGReg) ended at +0.984 — it never learned to spread its embeddings at all, rather than collapsing
  into a shared component from a spread state. **That reinforces D17; it says nothing about SIGReg.**
  J under SIGReg ended at **+0.0250** (`j_findings.md:204`) — SIGReg forces isotropy, the opposite
  failure. An earlier version of this entry attributed +0.984 to J and was wrong.
  The signature of genuine learning, measured at O3's overfit floor, is mean-cosine **+0.413** at
  effective rank **19.7** and std **1.56** (from 0.109); effective rank dips to 9.4 mid-run and
  recovers, so the dip is a transient and not the signal. **Caveat:** O3's effective rank is measured
  on a batch of 32 and erank is bounded by sample count, so cosine is comparable across runs and
  effective rank is not.
- [ ] **The lambda-robustness result is a positive replication** worth reporting: an 8x difference
  in weight gave indistinguishable AUC (0.9470 vs 0.9471) on a corpus, architecture and objective
  the paper did not test.

## Epic I — arXiv sweep `[parallel]`
- [x] First pass → `docs/related-work.md` (gap confirmed cautiously; Wu & Walmsley MAE pinned).
- [ ] (P1) Fetch Wu & Walmsley MAE card (licence, resolution, patch size, corpus).
- [ ] (P1) Verify all `*verify*`-tagged arXiv IDs; ADS full-text search for JEPA-morphology.

## Epic R — Rung controls / ablations `[control]`
- [ ] (P1) **Masking β-sweep** {0, 0.5, 1.0} — β=0 is the control. Downstream of the headline run.
- [ ] (P1) **8×8-patch (higher-res) ablation** — Rung-4 control (under-resolved vs absent). *(D11)*
- [ ] (P1) **Backbone sweep** — clean ViT → conv-stem hybrid (CCT/CvT) → E(2)-equivariant ViT. *(D2)*
- [ ] (P1) **Inclination conditioning** — recoverability/entanglement as a function of axis ratio
  (b/a); the confound-fingerprint deliverable. A new axis on the existing probe, not a redesign.
  The columns are in the corpus; the conditioning run is not written. *(D13)*
  **Watch:** they are in `data/probe/metadata.csv`, **not** in the compact probe-column sidecar
  the harness reads (`_probe_rows` → `load_probe_columns`), so any run through `prepare` sees
  `expAB_r` as absent. Brief R3 streams it from the CSV; the sidecar wants the two columns added.

## Deferred — Paper 2 `[P2-paper]`
- [ ] Multi-survey corpus (SDSS + DESI Legacy → space-based); homogenisation (degrade-down first).
- [ ] Survey-leakage probe + single-vs-multi merge experiment.
- [ ] Fine-tuning comparison; confidence-aware read-outs.
- [ ] Deferred controls: matched evaluation at scale, cross-split robustness.
- [ ] E(2)-equivariant ViT ablation; GalaxyMNIST external comparison.
- [ ] Parking lot: SAE on frozen JEPA embeddings; Spectra/Jacobian mechanistic angle.
