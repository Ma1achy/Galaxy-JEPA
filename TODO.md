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
  withdrawn: v1 needed it because v1 trained on the labels, and v2's encoder never sees one.
  Probe-target noise is conservative — it attenuates toward chance and cannot manufacture a
  direction — so a feature clearing the gate unfiltered is a *stronger* result. Frozen at **1**,
  the minimum where a vote fraction is defined, via `VoteCountFreeze` (hashed, stamped, refit
  refused). Sweep **{1, 5, 11, 21, 37}** pre-registered as a robustness claim, not a selection
  step. *(D8 superseded)*
- [ ] (P1) **Vote count is not merely noise for the uncertainty geometry — decide a LOCAL floor.**
  Separate sub-system, deliberately untouched by the corpus-wide decision above. A galaxy at
  50/50 on 60 votes is *genuinely ambiguous* — people looked and disagreed — while 2/2 on 4 votes
  may be obvious and merely undersampled; likewise `v = 1.0` on 3 votes is a weak
  consensus-extreme, not a strong one. The uncertainty-geometry test and the consensus-extreme
  split (`extreme_low`/`extreme_high`) both read the fraction as if it carried the same meaning
  at every depth, and it does not. Decide a vote-count floor **or** a weighting **local to those
  two**; do **not** impose it corpus-wide, which would delete the ambiguous middle that test
  exists to use. Bites hard at the frozen floor: 89.8% of t09 boxy's positives rest on ≤2 votes.
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
- [ ] (P1) Rotation/reflection augmentation pipeline (symmetry, augmentation-first). *(D10)*
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
- [ ] (P0) **The LR schedule is a measured cause of the rank fall; the recipe is not yet changed.**
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
- [ ] (P0) **Proposed schedule, awaiting the resolving run — `artifacts/h4_schedule_proposal.md`.
  Nothing merged into `configs/pretrain.yaml`.** The **reference recipe adapted**, each number by a
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

## Epic G — Baselines as controls `[baseline]` — *same probe ladder, all SDSS-trained*
- [ ] (P1) `[baseline]` **MAE** — reproduce the Wu & Walmsley recipe on our SDSS corpus. *(D12)*
- [ ] (P1) `[baseline]` **Contrastive (MoCo)** — same SDSS corpus; probe identically. *(D12 sub)*
- [ ] (P1) Cross-objective comparison table (rung per feature × objective).

## Epic H — Figures & eval `[P7]`
- [x] (P0) **FIG 1** — ladder / AUC bar chart (`eval.figures.figure_ladder`).
- [x] (P0) **FIG 2** — concept-direction cosine matrix (`figure_entanglement`).
- [x] (P0) **FIG 3** — uncertainty-geometry scatter (`figure_uncertainty`).
- [x] (P1) **FIG 4** — controls panel (`figure_controls`).
- [ ] (P1) Label-efficiency curve (SSL-pretrained vs supervised-from-scratch).
- [ ] (P1) v1-comparable evaluation, and the v1-vs-v2 comparison as a first-class deliverable.

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

## Deferred — Paper 2 `[P2-paper]`
- [ ] Multi-survey corpus (SDSS + DESI Legacy → space-based); homogenisation (degrade-down first).
- [ ] Survey-leakage probe + single-vs-multi merge experiment.
- [ ] Fine-tuning comparison; confidence-aware read-outs.
- [ ] Deferred controls: matched evaluation at scale, cross-split robustness.
- [ ] E(2)-equivariant ViT ablation; GalaxyMNIST external comparison.
- [ ] Parking lot: SAE on frozen JEPA embeddings; Spectra/Jacobian mechanistic angle.
