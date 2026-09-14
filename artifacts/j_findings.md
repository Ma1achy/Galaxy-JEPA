# Brief J — the medium pretraining run

**Status: J0–J5 complete.** Nothing here is a result. Every artefact carries `smoke` in
`escape_hatches_used`, `effect_floor_freeze` is still `None`, and no rung verdict was computed.

**The four things this run found that it was not looking for:**

1. `run_harness`'s post-train probe was still building the 4 GB metadata table — the third call
   site Brief I's sidecar fix missed, and the one that pays last. Fixed.
2. The run's own loss decomposition was never written to disk. Fixed; `traces.json` now exists.
3. **Training 16.7x longer made the frozen representation measurably worse** — 0.9470 at step
   3,000 against 0.9278 at step 50,000, on one verified-identical trajectory, intervals separated.
4. **The probing ladder cannot currently run.** The sky-noise control (3C-5) sits at 0.84 and
   enters the existence bar, so every feature fails existence on this encoder — including
   featured-ness. The representation encodes magnitude (0.87), size (0.85) and SNR (0.84) more
   strongly than any morphology feature probed.

---

## J0 — the SSD was already the SSD

Malachy asked for the SSD rather than the main disk. Going to move the 416 GB cache, I found **it
is already there**: `runs` is a symlink to `/Volumes/X10 Pro/galaxy-jepa/runs`, made 5 September,
exactly as `.gitignore:31` documents ("the corpora, runs and pull staging live on an external
SSD"). `artifacts/f0_preconditions.py:25` has hardcoded that path since Brief F.

**My pre-flight was wrong, and the way it was wrong is worth recording.** `ls -la runs/` and
`du -sh runs/full/cache` both *follow* a symlink, so both measured the SSD while I reported them
as the internal disk — and from that I concluded a 416 GB relocation was needed and a rsync was
launched that would have copied the SSD onto itself. It failed instantly on a flag macOS's rsync
does not support (`--info=progress2`) and wrote nothing; the four cache files carry their original
timestamps. `ls -la` on the *parent* is what distinguishes a link from a directory, and it is what
I should have run.

| | |
|---|---|
| `runs/` — all outputs, incl. the 415,757,500,416 B `stamps.f16` | **SSD**, 2,520 GiB free |
| `data/pretrain`, `data/probe` — 1.2 TiB raw FITS | **SSD** |
| internal container | 34.3 GB unallocated |

So there is **no disk constraint on this run**. Checkpoints (35 retained × ~381 MB ≈ 13 GiB) land
beside the cache with three orders of magnitude of headroom.

The internal disk's 93% is not this project's: ~183 GB home directory, 29 GB `/Applications`, 20 GB
`/private/tmp` (19 GB of it one unrelated work tree), 12 GB `/opt`. It matters only because macOS
**swap** shares the container — 15.0 GB consumed against 34.3 GB unallocated, on an 18 GB machine
running a 9.7 h job. Adequate, not generous. The 19 GB in `/private/tmp` is the obvious reclaim if
headroom is wanted; not deleted, not mine to remove.

**One finding, reported not fixed.** `index.json` carries `scalars_sha256` and
`probe_columns_sha256` but **no digest on `stamps.f16`**. The parity lock is the *directory name*
(`pipeline_hash`), so a silently corrupted byte in 416 GB would never be caught by anything. A
one-off `shasum` would record one; whether it enters the index is a separate decision.

---

## J1 — pre-flight, as checked facts

`artifacts/j1_preflight.py` asserts all of these and raises on the first failure. It sits on top of
`f0_preconditions.check()`, which already gates the four it owns, rather than restating them.

| fact | value |
|---|---|
| `code_sha` | on `sigreg-ablation`; `main` is **45 commits behind** and carries neither D17 nor D18 |
| `config_hash`, config file | `f561d7f5039f23d8` → `b5acc6779df49070` (D18) → `538bf997880a8767` (D17) |
| `config_hash`, **stamped** | `v2:bb9945b617b53e3b` on `mps` |
| normalisation | `75100066b3e0…`, q=4.0, n=826,141, trim 827; refit path refused |
| cache | 1,057,326 stamps = 826,968 pretrain + 230,358 probe, dense; sidecar 81 columns, digest verified |
| vote floor | frozen at 1, sweep [1, 5, 11, 21, 37] |
| effect floor | **open** — `headline=True` refused at load |
| halt conditions | hard rank < 2.0 after step 100 ×3; std floor; non-finite |
| soft rank floor | 2.5, **inert** (`sigreg_lambda=0.05`) — silence confirmed, not assumed |
| out_dir | `/Volumes/X10 Pro/galaxy-jepa/runs/full`, 2,520 GiB free |

**On not merging.** `main` does not carry D17 or D18. I did not merge: `RunStamp.code_sha` pins the
actual commit either way, which is the provenance guarantee, and a merge is wider than this brief.
Available on Malachy's word.

**Two hashes, not one.** D18 recorded `b5acc6779df49070`, which is the **config-file** hash with
`runtime.device` unresolved. `_make_stamp` resolves the backend *before* hashing (MPS/CPU/CUDA
differ numerically and must hash apart), so what an artefact actually carries is
`v2:bb9945b617b53e3b`. I had been treating the two as one number. The pre-flight prints both.

### The four defects

**(a) `keep=3` was still the production default.** `run_harness` built `TrainCheckpointer` with no
`keep`. That is the hole that ate Brief I's arms mid-measurement, and J1 names this run's
checkpoints as the effect floor's input. Now derived from the schedule —
`steps // checkpoint_every + 2` = 35, against 33 scheduled — the form `i2_sigreg_run.py` already
used. Deliberately not a config field: retention cannot change a number the run produces, and
`NON_DETERMINING` is a top-level-key deny-list, so a nested knob would move `config_hash` for pure
housekeeping. **Proved** the new test fails under `keep=3` (only steps 3–5 survive of 5) rather
than asserting that it would.

**(b) `smoke` was False.** Now `true`, with the consequence written where it will be read: being a
determining field it moves `config_hash`, and `TrainCheckpointer` records `config_hash` in every
payload so a resume refuses across the change. **These weights can never be promoted to a headline
run** — the headline costs its own full budget from scratch. That is the honest price of J's own
framing, and it should be known before launching rather than discovered after.

**(c) A fourth reproducibility hole, exactly where J1 predicted one would be.** J said: *"Three
reproducibility holes have already been found… If a fourth exists, this run is where it will
show."* It did. `run_harness`'s own post-train probe still built
`rows_by_id(DirectorySource(probe_dir).rows)` — 1.49 GB that `LabelProvider` copies to 2.99 GB,
the table Brief G2 fled and Brief I's sidecar replaced. Brief I routed `evaluate_probe` and
`probe_frozen_checkpoint` through the sidecar and **missed this third call site**, so the one path
that pays for the failure last — after 9.7 h of training has banked its checkpoint — was the one
still carrying it. Now reads the sidecar like the other two.

The split is provably unaffected, which is why this changes no number: `assign_three_way` hashes
each objID independently and returns frozensets, so it is order-blind, and the sidecar's
finite-label ids are **exactly** the probe corpus (measured: 230,358 finite of 1,057,326 baked —
the other 826,968 are pretrain stamps carrying no vote columns). Pinned by making the whole-table
path raise for the whole run, and **proved** the patch is live on that path rather than merely
unreached: hide the sidecar and the same run does hit it.

**(d) The trajectory was not an artefact.** `RunReport` keeps `final_loss`;
`_safe_collapse_plot` renders the collapse trace as *pixels*; and `prediction_losses` /
`sigreg_losses` are deliberately **not** checkpointed (diagnostics, not training state). So the
loss decomposition of a multi-hour run existed only in the returned object and died with the
process — and J3 asks for exactly that decomposition as a trajectory. `run_harness` now writes
`traces.json`: total, prediction and SIGReg losses index-aligned per step, plus the numeric
collapse trace. H5's lesson is that endpoints lie (`std_final` called two arms equivalent where
the peak separated them 2.87×), so the trajectory is written down.

---

## J2 — budget: 50,000 steps, ≈ 9.7 h

Measured on the production path with SIGReg on, immediately before launch
(`artifacts/f2_model_smoke.py throughput`, 300 steps after 20 warmup):

| | |
|---|---|
| throughput | **1.4279 steps/s** (45.7 stamps/s, 17.97 MB/s) |
| bottleneck | **compute-bound** — 13.9% data wait, 86.1% compute |
| host-side masking | 5.5% of compute (`mask.sample` 9.85 s of 180.8 s) |
| MPS driver peak | 7.00 GB; process RSS peak 2.62 GB |
| loss over 300 steps | 2.88 → 0.66, all finite |
| erank / cos | 22.1 → 18.0 / +0.93 → +0.54 |

**50,000 / 1.4279 = 35,016 s = 9.73 h.** This is 3.3% below Brief I's 1.4765 steps/s, and the gap
is the measurement harness rather than the recipe: the f2 loop synchronises on `loss.item()` every
step, instruments the masker with timers, and monitors every 25 steps instead of 100. So 9.7 h is
the conservative end.

**Samples-seen, which is the stronger argument than epochs.** 50,000 × 32 = **1,600,000 samples =
1.93 epochs** over 826,968 stamps. Against the pilot's 192 k (AUC 0.905) and the 3,000-step arms'
96 k (0.9470). G4 stands: I-JEPA's epoch counts do not transfer — the paper never pretrains a
ViT-S with I-JEPA and its batch is 64× larger.

**The budget is not a free parameter.** `lr_final`'s cosine and the `ema_start → ema_end` ramp are
both defined over `cfg.steps`, so moving 50,000 changes the recipe D17 and D18 were adopted with —
H5 and I2 both held `steps=50000` fixed for exactly this reason while running 3,000. No extension
will be requested mid-run, and the loss looking interesting is not a reason.

---

## J3 — the trajectory

**Completed end to end. 50,000/50,000 steps, no halt, no interruption, no watchdog trip.**

| | |
|---|---|
| total wall | **11.09 h** = 1.20 h setup + 9.40 h training + 0.47 h probe |
| training throughput | **1.4752 steps/s** mean over 9.40 h |
| checkpoints | 34 entries, steps 1,500–50,000, **none pruned** (`keep=35`) |
| resume gaps | **zero** — 0 NaN pads across all three 50,000-long loss traces |
| stamped | `v2:bb9945b617b53e3b…`, `escape_hatches_used: ["smoke"]`, seed 0, mps |
| headline | frozen-probe AUC **0.9324 [0.9283, 0.9366]**, n_test 21,974 |

**Setup cost 72 minutes, and it was not in the budget.** Step 0 began at 22:39 against a 21:27
launch. `_prepare` spends it on `DirectorySource` over both corpora, `resolve_corpora`, two
`bake_cache` scans and `write_probe_columns`. Both bakes logged `0 new, 1057326 total` — **no
re-bake**, so the parity lock held across the `smoke` hash change exactly as designed
(`pipeline_hash` covers only the stretch and the frozen normalisation). The 9.4 h estimate was
for *training* and was right; the total was the part I had not measured.

**The soft floor's silence was confirmed, not assumed.** Logged verbatim at start-up: *"SIGReg is
on (lambda=0.05), so the soft rank floor of 2.50 is not applied: effective rank is
constraint-satisfied under it and the criterion could never fire (D18). The hard floor (2.00) and
the std floor still apply."*

### Trajectory, not endpoints

| step | loss | std | erank | cos |
|---|---|---|---|---|
| 0 | 2.9861 | 0.303 | 21.9 | +0.926 |
| 2,000 | 0.4066 | 1.008 | 28.6 | +0.154 |
| 6,000 | 0.4747 | 0.962 | 44.4 | +0.085 |
| 14,000 | 0.4725 | 0.986 | 51.9 | +0.053 |
| 26,000 | 0.4801 | 0.939 | 55.4 | +0.032 |
| 38,000 | 0.4441 | 0.895 | 57.3 | +0.028 |
| 50,000 | 0.4460 | 0.885 | 57.6 | +0.025 |

Peak **and** endpoint, because H5 taught that `std_final` alone called two arms equivalent where
the peak separated them 2.87×:

* **effective rank** — min 16.43 @ 700, peak **57.61 @ 49,800**, endpoint 57.61. Monotone after
  step 700; peak *is* the endpoint. Against the 3,000-step SIGReg arm's 34.2.
* **std** — min 0.303 @ 0, peak **1.023 @ 3,100**, endpoint **0.885**. It overshoots the N(0,1)
  target early and settles 13% below it. Here the peak/endpoint gap is 1.16×, not H5's 2.87×, so
  the two readings agree — but only because they were both taken.
* **mean cosine** — peak +0.926 @ 0, min +0.0225 @ 38,600, endpoint **+0.0250**.

### The loss decomposition — what SIGReg bought and what it paid

| term | first 2k | minimum | last 2k |
|---|---|---|---|
| total | 0.5210 | **0.3865 @ 2,000** | 0.4474 |
| prediction | 0.3844 | **0.3244 @ 2,000** | 0.4138 |
| SIGReg | 3.1159 | **1.0863 @ 48,000** | 1.0863 |

Final 500 steps: `0.4479 = 0.95 × 0.3935 + 0.05 × 1.0876`.

**The SIGReg penalty reached 1.086 against a measured isotropic floor of ~1.0** (1.065 at n=32 in
the I1 tests). The constraint is essentially **satisfied** — the pooled penultimate embedding is
as close to an isotropic Gaussian as a batch of 32 can resolve. It bought that by **giving back
prediction accuracy**: the prediction term bottomed at 0.3244 around step 2,000 and rose 28% to
0.4138 by 50,000, while the penalty fell 3.1 → 1.09.

This is I4's *isotropic-but-empty* guard doing its work. Effective rank 57.6 and cosine 0.025 are
**constraint satisfaction, not evidence** — they are what the objective was told to produce. They
must be read alongside AUC, never instead of it, and the AUC is below.

### Thermal behaviour — unmeasured territory, now measured

Read as sustained-throughput drift from the monitor timestamps, which is what actually costs
wall-clock. Absolute die temperature needs `sudo powermetrics` and is **not** measured.

Fourteen ~40-minute windows across 9.4 h: min 1.4534, max 1.4925, **mean 1.4752**, total spread
**2.7%**. First window 1.4534, last 1.4811 — the run ended **1.9% faster** than it began. **No
thermal throttling, no data-wait drift, no page-cache degradation** against a 416 GB working set
on USB. F only ever ran four minutes; nine hours look the same as four minutes.

The mean lands on Brief I's 1.4765 steps/s. J2's pre-flight calibration of 1.4279 was 3.3%
conservative, exactly as attributed at the time — the f2 loop's per-step `loss.item()` sync, its
masker timers and its 4× monitor cadence, not the recipe.

### Memory

The watchdog (12 GB tree RSS, 30 s poll) **never fired**, so the tree stayed under 12 GB for
11 hours. A spot reading at 1 h 40 m was 842 MB for the training process. Peak was **not sampled
continuously** — that is a gap in this run's instrumentation, stated rather than estimated. System
memory ran near-full throughout with swap in use, which is the page cache streaming the working
set and is the kernel doing its job, not a leak (`f_proc_watchdog.sh` documents exactly this).
One of my own polling loops was reaped for the pressure; the run was untouched.

### One provenance blemish, mine

`stamp.json` records **`code_dirty: true`** against `code_sha 42843a6`. The J1 pre-flight at 21:27
saw a clean tree; I then corrected a throughput constant in `artifacts/j1_preflight.py` at ~21:30,
and `_make_stamp` runs *after* `_prepare` — at 22:39, by which time the tree was dirty. The edit
was a printed estimate in a pre-flight script and cannot have touched a number, but that is an
argument, not a guarantee, and the artefact correctly refuses to make the guarantee.

**The lesson is the general one: do not touch the tree while a run is initialising.** A 72-minute
setup window is a 72-minute window in which the stamp has not yet been taken.

## J4 — signs of life

### The comparison is exact, not merely matched

Before reading any number: **this run is Brief I's `sigreg_050` arm continued.** I2 ran the
production `train_jepa` with `steps=50000` in the config and stopped at 3,000, so its arm is the
first 3,000 steps of exactly this trajectory. Checked rather than assumed:

* all three per-step loss traces — total, prediction and SIGReg — agree over all 3,000 steps to
  **5.0e-7**, which is the half-ulp of I2's 6-decimal storage. `loss_deepest` 0.29385602 at step
  **1559** in both; `loss_last50_mean` 0.3863165 in both; `sigreg_first` 35.240631 in both.
* every collapse reading is **identical at every shared step** — step 0 erank 21.9437 / std 0.3030
  / cos 0.9259, step 1,000 17.9385 / 0.8589 / 0.3798, step 2,900 33.9038 / 0.9988 / 0.1405. Step 0
  matching means the **monitor batch is the same too**, since the weights there are identical by
  construction.

So the AUC comparison below is **one trajectory at two stopping points**, not two runs that
resemble each other.

> A correction of my own, recorded because it is the second time: I first read I2's `erank_final`
> (34.2324) against this run's step-3000 value (34.5603) and took it for divergence. I2's trace
> ends at step **2975** — it monitored every 25. That is the identical like-for-unlike error Brief
> I already caught once with H5's 0.3168. Comparing summary scalars across drivers is where it
> keeps happening; comparing traces at matched steps is what settles it.

### Featured-ness, like-for-like

`artifacts/h5_probe_lean.py` **untouched**, `--arm full --runs runs`: same deterministic 40,000
stride, same 25,305 consensus train, same 21,974 consensus test galaxies as every arm below.

| arm | steps | consensus | all held-out | ambiguous middle |
|---|---|---|---|---|
| pilot | 6,000 (10k corpus) | 0.905 | — | — |
| baseline (pre-D17) | 3,000 | 0.9043 [0.8988, 0.9097] | 0.8084 | 0.6358 |
| d17 (λ=0) | 3,000 | 0.9358 [0.9315, 0.9402] | 0.8420 | 0.6624 |
| sigreg_050 | 3,000 | 0.9470 [0.9435, 0.9506] | 0.8558 | 0.6734 |
| sigreg_006 | 3,000 | 0.9471 [0.9433, 0.9508] | 0.8573 | 0.6760 |
| **this run** | **50,000** | **0.9278 [0.9234, 0.9320]** | **0.8336** | **0.6582** |

**Training 16.7× longer made the frozen representation measurably worse.** 0.9320 < 0.9435: the
intervals are cleanly separated, and all three framings move together. It ends below even the
λ=0 arm at 3,000 steps (0.9358, lower CI 0.9315), and above the pre-D17 baseline.

The production harness's own probe agrees: 0.9324 [0.9283, 0.9366], fitting on 101,591 consensus
extremes rather than 25,305. Four times the probe training data buys +0.0046 and does not close
the gap, so the deficit is in the representation, not the probe.

**Not like-for-like with a headline, and stated as such.** This is stamped `smoke`, the effect
floor is open, no checkpoint-selection claim is made, and no rung verdict is computed. It is an
existence check that the pipeline produces a sensible number at scale — which it does — plus one
finding that was not the point of the run.

### What the trajectory already said

The loss decomposition put the **prediction term's minimum at ~step 2,000** (0.3244, rising 28% to
0.4138 by 50,000) while the SIGReg penalty fell 3.12 → 1.086 against a measured isotropic floor of
~1.0. The best AUC (step 3,000) sits beside the best prediction loss (step 2,000), and both are
early. The two readings are consistent, but **one trajectory is not a correlation** — Brief I's Q2
measured that question properly at n=6 per arm and found ρ = +0.657 / +0.086 / −0.086, every exact
permutation p ≥ 0.136. This is a single additional point in the same direction, and it is reported
as that and nothing more. **The 1C label-blind checkpoint rule is unaffected and stands.**

What this run does establish is that *"train to the end of the configured budget"* is not
automatically the right label-blind rule at this scale — the final checkpoint is measurably not
the best one on this trajectory. That is a question for a later brief, with its own controls; it
is **not** settled here, and nothing in this run licenses picking a checkpoint by AUC.

### The spread and the controls

Six Scheme-1 features (all binary there, so `GradedExistenceTestUndecided` cannot fire), on the
same 40,000-train / 34,829-test split as (A), assembled **below the ladder**: `extract_matrix` ×3
sources → `probe_auc_ci` per feature → `build_feature_controls`. No `run_ladder`, no
`existence_verdicts`, no rung verdict. Extraction 1,885 s for all three sources.

| feature | real AUC | shuf max | rand max | untrained | noise | selectivity | n_test (pos) |
|---|---|---|---|---|---|---|---|
| t01 featured-or-disk | **0.8365** [0.8316, 0.8414] | 0.5552 | 0.5094 | **0.7908** | 0.4948 | +0.3370 | 34,829 (25.9%) |
| t02 edge-on yes | 0.7320 [0.7248, 0.7390] | 0.5321 | 0.5107 | 0.6368 | 0.5059 | +0.2322 | 34,351 (16.0%) |
| t10 winding loose | 0.6098 [0.5997, 0.6196] | 0.5167 | 0.5118 | 0.5645 | 0.5055 | +0.1106 | 20,245 (16.9%) |
| t10 winding tight | 0.5740 [0.5662, 0.5815] | 0.5157 | 0.5092 | 0.5474 | 0.4963 | +0.0727 | 20,245 (45.0%) |
| t09 bulge boxy | 0.5534 [0.5371, 0.5711] | 0.5251 | 0.5226 | 0.5359 | 0.5086 | +0.0535 | 16,693 (7.2%) |
| t10 winding **medium** | **0.5161** [0.5084, 0.5240] | 0.5136 | 0.5089 | 0.5160 | 0.5036 | +0.0165 | 20,245 (36.7%) |

**The graded axis behaves like an axis.** Ordered tight → medium → loose, the AUCs run 0.5740 →
**0.5161** → 0.6098. The *middle* category is the minimum, which is the signature of a graded
quantity rather than three independent binaries: a middle class is bounded on both sides and so is
the hardest to isolate, while the ends separate. That was predicted from the structure of an
ordered question before the numbers were read, and it held.

**Three of the four chance-calibrated controls behave exactly.** Shuffled-label mean across
features **0.4999** against a construction chance of 0.5; noise-images 0.4948–0.5086;
random-embedding maxima 0.5089–0.5226. The battery is calibrated.

**The untrained encoder is the binding null in every single case** — never the shuffled, never the
random-embedding, never the noise. So the real question for every feature here is "did pretraining
beat random initialisation", and the margins are thin:

| feature | real | untrained | margin |
|---|---|---|---|
| t02 edge-on | 0.7320 | 0.6368 | **+0.0952** |
| t01 featured | 0.8365 | 0.7908 | +0.0457 |
| t10 loose | 0.6098 | 0.5645 | +0.0453 |
| t10 tight | 0.5740 | 0.5474 | +0.0267 |
| t09 boxy | 0.5534 | 0.5359 | +0.0176 |
| t10 medium | 0.5161 | 0.5160 | **+0.0000** |

A random-init ViT-S reaches 0.7908 on featured-ness. Most of that signal is available from
architecture and image statistics alone. Edge-on is where learning adds most in absolute terms,
despite its lower headline AUC.

### The controls do NOT behave — and it is structural

> J4: *"If the controls do not behave here, the ladder cannot be trusted later."* They do not.

**The sky-noise control sits at 0.8355–0.8416 on every feature, and it enters the existence bar.**
`nulls.five_null_samples` takes `max(noise_encoder, untrained_encoder, sky_noise)` as a constant
every draw must clear. At 0.84 that exceeds **every** real AUC measured here — including
featured-ness, whose 0.8365 falls below its own sky-noise control of 0.8373.

| feature | real | five-null max | verdict as wired |
|---|---|---|---|
| t01 featured-or-disk | 0.8365 | 0.8373 | **FAILS** |
| t02 edge-on | 0.7320 | 0.8381 | FAILS |
| t10 loose / tight / medium | 0.6098 / 0.5740 / 0.5161 | 0.8416 | FAILS |
| t09 boxy | 0.5534 | 0.8355 | FAILS |

**Run the designed ladder on this encoder and every feature returns R3/R4** — an all-fail
catalogue that reads like a scientific null and is nothing of the kind. That is precisely the
failure `nulls.assert_null_resolution` exists to prevent, arriving through a door it does not
watch.

**Not a wiring error in my driver.** `sky_noise_auc` and `nuisance_aucs["snr"]` come out
**bit-identical** for all six features (0.8373314431, 0.8381377539, …) — two independent code
paths, the same number. Control 3C-5 and the `snr` nuisance probe are literally the same
measurement computed twice, once as a null to beat and once as a diagnostic.

**Why it misbehaves.** Four of the five controls *break something* and ask whether the probe still
works — shuffle the labels, randomise the embeddings, noise the images, un-train the weights. Each
is calibrated to chance by construction. The fifth keeps everything real and substitutes a
**different label**, so its AUC is not a chance-level quantity at all: it measures how much
nuisance information the representation carries. Folding it into the null maximum encodes the
requirement *"a morphology feature is real only if it is more predictable than image quality"*,
which is not what existence means.

**The nuisance panel says the same thing louder.** Probing the real embeddings for each nuisance:

| | magnitude | size | snr | redshift | psf |
|---|---|---|---|---|---|
| t01 | **0.8733** | 0.8501 | 0.8373 | 0.7918 | 0.5813 |
| t09 | 0.8731 | 0.8463 | 0.8355 | 0.7776 | 0.5719 |

Apparent magnitude (0.87), size (0.85), signal-to-noise (0.84) and redshift (0.78) are **all more
predictable from the frozen embedding than any morphology feature probed**, featured-ness
included. Only PSF width (0.57) is near chance. The representation encodes observing conditions
strongly — which is not surprising physically, and is a serious confound for the whole probing
programme.

**Reported, not fixed.** Whether 3C-5 belongs in the null maximum is a design decision in
`docs/galaxy-jepa-spec.pdf` §3C, not something to change quietly inside a brief that was told not
to compute verdicts. It must be settled **before any ladder runs**.

## J5 — the effect floor

### The three quantities

**1. The null distribution**, across six features and 600 shuffled + 600 random-embedding draws:

| control | min | median | max |
|---|---|---|---|
| shuffled labels (max/feature) | 0.5136 | 0.5209 | **0.5552** |
| random embeddings (max/feature) | 0.5089 | 0.5101 | 0.5226 |
| noise images | 0.4948 | 0.5045 | 0.5086 |
| untrained encoder | 0.5160 | 0.5559 | **0.7908** |
| sky-noise label | 0.8355 | 0.8399 | 0.8416 |

The three chance-calibrated controls top out at **0.5552**. The untrained encoder is a different
animal, ranging to 0.7908 and binding on every feature. The sky-noise control is a different
animal again, as above.

**2. The real spread**: 0.5161 … 0.8365, median 0.5919.

**3. The gap**: per feature, against its own strongest chance-calibrated null, +0.0000 (t10
medium) to +0.0952 (t02 edge-on). Against the pooled chance ceiling of 0.5552, the weakest real
AUC is *below* it and the strongest is +0.2813 above.

### The proposal: do not freeze a floor from this run

The brief allows this reading explicitly — *"If the honest reading is that this run cannot support
a defensible floor, that is the finding, said as such."* It is that reading, for four reasons, in
descending order of force:

1. **The floor's input population is currently empty.** The effect floor partitions *clean from
   marginal among already-significant effects*; existence is upstream of it. With the sky-noise
   control in the null maximum, nothing passes existence on this encoder. A threshold that
   partitions an empty set cannot be calibrated, and fixing the partition after the control
   question is settled would mean setting it twice.
2. **This encoder is measurably not the best on its own trajectory.** Step 3,000 of this exact run
   scored 0.9470 on featured-ness; step 50,000 scored 0.9278, intervals separated. Freezing the
   catalogue's clean/marginal line against a representation known to be worse than one already in
   hand is the wrong anchor, and it is the sort of choice that is very hard to revisit later.
3. **An absolute AUC floor does not match what the evidence has structure in.** The binding null is
   the *untrained encoder*, and it varies per feature from 0.5160 to 0.7908 — a spread of 0.275,
   wider than the whole real spread's distance above chance. No single absolute AUC can encode
   "beat what architecture alone delivers". The measured structure argues for a **margin over the
   untrained-encoder control** rather than an absolute threshold; that is a change to the
   mechanism, which the spec records as settled, so it is raised rather than made.
4. **n = 6 is too few to locate a threshold in the spread.** The widest gap runs 0.6098 → 0.7320,
   midpoint 0.6709 — and that midpoint is a fact about which six features were chosen, not about
   the catalogue. The brief warns against exactly this: *"not a value that happens to include or
   exclude a feature of interest."*

### If a value must be set now

Recorded so the call is available rather than withheld. Each is measured, none is round:

| candidate | value | what it would mean |
|---|---|---|
| chance-calibrated null ceiling | **0.5552** | existence's job, not the floor's — too low |
| untrained-encoder ceiling | **0.7908** | "above anything architecture alone delivered here"; admits t01 only |
| widest-gap midpoint in the spread | 0.6709 | admits t01, t02; an artefact of this six |
| median of the real spread | 0.5919 | descriptive, not principled |
| the placeholder now in `probe.yaml` | 0.6500 | ungrounded |

Of these, **0.7908** is the one with a stated meaning that survives being questioned. It would
admit featured-ness alone of the six, which is a severe catalogue — and that severity is the
honest reflection of a representation whose nuisances outscore its morphology.

**Not frozen.** `effect_floor_freeze` remains `None`; `configs/probe.yaml` is unchanged;
`headline=True` is still refused at load. The value is Malachy's call and carries `frozen_by`.
