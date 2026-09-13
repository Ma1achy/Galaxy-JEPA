# Brief J — the medium pretraining run

**Status: J0–J2 complete, J3 running.** Nothing here is a result. Every artefact carries
`smoke` in `escape_hatches_used`, and `effect_floor_freeze` is still `None`.

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

*(running)*

## J4 — signs of life

*(pending)*

## J5 — the effect floor

*(pending)*
