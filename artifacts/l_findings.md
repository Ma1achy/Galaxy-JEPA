# Brief L — findings

Three phases. The reading rules for L1 and L2 are written **here, before either measurement
returned**, the way `artifacts/i_decision_rule.md` was committed before Brief I's arms ran and
K2's was written before its curve landed. `git log` is the proof.

---

## L0 — Prerequisites

**`EmbeddingMatrix.index` is cached.** K3-iv measured the defect: a plain `@property` rebuilding
a 74,829-entry dict on every read (5.6 ms), read *inside a comprehension's condition* by
`feature_ids`, so filtering 40,000 ids cost **225 s against 8.6 ms hoisted** — a factor of 26,000,
and ≈2.8 h of J4's 6.0 h.

Fixed on the property rather than at the call site. Only one site was actually broken
(`rows_for` already hoisted it), so caching the property is correct at every consumer including
ones not yet written; `functools.cached_property` writes straight into `__dict__` and so works on
the frozen dataclass, at ~6 MB retained per matrix. Three invariant tests pin it: cache identity
(which structurally forbids a rebuild), that the dataclass is still frozen, and a loose wall-clock
bound the quadratic form cannot meet.

**Pushed.** `origin/sigreg-ablation` now carries D19, the corrected 3C spec and Brief K's
findings. **The spec PDF is still stale** — no LaTeX toolchain in this environment; the `.tex` is
the corrected source.

---

## L1 — Is the information lost, or just less linearly accessible?

Effective rank rose 24.3 → 57.6 across the run while consensus AUC fell 0.9477 → 0.9278. Those may
be **one fact seen twice**: information spread isotropically across more directions is less
accessible to a fixed-capacity L2 probe without being *gone*. The distinction changes what the
whole ladder measures, because R1 means "clean **linear** direction".

### The reading rule, written before the numbers

| shape | reads as |
|---|---|
| MLP flat while linear declines | information intact; the problem is **linear accessibility** |
| both decline together | information genuinely **destroyed** |
| MLP declines *more* | something else is happening — say so, do not force a story |

Summary statistic per checkpoint: **the best MLP AUC at a width strictly below the selectivity
ceiling** — the ladder's own R3 convention, and the guardrail the spec insists on. An MLP that
holds up by memorising is not evidence of retained information, so `capacity_sweep` runs its
shuffled-label control at every width alongside the real one.

**Two guards, both pre-registered:**

- **Probe-adequacy gate.** At the peak checkpoint the best sub-ceiling MLP AUC must be **≥ the
  linear AUC**. `mlp_epochs: 200` is 200 *full-batch* Adam steps, a weak recipe against sklearn's
  lbfgs-to-convergence. If the MLP cannot match the linear probe where the representation is at
  its best, the instrument is too blunt to answer the question and the trajectory is
  **uninterpretable** — an instrument failure, not a negative result.
- **Ceiling sensitivity.** `selectivity_ceiling`'s predicate is FLAGGED in the spec as undecided.
  The curve is re-read with the ceiling one swept width lower and one higher; if the conclusion
  moves, it belongs to the flagged predicate and not to the data.

### Protocol

`l1a_cache_embeddings.py` extracts each checkpoint's frozen embeddings **once** and banks them
(74,829 × 384 float32, 115 MB each) to the SSD. K2 embedded the same stamps at ~10 min a
checkpoint, read three AUCs off each matrix and threw it away, so this brief pays that cost again
— **the lesson is that an extraction pass costing ten minutes must bank its output**, for the same
reason `keep=3` had to go. Banked, `l1b_mlp_ladder.py` is CPU-only and needs no encoder, which is
what lets it run while a training arm holds the GPU.

The split is K2's, imported rather than re-derived, with the same two assertions against J4(A)
(consensus train 25,305, held-out 34,829) fired before the first encode. Each banked matrix is
re-probed linearly and must reproduce K2's number to four decimals or the pass aborts — checked on
the first checkpoint at write time, and on all eight at read time.

Two features, each compared against **its own** linear probe under **one** convention, never
across two: **t01** under the H5/K2 `_extremes` consensus protocol, so its linear column lands
directly on K2's curve; **t02 edge-on** under the scheme's full labelled set, where its own linear
number (J4's 0.7320) comes from.

### Status — the extraction is done, the sweep is not

`l1a` banked all eight matrices: **882 MB, 8 x 74,829 x 384 float32, 6,167 s (1.71 h)**, on the
SSD at `runs/l1_embeddings/full`. Step 1,500 re-probed linearly to **0.9477**, reproducing K2
exactly, so the bank is faithful.

`l1b` was validated end to end on that first matrix before the full pass, and **both
pre-registered guards report cleanly**:

* **Probe-adequacy — passes.** At the peak checkpoint the best sub-ceiling MLP reaches **0.9558**
  against the linear probe's **0.9477**, +0.0081. The 200-full-batch-step recipe is strong enough
  to answer L1; had it come in under, the trajectory would have been an instrument failure rather
  than a result.
* **Ceiling — never fires.** The shuffled-label control sits at 0.49-0.53 at every width against a
  0.5475 threshold, so the whole swept range is valid and the sensitivity check has nothing to
  move. That is the bounded probe working as designed: 200 full-batch steps cannot memorise.

| width | t01 real | t01 control | t02 real | t02 control |
|---|---|---|---|---|
| 16 | 0.9398 | 0.5328 | 0.6937 | 0.4941 |
| 32 | 0.9449 | 0.5190 | 0.7098 | 0.4837 |
| 64 | 0.9506 | 0.5088 | 0.7285 | 0.4908 |
| 128 | 0.9535 | 0.5129 | 0.7365 | 0.4943 |
| 256 | 0.9550 | 0.5124 | 0.7440 | 0.4952 |
| 512 | **0.9558** | 0.4945 | **0.7515** | 0.4916 |

**A caveat the data forced, not one I planned.** The real curve rises monotonically and is still
climbing at width 512, the top of the sweep. The MLP is therefore **capacity-limited, not
ceiling-limited**, and its number is a *lower bound* on nonlinear accessibility rather than an
estimate of it. Now recorded per checkpoint (`capacity_limited`) and printed, because the read
changes if it holds at one end of the trajectory and not the other.

The full eight-checkpoint sweep is ~1.7 h of CPU and has not been run. **Parked** — see below.

### The trajectory — 8 checkpoints, both features, 1,121 s

`t01_consensus`, the headline, under K2's exact protocol (the linear column lands on K2's curve to
four decimals at every step, which is the cache-faithfulness check):

| step | linear | MLP (best sub-ceiling) | MLP − linear | best width | ceiling | capacity-limited |
|---|---|---|---|---|---|---|
| 1,500 | **0.9477** | **0.9558** | +0.0081 | 512 | none | **yes** |
| 3,000 | 0.9470 | 0.9575 | +0.0105 | 512 | none | **yes** |
| 6,000 | 0.9448 | 0.9555 | +0.0107 | 128 | none | no |
| 10,500 | 0.9412 | 0.9512 | +0.0100 | 128 | none | no |
| 18,000 | 0.9376 | 0.9462 | +0.0087 | 128 | none | no |
| 27,000 | 0.9295 | 0.9361 | +0.0067 | 128 | none | no |
| 37,500 | 0.9284 | 0.9342 | +0.0058 | 64 | none | no |
| 50,000 | 0.9278 | 0.9321 | +0.0042 | 64 | none | no |
| **net** | **−0.0199** | **−0.0238** | | | | |

`t02_edgeon_a04_yes`, the robustness feature, under its own scheme convention:

| step | linear | MLP | MLP − linear | best width |
|---|---|---|---|---|
| 1,500 | 0.7261 | 0.7515 | +0.0254 | 512 |
| 3,000 | 0.7558 | 0.8245 | +0.0687 | 512 |
| 6,000 | **0.7794** | **0.8465** | +0.0672 | 128 |
| 10,500 | 0.7687 | 0.8212 | +0.0526 | 128 |
| 18,000 | 0.7600 | 0.7985 | +0.0385 | 128 |
| 27,000 | 0.7369 | 0.7550 | +0.0181 | 128 |
| 37,500 | 0.7333 | 0.7461 | +0.0128 | 64 |
| 50,000 | 0.7320 | 0.7432 | +0.0112 | 64 |
| **net** | **+0.0059** | **−0.0083** | | |

### Both guards, reporting

**Probe-adequacy — passes.** At the peak the MLP beats the linear probe by +0.0081, so the
instrument is strong enough for the question. Had it come in under, the trajectory would have been
an instrument failure, not a result.

**Ceiling — never fires, at any checkpoint, for either feature.** Shuffled-label controls span
0.4646–0.5333 against thresholds of 0.5158–0.5496. The whole swept range is valid, so the
**ceiling-sensitivity check has nothing to move and the FLAGGED `selectivity_ceiling` predicate is
not load-bearing on this result.** No MLP anywhere on this trajectory holds up by memorising:
200 full-batch steps cannot, and the control measures that it did not.

### The reading — branch three fired, the one with no story attached

Pre-registered: *MLP flat while linear declines* → linear accessibility; *both decline* →
information destroyed; *MLP declines MORE* → something else, say so rather than force a story.

**The MLP declines more.** On t01, linear loses 0.0199 and the MLP loses **0.0238**. The result is
not that the information survives in a form a stronger probe can reach. The nonlinearly accessible
part is going **faster** than the linearly accessible part.

The cleanest statement of it is the gap itself — the nonlinear headroom, what the MLP buys over
the linear probe, which collapses monotonically on both features:

```
t01   +0.0081  +0.0105  +0.0107  +0.0100  +0.0087  +0.0067  +0.0058  +0.0042
t02   +0.0254  +0.0687  +0.0672  +0.0526  +0.0385  +0.0181  +0.0128  +0.0112
```

By step 50,000 the MLP is worth about half what it was worth early on t01, and a sixth on t02. The
best width falls with it, **512 → 128 → 64** on both features: early representations reward
capacity, late ones have less structure left to exploit.

**This kills the isotropisation-hides-it hypothesis, which was the reason L1 was asked.** Effective
rank rising 24.3 → 57.6 while AUC fell could have been one fact seen twice — the same information
spread across more directions, harder for a fixed-capacity L2 probe to reach but not gone. If that
were what happened, the MLP would have held while the linear probe fell. It did not. Rank growth
here is not re-packaging; information is leaving.

**The capacity limitation cuts the safe way.** The two earliest checkpoints are capacity-limited —
the sweep is still climbing at width 512, so those MLP numbers are *lower bounds*. The true early
values can only be higher, which can only make the decline **larger** than measured. The finding is
conservative in the direction that matters, and widening the sweep would strengthen it, not
threaten it.

**A warning the second feature paid for.** t02's linear probe ends **higher** than it started
(0.7261 → 0.7320, +0.0059) while its MLP ends **lower** (−0.0083). Read by the linear probe alone,
t02 looks mildly improved by 50,000 steps. Read with the MLP, it lost a third of its nonlinear
headroom. **A single-probe reading of this trajectory is not merely incomplete, it can carry the
wrong sign.** t02 also peaks at 6,000 rather than falling from the first checkpoint, so K2's
"monotonic from the earliest checkpoint" is a t01 fact and not a property of the run.

**What this does not say.** It does not name a cause — that is L2's job, and L1 was run first
precisely so the cause is not inferred from the same trajectory that raised the question. It does
not license checkpoint selection: 1C stands. And n=8 on one trajectory with one seed cannot bound
the peak, which is at or before step 1,500 for t01.

---

## L2 — The decisive arm: is it SIGReg at all?

λ=0 has never been run past 3,000 steps. "SIGReg is the suspect" rests on K2 eliminating the
decay's temporal signature, **not** on any evidence that λ=0 behaves differently. That gap closes
before a fix is chosen.

### The reading rule, written before the run

| λ=0 behaviour vs λ=0.05 | branch | reads as |
|---|---|---|
| rises or stays flat while λ=0.05 declines | **1** | **SIGReg confirmed** as the cause |
| declines with the same shape | **2** | **NOT SIGReg.** Something more fundamental — batch 32 against the reference's 2048, the EMA schedule at small batch, or the recipe generally. A different investigation |
| declines more slowly | **3** | SIGReg **accelerates a pre-existing problem**. Both need fixing |

Decision statistic, fixed now: **ΔAUC from each arm's own step-1,500 value to its step-10,500
value.** λ=0.05's is **−0.0065** (0.9477 → 0.9412). "Same shape" means the λ=0 interval overlaps
that; "flat" means its interval includes zero and excludes −0.0065. **The branch is named before
anything is proposed.**

n=4 per arm, one seed, one trajectory. A **shape test, not a replication** — it can say whether
λ=0 turns, not by how much.

### Protocol

`i2_sigreg_run.py` reused, with `--steps` / `--checkpoint-every` / `--runs` added and defaults
unchanged so the recorded I2 invocation still reproduces. Everything the comparison rests on is
held: one `seed_init`, one `ResumableShuffle` seed, the same masks, the same monitor batch.
**Only `sigreg_lambda` varies.**

`cfg.objective.steps` stays **50,000** and the run stops via `train_jepa`'s `stop_after`, which
"ends *this invocation*, leaving `config.steps` — and so the LR and EMA schedules — untouched".
Load-bearing: compressing the cosine into 10,500 steps would change the recipe D17 and D18 were
adopted with and destroy the comparison. H5 and I2 both did exactly this. Confirmed from `plan`:
lr@10,500 is **91.4% of peak**, matching K2's step-10,500 exactly.

**10,500, not 10,000.** `checkpoint_every: 1500` then lands checkpoints on 1,500 / 3,000 / 4,500 /
6,000 / 7,500 / 9,000 / 10,500, and 10,500 is *exactly* K2's fourth point — a four-point
like-for-like instead of comparing 9,000 against 10,500, for 5% more wall-clock.

**A provenance claim, wrong three times, and what finally closed it.** This cost an hour and the
sequence is the finding, so it is written out rather than tidied away.

*First*, the plan asserted the λ=0 config minus `smoke` must hash to D17's `538bf997880a8767`. It
cannot: D18's config commit added **four** keys — `sigreg_lambda` plus the inert `sigreg_slices`,
`sigreg_quad_points`, `sigreg_domain` — and all four are hashed whether or not the penalty runs.

*Second*, I corrected that to "the current config with `smoke: false` reproduces D18's
`b5acc6779df49070`", which returned `a456d91f…` instead and briefly looked like drift under D18. It
was not. There are **two dumps**, differing in exactly the field the invariant requires:

| dump | hash | what it is |
|---|---|---|
| `HarnessConfig(**yaml)` from the file, `smoke: false` | `b5acc6779df49070` | the **config-file** hash — what D18 recorded |
| the same, as written (`smoke: true`) | `f561d7f5039f23d8` | what Brief J's artefacts carry |
| the **resolved** config from `check()` | `bb9945b617b53e3b` | what a *run* stamps — matches `runs/full/stamp.json` |

`runtime.device` is `None` in the file and `'mps'` once resolved, and is hashed because backends
differ numerically. Both numbers are correct; quoting one without saying which dump produced it is
the error.

*Third, and the one worth keeping.* I then built a guard asserting the λ=0 arm must reproduce what
I2's `d17` arm stamped, `v2:b536709fababa819…`. **It fired and stopped the run before it spent two
hours** — correctly, by its own logic, and on a premise that was wrong. That hash reconstructs only
at `monitor_every: 100, checkpoint_every: 1500`, while the constants committed *before* that run
read 25 and 500. I concluded the stamp misdescribed its own run, from a dirty tree
(`code_dirty: true`), and that it was therefore untrustworthy.

**That conclusion was wrong, and `runs/i2/d17/config.json` says so.** The run recorded its own
config: `monitor_every: 100, checkpoint_every: 1500`, matching its stamp exactly. The stamp was
right. What was not the record was the *committed source* — the tree was dirty in precisely those
two constants, so git does not say what ran, and the config the run wrote beside its stamp does.

**The rule that falls out, and it generalises past this run:** check a run against what it
RECORDED, never against what the current source would produce. `code_dirty: true` does not mean
the artefacts are unrecoverable; it means the source is not the record and the run's own
`config.json` is. A hash alone could not have settled this. The hash plus the config beside it
could.

### The arm actually launched, against the reference that governs the comparison

The reference is not I2's `d17` — it is **`runs/full`**, the λ=0.05 run K2's trajectory was probed
from, because the shape test compares this arm against *those* numbers:

| field | `runs/full` (λ=0.05) | L2 (λ=0) | |
|---|---|---|---|
| `sigreg_lambda` | 0.05 | **0.0** | the variable under test |
| `checkpoint_every` | 1500 | 1500 | matches |
| `monitor_every` | 100 | 25 | **observation-only** |

`monitor_every` gates a `torch.no_grad()` forward of the monitor batch and the MPS pool release
(`objectives/jepa.py:405,429`). The forward draws no randomness — the models carry **no dropout and
no stochastic depth**, verified, so it is a deterministic read of the weights — and the pool release
is documented inert against a step-for-step loss comparison. It changes the trace's sampling rate
and nothing else, so **the arm was left running rather than restarted for 35 minutes**, and it buys
4x the trace resolution for watching the rank floor. The guard now licenses both cadence fields by
name with that reasoning attached, and refuses anything else.

**Halt conditions, stated before starting.** Under λ=0 the **soft rank floor becomes active** —
`train_jepa` derives `soft_rank_floor = sigreg_lambda <= 0.0` — at 2.5, after a grace of
`0.10 × 50,000 = 5,000` steps. **This is the first run in the project that crosses that
boundary**; D17's 3,000-step arm never reached it. Its effective rank at 3,000 was 11.77 so it
should sit far clear, but the floor's silence must be *confirmed from the log*, not assumed. The
hard floor (2.0) and the std floor apply throughout.

### Status — launched, stopped, nothing lost

The arm ran 600 of 10,500 steps and was stopped. **No checkpoint was written** (the first is due
at 1,500) and the trajectory is a pure function of `(seed, data order, schedule)`, so a relaunch
is identical rather than merely equivalent. Nothing is lost.

**Why it was stopped: the machine was oversubscribed before it started.** Measured after every
one of my own processes had exited:

| | |
|---|---|
| physical RAM | 19.33 GB |
| free | **0.08 GB** |
| held in the compressor | **39.72 GB** of logical memory, inside 7.05 GB of RAM |
| swap in use | **12.8 GB** |

The arm managed **0.045 steps/s** at the end — 25 steps in 555 s — against the **1.479 steps/s**
I2 measured for this identical arm. That is a 65 h projection for a 2 h run, and the process's RSS
had collapsed to 11 MB with CPU at 2.9%: the macOS memory-compressor signature, paged out and
stalled rather than working. Killing it moved swap by 190 MB, which is the proof that **it was
never the consumer** — the load is a Docker VM at 150-390% CPU, the Claude desktop app, several
Claude Code sessions and a `codex` process, none of them mine to stop.

**The result would have been valid either way.** The AUC trajectory depends on seed, data order
and schedule, not on wall-clock, so contention costs time and nothing else. The one thing it does
void is any throughput comparison against I2's 1.479 steps/s, and none is made.

### Two contention findings worth keeping

**1. A CPU sweep beside a training loop costs 2.8x; beside an extraction pass it costs 2%.** I
launched `l1b` alongside the arm on the reasoning that CPU work and MPS work coexist — which the
`l1a` smoke test had appeared to confirm, costing that pass 35 s out of 881. It does not
generalise. An *extraction* pass is MPS + bulk sequential disk; a *training loop* is MPS plus a
CPU dataloader feeding it every step, and a CPU-bound sweep starves the feeder. Measured 1.9 s/step
against 0.68 s/step for the same arm alone. **The plan's claim that the MLP sweep could ride
inside the training run was mine and it does not hold.** L1b runs after L2, or during L2's probe
phase where the profile is extraction-against-extraction again.

**2. Repeated extraction gets faster as the page cache warms.** Across `l1a`'s eight passes over
the same 29.4 GB of stamps: 881 -> 867 -> 785 -> 696 -> 655 -> 659 -> 682 s. The first pass is the
expensive one, and a multi-checkpoint probe should be costed on the later figure rather than the
first. (The 916 s second pass is the `l1b` smoke test, above.)

### The arm ran clean

10,500 steps, **no halt**, 1.515 steps/s, 6,931 s. All seven checkpoints written. The
pre-registered halt condition is closed: the soft rank floor became active at step 5,000 (grace
`0.10 x 50,000`) at a threshold of 2.5, and effective rank was **rising away from it** — 8.1 at
1,500 to 18.6 at 9,000 — so it never fired. `halted: false`.

| step | pred loss | std | erank | cos |
|---|---|---|---|---|
| 1,500 | 0.1871 | 3.14 | 8.1 | 0.409 |
| 3,000 | 0.2487 | 3.90 | 12.0 | 0.326 |
| 4,500 | 0.3188 | 4.50 | 15.0 | 0.307 |
| 6,000 | 0.4254 | 4.50 | 16.4 | 0.300 |
| 7,500 | 0.2634 | 4.59 | 17.6 | 0.285 |
| 9,000 | 0.3705 | 4.58 | 18.6 | 0.324 |

**Continuity, demonstrated rather than argued.** At step 3,000 this arm probes to **0.9358**, which
is *exactly* what D18 recorded for its λ=0 arm — `0.9358 [0.9315, 0.9402]` — to four decimals and
on both interval ends. After all the trouble the config hashes gave, the check that settled it was
a measurement: this run reproduces I2's `d17` where the two overlap, so it is that arm carried
further.

### The shape test — `t01_consensus`, the decision statistic

| step | λ=0 | λ=0.05 | λ=0 − λ=0.05 |
|---|---|---|---|
| 1,500 | 0.9193 [0.9145, 0.9241] | **0.9477** [0.9440, 0.9515] | −0.0285 |
| 3,000 | 0.9358 [0.9315, 0.9402] | 0.9470 [0.9435, 0.9506] | −0.0112 |
| 6,000 | 0.9458 [0.9420, 0.9495] | 0.9448 [0.9410, 0.9485] | +0.0010 |
| 10,500 | **0.9554** [0.9520, 0.9589] | 0.9412 [0.9372, 0.9451] | **+0.0142** |
| **Δ 1,500→10,500** | **+0.0361** | **−0.0065** | |

And `t02_edgeon_a04_yes`, which was added so the reading would not rest on one feature:

| step | λ=0 | λ=0.05 | λ=0 − λ=0.05 |
|---|---|---|---|
| 1,500 | 0.7080 | 0.7261 | −0.0181 |
| 3,000 | 0.7422 | 0.7558 | −0.0135 |
| 6,000 | 0.7850 | 0.7794 | +0.0056 |
| 10,500 | **0.8042** [0.7979, 0.8108] | 0.7687 [0.7619, 0.7755] | **+0.0355** |
| **Δ** | **+0.0962** | +0.0426 | |

### The branch: **1 — SIGReg confirmed**

Pre-registered: *rises or stays flat while λ=0.05 declines* → **branch 1, SIGReg confirmed**;
*same shape* → branch 2, not SIGReg; *declines more slowly* → branch 3, accelerant.

**λ=0 rises monotonically on both features** — every checkpoint above the last, +0.0361 on t01 with
the step-1,500 and step-10,500 intervals not overlapping — **while λ=0.05 falls monotonically.**
The decay, the batch size, the EMA schedule and the recipe are identical across the two arms. The
only thing that differs is whether the penalty is computed. Branch 1, with nothing to qualify it.

λ=0 does not merely avoid the decline: by step 10,500 it reaches **0.9554**, higher than λ=0.05
ever gets *anywhere on its 50,000-step trajectory* (best 0.9477, at the earliest checkpoint probed).

### The crossover is the whole of D18's error

At 3,000 steps λ=0.05 genuinely leads, 0.9470 against 0.9358 — **D18's evidence was correct**. By
6,000 the arms are level (+0.0010, intervals overlapping). By 10,500 λ=0 leads by 0.0142 with
intervals cleanly apart. **The crossover lies between 3,000 and 6,000 — immediately past the
horizon D18 measured to.** D18 did not misread its data; it read data that stopped one regime short
of the one the decision would run in.

### A second difference between the arms, found by the guardrail crashing

On **3 of 4** λ=0 t01 checkpoints there is **no admissible MLP at all**: the shuffled-label control
clears the selectivity threshold at *every* width down to 16, so no capacity in the sweep is
selective. Across all **eight** λ=0.05 checkpoints, on both features, the ceiling never fired once.

The λ=0 representation is memorisable at every capacity offered, and the λ=0.05 one is not. Its
embeddings carry std ≈ 3.1–4.6 against λ=0.05's ≈ 0.9–1.0, and an unnormalised, larger-magnitude
embedding is easier to overfit at fixed weight decay. Two consequences, and the second is a
warning:

* **The MLP column is not comparable across arms** for t01, and no nonlinear reading is taken from
  those three checkpoints. L1's conclusion is unaffected — it was drawn *within* the λ=0.05 arm.
* The effect **weakens as λ=0 trains**: by 10,500 the ceiling no longer fires and probe-adequacy
  passes (+0.0006). Early λ=0 is the memorisable regime, not λ=0 as such.

This surfaced as a crash — `mlp_best_sub_ceiling` was `None` and the print assumed a number. The
driver now reports "no admissible MLP" as the result it is. **A guardrail firing is an outcome, not
an error**, and a driver that dies on its own guardrail cannot report the thing it exists to catch.

### Limits, stated with the result

One seed, one trajectory, four points per arm. This is a **shape test**: it says λ=0 turns upward
where λ=0.05 turns down, not by how much at any other horizon. And the horizon caveat cuts both
ways — **λ=0 is measured to 10,500 and the headline run is 50,000.** λ=0.05's decline was invisible
at 3,000; nothing here proves λ=0 has no turn of its own past 10,500, and its rank is still climbing
at the last reading. That is the subject of the adoption decision in L3, not a footnote to it.

---

## L3 — Fix it

Branch 1 fired, so the remedy follows from the measurement rather than from a preference.

### The change

**`configs/pretrain.yaml`: `sigreg_lambda: 0.05 → 0.0`.** This reverses D18. Recorded as **D21**,
with **D20** carrying the process lesson separately.

Three candidates were listed in the brief, in order of principle. Only one is supported by a
measurement at the required horizon:

| candidate | status |
|---|---|
| accumulate embeddings across steps to raise the statistic's effective sample | a hypothesis about *why* SIGReg hurts. Untested. **Adopting it would repeat D18's pattern** — choosing a recipe from reasoning rather than from a run long enough to test it |
| reduce λ | same objection, and I2 already measured an 8× change in λ as indistinguishable (0.9470 vs 0.9471), so the dose-response needed to justify it is absent |
| **disable SIGReg, accept D17** | **measured, at 10,500 steps, on both features, intervals apart.** Adopted |

The three inert SIGReg fields are **kept, not deleted**. They are read only inside
`if self.config.sigreg_lambda > 0.0`, so they cost nothing at λ=0, and keeping them means the
penalty can be switched back on for the estimator experiment without moving anything else in the
hash.

**The soft rank floor is active again** — `train_jepa` derives it from `sigreg_lambda <= 0.0` — and
the config now says so where it previously said the opposite. L2 is also the evidence that 2.5 is
not set too high: effective rank rose 8.1 → 18.6 *away* from it across 10,500 steps and it did not
fire. Before Brief L this floor had never been reachable by any run in the project, because none
had passed the 5,000-step grace.

### What moved, and what deliberately did not

338 tests, ruff, ruff format and mypy all green. The two pinned config hashes moved to
`7ecf5dce5a1f60ba` (as written, `smoke: true`) and `de87b8f9704b7e2d` (the recipe), with the
reasoning written into the tests rather than the numbers merely updated.

**D17's stripped anchor `538bf997880a8767` is unchanged**, and that is the load-bearing assertion:
stripping the four SIGReg keys removes `sigreg_lambda` whatever its value, so the anchor holds
across both the adoption and the reversal and proves **nothing outside the SIGReg block moved**. An
artefact carrying `f561d7f5…` was written under D18's recipe and one carrying `7ecf5dce…` under
D21's; the two eras cannot be confused by reading a stamp.

### The adoption is provisional, in writing

D21's evidence reaches **10,500 steps. A headline run is 50,000.** λ=0.05's decline was invisible at
3,000, so nothing here proves λ=0 has no turn of its own later, and its effective rank was still
climbing at the last reading. **Adopting λ=0 for a 50,000-step run on 10,500-step evidence is
D18's error at a longer lever arm** — so D20 and D21 both name the missing measurement rather than
leaving it in someone's memory, and TODO carries it as P0.

The brief's bar was "at least 10,000 steps, long enough to see the turn", and that bar is met:
λ=0.05's turn is visible by 6,000 and λ=0 shows none by 10,500. **The headline-horizon arm is a
further ~9 h of GPU and was not launched** — it is outside what this brief asked for, and it is the
user's call whether to spend it before the headline or to accept the risk knowingly.

### The next experiment, named but not adopted

I1 pre-registered the risk that the Epps–Pulley statistic is badly estimated from 32 samples in 384
dimensions. That remains the live mechanistic hypothesis, and L2 is consistent with it without
testing it. The experiment: **accumulate embeddings across steps** so the statistic sees an
effective sample far larger than the batch, without touching the batch — then validate at ≥10,000
steps against this λ=0 arm. LeJEPA's own result stands in its regime, batch 2048 and eight views;
nothing here is evidence against it.

### One engineering lesson, paid for in a crash

`l1b` died on the first λ=0 checkpoint because `mlp_best_sub_ceiling` came back `None` — the
selectivity ceiling firing at the smallest width — and the print assumed a number. **A guardrail
firing is an outcome, not an error.** A driver that dies on its own guardrail cannot report the
thing it exists to catch, and the finding here (λ=0 is memorisable at every capacity, early) would
have been lost as a stack trace. It now reports "no admissible MLP" as the result it is.

---

## Parked — how to resume

Nothing is half-finished. Every asset a resume needs is on disk and every driver is committed.

**Freed first.** All Galaxy-JEPA processes are stopped; `runs/l1_embeddings/full` (882 MB, eight
matrices + manifest) is the resumable asset and must not be deleted. `l1a` is idempotent — it
skips a checkpoint whose `.npz` is present and well-formed — so re-running it is a no-op.

**On a quiet machine, in this order:**

```bash
# L1 — the answer L1a already paid for. CPU only, ~1.7 h, ~1 GB resident.
uv run python artifacts/l1b_mlp_ladder.py --tag full

# L2 — the decisive arm. MPS, ~2 h at 1.479 steps/s. NOTHING else running.
uv run python artifacts/i2_sigreg_run.py arm --name d17 --steps 10500 \
    --checkpoint-every 1500 --runs runs/l2

# L2's probe — note the arm's own subdirectory, `runs/l2/d17`, not `runs/l2`.
uv run python artifacts/l1a_cache_embeddings.py --run runs/l2/d17 \
    --steps 1500 3000 6000 10500 --tag l2
uv run python artifacts/l1b_mlp_ladder.py --tag l2
```

**Do not run L1b and L2 together** — finding 1 above.

**Free memory before L2.** The arm needs a machine that is not already 39.72 GB into its
compressor. Quitting Docker Desktop, the Claude desktop app and any spare CLI sessions is what
that means in practice; none of them are mine to stop.
