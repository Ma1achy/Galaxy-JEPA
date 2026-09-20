# Brief M — the long λ=0 baseline run

**Outcome: FLATTENS EARLY.** The pre-registered stopping rule fired at **4 epochs**, and the run
stopped at 101,308 of 253,270 steps. The encoder is in hand and the budget question is answered
with data rather than an estimate.

---

## M1 — The budget, fixed before launch

`lr_final`'s cosine and the EMA ramp are both defined over `cfg.steps`, so changing `steps`
changes the recipe. It was fixed before launch and never touched: stopping early was allowed,
extending was not.

| quantity | value | source |
|---|---|---|
| corpus | 826,968 stamps | cache index |
| monitor slice | 16,477 | `monitor_frac: 0.02` |
| **train split** | **810,491** | `_split_ids(cfg)` |
| batch | 32, `drop_last=True` | config |
| **steps/epoch** | **25,327** | 810,491 // 32 |
| **10-epoch budget** | **253,270 steps** | |
| **estimate** | **~46.4 h** at 1.515 steps/s | Brief L's λ=0 arm, same code path |

The brief's ~258,000 was an estimate against the full corpus; 253,270 is the figure against the
actual train split. The pre-flight's own epoch line was wrong in the same way at first — it
divided by the whole corpus and reported a 10-epoch budget as 9.80 epochs. An epoch is a pass
over what the model **trains** on, and the monitor slice is held out. Fixed before launch.

**Why 10 epochs.** Every run before this one was diagnostic: 3,000 steps is 0.12 epochs, 10,500 is
0.42, and even J's 50,000-step run was 1.97. The reference trains for 600. Ten was chosen to be
ambitious enough not to hit the ceiling and wish it had gone further, with the stopping rule
protecting against paying for a plateau. **It did exactly that: the rule fired at 4 and returned
~28 h.**

## M2 — Pre-flight, as checked facts

`M1 PASS — every fact checked, nothing assumed`, against the run's own recipe rather than J's:

* **D21 in the config**, `sigreg_lambda = 0.0`, asserted in **both** directions — a silent flip
  either way is an error.
* **The hash chain gained a link**, because `steps` and `checkpoint_every` are both determining
  and M moved the budget. Each strip names exactly one thing that moved:
  `046659910b5fd543 → 7ecf5dce5a1f60ba → de87b8f9704b7e2d → 538bf997880a8767`
  (as shipped → minus budget → minus smoke → minus SIGReg). **D17's anchor is unchanged**, as it
  has been through both D18's adoption and D21's reversal.
* Stamped hash `v2:61330a0012234374` on MPS — the resolved config, which is a different dump from
  the file chain above and is printed as such rather than conflated.
* Normalisation freeze loads, refit refused; cache dense at 1,057,326 stamps with the sidecar
  digest verified; `vote_count_min` frozen; **`effect_floor` unset and `headline=True` refused**;
  `smoke: true` stamped into `escape_hatches_used`.
* **Soft rank floor ACTIVE**, not inert — under D18 the penalty made it unfirable, and a criterion
  that cannot fire is decoration. This is the first run in the project to pass the grace period
  (0.10 × 253,270 = 25,327 steps) at all, so it is the first where the floor was ever reachable.
* `keep=49`, covering 40 scheduled checkpoints **and** the 7 segment-end ones. The harness derives
  `scheduled + 2`, which is short by one per segment — the `keep=3` failure class.
* `out_dir: runs/m`, **not** `runs/full`: that is Brief J's run and K2's whole trajectory was
  probed from its checkpoints. 2,448 GiB free.

### The `code_dirty` defect, fixed rather than carried

`_make_stamp` ran at `harness.py:568`, **after** `_prepare` at 551 — so dirtiness was sampled once
preparation had already taken minutes, and across a two-day run an edit made mid-flight would be
recorded as though the run had executed it. `code_sha` is now read once at import and cached per
repo. Pinned by an invariant test that dirties the tree and asserts the reading does not move.

Brief L met the same defect from the other side, when `runs/i2/d17`'s stamp could not be
reconciled with any committed state.

## M3 — The curve

Segmented so the stopping rule could act: training stops at each probe point, the probe runs with
the machine to itself, training resumes. Probing *alongside* training is the lesson already paid
for twice — a CPU sweep beside a training loop costs 2.8× because the loop's dataloader starves,
and the machine OOM'd once.

**Segmenting is inert, and that was measured, not asserted.** A 60-step run as 20/40/60 and as one
continuous invocation gave losses **identical step for step**; so did an interrupted-and-resumed
run against a continuous one. `config.steps` stayed 253,270 in every segment: segmenting changes
*when the process exits*, never the recipe.

| epochs | step | consensus AUC | interval | all held-out | ambiguous | ΔAUC | slope/epoch |
|---|---|---|---|---|---|---|---|
| 0.50 | 12,664 | 0.9593 | [0.9562, 0.9624] | 0.8702 | 0.6784 | — | — |
| 1.00 | 25,327 | 0.9631 | [0.9600, 0.9662] | 0.8781 | 0.6927 | +0.0038 | +0.0076 |
| 2.00 | 50,654 | 0.9642 | [0.9613, 0.9671] | 0.8810 | 0.6992 | +0.0010 | +0.0010 |
| **4.00** | **101,308** | **0.9646** | **[0.9619, 0.9675]** | **0.8818** | 0.6948 | **+0.0005** | **+0.0002** |

Same held-out split (34,829) and probe config as K2/L2, so these sit directly alongside:

| run | λ | epochs | consensus |
|---|---|---|---|
| J / K2, best point | 0.05 | 0.06 | 0.9477 |
| J / K2, endpoint | 0.05 | 1.97 | 0.9278 |
| L2 arm, endpoint | 0 | 0.42 | 0.9554 |
| **M, at the plateau** | **0** | **4.00** | **0.9646** |

M at half an epoch (0.9593) already exceeds everything λ=0.05 reached anywhere on its 50,000-step
trajectory, and it continues from there. All three measures — consensus, all-held-out and the
ambiguous middle — rose together through 2 epochs, so the gain is not the consensus cut flattering
itself. Only the ambiguous middle turned at 4 (0.6992 → 0.6948), against a rising consensus and
all-held-out; on one trajectory that is not attributed.

### Traces

| step | prediction loss | effective rank | std | mean cosine |
|---|---|---|---|---|
| 12,700 | 0.2984 | 22.2 | 5.06 | 0.380 |
| 25,327 | 0.1898 | 28.7 | 5.11 | 0.574 |
| 50,654 | 0.1357 | 33.5 | 3.29 | 0.711 |
| 101,300 | 0.1155 | 37.3 | 2.84 | 0.606 |

**Effective rank rose 22.2 → 37.3 while AUC also rose.** Under λ=0.05 rank rose 24.3 → 57.6 while
AUC *fell*. Rank growth is therefore not the thing that costs morphology — which is consistent
with L1's finding that the λ=0.05 decline was information leaving rather than being spread thinner,
and it removes rank growth on its own as a suspect. **std peaks at 5.11 (step 25,327) and ends at
2.84**; mean cosine peaks at 0.711 (step 50,654) and eases to 0.606.

**No halt.** Zero halt lines. The soft rank floor was active from step 25,327 and effective rank
was 37.3 and climbing away from the 2.5 threshold, so it never came close — which is also evidence
that 2.5 is not set too high.

### Throughput and thermal behaviour

| segment | steps | hours | steps/s |
|---|---|---|---|
| 1 | 12,664 | 2.30 | 1.5293 |
| 2 | 12,663 | 2.41 | 1.4586 |
| 3 | 25,327 | 4.86 | 1.4468 |
| 4 | 50,654 | 9.74 | 1.4448 |

**No thermal drift worth the name.** Throughput fell 5.5% between segment 1 and segment 2 and then
stayed flat across the next 14.6 h — a step, not an accumulation, which is the signature of memory
pressure rather than heat. The 11-hour run had shown none; this is 19.3 h and shows none either,
once that first step is accounted for. Absolute die temperature needs `sudo powermetrics` and is
deliberately not measured.

**Memory was the standing risk throughout.** The compressor held 6.84 GB of RAM storing 33.78 GB
of logical memory — the machine ran ~34 GB of demand in 19 GB of physical, with free RAM at
0.06–0.07 GB for most of the run. The training process itself was modest at ~1.25 GB. Five polling
waiters were reaped by the memory manager; **the training run never was**, and the probe phase —
the ~1 GB spike — survived every time because training is paused while it runs.

### Interruptions

Two, neither costing data.

1. **The SSD was unplugged** an hour in, at step 5,792 with the first checkpoint due at 6,331. That
   hour was lost; nothing was corrupted. It exposed that segment 1 always started fresh
   (`resume = i > 1`), which is wrong exactly when a run is interrupted inside its first segment —
   the case that costs most.
2. **The driver crashed reading the first probe's result**, having the filename inverted. The
   probe had **succeeded**; two hours of training survived in the checkpoint, and the relaunch
   recovered the probe point from the file the probe itself wrote rather than re-extracting.

Both fixed, and the resume path is now proven the same way segmenting was — by measurement.

## M4 — Which outcome fired

Pre-registered, before any number existed:

> **Stop early** if consensus AUC improves by less than ~0.002 across two consecutive probe
> intervals, and the trend is not still rising within intervals.

**FLAT fired at 4 epochs:** ΔAUC **+0.0010 then +0.0005**, both below 0.002, and not rising. The
run stopped at 101,308 of 253,270 steps — **4.00 of 10.00 epochs, returning ~28 h of the budget.**

**The caveat I raised in advance did not bite, and it is worth saying why.** Probe intervals are
unequal, so a raw ΔAUC favours stopping late when intervals are widest: the 2→4 epoch interval is
twice as wide as 1→2 and has twice the room to accrue. At the previous slope it would have
accrued ~+0.0020 and landed exactly on the threshold. It accrued **+0.0005** — a quarter of that.
The wider interval did not mask a plateau; it confirmed one, and the slope per epoch agrees
(+0.0076 → +0.0010 → +0.0002, a 38× collapse). Rule and slope point the same way, so the
disagreement clause never had to be used.

**The shape, in one line.** Three quarters of the total gain from 0.5 epochs to the plateau
(+0.0053) is already in by 1 epoch (+0.0038). The remaining 3 epochs bought +0.0015.

## What this run does not say

* **No ladder verdicts, no rung assignments.** `effect_floor` is unset, so the hard gate cannot
  fire, and nothing here imports `run_ladder` or `existence_verdicts`.
* **No effect-floor freezing.** J5's argument stands.
* **No AUC-based checkpoint selection.** 1C stands; Brief I measured that question and the answer
  was no. The plateau is not a licence to pick a checkpoint by its probe score.
* **No SIGReg conclusions.** D21 is a fix for the regime that can actually be run, not a verdict on
  the method. SIGReg estimates the embedding distribution from 32 samples in 384 dimensions, which
  is severely underdetermined and a small-batch problem as much as a SIGReg one; the paper used
  batch 2048. **A fair re-test needs this baseline, which now exists.**
* **One seed, one trajectory, four probe points.** The plateau is located between 2 and 4 epochs on
  this run. It is not bounded, and a second seed could place it elsewhere.

## The numbers a rental case would be argued from — reported, not acted on

The curve flattened, so the rental argument is **weak on this evidence**: the final slope is
**+0.0002 AUC/epoch**, and extrapolating it, another 10× of compute (100 epochs) would buy roughly
+0.002 if the slope held — about the width of one confidence interval, and slopes of a flattening
curve do not hold.

Where compute would plausibly pay instead, in order, none of them acted on here:

1. **Batch size.** 32 against the reference's 2048 is the largest unexamined divergence in the
   recipe, and it is the hypothesis behind SIGReg's failure here. A bigger batch is a different
   experiment, not more of this one.
2. **A second seed at 4 epochs**, to put an interval on the plateau rather than a point.
3. **The SIGReg re-test**, now that a λ=0 baseline at a real horizon exists to test against.

---

## Provenance

`runs/m` (20 checkpoints retained, `encoder.pt` 82 MB), `artifacts/out/m_curve.json`,
`artifacts/out/m{1,2,3,4}_trajectory.json`, `runs/m2.log` + `runs/m2b.log`.
Stamped `v2:61330a0012234374`, `smoke: true`, `escape_hatches_used: [smoke]`.
