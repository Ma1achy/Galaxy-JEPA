# H5 — the decision rule, written before the resolving run produces a number

Pre-registered, in the same spirit as the G5 collapse floor: a rule chosen after seeing the
curve is not a rule. Committed before the arms are launched; `git log` is the proof of order.

## What is being compared

Two arms, 3,000 steps each. Every H2 control unchanged: same seed, same initial weights via
`harness.seed_init`, same data order (`ResumableShuffle`, one seed), same per-step mask seeds,
same monitor batch, and **`steps=50000` held constant in both** so the EMA momentum ramp is
identical (H1: it moves four parts in ten million across such a window, and H2 confirmed it
empirically — `ema_momentum` agreed to all 16 digits across all six arms).

| | baseline | proposal |
|---|---|---|
| peak LR | 1e-3 | **1.25e-4** (√-scaling of I-JEPA's batch-2048 1e-3) |
| warmup | 100 steps | **1,250 steps** (the reference's relative 2.50%) |
| decay | none | **cosine to 1.25e-7** (both endpoints scaled by the same √ factor) |
| weight decay | 0.04 constant | 0.04 constant — *deliberately unchanged* |

The WD ramp stays out. H2's `wd_ramp` arm was indistinguishable from baseline, 500 steps cannot
speak to a regularisation schedule, and bundling an unmeasured change with two measured ones
would make the outcome unattributable.

## The rule

**AUC is the objective. Effective rank and std are diagnostics.** The rule is written so that a
diagnostic can never outvote the objective — this is the `linear`-arm trap from H2 (highest rank
of all six arms, worst loss of all six, rejected) restated at a longer horizon.

| outcome | decision |
|---|---|
| proposal **≥** baseline on AUC **and** materially better on rank/std | **adopt the proposal** |
| proposal holds rank/std but **loses** on AUC | **keep the baseline** — the linear-arm trap at 3,000 steps |
| AUC **indistinguishable** | **adopt the proposal, on the reference-recipe argument alone** — √-scaling for an AdamW-family optimiser, and the reference's relative warmup. Stated as such: the traces are not what decided it |
| proposal wins AUC but **loses** rank/std | **adopt the proposal**, and say plainly that the rank diagnostic did not predict the objective — which would be a finding about the diagnostic, not about the schedule |

"Materially better on rank/std" is read against H2's own separations, which were large: erank
7.19 vs 3.75 and std 1.80 vs 6.20 at step 500. A difference of that order is material; a few
tenths of erank is not.

**"Indistinguishable" on AUC** means the two arms' bootstrap 95% CIs overlap. This is a
two-checkpoint comparison with no repeated seeds, so overlapping CIs is the honest reading of
"no separation detected", not proof of equality.

## Extending to 6,000 steps

Only if **both**: the loss is still visibly descending at 3,000 **and** the arms have not
separated under the rule above. If either fails, 3,000 is the answer. Which applies will be
stated before any extension, not after.

## What this run cannot settle, stated in advance

- **3,000 steps against the pilot's 6,000 on a different corpus is not like-for-like.** The pilot
  is 10,000 stamps seen ~19× each; this is 827k seen once. The pilot's AUC 0.905 is an
  **existence proof** that the premise works, never a baseline to beat. Any comparison drawn to
  it is illustrative.
- **Both runs are stamped `smoke=True`.** The effect floor is still open, so nothing here is a
  ladder verdict, and the stamp makes that structural rather than a matter of remembering.
- **The G5 soft floor still cannot speak.** Its grace is 10% of `steps` = 5,000, and this run is
  3,000. Only the hard floor (erank < 2.0) can fire. If it reports `False` that is arithmetic,
  not a pass, and it will be reported as such.
