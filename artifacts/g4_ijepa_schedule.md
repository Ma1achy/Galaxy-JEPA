# G4 — what the I-JEPA paper actually says about its pretraining schedule

Checked against the paper rather than recalled, because the figure was anchoring a decision
spanning 10.7 h to 69 d. Source: Assran et al., *Self-Supervised Learning from Images with a
Joint-Embedding Predictive Architecture*, arXiv:2301.08243 (CVPR 2023) — abstract page and full
text via ar5iv.

## The schedule, as published

| model | dataset | epochs |
|---|---|---|
| ViT-B/16 | IN1K | 600 |
| ViT-L/16 | IN1K | 600 |
| ViT-H/14 | IN1K | 300 |
| ViT-H/14 @ 448 | IN1K | 300 |
| ViT-H/14 | IN22K | "equivalent of 900 IN1K epochs" |
| ViT-G/16 | IN22K | "equivalent of 600 IN1k epochs" |

- **Batch size 2048** ("our default batch-size is 2048").
- **LR**: "linearly increased from 1e-4 to 1e-3 during the first 15 epochs of pretraining, and
  decayed to 1e-6 following a cosine schedule".
- **Weight decay**: "linearly increased from 0.04 to 0.4 throughout pretraining".
- **EMA**: momentum 0.996, "linearly increase this value to 1.0 throughout pretraining".
- **Predictor**: embedding dimension 384; depth 6 for ViT-B/16, 12 for ViT-L/16 and ViT-H, 16 for
  ViT-G/16.
- **Masking**: 4 target blocks, scale (0.15, 0.20), aspect ratio (0.75, 1.50); context block scale
  (0.85, 1.00) at unit aspect ratio.
- **Hardware**: "we train a ViT-Huge/14 on ImageNet using 16 A100 GPUs in under 72 hours".

Converted to the units the budget decision is actually made in (IN1K = 1,281,167 images):

| configuration | samples seen | optimiser steps at batch 2048 |
|---|---|---|
| 600 epochs | 769M | 375,000 |
| 300 epochs | 384M | 188,000 |
| 900 IN1K-equivalent | 1.15B | 563,000 |

**The prior claim was approximately right and is now checked.** `TODO.md` said the configured
50,000 steps is "~1/300 of I-JEPA's published ImageNet epoch schedules". Against the 600-epoch
ViT-B/L schedule and the corrected 1.97-epoch figure below, the ratio is 1/305. The number held;
it is now a citation rather than a recollection.

## How comparable is a ViT-S on 827k?

**Not very, and the honest answer matters more than the epoch count.** Five divergences, worst
first.

1. **The paper never pretrains a ViT-S with I-JEPA at all.** Its smallest is ViT-B/16 (86M). Every
   ViT-S/16 number in the paper is a *baseline from another method* (iBOT), quoted for comparison.
   So there is no published I-JEPA schedule for this model scale to borrow. The paper in fact makes
   a point in the other direction — "a huge I-JEPA model (ViT-H/14) requires less compute than a
   small iBOT model (ViT-S/16)" — which is an argument about I-JEPA's efficiency, not a schedule
   for a 21.6M-parameter encoder.
2. **The batch is 64× smaller** — 32 here against 2048 — at the same peak LR of 1e-3. Their 1e-3
   is a batch-2048 learning rate. Nothing in this project has established that 1e-3 at batch 32 is
   the corresponding setting, and the usual scaling rules would put it far lower. A borrowed epoch
   count carries an unstated assumption that the optimisation is equivalent, and it is not.
3. **The LR and WD schedules are not the published ones.** `train_jepa` applies linear warmup and
   then holds `lr` constant; there is no cosine decay to 1e-6 and no 0.04 → 0.4 weight-decay ramp.
   In a 600-epoch run a large part of the final representation quality comes from the low-LR tail.
   Borrowing the epoch count without the anneal borrows the cost and not the mechanism.
4. **Different domain, and a modified objective.** 3-channel 256 px native-resolution SDSS stamps
   at 0.396″/px, no resampling, against natural images; and at the headline β = 0.5 the masking is
   bbox-biased, which is this project's contribution and not their objective. Only β = 0 is the
   published control.
5. **Corpus scale is the one thing that is close**: 810,491 training stamps against 1,281,167
   IN1K images, 0.63×. And the **masking geometry is identical** — 4 target blocks at scale
   (0.15, 0.20), aspect (0.75, 1.50), context (0.85, 1.00) — so `MaskConfig`'s defaults are
   faithful to the paper, which is worth knowing for the β = 0 control's integrity.

## What this means for the budget

**The paper cannot anchor it.** Matching 769M samples seen would be 24.0M steps at batch 32 —
214 days on the M3 Pro at the measured 1.297 steps/s, and 948 epochs over this corpus. That is not
a schedule to scale down by judgement; it is a different compute regime.

So the budget has to be argued on **samples seen** and **the collapse trace**, as the brief
anticipated. The two reference points this project actually owns, both from the pilot
(10,000 stamps, 6,000 steps, batch 32, frozen-probe AUC 0.905):

| framing | steps at batch 32 | samples | epochs over 810,491 | M3 Pro wall-clock |
|---|---|---|---|---|
| configured `steps: 50000` | 50,000 | 1.60M | 1.97 | 10.7 h |
| pilot **samples** parity | 6,000 | 0.19M | 0.24 | 1.3 h |
| pilot **epoch** parity (19.2×) | 486,000 | 15.6M | 19.2 | 4.3 d |
| I-JEPA ViT-B/L samples parity | 24,022,000 | 769M | 948 | 214 d |

The two pilot framings disagree by 81×, and the disagreement is the substance: the pilot saw
10,000 distinct stamps nineteen times each, which is a different learning problem from seeing
810,491 stamps twice. Neither framing is obviously right. What is defensible is that the
configured 50,000 steps already exceeds the pilot's samples seen by 8.3× on a corpus 81× larger,
and that the collapse trace — not a borrowed epoch count — is the signal that says whether more
steps are buying representation or burning hours.
