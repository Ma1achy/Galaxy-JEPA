# H5 — the resolving run. The proposal is adopted, and the loss was lying.

Two arms, 3,000 steps each, same seed, same data order, same masks, `steps=50000` fixed so the
EMA ramp could not move. Only the `(lr, weight_decay)` sequence differs. The rule below was
committed **before** the arms ran (`artifacts/h5_decision_rule.md`; `git log` is the proof).

## The deciding measurement

Frozen-probe AUC on featured-ness, both encoders probed with the identical split, the identical
40,000-galaxy train subset and the identical 34,829-galaxy held-out set.

| arm | consensus AUC (headline) | all held-out | ambiguous middle |
|---|---|---|---|
| baseline | 0.9043 `[0.8988, 0.9097]` | 0.8084 `[0.8031, 0.8136]` | 0.6358 `[0.6260, 0.6461]` |
| **proposal** | **0.9358** `[0.9315, 0.9402]` | **0.8420** `[0.8376, 0.8467]` | **0.6624** `[0.6524, 0.6720]` |

**The proposal wins on all three, and no confidence interval overlaps on any of them** — the
nearest approach is 0.9315 against 0.9097, a gap of 0.022 between interval edges. Under the
pre-registered rule this is the unambiguous branch: **adopt the proposal**.

## The loss was actively misleading, and by a factor of 19

This is the finding worth carrying forward past this run.

| arm | loss @3000 | deepest ever | shape over final 100 | mean-cosine | AUC |
|---|---|---|---|---|---|
| baseline | **0.0164** | **0.0152** @2981 | flat | +0.984 | 0.9043 |
| proposal | 0.3168 | 0.0996 @768 | rising | +0.286 | **0.9358** |

The baseline is **19× better on loss** — and on *both* framings H2 had to separate, which agree
at 3,000 steps where they disagreed at 500. It then lost the objective decisively.

The mechanism is visible in the cosine column. Latent MSE is measured against a moving EMA
target, so a predictor and target that co-adapt onto a large shared mean component score
beautifully while encoding little. The baseline ends with mean pairwise cosine **+0.984** —
embeddings 98% aligned — and a loss near zero. That is not a model that learned the task well;
it is a model that made the task easy. **Low latent MSE is a collapse signature, not a score.**

Anyone reading these two arms on loss alone would have adopted the worse recipe.

## What the diagnostics did and did not predict

| | baseline | proposal | predicted AUC? |
|---|---|---|---|
| erank final | 7.91 | 11.77 (1.49×) | **yes** |
| erank minimum | 3.50 | 7.59 (2.17×) | **yes, more strongly** |
| std maximum | 11.68 | 4.07 (2.87×) | **yes** |
| std final | 4.46 | 4.03 (1.07×) | **no — the arms converge** |
| mean-cosine | +0.984 | +0.286 | **yes, and by the widest margin of all** |
| loss | 0.0164 | 0.3168 | **inverted** |

Three corrections to H2's reading, all of them things 500 steps could not show:

1. **`std_final` is the wrong summary; the trajectory is the signal.** The two arms finish within
   10% of each other by opposite routes — the baseline climbed to 11.68 and came back down, the
   proposal rose monotonically and never exceeded 4.07. A single end-of-run std would have called
   these equivalent. The **peak** separates them 2.87×.
2. **Mean-cosine is not uninformative.** H2 found it carried no ordering across six arms and this
   run was told to watch for that changing. It changed: +0.984 against +0.286 is the widest
   separation of any diagnostic here, and it tracks the AUC. H2's finding was an artefact of
   stopping at 500 steps, before the shared mean component had grown.
3. **The baseline's 500-step picture was a transient.** It bottomed at erank 3.50, recovered to
   7.91, and reached AUC 0.9043 — statistically indistinguishable from the pilot's 0.905. The
   baseline is a working recipe, not a broken one. The proposal had to beat a real comparator.

## The extension clause: not triggered

It permitted 6,000 steps only if the loss was still visibly descending **and** the arms had not
separated. Neither holds: the baseline's loss is flat (−0.0004 over the final 100 steps) and the
proposal's is rising (+0.0091), so nothing is descending; and the AUC intervals are cleanly
separated. **3,000 steps is the answer.** Stated before acting on it, as the rule required.

## The G5 floor could not speak, again

Its soft floor of 5.0 applies only from step 5,000 (10% of `steps`), and this run is 3,000. Only
the hard floor (erank < 2.0) could fire, and neither arm went near it. `would_halt` was `False`
throughout for both arms — **arithmetic, not a pass**. Worth noting separately: the baseline sat
below the soft floor's *value* for 53 consecutive readings, from step 125 onwards, and went on to
score AUC 0.9043. Had the grace elapsed, the frozen criterion would have killed a run that was
working. See the re-derivation note below.

## What this run does not settle

- **The cosine decay is untested.** The proposal carries the real 50,000-step cosine, not H2's
  compressed proxy, which makes it nearly inert here: the LR is still **99.7% of peak** at step
  3,000. This run tested the **peak and the warmup**. The decay is adopted on the reference
  recipe's authority alone, and that is the weakest-supported part of the proposal.
- **3,000 steps is not 50,000.** Both arms were still moving. The proposal's rank was still
  climbing (7.59 → 11.77) and its loss still rising.
- **Not like-for-like with the pilot.** 6,000 steps on 10,000 stamps seen ~19× each, against
  3,000 steps on 827k seen once. The pilot is an existence proof, not a baseline.
- **Both arms are stamped `smoke=True`.** The effect floor is open; nothing here is a ladder
  verdict, and the single-feature AUC is not the probing battery.
- **One seed, one feature.** No repeated seeds, so "separated" means separated on bootstrap
  intervals over the test set, not across training runs.

## A defect found in the production probe path

`harness.evaluate_probe` was the first choice and was killed for memory before embedding a single
stamp. It builds `rows_by_id(DirectorySource(probe_dir).rows)` — `csv.DictReader` over 230,358
rows × ~150 columns — while `StampDataset` reads exactly **one** of those columns. This is the
same multi-gigabyte metadata table the *training* path already escaped via the scalar sidecar
(Brief G2); the probing path still carries it. `artifacts/h5_probe_lean.py` works around it for
this run by reading two columns with `pandas.usecols`, but **the production path retains the
defect** and should get the same treatment as the training path. Logged in `TODO.md`.
