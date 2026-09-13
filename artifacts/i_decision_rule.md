# Brief I — the decision rule, fixed before any arm runs

Committed ahead of the measurement, as H5's was. `git log` is the proof. Nothing below is
adjusted after a number is seen; if an arm falls outside what is written here, the write-up says
so rather than the rule moving.

## What is being asked

H5 measured the training loss running **against** the objective: the arm 19× better on latent
MSE (0.0164 vs 0.3168) lost frozen-probe AUC decisively (0.9043 vs 0.9358, no interval overlap).
Latent MSE against a moving EMA target rewards co-adaptation onto a shared mean component — the
losing-on-AUC arm ended at mean pairwise cosine **+0.984**. Selecting on loss would have kept the
worse recipe.

LeJEPA (Balestriero & LeCun, arXiv:2511.08544) claims to fix exactly that. So two questions:

1. **Does SIGReg improve the representation?** Read on AUC.
2. **Does SIGReg make the training loss usable as a label-free selection signal?** Read on the
   within-run rank correlation between loss and AUC across checkpoints.

This is an ablation *on I-JEPA*, not adoption of LeJEPA. Masking, ViT, EMA target, predictor,
optimiser and the D17 schedule are untouched.

## The arms

Three, all sharing H5's controls: one `seed_init` so every arm starts from identical weights, one
`ResumableShuffle` seed, the same per-step mask seeds, the same 64-stamp monitor batch, and
`steps=50000` held fixed so the EMA ramp cannot move. 3,000 steps each, checkpointed every 500.

| arm | λ | provenance |
|---|---|---|
| `d17` | 0.0 | the comparator — the adopted recipe, no penalty computed |
| `sigreg_050` | 0.05 | §6.1 verbatim: *"we thus recommend to use λ = 0.05, V_g = 2, V_l = 8, and batch size ≥ 128 as starting points"* |
| `sigreg_006` | 0.00625 | 0.05 × (1/8), from §6.1's note that peak performance comes from adjusting λ *"proportionally to the number of views"*. We have one context view against their eight |

**Both λ values are fixed here, before any arm launches.** This is a pre-registered pair, not a
sweep: no third value is added afterwards, and a λ chosen because it produced the best AUC would
be a researcher degree of freedom with a number attached. `sigreg_006` has the weaker provenance
of the two — it is extrapolated past the range of their figure 8 — and is labelled so in the
report whatever it returns.

## Question 1 — the AUC rule

Probe every arm's final frozen encoder identically: the same held-out split, the same
40,000-capped deterministic train stride, the same `ProbeConfig`, so the numbers sit directly
alongside H5's **0.9043 / 0.9358**. Report consensus, all-held-out and ambiguous-middle with
bootstrap intervals.

**A SIGReg arm wins only on non-overlapping intervals against `d17`.** Overlapping intervals are
"indistinguishable", and indistinguishable is not a win — there is no reference-recipe tiebreak
here as there was in H5, because the thing under test is the addition itself.

## Question 2 — the loss-usability read, and its limits stated first

Six checkpoints per arm (500…3000), each probed on a **reduced fixed subset** (8,000 train /
8,000 test, identical across every checkpoint and arm). Spearman ρ between each loss component
(total, prediction, penalty) and AUC, **within each arm**.

Stated before the numbers, because they bound what any result can mean:

- **n = 6 per arm.** |ρ| = 1 is p ≈ 0.0028 two-sided; anything less is not much. Reportable, weak.
- **It is not the paper's quantity.** Their ρ ≈ 0.99 is *across runs* over a hyperparameter sweep
  (figure 11: "across numerous hyper-parameters such as learning rate, weight decay, number of
  epochs, λ"). Ours is *within a run over training time*. Different things — and ours is the one
  the 1C checkpoint rule actually needs. It must not be reported as confirming their number.
- **The α-scaling law (eq. 8) is inapplicable.** It needs λ to vary within the correlated set.
  Report α = 0, the plain loss, only.
- **Across-run correlation is unavailable.** Loss values are not comparable between the λ=0 arm
  and the SIGReg arms, because the objective differs — one point per family.

## Guards

- **Isotropic-but-empty.** SIGReg drives effective rank up and mean-cosine towards zero *by
  construction*. That is constraint satisfaction and is **not** evidence. Rank and cosine are read
  alongside AUC, never instead of it. An arm that holds rank and cosine without improving AUC is
  the regulariser winning at the objective's expense — H2's `linear` trap at a longer horizon.
- **The constraint is loose at this batch size, measured.** With n=32 the empirical CF over random
  1-D projections pins the scale only weakly: in a direct-embedding probe the statistic reached
  its floor while the batch's std was still ≈ 1.7. Satisfaction accumulates across resampled
  directions over steps, not within one. So "the penalty is near its floor" does not by itself
  mean "the representation is isotropic", and the two are reported separately.
- **3,000 steps may not be enough for the penalty to land.** A direct-embedding probe at AdamW
  lr 1e-2 reached isotropy in ~1,200 steps; at lr 1e-3 it had barely moved after 3,000. The
  encoder's peak LR is 1.25e-4 and the penalty reaches it through the network, so a **partial
  effect is a real possible outcome**. If the penalty is still descending at 3,000 that is stated
  as a limit of the horizon — the same shape as H5's finding that the cosine decay was nearly
  inert inside 3,000 steps — and must not be read as "SIGReg does not help".
- **`smoke: true`** on every arm and every probe, so none can be read back as a result.

## Expected magnitudes, so a surprise is recognisable

At this project's embedding scale (std ≈ 4) the penalty should start near **≈ 26**; its floor for
a genuinely standard-Gaussian batch at n=32 is **≈ 1.0**, not zero — the finite-batch estimate is
positively biased (the paper's thm. 6). Values are read as differences, never absolutely.

## If a SIGReg arm wins

**Propose; do not merge.** `sigreg_lambda` stays 0.0 in `configs/pretrain.yaml` regardless of
outcome. A D18 draft must trace: what happens to `CollapseFloorFreeze` once effective rank is
constraint-satisfied rather than diagnostic (re-derive or retire — say which); whether the
logistic-vs-CAV cross-check survives, **measured in both arms rather than argued from
Σ_within = σ²I − Σ_between**; the D12 framing question of one arm carrying a distributional
constraint the others lack; and whether the 1C checkpoint rule becomes replaceable.

**Then stop.** The full pretraining run is not launched from this brief.
