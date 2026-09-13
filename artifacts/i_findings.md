# Brief I — SIGReg on I-JEPA. It improves the representation; it does not fix the loss.

Three arms at 3,000 steps, then eighteen frozen probes. The rule was committed before any arm ran
(`artifacts/i_decision_rule.md`, commit `1ff1df9`; `git log` is the proof).

**Two answers, and they point in different directions.** SIGReg improves AUC on non-overlapping
intervals at both pre-registered λ values. It does **not** make the training loss usable for
checkpoint selection: selecting on lowest loss still costs AUC in every arm, SIGReg included.

---

## I1 — What was built, and where it differs from the paper

`src/galaxy_jepa/objectives/sigreg.py` implements Definition 2 with the Epps–Pulley statistic —
the weighted L2 distance between the empirical characteristic function of random 1-D projections
and the CF of N(0, 1). **1024 slices, 17 quadrature points, domain [−5, 5]**, all from §6.1's
stated recommendation (their algorithm 1 listing defaults to 256 slices; the recommendation is the
later, ablated number). Directions are **resampled every step**, which their figure 7 shows
beating a fixed set substantially.

| # | Here | The paper | Why |
|---|---|---|---|
| 1 | Predictor + EMA target retained | LeJEPA removes both | This is an ablation *on* I-JEPA |
| 2 | bbox-biased multi-block masking, one context view | DINO multi-crop, V=8 | Keeping the masking geometry was the brief's scope discipline |
| 3 | Latent MSE against EMA targets | views-predict-global-mean | I-JEPA is the thing being ablated |
| 4 | **Attached at the penultimate block, pre-norm** | the encoder output | See below |
| 5 | batch 32 | recommends ≥ 128 | 18 GB machine; their O(1/N) bias is "not a concern… as small as 16" |
| 6 | **Real arithmetic** | complex ECF | MPS complex kernels; exact reformulation, and verified so |
| 7 | Sketch drawn on the host | device generator seeded per step | Backend-independent directions |

**(4) is the one real judgement call.** LeJEPA regularises the encoder output because for LeJEPA
that *is* what a probe reads. Here they differ: `DEFAULT_LAYER = -2`, so `encode()` reads the
penultimate block pre-norm, while the predictor's context is the final block post-`norm`. The
intent is to constrain the distribution that gets probed, so that is where it goes — which is
also the tensor `CollapseMonitor` reads, so its rank/std/cosine describe the constrained object.
The final block and `norm` stay free to specialise for the pretext task.

**(6) is exact, not an approximation.** The target CF is real, so `|φ̂ − φ|² = (C − φ)² + S²`.
Checked against a transcription of algorithm 1 across four input scales: agreement to 0.0e+00.

**λ and its provenance.** Both values fixed before any arm ran. `sigreg_050` = **0.05**, §6.1
verbatim. `sigreg_006` = **0.00625** = 0.05 × (1/8), from §6.1's note that peak performance comes
from adjusting λ "proportionally to the number of views" — one context view here against their
eight. The second has the weaker provenance, being extrapolated past figure 8's range.

**The in-domain prior is adjacent, not identical.** The paper's Galaxy10 result (figure 12, §6.3)
is 11,000 stamps, 10-class, DECaLS, ResNet/LeViT, 400 epochs. This is 827k SDSS stamps, binarised
vote fractions, ViT-S. Encouragement, not a prior on our number.

---

## I2 — The arms

Every H5 control unchanged: one `seed_init`, one `ResumableShuffle` seed, the same per-step mask
seeds, the same 64-stamp monitor batch, `steps=50000` fixed so the EMA ramp could not move. Only
`sigreg_lambda` varies. Unlike H5, the loop is the **production** `train_jepa` reading D17 from
`configs/pretrain.yaml`.

| | d17 (λ=0) | sigreg_050 (λ=0.05) | sigreg_006 (λ=0.00625) |
|---|---|---|---|
| steps/s | 1.4790 | 1.4765 | 1.4782 |
| loss @3000 | **0.3573** | 0.4285 | 0.4028 |
| loss deepest / at step | **0.0721** @851 | 0.2939 @1559 | 0.2276 @1070 |
| loss last-50 | **0.2951** | 0.3863 | 0.3533 |
| effective rank final | 11.77 | **34.23** | 29.87 |
| effective rank min | 7.59 | **15.99** | 14.67 |
| std final | 4.03 | **0.996** | 1.255 |
| std max | 4.07 | **1.024** | 1.357 |
| mean cosine | +0.286 | **+0.114** | +0.116 |
| penalty, first → final | — | 35.24 → 1.61 | 35.24 → 1.93 |

**The throughput cost of SIGReg is 0.17%.** 1024 slices per step at batch 32 is not measurable
against a ViT-S forward and backward.

**The pre-registered "it may not land in 3,000 steps" risk did not materialise.** The penalty fell
from 35.24 to 1.61 against a measured floor of ≈ 1.0, and std reached unit by roughly step 2,000.

**One correction to my own framing.** The plan said SIGReg would *shrink* embeddings from std ≈ 4
to ≈ 1. It does not. Both arms start at std 0.303; what SIGReg does is **prevent the fourfold
growth** `d17` undergoes. The penalty is holding a scale, not reducing one.

### The replication check passed exactly

`d17` is the same recipe H5 called `proposal`, so it had to reproduce it — and did, through a
different code path (production `train_jepa` + `configs/pretrain.yaml`, against H5's driver-local
schedule class):

| | H5 `proposal` | I2 `d17` |
|---|---|---|
| loss at the last monitor reading (step 2975) | 0.3168 | 0.3168 |
| loss at step 3000 / deepest / at step | 0.3573 / 0.0721 / 851 | 0.3573 / 0.0721 / 851 |
| erank final / std final / mean cosine | 11.7654 / 4.0252 / 0.2856 | 11.7654 / 4.0252 / 0.2856 |
| consensus AUC | 0.9358 [0.9315, 0.9402] | 0.9358 [0.9315, 0.9402] |

H5's headline "loss @3000 = 0.3168" is in fact the monitor reading at **step 2975**; a per-step
latent MSE swings enough between neighbours that the distinction is worth stating.

### Two defects this run found, both in production code

- **`train_jepa` never released the MPS allocator pool.** Measured at batch 32: it settles at
  6.9 GB while only 0.87 GB is live — 3.2 GB of an 18 GB machine held for nothing. A second arm
  was killed for memory after the first had finished. Releasing at the monitor interval holds it
  at 3.67 GB, costs nothing measurable (133 s vs 135 s over 200 steps), and is **numerically
  inert**: after the change, `d17` reproduced all 3,000 per-step losses bit-identically, and ran
  **5.5% faster** (1.402 → 1.479 steps/s) for the reduced pressure. H5's own driver did this; the
  production path did not.
- **`ObjectiveConfig.from_jepa_config` silently dropped `lr_final`.** A round trip turned D17's
  adopted cosine decay back into warmup-only while the config still looked clean — and that class
  is what gets *stamped*. Now pinned by comparing the shared field set rather than by hand.

---

## I3 — The two questions, read separately

### Question 1 — AUC. Both SIGReg arms win.

Identical probe to H5's: same held-out split, same 40,000-capped deterministic train stride, same
`ProbeConfig`. n_train 25,305; n_test 21,974 consensus / 34,829 all / 12,855 ambiguous.

| arm | consensus (headline) | all held-out | ambiguous middle |
|---|---|---|---|
| H5 `baseline` | 0.9043 [0.8988, 0.9097] | 0.8084 [0.8031, 0.8136] | 0.6358 [0.6260, 0.6461] |
| `d17` | 0.9358 [0.9315, 0.9402] | 0.8420 [0.8376, 0.8467] | 0.6624 [0.6524, 0.6720] |
| **`sigreg_050`** | **0.9470** [0.9435, 0.9506] | **0.8558** [0.8516, 0.8603] | 0.6734 [0.6637, 0.6829] |
| **`sigreg_006`** | **0.9471** [0.9433, 0.9508] | **0.8573** [0.8529, 0.8616] | 0.6760 [0.6661, 0.6854] |

Against the pre-registered rule — non-overlapping intervals or nothing:

- **consensus: separated, higher.** +0.0112 and +0.0113.
- **all held-out: separated, higher.** +0.0138 and +0.0153.
- **ambiguous middle: intervals overlap.** Both arms are numerically higher (+0.0110, +0.0136) but
  this reads as **indistinguishable**, and is reported as such.

**The two λ values are indistinguishable from each other**: 0.9470 against 0.9471 across an 8×
difference in weight. That is the paper's robustness claim reproduced rather than assumed, and it
is the strongest argument that the effect is SIGReg rather than a tuned coefficient.

**The loss inverted the answer again.** `d17` is best on loss on every framing — @3000, deepest,
and last-50 — and worst on AUC. H5's central finding replicates on a new axis.

### Question 2 — loss usability. No.

Six checkpoints per arm on a **reduced fixed subset**, identical at every checkpoint and in every
arm: n_train 5,084 / n_test 5,010 after the extremes filter. These AUCs are therefore **not**
comparable with Q1's — a smaller probe-train set is worth roughly 0.01 AUC — but they are
comparable *with each other*, which is what a rank correlation needs.

Exact two-sided permutation p, over all 720 orderings. **At n=6 the 5% threshold is |ρ| = 0.886.**

| arm | ρ(total loss, AUC) | p | ρ(prediction, AUC) | ρ(penalty, AUC) |
|---|---|---|---|---|
| `d17` | **+0.657** | 0.175 | +0.657 | — |
| `sigreg_050` | +0.086 | 0.919 | −0.143 | −0.314 |
| `sigreg_006` | −0.086 | 0.919 | −0.086 | **−0.714** (p 0.136) |

**No correlation here is significant.** Every p is ≥ 0.136. What can be said:

1. **The anti-correlation weakens under SIGReg.** `d17` reproduces H5's finding *within a single
   run* at ρ = +0.657 — lower loss, worse AUC — and both SIGReg arms sit at ≈ 0. Directionally
   this is the claimed effect. It is not measured to significance at n=6.
2. **It does not become a usable signal.** ≈ 0 is not the paper's ρ ≈ 0.99. Removing a misleading
   signal is not the same as supplying a good one.
3. **The practical read is clearer than the correlation, and it is the one that matters.**
   Selecting the lowest-loss checkpoint costs AUC in **every** arm:

   | arm | best-AUC checkpoint | lowest-loss checkpoint | cost of selecting on loss |
   |---|---|---|---|
   | `d17` | step 3000 (0.9245) | step 1000 (0.9135) | **−0.0110** |
   | `sigreg_050` | step 1000 (0.9413) | step 2000 (0.9341) | **−0.0072** |
   | `sigreg_006` | step 3000 (0.9394) | step 1000 (0.9213) | **−0.0181** |

   SIGReg roughly halves the cost at λ = 0.05 and makes it worse at λ = 0.00625. **On this
   evidence the 1C checkpoint rule must not change.**
4. **The penalty term alone is the most informative component** (−0.314, −0.714) — more so than
   the total loss it is part of. Suggestive only, and worth a properly powered test rather than a
   conclusion.

**What this is not.** The paper's ρ ≈ 0.99 is *across runs* over a hyperparameter sweep (figure
11: "across numerous hyper-parameters such as learning rate, weight decay, number of epochs, λ").
This is *within a run over training time* — a different quantity, though the one the 1C checkpoint
rule actually needs. The α-scaling law (eq. 8) requires λ to vary within the correlated set and is
inapplicable; α = 0 only. Across-run correlation is unavailable: loss values are not comparable
between the λ=0 arm and the SIGReg arms, leaving one point per family.

---

## I4 — The guards held

- **Isotropic-but-empty did not fire.** Rank and cosine moved enormously *by construction* — which
  is constraint satisfaction and not evidence — **and** AUC improved on separated intervals. Both
  moved together, so this is not H2's `linear` trap. Had AUC been flat, the rank would have been
  worth nothing.
- **λ was not tuned.** Both values were committed before any arm ran and no third was added. The
  two are indistinguishable on AUC, so there was nothing to tune towards.
- **Both arms and all probes are stamped `smoke: true`.**
- The scale confound was closed in advance: `probe_auc_ci` fits a `StandardScaler` before the
  logistic, so SIGReg holding std at 1.0 against `d17`'s 4.0 does not change the probe's effective
  regularisation. The measured `embedding_std` per checkpoint confirms the gap it would have been
  (d17 1.92 → 4.71; SIGReg arms flat at 0.80–0.96).

---

## I5 — What adoption would mean. Proposed, not merged.

`sigreg_lambda` remains **0.0** in `configs/pretrain.yaml`. A D18 draft is in `DECISIONS.md`.

- **The collapse machinery changes meaning.** Effective rank becomes constraint-satisfied rather
  than diagnostic: 34.23 against `d17`'s 11.77, and monotone. `CollapseFloorFreeze` was re-derived
  to 2.5 only last brief, on grounding H5 had already undercut — the H5 baseline sat below the old
  5.0 for 53 consecutive readings and still scored 0.9043. Under SIGReg it would never bind at
  all. **Recommendation: retire the soft rank floor for any SIGReg run and keep only the hard
  floor**, because a criterion that cannot fire is not a tripwire, it is decoration. Retiring is
  honest; re-deriving a number that the constraint guarantees is not.
- **The eigen-triangulation, measured rather than argued.** The logistic-vs-CAV disagreement
  **persists**. It declines in all three arms over training — `d17` 0.9804 → 0.9479, `sigreg_050`
  0.9738 → 0.9118, `sigreg_006` 0.9691 → 0.9142 — so enforced isotropy does bring the
  discriminative and marginal directions closer, but nowhere near collapse: at 0.91 the two
  definitions still disagree on 91% of the available angle. The algebra predicted this
  (`Σ_within = σ²I − Σ_between` is not isotropic) and the measurement agrees. **The cross-check
  survives.** The Marchenko–Pastur null becomes better justified, since its isotropy assumption
  would now be enforced rather than hoped for.
- **D12 stays open.** A SIGReg arm would carry a distributional constraint the MAE and contrastive
  arms lack, which is a real asymmetry in a cross-objective comparison. The counter-argument is
  equally real: entanglement *surviving* enforced isotropy is stronger evidence it is in the data
  than entanglement in an unconstrained representation. This is a framing decision, flagged here
  and not settled.
- **The 1C checkpoint rule must not change.** Question 2 does not support it. Selecting on lowest
  loss still costs AUC under SIGReg.

---

## What this does not settle

- **3,000 steps is not 50,000.** All three arms were still moving; `sigreg_050`'s rank was still
  climbing (29.9 → 34.2 over the last 500 steps) and its penalty still falling.
- **One seed, one feature.** "Separated" means separated on bootstrap intervals over the test set,
  not across training runs. No repeated seeds.
- **Question 2 is underpowered by design.** n=6 per arm cannot resolve |ρ| < 0.886. A real answer
  needs either many more checkpoints or the paper's across-run design.
- **Everything is `smoke: true`.** The effect floor is open; the single-feature AUC is not the
  probing battery, and no number here is a ladder verdict.
- **The attachment point was chosen, not tested.** SIGReg on the final block post-norm, or on
  per-token rather than pooled embeddings, might behave differently. Those are separate arms.
