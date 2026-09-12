# H2/H3 — the schedule *is* a cause of the rank fall. Measured, monotone, replicated.

Six arms, 500 steps each, on the real 415.8 GB cache. Every arm shares the same initial weights
(`harness.seed_init`), the same data order (`ResumableShuffle`, one seed), the same per-step mask
seeds, the same monitor batch and the same `steps=50000` so the EMA ramp is identical. Only the
`(lr, weight_decay)` sequence varies. Full read-out: `artifacts/out/h3_readout.txt`.

## The arms

| arm | peak LR | warmup | erank@0 | @100 | @175 | @300 | @end | min | loss@500 | deepest loss | trend, final 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1.000e-3 | 100 | 21.89 | 5.32 | 4.84 | 3.94 | **3.75** | 3.73 | 0.1878 | 0.0476 @178 | **+0.063 RISING** |
| linear | 1.563e-5 | 100 | 21.94 | 18.85 | 17.14 | 14.06 | **12.50** | 12.50 | 0.2026 | 0.1945 @486 | −0.043 descending |
| sqrt | 1.250e-4 | 100 | 21.94 | 9.66 | 9.46 | 7.44 | **7.19** | 6.88 | **0.0757** | 0.0713 @479 | −0.001 flat |
| cosine | 1.000e-3 | 100 | 21.89 | 5.32 | 4.96 | 4.62 | **5.87** | 4.26 | 0.1242 | 0.0471 @178 | +0.024 RISING |
| warmup1250 | 1.000e-3 | 1250 | 21.94 | 12.13 | 10.00 | 7.12 | **7.10** | 6.27 | 0.0941 | 0.0832 @362 | +0.008 RISING |
| wd_ramp | 1.000e-3 | 100 | 21.89 | 5.32 | 4.95 | 3.74 | **4.10** | 3.74 | 0.1514 | 0.0472 @178 | +0.034 RISING |

`would_halt` is `no` for every arm. That is structural, not a result: H1 established that inside 500
steps only the hard floor (erank < 2.0) can fire, because the soft floor's grace is 5,000 steps. It
confirms nothing collapsed to a single direction and says nothing about the criterion.

## The comparison is provably controlled

`baseline` and `cosine` share an identical schedule through step 100, and their effective ranks are
**bit-identical** there — 21.8913, 10.1448, 8.8392, 7.5701, 5.3205 at steps 0/25/50/75/100 — and
diverge only from step 125, when the cosine begins to act. Same init, same data, same masks, same
EMA. **Any difference between arms is the schedule**, not run-to-run noise.

## Five findings

**1. Monotone dose–response over a 64× LR range.** Mean LR across steps 0–175 against erank@175:

| arm | mean early LR | erank@175 |
|---|---|---|
| linear | 1.08e-5 | 17.14 |
| warmup1250 | 7.08e-5 | 10.00 |
| sqrt | 8.66e-5 | 9.46 |
| cosine | 6.76e-4 | 4.96 |
| baseline | 6.93e-4 | 4.84 |

Strictly monotone. This is as clear a causal signal as a controlled 500-step experiment can give.

**2. Replicated by two independent mechanisms.** `warmup1250` keeps the **same 1e-3 peak** and only
reaches it slowly; `sqrt` lowers the peak. Their mean early LRs are close (7.08e-5, 8.66e-5) and
their ranks are close (10.00, 9.46). So what matters is the **LR magnitude early**, not how it was
produced — which is a stronger claim than either arm alone supports.

**3. Three-quarters of the fall happens inside the 100-step warmup.** Baseline falls 21.89 → 5.32
over steps 0–100 while the LR ramps 1e-5 → 1e-3, then only 5.32 → 3.75 over the remaining 400 steps
at constant peak. The fall is not caused by *sitting* at the peak; it happens while *reaching* it.

**4. Decay does not prevent the fall — it partially recovers rank afterwards.** Exactly as H1's
arithmetic predicted: the compressed cosine is still at 91.6% of peak at step 175, so `cosine` tracks
`baseline` through the fall (4.96 vs 4.84). Then as its LR decays, rank **climbs back** from a
minimum of 4.26 to 5.87, while `baseline` continues down to 3.75. **This isolates peak from decay:
the early LR causes it, decay mitigates it.** It also shows the rank is not a one-way collapse — it
responds to the learning rate in both directions.

**5. The weight-decay ramp does essentially nothing.** `wd_ramp` reaches wd 0.383 and stays within
noise of baseline throughout (4.95 vs 4.84 at step 175; 4.10 vs 3.75 at the end). Not the cause.

## The trap, guarded

The brief warns against adopting the best-erank arm. **`linear` has the highest rank (12.50) and by
far the worst loss** — deepest 0.1945, **4.1× the baseline's**, still descending steeply at step 500.
It is the arm that holds rank by not having started. Its erank is not comparable and is reported as
such.

Read alongside the loss, two framings disagree and both are reported:

- **On deepest loss ever reached**, `baseline` wins (0.0476 vs `sqrt`'s 0.0713, 1.5×). By this
  measure the scaled arms have done less.
- **On loss at equal step 500**, `sqrt` wins on *both* axes — 0.0757 against 0.1878 (2.5× lower) and
  erank 7.19 against 3.75 (1.9× higher).

The second framing is the fairer one, because the baseline's minimum is a **transient it immediately
leaves**: it bottoms at step 178 and then climbs +0.063 over the final 100 steps while its rank falls
monotonically. A loss that dives then climbs as rank collapses has not "got further". But this is
read as *not settled*, not as a win — see the limits.

## Honest limits

- **500 steps.** The pilot's reference (erank 10.2–10.6, frozen-probe AUC 0.905) was at **6,000**.
  Whether `sqrt`'s flat 0.076 resumes falling or has plateaued is not knowable at this length.
- **A puzzle that must not be papered over.** The pilot ran the **baseline recipe** (lr 1e-3) on
  10,000 stamps for 6,000 steps and held erank 10.2–10.6 — far above the 3.75 the same recipe gives
  here. Same learning rate, very different rank. The difference is the corpus: 10k stamps seen ~19×
  each, against 827k seen once. So **the absolute rank level is not set by the LR alone.** Within
  this experiment the corpus is held constant across arms, so the LR effect is clean; the pilot
  comparison says the schedule is *a* cause, not necessarily the whole one.
- **Effective rank is a diagnostic, not the objective.** No arm was probed. The thing that matters
  is frozen-probe AUC, and this experiment does not measure it.

## Verdict

**Yes — the schedule is a demonstrated cause, and the recipe should be fixed before spending 12 h on
the current one.** The causal question is *not* ambiguous: monotone over 64×, replicated by two
mechanisms, mechanistically isolated (early LR causes, decay mitigates, weight decay irrelevant), on
a comparison proven controlled to bit-identity.

**Which** schedule is not settled at 500 steps, and must not be chosen off these traces.

## What would resolve it

Budget at G3's production rate of **1.067 steps/s** (checkpointing live). The six H2 arms themselves
ran at 1.01–1.51 steps/s — the baseline was the slow one at 1.006 because it shared the machine with
other load; the five later arms averaged 1.44 — so these are upper bounds.

| length | per arm | pair | note |
|---|---|---|---|
| 3,000 steps | 47 min | **1.6 h** | 17× past the baseline's turnover at step 178; enough to see whether the scaled arm's loss resumes falling or has genuinely plateaued |
| 6,000 steps | 94 min | 3.1 h | matches the pilot's own reference length — the only trace tied to AUC 0.905 |

Recommended: **3,000 steps for the pair** (baseline against the proposal), going to 6,000 only if the
loss is still flat and the choice still open. Better still, probe both frozen checkpoints — the
objective is AUC, not rank, and 500 steps of erank cannot stand in for it.
