# H4 — proposed schedule. A proposal, not a merge.

Nothing in `configs/pretrain.yaml` has been edited. This is the recommendation the brief asked for,
with its provenance stated and its consequences traced.

## Provenance: the reference recipe adapted, not a trace picked here

Every proposed number below is derived from I-JEPA (Assran et al. 2023) by a stated rule. **None is
read off an H2 trace.** The H2 arms are cited only as evidence that a change of this *kind* moves the
diagnostic — they did not choose the values. This matters because the brief's trap is real: the
highest-erank arm (`linear`, 12.50) is the arm being **rejected**, on its loss.

| quantity | now | proposed | rule |
|---|---|---|---|
| peak LR | 1e-3 | **1.25e-4** | √-scaling of the reference peak: 1e-3 × √(32/2048) |
| warmup | 100 steps (0.20%) | **1,250 steps (2.50%)** | the reference's *relative* warmup, 15 of 600 epochs |
| decay | none | **cosine to 1.25e-7** | the reference decays peak → peak/1000; both endpoints scaled by the same √ factor |
| weight decay | 0.04 constant | 0.04 constant *(unchanged — see below)* | — |
| EMA, steps, batch | — | unchanged | changing `steps` moves every momentum; batch is a throughput decision, not a schedule one |

**Why square-root and not linear.** Linear scaling is derived for SGD with momentum (Goyal et al.
2017), where the update is proportional to the gradient. AdamW normalises by the gradient's second
moment, so its step size does not scale with gradient magnitude the same way, and √-scaling is the
conventional choice for Adam-family optimisers. This is the reason to prefer 1.25e-4 over 1.563e-5 —
**not** that `sqrt` had a better trace than `linear`. That the measurement agrees is corroboration,
not the argument.

**Why the decay floor scales too.** The reference's 1e-6 endpoint is as much a batch-2048 quantity as
its 1e-3 peak; scaling only the peak would compress the decay range from 1000× to 125× and silently
change the recipe's shape. Scaling both keeps it. Stated because it is a judgement call: keeping the
published 1e-6 floor is the defensible alternative, and the difference between the two is confined to
the last few per cent of the run.

**Two smaller fidelity notes, for completeness.** The reference warms up *from 1e-4*, i.e. from
peak/10; ours ramps from peak/100 (`min(1, (step+1)/100) × lr` is 1e-5 at step 0). Under √-scaling both
endpoints move by the same factor, so the shape is preserved and no separate decision is needed. And
the reference's EMA rises *linearly* to 1.0 where `ema_momentum` uses a cosine — a pre-existing
divergence, out of scope here, and provably inert over any short window (H1: four parts in ten million
across 500 steps).

**Why the weight-decay ramp is *not* proposed.** The reference ramps 0.04 → 0.4, so there is a case
from authority. Against it: H2's `wd_ramp` arm was indistinguishable from baseline (erank 4.10 vs
3.75, floor crossed at the identical step 125), and 500 steps is far too short for a regularisation
schedule to show its effect — so the measurement neither supports nor refutes it. Bundling an
unmeasured change with two measured ones would make the result unattributable. **Recommendation:
leave it constant for the resolving run, and adopt it separately if at all.** This is the one place
the proposal deliberately departs from the reference, and says so.

## Cost: a small code change, not a non-trivial one

The brief asked for this to be reported as a finding if it were awkward. It is not. The schedule lives
entirely in three lines, `objectives/jepa.py:269-271`:

```python
lr = cfg.lr * min(1.0, (step + 1) / max(cfg.warmup_steps, 1))
```

The change is two new `JepaConfig` fields (`lr_final`, defaulting to `0.0` = today's behaviour) and
replacing those lines with warmup-then-cosine — roughly eight lines, one site, no new module, no
touch to the optimiser, the loader, the EMA or the checkpointer.

**Resume is unaffected**, and this is worth stating explicitly because it is the kind of thing that
breaks quietly: the schedule stays a pure function of `(step, cfg.steps)`, exactly as the warmup is
today, so a resumed run recomputes the same LR at the same step. `train_jepa`'s `stop_after` docstring
already records that invariant. No checkpoint field is needed.

Landing it inert — `lr_final = 0.0` meaning "warmup only" — means the config hash of an unchanged run
does not move, so the H2 arms and the F-series smokes stay comparable to anything run before the edit.

## Consequences, traced

**1. The provenance machinery holds.** H1 measured it: editing `lr`, `warmup_steps` or `weight_decay`
each moves `config_hash`, and `collapse_floor` is hashed alongside. A schedule change cannot be worn
by an old result.

**2. The G5 floor becomes empirically ungrounded, and that is the real cost.**
`CollapseFloorFreeze.derived_from` cites two traces — the pilot (22.6 → 10.2, held 10.2–10.6, AUC
0.900–0.905) and the F smoke (22.6 → 4.1 by step 175) — and **both were taken at lr = 1e-3 with no
decay**. The 5.0 threshold was "half the erank observed under this recipe". Under a new recipe that
sentence no longer parses. Two honest options:

- **Re-derive.** The resolving run below produces exactly the trace needed; freeze a new
  `CollapseFloorFreeze` from it, with `derived_from` citing the new recipe. **Recommended** — and it
  is nearly free, because the resolving run has to happen anyway.
- **Carry it across explicitly**, rewriting `derived_from` to say the threshold is inherited from a
  different recipe and why that is acceptable. Weaker, but honest.

What must not happen is the floor keeping its current `derived_from` text under a changed schedule.
That would be a stamped provenance claim that is no longer true.

**3. The β sweep gets *stronger*, not weaker.** Every β arm runs one recipe, so the sweep stays
internally controlled. And the β = 0 arm's standing as the published I-JEPA control **improves**: it
currently differs from the reference in β *and* in three schedule respects; under this proposal it
differs in β and in the weight-decay ramp alone.

**4. The pilot comparison is what genuinely degrades.** The pilot's AUC 0.905 was measured under the
current recipe on 10k stamps. After a schedule change it stops being a same-recipe reference and
becomes a different-recipe, different-corpus one. This is a real loss and should be recorded as such
rather than glossed: the pilot remains the *existence* proof that the premise works, not a
like-for-like baseline for the headline run.

## The decision record this needs

Per the brief, a D-series entry in `DECISIONS.md`, not a config tweak. It must carry: the reference's
published schedule; the √-scaling argument and why not linear; the relative-warmup calculation; the
decay-endpoint judgement; the explicit *non*-adoption of the WD ramp with H2's null result behind it;
H2's dose–response table as the evidence that the schedule is causal; and what became of the G5 floor.
Drafted only once the resolving run has reported — the D-entry should record a decision taken on
3,000-step evidence, not on 500.

## Gate before any of this lands

**The resolving run first: `baseline` vs this proposal, 3,000 steps, same seed, ~1.6 h for the pair.**
Two reasons, both from H3's limits. The absolute rank level is not set by the LR alone — the pilot held
10.2–10.6 at the *same* 1e-3 — so the proposal's rank at 500 steps does not predict its rank at
50,000. And `sqrt` has not yet reached the baseline's transient minimum of 0.0476; its flat 0.0757
needs to be shown either resuming its fall or genuinely plateaued. If the proposal's loss is still
flat at 3,000 steps while the baseline is still rising, the case is closed. If both are falling, probe
the two frozen checkpoints and decide on AUC — which is the actual objective, and the thing 500 steps
of effective rank cannot stand in for.
