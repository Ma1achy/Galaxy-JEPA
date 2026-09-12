# H1 — the configured schedule, verified from the code

Read out of the source rather than from the brief or from memory. Every claim below has a file and
line behind it.

## What the pretrain path actually does

| quantity | value | where |
|---|---|---|
| peak LR | **1e-3** | `configs/pretrain.yaml` `objective.lr` |
| batch size | **32** | `objective.batch_size` |
| warmup | **100 steps**, linear | `objectives/jepa.py:269` |
| LR decay | **none, anywhere** | see below |
| weight decay | **0.04, constant** | `objective.weight_decay` → `jepa.py:238`, set once at AdamW construction and never touched |
| EMA | **0.996 → 1.0, cosine over `steps`** | `ema_momentum`, `jepa.py:180-186` |
| steps | 50,000 | `objective.steps` |

**The only line in the package that ever mutates a learning rate** is `objectives/jepa.py:269-271`:

```python
lr = cfg.lr * min(1.0, (step + 1) / max(cfg.warmup_steps, 1))
for group in opt.param_groups:
    group["lr"] = lr
```

`min(1.0, …)` clamps at the peak. After step 100 this is a constant 1e-3 for the remaining 49,900
steps. Grepping the whole package for `lr_scheduler`, `CosineAnnealing`, `LambdaLR`, `OneCycle` and
`get_lr` returns **nothing** in the training path. The only other `lr=` sites are `probing/mlp.py`
(a different path) and `harness.calibrate:1016` (a throughput timer, not training). So: **confirmed,
warmup only.**

**Weight decay is never ramped.** It is passed to `AdamW(...)` once, at `jepa.py:236-239`, and no
code writes `group["weight_decay"]`. Confirmed constant at 0.04.

## The scaled equivalents

I-JEPA's published peak is **1e-3 at batch 2048**. At the configured batch 32:

| scaling | equivalent peak LR | configured LR is above it by |
|---|---|---|
| linear (`lr × B/B_ref`) | **1.563e-5** | **64.0×** |
| square-root (`lr × √(B/B_ref)`) | **1.250e-4** | **8.00×** |

At batch 64 — F2's measured-fastest batch, and a plausible change — linear gives 3.125e-5 (32.0×)
and sqrt gives 1.768e-4 (5.66×).

**The brief's figures are correct.** Reproduced independently here; no correction needed.

## Three divergences from the reference recipe, not one

G4 named the first two. Verifying the arithmetic turned up a third that matters more for the
hypothesis under test.

1. **Peak LR is 8–64× above the scaled equivalent** and, with no decay, stays there.
2. **No cosine decay to 1e-6, and no 0.04 → 0.4 weight-decay ramp.**
3. **The warmup is 12.5× shorter in relative terms.** I-JEPA warms up over 15 of 600 epochs =
   **2.50%** of its schedule (9,384 steps at batch 2048). Ours is 100 of 50,000 = **0.20%**. This is
   the divergence that lands *inside* the window where the smoke's rank fell (steps 25–175), which
   is why H2 carries a `warmup1250` arm — the reference's relative warmup — alongside the LR arms.

## Two schedule facts that constrain what a 500-step test can show

Computed from the schedules themselves, so these are arithmetic, not inference.

**The real cosine decay is inert in a 500-step window.** Decaying 1e-3 → 1e-6 over 50,000 steps:

| step | cosine factor | LR |
|---|---|---|
| 100 | 1.000000 | 1.000e-3 |
| 175 | 0.999994 | 9.99994e-4 |
| 500 | 0.999841 | 9.99842e-4 |

Still **99.98% of peak** at step 500. So no 500-step run can test the *actual* reference decay; a
decay arm must be a **compressed proxy** over the window, and must be labelled one. Even compressed,
that proxy is still at **91.6% of peak at step 175** — where the smoke's erank had already reached
4.1. Expect the decay arm to say little about the fall itself, and to speak only to whether
annealing later *recovers* rank.

**The EMA ramp cannot confound the arms.** Over a 500-step window of a 50,000-step schedule the
momentum moves from 0.99600000 to 0.99600099 — four parts in ten million. Provided every arm keeps
`steps=50000` (H2 does), the EMA schedule is effectively identical across arms. This is also G1's
point from the other side: changing `steps` would move every later momentum, so `steps` is exactly
the wrong thing to vary in a schedule experiment.

## H4 precondition, checked ahead of the result

The brief asks to confirm that a recipe change is handled by the provenance machinery. Measured:

| edit | `config_hash` | moved? |
|---|---|---|
| *(baseline)* | `157903bd5180788b…` | — |
| `objective.lr` → 1.25e-4 | `01c507d02f611990…` | yes |
| `objective.warmup_steps` → 1250 | `1ad25377c18ceea7…` | yes |
| `objective.weight_decay` → 0.4 | `9e040b48d6098448…` | yes |
| `collapse_floor.min_effective_rank` → 3.0 | `ce8656b572be229f…` | yes |

So the mechanics hold: any schedule edit restamps the run, the G5 floor is hashed alongside it, and
editing the floor itself restamps too. `CollapseFloorFreeze` carries no `content_hash` and correctly
does not need one — unlike `NormalisationFreeze`, whose constants must match a separate on-disk fit,
the floor's values *are* the artefact and are hashed directly.

**The real exposure is not the hash.** `CollapseFloorFreeze.derived_from` reads:

> Pilot run … effective rank 22.6 → 10.2 by step 100 and held 10.2–10.6 to the end; frozen-probe
> AUC 0.900–0.905. Brief F smoke (300 steps, 827k corpus): 22.6 → 4.1 by step 175, then flat …

Both of those traces were taken **at lr = 1e-3 with no decay**. A recipe change therefore leaves the
floor mechanically consistent and **empirically ungrounded**: the 5.0 threshold would no longer be
"half an erank observed under this recipe". Any H4 proposal has to say what happens to the floor —
re-derive it under the new recipe as a fresh freeze, or state explicitly that the old grounding is
being carried across and why. That is a stronger requirement than moving a hash.

## What `would_halt` can and cannot say at 500 steps

H2 asks for `would_halt` under the frozen G5 criterion per arm. Bounded first, because the answer is
structural rather than empirical:

| scenario | fires inside 500 steps? |
|---|---|
| total collapse, erank 1.5 | yes, at step 100 |
| the smoke's plateau, erank 4.1 | **never** |
| erank 2.5 | **never** |

The soft floor of 5.0 applies only from step **5,000** (10% of 50,000). So inside a 500-step window
only the **hard floor** — erank < 2.0 for three consecutive readings after step 100 — can fire.
Anything in the 2.0–5.0 band, the smoke's own regime included, reports `False`.

**So `would_halt = False` across every arm is expected and carries almost no information at this run
length.** It is reported because the brief asks for it, and it confirms no arm collapsed to a single
direction — but it must not be read as "every arm passed the criterion". The criterion has not been
given the chance to speak.
