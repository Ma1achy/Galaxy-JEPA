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
