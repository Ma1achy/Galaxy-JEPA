# Spec — callbacks (the training-loop hook surface)

*Status: design proposal for sign-off. Expands `docs/architecture.md` → "Package layout"
/ "Control-flow ownership". Built in `src/galaxy_jepa/callbacks/` (next phase). British
English.*

The split (Prism's executor- vs lifecycle-owned ownership):

- **Executor-owned (run config):** what the loop needs *before/during* `loss.backward()`
  — `seed`, EMA schedule, masking ratio, grad clip. These are config, not callbacks.
- **Lifecycle-owned (callbacks):** behaviour *inserted around* the loop — collapse
  monitor, sky-fraction logger, EMA updater, checkpointer. These are callbacks.

The encoder, objective, data pipeline, and probe stay independently importable and
testable with no orchestration present.

---

## 1. The hook surface — **decided: granular**

| Hook | Fires | Load-bearing? |
|---|---|---|
| `on_train_start` | once, before the first step | setup |
| `on_batch_start` | each step, before forward | — |
| `on_before_backward` | after loss, before `backward()` | — |
| `on_after_backward` | after `backward()`, before optimiser step | grad clip / inspection |
| `on_step_end` | after the optimiser step | **yes** (see §2) |
| `on_epoch_end` | end of a data epoch | **not** load-bearing (see §2) |
| `on_train_end` | once, after the final step | teardown / final checkpoint |

---

## 2. Step-based, not epoch-based

Large-corpus SSL pretraining runs by **iteration count**, not epochs — the unlabelled
SDSS set is large and a single "epoch" is huge. So **`on_step_end` is the load-bearing
hook**: validation, checkpointing, and the collapse monitor all key off **steps**.
`on_epoch_end` is retained for convenience but **must not** be load-bearing; nothing
required for correctness keys off it.

---

## 3. Field-availability contract

What exists at each hook (a callback reading an unavailable field is a loud error, not a
silent `None`):

| Field | `train_start` | `batch_start` | `before_backward` | `after_backward` | `step_end` | `epoch_end` | `train_end` |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `step`, `config`, `encoder` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `batch` | | ✓ | ✓ | ✓ | ✓ | | |
| `loss` | | | ✓ | ✓ | ✓ | | |
| `grads` | | | | ✓ | ✓ | | |
| `optimiser_stepped` | | | | | ✓ | ✓ | ✓ |
| `epoch_metrics` | | | | | | ✓ | ✓ |

---

## 4. Aborting a run — **recommendation: yes, via `StopRun`**

A callback may halt the run by raising a documented `StopRun` exception — the collapse
monitor raises it on divergence (rather than letting a doomed multi-hour run continue).
This is fail-loud: the run stops with a named reason, checkpoints what it can, and the
reason is recorded. (Maps to Tier-3 `T3.collapse-monitor` in `docs/spec/validation.md`.)

---

## 5. The named Paper-1 callbacks

| Callback | Hook(s) | Role |
|---|---|---|
| EMA updater | `on_step_end` | update the EMA target encoder per the schedule |
| Collapse monitor | `on_step_end` | track representation variance / rank / std; raise `StopRun` on collapse |
| Sky-fraction logger | `on_step_end` | log the realised sky-fraction of sampled target tokens vs β (`docs/masking.md` §7) |
| Checkpointer | `on_step_end`, `on_train_end` | write stamped checkpoints at step intervals + a final frozen-encoder export |

---

## 6. Forks

| Fork | Options | Recommendation | Status |
|---|---|---|---|
| Hook surface | minimal lifecycle / granular | **granular** | **decided** |
| Run-abort mechanism | `StopRun` exception / return-sentinel / flag | **`StopRun` exception** | proposed (recommendation stands) |
| Checkpoint cadence | every N steps / time-based | every N steps (config-driven) | open (minor) |
