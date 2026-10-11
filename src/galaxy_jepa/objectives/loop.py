"""The training loop every objective shares (baselines build, ``artifacts/baselines.md`` §9 item 2).

One loop, so the resume, stop, checkpoint, collapse-monitor and MPS-pool logic that trained M
exists once. An objective (:class:`Objective`) supplies only what differs between methods: its loss
terms, the optimiser over its trainable parameters, its per-step momentum update (JEPA's EMA target,
MoCo's key encoder, a no-op for MAE), whether M's soft rank floor applies to it, and the encoder it
exports. :func:`galaxy_jepa.objectives.jepa.train_jepa` is this loop with a :class:`Jepa`.

**The refactor is an identity, not an approximation.** The body below is ``train_jepa``'s as it
trained M, with the JEPA-specific lines replaced by calls that execute the same operations in the
same order; ``tests/test_objective_loop.py`` holds the verbatim pre-refactor loop and requires equal
losses and bit-identical weights.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from pathlib import Path
from typing import Any, Protocol

import torch

from galaxy_jepa.callbacks.checkpoint import TrainCheckpointer
from galaxy_jepa.callbacks.collapse import CollapseFloorFreeze, CollapseMonitor
from galaxy_jepa.models.vit import VisionTransformer, save_encoder

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class LossParts:
    """A step's loss and the terms behind it — the thing an arm's trace records.

    ``total`` is what backpropagates. ``sigreg`` is ``None``, not zero, when the penalty was
    not computed: a trace should be able to say "off" rather than imply a measured zero.
    For an objective without a regulariser, ``prediction`` is its pretext loss and equals ``total``.
    """

    total: torch.Tensor
    prediction: torch.Tensor
    sigreg: torch.Tensor | None


class ScheduleConfig(Protocol):
    """The config fields the loop reads. Every objective's config carries them, so the LR shape,
    the step budget and the per-step seeding are one definition across methods (F7: M's warmup and
    cosine shape for every arm)."""

    steps: int
    batch_size: int
    lr: float
    lr_final: float | None
    warmup_steps: int
    monitor_every: int
    seed: int


class Objective(Protocol):
    """What the loop needs from a method. ``name`` labels the progress bar (``jepa`` for M, so M's
    logs read as they always have)."""

    name: str
    config: Any
    encoder: VisionTransformer

    def to(self, device: str) -> Any: ...

    def loss_parts(self, batch: dict[str, Any], *, seed: int | None = None) -> LossParts: ...

    def make_optimiser(self) -> torch.optim.Optimizer: ...

    def momentum_update(self, step: int) -> None: ...

    def checkpoint_modules(self) -> dict[str, Any]: ...

    @property
    def soft_rank_floor(self) -> bool: ...

    def soft_floor_note(self, floor: CollapseFloorFreeze) -> str: ...


def learning_rate(step: int, cfg: ScheduleConfig) -> float:
    """The learning rate at ``step``: linear warmup, then optional cosine decay (D17).

    A pure function of ``(step, cfg)`` exactly as :func:`ema_momentum` is, which is what lets a
    resume recompute the same value from the checkpointed step without a stateful scheduler to
    restore. ``cfg.lr_final is None`` reproduces the pre-D17 warmup-only line byte for byte, so
    an unchanged config is an unchanged run.
    """
    lr = cfg.lr * min(1.0, (step + 1) / max(cfg.warmup_steps, 1))
    if cfg.lr_final is None or step + 1 <= cfg.warmup_steps:
        return lr
    t = (step + 1 - cfg.warmup_steps) / max(cfg.steps - cfg.warmup_steps, 1)
    return cfg.lr_final + (cfg.lr - cfg.lr_final) * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))


@dataclasses.dataclass
class TrainResult:
    """What the pretrain returns: where the checkpoint landed + the traces to read."""

    checkpoint: Path | None
    losses: list[float]
    collapse_trace: dict[str, list[float]]
    halted: bool
    #: The two terms behind ``losses``, index-aligned with it. Diagnostics, not training state,
    #: so they are **not** in the checkpoint: on a resumed run the restored steps come back as
    #: NaN rather than being dropped, so the lists stay aligned and the gap is visible instead
    #: of implied. ``sigreg_losses`` is all-NaN when the penalty is off.
    prediction_losses: list[float] = dataclasses.field(default_factory=list)
    sigreg_losses: list[float] = dataclasses.field(default_factory=list)
    #: Steps completed in total, restored ones included — not the same as ``len(losses)`` once a
    #: run has halted, and the thing a resume needs to know it landed where it meant to.
    steps_completed: int = 0


def train_objective(
    obj: Objective,
    loader: Any,
    *,
    device: str = "cpu",
    monitor_batch: dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
    autocast_dtype: torch.dtype | None = None,
    checkpointer: TrainCheckpointer | None = None,
    sampler: Any | None = None,
    collapse_floor: CollapseFloorFreeze | None = None,
    resume: bool = True,
    stop_after: int | None = None,
) -> TrainResult:
    """Run an objective's pretrain loop with the collapse monitor live; export a frozen checkpoint.

    ``loader`` yields batch dicts (from ``StampDataset``); it is cycled until ``config.steps``.
    ``monitor_batch`` is a fixed held-out batch (the ``pretrain-monitor`` slice) the collapse
    monitor reads. Halts on NaN/Inf loss, representation collapse, or the pre-registered
    ``collapse_floor``, recording why.

    ``checkpointer`` writes a resumable mid-run checkpoint every ``checkpointer.every`` steps and,
    with ``resume=True``, continues from the newest committed one. Resuming requires ``sampler``
    to be a :class:`~galaxy_jepa.data.dataset.ResumableShuffle`: the trajectory is a function of
    ``(seed, step)`` only because the data order is, and with a plain shuffling loader a resumed
    run trains on a different sequence from the same step and diverges silently.

    ``stop_after`` ends *this invocation* after that many steps, leaving ``config.steps`` — and so
    the LR and momentum schedules — untouched. It is a deliberate interruption: what a crash looks
    like from the checkpoint's side, and how you stop a multi-day job to free the machine. The
    resume identity proof uses it rather than reimplementing this loop, which would test the wrong
    thing.
    """
    cfg = obj.config
    obj.to(device)
    opt = obj.make_optimiser()
    losses: list[float] = []
    prediction_losses: list[float] = []
    sigreg_losses: list[float] = []
    halted = False
    start_step = 0
    restored: list[dict[str, float]] = []

    if checkpointer is not None and resume:
        from galaxy_jepa.data.dataset import ResumableShuffle

        if checkpointer.latest() is not None and not isinstance(sampler, ResumableShuffle):
            raise ValueError(
                "refusing to resume without a ResumableShuffle sampler. The checkpoint restores "
                "the weights, the optimiser and the RNG, but with a plain shuffling loader the "
                "data order is not recoverable, so the resumed run would train on a different "
                "sequence from the same step. A resume that runs but diverges is worse than none."
            )
        state = checkpointer.restore(objective=obj, optimiser=opt, map_location=device)
        if state is not None:
            assert isinstance(sampler, ResumableShuffle)  # guaranteed by the refusal above
            start_step, losses, restored = state.step, list(state.losses), state.collapse_history
            # the terms were never checkpointed; pad so the indices keep meaning what they say
            prediction_losses = [math.nan] * len(losses)
            sigreg_losses = [math.nan] * len(losses)
            sampler.start = start_step * cfg.batch_size
            logger.info("resuming at step %d of %d", start_step, cfg.steps)

    soft_rank_floor = obj.soft_rank_floor
    if not soft_rank_floor and collapse_floor is not None:
        logger.info(obj.soft_floor_note(collapse_floor))
    monitor = CollapseMonitor(
        floor=collapse_floor,
        total_steps=cfg.steps,
        history=restored,
        soft_rank_floor=soft_rank_floor,
    )
    stopped_early = False
    done = start_step
    data = _cycle(loader)
    bar = _progress(cfg.steps, start=start_step, desc=obj.name)
    for step in bar:
        batch = _to_device(next(data), device)
        lr = learning_rate(step, cfg)
        for group in opt.param_groups:
            group["lr"] = lr

        opt.zero_grad(set_to_none=True)
        if autocast_dtype is not None:
            with torch.autocast(device_type=device.split(":")[0], dtype=autocast_dtype):
                parts = obj.loss_parts(batch, seed=cfg.seed + step)
        else:
            parts = obj.loss_parts(batch, seed=cfg.seed + step)
        loss = parts.total

        if not torch.isfinite(loss):  # T3.no-nan-loss
            logger.error("step %d: non-finite loss (%s) — halting", step, loss.item())
            halted = True
            break
        loss.backward()
        opt.step()
        obj.momentum_update(step)
        loss_val = float(loss.item())
        losses.append(loss_val)
        prediction_losses.append(float(parts.prediction.item()))
        sigreg_losses.append(math.nan if parts.sigreg is None else float(parts.sigreg.item()))
        _set_postfix(bar, loss=loss_val)

        if monitor_batch is not None and step % cfg.monitor_every == 0:
            with torch.no_grad():
                emb = obj.encoder.encode(_to_device(monitor_batch, device)["image"].float())
            signals = monitor.update(step, emb)
            _set_postfix(
                bar,
                loss=loss_val,
                std=signals.std,
                erank=signals.effective_rank,
                cos=signals.mean_cosine,
            )
            logger.info(
                "step %d: loss=%.4f std=%.4f erank=%.1f cos=%.3f",
                step,
                loss_val,
                signals.std,
                signals.effective_rank,
                signals.mean_cosine,
            )
            if monitor.should_halt(signals):  # T3.collapse-monitor
                logger.error("step %d: halting — %s", step, monitor.halt_reason)
                halted = True
                break

        if device.startswith("mps") and step % cfg.monitor_every == 0:
            # Release the MPS allocator's cached-but-unused blocks. Measured on this machine at
            # batch 32: the pool settles at 6.9 GB while only 0.87 GB is live, so ~3.2 GB of an
            # 18 GB machine sits held for nothing; releasing every monitor interval holds it at
            # 3.67 GB. It is numerically inert — the losses are identical step for step — and
            # costs nothing measurable (133 s vs 135 s over 200 steps). Brief I found this the
            # hard way, when a second arm was killed for memory after the first had finished.
            torch.mps.empty_cache()

        if checkpointer is not None and (step + 1) % checkpointer.every == 0:
            checkpointer.save(
                step=step + 1,
                objective=obj,
                optimiser=opt,
                losses=losses,
                collapse_history=monitor.history,
            )
        done = step + 1
        if stop_after is not None and done - start_step >= stop_after:
            logger.info("stopping after %d steps this invocation (at step %d)", stop_after, done)
            stopped_early = True
            break

    if checkpointer is not None and stopped_early:
        # land the checkpoint exactly where the loop stopped, or the resume would repeat work
        checkpointer.save(
            step=done,
            objective=obj,
            optimiser=opt,
            losses=losses,
            collapse_history=monitor.history,
        )
    if checkpointer is not None and not halted and not stopped_early:
        # the last partial interval is worth keeping too; a completed run's final state is the
        # thing a probe re-run would want and re-deriving it costs the whole job
        checkpointer.save(
            step=cfg.steps,
            objective=obj,
            optimiser=opt,
            losses=losses,
            collapse_history=monitor.history,
        )
    checkpoint = None
    if checkpoint_path is not None:
        checkpoint = save_encoder(
            obj.encoder,
            checkpoint_path,
            extra={"steps": len(losses), "halted": halted, "halt_reason": monitor.halt_reason},
        )
    return TrainResult(
        checkpoint,
        losses,
        monitor.trace(),
        halted,
        steps_completed=done,
        prediction_losses=prediction_losses,
        sigreg_losses=sigreg_losses,
    )


def _progress(steps: int, *, start: int = 0, desc: str = "jepa") -> Any:
    """A tqdm progress bar over the step range (loss + collapse metrics + ETA in the postfix).

    tqdm is a core dependency, but fall back to a bare range if it is ever absent so the
    training loop never depends on a display library being importable.
    """
    todo = range(start, steps)
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - tqdm is a declared core dep
        return todo
    # `initial`/`total` so a resumed run's bar and ETA describe the whole job, not the remainder
    return tqdm(todo, desc=desc, unit="step", dynamic_ncols=True, initial=start, total=steps)


def _set_postfix(bar: Any, **fields: float) -> None:
    """Update the bar's live readout, tolerant of the bare-range fallback."""
    set_postfix = getattr(bar, "set_postfix", None)
    if set_postfix is not None:
        set_postfix(fields, refresh=False)


def _cycle(loader: Any) -> Any:
    while True:
        yield from loader


def _to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            # MPS has no float64; the collated scalar fields (petro/pixel_scale) come through
            # as float64 and would fail .to("mps"). Downcast to float32 before the move.
            if v.dtype == torch.float64:
                v = v.to(torch.float32)
            out[k] = v.to(device)
        else:
            out[k] = v
    return out
