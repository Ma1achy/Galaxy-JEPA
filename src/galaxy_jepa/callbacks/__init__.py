"""Callbacks (placeholder).

Implements ``docs/architecture.md`` "Package layout" / "Control-flow ownership":
behaviour *inserted around* the training loop — collapse monitor, sky-fraction
logger, EMA updater, checkpointer — lives here (lifecycle-owned), distinct from the
executor-owned config (seed, EMA schedule, masking ratio, grad clip). The hook
surface is specified in ``docs/spec/callbacks.md``.

No implementation yet — this phase builds only ``core/``.
"""
