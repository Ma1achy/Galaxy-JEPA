"""Objectives (placeholder).

Implements ``docs/architecture.md`` "Package layout": JEPA (masking + EMA target +
predictor + latent-MSE), MAE, and contrastive are each a config-driven *procedure
that produces a frozen* :class:`galaxy_jepa.core.encoder.Encoder`. The training
machinery lives here, never on the encoder. Each objective writes a
provenance-stamped checkpoint that ``probing/`` reloads frozen. See
``docs/spec/objectives.md``.

No implementation yet — this phase builds only ``core/`` (the Track-A specs gate
the objectives/encoders/masking/probing/data code).
"""
