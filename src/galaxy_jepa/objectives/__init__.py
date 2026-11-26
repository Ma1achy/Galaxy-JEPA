"""Self-supervised pretraining objectives (JEPA, MoCo, MAE, grouped MAE) and training loop.

Implements ``docs/architecture.md`` "Package layout": each objective is a config-driven
procedure that produces a frozen :class:`galaxy_jepa.models.vit.VisionTransformer`
checkpoint reloaded by ``probing/``. See ``docs/spec/objectives.md``.
"""
