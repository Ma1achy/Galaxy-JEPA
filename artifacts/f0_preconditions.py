"""Brief F0 — the preconditions every F-series measurement rests on, checked as a gate.

Imported by the F1 loader benchmark and the F2 model smoke so neither can run against a
half-closed guardrail: a benchmark taken over a recomputed normalisation, or a cache whose
provenance does not match the live freeze, measures something that is not the experiment.

Raises on the first failure. Investigation code: terse, excluded from lint/CI.

    uv run python artifacts/f0_preconditions.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from galaxy_jepa.data.cache import TensorCache, pipeline_hash
from galaxy_jepa.data.transforms import Normalise
from galaxy_jepa.harness import HarnessConfig, _build_pipeline

REPO = Path(__file__).resolve().parent.parent
# F0_CACHE_BASE / F0_NORM_PREFIX point this at a v2 run's cache and freeze (plan A8)
CACHE_BASE = Path(__import__("os").environ.get("F0_CACHE_BASE", "/Volumes/X10 Pro/galaxy-jepa/runs/full/cache"))
EXPECTED_NORM_PREFIX = __import__("os").environ.get("F0_NORM_PREFIX", "75100066b3e0")
# D22's floor and the write-once record that carries it (configs/effect_floor.json)
EXPECTED_FLOOR = 0.7267
EXPECTED_FLOOR_PREFIX = "0c048dc8dd35"


def _load(name: str) -> dict[str, Any]:
    return yaml.safe_load((REPO / "configs" / name).read_text())


def check(verbose: bool = True) -> tuple[HarnessConfig, TensorCache]:
    say = print if verbose else (lambda *a, **k: None)
    cfg = HarnessConfig(**_load("pretrain.yaml")).with_resolved_device()
    freeze = cfg.normalisation
    if freeze is None:
        raise SystemExit("F0: configs/pretrain.yaml carries no normalisation freeze")

    # 1. the freeze loads, is intact, and the pipeline takes its constants FROM the record
    freeze.assert_intact()
    pipeline = _build_pipeline(q=cfg.q, freeze=freeze)
    norm = next(t for t in pipeline.transforms if isinstance(t, Normalise))
    if tuple(norm.mean or ()) != tuple(freeze.mean) or tuple(norm.std or ()) != tuple(freeze.std):
        raise SystemExit("F0: _build_pipeline did not take its constants from the record")
    # 2. and it has no fitting path at all — a missing record must stop a run
    try:
        _build_pipeline(q=cfg.q, freeze=None)
    except ValueError as exc:
        say(f"F0 refit closed        : missing freeze refused — {str(exc).split('.')[0][:60]}…")
    else:
        raise SystemExit("F0: _build_pipeline FITTED without a record — the E5 defect reopened")
    say(f"F0 freeze intact       : {freeze.content_hash[:12]}… q={freeze.stretch_q} "
        f"n={freeze.n_sample} trim={freeze.trim_excluded}")

    # 3. the baked cache names the freeze that made it, and it is the live one
    ph = pipeline_hash(pipeline)
    cache = TensorCache(CACHE_BASE / ph)
    if cache.index.normalisation_hash != freeze.content_hash:
        raise SystemExit(
            f"F0: cache baked under {cache.index.normalisation_hash[:12] or '(unrecorded)'} "
            f"but the live freeze is {freeze.content_hash[:12]}"
        )
    if not freeze.content_hash.startswith(EXPECTED_NORM_PREFIX):
        raise SystemExit(f"F0: freeze is {freeze.content_hash[:12]}, brief expects "
                         f"{EXPECTED_NORM_PREFIX}")
    say(f"F0 cache provenance    : pipeline {ph[:12]}… normalisation "
        f"{cache.index.normalisation_hash[:12]}… n={cache.index.n:,} "
        f"{cache.index.shape} {cache.index.dtype}")

    # 4. BOTH pre-registration freezes are in place and each still bites.
    #
    # This check was written while the effect floor was open, and asserted the opposite: that
    # `headline=True` was REFUSED. Brief O0 froze the floor at 0.7267, so that refusal is gone and
    # the old assertion would fail on a correct config — a guardrail outliving the state it
    # guarded. Inverted rather than deleted: what matters now is that each freeze is present, that
    # the live value equals the stamped one, and that REMOVING either freeze still refuses. A gate
    # that stops biting once its neighbour is satisfied is not a gate.
    # The effect floor is no longer inline: probe.yaml points at the write-once record
    # (configs/effect_floor.json, floor.freeze_effect_floor), so it is checked as scoring reads it —
    # loaded intact (hash AND rule), at D22's value, at the pinned record hash.
    probe_cfg = _load("probe.yaml")
    from galaxy_jepa.probing.config import ProbingConfig
    from galaxy_jepa.probing.floor import load_effect_floor, resolve_effect_floor

    pc = ProbingConfig(**probe_cfg)
    if pc.vote_count_freeze is None:
        raise SystemExit("F0: probe.yaml carries no vote-count freeze")
    if pc.effect_floor_file is None:
        raise SystemExit("F0: probe.yaml names no effect_floor_file (the write-once floor record)")
    floor_path = Path(pc.effect_floor_file)
    floor_path = floor_path if floor_path.is_absolute() else REPO / floor_path
    try:
        record = load_effect_floor(floor_path)
        resolved = resolve_effect_floor(pc.model_copy(update={"effect_floor_file": str(floor_path)}))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"F0: the effect-floor record {floor_path} does not load intact: {exc}")
    if record.value != EXPECTED_FLOOR or resolved.effect_floor != EXPECTED_FLOOR:
        raise SystemExit(f"F0: the effect-floor record says {record.value}, D22 froze {EXPECTED_FLOOR}")
    if not record.content_hash.startswith(EXPECTED_FLOOR_PREFIX):
        raise SystemExit(f"F0: effect-floor record is {record.content_hash[:12]}, expected "
                         f"{EXPECTED_FLOOR_PREFIX}")
    for drop in ("effect_floor_file", "vote_count_freeze"):
        try:
            ProbingConfig(**{**probe_cfg, "headline": True, "smoke": False, drop: None})
        except Exception:  # pydantic ValidationError — the gate bit, which is the point
            pass
        else:
            raise SystemExit(f"F0: headline=True was ACCEPTED with {drop} removed")
    say(f"F0 vote floor frozen   : value={pc.vote_count_freeze.value:g} "
        f"sweep={list(pc.vote_count_freeze.sweep)} min={pc.vote_count_min:g}")
    say(f"F0 effect floor frozen : value={record.value:g} ({record.rule}) record "
        f"{record.content_hash[:12]}… frozen {record.frozen_at} by {record.frozen_by}; "
        f"both gates shut, headline is now loadable")
    return cfg, cache


if __name__ == "__main__":
    check()
    print("F0 PASS — all four preconditions hold")
