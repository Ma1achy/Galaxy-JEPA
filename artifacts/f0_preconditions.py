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
CACHE_BASE = Path("/Volumes/X10 Pro/galaxy-jepa/runs/full/cache")
EXPECTED_NORM_PREFIX = "75100066b3e0"


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

    # 4. the probing gates are still shut: the effect floor is open, so no headline
    probe_cfg = _load("probe.yaml")
    from galaxy_jepa.probing.config import ProbingConfig

    pc = ProbingConfig(**probe_cfg)
    if pc.vote_count_freeze is None:
        raise SystemExit("F0: probe.yaml carries no vote-count freeze")
    if pc.headline:
        raise SystemExit("F0: probe.yaml claims headline while the effect floor is open")
    try:
        ProbingConfig(**{**probe_cfg, "headline": True})
    except Exception as exc:  # pydantic ValidationError
        first = str(exc).splitlines()
        msg = next((ln.strip() for ln in first if "effect" in ln.lower()), first[-1].strip())
        say(f"F0 headline refused    : {msg[:88]}")
    else:
        raise SystemExit("F0: headline=True was ACCEPTED with the effect floor open")
    say(f"F0 vote floor frozen   : value={pc.vote_count_freeze.value:g} "
        f"sweep={list(pc.vote_count_freeze.sweep)} min={pc.vote_count_min:g}")
    return cfg, cache


if __name__ == "__main__":
    check()
    print("F0 PASS — all four preconditions hold")
