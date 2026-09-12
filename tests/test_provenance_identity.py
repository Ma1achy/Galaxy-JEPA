"""Invariants on what a run's identity *is* (``core.config`` + ``harness`` config split).

A run is determined by ``(config_hash, code_sha, data_snapshot, seed)``. These pin the two
halves of that claim that are easy to break silently:

* a corpus's **location** is not part of the experiment — moving it must not restamp the run;
* a corpus's **backend** is, because MPS / CPU / CUDA differ numerically.

Plus the trap underneath the scheme marker: ``data.cache.pipeline_hash`` is ``config_hash`` of
the fitted pipeline and *names the fp16 cache directory*, so the marker must never reach it.
"""

import pytest

from galaxy_jepa.callbacks.collapse import CollapseFloorFreeze
from galaxy_jepa.core.config import STAMP_SCHEME, config_hash
from galaxy_jepa.data.cache import pipeline_hash
from galaxy_jepa.data.transforms import AsinhStretch, Normalise, Pipeline
from galaxy_jepa.harness import (
    HarnessConfig,
    ObjectiveConfig,
    PathsConfig,
    RuntimeConfig,
    _make_stamp,
)

pytestmark = pytest.mark.invariant


def _cfg(**kwargs) -> HarnessConfig:
    base = {
        "paths": PathsConfig(
            pretrain_dir="data/pretrain", probe_dir="data/probe-40k", out_dir="runs/full"
        ),
        "runtime": RuntimeConfig(device="cpu"),
    }
    return HarnessConfig(**{**base, **kwargs})


def _hash(config: HarnessConfig) -> str:
    return _make_stamp(config, "manifest:fixed").config_hash


def test_moving_a_corpus_does_not_change_the_run_hash():
    """The whole point of the split: the same science on another disk is the same run."""
    here = _cfg()
    on_the_ssd = _cfg(
        paths=PathsConfig(
            pretrain_dir="/Volumes/X10 Pro/galaxy-jepa/raw/pretrain",
            probe_dir="/Volumes/X10 Pro/galaxy-jepa/raw/probe-40k",
            out_dir="/Volumes/X10 Pro/galaxy-jepa/runs/full",
        )
    )
    assert _hash(here) == _hash(on_the_ssd)


def test_the_backend_does_change_the_run_hash():
    """`runtime` is deliberately NOT in NON_DETERMINING — backends differ numerically."""
    assert _hash(_cfg(runtime=RuntimeConfig(device="cpu"))) != _hash(
        _cfg(runtime=RuntimeConfig(device="mps"))
    )


def test_an_unset_device_hashes_as_the_backend_actually_used():
    """`device: null` must resolve before hashing, or MPS and CUDA share one hash."""
    from galaxy_jepa.harness import pick_device

    assert _hash(_cfg(runtime=RuntimeConfig())) == _hash(
        _cfg(runtime=RuntimeConfig(device=pick_device()))
    )
    assert _make_stamp(_cfg(runtime=RuntimeConfig()), "manifest:fixed").device == pick_device()


def test_a_change_to_the_science_still_changes_the_run_hash():
    """The deny-list must not have widened into ignoring the experiment itself."""
    assert _hash(_cfg(objective=ObjectiveConfig(beta=0.0))) != _hash(
        _cfg(objective=ObjectiveConfig(beta=1.0))
    )
    assert _hash(_cfg(seed=0)) != _hash(_cfg(seed=1))


def test_only_paths_are_dropped_from_the_hashed_tree():
    """A deny-list, so a newly added field is hashed by default: a spurious 'different run'
    is recoverable, a false 'same run' is not."""
    cfg = _cfg()
    dumped, determining = cfg.model_dump(mode="json"), cfg.determining_dump()
    assert set(dumped) - set(determining) == {"paths"}
    assert "runtime" in determining


def test_the_stamped_hash_declares_its_scheme():
    stamp = _make_stamp(_cfg(), "manifest:fixed")
    assert stamp.config_hash.startswith(STAMP_SCHEME)


def test_the_cache_key_carries_no_scheme_marker():
    """The parity lock. `pipeline_hash` names the fp16 cache directory on disk; a marker here
    would rename the key and silently force a full re-bake (docs/spec/data.md)."""
    pipeline = Pipeline([AsinhStretch(q=4.0), Normalise(mean=(0.0,), std=(1.0,))])
    key = pipeline_hash(pipeline)
    assert ":" not in key
    assert len(key) == 64 and all(c in "0123456789abcdef" for c in key)
    assert not config_hash(pipeline.to_config()).startswith(STAMP_SCHEME)


class TestASmokeCanNeverBeReadAsAResult:
    """The probing path has carried this marking since D-series; the training path had none.

    A throughput measurement runs the real objective on the real corpus and writes real-looking
    artefacts. Without a marker it is distinguishable from a run only by remembering which is
    which — so the marker is a determining field (it moves the hash, so a smoke's artefacts
    cannot collide with a run's) *and* an escape hatch (so the stamp says so in words).
    """

    def test_the_marking_moves_the_run_hash(self):
        assert _hash(_cfg(smoke=True)) != _hash(_cfg())

    def test_the_marking_is_stamped_as_a_forfeit_not_only_hashed(self):
        hatches = _make_stamp(_cfg(smoke=True), "manifest:fixed").escape_hatches_used
        assert "smoke" in hatches
        assert "smoke" not in _make_stamp(_cfg(), "manifest:fixed").escape_hatches_used


class TestTheStampNamesWhatARunForfeited:
    """A guardrail left off is a choice, and the artefact has to carry it in words.

    `config_hash` already moves — an unset freeze is a different config — but a hash says only
    *that* something differs, never *what was given up*. These are the two forfeits Brief G added:
    a run with no pre-registered collapse floor can only halt on a non-finite embedding, so a dead
    representation costs days before anyone looks; a run with no mid-run checkpoint is betting the
    whole job on not crashing.
    """

    def test_an_unset_collapse_floor_is_recorded_as_forfeited(self):
        assert "collapse_floor_open" in _make_stamp(_cfg(), "m").escape_hatches_used
        floor = CollapseFloorFreeze(
            derived_from="pilot: erank 10.2-10.6 at 6k steps -> AUC 0.905",
            frozen_at="2026-09-12",
            frozen_by="tests",
            rationale="half the one trace tied to a working probe",
        )
        assert (
            "collapse_floor_open"
            not in _make_stamp(_cfg(collapse_floor=floor), "m").escape_hatches_used
        )

    def test_disabling_the_mid_run_checkpoint_is_recorded_as_forfeited(self):
        on = _cfg(objective=ObjectiveConfig(checkpoint_every=1500))
        off = _cfg(objective=ObjectiveConfig(checkpoint_every=0))
        assert "no_mid_run_checkpoint" not in _make_stamp(on, "m").escape_hatches_used
        assert "no_mid_run_checkpoint" in _make_stamp(off, "m").escape_hatches_used

    def test_a_clean_run_forfeits_nothing(self):
        floor = CollapseFloorFreeze(
            derived_from="pilot", frozen_at="2026-09-12", frozen_by="tests", rationale="r"
        )
        assert _make_stamp(_cfg(collapse_floor=floor), "m").escape_hatches_used == []

    def test_a_real_run_is_the_default(self):
        assert _cfg().smoke is False
