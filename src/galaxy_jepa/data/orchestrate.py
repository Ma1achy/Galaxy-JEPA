"""Split orchestration — the deferred machinery, built small on the guards.

``docs/spec/splits.md`` ships the structural *guards* (``data/splits.py``); this is the
orchestration that sits on top of them: it resolves the two corpora (exercising the
cross-corpus dedup guard), assigns the probe corpus to a deterministic three-way split,
and carves a fixed collapse-monitor slice out of the pretraining corpus. It is the
guards' first real outing — the dedup post-condition and the seed-reproducible assignment
are **run here, not bypassed** "because it's only the slice".

What is deliberately *not* here (deferred to corpus scale, per ``splits.md`` §4):
**stratification**. The slice splits uniformly by the hashed assignment coordinate; the
stratified draw (hard features × large-extended × uncertainty balance) lands with the
full pull. The uncertainty-geometry firewall is also not applied here — it composes
*inside* this split (``splits.md`` §3) and is exercised at fit time by the probe.

Pure stdlib + ``manifest`` (no network, no ``data`` extra), so the orchestration runs in
the fast/integration gate on fixture id-sets.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable
from pathlib import Path

from galaxy_jepa.data.manifest import manifest_hash
from galaxy_jepa.data.splits import (
    assert_no_cross_corpus_leak,
    assignment_unit,
    exclude_probe_from_pretrain,
    to_object_ids,
)

__all__ = [
    "ProbeSplit",
    "PretrainSplit",
    "resolve_corpora",
    "assign_three_way",
    "split_pretrain",
    "write_split_plan",
]

#: Namespaces for the deterministic assignment (``splits.md`` §5) — kept distinct so the
#: probe split and the monitor slice are not accidentally shifted copies of each other.
_PROBE_SALT = "probe"
_MONITOR_SALT = "pretrain-monitor"


@dataclasses.dataclass(frozen=True)
class ProbeSplit:
    """The probing corpus three-way split (``splits.md`` §1). Disjoint and exhaustive."""

    train: frozenset[int]
    val: frozenset[int]
    test: frozenset[int]

    def __post_init__(self) -> None:
        if not (self.train.isdisjoint(self.val) and self.train.isdisjoint(self.test)):
            raise ValueError("probe split sets overlap")
        if not self.val.isdisjoint(self.test):
            raise ValueError("probe split sets overlap")

    @property
    def total(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)


@dataclasses.dataclass(frozen=True)
class PretrainSplit:
    """The pretraining corpus, carved into the training set and a fixed monitor slice.

    ``monitor`` is the held-out slice the collapse monitor reads (``splits.md`` §1,
    decided: a *fixed* slice so the collapse/loss trend is comparable across checkpoints).
    It is disjoint from the probe corpus by construction — it is drawn from the *deduped*
    pretraining set produced by :func:`resolve_corpora`.
    """

    train: frozenset[int]
    monitor: frozenset[int]

    def __post_init__(self) -> None:
        if not self.train.isdisjoint(self.monitor):
            raise ValueError("pretrain train/monitor sets overlap")

    @property
    def total(self) -> int:
        return len(self.train) + len(self.monitor)


def resolve_corpora(pretrain_ids: Iterable[object], probe_ids: Iterable[object]) -> frozenset[int]:
    """Return the leak-free pretraining ID set, asserting the dedup post-condition.

    The collision-resolution operation (``exclude_probe_from_pretrain``) followed
    immediately by its merge-blocking post-condition (``assert_no_cross_corpus_leak``).
    Calling both here means a probe galaxy *cannot* survive into pretraining — the guard
    is exercised on the real manifests, not just in the unit test.
    """
    deduped = exclude_probe_from_pretrain(pretrain_ids, probe_ids)
    assert_no_cross_corpus_leak(deduped, probe_ids)  # must not raise — proves the dedup held
    return deduped


def assign_three_way(
    probe_ids: Iterable[object],
    *,
    seed: int,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> ProbeSplit:
    """Deterministically assign probe galaxies to train/val/test by the hashed coordinate.

    Each galaxy's home is ``assignment_unit(objID, seed, salt="probe") ∈ [0, 1)`` bucketed
    by the cumulative ratios — reproducible from the seed alone, no stored split file. The
    realised proportions approach ``ratios`` in the large-n limit; small id-sets will not
    hit them exactly (that is expected, not a bug).
    """
    train_hi, val_hi = _cumulative(ratios)
    train: set[int] = set()
    val: set[int] = set()
    test: set[int] = set()
    for oid in to_object_ids(probe_ids):
        u = assignment_unit(oid, seed, salt=_PROBE_SALT)
        if u < train_hi:
            train.add(oid)
        elif u < val_hi:
            val.add(oid)
        else:
            test.add(oid)
    return ProbeSplit(frozenset(train), frozenset(val), frozenset(test))


def split_pretrain(
    pretrain_ids: Iterable[object],
    *,
    seed: int,
    monitor_frac: float = 0.02,
) -> PretrainSplit:
    """Carve a fixed monitor slice (``monitor_frac``) out of the pretraining corpus.

    Uses a distinct salt from the probe split so the monitor is not a shifted copy of it.
    Pass the **deduped** pretraining ids (from :func:`resolve_corpora`) so the monitor is
    guaranteed disjoint from the probe corpus.
    """
    if not 0.0 < monitor_frac < 1.0:
        raise ValueError(f"monitor_frac must be in (0, 1); got {monitor_frac}")
    train: set[int] = set()
    monitor: set[int] = set()
    for oid in to_object_ids(pretrain_ids):
        u = assignment_unit(oid, seed, salt=_MONITOR_SALT)
        (monitor if u < monitor_frac else train).add(oid)
    return PretrainSplit(frozenset(train), frozenset(monitor))


def _cumulative(ratios: tuple[float, float, float]) -> tuple[float, float]:
    if len(ratios) != 3 or any(r < 0 for r in ratios):
        raise ValueError(f"ratios must be three non-negative numbers; got {ratios!r}")
    total = sum(ratios)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0; got {ratios!r} summing to {total}")
    return ratios[0], ratios[0] + ratios[1]


def write_split_plan(
    path: str | Path,
    *,
    probe: ProbeSplit,
    pretrain: PretrainSplit,
    seed: int,
    ratios: tuple[float, float, float],
    query: str = "",
) -> str:
    """Write a stamped JSON split plan and return its top-level ``data_snapshot`` hash.

    The plan is reproducible from ``(seed, ratios)`` alone, but persisting it gives a
    provenance trail and a stamped decision (``splits.md`` §5). Each split carries its own
    manifest hash over its id-set, and the top-level hash covers the whole assignment.
    """
    splits = {
        "probe-train": sorted(probe.train),
        "probe-val": sorted(probe.val),
        "probe-test": sorted(probe.test),
        "pretrain-train": sorted(pretrain.train),
        "pretrain-monitor": sorted(pretrain.monitor),
    }
    all_ids = [oid for ids in splits.values() for oid in ids]
    snapshot = manifest_hash(all_ids, f"split-plan|seed={seed}|ratios={ratios}|{query}")
    plan = {
        "data_snapshot": snapshot,
        "seed": seed,
        "ratios": list(ratios),
        "counts": {name: len(ids) for name, ids in splits.items()},
        "hashes": {name: manifest_hash(ids, name) for name, ids in splits.items()},
        "splits": splits,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2) + "\n")
    return snapshot
