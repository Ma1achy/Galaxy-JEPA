"""Split-policy structural guards — leak-prevention as code, not discipline.

Implements the enforceable core of ``docs/spec/splits.md``. The split *policy* (ratios,
stratification, the actual pull) is design; what lives here is the handful of guarantees
that must be **impossible to violate by hand**, each with a loud failure:

1. **Cross-corpus dedup (the leak guard).** A galaxy in the GZ2 *probing* corpus must
   never also sit in the large unlabelled *pretraining* corpus. If it did, the frozen
   encoder would have seen a probe galaxy during pretraining and the D6 decoupling — the
   premise that the representation *transfers* rather than *memorises* — would silently
   break. Keyed on **SDSS ``objID``** (the stable cross-catalogue identity), never on
   ra/dec floats.
2. **Collision resolution.** When a galaxy appears in both pulls, it is assigned to the
   **probing** corpus and removed from pretraining (proposed default — see the fork in
   ``splits.md``). :func:`exclude_probe_from_pretrain` is the operation;
   :func:`assert_no_cross_corpus_leak` is the post-condition that proves it held.
3. **Uncertainty-geometry firewall.** The ambiguous middle (``low < v < high``, default
   0.2–0.8) must never enter the concept-axis *fit* set — only high-consensus extremes
   fit, and the middle is the held-out test. Recovering a vote fraction from an axis
   trained on it is a tautology (scratchpad, "the non-circular protocol"), so the
   firewall is structural, not a convention.

Determinism (:func:`assignment_unit`) is a stable hash of ``(objID, seed)`` so a galaxy's
train/val/test home is reproducible from the seed alone — no stored split file to drift.

This module is pure (stdlib only) and import-light: the guards run in the fast unit gate
and as a merge-blocking invariant test, with no network and no ``data`` extra.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping

__all__ = [
    "LeakError",
    "to_object_ids",
    "exclude_probe_from_pretrain",
    "assert_no_cross_corpus_leak",
    "assert_uncertainty_firewall",
    "partition_uncertainty",
    "assignment_unit",
]


class LeakError(AssertionError):
    """A structural split guarantee was violated (a leak that must be impossible).

    Subclasses :class:`AssertionError` so it reads as a broken invariant, not a runtime
    mishap — the same category as the frozen-encoder and parity guards.
    """


def to_object_ids(values: Iterable[object]) -> frozenset[int]:
    """Normalise an iterable of object IDs to a ``frozenset[int]``.

    SDSS ``objID`` is a 64-bit integer; accepting ``int``/``str``/``float`` and coercing
    once here means the guards compare *identities*, never mixed types (a silent
    ``"123" != 123`` mismatch would defeat the leak check). Non-integral floats raise.
    """
    out: set[int] = set()
    for v in values:
        if isinstance(v, bool):  # bool is an int subclass — never a valid objID
            raise TypeError(f"object id must not be a bool: {v!r}")
        if isinstance(v, int):
            out.add(v)
        elif isinstance(v, str):
            out.add(int(v.strip()))
        elif isinstance(v, float):
            if not v.is_integer():
                raise ValueError(f"object id is not integral: {v!r}")
            out.add(int(v))
        else:
            raise TypeError(f"unsupported object id type: {type(v).__name__}")
    return frozenset(out)


def exclude_probe_from_pretrain(
    pretrain_ids: Iterable[object], probe_ids: Iterable[object]
) -> frozenset[int]:
    """Return the pretraining IDs with every probing galaxy removed (collision resolution).

    The proposed default resolution (``splits.md`` fork): a galaxy present in both pulls
    belongs to **probing** and is dropped from pretraining. Idempotent and order-free.
    Call this when building the pretraining manifest; then assert the post-condition with
    :func:`assert_no_cross_corpus_leak`.
    """
    pre = to_object_ids(pretrain_ids)
    probe = to_object_ids(probe_ids)
    return pre - probe


def assert_no_cross_corpus_leak(
    pretrain_ids: Iterable[object], probe_ids: Iterable[object]
) -> None:
    """Raise :class:`LeakError` if any probing galaxy also sits in the pretraining corpus.

    The post-condition of :func:`exclude_probe_from_pretrain`, and the merge-blocking
    invariant: it must hold for the manifests a run actually trains on, so a leak cannot
    survive to pretraining. The error names a sample of the offending IDs.
    """
    pre = to_object_ids(pretrain_ids)
    probe = to_object_ids(probe_ids)
    overlap = pre & probe
    if overlap:
        sample = sorted(overlap)[:10]
        raise LeakError(
            f"{len(overlap)} galaxy(ies) appear in BOTH the pretraining and probing "
            f"corpora — the frozen encoder would see probe galaxies during pretraining "
            f"(D6 decoupling broken). Resolve with exclude_probe_from_pretrain(). "
            f"Offending objID sample: {sample}"
        )


def partition_uncertainty(
    vote_fractions: Mapping[object, float],
    *,
    low: float = 0.2,
    high: float = 0.8,
) -> tuple[frozenset[int], frozenset[int]]:
    """Split galaxies into ``(fit_extremes, test_middle)`` by consensus vote fraction.

    ``v >= high`` or ``v <= low`` → the high-consensus **fit** set (the binary axis is
    estimated here); ``low < v < high`` → the ambiguous **test** middle (held out of
    estimation, projected afterwards). This is the partition the firewall protects; the
    bounds are the same ``0.2``/``0.8`` used across the scratchpad.
    """
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(f"require 0 <= low < high <= 1; got low={low}, high={high}")
    fit: set[int] = set()
    middle: set[int] = set()
    for oid, v in vote_fractions.items():
        (oid_int,) = to_object_ids([oid])
        if v <= low or v >= high:
            fit.add(oid_int)
        else:
            middle.add(oid_int)
    return frozenset(fit), frozenset(middle)


def assert_uncertainty_firewall(
    fit_vote_fractions: Iterable[float],
    *,
    low: float = 0.2,
    high: float = 0.8,
) -> None:
    """Raise :class:`LeakError` if any *fit-set* vote fraction is in the ambiguous middle.

    The non-circular protocol made structural: the axis is fitted on extremes only, so a
    middle (``low < v < high``) value in the fit set means the held-out gradient leaked
    into estimation — and the uncertainty-geometry result would be a tautology. Run this
    on the vote fractions of whatever rows enter the fit, before fitting.
    """
    offenders = [v for v in fit_vote_fractions if low < v < high]
    if offenders:
        raise LeakError(
            f"{len(offenders)} fit-set galaxy(ies) have an ambiguous vote fraction in "
            f"({low}, {high}) — the held-out uncertainty gradient leaked into axis "
            f"estimation, making uncertainty geometry circular. Fit on extremes only "
            f"(v<={low} or v>={high}). Offending values: {offenders[:10]}"
        )


def assignment_unit(object_id: object, seed: int, *, salt: str = "") -> float:
    """Deterministic ``[0, 1)`` assignment coordinate for a galaxy, from ``(objID, seed)``.

    A stable SHA-256 hash (not Python's salted ``hash()``, which varies per process), so
    a galaxy's train/val/test home is reproducible from the seed alone — no split file to
    store or drift. ``salt`` namespaces independent partitions (e.g. ``"probe"`` vs
    ``"pretrain-monitor"``) so they do not correlate. The downstream split machinery
    (ratios, stratification) is out of scope here; this is the reproducible primitive it
    will build on.
    """
    (oid,) = to_object_ids([object_id])
    digest = hashlib.sha256(f"{seed}:{salt}:{oid}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)
