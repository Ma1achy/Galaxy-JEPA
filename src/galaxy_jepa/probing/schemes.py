"""Feature schemes (D14) — the feature set as an *experiment over two configs*, one harness.

The feature set is not a fixed choice. Two schemes run the **same** ladder, and the comparison
between them is itself a result (same verdicts ⇒ the reduction is cosmetic; different verdicts ⇒
reducing changes what is expressible). So a scheme is a *config*, never a second harness:

* **Scheme 1 — full tree.** All 37 answers of ``metadata.GZ2_TREE``, one feature per answer,
  each probed within its conditional population. The honest baseline, and it runs **first**
  (transparent, no cherry-picking). BY family = 37.
  *Power confound:* per-bucket weakness on the deep features is confounded between genuine
  absence and split sample (~9,870 spirals ÷ 6 arm buckets), so per-bucket weakness must not be
  read as "absent" — Scheme 2 is the diagnostic that separates them.
* **Scheme 2 — reduced.** Ordered questions collapse to one graded axis each; well-posed binary
  questions become one binary feature each; t08's odd-subtypes are an exploratory, low-power set
  outside the primary family.

Two filters ride on the specs, and they are **different things**:

* the **vote-count filter** — a per-feature reach/reliability floor. A hard filter, and legitimately
  so: a galaxy nobody was asked about carries no measurement.
* the **conditional population** — the galaxies that reached the question. This is run as a
  **comparison** against the full population, *never* as a hard mask (D14): a no-bulge galaxy
  carrying boxy-bulge votes measures human disagreement, and masking it pre-imposes the tree's
  logic before testing whether it holds.

**The graded existence test is deliberately not implemented.** Whether a graded axis is tested by
AUC on a binarised fraction or by a Spearman/permutation correlation — which may be the *same*
measurement as the uncertainty geometry for that feature, collapsing the two — is an open design
question (spec §Feature-scheme experiment, open register item 1). :class:`FeatureSpec` carries the
ordered columns so the structure is wired, and any attempt to take a label from a graded feature
raises :class:`GradedExistenceTestUndecided`. Binary features keep AUC regardless, and Scheme 1 has
no graded features — so this blocks nothing until Scheme 2 runs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np

from galaxy_jepa.data.metadata import (
    GZ2_CONDITIONS,
    GZ2_TREE,
    vote_column,
)

__all__ = [
    "FeatureKind",
    "FeatureSpec",
    "FeatureScheme",
    "GradedExistenceTestUndecided",
    "get_scheme",
    "scheme_names",
    "full_tree_scheme",
    "reduced_scheme",
    "derive_vote_count_min",
    "DEFAULT_CONSENSUS_GATE",
]

FeatureKind = Literal["binary", "graded", "exploratory"]

#: Vote-fraction floor for "this galaxy consensus-reached the upstream answer". **OPEN** (spec open
#: register, item 2): which upstream vote, which threshold, consensus vs weighted. A per-run knob.
DEFAULT_CONSENSUS_GATE: float = 0.5


class GradedExistenceTestUndecided(NotImplementedError):
    """A graded axis was asked for a label before its existence test was decided.

    Raised loudly rather than defaulted, because the two candidate tests (AUC on a binarised
    fraction vs a Spearman/permutation correlation) are *different measurements* and one of them
    may be the uncertainty geometry itself. Defaulting would silently pick a scientific answer.
    """


@dataclasses.dataclass(frozen=True)
class FeatureSpec:
    """One probed feature: where its votes live, how it is tested, and who reaches its question."""

    name: str
    question: str
    kind: FeatureKind
    fraction_col: str | None = None  # binary / exploratory: the single answer fraction
    #: This answer's own vote count. **Not** the reach filter — see :meth:`reach_count_cols`,
    #: which sums every answer to the question, because a fraction's denominator is the total.
    count_col: str | None = None
    graded_cols: tuple[str, ...] = ()  # graded: the ordered answer fractions, low → high
    graded_count_cols: tuple[str, ...] = ()
    #: Conditions beyond the GZ2 tree's own reachability chain. Each entry is a group of
    #: columns whose **summed** vote fraction must clear the consensus gate, so a condition can
    #: name a set of answers ("a bulge is present" = rounded + boxy) rather than only one.
    extra_conditions: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        if self.kind == "graded":
            if not self.graded_cols:
                raise ValueError(f"graded feature {self.name!r} needs ordered graded_cols")
            if self.fraction_col is not None:
                raise ValueError(
                    f"graded feature {self.name!r} must not carry a single fraction_col — "
                    "collapsing the ordered answers to one scalar is part of the undecided "
                    "graded existence test, not a property of the spec"
                )
        elif self.fraction_col is None:
            raise ValueError(f"{self.kind} feature {self.name!r} needs a fraction_col")

    @property
    def conditions(self) -> tuple[tuple[str, str], ...]:
        """The upstream ``(question, answer)`` chain that reaches this feature's question."""
        return GZ2_CONDITIONS[self.question]

    @property
    def is_primary(self) -> bool:
        """Primary features carry the BY family; exploratory ones are reported separately."""
        return self.kind != "exploratory"

    def condition_groups(self) -> tuple[tuple[str, ...], ...]:
        """The consensus gates on this feature's population, as groups of columns to sum.

        The GZ2 reachability chain contributes one single-column group per upstream answer;
        :attr:`extra_conditions` contributes any further gate. Summing within a group is what
        lets a condition be "a bulge is present" (rounded + boxy) rather than only a single
        answer — the alternative, a negated gate on ``no_bulge``, reads the same but inverts
        which way the consensus threshold points, which is a different filter.
        """
        tree = tuple((vote_column(q, a),) for q, a in self.conditions)
        return tree + self.extra_conditions

    def reach_count_cols(self) -> tuple[str, ...]:
        """Every answer's vote-count column for this feature's **question** — the reach denominator.

        The floor asks "did enough people answer this question", so the denominator is the
        question total, not one answer's share of it. That distinction is not cosmetic. A GZ2
        vote fraction is ``count_answer / total_question``, so its precision depends on the
        total; and filtering on ``count_col`` instead drops the **well-defined zeros** — the
        galaxies where the question *was* answered and nobody chose this answer, which is most
        of the negatives. Measured on the 230,358-galaxy probe corpus at a floor of 1, that
        would have discarded 72.6% of t09 boxy's eligible galaxies, 85.6% of t11 4-arms and
        87.2% of t11 >4-arms, leaving probe sets of almost nothing but positives.

        This also puts binary specs back in step with graded ones, which already summed every
        answer, and with :func:`eligible_ids`' own contract ("no votes on this question").
        """
        return tuple(vote_column(self.question, a, "count") for a in GZ2_TREE[self.question])

    def condition_columns(self) -> tuple[str, ...]:
        """Flat view of :meth:`condition_groups` — every column any gate reads."""
        return tuple(col for group in self.condition_groups() for col in group)

    def require_testable(self) -> str:
        """The fraction column to test on, or raise if the test itself is undecided."""
        if self.kind == "graded":
            raise GradedExistenceTestUndecided(
                f"feature {self.name!r} is a graded axis over {list(self.graded_cols)}: its "
                "existence test (AUC on a binarised fraction vs Spearman/permutation, which may "
                "collapse into the uncertainty geometry) is an open design question — decide it "
                "before wiring Scheme 2. Scheme 1 is unaffected and runs first."
            )
        assert self.fraction_col is not None  # guaranteed by __post_init__
        return self.fraction_col


@dataclasses.dataclass(frozen=True)
class FeatureScheme:
    """A named feature set: the specs, plus the BY family size they imply."""

    name: str
    specs: tuple[FeatureSpec, ...]

    def __post_init__(self) -> None:
        seen = [s.name for s in self.specs]
        if len(set(seen)) != len(seen):
            raise ValueError(f"scheme {self.name!r} has duplicate feature names")

    def __len__(self) -> int:
        return len(self.specs)

    @property
    def by_name(self) -> dict[str, FeatureSpec]:
        return {s.name: s for s in self.specs}

    def primary(self) -> tuple[FeatureSpec, ...]:
        return tuple(s for s in self.specs if s.is_primary)

    def exploratory(self) -> tuple[FeatureSpec, ...]:
        return tuple(s for s in self.specs if not s.is_primary)

    def family_size(self) -> int:
        """The Benjamini–Yekutieli family count — **per-scheme**, never a global constant.

        Exploratory features are excluded: they are reported as an explicitly low-power set, so
        folding them into the primary family would dilute the correction for the tests that carry
        the claim (spec §Statistics (3): family count per-scheme).
        """
        return len(self.primary())

    def feature_cols(self) -> dict[str, str]:
        """Feature name → vote-fraction column, for the ``LabelProvider``.

        Graded features are omitted: they have no single fraction column by construction, and
        inventing one is the undecided part. A caller wanting them raises via
        :meth:`FeatureSpec.require_testable`.
        """
        return {s.name: s.fraction_col for s in self.specs if s.fraction_col is not None}

    def count_cols(self) -> dict[str, str]:
        return {s.name: s.count_col for s in self.specs if s.count_col is not None}

    def graded(self) -> tuple[FeatureSpec, ...]:
        return tuple(s for s in self.specs if s.kind == "graded")


# --- the two schemes ------------------------------------------------------------------------


def full_tree_scheme() -> FeatureScheme:
    """Scheme 1 — every answer of the GZ2 tree as its own per-bucket binary feature (37)."""
    specs = [
        FeatureSpec(
            name=f"{question}_{answer}",
            question=question,
            kind="binary",
            fraction_col=vote_column(question, answer),
            count_col=vote_column(question, answer, "count"),
        )
        for question, answers in GZ2_TREE.items()
        for answer in answers
    ]
    return FeatureScheme("full_tree", tuple(specs))


#: Scheme 2's well-posed binary questions → one binary feature each (the positive answer).
#:
#: **t09 bulge shape is here, not among the graded axes (D14).** Rounded / boxy / no-bulge is not
#: ordered: the four graded axes are all genuinely ordinal (1→2→3→4→5+, tight→medium→loose,
#: none→just-noticeable→obvious→dominant, round→in-between→cigar), whereas bulge shape is a
#: categorical contrast with an absence bolted on. Collapsing it to an axis would impose an order
#: that does not exist — the precise failure the graded framing exists to prevent. It enters as
#: the **boxy-versus-rounded** contrast, conditioned on edge-on *and* bulge-present, because
#: D13's confound-2 deliverable — does the encoder separate boxy from rounded where humans
#: cannot — lives entirely in that two-way contrast.
_REDUCED_BINARY: tuple[tuple[str, str, str], ...] = (
    ("featured", "t01_smooth_or_features", "a02_features_or_disk"),
    ("edge_on", "t02_edgeon", "a04_yes"),
    ("bar", "t03_bar", "a06_bar"),
    ("spiral", "t04_spiral", "a08_spiral"),
    ("odd", "t06_odd", "a14_yes"),
    ("bulge_boxy", "t09_bulge_shape", "a26_boxy"),
)

#: "A bulge is present" — the extra gate on the boxy-vs-rounded contrast, on top of t09's own
#: tree chain (featured → edge-on). Summed rather than expressed as a negated no-bulge gate so
#: the consensus threshold keeps pointing the same way as every other condition.
_BULGE_PRESENT: tuple[str, ...] = (
    vote_column("t09_bulge_shape", "a25_rounded"),
    vote_column("t09_bulge_shape", "a26_boxy"),
)

#: The graded axes: one per ordered question, named for the axis rather than an answer.
_REDUCED_GRADED: tuple[tuple[str, str], ...] = (
    ("bulge_prominence", "t05_bulge_prominence"),
    ("roundedness", "t07_rounded"),
    ("arms_winding", "t10_arms_winding"),
    ("arms_number", "t11_arms_number"),
)


def reduced_scheme() -> FeatureScheme:
    """Scheme 2 — graded questions as one axis each, binary questions as one feature each.

    t08's odd-subtypes stay per-answer but are marked **exploratory**: they are rare, so their
    power is poor, and they are reported outside the primary BY family rather than dragging it.

    Ten primaries: five well-posed binaries, the boxy-vs-rounded bulge-shape contrast, and four
    graded axes. **Deferred alternative:** t09 as three per-answer binaries (family 12). Its one
    real argument is that t05 and t09 are asked of *disjoint* populations (featured non-edge-on
    vs featured edge-on), so t09's no-bulge is not redundant with t05's low end — it is the same
    concept measured on the other branch, structurally v1's Q4/Q7 situation. If a cross-branch
    consistency check is wanted later, family 12 is where it lives.
    """
    specs: list[FeatureSpec] = [
        FeatureSpec(
            name=name,
            question=question,
            kind="binary",
            fraction_col=vote_column(question, answer),
            count_col=vote_column(question, answer, "count"),
            extra_conditions=(_BULGE_PRESENT,) if name == "bulge_boxy" else (),
        )
        for name, question, answer in _REDUCED_BINARY
    ]
    specs += [
        FeatureSpec(
            name=name,
            question=question,
            kind="graded",
            graded_cols=tuple(vote_column(question, a) for a in GZ2_TREE[question]),
            graded_count_cols=tuple(vote_column(question, a, "count") for a in GZ2_TREE[question]),
        )
        for name, question in _REDUCED_GRADED
    ]
    specs += [
        FeatureSpec(
            name=f"odd_{answer}",
            question="t08_odd_feature",
            kind="exploratory",
            fraction_col=vote_column("t08_odd_feature", answer),
            count_col=vote_column("t08_odd_feature", answer, "count"),
        )
        for answer in GZ2_TREE["t08_odd_feature"]
    ]
    return FeatureScheme("reduced", tuple(specs))


_BUILDERS = {"full_tree": full_tree_scheme, "reduced": reduced_scheme}


def scheme_names() -> list[str]:
    return sorted(_BUILDERS)


def get_scheme(name: str) -> FeatureScheme:
    """Resolve a scheme by name, failing loudly on an unknown one (never a silent default)."""
    try:
        builder = _BUILDERS[name]
    except KeyError:
        raise KeyError(f"unknown feature scheme {name!r}; known: {scheme_names()}") from None
    return builder()


# --- the vote-count reach filter --------------------------------------------------------------


def derive_vote_count_min(
    counts: Sequence[float] | np.ndarray, *, method: str = "mean_plus_2sigma"
) -> float:
    """Derive the reliability threshold from the corpus's own vote-count distribution.

    Carries v1's *method* (mean + 2σ) rather than its number, per D8. The distinction is the
    whole point: v1 read 21 off the PyPI ``galaxy-datasets`` release, and this corpus is a
    direct SciServer pull with different vote counts — the same method re-derived here lands at
    ~36.6 per question, and ~300 if taken over ``total_votes``. So this is the tool for the
    re-derivation, not a decision, and there is deliberately no default anywhere for it to
    quietly supply. See ``probing.config.VoteCountFreeze``.
    """
    values = np.asarray(counts, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("no finite vote counts to derive a threshold from")
    if method == "mean_plus_2sigma":
        return float(values.mean() + 2.0 * values.std())
    raise ValueError(f"unknown vote-count threshold method {method!r}")


def eligible_ids(
    rows: Mapping[int, Mapping[str, object]],
    spec: FeatureSpec,
    ids: Sequence[int],
    *,
    vote_count_min: float,
    conditional: bool = False,
    consensus_gate: float = DEFAULT_CONSENSUS_GATE,
) -> list[int]:
    """The galaxies that enter ``spec``'s probe set, under one population definition.

    Two filters, deliberately separable:

    * the **vote-count floor** always applies — a galaxy with no votes on this question carries
      no measurement to probe. Note what that is load-bearing *for* on this corpus: GZ2 stores an
      unreached question's fraction as a literal ``0.0``, not as a blank, so 51.6% of the probe
      corpus carries ``t09_..._fraction = 0.0`` meaning "never asked" — indistinguishable by
      value from "asked, nobody said boxy". This floor is the only thing between that and the
      probe, where it would land as 118,962 fabricated negatives;
    * the **conditional population** applies only when ``conditional`` is set. Running with it
      both off and on is the D14 comparison; running with it permanently on would be the hard
      mask the design forbids.
    """

    def _value(oid: int, col: str) -> float:
        raw = rows.get(int(oid), {}).get(col)
        try:
            return float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return float("nan")

    count_cols = spec.reach_count_cols()
    groups = spec.condition_groups() if conditional else ()

    def _group_share(oid: int, group: tuple[str, ...]) -> float:
        return sum(v for v in (_value(oid, c) for c in group) if np.isfinite(v))

    out: list[int] = []
    for oid in ids:
        if count_cols:
            total = sum(v for v in (_value(oid, c) for c in count_cols) if np.isfinite(v))
            if not total >= vote_count_min:
                continue
        if any(not _group_share(oid, g) >= consensus_gate for g in groups):
            continue
        out.append(int(oid))
    return out
