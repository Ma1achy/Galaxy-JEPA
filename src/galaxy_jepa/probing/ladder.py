"""The ladder — the gated cascade that assigns each feature its rung (design §2).

The per-feature R1/R2/R3/R4 classification is the core scientific output. The controls (3)
make each rung credible; this module is the structure that assigns them — a *gated cascade*
where each rung-up tests a *named* alternative hypothesis and is itself gated, and the rung is
a **deterministic function of the gate tree**, never a human read (the keystone of 3A).

Run-level phases (ordering forced by data dependencies):

0. Caller extracts the embeddings once and builds the control sources (``run.py``).
1. Per feature: linear probe + direction; assemble the five-null battery (``controls.py``).
2. Family-corrected existence verdict vs the five-null max + effect floor (``nulls.py``).
3. Global entanglement geometry over the existence-passing directions (``entanglement.py``).
4. Existence-passing features → entanglement R1/R2 + nuisance gate (+ triggered matching).
5. Existence-failing features → MLP capacity ladder → R3 / R4 (``mlp.py``).

Per feature the rung is emitted with its ``GateResult`` tree (so ``render()`` is the stamped,
pre-registered audit trail) and a named ``mechanism``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import numpy as np

from galaxy_jepa.core.gates import GateResult
from galaxy_jepa.core.gates import all as gate_all
from galaxy_jepa.probing import controls as ctl
from galaxy_jepa.probing import entanglement as ent
from galaxy_jepa.probing import matching as match
from galaxy_jepa.probing import mlp as mlp_mod
from galaxy_jepa.probing import nulls as nulls_mod
from galaxy_jepa.probing.config import ProbingConfig
from galaxy_jepa.probing.extract import LabelProvider, feature_embeddings
from galaxy_jepa.probing.gates import EXISTENCE_METRIC_FLOOR, build_gates
from galaxy_jepa.probing.logistic import ConceptDirection, Embeddings, probe_auc, probe_direction

__all__ = ["RungVerdict", "LadderResult", "run_ladder"]


@dataclasses.dataclass(frozen=True)
class RungVerdict:
    """One feature's place on the ladder, with the deterministic gate tree that produced it."""

    feature: str
    rung: str  # R1 | R2 | R3 | R4
    mechanism: str
    metrics: dict[str, float]
    gate_tree: GateResult
    nuisance_aucs: Mapping[str, float] = dataclasses.field(default_factory=dict)
    matched: match.MatchedVerdict | None = None
    sweep: list[mlp_mod.SweepRow] | None = None
    ceiling: int | None = None


@dataclasses.dataclass
class LadderResult:
    """The full ladder output: per-feature verdicts + the global entanglement geometry."""

    verdicts: dict[str, RungVerdict]
    existence: dict[str, nulls_mod.ExistenceVerdict]
    entanglement: ent.EntanglementGeometry | None
    feature_controls: dict[str, ctl.FeatureControls]
    directions: dict[str, ConceptDirection]


def _linear_probe(
    feature: str,
    controls: ctl.ControlEmbeddings,
    labels: LabelProvider,
    train_ids: Sequence[int],
    test_ids: Sequence[int],
    *,
    c: float,
) -> tuple[float, ConceptDirection | None, Embeddings, Embeddings]:
    """Stage-1 canonical linear probe: ROC-AUC + the concept direction (None if single-class)."""
    train = feature_embeddings(controls.real, labels, feature, train_ids)
    test = feature_embeddings(controls.real, labels, feature, test_ids)
    try:
        auc = probe_auc(train, test, c=c)
        direction = probe_direction(train, name=feature, c=c)
    except ValueError:  # a single-class train split — not linearly fittable
        return 0.5, None, train, test
    return auc, direction, train, test


def _entangled_map(
    geometry: ent.EntanglementGeometry | None,
    directions: Mapping[str, ConceptDirection],
    controls: ctl.ControlEmbeddings,
    labels: LabelProvider,
    train_ids: Sequence[int],
    test_ids: Sequence[int],
    *,
    config: ProbingConfig,
) -> dict[str, bool]:
    """Per-feature entangled (R2) flag, with the surgical conditional cross-check (2A).

    A flagged pair is *representational* entanglement only if the signal **survives** matching
    on the partner feature; if it vanishes under matching it was *world-correlation*
    (astrophysics) — itself a clean finding, so the feature is not marked entangled by it.
    """
    entangled: dict[str, bool] = {}
    if geometry is None:
        return entangled
    for a, b in geometry.entangled_pairs:
        # match feature A's probe on feature B's vote fraction (hold the world correlation fixed)
        a_train = feature_embeddings(controls.real, labels, a, train_ids)
        a_test = feature_embeddings(controls.real, labels, a, test_ids)
        b_train = labels.vote_fraction(b, [o for o in train_ids if o in controls.real.index])
        b_test = labels.vote_fraction(b, [o for o in test_ids if o in controls.real.index])
        verdict = match.matched_evaluation(
            a_train,
            a_test,
            b_train,
            b_test,
            survive_threshold=config.effect_floor,
            c=config.c,
            seed=config.seed,
        )
        if verdict.survived:  # representational entanglement — both partners are entangled
            entangled[a] = True
            entangled[b] = True
    return entangled


def _nuisance_clearance(
    feature: str,
    fc: ctl.FeatureControls,
    controls: ctl.ControlEmbeddings,
    labels: LabelProvider,
    train_ids: Sequence[int],
    test_ids: Sequence[int],
    *,
    config: ProbingConfig,
) -> tuple[bool, match.MatchedVerdict | None]:
    """Nuisance gate (3D-ii): clear iff no nuisance competitive, or a competitive one survives
    matched evaluation (the nuisance held constant within the matched set)."""
    competitive = [
        n
        for n, auc in fc.nuisance_aucs.items()
        if match.nuisance_competitive(fc.real_auc, auc, margin=config.nuisance_competitive_margin)
    ]
    if not competitive:
        return True, None
    # re-test on the most competitive nuisance, matched
    worst = max(competitive, key=lambda n: fc.nuisance_aucs[n])
    train = feature_embeddings(controls.real, labels, feature, train_ids)
    test = feature_embeddings(controls.real, labels, feature, test_ids)
    present_tr = [o for o in train_ids if o in controls.real.index]
    present_te = [o for o in test_ids if o in controls.real.index]
    verdict = match.matched_evaluation(
        train,
        test,
        labels.nuisance_value(worst, present_tr),
        labels.nuisance_value(worst, present_te),
        survive_threshold=config.effect_floor,
        c=config.c,
        seed=config.seed,
    )
    return verdict.survived, verdict


def _passing_rung(
    feature: str,
    existence: nulls_mod.ExistenceVerdict,
    fc: ctl.FeatureControls,
    entangled: bool,
    nuisance_cleared: bool,
    matched: match.MatchedVerdict | None,
    *,
    config: ProbingConfig,
) -> RungVerdict:
    """Assign R1/R2 to an existence-passing feature via the control gate tree (3A)."""
    gates = build_gates(config)
    nuisance_max = max(fc.nuisance_aucs.values(), default=0.5)
    metrics = {
        "auc": fc.real_auc,
        "exceeds_null": 1.0 if existence.exceeds_null else 0.0,
        "selectivity": fc.selectivity,
        "nuisance_cleared": 1.0 if nuisance_cleared else 0.0,
        "entangled": 1.0 if entangled else 0.0,
        "nuisance_auc_max": nuisance_max,
    }
    tree = gate_all(*gates.rung_inputs()).evaluate(metrics)
    clean_linear = (
        existence.exceeds_null
        and existence.clean
        and metrics["selectivity"] >= config.selectivity_floor
        and nuisance_cleared
        and not entangled
    )
    if clean_linear:
        rung, mechanism = "R1", "clean linear direction"
    elif not nuisance_cleared:
        worst = max(fc.nuisance_aucs, key=lambda n: fc.nuisance_aucs[n], default="?")
        rung, mechanism = "R2", f"confounded by {worst} (did not survive matching)"
    else:
        rung, mechanism = "R2", "entangled linear (present, not orthogonal)"
    return RungVerdict(
        feature=feature,
        rung=rung,
        mechanism=mechanism,
        metrics=metrics,
        gate_tree=tree,
        nuisance_aucs=fc.nuisance_aucs,
        matched=matched,
    )


def _failing_rung(
    feature: str,
    fc: ctl.FeatureControls,
    controls: ctl.ControlEmbeddings,
    labels: LabelProvider,
    train_ids: Sequence[int],
    test_ids: Sequence[int],
    nuisance_cleared: bool,
    *,
    config: ProbingConfig,
) -> RungVerdict:
    """Linear-failure → MLP bounded-capacity ladder → R3 (recoverable nonlinearly) / R4 (2D/2F)."""
    train = feature_embeddings(controls.real, labels, feature, train_ids)
    test = feature_embeddings(controls.real, labels, feature, test_ids)
    rng = np.random.default_rng(config.seed)
    ctrl_train = Embeddings(train.x, rng.permutation(train.y), train.fraction)
    ctrl_test = Embeddings(test.x, test.y, test.fraction)  # real test labels keep AUC defined
    sweep = mlp_mod.capacity_sweep(
        train,
        test,
        ctrl_train,
        ctrl_test,
        widths=config.mlp_widths,
        depth=config.mlp_depth,
        weight_decay=config.mlp_weight_decay,
        epochs=config.mlp_epochs,
        lr=config.mlp_lr,
        seed=config.seed,
    )
    # FLAGGED ceiling null threshold: the chance band for this probe — placeholder is the
    # configured quantile of the shuffled-label null (the linear control), floored at 0.5.
    null_threshold = max(0.5, float(np.quantile(fc.shuffled_nulls, config.ceiling_null_quantile)))
    ceiling = mlp_mod.selectivity_ceiling(sweep, null_threshold=null_threshold)
    rung = mlp_mod.rung_from_sweep(sweep, ceiling, decode_threshold=config.effect_floor)
    if rung == "R3" and not nuisance_cleared:
        rung = "R4"  # cannot rescue a feature by capacity while a nuisance is competitive
    decodes = 1.0 if rung == "R3" else 0.0
    metrics = {
        "auc": fc.real_auc,
        "exceeds_null": 0.0,
        "mlp_decodes_below_ceiling": decodes,
        "nuisance_cleared": 1.0 if nuisance_cleared else 0.0,
        "ceiling_width": float(ceiling) if ceiling is not None else float("nan"),
    }
    tree = gate_all(
        mlp_mod_gate("mlp_decodes_below_ceiling"),
        mlp_mod_gate("nuisance_cleared"),
    ).evaluate(metrics)
    if rung == "R3":
        mechanism = "recoverable nonlinearly (MLP below the selectivity ceiling)"
    else:
        # the resolution ablation (2E) is a cross-run diff of two encoders — out of a single
        # ladder run; this verdict is R4 *pending* that check (surfaced in the plan).
        mechanism = "not recoverable by linear or MLP (pending 8×8 resolution ablation)"
    return RungVerdict(
        feature=feature,
        rung=rung,
        mechanism=mechanism,
        metrics=metrics,
        gate_tree=tree,
        nuisance_aucs=fc.nuisance_aucs,
        sweep=sweep,
        ceiling=ceiling,
    )


def mlp_mod_gate(metric: str):
    """A 0/1-indicator gate (passes at ≥ 0.5) for the MLP branch's gate tree."""
    from galaxy_jepa.core.gates import MetricGate

    return MetricGate(metric, gte=EXISTENCE_METRIC_FLOOR)


def run_ladder(
    controls: ctl.ControlEmbeddings,
    labels: LabelProvider,
    train_ids: Sequence[int],
    test_ids: Sequence[int],
    *,
    config: ProbingConfig,
    sky_label_col: str,
) -> LadderResult:
    """Run the full gated cascade over every feature, returning per-feature rung verdicts.

    Uses the full labelled probe set (not the extremes filter) for the existence/entanglement
    tests — the nulls calibrate the bar; the extremes filter is the uncertainty geometry's job
    (``uncertainty.py``), which fits on extremes and tests the middle.
    """
    features = labels.features
    n_tests = config.n_primary_tests if config.n_primary_tests is not None else len(features)

    # Phase 1: per-feature linear probe + direction + the five-null battery.
    directions: dict[str, ConceptDirection] = {}
    real_aucs: dict[str, float] = {}
    feature_controls: dict[str, ctl.FeatureControls] = {}
    for i, feature in enumerate(features):
        auc, direction, _, _ = _linear_probe(
            feature, controls, labels, train_ids, test_ids, c=config.c
        )
        real_aucs[feature] = auc
        if direction is not None:
            directions[feature] = direction
        feature_controls[feature] = ctl.build_feature_controls(
            feature,
            real_auc=auc,
            train_ids=train_ids,
            test_ids=test_ids,
            controls=controls,
            labels=labels,
            sky_label_col=sky_label_col,
            c=config.c,
            seed=config.seed + i,
        )

    # Phase 2: family-corrected existence verdict (3B/2B) — the gate input.
    existence = nulls_mod.existence_verdicts(
        feature_controls,
        alpha=config.alpha,
        method=config.multiplicity,
        effect_floor=config.effect_floor,
        n_tests=n_tests,
    )

    # Phase 3: global entanglement geometry over the existence-passing directions (2A).
    passing = [f for f in features if existence[f].exceeds_null and f in directions]
    geometry: ent.EntanglementGeometry | None = None
    if len(passing) >= 2:
        geometry = ent.entanglement_geometry(
            [directions[f] for f in passing],
            controls.real.x,
            mp_method=config.mp_method,
            pair_quantile=config.entangled_pair_quantile,
        )
    entangled = _entangled_map(
        geometry, directions, controls, labels, train_ids, test_ids, config=config
    )

    # Phases 4 & 5: per-feature rung.
    verdicts: dict[str, RungVerdict] = {}
    for feature in features:
        fc = feature_controls[feature]
        cleared, matched = _nuisance_clearance(
            feature, fc, controls, labels, train_ids, test_ids, config=config
        )
        if existence[feature].exceeds_null and feature in directions:
            verdicts[feature] = _passing_rung(
                feature,
                existence[feature],
                fc,
                entangled.get(feature, False),
                cleared,
                matched,
                config=config,
            )
        else:
            verdicts[feature] = _failing_rung(
                feature, fc, controls, labels, train_ids, test_ids, cleared, config=config
            )

    return LadderResult(
        verdicts=verdicts,
        existence=existence,
        entanglement=geometry,
        feature_controls=feature_controls,
        directions=directions,
    )
