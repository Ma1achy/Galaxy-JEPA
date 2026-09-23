"""The ladder — the gated cascade that assigns each feature its rung (design §2).

The per-feature R1/R2/R3/R4 classification is the core scientific output. The controls (3)
make each rung credible; this module is the structure that assigns them — a *gated cascade*
where each rung-up tests a *named* alternative hypothesis and is itself gated, and the rung is
a **deterministic function of the gate tree**, never a human read (the keystone of 3A).

Run-level phases (ordering forced by data dependencies):

0. Caller extracts the embeddings once and builds the control sources (``run.py``).
1. Per feature: linear probe + direction; assemble the 3C battery (``controls.py``).
2. Family-corrected existence verdict vs the chance-calibrated null max + the effect floor
   (``nulls.py``; the sky/noise control is a diagnostic and not in that max — D19).
3. Global entanglement geometry over the existence-passing directions (``entanglement.py``).
4. Existence-passing features → entanglement R1/R2 + nuisance gate (+ triggered matching).
5. Existence-failing features → MLP capacity ladder → R3 / R4 (``mlp.py``).

Per feature the rung is emitted with its ``GateResult`` tree (so ``render()`` is the stamped,
pre-registered audit trail) and a named ``mechanism``.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from galaxy_jepa.core.gates import GateResult
from galaxy_jepa.core.gates import all as gate_all
from galaxy_jepa.probing import controls as ctl
from galaxy_jepa.probing import entanglement as ent
from galaxy_jepa.probing import matching as match
from galaxy_jepa.probing import mlp as mlp_mod
from galaxy_jepa.probing import nulls as nulls_mod
from galaxy_jepa.probing.config import ProbingConfig
from galaxy_jepa.probing.extract import (
    EmbeddingMatrix,
    LabelProvider,
    feature_embeddings,
    feature_ids,
)
from galaxy_jepa.probing.gates import EXISTENCE_METRIC_FLOOR, build_gates
from galaxy_jepa.probing.logistic import (
    ConceptDirection,
    Embeddings,
    probe_auc_ci_se,
    probe_direction,
)

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
    #: Power, per bucket. An R4 on 160 positives and an R4 on 9,019 are different findings, and
    #: nothing downstream could tell them apart without these. ``underpowered`` means the rung
    #: reads "cannot resolve at this N", NEVER "absent".
    n_train: int = 0
    positives_train: int = 0
    n_test: int = 0
    positives_test: int = 0
    real_se: float = 0.0
    resolvable_margin: float = 0.0
    underpowered: bool = False


@dataclasses.dataclass
class LadderResult:
    """The full ladder output: per-feature verdicts + the global entanglement geometry."""

    verdicts: dict[str, RungVerdict]
    existence: dict[str, nulls_mod.ExistenceVerdict]
    entanglement: ent.EntanglementGeometry | None
    feature_controls: dict[str, ctl.FeatureControls]
    directions: dict[str, ConceptDirection]
    #: The 2A pair adjudications behind every entangled flag, kept so a reader can see WHY a
    #: feature was marked — which of the four measures agreed, and which did not.
    pair_verdicts: list[ent.PairVerdict] = dataclasses.field(default_factory=list)


def _load_untrained_bank(
    config: ProbingConfig, features: Sequence[str]
) -> dict[str, np.ndarray] | None:
    """The K-seed untrained bar for D23's existence test, or ``None`` under the empirical method.

    Raises rather than falling back. A missing bank under ``existence_method='untrained_z'`` is a
    configuration error, and silently reverting to the point-mass estimator would produce a
    plausible-looking catalogue computed by a test the config said not to use.
    """
    if config.existence_method != nulls_mod.EXISTENCE_UNTRAINED_Z:
        return None
    if not config.untrained_bank_path:
        raise ValueError(
            "existence_method='untrained_z' needs ProbingConfig.untrained_bank_path. "
            "Build it with `uv run python artifacts/p1_untrained_bank.py`."
        )
    path = Path(config.untrained_bank_path)
    if not path.exists():
        raise FileNotFoundError(f"untrained bank not found at {path}")
    rec = json.loads(path.read_text())
    seeds = rec.get("seeds", {})
    if not seeds:
        raise ValueError(f"untrained bank at {path} is empty")
    bank: dict[str, np.ndarray] = {}
    for feature in features:
        vals = [s[feature] for s in seeds.values() if feature in s]
        if len(vals) != len(seeds):
            raise ValueError(
                f"untrained bank at {path} covers {len(vals)} of {len(seeds)} seeds for "
                f"{feature!r} — an uneven bank would give features different-sized denominators"
            )
        bank[feature] = np.asarray(vals, dtype=np.float64)
    return bank


def _linear_probe(
    feature: str,
    controls: ctl.ControlEmbeddings,
    labels: LabelProvider,
    train_ids: Sequence[int],
    test_ids: Sequence[int],
    *,
    c: float,
    seed: int = 0,
    n_boot: int = 2000,
) -> tuple[float, float, ConceptDirection | None, Embeddings, Embeddings]:
    """Stage-1 canonical linear probe: ROC-AUC, its bootstrap SE, and the concept direction.

    The SE arrived with D23: the ``untrained_z`` existence test needs the real AUC's sampling
    uncertainty on one side of the comparison, and the power rule needs it again to say what
    margin a bucket could have resolved. ``direction`` is ``None`` on a single-class split.
    """
    train = feature_embeddings(controls.real, labels, feature, train_ids)
    test = feature_embeddings(controls.real, labels, feature, test_ids)
    try:
        auc, _lo, _hi, se = probe_auc_ci_se(train, test, c=c, seed=seed, n_boot=n_boot)
        direction = probe_direction(train, name=feature, c=c)
    except ValueError:  # a single-class train split — not linearly fittable
        return 0.5, 0.0, None, train, test
    return auc, se, direction, train, test


def _entangled_map(
    geometry: ent.EntanglementGeometry | None,
    directions: Mapping[str, ConceptDirection],
    controls: ctl.ControlEmbeddings,
    labels: LabelProvider,
    train_ids: Sequence[int],
    test_ids: Sequence[int],
    *,
    config: ProbingConfig,
) -> tuple[dict[str, bool], list[ent.PairVerdict]]:
    """Per-feature entangled (R2) flag + the pair adjudications that produced it (2A).

    **The MP null is now a gate input.** The spec requires the R1/R2 verdict to clear the
    eigen-quantification *and*, for flagged pairs, the conditional test to attribute the
    entanglement representationally. `mp.significant` was computed and then never consulted, so
    the flag turned on the matched cross-check alone — half the evidence the design specifies.
    The decision now goes through `ent.adjudicate_pair`, the one pre-registered mapping from
    (eigen, cosine, CAV, conditional) to a verdict.

    A pair is entangled only on ``representational_entanglement``. ``world_correlation`` — the
    direction vanishing when the partner is held constant — is a *clean finding* about the sky,
    not a mark against the feature, and D13's bar+arms case is exactly that shape.
    """
    entangled: dict[str, bool] = {}
    pair_verdicts: list[ent.PairVerdict] = []
    if geometry is None:
        return entangled, pair_verdicts
    index = {name: i for i, name in enumerate(geometry.names)}
    for a, b in geometry.entangled_pairs:
        # match feature A's probe on feature B's vote fraction (hold the world correlation fixed)
        a_train = feature_embeddings(controls.real, labels, a, train_ids)
        a_test = feature_embeddings(controls.real, labels, a, test_ids)
        # feature A's eligible ids — the partner's fractions must line up with A's rows
        b_train = labels.vote_fraction(b, feature_ids(controls.real, labels, a, train_ids))
        b_test = labels.vote_fraction(b, feature_ids(controls.real, labels, a, test_ids))
        matched = match.matched_evaluation(
            a_train,
            a_test,
            b_train,
            b_test,
            survive_threshold=config.effect_floor,
            c=config.c,
            seed=config.seed,
        )
        # A degenerate match is a statement about the SAMPLE: it can neither confirm survival nor
        # attribute the entanglement to the world, so the conditional leg is recorded as unrun.
        survived = None if matched.degenerate else matched.survived
        pv = ent.adjudicate_pair(
            a,
            b,
            cosine=float(geometry.cosine[index[a], index[b]]),
            mp_significant=bool(geometry.mp.significant),
            cav_disagreement=geometry.cav_disagreement,
            survived_matching=survived,
        )
        pair_verdicts.append(pv)
        if pv.verdict == ent.REPRESENTATIONAL:
            entangled[a] = True
            entangled[b] = True
    return entangled, pair_verdicts


@dataclasses.dataclass(frozen=True)
class MatchedRows:
    """The rows a feature's matched re-probe runs over, before stratification.

    The feature's eligible rows, minus those whose ``nuisance`` measurement is unusable, plus that
    nuisance's values. Stratification (:func:`matching.matched_indices`) then depends only on these
    values, the labels and a seed — never on an embedding — so any matrix co-indexed with the real
    one can be scored on exactly the same galaxies. That is what the untrained bar re-measured on
    the matched rows (O1's ``C_m``) and the matched MLP both need.
    """

    nuisance: str
    keep_train: np.ndarray
    keep_test: np.ndarray
    values_train: np.ndarray
    values_test: np.ndarray

    def embeddings(
        self,
        matrix: EmbeddingMatrix,
        labels: LabelProvider,
        feature: str,
        train_ids: Sequence[int],
        test_ids: Sequence[int],
    ) -> tuple[Embeddings, Embeddings]:
        """``matrix``'s rows for this feature, restricted to the usable-nuisance rows."""
        full_tr = feature_embeddings(matrix, labels, feature, train_ids)
        full_te = feature_embeddings(matrix, labels, feature, test_ids)
        kt, ke = self.keep_train, self.keep_test
        return (
            Embeddings(full_tr.x[kt], full_tr.y[kt], full_tr.fraction[kt]),
            Embeddings(full_te.x[ke], full_te.y[ke], full_te.fraction[ke]),
        )


def matched_rows(
    nuisance_aucs: Mapping[str, float],
    real: EmbeddingMatrix,
    labels: LabelProvider,
    feature: str,
    train_ids: Sequence[int],
    test_ids: Sequence[int],
) -> MatchedRows:
    """The worst nuisance and the rows it can be matched over.

    The worst nuisance whether or not it cleared the competitive margin: with nothing competitive
    there is still a strongest one, and matching against it is the honest check.
    """
    worst = max(nuisance_aucs, key=lambda n: nuisance_aucs[n])
    present_tr = feature_ids(real, labels, feature, train_ids)
    present_te = feature_ids(real, labels, feature, test_ids)
    keep_tr = np.asarray(labels.nuisance_valid(worst, present_tr), dtype=bool)
    keep_te = np.asarray(labels.nuisance_valid(worst, present_te), dtype=bool)
    ids_tr = [o for o, k in zip(present_tr, keep_tr, strict=True) if k]
    ids_te = [o for o, k in zip(present_te, keep_te, strict=True) if k]
    return MatchedRows(
        nuisance=worst,
        keep_train=keep_tr,
        keep_test=keep_te,
        values_train=labels.nuisance_value(worst, ids_tr),
        values_test=labels.nuisance_value(worst, ids_te),
    )


def _nuisance_clearance(
    feature: str,
    fc: ctl.FeatureControls,
    controls: ctl.ControlEmbeddings,
    labels: LabelProvider,
    train_ids: Sequence[int],
    test_ids: Sequence[int],
    *,
    margin_established: bool,
    config: ProbingConfig,
) -> tuple[bool, match.MatchedVerdict | None]:
    """Nuisance gate (3D-ii): clear iff no nuisance is competitive, or the worst one's effect is
    RETAINED under matched evaluation (the nuisance held constant within the matched set).

    **Survival is retention, not the effect floor (D24).** This gate judged "survived matching" as
    *matched AUC ≥ effect floor*, so an answer already below the floor unmatched failed whatever
    matching did — 21 of Brief P's 22 "confounded" answers. It now applies O1's pre-registered rule
    (:func:`matching.retention_verdict`): does the margin over the untrained bar survive, measured
    on the same matched rows with the bar re-measured there? ``C`` and ``C_m`` average
    :data:`matching.RETENTION_SEEDS` untrained draws — Brief R0's construction, reproduced exactly.
    The effect floor is an absolute clean-vs-marginal threshold and never tests anything relative.

    **The margin floor reuses a gate rather than adding a parameter.** Retention is a ratio of
    margins, and where the unmatched margin is not itself established the ratio divides noise by
    noise (Brief R: ``t10 winding: medium``, margin 0.003 against a 0.006 bar spread, read COLLAPSES
    linearly and SURVIVES with an MLP). So retention is judged only where the unmatched margin has
    passed D23's existence test; otherwise the verdict is UNRESOLVED — a statement about the
    evidence, not the nuisance.

    **Matching runs unconditionally**, and competitiveness is recorded alongside rather than
    deciding whether to look: a trigger that always fires is not a trigger. **The `nuisance_valid`
    filter is applied**, so the panel and the matched re-probe are taken over the same rows.
    """
    competitive = [
        n
        for n, auc in fc.nuisance_aucs.items()
        if match.nuisance_competitive(fc.real_auc, auc, margin=config.nuisance_competitive_margin)
    ]
    if not fc.nuisance_aucs:
        return True, None
    seeds = controls.untrained_seeds
    if len(seeds) != match.RETENTION_SEEDS:
        raise ValueError(
            f"matched survival needs {match.RETENTION_SEEDS} untrained draws for C and C_m (D24), "
            f"got {len(seeds)}: pass ControlEmbeddings(untrained_extra=...)"
        )
    rows = matched_rows(fc.nuisance_aucs, controls.real, labels, feature, train_ids, test_ids)
    train, test = rows.embeddings(controls.real, labels, feature, train_ids, test_ids)
    i_tr, i_te = match.matched_indices(
        rows.values_train, train.y, rows.values_test, test.y, n_strata=5, seed=config.seed
    )

    def sub(e: Embeddings, i: np.ndarray) -> Embeddings:
        return Embeddings(e.x[i], e.y[i], e.fraction[i])

    mtr, mte = sub(train, i_tr), sub(test, i_te)
    degenerate = mte.y.size == 0 or len(np.unique(mtr.y)) < 2 or len(np.unique(mte.y)) < 2
    n_test = len(feature_ids(controls.real, labels, feature, test_ids))
    bar = float(
        np.mean(
            [
                ctl._safe_auc(
                    feature_embeddings(u, labels, feature, train_ids),
                    feature_embeddings(u, labels, feature, test_ids),
                    c=config.c,
                )
                for u in seeds
            ]
        )
    )
    m = m_lo = bar_m = None
    if not degenerate:
        m, m_lo, _, _ = probe_auc_ci_se(
            mtr, mte, c=config.c, n_boot=config.n_boot, seed=config.seed
        )
        bar_m_draws = []
        for u in seeds:
            utr, ute = rows.embeddings(u, labels, feature, train_ids, test_ids)
            bar_m_draws.append(ctl._safe_auc(sub(utr, i_tr), sub(ute, i_te), c=config.c))
        bar_m = float(np.mean(bar_m_draws))
    retention = match.retention_verdict(
        fc.real_auc, bar, m, m_lo, bar_m, n_matched_test=int(mte.y.size), n_test=n_test
    )
    if not margin_established and retention.verdict != match.UNRESOLVED:
        retention = dataclasses.replace(retention, verdict=match.UNRESOLVED)
    survived = retention.verdict == match.SURVIVES
    verdict = match.MatchedVerdict(
        matched_auc=0.5 if m is None else m,
        survived=survived,
        n_matched_train=int(mtr.y.size),
        n_matched_test=int(mte.y.size),
        n_train=int(train.y.size),
        n_test=int(test.y.size),
        degenerate=degenerate,
        retention=retention,
        matched_auc_lo=m_lo,
        bar_unmatched=bar,
        bar_matched=bar_m,
        k_bar=len(seeds),
        margin_established=margin_established,
    )
    # Nothing competitive is still a clearance — the matched verdict is then a measurement carried
    # for the record, not a gate that can fail the feature.
    return (True if not competitive else survived), verdict


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
    # The mechanism names the gate that actually failed. It used to call every cleared-but-not-clean
    # answer "entangled", which — once matching stopped failing below-floor answers by arithmetic
    # (D24) — would have relabelled twenty of them with a second wrong reason.
    if clean_linear:
        rung, mechanism = "R1", "clean linear direction"
    elif not nuisance_cleared:
        worst = max(fc.nuisance_aucs, key=lambda n: fc.nuisance_aucs[n], default="?")
        state = None if matched is None or matched.retention is None else matched.retention.verdict
        if state == match.UNRESOLVED:
            mechanism = f"nuisance clearance unresolved ({worst})"
        elif state == match.PARTIAL:
            mechanism = f"confounded by {worst} (partial retention under matching)"
        else:
            mechanism = f"confounded by {worst} (collapsed under matching)"
        rung = "R2"
    elif entangled:
        rung, mechanism = "R2", "entangled linear (present, not orthogonal)"
    elif not existence.clean:
        rung, mechanism = "R2", "present, below the effect floor"
    else:
        rung, mechanism = "R2", "present, not selective"
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
    matched: match.MatchedVerdict | None = None,
    *,
    config: ProbingConfig,
) -> RungVerdict:
    """Linear-failure → MLP bounded-capacity ladder → R3 (recoverable nonlinearly) / R4 (2D/2F).

    ``matched`` is carried onto the verdict so a failing feature's clearance is on the record too:
    under D24 it is UNRESOLVED by construction (no established margin), and a record that omits it
    cannot show that.
    """
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
        matched=matched,
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

    # Phase 1: per-feature linear probe + direction + the 3C battery.
    directions: dict[str, ConceptDirection] = {}
    real_aucs: dict[str, float] = {}
    real_ses: dict[str, float] = {}
    counts: dict[str, tuple[int, int, int, int]] = {}
    feature_controls: dict[str, ctl.FeatureControls] = {}
    for i, feature in enumerate(features):
        auc, se, direction, tr_emb, te_emb = _linear_probe(
            feature,
            controls,
            labels,
            train_ids,
            test_ids,
            c=config.c,
            seed=config.seed,
            n_boot=config.n_boot,
        )
        real_aucs[feature] = auc
        real_ses[feature] = se
        # N and positives per bucket, because an R4 on 160 positives and an R4 on 9,019 are
        # different findings and nothing downstream could tell them apart otherwise.
        counts[feature] = (
            int(len(tr_emb.y)),
            int(tr_emb.y.sum()),
            int(len(te_emb.y)),
            int(te_emb.y.sum()),
        )
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
            n_draws=config.n_null_draws,
            seed=config.seed + i,
        )

    # Phase 2: family-corrected existence verdict (3B/2B) — the gate input.
    bank = _load_untrained_bank(config, features)
    existence = nulls_mod.existence_verdicts(
        feature_controls,
        alpha=config.alpha,
        method=config.multiplicity,
        effect_floor=config.effect_floor,
        n_tests=n_tests,
        existence_method=config.existence_method,
        untrained_bank=bank,
        real_se=real_ses if bank is not None else None,
    )

    # Phase 3: global entanglement geometry over the existence-passing directions (2A).
    passing = [f for f in features if existence[f].exceeds_null and f in directions]
    geometry: ent.EntanglementGeometry | None = None
    if len(passing) >= 2:
        # The logistic-vs-CAV cross-check. It has existed in `entanglement.py` since 2A was
        # built and was never called from the ladder, so the verdict logic had only the eigen
        # half of its evidence. It is genuinely independent of the Gram: a different
        # *definition* of the concept axis, not a different statistic on the same one.
        cav_disagreement: dict[str, float] = {}
        for f in passing:
            tr = feature_embeddings(controls.real, labels, f, train_ids)
            if len(np.unique(tr.y)) < 2:
                continue
            cav_disagreement[f] = ent.logistic_cav_disagreement(
                directions[f].w_unit, ent.cav_direction(tr)
            )
        geometry = ent.entanglement_geometry(
            [directions[f] for f in passing],
            controls.real.x,
            mp_method=config.mp_method,
            pair_quantile=config.entangled_pair_quantile,
            cav_disagreement=cav_disagreement,
        )
    entangled, pair_verdicts = _entangled_map(
        geometry, directions, controls, labels, train_ids, test_ids, config=config
    )

    # Phases 4 & 5: per-feature rung.
    verdicts: dict[str, RungVerdict] = {}
    for feature in features:
        fc = feature_controls[feature]
        cleared, matched = _nuisance_clearance(
            feature,
            fc,
            controls,
            labels,
            train_ids,
            test_ids,
            margin_established=existence[feature].exceeds_null,
            config=config,
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
                feature, fc, controls, labels, train_ids, test_ids, cleared, matched, config=config
            )

        # Power annotation, attached to every verdict rather than only the failures: a feature
        # that passed is entitled to say how thin a margin it could have resolved, and a reader
        # comparing R1s across buckets needs it as much as a reader of an R4.
        n_tr, pos_tr, n_te, pos_te = counts[feature]
        sd_bar = float(np.std(bank[feature], ddof=1)) if bank is not None else 0.0
        margin = nulls_mod.resolvable_margin(
            real_ses[feature],
            sd_bar=sd_bar,
            alpha=config.alpha,
            method=config.multiplicity,
            n_tests=n_tests,
            df=(len(bank[feature]) - 1) if bank is not None else None,
        )
        verdicts[feature] = dataclasses.replace(
            verdicts[feature],
            n_train=n_tr,
            positives_train=pos_tr,
            n_test=n_te,
            positives_test=pos_te,
            real_se=real_ses[feature],
            resolvable_margin=margin,
            underpowered=nulls_mod.is_underpowered(margin),
        )

    return LadderResult(
        verdicts=verdicts,
        existence=existence,
        entanglement=geometry,
        feature_controls=feature_controls,
        directions=directions,
        pair_verdicts=pair_verdicts,
    )
