"""``run_probing`` — the probing harness entry point (the phases, stamped end-to-end).

Takes a **frozen** encoder + a label-free dataset over the probing corpus + a
``LabelProvider``, and runs the full design: extract once → control sources → the gated
ladder → uncertainty geometry on the R1/R2 features → emit the figures → stamp every artefact.
It stays objective-free (the freeze boundary): it imports ``models`` for the untrained-encoder
control, never ``objectives``.

The data-layer wiring (DirectorySource → cache → StampDataset → metadata rows) is the caller's
job — ``harness.py`` already builds exactly those pieces post-freeze, and a thin wrapper hands
them here. Keeping ``run_probing`` parameterised on the *prepared* dataset + labels means the
whole battery is exercised in the integration tier with a synthetic encoder, no network.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from galaxy_jepa.core.config import RunStamp, write_stamp
from galaxy_jepa.core.encoder import Encoder, assert_frozen
from galaxy_jepa.data.manifest import manifest_hash
from galaxy_jepa.data.orchestrate import assign_three_way
from galaxy_jepa.probing import controls as ctl
from galaxy_jepa.probing import uncertainty as unc
from galaxy_jepa.probing.config import ProbingConfig
from galaxy_jepa.probing.extract import LabelProvider, extract_matrix
from galaxy_jepa.probing.ladder import LadderResult, run_ladder

__all__ = ["ProbingReport", "run_probing"]


@dataclasses.dataclass
class ProbingReport:
    """The deliverable: the ladder + uncertainty geometry + the emitted figure paths."""

    ladder: LadderResult
    uncertainty: dict[str, unc.UncertaintyGeometry]
    figures: dict[str, str]
    out_dir: str
    data_snapshot: str
    scheme: str | None = None
    family_size: int | None = None
    conditional: LadderResult | None = None

    def rung_table(self) -> dict[str, str]:
        """Feature → rung, the one-line story (Figure 1's content)."""
        return {f: v.rung for f, v in self.ladder.verdicts.items()}

    def population_comparison(self) -> dict[str, dict[str, Any]]:
        """Per-feature full-vs-conditional comparison — the D14 result, not a filtered rerun.

        Empty when the conditional ladder was not run. ``rung_changed`` is the headline: a
        feature whose verdict is the same either way says the tree's conditioning is cosmetic
        *for that feature*; one that changes says the population definition is doing real work.
        """
        if self.conditional is None:
            return {}
        out: dict[str, dict[str, Any]] = {}
        for feature, full in self.ladder.verdicts.items():
            cond = self.conditional.verdicts.get(feature)
            if cond is None:
                continue
            out[feature] = {
                "rung_full": full.rung,
                "rung_conditional": cond.rung,
                "rung_changed": full.rung != cond.rung,
                "auc_full": full.metrics.get("auc"),
                "auc_conditional": cond.metrics.get("auc"),
            }
        return out


def run_probing(
    encoder: Encoder,
    dataset: Dataset,
    labels: LabelProvider,
    model_config: dict[str, Any],
    *,
    config: ProbingConfig,
    out_dir: str | Path,
    sky_label_col: str = "snr_r",
    emit_figures: bool = True,
) -> ProbingReport:
    """Run the full probing battery on a frozen encoder and stamp the artefacts.

    ``dataset`` yields ``image`` + ``object_id`` per item (a ``StampDataset``); ``model_config``
    is the encoder's constructor record (``VisionTransformer.config``), used to build the
    untrained-encoder control. There is no "pick best AUC checkpoint" path — the checkpoint is
    chosen label-blind upstream (design 1C), so labels never select the encoder.
    """
    assert_frozen(encoder)  # the probing freeze boundary
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Phase 0: extract the real embeddings ONCE; split the ids the cache actually holds.
    real = extract_matrix(encoder, dataset, device=config.device or "cpu")
    ids = [int(o) for o in real.object_ids]
    split = assign_three_way(ids, seed=config.seed, ratios=config.ratios)
    train_ids, test_ids = sorted(split.train), sorted(split.test)

    # Phase 0b: the per-encoder control sources (untrained encoder; noise through real encoder).
    device = config.device or "cpu"
    untrained = ctl.untrained_encoder_matrix(model_config, dataset, device=device, seed=config.seed)
    noise = ctl.noise_through_encoder_matrix(encoder, dataset, device=device, seed=config.seed)
    controls = ctl.ControlEmbeddings(real=real, untrained=untrained, noise=noise)

    # The BY family size is per-scheme, never a global constant: an explicit config value wins,
    # else the active scheme's primary count, else the number of features actually probed.
    scheme = getattr(labels, "scheme", None)
    if config.n_primary_tests is not None:
        family_size = config.n_primary_tests
    elif scheme is not None:
        family_size = scheme.family_size()
    else:
        family_size = len(labels.features)
    config = config.model_copy(update={"n_primary_tests": family_size})

    # Phases 1–5: the gated cascade, on the full population.
    ladder = run_ladder(
        controls, labels, train_ids, test_ids, config=config, sky_label_col=sky_label_col
    )

    # Phase 5b: the same ladder again on the consensus-conditional population (D14). This is a
    # *comparison*, so both ladders are kept and neither filters the other — the off-population
    # galaxies stay in the full run, where a concentrated disagreement is itself a finding.
    conditional: LadderResult | None = None
    if scheme is not None and config.compare_populations and labels.population == "full":
        conditional = run_ladder(
            controls,
            labels.with_population("conditional"),
            train_ids,
            test_ids,
            config=config,
            sky_label_col=sky_label_col,
        )

    # Phase 6: uncertainty geometry on the R1/R2 features only (4B) — gated on recoverability.
    uncertainty: dict[str, unc.UncertaintyGeometry] = {}
    for feature, verdict in ladder.verdicts.items():
        if verdict.rung in ("R1", "R2"):
            uncertainty[feature] = unc.uncertainty_geometry(
                real,
                labels,
                feature,
                ids,
                low=config.extreme_low,
                high=config.extreme_high,
                c=config.c,
                n_perm=config.n_perm,
                method=config.permutation_method,
                seed=config.seed,
            )

    # Phase 7: figures + stamped artefacts.
    figures: dict[str, str] = {}
    if emit_figures:
        figures = _emit_figures(ladder, uncertainty, out)
    report = ProbingReport(
        ladder=ladder,
        uncertainty=uncertainty,
        figures=figures,
        out_dir=str(out),
        data_snapshot=manifest_hash(ids, f"probe|seed={config.seed}|ratios={config.ratios}"),
        scheme=scheme.name if scheme is not None else None,
        family_size=family_size,
        conditional=conditional,
    )
    _write_summary(
        ladder,
        uncertainty,
        out,
        scheme=report.scheme,
        family_size=family_size,
        comparison=report.population_comparison(),
    )

    stamp = RunStamp.create(
        config.model_dump(mode="json"),
        data_snapshot=report.data_snapshot,
        seed=config.seed,
        # A declared deviation and a smoke marking both land in the power-path ledger, so an
        # artefact carries what it forfeited rather than looking like a clean run.
        escape_hatches_used=(
            [*config.escape_hatches]
            + (["smoke"] if config.smoke else [])
            # An unfrozen floor is a forfeited guarantee, not a neutral default: the run's
            # clean-vs-marginal line was set by a placeholder. Say so on the artefact.
            + ([] if config.effect_floor_freeze is not None else ["effect_floor_open"])
        )
        or None,
    )
    write_stamp(stamp, out, config.model_dump(mode="json"))
    return report


def _emit_figures(
    ladder: LadderResult, uncertainty: dict[str, unc.UncertaintyGeometry], out: Path
) -> dict[str, str]:
    """Emit the three headline figures + the control figures (best-effort; matplotlib lazy)."""
    from galaxy_jepa.eval import figures as fig

    figdir = out / "figures"
    emitted: dict[str, str] = {}
    try:
        emitted["ladder"] = str(fig.figure_ladder(ladder, figdir / "fig1_ladder.png"))
        if uncertainty:
            emitted["uncertainty"] = str(
                fig.figure_uncertainty(uncertainty, figdir / "fig2_uncertainty.png")
            )
        if ladder.entanglement is not None:
            emitted["entanglement"] = str(
                fig.figure_entanglement(ladder.entanglement, figdir / "fig3_entanglement.png")
            )
        emitted["controls"] = str(fig.figure_controls(ladder, figdir / "controls.png"))
    except Exception as exc:  # figures are a deliverable, not a gate — never abort the run
        emitted["error"] = f"{type(exc).__name__}: {exc}"
    return emitted


def _write_summary(
    ladder: LadderResult,
    uncertainty: dict[str, unc.UncertaintyGeometry],
    out: Path,
    *,
    scheme: str | None = None,
    family_size: int | None = None,
    comparison: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Persist the rung table + the gate verdict trees + uncertainty stats as JSON."""
    summary = {
        "rungs": {
            f: {
                "rung": v.rung,
                "mechanism": v.mechanism,
                "metrics": v.metrics,
                "gate_tree": v.gate_tree.render(),
            }
            for f, v in ladder.verdicts.items()
        },
        "existence": {
            f: {"real_auc": e.real_auc, "pvalue": e.pvalue, "exceeds_null": e.exceeds_null}
            for f, e in ladder.existence.items()
        },
        # Which control set the bar. Without this the verdict is unreadable: "p = 1.0" says the
        # feature lost, not *what to*, and the answer (an untrained encoder vs a shuffled label)
        # is a different scientific statement each time.
        "nulls": {
            f: {
                "shuffled_max": float(c.shuffled_nulls.max()) if c.shuffled_nulls.size else None,
                "random_embedding_max": (
                    float(c.random_embedding_nulls.max()) if c.random_embedding_nulls.size else None
                ),
                "noise_encoder": c.noise_encoder_auc,
                "untrained_encoder": c.untrained_encoder_auc,
                "sky_noise": c.sky_noise_auc,
                "selectivity": c.selectivity,
                "nuisance_aucs": dict(c.nuisance_aucs),
            }
            for f, c in ladder.feature_controls.items()
        },
        "uncertainty": {
            f: {"spearman": u.spearman, "pvalue": u.pvalue, "n_middle": u.n_middle}
            for f, u in uncertainty.items()
        },
        "scheme": scheme,
        "by_family_size": family_size,
        "population_comparison": comparison or {},
    }
    path = out / "ladder_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path
