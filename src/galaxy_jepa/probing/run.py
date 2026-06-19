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

    def rung_table(self) -> dict[str, str]:
        """Feature → rung, the one-line story (Figure 1's content)."""
        return {f: v.rung for f, v in self.ladder.verdicts.items()}


def run_probing(
    encoder: Encoder,
    dataset: Dataset,
    labels: LabelProvider,
    model_config: dict[str, Any],
    *,
    config: ProbingConfig,
    out_dir: str | Path,
    sky_label_col: str = "snr",
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
    untrained = ctl.untrained_encoder_matrix(model_config, dataset, device=device)
    noise = ctl.noise_through_encoder_matrix(encoder, dataset, device=device, seed=config.seed)
    controls = ctl.ControlEmbeddings(real=real, untrained=untrained, noise=noise)

    # Phases 1–5: the gated cascade.
    ladder = run_ladder(
        controls, labels, train_ids, test_ids, config=config, sky_label_col=sky_label_col
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
    _write_summary(ladder, uncertainty, out)

    data_snapshot = manifest_hash(ids, f"probe|seed={config.seed}|ratios={config.ratios}")
    stamp = RunStamp.create(
        config.model_dump(mode="json"), data_snapshot=data_snapshot, seed=config.seed
    )
    write_stamp(stamp, out, config.model_dump(mode="json"))

    return ProbingReport(
        ladder=ladder,
        uncertainty=uncertainty,
        figures=figures,
        out_dir=str(out),
        data_snapshot=data_snapshot,
    )


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
    ladder: LadderResult, uncertainty: dict[str, unc.UncertaintyGeometry], out: Path
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
        "uncertainty": {
            f: {"spearman": u.spearman, "pvalue": u.pvalue, "n_middle": u.n_middle}
            for f, u in uncertainty.items()
        },
    }
    path = out / "ladder_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path
