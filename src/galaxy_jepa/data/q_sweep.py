"""asinh Q-sweep — measure the stretch trade-off as a curve (docs/spec/data.md §2.3).

The asinh softening ``Q`` is a **constrained trade-off**, not a maximisation: as ``Q``
rises, faint structure retained goes *up* and the sky-noise floor goes *up* together.
Maximising "retention" alone walks ``Q`` into the noisy regime, because amplified sky
noise counts as retained "signal". So this sweep reports **two opposite-trending numbers
per Q** — annulus faint-retention and corner sky-floor (``sanity.galaxy_zone_metrics``)
— plus their **gap** (``faint - sky``), the honest signal-above-noise quantity: lines
*diverging* = real signal, *tracking together* = amplified noise.

Normalisation is fitted **post-stretch and re-fit per Q** (it interacts with ``Q``; a
single fit would tangle ``Q`` with statistics that suit only one ``Q``). The flux scale
is held fixed across the sweep so ``Q`` is isolated.

This phase **hands over a curve** — it does not pick or freeze ``Q`` (the user sets the
sky-noise ceiling and chooses the ``Q`` a follow-up freezes).

CLI (runs in the devcontainer, on a pulled corpus)::

    python -m galaxy_jepa.data.q_sweep --corpus data/probe_2k --out artifacts/q_sweep \
        --q 1,2,3,4,6,8,12 --contact-q 2,4,8 --contact-corpus data/probe_slice
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from galaxy_jepa.data.bbox import DEFAULT_K
from galaxy_jepa.data.sanity import galaxy_zone_metrics
from galaxy_jepa.data.sources import NATIVE_PIXEL_SCALE, DataSource, DirectorySource
from galaxy_jepa.data.transforms import AsinhStretch, Normalise, Pipeline

logger = logging.getLogger(__name__)

Array = np.ndarray
DEFAULT_Q_GRID: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
DEFAULT_CONTACT_Q: tuple[float, ...] = (2.0, 4.0, 8.0)


@dataclass(frozen=True)
class QRecord:
    """Aggregated metrics at one ``Q`` (median + IQR across the sample)."""

    q: float
    faint_med: float
    faint_lo: float
    faint_hi: float
    sky_med: float
    sky_lo: float
    sky_hi: float
    gap_med: float
    gap_lo: float
    gap_hi: float
    n_faint_valid: int
    n_total: int


def _radius(meta: dict[str, Any]) -> float | None:
    for key in ("petroRad_r", "petroRad"):
        if key in meta:
            return meta[key]
    return None


def _med_iqr(values: list[float]) -> tuple[float, float, float]:
    """Median, 25th, 75th percentile — or NaNs if empty."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return (
        float(np.median(arr)),
        float(np.percentile(arr, 25)),
        float(np.percentile(arr, 75)),
    )


def _sweep_one_q(
    pairs: list[tuple[Array, dict[str, Any]]],
    q: float,
    flux_scale: tuple[float, ...],
    k: float,
    pixel_scale: float,
) -> QRecord:
    """Metrics at a single ``Q`` (normalisation re-fit on *this* ``Q``'s stretched data)."""
    stretch = AsinhStretch(q=float(q), flux_scale=flux_scale)
    stretched = np.stack([stretch(image) for image, _ in pairs])
    norm = Normalise.fit(stretched)  # re-fit per Q — the requirement
    pipeline = Pipeline((stretch, norm))

    faint: list[float] = []
    sky: list[float] = []
    gap: list[float] = []
    for image, meta in pairs:
        zm = galaxy_zone_metrics(
            pipeline(image), _radius(meta), pixel_scale,
            k=k, object_id=meta.get("object_id"),
        )
        sky.append(zm.sky_floor)
        if zm.faint_valid:
            faint.append(zm.faint_retention)
            gap.append(zm.faint_retention - zm.sky_floor)

    fm, flo, fhi = _med_iqr(faint)
    sm, slo, shi = _med_iqr(sky)
    gm, glo, ghi = _med_iqr(gap)
    logger.info(
        "Q=%-4g faint=%.4f sky=%.4f gap=%.4f (%d/%d valid)",
        q, fm, sm, gm, len(faint), len(pairs),
    )
    return QRecord(float(q), fm, flo, fhi, sm, slo, shi, gm, glo, ghi, len(faint), len(pairs))


def sweep_q(
    source: DataSource,
    q_grid: Sequence[float] = DEFAULT_Q_GRID,
    *,
    flux_scale: tuple[float, ...] = (1.0, 1.0, 1.0),
    k: float = DEFAULT_K,
    pixel_scale: float = NATIVE_PIXEL_SCALE,
    workers: int = 4,
) -> list[QRecord]:
    """Sweep ``Q``, re-fitting normalisation per ``Q``, returning per-``Q`` aggregates.

    The `Q` values are independent, so they run on a thread pool — the heavy numpy ops
    (``arcsinh``/``median``/``percentile``) release the GIL, giving real parallelism with
    no pickling (the image list is shared). The per-galaxy **gap** ``faint - sky`` uses
    each galaxy's own sky floor and is aggregated only over galaxies with a valid annulus.
    """
    pairs: list[tuple[Array, dict[str, Any]]] = [source[i] for i in range(len(source))]
    if not pairs:
        raise ValueError("empty corpus — nothing to sweep")

    n_workers = max(1, min(workers, len(q_grid)))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        records = list(
            pool.map(lambda q: _sweep_one_q(pairs, q, flux_scale, k, pixel_scale), q_grid)
        )
    return sorted(records, key=lambda r: r.q)


def write_metrics_csv(records: Sequence[QRecord], out_path: str | Path) -> Path:
    """Write the per-``Q`` table as CSV."""
    out_path = Path(out_path)
    cols = [f.name for f in fields(QRecord)]
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(cols)
        for rec in records:
            writer.writerow([getattr(rec, c) for c in cols])
    return out_path


def write_curve(records: Sequence[QRecord], out_path: str | Path) -> Path:
    """Plot faint-retention, sky-floor and their gap vs ``Q`` (one axis — same units).

    All three are in normalised-flux units, so the **gap is literally the vertical
    distance** between the retention and sky-floor lines — divergence is visible directly.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qs = [r.q for r in records]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.plot(qs, [r.faint_med for r in records], "o-", color="tab:blue",
            label="faint-retention (annulus)")
    ax.fill_between(qs, [r.faint_lo for r in records], [r.faint_hi for r in records],
                    color="tab:blue", alpha=0.15)
    ax.plot(qs, [r.sky_med for r in records], "s-", color="tab:red",
            label="sky-noise floor (corner MAD)")
    ax.fill_between(qs, [r.sky_lo for r in records], [r.sky_hi for r in records],
                    color="tab:red", alpha=0.15)
    ax.plot(qs, [r.gap_med for r in records], "D--", color="tab:green",
            label="gap = signal above noise")

    ax.set_xlabel("asinh Q (softening)")
    ax.set_ylabel("post stretch+normalise (normalised-flux units)")
    ax.set_title("asinh Q-sweep — signal (gap) vs noise floor; nothing frozen")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def elbow_readout(records: Sequence[QRecord]) -> str:
    """Plain-text readout framed on the **gap** (signal above noise), not retention height.

    Reports where the gap plateaus (diminishing real-signal returns), where the sky-floor
    climbs steeply, and flags any ``Q`` where retention rises but the gap does not
    (amplified noise, the suspect case). Decides nothing.
    """
    if not records:
        return "no records"
    recs = sorted(records, key=lambda r: r.q)
    lines = ["Q-sweep readout — the GAP (faint − sky) is the signal, not retention height.", ""]
    lines.append(f"{'Q':>5} {'faint':>9} {'sky':>9} {'gap':>9}  valid/total")
    for r in recs:
        lines.append(
            f"{r.q:>5g} {r.faint_med:>9.4f} {r.sky_med:>9.4f} {r.gap_med:>9.4f}"
            f"  {r.n_faint_valid}/{r.n_total}"
        )
    lines.append("")

    # Gap plateau: the largest Q beyond which the gap gains < 10% of its biggest step.
    gap_steps = [recs[i + 1].gap_med - recs[i].gap_med for i in range(len(recs) - 1)]
    if gap_steps:
        max_step = max(abs(s) for s in gap_steps) or 1.0
        plateau_q = recs[-1].q
        for i, step in enumerate(gap_steps):
            if step < 0.10 * max_step:
                plateau_q = recs[i].q
                break
        lines.append(
            f"- Gap plateaus around Q≈{plateau_q:g}: beyond it, real signal-above-noise "
            "barely grows (diminishing returns)."
        )

    # Sky-floor steep climb: first Q whose incremental sky rise exceeds the median step.
    sky_steps = [recs[i + 1].sky_med - recs[i].sky_med for i in range(len(recs) - 1)]
    if sky_steps:
        typical = float(np.median([abs(s) for s in sky_steps])) or 0.0
        climb_q = None
        for i, step in enumerate(sky_steps):
            if step > 2.0 * typical and step > 0:
                climb_q = recs[i + 1].q
                break
        if climb_q is not None:
            lines.append(f"- Sky-noise floor starts climbing steeply at Q≈{climb_q:g}.")
        else:
            lines.append("- Sky-noise floor rises smoothly across the grid (no sharp knee).")

    # Suspect Q: retention up vs previous, but gap flat or down ⇒ the rise is noise.
    suspect = [
        recs[i + 1].q
        for i in range(len(recs) - 1)
        if recs[i + 1].faint_med > recs[i].faint_med
        and recs[i + 1].gap_med <= recs[i].gap_med
    ]
    if suspect:
        lines.append(
            "- SUSPECT (retention rises but gap does not — amplified noise): "
            f"Q={', '.join(f'{q:g}' for q in suspect)}."
        )
    else:
        lines.append("- No suspect Q: every retention rise is matched by a gap rise.")

    lines.append("")
    lines.append(
        "Decision (yours): pick the highest Q whose sky-floor stays under your ceiling "
        "while the gap is still near its plateau. Nothing is frozen here."
    )
    return "\n".join(lines)


def _fit_pipeline(source: DataSource, q: float, flux_scale: tuple[float, ...]) -> Pipeline:
    stretch = AsinhStretch(q=float(q), flux_scale=flux_scale)
    stretched = np.stack([stretch(source[i][0]) for i in range(len(source))])
    return Pipeline((stretch, Normalise.fit(stretched)))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="asinh Q-sweep — measured stretch curve.")
    parser.add_argument("--corpus", type=Path, required=True,
                        help="DirectorySource corpus for the sweep")
    parser.add_argument("--out", type=Path, required=True,
                        help="output dir for curve/csv/sheets/readout")
    parser.add_argument("--q", default=",".join(f"{q:g}" for q in DEFAULT_Q_GRID),
                        help="comma-separated Q grid")
    parser.add_argument("--contact-q", default=",".join(f"{q:g}" for q in DEFAULT_CONTACT_Q),
                        help="comma-separated Q values for contact sheets")
    parser.add_argument("--contact-corpus", type=Path,
                        help="corpus for contact sheets (defaults to --corpus)")
    parser.add_argument("--flux-scale", default="1,1,1",
                        help="per-channel flux scale (fixed across sweep)")
    args = parser.parse_args(argv)

    q_grid = tuple(float(x) for x in args.q.split(","))
    flux_scale = tuple(float(x) for x in args.flux_scale.split(","))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    source = DirectorySource(args.corpus)
    records = sweep_q(source, q_grid, flux_scale=flux_scale)

    write_metrics_csv(records, out / "metrics.csv")
    write_curve(records, out / "curve.png")
    readout = elbow_readout(records)
    (out / "readout.txt").write_text(readout + "\n")
    print("\n" + readout + "\n")

    # Contact sheets at a few Q for visual cross-check (reuse the eyeball-gate renderer).
    from galaxy_jepa.data.contact_sheet import build_contact_sheet

    contact_root = args.contact_corpus or args.corpus
    contact_source = DirectorySource(contact_root)
    for q in (float(x) for x in args.contact_q.split(",")):
        pipeline = _fit_pipeline(contact_source, q, flux_scale)
        build_contact_sheet(contact_source, pipeline, out / f"sheet_q{q:g}.png")
    logger.info("wrote curve.png, metrics.csv, readout.txt and %s contact sheets to %s",
                len(args.contact_q.split(",")), out)


if __name__ == "__main__":
    main()
