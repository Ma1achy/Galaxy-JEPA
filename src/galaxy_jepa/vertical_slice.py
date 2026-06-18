"""End-to-end vertical slice — pull → split → bake → pretrain → probe → figures.

The runnable wiring for the slice plan (one AUC + collapse trace + UMAP, then stop). It
sits on the pieces built around it and runs the strict sequence on the Mac (MPS) or any
device:

1. resolve the two corpora through the dedup guard, assign the probe three-way, carve the
   monitor slice (``data/orchestrate``);
2. fit the normalisation **once** on a pretrain subsample, freeze it, bake the parity-locked
   fp16 cache for both corpora under that one frozen pipeline (``data/cache``);
3. pretrain the from-scratch JEPA with the collapse monitor live (``objectives/jepa``);
4. reload the encoder frozen and run the L2-logistic probe on the confident extremes
   (``probing/logistic``) — the one headline AUC;
5. write the collapse-trace and UMAP figures (``eval/embed``) and a report JSON.

:func:`calibrate` is the cheap pre-flight: it times the dataloader against the model step
and reports the **disk-bound vs compute-bound** verdict (renting a GPU only helps the
compute-bound case — the slice plan's decision gate).

The pull itself is ``data/pull.py`` (networked); this module consumes the resulting
``DirectorySource`` corpora, so the whole orchestration is testable offline on fixtures.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from galaxy_jepa.data.cache import bake_cache, fit_normalise
from galaxy_jepa.data.dataset import StampDataset, rows_by_id
from galaxy_jepa.data.metadata import FEATURED_FRACTION_COL
from galaxy_jepa.data.orchestrate import (
    assign_three_way,
    resolve_corpora,
    split_pretrain,
    write_split_plan,
)
from galaxy_jepa.data.sources import DirectorySource
from galaxy_jepa.data.transforms import AsinhStretch, Pipeline
from galaxy_jepa.models.vit import VisionTransformer, load_frozen_encoder
from galaxy_jepa.objectives.jepa import Jepa, JepaConfig, _to_device, train_jepa
from galaxy_jepa.probing.logistic import Embeddings, extract_embeddings, probe_auc

logger = logging.getLogger(__name__)


def pick_device() -> str:
    """Prefer MPS on the Mac, else CUDA, else CPU (the slice runs on the Mac by default)."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclasses.dataclass
class SliceReport:
    """The deliverable: the headline AUC + the two gut-checks + provenance."""

    auc: float | None
    n_train: int
    n_test: int
    halted: bool
    final_loss: float | None
    checkpoint: str | None
    collapse_png: str | None
    umap_png: str | None
    note: str = ""

    def go_no_go(self) -> str:
        """A plain read of the result (positive / null-but-inconclusive / collapse)."""
        if self.halted:
            return "COLLAPSE — representation collapsed; a finding, not a pass (see trace + tries)."
        if self.auc is None:
            return f"INCONCLUSIVE — probe AUC not computed ({self.note})."
        if self.auc >= 0.6:
            return f"GO — probe AUC {self.auc:.3f} clearly above chance; the premise is alive."
        return (
            f"NULL (inconclusive) — probe AUC {self.auc:.3f} near chance. Not proof the premise "
            "is dead: at this scale it may be undertraining. Next: scale / warm-start."
        )


def _build_pipeline(
    pretrain_source: DirectorySource, *, q: float, n_sample: int, seed: int
) -> Pipeline:
    # fit normalisation ONCE on a pretrain subsample and freeze BEFORE any training, so the
    # pilot and the full run share one pipeline_hash and the cache tops up incrementally.
    stretch = AsinhStretch(q=q)
    norm = fit_normalise(pretrain_source, stretch, n_sample=n_sample, seed=seed)
    return Pipeline((stretch, norm))


@dataclasses.dataclass
class _Prepared:
    """The shared setup the full run and the calibration pre-flight both need."""

    out: Path
    device: str
    jepa: Jepa
    loader: DataLoader
    monitor_batch: dict[str, Any]
    cache: Any
    rows: dict[int, dict[str, Any]]
    probe_split: Any


def _prepare(
    pretrain_dir: str | Path,
    probe_dir: str | Path,
    out_dir: str | Path,
    *,
    config: JepaConfig,
    device: str,
    seed: int,
    q: float,
    norm_sample: int,
    monitor_frac: float,
    model_kwargs: dict[str, Any] | None,
) -> _Prepared:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info("vertical slice on device=%s, out=%s", device, out)

    pre_src = DirectorySource(pretrain_dir)
    probe_src = DirectorySource(probe_dir)
    pre_ids = [int(r["object_id"]) for r in pre_src.rows]
    probe_ids = [int(r["object_id"]) for r in probe_src.rows]

    # 1. splits — dedup guard exercised, monitor slice carved, probe split assigned
    deduped = resolve_corpora(pre_ids, probe_ids)
    pre_split = split_pretrain(deduped, seed=seed, monitor_frac=monitor_frac)
    probe_split = assign_three_way(probe_ids, seed=seed)
    write_split_plan(
        out / "split_plan.json",
        probe=probe_split,
        pretrain=pre_split,
        seed=seed,
        ratios=(0.70, 0.15, 0.15),
    )

    # 2. one frozen pipeline, baked into a shared hash-keyed cache (incremental top-up)
    pipeline = _build_pipeline(pre_src, q=q, n_sample=norm_sample, seed=seed)
    bake_cache(pre_src, pipeline, out / "cache")
    cache = bake_cache(probe_src, pipeline, out / "cache")  # same hash dir → appends probe
    rows = rows_by_id([*pre_src.rows, *probe_src.rows])

    # 3. encoder + loaders sized to the baked stamp
    stamp_px = cache.index.height
    encoder = VisionTransformer(img_size=stamp_px, **(model_kwargs or {}))
    jepa = Jepa(encoder, config)
    train_ds = StampDataset(cache, rows, sorted(pre_split.train))
    loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False)
    monitor_ds = StampDataset(cache, rows, sorted(pre_split.monitor) or sorted(pre_split.train))
    monitor_batch = next(iter(DataLoader(monitor_ds, batch_size=min(64, len(monitor_ds) or 1))))
    return _Prepared(out, device, jepa, loader, monitor_batch, cache, rows, probe_split)


def run_slice(
    pretrain_dir: str | Path,
    probe_dir: str | Path,
    out_dir: str | Path,
    *,
    config: JepaConfig | None = None,
    device: str | None = None,
    seed: int = 0,
    q: float = 4.0,
    norm_sample: int = 8000,
    monitor_frac: float = 0.02,
    model_kwargs: dict[str, Any] | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> SliceReport:
    """Run the whole slice on two pulled corpora; return the report and write artefacts."""
    config = config or JepaConfig()
    device = device or pick_device()
    prep = _prepare(
        pretrain_dir,
        probe_dir,
        out_dir,
        config=config,
        device=device,
        seed=seed,
        q=q,
        norm_sample=norm_sample,
        monitor_frac=monitor_frac,
        model_kwargs=model_kwargs,
    )
    out, cache, rows, probe_split = prep.out, prep.cache, prep.rows, prep.probe_split

    # pretrain with the collapse monitor live
    result = train_jepa(
        prep.jepa,
        prep.loader,
        device=device,
        monitor_batch=prep.monitor_batch,
        checkpoint_path=out / "encoder.pt",
        autocast_dtype=autocast_dtype,
    )

    # 4. freeze + probe the confident extremes → the one number (+ embeddings for the UMAP)
    auc: float | None = None
    n_train = n_test = 0
    note = ""
    umap_png: str | None = None
    if not result.halted and result.checkpoint is not None:
        frozen = load_frozen_encoder(result.checkpoint)
        train_probe = StampDataset(
            cache, rows, sorted(probe_split.train), label_fraction_col=FEATURED_FRACTION_COL
        )
        test_probe = StampDataset(
            cache, rows, sorted(probe_split.test), label_fraction_col=FEATURED_FRACTION_COL
        )
        train_emb = _extremes(extract_embeddings(frozen, train_probe, device=device))
        test_emb = _extremes(extract_embeddings(frozen, test_probe, device=device))
        try:
            auc = probe_auc(train_emb, test_emb)
            n_train, n_test = len(train_emb.y), len(test_emb.y)
        except ValueError as exc:  # too few confident extremes / single class in a tiny run
            note = f"probe skipped: {exc}"
            logger.warning(note)
        umap_png = _safe_umap_plot(test_emb, out / "umap.png")  # gut-check 1

    # 5. figures + report (collapse trace = gut-check 2)
    collapse_png = _safe_collapse_plot(result.collapse_trace, out / "collapse_trace.png")

    report = SliceReport(
        auc=auc,
        n_train=n_train,
        n_test=n_test,
        halted=result.halted,
        final_loss=result.losses[-1] if result.losses else None,
        checkpoint=str(result.checkpoint) if result.checkpoint else None,
        collapse_png=collapse_png,
        umap_png=umap_png,
        note=note,
    )
    (out / "report.json").write_text(
        json.dumps({**dataclasses.asdict(report), "go_no_go": report.go_no_go()}, indent=2) + "\n"
    )
    logger.info("slice report: %s", report.go_no_go())
    return report


def _extremes(emb: Embeddings, *, low: float = 0.2, high: float = 0.8) -> Embeddings:
    keep = (emb.fraction <= low) | (emb.fraction >= high)
    return Embeddings(emb.x[keep], emb.y[keep], emb.fraction[keep])


def _safe_collapse_plot(trace: dict[str, list[float]], path: Path) -> str | None:
    if not trace.get("std"):
        return None
    from galaxy_jepa.eval.embed import plot_collapse_trace

    return str(plot_collapse_trace(trace, path))


def _safe_umap_plot(test_emb: Embeddings, path: Path) -> str | None:
    """UMAP of the frozen probe-test embeddings, coloured by label — skipped if umap absent."""
    if len(test_emb.y) < 3 or len(set(test_emb.y.tolist())) < 2:
        return None
    try:
        from galaxy_jepa.eval.embed import plot_umap

        return str(plot_umap(test_emb.x, test_emb.y, path))
    except ImportError:  # pragma: no cover - umap is an eval extra
        logger.warning("umap-learn not installed; skipping the UMAP gut-check")
        return None


@dataclasses.dataclass
class CalibrationResult:
    """The pre-flight verdict that gates the full run (disk- vs compute-bound)."""

    data_ms: float
    step_ms: float
    bound: str  # "disk-bound" | "compute-bound"
    it_per_s: float

    def verdict(self) -> str:
        if self.bound == "disk-bound":
            return (
                f"DISK-BOUND (dataloader {self.data_ms:.1f}ms > step {self.step_ms:.1f}ms): a "
                "bigger batch or a rented GPU will NOT help — shrink the working set to fit RAM."
            )
        return (
            f"COMPUTE-BOUND (step {self.step_ms:.1f}ms > dataloader {self.data_ms:.1f}ms): a "
            f"bigger batch / a rented 4090 cut wall-clock. ~{self.it_per_s:.1f} it/s here."
        )


def calibrate(
    jepa: Jepa,
    loader: DataLoader,
    *,
    device: str | None = None,
    steps: int = 50,
    autocast_dtype: torch.dtype | None = None,
) -> CalibrationResult:
    """Time the dataloader against the model step and classify the bottleneck (slice plan)."""
    device = device or pick_device()
    jepa.to(device)
    opt = torch.optim.AdamW([*jepa.encoder.parameters(), *jepa.predictor.parameters()], lr=1e-3)
    data_times, step_times = [], []
    it = iter(loader)
    for i in range(steps):
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = _to_device(batch, device)  # casts float64 scalars to float32 (MPS-safe)
        _sync(device)
        t1 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        if autocast_dtype is not None:
            with torch.autocast(device_type=device.split(":")[0], dtype=autocast_dtype):
                loss = jepa.loss_step(batch, seed=i)
        else:
            loss = jepa.loss_step(batch, seed=i)
        loss.backward()
        opt.step()
        _sync(device)
        t2 = time.perf_counter()
        if i >= 2:  # skip warm-up iterations
            data_times.append((t1 - t0) * 1e3)
            step_times.append((t2 - t1) * 1e3)

    data_ms = statistics.median(data_times) if data_times else 0.0
    step_ms = statistics.median(step_times) if step_times else 0.0
    bound = "disk-bound" if data_ms > step_ms else "compute-bound"
    it_per_s = 1000.0 / max(data_ms + step_ms, 1e-6)
    result = CalibrationResult(data_ms=data_ms, step_ms=step_ms, bound=bound, it_per_s=it_per_s)
    logger.info(result.verdict())
    return result


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


def main(argv: list[str] | None = None) -> None:
    """CLI: run the slice (or just the calibration pre-flight) on two pulled corpora.

    Examples::

        python -m galaxy_jepa.vertical_slice --pretrain data/pretrain --probe data/probe \\
            --out artifacts/slice --calibrate
        python -m galaxy_jepa.vertical_slice --pretrain data/pretrain --probe data/probe \\
            --out artifacts/slice --steps 50000 --batch-size 256 --beta 0.5 --bf16
    """
    import argparse

    from galaxy_jepa.masking.blocks import MaskConfig

    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Galaxy-JEPA vertical slice (one AUC + gut-checks).")
    p.add_argument("--pretrain", required=True, type=Path)
    p.add_argument("--probe", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--bf16", action="store_true", help="bf16 autocast (MPS/CUDA throughput)")
    p.add_argument(
        "--calibrate",
        action="store_true",
        help="time data vs step, print the disk- vs compute-bound verdict, then exit",
    )
    p.add_argument("--calibrate-steps", type=int, default=50)
    args = p.parse_args(argv)

    cfg = JepaConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        mask=MaskConfig(beta=args.beta),
        seed=args.seed,
    )
    autocast = torch.bfloat16 if args.bf16 else None
    device = args.device or pick_device()

    if args.calibrate:
        prep = _prepare(
            args.pretrain,
            args.probe,
            args.out,
            config=cfg,
            device=device,
            seed=args.seed,
            q=4.0,
            norm_sample=8000,
            monitor_frac=0.02,
            model_kwargs=None,
        )
        result = calibrate(
            prep.jepa,
            prep.loader,
            device=device,
            steps=args.calibrate_steps,
            autocast_dtype=autocast,
        )
        print(result.verdict())
        return

    report = run_slice(
        args.pretrain,
        args.probe,
        args.out,
        config=cfg,
        device=device,
        seed=args.seed,
        autocast_dtype=autocast,
    )
    print(report.go_no_go())


if __name__ == "__main__":
    main()
