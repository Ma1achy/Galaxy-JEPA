"""Reusable train → freeze → probe → figures/artifacts harness (post-slice consolidation).

The vertical slice proved the premise; this is the slice's wiring lifted out of a one-off
script into a **config-driven, stamped** entrypoint the full run, the baselines, and the
ablations all reuse. One :class:`HarnessConfig` fully determines a run; :func:`run_harness`
executes it and stamps every artefact with its provenance (``core.config.RunStamp``).

What it adds over the original ``vertical_slice`` script:

* a typed, validated :class:`HarnessConfig` (objective / model scale / probe label all
  config-driven) that serialises into the run stamp;
* :func:`build_objective` — the **single switch point** a future MAE/MoCo baseline edits
  (no name-keyed registry until a second objective actually lands — the repo's
  "second consumer" rule);
* :func:`evaluate_probe` — a probe-only re-evaluation on an existing frozen checkpoint
  (no retraining), used to regenerate a report's headline AUC + bootstrap CI;
* web-ready static **explorer blobs** (embedding index, the fitted concept direction, UMAP
  coordinates) persisted alongside the figures, per ``docs/embedding-explorer.md`` — so the
  embedding explorer is a clean downstream consumer, not a retrofit.

``vertical_slice`` is now a thin preset over this module.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import Field
from torch.utils.data import DataLoader

from galaxy_jepa.core.config import RunConfig, RunStamp, write_stamp
from galaxy_jepa.data.cache import TensorCache, bake_cache, fit_normalise
from galaxy_jepa.data.dataset import StampDataset, rows_by_id
from galaxy_jepa.data.manifest import manifest_hash
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
from galaxy_jepa.probing.logistic import (
    Embeddings,
    ProbeResult,
    _extremes,
    extract_embeddings,
    probe_auc_ci,
    probe_direction,
)

logger = logging.getLogger(__name__)

_AUTOCAST = {"bf16": torch.bfloat16, "fp16": torch.float16}


def pick_device() -> str:
    """Prefer MPS on the Mac, else CUDA, else CPU (the slice runs on the Mac by default)."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# --- the typed run config (HarnessConfig's first real consumer is the slice preset) -------


class ModelConfig(RunConfig):
    """Encoder scale. ``img_size`` is not here — it is fixed by the baked stamp at build time."""

    patch_size: int = 16
    embed_dim: int = 384
    depth: int = 12
    heads: int = 6
    mlp_ratio: float = 4.0

    def model_kwargs(self) -> dict[str, Any]:
        return self.model_dump()


class ObjectiveConfig(RunConfig):
    """Objective hyperparameters — a flat, serialisable mirror of ``JepaConfig`` (+ mask β)."""

    steps: int = 1000
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.04
    warmup_steps: int = 100
    ema_start: float = 0.996
    ema_end: float = 1.0
    pred_dim: int = 192
    pred_depth: int = 6
    pred_heads: int = 6
    beta: float = 0.5  # the headline masking-bias knob (mask.beta); 0 ≡ standard I-JEPA
    petro_k: float = 2.5
    global_box_frac: float = 0.40
    monitor_every: int = 100

    def to_jepa_config(self, *, seed: int) -> JepaConfig:
        from galaxy_jepa.masking.blocks import MaskConfig

        return JepaConfig(
            steps=self.steps,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            ema_start=self.ema_start,
            ema_end=self.ema_end,
            pred_dim=self.pred_dim,
            pred_depth=self.pred_depth,
            pred_heads=self.pred_heads,
            mask=MaskConfig(beta=self.beta),
            petro_k=self.petro_k,
            global_box_frac=self.global_box_frac,
            monitor_every=self.monitor_every,
            seed=seed,
        )

    @classmethod
    def from_jepa_config(cls, cfg: JepaConfig) -> ObjectiveConfig:
        return cls(
            steps=cfg.steps,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            warmup_steps=cfg.warmup_steps,
            ema_start=cfg.ema_start,
            ema_end=cfg.ema_end,
            pred_dim=cfg.pred_dim,
            pred_depth=cfg.pred_depth,
            pred_heads=cfg.pred_heads,
            beta=cfg.mask.beta,
            petro_k=cfg.petro_k,
            global_box_frac=cfg.global_box_frac,
            monitor_every=cfg.monitor_every,
        )


class ProbeConfig(RunConfig):
    """The frozen-probe read-out: which label, the confident-extreme cut, the L2 strength."""

    label_col: str = FEATURED_FRACTION_COL
    extreme_low: float = 0.2
    extreme_high: float = 0.8
    c: float = 1.0


class HarnessConfig(RunConfig):
    """A complete, stamped run: data corpora + objective + model scale + probe + run knobs."""

    pretrain_dir: str
    probe_dir: str
    out_dir: str
    device: str | None = None
    seed: int = 0
    q: float = 4.0
    norm_sample: int = 8000
    monitor_frac: float = 0.02
    autocast: str | None = None  # None | "bf16" | "fp16"
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    probe: ProbeConfig = Field(default_factory=ProbeConfig)

    def autocast_dtype(self) -> torch.dtype | None:
        if self.autocast is None:
            return None
        if self.autocast not in _AUTOCAST:
            raise ValueError(
                f"autocast must be one of {sorted(_AUTOCAST)} or null, got {self.autocast!r}"
            )
        return _AUTOCAST[self.autocast]

    def to_jepa_config(self) -> JepaConfig:
        return self.objective.to_jepa_config(seed=self.seed)


# --- report ------------------------------------------------------------------------------


@dataclasses.dataclass
class RunReport:
    """The deliverable: the headline AUC (+ CI) + the two gut-checks + provenance."""

    auc: float | None
    auc_lo: float | None
    auc_hi: float | None
    n_train: int
    n_test: int
    halted: bool
    final_loss: float | None
    checkpoint: str | None
    collapse_png: str | None
    umap_png: str | None
    explorer_dir: str | None = None
    note: str = ""

    def go_no_go(self) -> str:
        """A plain read of the result (positive / null-but-inconclusive / collapse)."""
        if self.halted:
            return "COLLAPSE — representation collapsed; a finding, not a pass (see trace + tries)."
        if self.auc is None:
            return f"INCONCLUSIVE — probe AUC not computed ({self.note})."
        ci = f" (95% CI {self.auc_lo:.3f}–{self.auc_hi:.3f})" if self.auc_lo is not None else ""
        if self.auc >= 0.6:
            return f"GO — probe AUC {self.auc:.3f}{ci} clearly above chance; the premise is alive."
        return (
            f"NULL (inconclusive) — probe AUC {self.auc:.3f}{ci} near chance. Not proof the "
            "premise is dead: at this scale it may be undertraining. Next: scale / warm-start."
        )


# Back-compat alias: the slice's report type is the harness report.
SliceReport = RunReport


# --- objective seam ----------------------------------------------------------------------


def build_objective(config: JepaConfig, encoder: VisionTransformer) -> Jepa:
    """Construct the training objective for ``encoder``.

    The objective switch point. Today there is one path — the latent-MSE JEPA. This is a
    *partial* seam, not a one-line one: a future MAE / contrastive baseline adds its module
    class and a branch here **and** must also widen the config type (``JepaConfig`` →
    Protocol/union), add a second training loop (``train_jepa``'s EMA / latent-MSE loop is
    JEPA-specific), and adjust ``calibrate``'s predictor-reach — so don't under-scope a
    baseline as trivial. No name-keyed registry is built until that second consumer actually
    lands (the repo's "second consumer" rule).
    """
    return Jepa(encoder, config)


# --- shared setup ------------------------------------------------------------------------


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
    data_snapshot: str


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
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> _Prepared:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info("harness run on device=%s, out=%s", device, out)

    pre_src = DirectorySource(pretrain_dir)
    probe_src = DirectorySource(probe_dir)
    pre_ids = [int(r["object_id"]) for r in pre_src.rows]
    probe_ids = [int(r["object_id"]) for r in probe_src.rows]

    # 1. splits — dedup guard exercised, monitor slice carved, probe split assigned
    deduped = resolve_corpora(pre_ids, probe_ids)
    pre_split = split_pretrain(deduped, seed=seed, monitor_frac=monitor_frac)
    probe_split = assign_three_way(probe_ids, seed=seed, ratios=ratios)
    data_snapshot = write_split_plan(
        out / "split_plan.json",
        probe=probe_split,
        pretrain=pre_split,
        seed=seed,
        ratios=ratios,
    )

    # 2. one frozen pipeline, baked into a shared hash-keyed cache (incremental top-up)
    pipeline = _build_pipeline(pre_src, q=q, n_sample=norm_sample, seed=seed)
    bake_cache(pre_src, pipeline, out / "cache")
    cache = bake_cache(probe_src, pipeline, out / "cache")  # same hash dir → appends probe
    rows = rows_by_id([*pre_src.rows, *probe_src.rows])

    # 3. encoder + objective + loaders sized to the baked stamp
    stamp_px = cache.index.height
    encoder = VisionTransformer(img_size=stamp_px, **(model_kwargs or {}))
    jepa = build_objective(config, encoder)
    train_ds = StampDataset(cache, rows, sorted(pre_split.train))
    loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False)
    monitor_ids = sorted(pre_split.monitor) or sorted(pre_split.train)
    monitor_ds = StampDataset(cache, rows, monitor_ids)
    monitor_batch = next(iter(DataLoader(monitor_ds, batch_size=min(64, len(monitor_ds) or 1))))
    return _Prepared(
        out, device, jepa, loader, monitor_batch, cache, rows, probe_split, data_snapshot
    )


# --- the run -----------------------------------------------------------------------------


def run_harness(config: HarnessConfig) -> RunReport:
    """Execute a full stamped run: splits → bake → train → freeze → probe → figures/blobs."""
    device = config.device or pick_device()
    jcfg = config.to_jepa_config()
    prep = _prepare(
        config.pretrain_dir,
        config.probe_dir,
        config.out_dir,
        config=jcfg,
        device=device,
        seed=config.seed,
        q=config.q,
        norm_sample=config.norm_sample,
        monitor_frac=config.monitor_frac,
        model_kwargs=config.model.model_kwargs(),
        ratios=config.ratios,
    )
    out = prep.out

    # pretrain with the collapse monitor live
    result = train_jepa(
        prep.jepa,
        prep.loader,
        device=device,
        monitor_batch=prep.monitor_batch,
        checkpoint_path=out / "encoder.pt",
        autocast_dtype=config.autocast_dtype(),
    )

    stamp = _make_stamp(config, prep.data_snapshot)
    report = RunReport(
        auc=None,
        auc_lo=None,
        auc_hi=None,
        n_train=0,
        n_test=0,
        halted=result.halted,
        final_loss=result.losses[-1] if result.losses else None,
        checkpoint=str(result.checkpoint) if result.checkpoint else None,
        collapse_png=_safe_collapse_plot(result.collapse_trace, out / "collapse_trace.png"),
        umap_png=None,
    )

    # freeze + probe the confident extremes → the headline (+ CI, figures, explorer blobs)
    if not result.halted and result.checkpoint is not None:
        frozen = load_frozen_encoder(result.checkpoint)
        _probe_and_persist(
            frozen, prep.cache, prep.rows, prep.probe_split, config, device, stamp, report
        )

    _write_report(out, report)
    write_stamp(stamp, out, config.model_dump(mode="json"))
    logger.info("harness report: %s", report.go_no_go())
    return report


def evaluate_probe(config: HarnessConfig, *, checkpoint: str | Path | None = None) -> ProbeResult:
    """Probe-only re-evaluation on an existing frozen checkpoint — no retraining.

    Reuses the already-baked cache and the deterministic probe split, reloads the frozen
    encoder, and reports the converged AUC + bootstrap CI. Used to regenerate a run's
    headline figure (and refresh the explorer blobs) without spending a training run.
    """
    out = Path(config.out_dir)
    ckpt = Path(checkpoint) if checkpoint is not None else out / "encoder.pt"
    cache = _open_existing_cache(out)
    probe_src = DirectorySource(config.probe_dir)
    rows = rows_by_id(probe_src.rows)
    probe_ids = [int(r["object_id"]) for r in probe_src.rows]
    # The deterministic probe split — identical to the one the training run used (same seed +
    # ratios), so the headline reproduces exactly. No pretrain split is needed here.
    probe_split = assign_three_way(probe_ids, seed=config.seed, ratios=config.ratios)
    data_snapshot = _probe_data_snapshot(config.probe_dir, probe_ids)

    frozen = load_frozen_encoder(ckpt)
    stamp = _make_stamp(config, data_snapshot)
    report = RunReport(
        auc=None,
        auc_lo=None,
        auc_hi=None,
        n_train=0,
        n_test=0,
        halted=False,
        final_loss=None,
        checkpoint=str(ckpt),
        collapse_png=None,
        umap_png=None,
    )
    device = config.device or pick_device()
    _probe_and_persist(frozen, cache, rows, probe_split, config, device, stamp, report)
    # Probe-only re-eval doesn't see the training trace; carry the training-side fields over
    # from an existing report so regenerating the headline never discards them.
    existing = out / "report.json"
    if existing.exists():
        prior = json.loads(existing.read_text())
        report.final_loss = prior.get("final_loss")
        report.collapse_png = prior.get("collapse_png")
    _write_report(out, report)
    write_stamp(stamp, out, config.model_dump(mode="json"))
    logger.info("probe-only re-eval: %s", report.go_no_go())
    return ProbeResult(
        auc=report.auc if report.auc is not None else float("nan"),
        auc_lo=report.auc_lo if report.auc_lo is not None else float("nan"),
        auc_hi=report.auc_hi if report.auc_hi is not None else float("nan"),
        n_train=report.n_train,
        n_test=report.n_test,
    )


def _probe_and_persist(
    frozen: VisionTransformer,
    cache: Any,
    rows: dict[int, dict[str, Any]],
    probe_split: Any,
    config: HarnessConfig,
    device: str,
    stamp: RunStamp,
    report: RunReport,
) -> None:
    """Fit the frozen probe, attach AUC + CI to ``report``, write UMAP + explorer blobs."""
    pc = config.probe
    train_ds = StampDataset(cache, rows, sorted(probe_split.train), label_fraction_col=pc.label_col)
    test_ds = StampDataset(cache, rows, sorted(probe_split.test), label_fraction_col=pc.label_col)
    train_full = extract_embeddings(frozen, train_ds, device=device)
    test_full = extract_embeddings(frozen, test_ds, device=device)
    train_emb = _extremes(train_full, low=pc.extreme_low, high=pc.extreme_high)
    test_emb = _extremes(test_full, low=pc.extreme_low, high=pc.extreme_high)

    try:
        auc, lo, hi = probe_auc_ci(train_emb, test_emb, c=pc.c)
        report.auc, report.auc_lo, report.auc_hi = auc, lo, hi
        report.n_train, report.n_test = len(train_emb.y), len(test_emb.y)
    except ValueError as exc:  # too few confident extremes / single class in a tiny run
        report.note = f"probe skipped: {exc}"
        logger.warning(report.note)

    out = Path(config.out_dir)
    report.umap_png = _safe_umap_plot(test_emb, out / "umap.png")
    report.explorer_dir = _write_explorer_blobs(
        out, test_ds.object_ids, test_full, train_emb, pc, stamp, report
    )


# --- explorer blobs (numeric only; docs/embedding-explorer.md "web-ready artifacts") -------


def _write_explorer_blobs(
    out: Path,
    object_ids: list[int],
    reference: Embeddings,
    train_extremes: Embeddings,
    pc: ProbeConfig,
    stamp: RunStamp,
    report: RunReport,
) -> str | None:
    """Persist the embedding index, the fitted concept direction, and UMAP coords as static
    JSON/npz — the web-ready artefacts the embedding explorer consumes. No thumbnails here."""
    explorer = out / "explorer"
    explorer.mkdir(parents=True, exist_ok=True)
    encoder_stamp = {"config_hash": stamp.config_hash, "code_sha": stamp.code_sha}

    np.savez(
        explorer / "embeddings.npz",
        object_ids=np.asarray(object_ids, dtype=np.int64),
        x=reference.x.astype(np.float32),
        y=reference.y.astype(np.int64),
        fraction=reference.fraction.astype(np.float32),
    )

    directions: dict[str, Any] = {"encoder_stamp": encoder_stamp}
    try:
        d = probe_direction(train_extremes, name="featured", c=pc.c)
        directions["featured"] = {
            "w_unit": d.w_unit.tolist(),
            "w_raw": d.w_raw.tolist(),
            "bias": d.bias,
            "auc": report.auc,
            "auc_ci": [report.auc_lo, report.auc_hi],
        }
    except ValueError as exc:  # single-class train extremes — no direction to fit
        directions["featured"] = {"error": str(exc)}
    (explorer / "concept_directions.json").write_text(json.dumps(directions, indent=2) + "\n")

    coords = _safe_umap_coords(reference.x)
    if coords is not None:
        (explorer / "umap_coords.json").write_text(
            json.dumps(
                {"object_ids": [int(o) for o in object_ids], "coords": coords.tolist(), "seed": 0},
                indent=2,
            )
            + "\n"
        )

    (explorer / "index.json").write_text(
        json.dumps(
            {
                "n": len(object_ids),
                "embed_dim": int(reference.x.shape[1]) if reference.x.size else 0,
                "encoder_stamp": encoder_stamp,
                "files": {
                    "embeddings": "embeddings.npz",
                    "concept_directions": "concept_directions.json",
                    "umap_coords": "umap_coords.json" if coords is not None else None,
                },
            },
            indent=2,
        )
        + "\n"
    )
    return str(explorer)


# --- figures (lazy on the eval extras) ---------------------------------------------------


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


def _safe_umap_coords(x: np.ndarray) -> np.ndarray | None:
    """2-D UMAP coordinates for the explorer blob — skipped if umap absent or too few points."""
    if x.shape[0] < 3:
        return None
    try:
        from galaxy_jepa.eval.embed import umap_2d

        return umap_2d(x)
    except ImportError:  # pragma: no cover - umap is an eval extra
        return None


# --- helpers -----------------------------------------------------------------------------


def _make_stamp(config: HarnessConfig, data_snapshot: str) -> RunStamp:
    return RunStamp.create(
        config.model_dump(mode="json"), data_snapshot=data_snapshot, seed=config.seed
    )


def _probe_data_snapshot(probe_dir: str | Path, probe_ids: list[int]) -> str:
    """The probe corpus's data-snapshot hash — its pull manifest if present, else recomputed."""
    manifest = Path(probe_dir) / "manifest.json"
    if manifest.exists():
        snapshot = json.loads(manifest.read_text()).get("data_snapshot")
        if snapshot:
            return str(snapshot)
    return manifest_hash(probe_ids, "")


def _open_existing_cache(out_dir: str | Path) -> TensorCache:
    """Open the single hash-keyed cache under ``out_dir/cache`` without re-baking."""
    cache_root = Path(out_dir) / "cache"
    subdirs = [p for p in cache_root.iterdir() if p.is_dir()] if cache_root.exists() else []
    if len(subdirs) != 1:
        raise FileNotFoundError(
            f"expected exactly one baked cache under {cache_root}, found {len(subdirs)}; "
            "run the full harness first to bake the cache."
        )
    return TensorCache(subdirs[0])


def _write_report(out: Path, report: RunReport) -> None:
    (out / "report.json").write_text(
        json.dumps({**dataclasses.asdict(report), "go_no_go": report.go_no_go()}, indent=2) + "\n"
    )


# --- calibration pre-flight (disk- vs compute-bound; the slice plan's decision gate) -------


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
