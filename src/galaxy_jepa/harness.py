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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import Field
from torch.utils.data import DataLoader

from galaxy_jepa.callbacks.checkpoint import TrainCheckpointer
from galaxy_jepa.callbacks.collapse import CollapseFloorFreeze
from galaxy_jepa.core.config import RunConfig, RunStamp, write_stamp
from galaxy_jepa.data.cache import TensorCache, bake_cache, write_scalars
from galaxy_jepa.data.dataset import ResumableShuffle, StampDataset, rows_by_id
from galaxy_jepa.data.manifest import manifest_hash
from galaxy_jepa.data.metadata import FEATURED_FRACTION_COL
from galaxy_jepa.data.orchestrate import (
    assign_three_way,
    resolve_corpora,
    split_pretrain,
    write_split_plan,
)
from galaxy_jepa.data.sources import DirectorySource
from galaxy_jepa.data.transforms import AsinhStretch, NormalisationFreeze, Pipeline
from galaxy_jepa.models.vit import VisionTransformer, load_frozen_encoder
from galaxy_jepa.objectives.jepa import Jepa, JepaConfig, _to_device, train_jepa
from galaxy_jepa.probing.config import ProbingConfig
from galaxy_jepa.probing.extract import LabelProvider
from galaxy_jepa.probing.logistic import (
    Embeddings,
    ProbeResult,
    _extremes,
    extract_embeddings,
    probe_auc_ci,
    probe_direction,
)
from galaxy_jepa.probing.run import ProbingReport, run_probing
from galaxy_jepa.probing.schemes import (
    DEFAULT_CONSENSUS_GATE,
    FeatureScheme,
    get_scheme,
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
    checkpoint_every: int = 1500  # steps; ~19 min of work at the measured 1.297 steps/s

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
            checkpoint_every=self.checkpoint_every,
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


class ProbingStageConfig(RunConfig):
    """Whether the full probing battery runs at the end of a training run, and how.

    Off by default, deliberately. The battery is the expensive stage (per-feature nulls, the MLP
    capacity sweep, ≥10k-shuffle permutations) and the design's own sequencing runs it *after* a
    **label-blind** checkpoint choice (spec 1C) — not automatically at the end of every training
    run, where a training loop that ends badly would still burn the probe. Turning it on is a
    deliberate act; :func:`probe_frozen_checkpoint` is the standalone entry point.
    """

    enabled: bool = False
    scheme: str | None = None  # None ⇒ the LabelProvider default; else a probing.schemes name
    max_galaxies: int | None = None  # truncate the corpus (plumbing smokes only)
    # REQUIRED whenever the battery runs (D8). There is no default anywhere for this: v1's
    # mean+2σ *method* carries, its value 21 does not — that was read off a different data
    # release. Turning probing on means stating the floor. See `probing.config.VoteCountFreeze`.
    vote_count_min: float | None = None


class PathsConfig(RunConfig):
    """Where the run reads and writes.

    **Not** part of the experiment's identity: named in ``RunConfig.NON_DETERMINING``, so it is
    dropped from the stamped ``config_hash``. Moving a corpus to another disk is not a different
    experiment — *which* galaxies a run saw is carried by ``RunStamp.data_snapshot``, which hashes
    the object-id set rather than a location.
    """

    pretrain_dir: str
    probe_dir: str
    out_dir: str


class RuntimeConfig(RunConfig):
    """What the run executed on.

    Unlike :class:`PathsConfig` this **is** hashed. A backend is not a location: MPS, CPU and
    CUDA differ numerically, so two runs agreeing on everything but the device are not the same
    run and must not share a hash.

    ``device=None`` means :func:`pick_device`, so the stamp must record the *resolved* value
    (:meth:`HarnessConfig.with_resolved_device`) — hashing the literal null would hand MPS and
    CUDA one identical hash, precisely the collision this field exists to prevent.
    """

    device: str | None = None

    def resolved_device(self) -> str:
        """The concrete backend this run uses — the declared one, else :func:`pick_device`."""
        return self.device or pick_device()


class HarnessConfig(RunConfig):
    """A complete, stamped run: data corpora + objective + model scale + probe + run knobs."""

    paths: PathsConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    seed: int = 0
    q: float = 4.0
    #: The frozen normalisation statistic (E5). REQUIRED before any run that bakes a cache —
    #: `_build_pipeline` loads it and refuses to fit. There is deliberately no `norm_sample`
    #: knob any more: the sample size that produced these numbers is *in* the record, and a
    #: live knob would imply a run could change it, which is exactly the drift being closed.
    normalisation: NormalisationFreeze | None = None
    #: The pre-registered collapse kill criterion (G5). Unset forfeits the tripwire — a run can
    #: then only halt on a non-finite embedding or a std at zero, and would burn days on a dead
    #: representation — so the stamp records ``collapse_floor_open`` rather than implying a choice.
    collapse_floor: CollapseFloorFreeze | None = None
    #: A run that only exercises plumbing or measures throughput — NOT a result. Mirrors
    #: ``ProbingConfig.smoke``: being a determining field it changes ``config_hash``, so a
    #: smoke's artefacts can never collide with a real run's, and it is additionally written
    #: into ``escape_hatches_used`` so the stamp says so in words as well as in the hash.
    smoke: bool = False
    monitor_frac: float = 0.02
    autocast: str | None = None  # None | "bf16" | "fp16"
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    probe: ProbeConfig = Field(default_factory=ProbeConfig)
    probing: ProbingStageConfig = Field(default_factory=ProbingStageConfig)

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

    def with_resolved_device(self) -> HarnessConfig:
        """A copy whose ``runtime.device`` is concrete, so the stamp records what actually ran."""
        if self.runtime.device:
            return self
        return self.model_copy(
            update={"runtime": RuntimeConfig(device=self.runtime.resolved_device())}
        )


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


def seed_init(seed: int, stamp_px: int, model_kwargs: dict[str, Any] | None) -> VisionTransformer:
    """Build the encoder with the run's seed actually applied to the weight init.

    **This closes a provenance hole, not a convenience.** ``RunStamp`` asserts that a run is
    determined by ``(config_hash, code_sha, data_snapshot, seed)``, and ``seed`` reached the masker
    (``loss_step(seed=cfg.seed + step)``) and, since Brief G1, the data order
    (:class:`~galaxy_jepa.data.dataset.ResumableShuffle`) — but **nothing in the package ever
    called** ``torch.manual_seed``. So the ViT's parameters came from whatever ambient RNG state the
    process happened to hold, and two runs with byte-identical stamps produced different encoders.
    Found while comparing two Brief G3 throughput runs whose collapse traces diverged at the same
    seed; the resume-identity proof was unaffected because a restore overwrites the init entirely.

    Seeding here rather than inside ``Jepa.__init__`` keeps it one site and covers the whole
    construction sequence: the encoder, its deepcopy into the EMA target, and the predictor all draw
    from the one seeded stream.
    """
    torch.manual_seed(int(seed))
    return VisionTransformer(img_size=stamp_px, **(model_kwargs or {}))


# --- shared setup ------------------------------------------------------------------------


def _build_pipeline(*, q: float, freeze: NormalisationFreeze | None) -> Pipeline:
    """The parity-locked pipeline, **loaded** from the frozen record — never re-fitted.

    This used to call ``fit_normalise`` on every run, and that was a defect rather than a
    shortcut: the subsample is drawn as ``rng.choice(len(source), ...)``, so it moved when the
    pretraining corpus grew from 10,000 to 826,968 stamps, and nothing on disk recorded that
    the constants had changed. Two runs could differ in their input transform with identical
    configs and no way to tell afterwards.

    So the statistic is now an artefact (``NormalisationFreeze``) that this function reads. A
    missing record is a loud failure; re-fitting is not something a training run is allowed to
    do by accident.
    """
    if freeze is None:
        raise ValueError(
            "no frozen normalisation: `normalisation` is unset in the run config, and a run "
            "may not fit one for itself. The statistic is the parity lock across the "
            "pretraining corpus, the probing corpus and every baseline, and re-fitting it "
            "per run is how it drifted silently before. Fit it once with "
            "`artifacts/e5_fit_normalisation.py` and paste the block into the config."
        )
    if freeze.stretch_q != q:
        raise ValueError(
            f"the frozen normalisation was fitted after AsinhStretch(q={freeze.stretch_q}) but "
            f"this run stretches at q={q}. These are post-stretch statistics, so they do "
            "not transfer — re-fit under the new Q, deliberately, or put Q back."
        )
    return Pipeline((AsinhStretch(q=q), freeze.to_normalise()))


@dataclasses.dataclass
class _Prepared:
    """The shared setup the full run and the calibration pre-flight both need."""

    out: Path
    device: str
    jepa: Jepa
    loader: DataLoader
    monitor_batch: dict[str, Any]
    cache: Any
    #: The probe corpus's *location*, not its metadata. Building `rows_by_id` over both corpora
    #: up front cost 4.07 GB resident for the whole training phase, to serve one float per item
    #: that now comes from the cache's scalar sidecar. Probing rebuilds what it needs when it
    #: starts, which on a multi-hour run is a ten-second read at the far end (Brief G2).
    probe_dir: Path
    probe_split: Any
    data_snapshot: str
    sampler: ResumableShuffle


def _prepare(
    pretrain_dir: str | Path,
    probe_dir: str | Path,
    out_dir: str | Path,
    *,
    config: JepaConfig,
    device: str,
    seed: int,
    q: float,
    normalisation: NormalisationFreeze | None,
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
    pipeline = _build_pipeline(q=q, freeze=normalisation)
    # The cache holds NORMALISED stamps, so it records which freeze made them. `_build_pipeline`
    # has already refused a missing record, hence the assert rather than a branch.
    assert normalisation is not None
    norm_hash = normalisation.content_hash
    bake_cache(pre_src, pipeline, out / "cache", normalisation_hash=norm_hash)
    # same hash dir → appends probe under the one statistic; that sharing IS the parity rule
    cache = bake_cache(probe_src, pipeline, out / "cache", normalisation_hash=norm_hash)
    # 2b. the one per-item scalar the masking path needs, as a 4.2 MB array aligned with the
    # index. Written here because this is the only place that holds both corpora's metadata, and
    # written *from* it so the values are identical to what the row-dict path served (Brief G2).
    write_scalars(
        cache.cache_dir,
        {
            int(r["object_id"]): float(r.get("petroRad_r", float("nan")))
            for r in [*pre_src.rows, *probe_src.rows]
        },
    )
    del pre_src, probe_src  # 4.07 GB of metadata has done its job; the run does not need it
    # re-open: `write_scalars` rewrote the index to vouch for the sidecar, and the reader must
    # hold the index that carries the digest, not the one from before it existed
    cache = TensorCache(cache.cache_dir)

    # 3. encoder + objective + loaders sized to the baked stamp
    stamp_px = cache.index.height
    jepa = build_objective(config, seed_init(config.seed, stamp_px, model_kwargs))
    scalars = cache.scalars
    train_ds = StampDataset(cache, {}, sorted(pre_split.train), scalars=scalars)
    # A deterministic, resumable index stream rather than `shuffle=True`: the trajectory is a
    # function of (seed, step) only if the data order is, which is what makes a mid-run resume
    # provably identical rather than merely functional (Brief G1).
    sampler = ResumableShuffle(len(train_ds), seed=config.seed)
    loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler, drop_last=True)
    monitor_ids = sorted(pre_split.monitor) or sorted(pre_split.train)
    monitor_ds = StampDataset(cache, {}, monitor_ids, scalars=scalars)
    monitor_batch = next(iter(DataLoader(monitor_ds, batch_size=min(64, len(monitor_ds) or 1))))
    return _Prepared(
        out,
        device,
        jepa,
        loader,
        monitor_batch,
        cache,
        Path(probe_dir),
        probe_split,
        data_snapshot,
        sampler,
    )


# --- the run -----------------------------------------------------------------------------


def run_harness(config: HarnessConfig) -> RunReport:
    """Execute a full stamped run: splits → bake → train → freeze → probe → figures/blobs."""
    # Pin the backend before anything is stamped: `device: null` must not reach the hash, or
    # MPS and CUDA would share one config_hash despite differing numerically.
    config = config.with_resolved_device()
    device = config.runtime.resolved_device()
    jcfg = config.to_jepa_config()
    prep = _prepare(
        config.paths.pretrain_dir,
        config.paths.probe_dir,
        config.paths.out_dir,
        config=jcfg,
        device=device,
        seed=config.seed,
        q=config.q,
        normalisation=config.normalisation,
        monitor_frac=config.monitor_frac,
        model_kwargs=config.model.model_kwargs(),
        ratios=config.ratios,
    )
    out = prep.out

    # The stamp is made *before* training, not after: the checkpointer records `config_hash` in
    # every payload so a resume cannot silently continue a different experiment (Brief G1).
    stamp = _make_stamp(config, prep.data_snapshot)
    assert config.normalisation is not None  # `_build_pipeline` has already refused otherwise
    checkpointer = (
        TrainCheckpointer(
            out / "checkpoints",
            every=jcfg.checkpoint_every,
            config_hash=stamp.config_hash,
            normalisation_hash=config.normalisation.content_hash,
            schedule={
                "steps": jcfg.steps,
                "lr": jcfg.lr,
                "warmup_steps": jcfg.warmup_steps,
                "ema_start": jcfg.ema_start,
                "ema_end": jcfg.ema_end,
                "batch_size": jcfg.batch_size,
            },
        )
        if jcfg.checkpoint_every
        else None
    )

    # pretrain with the collapse monitor live, resuming if a committed checkpoint is present
    result = train_jepa(
        prep.jepa,
        prep.loader,
        device=device,
        monitor_batch=prep.monitor_batch,
        checkpoint_path=out / "encoder.pt",
        autocast_dtype=config.autocast_dtype(),
        checkpointer=checkpointer,
        sampler=prep.sampler,
        collapse_floor=config.collapse_floor,
    )
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
            # rebuilt here, not carried through training: probing needs the vote columns, the
            # pretrain loop needs one float, and only the loop runs for hours (Brief G2)
            frozen,
            prep.cache,
            rows_by_id(DirectorySource(prep.probe_dir).rows),
            prep.probe_split,
            config,
            device,
            stamp,
            report,
        )
        # The headline read-out above is one feature; the battery below is the full ladder +
        # controls + uncertainty geometry (design §2–§4). Both are kept: the headline feeds
        # RunReport / the explorer blobs, the battery is the scientific deliverable.
        if config.probing.enabled:
            probe_frozen_checkpoint(
                config,
                checkpoint=result.checkpoint,
                probing=ProbingConfig(
                    seed=config.seed,
                    ratios=config.ratios,
                    device=config.runtime.device,
                    vote_count_min=_required_vote_floor(config),
                ),
                scheme=get_scheme(config.probing.scheme) if config.probing.scheme else None,
                max_galaxies=config.probing.max_galaxies,
            )

    _write_report(out, report)
    write_stamp(stamp, out, config.model_dump(mode="json"))
    logger.info("harness report: %s", report.go_no_go())
    return report


def _required_vote_floor(config: HarnessConfig) -> float:
    """The reliable-label floor, or a loud refusal — never a quiet 21."""
    floor = config.probing.vote_count_min
    if floor is None:
        raise ValueError(
            "probing needs `probing.vote_count_min` and this config leaves it unset. The floor "
            "decides which galaxies count as measured at all, so it is stated, never inferred. "
            "D8 is SUPERSEDED, not open: the mean+2σ filter existed because v1 trained on the "
            "labels, and v2's encoder never sees one, so `configs/probe.yaml` freezes the floor "
            "at 1 — the minimum where a vote fraction is defined, since a question nobody "
            "answered is a 0/0 that GZ2 stores as a literal 0.0. Copy that block, including its "
            "`vote_count_freeze`; `derive_vote_count_min` remains only to re-derive the "
            "withdrawn heuristic for the registered sweep."
        )
    return float(floor)


def evaluate_probe(config: HarnessConfig, *, checkpoint: str | Path | None = None) -> ProbeResult:
    """Probe-only re-evaluation on an existing frozen checkpoint — no retraining.

    Reuses the already-baked cache and the deterministic probe split, reloads the frozen
    encoder, and reports the converged AUC + bootstrap CI. Used to regenerate a run's
    headline figure (and refresh the explorer blobs) without spending a training run.
    """
    config = config.with_resolved_device()  # as in `run_harness`: pin before stamping
    out = Path(config.paths.out_dir)
    ckpt = Path(checkpoint) if checkpoint is not None else out / "encoder.pt"
    cache = _open_existing_cache(out)
    probe_src = DirectorySource(config.paths.probe_dir)
    rows = rows_by_id(probe_src.rows)
    probe_ids = [int(r["object_id"]) for r in probe_src.rows]
    # The deterministic probe split — identical to the one the training run used (same seed +
    # ratios), so the headline reproduces exactly. No pretrain split is needed here.
    probe_split = assign_three_way(probe_ids, seed=config.seed, ratios=config.ratios)
    data_snapshot = _probe_data_snapshot(config.paths.probe_dir, probe_ids)

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
    device = config.runtime.resolved_device()
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


# --- the probing-battery load path (design §2-§4 via probing.run_probing) ------------------


def build_label_provider(
    rows: dict[int, dict[str, Any]],
    *,
    feature_cols: Mapping[str, str] | None = None,
    nuisance_cols: Mapping[str, str] | None = None,
    scheme: FeatureScheme | None = None,
    vote_count_min: float,
    consensus_gate: float = DEFAULT_CONSENSUS_GATE,
) -> LabelProvider:
    """Build the probing ``LabelProvider`` over a corpus's metadata rows.

    The one place the harness translates *its* data-layer view (``rows_by_id`` over a
    ``DirectorySource``) into the probing layer's label view. Kept a named function rather than
    inlined so the feature-scheme selection (D14) has exactly one seam to enter through.
    """
    return LabelProvider(
        rows,
        feature_cols=feature_cols,
        nuisance_cols=nuisance_cols,
        scheme=scheme,
        vote_count_min=vote_count_min,
        consensus_gate=consensus_gate,
    )


def probe_frozen_checkpoint(
    config: HarnessConfig,
    *,
    checkpoint: str | Path | None = None,
    probing: ProbingConfig | None = None,
    out_dir: str | Path | None = None,
    scheme: FeatureScheme | None = None,
    feature_cols: Mapping[str, str] | None = None,
    nuisance_cols: Mapping[str, str] | None = None,
    max_galaxies: int | None = None,
    emit_figures: bool = True,
) -> ProbingReport:
    """Run the **full probing battery** on a frozen checkpoint — the load-bearing joint.

    ``probing.run_probing`` was built and tested against a synthetic encoder; this is the wrapper
    that hands it a real one. It owns exactly the translation the probing layer refuses to do for
    itself (the freeze boundary): the baked cache + the corpus metadata → a ``StampDataset`` and a
    ``LabelProvider``. ``run_probing`` stays objective-free; this function lives in the harness
    because the harness is the only layer allowed to see both sides.

    Reuses the already-baked fp16 cache under ``config.paths.out_dir`` — the parity lock is the
    ``pipeline_hash`` the cache directory is keyed on, so probing cannot silently re-preprocess.
    A ``ProbingConfig`` with ``smoke=True`` marks the artefacts (see that field); ``max_galaxies``
    truncates the corpus for a plumbing smoke and is itself recorded, since a truncated corpus is
    a different ``data_snapshot``.
    """
    cfg = (
        probing
        if probing is not None
        else ProbingConfig(
            seed=config.seed,
            ratios=config.ratios,
            vote_count_min=_required_vote_floor(config),
        )
    )
    out = (
        Path(out_dir)
        if out_dir is not None
        else Path(config.paths.out_dir) / ("probing_smoke" if cfg.smoke else "probing")
    )
    ckpt = Path(checkpoint) if checkpoint is not None else Path(config.paths.out_dir) / "encoder.pt"

    cache = _open_existing_cache(config.paths.out_dir)
    probe_src = DirectorySource(config.paths.probe_dir)
    rows = rows_by_id(probe_src.rows)

    # Only galaxies the cache actually holds can be probed; StampDataset intersects, but doing it
    # here too keeps the truncation deterministic (sorted) rather than dependent on corpus order.
    probe_ids = sorted(int(r["object_id"]) for r in probe_src.rows)
    probe_ids = cache.present(probe_ids)
    if max_galaxies is not None:
        probe_ids = probe_ids[:max_galaxies]
    if not probe_ids:
        raise ValueError(
            f"no galaxy from {config.paths.probe_dir} is present in the cache under "
            f"{config.paths.out_dir}: "
            "the probe corpus was never baked into this run's cache (bake it first)"
        )

    dataset = StampDataset(cache, rows, probe_ids)
    labels = build_label_provider(
        rows,
        feature_cols=feature_cols,
        nuisance_cols=nuisance_cols,
        scheme=scheme,
        vote_count_min=cfg.vote_count_min,
        consensus_gate=cfg.consensus_gate,
    )
    frozen = load_frozen_encoder(ckpt)

    logger.info(
        "probing battery: %d galaxies, %d feature(s), %d nuisance(s), checkpoint=%s%s",
        len(probe_ids),
        len(labels.features),
        len(labels.nuisances),
        ckpt,
        " [SMOKE — not a result]" if cfg.smoke else "",
    )
    return run_probing(
        frozen,
        dataset,
        labels,
        frozen.config,
        config=cfg,
        out_dir=out,
        emit_figures=emit_figures,
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

    out = Path(config.paths.out_dir)
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
    """Stamp `config`. Hashes `determining_dump()` — the experiment minus where it ran — and
    records the resolved backend as metadata alongside the hash it entered."""
    resolved = config.with_resolved_device()
    forfeits = (
        (["smoke"] if resolved.smoke else [])
        # An unset floor is a forfeited guarantee, not a neutral default: the run can then only
        # halt on a non-finite embedding, so a dead representation costs days before anyone looks.
        + ([] if resolved.collapse_floor is not None else ["collapse_floor_open"])
        # No mid-run checkpoint on a multi-hour job is a bet, and the artefact should say so.
        + ([] if resolved.objective.checkpoint_every else ["no_mid_run_checkpoint"])
    )
    return RunStamp.create(
        resolved.determining_dump(),
        data_snapshot=data_snapshot,
        seed=resolved.seed,
        device=resolved.runtime.device,
        escape_hatches_used=forfeits or None,
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
