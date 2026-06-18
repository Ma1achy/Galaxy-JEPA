"""The vertical-slice preset — a thin wrapper over :mod:`galaxy_jepa.harness`.

The slice was the first consumer of the train→freeze→probe path; that path now lives in the
reusable, config-driven :mod:`galaxy_jepa.harness`. This module keeps the slice's familiar
CLI (``python -m galaxy_jepa.vertical_slice``) and the kwargs-style :func:`run_slice` /
:func:`calibrate` entrypoints, mapping them onto a :class:`~galaxy_jepa.harness.HarnessConfig`.

Run the slice (or just the calibration pre-flight) on two pulled corpora::

    python -m galaxy_jepa.vertical_slice --pretrain data/pretrain --probe data/probe \\
        --out runs/slice --calibrate
    python -m galaxy_jepa.vertical_slice --pretrain data/pretrain --probe data/probe \\
        --out runs/slice --steps 50000 --batch-size 256 --beta 0.5 --bf16
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from galaxy_jepa.harness import (
    CalibrationResult,
    HarnessConfig,
    ModelConfig,
    ObjectiveConfig,
    ProbeConfig,
    RunReport,
    SliceReport,
    _prepare,
    build_objective,
    calibrate,
    evaluate_probe,
    pick_device,
    run_harness,
)
from galaxy_jepa.objectives.jepa import JepaConfig

__all__ = [
    "SliceReport",
    "RunReport",
    "run_slice",
    "calibrate",
    "evaluate_probe",
    "pick_device",
    "build_objective",
    "_prepare",
    "main",
]

_DTYPE_NAME = {torch.bfloat16: "bf16", torch.float16: "fp16"}


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
) -> RunReport:
    """Run the whole slice on two pulled corpora (kwargs adapter over :func:`run_harness`)."""
    cfg = HarnessConfig(
        pretrain_dir=str(pretrain_dir),
        probe_dir=str(probe_dir),
        out_dir=str(out_dir),
        device=device,
        seed=seed,
        q=q,
        norm_sample=norm_sample,
        monitor_frac=monitor_frac,
        autocast=_DTYPE_NAME.get(autocast_dtype) if autocast_dtype is not None else None,
        objective=ObjectiveConfig.from_jepa_config(config or JepaConfig()),
        model=ModelConfig(**(model_kwargs or {})),
        probe=ProbeConfig(),
    )
    return run_harness(cfg)


def main(argv: list[str] | None = None) -> None:
    """CLI: run the slice (or just the calibration pre-flight) on two pulled corpora."""
    import argparse
    import logging

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

    cfg = HarnessConfig(
        pretrain_dir=str(args.pretrain),
        probe_dir=str(args.probe),
        out_dir=str(args.out),
        device=args.device,
        seed=args.seed,
        autocast="bf16" if args.bf16 else None,
        objective=ObjectiveConfig(steps=args.steps, batch_size=args.batch_size, beta=args.beta),
    )

    if args.calibrate:
        device = cfg.device or pick_device()
        prep = _prepare(
            cfg.pretrain_dir,
            cfg.probe_dir,
            cfg.out_dir,
            config=cfg.to_jepa_config(),
            device=device,
            seed=cfg.seed,
            q=cfg.q,
            norm_sample=cfg.norm_sample,
            monitor_frac=cfg.monitor_frac,
            model_kwargs=cfg.model.model_kwargs(),
            ratios=cfg.ratios,
        )
        result: CalibrationResult = calibrate(
            prep.jepa,
            prep.loader,
            device=device,
            steps=args.calibrate_steps,
            autocast_dtype=cfg.autocast_dtype(),
        )
        print(result.verdict())
        return

    report = run_harness(cfg)
    print(report.go_no_go())


if __name__ == "__main__":
    main()
