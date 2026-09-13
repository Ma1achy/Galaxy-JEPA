"""Brief J3 — the medium pretraining run: 50,000 steps through the production harness.

Not a results run. `configs/pretrain.yaml` carries `smoke: true`, which moves `config_hash` and
stamps the forfeit, so nothing this produces can be read back as a headline — and, because a
resume refuses across a hash change, these weights cannot be promoted to one either. What it is
for: end-to-end correctness at scale, signs of life, and the evidence the effect floor is chosen
against.

Nothing is reimplemented. This is `run_harness(HarnessConfig)` with two things arranged around it:

  * **logging with timestamps**, at INFO. `train_jepa` already logs every monitor reading
    (`step N: loss=… std=… erank=… cos=…`) and the soft-floor derivation, so a timestamped log IS
    the throughput-drift record — thermal behaviour over hours reads off the wall-clock between
    readings rather than off a separate instrument. Absolute die temperature needs `sudo
    powermetrics` and is deliberately not measured.
  * **the pre-flight as a gate**: `j1_preflight.main()` runs first and raises on any unchecked
    fact, so the run cannot start against a half-closed guardrail.

Halt conditions come from the config, not from here: the hard rank floor, the std floor and
non-finite embeddings, via `CollapseMonitor`. The SOFT rank floor is inert by design under
`sigreg_lambda > 0` (D18) — `train_jepa` derives that and logs it, so its silence is confirmed
rather than assumed.

Investigation code: terse, excluded from lint/CI.

    uv run python artifacts/j3_medium_run.py 2>&1 | tee runs/j3.log
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from j1_preflight import main as preflight  # noqa: E402

from galaxy_jepa.harness import HarnessConfig, run_harness  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    log = logging.getLogger("j3")

    preflight()  # raises on the first unchecked fact
    cfg = HarnessConfig(**yaml.safe_load((REPO / "configs/pretrain.yaml").read_text()))
    if not cfg.smoke:
        raise SystemExit("J3: smoke is False — this run is not a result and must say so")

    obj = cfg.objective
    log.info(
        "J3 launching: %d steps, batch %d, sigreg_lambda=%s, checkpoint every %d, monitor every %d",
        obj.steps, obj.batch_size, obj.sigreg_lambda, obj.checkpoint_every, obj.monitor_every,
    )
    t0 = time.perf_counter()
    report = run_harness(cfg)
    wall = time.perf_counter() - t0

    out = Path(cfg.paths.out_dir)
    log.info(
        "J3 done in %.2f h (%.4f steps/s): halted=%s auc=%s [%s, %s] n_test=%s",
        wall / 3600, obj.steps / wall, report.halted,
        report.auc, report.auc_lo, report.auc_hi, report.n_test,
    )
    log.info("J3 verdict: %s", report.go_no_go())
    (out / "j3_wall.json").write_text(json.dumps({
        "wall_seconds": wall, "steps": obj.steps, "steps_per_s": obj.steps / wall,
        "halted": report.halted, "auc": report.auc, "smoke": True,
    }, indent=2))


if __name__ == "__main__":
    main()
