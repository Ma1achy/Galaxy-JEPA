"""Kickoff E — the aligned encoders A1 and A2 (aligned_comparison.md, Common ground).

M's recipe on pretrain_v2: M's driver (`m2_long_run.main`, as O2 ran it), training seed 0 and 1,
config and split seed 0, `steps` left at 253,270 so the LR and EMA schedules are M's and O2's
exactly, and a **fixed stop at 101,308** (4.0 epochs × 25,327). One segment, no probe: nothing may
embed, probe or inspect A1 or A2 before aligned_comparison.md is hashed and the run is released.

Launched through artifacts/aligned_launch.sh, which puts TMPDIR, the log and the checkpoints on
the SSD and points GJ_CONFIG at configs/pretrain_v2.yaml (written at the v2 freeze).

    artifacts/aligned_launch.sh --resolve        # resolved paths, launches nothing
    artifacts/aligned_launch.sh A1 --plan
    artifacts/aligned_launch.sh A1               # then A2
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m1_preflight import M as M_EXPECT  # noqa: E402
from m2_long_run import Run, main as long_run  # noqa: E402

STOP_EPOCHS = (4.0,)  # round(4.0 × 25,327) = 101,308: the one segment end, where M and O2 stopped

# The pre-flight's answers (user, 2026-10-06). M's recipe, so M's chain, reached through a link that
# swaps in M's normalisation and must land on M's shipped hash: the v2 freeze is the only difference
# (aligned_comparison.md, Common ground). The counts come from the v2 manifests, and the overfit gate
# has its own record for this recipe so it never overwrites M's.
EXPECT = dataclasses.replace(
    M_EXPECT,
    shipped_hash="2e2f2877221e1356",  # configs/pretrain_v2.yaml as written
    config_path="configs/pretrain_v2.yaml",
    via=("configs/pretrain.yaml", M_EXPECT.shipped_hash, ("normalisation",)),
    counts_from_manifests=True,
    gate_record="artifacts/out/o3_overfit_gate_v2.json",
)

RUNS = {
    "A1": Run(label="A1", tag="m_v2", probe_epochs=STOP_EPOCHS, train_seed=0, out_dir="runs/m_v2",
              expect=dataclasses.replace(EXPECT, label="A1")),
    "A2": Run(label="A2", tag="m_v2_s1", probe_epochs=STOP_EPOCHS, train_seed=1,
              out_dir="runs/m_v2_s1", expect=dataclasses.replace(EXPECT, label="A2")),
}


def main() -> None:
    name = sys.argv.pop(1)
    if name not in RUNS:
        raise SystemExit(f"aligned_train: run must be one of {sorted(RUNS)}, not {name!r}")
    if "--plan" not in sys.argv and "--no-probe" not in sys.argv:
        sys.argv.append("--no-probe")  # the probe between segments would read the encoder
    long_run(RUNS[name])


if __name__ == "__main__":
    main()
