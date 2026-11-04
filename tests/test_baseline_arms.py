"""The third seed pair: MoCo s2 and MAE s2 train on A3's seed (baselines_comparison.md).

`baseline_run.py --plan` runs F0 and the config-equality check against A1 and launches nothing.
Each arm is its own process, because F0 reads its environment at import. Needs the v2 cache on the
external disk; skips when it is absent.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[1]
CACHE = Path("/Volumes/X10 Pro/galaxy-jepa/runs/m_v2/cache")


@pytest.mark.parametrize(
    ("arm", "seed", "paired", "out"),
    [
        ("moco0", 0, "A1 (runs/m_v2)", "runs/moco_s0"),
        ("mae1", 1, "A2 (runs/m_v2_s1)", "runs/mae_s1"),
        ("moco2", 2, "A3 (runs/m_v2_s2)", "runs/moco_s2"),
        ("mae2", 2, "A3 (runs/m_v2_s2)", "runs/mae_s2"),
    ],
)
def test_plan_names_the_paired_run_and_seed(arm, seed, paired, out):
    if not CACHE.exists():
        pytest.skip("the v2 cache is not mounted")
    r = subprocess.run(
        [sys.executable, "artifacts/baseline_run.py", arm, "--plan"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"pairs with  : {paired}" in r.stdout
    assert f"train={seed} " in r.stdout
    assert f"out_dir     : {out}" in r.stdout
    assert "nothing launched" in r.stdout
