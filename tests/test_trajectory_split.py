"""The baselines' trajectory probe scores on ladder v2's split, byte for byte.

Ladder v2's E2 scored A1–A3 on the identical set (`j4_spread_controls._identical_split`,
decision 1), under the manifest's `common_env`. `baseline_trajectory_probe.py` must land on exactly
those train and test galaxies, or a baseline's trajectory is read off different galaxies from the
encoder it is paired with. Each split is built in its own process, because F0 reads its environment
at import: one under the manifest's recorded environment, one under nothing but what the probe
module itself imposes.

At E2's uncapped size the config's own split happens to coincide with the identical set (probe_v2
is v1 minus nine galaxies and `assign_three_way` places each id by its own hash); the two part only
under a stride cap. The probe holds `LADDER_SPLIT=identical` so that stays true if either path
changes, and the second test shows the comparison can fail: at H5's 40,000 cap the config's own
stride differs.

Needs the v1 and v2 caches on the external disk; skips when they are absent.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "docs/ladder_v2/manifest.yaml"
CACHES = (
    Path("/Volumes/X10 Pro/galaxy-jepa/runs/m_v2/cache"),
    Path("/Volumes/X10 Pro/galaxy-jepa/runs/full/cache"),
)

SPLIT = """
import hashlib, sys
import numpy as np
sys.path.insert(0, "artifacts")
{prelude}
import j4_spread_controls as J
s = J.prepare(None, {max_train}, label="split-test", sources=1)
for name, ids in (("train", s.train_ids), ("test", s.test_ids)):
    b = np.asarray(ids, dtype=np.int64).tobytes()
    print(name, len(ids), hashlib.sha1(b).hexdigest())
"""


def _split(env: dict[str, str], prelude: str, max_train: int = 0) -> str:
    """max_train 0 is E2's uncapped split, the one the trajectory probe uses."""
    code = SPLIT.format(prelude=prelude, max_train=max_train)
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def _envs() -> tuple[dict[str, str], dict[str, str]]:
    """(the environment nothing ladder-shaped leaks into, the manifest's common_env)."""
    keep = {
        k: v for k, v in os.environ.items() if not k.startswith(("LADDER_", "GJ_", "F0_", "DD_"))
    }
    common = {k: str(v) for k, v in yaml.safe_load(MANIFEST.read_text())["common_env"].items()}
    assert common["LADDER_SPLIT"] == "identical"
    return keep, common


needs_disk = pytest.mark.skipif(
    not all(c.is_dir() for c in CACHES), reason="the v1/v2 caches are not mounted"
)


@needs_disk
def test_trajectory_probe_split_is_the_ladders_byte_for_byte():
    keep, common = _envs()
    ladder = _split({**keep, **common}, prelude="")
    probe = _split(dict(keep), prelude="import baseline_trajectory_probe")
    assert ladder.count("\n") == 2 and all(int(ln.split()[1]) > 0 for ln in ladder.splitlines())
    assert probe == ladder


@needs_disk
def test_the_comparison_sees_a_different_split():
    """Planted divergence: the config's own stride at a 40,000 cap is not the identical set's."""
    keep, common = _envs()
    ladder = _split({**keep, **common}, prelude="", max_train=40_000)
    own = _split(
        {**keep, **{k: v for k, v in common.items() if k != "LADDER_SPLIT"}},
        prelude="",
        max_train=40_000,
    )
    assert own != ladder
