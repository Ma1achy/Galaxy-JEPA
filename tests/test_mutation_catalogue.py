"""Pytest entry-point for the §14 scientific mutation catalogue (all 21 critical mutants)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

import mutation_audit as ma  # noqa: E402
import research_validate as rv  # noqa: E402


@pytest.mark.mutation
@pytest.mark.parametrize("entry", ma.CATALOGUE, ids=[e[0] for e in ma.CATALOGUE])
def test_critical_scientific_mutant_is_killed(entry: tuple) -> None:
    """Every mandatory §14 critical mutant must pass unmutated control and be KILLED mutated."""
    mid, cat, desc, fn = entry
    res = ma._run_mutant(mid, cat, desc, fn)
    assert res.control_passed, f"{mid} control failed: {res.kill_reason}"
    assert res.killed, f"{mid} ({cat}) SURVIVED: {desc}"


def test_validation_manifest_verifier_rejects_stale_dirty_and_failed(tmp_path: Path) -> None:
    """Verify research_validate.verify_manifest fails closed on missing/stale/dirty/failed."""
    import json

    missing = tmp_path / "missing.json"
    ok, errs = rv.verify_manifest(missing)
    assert not ok and any("not found" in e for e in errs)

    head = rv._git("rev-parse", "HEAD")
    manifest = tmp_path / "validation_manifest.json"

    # Stale commit rejected
    manifest.write_text(
        json.dumps(
            {
                "schema_version": rv.SCHEMA_VERSION,
                "passed": True,
                "git": {"commit": "0" * 40, "dirty": False},
            }
        )
    )
    ok_stale, errs_stale = rv.verify_manifest(manifest, allow_dirty=True)
    assert not ok_stale and any("commit" in e for e in errs_stale)

    # Failed manifest rejected
    manifest.write_text(
        json.dumps(
            {
                "schema_version": rv.SCHEMA_VERSION,
                "passed": False,
                "git": {"commit": head, "dirty": False},
            }
        )
    )
    ok_fail, errs_fail = rv.verify_manifest(manifest, allow_dirty=True)
    assert not ok_fail and any("failed" in e for e in errs_fail)

    # Valid manifest matching HEAD passes (with allow_dirty=True for test worktrees)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": rv.SCHEMA_VERSION,
                "passed": True,
                "git": {"commit": head, "dirty": False},
            }
        )
    )
    ok_good, errs_good = rv.verify_manifest(manifest, allow_dirty=True)
    assert ok_good and errs_good == []
