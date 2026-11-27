"""Pytest entry-point for the §14 scientific mutation catalogue (all 21 critical mutants)."""

from __future__ import annotations

import json
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
    mid, cat, desc, _fn = entry
    res = ma.run_one(entry)
    assert res.control_passed, f"{mid} control failed: {res.kill_reason}"
    assert res.killed, f"{mid} ({cat}) SURVIVED: {desc} ({res.kill_reason})"


def test_validation_manifest_verifier_rejects_stale_dirty_tampered_and_failed(
    tmp_path: Path,
) -> None:
    """Verify research_validate.verify_manifest fails closed on missing/stale/dirty/tampered."""
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

    # Dirty manifest rejected when allow_dirty=False
    manifest.write_text(
        json.dumps(
            {
                "schema_version": rv.SCHEMA_VERSION,
                "passed": True,
                "git": {"commit": head, "dirty": True},
            }
        )
    )
    ok_dirty, errs_dirty = rv.verify_manifest(manifest, allow_dirty=False)
    assert not ok_dirty and any("dirty" in e for e in errs_dirty)

    # Tampered pinned_file_sha256 rejected
    manifest.write_text(
        json.dumps(
            {
                "schema_version": rv.SCHEMA_VERSION,
                "passed": True,
                "git": {"commit": head, "dirty": False},
                "pinned_file_sha256": {"artifacts/dd_part4.py": "0" * 64},
            }
        )
    )
    ok_pin, errs_pin = rv.verify_manifest(manifest, allow_dirty=True)
    assert not ok_pin and any("sha256 mismatch" in e for e in errs_pin)

    # Valid manifest matching HEAD + real pinned sha256 passes
    real_sha = rv._sha256_file(REPO / "artifacts/dd_part4.py")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": rv.SCHEMA_VERSION,
                "passed": True,
                "git": {"commit": head, "dirty": False},
                "pinned_file_sha256": {"artifacts/dd_part4.py": real_sha},
            }
        )
    )
    ok_good, errs_good = rv.verify_manifest(manifest, allow_dirty=True)
    assert ok_good and errs_good == []
