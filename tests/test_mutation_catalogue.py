"""Pytest entry-point for the §14 scientific mutation catalogue (22 independent mutants)."""

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


def _valid_manifest(head: str) -> dict:
    return {
        "schema_version": rv.SCHEMA_VERSION,
        "passed": True,
        "git": {"commit": head, "dirty": False},
        "checks": [{"name": name, "passed": True, "returncode": 0} for name in rv.REQUIRED_CHECKS],
        "pinned_file_sha256": {rel: rv._sha256_file(REPO / rel) for rel in rv.REQUIRED_PINS},
    }


def test_validation_manifest_verifier_rejects_stale_dirty_tampered_and_failed(
    tmp_path: Path,
) -> None:
    """Verify research_validate.verify_manifest fails closed on malformed/contradictory records."""
    missing = tmp_path / "missing.json"
    ok, errs = rv.verify_manifest(missing)
    assert not ok and any("not found" in e for e in errs)

    head = rv._git("rev-parse", "HEAD")
    manifest = tmp_path / "validation_manifest.json"

    # Malformed JSON rejected
    manifest.write_text("{not-json")
    ok_bad, errs_bad = rv.verify_manifest(manifest, allow_dirty=True)
    assert not ok_bad and any("malformed JSON" in e for e in errs_bad)

    # Stale commit rejected
    m_stale = _valid_manifest("0" * 40)
    manifest.write_text(json.dumps(m_stale))
    ok_stale, errs_stale = rv.verify_manifest(manifest, allow_dirty=True)
    assert not ok_stale and any("commit" in e for e in errs_stale)

    # Contradictory manifest (passed=True yet pytest_validation failed!) MUST be rejected
    m_contra = _valid_manifest(head)
    m_contra["checks"][3] = {"name": "pytest_validation", "passed": False, "returncode": 1}
    manifest.write_text(json.dumps(m_contra))
    ok_contra, errs_contra = rv.verify_manifest(manifest, allow_dirty=True)
    assert not ok_contra and any("pytest_validation" in e for e in errs_contra)

    # Missing required checks rejected
    m_nochecks = _valid_manifest(head)
    m_nochecks["checks"] = []
    manifest.write_text(json.dumps(m_nochecks))
    ok_nc, errs_nc = rv.verify_manifest(manifest, allow_dirty=True)
    assert not ok_nc and any("checks" in e for e in errs_nc)

    # Missing/null required pin rejected
    m_nullpin = _valid_manifest(head)
    m_nullpin["pinned_file_sha256"]["artifacts/dd_part4.py"] = None
    manifest.write_text(json.dumps(m_nullpin))
    ok_np, errs_np = rv.verify_manifest(manifest, allow_dirty=True)
    assert not ok_np and any("missing or null required pin" in e for e in errs_np)

    # Dirty manifest rejected when allow_dirty=False
    m_dirty = _valid_manifest(head)
    m_dirty["git"]["dirty"] = True
    manifest.write_text(json.dumps(m_dirty))
    ok_dirty, errs_dirty = rv.verify_manifest(manifest, allow_dirty=False)
    assert not ok_dirty and any("dirty" in e for e in errs_dirty)

    # Tampered pinned_file_sha256 rejected
    m_pin = _valid_manifest(head)
    m_pin["pinned_file_sha256"]["artifacts/dd_part4.py"] = "0" * 64
    manifest.write_text(json.dumps(m_pin))
    ok_pin, errs_pin = rv.verify_manifest(manifest, allow_dirty=True)
    assert not ok_pin and any("sha256 mismatch" in e for e in errs_pin)

    # Complete valid manifest matching HEAD + all required checks & pins passes
    manifest.write_text(json.dumps(_valid_manifest(head)))
    ok_good, errs_good = rv.verify_manifest(manifest, allow_dirty=True)
    assert ok_good and errs_good == []


def test_research_validate_records_gpu_blocked_even_when_cuda_available() -> None:
    """Even on a CUDA host, CPU-only validation (`not gpu`) MUST record cuda_gpu_parity blocked."""
    from unittest import mock

    fake_pass = {
        "name": "stub",
        "command": "true",
        "returncode": 0,
        "passed": True,
        "output_tail": [],
    }

    def _fake_step(name: str, _cmd: list[str], **_kw: object) -> dict:
        return {**fake_pass, "name": name}

    with (
        mock.patch.object(
            rv,
            "_hardware_info",
            return_value={
                "platform": "Linux",
                "machine": "x86_64",
                "cpu_count": 8,
                "cuda_available": True,
                "mps_available": False,
            },
        ),
        mock.patch.object(rv, "_run_step", side_effect=_fake_step),
    ):
        manifest = rv.build_manifest(skip_slow=True)
    blocked_names = {b["check"] for b in manifest["blocked_checks"]}
    assert "cuda_gpu_parity" in blocked_names
    assert any("excluded" in b["reason"] for b in manifest["blocked_checks"])
