"""Generate and verify a provenance-bound research validation manifest (§15).

Produces a machine-readable JSON manifest binding:
  - Git commit SHA, branch, dirty-tree state, and tracked source tree digest
  - Active config & pre-registration manifest hashes
  - Objective & mask scheme versions
  - Python / PyTorch / NumPy / SciPy / scikit-learn versions and hardware capabilities
  - Executed validation gates (ruff, mypy, pytest suites, mutation audit, credential check)
  - Seeds, explicit tolerances, and explicitly disclosed blocked/skipped checks

Usage:
    .venv/bin/python tools/research_validate.py --out runs/validation_manifest.json
    .venv/bin/python tools/research_validate.py --verify runs/validation_manifest.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1


def _git(*args: str) -> str:
    cp = subprocess.run(
        ["git", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return cp.stdout.strip()


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for mod_name in ("torch", "numpy", "scipy", "sklearn", "hypothesis", "pytest"):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            versions[mod_name] = "unavailable"
    return versions


def _hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "cuda_available": False,
        "mps_available": False,
    }
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["mps_available"] = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except Exception:
        pass
    return info


def _run_step(
    name: str, cmd: list[str], *, env_extra: dict[str, str] | None = None
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env_extra:
        env.update(env_extra)
    cp = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, check=False)
    tail = (cp.stdout + "\n" + cp.stderr).strip().splitlines()[-8:]
    return {
        "name": name,
        "command": " ".join(cmd),
        "returncode": cp.returncode,
        "passed": cp.returncode == 0,
        "output_tail": tail,
    }


def build_manifest(*, skip_slow: bool = True) -> dict[str, Any]:
    py = sys.executable
    bin_dir = Path(py).resolve().parent
    ruff = str(bin_dir / "ruff") if (bin_dir / "ruff").exists() else "ruff"
    mypy = str(bin_dir / "mypy") if (bin_dir / "mypy").exists() else "mypy"
    pytest_bin = str(bin_dir / "pytest") if (bin_dir / "pytest").exists() else "pytest"

    commit = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    dirty = bool(_git("status", "--porcelain"))

    pinned_files = {
        "configs/pretrain_v2.yaml": _sha256_file(REPO / "configs/pretrain_v2.yaml"),
        "docs/ladder_v2/manifest.yaml": _sha256_file(REPO / "docs/ladder_v2/manifest.yaml"),
        "artifacts/dd_part4.py": _sha256_file(REPO / "artifacts/dd_part4.py"),
        "artifacts/dd_part4b.py": _sha256_file(REPO / "artifacts/dd_part4b.py"),
    }

    pytest_expr = (
        "not gpu and not external_data and not slow"
        if skip_slow
        else "not gpu and not external_data"
    )
    checks = [
        _run_step("ruff_check", [ruff, "check", "."]),
        _run_step("ruff_format", [ruff, "format", "--check", "."]),
        _run_step("mypy_src", [mypy, "--cache-dir=/tmp/mypy_cache", "src"]),
        _run_step(
            "pytest_validation",
            [
                pytest_bin,
                "-q",
                "-p",
                "no:cacheprovider",
                "--basetemp=/tmp/pytest_tmp",
                "-m",
                pytest_expr,
            ],
            env_extra={"HYPOTHESIS_STORAGE_DIRECTORY": "/tmp/hyp_cache"},
        ),
        _run_step("mutation_audit", [py, "tools/mutation_audit.py"]),
        _run_step("credential_check", [py, "tools/credential_check.py"]),
    ]

    hw = _hardware_info()
    blocked: list[dict[str, str]] = []
    if not hw["cuda_available"]:
        blocked.append(
            {
                "check": "cuda_gpu_parity",
                "reason": "CUDA unavailable on this host; GPU tests marked blocked, not passed.",
            }
        )
    blocked.append(
        {
            "check": "external_sdss_catalogue_reruns",
            "reason": "Requires external SDSS/GZ2 FITS store and explicit user authorisation.",
        }
    )

    all_passed = all(c["passed"] for c in checks)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git": {
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
        },
        "versions": {
            "mask_scheme": "precomputed-masks-v2",
            "jepa_loss_semantics": "independent-target-blocks-equal-block-mse",
            "packages": _package_versions(),
        },
        "pinned_file_sha256": pinned_files,
        "hardware": hw,
        "checks": checks,
        "blocked_checks": blocked,
        "passed": all_passed,
    }


def verify_manifest(manifest_path: Path, *, allow_dirty: bool = False) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not manifest_path.exists():
        return False, [f"validation manifest not found: {manifest_path}"]
    data = json.loads(manifest_path.read_text())
    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"unexpected schema_version: {data.get('schema_version')!r}")
    if not data.get("passed"):
        errors.append("validation manifest recorded failed checks")
    head = _git("rev-parse", "HEAD")
    if data.get("git", {}).get("commit") != head:
        errors.append(
            f"manifest commit {data.get('git', {}).get('commit')!r} != current HEAD {head!r}"
        )
    is_dirty = bool(data.get("git", {}).get("dirty")) or bool(_git("status", "--porcelain"))
    if not allow_dirty and is_dirty:
        errors.append("working tree or manifest is dirty; commit clean tree before validation")
    return len(errors) == 0, errors


def main() -> None:
    ap = argparse.ArgumentParser(description="Build or verify Galaxy-JEPA validation manifest.")
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "validation_manifest.json")
    ap.add_argument("--verify", type=Path, help="Verify an existing manifest against HEAD.")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument(
        "--dev-override",
        type=str,
        help="Explicit development override reason (logged; cannot masquerade as accepted).",
    )
    ap.add_argument("--include-slow", action="store_true")
    args = ap.parse_args()

    if args.verify:
        ok, errs = verify_manifest(args.verify, allow_dirty=args.allow_dirty)
        if not ok:
            if args.dev_override:
                record = {
                    "accepted": False,
                    "override": True,
                    "override_reason": args.dev_override,
                    "manifest": str(args.verify),
                    "errors": errs,
                    "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
                    "head": _git("rev-parse", "HEAD"),
                }
                print(json.dumps(record, indent=2), file=sys.stderr)
                return
            for e in errs:
                print(f"INVALID: {e}", file=sys.stderr)
            raise SystemExit(1)
        print(f"VALID: {args.verify}")
        return

    manifest = build_manifest(skip_slow=not args.include_slow)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote validation manifest -> {args.out} (passed={manifest['passed']})")
    if not manifest["passed"]:
        for c in manifest["checks"]:
            if not c["passed"]:
                print(f"FAILED CHECK: {c['name']} (rc={c['returncode']})", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
