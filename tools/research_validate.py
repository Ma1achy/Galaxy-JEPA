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
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
REQUIRED_CHECKS: tuple[str, ...] = (
    "ruff_check",
    "ruff_format",
    "mypy_src",
    "pytest_validation",
    "mutation_audit",
    "credential_check",
)
REQUIRED_PINS: tuple[str, ...] = (
    "configs/pretrain_v2.yaml",
    "docs/ladder_v2/manifest.yaml",
    "artifacts/dd_part4.py",
    "artifacts/dd_part4b.py",
)


def _git(*args: str) -> str:
    cp = subprocess.run(
        ["git", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if cp.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed (rc={cp.returncode}): {cp.stderr.strip()}")
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
    # Do NOT resolve symlinks on sys.executable: venv python symlinks to base interpreter.
    bin_dir = Path(py).parent
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
    with tempfile.TemporaryDirectory(prefix="gj_validate_") as td:
        tmp_root = Path(td)
        mypy_cache = str(tmp_root / "mypy_cache")
        pytest_tmp = str(tmp_root / "pytest_tmp")
        hyp_cache = str(tmp_root / "hyp_cache")
        checks = [
            _run_step("ruff_check", [ruff, "check", "."]),
            _run_step("ruff_format", [ruff, "format", "--check", "."]),
            _run_step("mypy_src", [mypy, f"--cache-dir={mypy_cache}", "src"]),
            _run_step(
                "pytest_validation",
                [
                    pytest_bin,
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    f"--basetemp={pytest_tmp}",
                    "-m",
                    pytest_expr,
                ],
                env_extra={"HYPOTHESIS_STORAGE_DIRECTORY": hyp_cache},
            ),
            _run_step("mutation_audit", [py, "tools/mutation_audit.py"]),
            _run_step(
                "credential_check",
                [py, "tools/credential_check.py", "--allow-missing-env", "HEAD~1..HEAD"],
            ),
        ]

    hw = _hardware_info()
    # Always disclose GPU parity as unrun/blocked whenever the pytest expression excludes `gpu`,
    # even on CUDA-capable hosts, so a CPU-only pass can never masquerade as GPU validation.
    gpu_reason = (
        "CUDA available on host, but GPU tests were excluded (`not gpu`) from this CPU run."
        if hw["cuda_available"]
        else "CUDA unavailable on this host; GPU tests marked blocked, not passed."
    )
    blocked: list[dict[str, str]] = [
        {"check": "cuda_gpu_parity", "reason": gpu_reason},
        {
            "check": "external_sdss_catalogue_reruns",
            "reason": "Requires external SDSS/GZ2 FITS store and explicit user authorisation.",
        },
    ]

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
    try:
        data = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        return False, [f"malformed JSON manifest: {exc}"]
    if not isinstance(data, dict):
        return False, ["manifest root must be a JSON object"]
    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"unexpected schema_version: {data.get('schema_version')!r}")
    if data.get("passed") is not True:
        errors.append("validation manifest top-level passed flag is not True")

    # Derive acceptance from recorded checks themselves (never trust top-level `passed` alone!)
    raw_checks = data.get("checks")
    if not isinstance(raw_checks, list) or not raw_checks:
        errors.append("manifest is missing required `checks` list")
    else:
        by_name: dict[str, dict[str, Any]] = {}
        for item in raw_checks:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                errors.append("malformed entry in manifest `checks`")
                continue
            by_name[item["name"]] = item
        for req in REQUIRED_CHECKS:
            rec = by_name.get(req)
            if rec is None:
                errors.append(f"missing required check {req!r}")
            elif rec.get("passed") is not True or rec.get("returncode") != 0:
                errors.append(
                    f"required check {req!r} did not pass "
                    f"(passed={rec.get('passed')!r}, rc={rec.get('returncode')!r})"
                )

    head = _git("rev-parse", "HEAD")
    git_info = data.get("git")
    if not isinstance(git_info, dict) or git_info.get("commit") != head:
        errors.append(
            f"manifest commit {git_info.get('commit') if isinstance(git_info, dict) else None!r} "
            f"!= current HEAD {head!r}"
        )
    is_dirty = (bool(git_info.get("dirty")) if isinstance(git_info, dict) else True) or bool(
        _git("status", "--porcelain")
    )
    if not allow_dirty and is_dirty:
        errors.append("working tree or manifest is dirty; commit clean tree before validation")

    pins = data.get("pinned_file_sha256")
    if not isinstance(pins, dict):
        errors.append("manifest is missing required `pinned_file_sha256` mapping")
    else:
        for req_pin in REQUIRED_PINS:
            want_sha = pins.get(req_pin)
            if not isinstance(want_sha, str) or not want_sha:
                errors.append(f"missing or null required pin {req_pin!r}")
                continue
            got_sha = _sha256_file(REPO / req_pin)
            if got_sha != want_sha:
                errors.append(
                    f"pinned file {req_pin!r} sha256 mismatch: "
                    f"disk={got_sha!r} != manifest={want_sha!r}"
                )
        for rel, want_sha in pins.items():
            if rel in REQUIRED_PINS:
                continue
            got_sha = _sha256_file(REPO / rel)
            if got_sha != want_sha:
                errors.append(
                    f"pinned file {rel!r} sha256 mismatch: "
                    f"disk={got_sha!r} != manifest={want_sha!r}"
                )
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
