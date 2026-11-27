"""Credential check before a commit or push: does a diff (or a file) carry a secret?

Scans for every value in the repo's `.env` (the SciServer token lives only there, CLAUDE.md
"Token-only-in-artifacts") and for four secret shapes: a JWT, a quoted `token=`/`secret=`/
`password=`/`api_key=` assignment, a UUID, and an Anthropic API key. Prints **hit counts only**,
never a value, and exits 1 on any hit, so it can gate a commit with `&&`.

    .venv/bin/python tools/credential_check.py                     # the staged diff
    .venv/bin/python tools/credential_check.py origin/main..HEAD   # an unpushed range
    .venv/bin/python tools/credential_check.py --file PATH         # one file

`.env` is found beside the repository's common git directory, so the check reads the main tree's
`.env` from a worktree too. A missing `.env` is an error (exit 2), not a clean pass: with nothing to
compare against, a pass would mean nothing.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

PATTERNS = (
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    re.compile(
        r"(?i)(?:token|secret|password|api[_-]?key)\s*[=:]\s*"
        r"(?:['\"][^'\"]{8,}|(?=[A-Za-z0-9._/+=-]{12,}(?:\s|$|#))"
        r"(?=[^ \t\r\n]*[0-9])[A-Za-z0-9._/+=-]{12,})"
    ),
    re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"),
    re.compile(r"sk-ant-[A-Za-z0-9_-]{8,}"),  # Anthropic API key; the bare prefix is no hit
    re.compile(r"(?i)Authorization\s*:\s*Bearer\s+[A-Za-z0-9._~+/=-]{12,}"),
)
MIN_VALUE_LEN = 8  # shorter .env values (flags, small ints) would match by accident


def env_values(path: Path) -> list[str]:
    """The values of `KEY=value` lines, quotes stripped; comments and blank lines skipped.

    For URL-like values (``http://`` or ``https://``), also includes the scheme-stripped form
    without trailing slashes so leaked endpoints without a scheme are caught.
    """
    out: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            raw = line.split("=", 1)[1].strip().strip("'\"")
            if raw not in seen:
                seen.add(raw)
                out.append(raw)
            if raw.startswith(("http://", "https://")):
                stripped = raw.split("://", 1)[1].rstrip("/")
                if len(stripped) >= MIN_VALUE_LEN and stripped not in seen:
                    seen.add(stripped)
                    out.append(stripped)
    return out


def scan(text: str, secrets: Iterable[str]) -> tuple[int, list[int]]:
    """(occurrences of any secret value, per-pattern match counts) in `text`."""
    value_hits = sum(text.count(v) for v in secrets if len(v) >= MIN_VALUE_LEN)
    return value_hits, [len(p.findall(text)) for p in PATTERNS]


def default_env() -> Path:
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return Path(common).parent / ".env"


def _tracked_files_text() -> str:
    ls = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True)
    chunks: list[str] = []
    for rel in ls.stdout.splitlines():
        p = Path(rel)
        if not p.is_file() or p.name == ".env":
            continue
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".pt", ".npz", ".npy", ".bin", ".fits"}:
            continue
        chunks.append(p.read_bytes().decode("utf-8", "replace"))
    return "\n".join(chunks)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Credential check: hit counts only, exit 1 on a hit.")
    ap.add_argument("range", nargs="*", help="git diff arguments (default: --cached)")
    ap.add_argument("--file", type=Path, help="scan this file instead of a diff")
    ap.add_argument("--tracked", action="store_true", help="scan all tracked text files")
    ap.add_argument("--env", type=Path, help="the .env to compare against (default: the repo's)")
    ap.add_argument(
        "--allow-missing-env",
        action="store_true",
        help="run pattern scans even when .env is absent (e.g. GitHub Actions CI)",
    )
    a = ap.parse_args(argv)
    env = a.env or default_env()
    secrets: list[str] = []
    if env.is_file():
        secrets = env_values(env)
    elif a.allow_missing_env:
        print(
            f"credential check: no .env at {env}; running pattern-only scan (--allow-missing-env)",
            file=sys.stderr,
        )
    else:
        print(f"credential check: no .env at {env}; refusing to report a pass", file=sys.stderr)
        return 2
    if a.file:
        text = a.file.read_bytes().decode("utf-8", "replace")
    elif a.tracked:
        text = _tracked_files_text()
    else:
        diff = subprocess.run(["git", "diff", *(a.range or ["--cached"])], capture_output=True)
        if diff.returncode:
            print(f"credential check: git diff failed ({diff.returncode})", file=sys.stderr)
            return 2
        text = diff.stdout.decode("utf-8", "replace")
    value_hits, pattern_hits = scan(text, secrets)
    print(f"env-value hits: {value_hits} pattern hits: {pattern_hits}")
    return 1 if value_hits or any(pattern_hits) else 0


if __name__ == "__main__":
    sys.exit(main())
