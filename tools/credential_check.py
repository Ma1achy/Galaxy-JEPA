"""Credential check before a commit or push: does a diff (or a file) carry a secret?

Scans for every value in the repo's `.env` (the SciServer token lives only there, CLAUDE.md
"Token-only-in-artifacts") and for three secret shapes: a JWT, a quoted `token=`/`secret=`/
`password=`/`api_key=` assignment, and a UUID. Prints **hit counts only**, never a value, and exits
1 on any hit, so it can gate a commit with `&&`.

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
    re.compile(r"(?i)(token|secret|password|api[_-]?key)\s*[=:]\s*['\"][^'\"]{8,}"),
    re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"),
)
MIN_VALUE_LEN = 8  # shorter .env values (flags, small ints) would match by accident


def env_values(path: Path) -> list[str]:
    """The values of `KEY=value` lines, quotes stripped; comments and blank lines skipped."""
    out = []
    for line in path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            out.append(line.split("=", 1)[1].strip().strip("'\""))
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Credential check: hit counts only, exit 1 on a hit.")
    ap.add_argument("range", nargs="*", help="git diff arguments (default: --cached)")
    ap.add_argument("--file", type=Path, help="scan this file instead of a diff")
    ap.add_argument("--env", type=Path, help="the .env to compare against (default: the repo's)")
    a = ap.parse_args(argv)
    env = a.env or default_env()
    if not env.is_file():
        print(f"credential check: no .env at {env}; refusing to report a pass", file=sys.stderr)
        return 2
    if a.file:
        text = a.file.read_text()
    else:
        diff = subprocess.run(["git", "diff", *(a.range or ["--cached"])], capture_output=True)
        if diff.returncode:
            print(f"credential check: git diff failed ({diff.returncode})", file=sys.stderr)
            return 2
        text = diff.stdout.decode("utf-8", "replace")
    value_hits, pattern_hits = scan(text, env_values(env))
    print(f"env-value hits: {value_hits} pattern hits: {pattern_hits}")
    return 1 if value_hits or any(pattern_hits) else 0


if __name__ == "__main__":
    sys.exit(main())
