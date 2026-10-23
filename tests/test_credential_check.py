"""The credential check flags a planted secret and passes a clean file.

`tools/credential_check.py` gates every commit and push, so a check that silently passes would let
the SciServer token out. The planted secrets are assembled at run time from fragments, so this
file's own diff does not trip the check it tests.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import credential_check as C  # noqa: E402

pytestmark = pytest.mark.invariant

FAKE_VALUE = "planted" + "-value-" + "0123456789abcdef"  # stands in for a .env token
JWT = "ey" + "J" + "a" * 12 + "." + "b" * 12
ASSIGN = "pass" + "word = " + "'" + "c" * 10 + "'"
UUID = "-".join(["0" * 8, "1" * 4, "2" * 4, "3" * 4, "4" * 12])
CLEAN = "def f(x):\n    return x + 1  # nothing secret here\n"


@pytest.fixture
def env(tmp_path: Path) -> Path:
    p = tmp_path / ".env"
    p.write_text(f"# comment\nSCISERVER_TOKEN={FAKE_VALUE}\nDEBUG=1\n")
    return p


def test_env_values_reads_values_and_skips_comments(env: Path) -> None:
    assert C.env_values(env) == [FAKE_VALUE, "1"]


def test_short_env_values_are_not_matched() -> None:
    assert C.scan("DEBUG=1 and 1 again", ["1"]) == (0, [0, 0, 0])


@pytest.mark.parametrize(
    "text, expect",
    [
        (f"x = {FAKE_VALUE}", (1, [0, 0, 0])),
        (JWT, (0, [1, 0, 0])),
        (ASSIGN, (0, [0, 1, 0])),
        (UUID, (0, [0, 0, 1])),
        (CLEAN, (0, [0, 0, 0])),
    ],
)
def test_scan_counts_each_kind(text: str, expect: tuple[int, list[int]]) -> None:
    assert C.scan(text, [FAKE_VALUE]) == expect


def test_cli_flags_a_planted_secret_and_passes_a_clean_file(tmp_path: Path, env: Path) -> None:
    dirty, clean = tmp_path / "dirty.py", tmp_path / "clean.py"
    dirty.write_text(f"TOKEN_FROM_ENV = '{FAKE_VALUE}'\n")
    clean.write_text(CLEAN)
    assert C.main(["--file", str(dirty), "--env", str(env)]) == 1
    assert C.main(["--file", str(clean), "--env", str(env)]) == 0


def test_cli_scans_the_staged_diff(tmp_path: Path, env: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    (repo / "a.py").write_text(CLEAN)
    subprocess.run([*git, "add", "a.py"], check=True)
    monkeypatch.chdir(repo)
    assert C.main(["--env", str(env)]) == 0
    (repo / "a.py").write_text(CLEAN + f"# {FAKE_VALUE}\n")
    subprocess.run([*git, "add", "a.py"], check=True)
    assert C.main(["--env", str(env)]) == 1


def test_a_missing_env_is_an_error_not_a_pass(tmp_path: Path) -> None:
    f = tmp_path / "clean.py"
    f.write_text(CLEAN)
    assert C.main(["--file", str(f), "--env", str(tmp_path / "absent")]) == 2
