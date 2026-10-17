"""Ladder v2's encoder/output contract is inert when unset and refuses a half-set pair.

`artifacts/ladder_env.py` is imported by path (as `test_probe_bank` does): it is the one site every
ladder driver reads its encoder, output directory and untrained bank through. Unset, each accessor
must return the driver's own default, so M's runs stay reproducible; `LADDER_ENCODER` without
`LADDER_OUT` would send a second encoder's records into M's directory, so it raises.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "artifacts"))
import ladder_env as L  # noqa: E402

pytestmark = pytest.mark.invariant

VARS = ("LADDER_ENCODER", "LADDER_OUT", "LADDER_UNTRAINED_BANK", "LADDER_SPLIT", "LADDER_TAG")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for v in VARS:
        monkeypatch.delenv(v, raising=False)


def test_unset_every_accessor_returns_the_drivers_default():
    out = Path("artifacts/out")
    assert not L.active()
    assert L.encoder("runs/m/encoder.pt") == Path("runs/m/encoder.pt")
    assert L.encoder(None) is None
    assert L.out(out) is out
    assert L.untrained_bank("artifacts/out/p1_untrained_bank.json") == (
        "artifacts/out/p1_untrained_bank.json"
    )
    assert not L.identical_split()
    assert L.tag("m1") == "m1"


def test_set_together_the_pair_overrides_the_literal(monkeypatch):
    monkeypatch.setenv("LADDER_ENCODER", "runs/m_v2_s1/encoder.pt")
    monkeypatch.setenv("LADDER_OUT", "/tmp/ladder/A2")
    assert L.active()
    assert L.encoder("runs/m/encoder.pt") == Path("runs/m_v2_s1/encoder.pt")
    assert L.out(Path("artifacts/out")) == Path("/tmp/ladder/A2")


@pytest.mark.parametrize("lone", ["LADDER_ENCODER", "LADDER_OUT"])
def test_one_of_the_pair_alone_refuses(monkeypatch, lone):
    monkeypatch.setenv(lone, "x")
    with pytest.raises(SystemExit, match="without its pair"):
        L.active()
    with pytest.raises(SystemExit):
        L.out(Path("artifacts/out"))


def test_only_the_identical_split_is_named(monkeypatch):
    monkeypatch.setenv("LADDER_SPLIT", "identical")
    assert L.identical_split()
    monkeypatch.setenv("LADDER_SPLIT", "v2")
    with pytest.raises(SystemExit, match="identical"):
        L.identical_split()


def test_the_nine_are_the_aligned_scorers_nine():
    import aligned_score

    assert frozenset(int(i) for i in aligned_score.NINE) == L.NINE
    assert len(L.NINE) == 9
