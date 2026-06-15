"""Shared test fixtures.

Materialises the seeded synthetic corpora (``tests/fixtures/generate.py``) into a
session tmp dir, so every integration test runs against deterministic, network-free
data (``docs/spec/testing.md`` §3). Two corpora with *different* galaxies stand in for
the decoupled pretraining and probing sets (D6) — the point of the parity tests is that
*one* fitted pipeline serves both.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_GEN_PATH = Path(__file__).parent / "fixtures" / "generate.py"
_spec = importlib.util.spec_from_file_location("fixture_generate", _GEN_PATH)
assert _spec is not None and _spec.loader is not None
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)
generate_fixture_corpus = _gen.generate_fixture_corpus


@pytest.fixture(scope="session")
def pretraining_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A dozen seeded stamps standing in for the unlabelled pretraining corpus."""
    return generate_fixture_corpus(tmp_path_factory.mktemp("pretrain"), seed=1)


@pytest.fixture(scope="session")
def probing_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A *different* dozen stamps standing in for the GZ2 probing corpus (D6)."""
    return generate_fixture_corpus(tmp_path_factory.mktemp("probe"), seed=2)
