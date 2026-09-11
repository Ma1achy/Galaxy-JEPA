"""Shared test fixtures.

Materialises the seeded synthetic corpora (``tests/fixtures/generate.py``) into a
session tmp dir, so every integration test runs against deterministic, network-free
data (``docs/spec/testing.md`` §3). Two corpora with *different* galaxies stand in for
the decoupled pretraining and probing sets (D6) — the point of the parity tests is that
*one* fitted pipeline serves both.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from galaxy_jepa.data.cache import fit_normalise
from galaxy_jepa.data.sources import DirectorySource
from galaxy_jepa.data.transforms import AsinhStretch, NormalisationFreeze
from galaxy_jepa.data.validity import MIN_REGION_PX

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


def make_freeze(
    mean, std, *, q: float = 4.0, n_sample: int = 0, corpus="pretrain", derived_from="fixture"
) -> NormalisationFreeze:
    """A well-formed freeze from given constants — for tests with no corpus to fit on."""
    freeze = NormalisationFreeze(
        mean=tuple(mean),
        std=tuple(std),
        corpus=corpus,
        n_sample=n_sample,
        seed=0,
        stretch_q=q,
        valid_pixels_only=True,
        detector=f"like-4-neighbour bit-identity, all channels, region >= {MIN_REGION_PX}px",
        # a fixture corpus has no heavy tail to trim; "nothing was trimmed" is stated, not omitted
        trim_rank="total valid-pixel sum-of-squares across all channels, post-asinh",
        trim_threshold=float("inf"),
        trim_excluded=0,
        trim_ids_sha256=hashlib.sha256(b"").hexdigest(),
        content_hash="",
        code_sha="test",
        derived_from=str(derived_from),
        frozen_at="2026-09-11",
        frozen_by="tests",
        rationale="fixture: pinned so the harness can load rather than fit",
    )
    return freeze.model_copy(update={"content_hash": freeze.expected_hash()})


def fit_freeze(pretrain_dir, *, q: float = 4.0, n_sample: int = 10_000) -> NormalisationFreeze:
    """A frozen normalisation for a fixture corpus — what the E5 entrypoint does, in miniature.

    Runs may no longer fit their own normalisation (``harness._build_pipeline`` refuses), so a
    test that exercises the harness has to pin one first. Fitting here rather than hard-coding
    constants keeps the tests honest about the order: fit once, freeze, then run.
    """
    fit = fit_normalise(DirectorySource(pretrain_dir), AsinhStretch(q=q), n_sample=n_sample, seed=0)
    return make_freeze(
        fit.valid.mean or (),
        fit.valid.std or (),
        q=q,
        n_sample=fit.n_stamps,
        derived_from=pretrain_dir,
    )
