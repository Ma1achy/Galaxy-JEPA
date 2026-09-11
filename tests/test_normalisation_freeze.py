"""The normalisation freeze (Brief E5) — the parity lock made unforgeable.

These pin a *defect that shipped*, not a hypothetical. ``harness._build_pipeline`` used to call
``fit_normalise`` on every run; the subsample is ``rng.choice(len(source), n)``, so when the
pretraining corpus grew from 10,000 stamps to 826,968 the same seed began drawing a different
sample and the constants moved with nothing on disk recording it. Two runs could differ in their
input transform, with identical configs, undetectably.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from galaxy_jepa.data.transforms import AsinhStretch, NormalisationFreeze, Normalise
from galaxy_jepa.harness import _build_pipeline

pytestmark = pytest.mark.invariant


def _freeze(**overrides) -> NormalisationFreeze:
    base: dict[str, Any] = dict(
        mean=(0.1, 0.2, 0.3),
        std=(1.1, 1.2, 1.3),
        corpus="pretrain",
        n_sample=8000,
        seed=0,
        stretch_q=4.0,
        valid_pixels_only=True,
        detector="like-4-neighbour bit-identity, all channels, region >= 256px",
        trim_rank="total valid-pixel sum-of-squares across all channels, post-asinh",
        trim_threshold=36779.561450152854,
        trim_excluded=827,
        trim_ids_sha256="bf2a8d0b" * 8,
        content_hash="",
        code_sha="abc123",
        derived_from="data/pretrain (826968 stamps)",
        frozen_at="2026-09-11",
        frozen_by="tests",
        rationale="fixture",
    )
    base.update(overrides)
    freeze = NormalisationFreeze(**base)
    if not overrides.get("content_hash"):
        freeze = freeze.model_copy(update={"content_hash": freeze.expected_hash()})
    return freeze


def test_a_run_without_a_frozen_statistic_is_refused_not_fitted():
    """The defect itself: a missing record must stop the run, never trigger a quiet refit."""
    with pytest.raises(ValueError, match="may not fit one for itself"):
        _build_pipeline(q=4.0, freeze=None)


def test_a_statistic_fitted_under_a_different_stretch_is_refused():
    """These are post-stretch statistics; another Q makes them meaningless, not merely stale."""
    with pytest.raises(ValueError, match="post-stretch statistics"):
        _build_pipeline(q=8.0, freeze=_freeze(stretch_q=4.0))


def test_building_twice_loads_the_same_constants_and_never_refits():
    freeze = _freeze()
    one, two = _build_pipeline(q=4.0, freeze=freeze), _build_pipeline(q=4.0, freeze=freeze)
    norms = [t for p in (one, two) for t in p.transforms if isinstance(t, Normalise)]
    assert len(norms) == 2
    assert norms[0].mean == norms[1].mean == freeze.mean
    assert norms[0].std == norms[1].std == freeze.std
    # and the constants are the record's, not something derived from a corpus at build time
    assert [t for t in one.transforms if isinstance(t, AsinhStretch)][0].q == 4.0


def test_a_hand_edited_record_is_refused():
    """Editing a value without re-fitting would make the provenance a lie — so it cannot."""
    tampered = _freeze().model_copy(update={"mean": (9.9, 9.9, 9.9)})
    with pytest.raises(ValueError, match="edited since it was fitted"):
        tampered.to_normalise()
    with pytest.raises(ValueError, match="edited since it was fitted"):
        _build_pipeline(q=4.0, freeze=tampered)


def test_the_content_hash_covers_what_the_statistic_depends_on():
    """Anything that changes the numbers must change the hash, or the guard is decorative."""
    base = _freeze()
    for field, value in (
        ("mean", (0.9, 0.2, 0.3)),
        ("std", (9.1, 1.2, 1.3)),
        ("corpus", "probe"),
        ("n_sample", 4000),
        ("seed", 7),
        ("stretch_q", 8.0),
        ("valid_pixels_only", False),
        ("detector", "something else"),
        # the trim is a degree of freedom, so it has to move the hash like any other
        ("trim_rank", "brightest pixel"),
        ("trim_threshold", 1.0),
        ("trim_excluded", 0),
        ("trim_ids_sha256", "00" * 32),
    ):
        assert base.model_copy(update={field: value}).expected_hash() != base.content_hash, field
    # provenance prose is *not* part of the statistic, so it must not move the hash
    for field in ("frozen_by", "rationale", "derived_from", "frozen_at", "code_sha"):
        assert base.model_copy(update={field: "changed"}).expected_hash() == base.content_hash


def test_the_freeze_rides_in_the_config_hash():
    """The record must be hashed and stamped, or a result cannot say where its inputs came from."""
    from galaxy_jepa.core.config import config_hash
    from galaxy_jepa.harness import HarnessConfig, PathsConfig

    paths = PathsConfig(pretrain_dir="a", probe_dir="b", out_dir="c")
    open_run = HarnessConfig(paths=paths)
    frozen = HarnessConfig(paths=paths, normalisation=_freeze())
    assert config_hash(open_run.determining_dump()) != config_hash(frozen.determining_dump())
    assert "normalisation" in frozen.determining_dump()


def test_valid_pixel_fit_differs_from_the_naive_one_when_padding_is_present():
    """The reason for the whole detour: constant padding drags the statistic."""
    from galaxy_jepa.data.cache import fit_normalise

    rng = np.random.default_rng(0)

    class _Src:
        def __len__(self):
            return 12

        def __getitem__(self, i):
            img = rng.normal(0.5, 0.2, size=(3, 64, 64)).astype(np.float32)
            img[:, :, :16] = 0.0  # a padded strip, as ~1 stamp in 7 really carries
            return img, {"object_id": i}

    fit = fit_normalise(_Src(), AsinhStretch(q=4.0), n_sample=12, seed=0)
    assert fit.valid_pixel_fraction == pytest.approx(0.75, abs=0.01)
    for naive_mean, valid_mean, _, _ in fit.contamination():
        assert valid_mean > naive_mean, "padding at 0 must drag the naive mean down"
