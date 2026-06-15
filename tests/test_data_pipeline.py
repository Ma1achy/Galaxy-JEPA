"""Integration test for the data pipeline (data/ on a fixture corpus).

Tier: integration (``docs/spec/testing.md`` §1.3) — FITS stamps → stretch → normalise →
tensor-ready arrays, with **no network**, plus the stretch-sanity property
(``docs/spec/validation.md`` ``T2.stretch-sanity``): a faint arm survives stretch +
normalise while the blank-sky floor stays controlled.
"""

import numpy as np
import pytest

from galaxy_jepa.data.sources import FixtureSource
from galaxy_jepa.data.transforms import AsinhStretch, Normalise, Pipeline


def _fit_pipeline(source: FixtureSource) -> Pipeline:
    stretch = AsinhStretch()
    stretched = np.stack([stretch(image) for image, _ in source])
    norm = Normalise.fit(stretched)
    return Pipeline((stretch, norm))


@pytest.mark.integration
def test_pipeline_end_to_end(pretraining_corpus):
    source = FixtureSource(pretraining_corpus)
    pipeline = _fit_pipeline(source)
    image, meta = source[2]
    out = pipeline(image)
    assert out.shape == image.shape
    assert np.all(np.isfinite(out))
    assert meta["pixel_scale"] == pytest.approx(0.396)


@pytest.mark.integration
def test_stretch_sanity_faint_arm_survives(pretraining_corpus):
    source = FixtureSource(pretraining_corpus)
    pipeline = _fit_pipeline(source)

    # By construction (tests/fixtures/generate.py): index 0 = blank sky, 1 = faint arm.
    sky_image, sky_meta = source[0]
    arm_image, arm_meta = source[1]
    assert sky_meta["kind"] == "sky"
    assert arm_meta["kind"] == "faint_arm"

    sky_out = pipeline(sky_image)
    arm_out = pipeline(arm_image)

    # Faint arm survives: its peak sits clearly above the sky control's 99th percentile.
    assert arm_out.max() > np.percentile(sky_out, 99) + 0.5
    # Floor controlled: the sky output isn't amplified into noise that swamps signal.
    assert np.std(sky_out) < 2.0
