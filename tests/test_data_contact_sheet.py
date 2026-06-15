"""Integration smoke test for the contact-sheet + stretch-sanity tooling.

Tier: integration (``docs/spec/testing.md`` §1.3) — renders the eyeball-gate sheet (with
the Petrosian box overlay) from the fixture corpus, and checks the stretch-sanity metric
on the planted faint-arm / sky-control fixtures. The *real* sheet needs the devcontainer
pull; this proves the generator runs.
"""

import numpy as np
import pytest

from galaxy_jepa.data.contact_sheet import build_contact_sheet
from galaxy_jepa.data.sanity import stretch_sanity
from galaxy_jepa.data.sources import DirectorySource
from galaxy_jepa.data.transforms import AsinhStretch, Normalise, Pipeline


def _fit_pipeline(source: DirectorySource) -> Pipeline:
    stretch = AsinhStretch()
    stretched = np.stack([stretch(image) for image, _ in source])
    return Pipeline((stretch, Normalise.fit(stretched)))


@pytest.mark.integration
def test_contact_sheet_renders_with_box_overlay(pretraining_corpus, tmp_path):
    source = DirectorySource(pretraining_corpus)
    pipeline = _fit_pipeline(source)
    out = build_contact_sheet(source, pipeline, tmp_path / "sheet.png", k=2.5)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.integration
def test_stretch_sanity_metric_on_fixtures(pretraining_corpus):
    source = DirectorySource(pretraining_corpus)
    pipeline = _fit_pipeline(source)
    sky_image, _ = source[0]
    arm_image, _ = source[1]
    result = stretch_sanity([arm_image], [sky_image], pipeline)
    assert result.passes
    assert result.margin > 0.5
