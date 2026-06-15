"""Unit tests for the transform pipeline (data/transforms.py).

Pins: asinh config round-trip + hash stability and its faint-end boost; the stateful
``Normalise`` fit/freeze contract (unfitted calls fail loud); fitted statistics enter
the config hash; pipeline composition + round-trip.
"""

import numpy as np
import pytest

from galaxy_jepa.core.config import config_hash
from galaxy_jepa.data.transforms import AsinhStretch, Normalise, Pipeline


def test_asinh_config_round_trip_and_hash_stable():
    stretch = AsinhStretch(q=6.0, flux_scale=(0.8, 1.0, 1.2))
    config = stretch.to_config()
    restored = AsinhStretch.from_config(config)
    assert restored.q == 6.0
    assert restored.flux_scale == (0.8, 1.0, 1.2)
    assert config_hash(config) == config_hash(restored.to_config())


def test_asinh_boosts_faint_end_and_anchors_flux_scale():
    stretch = AsinhStretch(q=8.0, flux_scale=(1.0,))
    faint = np.full((1, 1, 1), 0.05)
    anchor = np.full((1, 1, 1), 1.0)
    # Faint signal is lifted above its linear value; flux_scale maps to 1.
    assert stretch(faint)[0, 0, 0] > 0.05
    assert stretch(anchor)[0, 0, 0] == pytest.approx(1.0)


def test_asinh_rejects_bad_params_and_channel_mismatch():
    with pytest.raises(ValueError):
        AsinhStretch(q=0.0)
    with pytest.raises(ValueError):
        AsinhStretch(flux_scale=(1.0, -1.0))
    with pytest.raises(ValueError):
        AsinhStretch(flux_scale=(1.0, 1.0)).__call__(np.zeros((3, 4, 4)))


def test_normalise_unfitted_is_loud():
    with pytest.raises(RuntimeError):
        Normalise()(np.zeros((3, 4, 4)))


def test_normalise_fit_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    images = rng.normal(5.0, 2.0, size=(8, 3, 16, 16))
    norm = Normalise.fit(images)
    out = np.stack([norm(img) for img in images])
    assert np.allclose(out.mean(axis=(0, 2, 3)), 0.0, atol=1e-6)
    assert np.allclose(out.std(axis=(0, 2, 3)), 1.0, atol=1e-6)


def test_normalise_stats_enter_config_hash():
    a = Normalise(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
    b = Normalise(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 2.0))
    assert config_hash(a.to_config()) != config_hash(b.to_config())
    restored = Normalise.from_config(a.to_config())
    assert restored.fitted and restored.std == (1.0, 1.0, 1.0)


def test_normalise_rejects_zero_variance_channel():
    images = np.ones((4, 2, 8, 8))  # zero variance in both channels
    with pytest.raises(ValueError):
        Normalise.fit(images)


def test_pipeline_composition_and_round_trip():
    stretch = AsinhStretch(q=8.0, flux_scale=(1.0, 1.0, 1.0))
    norm = Normalise(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
    pipeline = Pipeline((stretch, norm))
    image = np.abs(np.random.default_rng(1).normal(size=(3, 8, 8)))
    expected = norm(stretch(image))
    assert np.allclose(pipeline(image), expected)
    restored = Pipeline.from_config(pipeline.to_config())
    assert np.allclose(restored(image), expected)
    assert config_hash(pipeline.to_config()) == config_hash(restored.to_config())
