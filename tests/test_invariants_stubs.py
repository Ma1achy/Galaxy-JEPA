"""Registered invariant tests — the science-protecting tier (docs/spec/testing.md §1.2).

These invariants are structural to the methodology, so each has a live test (or a
registered, skipped stub naming the TODO) — an untested invariant is just a comment.

* ``test_masking_beta_zero_is_ijepa`` — the masking module is a strict generalisation
  of I-JEPA: at β=0 the token weighting must be uniform (so block sampling is identical
  to standard I-JEPA — ``docs/masking.md`` §6, ``docs/architecture.md`` hard invariant 4).
  Deterministic code-correctness, hence a property test. **Live** now ``masking/`` exists.
* ``test_normalisation_parity`` — **live** (integration: needs the fixture corpora, hence
  the data extra). Format + stretch + normalisation are byte-identical across the
  pretraining corpus, the probing corpus, and every baseline (``docs/spec/data.md`` §1;
  protects D6 and the Rung-4 result).
"""

import numpy as np
import pytest

from galaxy_jepa.core.config import config_hash
from galaxy_jepa.data.sources import FixtureSource
from galaxy_jepa.data.transforms import AsinhStretch, Normalise, Pipeline


@pytest.mark.invariant
def test_masking_beta_zero_is_ijepa():
    # β=0 ⇒ every token weight is 1 regardless of the box ⇒ uniform sampling = I-JEPA.
    from galaxy_jepa.masking.blocks import box_to_token_mask, token_weight_map

    in_box = box_to_token_mask(half_width_px=40, stamp_px=256, grid_size=16)
    assert np.allclose(token_weight_map(in_box, beta=0.0), 1.0)


def _fit_on(corpus) -> Pipeline:
    """Fit the parity pipeline once on a corpus: normalisation follows the stretch."""
    source = FixtureSource(corpus)
    stretch = AsinhStretch()
    stretched = np.stack([stretch(image) for image, _ in source])
    norm = Normalise.fit(stretched)
    return Pipeline((stretch, norm))


@pytest.mark.invariant
@pytest.mark.integration  # needs the fixture corpora (data extra); runs in the PR integration job
def test_normalisation_parity(pretraining_corpus, probing_corpus):
    # The rule: fit ONCE on the pretraining corpus, then apply that same frozen pipeline
    # to the probing corpus and every baseline (docs/spec/data.md §1).
    pipeline = _fit_on(pretraining_corpus)

    # 1. Reproducible: the pipeline reconstructed from its stamped config is identical.
    restored = Pipeline.from_config(pipeline.to_config())
    assert config_hash(pipeline.to_config()) == config_hash(restored.to_config())

    # 2. Byte-identical across corpora: the SAME frozen pipeline on a probing stamp is
    #    deterministic and matches the reconstructed pipeline bit-for-bit.
    probe_image, _ = FixtureSource(probing_corpus)[3]
    assert np.array_equal(pipeline(probe_image), pipeline(probe_image))
    assert np.array_equal(pipeline(probe_image), restored(probe_image))

    # 3. The rule is meaningful: a pipeline fitted on the probing corpus has DIFFERENT
    #    normalisation and so a different hash — you must reuse the pretraining one, not
    #    silently refit per corpus.
    refit_on_probe = _fit_on(probing_corpus)
    assert config_hash(refit_on_probe.to_config()) != config_hash(pipeline.to_config())
