"""Registered invariant tests — the science-protecting tier (docs/spec/testing.md §1.2).

These invariants are structural to the methodology, so each has a live end-to-end test:

* ``test_masking_beta_zero_is_ijepa`` — Hypothesis property test calling
  :meth:`MultiBlockMasker.sample` at ``beta=0`` verifying uniform token weighting, box
  independence, and zero target/context overlap (``docs/masking.md`` §6, hard invariant 4).
* ``test_uncertainty_geometry_firewall_rejects_poisoned_middle`` — end-to-end check on
  :func:`galaxy_jepa.probing.uncertainty.uncertainty_geometry` confirming the ambiguous middle
  cannot leak into axis estimation even when poisoned or non-finite.
* ``test_probing_never_attaches_gradients_to_frozen_encoder`` — end-to-end check on a real
  parameterised :class:`VisionTransformer` confirming frozen probing leaves ``p.grad is None``
  and weights bit-identical (hard invariant 1).
* ``test_normalisation_parity`` — format + stretch + normalisation are byte-identical across
  corpora (``docs/spec/data.md`` §1; protects D6 and the Rung-4 result).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from galaxy_jepa.core.config import config_hash
from galaxy_jepa.data.sources import FixtureSource
from galaxy_jepa.data.transforms import AsinhStretch, Normalise, Pipeline
from galaxy_jepa.masking.blocks import (
    MaskConfig,
    MultiBlockMasker,
    box_to_token_mask,
    token_weight_map,
)
from galaxy_jepa.models.vit import VisionTransformer, load_frozen_encoder, save_encoder
from galaxy_jepa.probing.extract import EmbeddingMatrix, LabelProvider, extract_matrix
from galaxy_jepa.probing.uncertainty import uncertainty_geometry


@pytest.mark.invariant
@settings(max_examples=25, deadline=None)
@given(
    half_w_a=st.floats(min_value=16.0, max_value=120.0),
    half_w_b=st.floats(min_value=16.0, max_value=120.0),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_masking_beta_zero_is_ijepa(half_w_a: float, half_w_b: float, seed: int) -> None:
    # β=0 ⇒ every token weight is 1 regardless of box ⇒ MultiBlockMasker.sample is box-blind
    # and strictly non-overlapping (standard I-JEPA).
    in_a = box_to_token_mask(half_width_px=half_w_a, stamp_px=256, grid_size=16)
    in_b = box_to_token_mask(half_width_px=half_w_b, stamp_px=256, grid_size=16)
    w_a = token_weight_map(in_a, beta=0.0)
    w_b = token_weight_map(in_b, beta=0.0)
    assert np.allclose(w_a, 1.0)
    assert np.allclose(w_b, 1.0)

    masker = MultiBlockMasker(grid_size=16, config=MaskConfig())
    ctx_a, tgt_a = masker.sample(w_a[None, :, :], seed=seed)
    ctx_b, tgt_b = masker.sample(w_b[None, :, :], seed=seed)
    assert torch.equal(ctx_a, ctx_b)
    assert len(tgt_a) == len(tgt_b) == 4
    ctx_set = set(ctx_a[0].tolist())
    assert len(ctx_set) >= 1
    for ta, tb in zip(tgt_a, tgt_b, strict=True):
        assert torch.equal(ta, tb)
        assert ctx_set.isdisjoint(ta[0].tolist())


@pytest.mark.invariant
def test_uncertainty_geometry_firewall_rejects_poisoned_middle() -> None:
    """Poisoning the ambiguous middle must not alter the fitted axis (zero leakage)."""
    rng = np.random.default_rng(7)
    n = 160
    ids = list(range(n))
    # First 80 extremes (0.05 or 0.95), next 80 ambiguous middle (0.30..0.70)
    fracs = np.concatenate(
        [
            np.where(np.arange(80) % 2 == 0, 0.05, 0.95),
            np.linspace(0.30, 0.70, 80),
        ]
    )
    x = rng.normal(size=(n, 6))
    x[:, 0] += 20.0 * (fracs - 0.5)
    rows = {i: {"f_frac": float(fracs[i]), "f_n": 30.0} for i in ids}
    labels = LabelProvider(
        rows,
        feature_cols={"f": "f_frac"},
        nuisance_cols={},
        vote_count_min=1,
    )
    res_clean = uncertainty_geometry(
        EmbeddingMatrix(object_ids=np.asarray(ids, dtype=np.int64), x=x, encoder_name="clean"),
        labels,
        "f",
        ids,
        train_ids=ids[:120],
        test_ids=ids[120:],
        n_perm=200,
        seed=0,
    )
    # Poison the ambiguous middle in the training half with massive opposite-sign values.
    x_poison = x.copy()
    x_poison[80:120, 0] = -1e4
    res_poison = uncertainty_geometry(
        EmbeddingMatrix(
            object_ids=np.asarray(ids, dtype=np.int64), x=x_poison, encoder_name="poison"
        ),
        labels,
        "f",
        ids,
        train_ids=ids[:120],
        test_ids=ids[120:],
        n_perm=200,
        seed=0,
    )
    # Because the axis is fitted strictly on training extremes (ids[:80]), poisoning ids[80:120]
    # leaves the test-middle projections and Spearman correlation bit-identical.
    assert np.allclose(res_clean.distances, res_poison.distances)
    assert res_clean.spearman == pytest.approx(res_poison.spearman)
    assert res_clean.spearman > 0.5


@pytest.mark.invariant
def test_probing_never_attaches_gradients_to_frozen_encoder(tmp_path) -> None:
    """Hard invariant 1: frozen probing leaves every encoder parameter untouched with grad=None."""
    torch.manual_seed(0)
    enc = VisionTransformer(img_size=32, patch_size=8, embed_dim=16, depth=2, heads=2)
    ckpt = save_encoder(enc, tmp_path / "encoder.pt")
    frozen = load_frozen_encoder(ckpt)
    before = {k: v.detach().clone() for k, v in frozen.state_dict().items()}

    class _TinyDataset:
        def __init__(self, n: int = 8) -> None:
            self.object_ids = list(range(n))
            self._x = torch.randn(n, 3, 32, 32)

        def __len__(self) -> int:
            return len(self.object_ids)

        def __getitem__(self, i: int) -> dict[str, object]:
            return {"image": self._x[i], "object_id": self.object_ids[i]}

    mat = extract_matrix(frozen, _TinyDataset(), device="cpu", batch_size=4)
    assert mat.x.shape == (8, 16)
    for p in frozen.parameters():
        assert not p.requires_grad
        assert p.grad is None
    for k, v in frozen.state_dict().items():
        assert torch.equal(v, before[k])


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
