"""Integration tests for the probing harness — the full gated cascade end-to-end.

Runs ``run_probing`` on a synthetic encoder whose embeddings carry one recoverable signal
direction (a planted clean feature) among noise dims (a planted absent feature), so the ladder
must land R1 on the first and R4 on the second — with the controls catching that a probe can
score AUC > 0.5 on the absent feature yet fail the null. Needs sklearn/torch/matplotlib (the
``eval`` extra), so it is the integration tier, not the fast gate. No network, no GPU.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from galaxy_jepa.probing import controls as ctl
from galaxy_jepa.probing.config import ProbingConfig
from galaxy_jepa.probing.extract import LabelProvider, extract_matrix
from galaxy_jepa.probing.ladder import run_ladder
from galaxy_jepa.probing.run import run_probing

pytestmark = pytest.mark.integration

_N = 160
_BRIGHT = np.random.default_rng(0).uniform(0.0, 1.0, _N)
_MODEL_CONFIG = {
    "img_size": 16,
    "patch_size": 16,
    "in_chans": 3,
    "embed_dim": 8,
    "depth": 2,
    "heads": 2,
    "mlp_ratio": 4.0,
    "name": "untrained",
}


class _SignalEncoder(nn.Module):
    """A fixed random projection of the image — one recoverable direction (channel-0 brightness)
    among 16 noise dims, like a real frozen encoder (a shuffled-label probe cannot recover it)."""

    name = "signal_stub"
    embed_dim = 16

    def __init__(self) -> None:
        super().__init__()
        w = np.random.default_rng(7).normal(size=(16, 3 * 16 * 16)).astype(np.float32)
        self.register_buffer("W", torch.from_numpy(w))

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return images.float().reshape(images.shape[0], -1) @ self.W.t()


class _Corpus(Dataset):
    def __len__(self) -> int:
        return _N

    def __getitem__(self, i: int) -> dict:
        tex = torch.randn(3, 16, 16, generator=torch.Generator().manual_seed(100 + i)) * 0.4
        tex[0] += float(_BRIGHT[i])  # brightness on channel 0 → the recoverable direction
        return {"image": tex, "object_id": i}


def _labels() -> LabelProvider:
    rng = np.random.default_rng(1)
    rows = {
        i: {
            "bright_frac": float(_BRIGHT[i]),
            "noise_frac": float(rng.uniform(0.0, 1.0)),  # uncorrelated with the embedding
            "size": float(rng.uniform(1.0, 10.0)),
            "snr": float(rng.uniform(5.0, 40.0)),
        }
        for i in range(_N)
    }
    return LabelProvider(
        rows,
        feature_cols={"bright": "bright_frac", "noise_feat": "noise_frac"},
        nuisance_cols={"size": "size", "snr": "snr"},
    )


def _config() -> ProbingConfig:
    return ProbingConfig(mlp_widths=(8, 16, 32), mlp_epochs=40, n_perm=200, seed=1)


def _run(out_dir) -> object:
    return run_probing(
        _SignalEncoder(),
        _Corpus(),
        _labels(),
        _MODEL_CONFIG,
        config=_config(),
        out_dir=out_dir,
        sky_label_col="snr",
    )


def test_clean_feature_lands_r1_and_absent_feature_lands_r4(tmp_path):
    report = _run(tmp_path)
    verdicts = report.ladder.verdicts
    assert verdicts["bright"].rung == "R1"  # planted recoverable signal → clean linear
    assert verdicts["noise_feat"].rung == "R4"  # planted absent → not recoverable
    # every feature carries a deterministic gate tree (the stamped audit trail) + a named rung
    for v in verdicts.values():
        assert v.rung in {"R1", "R2", "R3", "R4"}
        assert v.gate_tree.render()  # non-empty verdict tree
        assert v.mechanism


def test_controls_catch_the_predict_anything_probe(tmp_path):
    report = _run(tmp_path)
    fc = report.ladder.feature_controls["noise_feat"]
    # the absent feature can score above chance (a probe predicts anything) ...
    assert fc.real_auc > 0.5
    # ... but it does NOT exceed the negative-control null after correction — the controls gate it
    assert report.ladder.existence["noise_feat"].exceeds_null is False


def test_uncertainty_geometry_runs_on_r1_features(tmp_path):
    report = _run(tmp_path)
    assert "bright" in report.uncertainty  # gated on R1/R2
    geom = report.uncertainty["bright"]
    # the axis fitted on the extremes orders the held-out ambiguous middle by vote fraction
    assert geom.n_middle > 0 and geom.spearman > 0.3


def test_figures_and_artefacts_are_emitted_and_stamped(tmp_path):
    report = _run(tmp_path)
    assert "ladder" in report.figures and "controls" in report.figures
    for path in report.figures.values():
        if path.endswith(".png"):
            assert (tmp_path / "figures").exists()
    assert (tmp_path / "stamp.json").exists()
    assert (tmp_path / "config.json").exists()
    summary = json.loads((tmp_path / "ladder_summary.json").read_text())
    assert summary["rungs"]["bright"]["rung"] == "R1"


def test_run_probing_rejects_a_trainable_encoder(tmp_path):
    from galaxy_jepa.models.vit import VisionTransformer

    trainable = VisionTransformer(img_size=16, patch_size=16, embed_dim=8, depth=2, heads=2)
    with pytest.raises(RuntimeError):  # the freeze boundary — labels never bend the encoder
        run_probing(
            trainable,
            _Corpus(),
            _labels(),
            _MODEL_CONFIG,
            config=_config(),
            out_dir=tmp_path,
            sky_label_col="snr",
        )


def test_embeddings_extracted_once_never_re_encoded_per_feature(tmp_path):
    """The cost spine: the ladder slices one embedding matrix — it never re-encodes per feature."""
    calls = {"n": 0}
    encoder = _SignalEncoder()
    base_encode = encoder.encode

    def counting_encode(images):
        calls["n"] += 1
        return base_encode(images)

    encoder.encode = counting_encode  # type: ignore[method-assign]

    dataset, labels, cfg = _Corpus(), _labels(), _config()
    real = extract_matrix(encoder, dataset)
    untrained = ctl.untrained_encoder_matrix(_MODEL_CONFIG, dataset)
    noise = ctl.noise_through_encoder_matrix(encoder, dataset, seed=cfg.seed)
    controls = ctl.ControlEmbeddings(real=real, untrained=untrained, noise=noise)

    calls["n"] = 0  # reset: all encoding is done; the ladder must add none
    ids = [int(o) for o in real.object_ids]
    run_ladder(controls, labels, ids[:120], ids[120:], config=cfg, sky_label_col="snr")
    assert calls["n"] == 0  # zero re-encodes across the whole cascade
