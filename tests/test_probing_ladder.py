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
    """Stands in for a *trained* encoder: it exposes a **nonlinear** image statistic linearly.

    The planted property is channel-0 **contrast** (its standard deviation), carried on one
    dimension among 15 random-projection noise dims. The choice matters: an earlier version
    planted channel-0 *brightness*, which is a linear function of the pixels — so the
    untrained-encoder control recovered it just as well, and with an honest five-null bar the
    "clean" feature correctly failed existence. A statistic a random projection cannot read
    linearly is what makes "the pretraining mattered" the actual thing under test.
    """

    name = "signal_stub"
    embed_dim = 16

    def __init__(self) -> None:
        super().__init__()
        w = np.random.default_rng(7).normal(size=(15, 3 * 16 * 16)).astype(np.float32)
        self.register_buffer("W", torch.from_numpy(w))

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        x = images.float()
        contrast = x[:, 0].reshape(x.shape[0], -1).std(dim=1, keepdim=True)
        noise = x.reshape(x.shape[0], -1) @ self.W.t()
        return torch.cat([contrast * 10.0, noise], dim=1)


class _Corpus(Dataset):
    def __len__(self) -> int:
        return _N

    def __getitem__(self, i: int) -> dict:
        tex = torch.randn(3, 16, 16, generator=torch.Generator().manual_seed(100 + i))
        tex[0] *= 0.2 + float(_BRIGHT[i])  # contrast on channel 0 → the recoverable direction
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
        vote_count_min=21,
        feature_cols={"bright": "bright_frac", "noise_feat": "noise_frac"},
        nuisance_cols={"size": "size", "snr": "snr"},
        # Synthetic sizes, no deblending tail — say so rather than let the default look for a
        # `petrorad_suspect` column this schema was never going to have.
        nuisance_flag_cols={},
    )


def _config() -> ProbingConfig:
    return ProbingConfig(
        vote_count_min=21,
        mlp_widths=(8, 16, 32),
        mlp_epochs=40,
        n_perm=200,
        # 50 draws (the default) cap the attainable p at 1/51 ≈ 0.0196, which cannot clear the
        # BY rank-1 bar even for a family of 2 — the null-resolution guard rejects it. 200 draws
        # give 0.005, so a planted signal can actually express significance.
        n_null_draws=200,
        # The grounded floor is ≥10,000 shuffles; a fixture that wants fewer must say so, and
        # the declaration is stamped onto the artefact rather than silently weakening the test.
        escape_hatches=("reduced_permutations",),
        seed=1,
    )


def _controls_and_ids(seed: int = 1):
    """The three embedding sources and the id list — built once, sliced per feature."""
    dataset, encoder = _Corpus(), _SignalEncoder()
    real = extract_matrix(encoder, dataset)
    untrained = ctl.untrained_encoder_matrix(_MODEL_CONFIG, dataset)
    noise = ctl.noise_through_encoder_matrix(encoder, dataset, seed=seed)
    extra = tuple(ctl.untrained_encoder_matrix(_MODEL_CONFIG, dataset, seed=k) for k in (1, 2))
    controls = ctl.ControlEmbeddings(
        real=real, untrained=untrained, noise=noise, untrained_extra=extra
    )
    return controls, [int(o) for o in real.object_ids]


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
    # and it is R1 *because* the trained stub beats the untrained-encoder null, not merely
    # because it beats chance — that is what the five-null bar is for
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
    extra = tuple(ctl.untrained_encoder_matrix(_MODEL_CONFIG, dataset, seed=k) for k in (1, 2))
    controls = ctl.ControlEmbeddings(
        real=real, untrained=untrained, noise=noise, untrained_extra=extra
    )

    calls["n"] = 0  # reset: all encoding is done; the ladder must add none
    ids = [int(o) for o in real.object_ids]
    run_ladder(controls, labels, ids[:120], ids[120:], config=cfg, sky_label_col="snr")
    assert calls["n"] == 0  # zero re-encodes across the whole cascade


def test_two_identical_invocations_produce_identical_verdicts(tmp_path):
    """A verdict must be determined by the run stamp and nothing else.

    ``(config_hash, code_sha, data_snapshot, seed)`` fully determines the outcome. This is the
    regression guard for the class of bug the untrained-encoder control had: an unseeded
    component moves the existence bar between runs, so a feature near the threshold flips rung
    on a rerun of the *same* config. The comparison is over the whole stamped summary — rungs,
    p-values, and every null component — not just the headline.
    """
    first = _run(tmp_path / "a")
    second = _run(tmp_path / "b")

    summary_a = json.loads((tmp_path / "a" / "ladder_summary.json").read_text())
    summary_b = json.loads((tmp_path / "b" / "ladder_summary.json").read_text())
    assert summary_a == summary_b

    stamp_a = json.loads((tmp_path / "a" / "stamp.json").read_text())
    stamp_b = json.loads((tmp_path / "b" / "stamp.json").read_text())
    assert stamp_a["config_hash"] == stamp_b["config_hash"]
    assert stamp_a["data_snapshot"] == stamp_b["data_snapshot"]
    assert first.rung_table() == second.rung_table()


# --- D23: the untrained-z existence path, end to end ----------------------------------------
#
# Every other ladder test runs the `empirical` default, so the construction the 37-feature
# catalogue will actually use had no end-to-end cover: the bank loader, the dispatch, the SE
# threading and the power annotation were each unit-tested and never exercised together.


def _bank_file(tmp_path, features, *, k=25, mean=0.5, sd=0.01, seed=0):
    rng = np.random.default_rng(seed)
    rec = {
        "model_key": "test",
        "split_key": "test",
        "seeds": {
            str(s): {f: float(mean + sd * rng.standard_normal()) for f in features}
            for s in range(k)
        },
    }
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(rec))
    return path


def _z_config(bank_path) -> ProbingConfig:
    cfg = _config()
    return cfg.model_copy(
        update={
            "existence_method": "untrained_z",
            "untrained_bank_path": str(bank_path),
            "n_untrained_seeds": 25,
            "n_boot": 200,  # the fixture is tiny; 2000 resamples buys nothing here
        }
    )


@pytest.mark.integration
def test_the_untrained_z_path_runs_and_separates_signal_from_absence(tmp_path):
    labels = _labels()
    cfg = _z_config(_bank_file(tmp_path, labels.features))
    controls, ids = _controls_and_ids()
    result = run_ladder(controls, labels, ids[:120], ids[120:], config=cfg, sky_label_col="snr")
    assert {v.method for v in result.existence.values()} == {"untrained_z"}
    # a p-value the empirical estimator could not have produced at this draw count
    assert all(0.0 <= e.pvalue <= 1.0 for e in result.existence.values())
    # power travels with every rung, passing or failing
    for v in result.verdicts.values():
        assert v.n_test > 0 and v.resolvable_margin > 0.0


@pytest.mark.integration
def test_a_missing_bank_raises_rather_than_falling_back_to_the_point_mass(tmp_path):
    """Silently reverting would produce a catalogue computed by a test the config forbade."""
    cfg = _z_config(tmp_path / "does_not_exist.json")
    controls, ids = _controls_and_ids()
    with pytest.raises(FileNotFoundError):
        run_ladder(controls, _labels(), ids[:120], ids[120:], config=cfg, sky_label_col="snr")


@pytest.mark.integration
def test_an_uneven_bank_is_refused(tmp_path):
    """Different-sized denominators across features is not a catalogue, it is a bug."""
    labels = _labels()
    path = _bank_file(tmp_path, labels.features)
    rec = json.loads(path.read_text())
    rec["seeds"]["0"].pop(labels.features[0])  # one feature short on one seed
    path.write_text(json.dumps(rec))
    controls, ids = _controls_and_ids()
    with pytest.raises(ValueError, match="covers"):
        run_ladder(
            controls,
            _labels(),
            ids[:120],
            ids[120:],
            config=_z_config(path),
            sky_label_col="snr",
        )


# D24: matched survival is O1's retention rule, over three untrained draws, and only where the
# unmatched margin is established — never "matched AUC ≥ the effect floor".


@pytest.mark.integration
def test_matched_survival_refuses_fewer_than_three_untrained_draws(tmp_path):
    """One untrained draw decided a verdict under D23; a silently single-draw bar is refused."""
    labels = _labels()
    controls, ids = _controls_and_ids()
    single = ctl.ControlEmbeddings(
        real=controls.real, untrained=controls.untrained, noise=controls.noise
    )
    with pytest.raises(ValueError, match="D24"):
        run_ladder(
            single,
            labels,
            ids[:120],
            ids[120:],
            config=_z_config(_bank_file(tmp_path, labels.features)),
            sky_label_col="snr",
        )


@pytest.mark.integration
def test_retention_is_judged_only_where_the_margin_is_established(tmp_path):
    """A margin that fails existence cannot have its retention judged: noise over noise."""
    labels = _labels()
    controls, ids = _controls_and_ids()
    result = run_ladder(
        controls,
        labels,
        ids[:120],
        ids[120:],
        config=_z_config(_bank_file(tmp_path, labels.features)),
        sky_label_col="snr",
    )
    judged = [(f, v.matched) for f, v in result.verdicts.items() if v.matched is not None]
    assert judged
    # failing features carry their clearance too — the record must be able to show UNRESOLVED
    failing = [f for f, e in result.existence.items() if not e.exceeds_null]
    assert all(result.verdicts[f].matched is not None for f in failing)
    for f, m in judged:
        assert m.retention is not None and m.k_bar == 3
        assert m.margin_established == result.existence[f].exceeds_null
        if not m.margin_established:
            assert m.retention.verdict == "UNRESOLVED"
        assert m.survived == (m.retention.verdict == "SURVIVES")
    assert not any("did not survive matching" in v.mechanism for v in result.verdicts.values())


# D25: the effect floor is an absolute clean-vs-marginal threshold, and nothing relative reads it.
# Five consumers were audited; these pin the two that were relative (2A's conditional leg and the
# MLP decode) so a sixth cannot quietly return.


def _stub_geometry(a: str, b: str):
    from types import SimpleNamespace

    return SimpleNamespace(
        names=[a, b],
        cosine=np.array([[1.0, 0.9], [0.9, 1.0]]),
        mp=SimpleNamespace(significant=True),
        cav_disagreement={a: 0.5, b: 0.5},
        entangled_pairs=[(a, b)],
    )


def _pair(cfg: ProbingConfig, *, a_auc: float, established: bool = True):
    from galaxy_jepa.probing import ladder as ladder_mod

    labels = _labels()
    controls, ids = _controls_and_ids()
    a, b = labels.features
    bars = {
        f: ladder_mod._untrained_bar(controls, labels, f, ids[:120], ids[120:], config=cfg)
        for f in (a, b)
    }
    return ladder_mod._entangled_map(
        _stub_geometry(a, b),
        {},
        controls,
        labels,
        ids[:120],
        ids[120:],
        real_aucs={a: a_auc, b: 0.9},
        bars=bars,
        established={a: established, b: True},
        config=cfg,
    )


@pytest.mark.integration
def test_the_conditional_leg_judges_retention_and_never_reads_the_effect_floor():
    """Brief P called nine pairs world-correlation because A sat below 0.7267 before matching."""
    low, high = (_config().model_copy(update={"effect_floor": f}) for f in (0.51, 0.99))
    _, (pv_low,) = _pair(low, a_auc=0.6)
    _, (pv_high,) = _pair(high, a_auc=0.6)
    assert pv_low.retention in {"SURVIVES", "PARTIAL", "COLLAPSES", "UNRESOLVED"}
    assert (pv_low.retention, pv_low.survived_matching, pv_low.verdict) == (
        pv_high.retention,
        pv_high.survived_matching,
        pv_high.verdict,
    )
    expected = {"SURVIVES": True, "COLLAPSES": False}.get(pv_low.retention)
    assert pv_low.survived_matching is expected, "PARTIAL/UNRESOLVED must not attribute the pair"


@pytest.mark.integration
def test_a_pair_whose_margin_is_not_established_is_refused():
    """Only existence-passing directions reach 2A; a pair without one is a wiring error."""
    with pytest.raises(ValueError, match="D25"):
        _pair(_config(), a_auc=0.6, established=False)


@pytest.mark.integration
def test_moving_the_effect_floor_moves_nothing_but_clean(tmp_path):
    """Every matched verdict and every failing rung is invariant to the floor; only R1/R2 moves."""
    labels = _labels()
    controls, ids = _controls_and_ids()
    bank = _bank_file(tmp_path, labels.features)
    runs = [
        run_ladder(
            controls,
            labels,
            ids[:120],
            ids[120:],
            config=_z_config(bank).model_copy(update={"effect_floor": f}),
            sky_label_col="snr",
        )
        for f in (0.51, 0.99)
    ]
    for f in labels.features:
        lo, hi = (r.verdicts[f] for r in runs)
        assert (lo.matched is None) == (hi.matched is None)
        if lo.matched is not None:
            assert lo.matched.retention == hi.matched.retention
        if lo.rung in {"R3", "R4"} or hi.rung in {"R3", "R4"}:
            assert lo.rung == hi.rung and lo.mechanism == hi.mechanism
    assert [(p.a, p.b, p.retention) for p in runs[0].pair_verdicts] == [
        (p.a, p.b, p.retention) for p in runs[1].pair_verdicts
    ]


@pytest.mark.integration
def test_the_mlp_rung_is_not_assigned_without_an_untrained_mlp_bar(tmp_path):
    """No R3 on a stand-in bar: the failing verdict says the MLP decode is unadjudicated."""
    labels = _labels()
    controls, ids = _controls_and_ids()
    result = run_ladder(
        controls,
        labels,
        ids[:120],
        ids[120:],
        config=_z_config(_bank_file(tmp_path, labels.features)),
        sky_label_col="snr",
    )
    failing = [v for f, v in result.verdicts.items() if not result.existence[f].exceeds_null]
    assert failing
    for v in failing:
        assert v.rung == "R4" and "unadjudicated" in v.mechanism and v.sweep
