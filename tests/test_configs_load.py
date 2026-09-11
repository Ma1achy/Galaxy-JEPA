"""The shipped configs must actually load.

`configs/*.yaml` spent a long time claiming "PLACEHOLDER — no training code yet" while the
training code existed. A doc that lies about the repo is the failure this pins: because
``RunConfig`` sets ``extra='forbid'``, a config naming a field that no longer exists is a
load-time error — so this test turns config drift into a red build rather than a surprise at
the start of a long run.
"""

from __future__ import annotations

import pytest

yaml = pytest.importorskip("yaml")

from galaxy_jepa.harness import HarnessConfig  # noqa: E402
from galaxy_jepa.probing.config import ProbingConfig, VoteCountFreeze  # noqa: E402

pytestmark = pytest.mark.invariant


def test_pretrain_config_loads():
    with open("configs/pretrain.yaml") as fh:
        config = HarnessConfig(**yaml.safe_load(fh))
    assert config.objective.beta == 0.5  # the headline; the sweep is downstream
    assert config.probing.scheme == "full_tree"  # D14: full tree runs first


def test_the_shipped_normalisation_freeze_is_intact_and_not_rounded():
    """The config's constants must be the ones that were fitted, to the last digit.

    This pins a trap that actually sprung: the emitter printed the block rounded to 10 places
    while ``content_hash`` covered full precision, so the pasted record was — correctly — refused
    as a hand-edit. A prettier config is a different config.
    """
    with open("configs/pretrain.yaml") as fh:
        config = HarnessConfig(**yaml.safe_load(fh))
    freeze = config.normalisation
    assert freeze is not None, "a shipped pretraining config must carry a frozen statistic"
    freeze.assert_intact()
    assert freeze.corpus == "pretrain"  # fitted on pretrain, applied to both (parity)
    assert freeze.valid_pixels_only
    assert freeze.stretch_q == config.q  # or the statistic does not belong to this stretch
    assert freeze.trim_excluded > 0 and freeze.trim_ids_sha256  # the trim is stated, not implied

    rounded = freeze.model_copy(update={"mean": tuple(round(v, 10) for v in freeze.mean)})
    with pytest.raises(ValueError, match="edited since it was fitted"):
        rounded.to_normalise()


def test_probe_config_loads_and_carries_the_grounded_decisions():
    with open("configs/probe.yaml") as fh:
        config = ProbingConfig(**yaml.safe_load(fh))
    assert config.multiplicity == "benjamini_yekutieli"
    assert config.n_perm >= 10_000
    assert config.permutation_method == "two_sided"
    assert config.mp_method == "upper_edge"
    assert config.compare_populations is True
    assert config.smoke is False  # a shipped config must never default to a smoke


def test_an_unknown_key_is_a_load_time_error():
    with pytest.raises(ValueError):
        ProbingConfig(seed=0, vote_count_min=21, multiplicty="benjamini_yekutieli")  # typo'd key


# D8: the floor is required, so every construction below states it. A frozen record is needed
# wherever headline=True, or the vote-floor gate fires before the one under test.
VOTE_FREEZE = VoteCountFreeze(
    value=21,
    derived_from="tests",
    frozen_at="2026-09-11",
    frozen_by="tests",
    rationale="fixture: pins the vote floor so the effect-floor gate is the one being tested",
)


class TestEffectFloorFreeze:
    """The floor has one honest window — the medium run — and the gate is mechanical."""

    def test_headline_is_refused_while_the_floor_is_open(self):
        with pytest.raises(ValueError, match="still OPEN"):
            ProbingConfig(headline=True, vote_count_min=21, vote_count_freeze=VOTE_FREEZE)

    def test_a_frozen_floor_unlocks_the_headline_and_carries_its_provenance(self):
        from galaxy_jepa.probing.config import EffectFloorFreeze

        freeze = EffectFloorFreeze(
            value=0.65,
            derived_from="runs/medium",
            frozen_at="2026-09-10",
            frozen_by="malachy",
            rationale="set from the medium run's per-feature AUC distribution",
        )
        config = ProbingConfig(
            headline=True,
            effect_floor=0.65,
            effect_floor_freeze=freeze,
            vote_count_min=21,
            vote_count_freeze=VOTE_FREEZE,
        )
        assert config.effect_floor_freeze.derived_from == "runs/medium"
        # the record is part of the config, so it is hashed and written with every artefact
        assert "effect_floor_freeze" in config.model_dump(mode="json")

    def test_a_floor_that_contradicts_its_record_is_refused(self):
        from galaxy_jepa.probing.config import EffectFloorFreeze

        freeze = EffectFloorFreeze(
            value=0.65, derived_from="r", frozen_at="2026-09-10", frozen_by="m", rationale="x"
        )
        with pytest.raises(ValueError, match="contradicts its freeze record"):
            ProbingConfig(effect_floor=0.70, effect_floor_freeze=freeze, vote_count_min=21)

    def test_a_run_cannot_be_both_smoke_and_headline(self):
        from galaxy_jepa.probing.config import EffectFloorFreeze

        freeze = EffectFloorFreeze(
            value=0.65, derived_from="r", frozen_at="2026-09-10", frozen_by="m", rationale="x"
        )
        with pytest.raises(ValueError, match="both a smoke and the headline"):
            ProbingConfig(
                headline=True,
                smoke=True,
                effect_floor_freeze=freeze,
                vote_count_min=21,
                vote_count_freeze=VOTE_FREEZE,
            )

    def test_an_open_floor_is_recorded_in_the_ledger(self):
        """An unfrozen floor is a forfeited guarantee, so the artefact must say so."""
        assert (
            ProbingConfig(vote_count_min=21).effect_floor_freeze is None
        )  # the shipped default is OPEN
