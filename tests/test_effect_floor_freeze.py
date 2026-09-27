"""The effect floor's write-once, hashed record — the gate scoring cannot get past.

The shipped floor (0.7267, D22) used to live inline in ``configs/probe.yaml``, where a text
editor could move it with nothing on disk saying so; it now lives in ``configs/effect_floor.json``.
These pin the mechanism: a floor is *computed* by a named rule over named inputs, written exactly
once, refused if edited, and the only floor scoring will use. ``existence_verdicts`` takes the
record itself, not a number. Only a smoke may score without one, and everything it writes is
marked FLOOR BYPASSED and refused by every verdict reader.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from galaxy_jepa.probing import controls as ctl
from galaxy_jepa.probing import floor as floor_mod
from galaxy_jepa.probing import ladder as ladder_mod
from galaxy_jepa.probing import run as run_mod
from galaxy_jepa.probing.config import EffectFloorFreeze, ProbingConfig
from galaxy_jepa.probing.controls import FeatureControls
from galaxy_jepa.probing.floor import (
    FLOOR_BYPASSED,
    EffectFloorRecord,
    FloorBypass,
    effect_floor_gate,
    freeze_effect_floor,
    load_effect_floor,
    resolve_effect_floor,
)
from galaxy_jepa.probing.gates import build_gates
from galaxy_jepa.probing.nulls import existence_verdicts

pytestmark = pytest.mark.invariant

#: The six pre-declared spread features and their real AUCs, as N2 reported them
#: (artifacts/out/n2_floor_evidence.txt, "REAL AUC SPREAD"; M's encoder, runs/m/encoder.pt).
N2_SPREAD = {
    "t10_arms_winding_a29_medium": 0.5226,
    "t09_bulge_shape_a26_boxy": 0.5847,
    "t10_arms_winding_a28_tight": 0.5889,
    "t10_arms_winding_a30_loose": 0.6538,
    "t02_edgeon_a04_yes": 0.7995,
    "t01_smooth_or_features_a02_features_or_disk": 0.8845,
}
#: What configs/probe.yaml froze from them by hand (Brief O0).
N2_FLOOR = 0.7267


def _freeze(path, **overrides):
    kwargs = dict(
        rule="widest_gap_midpoint",
        inputs=N2_SPREAD,
        frozen_by="tests",
        derived_from="fixture: the N2 spread",
        rationale="fixture",
        frozen_at="2026-09-27",
    )
    kwargs.update(overrides)
    return freeze_effect_floor(path, **kwargs)


def _hand_edit(path, mutate) -> None:
    """Edit the record on disk the way a person would — which takes undoing its read-only bit."""
    os.chmod(path, 0o644)
    raw = json.loads(path.read_text())
    mutate(raw)
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")


def _config(**overrides) -> ProbingConfig:
    return ProbingConfig(vote_count_min=1, **overrides)


class TestTheRule:
    def test_the_n2_inputs_give_the_floor_that_was_frozen_by_hand(self):
        """The registry must reproduce the recorded call, or it is a different rule."""
        assert floor_mod.RULES["widest_gap_midpoint"](N2_SPREAD) == N2_FLOOR

    def test_half_up_is_the_decision_because_half_even_would_move_the_call(self):
        """N2's midpoint is the exact tie 0.72665; only half-up gives the 0.7267 D22 recorded."""
        from decimal import ROUND_HALF_EVEN, Decimal

        tie = (Decimal("0.6538") + Decimal("0.7995")) / 2
        assert tie == Decimal("0.72665")
        assert float(tie.quantize(Decimal("0.0001"), rounding=ROUND_HALF_EVEN)) == 0.7266
        assert floor_mod.RULES["widest_gap_midpoint"](N2_SPREAD) == 0.7267

    def test_the_midpoint_is_taken_in_decimal_not_binary(self):
        # N2's own midpoint, 0.72665, is an exact half: the rule states 0.7267 by its rounding,
        # not by whichever side of the tie the float sum happens to land on. Here it lands low —
        # round((0.8155 + 0.9708) / 2, 4) is 0.8931 in binary — and the rule still says 0.8932.
        assert round((0.8155 + 0.9708) / 2, 4) == 0.8931
        assert floor_mod.RULES["widest_gap_midpoint"]({"a": 0.8155, "b": 0.9708}) == 0.8932

    def test_a_tie_for_the_widest_gap_is_refused_and_nothing_is_written(self, tmp_path):
        path = tmp_path / "floor.json"
        with pytest.raises(ValueError, match="tie"):
            _freeze(path, inputs={"a": 0.5, "b": 0.6, "c": 0.7})
        assert not path.exists()

    @pytest.mark.parametrize(
        ("rule", "inputs", "match"),
        [
            ("widest_gap_midpoint", {"a": 0.6}, "at least two"),
            ("widest_gap_midpoint", {"a": 0.6, "b": 1.2}, r"in \[0, 1\]"),
            ("widest_gap_midpoint", {"a": 0.6, "b": float("nan")}, r"in \[0, 1\]"),
            ("gut_feeling", N2_SPREAD, "unknown effect-floor rule"),
        ],
    )
    def test_an_undefined_rule_application_is_refused(self, tmp_path, rule, inputs, match):
        path = tmp_path / "floor.json"
        with pytest.raises(ValueError, match=match):
            _freeze(path, rule=rule, inputs=inputs)
        assert not path.exists()


class TestWriteOnce:
    def test_freezes_once_and_the_record_loads_intact(self, tmp_path):
        path = tmp_path / "floor.json"
        written = _freeze(path)
        assert written.value == N2_FLOOR and written.rule == "widest_gap_midpoint"
        loaded = load_effect_floor(path)
        assert loaded == written
        assert isinstance(loaded, EffectFloorFreeze)  # it is a freeze, so it can sit in the config
        assert not os.access(path, os.W_OK)  # read-only on disk as well

    @pytest.mark.parametrize(
        "second",
        [
            {},
            {"rationale": "a better reason"},
            {"inputs": {**N2_SPREAD, "t02_edgeon_a04_yes": 0.9}},
        ],
        ids=["identical", "different-prose", "different-inputs"],
    )
    def test_any_second_write_is_refused_and_the_first_survives(self, tmp_path, second):
        path = tmp_path / "floor.json"
        _freeze(path)
        before = path.read_bytes()
        with pytest.raises(FileExistsError, match="frozen once"):
            _freeze(path, **second)
        assert path.read_bytes() == before


class TestHandEdits:
    @pytest.mark.parametrize(
        "mutate",
        [
            lambda r: r.update(value=0.65),
            lambda r: r["inputs"].update(t02_edgeon_a04_yes=0.70),
            lambda r: r.update(rule="widest_gap_midpoint_v2"),
            lambda r: r.update(frozen_by="someone else"),
            lambda r: r.update(frozen_at="2026-09-01"),
        ],
        ids=["value", "input", "rule", "author", "date"],
    )
    def test_a_hand_edited_record_is_refused(self, tmp_path, mutate):
        path = tmp_path / "floor.json"
        _freeze(path)
        _hand_edit(path, mutate)
        with pytest.raises(ValueError, match="edited since it was frozen"):
            load_effect_floor(path)

    def test_a_rehashed_forgery_still_fails_its_own_rule(self, tmp_path):
        """Recomputing the hash over a moved value is not enough: the rule is re-run too."""
        path = tmp_path / "floor.json"
        forged = _freeze(path).model_copy(update={"value": 0.65})
        forged = forged.model_copy(update={"content_hash": forged.expected_hash()})
        _hand_edit(path, lambda r: r.update(forged.model_dump(mode="json")))
        with pytest.raises(ValueError, match="its rule 'widest_gap_midpoint' gives 0.7267"):
            load_effect_floor(path)


class TestScoringRefusal:
    def test_the_ladder_refuses_before_any_fit_when_the_file_is_absent(self, tmp_path):
        config = _config(effect_floor_file=str(tmp_path / "missing.json"))
        # controls/labels are None: the refusal must land before anything reads them
        with pytest.raises(ValueError, match="refusing to score"):
            ladder_mod.run_ladder(None, None, [], [], config=config, sky_label_col="snr")  # type: ignore[arg-type]

    def test_the_ladder_refuses_an_edited_file(self, tmp_path):
        path = tmp_path / "floor.json"
        _freeze(path)
        _hand_edit(path, lambda r: r.update(value=0.65))
        with pytest.raises(ValueError, match="refusing to score"):
            ladder_mod.run_ladder(
                None,  # type: ignore[arg-type]
                None,  # type: ignore[arg-type]
                [],
                [],
                config=_config(effect_floor_file=str(path)),
                sky_label_col="snr",
            )

    def test_the_ladder_scores_existence_at_the_files_value(self, tmp_path, monkeypatch):
        path = tmp_path / "floor.json"
        _freeze(path)
        seen: dict[str, float] = {}

        class _Stop(Exception):
            pass

        def _spy(*_args, floor, **_kw):
            seen["floor"] = floor
            raise _Stop

        monkeypatch.setattr(ladder_mod.nulls_mod, "existence_verdicts", _spy)
        config = _config(effect_floor_file=str(path))  # inline stays at the 0.65 default
        with pytest.raises(_Stop):
            ladder_mod.run_ladder(
                None,  # type: ignore[arg-type]
                SimpleNamespace(features=[]),  # type: ignore[arg-type]
                [],
                [],
                config=config,
                sky_label_col="snr",
            )
        # the record itself reaches existence, not a number: the file's, not the inline 0.65
        assert isinstance(seen["floor"], EffectFloorRecord) and seen["floor"].value == N2_FLOOR

    def test_the_ladder_refuses_a_non_smoke_with_no_file_at_all(self):
        """No file is not an open floor any more: only a smoke may score without the record."""
        with pytest.raises(ValueError, match="Only a smoke may score"):
            ladder_mod.run_ladder(None, None, [], [], config=_config(), sky_label_col="snr")  # type: ignore[arg-type]

    def test_an_inline_freeze_alone_is_not_the_gate(self):
        inline = EffectFloorFreeze(
            value=0.65, derived_from="r", frozen_at="2026-09-10", frozen_by="m", rationale="x"
        )
        with pytest.raises(ValueError, match="not the gate"):
            resolve_effect_floor(_config(effect_floor_freeze=inline))

    def test_an_inline_value_that_disagrees_with_the_record_is_refused(self, tmp_path):
        path = tmp_path / "floor.json"
        _freeze(path)
        with pytest.raises(ValueError, match="two that disagree"):
            resolve_effect_floor(_config(effect_floor=0.65, effect_floor_file=str(path)))
        # stating the record's own value is fine — it is what configs/probe.yaml does
        agreeing = _config(effect_floor=N2_FLOOR, effect_floor_file=str(path))
        assert resolve_effect_floor(agreeing).effect_floor == N2_FLOOR

    def test_the_gates_read_the_resolved_floor(self, tmp_path):
        path = tmp_path / "floor.json"
        _freeze(path)
        resolved = resolve_effect_floor(_config(effect_floor_file=str(path)))
        assert resolved.effect_floor == N2_FLOOR
        assert build_gates(resolved).clean_bar.evaluate({"auc": 0.7266}).passed is False
        assert build_gates(resolved).clean_bar.evaluate({"auc": N2_FLOOR}).passed is True
        assert resolve_effect_floor(resolved) == resolved  # idempotent


def _stub_probing(monkeypatch):
    """``run_probing`` down to its config handling and stamp — no extraction, no fits."""
    captured: dict[str, ProbingConfig] = {}
    monkeypatch.setattr(
        run_mod, "extract_matrix", lambda *a, **k: SimpleNamespace(object_ids=np.arange(40))
    )
    monkeypatch.setattr(ctl, "untrained_encoder_matrix", lambda *a, **k: None)
    monkeypatch.setattr(ctl, "noise_through_encoder_matrix", lambda *a, **k: None)
    monkeypatch.setattr(run_mod, "_write_summary", lambda *a, **k: None)

    def _ladder(*_args, config, **_kw):
        captured.setdefault("config", config)
        floor = effect_floor_gate(config)
        return SimpleNamespace(
            verdicts={}, floor=floor, floor_bypassed=not isinstance(floor, EffectFloorRecord)
        )

    monkeypatch.setattr(run_mod, "run_ladder", _ladder)
    return captured


def _probe(tmp_path, config):
    encoder = torch.nn.Linear(2, 2).requires_grad_(False).eval()
    labels = SimpleNamespace(features=["f"], population="full")
    out = tmp_path / "out"
    run_mod.run_probing(encoder, None, labels, {}, config=config, out_dir=out, emit_figures=False)  # type: ignore[arg-type]
    stamp = json.loads((out / "stamp.json").read_text())
    return stamp, json.loads((out / "config.json").read_text())


class TestRunProbingStamps:
    def test_a_resolved_record_is_hashed_and_stamped_and_nothing_is_forfeited(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "floor.json"
        record = _freeze(path)
        captured = _stub_probing(monkeypatch)
        stamp, written = _probe(tmp_path, _config(effect_floor_file=str(path)))
        assert captured["config"].effect_floor == N2_FLOOR
        assert stamp["escape_hatches_used"] == []  # RunStamp.create stores "none" as []
        assert written["effect_floor"] == N2_FLOOR
        assert written["effect_floor_freeze"]["content_hash"] == record.content_hash
        assert written["effect_floor_freeze"]["inputs"] == N2_SPREAD

    def test_a_smoke_bypasses_a_missing_file_and_is_stamped(self, tmp_path, monkeypatch):
        captured = _stub_probing(monkeypatch)
        config = _config(effect_floor_file=str(tmp_path / "missing.json"), smoke=True)
        with pytest.warns(UserWarning, match=FLOOR_BYPASSED):
            stamp, _ = _probe(tmp_path, config)
        assert captured["config"].effect_floor == 0.65  # the inline floor, openly
        assert stamp["escape_hatches_used"] == ["smoke", "effect_floor_open", FLOOR_BYPASSED]

    def test_a_smoke_with_no_file_is_stamped_bypassed(self, tmp_path, monkeypatch):
        _stub_probing(monkeypatch)
        stamp, _ = _probe(tmp_path, _config(smoke=True))
        assert FLOOR_BYPASSED in stamp["escape_hatches_used"]

    def test_a_smoke_with_an_intact_file_is_not_bypassed(self, tmp_path, monkeypatch):
        path = tmp_path / "floor.json"
        _freeze(path)
        _stub_probing(monkeypatch)
        stamp, _ = _probe(tmp_path, _config(effect_floor_file=str(path), smoke=True))
        assert stamp["escape_hatches_used"] == ["smoke"]

    def test_run_probing_refuses_before_extraction_without_a_bypass(self, tmp_path, monkeypatch):
        monkeypatch.setattr(run_mod, "extract_matrix", lambda *a, **k: pytest.fail("extracted"))
        with pytest.raises(ValueError, match="refusing to score"):
            _probe(tmp_path, _config(effect_floor_file=str(tmp_path / "missing.json")))


class TestConfigContracts:
    @pytest.mark.parametrize("smoke", [False, True])
    def test_effect_floor_open_can_no_longer_be_declared(self, smoke):
        """It used to let any run fall back past a missing or edited file; only smoke may now."""
        with pytest.raises(ValueError, match="no longer opens anything"):
            _config(
                effect_floor_file="floor.json", escape_hatches=("effect_floor_open",), smoke=smoke
            )

    def test_a_file_beside_an_inline_freeze_is_refused_at_load(self, tmp_path):
        inline = EffectFloorFreeze(
            value=0.65, derived_from="r", frozen_at="2026-09-10", frozen_by="m", rationale="x"
        )
        with pytest.raises(ValueError, match="two that could disagree"):
            _config(effect_floor_file="floor.json", effect_floor_freeze=inline)

    def test_a_headline_needs_the_file_not_an_inline_freeze(self):
        from galaxy_jepa.probing.config import VoteCountFreeze

        votes = VoteCountFreeze(
            value=1, derived_from="r", frozen_at="2026-09-10", frozen_by="m", rationale="x"
        )
        inline = EffectFloorFreeze(
            value=0.65, derived_from="r", frozen_at="2026-09-10", frozen_by="m", rationale="x"
        )
        with pytest.raises(ValueError, match="still OPEN"):
            _config(headline=True, vote_count_freeze=votes, effect_floor_freeze=inline)
        assert _config(headline=True, vote_count_freeze=votes, effect_floor_file="f.json").headline

    def test_the_path_is_not_identity_but_the_record_is(self, tmp_path):
        """D15: moving the file is the same run; a different record is a different one."""
        a, b = tmp_path / "a" / "floor.json", tmp_path / "b" / "floor.json"
        _freeze(a)
        _freeze(b)
        other = tmp_path / "c.json"
        _freeze(other, rationale="another call")
        dump = lambda p: resolve_effect_floor(_config(effect_floor_file=str(p))).determining_dump()  # noqa: E731
        assert "effect_floor_file" not in dump(a)
        assert dump(a) == dump(b)
        assert dump(a) != dump(other)


class TestUnsetFile:
    def test_an_unset_file_hands_a_smoke_back_untouched_and_refuses_anything_else(self):
        smoke = _config(effect_floor=0.61, smoke=True)
        assert resolve_effect_floor(smoke) is smoke
        bypass = effect_floor_gate(smoke)
        assert isinstance(bypass, FloorBypass) and bypass.value == 0.61
        with pytest.raises(ValueError, match="Only a smoke may score"):
            resolve_effect_floor(_config(effect_floor=0.61))

    def test_an_unset_file_does_not_enter_the_hash(self):
        # Every config that predates the field must hash exactly as it did before it.
        assert "effect_floor_file" not in _config().determining_dump()

    def test_an_inline_freeze_dumps_as_it_always_did(self):
        inline = EffectFloorFreeze(
            value=0.65, derived_from="r", frozen_at="2026-09-10", frozen_by="m", rationale="x"
        )
        dumped = _config(effect_floor_freeze=inline).determining_dump()["effect_floor_freeze"]
        assert set(dumped) == {"value", "derived_from", "frozen_at", "frozen_by", "rationale"}

    def test_the_record_type_survives_a_config_dump(self, tmp_path):
        record = _freeze(tmp_path / "floor.json")
        assert isinstance(record, EffectFloorRecord)
        resolved = resolve_effect_floor(_config(effect_floor_file=str(tmp_path / "floor.json")))
        assert set(resolved.model_dump(mode="json")["effect_floor_freeze"]) >= {
            "rule",
            "inputs",
            "content_hash",
        }


def _fc(real_auc: float) -> FeatureControls:
    """A feature that clears its (constant) null at any family of one."""
    return FeatureControls(
        feature="f",
        real_auc=real_auc,
        shuffled_nulls=np.full(400, 0.50),
        random_embedding_nulls=np.full(400, 0.50),
        noise_encoder_auc=0.50,
        untrained_encoder_auc=0.50,
        sky_noise_auc=0.50,
        selectivity=0.3,
        nuisance_aucs={},
    )


class TestExistenceTakesTheRecord:
    """``existence_verdicts`` is the floor's last reader; it cannot be handed a bare number."""

    def test_the_record_decides_clean_and_nothing_is_marked(self, tmp_path):
        record = _freeze(tmp_path / "floor.json")
        (below,) = existence_verdicts({"f": _fc(0.7266)}, floor=record, n_tests=1).values()
        (at,) = existence_verdicts({"f": _fc(N2_FLOOR)}, floor=record, n_tests=1).values()
        assert below.exceeds_null and not below.clean
        assert at.clean and not at.floor_bypassed

    @pytest.mark.parametrize(
        "floor",
        [
            0.65,
            EffectFloorFreeze(
                value=0.65, derived_from="r", frozen_at="2026-09-10", frozen_by="m", rationale="x"
            ),
        ],
        ids=["bare-float", "unhashed-inline-freeze"],
    )
    def test_anything_but_the_record_or_a_bypass_is_refused(self, floor):
        with pytest.raises(TypeError, match="a bare number is not a floor"):
            existence_verdicts({"f": _fc(0.9)}, floor=floor, n_tests=1)

    def test_the_floor_is_required(self):
        with pytest.raises(TypeError, match="floor"):
            existence_verdicts({"f": _fc(0.9)}, n_tests=1)  # type: ignore[call-arg]

    def test_an_in_memory_edit_of_the_record_is_refused(self, tmp_path):
        moved = _freeze(tmp_path / "floor.json").model_copy(update={"value": 0.5})
        with pytest.raises(ValueError, match="edited since it was frozen"):
            existence_verdicts({"f": _fc(0.9)}, floor=moved, n_tests=1)

    def test_a_bypass_marks_every_verdict(self):
        bypass = FloorBypass(value=0.65, reason="fixture")
        (v,) = existence_verdicts({"f": _fc(0.9)}, floor=bypass, n_tests=1).values()
        assert v.clean and v.floor_bypassed


class TestBypassedVerdictsAreNotReported:
    def test_a_bypassed_ladder_result_refuses_every_read_out(self):
        bypassed = ladder_mod.LadderResult(
            verdicts={},
            existence={},
            entanglement=None,
            feature_controls={},
            directions={},
            floor=FloorBypass(value=0.65, reason="fixture"),
        )
        report = run_mod.ProbingReport(
            ladder=bypassed, uncertainty={}, figures={}, out_dir="x", data_snapshot="x"
        )
        with pytest.raises(ValueError, match=FLOOR_BYPASSED):
            report.rung_table()
        with pytest.raises(ValueError, match=FLOOR_BYPASSED):
            report.population_comparison()

    def test_a_result_that_names_no_floor_is_refused_too(self):
        unknown = ladder_mod.LadderResult(
            verdicts={}, existence={}, entanglement=None, feature_controls={}, directions={}
        )
        with pytest.raises(ValueError, match="no floor record attached"):
            unknown.assert_reportable("the rung table")

    def test_a_frozen_result_reports(self, tmp_path):
        frozen = ladder_mod.LadderResult(
            verdicts={},
            existence={},
            entanglement=None,
            feature_controls={},
            directions={},
            floor=_freeze(tmp_path / "floor.json"),
        )
        report = run_mod.ProbingReport(
            ladder=frozen, uncertainty={}, figures={}, out_dir="x", data_snapshot="x"
        )
        assert report.rung_table() == {} and report.population_comparison() == {}

    def test_the_summary_carries_the_marker_beside_the_verdicts(self, tmp_path):
        bypassed = ladder_mod.LadderResult(
            verdicts={},
            existence={
                "f": existence_verdicts(
                    {"f": _fc(0.9)}, floor=FloorBypass(value=0.65, reason="fixture"), n_tests=1
                )["f"]
            },
            entanglement=None,
            feature_controls={},
            directions={},
            floor=FloorBypass(value=0.65, reason="fixture"),
        )
        path = run_mod._write_summary(bypassed, {}, tmp_path, floor_value=0.65)
        summary = json.loads(path.read_text())
        assert summary["effect_floor"]["status"] == FLOOR_BYPASSED
        assert summary["existence"]["f"]["floor"] == FLOOR_BYPASSED


class TestTheShippedRecord:
    """The migration: probe.yaml's inline 0.7267 now lives in a write-once record it points at."""

    def test_it_loads_intact_at_the_d22_value_from_the_n2_inputs(self):
        record = load_effect_floor("configs/effect_floor.json")
        assert record.value == N2_FLOOR and record.rule == "widest_gap_midpoint"
        assert record.inputs == N2_SPREAD
        assert "D22" in record.derived_from and "n2_floor_evidence.txt" in record.derived_from
        assert record.frozen_by == "malachy"

    def test_probe_yaml_scores_against_it(self):
        yaml = pytest.importorskip("yaml")
        with open("configs/probe.yaml") as fh:
            config = ProbingConfig(**yaml.safe_load(fh))
        assert config.effect_floor_freeze is None  # the inline block is gone
        resolved = resolve_effect_floor(config)
        assert resolved.effect_floor == N2_FLOOR
        assert resolved.effect_floor_freeze == load_effect_floor("configs/effect_floor.json")
