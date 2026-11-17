"""`artifacts/m_comparison.py` on synthetic inputs only (user, 17 Nov): no real ladder record or
bank is opened. The refusal to load a real record before the hash is tested by pointing the loader
at the real paths with the pre-registration unhashed: it must refuse before reading a byte.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "artifacts"))
import m_comparison as mc  # noqa: E402

from galaxy_jepa.probing import nulls  # noqa: E402

RNG_SEED = 20261117


def _world(rng, sigma=0.15):
    return mc.base_world(rng, sigma)


def _score(rng, z, **kw):
    recs, banks = mc.build(rng, z, **kw)
    return mc.score(recs, banks, k=1.1)


def test_unhashed_refuses_real_records(monkeypatch):
    monkeypatch.setattr(mc, "PREREG_SHA1", None)
    opened = []
    real_sha1 = mc._sha1
    monkeypatch.setattr(mc, "_sha1", lambda p: opened.append(p) or real_sha1(p))
    with pytest.raises(SystemExit, match="not hashed"):
        mc.load_inputs(mc.RECORDS, mc.BANKS, mc.PINS)
    assert opened == []


def test_hash_mismatch_refuses(monkeypatch):
    monkeypatch.setattr(mc, "PREREG_SHA1", "0" * 40)
    with pytest.raises(SystemExit, match="changed since the hash"):
        mc.assert_hashed()


def test_real_roots_cover_every_real_input():
    assert all(mc.is_real(p) for p in (*mc.RECORDS.values(), *mc.BANKS.values()))
    assert all(mc.is_real(REPO / "runs" / d / "encoder.pt") for d in ("m", "o2"))
    assert mc.is_real(REPO / "artifacts" / "out" / "t1_ladder.json")
    assert not mc.is_real(REPO / "docs" / "m_comparison" / "plants.json")


def test_score_real_refuses_unhashed(monkeypatch):
    monkeypatch.setattr(mc, "PREREG_SHA1", None)
    with pytest.raises(SystemExit, match="not hashed"):
        mc.score_real(go=True)


def test_prereg_section_bounds():
    sec = mc.prereg_section()
    assert sec.startswith("## Pre-registration\n") and sec.endswith("\n")
    assert "\n## " not in sec[3:]


def test_by_reject_matches_package():
    rng = np.random.default_rng(RNG_SEED)
    for _ in range(200):
        p = rng.uniform(0, 1, mc.N_FEATURES) ** rng.uniform(1, 12)
        p[rng.integers(0, mc.N_FEATURES, 3)] = p[0]  # ties
        pkg = nulls.family_significant(
            {f"f{j}": float(v) for j, v in enumerate(p)},
            alpha=0.05,
            method="benjamini_yekutieli",
            n_tests=mc.N_FEATURES,
        )
        assert list(mc.by_reject(p)) == [pkg[f"f{j}"] for j in range(mc.N_FEATURES)]


def test_z_matches_package_pvalue():
    rng = np.random.default_rng(RNG_SEED)
    bank = rng.normal(0.75, 0.005, mc.K_BANK)
    for auc, se in ((0.80, 0.004), (0.751, 0.01), (0.70, 0.002)):
        z = (auc - bank.mean()) / np.sqrt(bank.std(ddof=1) ** 2 + se**2)
        from scipy.stats import t

        assert abs(t.sf(z, mc.K_BANK - 1) - nulls.untrained_z_pvalue(auc, se, bank)) < 1e-12


def test_identical_world_agrees():
    rng = np.random.default_rng(RNG_SEED)
    w = _world(rng, 0.0)
    w = {r: {p: w["M"][p].copy() for p in mc.POPS} for r in mc.RUNS}
    res = _score(rng, w)
    for pop in mc.POPS:
        assert {c["state"] for c in res["populations"][pop]["cells"]} == {"AGREES"}
        assert res["populations"][pop]["aggregate"]["state"] == "INSUFFICIENT"


@pytest.mark.parametrize(
    "runs,cells,val,want",
    [
        (("A1", "A2", "A3"), range(8), 0.0, ("A-LOWER", -1)),
        (("A1", "A2"), range(5), 0.0, ("A-SPLIT", -1)),
        (("A1",), range(5), 0.0, ("ONE-PAIR", -1)),
        (("M", "A2"), range(25, 30), 10.0, ("PAIRS-OPPOSE", 0)),
        (("M", "A1"), range(25, 30), 10.0, ("SEED-SPLIT", 0)),
        (("A1", "A2", "A3"), range(25, 33), 10.0, ("A-HIGHER", 1)),
    ],
)
def test_cell_states(runs, cells, val, want):
    rng = np.random.default_rng(RNG_SEED)
    w = _world(rng)
    for r in runs:
        for c in cells:
            w[r]["full"][c] = val
    res = _score(rng, w)
    got = res["populations"]["full"]["cells"]
    assert all((got[c]["state"], got[c]["dir"]) == want for c in cells)


def test_pairing_removes_a_shared_seed_offset():
    """A per-seed offset common to both members of a pair moves no Δ, so no state moves."""
    rng = np.random.default_rng(RNG_SEED)
    w = _world(rng)
    for c in range(mc.N_FEATURES):
        for r in ("M", "A1"):
            w[r]["full"][c] += 0.4
    res0 = _score(np.random.default_rng(1), _world(np.random.default_rng(RNG_SEED)))
    res1 = _score(np.random.default_rng(1), w)
    s0 = [c["state"] for c in res0["populations"]["full"]["cells"]]
    s1 = [c["state"] for c in res1["populations"]["full"]["cells"]]
    assert s0 == s1


def test_aggregate_binomial_and_reach():
    assert mc.reach(2) is None and mc.reach(3) == 3
    rng = np.random.default_rng(RNG_SEED)
    w = _world(rng)
    for r in ("A1", "A2", "A3"):
        for c in range(8):
            w[r]["full"][c] = 0.0
    agg = _score(rng, w)["populations"]["full"]["aggregate"]
    assert agg["state"] == "CARRIED" and agg["n"] == 8 and abs(agg["p_down"] - 0.25**8) < 1e-15


def test_thin_precedes_direction():
    rng = np.random.default_rng(RNG_SEED)
    w = _world(rng)
    se = {r: mc._se() for r in mc.RUNS}
    for r in ("A1", "A2", "A3"):
        for c in range(3):
            w[r]["full"][c] = 0.0
    for c in range(3):
        se["O2"]["full"][c] = 0.03
        w["O2"]["full"][c] = 1.0  # O2 underpowered: THIN wins whatever the pattern
    res = _score(rng, w, se=se)
    assert all(res["populations"]["full"]["cells"][c]["state"] == "THIN" for c in range(3))


@pytest.mark.parametrize(
    "tamper,level",
    [
        (lambda r, b: r["A3"].update(k_bar=30), "all"),  # the bank's K is not the record's k_bar
        (lambda r, b: b["v1"].update(split_key="x"), "all"),
        (lambda r, b: r["A1"]["full"][4].update(rung="R3"), 4),
        (lambda r, b: r["M"]["full"][2].update(auc=None), 2),
    ],
)
def test_void_levels(tamper, level):
    rng = np.random.default_rng(RNG_SEED)
    recs, banks = mc.build(rng, _world(rng), tamper=tamper)
    res = mc.score(recs, banks, k=1.1)
    if level == "all":
        assert res["void"] and res["populations"]["full"]["cells"] == "all VOID"
    else:
        cells = res["populations"]["full"]["cells"]
        assert cells[level]["state"] == "VOID" and sum(c["state"] == "VOID" for c in cells) == 1


def test_p1_pin_mismatch_voids_everything(tmp_path):
    rng = np.random.default_rng(RNG_SEED)
    recs, banks = mc.build(rng, _world(rng))
    res = mc.run_files(recs, banks, pins_override={"bank_v1": "f" * 40}, k=1.1)
    assert res["void"] == ["P1: bank_v1"]


def test_untrained_bars_reported_without_state():
    rng = np.random.default_rng(RNG_SEED)
    res = _score(rng, _world(rng))
    c = res["populations"]["conditional"]["cells"][0]
    assert set(c["bar_d23"]) == {"v1", "v2"} and c["bar_d23"]["v1"]["K"] == mc.K_BANK
    assert set(c["bar_d24_C"]) == set(mc.RUNS)


def test_calibration_k_is_the_constant():
    cal = json.loads(mc.CALIBRATION.read_text())
    assert cal["k"] == mc.K_GUARD


def test_original_m_refused_unhashed(monkeypatch):
    monkeypatch.setattr(mc, "PREREG_SHA1", None)
    monkeypatch.setattr(mc, "_sha1", lambda p: pytest.fail("read before the hash"))
    with pytest.raises(SystemExit, match="not hashed"):
        mc.load_pinned(mc.ORIGINAL_M, mc.ORIGINAL_M_SHA1)


def test_p4_on_the_committed_manifests():
    """The two manifests (method files, no values) differ only in the two declared pairs."""
    assert mc.p4_status() == []


def test_p4_compare_refuses_an_undeclared_difference():
    a = {
        "artifacts/t1_ladder.py": {"x"},
        "src/galaxy_jepa/**.py": {mc.P4_DECLARED["src/galaxy_jepa/**.py"][0]},
    }
    v = {
        "artifacts/t1_ladder.py": {"x"},
        "src/galaxy_jepa/**.py": {mc.P4_DECLARED["src/galaxy_jepa/**.py"][1]},
    }
    assert mc.p4_compare(a, v) == []
    assert mc.p4_compare(a, {**v, "artifacts/t1_ladder.py": {"y"}})
    assert mc.p4_compare(a, {**v, "src/galaxy_jepa/**.py": {"z"}})
    assert mc.p4_compare({**a, "configs/probe.yaml": {"p"}}, v)


def test_outputs_pins_by_path(tmp_path):
    f = tmp_path / "outputs.sha1"
    m = mc.RECORDS["M"]
    f.write_text(f"{'a' * 40}  {m}\n{'b' * 40}  {mc.BANKS['v1']}\n")
    got = mc.outputs_pins(f, {"M": m, "bank_v1": mc.BANKS["v1"], "O2": mc.RECORDS["O2"]})
    assert got == {"M": "a" * 40, "bank_v1": "b" * 40, "O2": None}


def test_reproduction_pass_fail():
    rec = {p: [{"feature": f"f{j}", "rung": "R1"} for j in range(3)] for p in mc.POPS}
    other = json.loads(json.dumps(rec))
    assert mc.reproduction(rec, other) == {"result": "PASS"}
    other["full"][1]["rung"] = "R4"
    other["conditional"][2]["rung"] = "R2"
    assert mc.reproduction(rec, other) == {"result": "FAIL", "cells_rung_differs": 2}
    assert mc.reproduction(None, other)["result"] == "NOT RUN"


def test_drift_mean_and_se():
    d1 = np.array([1.0, 2.0, 3.0, 9.0])
    d2 = np.zeros(4)
    ok = np.array([True, True, True, False])
    r = mc.drift(d1, d2, ok, np.asarray(2.0))
    assert abs(r["mean"] - 2.0) < 1e-12 and abs(r["se"] - 1 / np.sqrt(3)) < 1e-12
    assert abs(r["scaled"] - 2.0) < 1e-12


GRID = [round(0.1 * i, 1) for i in range(16)]


def _curve(c, m, b=0.0):
    return [
        {"omega": w, "CARRIED": c(i), "MASKED": m(i), "BOTH WAYS": b} for i, w in enumerate(GRID)
    ]


def test_tipping_interpolates_between_bracketing_rows():
    cv = _curve(lambda i: 0.01 * i, lambda i: 0.004 + 0.003 * i)
    assert abs(mc.tipping(cv, "CARRIED", 0.025)["omega"] - 0.25) < 1e-12
    assert mc.tipping(cv, "CARRIED", 0.025)["bracket"] == [0.2, 0.3]
    # MASKED: 0.004 + 0.003 i reaches 0.025 at i = 7 exactly; 0.05 between i = 15 (0.049) and never
    assert abs(mc.tipping(cv, "MASKED", 0.025)["omega"] - 0.7) < 1e-12
    assert mc.tipping(cv, "MASKED", 0.05)["text"] == "> 1.5×"


def test_tipping_edge_cases():
    cv = _curve(lambda i: 0.03, lambda i: 0.0)
    assert mc.tipping(cv, "CARRIED", 0.025)["omega"] == 0.0
    assert mc.tipping(cv, "CARRIED", 0.025)["text"].startswith("≤ 0×")
    assert mc.tipping(cv, "BOTH WAYS", 0.025) == {
        "omega": None,
        "text": "> 1.5×",
        "bracket": [1.5, None],
    }


def test_rate_at_clamps_to_grid_end():
    cv = _curve(lambda i: 0.01 * i, lambda i: 0.0)
    assert abs(mc.rate_at(cv, "CARRIED", 0.55) - 0.055) < 1e-12
    assert mc.rate_at(cv, "CARRIED", 9.0) == cv[-1]["CARRIED"]


def test_a_bound_coverage_on_synthetic_runs():
    """ω known; the bound must cover it about 95% of the time (2 df, one-sided)."""
    rng = np.random.default_rng(RNG_SEED)
    R, n, om = 4000, 37, 0.5
    zA = rng.normal(0, 1, (3, R, n)) + rng.normal(0, om, (3, R, 1)) + rng.normal(0, 3, (1, R, n))
    ok = np.ones((R, n), bool)
    b = mc.a_bound(zA, ok, np.ones(R))
    cov = float((b["ratio"] >= om).mean())
    assert 0.93 <= cov <= 0.97


def test_seed_noise_centred_ignores_a_run_offset():
    rng = np.random.default_rng(RNG_SEED)
    d1 = rng.normal(0, 1, 20000)
    d2 = rng.normal(0, 1, 20000)
    ok = np.ones(20000, bool)
    a = mc.seed_noise_centred(d1, d2, np.ones(20000), ok)
    b = mc.seed_noise_centred(d1 + 5.0, d2, np.ones(20000), ok)
    assert abs(a - b) < 1e-9 and abs(a - 0.5) < 0.02  # Var(Δ1 − Δ2) = 2 = 4σ², so σ² = 0.5


def test_omega_grid_in_calibration():
    cal = json.loads(mc.CALIBRATION.read_text())
    curve = mc.omega_curve(cal["rows"])
    assert [c["omega"] for c in curve] == list(mc.OMEGA_GRID)


def test_score_prints_run_noise_block_beside_the_state():
    rng = np.random.default_rng(RNG_SEED)
    w = _world(rng)
    for r in ("A1", "A2", "A3"):
        for c in range(8):
            w[r]["full"][c] = 0.0
    recs, banks = mc.build(rng, w)
    res = mc.score(recs, banks, calibration=json.loads(mc.CALIBRATION.read_text()))
    full = res["populations"]["full"]
    rn = full["aggregate"]["run_noise"]
    assert full["aggregate"]["state"] == "CARRIED" and rn["applies_to"] == ["CARRIED"]
    assert rn["headline"].startswith("below 2.5% false calls if run-specific noise < ")
    assert len(rn["curve"]) == 16 and rn["a_bound"]["df"] == 2 and rn["a_bound"]["level"] == 0.95
    assert rn["drift"]["label"] == "one draw of the run-level noise; not an estimate of its size"
    assert "omega_row" not in full["aggregate"]
    assert set(full["chance"]["A-SPLIT"]) == {"per_answer", "expected_of_37", "expected_scoreable"}
