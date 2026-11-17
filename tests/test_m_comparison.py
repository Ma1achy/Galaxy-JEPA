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
