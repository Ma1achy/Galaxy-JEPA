"""`baselines_comparison.score` on synthetic inputs only (user, 4 Nov): no real trajectory file, bank or
ladder output is opened. Every path the scoring reads is a parameter; here each points into tmp_path.

Covers P3's file check (manifests and baselines.md §6.1 prefixes), P1's verdict, P3's exact refit
check, the score core (decide per objective and epoch, P1 voiding, the trajectory, MoCo against MAE
reported without a state), §N's labels and fits, and the --go refusal.

Importing the comparison's helpers pulls `aligned_c2`, which reads F0's environment at import; the
module imports `baseline_run` first, as the scoring does. Needs the v2 cache on the external disk.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[1]
CACHE = Path("/Volumes/X10 Pro/galaxy-jepa/runs/m_v2/cache")
if not CACHE.exists():
    pytest.skip("the v2 cache is not mounted", allow_module_level=True)
sys.path.insert(0, str(REPO / "artifacts"))
import baseline_run  # noqa: E402,F401  — the v2 environment before aligned_c2 reads it
import baselines_comparison as bc  # noqa: E402

J = 35
ANSWERS = [f"f{j:02d}" for j in range(J)]
W = np.ones(J)


def _tables(rng, delta_x: float, tau: float = 0.002, sig: float = 0.004, b: int = 400):
    """auc[e][j], boot[e][j] for the six encoders: a shared base, run-level and answer-level noise,
    `delta_x` on every baseline; bootstrap draws share one galaxy component (paired)."""
    base = rng.uniform(0.70, 0.95, J)
    shared = rng.normal(0, 0.004, (b, J))
    A, B = {}, {}
    for a, x in bc.PAIRS:
        for e, d in ((a, 0.0), (x, delta_x)):
            p = base + rng.normal(0, tau) + rng.normal(0, sig, J) + d
            A[e] = dict(zip(ANSWERS, p.tolist(), strict=True))
            draws = p + shared + rng.normal(0, 0.001, (b, J))
            B[e] = {j: draws[:, i] for i, j in enumerate(ANSWERS)}
    return A, B


PASS = {a: {"identity": True, "e2e": True, "passes": True} for a, _ in bc.PAIRS}


def test_score_core_reaches_better_same_and_the_trajectory():
    rng = np.random.default_rng(1)
    tables = {"moco": {ep: _tables(rng, 0.03) for ep in bc.EPOCHS},
              "mae": {ep: _tables(rng, 0.0, tau=0.0005, sig=0.002) for ep in bc.EPOCHS},
              "moco_vs_mae": {bc.PRIMARY: _tables(rng, 0.0)}}
    r = bc.score_core(tables, ANSWERS, W, PASS)
    assert r["moco"]["primary"]["state"] == "BETTER"
    assert r["moco"]["trajectory"] == "CONSISTENT"
    assert r["mae"]["primary"]["state"] == "SAME"
    mm = r["moco_vs_mae"][str(bc.PRIMARY)]
    assert "state" not in mm and all("state" not in v for v in mm["answers"].values())
    # per-answer states at 4 epochs only (§B); earlier epochs carry D_j and its CI
    assert all("state" in v for v in r["moco"]["primary"]["answers"].values())
    assert all("state" not in v and "ci" in v for v in r["moco"]["epochs"]["0.5"]["answers"].values())


def test_score_core_worse_and_crosses():
    rng = np.random.default_rng(2)
    seq = (0.03, 0.03, -0.03, -0.03)
    tables = {"moco": {ep: _tables(rng, d) for ep, d in zip(bc.EPOCHS, seq, strict=True)},
              "mae": {ep: _tables(rng, -0.03) for ep in bc.EPOCHS}}
    r = bc.score_core(tables, ANSWERS, W, PASS)
    assert r["moco"]["trajectory"] == "CROSSES"
    assert r["mae"]["primary"]["state"] == "WORSE"


def test_a_p1_failure_voids_the_early_epochs_and_the_trajectory_only():
    rng = np.random.default_rng(3)
    tables = {k: {ep: _tables(rng, 0.03) for ep in bc.EPOCHS} for k in bc.KINDS}
    p1 = {**PASS, "A2": {"identity": True, "e2e": False, "passes": False}}
    r = bc.score_core(tables, ANSWERS, W, p1)
    for k in bc.KINDS:
        assert [r[k]["epochs"][str(e)]["state"] for e in bc.EPOCHS[:3]] == ["VOID"] * 3
        assert r[k]["trajectory"] == "VOID"
        assert r[k]["primary"]["state"] == "BETTER"


def test_a_p3_stop_voids_the_missing_epochs():
    rng = np.random.default_rng(4)
    tables = {k: {bc.EPOCHS[0]: _tables(rng, 0.0)} for k in bc.KINDS}
    r = bc.score_core(tables, ANSWERS, W, PASS)
    assert r["moco"]["primary"]["state"] == "VOID"
    assert r["moco"]["trajectory"] == "VOID"


def test_too_few_answers_is_void():
    rng = np.random.default_rng(5)
    A, B = _tables(rng, 0.03)
    for j in ANSWERS[: J - bc.MIN_ANSWERS + 1]:
        A["X2"].pop(j)
    r = bc.score_core({k: {ep: (A, B) for ep in bc.EPOCHS} for k in bc.KINDS}, ANSWERS, W, PASS)
    assert r["moco"]["primary"]["state"] == "VOID"


def _sha(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()


def _files(tmp_path: Path) -> tuple[Path, Path]:
    out = tmp_path / "out"
    out.mkdir()
    for f in bc.needed_files(out):
        f.write_bytes(f.name.encode())
    man = tmp_path / "p3.sha1"
    man.write_text("# test\n" + "".join(f"{_sha(f)}  artifacts/out/{f.name}\n" for f in bc.needed_files(out)))
    return out, man


def test_p3_files_pass_then_catch_a_changed_file_and_a_wrong_prefix(tmp_path):
    out, man = _files(tmp_path)
    assert len(bc.needed_files(out)) == 4 * 9 * 2
    assert bc.p3_hashes(out, [man])["mismatches"] == []
    t = out / "moco_s01_trajectory.json"
    md = tmp_path / "b.md"
    rows = []
    for run in ("moco_s0", "mae_s0", "moco_s1", "mae_s1"):
        cells = [_sha(out / f"{run}{k}_trajectory.json")[:12] for k in range(1, 5)]
        rows.append(f"| `{run}` | " + " | ".join(f"`{c}`" for c in cells) + " |")
    md.write_text("\n".join(rows) + "\n")
    assert bc.p3_hashes(out, [man], md)["mismatches"] == []
    md.write_text(md.read_text().replace(_sha(out / "moco_s02_trajectory.json")[:12], "000000000000"))
    bad = bc.p3_hashes(out, [man], md)["mismatches"]
    assert bad == [f"moco_s02_trajectory.json: manifest {_sha(out / 'moco_s02_trajectory.json')[:12]} "
                   "!= baselines.md §6.1 000000000000"]
    md.write_text(rows[0] + "\n")  # a table that parses short is refused, never passed vacuously
    with pytest.raises(bc.P3Mismatch, match="prefixes"):
        bc.p3_hashes(out, [man], md)
    with pytest.raises(bc.P3Mismatch, match="twice"):
        bc.p3_hashes(out, [man, man])
    # a run whose P1 fails needs only its 4-epoch files
    assert len(bc.needed_files(out, ("A2",))) == len(bc.needed_files(out)) - 3 * 2
    t.write_text("changed")
    assert any(m.startswith("moco_s01_trajectory.json:") for m in bc.p3_hashes(out, [man])["mismatches"])
    (out / "mae_s24_trajectory.json").unlink()
    assert any("missing" in m for m in bc.p3_hashes(out, [man])["mismatches"])


def test_p1_status_needs_both_resumes(tmp_path):
    for a, _ in bc.PAIRS:
        for part in ("identity", "e2e"):
            d = tmp_path / f"{a}_{part}"
            d.mkdir()
            ok = not (a == "A3" and part == "e2e")
            (d / "p1_record.json").write_text(json.dumps({"amended": {"passes": ok}}))
    st = bc.p1_status(tmp_path)
    assert st["A1"]["passes"] and st["A2"]["passes"] and not st["A3"]["passes"]
    (tmp_path / "A1_e2e" / "p1_record.json").unlink()
    with pytest.raises(SystemExit, match="missing"):  # a missing record is an error, not a failed P1
        bc.p1_status(tmp_path)


def test_refit_requires_exact_equality_with_the_trajectory_file(tmp_path):
    from aligned_c2 import auc, weights
    rng = np.random.default_rng(6)
    test = list(range(1000, 1400))
    wt, pos = weights(test, b=50)
    y = (rng.random(len(test)) < 0.4).astype(int)
    s = y * 1.0 + rng.normal(0, 1, len(test))
    bank = "x_s01_000012664"
    fit = lambda tags: ({bank: {"f00": s}}, {"y": {"f00": y}, "rows": {"f00": test}}, test)  # noqa: E731
    rec = {"checkpoints": [{"step": 12664, "answers": {"f00": {"auc": auc(s, y)}}}]}
    (tmp_path / "x_s01_trajectory.json").write_text(json.dumps(rec))
    A, B = bc.refit("x_s01", 12664, tmp_path, ["f00", "f01"], fit, wt, pos)
    assert A == {"f00": auc(s, y)} and B["f00"].shape == (50,)
    rec["checkpoints"][0]["answers"]["f00"]["auc"] = math.nextafter(auc(s, y), 1.0)
    (tmp_path / "x_s01_trajectory.json").write_text(json.dumps(rec))
    with pytest.raises(bc.P3Mismatch):
        bc.refit("x_s01", 12664, tmp_path, ["f00"], fit, wt, pos)
    with pytest.raises(bc.P3Mismatch):
        bc.refit("x_s01", 25327, tmp_path, ["f00"], fit, wt, pos)
    with pytest.raises(bc.P3Mismatch):  # a different test split from the shared weights'
        bc.refit("x_s01", 12664, tmp_path, ["f00"], lambda t: (*fit(t)[:2], test[::-1]), wt, pos)
    rec["checkpoints"][0]["answers"]["f00"]["auc"] = auc(s, y)
    rec["checkpoints"][0]["answers"]["f01"] = {"auc": 0.7}  # filed, but the refit cannot score it
    (tmp_path / "x_s01_trajectory.json").write_text(json.dumps(rec))
    with pytest.raises(bc.P3Mismatch, match="different answer set"):
        bc.refit("x_s01", 12664, tmp_path, ["f00", "f01"], fit, wt, pos)


def test_nuisance_labels_exclude_sentinels_and_split_at_their_own_median():
    v = np.array([-9999.0, 1.0, 2.0, 3.0, 4.0, np.nan, 0.0])
    ok, y = bc.nuisance_labels(v, "psf")
    assert ok.tolist() == [False, True, True, True, True, False, True]
    assert y.tolist() == [0, 1, 1, 1, 0]  # median 2: ≥ is 1
    ok, y = bc.nuisance_labels(v, "sky")  # non-positive sky is excluded too
    assert ok.tolist() == [False, True, True, True, True, False, False]
    assert y.tolist() == [0, 0, 1, 1]


def test_nuisance_tables_carry_a_planted_signal():
    from aligned_c2 import weights
    rng = np.random.default_rng(7)
    tr, te = list(range(0, 600)), list(range(600, 900))
    wt, pos = weights(te, b=50)
    v = rng.normal(size=900)
    x_sig = np.c_[v + rng.normal(0, 0.3, 900), rng.normal(size=(900, 4))].astype(np.float32)
    x_none = rng.normal(size=(900, 5)).astype(np.float32)
    ids = np.arange(900)
    A, B = bc.nuisance_tables({"sig": (ids, x_sig), "none": (ids, x_none)}, tr, te,
                              {"psf": (v[:600], v[600:])}, wt, pos, c=1.0)
    assert A["sig"]["psf"] > 0.85 and abs(A["none"]["psf"] - 0.5) < 0.1
    assert B["sig"]["psf"].shape == (50,)


def test_score_refuses_without_go():
    r = subprocess.run([sys.executable, "artifacts/baselines_comparison.py", "score"], cwd=REPO,
                       capture_output=True, text=True)
    assert r.returncode == 1 and "refuses without --go" in r.stderr


def test_score_refuses_after_the_import_order_trap_and_under_a_ladder_environment():
    # this module has imported aligned_c2 already: in-process scoring must refuse before reading anything
    with pytest.raises(SystemExit, match="imported before"):
        bc.score(go=True)
    import os
    env = {**os.environ, "LADDER_ENCODER": "x.pt", "LADDER_OUT": "/tmp/nowhere"}
    r = subprocess.run([sys.executable, "artifacts/baselines_comparison.py", "score", "--go"], cwd=REPO,
                       capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "LADDER_ENCODER / LADDER_OUT are set" in r.stderr


def test_all_void_names_every_state():
    v = bc.all_void("P3: x")
    for k in bc.KINDS:
        assert v["comparison"][k]["trajectory"] == "VOID"
        assert {e["state"] for e in v["comparison"][k]["epochs"].values()} == {"VOID"}
        assert {n["state"] for n in v["nuisance"][k]["states"].values()} == {"VOID"}


def _ladder(tmp_path: Path, rung_of):
    """Synthetic ladder records for nine encoders and the two runners' state files."""
    groups = {"JEPA": ("a1", "a2", "a3"), "MoCo": ("moco_s0", "moco_s1", "moco_s2"),
              "MAE": ("mae_s0", "mae_s1", "mae_s2")}
    feats = [f"t{k:02d}_q_a{k:02d}_x" for k in range(bc.N_FEATURES)]
    files = {}
    for g in groups.values():
        for e in g:
            d = tmp_path / e
            d.mkdir()
            for f in bc.LADDER_FILES:
                rows = [{"feature": ft, "rung": rung_of(e, ft), "mechanism": "m", "retention": "SURVIVES",
                         "exceeds_null": True, "clean": True, "underpowered": False} for ft in feats]
                (d / f).write_text(json.dumps({"checkpoint": f"runs/ladder_v2/encoders/{e}.pt",
                                               "full": rows, "conditional": rows}))
                files[str(d / f)] = [_sha(d / f), "E"]
    st = tmp_path / "state.json"
    st.write_text(json.dumps({"entries": {"E": {"status": "done"}}, "files": files}))
    return groups, feats, st


def test_ladder_table_counts_agreement_and_forms_no_comparative_state(tmp_path):
    groups, feats, st = _ladder(tmp_path, lambda e, f: "R4" if (e == "moco_s1" and f.startswith("t00")) else "R1")
    t = bc.ladder_table(tmp_path, (st,), groups)
    row = t["populations"]["full"][feats[0]]
    assert row["agreement"] == {"JEPA": "3/3", "MoCo": "2/1 (moco_s1 differs)", "MAE": "3/3"}
    assert row["moco_s1"]["rung"] == "R4" and row["moco_s1"]["files_agree"]
    assert set(row) == {*[e for g in groups.values() for e in g], "agreement"}


def test_ladder_agreement_with_no_majority_says_so(tmp_path):
    rung = {"moco_s0": "R1", "moco_s1": "R2", "moco_s2": "R4"}
    groups, feats, st = _ladder(tmp_path, lambda e, f: rung.get(e, "R1"))
    t = bc.ladder_table(tmp_path, (st,), groups)
    assert t["populations"]["conditional"][feats[3]]["agreement"]["MoCo"] == "0/3 (all differ)"


def test_ladder_table_refuses_unfinished_tampered_or_short_records(tmp_path):
    groups, feats, st = _ladder(tmp_path, lambda e, f: "R1")
    d = json.loads(st.read_text())
    d["entries"]["E2"] = {"status": "running"}
    st.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="not done"):
        bc.ladder_table(tmp_path, (st,), groups)
    del d["entries"]["E2"]
    st.write_text(json.dumps(d))
    (tmp_path / "mae_s2" / "t1_ladder.json").write_text("{}")
    with pytest.raises(SystemExit, match="recorded SHA-1"):
        bc.ladder_table(tmp_path, (st,), groups)
