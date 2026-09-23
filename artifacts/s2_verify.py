"""Brief S2 — verify D24: the production ladder's nuisance clearance reproduces Brief R0.

Runs `run_ladder` (the production path, with D24's retention clearance) on P2's union and split,
both populations, and compares it row by row against two records:

  * artifacts/out/r_r0.json   — the retention verdict and every number it was computed from
                                (A, C, M, M_lo, C_m). Expected: identical, except where the feature fails existence (D24's margin floor) — `t10 winding:
                                medium` (full), `t04 spiral` + `no spiral` (conditional) -> UNRESOLVED; see --flips.
  * artifacts/out/p2_ladder.json — rung, AUC, matched AUC. Expected: rungs unchanged; mechanisms
                                relabelled to the gate that actually failed.

Real and untrained matrices come from the banks Brief R used (O1's M + untrained seed 0, and
seeds 1-2), co-indexed with P2's union; only the noise control is extracted, and banked.

    uv run python artifacts/s2_verify.py    -> artifacts/out/s2_ladder.json
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

sys.path.insert(0, "artifacts")
import r_nonlinear as R  # noqa: E402
from p2_ladder import _rung_row  # noqa: E402

from galaxy_jepa.models.vit import load_frozen_encoder  # noqa: E402
from galaxy_jepa.probing import controls as ctl  # noqa: E402
from galaxy_jepa.probing import ladder as ladder_mod  # noqa: E402
from galaxy_jepa.probing.extract import EmbeddingMatrix  # noqa: E402

NOISE_BANK = R.OUT / "s2_noise.npz"
RECORD = R.OUT / "s2_ladder.json"
TOL = 1e-9


def noise_matrix(setup, frozen, ids: np.ndarray) -> EmbeddingMatrix:
    if NOISE_BANK.exists():
        blob = np.load(NOISE_BANK, allow_pickle=False)
        if np.array_equal(blob["ids"], ids):
            return EmbeddingMatrix(ids, blob["x"], "noise")
    t0 = time.perf_counter()
    m = ctl.noise_through_encoder_matrix(frozen, setup.ds, device=setup.device, seed=setup.cfg.seed)
    if not np.array_equal(m.object_ids, ids):
        raise SystemExit("S2: noise matrix is not co-indexed with the bank")
    np.savez(NOISE_BANK, ids=ids, x=m.x)
    print(f"S2 noise      : extracted in {time.perf_counter() - t0:.0f}s", file=sys.stderr)
    return m


def row(f: str, v) -> dict:
    out = _rung_row(f, v)
    m = v.matched
    if m is not None and m.retention is not None:
        out.update({"retention": m.retention.verdict, "retained": m.retention.retained,
                    "M_lo": m.matched_auc_lo, "C": m.bar_unmatched, "C_m": m.bar_matched,
                    "k_bar": m.k_bar, "margin_established": m.margin_established})
    return out


def main() -> None:
    setup = R.prepare("runs/m/encoder.pt", R.MAX_TRAIN, label="S2", sources=1)
    frozen = load_frozen_encoder(setup.ckpt)
    real, untrained = R.load_matrices(setup, frozen)
    noise = noise_matrix(setup, frozen, real.object_ids)
    controls = ctl.ControlEmbeddings(real=real, untrained=untrained[0], noise=noise,
                                     untrained_extra=tuple(untrained[1:]))
    labels = setup.labels
    out: dict[str, object] = {"checkpoint": str(setup.ckpt), "k_bar": len(controls.untrained_seeds)}
    for pop in ("full", "conditional"):
        lab = labels if pop == "full" else labels.with_population("conditional")
        t0 = time.perf_counter()
        res = ladder_mod.run_ladder(controls, lab, setup.train_ids, setup.test_ids,
                                    config=setup.pc, sky_label_col="snr_r")
        out[pop] = [row(f, v) for f, v in res.verdicts.items()]
        RECORD.write_text(json.dumps(out, indent=1, default=float))
        print(f"S2 {pop:<12s} {time.perf_counter() - t0:6.0f}s", file=sys.stderr)
    compare(out)


def compare(out: dict) -> None:
    p2 = json.loads((R.OUT / "p2_ladder.json").read_text())
    r0 = json.loads((R.OUT / "r_r0.json").read_text())
    for pop in ("full", "conditional"):
        old = {r["feature"]: r for r in p2[pop]}
        rungs_new: dict[str, int] = {}
        rungs_old: dict[str, int] = {}
        print(f"\nS2 {pop}", file=sys.stderr)
        for r in out[pop]:
            f, o, z = r["feature"], old[r["feature"]], r0.get(f"{pop}:{r['feature']}", {})
            rungs_new[r["rung"]] = rungs_new.get(r["rung"], 0) + 1
            rungs_old[o["rung"]] = rungs_old.get(o["rung"], 0) + 1
            notes = []
            if r["rung"] != o["rung"]:
                notes.append(f"RUNG {o['rung']}->{r['rung']}")
            if abs((r["auc"] or 0) - (o["auc"] or 0)) > TOL:
                notes.append(f"AUC {o['auc']:.6f}->{r['auc']:.6f}")
            if (r["matched_auc"] is None) != (o["matched_auc"] is None) or (
                    r["matched_auc"] is not None and abs(r["matched_auc"] - o["matched_auc"]) > TOL):
                notes.append(f"matched {o['matched_auc']}->{r['matched_auc']}")
            if z and r.get("retention") != z.get("verdict"):
                notes.append(f"VERDICT {z.get('verdict')}->{r.get('retention')}")
            for k_new, k_old in (("C", "C"), ("C_m", "C_m"), ("M_lo", "M_lo"), ("auc", "A")):
                a, b = r.get(k_new), z.get(k_old)
                if a is not None and b is not None and abs(a - b) > TOL:
                    notes.append(f"{k_old} {b:.6f}->{a:.6f}")
            flag = "  " + "; ".join(notes) if notes else ""
            print(f"  {f:<44s} {r['rung']:<3s} {str(r.get('retention')):<10s} "
                  f"{r['mechanism'][:44]:<44s}{flag}", file=sys.stderr)
        same = "UNCHANGED" if rungs_new == rungs_old else "CHANGED"
        print(f"S2 {pop} rung counts {dict(sorted(rungs_old.items()))} -> "
              f"{dict(sorted(rungs_new.items()))}  {same}", file=sys.stderr)


FLIPS = (("full", "t10_arms_winding_a29_medium"), ("conditional", "t04_spiral_a08_spiral"),
         ("conditional", "t04_spiral_a09_no_spiral"))


def flips() -> None:
    """The predicted flips, through the production clearance: with the margin floor lifted it must
    give R0's verdict; with the feature's real existence status (failed) it must give UNRESOLVED.
    Needed because P2-era failing verdicts did not carry their clearance onto the record."""
    from types import SimpleNamespace

    from galaxy_jepa.probing.extract import feature_embeddings, feature_ids

    setup = R.prepare("runs/m/encoder.pt", R.MAX_TRAIN, label="S2", sources=1)
    frozen = load_frozen_encoder(setup.ckpt)
    real, untrained = R.load_matrices(setup, frozen)
    controls = ctl.ControlEmbeddings(real=real, untrained=untrained[0], noise=real,
                                     untrained_extra=tuple(untrained[1:]))
    s2 = json.loads(RECORD.read_text())
    r0 = json.loads((R.OUT / "r_r0.json").read_text())
    out = {}
    for pop, f in FLIPS:
        lab = setup.labels if pop == "full" else setup.labels.with_population("conditional")
        tr = feature_embeddings(real, lab, f, setup.train_ids)
        te = feature_embeddings(real, lab, f, setup.test_ids)
        panel = ctl.nuisance_panel(tr, te, lab, feature_ids(real, lab, f, setup.train_ids),
                                   feature_ids(real, lab, f, setup.test_ids), c=setup.pc.c)
        auc = next(r["auc"] for r in s2[pop] if r["feature"] == f)
        fc = SimpleNamespace(nuisance_aucs=panel, real_auc=auc)
        got = {}
        for established in (True, False):
            _, m = ladder_mod._nuisance_clearance(f, fc, controls, lab, setup.train_ids,
                                                  setup.test_ids, margin_established=established,
                                                  config=setup.pc)
            got[established] = m.retention.verdict
        want = r0[f"{pop}:{f}"]["verdict"]
        ok = got[True] == want and got[False] == "UNRESOLVED"
        out[f"{pop}:{f}"] = {"r0": want, "floor_lifted": got[True], "production": got[False],
                             "ok": ok}
        print(f"  {pop:<12s} {f:<34s} R0 {want:<10s} floor lifted {got[True]:<10s} "
              f"production {got[False]:<10s} {'OK' if ok else 'MISMATCH'}", file=sys.stderr)
    (R.OUT / "s2_flips.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1:] == ["--compare"]:
        compare(json.loads(RECORD.read_text()))
    elif sys.argv[1:] == ["--flips"]:
        flips()
    else:
        main()
