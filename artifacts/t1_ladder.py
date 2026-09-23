"""Brief T1 — D25: nothing relative reads the effect floor. The production ladder, rerun.

Three modes, in the order the brief requires:

    uv run python artifacts/t1_ladder.py --old-pairs   -> out/t1_old_pairs.json
        The 2A pair list per population under the OLD conditional leg (matched AUC >= 0.7267),
        rebuilt from S2's record — P2 only kept the full population's. Input to the prediction;
        it computes nothing under the new rule.
    uv run python artifacts/t1_ladder.py               -> out/t1_ladder.json
        run_ladder with D25 (retention on the conditional leg; MLP decode unadjudicated), both
        populations, with every pair verdict, the MLP sweep/ceiling and the competitive list.
    uv run python artifacts/t1_ladder.py --compare
        against S2's record (artifacts/out/s2_ladder.json), rung by rung, pair by pair.

The pre-registration is in artifacts/t_findings.md, committed-before-run by hash.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time

import numpy as np

sys.path.insert(0, "artifacts")
import r_nonlinear as R  # noqa: E402
from s2_verify import noise_matrix, row  # noqa: E402

from galaxy_jepa.models.vit import load_frozen_encoder  # noqa: E402
from galaxy_jepa.probing import controls as ctl  # noqa: E402
from galaxy_jepa.probing import entanglement as ent  # noqa: E402
from galaxy_jepa.probing import ladder as ladder_mod  # noqa: E402
from galaxy_jepa.probing import matching as match  # noqa: E402
from galaxy_jepa.probing.extract import feature_embeddings, feature_ids  # noqa: E402
from galaxy_jepa.probing.logistic import probe_direction  # noqa: E402

RECORD = R.OUT / "t1_ladder.json"
OLD_PAIRS = R.OUT / "t1_old_pairs.json"
S2 = R.OUT / "s2_ladder.json"


def question(f: str) -> str:
    return f.split("_a")[0]


def _setup():
    setup = R.prepare("runs/m/encoder.pt", R.MAX_TRAIN, label="T1", sources=1)
    frozen = load_frozen_encoder(setup.ckpt)
    real, untrained = R.load_matrices(setup, frozen)
    return setup, frozen, real, untrained


def old_pairs() -> None:
    """The S2-state pair list: the geometry over S2's existence-passing answers, and the old leg."""
    setup, _, real, _ = _setup()
    s2 = json.loads(S2.read_text())
    pc, out = setup.pc, {}
    for pop in ("full", "conditional"):
        lab = setup.labels if pop == "full" else setup.labels.with_population("conditional")
        passing = [r["feature"] for r in s2[pop] if r["rung"] in ("R1", "R2")]
        dirs, cav = [], {}
        for f in passing:
            tr = feature_embeddings(real, lab, f, setup.train_ids)
            d = probe_direction(tr, name=f, c=pc.c)
            dirs.append(d)
            cav[f] = ent.logistic_cav_disagreement(d.w_unit, ent.cav_direction(tr))
        geo = ent.entanglement_geometry(dirs, real.x, mp_method=pc.mp_method,
                                        pair_quantile=pc.entangled_pair_quantile,
                                        cav_disagreement=cav)
        idx = {n: i for i, n in enumerate(geo.names)}
        rows = []
        for a, b in geo.entangled_pairs:
            tr = feature_embeddings(real, lab, a, setup.train_ids)
            te = feature_embeddings(real, lab, a, setup.test_ids)
            btr = np.asarray(lab.vote_fraction(b, feature_ids(real, lab, a, setup.train_ids)))
            bte = np.asarray(lab.vote_fraction(b, feature_ids(real, lab, a, setup.test_ids)))
            i_tr, i_te = match.matched_indices(btr, tr.y, bte, te.y, n_strata=5, seed=pc.seed)
            ok = i_te.size and len(np.unique(tr.y[i_tr])) == 2 and len(np.unique(te.y[i_te])) == 2
            survived = (match.matched_auc(tr, te, btr, bte, n_strata=5, c=pc.c, seed=pc.seed)
                        >= pc.effect_floor) if ok else None
            pv = ent.adjudicate_pair(a, b, cosine=float(geo.cosine[idx[a], idx[b]]),
                                     mp_significant=bool(geo.mp.significant),
                                     cav_disagreement=geo.cav_disagreement,
                                     survived_matching=survived)
            rows.append({"a": a, "b": b, "verdict": pv.verdict, "survived": survived,
                         "cosine": pv.cosine, "within_question": question(a) == question(b),
                         "n_matched_test": int(i_te.size)})
        out[pop] = {"passing": passing, "mp_significant": bool(geo.mp.significant), "pairs": rows}
        OLD_PAIRS.write_text(json.dumps(out, indent=1))
        print(f"T1 old pairs {pop}: {len(rows)} pairs over {len(passing)} passing", file=sys.stderr)
    if json.loads(S2.read_text()) and "full" in out:
        p2 = json.loads((R.OUT / "p2_ladder.json").read_text())["pair_verdicts"]
        want = {(p["a"], p["b"]): p["verdict"] for p in p2}
        got = {(p["a"], p["b"]): p["verdict"] for p in out["full"]["pairs"]}
        print(f"T1 old pairs full reproduce P2's recorded 53: {want == got}", file=sys.stderr)


def main() -> None:
    setup, frozen, real, untrained = _setup()
    noise = noise_matrix(setup, frozen, real.object_ids)
    controls = ctl.ControlEmbeddings(real=real, untrained=untrained[0], noise=noise,
                                     untrained_extra=tuple(untrained[1:]))
    out: dict[str, object] = {"checkpoint": str(setup.ckpt), "k_bar": len(controls.untrained_seeds)}
    for pop in ("full", "conditional"):
        lab = setup.labels if pop == "full" else setup.labels.with_population("conditional")
        t0 = time.perf_counter()
        res = ladder_mod.run_ladder(controls, lab, setup.train_ids, setup.test_ids,
                                    config=setup.pc, sky_label_col="snr_r")
        rows = []
        for f, v in res.verdicts.items():
            r = row(f, v)
            fc = res.feature_controls[f]
            r["competitive"] = [n for n, a in fc.nuisance_aucs.items()
                                if match.nuisance_competitive(
                                    fc.real_auc, a, margin=setup.pc.nuisance_competitive_margin)]
            r["exceeds_null"] = bool(res.existence[f].exceeds_null)
            r["clean"] = bool(res.existence[f].clean)
            if v.sweep:
                r["sweep"] = [dataclasses.asdict(s) for s in v.sweep]
                r["ceiling"] = v.ceiling
            rows.append(r)
        out[pop] = rows
        out[f"{pop}_pairs"] = [dataclasses.asdict(p) for p in res.pair_verdicts]
        RECORD.write_text(json.dumps(out, indent=1, default=float))
        print(f"T1 {pop:<12s} {time.perf_counter() - t0:6.0f}s", file=sys.stderr)
    compare(out)


def compare(out: dict) -> None:
    s2 = json.loads(S2.read_text())
    old = json.loads(OLD_PAIRS.read_text())
    floor = 0.7267
    for pop in ("full", "conditional"):
        before = {r["feature"]: r for r in s2[pop]}
        counts_old: dict[str, int] = {}
        counts_new: dict[str, int] = {}
        print(f"\nT1 {pop} — rungs and mechanisms that moved", file=sys.stderr)
        for r in out[pop]:
            f, o = r["feature"], before[r["feature"]]
            counts_old[o["rung"]] = counts_old.get(o["rung"], 0) + 1
            counts_new[r["rung"]] = counts_new.get(r["rung"], 0) + 1
            notes = []
            if r["rung"] != o["rung"]:
                notes.append(f"RUNG {o['rung']}->{r['rung']}")
            if r["mechanism"] != o["mechanism"]:
                notes.append(f"'{o['mechanism'][:34]}' -> '{r['mechanism'][:40]}'")
            if abs((r["auc"] or 0) - (o["auc"] or 0)) > 1e-9:
                notes.append("AUC MOVED")
            if r.get("retention") != o.get("retention"):
                notes.append(f"clearance {o.get('retention')}->{r.get('retention')}")
            if notes:
                print(f"  {f:<44s} " + "; ".join(notes), file=sys.stderr)
        same = "UNCHANGED" if counts_old == counts_new else "CHANGED"
        print(f"T1 {pop} rung counts {dict(sorted(counts_old.items()))} -> "
              f"{dict(sorted(counts_new.items()))}  {same}", file=sys.stderr)
        was = {(p["a"], p["b"]): p for p in old[pop]["pairs"]}
        print(f"T1 {pop} pairs — old verdict -> retention -> new verdict", file=sys.stderr)
        for p in out[f"{pop}_pairs"]:
            w = was.get((p["a"], p["b"]))
            a_auc = next(r["auc"] for r in out[pop] if r["feature"] == p["a"])
            flag = "" if w and w["verdict"] == p["verdict"] else "  MOVED"
            print(f"  {'WQ' if question(p['a']) == question(p['b']) else '  '} "
                  f"{p['a'][:30]:<30s} {p['b'][:30]:<30s} A={a_auc:.3f}"
                  f"{'<F' if a_auc < floor else '  '} {w['verdict'][:5] if w else '?':<5s} -> "
                  f"{str(p['retention']):<10s} ret={p['retained'] if p['retained'] is None else round(p['retained'], 2)!s:<6s} -> "
                  f"{p['verdict'][:5]}{flag}", file=sys.stderr)
        # the MLP leg: would the old absolute rule have produced R3 anywhere (before demotion)?
        for r in out[pop]:
            if "sweep" not in r:
                continue
            ceil = r["ceiling"]
            hit = next((s["width"] for s in r["sweep"]
                        if (ceil is None or s["width"] < ceil) and s["real_auc"] >= floor), None)
            print(f"  MLP {r['feature'][:40]:<40s} old-rule decode at width {hit}; "
                  f"competitive {r['competitive'] or '-'}; cleared "
                  f"{'yes' if not r['competitive'] else r.get('retention')}", file=sys.stderr)


if __name__ == "__main__":
    if sys.argv[1:] == ["--old-pairs"]:
        old_pairs()
    elif sys.argv[1:] == ["--compare"]:
        compare(json.loads(RECORD.read_text()))
    else:
        main()
