"""Aligned comparison: the scoring run, exactly as hashed (`aligned_comparison.md`, 637dddda…).

Every statistic is the hashed machinery's own (`aligned_c2`, `aligned_c13`), called unchanged. This
driver adds only what the hashed scripts left to the run (user, 2026-10-09):

- **The identical set** (declared 2026-10-06): all four encoders are read on the 230,349 galaxies
  probe_v2 holds, so the 9 excluded galaxies are dropped from M1's and M2's v1 banks. Criterion 2's
  full train split drops them directly (160,849 / 34,828). Criteria 1 and 3 keep the DD cap: the
  deterministic 40,000-stride is taken on the v1 split first and the 9 dropped after, as the hashed
  confound floor did ("the 40,000 cap less 3"), so the stride never shifts.
- **Powered answers** are fixed from M1 and M2 alone on the identical set and written before any
  aligned bank is read (`powered`); every later step reads that file.
- **M1 on probe_v2** (reported, no state): probe_v2's FITS through M's own frozen pipeline
  (`aligned_c13._V2Stamps`, its v1 parity check included), so only the pixels change.
- **Flags** (criterion 3): the offset-flagged fraction of live latents, the rule that gave the hashed
  M1 0.2747 / M2 0.5036: flagged (strongest |ρ| is a nuisance) and the strongest nuisance is a band
  offset (`offset_rho` ≥ `best_nuisance`), over latents with density ≥ 1e-4. Read from each
  encoder's 4,999-panel DD outputs.

    uv run python artifacts/aligned_score.py powered        # M1, M2 only — before any aligned bank
    uv run python artifacts/aligned_score.py embed-m1v2
    uv run python artifacts/aligned_score.py c1a | c2 | c3
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import aligned_c2 as C2  # noqa: E402
import aligned_c13 as C13  # noqa: E402
from j4_spread_controls import OUT  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = OUT / "aligned"
#: the 9 probe galaxies probe_v2 lacks (GZ2 position > 3″ from PhotoObj; `aligned_comparison.md`,
#: declared 2026-10-06)
NINE = frozenset({1237651273510813752, 1237655472895099027, 1237658423543595058, 1237658802034900998,
                  1237661976553652249, 1237662263251042419, 1237662264854904841, 1237662528990674970,
                  1237665329849958421})
N_IDENT, N_TRAIN, N_TEST = 230_349, 160_849, 34_828
ENC = ("a1", "a2", "m1", "m2")
#: each encoder's DD outputs on the 4,999 panel (M1's and M2's: the 2026-10-09 recompute)
DD_RUN = {"m1": "runs/dd_v4ref_4999", "m2": "runs/dd_o2_4999", "a1": "runs/dd_a1", "a2": "runs/dd_a2"}
LIVE = 1e-4


def _write(name: str, obj: dict) -> dict:
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))
    return obj


def _identical_prepare(*a, **k):
    """j4's prepare, its split cut to the identical set (no cap involved: criterion 2's full split)."""
    s = _prepare(*a, **k)
    tr, te = [i for i in s.train_ids if i not in NINE], [i for i in s.test_ids if i not in NINE]
    if (len(tr), len(te)) != (N_TRAIN, N_TEST):
        raise SystemExit(f"identical set: {len(tr):,} train / {len(te):,} test, expected {N_TRAIN:,} / {N_TEST:,}")
    return dataclasses.replace(s, train_ids=tr, test_ids=te, union=sorted({*tr, *te}))


_prepare = C2.prepare
C2.prepare = _identical_prepare  # fit_scores reads the split through this name; the statistic is untouched


def setup13() -> dict:
    """aligned_c13.setup (the v1 split, capped at 40,000 by its stride), then the 9 dropped."""
    from dd_sae_cards import _panel
    d = C13.setup()
    tr, te = [i for i in d["train"] if i not in NINE], [i for i in d["test"] if i not in NINE]
    if len(te) != N_TEST:
        raise SystemExit(f"identical set: {len(te):,} test, expected {N_TEST:,}")
    d.update(train=tr, test=te, pan_train=_panel(np.array(tr)), pan_test=_panel(np.array(te)),
             dropped_from_capped_train=len(d["train"]) - len(tr))
    return d


# ── powered answers: M alone, before any aligned bank ────────────────────────────────────────────────

def powered() -> dict:
    if (RES / "powered.json").exists():
        raise SystemExit("powered.json exists: the list is fixed once, before any aligned bank is read")
    sc, meta, test_ids = C2.fit_scores(["m1", "m2"])  # reads m1 and m2 only
    ans = C2.powered(sc, meta["y"])
    hashed = json.loads((OUT / "c2_plants.json").read_text())["powered"]
    return _write("powered", {"powered": ans, "n": len(ans), "n_test": len(test_ids),
                              "same_as_hashed_plants": ans == hashed,
                              "added": sorted(set(ans) - set(hashed)), "dropped": sorted(set(hashed) - set(ans)),
                              "m_auc": {e: {f: C2.auc(sc[e][f], meta["y"][f]) for f in meta["y"]} for e in ("m1", "m2")}})


def _powered() -> list[str]:
    return json.loads((RES / "powered.json").read_text())["powered"]


# ── M1 on probe_v2 (reported, no state) ──────────────────────────────────────────────────────────────

def embed_m1v2() -> dict:
    from f0_preconditions import check
    from galaxy_jepa.harness import _build_pipeline
    from galaxy_jepa.models.vit import load_frozen_encoder
    from galaxy_jepa.probing.extract import extract_matrix
    from probe_bank import write_bank
    cfg, cache = check(verbose=False)
    if not cfg.normalisation.content_hash.startswith("75100066b3e0"):
        raise SystemExit("embed-m1v2 must run under M's config and freeze (unset GJ_CONFIG)")
    pipe = _build_pipeline(q=cfg.q, freeze=cfg.normalisation)
    s = _identical_prepare(None, 0, label="C2-m1v2", sources=1)
    ids = s.union
    v1 = C13._V2Stamps(ids[:16], pipe, C13.REPO_DATA / "probe")
    parity = max(float(np.abs(v1[k]["image"].numpy().astype(np.float32) - np.asarray(cache.get(o), np.float32)).max())
                 for k, o in enumerate(ids[:16]))
    if parity != 0.0:
        raise SystemExit(f"FITS path differs from M's cache on v1 stamps (max |Δ| {parity})")
    ckpt = REPO / "runs/m/encoder.pt"
    m = extract_matrix(load_frozen_encoder(ckpt), C13._V2Stamps(ids, pipe, C13.REPO_DATA / "probe_v2"),
                       device=cfg.runtime.resolved_device())
    write_bank(C2._bank("m1v2"), ids=m.object_ids, x=m.x.astype(np.float32), checkpoint=str(ckpt))
    return _write("m1v2_embed", {"n": len(m.object_ids), "parity_v1_max_abs": parity,
                                 "normalisation": cfg.normalisation.content_hash})


# ── criterion 1a ─────────────────────────────────────────────────────────────────────────────────────

def c1a() -> dict:
    d = setup13()
    out = {"n_train_capped": len(d["train"]), "dropped_from_capped_train": d["dropped_from_capped_train"],
           "n_test": len(d["test"]), "encoders": {}}
    for t in ("a1", "a2", "m1v2"):
        x_tr, x_te = C13._load(t, d["train"]), C13._load(t, d["test"])
        r = {}
        for name in C13.OFFSETS:
            xtr, ytr, xte, yte, _, _ = C13.c1_arrays(d, x_tr, x_te, name)
            r[name] = C13.offset_state(xtr, ytr, xte, yte) | {"n_train": len(ytr)}
        out["encoders"][t] = r
    return _write("c1a", out)


# ── criterion 2 ──────────────────────────────────────────────────────────────────────────────────────

ELONGATION = ("t02_edgeon_a04_yes", "t02_edgeon_a05_no", "t07_rounded_a18_cigar_shaped",
              "t07_rounded_a16_completely_round")


def c2() -> dict:
    ans = _powered()
    sc, meta, test_ids = C2.fit_scores([*ENC, "m1v2"])
    w, pos = C2.weights(test_ids)
    r = C2.statistic(sc, meta, ans, w, pos)
    pt = {e: {f: C2.auc(sc[e][f], meta["y"][f]) for f in meta["y"] if f in sc[e]} for e in (*ENC, "m1v2")}
    allf = sorted(set.intersection(*(set(pt[e]) for e in ENC)))
    per_all = {f: {"D": (pt["a1"][f] + pt["a2"][f]) / 2 - (pt["m1"][f] + pt["m2"][f]) / 2,
                   "S": (abs(pt["m1"][f] - pt["m2"][f]) + abs(pt["a1"][f] - pt["a2"][f])) / 2,
                   **{e: pt[e][f] for e in (*ENC, "m1v2")}} for f in allf}
    a_spread = float(np.mean([abs(pt["a1"][f] - pt["a2"][f]) for f in ans]))
    m_spread = float(np.mean([abs(pt["m1"][f] - pt["m2"][f]) for f in ans]))
    return _write("c2", {"n_test": len(test_ids), "powered": ans, "family": {k: r[k] for k in ("state", "D_bar", "S_bar", "ci")},
                         "per_answer_powered": {f: {"state_B_j (hashed bar)": v["state_shrunk_bar"], "D": v["D"],
                                                    "S": v["S"], "B": (v["S"] + r["S_bar"]) / 2, "ci": v["ci"],
                                                    "state_own_S_j": v["state"], "state_pooled_S_bar": v["state_pooled_bar"]}
                                                for f, v in r["per_answer"].items()},
                         "aligned_seed_spread_mean_abs": a_spread, "M_seed_spread_mean_abs": m_spread,
                         "per_answer_all (D_j, exploratory beyond powered)": per_all,
                         "elongation (exploratory)": {f: per_all.get(f) for f in ELONGATION},
                         "m1_on_probe_v2 (reported)": {"mean_auc_powered": float(np.mean([pt["m1v2"][f] for f in ans])),
                                                       "mean_auc_powered_m1_v1": float(np.mean([pt["m1"][f] for f in ans]))}})


# ── criterion 3 ──────────────────────────────────────────────────────────────────────────────────────

def flag_fraction(run: str) -> dict:
    f = np.load(REPO / run / "sae/flags.npz")
    st = np.load(REPO / run / "sae/part4b_stats.npz")
    if st["abar"].shape[0] != 4_999:
        raise SystemExit(f"{run}: part4b_stats on {st['abar'].shape[0]} galaxies, expected the 4,999 panel")
    live = f["density"] >= LIVE
    fl = f["flagged"]
    off = fl & (st["offset_rho"] >= f["best_nuisance"] - 1e-12)
    return {"live": int(live.sum()), "flagged": float(fl[live].mean()), "flag": float(off[live].mean()),
            "other_flagged": float((fl & ~off)[live].mean())}


def c3() -> dict:
    d = setup13()
    d["r_flux"] = C13._r_flux(d)
    fam = C13.families(d)
    reps, flags = {}, {}
    for t in ENC:
        reps[t] = C13.representation(C13._load(t, d["test"]), fam)
        flags[t] = flag_fraction(DD_RUN[t])
        reps[t]["flag"] = flags[t]["flag"]
    states = C13.c3_states((reps["a1"], reps["a2"]), (reps["m1"], reps["m2"]))
    floors = {q: C13.c3_floor((reps["m1"][q], reps["m2"][q])) for q in ("N", "Mo", "PR", "flag")}
    slim = {t: {k: v for k, v in r.items() if k != "pcs"} |
            {"pcs": [(p["pc"], round(p["share"], 4), p["top_family"], p["tracks"], {k: round(v, 3) for k, v in p["rho"].items()})
                     for p in r["pcs"]]} for t, r in reps.items()}
    return _write("c3", {"n_test": len(d["test"]), "states": states, "floors": floors, "encoders": slim, "flags": flags})


if __name__ == "__main__":
    fn = {"powered": powered, "embed-m1v2": embed_m1v2, "c1a": c1a, "c2": c2, "c3": c3}[sys.argv[1]]
    print(json.dumps(fn(), indent=1, default=float)[:20_000])
