"""Brief R — nonlinear structure and concept geometry, on M's 4-epoch checkpoint.

Descriptive, not gating: nothing here changes a rung. Stages run in a fixed order, because the
later ones read what the earlier ones decided:

  bank  two further untrained seeds, extracted once and banked (seed 0 is O1's bank)
  r0    Brief P's matched verdicts re-adjudicated under O1's pre-registered retention rule — the
        ladder judged "survived matching" against the effect floor, so 18 of 19 "confounded by
        size" features failed by arithmetic. R0 defines "confounded" for everything below.
  r3    circular structure (orientation) — run BEFORE R2's verdicts are read, because it is the
        instrument's validation: a "straight" verdict means nothing unless the same statistics
        find a circle that is certainly there
  r1    MLP headroom on all 37, both populations, with the cluster control task
  r2    centroid geometry of E[z | vote fraction]: straight, curved, or tangled

Each stage writes its own record under artifacts/out/ and is skipped if the record exists
(``--force STAGE`` to redo one). Investigation code: terse, excluded from lint/CI.

    uv run python artifacts/r_nonlinear.py --plan
    uv run python artifacts/r_nonlinear.py --stages bank
    uv run python artifacts/r_nonlinear.py --stages r0 r3 --features 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from j4_spread_controls import OUT, _release, prepare  # noqa: E402

from galaxy_jepa.models.vit import load_frozen_encoder  # noqa: E402
from galaxy_jepa.probing import controls as ctl  # noqa: E402
from galaxy_jepa.probing import geometry as geo  # noqa: E402
from galaxy_jepa.probing import ladder as ladder_mod  # noqa: E402
from galaxy_jepa.probing import matching as match  # noqa: E402
from galaxy_jepa.probing import mlp as mlp_mod  # noqa: E402
from galaxy_jepa.probing import nulls as nz  # noqa: E402
from galaxy_jepa.probing.extract import (  # noqa: E402
    EmbeddingMatrix,
    feature_embeddings,
    feature_ids,
)
from galaxy_jepa.probing.logistic import (  # noqa: E402
    Embeddings,
    _fit,
    paired_auc_bootstrap,
    probe_auc_ci_se,
    probe_scores,
    weighted_auc,
)
from galaxy_jepa.probing.orientation import second_moment_orientation  # noqa: E402

MAX_TRAIN = 40_000
O1_BANK = OUT / "o1_embeddings.npz"  # M real + untrained seed 0, co-indexed over the union
SEED_BANK = OUT / "r_untrained_seeds.npz"
EXTRA_SEEDS = (1, 2)  # with O1's seed 0: three untrained draws, reported as a RANGE, never a spread
STAGES = ("bank", "r0", "r3", "r1", "r2")
POPULATIONS = ("full", "conditional")
CLUSTER_COUNTS = (100, 1_000)  # declared, an order of magnitude apart; a C-flip is its own state
N_BOOT = 2_000
R2_EDGES = np.linspace(0.0, 1.0, 11)
R3_EDGES = np.linspace(0.0, 180.0, 13)
R3_MAX_AB = 0.6  # photometric axis ratio: orientation is undefined for round objects
R3_MAX_Q = 0.8  # moment axis ratio: a contaminated stamp's moments disagree with the fit


def _key(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def load_matrices(setup, frozen) -> tuple[EmbeddingMatrix, list[EmbeddingMatrix]]:
    """M's real matrix and the three untrained ones, all co-indexed over P2's union.

    Refuses rather than realigns: a bank built on another checkpoint, another architecture or
    another union would put two different populations into one comparison.
    """
    blob = np.load(O1_BANK, allow_pickle=False)
    if str(blob["checkpoint"]) != str(setup.ckpt):
        raise SystemExit(f"R: {O1_BANK.name} is for {blob['checkpoint']}, not {setup.ckpt}")
    ids = blob["ids"]
    if not np.array_equal(ids, np.asarray(setup.union, dtype=ids.dtype)):
        raise SystemExit("R: O1's bank is not over P2's union — refusing to realign silently")
    real = EmbeddingMatrix(ids, blob["real"], str(blob["encoder_name"]))
    untrained = [EmbeddingMatrix(ids, blob["untrained"], "untrained-s0")]

    seeds = np.load(SEED_BANK, allow_pickle=False)
    if str(seeds["model_key"]) != _key(dict(frozen.config)):
        raise SystemExit("R: the seed bank was built for a different architecture")
    if not np.array_equal(seeds["ids"], ids):
        raise SystemExit("R: the seed bank is not co-indexed with O1's")
    for s in EXTRA_SEEDS:
        untrained.append(EmbeddingMatrix(ids, seeds[f"seed{s}"], f"untrained-s{s}"))
    return real, untrained


def stage_bank(setup, frozen) -> None:
    """Extract untrained seeds 1 and 2. Architecture-determined, so reusable by any M-shaped run."""
    have = dict(np.load(SEED_BANK, allow_pickle=False)) if SEED_BANK.exists() else {}
    ids = np.load(O1_BANK, allow_pickle=False)["ids"]
    out = {"ids": ids, "model_key": _key(dict(frozen.config))}
    for s in EXTRA_SEEDS:
        if f"seed{s}" in have:
            out[f"seed{s}"] = have[f"seed{s}"]
            print(f"  seed {s}: banked", file=sys.stderr)
            continue
        t0 = time.perf_counter()
        _release(setup.device)
        mat = ctl.untrained_encoder_matrix(frozen.config, setup.ds, device=setup.device, seed=s)
        if not np.array_equal(mat.object_ids, ids):
            raise SystemExit(f"R: untrained seed {s} is not co-indexed with O1's bank")
        out[f"seed{s}"] = mat.x
        np.savez(SEED_BANK, **out)  # after EACH seed — an interruption costs one extraction
        print(f"  seed {s}: {time.perf_counter() - t0:.0f}s", file=sys.stderr)
        del mat
        _release(setup.device)


# ---------------------------------------------------------------------------------- shared

class Ctx:
    """Everything the stages share, loaded once."""

    def __init__(self, setup, frozen, n_features: int, dry: bool, only=()) -> None:
        self.setup, self.frozen = setup, frozen
        self.pc = setup.pc
        self.labels = {"full": setup.labels,
                       "conditional": setup.labels.with_population("conditional")}
        feats = list(setup.labels.features)
        self.features = feats[:n_features] if n_features else feats
        if only:
            unknown = sorted(set(only) - set(feats))
            if unknown:
                raise SystemExit(f"R: --only names unknown features {unknown}")
            self.features = [f for f in feats if f in set(only)]
        self.suffix = "_dry" if dry else ""
        self.real, self.untrained = load_matrices(setup, frozen)
        self._z: dict[str, EmbeddingMatrix] = {}
        self.selection: dict[tuple[str, str], tuple] = {}

    def record(self, stage: str) -> Path:
        return OUT / f"r_{stage}{self.suffix}.json"

    def z(self, m: EmbeddingMatrix) -> EmbeddingMatrix:
        """``m`` z-scored per dimension over the whole union — label-free, cached."""
        if m.encoder_name not in self._z:
            self._z[m.encoder_name] = EmbeddingMatrix(m.object_ids, geo.standardise(m.x),
                                                      m.encoder_name)
        return self._z[m.encoder_name]


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _save(path: Path, rec: dict) -> None:
    path.write_text(json.dumps(rec, indent=1, default=float))


def _two_class(*ys: np.ndarray) -> bool:
    return all(len(np.unique(y)) > 1 for y in ys)


def _auc(y, s) -> float:
    return weighted_auc(np.asarray(y), np.asarray(s))


def _matched(ctx: Ctx, pop: str, f: str, matrix: EmbeddingMatrix):
    """The rows R0's re-probe uses, for ``matrix``: nuisance-valid, then stratified-matched.

    The worst nuisance is found on M's real embeddings (as the ladder does) and the selection
    depends only on nuisance values, labels and the seed — so every matrix, and every probe, is
    scored on exactly the same galaxies.
    """
    lab, s = ctx.labels[pop], ctx.setup
    if (pop, f) not in ctx.selection:  # M-only, so fitted once and applied to every matrix
        tr = feature_embeddings(ctx.real, lab, f, s.train_ids)
        te = feature_embeddings(ctx.real, lab, f, s.test_ids)
        el_tr = feature_ids(ctx.real, lab, f, s.train_ids)
        el_te = feature_ids(ctx.real, lab, f, s.test_ids)
        panel = ctl.nuisance_panel(tr, te, lab, el_tr, el_te, c=ctx.pc.c)
        rows = ladder_mod.matched_rows(panel, ctx.real, lab, f, s.train_ids, s.test_ids)
        rtr, rte = rows.embeddings(ctx.real, lab, f, s.train_ids, s.test_ids)
        ctx.selection[(pop, f)] = (rows, *match.matched_indices(
            rows.values_train, rtr.y, rows.values_test, rte.y, n_strata=5, seed=ctx.pc.seed))
    rows, i_tr, i_te = ctx.selection[(pop, f)]
    mtr, mte = rows.embeddings(matrix, lab, f, s.train_ids, s.test_ids)
    sub = lambda e, i: Embeddings(e.x[i], e.y[i], e.fraction[i])  # noqa: E731
    return rows.nuisance, sub(mtr, i_tr), sub(mte, i_te)


# ---------------------------------------------------------------------------------- R0

def stage_r0(ctx: Ctx) -> None:
    """Brief P's matched verdicts, re-adjudicated under O1's pre-registered retention rule."""
    path = ctx.record("r0")
    rec = _load(path)
    p2 = json.loads((OUT / "p2_ladder.json").read_text())
    p2_rows = {pop: {r["feature"]: r for r in p2[pop]} for pop in POPULATIONS}
    floor = ctx.pc.effect_floor
    for pop in POPULATIONS:
        lab, s = ctx.labels[pop], ctx.setup
        for f in ctx.features:
            key = f"{pop}:{f}"
            if key in rec:
                continue
            t0 = time.perf_counter()
            tr = feature_embeddings(ctx.real, lab, f, s.train_ids)
            te = feature_embeddings(ctx.real, lab, f, s.test_ids)
            if not _two_class(tr.y, te.y):
                rec[key] = {"population": pop, "feature": f, "verdict": "DEGENERATE"}
                _save(path, rec)
                continue
            a, a_lo, a_hi, _ = probe_auc_ci_se(tr, te, c=ctx.pc.c, n_boot=N_BOOT, seed=ctx.pc.seed)
            c_seeds = []
            for u in ctx.untrained:
                utr = feature_embeddings(u, lab, f, s.train_ids)
                ute = feature_embeddings(u, lab, f, s.test_ids)
                c_seeds.append(float(ctl._safe_auc(utr, ute, c=ctx.pc.c)))
            nuisance, mtr, mte = _matched(ctx, pop, f, ctx.real)
            m = m_lo = m_hi = None
            cm_seeds: list[float] = []
            if mtr.y.size and mte.y.size and _two_class(mtr.y, mte.y):
                m, m_lo, m_hi, _ = probe_auc_ci_se(mtr, mte, c=ctx.pc.c, n_boot=N_BOOT,
                                                   seed=ctx.pc.seed)
                for u in ctx.untrained:
                    _, utr, ute = _matched(ctx, pop, f, u)
                    cm_seeds.append(float(ctl._safe_auc(utr, ute, c=ctx.pc.c)))
            c = float(np.mean(c_seeds))
            c_m = float(np.mean(cm_seeds)) if cm_seeds else None
            v = match.retention_verdict(a, c, m, m_lo, c_m, n_matched_test=int(mte.y.size),
                                        n_test=int(te.y.size))
            prior = p2_rows[pop].get(f, {})
            rec[key] = {
                "population": pop, "feature": f, "nuisance": nuisance,
                "A": a, "A_lo": a_lo, "A_hi": a_hi, "C": c, "C_seeds": c_seeds,
                "M": m, "M_lo": m_lo, "M_hi": m_hi, "C_m": c_m, "C_m_seeds": cm_seeds,
                "n_matched_test": int(mte.y.size), "n_test": int(te.y.size),
                "verdict": v.verdict, "retained": v.retained,
                "p2_mechanism": prior.get("mechanism"), "p2_matched_auc": prior.get("matched_auc"),
                "below_floor_unmatched": bool(a < floor),
            }
            _save(path, rec)
            # Faithfulness: the real matched AUC must reproduce P2's to the digit — same rows,
            # same seed, same fit. A mismatch means R0 is judging a different re-probe.
            pm = prior.get("matched_auc")
            flag = "" if pm is None or m is None or abs(pm - m) < 1e-9 else "  MISMATCH vs P2"
            ms = "  -   " if m is None else f"{m:.4f}"
            print(f"  r0 {pop:<11s} {f:<44s} {v.verdict:<10s} A {a:.4f} M {ms} "
                  f"({nuisance}) {time.perf_counter() - t0:5.0f}s{flag}", file=sys.stderr)
    _summarise_r0(rec)


def _summarise_r0(rec: dict) -> None:
    full = [r for r in rec.values() if r["population"] == "full" and "A" in r]
    p2_size = [r for r in full if (r.get("p2_mechanism") or "").startswith("confounded")]
    counts: dict[str, int] = {}
    for r in p2_size:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    below = sum(r["below_floor_unmatched"] for r in p2_size)
    print(f"\nR0 P2's {len(p2_size)} 'confounded' (full): {below} below the floor unmatched; "
          f"under retention: {dict(sorted(counts.items()))}", file=sys.stderr)


def confounded(ctx: Ctx) -> list[str]:
    """R0's PARTIAL or COLLAPSES features, full population — what amendments 5 and 6 act on."""
    rec = _load(ctx.record("r0"))
    return [r["feature"] for k, r in rec.items()
            if r["population"] == "full" and r["verdict"] in (match.PARTIAL, match.COLLAPSES)
            and r["feature"] in ctx.features]


# ---------------------------------------------------------------------------------- R3

def _axis_ratio(ctx: Ctx) -> np.ndarray:
    """``expAB_r`` over the union, streamed from the probe catalogue one row at a time.

    The compact probe-column sidecar ``prepare`` reads does not carry it; the catalogue does.
    """
    import csv

    csv.field_size_limit(1 << 24)
    ids = ctx.real.object_ids
    pos = {int(o): i for i, o in enumerate(ids)}
    ab = np.full(ids.size, np.nan)
    with open(Path(ctx.setup.cfg.paths.probe_dir) / "metadata.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            i = pos.get(int(row["object_id"]))
            if i is not None and row.get("expAB_r"):
                ab[i] = float(row["expAB_r"])
    return ab


def _orientations(ctx: Ctx) -> dict[str, np.ndarray]:
    """θ and q for every union galaxy, from the cached stamps. Banked: ~10 min to compute."""
    path = OUT / "r3_orientation.npz"
    ids = ctx.real.object_ids
    if path.exists():
        blob = np.load(path, allow_pickle=False)
        if np.array_equal(blob["ids"], ids):
            return {k: blob[k] for k in blob.files}
    s = ctx.setup
    theta = np.full(ids.size, np.nan)
    q = np.full(ids.size, np.nan)
    t0 = time.perf_counter()
    for i in range(len(s.ds)):
        item = s.ds[i]
        if int(item["object_id"]) != int(ids[i]):
            raise SystemExit("R3: the dataset order is not the matrix order")
        radius = 1.5 * float(item["petro_rad_arcsec"]) / float(item["pixel_scale"])
        if not np.isfinite(radius):
            continue
        theta[i], q[i] = second_moment_orientation(item["image"].float().numpy(),
                                                   radius_px=float(np.clip(radius, 4.0, 120.0)))
        if i % 10_000 == 0:
            print(f"  r3 moments {i:>6d}/{len(s.ds)}  {time.perf_counter() - t0:5.0f}s",
                  file=sys.stderr)
    if not np.array_equal(np.asarray(s.union), ids):
        raise SystemExit("R3: the dataset order is not the matrix order")
    ab = _axis_ratio(ctx)
    np.savez(path, ids=ids, theta=theta, q=q, ab=ab)
    return {"ids": ids, "theta": theta, "q": q, "ab": ab}


def stage_r3(ctx: Ctx) -> None:
    """Orientation as a circle — the instrument's validation, and a D10 nuisance measurement."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    o = _orientations(ctx)
    gate = np.isfinite(o["theta"]) & (o["ab"] <= R3_MAX_AB) & (o["q"] <= R3_MAX_Q)
    idx = np.nonzero(gate)[0]
    theta = o["theta"][idx]
    train_set = set(int(i) for i in ctx.setup.train_ids)
    is_train = np.array([int(ctx.real.object_ids[i]) in train_set for i in idx])
    target = np.stack([np.cos(np.radians(2 * theta)), np.sin(np.radians(2 * theta))], axis=1)
    print(f"R3 gated      : {idx.size:,} of {gate.size:,} (expAB_r <= {R3_MAX_AB}, moment q <= "
          f"{R3_MAX_Q}; expAB_r finite on {np.isfinite(o['ab']).mean():.1%})", file=sys.stderr)
    out: dict[str, object] = {"n_gated": int(idx.size), "n_union": int(gate.size),
                              "max_ab": R3_MAX_AB, "max_q": R3_MAX_Q,
                              "ab_finite": float(np.isfinite(o["ab"]).mean()), "matrices": {}}
    for m in [ctx.real, *ctx.untrained]:
        x = ctx.z(m).x[idx]
        stats = geo.path_statistics(x, theta, R3_EDGES, seed=ctx.pc.seed)
        if stats is None:
            out["matrices"][m.encoder_name] = {"characterised": False}
            continue
        test = geo.curvature_test(x, theta, R3_EDGES, stats, top_up_below=0.0,
                                  seed=ctx.pc.seed)
        curved = test.p_curved <= 0.05 and (1 - stats.straightness) >= geo.CURVED_MIN_BEND
        circle = geo.circle_test(stats, curved=curved, exists=test.exists)
        ridge = Ridge(alpha=1.0).fit(x[is_train], target[is_train])
        r2 = float(r2_score(target[~is_train], ridge.predict(x[~is_train])))
        out["matrices"][m.encoder_name] = {
            "characterised": True, "total": stats.total, "straightness": stats.straightness,
            "effective_dim": stats.effective_dim, "monotonicity": stats.monotonicity,
            "exists": test.exists, "p_curved": test.p_curved, "curved": bool(curved),
            "circle_recovered": circle.recovered, "circle_reason": circle.reason,
            "winding_turns": circle.winding_turns, "violations": circle.winding_violations,
            "closes": circle.closes, "ridge_r2_doubled_angle": r2,
            "occupancy": stats.occupancy.tolist(),
        }
        print(f"  r3 {m.encoder_name:<14s} circle={circle.recovered!s:<5s} {circle.reason}  "
              f"ridge R2 {r2:.3f}", file=sys.stderr)
    out["instrument_validated"] = bool(
        out["matrices"].get(ctx.real.encoder_name, {}).get("circle_recovered", False))
    _save(ctx.record("r3"), out)
    print(f"R3 instrument : {'VALIDATED' if out['instrument_validated'] else 'NOT VALIDATED'} "
          f"on M", file=sys.stderr)


# ---------------------------------------------------------------------------------- R1

def _mlp_kw(pc) -> dict:
    return dict(depth=pc.mlp_depth, weight_decay=pc.mlp_weight_decay, epochs=pc.mlp_epochs,
                lr=pc.mlp_lr, seed=pc.seed)


def _clusters(ctx: Ctx) -> dict[int, np.ndarray]:
    """k-means on the 40,000 TRAIN embeddings only; every union galaxy takes a train centroid."""
    path = OUT / "r1_clusters.npz"
    ids = ctx.real.object_ids
    if path.exists():
        blob = np.load(path, allow_pickle=False)
        if np.array_equal(blob["ids"], ids):
            return {c: blob[f"c{c}"] for c in CLUSTER_COUNTS}
    train_rows = ctx.real.rows_for(ctx.setup.train_ids)
    out = {}
    for c in CLUSTER_COUNTS:
        t0 = time.perf_counter()
        out[c] = ctl.cluster_assignment(ctx.real.x[train_rows], ctx.real.x, n_clusters=c,
                                        seed=ctx.pc.seed)
        print(f"  r1 k-means C={c}: {time.perf_counter() - t0:.0f}s", file=sys.stderr)
    np.savez(path, ids=ids, **{f"c{c}": v for c, v in out.items()})
    return out


def _three_state(headroom_lo: float, sel: dict[int, tuple | None]) -> str:
    if not headroom_lo > 0:
        return "NO EVIDENCE"
    passing = [c for c, v in sel.items() if v is not None and v[1] > 0]
    if len(passing) == len(CLUSTER_COUNTS):
        return "EVIDENCE"
    if len(passing) == 1:
        return "SENSITIVE TO C"
    return "NO EVIDENCE"


def stage_r1(ctx: Ctx) -> None:
    """MLP headroom on all 37, both populations, with its guards; then untrained and matched."""
    path = ctx.record("r1")
    rec = _load(path)
    pc, s = ctx.pc, ctx.setup
    kw = _mlp_kw(pc)
    clusters = _clusters(ctx)
    for pop in POPULATIONS:
        lab = ctx.labels[pop]
        for f in ctx.features:
            key = f"{pop}:{f}"
            if key in rec:
                continue
            t0 = time.perf_counter()
            tr = feature_embeddings(ctx.real, lab, f, s.train_ids)
            te = feature_embeddings(ctx.real, lab, f, s.test_ids)
            if not _two_class(tr.y, te.y):
                rec[key] = {"population": pop, "feature": f, "verdict": "DEGENERATE"}
                _save(path, rec)
                continue
            lin = probe_scores(tr, te, c=pc.c)
            width, val = mlp_mod.select_width(tr, widths=tuple(pc.mlp_widths), **kw)
            real = mlp_mod.mlp_fit(tr, te, width=width, **kw)
            head = paired_auc_bootstrap([te.y, te.y], [real.test_scores, lin], [1.0, -1.0],
                                        n_boot=N_BOOT, seed=pc.seed)
            # Hewitt–Liang as the ladder builds it: permuted train labels. Blind to memorisation
            # on non-recurring galaxies, reported because asked; its TRAIN AUC is the real measure.
            rng = np.random.default_rng(pc.seed)
            perm = mlp_mod.mlp_fit(Embeddings(tr.x, rng.permutation(tr.y), tr.fraction), te,
                                   width=width, **kw)
            sel: dict[int, tuple | None] = {}
            ctrl_aucs: dict[int, dict] = {}
            tr_rows = ctx.real.rows_for(feature_ids(ctx.real, lab, f, s.train_ids))
            te_rows = ctx.real.rows_for(feature_ids(ctx.real, lab, f, s.test_ids))
            for c in CLUSTER_COUNTS:
                both = clusters[c][np.concatenate([tr_rows, te_rows])]
                yc = ctl.cluster_control_labels(both, base_rate=float(tr.y.mean()),
                                                seed=pc.seed + c)
                yc_tr, yc_te = yc[: tr_rows.size], yc[tr_rows.size:]
                if not _two_class(yc_tr, yc_te):
                    sel[c] = None
                    continue
                ctr = Embeddings(tr.x, yc_tr, tr.fraction)
                cte = Embeddings(te.x, yc_te, te.fraction)
                lin_c = probe_scores(ctr, cte, c=pc.c)
                mlp_c = mlp_mod.mlp_fit(ctr, cte, width=width, **kw)
                sel[c] = paired_auc_bootstrap(
                    [te.y, te.y, yc_te, yc_te], [real.test_scores, lin, mlp_c.test_scores, lin_c],
                    [1.0, -1.0, -1.0, 1.0], n_boot=N_BOOT, seed=pc.seed)
                ctrl_aucs[c] = {"mlp": _auc(yc_te, mlp_c.test_scores), "linear": _auc(yc_te, lin_c)}
            verdict = _three_state(head[1], sel)
            rec[key] = {
                "population": pop, "feature": f, "width": width, "val_auc": val,
                "linear_auc": _auc(te.y, lin), "mlp_auc": _auc(te.y, real.test_scores),
                "headroom": head[0], "headroom_lo": head[1], "headroom_hi": head[2],
                "hl_gap": _auc(te.y, real.test_scores) - _auc(te.y, perm.test_scores),
                "perm_test_auc": _auc(te.y, perm.test_scores),
                "perm_train_auc": perm.train_auc, "real_train_auc": real.train_auc,
                "selective": {str(c): (None if v is None else
                                       {"point": v[0], "lo": v[1], "hi": v[2]})
                              for c, v in sel.items()},
                "control_aucs": {str(c): v for c, v in ctrl_aucs.items()},
                "verdict": verdict, "n_train": int(tr.y.size), "n_test": int(te.y.size),
            }
            _save(path, rec)
            print(f"  r1 {pop:<11s} {f:<44s} w={width:<4d} head {head[0]:+.4f} "
                  f"[{head[1]:+.4f},{head[2]:+.4f}] {verdict:<15s} "
                  f"{time.perf_counter() - t0:5.0f}s", file=sys.stderr)

    # Untrained headroom — descriptive, out of the verdict: three seeds, a RANGE.
    for f in ctx.features:
        key = f"untrained:{f}"
        if key in rec:
            continue
        lab = ctx.labels["full"]
        vals = []
        for u in ctx.untrained:
            tr = feature_embeddings(u, lab, f, s.train_ids)
            te = feature_embeddings(u, lab, f, s.test_ids)
            if not _two_class(tr.y, te.y):
                continue
            lin = probe_scores(tr, te, c=pc.c)
            width, _ = mlp_mod.select_width(tr, widths=tuple(pc.mlp_widths), **kw)
            fit = mlp_mod.mlp_fit(tr, te, width=width, **kw)
            vals.append({"seed": u.encoder_name, "linear": _auc(te.y, lin),
                         "mlp": _auc(te.y, fit.test_scores), "width": width})
        rec[key] = {"population": "untrained", "feature": f, "seeds": vals}
        _save(path, rec)
        hr = [v["mlp"] - v["linear"] for v in vals]
        print(f"  r1 untrained {f:<44s} headroom range "
              f"[{min(hr, default=float('nan')):+.4f}, {max(hr, default=float('nan')):+.4f}]",
              file=sys.stderr)

    # Matched MLP on R0's confounded set: same rows, same test, same rule as the linear re-probe.
    r0 = _load(ctx.record("r0"))
    for f in confounded(ctx):
        key = f"matched:{f}"
        if key in rec:
            continue
        full = rec.get(f"full:{f}", {})
        unt = rec.get(f"untrained:{f}", {}).get("seeds", [])
        _, mtr, mte = _matched(ctx, "full", f, ctx.real)
        if not (mtr.y.size and mte.y.size and _two_class(mtr.y, mte.y)) or "mlp_auc" not in full:
            rec[key] = {"population": "matched", "feature": f, "verdict": match.UNRESOLVED}
            _save(path, rec)
            continue
        width, _ = mlp_mod.select_width(mtr, widths=tuple(pc.mlp_widths), **kw)
        fit = mlp_mod.mlp_fit(mtr, mte, width=width, **kw)
        m, m_lo, _ = paired_auc_bootstrap([mte.y], [fit.test_scores], [1.0], n_boot=N_BOOT,
                                          seed=pc.seed)
        cm = []
        for u in ctx.untrained:
            _, utr, ute = _matched(ctx, "full", f, u)
            if _two_class(utr.y, ute.y):
                w_u, _ = mlp_mod.select_width(utr, widths=tuple(pc.mlp_widths), **kw)
                cm.append(_auc(ute.y, mlp_mod.mlp_fit(utr, ute, width=w_u, **kw).test_scores))
        a, c = full["mlp_auc"], float(np.mean([v["mlp"] for v in unt])) if unt else None
        c_m = float(np.mean(cm)) if cm else None
        v = (match.retention_verdict(a, c, m, m_lo, c_m, n_matched_test=int(mte.y.size),
                                     n_test=int(full["n_test"]))
             if c is not None else match.RetentionVerdict(match.UNRESOLVED, None, 0.0, None))
        lin = r0.get(f"full:{f}", {})
        rec[key] = {"population": "matched", "feature": f, "A_mlp": a, "C_mlp": c, "M_mlp": m,
                    "M_mlp_lo": m_lo, "C_m_mlp": c_m, "verdict": v.verdict,
                    "retained": v.retained, "linear_verdict": lin.get("verdict"),
                    "separable_nonlinearly": bool(
                        lin.get("verdict") in (match.PARTIAL, match.COLLAPSES)
                        and v.verdict == match.SURVIVES)}
        _save(path, rec)
        print(f"  r1 matched {f:<44s} linear {lin.get('verdict')} -> MLP {v.verdict}",
              file=sys.stderr)


# ---------------------------------------------------------------------------------- R2

def _direction_z(tr: Embeddings, c: float, std_union: np.ndarray) -> np.ndarray:
    """The ladder's logistic direction, expressed in union-z-scored coordinates."""
    scaler, clf = _fit(tr, c=c)
    return std_union * clf.coef_[0] / scaler.scale_


def _r2_family(ctx: Ctx, entries: dict[str, tuple], top_up_below: float) -> dict:
    """Paths, nulls and BY-corrected verdicts for one family. ``entries``: f -> (x, values, dir)."""
    stats: dict[str, geo.PathStats | None] = {}
    tests: dict[str, geo.CurvatureTest] = {}
    for f, (x, v, d) in entries.items():
        t0 = time.perf_counter()
        st = geo.path_statistics(x, v, R2_EDGES, direction=d, seed=ctx.pc.seed)
        stats[f] = st
        if st is not None:
            tests[f] = geo.curvature_test(x, v, R2_EDGES, st, top_up_below=top_up_below,
                                          seed=ctx.pc.seed)
        print(f"  r2 {f:<44s} {'-' if st is None else f'{st.straightness:.3f}':>6s} "
              f"{time.perf_counter() - t0:5.0f}s", file=sys.stderr)
    verdicts = geo.curvature_verdicts(stats, tests, alpha=ctx.pc.alpha)
    family = [f for f in tests if tests[f].exists]
    topped = [f for f in tests if tests[f].n_draws > geo.B_FIRST]
    if family and topped:
        nz.assert_null_resolution(geo.B_TOP, alpha=ctx.pc.alpha, method="benjamini_yekutieli",
                                  n_tests=len(family))
    out = {}
    for f, st in stats.items():
        row: dict[str, object] = dict(verdicts[f])
        if st is not None:
            t = tests[f]
            row.update({"straightness": st.straightness, "curvature": st.curvature,
                        "total": st.total, "monotonicity": st.monotonicity,
                        "effective_dim": st.effective_dim, "readout": st.readout,
                        "n_bins": st.n_bins, "occupancy": st.occupancy.tolist(),
                        "exists": t.exists, "p_curved": t.p_curved, "n_draws": t.n_draws,
                        "total_null_q95": t.total_null_q95})
        out[f] = row
    return {"features": out, "family_size": len(family),
            "expected_false_curved_at_5pct": 0.05 * len(family), "topped_up": topped}


def stage_r2(ctx: Ctx) -> None:
    """E[z | vote fraction]: straight, curved or tangled — full, matched, untrained reference."""
    r3 = _load(ctx.record("r3"))
    s, pc = ctx.setup, ctx.pc
    lab = ctx.labels["full"]
    h = float(np.sum(1.0 / np.arange(1, len(ctx.features) + 1)))
    top_up_below = pc.alpha / h  # the loosest BY threshold, reached only at the highest rank
    union = list(s.union)
    out: dict[str, object] = {"instrument_validated": r3.get("instrument_validated"),
                              "top_up_below": top_up_below}
    zr = ctx.z(ctx.real)
    std_union = ctx.real.x.std(axis=0) + 1e-8

    def entries_for(matrix: EmbeddingMatrix, with_dir: bool) -> dict[str, tuple]:
        zm = ctx.z(matrix)
        ent = {}
        for f in ctx.features:
            e = feature_embeddings(zm, lab, f, union)
            d = None
            if with_dir:
                tr = feature_embeddings(ctx.real, lab, f, s.train_ids)
                if _two_class(tr.y):
                    d = _direction_z(tr, pc.c, std_union)
            ent[f] = (e.x, e.fraction, d)
        return ent

    print("R2 full population, M:", file=sys.stderr)
    out["full"] = _r2_family(ctx, entries_for(ctx.real, True), top_up_below)

    conf = confounded(ctx)
    if conf:
        print("R2 matched rows (R0's PARTIAL/COLLAPSES):", file=sys.stderr)
        ent = {}
        for f in conf:
            _, mtr, mte = _matched(ctx, "full", f, zr)
            ent[f] = (np.vstack([mtr.x, mte.x]), np.concatenate([mtr.fraction, mte.fraction]),
                      None)
        h_m = float(np.sum(1.0 / np.arange(1, len(conf) + 1)))
        out["matched"] = _r2_family(ctx, ent, pc.alpha / h_m)

    out["untrained"] = {}
    for u in ctx.untrained:
        print(f"R2 reference, {u.encoder_name}:", file=sys.stderr)
        out["untrained"][u.encoder_name] = _r2_family(ctx, entries_for(u, False), 0.0)
    _save(ctx.record("r2"), out)

    full = out["full"]["features"]
    curved = [f for f, r in full.items() if r.get("verdict") == geo.CURVED]
    print(f"\nR2 curved (BY, M, full): {len(curved)} of {out['full']['family_size']} tested; "
          f"{out['full']['expected_false_curved_at_5pct']:.1f} expected false at 5% uncorrected",
          file=sys.stderr)
    if not r3.get("instrument_validated"):
        print("R2 NOTE: R3 did not validate the instrument — 'straight' verdicts are "
              "UNINTERPRETABLE, not evidence for feature = direction.", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/m/encoder.pt")
    ap.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    ap.add_argument("--features", type=int, default=0, help="cap for a dry run; 0 = all 37")
    ap.add_argument("--force", nargs="*", default=[], choices=STAGES)
    ap.add_argument("--only", nargs="*", default=[], help="named features; a dry run")
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()

    setup = prepare(args.checkpoint, MAX_TRAIN, label="R", sources=1)
    frozen = load_frozen_encoder(setup.ckpt)
    print(f"R stages      : {' '.join(args.stages)}", file=sys.stderr)
    print(f"R O1 bank     : {O1_BANK.name} ({'present' if O1_BANK.exists() else 'MISSING'})",
          file=sys.stderr)
    print(f"R seed bank   : {SEED_BANK.name} ({'present' if SEED_BANK.exists() else 'to build'})",
          file=sys.stderr)
    if args.plan:
        print("\nR --plan: nothing run", file=sys.stderr)
        return

    if "bank" in args.stages:
        stage_bank(setup, frozen)
    rest = [st for st in args.stages if st != "bank"]
    if not rest:
        return
    dry = bool(args.features or args.only)
    ctx = Ctx(setup, frozen, args.features, dry=dry, only=args.only)
    print(f"R features    : {len(ctx.features)}{' (DRY RUN, *_dry.json)' if dry else ''}",
          file=sys.stderr)
    runners = {"r0": stage_r0, "r3": stage_r3, "r1": stage_r1, "r2": stage_r2}
    for st in ("r0", "r3", "r1", "r2"):  # fixed order: R0 defines "confounded", R3 validates R2
        if st not in rest:
            continue
        if st in args.force and ctx.record(st).exists():
            ctx.record(st).unlink()
        t0 = time.perf_counter()
        print(f"\n=== {st} ===", file=sys.stderr)
        runners[st](ctx)
        print(f"=== {st} done in {time.perf_counter() - t0:.0f}s ===", file=sys.stderr)


if __name__ == "__main__":
    main()
