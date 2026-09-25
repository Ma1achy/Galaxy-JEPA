"""Brief DD, Part 1: occlusion maps on M and their tool-validation criteria V1–V4. TOOL VALIDATION ONLY.

Criteria, states and plants are pre-registered and hashed in artifacts/interp_tooling.md
("Part 1 pre-registration"). Every map comes from `dd_core.Occluder`; every plant goes through the same
scoring function as the real test.

  uv run python artifacts/dd_part1.py maps        # M on the occlusion sample (drop/noise × 1/2×2)
  uv run python artifacts/dd_part1.py cascade     # V1: M's maps at every randomisation level + plant encoder
  uv run python artifacts/dd_part1.py v2          # V2: g shifted by 1 px, and the joint-shift control
  uv run python artifacts/dd_part1.py v3          # V3: M, untrained on the hashed GZ3D list
  uv run python artifacts/dd_part1.py v4          # V4: rotations and flip
  uv run python artifacts/dd_part1.py plants      # D28: every plant, before the hash
  uv run python artifacts/dd_part1.py score       # the verdicts
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
import dd_core as D  # noqa: E402

P1 = D.LOCAL / "part1"
SEED = 20260925
V1_ANSWERS = ("t03_bar_a06_bar", "t04_spiral_a08_spiral", "t02_edgeon_a04_yes")
V1_BAR, V1_PLANT_SEED = 0.3, 2
V2_N, V2_TOP, ALPHA = 50, 0.10, 0.01
V2_MIN_DELTA = 0.02  # mass units; significance without this magnitude reads WEAK
V3_MIN_N, V3_MIN_DAUC = 30, 0.02
V3_ANSWER = {"bar": "t03_bar_a06_bar", "spiral": "t04_spiral_a08_spiral"}
V4_N = 100
V4_OPS = {"rot90": lambda a: np.rot90(a, 1, axes=(-2, -1)),
          "rot180": lambda a: np.rot90(a, 2, axes=(-2, -1)),
          "flip": lambda a: a[..., ::-1]}
CASCADE_LEVELS = tuple(range(0, 14))  # 0 = M; 1 = block 12 (unread at block 11); 13 = + patch embedding
MODES = (("drop", 1), ("drop", 2), ("noise", 1), ("noise", 2))
N_SECONDARY = 400  # the secondary modes (2×2 drop, noise fill) run on a seeded subset: GPU budget


# ── shared setup ─────────────────────────────────────────────────────────────────────────────────

class Ctx:
    """Encoders, read-outs and samples, built once per process."""

    def __init__(self) -> None:
        import r_nonlinear as R
        from j4_spread_controls import prepare
        self.setup = prepare(None, R.MAX_TRAIN, label="DD1", sources=1)
        self.m = D.m_encoder()
        self.ids, self.st = D.stamps("occl")
        self.groups = json.loads((D.LOCAL / "occl_groups.json").read_text())
        self.bank = np.load(D.O1_BANK, allow_pickle=False)
        self.pos = {int(o): i for i, o in enumerate(self.bank["ids"])}
        occl_rows = [self.pos[int(i)] for i in self.ids]
        self.probes = D.probe_readout(self.setup, "real")
        self.band = D.band_offset_readout(self.setup)
        for ro in (self.probes, self.band):
            ro.calibrate(self.bank["real"][occl_rows].astype(np.float64))
        self.names = [*self.probes.names, "band_offset_pc1", "norm"]
        self.__untrained = None
        self.occl_rows = occl_rows

    def untrained(self):
        """The seed-0 encoder with its own 37 probes, calibrated on its own bank rows."""
        if self.__untrained is None:
            enc = D.untrained_encoder(self.m.config, seed=0)
            ro = D.probe_readout(self.setup, "untrained")
            ro.calibrate(self.bank["untrained"][self.occl_rows].astype(np.float64))
            self.__untrained = (enc, ro)
        return self.__untrained

    def members(self, answer: str) -> list[int]:
        """Row indices (in the occl sample) of the answer's stratified positives and negatives."""
        return [k for k, o in enumerate(self.ids) if self.groups[str(int(o))].rsplit(":", 1)[0] == answer]


def _stack(maps: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    return np.stack([maps[n] for n in names]).astype(np.float32)


def _run(occ, imgs, names, mode="drop", scale=1, label=""):
    out, t0 = [], time.time()
    for k, img in enumerate(imgs):
        out.append(_stack(occ.maps(np.asarray(img, np.float32), mode, scale, seed=SEED + k), names))
        if (k + 1) % 200 == 0:
            print(f"  {label} {k + 1}/{len(imgs)}  {time.time() - t0:.0f}s", flush=True)
    return np.stack(out)


# ── compute stages ───────────────────────────────────────────────────────────────────────────────

def maps(ctx: Ctx) -> None:
    """Primary (token drop, single patch) on all 2,000; the secondary modes on N_SECONDARY rows."""
    occ = D.Occluder(ctx.m, [ctx.probes, ctx.band])
    sec = sorted(int(k) for k in np.random.default_rng(SEED + 1).choice(len(ctx.ids), N_SECONDARY, replace=False))
    np.save(P1 / "secondary_rows.npy", np.asarray(sec))
    for mode, scale in MODES:
        f = P1 / f"m_{mode}{scale}.npy"
        if not f.exists():
            imgs = ctx.st if (mode, scale) == ("drop", 1) else [ctx.st[k] for k in sec]
            np.save(f, _run(occ, imgs, ctx.names, mode, scale, f"M {mode}{scale}"))


def _cascade_readout(ctx: Ctx, enc, rows: list[int]) -> list:
    """M's probe directions and band axis, re-calibrated (SD only) on this encoder's pooled
    embeddings of the V1 galaxies — maps are compared by rank, so the SD is cosmetic."""
    x = D.base_pooled(enc, ctx.st[rows])
    out = []
    for ro in (ctx.probes, ctx.band):
        r2 = D.Readout(ro.names, ro.w, ro.b)
        r2.calibrate(x)
        out.append(r2)
    return out


def cascade(ctx: Ctx) -> None:
    rows = sorted({k for a in V1_ANSWERS for k in ctx.members(a)})
    np.save(P1 / "v1_rows.npy", np.asarray(rows))
    imgs = [ctx.st[k] for k in rows]
    for level in CASCADE_LEVELS:
        f = P1 / f"cascade_L{level:02d}.npy"
        if f.exists():
            continue
        enc = D.cascade(ctx.m, level, seed=1)
        np.save(f, _run(D.Occluder(enc, _cascade_readout(ctx, enc, rows)), imgs, ctx.names, label=f"L{level}"))
    f = P1 / "cascade_plant_seed2.npy"
    if not f.exists():
        enc = D.cascade(ctx.m, 13, seed=V1_PLANT_SEED)
        np.save(f, _run(D.Occluder(enc, _cascade_readout(ctx, enc, rows)), imgs, ctx.names, label="plant"))


def _shift_g(img: np.ndarray, bands=(0,)) -> np.ndarray:
    """Integer 1-px shift (+column) of the named bands: exact, no resampling; the wrapped column is
    replaced by the original first column."""
    out = np.array(img, np.float32, copy=True)
    for b in bands:
        out[b] = np.roll(out[b], 1, axis=-1)
        out[b][:, 0] = img[b][:, 0]
    return out


def v2(ctx: Ctx) -> None:
    rng = np.random.default_rng(SEED)
    rows = sorted(int(k) for k in rng.choice(len(ctx.ids), V2_N, replace=False))
    np.save(P1 / "v2_rows.npy", np.asarray(rows))
    occ = D.Occluder(ctx.m, [ctx.probes, ctx.band])
    for tag, bands in (("gshift", (0,)), ("allshift", (0, 1, 2))):
        f = P1 / f"v2_{tag}.npy"
        if not f.exists():
            np.save(f, _run(occ, [_shift_g(ctx.st[k], bands) for k in rows], ctx.names, label=tag))


def v3(ctx: Ctx) -> None:
    import dd_v3
    lst = json.loads((D.LOCAL / "v3_list.json").read_text())["lists"]
    ids = sorted(set().union(*lst.values()))
    np.save(P1 / "v3_ids.npy", np.asarray(ids))
    st = dd_v3._stamps_for(ids)
    enc_u, ro_u = ctx.untrained()
    runs = {"m_drop1": (ctx.m, [ctx.probes, ctx.band], ctx.names, "drop", 1),
            "m_drop2": (ctx.m, [ctx.probes, ctx.band], ctx.names, "drop", 2),
            "m_noise1": (ctx.m, [ctx.probes, ctx.band], ctx.names, "noise", 1),
            "u_drop1": (enc_u, [ro_u], [*ro_u.names, "norm"], "drop", 1)}
    for tag, (enc, ros, names, mode, scale) in runs.items():
        f = P1 / f"v3_{tag}.npy"
        if not f.exists():
            np.save(f, _run(D.Occluder(enc, ros), st, names, mode, scale, f"v3 {tag}"))
    (P1 / "v3_names.json").write_text(json.dumps({"m": ctx.names, "u": [*ro_u.names, "norm"]}))


def v4(ctx: Ctx) -> None:
    rng = np.random.default_rng(SEED + 4)
    rows = sorted(int(k) for k in rng.choice(len(ctx.ids), V4_N, replace=False))
    np.save(P1 / "v4_rows.npy", np.asarray(rows))
    occ = D.Occluder(ctx.m, [ctx.probes, ctx.band])
    for op, fn in V4_OPS.items():
        f = P1 / f"v4_{op}.npy"
        if not f.exists():
            np.save(f, _run(occ, [np.ascontiguousarray(fn(np.asarray(ctx.st[k], np.float32))) for k in rows],
                            ctx.names, label=op))


# ── statistics (the one path for real tests and plants) ─────────────────────────────────────────

def _rho(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(a.ravel(), b.ravel())[0])


def _boot_median(v: np.ndarray, n: int = 10_000, seed: int = SEED) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    b = np.median(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
    return float(np.median(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def v1_stat(maps_a: np.ndarray, maps_b: np.ndarray) -> dict:
    """Per-galaxy |Spearman| over the 256 patches; the median, its bootstrap CI, and the state
    against the 0.3 bar: PASS (median < 0.3, CI below), FRAGILE (median < 0.3, CI straddles),
    FAIL (median ≥ 0.3)."""
    r = np.abs([_rho(a, b) for a, b in zip(maps_a, maps_b, strict=True)])  # a random map's sign is arbitrary
    med, lo, hi = _boot_median(r)
    state = "FAIL" if med >= V1_BAR else ("FRAGILE" if hi >= V1_BAR else "PASS")
    return {"median": med, "ci": [lo, hi], "n": len(r), "state": state}


def bright_edge(img: np.ndarray) -> np.ndarray:
    """The top 10% of patches (26 of 256) by summed r-band gradient magnitude of the unshifted stamp:
    the edges of bright sources (galaxy and stars alike)."""
    gy, gx = np.gradient(np.asarray(img[1], np.float64))
    g = np.hypot(gx, gy).reshape(D.GRID, 16, D.GRID, 16).sum(axis=(1, 3))
    k = int(round(V2_TOP * g.size))
    out = np.zeros(g.size, bool)
    out[np.argsort(g.ravel())[-k:]] = True
    return out.reshape(g.shape)


def edge_mass(m: np.ndarray, be: np.ndarray) -> float:
    a = np.abs(m)
    return float(a[be].sum() / a.sum())


def v2_stat(base: np.ndarray, shifted: np.ndarray, control: np.ndarray, masks: list[np.ndarray]) -> dict:
    """Band-offset map mass on bright-edge patches, shifted vs unshifted (paired, one-sided).
    States, in precedence: FAIL (no rise at α) > CONFOUNDED (the joint shift of all three bands also
    rises at α with a median Δ ≥ half the g-shift's) > WEAK (rises, median Δ < 0.02) > PASS."""
    m0 = np.array([edge_mass(a, be) for a, be in zip(base, masks, strict=True)])
    m1 = np.array([edge_mass(a, be) for a, be in zip(shifted, masks, strict=True)])
    mc = np.array([edge_mass(a, be) for a, be in zip(control, masks, strict=True)])
    p1 = float(wilcoxon(m1 - m0, alternative="greater").pvalue)
    pc = float(wilcoxon(mc - m0, alternative="greater").pvalue)
    d1, dc = float(np.median(m1 - m0)), float(np.median(mc - m0))
    if p1 >= ALPHA or d1 <= 0:
        state = "FAIL"
    elif pc < ALPHA and dc >= 0.5 * d1:
        state = "CONFOUNDED"
    elif d1 < V2_MIN_DELTA:
        state = "WEAK"
    else:
        state = "PASS"
    return {"mass_unshifted": float(np.median(m0)), "mass_gshift": float(np.median(m1)),
            "mass_joint": float(np.median(mc)), "delta_g": d1, "p_g": p1, "delta_joint": dc, "p_joint": pc,
            "uniform_mass": V2_TOP, "state": state}


def auc(score: np.ndarray, lab: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(lab, score))


def centre_prior() -> np.ndarray:
    c = (np.arange(D.GRID) + 0.5) * 16 - 128
    return -np.hypot(*np.meshgrid(c, c))


def v3_stat(m_maps: list[np.ndarray], u_maps: list[np.ndarray], labels: list[np.ndarray],
            domains: list[np.ndarray]) -> dict:
    """Per-galaxy patch-level AUC inside the GZ3D footprint; median across galaxies. M against the
    untrained encoder and against the centre prior, each a one-sided paired Wilcoxon; BY over the 4
    V3 tests (2 structures × 2 comparisons) is applied in `score`. States per structure, in
    precedence: INSUFFICIENT (n < 30) > FAIL (either comparison not significant) > WEAK (both
    significant, either median paired ΔAUC < 0.02) > PASS."""
    n = len(labels)
    if n < V3_MIN_N:
        return {"n": n, "state": "INSUFFICIENT"}
    cp = centre_prior()
    am = np.array([auc(m[d], lab[d]) for m, lab, d in zip(m_maps, labels, domains, strict=True)])
    au = np.array([auc(u[d], lab[d]) for u, lab, d in zip(u_maps, labels, domains, strict=True)])
    ac = np.array([auc(cp[d], lab[d]) for lab, d in zip(labels, domains, strict=True)])
    out = {"n": n, "auc_m": _boot_median(am), "auc_untrained": _boot_median(au), "auc_centre": _boot_median(ac)}
    for tag, other in (("untrained", au), ("centre", ac)):
        d = am - other
        out[f"p_vs_{tag}"] = float(wilcoxon(d, alternative="greater").pvalue) if np.any(d) else 1.0
        out[f"dauc_vs_{tag}"] = float(np.median(d))
    return out


def by_adjust(p: list[float]) -> list[float]:
    """Benjamini–Yekutieli adjusted p-values."""
    p_arr = np.asarray(p, float)
    m = len(p_arr)
    cm = np.sum(1.0 / np.arange(1, m + 1))
    o = np.argsort(p_arr)
    adj = np.minimum.accumulate((p_arr[o] * m * cm / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(adj, 1.0)
    return out.tolist()


def v3_states(res: dict) -> dict:
    keys = [(s, c) for s in res if res[s].get("state") != "INSUFFICIENT" for c in ("untrained", "centre")]
    adj = by_adjust([res[s][f"p_vs_{c}"] for s, c in keys]) if keys else []
    for (s, c), q in zip(keys, adj, strict=True):
        res[s][f"q_vs_{c}"] = q
    for s, r in res.items():
        if r.get("state") == "INSUFFICIENT":
            continue
        sig = all(r[f"q_vs_{c}"] < ALPHA for c in ("untrained", "centre"))
        big = all(r[f"dauc_vs_{c}"] >= V3_MIN_DAUC for c in ("untrained", "centre"))
        r["state"] = "FAIL" if not sig else ("WEAK" if not big else "PASS")
    return res


def v3_inputs(which: str = "m_drop1", votes: int = 3, suffix: str = "") -> dict:
    """`suffix` '_f25' scores the ≥ 25%-coverage lists against their own labels."""
    """Per structure: (maps, untrained maps, labels, domains) for the listed galaxies."""
    lst = json.loads((D.LOCAL / "v3_list.json").read_text())["lists"]
    ids = [int(i) for i in np.load(P1 / "v3_ids.npy")]
    names = json.loads((P1 / "v3_names.json").read_text())
    mm, uu = np.load(P1 / f"v3_{which}.npy"), np.load(P1 / "v3_u_drop1.npy")
    lab = np.load(D.LOCAL / "v3_patch_labels.npz")
    row = {o: k for k, o in enumerate(ids)}
    out = {}
    for s, ans in V3_ANSWER.items():
        im, iu = names["m"].index(ans), names["u"].index(ans)
        gal = lst[s + suffix]
        ks = [row[o] for o in gal]
        labs = [lab[f"{o}__{s}_{votes}{suffix}"] for o in gal]
        doms = [lab[f"{o}__footprint"] for o in gal]
        keep = [j for j, (lb, d) in enumerate(zip(labs, doms, strict=True)) if lb[d].any() and (~lb[d]).any()]
        out[s] = ([mm[ks[j], im] for j in keep], [uu[ks[j], iu] for j in keep],
                  [labs[j] for j in keep], [doms[j] for j in keep])
    return out


# ── plants (D28) ─────────────────────────────────────────────────────────────────────────────────

def plants() -> dict:
    """Each criterion's plant through its own scoring function. V1: a second fully randomised network
    (seed 2) in M's place must FAIL against the seed-1 randomised maps, OR an image-only map (patch
    r-band flux) must — else V1 for that answer is UNREACHABLE. V2: an injected edge signal must PASS
    and a null (no change) must FAIL. V3: an oracle (mask fraction + noise) must PASS; the centre prior
    + noise in M's place must not PASS."""
    rng = np.random.default_rng(SEED + 28)
    out: dict = {}
    ids, st = D.stamps("occl")
    groups = json.loads((D.LOCAL / "occl_groups.json").read_text())
    names = json.loads((P1 / "names.json").read_text())
    rows = [int(k) for k in np.load(P1 / "v1_rows.npy")]
    rand1, rand2 = np.load(P1 / "cascade_L13.npy"), np.load(P1 / "cascade_plant_seed2.npy")
    flux = np.stack([np.asarray(st[k][1], np.float64).reshape(16, 16, 16, 16).sum(axis=(1, 3)) for k in rows])
    for a in V1_ANSWERS:
        sel = [j for j, k in enumerate(rows) if groups[str(int(ids[k]))].rsplit(":", 1)[0] == a]
        ia = names.index(a)
        pa = v1_stat(rand2[sel, ia], rand1[sel, ia])
        pb = v1_stat(flux[sel], rand1[sel, ia])
        # arithmetic reachability (reported; does not lift UNREACHABLE): the randomised map itself
        # plus equal-variance noise in M's place
        pc = v1_stat(np.stack([x + rng.normal(0, x.std(), x.shape) for x in rand1[sel, ia]]), rand1[sel, ia])
        out[f"V1 {a}"] = {"seed2_vs_seed1": pa, "flux_vs_seed1": pb, "arithmetic": pc,
                          "reachable": pa["state"] == "FAIL" or pb["state"] == "FAIL"}
    # V2: plant = the unshifted band maps with their bright-edge patches scaled up by 50%; null = identical
    v2rows = [int(k) for k in np.load(P1 / "v2_rows.npy")]
    base = np.load(P1 / "m_drop1.npy", mmap_mode="r")[v2rows, names.index("band_offset_pc1")]
    be = [bright_edge(st[k]) for k in v2rows]
    planted = np.stack([np.where(m, 1.5 * b, b) for b, m in zip(base, be, strict=True)])
    noise = lambda a: a + rng.normal(0, 1e-6, a.shape)  # noqa: E731 — breaks exact ties for Wilcoxon
    out["V2 plant"] = v2_stat(base, noise(planted), noise(base), be)
    out["V2 null"] = v2_stat(base, noise(base), noise(base), be)
    # V3
    if (P1 / "v3_u_drop1.npy").exists():
        for suffix in ("", "_f25"):
            orc, cen = {}, {}
            for s, (_mm, uu, lab, dom) in v3_inputs(suffix=suffix).items():
                oracle = [lb.astype(float) + rng.normal(0, 0.5, lb.shape) for lb in lab]
                fake = [centre_prior() + rng.normal(0, 1.0, (16, 16)) for _ in lab]
                orc[s] = v3_stat(oracle, uu, lab, dom)
                cen[s] = v3_stat(fake, uu, lab, dom)
            out[f"V3 oracle{suffix}"] = v3_states(orc)
            out[f"V3 centre-as-M{suffix}"] = v3_states(cen)
    (P1 / "plants.json").write_text(json.dumps(out, indent=1, default=float))
    return out


# ── verdicts ─────────────────────────────────────────────────────────────────────────────────────

def score(with_v3: bool = False) -> dict:
    ids, st = D.stamps("occl")
    groups = json.loads((D.LOCAL / "occl_groups.json").read_text())
    names = json.loads((P1 / "names.json").read_text())
    pl = json.loads((P1 / "plants.json").read_text())
    res: dict = {"V1": {}, "V1_curve": {}}
    rows = [int(k) for k in np.load(P1 / "v1_rows.npy")]
    rand = np.load(P1 / "cascade_L13.npy")
    m0 = np.load(P1 / "cascade_L00.npy")
    for a in V1_ANSWERS:
        sel = [j for j, k in enumerate(rows) if groups[str(int(ids[k]))].rsplit(":", 1)[0] == a]
        ia = names.index(a)
        r = v1_stat(m0[sel, ia], rand[sel, ia])
        if not pl[f"V1 {a}"]["reachable"] and r["state"] != "FAIL":
            r["state"] = "UNREACHABLE"
        res["V1"][a] = r
        res["V1_curve"][a] = [v1_stat(m0[sel, ia], np.load(P1 / f"cascade_L{lv:02d}.npy")[sel, ia])["median"]
                              for lv in CASCADE_LEVELS]
    order = ["FAIL", "UNREACHABLE", "FRAGILE", "PASS"]
    res["V1_state"] = min((r["state"] for r in res["V1"].values()), key=order.index)
    v2rows = [int(k) for k in np.load(P1 / "v2_rows.npy")]
    ib = names.index("band_offset_pc1")
    base = np.load(P1 / "m_drop1.npy", mmap_mode="r")[v2rows, ib]
    be = [bright_edge(st[k]) for k in v2rows]
    res["V2"] = v2_stat(base, np.load(P1 / "v2_gshift.npy")[:, ib], np.load(P1 / "v2_allshift.npy")[:, ib], be)
    allm = np.load(P1 / "m_drop1.npy", mmap_mode="r")
    res["V2_concentration_occl"] = float(np.median([edge_mass(allm[k, ib], bright_edge(st[k]))
                                                    for k in range(len(ids))]))
    res["V3"] = {}
    if not with_v3:  # withheld (D28): the pre-registered comparison is saturated; see interp_tooling.md
        votes_list: tuple[int, ...] = ()
    else:
        votes_list = (3, 2, 5)
    for votes in votes_list:
        for which in ("m_drop1", "m_drop2", "m_noise1"):
            key = f"{which}_v{votes}"
            r = {s: v3_stat(*args) for s, args in v3_inputs(which, votes).items()}
            res["V3"][key] = v3_states(r)
    if with_v3:
        res["V3"]["m_drop1_v3_f25"] = v3_states({s: v3_stat(*a) for s, a in v3_inputs("m_drop1", 3, "_f25").items()})
        res["V3_state"] = {s: res["V3"]["m_drop1_v3"][s]["state"] for s in V3_ANSWER}
    else:
        res["V3_state"] = "WITHHELD (plants fail, D28)"
    res["V4"] = {}
    v4rows = [int(k) for k in np.load(P1 / "v4_rows.npy")]
    for op, fn in V4_OPS.items():
        tr = np.load(P1 / f"v4_{op}.npy")
        for a in (*V1_ANSWERS, "band_offset_pc1", "norm"):
            ia = names.index(a)
            r = [_rho(tr[j, ia], fn(allm[k, ia])) for j, k in enumerate(v4rows)]
            res["V4"][f"{op} {a}"] = _boot_median(np.asarray(r))
    (P1 / "score.json").write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    P1.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    if cmd in ("maps", "cascade", "v2", "v3", "v4", "all"):
        ctx = Ctx()
        (P1 / "names.json").write_text(json.dumps(ctx.names))
        stages = {"maps": maps, "cascade": cascade, "v2": v2, "v3": v3, "v4": v4}
        for name, fn in stages.items():
            if cmd in (name, "all"):
                t = time.time()
                fn(ctx)
                print(f"[{name}] {time.time() - t:.0f}s", flush=True)
    elif cmd == "plants":
        print(json.dumps(plants(), indent=1, default=float))
    elif cmd == "score":
        print(json.dumps(score(), indent=1, default=float))
