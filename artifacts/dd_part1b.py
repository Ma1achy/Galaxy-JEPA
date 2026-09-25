"""Brief DD, Stop-2 follow-ups (user decisions 2026-09-25). TOOL VALIDATION ONLY.

V1 exploratory:
  (a) direct / indirect split of the token-drop map. Dropping token j changes the mean-pooled block-11
      embedding p by a DIRECT term, j's own absence from the mean holding every other token fixed:
          p − mean_{i≠j} h_i = (h_j − p) / 255,
      plus an INDIRECT term, the change in the other 255 tokens (attention no longer sees j). Mapped
      through a read-out w, direct_j = w·(h_j − p) / 255 / SD. It is rank-identical to the cheap
      per-token read-out map w·h_j.
  (b) concept specificity: median |ρ| between the bar, spiral and edge-on maps of one galaxy, against
      each concept's agreement with itself across removal modes (drop vs noise fill, 1 patch vs 2×2).
  (c) brightness baseline: summed r flux per patch, reported wherever the centre prior is.
V3 amendment (post hoc, D27; before M's V3 maps were scored): ring-stratified AUC.

  uv run python artifacts/dd_part1b.py direct     # block-11 direct maps: occl (M) + V3 list (M, untrained)
  uv run python artifacts/dd_part1b.py v1x        # (a) (b) (c)
  uv run python artifacts/dd_part1b.py plants     # V3 amendment plants (D28), before its hash
  uv run python artifacts/dd_part1b.py v3         # V3 amendment verdicts, after its hash
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import dd_core as D  # noqa: E402
import dd_part1 as P  # noqa: E402

N_BOOT, MIN_DAUC = 10_000, 0.02
_c = (np.arange(D.GRID) + 0.5) * 16 - 128
RING = np.floor(np.hypot(*np.meshgrid(_c, _c)) / 16).astype(int)  # 16-px annuli of patch centres


# ── maps ─────────────────────────────────────────────────────────────────────────────────────────

def flux_map(stamp: np.ndarray) -> np.ndarray:
    """Brightness baseline: summed (normalised) r flux per 16-px patch."""
    return np.asarray(stamp[1], np.float64).reshape(D.GRID, 16, D.GRID, 16).sum(axis=(1, 3))


@torch.no_grad()
def direct_maps(model, readouts: list, imgs, batch: int = 64) -> np.ndarray:
    """(N, K, 16, 16): w·(h_j − p) / 255 / SD for every read-out column."""
    w = np.concatenate([ro.w for ro in readouts], axis=1)
    sd = np.concatenate([ro.sd for ro in readouts])
    out = []
    for a in range(0, len(imgs), batch):
        x = torch.from_numpy(np.stack([np.asarray(i, np.float32) for i in imgs[a:a + batch]])).to(D.DEVICE)
        h = D.block_tokens(model, x, (D.READ_BLOCK,))[D.READ_BLOCK].cpu().double().numpy()  # (b, 256, D)
        p = h.mean(1, keepdims=True)
        d = ((h - p) @ w) / (h.shape[1] - 1) / sd  # (b, 256, K)
        out.append(d.transpose(0, 2, 1).reshape(len(x), -1, D.GRID, D.GRID).astype(np.float32))
    return np.concatenate(out)


def direct(ctx: P.Ctx) -> None:
    np.save(P.P1 / "direct_m_occl.npy", direct_maps(ctx.m, [ctx.probes, ctx.band], ctx.st))
    import dd_v3
    ids = [int(i) for i in np.load(P.P1 / "v3_ids.npy")]
    st = dd_v3._stamps_for(ids)
    np.save(P.P1 / "direct_m_v3.npy", direct_maps(ctx.m, [ctx.probes, ctx.band], st))
    enc_u, ro_u = ctx.untrained()
    np.save(P.P1 / "direct_u_v3.npy", direct_maps(enc_u, [ro_u], st))
    np.save(P.P1 / "flux_v3.npy", np.stack([flux_map(s) for s in st]).astype(np.float32))


# ── V1 exploratory ───────────────────────────────────────────────────────────────────────────────

def _share(total: np.ndarray, dmap: np.ndarray) -> float:
    """Fraction of the map's variance the direct term carries: 1 − var(total − direct) / var(total)."""
    return float(1 - np.var(total - dmap) / np.var(total))


def v1x() -> dict:
    ids, st = D.stamps("occl")
    groups = json.loads((D.LOCAL / "occl_groups.json").read_text())
    names = json.loads((P.P1 / "names.json").read_text())
    m1 = np.load(P.P1 / "m_drop1.npy", mmap_mode="r")
    dm = np.load(P.P1 / "direct_m_occl.npy", mmap_mode="r")
    sec = [int(k) for k in np.load(P.P1 / "secondary_rows.npy")]
    modes = {t: np.load(P.P1 / f"m_{t}.npy", mmap_mode="r") for t in ("drop2", "noise1")}
    concepts = ("t03_bar_a06_bar", "t04_spiral_a08_spiral", "t02_edgeon_a04_yes")
    out: dict = {"direct": {}, "specificity": {}, "brightness": {}}
    # (a) on each concept's own 54 galaxies, and the band axis on all 2,000
    for a in (*concepts, "band_offset_pc1"):
        ia = names.index(a)
        rows = [k for k, o in enumerate(ids) if groups[str(int(o))].rsplit(":", 1)[0] == a] if a in concepts \
            else range(len(ids))
        sh = np.array([_share(m1[k, ia], dm[k, ia]) for k in rows])
        rho = np.array([P._rho(m1[k, ia], dm[k, ia]) for k in rows])
        out["direct"][a] = {"share_median": float(np.median(sh)), "share_iqr": np.percentile(sh, [25, 75]).tolist(),
                            "rho_median": float(np.median(rho)), "n": len(sh)}
    # (b) cross-concept |ρ| on all 2,000 (occlusion and direct maps) vs self-agreement across modes
    for tag, arr in (("occlusion", m1), ("direct", dm)):
        for i, a in enumerate(concepts):
            for b in concepts[i + 1:]:
                r = [abs(P._rho(arr[k, names.index(a)], arr[k, names.index(b)])) for k in range(len(ids))]
                out["specificity"][f"{tag}: {a[4:10]} vs {b[4:10]}"] = float(np.median(r))
    for a in concepts:
        ia = names.index(a)
        for t, arr in modes.items():
            r = [abs(P._rho(m1[k, ia], arr[j, ia])) for j, k in enumerate(sec)]
            out["specificity"][f"self: {a[4:10]} drop1 vs {t}"] = float(np.median(r))
    # (c) brightness: how much of each map is the flux map
    for a in (*concepts, "band_offset_pc1", "norm"):
        ia = names.index(a)
        out["brightness"][a] = {
            "occlusion": float(np.median([abs(P._rho(m1[k, ia], flux_map(st[k]))) for k in range(len(ids))])),
            **({"direct": float(np.median([abs(P._rho(dm[k, ia], flux_map(st[k]))) for k in range(len(ids))]))}
               if a != "norm" else {})}
    (P.P1 / "v1x.json").write_text(json.dumps(out, indent=1))
    return out


# ── V3 amendment: ring-stratified AUC ────────────────────────────────────────────────────────────

def ring_auc(score: np.ndarray, lab: np.ndarray, dom: np.ndarray) -> float:
    """Mann–Whitney AUC over (positive, negative) patch pairs in the same 16-px annulus, pooled over
    annuli (each annulus weighted by its pair count). NaN if no annulus holds both classes."""
    num = den = 0.0
    for r in np.unique(RING[dom]):
        m = dom & (RING == r)
        p, n = score[m & lab], score[m & ~lab]
        if len(p) and len(n):
            d = p[:, None] - n[None, :]
            num += float((d > 0).sum() + 0.5 * (d == 0).sum())
            den += d.size
    return num / den if den else float("nan")


def _boot_ci(v: np.ndarray, seed: int = P.SEED) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    b = np.median(v[rng.integers(0, len(v), (N_BOOT, len(v)))], axis=1)
    return float(np.median(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def v3b_stat(m_maps, others: dict[str, list], labels, domains) -> dict:
    """Ring AUC per galaxy; for each comparator, the paired ΔAUC (M − comparator) with a bootstrap 95%
    CI of its median over galaxies. States, in precedence: INSUFFICIENT (< 30 usable galaxies) >
    REVERSED (either CI wholly below 0) > FAIL (either CI reaches 0) > WEAK (both CIs above 0, either
    median Δ < 0.02) > PASS."""
    am = np.array([ring_auc(m, lb, d) for m, lb, d in zip(m_maps, labels, domains, strict=True)])
    ok = ~np.isnan(am)
    out: dict = {"n_usable": int(ok.sum()), "n": len(am), "auc_m": _boot_ci(am[ok])}
    if ok.sum() < P.V3_MIN_N:
        out["state"] = "INSUFFICIENT"
        return out
    los, meds = [], []
    for name, maps in others.items():
        ao = np.array([ring_auc(x, lb, d) for x, lb, d in zip(maps, labels, domains, strict=True)])[ok]
        med, lo, hi = _boot_ci(am[ok] - ao)
        out[f"auc_{name}"] = _boot_ci(ao)
        out[f"dauc_vs_{name}"] = [med, lo, hi]
        los.append((lo, hi))
        meds.append(med)
    if any(hi < 0 for _, hi in los):
        out["state"] = "REVERSED"
    elif any(lo <= 0 for lo, _ in los):
        out["state"] = "FAIL"
    elif any(m < MIN_DAUC for m in meds):
        out["state"] = "WEAK"
    else:
        out["state"] = "PASS"
    return out


def v3b_inputs(which: str = "m_drop1", votes: int = 3, suffix: str = ""):
    """Per structure: (M maps, {untrained, brightness}, labels, domains); M maps from `which`
    ('direct' = the direct-term map)."""
    lst = json.loads((D.LOCAL / "v3_list.json").read_text())["lists"]
    ids = [int(i) for i in np.load(P.P1 / "v3_ids.npy")]
    names = json.loads((P.P1 / "v3_names.json").read_text())
    mm = np.load(P.P1 / ("direct_m_v3.npy" if which == "direct" else f"v3_{which}.npy"), mmap_mode="r")
    uu = np.load(P.P1 / "v3_u_drop1.npy", mmap_mode="r")
    fx = np.load(P.P1 / "flux_v3.npy")
    lab = np.load(D.LOCAL / "v3_patch_labels.npz")
    row = {o: k for k, o in enumerate(ids)}
    out = {}
    for s, ans in P.V3_ANSWER.items():
        im, iu = names["m"].index(ans), names["u"].index(ans)
        gal = lst[s + suffix]
        ks = [row[o] for o in gal]
        out[s] = ([np.asarray(mm[k, im]) for k in ks],
                  {"untrained": [np.asarray(uu[k, iu]) for k in ks], "brightness": [fx[k] for k in ks]},
                  [lab[f"{o}__{s}_{votes}{suffix}"] for o in gal], [lab[f"{o}__footprint"] for o in gal])
    return out


def plants() -> dict:
    """D28 for the amendment: the oracle (label + N(0, 0.5)) must PASS; the brightness map + N(0, 0.1·SD)
    in M's place must not PASS; and neither comparator may saturate (median ring AUC < 0.95)."""
    rng = np.random.default_rng(P.SEED + 29)
    out = {}
    for suffix in ("", "_f25"):
        for s, (_m, others, lab, dom) in v3b_inputs(suffix=suffix).items():
            oracle = [lb.astype(float) + rng.normal(0, 0.5, lb.shape) for lb in lab]
            fake = [f + rng.normal(0, 0.1 * f.std(), f.shape) for f in others["brightness"]]
            o, f = v3b_stat(oracle, others, lab, dom), v3b_stat(fake, others, lab, dom)
            out[f"{s}{suffix}"] = {"oracle": o["state"], "oracle_dauc": {k: o[k][0] for k in o if k.startswith("dauc")},
                                   "brightness_as_M": f["state"],
                                   "comparators": {k: o[k][0] for k in ("auc_untrained", "auc_brightness")}}
    (P.P1 / "v3b_plants.json").write_text(json.dumps(out, indent=1))
    return out


def v3() -> dict:
    res = {}
    for which in ("m_drop1", "m_drop2", "m_noise1", "direct"):
        for votes, suffix in ((3, ""), (2, ""), (5, ""), (3, "_f25")):
            if which != "m_drop1" and (votes, suffix) != (3, ""):
                continue
            res[f"{which}_v{votes}{suffix}"] = {s: v3b_stat(*a) for s, a in v3b_inputs(which, votes, suffix).items()}
    res["state"] = {s: res["m_drop1_v3"][s]["state"] for s in P.V3_ANSWER}
    (P.P1 / "v3b_score.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "direct":
        direct(P.Ctx())
    elif cmd == "v1x":
        print(json.dumps(v1x(), indent=1))
    elif cmd == "plants":
        print(json.dumps(plants(), indent=1))
    elif cmd == "v3":
        r = v3()
        print(json.dumps({k: {s: (v[s]["state"], v[s]["auc_m"][0], v[s].get("dauc_vs_untrained"),
                                  v[s].get("dauc_vs_brightness")) for s in v} for k, v in r.items() if k != "state"},
                         indent=1))
