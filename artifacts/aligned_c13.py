"""Aligned comparison, criteria 1 (fix check, probe arm) and 3 (representation): machinery and D28 plants.

Read on the probe-test split (criteria 3–4 keep the DD cap of 40,000 train galaxies; criterion 1's
offset probes train on the capped train rows with recorded offsets). Embeddings are the c2 banks
(`aligned_c2.py embed`), which hold every train and test galaxy.

  uv run python artifacts/aligned_c13.py plants
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from aligned_c2 import _bank  # noqa: E402
from j4_spread_controls import OUT, prepare  # noqa: E402
from probe_bank import load_bank  # noqa: E402

SEED, N_BOOT, CAP = 20260926, 10_000, 40_000  # N_BOOT: 1a's 10,000-draw galaxy bootstrap
CHANCE_HALF, CI_BAND, TRACK = 0.05, (0.40, 0.60), 0.3
OFFSETS = ("g-r x", "g-r y", "i-r x", "i-r y")


def _load(tag: str, ids: list[int]) -> np.ndarray:
    b = load_bank(_bank(tag))
    pos = {int(o): k for k, o in enumerate(b["ids"])}
    return b["x"][[pos[i] for i in ids]].astype(np.float64)


def setup() -> dict:
    from dd_sae_cards import _panel
    from j4_spread_controls import _capped_train
    s = prepare(None, 0, label="C13", sources=1)
    train, test = _capped_train(sorted(s.train_ids), CAP), sorted(s.test_ids)
    return {"s": s, "train": train, "test": test,
            "pan_train": _panel(np.array(train)), "pan_test": _panel(np.array(test))}


# ── criterion 1a ────────────────────────────────────────────────────────────────────────────────────

def offset_state(x_tr, y_tr, x_te, y_te, seed: int = SEED) -> dict:
    """sign(offset) probe (standardised L2 logistic, C = 1), AUC with a galaxy-bootstrap CI."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(x_tr)
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(x_tr), y_tr)
    s = clf.decision_function(sc.transform(x_te))
    a = float(roc_auc_score(y_te, s))
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(s), len(s))
        if len(np.unique(y_te[i])) == 2:
            bs.append(roc_auc_score(y_te[i], s[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    if abs(a - 0.5) <= CHANCE_HALF and CI_BAND[0] <= lo and hi <= CI_BAND[1]:
        st = "NEAR CHANCE"
    elif lo > CI_BAND[1] or hi < CI_BAND[0]:
        st = "READABLE"
    else:
        st = "UNRESOLVED"
    return {"auc": a, "ci": [float(lo), float(hi)], "state": st, "n_test": len(y_te)}


def c1_arrays(d: dict, x_tr: np.ndarray, x_te: np.ndarray, name: str):
    tr_off, te_off = d["pan_train"][name], d["pan_test"][name]
    a, b = ~np.isnan(tr_off), ~np.isnan(te_off)
    return x_tr[a], (tr_off[a] > 0).astype(int), x_te[b], (te_off[b] > 0).astype(int), te_off[b], b


# ── criterion 3 ─────────────────────────────────────────────────────────────────────────────────────

def _rho(u: np.ndarray, v: np.ndarray) -> float:
    from scipy.stats import spearmanr
    ok = ~np.isnan(v)
    return float(abs(spearmanr(u[ok], v[ok])[0])) if ok.sum() >= 50 else 0.0


def families(d: dict) -> dict[str, list[np.ndarray]]:
    s, test = d["s"], d["test"]
    morph = []
    for f in s.labels.features:
        el = set(s.labels.eligible(f, test))
        v = s.labels.vote_fraction(f, test).astype(float)
        v[[k for k, i in enumerate(test) if i not in el]] = np.nan
        morph.append(v)
    return {"offsets": [d["pan_test"][n] for n in OFFSETS],
            "brightness": [d["pan_test"]["modelMag_r"], d["r_flux"]], "morphology": morph}


def representation(x: np.ndarray, fam: dict) -> dict:
    xc = x - x.mean(0)
    lam, vec = np.linalg.eigh(np.cov(xc, rowvar=False))
    lam, vec = lam[::-1], vec[:, ::-1]
    share = lam / lam.sum()
    pcs = []
    for k in range(10):
        z = xc @ vec[:, k]
        best = {f: max(_rho(z, v) for v in vs) for f, vs in fam.items()}
        top = max(best, key=best.get)
        pcs.append({"pc": k + 1, "share": float(share[k]), "rho": best, "top_family": top,
                    "tracks": top if best[top] >= TRACK else "untracked"})
    n = sum(p["share"] for p in pcs if p["tracks"] in ("offsets", "brightness"))
    mo = sum(p["share"] for p in pcs if p["tracks"] == "morphology")
    return {"N": n, "Mo": mo, "PR": float(lam.sum() ** 2 / (lam ** 2).sum()),
            "offset_pc": any(p["tracks"] == "offsets" for p in pcs), "pcs": pcs}


def seed_range(a: tuple[float, float], m: tuple[float, float], hi: str, same: str, lo: str,
               floor: float = 0.0) -> str:
    """Both aligned seeds beyond both M seeds by more than `floor` (v4: F_q = k·σ_q), else `same`."""
    if min(a) - max(m) > floor:
        return hi
    if min(m) - max(a) > floor:
        return lo
    return same


def c3_floor(m: tuple[float, float]) -> float:
    """F_q = k · |M1 − M2| / √2, with k calibrated by aligned_c3_floor.py (≤ 5% null false calls)."""
    k = json.loads((OUT / "c3_floor.json").read_text())["k_95"]
    return k * abs(m[0] - m[1]) / np.sqrt(2)


def c3_states(ra: tuple[dict, dict], rm: tuple[dict, dict]) -> dict:
    def rule(q, hi, same, lo):
        m = (rm[0][q], rm[1][q])
        return seed_range((ra[0][q], ra[1][q]), m, hi, same, lo, c3_floor(m))
    n = rule("N", "DIRTIER", "SAME", "LOWER")
    nuis = "CLEANER" if n == "LOWER" and not (ra[0]["offset_pc"] or ra[1]["offset_pc"]) else ("SAME" if n == "LOWER" else n)
    out = {"nuisance": nuis,
           "morphology_share": rule("Mo", "GAINED", "SAME", "LOST"),
           "dimensionality": rule("PR", "HIGHER", "SAME", "LOWER")}
    if all("flag" in r for r in (*ra, *rm)):
        out["flags"] = rule("flag", "HIGHER", "SAME", "LOWER")
    return out


def _r_flux(d: dict) -> np.ndarray:
    from f0_preconditions import check
    _, cache = check(verbose=False)
    rows = np.array([cache._row_of[i] for i in d["test"]])
    o = np.argsort(rows)
    out = np.empty(len(rows))
    for a in range(0, len(rows), 512):
        sel = o[a:a + 512]
        out[sel] = np.asarray(cache.data[rows[sel]][:, 1], np.float64).sum((1, 2))
    return out


def _c1_plants(d, out, rng, m1_te, xtr, ytr, xte, yte, off_te) -> None:
    """Criterion 1a — M1 real (must READABLE), permuted (must NEAR CHANCE), planted on the permuted null."""
    out["c1"]["M1 real i-r x (must READABLE)"] = offset_state(xtr, ytr, xte, yte)
    ptr, pte = rng.permutation(ytr), rng.permutation(yte)
    out["c1"]["offsets permuted (must NEAR CHANCE)"] = offset_state(xtr, ptr, xte, pte)
    u = rng.standard_normal(m1_te.shape[1])
    u /= np.linalg.norm(u)
    tr_off = d["pan_train"]["i-r x"][~np.isnan(d["pan_train"]["i-r x"])]
    # the component's scale is set so the planted coordinate has ρ ≈ 0.6 with the offset
    k = 0.75 * np.std(xtr @ u) / np.std(tr_off)
    ptr_x = xtr + k * np.outer(tr_off - tr_off.mean(), u)
    pte_x = xte + k * np.outer(off_te - off_te.mean(), u)
    planted = offset_state(ptr_x, ytr, pte_x, yte)
    planted["rho_planted_coordinate"] = _rho(pte_x @ u, off_te)
    out["c1"]["planted rho~0.6 (must READABLE)"] = planted
    out["c1"]["fire_as_expected"] = (out["c1"]["M1 real i-r x (must READABLE)"]["state"] == "READABLE"
                                     and out["c1"]["offsets permuted (must NEAR CHANCE)"]["state"] == "NEAR CHANCE"
                                     and planted["state"] == "READABLE")


def plants(only_c3: bool = False) -> dict:
    d = setup()
    d["r_flux"] = _r_flux(d)
    rng = np.random.default_rng(SEED)
    m1_tr, m1_te = _load("m1", d["train"]), _load("m1", d["test"])
    m2_te = _load("m2", d["test"])
    out: dict = {"c1": {}, "c3": {}}
    if only_c3:
        out["c1"] = json.loads((OUT / "c13_plants.json").read_text())["c1"]
    xtr, ytr, xte, yte, off_te, keep = c1_arrays(d, m1_tr, m1_te, "i-r x")
    if only_c3:
        # c1's draws come first from the shared rng; replay them so c3's plants see the stream they did
        rng.permutation(ytr), rng.permutation(yte), rng.standard_normal(m1_te.shape[1])
    else:
        _c1_plants(d, out, rng, m1_te, xtr, ytr, xte, yte, off_te)

    # criterion 3 — M1 and M2 as the 'M' pair; planted 'aligned' pairs built from them
    fam = families(d)
    rm = (representation(m1_te, fam), representation(m2_te, fam))
    out["c3"]["M1"] = {k: v for k, v in rm[0].items() if k != "pcs"} | {"pcs": [(p["pc"], round(p["share"], 3), p["top_family"], p["tracks"]) for p in rm[0]["pcs"]]}
    out["c3"]["floors (F_q)"] = {q: c3_floor((rm[0][q], rm[1][q])) for q in ("N", "Mo", "PR")}
    out["c3"]["M2"] = {k: v for k, v in rm[1].items() if k != "pcs"} | {"pcs": [(p["pc"], round(p["share"], 3), p["top_family"], p["tracks"]) for p in rm[1]["pcs"]]}
    off_full = d["pan_test"]["i-r x"]
    off_fill = np.where(np.isnan(off_full), np.nanmean(off_full), off_full)

    def inject(x):  # an offset-correlated component carrying 20% of the (new) total variance
        v = rng.standard_normal(x.shape[1])
        v /= np.linalg.norm(v)
        z = (off_fill - off_fill.mean()) / off_fill.std()
        tot = np.var(x - x.mean(0), axis=0).sum()
        return x + np.sqrt(0.25 * tot) * np.outer(z, v)

    def clean(x, r):  # the offset-tracking top-10 PCs projected out
        xc = x - x.mean(0)
        _, vec = np.linalg.eigh(np.cov(xc, rowvar=False))
        vec = vec[:, ::-1]
        ks = [p["pc"] - 1 for p in r["pcs"] if p["tracks"] == "offsets"]
        if not ks:
            return x
        v = vec[:, ks]
        return x - xc @ v @ v.T

    dirty = (representation(inject(m1_te), fam), representation(inject(m2_te), fam))
    cleaned = (representation(clean(m1_te, rm[0]), fam), representation(clean(m2_te, rm[1]), fam))
    out["c3"]["plant DIRTIER (offset component, 20% of variance)"] = c3_states(dirty, rm)
    out["c3"]["plant CLEANER (offset PCs projected out)"] = c3_states(cleaned, rm)
    out["c3"]["plant SAME (M1, M2 swapped)"] = c3_states((rm[1], rm[0]), rm)
    iso = np.random.default_rng(1).standard_normal((20_000, 50))
    rk2 = np.random.default_rng(2).standard_normal((20_000, 2)) @ np.random.default_rng(3).standard_normal((2, 50))
    empty = {"offsets": [], "brightness": [], "morphology": []}

    def pr_only(x):
        lam = np.linalg.eigvalsh(np.cov(x - x.mean(0), rowvar=False))
        return float(lam.sum() ** 2 / (lam ** 2).sum())
    out["c3"]["PR isotropic 50 (≈50)"] = pr_only(iso)
    out["c3"]["PR rank-2 (≈2 if equal)"] = pr_only(rk2)
    del empty
    # the flag rule on a synthetic latent: ρ ≈ 0.7 with i−r x must be offset-flagged
    z = np.where(np.isnan(off_full), np.nan, off_full)
    lat = np.nan_to_num(z, nan=0.0) * 0.7 / max(np.nanstd(z), 1e-9) + rng.standard_normal(len(z)) * 0.7
    best_n = max(_rho(lat, v) for v in fam["offsets"] + fam["brightness"])
    best_v = max(_rho(lat, v) for v in fam["morphology"])
    out["c3"]["synthetic offset latent flagged (must True)"] = bool(best_n > best_v)
    c3p = out["c3"]
    out["c3"]["fire_as_expected"] = (c3p["plant DIRTIER (offset component, 20% of variance)"]["nuisance"] == "DIRTIER"
                                     and c3p["plant CLEANER (offset PCs projected out)"]["nuisance"] == "CLEANER"
                                     and c3p["plant SAME (M1, M2 swapped)"] == {"nuisance": "SAME", "morphology_share": "SAME", "dimensionality": "SAME"}
                                     and c3p["synthetic offset latent flagged (must True)"])
    (OUT / "c13_plants.json").write_text(json.dumps(out, indent=1, default=float))
    return out


class _V2Stamps:
    """probe_v2 FITS through M's frozen pipeline, cast to fp16 as the cache stores them. probe_v2 has
    no freeze of its own until plan A8, so M's (v1) is the only one there is — declared, not hidden."""

    def __init__(self, ids: list[int], pipeline, root: Path):
        self.ids, self.pipeline, self.root = ids, pipeline, root

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, k: int) -> dict:
        import torch
        from galaxy_jepa.data.sources import load_fits_stamp
        x = np.asarray(self.pipeline(load_fits_stamp(self.root / f"{self.ids[k]}.fits")), np.float16)
        return {"image": torch.from_numpy(x), "object_id": self.ids[k]}


def untrained_floor() -> dict:
    """Criterion 1, reported with no state: the four sign(offset) probes (1a's, unchanged) on an
    untrained encoder (M's architecture, seed 0) read on probe_v2 — offset readability from galaxy
    properties correlated with the recorded offsets, not from the misregistration."""
    from f0_preconditions import check
    from galaxy_jepa.harness import _build_pipeline
    from galaxy_jepa.models.vit import load_frozen_encoder
    from galaxy_jepa.probing.controls import untrained_encoder_matrix
    d = setup()
    cfg, cache = check(verbose=False)
    pipe = _build_pipeline(q=cfg.q, freeze=cfg.normalisation)
    root = REPO_DATA / "probe_v2"
    # the FITS path reproduces the cache exactly on v1 stamps, so the only change on v2 is the pixels
    probe_ids = d["test"][:16]
    v1 = _V2Stamps(probe_ids, pipe, REPO_DATA / "probe")
    parity = max(float(np.abs(v1[k]["image"].numpy().astype(np.float32) - np.asarray(cache.get(o), np.float32)).max())
                 for k, o in enumerate(probe_ids))
    if parity != 0.0:
        raise SystemExit(f"FITS path differs from M's cache on v1 stamps (max |Δ| {parity})")
    have = {int(p.stem) for p in root.glob("*.fits")}
    train, test = [i for i in d["train"] if i in have], [i for i in d["test"] if i in have]
    config = load_frozen_encoder(Path(cfg.paths.out_dir) / "encoder.pt").config
    mat = untrained_encoder_matrix(config, _V2Stamps(train + test, pipe, root),
                                   device=cfg.runtime.resolved_device(), seed=0)
    x = mat.x.astype(np.float64)
    pos = {int(o): k for k, o in enumerate(mat.object_ids)}
    x_tr, x_te = x[[pos[i] for i in train]], x[[pos[i] for i in test]]
    from dd_sae_cards import _panel
    d2 = {"pan_train": _panel(np.array(train)), "pan_test": _panel(np.array(test))}
    out: dict = {"encoder": "untrained, M's architecture, seed 0", "corpus": "probe_v2",
                 "normalisation": f"M's freeze {cfg.normalisation.content_hash[:12]} (no v2 freeze yet)",
                 "fits_path_parity_v1_max_abs": parity, "n_train_embedded": len(train),
                 "n_test_embedded": len(test), "missing_v2": {"train": len(d["train"]) - len(train),
                                                               "test": len(d["test"]) - len(test)},
                 "n_boot": N_BOOT, "offsets": {}}
    for name in OFFSETS:
        xtr, ytr, xte, yte, _, _ = c1_arrays(d2, x_tr, x_te, name)
        out["offsets"][name] = offset_state(xtr, ytr, xte, yte) | {"n_train": len(ytr)}
    (OUT / "c1_untrained_v2.json").write_text(json.dumps(out, indent=1, default=float))
    return out


REPO_DATA = Path(__file__).resolve().parent.parent / "data"


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plants"
    fn = {"untrained": untrained_floor, "plants": plants, "c3": lambda: plants(only_c3=True)}[cmd]
    print(json.dumps(fn(), indent=1, default=float))
