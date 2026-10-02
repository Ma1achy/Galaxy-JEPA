"""Kickoff C: real data for the Plate (Almagest), from encoder M only. Compact, no images.

The steps, each deterministic (SEED), each taking the pool from the previous one's output:

  subset    of probe train+test (195,686): MiniBatchKMeans(400) on M's STANDARDISED
            block-11 embedding (per-dimension z-score over the pool — the probes' own space, so no
            handful of high-variance dimensions picks the clusters), a fixed cap per cluster
            (water-filled so the total lands exactly on --size), plus rare-class oversampling.
            Rare-class member = the repo's conditional population (D14, `consensus_gate` 0.5 on the
            upstream chain: t08 needs t06 odd >= 0.5; t09 needs featured >= 0.5 AND edge-on >= 0.5)
            AND raw vote fraction >= 0.5 (the probe label's threshold). Unconditional t08 >= 0.5
            is not rare — 6.6% of the pool are "mergers", mostly one vote of one oddity.
            Shrinking --size shrinks the per-cluster fill only: the rare picks are drawn first from
            the same seeded stream, so at a fixed --rare-cap they are the same galaxies at any size.
  pcasweep  RECORD of the rejected plain PCA: probe-AUC cost of the PCA-d re-expressed probe vs
            the full 384-D one, for each --dims, on the subset's test galaxies. No d up to 128 met
            mean loss <= 0.02 and worst <= 0.06 (plate_pca_sweep.json), hence the basis below.
  basiseval the 48-D basis (`_basis`: the 37 probe directions by QR + 11 residual principal
            components, in the standardised space) measured on the subset BEFORE export, against
            the v1 plate.bin of the same subset: AUC loss, kNN overlap beside PCA-32's, variance.
  export    one file: magic + JSON header + planar binary sections (plate_data_format.md).
            Probes: `_fit` with C from probe.yaml on the pool's probe-train split, exactly as
            aligned_c2.fit_scores; read-outs on train galaxies are in-sample (split flag). The fit
            does not depend on the subset, so it is cached (plate_probes.npz, keyed on the pool).
            Read-out byte = the decision, affine per answer onto 0-255 over its [0.1, 99.9]
            percentile range on the subset — NOT sigmoid x 255, which puts most of a rare answer's
            galaxies on a dozen levels (both are measured; see the report's `readout_mapping`).
            Coordinates on the 48-D basis: int8, per-dimension scale = max |z| over the file's
            galaxies / 127. Each probe is exact on the float coordinates.
  testfile  the first 500 galaxies (the file order is a seeded shuffle, so any prefix is a fair
            sample), standalone: kNN recomputed among the 500 in 384-D.
  sanity    from the exported file ALONE.

Raw GZ2 `_fraction` / `_count` only (via LabelProvider and `vote_column`, which refuses
`_debiased`). Nothing from pretrain_v2 or an aligned encoder: the only embedding bank read is
c2_m1 (runs/m/encoder.pt).

  uv run python artifacts/plate_export.py subset [--limit N --size S --clusters K --rare-cap R] [--out DIR]
  uv run python artifacts/plate_export.py pcasweep [--dims 32,48,64,96,128] [--out DIR]
  uv run python artifacts/plate_export.py basiseval [--out DIR]
  uv run python artifacts/plate_export.py export   [--out DIR] [--file NAME]
  uv run python artifacts/plate_export.py testfile [--out DIR]
  uv run python artifacts/plate_export.py sanity   [--out DIR] [--file plate.bin]
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from f0_preconditions import REPO  # noqa: E402

SEED = 20260927
MAGIC = b"ALMGPLT1"
N_PCA, K_NN, N_TEST_FILE = 64, 6, 500  # N_PCA: pcasweep's plain-PCA record only
N_RESIDUAL = 11  # basis = 37 probe directions (QR) + 11 residual principal components = 48
STD_REF = 0  # the standardisation every probe is re-expressed in: t01's, fitted on the whole train split
FORMAT_VERSION = 2  # 2: int8 coordinates on the probe-QR + residual-PCA basis (v1: float16 PCA-32)
PCA_LOSS_MEAN, PCA_LOSS_WORST = 0.02, 0.06  # pcasweep's acceptance bar (AUC, full 384 - PCA-d)
RO_PCT = (0.1, 99.9)  # read-out byte range, percentiles of each answer's decision on the subset
MEMBER_FRACTION = 0.5
#: (reason code - 1) -> (short name, Scheme-1 feature). Processed rarest-first; a galaxy's reason
#: is the first class that picked it, so the rarest classes keep their own label.
RARE = (
    ("merger", "t08_odd_feature_a24_merger"),
    ("ring", "t08_odd_feature_a19_ring"),
    ("dust_lane", "t08_odd_feature_a38_dust_lane"),
    ("edgeon_bulge_rounded", "t09_bulge_shape_a25_rounded"),
    ("edgeon_bulge_boxy", "t09_bulge_shape_a26_boxy"),
    ("edgeon_no_bulge", "t09_bulge_shape_a27_no_bulge"),
    ("star_or_artifact", "t01_smooth_or_features_a03_star_or_artifact"),
)
#: The Plate's own answer ids (its TASKS table), in GZ2_TREE order — the file's answer order.
PLATE_IDS = (
    "t01_smooth", "t01_feat", "t01_star", "t02_yes", "t02_no", "t03_yes", "t03_no", "t04_yes",
    "t04_no", "t05_none", "t05_notice", "t05_obvious", "t05_dominant", "t06_yes", "t06_no",
    "t07_round", "t07_between", "t07_cigar", "t08_ring", "t08_lens", "t08_disturbed",
    "t08_irregular", "t08_other", "t08_merger", "t08_dust", "t09_rounded", "t09_boxy", "t09_none",
    "t10_tight", "t10_medium", "t10_loose", "t11_1", "t11_2", "t11_3", "t11_4", "t11_more",
    "t11_cant",
)


# --- inputs -----------------------------------------------------------------------------------


def _bank() -> tuple[np.ndarray, np.memmap, str]:
    """M's embeddings, x memory-mapped out of the (uncompressed) npz — a --limit run reads only
    its own rows, and the full run never holds a second copy of the float32 block."""
    from numpy.lib import format as npf

    from aligned_c2 import _bank as bank_path
    from probe_bank import load_bank
    p = bank_path("m1")
    with load_bank(p, expected_checkpoint="runs/m/encoder.pt") as b:  # M by SHA-1, not only by path
        ids, ckpt = b["ids"].astype(np.int64), str(b["checkpoint"])
    if not ckpt.endswith("runs/m/encoder.pt"):
        raise SystemExit(f"plate: {p} is from {ckpt}, not M")
    if not np.all(np.diff(ids) > 0):
        raise SystemExit("plate: bank ids are not strictly ascending")
    info = zipfile.ZipFile(p).getinfo("x.npy")
    if info.compress_type != zipfile.ZIP_STORED:
        raise SystemExit("plate: bank is compressed; cannot memory-map")
    with open(p, "rb") as f:
        f.seek(info.header_offset)
        n, e = struct.unpack("<HH", f.read(30)[26:30])
        f.seek(info.header_offset + 30 + n + e)
        v = npf.read_magic(f)
        shape, fortran, dt = (npf.read_array_header_1_0 if v == (1, 0) else npf.read_array_header_2_0)(f)
        off = f.tell()
    assert not fortran and shape == (len(ids), 384)
    return ids, np.memmap(p, dtype=dt, mode="r", offset=off, shape=shape), ckpt


def _rows(ids: np.ndarray, x: np.memmap, want: np.ndarray) -> np.ndarray:
    i = np.searchsorted(ids, want)
    if not np.array_equal(ids[i], want):
        raise SystemExit("plate: ids missing from the embedding bank")
    return np.asarray(x[i], dtype=np.float64)


def _setup():
    from j4_spread_controls import prepare
    return prepare(None, 0, label="plate", sources=1)


def _pool(s, limit: int) -> np.ndarray:
    """Probe train+test, or a seeded draw of `limit` of them for development."""
    ids = np.array(sorted({*s.train_ids, *s.test_ids}), dtype=np.int64)
    if limit:
        ids = np.sort(np.random.default_rng(SEED).choice(ids, limit, replace=False))
    return ids


def _members(s, pool: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean membership over `pool` per rare class (conditional population, fraction >= 0.5)."""
    from galaxy_jepa.probing.schemes import eligible_ids

    # eligible_ids directly, written when labels.with_population copied the whole ProbeColumns
    # sidecar into dicts (measured 5.7 GB peak); it now shares the rows, so either is fine
    L = s.labels
    out = {}
    for name, f in RARE:
        e = np.array(eligible_ids(L.rows, L.scheme.by_name[f], pool.tolist(), vote_count_min=L.vote_count_min,
                                  conditional=True, consensus_gate=L.consensus_gate), dtype=np.int64)
        pos = e[L.vote_fraction(f, e.tolist()) >= MEMBER_FRACTION]
        out[name] = np.isin(pool, pos)
    return out


def _git() -> str:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         cwd=Path(__file__).parent).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "artifacts/plate_export.py"],
                           capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
    return sha + ("+plate_export-uncommitted" if dirty else "")


# --- 1. subset --------------------------------------------------------------------------------


def subset(a) -> dict:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler
    s = _setup()
    pool = _pool(s, a.limit)
    if a.size > len(pool):
        raise SystemExit(f"plate: --size {a.size} exceeds the pool ({len(pool)})")
    bids, bx, _ = _bank()
    xs = StandardScaler().fit_transform(_rows(bids, bx, pool)).astype(np.float32)
    km = MiniBatchKMeans(n_clusters=a.clusters, random_state=SEED, batch_size=4096, n_init=3,
                         max_iter=200).fit(xs)
    cl = km.labels_.astype(np.int64)
    del xs
    rng = np.random.default_rng(SEED)
    mem = _members(s, pool)
    reason = np.full(len(pool), -1, dtype=np.int64)  # -1 = not chosen, 0 = cluster, k = RARE[k-1]
    code = {name: k + 1 for k, (name, _) in enumerate(RARE)}
    for name in sorted(mem, key=lambda n: (mem[n].sum(), n)):
        cand = np.flatnonzero(mem[name] & (reason < 0))
        take = rng.choice(cand, min(a.rare_cap, len(cand)), replace=False) if len(cand) else cand
        reason[take] = code[name]
    left = a.size - int((reason > 0).sum())
    avail = [np.flatnonzero((cl == c) & (reason < 0)) for c in range(a.clusters)]
    n_av = np.array([len(v) for v in avail])
    q = 0  # the largest per-cluster cap whose water-fill fits
    while np.minimum(n_av, q + 1).sum() <= left and q < n_av.max():
        q += 1
    extra = left - int(np.minimum(n_av, q).sum())
    bump = {int(c) for c in [c for c in rng.permutation(a.clusters) if n_av[c] > q][:extra]}
    for c in range(a.clusters):
        k = min(n_av[c], q) + (c in bump)
        reason[rng.choice(avail[c], k, replace=False) if k else avail[c][:0]] = 0
    pick = reason >= 0
    assert pick.sum() == a.size

    test = np.isin(pool, np.array(s.test_ids, dtype=np.int64))
    bal = {}
    for name, _ in RARE:
        m = mem[name]
        bal[name] = {"pool": int(m.sum()), "pool_share": float(m.mean()),
                     "subset": int((m & pick).sum()), "subset_share": float((m & pick).sum() / a.size),
                     "picked_for_this_reason": int((reason == code[name]).sum())}
    rep = {"pool": len(pool), "limit": a.limit, "size": a.size, "clusters": a.clusters,
           "rare_cap": a.rare_cap, "per_cluster_cap": q, "clusters_bumped": len(bump),
           "cluster_fill": int((reason == 0).sum()), "rare_fill": int((reason > 0).sum()),
           "subset_test": int((pick & test).sum()), "subset_train": int((pick & ~test).sum()),
           "pool_test_share": float(test.mean()),
           "cluster_sizes": {"min": int(np.bincount(cl).min()), "median": float(np.median(np.bincount(cl))),
                             "max": int(np.bincount(cl).max())},
           "kmeans": {"space": "M block-11 mean-pool, standardised per dimension over the pool",
                      "n_clusters": a.clusters, "batch_size": 4096, "n_init": 3, "random_state": SEED},
           "member_rule": "conditional population (consensus_gate 0.5 on the upstream chain) AND raw "
                          f"vote fraction >= {MEMBER_FRACTION}",
           "rare_classes": bal}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "plate_subset.npz", ids=pool[pick], reason=reason[pick].astype(np.uint8),
             cluster=cl[pick].astype(np.uint16), test=test[pick], limit=a.limit, seed=SEED)
    (out / "plate_subset.json").write_text(json.dumps(rep, indent=1))
    return rep


# --- file format ------------------------------------------------------------------------------


def write_plate(path: Path, header: dict, sections: list[tuple[str, np.ndarray, str]]) -> None:
    """magic(8) | u32 header bytes | u32 0 | JSON header (space-padded to 8) | 8-aligned sections."""
    body, secs, off = [], [], 0
    for name, arr, meaning in sections:
        arr = np.ascontiguousarray(arr)
        pad = -off % 8
        body.append(b"\0" * pad)
        off += pad
        raw = arr.astype(arr.dtype.newbyteorder("<"), copy=False).tobytes()
        secs.append({"name": name, "dtype": arr.dtype.name, "shape": list(arr.shape), "offset": off,
                     "bytes": len(raw), "meaning": meaning})
        body.append(raw)
        off += len(raw)
    h = json.dumps({**header, "sections": secs}, separators=(",", ":"), allow_nan=False).encode()
    h += b" " * (-(16 + len(h)) % 8)
    path.write_bytes(MAGIC + struct.pack("<II", len(h), 0) + h + b"".join(body))


def read_plate(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    buf = Path(path).read_bytes()
    if buf[:8] != MAGIC:
        raise SystemExit(f"plate: {path} is not a Plate file")
    hl = struct.unpack("<I", buf[8:12])[0]
    h = json.loads(buf[16:16 + hl])
    base = 16 + hl
    arrs = {}
    for sec in h["sections"]:
        dt = np.dtype(sec["dtype"]).newbyteorder("<")
        n = int(np.prod(sec["shape"]))
        assert n * dt.itemsize == sec["bytes"] and sec["offset"] % 8 == 0
        arrs[sec["name"]] = np.frombuffer(buf, dtype=dt, count=n, offset=base + sec["offset"]).reshape(sec["shape"])
    return h, arrs


def _knn(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """6 nearest neighbours (Euclidean, self excluded by index, not by distance 0)."""
    from sklearn.neighbors import NearestNeighbors
    d, i = NearestNeighbors(n_neighbors=K_NN + 1, algorithm="brute").fit(x).kneighbors(x)
    keep = i != np.arange(len(x))[:, None]
    keep[keep.sum(1) > K_NN, K_NN] = False  # self absent (exact duplicate): drop the 7th
    return i[keep].reshape(len(x), K_NN), d[keep].reshape(len(x), K_NN)


def _auc(s: np.ndarray, y: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def _u8(f: np.ndarray) -> np.ndarray:
    """floor(255 f + 0.5): f >= 0.5 <=> byte >= 128 exactly (0.5 * 255 = 127.5 rounds up)."""
    return np.floor(255 * np.clip(f, 0, 1) + 0.5).astype(np.uint8)


def _reference(feats, qof, test, counts, votes, fitted, **scores) -> list[dict]:
    """Per-answer AUC of each [37, n] score table on the test galaxies eligible for the answer."""
    out = []
    for j, f in enumerate(feats):
        e = test & (counts[qof[j]] >= 1)
        y = (votes[j][e] >= 128).astype(int)
        out.append({"answer": f, "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()),
                    **{k: _auc(np.asarray(v[j], dtype=np.float64)[e], y) if fitted[j] else None
                       for k, v in scores.items()}})
    return out


# --- shared by pcasweep and export ------------------------------------------------------------


def _probes(out: Path, s, pool: np.ndarray, bids: np.ndarray, bx: np.memmap) -> list[dict | None]:
    """The 37 probes, `_fit` on the pool's probe-train split exactly as aligned_c2.fit_scores.
    They depend on the pool (and C), never on the subset, so they are fitted once and cached;
    a cache whose key differs is refitted, not trusted."""
    import hashlib

    from galaxy_jepa.probing.extract import EmbeddingMatrix, feature_embeddings
    from galaxy_jepa.probing.logistic import _fit
    L = s.labels
    feats = L.features
    key = hashlib.sha1(pool.tobytes() + repr((s.pc.c, list(feats))).encode()).hexdigest()
    path = out / "plate_probes.npz"
    if path.exists():
        c = np.load(path, allow_pickle=False)
        if str(c["key"]) == key:
            return [{"w": c["w"][j], "b": float(c["b"][j]), "mu": c["mu"][j], "sd": c["sd"][j],
                     "n_train": int(c["n_train"][j]), "n_train_pos": int(c["n_train_pos"][j]),
                     "auc_probe_test": float(c["auc_probe_test"][j])} if c["fitted"][j] else None
                    for j in range(len(feats))]
        print("  plate_probes.npz is for another pool — refitting", file=sys.stderr)
    t0 = time.perf_counter()
    mat = EmbeddingMatrix(pool, _rows(bids, bx, pool), "m1")
    tr_ids = [o for o in s.train_ids if o in mat.index]
    te_ids = [o for o in s.test_ids if o in mat.index]
    probes = []
    for j, f in enumerate(feats):
        tr = feature_embeddings(mat, L, f, tr_ids)
        if len(np.unique(tr.y)) < 2:
            probes.append(None)
            print(f"  {f}: single-class train set — no probe", file=sys.stderr)
            continue
        sc, clf = _fit(tr, c=s.pc.c)
        te = feature_embeddings(mat, L, f, te_ids)
        probes.append({"w": clf.coef_[0].astype(np.float64), "b": float(clf.intercept_[0]),
                       "mu": sc.mean_, "sd": sc.scale_, "n_train": len(tr.y), "n_train_pos": int(tr.y.sum()),
                       "auc_probe_test": _auc(clf.decision_function(sc.transform(te.x)), te.y)})
        print(f"  probe {j + 1:2d}/37 {f} {time.perf_counter() - t0:6.0f}s", file=sys.stderr)
    z384, one = np.zeros((len(feats), 384)), np.ones((len(feats), 384))
    get = lambda k, fill: np.stack([p[k] if p else fill for p in probes])  # noqa: E731
    np.savez(path, key=key, fitted=np.array([p is not None for p in probes]), w=get("w", z384[0]),
             b=np.array([p["b"] if p else 0.0 for p in probes]), mu=get("mu", z384[0]), sd=get("sd", one[0]),
             n_train=np.array([p["n_train"] if p else 0 for p in probes]),
             n_train_pos=np.array([p["n_train_pos"] if p else 0 for p in probes]),
             auc_probe_test=np.array([p["auc_probe_test"] if p and p["auc_probe_test"] is not None
                                      else np.nan for p in probes]))
    return probes


def _decisions(probes: list[dict | None], x: np.ndarray) -> np.ndarray:
    """[37, n] probe decisions: w.((x - mu)/sd) + b, as clf.decision_function(sc.transform(x))."""
    dec = np.zeros((len(probes), len(x)))
    for j, p in enumerate(probes):
        if p is not None:
            dec[j] = ((x - p["mu"]) / p["sd"]) @ p["w"] + p["b"]
    return dec


def _reexpress(probes: list[dict | None], P: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each probe on the PCA scores z: exact on the reconstruction x = m + P^T z.
    score(x) = w.((x - mu)/sd) + b = (P (w/sd)).z + [b + (w/sd).(m - mu)]."""
    wp, bp = np.zeros((len(probes), len(P))), np.zeros(len(probes))
    for j, p in enumerate(probes):
        if p is not None:
            v = p["w"] / p["sd"]
            wp[j], bp[j] = P @ v, p["b"] + v @ (m - p["mu"])
    return wp, bp


def _quantise(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """int8 per dimension: scale = max |z| over these galaxies / 127, q = round(z / scale). No
    clipping is needed (|q| <= 127 by construction), so no galaxy saturates; z ~ q * scale."""
    scale = np.abs(z).max(0) / 127
    scale = np.where(scale > 0, scale, 1.0)
    return np.clip(np.round(z / scale), -127, 127).astype(np.int8), scale


def _basis(probes: list[dict | None], xs: np.ndarray, n_res: int = N_RESIDUAL) -> dict:
    """The 48-D basis, in the standardised space the probes read.

    The probes' scalers differ (each is fitted on its own answer's eligible train galaxies), so one
    is chosen — STD_REF, t01's, fitted on the whole train split — and every probe is re-expressed in
    it exactly (a change of scaler is affine): u_j = w_j sd0/sd_j, c_j = b_j + (w_j/sd_j).(mu0 - mu_j).
    Basis rows: the 37 u_j orthonormalised by QR in the file's answer order, then the top n_res
    principal components of the embedding with that span projected out. Every u_j lies in the
    basis's row space, so each probe on the coordinates is EXACT for every galaxy, not only on a
    reconstruction; the residual components are there for the neighbours."""
    from sklearn.decomposition import PCA
    if any(p is None for p in probes):
        raise SystemExit("plate: an unfitted probe has no direction to put in the basis")
    mu0, sd0 = probes[STD_REF]["mu"], probes[STD_REF]["sd"]
    U = np.stack([p["w"] * sd0 / p["sd"] for p in probes])
    c = np.array([p["b"] + (p["w"] / p["sd"]) @ (mu0 - p["mu"]) for p in probes])
    s = (xs - mu0) / sd0
    Q, R = np.linalg.qr(U.T)
    res = s - (s @ Q) @ Q.T
    V = PCA(n_components=n_res, svd_solver="full").fit(res).components_
    B, R2 = np.linalg.qr(np.hstack([Q, V.T]))  # re-orthonormalise; signs pinned to Q's and V's
    B = (B * np.sign(np.diag(R2))).T
    assert np.abs(B[:len(U)] - Q.T).max() < 1e-8 and np.abs(B[len(U):] - V).max() < 1e-6
    sbar = s.mean(0)
    centre = B @ sbar
    z = s @ B.T - centre
    wz = U @ B.T
    bz = c + wz @ centre
    return {"mu": mu0, "sd": sd0, "B": B, "centre": centre, "z": z, "w": wz, "b": bz,
            "r_diag_min_over_max": float(np.abs(np.diag(R)).min() / np.abs(np.diag(R)).max()),
            "variance_kept": float(z.var(0).sum() / s.var(0).sum()),
            "variance_kept_probe_span": float(z[:, :len(U)].var(0).sum() / s.var(0).sum()),
            "std_total_variance": float(s.var(0).sum())}


def _labels(L, gid: np.ndarray, feats) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """votes [37, n] u8 raw fractions, counts [11, n] u8 question totals, qof [37], questions."""
    from galaxy_jepa.data.metadata import GZ2_TREE, vote_column
    frac = np.stack([L.vote_fraction(f, gid.tolist()) for f in feats])
    if not np.isfinite(frac).all():
        raise SystemExit("plate: missing vote fractions in the subset")
    qs = list(GZ2_TREE)
    qcount = np.stack([sum(L._column(gid.tolist(), vote_column(q, ans, "count")) for ans in GZ2_TREE[q])
                       for q in qs])
    votes, counts = _u8(frac), np.minimum(qcount, 255).astype(np.uint8)
    qof = np.array([next(k for k, q in enumerate(qs) if f.startswith(q + "_")) for f in feats])
    # the file's label rule must BE the provider's — checked here, where both are in reach
    for j, f in enumerate(feats):
        el = np.array(L.eligible(f, gid.tolist()), dtype=np.int64)
        assert np.array_equal(np.isin(gid, el), counts[qof[j]] >= 1), f
        assert np.array_equal(votes[j] >= 128, L.binary_label(f, gid.tolist()) == 1), f
    return votes, counts, qof, qs


def _loss(per: list[dict], key: str, ref: str = "auc_384") -> dict:
    """AUC loss ref - key over fitted answers with a defined AUC (signed: > 0 = the PCA probe ranks worse)."""
    d = np.array([p[ref] - p[key] for p in per if p[ref] is not None and p[key] is not None])
    return {"mean": float(d.mean()), "median": float(np.median(d)), "worst": float(d.max()),
            "min": float(d.min()), "mean_abs": float(np.abs(d).mean()), "n_answers": len(d),
            "worst_answer": max((p for p in per if p[ref] is not None and p[key] is not None),
                                key=lambda p: p[ref] - p[key])["answer"]}


# --- 2. pcasweep ------------------------------------------------------------------------------


def pcasweep(a) -> dict:
    from sklearn.decomposition import PCA
    out = Path(a.out)
    sub = np.load(out / "plate_subset.npz")
    s = _setup()
    pool = _pool(s, int(sub["limit"]))
    gid, test = sub["ids"], sub["test"]
    bids, bx, _ = _bank()
    probes = _probes(out, s, pool, bids, bx)
    xs = _rows(bids, bx, gid)
    dec = _decisions(probes, xs)
    feats = s.labels.features
    votes, counts, qof, _ = _labels(s.labels, gid, feats)
    fitted = [p is not None for p in probes]
    dims = sorted(int(d) for d in a.dims.split(","))
    pca = PCA(n_components=max(dims), svd_solver="full").fit(xs)  # the d-component PCA = its first d rows
    rows, chosen = {}, None
    for d in dims:
        P, m = pca.components_[:d], pca.mean_
        z = (xs - m) @ P.T
        q, scale = _quantise(z)
        wp, bp = _reexpress(probes, P, m)
        per = _reference(feats, qof, test, counts, votes, fitted, auc_384=dec, auc_pca=wp @ z.T,
                         auc_pca_f16=wp @ z.astype(np.float16).astype(np.float64).T,
                         auc_pca_i8=wp @ (q * scale).T)
        rows[d] = {"loss_float": _loss(per, "auc_pca"), "loss_f16": _loss(per, "auc_pca_f16"),
                   "loss_i8": _loss(per, "auc_pca_i8"), "quantisation_i8_vs_f16": _loss(per, "auc_pca_i8", "auc_pca_f16"),
                   "quantisation_i8_vs_float": _loss(per, "auc_pca_i8", "auc_pca"),
                   "explained_variance_ratio": float(pca.explained_variance_ratio_[:d].sum()),
                   "per_answer": per}
        lf = rows[d]["loss_float"]
        ok = lf["mean"] <= PCA_LOSS_MEAN and lf["worst"] <= PCA_LOSS_WORST
        rows[d]["qualifies"] = ok
        print(f"  PCA-{d}: mean {lf['mean']:.4f} median {lf['median']:.4f} worst {lf['worst']:.4f}"
              f" ({lf['worst_answer']}) {'OK' if ok else '--'}", file=sys.stderr)
        if ok and chosen is None:
            chosen = d
    rep = {"n": len(gid), "n_test": int(test.sum()), "bar": {"mean": PCA_LOSS_MEAN, "worst": PCA_LOSS_WORST},
           "loss": "auc_384 - auc_pca_d per fitted answer, on the subset's test galaxies eligible for it; "
                   "auc_pca_d = the re-expressed probe on the float PCA-d scores (exact on the reconstruction)",
           "chosen": chosen, "dims": rows}
    (out / "plate_pca_sweep.json").write_text(json.dumps(rep, indent=1))
    return {**rep, "dims": {d: {k: v for k, v in r.items() if k != "per_answer"} for d, r in rows.items()}}


def _overlap(z: np.ndarray, q: np.ndarray, nb: np.ndarray) -> np.ndarray:
    """Per query: how many of its 6 nearest (Euclidean in z, self excluded) are in nb[query]."""
    sq = (z ** 2).sum(1)
    d2 = np.maximum(sq[q, None] + sq[None, :] - 2 * z[q] @ z.T, 0)
    d2[np.arange(len(q)), q] = np.inf
    fresh = np.argsort(d2, axis=1, kind="stable")[:, :K_NN]
    return np.array([len(set(fresh[i]) & set(nb[g])) for i, g in enumerate(q)])


def basiseval(a) -> dict:
    """The 48-D basis on the current subset, before any export: probe-AUC loss (float and int8),
    kNN overlap against the stored 384-D neighbours of the v1 plate.bin (same galaxies, same file
    order, same 1,000 seeded queries as `sanity`), variance kept, and the int8 section's gzip."""
    out = Path(a.out)
    sub = np.load(out / "plate_subset.npz")
    s = _setup()
    pool = _pool(s, int(sub["limit"]))
    order = np.random.default_rng(SEED).permutation(len(sub["ids"]))
    gid, test = sub["ids"][order], sub["test"][order]
    h, A = read_plate(out / "plate.bin")
    if not np.array_equal(A["objid"].astype(np.int64), gid):
        raise SystemExit("plate: plate.bin is not this subset in file order")
    bids, bx, _ = _bank()
    probes = _probes(out, s, pool, bids, bx)
    o = np.argsort(gid)
    xs = np.empty((len(gid), 384))
    xs[o] = _rows(bids, bx, gid[o])
    dec = _decisions(probes, xs)
    feats = s.labels.features
    votes, counts, qof, _ = _labels(s.labels, gid, feats)
    bas = _basis(probes, xs)
    z = bas["z"]
    assert np.allclose(z @ bas["w"].T + bas["b"], dec.T, atol=1e-6 * max(1, np.abs(dec).max()))
    zq, zscale = _quantise(z)
    zi = zq * zscale
    per = _reference(feats, qof, test, counts, votes, [True] * 37, auc_384=dec,
                     auc_basis=bas["w"] @ z.T + bas["b"][:, None], auc_basis_i8=bas["w"] @ zi.T + bas["b"][:, None])
    lf, li = _loss(per, "auc_basis"), _loss(per, "auc_basis_i8")
    # kNN: the stored neighbours are raw 384-D Euclidean among the 40,000 (v1 plate.bin) — the
    # reference the 5.22 of 6 for PCA-32 was measured against; the same queries as `sanity`
    q = np.random.default_rng(SEED).choice(len(gid), min(1000, len(gid)), replace=False)
    nb = A["knn"].T.astype(np.int64)
    ov = {"basis48_float": _overlap(z, q, nb), "basis48_int8": _overlap(zi, q, nb),
          "pca32_float16_v1_file": _overlap(A["pca"].T.astype(np.float64), q, nb)}
    # and against neighbours in the full STANDARDISED 384-D space (the space the basis lives in)
    sfull = (xs - bas["mu"]) / bas["sd"]
    nb_std = np.array([np.argsort(np.where(np.arange(len(gid)) == g, np.inf,
                                           ((sfull - sfull[g]) ** 2).sum(1)), kind="stable")[:K_NN] for g in q])
    ov_std = _overlap(zi, q, _scatter(nb_std, q, len(gid)))
    rep = {"n": len(gid), "n_test": int(test.sum()), "bar": {"mean": PCA_LOSS_MEAN, "worst": PCA_LOSS_WORST},
           "d": int(z.shape[1]), "standardisation": f"probe {STD_REF} ({feats[STD_REF]}): StandardScaler "
           f"fitted on its {probes[STD_REF]['n_train']:,} train galaxies; the other 36 re-expressed exactly",
           "qr_order": list(feats), "qr_r_diag_min_over_max": bas["r_diag_min_over_max"],
           "loss_float": lf, "loss_int8": li, "quantisation_int8_vs_float": _loss(per, "auc_basis_i8", "auc_basis"),
           "passes": li["mean"] <= PCA_LOSS_MEAN and li["worst"] <= PCA_LOSS_WORST,
           "variance_kept_std": bas["variance_kept"], "variance_kept_probe_span_only": bas["variance_kept_probe_span"],
           "knn": {"queries": len(q), "reference": "stored kNN of plate.bin v1: raw (unstandardised) 384-D "
                   f"Euclidean, among its {len(gid):,} galaxies, self excluded", "query_rule": "default_rng(SEED).choice(n, 1000)",
                   **{k: {"mean_overlap_of_6": float(v.mean()), "hist": np.bincount(v, minlength=K_NN + 1).tolist()}
                      for k, v in ov.items()},
                   "basis48_int8_vs_standardised384": {"mean_overlap_of_6": float(ov_std.mean()),
                                                       "hist": np.bincount(ov_std, minlength=K_NN + 1).tolist()}},
           "int8_section": {"bytes": zq.nbytes, "gzip9_alone": len(gzip.compress(np.ascontiguousarray(zq.T).tobytes(),
                                                                                  9, mtime=0))},
           "per_answer": per}
    (out / "plate_basis_eval.json").write_text(json.dumps(rep, indent=1))
    return {k: v for k, v in rep.items() if k not in ("per_answer", "qr_order")}


def _scatter(nb_q: np.ndarray, q: np.ndarray, n: int) -> np.ndarray:
    full = np.zeros((n, K_NN), dtype=np.int64)
    full[q] = nb_q
    return full


# --- 3. export --------------------------------------------------------------------------------


def export(a) -> dict:
    import pandas as pd
    import umap
    t0 = time.perf_counter()
    out = Path(a.out)
    sub = np.load(out / "plate_subset.npz")
    s = _setup()
    pool = _pool(s, int(sub["limit"]))
    order = np.random.default_rng(SEED).permutation(len(sub["ids"]))  # file order: any prefix is fair
    gid, reason, test = sub["ids"][order], sub["reason"][order], sub["test"][order]
    n = len(gid)
    assert n <= 0xFFFF, "uint16 kNN indices"
    assert np.array_equal(test, np.isin(gid, np.array(s.test_ids)))
    bids, bx, ckpt = _bank()
    probes = _probes(out, s, pool, bids, bx)
    o = np.argsort(gid)
    xs = np.empty((n, 384))
    xs[o] = _rows(bids, bx, gid[o])
    dec = _decisions(probes, xs)
    L = s.labels
    feats = L.features
    assert len(feats) == 37 and [f.split("_")[0] for f in feats] == [p.split("_")[0] for p in PLATE_IDS]

    # the 48-D probe-QR + residual-PCA basis in the standardised space; int8 per-dimension scale.
    # Each probe is exact on the float coordinates, for every galaxy
    bas = _basis(probes, xs)
    z, wp, bp = bas["z"], bas["w"], bas["b"]
    d = z.shape[1]
    assert np.allclose(z @ wp.T + bp, dec.T, atol=1e-6 * max(1, np.abs(dec).max()))
    zq, zscale = _quantise(z)

    # read-out bytes: affine on the decision per answer; sigmoid x 255 measured alongside
    lo = np.percentile(dec, RO_PCT[0], axis=1)
    hi = np.percentile(dec, RO_PCT[1], axis=1)
    hi = np.where(hi > lo, hi, lo + 1)
    ro = np.floor(255 * np.clip((dec - lo[:, None]) / (hi - lo)[:, None], 0, 1) + 0.5).astype(np.uint8)
    ro_sig = np.floor(255 / (1 + np.exp(-dec)) + 0.5).astype(np.uint8)

    votes, counts, qof, qs = _labels(L, gid, feats)

    # AUC cost on the subset's test galaxies, eligible per answer
    fitted = [p is not None for p in probes]
    per = _reference(feats, qof, test, counts, votes, fitted, auc_384=dec, auc_basis=wp @ z.T,
                     auc_basis_i8=wp @ (zq * zscale).T, auc_u8=ro, auc_u8_sigmoid=ro_sig)
    for r, p in zip(per, probes, strict=True):
        r["auc_probe_test_full"] = p["auc_probe_test"] if p else None
    # the full-precision decisions, so a standalone slice can carry references for ITS galaxies
    np.savez(out / "plate_decisions.npz", dec=dec.astype(np.float32), objid=gid)
    print(f"  probes + PCA {time.perf_counter() - t0:6.0f}s", file=sys.stderr)

    nb, nd = _knn(xs)
    print(f"  kNN {time.perf_counter() - t0:6.0f}s", file=sys.stderr)
    um = dict(n_neighbors=15, min_dist=0.1, metric="euclidean", n_components=2, random_state=SEED)
    xy = umap.UMAP(**um).fit_transform(xs).astype(np.float64)
    print(f"  UMAP {time.perf_counter() - t0:6.0f}s", file=sys.stderr)

    md = pd.read_csv(REPO / s.cfg.paths.probe_dir / "metadata.csv",
                     usecols=["object_id", "ra", "dec", "modelMag_r", "expAB_r"]).set_index("object_id").loc[gid]
    mag = md["modelMag_r"].to_numpy()
    ba = md["expAB_r"].to_numpy()
    mag_u = np.where(np.isfinite(mag), np.clip(np.round((mag - 10) * 1000), 0, 65534), 65535).astype(np.uint16)
    ba_u = np.where(np.isfinite(ba) & (ba > 0), np.clip(np.round(ba * 254), 0, 254), 255).astype(np.uint8)
    flags = (test.astype(np.uint8) | (reason.astype(np.uint8) << 1)).astype(np.uint8)

    header = {
        "format": "almagest-plate", "version": FORMAT_VERSION, "n": n, "encoder": "M",
        "provenance": {"checkpoint": ckpt, "embeddings": "artifacts/out/c2_m1_embeddings.npz (block-11 mean-pool)",
                       "code": _git(), "seed": SEED, "pool": len(pool), "limit": int(sub["limit"]),
                       "votes": "GZ2 raw _fraction / _count (never _debiased)"},
        "layout": "little-endian; multi-column fields are PLANAR: shape [cols, n], value(i, j) = a[j*n + i]",
        "answers": [{"id": f, "plate_id": PLATE_IDS[j], "question": qs[qof[j]], "q": int(qof[j])}
                    for j, f in enumerate(feats)],
        "questions": qs,
        "readout": {"map": "byte = floor(255*clip((d-lo)/(hi-lo),0,1)+0.5); d = lo + byte*(hi-lo)/255 "
                           "is the probe decision (logit); P(label) = 1/(1+exp(-d))",
                    "lo": lo.tolist(), "hi": hi.tolist(), "range_percentiles": list(RO_PCT),
                    "fitted": fitted,
                    "probe": "L2 logistic, C=%g, StandardScaler, fitted on the pool's probe-train split "
                             "(label: raw fraction >= 0.5; eligible: question total >= 1)" % s.pc.c,
                    "in_sample": "read-outs on train-split galaxies (flags bit0 = 0) are in-sample"},
        "probe_basis": {"rule": "d = w[j] . z + b[j] on the coordinates z = q * coords.scale; exact on the "
                                "float coordinates for every galaxy (each probe direction is in the basis)",
                        "w": wp.round(8).tolist(), "b": bp.round(8).tolist()},
        "coords": {"n_components": d, "n_probe": len(feats), "n_residual": d - len(feats),
                   "standardise": f"s = (x - std_mean) / std_scale: probe {STD_REF}'s ({feats[STD_REF]}) "
                                  f"StandardScaler, fitted on its {probes[STD_REF]['n_train']} train galaxies (the "
                                  "whole probe-train split); every other probe re-expressed in it exactly",
                   "construction": "basis rows 0..36: the 37 probe weight vectors in that space, orthonormalised "
                                   "by QR in qr_order; rows 37..47: the top principal components of the "
                                   "subset's standardised embedding with the probe span projected out",
                   "qr_order": list(feats),
                   "project": "z = basis @ ((x - std_mean) / std_scale) - centre",
                   "scale": zscale.tolist(),
                   "quantisation": "int8 q = round(z / scale); scale[k] = max |z[:, k]| over the file's galaxies "
                                   "/ 127 (no clipping); decode z[k] = q[k] * scale[k]",
                   "variance_kept": bas["variance_kept"], "variance_kept_probe_span": bas["variance_kept_probe_span"],
                   "variance_of": "the standardised embedding over the file's galaxies (sum of per-dimension variances)"},
        "map": {"method": "umap", "umap_learn": umap.__version__, "input": "raw 384-D embedding of the subset",
                **{k: v for k, v in um.items()}, "min": xy.min(0).tolist(), "max": xy.max(0).tolist()},
        "knn": {"k": K_NN, "space": "raw 384-D, Euclidean, among the file's galaxies", "self_excluded": True,
                "order": "nearest first"},
        "flags": {"bit0": "1 = probe-test split, 0 = probe-train (read-out in-sample)",
                  "bits1_3": {"0": "cluster-stratified", **{str(k + 1): r[0] for k, r in enumerate(RARE)}}},
        "reference_auc": [{k: v for k, v in r.items() if k != "auc_u8_sigmoid"} for r in per],
    }
    sections = [
        ("objid", gid.astype(np.uint64), "SDSS DR8+ objID (object_id; not dr7objid)"),
        ("ra", md["ra"].to_numpy().astype(np.float32), "degrees J2000"),
        ("dec", md["dec"].to_numpy().astype(np.float32), "degrees J2000"),
        ("readout", ro, "[37,n] probe read-out byte, see header.readout"),
        ("map", xy.T.astype(np.float16), "[2,n] UMAP x, y"),
        ("coords", zq.T.copy(), f"[{d},n] int8 coordinates on the basis; z = q * header.coords.scale[k]"),
        ("knn", nb.T.astype(np.uint16), "[6,n] row indices of the 6 nearest neighbours, nearest first"),
        ("knn_d6", nd[:, -1].astype(np.float16), "384-D Euclidean distance to the 6th neighbour"),
        ("votes", votes, "[37,n] raw GZ2 vote fraction, byte/255"),
        ("counts", counts, "[11,n] question total votes (sum of its answers' counts), capped 255"),
        ("mag_r", mag_u, "modelMag_r = 10 + v/1000; 65535 missing"),
        ("ba", ba_u, "expAB_r (exponential-fit b/a, r band) = v/254; 255 missing"),
        ("flags", flags, "bit0 split, bits1-3 inclusion reason"),
        ("std_mean", bas["mu"].astype(np.float32), "[384] standardisation mean (probe 0's scaler)"),
        ("std_scale", bas["sd"].astype(np.float32), "[384] standardisation scale (probe 0's scaler)"),
        ("basis", bas["B"].astype(np.float32), f"[{d},384] basis rows (orthonormal), standardised space"),
        ("centre", bas["centre"].astype(np.float32), f"[{d}] subtracted after projecting"),
    ]
    name = a.file or "plate.bin"
    write_plate(out / name, header, sections)
    rep = {"file": str(out / name), "n": n, "d": d, "seconds": round(time.perf_counter() - t0, 1),
           **_sizes(out / name), "readout_mapping": _cmp(per), "per_answer": per}
    (out / "plate_export_report.json").write_text(json.dumps(rep, indent=1))
    return rep


def _cmp(per: list[dict]) -> dict:
    d = {k: [abs(p[k] - p["auc_384"]) for p in per if p["auc_384"] is not None and p[k] is not None]
         for k in ("auc_u8", "auc_u8_sigmoid", "auc_basis", "auc_basis_i8")}
    out = {k: {"max_abs_delta": max(v) if v else None, "mean_abs_delta": float(np.mean(v)) if v else None}
           for k, v in d.items()}
    return {**out, "loss_basis": _loss(per, "auc_basis"), "loss_basis_i8": _loss(per, "auc_basis_i8"),
            "quantisation_i8_vs_float": _loss(per, "auc_basis_i8", "auc_basis")}


def _sizes(path: Path) -> dict:
    h, arrs = read_plate(path)
    raw = path.read_bytes()
    g = len(gzip.compress(raw, compresslevel=9, mtime=0))
    br = shutil.which("brotli")
    b = len(subprocess.run([br, "-q", "11", "-c", str(path)], capture_output=True, check=True).stdout) if br else None
    hl = struct.unpack("<I", raw[8:12])[0]
    fields = {k: {"bytes": v.nbytes, "per_galaxy": v.nbytes / h["n"],
                  "gzip9_alone": len(gzip.compress(v.tobytes(), 9, mtime=0))} for k, v in arrs.items()}
    fields["header_json"] = {"bytes": hl, "gzip9_alone": len(gzip.compress(raw[16:16 + hl], 9, mtime=0))}
    return {"bytes": len(raw), "gzip9": g, "brotli11": b, "brotli_tool": br, "fields": fields}


# --- 4. testfile ------------------------------------------------------------------------------


def testfile(a) -> dict:
    out = Path(a.out)
    h, arrs = read_plate(out / "plate.bin")
    k = min(N_TEST_FILE, h["n"])
    bids, bx, _ = _bank()
    gid = arrs["objid"][:k].astype(np.int64)
    o = np.argsort(gid)
    x = np.empty((k, 384))
    x[o] = _rows(bids, bx, gid[o])
    nb, nd = _knn(x)
    per_gal = {"objid", "ra", "dec", "readout", "map", "coords", "knn", "knn_d6", "votes", "counts",
               "mag_r", "ba", "flags"}
    sec = []
    for s in h["sections"]:
        v = arrs[s["name"]]
        if s["name"] == "knn":
            v = nb.T.astype(np.uint16)
        elif s["name"] == "knn_d6":
            v = nd[:, -1].astype(np.float16)
        elif s["name"] in per_gal:
            v = v[..., :k]
        sec.append((s["name"], v, s["meaning"]))
    hd = {kk: vv for kk, vv in h.items() if kk != "sections"}
    hd["n"] = k
    hd["knn"] = {**h["knn"], "space": "raw 384-D, Euclidean, among these 500 only (recomputed)"}
    hd["parent"] = {"file": "plate.bin", "n": h["n"], "rows": f"first {k}"}
    side = np.load(out / "plate_decisions.npz")
    assert np.array_equal(side["objid"][:k], arrs["objid"][:k])
    qof = np.array([a_["q"] for a_ in h["answers"]])
    zk = arrs["coords"][:, :k].astype(np.float64) * np.array(h["coords"]["scale"])[:, None]
    pb = h["probe_basis"]
    hd["reference_auc"] = _reference([a_["id"] for a_ in h["answers"]], qof, (arrs["flags"][:k] & 1).astype(bool),
                                     arrs["counts"][:, :k], arrs["votes"][:, :k], h["readout"]["fitted"],
                                     auc_384=side["dec"][:, :k], auc_basis_i8=np.array(pb["w"]) @ zk + np.array(pb["b"])[:, None],
                                     auc_u8=arrs["readout"][:, :k])
    write_plate(out / "plate_test.bin", hd, sec)
    return {"file": str(out / "plate_test.bin"), "n": k, **_sizes(out / "plate_test.bin")}


# --- 5. sanity (from the file alone) ----------------------------------------------------------


def sanity(a) -> dict:
    """Needs nothing but the file: labels come from the stored fractions (byte >= 128 <=> raw
    fraction >= 0.5) and eligibility from the stored question totals (>= 1 <=> vote_count_min 1;
    the 255 cap cannot touch a >= 1 test) — both checked against LabelProvider at export."""
    path = Path(a.out) / (a.file or "plate.bin")
    h, A = read_plate(path)
    n = h["n"]
    test = (A["flags"] & 1).astype(bool)
    lo, hi = np.array(h["readout"]["lo"]), np.array(h["readout"]["hi"])
    scale = np.array(h["coords"]["scale"])
    z = A["coords"].T.astype(np.float64) * scale
    wp, bp = np.array(h["probe_basis"]["w"]), np.array(h["probe_basis"]["b"])
    ref = {r["answer"]: r for r in h["reference_auc"]}
    per, fails = [], []
    for j, ans in enumerate(h["answers"]):
        e = test & (A["counts"][ans["q"]] >= 1)
        y = (A["votes"][j][e] >= 128).astype(int)
        r = ref[ans["id"]]
        u8 = _auc(A["readout"][j][e].astype(float), y) if h["readout"]["fitted"][j] else None
        pp = _auc((z @ wp[j] + bp[j])[e], y) if h["readout"]["fitted"][j] else None
        d = None if u8 is None or r["auc_384"] is None else u8 - r["auc_384"]
        per.append({"answer": ans["id"], "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()),
                    "auc_384_ref": r["auc_384"], "auc_u8": u8, "delta_u8": d, "auc_basis_page": pp,
                    "auc_basis_i8_ref": r.get("auc_basis_i8")})
        if d is not None and abs(d) > 0.005:
            fails.append(ans["id"])
    # the read-out byte and the page-side PCA score describe the same probe: compare decisions
    d_u8 = lo[:, None] + A["readout"].astype(np.float64) * ((hi - lo) / 255)[:, None]
    d_p = wp @ z.T + bp[:, None]
    fit = np.array(h["readout"]["fitted"])
    inside = (d_p > lo[:, None]) & (d_p < hi[:, None])
    corr = [float(np.corrcoef(d_u8[j], d_p[j])[0, 1]) for j in np.flatnonzero(fit)]

    # kNN: stored (raw 384-D Euclidean) vs fresh on the stored, dequantised coordinates. The old
    # contraction checks (PCA distance <= 384-D distance <= d6) do not carry over: the coordinates
    # live in the STANDARDISED space and d6 in the raw one, so no inequality links them. What the
    # file can check instead: the basis is orthonormal, and the page's probe on the coordinates
    # reproduces the export's reference AUC for it
    rng = np.random.default_rng(SEED)
    q = rng.choice(n, min(1000, n), replace=False)
    nb = A["knn"].T.astype(np.int64)
    overlap = _overlap(z, q, nb)
    valid = bool(((nb >= 0) & (nb < n)).all() and (nb != np.arange(n)[:, None]).all()
                 and all(len(set(r)) == K_NN for r in nb))
    B = A["basis"].astype(np.float64)
    page_vs_ref = [abs(p["auc_basis_page"] - p["auc_basis_i8_ref"]) for p in per
                   if p["auc_basis_page"] is not None and p["auc_basis_i8_ref"] is not None]
    rep = {"file": str(path), "n": n, "n_test": int(test.sum()),
           "auc": {"target_abs_delta": 0.005, "max_abs_delta_u8": max((abs(p["delta_u8"]) for p in per
                                                                         if p["delta_u8"] is not None), default=None),
                   "answers_over_target": fails,
                   "basis_page_vs_reference_max_abs": max(page_vs_ref, default=None),
                   "basis_loss_vs_384": _loss([{"answer": p["answer"], "a": p["auc_384_ref"], "b": p["auc_basis_page"]}
                                               for p in per], "b", "a"),
                   "per_answer": per},
           "readout_vs_basis_decision": {"median_corr": float(np.median(corr)), "min_corr": float(np.min(corr)),
                                         "share_inside_readout_range": float(inside[fit].mean())},
           "basis": {"max_abs_BBt_minus_I": float(np.abs(B @ B.T - np.eye(len(B))).max()), "d": len(B)},
           "knn": {"queries": len(q), "reference": "stored: raw 384-D Euclidean among the file's galaxies",
                   "fresh": "Euclidean on the dequantised int8 coordinates",
                   "mean_overlap_of_6": float(overlap.mean()),
                   "overlap_hist": np.bincount(overlap, minlength=K_NN + 1).tolist(),
                   "indices_valid": valid}}
    (Path(a.out) / (path.stem + "_sanity.json")).write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("subset", "pcasweep", "basiseval", "export", "testfile", "sanity"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "out" / "plate"))
    ap.add_argument("--limit", type=int, default=0, help="subset: seeded draw of N pool galaxies (dev)")
    ap.add_argument("--size", type=int, default=40_000)
    ap.add_argument("--clusters", type=int, default=400)
    ap.add_argument("--rare-cap", type=int, default=1_000, help="max oversampled members per rare class")
    ap.add_argument("--dims", default="32,48,64", help="pcasweep: PCA dimensions to evaluate")
    ap.add_argument("--file", default=None, help="export: output name; sanity: file to check")
    a = ap.parse_args()
    r = {"subset": subset, "pcasweep": pcasweep, "basiseval": basiseval, "export": export, "testfile": testfile,
         "sanity": sanity}[a.cmd](a)
    print(json.dumps(r if a.cmd != "export" else {k: v for k, v in r.items() if k != "per_answer"}, indent=1))
