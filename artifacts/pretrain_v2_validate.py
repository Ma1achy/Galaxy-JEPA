"""Validate pretrain_v2 as it arrives: integrity, identity, registration, coverage. Rerunnable.

Incremental: every stamp's integrity record is kept in WORK/pretrain_v2_validate_state.json keyed by
(name, size, mtime), so a rerun reads only the stamps that landed since. Read-only on the corpus.

  uv run python artifacts/pretrain_v2_validate.py            # all stamps so far + 1,000-galaxy registration
  uv run python artifacts/pretrain_v2_validate.py --quick N  # integrity on N random stamps only (a dry run)

No per-stamp checksum exists anywhere to check against: the manifest (written at the pull's end)
records only IDs + query, and chunk tarballs are deleted on extraction. So integrity is checked on
the stamps themselves: exact file size, header, shape, dtype, finite, no empty band, padding
consistent with the cut log's valid fraction, and band order against v1's flux per channel.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
V2, V1 = REPO / "data" / "pretrain_v2", REPO / "data" / "pretrain"
WORK = REPO / ".sciserver_work"
STATE = WORK / "pretrain_v2_validate_state.json"
OUT = REPO / "artifacts" / "out" / "pretrain_v2_validation.json"
FILE_BYTES = 2880 + 3 * 256 * 256 * 4 + 0  # one header block + the float32 cube, padded to 2880
FILE_BYTES += (-FILE_BYTES) % 2880
N_CHUNKS, SEED, N_REG = 827, 20260927, 1_000
BANDS = ("g", "r", "i")


def _load(path: Path) -> np.ndarray:
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        if len(h) != 1:
            raise ValueError(f"{len(h)} HDUs")
        return np.asarray(h[0].data)


def integrity(path: Path, valid_frac: float | None) -> dict:
    """One stamp: exact size, shape, dtype, finite, no empty band, zero pad ≈ 1 − valid_frac."""
    rec: dict = {"size_ok": path.stat().st_size == FILE_BYTES}
    try:
        x = _load(path)
    except Exception as exc:  # noqa: BLE001 — record, never skip
        return rec | {"ok": False, "error": repr(exc)[:200]}
    rec["shape_ok"] = x.shape == (3, 256, 256)
    rec["dtype_ok"] = x.dtype == np.dtype(">f4")
    rec["finite"] = bool(np.isfinite(x).all())
    xs = x.astype(np.float32)
    rec["empty_band"] = bool(any((xs[b] == 0).all() for b in range(3)))
    zero_all = (xs == 0).all(0)  # the cutter's pad is exactly 0.0 in all three bands
    rec["pad_frac"] = float(zero_all.mean())
    # the cutter pads the intersection of the band masks, so the pad is at least 1 − the tightest
    # band's valid fraction; valid_frac (the logged intersection) bounds it exactly up to the
    # mask threshold at the boundary
    rec["pad_ok"] = valid_frac is None or abs(rec["pad_frac"] - (1 - valid_frac)) <= 0.02
    rec["flux"] = [float(xs[b][~zero_all].sum()) for b in range(3)]
    rec["ok"] = all(rec[k] for k in ("size_ok", "shape_ok", "dtype_ok", "finite", "pad_ok")) and not rec["empty_band"]
    return rec


def _ids(path: Path, col: str) -> list[str]:
    with path.open(newline="") as fh:
        return [r[col] for r in csv.DictReader(fh)]


def _cut_log() -> dict[str, dict]:
    with (V2 / "cut_log.csv").open(newline="") as fh:
        return {r["object_id"]: r for r in csv.DictReader(fh)}


def centroid(img: np.ndarray, r: float = 12.0) -> tuple[float, float]:
    """Flux-weighted centroid (x, y) inside radius r of the stamp centre, border-median sky removed,
    iterated once about the first estimate. 0-based pixel centres; the target sits at 127.5."""
    sky = np.median(np.concatenate([img[:8].ravel(), img[-8:].ravel(), img[:, :8].ravel(), img[:, -8:].ravel()]))
    yy, xx = np.mgrid[:256, :256]
    cx = cy = 127.5
    for _ in range(2):
        m = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
        w = np.clip(img - sky, 0, None) * m
        s = w.sum()
        if s <= 0:
            return np.nan, np.nan
        cx, cy = float((w * xx).sum() / s), float((w * yy).sum() / s)
    return cx, cy


def registration(ids: list[str], log: dict[str, dict]) -> dict:
    """1,000 random galaxies: logged shifts recompute from the logged positions; measured g−r and
    i−r centroid offsets ~0 on v2 and track the recorded v1 in-stamp offsets on v1."""
    rng = np.random.default_rng(SEED)
    pick = sorted(rng.choice(ids, min(N_REG, len(ids)), replace=False).tolist())
    shift_err, rows = [], []
    for oid in pick:
        r = log[oid]
        for b in BANDS:  # s_b = 159.5 − (pos_b − origin_b): the padded 320² cut's centre
            shift_err.append(abs(159.5 - (float(r[f"{b}_x"]) - float(r[f"{b}_ox"])) - float(r[f"{b}_sx"])))
            shift_err.append(abs(159.5 - (float(r[f"{b}_y"]) - float(r[f"{b}_oy"])) - float(r[f"{b}_sy"])))
        v2, v1 = _load(V2 / f"{oid}.fits").astype(np.float64), _load(V1 / f"{oid}.fits").astype(np.float64)
        c2 = [centroid(v2[k]) for k in range(3)]
        c1 = [centroid(v1[k]) for k in range(3)]
        rec = {"id": oid}
        for pair, (a, b) in (("g-r", (0, 1)), ("i-r", (2, 1))):
            for ax, k in (("x", 0), ("y", 1)):
                rec[f"v2 {pair} {ax}"] = c2[a][k] - c2[b][k]
                rec[f"v1 {pair} {ax}"] = c1[a][k] - c1[b][k]
                ba, bb = BANDS[a], BANDS[b]
                rec[f"rec {pair} {ax}"] = float(r[f"{ba}_v1_rel{ax}"]) - float(r[f"{bb}_v1_rel{ax}"])
        rows.append(rec)
    out: dict = {"n": len(pick), "max_abs_logged_shift_mismatch_px": float(max(shift_err)), "offsets": {}}
    for pair in ("g-r", "i-r"):
        for ax in ("x", "y"):
            v2 = np.array([q[f"v2 {pair} {ax}"] for q in rows])
            v1 = np.array([q[f"v1 {pair} {ax}"] for q in rows])
            rc = np.array([q[f"rec {pair} {ax}"] for q in rows])
            ok = np.isfinite(v2) & np.isfinite(v1)
            fit = lambda y: float(np.polyfit(rc[ok], y[ok], 1)[0])  # noqa: E731
            out["offsets"][f"{pair} {ax}"] = {
                "v2_median_px": float(np.median(v2[ok])), "v2_slope_on_recorded": fit(v2),
                "v1_median_minus_recorded_px": float(np.median(v1[ok] - rc[ok])), "v1_slope_on_recorded": fit(v1),
                "recorded_sd_px": float(rc[ok].std()), "n": int(ok.sum())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    v1_ids = _ids(V1 / "metadata.csv", "object_id")
    targets = _ids(WORK / "pretrain_v2_all_targets.csv", "objID")
    log = _cut_log()
    names = {e.name[:-5]: e for e in os.scandir(V2) if e.name.endswith(".fits")}
    failed = _ids(V2 / "failed.csv", "object_id")
    s1, st, landed = set(v1_ids), set(targets), set(names)
    identity = {"v1_pretrain_n": len(v1_ids), "v1_unique": len(s1), "v2_targets_n": len(targets),
                "v2_targets_unique": len(st), "targets_symmetric_difference_vs_v1": len(s1 ^ st),
                "landed_n": len(landed), "landed_not_in_v1": len(landed - s1), "failed_n": len(failed),
                "cut_log_rows": len(log), "cut_log_equals_landed": set(log) == landed,
                "landed_symmetric_difference_vs_v1": len(s1 ^ landed)}

    state = json.loads(STATE.read_text()) if STATE.exists() and not args.quick else {}
    todo = list(names)
    if args.quick:
        todo = np.random.default_rng(SEED).choice(todo, min(args.quick, len(todo)), replace=False).tolist()
    n_new = 0
    for oid in todo:
        e = names[oid]
        st_ = e.stat()
        key = f"{st_.st_size}:{st_.st_mtime_ns}"
        if state.get(oid, {}).get("key") == key:
            continue
        vf = log.get(oid, {}).get("valid_frac")
        state[oid] = integrity(Path(e.path), float(vf) if vf else None) | {"key": key}
        n_new += 1
        if not args.quick and n_new % 20_000 == 0:
            STATE.write_text(json.dumps(state))
            print(f"  integrity {n_new:,} new ({time.time() - t0:.0f}s)", file=sys.stderr)
    if not args.quick:
        STATE.write_text(json.dumps(state))
    recs = [state[o] for o in todo]
    bad = {o: {k: v for k, v in state[o].items() if k not in ("flux", "key")} for o in todo if not state[o]["ok"]}
    integ = {"checked": len(recs), "new_this_run": n_new, "failures": len(bad), "failure_examples": dict(list(bad.items())[:10]),
             "expected_file_bytes": FILE_BYTES,
             "pad_frac_max": float(max(r.get("pad_frac", 0) for r in recs)),
             "checksums": "none exist to check against (manifest holds IDs + query only; tarballs deleted on extraction)"}

    # band order: each v2 channel's flux against the same v1 channel, 300 random landed galaxies
    rng = np.random.default_rng(SEED + 1)
    f1s, f2s = [], []
    for oid in rng.choice(sorted(landed), min(300, len(landed)), replace=False):
        v1 = _load(V1 / f"{oid}.fits").astype(np.float64)
        v2 = _load(V2 / f"{oid}.fits").astype(np.float64)
        z = (v1 == 0).all(0) | (v2 == 0).all(0)  # compare on pixels valid in both cuts
        f1s.append([v1[b][~z].sum() for b in range(3)])
        f2s.append([v2[b][~z].sum() for b in range(3)])
    f1s, f2s = np.array(f1s), np.array(f2s)
    perm_err = {}
    for name, p in (("identity g,r,i", (0, 1, 2)), ("g↔i swapped", (2, 1, 0)), ("g↔r swapped", (1, 0, 2)),
                    ("r↔i swapped", (0, 2, 1))):
        perm_err[name] = float(np.median(np.abs(f2s[:, list(p)] / f1s - 1)))
    integ["band_order"] = {"n": len(f1s), "median_flux_ratio_v2_over_v1_per_band": np.median(f2s / f1s, 0).tolist(),
                           "median_abs_ratio_error_by_ordering": perm_err,
                           "ok": min(perm_err, key=perm_err.get) == "identity g,r,i"}

    js = json.loads((WORK / "pretrain_v2.job.json").read_text())
    done = sum(c["status"] == "done" for c in js["chunks"])
    mt = sorted(e.stat().st_mtime for e in names.values())
    win = 6 * 3600
    recent = sum(1 for m in mt if m >= mt[-1] - win)
    rate = recent / win  # stamps per second over the last six hours of arrivals
    remaining = len(targets) - len(landed)
    cover = {"chunks_done": done, "chunks_total": len(js["chunks"]), "stamps_landed": len(landed),
             "rate_stamps_per_hour_last_6h": rate * 3600,
             "eta_hours": remaining / rate / 3600 if rate else None,
             "eta_local": time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + remaining / rate)) if rate else None}

    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "integrity": integ, "identity": identity, "coverage": cover}
    if not args.quick:
        out["registration"] = registration(sorted(landed), log)
        OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"{time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
