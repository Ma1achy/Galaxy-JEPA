"""Shipping estimate inputs (user brief, 21 Nov 2026): the minimum cache set, and zstd ratios and speeds.

Estimate only: creates no bucket, uploads nothing, writes no sample files (samples stream into zstd's stdin and
only the compressed byte count is kept). Reads the cache and the two corpora's metadata.

    .venv/bin/python artifacts/ship_measure.py ids    # which cache rows the grid reads, by corpus and split
    .venv/bin/python artifacts/ship_measure.py zstd   # ratio and Mac time per GB, levels 3 and 19, 2 GB samples

Investigation code: terse, excluded from lint/CI. Writes docs/ship_measure.json only.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE = Path("/Volumes/X10 Pro/galaxy-jepa/runs/m_v2/cache/eab81d179579f2f083f11b2301a662db781bd395e664f5ba5d71041163d5abfd")
STAMP = 3 * 256 * 256 * 2
OUT = REPO / "docs/ship_measure.json"
SAMPLE = 2 * 1024**3
PIECE = 64 * 1024**2  # stamps.f16 is sampled as 32 pieces of 64 MiB at random stamp-aligned offsets


def ids_of(corpus: str) -> np.ndarray:
    with open(REPO / f"data/{corpus}/metadata.csv", newline="") as h:
        return np.array([int(r["object_id"]) for r in csv.DictReader(h)], dtype=np.int64)


def step_ids() -> dict:
    sys.path.insert(0, str(REPO / "src"))
    from galaxy_jepa.data.orchestrate import resolve_corpora, split_pretrain
    idx = json.loads((CACHE / "index.json").read_text())
    cache_ids = np.asarray(idx["object_ids"], dtype=np.int64)
    pre, probe = ids_of("pretrain_v2"), ids_of("probe_v2")
    sp = split_pretrain(resolve_corpora([int(i) for i in pre], [int(i) for i in probe]), seed=0, monitor_frac=0.02)
    train, monitor = np.array(sorted(sp.train), np.int64), np.array(sorted(sp.monitor), np.int64)
    sets = {"pretrain_train (A1's split)": train, "pretrain_monitor (2%)": monitor, "probe_v2 (all rows)": probe}
    incache = {k: int(np.isin(v, cache_ids).sum()) for k, v in sets.items()}
    need = np.union1d(np.union1d(train, monitor), probe)
    extra = np.setdiff1d(cache_ids, need)
    res = {"cache_stamps": int(cache_ids.size), "stamps_f16_bytes": os.path.getsize(CACHE / "stamps.f16"),
           "stamp_bytes": STAMP, "pretrain_rows": int(pre.size), "probe_rows": int(probe.size),
           "pretrain_probe_overlap": int(np.intersect1d(pre, probe).size),
           "sets": {k: {"n": int(v.size), "in_cache": incache[k], "gb": v.size * STAMP / 1e9} for k, v in sets.items()},
           "union_needed": {"n": int(need.size), "in_cache": int(np.isin(need, cache_ids).sum()), "gb": need.size * STAMP / 1e9},
           "cache_rows_not_needed": {"n": int(extra.size), "gb": extra.size * STAMP / 1e9},
           "excluded_pretrain_not_in_split": int(np.setdiff1d(pre, np.union1d(train, monitor)).size)}
    return res


def zstd_size(chunks, level: int, threads: int) -> tuple[int, float]:
    """Stream the byte chunks through `zstd -<level> -T<threads> -c`; return (compressed bytes, seconds)."""
    cmd = ["zstd", f"-{level}", f"-T{threads}", "-c", "-q"] + (["--ultra"] if level > 19 else [])
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    n_out = 0
    import threading

    def drain():
        nonlocal n_out
        while True:
            b = p.stdout.read(1 << 20)
            if not b:
                break
            n_out += len(b)
    th = threading.Thread(target=drain)
    th.start()
    for c in chunks():
        p.stdin.write(c)
    p.stdin.close()
    th.join()
    p.wait()
    return n_out, time.perf_counter() - t0


def stamp_chunks(shuffle: bool):
    n = os.path.getsize(CACHE / "stamps.f16") // STAMP
    rng = np.random.default_rng(20261121)
    per = PIECE // STAMP
    starts = np.sort(rng.choice(n - per, SAMPLE // PIECE, replace=False))
    mm = np.memmap(CACHE / "stamps.f16", dtype=np.uint8, mode="r")

    def gen():
        for s in starts:
            b = np.asarray(mm[s * STAMP:(s + per) * STAMP])
            if shuffle:  # byte planes: all high bytes of the fp16 values, then all low bytes
                b = np.ascontiguousarray(b.reshape(-1, 2).T)
            yield b.tobytes()
    return gen, per * STAMP * (SAMPLE // PIECE)


def file_chunks(path: Path):
    size = os.path.getsize(path)

    def gen():
        with open(path, "rb") as h:
            while True:
                b = h.read(1 << 24)
                if not b:
                    break
                yield b
    return gen, size


def step_zstd() -> dict:
    comps = {"stamps.f16 (32 × 64 MiB random pieces)": stamp_chunks(False),
             "stamps.f16, byte-plane shuffled (same pieces)": stamp_chunks(True),
             "probe_columns.f64 (whole)": file_chunks(CACHE / "probe_columns.f64"),
             "index.json (whole)": file_chunks(CACHE / "index.json"),
             "petro_rad_arcsec.f64 (whole)": file_chunks(CACHE / "petro_rad_arcsec.f64"),
             "pretrain_v2/metadata.csv (whole)": file_chunks(REPO / "data/pretrain_v2/metadata.csv"),
             "probe_v2/metadata.csv (whole)": file_chunks(REPO / "data/probe_v2/metadata.csv")}
    res = {"zstd": subprocess.run(["zstd", "--version"], capture_output=True, text=True).stdout.strip(),
           "threads": "-T0 (all cores)", "load_start": os.getloadavg()}
    for name, (gen, raw) in comps.items():
        r = {"raw_bytes": raw}
        for lv in (3, 19):
            if lv == 19 and "shuffled" in name:
                continue
            out, sec = zstd_size(gen, lv, 0)
            r[f"level_{lv}"] = {"ratio": raw / out, "s_per_gb": sec / (raw / 1e9), "seconds": sec}
            print(name, lv, json.dumps(r[f"level_{lv}"]), file=sys.stderr, flush=True)
        res[name] = r
    res["load_end"] = os.getloadavg()
    return res


def main() -> None:
    step = sys.argv[1]
    r = {"ids": step_ids, "zstd": step_zstd}[step]()
    allr = json.loads(OUT.read_text()) if OUT.exists() else {}
    allr[step] = r
    OUT.write_text(json.dumps(allr, indent=1, default=float))
    print(json.dumps(r, indent=1, default=float))


if __name__ == "__main__":
    main()
