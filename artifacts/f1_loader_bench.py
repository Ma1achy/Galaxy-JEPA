"""Brief F1 — what stamps/s can the pipeline deliver? Measured alone, no model.

The cache is 415.8 GB on a USB SSD. If the loader starves the device then every F2 number
measures the drive, not the machine — and the M3/M5 comparison would be void without saying so.
So this runs first and in isolation, in two layers:

**A. device layer** — random and sequential 393,216-byte reads straight out of ``stamps.f16``
with ``F_NOCACHE`` set, so the page cache is bypassed and the figure is the drive's, not RAM's.
Run against the external cache and against a copy on the internal SSD, which is the ceiling
reference. Without ``F_NOCACHE`` an internal subset small enough to fit the disk also fits RAM,
and the "internal SSD" number would silently be a memory-bandwidth number.

**B. pipeline layer** — the real ``TensorCache`` → ``StampDataset`` → ``DataLoader`` at the real
batch size, over the real pretraining train split, shuffled (the training access pattern) and
sequential (the contrast), with ``num_workers`` swept — the production loader
(``harness._prepare``) passes none, so it runs single-process.

**Each point runs in its own subprocess.** The production dataset is 4.07 GB resident on this
corpus (``rows_by_id`` over both metadata tables), so measurements taken in one process
accumulate until the machine dies — it did. A subprocess per point returns memory to zero
between them, and the affordability of ``num_workers`` is projected from the *measured* RSS of
the single-process point rather than by serialising the dataset to find out how big it is.

Investigation code: terse, excluded from lint/CI.

    uv run python artifacts/f1_loader_bench.py device
    uv run python artifacts/f1_loader_bench.py loader
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f0_preconditions import check  # noqa: E402

BYTES_PER_STAMP = 3 * 256 * 256 * 2
OUT = Path(__file__).resolve().parent / "out"
INTERNAL_SUBSET = Path.home() / ".cache" / "galaxy-jepa-f1" / "subset.f16"
SUBSET_STAMPS = 24_000
DEVICE_SECONDS = 45
WORKER_SWEEP = (0, 2, 4, 8)
HEADLINE_SECONDS = 130  # > 2 min on the production configuration
SWEEP_SECONDS = 55
WINDOW = 10.0
#: Projected (parent + workers) dataset residency above this share of RAM is not attempted.
#: Under ``spawn`` — the macOS default — every worker gets its own copy of the dataset. Two
#: ceilings: what each copy *retains* for the whole run, and the transient build *peak* they hit
#: together at startup. Both must clear, because getting this wrong kills the machine — it did.
RSS_GUARD = 0.40
RSS_GUARD_PEAK = 0.65


def rss_bytes() -> int:
    """Peak RSS of this process. macOS reports ``ru_maxrss`` in bytes."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def child_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)


def current_rss_bytes() -> int:
    """Resident *now*, not the high-water mark — what each worker actually holds for the run.

    ``ru_maxrss`` only ever rises, and the dataset build churns a lot of transient CSV garbage,
    so guarding on the peak alone would refuse a worker count that genuinely fits.
    """
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                         capture_output=True, text=True).stdout.strip()
    return int(out or 0) * 1024


def _memsize() -> int:
    return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                              text=True).stdout or 0)


def _machine() -> dict:
    def sysctl(k: str) -> str:
        return subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()

    return {
        "cpu": sysctl("machdep.cpu.brand_string"),
        "ram_bytes": int(sysctl("hw.memsize") or 0),
        "ncpu": int(sysctl("hw.ncpu") or 0),
        "perf_cores": int(sysctl("hw.perflevel0.logicalcpu") or 0),
        "eff_cores": int(sysctl("hw.perflevel1.logicalcpu") or 0),
        "os": platform.mac_ver()[0] or platform.platform(),
        "torch": torch.__version__,
        "mps": bool(torch.backends.mps.is_available()),
        "start_method": __import__("multiprocessing").get_start_method(),
    }


# --- A. device layer ---------------------------------------------------------------------


def _uncached_fd(path: Path) -> int:
    fd = os.open(path, os.O_RDONLY)
    fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)  # macOS: do not populate the page cache
    return fd


def device_read(path: Path, *, shuffled: bool, seconds: int, seed: int = 0) -> dict:
    size = path.stat().st_size
    n = size // BYTES_PER_STAMP
    rng = np.random.default_rng(seed)
    fd = _uncached_fd(path)
    windows: list[float] = []
    try:
        t0 = last = time.perf_counter()
        reads = wreads = 0
        cursor = int(rng.integers(0, n))
        while True:
            row = int(rng.integers(0, n)) if shuffled else (cursor + reads) % n
            got = os.pread(fd, BYTES_PER_STAMP, row * BYTES_PER_STAMP)
            if len(got) != BYTES_PER_STAMP:
                raise RuntimeError(f"short read at row {row}")
            reads += 1
            wreads += 1
            now = time.perf_counter()
            if now - last >= WINDOW:
                windows.append(wreads / (now - last))
                last, wreads = now, 0
            if now - t0 >= seconds:
                break
    finally:
        os.close(fd)
    elapsed = time.perf_counter() - t0
    return {
        "path": str(path), "access": "shuffled" if shuffled else "sequential",
        "n_stamps_in_file": n, "reads": reads, "seconds": elapsed,
        "stamps_per_s": reads / elapsed,
        "mb_per_s": reads * BYTES_PER_STAMP / elapsed / 1e6,
        "windows_stamps_per_s": [round(w, 1) for w in windows],
    }


def ensure_internal_subset(src: Path) -> Path | None:
    want = SUBSET_STAMPS * BYTES_PER_STAMP
    if INTERNAL_SUBSET.exists() and INTERNAL_SUBSET.stat().st_size >= want:
        return INTERNAL_SUBSET
    st = os.statvfs(Path.home())
    if st.f_bavail * st.f_frsize < want * 1.5:
        print(f"F1: internal SSD too full for a {want/1e9:.1f} GB reference — skipping")
        return None
    INTERNAL_SUBSET.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(src, "rb") as fin, open(INTERNAL_SUBSET, "wb") as fout:
        left = want
        while left > 0:
            chunk = fin.read(min(1 << 26, left))
            if not chunk:
                break
            fout.write(chunk)
            left -= len(chunk)
    dt = time.perf_counter() - t0
    print(f"F1: staged {want/1e9:.1f} GB to internal in {dt:.0f}s ({want/dt/1e6:.0f} MB/s copy)")
    return INTERNAL_SUBSET


def part_a(cache_dir: Path) -> list[dict]:
    ext = cache_dir / "stamps.f16"
    rows = []
    for drive, path in (("external (USB SSD)", ext),
                        ("internal (ceiling reference)", ensure_internal_subset(ext))):
        if path is None:
            continue
        for shuffled in (True, False):
            r = device_read(path, shuffled=shuffled, seconds=DEVICE_SECONDS)
            r["drive"] = drive
            rows.append(r)
            print(f"  {drive.split()[0]:<9} {r['access']:<10} {r['stamps_per_s']:7.1f} stamps/s  "
                  f"{r['mb_per_s']:7.1f} MB/s   windows={r['windows_stamps_per_s']}")
    return rows


# --- B. pipeline layer: one subprocess per point -------------------------------------------


def _split_ids(cfg) -> tuple[list[int], list[int], list[int]]:
    """Pretrain train ids + both corpora's ids, read without building full row dicts."""
    from galaxy_jepa.data.orchestrate import resolve_corpora, split_pretrain

    def ids(root: str) -> list[int]:
        with open(Path(root) / "metadata.csv", newline="") as fh:
            return [int(r["object_id"]) for r in csv.DictReader(fh)]

    pre_ids, probe_ids = ids(cfg.paths.pretrain_dir), ids(cfg.paths.probe_dir)
    split = split_pretrain(resolve_corpora(pre_ids, probe_ids), seed=cfg.seed,
                           monitor_frac=cfg.monitor_frac)
    return sorted(split.train), sorted(split.monitor), probe_ids


def build_production_dataset(cfg, cache):
    """Exactly what ``harness._prepare`` builds — the whole metadata table for both corpora."""
    from galaxy_jepa.data.dataset import StampDataset, rows_by_id
    from galaxy_jepa.data.orchestrate import resolve_corpora, split_pretrain
    from galaxy_jepa.data.sources import DirectorySource

    t0 = time.perf_counter()
    pre, probe = DirectorySource(cfg.paths.pretrain_dir), DirectorySource(cfg.paths.probe_dir)
    pre_ids = [int(r["object_id"]) for r in pre.rows]
    probe_ids = [int(r["object_id"]) for r in probe.rows]
    split = split_pretrain(resolve_corpora(pre_ids, probe_ids), seed=cfg.seed,
                           monitor_frac=cfg.monitor_frac)
    rows = rows_by_id([*pre.rows, *probe.rows])
    del pre, probe
    gc.collect()
    ds = StampDataset(cache, rows, sorted(split.train))
    return ds, sorted(split.monitor), {"setup_seconds": time.perf_counter() - t0,
                                       "n_train": len(ds), "n_rows": len(rows),
                                       "n_monitor": len(split.monitor)}


def build_lean_dataset(cfg, cache):
    """The same dataset carrying only the column ``__getitem__`` reads: ``petroRad_r``.

    ``harness._prepare`` hands ``StampDataset`` both metadata tables in full — 4.07 GB resident
    on this corpus — and ``__getitem__`` reads one key out of them. Under ``spawn`` that table is
    copied into every worker, which was believed to be what makes ``num_workers`` unaffordable.
    **It is not** — Brief G3 re-ran this sweep against the 8.46 MB scalar sidecar and 2/4/8 workers
    still drive the machine into swap, with each worker holding 0.01 GB. Measured as a
    separate variant, not fixed here: it says what the drive would give if the dataset were
    worker-shaped, which is the number the run plan needs.
    """
    from galaxy_jepa.data.dataset import StampDataset

    t0 = time.perf_counter()
    lean: dict[int, dict[str, float]] = {}
    for root in (cfg.paths.pretrain_dir, cfg.paths.probe_dir):
        with open(Path(root) / "metadata.csv", newline="") as fh:
            for r in csv.DictReader(fh):
                v = r.get("petroRad_r", "")
                lean[int(r["object_id"])] = {
                    "petroRad_r": float(v) if v not in ("", "nan", None) else float("nan")
                }
    train, monitor, _ = _split_ids(cfg)
    gc.collect()
    ds = StampDataset(cache, lean, train)
    return ds, monitor, {"setup_seconds": time.perf_counter() - t0, "n_train": len(ds),
                         "n_rows": len(lean), "n_monitor": len(monitor)}


BUILDERS = {"production": build_production_dataset, "lean": build_lean_dataset}


def loader_run(ds, *, batch_size: int, workers: int, shuffle: bool, seconds: float) -> dict:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False,
                        num_workers=workers, persistent_workers=False, pin_memory=False)
    it = iter(loader)
    t_first = time.perf_counter()
    first = next(it)
    startup = time.perf_counter() - t_first
    assert first["image"].shape[0] == batch_size
    windows: list[float] = []
    t0 = last = time.perf_counter()
    stamps = wstamps = 0
    for batch in it:
        k = batch["image"].shape[0]
        stamps += k
        wstamps += k
        now = time.perf_counter()
        if now - last >= WINDOW:
            windows.append(wstamps / (now - last))
            last, wstamps = now, 0
        if now - t0 >= seconds:
            break
    elapsed = time.perf_counter() - t0
    del it, loader
    return {
        "batch_size": batch_size, "num_workers": workers,
        "access": "shuffled" if shuffle else "sequential",
        "startup_to_first_batch_s": startup, "stamps": stamps, "seconds": elapsed,
        "stamps_per_s": stamps / elapsed, "steps_per_s": stamps / elapsed / batch_size,
        "mb_per_s": stamps * BYTES_PER_STAMP / elapsed / 1e6,
        "windows_stamps_per_s": [round(w, 1) for w in windows],
    }


def run_point(args) -> None:
    """One measurement, in its own process. Prints a single ``RESULT <json>`` line."""
    cfg, cache = check(verbose=False)
    base = rss_bytes()
    ds, _monitor, meta = BUILDERS[args.variant](cfg, cache)
    gc.collect()
    retained, peak = current_rss_bytes(), rss_bytes()
    copies = args.workers + 1
    ram = _memsize()
    over = (retained * copies > RSS_GUARD * ram) or (peak * copies > RSS_GUARD_PEAK * ram)
    if over:
        print("RESULT " + json.dumps({
            "variant": args.variant, "num_workers": args.workers, "access": args.access,
            "skipped": True, "reason": "projected dataset residency exceeds the RSS guard",
            "dataset_rss_bytes": retained, "build_peak_rss_bytes": peak,
            "projected_bytes": retained * copies, "projected_peak_bytes": peak * copies,
            "guard_bytes": int(RSS_GUARD * ram), "guard_peak_bytes": int(RSS_GUARD_PEAK * ram),
            **meta}))
        return
    dataset_rss = retained
    r = loader_run(ds, batch_size=cfg.objective.batch_size, workers=args.workers,
                   shuffle=args.access == "shuffled", seconds=args.seconds)
    r.update({"variant": args.variant, "skipped": False, "import_rss_bytes": base,
              "dataset_rss_bytes": dataset_rss, "build_peak_rss_bytes": peak,
              # file-backed memmap pages land in RSS as the loader touches them; they are clean
              # and evictable, so this figure is not a memory *cost*, only a footprint
              "peak_rss_bytes": rss_bytes(), "peak_worker_rss_bytes": child_rss_bytes(), **meta})
    print("RESULT " + json.dumps(r))


def _spawn_point(variant: str, workers: int, access: str, seconds: float) -> dict:
    cmd = [sys.executable, __file__, "point", "--variant", variant, "--workers", str(workers),
           "--access", access, "--seconds", str(seconds)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
    if line is None:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-6:]
        return {"variant": variant, "num_workers": workers, "access": access, "failed": True,
                "returncode": proc.returncode, "tail": tail}
    return json.loads(line[len("RESULT "):])


def _say(r: dict) -> None:
    tag = f"{r.get('variant','?'):<10} w={r.get('num_workers','?'):<2} {r.get('access','')[:4]:<4}"
    if r.get("failed"):
        print(f"  {tag} FAILED rc={r['returncode']}: {r['tail'][-1] if r['tail'] else ''}")
    elif r.get("skipped"):
        print(f"  {tag} SKIPPED — {r['dataset_rss_bytes']/1e9:.2f} GB retained "
              f"(peak {r['build_peak_rss_bytes']/1e9:.2f}) x {r['num_workers']+1} = "
              f"{r['projected_bytes']/1e9:.1f}/{r['projected_peak_bytes']/1e9:.1f} GB vs guard "
              f"{r['guard_bytes']/1e9:.1f}/{r['guard_peak_bytes']/1e9:.1f}. That cost IS "
              f"the finding.")
    else:
        print(f"  {tag} {r['stamps_per_s']:7.1f} stamps/s  {r['mb_per_s']:7.1f} MB/s  "
              f"first batch {r['startup_to_first_batch_s']:5.1f}s  "
              f"rss {r['dataset_rss_bytes']/1e9:.2f}->{r['peak_rss_bytes']/1e9:.2f} GB  "
              f"windows={r['windows_stamps_per_s']}")


def part_b() -> dict:
    runs: list[dict] = []
    for variant in ("production", "lean"):
        for workers in WORKER_SWEEP:
            secs = HEADLINE_SECONDS if workers == 0 else SWEEP_SECONDS
            r = _spawn_point(variant, workers, "shuffled", secs)
            runs.append(r)
            _say(r)
    live = [r for r in runs if not r.get("skipped") and not r.get("failed")]
    best = max(live, key=lambda r: r["stamps_per_s"]) if live else None
    for variant, workers in dict.fromkeys(
        [("production", 0)] + ([(best["variant"], best["num_workers"])] if best else [])
    ):
        r = _spawn_point(variant, workers, "sequential", SWEEP_SECONDS)
        runs.append(r)
        _say(r)
    return {"runs": runs, "rss_guard_fraction": RSS_GUARD,
            "best": None if best is None else {"variant": best["variant"],
                                               "workers": best["num_workers"],
                                               "stamps_per_s": best["stamps_per_s"]}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["device", "loader", "all", "point"])
    ap.add_argument("--variant", choices=list(BUILDERS), default="production")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--access", choices=["shuffled", "sequential"], default="shuffled")
    ap.add_argument("--seconds", type=float, default=SWEEP_SECONDS)
    args = ap.parse_args()
    if args.mode == "point":
        run_point(args)
        return

    cfg, cache = check()
    OUT.mkdir(parents=True, exist_ok=True)
    result: dict = {"machine": _machine(), "mode": args.mode,
                    "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "cache": {"dir": str(cache.cache_dir), "n": cache.index.n,
                              "shape": list(cache.index.shape),
                              "bytes": cache.index.n * BYTES_PER_STAMP,
                              "normalisation_hash": cache.index.normalisation_hash},
                    "batch_size": cfg.objective.batch_size}
    if args.mode in ("device", "all"):
        print("\nF1-A  device layer (F_NOCACHE — page cache bypassed)")
        result["device"] = part_a(cache.cache_dir)
    if args.mode in ("loader", "all"):
        print(f"\nF1-B  pipeline layer (one subprocess per point; guard {RSS_GUARD:.0%} of "
              f"{_memsize()/1e9:.1f} GB)")
        del cache
        gc.collect()
        result["loader"] = part_b()
    tag = _machine()["cpu"].replace(" ", "-").lower() or "unknown"
    path = OUT / f"f1_{tag}_{args.mode}.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
