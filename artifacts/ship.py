"""Ship the fp16 cache to Cloudflare R2, and pull it onto the rented box (docs/shipping_plan.md; user's decisions of 25 Nov 2026).

    python artifacts/ship.py plan                               # what would go, no network
    python artifacts/ship.py selftest                           # chunk → planes → zstd → back, and a whole push/pull of a toy cache through a local "bucket"
    python artifacts/ship.py push --only 0 --no-cap             # one chunk, uncapped: the real R2 rate
    python artifacts/ship.py testpull --only 0                  # that chunk back, with the read-only token: hashes checked, nothing kept
    python artifacts/ship.py push                               # everything missing; safe to stop and rerun (6 Mbit/s 08:00–23:00, uncapped overnight)
    python artifacts/ship.py pull --manifest-sha1 SHA --dest DIR [--data DIR] [--workers 8]     # on the box

One chunk is staged at a time: read 5,461 stamps, split the fp16 byte planes, zstd, upload, check, delete. An interruption costs at most the chunk in flight.
The manifest goes last and its SHA-1 is printed: the box is given that SHA-1 by hand, and refuses anything else.

Credentials, in .env only (never in the package, never printed): R2_ENDPOINT (https://<account>.r2.cloudflarestorage.com), and
R2_RW_ACCESS_KEY_ID / R2_RW_SECRET_ACCESS_KEY (Mac: push) or R2_RO_ACCESS_KEY_ID / R2_RO_SECRET_ACCESS_KEY (box: pull, testpull).
The bucket's name is not a secret and is not in .env: gj-cache-<first 16 hex of the pipeline hash>. Investigation code: terse.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PH = "eab81d179579f2f083f11b2301a662db781bd395e664f5ba5d71041163d5abfd"
CACHE = REPO / "runs/m_v2/cache" / PH
STAMP = 3 * 256 * 256 * 2
CHUNK = 5461  # stamps: 2.147 GB
BLOCK = 160  # stamps per byte-plane block: 63 MB, the only raw data held in memory
LEVEL, LEVEL_SMALL = 3, 19
STATE = Path("/Volumes/X10 Pro/galaxy-jepa/runs/almagest/ship") / PH[:16]
BUCKET = f"gj-cache-{PH[:16]}"
V1 = Path("/Volumes/X10 Pro/galaxy-jepa/runs/full/cache")
SWAP_FLOOR_MB = 300
PAUSED = 75


def say(*a) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def sha1_file(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


# --- byte planes and zstd: block by block, so a chunk never sits in memory ----------------------------------------------
def split(raw: bytes) -> bytes:
    a = np.frombuffer(raw, np.uint8)
    return a[1::2].tobytes() + a[0::2].tobytes()  # little-endian fp16: high bytes, then low


def merge(pl: bytes) -> bytes:
    a = np.frombuffer(pl, np.uint8)
    h, out = len(a) // 2, np.empty(len(a), np.uint8)
    out[1::2], out[0::2] = a[:h], a[h:]
    return out.tobytes()


def pack(src: Path, off: int, size: int, dst: Path, *, planes: bool, level: int) -> dict:
    h, left, t0 = hashlib.sha1(), size, time.time()
    with open(src, "rb") as f, open(dst, "wb") as o:
        z = subprocess.Popen(["zstd", "-q", f"-{level}", "-T2", "-c"], stdin=subprocess.PIPE, stdout=o)
        f.seek(off)
        while left:
            b = f.read(min(BLOCK * STAMP, left))
            if not b:
                raise SystemExit(f"{src.name}: short read")
            h.update(b)
            z.stdin.write(split(b) if planes else b)
            left -= len(b)
        z.stdin.close()
        if z.wait():
            raise SystemExit("zstd failed")
    return {"raw_size": size, "raw_sha1": h.hexdigest(), "comp_size": dst.stat().st_size, "comp_sha1": sha1_file(dst), "planes": planes, "level": level, "pack_s": round(time.time() - t0, 1)}


def unpack(comp: Path, rec: dict, out, off: int = 0) -> None:
    """Decompress into the open binary file `out` at `off` (None: check only); the raw SHA-1 must match."""
    if comp.stat().st_size != rec["comp_size"] or sha1_file(comp) != rec["comp_sha1"]:
        raise ValueError("compressed SHA-1")
    h, n = hashlib.sha1(), 0
    z = subprocess.Popen(["zstd", "-q", "-d", "-c", str(comp)], stdout=subprocess.PIPE)
    try:
        if out is not None:
            out.seek(off)
        while n < rec["raw_size"]:
            want = min(BLOCK * STAMP, rec["raw_size"] - n)
            b = z.stdout.read(want)
            if len(b) != want:
                raise ValueError("short decompressed stream")
            b = merge(b) if rec["planes"] else b
            h.update(b)
            n += len(b)
            if out is not None:
                out.write(b)
        if z.stdout.read(1) or z.wait():
            raise ValueError("trailing data or zstd failure")
    except Exception:
        z.kill()
        z.wait()
        raise
    finally:
        if z.stdout is not None:
            z.stdout.close()
    if h.hexdigest() != rec["raw_sha1"]:
        raise ValueError("raw SHA-1")


# --- rclone: the remote comes from environment variables, so no rclone.conf ever holds a key ------------------------------
def dotenv() -> dict[str, str]:
    f = REPO / ".env"
    out = {}
    for ln in f.read_text().splitlines() if f.exists() else []:
        if "=" in ln and not ln.lstrip().startswith("#"):
            k, v = ln.split("=", 1)
            out[k.strip()] = v.strip().strip("'\"")
    return out


class Remote:
    def __init__(self, role: str, local: Path | None = None):
        self.local, self.env, self.hide = local, dict(os.environ), []
        if local:
            self.base = str(local)
            return
        e = dotenv()
        ks = ["R2_ENDPOINT", f"R2_{role}_ACCESS_KEY_ID", f"R2_{role}_SECRET_ACCESS_KEY"]
        missing = [k for k in ks if not e.get(k)]
        if missing:
            raise SystemExit("ship: .env has no " + ", ".join(missing))
        self.hide = [e[k] for k in ks]
        self.env |= {"RCLONE_CONFIG_R2_TYPE": "s3", "RCLONE_CONFIG_R2_PROVIDER": "Cloudflare", "RCLONE_CONFIG_R2_ENDPOINT": e[ks[0]].rstrip("/"),
                     "RCLONE_CONFIG_R2_ACCESS_KEY_ID": e[ks[1]], "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": e[ks[2]], "RCLONE_CONFIG_R2_REGION": "auto",
                     "RCLONE_CONFIG_R2_NO_CHECK_BUCKET": "true", "RCLONE_CONFIG_R2_ACL": "private", "RCLONE_CONFIG": "/dev/null"}
        self.base = f"R2:{BUCKET}"

    def clean(self, s: str) -> str:
        for v in self.hide:
            s = s.replace(v, "<hidden>").replace(v.replace("https://", "").rstrip("/"), "<hidden>")
        return s

    def run(self, *args, tick=None) -> str:
        """rclone with JSON stats on stderr; `tick(bytes, speed)` is called as they arrive."""
        p = subprocess.Popen(["rclone", *args, "--use-json-log", "--stats", "10s", "--stats-log-level", "NOTICE", "--log-level", "NOTICE", "--retries", "10", "--low-level-retries", "50",
                              "--contimeout", "30s", "--timeout", "2m", "--s3-chunk-size", "16M", "--s3-upload-concurrency", "4"],
                             env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        err = []

        def read():
            for ln in p.stderr:
                try:
                    d = json.loads(ln)
                except ValueError:
                    d = {"msg": ln}
                if "stats" in d:
                    if tick:
                        tick(d["stats"].get("bytes", 0), d["stats"].get("speed", 0.0))
                else:
                    err.append(self.clean(str(d.get("msg", "")))[:300])

        th = threading.Thread(target=read, daemon=True)
        th.start()
        out = p.stdout.read()
        rc = p.wait()
        th.join()
        if rc:
            raise RuntimeError(f"rclone {args[0]} exited {rc}: " + " | ".join(err[-3:]))
        return out

    def put(self, f: Path, key: str, cap: list[str], tick=None) -> None:
        self.run("copyto", str(f), f"{self.base}/{key}", "--ignore-times", *cap, tick=tick)
        md5 = hashlib.md5()
        with open(f, "rb") as fh:
            while b := fh.read(1 << 22):
                md5.update(b)
        st = json.loads(self.run("lsjson", f"{self.base}/{key}", "--stat", "--hash"))
        if st["Size"] != f.stat().st_size or st.get("Hashes", {}).get("md5") != md5.hexdigest():
            raise RuntimeError(f"{key}: the bucket's size or MD5 is not the file's")

    def get(self, key: str, f: Path) -> None:
        self.run("copyto", f"{self.base}/{key}", str(f), "--ignore-times", "--multi-thread-streams", "4")


# --- what goes -----------------------------------------------------------------------------------------------------------
def items(cache: Path, chunk: int, extras: list[tuple[str, Path]]) -> list[dict]:
    n = (cache / "stamps.f16").stat().st_size // STAMP
    out = [{"key": f"chunk-{k:05d}.f16.zst", "kind": "chunk", "src": str(cache / "stamps.f16"), "first": s, "n": min(chunk, n - s), "off": s * STAMP, "size": min(chunk, n - s) * STAMP}
           for k, s in enumerate(range(0, n, chunk))]
    for f in ("index.json", "probe_columns.f64", "petro_rad_arcsec.f64"):
        if (cache / f).exists():
            out.append({"key": f"cache/{f}.zst", "kind": "cache", "src": str(cache / f), "off": 0, "size": (cache / f).stat().st_size})
    for rel, p in extras:
        out.append({"key": f"{rel}.zst", "kind": rel.split("/")[0], "src": str(p), "off": 0, "size": p.stat().st_size})
    return out


def extras() -> list[tuple[str, Path]]:
    out = []
    for c in ("pretrain_v2", "probe_v2"):
        for f in ("metadata.csv", "manifest.json"):
            p = REPO / "data" / c / f
            if p.exists():
                out.append((f"data/{c}/{f}", p))
    # v1's split inputs (user, 24 Nov: they stay in the plan). Stamps not shipped; the identical-split probes run on the Mac
    for d in sorted(V1.glob("*")) if V1.exists() else []:
        out += [(f"v1/{d.name}/{f}", d / f) for f in ("index.json", "probe_columns.f64") if (d / f).exists()]
    if (REPO / "data/probe/metadata.csv").exists():
        out.append(("v1/probe/metadata.csv", REPO / "data/probe/metadata.csv"))
    return out


def load_state(st: Path) -> dict[str, dict]:
    f = st / "state.jsonl"
    out = {}
    for ln in f.read_text().splitlines() if f.exists() else []:
        try:
            r = json.loads(ln)
            out[r["key"]] = r
        except ValueError:  # a line cut short by a kill: that chunk is sent again
            pass
    return out


def report(**kw) -> None:
    """The status line's run file (claude-statusline v1): numbers and a name only."""
    d = Path(os.path.expanduser(os.environ.get("SL_RUNS_DIR", "~/.claude/runs")))
    try:
        d.mkdir(parents=True, exist_ok=True)
        f = d / "r2-upload.json"
        try:
            r = json.loads(f.read_text())
        except (OSError, ValueError):
            r = {}
        r.update(kw, v=1, name="R2 upload", updated=int(time.time() * 1000), pid=os.getpid())
        r.setdefault("started", int(time.time() * 1000))
        t = d / f".r2-upload.{os.getpid()}.tmp"
        t.write_text(json.dumps(r))
        os.replace(t, f)
    except OSError:
        pass


def swap_guard() -> None:
    """Stop (paused, resumable) if swap is allocated (total > 0) and free swap stays under 300 MB for three readings 10 s apart."""
    low = 0
    while True:
        try:
            out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
            mt = re.search(r"total = ([\d.]+)M", out)
            mf = re.search(r"free = ([\d.]+)M", out)
            low = low + 1 if mt and mf and float(mt.group(1)) > 0 and float(mf.group(1)) < SWAP_FLOOR_MB else 0
        except OSError:
            return
        if low >= 3:
            say("swap guard: free swap under 300 MB; stopping, resumable")
            report(status="paused", message="swap guard")
            subprocess.run(["pkill", "-P", str(os.getpid())])
            os._exit(PAUSED)
        time.sleep(10)


def cap_args(a) -> list[str]:
    if a.no_cap:
        return []
    night = f"{int(a.night_mbit * 125_000)}B" if a.night_mbit else "off"  # user, 26 Nov: leave the uplink room for other use
    return ["--bwlimit", f"{a.day_start},{int(a.day_mbit * 125_000)}B {a.day_end},{night}"]


def push(a, cache=CACHE, st=STATE, remote: Remote | None = None, chunk=CHUNK, ex=None) -> str | None:
    """Send what is missing; returns the manifest SHA-1 once everything is up."""
    st.mkdir(parents=True, exist_ok=True)
    lock = open(st / "lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("ship: another push is running")
    r = remote or Remote("RW")
    todo = items(cache, chunk, extras() if ex is None else ex)
    if a.only is not None:
        todo = [i for i in todo if i["key"] == f"chunk-{a.only:05d}.f16.zst"]
    done = load_state(st)
    sig = {"stamps_size": (cache / "stamps.f16").stat().st_size, "stamps_mtime_ns": (cache / "stamps.f16").stat().st_mtime_ns}
    sf = st / "source.json"
    if sf.exists() and json.loads(sf.read_text()) != sig:
        raise SystemExit("ship: stamps.f16 has changed since the upload began; refusing to mix two caches")
    sf.write_text(json.dumps(sig))
    if not r.local and sys.platform == "darwin":
        subprocess.Popen(["caffeinate", "-ims", "-w", str(os.getpid())])
        threading.Thread(target=swap_guard, daemon=True).start()
    tot = sum(i["size"] for i in todo)
    got = sum(i["size"] for i in todo if i["key"] in done)
    t_all, sent = time.time(), 0
    say(f"{len(todo)} files, {tot / 1e9:.1f} GB raw; {got / 1e9:.1f} GB already up; cap {cap_args(a) or 'none'}")
    tmp = st / "staging"
    tmp.mkdir(exist_ok=True)
    for old in tmp.iterdir():  # this program's own half-made chunk from a killed run
        old.unlink()
    for k, i in enumerate(todo):
        if i["key"] in done:
            continue
        f = tmp / i["key"].replace("/", "__")
        rec = {**{x: i[x] for x in ("key", "kind", "first", "n") if x in i}, **pack(Path(i["src"]), i["off"], i["size"], f, planes=i["kind"] == "chunk", level=LEVEL if i["kind"] == "chunk" else LEVEL_SMALL)}
        fails = 0
        while True:
            t0 = time.time()

            def tick(b, sp, c=rec["comp_size"], g=got, sz=i["size"]):
                report(status="running", percent=100 * (g + sz * min(1, b / max(c, 1))) / tot, message=f"{k + 1}/{len(todo)} {sp * 8 / 1e6:.1f} Mbit/s")

            try:
                r.put(f, i["key"], cap_args(a), tick)
                break
            except RuntimeError as e:  # a dropped connection: wait and send this chunk again; a refusal (403, 404) stops
                fails += 1
                say(f"{i['key']}: {e}")
                if fails >= a.max_fails or re.search(r"AccessDenied|Forbidden|NoSuchBucket|InvalidAccessKeyId|SignatureDoesNotMatch|Unauthorized|\b40[134]\b", str(e)):
                    report(status="failed", message="upload stopped; see log")
                    raise SystemExit(f"ship: stopped at {i['key']} (rerun to resume)")
                time.sleep(min(600, 15 * 2**fails))
        dt = max(time.time() - t0, 1e-6)
        f.unlink()
        rec |= {"upload_s": round(dt, 1), "mbit_s": round(rec["comp_size"] * 8 / dt / 1e6, 2), "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        with open(st / "state.jsonl", "a") as s:
            s.write("\n" + json.dumps(rec) + "\n")
            s.flush()
            os.fsync(s.fileno())
        done[i["key"]] = rec
        got, sent = got + i["size"], sent + rec["comp_size"]
        eta = (tot - got) / max(1, got) * sum(done[j["key"]]["upload_s"] for j in todo if j["key"] in done)
        say(f"{k + 1}/{len(todo)} {i['key']}: {rec['raw_size'] / 1e6:.0f} → {rec['comp_size'] / 1e6:.0f} MB (×{rec['raw_size'] / rec['comp_size']:.3f}), packed {rec['pack_s']} s, "
            f"sent {rec['upload_s']} s = {rec['mbit_s']} Mbit/s; {got / tot:.1%} up, about {eta / 3600:.1f} h left at the mean so far")
    if (cache / "stamps.f16").stat().st_mtime_ns != sig["stamps_mtime_ns"]:
        raise SystemExit("ship: stamps.f16 changed during the upload")
    if a.only is not None:
        report(status="done")
        return None
    ix = json.loads((cache / "index.json").read_text())
    man = {"version": 1, "pipeline_hash": cache.name, "bucket": BUCKET, "n_stamps": sig["stamps_size"] // STAMP, "stamp_bytes": STAMP, "chunk_stamps": chunk,
           "index": {x: ix[x] for x in ("pipeline_hash", "normalisation_hash", "scalars_sha256", "probe_columns_sha256") if x in ix},
           "files": [{x: done[i["key"]][x] for x in ("key", "kind", "first", "n", "raw_size", "raw_sha1", "comp_size", "comp_sha1", "planes", "level") if x in done[i["key"]]} for i in todo],
           "ship_sha1": sha1_file(Path(__file__)), "rclone": subprocess.run(["rclone", "version"], capture_output=True, text=True).stdout.splitlines()[0],
           "zstd": subprocess.run(["zstd", "--version"], capture_output=True, text=True).stdout.strip(), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    mf = st / "manifest.json"
    if mf.exists():  # written once: a second finished run returns the same manifest
        old = json.loads(mf.read_text())
        if old["files"] != man["files"]:
            raise SystemExit("ship: manifest.json exists and names other files; refusing to rewrite it")
    else:
        mf.write_text(json.dumps(man, indent=1))
    r.put(mf, "manifest.json", [])
    report(status="done", percent=100)
    say(f"COMPLETE. {sent / 1e9:.1f} GB sent this run in {(time.time() - t_all) / 3600:.1f} h. Manifest SHA-1 (give this to the box by hand): {sha1_file(mf)}")
    return sha1_file(mf)


# --- pull ----------------------------------------------------------------------------------------------------------------
def fetch(job) -> str:
    rec, dst, off, stage, local = job
    r = Remote("RO", local)
    f = Path(stage) / rec["key"].replace("/", "__")
    why = ""
    for _ in range(2):  # re-fetched once, then the pull stops
        try:
            r.get(rec["key"], f)
            if dst is None:
                unpack(f, rec, None)
            else:
                Path(dst).parent.mkdir(parents=True, exist_ok=True)
                with open(dst, "r+b" if rec["kind"] == "chunk" else "wb") as o:
                    unpack(f, rec, o, off)
                    o.flush()
                    os.fsync(o.fileno())
            f.unlink()
            return rec["key"]
        except (ValueError, RuntimeError) as e:
            why = str(e)
            f.unlink(missing_ok=True)
    raise RuntimeError(f"{rec['key']} failed twice: {why}")


def pull(a, local: Path | None = None, parity: bool = True) -> None:
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    stage = dest / "_staging"
    stage.mkdir(exist_ok=True)
    mf = dest / "ship_manifest.json"
    Remote("RO", local).get("manifest.json", mf)
    if sha1_file(mf) != a.manifest_sha1:
        raise SystemExit(f"ship: the bucket's manifest is {sha1_file(mf)[:12]}, not the {a.manifest_sha1[:12]} given by hand. Stop.")
    man = json.loads(mf.read_text())
    cache = dest / man["pipeline_hash"]
    cache.mkdir(exist_ok=True)
    (cache / "cache_verified.json").unlink(missing_ok=True)  # a pull that stops half-way must not leave an old pass behind
    stamps, data = cache / "stamps.f16", Path(a.data or REPO)
    size = man["n_stamps"] * man["stamp_bytes"]
    if not stamps.exists():
        with open(stamps, "wb") as o:
            o.truncate(size)
    if stamps.stat().st_size != size:
        raise SystemExit("ship: stamps.f16 on disk has the wrong size")
    row = 0
    for c in (f for f in man["files"] if f["kind"] == "chunk"):
        if c["first"] != row or c["raw_size"] != c["n"] * man["stamp_bytes"]:
            raise SystemExit("ship: the manifest's chunks don't tile the stamp file")
        row += c["n"]
    if row != man["n_stamps"]:
        raise SystemExit("ship: the manifest's chunks don't cover the stamp file")
    sf = dest / "pull_state.json"
    done = set(json.loads(sf.read_text())["done"]) if sf.exists() and json.loads(sf.read_text())["manifest"] == a.manifest_sha1 else set()
    jobs = []
    for f in man["files"]:
        if f["key"] in done:
            continue
        name = f["key"][: -len(".zst")]
        dst = stamps if f["kind"] == "chunk" else cache / Path(name).name if f["kind"] == "cache" else data / name if f["kind"] == "data" else dest / name
        jobs.append((f, str(dst), f.get("first", 0) * man["stamp_bytes"], str(stage), local))
    say(f"{len(jobs)} of {len(man['files'])} files to pull into {dest}")
    t0, n = time.time(), 0
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for key in pool.imap_unordered(fetch, jobs):
            done.add(key)
            n += next(f["comp_size"] for f in man["files"] if f["key"] == key)
            t = sf.with_suffix(".tmp")
            t.write_text(json.dumps({"manifest": a.manifest_sha1, "done": sorted(done)}))
            os.replace(t, sf)
            say(f"{len(done)}/{len(man['files'])} {key}; {n * 8 / 1e6 / max(time.time() - t0, 1e-6):.0f} Mbit/s")
    if parity:
        sys.path.insert(0, str(REPO / "artifacts"))
        os.environ |= {"F0_CACHE_BASE": str(dest)}
        os.environ.setdefault("GJ_CONFIG", "configs/pretrain_v2.yaml")
        os.environ.setdefault("F0_NORM_PREFIX", "246d8de6ba9a")
        from f0_preconditions import check
        _, c = check(verbose=False)  # opens <dest>/<the hash this code resolves>, so a different pipeline finds no cache
        if c.cache_dir.name != man["pipeline_hash"] or c.index.n != man["n_stamps"] or c.index.normalisation_hash != man["index"]["normalisation_hash"]:
            raise SystemExit("ship: pipeline parity failed")
        _ = c.scalars  # rechecks its own SHA-256
    (cache / "cache_verified.json").write_text(json.dumps({"manifest_sha1": a.manifest_sha1, "pipeline_hash": man["pipeline_hash"], "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "files": len(done)}))
    shutil.rmtree(stage)
    say(f"VERIFIED: {len(done)} files, every SHA-1 passed; {cache / 'cache_verified.json'}")


def testpull(a) -> None:
    key = f"chunk-{a.only or 0:05d}.f16.zst"
    rec = load_state(STATE)[key]
    with tempfile.TemporaryDirectory(dir=STATE) as t:
        t0 = time.time()
        fetch((rec, None, 0, t, None))
        dt = time.time() - t0
    say(f"{key}: pulled with the read-only token and verified to the raw SHA-1; {rec['comp_size'] * 8 / dt / 1e6:.1f} Mbit/s including the checks")


def plan(a) -> None:
    it, done = items(CACHE, CHUNK, extras()), load_state(STATE)
    ch = [i for i in it if i["kind"] == "chunk"]
    print(f"bucket {BUCKET}; {len(ch)} chunks, {sum(i['size'] for i in ch) / 1e9:.2f} GB; {len(it) - len(ch)} small files, {sum(i['size'] for i in it if i['kind'] != 'chunk') / 1e9:.2f} GB raw; {len(done)} already up")
    for i in it:
        if i["kind"] != "chunk":
            print(f"  {i['key']}  {i['size'] / 1e6:.1f} MB")
    print("cap:", cap_args(a) or "none")


def selftest(a) -> None:
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(dir=STATE.parent if STATE.parent.exists() else None) as t:
        t = Path(t)
        # 1. real stamps: pack → unpack, raw SHA-1 and the bytes themselves
        rec = pack(CACHE / "stamps.f16", 1234 * STAMP, 333 * STAMP, t / "c.zst", planes=True, level=LEVEL)
        with open(t / "back", "wb") as o:
            unpack(t / "c.zst", rec, o)
        with open(CACHE / "stamps.f16", "rb") as f:
            f.seek(1234 * STAMP)
            assert f.read(333 * STAMP) == (t / "back").read_bytes()
        print(f"333 real stamps: ×{rec['raw_size'] / rec['comp_size']:.3f}, {rec['raw_size'] / 1e6 / max(rec['pack_s'], .1):.0f} MB/s, bytes identical")
        for what in ("comp", "raw"):  # planted corruption must be refused
            bad = dict(rec)
            if what == "comp":
                b = bytearray((t / "c.zst").read_bytes())
                b[len(b) // 2] ^= 1
                (t / "d.zst").write_bytes(b)
            else:
                shutil.copy(t / "c.zst", t / "d.zst")
                bad["raw_sha1"] = "0" * 40
            try:
                unpack(t / "d.zst", bad, None)
                raise AssertionError(f"{what} corruption accepted")
            except ValueError as e:
                print(f"planted {what} corruption refused: {e}")
        # 2. a toy cache through a local "bucket": push in two goes (one chunk, then the rest, resumed), pull, refuse a wrong manifest SHA-1, refuse a tampered chunk
        c = t / "cache" / PH
        c.mkdir(parents=True)
        (c / "stamps.f16").write_bytes(rng.integers(0, 256, 25 * STAMP, dtype=np.uint8).tobytes())
        (c / "index.json").write_text(json.dumps({"pipeline_hash": PH, "normalisation_hash": "n"}))
        (c / "probe_columns.f64").write_bytes(rng.random(1000).tobytes())
        (t / "m.csv").write_text("id\n1\n")
        b, st = t / "bucket", t / "state"
        b.mkdir()
        ns = argparse.Namespace(only=1, no_cap=True, max_fails=2, day_mbit=6, night_mbit=0.0, day_start="08:00", day_end="23:00")
        ex = [("data/x/metadata.csv", t / "m.csv")]
        push(ns, c, st, Remote("RW", b), 10, ex)
        with open(st / "state.jsonl", "a") as f:
            f.write('{"key": "chunk-00002')  # a kill mid-write
        ns.only = None
        n0 = len(load_state(st))
        assert n0 == 1  # the truncated line is not a record
        sha = push(ns, c, st, Remote("RW", b), 10, ex)
        assert len(load_state(st)) == 6  # and it did not swallow the record after it
        assert push(ns, c, st, Remote("RW", b), 10, ex) == sha and not list((st / "staging").iterdir())
        pa = argparse.Namespace(dest=str(t / "box"), data=str(t / "box"), workers=2, manifest_sha1=sha)
        pull(pa, b, parity=False)
        assert (t / "box" / PH / "stamps.f16").read_bytes() == (c / "stamps.f16").read_bytes() and (t / "box/data/x/metadata.csv").read_text() == "id\n1\n"
        assert (t / "box" / PH / "probe_columns.f64").read_bytes() == (c / "probe_columns.f64").read_bytes()
        assert json.loads((t / "box" / PH / "cache_verified.json").read_text())["manifest_sha1"] == sha
        for name, fn in (("wrong manifest SHA-1", lambda: pull(argparse.Namespace(**{**vars(pa), "dest": str(t / "b2"), "manifest_sha1": "0" * 40}), b, parity=False)),):
            try:
                fn()
                raise AssertionError(name)
            except SystemExit as e:
                print(f"{name} refused: {e}")
        x = bytearray((b / "chunk-00001.f16.zst").read_bytes())
        x[100] ^= 1
        (b / "chunk-00001.f16.zst").write_bytes(x)
        try:
            pull(argparse.Namespace(**{**vars(pa), "dest": str(t / "b3"), "data": str(t / "b3")}), b, parity=False)
            raise AssertionError("tampered chunk accepted")
        except RuntimeError as e:
            assert not (t / "b3" / PH / "cache_verified.json").exists()
            print(f"tampered chunk refused: {e}; no cache_verified.json left")
        (c / "stamps.f16").write_bytes(bytes(25 * STAMP))
        try:
            push(ns, c, st, Remote("RW", b), 10, ex)
            raise AssertionError("changed source accepted")
        except SystemExit as e:
            print(f"changed source refused: {e}")
    print("SELFTEST PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "selftest", "push", "testpull", "pull"])
    ap.add_argument("--only", type=int, help="push/testpull: this chunk only")
    ap.add_argument("--no-cap", action="store_true")
    ap.add_argument("--day-mbit", type=float, default=6.0)
    ap.add_argument("--night-mbit", type=float, default=0.0, help="cap outside the day window too (0 = uncapped)")
    ap.add_argument("--day-start", default="08:00")
    ap.add_argument("--day-end", default="23:00")
    ap.add_argument("--max-fails", type=int, default=20)
    ap.add_argument("--manifest-sha1")
    ap.add_argument("--dest")
    ap.add_argument("--data", help="pull: where data/<corpus>/ goes (default: the repository)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.mode == "pull" and not (a.manifest_sha1 and a.dest):
        raise SystemExit("pull needs --manifest-sha1 (by hand) and --dest")
    globals()[a.mode](a)


if __name__ == "__main__":
    main()
