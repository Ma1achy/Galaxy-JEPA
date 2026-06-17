"""Local driver for the SciServer native-stamp corpus pull (the scalable pull path).

Runs the metadata SQL locally over the public SkyServer REST endpoint (fast, no token),
ships the target list + the server-side cutter (``sciserver_cut.py``) to SciServer Compute,
runs the cut server-side against the mounted SDSS SAS frames, and streams back a single
``corpus.tar`` that unpacks into the ``DirectorySource`` layout the rest of the pipeline
already consumes (``metadata.csv`` + ``<object_id>.fits``).

Why this and not ``data/pull.py``: the HTTP frame-download path measured ~5.2 s/galaxy
(downloads 10 MB frames over the wire); cutting server-side touches the frames locally on
SciServer and only the small stamps leave. See memory: corpus-pull-path-sciserver.

The token is short-lived (SSO account — see memory: sciserver-token-is-sso-and-goes-stale),
so submit and fetch are **decoupled**: the job runs server-side independent of the local
token. A long pull (the cut is ~3.8 gal/s, so 30k ≈ 2 h) outlives one token — submit now,
refresh the token, fetch later. Every SciServer call retries with a re-auth on a transient
401. ``full`` mode does both in one process (fine for short pulls).

Usage (token in .env)::

    python artifacts/sciserver_pull.py --corpus probe --limit 20 --out data/probe   # full
    python artifacts/sciserver_pull.py --corpus pretrain --limit 30000 --out data/pretrain --mode submit
    # ... wait ~2 h, refresh SCISERVER_TOKEN in .env ...
    python artifacts/sciserver_pull.py --corpus pretrain --mode fetch
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sciserver_auth import authenticate  # noqa: E402

from galaxy_jepa.data.metadata import pretrain_sql, probe_sql, run_sql  # noqa: E402
from galaxy_jepa.data.pull import check_join  # noqa: E402

CUTTER = Path(__file__).with_name("sciserver_cut.py")
WORK = Path(".sciserver_work")

# SciServer job status codes.
_DONE = {32, 64}  # SUCCESS, ERROR


def safe(fn, *args, _tries: int = 6, **kwargs):
    """Call a SciServer function, re-authenticating on a transient error (401 mid-poll)."""
    last: Exception | None = None
    for _ in range(_tries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — SciServer raises bare Exception on HTTP errors
            last = exc
            try:
                authenticate(verbose=False)
            except SystemExit:
                raise
            time.sleep(3)
    raise last  # type: ignore[misc]


# --- target list (local, public REST) ---------------------------------------------


def _net_retry(fn, *args, _tries: int = 4, **kwargs):
    """Retry a SkyServer REST call with backoff — the endpoint occasionally times out."""
    for attempt in range(_tries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — requests timeouts/5xx are transient
            if attempt == _tries - 1:
                raise
            print(f"[submit] SkyServer call failed ({type(exc).__name__}); retrying ...")
            time.sleep(5 * (attempt + 1))


def _targets(corpus: str, limit: int) -> list[dict]:
    if corpus == "probe":
        _net_retry(check_join, limit=10)  # 10-row ra/dec guard before any bulk pull
        sql = probe_sql(limit)
    else:
        sql = pretrain_sql(limit)
    rows = _net_retry(run_sql, sql)
    if not rows:
        sys.exit(f"no rows from {corpus} SQL (limit={limit})")
    return rows


def _write_targets(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# --- SciServer plumbing -------------------------------------------------------------


def _pick_domain(Jobs, prefer: str):
    def has_sas(d):
        return any((v.get("name") or "").lower() == "sdss sas" for v in d.get("volumes", []))

    def has_astro(d):
        return any("astro" in (i.get("name") or "").lower() for i in d.get("images", []))

    cands = [d for d in safe(Jobs.getDockerComputeDomains) if has_sas(d) and has_astro(d)]
    if not cands:
        sys.exit("no compute domain mounts SDSS SAS with an Astronomy image")
    chosen = next((d for d in cands if prefer.lower() in (d.get("name") or "").lower()), cands[0])
    image = next(i for i in chosen.get("images", []) if "astro" in (i.get("name") or "").lower())
    sdss = next(v for v in chosen.get("volumes", []) if (v.get("name") or "").lower() == "sdss sas")
    return chosen, image, sdss


def _job_output_dir(Files, fs, rel: str, jid) -> str:
    """Resolve <rel>/<date>/<datetime>-<jobid>/ — each job runs in its own dated subdir.

    dirList only populates ``folders`` at level>=2 (level=1 lists files at the queried path
    but no subfolders), so every listing here passes level=2.
    """
    root = safe(Files.dirList, fs, rel, level=2)["root"]
    dates = sorted(f["name"] for f in root.get("folders", []))
    if not dates:
        return rel
    for date in reversed(dates):
        subs = safe(Files.dirList, fs, f"{rel}/{date}", level=2)["root"].get("folders", [])
        names = sorted(s["name"] for s in subs)
        match = next((n for n in names if n.endswith(f"-{jid}")), None)
        if match:
            return f"{rel}/{date}/{match}"
        if names:
            return f"{rel}/{date}/{names[-1]}"
    return f"{rel}/{dates[-1]}"


def _file_size(Files, fs, dir_rel: str, name: str) -> int:
    """Size of <dir_rel>/<name> via the Files API, or 0 if absent/unreadable.

    dirList only populates a directory's children at level>=2 (level=1 returns the node
    with empty files/folders), so query at level=2.
    """
    try:
        files = safe(Files.dirList, fs, dir_rel, level=2)["root"].get("files", [])
        return next((int(f.get("size", 0)) for f in files if f.get("name") == name), 0)
    except Exception:  # noqa: BLE001
        return 0


def _state_path(corpus: str) -> Path:
    return WORK / f"{corpus}.job.json"


# --- phases -------------------------------------------------------------------------


def submit(corpus: str, limit: int, out_dir: Path, *, stamp_px: int, domain_pref: str) -> dict:
    from SciServer import Files, Jobs

    authenticate()
    print(f"[submit] querying {corpus} metadata (limit={limit}) over SkyServer REST ...")
    rows = _targets(corpus, limit)
    print(f"[submit] {len(rows)} targets")

    WORK.mkdir(exist_ok=True)
    targets_local = WORK / f"{corpus}_targets.csv"
    _write_targets(rows, targets_local)

    domain, image, sdss = _pick_domain(Jobs, domain_pref)
    uservols = domain.get("userVolumes", [])
    tmp = next((v for v in uservols if (v.get("name") or "") == "scratch"), None) or next(
        (v for v in uservols if "temporary" in (v.get("rootVolumeName") or "").lower()),
        uservols[0],
    )
    rel = f"{tmp['rootVolumeName']}/{tmp['owner']}/{tmp['name']}/galaxy_pull_{corpus}"
    results = f"/home/idies/workspace/{rel}"
    print(f"[submit] domain={domain.get('name')!r} image={image.get('name')!r} results={results}")

    fs = Files.getFileServices(verbose=False)[0]
    safe(Files.createDir, fs, rel, quiet=True)
    safe(Files.upload, fs, f"{rel}/targets.csv", localFilePath=str(targets_local), quiet=True)
    print("[submit] uploaded targets.csv; submitting cut job ...")

    b64 = base64.b64encode(CUTTER.read_text().encode()).decode()
    # The job runs in a dated subfolder, so targets.csv (uploaded to the results root) must be
    # referenced by absolute path; outputs (cut.log, out/, corpus.tar) land in the CWD subfolder.
    cmd = (
        f"echo {b64} | base64 -d > /tmp/cut.py && "
        f"STAMP_PX={stamp_px} TARGETS_CSV={results}/targets.csv python3 /tmp/cut.py 2>&1 "
        f"| tee cut.log"
    )
    job = safe(
        Jobs.submitShellCommandJob,
        cmd,
        dockerComputeDomain=domain,
        dockerImageName=image.get("name"),
        dataVolumes=[sdss],
        userVolumes=uservols,
        resultsFolderPath=results,
        jobAlias=f"galaxy_cut_{corpus}",
    )
    jid = job if isinstance(job, int) else job.get("id", job)
    state = {
        "corpus": corpus,
        "jid": jid,
        "rel": rel,
        "out_dir": str(out_dir),
        "limit": limit,
        "stamp_px": stamp_px,
        "n_targets": len(rows),
    }
    _state_path(corpus).write_text(json.dumps(state, indent=2))
    print(f"[submit] job {jid} submitted; state -> {_state_path(corpus)}")
    print(f"[submit] cut ~{len(rows) / 3.8 / 60:.0f} min server-side; fetch with: "
          f"python artifacts/sciserver_pull.py --corpus {corpus} --mode fetch")
    return state


def fetch(corpus: str, *, poll_s: int = 15, max_wait_min: int = 600) -> None:
    from SciServer import Files, Jobs

    sp = _state_path(corpus)
    if not sp.exists():
        sys.exit(f"no submit state at {sp}; run --mode submit first")
    state = json.loads(sp.read_text())
    jid, rel, out_dir = state["jid"], state["rel"], Path(state["out_dir"])
    authenticate()

    print(f"[fetch] polling job {jid} ...")
    t0 = time.time()
    status = None
    while (time.time() - t0) < max_wait_min * 60:
        status = safe(Jobs.getJobDescription, jid).get("status")
        if status in _DONE:
            break
        time.sleep(poll_s)
    print(f"[fetch] job {jid} status={status} (32=ok,64=err) after {time.time() - t0:.0f}s")
    if status == 64:
        sys.exit("[fetch] job ERRORED server-side — check cut.log in the SciServer workspace")

    fs = Files.getFileServices(verbose=False)[0]
    # The job marks SUCCESS before its results subfolder + corpus.tar sync to the Files API,
    # so poll until corpus.tar actually appears (avoids downloading an empty fallback path).
    # The results subfolder can lag the status flip by several minutes, so poll patiently.
    job_rel = rel
    for _ in range(90):  # up to ~15 min
        job_rel = _job_output_dir(Files, fs, rel, jid)
        if job_rel != rel and _file_size(Files, fs, job_rel, "corpus.tar.gz") > 0:
            break
        time.sleep(10)
    size = _file_size(Files, fs, job_rel, "corpus.tar.gz")
    print(f"[fetch] job output dir: {job_rel} (corpus.tar={size / 1e6:.1f} MB)")
    if size == 0:
        log = ""
        try:
            log = safe(Files.download, fs, f"{job_rel}/cut.log", format="txt")
        except Exception:  # noqa: BLE001
            pass
        sys.exit(f"[fetch] no corpus.tar produced. cut.log:\n{log[-800:]}")
    try:
        log = safe(Files.download, fs, f"{job_rel}/cut.log", format="txt")
        print("----- cut.log (tail) -----\n" + "\n".join(log.splitlines()[-12:]))
    except Exception as exc:  # noqa: BLE001
        print(f"[fetch] could not read cut.log: {exc}")

    out_dir.mkdir(parents=True, exist_ok=True)
    local_tar = WORK / f"{corpus}_corpus.tar.gz"
    print(f"[fetch] streaming corpus.tar.gz -> {local_tar} ...")
    t1 = time.time()
    res = safe(Files.download, fs, f"{job_rel}/corpus.tar.gz", format="response")
    n = 0
    with local_tar.open("wb") as fh:
        for chunk in res.iter_content(chunk_size=8 << 20):
            fh.write(chunk)
            n += len(chunk)
    dl_s = time.time() - t1
    print(f"[fetch] downloaded {n / 1e6:.1f} MB in {dl_s:.0f}s ({n / 1e6 / max(dl_s, 1e-3):.1f} MB/s)")
    with tarfile.open(local_tar) as tar:
        tar.extractall(out_dir)
    local_tar.unlink(missing_ok=True)  # reclaim the archive (9.5 GB at 30k) — extracted now
    n_fits = len(list(out_dir.glob("*.fits")))
    print(f"[fetch] DONE — {n_fits} stamps under {out_dir} (+ metadata.csv); archive removed")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="SciServer native-stamp corpus pull.")
    p.add_argument("--corpus", choices=["pretrain", "probe"], required=True)
    p.add_argument("--limit", type=int, help="required for submit/full")
    p.add_argument("--out", type=Path, help="required for submit/full")
    p.add_argument("--stamp-px", type=int, default=256)
    # Small Jobs Domain actually runs jobs promptly (32 cores, proven by native_test); the
    # Large Jobs Domain sat PENDING for 20+ min for this account. Default to Small.
    p.add_argument("--domain", default="Small", help="substring of the compute domain name")
    p.add_argument("--mode", choices=["full", "submit", "fetch"], default="full")
    args = p.parse_args(argv)

    if args.mode in ("full", "submit"):
        if args.limit is None or args.out is None:
            p.error("--limit and --out are required for submit/full")
        submit(args.corpus, args.limit, args.out, stamp_px=args.stamp_px, domain_pref=args.domain)
    if args.mode in ("full", "fetch"):
        fetch(args.corpus)


if __name__ == "__main__":
    main()
