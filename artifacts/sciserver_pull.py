"""Local driver for the SciServer native-stamp corpus pull (the scalable pull path).

Runs the metadata SQL locally over the public SkyServer REST endpoint (fast, no token),
ships the target list + the server-side cutter (``sciserver_cut.py``) to SciServer Compute,
runs the cut server-side against the mounted SDSS SAS frames, and streams back a single
``corpus.tar`` per chunk that unpacks into the ``DirectorySource`` layout the rest of the
pipeline already consumes (``metadata.csv`` + ``<object_id>.fits``).

Why this and not ``data/pull.py``: the HTTP frame-download path measured ~5.2 s/galaxy
(downloads 10 MB frames over the wire); cutting server-side touches the frames locally on
SciServer and only the small stamps leave. See memory: corpus-pull-path-sciserver.

**Auth + orchestration stay here, in ``artifacts/`` — the token never enters the importable
package.** The package contributes only the *pure* pull helpers
(:func:`galaxy_jepa.data.sciserver.chunk_target_ids` /
:func:`~galaxy_jepa.data.sciserver.merge_corpora`); everything that touches the token or the
SciServer Jobs API lives in this file (memory: token-only-in-artifacts).

The token is short-lived (SSO account — see memory: sciserver-token-is-sso-and-goes-stale),
so submit and fetch are **decoupled**: the jobs run server-side independent of the local
token. A large pull is split into **chunks** (``--max-per-job``) so each SciServer job stays
under the Small-domain ~1 h timeout cap; submit fires all chunk jobs, then — after a token
refresh — fetch polls and downloads each and :func:`merge_corpora` stitches them into one
corpus. (The single-chunk path is the one exercised live; the multi-chunk loop reuses the
same per-job primitives.)

Usage (token in .env)::

    python artifacts/sciserver_pull.py --corpus probe --limit 20 --out data/probe   # full
    python artifacts/sciserver_pull.py --corpus pretrain --limit 100000 \\
        --out data/pretrain --max-per-job 12000 --mode submit
    # ... wait for the jobs, refresh SCISERVER_TOKEN in .env ...
    python artifacts/sciserver_pull.py --corpus pretrain --mode fetch
"""

from __future__ import annotations

import argparse
import base64
import csv
import itertools
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sciserver_auth import authenticate  # noqa: E402

from galaxy_jepa.data.manifest import manifest_hash  # noqa: E402
from galaxy_jepa.data.metadata import pretrain_sql, probe_sql, run_sql  # noqa: E402
from galaxy_jepa.data.pull import check_join, with_derived_columns  # noqa: E402
from galaxy_jepa.data.sciserver import (  # noqa: E402
    align_append_fieldnames,
    chunk_target_ids,
    merge_corpora,
)

CUTTER = Path(__file__).with_name("sciserver_cut.py")
WORK = Path(".sciserver_work")

# A *stalled* transfer is the failure mode the retry loop in `_fetch_one` cannot see. It
# handles a connection that drops loudly (ChunkedEncodingError), but when the file service
# stops sending without closing, `iter_content` blocks in a socket read forever and no
# exception ever reaches the `except` — the pull simply stops, alive and at 0% CPU.
# (Observed twice, ~3 h lost: socket ESTABLISHED, zero bytes for 45 min.) A stall always
# lands on a multiple of the 8 MiB read size, so a repeated byte count means nothing — it
# is not a server-side cap to route around.
#
# `socket.setdefaulttimeout` does NOT fix this: `SciServer.Files.download` calls
# `requests.get(..., stream=True)` with no timeout, and requests then explicitly applies
# `settimeout(None)` to the connection, overriding the process default. The timeout has to
# be injected where requests actually reads it — the transport adapter, which is the one
# seam every verb and Session funnels through.
_CONNECT_TIMEOUT, _READ_TIMEOUT = 30, 180  # 8 MiB lands in ~6-8 s at 1.0-1.4 MiB/s


def _install_default_timeouts() -> None:
    """Give every SciServer HTTP read a deadline, so a half-open socket raises instead of hanging."""
    import requests.adapters

    original = requests.adapters.HTTPAdapter.send

    def send(self, request, **kwargs):  # type: ignore[no-untyped-def]
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = (_CONNECT_TIMEOUT, _READ_TIMEOUT)
        return original(self, request, **kwargs)

    requests.adapters.HTTPAdapter.send = send  # type: ignore[method-assign]


_install_default_timeouts()

# SciServer job status codes. CANCELED (128) is how the Small domain reports a job that hit
# its hard ~60-min wall-clock cap — it is a TERMINAL FAILURE, not a transient: treating it as
# non-terminal made `fetch` poll a dead job until max_wait_min (10 h hang). It is terminal.
_SUCCESS = 32
_FAILED = {64, 128}  # ERROR, CANCELED/TIMEOUT
_TERMINAL = {_SUCCESS} | _FAILED

#: A graceful stop, for rotating the token without interrupting a wave. Touch
#: ``.sciserver_work/<corpus>.stop`` and the driver finishes the wave already in flight —
#: downloading and merging it — then exits without submitting more. Killing the process
#: instead would abandon a part-downloaded chunk and, worse, leave new jobs cut against a
#: token that dies before they can be fetched. Resume with the same --mode full.
def stop_path(corpus: str) -> Path:
    return WORK / f"{corpus}.stop"


def _stop_requested(corpus: str) -> bool:
    """True once, consuming the sentinel so a stale file cannot stop the next run."""
    f = stop_path(corpus)
    if f.exists():
        f.unlink(missing_ok=True)
        return True
    return False


#: The fail-safe behind the planned rotation. If a token dies mid-run anyway, an expired
#: token is neither a dead job nor a dead transfer: requeueing on it burns a bounded retry
#: pass, and across a whole wave that can end a run while every job is still alive
#: server-side. Hold position and re-read .env instead -- drop a fresh token in and the pull
#: carries on with nothing lost. Bounded, so it can never hang silently forever.
_AUTH_RETRIES = 1
_TOKEN_WAIT_H = 12.0


def _is_auth_error(exc: BaseException) -> bool:
    t = str(exc).lower()
    return "not authenticated" in t or "x-auth-token" in t or "401" in t


def _wait_for_token(corpus: str, *, budget_h: float = _TOKEN_WAIT_H) -> None:
    """Block until .env holds a valid token again, logging so a wait never looks like a stall."""
    deadline = time.time() + budget_h * 3600
    tick = 0
    while time.time() < deadline:
        try:
            authenticate(verbose=False)
            print("[full] token restored — resuming")
            return
        except SystemExit:
            pass
        if tick % 5 == 0:  # ~every 5 min: visible progress, and the log-silence watchdog stays quiet
            left = int((deadline - time.time()) / 60)
            print(f"[full] WAITING for a fresh SCISERVER_TOKEN in .env — {left} min of budget left")
        tick += 1
        time.sleep(60)
    raise SystemExit(
        f"[full] no valid token within {budget_h:g} h. Refresh SCISERVER_TOKEN in .env and "
        f"re-run --mode full; {corpus} state is checkpointed, nothing is lost."
    )


#: Retry passes over chunks a pass requeued (a dead job, a connection dropped by the
#: laptop sleeping). Bounded, and a pass that heals nothing ends the loop, so a
#: permanently-broken chunk can never spin the run forever.
_MAX_PASSES = 6


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


def _targets(corpus: str, limit: int) -> tuple[list[dict], str]:
    if corpus == "probe":
        _net_retry(check_join, limit=10)  # 10-row ra/dec guard before any bulk pull
        sql = probe_sql(limit)
    else:
        sql = pretrain_sql(limit)
    rows = _net_retry(run_sql, sql)
    if not rows:
        sys.exit(f"no rows from {corpus} SQL (limit={limit})")
    # The package's single derivation site (object_id + snr_r). sciserver_cut.py passes every
    # target column straight through to metadata.csv, so deriving here is what puts the SNR
    # nuisance in the corpus — the HTTP path in data/pull.py calls the same function.
    return with_derived_columns(rows), sql


def _write_targets(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# --- SciServer plumbing -------------------------------------------------------------


def _pick_domain(Jobs, prefer: str):
    # SciServer renamed the mount: the volume that was plain "SDSS SAS" is now published per
    # release ("SDSS SAS DR19", "SDSS SAS DR20"), so an exact match finds nothing and the pull
    # dies with "no compute domain mounts SDSS SAS". Prefix-match instead. This does NOT relax
    # the release the stamps are cut from: `sciserver_cut.py` globs `**/dr17/**/frame-r-*` and
    # exits loudly if that tree is absent, so the DR17 frames our existing corpus was cut from
    # stay the only thing that can be cut. That guard is the format-parity lock, not this name.
    def has_sas(d):
        return any((v.get("name") or "").lower().startswith("sdss sas")
                   for v in d.get("volumes", []))

    def has_astro(d):
        return any("astro" in (i.get("name") or "").lower() for i in d.get("images", []))

    cands = [d for d in safe(Jobs.getDockerComputeDomains) if has_sas(d) and has_astro(d)]
    if not cands:
        sys.exit("no compute domain mounts SDSS SAS with an Astronomy image")
    chosen = next((d for d in cands if prefer.lower() in (d.get("name") or "").lower()), cands[0])
    image = next(i for i in chosen.get("images", []) if "astro" in (i.get("name") or "").lower())
    # Lowest-numbered release first: the older mount is likeliest to still carry the dr17 tree.
    sas = sorted((v for v in chosen.get("volumes", [])
                  if (v.get("name") or "").lower().startswith("sdss sas")),
                 key=lambda v: (v.get("name") or ""))
    sdss = sas[0]
    print(f"[full] SAS volume={sdss.get('name')!r} (cutter still requires the dr17 frame tree)")
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


# --- per-chunk submit / fetch primitives -------------------------------------------


def _submit_chunk(Jobs, Files, *, corpus: str, k: int, rows: list[dict], stamp_px: int,
                  domain, image, sdss) -> dict:
    """Submit one cut job for one chunk of targets; return its resumable state record."""
    WORK.mkdir(exist_ok=True)
    targets_local = WORK / f"{corpus}_targets_{k}.csv"
    _write_targets(rows, targets_local)

    uservols = domain.get("userVolumes", [])
    tmp = next((v for v in uservols if (v.get("name") or "") == "scratch"), None) or next(
        (v for v in uservols if "temporary" in (v.get("rootVolumeName") or "").lower()),
        uservols[0],
    )
    rel = f"{tmp['rootVolumeName']}/{tmp['owner']}/{tmp['name']}/galaxy_pull_{corpus}_{k}"
    results = f"/home/idies/workspace/{rel}"

    fs = Files.getFileServices(verbose=False)[0]
    safe(Files.createDir, fs, rel, quiet=True)
    safe(Files.upload, fs, f"{rel}/targets.csv", localFilePath=str(targets_local), quiet=True)

    b64 = base64.b64encode(CUTTER.read_text().encode()).decode()
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
        jobAlias=f"galaxy_cut_{corpus}_{k}",
    )
    jid = job if isinstance(job, int) else job.get("id", job)
    print(f"[submit] chunk {k}: job {jid} ({len(rows)} targets)")
    return {"k": k, "jid": jid, "rel": rel, "n_targets": len(rows)}


def _fetch_chunk(Jobs, Files, chunk: dict, dest: Path, *, poll_s: int, max_wait_min: int) -> Path:
    """Poll one chunk's job, download its corpus.tar.gz, extract into ``dest``."""
    jid, rel = chunk["jid"], chunk["rel"]
    print(f"[fetch] chunk {chunk['k']}: polling job {jid} ...")
    t0 = time.time()
    status = None
    while (time.time() - t0) < max_wait_min * 60:
        status = safe(Jobs.getJobDescription, jid).get("status")
        if status in _TERMINAL:
            break
        time.sleep(poll_s)
    if status in _FAILED:
        msg = (safe(Jobs.getJobDescription, jid).get("messages") or [{}])[-1].get("content", "")
        raise RuntimeError(
            f"[fetch] chunk {chunk['k']} job {jid} did not succeed (status {status}: {msg!r}). "
            "A CANCELED/TIMEOUT means the cut exceeded the domain's per-job wall-clock cap — "
            "lower --max-per-job. Re-run --mode full to resume the unfinished chunks."
        )
    if status != _SUCCESS:
        raise RuntimeError(
            f"[fetch] chunk {chunk['k']} job {jid} still status {status} after {max_wait_min} "
            "min — giving up. Re-run --mode full to resume."
        )

    fs = Files.getFileServices(verbose=False)[0]
    job_rel = rel
    for _ in range(90):  # results subfolder + corpus.tar.gz can lag the status flip by minutes
        job_rel = _job_output_dir(Files, fs, rel, jid)
        if job_rel != rel and _file_size(Files, fs, job_rel, "corpus.tar.gz") > 0:
            break
        time.sleep(10)
    size = _file_size(Files, fs, job_rel, "corpus.tar.gz")
    print(f"[fetch] chunk {chunk['k']} output: {job_rel} (corpus.tar={size / 1e6:.1f} MB)")
    if size == 0:
        log = ""
        try:
            log = safe(Files.download, fs, f"{job_rel}/cut.log", format="txt")
        except Exception:  # noqa: BLE001
            pass
        sys.exit(f"[fetch] chunk {chunk['k']}: no corpus.tar produced. cut.log:\n{log[-800:]}")

    dest.mkdir(parents=True, exist_ok=True)
    local_tar = WORK / f"{chunk['k']}_corpus.tar.gz"
    # A multi-hundred-MB transfer at ~1-2 MB/s runs for minutes and does drop
    # (ChunkedEncodingError: IncompleteRead). Retry the transfer itself, and check the byte
    # count against what the file service advertised — a truncated tarball that still opens
    # would merge a short chunk and mark it done, silently losing galaxies.
    for attempt in range(4):
        try:
            res = safe(Files.download, fs, f"{job_rel}/corpus.tar.gz", format="response")
            with local_tar.open("wb") as fh:
                for piece in res.iter_content(chunk_size=8 << 20):
                    fh.write(piece)
            got = local_tar.stat().st_size
            if got != size:
                raise OSError(f"truncated transfer: {got} of {size} bytes")
            break
        except Exception as exc:  # noqa: BLE001 — transfers drop; that is what this retries
            local_tar.unlink(missing_ok=True)
            if attempt == 3:
                raise
            print(f"[fetch] chunk {chunk['k']} transfer failed ({exc}); retrying ...")
            time.sleep(10 * (attempt + 1))
    with tarfile.open(local_tar) as tar:
        tar.extractall(dest)
    local_tar.unlink(missing_ok=True)
    return dest


# --- phases -------------------------------------------------------------------------


def submit(corpus: str, limit: int, out_dir: Path, *, stamp_px: int, domain_pref: str,
           max_per_job: int) -> dict:
    from SciServer import Files, Jobs

    authenticate()
    print(f"[submit] querying {corpus} metadata (limit={limit}) over SkyServer REST ...")
    rows, sql = _targets(corpus, limit)
    ids = [int(r.get("objID", r.get("dr8objid", r.get("object_id")))) for r in rows]
    id_chunks = chunk_target_ids(ids, max_per_job)
    print(f"[submit] {len(rows)} targets → {len(id_chunks)} chunk(s) of ≤{max_per_job}")

    domain, image, sdss = _pick_domain(Jobs, domain_pref)
    print(f"[submit] domain={domain.get('name')!r} image={image.get('name')!r}")

    chunks: list[dict] = []
    offset = 0
    for k, id_chunk in enumerate(id_chunks):
        chunk_rows = rows[offset : offset + len(id_chunk)]
        offset += len(id_chunk)
        chunks.append(
            _submit_chunk(Jobs, Files, corpus=corpus, k=k, rows=chunk_rows, stamp_px=stamp_px,
                          domain=domain, image=image, sdss=sdss)
        )

    WORK.mkdir(exist_ok=True)
    state = {"corpus": corpus, "out_dir": str(out_dir), "stamp_px": stamp_px, "query": sql,
             "limit": limit, "chunks": chunks}
    _state_path(corpus).write_text(json.dumps(state, indent=2))
    print(f"[submit] {len(chunks)} job(s) submitted; state -> {_state_path(corpus)}")
    print(f"[submit] cut ~{len(rows) / 3.8 / 60:.0f} min server-side; fetch with: "
          f"python artifacts/sciserver_pull.py --corpus {corpus} --mode fetch")
    return state


def fetch(corpus: str, *, poll_s: int = 15, max_wait_min: int = 600) -> None:
    from SciServer import Files, Jobs

    sp = _state_path(corpus)
    if not sp.exists():
        sys.exit(f"no submit state at {sp}; run --mode submit first")
    state = json.loads(sp.read_text())
    out_dir = Path(state["out_dir"])
    chunks = state["chunks"]
    authenticate()

    if len(chunks) == 1:
        # Single chunk: extract straight into out_dir (identical to the original path).
        _fetch_chunk(Jobs, Files, chunks[0], out_dir, poll_s=poll_s, max_wait_min=max_wait_min)
        n_fits = len(list(out_dir.glob("*.fits")))
        print(f"[fetch] DONE — {n_fits} stamps under {out_dir} (+ metadata.csv)")
        return

    # Multiple chunks: fetch each into its own staging dir, then merge into one corpus.
    chunk_dirs: list[Path] = []
    for chunk in chunks:
        dest = WORK / f"{corpus}_chunk_{chunk['k']}"
        _fetch_chunk(Jobs, Files, chunk, dest, poll_s=poll_s, max_wait_min=max_wait_min)
        chunk_dirs.append(dest)
    print(f"[fetch] merging {len(chunk_dirs)} chunks -> {out_dir} ...")
    merge_corpora(chunk_dirs, out_dir, query=state.get("query"))
    n_fits = len(list(out_dir.glob("*.fits")))
    print(f"[fetch] DONE — {n_fits} stamps merged under {out_dir} (+ metadata.csv)")


# --- throttled wave orchestration (the scalable path) ------------------------------


def _all_targets_path(corpus: str) -> Path:
    return WORK / f"{corpus}_all_targets.csv"


def _save_state(corpus: str, state: dict) -> None:
    WORK.mkdir(exist_ok=True)
    _state_path(corpus).write_text(json.dumps(state, indent=2))


def _read_all_targets(corpus: str) -> list[dict]:
    with _all_targets_path(corpus).open(newline="") as fh:
        return list(csv.DictReader(fh))


def _read_target_slice(corpus: str, offset: int, n: int) -> list[dict]:
    """Rows ``[offset, offset+n)`` of the target list, read a row at a time.

    Not ``_read_all_targets(...)[a:b]``: at the full-GZ2 scale the target list is 230k rows x
    ~135 columns, which is several GB as dicts and OOM'd an 18 GiB machine mid-wave. A wave only
    ever needs its own chunks, so hold only those.
    """
    with _all_targets_path(corpus).open(newline="") as fh:
        return list(itertools.islice(csv.DictReader(fh), offset, offset + n))


def _plan_chunks(n_rows: int, max_per_job: int) -> list[dict]:
    """Contiguous, order-preserving chunks recorded as (offset, n_targets) into the target list."""
    chunks, off, k = [], 0, 0
    while off < n_rows:
        n = min(max_per_job, n_rows - off)
        chunks.append(
            {"k": k, "offset": off, "n_targets": n, "status": "pending", "jid": None, "rel": None}
        )
        off += n
        k += 1
    return chunks


def _load_or_plan(corpus: str, limit: int, out_dir: Path, *, stamp_px: int,
                  max_per_job: int) -> dict:
    """Resume an in-progress run, or query + chunk a fresh one (state in WORK/<corpus>.job.json)."""
    sp = _state_path(corpus)
    if sp.exists():
        st = json.loads(sp.read_text())
        if (st.get("limit") == limit and st.get("out_dir") == str(out_dir)
                and st.get("max_per_job") == max_per_job and "chunks" in st):
            done = sum(c.get("status") == "done" for c in st["chunks"])
            print(f"[full] resuming {sp}: {done}/{len(st['chunks'])} chunks already done")
            return st
        sys.exit(
            f"[full] {sp} is for a different run (limit/out/max_per_job differ). Clear it or use "
            "a fresh --out before starting a new pull."
        )
    WORK.mkdir(exist_ok=True)
    print(f"[full] querying {corpus} metadata (limit={limit}) over SkyServer REST ...")
    rows, sql = _targets(corpus, limit)
    _write_targets(rows, _all_targets_path(corpus))
    st = {"corpus": corpus, "out_dir": str(out_dir), "stamp_px": stamp_px, "query": sql,
          "limit": limit, "max_per_job": max_per_job,
          "chunks": _plan_chunks(len(rows), max_per_job)}
    _save_state(corpus, st)
    print(f"[full] {len(rows)} targets → {len(st['chunks'])} chunk(s) of ≤{max_per_job}")
    return st


def _accumulate(chunk_dir: Path, out: Path) -> int:
    """Copy a fetched chunk's stamps into ``out`` and append its metadata rows (incremental merge).

    Per-wave accumulation keeps peak disk at out_dir + one wave's staging, rather than holding
    every chunk's staging dir until a single final merge. The manifest is written once at the
    end over the full accumulated object-ID set.

    The column order comes from :func:`align_append_fieldnames`, never from the chunk -- a
    chunk-ordered row appended beneath the corpus's alphabetised header misaligns every column
    silently, which is what mislabelled 190,358 galaxies here. See that function for why.
    """
    meta = chunk_dir / "metadata.csv"
    if not meta.exists():
        raise FileNotFoundError(f"{chunk_dir} has no metadata.csv")
    with meta.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    out_meta = out / "metadata.csv"
    fieldnames, write_header = align_append_fieldnames(out_meta, fieldnames)
    with out_meta.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        if write_header:
            w.writeheader()
        for r in rows:
            oid = r["object_id"]
            src = chunk_dir / f"{oid}.fits"
            if not src.exists():
                raise FileNotFoundError(f"{chunk_dir} lists {oid} but {src} is missing")
            _relocate(src, out / f"{oid}.fits")
            w.writerow(r)
    # Force writeback before the next chunk: a wave writes several GB to an external disk, and
    # dirty pages awaiting writeback are what put an 18 GiB machine under enough memory pressure
    # to get this process killed mid-fetch.
    os.sync()
    return len(rows)


def _relocate(src: Path, dst: Path) -> None:
    """Move a stamp into the corpus — a rename when possible, a copy only across devices.

    Staging (``.sciserver_work/``) and the corpus normally sit on the same filesystem, so this
    is a metadata operation. It used to be ``shutil.copy2``, which re-read and re-wrote every
    stamp: ~1.4 GB of pointless I/O per chunk, and roughly 2.7 GB of page-cache churn that was
    the local half of the memory kills. Extraction still lands in staging first, so a chunk that
    dies mid-fetch cannot leave half its stamps in the corpus.
    """
    try:
        os.replace(src, dst)
    except OSError:  # EXDEV — staging and corpus on different filesystems
        shutil.copy2(src, dst)
        src.unlink(missing_ok=True)


def _finalize_manifest(out: Path, query: str | None) -> None:
    """Write the data_snapshot manifest over the full accumulated corpus (same hash as merge)."""
    with (out / "metadata.csv").open(newline="") as fh:
        ids = [int(r["object_id"]) for r in csv.DictReader(fh)]
    (out / "manifest.json").write_text(
        json.dumps({"data_snapshot": manifest_hash(ids, query or ""), "n": len(ids),
                    "query": query or ""}, indent=2) + "\n"
    )


def run_full(corpus: str, limit: int, out_dir: Path, *, stamp_px: int, domain_pref: str,
             max_per_job: int, max_concurrent: int, max_waves: int) -> None:
    """Throttled, resumable pull: submit chunks in waves of ``max_concurrent``, fetch+merge each
    wave before launching the next.

    The Small domain has NO concurrency cap and a hard ~60-min per-job wall-clock cap; firing
    every chunk at once oversubscribes the 32 cores so badly that even small chunks time out
    (observed: 4×12k jobs all CANCELED at 60 min). Throttling to ``max_concurrent`` keeps each
    job near the staging-measured ~2.63 gal/s so it lands under the cap. State is checkpointed
    per chunk, so a crash / token expiry / ``--max-waves`` stop resumes cleanly with --mode full.
    """
    from SciServer import Files, Jobs

    authenticate()
    state = _load_or_plan(corpus, limit, out_dir, stamp_px=stamp_px, max_per_job=max_per_job)
    out = Path(state["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    domain, image, sdss = _pick_domain(Jobs, domain_pref)
    print(f"[full] domain={domain.get('name')!r} image={image.get('name')!r}")

    n_chunks = len(state["chunks"])

    def _submit_wave(wave: list[dict], label: str) -> None:
        print(f"[full] === {label}: submit chunks {[c['k'] for c in wave]} ===")
        for c in wave:
            # A chunk already submitted has a job running server-side, independent of this
            # process and of the local token (see the module docstring). Resuming after a crash
            # or a token expiry must adopt that job, not fire a duplicate and orphan it.
            if c["status"] == "submitted" and c.get("jid"):
                print(f"[full] chunk {c['k']}: adopting live job {c['jid']} (already submitted)")
                continue
            rows = _read_target_slice(corpus, c["offset"], c["n_targets"])
            # Submitting is the third place a 401 lands, and the one that actually killed a
            # run: `_submit_chunk` asks for the file-service list before cutting, and an
            # expired token there raised straight through the wave loop and out of main().
            # Hold for a fresh token here too — the alternative is dying with 600 chunks left.
            for attempt in range(_AUTH_RETRIES + 1):
                try:
                    rec = _submit_chunk(Jobs, Files, corpus=corpus, k=c["k"], rows=rows,
                                        stamp_px=state["stamp_px"], domain=domain, image=image,
                                        sdss=sdss)
                except Exception as exc:  # noqa: BLE001
                    if _is_auth_error(exc) and attempt < _AUTH_RETRIES:
                        _wait_for_token(corpus)
                        continue
                    raise
                c.update(jid=rec["jid"], rel=rec["rel"], status="submitted")
                break
        _save_state(corpus, state)

    def _await_wave(wave: list[dict]) -> None:
        """Block until every job in ``wave`` has left the cluster (succeeded or failed).

        This is what keeps the pipelining honest: submitting the next wave the moment the
        previous one is *submitted* would put 2x max_concurrent jobs on a shared domain, and
        the contention pushes every job toward the per-job wall-clock cap. Waiting for the cut to
        finish — but not for the download — overlaps transfer with compute and nothing else.
        """
        for c in wave:
            t0 = time.time()
            while (time.time() - t0) < 90 * 60:
                try:
                    status = safe(Jobs.getJobDescription, c["jid"]).get("status")
                except Exception as exc:  # noqa: BLE001
                    if not _is_auth_error(exc):
                        raise
                    _wait_for_token(corpus)
                    continue
                if status in _TERMINAL:
                    break
                time.sleep(15)

    def _drain_wave(wave: list[dict]) -> None:
        for c in wave:
            dest = WORK / f"{corpus}_chunk_{c['k']}"
            for attempt in range(_AUTH_RETRIES + 1):
                try:
                    _fetch_chunk(Jobs, Files, {"k": c["k"], "jid": c["jid"], "rel": c["rel"]},
                                 dest, poll_s=15, max_wait_min=90)
                    n = _accumulate(dest, out)
                except Exception as exc:  # noqa: BLE001 — nothing here may abort the whole pull
                    # An expired token is neither of the two failures below: the job is alive
                    # and its output is intact, only this process cannot ask for it. Hold for a
                    # fresh one rather than requeueing, which would spend a bounded retry pass.
                    if _is_auth_error(exc) and attempt < _AUTH_RETRIES:
                        shutil.rmtree(dest, ignore_errors=True)
                        _wait_for_token(corpus)
                        continue
                    # A dead job must not wedge the run. Leaving the chunk `submitted` with its
                    # jid meant every later invocation re-adopted the corpse and aborted on it —
                    # one transient server-side failure (or a laptop sleeping mid-poll) blocked
                    # the whole pull permanently. Reset it to pending so a later pass resubmits
                    # it, and carry on with the rest of the wave.
                    if isinstance(exc, RuntimeError):  # the *job* died — it needs cutting again
                        print(f"[full] chunk {c['k']} job failed, requeued for a fresh cut: {exc}")
                        c.update(status="pending", jid=None, rel=None)
                    else:  # the *transfer* died — the job's output is still there to re-download
                        print(f"[full] chunk {c['k']} transfer failed, will retry download: {exc}")
                    _save_state(corpus, state)
                    shutil.rmtree(dest, ignore_errors=True)
                    break
                c["status"], c["n_written"] = "done", n
                _save_state(corpus, state)
                shutil.rmtree(dest, ignore_errors=True)
                print(f"[full] chunk {c['k']} merged ({n} stamps) → {out}; staging freed")
                break

    # Submit the NEXT wave before draining the current one, so the cluster is cutting while we
    # download. The two phases are independent — jobs run server-side, detached from this
    # process — but were serialised, which left the cluster idle for the whole transfer. The
    # file service moves ~1-2 MB/s, so at full-GZ2 scale that idle time is ~8 h of the run.
    # In-flight cutting jobs are still `max_concurrent`: the wave being drained has already
    # finished cutting.
    for p_no in range(_MAX_PASSES):
        pending = [c for c in state["chunks"] if c["status"] != "done"]
        before = len(pending)
        if not pending:
            break
        waves = [pending[i : i + max_concurrent] for i in range(0, before, max_concurrent)]
        print(f"[full] pass {p_no + 1}: {n_chunks} chunks, {before} pending → {len(waves)} "
              f"wave(s) of ≤{max_concurrent}"
              + (f"; running {max_waves} this invocation" if max_waves else ""))

        ran, in_flight, stopping = 0, None, False
        for w, wave in enumerate(waves):
            if max_waves and ran >= max_waves:
                print(f"[full] stopping after {ran} wave(s) (--max-waves); re-run --mode full to resume")
                break
            if _stop_requested(corpus):
                stopping = True
                print("[full] stop requested — draining the wave in flight, submitting no more")
                break
            if in_flight is not None:
                _await_wave(in_flight)  # cluster free again — never more than max_concurrent cutting
            _submit_wave(wave, f"wave {w + 1}/{len(waves)}")
            if in_flight is not None:
                _drain_wave(in_flight)  # download the finished wave while this one cuts
                ran += 1
            in_flight = wave
        if in_flight is not None and not (max_waves and ran >= max_waves):
            _drain_wave(in_flight)
            ran += 1

        if max_waves:
            break  # a bounded invocation runs its waves once and reports
        if stopping:
            print("[full] STOPPED at a wave boundary; re-run --mode full to resume")
            break

        still = [c for c in state["chunks"] if c["status"] != "done"]
        if not still or len(still) >= before:
            break  # finished, or a pass that healed nothing — don't spin
        print(f"[full] pass {p_no + 1} done; {len(still)} chunk(s) requeued, retrying")

    remaining = [c for c in state["chunks"] if c["status"] != "done"]
    if not remaining:
        _finalize_manifest(out, state.get("query"))
        n_fits = len(list(out.glob("*.fits")))
        print(f"[full] DONE — {n_fits} stamps under {out} (+ metadata.csv + manifest.json)")
    else:
        print(f"[full] PARTIAL — {len(remaining)} chunk(s) remain; re-run --mode full to resume")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="SciServer native-stamp corpus pull.")
    p.add_argument("--corpus", choices=["pretrain", "probe"], required=True)
    p.add_argument("--limit", type=int, help="required for submit/full")
    p.add_argument("--out", type=Path, help="required for submit/full")
    p.add_argument("--stamp-px", type=int, default=256)
    # Per-job size is bounded by the Small-domain ~60-min wall-clock cap at the MEASURED cut
    # rate (~2.63 gal/s under 2-way contention, not the earlier optimistic 3.8): 5000 ≈ 32 min,
    # wide margin for the unknown sustained rate over a full-length job. The old 12k default ran
    # 4 jobs straight into the cap. Raise it only on a domain with a longer cap / more cores.
    p.add_argument("--max-per-job", type=int, default=5000, help="targets per SciServer job")
    # The Small domain runs ALL submitted jobs at once (no cap) and they contend for its 32
    # cores; throttling to 2 concurrent keeps each near the measured rate. `full` honours this
    # (waves); `submit` fires all chunks at once (only safe on a capped/uncontended domain).
    p.add_argument("--max-concurrent", type=int, default=2, help="jobs per wave (full mode)")
    p.add_argument("--max-waves", type=int, default=0, help="run at most N waves then stop (0=all)")
    # Small Jobs Domain runs jobs promptly (32 cores); the Large Jobs Domain sat PENDING for
    # 20+ min for this account. Default to Small.
    p.add_argument("--domain", default="Small", help="substring of the compute domain name")
    p.add_argument("--mode", choices=["full", "submit", "fetch"], default="full")
    args = p.parse_args(argv)

    try:
        if args.mode == "full":
            if args.limit is None or args.out is None:
                p.error("--limit and --out are required for full")
            run_full(args.corpus, args.limit, args.out, stamp_px=args.stamp_px,
                     domain_pref=args.domain, max_per_job=args.max_per_job,
                     max_concurrent=args.max_concurrent, max_waves=args.max_waves)
        elif args.mode == "submit":
            if args.limit is None or args.out is None:
                p.error("--limit and --out are required for submit")
            submit(args.corpus, args.limit, args.out, stamp_px=args.stamp_px,
                   domain_pref=args.domain, max_per_job=args.max_per_job)
        elif args.mode == "fetch":
            fetch(args.corpus)
    except RuntimeError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
