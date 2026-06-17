"""Local driver for the SciServer native-stamp corpus pull (the scalable pull path).

Runs the metadata SQL locally over the public SkyServer REST endpoint (fast, no token),
ships the target list + the server-side cutter (``sciserver_cut.py``) to SciServer Compute,
runs the cut server-side against the mounted SDSS SAS frames, and streams back a single
``corpus.tar`` that unpacks into the ``DirectorySource`` layout the rest of the pipeline
already consumes (``metadata.csv`` + ``<object_id>.fits``).

Why this and not ``data/pull.py``: the HTTP frame-download path measured ~5.2 s/galaxy
(downloads 10 MB frames over the wire); cutting server-side touches the frames locally on
SciServer and only the small stamps leave. See memory: corpus-pull-path-sciserver.

Usage (token in .env; see _sciserver_auth)::

    python artifacts/sciserver_pull.py --corpus probe   --limit 20   --out data/probe     # smoke
    python artifacts/sciserver_pull.py --corpus pretrain --limit 30000 --out data/pretrain
"""

from __future__ import annotations

import argparse
import base64
import csv
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sciserver_auth import authenticate  # noqa: E402

from galaxy_jepa.data.metadata import (  # noqa: E402
    pretrain_sql,
    probe_sql,
    run_sql,
)
from galaxy_jepa.data.pull import check_join  # noqa: E402

CUTTER = Path(__file__).with_name("sciserver_cut.py")


def _targets(corpus: str, limit: int) -> list[dict]:
    if corpus == "probe":
        check_join(limit=10)  # 10-row ra/dec guard before any bulk pull
        sql = probe_sql(limit)
    else:
        sql = pretrain_sql(limit)
    rows = run_sql(sql)
    if not rows:
        sys.exit(f"no rows from {corpus} SQL (limit={limit})")
    return rows


def _write_targets(rows: list[dict], path: Path) -> None:
    cols = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _job_output_dir(files_mod, fs, rel: str, jid) -> str:
    """Resolve <rel>/<date>/<datetime>-<jobid>/ — SciServer runs each job in a dated subdir.

    Falls back to the newest date/subfolder if no folder matches the job id exactly.
    """
    dates = sorted(f["name"] for f in files_mod.dirList(fs, rel)["root"].get("folders", []))
    if not dates:
        return rel  # nothing nested — use the root
    for date in reversed(dates):
        subs = files_mod.dirList(fs, f"{rel}/{date}")["root"].get("folders", [])
        names = sorted(s["name"] for s in subs)
        match = next((n for n in names if n.endswith(f"-{jid}")), None)
        if match:
            return f"{rel}/{date}/{match}"
        if names:  # newest date, newest run
            return f"{rel}/{date}/{names[-1]}"
    return f"{rel}/{dates[-1]}"


def _pick_domain(jobs_mod, prefer: str):
    domains = jobs_mod.getDockerComputeDomains()

    def has_sas(d):
        return any((v.get("name") or "").lower() == "sdss sas" for v in d.get("volumes", []))

    def has_astro(d):
        return any("astro" in (i.get("name") or "").lower() for i in d.get("images", []))

    cands = [d for d in domains if has_sas(d) and has_astro(d)]
    if not cands:
        sys.exit("no compute domain mounts SDSS SAS with an Astronomy image")
    chosen = next((d for d in cands if prefer.lower() in (d.get("name") or "").lower()), cands[0])
    image = next(i for i in chosen.get("images", []) if "astro" in (i.get("name") or "").lower())
    sdss = next(v for v in chosen.get("volumes", []) if (v.get("name") or "").lower() == "sdss sas")
    return chosen, image, sdss


def pull(corpus: str, limit: int, out_dir: Path, *, stamp_px: int, domain_pref: str) -> None:
    from SciServer import Files, Jobs

    authenticate()

    print(f"[pull] querying {corpus} metadata (limit={limit}) over SkyServer REST ...")
    rows = _targets(corpus, limit)
    print(f"[pull] {len(rows)} targets")

    work = Path(".sciserver_work")
    work.mkdir(exist_ok=True)
    targets_local = work / "targets.csv"
    _write_targets(rows, targets_local)

    domain, image, sdss = _pick_domain(Jobs, domain_pref)
    uservols = domain.get("userVolumes", [])
    tmp = next(
        (v for v in uservols if "temporary" in (v.get("rootVolumeName") or "").lower()),
        uservols[0],
    )
    owner, root = tmp.get("owner"), tmp.get("rootVolumeName")
    rel = f"{root}/{owner}/galaxy_pull_{corpus}"
    results = f"/home/idies/workspace/{rel}"
    print(
        f"[pull] domain={domain.get('name')!r} image={image.get('name')!r} "
        f"results={results}"
    )

    fs = Files.getFileServices(verbose=False)[0]
    Files.createDir(fs, rel, quiet=True)
    Files.upload(fs, f"{rel}/targets.csv", localFilePath=str(targets_local), quiet=True)
    print("[pull] uploaded targets.csv; submitting cut job ...")

    b64 = base64.b64encode(CUTTER.read_text().encode()).decode()
    cmd = (
        f"echo {b64} | base64 -d > /tmp/cut.py && "
        f"STAMP_PX={stamp_px} python3 /tmp/cut.py 2>&1 | tee cut.log"
    )
    job = Jobs.submitShellCommandJob(
        cmd,
        dockerComputeDomain=domain,
        dockerImageName=image.get("name"),
        dataVolumes=[sdss],
        userVolumes=uservols,
        resultsFolderPath=results,
        jobAlias=f"galaxy_cut_{corpus}",
    )
    jid = job if isinstance(job, int) else job.get("id", job)
    print(f"[pull] job {jid} submitted; waiting ...")
    t0 = time.time()
    Jobs.waitForJob(jid, verbose=True)
    desc = Jobs.getJobDescription(jid)
    print(f"[pull] job {jid} status={desc.get('status')} (32=ok,64=err) in {time.time() - t0:.0f}s")

    # The job runs in a per-job subfolder <results>/<date>/<datetime>-<jobid>/, NOT the
    # results root — outputs (cut.log, corpus.tar) land there.
    job_rel = _job_output_dir(Files, fs, rel, jid)
    print(f"[pull] job output dir: {job_rel}")

    try:
        print("\n----- cut.log (tail) -----")
        log = Files.download(fs, f"{job_rel}/cut.log", format="txt")
        print("\n".join(log.splitlines()[-12:]))
    except Exception as exc:  # noqa: BLE001
        print(f"[pull] could not read cut.log: {exc}")

    # Stream corpus.tar back (chunked — Files.download(localFilePath) buffers in RAM).
    out_dir.mkdir(parents=True, exist_ok=True)
    local_tar = work / f"{corpus}_corpus.tar"
    print(f"[pull] downloading corpus.tar -> {local_tar} ...")
    res = Files.download(fs, f"{job_rel}/corpus.tar", format="response")
    n = 0
    with local_tar.open("wb") as fh:
        for chunk in res.iter_content(chunk_size=8 << 20):
            fh.write(chunk)
            n += len(chunk)
    print(f"[pull] downloaded {n / 1e6:.1f} MB; extracting to {out_dir} ...")
    with tarfile.open(local_tar) as tar:
        tar.extractall(out_dir)
    n_fits = len(list(out_dir.glob("*.fits")))
    print(f"[pull] DONE — {n_fits} stamps under {out_dir} (+ metadata.csv)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="SciServer native-stamp corpus pull.")
    p.add_argument("--corpus", choices=["pretrain", "probe"], required=True)
    p.add_argument("--limit", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stamp-px", type=int, default=256)
    p.add_argument("--domain", default="Large", help="substring of the compute domain name")
    args = p.parse_args(argv)
    pull(args.corpus, args.limit, args.out, stamp_px=args.stamp_px, domain_pref=args.domain)


if __name__ == "__main__":
    main()
