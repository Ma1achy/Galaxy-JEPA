"""Drive SciServer Compute from the devcontainer via the Jobs API.

Token comes from env SCISERVER_TOKEN (never written to disk). Two phases:
  python sciserver_driver.py discover   # dump compute domains, data volumes, images
  python sciserver_driver.py run        # submit the native-cutout test, wait, fetch stdout

`run` base64-embeds artifacts/sciserver_native_test.py into a shell-command job, mounts the
SDSS SAS data volume, runs it server-side, then downloads the job's stdout and prints it.
"""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from SciServer import Authentication, Files, Jobs

load_dotenv()  # read SCISERVER_TOKEN from the gitignored .env
TOKEN = os.environ.get("SCISERVER_TOKEN", "").strip()
_script = os.environ.get("SCISERVER_SCRIPT") or str(Path(__file__).with_name("sciserver_native_test.py"))
TEST_SCRIPT = Path(_script).read_text()


def _auth() -> None:
    if not TOKEN:
        sys.exit("SCISERVER_TOKEN env var is empty")
    Authentication.setToken(TOKEN)
    print("authenticated as:", Authentication.getKeystoneUserWithToken(TOKEN).userName)


def discover() -> None:
    _auth()
    domains = Jobs.getDockerComputeDomains()
    for d in domains:
        print("\n=== compute domain:", d.get("name"))
        print("  images:", [i.get("name") for i in d.get("dockerImages", [])])
        print("  dataVolumes:", [v.get("name") for v in d.get("volumes", d.get("dataVolumes", []))])
        print("  userVolumes:", [f"{v.get('rootVolumeName')}/{v.get('owner')}"
                                  for v in d.get("userVolumes", [])][:5])


def _pick(items, *needles):
    for it in items:
        nm = (it.get("name") or "").lower()
        if any(n in nm for n in needles):
            return it
    return items[0] if items else None


def _has_sas(domain) -> bool:
    return any((v.get("name") or "").lower() == "sdss sas" for v in domain.get("volumes", []))


def run() -> None:
    _auth()
    domains = Jobs.getDockerComputeDomains()
    # The domain must mount SDSS SAS and offer images (e.g. "Large Jobs Domain").
    # Need SDSS SAS + an image with astropy ("Astronomy").
    cands = [d for d in domains if _has_sas(d)
             and any("astro" in (i.get("name") or "").lower() for i in d.get("images", []))]
    domain = cands[0] if cands else domains[0]
    image = next((i for i in domain.get("images", []) if "astro" in (i.get("name") or "").lower()), None)
    vols = domain.get("volumes", [])
    sdss = next((v for v in vols if (v.get("name") or "").lower() == "sdss sas"), None)
    uservols = domain.get("userVolumes", [])
    # Results must live in a mounted user volume under /home/idies/workspace/. Prefer Temporary.
    tmp = next((v for v in uservols if "temporary" in (v.get("rootVolumeName") or "").lower()),
               uservols[0])
    owner, root = tmp.get("owner"), tmp.get("rootVolumeName")
    rel = f"{root}/{owner}/native_test"                       # Files-API path
    results = f"/home/idies/workspace/{rel}"                   # absolute path for the job
    print("domain:", domain.get("name"), "| image:", image and image.get("name"),
          "| sdss:", sdss and sdss.get("name"), "| results:", results)

    b64 = base64.b64encode(TEST_SCRIPT.encode()).decode()
    # tee stdout+stderr into out.txt in the results CWD so we can retrieve it via Files.
    cmd = f"echo {b64} | base64 -d > /tmp/t.py && python3 /tmp/t.py 2>&1 | tee out.txt"

    job = Jobs.submitShellCommandJob(
        cmd, dockerComputeDomain=domain, dockerImageName=image and image.get("name"),
        dataVolumes=[sdss] if sdss else None, userVolumes=uservols or None,
        resultsFolderPath=results, jobAlias="native_cutout_test",
    )
    jid = job if isinstance(job, int) else job.get("id", job)
    print("submitted job:", jid, "— waiting...")
    Jobs.waitForJob(jid, verbose=True)
    desc = Jobs.getJobDescription(jid)
    print("status:", desc.get("status"), "(32=success, 64=error)")
    for m in desc.get("messages", []):
        print("  msg:", m.get("label"), m.get("content")[:200])

    fs = Files.getFileServices(verbose=False)[0]
    for name in ("out.txt", "stdout.txt", "command.out"):
        try:
            print(f"\n===== {name} =====\n{Files.download(fs, f'{rel}/{name}', format='txt')}")
            return
        except Exception:  # noqa: BLE001 — try the next candidate filename
            continue
    print("no stdout file; dirList:")
    print(Files.dirList(fs, rel, level=2))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "discover"
    {"discover": discover, "run": run}.get(mode, discover)()
