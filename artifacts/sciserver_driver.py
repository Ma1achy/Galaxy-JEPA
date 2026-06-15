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

from SciServer import Authentication, Files, Jobs

TOKEN = os.environ.get("SCISERVER_TOKEN", "").strip()
TEST_SCRIPT = Path(__file__).with_name("sciserver_native_test.py").read_text()


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


def run() -> None:
    _auth()
    domains = Jobs.getDockerComputeDomains()
    domain = _pick(domains, "small", "compute", "interactive") or domains[0]
    images = domain.get("dockerImages", [])
    image = _pick(images, "astro", "essentials", "python")
    vols = domain.get("volumes", domain.get("dataVolumes", []))
    sdss = _pick(vols, "sdss", "sas")
    uservols = domain.get("userVolumes", [])
    print("domain:", domain.get("name"), "| image:", image and image.get("name"),
          "| sdss vol:", sdss and sdss.get("name"))

    b64 = base64.b64encode(TEST_SCRIPT.encode()).decode()
    cmd = f"echo {b64} | base64 -d > /tmp/t.py && python3 /tmp/t.py"

    results = "jobsResults/native_test"
    job = Jobs.submitShellCommandJob(
        cmd, dockerComputeDomain=domain, dockerImageName=image and image.get("name"),
        dataVolumes=[sdss] if sdss else None, userVolumes=uservols or None,
        resultsFolderPath=results, jobAlias="native_cutout_test",
    )
    jid = job if isinstance(job, int) else job.get("id", job)
    print("submitted job:", jid, "— waiting...")
    Jobs.waitForJob(jid, verbose=True)
    desc = Jobs.getJobDescription(jid)
    folder = desc.get("resultsFolderPath", results)
    print("status:", desc.get("status"), "results:", folder)

    fs = Files.getFileServices(verbose=False)[0]
    for name in ("stdout.txt", "command.out", "out.txt"):
        try:
            data = Files.download(fs, f"{folder}/{name}", format="txt")
            print(f"\n===== {name} =====\n{data}")
            return
        except Exception:  # noqa: BLE001 — try the next candidate filename
            continue
    print("could not find stdout; dirList of results folder:")
    print(Files.dirList(fs, folder, level=2))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "discover"
    {"discover": discover, "run": run}.get(mode, discover)()
