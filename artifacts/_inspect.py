import json
import os

from dotenv import load_dotenv

load_dotenv("/workspaces/Galaxy-JEPA/.env")
from SciServer import Authentication, Jobs  # noqa: E402

Authentication.setToken(os.environ["SCISERVER_TOKEN"])

# Which domains pair SDSS SAS with an Astronomy image?
print("=== domains: SAS + Astronomy image? ===")
for d in Jobs.getDockerComputeDomains():
    vols = [v.get("name") for v in d.get("volumes", [])]
    imgs = [i.get("name") for i in d.get("images", [])]
    has_sas = "SDSS SAS" in vols
    astro = [i for i in imgs if "astro" in i.lower()]
    if has_sas:
        print(f"  {d.get('name')!r}: SAS=yes astro_imgs={astro} n_imgs={len(imgs)}")

# The failed job's real status + results location + any message.
print("\n=== job 474307 description ===")
desc = Jobs.getJobDescription(474307)
for k in ("status", "resultsFolderPath", "command", "dockerImageName", "submitterDID", "messages"):
    print(f"  {k}: {json.dumps(desc.get(k))[:300]}")
