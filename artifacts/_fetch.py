import os

from dotenv import load_dotenv

load_dotenv("/workspaces/Galaxy-JEPA/.env")
from SciServer import Authentication, Files, Jobs  # noqa: E402

Authentication.setToken(os.environ["SCISERVER_TOKEN"])

# Most recent native_cutout_test job.
jobs = [j for j in Jobs.getJobsList(top=20) if j.get("submitterDID") == "native_cutout_test"]
job = sorted(jobs, key=lambda j: j.get("id", 0))[-1]
jid = job.get("id")
desc = Jobs.getJobDescription(jid)
print("job", jid, "status", desc.get("status"), "resultsFolderPath:", desc.get("resultsFolderPath"))

fs = Files.getFileServices(verbose=False)[0]
rel = "Temporary/k24085112/native_test"
print("\n=== dirList", rel, "===")
try:
    listing = Files.dirList(fs, rel, level=3)
    import json
    print(json.dumps(listing, indent=1)[:1500])
except Exception as e:
    print("dirList failed:", str(e)[:200])

# Try to download every plausible output file.
for name in ("out.txt", "stdout", "stderr", "command.out", "command.err",
             "notebook.out", "jobInfo.json"):
    try:
        data = Files.download(fs, f"{rel}/{name}", format="txt")
        print(f"\n===== {name} ({len(data)} chars) =====\n{data}")
    except Exception:
        pass
