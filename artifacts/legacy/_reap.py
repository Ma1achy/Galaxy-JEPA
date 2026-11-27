"""Requeue chunks whose SciServer job is dead, so a corpse cannot wedge the pull."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path("artifacts").resolve()))
from _sciserver_auth import authenticate
authenticate(verbose=False)
from SciServer import Jobs

FAILED, SUCCESS = {64, 128}, 32
STATE = Path(".sciserver_work/probe.job.json")
st = json.loads(STATE.read_text())
reaped = 0
for c in st["chunks"]:
    if c["status"] != "submitted" or not c.get("jid"):
        continue
    try:
        s = Jobs.getJobDescription(c["jid"]).get("status")
    except Exception as exc:  # noqa: BLE001
        print(f"chunk {c['k']} job {c['jid']}: {exc} -> requeue"); s = 64
    if s in FAILED:
        c.update(status="pending", jid=None, rel=None); reaped += 1
        print(f"chunk {c['k']}: job dead (status {s}) -> requeued")
    else:
        print(f"chunk {c['k']}: job {c['jid']} status {s} — kept")
STATE.write_text(json.dumps(st, indent=2))
print(f"requeued {reaped}")
