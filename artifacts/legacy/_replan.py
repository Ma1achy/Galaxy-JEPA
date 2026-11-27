"""Re-chunk an in-flight pull at a new --max-per-job, keeping everything already on disk.

Chunk boundaries are offsets into the (unchanged) target list, so re-chunking is safe as long as
the number of stamps already merged divides evenly by the new size — otherwise a chunk would
straddle the done/pending boundary and either re-pull or skip galaxies.
"""

import csv
import json
import sys
from pathlib import Path

WORK = Path(".sciserver_work")
OUT = Path("data/probe")
STATE = WORK / "probe.job.json"
NEW = int([a for a in sys.argv[1:] if a.isdigit()][0])

st = json.loads(STATE.read_text())
limit = st["limit"]
n_have = sum(1 for _ in csv.DictReader((OUT / "metadata.csv").open()))
if n_have % NEW:
    sys.exit(f"{n_have} stamps on disk is not a whole number of {NEW}-target chunks — refusing")

# Cancel anything still queued/running under the old plan: its chunk boundaries are about to
# stop existing, so its output could no longer be merged at a known offset.
live = [c for c in st["chunks"] if c["status"] == "submitted" and c.get("jid")]
if live:
    sys.path.insert(0, str(Path("artifacts").resolve()))
    from _sciserver_auth import authenticate
    authenticate(verbose=False)
    from SciServer import Jobs
    for c in live:
        try:
            Jobs.cancelJob(c["jid"])
            print(f"cancelled job {c['jid']} (old chunk {c['k']})")
        except Exception as exc:  # noqa: BLE001 — a job that already ended cannot be cancelled
            print(f"job {c['jid']}: {exc}")

chunks, off, k = [], 0, 0
while off < limit:
    m = min(NEW, limit - off)
    chunks.append({"k": k, "offset": off, "n_targets": m,
                   "status": "done" if off + m <= n_have else "pending", "jid": None, "rel": None})
    off += m
    k += 1
st["max_per_job"], st["chunks"] = NEW, chunks
STATE.write_text(json.dumps(st, indent=2))
done = sum(c["status"] == "done" for c in chunks)
print(f"re-planned at {NEW}/chunk: {len(chunks)} chunks, {done} done ({n_have:,} stamps), "
      f"{len(chunks) - done} pending ({limit - n_have:,} stamps)")
