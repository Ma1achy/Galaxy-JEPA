"""Launch a long pull fully detached from the calling shell/session.

The pull runs for hours. Anything started as a managed background task of the calling session
gets culled when that session decides the machine is under memory pressure — indiscriminately,
including trivial `sleep` loops, and regardless of how little the pull itself uses (measured:
~124 MB across its whole process tree, against 43% system memory free). Double-fork + `setsid`
puts it in its own session so it outlives the caller; progress is in the log and, structurally,
in `<corpus>.job.json`, which the driver rewrites after every chunk.

Each line is timestamped here rather than through a shell pipeline, because a pipeline is one
more process for the caller's session to take down with it.

    .venv/bin/python artifacts/_detach.py <log> -- <argv...>
"""

import os
import subprocess
import sys
import time

log_path, rest = sys.argv[1], sys.argv[2:]
if rest and rest[0] == "--":
    rest = rest[1:]

if os.fork():
    time.sleep(2)
    print(f"detached: {' '.join(rest)}\n  log: {log_path}")
    raise SystemExit(0)

os.setsid()  # own session: no controlling terminal, no process group shared with the caller
with open(log_path, "a", buffering=1) as fh:
    fh.write(f"\n=== detached launch {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    proc = subprocess.Popen(
        rest, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, text=True, bufsize=1,
    )
    fh.write(f"pid {proc.pid}\n")
    assert proc.stdout is not None
    for line in proc.stdout:
        fh.write(time.strftime("%H:%M:%S ") + line)
    proc.wait()
os._exit(0)
