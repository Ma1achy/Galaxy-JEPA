"""Launch a long pull fully detached from the calling shell/session.

The pull runs for hours. Anything started as a managed background task of the calling session
gets culled when that session decides the machine is under memory pressure — indiscriminately,
including trivial `sleep` loops, and regardless of how little the pull itself uses (measured:
~124 MB across its whole process tree, against 43% system memory free). Double-fork + `setsid`
puts it in its own session so it outlives the caller; progress is in the log and, structurally,
in `<corpus>.job.json`, which the driver rewrites after every chunk.

Each line is timestamped here rather than through a shell pipeline, because a pipeline is one
more process for the caller's session to take down with it.

    .venv/bin/python artifacts/_detach.py <log> [watch flags...] -- <argv...>

The job also gets a status-line bar with no sidecar to start (claude-statusline's `--watch`, when
~/.claude/statusline.js is installed): named after the log unless `--name` is among the watch flags,
which pass through as given (`--stage RE --stages N --inner TPL` for a chain). This process is the
job's parent, so it writes the true end, done or failed by exit code, once the job exits.
"""

import os
import shutil
import subprocess
import sys
import time

log_path, rest = sys.argv[1], sys.argv[2:]
watch = rest[:rest.index("--")] if "--" in rest else []
rest = rest[len(watch) + 1:] if "--" in rest else rest
SL = os.path.expanduser("~/.claude/statusline.js")
name = os.path.splitext(os.path.basename(log_path))[0]
if "--name" in watch:
    i = watch.index("--name")
    name, watch = watch[i + 1], watch[:i] + watch[i + 2:]

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
    node = shutil.which("node") or "/usr/local/bin/node"
    watcher = None
    if os.path.exists(SL) and os.path.exists(node):
        watcher = subprocess.Popen([node, SL, "--watch", os.path.abspath(log_path), "--name", name,
                                    "--pid", str(proc.pid), *watch],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert proc.stdout is not None
    for line in proc.stdout:
        fh.write(time.strftime("%H:%M:%S ") + line)
    rc = proc.wait()
    if watcher is not None:
        try:  # it sees the pid go and takes a last reading itself
            watcher.wait(timeout=30)
        except subprocess.TimeoutExpired:
            watcher.terminate()
        subprocess.run([node, SL, "--run", name, "--status", "done" if rc == 0 else "failed",
                        "--message", f"exit {rc}"], stdin=subprocess.DEVNULL, capture_output=True)
os._exit(0)
