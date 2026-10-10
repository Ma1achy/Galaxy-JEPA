"""Report a detached job's progress to the status line (claude-statusline's run-progress API, v1).

The jobs here write logs, not progress files, and a running script is never edited (bash reads it
incrementally). So this sidecar tails the job's log every few seconds and writes
`~/.claude/runs/<slug>.json` atomically, with the job's own pid (a dead job reads `stopped?`). It
exits when the job reports done or failed, or its pid is gone.

    artifacts/run_report.py sae   <name> <log> <pid> [--epochs 8]   # dd_sae.train epoch lines
    artifacts/run_report.py chain <name> <log> <pid> --steps N      # chain.sh '=== step' / 'exit' lines

In chain mode the current step's own log (`log_<step>.txt` beside the chain log) feeds the inner
progress when it carries SAE epoch lines.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import tempfile
import time
from pathlib import Path

RUNS = Path(os.path.expanduser(os.environ.get("SL_RUNS_DIR", "~/.claude/runs")))
EPOCH = re.compile(r"×(\d+) epoch (\d+): (\{.*\})")
STEP, EXIT = re.compile(r"=== (\S+)"), re.compile(r"\bexit (\d+)")


def report(name: str, **fields) -> None:
    """Merge `fields` into the run's progress file (the API's reference helper, atomically)."""
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / ((re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:64] or "run") + ".json")
    try:
        run = json.loads(path.read_text())
    except (OSError, ValueError):
        run = {}
    now = int(time.time() * 1000)
    metrics = run.get("metrics", {})
    for group, kv in fields.pop("metrics", {}).items():
        metrics[group] = {**metrics.get(group, {}), **kv}
    run.update(fields, v=1, name=name, metrics=metrics, updated=now)
    run.setdefault("started", now)
    fd, tmp = tempfile.mkstemp(dir=RUNS, prefix=".", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(run, f)
    os.replace(tmp, path)


def _tail(p: Path, n: int = 65_536) -> str:
    try:
        with open(p, "rb") as f:
            f.seek(max(0, p.stat().st_size - n))
            return f.read().decode(errors="replace")
    except OSError:
        return ""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def sae_fields(text: str, epochs: int) -> dict:
    """The last epoch line: epochs completed, and the SAE's own train metrics."""
    m = None
    for m in EPOCH.finditer(text):
        pass
    if m is None:
        return {"epoch": 0, "epochs": epochs}
    st = ast.literal_eval(m.group(3))
    return {"epoch": st["epoch"] + 1, "epochs": epochs, "unit": "epoch",
            "metrics": {"train": {"nmse": st["nmse"], "aux": st["aux"], "dead %": 100 * st["dead"]}}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("sae", "chain"))
    ap.add_argument("name")
    ap.add_argument("log", type=Path)
    ap.add_argument("pid", type=int)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--every", type=float, default=5.0)
    a = ap.parse_args()
    report(a.name, status="running", pid=a.pid)
    gone = False
    while True:
        text = _tail(a.log)
        if a.mode == "sae":
            f = sae_fields(text, a.epochs)
            done = re.search(r"^\S+ A1 \{|\bA1 \{", text) is not None or "epoch %d:" % (a.epochs - 1) in text
            failed = "out of memory" in text or "Traceback" in text
        else:
            steps = STEP.findall(text)
            ok = sum(1 for c in EXIT.findall(text) if c == "0")
            cur = steps[-1] if steps else ""
            f = {"stage": {"current": ok, "total": a.steps, "label": "step"}, "message": cur}
            inner = a.log.parent / f"log_{cur}.txt"
            if cur and inner.exists():
                f |= sae_fields(_tail(inner), a.epochs) if EPOCH.search(_tail(inner)) else {}
            done, failed = "ALLDONE" in text, "STOPPED" in text
        if failed:
            report(a.name, **f, status="failed", message="out of memory" if "out of memory" in text else "stopped")
            return
        if done:
            report(a.name, **f, status="done")
            return
        if gone:
            report(a.name, **f)  # status stays running: the status line shows `stopped?`
            return
        if not _alive(a.pid):
            gone = True  # the job may have written its last line after this pass's read: read once more
            continue
        report(a.name, **f, status="running", pid=a.pid)
        time.sleep(a.every)


if __name__ == "__main__":
    main()
