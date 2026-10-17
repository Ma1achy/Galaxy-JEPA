"""Ladder v2 control: pause and resume the runner safely (status, pause, resume). Prints no values.

    .venv/bin/python artifacts/ladder_v2_ctl.py status
    .venv/bin/python artifacts/ladder_v2_ctl.py pause            # at the next entry boundary: nothing lost
    .venv/bin/python artifacts/ladder_v2_ctl.py pause --now      # interrupt the entry; it reruns on resume
    .venv/bin/python artifacts/ladder_v2_ctl.py pause --now --eject   # …then eject the runs disk
    .venv/bin/python artifacts/ladder_v2_ctl.py resume           # preflight, relaunch, watchdog

A boundary pause drops `PAUSE` beside the state; the runner honours it before its next entry. A pause
`--now` also sends the runner SIGTERM: it takes the entry's process group down, clears what the entry
wrote and puts it back to pending (artifacts/ladder_v2_run.py, "Pause"). Both end with the runner
printing `PAUSED` and exiting 75, which `_detach.py` and the status line read as paused.

`--eject` waits for the runner and the heavy lock, then ejects the volume holding the manifest's root
only if nothing has a file open on it (holders are listed, never killed).

`resume` refuses unless: the runs root is mounted; no runner is alive; the heavy lock is free; and the
runner's own `--plan` passes (every pin unmoved, the manifest well formed). Then it removes `PAUSE`,
relaunches the runner detached on a new log under the same status-line name (so the bar resumes), and
starts the watchdog on that log. The new log's path is written to `runner_log` beside the state.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv" / "bin" / "python"
RUNNER = "artifacts/ladder_v2_run.py"
LOCK = Path("/private/tmp/claude-501/galaxy-jepa-heavy.lock")  # _heavy.sh's
NAME = "ladder_v2"
STAGE = r"=== \d+/\d+ (\S+)"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _cmdline(pid: int) -> str:
    r = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True)
    return r.stdout.strip()


class Ladder:
    def __init__(self, manifest: Path) -> None:
        self.manifest = manifest
        self.doc = yaml.safe_load(manifest.read_text())
        self.n = len(self.doc["entries"])
        self.est = {e["id"]: e["est_h"] for e in self.doc["entries"]}
        root = Path(self.doc["root"])
        self.root = root if root.is_absolute() else REPO / root
        self.state_path = manifest.with_name("state.json")
        self.pause_file = manifest.with_name("PAUSE")
        self.log_ptr = manifest.with_name("runner_log")

    def state(self) -> dict:
        return json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    def runner_pid(self) -> int | None:
        """The live runner for this state, by the pid it recorded and its command line."""
        pid = self.state().get("runner", {}).get("pid")
        if pid and _alive(pid) and RUNNER in _cmdline(pid):
            return pid
        # a runner started before it recorded its pid (or by hand): find it by command line
        r = subprocess.run(["pgrep", "-f", RUNNER], capture_output=True, text=True)
        pids = [int(x) for x in r.stdout.split() if int(x) != os.getpid()]
        pids = [p for p in pids if "--plan" not in _cmdline(p)]
        return pids[0] if pids else None

    def lock_holder(self) -> str | None:
        pid = (LOCK / "pid").read_text().strip() if (LOCK / "pid").exists() else ""
        if pid and _alive(int(pid)):
            label = (LOCK / "label").read_text().strip() if (LOCK / "label").exists() else "?"
            return f"{label} (pid {pid})"
        return None

    def volume(self) -> Path | None:
        """The mount point holding the runs root (resolved through the repo's symlink)."""
        try:
            real = self.root.resolve(strict=True)
        except (FileNotFoundError, OSError):
            return None
        parts = real.parts
        return Path(*parts[:3]) if len(parts) > 2 and parts[1] == "Volumes" else None

    def runs_mounted(self) -> bool:
        try:
            return self.root.resolve(strict=True).is_dir()
        except (FileNotFoundError, OSError):
            return False


def status(L: Ladder) -> int:
    st = L.state()
    entries = st.get("entries", {})
    counts: dict[str, int] = {}
    for rec in entries.values():
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
    done = sum(v for k, v in counts.items() if k in ("done", "failed", "skipped(dep)"))
    pid = L.runner_pid()
    print(f"ladder v2: {done}/{L.n} entries settled — {counts}")
    print(f"  runner: {'alive, pid ' + str(pid) if pid else 'not running'}")
    for eid, rec in entries.items():
        if rec["status"] == "running":
            since = rec.get("started", "?")
            print(f"  {'current' if pid else 'interrupted'} entry: {eid} (since {since}, est {L.est[eid]:.2f} h)"
                  + ("" if pid else " — reruns on resume"))
    if st.get("paused"):
        p = st["paused"]
        print(f"  PAUSED {p['at']} before {p['before']}/{L.n} {p['entry']} ({p['how']})")
    if L.pause_file.exists():
        print(f"  pause requested ({L.pause_file.name} present): "
              + ("the runner stops before its next entry" if pid else "`resume` removes it"))
    print(f"  heavy lock: {L.lock_holder() or 'free'}")
    print(f"  runs root {L.root}: {'mounted' if L.runs_mounted() else 'NOT MOUNTED'}")
    if L.log_ptr.exists():
        print(f"  runner log: {L.log_ptr.read_text().strip()}")
    return 0


def _wait_stopped(L: Ladder, timeout: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not L.runner_pid() and not L.lock_holder():
            return True
        time.sleep(2)
    return False


def _eject(L: Ladder) -> int:
    vol = L.volume()
    if vol is None:
        print("eject: the runs root is not on an external volume (or not mounted); nothing to eject")
        return 1
    r = subprocess.run(["lsof", "-Fpc", "+f", "--", str(vol)], capture_output=True, text=True)
    holders, pid = [], None
    for line in r.stdout.splitlines():
        if line.startswith("p"):
            pid = line[1:]
        elif line.startswith("c") and pid:
            holders.append(f"{line[1:]} (pid {pid})")
    if holders:
        print(f"eject: NOT ejecting {vol} — files are open on it:\n  " + "\n  ".join(sorted(set(holders))))
        return 1
    r = subprocess.run(["diskutil", "eject", str(vol)], capture_output=True, text=True)
    print(f"eject: {(r.stdout or r.stderr).strip()}")
    return r.returncode


def pause(L: Ladder, now: bool, wait: bool, eject: bool) -> int:
    pid = L.runner_pid()
    if not pid:
        print("pause: no runner is alive — the ladder is already stopped")
        status(L)
        return _eject(L) if eject else 0
    L.pause_file.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + (" now\n" if now else " boundary\n"))
    cur = next((eid for eid, rec in L.state().get("entries", {}).items() if rec["status"] == "running"), None)
    if now:
        os.kill(pid, signal.SIGTERM)
        print(f"pause --now: SIGTERM to runner {pid}; {cur or 'no entry'} is interrupted and reruns on resume")
    else:
        print(f"pause: requested; the runner stops after {cur or 'its current step'}"
              + (f" (est {L.est[cur]:.2f} h, since {L.state()['entries'][cur].get('started', '?')})" if cur else ""))
    if not (now or wait or eject):
        print("  (pause --wait to block until it has stopped; ladder_v2_ctl.py status to check)")
        return 0
    timeout = 300 if now else 7 * 86400
    print("  waiting for the runner to exit and the heavy lock to free…", flush=True)
    if not _wait_stopped(L, timeout):
        print(f"pause: still running after {timeout:.0f} s — not ejecting. Check with `status`.")
        return 1
    print("  stopped.")
    status(L)
    return _eject(L) if eject else 0


def resume(L: Ladder, name: str) -> int:
    problems = []
    if not L.runs_mounted():
        problems.append(f"the runs root {L.root} is not mounted (plug the disk in)")
    if L.runner_pid():
        problems.append(f"a runner is alive (pid {L.runner_pid()})")
    if L.lock_holder():
        problems.append(f"the heavy lock is held by {L.lock_holder()}")
    if problems:
        print("resume: refusing —\n  " + "\n  ".join(problems))
        return 1
    plan = subprocess.run([str(PY), RUNNER, "--manifest", str(L.manifest), "--plan"], cwd=REPO,
                          capture_output=True, text=True)
    summary = [ln for ln in plan.stdout.splitlines() if ln.startswith(("pins:", "structure:", "  encoder "))]
    print("preflight (--plan):\n  " + "\n  ".join(summary))
    if plan.returncode != 0:
        print("resume: refusing — the runner's --plan did not pass (above); nothing launched")
        return 1
    L.pause_file.unlink(missing_ok=True)
    logs = L.root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"runner_{time.strftime('%Y%m%d_%H%M')}.log"
    if log.exists():
        log = logs / f"runner_{time.strftime('%Y%m%d_%H%M%S')}.log"
    rel = os.path.relpath(log, REPO) if log.is_relative_to(REPO) else str(log)
    subprocess.run([str(PY), "artifacts/_detach.py", rel, "--name", name, "--stage", STAGE, "--stages", str(L.n),
                    "--", str(PY), RUNNER, "--manifest", str(L.manifest)], cwd=REPO, check=True)
    L.log_ptr.write_text(rel + "\n")
    wd = REPO / "artifacts" / "ladder_v2_watchdog.sh"
    if L.manifest.resolve() == (REPO / "docs/ladder_v2/manifest.yaml").resolve():
        subprocess.run([str(PY), "artifacts/_detach.py", str(logs / "watchdog.log"), "--name", f"{name}_watchdog",
                        "--", str(wd), rel], cwd=REPO, check=True)
    time.sleep(3)
    pid = L.runner_pid()
    print(f"resume: runner {'alive, pid ' + str(pid) if pid else 'NOT ALIVE — read ' + rel}; log {rel}")
    return 0 if pid else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=REPO / "docs/ladder_v2/manifest.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p = sub.add_parser("pause")
    p.add_argument("--now", action="store_true", help="interrupt the current entry (it reruns on resume)")
    p.add_argument("--wait", action="store_true", help="block until the runner has stopped")
    p.add_argument("--eject", action="store_true", help="then eject the runs disk if nothing holds it")
    r = sub.add_parser("resume")
    r.add_argument("--name", default=NAME, help="status-line name (the paused bar under it resumes)")
    a = ap.parse_args()
    L = Ladder(a.manifest.resolve())
    if a.cmd == "status":
        sys.exit(status(L))
    if a.cmd == "pause":
        sys.exit(pause(L, a.now, a.wait, a.eject))
    sys.exit(resume(L, a.name))


if __name__ == "__main__":
    main()
