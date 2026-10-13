"""Ladder v2 runner (docs/ladder_v2/plan.md, Task 2): the manifest, strictly in order, blind.

    .venv/bin/python artifacts/ladder_v2_run.py --plan                  # dry run: resolve, pin-check, estimate
    .venv/bin/python artifacts/_detach.py runs/ladder_v2/logs/runner_<ts>.log \
        --name ladder_v2 --stage '=== \\d+/\\d+ (\\S+)' --stages <N> -- \
        .venv/bin/python artifacts/ladder_v2_run.py                     # the run (second approval)

One entry at a time, each a subprocess under `artifacts/_heavy.sh` (the 18 GB rule: memory returns
between steps, and no other heavy job overlaps). The run does not branch on any output.

- **Pins.** Before every entry each pinned SHA-1 (driver and its imported artifacts modules, the
  package tree, configs, the encoder) is re-checked; a moved pin stops the runner — the code under
  the ladder changed, which is not an entry's failure.
- **Outputs.** The entry's directory is snapshotted before and after; every file it created or
  changed is SHA-1'd into the state (`outputs.sha1` mirrors it). Before every entry every recorded
  file is re-verified (stat-cached within a process, fully re-hashed at start); a file that changed
  *between* entries stops the runner. A rewrite *by* an entry of an earlier entry's file is
  attributed to it and listed.
- **Failure.** Non-zero exit, or an `expect`ed file absent: `failed`, with the exit code and the log
  path; every entry reading it, transitively, `skipped(dep)`; the runner continues. A failed entry is
  not retried on restart unless `--retry-failed` names it.
- **Resume.** The state is rewritten atomically after every transition. An entry found `running`
  was interrupted: files it created are removed (bar its `keep` globs, which its driver resumes from)
  and it runs again.
- **Blind.** Entry stdout/stderr go only to `<root>/logs/<id>.log`. The runner prints and records id,
  status, wall-clock, peak RSS and output paths — never a value read from an output or a log.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
HEAVY = REPO / "artifacts" / "_heavy.sh"
LOCK = Path("/private/tmp/claude-501/galaxy-jepa-heavy.lock")  # _heavy.sh's
#: Directories no entry may write: M's records and the shared DD inputs. Snapshotted around every
#: entry; a change stops the runner (an entry escaped its directory).
GUARDED = [REPO / "artifacts" / "out", REPO / "runs" / "dd_local_v2"]


class Stop(SystemExit):
    """The ladder cannot go on as declared (pins moved, recorded outputs altered, state mismatch)."""


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def _sha1_file(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _sha1_tree(root: Path, pattern: str = "*") -> str:
    h = hashlib.sha1()
    for p in sorted(root.rglob(pattern)):
        if p.is_file():
            h.update(f"{p.relative_to(root)}\0{_sha1_file(p)}\n".encode())
    return h.hexdigest()


class Hasher:
    """SHA-1 by path, cached on (size, mtime_ns) for the life of the process."""

    def __init__(self) -> None:
        self.cache: dict[str, tuple[int, int, str]] = {}

    def __call__(self, p: Path) -> str:
        st = p.stat()
        hit = self.cache.get(str(p))
        if hit and hit[:2] == (st.st_size, st.st_mtime_ns):
            return hit[2]
        sha = _sha1_tree(p) if p.is_dir() else _sha1_file(p)
        self.cache[str(p)] = (st.st_size, st.st_mtime_ns, sha)
        return sha


def _pin_sha(rel: str, sha: Hasher) -> str:
    if rel.endswith("/**.py"):
        return _sha1_tree_pkg(_abs(rel[: -len("/**.py")]))
    return sha(_abs(rel))


def _sha1_tree_pkg(root: Path) -> str:  # artifacts/ladder_v2_manifest.tree_sha1, verbatim
    h = hashlib.sha1()
    for p in sorted(root.rglob("*.py")):
        h.update(f"{p.relative_to(root)}\0{_sha1_file(p)}\n".encode())
    return h.hexdigest()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _snapshot(d: Path) -> dict[str, tuple[int, int]]:
    """Every file under d (symlinks recorded as themselves, not followed into directories)."""
    out: dict[str, tuple[int, int]] = {}
    if not d.exists():
        return out
    for root, dirs, files in os.walk(d, followlinks=False):
        for name in files + [x for x in dirs if (Path(root) / x).is_symlink()]:
            p = Path(root) / name
            st = p.stat() if p.exists() else p.lstat()
            out[str(p)] = (st.st_size, st.st_mtime_ns)
    return out


class Runner:
    def __init__(self, manifest: Path, state: Path, *, retry_failed: list[str], dry: bool) -> None:
        self.manifest_path = manifest
        raw = manifest.read_bytes()
        self.doc = yaml.safe_load(raw)
        self.entries: list[dict] = self.doc["entries"]
        self.by_id = {e["id"]: e for e in self.entries}
        self.root = _abs(self.doc["root"])
        self.logs = self.root / "logs"
        self.state_path = state
        self.sha = Hasher()
        self.manifest_sha1 = hashlib.sha1(raw).hexdigest()
        self.retry = set(retry_failed)
        self.child: int | None = None
        self.dry = dry
        self.state = self._load_state()

    # ---- state -----------------------------------------------------------------------------
    def _load_state(self) -> dict:
        if self.state_path.exists():
            st = json.loads(self.state_path.read_text())
            if st["manifest_sha1"] != self.manifest_sha1:
                raise Stop(f"state {self.state_path} belongs to another manifest "
                           f"({st['manifest_sha1'][:12]} ≠ {self.manifest_sha1[:12]}); refusing to mix")
            return st
        return {"manifest": str(self.manifest_path), "manifest_sha1": self.manifest_sha1,
                "entries": {e["id"]: {"status": "pending"} for e in self.entries}, "files": {}}

    def _save(self) -> None:
        if self.dry:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        os.replace(tmp, self.state_path)
        sums = self.state_path.with_name("outputs.sha1")
        lines = [f"{v[0]}  {path}" for path, v in sorted(self.state["files"].items())]
        tmp = sums.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + ("\n" if lines else ""))
        os.replace(tmp, sums)

    def _log_row(self, eid: str, rec: dict) -> None:
        if self.dry:
            return
        path = self.state_path.with_name("run_log.md")
        if not path.exists():
            path.write_text("# Ladder v2 run log (generated by artifacts/ladder_v2_run.py; no result values)\n\n"
                            "| entry | status | wall | peak RSS | outputs |\n|---|---|---|---|---|\n")
        outs = ", ".join(f"`{Path(p).relative_to(self.root) if Path(p).is_relative_to(self.root) else p}`"
                         for p in rec.get("outputs", []))
        wall = f"{rec['wall_s'] / 60:.1f} min" if "wall_s" in rec else "—"
        rss = f"{rec['peak_rss_mb']:,.0f} MB" if "peak_rss_mb" in rec else "—"
        status = rec["status"] + (f" (exit {rec['exit']})" if rec.get("exit") not in (None, 0) else "")
        with path.open("a") as fh:
            fh.write(f"| {eid} | {status} | {wall} | {rss} | {outs or '—'} |\n")

    # ---- checks ----------------------------------------------------------------------------
    def check_pins(self, e: dict) -> list[str]:
        moved = []
        for rel, want in e["pins"].items():
            try:
                got = _pin_sha(rel, self.sha)
            except FileNotFoundError:
                got = "MISSING"
            if got != want:
                moved.append(rel)
        return moved

    def verify_files(self) -> None:
        for path, (want, owner) in self.state["files"].items():
            p = Path(path)
            if not (p.exists() or p.is_symlink()):
                raise Stop(f"recorded output {path} (from {owner}) has gone")
            got = self._file_sha(p)
            if got != want:
                raise Stop(f"recorded output {path} (from {owner}) changed between entries")

    def _file_sha(self, p: Path) -> str:
        if p.is_symlink() and not p.exists():
            return "dangling:" + os.readlink(p)
        return self.sha(p.resolve() if p.is_symlink() else p)

    def _deps_ok(self, e: dict) -> str | None:
        for i in e["inputs"]:
            s = self.state["entries"][i]["status"]
            if s != "done":
                return i
        return None

    # ---- execution ---------------------------------------------------------------------------
    def _env(self, e: dict) -> dict:
        env = dict(os.environ)
        for k in [k for k in env if k.startswith(("LADDER_", "DD_", "F0_")) or k == "GJ_CONFIG"]:
            del env[k]  # only the manifest decides these
        env.update({k: str(v) for k, v in self.doc["common_env"].items()})
        env.update({k: str(v) for k, v in e.get("env", {}).items()})
        return env

    def _cleanup_interrupted(self, e: dict, rec: dict) -> None:
        pre = set(rec.get("pre", []))
        keep = e.get("keep", [])
        for path in sorted(_snapshot(_abs(e["out_dir"]))):
            if path in pre or any(fnmatch.fnmatch(Path(path).name, g) for g in keep):
                continue
            Path(path).unlink()
            print(f"    cleared interrupted output {path}", flush=True)

    def _link(self, e: dict, out_dir: Path) -> list[Path]:
        spec, made = e["link"], []
        if "src" in spec:
            pairs = [(_abs(spec["src"]), out_dir / spec["dst"])]
        else:
            pairs = []
            for pat in spec["glob"]:
                matches = sorted(_abs(Path(pat).parent).glob(Path(pat).name))
                if not matches:
                    raise FileNotFoundError(f"link: no match for {pat}")
                sub = next(v for k, v in spec["dst_dir"].items() if f"/{k}" in pat)
                pairs += [(m, out_dir / sub / m.name) for m in matches]
        for src, dst in pairs:
            if not src.exists():
                raise FileNotFoundError(f"link: {src} is absent")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink() and dst.resolve() == src.resolve():
                made.append(dst)
                continue
            if dst.exists() or dst.is_symlink():
                raise FileExistsError(f"link: {dst} exists and is not a link to {src}")
            dst.symlink_to(src.resolve())
            made.append(dst)
        return made

    def run_entry(self, k: int, n: int, e: dict) -> None:
        eid, rec = e["id"], self.state["entries"][e["id"]]
        out_dir = _abs(e["out_dir"])
        print(f"=== {k}/{n} {eid}", flush=True)
        if rec["status"] == "running":
            holder = (LOCK / "label").read_text().strip() if (LOCK / "label").exists() else ""
            pid = (LOCK / "pid").read_text().strip() if (LOCK / "pid").exists() else ""
            if holder == eid and pid and _alive(int(pid)):
                raise Stop(f"{eid}: the interrupted attempt is still running (pid {pid}); stop it first")
            self._cleanup_interrupted(e, rec)
        moved = self.check_pins(e)
        if moved:
            raise Stop(f"{eid}: pinned files moved since the manifest was generated: {moved}")
        self.verify_files()
        out_dir.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        before = _snapshot(out_dir)
        guard = {str(d): _snapshot(d) for d in GUARDED}
        rec.clear()
        rec.update({"status": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S"), "pre": sorted(before)})
        self._save()
        t0, rc, rss = time.time(), 0, None
        log = self.logs / f"{eid}.log"
        if e["kind"] == "link":
            try:
                self._link(e, out_dir)
            except (FileNotFoundError, FileExistsError, StopIteration) as err:
                rc = 1
                log.write_text(f"link failed: {err}\n")
        else:
            with open(log, "w") as fh:
                fh.write(f"# {eid}\n# cmd: {' '.join(e['cmd'])}\n# env: {json.dumps(e.get('env', {}))}\n")
                fh.flush()
                proc = subprocess.Popen([str(HEAVY), eid, *e["cmd"]], cwd=REPO, env=self._env(e),
                                        stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        start_new_session=True)
                self.child = proc.pid
                _, status, usage = os.wait4(proc.pid, 0)
                self.child = None
                proc.returncode = os.waitstatus_to_exitcode(status)
                rc = proc.returncode
                rss = usage.ru_maxrss / 2**20 if sys.platform == "darwin" else usage.ru_maxrss / 2**10
        escaped = [p for d in GUARDED for p, st in _snapshot(d).items() if guard[str(d)].get(p) != st]
        escaped += [p for d in GUARDED for p in guard[str(d)] if not Path(p).exists()]
        if escaped:
            rec.update({"status": "failed", "exit": rc, "escaped": len(escaped)})
            self._save()
            self._log_row(eid, rec)
            raise Stop(f"{eid} wrote {len(escaped)} files outside its directory, e.g. {escaped[0]}")
        after = _snapshot(out_dir)
        changed = [p for p, st in after.items() if before.get(p) != st]
        rewrote = sorted({self.state["files"][p][1] for p in changed if p in self.state["files"]} - {eid})
        missing = [x for x in e.get("expect", []) if not (out_dir / x).exists()]
        outputs = {}
        for p in sorted(changed):
            outputs[p] = self._file_sha(Path(p))
            self.state["files"][p] = [outputs[p], eid]
        rec.pop("pre", None)
        rec.update({"status": "done" if rc == 0 and not missing else "failed", "exit": rc,
                    "wall_s": round(time.time() - t0, 1), "outputs": sorted(outputs), "log": str(log)})
        if rss is not None:
            rec["peak_rss_mb"] = round(rss, 1)
        if missing:
            rec["missing_expected"] = missing
        if rewrote:
            rec["rewrote_files_of"] = rewrote
        self._save()
        self._log_row(eid, rec)
        print(f"    {rec['status']}  {rec['wall_s'] / 60:.1f} min"
              + (f"  rss {rec['peak_rss_mb']:,.0f} MB" if "peak_rss_mb" in rec else "")
              + (f"  exit {rc}" if rc else "") + (f"  missing {missing}" if missing else "")
              + f"  outputs {len(outputs)}", flush=True)
        if rec["status"] == "failed":
            self._skip_dependants(eid)

    def _skip_dependants(self, failed: str) -> None:
        bad = {failed}
        for e in self.entries:
            if any(i in bad for i in e["inputs"]) and self.state["entries"][e["id"]]["status"] != "done":
                bad.add(e["id"])
                if e["id"] != failed:
                    self.state["entries"][e["id"]] = {"status": "skipped(dep)", "because": failed}
                    self._log_row(e["id"], self.state["entries"][e["id"]])
        self._save()

    def run(self) -> int:
        n = len(self.entries)
        print(f"ladder v2 runner: {n} entries, manifest {self.manifest_sha1[:12]}, state {self.state_path}",
              flush=True)
        print("verifying recorded outputs (full re-hash)…", flush=True)
        self.verify_files()
        for eid in self.retry:
            st = self.state["entries"].get(eid, {}).get("status")
            if st != "failed":
                raise Stop(f"--retry-failed {eid}: it is {st!r}, not failed")
            # the failed attempt's leftovers neither feed the retry nor stay on the record
            keep = self.by_id[eid].get("keep", [])
            for path in [p for p, v in self.state["files"].items() if v[1] == eid]:
                del self.state["files"][path]
                if not any(fnmatch.fnmatch(Path(path).name, g) for g in keep):
                    Path(path).unlink(missing_ok=True)
                    print(f"    cleared failed attempt's output {path}", flush=True)
            self.state["entries"][eid] = {"status": "pending", "retried": True}
            for e in self.entries:  # its dependants come back with it
                rec = self.state["entries"][e["id"]]
                if rec["status"] == "skipped(dep)":
                    self.state["entries"][e["id"]] = {"status": "pending"}
        self._save()
        for k, e in enumerate(self.entries, 1):
            st = self.state["entries"][e["id"]]["status"]
            if st in ("done", "failed", "skipped(dep)"):
                print(f"=== {k}/{n} {e['id']} ({st}, earlier)", flush=True)
                continue
            blocker = self._deps_ok(e)
            if blocker:
                print(f"=== {k}/{n} {e['id']}", flush=True)
                self.state["entries"][e["id"]] = {"status": "skipped(dep)", "because": blocker}
                self._save()
                self._log_row(e["id"], self.state["entries"][e["id"]])
                print(f"    skipped(dep) — {blocker}", flush=True)
                continue
            self.run_entry(k, n, e)
        counts: dict[str, int] = {}
        for rec in self.state["entries"].values():
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
        print(f"ladder v2 runner: finished — {counts}", flush=True)
        return 0

    # ---- dry run ---------------------------------------------------------------------------
    def plan(self) -> int:
        n = len(self.entries)
        print(f"manifest {self.manifest_path} ({self.manifest_sha1[:12]}): {n} entries; root {self.root}")
        print(f"state: {'none yet' if not self.state_path.exists() else self.state_path}\n")
        bad: list[str] = []
        ids_so_far: set[str] = set()
        pinned: dict[str, str] = {}
        for e in self.entries:
            for i in e["inputs"]:
                if i not in ids_so_far:
                    bad.append(f"{e['id']}: input {i} does not precede it")
            ids_so_far.add(e["id"])
            for rel, want in e["pins"].items():
                if pinned.setdefault(rel, want) != want:
                    bad.append(f"{e['id']}: pins {rel} at two different SHA-1s within the manifest")
            if e["kind"] == "run":
                script = REPO / e["cmd"][1]
                if not script.exists():
                    bad.append(f"{e['id']}: {e['cmd'][1]} absent")
            else:
                spec = e["link"]
                srcs = [spec["src"]] if "src" in spec else spec["glob"]
                for s in srcs:
                    made_by = [x["id"] for x in self.entries if str(_abs(s)).startswith(str(_abs(x["out_dir"])) + "/")
                               and x["id"] in e["inputs"]]
                    if not made_by and not list(_abs(Path(s).parent).glob(Path(s).name)):
                        bad.append(f"{e['id']}: link source {s} absent and produced by no input")
        moved = []
        for rel, want in sorted(pinned.items()):
            try:
                got = _pin_sha(rel, self.sha)
            except FileNotFoundError:
                got = "MISSING"
            if got != want:
                moved.append(rel)
        print(f"pins: {len(pinned)} distinct files/trees, {len(moved)} moved"
              + (f": {moved}" if moved else " — all match"))
        enc_ckpt = {k: v for e in self.entries for k, v in e["pins"].items() if k.endswith("encoder.pt")}
        for rel in sorted(enc_ckpt):
            print(f"  encoder {rel}: SHA-1 {enc_ckpt[rel][:12]} {'ok' if rel not in moved else 'MOVED'}")
        print(f"structure: {len(bad)} problems" + ("" if not bad else ":\n  " + "\n  ".join(bad)))

        print("\norder (k  id  heavy  est):")
        tot: dict[str, dict[str, float]] = {}
        for k, e in enumerate(self.entries, 1):
            st = self.state["entries"][e["id"]]["status"]
            print(f"  {k:3d}  {e['id']:<44s} {e['heavy']:<4s} {e['est_h']:5.2f} h"
                  + ("" if st == "pending" else f"  [{st}]"))
            t = tot.setdefault(e["encoder"], {"mps": 0.0, "cpu": 0.0})
            t[e["heavy"]] += e["est_h"]
        print("\nestimates (serial; ±50%):")
        g_m = g_c = 0.0
        for enc, t in tot.items():
            g_m, g_c = g_m + t["mps"], g_c + t["cpu"]
            print(f"  {enc:<7s} MPS {t['mps']:5.1f} h   CPU {t['cpu']:5.1f} h   = {t['mps'] + t['cpu']:5.1f} h")
        print(f"  total   MPS {g_m:5.1f} h   CPU {g_c:5.1f} h   = {g_m + g_c:5.1f} h ≈ {(g_m + g_c) / 24:.1f} days")
        print("\nexcluded:")
        for k, v in self.doc["excluded"].items():
            print(f"  - {k}: {v}")
        print("not implemented (reported as such):\n  - " + "\n  - ".join(self.doc["not_implemented"]))
        print("\n--plan: nothing run")
        return 1 if (bad or moved) else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=REPO / "docs/ladder_v2/manifest.yaml")
    ap.add_argument("--state", type=Path, default=None, help="default: state.json beside the manifest")
    ap.add_argument("--plan", action="store_true", help="resolve, pin-check and estimate; run nothing")
    ap.add_argument("--retry-failed", nargs="*", default=[])
    a = ap.parse_args()
    state = a.state or a.manifest.with_name("state.json")
    r = Runner(a.manifest, state, retry_failed=a.retry_failed, dry=a.plan)

    def _term(signum, _frame):  # take the entry's process group down too: no orphan keeps writing
        if r.child:
            try:
                os.killpg(r.child, signal.SIGTERM)
            except ProcessLookupError:
                pass
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    sys.exit(r.plan() if a.plan else r.run())


if __name__ == "__main__":
    main()
