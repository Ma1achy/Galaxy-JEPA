"""The results bucket (gj-results) for the rental box: run.sh's only way out. Investigation code: terse.

    python artifacts/results.py writetest                       # a 1-byte object up, read back, removed: the token can write here
    python artifacts/results.py put LOCAL KEY                   # one file, size + MD5 checked against the bucket
    python artifacts/results.py putdir DIR PREFIX               # every file under DIR not already there with the same size and MD5
    python artifacts/results.py get KEY LOCAL | getdir PREFIX DIR | exists KEY (exit 0/1) | ls PREFIX
    python artifacts/results.py ckpt-push RUNDIR PREFIX         # the newest committed checkpoint, then its manifest; older remote ones removed
    python artifacts/results.py ckpt-pull PREFIX RUNDIR         # the manifest and the checkpoint it names, sha256 checked; exit 3 if none
    python artifacts/results.py selftest                        # all of the above against a local directory standing in for the bucket

Credentials, in the box's .env only (never printed): R2_ENDPOINT, R2_RES_ACCESS_KEY_ID, R2_RES_SECRET_ACCESS_KEY: Object Read & Write on gj-results
only (user, 25 Nov). The cache bucket stays read-only to the box (ship.py, R2_RO_*). RES_LOCAL=DIR swaps the bucket for a directory (tests, the Mac's kill test).
A checkpoint is pushed file first, manifest second, so a manifest in the bucket never names a file that isn't there; the restore needs only the newest entry.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ship  # noqa: E402

BUCKET = "gj-results"


def remote() -> ship.Remote:
    if os.environ.get("RES_LOCAL"):
        Path(os.environ["RES_LOCAL"]).mkdir(parents=True, exist_ok=True)
        return ship.Remote("RES", local=Path(os.environ["RES_LOCAL"]))
    r = ship.Remote("RES")
    r.base = f"R2:{BUCKET}"
    return r


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


def ls(r, prefix: str) -> dict[str, int]:
    try:
        out = r.run("lsjson", f"{r.base}/{prefix}", "--recursive", "--files-only")
    except RuntimeError as e:  # a prefix with nothing under it: rclone says "directory not found" on a local remote
        if "not found" in str(e):
            return {}
        raise
    return {d["Path"]: d["Size"] for d in json.loads(out or "[]")}


def exists(r, key: str) -> bool:
    d, _, name = key.rpartition("/")
    return name in ls(r, d) if d else key in ls(r, "")


def put(r, f: Path, key: str) -> None:
    r.put(Path(f), key, [])


def putdir(r, d: Path, prefix: str) -> int:
    have, n = ls(r, prefix), 0
    for f in sorted(p for p in Path(d).rglob("*") if p.is_file() and not p.name.endswith(".tmp")):
        rel = f.relative_to(d).as_posix()
        if have.get(rel) == f.stat().st_size and rel.endswith((".pt", ".npz")):  # big and written once: size is enough to skip; small files always go
            continue
        put(r, f, f"{prefix}/{rel}")
        n += 1
    return n


def get(r, key: str, f: Path) -> None:
    Path(f).parent.mkdir(parents=True, exist_ok=True)
    r.get(key, Path(f))


def getdir(r, prefix: str, d: Path) -> int:
    files = ls(r, prefix)
    for rel in files:
        get(r, f"{prefix}/{rel}", Path(d) / rel)
    return len(files)


def ckpt_push(r, run: Path, prefix: str) -> int | None:
    """Push the newest committed checkpoint of RUNDIR/checkpoints. The manifest is snapshotted first: the trainer may commit a newer one meanwhile."""
    cdir = Path(run) / "checkpoints"
    mf = cdir / "checkpoints.json"
    if not mf.exists():
        return None
    snap = json.loads(mf.read_text())
    if not snap.get("entries"):
        return None
    e = max(snap["entries"], key=lambda x: int(x["step"]))
    have = ls(r, f"{prefix}/checkpoints")
    if have.get(e["file"]) != e["bytes"]:
        put(r, cdir / e["file"], f"{prefix}/checkpoints/{e['file']}")
    with tempfile.TemporaryDirectory() as t:
        (Path(t) / "checkpoints.json").write_text(json.dumps({**snap, "entries": [e]}, indent=2) + "\n")  # names only what the bucket holds
        put(r, Path(t) / "checkpoints.json", f"{prefix}/checkpoints/checkpoints.json")
    for name in have:
        if name.startswith("ckpt_") and name != e["file"]:
            r.run("deletefile", f"{r.base}/{prefix}/checkpoints/{name}")
    return int(e["step"])


def ckpt_pull(r, prefix: str, run: Path) -> int | None:
    """Restore RUNDIR/checkpoints from the bucket when the local disk has none (a fresh box after a preemption). Never overwrites a local manifest."""
    cdir = Path(run) / "checkpoints"
    if (cdir / "checkpoints.json").exists():
        raise SystemExit(f"{cdir} already has a manifest: the local run is the newer one, nothing pulled")
    if "checkpoints.json" not in ls(r, f"{prefix}/checkpoints"):
        return None
    cdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cdir) as t:
        get(r, f"{prefix}/checkpoints/checkpoints.json", Path(t) / "m.json")
        m = json.loads((Path(t) / "m.json").read_text())
        e = max(m["entries"], key=lambda x: int(x["step"]))
        get(r, f"{prefix}/checkpoints/{e['file']}", Path(t) / e["file"])
        if sha256(Path(t) / e["file"]) != e["sha256"]:
            raise SystemExit(f"{prefix}/checkpoints/{e['file']}: sha256 is not the manifest's: refusing to resume from it")
        os.replace(Path(t) / e["file"], cdir / e["file"])
        os.replace(Path(t) / "m.json", cdir / "checkpoints.json")  # the commit point, last
    return int(e["step"])


def writetest(r) -> None:
    with tempfile.TemporaryDirectory() as t:
        f = Path(t) / "probe"
        f.write_bytes(b"x")
        put(r, f, "_writetest/probe")
        get(r, "_writetest/probe", Path(t) / "back")
        assert (Path(t) / "back").read_bytes() == b"x"
        r.run("deletefile", f"{r.base}/_writetest/probe")
    print("results bucket: write, read back and delete OK")


def selftest() -> None:
    import torch
    sys.path.insert(0, str(ship.REPO / "src"))
    from galaxy_jepa.callbacks.checkpoint import TrainCheckpointer
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        os.environ["RES_LOCAL"] = str(t / "bucket")
        r = remote()
        writetest(r)
        run = t / "run"
        class Obj(torch.nn.Module):  # the least an objective must be for TrainCheckpointer
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(4, 4)
                self.encoder.config = {"toy": True}

            def checkpoint_modules(self):
                return {"encoder": self.encoder}

        net = Obj()
        opt = torch.optim.Adam(net.parameters())
        ck = TrainCheckpointer(run / "checkpoints", every=10, keep=99, config_hash="h", normalisation_hash="n", schedule={})
        assert ckpt_push(r, run, "arms/x") is None
        for s in (10, 20):
            ck.save(step=s, objective=net, optimiser=opt, losses=[1.0] * s, collapse_history=[])
            assert ckpt_push(r, run, "arms/x") == s
        assert sorted(ls(r, "arms/x/checkpoints")) == ["checkpoints.json", "ckpt_000000020.pt"], ls(r, "arms/x/checkpoints")
        fresh = t / "fresh"
        assert ckpt_pull(r, "arms/x", fresh) == 20
        st = TrainCheckpointer(fresh / "checkpoints", every=10, keep=99, config_hash="h", normalisation_hash="n", schedule={}).restore(optimiser=torch.optim.Adam(net.parameters()), objective=net)
        assert st.step == 20 and len(st.losses) == 20
        assert ckpt_pull(r, "arms/none", t / "none") is None
        bad = t / "bucket/arms/x/checkpoints"
        f = next(bad.glob("ckpt_*.pt"))
        b = bytearray(f.read_bytes())
        b[len(b) // 2] ^= 0xFF
        f.write_bytes(bytes(b))
        try:
            ckpt_pull(r, "arms/x", t / "tampered")
            raise AssertionError("a tampered checkpoint was accepted")
        except SystemExit as e:
            assert "sha256" in str(e)
        (t / "d/a").mkdir(parents=True)
        (t / "d/a/e.npz").write_bytes(b"12")
        (t / "d/s.json").write_text("{}")
        assert putdir(r, t / "d", "out") == 2 and putdir(r, t / "d", "out") == 1  # the .npz is skipped the second time
        assert getdir(r, "out", t / "back") == 2 and (t / "back/a/e.npz").read_bytes() == b"12"
        assert exists(r, "out/s.json") and not exists(r, "out/t.json")
    print("results selftest: PASS")


def main() -> None:
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "selftest":
        return selftest()
    r = remote()
    m = a[0]
    if m == "writetest":
        writetest(r)
    elif m == "put":
        put(r, Path(a[1]), a[2])
    elif m == "putdir":
        print(f"{putdir(r, Path(a[1]), a[2])} files up")
    elif m == "get":
        get(r, a[1], Path(a[2]))
    elif m == "getdir":
        print(f"{getdir(r, a[1], Path(a[2]))} files down")
    elif m == "exists":
        raise SystemExit(0 if exists(r, a[1]) else 1)
    elif m == "ls":
        print("\n".join(f"{v:>14,}  {k}" for k, v in sorted(ls(r, a[1] if len(a) > 1 else "").items())))
    elif m == "ckpt-push":
        print(f"checkpoint step {ckpt_push(r, Path(a[1]), a[2])} up")
    elif m == "ckpt-pull":
        s = ckpt_pull(r, a[1], Path(a[2]))
        print(f"checkpoint step {s} restored" if s else "no checkpoint in the bucket")
        raise SystemExit(0 if s else 3)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
