"""Probe banks (.npz embedding banks) that carry their encoder's SHA-1 — written once, checked on load.

A bank used to record its checkpoint as a path string only, so a checkpoint overwritten in place (same
path, new weights) left every bank built on the old weights looking current, and a loader handed an
encoder by environment variable (dd_core's DD_ENCODER) never compared it to the bank at all. Now:

  * `write_bank` stores `checkpoint` (the path, or an untrained encoder's descriptor), `checkpoint_sha1`
    (SHA-1 of the checkpoint file's bytes; empty for an untrained encoder, which has no file) and
    `checkpoint_provenance` ('built' | 'backfilled YYYY-MM-DD: …' | 'untrained'). Uncompressed, via a
    temporary file and an atomic rename, so plate_export can still memory-map `x`.
  * `load_bank` recomputes the SHA-1 of the recorded checkpoint when the file is present and raises on
    a mismatch; given `expected_checkpoint` it also requires that checkpoint's SHA-1 (or, for an
    untrained bank, that descriptor) to be the bank's. A bank without the field raises: the one-off
    backfill below is the only legacy path, so there is no `legacy=True` for a caller to reach for.

Backfill (one-off, for banks built before this module):

  uv run python artifacts/probe_bank.py backfill [bank.npz ...]   # default: every bank in artifacts/out

A bank that records a checkpoint path gets the SHA-1 of that file *as it is today*. Whether the bank
really came from those bytes cannot be checked cheaply (it would mean re-embedding), so the provenance
says 'backfilled' with the date rather than 'built'. The arrays are not rewritten: every existing member
is copied byte-for-byte into a new file, which is verified member-by-member and array-by-array before
the atomic rename.
"""

from __future__ import annotations

import datetime as dt
import functools
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "out"
META = ("checkpoint", "checkpoint_sha1", "checkpoint_provenance")
UNTRAINED = "untrained"


class BankProvenanceError(RuntimeError):
    """A bank whose encoder cannot be shown to be the one it claims (or the one the caller expects)."""


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p  # banks record checkpoints relative to the repo root


@functools.lru_cache(maxsize=16)
def _sha1_cached(path: str, mtime_ns: int, size: int) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha1(path: str | Path) -> str:
    """SHA-1 of a file's bytes; cached on (path, mtime, size) since several loaders read one bank."""
    p = _resolve(path).resolve()
    st = p.stat()
    return _sha1_cached(str(p), st.st_mtime_ns, st.st_size)


def untrained_descriptor(config: dict, seed: int | str) -> str:
    """What stands in for a checkpoint when the encoder is a fresh init: seed + architecture."""
    blob = json.dumps(dict(config), sort_keys=True, default=str).encode()
    return f"{UNTRAINED} seed {seed}, config sha1 {hashlib.sha1(blob).hexdigest()}"


def _atomic_npz(path: Path, write, verify=None) -> None:
    """Write to a sibling temporary file, optionally verify it, then rename over `path` — a reader never
    sees a half-written bank, and a failed verification leaves the original untouched."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        if verify is not None:
            verify(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_bank(path: str | Path, *, ids: np.ndarray, x: np.ndarray | None, checkpoint: str | Path,
               **extra: Any) -> None:
    """Write a bank stamped with its encoder. `checkpoint` is a file (hashed now) or an
    `untrained_descriptor`. `x` may be None for banks whose matrices live under other keys (O1's
    'real'/'untrained', R's 'seed1'/'seed2')."""
    clash = set(extra) & {*META, "ids", "x"}
    if clash:
        raise ValueError(f"write_bank: {sorted(clash)} are reserved")
    ck = str(checkpoint)
    if ck.startswith(UNTRAINED):
        sha, prov = "", UNTRAINED
    else:
        sha, prov = file_sha1(ck), "built"
    arrays = {"ids": ids, **({} if x is None else {"x": x}), **extra,
              "checkpoint": ck, "checkpoint_sha1": sha, "checkpoint_provenance": prov}
    _atomic_npz(Path(path), lambda f: np.savez(f, **arrays))


def load_bank(path: str | Path, expected_checkpoint: str | Path | None = None):
    """np.load(path) once the bank's encoder checks out; returns the (lazy) NpzFile.

    Raises BankProvenanceError when: the bank has no `checkpoint_sha1`; the recorded checkpoint file
    exists and no longer hashes to it; `expected_checkpoint` names a file whose SHA-1 differs, or a
    descriptor that differs, or names a file for an untrained bank (or vice versa)."""
    path = Path(path)
    b = np.load(path, allow_pickle=False)
    try:
        if "checkpoint_sha1" not in b.files:
            raise BankProvenanceError(
                f"{path.name} has no checkpoint_sha1 — built before banks were stamped. Run "
                f"`uv run python artifacts/probe_bank.py backfill {path}` (records it as backfilled).")
        ck, sha = str(b["checkpoint"]), str(b["checkpoint_sha1"])
        untrained = ck.startswith(UNTRAINED)
        if untrained != (sha == ""):
            raise BankProvenanceError(f"{path.name}: checkpoint {ck!r} and checkpoint_sha1 {sha!r} disagree")
        if not untrained and _resolve(ck).exists() and file_sha1(ck) != sha:
            raise BankProvenanceError(
                f"{path.name} was built on {ck} with SHA-1 {sha}, but that file now hashes to "
                f"{file_sha1(ck)} — the checkpoint changed under the bank; re-embed")
        if expected_checkpoint is not None:
            want = str(expected_checkpoint)
            if want.startswith(UNTRAINED) or untrained:
                if want != ck:
                    raise BankProvenanceError(f"{path.name} is for {ck!r}, not {want!r}")
            elif file_sha1(want) != sha:
                raise BankProvenanceError(
                    f"{path.name} was built on {ck} (SHA-1 {sha}); the expected checkpoint {want} hashes "
                    f"to {file_sha1(want)} — a different encoder")
        if not untrained and not _resolve(ck).exists() and expected_checkpoint is None:
            print(f"load_bank: {path.name}'s checkpoint {ck} is not on disk; SHA-1 not re-verified",
                  file=sys.stderr)
    except BaseException:
        b.close()
        raise
    return b


# ── backfill ──────────────────────────────────────────────────────────────────────────────────────────

def _npy_bytes(value: Any) -> bytes:
    buf = io.BytesIO()
    np.lib.format.write_array(buf, np.asanyarray(value), allow_pickle=False)
    return buf.getvalue()


def add_members(path: Path, new: dict[str, Any]) -> None:
    """Append members to an npz without touching the existing ones: copy each byte-for-byte (same
    compression) into a new file, add `new`, verify, then atomically replace."""
    with zipfile.ZipFile(path) as z:
        old = [(i, z.read(i.filename)) for i in z.infolist()]
    names = {i.filename for i, _ in old}
    if any(f"{k}.npy" in names for k in new):
        raise ValueError(f"{path.name} already has one of {sorted(new)}")

    def write(f):
        with zipfile.ZipFile(f, "w", allowZip64=True) as z:
            for info, data in old:
                z.writestr(info, data, compress_type=info.compress_type)
            for k, v in new.items():
                z.writestr(f"{k}.npy", _npy_bytes(v), compress_type=zipfile.ZIP_STORED)

    def verify(tmp: Path) -> None:  # member bytes and compression, then the arrays themselves
        with zipfile.ZipFile(tmp) as z:
            for info, data in old:
                if z.read(info.filename) != data or z.getinfo(info.filename).compress_type != info.compress_type:
                    raise RuntimeError(f"{path.name}: member {info.filename} changed in the copy")
        with np.load(path, allow_pickle=False) as a, np.load(tmp, allow_pickle=False) as c:
            if set(c.files) != set(a.files) | set(new):
                raise RuntimeError(f"{path.name}: keys changed in the copy")
            for k in a.files:
                if a[k].dtype != c[k].dtype or not np.array_equal(a[k], c[k]):
                    raise RuntimeError(f"{path.name}: array {k} changed in the copy")

    _atomic_npz(path, write, verify)


def backfill(path: Path, *, today: str | None = None, untrained_config: dict | None = None,
             model_key=None) -> str:
    """Stamp one legacy bank. Returns what was done, for the report."""
    today = today or dt.date.today().isoformat()
    with np.load(path, allow_pickle=False) as b:
        files = set(b.files)
        if "checkpoint_sha1" in files:
            return f"skip   {path.name}: already stamped ({b['checkpoint_provenance']})"
        if "checkpoint" in files:
            ck = str(b["checkpoint"])
            if ck.startswith(UNTRAINED):
                return f"skip   {path.name}: untrained bank with a free-text descriptor {ck!r}; rebuild it"
            if not _resolve(ck).exists():
                return f"skip   {path.name}: recorded checkpoint {ck} is not on disk"
            sha = file_sha1(ck)
            add_members(path, {"checkpoint_sha1": sha, "checkpoint_provenance":
                               f"backfilled {today}: SHA-1 of {ck} at backfill time; not verified to be "
                               f"the bytes the bank was built from"})
            return f"stamp  {path.name}: {ck} sha1 {sha}"
        if "model_key" in files and untrained_config is not None and model_key is not None:
            # R's untrained-seed bank: the architecture key IS checkable against the config
            if str(b["model_key"]) != model_key(untrained_config):
                return f"skip   {path.name}: model_key does not match the given architecture"
            seeds = "+".join(sorted(k.removeprefix("seed") for k in files if k.startswith("seed")))
            desc = untrained_descriptor(untrained_config, seeds)
            add_members(path, {"checkpoint": desc, "checkpoint_sha1": "", "checkpoint_provenance":
                               f"backfilled {today}: descriptor from the architecture the model_key "
                               f"matches; seeds read off the keys"})
            return f"stamp  {path.name}: {desc}"
    return f"skip   {path.name}: records no checkpoint — cannot be backfilled"


def main(argv: list[str]) -> None:
    if not argv or argv[0] != "backfill":
        raise SystemExit(__doc__)
    paths = [Path(p) for p in argv[1:]] or sorted(OUT.glob("*.npz"))
    # the untrained seeds in R's bank are M's architecture; its model_key says so or the stamp skips
    import torch
    cfg = torch.load(REPO / "runs/m/encoder.pt", map_location="cpu", weights_only=False)["config"]
    from r_nonlinear import _key
    for p in paths:
        print(backfill(p, untrained_config=cfg, model_key=_key))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main(sys.argv[1:])
