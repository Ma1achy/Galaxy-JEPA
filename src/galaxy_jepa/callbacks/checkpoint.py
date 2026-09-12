"""Mid-run checkpointing — so a crash at hour eight costs twenty minutes, not the run.

``train_jepa`` wrote one checkpoint, at the end. Brief F measured the configured budget at
**10.7 h** on the M3 Pro and anything larger in days, so that arrangement made every long run an
all-or-nothing bet. This is the behaviour *inserted around* the loop, which is why it lives in
``callbacks/`` (``docs/architecture.md`` control-flow ownership) rather than inside the objective.

**Resuming an I-JEPA needs more than the model.** The EMA target encoder is a separate set of
weights with its own trajectory; losing it and re-deriving it from the online encoder would
restart the EMA from a copy, silently changing training while appearing to work. So the payload
carries the online encoder, the target encoder, the predictor, the AdamW moments, the step (which
*is* the position in both the LR warmup and the EMA cosine ramp), the RNG states, and the two
hashes that say which experiment and which normalisation statistic this was.

**The manifest is the commit point**, exactly as ``index.json`` is for the fp16 cache. A payload
is written to a temp path, fsynced, renamed (atomic on POSIX), digested and read back; only then
is the manifest rewritten to name it. A payload file the manifest does not vouch for is
uncommitted and ignored; a manifest entry whose file is missing or whose digest disagrees is
**refused loudly**. This is the torn-cache defect (``data.cache._assert_untorn``) applied to a
different artefact: an interrupted write must fail at load, never half-load.

**A resume cannot silently cross a freeze.** ``config_hash`` and ``normalisation_hash`` are
checked before any weights are touched, and so are the schedule parameters — resuming a run whose
``steps`` changed would move every later EMA momentum and LR, which is a different experiment
wearing the same step number.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = ["ResumeState", "TrainCheckpointer", "CHECKPOINT_SCHEMA"]

#: Bumped when the payload's shape changes. A mismatch is refused, not migrated — a resume is
#: not the place to discover that two versions disagree about what a checkpoint contains.
CHECKPOINT_SCHEMA = 1

_MANIFEST = "checkpoints.json"
_STEM = "ckpt_"


@dataclasses.dataclass(frozen=True)
class ResumeState:
    """What a resume recovers beyond the weights: where it was, and what it had recorded."""

    step: int
    losses: list[float]
    collapse_history: list[dict[str, float]]
    path: Path


def _sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _fsync_dir(path: Path) -> None:
    """Durability of a rename is the *directory's* metadata, not the file's."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - some filesystems refuse; the rename still happened
        pass
    finally:
        os.close(fd)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    with tmp.open("rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def _set_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


class TrainCheckpointer:
    """Periodic, atomic, self-verifying checkpoints under one directory.

    ``every`` is in steps. Brief F measured 1.297 steps/s at batch 32, so the default of 1,500
    bounds a crash's cost to about 19 minutes of work; the write itself was measured at well
    under a percent of throughput at that interval (see the G1 report).
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        every: int,
        config_hash: str,
        normalisation_hash: str,
        schedule: dict[str, Any],
        keep: int = 3,
    ):
        if every <= 0:
            raise ValueError("checkpoint interval must be positive; use `every=None` upstream")
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.every = int(every)
        self.keep = max(1, int(keep))
        self.config_hash = config_hash
        self.normalisation_hash = normalisation_hash
        #: The LR warmup and the EMA cosine ramp are pure functions of ``(step, these)``, so
        #: recording them is recording the schedule *position*, and a change to them invalidates
        #: every later momentum. Checked on resume.
        self.schedule = dict(schedule)
        self.write_seconds = 0.0
        self.writes = 0

    # --- reading -------------------------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.dir / _MANIFEST

    def _manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        return dict(json.loads(self.manifest_path.read_text()))

    def latest(self) -> dict[str, Any] | None:
        """The newest *committed* entry, or ``None``. Uncommitted payload files are ignored."""
        manifest = self._manifest()
        entries = list(manifest.get("entries") or []) if manifest is not None else []
        if not entries:
            return None
        return max(entries, key=lambda e: int(e["step"]))

    # --- writing -------------------------------------------------------------------------

    def save(
        self,
        *,
        step: int,
        jepa: Any,
        optimiser: torch.optim.Optimizer,
        losses: list[float],
        collapse_history: list[dict[str, float]],
    ) -> Path:
        """Write a checkpoint for ``step`` completed steps, then commit it in the manifest."""
        t0 = time.perf_counter()
        path = self.dir / f"{_STEM}{step:09d}.pt"
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "step": int(step),
            "encoder_config": jepa.encoder.config,
            "encoder": jepa.encoder.state_dict(),
            # separate on purpose: re-deriving this from the online encoder would restart the
            # EMA from a copy and change training while looking like a clean resume
            "target_encoder": jepa.target_encoder.state_dict(),
            "predictor": jepa.predictor.state_dict(),
            "optimiser": optimiser.state_dict(),
            "schedule": self.schedule,
            "rng": _rng_state(),
            "config_hash": self.config_hash,
            "normalisation_hash": self.normalisation_hash,
            "losses": list(losses),
            "collapse_history": [dict(r) for r in collapse_history],
        }
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        with tmp.open("rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        _fsync_dir(self.dir)

        digest = _sha256_file(path)
        self._assert_loadable(path, step)

        manifest = self._manifest() or {}
        entries = [e for e in (manifest.get("entries") or []) if int(e["step"]) != int(step)]
        entries.append(
            {
                "step": int(step),
                "file": path.name,
                "sha256": digest,
                "bytes": path.stat().st_size,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
        entries.sort(key=lambda e: int(e["step"]))
        keep = entries[-self.keep :]
        _write_json_atomic(
            self.manifest_path,
            {
                "schema": CHECKPOINT_SCHEMA,
                "config_hash": self.config_hash,
                "normalisation_hash": self.normalisation_hash,
                "schedule": self.schedule,
                "entries": keep,
            },
        )
        # only now that the manifest no longer names them: anything uncommitted is prunable
        self._prune({e["file"] for e in keep})
        self.write_seconds += time.perf_counter() - t0
        self.writes += 1
        logger.info(
            "checkpoint step %d: %.0f MB, sha %s, %.1fs",
            step,
            path.stat().st_size / 1e6,
            digest[:12],
            time.perf_counter() - t0,
        )
        return path

    def _assert_loadable(self, path: Path, step: int) -> None:
        """Read it back before vouching for it. ``mmap`` so a 400 MB payload is not materialised
        just to check that it is a well-formed archive naming the right step."""
        try:
            probe = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except Exception as exc:  # pragma: no cover - a torn write is what this guards
            raise RuntimeError(
                f"checkpoint {path.name} was written but does not load back ({exc}). Not "
                "committing it to the manifest — a checkpoint that cannot be read is worse than "
                "none, because it would be trusted."
            ) from exc
        if int(probe.get("step", -1)) != int(step) or probe.get("schema") != CHECKPOINT_SCHEMA:
            raise RuntimeError(
                f"checkpoint {path.name} read back as step {probe.get('step')} "
                f"schema {probe.get('schema')}, expected {step}/{CHECKPOINT_SCHEMA}"
            )

    def _prune(self, keep_files: set[str]) -> None:
        for f in self.dir.glob(f"{_STEM}*.pt"):
            if f.name not in keep_files:
                f.unlink(missing_ok=True)
        for f in self.dir.glob(f"{_STEM}*.tmp"):  # an interrupted write, never committed
            f.unlink(missing_ok=True)

    # --- resuming ------------------------------------------------------------------------

    def restore(
        self, *, jepa: Any, optimiser: torch.optim.Optimizer, map_location: str = "cpu"
    ) -> ResumeState | None:
        """Load the newest committed checkpoint into ``jepa``/``optimiser``, or return ``None``.

        Every refusal here is loud. A resume that proceeds past a mismatch is the failure mode
        this whole module exists to prevent: it would look like the same run and not be one.
        """
        entry = self.latest()
        if entry is None:
            return None
        path = self.dir / str(entry["file"])
        if not path.exists():
            raise FileNotFoundError(
                f"the manifest commits step {entry['step']} to {path.name} but the file is gone. "
                "Refusing to fall back to an older checkpoint silently — say which one to use."
            )
        digest = _sha256_file(path)
        if digest != entry["sha256"]:
            raise RuntimeError(
                f"{path.name} hashes to {digest[:12]}, not the {str(entry['sha256'])[:12]} its "
                "manifest vouches for. The payload changed under a manifest that did not; this is "
                "a torn or tampered checkpoint and it will not be loaded."
            )
        payload = torch.load(path, map_location=map_location, weights_only=False)
        self._assert_compatible(payload, path)

        jepa.encoder.load_state_dict(payload["encoder"])
        jepa.target_encoder.load_state_dict(payload["target_encoder"])
        jepa.predictor.load_state_dict(payload["predictor"])
        optimiser.load_state_dict(payload["optimiser"])
        _set_rng_state(payload["rng"])
        step = int(payload["step"])
        logger.info("resumed from %s at step %d", path.name, step)
        return ResumeState(
            step=step,
            losses=list(payload["losses"]),
            collapse_history=[dict(r) for r in payload["collapse_history"]],
            path=path,
        )

    def _assert_compatible(self, payload: dict[str, Any], path: Path) -> None:
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise RuntimeError(
                f"{path.name} is schema {payload.get('schema')}, this code writes "
                f"{CHECKPOINT_SCHEMA}. Refusing to guess what moved."
            )
        if payload.get("config_hash") != self.config_hash:
            raise RuntimeError(
                f"{path.name} was written under config_hash "
                f"{str(payload.get('config_hash'))[:16]} but this run is "
                f"{self.config_hash[:16]}. That is a different experiment; resuming across it "
                "would produce a run whose stamp describes neither half."
            )
        if payload.get("normalisation_hash") != self.normalisation_hash:
            raise RuntimeError(
                f"{path.name} was trained on stamps normalised under "
                f"{str(payload.get('normalisation_hash'))[:12] or '(unrecorded)'} but this run "
                f"carries {self.normalisation_hash[:12]}. A resume may not cross a freeze: the "
                "encoder's inputs would change mid-training with nothing recording it."
            )
        theirs, mine = dict(payload.get("schedule") or {}), self.schedule
        moved = {k: (theirs.get(k), mine.get(k)) for k in mine if theirs.get(k) != mine.get(k)}
        if moved:
            raise RuntimeError(
                f"{path.name}'s schedule differs from this run's: {moved}. The LR warmup and the "
                "EMA cosine ramp are functions of (step, these), so every momentum after the "
                "resume point would differ from the run being continued."
            )
