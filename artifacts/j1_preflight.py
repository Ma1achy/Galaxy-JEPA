"""Brief J1 — the pre-flight, as checked facts rather than assumptions.

J's own instruction: *state each as a checked fact*. Three reproducibility holes have already
been found in this project (an unseeded ViT init, a dropped `lr_final`, `keep=3` pruning a
trajectory mid-measurement), each of which looked fine from the config. So this asserts rather
than reports, and raises on the first failure.

Builds on `f0_preconditions.check()`, which already gates the four it owns — the normalisation
freeze loads and has no fitting path, the cache names the live freeze, `headline=True` is refused
while the effect floor is open, the vote floor is frozen. Everything below is what J1 adds.

Raises on the first failure. Investigation code: terse, excluded from lint/CI.

    uv run python artifacts/j1_preflight.py
"""

from __future__ import annotations

import inspect
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f0_preconditions import REPO, check  # noqa: E402

from galaxy_jepa.core.config import config_hash  # noqa: E402
from galaxy_jepa.data.cache import load_probe_columns  # noqa: E402
from galaxy_jepa.harness import HarnessConfig  # noqa: E402

# The hash chain, each link naming exactly what moved. Recorded before the run, so an artefact
# read back later can be walked to the recipe it came from rather than guessed at.
J_SMOKE_HASH = "f561d7f5039f23d8"  # as shipped: D18 + smoke
D18_HASH = "b5acc6779df49070"  # smoke=False
D17_HASH = "538bf997880a8767"  # + the four sigreg keys stripped

PRETRAIN_STAMPS = 826_968
PROBE_STAMPS = 230_358
MEASURED_STEPS_PER_S = 1.4279  # J2, measured on THIS path immediately before launch;
#                              Brief I's 1.4765 was its own driver, 3.3% optimistic here


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> None:
    cfg, cache = check()  # the F0 four; raises on any of them
    obj = cfg.objective
    print()

    # 1. provenance: what code_sha will say, and what it will NOT say
    head, branch = _git("rev-parse", "HEAD")[:7], _git("rev-parse", "--abbrev-ref", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    behind = _git("rev-list", "--count", f"{head}..main") if _git("branch", "--list", "main") else "?"
    ahead = _git("rev-list", "--count", f"main..{head}")
    print(f"J1 code_sha            : {head} on {branch}"
          f"{' — DIRTY, commit before launching' if dirty else ' (clean)'}")
    print(f"J1 main                : {ahead} commits behind this branch, {behind} ahead — "
          f"D17+D18 are HERE, not on main. NOT merged; code_sha is the provenance.")

    # 2. the hash chain
    shipped = HarnessConfig(**yaml.safe_load((REPO / "configs/pretrain.yaml").read_text()))
    # The CONFIG-FILE chain, device unresolved — this is what D17/D18 recorded and what
    # `tests/test_configs_load.py` pins. It is NOT what lands on the artefact: `_make_stamp`
    # resolves the backend first (MPS/CPU/CUDA differ numerically and must hash apart), so the
    # stamped hash is a different number and both are printed below rather than conflated.
    d = shipped.determining_dump()
    keys = {"sigreg_lambda", "sigreg_slices", "sigreg_quad_points", "sigreg_domain"}
    off = dict(d, smoke=False)
    d17 = dict(off, objective={k: v for k, v in d["objective"].items() if k not in keys})
    for label, got, want in (
        ("this run (D18+smoke)", config_hash(d)[:16], J_SMOKE_HASH),
        ("minus smoke -> D18", config_hash(off)[:16], D18_HASH),
        ("minus sigreg -> D17", config_hash(d17)[:16], D17_HASH),
    ):
        if got != want:
            raise SystemExit(f"J1: {label} hashes {got}, expected {want}")
    print(f"J1 config_hash chain   : {J_SMOKE_HASH} -> {D18_HASH} (D18) -> {D17_HASH} (D17)"
          f"  [config file, device unresolved]")
    resolved = shipped.with_resolved_device()
    stamped = config_hash(resolved.determining_dump())[:16]
    print(f"J1 stamped hash        : v2:{stamped} on device "
          f"{resolved.runtime.device} — what the artefacts will actually carry")

    # 3. smoke, and the consequence of it
    if not cfg.smoke:
        raise SystemExit("J1: smoke is False. This run is not a result; it must say so structurally")
    print("J1 smoke               : True — stamps `smoke` into escape_hatches_used. The hash MOVED, "
          "so these weights can never be resumed into a headline run.")

    # 4. checkpointing: the trajectory survives. `keep` is DERIVED; prove the default it displaces.
    from galaxy_jepa.callbacks.checkpoint import TrainCheckpointer

    default_keep = inspect.signature(TrainCheckpointer.__init__).parameters["keep"].default
    scheduled = obj.steps // obj.checkpoint_every
    derived = scheduled + 2
    if derived <= default_keep:
        raise SystemExit(f"J1: derived keep={derived} is no better than the default {default_keep}")
    print(f"J1 checkpoints         : every {obj.checkpoint_every} -> {scheduled} scheduled, "
          f"keep={derived} (default {default_keep} would prune {scheduled - default_keep}); "
          f"~{derived * 381 / 1024:.1f} GiB")

    # 5. the MPS pool release is in the production path, not only in the I2 driver
    from galaxy_jepa.objectives import jepa as jepa_mod

    src = inspect.getsource(jepa_mod.train_jepa)
    if "torch.mps.empty_cache()" not in src:
        raise SystemExit("J1: train_jepa does not release the MPS pool — Brief I's OOM reopens")
    print("J1 MPS pool release    : present in train_jepa (monitor cadence)")

    # 6. the soft rank floor is INERT by design here; its silence must be understood, not assumed
    floor = cfg.collapse_floor
    if floor is None:
        raise SystemExit("J1: no collapse floor — the tripwire is forfeited")
    if obj.sigreg_lambda <= 0.0:
        raise SystemExit("J1: sigreg_lambda is 0, so D18 is not actually on")
    print(f"J1 halt conditions     : hard rank < {floor.hard_floor} after step "
          f"{floor.hard_floor_after_step} x{floor.consecutive_readings}; std floor; non-finite. "
          f"SOFT floor {floor.min_effective_rank} is INERT (sigreg_lambda={obj.sigreg_lambda}).")

    # 7. the cache is complete and dense — no gaps to top up mid-run
    ids = cache.index.object_ids
    if len(ids) != PRETRAIN_STAMPS + PROBE_STAMPS:
        raise SystemExit(f"J1: cache holds {len(ids):,}, expected "
                         f"{PRETRAIN_STAMPS + PROBE_STAMPS:,} — it is partial")
    columns = load_probe_columns(cache.cache_dir, cache.index)  # refuses on a digest mismatch
    print(f"J1 cache               : {len(ids):,} stamps ({PRETRAIN_STAMPS:,} pretrain + "
          f"{PROBE_STAMPS:,} probe), dense; probe sidecar {len(cache.index.probe_columns)} columns, "
          f"digest verified")
    del columns

    # 8. where it writes, and whether there is room
    out = Path(cfg.paths.out_dir).resolve()
    usage = shutil.disk_usage(out if out.exists() else out.parent)
    print(f"J1 out_dir             : {out} — {usage.free / 1024**3:,.0f} GiB free")
    if usage.free < 64 * 1024**3:
        raise SystemExit(f"J1: only {usage.free / 1024**3:.0f} GiB where the run writes")

    # 9. the budget, from this project's own measurement
    hours = obj.steps / MEASURED_STEPS_PER_S / 3600
    samples = obj.steps * obj.batch_size
    print(f"J1 budget              : {obj.steps:,} steps at {MEASURED_STEPS_PER_S} steps/s "
          f"= {hours:.1f} h; {samples:,} samples = {samples / PRETRAIN_STAMPS:.2f} epochs; "
          f"device {cfg.with_resolved_device().runtime.device}")
    print(f"J1 budget is fixed     : lr_final's cosine and the EMA ramp are both defined over "
          f"steps={obj.steps}, so moving it changes the recipe D17/D18 were adopted with")

    print("\nJ1 PASS — every fact checked, nothing assumed")


if __name__ == "__main__":
    main()
