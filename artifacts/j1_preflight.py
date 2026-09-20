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

import dataclasses
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
D17_HASH = "538bf997880a8767"  # + the four sigreg keys stripped, and UNCHANGED by D21 --
#                              stripping removes sigreg_lambda whatever its value, so this
#                              anchor holds across both the adoption and the reversal


@dataclasses.dataclass(frozen=True)
class Expect:
    """What a given brief's pre-flight asserts. The checks are shared; the answers are not.

    The second consumer arrived at Brief M, which runs the SAME gate against a different recipe:
    D21 turns SIGReg off, so `sigreg_on=False` inverts check 6 and the hash chain moves. Rather
    than fork the file — J1 is the record of what Brief J actually checked, and editing it would
    rewrite that — the answers become an argument and the default reproduces J exactly.
    """

    label: str
    shipped_hash: str  # the config exactly as written
    smoke_hash: str  # the RECIPE with smoke still on (== shipped_hash when the budget is stock)
    recipe_hash: str  # minus smoke
    sigreg_on: bool  # is the penalty meant to be computed?
    steps_per_s: float  # measured on THIS path, for the budget line
    decisions: str  # which D-entries this recipe carries, for the provenance line
    #: The stock budget, when this brief moved it. Strip it and the recipe chain below is
    #: reachable again; leave it None when the brief runs at the shipped budget. `steps` and
    #: `checkpoint_every` are both determining, so a brief that changes the horizon changes the
    #: hash — the chain gains a LINK rather than losing its anchors, and each strip still names
    #: exactly one thing that moved.
    stock_budget: dict[str, int] | None = None


J = Expect(
    label="J1",
    shipped_hash=J_SMOKE_HASH,
    smoke_hash=J_SMOKE_HASH,
    recipe_hash=D18_HASH,
    sigreg_on=True,
    steps_per_s=1.4279,  # J2, measured immediately before launch; Brief I's 1.4765 was its own
    #                      driver and 3.3% optimistic here
    decisions="D17+D18",
)

PRETRAIN_STAMPS = 826_968
PROBE_STAMPS = 230_358


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def main(expect: Expect = J) -> None:
    cfg, cache = check()  # the F0 four; raises on any of them
    obj = cfg.objective
    lb = expect.label
    print()

    # 1. provenance: what code_sha will say, and what it will NOT say
    head, branch = _git("rev-parse", "HEAD")[:7], _git("rev-parse", "--abbrev-ref", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    behind = _git("rev-list", "--count", f"{head}..main") if _git("branch", "--list", "main") else "?"
    ahead = _git("rev-list", "--count", f"main..{head}")
    print(f"{lb} code_sha            : {head} on {branch}"
          f"{' — DIRTY, commit before launching' if dirty else ' (clean)'}")
    print(f"{lb} main                : {ahead} commits behind this branch, {behind} ahead — "
          f"{expect.decisions} are HERE, not on main. NOT merged; code_sha is the provenance.")

    # 2. the hash chain
    shipped = HarnessConfig(**yaml.safe_load((REPO / "configs/pretrain.yaml").read_text()))
    # The CONFIG-FILE chain, device unresolved — this is what D17/D18 recorded and what
    # `tests/test_configs_load.py` pins. It is NOT what lands on the artefact: `_make_stamp`
    # resolves the backend first (MPS/CPU/CUDA differ numerically and must hash apart), so the
    # stamped hash is a different number and both are printed below rather than conflated.
    d = shipped.determining_dump()
    keys = {"sigreg_lambda", "sigreg_slices", "sigreg_quad_points", "sigreg_domain"}
    links = [(f"this run ({expect.decisions}+smoke)", d, expect.shipped_hash)]
    if expect.stock_budget is not None:
        d = dict(d, objective={**d["objective"], **expect.stock_budget})
        links.append(("minus budget -> recipe+smoke", d, expect.smoke_hash))
    off = dict(d, smoke=False)
    d17 = dict(off, objective={k: v for k, v in off["objective"].items() if k not in keys})
    links += [
        ("minus smoke -> recipe", off, expect.recipe_hash),
        ("minus sigreg -> D17", d17, D17_HASH),
    ]
    for label, dump, want in links:
        got = config_hash(dump)[:16]
        if got != want:
            raise SystemExit(f"{lb}: {label} hashes {got}, expected {want}")
    print(f"{lb} config_hash chain   : " + " -> ".join(w for _, _, w in links)
          + f"  [{len(links)} links, config file, device unresolved]")
    resolved = shipped.with_resolved_device()
    stamped = config_hash(resolved.determining_dump())[:16]
    print(f"{lb} stamped hash        : v2:{stamped} on device "
          f"{resolved.runtime.device} — what the artefacts will actually carry")

    # 3. smoke, and the consequence of it
    if not cfg.smoke:
        raise SystemExit(f"{lb}: smoke is False. This run is not a result; it must say so structurally")
    print(f"{lb} smoke               : True — stamps `smoke` into escape_hatches_used. The hash MOVED, "
          "so these weights can never be resumed into a headline run.")

    # 4. checkpointing: the trajectory survives. `keep` is DERIVED; prove the default it displaces.
    from galaxy_jepa.callbacks.checkpoint import TrainCheckpointer

    default_keep = inspect.signature(TrainCheckpointer.__init__).parameters["keep"].default
    scheduled = obj.steps // obj.checkpoint_every
    derived = scheduled + 2
    if derived <= default_keep:
        raise SystemExit(f"{lb}: derived keep={derived} is no better than the default {default_keep}")
    print(f"{lb} checkpoints         : every {obj.checkpoint_every} -> {scheduled} scheduled, "
          f"keep={derived} (default {default_keep} would prune {scheduled - default_keep}); "
          f"~{derived * 381 / 1024:.1f} GiB")

    # 5. the MPS pool release is in the production path, not only in the I2 driver
    from galaxy_jepa.objectives import jepa as jepa_mod

    src = inspect.getsource(jepa_mod.train_jepa)
    if "torch.mps.empty_cache()" not in src:
        raise SystemExit(f"{lb}: train_jepa does not release the MPS pool — Brief I's OOM reopens")
    print(f"{lb} MPS pool release    : present in train_jepa (monitor cadence)")

    # 6. the soft rank floor is INERT by design here; its silence must be understood, not assumed
    floor = cfg.collapse_floor
    if floor is None:
        raise SystemExit(f"{lb}: no collapse floor — the tripwire is forfeited")
    on = obj.sigreg_lambda > 0.0
    if on != expect.sigreg_on:
        want = "on" if expect.sigreg_on else "off"
        raise SystemExit(
            f"{lb}: sigreg_lambda is {obj.sigreg_lambda}, but this brief runs with SIGReg {want}"
        )
    # The soft rank floor's state follows from lambda and `train_jepa` derives it the same way.
    # Under D18 it was INERT and its silence had to be understood rather than assumed; under D21
    # it is ACTIVE and is a real halt path. Brief L is the evidence 2.5 is not set too high --
    # effective rank rose 8.1 -> 18.6 across 10,500 steps, away from it, and it did not fire.
    soft = "INERT" if on else "ACTIVE — it can fire, and this is the first project regime where it could"
    print(f"{lb} halt conditions     : hard rank < {floor.hard_floor} after step "
          f"{floor.hard_floor_after_step} x{floor.consecutive_readings}; std floor; non-finite. "
          f"SOFT floor {floor.min_effective_rank} is {soft} (sigreg_lambda={obj.sigreg_lambda}).")

    # 7. the cache is complete and dense — no gaps to top up mid-run
    ids = cache.index.object_ids
    if len(ids) != PRETRAIN_STAMPS + PROBE_STAMPS:
        raise SystemExit(f"{lb}: cache holds {len(ids):,}, expected "
                         f"{PRETRAIN_STAMPS + PROBE_STAMPS:,} — it is partial")
    columns = load_probe_columns(cache.cache_dir, cache.index)  # refuses on a digest mismatch
    print(f"{lb} cache               : {len(ids):,} stamps ({PRETRAIN_STAMPS:,} pretrain + "
          f"{PROBE_STAMPS:,} probe), dense; probe sidecar {len(cache.index.probe_columns)} columns, "
          f"digest verified")
    del columns

    # 8. where it writes, and whether there is room
    out = Path(cfg.paths.out_dir).resolve()
    usage = shutil.disk_usage(out if out.exists() else out.parent)
    print(f"{lb} out_dir             : {out} — {usage.free / 1024**3:,.0f} GiB free")
    if usage.free < 64 * 1024**3:
        raise SystemExit(f"{lb}: only {usage.free / 1024**3:.0f} GiB where the run writes")

    # 9. the budget, from this project's own measurement
    hours = obj.steps / expect.steps_per_s / 3600
    samples = obj.steps * obj.batch_size
    # An epoch is a pass over what the model TRAINS on, which is the corpus less the monitor
    # slice — dividing by the whole corpus reports 9.80 epochs for a 10-epoch budget.
    train_n = PRETRAIN_STAMPS - round(PRETRAIN_STAMPS * cfg.monitor_frac)
    print(f"{lb} budget              : {obj.steps:,} steps at {expect.steps_per_s} steps/s "
          f"= {hours:.1f} h; {samples:,} samples = {samples / train_n:.2f} epochs "
          f"({train_n:,} train, corpus less the {cfg.monitor_frac:.0%} monitor slice); "
          f"device {cfg.with_resolved_device().runtime.device}")
    print(f"{lb} budget is fixed     : lr_final's cosine and the EMA ramp are both defined over "
          f"steps={obj.steps}, so moving it changes the recipe. Stopping early is allowed; extending is not")

    # 10. the training path has been shown to memorise a batch, for THIS recipe.
    #
    # Brief O3. A check that can be skipped silently is not a check, so the gate is required here
    # rather than left as a script someone remembers to run: no long job starts without a passing
    # record keyed on the recipe hash. Imported lazily — o3 pulls in the objective and the masker,
    # which a pre-flight has no other reason to build.
    from o3_overfit_gate import assert_gate_passed  # noqa: PLC0415

    assert_gate_passed(cfg, label=lb)

    print(f"\n{lb} PASS — every fact checked, nothing assumed")


if __name__ == "__main__":
    main()
