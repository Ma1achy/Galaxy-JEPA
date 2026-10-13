"""The generic-loop refactor against the code that trained M, on M's own config (user, 2026-10-11).

Baselines build step 2a: `train_jepa` became `objectives/loop.py`'s `train_objective`. The unit test
(`tests/test_objective_loop.py`) proves the identity on a tiny model; this proves it on M's recipe:
M's config and v1 cache, training seed 0, M's monitor batch, collapse floor and autocast, **500
steps** of M's schedule (`stop_after`, so the LR and EMA are M's), no checkpointer. One driver, two
code trees: each run is a subprocess whose `galaxy_jepa` comes from the tree's own `src/` (first on
PYTHONPATH, ahead of the shared venv's editable install), recorded in the output.

    loop_identity.py run <tree> <cpu|mps> <steps> <out-stem>   # one run; writes <out-stem>.json/.pt
    loop_identity.py compare <device> <old.json> <new.json> [<old-again.json>]

**The bar, fixed before any run.** CPU: losses (total, prediction), the collapse trace and every
weight of the context encoder, target encoder and predictor **bit-identical**. MPS: the same, or —
only if the pre-refactor code does not reproduce *itself* on MPS — the new run may differ from the
old by no more than the old run differs from its own repeat, on every quantity; anything larger is a
failure. So the MPS tolerance is the device's own run-to-run spread, measured, not chosen.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "runs" / "loop_identity"


def run(tree: str, device: str, steps: int, stem: str) -> None:
    env = {**os.environ, "PYTHONPATH": f"{Path(tree).resolve()}/src"}
    code = f"import sys; sys.path.insert(0, {str(HERE)!r}); import loop_identity as L; L._child({device!r}, {steps}, {stem!r})"
    subprocess.run([sys.executable, "-c", code], env=env, check=True, cwd=tree)


def _child(device: str, steps: int, stem: str) -> None:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    import galaxy_jepa
    from galaxy_jepa.data.dataset import ResumableShuffle, StampDataset
    from galaxy_jepa.harness import build_objective, seed_init
    from galaxy_jepa.objectives.jepa import _to_device, train_jepa
    from m2_long_run import MONITOR_BATCH, _split_ids, check

    cfg, cache = check(verbose=False)
    jcfg = dataclasses.replace(cfg.to_jepa_config(), seed=cfg.seed)
    train_ids, monitor_ids, _ = _split_ids(cfg)
    ds = StampDataset(cache, {}, train_ids, scalars=cache.scalars)
    sampler = ResumableShuffle(len(ds), seed=cfg.seed)
    loader = DataLoader(ds, batch_size=jcfg.batch_size, sampler=sampler, drop_last=True)
    jepa = build_objective(jcfg, seed_init(cfg.seed, cache.index.height, cfg.model.model_kwargs()))
    mon_ds = StampDataset(cache, {}, monitor_ids[:MONITOR_BATCH], scalars=cache.scalars)
    mon_batch = _to_device(next(iter(DataLoader(mon_ds, batch_size=MONITOR_BATCH))), device)
    res = train_jepa(jepa, loader, device=device, monitor_batch=mon_batch,
                     autocast_dtype=cfg.autocast_dtype(), collapse_floor=cfg.collapse_floor,
                     stop_after=steps)
    state = {f"{part}.{k}": v.detach().cpu() for part in ("encoder", "target_encoder", "predictor")
             for k, v in getattr(jepa, part).state_dict().items()}
    torch.save(state, f"{stem}.pt")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    Path(f"{stem}.json").write_text(json.dumps({
        "tree": os.getcwd(), "commit": sha, "galaxy_jepa": galaxy_jepa.__file__, "device": device,
        "steps": res.steps_completed, "halted": res.halted, "autocast": str(cfg.autocast_dtype()),
        "losses": res.losses, "prediction_losses": res.prediction_losses,
        "collapse_trace": res.collapse_trace,
        "weights_sha1": hashlib.sha1(b"".join(state[k].numpy().tobytes() for k in sorted(state))).hexdigest(),
        "finite": bool(np.isfinite(res.losses).all()),
    }, indent=1))


def _diff(a: dict, b: dict, wa, wb) -> dict:
    import numpy as np
    import torch

    def d(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if x.shape != y.shape:
            return {"abs": float("inf"), "rel": float("inf")}
        e = np.abs(x - y)
        return {"abs": float(np.nanmax(e)) if e.size else 0.0,
                "rel": float(np.nanmax(e / np.maximum(np.abs(x), 1e-30))) if e.size else 0.0}

    out = {"losses": d(a["losses"], b["losses"]),
           "prediction_losses": d(a["prediction_losses"], b["prediction_losses"])}
    for k in a["collapse_trace"]:
        out[f"collapse.{k}"] = d(a["collapse_trace"][k], b["collapse_trace"].get(k, []))
    worst = {"abs": 0.0, "rel": 0.0}
    for k in wa:
        e = (wa[k].double() - wb[k].double()).abs()
        worst["abs"] = max(worst["abs"], float(e.max()))
        worst["rel"] = max(worst["rel"], float((e / wa[k].double().abs().clamp_min(1e-30)).max()))
    out["weights"] = worst
    out["weights_bit_identical"] = a["weights_sha1"] == b["weights_sha1"] and all(
        torch.equal(wa[k], wb[k]) for k in wa)
    out["losses_bit_identical"] = a["losses"] == b["losses"] and a["prediction_losses"] == b["prediction_losses"]
    return out


def compare(device: str, old: str, new: str, again: str | None = None) -> dict:
    import torch

    load = lambda p: (json.loads(Path(p).read_text()), torch.load(Path(p).with_suffix(".pt")))  # noqa: E731
    (a, wa), (b, wb) = load(old), load(new)
    res = {"device": device, "old": {k: a[k] for k in ("tree", "commit", "galaxy_jepa", "steps")},
           "new": {k: b[k] for k in ("tree", "commit", "galaxy_jepa", "steps")},
           "new_vs_old": _diff(a, b, wa, wb)}
    identical = res["new_vs_old"]["losses_bit_identical"] and res["new_vs_old"]["weights_bit_identical"]
    if device == "cpu" or identical or again is None:
        res["pass"] = identical
        res["rule"] = "bit-identical"
    else:
        c, wc = load(again)
        res["old_vs_old_again"] = noise = _diff(a, c, wa, wc)
        nv = res["new_vs_old"]
        res["pass"] = all(nv[k][m] <= noise[k][m] for k in nv if isinstance(nv[k], dict) for m in ("abs", "rel"))
        res["rule"] = "MPS: new-vs-old within the old code's own run-to-run spread"
    return res


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        tree, device, steps, stem = sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
        run(tree, device, steps, stem)
    elif cmd == "compare":
        r = compare(*sys.argv[2:])
        print(json.dumps(r, indent=1))
    else:
        raise SystemExit(__doc__)
