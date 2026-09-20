"""Brief O3 — the overfit-one-batch gate: can the training path memorise 32 images?

`TODO.md` has carried this as P0 and unwritten since the start, and five real runs have happened
without it. It is **insurance rather than a blocker** — AUC 0.9646, a clean six-arm LR
dose-response, bit-identical resume and a SIGReg penalty that responded correctly to λ are not what
a structurally broken path produces. But it catches what those do not, and it yields a number this
project has never measured: **the achievable prediction-loss floor**.

**A gate, not a script.** Same posture as the freezes: a check that can be skipped silently is not
a check. So this writes a stamped record and `j1_preflight` asserts a passing one exists for the
current recipe before any long job starts — see `assert_gate_passed`.

**The pre-registered pass condition, fixed before running:**

    prediction loss falls below 0.05 within 2,000 steps on a batch of 32

Deliberately conservative. A working path should memorise 32 images to near zero; 0.05 sits below
**0.0721**, the deepest prediction-loss minimum any real run reached (I2/H5), so the gate asks the
path to do on 32 images something no run managed on 810,491 — a real bar that will not cry wolf.
The floor actually reached is the number that matters and is reported whatever the verdict.

**Why this is worth running even though nothing looks broken.** In I-JEPA the loss is between the
online encoder and its own EMA target, so it can fall because the model learned *or* because the
two collapsed toward each other — which is exactly what mean-cosine +0.984 under SIGReg looked
like. Driving one batch to genuine memorisation gives a **reference for what "learned" looks like**
in rank / std / cosine, which no run so far has had. The online-vs-target cosine is reported
alongside, because that is the quantity the collapse worry is actually about.

**This is not a run and cannot be mistaken for one.** `steps` is overridden to 2,000, and `steps`
is determining — so it hashes apart from every recipe and no checkpoint it writes could be resumed
into a real job. It writes no checkpoints at all.

    uv run python artifacts/o3_overfit_gate.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f0_preconditions import REPO, check  # noqa: E402
from f1_loader_bench import OUT, _split_ids  # noqa: E402

from galaxy_jepa.callbacks.collapse import collapse_signals  # noqa: E402
from galaxy_jepa.core.config import code_sha, config_hash  # noqa: E402
from galaxy_jepa.data.dataset import StampDataset  # noqa: E402
from galaxy_jepa.harness import build_objective, seed_init  # noqa: E402
from galaxy_jepa.objectives.jepa import _to_device, train_jepa  # noqa: E402

RECORD = OUT / "o3_overfit_gate.json"

#: PRE-REGISTERED, before any number existed.
BATCH = 32  # the production batch size — the recipe is not altered to make the gate easier
STEPS = 2000
LOSS_TARGET = 0.05
#: The deepest prediction-loss minimum any real run reached (I2/H5). The bar sits below it.
DEEPEST_REAL_MINIMUM = 0.0721

#: Context for the floor, from the record. Reported beside it because the comparison is the point.
REAL_RUNS = (
    ("I2/H5 d17 arm, deepest real minimum", 0.0721),
    ("M at its 4-epoch plateau (step 101,300)", 0.1155),
    ("M at 2 epochs (step 50,654)", 0.1357),
    ("M at 0.5 epochs (step 12,700)", 0.2984),
    ("J under SIGReg, minimum ~step 2,000", 0.3244),
    ("J under SIGReg, endpoint step 50,000", 0.4138),
)


def _recipe_hash(cfg) -> str:
    """The SHIPPED recipe's hash — not the gate's overridden one, which is deliberately apart."""
    return config_hash(cfg.determining_dump())


def assert_gate_passed(cfg, *, label: str) -> None:
    """Refuse to start a long job without a passing overfit gate for this recipe.

    Raises on a missing record, a failing one, or one taken against a different recipe. A code
    drift since the gate ran is reported LOUDLY rather than raised on: the recipe hash is what
    determines the training path's shape, the suite covers code changes, and making a ten-minute
    gate mandatory per commit would mean it gets disabled rather than run. Naming the drift is not
    the same as skipping the check.
    """
    if not RECORD.exists():
        raise SystemExit(
            f"{label}: no overfit-gate record at {RECORD}. The training path has never been shown "
            f"to memorise a batch. Run `uv run python artifacts/o3_overfit_gate.py` first."
        )
    rec = json.loads(RECORD.read_text())
    if not rec.get("passed"):
        raise SystemExit(f"{label}: the overfit gate FAILED ({rec.get('verdict')}) — fix the "
                         f"training path before spending hours on it")
    want = _recipe_hash(cfg)
    if rec.get("recipe_hash") != want:
        raise SystemExit(f"{label}: the overfit gate ran against recipe {rec.get('recipe_hash')}, "
                         f"this run is {want}. Re-run the gate.")
    sha, _dirty = code_sha()
    drift = "" if rec.get("code_sha") == sha else (
        f"  *** CODE HAS MOVED since the gate ran ({str(rec.get('code_sha'))[:8]} -> {sha[:8]}); "
        f"re-run it if the training path was touched ***")
    print(f"{label} overfit gate      : PASS, floor {rec['floor']:.6f} at step {rec['floor_step']} "
          f"(bar {rec['loss_target']}){drift}")


def _masking_integrity(jepa, batch, *, seeds=(0, 1, 2, 3, 4)) -> dict:
    """What was checked, not merely that it passed.

    Two failure modes, both of which would make the pretext task easier than intended and so make
    a low floor meaningless: context and target blocks OVERLAPPING (the model can copy rather than
    predict), and the bbox bias DEGENERATING so targets land on empty sky (trivially predictable).
    The second is measured as the mean galaxy-weight at target tokens against the map's own mean —
    a ratio near or below 1 means targets are no longer preferentially on the galaxy.
    """
    petro = batch["petro_rad_arcsec"].cpu().numpy()
    pscale = batch["pixel_scale"].cpu().numpy()
    maps = jepa.weight_maps(petro, pscale)
    flat = maps.reshape(maps.shape[0], -1)
    overlaps, ctx_n, tgt_n, ratios = [], [], [], []
    for s in seeds:
        c_idx, t_idx = jepa.masker.sample(maps, seed=s)
        c, t = c_idx.cpu().numpy(), t_idx.cpu().numpy()
        for b in range(c.shape[0]):
            overlaps.append(len(np.intersect1d(c[b], t[b])))
        ctx_n.append(int(c.shape[1]))
        tgt_n.append(int(t.shape[1]))
        w_tgt = np.mean([flat[b, t[b]].mean() for b in range(t.shape[0])])
        ratios.append(float(w_tgt / max(flat.mean(), 1e-12)))
    return {
        "seeds_checked": list(seeds),
        "max_context_target_overlap_tokens": int(max(overlaps)),
        "total_overlap_tokens": int(sum(overlaps)),
        "context_tokens_per_seed": ctx_n,
        "target_tokens_per_seed": tgt_n,
        "grid_tokens": int(flat.shape[1]),
        "target_weight_ratio_per_seed": [round(r, 4) for r in ratios],
        "disjoint": max(overlaps) == 0,
        "bbox_bias_live": min(ratios) > 1.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()

    print(f"O3 pre-registered pass condition (fixed before running):\n"
          f"    prediction loss < {LOSS_TARGET} within {args.steps:,} steps on a batch of "
          f"{args.batch}\n"
          f"    ({LOSS_TARGET} sits below {DEEPEST_REAL_MINIMUM}, the deepest minimum any real run "
          f"reached)\n", file=sys.stderr)

    cfg, cache = check(verbose=False)
    recipe = _recipe_hash(cfg)
    sha, dirty = code_sha()
    train_ids, _monitor_ids, _ = _split_ids(cfg)
    ids = sorted(train_ids)[: args.batch]  # deterministic, and the same batch every step

    # `steps` is determining, so this hashes APART from every real recipe — the gate can never be
    # mistaken for a run, and writes no checkpoints at all.
    obj = cfg.objective.model_copy(update={"steps": args.steps, "monitor_every": 25})
    gate_cfg = cfg.model_copy(update={"objective": obj})
    jcfg = gate_cfg.to_jepa_config()
    device = cfg.runtime.resolved_device()

    ds = StampDataset(cache, {}, ids, scalars=cache.scalars)
    loader = DataLoader(ds, batch_size=args.batch, drop_last=True)
    batch = _to_device(next(iter(loader)), device)
    jepa = build_objective(jcfg, seed_init(cfg.seed, cache.index.height, cfg.model.model_kwargs()))
    jepa.to(device) if hasattr(jepa, "to") else None

    masks = _masking_integrity(jepa, batch)
    print(f"O3 masking: context/target disjoint={masks['disjoint']} "
          f"(max overlap {masks['max_context_target_overlap_tokens']} tokens over "
          f"{len(masks['seeds_checked'])} seeds); context {masks['context_tokens_per_seed'][0]} + "
          f"target {masks['target_tokens_per_seed'][0]} of {masks['grid_tokens']} grid tokens; "
          f"target/grid galaxy-weight ratio {masks['target_weight_ratio_per_seed']}",
          file=sys.stderr)

    t0 = time.perf_counter()
    result = train_jepa(
        jepa, loader, device=device, monitor_batch=batch,
        checkpoint_path=None, autocast_dtype=cfg.autocast_dtype(),
        checkpointer=None, collapse_floor=cfg.collapse_floor,
    )
    secs = time.perf_counter() - t0

    pred = [p for p in result.prediction_losses if p == p]
    floor = min(pred) if pred else float("nan")
    floor_step = int(np.argmin(result.prediction_losses)) if pred else -1
    passed = bool(pred) and floor < LOSS_TARGET and not result.halted

    with torch.no_grad():
        emb = jepa.encoder.encode(batch["image"].float())
        sig = collapse_signals(emb)
        tgt = jepa.target_encoder.encode(batch["image"].float())
        ot_cos = float(torch.nn.functional.cosine_similarity(emb, tgt, dim=1).mean())

    verdict = ("PASS" if passed else
               ("HALTED — the collapse monitor fired" if result.halted else
                f"FAIL — floor {floor:.6f} did not reach {LOSS_TARGET}"))
    record = {
        "verdict": verdict, "passed": passed, "halted": bool(result.halted),
        "floor": float(floor), "floor_step": floor_step,
        "loss_target": LOSS_TARGET, "steps": args.steps, "batch": args.batch,
        "seconds": secs, "recipe_hash": recipe, "code_sha": sha, "code_dirty": dirty,
        "gate_config_hash": config_hash(gate_cfg.determining_dump()),
        "at_floor": {"std": sig.std, "effective_rank": sig.effective_rank,
                     "mean_cosine": sig.mean_cosine, "online_target_cosine": ot_cos},
        "masking": masks,
        "first_last_prediction": [pred[0], pred[-1]] if pred else None,
        "real_runs_for_comparison": {k: v for k, v in REAL_RUNS},
    }
    RECORD.write_text(json.dumps(record, indent=2))

    print(f"\nO3 {verdict}")
    print(f"O3 LOSS FLOOR          : {floor:.6f} at step {floor_step} "
          f"(started {pred[0]:.4f}) — measured for the first time in this project")
    print(f"O3 collapse signature  : effective_rank {sig.effective_rank:.1f}  std {sig.std:.3f}  "
          f"mean_cosine {sig.mean_cosine:+.3f}  online-vs-EMA cosine {ot_cos:+.4f}")
    print("O3 against the record  : the real runs' prediction-loss minima")
    for name, v in REAL_RUNS:
        print(f"   {name:44s} {v:.4f}   ({v / floor:>6.1f}x the overfit floor)")
    print(f"\nrecord written to {RECORD}")
    if not passed:
        raise SystemExit("O3 FAILED — a broken training path outranks every downstream analysis")


if __name__ == "__main__":
    main()
