"""Brief I, question 2 — is the training loss usable to pick a checkpoint?

H5's finding was that it is not: the arm 19x better on latent MSE lost frozen-probe AUC
decisively. LeJEPA claims its loss correlates with downstream accuracy; this measures whether
that holds here, by probing every checkpoint of every arm and correlating loss against AUC
*within* each arm.

**What this can and cannot say, stated before the numbers** (and in
`artifacts/i_decision_rule.md`, committed before any arm ran):

  * n = 6 per arm. |rho| = 1 is p ~ 0.0028 two-sided; anything less is not much.
  * It is not the paper's quantity. Their rho ~ 0.99 is ACROSS runs over a hyperparameter sweep;
    this is WITHIN a run over training time. Different things — though this is the one the 1C
    checkpoint rule actually needs.
  * Their alpha-scaling law needs lambda to vary within the correlated set, so it does not apply.

The probe here is deliberately **smaller** than question 1's: a fixed 8,000/8,000 subset, the
same one at every checkpoint and in every arm, because Q2 needs ordering within an arm rather
than a number comparable with 0.9043 / 0.9358. Q1 keeps the full-size probe for that.

The logistic-vs-CAV disagreement (I5) rides along: it is a mean difference and a dot product on
embeddings that have already been extracted, so it costs nothing and comes out as a trajectory
per arm rather than a single endpoint.

    uv run python artifacts/i3_loss_usability.py sweep --arm d17
    uv run python artifacts/i3_loss_usability.py read
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f0_preconditions import check  # noqa: E402
from f1_loader_bench import OUT  # noqa: E402

from galaxy_jepa.data.dataset import StampDataset  # noqa: E402
from galaxy_jepa.data.orchestrate import assign_three_way  # noqa: E402
from galaxy_jepa.models.vit import load_frozen_encoder  # noqa: E402
from galaxy_jepa.probing.entanglement import cav_direction, logistic_cav_disagreement  # noqa: E402
from galaxy_jepa.probing.logistic import (  # noqa: E402
    _extremes,
    extract_embeddings,
    probe_auc_ci,
    probe_direction,
)

RUNS = Path(__file__).resolve().parent.parent / "runs" / "i2"
ARMS_JSONL = OUT / "i2_arms.jsonl"
SWEEP = OUT / "i3_sweep.jsonl"
N_TRAIN, N_TEST = 8000, 8000


def _stride(ids: list[int], n: int) -> list[int]:
    """A deterministic even subsample — the identical one for every checkpoint and arm."""
    if len(ids) <= n:
        return ids
    step = len(ids) / n
    return [ids[int(i * step)] for i in range(n)]


def frozen_from(ckpt: Path, scratch: Path):
    """A training checkpoint, re-exported so the freeze boundary still runs through disk.

    `TrainCheckpointer` writes `encoder_config` + `encoder`; `save_encoder` writes `config` +
    `state_dict`. Same two objects under different keys. Re-exporting rather than constructing
    the module here keeps `load_frozen_encoder` — and so the `eval()` + `requires_grad_(False)`
    that the probing path relies on — the only way an encoder reaches a probe.
    """
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    scratch.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": payload["encoder_config"],
            "state_dict": payload["encoder"],
            "extra": {"step": payload["step"], "smoke": True, "from": str(ckpt)},
        },
        scratch,
    )
    return load_frozen_encoder(scratch), int(payload["step"])


def sweep(arm: str) -> list[dict]:
    cfg, cache = check(verbose=False)
    pc = cfg.probe
    device = cfg.runtime.resolved_device()

    meta = Path(cfg.paths.probe_dir) / "metadata.csv"
    df = pd.read_csv(meta, usecols=["object_id", pc.label_col])  # two columns, not ~150
    probe_ids = [int(o) for o in df["object_id"]]
    rows = {
        int(o): {pc.label_col: float(f)}
        for o, f in zip(df["object_id"], df[pc.label_col], strict=True)
    }
    del df

    split = assign_three_way(probe_ids, seed=cfg.seed, ratios=cfg.ratios)
    train_ids = _stride(sorted(split.train), N_TRAIN)
    test_ids = _stride(sorted(split.test), N_TEST)
    train_ds = StampDataset(cache, rows, train_ids, label_fraction_col=pc.label_col)
    test_ds = StampDataset(cache, rows, test_ids, label_fraction_col=pc.label_col)

    losses = _arm_record(arm)
    out: list[dict] = []
    for ckpt in sorted((RUNS / arm / "checkpoints").glob("ckpt_*.pt")):
        t0 = time.perf_counter()
        frozen, step = frozen_from(ckpt, RUNS / arm / "frozen" / f"{ckpt.stem}_encoder.pt")
        train_full = extract_embeddings(frozen, train_ds, device=device)
        test_full = extract_embeddings(frozen, test_ds, device=device)
        train_emb = _extremes(train_full, low=pc.extreme_low, high=pc.extreme_high)
        test_emb = _extremes(test_full, low=pc.extreme_low, high=pc.extreme_high)
        auc, lo, hi = probe_auc_ci(train_emb, test_emb, c=pc.c, seed=cfg.seed)

        direction = probe_direction(train_emb, name=pc.label_col, c=pc.c)
        rec = {
            "arm": arm,
            "step": step,
            "auc": auc,
            "auc_lo": lo,
            "auc_hi": hi,
            "n_train": int(len(train_emb.y)),
            "n_test": int(len(test_emb.y)),
            # the loss AT the checkpoint's step, plus the window around it — a single step is
            # noisy, and the question is about the signal a selector would actually read
            "loss": losses["losses"][step - 1],
            "loss_prediction": losses["losses_prediction"][step - 1],
            "loss_sigreg": losses["losses_sigreg"][step - 1],
            "loss_mean50": float(np.mean(losses["losses"][max(0, step - 50) : step])),
            "loss_prediction_mean50": float(
                np.mean(losses["losses_prediction"][max(0, step - 50) : step])
            ),
            # I5: does enforced isotropy collapse the discriminative-vs-marginal cross-check?
            "logistic_cav_disagreement": logistic_cav_disagreement(
                direction.w_unit, cav_direction(train_emb)
            ),
            "embedding_std": float(np.std(train_full.x)),
            "seconds": time.perf_counter() - t0,
            "smoke": True,
        }
        out.append(rec)
        with SWEEP.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(
            f"  {arm:<11} step {step:>4}  auc {auc:.4f} [{lo:.4f}, {hi:.4f}]  "
            f"loss {rec['loss_mean50']:.4f}  sigreg {rec['loss_sigreg']}  "
            f"cav-disagree {rec['logistic_cav_disagreement']:.4f}  "
            f"emb-std {rec['embedding_std']:.3f}  {rec['seconds'] / 60:.1f} min",
            file=sys.stderr,
            flush=True,
        )
    return out


def _arm_record(arm: str) -> dict:
    """The newest record for ``arm`` — the per-step loss traces the sweep reads."""
    best = None
    for line in ARMS_JSONL.read_text().splitlines():
        rec = json.loads(line)
        if rec["arm"] == arm:
            best = rec
    if best is None:
        raise SystemExit(f"no arm record for {arm!r} in {ARMS_JSONL}")
    return best


def _spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation without scipy, which is an optional dependency here."""
    ra, rb = _ranks(a), _ranks(b)
    ra, rb = np.asarray(ra), np.asarray(rb)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / denom) if denom else float("nan")


def _ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(order):  # average ties, as Spearman requires
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return out


def read() -> None:
    by_arm: dict[str, list[dict]] = {}
    for line in SWEEP.read_text().splitlines():
        rec = json.loads(line)
        by_arm.setdefault(rec["arm"], []).append(rec)
    for arm, recs in by_arm.items():
        recs = sorted({r["step"]: r for r in recs}.values(), key=lambda r: r["step"])
        auc = [r["auc"] for r in recs]
        print(f"\n{arm}  (n={len(recs)} checkpoints)")
        print("  step   auc      loss(mean50)  pred(mean50)  sigreg    cav-disagree  emb-std")
        for r in recs:
            sig = "—" if r["loss_sigreg"] is None else f"{r['loss_sigreg']:.4f}"
            print(f"  {r['step']:>4}  {r['auc']:.4f}   {r['loss_mean50']:>10.4f}  "
                  f"{r['loss_prediction_mean50']:>11.4f}  {sig:>8}  "
                  f"{r['logistic_cav_disagreement']:>11.4f}  {r['embedding_std']:>6.3f}")
        for label, key in (
            ("total loss", "loss_mean50"),
            ("prediction term", "loss_prediction_mean50"),
            ("sigreg term", "loss_sigreg"),
        ):
            vals = [r[key] for r in recs]
            if any(v is None for v in vals):
                print(f"  spearman(AUC, {label:<16}) = —   (the penalty was not computed)")
                continue
            rho = _spearman(vals, auc)
            print(f"  spearman(AUC, {label:<16}) = {rho:+.3f}"
                  f"   ({'lower loss = better' if rho < 0 else 'lower loss = WORSE'})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sweep", "read"])
    ap.add_argument("--arm", default="d17")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "read":
        read()
        return
    sweep(args.arm)


if __name__ == "__main__":
    main()
