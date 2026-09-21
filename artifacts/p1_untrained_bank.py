"""Brief P1 — the untrained-encoder bar, measured across K seeds.

D23 replaces the point-mass existence null with a two-sided comparison::

    z = (AUC_real - mean(C)) / sqrt( sd(C)^2 + se_real^2 )

``C`` is the untrained-encoder AUC for a feature, and it is **not** a constant: Brief N2 measured
per-feature ranges of 0.0026-0.0209 across three seeds. This builds the bank that ``mean(C)`` and
``sd(C)`` are estimated from, for all 37 Scheme 1 features, over the same 40,000/34,829 split the
whole N/O series used.

Three properties that make this infrastructure rather than a one-off:

* **It never touches a trained checkpoint.** ``ctl.untrained_encoder_matrix`` builds from the model
  *constructor record* and a seed (`controls.py:140-168`). The encoder is loaded only to read
  ``.config`` off it. So the bar is a property of (architecture, seed, galaxies) alone, and the
  bank is **reusable across every future encoder of the same architecture** — M's untrained nulls
  came back identical to J's to four decimals.
* **It is keyed on what it depends on.** The record carries a hash of the model config and of the
  split. A bank built against a different architecture or a different split refuses to be topped
  up, rather than silently mixing two populations into one standard deviation.
* **It is resumable.** Seeds are independent and the record is written after each one, so an
  interrupted build loses the seed in flight, not the bank. This matters: it is hours of work on
  a machine that has OOM'd before.

Investigation code: terse, excluded from lint/CI.

    uv run python artifacts/p1_untrained_bank.py --plan     # cost and keys; builds nothing
    uv run python artifacts/p1_untrained_bank.py --seeds 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from j4_spread_controls import OUT, _release, prepare  # noqa: E402

from galaxy_jepa.models.vit import load_frozen_encoder  # noqa: E402
from galaxy_jepa.probing import controls as ctl  # noqa: E402
from galaxy_jepa.probing.extract import feature_embeddings  # noqa: E402
from galaxy_jepa.probing.nulls import K_MIN  # noqa: E402

BANK = OUT / "p1_untrained_bank.json"
MAX_TRAIN = 40_000  # the N/O-series split, deliberately — the effect floor is frozen against it


def _key(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _load(model_key: str, split_key: str) -> dict:
    """The bank, or a fresh one. Refuses to top up a bank built on different foundations."""
    if not BANK.exists():
        return {"model_key": model_key, "split_key": split_key, "max_train": MAX_TRAIN, "seeds": {}}
    rec = json.loads(BANK.read_text())
    for name, want, got in (("model", model_key, rec.get("model_key")),
                            ("split", split_key, rec.get("split_key"))):
        if got != want:
            raise SystemExit(
                f"P1: the bank at {BANK} was built against {name} {got}, this run is {want}. "
                f"Mixing them would put two populations into one standard deviation. Move the "
                f"old bank aside rather than topping it up."
            )
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30, help="K — how many untrained seeds in total")
    ap.add_argument("--checkpoint", default="runs/m/encoder.pt")
    ap.add_argument("--plan", action="store_true", help="keys and cost; builds nothing")
    args = ap.parse_args()

    setup = prepare(args.checkpoint, MAX_TRAIN, label="P1", sources=1)
    pc, device = setup.pc, setup.device
    labels, train_ids, test_ids, ds = setup.labels, setup.train_ids, setup.test_ids, setup.ds
    features = list(labels.features)

    # The encoder is loaded ONLY for its constructor record. No weights reach the bar.
    frozen = load_frozen_encoder(setup.ckpt)
    model_key = _key(dict(frozen.config))
    split_key = _key([len(train_ids), len(test_ids), setup.union[:1], setup.union[-1:]])

    rec = _load(model_key, split_key)
    have = sorted(int(s) for s in rec["seeds"])
    todo = [s for s in range(args.seeds) if s not in have]

    print(f"P1 features    : {len(features)} (scheme {pc.scheme_name!r})", file=sys.stderr)
    print(f"P1 split       : {len(train_ids):,} train + {len(test_ids):,} test", file=sys.stderr)
    print(f"P1 keys        : model {model_key}  split {split_key}", file=sys.stderr)
    print(f"P1 bank        : {len(have)} seed(s) present, {len(todo)} to build -> K={args.seeds}",
          file=sys.stderr)
    print(f"P1 K_MIN       : {K_MIN} — the existence gate refuses a bank below this",
          file=sys.stderr)
    if args.seeds < K_MIN:
        print(f"P1 WARNING     : K={args.seeds} is below K_MIN={K_MIN}; the ladder will refuse it",
              file=sys.stderr)
    if args.plan:
        print("\nP1 --plan: nothing built", file=sys.stderr)
        return

    for n, seed in enumerate(todo, start=1):
        t0 = time.perf_counter()
        _release(device)
        mat = ctl.untrained_encoder_matrix(frozen.config, ds, device=device, seed=seed)
        t_ex = time.perf_counter() - t0
        aucs: dict[str, float] = {}
        for feature in features:
            tr = feature_embeddings(mat, labels, feature, train_ids)
            te = feature_embeddings(mat, labels, feature, test_ids)
            aucs[feature] = float(ctl._safe_auc(tr, te, c=pc.c))
        del mat
        _release(device)

        rec["seeds"][str(seed)] = aucs
        BANK.write_text(json.dumps(rec, indent=1))  # after EACH seed — an interruption costs one
        dt = time.perf_counter() - t0
        left = (len(todo) - n) * dt / 3600
        print(f"  seed {seed:<3d} {dt:6.0f}s (extract {t_ex:.0f}s, {len(features)} fits "
              f"{dt - t_ex:.0f}s)  {n}/{len(todo)} done, ~{left:.1f} h left", file=sys.stderr)

    k = len(rec["seeds"])
    print(f"\nP1 bank: K={k} seeds x {len(features)} features -> {BANK}", file=sys.stderr)
    if k >= 2:
        import numpy as np

        spread = {
            f: float(np.std([rec["seeds"][s][f] for s in rec["seeds"]], ddof=1)) for f in features
        }
        worst = sorted(spread, key=lambda f: -spread[f])[:5]
        print("P1 widest per-feature sd of the bar (the z-denominator's own term):", file=sys.stderr)
        for f in worst:
            vals = [rec["seeds"][s][f] for s in rec["seeds"]]
            print(f"    {f:<48s} sd {spread[f]:.4f}  mean {np.mean(vals):.4f} "
                  f"range {max(vals) - min(vals):.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
