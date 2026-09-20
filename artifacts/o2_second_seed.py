"""Brief O2 — a second training seed on M's plateau.

M is one seed, one trajectory, four probe points. Its stopping rule fired at 4 epochs, which
*locates* the plateau between 2 and 4 — it does not bound it. O2 asks one narrow question:

    **Is the plateau a property of the recipe, or of one training draw?**

Answering that requires holding the evaluation fixed. `config.seed` in this project drives four
different things — the weight init (`harness.seed_init`), the data order (`ResumableShuffle`), the
masker, *and both splits*: `split_pretrain` for pretrain/monitor and `assign_three_way` for the
probe three-way. Turning the single knob would move the ruler at the same time as the thing being
measured, giving two draws from one process rather than a controlled contrast.

So O2 moves **the training seed only**. The splits stay on `cfg.seed = 0`, so O2's four probe points
land on the **identical 34,829 held-out galaxies** as M's 0.9593 / 0.9631 / 0.9642 / 0.9646 and the
difference between the curves is training-draw variance and nothing else.

**What this costs, stated rather than discovered later.**

* **Two seeds now exist and the record carries both.** `cfg.seed` is hashed into `config_hash`;
  `train_seed` is not, because the config does not carry it. A stamp naming only one of them would
  assert a determinism this run does not have — the defect class already closed three times here.
  `m2_long_run.main` therefore prints and writes `train_seed`, `split_seed` and `config_seed` as
  three distinct keys, and this driver writes a `seeds.json` beside the checkpoints as well.
* **O2's `config_hash` is IDENTICAL to M's**, which is correct — it is the same recipe — but it
  means `TrainCheckpointer`'s hash guard cannot tell the two runs apart. The output directory is
  the only thing separating them, so the driver refuses to run unless it is writing to `runs/o2`.
* **This is not the production path.** `run_harness` moves every seed together; this driver splits
  them deliberately. O2 therefore measures the recipe's variance *under a driver that differs from
  the one a headline run would use*. Small, confined to which seed reaches three call sites, and a
  real caveat on transferring the interval to a production run.

**What the interval it produces is, and is not.** It is **training-draw variance with the splits
held fixed**. It is *not* a full interval on the headline AUC: the honest interval on a published
number should also include which galaxies landed in the test set, and O2 deliberately excludes that
so it can answer its own question. If the plateau reproduces, the claim supported is "the plateau is
a property of the recipe, not of one training draw" — and no more than that.

Budget: `steps` stays **253,270**, so the LR and EMA schedules are M's exactly; the probe points stop
at 4 epochs (101,308) because that is where M stopped. ~19.5 h plus four probes.

    uv run python artifacts/o2_second_seed.py --plan     # budget and schedule, launches nothing
    uv run python artifacts/o2_second_seed.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m2_long_run import Run, main as long_run  # noqa: E402

#: M's four probe points, up to and including where its rule fired. Not beyond: O2 is a replication
#: of M's trajectory, not an extension of it, and extending would answer a different question.
PROBE_EPOCHS = (0.5, 1.0, 2.0, 4.0)

O2 = Run(
    label="O2",
    tag="o2",
    probe_epochs=PROBE_EPOCHS,
    train_seed=1,  # splits stay on cfg.seed = 0 — that is the whole design
    out_dir="runs/o2",
)


def main() -> None:
    long_run(O2)


if __name__ == "__main__":
    main()
