"""Brief N1 — the controls battery on M's settled encoder.

Same battery as J4, different encoder and one addition. The checks are J4's and are not restated;
what N1 declares is what it runs them against.

* **M's encoder, not J's.** J's controls ran on the 50,000-step SIGReg run — measurably worse than
  step 3,000 of its own trajectory (0.9278 against 0.9470, intervals separated). Everything the
  catalogue rests on was therefore read off a representation nobody wants to publish. M's is
  settled: λ=0 (D21), D17 schedule, stopped by its own pre-registered rule at 4 epochs, consensus
  0.9646. The checkpoint is left for `cfg.paths.out_dir` to resolve rather than named absolutely —
  a run's location is not its identity (D15), and M's `out_dir` is already `runs/m`.
* **The D19 fix, exercised for the first time on a good encoder.** K1 removed 3C-5 from the
  existence bar after J4 measured it bit-identical to the `snr` nuisance probe — one measurement
  entered twice, once as a null and once as a diagnostic. Under the old bar *every* feature failed,
  featured-ness included. That correction has never met a representation worth testing.
* **Three untrained seeds.** J measured one. Because the two resamplable controls never cleared
  0.5552 while the untrained singleton reached 0.7908, `existence_null_samples` collapses to a
  point mass at that singleton: the existence test is exactly `real_auc > untrained_encoder_auc`,
  and one draw of a random ViT is the entire bar. Its variability under reseeding has never been
  measured, and it is the base quantity a margin-form effect floor would consume — so it is
  measured *before* N2 proposes anything on top of it, not after. `cfg.seed` stays primary, so the
  bar, the selectivity and every number comparable to J are unchanged in construction.

The spread is J4's, unedited and unreordered: chosen to span the difficulty range and declared
before any number existed. Re-picking features on a new encoder would be choosing them after seeing
results, which is the failure this whole apparatus is built to prevent.

**No rungs, no verdicts, no p-values, no floor.** `assert_null_resolution` would refuse at this
draw budget anyway (Scheme 1's BY family of 37 needs >= 3,109), and these draws are
characterisation for a floor *proposal*, not an existence test.

Investigation code: terse, excluded from lint/CI.

    uv run python artifacts/n1_controls.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from j4_spread_controls import SPREAD, Spec, main as battery  # noqa: E402

N = Spec(
    label="N1",
    tag="n1",
    floor_label="N2",
    checkpoint=None,  # <out_dir>/encoder.pt — M's, because pretrain.yaml's out_dir is runs/m
    spread=SPREAD,  # J4's, verbatim: declared before the numbers, so it stays declared
    extra_untrained_seeds=(1, 2),  # primary is cfg.seed = 0; three in total
)


def main() -> None:
    battery(N)


if __name__ == "__main__":
    main()
