"""Ladder (a)'s six baseline encoders, re-exported once (baselines_comparison.md §D).

§D runs the ladder with `LADDER_ENCODER` set to each baseline "re-exported through `frozen_from`, as the
trajectory probe does": the final training checkpoint (step 101,308, the arm's fixed stop) goes through
`i3_loss_usability.frozen_from`, so `load_frozen_encoder` stays the only way an encoder reaches a probe.
The re-export lands in `runs/ladder_v2/encoders/<tag>.pt`, outside every entry's output directory, and
is pinned by SHA-1 in the baselines manifest. As a check (not a substitute), its weights must equal the
run's own `encoder.pt` exactly.

    .venv/bin/python artifacts/ladder_v2_baselines_export.py   # -> runs/ladder_v2/encoders/{tag}.pt + .json

Opens weights only; no bank, probe or result. Investigation code: terse, excluded from lint/CI.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from i3_loss_usability import frozen_from  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TAGS = ("moco_s0", "mae_s0", "moco_s1", "mae_s1", "moco_s2", "mae_s2")
STOP = 101_308  # baseline_run.STOP (not imported: it sets the v2 environment at import)
DEST = REPO / "runs/ladder_v2/encoders"


def sha1(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    rec = {}
    for tag in TAGS:
        ckpt = REPO / f"runs/{tag}/checkpoints/ckpt_{STOP:09d}.pt"
        out = DEST / f"{tag}.pt"
        if out.exists():
            raise SystemExit(f"export: {out} exists; the re-export is made once and pinned")
        enc, step = frozen_from(ckpt, out)
        if step != STOP:
            raise SystemExit(f"export: {ckpt.name} says step {step}")
        own = torch.load(REPO / f"runs/{tag}/encoder.pt", map_location="cpu", weights_only=False)
        got = torch.load(out, map_location="cpu", weights_only=False)
        same = (own["config"] == got["config"] and own["state_dict"].keys() == got["state_dict"].keys()
                and all(torch.equal(own["state_dict"][k], got["state_dict"][k]) for k in own["state_dict"]))
        if not same:
            raise SystemExit(f"export: {tag}'s re-export differs from runs/{tag}/encoder.pt")
        frozen = not enc.training and not any(p.requires_grad for p in enc.parameters())
        rec[tag] = {"checkpoint": str(ckpt.relative_to(REPO)), "checkpoint_sha1": sha1(ckpt), "step": step,
                    "encoder": str(out.relative_to(REPO)), "encoder_sha1": sha1(out),
                    "equals_run_encoder_pt": same, "loads_frozen": frozen}
        print(f"  {tag}: {rec[tag]['encoder_sha1'][:12]}  equal={same} frozen={frozen}", flush=True)
    (DEST / "export.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
