"""Ladder v2, E1: the encoder's full-split bank — reused only when it is provably the one to use.

    LADDER_* env set by the runner; .venv/bin/python artifacts/ladder_v2_c2.py <tag>

The aligned-scoring banks `artifacts/out/c2_<tag>_embeddings.npz` were embedded from the same
checkpoints. One is linked into the encoder's directory only if (a) its recorded checkpoint still
hashes to its SHA-1 and is the encoder ladder_env names (`probe_bank.load_bank`), and (b) its ids are
exactly the union of the split this ladder uses (decision 1's identical set, uncapped, as
`aligned_c2.embed` takes it). Anything else re-embeds through `aligned_c2.embed` unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ladder_env as L  # noqa: E402
from aligned_c2 import _bank, embed  # noqa: E402
from f0_preconditions import REPO  # noqa: E402
from j4_spread_controls import prepare  # noqa: E402
from probe_bank import BankProvenanceError, load_bank  # noqa: E402


def main() -> None:
    tag = sys.argv[1]
    if not (L.active() and L.identical_split()):
        raise SystemExit("ladder_v2_c2: runs only under ladder v2 (LADDER_* and LADDER_SPLIT=identical)")
    dst = _bank(tag)
    if dst.exists() or dst.is_symlink():
        raise SystemExit(f"ladder_v2_c2: {dst} exists; the runner clears an interrupted entry")
    src = REPO / "artifacts" / "out" / f"c2_{tag}_embeddings.npz"
    s = prepare(None, 0, label=f"E1-{tag}", sources=1)
    union = np.asarray(s.union, dtype=np.int64)
    why = "absent"
    if src.exists():
        try:
            with load_bank(src, expected_checkpoint=s.ckpt) as b:
                ids = b["ids"].astype(np.int64)
            why = "reused" if np.array_equal(ids, union) else f"ids differ ({len(ids):,} vs {len(union):,})"
        except BankProvenanceError as e:
            why = f"provenance: {e}"
    if why == "reused":
        dst.symlink_to(src.resolve())
        print(f"E1 {tag}: linked {src.name} (checkpoint SHA-1 and ids match the split)", file=sys.stderr)
        return
    print(f"E1 {tag}: re-embedding — {why}", file=sys.stderr)
    embed(tag, str(s.ckpt))


if __name__ == "__main__":
    main()
