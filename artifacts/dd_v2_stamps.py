"""Brief DD's three samples re-baked under the aligned pipeline (Kickoff E, data step 1d).

The ID lists are Step 0's, unchanged (`runs/dd/{occl,sae,sae_eval}_ids.npy`, checked against the
SHA-256s in `runs/dd/step0.json`): the aligned rerun reads the same galaxies as M's. Only the pixels
change: each stamp is copied from the v2 cache (probe_v2 through the v2 freeze), so the copy is
byte-identical to what A1 and A2 will see. Written beside M's, never over them:
`runs/dd_v2/{name}_ids.npy`, `{name}_stamps.npy`, and `step0_v2.json` with the cache provenance.

    GJ_CONFIG=configs/pretrain_v2.yaml F0_CACHE_BASE=... F0_NORM_PREFIX=... \
        uv run python artifacts/dd_v2_stamps.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dd_step0 import LOCAL as V1, Sample  # noqa: E402
from f0_preconditions import check  # noqa: E402

V2 = V1.parent / "dd_v2"
NAMES = ("occl", "sae", "sae_eval")


def main() -> None:
    if "pretrain_v2" not in os.environ.get("GJ_CONFIG", ""):
        raise SystemExit("dd_v2_stamps: GJ_CONFIG must name the v2 config; this copies from the v2 cache")
    step0 = json.loads((V1 / "step0.json").read_text())["lists"]
    lists = {}
    for name in NAMES:
        ids = [int(i) for i in np.load(V1 / f"{name}_ids.npy")]
        if Sample.digest(ids) != step0[name]["sha256"]:
            raise SystemExit(f"dd_v2_stamps: {name}'s ID list no longer matches Step 0's SHA-256")
        lists[name] = ids
    cfg, cache = check(verbose=False)
    missing = {n: len(set(ids) - set(cache._row_of)) for n, ids in lists.items()}
    if any(missing.values()):
        raise SystemExit(f"dd_v2_stamps: the v2 cache lacks Step 0 galaxies: {missing}")
    V2.mkdir(parents=True, exist_ok=True)
    gb = {}
    for name, ids in lists.items():  # Step 0's copy, row-ordered reads, written to runs/dd_v2
        np.save(V2 / f"{name}_ids.npy", np.asarray(ids, np.int64))
        rows = np.array([cache._row_of[i] for i in ids])
        order = np.argsort(rows)
        dst = np.lib.format.open_memmap(V2 / f"{name}_stamps.npy", mode="w+", dtype=np.float16,
                                        shape=(len(ids), *cache.index.shape))
        for a in range(0, len(ids), 512):
            sel = order[a:a + 512]
            dst[sel] = cache.data[rows[sel]]
        dst.flush()
        del dst
        gb[name] = round((V2 / f"{name}_stamps.npy").stat().st_size / 1e9, 2)
        print(f"  copied {name}: {len(ids):,} stamps, {gb[name]} GB", flush=True)
    rec = {"lists": {n: {"n": len(ids), "sha256": Sample.digest(ids)} for n, ids in lists.items()},
           "local_gb": gb, "cache": str(cache.cache_dir), "pipeline_hash": cache.cache_dir.name,
           "normalisation_hash": cache.index.normalisation_hash,
           "freeze_corpus": cfg.normalisation.corpus if cfg.normalisation else None,
           "stamps_sha1": {n: hashlib.sha1((V2 / f"{n}_stamps.npy").read_bytes()).hexdigest() for n in lists}}
    (V2 / "step0_v2.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
