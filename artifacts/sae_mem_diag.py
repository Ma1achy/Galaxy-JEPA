"""Why A1's SAE training runs out of MPS memory (2026-10-10). Diagnostic only: nothing is saved.

A1's block-11 8× training failed twice at epoch 6 of 8 with identical numbers (MPS "other
allocations" 21.55 GiB; peak footprint 25.6 GB). O2's identical training peaked at 3.0 GB on
2026-09-27, before macOS 27.0.1 (installed 2026-10-05). Here, M's own SAE and A1's, three epochs each,
in memory (`dd_sae.train(..., tokens=...)` never writes), with the MPS driver's allocation sampled
every 10 s and the distinct dead-latent counts tallied — each is a distinct `topk(k)` shape in AuxK.

    uv run python artifacts/sae_mem_diag.py               # M and A1, 3 epochs -> sae_mem_diag.json
    uv run python artifacts/sae_mem_diag.py A1 8          # A1 alone, all 8 -> sae_mem_diag_A1_8.json
Each sample: (seconds, driver GiB, allocator GiB, dead latents, distinct dead counts so far).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import dd_sae as S  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "out" / "aligned" / "sae_mem_diag.json"
EPOCHS = 3
RUNS = {"M": REPO / "runs/dd/sae_tokens/M_b11_sae.npy", "A1": REPO / "runs/dd_a1/sae_tokens/A1_b11_sae.npy"}
GIB = 2 ** 30


def one(tag: str, path: Path, epochs: int = EPOCHS) -> dict:
    tokens = np.load(path, mmap_mode="r")
    samples, stop = [], threading.Event()
    t0 = time.time()
    dead_counts: set[int] = set()
    last_dead = [0]

    def poll():
        while not stop.is_set():
            samples.append((round(time.time() - t0), torch.mps.driver_allocated_memory() / GIB,
                            torch.mps.current_allocated_memory() / GIB, last_dead[0], len(dead_counts)))
            stop.wait(10)

    real = S.losses

    def counting(sae, x, seen):
        loss, st = real(sae, x, seen)
        last_dead[0] = round(st["dead"] * sae.W_enc.shape[0])
        dead_counts.add(last_dead[0])
        return loss, st

    S.losses = counting
    th = threading.Thread(target=poll, daemon=True)
    th.start()
    try:
        S.train(tag, 11, 8, tokens=tokens, epochs=epochs)
        err = None
    except RuntimeError as e:
        err = str(e)[:300]
    finally:
        stop.set()
        th.join()
        S.losses = real
    torch.mps.empty_cache()
    drv = [s[1] for s in samples]
    per_min = np.polyfit([s[0] for s in samples], drv, 1)[0] * 60 if len(samples) > 2 else None
    return {"tokens": str(path), "epochs": epochs, "seconds": round(time.time() - t0), "error": err,
            "driver_gib_start_peak_end": [drv[0], max(drv), drv[-1]] if drv else None,
            "driver_growth_gib_per_min": per_min, "allocator_gib_peak": max(s[2] for s in samples) if samples else None,
            "distinct_dead_counts (topk shapes)": len(dead_counts), "samples": samples}


if __name__ == "__main__":
    import platform
    res = {"macos": platform.mac_ver()[0], "torch": torch.__version__,
           "pid": os.getpid(), "runs": {}}
    pick = {sys.argv[1]: RUNS[sys.argv[1]]} if len(sys.argv) > 1 else RUNS
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else EPOCHS
    out = OUT if len(sys.argv) == 1 else OUT.with_name(f"sae_mem_diag_{sys.argv[1]}_{epochs}.json")
    for tag, path in pick.items():
        res["runs"][tag] = one(tag, path, epochs)
        print(tag, {k: v for k, v in res["runs"][tag].items() if k != "samples"}, flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1))  # written per run, so an out-of-memory stop keeps the curve
