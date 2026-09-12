"""Brief E5.4 — per-stamp moments over the **whole** pretraining corpus, in one parallel pass.

Fitting the normalisation on a subsample forces a question that has no good answer: how big must
the subsample be? The measured curve (``e5_sample_size.py``) says nothing reachable clears the 1%
tolerance. Fitting on the whole corpus dissolves the question rather than answering it — but then
the stability check has to be run at *that* scale too, or freezing loses its evidence.

Both come out of one read. Each stamp contributes only its per-channel sums, so the corpus
statistic, the disjoint-half check at 413k a side, and the whole n-vs-stability curve are exact
arithmetic afterwards on a 93 MB array. The pass is embarrassingly parallel; the reduction is
associative, so a worker pool changes nothing about the numbers.

Investigation code: terse, excluded from lint/CI.

    uv run python artifacts/e5_corpus_moments.py [n_workers]
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time

import numpy as np

from galaxy_jepa.data.sources import DirectorySource, load_fits_stamp
from galaxy_jepa.data.transforms import AsinhStretch
from galaxy_jepa.data.validity import validity_mask

CORPUS, Q = "pretrain", 4.0
OUT = ("/private/tmp/claude-501/-Users-malachy-Documents-Galaxy-JEPA/"
       "519f0c5e-3b4c-4159-bf72-4f179da196ae/scratchpad/corpus_moments.npz")
CHUNK = 500


def _one_chunk(object_ids: list[int]) -> tuple[np.ndarray, ...]:
    stretch = AsinhStretch(q=Q)
    vs, vq, vc, ns, nq, npx = [], [], [], [], [], []
    for oid in object_ids:
        raw = load_fits_stamp(f"data/{CORPUS}/{oid}.fits")
        st = np.asarray(stretch(raw), dtype=np.float64)
        ns.append(st.sum(axis=(1, 2)))
        nq.append((st**2).sum(axis=(1, 2)))
        npx.append(float(st.shape[1] * st.shape[2]))
        keep = validity_mask(raw)
        m = st * keep
        vs.append(m.sum(axis=(1, 2)))
        vq.append((m**2).sum(axis=(1, 2)))
        vc.append(float(keep.sum()))
    return (np.array(vs), np.array(vq), np.array(vc),
            np.array(ns), np.array(nq), np.array(npx))


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    src = DirectorySource(f"data/{CORPUS}")
    oids = [int(r["object_id"]) for r in src.rows]
    chunks = [oids[i : i + CHUNK] for i in range(0, len(oids), CHUNK)]
    print(f"corpus {CORPUS}: {len(oids)} stamps | Q={Q} | {len(chunks)} chunks x {CHUNK} "
          f"| {workers} workers", flush=True)

    parts: list[tuple[np.ndarray, ...]] = []
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for j, part in enumerate(pool.imap(_one_chunk, chunks), start=1):
            parts.append(part)
            if j % 40 == 0 or j == len(chunks):
                done = sum(len(p[2]) for p in parts)
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(oids)}  {rate:.0f}/s  "
                      f"eta {(len(oids)-done)/max(rate,1)/60:.1f} min", flush=True)

    stacked = [np.concatenate([p[i] for p in parts]) for i in range(6)]
    assert len(stacked[2]) == len(oids), (len(stacked[2]), len(oids))
    np.savez(OUT, object_id=np.array(oids, dtype=np.int64), vsum=stacked[0], vsq=stacked[1],
             vcnt=stacked[2], nsum=stacked[3], nsq=stacked[4], npx=stacked[5])
    print(f"\nwrote {OUT} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
