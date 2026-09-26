"""Brief DD Stop-3 item 4: latent 2336 (M, block 11, 8× SAE) across the whole probe corpus.

Its top-activating tokens on the held-out set include red, i-band-only streaks. Stated threshold: a
stamp is flagged if any token's activation ≥ T = 15.0, the 99.99th percentile of 2336's non-zero
token activations on sae_eval (1.28 M tokens). M reads its own (v1) cache; the flagged stamps are
eyeballed in their probe_v2 re-cuts (g, r, i separately).

  uv run python artifacts/dd_l2336_scan.py scan
  uv run python artifacts/dd_l2336_scan.py eyeball
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import dd_core as D  # noqa: E402
import dd_sae as S  # noqa: E402

LATENT, T = 2336, 15.0
OUT = D.LOCAL / "sae" / "l2336_scan.npz"


@torch.no_grad()
def scan() -> None:
    from f0_preconditions import check
    _, cache = check(verbose=False)
    ids = []
    with open(D.REPO / "data" / "probe" / "metadata.csv") as fh:
        ids = [int(r["object_id"]) for r in csv.DictReader(fh)]
    have = [i for i in ids if i in cache._row_of]
    rows = np.array([cache._row_of[i] for i in have])
    order = np.argsort(rows)
    m, sae = D.m_encoder(), S.load("M", 11, 8).to(D.DEVICE)
    mx = np.zeros(len(have), np.float32)
    n_hi = np.zeros(len(have), np.int16)
    arg = np.zeros(len(have), np.int16)
    for a in range(0, len(have), 128):
        sel = order[a:a + 128]
        x = torch.from_numpy(np.asarray(cache.data[rows[sel]], np.float32)).to(D.DEVICE)
        t = D.block_tokens(m, x, (11,))[11]
        z = sae.encode(t.reshape(-1, t.shape[-1]))[:, LATENT].reshape(len(sel), 256).cpu().numpy()
        mx[sel], arg[sel], n_hi[sel] = z.max(1), z.argmax(1), (z >= T).sum(1)
        if (a // 128) % 200 == 0:
            print(f"  {a + len(sel):,}/{len(have):,}", flush=True)
    np.savez(OUT, ids=np.array(have), max=mx, argmax=arg, n_hi=n_hi)
    print(f"scanned {len(have):,} of {len(ids):,} probe galaxies; flagged {(mx >= T).sum():,}")


def eyeball(n: int = 30) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits

    d = np.load(OUT)
    flag = np.where(d["max"] >= T)[0]
    meta = {}
    with open(D.REPO / "data" / "probe" / "metadata.csv") as fh:
        for r in csv.DictReader(fh):
            meta[int(r["object_id"])] = (int(r["run"]), int(r["camcol"]), int(r["field"]))
    fl_ids = [int(d["ids"][k]) for k in flag]
    frames = Counter(meta[i] for i in fl_ids)
    runs = Counter(meta[i][0] for i in fl_ids)
    all_runs = Counter(meta[int(i)][0] for i in d["ids"])
    rng = np.random.default_rng(20260926)
    v2dir = D.REPO / "data" / "probe_v2"
    pick = [k for k in rng.permutation(flag) if (v2dir / f"{int(d['ids'][k])}.fits").exists()][:n]
    fig, axes = plt.subplots(n // 5 * 1, 20, figsize=(26, 1.45 * (n // 5)))
    axes = axes.reshape(-1, 4)
    for j, k in enumerate(pick):
        oid = int(d["ids"][k])
        img = np.asarray(fits.getdata(v2dir / f"{oid}.fits"), np.float32)
        p = int(d["argmax"][k])
        for c in range(3):
            im = img[c]
            axes[j, c].imshow(im, origin="lower", cmap="gray", vmin=np.percentile(im, 5), vmax=np.percentile(im, 99.5))
            axes[j, c].add_patch(plt.Rectangle((p % 16 * 16 - 0.5, p // 16 * 16 - 0.5), 16, 16, fill=False, ec="r", lw=0.6))
            axes[j, c].set_title(f"{'gri'[c]}  {oid}" if c == 0 else "gri"[c], fontsize=5)
        rgb = np.clip(np.arcsinh((img[[2, 1, 0]] - np.median(img)) / (img.std() + 1e-6)).transpose(1, 2, 0) / 3, 0, 1)
        axes[j, 3].imshow(rgb, origin="lower")
        axes[j, 3].set_title(f"max {d['max'][k]:.1f}", fontsize=5)
    for a in axes.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"Latent {LATENT} ≥ {T} — 30 flagged probe stamps, probe_v2 re-cut: g | r | i | RGB; red box = the "
                 "strongest token. Is it a single-band streak?", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = D.OUT / "dd_l2336_eyeball.png"
    fig.savefig(path, dpi=90)
    rec = {"threshold": T, "scanned": int(len(d["ids"])), "flagged": int(len(flag)),
           "flagged_frac": float(len(flag) / len(d["ids"])), "distinct_frames": len(frames),
           "top_frames": [[list(k), v] for k, v in frames.most_common(10)],
           "top_runs": [[k, v, all_runs[k], round(v / all_runs[k], 4)] for k, v in runs.most_common(10)],
           "eyeball_ids": [int(d["ids"][k]) for k in pick], "figure": str(path)}
    (D.LOCAL / "sae" / "l2336_eyeball.json").write_text(json.dumps(rec, indent=1))
    return rec


if __name__ == "__main__":
    scan() if sys.argv[1] == "scan" else print(json.dumps(eyeball(), indent=1))
