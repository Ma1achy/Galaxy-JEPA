"""Brief DD, Step 0: the interpretability samples on M, drawn, hashed and copied local.

Tool validation only (artifacts/interp_tooling.md) — no morphology claims.

Three disjoint ID lists over M's own probe split (`j4_spread_controls.prepare`, the one site):
  occl      2,000 TEST galaxies for occlusion maps and patch parts, stratified by answer and
            visibility: per answer, up to 27 positives and 27 negatives among the galaxies that
            reached its question (the conditional population), drawn without replacement across
            answers; topped up at random to 2,000.
  sae       20,000 of the probes' own TRAIN galaxies — SAE tokens (never scored by a probe).
  sae_eval  5,000 further TEST galaxies — SAE faithfulness (reconstruct, re-pool, re-probe).

Each list is written as .npy + a SHA-256 over its sorted IDs. Their normalised fp16 stamps are
read once from M's cache in row order into a compact local copy (runs/dd, on the X10 Pro), so later steps
read 22k contiguous stamps instead of seeking through the 230k-row cache.

  uv run python artifacts/dd_step0.py            # sample, hash, copy, figure
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import r_nonlinear as R  # noqa: E402
from f0_preconditions import check  # noqa: E402
from j4_spread_controls import prepare  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO / "runs" / "dd"  # runs/ → the X10 Pro (gitignored)
FIG = REPO / "artifacts" / "out" / "dd"
N_OCCL, PER_CLASS, N_SAE, N_EVAL = 2_000, 27, 20_000, 5_000
SEED = 20260925


class Sample:
    """The three ID lists and their provenance; stamps copied in cache-row order."""

    def __init__(self, setup) -> None:
        self.setup = setup
        self.rng = np.random.default_rng(SEED)
        self.groups: dict[int, str] = {}
        self.lists: dict[str, list[int]] = {}

    def draw(self) -> dict[str, list[int]]:
        self.lists["occl"] = self.__occlusion()
        rest = sorted(set(self.setup.test_ids) - set(self.lists["occl"]))
        self.lists["sae_eval"] = sorted(int(i) for i in self.rng.choice(rest, N_EVAL, replace=False))
        self.lists["sae"] = sorted(int(i) for i in self.rng.choice(self.setup.train_ids, N_SAE, replace=False))
        return self.lists

    def __occlusion(self) -> list[int]:
        cond = self.setup.labels.with_population("conditional")
        test = sorted(self.setup.test_ids)
        chosen: list[int] = []
        taken: set[int] = set()
        for f in cond.features:
            el = [o for o in cond.eligible(f, test) if o not in taken]
            y = cond.binary_label(f, el)
            for cls in (1, 0):
                pool = [o for o, v in zip(el, y, strict=True) if v == cls]
                k = min(PER_CLASS, len(pool))
                for o in (self.rng.choice(pool, k, replace=False) if k else []):
                    chosen.append(int(o))
                    taken.add(int(o))
                    self.groups[int(o)] = f"{f}:{'pos' if cls else 'neg'}"
        fill = [o for o in test if o not in taken]
        for o in self.rng.choice(fill, N_OCCL - len(chosen), replace=False):
            chosen.append(int(o))
            self.groups[int(o)] = "random"
        return sorted(chosen)

    @staticmethod
    def digest(ids: list[int]) -> str:
        return hashlib.sha256("\n".join(str(i) for i in sorted(ids)).encode()).hexdigest()

    def write(self) -> dict:
        LOCAL.mkdir(parents=True, exist_ok=True)
        rec = {}
        for name, ids in self.lists.items():
            np.save(LOCAL / f"{name}_ids.npy", np.asarray(ids, np.int64))
            rec[name] = {"n": len(ids), "sha256": self.digest(ids)}
        (LOCAL / "occl_groups.json").write_text(json.dumps({str(k): v for k, v in self.groups.items()}))
        return rec

    def copy_stamps(self, cache) -> dict:
        """Row-ordered reads out of the memmap; one local fp16 file per list, in list order."""
        out = {}
        for name, ids in self.lists.items():
            rows = np.array([cache._row_of[i] for i in ids])
            order = np.argsort(rows)
            dst = np.lib.format.open_memmap(LOCAL / f"{name}_stamps.npy", mode="w+",
                                            dtype=np.float16, shape=(len(ids), *cache.index.shape))
            for a in range(0, len(ids), 512):  # sorted rows → near-sequential reads on the X10
                sel = order[a:a + 512]
                dst[sel] = cache.data[rows[sel]]
            dst.flush()
            out[name] = round((LOCAL / f"{name}_stamps.npy").stat().st_size / 1e9, 2)
            del dst
            print(f"  copied {name}: {len(ids):,} stamps, {out[name]} GB", flush=True)
        return out


def _rgb(stamp: np.ndarray) -> np.ndarray:
    """i, r, g → R, G, B; asinh stretch on the stamp's own sky level and bright core."""
    img = np.asarray(stamp, np.float32)[[2, 1, 0]].transpose(1, 2, 0)
    img = np.arcsinh((img - np.median(img)) / (np.std(img[:24]) + 1e-6) / 3)
    return np.clip(img / np.percentile(img, 99.8), 0, 1)


def _short(f: str) -> str:
    return f.split("_", 1)[1]


def figure(ids: list[int], groups: dict[int, str], features: list[str]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    st = np.load(LOCAL / "occl_stamps.npy", mmap_mode="r")
    want = [f for key in ("spiral_a08", "bar_a06", "edgeon_a04", "smooth_or_features_a01")
            for f in features if key in f]
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.28)
    for j, f in enumerate(want):
        k = next(i for i, o in enumerate(ids) if groups.get(o) == f"{f}:pos")
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(_rgb(st[k]), origin="lower")
        for t in range(0, 257, 16):
            ax.axhline(t - 0.5, color="w", lw=0.3, alpha=0.35)
            ax.axvline(t - 0.5, color="w", lw=0.3, alpha=0.35)
        ax.set_title(f"{_short(f)}: positive\n{ids[k]}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    ax = fig.add_subplot(gs[1, :])
    pos = [sum(1 for v in groups.values() if v == f"{f}:pos") for f in features]
    neg = [sum(1 for v in groups.values() if v == f"{f}:neg") for f in features]
    x = np.arange(len(features))
    ax.bar(x - 0.2, pos, 0.4, label="positive", color="#c0504d")
    ax.bar(x + 0.2, neg, 0.4, label="negative", color="#4f81bd")
    ax.axhline(PER_CLASS, color="k", lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([_short(f) for f in features], rotation=90, fontsize=6.5)
    ax.set_ylabel("galaxies in occlusion sample")
    ax.legend(fontsize=8, loc="lower left")
    n_rand = sum(1 for v in groups.values() if v == "random")
    ax.set_title(f"occlusion sample: {len(ids):,} held-out test galaxies — up to {PER_CLASS} per class per answer, "
                 f"among galaxies that reached the question; {n_rand} random top-up", fontsize=9)
    fig.suptitle("Brief DD Step 0 — M is ViT-S/16 at 256²: 16×16 = 256 patch tokens, width 384, 12 blocks. "
                 "Grid = the 16-px patches that occlusion drops.", fontsize=10, y=0.995)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.24)
    path = FIG / "dd_step0_sample.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    setup = prepare(None, R.MAX_TRAIN, label="DD0", sources=1)
    _, cache = check(verbose=False)
    s = Sample(setup)
    s.draw()
    rec = {"lists": s.write()}
    rec["local_gb"] = s.copy_stamps(cache)
    rec["pos_neg_short"] = {f: [sum(1 for v in s.groups.values() if v == f"{f}:{c}") for c in ("pos", "neg")]
                            for f in setup.labels.features}
    rec["figure"] = str(figure(s.lists["occl"], s.groups, setup.labels.features))
    (LOCAL / "step0.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "pos_neg_short"}, indent=1))
    short = {f: v for f, v in rec["pos_neg_short"].items() if min(v) < PER_CLASS}
    print("answers short of 27/27:", short)


def redraw() -> None:
    """The figure alone, from the saved lists (no resampling, no copy)."""
    ids = [int(i) for i in np.load(LOCAL / "occl_ids.npy")]
    groups = {int(k): v for k, v in json.loads((LOCAL / "occl_groups.json").read_text()).items()}
    feats = list(dict.fromkeys(v.rsplit(":", 1)[0] for v in groups.values() if v != "random"))
    print(figure(ids, groups, feats))


if __name__ == "__main__":
    redraw() if sys.argv[1:] == ["--figure"] else main()
