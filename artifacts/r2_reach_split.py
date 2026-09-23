"""Brief R2 follow-up — is a curved path the concept's shape, or a mixture of vote-reach groups?

GZ2's conditional questions reach a median of 5-8 volunteers, so half their vote fractions rest on a
handful of votes, and reach tracks the parent answer (featuredness). Recompute each path's cross-
fitted bend on the well-voted half (question reach >= its median) and the rest. DESCRIPTIVE: no null,
no BY, and splitting by reach also shifts the population. Reach is the QUESTION's total (sum over its
answers' counts), never the answer's own count, which would condition on the label.

    uv run python artifacts/r2_reach_split.py      -> artifacts/out/r2_reach_split.json
"""

import json
import sys

import numpy as np

sys.path.insert(0, "artifacts")
import r_nonlinear as R
from galaxy_jepa.models.vit import load_frozen_encoder
from galaxy_jepa.probing import geometry as geo
from galaxy_jepa.probing.extract import feature_ids
from scipy.stats import spearmanr
setup = R.prepare("runs/m/encoder.pt", R.MAX_TRAIN, label="R", sources=1)
ctx = R.Ctx(setup, load_frozen_encoder(setup.ckpt), 0, dry=True)
zr = ctx.z(ctx.real); lab = ctx.labels["full"]; union = list(setup.union)
feats = list(ctx.features)
rows = {}
print(f"{'feature':<34s} {'bend':>6s} {'rho(|off-line|,reach)':>22s} {'bend@reach>=median':>19s} {'kept':>6s}  median reach")
for f in feats:
    ids = feature_ids(zr, lab, f, union)
    x = zr.x[zr.rows_for(ids)]
    v = lab.vote_fraction(f, ids)
    # the QUESTION's reach (sum over its answers), never the answer's own count — that is the label
    cnt = sum(lab._column(ids, c) for c in lab.scheme.by_name[f].reach_count_cols())
    st = geo.path_statistics(x, v, R.R2_EDGES, seed=0)
    if st is None:
        continue
    b = geo._bin_of(v, R.R2_EDGES)
    keep_bins = [k for k in range(10) if (b == k).sum() >= geo.MIN_OCCUPANCY]
    mean_cnt = np.array([np.nanmean(cnt[b == k]) for k in keep_bins])
    c = st.centroids - st.centroids.mean(0)
    line = st.line / np.linalg.norm(st.line)
    off = np.linalg.norm(c - np.outer(c @ line, line), axis=1)
    rho = spearmanr(off, mean_cnt).statistic
    hi = cnt >= np.nanmedian(cnt)
    st2 = geo.path_statistics(x[hi], v[hi], R.R2_EDGES, seed=0)
    b2 = "   -  " if st2 is None else f"{1 - st2.straightness:+.3f}"
    lo = geo.path_statistics(x[~hi], v[~hi], R.R2_EDGES, seed=0)
    rows[f] = {"bend_all": 1 - st.straightness, "rho_offline_reach": float(rho),
               "bend_hi_reach": None if st2 is None else 1 - st2.straightness,
               "bend_lo_reach": None if lo is None else 1 - lo.straightness,
               "median_reach": float(np.nanmedian(cnt)), "kept_hi": float(hi.mean())}
    print(f"{f[:34]:<34s} {1 - st.straightness:+.3f} {rho:>+22.2f} {b2:>19s} {hi.mean():>6.1%}  {np.nanmedian(cnt):.0f}")

(R.OUT / "r2_reach_split.json").write_text(json.dumps(rows, indent=1))
