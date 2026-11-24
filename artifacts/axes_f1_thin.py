"""F1 supplement (24 Nov 2026, POST HOC after seeing F1): the thinning control at the low bin's real vote count. F1's control thinned the 17+ bin
to 4 votes, but the 1–4 bin's median is 1 vote for arm count and winding. Thin the same bin to 1 and 2 votes (train and test labels), refit and
test in the bin, and read τ against the thinned labels AND against the full labels. Same code, same seeds. → docs/axes/f1_thin.json"""
import json
import numpy as np
from scipy.stats import kendalltau
import axes_a_ordinal as A
import axes_f as F

AXS, d = A.load()
out = {"questions": {}}
for qi, name in enumerate(A.QUESTIONS):
    r = A.question_rows(AXS, d, name)
    rows, p, n, ct, tr, te = (r[k] for k in ("rows", "p", "n", "ct", "tr", "te"))
    K, cov = p.shape[1], A.QUESTIONS[name]["cov"] is not None
    ecat = p @ np.arange(1, K + 1)
    bins, _ = F.merge_bins(n, tr, te)
    inb = n >= bins[-1][0]
    rec = {"bin": list(bins[-1]), "levels": {}}
    for kind in ("cleaned", "raw"):
        for k in (1, 2):
            pt, nt = F.thin(p[inb], n[inb], k, np.random.default_rng(F.SEED + 20 + k))
            P_, N_ = p.copy(), n.copy()
            P_[inb], N_[inb] = pt, nt
            e4 = P_ @ np.arange(1, K + 1)
            for enc in F.ENC:
                X = (d.clean if kind == "cleaned" else d.mats)[enc][rows]
                sel = te & inb
                a, (Z, m, _, _) = F.fit_test(X, ct, P_, N_, tr & inb, sel, F.SEED + qi, cov)
                s = (Z @ m["ord"]["w"])[sel]
                rec["levels"].setdefault(f"{kind}/{k}", {})[enc] = {"gap": a["gap_free_base"], "gap_ci": a["gap_ci95"], "share": a["share"], "reading": a["reading"],
                                                                     "tau_thin_labels": float(kendalltau(s, e4[sel])[0]), "tau_full_labels": float(kendalltau(s, ecat[sel])[0])}
            A.log(f"F1thin {name} {kind} k={k} " + " ".join(f"{e}:{v['tau_thin_labels']:.3f}/{v['tau_full_labels']:.3f}" for e, v in rec["levels"][f"{kind}/{k}"].items()))
    out["questions"][name] = rec
    (A.DOC / "f1_thin.json").write_text(json.dumps(out, indent=1, default=float))
