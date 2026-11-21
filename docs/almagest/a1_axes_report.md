# A1: all 37 answer axes through the spiral pipeline

Branch `almagest-geometry`. Descriptive only; nothing here enters the science record.

**Files**
- Script: `artifacts/almagest_axes.py`, stages `fit`, `assemble`, `sheet`, `relabel` and `relabel2` (21 Nov).
- Per-answer records: `runs/almagest/a1/axes/<answer>.{json,npz}`. Shuffled-label probes: `axes/_shuffled/`.
- Summary `a1_axes.json`; table `a1_axes_table.md`.
- Figures:
  - `a1_axes_score_corr.png` (score correlation beside Σw cosine);
  - `a1_axes_sigma_w_cos.png` (the first pass);
  - `a1_axes_contact_sheet.png` (the record) and `.jpg` (a viewing copy).
- Plate: `runs/almagest/a1/plate_axes.bin`, documented in `artifacts/plate_data_format.md` ("Version 3, all answer axes").

**History.** The first pass (`efcd0ef`) labelled proxies and planes by Σw cosine. That rule was symmetric, and
Σw cosines are inflated by the shared covariance (see "Why the Σw rules were replaced"). The `relabel` stage
replaces both rules without refitting any axis; the Σw labels are kept beside the new ones. `relabel2` (21 Nov,
later) adds a ratio floor, replaces the residual with a conditional test, and follows chains to a root; the
transfer-v1 labels are kept beside it as a second old column.

## Summary

- **Footing.** A1 (`runs/m_v2/encoder.pt`, SHA-1 `34bb30a7…`), union of 74,825 galaxies (39,997 train,
  34,828 test). The cleaner is the spiral run's own, union-fitted and reused, not refitted. Reuse check: max
  |Δ| 3.6e−6. The three untrained seeds are cleaned the same way.
- **No answer is THIN.** Every answer has ≥ 23k train galaxies, so every γ uses 20 equal-count bins.
- **Base labels** (unchanged): 33 STRONG, 3 NOISE (boxy bulge, 3 arms, more than 4 arms), 1 WEAK
  (star/artifact).
- **Labels, current (`relabel2`): 20 STRONG, 10 SEPARATE, 3 PROXY, 1 WEAK, 3 NOISE.** Transfer v1:
  21 STRONG / 13 PROXY / 3 NOISE / 0 WEAK. Superseded Σw rule: 12 STRONG / 21 PROXY / 3 NOISE / 1 WEAK.
- **10 of v1's 13 proxies carry information beyond their X** (SEPARATE). The 3 still PROXY are the three
  lowest-powered candidates. Each is a failure to show separation, not a demonstrated proxy (see Labels).
- **Linearity.** Only star/artifact shows MLP headroom beyond the CI: +0.166 [0.048, 0.286], from the
  ladder's A1 `r_r1` record.
- **Cleaning** lowers AUC by more than 0.05 on 12 answers. It *raises* AUC on 11, among them edge-on, cigar,
  odd and dust lane. Raw–cleaned filter cosine: median 0.89, minimum 0.70.
- **Bend (1 − chord/arc).** A1's γ is straighter than every untrained seed on 26 answers, within range on 2,
  and more bent on 9 (listed below). Cleaning straightens γ on 25 of 37.
- **Poorly-voted tail flag** on 4 answers: edge-on/no, spiral, just-noticeable bulge, disturbed.
- **Pattern vs filter.** cos(chord, Σw) is 0.92–0.996 and cos(chord, w) is 0.03–0.25: γ follows the pattern.

## Labels

- **NOISE:** split-half cos(w) is not above the maximum over 5 shuffled-label fits on the same halves.
- **WEAK / STRONG:** stable, and the cleaned test AUC is within/below vs above the untrained seeds' maximum.
- **Ratio floor.** If AUC_Y − 0.5 < 0.05, Y is WEAK. Its best transfer is reported but not used.
  This floors medium winding: AUC 0.514; best transfer from dominant bulge, ratio 0.74.
- **Transfer.** For each pair in different questions, d_X is X's cleaned score on Y's eligible test galaxies,
  sign-aligned. Ratio = (AUC_{X→Y} − 0.5) / (AUC_Y − 0.5).
  - Y is a *candidate* proxy of X if the ratio is ≥ 0.9, X is STRONG, and Y is neither NOISE nor floored.
  - If several X qualify, the highest ratio names X.
  - If both directions qualify, the answer whose question gates the other's on the GZ2 decision path stays
    STRONG. The tree comes from the label scheme's own `conditions`. If neither gates the other, the higher
    AUC stays STRONG.
- **Conditional test** (replaces the residual):
  - Within 10 equal-count bins of d_X, on Y's eligible test galaxies, compute Y's own cleaned score's AUC on
    Y's labels and average the bins by their counts. Bins holding one class only are dropped.
  - 95% CI from 1,000 galaxy bootstrap resamples.
  - The null is the maximum, over Y's 5 shuffled-label probes, of the same conditional AUC, folded to
    max(a, 1 − a).
  - **SEPARATE** if the CI's lower end is above that maximum; otherwise **PROXY of X**.
- **Chains.** Each PROXY is followed through X until it reaches a STRONG or SEPARATE answer, which is its
  *root*. The Plate stores the root.

**Mutual pairs** (unchanged; both ratios 0.93–0.98):
- smooth and features against spiral and no-spiral: t01 gates t04, so smooth and features stay STRONG.
- edge-on yes/no against cigar: neither gates the other (t02 sits under features, t07 under smooth). Cigar
  stays STRONG on the higher AUC (0.828 against 0.796 and 0.821).

**The 13 candidates.** Old labels sit beside the new ones.
- "Cond. AUC" is Y's conditional AUC within bins of X, with its 95% CI.
- "Null" is the shuffled maximum.
- "Ratio" is the transfer ratio.

| answer | X | ratio | own AUC | cond. AUC [95% CI] | null | **label** | transfer v1 | Σw rule |
|---|---|---|---|---|---|---|---|---|
| edge-on yes | cigar | 0.97 | 0.796 | 0.604 [0.594, 0.616] | 0.518 | **SEPARATE** | PROXY of cigar | PROXY of cigar |
| edge-on no | cigar | 0.98 | 0.821 | 0.590 [0.578, 0.604] | 0.508 | **SEPARATE** | PROXY of cigar | PROXY of cigar |
| spiral | smooth | 0.95 | 0.746 | 0.599 [0.591, 0.606] | 0.511 | **SEPARATE** | PROXY of smooth | PROXY of smooth |
| no-spiral | features | 0.95 | 0.772 | 0.608 [0.600, 0.618] | 0.512 | **SEPARATE** | PROXY of features | PROXY of smooth |
| dominant bulge | spiral | 1.05 | 0.594 | 0.547 [0.533, 0.560] | 0.511 | **SEPARATE** | PROXY of spiral | PROXY of smooth |
| lens/arc | no-spiral | 0.98 | 0.670 | 0.562 [0.547, 0.577] | 0.514 | **SEPARATE** | PROXY of no-spiral | PROXY of in-between |
| dust lane | edge-on no | 0.98 | 0.762 | 0.619 [0.573, 0.659] | 0.549 | **SEPARATE** | PROXY of edge-on no | STRONG |
| loose winding | just-noticeable | 0.91 | 0.618 | 0.567 [0.556, 0.579] | 0.512 | **SEPARATE** | PROXY of just-noticeable | PROXY of just-noticeable |
| 2 arms | bar | 0.94 | 0.568 | 0.539 [0.531, 0.547] | 0.506 | **SEPARATE** | PROXY of bar | STRONG |
| can't-tell arms | features | 0.95 | 0.597 | 0.549 [0.541, 0.557] | 0.505 | **SEPARATE** | PROXY of features | PROXY of in-between |
| 1 arm | just-noticeable | 1.36 | 0.568 | 0.534 [0.518, 0.551] | 0.529 | **PROXY**, root just-noticeable | same | STRONG |
| 4 arms | features | 1.19 | 0.565 | 0.533 [0.479, 0.584] | 0.583 | **PROXY**, root features | same | STRONG |
| star/artifact | disturbed | 2.48 | 0.606 | 0.530 [0.369, 0.728] | 0.661 | **PROXY**, root disturbed | same | WEAK |

**Labels that changed between columns, outside this table:**
- Medium winding: STRONG under transfer v1 and Σw; now WEAK (floored).
- All other answers keep their transfer-v1 label. The Σw column differs for many STRONG answers;
  see `a1_axes_table.md`.

**Four things to read with the labels:**

1. **Every chain has length 1.** All three PROXY answers point straight at a STRONG root. The v1 chains
   (dust lane → edge-on no → cigar; dominant → spiral → smooth; lens/arc → no-spiral → features) are gone,
   because the middle links are now SEPARATE.
2. **The remaining PROXYs mean "not shown separate", not "shown to be X".**
   - Each has a ratio above 1 and a small own AUC: X's score separates Y's labels better than Y's own probe.
   - Their conditional CIs are wide or touch the null: star/artifact (37 positives) spans 0.37–0.73
     against 0.66; 4 arms spans 0.48–0.58 against 0.58; 1 arm's lower end is 0.518 against 0.529.
   - More data, not a different rule, would move them.
3. **SEPARATE is a modest bar.** Y's own score still ranks Y's labels within narrow slices of d_X at
   conditional AUC 0.53–0.62, against nulls of 0.51–0.55.
   - So Y carries something X does not. That is not the same as Y being independent of X: the transfer
     ratios of 0.9–1.05 still say most of Y's ranking is shared with X.
   - Dust lane's CI is wide (0.57–0.66) but its lower end clears its null (0.549).
4. **The old residual test** (projecting X's unit w out and refitting Y) is retired. It barely moved any
   AUC (spiral 0.746 → 0.744) and could not separate "beyond X" from "X read along other directions".
   Its numbers remain in the per-answer records.

## Pattern geometry and planes

`a1_axes_score_corr.png` shows corr(d_X, d_Y) over the test galaxies eligible for both, clustered on 1 − |r|.
Beside it is the Σw cosine in the same order.

The score correlations are sparse: 84 of 666 pairs exceed |r| 0.5. The large blocks are:
- the disk/smooth/spiral block (smooth, features, spiral, no-spiral, just-noticeable and obvious bulge),
  |r| 0.6–1.0 (the bulge answers 0.61–0.67);
- edge-on ↔ cigar, |r| ≈ 1.0, with dust lane and completely round attached;
- complements within a question (odd yes/no, bar/no bar, bulge-shape rounded/no bulge), r ≈ −1;
- weaker ties: bar ↔ ring and bar ↔ 2 arms.

The Σw panel is saturated almost everywhere by comparison, which is the inflation the old rules read.

**Planes.** A pair is a candidate if both answers are STRONG or SEPARATE and |r| < null + 0.1. The null is the maximum
|r| over 5 *paired* shuffled-label probes: shuffle k of X against shuffle k of Y. Results:
- Null maxima run 0.013–0.32 (median 0.088). Random-label scores correlate because they share the data's
  variance.
- **157 candidates** (transfer v1: 82; Σw rule: 30). The margin of 0.1 admits pairs up to |r| ≈ 0.23.
  - From v1, 63 are kept and 19 are dropped: all 19 involve medium winding, now WEAK.
  - 94 are added, all involving a now-SEPARATE answer.
- The cleanest pairs, |r| < 0.01:
  - cigar × merger;
  - merger × bulge-shape no-bulge;
  - no bar × odd;
  - no bulge × odd.
- First-order pairs also qualify, e.g. smooth × cigar (−0.075) and spiral × bulge-shape no-bulge (+0.001).
  Features × medium winding (+0.011) left with the floor.
- Full list in the table file.

## A1 bends more than every untrained seed (1 − chord/arc)

| answer | label | A1 (untrained) | max off chord A1 (untrained) | uncleaned | tail flag |
|---|---|---|---|---|---|
| dominant bulge | SEPARATE | 0.507 (0.417–0.496) | 0.368 (0.300–0.352) | 0.295 | — |
| ring | STRONG | 0.174 (0.034–0.051) | 0.158 (0.074–0.093) | 0.434 | — |
| disturbed | STRONG | 0.437 (0.019–0.028) | 0.351 (0.075–0.091) | 0.438 | **FLAG** |
| irregular | STRONG | 0.265 (0.056–0.095) | 0.292 (0.077–0.094) | 0.299 | — |
| other (odd feature) | STRONG | 0.396 (0.124–0.213) | 0.331 (0.109–0.124) | 0.325 | — |
| dust lane | SEPARATE | 0.363 (0.228–0.254) | 0.238 (0.108–0.141) | 0.553 | — |
| tight winding | STRONG | 0.550 (0.150–0.245) | 0.382 (0.132–0.167) | 0.576 | — |
| medium winding | WEAK (floored) | 0.681 (0.532–0.631) | 0.400 (0.230–0.283) | 0.536 | — |
| 3 arms | NOISE | 0.512 (0.272–0.321) | 0.292 (0.153–0.176) | 0.534 | — |

Six of the nine are t08 odd features or t10 winding. On those questions the untrained curves are almost
straight (0.02–0.25), so A1's γ there is curved where the untrained encoder's is not. Only disturbed carries a
tail flag. For dominant bulge, cleaning *adds* the bend (uncleaned 0.295 against 0.507 cleaned). For the
others, cleaning lowers it or leaves it about the same.

## Things to treat carefully

- **Marginal answers.** STRONG means "above the untrained seeds", not "large".
  - Medium winding: split-half w cosine 0.081 against a shuffle maximum of 0.074; cleaned AUC 0.514
    against an untrained maximum of 0.508. It is now WEAK under the ratio floor (0.514 − 0.5 < 0.05).
  - 4 arms (PROXY): split-half w cosine 0.086 against 0.055.
- **NOISE despite AUC.** More than 4 arms has raw AUC 0.657, but its split-half w cosine is 0.028 against
  0.087.
- **Star/artifact** has 37 test positives and the only MLP headroom.

## Why the Σw rules were replaced (first pass, kept for the record)

- **The proxy rule was symmetric.** Two STRONG answers in different questions with |cos Σw| ≥ 0.8 each
  became the other's proxy, so no first-order axis stayed STRONG.
- **The Σw cosine has a high floor.** Σw = cov · w, so the covariance's leading directions pull every
  pattern towards them. Split-half Σw cosines under shuffled labels reach 0.14–0.93 (median about 0.6),
  against −0.03–0.20 for w. Stability is therefore judged on w.
- **The old threshold sat inside the null.** The 0.8 proxy threshold and the 0.3 plane threshold both sat
  inside the range random-label patterns reach.
- The Σw labels are kept per answer (`label_sigma_w`, `proxy_sigma_w`) and in the table's third column.

## Plate (`plate_axes.bin`, a new file)

- **The spiral v3 `plate.bin` is not rewritten.** `plate_axes.bin` carries every v2/v3 section
  byte-identical, which is asserted, and adds `axes_d/s/r` [37, n] uint8, `axes_gamma` [37, 101, 80],
  `axes_d_knots`, `axes_tangents`, `axes_filter_w` and `axes_pattern`.
- **Header.** It adds `axes_all`, now carrying the `relabel2` labels (including SEPARATE), `proxy_of` and
  `proxy_root`, plus `axes_all_note`.
- **Row 7.** It equals `spiral_d/s/r` byte for byte.
- **Old axes.** The QR `probe_basis` is untouched.
- **Retention in the 80-D basis.** w 0.57–0.92, Σw 0.94–1.00.
- **Spiral-row AUC on the Plate test galaxies.** Old QR 0.649, w[j] 0.797, cleaned d 0.718.

## Contact sheet

`a1_axes_contact_sheet.png` is the record. It is lossless, 11.2 MB, and regenerated with the `relabel2`
labels (SEPARATE in green); `.jpg` is a q90 viewing copy (2.0 MB).

Layout: one strip per answer; 11 s-slices (0 … 1, ±0.05) × the 5 test galaxies nearest γ(s); the label is
printed on each strip.

The first-order axes order visibly by eye: smooth runs from disks to round ellipticals, and dominant bulge
from open spirals to compact cores. Most finer answers show little ordering.

## Decisions and deviations, declared

1. **Stability is judged on w,** not Σw; both are recorded.
2. **The proxy partner X is judged on its base label,** so chains can occur. Under `relabel2` none remain;
   the Plate stores the root.
3. **The transfer AUC uses Y's eligible *test* galaxies;** AUC_Y is Y's own cleaned test AUC, recomputed
   from the stored filter and asserted equal to the record.
4. **The residual test is retired** for the conditional test.
   - Bins drop when one class only; the AUC is count-weighted.
   - The bootstrap resamples galaxies and recomputes the bins in each resample.
   - The null uses the same 5 shuffled probes as NOISE. Each conditional AUC is folded to max(a, 1 − a),
     since a shuffled probe has no meaningful sign. Folding raises the null, so it leans towards PROXY.
5. **The plane null pairs shuffle k of X with shuffle k of Y** (5 draws), rather than all 25 cross pairs.
   The cross pairs would raise the null and admit more planes.
6. **Clustering is on 1 − |r| (and 1 − |cos|),** so complements sit together.
7. **The Plate goes in a new file,** `plate_axes.bin`.
8. **THIN strips would run along d.** None occurred.
9. **The ratio floor (0.05) also removes X candidates.** Medium winding was the only STRONG answer below it,
   and as a floored WEAK answer it cannot be an X.
