# A1: all 37 answer axes through the spiral pipeline

Branch `almagest-geometry`. Descriptive only; nothing here enters the science record.

**Files**
- Script: `artifacts/almagest_axes.py`, stages `fit`, `assemble`, `sheet` and `relabel` (21 Nov).
- Per-answer records: `runs/almagest/a1/axes/<answer>.{json,npz}`. Shuffled-label probes: `axes/_shuffled/`.
- Summary `a1_axes.json`; table `a1_axes_table.md`.
- Figures:
  - `a1_axes_score_corr.png` (score correlation beside Σw cosine);
  - `a1_axes_sigma_w_cos.png` (the first pass);
  - `a1_axes_contact_sheet.png` (the record) and `.jpg` (a viewing copy).
- Plate: `runs/almagest/a1/plate_axes.bin`, documented in `artifacts/plate_data_format.md` ("Version 3, all answer axes").

**History.** The first pass (`efcd0ef`) labelled proxies and planes by Σw cosine. That rule was symmetric, and
Σw cosines are inflated by the shared covariance (see "Why the Σw rules were replaced"). The `relabel` stage
replaces both rules without refitting any axis; the Σw labels are kept beside the new ones.

## Summary

- **Footing.** A1 (`runs/m_v2/encoder.pt`, SHA-1 `34bb30a7…`), union of 74,825 galaxies (39,997 train,
  34,828 test). The cleaner is the spiral run's own, union-fitted and reused, not refitted. Reuse check: max
  |Δ| 3.6e−6. The three untrained seeds are cleaned the same way.
- **No answer is THIN.** Every answer has ≥ 23k train galaxies, so every γ uses 20 equal-count bins.
- **Base labels** (unchanged): 33 STRONG, 3 NOISE (boxy bulge, 3 arms, more than 4 arms), 1 WEAK
  (star/artifact).
- **Labels, transfer rule: 21 STRONG, 13 PROXY, 3 NOISE, 0 WEAK.** Under the superseded Σw rule they were
  12 / 21 / 3 / 1.
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
- **PROXY of X** (21 Nov rule):
  - *Transfer.* For each pair in different questions, d_X is X's cleaned score on Y's eligible test
    galaxies, sign-aligned. Ratio = (AUC_{X→Y} − 0.5) / (AUC_Y − 0.5).
  - *Proxy.* Y is PROXY of X if the ratio is ≥ 0.9, X is base-STRONG and Y is not NOISE. Among several
    qualifying X, the highest ratio names X.
  - *Both directions qualify.* The answer whose question gates the other's on the GZ2 decision path stays
    STRONG. The tree is the label scheme's own `conditions`, not hand-written. If neither gates the other,
    the higher AUC stays STRONG.

**Mutual pairs** (both ratios between 0.93 and 0.98):
- smooth and features each against spiral and no-spiral: t01 gates t04, so smooth and features stay STRONG;
- edge-on yes/no against cigar: neither gates the other (t02 sits under features, t07 under smooth), so cigar
  stays STRONG on the higher AUC (0.828 against 0.796 and 0.821).

**The 13 PROXY** (transfer ratio in brackets):

| answer | proxy of |
|---|---|
| spiral | smooth (0.95) |
| no-spiral | features (0.95) |
| edge-on yes | cigar (0.97) |
| edge-on no | cigar (0.98) |
| dominant bulge | spiral (1.05) |
| lens/arc | no-spiral (0.98) |
| dust lane | edge-on no (0.98) |
| loose winding | just-noticeable bulge (0.91) |
| 1 arm | just-noticeable bulge (1.36) |
| 2 arms | bar (0.94) |
| 4 arms | features (1.19) |
| can't-tell arms | features (0.95) |
| star/artifact | disturbed (2.48) |

**Three things to read with the labels:**

1. **Ratios above 1.** For star/artifact, 1 arm, 4 arms, boxy (NOISE, 1.60) and 3 arms (NOISE, 2.98), another
   answer's score separates Y's labels *better* than Y's own probe does. Y's own AUC is close to 0.5, so the
   denominator is small and the ratio is unbounded. Star/artifact: disturbed's score reaches 0.762 on its
   labels, its own probe 0.606.
2. **Chains.** X is judged on its base label, so three proxies point at an answer that is itself PROXY:
   - dust lane → edge-on no → cigar;
   - dominant → spiral → smooth;
   - lens/arc → no-spiral → features.

   The records keep every qualifying X (`proxy.all_qualifying`), so a chain can be followed to its STRONG end.
3. **The residual barely moves.** Y is refitted after projecting X's unit filter w out of the cleaned
   embeddings. Its AUC is almost unchanged: spiral 0.746 → 0.744, edge-on yes 0.796 → 0.777. Removing one
   direction does not remove X's information, because X is readable from many correlated directions and Y's
   refit finds them. **This residual does not test "Y beyond X".** A test that does would condition on d_X
   itself: Y's AUC within d_X bins, or the gain of adding Y's score to d_X. Residual split-half cosines
   sit above their shuffle maxima for 12 of 13; 2 arms falls just below (0.082 against 0.086).

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

**Planes.** A pair is a candidate if both answers are STRONG and |r| < null + 0.1. The null is the maximum
|r| over 5 *paired* shuffled-label probes: shuffle k of X against shuffle k of Y. Results:
- Null maxima run 0.013–0.32 (median 0.088). Random-label scores correlate because they share the data's
  variance.
- **82 candidates** (30 under the Σw rule). The margin of 0.1 admits pairs up to |r| ≈ 0.23.
- The cleanest pairs, |r| < 0.01:
  - cigar × merger;
  - merger × bulge-shape no-bulge;
  - no bar × odd;
  - no bulge × odd.
- First-order pairs also qualify: features × medium winding (+0.011) and smooth × cigar (−0.075).
- Full list in the table file.

## A1 bends more than every untrained seed (1 − chord/arc)

| answer | label | A1 (untrained) | max off chord A1 (untrained) | uncleaned | tail flag |
|---|---|---|---|---|---|
| dominant bulge | PROXY of spiral | 0.507 (0.417–0.496) | 0.368 (0.300–0.352) | 0.295 | — |
| ring | STRONG | 0.174 (0.034–0.051) | 0.158 (0.074–0.093) | 0.434 | — |
| disturbed | STRONG | 0.437 (0.019–0.028) | 0.351 (0.075–0.091) | 0.438 | **FLAG** |
| irregular | STRONG | 0.265 (0.056–0.095) | 0.292 (0.077–0.094) | 0.299 | — |
| other (odd feature) | STRONG | 0.396 (0.124–0.213) | 0.331 (0.109–0.124) | 0.325 | — |
| dust lane | PROXY of edge-on no | 0.363 (0.228–0.254) | 0.238 (0.108–0.141) | 0.553 | — |
| tight winding | STRONG | 0.550 (0.150–0.245) | 0.382 (0.132–0.167) | 0.576 | — |
| medium winding | STRONG | 0.681 (0.532–0.631) | 0.400 (0.230–0.283) | 0.536 | — |
| 3 arms | NOISE | 0.512 (0.272–0.321) | 0.292 (0.153–0.176) | 0.534 | — |

Six of the nine are t08 odd features or t10 winding. On those questions the untrained curves are almost
straight (0.02–0.25), so A1's γ there is curved where the untrained encoder's is not. Only disturbed carries a
tail flag. For dominant bulge, cleaning *adds* the bend (uncleaned 0.295 against 0.507 cleaned). For the
others, cleaning lowers it or leaves it about the same.

## Things to treat carefully

- **Marginal STRONGs.** Both pass by a hair. STRONG means "above the untrained seeds", not "large".
  - Medium winding: split-half w cosine 0.081 against a shuffle maximum of 0.074; cleaned AUC 0.514
    against an untrained maximum of 0.508.
  - 4 arms (now PROXY): 0.086 against 0.055.
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
- **Header.** It adds `axes_all`, now carrying the transfer-rule labels and `proxy_of`, and
  `axes_all_note`.
- **Row 7.** It equals `spiral_d/s/r` byte for byte.
- **Old axes.** The QR `probe_basis` is untouched.
- **Retention in the 80-D basis.** w 0.57–0.92, Σw 0.94–1.00.
- **Spiral-row AUC on the Plate test galaxies.** Old QR 0.649, w[j] 0.797, cleaned d 0.718.

## Contact sheet

`a1_axes_contact_sheet.png` is the record. It is lossless, 11.2 MB, and regenerated with the transfer-rule
labels; `.jpg` is a q90 viewing copy (2.0 MB).

Layout: one strip per answer; 11 s-slices (0 … 1, ±0.05) × the 5 test galaxies nearest γ(s); the label is
printed on each strip.

The first-order axes order visibly by eye: smooth runs from disks to round ellipticals, and dominant bulge
from open spirals to compact cores. Most finer answers show little ordering.

## Decisions and deviations, declared

1. **Stability is judged on w,** not Σw; both are recorded.
2. **The proxy partner X is judged on its base label,** so chains can occur (listed above).
3. **The transfer AUC uses Y's eligible *test* galaxies;** AUC_Y is Y's own cleaned test AUC, recomputed
   from the stored filter and asserted equal to the record.
4. **The residual projects out X's *filter* w,** as the brief specifies. It is weak as a test (above).
5. **The plane null pairs shuffle k of X with shuffle k of Y** (5 draws), rather than all 25 cross pairs.
   The cross pairs would raise the null and admit more planes.
6. **Clustering is on 1 − |r| (and 1 − |cos|),** so complements sit together.
7. **The Plate goes in a new file,** `plate_axes.bin`.
8. **THIN strips would run along d.** None occurred.
