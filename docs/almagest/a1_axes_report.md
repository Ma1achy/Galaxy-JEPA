# A1 — all 37 answer axes through the spiral pipeline

Branch `almagest-geometry`. Descriptive only; nothing here enters the science record.
Script `artifacts/almagest_axes.py` (stages `fit`, `assemble`, `sheet`); per-answer records in
`runs/almagest/a1/axes/<answer>.{json,npz}`; summary `a1_axes.json`; table `a1_axes_table.md`;
figures `a1_axes_sigma_w_cos.png`, `a1_axes_contact_sheet.jpg`; Plate `runs/almagest/a1/plate_axes.bin`.

## Summary

- **Footing.** A1 (`runs/m_v2/encoder.pt`, SHA-1 `34bb30a7…`), union of 74,825 galaxies (39,997 train, 34,828
  test). The cleaner is the spiral run's own, union-fitted and reused, not refitted. Reuse check: max |Δ| 3.6e−6
  against the saved cleaned matrix. The three untrained seeds are cleaned the same way.
- **No answer is THIN.** Every answer has ≥ 23k train galaxies, so every γ uses 20 equal-count bins.
- **Base labels:** 33 STRONG, 3 NOISE (boxy bulge, 3 arms, more than 4 arms), 1 WEAK (star/artifact).
  After the proxy rule: **12 STRONG, 21 PROXY, 3 NOISE, 1 WEAK.**
- **Linearity.** Only star/artifact shows MLP headroom beyond the CI: +0.166 [0.048, 0.286], from the ladder's A1
  `r_r1` record.
- **Cleaning** lowers cleaned AUC by more than 0.05 on 12 answers and *raises* it on 11, among them
  edge-on, cigar, odd and dust lane. Raw–cleaned filter cosine: median 0.89, minimum 0.70 (features/disk).
- **Bend (1 − chord/arc).** A1's γ is straighter than every untrained seed on 26 answers, within their range on 2,
  and more bent on 9. Cleaning straightens γ on 25 of 37.
- **Poorly-voted tail flag** on 4 answers: edge-on/no, spiral, just-noticeable bulge, disturbed.
- **Pattern vs filter.** cos(chord, Σw) is 0.92–0.996 on every answer; cos(chord, w) is 0.03–0.25. As in the spiral
  run, γ follows the pattern, not the filter.

## Two properties of the label rules that shape the table

**1. The proxy rule is symmetric.** When two STRONG answers in different questions share |cos Σw| ≥ 0.8,
each satisfies "PROXY of X with X STRONG in another question", so each becomes the other's proxy. The
first-order axes therefore all carry PROXY: smooth ↔ no_spiral, features ↔ spiral, edge-on ↔ cigar, and
bar ↔ completely round. None of them is left STRONG. This is the rule as written, applied literally, with X judged
on its *base* label. If the intent was one direction (a parent question, or the higher AUC, as the original), that
rule needs choosing; the records keep `base_label` so it can be re-labelled without refitting.

**2. The Σw cosine has a high floor.** Σw = cov · w, so the covariance's leading directions pull every
pattern towards them, including random ones. Split-half Σw cosines under *shuffled* labels reach 0.14–0.93
(median about 0.6), against −0.03–0.20 for w. Two consequences:
- Judging stability on w rather than Σw was necessary. On Σw almost every answer would have passed, the
  shuffles included.
- The 0.8 proxy threshold sits inside the range random-label patterns reach. The near-±1 blocks in the matrix
  are partly shared covariance, not shared concepts. No cross-answer null was computed. That would take
  shuffled fits for pairs of answers, and it is the natural next check if the proxy calls are to be relied on.

The residual refits argue that most proxies are not *only* their partner. After Σw_X is projected out:
- residual AUCs are 0.56–0.69;
- the residual split-half w cosine sits above the residual fit's own shuffle maximum on 20 of 21 proxies.
- The exception is smooth (0.022 against 0.075): with no-spiral removed, nothing stable is left of it.

## Pattern geometry

`a1_axes_sigma_w_cos.png` holds the 37 × 37 cosine matrix of cleaned Σw, in embedding units, clustered on
1 − |cos| with average linkage. Three blocks carry most of the structure:
- **Disk vs smooth:** smooth, features, spiral, the three bulge-prominence proxies, loose winding, in-between,
  lens/arc, other and cant-tell, all at |cos| 0.8–1.0.
- **Edge-on / elongation:** edge-on, cigar, odd, dust lane and tight winding.
- **Small faint objects:** merger, more-than-4 arms, boxy, 1 arm, 4 arms, star/artifact and irregular.

**Candidate two-concept planes** are STRONG pairs with |cos Σw| < 0.3: 30 pairs, listed in the table file. All of them
involve the finer answers: bulge shape, arm number and winding, and the odd features. The tightest are
bulge-shape-no-bulge × 2 arms (+0.03) and × 1 arm (+0.04). Read these with property 2 in mind: a low cosine
is the informative direction, because it sits *below* the floor.

## Things to treat carefully

- **Marginal STRONGs.** Medium winding: split-half w cosine 0.081 against a shuffle maximum of 0.074, cleaned
  AUC 0.514 against an untrained maximum of 0.508. 4 arms: 0.086 against 0.055. Both pass by a hair. STRONG
  means "above the untrained seeds", not "large".
- **NOISE despite AUC.** More than 4 arms has raw AUC 0.657, but split-half w cosine 0.028 against 0.087: the
  filter that separates it is not reproducible.
- **Star/artifact** has 37 test positives, and the only clear MLP headroom. Its axis is not well described
  by a line.

## Plate (`plate_axes.bin`, a new file)

The spiral v3 `plate.bin` is left as it was. `plate_axes.bin` carries every v2/v3 section byte-identical, which is
asserted on write, and adds:

| section | shape | meaning |
|---|---|---|
| `axes_d` | [37, n] uint8 | cleaned probe decision per answer, over its 0.1–99.9 percentiles (`header.axes_all[j].d_lo/hi`) |
| `axes_s` | [37, n] uint8 | arc-length coordinate s on γ (255 = missing / THIN) |
| `axes_r` | [37, n] uint8 | off-axis \|z − γ(d)\|, scaled by `r_scale` |
| `axes_gamma` | [37, 101, 80] f32 | γ at s = 0, 0.01, …, 1 in the 80-D coordinates |
| `axes_d_knots` | [37, 101] f32 | d at those s |
| `axes_tangents` | [37, 2, 80] f32 | end tangents of γ (for later extrapolation past s = 1) |
| `axes_filter_w` | [37, 80] f32 | unit filter w: reads a position |
| `axes_pattern` | [37, 80] f32 | unit pattern Σw: walks |

The header gains `axes_all`, one entry per answer: id, j, label, proxy_of, thin, n_bins, d_lo/hi, r_scale, ds_dd,
retention. It also gains `axes_all_note`. Row 7 equals `spiral_d` exactly, which is asserted. The old QR
`probe_basis` is untouched. Retention in the 80-D basis is w 0.57–0.92 and Σw 0.94–1.00. Spiral row fidelity
on the Plate test galaxies (AUC): old QR 0.649, w[j] 0.797, cleaned d 0.718, the same as the spiral report.

## Contact sheet

`a1_axes_contact_sheet.jpg` has one strip per answer: 11 s-slices (0 … 1, ±0.05) × the 5 test galaxies nearest each
γ(s), with the label printed on the strip. The first-order axes order visibly by eye: smooth runs from disks
to round ellipticals, and dominant bulge from open spirals to compact cores. Most finer answers show little
ordering by eye. The sheet was saved as PNG (11.2 MB) and converted losslessly from the same pixels to JPEG
q90 (2.0 MB). The script now writes JPEG, and the PNG is kept in `runs/almagest/a1/`.

## Decisions and deviations, declared

1. **Stability is judged on w**, not Σw (property 2 above). Both are recorded.
2. **The proxy partner X is judged on its base label.** Without this, the rule would depend on processing order.
3. **Clustering is on 1 − |cos|.** Complementary answers (cos ≈ −1) sit together.
4. **The Plate goes in a new file**, `plate_axes.bin`, so the committed spiral v3 file is never rewritten.
5. **THIN strips would run along d.** None occurred.
6. **The contact sheet is JPEG** (size), as above.
