# Part G1–G3: expert gold sets (25 Nov 2026)

Branch `almagest-geometry`. Descriptive, not hashed. Frozen A1 embeddings; no Claude API. Code: `artifacts/axes_g.py` (G1, G2),
`artifacts/axes_g3.py` (G3). Numbers: `g1.json`, `g2.json`, `g2pair.json`, `g3.json`. **Stopped before G4.**

## Sources

| catalogue | where | rows | licence | SHA-1 |
|---|---|---|---|---|
| EFIGI (Baillard et al. 2011) | VizieR `J/A+A/532/A74` (tables `efigi`, `sdss`), Harvard mirror | 4,458 | none stated beyond VizieR terms; cite the paper | in `g1.json` |
| Nair & Abraham 2010 | VizieR `J/ApJS/186/427/table2`, Harvard mirror; codes from the NAOJ mirror's ReadMe | 14,034 | same | in `g1.json` |
| Galaxy Zoo DESI v1.0.1 ("friendly") | Zenodo 8360385, MD5 `114785d0…` verified | 8,689,370 | CC-BY-4.0 | Zenodo MD5 |

## G1: the matches (nearest of ours within 3″, one per galaxy)

| | probe corpus | partition (train / val / test) | in the union | pretraining corpus | chance (+60″) |
|---|---|---|---|---|---|
| EFIGI | 2,373 | 1,662 / 338 / 373 | 790 | 219 | 1 |
| Nair & Abraham | 11,601 | 8,099 / 1,793 / 1,709 | 3,692 | 3,208 | 1 |

- EFIGI galaxies by resolution decile: 22, 2, 6, 6, 8, 17, 28, 55, 147, **2,082**. N&A: 89, 155, 219, 346, 451, 677, 914, 1,406, 2,534, **4,810**.
  As expected, nearby bright galaxies, mostly in the top decile.
- The DR7 objID that EFIGI publishes agrees with ours for 2,335 of 2,373 matches (98.4%): a cross-check only, not a filter.

## G2: EFIGI attributes as probe targets

Part A's cumulative logit and free multinomial, raw and cleaned, A1 and three untrained seeds. Train = the probe corpus' train partition,
not only the union (1,245 EFIGI galaxies outside it were embedded through A1's banked file and the untrained seeds; the code reproduces the bank to
4e-6 and the seeds exactly). EFIGI's value as a one-hot over its ordered categories, weight 1; sparse end categories merged. Test is **almost entirely decile 10**
(326 of 373), so the decile breakdown says only that: decile-10 τ is within 0.04 of the overall τ for every priority attribute, and no other decile reaches 30 galaxies.

Priority attributes (τ with EFIGI's value; `Δ` = A1 minus the best untrained seed, paired bootstrap; *added after G2's table was read, statistic fixed before computing it*):

| attribute | n train/test | τ cleaned | Δ cleaned [95%] | τ raw | Δ raw [95%] | gap (cleaned) | share | reading cleaned / raw |
|---|---|---|---|---|---|---|---|---|
| Arm Curvature | 863/191 | 0.38 | +0.09 [+0.02, +0.15] | 0.47 | +0.05 [−0.01, +0.10] | 0.08 | 1.49 | ONE AXIS / ONE AXIS |
| Arm Strength | 1441/325 | 0.47 | +0.06 [−0.00, +0.10] | 0.54 | +0.03 [−0.02, +0.07] | 0.22 | 0.95 | MOSTLY ONE AXIS / MOSTLY ONE AXIS |
| B/T | 1660/372 | 0.50 | **+0.19 [+0.13, +0.24]** | 0.70 | **+0.13 [+0.09, +0.16]** | 0.33 | 0.82 | MOSTLY ONE AXIS / ONE AXIS |
| Flocculence | 1662/373 | 0.44 | +0.06 [+0.00, +0.09] | 0.60 | +0.05 [+0.01, +0.08] | 0.20 | 0.96 | MOSTLY ONE AXIS / ONE AXIS |
| Inclination/Elongation | 1662/373 | 0.63 | **+0.16 [+0.11, +0.19]** | 0.63 | **+0.17 [+0.13, +0.21]** | 0.43 | 1.10 | ONE AXIS / ONE AXIS |
| Perturbation | 1662/373 | 0.34 | +0.01 [−0.03, +0.05] | 0.37 | +0.03 [−0.01, +0.06] | 0.09 | 1.17 | ONE AXIS / ONE AXIS |
| Visible Dust | 1662/373 | 0.43 | +0.04 [−0.01, +0.08] | 0.48 | +0.04 [−0.00, +0.08] | 0.24 | 0.74 | SEVERAL / SEVERAL |

- **A1 clearly beats untrained on B/T and Inclination, in both matrices.** Arm Curvature clears it in cleaned (n = 191), not in raw, Arm Strength touches zero, Flocculence barely clears it in both, Perturbation and Visible Dust do not.
  Untrained encoders carry a lot of this: their cleaned τ is 0.24–0.47 across the priority attributes, so most of a τ of 0.4 is pixel statistics and size.
- The readings (ONE AXIS etc.) describe the shape of what is learned and can be reached by an untrained encoder too; a reading says nothing about A1 unless its Δ is positive. Shares above 1 mean the ordinal model beats the free multinomial out of sample,
  which is the free model overfitting at 900–1,700 galaxies.
- Other attributes (not priority; `g2.json`): Hot Spots, Inner Ring, Outer Ring and Contamination sit on or near the untrained range; Bar Length, Pseudo-Ring and Multiplicity are no better than untrained. **Arm Rotation** has τ ≈ 0 for every encoder, with a
  large free-model gap: the free model separates "no rotation" (0.5) from the two handed classes, not left from right.
- EFIGI's own confidence intervals are wide (median width 0.25 for most attributes, 0.5 for Arm Curvature and Visible Dust), so τ is capped below 1 for everyone; the cap is not estimated here.

Transfer, cleaned A1, GZ2-trained ordinal axis against the EFIGI attribute:

| EFIGI | GZ2 axis | τ(GZ2 axis, attribute) | r(scores) on EFIGI test | r on union test eligible |
|---|---|---|---|---|
| Arm Curvature | winding (tight → loose) | −0.24 [−0.35, −0.12] | −0.60 | −0.28 (n 20,244) |
| Arm Strength | winding | −0.25 | −0.51 | −0.11 |
| B/T | bulge prominence | +0.23 | +0.39 | +0.12 (n 33,080) |
| Inclination | roundness | +0.43 | +0.75 | +0.37 (n 34,055) |

Signs are as expected (Curvature is 1 = tight; GZ2 winding is tight → loose). The GZ2 bulge axis recovers much less of EFIGI's B/T (τ 0.23) than EFIGI's own fit (0.50); the two axes share only about 15% of variance on the EFIGI test galaxies.
Reported as they come; not reconciled.

## G3: scoring the label sources against the expert grades

Kendall τ-b against the expert grade on galaxies **every named source covers** (GZ2 votes ≥ 5; GZD-5/8 ≥ 5 pooled; GZ DESI predictions, defined where its model says the question was reached). 1,000-resample paired bootstrap for the differences. The full tables (own coverage, ≥ 1 vote arm,
GZD-1/2, N&A, AUCs) are in `g3.json`.

| expert grade ~ label | n (GZ2∩DESI) | GZ2 | GZ DESI | DESI − GZ2 [95%] |
|---|---|---|---|---|
| EFIGI Arm Curvature ~ winding | 1,002 | **0.49** | 0.44 | −0.05 [−0.08, −0.02] |
| EFIGI Arm Strength ~ spiral | 1,314 | 0.55 | 0.56 | +0.01 [−0.02, +0.03] |
| EFIGI B/T ~ bulge | 1,318 | **0.71** | 0.68 | −0.03 [−0.05, −0.02] |
| EFIGI Bar Length ~ bar | 1,311 | 0.67 | 0.67 | +0.00 [−0.02, +0.02] |
| EFIGI Inclination ~ roundness | 299 | 0.66 | 0.67 | +0.01 [−0.01, +0.04] |
| EFIGI Inclination ~ edge-on | 1,787 | 0.65 | **0.73** | +0.08 [+0.06, +0.10] |
| EFIGI Perturbation ~ disturbed | 2,175 | **0.41** | 0.38 | −0.03 [−0.06, −0.01] |
| N&A T-type ~ featured | 11,091 | 0.61 | **0.62** | +0.01 [+0.01, +0.01] |
| N&A T-type ~ bulge | 5,639 | −0.54 | −0.53 | +0.01 [+0.00, +0.03] |
| N&A bar (ordinal) ~ bar | 5,780 | 0.58 | 0.57 | −0.00 [−0.01, +0.00] |
| N&A interaction ~ disturbed | 11,342 | 0.25 | **0.29** | +0.04 [+0.03, +0.05] |

- **GZ2 votes are the better tracker of winding and bulge; DESI is the better tracker of inclination and interaction; the rest is level.** N&A T-type and winding are unrelated for both (τ −0.08 to −0.01): winding is not a Hubble stage.
- Human DECaLS votes (GZD-5/8) are the weakest source on the same galaxies (e.g. winding 0.40 against GZ2 0.46 and DESI 0.43 on n = 327; disturbed 0.24 against 0.41). They are ≥ 5 votes on a tree with "can't tell" answers GZ2 does not have; the source, not the people, may differ.
- **What G3 measures.** The experts and the GZ2 voters graded the same SDSS images, so G3 measures consistency under a shared visibility limit, not accuracy against the galaxy. DESI's deeper-image advantage on inclination and interaction is the one hint of an image effect.
- **Caveats on reading this as "labeller quality":** experts graded SDSS images, GZ2 voters SDSS images, DESI predictions and GZD voters deeper Legacy ones; DESI's model was trained on GZD votes; the expert grade has its own uncertainty.
  A gap can be the image.
- Arm count has no expert grade in either catalogue, and no label source could be scored on it.

## For the decision before G4

GZ2 votes already track the experts at τ 0.49 (winding), 0.71 (bulge), 0.55–0.67 (spiral, bar); DESI is level or better on inclination and interaction only. The labels that looked weakest to A1 (winding, arm count) do not have a better
alternative here: winding is 0.49, arm count unscorable. On this evidence G4's expected gain is small where it can be scored, and arm count, where it might matter, cannot be scored by these gold sets. I'd hold G4; if it runs, start with arm curvature and bulge on the
~1,000 EFIGI galaxies with a GZ2-∩-DESI match, and price it first.
