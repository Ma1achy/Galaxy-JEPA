# Part C — star/artifact: lumped label or blind encoder?

24 Nov 2026, branch `almagest-geometry`. **Descriptive; not hashed; nothing here enters the science record.**
Scripts `artifacts/axes_c_star.py` (pull, embed, run, sheet) and `axes_c_matched.py` (post hoc). Numbers
`c_star.json`. Sheets `c_sheet_raw.png`, `c_sheet_cleaned.png`. Working files `runs/almagest/axes_phys/c/`.

**Result: a partial yes.** A1 is not blind: its clusters are visibly different kinds of scene, and they match the flag
groups above chance. The stated rule ("clearly better than untrained") returns yes, but most of that is a difference in
the number of clusters. At matched k the raw advantage is small and the cleaned one is mostly gone. The untrained
encoder is *better* at frame edges; A1 is better at saturation. The label is certainly lumped.

## Sample

- The full train + test bank (195,677 galaxies) has 145 at star/artifact fraction ≥ 0.5 (raw `_fraction`), below 200, so
  the fallback **≥ 0.3** applies: **n = 467**. The 74,825-galaxy union alone has 184; the whole corpus 551.
- Flags: public SkyServer SQL, DR17, no token; 934 of 934 rows (467 + 467 random controls with fraction < 0.1, used for
  base rates only). Pinned by SHA-1 `af3ecbf3…` in `flags_record.json`. Bit values came from the server's `PhotoFlags`.
- Embeddings: A1 from the bank. The three untrained seeds were embedded here; every one equals its union bank exactly
  on the 184 shared galaxies (max |Δ| 0.0), and so does A1.

## Flags and types

| | star/artifact | control |
|---|---|---|
| SATURATED | 52% | 2% |
| SUBTRACTED (wings of a bright object) | 41% | 1% |
| NOTCHECKED | 29% | 1% |
| EDGE | 24% | 1% |
| BLENDED | 22% | 14% |
| DEBLENDED_AS_PSF | 9% | 0% |
| BRIGHT, DEBLENDED_AS_MOVING | 0 | 0 |

- **These are galaxies, not stars.** Photometric type is galaxy for 466 of 467 (r-band type: 460). Of the 335 with a
  spectrum, 327 are GALAXY, 6 QSO, 2 STAR. The votes are about the *scene*: a bright neighbour, a frame edge, a light
  gradient, or a shred of a large nearby galaxy.
- `DEBLENDED_AT_EDGE` equals `CHILD` on every row, and is not used.

## Clusters against flag groups

Groups, fixed before any clustering: the pattern of SATURATED × EDGE × BLENDED, with patterns under 10 galaxies merged
(8 groups). k-means on embeddings standardised by the union's train rows; k = 2–8 by silhouette.

| | A1 raw | untrained raw | A1 cleaned | untrained cleaned |
|---|---|---|---|---|
| k chosen (silhouette) | 7 (0.13) | 2, 2, 2 (0.52–0.54) | 3 (0.12) | 2, 2, 2 (0.50–0.52) |
| ARI, pattern groups | 0.153 | 0.061, 0.050, 0.055 | 0.139 | 0.050, 0.038, 0.039 |
| ARI, SATURATED alone | 0.074 | 0.002–0.003 | 0.160 | 0.007–0.011 |
| ARI, EDGE alone | 0.133 | 0.201–0.240 | 0.080 | 0.141–0.161 |

Permutation 97.5th percentile for A1: 0.013 raw, 0.012 cleaned. **Stated rule: clearly better in both spaces.**

**Matched k (post hoc).** A two-way split cannot score high against eight groups, so the rule favours whichever encoder
silhouette gives more clusters. Re-clustering 100 80% subsamples at every k, A1 minus the best untrained seed:

| k | raw, pattern ARI A1 / best untrained, 95% | cleaned |
|---|---|---|
| 2 | 0.087 / 0.061, [−0.009, 0.054] | 0.096 / 0.049, [0.020, 0.077] |
| 3 | 0.158 / 0.081, [0.043, 0.108] | 0.124 / 0.078, [−0.007, 0.088] |
| 4 | 0.148 / 0.087, [0.033, 0.103] | 0.114 / 0.092, [−0.012, 0.072] |
| 5 | 0.148 / 0.097, [0.032, 0.078] | 0.140 / 0.118, [−0.031, 0.049] |
| 6 | 0.141 / 0.106, [0.008, 0.062] | 0.138 / 0.125, [−0.018, 0.051] |
| 7 | 0.140 / 0.112, [0.000, 0.062] | 0.134 / 0.131, [−0.033, 0.031] |
| 8 | 0.139 / 0.119, [−0.009, 0.050] | 0.140 / 0.132, [−0.026, 0.041] |

The intervals are subsample spreads, not confidence intervals over new galaxies (80% subsamples overlap).

- **Raw:** A1 is ahead at k = 3–6. Its median subsample ARI is about 0.14–0.16 either way, so the gap closes because the
  untrained encoders catch up as k grows.
- **Cleaned:** ahead only at k = 2. Part of A1's raw advantage goes with the nuisance cleaning.
- **SATURATED** is A1's (raw ahead at k = 2–6, cleaned at k = 2–3; at k = 7 cleaned it is behind). **EDGE** is the
  untrained encoder's (ahead at k = 2–5 raw): a large blank region is a low-level pixel property.

## What the clusters look like (16 random galaxies each; my descriptions)

Raw, seven clusters: (1) frame edges and bleed at an edge; (2, 3, 6) a bright star with diffraction spikes beside a small
galaxy, three clusters of one kind; (4) light gradients, resolved pieces of large nearby galaxies, and crowded bright
neighbours; (5) compact or close multiple sources with few flags; (7) strong coloured backgrounds, satellite trails and
red halos. The cleaned solution collapses this to three: coloured halo / edge, few-flag compact sources, spiked stars.

**A limit on the ARI:** flags describe the target object, and these artefacts live around it. The low ARI (0.15) may
partly be the flags' fault. It is a lower bound on what the clusters separate.

## Reading (the brief's, descriptive)

- **The lumped label is a real problem.** 467 galaxies flagged "star or artifact" cover at least four visibly different
  causes, 99.8% of them catalogued as galaxies. One direction cannot be expected to carry all of them, which fits the
  one answer with MLP headroom (+0.166 [0.048, 0.286]).
- **"The encoder separates the subtypes" is supported, weakly.** A1's clusters differ visibly, and separate
  saturated from unsaturated scenes better than untrained. The match to flags is low, and the advantage over untrained
  depends on k. It is not a clear yes.
- **Not shown:** whether a split label would probe better. There are too few galaxies (37 train positives at ≥ 0.5 in
  the probe's union).
