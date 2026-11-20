# A1's spiral axis: cleaned, fitted and straightened (Almagest geometry)

*20 Nov 2026, branch `almagest-geometry`. Descriptive, for the Plate only: no claim, nothing in the science
record, no rung change. Answer `t04_spiral_a08_spiral`, full population, on the ladder's R footing (A1's real
embeddings and the three banked untrained seeds over P2's union, identical split; penultimate-block mean-pooled
384-d vector). Fitted on train (38,000 eligible galaxies), measured on test (33,081). Code
`artifacts/almagest_spiral.py`; numbers in `docs/almagest/a1_spiral_{r1,clean,curve,plate,figs}.json` (and
`_logsens` for the sensitivity reading). Encoder `runs/m_v2/encoder.pt` (SHA-1 `34bb30a7…`); bank SHA-1s in each
JSON's `provenance`.*

## Summary

- The linear readout holds on A1 (the MLP does not beat it).
- **Finding: about 0.06 of A1's raw spiral readout rides on size and brightness.** Removing five nuisances
  (size, magnitude, S/N, PSF, redshift) takes the probe from AUC 0.806 to **0.746**, and turns its direction by
  cos **0.83**. The cleaned axis is "spiral at fixed size and brightness".
- The axis is now the cleaned probe's score d, and γ(d) is the mean galaxy's path along it. s (arc length of
  γ) is monotone in d, so it reads exactly as the probe does (AUC 0.7462, asserted equal).
- γ runs along **Σw** (the pattern), not w (the filter): cos(chord, Σw) **0.99**, cos(chord, w) 0.06. Against
  its own chord it bends by at most **0.18** path lengths (1 − chord/arc **0.134**), less than the untrained
  seeds (0.26–0.29; 0.40–0.41).
- **The R2 reach mixture recurs.** γ fitted on the poorly-voted half (question reach < 6) departs from the
  full fit by ~10× the random-half noise in Hausdorff distance (~5× in mean gap), mostly in the high-spiral
  tail (flagged); γ is fitted on the full population as it is.
- The Plate (format v3, additive) carries d, s and |r| per galaxy, γ, the end tangents, the d → s map, and both
  unit vectors. The old spiral axis (QR coordinate 7) read AUC **0.649** on the Plate's test galaxies, against
  **0.797** for the probe direction w[7] it stood in for.

## Step 1: linearity re-check

Replicates R1 (`r_nonlinear.stage_r1`) for spiral, full population, with the shuffled-label and cluster controls.

| | value |
|---|---|
| linear probe AUC | 0.8060 |
| MLP AUC (width 16 by validation; 16→512: 0.802, 0.802, 0.801, 0.797, 0.794, 0.788) | 0.8033 |
| headroom (MLP − linear), 95% CI | −0.0027 [−0.0048, −0.0007] |
| verdict as R1 | NO EVIDENCE that the MLP beats linear |
| shuffled-label MLP: test / train AUC | 0.497 / 0.609 (real train 0.819) |
| cluster-selectivity, C = 100 / 1,000 | −0.057 [−0.060, −0.053] / −0.112 [−0.118, −0.106] |
| ladder record `r_r1.json` (cross-check) | linear 0.805995, MLP 0.803268, headroom −0.002726 — matches |

The straightening plan's assumption (a linear readout) stands.

## Step 2: ridge cleaning

**Construction.** Each nuisance (petroRad_r, modelMag_r, snr_r, psfWidth_r, specz) becomes four covariates:
its train-ECDF rank − 0.5 (rank held inside [0.5/n, 1 − 0.5/n]), that squared, its normal score Φ⁻¹(rank), that
squared — 20 columns, centred on train. A ridge (α = 1) of the 384-d embedding on them is fitted on the ladder
union's train rows and subtracted from every row; centring keeps the embedding's own mean.

**Dropped:** 0 rows (union, spiral train, spiral test — every row has all five nuisances). 710 union rows with
a suspect size value (`petrorad_suspect`) are kept and counted.

| check | value |
|---|---|
| (a) spiral AUC, raw → cleaned | 0.8060 → **0.7462** |
| (b) cosine, raw vs cleaned probe direction | **0.827** |
| (c) nuisance decode R² after cleaning (test; target = raw value / rank) | size 0.0039 / ≈0; magnitude 0.0004 / ≈0; S/N ≈0 / ≈0; PSF ≈0 / ≈0; redshift 0.0014 / ≈0. None > 0.05 (before cleaning, rank targets: 0.61, 0.61, 0.55, 0.58, 0.49) |
| (d) planted Gaussian nuisance (gate): corr with its direction, decode R² | 0.9999 → **0.0115**; 1.000 → **−0.0001**: removed |
| planted lognormal (stress reading, not a gate) | corr 0.9999 → 0.176; R² 1.000 → 0.031 |

The lognormal plant is linear in its raw value e^g. Every rank-based column (and a log column) sees only g,
so none reaches it; only a raw-value covariate would. The real nuisances are clean (≤ 0.004) under this basis,
so the lognormal plant is harsher than the data.

**Log-value sensitivity** (reported, not a gate): the 20 columns plus each nuisance's log value (size, S/N,
PSF: log; redshift: log(1 + z), because 14 union specz are below zero; magnitude as is), all 25 standardised,
no clipping. Spiral AUC 0.745, cosine 0.825, worst decode R² 0.0013, Gaussian plant 0.0116, lognormal plant
0.175 / 0.031. Step 3 under it: see *Cross-fits*.

## Step 3: γ(d), the straightened axis

**Construction.** d = the cleaned probe's decision (L2 logistic, train), d = w·z + b, with z the cleaned
embedding in the probe's own standardisation. γ(d) = the mean z in **20 equal-count bins of d** (train, 1,900
galaxies each), smoothed by a cubic smoothing spline (scipy `make_smoothing_spline`, GCV) through the bin means
in their top-k PCs, with the bin-mean d as parameter. s = the arc length of γ from the first bin mean's d, over
the arc length to the last's. Beyond the end bin means γ continues straight along the secant of the last two
bin means (the extrapolation rule), and s with it, below 0 and above 1.

| | value |
|---|---|
| bins | 20 × 1,900 |
| first bin: count, d range | 1,900, [−9.654, −2.793]; bin-mean d −3.451 |
| last bin: count, d range | 1,900, [0.880, 7.306]; bin-mean d 1.418 |
| s span (d) | −3.451 to 1.418 (train range of d: −9.654 to 7.306) |
| k (99% of the bin means' variance) | **7** (PC1 89.4%, then 6.8, 1.8, 0.4, 0.3, 0.3, 0.2%) |
| arc length / chord (z units) | 13.43 / 11.62 |
| **check: s strictly monotone in d** | asserted, holds |
| **check: AUC(s) = AUC(d)** | **0.7462 = 0.7462** (asserted equal; = step 2's cleaned probe) |
| test s range; outside [0, 1] | −1.12 to 2.01; 1,212 of 33,081 (3.7%) |

**Bend** (γ sampled at 1,001 equal arc-length points; path-length units):

| | A1 | untrained seeds (own spans) |
|---|---|---|
| max displacement off γ's own chord | **0.182** (at s ≈ 0.48) | 0.263–0.287 |
| 1 − chord/arc | **0.134** | 0.397–0.407 |
| cos(chord, Σw) — the pattern | **0.992** | 0.998–1.000 |
| cos(chord, w) — the filter | 0.058 | 0.039–0.043 |
| cos(w, Σw) | 0.056 | 0.037–0.040 |
| max displacement off the Σw line | 0.213 | 0.250–0.282 |
| k | 7 | 2 |
| probe AUC (d) | 0.746 | 0.658–0.659 |

A straight mean path E[z | d] would run along Σw; γ does, to cos 0.99. The probe direction w is almost
orthogonal to it (cos 0.06): the probe reads a low-variance direction, and the galaxies move along a
high-variance one as d grows. A1's γ bends less than an untrained encoder's, against its chord and against
Σw; it uses more dimensions to do it (k 7 against 2). The raw (uncleaned) γ bends more: 1 − chord/arc 0.370,
k 5, with a sharp hook at the high-d end that the cleaning removes (figure 1).

**Off-axis residual** r = z − γ(d) (384-d): median |r| = 1.07 path lengths. Spearman ρ of |r| (test, n = 33,081)
with S/N −0.126, magnitude +0.119, PSF −0.091, size −0.083, expAB_r −0.067, redshift −0.019, question reach
+0.008. Descriptive: the off-axis spread grows a little for fainter, lower-S/N galaxies.

**Cross-fits** (γ refitted, same d; distances in path lengths of the full fit, on its d grid; s over each
fit's own bin-mean span, so |Δs| also carries the difference in span):

| refit on | n | Hausdorff, all / s ≥ 0.8 | mean gap at matched d, all / s ≥ 0.8 | median \|Δs\| (test) | own span (d) | bend |
|---|---|---|---|---|---|---|
| well-voted half (reach ≥ 6) | 19,530 | 0.077 / 0.042 | 0.046 / 0.038 | 0.082 | −3.01 to 1.74 | 0.116 |
| **poorly-voted half (reach < 6)** | 18,470 | **0.384 / 0.384** | **0.084 / 0.252** | 0.091 | −3.71 to 0.62 | 0.211 |
| random half A | 19,000 | 0.031 / 0.023 | 0.018 / 0.019 | 0.010 | −3.43 to 1.44 | 0.137 |
| random half B | 19,000 | 0.040 / 0.040 | 0.019 / 0.022 | 0.005 | −3.48 to 1.40 | 0.141 |
| well-voted vs poorly-voted | | 0.422 / 0.422 | 0.127 / 0.288 (tail 2.26×) | | | |
| random A vs B | | 0.047 / 0.047 | 0.031 / 0.036 (tail 1.14×) | | | |
| log-value basis vs primary (cleaned-embedding units, matched s) | 33,081 test | 0.010 / 0.010 | 0.0043 / 0.0059 (1.37×) | 0.0033 / 0.0034 | | 0.134 |

**Tail flag** (set before the run: the tail's mean gap more than twice the whole's): **flagged** for the
poorly-voted half against the full fit (3.0×) and against the well-voted half (2.26×); not for the random
halves (1.06–1.18×) or the log-value basis (1.37×; s Pearson 0.997 all, 0.913 in the tail, Spearman 0.999 /
0.985, |Δs| p90 0.009). Not chosen between: γ is fitted on the full population. Two notes. Hausdorff is a
maximum, so its tail value can hardly exceed the whole's (several ratios read exactly 1.0): the flag reads the
mean gap, and Hausdorff is reported only. And the poorly-voted half's last bin mean sits at d 0.62, below the
full fit's 1.42: across most of the tail its γ is its own end tangent, because few poorly-voted galaxies reach
high d.

**Finding: the R2 reach mixture recurs.** Galaxies whose spiral question reached few voters are a different
population (most voters routed them elsewhere), and their mean path along the same probe score departs from
the well-voted galaxies' by ~10× the random-half noise (Hausdorff 0.42 against 0.047), most of it in the
high-spiral tail.

**Against the vote-fraction path** (the first design's curve: 10 equal-width vote-fraction bins, same z space;
each path by its own arc length):

| | γ(d) | vote-fraction path |
|---|---|---|
| k / 1 − chord/arc / max off own chord | 7 / 0.134 / 0.182 | 8 / 0.107 / 0.127 |
| cos(chord, Σw) / cos(chord, w) | 0.992 / 0.058 | 0.996 / 0.055 |
| cos of the two chords | 0.988 | |
| cos of their sideways displacements (off own chord), all 1,001 points / at each one's maximum | **0.53 / 0.53** | |
| vote path's displacement inside γ's 7 PCs | 98.7% | |

They run the same way (chords cos 0.99, the vote path almost wholly inside γ's PCs) and bend partly the same
way (cos 0.53). Under the w yardstick the earlier run gave 0.87 for the same comparison; that number was mostly
the shared tilt from w towards Σw, not bend.

**Extrapolation** (defined, not rendered): below s = 0 or above s = 1, γ(s) = γ(end) + (s − end)·L·t_end, with
t_end the outward unit secant of the last two bin means and L the arc length; s continues at ds/dd = 0.206
below and 0.194 above. The tangents are in the curve file (`runs/almagest/a1/a1_spiral_gamma.npz`) and, in
80-D, in the Plate header.

## Step 4: the Plate, format v3 (additive)

`runs/almagest/a1/plate.bin` (SHA-1 `a29587ef…`, 8,483,728 bytes) = A1's ladder Plate
`runs/ladder_v2/a1/plate/plate.bin` (read only; SHA-1 `e922ba27…`, 8,286,352 bytes; 40,000 galaxies) with
every v2 section byte-identical (asserted), plus three per-galaxy sections and a `spiral` header block.
Documented in `artifacts/plate_data_format.md`, "Version 3".

- Per galaxy: `spiral_d` (cleaned d, uint8 over its 0.1–99.9 percentiles, 80 clamped), `spiral_s` (uint8,
  clamped to [0, 1]; 1,615 galaxies lie outside), `spiral_r` (|r|, uint8 with a header scale). No galaxy lacks
  a nuisance.
- Header: γ at s = 0, 0.01, …, 1 in the file's 80-D coordinates; the end tangents in arc-length units; the
  d → s map (101 knots, linear slopes beyond; reproduces s exactly inside); the two unit vectors:
  **filter w** (for reading a position) and **pattern Σw** (for walks), with a note saying which is which. In
  80-D they keep 83.7% (w) and 99.4% (Σw) of their norm; cos between them 0.064.
- The spiral axis now reads `spiral_d`; `probe_basis` and the v2 fields are unchanged.

**The axis fix**, on the Plate's 6,622 probe-test galaxies eligible for spiral (2,165 positive):

| axis | AUC |
|---|---|
| old: QR coordinate 7 (spiral's direction with answers 0–6 removed) | **0.649** |
| `probe_basis.w[7]` on the stored coords (the raw probe direction) | 0.797 |
| raw probe, stored read-out byte / full 384-d | 0.797 / 0.797 |
| filter w (cleaned) projected to 80-D, on the raw coords | 0.786 |
| **new: cleaned d (`spiral_d`), float / byte; `spiral_s` byte** | **0.718 / 0.718 / 0.718** |

Reading QR coordinate 7 as the spiral axis cost 0.148 AUC against the probe it stood for. The new axis reads
lower than the raw probe by design: it is spiral at fixed size and brightness (the drop here, 0.079, is larger
than on the full test split, 0.060, on the Plate's cluster-stratified, rare-class-oversampled subset).

## Step 5: figures

- `a1_spiral_path.png`: γ on its top two PCs, raw (k 5, bend 0.370) and cleaned (k 7, bend 0.134), with the 20
  bin means and s marked.
- `a1_spiral_sideways.png`: γ's displacement against s, off its own chord and off the Σw line, A1 against the
  untrained range (each seed on its own span).
- `a1_spiral_strip.png`: real test galaxies at s = 0, 0.1, …, 1, the five nearest γ(s) among those with s within
  ±0.05 (469–6,011 per slice). s 0–0.7 is smooth and compact galaxies (vote fractions mostly 0); spiral
  structure appears from s 0.8 (fractions 0.2–0.93) and is clear at 0.9–1.0 (0.52–1.0). Object IDs, s and vote
  fractions in `a1_spiral_figs.json`.

## Deviations from the brief, declared

1. **Cleaning basis.** The brief's rank − 0.5 and its square flatten out at the ends and cannot follow an
   effect that grows into a nuisance's tails; check (d) caught it (planted correlation left at 0.199; size raw
   value R² 0.12). Adopted: the 20-column basis above (user decision). Diagnostic across four bases in the
   session; the spiral readout did not depend on the basis (cleaned AUC 0.746–0.754, cosine 0.83–0.85), the
   leftover nuisance did.
2. **Covariates centred.** Uncentred columns moved the cleaned space by a constant; harmless within a space,
   but it inflated a cross-fit Hausdorff from 0.016 to 0.73. Fixed before every number here.
3. **log(1 + z)** for redshift in the sensitivity basis (14 specz below zero).
4. **The coordinate.** The first design read s off the nearest point of a vote-fraction curve: s AUC 0.672
   against the cleaned probe's 0.746 (check (a) failed), 96% of galaxies ambiguous, because the within-bin spread
   (3–4 SD per PC) dwarfs the path (3.6 on PC1, ≤ 0.64 elsewhere) and the curve's 8-PC subspace caps even a
   linear probe at 0.691. Kept in `a1_spiral_curve_nearest*.json`. Replaced by γ(d) (user decision).
5. **s span.** First specified over the train range of d; 71% of γ's length then lay in straight extrapolation
   to a few outliers and test s bunched into 0.35–0.68. Now over the first to last bin mean (user decision).
6. **Bend yardstick.** First read against w; now against γ's own chord and Σw (user decision).
7. **Cross-fit tail measure.** The flag reads the mean gap at matched d (or s), not Hausdorff (see above); a
   random-halves pair is added as the noise reference. "The other train half" is read as the complement of
   the well-voted half.

Not done (out of scope): diffusion, other concepts, two-concept planes, any page change.
