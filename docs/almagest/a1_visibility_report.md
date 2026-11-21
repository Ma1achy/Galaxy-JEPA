# A1 visibility tests B0–B2 (21 Nov 2026)

Exploratory, and descriptive only. Nothing here is a claim; it feeds the 8×8 pre-registration.
Branch `almagest-geometry`. Code: `artifacts/almagest_visibility.py` (stages `b1`, `b2`).
Numbers: `a1_visibility_b1.json`, `a1_visibility_b2.json`. Figures: `a1_visibility_b1.png` and
`a1_visibility_b2_examples.png` (columns: own | planted | z = 0.07 | 0.10 | 0.13).
Post-hoc diagnostics, outside the pre-stated plan and labelled as such below, are under
`runs/almagest/a1/visibility/` (`b2_support.{py,json}`, `b2_diag.py`, `b2_edge.py`).

## B0: stamp scale (answered earlier)

Stamps are a fixed 256 px at native 0.396″/px, **not** scaled to galaxy size. petroRad_r is used
only for JEPA's target-sampling box (2.5 × R_petro, `objectives/jepa.py` via `data/bbox.py`).
**For 8×8:** that box is a held-fixed variable (`TODO.md`, Epic R).

## B1: AUC by resolution decile (petroRad_r / psfWidth_r)

**Method.**
- Answers: the 30 that are STRONG or SEPARATE.
- Deciles are taken over each answer's eligible test galaxies.
- Probes are fitted on train rows, for A1 and the 3 untrained seeds, cleaned and raw.
- CIs: 1,000 galaxy bootstrap resamples within each decile.
- **Knee:** the first decile, in ascending resolution, where A1's AUC exceeds the untrained maximum
  by 0.05. This uses the point estimate, as specified.
- Decile 1 runs up to R_petro ≈ 2.9 × the PSF FWHM (t09–t11: 3.0–3.2). Decile 10 starts at about 9×.

| answer | label | knee cleaned / raw | deciles clearing, cleaned / raw | cleaned AUC, decile 1 (untrained max) | decile 10 |
|---|---|---|---|---|---|
| smooth | STRONG | 1 / 1 | 9 / 10 | 0.658 (0.594) | 0.875 (0.825) |
| features | STRONG | 2 / 1 | 9 / 10 | 0.661 (0.629) | 0.869 (0.817) |
| edge-on yes | SEPARATE | 1 / 1 | 10 / 10 | 0.705 (0.591) | 0.899 (0.801) |
| edge-on no | SEPARATE | 1 / 1 | 10 / 10 | 0.730 (0.604) | 0.909 (0.809) |
| bar | STRONG | 4 / 2 | 7 / 9 | 0.533 (0.494) | 0.632 (0.549) |
| no bar | STRONG | 4 / 1 | 7 / 10 | 0.548 (0.510) | 0.644 (0.556) |
| spiral | SEPARATE | 3 / 3 | 8 / 8 | 0.562 (0.533) | 0.844 (0.778) |
| no spiral | SEPARATE | 3 / 1 | 8 / 9 | 0.590 (0.559) | 0.853 (0.782) |
| no bulge | STRONG | 1 / 1 | 8 / 8 | 0.653 (0.574) | 0.596 (0.583) |
| just noticeable | STRONG | none / none | 0 / 0 | 0.539 (0.505) | 0.654 (0.614) |
| obvious | STRONG | 3 / 3 | 5 / 4 | 0.571 (0.537) | 0.634 (0.634) |
| dominant | SEPARATE | 6 / 4 | 4 / 7 | 0.529 (0.511) | 0.774 (0.678) |
| odd yes | STRONG | 1 / 1 | 7 / 8 | 0.743 (0.663) | 0.658 (0.594) |
| odd no | STRONG | 1 / 1 | 8 / 7 | 0.742 (0.663) | 0.657 (0.589) |
| completely round | STRONG | 1 / 1 | 10 / 10 | 0.665 (0.583) | 0.717 (0.641) |
| in between | STRONG | 3 / 3 | 7 / 6 | 0.523 (0.526) | 0.668 (0.573) |
| cigar | STRONG | 1 / 1 | 10 / 10 | 0.733 (0.611) | 0.877 (0.789) |
| ring | STRONG | 4 / 1 | 7 / 9 | 0.585 (0.554) | 0.726 (0.628) |
| lens/arc | SEPARATE | 4 / 4 | 6 / 6 | 0.586 (0.548) | 0.752 (0.677) |
| disturbed | STRONG | 5 / 5 | 1 / 2 | 0.617 (0.580) | 0.625 (0.576) |
| irregular | STRONG | 5 / 8 | 5 / 1 | 0.598 (0.590) | 0.714 (0.652) |
| other | STRONG | 4 / 6 | 6 / 4 | 0.587 (0.568) | 0.668 (0.617) |
| merger | STRONG | 7 / 6 | 1 / 3 | 0.618 (0.612) | 0.652 (0.610) |
| dust lane | SEPARATE | 5 / 6 | 5 / 4 | 0.643 (0.613) | 0.877 (0.793) |
| bulge rounded | STRONG | 5 / 5 | 6 / 5 | 0.517 (0.498) | 0.706 (0.623) |
| bulge-shape no bulge | STRONG | 3 / 2 | 8 / 8 | 0.524 (0.523) | 0.679 (0.618) |
| tight | STRONG | 1 / 8 | 2 / 1 | 0.561 (0.510) | 0.617 (0.589) |
| loose | SEPARATE | 4 / 4 | 6 / 6 | 0.522 (0.531) | 0.668 (0.590) |
| 2 arms | SEPARATE | 3 / 2 | 4 / 6 | 0.527 (0.512) | 0.570 (0.547) |
| can't tell (arms) | SEPARATE | 6 / none | 4 / 0 | 0.561 (0.540) | 0.658 (0.578) |

**Reading** (descriptive):
- **Clear above the untrained encoders at every resolution:** the first-order answers (smooth, features,
  edge-on, cigar, completely round, odd).
- **Knee at decile 3–4,** i.e. R_petro ≳ 4–5 × PSF: bar, spiral, ring, lens/arc, bulge obvious, loose
  winding.
- **Knee at decile 5–7:** dominant bulge, dust lane, disturbed, irregular, merger, bulge rounded.
- **The knee rule is fragile where curves are flat.** It uses the point estimate, not the CI, and a
  curve can clear once and fall back:
  - tight winding: knee 1 cleaned, but only 2 of 10 deciles clear;
  - disturbed and merger: 1 decile each.
  - Read "deciles clearing" beside the knee; for a pre-registration, the knee should be stated on the
    CI or as a run of k deciles.
- **Cleaning sometimes moves the comparison.** The cleaned no-bulge AUC *falls* with resolution
  (0.71 → 0.60) while raw rises (0.73 → 0.83). The cleaner removes size, and size carries the label in
  large galaxies.
- **Label prevalence shifts with resolution.** For example, smooth is 95% positive in decile 1. Within-decile
  AUCs compare different populations across deciles.

## B2: redshifting intervention (simplified FERENGI)

**Declared:**
- **This resamples images. It is an intervention test, separate from the training pipeline's
  no-rebin rule.** The training rule is untouched.
- Angular size × D_A(z0)/D_A(z1) and flux × (D_L(z0)/D_L(z1))², Planck18. **Together these give the
  (1+z)⁴ surface-brightness dimming exactly** (D_L = (1+z)² D_A); **only the k-correction is missing.**
- A Gaussian PSF brings the result up to the median r-band psfWidth of real test galaxies at z1, applied
  to all bands. The added blur is σ_add² = σ_target² − (s·σ_orig)² − (anti-alias σ after the shrink)²,
  s = D_A(z0)/D_A(z1).
  - **Point-source check (built in, ±5%):** +0.4%, +0.1%, −0.9%, +1.2%. That includes
    0.02 → 0.13, where the shrunk PSF is 0.21″ and the added blur does nearly all the work.
- Gaussian noise brings the result up to the median per-band sky σ at z1, measured on 400 real galaxies.
  The shrunk image sits centred on a 256-px canvas whose outside is noise only.
- **Inverse stretch:** max error 3.6e−15. **Planted run** (z1 = z0): cos(planted, own) median 1.0,
  min 0.99999988; cos(own, bank) min 0.99999987.
  - This confirms the embedding path only. A planted stamp is a copy, so it cannot test the resampling.

**Sample.**
- Test galaxies with specz < 0.05: 8,897.
- petroRad_r above that subset's median: 4,448.
- **Declared deviation: source specz ≥ 0.01**, applied after the size cut. n = **4,123**.
  - Why: z ≤ 0 (4 galaxies) has no D_A. Below 0.01, peculiar velocities put D_A from redshift ≥ 10%
    off, and the shrink to 0.13 leaves fewer than 23 px.
  - The first run crashed on these. The script now asserts a shrink of ≥ 16 px.
- Targets 0.07, 0.10 and 0.13 are all below the test p95 of specz (0.152), as asserted.
- Target PSF is 1.33–1.34″ at all three. Sky σ is nearly constant too.
- Median shrink f: 0.47, 0.34, 0.27. Median flux factor g: 0.19, 0.09, 0.05.

### Score movement (mean Δ against planted, in test-SD units of each score)

**Raw scores move a lot, and monotonically in z** (full sample, n = 4,123; bootstrap CIs in the json,
all narrow at ±0.03–0.06 SD). A selection, at 0.07 / 0.10 / 0.13:

| answer | raw Δ / SD |
|---|---|
| odd yes / no | −1.26 −1.88 −2.19 / +1.27 +1.92 +2.23 |
| smooth / features | +0.95 +1.24 +1.38 / −0.95 −1.23 −1.37 |
| edge-on yes | +0.72 +1.19 +1.41 |
| can't tell (arms) | +1.12 +1.60 +1.82 |
| irregular | +1.21 +1.49 +1.61 |
| disturbed | +0.99 +1.35 +1.54 |
| no bulge | +0.76 +1.12 +1.30 |
| dominant | +0.90 +1.19 +1.35 |
| bar / no bar | −0.60 −0.92 −1.09 / +0.64 +1.06 +1.29 |
| ring, lens/arc, loose, 2 arms | within ±0.5 at every z |

The direction is consistent: under redshifting, A1's raw scores move towards smooth, no bar, edge-on,
no bulge, odd "no", but also disturbed, irregular and can't-tell. That fits a "less resolved / harder to
call" reading.

**The cleaned scores, as computed in the stage, are not interpretable.** A post-hoc diagnostic found why:

1. **The simulated galaxies leave the cleaner's support.** Dimming takes them to r 21–23, while the SDSS
   spectroscopic sample stops near r = 17.77 (train p99 17.2). Within that limit there are 67% of sims at
   0.07, 42% at 0.10, and **24% at 0.13**. A large nearby galaxy moved to 0.13 is fainter than almost any
   real galaxy observed there.
2. **SNR has no valid match.** The stage looks SNR up from real galaxies near (z, mag). Out of support it
   falls back to wide windows, and 348 galaxies at 0.10 got NaN.
3. **The cleaner imputed those NaNs silently.** The old `almagest_spiral.rank_fit` mapped a NaN to the
   **top** rank (1 − 0.5/n): `searchsorted` places NaN after every value, and the clip held it there. A
   first draft of this report said "median"; that was wrong, corrected 21 Nov. Every NaN SNR was therefore
   read as the highest SNR in train. It broke the fail-loudly rule; **fixed 21 Nov**: `rank_fit` now
   raises on any non-finite value, at fit and at apply (`tests/test_almagest_rank_fit.py`). Rerunning
   this stage as written would now stop at that line, which is the intended behaviour.

   **No earlier result had NaN inputs.** Every other fit and application of the cleaner (spiral steps
   1–5, the 37 axes, relabel2, B1, and B2's own-nuisance "planted" row) passes only rows from the
   all-finite nuisance mask (`np.isfinite(N).all(1)`: `fit_cleaner`, `clean_matrix`, `Fit`, the c2 bank,
   `almagest_axes`'s `ok`, B2's `test_rows`). Rows with a missing nuisance were dropped and counted
   (`dropped_missing` in the spiral JSON), never ranked. The spiral step's planted column is Gaussian
   draws. B2's simulated SNR is the only path that ever handed `rank_fit` a NaN.
4. The result is that the SNR term alone swings the ridge shift between +0.5 and +2.1 (smooth) across
   targets, so the stage's cleaned Δ jumps: smooth +0.12, −1.75, −0.47.

**Post hoc, in-support subset.** Kept: simulated r ≤ 17.77 and ≥ 20 real SNR matches within ±0.25 mag
and ±0.01 z. n = 2,468 / 1,594 / 708. These are brighter sources, and **a different subset at each z.**
Exploratory; not the pre-stated test.

| answer | label | raw Δ/SD, all (0.07 / 0.10 / 0.13) | raw Δ/SD, in-support | cleaned Δ/SD, in-support [95% CI at 0.13] |
|---|---|---|---|---|
| smooth | STRONG | +0.95 / +1.24 / +1.38 | +0.85 / +1.11 / +1.03 | −0.06 / −0.26 / −1.02 [−1.16, −0.89] |
| features | STRONG | −0.95 / −1.23 / −1.37 | −0.85 / −1.11 / −1.00 | +0.07 / +0.28 / +1.08 [+0.94, +1.22] |
| edge-on yes | SEPARATE | +0.72 / +1.19 / +1.41 | +0.29 / +0.66 / +0.71 | −0.09 / +0.13 / +0.11 [+0.03, +0.18] |
| edge-on no | SEPARATE | −0.63 / −1.05 / −1.25 | −0.21 / −0.55 / −0.59 | +0.12 / −0.11 / −0.09 [−0.16, −0.02] |
| bar | STRONG | −0.60 / −0.92 / −1.09 | −0.42 / −0.70 / −0.52 | +0.28 / +0.33 / +0.88 [+0.76, +1.01] |
| no bar | STRONG | +0.64 / +1.06 / +1.29 | +0.45 / +0.86 / +0.82 | −0.32 / −0.29 / −0.81 [−0.95, −0.69] |
| spiral | SEPARATE | −0.33 / −0.48 / −0.57 | −0.22 / −0.25 / +0.00 | +0.24 / +0.59 / +1.36 [+1.27, +1.47] |
| no spiral | SEPARATE | +0.42 / +0.59 / +0.69 | +0.32 / +0.38 / +0.19 | −0.19 / −0.52 / −1.29 [−1.38, −1.18] |
| no bulge | STRONG | +0.76 / +1.12 / +1.30 | +0.73 / +1.28 / +1.55 | −0.02 / +0.31 / +0.11 [−0.01, +0.21] |
| just noticeable | STRONG | +0.28 / +0.52 / +0.65 | +0.26 / +0.69 / +1.20 | +0.22 / +0.97 / +1.83 [+1.74, +1.92] |
| obvious | STRONG | −0.51 / −0.66 / −0.72 | −0.54 / −0.89 / −1.17 | −0.47 / −1.09 / −1.72 [−1.79, −1.64] |
| dominant | SEPARATE | +0.90 / +1.19 / +1.35 | +0.81 / +1.03 / +0.95 | +0.22 / −0.19 / −0.72 [−0.82, −0.61] |
| odd yes | STRONG | −1.26 / −1.88 / −2.19 | −1.02 / −1.67 / −1.95 | −0.06 / −0.31 / −0.39 [−0.48, −0.30] |
| odd no | STRONG | +1.27 / +1.92 / +2.23 | +1.03 / +1.70 / +1.99 | +0.05 / +0.31 / +0.41 [+0.31, +0.50] |
| completely round | STRONG | −0.60 / −0.88 / −1.04 | −0.32 / −0.57 / −0.74 | −0.37 / −0.74 / −1.10 [−1.17, −1.03] |
| in between | STRONG | +0.37 / +0.13 / +0.04 | +0.60 / +0.43 / +0.52 | +0.66 / +0.66 / +0.99 [+0.90, +1.07] |
| cigar | STRONG | +0.28 / +0.60 / +0.76 | −0.04 / +0.26 / +0.42 | −0.17 / +0.15 / +0.31 [+0.24, +0.38] |
| ring | STRONG | +0.17 / +0.12 / +0.09 | +0.08 / −0.12 / −0.12 | +0.48 / +0.69 / +1.26 [+1.16, +1.37] |
| lens/arc | SEPARATE | −0.06 / −0.00 / +0.05 | −0.12 / −0.24 / −0.49 | −0.31 / −0.57 / −1.09 [−1.18, −1.00] |
| disturbed | STRONG | +0.99 / +1.35 / +1.54 | +1.03 / +1.64 / +2.28 | +0.28 / +0.61 / +0.97 [+0.88, +1.06] |
| irregular | STRONG | +1.21 / +1.49 / +1.61 | +1.47 / +2.13 / +2.62 | +0.95 / +1.41 / +1.64 [+1.57, +1.70] |
| other | STRONG | −0.20 / +0.00 / +0.12 | −0.44 / −0.42 / −0.59 | −0.72 / −0.94 / −1.52 [−1.61, −1.44] |
| merger | STRONG | −0.24 / −0.36 / −0.43 | −0.40 / −0.71 / −1.00 | −0.43 / −0.87 / −1.34 [−1.42, −1.27] |
| dust lane | SEPARATE | +0.17 / +0.35 / +0.38 | −0.21 / −0.16 / −0.23 | +0.05 / +0.16 / +0.12 [+0.04, +0.22] |
| bulge rounded | STRONG | −0.60 / −0.93 / −1.05 | −0.44 / −0.91 / −1.13 | +0.09 / −0.32 / −0.35 [−0.45, −0.26] |
| bulge-shape no bulge | STRONG | +0.39 / +0.68 / +0.81 | +0.23 / +0.61 / +0.80 | −0.04 / +0.43 / +0.58 [+0.49, +0.66] |
| tight | STRONG | +0.37 / +0.91 / +1.19 | −0.19 / +0.13 / +0.27 | −0.17 / +0.31 / +0.60 [+0.50, +0.70] |
| loose | SEPARATE | +0.26 / +0.17 / +0.11 | +0.41 / +0.33 / −0.04 | −0.03 / −0.53 / −1.29 [−1.39, −1.19] |
| 2 arms | SEPARATE | +0.45 / +0.24 / +0.13 | +0.69 / +0.61 / +0.76 | +0.31 / +0.26 / +0.57 [+0.48, +0.65] |
| can't tell (arms) | SEPARATE | +1.12 / +1.60 / +1.82 | +0.77 / +1.20 / +1.14 | −0.01 / +0.00 / −0.66 [−0.80, −0.54] |

**Reading** (descriptive):
- At 0.07, cleaning removes most of the raw movement on the first-order answers. Smooth goes from
  +0.85 raw to −0.06 cleaned; edge-on and odd are within ±0.12.
- At 0.13, **the cleaned scores move by ~1–1.8 SD for many answers, often opposite to raw**: smooth
  −1.02, spiral +1.36, just-noticeable +1.83. So the cleaner over-corrects: it subtracts more
  "redshift/brightness" structure than the simulated image carries.
- The likely, untested reason is the **missing k-correction**. In real data, the cleaner's redshift and
  magnitude terms also absorb colour change (real z = 0.13 galaxies are redder in g−r). The simulations
  are not redder, so subtracting the real-data redshift term over-shoots. The nuisance movement below
  points the same way.

### Movement along the nuisance directions

**Method.**
- OLS of the raw A1 embedding on the 5 standardised nuisances (train rows).
- "Implied" is each sim's embedding change projected onto those directions, in nuisance SDs.
- "Actual" is the simulated nuisance change.
- The 5 directions are correlated, so read them jointly.

| | size (implied / actual) | magnitude | SNR | PSF | redshift |
|---|---|---|---|---|---|
| z = 0.07 | −2.21 / −1.60 | +2.69 / +2.40 | −1.53 / −1.34 | +0.86 / −0.09 | +0.09 / +0.87 |
| z = 0.10 | −2.24 / −1.98 | +3.17 / +3.46 | −2.14 / −1.98 | +1.04 / −0.02 | +0.83 / +1.55 |
| z = 0.13 | −2.19 / −2.18 | +3.30 / +4.24 | −2.45 / −2.30 | +1.19 / −0.07 | +1.32 / +2.24 |

(Implied CIs are ±0.06–0.12; see the json.)

- **Size, magnitude and SNR move about as the real-data directions predict.**
- **Along "PSF" the embedding moves ~1 SD with no PSF change.** The galaxy shrinks relative to an unchanged
  PSF, and in real data that resolution loss is what the PSF direction carries.
- **Along "redshift" the embedding moves only 10–60% of the actual change.** This is consistent with
  the missing colour (k-correction) part of the redshift signal.

### Nearest-neighbour specz (cosine among the union's real galaxies; median of the NN medians)

| | planted | 0.07 | 0.10 | 0.13 |
|---|---|---|---|---|
| NN specz | 0.049 (sample's own z: 0.032) | 0.069 | 0.074 | 0.076 |

- The neighbours' z rises with the intervention, but it saturates at the union median (0.075).
- It is also pulled up even for the planted run (0.049 against 0.032).
- So this check shows the sims become "typical-z-like", not that they match their target z.

### Noise across the footprint edge (post hoc; `b2_edge.py`, 60 galaxies)

- `a1_visibility_b2_examples.png` shows the shrunk footprint as a visible square at z = 0.07.
- **Pixel statistics match across the edge:** σ inside/outside 1.02–1.03 / 1.00 of target in every
  band, level difference ≤ 0.04 σ.
- So the square is texture. Inside sits correlated, shrunk original noise topped up with white noise;
  outside is white only. The per-pixel top-up cannot match the spatial correlation.
- A1's response to that edge is untested. A noise-only footprint control would measure it.

## Feeding the 8×8 pre-registration (notes, not decisions)

1. **B1:** state the knee on the CI, or as k consecutive deciles clearing; the point-estimate knee is
   fragile on flat curves.
2. **B2, keeping the sims inside the observed population:** apply the r ≤ 17.77 limit, or draw sources
   so their dimmed magnitude stays in support. Otherwise the cleaner extrapolates.
3. **B2, the missing k-correction is not cosmetic for cleaned scores:** add a k-correction (or a colour
   shift matched to real galaxies at z1), or compare raw scores only.
4. **B2, correlated-noise control:** a footprint with no galaxy, to measure A1's response to the texture edge.
5. **Cleaner:** `rank_fit` raises on NaN (done 21 Nov); any 8×8 nuisance lookup must yield finite values or drop the row, declared.
6. **Held fixed:** the 2.5 × R_petro JEPA masking box.
