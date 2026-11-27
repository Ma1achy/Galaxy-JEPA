# Part B: which physical quantities does A1 store as linear directions? (25 Nov 2026)

Branch `almagest-geometry`. Frozen embeddings, catalogues and statmorph. Descriptive, not hashed. Code: `artifacts/axes_b.py`, `axes_b_ids.py`,
`axes_b_statmorph.py`, `axes_b_mapnull.py`. Numbers: `b_fit.json`, `b_extras.json`, `b_map.json`, `b_map_null.json`, `b_catalogue.json`; figure `b_map.png`.

## Answer

**Every one of the 18 non-nuisance targets reads STRONG on the fixed rule (stable w, A1 above every untrained seed), but they are not equally strong.** In cleaned A1 the largest are
colour (R² 0.33), concentration, sSFR, axis ratio and asymmetry (R² 0.26–0.28). Bulge n, Re, disc Rd and stellar mass are nominal STRONGs with R² 0.03–0.06: real against the null, small in effect.
Raw R² is much higher (0.56–0.66 for colour, concentration, mass, sSFR), and the difference is the nuisance cleaning: the untrained seeds also reach raw ρ 0.5–0.6 on most of these, so raw numbers overstate A1.

**Which Galaxy Zoo axes are physics.** The score of each human axis (cleaned, test galaxies eligible for it) regressed on the 18 physical scores, half fitted and half tested:
- **Physics (R² ≥ 0.8):** edge-on and cigar/round (inclination, axis ratio), spiral / no-spiral (concentration, sSFR), smooth / features (concentration, bulge ellipticity), bulge *obvious* and *just noticeable*.
- **Mostly physics (0.6–0.8):** no bulge, ring, bar, irregular, in-between, lens/arc, bulge shape, "other".
- **Mixtures or not captured (R² < 0.6):** **dominant bulge (0.48), disturbed, merger, dust lane, odd, arm winding, arm number, can't tell.** What A1 reads there is not a sum of catalogue photometry.
- **What the R² measures:** alignment of directions inside A1 (how much of a human axis's score the physical axes' scores reproduce), not that the human label *is* that physical quantity.
- The control is in the table: 18 shuffled-target directions give R² 0.02–0.07 and 18 random directions 0.02–0.08 on the same rows, so these R² are not what any 18 directions would give.

A correlation of scores says the two directions agree in A1; it does not say A1 *uses* the physical quantity to answer the question. Bar and ring correlate with Simard's bulge Re and ellipticity (|r| 0.5–0.7): plausibly Simard's bulge absorbing bar and ring light rather than a bar–bulge relation. Not tested here.

## 1. The targets

| target | rows (train/test) | cleaned R² | cleaned ρ [95%] | partial ρ | untrained ρ | raw ρ | untrained raw ρ | split-half cos / shuffle max | label |
|---|---|---|---|---|---|---|---|---|---|
| concentration R90/R50 | 39,992/34,821 | 0.27 | 0.53 [0.52, 0.54] | 0.64 | 0.37–0.37 | 0.77 | 0.58–0.60 | 0.72 / 0.17 | STRONG |
| expAB_r | 39,997/34,828 | 0.26 | 0.47 [0.47, 0.48] | 0.49 | 0.23–0.25 | 0.49 | 0.25–0.27 | 0.80 / 0.12 | STRONG |
| deVAB_r | 39,997/34,828 | 0.27 | 0.49 [0.48, 0.49] | 0.50 | 0.24–0.26 | 0.51 | 0.27–0.29 | 0.84 / 0.13 | STRONG |
| fracDeV_r | 39,997/34,828 | 0.19 | 0.47 [0.47, 0.48] | 0.59 | 0.33–0.34 | 0.75 | 0.58–0.61 | 0.76 / 0.17 | STRONG |
| petroRad_r (size) *(is a nuisance)* | 39,997/34,828 | – | – | – | – | 0.79 | 0.69–0.70 | – | nuisance |
| modelMag_r (magnitude) *(is a nuisance)* | 39,997/34,828 | – | – | – | – | 0.78 | 0.72–0.73 | – | nuisance |
| g-r, k-corrected z=0 | 31,295/27,410 | 0.33 | 0.55 [0.54, 0.56] | 0.64 | 0.33–0.36 | 0.77 | 0.56–0.59 | 0.76 / 0.12 | STRONG |
| log M* (MPA-JHU) | 31,409/27,475 | 0.04 | 0.19 [0.18, 0.20] | 0.46 | 0.11–0.12 | 0.74 | 0.51–0.54 | 0.23 / 0.13 | STRONG |
| log sSFR (MPA-JHU) | 30,822/26,995 | 0.28 | 0.53 [0.52, 0.54] | 0.64 | 0.33–0.34 | 0.80 | 0.58–0.61 | 0.71 / 0.14 | STRONG |
| Simard B/T r | 39,196/34,131 | 0.19 | 0.45 [0.44, 0.46] | 0.59 | 0.30–0.30 | 0.74 | 0.55–0.57 | 0.66 / 0.14 | STRONG |
| Simard disc inclination | 35,642/31,022 | 0.24 | 0.49 [0.48, 0.50] | 0.51 | 0.22–0.24 | 0.49 | 0.24–0.25 | 0.84 / 0.20 | STRONG |
| Simard bulge n (free n) | 35,052/30,510 | 0.06 | 0.24 [0.23, 0.25] | 0.27 | 0.15–0.17 | 0.36 | 0.27–0.29 | 0.60 / 0.11 | STRONG |
| Simard bulge Re (log kpc) | 32,138/28,084 | 0.06 | 0.26 [0.25, 0.27] | 0.27 | 0.15–0.16 | 0.46 | 0.31–0.33 | 0.54 / 0.13 | STRONG |
| Simard disc Rd (log kpc) | 35,636/31,019 | 0.03 | 0.20 [0.19, 0.21] | 0.32 | 0.14–0.15 | 0.63 | 0.44–0.46 | 0.39 / 0.15 | STRONG |
| Simard bulge ellipticity | 32,140/28,087 | 0.10 | 0.33 [0.32, 0.34] | 0.33 | 0.16–0.18 | 0.36 | 0.21–0.22 | 0.47 / 0.11 | STRONG |
| statmorph asymmetry *(half)* | 9,813/9,813 | 0.27 | 0.40 [0.38, 0.42] | 0.43 | 0.26–0.28 | 0.47 | 0.34–0.37 | 0.46 / 0.11 | STRONG |
| statmorph gini *(half)* | 9,813/9,813 | 0.23 | 0.49 [0.48, 0.51] | 0.55 | 0.33–0.34 | 0.69 | 0.52–0.54 | 0.47 / 0.17 | STRONG |
| statmorph m20 *(half)* | 9,813/9,813 | 0.12 | 0.39 [0.37, 0.41] | 0.47 | 0.27–0.28 | 0.68 | 0.52–0.54 | 0.42 / 0.16 | STRONG |
| statmorph smoothness *(half)* | 9,813/9,813 | 0.26 | 0.32 [0.31, 0.34] | 0.34 | 0.17–0.19 | 0.46 | 0.31–0.33 | 0.33 / 0.09 | STRONG |
| statmorph sersic_n *(half)* | 9,785/9,785 | 0.19 | 0.47 [0.46, 0.49] | 0.57 | 0.32–0.33 | 0.73 | 0.56–0.57 | 0.41 / 0.13 | STRONG |

- Cleaned R² and ρ are on the test rows; partial ρ is the partial Spearman given the five nuisances. Size and magnitude are the nuisances themselves: raw only, marked.
- **Note (26 Nov 2026 audit fix):** `b_fit.json`'s `partial_ci` previously re-ranked residuals inside `boot()` (Spearman of residuals instead of Pearson of residualised ranks, placing 4/38 raw point estimates outside their 95% CIs). The point estimates above are unaffected; corrected partial-ρ gains over the best untrained seed with Pearson-on-residual bootstrap CIs are in `docs/axes/b_partial.json` (all 18 non-nuisance targets clear untrained by +0.062 to +0.256 partial ρ, 95% CIs wholly positive). `axes_b.py` and `axes_b_mapnull.py` (which previously sampled whitened `inv(S).T @ z` instead of covariance-weighted `S @ z`) are fixed for future runs.
- **What the label means:** STRONG is a threshold (w's split-half cosine above the shuffles' maximum, A1's test ρ above every untrained seed's), not a size. Bulge Rd, Re, n and mass pass it with R² 0.03–0.06.
  Stellar mass has split-half cosine 0.23 against the shuffle maximum 0.13, so its w is barely stable.
- **Rules fixed in the docstring before any fit.** One difference from the 37-axis run: 39 shuffles rather than 5 (with 5, pure noise reads "stable" about 1 time in 6; 39 measured 1 false-stable in 40 noise targets). Made in the self-test, before any real fit.
- statmorph exists for 20,000 test galaxies only (our r-band stamps): its targets are fitted on a fixed random half and tested on the other, so their n is 9.8k and their split-half cosines are not comparable to the catalogue targets'. statmorph 0.7.3 (BSD), benchmark 1.1 s per galaxy,
  374 flagged of 20,000 dropped (292 flag 1, 82 flag 2), 86 Sérsic-flagged; median S/N per pixel 7.9 (5th percentile 3.7).
- Sources: Simard et al. 2011 from VizieR's Harvard mirror (CDS hosts did not resolve); 1.12 M rows each of tables 1 and 2; P_pS ≤ 0.32 for 48% and 52%, not cut. g−r k-corrected to z = 0 for 58,705 of 74,825 (needs a spectrum). Details in `b_catalogue.json`.
- The measures behind Simard are SDSS g, r imaging and A1 saw g, r, i; they share photons with A1's input but not the pipeline.

## 2. Addition 1: is roundness inclination?

Yes, in A1. Roundness (ordinal, Part A) against Simard disc inclination: cleaned ρ 0.48 [0.47, 0.49], partial given nuisances 0.50; against `expAB_r` −0.48, partial −0.49; raw almost identical (0.47, −0.48). The A1 axis for `expAB_r` itself reaches ρ 0.47, so the roundness direction *is* as good an axis-ratio reader as a probe trained on it,
and the two directions correlate at 0.91–0.95 in score (edge-on 0.88–0.90). It is inclination up to what a ViT reads from r-band light.

## 3. Addition 2: the ordinal bulge score and "dominant − obvious"

| score | vs Simard B/T | vs bulge n (free n) | vs concentration |
|---|---|---|---|
| ordinal bulge, cleaned | 0.22 | 0.17 | 0.25 |
| ordinal bulge, raw | 0.69 | 0.33 | 0.72 |
| dominant − obvious, cleaned | 0.06 | −0.08 | 0.13 |
| dominant − obvious, raw | 0.15 | −0.06 | 0.08 |

(test rows, 30.5–34.8k; CIs ±0.01.) The bulge axis tracks concentration and B/T in raw but mostly through the nuisances: cleaning takes it from ~0.7 to 0.2–0.25. **"Dominant − obvious" does not track bulge size in any of the three measures**, consistent with Part A's bulge SEVERAL: the dominant answer is not the end of a bulge-size ladder. Its R² from all 18 physical scores is 0.48, against 0.80 and 0.92 for *just noticeable* and *obvious*.

## 4. Addition 3: SDSS photo flags as binary targets

Union rows only, fixed split, A1 against the three untrained seeds. No flag was rare enough to fall back to counts. Gain = AUC minus the best untrained, paired bootstrap.

| flag | positives train/test | raw AUC [95%] (best untrained) | gain | cleaned AUC [95%] (best untrained) | gain [95%] |
|---|---|---|---|---|---|
| SATURATED | 1,131/1,019 | 0.955 [0.950, 0.961] (0.921) | +0.035 | 0.938 [0.929, 0.947] (0.899) | +0.039 [+0.029, +0.049] |
| EDGE | 879/775 | 0.958 [0.951, 0.964] (0.917) | +0.040 | 0.908 [0.894, 0.922] (0.812) | +0.096 [+0.078, +0.116] |
| BLENDED | 6,347/5,498 | 0.673 [0.665, 0.680] (0.641) | +0.032 | 0.576 [0.568, 0.586] (0.556) | +0.020 [+0.011, +0.030] |
| DEBLENDED_AS_MOVING | 432/439 | 0.591 [0.562, 0.617] (0.547) | +0.044 | 0.570 [0.540, 0.597] (0.542) | +0.028 [-0.006, +0.061] |
| BAD_MOVING_FIT | 6,148/5,336 | 0.688 [0.680, 0.696] (0.645) | +0.043 | 0.633 [0.625, 0.641] (0.590) | +0.043 [+0.034, +0.052] |
| COSMIC_RAY | 25,920/22,462 | 0.746 [0.741, 0.750] (0.726) | +0.020 | 0.627 [0.621, 0.633] (0.594) | +0.033 [+0.028, +0.038] |
| DEBLEND_NOPEAK | 2,144/1,965 | 0.754 [0.743, 0.764] (0.711) | +0.043 | 0.626 [0.614, 0.640] (0.596) | +0.030 [+0.016, +0.044] |
| DEBLENDED_AS_PSF | 220/193 | 0.828 [0.797, 0.856] (0.763) | +0.066 | 0.872 [0.839, 0.902] (0.767) | +0.105 [+0.061, +0.154] |
| INTERP_CENTER | 4,387/3,777 | 0.528 [0.517, 0.538] (0.506) | +0.021 | 0.526 [0.515, 0.537] (0.507) | +0.019 [+0.005, +0.032] |
| MANYPETRO | 5,996/5,242 | 0.650 [0.642, 0.657] (0.614) | +0.036 | 0.572 [0.565, 0.580] (0.542) | +0.030 [+0.020, +0.040] |
| MOVED | 11,001/9,537 | 0.587 [0.581, 0.594] (0.557) | +0.031 | 0.583 [0.577, 0.590] (0.565) | +0.018 [+0.010, +0.026] |
| NODEBLEND | 6,347/5,498 | 0.673 [0.665, 0.680] (0.641) | +0.032 | 0.576 [0.568, 0.586] (0.556) | +0.020 [+0.011, +0.030] |
| NOPETRO | 5,622/4,950 | 0.746 [0.740, 0.755] (0.696) | +0.050 | 0.683 [0.675, 0.691] (0.613) | +0.070 [+0.061, +0.079] |
| NOTCHECKED | 947/834 | 0.952 [0.945, 0.958] (0.919) | +0.032 | 0.910 [0.896, 0.922] (0.841) | +0.069 [+0.053, +0.085] |
| PSF_FLUX_INTERP | 3,656/3,207 | 0.534 [0.523, 0.546] (0.510) | +0.025 | 0.520 [0.509, 0.532] (0.498) | +0.022 [+0.008, +0.036] |
| STATIONARY | 36,784/32,029 | 0.720 [0.711, 0.730] (0.666) | +0.054 | 0.658 [0.647, 0.669] (0.612) | +0.045 [+0.034, +0.057] |
| SUBTRACTED | 1,018/872 | 0.975 [0.970, 0.979] (0.970) | +0.006 | 0.959 [0.950, 0.967] (0.942) | +0.017 [+0.010, +0.024] |

- A1 gains on every flag in raw, and on all but DEBLENDED_AS_MOVING in cleaned (gain CI includes zero). **BLENDED and NODEBLEND are the same bit on every galaxy in the union**; that is one flag, so 16 distinct. SUBTRACTED untrained 0.97 already.
- What A1 learned beyond pixel statistics is smaller than the AUCs suggest: cleaned gains are +0.02 to +0.10, largest for DEBLENDED_AS_PSF (193 test positives), EDGE (+0.10), NOPETRO and NOTCHECKED (+0.07). Saturation and edges are mostly read by an untrained encoder too.

## 5. The map

`b_map.png`: score correlation of 18 physical and 30 STRONG/SEPARATE GZ2 axes, clustered on 1 − |r|. Clusters at distance 0.5: (i) concentration, B/T, Gini, sSFR, colour, mass, smooth, spiral, bulge obvious, winding loose; (ii) edge-on, round, cigar, axis ratios, inclination, dust lane; (iii) bar, ring, two arms, bulge Re; (iv) no bulge and bulge shapes with bulge n; (v) smaller groups for in-between/other/can't tell, odd, asymmetry + smoothness; disturbed, merger and tight winding alone.

## 6. The Plate

`plate_phys.bin` (beside `plate_axes.bin`) (14.2 MB): a copy of `plate_axes.bin` with two additive sections, `phys_filter_w` and `phys_pattern` (18 × 80, unit, cleaned), and a `phys_axes` header with test ρ and the fraction of w and Σw the 80-D keeps. The original plate and all its sections are unchanged (checked).

## 7. Table: R² of each GZ2 axis by the 18 physical scores

R² here is alignment of directions inside A1, not identity of the label with the quantity.

| GZ2 axis | R² by 18 physical scores | R² by shuffled-target / random directions | strongest physical (r) |
|---|---|---|---|
| rounded_a18_cigar_shaped | 0.95 | 0.05 / 0.03 | expAB_r (-0.95) |
| edgeon_a04_yes | 0.94 | 0.04 / 0.03 | expAB_r (-0.88) |
| edgeon_a05_no | 0.94 | 0.05 / 0.03 | expAB_r (+0.90) |
| rounded_a16_completely_round | 0.93 | 0.07 / 0.04 | deVAB_r (+0.92) |
| bulge_prominence_a12_obvious | 0.92 | 0.04 / 0.05 | fracDeV_r (+0.94) |
| spiral_a08_spiral | 0.88 | 0.03 / 0.06 | concentration R90/R50 (-0.86) |
| spiral_a09_no_spiral | 0.88 | 0.04 / 0.05 | concentration R90/R50 (+0.85) |
| smooth_or_features_a01_smooth | 0.84 | 0.03 / 0.04 | concentration R90/R50 (+0.75) |
| smooth_or_features_a02_features_or_disk | 0.82 | 0.04 / 0.05 | concentration R90/R50 (-0.74) |
| bulge_prominence_a11_just_noticeable | 0.80 | 0.07 / 0.03 | concentration R90/R50 (-0.86) |
| odd_feature_a19_ring | 0.72 | 0.07 / 0.03 | Simard bulge Re (log kpc) (-0.67) |
| bulge_prominence_a10_no_bulge | 0.71 | 0.02 / 0.04 | Simard bulge n (free n) (-0.68) |
| odd_feature_a22_irregular | 0.69 | 0.03 / 0.04 | g-r, k-corrected z=0 (-0.69) |
| bar_a06_bar | 0.69 | 0.04 / 0.03 | Simard bulge ellipticity (+0.60) |
| rounded_a17_in_between | 0.68 | 0.06 / 0.05 | concentration R90/R50 (-0.56) |
| bar_a07_no_bar | 0.67 | 0.04 / 0.04 | Simard bulge ellipticity (-0.61) |
| bulge_shape_a25_rounded | 0.66 | 0.05 / 0.06 | expAB_r (+0.47) |
| bulge_shape_a27_no_bulge | 0.65 | 0.05 / 0.06 | expAB_r (-0.51) |
| odd_feature_a20_lens_or_arc | 0.61 | 0.04 / 0.06 | concentration R90/R50 (+0.67) |
| odd_feature_a23_other | 0.60 | 0.06 / 0.05 | log sSFR (MPA-JHU) (-0.59) |
| arms_winding_a30_loose | 0.59 | 0.04 / 0.05 | concentration R90/R50 (+0.70) |
| arms_number_a37_cant_tell | 0.57 | 0.03 / 0.07 | log sSFR (MPA-JHU) (-0.45) |
| odd_feature_a38_dust_lane | 0.55 | 0.04 / 0.04 | g-r, k-corrected z=0 (+0.53) |
| odd_a15_no | 0.53 | 0.02 / 0.03 | expAB_r (-0.36) |
| odd_a14_yes | 0.52 | 0.03 / 0.03 | expAB_r (+0.36) |
| bulge_prominence_a13_dominant | 0.48 | 0.07 / 0.04 | concentration R90/R50 (+0.64) |
| odd_feature_a24_merger | 0.48 | 0.04 / 0.03 | Simard disc Rd (log kpc) (+0.48) |
| odd_feature_a21_disturbed | 0.43 | 0.05 / 0.02 | statmorph smoothness (+0.48) |
| arms_number_a32_2 | 0.42 | 0.03 / 0.07 | Simard bulge Re (log kpc) (-0.46) |
| arms_winding_a28_tight | 0.36 | 0.04 / 0.08 | expAB_r (-0.26) |
