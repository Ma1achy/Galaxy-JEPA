# Aligned re-pull — findings

Status: in progress. The corpora are re-cut with every band registered to the target
(`artifacts/sciserver_cut_v2.py`). The old corpora stay untouched as M's data. Each pre-registration
is hashed: SHA-1 over the section text, from its heading to the footer line, plus a trailing
newline. Plan: `~/.claude/plans/magical-crafting-feather.md`.

**Why.** v1 (`sciserver_cut.py`) cut g, r, i with `Cutout2D` on each band's own frame WCS. That
snaps each band to its own integer grid, so the bands sat up to ±0.5 px apart (0.20–0.22 px per
component). The misregistration became M's PC1/PC2 (`aa_findings.md` §AA3a; the INTER offsets
predict PC1/PC2 at CV R² 0.83/0.84). The float positions were discarded, so the nuisance panel
never held the pipeline's own variables.

## Preflight facts (recorded before the pilot)

- **Pretrain count.** 826,984 targets (`pretrain_all_targets.csv`); 826,968 landed. The 16
  missing are per-object `cut_one` failures that v1 swallowed, in chunks 152, 334 ×2, 336,
  339 ×2, 624, 736, 766, 790 and 805 ×6. They are not chunk deaths, as `DECISIONS.md:141` says.
  Their IDs are in `.sciserver_work/pretrain_v1_failed.csv`. Probe: 230,358 = targets = rows =
  files.
- **Probe centring.** v1 probe stamps were centred on GZ2's `ra/dec`, not PhotoObj's.
  - Over all 230,358, the GZ2 → PhotoObj separation has median 0.078″ and 99th percentile 0.23″.
  - 891 are over 1 px (0.396″), 64 over 1″, 9 over 3″, and 1 at 43.1″ (objID 1237661976553652249).
  - In the large-separation cases, GZ2's `dr8objid` names a PhotoObj source away from the
    position the volunteers classified. So v1's catalogue columns (joined by `dr8objid`) and its
    pixels (centred on GZ2's position) can describe different sources.
  - v2 centres on PhotoObj, so a stamp and its catalogue row describe one source. The separation
    is kept per object (`probe_v2_targets.csv`, `sep_arcsec`). Whether to drop the 9 over 3″ is
    left for the user at Stop 1.
- **Pretrain coordinates** are already full-precision PhotoPrimary values (13 decimals).
- **Disk.** 2.4 TiB free on the X10. v2 raw needs 0.76 TiB (1,057,326 × 792,000 B) and the fp16
  cache 0.38 TiB.

## Pilot (pre-registered)

**Question.** Does the v2 cutter register every band to the target, preserve the noise, and
leave no processing fingerprint? Is the pipeline one pipeline across both corpora? READY lets
the full pull start (after the user's go-ahead). Nothing here trains anything.

**Code, frozen by hash.**
- Cutter `sciserver_cut_v2.py` sha256 `327c4213a0008514…`.
- Analysis `repull_pilot.py` sha256 `71ff230ba65c766c…`.
- Targets `repull_pilot_all_targets.csv` sha256 `74e914c3f62c8fe9…`.
- The SciServer job is submitted after this hash. No pilot pixel exists before it.

**Cutter convention.**
- Per band: `(x_b, y_b) = skycoord_to_pixel(…, origin=0, mode="all")`, 0-based, with pixel
  centres on integers.
- Padded cut of 321 px (256 + 2·32 + 1; odd, so the phase ramp has no unpaired Nyquist bin and is
  exactly unitary) at `o_b = floor(x_b − 159)`.
- Off-frame pixels are filled with the band median.
- Fourier shift `s_b = 159.5 − (x_b − o_b) ∈ (−0.5, 0.5]`, applied to every band, r included.
- The mask is shifted by the same operator and thresholded at 0.5. Beside a frame edge it is
  eroded by 2 px (the sinc tails mix the fill into the nearest real pixels).
- Crop the centre. **The target lands on (127.5, 127.5)**, the stamp's geometric centre.
- Pad = the union of the bands' invalid masks, set to exactly 0 in all three.
- Logged per object and band: `x, y, ox, oy, sx, sy`, valid fraction, distance to the frame edge,
  frame header SHA-256, and v1's own `Cutout2D` origin and target position (`v1_ox, v1_oy,
  v1_relx, v1_rely`). Failures are logged with their exception.

**Sample: 776 objects, one SciServer job.**

| Group | Count |
|---|---|
| Random, camcol-stratified (42/41 per camcol), probe | 250 |
| Random, camcol-stratified (42/41 per camcol), pretrain | 250 |
| Edge (target within 100 px of a frame edge), probe | 30 |
| Edge (target within 100 px of a frame edge), pretrain | 30 |
| AA3a galaxies (probe) | 200 |
| v1's pretrain failures | 16 |

Four variants are cut per object:
- `v2`;
- `nomargin` (no pad; plant for 3);
- `bilinear` (bilinear shift; plant for 2 and 3b);
- `margin64` (64 px pad; the border-convergence reference).

v1 stamps come from disk. "Paired" means an object with a v1 stamp.

**Checks.** "Depends" means Benjamini–Yekutieli-significant (q 0.05, within the check's family)
AND |Spearman ρ| ≥ 0.1. If a test is significant but |ρ| < 0.1, it is named SIGNIFICANT BELOW BAR
and is not a defect.

| # | Check | Passes iff | In-run plant (must fire) |
|---|---|---|---|
| 1a | Recorded registration | For every object and band, the logged `(x−ox)+sx−32` = 127.5 ± 0.01 px, both axes | r's logged shift perturbed by (0.3, −0.2) is recovered ± 0.01 |
| 1b | Empirical registration (field stars; windowed centroids g−r, i−r) | Median offset per component \|·\| < 0.03 px, AND slope of offset on v1's predicted misregistration \|β\| < 0.1 | The same slope on v1 stamps > 0.7 in every component |
| 1c | Residual independence | Per-stamp star offset does not depend on frame x, frame y, edge distance, camcol, run, or that band's shift (24 tests) | v1's per-stamp offsets depend on v1's predicted misregistration (all 4) |
| 2 | Noise and resampling fingerprint | Lag-1 autocorrelation and top-octave noise share do not depend on h_b = \|sx_b\|+\|sy_b\| (6 tests); median v2/v1 sky variance ∈ [0.98, 1.02], high-frequency ratio ∈ [0.95, 1.05], \|Δ lag-1\| < 0.02, per band | The bilinear copy's lag-1 depends on h_b (bar 0.2) in every band |
| 3 | Edges | Median rms(v2 − margin64) in the outer 4 px ≤ 0.1σ per band, AND median sky variance 1–4 px inside the pad / far sky ∈ [0.9, 1.1] per band | nomargin's border rms > 0.1σ, AND a copy with a planted ±0.6σ damped ring on the pad boundary reads > 1.1, per band |
| 3b | Star ringing | Per star, the paired radial residual profile (v2 − the same star in v1, flux-normalised, 1 px bins 0–8 px) does not depend on h_b at any radius (24 tests) | The bilinear copy's residual depends on h_b (bar 0.2) at some radius in every band |
| 4 | Old vs new | Median rms(v2 − v1 Fourier-shifted by its recorded offset), central 176², outside pads, ≤ 0.1σ per band | v1 with g moved a further 0.5 px: median rms > 0.2σ |
| 5 | Centring | **Decisive:** KS on \|Δ\| of the r-core centroid (peak of the σ = 1.3 px-smoothed r image, 3×3 quadratic) from (127.5, 127.5), probe vs pretrain random groups, p > 0.01. Median \|Δ\| is reported only: this approximates SDSS's convention and is not it, and asymmetric galaxies' cores legitimately sit off the catalogue position | Probe r moved 0.5 px in a random direction: KS p < 0.01 |
| 6 | One pipeline | Checks 1a–4 give the same pass/fail computed on each corpus alone | Covered by 5's plant |

**Minimums.**
- 200 stars (1b, 1c) and 200 star pairs (3b).
- 30 edge objects with a pad (3).
- 300 paired objects (2, 4).

**States and precedence (D27).**
- **INCONCLUSIVE** (naming the checks whose plant did not fire) takes precedence over everything:
  an instrument that cannot see a planted defect cannot clear the data.
- Then **DEFECT** (naming the failed checks, and any check 6 difference). A demonstrated failure
  is informative even where another check is thin.
- Then **INSUFFICIENT** (naming the thin checks).
- Then **READY**.
- Also reported, without a state:
  - per-band spreads;
  - median |Δ| per corpus;
  - the v2 cut rate (gal/s) and the projected full-pull wall-clock at waves of 6;
  - each of the 16 v1 failures' outcome under v2, with its exception;
  - the probe objects over 3″.

**D28, shown before this hash** (`repull_pilot.py plant`, `out/repull_pilot_plant.json`).
Synthetic frames went through the real `cut_one`, all four variants, and the identical checks:
360 objects with stars, a galaxy, per-band WCS offsets and 0.02 px per-star astrometric scatter.
- **Aligned → READY.** Every plant fired:
  - 1b: v1 slope 1.00, v2 slope 0.001;
  - 2: bilinear ρ ≈ 0.99;
  - 3: v2 border 0.04σ vs nomargin 0.33σ; pad ring 0.98–1.00 vs plant 1.15–1.17;
  - 3b: bilinear core ρ ≈ −0.5;
  - 4: 0.05σ vs plant 1.06σ;
  - 5: plant p ≈ 1e-84.
- **A synthetic "v2" with g misregistered 0.3 px in the frame → DEFECT (1b)**, median g−r = 0.299 px.
- **An earlier draft of 3b** used lobe depth over peak, unpaired. It reached **INCONCLUSIVE**: the
  bilinear plant only reached ρ ≈ 0.1, swamped by star-to-star seeing. That is why 3b is paired
  against v1.

**Changes from the approved plan, made before this hash.**
- **Check 3.** A Fourier shift is unitary on sky noise, so "nomargin rings at the border" cannot
  show on a variance statistic. The test became convergence in the margin (v2 vs margin64),
  with the pad-boundary variance plus a planted ring.
- **Check 3b** is the paired residual profile, per the amendment's "radial residual profile".
  Its plant is the bilinear copy (a shift-dependent profile change).
- **Check 1b's** "spread ≤ 1.2× photon noise" became a slope on v1's predicted misregistration.
  The spread is dominated by per-band astrometric calibration, which has no clean expectation; the
  slope isolates the defect.
- **Check 5's plant** is a planted 0.5 px move. v1 probe vs v1 pretrain differ by about
  0.26 px rms, too little for a guaranteed KS firing at n = 250.
- **Magnitude gate.** A significant ρ below 0.1 is named, not a defect.

*Pilot pre-registration ends here: the Pilot section above (108 lines from "## Pilot"), SHA-1 `1f6ffd513232c679bee0f5e6d0a52492b62b3309`.*

## Layer 3a — when does M's band axis appear? (pre-registered)

**Question.** AA3a found M's PC1/PC2 track band misregistration at the final encoder (INTER →
PC1/PC2 CV R² 0.83/0.84). The re-pull's Layer 3 needs to know when that axis first appears during
training. If it appears early, short smoke trainings on v1 and on v2 data can test the fix before
any full retrain: v1 must show the axis, v2 must not.

**Method** (`artifacts/m_band_axis.py`, sha256 `2ab9c2011527ba2b…`; `--scan`).
- Every checkpoint in `runs/m/checkpoints/` (20 entries, steps 6,331 to 101,308), then the final
  `runs/m/encoder.pt`.
- Each encoder is frozen through disk (`frozen_from`).
- It embeds a fixed 5,000-galaxy PCA sample (a deterministic stride of the probe-train ids) and
  AA3a's 2,000.
- PCA is fitted on the sample, the 2,000 are projected, and the top 10 PCs are kept.
- Statistic: AA3a's `cv_r2` of each PC on the INTER offsets it measured (rank-normal, 5-fold).
  "Best" is the maximum over the 10 PCs.
- Offsets are AA3a's pixel-measured ones. Exact per-band fractional positions from frame headers
  (the A9 header job, limited to these 2,000) are added descriptively when that job has run.

**States** (first match; D27).
- **SCAN DEFECT** if the final encoder does not reproduce AA3a (|R²(PC1) − 0.83| or
  |R²(PC2) − 0.84| > 0.10, with PCA fitted on this 5,000-sample and not AA3a's 40k). Then nothing
  else is read.
- **EARLY:** best ≥ 0.30 at a checkpoint ≤ step 12,664 (the first two).
  → Smoke trainings to that step, on v1 and on v2, after Stop 2.
- **LATE:** best first ≥ 0.30 at a later checkpoint.
  → No smoke training. The test waits for the aligned encoder's full run.
- **NEVER:** never ≥ 0.30, the final encoder included. That contradicts AA3a and is read as
  SCAN DEFECT.
- Reported without a state: the PC1, PC2 and best trajectories, and the index of the best PC.

**D28 before this hash** (`--planted`, `out/m_band_axis_planted.json`).
- A planted axis on PC1 is recovered: R² 0.997 / 0.50 / 0.31 / 0.09 at noise 0 / 1 / 1.5 / 3.
- Buried on PC7, it is found there (0.51, PC7).
- With no axis, best-of-10 is −0.001: the null does not saturate.
- EARLY, LATE and NEVER are all reached from constructed trajectories.

**Runs** after the BB2 pilot (one heavy process at a time).

*Layer 3a pre-registration ends here: the Layer 3a section above (38 lines from "## Layer 3a"), SHA-1 `4fadea31096260756d17e015cfe12924bf8c77ae`.*

### Layer 3a result — **LATE** (`out/m_band_axis.json`)

The final encoder reproduces AA3a: INTER → PC1 0.834, PC2 0.834 (AA3a 0.83/0.84, with PCA on this
5,000-sample). So the state is readable.

| Step | PC1 | PC2 | Best of top 10 |
|---|---|---|---|
| 6,331 – 25,327 (6 checkpoints) | ≈ −0.01 | ≈ −0.01 | ≤ 0.004 |
| 31,655 | 0.688 | 0.003 | 0.688 (PC1) |
| 37,986 | 0.859 | −0.005 | 0.859 (PC1) |
| 44,317 | 0.840 | −0.008 | 0.870 (PC3) |
| 50,648 / 50,654 | 0.140 / −0.002 | 0.646 / 0.810 | PC2 |
| 56,979 – 94,965 | 0.78 – 0.86 | 0.00 – 0.17 | PC1 |
| 101,296 / 101,308 | 0.836 / 0.834 | 0.824 / 0.834 | PC1 |

- **The axis is absent** (R² ≈ 0 on every one of the top 10 PCs) for the first quarter of training.
  It **appears abruptly between steps 25,327 and 31,655** at R² ≈ 0.7 and holds ~0.8–0.87 from
  then on.
- **For most of training, one leading PC carries it** (PC1, briefly PC2 or PC3). The second
  component that makes (PC1, PC2) the two-axis "pose" pair AA3a found reaches R² ≈ 0.83 **only at
  the last checkpoints** (step 101,296 onward).
- **Per the pre-registration: LATE → no smoke trainings.** The v1-vs-v2 test waits for the aligned
  encoder's full run (Layer 3 on the final aligned encoder). A smoke run long enough to show the
  axis would need more than 31k steps (~5.5 h per arm at M's rate); that is not a smoke run.
- The scan used AA3a's 2,000 as measured, including …3595058 (one of the 9 excluded below). This
  is an M-only scan, not an M-vs-v2 comparison.

## Probe exclusion — GZ2 position more than 3″ from PhotoObj (user, 2026-09-25)

Volunteers classified whatever sat at GZ2's position. So re-centring on PhotoObj could put a
different object under their votes; the 43″ case is a different source outright.
- **These 9 are dropped from probe_v2 (230,349 targets) and from every M-vs-v2 comparison**, so
  the sets stay one-to-one.
- The 55 between 1″ and 3″ are kept and flagged in `cut_log.csv` (`gz2_sep_arcsec`,
  `gz2_sep_flag = 1-3arcsec`).
- None of the 9 is in the pilot. One (…3595058) is in AA3a's 2,000: the header job logs it, and it
  is excluded from M-vs-v2 comparisons.
- List: `.sciserver_work/probe_v2_excluded_gz2_sep.csv`.

| objID | GZ2 → PhotoObj | PhotoObj ra, dec | GZ2 ra, dec |
|---|---|---|---|
| 1237661976553652249 | 43.14″ | 199.877073107256, 7.893785734356 | 199.8816, 7.882674 |
| 1237655472895099027 | 8.21″ | 249.469288345408, 39.248597317510 | 249.4718, 39.24979 |
| 1237651273510813752 | 7.18″ | 182.964132373112, 67.928288676307 | 182.9598, 67.92944 |
| 1237662264854904841 | 6.10″ | 217.844222619509, 9.276042880989 | 217.8453, 9.277361 |
| 1237662528990674970 | 4.67″ | 210.081387630479, 12.957535946330 | 210.0826, 12.957 |
| 1237658423543595058 | 4.51″ | 136.560727516127, 4.975111418485 | 136.5613, 4.976228 |
| 1237662263251042419 | 3.71″ | 233.274712607773, 6.423783396951 | 233.274, 6.424534 |
| 1237658802034900998 | 3.62″ | 172.709229452376, 54.393779581024 | 172.7103, 54.39457 |
| 1237665329849958421 | 3.46″ | 178.548614924114, 31.933648555893 | 178.5487, 31.93269 |

## Pilot result (Stop 1, 2026-09-25)

Job 486109: 776 targets, **760 cut, 16 failed**. The 16 are exactly v1's 16 silent pretrain
failures, each `NoOverlapError` (the target lies off at least one band's frame). Nothing new failed.
Code at run time matched the hashed SHAs: analysis `71ff230b`, cutter `327c4213`. The run is in
`out/repull_pilot.json`, copied to `out/repull_pilot_hashed_run.json`.

### Hashed state — **INCONCLUSIVE (plant did not fire: 1b)**

| check | pass | plant fired | reading |
|---|---|---|---|
| 1a recorded registration | ✓ (max dev 0.000 px) | ✓ | |
| 1b empirical registration | ✗ | **✗** | v1 slopes 0.39 / 0.02 / 0.66 / 0.20; plant needs > 0.7 |
| 1c residual independence | ✗ (camcol: g−r x ρ 0.35, i−r x ρ 0.21) | ✓ | |
| 2 noise fingerprint | ✗ (pooled medians NaN) | ✓ | check 6 flags 2 |
| 3 edges | ✓ (border 0.04σ vs nomargin 0.33σ) | ✓ | |
| 3b star ringing | ✓ | ✓ | |
| 4 old vs new | ✓ (0.053σ) | ✓ | |
| 5 centring KS | ✗ (p = 3.8e−5; median \|Δ\| probe 0.132, pretrain 0.172 px) | ✓ (p 1.8e−94) | |

### Why 1b's plant did not fire — a logging fault in the cutter, not in the v2 pixels

The cutter logs v1's origin as `Cutout2D.origin_original`, the origin of the overlap *in the
frame*. At a frame edge (`mode="partial"`) that clamps to 0: e.g. `g_v1_oy = 0`, `rel_y = 74.9`
where the true in-stamp position is ≈ 128. So v1's "predicted misregistration" was wrong for every
edge-touching stamp (231 band-axis origins in the pilot).

The true virtual origin is `ceil(v1x − 128)`, with `v1x` = the logged origin + the logged rel. This
is exact against astropy's `input_position_cutout` on 20,000 random positions, clipped ones
included. So it is recoverable from existing logs without re-cutting. The same fault is in the
AA3a header job (`raw/aa3a_hdr`).

### EXPLORATORY — the hashed analysis on the corrected log (`out/repull_pilot_explore_v1fix.json`)

Not the pre-registered result; the state it reaches is **DEFECT (1b, 1c, 2, 5; 6:2)**, with every
plant firing.

- **The cutter's misregistration is gone.**
  - Star-offset slope on v1's predicted snapping: **v1 0.99–1.03, v2 0.001–0.026**.
  - 1c's plant reaches ρ 0.94–0.97.
- **1b / 1c — the residual is SDSS's, and both cutters carry it identically.**
  - v2's star offsets equal v1's star offsets minus the predicted snapping, to about 0.01 px.
  - In the median: g−r y −0.049 against −0.050.
  - Per camcol, g−r x runs −0.080 / −0.027 / −0.045 / +0.008 / +0.052 / +0.050 (v2) against
    −0.067 / −0.014 / −0.045 / +0.004 / +0.050 / +0.052 (v1 residual).
  - So the frame WCSs disagree between bands by about 0.05 px (0.02″), per camcol. Star colour (DCR)
    could contribute to the y term.
  - v2 applies the headers exactly. The bars (|median| < 0.03, no camcol dependence) assumed the
    WCS is exact between bands.
- **2 — an analysis defect.**
  - 5 of 760 paired stamps (all pretrain) give NaN, and `np.median` carries it into the pooled and
    pretrain values.
  - Without them: variance ratio 1.0003 / 1.0005 / 1.0007, top-octave 1.0008 / 1.0009 / 1.0005,
    Δlag-1 −0.0006 / −0.0006 / −0.0004.
  - No |s| dependence survives BY.
- **5 — the corpus difference follows SNR.**
  - Random groups: probe median r = 16.47 / SNR 243; pretrain 17.92 / SNR 93.
  - |Δ| ~ SNR ρ = −0.28 (p ≈ 0); ~ petroRad ρ = −0.01.
  - Nearest-neighbour matched on (mag, log size): KS p = 0.26, but only 59 pairs, and the medians
    still read 0.123 against 0.148.
  - Suggestive, underpowered.

### Header job, corrected (AA3a's 2,000, the 1 excluded object dropped)

Exact v1 g−r / i−r offsets:
- sd 0.396–0.408 px, the triangular sd of two independent ±0.5 px snaps (0.408);
- against AA3a's measured light-centroid offsets: ρ **0.87 / 0.88 / 0.91 / 0.90**, slope 0.65–0.67;
- the uncorrected log gave ρ 0.79–0.88.

### Rate

| job | cut rate | objects | notes |
|---|---|---|---|
| pilot (4 variants) | 0.89 gal/s | scattered | ~3 fresh bz2 frames per object |
| header-only | 1.30 gal/s | scattered | |
| contiguous timing (job 486112; first 1,000 pretrain_v2 targets, v2 only) | **4.65 gal/s**, 215 s, 0 failed | contiguous | |

- The contiguous rate is at the top of v1's 2.6–4.7.
- Tarballs are 714 MB per 1,000, the same as v1, so the pull stays download-bound like v1's.
- Projected at waves of 6: pretrain ≈ 4.1 d (v1's measured), probe ≈ 1.1 d (scaled by chunk count).
  **About 5.2 days together**, with a token refresh every 24 h.

## Pilot amendment A — post hoc (D27; user decisions at Stop 1, 2026-09-25)

**Post hoc.** Checks 1b, 1c and 2 below were rewritten *after* the hashed pilot result was seen,
and they are re-run on the same pilot stamps. They apply to this re-pull and to future runs. They
are labelled exploratory beside the hashed record, which stays **INCONCLUSIVE (1b)**. Check 5′
alone runs on fresh data cut after this hash.

**Code.**
- Cutter `0dc4aeedba67a234`. It logs v1's virtual origin and `input_position_cutout`; the stamps
  are unchanged. A unit-test corner case, target at frame (20.4, 15.7), asserts origin < 0 and rel
  in (127, 128].
- Analysis `repull_pilot.py` `4d9d9f4e69ede751`: `analyse_amended`; the hashed functions are
  unchanged.
- Log correction `repull_fix_v1log.py` `1d847c5e50957ae1`, applied in post to `data/repull_pilot`
  (231 origins) and `raw/aa3a_hdr` (417). The originals are kept as `cut_log.uncorrected.csv`.
- Fresh sample `.sciserver_work/repull_check5_all_targets.csv` `74b7d48a14aef1c4`.

**1b′ — registration.**
- Per v2 star, the same star in v1 at r's in-stamp offset, both through `star_offsets`:
  d = off_v2 − (off_v1 − v1_pred).
- READY iff the pooled median d is within ±0.02 px (TOL_PAIRED) in each of g−r and i−r, x and y,
  **and** the hashed slope bar holds: |slope of v2 on v1_pred| < 0.1.
- The slope stays because a median-only bar would pass a v2 that kept v1's snapping (the snapping
  is zero-mean).
- Plants:
  - the hashed v1-slope plant, needing slope > 0.7;
  - v2's g moved +0.05 px, which must give |pooled median d(g−r x)| > 0.02.
- INSUFFICIENT below 200 star pairs.

**1c′ — independence.**
- READY iff the median d is within ±0.02 px in every camcol × component. This replaces the camcol
  Spearman test.
- The hashed tests on x, y, edge distance, run and |s| are kept as they were.
- Plant: camcol 3's stamps alone take the g-moved d. It must flag camcol 3 (g−r x) and no other
  camcol.

**2′ — noise fingerprint.**
- `nanmedian`, with each dropped stamp named along with the quantity and band that went NaN.
- The bars and plant are as hashed.

**5′ — centring, decisive (replaces 5).**
- Fresh data: 250 random probe objects outside the pilot and outside the 1″ flag, all within
  pretrain's selection box (petroRad_r ≥ 5″, 14 ≤ r ≤ 19; 25% of the probe lies below 5″ with no
  pretrain counterpart).
- The pretrain side: 250 objects outside the pilot, nearest-neighbour matched without replacement on
  standardised (modelMag_r, log petroRad_r, log snr_r). Balance: SMD 0.001 / 0.005 / 0.0005.
- READY iff KS(|Δ| probe, |Δ| matched pretrain) p > 0.01. |Δ| is the r core centroid − (127.5,
  127.5), as hashed.
- INSUFFICIENT, with the KS not read, if either side has fewer than 200 or any |SMD| ≥ 0.1. It takes
  precedence over the KS: an unbalanced KS is neither a pass nor a DEFECT.
- Plant: the hashed 0.5 px probe move must give p < 0.01.

**Flag exercise.** The same job cuts the 55 probe objects at 1″–3″ (group `sep_flag`). Reported,
not gating:
- all 55 are cut, with `gz2_sep_flag = 1-3arcsec`;
- no other row carries a flag;
- every v1 rel in the fresh log lies in (127, 128].

**Check 6** is unchanged: each check's state is the same in each corpus, 5′ excluded. **Precedence**
is unchanged: INCONCLUSIVE > DEFECT > INSUFFICIENT > READY.

**D28 before this hash** (`out/repull_pilot_amendA_dry.json`, the pilot with 5′'s code path run on
the pilot's own random groups, plumbing only):
- every plant fires;
- 1b′ plant d = 0.0500;
- 1c′ plant flags `camcol3:gr_dx` only;
- 5′ plant p = 1.8e−94;
- the unbalanced dry-run sample (SMD −1.75 / 2.01) reads INSUFFICIENT.

The real paired d is 0.0000 (4 d.p.) pooled and in every camcol, over 3,501 pairs. At star positions
v2 is an exact translate of v1, so the 0.02 bar is far from binding. That is recorded as it is.

**Check 2's 5 dropped stamps** (all pretrain):

| objID | group | band(s) |
|---|---|---|
| 1237649768110621418 | random | g, r |
| 1237649770795958852 | random | r |
| 1237653742021116162 | edge | r |
| 1237654896833069438 | random | r |
| 1237672003693249280 | random | i |

- Each band carries a background pedestal 1.5–9σ above zero across the whole stamp (1237649768110621418
  r: median 1.05 against σ 0.11). That is a sky-subtraction residual or scattered light.
- `sky_mask` thresholds at +1σ above *zero*, not above the local sky, so it finds no sky pixels. The v1
  stamp gives the identical empty mask, so this is not the cut.

**If READY:** the full pull, probe_v2 then pretrain_v2, at waves of 6.

**Staged fallback for camcol (recorded now).** If the Layer 2 audit reads LEAK on camcol:
1. The next step is a per-camcol, per-band correction table, measured on field stars and fed into
   the Fourier shift. Not per-frame star registration.
2. Only if that fails does a new cutter design come up.

*Amendment A pre-registration ends here: the section above (93 lines from "## Pilot amendment A"), SHA-1 `a0f56905deb71246bf489c4965116e96a5075f49`.*

### Amendment A result — **READY** (`out/repull_pilot_amendA.json`)

Code at run time: analysis `4d9d9f4e`, cutter `0dc4aeed`. Fresh job 486114: 555 cut, 0 failed.

| check | pass | plant fired | reading |
|---|---|---|---|
| 1a | ✓ | ✓ | 0.000 px |
| 1b′ | ✓ | ✓ | paired d 0.0000 over 3,501 pairs; slopes 0.026 / −0.010 / 0.001 / 0.020; plant 0.0500 |
| 1c′ | ✓ | ✓ | every camcol 0.0000; plant flags `camcol3:gr_dx` only |
| 2′ | ✓ | ✓ | 5 dropped (above); variance 1.0003–1.0007, top octave 1.0005–1.0010, Δlag −0.0006 |
| 3, 3b, 4 | ✓ | ✓ | as in the hashed run |
| 5′ | ✓ | ✓ | KS p = **0.94**; median \|Δ\| 0.132 probe vs 0.128 matched pretrain; n 250/250; plant p 2.9e−124 |
| 6 | ✓ | | the same state in each corpus |

- **Flag exercise:**
  - all 55 of the 1″–3″ objects were cut, each with `gz2_sep_flag = 1-3arcsec`;
  - 0 flags on any other row;
  - every v1 rel in the fixed cutter's log lies in (127, 128].
- **The unmatched check 5's corpus difference was SNR, not the pipeline.** Matched on magnitude, size
  and SNR, the centring residuals are indistinguishable.
- **Note.** Pretrain's own selection also caps petroRad_r ≤ 25″, which the support filter did not
  apply. The balance held regardless (every |SMD| ≤ 0.005).
