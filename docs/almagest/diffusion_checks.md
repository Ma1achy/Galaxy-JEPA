# Almagest diffusion — normalisation and native-crop checks (21 Nov 2026)

VISUALISATION ONLY. No claim, nothing in the science record. Branch `almagest-geometry`. No training.
Code: `artifacts/almagest_diffusion_checks.py` (`norm`, `crop`); it reads the pilot's banks and checkpoint
and changes neither. Numbers: `diffusion_checks.json`. Arrays: `runs/almagest/diffusion/checks/`.

Run conditions: load 6.0–6.4, from other sessions. Swap was near full (about 700 MB free). The first
`norm` attempt was stopped by my swap guard at 134 MB free. It was rerun with MPS batches cut to 16 for
sampling and 32 for embedding; the data stages peak at 0.76 GB RSS.

## 1. Normalisation sanity

The U-Net is fed y = asinh(x) / 0.8, where x is the normalised stamp (A1's input) pooled 4 × 4 to 64 px.

- Background is the pixels outside the 2.5 × R_petro box (`bbox.petrosian_box`, k = 2.5). 11 of 4,096
  galaxies have no background pixels.
- The samples have no R_petro, so their background is the outer 8-px ring. That ring lies outside the
  box for 97% of the 100k, and the inputs' ring statistics match their outside-box ones.

Per channel g / r / i, in model units y. Inputs are a random 4,096 of the 100k; a batch of 32 gives the
same picture (`inputs_batch_32`).

| | inputs | unconditional samples (w = 0, 64) |
|---|---|---|
| whole-image mean | −0.05 / −0.06 / −0.06 | **+0.26 / +0.71 / +0.67** |
| whole-image SD | **0.38 / 0.44 / 0.47** | 0.85 / 1.21 / 1.44 |
| background: median level per image, IQR across images | 0.04 / 0.05 / 0.07 | **0.61 / 1.35 / 1.49** |
| background: per-image pixel noise (1.4826·MAD) | 0.13 / 0.16 / 0.20 | 0.22 / 0.29 / 0.37 |
| images with a channel's ring level > 1 own-noise SD from the inputs' median | 1.6% | **80%** |

In x units the inputs are mean 0.00 with SD 0.79–0.91 per channel. Their background noise is
0.11–0.16 and the level IQR 0.03–0.06.

**Reading.**

- **The inputs are roughly zero-mean (−0.05) and their backgrounds are clean.** Levels are tight across
  images, and only 1.6% have an offset above 1 noise SD. The casts do not come from the input's
  background.
- **The inputs are not unit-variance.** Their SD is 0.38–0.47 in model units. This was set deliberately
  by the squash (the code's comment says SD ≈ 0.42), but the cosine noise schedule assumes data of about
  unit variance. At SD 0.44 every timestep's signal-to-noise is about 0.19× the schedule's (log-SNR
  shifted by −1.6).
- **The per-channel sky level is a tiny signal.** It varies by only about 0.05 across images, against
  unit diffusion noise, so it is set only in the last steps.
- **The samples err in one direction:** too bright (+0.3 to +0.7), with the error growing g < r < i.
  That matches the pilot's conditional samples. The pilot's x0 clamp, (−2, 6) in y, is asymmetric and
  lets positive x0 predictions through early.

  Candidate causes, none tested and nothing fixed:
  - the variance mismatch;
  - the weak per-channel level signal;
  - the asymmetric clamp.

## 2. Ceiling of a native 128-px crop (no resampling)

**Format.** The 128 × 128 centre crop of the 256-px stamp at native 0.396″/px. For A1, it is padded back
to 256 px with Gaussian sky per band, matched to that stamp's background.

- The background is measured outside the 2.5 R box, 3σ-clipped against field stars. All 256 test
  galaxies had enough pixels.
- Variant: the sky measured from the crop's own outer 8 px, which is all a generated crop would have.

**Fit.** The 2.5 × R_petro box fits inside the crop (half-width ≤ 64 px) for **88.7% of the 100k**
(no R_petro fallbacks). Box half-width percentiles 5 / 50 / 95 are 32 / 39 / 77 px. Of the 256 test
galaxies, 233 fit.

**Re-embedding, the pilot's 256 test galaxies** (the pilot's eval draw). Native embeddings are recomputed
and match the bank exactly. "Centred" subtracts the train bank's mean. The rank is the source's rank
among the 256 native embeddings, as in the pilot; chance is 128.5.

| | cos to own, raw | centred | rank median | rank mean | top-1 | top-5 | random pairs, centred |
|---|---|---|---|---|---|---|---|
| **128 crop, padded (sky from full stamp)** | 0.92 | **0.79** | **1** | 18.7 | **50%** | 68% | 0.17 |
| 128 crop, padded (sky from crop ring) | 0.90 | 0.75 | 2 | 20.1 | 46% | 64% | 0.18 |
| 64-px round trip (pilot ceiling) | 0.53 | 0.12 | 60.5 | 87.6 | 6.6% | 17% | 0.89 |
| native, for scale | 1 | 1 | 1 | 1 | 100% | 100% | −0.03 |

The 64-px row reproduces the pilot's ceiling (0.12, rank 60.5).

- The padded crop keeps galaxies apart: random pairs sit at 0.17, against 0.89 after the 64-px round
  trip, which collapsed them together.
- By box fit: 0.78 centred where the box fits (n = 233), 0.88 where it does not (n = 23). Large galaxies
  fill the crop, so less of the image is replaced.

**What the format loses (by eye, `diffusion_checks_padded_crops.png`: native | padded).** Every field
source beyond ±25″ is gone: stars, neighbours, and a neighbour cut at the crop edge. That is the likely
source of most of the remaining gap (rank mean 18.7, top-1 50%), but it is untested.

**Planted pure background** (the pass rule was declared in the code before the run).

- Setup: 64 Gaussian-sky stamps, each with one test stamp's measured sky. Crop and pad each one, and
  compare with an independent full redraw.
- Result: cos(original, padded) = **0.994** against cos(original, redraw) = 0.990 (centred medians).
  **Pass.**
- **It can fail.** Padding with the wrong sky (level + 1 SD, SD × 2) drops the cosine to **−0.19**. The
  matched pad is as good as a fresh draw, and a mismatched one is caught.

## For the next brief (not decided here)

- The padded 128 crop ranks the source **first in median (rank 1, top-1 50%)**, against 60.5 at 64 px.
  It meets the condition set for a 128-px crop model.
- In a crop model, the sky for padding has to come from the crop itself (the crop-ring row: rank 2,
  top-1 46%). Field sources beyond ±25″ cannot be rendered.
- Check 1's candidates (input variance 0.44 against the schedule, the weak per-channel level signal, the
  asymmetric clamp) are worth a decision before rented compute, since they would carry into a 128-px run.
