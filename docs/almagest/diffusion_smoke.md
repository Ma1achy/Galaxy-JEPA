# Almagest diffusion — 64-px smoke run of three input fixes (21 Nov 2026)

VISUALISATION ONLY. No claim, nothing in the science record. Branch `almagest-geometry`. Stats only;
no re-embedding judgement at 64 px. Code: `artifacts/almagest_diffusion_smoke.py`. Numbers:
`diffusion_smoke.json`. Runs and diagnostics: `runs/almagest/diffusion/smoke_{full,nosky}/`.

**Result: the smoke does not pass.** With all three fixes, the unconditional samples still differ from the
inputs in level, spread and background. Two diagnostics locate the causes:
- **The percentile clamp (fix 3) causes the positive level bias.** Its upper limit (+8 to +11) lies far
  above its lower one (about −1.5).
- **The remaining excess background spread** is the network at 2,000 steps; the sampler is not at fault.

## Set-up

- The pilot's U-Net (33.1M), schedule, loss and CFG dropout are imported unchanged. Training starts from
  scratch: 2,000 steps at batch 32, AdamW 2e-4 with EMA, about 0.89 h per arm at 0.62–0.64 steps/s.
- **Fixes**, each switchable:
  1. **Unit variance:** per-channel mean and SD from a frozen draw of 8,192 train galaxies, stored in the
     checkpoint.
  2. **Sky subtraction:** per image and band, the 3σ-clipped median of the outer 8-px ring of the
     64-px stamp, subtracted before the squash and scaling. The 100k levels are stored
     (`smoke_sky_levels.npz`); renders add back the median sky.
  3. **x0 clamp:** at the training data's per-channel 0.1 / 99.9 percentiles after fixes 1 and 2. For the
     full arm that is [−1.54, 11.27] / [−1.50, 9.66] / [−1.70, 8.48] (g / r / i).
- **Arms:** `full` (1 + 2 + 3) and `nosky` (1 + 3, the ablation).
- **Evaluation:** 64 unconditional samples (w = 0, DDIM 100 steps) against 64 train inputs, in a fresh
  process. Background is the outer 8-px ring. Offset = a channel's ring level more than 1 own-noise SD
  from the inputs' median ring level.
- **Running it:** the swap guard (300 MB) killed the first nosky attempt at step about 1,600, from other
  sessions' memory, and an earlier probe at sampling. The trainer now checkpoints every 500 steps and
  resumes. The rerun completed without a kill, and the full arm ran clean. Nothing else ran beside either.

## Results (g / r / i)

**Model units** (what the network sees; inputs are about zero-mean and unit-variance by construction):

| | mean | SD | ring level IQR | ring noise | offset > 1 SD |
|---|---|---|---|---|---|
| inputs, full arm's transform | +0.03 / +0.04 / +0.02 | 1.05 / 1.05 / 1.03 | 0.019 / 0.019 / 0.023 | 0.36 / 0.38 / 0.47 | 0% |
| **samples, full (1 + 2 + 3)** | **+1.16 / +1.29 / +1.30** | 1.78 / 1.71 / 1.70 | **0.82 / 0.69 / 1.05** | 0.63 / 0.70 / 0.81 | **77%** |
| inputs, nosky arm's transform | −0.01 / −0.01 / +0.02 | 1.02 / 0.98 / 1.04 | 0.092 / 0.092 / 0.137 | 0.34 / 0.34 / 0.42 | 2% |
| **samples, nosky (1 + 3)** | **+1.45 / +1.82 / +1.34** | 1.83 / 1.83 / 1.98 | **1.47 / 1.95 / 1.91** | 0.57 / 0.53 / 0.63 | **92%** |

**Rendered to normalised-stamp units x** (inputs as stored; the full arm adds back the median sky):

| | mean | SD | ring level IQR | ring noise | offset > 1 SD |
|---|---|---|---|---|---|
| inputs | −0.00 / −0.00 / +0.01 | 0.87 / 0.84 / 0.77 | 0.028 / 0.033 / 0.052 | 0.10 / 0.12 / 0.16 | 2% |
| samples, full | +0.42 / +0.53 / +0.58 | 1.16 / 1.24 / 1.17 | 0.24 / 0.23 / 0.39 | 0.19 / 0.24 / 0.31 | 73% |
| samples, nosky | +0.55 / +0.85 / +0.68 | 1.21 / 1.34 / 1.28 | 0.46 / 0.77 / 0.75 | 0.19 / 0.22 / 0.28 | 92% |
| pilot (no fixes, 6,413 steps; check 1) | +0.47 / +1.24 / +1.46 | 2.59 / 3.69 / 4.19 | 0.50 / 1.25 / 1.32 | 0.18 / 0.31 / 0.44 | 80% |

**Does the sky fix carry most of it?** It carries some. Against nosky, it roughly halves the rendered
background spread (IQR 0.24–0.39 against 0.46–0.77) and cuts offsets from 92% to 77%. It leaves the level
bias (+0.4 to +0.6 in x) and the excess noise untouched. Against the pilot, the fixed model is better in SD
and r/i background, though the step counts differ (2,000 against 6,413). No arm is near the inputs.

## Diagnostics (on the full arm's checkpoint; post hoc)

1. **Conditional samples (w = 1)**, on the 64 evaluation galaxies' own embeddings, look the same: mean
   +1.14 / +1.26 / +1.31, ring IQR 0.73 / 0.66 / 1.04, 78% offset. So the null branch (10% of batches)
   is not the cause, and the conditioning makes no difference to these statistics at 2,000 steps.
2. **The sampler is correct (planted check).** The same DDIM loop and schedule, given the exact ε for
   Gaussian data, returns the target:
   - N(0, 1) gives −0.001 / 0.988;
   - N(0, 0.44) gives −0.001 / 0.433;
   - N(0.5, 0.3) gives +0.499 / 0.293.
   Per-image levels match √(1/4096) noise.
3. **Single-step ε has no level bias.** On noised real inputs at t = 999 … 10, the per-image, per-channel
   mean ε error is 0.000–0.006 against an SD of 0.011–0.021. The implied x0 level error is ≤ 0.01 for
   t ≤ 900, conditional and unconditional alike (`smoke_full/eps_bias.json`). The drift accumulates along
   the trajectory.
4. **The clamp makes the bias.** Same checkpoint, same noise, 64 unconditional samples
   (`smoke_full/clamp_variants.json`):

| sampler variant | mean (model units) | SD | ring IQR | offset |
|---|---|---|---|---|
| as run: percentile clamp [−1.5, +11], raw ε | +1.16 / +1.29 / +1.30 | 1.78 / 1.71 / 1.70 | 0.82 / 0.69 / 1.05 | 77% |
| same clamp, ε recomputed from the clamped x0 | +0.98 / +1.11 / +1.11 | 1.93 / 1.87 / 1.90 | 0.93 / 0.77 / 1.17 | 73% |
| **symmetric clamp ± 11.3** | **−0.06 / +0.06 / −0.10** | 2.40 / 2.48 / 2.78 | 1.34 / 1.35 / 2.00 | 67% |
| **no clamp** | **+0.10 / +0.21 / −0.01** | 4.21 / 4.09 / 4.27 | 1.61 / 1.45 / 2.30 | 59% |

   - At the first steps, the x0 estimate scales ε errors by up to √((1−ᾱ)/ᾱ) ≈ 300. Many pixels hit the
     clamp, and a clamp lopsided upwards turns that into a positive level.
   - Removing the asymmetry removes the bias but exposes the spread: SD 2.4–4.3 against 1.05, ring IQR
     1.3–2.3 against 0.02. That spread is the network's error, which the clamp was hiding.

## What this means for the 128-px brief (not decided here)

- **Fix 3 as specified is harmful** for data whose 99.9th percentile is a bright core. A clamp that cannot
  add a positive level is needed: symmetric, or a narrower one (for example the 99th percentile for both
  limits), or dynamic thresholding. Any choice should be shown on the oracle check before use.
- **Fixes 1 and 2 hold their place.** Unit variance is reached exactly (input SD 1.02–1.05), and the sky
  subtraction halves the background spread.
- **The remaining gap is training.** The 2,000-step model's ε is accurate one step at a time but not over
  the trajectory. Whether 100k+ steps closes it is the open question; the Mac cannot answer it cheaply
  (about 44 h per 100k steps).
- **Suggested check for the rented run:** the same statistics on unconditional samples every 10k steps,
  with a stop rule written in advance.
