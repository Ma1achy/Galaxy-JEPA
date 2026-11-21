# Almagest diffusion pilot — feasibility report (21 Nov 2026)

VISUALISATION ONLY. No claim, nothing in the science record. Branch `almagest-geometry`.
Code: `artifacts/almagest_diffusion.py`. Numbers: `diffusion_pilot_eval.json`. Run directory:
`runs/almagest/diffusion/` (`train_log.jsonl`, `bench_b32.json`, `eval.log`).

## Set-up

- An RCDM-style conditional U-Net: 33.1M parameters, 64 × 64 × 3 (g r i). The model is
  conditioned on **raw A1**: `runs/m_v2/encoder.pt`, penultimate block, mean-pooled, 384-d.
- Classifier-free guidance with 10% condition dropout. 1000-step schedule; sampled with DDIM
  (η = 0, 100 steps).
- Training images: ~100k train galaxies, 4 × 4 area-averaged from the native 256-px stamps.
  This is display-only: A1 itself always embeds at native 256 px, and the no-rebin rule is
  untouched.
- Training: batch 32, lr 2e-4, EMA weights. 3.0 h on MPS, **6,413 steps (≈ 2 epochs)**.
  The loss EMA plateaued at 0.12–0.15 from step 500 and ended at 0.122.
- Test galaxies come from A1's monitor slice (2,048 drawn), seen by neither A1's training nor the diffusion model.
- Eval: 256 held-out test galaxies × 4 samples, at w = 1 and w = 3. Each sample is mapped
  back to normalised-stamp units, upsampled bilinear to 256 px and re-embedded with A1.

## (a) Reconstruction sheets

- `diffusion_pilot_recon_w1.png`, `diffusion_pilot_recon_w3.png`.
- Layout: 64 galaxies, each as original | 4 samples; i r g bands, per-image percentile stretch.
- **By eye, the samples are not reconstructions.** Most have strong colour casts (red, green,
  blue or teal fields) and grainy, star-field-like backgrounds. The originals are dark-sky
  stamps with one central galaxy.
- **The casts are real, not a display artefact.** A border check on 32 w = 1 samples
  (scratch diagnostic, not committed):

| per channel g / r / i | real | sample |
|---|---|---|
| border median | −0.10 / −0.12 / −0.13 | −0.05 / 0.03 / 0.13 |
| border-median IQR across images | 0.03 / 0.04 / 0.11 | 0.40 / 1.77 / 2.07 |
| border noise (1.4826·MAD) | 0.12 / 0.13 / 0.17 | 0.25 / 0.42 / 0.44 |
| centre peak above background | 3.5 / 5.2 / 5.2 | 9.1 / 13.7 / 11.6 |
| images with a channel offset > 1 noise SD | 3% | 55% |

- What the samples do get right: there is usually a compact central source, and some show
  elongated or edge-on shapes. Guidance (w = 3) makes the 4 samples of one galaxy agree
  with each other more: shared field colour, shared crowding, and in a few rows a shared
  bright elliptical.

## (b) Re-embedding with A1 (medians over 256 galaxies)

"Centred" subtracts the train bank's mean embedding; the raw cosines share a large common
mean. The rank is the conditioning galaxy's rank among the 256 by cosine; chance is 128.5.

| | cos to condition, raw | centred | rank (centred), median | top-1 | cos between 4 samples of one galaxy (centred) |
|---|---|---|---|---|---|
| samples, w = 1 | 0.22 | −0.01 | 110 | 2.4% | 0.39 |
| samples, w = 3 | 0.30 | 0.02 | 86.5 | 2.9% | 0.45 |
| **ceiling:** the real galaxy pooled to 64 px, upsampled, re-embedded | 0.53 | 0.12 | 60.5 | 6.6% | — |
| random real pairs (native) | 0.61 | −0.02 | — | — | — |
| random real pairs, both round-tripped | — | **0.90** | — | — | — |
| nearest real neighbour, train 100k (native) | 0.97 | 0.93 | — | — | — |

**The reading that matters: at 64 px, this test can barely tell a good sample from a bad
one.**

- Pool a *real* galaxy to 64 px and upsample it: it lands 0.12 (centred) from its own native
  embedding, but 0.90 from any other round-tripped galaxy.
- So A1 sends every 64-px-upsampled image into one common region. The identity it keeps is
  small: rank 60 of 256, top-1 7%.
- The samples' raw cosines (0.22 and 0.30) sit *below* random real pairs (0.61). Both
  samples and round trips sit off the real manifold.

What the pilot does show:
- **Conditioning carries some information.** Samples of the same galaxy agree with each other
  (0.39–0.45) far more than samples agree with their condition.
- **Guidance moves the rank from 110 to 86.5**, against the ceiling of 60.5.

Neither number shows the model renders *the* galaxy, and at 64 px no sample could show it
under A1.

## (c) What varies across the 4 samples of one galaxy

By eye, from the sheets:
- **Varies most:** background level and colour, field crowding (star-like specks), and noise
  texture.
- **Then:** the central source's brightness and, less often, its shape.
- **More stable:** whether there is one compact central object; at w = 3, the field colour
  and crowding within a row.

So the variance is spent on the nuisance the model has not learned (sky level and colour
per channel), not on morphology.

## Timings

**Training** (MPS; batch 32; 2.67 TFLOP per step, counted on CPU). Rates are per 500-step
window, from `train_log.jsonl`:

| windows | steps/s | load (1 / 5 / 15) |
|---|---|---|
| steps 0→500 | 0.613 | 9.3 / 9.2 / 9.0 |
| **500→1000, ending 01:34: contended** (my relabel and sheet jobs ran beside it). Excluded. | 0.505 | 15.6 / 11.9 / 10.6 |
| **1500→2000: slow** (load from other sessions; no job of mine). Shown, not excluded. | 0.456 | 8.9 / 10.2 / 10.3 |
| the other 9 windows | 0.546–0.647 | 3.3–11.5 (1-min) |
| **estimate rate: median, 01:34 window excluded** | **0.636** | — |

- Excluding the 1500→2000 window as well changes the median by < 0.001.
- No window had its 5-min load above 16. The start was at load 17.7, from the bench's own
  tail, and is stamped `under_load`.
- Memory: MPS driver 8.2–8.3 GB, process RSS 0.93 GB. Batch 64 spills into swap on this
  18 GB machine, which caused the 54-min stalled bench, so batch 32 is the Mac ceiling.
- Effective throughput: 0.636 × 2.67 TFLOP ≈ **1.7 TFLOP/s**. bf16 autocast was slower on MPS.

**Sampling:** w = 1 takes 1.42 s per image. w = 3 takes 4.23 s per image, because
`sample()` runs the unconditional pass twice per step (3 forwards rather than 2). That is a
⅓ waste at w ≠ 1, to fix before any reuse.

**Mac vs rented GPU.** "Full run" is not fixed in the brief, so the table gives hours per
100k steps at batch 32; RCDM-like runs are 100k–300k steps.

- GPU rows use FLOPs ÷ assumed sustained throughput: A100 at 15–40% of 312 bf16 TFLOP/s,
  H100 at 15–35% of 990.
- A 33M U-Net at batch 32 will sit at the low end, so on a GPU raise the batch rather than
  trust the upper bound.
- Higher resolutions scale FLOPs ≈ 4× per doubling, assuming the same architecture. Self-attention
  at higher resolution grows faster, so those rows are a lower bound.

| per 100k steps, batch 32 | Mac (measured rate) | A100 (assumed) | H100 (assumed) |
|---|---|---|---|
| 64 px (this pilot) | **43.7 h** | 0.6–1.6 h | 0.2–0.5 h |
| 128 px (≈ 4×) | ≈ 175 h | 2.4–6.3 h | 0.9–2.1 h |
| 256 px (≈ 16×) | ≈ 700 h | 10–25 h | 3.4–8.0 h |

These are estimates, not measurements: no GPU was rented and nothing was launched.

## Feasibility, plainly

1. **This pilot does not render galaxies from their embeddings.** At 2 epochs, the samples
   carry per-channel sky offsets and excess noise, and do not match their conditions by eye.
   Candidate causes, none tested:
   - undertraining (6.4k steps);
   - the model-space squash with a (−2, 6) DDIM clamp, letting sky offsets through;
   - no per-channel background term in the loss.
2. **64 px cannot pass test (b) even when perfect.** A1 does not recognise its own galaxies
   after a 64-px round trip (centred 0.12, rank 60/256). A rendering for A1 walks should be
   ≥ 128 px and preferably native 256, which per the table is a rented-GPU job, not a Mac one.
   An alternative is to judge 64-px samples with a 64-px reference: compare against round-tripped
   real galaxies, centred on their own mean, rather than against native embeddings.
3. The Mac is fine for plumbing: batch 32, 1.7 TFLOP/s, about 44 h per 100k steps at 64 px.

No walks, extrapolation or Plate wiring were done.
