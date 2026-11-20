# 8×8-patch ablation (D11): cost on a rented GPU

Estimate only (20 Nov 2026); nothing launched, rented or measured on a GPU. The run: A1's recipe
(`configs/pretrain_v2.yaml`, training seed 0, fixed stop at 101,308 steps = 4 epochs) with
`model.patch_size: 8` instead of 16, everything else unchanged. Purpose: the Rung-4 control — are the R4
answers (medium winding; star/artifact; 3 and 4 arms in A1/A3) under-resolved at 16-pixel patches, or absent?

## Tokens and compute

- **Tokens per image:** 256 px stamps (no-rebin, native 0.396″/px) → 32 × 32 = **1,024 tokens** against
  16 × 16 = 256. The code is grid-generic: `VisionTransformer` derives `grid_size` from `img_size //
  patch_size`, positional embeddings are sin-cos from the grid, and the masker's block sizes are
  fractions of the grid (`MaskConfig.target_scale` 0.15–0.20, `context_scale` 0.85–1.0). No code change.
- **FLOPs per step**, counted analytically for ViT-S (dim 384, depth 12) + predictor (dim 192, depth 6),
  batch 32, four target blocks, context ≈ 45% of tokens:
  - patch 16: ≈ 39 GFLOP per image, **≈ 1.25 TFLOP per step**;
  - patch 8: ≈ 197 GFLOP per image, **≈ 6.3 TFLOP per step**, about **5×**: 4× from the tokens, the rest
    from attention, which grows with the square of the token count.
- **Cross-check on the Mac:** A1's 1.515 training steps/s × 1.25 TFLOP ≈ 1.9 TFLOP/s effective (fp32,
  MPS). At 5× the work the same machine would manage ≈ 0.3 steps/s: **≈ 94 h for 101,308 steps** before
  probes and checkpoints. That is why it goes to a rented GPU.

## Step time and hours for one seed (101,308 steps)

A1 trained in **full fp32** (no `autocast` in the config). Precision is the biggest lever, and the
choice is a recipe decision, not only a speed one:

| precision on the GPU | A100 80 GB, s/step | A100 hours | H100, s/step | H100 hours |
|---|---|---|---|---|
| strict fp32 (as A1) | 0.5–0.7 | **14–20 h** | 0.2–0.25 | **6–7 h** |
| TF32 matmuls | 0.12–0.16 | 3.5–4.5 h | 0.05–0.07 | 1.5–2 h |
| bf16 autocast | 0.06–0.08 | 1.7–2.3 h | 0.03–0.04 | ≈ 1 h |

Assumed effective throughput: 25–50% of peak, with the ranges reflecting that. Add ≈ 1–2 h for set-up,
the data copy check and the probe points.

- **Memory:** activations at 1,024 tokens × batch 32 fit comfortably in 40–80 GB; SDPA (fused attention)
  avoids materialising the 1,024² attention maps.
- **Data rate:** 32 stamps × 393 KB ≈ 12.6 MB per step, so 100–200 MB/s of random reads at bf16
  speeds. Fine from local NVMe; not from network storage.

## Data to move

- **The cache is unchanged.** Patch size is a model setting; the fp16 cache is keyed on the pixel
  pipeline (`pipeline_hash`, freeze `246d8de6ba9a`), so A1's cache serves as-is. Directory
  `runs/m_v2/cache/eab81d17…`: `stamps.f16` **416 GB** (387 GiB; 3 bands × 256² fp16 ≈ 393 KB per stamp,
  pretraining plus probe stamps), plus `index.json` 26 MB, `probe_columns.f64` 685 MB,
  `petro_rad_arcsec.f64` 8 MB. Total **≈ 417 GB**.
- **Upload time:** ≈ 1 h at 1 Gbit/s, ≈ 9 h at 100 Mbit/s, ≈ 23 h at 40 Mbit/s. On a slow uplink the
  copy costs more wall-clock than the training. A drive shipped or a staging bucket near the GPU are the
  alternatives.
- **Checkpoints back:** ≈ 0.4 GB each; A1's schedule keeps one every 2,000 steps (51 to step 101,308,
  ≈ 20 GB). Only the 4-epoch checkpoint (and the four probe-point ones) need to come home.

## What carries over, and what changes

**Carries over unchanged:**
- `artifacts/aligned_train.py` / `m2_long_run.Run`: training seed 0, the fixed stop at 101,308;
- `artifacts/aligned_launch.sh`, with paths adapted to the rented box;
- the normalisation freeze `246d8de6ba9a`, the cache, the identical split (ids `17296e0f`);
- the frozen effect floor 0.7267, and the probe and ladder drivers (`ladder_env` contract).

**Changes (new pins):**
- a config `configs/pretrain_v2_p8.yaml` (`patch_size: 8`), so a new shipped config hash in the
  launcher's `EXPECT` (A1's is `2e2f2877221e1356`);
- `runtime.device` is hashed, so `cuda` gives a different config hash from A1's MPS run regardless;
- the **untrained bank (D23's bar) must be rebuilt.** It is keyed on the model config, and patch 8 is
  a different architecture, so a new K = 30 bank (E4) is needed. It took 6.6 h (v2) and 10.3 h (v1) on the
  Mac at patch 16; on the GPU it is minutes to an hour;
- the ladder subset for the new encoder (E3, E7 bank, p2, r0, s2, t1; ≈ 9 h per patch-16 encoder on the
  Mac, longer at patch 8 for the embedding steps), via a manifest like `ladder_v2_m_manifest.py`.

**A confound to design out.** Comparing a CUDA patch-8 run with A1 (MPS, fp32) changes device, and
possibly precision, along with patch size. A **patch-16 control on the same GPU at the same precision**
isolates the patch size. It costs about a fifth of the patch-8 run: ≈ 3–4 h in strict fp32 on an A100,
under an hour in bf16. Since seed 1 runs high on v2 (findings summary), seed 0 (A1's) is the cleaner
single seed.

**Recommended shape:** one A100 or H100, TF32 or strict fp32 chosen before launch and written into the
pre-registration; patch 8 and the patch-16 control, both at seed 0; the E4 bank and the ladder subset on
the same box. In TF32 on an A100: ≈ 4.5 + 1 + ≈ 2 (bank and ladder embeddings) ≈ **8 GPU-hours**, plus
the data copy.
