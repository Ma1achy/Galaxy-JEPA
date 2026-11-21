# Patch-size ablation 16 → 8 → 4 (D11) and the 128-px renderer: rented-GPU cost

Estimate only (21 Nov 2026). Nothing booked, rented, launched or measured on a GPU. This supersedes
`patch8_cost.md` for the plan as changed on 21 Nov: full 256-px stamps, no crops for the encoders; ViT-S at
patch 8 and 4; JEPA plus the D12 baselines (MoCo, MAE); 2 seeds each.

## Recipe held fixed

- **A1's run:** 810,491 train stamps, batch 32, **101,308 steps (4 epochs)**. The cosine is defined over
  253,270 steps and stopped at 101,308 via `stop_after`, as every arm in `artifacts/baselines.md`.
- **Same images for every arm:** the cache is unchanged, because patch size is a model setting and the fp16
  cache is keyed on the pixel pipeline.
- **Masking fixed in pixels.** Target blocks and the 2.5 × R_petro box cover the same sky at every patch
  size, so token counts scale by (16/p)²:

  | patch | tokens | JEPA context | JEPA targets | MoCo window | MAE visible |
  |---|---|---|---|---|---|
  | 16 | 256 | 81 | 180 | 146 | 64 |
  | 8 | 1,024 | 324 | 720 | 584 | 256 |
  | 4 | 4,096 | 1,296 | 2,880 | 2,336 | 1,024 |

- **Positional embeddings are re-initialised per size.** They are sin-cos from the grid, so this holds by
  construction; the patch-embedding layer is new at each size.

## Compute per image

Model: per block N·(24d² + 4Nd) FLOP; training costs 3× a forward pass; the no-grad target or key encoder
costs 1×.

- This is the model of `baselines.md` §5.1, and it reproduces that table at patch 16: 27.9 / 26.4 / 50.3
  GFLOP against 28.5 / 26.7 / 50.7.
- `patch8_cost.md` put JEPA at patch 8 at 197 GFLOP, assuming a 45% context. The masker's measured
  81-token context gives **141.6**, so that document's patch-8 hours were about 1.4× high.

| GFLOP per image | patch 16 | patch 8 | patch 4 |
|---|---|---|---|
| JEPA (ViT-S + predictor 192 × 6) | 27.9 | 141.6 | 1,049 |
| MoCo (asymmetric, windows) | 26.4 | 124.3 | 799 |
| MAE (decoder 8 × 512) | 50.3 | 242.4 | 1,632 |

At patch 4, attention is about 64% of the FLOPs (4N·d against 24d²). Flash attention keeps memory linear in
N but does not reduce these FLOPs.

## Peak memory at batch 32

Assumptions:
- activations are 16-bit, with flash attention (no N² maps) and no recomputation: 34·s·b·h bytes per layer;
- × 1.3 for the fp32 norm and softmax statistics;
- + 1.5 GB for weights, optimiser, EMA and the CUDA context;
- ± 30%.

| per GPU | patch 8 | **patch 4** | largest batch at patch 4 that fits: 40 GB / 80 GB |
|---|---|---|---|
| JEPA | 5.6 GB | **≈ 18 GB** | 67 / 138 |
| MoCo | 5.4 GB | **≈ 17 GB** | 71 / 145 |
| MAE | 9.1 GB | **≈ 32 GB** | 36 / 74 |

**Batch stays 32** (A1's recipe; a bigger batch changes the LR rule and the step count). Every arm fits on
one GPU with no checkpointing or accumulation. MAE at patch 4 is close to a 40-GB card's limit, so take
80 GB.

## Hours per seed

Assumptions:
- bf16 autocast plus flash attention;
- effective throughput of 25–45% of peak on an A100 (312 TFLOP/s) and 18–35% on an H100 SXM
  (989 TFLOP/s). ViT-S's narrow d = 384 keeps the matmuls small;
- a per-step floor of 25–40 ms for the loader and kernel launches. That floor sets patch 8 on an H100;
- MoCo + 5–15% for on-device augmentation (`baselines.md` §5.2);
- ± 30% model uncertainty.

| hours per seed | A100 80 GB | H100 80 GB |
|---|---|---|
| JEPA p8 | 0.9–1.6 | 0.7–1.1 (floor-bound) |
| MoCo p8 | 0.8–1.7 | 0.7–1.3 |
| MAE p8 | 1.6–2.8 | 0.7–1.2 |
| **JEPA p4** | **6.7–12.1** | **2.7–5.3** |
| MoCo p4 | 5.4–10.6 | 2.2–4.6 |
| MAE p4 | 10.5–18.8 | 4.2–8.3 |
| JEPA p16 (the control below) | 0.7–1.1 | 0.7–1.1 (floor-bound) |

**Evaluation on the same box:**
- an untrained bank (D23's bar, K = 30) per patch size, shared by the three objectives because they use
  the same encoder architecture: p8 0.1–0.5 h, p4 0.9–3.9 h;
- ladder embeddings per trained encoder, about 3 passes over the 74,825-galaxy union: p8 1–3 min,
  p4 5–23 min.

## Totals (2 seeds per arm)

Each total includes:
- the training runs;
- the untrained banks for patch 8 and 4;
- the ladder embeddings for every encoder;
- 2–3 h of set-up: environment, R2 pull and check, smoke steps.

Prices are **assumed** on-demand single-GPU rates, not checked live: **A100 80 GB $1.30–2.00/h, H100 SXM
$2.00–3.50/h.** Confirm them at booking.

| scenario | runs | A100 GPU-h | A100 cost | H100 GPU-h | H100 cost |
|---|---|---|---|---|---|
| **(a) JEPA only** | JEPA p8, p4 × 2 | 20–36 | $26–71 | **10–18** | **$20–64** |
| **(b) JEPA + 8×8 baselines** | (a) + MoCo p8, MAE p8 × 2 | 25–45 | $33–90 | **13–23** | **$26–82** |
| **(c) full grid** | (b) + MoCo p4, MAE p4 × 2 | 58–105 | $75–211 | **26–50** | **$52–174** |
| + 128-px renderer, if the smoke passes | 200k steps (100k–300k) | 4.2–7.6 (2.1–11.4) | $5–15 | 1.7–3.3 (0.9–5.0) | $3–12 |
| + patch-16 GPU control (recommended) | JEPA × 2 / all three objectives × 2 | 1.4–2.3 / 4.3–7.1 | $2–14 | 1.4–2.3 / 4.3–7.1 | $3–25 |

**H100 is the better buy at patch 4,** at about 2.3–2.5× the speed for about 1.5–1.75× the price. At patch
8 it hits the step floor and buys little. Patch 4 dominates every scenario: in (c), 75–80% of the hours.

On one GPU, serially, the wall-clock is (c) 1–2 days on an H100. Two to four GPUs on one box cut it
proportionally at the same cost, since the arms are independent runs. The patch-8 runs use under 10 GB, so
two can share one 80-GB GPU.

## Storage and transfer (R2)

The R2 shipping layer was designed but never built (user, 21 Nov); building it waits on the upload-speed
decision. These costs use Cloudflare R2's public price list as I know it (storage $0.015 / GB-month, free
egress, operations $4.50 / $0.36 per million). **Confirm the price list at booking.**

- **Up:** the cache is about **417 GB** (`stamps.f16` 416 GB plus index, probe columns and scalars),
  uploaded once.
  - Ingress is free.
  - The time is set by the home uplink: about 1 h at 1 Gbit/s, 9 h at 100 Mbit/s, 23 h at 40 Mbit/s.
  - Storage is about **$6.30 per month**; multipart operations come to cents.
- **To the box:** free egress, about 6–60 min at 1–10 Gbit/s.
  - The box needs **≥ 0.5 TB local NVMe**. At the H100 patch-8 step floor the loader reads about 500 MB/s
    at random (32 × 393 KB per 25 ms), which network storage cannot serve.
  - Some providers bill that volume separately (around $0.10 / GB-month, so about $50 per month).
- **Back:** each run returns its 4-epoch checkpoint and the four probe-point ones (about 0.4 GB each), plus
  embeddings. That is **≤ 30 GB for (c)**, free egress, and pennies to store.
- **Transfer and storage come to under $10 in every scenario.** The real cost is the upload time on a slow
  uplink.

## Decided 21 Nov (user): plan and pre-registration choices

**Plan:** the full grid in one rental on an **H100 80 GB**:
- JEPA, MoCo and MAE at 8×8 and 4×4, 2 seeds each;
- the **patch-16 bf16 GPU control for all three objectives**, 2 seeds each;
- the 128-px renderer if the diffusion smoke passes;
- **first, a 200-step timed smoke per arm** on the rented box.

**Budget** (from the tables above): 26–50 + 4.3–7.1 + 1.7–3.3 ≈ **32–60 H100-hours, about $65–210** at the
assumed rates. Nothing is booked.

**Pre-registration choices:**
1. A patch-16 GPU control (bf16) for all three objectives: **yes**.
2. **MoCo windows scale with pixels:** 20–28 tokens at 8×8, 40–56 at 4×4.
3. **MAE at 8×8 and 4×4:** mask in 16-px groups, with the normalised target computed over the 16-px group,
   identical to 16×16. Only the encoder's token size changes. This replaces design note 2's MAE item.
4. **JEPA:** unchanged.

**R2:** the shipping layer was designed but never built. Do not build it yet; the transfer route depends on
the upload speed. The storage and transfer figures below stand as the R2 option's cost.

## Design notes the cost depends on (written before the decisions above)

1. **Flash attention means bf16 or fp16.** The fused kernels do not run in fp32, and PyTorch's SDPA falls
   back to its memory-efficient kernel in fp32. A1 ran fp32 on MPS, so a GPU run changes device and
   precision along with patch size. The patch-16 GPU control at the same precision isolates the patch
   size, at about 5% of (c).
2. **Pixel-fixed masking per objective:**
   - **JEPA:** the masker's block scales are fractions of the grid, and the 2.5 R box maps to tokens from
     pixels, so this is fixed in pixels already.
   - **MoCo:** the window side is in tokens (10–14), so it must scale to 20–28 at patch 8 and 40–56 at
     patch 4. That is a config change.
   - **MAE:** a ratio of 0.75 over 4-px tokens is far easier than over 16-px ones, because neighbours
     interpolate. Pixel-fixed means masking in 16-px groups.
   - The norm-pix target over a 4 × 4 × 3 patch (48 values) is much noisier per patch. Both MAE points are
     decisions; neither changes the cost.
3. **Batch 32 leaves GPUs underused at patch 8.** Sharing a GPU between runs, or MIG, is the lever, not a
   bigger batch.
4. **Model uncertainty is ± 30%** on every figure above. A 200-step timed smoke per arm on the rented box
   is the first spend, and replaces these numbers before the main runs start.
