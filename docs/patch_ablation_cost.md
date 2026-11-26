# Patch-size ablation 16 → 8 → 4 (D11) and the 128-px renderer: rented-GPU cost

**Current plan: the full grid only, 27 encoders, 45–87 H100-hours ($90–305) on one GPU ("Decided 25 Nov"), 51–97 billed ($102–340) as now planned on two boxes ("Decided 25 Nov, later").** Everything above that section is the history of the estimate.

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

## Decided 24 Nov (user): three seeds per arm

The grid is now **three seeds (train_seed 0, 1, 2) for every arm, including all three patch-16 GPU references** (27 trained encoders),
because the two-seed design could not resolve the JEPA's and MAE's effects (`docs/patch_grid/flags.md`). Seed 1 stays in.

The extra seed, per objective-and-size, from the hours-per-seed table:

| extra (third) seed of | H100 | A100 |
|---|---|---|
| patch 8, three objectives | 2.1–3.6 h | 3.3–6.1 h |
| patch 4, three objectives | 9.1–18.2 h | 22.6–41.5 h |
| patch 16, three objectives | 2.1–3.6 h | 2.1–3.3 h |
| **total added** | **13–25 h** | **28–51 h** |

| | H100 GPU-h | H100 cost | A100 GPU-h | A100 cost |
|---|---|---|---|---|
| full grid, 2 seeds, + patch-16 controls (as decided 21 Nov) | 30–57 | $60–200 | 62–112 | $81–224 |
| **full grid, 3 seeds, + patch-16 references** | **44–84** | **$88–294** | **91–165** | **$118–330** |
| the same + the 128-px renderer | 46–87 | $92–305 | 95–173 | $124–346 |

- The 3-seed totals allow 0.5–2 h for the nine extra encoders' ladder embeddings. Untrained banks are not per seed and don't change.
- Wall-clock on one H100, serially: about 2–4 days. Two to four GPUs on one box cut it proportionally.
- Retained output grows with the encoders: ≤ 45 GB for the three-seed grid (was ≤ 30 GB).
- **The masker's CPU cost at patch 4 is not in these hours.** On this loaded Mac, the JEPA masker takes 0.50 s per batch at a 64 × 64 grid, against 0.044 s at
  16 × 16 (measured 24 Nov). Over 101,308 steps that is about 14 h of CPU time per run. It can hide in loader workers on a rented box, but a
  serial masker would dominate the JEPA patch-4 wall-clock. The 200-step timed smoke must measure it.

The prices are the assumed rates above, unconfirmed.

## Decided 25 Nov (user): the full grid, and only the full grid

The blind ρ gate is gone (and with it the staged cost table). All 27 encoders run: JEPA, MoCo and MAE at patch 16, 8 and 4, seeds 0, 1 and 2, all bf16 on the GPU. Stage 1 (patch 16 and 8) goes first as an
engineering check with no decision attached, so **there is one budget, and no cheaper branch to plan for.**

| | H100 GPU-h | H100 cost | A100 GPU-h | A100 cost |
|---|---|---|---|---|
| training: 9 patch-16 + 9 patch-8 + 9 patch-4 arms | 40–76 | $80–266 | 84–153 | $109–306 |
| set-up (environment, R2 pull, pins, timed smokes) | 2–3 | $4–11 | 2–3 | $3–6 |
| embeddings: 27 trained encoders + 9 untrained (3 seeds × 3 sizes) over the 74,825 union | 3–8 | $6–28 | 10–25 | $13–50 |
| **full grid** | **45–87** | **$90–305** | **96–181** | **$125–362** |
| + the 128-px renderer | 47–90 | $94–315 | 100–189 | $130–378 |

- **The embedding row is new in size.** The grid keeps the ladder's fp32 embedding recipe (§3). An H100 has no tensor-core speed-up for fp32 (about 67 TFLOP/s peak against 989 in bf16), and a patch-4 pass over the
  union is about 36 PFLOP: roughly 18 minutes per encoder on an H100 and an hour on an A100, for 12 patch-4 encoders. The untrained banks are 3 seeds (not the 30 of the older tables), which takes off most of what they cost
  before. bf16 embeddings would be quicker; they would also not be the ladder's recipe.
- **JEPA's patch-4 masks are precomputed on the box's CPUs.** Measured on the loaded Mac: 0.6 s per batch at grid 64 (64 batches in 40 s), so 101,308 steps are about 17 core-hours per seed, 50 for three. On 16 cores it is about
  an hour per seed, run beside stage 1's GPU work. About 33 MB of output per seed. A box with few CPU cores is the wrong box.
- **Wall-clock:** about 2–4 days serially on one H100; two to four GPUs on one box cut it in proportion.
- **Retained output:** ≤ 45 GB, plus about 0.12 GB of embeddings per encoder (36 encoders ≈ 4 GB), which come home for the probes.
- The ± 30% model uncertainty and the assumed, unconfirmed prices are as everywhere above. The older tables (two seeds, the scenarios (a)–(c), the 24 Nov three-seed table) are history, kept for the arithmetic.

## Decided 25 Nov, later (user): stage 1 on 1 × H100, stage 2 on 4 × H100

The arms are unchanged. Two boxes mean a second environment and pull on four billed GPUs, and a small idle tail: **51–97 GPU-hours billed ($102–340)** against the 45–87 above, for a wall-clock of 25–44 h instead of 2–4 days. The breakdown is in
`docs/patch_grid/runbook.md` §1. Checkpoints stay on the boxes and die with them: about 6 GB comes home (embeddings, encoders, traces, the checklist), not the ≤ 45 GB above.

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

## 25 Nov, third version: on demand against spot (prices read 25 Nov 2026; nothing booked)

Prices are from the providers' public pages. RunPod's are dated 12 Nov 2026; Nebius's carry no date.

| | H100 on demand | H100 spot / preemptible | billing | preemption |
|---|---|---|---|---|
| Nebius | $4.50 /GPU-h (SXM; presets of 1 or 8 GPUs, **no 4-GPU**) | **from $0.79**, a spot price that can move every 15 min | per second; egress free | SIGTERM, then 60 s, then stop. Volumes survive; no local NVMe is listed for H100. **No restart on its own.** An in-VM `shutdown` does not stop the bill |
| RunPod | Secure SXM $4.39; Community SXM from $2.69; PCIe $2.89 / $1.99 | **not published** (the API has no spot field; a blog summary says about 50% off, unverified) | per second; no egress fees | **undocumented**: no notice period, no restart rule. Volume disk survives a stop; container disk does not |

GPU-hours from the runbook, all estimates: 51–97 for two boxes (box 2's four GPUs bill while it sets up and idles), or 46–89 for everything on one 1-GPU box with `STAGES=all` (2–4 days of wall-clock).

| plan | GPU-h | rate | cost |
|---|---|---|---|
| Nebius on demand, one 1-GPU box, `STAGES=all` | 46–89 | $4.50 | $207–401 |
| RunPod Secure, two boxes | 51–97 | $4.39 | $224–426 |
| RunPod Community SXM, two boxes | 51–97 | $2.69 | $137–261 |
| **Hybrid:** stage 1 + smokes on demand (Nebius 1 GPU) **+** stage 2 + embeddings preemptible (Nebius, at the $0.79 floor, 5–25% overhead) | 16–26 + 30–63 | $4.50 + $0.79 | $97–179 |
| the same hybrid, preemptible at $2.00 | | $4.50 + $2.00 | $135–275 |
| all preemptible at $0.79 (stage 1 judged on unresumed arms only: see below) | 46–89 ×1.05–1.25 | $0.79 | $38–88 |

**Expected preemption overhead.** One preemption costs:
- half a checkpoint interval of lost training: 1.3–2.5 min at patch 16 and 8, and 4–16 min at patch 4 (MAE worst);
- a restart: boot, `uv sync`, preflight and parity, about 10–25 min if the disk survives. If it doesn't, add the re-pull: about 9 min at 5 Gbit/s, 46 min at 1 Gbit/s.

At one preemption per 10 h that is 2–4% (disk kept) to 4–12% (disk lost). At one per 2 h it is 8–20% to 17–58%. Spot at half price pays unless preemptions come more often than about once an hour at patch 4. Nebius won't restart a preempted VM, so wall-clock waits for you, but the stopped GPU doesn't bill. On a 4-GPU box one preemption costs four arms' half-intervals at once.

**Resume, checked.** `artifacts/patch_grid_killtest.py` runs the real launcher (`patch_grid_run.main`, with the pins and cache gates stubbed, STOP and the checkpoint interval set from outside) on the Mac for JEPA, MoCo and MAE at patch 16:
- uninterrupted;
- again, as the determinism control;
- SIGKILLed after its first checkpoint, then restarted;
- with another checkpoint interval.

It compares every tensor of the final checkpoint, `encoder.pt` and the losses. It is queued behind the load gate, after the 25 Nov queue; result pending. Patch 4 is not in it: JEPA at patch 4 needs masks, and MPS at 4,096 tokens is slow. `tests/test_checkpoint_resume.py` covers the loop on CPU.

**Why the hybrid.** The judge refuses a resumed stage-1 arm's throughput row: the rate covers only the last invocation, so it would be overstated. One stage-1 preemption therefore stops the run at the judge. Two ways out, both yours to decide:
- (a) stage 1 on demand, which is the hybrid above;
- (b) amend the checker (unpinned) to measure a resumed arm's rate over the steps that invocation ran. That changes how a bar is measured, so it is not done.

### Priced, not adopted: dropping MAE at patch 4

- **Saves:** 3 × 4.2–8.3 = 12.6–24.9 GPU-h, plus three embeddings and one smoke, about 13–26 GPU-h in all.
  - In money: $58–117 at $4.50, $35–70 at $2.69, $10–21 at $0.79.
  - Stage 2 on four GPUs shortens from 6.9–13.8 h to about 4.9–9.9 h of wall-clock.
  - The largest remaining arm is JEPA at patch 4 (18 GB), so 80-GB cards stop being necessary. Another card is another device, though, and the throughput bars are H100 figures.
- **Removes:**
  - MAE's three Q1 readings (G1–G3). Q1 needs both steps, and "INSUFFICIENT in either" makes a reading INSUFFICIENT, so without 8→4 none can be read.
  - The 8→4 step of the JEPA–MAE and MAE–MoCo Q2 pairs: 6 of the 18 Q2 step-states.
  - MAE's 4×4 secondaries.
- **Keeps:** MAE's 16→8 step, all of JEPA and MoCo, and the JEPA–MoCo pair.
- MAE is also the objective whose Q1 the prereg already expects to read UNRESOLVED at low ρ (Δ80 0.047–0.164 against E = 0.02).
- Adopting it is an amendment to the hashed pre-registration: a new section hash, and the plan-hash table in the runbook rewritten. Yours to make.

**Decided 26 Nov (user):** Nebius. On demand for the stage 1 checklist only (nine uninterrupted smokes). All 27 arms and the embeddings on one preemptible VM, capped at $2.00/h. No MAE amendment: the full pre-registered grid. RunPod no longer sells spot pods (its v2 API lists them as removed). Re-priced: $50–106 at the $0.79 floor, $106–234 if spot sits at the cap, disk included. The table and steps are in `docs/patch_grid/runbook.md`, "Recommended: Nebius".
