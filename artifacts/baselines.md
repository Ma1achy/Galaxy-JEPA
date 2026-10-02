# Baselines — MoCo and MAE, matched to M (Kickoff F)

*Status: draft, 2026-10-02 (first draft 2026-09-27). Nothing hashed, nothing trained. **Settled
(user, 2026-09-28):** 16×16 patches for MAE (D12 amended), F2 (MAE decoder 8×512) and the MoCo
settings of F3 (K = 65,536, m = 0.999, τ = 0.2). F1, F4–F9 and the residue of F3 await review.
Design sources: `DECISIONS.md` D10 revised (`:377`), D12 (`:414`), `docs/spec/objectives.md`, `TODO.md` Epic G
(`:467`). A choice is recorded as decided only once the user has signed it off; every open fork is
in §8. Claims marked *verify* are from memory of the primary papers and are checked against the
PDFs before sign-off.*

## 1. What "matched to M" means

D12 holds the data fixed and varies only the objective (`DECISIONS.md:418-423`). Each baseline
keeps everything in M's recipe that is not intrinsic to its objective. The matched items are below;
§7 lists every place where matching was impossible.

| quantity | M (`configs/pretrain.yaml`) | MoCo | MAE |
|---|---|---|---|
| backbone | ViT-S/16: dim 384, depth 12, heads 6, mlp 4.0 (`:62-67`), 21.6M params, no CLS | same | same |
| stamp / tokens | 256 px native, 16×16 = 256 tokens | same at read-out | same |
| corpus | v1 pretrain, 810,491 train / 16,477 monitor (`m_findings.md:13-21`) | same split (`_split_ids`, split seed 0) | same |
| normalisation | frozen, `content_hash 75100066b3e0…` (`:59`) | the same block verbatim, so the same fp16 cache (`pipeline_hash` is keyed on the pipeline only, `data/cache.py:60`) | same |
| batch | 32, `drop_last` (`:80`) | 32 | 32 |
| `steps` (schedule length) | 253,270 (`:73`) | 253,270 | 253,270 |
| stop | 101,308 via `stop_after` (`objectives/jepa.py:447`) | 101,308 | 101,308 |
| optimiser | AdamW, torch default betas (0.9, 0.999), wd 0.04 constant, no grad clip (`jepa.py:326`) | same | same |
| LR | peak 1.25e-4, warmup 1,250, cosine to 1.25e-7 (`:86-90`, `jepa.py:250`) | same | same |
| LR at the stop | 8.24e-5 (65.9% of peak) | same, by construction | same |
| EMA | 0.996 → 1.0 cosine over 253,270 (`:91-92`, `jepa.py:265`); 0.99738 at the stop | momentum encoder 0.999 → 1.0 on the same ramp (§3.3) | none (objective-intrinsic) |
| precision | fp32 (`autocast: null`, `:34`) | fp32 | fp32 |
| SIGReg | off, λ = 0 (`:121`, D21) | n/a | n/a |
| status | `smoke: true` (`:30`), so not a headline | `smoke: true` (§8, F1) | `smoke: true` |
| weight init | `seed_init(train_seed)` (`harness.py:372`) | byte-identical to M's for the same seed: the encoder is built first from the seeded stream | same |
| data order | `ResumableShuffle(seed=train_seed)` | the identical sequence to M's | same |
| read-out | penultimate block (`DEFAULT_LAYER = -2`: block index 10, the 11th of 12, before the final norm), mean over all 256 tokens (`vit.py:172`, `core/encoder.py:20-36`) | backbone `encode()` only; projector discarded | backbone `encode()` over all 256 tokens, no mask; decoder discarded |

"Block 11" throughout the project means the 11th block, index 10. Nobody writes `blocks[11]`.

**Seeds.** Two per baseline, paired to M's: training seed 0 pairs with M, training seed 1 with O2.
The split seed is 0 for both, exactly as in O2 (`o_findings.md:343-347`). Within a pair, the initial
encoder weights and the order of stamps are identical to the JEPA run, so the seed-0 MoCo run
differs from M in its objective and in nothing else the harness controls.

**The SIGReg caveat for D12 (`TODO.md:498`, `DECISIONS.md:945`) does not apply to M.** M runs at
λ = 0 (D21). No arm carries a distributional constraint, so the "you constrained its geometry"
objection does not arise. It stays live only if a SIGReg JEPA arm is ever added.

### 1.1 "Matched compute": what is matched, and why

**Matched: data exposure and the optimiser schedule.** Every arm takes 101,308 optimiser steps of
32 stamps: 3,241,856 stamp draws, 4.00 epochs of the train split, in the same order. A MoCo step
draws each stamp once and augments it into two views, which count as one image seen. An MAE step
draws each stamp once, however many of its tokens the encoder sees. The same step count puts the LR
and EMA at the same point on their schedules when training stops.

**Not matched: FLOPs and wall-clock.** MoCo's forward and backward over one view plus the momentum
encoder over the other, and MAE's encoder on 25% of tokens plus a full-token decoder, cost
different amounts per step (§5).

**Why exposure rather than FLOPs.** Matching FLOPs would give MAE about 1.8× fewer steps (8-layer
decoder) and MoCo about 1.1× more. That changes the number of epochs, so it changes which data and
how much of it each arm saw, which is the variable D12 holds fixed. It would also stop each arm's
cosine at a different point. The price: "JEPA beats MAE" cannot be read as "per FLOP"; the write-up
says "per image seen".

**Why keep M's truncated schedule.** M stopped at 40% of its 10-epoch cosine (`m_findings.md:165`);
the annealing tail never ran (`o_findings.md` P4, `:466`). A cosine completed over 101,308 steps
would give the baselines an anneal M never had. So every arm keeps `steps: 253,270` and stops at
101,308 through `stop_after`, which leaves "`config.steps` — and so the LR and EMA schedules —
untouched" (`jepa.py:319`). P4's cooldown question then applies to all three arms equally.

**The stop is fixed, not a rule.** M's 4-epoch stop was chosen by an AUC-reading rule
(`m_findings.md:162`). The baselines inherit its step count as a fixed number and apply no rule of
their own, so no baseline checkpoint is chosen by looking at labels (spec 1C). They are probed at
M's epoch points (0.5, 1, 2, 4) for trajectories only; those probes are reported, never used to
select a checkpoint.

## 2. Symmetry and augmentation policy (D10 revised)

D10 revised: "No rotation or reflection augmentation, across the whole encoder family (M and every
D12 baseline)" (`DECISIONS.md:377-378`); only objective-intrinsic augmentation is permitted
(`:397-402`). Two consequences:

- **MoCo:** no horizontal or vertical flips, and **no rotations of any kind, including multiples of
  90°**. This also rules out the rotation and flip augmentations of Hayat et al. 2021 and Zoobot,
  the galaxy-SSL precedents (*verify Hayat's list against arXiv 2012.13083 §3*).
- **MAE:** He et al. 2022's default pipeline is RandomResizedCrop(0.2–1.0) plus a horizontal flip
  (*verify: the MAE repo's `main_pretrain.py`*). MAE functions without augmentation, so none of it
  is objective-intrinsic. **MAE runs with no augmentation, exactly as M does.**

The no-rebin rule (`docs/spec/data.md:252-264, :400`) governs the stored stamps and every encoder's
read-out input, and a resampling crop is resampling. So MoCo's crops are native-pixel token-grid
windows (§3.4). Blur and noise touch only the transient training views, never the stored data or
the read-out input.

## 3. MoCo (contrastive)

### 3.1 Variant: the v2 queue, not v3's in-batch negatives (recommended)

| | MoCo v2 (Chen, Fan, Girshick, He 2020, arXiv 2003.04297) | MoCo v3 (Chen, Xie, He 2021, arXiv 2104.02057) |
|---|---|---|
| negatives | queue of 65,536 | in-batch only; batch 4,096 |
| negatives at batch 32 | K (the queue size) | **31** |
| head | 2-layer MLP projector, no predictor | 3-layer projector + 2-layer predictor, BN |
| loss | asymmetric InfoNCE | symmetrised |
| τ | 0.2 | 0.2 (v1: 0.07) |
| momentum | 0.999 constant | 0.99 → 1 cosine |
| galaxy precedent | Hayat et al. 2021 (MoCo v2 on SDSS, ResNet) | none known |

**Recommendation: v2's queue on M's ViT-S** (the queue settings are decided, §3.2). The queue exists to decouple the number of negatives
from the batch (MoCo v1, He et al. 2020, §3.2); batch 32 is the case it was built for. v3 at batch
32 has 31 negatives and an InfoNCE ceiling of log 32 ≈ 3.5 nats: a starved MoCo, a straw man. v2 is
also the variant of the galaxy precedent D12 cites.

The cost: v2 was published on ResNets, and v3 exists partly because ViTs trained unstably under the
older recipes. That instability grows with batch and LR (v3 §4), both small here. The 500-step
rehearsal (§6) is where it would first show.

### 3.2 Objective settings

| setting | value | source / justification |
|---|---|---|
| queue K | **65,536** (decided, user, 2026-09-28) | MoCo v1's queue (He et al. 2020, arXiv 1911.05722, *verify the section*), kept unchanged in v2. Replaces the draft's 8,192; the consequence is below. |
| τ | **0.2** (decided, user, 2026-09-28) | v2 (MLP head, τ = 0.2; *verify the table*); v3 the same. 0.07 is v1. |
| projector | Linear(384 → 2048), ReLU, Linear(2048 → 128) | v2's MLP head |
| predictor | **none** | v2. v3's predictor adds a BYOL-like path; D12 rejected BYOL as too close to JEPA (`DECISIONS.md:446-449`). |
| BatchNorm in the head | none | v2's head; the ViT uses LayerNorm only, so shuffle-BN is not needed |
| loss | asymmetric InfoNCE: query = view 1, key = view 2, positive at index 0, K queue negatives | v2; symmetrising is fork F4 |
| projector input | mean over the final block's tokens, after the final norm | no CLS token (`vit.py` docstring); v2's global-average-pool analogue |
| read-out | `encode()`: block index 10, before the norm, mean-pooled over the full 256-token grid | `core/encoder.py:20-25` |

**The settings are v2's, decided by the user (2026-09-28): queue K = 65,536, momentum 0.999,
τ = 0.2.** Sources: MoCo v2 is Chen, Fan, Girshick & He 2020, "Improved Baselines with Momentum
Contrastive Learning" (arXiv:2003.04297), which adds to MoCo the MLP projection head, τ = 0.2, the
aug+ augmentation (adding blur) and a cosine LR schedule; K = 65,536 and m = 0.999 come from MoCo
v1, He, Fan, Wu, Xie & Girshick 2020, "Momentum Contrast for Unsupervised Visual Representation
Learning" (CVPR 2020, arXiv:1911.05722), and v2 keeps them. v2 was run at batch 256 on 8 GPUs
(*verify*); it is the small-batch member of the family, which is why it is the template here.

**Known consequence, not a reason to deviate: key staleness at batch 32.** The draft proposed
K = 8,192 to hold v2's key age. v2 at batch 256 enqueues 256 keys per step, so the oldest key is
256 steps old. Here at batch 32 the oldest is 65,536 / 32 = **2,048 steps** old (mean 1,024), 8×
v2's. At m = 0.999 the key encoder's time constant is about 1,000 steps, so the oldest keys are
about two time constants stale, against about 0.26 in v2; under the ramp (§3.3) the time constant
is about 1,540 steps at the stop, about 1.3 time constants. The decision keeps the published
values; the rehearsal (§6) and the InfoNCE trace are where an effect of stale negatives would first
show.

### 3.3 Momentum encoder

**m = 0.999, decided (user, 2026-09-28).** The value is v1's, kept in v2 (MoCo v1 Table 4: about
0.999 best with a queue, 0.9 failing; *verify*). M's 0.996 is I-JEPA's value and has no queue behind
it: at 0.996 the time constant is 250 steps, so 65,536 keys at batch 32 (2,048 steps) would be
about eight time constants stale. The draft's matched alternative (0.996 with K = 2,048) is not
taken.

**Still open (F3 residue): the schedule around 0.999.** The draft proposed **0.999 → 1.0, cosine
over 253,270 steps (0.99935 at the stop)**, i.e. v2's start on M's ramp shape. v1 and v2 hold m
constant at 0.999 (*verify*). The decision fixes the value; whether it ramps as M's EMA does or
stays constant as in v2 is for review.

### 3.4 Augmentation: minimal, objective-intrinsic, no flips or rotations, no resampling

Every parameter is drawn from a generator seeded with `(train_seed, step)`, as the masker is
(`jepa.py:388`), so a resume reproduces the same views with no saved augmentation RNG. It runs on
the device inside the loss step (the loader has `num_workers=0`).

1. **Crop = a token-grid window.** Per step, one side s ∈ {10, …, 14} tokens, shared across the
   batch so shapes line up (160–224 px, 39–77% of the area); per image and per view, an independent
   top-left position. The view is `patch_embed_tokens(augmented_image)` then a gather of the
   window: the hook JEPA's context path uses (`jepa.py:53, :199-200`). There is **no resampling and
   no new positional code**: positions are an exact subset of the fixed 16×16 sin-cos grid, so the
   full-grid read-out is positionally in-distribution, and M's tokenisation phase is kept. This
   replaces RandomResizedCrop, whose scale jitter resamples.
2. **Per-band gain** on the normalised tensor, independent per band and view: the gri analogue of
   v2's colour jitter and Hayat's reddening. *Magnitude to fix (proposed ±10%) once Hayat's
   parameterisation is verified.*
3. **Gaussian blur**, σ ∈ [0.1, 2.0] px at p = 0.5: v2's aug+ (from SimCLR), standing in for
   Hayat's varying PSF. SDSS seeing ≈ 1.4″ ≈ σ 1.5 px, so σ up to 2 px roughly doubles the PSF.
4. **Additive Gaussian noise** (Hayat). *σ to fix as a fraction of the normalised sky σ.*

Dropped: flips and rotations (D10); grayscale (destroys colour); hue and saturation (meaningless for
gri); resized crops (no-rebin).

**Known shortcut risk: band misregistration.** v1 stamps carry per-galaxy sub-pixel g/r/i offsets
(AA3a; `TODO.md:447-455`). Crops preserve them, so instance discrimination could key on them.
Default (F6): measure, not mitigate. Probe the band offsets from the MoCo encoder against M's.

## 4. MAE (reconstructive)

### 4.1 D12 amended: 16×16 patches, matching M

D12 said reproduce Wu & Walmsley: ViT ~30M, 3-layer decoder, **8×8 patches** (`DECISIONS.md` D12,
`docs/spec/objectives.md` §2). The user decided (2026-09-28) on M's 16×16 patches, and matching M
means ViT-S (21.6M); D12 now carries the amendment ("D12 amended", 2026-10-02), with the original
struck through beside it. With both, the arm is **He et al. 2022 MAE on M's backbone**, with the
decoder at 8×512 (F2, settled). The released Euclid MAE's role as a way to validate the
reimplementation shrinks to a loose sanity reference. `docs/spec/objectives.md` §2 still says 8×8
and needs its line.

### 4.2 Settings

| setting | value | source / justification |
|---|---|---|
| mask ratio | **0.75**, uniform random per sample (64 visible, 192 masked, exact) | He et al. 2022 (arXiv 2111.06377) §4, Table 1b |
| encoder input | visible tokens only: `patch_embed_tokens` → gather → `run_tokens` | He §3; mask tokens live in the decoder only, so the saved encoder is a plain `VisionTransformer` |
| decoder | **8 blocks × 512 wide, 16 heads** (decided, user, 2026-09-28; F2) | He's default. He Table 1a: linear-probe accuracy rises with decoder depth (*verify*: about 65.5 at depth 1, 73.5 at 8), because a deep decoder absorbs the pixel specialisation. The read-out here is a frozen linear probe, so a shallow decoder would handicap MAE exactly where it is measured. |
| decoder build | Linear(384 → 512), mask token, 2-D sin-cos positions, `_Block`s, norm, Linear(512 → 16·16·3 = 768) | JEPA's `Predictor` structure (`jepa.py:59-96`) with a pixel head over all 256 positions |
| target | the cached normalised stamp, patchified; **norm-pix** (per-patch mean and variance, eps 1e-6) | He Table 1c (default, better); the sky hazard is below |
| loss | MSE on masked patches only | He §3 |
| augmentation | **none** | §2. He Table 1e reports a linear-probe cost for dropping crops (*verify*): a named handicap. |
| read-out | `encode()` over **all 256 tokens, unmasked**, block index 10, mean-pooled | He's linear probe uses all patches; `core/encoder.py:21-23` already expects MAE's final block to specialise |

**Sky hazard.** M's target sampling is biased onto the galaxy (β = 0.5, `docs/masking.md`); MAE's
is uniform, so most masked patches are sky, and norm-pix rescales each near-flat sky patch's noise
to unit variance. Much of the loss becomes irreducible noise prediction. This is the published
method and the default keeps it; the rehearsal logs the sky/galaxy split of the loss (the
`petrosian_box` token mask). Plain-pixel MSE is fork F5.

## 5. Compute estimate

### 5.1 Analytic forward FLOPs per image

Per block a ViT costs N·(24d² + 4Nd); training is 3× forward, a no-grad pass 1×. M's context length
comes from sampling M's masker (β = 0.5, synthetic Petrosian radii, 300 batches of 32): **81 context
tokens** (5–95% range 61–102), 180 target tokens.

| arm | composition | GFLOP / image | ratio to M |
|---|---|---|---|
| **M** | context encoder 81 tokens (train) + target encoder 256 (fwd) + predictor 192×6 over 261 (train) | 28.5 | 1.00 |
| MoCo, asymmetric, windows 10–14 (mean 146 tokens) | query encoder (train) + key encoder (fwd) | 26.7 | **0.94** |
| MoCo, symmetrised, windows | 2 × (train + fwd) | 53.4 | 1.88 |
| MoCo, asymmetric, 224 px (196 tokens) | | 36.6 | 1.29 |
| **MAE, decoder 8×512** | encoder 64 tokens (train) + decoder 256 (train) | 50.7 | **1.78** |
| MAE, decoder 3×512 | | 24.5 | 0.86 |
| MAE, decoder 3×256 | | 13.1 | 0.46 |

The projector, the queue logits (128 × 65,536: about 17 MFLOP forward per image, under 0.1% of
26.7 GFLOP) and the EMA update are negligible or common to M. The queue itself is 65,536 × 128 fp32,
32 MiB.

### 5.2 Wall-clock on this M3 Pro (MPS, fp32, batch 32)

Base: M's measured 1.4448–1.4586 steps/s over the long segments (`m_findings.md:127-130`) and O2's
1.42 (`o_findings.md:348`). O2's "9.88 h to step 101,308" is its final 50,654-step segment; the
whole run is about 19.8 h. Model: t = f·t_M + (1 − f)·t_M·ratio, fixed per-step overhead f
bracketed in [0, 0.3].

| arm | steps/s | h per seed (101,308 steps) |
|---|---|---|
| M (measured) | 1.42–1.45 | 19.3–19.8 |
| **MoCo asymmetric, windows** (recommended) | 1.48–1.54 | 18.3–19.0, **+5–15% for on-device augmentation → ~19–22 h** |
| MoCo symmetrised | 0.76–0.90 | 31–37 |
| **MAE 8×512** (decided, F2) | 0.80–0.93 | **30–35** |
| MAE 3×512 (not taken) | 1.57–1.68 | 17–18 |

The accepted cost of the 8×512 decoder: 30–35 h per seed against 17–18 h for 3×512, **about +12–18 h
per seed** (the "~15 h" of the decision), before the ±30% model uncertainty; about +25–35 h over the
two MAE seeds.

**Four runs, serial** (one heavy process at a time on this 18 GB machine):
- the settled pair (MoCo asymmetric, MAE 8×512): 2 × (19–22) + 2 × (30–35) = **98–114 h ≈ 4.1–4.8
  days**, plus probe pauses;
- for the record, a 3-layer MAE decoder would have been **72–80 h ≈ 3–3.3 days**;
- model uncertainty **±30%**.

Uncertainty sources:
1. FLOP share is not time share on MPS: small-N matmuls (MAE's 64-token encoder) run at lower
   utilisation than M's mix, a 512-wide decoder at higher. M runs at about 1.3 TFLOP/s effective.
2. M's host-side numpy masking is a cost MAE does not pay and MoCo pays differently.
3. M ran under heavy memory pressure (34 GB of demand in 19 GB physical, `m_findings.md:136-140`);
   the rehearsal's conditions will differ, and the machine is currently loaded by other work.
4. A one-time drop of about 5% after the first segment (`m_findings.md:132-134`).

The rehearsal replaces all of these with measurements.

## 6. The 500-step timing rehearsal (proposed, not run)

Modelled on `artifacts/rehearsal_run.py`, on the **v1** cache and freeze. Each arm is its own
subprocess, `smoke: true`, into a scratch `out_dir`, with no bake or pull running locally alongside.

1. **Reference:** M's exact recipe for 500 steps in the same session, so ratios are against today's
   machine state.
2. **MoCo** (the settled settings, §3.2), 500 steps.
3. **MAE 8×512**, 500 steps.
4. **Resume identity per baseline:** 250 steps, stop, resume to 500 in a second process; losses
   must match the straight run step for step (this exercises the queue in the checkpoint).

**Record:** median steps/s over steps 100–500; peak RSS and MPS allocation; per-phase timing
(augment/mask, forward, backward, optimiser, EMA); collapse-monitor readings; MoCo's InfoNCE against
log(K + 1) = 11.09 and its contrastive top-1; MAE's sky/galaxy loss split.

All 500 steps sit inside the 1,250-step warm-up, so the rehearsal measures timing and plumbing, not
learning. Cost: about 6–12 min per arm, under 1 h in all.

**Gate before launch:** if MoCo's top-1 is near 1.0 by step 500, the pretext task is too easy (a
shortcut, possibly the band offsets, F6); revisit §3.4 before committing about 40 h.

## 7. Where matching M was impossible

| # | what | why it cannot match | proposed |
|---|---|---|---|
| 1 | MoCo needs augmentation; M has none | contrastive learning without views is undefined | the minimal objective-intrinsic set (§3.4); no flips, rotations or resampling |
| 2 | negatives at batch 32 | v3's in-batch design gives 31 | v2's queue, K = 65,536 (decided; keys 2,048 steps old at the oldest, §3.2) |
| 3 | MoCo momentum | M's 0.996 has no queue behind it | 0.999 (decided); constant as v2 or on M's ramp to 1.0 still open (F3) |
| 4 | view size | MoCo trains on 100–196-token windows, reads out 256 | grid-aligned windows, positions a subset of the read-out grid (M trains on ~81 tokens and reads 256; MAE trains on 64) |
| 5 | pooling into the loss | v3 uses a CLS token; there is none | mean-pooled final block into the projector |
| 6 | MAE decoder capacity | M has no decoder; its predictor is 192×6 | He 8×512 (decided, F2) |
| 7 | masking distribution | M's bbox-biased multi-block masking belongs to its objective | uniform (published); log the sky share of the loss (F5) |
| 8 | patch size / encoder vs D12 | D12 specified 8×8, ~30M | 16×16 ViT-S by the user's decision; **D12 amended** (§4.1) |
| 9 | MAE augmentation | He's default uses resized crops and flips | none, as M (D10, no-rebin); He's linear-probe cost named |
| 10 | optimiser details | MAE: β2 0.95, wd 0.05, blr 1.5e-4·bs/256; v3: wd 0.1. At batch 32 those give 1.9e-5 (linear scaling) to ~2.1e-4 (sqrt from 4,096) | M's 1.25e-4, wd 0.04, β2 0.999: within the published family; stability checked at rehearsal (F7) |
| 11 | FLOPs / wall-clock | objectives differ in cost per image | exposure and schedule matched instead (§1.1); reported "per image seen" |
| 12 | schedule completion | published recipes anneal to completion; M stopped at 65.9% of peak LR | inherit M's truncation; P4 applies to all arms |
| 13 | collapse kill criterion | the soft floor 2.5 is grounded on JEPA traces only (`pretrain.yaml` `collapse_floor`) | F8 |
| 14 | band misregistration | v1 stamps carry it (`TODO.md:453`) | train on v1 to match M; measure the shortcut (F6) |

## 8. Forks for the user

- **F1** — `smoke: true` for both baselines, as for M (**recommended**): the D12 comparison on M is
  exploratory until a headline run exists.
- **F2** — *Settled (user, 2026-09-28):* MAE decoder **8 blocks × 512 wide, 16 heads**, He's
  standard configuration, which matters for the linear-probe read-out; the extra cost is accepted
  (30–35 h per seed against 17–18 h for 3 layers, about +12–18 h per seed, §5.2). The 3-layer
  alternative (D12 / Wu & Walmsley, about M's cost) is not taken.
- **F3** — *Settled (user, 2026-09-28):* MoCo v2-style settings for a small batch on one GPU,
  **queue K = 65,536, momentum 0.999, τ = 0.2** (Chen, Fan, Girshick & He 2020, arXiv:2003.04297;
  K and m from He et al. 2020, arXiv:1911.05722; §3.2). This replaces the draft's K = 8,192 and the
  0.996 / K 2,048 alternative; keys up to 2,048 steps old at batch 32 are a known consequence
  (§3.2). *Open residue:* m constant at 0.999 as in v2, or 0.999 → 1.0 on M's cosine ramp as the
  draft proposed (§3.3).
- **F4** — MoCo loss: **asymmetric (v2, recommended)** vs symmetrised (v3, about 2× the cost).
- **F5** — MAE target: **norm-pix (recommended)** vs plain-pixel MSE.
- **F6** — band-offset shortcut: **measure only (recommended)** vs an integer ±1 px per-band shift
  in MoCo's views.
- **F7** — LR: **M's 1.25e-4 (recommended)** vs the published scaling rules.
- **F8** — collapse criterion: reuse M's floors with their JEPA-only grounding stated (recommended)
  vs the hard floor only, stamping the forfeit.
- **F9** — test-time D4 averaging (`TODO.md` P2): whatever is chosen applies to all encoders alike;
  not decided here.

## 9. Build list

No MoCo or MAE code exists under `src/`. Estimates in working days of build and review.

| # | item | files | effort | what bears weight |
|---|---|---|---|---|
| 1 | Separate top-level configs `MocoRunConfig`, `MaeRunConfig` reusing `PathsConfig`, `RuntimeConfig`, `ModelConfig`, `NormalisationFreeze`, `CollapseFloorFreeze`; `configs/moco.yaml`, `configs/mae.yaml` | `harness.py` or new `objectives/config.py` | 0.5 d | **Do not widen `HarnessConfig.objective`**: `determining_dump` hashes every field including defaults (`core/config.py:212-221`), so a new field moves M off `v2:61330a0012234374` and M's checkpoints refuse to resume. A test pins M's hash. |
| 2 | Generic loop `train_objective(obj, …)`, `train_jepa` a thin wrapper; the objective supplies `loss_parts`, `trainable_parameters`, `momentum_update` (no-op for MAE), `checkpoint_modules`, `extra_state`, `encoder` | `objectives/jepa.py` → `objectives/loop.py` | 1 d | **Highest risk**: M's hot path. A JEPA regression test (tiny CPU model, 20 steps, identical losses before and after). The alternative, a second copy of the loop, duplicates the resume and stop logic. `harness.py:362-366` anticipates the second loop; the second-consumer rule now applies. |
| 3 | Generalise `TrainCheckpointer` from the fixed keys (`checkpoint.py:196-201, :309-311`) to `checkpoint_modules()` + `extra_state` (MoCo queue and pointer) | `callbacks/checkpoint.py` | 0.5 d | JEPA keeps writing the same keys, so schema 1 stays valid and M's checkpoints still load (`:323-329`). The queue is training state; left out, a resume trains on different negatives. |
| 4 | `objectives/mae.py`: seeded 75% masking, decoder from `_Block`/`_sincos_2d`, norm-pix loss | new | 0.5–1 d | The encoder holds no mask token, so `load_frozen_encoder` and `encode()` work unchanged |
| 5 | `objectives/moco.py`: projector, queue buffer, InfoNCE, momentum encoder by deepcopy (as `jepa.py:152`) | new | 0.5–1 d | Queue in the checkpoint; positive at index 0; key encoder without grad |
| 6 | Augmentation: token windows, per-band gain, blur, noise; on device; seeded from `(train_seed, step)`; no flip or rotation op exists | `objectives/moco.py` (single consumer) | 0.5–1 d | Per-step seeding keeps resume identity; a seeded `torch.Generator('mps')` to check |
| 7 | `build_objective` branch; `calibrate` generalised (it hard-codes `jepa.predictor`, `harness.py:1132`) | `harness.py` | 0.25 d | |
| 8 | Driver `artifacts/baseline_run.py` reusing m2's `Run`, segmenting, `keep` and the k2 probe subprocess; fixed stop 101,308; probes at 0.5/1/2/4 epochs for trajectories only | new artefact | 0.5 d | out-dir guard as in m2 (`m2_long_run.py:205`) |
| 9 | Rehearsal (§6) | `rehearsal_run.py` pattern | 0.25 d + ~1 h machine | |
| 10 | Tests (below) | `tests/test_objectives_moco.py`, `test_objectives_mae.py`, `test_objective_loop.py` | 1 d | |
| 11 | D-entries: D12 amendment (16×16, ViT-S, decoder; done 2026-10-02); the MoCo recipe; `docs/spec/objectives.md` §2 | docs | 0.25 d | |

**Total: about 5.5–7 working days of build**, then about 4–5 machine-days for the four runs.

Tests (invariant tier unless marked): M's `config_hash` unchanged; JEPA losses identical step for
step after the loop refactor; a baseline's encoder init byte-identical to M's for the same seed; a
baseline's data order equal to M's; no flip or rotation in the augmentation (windows are
axis-aligned token sub-grids, a chiral test image is never mirrored); MoCo queue enqueue/dequeue and
no grad on the key encoder; MAE exactly 64 visible tokens, loss only on masked patches, norm-pix
against a hand computation; the baseline checkpoint loads through `load_frozen_encoder`, passes
`assert_frozen`, and `encode()` uses all 256 tokens; integration: resume identity per baseline
(250 + 250 = 500); planted-positive overfit on a tiny fixed set.

## 10. Pre-registration still owed

The cross-objective comparison (`TODO.md` Epic G) needs its own D27 outcome enumeration and D28
planted positives before any baseline number is read. This document fixes training only.
