# Baselines — MoCo and MAE, matched to A1 (Kickoff F)

*Status: draft, 2026-10-02 (first draft 2026-09-27). Nothing hashed, nothing trained. **Retargeted
to pretrain_v2 (user, 2026-10-13):** the baselines are matched to A1 (M's recipe on the aligned v2
stamps, `configs/pretrain_v2.yaml`, freeze `246d8de6…`), not to M on v1; seed 0 pairs with A1, seed 1
with A2; σ_sky re-measured on the v2 cache; trajectory probes on E2's protocol; F6 moot on aligned
stamps; the rehearsal held (§1, §3.4, §5.3, §6, §7, §8). Ladder plan Task 5 (`docs/ladder_v2/plan.md`) was carried out on branch `baselines-v2`, commit `3767663`. **Settled
(user, 2026-09-28):** 16×16 patches for MAE (D12 amended), F2 (MAE decoder 8×512) and the MoCo
settings of F3 (K = 65,536, m = 0.999, τ = 0.2); **F3's residue settled (user, 2026-10-01 evening):
m constant at 0.999, v2-style.** **F1, F4–F8 settled (user, 2026-10-02); F9 settled (user, 2026-10-09): D4 averaging off** (§8). **F7's MoCo
question settled (user, 2026-10-03): option (b)**, MoCo v3's ViT optimiser rule on v2's objective.
Design sources: `DECISIONS.md` D10 revised (`:377`), D12 (`:414`), `docs/spec/objectives.md`, `TODO.md` Epic G
(`:467`). A choice is recorded as decided only once the user has signed it off; every open fork is
in §8. Claims marked *verify* are from memory of the primary papers and are checked against the
PDFs before sign-off.*

## 1. What "matched to A1" means

D12 holds the data fixed and varies only the objective (`DECISIONS.md:418-423`). Each baseline
keeps everything in A1's recipe that is not intrinsic to its objective. A1 is M's recipe verbatim on
the aligned pretrain_v2 stamps (`configs/pretrain_v2.yaml` is `pretrain.yaml` bar the three `paths`
and the `normalisation` block), so "M's recipe" below means A1's too. The baselines were first
matched to M on v1; the user retargeted them to A1 on 2026-10-13, so they train on the stamps every
later encoder is compared on. The matched items are below; §7 lists every place where matching was
impossible.

| quantity | A1 (`configs/pretrain_v2.yaml`; M's recipe) | MoCo | MAE |
|---|---|---|---|
| backbone | ViT-S/16: dim 384, depth 12, heads 6, mlp 4.0 (`:66-71`), 21.6M params, no CLS | same | same |
| stamp / tokens | 256 px native, 16×16 = 256 tokens | same at read-out | same |
| corpus | pretrain_v2, the aligned re-pull (826,968 stamps, none shared with probe_v2); train / monitor by `split_pretrain` (`harness._prepare`; split seed 0, the config's, for every run, `monitor_frac` 0.02): **810,491 train / 16,477 monitor** (measured 2026-10-13 from the two corpora's `metadata.csv`) | same split | same |
| normalisation | frozen v2 freeze, `content_hash 246d8de6ba9a…` (`:51`); the v2 fp16 cache under `runs/m_v2/cache` | the same block verbatim, so the same fp16 cache (`pipeline_hash` is keyed on the pipeline only, `data/cache.py:60`) | same |
| batch | 32, `drop_last` (`:84`) | 32 | 32 |
| `steps` (schedule length) | 253,270 (`:77`) | 253,270 | 253,270 |
| stop | 101,308 via `stop_after` (`objectives/jepa.py:447`; `aligned_train.py` `STOP_EPOCHS`) | 101,308 | 101,308 |
| optimiser | AdamW, torch default betas (0.9, 0.999), wd 0.04 constant, no grad clip (`jepa.py:326`) | **AdamW, torch default betas (0.9, 0.999), wd 0.1** (F7 (b): MoCo v3's ViT recipe) | **AdamW, betas (0.9, 0.95), wd 0.05** (F7, He et al.'s recipe) |
| LR | peak 1.25e-4, warmup 1,250, cosine to 1.25e-7 (`:90-94`, `jepa.py:250`) | **peak 1.875e-5** = 1.5e-4 × 32/256 (F7 (b), MoCo v3); M's warmup and cosine shape | **peak 1.875e-5** = 1.5e-4 × 32/256 (F7); M's warmup and cosine shape |
| LR at the stop | 8.24e-5 (65.9% of peak) | 65.9% of its own peak, by construction | 65.9% of its own peak |
| EMA | 0.996 → 1.0 cosine over 253,270 (`:95-96`, `jepa.py:265`); 0.99738 at the stop | momentum encoder 0.999, constant (v2; §3.3) | none (objective-intrinsic) |
| precision | fp32 (`autocast: null`, `:38`) | fp32 | fp32 |
| SIGReg | off, λ = 0 (`:125`, D21) | n/a | n/a |
| status | `smoke: true` (`:34`), so not a headline | `smoke: true` (§8, F1) | `smoke: true` |
| weight init | `seed_init(train_seed)` (`harness.py:372`) | byte-identical to A1's for the same seed: the encoder is built first from the seeded stream | same |
| data order | `ResumableShuffle(seed=train_seed)` | the identical sequence to A1's | same |
| read-out | penultimate block (`DEFAULT_LAYER = -2`: block index 10, the 11th of 12, before the final norm), mean over all 256 tokens (`vit.py:172`, `core/encoder.py:20-36`) | backbone `encode()` only; projector discarded | backbone `encode()` over all 256 tokens, no mask; decoder discarded |

"Block 11" throughout the project means the 11th block, index 10. Nobody writes `blocks[11]`.

**Checked, not trusted.** `configs/moco.yaml` and `configs/mae.yaml` are `pretrain_v2.yaml` with the
`objective` block swapped (and their own `out_dir`); `tests/test_objectives_baselines.py` pins the
equality on every determining field and names the v2 freeze. `artifacts/baseline_run.py` runs under
A1's launch environment (`aligned_launch.sh`): `GJ_CONFIG=configs/pretrain_v2.yaml`,
`F0_CACHE_BASE=…/runs/m_v2/cache`, `F0_NORM_PREFIX=246d8de6ba9a`. Unset, it sets them; set to anything
else, it refuses. F0 then checks the cache against the live v2 freeze, and the driver refuses a
config that differs from A1's outside `objective`.

**Seeds.** Two per baseline, paired to A1's and A2's: training seed 0 pairs with A1 (`runs/m_v2`),
training seed 1 with A2 (`runs/m_v2_s1`). The split seed is 0 for both, exactly as in A1 and A2.
Within a pair, the initial encoder weights and the order of stamps are identical to the JEPA run, so
the seed-0 MoCo run differs from A1 in its objective and in nothing else the harness controls. A3
(seed 2, exploratory) gets no baseline partner; pairing one would add about 50 h.

**The SIGReg caveat for D12 (`TODO.md:498`, `DECISIONS.md:945`) does not apply to A1.** A1 runs at
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
the annealing tail never ran (`o_findings.md` P4, `:466`). A1 and A2 inherit that truncation as a
fixed stop. A cosine completed over 101,308 steps would give the baselines an anneal A1 never had.
So every arm keeps `steps: 253,270` and stops at 101,308 through `stop_after`, which leaves
"`config.steps` — and so the LR and EMA schedules — untouched" (`jepa.py:319`). P4's cooldown
question then applies to all three arms equally.

**The stop is fixed, not a rule.** M's 4-epoch stop was chosen by an AUC-reading rule
(`m_findings.md:162`); A1 and A2 took its step count as a fixed number. The baselines do the same
and apply no rule of their own, so no baseline checkpoint is chosen by looking at labels (spec 1C).
They are probed at M's epoch points (0.5, 1, 2, 4) for trajectories only; those probes are
reported, never used to select a checkpoint.

**Trajectory probes: E2's protocol (user, 2026-10-13).** m2's k2 probe (`k2_trajectory_probe.py`)
is a v1 like-for-like guard: it asserts J4(A)'s split counts and so refuses probe_v2. The trajectory
probes use E2's protocol instead, read unchanged from `aligned_c2.py` as ladder v2's E2 entry reads
it: the encoder's full-split bank (`embed`: uncapped probe-train plus probe-test), `fit_scores`
(`_fit`, C from `probe.yaml`), then per answer the held-out `auc` and a 95% interval from `boot_aucs`
under `weights` (2,000 Poisson galaxy draws, seed fixed there). **All 37 Scheme-1 answers, no powered
filter, no multiplicity correction, and no aggregation over answers** — in particular none over an
M-derived list such as `powered`. `baseline_run.py` calls `artifacts/baseline_trajectory_probe.py` at
each probe point with one checkpoint; it re-exports the checkpoint through `frozen_from`, so
`load_frozen_encoder` stays the only way in, writes a bank per checkpoint (fp32 × 384 over the probe-train and probe-test union: about 0.3 GB
each by arithmetic, so about 5 GB over the four runs' 16 probe points) and
`artifacts/out/<arm><i>_trajectory.json`. The driver logs how many answers were scored, never an
AUC.

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
about two time constants stale, against about 0.26 in v2; m is held constant (§3.3), so this holds
throughout the run. The decision keeps the published
values; the rehearsal (§6) and the InfoNCE trace are where an effect of stale negatives would first
show.

### 3.3 Momentum encoder

**m = 0.999, decided (user, 2026-09-28).** The value is v1's, kept in v2 (MoCo v1 Table 4: about
0.999 best with a queue, 0.9 failing; *verify*). M's 0.996 is I-JEPA's value and has no queue behind
it: at 0.996 the time constant is 250 steps, so 65,536 keys at batch 32 (2,048 steps) would be
about eight time constants stale. The draft's matched alternative (0.996 with K = 2,048) is not
taken.

**Schedule: constant, decided (user, 2026-10-01 evening).** m stays at 0.999 for the whole run, as
in v1 and v2 (*verify*). The draft's alternative, 0.999 → 1.0 cosine over 253,270 steps (0.99935 at
the stop, v2's start on M's ramp shape), is not taken.

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
2. **Gain** on the normalised tensor, per image and view: a **global** gain ×U(0.9, 1.1), the same
   for all bands, and a **per-band** gain ×U(0.98, 1.02), independent per band (settled 2026-10-13,
   below).
3. **Gaussian blur**, σ ∈ [0.1, 2.0] px at p = 0.5: v2's aug+ (from SimCLR), standing in for
   Hayat's varying PSF. SDSS seeing ≈ 1.4″ ≈ σ 1.5 px, so σ up to 2 px roughly doubles the PSF.
4. **Additive Gaussian noise** (Hayat): σ_aug = u × σ_sky per band, u ~ U(0, 1) per image and view
   (settled 2026-10-13, below).

Dropped: flips and rotations (D10); grayscale (destroys colour); hue and saturation (meaningless for
gri); resized crops (no-rebin).

**Settled (user, 2026-10-13): gain and noise.**

- Global gain ×U(0.9, 1.1), applied identically to all bands: it changes brightness, not colour.
- Per-band gain ×U(0.98, 1.02), independent per band: about SDSS's photometric calibration, and small
  enough to keep colour information.
- Additive Gaussian noise σ_aug = u × σ_sky per band, u ~ U(0, 1).
- No translation between bands and no offset augmentation (F6).
- **Reason.** Augmentations define MoCo's invariances. A ±10% per-band gain (the earlier proposal)
  would make MoCo invariant to colour, a strong morphology signal, and so handicap it on exactly what
  is probed.
- **σ_sky, measured on the cache the baselines train on** (`artifacts/sky_sigma.py` →
  `artifacts/out/sky_sigma_v2.json`, 2026-10-13): A1's v2 fp16 cache (asinh q = 4, frozen v2 Normalise
  `246d8de6…`, read through F0 under `baseline_run.py`'s environment); 2,000 stamps from the
  pretrain_v2 train split (numpy seed 20261013); each stamp's outer 16-px frame (≥ 44″ from the
  centre); per stamp and band a 1.4826 × MAD σ after iterative 3σ clipping; the median over stamps.
  **g 0.395, r 0.466, i 0.598** normalised units, in `MocoConfig`, `MocoObjectiveConfig` and
  `configs/moco.yaml`; the spread, sky level and clipped fraction are in the record. The method is the
  first measurement's, unchanged; that one, on M's v1 cache (`75100066…`, `sky_sigma_v1.json`), is
  superseded with the retarget. The frozen Normalise std is dominated by sky pixels, so σ_sky is a
  large fraction of a unit: at u = 1 a view carries √2 × the native sky noise.
- **No rotations and no flips: D10 revised stands (user, 2026-10-13).** The decision first listed
  rotations by multiples of 90° and flips (citing D4, which is the from-scratch decision); asked
  against D10 revised (2026-09-23), which rules out rotation and reflection for M and every D12
  baseline (its reason 4: one symmetry policy across the encoder family) and whose flips would erase
  handedness, the user kept D10. The code has no flip or rotation op, and a test pins that.

**Band misregistration: moot on aligned stamps (user, 2026-10-13).** v1 stamps carry per-galaxy
sub-pixel g/r/i offsets (AA3a; `TODO.md:447-455`), which crops preserve, so on v1 instance
discrimination could have keyed on them; F6 was to measure that shortcut. The baselines now train on
pretrain_v2, the aligned re-pull, so the shortcut F6 named is not in the data and its measurement is
dropped (§8, F6). The rehearsal's top-1 gate (§6) stays, as a check that the pretext task is not too
easy for any reason.

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
| decoder init | **He et al.'s** (user, 2026-10-13; replaces the earlier PyTorch-default deviation): xavier_uniform on every Linear with zero bias, LayerNorm weight 1 / bias 0, the mask token normal(0.02); fixed sin-cos decoder positions. The encoder keeps M's init, byte-identical for the same seed | `facebookresearch/mae` `models_mae.py`, `MaskedAutoencoderViT.initialize_weights` and `_init_weights` |
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

### 5.3 Total compute and order after A1 and A2 (user, 2026-10-02; pairing retargeted 2026-10-13)

Serial, one heavy process at a time. Rates as §5.2 (±30% model uncertainty until the rehearsal
measures them). A1 and A2 are M's recipe on the v2 cache, so they cost M's 19.3–19.8 h each; the
baselines train on the same cache and are paired with them (§1, Seeds), so the retarget moves no
cost.

| # | job | steps | h (range) | why here |
|---|---|---|---|---|
| — | A1, A2 (aligned v2, seeds 0 and 1) | 2 × 101,308 | 39–40 | trained under Kickoff E (`aligned_train.py`); the pairing partners, not part of this budget |
| 1 | rehearsal (§6): A1-recipe reference, MoCo, MAE, resume identity | 3 × 500 + resumes | ~1 | timing and plumbing; the top-1 pretext-ease gate before anything long |
| 2 | F7 check, MoCo: 1/3×, 1×, 3× the rule's rate | 3 × 5,000 | ~2.8 (at ~1.5 steps/s) | fixes MoCo's LR before its seed runs; cheaper arm first |
| 3 | F7 check, MAE: the same three rates | 3 × 5,000 | ~4.8 (at ~0.87 steps/s) | fixes MAE's LR |
| 4 | **MoCo seed 0** (pairs with A1, `runs/m_v2`) | 101,308 | 19–22 | first full D12 contrast at the earliest point; the cheaper arm, so a fault shows early |
| 5 | **MAE seed 0** (pairs with A1) | 101,308 | 30–35 | completes the seed-0 ladder (A1, MoCo, MAE) |
| 6 | **MoCo seed 1** (pairs with A2, `runs/m_v2_s1`) | 101,308 | 19–22 | second seed: the J1–J2-style spread for MoCo |
| 7 | **MAE seed 1** (pairs with A2) | 101,308 | 30–35 | second seed for MAE |

- **Baselines in all (rows 1–7): about 107–123 h, ≈ 4.5–5.1 days** of machine time (±30%), plus probe
  pauses (the E2 trajectory probe subprocess at 0.5/1/2/4 epochs per run, §1.1, §9 item 8).
- **A1 and A2 are already trained**, so their 39–40 h is not owed; a baseline partner for A3 (seed
  2, exploratory) would add about 50 h and is not planned.
- **Build time is separate:** about 5.5–7 working days (§9), which can run while A1 and A2 train;
  the F7 checks need build items 1–8, so the rows above start when both A2 and the build are done.
- **Order rule:** seed-0 pair before seed-1 pair, so one complete objective ladder exists before any
  second seed is spent; within a pair, MoCo before MAE (cheaper, and its rehearsal gate is the one
  that can stop a run). Row 2's rates are MoCo v3's ViT rule (F7 (b), settled 2026-10-03):
  6.25e-6, 1.875e-5 and 5.625e-5.

## 6. The 500-step timing rehearsal (run 2026-10-17; gate passed)

**Released (user, 2026-10-17) and run 14:26–15:10** by `artifacts/baseline_rehearsal.py`, record
`runs/_rehearsal_baselines/rehearsal.json` (SHA-1 `6fb332be32f9`). All nine child processes exited 0.

| arm | steps/s (median, steps 100–500) | × ref | 5,000 steps | 101,308 steps | peak RSS | MPS live / driver peak | resume 250 + 250 |
|---|---|---|---|---|---|---|---|
| reference (A1's recipe) | 1.454 | 1.00 | 0.96 h | 19.4 h | 1.23 GB | 0.85 / 13.26 GB | identical, step for step |
| MoCo | 1.736 | 1.19 | 0.80 h | 16.2 h | 2.20 GB | 2.03 / 7.86 GB | identical, step for step |
| MAE 8×512 | 1.052 | 0.72 | 1.32 h | 26.7 h | 0.98 GB | 1.91 / 7.72 GB | identical, step for step |

- **Phases** (median ms per step, from the resumed child, steps 250–500, a device sync either side of
  each phase): reference — data 62, mask 30, forward 353, backward 254, optimiser 72, EMA 4; MoCo —
  data 76, augment 48 (inside forward), forward 257, backward 267, optimiser 138, momentum 6; MAE —
  data 59, mask 1, forward 305, backward 557, optimiser 134.
- **Collapse monitor at step 400:** reference std 0.600, effective rank 13.0, mean cosine 0.867; MoCo
  0.444, 19.2, 0.822; MAE 0.376, 18.5, 0.934. No halt.
- **MoCo:** InfoNCE (mean of the last 50 steps) 9.18 against log(K + 1) = 11.09. Top-1 by 100 steps:
  0.106, 0.019, 0.014, 0.018, 0.018 (the first block is the queue's initial random keys).
- **The gate** (fixed in the driver before the run: mean top-1 over steps 450–500 ≥ 0.95 stops the
  sequence): **0.017 — pass.** The pretext is not too easy.
- **MAE's sky/galaxy split (F5)** at step 499, masked tokens only, galaxy = inside A1's Petrosian box:
  per-token loss galaxy 0.998, sky 0.999; the sky's share of the masked loss 0.898, equal to its share
  of masked tokens (0.898). At this point in the warm-up the loss is spread evenly per token, so the
  sky carries the loss in proportion to its area. Descriptive (F5 settled norm-pix); no gate.
- **After the gate (MoCo seed 0, recorded 2026-10-20; user: continue as planned).** The gate passed at
  step 500, but MoCo's contrastive top-1 (one batch of 32, logged at each segment end) then reached
  0.938 at step 12,664 and **1.000 at 25,327, 50,654 and 101,308**: by ranking, the two views of a
  galaxy are told apart from all 65,536 queued keys every time. The run did not stall: InfoNCE
  (500-step means) fell from 8.55 (steps 0–500) through 6.80 (12,664), 6.51 (25,327), 6.36 (50,654)
  to 6.28 (101,308), against log(K + 1) = 11.09, so the softmax was still sharpening while the
  argmax was already right. This is the "too easy" the gate was written for, arriving after the
  window the gate looked at. It is recorded as a property of this baseline under §3.4's mild,
  symmetry-free augmentation; MoCo is judged by its trajectory probes, never by its pretext top-1.
  - **The pre-registered gate passed** (mean top-1 0.017 over steps 450–500, above); saturation
    came later, by 1 epoch, and the gate did not cover it.
  - **The augmentation is unchanged, by decision (user, 2026-10-20).** It was set on physical grounds
    (§2, §3.4). Stronger augmentation would add invariances the design excludes: orientation (flips and
    rotations, D10 revised) and colour (a wider per-band gain, grayscale or hue jitter, against the
    colour argument settled 2026-10-13). It is not tuned to the pretext's difficulty.
  - **For interpretation.** With augmentation this mild, the crop, gain, blur and noise leave much of
    a galaxy's low-level identity intact across its two views. Instance discrimination may therefore
    rest on low-level identity cues, such as field stars and the position of the galaxy and its
    neighbours in the frame, rather than on morphology. A saturated top-1 does not say which. **The
    trajectory probes are the measure of what MoCo learned;** its pretext top-1 is not.
- **Against §5.2's model:** MoCo is faster than modelled (16.2 h, not 19–22) and MAE too (26.7 h, not
  30–35). The four seed runs come to about 86 h at these rates, before probe pauses.

The plan as proposed (kept for the record):

Modelled on
`artifacts/rehearsal_run.py`, on **A1's v2 cache and freeze** (`baseline_run.py`'s environment, §1).
Each arm is its own subprocess, `smoke: true`, into a scratch `out_dir`, with no bake or pull running
locally alongside.

1. **Reference:** A1's recipe (`configs/pretrain_v2.yaml`: M's recipe on the v2 cache) for 500 steps
   in the same session, so ratios are against today's machine state.
2. **MoCo** (the settled settings, §3.2), 500 steps.
3. **MAE 8×512**, 500 steps.
4. **Resume identity per baseline:** 250 steps, stop, resume to 500 in a second process; losses
   must match the straight run step for step (this exercises the queue in the checkpoint).

**Record:** median steps/s over steps 100–500; peak RSS and MPS allocation; per-phase timing
(augment/mask, forward, backward, optimiser, EMA); collapse-monitor readings; MoCo's InfoNCE against
log(K + 1) = 11.09 and its contrastive top-1; MAE's sky/galaxy loss split.

All 500 steps sit inside the 1,250-step warm-up, so the rehearsal measures timing and plumbing, not
learning. Cost: about 6–12 min per arm, under 1 h in all.

**Gate before launch: a pretext-ease check.** If MoCo's top-1 is near 1.0 by step 500, the pretext
task is too easy, whatever the cause; revisit §3.4 before committing about 40 h. The gate was first
written against one named shortcut, v1's band offsets (F6); on the aligned v2 stamps that shortcut is
not in the data (§3.4), so the gate no longer names a mechanism and stays as the generic check.

### 6.1 The four seed runs (run 2026-10-18 to 2026-10-23; all complete)

§5.3 rows 4–7. They ran serially under the heavy-job lock as one chain:
`runs/_seed_chain.sh`, which calls `baseline_run.py <arm>` with `GJ_CHECKPOINT_EVERY=2000` and stops at
the first arm that fails. The log is `runs/_seed_chain.log`.
- **Learning rate:** the rule's rate, 1.875e-5, for both arms, as F7 chose (§8).
- **Code:** at `41e97a9`. No commit has touched `src/`, `configs/`, `baseline_run.py` or
  `baseline_trajectory_probe.py` since.
- **Outcome:** all four reached step 101,308 with no halt, no non-finite loss and no watchdog alert.

| run | pairs with | start → end | wall-clock (of it training) | `encoder.pt` SHA-1 | `ckpt_000101308.pt` SHA-1 |
|---|---|---|---|---|---|
| `runs/moco_s0` | A1 | 18 Oct 06:32 → 19 Oct 02:56 | 20 h 24 min (18.2 h) | `bf0f0f60cae41ec48acb075bcf30d46009f02a54` | `05963418539f30e084bf0cf8b264f157a01cd92f` |
| `runs/mae_s0` | A1 | 19 Oct 02:56 → 20 Oct 14:39 | 35 h 44 min (31.1 h) | `fb3d03d1bd476e1b65d171ef8fbd6f6e35250da8` | `46097653f7fc6fa0c7b303dd41247ebb8fd5c2ed` |
| `runs/moco_s1` | A2 | 20 Oct 14:39 → 21 Oct 17:10 | 26 h 30 min (24.1 h) | `f5e2a9f6fd5a5a0a2583216d3ae2f14248c3417a` | `ea758c0c01ee740ed5492ecb0a5b354de389b553` |
| `runs/mae_s1` | A2 | 21 Oct 17:10 → 23 Oct 03:01 | 33 h 51 min (31.6 h) | `f375c9e530f78983d81bb0d3b7e6d135b5663778` | `248574a5fdc9e3f8c50dcd977025d2b712521c70` |

**The whole chain:** 18 Oct 06:32 → 23 Oct 03:01, about 116.5 h. The rehearsal had projected about
86 h of training. The excess comes mostly from other jobs sharing the machine (load averages of
30–57 at times); the rest is the probe pauses and the restarts between segments.

**Pretext traces at each probe point.** Each figure is the mean of the 500 steps before the point,
read from each run's `traces.json`. MoCo's loss is InfoNCE, against log(K + 1) = 11.09; MAE's is the
norm-pix masked-patch loss. MoCo's top-1 is one batch of 32, logged at each segment's end. Segment
speed is in steps/s.

| step (epochs) | MoCo s0 loss / top-1 / speed | MoCo s1 loss / top-1 / speed | MAE s0 loss / speed | MAE s1 loss / speed |
|---|---|---|---|---|
| 12,664 (0.5) | 6.801 / 0.938 / 1.79 | 6.789 / 0.906 / 1.12 | 0.959 / 1.10 | 0.960 / 0.98 |
| 25,327 (1) | 6.509 / 1.000 / 1.27 | 6.501 / 1.000 / 1.12 | 0.945 / 1.04 | 0.947 / 0.66 |
| 50,654 (2) | 6.358 / 1.000 / 1.40 | 6.351 / 1.000 / 1.03 | 0.943 / 0.86 | 0.943 / 0.90 |
| 101,308 (4) | 6.283 / 1.000 / 1.66 | 6.279 / 1.000 / 1.27 | 0.940 / 0.86 | 0.941 / 0.94 |

**Trajectory probes.** These are the E2 protocol on ladder v2's identical split, 37 answers each, in
`artifacts/out/<run><i>_trajectory.json` with i = 1–4 for 0.5, 1, 2 and 4 epochs. All 16 probes ran.
Their values have **not been read**. They stay unread until the §10 pre-registration (D27/D28) is
hashed. The SHA-1s (first 12 hex digits) are recorded so that what is read later is what was written:

| run | i = 1 | i = 2 | i = 3 | i = 4 |
|---|---|---|---|---|
| `moco_s0` | `f704d7b7482a` | `96b2fe92be89` | `43620ad7ebae` | `2f09f95583c6` |
| `mae_s0` | `30f12cd20a20` | `ed70365f8f5b` | `e841aa0e330e` | `ab340ad11339` |
| `moco_s1` | `1b7f6799ae41` | `f347544d196d` | `e3576ff5399d` | `556e404d9452` |
| `mae_s1` | `83b3846ea79f` | `5f9be0065a4a` | `3dad1148bdf7` | `650943208b93` |

## 7. Where matching A1 was impossible

A1 is M's recipe on v2 (§1), so "M" in this table reads as A1 as well.

| # | what | why it cannot match | proposed |
|---|---|---|---|
| 1 | MoCo needs augmentation; M has none | contrastive learning without views is undefined | the minimal objective-intrinsic set (§3.4); no flips, rotations or resampling |
| 2 | negatives at batch 32 | v3's in-batch design gives 31 | v2's queue, K = 65,536 (decided; keys 2,048 steps old at the oldest, §3.2) |
| 3 | MoCo momentum | M's 0.996 has no queue behind it | 0.999, constant as v2 (decided, F3) |
| 4 | view size | MoCo trains on 100–196-token windows, reads out 256 | grid-aligned windows, positions a subset of the read-out grid (M trains on ~81 tokens and reads 256; MAE trains on 64) |
| 5 | pooling into the loss | v3 uses a CLS token; there is none | mean-pooled final block into the projector |
| 6 | MAE decoder capacity | M has no decoder; its predictor is 192×6 | He 8×512 (decided, F2) |
| 7 | masking distribution | M's bbox-biased multi-block masking belongs to its objective | uniform (published); log the sky share of the loss (F5) |
| 8 | patch size / encoder vs D12 | D12 specified 8×8, ~30M | 16×16 ViT-S by the user's decision; **D12 amended** (§4.1) |
| 9 | MAE augmentation | He's default uses resized crops and flips | none, as M (D10, no-rebin); He's linear-probe cost named |
| 10 | optimiser details | each method's recipe is tuned with its own optimiser and LR rule | **each method's published LR rule at batch 32, with its own optimiser form** (F7, settled): MAE AdamW (0.9, 0.95), wd 0.05, 1.875e-5; MoCo AdamW (MoCo v3's ViT rule), wd 0.1, 1.875e-5. A 5,000-step check at 1/3×, 1× and 3× the rule's rate, judged on pretext loss only. **Not taken from v3's ViT recipe** (review of PR #3, 2026-10-13): `--stop-grad-conv1`, v3's frozen random patch projection, its stability fix for ViTs (v3 §4–5); the patch embedding trains as M's does. v3's `--moco-m-cos` and 40-epoch warmup are replaced by F3 and M's schedule. If the F7 check shows instability, the frozen patch projection is the named lever |
| 11 | FLOPs / wall-clock | objectives differ in cost per image | exposure and schedule matched instead (§1.1); reported "per image seen" |
| 12 | schedule completion | published recipes anneal to completion; M, and A1 and A2 after it, stopped at 65.9% of peak LR | inherit the truncation; P4 applies to all arms |
| 13 | collapse kill criterion | the soft floor 2.5 is grounded on JEPA traces only (`pretrain_v2.yaml` `collapse_floor`, as `pretrain.yaml`'s) | F8 |
| 14 | band misregistration | v1 stamps carry it (`TODO.md:453`); the first plan trained on v1 to match M and measured the shortcut (F6) | **moot (user, 2026-10-13):** every arm trains on A1's aligned v2 stamps, so the offsets are not in the data and F6's measurement is dropped; the rehearsal's top-1 gate stays as a pretext-ease check (§6) |
| 15 | trajectory probes | m2's k2 probe is a v1 like-for-like guard and refuses probe_v2 | E2's protocol (`baseline_trajectory_probe.py`, §1.1): per-answer held-out AUC on the full split, all 37 answers, no powered filter, no aggregate |

## 8. Forks for the user

**Settled by the user, 2026-10-02** (F2 and F3 earlier). Every arm carries these; they are fixed before
any baseline trains, and no probe number enters any of them.

- **F1** — *Settled:* `smoke: true` for both baselines, as for M and A1. The D12 comparison on A1
  is exploratory until a headline run exists.
- **F2** — *Settled (user, 2026-09-28):* MAE decoder **8 blocks × 512 wide, 16 heads**, He's
  standard configuration, which matters for the linear-probe read-out; the extra cost is accepted
  (30–35 h per seed against 17–18 h for 3 layers, about +12–18 h per seed, §5.2). The 3-layer
  alternative (D12 / Wu & Walmsley, about M's cost) is not taken.
- **F3** — *Settled (user, 2026-09-28):* MoCo v2-style settings for a small batch on one GPU,
  **queue K = 65,536, momentum 0.999, τ = 0.2** (Chen, Fan, Girshick & He 2020, arXiv:2003.04297;
  K and m from He et al. 2020, arXiv:1911.05722; §3.2). This replaces the draft's K = 8,192 and the
  0.996 / K 2,048 alternative; keys up to 2,048 steps old at batch 32 are a known consequence
  (§3.2). *Residue settled (user, 2026-10-01 evening):* m **constant** at 0.999 as in v2, not M's
  cosine ramp to 1.0 (§3.3).
- **F4** — *Settled:* MoCo loss **asymmetric** (v2): query = view 1, key = view 2. Not v3's
  symmetrised loss (about 2× the cost, §5.1).
- **F5** — *Settled:* MAE target **norm-pix** (He et al. 2022, Table 1c). The sky hazard (§4.2) is
  accepted; the rehearsal logs the loss's sky/galaxy split.
- **F6** — *Settled (2026-10-02):* **measure only**; no anti-offset augmentation. Every encoder faces
  the same nuisance (v1's band misregistration), so none is given a remedy the others lack. **Moot
  (user, 2026-10-13):** the baselines train on the aligned v2 stamps (§1), which do not carry the
  offsets, so there is no shortcut to measure and F6's measurement is dropped. Still no anti-offset
  augmentation. The rehearsal's top-1 gate (§6) stays as a pretext-ease check.
- **F7** — *Settled:* **each method's published learning-rate scaling rule at our batch size (32)**,
  stated with its source, in the method's own optimiser form; M's warmup (1,250 steps) and cosine
  shape over 253,270 steps, stopped at 101,308, for every arm (§1).
  - **MAE** (He et al. 2022, arXiv:2111.06377, App. A.1; code `facebookresearch/mae`, `PRETRAIN.md`
    and `main_pretrain.py`): AdamW, betas (0.9, 0.95), weight decay 0.05 (0 on biases and norms, as
    the code does), base lr 1.5e-4, rule **lr = blr × batch / 256**. At batch 32: **1.875e-5**.
    (The code's default `--blr` is 1e-3; the recipe passes 1.5e-4 explicitly. Verified 2026-10-03:
    code and `PRETRAIN.md` verbatim; the paper's table through a summary only.)
  - **MoCo (v2): the published recipe is SGD, not AdamW.** MoCo v2 reuses MoCo v1's
    hyper-parameters ("the same hyper-parameters … and codebase as MoCo"); MoCo v1 (He et al. 2020,
    arXiv:1911.05722, §4): SGD, momentum 0.9, weight decay 1e-4, lr 0.03 at batch 256; v2 adds the
    cosine schedule. **Neither paper states a scaling rule for the recipe:** v1 invokes linear scaling
    only for its end-to-end ablation, though its IG-1B run (0.12 at batch 1,024) is consistent with
    it. The official repository could not be read (its contents return empty through the API, its
    pages 404), so the README's command line is unverified. Linear scaling at batch 32 gives
    **0.00375 (SGD)**.
    - **Settled (user, 2026-10-03): option (b), a declared hybrid.** MoCo's optimiser and LR follow
      **MoCo v3's rule for ViTs** (Chen, Xie & He 2021, arXiv:2104.02057; `facebookresearch/moco-v3`
      `CONFIG.md` and `main_moco.py`): **AdamW, base lr 1.5e-4 × batch/256 = 1.875e-5 at batch 32,
      weight decay 0.1**, betas torch's default (0.9, 0.999), which is what MoCo v3's code uses (it
      passes none; the paper states none). The **objective stays v2's** (F3): queue K = 65,536,
      momentum 0.999 constant, τ = 0.2, asymmetric loss (F4), no predictor.
      - **Why the hybrid.** v2's own recipe is SGD at batch 256 on ResNets. SGD-era recipes train
        ViTs unstably, which is why MoCo v3 moved ViTs to AdamW (v3 §4–5); our backbone is a ViT-S.
        The objective is v2's because at batch 32 the queue is what gives MoCo negatives (§3.1).
      - **Not taken:** (a) SGD at 0.00375 (MoCo v1/v2's 0.03 at batch 256, scaled linearly).
      - The F7 sanity check stands for both baselines: MoCo's three rates are 6.25e-6, 1.875e-5 and
        5.625e-5; MAE's the same three (its rule gives the same 1.875e-5).
  - **Sanity check** (per arm, before any seed run): about **5,000 steps** at the rule's rate, at
    **3×** and at **1/3×**, from the seed-0 init and data order, `smoke: true`, scratch `out_dir`.
    **Chosen by pretext loss only** (MAE's masked-patch loss; MoCo's InfoNCE), **never probe AUC**.
    The run departs from the rule's rate **only if that rate is unstable** (divergence, NaN, or a
    collapse-floor breach under F8); then the nearest stable of the three, stated. Recorded: the
    loss curves, the final 500-step mean loss at each rate, any instability, and the choice.
    **Results (run 2026-10-18 00:02–06:31, `artifacts/baseline_f7.py`, record `runs/_f7/f7.json`,
    SHA-1 `2bf5f3e5dfde`), written before the seed runs.** "Unstable" was made numeric in the driver
    before any F7 run: a halt (non-finite loss, the hard collapse floor, or short of 5,000 steps), or
    divergence — the final 500-step mean above the run's lowest 500-step rolling mean by more than
    50%. Nearest is by rate ratio, a tie going to the lower rate. All six reached step 5,000 with no
    halt and no divergence:

    | arm | rate | final 500-step mean loss | lowest 500-step rolling mean | rise over lowest | top-1 at 5,000 | steps/s |
    |---|---|---|---|---|---|---|
    | MoCo | 6.25e-6 (1/3×) | 7.4835 | 7.4699 | 0.18% | 0.469 | 1.67 |
    | MoCo | **1.875e-5 (rule)** | 7.2924 | 7.2924 | 0 | 0.562 | 1.33 |
    | MoCo | 5.625e-5 (3×) | 7.2887 | 7.2829 | 0.08% | 0.438 | 1.77 |
    | MAE | 6.25e-6 (1/3×) | 0.9679 | 0.9679 | 0 | — | 1.10 |
    | MAE | **1.875e-5 (rule)** | 0.9641 | 0.9641 | 0 | — | 1.10 |
    | MAE | 5.625e-5 (3×) | 0.9622 | 0.9622 | 0 | — | 1.10 |

    **Choice: the rule's rate, 1.875e-5, for both arms — it is stable,** so the seed runs train at
    the configured rate with no departure. MoCo's InfoNCE rises during the first ~1,000 steps at every
    rate (to 8.3–9.2 at its peak, from 7.45) as the queue's initial random keys are replaced by real
    ones, then falls; the lowest rolling mean comes after that rise, so it is not counted as
    divergence. The rule-rate MoCo arm ran slower (1.33 steps/s) than its neighbours and the rehearsal
    (1.74): machine load at the time, not the recipe. Recorded for the record only: pretext loss
    chose nothing here, because the rule's rate was not unstable.
    About 7.6 h of machine time in all (§5.2).
- **F8** — *Settled:* **the method-agnostic hard floor gates** a run (a collapse that stops it);
  M's JEPA-grounded soft floors (`collapse_floor`, 2.5) are **reported descriptively, not gating**:
  they were fitted to JEPA traces only.
- **F9** — *Settled (user, 2026-10-09):* **test-time D4 averaging is off for every encoder** (`TODO.md`
  P2). It may be run later as a sensitivity analysis, applied to every encoder alike, never to one
  arm alone.

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
| 8 | Driver `artifacts/baseline_run.py` reusing m2's `Run`, segmenting and `keep`, under A1's v2 environment (§1); fixed stop 101,308; probes at 0.5/1/2/4 epochs for trajectories only, through `artifacts/baseline_trajectory_probe.py` (E2's protocol, §1.1; k2 refuses probe_v2) | new artefact | 0.5 d | out-dir guard as in m2 (`m2_long_run.py:205`) |
| 9 | Rehearsal (§6) | `rehearsal_run.py` pattern | 0.25 d + ~1 h machine | |
| 10 | Tests (below) | `tests/test_objectives_moco.py`, `test_objectives_mae.py`, `test_objective_loop.py` | 1 d | |
| 11 | D-entries: D12 amendment (16×16, ViT-S, decoder; done 2026-10-02); the MoCo recipe; `docs/spec/objectives.md` §2 | docs | 0.25 d | |

**Total: about 5.5–7 working days of build**, then about 4–5 machine-days for the four runs.

Tests (invariant tier unless marked): M's `config_hash` unchanged; each baseline config equal to
A1's `pretrain_v2.yaml` outside `objective`, on the v2 freeze; JEPA losses identical step for step
after the loop refactor; a baseline's encoder init byte-identical to M's (and so A1's) for the same
seed; a baseline's data order equal to M's; no flip or rotation in the augmentation (windows are
axis-aligned token sub-grids, a chiral test image is never mirrored); MoCo queue enqueue/dequeue and
no grad on the key encoder; MAE exactly 64 visible tokens, loss only on masked patches, norm-pix
against a hand computation; the baseline checkpoint loads through `load_frozen_encoder`, passes
`assert_frozen`, and `encode()` uses all 256 tokens; integration: resume identity per baseline
(250 + 250 = 500); planted-positive overfit on a tiny fixed set.

## 10. Pre-registration still owed

The cross-objective comparison (`TODO.md` Epic G) needs its own D27 outcome enumeration and D28
planted positives before any baseline number is read. This document fixes training only.
