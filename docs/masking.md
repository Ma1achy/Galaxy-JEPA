# Masking design note — bounding-box-biased multi-block masking

*Status: design proposal for sign-off. **Propose, don't implement.** No code is
written until this note is approved.*

This is the one genuinely novel architectural piece of Galaxy-JEPA, and two review
rounds converged on it as *the* open item. This note specifies a concrete
sampling scheme, its parameters, its diagnostics, and — critically — how it
**degrades cleanly back to standard I-JEPA**.

---

## 1. The problem

I-JEPA's multi-block masking samples target blocks uniformly across the image. An
SDSS / GZ2 cutout is **mostly black sky** with the galaxy in a central region
(v1's average-image bounding box sat at ~227/256 px ≈ the central ~65–78% of the
frame). Uniform sampling therefore spends a large share of the prediction budget
on **empty sky**, where predicting the masked target embedding is trivial
(predict "black") and teaches the encoder nothing about morphology. We want the
budget spent on the galaxy.

---

## 2. Recap — standard I-JEPA masking (the baseline we generalise)

Operating on the **token grid** (for 256-px input, 16-px patches → a 16×16 = 256
token grid):

- Sample **M = 4 target blocks**, each with **scale 0.15–0.20** of the image area
  and **aspect ratio 0.75–1.5**.
- Sample **1 context block** with **scale 0.85–1.00**.
- **Remove** from the context any tokens overlapping *any* target block (prevents
  trivial copy-through).
- The predictor receives context tokens + mask tokens (positionally encoded at
  the target locations) and predicts the **target blocks' embeddings** from the
  **EMA target encoder**; loss is **latent MSE**.

All block geometry is at token granularity. Our scheme changes only **where the
blocks are sampled from** — nothing else in the I-JEPA recipe moves.

---

## 3. The galaxy prior — a bounding box from the average image

Reuse v1's idea (documented in the v1 README; the computation code was not in the
v1 repo, so we rebuild it — it is a few lines of NumPy):

1. Take a random sample of **~2,000** cutouts.
2. Compute the **mean image** (per-pixel average over the sample).
3. Threshold at a **brightness fraction τ** (v1 used **τ = 8.5%** of peak).
4. Take the **centred bounding box** of the supra-threshold region.
5. Express it as a **fractional box** `(r0, c0, r1, c1) ∈ [0,1]²` and **project
   onto the token grid** → a boolean `G×G` mask `B`, where `B[i,j] = 1` if token
   `(i,j)` lies inside the galaxy box.

This is a **dataset-level prior** computed **once**: cheap, robust, and
deterministic. Because GZ2 cutouts are centred on the target galaxy, a single
shared box is a reasonable approximation — but it is the *floor*, not the ceiling
(see §3.1).

### 3.1 Paper-1 default — per-galaxy box from the Petrosian radius

A single shared box is **too loose for small, distant galaxies** (it includes a
lot of sky) and can **clip large, nearby ones**. GZ2 cutouts vary substantially
in apparent galaxy size, so a fixed box leaves accuracy on the table. The fix is
nearly free: we are **already pulling the Petrosian radius** from CasJobs for the
nuisance battery (P2 data layer), so we can **scale the box per galaxy** by it —
e.g. a box of half-width `k · R_petro` (in pixels, via the cutout's arcsec/pixel),
clamped to the frame, with `k` a small multiplier (~2–3) tuned so the box
comfortably contains the galaxy.

This gives a tight, per-image galaxy prior at near-zero extra cost. **Decision:
the Petrosian-scaled per-galaxy box is the Paper-1 default.** The global
average-image box (§3) is retained as the **β / ablation fallback and degradation
reference** — and as the fallback for any galaxy whose Petrosian radius is missing.
*(A per-image flux-threshold saliency box remains a further option.)*

---

## 4. The biased sampling scheme

### 4.1 Token weight map

Define a sampling weight over tokens from the box `B` and a single **bias
strength β ∈ [0, 1]**:

```
w[i,j] = 1            if B[i,j] = 1      (inside the galaxy box)
w[i,j] = 1 − β        if B[i,j] = 0      (sky)
```

- **β = 0** → all weights equal → **uniform sampling = standard I-JEPA**.
- **β = 1** → sky weight 0 → targets sampled **strictly inside** the box.
- **0 < β < 1** → a soft preference for the galaxy, sky still reachable.

### 4.2 Target-block sampling (the core change)

For each of the M target blocks:

1. Draw block **size** (`scale ∈ 0.15–0.20`, `aspect ∈ 0.75–1.5`) → token
   height/width `(h, w)` — **exactly as I-JEPA**.
2. Enumerate the valid top-left positions for a block of that size on the grid.
3. Score each candidate position by the **mean token weight under the block**,
   `s = mean(w[block])`.
4. **Sample the position with probability ∝ s.** Optionally enforce a
   **min-overlap floor** `φ`: reject candidates whose fraction of in-box tokens is
   `< φ` (re-draw size if no candidate qualifies).

This biases target blocks to land **on the galaxy** while preserving I-JEPA's
block-size distribution and count.

### 4.3 Context block

Keep the context block **large** (`scale 0.85–1.00`). Because it is large it
already covers the galaxy, so it needs no bias; apply the **standard removal** of
tokens overlapping the chosen targets. *(Optional, flagged: bias the context
toward the box too — but with large context this adds little and is not in the
default.)*

---

## 5. Parameters

| Parameter | Symbol | Default | Notes |
|---|---|---|---|
| Bias strength | `β` | sweep `{0, 0.5, 1.0}` | β=0 reproduces I-JEPA; the ablation lever |
| Brightness threshold | `τ` | 0.085 | v1's value; controls box size |
| Sample for mean image | — | 2,000 | for the dataset-level box |
| Number of target blocks | `M` | 4 | I-JEPA default |
| Target scale | — | 0.15–0.20 | I-JEPA default |
| Target aspect | — | 0.75–1.50 | I-JEPA default |
| Context scale | — | 0.85–1.00 | I-JEPA default |
| Min in-box overlap floor | `φ` | 0.0 (off) | optional hard constraint |
| Token grid | `G` | 16 | 256-px input ÷ 16-px patch |

Everything except `β`, `τ`, and `φ` is inherited from I-JEPA, so the scheme adds
**three** new knobs and reuses the rest.

---

## 6. Graceful degradation (the key property)

The scheme is a **strict generalisation** of I-JEPA:

- **`β = 0`** ⇒ uniform weights ⇒ identical to standard I-JEPA.
- **box = full frame** (every token in-box) ⇒ uniform weights ⇒ identical to
  standard I-JEPA, for any `β`.

So the **β ∈ {0, 0.5, 1.0}** sweep is a clean, controlled ablation: β=0 *is* the
I-JEPA control, and any representation-quality change as β rises is attributable
to the galaxy bias alone. This also means the same module ships the baseline — no
separate code path.

---

## 7. Diagnostics (ship with the module)

- **Sky-waste metric.** Fraction of sampled **target tokens** that are "sky"
  (below τ on the mean image), as a function of β. Expect it to fall sharply from
  β=0 → β=1; this is the direct evidence the scheme does its job.
- **Coverage check.** Distribution of in-box overlap per target block; confirms
  φ behaves.
- **Mask-overlay visualisations.** Render context (kept), target blocks, and the
  galaxy box over example cutouts at each β — eyeball that targets sit on
  structure, not sky.
- **β shifts the prediction-difficulty distribution — re-tune per β.** Biasing
  all targets onto the galaxy **removes the easy sky-prediction task**, so every
  target becomes "hard". That is the point, but it changes the **loss scale** and
  can **move the collapse / EMA sweet spot**: predicting constant sky is the most
  collapse-adjacent signal there is, so removing it is not necessarily bad — just
  *different*. **Do not assume β=0 tuning transfers to β=1**; re-check the EMA
  schedule / masking ratio at each β. The collapse monitor (P4) covers this; track
  the masking ratio actually realised and flag it.

---

## 8. Scope / non-goals

- This note covers **target/context sampling geometry only**. The encoder,
  predictor, EMA target, and latent-MSE loss are unchanged from I-JEPA.
- Rotational symmetry is handled separately (augmentation first; E(2)-equivariant
  ViT as a later ablation — see `DECISIONS.md`).
- The **per-galaxy Petrosian-scaled box** (§3.1) is the **Paper-1 default**; the
  global average-image box is the β / ablation fallback and the missing-radius
  fallback. Flux-threshold saliency boxes, context biasing, and learned masking
  remain **future options**, out of scope for the first run.

---

## 9. Open question for sign-off

- Default **β** for the headline run (recommend reporting the **β sweep** rather
  than committing to one value, with β=0 as the published control).
- Box granularity is **decided: per-galaxy Petrosian-scaled box (§3.1) is the
  Paper-1 default**, global average-image box as the β / ablation + missing-radius
  fallback. Remaining sub-question for sign-off: the multiplier `k` (~2–3) and the
  exact missing-radius fallback behaviour.
