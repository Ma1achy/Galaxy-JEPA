# Interpretability tooling on M — TOOL VALIDATION (Brief DD)

**Everything here is tool validation on M, and exploratory. No morphology claims.** The same code
re-runs on the aligned encoder later, where claims are pre-registered separately. Nothing from this
file enters the findings.

## Step 0 — architecture and samples (2026-09-25)

**M** (`runs/m/encoder.pt`, `models/vit.py`)
- ViT-S/16 at 256²: 16-px patches, a 16×16 grid of **256 tokens**, width **384**, **12 blocks**,
  6 heads, MLP ratio 4, fixed 2-D sin-cos positions, no CLS token.
- The saved weights are the **context encoder** (`jepa.encoder`, `objectives/jepa.py:473`), not the
  EMA target.

**Token dropping is supported and in-distribution.**
- `patch_embed_tokens` → gather any subset → `run_tokens_layers` is exactly the path the context
  encoder was trained through.
- M's contexts were one block at scale 0.85–1.0 minus four target blocks (`docs/masking.md` §4.3),
  typically well under 256 tokens.
- The probe path feeds all 256, which the context encoder never saw at once. Dropping 1 or 4
  patches moves *toward* the training regime, not away from it.

**The probes.**
- The read-out is block 11 of 12 (`DEFAULT_LAYER = −2`, before the final norm), **mean-pooled over
  the 256 tokens**.
- The 37 answer directions are **not stored**. Every brief refits them deterministically: the
  ladder's logistic (`r_nonlinear._direction_z` / `_fit`) on `artifacts/out/o1_embeddings.npz`.
- That bank holds M's and untrained seed 0's pooled embeddings on the fixed split: 40,000 train +
  34,829 test (`j4_spread_controls.prepare`, the one site).

**Samples** (`artifacts/dd_step0.py`, seed 20260925; copies in `runs/dd/`, which is on the X10 Pro)

| list | n | from | sha256 (sorted IDs) | fp16 stamps |
|---|---|---|---|---|
| occl | 2,000 | test | `404859c3a71c04bd…` | 0.79 GB |
| sae | 20,000 | probe-train | `76b4496eba4c8fd0…` | 7.86 GB |
| sae_eval | 5,000 | test, disjoint from occl | `563743fbc9e72851…` | 1.97 GB |

- occl is stratified by answer and visibility: per answer, up to 27 positives and 27 negatives among
  galaxies in its conditional population (those that reached the question), without replacement
  across answers, plus 69 random to make 2,000.
- Five answers are short of positives: lens/arc 5, boxy bulge 9, disturbed 13, dominant bulge 20,
  dust lane 21.
- Figure: `out/dd/dd_step0_sample.png`.

**GZ3D (for V3).**
- 29,813 galaxies in the DR17 VAC (`galaxyzoo3d/v4_0_0`). Positions come from `mangaDrpAll` for the
  9,314 observed and from `mangaTarget` for the rest.
- **23,141 fall in the probe corpus (< 3″)**, but only 274 in the occl sample. V3 therefore needs its
  own draw: test ∩ GZ3D, with volunteer bar or spiral masks.
- `runs/dd/gz3d_positions.csv`.
