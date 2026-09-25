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

## Stop 1 decisions (user, 2026-09-25)

- **SAE layers: block 11 and block 6**, fp16.
  - The brief's "last layer" is corrected to the **probe layer**: block 11 of 12, which the probes
    pool and where faithfulness is measured. Block 12 is read by nothing.
  - Block 6 is the middle layer.
- **V3.** Test-split galaxies in GZ3D with bar or spiral masks, up to 500 of each, the list hashed
  before running.
  - Masks are reprojected into each stamp's frame through both WCSs, with 20 overlays eyeballed first.
  - A pixel is bar or arm at ≥ 3 volunteers; sensitivity at ≥ 2 and ≥ 5.
  - A patch is positive at ≥ 50% of its pixels inside the mask.
  - Metric: per-galaxy patch-level ROC AUC, median across galaxies. Bar maps are scored against bar
    masks and spiral maps against arm masks, each compared with the untrained encoder and the
    centre prior.
- **Cascading randomisation:** blocks 12 → 1 top-down, each block's norms included, then the patch
  embedding last. Pooling unchanged.
- **Answers with fewer than 20 positives** in the occlusion sample (lens/arc 5, boxy bulge 9,
  disturbed 13): example maps only, no summary statistics.

## Part 1 pre-registration — occlusion-map tool validation (V1–V4)

Code: `artifacts/dd_core.py` (encoders, read-outs, occlusion) and `artifacts/dd_part1.py` (stages,
statistics, plants). Every plant is scored by the same function as the real test. Nothing below is a
morphology claim.

**Maps.**
- Score = probe direction · block-11 mean-pooled embedding (the 37 ladder probes, refitted by `_fit`
  on O1's bank train rows, raw coordinates). Plus AA3a's PC1 (the band-offset axis: PC1 of M's bank
  train embeddings) and the concept-free map (Δ‖pooled‖ / ‖pooled‖).
- Map value per patch = (score with all 256 tokens − score with the patch removed) / SD of that
  score over the occlusion sample. Positive = the patch supports the concept.
- Primary: token dropping, single patches, on all 2,000 occl galaxies.
- Secondary: 2×2 blocks at stride 1 (each patch = the mean over the blocks covering it), and sky-noise
  fill (per-band median and robust σ of the stamp's own sky outside r = 64 px). The secondary modes
  run on a seeded 400-galaxy subset (GPU budget).
- Cascade: M's blocks re-initialised top-down from seed 1 (12 → 1, each with its norms), then the
  patch embedding (level 13 = fully randomised). M's probe directions, pooling unchanged. Level 1
  (block 12) cannot change a block-11 read-out; it is run and reported as the check that it doesn't.

**V1 — model sensitivity.**
- Per answer (bar a06, spiral a08, edge-on a04): its 54 stratified galaxies (27 positive,
  27 negative). Per galaxy, the **|Spearman|** over 256 patches between M's primary map and the
  level-13 map; the median across galaxies, with a 10,000-draw bootstrap 95% CI.
- |ρ|, not the brief's signed ρ: a randomised network's map has an arbitrary sign, so an
  image-driven map anticorrelated with M's would pass a signed bar. The absolute reading is stricter.
  (Changed before the hash, before any M-vs-randomised statistic was computed.)
- States per answer:
  - **FAIL**: median ≥ 0.3;
  - **FRAGILE**: median < 0.3 but the CI reaches 0.3;
  - **PASS**: the CI lies below 0.3;
  - **UNREACHABLE**: neither V1 plant reaches FAIL for that answer (the statistic has not been shown
    able to fail).
- Precedence: FAIL > UNREACHABLE > FRAGILE > PASS. V1 overall takes the worst answer.
- Plants (D28), per answer, the same statistic:
  - (a) a second fully randomised network (seed 2) in M's place;
  - (b) an image-only map (per-patch summed r flux) in M's place.
  - The FAIL state is reachable if (a) or (b) reads FAIL.
  - Also recorded: an arithmetic plant (the level-13 map + equal-variance noise in M's place), which
    shows the pipeline can reach FAIL. It does not lift UNREACHABLE.
- **Plant results, run before the hash** (`runs/dd/part1/plants.json`), median |ρ| against the
  seed-1 randomised maps:

  | answer | (a) seed-2 network | (b) flux map | arithmetic |
  |---|---|---|---|
  | bar | 0.272 (CI reaches 0.3: FRAGILE) | 0.221 | 0.670 |
  | spiral | 0.077 | 0.198 | 0.605 |
  | edge-on | 0.180 | 0.101 | 0.688 |

  Neither image-driven plant reaches 0.3. Two randomised networks' maps barely agree with each
  other or with the image. So **V1 reads UNREACHABLE unless M's own |ρ| is ≥ 0.3 (FAIL)**: at this
  layer and scale, V1's failure mode (maps that show the image, not the model) is not something the
  randomised maps exhibit. The cascade curve is reported beside it.
- Reported: the full cascade curve (median ρ against M per level), and the same statistic for the
  concept-free map.

**V2 — planted positive: the band-offset map on bright-source edges.**
- 50 occl galaxies, seeded.
- Bright-edge patches = the top 10% (26) of patches by summed r-band gradient magnitude of the
  unshifted stamp.
- Mass = Σ|map| on bright-edge patches / Σ|map|, for the band-offset (PC1) map.
- Conditions:
  - unshifted;
  - g rolled +1 px in columns (an integer shift: exact, no resampling; the wrapped column replaced by
    the original first column);
  - control: all three bands rolled +1 px together (no change in misregistration).
- Test: one-sided paired Wilcoxon, shifted > unshifted, α = 0.01.
- States, in precedence:
  - **FAIL**: no rise at α, or median Δ ≤ 0;
  - **CONFOUNDED**: the joint-shift control also rises at α, with a median Δ ≥ half the g-shift's;
  - **WEAK**: rises, but median Δ < 0.02;
  - **PASS**.
- Plants:
  - the unshifted maps with bright-edge values × 1.5 must PASS;
  - unchanged maps must FAIL.
- **Plant results, run before the hash:** the ×1.5 plant reads PASS (median Δ 0.098, p 9e−16;
  joint-control Δ 0); the null reads FAIL (Δ 0, p 0.59). Both states are reachable.
- Reported: the median mass on the full occl sample against the uniform 0.10.

**V3 — beats the centre (GZ3D; Masters et al. 2021, DR17 VAC v4_0_0).**
- Overlap: 3,521 test-split galaxies have a GZ3D file.
- Masks are reprojected GZ3D WCS → sky → the SDSS r-frame WCS → v1's stamp origin ceil(x − 128),
  4×4 sub-sampling per stamp pixel. The 20-galaxy overlay figure is eyeballed before selection.
- A pixel is bar/arm at ≥ 3 volunteers; sensitivities at ≥ 2 and ≥ 5. A patch is positive at ≥ 50%
  of its pixels. The evaluation domain is the patches wholly inside the GZ3D image footprint.
- Eligible = at least one positive and one negative patch in the domain at ≥ 3. Up to 500 per
  structure, seeded; the lists are hashed before any map is computed on them.
- Metric: per-galaxy patch-level ROC AUC, median across galaxies. Bar maps (a06) against bar masks,
  spiral maps (a08) against arm masks.
- Comparisons:
  - M's primary map vs the untrained encoder's map (seed 0, its own probes);
  - M's primary map vs the centre prior (−distance from the stamp centre).
  - Each is a one-sided paired Wilcoxon; BY over the 4 tests at q = 0.01.
- States per structure, in precedence:
  - **INSUFFICIENT**: fewer than 30 eligible;
  - **FAIL**: either comparison not significant;
  - **WEAK**: both significant, but either median paired ΔAUC < 0.02;
  - **PASS**.
- Plants:
  - an oracle map (the patch mask label + N(0, 0.5)) must PASS both structures;
  - the centre prior + N(0, 1) in M's place must not PASS.
- Reported: 2×2 and noise-fill maps, the ≥ 2 and ≥ 5 sensitivities, and a patch-coverage
  sensitivity at ≥ 25% (declared before the hash: on galaxies already fetched, ≥ 3-vote bar masks
  covered no patch at ≥ 50%, because a bar is narrower than a 6.3″ patch). The sensitivity has its
  own eligible lists (up to 500 per structure, seeded, hashed with the primary lists), scored against
  its own labels by the same function; it is reported beside the primary verdict and never replaces it.

- **Lists (drawn from masks only, before any map on them; `runs/dd/v3_list.json`):**
  - bar ≥ 50%: 165 eligible, all used, sha256 `54befdeaa88d4d193ccb453cfb73bc25ce5dc40156c2246fb5e1ae3172934f1d`;
  - spiral ≥ 50%: 552 eligible, 500 used, `eddf11955b08624945c9484f79efa7e9d9b3a1e43bb4c05d748c532dfd125231`;
  - bar ≥ 25%: 426 eligible, all used, `e5db0bce1f9a5243391dc7e5aa1b18cc46f2183a63382a4a6276ff54d2ce2d3d`;
  - spiral ≥ 25%: 703 eligible, 500 used, `ff345d01cc12077f2cb9ba78a0e2b06a942ff7a689ad84401f546069afdf422c`.
  - Union 794 galaxies. Fetch: 3,521 GZ3D files and r-frame headers, 0 failed.
- Overlays (`out/dd/dd_v3_overlays.png`, 20 galaxies with a ≥ 3-vote mask): the reprojected contours
  sit on the same arms and bars in M's stamps as in the GZ3D images, through the ~20–40° rotation
  between the two WCSs; no flip or offset seen.

- **Plant results, run before the hash — V3 is WITHHELD (D28).** Inside the footprint (the central
  8×8 patches), the centre prior alone scores median per-galaxy AUC **0.978 (bar)** and
  **0.956 (arms)**, and the untrained encoder about 0.95. The oracle (label + N(0, 0.5)) scores 0.954
  / 0.942 and reads **FAIL** against the centre prior, in both structures and both coverage rules.
  The comparison is saturated: a good map cannot reach PASS. **V3 is not hashed and not scored on M
  under this design.** The redesign goes to the user at Stop 2.
- Candidate redesign (plants only, M unscored): the ring-stratified AUC. Patch pairs are compared
  only within the same 16-px radial ring, so the centre prior is 0.5 by construction. Median over
  galaxies:

  | list | usable | oracle | centre | centre + noise | untrained |
  |---|---|---|---|---|---|
  | bar ≥ 50% | 164 / 165 | 1.00 | 0.50 | 0.50 | 0.67 |
  | spiral ≥ 50% | 464 / 500 | 1.00 | 0.50 | 0.58 | 0.75 |
  | bar ≥ 25% | 390 / 426 | 1.00 | 0.50 | 0.50 | 0.67 |
  | spiral ≥ 25% | 450 / 500 | 0.96 | 0.50 | 0.62 | 0.75 |

**V4 — equivariance (exploratory, no bar).**
- 100 occl galaxies, seeded; rot90, rot180 and a left–right flip of the stamp.
- Per galaxy, the Spearman of the transformed stamp's map against the transformed original map; the
  median with bootstrap CI.
- Maps: bar, spiral, edge-on, band offset, concept-free.

**Answers with fewer than 20 positives** (lens/arc, boxy bulge, disturbed): example maps only.

*Hashed 2026-09-25, before any M statistic for V1, V2 or V4 was computed. SHA-1 over the section from its heading through the line above the blank line before this footer, plus a trailing newline: `b19ee5ee8b417ddd906447d2a253213bcf51830d`. V3 is withheld (above).*

## Stop 2 — Part 1 results and patch parts (2026-09-25; TOOL VALIDATION, no morphology claims)

Scored against the hash above (`runs/dd/part1/score.json`, `plants.json`). Compute:
- M maps: 2,000 galaxies primary + 400 secondary, 65 min.
- Cascade: 14 levels + the plant network on 162 galaxies, 51 min.
- V3 maps: M × 3 modes + untrained on 794 galaxies, 68 min.

**V1 — UNREACHABLE (as pre-registered).**
- M against fully randomised, median |ρ| (CI): bar 0.090 (0.063–0.109), spiral 0.180 (0.163–0.214),
  edge-on 0.111 (0.085–0.130). This would read PASS if the image-driven plants had been able to fail.
- **The cascade is the informative part** (`out/dd/dd_part1_cascade.png`):
  - Level 1 (block 12) equals M, as it must.
  - The maps stay at median |ρ| ≥ 0.6 until blocks 12 → 5 are all random.
  - They cross 0.3 only once blocks 3 and 2 go. Bar: 1.00, 0.92, 0.87, 0.82, 0.78, 0.77, 0.69, 0.62,
    0.60, 0.39, 0.16, 0.07, 0.09.
- Through the residual stream, these occlusion maps are dominated by what blocks 1–4 find salient,
  not by M's trained top blocks. A map on the aligned encoder should be read with that in mind; a
  per-block attribution (Part 4's hooks) is the tool for it.

**V2 — FAIL (as pre-registered).**
- Rolling g by 1 px leaves the band-offset map's bright-edge mass where it was: median 0.541 →
  0.541, Δ 0.003, p 0.52. The joint-roll control gives Δ −0.002.
- Exploratory reasons, measured after the verdict:
  - The shift does act. It moves M's pooled PC1 by **+0.61 SD** (AA3a measured +0.67 SD/px); the
    joint roll moves it 0.03 SD.
  - It rewrites the band-offset map: Σ|Δmap| = 0.60 Σ|map|, 58% of it on bright-edge patches.
  - But the map already concentrates there: 0.54 of its mass on the bright-edge decile, against
    0.36 for the concept-free map, 0.22 for bar and spiral, and 0.10 uniform. So the *share* can't
    rise.
- The plant modelled "mass moves onto edges", not "the map changes where it already sits". A plant
  must copy the mechanism, not the statistic.
- The band map's first clause, concentration on bright-source edges, holds descriptively. It was
  not the tested clause.

**V3 — WITHHELD (plants fail, D28).**
- See the pre-registration. The centre prior saturates the patch AUC inside the GZ3D footprint.
- M's V3 maps are computed (`runs/dd/part1/v3_*.npy`) but unscored.
- The ring-stratified redesign and its plants are above; its adoption is the user's call.
- Figure: `out/dd/dd_v3_saturation.png`; overlays `out/dd/dd_v3_overlays.png`.

**V4 — exploratory.** Median ρ (100 galaxies; rot90 / rot180 / flip):

| map | rot90 | rot180 | flip |
|---|---|---|---|
| bar | 0.37 | 0.26 | 0.43 |
| spiral | 0.48 | 0.53 | 0.65 |
| edge-on | 0.46 | 0.44 | 0.53 |
| band offset | 0.15 | **−0.29** | **−0.26** |
| concept-free | 0.45 | 0.44 | 0.53 |

- The concept maps are partially equivariant; the fixed sin-cos positions break exact equivariance.
- The band-offset map *reverses sign* under rot180 and a flip, the transforms that reverse an
  x-offset vector, and is near zero under rot90. That is what a directional misregistration code
  does.

**Removal modes.**
- Sky-noise fill and 2×2 blocks give maps of the same form as single-token drops
  (`out/dd/dd_part1_modes.png`).
- Noise fill gives larger positive values (bar map range to 0.71 SD, against 0.14 for drop, on the
  first stamp).

**Seen in the example maps** (`out/dd/dd_part1_maps.png`):
- Field stars carry large values in *every* read-out, including the concept-free one.
- On a padded stamp, the patches along the pad boundary carry map mass.
- Both are the kind of nuisance the SAE feature cards will flag.

**Answers with < 20 positives:** example maps only (`out/dd/dd_part1_rare.png`).

**Part 2 — patch parts (exploratory; `runs/dd/part2/part2.json`, `out/dd/dd_part2_patch_parts.png`,
`out/dd/dd_part2_components_M.png`).**
- **The band offsets: yes, strongly, and only on M.** v1's recorded offsets come from probe_v2's
  cut_log, for the 909 occl galaxies re-cut so far. Galaxy-mean token PC scores against them:

  | M component | i−r x | g−r x | i−r y | g−r y |
  |---|---|---|---|---|
  | PC2 | −0.85 | −0.62 | 0.03 | 0.03 |
  | PC3 | −0.77 | −0.57 | 0.08 | 0.07 |
  | PC1 | 0.59 | 0.40 | 0.05 | 0.03 |
  | PC5 | −0.20 | −0.20 | **0.75** | 0.58 |
  | PC7 | −0.20 | −0.18 | −0.53 | −0.42 |

  - The x offsets take PC1–3 (32% of token variance); the y offsets take PC5 and PC7.
  - Every untrained component is |ρ| ≤ 0.08.
  - A 1-px g roll moves M's PC2 by 0.46 token-SD on non-edge tokens and 0.32 on edge tokens; the
    untrained encoder moves ≤ 0.14.
  - So the offset is written into *every* token (the sky included), not only at edges. The RGB
    overlays show it as a whole-stamp hue that differs galaxy to galaxy.
- **Parts:**
  - Beneath that hue, M's components split a core/bulge region from a disc annulus and from the
    sky, following the galaxy's shape (the edge-on disc in column 9).
  - The untrained encoder's PC1 (33%) is a fixed left-to-right positional gradient, plus a
    bright-core blob. It has no disc/sky split.
- **Arms** are not separated at this resolution: 16-px patches, and galaxies 3–8 patches across.

**Part 3 / Part 4 status (built, not run for results).**
- `dd_sae.py`: TopK SAE; synthetic smoke gives VE 0.978, unit-norm decoder held, L0 = 32. The
  smoke needs ~490 steps; 3 epochs at 2e-4 reached only 0.46, so the real runs log convergence.
- `dd_circuits.py`: hooks and latent ablation; `test_dd_circuits.py` gives 3 passed.
- Token disk: 20,000 × 256 × 384 × 2 B = **3.93 GB per layer per encoder** (sae), 0.98 GB (sae_eval).
  Blocks 11 + 6, for M and untrained: **19.7 GB** on the X10.
