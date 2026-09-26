# Interpretability tooling on M — TOOL VALIDATION (Brief DD)

**Closed 2026-09-26.** Summary table, rerun queue and method limits: see "Brief DD on M — closed" at the end.

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

## Stop 2 decisions (user, 2026-09-25)

- **V1:** recorded as UNREACHABLE, with the randomisation curve as the headline (the maps mostly
  reflect blocks 1–4). Exploratory additions (a)–(c) below.
- **V2:** stays FAIL; the plant modelled the wrong mechanism. **For the aligned-encoder rerun,
  pre-register a localised plant:** misalign g by 1 px only within one source's footprint (or one
  quadrant). Pass if the band-offset map's mass inside the planted region rises against the
  unplanted stamp.
- **V3:** adopt the same-radius comparison as a post-hoc amendment (next section).
- **V4:** noted. The band-offset map's sign flip under rot180 and flip is a consistency check.
- **Part 2:** noted. The offset is a whole-stamp colour cast in every token.
- **Part 3:** go, with a disk-first check and an amended S1 (see the Part 3 pre-registration).

### V1 exploratory (a)–(c) (`artifacts/dd_part1b.py`, `runs/dd/part1/v1x.json`)

**(a) Direct / indirect split.**
- Dropping token j moves the mean-pooled embedding by a direct term, (h_j − p)/255 (j's own
  absence, the other tokens held fixed), plus an indirect term (the other 255 tokens change,
  because attention no longer sees j).
- The direct term's share of map variance, median (IQR):

  | map | direct share | ρ(direct, occlusion) |
  |---|---|---|
  | bar | 0.15 (0.05–0.27) | 0.57 |
  | spiral | 0.24 (0.17–0.32) | 0.67 |
  | edge-on | 0.07 (0.01–0.18) | 0.65 |
  | band offset | −0.02 (≈ 0) | 0.11 |

- **The maps are mostly indirect.** The cheap per-token read-out map (probe · token, rank-identical
  to the direct term) is therefore *not* promoted to primary. It is reported beside the occlusion
  map.
- The band-offset map is wholly indirect: the offset code is not in any one token's own projection
  but in how the tokens respond to each other. That fits Part 2's finding: the cast is in every
  token.

**(b) Concept specificity** (median |ρ| per galaxy, all 2,000):

| pair | occlusion | direct |
|---|---|---|
| bar vs spiral | 0.34 | 0.19 |
| bar vs edge-on | 0.13 | 0.13 |
| spiral vs edge-on | 0.07 | 0.12 |

- Self-agreement of each concept across removal modes, on the 400 secondary: bar 0.68 / 0.60
  (drop 1 vs 2×2 / vs noise fill); spiral 0.77 / 0.56; edge-on 0.71 / 0.57.
- **The concept maps agree with themselves far more than with each other, so they are not generic
  salience.** Bar and spiral share the most, at 0.34.

**(c) Brightness baseline** (median |ρ| with the r flux per patch):
- Concept occlusion maps: 0.09 (bar), 0.09 (spiral), 0.10 (edge-on).
- Band offset: 0.13. Concept-free: 0.23.
- Direct maps: 0.39, 0.18, 0.22; band offset 0.41.

## V3 amendment — ring-stratified AUC — post hoc (D27)

*Made on 2026-09-25, after the pre-registered V3 plants failed and before any M V3 map was scored.
It applies to this run and is labelled post hoc. The withheld V3 above was never scored.*

**Reason.**
- Inside the GZ3D footprint the positive patches are the central ones. The centre prior alone scored
  median per-galaxy AUC 0.978 (bar) and 0.956 (arms).
- A noisy perfect mask (label + N(0, 0.5)) scored 0.954 / 0.942 and could not beat it. The
  comparison was saturated.

**Metric.**
- Per galaxy, a Mann–Whitney AUC over (positive, negative) patch pairs *in the same 16-px annulus*
  of patch-centre radius, pooled over annuli (each annulus weighted by its pair count). The centre
  prior is 0.5 by construction.
- A galaxy is usable if at least one annulus holds both classes. Median over galaxies.
- Lists, labels and footprint are unchanged (hashed above). Code: `dd_part1b.v3b_stat`.

**Criterion, per structure (bar, spiral separately).**
- M's primary map (token drop, single patch) must beat **both** the untrained encoder's map (its own
  probe) **and** the brightness map (r flux per patch).
- For each comparator: per-galaxy paired ΔAUC = M − comparator, with a 10,000-draw bootstrap 95% CI
  of its median over galaxies.
- States, in precedence:
  - **INSUFFICIENT**: < 30 usable galaxies;
  - **REVERSED**: either CI wholly below 0;
  - **FAIL**: either CI reaches 0;
  - **WEAK**: both CIs above 0, but either median Δ < 0.02;
  - **PASS**.

**Plants, run before this hash** (`runs/dd/part1/v3b_plants.json`; D28):
- oracle (label + N(0, 0.5)) must PASS;
- the brightness map + N(0, 0.1 SD) in M's place must not PASS;
- no comparator may saturate (median ring AUC < 0.95).

| list | oracle | brightness as M | untrained ring AUC | brightness ring AUC |
|---|---|---|---|---|
| spiral ≥ 50% | PASS | FAIL | 0.75 | 0.81 |
| spiral ≥ 25% | PASS | FAIL | 0.75 | 0.86 |
| bar ≥ 50% | **FAIL** | FAIL | 0.67 | **1.00** |
| bar ≥ 25% | **FAIL** | FAIL | 0.67 | **1.00** |

- **Spiral is reachable and is scored.**
- **Bar is UNREACHABLE under this metric, and withheld.** Within an annulus the bar's patches are
  the brightest, so the brightness map already ranks them perfectly (median 1.00) and no map can
  beat it. Its M statistic is reported descriptively, never as a verdict.
- A bar test needs a brightness-matched comparison (pairs matched on patch flux as well as radius).
  That is left for the aligned-encoder pre-registration.

**Reported beside the spiral verdict:**
- the 2×2, noise-fill and direct-term maps;
- the ≥ 2 and ≥ 5 vote sensitivities;
- the ≥ 25% coverage list;
- bar, all of the above, descriptively.

*Amendment hashed 2026-09-25, before any M V3 statistic: SHA-1 over this section from its heading through the line above the blank line before this footer, plus a trailing newline: `97e3ed526d412bb61dfeffb26d036b841249030b`.*

### V3 amendment result (`runs/dd/part1/v3b_score.json`) — spiral **REVERSED**; bar withheld

| map / list | structure | state | ring AUC: M / untrained / brightness | ΔAUC vs untrained (CI) | ΔAUC vs brightness (CI) |
|---|---|---|---|---|---|
| **drop 1 patch, ≥3 votes (primary)** | **spiral** | **REVERSED** | 0.665 / 0.748 / 0.810 | −0.045 (−0.086, 0.000) | −0.139 (−0.182, −0.097) |
| drop 2×2 | spiral | REVERSED | 0.667 / 0.748 / 0.810 | −0.027 (−0.063, 0.000) | −0.109 (−0.145, −0.073) |
| noise fill | spiral | REVERSED | 0.575 / 0.748 / 0.810 | −0.116 (−0.151, −0.071) | −0.200 (−0.241, −0.163) |
| direct term | spiral | REVERSED | 0.600 / 0.748 / 0.810 | −0.106 (−0.157, −0.077) | −0.188 (−0.226, −0.143) |
| drop, ≥2 votes | spiral | REVERSED | 0.711 / 0.789 / 0.882 | −0.034 (−0.061, 0.000) | −0.133 (−0.154, −0.104) |
| drop, ≥5 votes | spiral | REVERSED | 0.667 / 0.727 / 0.765 | −0.100 (−0.192, 0.000) | −0.143 (−0.231, −0.079) |
| drop, ≥25% coverage | spiral | REVERSED | 0.720 / 0.748 / 0.857 | 0.000 (−0.017, 0.000) | −0.105 (−0.130, −0.083) |
| drop (descriptive) | bar | — | 0.500 / 0.667 / 1.000 | — | — |

- **Tool reading, on M:** at 16-px patches, the spiral occlusion maps rank volunteer-marked arm
  patches *below* the patch brightness within an annulus, and not above the untrained encoder's
  maps. Every mode, threshold and coverage rule agrees.
- This fits (a) and the cascade. The maps are mostly indirect and mostly shaped by early blocks. What
  they rank is not where the arms are, but which removals most perturb the other tokens.
- On the aligned encoder, occlusion maps are not to be read as arm localisers without a new V3 that
  passes.
- Medians are quantised because many galaxies have few same-annulus pairs, hence CI endpoints at
  exactly 0.000.

## Part 3 pre-registration — sparse autoencoders (S1–S3)

Code: `artifacts/dd_sae.py` (tokens, training, evaluation) and `artifacts/dd_sae_score.py` (S1–S3,
plants).

**Disk, first** (user condition).
- X10 free: 2,329 GiB.
- The rest of the re-pull needs about 1,080 GiB: probe_v2's remaining ~109k stamps (87 GB),
  pretrain_v2 (655 GB), and the fp16 cache for both (416 GB).
- The extraction adds 19.7 GB.
- Headroom after both: about 1,230 GiB, well above the 50 GB floor.
- The internal SSD has 13 GB free, so the extraction goes to the X10 (`runs/dd/sae_tokens`).

**Data and training (fixed now).**
- Tokens: blocks 11 (the probe layer) and 6, from M and the untrained encoder (seed 0), fp16.
  Train on sae (20,000 probe-train galaxies, 5.12 M tokens); evaluate on sae_eval (5,000 test).
- TopK SAE:
  - k = 32; dictionary 8× (3,072) and 16× (6,144);
  - unit-norm decoder (renormalised each step, parallel gradient removed);
  - pre-encoder bias initialised at the token mean; one input scale so that E‖x‖² = 384;
  - AuxK: k_aux = 256, coefficient 1/32, dead after 1 M tokens unfired.
- Optimiser: Adam (0.9, 0.999, ε 6.25e−10), batch 4,096 tokens, 8 epochs (~10,000 steps). The step
  follows Gao et al.'s size scaling: 2e−4 · √(2¹⁴ / n) = 4.6e−4 (8×), 3.3e−4 (16×).
- Batches are shuffled within 65,536-token contiguous blocks, drawn in random order.
- Eight SAEs: {M, untrained} × {b11, b6} × {8×, 16×}.
- Reported: loss curves, variance explained, and % dead on sae_eval.

**S3 — faithfulness.**
- Replace block-L tokens with their reconstructions, run the rest of the blocks to 11, mean-pool,
  and apply the 37 fixed probes (O1 bank refits). AUC on sae_eval galaxies (each answer's eligible
  population), original against reconstructed.
- States, in precedence:
  - **FAIL**: mean drop > 0.02;
  - **UNEVEN**: mean drop ≤ 0.02 but some answer drops > 0.05;
  - **PASS**.
- **Chosen dictionary** (used for S1, S2 and the cards): 8× at block 11, unless 8× FAILs and 16×
  doesn't. The S3 verdict is the chosen size's. Everything else is reported.

**S1 — planted positive, amended by the user in light of Part 2.**
- At least one latent (block 11, chosen size) has galaxy-level |Spearman| ≥ 0.5 with a per-band
  offset: v1's recorded g−r and i−r in-stamp offsets, x and y, from probe_v2's cut_log. Galaxy-level
  activation is the mean over the galaxy's 256 tokens.
- And its **nearest match** in the untrained-encoder SAE (same layer and size) correlates with the
  same offset at |ρ| < 0.2.
- **"Nearest match" is by activation, not decoder.** The two SAEs' decoders live in different
  networks' embedding spaces, where a cosine means nothing. So the match is the untrained latent with
  the highest |Pearson| of token-level activations with the M latent, over the same sae_eval tokens.
  This is a clarification of the user's wording, flagged to them.
- Location (centre vs edge, bright edge vs other) is reported, not required.
- States, in precedence:
  - **INSUFFICIENT**: fewer than 500 sae_eval galaxies with recorded offsets (2,658 now);
  - **FAIL**: no latent at |ρ| ≥ 0.5;
  - **NOT-SPECIFIC**: every such latent's match reaches |ρ| ≥ 0.2;
  - **PASS**.
- Also reported: the untrained SAE's best |ρ| over *all* its latents per offset, and block 6.

**S2 — injected positive (supporting, not decisive).**
- Synthetic satellite trails in 2% of sae_eval (100 galaxies, seeded):
  - a straight line in all three bands, Gaussian FWHM 3.5 px;
  - peak 3–10× each band's own sky σ (normalised units);
  - random angle, passing within 64 px of the centre.
- Trail patches: ≥ 16 pixels within 2 px of the centre line. The other 98% are the clean tokens.
- Per latent, precision = fires on trail patches / all fires; recall = fires on trail patches / trail
  patches.
- States: **DETECTED** if any latent reaches precision ≥ 0.8 with recall ≥ 0.2 (the floor, so that a
  latent firing once cannot count); else **NOT DETECTED**.

**Plants, run before this hash** (`runs/dd/sae/plants.json`). Every state is reached through its own
scoring function:
- S1 planted latent → PASS;
- null → FAIL;
- untrained match also tracking → NOT-SPECIFIC;
- n = 300 → INSUFFICIENT;
- S2 oracle latent → DETECTED; random → NOT DETECTED;
- S3 identity → PASS; lossy → FAIL; one answer collapsing → UNEVEN.
- A real stamp's injected trail covers 16 patches.

**Feature cards (Stop 3; chosen size, block 11, M; exploratory).** Per latent:
- top-activating patches with 3×3-patch context crops;
- activation density;
- spatial histogram (radius of the firing patch);
- galaxy-level Spearman with the 37 vote fractions;
- galaxy-level Spearman with the nuisance panel: per-band offsets, psfWidth_r, modelMag_r, specz,
  camcol, frame-edge distance, valid fraction;
- **brightness:** galaxy-level with total r flux, and token-level with patch r flux.
- **Nuisance flag:** the strongest |ρ| is a nuisance variable or brightness.
- "Most interpretable": among unflagged latents with density in [1e−4, 0.1], ranked by their
  strongest |ρ| with a vote fraction; the top 20.
- The 20 top nuisance-flagged latents are ranked by that nuisance |ρ|.

*Hashed 2026-09-25, before any token was extracted: SHA-1 over this section from its heading through the line above the blank line before this footer, plus a trailing newline: `e25e1e27af4114de4c639e9c3022c69ce4f4a3b3`.*

## Stop 3 — Part 3 results (2026-09-25; TOOL VALIDATION, no morphology claims)

Scored against the Part 3 hash (`runs/dd/sae/score.json`, `cards.json`).
- Extraction: 19.7 GB, 5 min.
- Training: eight SAEs, 8 epochs each, 8–19 min apiece.
- Final training NMSE: 0.037 for every M SAE, 0.128–0.137 for the untrained.

**S3 — FAIL (chosen = 8×, block 11: both sizes fail, so the rule keeps 8×).**

| SAE | VE (eval) | dead on eval | mean AUC drop | max drop | state |
|---|---|---|---|---|---|
| **M b11 8×** | 0.965 | 37% | **0.029** | 0.093 | **FAIL** |
| M b11 16× | 0.966 | 30% | 0.031 | 0.097 | FAIL |
| M b6 8× | 0.966 | 22% | 0.016 | 0.052 | UNEVEN |
| M b6 16× | 0.967 | 16% | 0.015 | 0.054 | UNEVEN |
| untrained b11 8× / 16× | 0.881 / 0.884 | 0% | 0.064 / 0.061 | 0.139 / 0.137 | FAIL |
| untrained b6 8× / 16× | 0.882 / 0.885 | 0% | 0.023 / 0.027 | 0.068 / 0.073 | FAIL |

- At 96.5% of variance, the missing 3.5% still carries 0.03 AUC of the probes' signal at block 11.
  Per the brief, the block-11 SAE is too lossy to interpret. Block 6 comes closer (UNEVEN).
- **Dead latents on held-out data (37%) far exceed dead-in-training (6.6% at the end).** Many
  latents fire on training galaxies but never on the test set. That is consistent with latents
  keyed to individual galaxies' whole-stamp cast, and a reason to prefer fewer, denser latents, or
  to train on the aligned corpus.

**S1 — PASS (block 11; block 6 also PASS).**
- 40 latents reach |ρ| ≥ 0.5 with a per-band offset (n = 3,116 galaxies with recorded offsets).
  Latent 328 has ρ +0.91 with i−r x; latent 1472 has −0.88 with i−r y.
- Their activation matches in the untrained SAE correlate at |ρ| ≤ 0.01.
- Reported: the untrained SAE's best latent over *all* of them reaches only 0.21 (g−r x), 0.20,
  0.23 (i−r x), 0.22 at block 11, and 0.19–0.29 at block 6.
- Location (reported): the offset latents are **dense**, firing on 23–48% of all tokens, mostly bare
  sky (`dd_sae_cards_nuisance_1.png`). That is Part 2's whole-stamp cast, now as SAE features. i−r
  dominates g−r.
- One caveat: several M candidates share one untrained match (latent 755), a generic high-activity
  latent. Activation matching is weak across networks with nothing in common.

**S2 — NOT DETECTED (supporting).**
- The best latent reaches precision 0.09 at recall 0.55 on 1,965 injected trail patches, at both
  layers.
- By contrast, **latent 2336 (i−r x, ρ 0.81) fires on real red, i-only streaks in the data**: natural
  single-band trails, coded as a band-offset feature.

**Feature cards** (`out/dd/dd_sae_cards_{interpretable,nuisance}_{1,2}.png`).
- 886 of 3,072 latents are live in the density window. **94% of them are nuisance-flagged.**
- Top nuisance latents: the band offsets (dense sky latents); padding (valid fraction ρ −0.80 to
  −0.83, e.g. 2743, 1806); brightness (1741, 2968, 2104, ρ ±0.73–0.78).
- **Latent 2452 fires on frame-edge stripes where v1's three bands are padded on different rows**, a
  v1 per-band padding artefact the aligned cutter's shared pad removes.
- The top "interpretable" latents are spatial motifs with |ρ| 0.30–0.52 against features-or-disk,
  spiral, bulge and arm number, each with specz close behind (0.3–0.4):
  - the sky just beside a galaxy's edge (983, 2933);
  - edge-on disc cores (1341);
  - disc and spiral texture (207, 2797);
  - stamp corners (825: positional, which the panel doesn't carry).
- The radial histograms are counts, not normalised by ring area, so they lean to the edge by
  construction.

**Tool reading for the aligned encoder.**
- The SAE and card pipeline works end to end. The S1 plant fires, with the same offset seen by
  Part 2's PCA and V4's sign flip.
- On M, the representation's sparse features are dominated by processing variables (offset,
  padding, brightness), as AA3a predicted.
- The rerun on the aligned encoder should:
  - use block 6 or a larger k if block 11 stays lossy;
  - add stamp position and ring area to the panel;
  - pre-register S3 against the chosen layer.

## Stop 3 decisions (user, 2026-09-26)

### 1. S1 matching — amendment recorded after the hash

- **What changed.** The hashed S1 said "nearest decoder match in the untrained-encoder SAE". It was
  scored with **activation-based matching** instead: the untrained latent with the highest |Pearson|
  of token-level activations over the same sae_eval tokens.
- **Why.** Decoder vectors of two SAEs trained on two different networks live in unrelated
  embedding spaces, so a decoder cosine between them is undefined.
- The change was made after the hash, disclosed at Stop 3, and accepted by the user as a recorded
  amendment (D27: labelled, not retroactive to the hash).
- **Primary S1 evidence, needing no matching:** at block 11, **no untrained-SAE latent exceeds
  |ρ| 0.23 with any band offset** (best per offset: g−r x 0.21, g−r y 0.20, i−r x 0.23, i−r y 0.22).
  M's best latent reaches 0.91, and 40 of M's latents reach ≥ 0.5.
- At block 6 the untrained best reaches 0.29 (g−r x); the statement is block 11's.
- The matched-pair figures (M latent against its activation match, |ρ| ≤ 0.01) are supporting only.
  Several M latents share one weakly matched untrained latent (755).

### 2. S3 — FAIL, recorded; no rescue on M

- **S3 stands as FAIL** at the chosen 8× block-11 SAE: mean AUC drop 0.029, against the 0.02 bar.
- Dictionary size, sparsity, block and training settings are **not** varied on M in search of a
  pass.
- **Rerun queue (aligned encoder):** S3 with identical settings (k = 32; 8× and 16×; blocks 11 and
  6; 8 epochs; Gao-scaled LR; AuxK as hashed) and the same hashed criteria and chosen-size rule.
- **Plausible reasons, not tested:**
  - The colour-cast latents consume dictionary capacity. The band-offset latents are dense (23–48%
    of all tokens) and 94% of live latents are nuisance-flagged, which leaves fewer latents for the
    residual the probes read.
  - The held-out dead rate (37%, against 6.6% at the end of training) fits galaxy-specific
    colour-cast features. They fire on the training galaxies whose cast they encode and never on
    held-out ones.
  - On the aligned corpus, with the cast gone, both should change. That is the rerun's question.

### 3. S2 diagnostic — pre-registration: is the trail represented at all?

Code: `artifacts/dd_s2_probe.py`.

**Data.** The same injected-trail set as S2, regenerated with identical seeds and draw order: 100
sae_eval galaxies and 1,965 trail patches (the count matches S2's, confirming it is the same set).

**Probe.**
- Token level on M's block-11 tokens. Label = trail patch (S2's definition); negatives = every
  other token of the same 100 galaxies.
- Standardised L2 logistic regression, C = 1. Split by galaxy, 70 train / 30 test (seeded).

**Criterion.** **DETECTED** if the held-out token AUC ≥ 0.75, else **NOT DETECTED**. Reading:
- DETECTED: S2's NOT DETECTED stands as an **SAE sensitivity limitation**. The absence of a latent
  does not imply absence from the model.
- NOT DETECTED: the plant was not represented at block 11, so S2 is **INVALID** as a test, not a miss.
- No redesign of the plant on M either way.

**Reported, not decisive:** the same probe on block 6, and on a pixel baseline (per patch, each band's
mean and SD).

**Plants, run before this hash** (`runs/dd/sae/s2_probe_plants.json`):
- tokens plus a noisy label column → AUC 0.997, DETECTED;
- labels scrambled within galaxy → AUC 0.503, NOT DETECTED.

*Hashed 2026-09-26, before the probe ran on the real labels: SHA-1 over this section from its heading through the line above the blank line before this footer, plus a trailing newline: `3355cac506152336415546f60a2f9d1a2d0fa367`.*

**Result: DETECTED** (`runs/dd/sae/s2_probe.json`).
- A linear probe on M's block-11 tokens finds the trail patches at held-out AUC **0.991** (block 6
  also 0.991; the pixel baseline 0.959).
- The trail is plainly, linearly represented in M's tokens. **S2's NOT DETECTED is an SAE sensitivity
  limitation:** no single one of the 3,072 latents (k = 32) picks it out at precision ≥ 0.8. The
  absence of a latent does not imply absence from the model.
- A trail is 2% of galaxies and ~0.15% of tokens, far rarer than the dense colour-cast features that
  take the capacity (item 2). A sparse dictionary trained without trails has little reason to
  spend a latent on one.

### 5. Part 4 pre-registration — ablating latent 328 through the hooks

Code: `artifacts/dd_circuits.py` (hooks, `latent_patch`) and `artifacts/dd_part4.py`.

**Intervention.**
- At block 11, M's 8× SAE latent 328 (S1's top latent: ρ +0.91 with i−r x) is removed from every token
  of every sae_eval galaxy: x′ = x − W_dec[:, 328] · z₃₂₈ / scale. The SAE's reconstruction error is
  kept, so nothing else moves.
- The ablated tokens are mean-pooled and read out.

**Read-outs.**
- **The i−r x offset probe:** sign(i−r x) from the block-11 pooled embedding. Standardised L2
  logistic, C = 1, trained on the SAE-training galaxies (probe-train split) with recorded offsets,
  and scored by AUC on sae_eval galaxies with recorded offsets.
- **The 37 morphology probes:** the ladder refits, AUC on each answer's eligible sae_eval galaxies.

**Expectation, with numeric bounds.**
- **E1:** the offset probe moves towards chance, |AUC − 0.5| falling by **≥ 0.05**.
- **E2:** the morphology probes barely move, mean |ΔAUC| over the 37 **≤ 0.01** and max **≤ 0.03**.
- States:
  - **AS EXPECTED**: E1 and E2;
  - **NO EFFECT**: E2 only;
  - **NOT SELECTIVE**: E1 only;
  - **NEITHER**.

**Reported, not decisive:**
- a density-matched control latent (random, within ±25% of 328's density) ablated the same way;
- all S1 candidate offset latents ablated together;
- the shift in AA3a's PC1 score;
- hook-path pooled against token-arithmetic pooled (consistency).

**Reachability** (D28; `artifacts/test_dd_circuits.py`, 4 passed):
- a planted latent whose decoder is a read-out direction moves that read-out and leaves an orthogonal
  one within 1e−4 (the AS EXPECTED shape);
- ablating a dead latent changes every read-out by exactly 0 (the NO EFFECT shape);
- capture matches `block_tokens`, and an identity patch is a no-op, with hooks detached.

**The prior is uncertain.** 328 is one of about 40 offset latents, several on i−r x (2389, 2336,
2725, …). Redundancy could leave the probe at its baseline, which would read NO EFFECT.

*Hashed 2026-09-26, before the ablation ran: SHA-1 over this section from its heading through the line above the blank line before this footer, plus a trailing newline: `185e1897886dd754bbf21b2b620adee865da37d5`.*

**Part 4 result: NEITHER** (`runs/dd/sae/part4.json`).

| ablation (block 11) | offset-probe AUC | towards chance | morph mean \|ΔAUC\| | morph max \|ΔAUC\| (answer) | PC1 shift (median \|Δ\|, SD) | state |
|---|---|---|---|---|---|---|
| **latent 328** | 0.973 → 0.948 | **0.024** (< 0.05) | 0.005 | **0.037** (arms = 4) | 0.52 | **NEITHER** |
| control 2526 (density-matched draw) | 0.973 → 0.973 | 0.000 | 0.001 | 0.004 | 0.05 | NO EFFECT |
| all 40 S1 offset latents | 0.973 → 0.690 | 0.283 | 0.010 | 0.057 (arms = 4) | 17.0 | NOT SELECTIVE |

- **E1 fails.** Removing 328 alone costs the i−r x probe only 0.024 AUC. The offset is held
  redundantly across about 40 latents, several of them on i−r x, as the pre-registration's prior
  warned. All 40 together take the probe most of the way to chance (0.69).
- **E2 fails narrowly on one answer:** "four arms" moves 0.037 against a max bound of 0.03, while the
  mean (0.005) is well inside. The offset latents are not perfectly orthogonal to the morphology
  read-outs, and removing all 40 costs 0.010 on average.
- **The control draw was not neutral.** The random density-matched latent, 2526, is itself an
  offset latent (i−r y, ρ 0.83), because the dense latents at 328's density are mostly offset
  latents. It leaves the i−r x probe exactly where it was, which reads as an axis-specificity check.
- **Hook plumbing is consistent:** block-11 pooled via hooks matches the stored-token arithmetic to
  0.0024 (the fp16 storage of the tokens). The unit tests give 4 passed, including the new
  dead-latent test.
- **Tool reading:** single-latent ablation on M's SAE is weak evidence of anything, because of
  redundancy. Feature-set ablation is the working unit.

## Stop 4 decisions (user, 2026-09-26)

Stop 3 results were accepted as reported.

### 1. Part 4 — NEITHER stands as hashed

#### Standing rule — random controls for ablations (all future ablations)

- Random controls are drawn from latents that **do not carry the ablated property**.
- They are matched on **removed energy** where feasible, else on density.
- Use **≥ 20 draws**, and report the distribution, not one draw.

Why: Part 4's single density-matched control, 2526, was itself an offset latent.

This amends the user's first wording ("not flagged as offset/nuisance, density-matched"), which was
infeasible on M:
- only 48 non-flagged latents lie in the offset set's density range, so 50 sets of 40 overlapped at
  Jaccard ≈ 0.6;
- non-flagged latents are vote-enriched by construction, which biases the null towards GENERIC.

#### Standing rule — only powered answers can trip a collateral bound (all future collateral bounds)

- An answer is **powered** on the evaluation galaxies iff its smaller class has **≥ 100** galaxies
  and its baseline probe AUC is **≥ 0.6**.
- Only powered answers can trip a collateral bound. The rest are reported, exploratory.
- On sae_eval, 28 of the 36 scoreable answers are powered.

Why: four-arms, the answer that tripped Part 4's E2, has 14 positives on sae_eval and a baseline AUC
of 0.505 (Hanley–McNeil SE ≈ 0.078). An at-chance probe cannot register a drop; it can only move by
noise.

#### Part 4 note (post hoc, D27; the verdict is unchanged)

- Part 4 reads **NEITHER as hashed.**
- Its E2 breach (four-arms, 0.037 against the 0.03 bound) lies within one SE of an unpowered answer
  (base AUC 0.505).
- Under the powered-answer rule that breach would not trip the bound. The NEITHER then rests on
  latent 328's E1 shortfall alone: 0.024 towards chance, against the 0.05 bound.
- 328's figures on the powered answers are given with the Part 4b result below.

The same caution applies to the "all 40 offset latents" row: its 0.057 maximum is also four-arms.

### Part 4b pre-registration — is the offset set's collateral SPECIFIC or GENERIC?

Code: `artifacts/dd_part4b.py`.

**Question.** Does removing M's offset features hurt morphology more than removing an equal amount of
non-offset representation?

**Tested set.** The 40 S1 offset latents at block 11 of M's 8× SAE, ablated together (Part 4's set).

**Exact token arithmetic.**
- Block 11 is M's last block, and the probes read its mean-pooled tokens.
- Ablating a set S is therefore pooled − (Ā_S · W_dec,Sᵀ) / scale, where Ā is the galaxy-mean SAE
  activation on the stored sae_eval tokens.
- Part 4 checked the hook path against this arithmetic to 0.0024, the fp16 token storage.
- The run reports the offset set's all-answer mean and max |ΔAUC| beside Part 4's hook values
  (0.0102, 0.0570) as a consistency check.

**Removed energy.**
- E(S) = 1_Sᵀ C 1_S, with C = (ZᵀZ / n) ⊙ (W_decᵀ W_dec / scale²).
- Z is taken over all 1.28 M held-out tokens. E is the exact mean ‖Σ_{j∈S} z_j d_j / scale‖² per
  token.
- The offset set's E is 1,164. It is 97% diagonal, and 85% of it sits in four dense latents
  (370, 275, 194 and 154).

**Primary pool.**
- Live latents (density ≥ 1e−4 on sae_eval), not in the offset set, whose galaxy-level |ρ| with every
  band offset (g−r x/y, i−r x/y) is < 0.3.
- Brightness- and nuisance-flagged latents are included.
- 852 latents.

**Primary sets.**
- 50 sets of 40 distinct pool latents, each within ±15% of the offset set's E, by rejection sampling.
- Proposal: latents drawn with probability ∝ C_jj^0.5.
  - α = 0.5 was chosen from the energies alone, so the median proposal sits near the target.
  - Per-latent pairing is infeasible: the offset set's four densest latents each carry more energy
    than any pool latent (the pool's maximum is 182).
- Achieved: 50 sets from 97 proposals, energy ratio 0.857–1.142, mean pairwise Jaccard 0.084.

**Statistics** (over the 28 powered answers; ΔAUC is ablated minus baseline, 37 probes refit as in
Part 4):
- **(a)** max |ΔAUC| over the powered answers;
- **(b)** mean |ΔAUC| over the powered answers.

**States and precedence:**
1. **INSUFFICIENT** if fewer than 20 accepted random sets.
2. Otherwise **SPECIFIC** iff the offset set's (a) exceeds the random sets' 95th percentile of (a)
   **and** its (b) exceeds their 95th percentile of (b). SPECIFIC means removing the offset features
   hurts morphology more than removing an equal amount of non-offset representation.
3. Otherwise **GENERIC**: removing any 40 latents of that energy costs about that much.
   - The run reports which of (a) and (b) cleared, when exactly one did. That split is descriptive,
     not a separate state.

**Reported, exploratory, no verdict:**
- **Per answer, all 36:** the offset set's signed ΔAUC against the random sets' 2.5–97.5% band,
  marked powered or not. Any answer outside the band is flagged. 36 answers are scanned, so about
  2 flags are expected by chance.
- **Secondary arm:** the cards' non-flagged pool (72 live latents outside the offset set), each
  offset latent density-matched within ×3.
  - ×3 is the loosest band at which a draw is feasible; draws that run out of candidates are
    redrawn.
  - 50 sets, Jaccard 0.62, median energy 0.44× the offset set's.
  - The pool is vote-enriched by construction (its latents' strongest |ρ| is with a vote), and its
    sets overlap heavily. It is reported for comparison only, with the state it would read.
- The offset probe's AUC under each random set, as an axis-specificity check.
- **Latent 328 alone on the powered answers,** post hoc, for the Part 4 note.

**Reachability** (D28; `runs/dd/sae/part4b_plants.json`, identical code path, same 50 sets):
- **SPECIFIC plant: fires.** The offset set's removal, plus 0.75 of the 37-probe weight subspace
  replaced by a permuted galaxy's, reads SPECIFIC.
  - A blind dose check (booleans only) found (b) clears from λ = 0.5 and (a) from λ = 0.75. λ = 0.75
    is therefore the smallest tested dose at which both clear.
- **GENERIC plant:** each random set tested against the other 49 reads SPECIFIC at a rate of
  **0.04**, bound ≤ 0.10.
- **Not saturated:** every random set's mean |ΔAUC| is > 0, and the offset probe's baseline sits off
  chance and off ceiling.

**Blinding.**
- The offset set's all-answer statistics were already known from Part 4. The null's quantiles were
  therefore never printed before this hash; the plants report states and booleans only.
- The offset set's powered-answer statistics have not been computed.

*Hashed 2026-09-26, before the run: SHA-1 over this section from its heading through the line above the blank line before this footer, plus a trailing newline: `81e5b6905c9b073b84d36508888992310c8ecf93`.*

**Part 4b result: GENERIC** (`runs/dd/sae/part4b.json`). Hash `81e5b690` re-verified after the run.

| statistic (28 powered answers) | offset set | random 95th pct (50 energy-matched sets) | cleared |
|---|---|---|---|
| (a) max \|ΔAUC\| | 0.038 (edge-on: no) | 0.174 | no |
| (b) mean \|ΔAUC\| | 0.009 | 0.060 | no |

- **Neither statistic clears.** Removing the offset set costs morphology no more than removing any 40
  non-offset latents of equal energy; per answer it mostly costs less (below).
- **Per answer (exploratory, 36 scanned):** the offset set lies **above** the random 2.5–97.5% band
  on 20 answers (19 powered) and **below** it on none. Above the band means less damage than 97.5%
  of the random sets.
  - Given the energy it removes, the offset set is unusually morphology-sparing. This fits the
    offset being a colour-cast direction largely orthogonal to the morphology read-outs (Part 2).
- **Axis-specificity:** the random sets leave the i−r x offset probe at 0.969–0.973 (baseline
  0.973), while the offset set takes it to 0.681.
- **Consistency:** the token arithmetic reproduces Part 4's hook values for the offset set on all
  answers: mean 0.010175 against 0.010175, max 0.057005 against 0.057005, a difference of 4 × 10⁻⁷.
- **Latent 328 alone on the powered answers** (post hoc, for the Part 4 note): max |ΔAUC| 0.010
  (bar), mean 0.003.
  - Under the powered-answer rule it would pass E2 (≤ 0.03, ≤ 0.01).
  - Part 4's NEITHER therefore rests on E1 alone, as the note states.

**Secondary arm** (exploratory, no verdict; the non-flagged pool at ×3 density, 0.44× the energy,
Jaccard 0.62): it would read **GENERIC**.
- Mean |ΔAUC| 0.009 against a p95 of 0.018; max 0.038 against 0.043.
- The offset set lies below this arm's band on four powered shape answers: edge-on yes/no,
  completely round and cigar-shaped (ΔAUC −0.034 to −0.038, against bands down to about −0.025).
  - This arm's sets carry less than half the energy, so the comparison is not like for like.
  - The primary arm's energy-matched sets hold all four inside their bands.
- It is a hint only, not a finding: the offset removal touches elongation read-outs somewhat more
  than a light, vote-enriched set does. A sub-pixel band misregistration elongating the colour
  image would do that.
- The other three below-band flags are unpowered (four arms, boxy bulge, medium winding).

**Tool reading.** The ablation machinery now supports set-level, energy-matched controls, and the
standing rules keep single draws and unpowered answers from carrying a verdict.

## Brief DD on M — closed (2026-09-26)

These tests validate the tools; none of them makes a morphology claim. Every verdict is M's, scored
against its hash. "Rerun" means the aligned encoder (trained on probe_v2 and pretrain_v2), run under
the same hashed criterion unless an amendment is recorded.

| test | hashed criterion (short) | hash | verdict on M | aligned rerun |
|---|---|---|---|---|
| **V1** model sensitivity | median \|ρ\| (M map against fully randomised) < 0.3, with CI; UNREACHABLE if the plants cannot reach FAIL | `b19ee5ee` | **UNREACHABLE**. The cascade shows the maps are set by blocks 1–4 | queued, same criterion |
| **V2** band-map planted positive | bright-edge mass of the band-offset map rises under a 1 px g roll (paired Wilcoxon, α 0.01) | `b19ee5ee` | **FAIL**. The plant modelled the wrong mechanism | queued, **amended**: localised plant (g misaligned within one footprint or quadrant; mass inside the region must rise). A new pre-registration is hashed before the rerun |
| **V3** maps locate GZ3D structure | original: patch AUC against the centre prior | withheld (D28, saturated) | **WITHHELD** | superseded by V3′ |
| **V3′** same-radius (post hoc) | ring-stratified ΔAUC against the untrained map **and** the brightness map, bootstrap CIs | `97e3ed52` | spiral **REVERSED** (0.665 against 0.748 and 0.810); bar **withheld** (brightness comparator at 1.00) | queued, amended criterion. Bar stays withheld unless a non-saturating comparator is pre-registered |
| **V4** equivariance | exploratory, no bar | `b19ee5ee` | median ρ 0.26–0.65 across rot90/rot180/flip; the band map's sign flips under rot180 and flip | queued, exploratory |
| **S1** offset latent | a latent at \|ρ\| ≥ 0.5 with a band offset, whose untrained counterpart is < 0.2 | `e25e1e27` + post-hash amendment (activation matching) | **PASS**. Matching-free: untrained best 0.23 against M 0.91; 40 M latents ≥ 0.5 | queued, amended (matching-free statement primary) |
| **S2** injected trail | a latent with precision ≥ 0.8 at recall ≥ 0.2 on injected trails | `e25e1e27` | **NOT DETECTED** (best precision 0.09) | queued |
| **S2 diagnostic** | token-level linear probe for the trail, held-out AUC ≥ 0.75 | `3355cac5` | **DETECTED** (0.991). S2's miss is an SAE limitation | queued with S2 |
| **S3** faithfulness | mean probe-AUC drop ≤ 0.02 (UNEVEN if any answer drops > 0.05) | `e25e1e27` | **FAIL** (0.029; 37% dead on held-out) | queued, identical settings; **amended**: only powered answers can trip UNEVEN |
| **Part 4** ablate latent 328 | E1: ≥ 0.05 towards chance; E2: morphology mean ≤ 0.01, max ≤ 0.03 | `185e1897` | **NEITHER**. It rests on E1 (0.024); the E2 breach was an unpowered answer (note, post hoc) | **not queued**; superseded by set ablation |
| **Part 4b** offset-set collateral | offset set's max **and** mean \|ΔAUC\| (powered answers) above the 95th percentile of 50 energy-matched non-offset sets | `81e5b690` | **GENERIC**. The offset set costs less than energy-matched random sets on 20 answers | queued, with the standing control rule |
| Part 2, feature cards, V1 (a)–(c), latent 2336 | exploratory, not hashed | — | the offset is a whole-stamp colour cast; 94% of live latents are nuisance-flagged; 2336 is not a streak flag | cards and Part 2 rerun as exploratory |

**Standing rules that bind every rerun:**
- random controls come from latents without the ablated property, energy-matched where feasible,
  with ≥ 20 draws;
- only powered answers can trip a collateral bound. Powered status is recomputed on the aligned
  encoder's evaluation galaxies, since baseline AUCs change with the encoder;
- plants are run and pass before each hash (D28).

**Settled (user, 2026-09-26):** the powered-answer rule governs S3's UNEVEN clause for the aligned
rerun. See the S3 amendment below.

**Finding (Brief DD, Part 4b; user, 2026-09-26).** On M, the colour-offset features form a separable nuisance subspace: removing them costs morphology less than energy-matched random ablation (4b GENERIC; less damage than 97.5% of random sets on 20 of 36 answers, 19 of them among the 28 powered). 85% of the offset energy sits in 4 latents. (The user's wording said "20 of 28 powered"; the 20 answers above the band include one unpowered answer, two arms, so the powered count is 19 of 28.)

**Method limits on M, plainly:**
- **Occlusion maps do not locate arms at 16 px on M.** On GZ3D spiral masks, M's map ranks arm
  patches below both the untrained encoder's map and plain r-band brightness at the same radius
  (V3′ REVERSED). The cascade shows the maps are largely set by blocks 1–4, not by M's trained top
  blocks (V1).
- **The SAE misses features that are linearly present.** No latent picks out an injected trail
  (best precision 0.09), yet a linear probe on the same block-11 tokens finds it at AUC 0.991 (S2
  and its diagnostic). At block 11 the SAE is also too lossy for the probes' signal (S3). Absence of
  a latent is not absence of a feature.
- **Single-latent ablation is ineffective where features are redundant.** Latent 328 alone moves
  its offset probe 0.024 towards chance; all 40 offset latents move it 0.283 (Part 4). Set ablation
  with energy-matched controls is the working unit (Part 4b).

## Stop 5 decisions (user, 2026-09-26)

### S3 — pre-rerun amendment: only powered answers can trip UNEVEN

- **Amendment, for the aligned rerun only.** S3's UNEVEN clause ("mean drop ≤ 0.02 but some answer
  drops > 0.05") counts **powered answers only**: smaller class ≥ 100 and baseline AUC ≥ 0.6, on the
  aligned encoder's own evaluation galaxies. Unpowered answers' drops are reported, exploratory.
- **Reason:** an unpowered per-answer bound fires on noise, as Part 4's four-arms breach did (14
  positives, base AUC 0.505, SE ≈ 0.08).
- It is recorded before the rerun's hash, so it binds that hash and is not post hoc for it.
- **M's S3 stays FAIL**, which rests on the mean drop (0.029 against 0.02) and is untouched by the
  amendment.

**Retrospective note on block 6's UNEVEN** (note only; verdicts unchanged; `runs/dd/sae/M_b6_x*.eval.json`,
powered status from Part 4b's list):
- **8×:** the answers over 0.05 are edge-on yes (0.052) and edge-on no (0.050). **Both powered.**
- **16×:** edge-on no (0.054), edge-on yes (0.054) and cigar-shaped (0.050). **All powered.**
- So block 6's UNEVEN would stand under the amended rule too. It was not a noise trip.
- **Descriptive:** the same elongation answers (edge-on, cigar-shaped, completely round) head the
  per-answer losses at block 11 as well (0.054–0.097). Part 4b's secondary arm flagged the same
  family as the one the offset removal touches. Both observations are exploratory. Whether shape
  read-outs sit in the directions an SAE (or a colour-cast removal) disturbs first is a question for
  the aligned rerun, not a finding.
