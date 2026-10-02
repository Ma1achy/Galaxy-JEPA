# Layer 2 leakage audit — pre-registration (DRAFT, not hashed)

**Question.** Can the pixels of the re-pulled corpora (pretrain_v2, probe_v2) say how a stamp was
processed, beyond what the galaxy's own physics already says?

**Status.** Draft v4, 2026-10-02. v3 applied the user's 2026-09-28 decisions: TRACE is reported, not
blocking; sky is in the physics baseline, with the plan's baseline (no sky) reported beside it; the
baseline's catalogue physics is pulled now and pinned. v4 applies the 2026-10-01 decisions (see
"Settled (user, 2026-10-01)"): the redshift proxy is dropped, `petroRad_r` joins the baseline, and
S5's revised requirement is approved. The plants were rerun under v4 (results at the end). Two plants failed as first written (v2), and the
fixes are declared in "Revised before the hash after a plant failed". The user approves before
hashing. Once settled, the section from "## Pre-registration" through its last content line is
hashed (SHA-1, trailing newline) **before any pretrain_v2 or probe_v2 pixel is read for the audit**.
The hash is written into `leakage_audit.py`'s `PREREG_SHA1`; `audit` refuses to run until the
section matches it (`assert_hashed`). Code: `artifacts/leakage_audit.py`.

## Pre-registration

### What "leakage" means here

The plan's definition (plan A7, "Layer 2 — leakage audit", quoted in full) is primary:

> - **Sample.** 20k per corpus, held-out folds.
> - **Predictors.**
>   - Ridge on image statistics: per-band moments, centroids, cross-band correlation, lag-1
>     autocorrelation, border statistics.
>   - A small CNN: 4 conv layers, 20k stamps, MPS, a few minutes per target.
>   - Both are fitted to recover each processing variable from the pixels alone.
> - **Pixels must not beat a physics-only baseline (amendment 3).** For each processing variable, a
>   baseline predictor is given only catalogue physics: magnitude, size, SNR, colour, surface
>   brightness, a redshift proxy and PSF width. LEAK = the pixel predictors' held-out excess
>   (ΔR² / ΔAUC) over that baseline, with a bootstrap CI excluding 0.
>   - per-band s_x and s_y;
>   - frac(origin);
>   - corpus membership. The corpora differ by design (D6), so magnitude/size matching alone would
>     read LEAK on physics; the baseline absorbs that.
> - **Readable, reported only:** psfWidth_r, sky level, camcol.
> - **Planted.** A copy of the sample with v1's recorded per-band offsets re-injected must read LEAK
>   on the shift variables.
> - **States:** CLEAN / LEAK (variable, R²/AUC, CI). **No training on LEAK.**

So **leakage** here is *processing-variable leakage*: information about how a stamp was cut
(its per-band sub-pixel shift, where the target fell on the frame grid, which corpus's pull it came
from) that a model could read from the pixels and that the galaxy's physics does not explain. It is
the failure that produced M's PC1/PC2 (37% of variance on v1's misregistration, AA3a): an encoder
spends capacity on whatever processing leaves in the pixels.

**A second, separate criterion: identity leakage** (the split-leak sense a reader may expect). The
same galaxy reaching both the pretraining corpus and the probe corpus, under one objID or two.
- It is included because it is cheap (coordinates only, no pixels) and it is what "leakage" means
  to most readers.
- It is kept separate because it is a different failure, with a different guard and remedy. The
  exact-objID case is already made impossible in code (`data/orchestrate.resolve_corpora`,
  `docs/spec/splits.md` §2); the audit re-asserts it on the manifests it reads and adds what objID
  dedup cannot see.
- Its state is reported beside Layer 2's and does not change it (precedence below).

### Common ground

- **Data.** pretrain_v2, deduplicated against probe_v2 by `resolve_corpora` (a shared galaxy
  belongs to probing), and probe_v2. No encoder is involved; no GZ2 vote column is read.
- **Completeness (INSUFFICIENT otherwise).** Each corpus's pull is complete: its `cut_log.csv` rows
  plus its `failed.csv` rows equal its target list (`.sciserver_work/<corpus>_all_targets.csv`),
  and `manifest.json` exists. The audit does not run on a partial pull.
- **Sample.** 20,000 per corpus: the 20,000 smallest `assignment_unit(objID, 0, salt="leakage")`.
  probe_v2 is sampled from all three splits (no labels are read, so there is nothing to leak into
  the probe splits).
- **Folds.** 5, by `floor(5 · assignment_unit(objID, 0, salt="leakage-fold"))`. Every prediction
  scored is out-of-fold. A galaxy keeps its fold in the pooled corpus-membership block.
- **Seeds.** 0 throughout: sample and folds as above; CNN seed 0 + fold; bootstrap seeds fixed per
  variable in the code.

### The processing variables

Stated (each gets a state):

| variable | source (`cut_log.csv`) | per corpus |
|---|---|---|
| s_x, s_y per band (6) | `g_sx` … `i_sy`: the v2 cutter's applied shift, ∈ (−0.5, 0.5] | yes |
| v1 in-stamp offsets, g−r and i−r, x and y (4) | `{g,r,i}_v1_rel{x,y}` differences | yes |
| corpus membership | pretrain_v2 = 0, probe_v2 = 1 | pooled 40,000 |

- **frac(origin) is s_b.** The cutter sets s_b = 159.5 − (x_b − o_b) (`sciserver_cut_v2.py`,
  `padded_cut`), so the target's sub-pixel phase on the frame grid is s_b to rounding. It is not a
  separate test. The audit asserts the identity on every row and fails loudly if it breaks.
- **The v1 offsets are added.** They are the misregistration M read (the aligned comparison's four
  offsets) and the variables the plan's plant re-injects. They are linear in the per-band phases but
  differences can be readable where single bands are not.
- That is 2 corpora × 10 + 1 = 21 stated variables, × 2 predictors = **42 tests: the family**.

Readable, reported only (no state; the pixels' own R², or AUC for camcol):
- psfWidth_r, sky_r, camcol (one-vs-rest AUC per camcol), as the plan says;
- `valid_frac` and `edge_dist` (the smallest over bands). The padding is visible by design (the
  validity detector must find it), so a state on them would fire by construction. Whether the
  padding carries s_b is covered by the s_b tests themselves.

### Predictors

- **Ridge on image statistics** (`image_stats`, 65 features). Per band:
  - mean, SD, skew and kurtosis over the whole stamp and over the central 32²;
  - windowed centroid (Gaussian σ = 3 px, 6 iterations, background-subtracted) and second moments;
  - lag-1 autocorrelation on the border strips (x on the top strip, y on the left);
  - border-ring (outer 8 px) median, SD and zero fraction.

  Across bands:
  - centroid differences g−r and i−r;
  - central Pearson correlation for g–r, r–i and g–i;
  - the ±1 px correlation asymmetry, g–r and i–r, in x and y (the shift-sensitive part).

  Standardised; `RidgeCV` over α ∈ 10^[−2, 4] (13 values), one fit per target, per fold.
- **Small CNN** (`_net`).
  - Architecture: 4 conv layers (5×5 stride 2, then 3×3 stride 2; widths 32, 64, 64, 128; BN, GELU),
    4×4 average pool, MLP 2048 → 128 → outputs.
  - Input: the three bands through asinh on a fixed per-band scale, plus two coordinate channels.
    The scale is the sample's median border-noise σ, with no per-stamp normalisation, so the noise
    level stays readable. The coordinate channels make absolute position visible.
  - Budget: 8 epochs, batch 128, AdamW (lr 2×10⁻³, wd 10⁻⁴), one-cycle schedule, on MPS. The best
    epoch on a 10% inner split of the training folds is kept. No augmentation: a flip would reverse
    s_x.
  - Measured on this machine: 46 ms per batch at 64², 198 ms at 256². The audit's CNN is therefore
    about 17 minutes per corpus and 35 for the pooled pair, and runs twice (the audit, then the
    re-injected plant).
  - One multi-output network per corpus and fold predicts every stated and reported target (masked
    MSE on standardised targets). Corpus membership has its own network on the pooled pair (BCE).
- **Declared: multi-output, not one network per target.** The plan says "a few minutes per target";
  one network per target is 21 × 5 fits. Multi-output keeps the budget and loses sensitivity only if
  the targets compete for capacity; the plants measure the result.

### The physics baseline and the excess

- **Baseline inputs** (catalogue physics only; the per-corpus metadata lacks most of them, see the
  table). They come from one public SkyServer SQL pull (no token; `run_sql`, as the pilot's
  `_rowcol`) for the audit's 40,000 sampled objects, made before the hash (`leakage_audit.py pull`)
  and pinned: `out/leakage/audit_physics.csv`, with its query and SHA-1 in
  `out/leakage/audit_physics.json` (see "Plant evidence"). `audit` refuses a physics file that no
  longer matches the record. The sample is drawn from the cut logs alone (`_samples`, shared by
  `pull` and `audit`); no pixel is read.

  | plan's input | column used | source |
  |---|---|---|
  | magnitude | `modelMag_r` | pull (also in both metadata files) |
  | size | `petroR50_r` | pull (metadata has `petroRad_r` only) |
  | SNR | `snr_r` = f(`modelMagErr_r`) | the package's one derivation site (`with_derived_columns`) |
  | colour | g−r, r−i from `modelMag_{g,r,i}` | pull (neither metadata file has g or i) |
  | size (Petrosian radius) | `petroRad_r` | each corpus's own `metadata.csv` (v4; ≤ 0 or −9999 read as missing) |
  | surface brightness | `modelMag_r` + 5 log₁₀ `petroR50_r` | derived |
  | ~~redshift proxy~~ | ~~`Photoz.z`~~ | **dropped (v4)**: see below |
  | PSF width | `Field.psfWidth_r` | pull (pretrain metadata has none) |
  | sky level | `sky_r` (PhotoObjAll) | pull; **added to the plan's list** (below) |

  **INSUFFICIENT** if more than 2% of either corpus's sample lacks any baseline input. Missing
  values are SDSS's −9999, read as missing.
- **No redshift proxy (v4; user, 2026-10-01). This departs from the plan's list.** The pull returned
  every object, but DR17's `Photoz` has no usable `z` for 21.2% of probe_v2's sample and 2.3% of
  pretrain_v2's, which fails the 2% gate. `Photoz.z` stays in the pinned pull (the query is
  unchanged) and is not read.
  - **Why dropping is the safe direction.** A weaker baseline can only cause a false alarm on corpus
    membership (more of the corpus difference is left for the pixels to explain); it can never hide
    a leak. The shift variables do not depend on redshift.
  - **Why not keep it with the gaps.** The missingness pattern itself (21.2% against 2.3%) would
    separate the corpora, so a baseline allowed to read it would gain corpus information from the
    catalogue's holes and could absorb a real corpus leak.
  - **Not filled from specz.** Every probe_v2 galaxy has one and pretrain_v2 has none, so the input
    would differ by corpus in kind.
  - With the proxy dropped, the baseline misses 0% of pretrain_v2's sample and 0.03% of probe_v2's
    (6 galaxies without `petroR50_r`), inside the 2% gate.
- **Magnitude and size, confirmed (v4).** Apparent magnitude is `modelMag_r`. Angular size was in as
  `petroR50_r` (the Petrosian half-light radius); `petroRad_r` (the Petrosian radius itself) was not,
  and is **added** in v4 from the corpora's metadata. The synthetic plants keep their single `size`.
- **Sky is in the baseline, stated on principle.** Sky level is an observing condition, like PSF
  width, which the plan's baseline already holds: it is not something the cutter did. A corpus
  difference in sky is therefore physics-explained, not a leak.
  - **This departs from the plan's list** (A7: magnitude, size, SNR, colour, surface brightness, a
    redshift proxy and PSF width). So the plan's baseline, without sky, is scored beside it **from the
    same pixel predictions** (`run(..., alt=...)`, read with `view`), and both results are reported.
    The baseline with sky decides the state; the one without is reported, with no state of its own.
- **Conditions (corpus membership only):** camcol one-hot. Under the reported baseline without sky,
  the conditions are `sky_r` plus camcol one-hot, as in v2.
- **Model.** Gradient-boosted trees (`HistGradientBoosting`: 300 iterations, learning rate 0.05,
  15 leaves, ≥ 100 per leaf, no early stopping, seed 0), cross-fitted on the same 5 folds. The
  selection that separates the corpora is non-linear (magnitude and size cuts), so a linear
  baseline would under-absorb it and hand the difference to the pixels.
- **Excess.** Per variable and pixel predictor: the out-of-fold predictions of trees on
  **physics ⊕ the pixel predictor's out-of-fold prediction**, against trees on physics alone:
  ΔR² (ΔAUC for corpus membership).
  - This is the information in the pixels beyond physics. The literal "pixel R² minus physics R²"
    would go negative whenever physics predicts the variable better than pixels, and would hide a
    leak.
- **Bootstrap.** 20,000 Poisson galaxy-bootstrap draws of the out-of-fold predictions, shared
  across the models being compared (paired by galaxy).
  - The bound: the one-sided lower bound of Δ at α = 0.05 / 42 (Bonferroni over the family).
  - The plan's "CI excluding 0" is this bound above 0; a pixel predictor cannot leak by being worse
    than physics.
  - **Resolution** (`assert_resolution`): 20,000 × 0.05/42 = 23.8 draws lie beyond the bound (the
    floor is 10). With fewer, every variable would read CLEAN by construction, so the audit refuses
    to run.
  - **Declared:** the bootstrap resamples galaxies around fixed fits, so fit-to-fit variance is not
    in the bound.
- **Magnitude floor ε = 0.01** (ΔR² or ΔAUC). ΔR² 0.01 is |ρ| ≈ 0.1, the bound every pilot check
  used (`repull_findings.md`).

### States

**Per stated variable, per predictor** (Δ = excess, lo = the family-corrected lower bound):

| state | rule |
|---|---|
| **LEAK** | lo > 0 and Δ ≥ ε |
| **TRACE** | lo > 0 and Δ < ε: significant, below the floor (D27: significance disagreeing with magnitude) |
| **UNRESOLVED** | lo ≤ 0 and Δ ≥ ε: the magnitude is there, the resolution is not |
| **CLEAN** | lo ≤ 0 and Δ < ε |

- A variable's state is the worse of its two predictors', in the order LEAK > UNRESOLVED > TRACE >
  CLEAN.
- **INSUFFICIENT** (per variable): fewer than 15,000 rows with the variable recorded.
- **Labels (reported with the state; they change no state).**
  - **PHYSICS-EXPLAINED:** CLEAN, but the pixel predictor's own R² (or AUC − 0.5) has lo > 0. The
    pixels read the variable, and physics accounts for it. Corpus membership is expected to read
    this.
  - **CONDITIONS-EXPLAINED / NOT CONDITIONS-EXPLAINED** (corpus membership, when not CLEAN): whether
    the excess survives adding the conditions (camcol; under the reported baseline, sky_r and camcol)
    to the baseline.
    - The plan's baseline has PSF width but not sky. So under it a corpus difference in sky reads
      LEAK, and this label says so; under the gating baseline sky is physics and reads CLEAN.
    - **Explained** iff, for every predictor that is not CLEAN, the excess over physics + conditions
      is below ε (CLEAN or TRACE). Revised after S5; see below.
  - **CAMCOL-STRUCTURED (camcol c) / UNSTRUCTURED:** see the routing below.

**The audit's state, in precedence:**
1. **INSUFFICIENT:** a pull is incomplete, the physics pull misses > 2% of a sample, or a stated
   variable is INSUFFICIENT. Not read; nothing downstream runs.
2. **INVALID:** the re-injected plant (below) does not read LEAK on all 20 shift variables. The audit
   has not shown it can see the leak it exists to see, so a CLEAN would be unreadable. Nothing
   downstream runs.
3. **LEAK:** any stated variable LEAK. **No training.** Nothing downstream is scored.
4. **UNRESOLVED:** any variable UNRESOLVED, none LEAK. Treated as LEAK: a leak of material size
   cannot be excluded.
5. **TRACE:** any variable TRACE, none worse. **Reported, not blocking:** A8 and training proceed,
   and the TRACE variables, their excesses and bounds are reported with the state.
   - **Why.** The audit resolves far below anything an encoder could exploit. At α = 0.01 (1% of v1's
     offsets, about 0.003 px rms) 19 of 20 shift variables read LEAK (plant S3b); M's misregistration
     was about 0.4 px rms. At n = 40,000 a significant excess below ε is expected even for clean data
     (S0's corpus bound sat at −0.0002, a hair from TRACE).
   - **This departs from the plan's literal wording**, whose rule is "CI excluding 0" with no floor.
     Under the plan's rule TRACE would block. The departure is the user's decision (2026-09-28), made
     before the hash; it is declared here rather than taken silently.
6. **CLEAN:** every stated variable CLEAN. A8 (freeze, cache, identities) may proceed.

### Camcol routing (the staged fallback, `repull_findings.md:408`)

- The recorded fallback is triggered "if the Layer 2 audit reads LEAK on camcol". Under A7, camcol is
  *readable, reported only*, and camcol is expected to be readable anyway (each camcol is a
  different CCD, with its own gain and noise). A LEAK state on camcol itself would fire for reasons
  no per-band shift table can correct.
- **Reading chosen.** The fallback fires when a **shift variable** reads LEAK, TRACE or UNRESOLVED
  and its excess is **camcol-structured**. The remedy is a per-camcol, per-band shift table, so this
  is the case it can correct.
- **The test** (`camcol_structure`), per corpus, over its non-CLEAN shift variables, each at its
  larger-excess predictor:
  - d̄_c = the mean over variables of (Δ inside camcol c − Δ outside c), on 4,000 shared Poisson
    draws;
  - **CAMCOL-STRUCTURED (camcol c)** iff some d̄_c's one-sided bound at α / (6 camcols × 2 corpora)
    exceeds 0, otherwise **UNSTRUCTURED**.
- **Routing.**
  - LEAK + CAMCOL-STRUCTURED → fallback step 1: the per-camcol, per-band correction table, measured on
    field stars and fed into the Fourier shift.
  - LEAK + UNSTRUCTURED → no pre-set remedy. Stop and report; a new cutter design is fallback step 2.

### Identity criterion (coordinates only)

Read from both cut logs' `object_id`, `ra` and `dec`; the probe splits come from
`assign_three_way(seed=0)`.
- **Shared objIDs.** `resolve_corpora` removes every probe galaxy from pretrain_v2, and its
  post-condition raises `LeakError` if one survives. That is an **error, not a state**. The raw
  overlap before dedup and the number removed are reported.
- **Near-duplicates:** a probe_v2 galaxy with a deduplicated pretrain_v2 target within **1.0″**
  under a different objID. That is the same galaxy under two IDs (a secondary detection or a
  deblend variant), which objID dedup cannot see.
  - **NEAR-DUPLICATES (n per split)** if any; **CLEAN** otherwise.
  - Pairs at 1–3″ are reported. 3″ is the probe_v2 GZ2-separation flag's bound.
- **Footprint overlap** (reported, no state): the probe_v2 galaxies per split lying inside some
  pretrain_v2 stamp's 256-px footprint (|Δ| < 50.7″ on both axes).
  - At SDSS density most will be. Pretraining is label-free, and seeing a probe galaxy off-centre
    in someone else's stamp is inherent to cutting stamps from a survey.
  - A threshold would have no grounding, so this carries no state.
- **Consequence.** NEAR-DUPLICATES is reported and does not gate Layer 2 or A8. Remedying it would
  exclude those targets from pretrain_v2. But pretrain_v2's target list is M's (v1) pretrain list, so
  M saw the same near-duplicates. Excluding them would add a corpus difference to the aligned
  comparison, which is built to differ from M only by the alignment. **Whether to exclude them is
  the user's call before the hash** (flagged).

### Precedence over everything downstream

- The audit runs **after the pull and before any training** (plan A7). Its state is the
  precondition for A8 and for the aligned comparison (`aligned_comparison.md`, Common ground,
  "Precondition").
  - **CLEAN** and **TRACE** let A8 (freeze, cache, identities), A1/A2 training, and the aligned
    comparison proceed. TRACE is reported with the state (above).
  - **INSUFFICIENT, INVALID, LEAK and UNRESOLVED:** nothing in the aligned comparison is scored, and
    no encoder trains on v2.
- No downstream result can revise the audit's state. A state added after the audit reads is post
  hoc (D27): it applies to future audits only.

### The re-injected plant (the plan's, run inside the audit)

- A copy of each corpus's sample, with v1's recorded per-band offsets re-injected: each band moves
  by (v1_rel − 127.5) px (`reinject`: reflect-pad to odd P, the cutter's Fourier shift, crop).
- It goes through the identical path, with the same folds, seeds and family.
- **Must read LEAK on all 20 shift variables.** Otherwise the audit is **INVALID**.

### Plants (D28; `leakage_audit.py plants`, run before the hash)

The audit's data cannot be read before the hash, so the plants run on synthetic stamps and on
held-aside real data. That data is **v1 probe stamps (`data/probe`, M's corpus), 4,000 of them**,
not pretrain_v2 or probe_v2. Every plant goes through `run`, the audit's path, with the audit's
family (42) and bootstrap.

**Synthetic** (`synth`): two corpora. S0, S3, S3b and the calibration use 20,000 each (the audit's
n). S1, S2, S4 and S5 use 10,000 each: they are gross plants, and halving them saved about 1.5 h under
the shared lock.
- Stamps are 64² rather than 256², to fit memory; the code path is otherwise identical.
- Each galaxy is a bulge + disc with a colour gradient, a per-band PSF and white sky noise. It is
  rendered on the frame grid at a random sub-pixel phase per band, then cut as the v2 cutter cuts:
  padded to odd P, `fourier_shift` by s_b, cropped.
- Corpus B is probe-like: brighter, larger and redder.
- The catalogue physics carries measurement noise.

| plant | construction | must read |
|---|---|---|
| S0 clean | the v2 pipeline in both corpora | every variable CLEAN; corpus PHYSICS-EXPLAINED |
| S1 v1 re-injected | α = 1 re-injection of v1's offsets (−s_b) in both corpora | all 20 shift variables LEAK; both corpora UNSTRUCTURED; corpus CLEAN |
| S2 pipeline differs | corpus B cut with the bilinear shift (the pilot's resampling plant) | corpus LEAK, NOT CONDITIONS-EXPLAINED; A's shift variables CLEAN |
| S3 threshold | re-injection at α*, the dose putting the median shift-variable excess at ε | every shift variable LEAK or TRACE, never CLEAN (**failed**; superseded by S3b) |
| S3b threshold (revised) | α = 0.01 re-injection, fresh seed | no shift variable CLEAN; at least one variable at the threshold (its larger excess in [ε/2, 2ε]), and every such variable LEAK or TRACE |
| S4 camcol-structured | α = 1 re-injection in camcol 3 only | every shift variable LEAK or TRACE; each corpus CAMCOL-STRUCTURED at camcol 3 only |
| S5 conditions | corpus B's sky carries a +0.3σ pedestal, and the catalogue sky records it | corpus not blocking, CLEAN or TRACE (v3: the baseline's sky absorbs it; revised, see below); shift variables CLEAN. Without sky (reported): corpus LEAK, CONDITIONS-EXPLAINED |

- **What "at the threshold" should read, and why** (the reasoning that follows is S3's, as first
  written; S3b keeps it per variable). α* is set so the median shift-variable excess sits at ε.
  - At the audit's n, the bound at ε is well above 0, so the threshold plant is significant and
    must never read CLEAN.
  - Whether a given variable lands LEAK or TRACE is a coin flip by construction. That is the
    point: the floor separates those two names, and both block.
  - α* is calibrated on a separate seed (`calibrate`, seed 900) and never on S3's realisation. The
    rule is a log–log line through the sweep α ∈ {0.01, 0.02}, since R² ∝ α² for a weak shift.

**Real, held-aside** (plant R): 4,000 v1 probe stamps (256², the audit's size).
- The stamps are the 4,000 smallest `assignment_unit(objID, 0, "leakage-plant")` with finite
  physics.
- g is Fourier-shifted by t_g ~ U(−0.5, 0.5]². t_r and t_i are drawn and **not** injected.
- phys_mix = z(modelMag_r) + z(log petroRad_r) + N(0, 1).
- The baseline uses the probe metadata's physics (modelMag_r, snr_r, petroRad_r, the surface
  brightness from them, psfWidth_r). **No colour**: v1 metadata has none. **No redshift** (v4: specz
  dropped with the audit's proxy).
- **Must read:** t_g x and y LEAK; t_r and t_i CLEAN; phys_mix CLEAN and PHYSICS-EXPLAINED.
- **Declared:** at n = 4,000 plant R tests the logic on real pixels, not the audit's power.
- **Declared (v3):** v1 probe metadata has no sky column, so R's baseline has no sky and R has no
  without-sky reading. R tests the pixel predictors and the excess on real stamps; sky plays no part
  in what it plants.

**Identity** (plant I, synthetic catalogues in a 4 deg² patch: 5,000 probe, about 20,000 pretrain):
- clean → CLEAN;
- 7 pretrain rows placed 0.5″ from probe galaxies under new IDs → NEAR-DUPLICATES (7);
- 10 shared IDs → the raw guard raises `LeakError`, and `resolve_corpora` removes exactly those 10.

### Revised before the hash after a plant failed

Each failure is kept in `plants.json` beside its fix. Neither fix is a state applied to audit data;
the audit has not run.

1. **S3, the threshold plant, failed: every variable read CLEAN.** The calibration's two points (α
   0.01 → median excess 0.105; 0.02 → 0.285) gave a log–log slope of 1.44. Extrapolated 5× below
   the sweep, that put α* at 0.0019. There every excess was |Δ| ≤ 0.004 and every state CLEAN: the
   response falls much faster below α = 0.01 than the line assumed.
   - At α = 0.01 the variables' excesses already spanned 0.006–0.27, so no single dose puts every
     variable at the floor. A median rule is the wrong target.
   - **S3b** replaces it. The dose is α = 0.01, the sweep point where the weakest variables (A's g
     band) sat at the floor and read TRACE in the calibration run. The seed is fresh (350). The
     requirement is per variable: the variables whose excess lands in [ε/2, 2ε] must not read CLEAN.
   - S3b was defined before it ran. `calibrate` now also stores per-variable excesses.
2. **S5 failed as first labelled** (NOT CONDITIONS-EXPLAINED, where CONDITIONS-EXPLAINED was
   required).
   - The conditions baseline reached AUC 0.99999. The pixels added 1.4 × 10⁻⁵ AUC over it, with a
     bound of 2.8 × 10⁻⁶: significant, and 700× below ε. The draft rule demanded CLEAN.
   - The rule now uses the floor (above). S2's and S5's labels were recomputed from the stored
     numbers without a re-run. S2 is unchanged: NOT CONDITIONS-EXPLAINED, with a conditions excess of
     0.136.

#3. **v3's S5 failed as first written** (corpus CLEAN required under the sky baseline; it read
   TRACE).
   - With sky in the baseline, physics alone separates the corpora at AUC 0.999987. The pixels add
     1.35 × 10⁻⁵ AUC (ridge and CNN alike), with a bound of 2.7 × 10⁻⁶: significant, about 740× below
     ε. This is v2's S5 residue again, now on the corpus state itself rather than the conditions
     label.
   - The requirement is now "corpus not blocking (CLEAN or TRACE); shift variables CLEAN", which is
     what the plant exists to show: a corpus difference in an observing condition the baseline
     holds does not block. The first-written result is kept in `plants.json` (`S5.first_written_v3`)
     and S5 was re-scored from the stored numbers, not re-run.

## Plant evidence (pinned)

SHA-1 of each file as it stands after the 2026-10-02 v4 plant runs. Re-pin if anything changes
before the hash.

| file | role | SHA-1 |
|---|---|---|
| `artifacts/out/leakage/plants.json` | v4: I, S0, S1, S3b, R, S4, S5, S2 (each scored with and without sky) | `eccf15b9a6df0e30d4bc7ede91d10fe5b8b6a6ad` |
| `artifacts/out/leakage/plants_v3.json` | v3's plants (2026-10-01), kept: S5's first-written failure lives here | `8832345f8be73f5c9939825cdbebc244f116a4ed` |
| `artifacts/out/leakage/plants_v2.json` | v2's plants (2026-09-28), kept: S3's failure lives here | `33e5b0cd09713a3141702aa0ac9d0fee4721efbb` |
| `artifacts/out/leakage/calibrate.json` | S3's dose sweep (v2; not re-run) | `a153b9f7261f44385309ece23b75261b56f16631` |
| `artifacts/out/leakage/audit_physics.csv` | the baseline's physics for the audit's 40,000 sampled objects | `c04df4ee52d7df0289f2f326705bda39526653cb` |
| `artifacts/out/leakage/audit_physics.json` | the pull's record: query, batch size, sample SHA-1s, output SHA-1, missingness | `0a2687f3a40c54eb49496af0bc1e7dd7bf01fe3e` |
| `artifacts/leakage_audit.py` | the script, as it stands after the v4 runs | `c309121f85740333736b4f10f610618df4788502` |

**The physics pull** (`leakage_audit.py pull`, 2026-10-01, public SkyServer DR17 `SqlSearch`, no
token, batches of 100 IDs): the query is `PHYS_SQL` in the script and in the record. 40,000 IDs
requested, 40,000 rows returned, 0 missing. Sample SHA-1s: pretrain_v2 `d0cad1ca…`, probe_v2
`d039d531…`.

The plants ran on successive revisions of the script. The revisions differ only in:
- where the MPS cache is released (after S0, S1, S3 and R);
- the S3b entry and the calibration's per-variable field;
- the conditions-label rule, applied to S2 and S5 by recomputation.

## Plant results (v4, run 2026-10-02, before the hash)

Everything below ran under `artifacts/_heavy.sh` (`out/leakage/chain_v4.sh`; detail log
`chain_v4_detail.log`). v4's baselines: no redshift proxy; `petroRad_r` in the audit's (the synthetic
plants keep one size). Each plant is scored with sky (gating) and without (reported) from the same
pixel predictions. v3's results are in `plants_v3.json`, v2's in `plants_v2.json`.

| plant | n | required (with sky) | result with sky | without sky | fires |
|---|---|---|---|---|---|
| I identity | 5,000 probe, ~20,000 pretrain | clean CLEAN; near-dups NEAR-DUPLICATES (7); shared IDs raise | as required | (no baseline) | yes |
| S0 clean | 2 × 20,000 | all CLEAN; corpus PHYSICS-EXPLAINED | 20 CLEAN, **B i_sy TRACE** (ridge Δ 0.0041, bound +0.0003); corpus CLEAN, PHYSICS-EXPLAINED (ridge Δ +0.0008, bound −0.0004) | all CLEAN | **no** / yes |
| S1 v1 re-injected | 2 × 10,000 | 20 shift LEAK; UNSTRUCTURED; corpus CLEAN | 20 LEAK; both UNSTRUCTURED; corpus CLEAN | the same | yes / yes |
| S3b threshold | 2 × 20,000 | at-threshold variables not CLEAN | 19 LEAK, 1 TRACE; corpus CLEAN | the same states | yes / yes |
| S4 camcol 3 only | 2 × 10,000 | LEAK/TRACE; CAMCOL-STRUCTURED (3) only | 20 LEAK; both CAMCOL-STRUCTURED at camcol 3 only | the same | yes / yes |
| S5 sky pedestal | 2 × 10,000 | corpus not blocking; shifts CLEAN | corpus TRACE (physics AUC 0.99999), CONDITIONS-EXPLAINED; **B gr_v1y TRACE** (ridge Δ 0.0087, bound +0.0012) | corpus LEAK (Δ 0.133), CONDITIONS-EXPLAINED; shifts CLEAN | **no** / yes |
| S2 B bilinear | 2 × 10,000 | corpus LEAK, NOT CONDITIONS-EXPLAINED; A CLEAN | corpus LEAK (Δ 0.135, bound 0.128), NOT CONDITIONS-EXPLAINED; A all CLEAN | the same | yes / yes |
| R real v1 probe | 4,000 (256²) | t_g LEAK; t_r, t_i CLEAN; phys_mix CLEAN, PHYSICS-EXPLAINED | t_g x, y LEAK (ridge Δ 0.40, 0.37); t_r, t_i CLEAN; phys_mix CLEAN, PHYSICS-EXPLAINED | (no sky in v1 metadata) | yes |

**Two plants fail as written under the gating baseline (open for the user, below).** In each, one
shift variable with nothing planted reads TRACE: significant, below ε, by the ridge only. Neither
reads TRACE without sky, and neither did in v3.
- **Mechanism.** On a target with no signal the trees overfit: the physics-only out-of-fold R² of the
  shift variables runs −0.016 to −0.039 across the plants. The excess compares trees on physics ⊕
  the pixel prediction against trees on physics alone, so a difference in how the two overfit leaks
  into it. Sky, a ninth input carrying nothing about the shift, makes the baseline overfit slightly
  more on some variables (B gr_v1y: −0.039 with sky, −0.032 without), and the excess grows by about
  that much (0.0087 against 0.0048). The two TRACEs measure that, not pixel information.
- **What it costs.** TRACE does not block, so no state that gates changes. But a null shift variable
  reads TRACE in 2 of the 51 the plants require CLEAN (S0's 21, S5's 20, S2's A-side 10), so a TRACE
  on the real audit would mean little.

**Findings that bear on the audit's design (open below):**
- **The audit is very sensitive.** At α = 0.01 (1% of v1's offsets, about 0.003 px rms) 19 of 20
  shift variables read LEAK. At α = 0.0019 nothing does.
- **The CNN adds nothing beyond the ridge except on gross leaks.**
  - It learned S1 (pixel R² 0.70–0.93).
  - In S2, S3b and S4 its excess never exceeded +0.015. On real 256² stamps (R) its pixel R² was
    ≈ 0 for t_g, where the ridge reached 0.37.
  - At this budget it mainly doubles the family (and halves α per test).
- **The camcol test false-fired on real data.** R injects t_g uniformly, yet read CAMCOL-STRUCTURED
  at camcol 2 (bound +0.0019).
  - The test compares excess R² across camcols, and detectability differs by camcol (PSF, noise).
  - The synthetic plants, which have no per-camcol data quality, cannot show this.
- **The baseline's out-of-fold R² on a pure-noise target is slightly negative** (−0.01 to −0.07):
  the trees overfit. It cancels in the excess (S0), but inflates the gross plants' Δ above 1.

## Settled (user, 2026-09-28)

1. **TRACE is reported, not blocking** (was open item 2). The reason and the departure from the plan's
   wording are stated under "States", item 5.
2. **Sky is in the physics baseline** (was open item 3), stated on principle as an observing
   condition like PSF width. The audit's result is reported with and without sky; with sky decides.
3. **The physics pull is made now** (was open item 4): colour, PSF width and the redshift proxy (with
   R50, SNR and sky from the same query) for the 40,000 sampled objects, from public SkyServer, with
   the query and the output's SHA-1 recorded.

## Settled (user, 2026-10-01)

1. **The redshift proxy is dropped** (was open item 0 in v3), with the reasons under "The physics
   baseline and the excess".
2. **Magnitude and angular size are confirmed in the baseline**; `petroRad_r` is added beside
   `petroR50_r`.
3. **S5's requirement revised to "not blocking" is approved**, with v3's failed version kept on
   record (`plants_v3.json`, `S5.first_written_v3`).

## Open for the user before the hash

0. **v4's S0 and S5 fail as written: a null shift variable reads TRACE under the sky baseline**
   (mechanism above). Options:
   - (a) **Matched base** (recommended): measure the excess against trees on physics ⊕ a *permuted*
     copy of the pixel prediction, so both models have the same inputs and the same room to overfit;
     only the pixel prediction's alignment with the galaxy differs. It removes the mechanism rather
     than the symptom. Every plant reruns (about 2 h 10 min).
   - (b) Regularise the trees (larger leaves or fewer iterations) until a null target's out-of-fold
     R² sits at about 0. It shrinks the mechanism without removing it, and changes the baseline's
     power on the real corpus-membership target. Every plant reruns.
   - (c) Revise S0 and S5 to "no blocking state", consistent with TRACE being non-blocking, and
     report the null TRACE rate (2 of 51). Cheapest, but it leaves TRACE uninformative.

1. **The identity criterion is in, as a separate criterion that does not gate.** Should
   NEAR-DUPLICATES gate, or exclude those targets from pretrain_v2? That would cost the aligned
   comparison its "M's corpus, aligned" symmetry.
2. **The camcol fallback's trigger** is read as "a shift-variable leak that is camcol-structured".
   A literal LEAK on camcol is impossible under A7's reported-only camcol, and would fire on CCD
   gain alone.
   - The test as built false-fired on real stamps (plant R).
   - Proposed fix (not built): structured iff the leak is *present* in some camcol (bound > 0) and
     *absent* in another (CLEAN within it). That compares presence, not R², so uneven data quality
     does not read as structure. It needs re-running R, S1 and S4.
3. **The v1 offsets are added as stated variables**, beyond the plan's per-band s and frac(origin).
   That takes the family from 26 to 42 tests.
4. **The CNN.** At 8 epochs it is at chance on real 256² stamps. The options:
   - (a) keep it as the plan asks, accepting that it mostly costs multiplicity;
   - (b) give it a real budget (more epochs, or a central 128² crop to afford them), and re-run R to
     show it learns t_g;
   - (c) drop it and halve the family.
5. **The audit's data path (`audit()`) has not been run.** It cannot run before the hash, and it
   cannot read v2 before the hash. Its loaders are:
   - the completeness gate;
   - the physics pull (v3: run once, before the hash, and pinned; `audit` checks the pin);
   - the cut_log targets, with the s_b ≡ frac(origin) assertion;
   - the re-injected copy.

   Only `run()`, which they feed, is exercised by the plants. The first real run is also the loaders'
   first run, apart from the sample and the pull.
