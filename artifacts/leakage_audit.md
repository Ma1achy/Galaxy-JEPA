# Layer 2 leakage audit — pre-registration (v7, hashed)

**Question.** Can the pixels of the re-pulled corpora (pretrain_v2, probe_v2) say how a stamp was
processed, beyond what the galaxy's own physics already says?

**Status.** **v7, 2026-10-06: a declared post-hoc revision**, made after the hashed v6 audit read
**INVALID** (user, 2026-10-05). The in-situ plant now scores each shift variable against the shift
it actually injected; the injection is unchanged. Re-hashed: the new SHA-1 is in `PREREG_SHA1`.
Revision 6 gives the diagnosis and the lesson; "Audit v6 (run 2026-10-03 → 05): INVALID" gives v6's
reading. Re-scored by `rescore-check` (probe_v2's check block only), behind an acceptance test.
v6's record is kept, read-only. *Earlier:* **Hashed 2026-10-03** (v6): the Pre-registration section's SHA-1 is `98d67f7258fe0d7cf936acf9f2d2bb4b4a0d529c` (in `PREREG_SHA1`; `audit` refuses on any change). Approved by the user 2026-10-03 (final approval). Draft v6, 2026-10-03. v6 applies the user's 3 October decision
(option (a)): the **fit bound gates**, and S3b's requirement is restated against the audit's measured
resolution (declared as revised after v5b's result; v5b's scoring is kept). Every plant was re-scored
from the stored v5b numbers, without a re-run: **all fire**, with and without sky (results at the
end). Draft v5b, 2026-10-03: every plant rerun (results at the end). The bound now carries
fit-to-fit noise (user, 2026-10-02): three bounds are scored from one run, and **no plant's blocking
state differs between them**. Under the gating bound S0 still fails as written (two null TRACEs);
under the fit-noise bound S0 fires and S3b fails (one at-threshold variable CLEAN). **Open for the
user before the hash:** which bound gates, and what S0/S3b require (item 0 below). Draft v5,
2026-10-03: v5 applies the user's 2026-10-01 evening decision (option (a)): the
excess is measured against a **matched base**, trees on physics ⊕ the same pixel predictor retrained
on shuffled labels (5 shuffles), so a null variable's TRACE from the trees' extra flexibility (v4's S0
and S5 failures) cancels. Every plant is rerun under both baselines (results at the end); v4's output
is kept as `plants_v4.json`. Draft v4, 2026-10-02: v3 applied the user's 2026-09-28 decisions: TRACE is reported, not
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
  - Measured on this machine: 46 ms per batch at 64², 198 ms at 256². One CNN pass is about 17
    minutes per corpus and 35 for the pooled pair. With the matched base (v5) each is fitted 6 times
    (once on the labels, 5 times on shuffles), and the audit runs twice (the audit, then the
    re-injected plant): about 14 h of CNN in all, plus the trees.
  - **Declared (user, 2026-10-03): the CNN is at chance on real 256² stamps** (plant R: pixel R² ≈ 0
    for t_g, where the ridge reached 0.37). It is kept as the plan asks, but **the audit's
    sensitivity rests on the ridge predictor**; the CNN mainly adds tests to the family.
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
- **Excess (v5: against a matched base).** Per variable and pixel predictor: the out-of-fold
  predictions of trees on **physics ⊕ the pixel predictor's out-of-fold prediction**, against the
  **matched base**: ΔR² (ΔAUC for corpus membership).
  - **The matched base.** The same pixel predictor (ridge or CNN), on the same folds, is retrained on
    the target rows **shuffled jointly** across galaxies (for corpus membership, the corpus labels
    shuffled), **5 times** (shuffle seeds 500–504). Its out-of-fold prediction is then a column with
    the predictor's own form and noise but no information about the target. Trees on physics ⊕ that
    column are fitted per shuffle, and the base is their metric **averaged over the 5 shuffles**.
    Both sides of the comparison then have the same inputs, the same predictor and the same room to
    overfit; only whether the pixel predictor learned from the true labels differs. The excess is the
    pixels' advantage beyond what the model's flexibility alone gives.
  - **Why (v4).** Against trees on physics alone, a null shift variable read TRACE in 2 of the 51
    the plants require CLEAN (S0 B i_sy, S5 B gr_v1y; v4 results below): on a target with no signal
    the trees overfit, and one more input column changes how much. The matched base holds that
    fixed.
  - **Reported, no state:** the v4 comparison, trees on physics ⊕ pixels against physics alone
    (`excess_vs_physics`, its bound and the state it would give), beside each matched excess.
  - **Why the 5 shuffles average the metric.** Each shuffle is one draw of a null column; averaging
    the 5 metrics lowers the base's own noise about √5-fold. The bootstrap draws are shared across
    all of them (paired by galaxy), so the bound covers the base's galaxy sampling; the 5 shuffles'
    fit-to-fit spread is reported (`matched_base_per_shuffle`), as the declared limitation below.
  - The literal "pixel R² minus physics R²" would go negative whenever physics predicts the
    variable better than pixels, and would hide a leak; the second stage avoids that.
  - **Conditions** (the CONDITIONS-EXPLAINED label) use the same construction: trees on physics ⊕
    conditions ⊕ pixels against physics ⊕ conditions ⊕ the shuffled-label column.
  - **The camcol test** uses the matched base too: its per-camcol R² takes the base's squared error
    averaged over the 5 shuffles, which is exactly the averaged metric.
- **Bootstrap.** 20,000 Poisson galaxy-bootstrap draws of the out-of-fold predictions, shared
  across the models being compared (paired by galaxy).
  - The bound: the one-sided lower bound of Δ at α = 0.05 / 42 (Bonferroni over the family).
  - The plan's "CI excluding 0" is this bound above 0; a pixel predictor cannot leak by being worse
    than physics.
  - **Resolution** (`assert_resolution`): 20,000 × 0.05/42 = 23.8 draws lie beyond the bound (the
    floor is 10). With fewer, every variable would read CLEAN by construction, so the audit refuses
    to run.
  - **Fit-to-fit noise in the bound (v5b; user, 2026-10-02).** The bootstrap resamples galaxies
    around fixed fits. In v5's S0 the null excesses spread with sd 0.0018 against a bootstrap sd of
    about 0.00115, so the bootstrap alone is too narrow. One fit's metric noise σ is measured by the
    spread of the 5 shuffled-label bases (sd per variable, pooled as an RMS over the block's stated
    variables per predictor; a one-variable block uses its own, 4 df). At full n σ is 0.0012–0.0027
    for the shift variables (S0: ridge 0.00119, CNN 0.0021). It enters in quadrature with the
    bootstrap margin, lo = Δ − √((Δ − lo_boot)² + (z_q σ f)²), z_q = Φ⁻¹(1 − α/42), under three
    readings scored from the same run:
    - **bootstrap**, f = 0: galaxies only (v5 as first written);
    - **shuffle**, f = 1/√5: the matched base's own noise. The 5-shuffle mean's sd is about 0.00054,
      half the bootstrap sd, so not negligible: included (user, 2026-10-02); it gated v5b;
    - **fit**, f = √(1 + 1/5): **gating from v6 (user, 2026-10-03).** Why: the trained pixel
      predictor is one fit, with the same noise as each shuffled-label base, so the excess carries
      that fit's noise plus the 5-base mean's. With it included, the predicted null sd,
      √(0.00115² + (0.0012 × 1.1)²) ≈ 0.0017, matches the observed 0.0018; without it the bound is
      too narrow and null variables read TRACE (v5b's S0).
    - All three readings are stored for every scored variable (`lo_by_bound`, `state_by_bound`); the
      state, labels and verdict are the fit reading's.
  - **Resolution, stated:** with fit noise counted the audit separates a pixel excess from noise at
    about **0.008** in ΔR², a little under ε. Below it TRACE versus CLEAN is not reliably readable
    either way, which is one reason TRACE does not block (States, item 5).
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
- **Reported only, not to be interpreted** (user, 2026-10-03): the test false-fired on real stamps in
  plant R (CAMCOL-STRUCTURED at camcol 2 on a uniform injection), because detectability differs by
  camcol. Its label is recorded with the audit and carries no conclusion; the routing below is the
  plan's record of what the label was meant to select, and on any blocking state the audit stops and
  reports, with the remedy chosen by the user, not by this label.
- **Routing (as planned; not acted on, above).**
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
- **Consequence: NEAR-DUPLICATES is reported and does not gate** Layer 2 or A8 (settled, user,
  2026-10-03). Why: the encoders never see labels, and M and the aligned encoders pretrain on the
  same target list (pretrain_v2's is M's v1 list), so any overlap affects both arms equally.
  Excluding them would add a corpus difference to a comparison built to differ from M only by the
  alignment.
  - **If the near-duplicate rate exceeds 1% of probe_v2, it is reported prominently** (a top-level
    flag in the audit's output and a banner line in its log), still without a state.

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

> **Revised in v7 (2026-10-05/06), post hoc: after the hashed v6 audit read INVALID.** v6's text
> is kept beside each change, marked superseded. Why, and what v6 read: revision 6
> under "Revised before the hash after a plant failed", and "Audit v6 (run 2026-10-03 → 05):
> INVALID" after the pins.

- A copy of each corpus's sample, with v1's recorded per-band offsets re-injected: each band moves
  by (v1_rel − 127.5) px (`reinject`: reflect-pad to odd P, the cutter's Fourier shift, crop).
  **Unchanged in v7:** the injection is M's real v1 misregistration.
- It goes through the identical path, with the same folds, seeds and family.
- **v7: what each shift variable is scored against.** On the re-injected copy, each s_b (per band,
  per axis) is scored against **the shift actually injected, −(v1_rel_b − 127.5)**, not the cut
  log's s_b (`check_targets`). gr_v1 and ir_v1 are unchanged: they are already the injected offsets'
  differences. The two definitions agree to 1.4 × 10⁻¹⁴ px on pretrain_v2, whose v1 position was
  taken at v2's own coordinates. They differ by up to 1 px on probe_v2, whose v1 was centred on
  GZ2's 4-decimal ra/dec.
  - *v6, superseded:* scored against the cut log's s_b, which assumed injection ≡ −s_b.
- **Must read LEAK on all 20 shift variables.** Otherwise the audit is **INVALID**.
- **v7: the re-score path** (`leakage_audit.py rescore-check`). Only probe_v2's check block is
  re-run, with the v7 targets. Everything else is read from v6's record (`audit_v6.json`, SHA-1
  pinned; it refuses any other):
  - the main run, which does not touch the injection;
  - pretrain_v2's check block, where v7's targets equal v6's to 1.4 × 10⁻¹⁴;
  - the pooled corpus block, which has no shift target.

  Before any fit, the sample's SHA-1 must equal v6's, and 50 stored re-injected stamps (fixed
  seed) must be byte-identical to a fresh re-injection from the FITS. The CNN's scale is the mean
  border σ over both corpora's stored re-injected copies, as in `run`.
- **v7: the acceptance test, before the re-score's result counts.** In the re-scored probe_v2
  check block, gr_v1x, gr_v1y, ir_v1x and ir_v1y (which v7 does not touch) are compared with v6's
  stored values.
  - **Ridge side: must be bit-identical.** This covers physics alone, and the ridge's pixel,
    pixel_lo, both, matched base (per shuffle and mean), excess, lo_boot, sd_shuffle and the
    excess over physics with its bound. The ridge is fitted per target, the trees are seeded and
    single-threaded, and the bootstrap seed is the variable's own. A difference up to 10⁻⁹ is
    accepted and declared as floating-point noise; anything larger fails.
  - The ridge's fit bound and state are **not** compared. They use σ pooled over the block's ten
    variables, and v7 changes six of those.
  - **CNN side: not reproducible by design.** One multi-output network is fitted per fold over the
    block's 20 targets. Revising six of them changes the shared fit, so gr_v1's and ir_v1's CNN
    numbers change in any v7 run, the full run included. Their changes in pixel R² and excess are
    reported against 3·√2·σ_fit (v6's pooled CNN σ for the block: 0.0076), but do not gate. **Their
    CNN states must stay LEAK.**
  - **If the test fails, no state is written**: the result does not count, and the fallback is the
    full audit (`audit`), which applies `check_targets` in its check run.
- **v7: the state.** If the test passes and every check variable reads LEAK, the audit's state is
  the stored main run's; otherwise **INVALID**. The output is `audit_v7.json`; v6's record is
  kept, read-only.

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
| S3b threshold (revised; restated v6) | α = 0.01 re-injection, fresh seed | **v6:** every shift variable with Δ ≥ ε LEAK or TRACE (never CLEAN); at least one variable at the threshold (Δ in [ε/2, 2ε]), at least one of them with Δ ≥ ε; those with ε/2 ≤ Δ < ε reported only. (v2–v5b: no shift variable CLEAN; every variable in [ε/2, 2ε] LEAK or TRACE.) |
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

3. **v3's S5 failed as first written** (corpus CLEAN required under the sky baseline; it read
   TRACE).
   - With sky in the baseline, physics alone separates the corpora at AUC 0.999987. The pixels add
     1.35 × 10⁻⁵ AUC (ridge and CNN alike), with a bound of 2.7 × 10⁻⁶: significant, about 740× below
     ε. This is v2's S5 residue again, now on the corpus state itself rather than the conditions
     label.
   - The requirement is now "corpus not blocking (CLEAN or TRACE); shift variables CLEAN", which is
     what the plant exists to show: a corpus difference in an observing condition the baseline
     holds does not block. The first-written result is kept in `plants_v3.json` (`S5.first_written_v3`)
     and S5 was re-scored from the stored numbers, not re-run.
4. **v5's S0 failed as first written, under the bootstrap and then the shuffle bound** (null shift
   variables read TRACE: v5 B i_sy, gr_v1y, ir_v1y; v5b, shuffle bound, B i_sy and gr_v1y, Δ 0.0039–0.0040,
   bounds +0.0001/+0.0002). The bootstrap omitted fit-to-fit noise (above). Fixed by the fit bound
   (user, 2026-10-03), not by a change to S0's requirement. v5's run is `plants_v5a.json`; v5b's
   scoring is `plants_v5b.json` and each plant's `scored_v5b` in `plants.json`.
5. **v5b's S3b fails under the fit bound as written** (A g_sy, 1% of v1's offsets, Δ 0.0078, bound
   −0.0002: CLEAN, inside [ε/2, 2ε]). **S3b is restated after seeing this result** (user, 2026-10-03):
   only variables with Δ ≥ ε must not read CLEAN; those in [ε/2, ε) are reported. Reason: the audit's
   resolution with fit noise counted is about 0.008, so a requirement below it would demand what the
   measurement cannot deliver. A clause is added so the restated test is not vacuous: at least one
   threshold variable must have Δ ≥ ε (here A g_sx 0.0122 and B g_sx 0.0173, both LEAK). **This is a
   post hoc revision**, declared as such; v5b's result stays on record (`plants_v5b.json`;
   `S3b.scored_v5b` in `plants.json`).
6. **v7: the hashed v6 audit read INVALID, and the in-situ plant's targets are revised after that
   result** (user, 2026-10-05). **This is a post hoc revision, declared as such.**
   - What v6 read: probe_v2's g_sy and i_sy were UNRESOLVED in the check run; the other 18 LEAK.
   - The diagnosis was data only (`leakage_diag_invalid.py`, output `diag_invalid.json`; both
     pinned).
     - **The pixels carry the intended shift exactly** in both corpora: measured against intended,
       slope 0.994–0.998, r = 1.000, residual 0.001–0.003 px RMS.
     - **The rows align by objID**, and the cut logs predate the run.
     - **The targets did not match the injection.** The plant injects v1's offsets but scored s_b,
       assuming the two are equal. On pretrain_v2 they are (r = −1.000). On probe_v2, v1 was
       centred on GZ2's 4-decimal ra/dec (`repull_targets.py`), so the injected offset correlates
       with s_b at r ≈ −0.19 (y) and −0.64 (x).
     - The readability this predicts (r² × the ridge's 0.9 on the injected offsets: 0.035 for y,
       0.37 for x) matches what was observed (0.030–0.036 and 0.33–0.36).
   - **The lesson.** Every planted positive before the hash was synthetic, with injection ≡ target
     by construction (S1: "v1's offsets (−s_b)"), so none of them could catch a mismatch between
     the shift injected and the target scored. A plant must also be checked on the real
     corpora's own definitions: that the target it scores is the quantity it injects.
   - v6's record (`audit_v6.json`, state INVALID) is kept on file, pinned and read-only.

## Plant evidence (pinned)

SHA-1 of each file as it stands after the v6 re-score (2026-10-03, from v5b's stored numbers).
Re-pin if anything changes before the hash.

| file | role | SHA-1 |
|---|---|---|
| `artifacts/out/leakage/plants.json` | v6: v5b's run re-scored under the fit bound and the restated S3b (`rescore`); each plant keeps `scored_v5b` | `b324e292ab3ef266c5e35d98ea66980682a1779d` |
| `artifacts/out/leakage/plants_v5b.json` | v5b as scored (shuffle bound gating): I, S0, S1, S3b, R, S4, S5, S2; with and without sky; three bounds | `019084dc982efecae6ae7306baaf09a1a259aa9b` |
| `artifacts/out/leakage/plants_v5a.json` | v5 as first written (bootstrap bound): I, S0 only, cut short by a macOS access loss | `2c870837971b36ca8fc6d45ec580ded117895716` |
| `artifacts/out/leakage/plants_v4.json` | v4's plants (2026-10-02), kept: the null TRACEs against physics alone | `eccf15b9a6df0e30d4bc7ede91d10fe5b8b6a6ad` |
| `artifacts/out/leakage/plants_v3.json` | v3's plants (2026-10-01), kept: S5's first-written failure lives here | `8832345f8be73f5c9939825cdbebc244f116a4ed` |
| `artifacts/out/leakage/plants_v2.json` | v2's plants (2026-09-28), kept: S3's failure lives here | `33e5b0cd09713a3141702aa0ac9d0fee4721efbb` |
| `artifacts/out/leakage/calibrate.json` | S3's dose sweep (v2; not re-run) | `a153b9f7261f44385309ece23b75261b56f16631` |
| `artifacts/out/leakage/audit_physics.csv` | the baseline's physics for the audit's 40,000 sampled objects | `c04df4ee52d7df0289f2f326705bda39526653cb` |
| `artifacts/out/leakage/audit_physics.json` | the pull's record: query, batch size, sample SHA-1s, output SHA-1, missingness | `0a2687f3a40c54eb49496af0bc1e7dd7bf01fe3e` |
| `artifacts/leakage_audit.py` | the script: v6 (BOUND = "fit", S3b restated, `rescore`) plus the near-duplicate flag, `PREREG_SHA1` filled (the script `audit` runs) | `e1c1a16c01c23d2bc41f646b9091b5a27b85b755` |
| `artifacts/leakage_audit.py` before `PREREG_SHA1` | v6 plus the near-duplicate flag, as committed (`c5d569d`) | `962901aeae62f3b40e85e3958dd64d05c7de4123` |
| `artifacts/leakage_audit.py` at the v6 re-score | before the near-duplicate flag | `b0958b69d456851ede6e32fd4f37f212ef222d19` |
| `artifacts/out/leakage/audit_v6.json` | **v6's audit record (INVALID)**: a read-only copy of `audit.json` as v6 wrote it; `rescore-check` refuses any other | `e469c1d7165c4f608217bd6342bc0f2be4808f04` |
| `artifacts/leakage_diag_invalid.py` | v6's INVALID diagnosed, data only (run 2026-10-06 from the scratchpad; filed with its output path set) | `46281d54e71d381221f21dbb84ed109c1dda9778` |
| `artifacts/out/leakage/diag_invalid.json` | its output | `5135207f6910ac3e296e5a48e493ee1b6cba73a7` |
| `artifacts/leakage_datacheck.py` | the loaders' data check | `4a4bfad671863005c144ac56623554548cd115bc` |
| `artifacts/out/leakage/datacheck.json` | its output | `393ab8f52ced3ee0bc6abec8cdc700a487da0df3` |
| `artifacts/leakage_audit.py` at v5b | the script the v5b plants ran (commit `8700b41`) | `26179731c58d939ebdb04eb78a381e0d34934e0b` |

**The physics pull** (`leakage_audit.py pull`, 2026-10-01, public SkyServer DR17 `SqlSearch`, no
token, batches of 100 IDs): the query is `PHYS_SQL` in the script and in the record. 40,000 IDs
requested, 40,000 rows returned, 0 missing. Sample SHA-1s: pretrain_v2 `d0cad1ca…`, probe_v2
`d039d531…`.

The plants ran on successive revisions of the script. The revisions differ only in:
- where the MPS cache is released (after S0, S1, S3 and R);
- the S3b entry and the calibration's per-variable field;
- the conditions-label rule, applied to S2 and S5 by recomputation.

## Plant results (v6, re-scored 2026-10-03 from v5b's stored numbers, before the hash)

`leakage_audit.py rescore`: v5b's stored excesses, bootstrap bounds and shuffle spreads read under the
fit bound and the restated S3b; nothing refitted.

| plant | required (v6) | with sky (gating) | without sky | v5b as scored (shuffle bound) |
|---|---|---|---|---|
| I identity | as v4 | fires | — | fires |
| S0 clean | every variable CLEAN; corpus PHYSICS-EXPLAINED | **fires**: all CLEAN; corpus CLEAN, PHYSICS-EXPLAINED | fires | did not fire (2 null TRACEs) |
| S1 v1 re-injected | 20 LEAK; UNSTRUCTURED; corpus CLEAN | **fires** | fires | fired |
| S3b threshold | restated (above) | **fires**: A g_sx (0.0122), B g_sx (0.0173) LEAK; A g_sy (0.0078) CLEAN, reported | fires | fired as then written; would not under the fit bound |
| R real v1 probe | t_g LEAK; t_r, t_i, phys_mix CLEAN, PHYSICS-EXPLAINED | **fires** | — | fired |
| S4 camcol 3 | LEAK/TRACE; CAMCOL-STRUCTURED at camcol 3 only | **fires** | fires | fired |
| S5 sky pedestal | corpus not blocking; shifts CLEAN | **fires**: corpus TRACE, CONDITIONS-EXPLAINED | fires | fired |
| S2 B bilinear | corpus LEAK, NOT CONDITIONS-EXPLAINED; A CLEAN | **fires** | fires | fired |

- **Declared (camcol under re-scoring).** The camcol test is not recomputed: it needs the predictions,
  which are not stored. In S0 and S3b its tested set was v5b's (S0 B: i_sy, gr_v1y; S3b: all 20 shift
  variables); under the fit bound S0 would test none and S3b would drop A g_sy. Both stored results
  read UNSTRUCTURED, and neither plant's requirement involves camcol. On the audit itself the camcol
  test runs live under the gating bound.

## Plant results (v5b, run 2026-10-03, before the hash)

`out/leakage/chain_v5b.sh` under `_heavy.sh`, 14:16–18:13 (detail log `chain_v5b_detail.log`). Matched
base from 5 shuffled-label pixel predictors; each plant scored with sky (gating) and without
(reported), and under all three bounds. "fires" is the plant's requirement as written.

| plant | n | bootstrap | **shuffle (gating)** | fit (proposed) | without sky (shuffle) |
|---|---|---|---|---|---|
| I identity | 5,000 / ~20,000 | yes | **yes** | yes | — |
| S0 clean | 2 × 20,000 | no: TRACE B i_sy, gr_v1y, ir_v1y | **no: TRACE B i_sy, gr_v1y** | yes: all CLEAN | yes |
| S1 v1 re-injected | 2 × 10,000 | yes | **yes**: 20 LEAK, UNSTRUCTURED, corpus CLEAN | yes | yes |
| S3b threshold | 2 × 20,000 | yes | **yes** | no: A g_sy (Δ 0.0078) CLEAN | yes |
| R real v1 probe | 4,000 | yes | **yes**: t_g LEAK; t_r, t_i, phys_mix CLEAN | yes | — |
| S4 camcol 3 | 2 × 10,000 | yes | **yes**: CAMCOL-STRUCTURED at camcol 3 only | yes | yes |
| S5 sky pedestal | 2 × 10,000 | no: TRACE B i_sx | **yes**: corpus TRACE, CONDITIONS-EXPLAINED | yes | yes |
| S2 B bilinear | 2 × 10,000 | yes | **yes**: corpus LEAK (Δ 0.127, bound 0.111), NOT CONDITIONS-EXPLAINED; A CLEAN | yes | yes |

- **No blocking state differs between the bounds, in any plant.** Every LEAK plant reads LEAK under
  all three; no null variable reads LEAK or UNRESOLVED under any. The bounds disagree only on TRACE
  versus CLEAN, which does not gate.
- **The null side (S0).** The two null TRACEs under the gating bound have Δ 0.0039–0.0040 and bounds
  +0.0001 / +0.0002. The fit bound clears them (bounds −0.0013).
- **The planted side (S3b).** A g_sy carries 1% of v1's offsets and reads Δ 0.0078: TRACE under the
  gating bound (lo +0.0018), CLEAN under the fit bound (lo −0.0002). The other at-threshold variables
  (A g_sx Δ 0.0122, B g_sx 0.0173) read LEAK under all three.
- **What the two failures say together.** With fit noise counted, null excesses reach about 0.004 and
  a planted 0.0078 sits at the edge: the audit's honest resolution is about 0.008 in ΔR², a little
  under ε (0.01). Below it, TRACE versus CLEAN cannot be read reliably either way, which is why TRACE
  was made non-blocking.
- **R's camcol test** again reads CAMCOL-STRUCTURED at camcol 2 on a uniform injection (open item 2).
- Per shuffle, the matched base's spread and the pooled σ are in `plants.json`
  (`matched_base_per_shuffle`, `sigma_fit_pooled`, `lo_by_bound`, `state_by_bound`).

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

## Settled (user, 2026-10-01 evening)

1. **Option (a), the matched base** (was open item 0 in v4: S0 and S5 failed as written, a null shift
   variable reading TRACE under the sky baseline). Built as the user specified: the comparison is a
   pixel predictor trained on **shuffled labels**, 5 shuffles, not the v4 draft's permuted copy of
   the trained prediction. A permuted copy keeps the trained predictor's distribution but not its
   fitting noise; retraining on shuffled labels keeps both. Defined under "The physics baseline and
   the excess". v4's failures stay on record (`plants_v4.json`); options (b) regularised trees and
   (c) relaxed requirements are not taken. Every plant is rerun under both baselines.

## Settled (user, 2026-10-02)

1. **The matched base's shuffle-to-shuffle variation is reported at full n and, being not negligible
   (sd of the 5-shuffle mean ≈ 0.00054 against a bootstrap sd ≈ 0.00115), is included in the bound**
   (the "shuffle" reading above, gating).

## Settled (user, 2026-10-03)

1. **Option (a)** (was open item 0 in v5b): **the fit bound gates**, for the reason under "Bootstrap";
   **S3b is restated** (variables with Δ ≥ ε never CLEAN; ε/2 ≤ Δ < ε reported only), declared as
   revised after v5b's result, with v5b's result kept. Every plant re-scored from v5b's stored numbers;
   no re-run.

## Settled (user, 2026-10-03, final approval)

1. **Approved:** the fit bound gating; S3b as restated, with the added clause; open items 1–4 as
   written, with these additions (made above): the reason near-duplicates do not gate, and a
   prominent report above 1% (Identity); the camcol result reported only and not interpreted
   (Camcol routing); the CNN declared at chance on real stamps, the audit's sensitivity resting on
   the ridge (Predictors). The v1 offsets stay as stated variables (family 42).
2. **Condition on item 5:** before the hash, the audit's data loaders are run on the real corpora as
   a data check only (no audit statistic, no predictor fitted, no state computed): 2,000 galaxies of
   each corpus. Result below.

## The loaders' data check (before the hash)

Run 2026-10-03 (22:49, 204 s), `artifacts/leakage_datacheck.py`. It goes through `audit()`'s own
loaders and computes no audit statistic: no predictor is fitted and no state is computed. It covers
the first 2,000 galaxies of each corpus's audit sample, in the audit's own order. Output:
`artifacts/out/leakage/datacheck.json`. **Every check passes, and no loader needed fixing.**

| check | pretrain_v2 | probe_v2 |
|---|---|---|
| completeness gate | 826,968 / 826,968 | 230,349 / 230,349 |
| cut log: ra, dec finite; object_id unique | yes (48 columns) | yes (50 columns) |
| sample: n, SHA-1 against the pull record | 20,000, `d0cad1ca…` matches | 20,000, `d039d531…` matches |
| physics file SHA-1 against its record | `c04df4ee…` matches | (same file) |
| join: rows = sample; physics rows found; metadata rows found | yes; 100%; 100% | yes; 100%; 100% |
| missing any baseline input (full 20,000) | 0% | 0.03% (6 galaxies: petroR50_r, hence sb50_r); within the 2% gate; enters the trees as NaN |
| `_targets`: the s_b ≡ frac(origin) assertion | holds | holds |
| stated targets: finite rows (min over the 10) | 2,000 / 2,000 | 2,000 / 2,000 |
| v1 offsets (2000, 3, 2) float64 | finite, within ±0.5 | finite, within ±0.5 |
| folds (5) | 428 / 432 / 373 / 402 / 365 | 393 / 404 / 403 / 386 / 414 |
| stamps (2000, 3, 256, 256) float16 | all finite; −59.8 to 394; 0 at the fp16 limit | all finite; −50.4 to 473; 0 at the fp16 limit |
| re-injected copy (200, 3, 256, 256) float16 | all finite; −15.5 to 356.5 | all finite; −16.9 to 425.3 |

Family: 10 stated variables, 42 tests, `assert_resolution` passes. The temporary stamp memmaps were
deleted afterwards. A pandas mixed-type warning on `gz2_sep_flag` is harmless: the audit does not
read that column.

Added with the check (user, 2026-10-03, item 1): `identity()` records `near_dup_rate`. Above 1% of
probe_v2, `audit()` prints it prominently and writes `NEAR_DUPLICATES_OVER_1PCT` into the report. It
does not gate. The script was re-pinned after this change.

## Audit v6 (run 2026-10-03 → 05): INVALID

The audit was run as hashed (SHA-1 `98d67f72…`, script `e1c1a16c…`) from 2026-10-03 23:03 to
2026-10-05 06:54, in one uninterrupted process lasting 31 h 51 min: the main run, then the full
check run. Its record is `audit_v6.json`.

- **The main run read TRACE.**
  - All 20 per-corpus variables read CLEAN under the ridge and the CNN. The largest fit-bound lower
    limit was −0.0013, and no excess reached 0.004.
  - Corpus membership read TRACE: the CNN's excess was 0.0049 AUC, with lower limit +0.0026, below
    ε. The ridge read CLEAN.
- **The check run read 18 of 20 LEAK.** probe_v2's **g_sy** (ridge excess 0.035, lower limit
  −0.092) and **i_sy** (0.030, −0.096) read UNRESOLVED, so `plant_fired` was false: **INVALID**.
  The bounds were wide because the ridge's pooled σ was 0.038, dominated by the v1 targets' shuffle
  spread.
- **Near-duplicates:** 11.94% of probe_v2 (within 1″, distinct objIDs), roughly uniform across the
  splits (test 4,082 of 34,828; val 4,254 of 34,672; train 19,174 of 160,849). Reported
  prominently; they do not gate (item 1).
- The cause and the revision are in revision 6 and "The re-injected plant" (v7).

## Settled (user, 2026-10-05): revision v7

1. On the re-injected copy, each s_b is scored against the shift actually injected,
   −(v1_rel − 127.5). The injection and gr_v1/ir_v1 are unchanged. Declared post hoc, with the
   diagnosis and the lesson recorded. v6's record is kept, pinned.
2. `rescore-check` re-runs only probe_v2's check block against the stored run, behind an acceptance
   test on gr_v1/ir_v1. If it fails, stop and report, and fall back to the full run. **The
   acceptance test as written (above) departs from the literal instruction ("reproduce exactly, or
   within the documented fit-noise tolerance") on the CNN side.** The CNN's gr_v1/ir_v1 numbers
   cannot reproduce in any v7 run, because its fit is shared with the revised targets. They are
   therefore reported against 3·√2·σ_fit, and only their states (LEAK) gate. The ridge side must be
   bit-identical.
3. Commit, hash v7, update `aligned_comparison.md`'s precondition SHA-1 (a change of reference
   only), re-hash it.
4. Run the re-score with a low-memory waiter. Report the acceptance test, then the verdict.
5. CLEAN or TRACE: the user restarts Terminal, then A1 (with the watchdog) and A2, after confirming
   nothing heavy holds memory. LEAK, UNRESOLVED, INVALID or INSUFFICIENT: stop and report. The
   stamp memmaps may be deleted once the re-score has read its verdict.
