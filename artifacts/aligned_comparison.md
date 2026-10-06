# Aligned encoder against M — pre-registration (DRAFT, not hashed)

**Question.** With the band misregistration fixed, does the encoder spend the freed capacity on
morphology?

**Status.** **Re-hashed 2026-10-06**: the Pre-registration section's SHA-1 is `637ddddaa0e07e5907687d64441eb1b5c41a0b5d`. The only change from the 2026-10-03 hash (`3c7ae3b7…`) is the precondition's reference to the leakage audit, now v7 (`f2fe917d…`): a change of reference only. Draft v5, 2026-10-01 (v4, 2026-09-27, plus criterion 3's plant sweep, user 2026-09-28). The user's settlements are folded in; the plants run before the
hash (D28), and the user approves before hashing. Once settled, the section from "## Pre-registration"
through its last content line is hashed (SHA-1, trailing newline), **before any aligned-encoder
embedding is read**.

## Pre-registration

### Common ground

- **Encoders.**
  - **A1, A2**: aligned encoders on pretrain_v2, seeds 0 and 1. Config identical to M's except the
    corpus, its frozen normalisation and `out_dir` (`runs/m_v2`, `runs/m_v2_s1`).
  - **M1** = M (seed 0, v1, 101,308 steps).
  - **M2** = O2 (`runs/o2`): training seed 1, with M's `config_hash` (`v2:61330a00…`), splits and
    v1 data, stopped at 101,308 steps (Brief O2, `o_findings.md`). An earlier draft wrongly said
    M2 did not exist.
  - **Training length:** A1 and A2 run M's driver (`m2_long_run.py`, as O2 did) with training seed
    0 and 1, config and split seed 0, and a **fixed stop at 101,308 steps**. The schedule stays
    sized for 253,270 steps, so the LR and EMA schedules match M's and O2's exactly. The stopping
    rule does not pick their length; that keeps the 2×2 balanced.
  - **Corpus.** pretrain_v2's target list is identical to v1's pretrain corpus (826,968 object IDs;
    none on one side only). Training starts only after `artifacts/pretrain_v2_validate.py` confirms
    all 826,968 stamps landed, or lists any that did not.
    - **Validated (2026-10-01):** all 826,968 landed, 0 failed, the ID set identical to v1's.
    - **Declared: one blank stamp is kept.** Object 1237666310706430371 sits at a frame edge where
      the three bands' valid regions do not intersect (cut-log `valid_frac` 0), so the v2 cutter pads
      the whole stamp to zero, as designed. It stays in pretrain_v2 so that the corpus matches M's ID
      set exactly; one stamp in 826,968 (0.0001%).
  - **Normalisation.** The v2 freeze is fitted only on the complete pretrain_v2 corpus, by v1's
    procedure (`e5_corpus_moments.py`, `e5_fit_normalisation.py`), under v1's stability gate: 200
    disjoint half-splits, tolerance 1% relative, at most 5% of splits may breach (v1: median
    0.314%, worst 0.861%, 0% breaching). If the gate fails, nothing trains and the user decides.
    - **Fitted (2026-10-01): passes.** Median 0.320%, worst 0.941%, 0% breaching; content hash
      `246d8de6…` (`configs/pretrain_v2.yaml`). Run 1000 again supplies most of the trim (543 of
      827, 108× the corpus rate; v1: 67.4%, 111×).
    - **Reported, no state: v2's per-band std is about 1.6% above v1's, and the stamps carry it, not
      the trims.** On the 826,122 stamps neither fit trimmed (the trims share 808 of 827), v2/v1 std
      is 1.0171, 1.0166, 1.0154 (g, r, i), against 1.0172, 1.0168, 1.0155 between the freezes; the
      means agree to 0.15% (`artifacts/v1_v2_std_common.py` → `artifacts/out/v1_v2_std_common.json`).
      Each encoder is normalised by its own corpus's freeze, so the difference is absorbed, not
      carried into training.
- **Corpora.** Each encoder reads the corpus version it was trained on: M on probe (v1), the aligned
  encoders on probe_v2.
  - The object IDs are one-to-one, and the splits are identical (`assign_three_way` hashes objID
    and seed).
  - Reported, no state: M1 read on probe_v2. This separates a test-time effect of the cleaner
    pixels from a training effect.
- **Read-out.** M's convention throughout: block 11, mean-pooled.
  - The 37 probes are `_fit` logistic regressions with C from `probe.yaml`.
  - **Criterion 2** uses the headline protocol's training size, the full probe-train split
    (160,857 galaxies). **Criteria 1, 3 and 4** keep the DD cap (40,000, H5's deterministic
    stride).
  - All criteria score AUC on the whole probe-test split (34,829 galaxies), each answer on its
    eligible galaxies.
- **Powered answers.** An answer is powered if its smaller class in the test split has ≥ 100
  galaxies **and** the mean of M1's and M2's baseline AUCs is ≥ 0.6.
  - The list is fixed from M alone, blind to the aligned results, and is recorded before any
    aligned embedding is read.
  - Only powered answers carry per-answer claims.
- **Offsets.** v1's recorded per-band in-stamp offsets from probe_v2's `cut_log.csv`: g−r x, g−r y,
  i−r x, i−r y (the misregistration M saw). v2's applied shifts s_b are reported alongside, with no
  state.
- **Precondition.** The leakage audit (`artifacts/leakage_audit.md`, SHA-1 `f2fe917d7c65fd8f36b40118a5ab56a0ee64e543`, v7; it was `98d67f72…` (v6) until 2026-10-06, and the change is of reference only: the audit was revised post hoc after v6 read INVALID, and nothing in this document changed) reads **CLEAN or TRACE** on v2 (TRACE is reported, not blocking; user, 2026-09-28). If it
  reads **LEAK, UNRESOLVED, INVALID or INSUFFICIENT**, nothing trains and nothing below is scored
  (the audit's blocking states, `leakage_audit.md` "States").

### 1. Fix check — is the offset gone from the aligned representation?

- **1a, probes.** For each of the four offsets, a sign(offset) probe is fitted on each aligned
  encoder's pooled embedding. It is the same probe as Part 4's: standardised L2 logistic, C = 1,
  trained on probe-train galaxies with recorded offsets and scored by AUC on test galaxies. The CI is
  a 10,000-draw galaxy bootstrap.
  - **NEAR CHANCE:** |AUC − 0.5| ≤ 0.05 **and** the 95% CI lies inside [0.40, 0.60].
  - **READABLE:** the CI lies wholly outside [0.40, 0.60].
  - **UNRESOLVED:** otherwise.
- **1b, SAE.** Each aligned encoder gets its SAE by **the rule that chose M's**, applied unchanged
  to A1 and A2 (and to M2 = O2 for criterion 3's floor):
  - **Width:** TopK SAEs at 8× (3,072) and 16× (6,144) are trained at block 11 with Part 3's hashed
    settings (`interp_tooling.md`, Part 3, `e25e1e27`: k = 32, Adam, 8 epochs, the size-scaled step,
    AuxK) on the encoder's own sae tokens. Each is scored by S3 on sae_eval. **The chosen size is 8×
    unless 8× FAILs and 16× does not** (`dd_sae_score.chosen_size`). On M both FAIL, so 8×.
  - **Selection statistic:** per latent, galaxy-level activation = the mean over the galaxy's 256
    tokens; its |Spearman| with each of the four offsets on the sae_eval galaxies with recorded
    offsets; the latent's score is the largest of the four (`dd_sae_score.s1_stat`).
  - **The selected latent** (Part 4's single latent) is the first of S1's specific latents: the
    highest score among latents whose activation-matched untrained latent reaches |ρ| < 0.2 with the
    same offset. **Tie-break:** equal scores go to the lower latent index.
  - **Reproduction on M:** the rule gives 8× and latent 328 (ρ +0.912 with i−r x; next 1472 at
    −0.883) on the 3,116 galaxies it was first run on, and again 8× and latent 328 (ρ +0.910;
    next 2389 at −0.881) on the complete probe_v2 `cut_log.csv` (4,999 galaxies).

  The test: **no latent** of the chosen SAE has galaxy-level |ρ| ≥ 0.5 with any offset. For
  reference, M's best is 0.91, and the untrained encoder's best over all latents is 0.23.
- **Reported, no state: the confound floor.** The four sign(offset) probes (the identical 1a probe:
  standardised L2 logistic, C = 1, same splits, 10,000-draw galaxy bootstrap) on an **untrained
  encoder** (M's architecture, random initialisation, seed 0), read on probe_v2. This is the offset
  readability that comes from galaxy properties correlated with the recorded offsets rather than
  from the misregistration, and is the reference for interpreting a RESIDUAL state.
- **States, in precedence:**
  1. **INSUFFICIENT:** fewer than 500 test galaxies with recorded offsets.
  2. **NOT FIXED:** any offset READABLE in either seed. The fix did not remove the offset, and the
     question cannot be answered; 2–4 are reported as exploratory.
  3. **FIXED:** all four NEAR CHANCE and no latent ≥ 0.5, in both seeds. The misregistration signal
     is gone.
  4. **RESIDUAL:** anything else, for example the probes near chance but a latent ≥ 0.5, or an
     UNRESOLVED probe. The signal is mostly gone but not cleanly; 2–4 are scored with this caveat.
- **Plants (D28; `artifacts/aligned_c13.py`, run before the hash).**
  - Positive: M1 on v1's real i−r x offset must read READABLE.
  - Null: the offsets permuted across galaxies must read NEAR CHANCE.
  - Planted: the permuted-null embeddings plus a random unit direction carrying the real offset
    (measured ρ reported) must read READABLE. The aligned encoders do not exist yet, so M1 is the
    stand-in.
  - SAE: Part 3's planted latent (ρ ≈ 0.7) must trip 1b (`runs/dd/sae/plants.json`).

### 2. Morphology — is the aligned encoder better?

- **Per answer j:** AUC_ej for e ∈ {A1, A2, M1, M2}, a balanced 2×2.
  - D_j = mean(A1, A2) − mean(M1, M2), the aligned-minus-M difference.
  - S_j = (|M1_j − M2_j| + |A1_j − A2_j|) / 2, the pooled seed-to-seed spread of both conditions.
- **Primary, the family statistic over the powered answers:**
  - D̄ = mean_j D_j and S̄ = mean_j S_j (point estimate).
  - A 2,000-draw Poisson galaxy bootstrap of the test split recomputes all four encoders' AUCs per
    resample, with weights shared across answers and encoders (paired by galaxy). That gives the
    95% CI of D̄.
- **States:**
  - **BETTER:** CI(D̄) lies wholly above +S̄. The aligned encoders beat M by more than M differs
    from itself.
  - **WORSE:** CI(D̄) lies wholly below −S̄. Removing the misregistration cost morphology; M's
    offset code may have carried usable colour or position information.
  - **SAME:** CI(D̄) lies inside [−S̄, +S̄]. Any change is within seed noise.
  - **UNRESOLVED:** otherwise. Not settled at two seeds.
- **Per answer, powered only:** the same four states per answer, judged against the **shrunk bar
  B_j = (S_j + S̄) / 2** (not the answer's own S_j alone, nor the pooled S̄ alone). The bootstrap
  p-values of the BETTER and WORSE directions are BY-corrected across the powered answers at
  q = 0.05. Unpowered answers are reported, exploratory.
  - Why B_j: with two seeds per arm, an answer's own S_j is a noisy bar; the homoscedastic null
    plant showed it false-calls 16.4% of answers, against 4.1% for the pooled S̄. The pooled S̄ was
    then tested against a heteroscedastic null (seed noise varying by answer) under a rule set
    before it ran: at most 6% false calls keeps the pooled S̄, above 6% replaces it with B_j. It
    false-called 6.5%, concentrated in the noisiest third of answers (17.3% there, 1.1% in the
    rest), so B_j is the bar. Under the heteroscedastic null B_j false-calls 4.1%; the full rates
    for all four plants are in the plant results below.
  - Each σ_j comes from a single |M1_j − M2_j|, so the heteroscedastic null overstates how much
    the real seed noise varies between answers, and the homoscedastic null understates it. The true
    false-call rates lie between the two nulls, for both bars: 4.1–7.3% for B_j, 4.1–6.5% for S̄.
- **Reported:**
  - the aligned seed-to-seed |A1 − A2| beside S̄;
  - M1 on probe_v2;
  - per-answer D_j for all 37 (all 37 answers are scoreable on the full train and test splits;
    the 36 elsewhere in Brief DD is the sae_eval subset, where one answer is not).
- **Exploratory, no state: the elongation group across the 2×2.** Per-answer AUCs for edge-on
  yes/no, cigar-shaped and completely round, for all four encoders, with D_j and S_j. The group was
  singled out after M's S3 losses and Part 4b's secondary arm; it is noted, not tested.
- **Plants (D28; `artifacts/aligned_c2.py`, run before the hash).**
  - Three plants, each through the identical statistic path, over 20 realisations. The "aligned"
    pair is M1 and M2 with each powered answer's AUC moved to M_e + Δ + ε, where
    ε ~ N(0, σ_seed) is drawn independently per seed and answer and σ_seed comes from M1 − M2.
    - Δ = +0.02 must read BETTER;
    - Δ = −0.02 must read WORSE;
    - Δ = 0 must read SAME.
    - Δ = 0, heteroscedastic: as the Δ = 0 plant, but ε_j ~ N(0, σ_j) with
      σ_j = |M1_j − M2_j| / √2, floored at 0.25 σ_seed. Must read SAME; it tests the per-answer bar
      when seed noise varies by answer, under the 6% rule above.
  - Reported for each: the family detection rate and the per-answer BETTER, WORSE and SAME rates
    under the own S_j, pooled S̄ and shrunk B_j bars. The false-call rate is the rate of BETTER or
    WORSE under a Δ = 0 plant, also split into the third of answers with the largest σ_j and the
    rest.
  - **Declared limitation:** the planted encoders share M's per-galaxy errors, so their bootstrap
    CIs are narrower than a real aligned-against-M comparison's. The plants test the logic, not
    the power.
  - **Realistic power check:** the bootstrap SD of mean_j(M1_j − M2_j), two genuinely independent
    draws, gives sd(D̄) ≈ that / √2. The minimum detectable D̄ ≈ S̄ + 1.96 sd(D̄), with S̄ taken as
    M's spread. If it exceeds 0.02, criterion 2 is declared underpowered for a 0.02 effect before
    the hash.

### 3. Representation — where did the capacity go?

Computed on the test split's pooled block-11 embeddings, per encoder.

- **Top 10 PCs.** Each PC's variance share of the total, and its top family by |ρ|, always
  reported whether or not it clears the bar. For each PC, the galaxy-level |Spearman|
  with three families:
  - offsets: the four above;
  - brightness: modelMag_r and total r flux;
  - morphology: the 37 vote fractions, each on its eligible galaxies.

  A PC **tracks** the family with its largest |ρ| if that |ρ| ≥ 0.3; otherwise it is **untracked**.
  - N = the top-10 variance share tracking offsets or brightness.
  - Mo = the share tracking morphology.
- **Effective dimensionality:** the participation ratio PR = (Σλ)² / Σλ² over the full spectrum.
- **SAE flags** (the 1b SAE, the cards' rule unchanged): a latent is flagged if its strongest
  |ρ| is with a nuisance or brightness variable rather than a vote. Reported:
  - the live fraction;
  - the flagged fraction, split into offset-flagged and other-nuisance-flagged;
  - the interpretable fraction.

  M's figures: 94% of live latents flagged; PC1 and PC2 track offsets (37% of variance, AA3a).
- **The seed-range rule with floor F** (every state below). A direction is called only if both
  aligned seeds lie beyond both M seeds **by more than F_q**: min(A1, A2) − max(M1, M2) > F_q for
  "higher", max(A1, A2) < min(M1, M2) − F_q for "lower"; anything else reads SAME. Without the floor
  the rule false-calls 1/3 of the time when the four seeds are exchangeable (4/24 per direction).
  - **F_q = k · σ_q**, σ_q = |M1_q − M2_q| / √2 (M's observed spread for quantity q). k = 1.110 is
    the smallest multiplier for which the seed-noise null (all four seeds exchangeable,
    ~ N(μ_q, σ_q²)) false-calls ≤ 5%; it is the 95th percentile of the larger of the two gaps over
    10⁶ draws (`artifacts/aligned_c3_floor.py`, seed 20260927). Every quantity is location-scale, so
    one k serves all four.
  - **Plant (D28):** 20 null realisations per quantity at F_q, from the observed M1–M2 spread, must
    false-call ≤ 5% (≤ 1 of 20); the rate over 10⁵ draws is reported beside it. Power is reported at
    two effect sizes per quantity, named before the calibration ran: N −0.10 and −0.25; PR +3 and +6;
    offset-flagged fraction −0.10 and −0.25; Mo +0.05 and +0.10.
  - The floors, from M1 and M2 (`artifacts/out/c3_floor.json`):

    | q | M1 | M2 | σ_q | F_q |
    |---|---|---|---|---|
    | N | 0.541 | 0.480 | 0.0429 | **0.048** |
    | PR | 9.63 | 13.19 | 2.52 | **2.79** |
    | offset-flagged fraction | 0.275 | 0.504 | 0.162 | **0.180** |
    | Mo | 0 | 0 | 0 | **0** (degenerate, declared below) |

  - **Declared: the flag rule is weak.** M's two seeds differ widely in offset-flagged fraction
    (M1 0.275, M2 0.504, among 972 and 1,096 live latents), so F_flag = 0.180 and the rule detects a
    −0.10 change 8% of the time and −0.25 28%. LOWER on flags needs a very large drop.
  - **Declared: Mo's floor is degenerate.** Neither M has a morphology-tracking top-10 PC, so
    σ_Mo = 0 and F_Mo = 0. GAINED then means a morphology-tracking PC (|ρ| ≥ 0.3) in the top 10 of
    both aligned seeds; the |ρ| bar, not seed noise, is the only guard. LOST stays unreachable.
- **States, each by the seed-range rule with floor F_q:**
  - **Nuisance:** **CLEANER** (N lower by > F_N, and no top-10 PC tracks offsets in either aligned
    seed) / **SAME** / **DIRTIER** (N higher by > F_N).
  - **Morphology share:** **GAINED** (Mo higher by > F_Mo) / **SAME** / **LOST**.
  - **Dimensionality:** **HIGHER** / **SAME** / **LOWER** PR, by > F_PR.
  - **Flags:** the offset-flagged fraction **LOWER** / **SAME** / **HIGHER**, by > F_flag. Each seed
    has one SAE (M2's is O2's, trained for this floor under the same rule), so the rule applies here
    too.
- **Meaning:**
  - CLEANER + GAINED: the variance M spent on the offset moved to morphology-tracking directions.
  - CLEANER + SAME Mo: it moved to untracked directions, not to the probed morphology; PR says
    whether it spread out or concentrated.
  - DIRTIER: a new nuisance took its place; the flags and the per-PC table name it.
- **Plants (D28; `artifacts/aligned_c13.py`, run before the hash).** M1 and M2 play the "M" pair.
  - **DIRTIER and CLEANER are power curves, not single plants (v5; user, 2026-09-28).** F_N stays
    0.048. Each is swept over an i−r x-correlated component carrying 10%, 15%, 20%, 25%, 30%, 35% and
    40% of variance, 20 realisations per strength (a fresh direction per seed per realisation;
    `artifacts/aligned_c3_sweep.py`, seed 20261001). The smallest strength read correctly in ≥ 80% of
    realisations is criterion 3's **declared detectable effect**.
    - **DIRTIER:** the "aligned" pair = M1 and M2 each plus the component; the "M" pair = M1 and M2.
    - **CLEANER:** the base = M1 and M2 with their offset-tracking top-10 PCs projected out (N 0.273
      and 0.178, no offset-tracking PC), so CLEANER's second clause can hold. The "aligned" pair = the
      base; the "M" pair = the base plus the component. The pairs differ only by the planted
      component, mirroring DIRTIER.
    - Both read through `aligned_c13`'s own `representation` and `seed_range`, with the floors fixed
      at real M's.

    | strength (share of variance) | 10% | 15% | 20% | 25% | 30% | 35% | 40% |
    |---|---|---|---|---|---|---|---|
    | DIRTIER read (of 20) | 0 | 0 | 0 | **18** | 20 | 20 | 20 |
    | mean ΔN, aligned − M | +0.014 | +0.061 | +0.085 | +0.107 | +0.136 | +0.159 | +0.184 |
    | CLEANER read (of 20) | 9 | 7 | **20** | 20 | 20 | 20 | 20 |
    | mean ΔN, aligned − M | −0.113 | −0.126 | −0.164 | −0.200 | −0.239 | −0.276 | −0.315 |

    - **Declared detectable effects.** Two seeds can detect a nuisance gain of about **25% of
      variance** (DIRTIER in 18 of 20; a mean rise in N of about 0.11) and a nuisance loss of about
      **20% of variance** (CLEANER in 20 of 20; a mean fall in N of about 0.16). A smaller change
      reads SAME, and SAME means "no change at least this large", not "no change".
    - CLEANER at 15% (7 of 20) sits below 10% (9 of 20). Both are below the bar; at 20 realisations
      the difference is noise. CLEANER's bar is set by the base's own seed spread: both "M" seeds must
      clear the larger aligned N (0.273) by F_N.
  - M1 and M2 swapped: must read SAME on all three.
  - PR checks: an isotropic 50-d cloud gives about 50, and a rank-2 cloud gives about 2.
  - A synthetic latent column at ρ ≈ 0.7 with i−r x must be offset-flagged.

### 4. Interpretability rerun (as queued in `interp_tooling.md`, same hashes plus recorded amendments)

- **S3** (`e25e1e27`, with the powered-UNEVEN amendment):
  - the aligned state, against M's FAIL (mean drop 0.029; 37% dead on held-out);
  - **IMPROVED** if the aligned encoder reads PASS or UNEVEN in both seeds; **SAME** if FAIL in
    both; **MIXED** otherwise.
  - Reported: the mean drop and the held-out dead rate, against M's.
- **V3′ spiral** (`97e3ed52`; same-radius arm AUC against brightness and the untrained map):
  - the aligned state (PASS / WEAK / FAIL / REVERSED / INSUFFICIENT), against M's REVERSED;
  - PASS means the maps locate arms beyond brightness at the same radius;
  - bar stays withheld unless a non-saturating comparator is pre-registered.
- **4b set ablation** (`81e5b690`, with the standing control rule):
  - the offset set is S1's rule applied to the aligned SAE;
  - **INAPPLICABLE** if no latent reaches |ρ| ≥ 0.5, which is expected under FIXED (nothing to
    remove); otherwise SPECIFIC / GENERIC / INSUFFICIENT.
- **Plants:** each test's hashed plants are rerun on the aligned encoder before it is scored.
- **M's reference values** (the pipeline rerun on the complete probe_v2 `cut_log.csv`, 4,999
  sae_eval galaxies with offsets; `runs/dd_v4ref`, recorded in `interp_tooling.md`'s amendment of
  2026-09-28). The aligned states are read against these, not against the 3,116-galaxy figures:

  | test | M's value | output | SHA-1 |
  |---|---|---|---|
  | S1 (selects 4b's offset set) | PASS; 40 latents; top 328, ρ +0.910 | `runs/dd_v4ref/sae/score.json` | `d64c782343a5d8b3a5dbce1f62a3ba0a13d4c840` |
  | S3 | FAIL; mean drop 0.029; 37% dead on held-out | `runs/dd_v4ref/sae/M_b11_x8.eval.json` | `f5a70809954e26d5c5e661aaac42d18ed7e019c4` |
  | V3′ | spiral REVERSED; bar REVERSED (withheld) | `runs/dd_v4ref/part1/v3b_score.json` | `0847b75a835e9386f2ea42f5776e83dd3c439376` |
  | 4b | GENERIC; max 0.033 vs p95 0.171, mean 0.011 vs 0.061; offset probe → 0.690 | `runs/dd_v4ref/sae/part4b.json` | `247b9d860f1c1ae586c79b39d023b85e4e6c6583` |
  | card flags (criterion 3) | 972 live; flagged 0.926; offset-flagged 0.275 | `runs/dd_v4ref/sae/flags.npz`, `part4b_stats.npz` | `766ae8e2f0fd7c9add3e7a4a45e725e260716bee`, `491c3c04941ed2630cf55c9d1d2d7ce0dfe70a2f` |
- V1, V2 (with the localised plant, pre-registered separately), V4, S1 and S2 run as queued. They
  are not part of this comparison's reading.

### Reading — the answer to the question

| 1 (fix) | 2 (morphology) | reading |
|---|---|---|
| FIXED | BETTER | Yes: the freed capacity went, at least in part, to probed morphology. 3 says where |
| FIXED | SAME | The fix worked but probed morphology did not improve; 3 says where the capacity went |
| FIXED | WORSE | The fix cost morphology; the offset code carried something usable |
| FIXED | UNRESOLVED | Not settled at two seeds per arm |
| RESIDUAL | any | As above, caveated; the residual offset is named |
| NOT FIXED / INSUFFICIENT | any | Unanswered; 2–4 exploratory |

Precedence: criterion 1's state governs how 2–4 are read. No criterion's state is revised after
another's is seen.

### Plant evidence (pinned)

SHA-1 of each file as it stands after the 2026-09-27 plant runs. The hash of this section covers
these pins, so the plant evidence cannot change after the hash without breaking it.

| file | role | SHA-1 |
|---|---|---|
| `artifacts/out/c2_plants.json` | criterion 2 plants, including the heteroscedastic null | `908479f6fdc155dd14d1f0aec0a14bded7c0e724` |
| `artifacts/out/c13_plants.json` | criterion 1 (1a) and criterion 3 plants (c3 rerun under the floor, v4) | `491e504fd2ab6aa479d5dc43e798305c49959248` |
| `artifacts/out/c3_floor.json` | criterion 3 floor calibration, null and power (v4) | `65fcffb6bc27f2363a72b61b6c3607514f9532cd` |
| `artifacts/out/c3_sweep.json` | criterion 3 DIRTIER and CLEANER power curves, 7 strengths × 20 realisations (v5) | `ee8b3ae64a0e8c256ec6166de7b0a3eccf0e69b0` |
| `artifacts/aligned_c3_sweep.py` | criterion 3 sweep script (v5) | `99d6107b29f1efde3c339d26fa3c44195140bbca` |
| `artifacts/out/c1_untrained_v2.json` | criterion 1 confound floor (untrained encoder on probe_v2) | `e1af6ed59112ab5545166d755f0b535ced0d7265` |
| `runs/dd/sae/plants.json` | 1b's SAE plant (Part 3) | `9cc2f2b6213468b29461a76a2b3f13087e6f6d5d` |
| `artifacts/aligned_c2.py` | criterion 2 script (v4: banks through `probe_bank`; statistic unchanged) | `029adeb522a3026603e36f7bf9187e3e9ae39c8a` |
| `artifacts/aligned_c13.py` | criteria 1 and 3 script, as run (v4: floored `seed_range`, banks through `probe_bank`) | `14361e7ac657b2dffb88782e9ea977f9cfd919ba` |
| `artifacts/aligned_c3_floor.py` | criterion 3 floor calibration | `cd677c466f4f445309b67d11e43375f21bb142e4` |

**Criterion 3's DIRTIER plant, resolved (v5).** In v4 the single 20% plant read SAME under the floored
rule (the planted pair's N 0.603 and 0.574 against M's 0.541 and 0.480, a gap of 0.033 below
F_N = 0.048). The user kept F and replaced the plant with the sweep (2026-09-28); the sweep shows the
20% plant sits below the rule's reach (0 of 20) and declares the reach instead: 25% for DIRTIER, 20%
for CLEANER.

## Settled (user, 2026-09-26)

- M2 = O2, which already exists (see Common ground). The noise term is the pooled seed spread of
  both conditions, in a balanced 2×2.
- Every plant runs before the hash. Criterion 2 has three plants (+0.02, −0.02, 0), with detection
  and false-call rates reported.
- Bounds confirmed:
  - near chance = |AUC − 0.5| ≤ 0.05 with the CI inside [0.40, 0.60];
  - a PC tracks a family at |ρ| ≥ 0.3, and each PC's top family is reported.
- Train size: criterion 2 uses the full train split (the headline protocol); criteria 1, 3 and 4
  keep the DD cap.
- The elongation group is read per answer across the 2×2, as exploratory only.

## Settled (user, 2026-09-27)

- The per-answer bar: the shrunk B_j = (S_j + S̄) / 2, as the 6% rule decided on the
  heteroscedastic null (the pooled S̄ false-called 6.5%). Approved in advance through the rule.
- The fixed training length: A1 and A2 stop at exactly 101,308 steps. Approved.
- The confound floor is added to criterion 1 as reported, with no state.

## Plant results (run 2026-09-26, extended 2026-09-27, before the hash)

M1 = `runs/m/encoder.pt`, M2 = `runs/o2/encoder.pt`. Both are embedded on the full probe train and
test splits (195,686 galaxies each; `artifacts/out/c2_m{1,2}_embeddings.npz`). The M1 bank equals
O1's M embeddings exactly (max |Δ| 0.0 over 5,000 shared galaxies). The weighted bootstrap AUC
equals sklearn's on unit and on expanded Poisson weights.

### Criterion 1 — `artifacts/out/c13_plants.json`: fires

| plant | AUC (95% CI) | state | required |
|---|---|---|---|
| M1, real i−r x | 0.975 (0.973–0.976) | READABLE | READABLE |
| offsets permuted | 0.501 (0.495–0.507) | NEAR CHANCE | NEAR CHANCE |
| permuted + planted coordinate (ρ 0.64) | 0.999 | READABLE | READABLE |

- Resolution: the null's CI half-width is 0.006. An offset AUC above about 0.61 reads READABLE, and
  0.55–0.61 reads UNRESOLVED.
- Rerun 2026-09-27 at 1a's 10,000 bootstrap draws (the draft-v2 run used 2,000). The AUCs and
  states are unchanged, and the CIs move in the fourth decimal.
- 1b's SAE plant is Part 3's (`runs/dd/sae/plants.json`, S1 planted latent → reaches |ρ| ≥ 0.5).

**Confound floor** (reported, no state; `artifacts/out/c1_untrained_v2.json`). Untrained encoder, M's
architecture, seed 0, read on probe_v2; 39,997 train (the 40,000 cap less 3 galaxies with no v2
stamp) and 34,828 test.

| offset | AUC (95% CI, 10,000 draws) | 1a state, for reference |
|---|---|---|
| g−r x | 0.499 (0.493–0.505) | NEAR CHANCE |
| g−r y | 0.503 (0.497–0.509) | NEAR CHANCE |
| i−r x | 0.501 (0.494–0.507) | NEAR CHANCE |
| i−r y | 0.503 (0.497–0.509) | NEAR CHANCE |

- The recorded offsets are not readable from galaxy properties through an untrained encoder on
  aligned pixels. So an aligned-encoder offset AUC above about 0.51 is not explained by this confound.
- Declared: probe_v2 has no frozen normalisation of its own until plan A8, so the stamps pass through
  M's freeze (`75100066b3e0`), cast to fp16 as the cache stores them. That FITS path reproduces M's
  cache exactly on v1 stamps (max |Δ| 0.0 over 16).

### Criterion 2 — `artifacts/out/c2_plants.json`: fires (20 realisations per plant)

- 33 answers are powered: test smaller class ≥ 100 and mean M AUC ≥ 0.6, with the full train
  split.
- σ_seed = 0.0052, and M's mean |M1 − M2| = 0.0063.

| plant | family state (of 20) | per answer, own S_j bar (BETTER / WORSE / SAME) | per answer, pooled S̄ bar | per answer, shrunk B_j bar |
|---|---|---|---|---|
| +0.02 | BETTER 20 | 97.9% / 0% / 1.5% | **100%** / 0% / 0% | **99.5%** / 0% / 0.2% |
| −0.02 | WORSE 20 | 0% / 98.3% / 1.1% | 0% / **100%** / 0% | 0% / **100%** / 0% |
| 0 | SAME 20 | **9.2% / 7.1%** / 82.4% | **2.0% / 2.1%** / 95.0% | **3.9% / 3.3%** / 91.5% |
| 0, heteroscedastic | SAME 20 | **5.6% / 4.7%** / 88.3% | **2.9% / 3.6%** / 92.4% | **2.0% / 2.1%** / 94.5% |

- Family: detection 100% at ±0.02, and no false calls at either Δ = 0 plant.
- Per answer, false calls (BETTER or WORSE) under the two nulls:

  | bar | homoscedastic: all / noisiest third / rest | heteroscedastic: all / noisiest third / rest |
  |---|---|---|
  | own S_j | 16.4% / 0.5% / 24.3% | 10.3% / 9.1% / 10.9% |
  | pooled S̄ | 4.1% / 2.7% / 4.8% | **6.5%** / 17.3% / 1.1% |
  | shrunk B_j | 7.3% / 0% / 10.9% | **4.1%** / 9.1% / 1.6% |

  The noisiest third is the 11 answers with the largest σ_j (0.0055–0.0108; median σ_j over the 33
  is 0.0047, and 5 answers sit at the 0.25 σ_seed floor of 0.0013).
- **The 6% rule, applied.** The pooled S̄ false-called 6.5% under the heteroscedastic null, above
  6%, so the per-answer bar is **B_j = (S_j + S̄) / 2**. With B_j: +0.02 detected 99.5%, −0.02
  detected 100%, and false calls of 7.3% (homoscedastic) and 4.1% (heteroscedastic).
- The three homoscedastic plants reproduce the draft-v2 run exactly (the heteroscedastic plant runs
  last on the same RNG stream).
- **Realistic power:** the bootstrap SD of mean_j(M1_j − M2_j), from two genuinely independent
  draws, is 0.0011, so sd(D̄) ≈ 0.0008. The minimum detectable D̄ ≈ S̄ + 1.96 sd ≈ **0.0078**.
  That is under 0.02, so criterion 2 is powered for a 0.02 family effect.
  - The bar is set by the seed spread (0.0063), not by galaxy sampling (0.0008). Two seeds per arm,
    not the test-set size, limit resolution.
- Declared: the plants' CIs are narrower than a real comparison's, because the planted encoders
  share M's per-galaxy errors. The realistic-width check covers the power question.

### Criterion 3 — `artifacts/out/c13_plants.json`: fires

M's baseline on the probe-test split: each PC's variance share and top family; "tracks" means
|ρ| ≥ 0.3.

| | N (nuisance share, top 10) | Mo (morphology share) | PR | tracking PCs |
|---|---|---|---|---|
| M1 | 0.541 | **0** | 9.6 | PC1 offsets 25.1%, PC2 offsets 11.8%, PC3–4 brightness |
| M2 | 0.480 | **0** | 13.2 | PC1–2 and PC4 offsets, PC3 brightness |

| plant | nuisance | morphology share | dimensionality | required | fires |
|---|---|---|---|---|---|
| i−r x component at 20% of variance | **SAME** (v3, no floor: DIRTIER) | SAME | SAME (v3: LOWER) | DIRTIER | **no** (v4); superseded by the v5 sweep: DIRTIER detectable from 25% |
| offset-tracking PCs projected out | CLEANER | SAME | SAME (v3: HIGHER) | CLEANER | yes |
| M1 and M2 swapped | SAME | SAME | SAME | all SAME | yes |

v4 reruns these three plants through the floored `seed_range`. The DIRTIER plant raises N by 0.062
and 0.093 (to 0.603 and 0.574), but the smaller planted value clears the larger M value by only
0.033, under F_N = 0.048. Its PR falls to 6.3 and 8.2, a gap of 1.4 below M's smaller value, under
F_PR = 2.79.

- PR unit checks: isotropic 50-d gives 49.9; rank 2 gives 1.97.
- A synthetic latent at ρ ≈ 0.7 with i−r x is offset-flagged.
- **Declared reachability:**
  - No top-10 PC tracks morphology in either M (Mo = 0), so **LOST is unreachable**. GAINED needs
    any morphology-tracking PC in both aligned encoders' top 10.
  - M's two seeds span PR 9.6–13.2, so HIGHER needs both aligned encoders above 16.0 and LOWER
    both below 6.8 (the range widened by F_PR = 2.79).

### Criterion 4

Each test's own hashed plants are rerun on the aligned encoder before it is scored, as recorded in
`interp_tooling.md`. Nothing new is needed before this hash.
