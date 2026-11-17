# Ladder v2 against M's ladder: pre-registration (DRAFT v0, not hashed)

**Question.** For each of the 37 Scheme-1 answers, in the full and in the conditional population, does
the aligned encoders' (A1–A3) existence reading match M's? Which of M's existence results look as if
they were carried by the v1 band misregistration?

**Status.** Draft for the user. Nothing here has been hashed, coded, committed or run. The scoring code,
calibration and plants are specified, not written. The hash covers the section from
"## Pre-registration" through its last content line (SHA-1, trailing newline). It comes **before any M
ladder record is opened**, and before any A1–A3 rung, AUC or retention value is read.

**What was read to write this, and what was not.**
- **Read (method text and code only):**
  - `CLAUDE.md`;
  - `docs/ladder_v2/plan.md`;
  - `docs/ladder_v2/inventory.md` lines 40–114 and 175–280;
  - `artifacts/baselines_comparison.md` lines 1–427: the preamble and the hashed Pre-registration
    section;
  - `artifacts/aligned_comparison.md` lines 11–343, skipping the inventory's flagged lines
    (101–109, 133–141, 199, 214–228, 245–262, 281–290, 336–340);
  - `artifacts/t1_ladder.py` (line 104 skipped);
  - `artifacts/p2_ladder.py` (line 63 and the `_rung_row` block skipped at first; the block was read
    later, in a grep of `_rung_row`);
  - `artifacts/s2_verify.py`, `row()` and `main()` only;
  - `src/galaxy_jepa/probing/ladder.py` lines 60–100, 505–660 and 770–830;
  - `artifacts/ladder_env.py`;
  - `docs/ladder_v2/outputs.sha1`, hash lines only.
- **Listed only (filenames, sizes, dates):** `artifacts/out` (→ SSD `artifacts_out`), `runs/`,
  `runs/ladder_v2/{a1,a2,a3,shared}`, `runs/o2`.
- **Schema only:** a key/type walk of `runs/ladder_v2/a1/t1_ladder.json`. It printed no value.
- **Not read:** any M or O2 record, any `*_findings.md`, `aligned_comparison.md` Results, the ladder v2
  report and audit, the baselines report, the memory files.
- **Slips, declared.** Each was seen and is not repeated here. Nothing in this draft depends on any of
  them.
  1. `p2_ladder.py`'s `HEADLINE` constant (it sits at line 60 now, not 63; the inventory's line number
     is stale). It quotes M's E0 headline AUC and O2's range. That is not a ladder result.
  2. `t1_ladder.py` line 98, a print string, names the number of entangled pairs in M's P2 full
     population.
  3. The A1 schema walk printed the lengths of `full_pairs` and `conditional_pairs`. The pair list is
     the top 10% of pairs over existence-passing answers, so its length roughly implies how many
     answers pass existence in A1 in each population.
  4. `aligned_comparison.md`'s hashed pre-registration quotes M's DD figures: S1, S3, V3′, 4b, the card
     flags and the AA3a PC shares. None is a ladder result.

## Pre-registration

### Common ground

- **Encoders.**
  - **M:** `runs/m/encoder.pt`, seed 0, the v1 corpus (misregistered bands).
  - **A1, A2, A3:** M's recipe on pretrain_v2 (band-aligned), training seeds 0, 1 and 2
    (`runs/m_v2`, `runs/m_v2_s1`, `runs/m_v2_s2`).
  - **O2** (M's second v1 seed, `runs/o2`) has **no ladder record**: there is no `p2_ladder`,
    `s2_ladder` or `t1_ladder` for O2 anywhere under `artifacts_out` or `runs/`. Only its O1 matched
    evaluation, union bank and trajectories exist. So the v1 arm is **one seed** (but see open
    decision 2).
- **The record compared: `t1_ladder.json`** (E5, the production ladder: D23 existence, D24 retention
  clearance, D25 "MLP decode unadjudicated"), on both sides.
  - Per population (`full`, `conditional`) it holds 37 rows. Fields used: `feature`, `rung`,
    `mechanism`, `auc`, `real_se`, `n_test`, `positives_test`, `underpowered`, `resolvable_margin`,
    `exceeds_null`, `clean`, `retention`, `retained`, `margin_established`, `competitive`,
    `matched_degenerate`, `k_bar`.
  - The schema was confirmed on A1 by key/type walk only. The same driver wrote M's record.
- **The rung scale.** R1 > R2 > R4, ordered by strength of evidence:
  - **R1:** exists, clean linear direction.
  - **R2:** exists, with a named caveat: entangled, below the effect floor, not selective, or not
    cleared of a nuisance.
  - **R4:** fails linear existence; the MLP decode is unadjudicated.
  - **R3 is unreachable in this code** (no untrained-MLP bar, D25). A recorded R3 is therefore a VOID.
- **Two tiers per cell** (cell = answer × population; 74 cells):
  - **Existence tier (primary):** E = 1 if the rung is R1 or R2 (`exceeds_null`), else 0. This is the
    reading the question is about.
  - **Grade tier (secondary):** R1 against R2, only where all four encoders have E = 1.
- **Populations.** Full is primary: its gate (`vote_count_min = 1`) is frozen. Conditional is
  secondary: its `consensus_gate = 0.5` is still OPEN (p2_ladder's docstring; spec register item 2).
- **What differs between M's ladder and A's, besides the training corpus.** All of it is declared. The
  comparison is of **the v1 pipeline against the v2 pipeline, whose designed difference is band
  alignment**. Nothing below separates these parts.
  1. **Probe pixels:** M is read on probe (v1), A on probe_v2.
  2. **Untrained bar:** M's existence is judged against the v1 K = 30 untrained bank, A's against the
     v2 bank (`runs/ladder_v2/shared/p1_untrained_bank.json`). An untrained encoder on v1 pixels may
     itself read some misregistration. So a change in E is a change in the encoder's margin over its
     own bar, not in its raw AUC.
  3. **Split:** M ran on v1's 40,000-stride / 34,829 split; A on the identical set, which is that split
     minus `ladder_env.NINE`.
  4. **Normalisation freeze:** v1 `75100066…`, v2 `246d8de6…`.
  5. **Code version (precondition P4):** M's `t1_ladder.json` is dated 23 Sep 23:53. Three commits
     touching the ladder code came after it:
     - `b9edeec` (24 Sep 00:01): `ladder.py`, `matching.py`, `mlp.py`, `entanglement.py`, and
       `t1_ladder.py` first committed;
     - `ed876a4` (27 Sep): `floor.py` added; `ladder.py`, `nulls.py`, `config.py` and `probe.yaml`
       changed;
     - `e130fe1` (13 Oct): ladder v2 plumbing.

     A1–A3 ran after all three.

### A. What can and cannot be claimed, with one M seed

- **The seed problem.** The A seeds' disagreement is the only available estimate of seed noise. A
  difference between M and A in one cell cannot be separated from M's own seed noise.
- **A per-cell directional state** (A-HIGHER or A-LOWER) says only this: all three aligned runs read
  the same, M's run reads differently, and the existence statistic of every A run lies beyond M's by
  more than A's seed floor.
  - It is a statement about **the M run**, not the v1 recipe.
  - No per-cell state is a significance claim. Every per-cell count is read against its chance count
    (§D).
- **The aggregate state** (§E) asks whether M is the odd run out among four more often than chance.
  - Under the exchangeable null, M is the odd one in a fraction 1/4 of the cells where exactly one run
    differs.
  - That test assumes the seed noise is **cell-level**: independent between answers.
  - **Run-level seed noise** (one run better or worse on many answers at once) cannot be told apart
    from a v1/v2 effect with one M seed. The best attainable evidence against it is that M is the most
    odd of the four runs, which happens 1 time in 4 under the null. So no claim about the v1 *recipe*
    can reach 5% against run-level noise in this design. The draft says so beside every aggregate
    state.
- **"Carried by the misregistration"** is the reading only of a cell that reads A-LOWER under an
  aggregate M-ODD (down) state, and then "carried by the v1 pipeline", with the five differences
  above. An A-LOWER cell with no such aggregate is listed as "M stands alone here; not distinguishable
  from seed noise in this design".

### B. Preconditions (any failure is VOID; no partial scoring)

- **P1, integrity.** Each input's SHA-1 must match:
  - A: `runs/ladder_v2/{a1,a2,a3}/t1_ladder.json` (`705154a1…`, `aee7d218…`, `2c30f8f2…`) and
    `shared/p1_untrained_bank.json` (`7a8efc74…`), against `docs/ladder_v2/outputs.sha1`;
  - M: `artifacts/out/t1_ladder.json` and `artifacts/out/p1_untrained_bank.json` (v1 bank). These
    SHA-1s are recorded into this section **before the hash, by `shasum` only**: digest printed,
    contents never displayed. If M is re-laddered (open decision 1), the runner's own recorded SHA-1s
    of M′'s outputs are used instead.
  - `score` refuses unless the Pre-registration section's SHA-1 equals the pinned value (house rule).
- **P2, completeness and schema.**
  - Every record has `full` and `conditional`, 37 rows each, with the same feature set.
  - Every row's key set equals the schema above.
  - `k_bar == 30` on every row.
  - Each `checkpoint` names its expected encoder.
  - Every `rung` is in {R1, R2, R4}.
  - A row failing any of these makes that cell VOID. A failure at record level makes everything VOID.
- **P3, split.** For every answer and population, the scoring code computes from the labels and split
  ids alone the eligible test count and positives on each side's split. Each record's `n_test` and
  `positives_test` must equal those exactly.
  - M's count must exceed A's by no more than the number of NINE galaxies eligible for that answer.
  - Any mismatch makes the cell VOID.
- **P4, code version.** Either:
  - (a) M is re-laddered under the code A ran (open decision 1, recommended), when P4 holds by
    construction; or
  - (b) a blinded code-diff review of `b9edeec`, `ed876a4` and `e130fe1`, restricted to the ladder
    path, by an agent who opens no record. It classifies each change as rung-inert or not. Any
    non-inert change makes the whole comparison VOID.
- **P5, existence reproduction.** For every row of all four records, the scoring code recomputes D23's
  `untrained_z` from `auc`, `real_se` and that record's own untrained bank, then BY over the 37 at
  α = 0.05. It calls the package's `nulls` functions, not a re-implementation.
  - The resulting `exceeds_null` must equal the recorded flag.
  - The rung must be consistent with the recorded flags: R4 iff `exceeds_null` is false; R1 only if
    `clean`.
  - Any mismatch makes the record VOID.
  - This also supplies z, the continuous statistic under the existence tier, which the record does not
    store.
- **P6, constants.** The effect floor 0.7267 and `probe.yaml`'s SHA-1 are the same on both sides. A
  difference makes everything VOID.

### C. Per-cell states

**The continuous guard** (existence tier only). Write z_e for encoder e's recomputed D23 z in the
cell.
- **A's seed spread.** In each population, pool the within-cell residuals of A's AUCs over the
  non-VOID cells (2 df per cell), each scaled by a count weight w_c = se_c² / mean(se²), as in the
  baselines' σ̂². Here se_c is the mean `real_se` over the A seeds. This gives σ̂²_AUC, and the cell's
  seed SD on the z scale is σ_z,c = √(w_c σ̂²_AUC) / d_c, where d_c = √(sd_bank,c² + se_c²) is z's
  denominator.
- **Two different banks.** M's bar and A's bar are estimated from different K = 30 banks, which adds
  at most 2/K to the variance of a z difference. So the effective SD is
  σ_eff,c = √(σ_z,c² + 2/K).
- **Floor:** F_c = k_g · σ_eff,c, with **k_g = 1.60**: the smallest k on a 0.05 grid for which one
  exchangeable normal draw lands beyond all three others by more than kσ, in either direction, ≤ 5% of
  the time. A quick Monte Carlo puts the rate at 4.8% at 1.60 and 5.3% at 1.55. The calibration (§G)
  fixes k_g.
- **"Beyond":**
  - A is below M if max_s z_A,s < z_M − F_c;
  - A is above M if min_s z_A,s > z_M + F_c.

**Existence-tier states: first match wins.**

| # | state | condition |
|---|---|---|
| 1 | **VOID** | P1–P6 failed for this cell or its record |
| 2 | **THIN** | M's row, or at least two of A's rows, has `underpowered = true`: "cannot resolve at this N", so a rung there means nothing about presence |
| 3 | **A-SPLIT** | the A seeds are not unanimous on E (2/1). No direction is claimed. It is recorded whether M sides with the A majority (feeds §D) |
| 4 | **A-LOWER** | M has E = 1, all three A have E = 0, and A is below M. Existence M had and A lacks |
| 5 | **A-HIGHER** | M has E = 0, all three A have E = 1, and A is above M |
| 6 | **NEAR-BAR** | A is unanimous and differs from M on E, but the guard fails: the rung moved and the statistic did not move beyond seed noise. Labelled with the direction the rung moved |
| 7 | **SHIFTED** | A is unanimous and equal to M on E, but A is above or below M by the guard (labelled up/down): same reading, the margin moved |
| 8 | **AGREES** | A is unanimous and equal to M on E, within the guard |

- States 4 and 5 are exclusive. The order between them is immaterial.
- **D27 checklist:**
  - **sign flips** across seeds: A-SPLIT;
  - **thin samples:** THIN, plus VOID on count mismatch;
  - **significance (the rung) without magnitude (z):** NEAR-BAR;
  - **magnitude without significance:** SHIFTED;
  - **precedence:** the table.

**Grade-tier states** (cells where all four encoders have E = 1 and the existence state is AGREES or
SHIFTED; first match wins):
1. **G-VOID**: inherits VOID or THIN.
2. **G-SPLIT**: the A seeds are not unanimous on R1/R2.
3. **G-HIGHER** (M R2, all A R1) or **G-LOWER** (M R1, all A R2). M's and A's `mechanism` strings are
   reported beside each.
4. **G-AGREES.**

The grade tier is **unguarded**. R1 against R2 turns on several gates, and the record does not store
the statistic under most of them (selectivity, for example), so no common magnitude guard exists.
This is declared (open decision 5).

**Retention under matching (S/P/C/U): reported, no state.** Beside every cell: each encoder's
`retention`, `retained`, `margin_established` and `competitive`, with an agreement column (`4/4`,
`3/1 (M differs)`, `3/1 (A? differs)`, or other).
- Why there is no state:
  1. In R4 cells retention is UNRESOLVED by construction (no established margin).
  2. Where it matters, it already enters the rung through `nuisance_cleared` (R2's "confounded by …"
     mechanism). A separate state would count it twice.
  3. The matched nuisance is each encoder's own worst, and v1 and v2 pixels may differ in which one
     that is. A retention verdict against a different nuisance is not the same measurement.
  4. UNRESOLVED is not a point on the S > P > C order.
- Open decision 4 offers a restricted state instead.

### D. Chance counts (multiplicity across 37 × 2)

- **The null:** in each cell the four runs (M, A1, A2, A3) are exchangeable draws.
- **The odd-one-out set.** For a population and direction (down: one run has E = 1 and three have
  E = 0; up: the reverse), let O be the non-VOID, non-THIN cells where exactly one of the four runs
  differs from the other three in that direction. Write n = |O|.
- **Under the null,** each run is equally likely to be the odd one in each such cell. So:
  - E[# A-LOWER, unguarded] = n_down / 4, and likewise for up;
  - **estimated from A's disagreement alone:** E = N_A-odd / 3, where N_A-odd is the number of cells in
    O whose odd run is an A seed (M sides with the A majority).
- **A bound that needs only A's records:** E[# unanimous-A cells differing from M] ≤ N_nonunan / 3,
  where N_nonunan is the number of cells with the A seeds not unanimous.
  - For a cell whose rung distribution is q, the ratio of the two probabilities is
    (Σq³ − Σq⁴) / (1 − Σq³). That is ≤ 1/3 for every q: proved for two categories, where it equals
    (p² + (1 − p)²)/3, and checked numerically over the three-category simplex (max 0.3333, 200,000
    Dirichlet points).
  - The scoring code computes all three numbers mechanically after P1–P6 pass. It prints them beside
    the per-cell state counts **before** any per-cell list.
- **Guarded counts** (A-LOWER/A-HIGHER after the z guard) are read against the null rate the plants
  measure for the guarded rule (§G), not against n/4.
- **No per-cell state is pooled.** Each population's counts are reported separately. The full
  population is primary.

### E. Aggregate states (per population and direction; the full population is primary)

- **Statistic:** m = the number of cells in O whose odd run is M. Under the cell-level null,
  m ~ Binomial(n, 1/4), and the p-value is P(X ≥ m).
- **Run-level guard:** m > m_s for every A seed s, where m_s is the number of O cells whose odd run is
  A_s. M must be strictly the most odd of the four runs.
- **α = 0.025 per direction, one-sided.** Two directions are tested in the primary population, so
  this is 5% Bonferroni-split.
- **"The count of answers moving R1/R2 → R4"** is m_down for the full population.

**States: first match wins.**

| # | state | condition |
|---|---|---|
| 1 | **VOID** | any of P1, P2 (record level), P4, P5 or P6 failed, or more than 3 of the population's 37 cells are VOID |
| 2 | **INSUFFICIENT** | n < 3. The binomial's smallest attainable p, 4⁻ⁿ, cannot reach 0.025 below n = 3 (4⁻³ = 0.016), so the bar is unreachable. This is an error state, not a null (the D28 rule) |
| 3 | **M-ODD BOTH WAYS** | both directions satisfy state 4's conditions. M is a different run, and no direction is claimed |
| 4 | **M-ODD** (down: "CARRIED"; up: "MASKED") | p ≤ 0.025 and the run-level guard holds |
| 5 | **SEED-LIKE** | p ≤ 0.025 but some A seed is the odd run at least as often as M. M's oddness is the size seed-to-seed differences show |
| 6 | **NOT-ODD** | p > 0.025. Reported with the smallest m that would have fired at this n, as its reach |

- **Beside every M-ODD**, the run-level null rate from §G is printed, i.e. the false-call rate under
  run-level seed noise at the observed-like grid row. The sentence "against run-level seed noise this
  is capped at 1 in 4 with one M seed" goes with it.
- **The conditional population** gets the same table as a secondary reading.
- **The grade tier** gets the same binomial on R1/R2 odd-one-out cells. It is secondary and reported
  with its state.

### F. Reading

- **Primary:** the full population's two aggregate states, CARRIED (down) and MASKED (up).
  - "Some of M's existence results were carried by the v1 pipeline" is claimed only on M-ODD (down).
    The cells named are then its A-LOWER cells, the guarded ones first.
  - M-ODD BOTH WAYS, SEED-LIKE, NOT-ODD and INSUFFICIENT name no answer as carried.
  - NOT-ODD is "no evidence at this reach", never "nothing was carried".
- **Secondary:**
  - the per-cell table (74 rows: the existence state, the grade state, the four rungs, the four z
    values, F_c, the mechanisms, and the retention columns), read against §D's chance counts;
  - the conditional population's aggregates;
  - the grade tier.
- **Reported, no state:**
  - retention agreement;
  - the population changes (full → conditional rung changes) per encoder, side by side;
  - AUC differences A − M per answer.
- **Out of scope:** no state about the v1 recipe as against the v2 recipe (that needs a second v1
  seed, open decision 2); no state on E6 pair verdicts, R-series, U-series or DD.

### G. Calibration and plants (D28): specified here, run before the hash, on synthetic inputs only

**Scoring entry point.** One function, `score(records, banks, labels, split)`, used by the plants and
the real run alike. Plants build records **from the schema**: the full t1 key set, 37 rows × 2
populations × 4 encoders, synthetic `auc`, `real_se` and synthetic banks. P5's z recomputation and BY
are therefore exercised, not bypassed. Expectations are fixed in the script before it runs.

**Calibration** (`calibrate`, seed fixed in the script, ≥ 20,000 draws per row):
- **Latent per cell:** μ_c, the cell's true z position relative to its BY critical value, drawn from a
  mixture: mass near 0 (borderline answers) and mass far above or below.
- **Each run's z:** z_ec = μ_c + τ·u_e + σ·ε_ec, where u_e ~ N(0, 1) is a run-level offset shared
  across cells and ε_ec is cell-level noise.
- **Rung:** E from the threshold. The grade tier is drawn from a second latent in the same way.
- **Grid:**
  - σ ∈ {0.5, 1, 2} in z units;
  - τ ∈ {0, 0.5σ, σ, 2σ};
  - borderline mass tuned so the A seeds' non-unanimity rate is 5%, 10%, 20% or 30% of cells.
- **Measured per row:**
  1. the per-cell directional false-call rate, unguarded and guarded, and k_g by the rule in §C;
  2. the false-call rate of the §D chance-count estimators, i.e. their bias against the true
     expectation;
  3. the aggregate M-ODD false-call rate per direction. It must be ≤ 2.5% at τ = 0, which the binomial
     guarantees, so this is a code check. At τ > 0 it is reported, not bounded: this is the run-level
     limitation, measured;
  4. SEED-LIKE's and INSUFFICIENT's rates under the null.
- **The observed-like row,** fixed now: after P1–P6 pass, the scoring code selects the grid row whose
  A non-unanimity rate is nearest the observed one (ties go to the higher τ). It quotes that row's
  rates beside every state. This is the "observed-like" null without anyone looking at values before
  the hash.

**Full-path plants** (each must produce the named state):

| plant | construction | must read |
|---|---|---|
| identical | four copies of one synthetic record | every cell AGREES; aggregate INSUFFICIENT (n = 0) |
| lost-8 | 8 cells: M E = 1 with z = crit + 4; all A E = 0 with z = crit − 4; elsewhere identical | those 8 A-LOWER; aggregate M-ODD (down), p = 4⁻⁸ |
| gained-8 | the mirror | 8 A-HIGHER; M-ODD (up) |
| both | lost-4 and gained-4 together | M-ODD BOTH WAYS |
| split | 6 cells with A at 2/1 | A-SPLIT in those 6 |
| near-bar | 5 cells where the rung flips but each z is within 0.2 F_c of M's | NEAR-BAR in those 5 |
| shifted | 5 cells with the same rung and A's z = M's + 3 F_c | SHIFTED (up) in those 5 |
| thin | `underpowered` set on M in 3 cells and on two A seeds in 3 others | THIN in all 6 |
| grade | 5 R1/R2 cells, M R1 and all A R2 | G-LOWER in those 5 |
| seed-like | lost-8 on M, plus A2 odd-one-down in 8 other cells | SEED-LIKE (down) |
| below-reach | lost-2 only | INSUFFICIENT |
| null (cell-level) | 1,000 realisations at the observed-like grid row, τ = 0 | aggregate M-ODD ≤ 2.5% per direction; per-cell guarded false calls ≤ the calibrated rate + 2 binomial SE |
| null (run-level) | the same at τ = σ | rate reported beside M-ODD; no bound (the declared limitation) |
| power | lost-k, k ∈ {2, 4, 8}, at the observed-like noise | detection rate reported |
| VOID set | each separately: a wrong SHA-1; a missing answer; a recorded R3; an `n_test` off by 10; `k_bar` 29; `exceeds_null` flipped against P5's recomputation; a different `probe.yaml` SHA-1 | VOID at the specified level |

The plant JSON's SHA-1 is pinned into this section before the hash, as in the aligned comparison's
"Plant evidence" table.

## Open decisions for the user

1. **M's record (P4).** M's `t1_ladder.json` predates three commits on the ladder path, including
   `ed876a4` (the floor resolution, `nulls.py`, `probe.yaml`).
   - (a), **recommended:** re-run E5 `t1_ladder` for M under the current code into
     `runs/ladder_v2/m/` with `LADDER_SPLIT=identical`. This needs M's union bank, v1 untrained bank
     and seeds bank, and its noise matrix keyed by encoder.
     - It removes the code, split and NINE differences in one step.
     - It costs about 2–3 h CPU, on A's t1 timings, plus s2_verify's noise matrix if it is not
       reusable.
     - The runner prints no values. M's original record is then used only for a reproduction check,
       reported pass/fail with no count.
     - It must be checked first that `LADDER_SPLIT=identical` works on the v1 probe corpus.
   - (b) Use the original record and run the blinded diff review.
2. **O2's ladder.** No O2 ladder record exists. Running O2's E3 (already done:
   `artifacts/out/o2_matched.json` and `o2_embeddings.npz` exist), then s2_verify and t1, about 5–7 h,
   would give the v1 arm two seeds.
   - Then a 2-against-3 seed-range rule could replace the binomial's single-run caveat, and "M's
     recipe" claims become possible.
   - Doing it changes this design. Decide before the hash.
3. **M read on probe_v2 through the ladder** (M on aligned pixels), to separate a test-time pixel
   effect from a training effect. About 12 h. Reported with no state, if run.
4. **Retention.** No state (recommended, §C), or a restricted state: S > P > C, only in cells where
   all four encoders have E = 1, a competitive nuisance and the same worst nuisance, with UNRESOLVED as
   VOID for retention.
5. **The grade tier is unguarded.** Accept that, or guard only "below the effect floor" moves by
   |AUC − 0.7267| against A's AUC seed SD.
6. **THIN's rule:** M, or at least two A seeds, underpowered. The alternatives are "any of the four" or
   "M or all three A".
7. **Aggregate α:** 0.025 per direction, full population only. Or 0.0125, if the conditional
   population's aggregates are to be confirmatory too.
8. **The run-level guard in M-ODD** (M must be the strictly most odd run). Keep it (recommended), or
   report it beside the state without gating on it.
9. **The VOID allowance:** at most 3 of 37 VOID cells per population before the aggregate is VOID.
10. **Whether `shasum` on M's records counts as blind** for P1's pins. It reads bytes and prints only a
    digest.
11. **k_g:** take it from §G's calibration rule (indicatively 1.60), or set it by judgement.
