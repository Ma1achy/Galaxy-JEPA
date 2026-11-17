# M and O2 against A1–A3, seed-matched pairs: pre-registration (DRAFT v3, final before the hash; not hashed)

**Question.** For each of the 37 Scheme-1 answers, in the full and in the conditional population, does
the existence reading change between the v1 pipeline (misregistered bands) and the v2 pipeline
(band-aligned) when the training seed is held fixed? Which of v1's existence results look as if they
were carried by the v1 pipeline?

**Status.** Draft v3 for the user. v1 was committed as d4d0c86. v2 applied the user's decisions on v1, and v3 applies the user's final changes before the hash (both 17 Nov). Nothing here has been hashed; v3 is uncommitted. The
scoring code, calibration and plants are written and have been run on synthetic inputs only
(`artifacts/m_comparison.py`; results below the hashed section). The hash covers the section from
"## Pre-registration" to the next level-2 heading (SHA-1, trailing newline). It comes **before any M or
O2 ladder record is opened**, and before any A1–A3 rung, AUC or retention value is read.

**What was read to write this, and what was not.**
- **v0 (earlier agent):** as listed in v0 (b5a444b): `CLAUDE.md`; `docs/ladder_v2/plan.md`;
  `docs/ladder_v2/inventory.md` 40–114 and 175–280; `artifacts/baselines_comparison.md` 1–427;
  `artifacts/aligned_comparison.md` 11–343 less flagged lines; `t1_ladder.py` (line 104 skipped);
  `p2_ladder.py`; `s2_verify.py` `row()`/`main()`; `ladder.py` 60–100, 505–660, 770–830;
  `ladder_env.py`; `docs/ladder_v2/outputs.sha1` hash lines; a key/type walk of
  `runs/ladder_v2/a1/t1_ladder.json`.
- **v1 (this agent), code and method text only:**
  - `docs/m_comparison/prereg.md` (v0, in full);
  - `artifacts/t1_ladder.py` lines 1–90 and 105–191 (91–104 skipped: v0's slip 2 sits there);
  - `artifacts/s2_verify.py` lines 1–90; `artifacts/p2_ladder.py` `_rung_row` (85–102);
  - `artifacts/p1_untrained_bank.py` in full; `artifacts/ladder_env.py` in full;
  - `src/galaxy_jepa/probing/ladder.py` 55–167, 356–388, 511–572, 655–834, plus greps;
  - `src/galaxy_jepa/probing/nulls.py` 106–260 and 370–529; greps of `floor.py`, `matching.py`;
  - `configs/probe.yaml`, grep lines only (alpha, effect floor, multiplicity, existence method);
  - `artifacts/baselines_comparison.py` 1–72, 185–260, 660–800, 880–909;
  - `artifacts/baselines_comparison.md` headings, §F (354–427) and "Scoring code of record" (657–693);
  - `docs/ladder_v2/manifest.yaml` lines 1–35 and its `LADDER_*` lines;
  - `docs/ladder_v2/outputs.sha1`, the four A/shared hash lines only;
  - the uncommitted working-tree diff of `artifacts/j4_spread_controls.py` (`_minus_nine`).
- **v2 (this agent), code and method text only:**
  - `docs/ladder_v2/manifest.yaml` E5.t1_ladder.a1 entry, plus a grep for entry ids;
  - `docs/ladder_v2_m/manifest.yaml` lines 1–28 and its E5.t1_ladder.m entry (the code pins);
  - a script comparing the two manifests' code pins (keys and SHA-1s only);
  - `git log` / `git show --stat 17945d9`, and `git diff fcea3b2 17945d9 -- src/galaxy_jepa` (harness.py,
    objectives/moco.py);
  - `shasum artifacts/out/t1_ladder.json` (digest only, §I) and `shasum` of both manifests.
  - Listed only: `docs/ladder_v2_m/` (manifest.yaml, outputs.sha1, runner_log, state.json). The last
    three were not opened.
- **Listed only (filenames):** `docs/`, `docs/ladder_v2*`, `runs/`, `runs/ladder_v2/`,
  `runs/ladder_v2/{a1,a2,a3,shared,encoders}`, `runs/m`, `runs/o2`. `runs/ladder_v2/{m,o2,shared_v1}`
  do not exist yet.
- **Schema:** no walk this time. The row schema is taken from the code that writes it (`_rung_row`,
  `s2_verify.row`, `t1_ladder.main`).
- **Not read:** any M, O2 or A1–A3 record value; any `*_findings.md`; the ladder v2 report and audit;
  the baselines report; `aligned_comparison.md`'s Results; the memory files.
- **Slips, declared.** Each was seen and is not repeated here. Nothing in this draft depends on any of
  them.
  1. (v0) `p2_ladder.py`'s `HEADLINE` constant: M's E0 headline AUC and O2's range. Not a ladder result.
  2. (v0) `t1_ladder.py` line 98, a print string: the number of entangled pairs in M's P2 full
     population.
  3. (v0) The A1 schema walk printed the lengths of `full_pairs` and `conditional_pairs`, which roughly
     imply how many answers pass existence in A1 in each population.
  4. (v0) `aligned_comparison.md`'s hashed pre-registration quotes M's DD figures (S1, S3, V3′, 4b, card
     flags, AA3a PC shares). None is a ladder result.
  5. (v1) A `grep` of `p2_ladder.py` printed the `HEADLINE` line again (the same content as slip 1).
  6. (v1) The filename listing of `runs/ladder_v2/{a1,a2,a3}` shows the `v3_pc*_high/low.png` names,
     which reveal which principal components each A run's V3 rendered. Not an existence or ladder value.
  7. (v1) `baselines_comparison.md` §F and "Scoring code of record" print synthetic calibration and
     plant rates from that study. They are synthetic, not ladder values; listed for completeness.
  8. (v2) The commit message of 17945d9, shown by `git show --stat`, gives the identical split's
     galaxy counts (train + test, v1 against v2) and the split-ids prefix. These are split sizes, not a
     ladder value.
  9. (v2) The M/O2 manifest's pins include `runs/m/encoder.pt`'s SHA-1. This is a checkpoint
     fingerprint, not a value.

## Pre-registration

### Common ground

- **Encoders and pairs** (user, 17 Nov, confirmed from the drivers):

  | run | corpus | train seed | checkpoint | pair |
  |---|---|---|---|---|
  | M | v1 | 0 | `runs/m/encoder.pt` | P1 (with A1) |
  | A1 | v2 | 0 | `runs/m_v2/encoder.pt` | P1 |
  | O2 | v1 | 1 | `runs/o2/encoder.pt` | P2 (with A2) |
  | A2 | v2 | 1 | `runs/m_v2_s1/encoder.pt` | P2 |
  | A3 | v2 | 2 | `runs/m_v2_s2/encoder.pt` | unpaired |

  - Init weights are identical within each pair (`artifacts/o2_second_seed.py:62`,
    `artifacts/aligned_train.py:42-48`). Splits are on cfg.seed 0 for all five.
  - **The primary unit is the matched pair (v1, v2) with the seed held fixed:** (M, A1) and (O2, A2).
    A3 is the third v2 seed. It is unpaired and estimates v2 seed noise; it can veto a direction
    (A-SPLIT) but cannot confirm one.
- **The records compared: `t1_ladder.json`** (E5, the production ladder: D23 existence, D24 retention
  clearance, D25 "MLP decode unadjudicated"), all five written by the same drivers:
  - M and O2: `runs/ladder_v2/{m,o2}/t1_ladder.json`, re-laddered under the current code on v1 pixels,
    `LADDER_SPLIT=identical`, against the v1 untrained bank (K = 30) built on the identical split,
    `runs/ladder_v2/shared_v1/p1_untrained_bank.json`.
  - A1–A3: `runs/ladder_v2/{a1,a2,a3}/t1_ladder.json`, against
    `runs/ladder_v2/shared/p1_untrained_bank.json` (v2, K = 30).
  - Per population (`full`, `conditional`) each record holds 37 rows. Fields used: `feature`, `rung`,
    `mechanism`, `auc`, `real_se`, `n_test`, `positives_test`, `underpowered`, `resolvable_margin`,
    `exceeds_null`, `clean`, `competitive`; where present, `retention`, `retained`,
    `margin_established`, `C`, `k_bar`.
  - Top level: `checkpoint`, and `k_bar`, which is the number of **D24** untrained draws
    (`matching.RETENTION_SEEDS` = 3), not the bank's K.
- **The rung scale.** R1 > R2 > R4. R1: exists, clean linear direction. R2: exists, with a named caveat.
  R4: fails linear existence; MLP decode unadjudicated. R3 is unreachable in this code (no untrained-MLP
  bar, D25), so a recorded R3 is a VOID.
- **Two tiers per cell** (cell = answer × population; 37 per population):
  - **Existence tier (primary):** E = `exceeds_null` (R1 or R2).
  - **Grade tier (secondary, unguarded, exploratory):** R1 against R2, only where all five runs have
    E = 1.
- **Populations.** Full is the only confirmatory population (`vote_count_min = 1`, frozen).
  Conditional is secondary (its `consensus_gate = 0.5` is still open).
- **What differs between the two sides of a pair, besides the training corpus.** All of it is declared.
  The comparison is of **the v1 pipeline against the v2 pipeline**, whose designed difference is band
  alignment. Nothing below separates these parts.
  1. **Probe pixels:** v1 runs are read on probe (v1), v2 runs on probe_v2.
  2. **Untrained bar:** each side's existence is judged against its own pixels' K = 30 bank. An untrained
     encoder on v1 pixels may itself read some misregistration. So a change in E is a change in the
     encoder's margin over its own bar, not in its raw AUC. Both bars are reported beside every cell
     (§C).
  3. **Normalisation freeze:** v1 `75100066…`, v2 `246d8de6…`.
  - **Equalised by the re-ladder, and checked:** the split (identical set, v1's split minus
    `ladder_env.NINE`, on both sides: P3) and the code (P4).
- **Caveat: the conditional population's bar.** Each side's D23 bank is built on the full population's
  rows (`p1_untrained_bank.py` uses the default labels), and `run_ladder` reuses that same per-answer
  bank when it judges the conditional population. This holds identically on both sides, so it does not
  bias the comparison. It is recorded in the ladder v2 register (user, 17 Nov). Conditional existence
  here is therefore "margin over the full population's untrained bar".

### A. What two pairs can and cannot claim

- **Can claim, per cell:** that in both seed-matched pairs the existence reading moved the same way,
  that each pair's move in the continuous statistic exceeds the within-pair noise by the calibrated
  guard, and that the unpaired v2 seed agrees. This is a statement about **two seeds of the v1 recipe
  against the same two seeds of the v2 recipe**. No per-cell state is a significance claim. Every
  per-cell count is read against its chance counts (§E).
- **Can claim, in aggregate:** that more cells flip down (or up) in both pairs than the **paired
  sign-flip null** allows. Under that null, v1 and v2 are exchangeable within each pair, so for a cell
  where both pairs flip, the four flip patterns are equally likely. This null is exact for noise that
  is independent across cells, and it is **untouched by a shared per-seed offset u_seed**, cell-level
  or run-level, because u_seed is common to both members of a pair and cancels.
- **Cannot claim:**
  1. **Against corpus-specific run-level noise** (an offset belonging to one run alone, not shared with
     its pair partner), two pairs give little protection. Each pair's run-level tilt has a random sign,
     so both pairs tilt the same way half the time, and both tilt down a quarter of the time. The
     aggregate's false-call rate under such noise is measured (§F) and reported beside the state, not
     bounded and not gated, with no change to α (user, 17 Nov). No single "observed" rate is printed.
     What is printed beside the state is how much run-specific noise it would take to break it, and
     an upper bound on that noise from A1–A3 (§E).
  2. **A corpus effect against a seed × corpus interaction.** With two seeds, an effect present in one
     seed only reads ONE-PAIR. It cannot be told from noise.
  3. **The training corpus against the read-out pixels.** M read on probe_v2 is deferred (user,
     17 Nov), so every directional state is about the v1 *pipeline*: the training corpus, the probe
     pixels and the untrained bar together.
  4. **A3 cannot confirm.** It is not paired. A3 disagreeing turns a two-pair move into A-SPLIT (the
     veto, kept by the user, 17 Nov). A3 agreeing adds nothing beyond the requirement. A-SPLIT's own
     chance rate is printed beside it (§E).
  5. **A null is not an absence.** WITHIN-CHANCE and INSUFFICIENT are "no evidence at this reach",
     never "nothing was carried". Calibrated power is given in §F.

### B. Preconditions (any failure is VOID at the stated level; no partial scoring)

- **P1, integrity (everything).** Each input's SHA-1 must equal its pin:
  - A1–A3 `t1_ladder.json` and the v2 bank, from `docs/ladder_v2/outputs.sha1`: `705154a1…`,
    `aee7d218…`, `2c30f8f2…`, `7a8efc74…` (full digests in `PINS`);
  - M, O2 and the v1 bank (`runs/ladder_v2/{m,o2}/t1_ladder.json`,
    `runs/ladder_v2/shared_v1/p1_untrained_bank.json`): **from `docs/ladder_v2_m/outputs.sha1`**, the
    M/O2 runner's own record, matched by path. That file's own SHA-1 is recorded outside the hash when
    the runner finishes, by fingerprint only, before any scoring (`V1_OUTPUTS_SHA1`). Scoring refuses
    unless the file matches it. `shasum` counts as blind (user, 17 Nov).
  - At scoring, A's four pins are also re-read from `docs/ladder_v2/outputs.sha1`. A disagreement is a
    P1 failure.
  - The loader refuses to read any path under `runs/`, `artifacts/out` or their SSD targets unless this
    section's SHA-1 equals `PREREG_SHA1` (house rule). A refused read happens before a byte is opened.
- **P2, completeness and schema.**
  - Record level (everything VOID): each record has `full` and `conditional`, 37 distinct answers each,
    the same answer set in all five; top-level `k_bar == 3`; `checkpoint` ends with the run's checkpoint
    above. Each bank has K = 30 seeds covering every answer; the two banks have the **same** `model_key`
    and `split_key` (same architecture, identical split) and **different** `pixels_key`s.
  - Row level (that cell VOID): every key of the field list above present; `rung` in {R1, R2, R4};
    `auc` and `real_se` finite, `real_se` > 0; a row-level `k_bar`, if present, equals 3.
- **P3, split (that cell VOID).** All five records share the identical split and the same labels, so
  `n_test` and `positives_test` must be **equal across all five** in every cell.
- **P4, code (everything).** **The code pins come from the two runners' manifests**, which pin every
  code file per entry, and which the runner checks at launch. A1–A3 ran at the ladder v2 code; M and
  O2 run at commit 17945d9 (the `_minus_nine` fix and the M/O2 manifest).
  - The manifests are read through `git show` at their committing commits, each checked against its
    SHA-1:
    - `docs/ladder_v2/manifest.yaml` at fcea3b2: `6c2f462009ce7f60a81b85e9eca02b8b2662732c`;
    - `docs/ladder_v2_m/manifest.yaml` at 17945d9: `b4dbd5171c8c96cbe88e83cfebeedb921d7c4ae4`.
  - **Scope:** the comparison's entries (t1_ladder, s2_verify, p2_ladder, p1_untrained_bank, r_bank,
    r0, o1_matched), for a1–a3/shared on one side and m/o2/shared_v1 on the other. Every code pin under
    `artifacts/`, `src/` and `configs/` must hold one SHA-1 per side and appear on both sides. Each must
    be equal across the sides, except two declared differences, each fixed as its exact (A, M/O2) pair
    and classified before the hash from the code diff alone:
    1. `artifacts/j4_spread_controls.py` `9598302b…` → `de657854…`: `_minus_nine` (17945d9). On v2 the
       nine are absent, so A's set is unchanged (its test). On v1 it makes the split identical to A's.
    2. `src/galaxy_jepa/**.py` `9400ddf4…` → `afc8ce9b…`: between fcea3b2 and 17945d9 the package
       changes only in `harness.py` and `objectives/moco.py`. Both changes are MoCo's `sky_sigma` default
       and docstrings: training-side, off the probing path of a JEPA encoder. Classified rung-inert.
  - Any other difference VOIDs everything.
- **P5, existence reproduction (everything).** For every well-formed row of all five records, D23 is
  recomputed through the package (`nulls.untrained_z_pvalue` with that side's bank, then
  `family_significant`, BY, α 0.05, family 37). The checks:
  - the recomputed `exceeds_null` equals the recorded flag;
  - R4 iff not `exceeds_null`; R1 only if `clean`;
  - `clean` == `exceeds_null` and `auc` ≥ 0.7267, which also pins the effect floor;
  - `resolvable_margin` equals the package's (|Δ| ≤ 1e-9), and `underpowered` equals
    `is_underpowered` of it.
  - Any mismatch VOIDs everything.
- **P6, VOID allowance (that population's aggregate).** At most 3 of 37 VOID cells per population
  (user, 17 Nov). More, and that population's aggregate is VOID. Its cells are still listed. (v0's P6,
  constants, is folded into P4 and P5.)

### C. The statistic, the guard, and the untrained bars

- **z.** For run r in cell c, z_rc = (AUC − mean C_side) / √(sd C_side² + se_r²), with mean C_side and
  sd C_side from that side's K = 30 bank: D23's statistic, which P5 ties to the recorded flag.
- **The pair differences:** Δ1 = z_A1 − z_M and Δ2 = z_A2 − z_O2.
- **Within-pair noise.** σ̂² = mean over the population's non-VOID cells of (Δ1 − Δ2)² / (2 ν_c).
  - In Δ1 − Δ2, u_seed cancels, a corpus effect common to both seeds cancels, and both banks' errors
    cancel. What remains is run noise.
  - The estimate is uncentred, so a pair-level drift inflates it, which is conservative.
  - ν_c is the cell's relative z-noise: the mean over the five runs of (se/d)², normalised to mean 1,
    where d is the z denominator. This is the house assumption that seed noise scales with the
    sampling SE, carried to the z scale.
- **The bank term.** b_c² = Σ_side sd C_side² / (K · d̄_side²). This is the error of the two bank means,
  which is common to both pairs and does not cancel.
- **The guard:** F_c = k · √(ν_c σ̂² + b_c²), with **k = 1.1** from the rule in §F.
  - A pair moves down "beyond the guard" if Δ_p < −F_c, and up if Δ_p > F_c.
- **The untrained bars, reported beside every cell, with no state** (user, 17 Nov):
  - **D23's bar** is the bar existence is judged against. It is held in each side's bank,
    `p1_untrained_bank.json`, under `seeds[s][answer]`: per side, mean, sd and K. The bank is built on
    the full population's rows, and `run_ladder` applies the same per-answer bank in both populations.
    So one D23 bar per answer and side is printed in both populations.
  - **D24's bar `C`** is the mean untrained AUC over the three D24 draws on this population's rows. It is
    in each record's row, only where the row carries a retention verdict. It is printed per run, or "—".

### D. Per-cell states

Write c1 = E_A1 − E_M and c2 = E_A2 − E_O2, each in {−1, 0, +1}.

**Existence tier: first match wins.**

| # | state | condition |
|---|---|---|
| 1 | **VOID** | P2 (row) or P3 failed for this cell, or P1/P2 (record)/P4/P5 failed |
| 2 | **THIN** | M or O2 has `underpowered`, or at least two of A1–A3 do: "cannot resolve at this N" |
| 3 | **PAIRS-OPPOSE** | c1 · c2 = −1: the two pairs flipped E in opposite directions |
| 4 | **ONE-PAIR** (↓/↑) | exactly one pair flipped E. Labelled with its direction, not claimed |
| 5 | **A-SPLIT** (↓/↑) | c1 = c2 ≠ 0, but A3's E differs from A1's and A2's: the v2 seeds are not unanimous |
| 6 | **A-LOWER** | c1 = c2 = −1, A3 agrees (E = 0), and Δ1 < −F_c and Δ2 < −F_c: existence both v1 seeds had and every v2 seed lacks |
| 7 | **A-HIGHER** | c1 = c2 = +1, A3 agrees (E = 1), and Δ1 > F_c and Δ2 > F_c |
| 8 | **NEAR-BAR** (↓/↑) | c1 = c2 ≠ 0 and A3 agrees, but the guard fails in at least one pair: the rung moved, the statistic did not move beyond pair noise |
| 9 | **SHIFTED** (↓/↑) | c1 = c2 = 0, and both pairs lie beyond F_c in the same direction: same reading, the margin moved |
| 10 | **SEED-SPLIT** | c1 = c2 = 0, not SHIFTED, and E_M ≠ E_O2: the reading follows the seed, identically on both corpora |
| 11 | **AGREES** | c1 = c2 = 0, not SHIFTED, and E_M = E_O2. Whether A3 differs is printed beside |

- States 6 and 7 are exclusive. The table is exhaustive: 11 is the remainder.
- **D27 checklist:**
  - **sign flips:** PAIRS-OPPOSE (across pairs), A-SPLIT (across v2 seeds), SEED-SPLIT (across seeds,
    within corpus);
  - **one seed only:** ONE-PAIR;
  - **thin samples:** THIN; VOID on a count mismatch; the aggregate's INSUFFICIENT;
  - **significance (E) without magnitude (Δz):** NEAR-BAR;
  - **magnitude without significance:** SHIFTED;
  - **precedence:** the table.

**Grade tier** (secondary, unguarded, exploratory; user, 17 Nov). Only in cells whose existence state
is AGREES or SHIFTED with all five E = 1; every other cell is G-NA. With g = 1 for R1 and 0 for R2, the
same pattern logic, first match wins:
G-OPPOSE, G-ONE-PAIR, G-SPLIT (A3 differs), G-LOWER (v1 R1 → v2 R2 in both pairs), G-HIGHER,
G-SEED-SPLIT, G-AGREES.
- There is **no magnitude guard.** R1 against R2 turns on several gates whose statistics the record
  does not store.
- Each run's `mechanism` is printed beside.

**Retention under matching (S/P/C/U): reported, no state** (user, 17 Nov). Each run's `retention`,
`retained`, `margin_established` and `competitive` are printed beside every cell, with whether each pair
agrees on `retention`.

### E. Aggregate states and chance counts (per population; the full population is the only confirmatory one)

- **The scoreable set:** non-VOID, non-THIN cells. O is the scoreable cells where **both** pairs flip E
  (c1 ≠ 0 and c2 ≠ 0), and n = |O|.
- **Statistics:** m_down = #O with c1 = c2 = −1, and m_up likewise. Under the paired sign-flip null,
  m_down ~ Binomial(n, 1/4), so p_down = P(X ≥ m_down); likewise for up.
- **α = 0.025 per direction**, one-sided, so 5% over the two directions of the full population. The
  conditional population gets the same table as a secondary reading.
- A3 does not enter the aggregate. That keeps the null exact.

**States: first match wins.**

| # | state | condition |
|---|---|---|
| 1 | **VOID** | a record-level failure (P1, P2 record, P4, P5), or P6: more than 3 of the population's 37 cells VOID |
| 2 | **INSUFFICIENT** | n < 3. The smallest attainable p, 4⁻ⁿ, cannot reach 0.025 below n = 3, so the bar is unreachable. This is an error state, not a null (D28) |
| 3 | **BOTH WAYS** | p_down ≤ 0.025 and p_up ≤ 0.025: the pipelines differ, and no direction is claimed |
| 4 | **CARRIED** (down) / **MASKED** (up) | p ≤ 0.025 in that direction |
| 5 | **WITHIN-CHANCE** | otherwise. Printed with its reach: the smallest m that would have fired at this n |

- **"The count of answers moving R1/R2 → R4"** is m_down in the full population.
- **Run-specific noise, printed beside every overall state** (full population; user, 17 Nov: no gate,
  no change to α, no single "observed" rate). The unit throughout is ω × per-answer run noise (σ), the per-run cell-level
  noise σ of §F's grid.
  - **(a) Tipping points.** For each of CARRIED, MASKED and BOTH WAYS: the ω at which that state's
    false-call rate first reaches 2.5% and 5%. They are read from §F's 16-row ω grid by linear
    interpolation between the bracketing rows: ω = ω_lo + (t − r_lo) / (r_hi − r_lo) × 0.1.
    - If the first row (ω 0) already reaches the threshold, it prints "≤ 0× (already r at 0)".
    - If no row reaches it, it prints "> 1.5×".
    - **Which rate applies:**
      - CARRIED: the false CARRIED rate (per direction);
      - MASKED: the false MASKED rate;
      - BOTH WAYS: its own null rate, the false BOTH WAYS rate. It is zero on every grid row, so it
        prints "> 1.5×".
      - WITHIN-CHANCE and INSUFFICIENT make no claim and carry no false call. "No false call to bound"
        is printed, with the CARRIED and MASKED tipping points beside it for reference only.
      - VOID prints nothing.
  - **(b) The full curve:** all 16 grid rows, ω with its CARRIED, MASKED and BOTH WAYS rates.
  - **(c) An upper bound on ω from A1–A3 alone.**
    - **Computation.** Over the full population's non-VOID cells (n cells), let m_r be run r's mean z,
      for r ∈ {A1, A2, A3}, with grand mean m̄.
      - s_m² = Σ_r (m_r − m̄)² / 2, which has **2 df**.
      - s_int² = Σ_r,c (z_rc − m_r − z̄_c + m̄)² / (2(n − 1)) is the run × cell residual.
      - ω_up² = max(2 s_m² / χ²₂(0.05) − s_int² / n, 0). This is a **one-sided 95%** bound, with
        χ²₂(0.05) = 0.1026.
      - It is printed in grid units, ω_up / σ̂_c, where σ̂_c² = Σ_c (x_c − x̄)² / (2ν_c) / (2(n − 1)),
        with x_c = Δ1 − Δ2. This is the per-run noise from the pairs' difference **centred** across
        cells, so a run-level offset does not inflate the unit.
      - The grid's CARRIED, MASKED and BOTH WAYS rates are printed at ω_up. Beyond 1.5× it prints the
        1.5× row's rate and says "beyond the grid".
    - **What it bounds, and why.** A run's mean over cells carries its run-level offsets: its own ω_r,
      and the run-level part of its seed offset u_seed. Cell-level terms average to s_int²/n. A1–A3
      have three different seeds, so E[s_m²] = ω² + τ_run² + (cell term)/n, and 2s_m²/E[s_m²] ~ χ²₂.
      The bound therefore covers ω² + τ_run² ≥ ω² at 95%.
      - A real corpus effect and the v2 bank's error are common to A1–A3, so they cancel.
      - **The assumption:** run-specific noise is the same size on v1 and v2 runs. The bound sees v2
        runs only. That is the assumption the grid itself makes.
      - With 2 df the bound is wide. It is printed as a bound, not an estimate.
    - **Coverage on synthetic inputs** (§F, and full-path plants in §G): over the 15 ω rows with ω > 0 (20,000 draws each), coverage is 94.8%–95.3%. With u_seed it is 94.9% at ω 0.5σ, τ 1.0σ, λ 0; 99.0% at ω 0.5σ, τ 1.0σ, λ 1; 97.5% at ω 1.0σ, τ 1.0σ, λ 1; 95.0% at ω 0.5σ, τ 2.0σ, λ 0. These are higher where seed offsets have a run-level part, which the bound also absorbs. The median bound is 0.58× at ω 0 and passes 1.5× near ω 0.4σ.
  - **(d) The drift D̄** (the mean over non-VOID cells of Δ1 − Δ2, with SE) is printed only as a number,
    labelled exactly "one draw of the run-level noise; not an estimate of its size". It is not used to
    choose a rate.
    - **Why:** D̄ is one draw of (ω_A1 − ω_M) − (ω_A2 − ω_O2). It is also blind to same-direction
      tilts, which are exactly the ones that make false CARRIED/MASKED calls.
    - **The plants showed it** when it was used as a lookup (draft v2): at a true ω of σ it picked ω 0
      in 24 of 300 runs; at a true ω of 0 it picked ≥ 0.2σ in 100 of 300.
  - **Headline form for any CARRIED or MASKED:** "below 2.5% false calls if run-specific noise < X×
    per-answer run noise (σ); up to Y at 1.5×", where X is (a)'s 2.5% tipping point and Y the curve's rate at 1.5×.

**Chance counts, printed beside the per-cell state counts and before any per-cell list.** Computed
mechanically after P1–P6 pass.
- **Paired symmetry (needs no model).** Under the null, each opposite-flip order is as likely as each
  same-direction pattern. So:
  - unguarded: E[#(c1 = c2 = −1)] = E[#(c1 = c2 = +1)] = N_oppose / 2;
  - guarded, before A3's veto: E[# guarded same-direction] = N_oppose,guarded / 2, where
    N_oppose,guarded counts opposite flips with |Δ1| > F_c and |Δ2| > F_c.
- **Calibrated.** For every state, n_scoreable × the observed-like row's per-cell rate (§F).
- **A-SPLIT** (the A3 veto, kept by the user, 17 Nov) gets its own chance line. The observed-like row's
  A-SPLIT rate under no effect, per answer, and the expected count across 37 (rate × 37) and across the
  scoreable cells. Its range over the grid is in §F.

### F. The null and its calibration (`m_comparison.py calibrate`; synthetic z only)

- **Model** (decision layer, 20,000 draws per row, seed 20261117, one generator per row). For run r in
  cell c:

  z_rc = μ_c + δ_c [r ∈ v2] + u_seed(r),c + ω_r + β_side(r),c + σ_side √ν_c ε_rc

  - **μ_c:** about half the non-borderline cells far above the bar (U(8, 15)), the rest far below
    (U(−2, 1)). A fraction β ∈ {0.1, 0.3} of the cells sits at the BY critical z of that configuration
    ± N(0, 0.3).
  - **u_seed,c:** shared by both members of a pair; A3 has its own. u = τ(√λ a_s + √(1−λ) g_sc), where
    λ is the run-level share.
  - **ω_r:** corpus-specific run-level noise, per run.
  - **β_side:** each bank's mean error, var b_c²/2 per side, with (sd/d)² = 0.5.
  - **ε:** cell-level run noise, with ν_c lognormal (σ_log 0.5).
  - E is computed by BY on t₂₉ p-values, then `decide()`, the function `score()` calls.
- **Grid.**
  - **Base:** σ ∈ {0.5, 1, 2}; τ/σ ∈ {0, 1, 2}; λ ∈ {0, 1}; β ∈ {0.1, 0.3}; equal corpus noise; ω = 0;
    bank term on.
  - **Stress:** v2/v1 noise ratio 2 (τ 0, σ); ω 0.5σ with u_seed (τ σ); noise weights unrelated to ν
    ("mis"); no bank term.
  - **The ω grid, for §E's tipping points and curve:** ω/σ from 0 to 1.5 in steps of 0.1 (16 rows), at
    σ 1, τ 0, β 0.3, equal corpus noise. τ 0 is the worst case, since u_seed lowers every rate. Linear
    interpolation between 0.1 steps is used. The curve is smooth and nearly linear there (see the table).
  - **A-bound coverage rows:** ω 0.5σ and σ with run-level u_seed (τ σ, λ 1), and ω 0.5σ at τ 2σ, λ 0.
    These are appended after the grid.
- **The rule for k (fixed before the sweep):** the smallest k on a 0.05 grid such that, in every base
  regime, with and without u_seed, the per-cell rate of "both pairs beyond F_c in the same direction" is
  ≤ 0.025 for each direction. That is SHIFTED's false-call rate exactly, and an upper bound on
  A-LOWER's and A-HIGHER's.
- **Outcome: k = 1.1** (`docs/m_comparison/calibration.json`, SHA-1 `9c598007f9d96883ca0be231e04fefb16a10e35b`). Over the base regimes:
  - **The magnitude condition:** 2.1–2.5% per direction, the same with u_seed and without it, as it must
    be, since u_seed cancels in Δ.
  - **Per borderline cell:**
    - **A-LOWER / A-HIGHER:** without u_seed, 0.69–0.81%. With u_seed (τ = σ) 0.34–0.45%, and 0.13–0.20%
      at τ = 2σ. Pairing helps: a shared seed offset moves both members of a pair together, so E flips
      less often.
    - **SHIFTED** (both directions, per cell): 2.3–4.5%.
  - **A-SPLIT under no effect** (both directions):
    - per answer 0.50–2.03% without u_seed, an expected 0.18–0.75 of 37;
    - 0.11–1.07% with u_seed, 0.04–0.40 of 37;
    - noise ratio 2: 2.09%, 0.77 of 37;
    - ω grid: 1.65–2.00%, 0.61–0.74 of 37.
    - It is highest where many answers sit at the bar (β 0.3) and seed noise is large.
  - **Aggregate false calls, per direction:**
    - **CARRIED:** ≤ 0.92% without u_seed and ≤ 0.32% with it; **MASKED** ≤ 0.89% and ≤ 0.29%.
      **BOTH WAYS:** 0.
    - The binomial is conservative here. Most null draws are INSUFFICIENT (40–100%): under the null,
      cells where both pairs flip are rare.
  - **Stress:**
    - **Noise ratio 2:** CARRIED 0.83%, MASKED 1.25%.
    - **"mis" weights:** ≤ 0.78%.
    - **No bank term:** 0.73–0.84%.
    - **ω 0.5σ with u_seed (τ σ):** CARRIED 1.00%, MASKED 0.93%.
  - **The ω grid** (σ 1, τ 0, β 0.3), with the A-bound's coverage and median per row:

    | ω/σ | CARRIED | MASKED | BOTH WAYS | A-bound coverage | A-bound median |
    |---|---|---|---|---|---|
    | 0.0 | 0.74% | 0.75% | 0.00% | 100.0% | 0.58× |
    | 0.1 | 0.82% | 0.97% | 0.00% | 95.3% | 0.70× |
    | 0.2 | 1.17% | 1.18% | 0.00% | 95.2% | 0.95× |
    | 0.3 | 1.65% | 1.54% | 0.00% | 95.2% | 1.26× |
    | 0.4 | 2.32% | 2.27% | 0.00% | 94.8% | 1.59× |
    | 0.5 | 3.12% | 2.87% | 0.00% | 95.0% | 1.94× |
    | 0.6 | 3.82% | 3.76% | 0.00% | 95.2% | 2.29× |
    | 0.7 | 4.52% | 4.62% | 0.00% | 95.1% | 2.66× |
    | 0.8 | 5.18% | 5.25% | 0.00% | 95.1% | 2.97× |
    | 0.9 | 5.64% | 5.70% | 0.00% | 95.0% | 3.34× |
    | 1.0 | 6.38% | 6.72% | 0.00% | 95.0% | 3.74× |
    | 1.1 | 6.64% | 6.69% | 0.00% | 95.0% | 4.12× |
    | 1.2 | 7.54% | 7.04% | 0.00% | 94.9% | 4.46× |
    | 1.3 | 7.60% | 7.78% | 0.00% | 95.0% | 4.82× |
    | 1.4 | 7.97% | 7.87% | 0.00% | 94.8% | 5.23× |
    | 1.5 | 8.48% | 8.73% | 0.00% | 94.9% | 5.59× |

    The rate passes 2.5% between ω 0.4σ and 0.5σ. This is the limitation §A.1 declares: measured, not
    bounded.
- **Power, with the A3 veto in** (σ = 1, β = 0.3, 4,000 draws; planted: v1 at the bar + 2, all three v2
  runs moved down by Δz; the aggregate ignores A3, the per-cell state does not):

  | lost | Δz | per-cell A-LOWER, τ 0 / τ σ | vetoed to A-SPLIT, τ 0 / τ σ | CARRIED, τ 0 / τ σ |
  |---|---|---|---|---|
  | 2 | 3 | 54% / 33% | 9.4% / 11.2% | 9% / 4% |
  | 2 | 4 | 84% / 64% | 2.1% / 5.5% | 13% / 8% |
  | 2 | 6 | 95% / 85% | 0.1% / 0.3% | 13% / 11% |
  | 4 | 3 | 56% / 35% | 8.6% / 10.8% | 37% / 25% |
  | 4 | 4 | 84% / 65% | 1.9% / 5.1% | 56% / 50% |
  | 4 | 6 | 95% / 84% | 0.0% / 0.2% | 59% / 65% |
  | 8 | 3 | 60% / 37% | 7.6% / 10.5% | 92% / 75% |
  | 8 | 4 | 86% / 67% | 1.5% / 4.4% | 99% / 96% |
  | 8 | 6 | 95% / 84% | 0.0% / 0.2% | 100% / 99% |

  - **The aggregate** needs about eight answers moved by ~3 z or more. Four answers are found about half
    the time. Two answers can be named per cell but never carry the aggregate.
  - **The veto's cost** is ≤ 11% of truly lost answers at Δz 3, and ≤ 0.3% at Δz 6: under a real effect
    A3 is shifted too, so it rarely dissents.
- **The observed-like row,** fixed now and chosen mechanically after P1–P5 pass, by three keys in turn:
  1. among the base rows, the nearest σ to √(σ̂²/2), on the log scale;
  2. then the nearest τ/σ, from the A seeds' spread: τ/σ = √max(s²_A / (σ̂²/2) − 1, 0), where s²_A is the
     mean over scoreable cells of the A1–A3 z variance / ν_c;
  3. then the nearest rate of cells with any pair flip.

  Ties go to the row with the higher A-LOWER + A-HIGHER rate. That row's rates are printed beside
  every state, as "expected under the null".

### G. Plants (D28; `m_comparison.py plants`; synthetic records only)

- **How they are built.**
  - Synthetic `t1_ladder.json` records carry the full row schema: 37 answers × 2 populations × 5 runs.
  - Two synthetic K = 30 banks share one centre per answer, with the same model and split keys and
    different pixels keys.
  - D23 is computed through the package exactly as the ladder does: `untrained_z_pvalue`,
    `family_significant`, `resolvable_margin` and `is_underpowered`. The rung follows R4 / R1-if-clean /
    R2.
  - The records and banks are written to a temp directory, pinned, and scored through `load_inputs` →
    `score()`, the same path as the real run, so P1–P5 are exercised, not bypassed.
  - **Background:** 22 answers far above the bar and 15 below, with per-run noise 0.15 z. Expectations
    are fixed in the script before it runs.

| plant | construction (full population) | must read |
|---|---|---|
| identical | five copies of one world, no run noise | every cell AGREES in both populations; aggregate INSUFFICIENT |
| lost-8 | 8 strong cells, all three v2 runs at z 0 | those 8 A-LOWER; CARRIED (p = 4⁻⁸) |
| gained-8 | 8 weak cells, all v2 at z 10 | those 8 A-HIGHER; MASKED |
| both | lost-10 and gained-10 | 20 directional cells; BOTH WAYS |
| below-reach | lost-2 | A-LOWER ×2; INSUFFICIENT |
| pairs-oppose | 5 weak cells, M and A2 at z 10 | PAIRS-OPPOSE ×5 |
| one-pair | 5 strong cells, A1 alone at z 0 | ONE-PAIR ↓ ×5 |
| a-split | 5 strong cells, A1 and A2 at z 0, A3 strong | A-SPLIT ↓ ×5 |
| near-bar | 5 cells straddling the BY bar at rank 27, v1 at +0.05 z and v2 at −0.05 z | NEAR-BAR ↓ ×5 |
| shifted | 5 strong cells, v2 +4 z | SHIFTED ↑ ×5 |
| seed-split | 5 weak cells, M and A1 at z 10 | SEED-SPLIT ×5 |
| thin | se 0.03 on M in 3 cells, on A1 and A3 in 3 others | THIN ×6 |
| grade | 5 strong cells, R2 on all v2 runs | G-LOWER ×5 |
| VOID set (each separately) | wrong pin; a missing answer; `k_bar` 2; an `exceeds_null` flipped; bank K 29; bank split key differs; banks share a pixels key; a wrong checkpoint | everything VOID |
| VOID cells | a recorded R3; `n_test` off by 10 on O2 | that one cell VOID |
| VOID allowance | 4 cells VOID in the full population | full aggregate VOID |
| null, no u_seed | 300 realisations (600 populations) from §F's model at σ 1, τ 0, β 0.3, through the full path | CARRIED and MASKED each ≤ 2.5% + 2 SE; SHIFTED per cell ≤ 2 × (2.5% + 2 SE) |
| null, u_seed cell-level | the same at τ = σ, λ = 0 | the same bounds |
| null, u_seed run-level | the same at τ = σ, λ = 1 | the same bounds |
| null, ω 0.5σ and ω σ, and ω 0.5σ with run-level u_seed | the same with corpus-specific run noise (§F's ω rows) | false-call rates reported, no bound (§A.1); the A-bound must cover the true ω in ≥ 95% − 2 SE of runs |
| printing | tipping points on a toy curve (hand values 0.25×, 0.50×), the real grid against the hand formula, "never crosses" ("> 1.5×"), "above at 0", BOTH WAYS on the real grid ("> 1.5×"); and the block with its headline printed by `score()` beside a CARRIED | all exact |
| power | lost-2/4/8 at Δz 4 through the full path | reported |

- **The plant evidence:** `docs/m_comparison/plants.json`, SHA-1 `95416bac703713cdcd54d80f84b32278ca51814c`.
- **The scoring code's SHA-1** is recorded outside the hash ("scoring code of record"), with
  `PREREG_SHA1` and `V1_OUTPUTS_SHA1`, because the file carries both constants. The code that
  produced the plants is the code of record less those constants.

### H. Reading

- **Primary:** the full population's aggregate state.
  - "Some of v1's existence results were carried by the v1 pipeline" is claimed only on CARRIED. The
    cells named are then its A-LOWER cells. Its A-SPLIT ↓ and NEAR-BAR ↓ cells are listed after them, as
    "flipped in both pairs, not guarded or not unanimous".
  - BOTH WAYS, WITHIN-CHANCE and INSUFFICIENT name no answer as carried.
  - **Beside the full population's state:** the run-specific-noise block of §E: the tipping points, the
    curve, the A-bound and the grid rate at it, and the drift as a labelled number. On a CARRIED or
    MASKED, the headline sentence of §E.
  - Beside A-SPLIT: its chance rate per answer and expected count of 37 (§E).
  - Every statement says "the v1 pipeline" (training corpus, probe pixels and untrained bar together),
    never "the misregistration" alone (§A.3).
- **Secondary:**
  - the per-cell table, 74 rows: existence state and direction, grade state, five rungs, five z, Δ1, Δ2,
    F_c, mechanisms, both untrained bars, retention columns. It is read against §E's chance counts;
  - the conditional population's aggregate;
  - the grade tier (exploratory).
- **Reported, no state:**
  - retention and its pair agreement;
  - the untrained bars;
  - the drift D̄ ± SE (full population), labelled as one draw, not an estimate;
  - the A seeds' spread (τ/σ);
  - full → conditional changes per run, side by side.
- **Out of scope:** M on probe_v2 (deferred); E6 pair verdicts, R-, U-series and DD.

### I. Reproduction check against M's original record (does not gate)

- **The record:** M's original `artifacts/out/t1_ladder.json` (pre-ladder-v2 code, v1 split), pinned
  now by `shasum` (digest only): `d71ce4ee858cc35f1531107a1db62b6f6133228f`.
- **When it is read:** in `score --go` only, after the hash, through the same refusal as every real
  record, and after the comparison is scored.
- **It does not gate.** Scoring uses the re-laddered M whatever this says.
- **What is reported:** rung by rung over 74 cells, against the re-laddered M:
  - **PASS** if every rung agrees;
  - **FAIL** with the number of cells whose rung differs (or "answer sets differ");
  - **NOT RUN** if the file no longer matches its pin.
- **A FAIL is expected to be possible.** The ladder code (`ed876a4`, `e130fe1`, …) and the split
  (minus NINE) changed since that record was written. Nothing is read from it but the rungs, and the
  count is given only after scoring.
- **The count stays combined.** It mixes the code changes with the nine-galaxy removal. They are not
  separated because the check does not gate. A separation would need its own rule and inputs, and it
  would decide nothing here.

## Plant and calibration results (synthetic, before the hash; v3 code of record, A3 veto in)

`m_comparison.py plants` → `docs/m_comparison/plants.json` (`55edf975…`, `all_fired: true`, k 1.1,
calibration `9c598007…`). Expectations were fixed in the script before the run. Every value below is
synthetic.

| plant | must read | got (full population) | fired |
|---|---|---|---|
| identical | all AGREES, both populations; INSUFFICIENT | AGREES 37; INSUFFICIENT | yes |
| lost-8 | A-LOWER ×8; CARRIED | as expected | yes |
| gained-8 | A-HIGHER ×8; MASKED | as expected | yes |
| both | A-LOWER ×10, A-HIGHER ×10; BOTH WAYS | as expected | yes |
| below-reach | A-LOWER ×2; INSUFFICIENT | as expected | yes |
| pairs-oppose | PAIRS-OPPOSE ×5 | as expected | yes |
| one-pair | ONE-PAIR ↓ ×5 | as expected | yes |
| a-split (the veto) | A-SPLIT ↓ ×5 | as expected | yes |
| near-bar | NEAR-BAR ↓ ×5 | as expected | yes |
| shifted | SHIFTED ↑ ×5 | as expected | yes |
| seed-split | SEED-SPLIT ×5 | as expected | yes |
| thin | THIN ×6 | as expected | yes |
| grade | G-LOWER ×5 | as expected | yes |
| VOID, everything (8 causes) | everything VOID, naming the cause | as expected | 8/8 |
| VOID, one cell (recorded R3; `n_test` off by 10) | that cell alone | as expected | 2/2 |
| P6: 4 VOID cells | full aggregate VOID | VOID | yes |
| printing: tipping points | toy curve 0.25× and 0.50× (hand); real grid equal to the hand formula at 2.5% and 5% for CARRIED and MASKED; "> 1.5×" for never-crosses and for BOTH WAYS on the real grid; "≤ 0×" above at 0 | all exact | 9/9 |
| printing: through `score()` on a CARRIED | the block with the headline, a 16-row curve, the A-bound (2 df, 95%) and the labelled drift | "below 2.5% false calls if run-specific noise < 0.42× per-answer run noise (σ); up to 8.5% at 1.5×" | yes |

**Nulls through the full path** (600 populations each). The bounds are 3.77% per direction for the
aggregate, 5.42% for SHIFTED per cell, and A-bound coverage ≥ 92.5% (95% − 2 SE at 300 runs).

| null | CARRIED | MASKED | SHIFTED / cell | A-LOWER / cell | A-HIGHER / cell | A-SPLIT / cell | A-bound coverage (median) | within bounds |
|---|---|---|---|---|---|---|---|---|
| no u_seed | 0.83% | 0.67% | 3.01% | 0.22% | 0.21% | 1.67% | — (ω 0; median 0.60×) | yes |
| u_seed cell-level | 0.17% | 0.67% | 2.95% | 0.13% | 0.16% | 0.88% | — (0.86×) | yes |
| u_seed run-level | 0.17% | 0.33% | 2.90% | 0.15% | 0.06% | 0.79% | — (3.52×) | yes |
| ω 0.5σ | 3.83% | 3.50% | 3.11% | 0.32% | 0.15% | 1.86% | 96.0% (1.88×) | rates reported; bound covers |
| ω σ | 4.83% | 8.83% | 4.84% | 0.32% | 0.40% | 1.81% | 97.0% (3.71×) | rates reported; bound covers |
| ω 0.5σ + run-level u_seed | 1.33% | 0.33% | 3.42% | 0.16% | 0.09% | 0.86% | 98.3% (4.23×) | rates reported; bound covers |

- **BOTH WAYS:** 0 in every null.
- **MASKED 8.83% at ω σ** (600 populations, SE ≈ 1.2%) against 6.7% on the 20,000-draw grid is sampling
  noise, within 2 SE, and not an asymmetry.
- **The A-bound with run-level u_seed and ω 0** has a median of 3.5×. It absorbs τ_run, so it is
  conservative there, as §E says.

**Power through the full path, veto in** (σ 1, τ 0, Δz 4, 120 populations). For 2 / 4 / 8 lost answers:
- per planted cell, A-LOWER 84% / 86% / 88%, and vetoed to A-SPLIT 2.1% / 2.1% / 1.7%;
- CARRIED 12% / 48% / 100%.

**Tipping points under no effect** (§F's grid; hand-checkable):

| state, threshold | bracketing rows (ω: rate) | interpolated |
|---|---|---|
| CARRIED 2.5% | 0.4: 2.320%, 0.5: 3.125% | 0.4 + (2.5 − 2.320)/(3.125 − 2.320) × 0.1 = 0.422× |
| CARRIED 5% | 0.7: 4.525%, 0.8: 5.185% | 0.7 + (5 − 4.525)/(5.185 − 4.525) × 0.1 = 0.772× |
| MASKED 2.5% | 0.4: 2.275%, 0.5: 2.870% | 0.4 + (2.5 − 2.275)/(2.870 − 2.275) × 0.1 = 0.438× |
| MASKED 5% | 0.7: 4.625%, 0.8: 5.250% | 0.7 + (5 − 4.625)/(5.250 − 4.625) × 0.1 = 0.760× |
| BOTH WAYS 2.5% and 5% | all rows 0 | > 1.5× |

## Changelog v0 → v1 (outside the hash; kept)

- **Design (user, 17 Nov):**
  - **Primary unit:** the seed-matched pair (M, A1), (O2, A2). It was M against A1–A3. O2 now has a
    ladder record (re-laddered with M).
  - **The null:** the exchangeable-four null and the run-level "most odd" guard (v0's decision 8) are
    replaced by the paired sign-flip null. It is calibrated with a shared u_seed and corpus-specific run
    noise, with rates reported with and without u_seed.
  - **Directional per-cell states need a same-direction move in both pairs beyond the guard.** New
    states: PAIRS-OPPOSE, ONE-PAIR, SEED-SPLIT. A-SPLIT now means "both pairs moved, A3 dissents".
  - **The aggregate:** M-ODD / M-ODD BOTH WAYS / SEED-LIKE / NOT-ODD become CARRIED, MASKED / BOTH
    WAYS / (removed) / WITHIN-CHANCE.
  - **The guard:**
    - it is in units of within-pair noise estimated from Δ1 − Δ2, plus the bank term (v0: A's three-seed
      spread plus 2/K);
    - **k = 1.1** from the stated rule (v0: k_g 1.60, indicative);
    - v0's estimator is gone.
  - **Untrained bars** (D23 bank and D24 C) are reported beside every cell, with no state.
- **Inputs:** M and O2 are re-laddered under the current code on the identical split, with a v1 K = 30
  bank on that split. v0's open decisions 1 and 2 are resolved, and P3/P4 are rewritten: exact count
  equality across all five, and code identity by pinned commits.
- **Corrections to v0:**
  - v0's P2 required `k_bar == 30`. The record's `k_bar` is the D24 draw count (3); the bank's K (30)
    is checked on the bank.
  - v0's P6 (floor and `probe.yaml`) is folded into P5's `clean` check and P4's path list.
  - P2 now also checks the two banks' keys.
- **Decided (user, 17 Nov):**
  - M on probe_v2 deferred;
  - retention has no state;
  - grade tier unguarded, exploratory, secondary;
  - THIN rule as stated;
  - α 0.025 per direction, full population the only confirmatory one;
  - VOID allowance 3 of 37;
  - `shasum` is blind;
  - k from the rule.

## Changelog v1 → v2 (outside the hash; user's decisions on v1, 17 Nov)

- **The A3 veto is kept.** The calibration and plants run the full path with it in.
  - §E adds A-SPLIT's chance line: the rate per answer and the expected count of 37, printed by
    `score()`.
  - §F gives A-SPLIT's null range, and restates power with the veto in, including how often it vetoes
    a true loss.
- **Run-specific noise: no gate and no α change.**
  - The pairs-drift t is replaced by the measured drift D̄ ± SE (full population, non-VOID cells).
  - D̄ selects an ω row (a 16-row grid, ω/σ 0–1.5 in steps of 0.1, added to `calibration.json`). That
    row's CARRIED/MASKED rate is printed beside the overall state, and `score()` implements the lookup.
  - The ω stress rows at τ 0 are folded into the grid.
- **Pins.**
  - M, O2 and the v1 bank: from `docs/ladder_v2_m/outputs.sha1`, whose own SHA-1 is recorded outside
    the hash when the runner finishes.
  - A: from `docs/ladder_v2/outputs.sha1`, as before, and now cross-checked at scoring.
  - P4 now reads the two runners' manifests at their commits (fcea3b2, 17945d9), each pinned by SHA-1.
    It allows exactly two declared and classified code differences. The invented `A_CODE_SHA` /
    `V1_CODE_SHA` / `P4_INERT` constants are gone.
  - The VOID allowance is now numbered P6.
- **The conditional bar:** stated as a caveat under Common ground. The full-population bank is reused
  for conditional, on both sides.
- **§I (new): a reproduction check against M's original record.** It does not gate. It is pinned now
  by `shasum` and reported after scoring as PASS, FAIL with a count, or NOT RUN.
- **The code's SHA-1** is no longer in the hashed text, since the file carries `PREREG_SHA1`. It is
  recorded outside the hash as the scoring code of record.

## Changelog v2 → v3 (outside the hash; the user's final changes, 17 Nov)

- **The drift lookup is removed.** No single "observed" rate is printed. Beside every overall state,
  `score()` now prints:
  - the tipping points (the ω at which CARRIED's, MASKED's and BOTH WAYS's false-call rates reach 2.5%
    and 5%, by linear interpolation on the 16-row grid, with "> 1.5×" and "≤ 0×" for the edges);
  - the full curve;
  - a one-sided 95% upper bound on ω from A1–A3's run-level spread (2 df), with the grid rate at it;
  - the drift, as a labelled number only.
- **Applicability:**
  - BOTH WAYS is read against its own null rate.
  - WITHIN-CHANCE and INSUFFICIENT carry no false call; their CARRIED/MASKED tipping points are shown
    for reference.
  - The CARRIED/MASKED headline sentence is fixed.
- **The A-bound's unit** is a centred per-run noise estimate. With the uncentred σ̂ of the guard, ω
  inflated the unit and coverage fell to 86% at ω 1.5σ (trial run). Centred, coverage holds across the
  grid.
- **§I:** a sentence on why the reproduction count stays combined.
- **Grid spacing:** unchanged. Three A-bound coverage rows are appended after the ω grid, so the grid
  rows' seeds, and their rates, are unchanged.

**v3, last change before the hash (user, 17 Nov).** Wording only: the ω unit is named "per-answer run
noise (σ)" wherever it was called "seed noise" (hashed text, open decisions, the printed headline);
uses of "seed noise" meaning the spread between seeds are unchanged, as is the identifier
`seed_noise_centred`. A sampling-noise note on MASKED at ω σ was added beside the null table. Plants
rerun on the final code: every plant fired; `plants.json` differs from v3's only in the headline text
and run timings (re-pinned `95416bac…`); `calibration.json` byte-identical (`9c598007…`).

## Open decisions for the user

1. **The A-bound rests on run-specific noise being the same size on v1 and v2 runs.** It sees only
   A1–A3. A v1-only bound would need a third v1 seed. If v1 runs could be noisier at run level (for
   example, the misregistration adding run-to-run variability), the bound understates ω for v1.
2. **The A-bound is wide.** With 2 df, its median is about 0.6× per-answer run noise (σ) when ω is truly 0, and it
   passes the grid's end (1.5×) once ω is about 0.4–0.5×. So in practice it says "small" only when
   A1–A3's run means agree closely. It also absorbs run-level seed variance (τ_run), so it is
   conservative whenever seed offsets have a run-level part.
