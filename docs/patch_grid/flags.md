# Patch-grid pre-registration: flags (25 Nov 2026, third set of decisions applied)

What I found while completing `prereg.md`: things that are ambiguous, things that don't match the code or the record, and the power. Nothing was reconciled
silently. Where the scoring code had to choose, the choice is named (`[AMBIGUOUS n]` in `artifacts/patch_grid_score.py`) and listed here.
**Hashed 25 Nov (`docs/patch_grid/prereg.sha1`); no rental is booked.** Numbers below that differ from the 24 Nov version do so because the groups changed (section B).

## A. Settled 25 Nov (user)

1. **Divergence:** F7's rule. §7 says why the drafted rule was wrong (JEPA's loss dips and rises; it would have voided A1–A3). Applied to the nine Mac runs it fires on none (A1 5%, A2 36%, A3 28%,
   MoCo and MAE under 0.2%); the planted rises of 51% and 60% and a non-finite loss fire, a 49% rise and a JEPA-like dip don't (`plants_divergence.json`). It is checked against training traces only (`patch_grid_auc.py valid`).
   **Its margin is narrow for JEPA:** A2's rise was 36% against the 50% line, so a run with a bigger late rise would be voided.
2. **The ρ gate is removed** (code, NOT RUN states and one-step K with it; `plants_gate.json` is left in the folder as the superseded record). The full grid runs. ρ̂ is reported after the final scoring, descriptively, beside its planted bias and SD.
3. **Collapse:** A1–A3's bar for JEPA; F8's hard floor for every objective.
4. **SAME** is |D| + b ≤ E, E = 0.02. Plants re-run; power for pairs is below.
5. **Embedding code** written, pinned and smoked (section F).
6. **Groups** from A1's raw B1 record by a stated rule (section B). 3 untrained seeds per patch size. The ladder's 39,997-galaxy train split.
7. **JEPA's "same sky" is in distribution,** now said in §3 (measured: ctx / target tokens p16 80 / 180, p8 329 / 698, p4 1,053 / 2,897; at p4 the context is 18% below 16 × the p16 count).

**Added before the hash (25 Nov, the user's five additions and what checking them turned up):**

8. **§6 states the expected UNRESOLVED readings,** with the Δ80 tables for Q1 and Q2. They sit beside the predicted readings, not in place of them: JEPA's "PSF FLOOR" prediction for G2 and G3 needs a NO CHANGE that a true null reaches 1–14% of the time at ρ = 0.
9. **Three gates the draft promised were not in code:** the cache-parity refusal, the `outputs.sha1` manifest, and VOID from a halted run. They now are, with refusal tests. `auc` also refuses an embedding file whose own name isn't the arm's.
10. **Two launcher faults, found by writing the checklist:** a smoke monitored two stamps, whose effective rank is 1, so a 2,000-step smoke would have halted itself; and a finished or halted run started again would have trained a step past its stop, or resumed past its halt. Both closed.
11. **The pins cover the whole import closure** (101 files with the lock file), and `patch_grid_pins.py` checks the section's own hash, so the table can't be edited either. Consequence: any later edit to a pinned file on this branch (`almagest_axes.py`, `plate_export.py`, anything in
    `src/`) refuses the launcher and the scoring. **Both must run from a git worktree at the hash commit.** `spread.json` (the Mac record, not in git) is pinned by value in the scoring code.

## B. The evaluation set

**8. The groups.** Rule (§4): raw knee 1 and ≥ 7 clearing deciles → G1; else raw knee ≤ 4 and ≥ 4 clearing → G2; else G3. Reproduced from `a1_visibility_b1.json` by the scoring code.
   - "The same thresholds as B1's reading" is my translation: B1 gave no numeric count thresholds, only a clearing margin of 0.05 and knee ranges. **The counts 7 and 4 are mine**, taken from B1's own
     examples (odd no, obvious bulge). On A1's raw record they never decide a case, so any lower threshold gives the same groups.
   - Moved against the draft: no bar, no spiral, ring (G2 → G1), dominant bulge (G3 → G2). G1 / G2 / G3 = 12 / 9 / 9.
   - A caution: the knee is a point estimate and B1 itself called it fragile. Raw knees of 8 for tight winding and irregular come from a single clearing decile each.

**10. A decile with few of a class.** Dust lane has only 24–34 minority-class test galaxies per decile in deciles
    1–5; the other 29 have at least 111. A decile-level AUC from 24 is rough (the 95% CI is about ± 0.12), and
    L for dust lane may be dominated by it. Its raw L for A1 is −0.024, A1's only negative one.
    [AMBIGUOUS 1]: I count an answer as defined if every decile 1–5 has ≥ 10 of each class.

**14. The ordinal-τ secondaries need a Part A ordinal fit per arm,** which is heavy. The scoring
    code for these is not written; they are descriptive.

## C. The scoring code and plants

`artifacts/patch_grid_score.py` implements §5–§6. It reads no image, embedding or checkpoint. `artifacts/patch_grid_auc.py` takes embeddings to readings.

**Choices the draft left open, as coded now:**
1. An answer's decile counts as defined with ≥ 10 of each class; L is defined only if deciles 1–5 all are.
2. An answer counts in a step only if its L is defined at both sizes in every valid seed.
3. Fewer than 2 seeds valid at both sizes = INSUFFICIENT; with exactly 2, K₂ applies and the result is marked.
4. The Mac spread in the device control is the median over answers of (max − min) over A1–A3.
5. ρ̂ (descriptive) centres each answer by its own seed mean at each patch size, raw, over the 30 answers.
6. In the paired rule: S is the larger of the two steps' SDs; G is the mean over seeds of per-seed group medians; objective pairs use max(S_o1, S_o2) · √(2/n); the 3-seed untrained maximum enters L at both
   sizes, so its noise is in L and is not separately modelled.
7. The group table has no row for, say, GAIN then LOSS beyond WORSE, or NO CHANGE then UNRESOLVED beyond "otherwise"; taken as written.

**Plants (D28).** At low noise (SD 0.001; 0.02 for UNRESOLVED): every step state, every group reading, every objective state (SAME in the new form) and both device states are reached (`plants.json`, nothing missing).
With a bar of 0.01 and E = 0.02, NO CHANGE and GAIN meet at 0.01 (as do SAME and DIFFERS), so UNRESOLVED needs real seed noise and its plants use it. The scalar and vector paths disagree on 0 of 5,400 steps.
`final` on a planted grid built from the Mac's own patch-16 records (+0.05 at patch 8, level at patch 4) reads PATCH-LIMITED TO A PSF FLOOR in every cell, with SAME at every pair sampled; a missing arm is refused, and a named VOID arm is scored from two seeds and marked.

## D. Calibration and power (25 Nov, new groups)

Nine Mac encoders at patch 16 (A1–A3, MoCo and MAE seeds 0–2), B1's probes and deciles, raw and cleaned (`runs/almagest/patch_grid/spread.json`, local).

**Noise model.** Per objective and group, from the three seeds' L values: run-level SD τ (encoder-wide shift of the group mean) and answer-level residual SD σ. **ρ** is how much of a seed's run-level effect is shared between patch
sizes (0 none, 1 all); patch 16 alone can't give it, so every number is at ρ = 0 / 0.5 / 1.

**Calibration.** K₃ = 3.2, K₂ = 5.6, unchanged by the new groups (20,000 draws per regime; half, one and twice the observed noise; Gaussian and "one seed in three far off" noise; three ρ). False GAIN + LOSS at the observed
noise is at most 2.4% in any cell; false DIFFERS at most 0.8%.

**Power, 3 paired seeds, raw (cleaned in `power.json`, similar).** Δ80 = the group-median Δ that reads GAIN with 80% probability. P(NC | 0) = the probability that a true Δ of 0 reads NO CHANGE under |G| + b ≤ 0.02.
Cells are ρ = 0 / 0.5 / 1.

| | Δ80 | P(NC \| 0) | smallest E for P(NC \| 0) = 80% |
|---|---|---|---|
| JEPA G1 | 0.124 / 0.090 / 0.024 | 0.00 / 0.00 / 0.48 | 0.138 / 0.099 / 0.027 |
| JEPA G2 | 0.068 / 0.050 / 0.021 | 0.01 / 0.04 / 0.65 | 0.075 / 0.056 / 0.023 |
| JEPA G3 | 0.037 / 0.027 / 0.013 | 0.14 / 0.36 / 1.00 | 0.041 / 0.030 / 0.013 |
| MoCo G1 | 0.015 / 0.014 / 0.013 | 0.92 / 0.99 / 1.00 | 0.017 / 0.014 / 0.012 |
| MoCo G2 | 0.014 / 0.014 / 0.013 | 0.98 / 1.00 / 1.00 | 0.015 / 0.014 / 0.012 |
| MoCo G3 | 0.013 / 0.013 / 0.013 | 1.00 / 1.00 / 1.00 | 0.012 / 0.012 / 0.012 |
| MAE G1 | 0.164 / 0.117 / 0.035 | 0.00 / 0.00 / 0.16 | 0.180 / 0.130 / 0.039 |
| MAE G2 | 0.076 / 0.055 / 0.014 | 0.01 / 0.03 / 0.98 | 0.085 / 0.061 / 0.015 |
| MAE G3 | 0.047 / 0.034 / 0.014 | 0.06 / 0.18 / 0.99 | 0.052 / 0.038 / 0.014 |

**Objective pairs** (SAME = |D| + b ≤ 0.02). Δ80 = the difference that reads DIFFERS with 80% probability; P(SAME | 0) = a true difference of 0 reading SAME.

| | Δ80 | P(SAME \| 0) |
|---|---|---|
| JEPA–MAE G1 | 0.251 / 0.175 / 0.049 | 0.00 / 0.00 / 0.01 |
| JEPA–MAE G2 | 0.116 / 0.085 / 0.029 | 0.00 / 0.00 / 0.25 |
| JEPA–MAE G3 | 0.069 / 0.050 / 0.018 | 0.00 / 0.01 / 0.84 |
| JEPA–MoCo G1 | 0.174 / 0.122 / 0.033 | 0.00 / 0.00 / 0.21 |
| JEPA–MoCo G2 | 0.094 / 0.068 / 0.029 | 0.00 / 0.01 / 0.32 |
| JEPA–MoCo G3 | 0.051 / 0.037 / 0.015 | 0.05 / 0.14 / 0.96 |
| MAE–MoCo G1 | 0.233 / 0.163 / 0.048 | 0.00 / 0.00 / 0.05 |
| MAE–MoCo G2 | 0.104 / 0.075 / 0.019 | 0.00 / 0.01 / 0.78 |
| MAE–MoCo G3 | 0.064 / 0.047 / 0.018 | 0.02 / 0.06 / 0.86 |

**Reading.**
- **At ρ = 1** JEPA and MAE resolve G3 (Δ80 = 0.013–0.014, NO CHANGE reached 99–100%), and G2 only for MAE (0.014, 98%). **JEPA G2 is weaker under the new groups** (Δ80 = 0.021, NO CHANGE 65%; it was 0.018 and 83%
  before the group change). **JEPA and MAE G1 stay poor** (NO CHANGE 48% and 16%). MoCo is resolved everywhere at any ρ.
- **At ρ = 0** nothing but MoCo is resolved: JEPA's NO CHANGE is reached 0 / 1 / 14% (G1 / G2 / G3) and MAE's 0 / 1 / 6%. A JEPA G2 gain needs 0.068.
- **SAME between objectives is all but unreachable at ρ = 0** (at most 5%), and only G3 is reached at ρ = 1 for every pair (84–96%). Q2's likely readings are UNRESOLVED.
- E is 0.02 as decided. The last column of the first table shows what it would take; nothing was widened.

**What the observed noise reaches** (largest probability of each group reading over the planted cases, `plants_observed.json`, raw):

| reading | JEPA ρ = 0 | JEPA ρ = 1 | MoCo ρ = 0 / 1 | MAE ρ = 0 | MAE ρ = 1 |
|---|---|---|---|---|---|
| PATCH-LIMITED BELOW 8 px | 1.00 | 1.00 | 1.00 / 1.00 | 1.00 | 1.00 |
| WORSE WITH SMALLER PATCHES | 0.99 | 1.00 | 1.00 / 1.00 | 0.94 | 1.00 |
| PATCH-LIMITED; FLOOR UNRESOLVED | 0.97 | 0.65 | 0.27 / 0.02 | 0.97 | 0.84 |
| PATCH-LIMITED TO A PSF FLOOR | **0.14** | 1.00 | 1.00 / 1.00 | **0.05** | 0.99 |
| DATA-LIMITED AT 16 px | **0.09** | 1.00 | 1.00 / 1.00 | **0.03** | 0.98 |
| LIMITED BELOW 8 px ONLY | **0.14** | 1.00 | 1.00 / 1.00 | **0.05** | 0.99 |

So at ρ = 0 the predicted reading for G2 and G3 (PSF FLOOR) came out for JEPA at most 14% of the time over the planted cases.

**ρ̂ (descriptive; `plants_rho.json`).** Unchanged by any planted Δ (largest change 1.7 × 10⁻¹⁶ over 6,000 shifts), correlation with the mean Δ at most 0.02.

| | JEPA | MAE | MoCo |
|---|---|---|---|
| true pooled correlation at run-level ρ = 0 / 0.5 / 1 | 0 / .36 / .72 | 0 / .36 / .73 | 0 / .10 / .20 |
| mean ρ̂ | .00 / .27 / .59 | .00 / .28 / .60 | .00 / .09 / .17 |
| SD of ρ̂ | .43 / .40 / .25 | .44 / .40 / .26 | .17 / .17 / .18 |

It is biased towards 0 (by up to 0.13) and its SD on three seeds is 0.25–0.44 for JEPA and MAE, so one ρ̂ says little. The seed-1 effect (new groups; median |L(i) − L(j)|, raw): JEPA pair 0–1 0.048 / 0.016 / 0.012,
0–2 0.005 / 0.004 / 0.006, 1–2 0.043 / 0.031 / 0.018 (G1 / G2 / G3); MAE 0.051 / 0.031 / 0.018, 0.007 / 0.004 / 0.007, 0.065 / 0.032 / 0.020; MoCo ≤ 0.006. Seed 1 stays in.

**Device control.** The GPU–Mac paired difference is compared with the Mac A1–A3 seed spread, a median over answers of 0.027 AUC (G1 0.039, G2 0.027, G3 0.022), most of it seed 1. A real device effect smaller than
that reads NO DEVICE EFFECT. Not changed; flagged.

## E. Frozen evaluation set

`docs/patch_grid/frozen_deciles.json` (SHA-1 `f245f70a19d435a6e0b619e194f0a423c3df48d4`): per answer, the decile edges, the class
counts per decile, and the SHA-1 of the train and test row lists, from B1's own rows. Minimum class count in deciles 1–5:
dust lane 24, features 111, no bar 122, smooth 136, dominant bulge 140. `docs/patch_grid/union_ids.npy` (the 74,825 ids in ladder order, SHA-1 pinned) fixes which galaxies are embedded.

## F. The GPU code of record: tests and smokes

**Smokes (24 Nov): all nine arms, seed 0, 3 steps, batch 2, bf16, CPU with 2 threads, on the loaded Mac.** Plumbing only: no quality reading, and no timing that says anything about a GPU.

| arm | grid | last loss (step 3) | CPU s for 3 steps |
|---|---|---|---|
| jepa p16 / p8 / p4 | 16 / 32 / 64 | 1.294 / 1.266 / 1.356 | 26 / 103 / 842 |
| moco p16 / p8 / p4 | 16 / 32 / 64 | 6.272 / 6.677 / 6.476 | 26 / 166 / 666 |
| mae (grouped) p16 / p8 / p4 | 16 / 32 / 64 | 1.756 / 2.516 / 2.160 | 60 / 291 / 1,616 |

- **The smokes found a fault that would have killed every MAE arm at step 0.** `MaeDecoder` scattered bf16 tokens into an fp32 mask-token tensor under autocast, which no run had exercised.
  Fixed with one cast in `objectives/mae.py` (a no-op in fp32, so the Mac MAE runs' arithmetic stands), with a bf16 test at all three patch sizes. All three MAE arms then passed.
- **One smoke was refused at first:** `jepa_p16` met a checkpoint left by an earlier MPS smoke under a different config hash, and the resume guard stopped it, as it should. Rerun in a clean folder.
- **Not smoked: CUDA, flash attention, batch 32, the loader workers.** MoCo p4 on MPS pushed swap up by about 8 GB, so patch-4 smokes moved to the CPU. The 200-step timed smoke per arm on the rented box
  (§7) is the first real test of those.
- **Mask check:** 64 steps at grid 64 built and 1,024 galaxies compared with `MultiBlockMasker.sample`: 0 batches differ. The rental repeats the check on the full build for each seed.
- **Embedding code** (`artifacts/patch_grid_embed.py`), smoked on the CPU: trained and untrained, patch 16, 8 and 4 (JEPA, MoCo and MAE encoders from the smokes and the Mac). On 64 union galaxies it reproduces the
  ladder's banked vectors: A2 trained, largest difference 1.1e-5 (mean |x| 2.5); MoCo s0, 2.1e-6; untrained seed 1, 9.5e-7. It refuses a patch-16 checkpoint offered as a patch-8 arm. It does not yet run on CUDA.
  **fp32 and no autocast is my choice** (the ladder's recipe); at patch 4 it costs about 18 minutes per encoder on an H100 and an hour on an A100 (cost doc).
- **`patch_grid_auc.py auc` on A2's banked embedding** reproduces `spread.json` exactly, raw and cleaned, per decile and full sample (0.0 over 30 answers). It needs this Mac's ladder data (A1's encoder and the label sidecars),
  so the 36 embedding files (about 4 GB) come home and the probes run here.
- **Untrained banks at patch 8 and 4 exist only as code.** They are made on the box by the same program, from seeds 0–2, with the arm's own encoder constructor.
- **Pins:** `artifacts/patch_grid_pins.py` checks the files in §3's table; the launcher and the embedding program call it before any real run (not for smokes or `--plan`).
- The full test suite passes (647).

## For the user, in one place

All seven items of 25 Nov are applied. What is left to decide or to know before the hash:

1. **Power at ρ = 0 (D).** Nothing but MoCo is resolved. The design answers Q1 for JEPA and MAE only if a seed's effect carries across patch sizes; whether it does is unknown until the runs exist. This is a limit you should accept knowingly.
2. **My choices in the group rule (B.8):** the clearing counts 7 and 4. They decide no case.
3. **fp32 embeddings (F):** they cost 3–8 H100-hours in all. Say if you would rather have bf16; it would no longer be the ladder's recipe.
4. **F7's 50% line is close to A2's 36%** (A.1). A JEPA arm at patch 8 or 4 that rises by more than half is VOID and rerun once.
5. The ordinal-τ secondaries (item 14) and the relabel2 secondaries are named in §8 and have no code here; they are descriptive.
6. The 200-step timed smoke is the first test of CUDA, flash attention, batch 32 and the loader workers.

## G. The stage 1 checklist (outside the hash): where its literal form may stop a sound run

Written as approved, two-sided, nothing loosened (`artifacts/patch_grid_stage1.py`; planted failures reach every item). Three places it may stop the run for a reason other than a fault; the call then is the user's.
1. **Utilisation > 85% at patch 16 and 8.** The cost doc says those arms are step-floor-bound on an H100 (25–40 ms a step of loader and kernel launches), so the GPU may idle even with a healthy loader.
2. **± 30% is two-sided.** An arm faster, or smaller in memory, than the doc by more than 30% also fails. The hours are read against the doc's range widened by 30%; memory against its point. Patch 16's memory isn't in the doc: it is the doc's own model at a quarter of
   patch 8's tokens (2.5 / 2.5 / 3.4 GB for JEPA / MoCo / MAE), and patch 16's MoCo and MAE hours take the patch-8 figures, as the doc's third-seed table does.
3. **"Falling" is mean(steps 1,501–2,000) < mean(1–500).** A monotone reading fails every JEPA and MoCo run on the Mac.
