# Patch-grid pre-registration: flags (24 Nov 2026, second set of decisions applied)

What I found while completing `prereg.md`: things that are ambiguous, and things that don't match the code or
the record. Nothing was reconciled silently: the draft's own text is unchanged except for the completed
[CC] items. Where the scoring code had to choose, the choice is named (`[AMBIGUOUS n]` in
`artifacts/patch_grid_score.py`) and listed here. Nothing is hashed and no rental is booked.

## A. Must be settled before the hash

**1–3. Written (24 Nov).** `configs/patch_grid/` (nine overlays), `artifacts/patch_grid_run.py` (launcher), `objectives/mae_grouped.py`
   (masks and normalises in 16-px groups at every patch size; at patch 16 it gives the Mac MAE's masks and loss exactly, tested), and the scaled
   MoCo windows (10–14, 20–28, 40–56 tokens; a test checks the pixel extent). Nine smokes of three steps each: section F.
   Notes: **`mae_grouped` at patch 16 draws the same masks as the Mac MAE, so the patch-16 reference stays the same recipe;** the launcher does not import
   `baseline_run.py` (it hard-codes the SSD path), but `artifacts/f0_preconditions.check` still reads `GJ_CONFIG`, `F0_CACHE_BASE` and `F0_NORM_PREFIX`, which the rental must set.

**4. "The same sky at every patch size" holds for JEPA only in distribution.** The block scales are fractions of
   the grid, and the RNG draws differ with the grid, so the same galaxy and step do not get the same boxes.
   Measured token counts (ctx / target): p16 80 / 180, p8 329 / 698, p4 1,053 / 2,897. The fraction of sky is right to
   within about 3% at p8, but at p4 the context is 18% below 16 × the p16 count (1,280 against 1,053), while
   the target matches within 1%. If §3 means identical masks, the masker has to be rewritten to sample in pixels.

**5. ⚠ THE DIVERGENCE RULE, KEPT AS DRAFTED (user, 24 Nov), VOIDS EVERY JEPA SEED WE HAVE.** It says a loss above the step-1,000 value
   after step 20,000. Applied to the nine Mac traces (`runs/*/traces.json`):

   | run | step-1,000 loss | steps after 20k above it | fires? |
   |---|---|---|---|
   | A1 | 0.106 | 92.6% | yes |
   | A2 | 0.110 | 99.5% | yes |
   | A3 | 0.066 | 100% | yes |
   | MoCo s0–s2 | 8.87–9.09 | 0% | no |
   | MAE s0–s2 | 0.99–1.00 | 0% | no |

   JEPA's loss starts near 1.3, drops fast, and later rises with the EMA target (A1's 500-step mean peaks
   at 0.258 and ends at 0.114), so a step-1,000 reference has no meaning for it. For MoCo, step 1,000 sits on the
   start-up hump, so the rule could hardly fire at all.
   **I kept it because you said as drafted, and I have not changed it. But applied as written, every JEPA arm of the rental would be VOID, rerun once, VOID again, and the JEPA
   groups scored with nothing: the JEPA half of Q1 and Q2 would be lost. Please read this before the hash.** If you meant the F7 proposal below (which you saw in the flags), say so.
   **Proposal (unchanged):** F7's rule, already used and written down (`artifacts/baselines.md`): a run has diverged if
   its final 500-step mean loss is more than 50% above its lowest 500-step rolling mean, or the loss is non-finite.
   It fires on none of the nine (rises: A1 5%, A2 36%, A3 28%, MoCo and MAE 0%).

**6. The seed spread at 8 and 4 is unknown.** The power in section D applies the patch-16 noise at all sizes, with three values of ρ.

**7. The collapse bar.** The one A1–A3 used is fixed and cited in §7 (`configs/pretrain_v2.yaml`, via `m2_long_run.py:361`; unchanged since `11b9067`, so nothing was invented).
   **One question remains: is it meant for all three objectives?** The code of record applies it whole to JEPA only. F8 settled
   that for MoCo and MAE only the hard floor (2.0) gates; the soft floor (2.5), fitted to JEPA traces, is
   reported but does not gate. If §7 means the same bar for all three, F8 is being reversed. In the nine
   traces the effective rank's lowest value is 7.7 (JEPA, early steps) and at least 19.1 after the grace period, so
   either reading would have passed them all. **As written, JEPA arms use A1–A3's bar and MoCo and MAE use F8's hard floor; reversing F8 would mean changing two objectives.**

**8. The GPU seeds share train_seed with the Mac runs, so the "third and fourth seed's worth" in §8 is not
   new seeds.** GPU seeds 0, 1 and 2 repeat A1's, A2's and A3's init and data order on a different device and precision.
   They add device noise, not independent seed draws, and can't be pooled with A1–A3 as a seed spread.

## B. The evaluation set

**9. Group membership was read off two different probes.** §4 says groups are fixed from A1's B1 results; §4's
   primary is raw, but the groups match the cleaned knees better. Where raw and cleaned disagree
   (`a1_visibility_report.md`, knees cleaned / raw):
   - G2 (knee 2–4): bar 4 / 2, **no bar 4 / 1**, **no spiral 3 / 1**, **ring 4 / 1**, bulge-shape no bulge 3 / 2, others agree.
     In raw, no bar, no spiral and ring would belong to G1.
   - G3 (knee ≥ 5 or none): **other 4 / 6** (cleaned is below 5), **tight 1 / 8**, **dominant 6 / 4**, the rest agree or are
     "none" somewhere.
   Counts: 9 + 11 + 10 = 30, all distinct, and each is STRONG or SEPARATE.

**10. A decile with few of a class.** Dust lane has only 24–34 minority-class test galaxies per decile in deciles
    1–5; the other 29 have at least 111. A decile-level AUC from 24 is rough (the 95% CI is about ± 0.12), and
    L for dust lane may be dominated by it. Its raw L for A1 is −0.024, A1's only negative one.
    [AMBIGUOUS 1]: I count an answer as defined if every decile 1–5 has ≥ 10 of each class.

**11. The probes' train split is the 39,997-galaxy capped one, not the full 160,849.** "The ladder's recipe" used
    both at different points (B1 used 39,997). The frozen split has to be named. The scoring code and the spread
    run use 39,997.

**12. The decile edges and row lists were only computed on the fly in B1.** They are now frozen on disk (section E), so the hash
    can cover them.

**13. Untrained banks: "3 seeds per patch size per objective architecture" reads as 27 banks; the code has one
    architecture.** The encoder is shared by all three objectives, so untrained banks can be shared per patch
    size (9 banks), which is also what the cost document's K = 30 per patch assumes. The K = 30 bank is D23's
    bar, not §5's. Say whether L uses 3 seeds or 30: max-over-seeds grows with K, so the two give different L.

**14. The ordinal-τ secondaries need a Part A ordinal fit per arm,** which is heavy. The scoring
    code for these is not written; they are descriptive.

## C. The scoring code and plants

`artifacts/patch_grid_score.py` implements §5–§6. It reads no image, embedding or checkpoint.

**Choices the draft left open, as coded now:**
1. An answer's decile counts as defined with ≥ 10 of each class; L is defined only if deciles 1–5 all are.
2. An answer counts in a step only if its L is defined at both sizes in every valid seed.
3. Fewer than 2 seeds valid at both sizes = INSUFFICIENT; with exactly 2, K₂ applies and the result is marked.
4. The Mac spread in the device control is the median over answers of (max − min) over A1–A3.
   (The drafted spread-based items 2, 3, 4 and 5 are gone: they belonged to the replaced rule.)
5. Staged design: ρ̂ centres each answer by its own seed mean at each patch size (§2a's "seed mean" read per answer); the gate uses raw L only, over the 30 answers.
6. A ρ̂ that isn't estimable has its own outcome (don't run; recorded), and NOT RUN has its own group readings: both are mine, for D27.
7. **SAME for objective pairs is still the drafted form** (|D| ≤ 0.005 and both step bars ≤ SESOI): decision 1 replaced the NO CHANGE rule only. Change it if you meant both.

**Also open:** §6's group table has no row for GAIN then LOSS or NO CHANGE then UNRESOLVED beyond "WORSE" and "otherwise", which I take as written.

**Plants (D28).** *Synthetic inputs, seed noise SD 0.001:* every step state, every group reading (including WORSE at both
steps and LIMITED BELOW 8 px ONLY), every objective-comparison state and both device states are reached
(`plants.json`). *At the observed seed spread:* see the numbers below.

## D. The paired design (user, 24 Nov): K, plants and power

Nine Mac encoders at patch 16 (A1–A3, MoCo and MAE seeds 0–2), B1's probes and deciles, raw and cleaned (`runs/almagest/patch_grid/spread.json`,
local). The JEPA AUCs match the ladder's record to the digit (edge-on yes 0.765 / 0.837 / 0.776 for A1 / A2 / A3). The earlier two-seed,
spread-based design had to be replaced: its s_g was 0.013–0.047 for JEPA and MAE, so at two seeds Δ80 was 0.070 or not
reached by 0.08, and NO CHANGE was unreachable (git `09346a5`). The reason is seed 1, which runs high for JEPA and MAE (see "The seed-1 effect").

**Noise model.** Per objective and group, from the three seeds' L values: run-level SD τ (encoder-wide shift of the group mean) and answer-level
residual SD σ. **ρ** is how much of a seed's run-level effect is shared between patch sizes (0 none, 1 all). Patch 16 alone can't give it, so every
number below is at ρ = 0, 0.5 and 1. If a seed runs high because of its data order or initialisation in a way that persists across patch sizes, ρ is near
1; if not, near 0. The seed-1 effect appearing in two objectives, for the same seed, suggests data order or a shared seed state, but
MoCo (same seed) doesn't show it. This is not evidence for either end.

**Calibration.** K₃ = 3.2, K₂ = 5.6, and for an objective whose 8 → 4 isn't run (S from one step) K₃ₛ = 6.4, K₂ₛ = 26.05 (1,728 regime cells, 20,000 draws each; half, one and twice the
observed noise; Gaussian and "one seed in three far off" noise; three ρ; the baselines' K is 3.0). False GAIN-or-LOSS per cell ≤ 2.5% by construction;
at the observed noise it is 0–2.4%.

**Plants.** Every step state, every group reading (the five NOT RUN rows included), every objective state and both device states reached (`plants.json`; nothing missing). Scalar and
vector paths agree on 5,400 of 5,400 steps. Under |G| + b ≤ 0.02 and a bar of 0.01 NO CHANGE and GAIN meet at 0.01, so UNRESOLVED can't be reached at low noise; its plants use a
seed noise of SD 0.02. At the observed noise, the largest probability of each reading over the planted cases (`plants_observed.json`, raw):

| reading | JEPA ρ = 0 | JEPA ρ = 1 | MoCo (either) | MAE ρ = 0 | MAE ρ = 1 |
|---|---|---|---|---|---|
| PATCH-LIMITED BELOW 8 px | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| WORSE WITH SMALLER PATCHES | 0.99 | 1.00 | 1.00 | 0.91 | 1.00 |
| PATCH-LIMITED; FLOOR UNRESOLVED | 0.97 | 0.66 | 0.36 / 0.01 | 0.97 | 0.91 |
| PATCH-LIMITED TO A PSF FLOOR | **0.15** | 1.00 | 1.00 | **0.04** | 1.00 |
| DATA-LIMITED AT 16 px | **0.09** | 1.00 | 1.00 | **0.02** | 0.99 |
| LIMITED BELOW 8 px ONLY | **0.14** | 1.00 | 1.00 | **0.04** | 0.99 |

**Power, 3 paired seeds, raw (cleaned in `power.json`, similar).** Δ80 = the group-median Δ that reads GAIN with 80% probability. P(NC | 0) =
the probability that a true Δ of 0 reads NO CHANGE under |G| + b ≤ 0.02. Cells are ρ = 0 / 0.5 / 1. "Two steps" is the full grid; "one step" is 16 → 8 alone (8 → 4 not run).

| | two steps: Δ80 | two steps: P(NC \| 0) | one step: Δ80 | one step: P(NC \| 0) |
|---|---|---|---|---|
| JEPA G1 | 0.148 / 0.107 / 0.025 | 0.00 / 0.00 / 0.43 | 0.256 / 0.179 / 0.041 | 0.00 / 0.00 / 0.23 |
| JEPA G2 | 0.067 / 0.049 / 0.018 | 0.01 / 0.05 / 0.83 | 0.111 / 0.080 / 0.028 | 0.02 / 0.04 / 0.47 |
| JEPA G3 | 0.036 / 0.026 / 0.013 | 0.15 / 0.39 / 1.00 | 0.060 / 0.044 / 0.014 | 0.09 / 0.20 / 0.96 |
| MoCo G1 | 0.017 / 0.014 / 0.013 | 0.86 / 0.98 / 1.00 | 0.028 / 0.022 / 0.014 | 0.49 / 0.70 / 0.97 |
| MoCo G2 | 0.013 / 0.013 / 0.013 | 1.00 / 1.00 / 1.00 | 0.017 / 0.015 / 0.015 | 0.89 / 0.92 / 0.95 |
| MoCo G3 | 0.013 / 0.013 / 0.013 | 1.00 / 1.00 / 1.00 | 0.014 / 0.014 / 0.014 | 0.98 / 0.98 / 0.98 |
| MAE G1 | 0.190 / 0.138 / 0.042 | 0.00 / 0.00 / 0.08 | not reached / 0.237 / 0.068 | 0.00 / 0.00 / 0.07 |
| MAE G2 | 0.075 / 0.054 / 0.014 | 0.01 / 0.03 / 0.99 | 0.126 / 0.090 / 0.019 | 0.01 / 0.03 / 0.77 |
| MAE G3 | 0.051 / 0.037 / 0.014 | 0.04 / 0.14 / 0.99 | 0.085 / 0.061 / 0.019 | 0.04 / 0.09 / 0.79 |

False GAIN + LOSS at 0 is at most 2.4% in every cell.

**Reading.**
- **G1 may read UNRESOLVED for JEPA and MAE at any ρ** (accepted: E is not widened). G1 was predicted DATA-LIMITED, so it is the null we could most want and the one the design sees least well.
- **The G2 answer depends on ρ.** If a seed's effect carries across sizes, JEPA G2 resolves at Δ ≈ 0.018 and NO CHANGE is reached 83% of the time; if it doesn't, 0.067 and 1%.
- **A gate that stops an objective also costs its 16 → 8 step most of its power** (the table's right half): with one step, S has 2 degrees of freedom, and a K of 6.4 is the price. MAE's and JEPA's
  16 → 8 would then be readable only if ρ is near 1, which is the case in which the gate would have passed. So the gate's cost is not only the missing 8 → 4.

### D2. The ρ gate's plants (`plants_gate.json`, `patch_grid_gate.py`)

| | JEPA | MAE | MoCo |
|---|---|---|---|
| true pooled correlation at run-level ρ = 0 / 0.25 / 0.5 / 0.75 / 1 | 0 / .20 / .40 / .60 / .80 | 0 / .19 / .39 / .58 / .78 | 0 / .04 / .07 / .11 / .15 |
| mean ρ̂ at the same ρ | .00 / .15 / .31 / .47 / .67 | .00 / .14 / .30 / .46 / .64 | 0 / .03 / .07 / .10 / .13 |
| SD of ρ̂, ρ = 0 → 1 | 0.48 → 0.24 | 0.47 → 0.26 | 0.16 |
| P(ρ̂ ≥ 0.5) at the same ρ | .20 / .29 / .42 / .57 / .79 | .18 / .28 / .40 / .55 / .76 | 0–.01 (runs regardless) |

- **Independence holds.** Planted Δ ≠ 0 leaves ρ̂ unchanged to 1.7 × 10⁻¹⁶ (6,000 shifts), and corr(ρ̂, mean Δ) is at most 0.02 in absolute value.
- **⚠ Unbiasedness does not hold.** ρ̂ is biased towards 0 (by up to 0.13 at ρ = 1 for JEPA) because answers are few and only two degrees of freedom of seed effect are in it, and its SD is
  0.24–0.48. You asked for a plant showing it unbiased; the plant says it isn't, and I have not altered the estimator. At the planted true correlations, ρ̂ ≥ 0.5 happens 20% of the time at ρ = 0 and
  fails 21–24% of the time at ρ = 1. MoCo has almost no shared run-level effect, so its ρ̂ means little (it runs regardless).
- **Conditional false calls.** Given that the gate passes with a true ρ of 0, the false GAIN-or-LOSS rate at 8 → 4 rises from 2.2 / 2.2 / 1.6% to 4.1 / 4.0 / 2.8% (G1 / G2 / G3), above the 2.5% the bar was
  calibrated to (the gate selects draws in which the noise happened to be small). At ρ = 0.5 it is 3.5 / 2.8 / 1.6%, and 0.7% or less at ρ = 1. Reported; the bar is unchanged.
- The gate's four decisions (including not estimable) are all reached. The path test of `patch_grid_gate.py` on Mac patch-16 records gives ρ̂ = 1 when the second size's seeds equal the first's and
  −0.5 when they are rotated, unmoved by adding 0.07 or −0.05 to one size.

**15. The old equivalence form, for the record.** Decided 24 Nov: |G| + b ≤ E with E = 0.02. The smallest E at which P(NC | 0) ≥ 80%, ρ = 0 / 0.5 / 1: JEPA G1 0.165 / 0.118 / 0.028, G2 0.074 / 0.054 / 0.019,
   G3 0.041 / 0.029 / 0.012; MAE G1 0.211 / 0.152 / 0.047, G2 0.084 / 0.060 / 0.014, G3 0.057 / 0.041 / 0.014; MoCo ≤ 0.019 everywhere. Nothing was widened.

**The seed-1 effect** (parked; not resolved here). Median |L(seed i) − L(seed j)| per group, raw:

| | pair 0–1 | pair 0–2 | pair 1–2 |
|---|---|---|---|
| JEPA G1 / G2 / G3 | 0.053 / 0.016 / 0.012 | 0.006 / 0.005 / 0.005 | 0.046 / 0.031 / 0.017 |
| MAE G1 / G2 / G3 | 0.063 / 0.033 / 0.019 | 0.008 / 0.003 / 0.007 | 0.071 / 0.032 / 0.022 |
| MoCo G1 / G2 / G3 | 0.006 / 0.004 / 0.003 | 0.006 / 0.004 / 0.003 | 0.006 / 0.001 / 0.002 |

- Seed 1 stays in (user, 24 Nov). ρ is the quantity that decides whether it costs us the result.
- **A cheap test before the rental:** can't be done at patch 16, because all nine patch-16 runs exist. The first paired step (JEPA patch 16 vs 8, seeds 0 and 1) on
  the rented box is the earliest estimate of ρ. Once estimated, ρ is recorded, but the rule and its K stay as hashed.

**16. Device control.** The rule compares the GPU–Mac paired difference with the Mac A1–A3 seed spread, a median over answers of 0.027 AUC (0.040 in G1), most of it seed 1.
   Against that, a real device effect smaller than that reads NO DEVICE EFFECT. Not changed; flagged.

**17. Ambiguities in the paired rule, coded as follows.** (a) S is the larger of the two steps' SDs; (b) G is the mean over seeds of the per-seed group medians;
   (c) with one seed VOID, K₂ (or K₂ₛ) applies and the result is marked; (d) the objective comparison uses the paired formula with
   max(S_o1, S_o2) · √(2/n); (e) the untrained bank's 3-seed maximum enters L at both sizes, so its noise is in L at each size and is not separately modelled.

## E. Frozen evaluation set

`docs/patch_grid/frozen_deciles.json` (SHA-1 `f245f70a19d435a6e0b619e194f0a423c3df48d4`): per answer, the decile edges, the class
counts per decile, and the SHA-1 of the train and test row lists, from B1's own rows. Minimum class count in deciles 1–5:
dust lane 24, features 111, no bar 122, smooth 136, dominant bulge 140.

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
- **Pins:** `artifacts/patch_grid_pins.py` checks the files in §3's table; the launcher calls it before any real run (not for smokes or `--plan`).
- The full test suite passes (647).

## For the user, in one place

Decided 24 Nov: paired design, three seeds, raw primary, B2b out of §8; |G| + b ≤ 0.02; the staged rental with the ρ gate; the divergence rule as drafted; the GPU code of record. Still open, none settled by me:

1. **The divergence rule (item 5).** As drafted it voids every JEPA arm. Confirm, or say F7's.
2. **The collapse bar for MoCo and MAE (item 7).** A1–A3's bar whole, or F8's hard floor only (as coded)?
3. **The gate is noisy and biased towards 0 (D2).** Keep it as specified (as coded), or change the threshold? My recommendation is none; the numbers are there.
4. **SAME for objective pairs (C.7)** stays in the drafted form unless you say otherwise.
5. The group memberships that mix raw and cleaned knees (item 9), the untrained bank of 3 seeds or 30 (item 13) and the 39,997 train split (item 11).
6. JEPA's masking is the same sky only in distribution (item 4).
