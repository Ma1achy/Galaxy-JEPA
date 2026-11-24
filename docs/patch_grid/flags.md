# Patch-grid pre-registration: flags (24 Nov 2026)

What I found while completing `prereg.md`: things that are ambiguous, and things that don't match the code or
the record. Nothing was reconciled silently: the draft's own text is unchanged except for the completed
[CC] items. Where the scoring code had to choose, the choice is named (`[AMBIGUOUS n]` in
`artifacts/patch_grid_score.py`) and listed here. Nothing is hashed and no rental is booked.

## A. Must be settled before the hash

**1. The patch-8 and patch-4 configs and the GPU launcher don't exist.** No config for patch 8 or 4, no bf16
   launcher and no MAE grouping code is in the repo (grep, 24 Nov), so §3's SHA-1 pin covers the code as it
   stands, not the code that will run. Every file in this set must be written, tested and pinned before the
   hash. `autocast` is a config field (null | bf16 | fp16), null in all three configs.

**2. MAE as coded masks per encoder patch, not in 16-px groups,** and normalises its target per encoder patch
   (`objectives/mae.py`, `uniform_mask`, `norm_pix`). §3's grouping and 16-px target are new code. The
   masking is seeded by `cfg.seed + step`; grouped masking changes the draws even at patch 16, so a grouped
   MAE at patch 16 equals the Mac MAE only if the grouping is the identity there (it should be, group = patch).

**3. MoCo's window is a config field in tokens** (`window_min: 10`, `window_max: 14`). At patch 8 and 4 it must be
   20–28 and 40–56 in the configs. This is config work with a test that the pixel extent matches.

**4. "The same sky at every patch size" holds for JEPA only in distribution.** The block scales are fractions of
   the grid, and the RNG draws differ with the grid, so the same galaxy and step do not get the same boxes.
   Measured token counts (ctx / target): p16 80 / 180, p8 329 / 698, p4 1,053 / 2,897. The fraction of sky is right to
   within about 3% at p8, but at p4 the context is 18% below 16 × the p16 count (1,280 against 1,053), while
   the target matches within 1%. If §3 means identical masks, the masker has to be rewritten to sample in pixels.

**5. The divergence rule in §7 would VOID every JEPA seed we have.** It says a loss above the step-1,000 value
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
   **Proposal:** F7's rule, already used and written down (`artifacts/baselines.md`): a run has diverged if
   its final 500-step mean loss is more than 50% above its lowest 500-step rolling mean, or the loss is non-finite.
   It fires on none of the nine (rises: A1 5%, A2 36%, A3 28%, MoCo and MAE 0%).

**6. Two-seed spread at 8 and 4 is unknown.** The power in section C applies the patch-16 seed spread at all sizes.

**7. The collapse bar differs by objective in the code of record.** The draft cites one bar. F8 settled
   that for MoCo and MAE only the hard floor (2.0) gates; the soft floor (2.5), fitted to JEPA traces, is
   reported but does not gate. If §7 means the same bar for all three, F8 is being reversed. In the nine
   traces the effective rank's lowest value is 7.7 (JEPA, early steps) and at least 19.1 after the grace period, so
   either reading would have passed them all.

**8. The seeds at 8 and 4 share train_seed with the Mac runs, so the "third and fourth seed's worth" in §8 is not
   new seeds.** GPU seeds 0 and 1 repeat A1's and A2's init and data order on a different device and precision.
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
    L for dust lane may be dominated by it. Its raw L for A1 is −0.024, the only negative one.
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

**[AMBIGUOUS 1–6], each a choice the draft leaves open:**
1. An answer's decile counts as defined with ≥ 10 of each class; L is defined only if deciles 1–5 all are.
2. With three seeds, the per-answer spread is the mean over the three pairs' |difference|. With two it is
   |L(0) − L(1)|, as drafted.
3. "Pooled across the two patch sizes" = one median over both sizes' lists of per-answer spreads.
4. If a size has one valid seed (the other voided), its spread list is empty; if neither size yields any spread the
   step is INSUFFICIENT. A step with one valid seed on one side is scored and marked.
5. D(o1, o2) is the median over answers of the per-answer Δ difference, not the difference of the medians.
6. The Mac spread in the device control is the median over answers of (max − min) over A1–A3.

**Also open:** "seeds pair by train_seed" changes nothing in Δ, which is a difference of seed means; §6's group table
has no row for GAIN then LOSS or NO CHANGE then UNRESOLVED beyond "WORSE" and "otherwise", which I take as written.

**Plants (D28).** *Synthetic inputs, seed noise SD 0.001:* every step state, every group reading (including WORSE at both
steps and LIMITED BELOW 8 px ONLY), every objective-comparison state and both device states are reached
(`plants.json`). *At the observed seed spread:* see the numbers below.

## D. Power at the observed patch-16 seed spread, and what it says about the plants

Nine Mac encoders at patch 16 (A1–A3, MoCo and MAE seeds 0–2), B1's probes and deciles, raw and cleaned.
`runs/almagest/patch_grid/spread.json` (local), `power.json`, `plants_observed.json`. The JEPA AUCs match the ladder's
record to the digit (edge-on yes: 0.765 / 0.837 / 0.776 for A1 / A2 / A3). The power model applies the patch-16 seed SD, and the share
of it that is common to all answers, at patch 8 and 4. Neither is known for those sizes (flag 6).

**15. Two-seed JEPA and MAE cannot resolve the expected effects, and their NO CHANGE is unreachable.**
   Raw embeddings, 2 seeds. Δ80 is the group-median Δ that reads GAIN with 80% probability.
   P(NC | 0) is the probability that a true Δ of 0 reads NO CHANGE:

   | arm | group | s_g (median, all pairs) | Δ80 | P(NC \| 0) | false GAIN + LOSS at 0 |
   |---|---|---|---|---|---|
   | JEPA | G1 | 0.035 | not reached by 0.08 | 0.00 | 5% |
   | JEPA | G2 | 0.021 | 0.072 | 0.01 | 6% |
   | JEPA | G3 | 0.013 | 0.070 | 0.01 | 11% |
   | MoCo | G1 | 0.006 | 0.0135 | 0.98 | 0% |
   | MoCo | G2 | 0.004 | 0.013 | 1.00 | 0% |
   | MoCo | G3 | 0.003 | 0.013 | 1.00 | 0% |
   | MAE | G1 | 0.047 | not reached | 0.00 | 4% |
   | MAE | G2 | 0.023 | not reached | 0.01 | 7% |
   | MAE | G3 | 0.016 | not reached | 0.01 | 12% |

   - Cleaned embeddings are the same to within 0.015 in Δ80 (JEPA G2 0.063, G3 0.063; G1 0.073).
   - §9's own trigger is met: G2 needs Δ ≈ 0.072 for JEPA, above 0.03, and MAE is not reached by 0.08.
   - **A third seed does not help under the drafted bar.** Three seeds give JEPA 0.078 / 0.070 / 0.061 for G1 / G2 / G3.
     The bar is 2 × the *spread*, not 2 × a standard error, so it does not shrink with more seeds.
   - **It does reach GAIN and LOSS for large effects,** but the states that carry H_psf (NO CHANGE at 8 → 4, so PATCH-LIMITED TO
     A PSF FLOOR and DATA-LIMITED AT 16 px) need s_g ≤ 0.010 and do not occur in practice for JEPA or MAE. Run at the observed spread,
     the planted "no change" reads UNRESOLVED; "gain then floor" reads PATCH-LIMITED; FLOOR UNRESOLVED. SAME between
     objectives is rare (about 0.6% of pair-reads). That is a D28 failure for those states.
   - The false-call rate at Δ = 0 is 4–12% per JEPA or MAE group with two seeds. s_g from two draws is sometimes small by luck,
     and the bar then drops to its floor of 0.01.

**16. Almost all of that spread is seed 1.** The known, parked seed-1 puzzle (`docs/findings_summary.md`): training seed 1 runs high on v2
   for JEPA (A2) and MAE, not for MoCo. Median |L(seed i) − L(seed j)| per group, raw:

   | | pair 0–1 | pair 0–2 | pair 1–2 |
   |---|---|---|---|
   | JEPA G1 / G2 / G3 | 0.053 / 0.016 / 0.012 | 0.006 / 0.005 / 0.005 | 0.046 / 0.031 / 0.017 |
   | MAE G1 / G2 / G3 | 0.063 / 0.033 / 0.019 | 0.008 / 0.003 / 0.007 | 0.071 / 0.032 / 0.022 |
   | MoCo G1 / G2 / G3 | 0.006 / 0.004 / 0.003 | 0.006 / 0.004 / 0.003 | 0.006 / 0.001 / 0.002 |

   - The draft fixes seeds 0 and 1, so s_g is inflated about 2.5–10 × for JEPA and MAE by one seed.
   - Swapping seed 1 for seed 2 would shrink it, but the seeds would then have been chosen after seeing this. Keep all three.
   - The same seed runs high in two objectives, so data order and initial weights (shared within a seed) are suspects. If the
     effect carries across patch sizes, a **paired** Δ (same seed, two sizes) cancels it, and the draft's s_g does not use the pairing.

**17. Proposed changes to the rule, for the user to decide (none applied):**
   - **(a) Bar on the paired Δ.** Seeds pair by train_seed, so take Δ_s per seed and use the spread of Δ_s across seeds, not of L.
     Whether it is smaller depends on the seed effect being the same at every patch size, which patch 16 alone can't tell.
     Plants must cover both cases.
   - **(b) Calibrate the bar** (K × the standard error of the paired mean, as in `baselines_comparison.md`, where K = 3.0 was swept to a
     2.5% false-call rate) instead of 2 × spread. This makes the bar shrink with seeds and fixes the false-call rate.
   - **(c) Three seeds (0, 1, 2)**, all with Mac partners, instead of two. It adds 9 encoders: about 50% more GPU time.
   - **(d) Restate the expected outcome** if the rule stays: JEPA and MAE will read GAIN, LOSS or UNRESOLVED; only MoCo can
     establish NO CHANGE.
   - My recommendation: (a) + (b) + (c), with the null plant at the observed spread re-run before any hash.

## E. Frozen evaluation set

`docs/patch_grid/frozen_deciles.json` (SHA-1 `f245f70a19d435a6e0b619e194f0a423c3df48d4`): per answer, the decile edges, the class
counts per decile, and the SHA-1 of the train and test row lists, from B1's own rows. Minimum class count in deciles 1–5:
dust lane 24, features 111, no bar 122, smooth 136, dominant bulge 140.

## For the user, in one place

1. Raw or cleaned as primary (the draft says raw; nothing in the power differs much).
2. Seed count and bar (flags 15–17).
3. B2b: whether it stays in §8 (no token, so not run).
4. The divergence rule (flag 5) and whether F8's split collapse bar (flag 7) holds.
5. Permission to write and test the new GPU configs, grouped MAE masking and pixel-scaled MoCo windows (flags 1–4) before the hash.
