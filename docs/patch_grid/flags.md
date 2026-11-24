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

**6. The seed spread at 8 and 4 is unknown.** The power in section D applies the patch-16 noise at all sizes, with three values of ρ.

**7. The collapse bar differs by objective in the code of record.** The draft cites one bar. F8 settled
   that for MoCo and MAE only the hard floor (2.0) gates; the soft floor (2.5), fitted to JEPA traces, is
   reported but does not gate. If §7 means the same bar for all three, F8 is being reversed. In the nine
   traces the effective rank's lowest value is 7.7 (JEPA, early steps) and at least 19.1 after the grace period, so
   either reading would have passed them all.

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

**Calibration.** K₃ = 3.2, K₂ = 5.55 (1,080 regime cells, 20,000 draws each; half, one and twice the observed noise; Gaussian and "one seed in three far off" noise;
three ρ; the baselines' K is 3.0). False GAIN-or-LOSS per cell ≤ 2.5% by construction; at the observed noise it is 0–2.4%. Over
the 18 group-and-step cells of one objective family that is at most 0.45 expected false calls.

**Plants.** Low-noise synthetic: all step states, all group readings, all objective states and both device states reached (`plants.json`). Scalar and
vector paths agree on 5,400 of 5,400 steps. At the observed noise, the largest probability with which each reading occurs over the planted cases
(`plants_observed.json`, raw):

| reading | JEPA ρ = 0 | JEPA ρ = 1 | MoCo (either) | MAE ρ = 0 | MAE ρ = 1 |
|---|---|---|---|---|---|
| PATCH-LIMITED BELOW 8 px | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| WORSE WITH SMALLER PATCHES | 0.99 | 1.00 | 1.00 | 0.91 | 1.00 |
| PATCH-LIMITED; FLOOR UNRESOLVED | 0.97 | 0.85 | 0.89 | 0.97 | 0.89 |
| PATCH-LIMITED TO A PSF FLOOR | **0.16** | 1.00 | 1.00 | **0.04** | 0.97 |
| DATA-LIMITED AT 16 px | **0.09** | 0.99 | 1.00 | **0.02** | 0.95 |
| LIMITED BELOW 8 px ONLY | **0.16** | 1.00 | 1.00 | **0.05** | 0.97 |

**Power, 3 paired seeds, raw (cleaned in `power.json`, similar).** Δ80 = the group-median Δ that reads GAIN with 80% probability. P(NC | 0) =
the probability that a true Δ of 0 reads NO CHANGE. Cells are ρ = 0 / 0.5 / 1.

| | Δ80 | P(NC \| 0) | false GAIN + LOSS at 0 |
|---|---|---|---|
| JEPA G1 | 0.148 / 0.107 / 0.025 | 0.00 / 0.00 / 0.45 | 2.4 / 2.4 / 0.9% |
| JEPA G2 | 0.067 / 0.049 / 0.018 | 0.02 / 0.05 / 0.80 | 2.4 / 2.2 / 0.1% |
| JEPA G3 | 0.036 / 0.026 / 0.013 | 0.16 / 0.40 / 1.00 | 1.9 / 1.2 / 0% |
| MoCo G1 | 0.017 / 0.014 / 0.013 | 0.82 / 0.94 / 1.00 | 0.2 / 0 / 0% |
| MoCo G2 | 0.013 / 0.013 / 0.013 | 0.99 / 0.99 / 1.00 | 0% |
| MoCo G3 | 0.013 / 0.013 / 0.013 | 1.00 / 1.00 / 1.00 | 0% |
| MAE G1 | 0.190 / 0.138 / 0.042 | 0.00 / 0.00 / 0.09 | 2.4 / 2.4 / 1.7% |
| MAE G2 | 0.075 / 0.054 / 0.014 | 0.01 / 0.04 / 0.97 | 2.4 / 2.4 / 0% |
| MAE G3 | 0.051 / 0.037 / 0.014 | 0.05 / 0.15 / 0.97 | 2.2 / 2.0 / 0% |

**Reading.**
- **The G2 answer depends on ρ.** If a seed's effect carries across sizes, JEPA G2 is resolved at Δ ≈ 0.018 and its NO CHANGE is reached 80% of the
  time. If it doesn't, JEPA G2 needs 0.067 and NO CHANGE is all but unreachable (2%), as with the old design.
- **The design cannot say which it is before the first runs.** After them, the paired differences' own spread will show: ρ is estimable
  from the GPU seeds' L at two sizes, and S is computed from that, so the bar follows it.
- **JEPA and MAE G1 stay poor even at ρ = 1** (NO CHANGE reached 45% and 9%), because G1's answer-level noise is also large. G1 was predicted DATA-LIMITED, so
  a null for it is the result we could most want, and it is the one we are least able to see.

**15. NO CHANGE is unreachable for JEPA and MAE at ρ = 0, and for G1 at any ρ. The smallest equivalence bounds (nothing loosened).**
   Two ways to loosen, reported only. The bound is E. "Reached" = P(NC | 0) ≥ 80%.
   - *Equivalence form,* |G| + b ≤ E (as the baselines' SAME). The smallest E for 80% (50%), ρ = 0 / 0.5 / 1:

     | | JEPA | MAE | MoCo |
     |---|---|---|---|
     | G1 | 0.165 / 0.118 / 0.028 (0.124 / 0.089 / 0.021) | 0.211 / 0.152 / 0.047 (0.159 / 0.115 / 0.035) | 0.019 / 0.015 / 0.012 |
     | G2 | 0.074 / 0.054 / 0.019 (0.056 / 0.041 / 0.015) | 0.084 / 0.060 / 0.014 (0.063 / 0.045 / 0.012) | 0.013 / 0.013 / 0.012 |
     | G3 | 0.041 / 0.029 / 0.012 (0.031 / 0.022 / 0.011) | 0.057 / 0.041 / 0.014 (0.043 / 0.031 / 0.012) | 0.012 at all ρ |

     At E = SESOI = 0.02 it is reached for JEPA G2 and G3 and MAE G2 and G3 only if ρ is near 1. A ρ = 0 world needs E ≥ 0.041 (JEPA G3) to 0.211 (MAE G1).
   - *Keeping the drafted form* (|G| ≤ NEGLIGIBLE and b ≤ E): E can't carry it past the probability that |G| ≤ 0.005, which is the ceiling whatever E is.
     For JEPA G1 / G2 / G3 that is 14 / 29 / 51% at ρ = 0 and 69 / 86 / 100% at ρ = 1 (MAE 11 / 26 / 38% and 46 / 97 / 97%). So
     80% would need NEGLIGIBLE widened too.

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
   (c) with one seed VOID, K₂ applies and the result is marked; (d) the objective comparison uses the paired formula with
   max(S_o1, S_o2) · √(2/n); (e) the untrained bank's 3-seed maximum enters L at both sizes, so its noise is in L at each size and is not separately modelled.

## E. Frozen evaluation set

`docs/patch_grid/frozen_deciles.json` (SHA-1 `f245f70a19d435a6e0b619e194f0a423c3df48d4`): per answer, the decile edges, the class
counts per decile, and the SHA-1 of the train and test row lists, from B1's own rows. Minimum class count in deciles 1–5:
dust lane 24, features 111, no bar 122, smooth 136, dominant bulge 140.

## For the user, in one place

Decided 24 Nov: paired design, three seeds, raw primary, B2b out of §8. Still open, none of them settled by me:

1. **Equivalence bound (item 15).** Keep the form as it is (NO CHANGE then needs ρ near 1 for JEPA and MAE), or move to |G| + b ≤ E, with which E.
2. **The divergence rule (item 5)** and whether F8's split collapse bar (item 7) holds.
3. **Permission to write and test** the patch-8/4 configs, the GPU launcher, MAE's 16-px grouping and the scaled MoCo windows (items 1–4),
   which must exist before the hash.
4. The group memberships that mix raw and cleaned knees (item 9), the untrained bank (item 13: 3 seeds or 30) and the 39,997 train split (item 11).
5. The "sky per patch" reading of JEPA's masking (item 4).
