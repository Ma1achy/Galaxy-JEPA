# Baselines against their paired JEPA seeds — pre-registration (DRAFT, not hashed)

**Question.** D12's Rung-3 control: with the data, backbone, exposure, initial weights and data
order held fixed, does the JEPA objective give a representation from which GZ2's morphology is
read better, the same, or worse than MoCo's and MAE's?

**Status.** Draft v1, 2026-10-23 (`baselines.md` §10). Nothing hashed, nothing read. The plants
ran before any hash (D28) and are recorded below. The user reviews the draft and the plant results
and decides §D and §E. Only then is the section from "## Pre-registration" through its last content
line hashed (SHA-1, trailing newline). The hash comes **before any trajectory probe file, bank-derived
score or ladder output of a baseline is read**.

**What was read to write this, and what was not.**
- **Read:** method text only: `baselines.md` §1, §6, §10; `DECISIONS.md` D12, D27, D28; the method
  sections of `aligned_comparison.md` (Common ground, criterion 2's design; its lines 133–141 and
  every Results, Plant-results and A3 section were skipped); the preamble and Common ground of
  `distractor_sensitivity.md`; the probe drivers' code; the ladder runner's per-entry wall-clock
  (`docs/ladder_v2/state.json`, metadata).
- **Not read:** any trajectory probe file (only their SHA-1s, `baselines.md` §6.1), any bank, the
  ladder's report or outputs, the aligned comparison's results, any `*_findings.md`.
- **One slip, declared.** `distractor_sensitivity.md` line 72, inside a method bullet, quotes five of
  M's N1 nuisance AUCs on v1. They were seen. They are not used here and are not repeated. Nothing in
  this draft depends on them: §E's proposal drops the variables' v1 numbers altogether.

## Pre-registration

### Common ground

- **Encoders, in pairs (`baselines.md` §1, Seeds).** Within a pair, initial weights and stamp order
  are identical, so the two encoders differ only in the objective.

  | pair | JEPA | MoCo | MAE |
  |---|---|---|---|
  | seed 0 | A1 `runs/m_v2` (`encoder.pt` `34bb30a7…`) | `runs/moco_s0` (`bf0f0f60…`) | `runs/mae_s0` (`fb3d03d1…`) |
  | seed 1 | A2 `runs/m_v2_s1` (`380cdb7b…`) | `runs/moco_s1` (`f5e2a9f6…`) | `runs/mae_s1` (`f375c9e5…`) |

  A3 (seed 2) has no baseline partner and is not used.
- **Epoch points:** steps 12,664, 25,327, 50,654 and 101,308 (0.5, 1, 2, 4 epochs of the 810,491
  train split). The baselines have checkpoints at exactly these steps. **A1 and A2 do not:** they kept
  checkpoints every 2,000 steps plus the final one.
- **Precondition P1: the A-side checkpoints are made by exact resume, and the resume is shown identical
  first.**
  - **Identity check:** resume A1 from `ckpt_000012000.pt` (`858942a4…`) under A1's recipe and launch
    environment to step 14,000. Every tensor (model, EMA, optimiser state) must equal the existing
    `ckpt_000014000.pt` (`e08c1e0a…`) bit for bit; the same for A2.
  - **Then:** resume from 12,000 → 12,664, 24,000 → 25,327 and 50,000 → 50,654, per seed, into scratch
    run directories. A1's and A2's own directories are not written. Step 101,308 uses the existing
    final checkpoints (`29a07f0b…`, `1b1013d3…`).
  - **If the identity check fails:** the 0.5, 1 and 2-epoch A-side points are VOID, and with them the
    secondary readout. The primary (4 epochs) is unaffected. No nearest-checkpoint substitute is
    used.
- **Precondition P2: the A side is probed exactly as the baselines were.** `baseline_trajectory_probe.py`
  (E2's protocol on ladder v2's identical split) runs on each A-side checkpoint. It writes a bank per
  point and `<tag>_trajectory.json`, with tags `a1_traj1…4` and `a2_traj1…4`. The probe files are not
  read before scoring.
- **Precondition P3: integrity.** Every baseline trajectory file and bank matches its SHA-1
  (`baselines.md` §6.1 for the probe files; banks hashed when P2 completes). Scoring refits the probes
  from the banks (`aligned_c2.fit_scores`, the call the probe made). Every refitted held-out AUC must
  equal the trajectory file's `auc` for that encoder, answer and step **exactly**. A mismatch stops
  scoring, with the state VOID for that epoch.
- **Split and read-out.** The identical split: 160,849 probe-train and 34,828 probe-test galaxies
  (`baselines_answer_set.json` records the ids' SHA-1). Block 11, mean-pooled; `_fit` logistic
  regression, C from `probe.yaml`; AUC on each answer's eligible test galaxies.
- **Bootstrap.** 2,000 Poisson galaxy draws (`aligned_c2.weights`, seed 20260926), one weight matrix
  shared by all six encoders, all answers and all four epochs, so every difference is paired by galaxy.

### A. The answer set: a rule on label counts only

- **Rule** (fixed in `artifacts/baselines_answer_set.py` before any count was printed): a Scheme-1
  answer is in the set if, on the identical split's probe-test galaxies,
  - its minority class has **≥ 200 galaxies**, and
  - the Hanley–McNeil standard error of an AUC of 0.75 at its class counts is **≤ 0.02**,

  and both classes occur in probe-train. The rule reads the labels and the split only; no encoder,
  bank or probe output is opened (`artifacts/out/baselines_answer_set.json`).
- **Result: 35 of the 37 answers.**
  - **Out:** `t01_smooth_or_features_a03_star_or_artifact` (test minority 28) and
    `t11_arms_number_a34_4` (161; SE 0.022).
  - **In:** every other Scheme-1 answer. Test minorities run from 415 (`t11_arms_number_a36_more_than_4`)
    to 16,021 (`t07_rounded_a17_in_between`).
- The count-based SE also sets the per-answer **weights** w_j = SE_j² / mean(SE²) in §C. These tell
  the per-answer bar how noisy each answer's AUC is expected to be, from its counts alone.
- No list derived from any encoder's results (in particular not `powered`) enters anywhere.

### B. The comparison

- **Per objective X ∈ {MoCo, MAE}, per epoch point, four encoders:** A1, A2, X0 and X1, scored on the
  35 answers.
  - **Paired difference per answer and seed pair:** d_sj = AUC(X_s, j) − AUC(A_{s+1}, j).
  - **Per-answer effect:** D_j = (d_0j + d_1j) / 2.
  - **Family effect:** D̄ = mean_j D_j.
- **Primary readout: 4 epochs (step 101,308),** the stop every arm shares. One family state per
  objective (§C), so two primary states: MoCo against JEPA and MAE against JEPA.
- **Secondary readouts.**
  - The per-answer states at 4 epochs (§C).
  - **The trajectory:** the family state at 0.5, 1, 2 and 4 epochs, read into one trajectory state per
    objective (§C).
- **Reported, no state.**
  - Every per-answer D_j with its CI, at every epoch.
  - MoCo against MAE: the same D_j and D̄ with X = MAE and the "JEPA" slot taken by MoCo.
  - The run-level and answer-level seed spreads of every arm (§C).

### C. Seed-noise bars and states

**Why not the aligned comparison's bar.** With two seeds per arm, a bar built from per-answer seed
differences (mean_j |A1_j − A2_j|) false-calls badly once part of the seed noise is **run-level**: one
run better or worse on every answer. In this design a uniform advantage of X over A is exactly a
run-level shift. Before any data, the decision-layer Monte Carlo (§F) put that bar's false-call rate
at 22–34% under purely run-level seed noise, and 10–14% when the two kinds are equal. That fails the
D28 bound, so the bar here estimates the run-level spread directly.

**Estimators** (per objective and epoch; P = point AUCs on the answers scoreable for all four
encoders):

- **Run-level seed spread** of each arm: r_A = mean_j (A1_j − A2_j) and r_X = mean_j (X0_j − X1_j).
  Answer-level noise averages out over 35 answers. **s_run = max(|r_A|, |r_X|).**
- **Answer-level residual spread**, pooled over answers and scaled by each answer's count weight:
  σ̂²_A = mean_j [((A1_j − A2_j) − r_A)² / w_j] / 2 × J/(J−1), and likewise σ̂²_X.
  Then ŝ_j = √(w_j (σ̂²_A + σ̂²_X) / 2), the answer-level SD of D_j.
- **Family bar:** b = K × s_run, with **K = 3.5**.
- **Per-answer bar:** b_j = √(b² + (Z ŝ_j)²), with **Z = 2.24**.
- **CIs** from the shared galaxy bootstrap of D̄ (and of D_j): **97.5%** for the family (95%,
  Bonferroni-split over the two objectives) and **95%** per answer.
- **Constants:** NEGLIGIBLE = 0.005 AUC; SESOI (smallest effect of interest) = 0.02 AUC.

**When the baseline's and the JEPA's spreads differ.** Both bars take the larger arm's run-level
spread and the mean of the two arms' answer-level variances. The noise of D̄ is driven by both arms,
so taking the larger spread keeps the bar from being set by the quieter arm. Calibrated at a 3× ratio,
the family false-call rate is 3.8% (§F). If either the run-level or the answer-level spread differs
between the arms by more than 2×, the flag **SPREADS DIFFER** is reported with both ratios beside the
state. It does not change the state.

**Family states: first match wins.**

| # | state | condition |
|---|---|---|
| 1 | **VOID** | P1–P3 failed for this epoch, or any of the four encoders has fewer than 30 of the 35 answers scoreable |
| 2 | **SPLIT** | the two seed pairs' family differences D̄_0 = mean_j d_0j and D̄_1 = mean_j d_1j have opposite signs, each of size ≥ SESOI/2 (0.01): the objective's effect reverses with the seed |
| 3 | **BETTER** | CI(D̄) lies wholly above +b, and D̄ ≥ NEGLIGIBLE |
| 4 | **WORSE** | CI(D̄) lies wholly below −b, and D̄ ≤ −NEGLIGIBLE |
| 5 | **NEGLIGIBLE** | CI(D̄) lies wholly beyond ±b, but \|D̄\| < NEGLIGIBLE: significant against seed noise, too small to matter |
| 6 | **SAME** | CI(D̄) lies within [−b, +b], **and b ≤ SESOI**, so an effect of 0.02 would have shown. If the CI excludes 0, it is labelled "detectable but below seed noise" |
| 7 | **UNRESOLVED** | everything else, labelled with why: **underpowered** (b > SESOI, so SAME is unreachable whatever the CI); **large but not resolved** (\|D̄\| ≥ SESOI but the CI crosses the bar); or **CI straddles the bar** |

"X better than JEPA" is BETTER for that objective; WORSE means the JEPA is better. The sign is fixed:
D is baseline minus JEPA.

**D27 checklist.**
- **Sign flips:** a reversal between seed pairs is SPLIT, and either direction is a named state.
- **Insufficient sample:** VOID below 30 answers, and the answer-set rule gates every answer by counts.
- **Significance against magnitude:** NEGLIGIBLE covers significant but tiny; UNRESOLVED (large but not
  resolved) covers large but not significant; SAME requires a bar that could have seen 0.02.
- **Precedence:** the ordered table.

**Per-answer states (secondary, 4 epochs).** The same table on D_j, using b_j and the 95% CI. SPLIT
uses the answer's own pair differences d_0j and d_1j. VOID applies if any of the four encoders lacks
the answer. **No state is pooled across answers.** Under the null, the calibrated per-answer
false-call rate is ≤ 2.5% (§F), so about 0.9 of 35 answers are expected to read BETTER or WORSE by
chance, per objective. Every per-answer reading is reported against that number.

**Trajectory states (secondary).** The family states at 0.5, 1, 2 and 4 epochs give one trajectory
state per objective. First match wins:

| # | state | condition |
|---|---|---|
| 1 | **VOID** | any epoch is VOID (including a P1 failure) |
| 2 | **CROSSES** | BETTER at one epoch and WORSE at another |
| 3 | **CONSISTENT** | the same state at all four |
| 4 | **DIVERGES** | BETTER or WORSE at 4 epochs, and neither at 0.5 or 1 |
| 5 | **CONVERGES** | neither BETTER nor WORSE at 4 epochs, but one of them at 0.5 or 1 |
| 6 | **MIXED** | anything else, with the four states listed |

The probes at earlier epochs are trajectories only. No checkpoint is chosen by them (spec 1C;
`baselines.md` §1.1).

### D. Ladder experiments on the four baselines: proposal, for the user's decision

D12 asks for "the same probe ladder across JEPA, MAE, contrastive". Ladder v2 ran it on A1–A3. Below
are measured costs per encoder, A2's wall-clock from the runner (`docs/ladder_v2/state.json`), to be
multiplied by four.

| option | experiments | per encoder | × 4 baselines |
|---|---|---|---|
| **(a) existence (recommended)** | E3 matched evaluation, then E5 the existence ladder (Scheme 1, D23 untrained bar, BY over 37, effect floor, D24 nuisance clearance, R4 MLP sweep), with E6 entanglement inside it | 0.6 + 11.5 ≈ **12 h** | **≈ 48 h** |
| (b) existence + geometry | (a) + E7 (r0–r3), E11 U2, E12 U3 Scheme 2, E15 V3, E20 AA1 | ≈ 18 h | ≈ 73 h |
| (c) full ladder | everything ladder v2 ran on A2, DD's SAEs (E24) included | ≈ 25 h | ≈ 100 h |

- The shared entries carry over: the v2 untrained bank (E4) and E7's untrained seeds bank. The
  untrained encoders share the baselines' architecture and pixels.
- **Plumbing (about half a day):** the runner takes `LADDER_ENCODER` = each baseline's checkpoint,
  re-exported through `frozen_from`, as the trajectory probe does, with outputs under
  `runs/ladder_v2/<experiment>/<moco_s0|…>/`.
- **Recommendation: (a).**
  - Existence is the rung D12 attributes to objective or images.
  - (b)'s geometry and (c)'s SAEs are interpretability of each encoder, not the cross-objective
    attribution.
- **How it would be read, if added (also to decide):** each baseline's own ladder states, beside
  A1/A2's with an agreement column, with no new comparative state. A comparative state, such as
  "existence survives in JEPA and not MoCo", would need its own D27 table and plants before this hash.

### E. The distractor analysis: proposal, revise and fold in

The v1 draft (`distractor_sensitivity.md`, unhashed) was written for v1 baselines against M/O2. Three
of its premises no longer hold:
- the baselines train on aligned v2 stamps;
- the JEPA pair is A1/A2;
- the four band-offset variables are gone from the data, because v2's stamps carry no misregistration
  and the baselines never saw any.

**Proposal: fold a revised criterion into this pre-registration as §N,** one hash and one scoring run:
- **Variables:** PSF (`psfWidth_r`), brightness (`modelMag_r`) and sky (`sky_r`), each binarised at
  its split's own median, as the v1 draft did. `snr_r`, redshift and size are reported with no state.
  The four offsets are dropped.
- **Galaxies:** `sky_r` was pulled only for v1's 40,000-capped train and the test split. So §N uses
  the 40,000 stride (a subset of the identical split's train set) and the whole test split. This
  needs no new pull. A pull for the other ~121,000 train galaxies (public SkyServer, about 1,200
  calls) is the alternative.
- **Encoders and statistic:** the same six banks at 4 epochs. Per variable v,
  D_v = mean of the two paired differences, with the same states as the family table. The bar is
  b_v = K × max(|A1_v − A2_v|, |X0_v − X1_v|): with only three variables there is no answer-level
  pool, and this is exactly the run-level structure K = 3.5 was calibrated on.
- **The v1 draft's criterion 2** (the nuisance share of the leading variance): reported with no state
  in this round, to keep the hash small.
- **Plants:** as §F, with synthetic nuisance labels. None of the v1 plants is reused, because they
  ran on M and O2.

If the user prefers, the alternative is to keep `distractor_sensitivity.md` as its own document,
revised along the same lines and hashed separately.

### F. Plants (D28): run before the hash, using no real result

**Decision-layer calibration** (`baselines_comparison.py calibrate` → `artifacts/out/baselines_calibration.json`;
60,000 draws per row; seed 20261023; the 35 answers' count weights). Synthetic AUC tables are built
as a run-level offset c ~ N(0, τ) per run plus answer-level noise ~ N(0, σ √w_j) (or lognormal σ_j,
deliberately unrelated to the counts), plus a planted Δ on X. They are fed to the §C estimators. This
fixed K and Z.

| τ (run) | σ (answer) | X/A spread | answer noise | family false calls | per-answer false calls | family power at Δ 0.01 / 0.02 / 0.03 | median bar |
|---|---|---|---|---|---|---|---|
| 0.006 | 0.006 | 1× | ∝ SE | 2.5% | 1.1% | 9% / 25% / 46% | 0.032 |
| 0.006 | 0 | 1× | — | 2.5% | 2.5% | 9% / 26% / 47% | 0.031 |
| 0 | 0.006 | 1× | ∝ SE | 2.5% | 1.3% | 90% / 100% / 100% | 0.005 |
| 0.002 | 0.008 | 1× | ∝ SE | 2.5% | 0.9% | 36% / 81% / 97% | 0.013 |
| 0.004 | 0.006 | 1× | lognormal | 2.4% | 1.6% | 15% / 44% / 71% | 0.022 |
| 0.004 | 0.004 | 3× | ∝ SE | 3.8% | 1.7% | 8% / 19% / 32% | 0.043 |
| 0.002 | 0.004 | 1× | ∝ SE | 2.5% | 0.8% | 44% / 88% / 99% | 0.011 |
| 0.001 | 0.003 | 1× | ∝ SE | 2.5% | 0.8% | 85% / 100% / 100% | 0.006 |

- **False calls** stay at 2.4–2.5% for the family in every regime with equal spreads, and at 3.8% when
  the baseline is 3× noisier. Per answer, they are 0.8–2.5%. The bound is **≤ 5% per objective**,
  and it holds.
- **Power is the honest limit of two seeds.** It depends almost entirely on the run-level spread τ,
  which only the real runs will show. At τ ≈ 0.006 AUC, a 0.02 effect is found a quarter of the time.
  The rule does not hide this: when the bar exceeds 0.02, SAME is unreachable and the state is
  UNRESOLVED (underpowered).
- A third seed pair is the only way to buy power. A3 + MoCo/MAE seed 2 would cost about 50 h of
  baseline training. It is not proposed now.

**Full-path plants** (`baselines_comparison.py plants` → `artifacts/out/baselines_plants.json`).
- **How they are built.** Synthetic per-galaxy scores are generated on the real labels of each
  answer's eligible test galaxies: binormal, with a galaxy noise component shared across the four
  encoders (ρ = 0.5) so the bootstrap pairing is exercised. They go through `aligned_c2.auc`,
  `boot_aucs` and `weights` (2,000 draws), then `decide()`, the function the real scoring calls.
- **Synthetic inputs.** Base AUCs ~ U(0.70, 0.95), drawn once with seed 20261024. Seed noise is
  τ = 0.002 and σ = 0.006 unless stated. The realised AUCs also carry finite-test-set noise, as real
  ones do.

*(Results, including two plants that missed their written expectation: see "Plant results" below.)*

### Reading

- **Primary.** Two family states at 4 epochs: MoCo against JEPA and MAE against JEPA. Each is BETTER,
  WORSE, SAME, NEGLIGIBLE, SPLIT, UNRESOLVED (with its reason) or VOID, with D̄, its 97.5% CI, the bar,
  s_run and any SPREADS DIFFER flag. "The JEPA objective reads morphology better than X" is claimed
  only on WORSE. "No different" is claimed only on SAME.
- **Secondary.** The per-answer states, read against the expected 0.9 chance calls, and the trajectory
  state.
- **Out of scope.** No state is formed across objectives beyond these two. No comparison with M.

## Plant results (before the hash)

Run 23 Oct, about 2 h, through the full path (`baselines_comparison.py plants` →
`artifacts/out/baselines_plants.json`, SHA-1 `1ff6252a…`). Every plant is recorded below, failures
included (D28). The expectation column is the one written into the script before the run.

| plant | set-up (τ, σ, Δ on X) | expected | family states (count) | per-answer states, other answers (rate) | median bar | verdict |
|---|---|---|---|---|---|---|
| null | 0.002, 0.006, 0 | SAME | SAME 9, UNRESOLVED 1 | SAME 0.69, UNRESOLVED 0.25, SPLIT 0.05; **no BETTER/WORSE** | 0.009 | fires |
| better_0.03 | 0.002, 0.006, +0.03 | BETTER | BETTER 10 | BETTER 0.72, UNRESOLVED 0.27 | 0.015 | fires |
| worse_0.03 | 0.002, 0.006, −0.03 | WORSE | WORSE 10 | WORSE 0.81, UNRESOLVED 0.18 | 0.011 | fires |
| split | 0.002, 0.006, X0 +0.03 / X1 −0.03 | SPLIT | SPLIT 10 | SPLIT 0.95 | 0.208 | fires |
| negligible | 0.0003, 0.0005, +0.003 | NEGLIGIBLE | NEGLIGIBLE 2, SAME 5, UNRESOLVED 3 | SAME 0.39, UNRESOLVED 0.61 | 0.004 | **reached, not reliably** |
| underpowered | 0.015, 0.006, 0 | UNRESOLVED | UNRESOLVED 7, **SPLIT 3** | UNRESOLVED 0.69, SPLIT 0.31 | 0.121 | **fails expectation 3/10** |
| void | X1 lacks 6 answers | VOID | VOID 3 | — | — | fires |
| extreme | 0.002, 0.006, +0.15 | BETTER | BETTER 5 | BETTER 1.00 | 0.015 | fires; not saturated |
| spreads_differ_null | X 3× noisier, 0 | not BETTER/WORSE/SPLIT | UNRESOLVED 6, SAME 3, **SPLIT 1** | UNRESOLVED 0.82, SPLIT 0.15, SAME 0.03; no BETTER/WORSE | 0.022 | **fails expectation 1/10** |
| five_answers_0.03 | 0.002, 0.006, +0.03 on 5 answers | per-answer: 5 BETTER | SAME 9, UNRESOLVED 1 | planted: **BETTER 0.90**, UNRESOLVED 0.10; rest: no BETTER/WORSE | 0.011 | fires |

Trajectory plants, three realisations each (τ 0.0003, σ 0.002, Δ per epoch as named):

| planted sequence (Δ at 0.5/1/2/4 ep) | got |
|---|---|
| CONSISTENT (0.03 ×4) | CONSISTENT ×3 |
| CONVERGES (0.03, 0.03, 0, 0) | CONVERGES ×3 |
| DIVERGES (0, 0, 0.03, 0.03) | DIVERGES ×3 |
| CROSSES (0.03, 0.03, −0.03, −0.03) | CROSSES ×3 |

MIXED is not planted. It is the remainder by construction.

**What holds.**
- Under every null (null, underpowered, spreads_differ_null, and the 30 unplanted answers of
  five_answers) **no BETTER or WORSE was called**, at family or answer level.
- Each directional state fires 10/10 at Δ 0.03. The extreme plant gives BETTER on every answer
  without the CI touching the range's edge.
- A planted effect on 5 of 35 answers is found per answer (90%) and correctly stays SAME at family
  level, where it dilutes to 0.004.

**Negligible: re-planted, the first run kept.** In the first plant, the seed bar (3.5 × the A-arm seed
difference) is set at τ = 0.0003 mostly by finite-test-set AUC noise, and it came out about the size
of Δ = 0.003. So the plant sat on the edge. The rule is unchanged. Only the plant was strengthened:
more shared galaxy noise (ρ 0.9) and Δ 0.004, still below the 0.005 floor (`artifacts/baselines_plant_negligible_v2.py` →
`artifacts/out/baselines_plants_negligible_v2.json`, SHA-1 `4945557a…`; noise seed 20261025).
Result: **NEGLIGIBLE 9, SAME 1**. The SAME draw had a bar of 0.0047, a large half-normal draw of the
seed difference, and SAME still makes no directional claim.

**SPLIT under large seed noise: a limit of two seeds, not a bug.** SPLIT is checked before the bar.
D̄_0 and D̄_1 each carry a run-level difference of SD ≈ τ√2. At τ = 0.015 (underpowered) they disagree
in sign by ≥ 0.01 in about a third of draws, and they did, 3/10. A genuinely seed-dependent effect and
large run-level noise are the same observation at two seeds: the split plant drives the X arm's own
seed difference to 0.06 and its bar to 0.21. No bar gate can separate the two. SPLIT claims no
direction, so its false calls are never directional false calls. But it fired where UNRESOLVED was
expected (3/10 underpowered, 1/10 spreads_differ_null; per answer 5% under the null and 15% with a
3×-noisier baseline). This is on record as written. The resolution is for the user (below); nothing
has been changed.

## For the user

1. §C's design, constants (K, Z, NEGLIGIBLE, SESOI) and the power statement.
1a. **SPLIT's meaning** (Plant results). (a) *Recommended:* keep SPLIT first. Restate it as
    non-directional: "the seed pairs disagree; no direction is claimed; at two seeds, not
    distinguishable from large run-level noise". Report its null rates (family 0/10 at τ 0.002; per
    answer about 5%, 15% with a 3×-noisier baseline) beside it, as the 0.9 chance calls are. The
    plant expectations for underpowered and spreads_differ_null become "UNRESOLVED or SPLIT". That
    change is written beside the record, not over it. (b) Move SPLIT below SAME/UNRESOLVED, so it
    fires only when nothing else does. The split plant would then read UNRESOLVED (its bar is 0.21),
    which leaves SPLIT reachable only in theory and fails D28. (c) Gate SPLIT on the A arm's own seed
    spread. That needs a new calibration and a plant re-run (about 2 h).
2. §D: (a), (b) or (c), or none; and whether to add a comparative ladder state.
3. §E: fold in as §N (recommended), or revise `distractor_sensitivity.md` separately; and 40,000-capped
   train or a new `sky_r` pull.
4. P1's resume-based A-side checkpoints (about 2 h of training, with the identity check first), then
   P2's eight A-side probes (about 5 h). Neither reads a result; both are needed before scoring, and
   neither needs the hash.
