# Baselines against their paired JEPA seeds — pre-registration (DRAFT, not hashed)

**Question.** D12's Rung-3 control: with the data, backbone, exposure, initial weights and data
order held fixed, does the JEPA objective give a representation from which GZ2's morphology is
read better, the same, or worse than MoCo's and MAE's?

**Status.** Final draft (v3), 2026-10-26: the three-seed design (user, 23 Oct); SAME in the
equivalence form, the constants approved and P1's identity rule amended (user, 25 Oct). Nothing hashed, nothing read.
Draft v1 (two seed pairs) and its plants are in git at `e545a0c`. Its plant record is kept below.
The decision layer was recalibrated and the plants rerun under three pairs, before any hash (D28).
The user reviews this, and only then is the section from "## Pre-registration" through its last
content line hashed (SHA-1, trailing newline). The hash comes **before any trajectory probe file,
bank-derived score or ladder output of a baseline is read**.

**What was read to write this, and what was not.**
- **Read:** method text only:
  - `baselines.md` §1, §6, §10; `DECISIONS.md` D12, D27, D28;
  - the method sections of `aligned_comparison.md` (Common ground, criterion 2's design, the A3
    declaration's launch lines; its lines 133–141 and every Results, Plant-results and A3 report
    section were skipped);
  - the preamble and Common ground of `distractor_sensitivity.md`;
  - the probe, training and pull drivers' code;
  - the ladder runner's per-entry metadata (`docs/ladder_v2/state.json`: status, wall-clock, peak
    memory), the manifest and the inventory's method text;
  - the A runs' launch-header log lines (code_sha only).
- **Not read:** any trajectory probe file (only their SHA-1s, `baselines.md` §6.1), any bank, the
  ladder's report or outputs, the aligned comparison's results, any `*_findings.md`.
- **Two slips, declared.** Both were seen. They are not used here and are not repeated, and nothing
  in this draft depends on either.
  - `distractor_sensitivity.md` line 72, inside a method bullet, quotes five of M's N1 nuisance AUCs
    on v1. §N drops the variables' v1 numbers altogether.
  - `aligned_comparison.md` line 202 appeared in a grep's output. It quotes figures from M's AA3a.

## Pre-registration

### Common ground

- **Encoders, in seed pairs (`baselines.md` §1, Seeds).** Within a pair, initial weights and stamp
  order are identical, so the two encoders differ only in the objective. The third pair is the user's
  design change of 23 Oct.

  | pair | JEPA | MoCo | MAE |
  |---|---|---|---|
  | seed 0 | A1 `runs/m_v2` (`encoder.pt` `34bb30a7…`) | `runs/moco_s0` (`bf0f0f60…`) | `runs/mae_s0` (`fb3d03d1…`) |
  | seed 1 | A2 `runs/m_v2_s1` (`380cdb7b…`) | `runs/moco_s1` (`f5e2a9f6…`) | `runs/mae_s1` (`f375c9e5…`) |
  | seed 2 | A3 `runs/m_v2_s2` (`4c3e1c7e…`) | `runs/moco_s2` (to train) | `runs/mae_s2` (to train) |

  - A3 was declared exploratory for the aligned comparison (`aligned_comparison.md`, 11 Oct). Here it
    is a confirmatory arm, by this pre-registration, before any of its baseline partners exists.
  - A3 ran M's recipe from its own worktree at `a36994f`. That commit's `src/` is identical to
    A1/A2's `47e4a0f`.
- **Epoch points:** steps 12,664, 25,327, 50,654 and 101,308 (0.5, 1, 2 and 4 epochs of the 810,491
  train split). The baselines checkpoint at exactly these steps. **A1–A3 do not:** they kept a
  checkpoint every 2,000 steps plus the final one.
- **Precondition P1: the A-side checkpoints are made by exact resume, and each resume is shown
  identical first** (`artifacts/p1_resume.py`, chained by `artifacts/p1p2_chain.sh`).
  - **The construction** is `m2_long_run.main`'s, without the probe or the curve:
    - config through F0;
    - `_split_ids`, `seed_init(train_seed)` and `ResumableShuffle(train_seed)`;
    - the checkpointer's schedule dict, checked against the run's manifest, as is the config hash;
    - `train_jepa(resume=True, stop_after=…)`.
  - It writes only to scratch directories under `runs/_p1/`.
  - **The package is the current checkout's** (`e545a0c`), where the baselines trained. Its loop
    (build 2a) was proven identical to the loop A1–A3 ran by `tests/test_objective_loop.py`. The
    identity check is the empirical proof for these runs.
  - **Identity check, per run — as amended (user, 25 Oct):** resume from `ckpt_000012000` to step
    14,000. The **training state** must equal the run's own `ckpt_000014000` bit for bit: encoder,
    predictor and target-encoder weights, optimiser state, and loss history. So must **every random
    generator training draws from**.
    - A global generator counts as one training draws from unless the original run's own state of it
      is unchanged between its 12,000 and 14,000 checkpoints. That is checked per run, in code
      (`p1_resume.amended`).
    - **Original wording (draft v2, kept beside the amendment):** "Every leaf of the training state
      (encoder, predictor, target encoder, optimiser state, the torch/MPS/NumPy/Python RNG states) and
      the loss history must equal the run's own `ckpt_000014000` bit for bit."
    - **Why it was amended.** Under the original wording the three resumes failed on one leaf of
      15,933, `rng/torch`, the CPU generator. That generator is **unchanged from step 12,000 to 14,000
      in all three original runs**, so training never draws from it. The resumed state is the 12,000
      state advanced by exactly one draw: the base seed a DataLoader takes when the resumed process
      opens it. In fact all four global generators (CPU torch, MPS, NumPy, Python) are unchanged
      across the window in the originals: training's randomness is seeded from (seed, step), not
      drawn from them. In the resumes, the MPS, NumPy and Python states matched anyway.
    - **End-to-end check (user, 25 Oct):** a fresh resume from 12,000, run separately from the
      identity resumes, must reproduce the original step-14,000 checkpoint under the amended rule, per
      run, reported pass/fail (P1/P2 status, below).
    - A1: `858942a4…` → `e08c1e0a…`
    - A2: `9c8f2467…` → `014eb19f…`
    - A3: `89daa79e…` → `7cf1a1ea…`
  - **Then, only for a run that passes:** resume from 12,000 → 12,664, 24,000 → 25,327 and
    50,000 → 50,654, from the run's own checkpoints:
    - A1: `858942a4…`, `49fa5fe6…`, `08aa2884…`
    - A2: `9c8f2467…`, `78503155…`, `49e999a6…`
    - A3: `89daa79e…`, `cc92aae3…`, `ec626029…`

    Step 101,308 uses the runs' final checkpoints (`29a07f0b…`, `1b1013d3…`, `49e547f9…`).
  - **If a run's identity check fails:** that run's 0.5, 1 and 2-epoch points are VOID, and with
    them the trajectory readout (any VOID epoch voids the trajectory, §C). The primary readout
    (4 epochs) is unaffected. No nearest-checkpoint substitute is used.
- **Precondition P2: the A side is probed exactly as the baselines were.**
  - `baseline_trajectory_probe.py` (E2's protocol on ladder v2's identical split) runs on each A-side
    point.
  - It writes a bank per point and `<tag>_trajectory.json`, with tags `a1_traj1…4`, `a2_traj1…4`
    and `a3_traj1…4`.
  - The probe prints no AUC, and nothing reads the files before scoring.
- **Precondition P3: integrity.**
  - Every trajectory file and bank matches its SHA-1: `baselines.md` §6.1 for the baselines' probe
    files; the A-side files and all banks are hashed when P2 completes.
  - Scoring refits the probes from the banks (`aligned_c2.fit_scores`, the call the probe made).
    Every refitted held-out AUC must equal the trajectory file's `auc` for that encoder, answer and
    step **exactly**.
  - A mismatch stops scoring, and that epoch's state is VOID.
- **Split and read-out.** The identical split: 160,849 probe-train and 34,828 probe-test galaxies
  (`baselines_answer_set.json` records the ids' SHA-1, `17296e0f…`). Block 11, mean-pooled; `_fit`
  logistic regression with C from `probe.yaml`; AUC on each answer's eligible test galaxies.
- **Bootstrap.** 2,000 Poisson galaxy draws (`aligned_c2.weights`, seed 20260926), with one weight
  matrix shared by all nine encoders, all answers and all four epochs, so every difference is paired
  by galaxy.

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
  - **In:** every other Scheme-1 answer. Test minorities run from 415
    (`t11_arms_number_a36_more_than_4`) to 16,021 (`t07_rounded_a17_in_between`).
- **Weights.** The count-based SE also sets the per-answer weights w_j = SE_j² / mean(SE²) in §C.
  These tell the per-answer bar how noisy each answer's AUC is expected to be, from its counts alone.
- No list derived from any encoder's results (in particular not `powered`) enters anywhere.

### B. The comparison

- **Per objective X ∈ {MoCo, MAE}, per epoch point, six encoders:** A1–A3 and X0–X2, scored on the
  35 answers. Write A_s for the JEPA of training seed s (A1 for s = 0, and so on).
  - **Paired difference per answer and seed pair:** d_sj = AUC(X_s, j) − AUC(A_s, j), s = 0, 1, 2.
  - **Per-answer effect:** D_j = mean_s d_sj.
  - **Family effect:** D̄ = mean_j D_j.
  - **Pair means:** D̄_s = mean_j d_sj.
- **Primary readout: 4 epochs (step 101,308),** the stop every arm shares. One family state per
  objective (§C), so two primary states: MoCo against JEPA and MAE against JEPA.
- **Secondary readouts.**
  - The per-answer states at 4 epochs (§C).
  - **The trajectory:** the family state at 0.5, 1, 2 and 4 epochs, read into one trajectory state
    per objective (§C).
  - **§N:** the nuisance readout at 4 epochs.
- **Reported, no state.**
  - Every per-answer D_j with its CI, at every epoch.
  - MoCo against MAE: the same D_j and D̄, with X = MAE and the "JEPA" slot taken by MoCo.
  - The run-level and answer-level seed spreads of every arm (§C).

### C. Seed-noise bars and states

**Why the bar is built this way.** With few seeds per arm, a bar built from per-answer seed
differences (the aligned comparison's mean_j |A1_j − A2_j|) false-calls badly once part of the seed
noise is **run-level**, meaning one run is better or worse on every answer. In this design a uniform
advantage of X over A is exactly a run-level shift. Before any data, the decision-layer Monte Carlo
put that bar's false-call rate at 22–34% under purely run-level noise, and 10–14% when the two kinds
are equal (draft v1, §F). So the bar here estimates the run-level spread directly, from the seeds.

**Estimators** (per objective and epoch; n = 3 seed pairs; J = the answers scoreable for all six
encoders):

- **Run-level seed spread** of each arm: S_A is the sample SD (n − 1 = 2 df) over seeds of
  m_{A,s} = mean_j AUC(A_s, j), and likewise S_X. Answer-level noise averages out over 35 answers.
- **SE of D̄ from seed noise:** max(S_A, S_X) × √(2/n). D̄ is a mean of n pair differences, each
  carrying both arms' run-level noise. The larger arm is used, so the quieter arm cannot set the bar.
- **Answer-level residual spread** of each arm: the two-way (seed × answer) residual of its AUC table,
  with each answer's squared residual divided by its count weight w_j and pooled over
  (n − 1)(J − 1) df. That gives σ̂²_A and σ̂²_X. Then ŝ_j = √(w_j (σ̂²_A + σ̂²_X) / n), the
  answer-level SD of D_j.
- **Family bar:** b = K × max(S_A, S_X) × √(2/n).
- **Per-answer bar:** b_j = √(b² + (Z ŝ_j)²).
- **CIs** come from the shared galaxy bootstrap of D̄ (and of D_j): 97.5% for the family, 95% per
  answer.
- At n = 2 these estimators reduce exactly to draft v1's (b = K/√2 × |r|).

**The constants, and why each has its value** (approved by the user, 25 Oct).

| constant | value | reason |
|---|---|---|
| **K** | **3.0** | **Calibrated, not chosen.** The rule was fixed before the sweep: the smallest K on a 0.05 grid whose family false-call rate (a BETTER or WORSE call under no effect) is ≤ 2.5% in every equal-spread noise regime of §F. That is half the 5% two-sided budget per objective. K is large for a "z" because each arm's spread has only 2 df, so its estimate is often small by chance; K absorbs that. Cross-check: the same rule at n = 2 gives 5.05, against draft v1's hand-calibrated 3.5 × √2 = 4.95. |
| **Z** | **2.1** | **Calibrated the same way, with K fixed:** the smallest Z keeping each answer's false-call rate ≤ 2.5% in every equal-spread regime. It is close to a normal 97.5% quantile (1.96), because the answer-level variance is pooled over (n − 1)(J − 1) = 68 df and is estimated well. |
| **NEGLIGIBLE** | **0.005 AUC** | **A judgement, fixed in draft v1 before any data; not calibrated.** It is about one test-set SE of a typical answer: the count-based SE at AUC 0.75 has median 0.0039 over the 35 answers (range 0.0026–0.014). A family effect smaller than that cannot be seen on a typical single answer, however many answers average it into significance. BETTER and WORSE need \|D̄\| ≥ 0.005, and a significant smaller effect is NEGLIGIBLE. |
| **SESOI** | **0.02 AUC** | **A judgement, fixed in draft v1 before any data; not calibrated.** It is the smallest effect of interest. It equals the answer-set rule's SE ceiling, so a difference as large as the noisiest admitted answer's own SE, and about five SEs of the median answer. A family-wide shift of 0.02 would therefore be plain on most single answers. SAME needs a bar ≤ 0.02, so that a 0.02 effect could have been seen; SPLIT needs pairs ≥ 0.01 = SESOI/2 each way. |
| CI_FAM | 97.5% | 95% Bonferroni-split over the two objectives (two primary states). |
| CI_ANS | 95% | Per-answer readings are secondary, are reported against their expected chance count, and are never pooled. |
| MIN_ANSWERS | 30 of 35 | Fewer scoreable answers for any encoder means the family mean is no longer over the same set, so VOID. |
| SPREAD_FLAG | 2× | Reported, never decisive; the bar already takes the larger arm. |

**Family states: first match wins.**

| # | state | condition |
|---|---|---|
| 1 | **VOID** | P1–P3 failed for this epoch, or any of the six encoders has fewer than 30 of the 35 answers scoreable |
| 2 | **SPLIT** | **the seeds disagree; no direction is claimed.** Some pair mean D̄_s ≥ +SESOI/2 (0.01) and some pair mean ≤ −SESOI/2. Reported with its null rate beside it (below) |
| 3 | **BETTER** | CI(D̄) lies wholly above +b, and D̄ ≥ NEGLIGIBLE |
| 4 | **WORSE** | CI(D̄) lies wholly below −b, and D̄ ≤ −NEGLIGIBLE |
| 5 | **NEGLIGIBLE** | CI(D̄) lies wholly beyond ±b, but \|D̄\| < NEGLIGIBLE: significant against seed noise, too small to matter |
| 6 | **SAME** | **Equivalence form (user, 25 Oct):** CI(D̄) lies within [−b, +b], **and \|D̄\| + b ≤ SESOI**: the whole seed-noise interval around D̄ lies inside ±0.02, so a real 0.02 effect is not called SAME. If the CI excludes 0, it is labelled "detectable but below seed noise" |
| 7 | **UNRESOLVED** | everything else, labelled with why: **underpowered** (b > SESOI, so SAME is unreachable whatever the CI); **not equivalent** (CI inside the bar, but \|D̄\| + b > SESOI); **large but not resolved** (\|D̄\| ≥ SESOI but the CI crosses the bar); or **CI straddles the bar** |

- "X better than JEPA" is BETTER for that objective, and WORSE means the JEPA is better. The sign is
  fixed: D is baseline minus JEPA.
- **SPLIT (user's decision (a), 23 Oct).** SPLIT says only that the seed pairs disagree in sign by at
  least SESOI/2 each way. A seed-dependent effect and large run-level seed noise are the same
  observation at this number of seeds, and no bar can separate them. So SPLIT never claims a
  direction, and its false calls are never directional false calls.
  - It is checked first, so it pre-empts some calls that would otherwise be BETTER or WORSE. The
    power figures in §F include that cost.
  - **Its null rate is reported beside every SPLIT** (§F, three seeds): at family level 0–1% when the
    run-level spread τ ≤ 0.004, and 8% at τ = 0.006; per answer 1–17% across the equal-spread
    regimes; 22% / 32% when the baseline is 3× noisier.
  - With three pairs, "some pair each way" is easier to meet than with two, so the rates are higher
    than draft v1's.

**D27 checklist.**
- **Sign flips:** a reversal across seed pairs is SPLIT, and either direction is a named state.
- **Insufficient sample:** VOID below 30 answers, and the answer-set rule gates every answer by
  counts.
- **Significance against magnitude:** NEGLIGIBLE covers significant but tiny. UNRESOLVED (large but
  not resolved) covers large but not significant. SAME requires the seed-noise interval to lie
  within ±0.02 (equivalence), so it can be called on a real 0.02 effect at most 1.8% of the time
  (below).
- **Precedence:** the ordered table.

**SAME's two rates (user, 25 Oct: recorded here).** Decision layer, three pairs, K 3.0, the CI taken
as its point (`artifacts/baselines_false_same.py` → `artifacts/out/baselines_false_same_3seed.json`,
`81d02b4f…`, 60,000 draws per cell). The full-path plants measure the same two rates on real labels
("Plant results" below).

| seed noise τ, σ, X/A | **SAME wrongly called** on a real Δ = 0.02 / 0.03 | **SAME reached** at a true null (Δ = 0) | for contrast, the old rule: wrongly at 0.02 / reached at 0 |
|---|---|---|---|
| 0.001–0.002, 0.003–0.008, 1× | 1.0–1.2% / 0% | 100% | 0% / 98% |
| 0, 0.006, 1× | 1.2% / 0% | 100% | 0% / 98% |
| 0.004, 0.006 (lognormal), 1× | 1.1% / 0% | 86% | 4.5% / 92% |
| 0.006, 0.006, 1× | 1.2% / 0% | 46% | 11.1% / 66% |
| 0.006, 0, 1× | 1.1% / 0% | 48% | 11.1% / 68% |
| 0.004, 0.004, 3× | 1.8% / 0.1% | 19% | 8.3% / 31% |
| 0.010, 0.006, 1× | 1.1% / 0.1% | 9% | 7.0% / 21% |
| 0.015, 0.006, 1× | 0.5% / 0.1% | 2% | 2.3% / 5% |

- **"No different" is now a claim with a bounded error:** SAME is called on a real 0.02 effect at most
  1.8% of the time in any regime, and on 0.03 at most 0.1%.
- **The price is reach.** At a true null, SAME is reached nearly always when τ ≤ 0.002, about half
  the time at τ = 0.006, and rarely once τ ≥ 0.010. The rest read UNRESOLVED ("not equivalent" or
  "underpowered"), never a directional state.

**Per-answer states (secondary, 4 epochs).** The same table on D_j, using b_j and the 95% CI. SPLIT
uses the answer's own pair differences d_sj, and VOID applies if any of the six encoders lacks the
answer. **No state is pooled across answers.**
- Under the null, the calibrated per-answer false-call rate is ≤ 2.5% (§F). That means at most
  about 0.9 of 35 answers are expected to read BETTER or WORSE by chance, per objective.
- SPLIT's per-answer null rate is 1–17%, so 0.4–6 answers per objective by chance, depending on the
  seed noise the real runs show.
- Every per-answer reading is reported against these two numbers.

**Trajectory states (secondary).** The family states at 0.5, 1, 2 and 4 epochs give one trajectory
state per objective. First match wins:

| # | state | condition |
|---|---|---|
| 1 | **VOID** | any epoch is VOID (including a P1 failure for any A run) |
| 2 | **CROSSES** | BETTER at one epoch and WORSE at another |
| 3 | **CONSISTENT** | the same state at all four |
| 4 | **DIVERGES** | BETTER or WORSE at 4 epochs, and neither at 0.5 or 1 |
| 5 | **CONVERGES** | neither BETTER nor WORSE at 4 epochs, but one of them at 0.5 or 1 |
| 6 | **MIXED** | anything else, with the four states listed |

The probes at earlier epochs are trajectories only. No checkpoint is chosen by them (spec 1C;
`baselines.md` §1.1).

### D. Ladder experiments on the six baselines: option (a), provisional (user, 23 Oct)

D12 asks for "the same probe ladder across JEPA, MAE, contrastive". Ladder v2 ran it on A1–A3.
**Option (a), existence, is chosen provisionally.** It covers exactly the following ladder-v2
manifest entries, per baseline encoder, run by the ladder runner with `LADDER_ENCODER` set to that
encoder (re-exported through `frozen_from`, as the trajectory probe does) and outputs under
`runs/ladder_v2/<experiment>/<moco_s0|…|mae_s2>/`.

Measured wall-clock (`docs/ladder_v2/state.json`) on A1 / A2 / A3:

| entry | what it is | A1 | A2 | A3 | peak memory |
|---|---|---|---|---|---|
| E3 `o1_matched` | the union bank (real + untrained seed 0) and matched evaluation O1 | 0.68 h | 0.59 h | 0.41 h | 3.3–6.7 GB |
| E7 `r_bank_link` | links the shared untrained seeds 1–2 bank (input to E5) | 0 | 0 | 0 | — |
| E5 `p2_ladder` | `run_ladder`, phases 1–4: per-answer logistic AUC and bootstrap SE; the 3C battery (50 shuffled + 50 Gaussian nulls, noise, untrained seed 0, five-nuisance panel); existence by D23 `untrained_z` against the K = 30 untrained bank, BY over 37, effect floor 0.7267; E6 entanglement inside it | 6.59 h | 6.08 h | 5.34 h | 3.1–6.8 GB |
| E7 `r0` | r_nonlinear stage r0, an input to s2_verify | 0.54 h | 0.44 h | 0.96 h | 2.6–3.3 GB |
| E5 `s2_verify` | D24 nuisance-clearance verification; banks the noise matrix | 2.47 h | 2.46 h | 4.01 h | 3.4–4.8 GB |
| E5 `t1_ladder` | production ladder: D24 retention clearance, R4 MLP capacity sweep (D25 decode unadjudicated) | 1.93 h | 2.98 h | 1.97 h | 2.6–3.2 GB |
| **per encoder** | | **12.2 h** | **12.6 h** | **12.7 h** | |

- **Total for six baselines: about 73–76 h, strictly serial,** on the machine as it was while the
  ladder ran (to 17 Oct). Under today's shared load from other sessions, P1's training runs 5–7×
  slower than A's measured rate.
- **Shared inputs carry over and are not re-run:** the v2 untrained bank (E4, `p1_untrained_bank`,
  6.6 h once) and E7's untrained seeds bank. The untrained encoders share the baselines'
  architecture and pixels.
- **Plumbing, about half a day:**
  - manifest entries for six new encoders;
  - `t1_ladder`'s `compare()` against M's records stays skipped under `LADDER_*` (Task 1b.3);
  - `s2_verify`'s noise bank is keyed by encoder checkpoint (Task 1b.2), so no baseline inherits
    another's.
- **How it is read:** each baseline's own ladder states, beside A1–A3's with an agreement column,
  with **no comparative state**. A comparative state, such as "existence survives in JEPA and not
  MoCo", would need its own D27 table and plants before this hash.
- Not in (a): E7 r1–r3, E11 U2, E12 U3, E15 V3, E20 AA1 (option (b), about +6 h per encoder) and
  DD's SAEs (E24, option (c)).

### N. Nuisance readout (folded in from `distractor_sensitivity.md`; user, 23 Oct)

The v1 draft was written for v1 baselines against M/O2. Here the baselines and the JEPA seeds are all
on aligned v2 stamps, so the four band-offset variables are gone. What remains is whether the
objectives differ in how much of three imaging nuisances their representations carry.

- **Variables (three, with states):**
  - **PSF:** `psfWidth_r` (SDSS `Field`, from probe_v2's metadata);
  - **brightness:** `modelMag_r` (`PhotoObjAll`, from probe_v2's metadata);
  - **sky:** `sky_r` (`PhotoObjAll`), pulled from public SkyServer for exactly the identical split's
    probe-train and probe-test galaxies (`artifacts/baselines_nuisance_pull.py` →
    `artifacts/out/baselines_nuisance.{csv,json}`; query and output SHA-1 in "Nuisance pull" below).
  - The same pull returns `psfWidth_r` and `modelMag_r` and is checked against the metadata (below).
  - Each variable is binarised at its own split's median, train and test separately, as N1's panel
    does (`nuisance_label`: `nanmedian`, ≥ is 1). Sentinels (≤ −9000), non-positive sky and missing
    values are excluded, per variable.
- **Reported, no state:** `snr_r`, redshift (`specz`) and size (`petroRad_r`), in the same way.
- **Statistic:** per encoder and variable v, the probe AUC for the binarised v, using the same banks
  at 4 epochs, `_fit` with C from `probe.yaml`, trained on the probe-train galaxies and scored on the
  probe-test galaxies, with the shared bootstrap weights.
  - D_v = mean over the three seed pairs of AUC(X_s, v) − AUC(A_s, v). Positive means the baseline
    carries **more** of the nuisance.
- **Bar:** b_v = K × max(S_A,v, S_X,v) × √(2/3), with K = 3.0, where S is the SD over the three
  seeds of the arm's AUC for v.
  - There is no answer-level pool and no averaging over answers. §F's pure run-level regime
    (σ = 0) is exactly this structure, and at K = 3.0 it false-calls 2.5%.
- **States:** the family table with **MORE** for BETTER and **LESS** for WORSE: X carries more or
  less of v than the JEPA. SPLIT, NEGLIGIBLE, SAME (equivalence: \|D_v\| + b_v ≤ 0.02), UNRESOLVED and VOID are as in the
  table, with a 97.5% CI.
  - Six readings (three variables × two objectives), so 0.15 chance MORE/LESS calls are expected in
    all. Each reading is reported against that.
- **The v1 draft's criterion 2** (the nuisance share of the leading variance): reported with no state
  in this round, to keep the hash small.
- **Plants:** run (see "Nuisance pull" below). The same decision code is used, with synthetic
  scores on the median-split labels of the pulled values: a null, MORE and LESS at Δ 0.03, and a
  SPLIT. None of the v1 plants is reused, because they ran on M and O2.

### F. Calibration and plants (D28): run before the hash, using no real result

**Decision-layer calibration** (`baselines_comparison.py calibrate` →
`artifacts/out/baselines_calibration_3seed.json`, SHA-1 `73928255…`).
- **Set-up:** 60,000 draws per row, seed 20261024, the 35 answers' count weights.
- **Synthetic AUC tables:**
  - a run-level offset c ~ N(0, τ) per run;
  - answer-level noise ~ N(0, σ √w_j), or lognormal σ_j deliberately unrelated to the counts;
  - a planted Δ on X.
- **Scored by** the §C estimators and the §C precedence: SPLIT first, a directional call only with
  |D| ≥ NEGLIGIBLE. The CI condition is the plants' job.
- **Outcome:** K and Z come from the rule in §C. Both designs are shown, each with its own calibrated
  K and Z (n = 2: K 5.05, Z 1.95; n = 3: K 3.0, Z 2.1).

| τ (run) | σ (answer) | X/A | noise | false calls, family / answer (n = 3) | SPLIT under null, family / answer (n = 3) | family power Δ 0.01 / 0.02 / 0.03, **n = 3** | same, n = 2 | per-answer power Δ 0.02 / 0.03, n = 3 | median bar, n = 3 |
|---|---|---|---|---|---|---|---|---|---|
| 0.006 | 0.006 | 1× | ∝ SE | 2.3% / 1.6% | 7.7% / 17.0% | 19% / **65%** / **94%** | 8% / 24% / 45% | 53% / 86% | 0.016 |
| 0.006 | 0 | 1× | — | 2.5% / 2.5% | 7.5% / 7.5% | 20% / **67%** / **94%** | 8% / 25% / 45% | 67% / 94% | 0.016 |
| 0 | 0.006 | 1× | ∝ SE | 2.3% / 2.3% | 0.0% / 6.1% | 100% / 100% / 100% | 89% / 100% / 100% | 92% / 97% | 0.003 |
| 0.002 | 0.008 | 1× | ∝ SE | 2.5% / 1.9% | 0.0% / 12.2% | 84% / **100%** / 100% | 35% / 80% / 97% | 83% / 94% | 0.007 |
| 0.004 | 0.006 | 1× | lognormal | 2.3% / 2.4% | 1.2% / 13.5% | 39% / **92%** / 100% | 14% / 43% / 70% | 53% / 83% | 0.012 |
| 0.002 | 0.004 | 1× | ∝ SE | 2.3% / 1.6% | 0.0% / 2.9% | 92% / 100% / 100% | 42% / 87% / 99% | 96% / 100% | 0.006 |
| 0.001 | 0.003 | 1× | ∝ SE | 2.3% / 1.7% | 0.0% / 1.2% | 100% / 100% / 100% | 84% / 100% / 100% | 99% / 100% | 0.003 |
| 0.004 | 0.004 | **3×** | ∝ SE | 3.6% / 2.2% | 22.2% / 32.2% | 12% / 36% / 61% | 8% / 18% / 32% | 26% / 51% | 0.025 |

- **False calls hold:** 2.3–2.5% per objective in every equal-spread regime, and 3.6% with a 3×
  noisier baseline, against a 5% two-sided budget per objective. Per answer they are ≤ 2.5%.
- **Power statement, three paired seeds.** Power depends almost entirely on the run-level seed spread
  τ, which only the real runs will show; the A seeds' own spread is a result and was not looked at.
  - At **τ = 0.004**: Δ = 0.02 is detected **92%** of the time, and Δ = 0.03 always.
  - At **τ = 0.006**: Δ = 0.02 is detected **65%** of the time (two seeds: 24%), and Δ = 0.03
    **94%** (two seeds: 45%).
  - At τ ≤ 0.002, both are detected essentially always.
  - When the bar exceeds 0.02, SAME is unreachable and the state is UNRESOLVED (underpowered). The
    rule does not hide this.
  - Most of the third seed's gain comes from the spread estimate going from 1 to 2 df, which lets K
    fall from 5.05 to 3.0. Only a small part comes from √(2/3) against √(2/2).
- **When the X arm is noisier,** power falls with it (36% / 61% at 3×), because the bar takes the
  noisier arm. The SPREADS DIFFER flag says so beside the state.

**Full-path plants** (`baselines_comparison.py plants` → `artifacts/out/baselines_plants_3seed_equiv.json`,
SAME in the equivalence form; two plants added on 25 Oct: a real 0.02 effect, and a null at τ 0.006).
- **How they are built:**
  - synthetic per-galaxy scores on the real labels of each answer's eligible test galaxies;
  - binormal, with a galaxy noise component shared across the six encoders (ρ = 0.5), so the
    bootstrap pairing is exercised;
  - passed through `aligned_c2.auc`, `boot_aucs` and `weights` (2,000 draws), then `decide()`, the
    function the real scoring calls.
- **Synthetic inputs:**
  - base AUCs ~ U(0.70, 0.95), drawn once with seed 20261026;
  - seed noise τ = 0.002 and σ = 0.006 unless stated;
  - the realised AUCs also carry finite-test-set noise, as real ones do.
- **The split is guarded.** The labels come from the identical split, and the run refuses unless the
  split's SHA-1 equals the answer set's. Draft v1's plants did not have this guard and ran on v1's
  probe population (below).
- **Expectations are fixed in the script before the run.** Two changed from v1, by the user's
  decision (a) on SPLIT, and are marked as changed:
  - underpowered: "UNRESOLVED or SPLIT";
  - spreads_differ_null: "not BETTER/WORSE".
- **The negligible plant** uses draft v1's strengthened version (ρ 0.9, Δ 0.004).

### Reading

- **Primary.** Two family states at 4 epochs: MoCo against JEPA and MAE against JEPA.
  - Each is BETTER, WORSE, SAME, NEGLIGIBLE, SPLIT (with its null rate), UNRESOLVED (with its
    reason) or VOID.
  - Each is given with D̄, its 97.5% CI, the bar, S_A and S_X, the three pair means and any SPREADS
    DIFFER flag.
  - "The JEPA objective reads morphology better than X" is claimed only on WORSE. "No different" is
    claimed only on SAME.
- **Secondary.** The per-answer states, read against the expected 0.9 chance directional calls and
  the SPLIT null rate; the trajectory state; §N's six readings.
- **Out of scope.** No state is formed across objectives beyond these two. No comparison with M.

## Plant results, three seed pairs, SAME in the equivalence form (before the hash)

Run 25–26 Oct, 21:58–01:28, through the full path (`baselines_comparison.py plants` →
`artifacts/out/baselines_plants_3seed_equiv.json`, **SHA-1 `e54aff7c…`**).
- K 3.0, Z 2.1, six encoders, SAME in the equivalence form. The split was guarded: its SHA-1 equals
  the answer set's.
- Seeds are as in the 24 Oct run, and the two new plants come last, so every earlier plant sees the
  same synthetic AUCs as before; only the SAME rule differs.
- Every plant is recorded, misses included (D28). Expectations are as written in the script before
  the run.

| plant | set-up (τ, σ, Δ on X) | expected | family states (count) | per-answer states, other answers (rate) | median bar | verdict |
|---|---|---|---|---|---|---|
| null | 0.002, 0.006, 0 | SAME | SAME 9, UNRESOLVED 1 | SAME 0.68, UNRESOLVED 0.22, SPLIT 0.10; **no BETTER/WORSE** | 0.006 | fires |
| better_0.03 | 0.002, 0.006, +0.03 | BETTER | **BETTER 10** | BETTER 0.92, UNRESOLVED 0.07, SPLIT 0.01 | 0.007 | fires |
| worse_0.03 | 0.002, 0.006, −0.03 | WORSE | **WORSE 10** | WORSE 0.92, UNRESOLVED 0.08 | 0.007 | fires |
| split | X0 +0.03, X1 −0.03, X2 0 | SPLIT | **SPLIT 10** | SPLIT 0.98 | 0.074 | fires |
| negligible | 0.0003, 0.0005, +0.004 (ρ 0.9) | NEGLIGIBLE | **NEGLIGIBLE 10** | UNRESOLVED 0.54, NEGLIGIBLE 0.34, BETTER 0.10, SAME 0.02 | 0.001 | fires |
| underpowered | 0.015, 0.006, 0 | UNRESOLVED or SPLIT | SPLIT 5, UNRESOLVED 4, **SAME 1** | SPLIT 0.51, UNRESOLVED 0.45, SAME 0.04 | 0.050 | **misses its expectation 1/10** (below) |
| void | X2 lacks 6 answers | VOID | VOID 3 | — | — | fires |
| extreme | 0.002, 0.006, +0.15 | BETTER | **BETTER 5** | BETTER 1.00 | 0.005 | fires; not saturated |
| spreads_differ_null | X 3× noisier, 0 | not BETTER/WORSE | SAME 7, UNRESOLVED 3 | UNRESOLVED 0.56, SPLIT 0.29, SAME 0.13, BETTER 0.009 | 0.009 | fires (family); per answer 0.9% false BETTER, within the 2.5% bound |
| five_answers_0.03 | +0.03 on 5 answers | per-answer: 5 BETTER | SAME 9, NEGLIGIBLE 1 | planted: **BETTER 1.00**; rest: SAME 0.71, UNRESOLVED 0.17, SPLIT 0.12, no BETTER/WORSE | 0.006 | fires |
| **real_0.02** (new) | **0.006, 0.006, +0.02** | **not SAME** | BETTER 14, UNRESOLVED 6, **SAME 0 of 20** | BETTER 0.46, UNRESOLVED 0.49, SPLIT 0.04, **SAME 0** | 0.016 | fires: **SAME wrongly called 0/20 (family), 0% (answers)** |
| **null_tau0.006** (new) | **0.006, 0.006, 0** | SAME reachable | **SAME 10**, UNRESOLVED 8, SPLIT 2 (of 20) | UNRESOLVED 0.57, SAME 0.22, SPLIT 0.21, BETTER 0.006 | 0.016 | fires: **SAME reached 10/20 (family)** |

Trajectory plants, three realisations each (τ 0.0003, σ 0.002):

| planted sequence (Δ at 0.5/1/2/4 ep) | got |
|---|---|
| CONSISTENT (0.03 ×4) | CONSISTENT ×3 |
| CONVERGES (0.03, 0.03, 0, 0) | CONVERGES ×3 |
| DIVERGES (0, 0, 0.03, 0.03) | DIVERGES ×3 |
| CROSSES (0.03, 0.03, −0.03, −0.03) | CROSSES ×3 |

**SAME's two rates on the full path (user, 25 Oct).** These match §C's decision-layer figures.

| rate | regime | family | per answer | §C, decision layer |
|---|---|---|---|---|
| SAME wrongly called on a real 0.02 effect | τ 0.006, σ 0.006 | **0 / 20** | **0%** of 700 readings | 1.2% |
| SAME reached at a true null | τ 0.006, σ 0.006 | **10 / 20** | 22% | 46% |
| SAME reached at a true null | τ 0.002, σ 0.006 | 9 / 10 | 68% | 100% (σ 0.008) |
| SAME reached at a true null | 3× noisier baseline | 7 / 10 | 13% | 19% |

Per-answer SAME is reached less often than family SAME, because each answer's bar also carries its
answer-level noise.

**What holds.**
- **No false directional call at family level** under any null.
- Each directional state fires 10/10 at Δ 0.03, and per answer at 92%. NEGLIGIBLE fires 10/10.
- The five-answer plant is found on every planted answer. Its single family NEGLIGIBLE is correct:
  the planted family effect is 5 × 0.03 / 35 = 0.0043, below the NEGLIGIBLE floor.
- Per-answer readings in the negligible plant: every answer carries a true +0.004, so its 10% BETTER
  calls point in the true direction. They are calls where the realised D_j crossed 0.005 by test-set
  noise, not false directions.
- Per-answer false BETTER under the nulls is 0.6% (τ 0.006) and 0.9% (3× noisier baseline), within
  the 2.5% bound.
- **SPLIT per answer under the nulls:** 10% (τ 0.002), 12% (five-answer rest), 21% (τ 0.006) and 29%
  (3×-noisier baseline). It is reported beside every SPLIT.

**The one miss, recorded.**
- The underpowered plant (τ 0.015, no effect) read SAME once in ten. It is the same realisation as
  in the 24 Oct run: the seeds reuse their streams. Its bar came out small enough that even
  |D̄| + b ≤ 0.02.
- The expectation, written before the run, assumed the bar would exceed 0.02 at this noise. Under a
  true null SAME is a true statement. The decision-layer rate of SAME at a null in this regime is
  1.6%, and of SAME on a real 0.02 effect 0.5%.
- Left as a miss against the written expectation. No rule changed.

**The 24 Oct three-seed run (old SAME rule), kept on record.** `artifacts/out/baselines_plants_3seed.json`,
`e5127982…`.
- Identical except for SAME. Its one miss (the same underpowered realisation reading SAME) led to the
  false-SAME measurement in §C, and to the user's adoption of the equivalence form on 25 Oct.

## P1/P2 status (final, 26 Oct)

- **First chain** (`artifacts/p1p2_chain.sh`, 24 Oct 18:15 → 25 Oct 01:08): the three identity
  resumes trained to step 14,000, then each crashed in the comparator.
  - The crash: the NumPy RNG state's array leaf was compared with `==`. That is a tooling bug.
  - The chain read the non-zero exit as a failed identity check. That conflated a crash with a
    result. The later chains report crashes as technical failures.
  - It ran P2's 4-epoch probes.
- **The comparator was fixed** (NumPy arrays compared by dtype, shape and `array_equal`). The three
  trained checkpoints were compared again without retraining (`p1_resume.py --recompare`).
  - In each run 15,932 of 15,933 leaves are bit-identical. The one that differs is `rng/torch`.
  - That fails P1's original wording, and passes the amended rule (§ Common ground,
    `runs/_p1/A*_identity/p1_record.json`).
- **End-to-end check under the amended rule** (`artifacts/p1_e2e_chain.sh`, 25 Oct 21:40 → 26 Oct
  00:11): a fresh resume from 12,000 → 14,000 per run, into `runs/_p1/A*_e2e`.
  - **A1 PASS, A2 PASS, A3 PASS** (61, 48 and 40 min).
  - Training state bit-identical to the original `ckpt_000014000`. The single differing leaf
    (`rng/torch`) is allowed under the amended rule only because each original run's own CPU
    generator is unchanged from 12,000 to 14,000 (`p1_record.json`, `amended`).
  - Each driver's log line printed "IDENTICAL" for this amended verdict. The exact statement is "1
    leaf of 15,933 differs, allowed; amended P1 passes".
- **The early-epoch points are therefore valid.** The continuation chain (`artifacts/p1p2_chain2.sh`,
  25 Oct 01:26 → 06:58, all 18 steps rc 0) made the 12,664 / 25,327 / 50,654 resumes and probed them.
- **P2 is complete:** twelve A-side probe files, `a{1,2,3}_traj{1..4}`, none opened.
- **P3, SHA-1s recorded before scoring:** `artifacts/baselines_p3.sha1` lists all 28 trajectory
  files (12 A-side, 16 baseline) and their 28 banks, hashed unread on 26 Oct. The 16 baseline
  trajectory files still match the SHA-1s recorded in `baselines.md` §6.1.

## Nuisance pull

Pulled 24 Oct, 18:28–19:14 (2,709 s), by `artifacts/baselines_nuisance_pull.py` →
`artifacts/out/baselines_nuisance.csv`, **SHA-1 `6bea1443…`** (record `baselines_nuisance.json`).
- **Service:** public SkyServer SQL, DR17 (`galaxy_jepa.data.metadata.run_sql`), no token; 100-ID
  batches with backoff.
- **Query, per batch:**
  `SELECT CAST(p.objID AS varchar(20)) AS objID, p.sky_r, f.psfWidth_r, p.modelMag_r FROM PhotoObjAll
  AS p JOIN Field AS f ON f.fieldID = p.fieldID WHERE p.objID IN (…)`.
  `psfWidth_r` lives in `Field`, joined exactly as the probe corpus's own pull joins it
  (`metadata.PROBE_SQL_TEMPLATE`).
- **Galaxies:** exactly the identical split's 160,849 probe-train + 34,828 probe-test. The ids hash to
  the answer set's `17296e0f…`, and the request list to `20c7b059…`.
- **Returned:** 195,677 of 195,677 rows. No duplicates, no missing ids, and no sentinel, non-positive
  or unparseable `sky_r`.
- **Confirmation of the other two columns:** `psfWidth_r` and `modelMag_r` agree with probe_v2's
  `metadata.csv` on **all 195,677 galaxies, maximum absolute difference 0.0**, with none missing or
  sentinel on either side. §N reads them from the metadata, which is now confirmed identical to the
  catalogue.
- `snr_r`, `specz` and `petroRad_r` (reported, no state) come from the same metadata.

**§N plants**, rerun with SAME in the equivalence form (`baselines_comparison.py nplants` →
`artifacts/out/baselines_nplants_3seed_equiv.json`, SHA-1 `6e87d111…`; the 24 Oct run under the old
SAME, `878f4571…`, read the same except null SAME 24 / UNRESOLVED 6).
- **Labels:** the pulled values' median-split labels on all 34,828 test galaxies, for each variable.
- **Synthetic inputs:** base AUCs ~ U(0.60, 0.90), seed 20261027; per-encoder seed noise τ = 0.004.
- **Path:** the shared bootstrap weights, then `decide_nuisance()`, which calls the same `state()` as
  the family table.
- Ten realisations × three variables per plant. The expectations were fixed in the script:

| plant | expected | readings (of 30) | median bar |
|---|---|---|---|
| null | not MORE/LESS | SAME 23, UNRESOLVED 7 | 0.010 |
| more_0.03 | MORE | **MORE 30** | 0.012 |
| less_0.03 | LESS | **LESS 30** | 0.011 |
| split (X0 +0.03, X1 −0.03) | SPLIT | **SPLIT 30** | 0.071 |

Every §N state that carries a claim fires, and the null makes no directional call.

## Draft v1's plants (two seed pairs; superseded, kept on record)

The full v1 record is in `e545a0c` (`artifacts/out/baselines_plants.json`, `1ff6252a…`;
`baselines_plants_negligible_v2.json`, `4945557a…`). In brief:
- **What fired.** No BETTER or WORSE under any null. Each directional state fired 10/10 at Δ 0.03.
  Extreme, VOID, the five-answer plant (90% per answer) and all four trajectory states fired.
- **Negligible** reached NEGLIGIBLE only 2/10. It was re-planted stronger (ρ 0.9, Δ 0.004) with the
  rule unchanged, and then fired 9/10.
- **SPLIT** fired where UNRESOLVED was expected: 3/10 under τ 0.015, and 1/10 with a 3×-noisier
  baseline. The user's decision (a) on 23 Oct restates SPLIT as non-directional, as above.
- **Found 24 Oct, recorded here: v1's plants ran on v1's probe population, not the identical
  split.** `plants()` imported `aligned_c2` before `_labels()` set the v2 environment. That import
  makes F0 read `F0_CACHE_BASE` before `baseline_run` sets it, so it fell back to v1's cache. So the
  labels came from v1's test split, which includes one of the nine galaxies probe_v2 lacks (34,829
  test galaxies, not 34,828), with the same GZ2 labels.
  - The decision layer is unaffected, and the plants' verdicts stand as a demonstration of the
    states.
  - The three-seed plants fix the import order and refuse to run unless the split's SHA-1 equals the
    answer set's.
  - `baseline_trajectory_probe.py` imports `baseline_run` first and is checked by
    `tests/test_trajectory_split.py`, so the trajectory files were never affected.

## Cost of the third seed

- **Training:** `baseline_run.py` moco2 and mae2 on A3's recipe (seed 2: A3's init and data order).
  - The four seed-0/1 runs took MoCo 20.4 h and 26.5 h, and MAE 35.7 h and 33.9 h, inline probes
    included (`baselines.md` §6.1).
  - So about **MoCo 20–27 h + MAE 34–36 h ≈ 54–63 h**, strictly serial, on the machine as it was on
    18–23 Oct. Under today's shared load (P1's training is running 5–7× slower) it would be several
    times that.
- **Code, about 2 h, with a test:**
  - `baseline_run.ARMS` gains seed 2;
  - the "pairs with" line names A3;
  - the config-equality check is unchanged, since A3's config hash equals A1's (`dae9709b…`).
- **A3's side:** done. P1 passes for A3, and its four probe files are made and hashed.
- **Ladder (a):** +2 encoders, about 25 h, included in §D's 73–76 h.

## Decided (user, 25 Oct) and what remains

- **Decided:**
  - SAME in the equivalence form, at family and answer level, with both rates recorded in §C;
  - K, Z, NEGLIGIBLE and SESOI approved as stated;
  - the P1 identity rule amended (original wording kept beside it), with the end-to-end check, which
    passes for all three runs.
- **Before the hash:** the user's review of this final draft. Then the section from
  "## Pre-registration" through its last content line is hashed.
- **After the hash, not before:**
  - train MoCo s2 and MAE s2 (paired with A3);
  - then ladder subset (a) on all six baselines.
  - Before launching either, free memory and load are reported, and the user is told first if other
    sessions are loading the machine heavily.
