# Ladder v2 — the full experimental ladder on A1, A2, A3

**Status.** Plan approved by the user, 2026-10-13: build and dry-run only; the ladder runs on a second approval.

## Context
M was trained on band-misregistered stamps; A1/A2/A3 are M's exact recipe on the band-aligned corpus
(pretrain_v2, seeds 0/1/2). The whole experimental ladder is to be re-run on all three, mechanically
and blind: no branching on intermediate results, nobody reads results until the manifest has run,
technical failures recorded and skipped around, never "fixed" by changing method without approval.
The §6 rehearsal and all baseline work are held.

**Blinding.** This session's main context already holds earlier results, so: the inventory was built
by a fresh subagent under the user's blinding rules (contaminated sources below); this plan cites no
earlier result; the end-of-ladder report is written by a fresh subagent from the result files alone.

## Task 0 — MAE decoder init to He et al.'s (first action on approval; no training)
Reference: `facebookresearch/mae` `models_mae.py`, `initialize_weights` + `_init_weights`: every
`nn.Linear` xavier_uniform weight, zero bias; `nn.LayerNorm` weight 1, bias 0; `mask_token`
`normal_(std=.02)`. Our `_Attention.qkv` is the same fused `Linear(dim, 3·dim)` as timm's, so xavier on
it matches.
- `src/galaxy_jepa/objectives/mae.py` `MaeDecoder.__init__`: mask token `nn.init.normal_(std=0.02)`;
  `self.apply(_he_init)` on the **decoder only**. The encoder is built first from the seeded stream and
  untouched, so its init stays byte-identical to M's (the existing test keeps that pinned).
- Test: every decoder Linear weight within xavier's bound ±√(6/(fan_in+fan_out)) with std ≈ √(2/(fan_in+fan_out)),
  biases exactly 0, LayerNorm 1/0, mask-token std ≈ 0.02; encoder-identity test still passes.
- `artifacts/baselines.md` §4.2 "decoder init" row rewritten to He's init (replacing D2), citing the file.
- Commit on `main` after the credential check. Rehearsal and baselines stay held.

## Task 1 — Inventory (blind subagent; full text → `docs/ladder_v2/inventory.md` on approval)
Plan documents: `docs/galaxy-jepa-spec.tex` (`b6ec7a98…`, PDF `151c4612…`), `docs/probing-harness-design.md`
(`6ac2f62f…`), `PROJECT_PLAN.md` (`e356b2d1…`), `TODO.md` method items, DECISIONS D13/D14/D23–D28,
`artifacts/interp_tooling.md` (DD pre-registrations), `configs/probe.yaml` (`c0ddb4e3…`),
`configs/effect_floor.json` (value 0.7267).

| id | experiment | driver (SHA-1 prefix) | granularity | one-encoder compute |
|---|---|---|---|---|
| E0 | headline single-feature probe | `harness.evaluate_probe` | 1 feature | ~0.5 h MPS |
| E1 | full-split embedding bank (block 11, mean-pooled) | `aligned_c2.py embed` (`029adeb5`) | per encoder | ~0.5 h MPS (exists for A1–A3, SHA-1s match) |
| E2 | 37-answer AUC, full split | `aligned_c2.fit_scores` | per answer | ~0.5 h CPU |
| E3 | matched evaluation O1 + the union bank (real + untrained s0) | `o1_matched.py --tag` (`6ffd24bb`) | 6 features × 6 matches | ~0.4 h MPS + 0.2 h CPU |
| E4 | untrained bank K = 30 (D23's bar) | `p1_untrained_bank.py` (`b04721ca`) | shared | ~5–6.5 h MPS, once |
| E5 | **existence ladder, Scheme 1**: 3C battery (50 shuffled + 50 Gaussian nulls, noise, untrained), D23 `untrained_z` vs K = 30, BY over 37, effect floor 0.7267, D24 nuisance clearance, MLP capacity sweep (R4) | `t1_ladder.py` (`c9f70847`) → `probing/ladder.run_ladder` | per answer × {full, conditional} | ~0.2 h MPS + 3–6 h CPU |
| E6 | entanglement: Marchenko–Pastur edge, Gram rank, probe-direction geometry, logistic-vs-CAV, D25 pair adjudication, vote-structure comparison, bar×winding | inside E5 (`entanglement.py`) | pairs | — |
| E7 | Brief R: untrained seeds 1–2 bank; r0, r3 orientation instrument, r1 MLP headroom + cluster control, r2 centroid path/bend | `r_nonlinear.py` (`9e7aeaa2`), `r2_reach_split.py` | per answer × pop | 0.4 h MPS (bank, shared) + 4–7 h CPU |
| E8 | spiral bend vs inclination (b/a) S1 | `s1_spiral_inclination.py` | 1 axis | 0.3 h CPU |
| E10 | bar×winding / bar×arms cosines U1 | `u1_bar_winding.py` | pairs | minutes |
| E11 | uncertainty geometry U2 (on/off-axis, visibility partials, 10,000 perms, BY) | `u2_uncertainty.py` | R1/R2 answers from E5 | 1–3 h CPU |
| E12 | **Scheme 2** graded existence U3 / D26 (Jonckheere–Terpstra, BY n = 4) | `u3_graded.py` | 4 axes | 0.5 h CPU |
| E13 | V1 suppression / escape | `v1_loose_ends.py` | see decision 4 | 0.2 h |
| E14 | V2 vote-independent: inclination proxies, axis-ratio direction test (edge-on × cigar), Simard B/T | `v2_independent.py` | — | 0.5 h CPU |
| E15 | V3 own directions: PCA, PR, salience vs 10,000 shuffled CAVs, principal angles | `v3_own_directions.py` | — | 0.5–1 h CPU |
| E16 | W2 naming PC1/PC2 | `w2_name_pcs.py` | — | 0.5 h CPU |
| E17 | X1 handedness / pose symmetry | `x1_handedness.py` | — | 0.1 h MPS + CPU |
| E18 | Y3 pitch angle T1–T5 | `y3_science.py` | — | 0.2 h MPS + CPU |
| E19 | Z3 B/T 2×2 | `z3_bt.py` | — | 0.3 h CPU |
| E20 | AA1 label efficiency | `aa1_label_efficiency.py` | 7 n × 5 × 37 × 4 | 1–2 h CPU |
| E22 | AA3a pose code | `aa3a_pose.py` | — | 0.5 h |
| E23 | AA3b dihedral pose averaging | `aa3b_pose_average.py` | — | 1.6 h MPS + CPU |
| E24 | **DD interpretability**: SAE tokens, TopK SAEs {11, 6} × {8×, 16×}, S3 faithfulness, S1 offset latent, 4b, V3′ maps, Part 1b | `dd_*` chain (env `DD_*`) | per encoder | 5–7 h mostly MPS |
| E26 | Plate export (what Almagest reads) | `plate_export.py` | per encoder | 0.5 h CPU |

Not re-run (encoder-free or one-off): N1/N2 (floor frozen), J/K/L/O3 trajectory diagnostics, leakage
audit, BB/BB2, distractor sensitivity (v1 by design), Z1, AA2, Y1/Y2. Excluded as comparison-with-M
(the separate later step): E25 (aligned criteria c1a/c2/c3, A3 report).

Dependencies: E1 → E2, E26. E3 → E5, E7, E20, E24. E4 + E7-bank → E5. E5 → E7 r0, E11, E12, E10, E14,
E15. E7 r3 → r2 → E8, E11. E15 → E16 → E17 → E22 → E23. E12 → E18 → E19.

**Never implemented / divergent (flagged, not fixed):** 8×8 ablation (D11); untrained-MLP bar (D25, so
R3 is unreachable); Tracy–Widom (raises); the vote-floor sweep [1, 5, 11, 21, 37]; **inclination
conditioning "recoverability/entanglement vs b/a"** (only proxies E8, E14 exist); **confound 2
(recoverability vs imaging depth) has no dedicated test**; Scheme 2 as a 10-family ladder (package
raises `GradedExistenceTestUndecided`; only U3's four axes exist); existence is D23's `untrained_z`, not
the spec text's "max of four nulls"; matching always runs (spec says "triggered"); the package's own
uncertainty geometry is off (U2 replaces it); free probe replication (1B), confidence-as-target (D9),
GalaxyMNIST, the supervised label-efficiency arm. The package battery (`probe_frozen_checkpoint`) has
never produced the catalogue and has no CLI.

## Contaminated sources (seen by the inventory subagent; not used)
`docs/probing-harness-design.md:161-168,196-221`; `docs/galaxy-jepa-spec.tex` ~736-744, 760-788,
896-898; `TODO.md` ~410-414, 434-462, 484-492, 908-910, 937-959; `configs/effect_floor.json` (all but
`value`); `configs/pretrain_v2.yaml` comment blocks ~1-7, 75-175; `artifacts/aligned_comparison.md`
~101-109, 133-141, 199, 214-228, 245-262, 281-290, 336-340, 528-543, 583-589, 614-647, 825-827;
`artifacts/interp_tooling.md` 125-136, 159-160, 197-213, 378-381, 405-410, 672-681, 725-747, 776-777,
802-820; `artifacts/distractor_sensitivity.md:20-23`; `artifacts/framing_a_claims.md:1-31`; result
remarks in the docstrings of `probing/nulls.py`, `ladder.py`, `controls.py`, `matching.py`, and of
`p2_ladder`, `j4_spread_controls`, `p1_untrained_bank`, `o1_matched`, `n1_controls`, `l1b_mlp_ladder`,
`o3_overfit_gate`, `k2_trajectory_probe`, `r_nonlinear`, `s2_verify`, `t1_ladder`, `t2_other`,
`x1_vector_posthoc`, `z3_bt`, `aa3a_pose`, `plate_export`. Avoided entirely: every `*_findings.md`,
result JSON/NPZ, run logs, A3's report, the memory files.

## Task 1b — Code changes the ladder needs (branch `ladder-v2`; the user expected none — these are why)
The inventory found the drivers are wired to M. Without these, a run would overwrite or skip M's
records, read M's outputs as inputs, or silently use the wrong control. All are plumbing or
fail-loudly fixes; none changes a statistic. Each lands with a test, then the full suite.
1. **One encoder/output contract.** `artifacts/ladder_env.py`: `encoder()`, `tag()`, `out_dir()`
   from `LADDER_ENCODER` / `LADDER_TAG` / `LADDER_OUT`, defaulting to today's M values (so M's runs
   stay reproducible). Every hard-coded `R.prepare("runs/m/encoder.pt", …)` (25 sites, e.g.
   `t1_ladder.py:49`, `u2_uncertainty.py:232`, `v3_own_directions.py:103`, `aa3b_pose_average.py:399`),
   the `--checkpoint` defaults (`p2_ladder.py:105`, `r_nonlinear.py:654`), the fixed bank names
   (`r_nonlinear.py:62-63`) and every `OUT/<name>.json` write/read go through it; outputs land in
   `runs/ladder_v2/<experiment>/<encoder>/` (on the SSD), never `artifacts/out`.
2. **Fail-loudly fixes.** `ladder._load_untrained_bank` (`ladder.py:113-145`) checks the bank's model and
   split key and refuses a mismatch (package code; invariant test). `s2_verify.py:41-44`'s noise bank is
   keyed by the encoder's checkpoint SHA-1 (today A2/A3 would get A1's noise control).
3. **Blinding.** `t1_ladder.main`'s `compare()` against M's records skipped when `LADDER_*` is set.
   `aligned_score`-style M reads are not in the manifest.
4. **D27.** `u2_uncertainty.py:51` `REVERSAL_STATE = False` → the REVERSES state, as D27 requires for
   future runs.
5. **v1 metadata paths** in encoder-dependent steps (`v2_independent.py:56`, `dd_sae_cards.py:43`,
   `dd_v3.py:75`) read the active probe corpus's metadata (probe_v2's carries `expAB_r`/`deVAB_r`).
6. The runner and the manifest generator (Task 2).

## Decisions (user, 2026-10-13)
1. **Probe split: (a)** the identical set — v1's 40,000-stride and test split, minus the 9 galaxies
   probe_v2 lacks (`aligned_score._identical_prepare`).
2. **Untrained bar: (a)** build the v2 untrained bank (K = 30), once.
3. **Package battery (E27): (a)** excluded; recorded under "not implemented".
4. **M-chosen target lists: (a)** every target list chosen from M's results is excluded. Where a driver
   derives its targets mechanically from the new encoder's own upstream output by a rule fixed in code
   (e.g. U2 from E5's R1/R2), that is used. Any experiment that can only run on an M-chosen list is
   excluded entirely and listed.
5. **DD: (a)** E24 rerun for A1, A2 and A3 into `runs/ladder_v2/`; the aligned-scoring DD runs stay
   untouched.
6. **Untested items** reported as "not implemented", not built.

**Additions (user, 2026-10-13).**
- **The one non-plumbing change: D27.** `u2_uncertainty.py` `REVERSAL_STATE` → the REVERSES state, as
  D27 requires for future runs (a state added after a result applies to future runs only). Every other
  change in Task 1b is plumbing or a fail-loudly fix.
- **Inertness test for 1b:** with `ladder_env` defaults (M), rerun E10 (`u1_bar_winding`) and confirm its
  output is byte-identical to M's existing output file by SHA-1 only (the file is neither opened nor
  printed). Reported pass/fail only.
- **Baselines train on pretrain_v2** (separate branch, while the ladder runs, no GPU use): configs
  regenerated from `pretrain_v2.yaml`; `baseline_run.py` matches A1's config and the v2 env; seed 0 pairs
  with A1, seed 1 with A2; σ_sky re-measured on the v2 cache (same method); k2 trajectory probes use the
  ladder's E2 protocol; F6 recorded as moot on aligned stamps (the top-1 gate kept as a pretext-ease
  check); `baselines.md` §1, §5.3 and §7 updated. The rehearsal is held until the ladder has finished.

## Task 2 — Manifest and runner (written on approval; run only on a second approval)
`docs/ladder_v2/manifest.yaml`, generated by `artifacts/ladder_v2_manifest.py` from the inventory, one
entry per (experiment × encoder [× population/scheme]), shared entries once:
```yaml
- id: E5.t1_ladder.A2          # experiment.step.encoder
  cmd: [artifacts/t1_ladder.py]
  env: {LADDER_ENCODER: runs/m_v2_s1/encoder.pt, LADDER_TAG: A2, LADDER_OUT: runs/ladder_v2/E5/A2,
        GJ_CONFIG: configs/pretrain_v2.yaml, F0_CACHE_BASE: ..., F0_NORM_PREFIX: 246d8de6ba9a}
  pins: {artifacts/t1_ladder.py: <sha1>, configs/probe.yaml: <sha1>, runs/m_v2_s1/encoder.pt: <sha1>}
  inputs: [E3.o1.A2, E4.untrained_bank, E7.bank]     # entry ids; their output SHA-1s are verified
  outputs: [runs/ladder_v2/E5/A2/t1_ladder.json]
  heavy: cpu            # mps | cpu
  status: pending       # pending | running | done | failed | skipped(dep)
```
Order: shared (E4, E7-bank) → per encoder A1, A2, A3 in dependency order: E0, E1 (reuse if bank SHA-1
matches the checkpoint, else re-embed), E2, E3, E5/E6, E7 r3→r2→r1→r0, E8, E10–E20, E22, E23, E24, E26.

Runner `artifacts/ladder_v2_run.py`:
- strictly in order, one heavy job at a time under `artifacts/_heavy.sh`'s lock; each entry a
  subprocess (the 18 GB rule: one subprocess per step, memory returned between steps);
- resumable: state in `docs/ladder_v2/state.json` (atomic rewrite), starts at the first non-done entry;
  a `running` entry on restart is re-run from scratch into a cleared output dir;
- before an entry: verify every pin and every input entry's recorded output SHA-1; after: SHA-1 every
  output into the state and `docs/ladder_v2/outputs.sha1`;
- on failure: mark `failed` with exit code and the last 50 lines of stderr **written to the entry's own
  log file**, mark dependants `skipped(dep)`, continue;
- each entry's stdout/stderr go to `runs/ladder_v2/logs/<id>.log` (never displayed); the run log
  `docs/ladder_v2/run_log.md` and the console carry only id, status, wall-clock, peak RSS, output paths;
  progress `k/N` + current id to the status line via `_detach.py`; **no result value is printed,
  summarised or logged by the runner**;
- launched detached via `_detach.py` with the watchdog.

## Task 3 — Estimates (from code and data sizes; ±50%)
| | MPS | CPU | peak RAM |
|---|---|---|---|
| shared once (E4 bank, E7 bank) | ~6–7 h | — | ~4 GB |
| per encoder (E0–E26 as above) | ~8–10 h | ~15–25 h | ladder 3–4 GB; embeds 4.2 GB footprint; SAE ~1.6 GB; plate 2–3 GB |
| **total, strictly serial** | | | **~80–110 h ≈ 3.5–4.5 days** |

Order for memory safety: strictly serial (no CPU stage overlapping an MPS stage) — peak stays ≤ ~4–5 GB
on the 18 GB machine; MPS pool emptied between entries by process exit. Disk: DD tokens ~10 GB per
encoder (SSD has ample space).

## Task 4 — Report format (written at the end by a fresh, blinded subagent from result files only)
`docs/ladder_v2/report.md`: per experiment, the ladder's own criteria, A1 | A2 | A3 side by side —
the state (e.g. R1/R2/R4, clean/marginal, SURVIVES/COLLAPSES, PASS/FAIL), the statistic with its CI and
the null/floor it was judged against, per seed. An explicit **agreement column**: `3/3` or `2/1 (A? differs)`,
never a mean across seeds. A front matrix (answers × experiments, states per seed), a technical-failures
section (entry, error, skipped dependants), a not-implemented section (decision 6), and every number
traced to its output file and SHA-1. No comparison with M.

## Task 5 — Baselines on pretrain_v2 instead of v1 (costed; decide later)
Compute is unchanged (same model, steps 101,308, batch 32): rehearsal ~1 h, F7 ~8 h, four runs
~107–123 h. Build changes ~0.5 day. What assumes v1 today:
- `configs/moco.yaml`, `configs/mae.yaml`: copies of `pretrain.yaml` (v1 paths, freeze `75100066…`) →
  regenerate from `pretrain_v2.yaml` (freeze `246d8de6…`); the equality test then pins against A's config.
- `artifacts/baseline_run.py`: forces `GJ_CONFIG=configs/pretrain.yaml`, refuses `F0_*`, matches M →
  match A1's config, v2 env; seed 0 pairs with A1, seed 1 with A2 (seed 2 with A3 would add ~50 h).
- σ_sky measured on the v1 cache (`artifacts/sky_sigma.py`, values in `MocoConfig` and `moco.yaml`) →
  re-measure on the v2 cache (minutes; same method).
- The k2 trajectory probe (`k2_trajectory_probe.py:74`) is a v1 like-for-like guard and refuses probe_v2
  → trajectory probes use the ladder's E2 protocol instead.
- Rehearsal: the reference arm becomes A1's recipe; the top-1 gate's named risk (band-offset shortcut,
  F6) is largely removed on aligned stamps — the gate stays as a pretext-ease check; F6's measurement
  becomes moot.
- `baselines.md` §1 table, §5.3 pairing and §7 rows 14 (band misregistration) rewritten.

## Verification
- Task 0: new init test + full suite, ruff, mypy; encoder-identity test unchanged.
- Task 1b: each fix with a test (bank-key refusal; noise bank keyed by encoder; `ladder_env` defaults
  reproduce M's paths byte-for-byte); full suite.
- Runner: a dry run (`--plan`) that resolves every entry, verifies every pin, and prints the order and
  estimates without executing; a 2-entry plumbing run on a scratch manifest (tiny CPU step) proving
  resume, failure-skip and SHA-1 pinning.
- Then stop and report; the ladder itself runs only on a second approval.
