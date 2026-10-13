# Ladder v2 — inventory (blind subagent, 2026-10-13)

Built by a fresh subagent with no access to the session's history, under the user's blinding rules:
from plan documents, configs, code and file metadata only. It contains no scores, verdicts or
comparisons from any earlier encoder. Compute figures are estimates from code and data sizes plus two
resource facts (an embedding pass of 195,677 stamps ≈ 30 min MPS at ≈ 2 GB RSS / 4.2 GB footprint;
37 logistic fits on 160,849 × 384 for five encoders ≈ 15–20 min CPU). SHA-1 prefixes are 12 hex
characters, as of 2026-10-13. "Union" = the 40,000 capped train subset (H5's deterministic stride) +
the probe-test split (on v2's own stride: 40,000 + 34,828 = 74,828). The approved plan
(`plan.md`, decision 1) uses the identical set instead.

## 1. Plan documents
| Role | File | SHA-1 |
|---|---|---|
| Science spec | `docs/galaxy-jepa-spec.pdf` | `151c46124015442d8898c3c47a42ad4220fcc5cd` |
| Spec source (read) | `docs/galaxy-jepa-spec.tex` | `b6ec7a9821aa46d25f9d14621ed619fdcd062bfc` |
| Locked probing design (3A–3D controls, 2A–2F ladder, uncertainty geometry, full run, figures) | `docs/probing-harness-design.md` | `6ac2f62f3672202cc282f1625ecd3b671ad82e40` |
| Phases, deliverables, sanity gates | `PROJECT_PLAN.md` | `e356b2d14cdd5146f6ee716d6c926455f566e504` |
| Method backlog (Epic F, Brief P, Epic R) | `TODO.md` | `59e948c6dd41a8bc20a8d78e77ef267931a3c23f` |
| Method decisions D13, D14, D23–D28 | `DECISIONS.md` | `9a53c6bb65c0048526fe7356bd0579703f3417ae` |
| A1/A2/A3 pre-registration | `artifacts/aligned_comparison.md` | `ab9f812a1b0ccdc40887b8433d8dedfb03d6392f` |
| DD interpretability pre-registrations (V1–V4, S1–S3, Part 4/4b) | `artifacts/interp_tooling.md` | `9d8e1255f121d1373a254a713d309076f5fb7586` |
| Probe config | `configs/probe.yaml` | `c0ddb4e322b70e90adfaf6b2d7760ba42003d28b` |
| Frozen effect floor (only `value` used) | `configs/effect_floor.json` | `e6b8b8ab7f0f0a9fea733a5d7d68135098b43cd1` |

Briefs S–BB's pre-registrations live inside `*_findings.md` (off limits); their method was taken from
script docstrings and code only.

## 2. Shared setup
- Split, labels, dataset: `artifacts/j4_spread_controls.py:128 prepare()` (`305a01847c4b`): reads
  `configs/probe.yaml`, scheme, vote floor 1, consensus gate 0.5; the 40,000-stride train cap and the
  full test split.
- Environment: `artifacts/f0_preconditions.py` (`b5491b4f50f4`) reads `GJ_CONFIG`, `F0_CACHE_BASE`,
  `F0_NORM_PREFIX`; defaults to v1 (lines 25, 26, 31). `artifacts/aligned_launch.sh` (`e7566a33ef4e`)
  sets the v2 values (`configs/pretrain_v2.yaml`, `runs/m_v2/cache`, `246d8de6ba9a`).
- Banks: `artifacts/probe_bank.py` (`b5b854f9011b`) `write_bank` / `load_bank` record and check the
  checkpoint SHA-1. On disk (metadata only): `c2_a1/a2/a3_embeddings.npz` (195,677 × 384 float32, 302 MB
  each; checkpoints `34bb30a74d38…` `runs/m_v2/encoder.pt`, `380cdb7b67a9…` `runs/m_v2_s1/encoder.pt`,
  `4c3e1c7e88f6…` `runs/m_v2_s2/encoder.pt`, all matching today); `a1_embeddings.npz`,
  `a2_embeddings.npz` (real + untrained, union). A3 has no union bank and no DD run.

## 3. Experiments
**E0 Headline single-feature probe.** Featured-ness consensus extremes (spec Stage 2, PROJECT_PLAN P6).
`harness.evaluate_probe` (`harness.py:843`, `326bb642e89f`) / `artifacts/h5_probe_lean.py`
(`703bcecb093d`). Input: checkpoint, probe train/test, t01 a02 fraction at 0.2/0.8. Output `report.json`.
Bootstrap CI. ~30 min MPS. A1–A3 were trained `--no-probe`.

**E1 Full-split bank.** `aligned_c2.py embed <tag> <ckpt>` (`029adeb522a3`) → `c2_<tag>_embeddings.npz`
(block 11, mean-pooled). Feeds E2, E25, E26. ~30 min MPS. Exists for A1–A3.

**E2 Per-answer AUC, 37 answers, full split.** `aligned_c2.fit_scores` (via `aligned_score.py c2`,
`acf02d30450c`): E1 bank, `_fit` C = 1, 160,849 / 34,828. 2,000-draw Poisson galaxy bootstrap (weights
shared across answers/encoders); BY over the powered answers at q = 0.05 (powered: min class ≥ 100 and
baseline AUC ≥ 0.6, fixed in `artifacts/out/aligned/powered.json`). Per answer, per seed. ~3–4 min CPU
per encoder for fits; bootstrap ~10–30 min; ~2 GB RAM.

**E3 Matched evaluation O1.** `o1_matched.py --checkpoint <ckpt> --tag <t>` (`6ffd24bbb7e2`): union; six
features in `SPREAD`; five single nuisances (magnitude, size, SNR, redshift, PSF) and a joint
magnitude×size 5×5 match; quantile stratification into 5 strata, class balancing; `petrorad_suspect`
filter. O1's retention rule (SURVIVES / PARTIAL / COLLAPSES / UNRESOLVED; retain 0.5; ≥ 500 matched
test; ≥ 10% share). Outputs `<t>_embeddings.npz` (real + untrained seed 0 — the union bank everything
downstream reads) and `<t>_matched.json`. Two passes over 74,828 ≈ 23 min MPS + 5–10 min CPU.

**E4 Untrained bank, K seeds.** `p1_untrained_bank.py --seeds 30` (`b04721ca714f`): K = 30 random-init
ViTs; per seed the 37 Scheme-1 AUCs on the union; keyed on a model-config hash and a split hash. Output
`artifacts/out/p1_untrained_bank.json` (fixed path). Architecture + pixels + split only: one v2 build
serves A1–A3. ~5–6.5 h MPS, resumable per seed. The existing bank is v1.

**E5 The ladder, Scheme 1, full and conditional populations.** `p2_ladder.py` (`9bf8ff5bb888`, original)
/ `t1_ladder.py` (`c9f70847fc17`, production: D24 retention clearance + D25 MLP decode unadjudicated) /
`s2_verify.py` (`b4fb99a60533`, D24 verification); all call `probing/ladder.run_ladder`
(`672428fd1984`). Phases: (1) per answer L2 logistic (C = 1), held-out AUC, bootstrap SE (2,000);
(2) the 3C battery (`controls.py`, `65ea392e90ea`): 50 shuffled-label + 50 random-Gaussian nulls, noise
through the encoder, untrained seed 0, the sky/noise label (diagnostic, D19), selectivity = AUC − mean
shuffled (gate 0.10), the five-nuisance median-split panel; (3) existence = `untrained_z`
(`nulls.py:208`, `10c7f06d43bb`): z = (A − mean C)/√(sd C² + se²), t with df = K − 1, refuses K < 20,
BY α = 0.05 over the scheme's 37, then the effect floor 0.7267 (`floor.resolve_effect_floor`, clean vs
marginal), Shapiro–Wilk reported, `resolvable_margin` / `underpowered` on every rung; (4) entanglement
(E6); (5) nuisance clearance (D24): always matches on the worst nuisance, C and C_m averaged over 3
untrained draws; (6) existence failures → MLP capacity sweep (widths 16–512, depth 1, 200 epochs;
ceiling at the 0.95 quantile of shuffled nulls) → R4 "MLP decode unadjudicated". Inputs: union bank
(E3), untrained seeds 1–2 (E7 bank); t1/s2 bank the noise matrix as `s2_noise.npz`. Outputs (fixed,
untagged): `p2_ladder.json`, `s2_ladder.json`, `t1_ladder.json`. ~12 min MPS noise pass + 3–6 h CPU;
3–4 GB RAM. `t1_ladder.main` ends by calling `compare()` against M's records (flagged).

**E6 Entanglement: Marchenko–Pastur, probe-direction geometry.** Inside E5 (`entanglement.py`,
`1cc5f7e36ba0`; `p2_ladder.py:202-247`): cosine and Gram matrices of existence-passing unit directions;
Gram effective rank and loadings; embedding covariance spectrum and span ratio; the MP upper edge
(1 + √(k/D))² for k × 384 (one global test; no Tracy–Widom); logistic-vs-CAV disagreement; the top 10% of
pairs by |cos| to D25 retention matching (A matched on B's vote fraction), then `adjudicate_pair`. Also
`human_vote_correlation` / `compare_to_human_structure` and `bar_winding_alignment` (D13 stage 2).

**E7 Brief R: nonlinear structure and geometry.** `r_nonlinear.py --stages bank r0 r3 r1 r2`
(`9e7aeaa276a4`). bank: untrained seeds 1–2 over the union → `r_untrained_seeds.npz` (shared; ~23 min
MPS). r0: re-adjudicates P2's matched verdicts under the retention rule (reads `p2_ladder.json`;
superseded by D24). r3: orientation-circle validation of the instrument (stamp second moments,
`expAB_r` ≤ 0.6 and moment q ≤ 0.8; `geometry.curvature_test` null 200 / 4,000 draws, BY; 30–60 min CPU).
r1: MLP headroom on 37 × 2 populations with a k-means control task (100 and 1,000 clusters; 2–4 h CPU).
r2: centroid path E[z | vote fraction] (10 bins, split-half cross-fitting, shuffled-value null 200 /
4,000, BY; "curved" needs bend ≥ 0.10; 1–2 h CPU). Outputs `r_r0..r3.json`, `r3_orientation.npz`,
`r1_clusters.npz` (untagged; a stage is skipped if its record exists). `r2_reach_split.py`
(`c8e3633d4245`): descriptive split of bend by vote reach.

**E8 S1 spiral bend vs inclination.** `s1_spiral_inclination.py` (`f2063bf02b89`): bend direction from
R2's covariance on train; partial correlation with `expAB_r` after removing 20 bin means; 10,000-perm
null, 2,000 bootstrap; full and well-voted half; encoder + 3 untrained. Depends on E7 r2, E3, E7 bank.
10–20 min CPU.

**E9 T2.** `t2_other.py`: a follow-up on one answer chosen from M's ladder output. Not generic.

**E10 U1 bar×winding / bar×arm-count cosines.** `u1_bar_winding.py` (`1d651c5e6988`): encoder cosine vs
the human vote correlation on the same test galaxies; null band = max(isotropic sd 1/√384, 30
shuffled-partner fits); untrained draws descriptive. Minutes.

**E11 U2 uncertainty geometry.** `u2_uncertainty.py --counts` then default (`f5de7ee39851`): answers R1/R2
in E5's `t1_ladder.json`, both populations; local reach floor 21 (10, 37 sensitivity); on-axis: logistic
on train extremes (≥ 0.8 vs ≤ 0.2) projected on the test middle (0.2–0.8), Spearman; off-axis: bend
coordinate from `r_r2.json` and distance from the extreme-centroid line vs ambiguity; visibility:
Freedman–Lane partial Spearman controlling magnitude, SNR, size, redshift; 10,000 perms, BY, `N_MIN` from
a power formula. 1–3 h CPU. The package's own Phase 6 is off (`probe.yaml uncertainty_geometry: false`).

**E12 U3 graded existence, Scheme 2 (D26).** `u3_graded.py` (`808687d15983`): four ordered axes,
plurality category, reach ≥ 21. Test A: endpoint-fit axis, Jonckheere–Terpstra as Kendall τ-b, 10,000
one-sided perms, BY n = 4, untrained ×3. Test B: Spearman vs Masters' w_avg and B_avg. Test C: where the
middle sits vs a 200-draw shuffled null, 1,000 bootstraps. 20–40 min CPU.

**E13 V1.** `v1_loose_ends.py` (`467112f52a66`): suppression check on two answers named from U2 (10,000
Freedman–Lane perms); escape test (votes vs photometry, no embedding). ~10 min.

**E14 V2 vote-independent measurements.** `v2_independent.py` (`5b8eb4a2e459`): ridge (GCV) decoding of
`expAB_r`, `deVAB_r`; logistic direction on vote labels vs base-rate-matched axis-ratio labels for
edge-on and cigar; Simard B/T (`out/ext/simard2011_table1.tsv`) with 2,000 bootstrap and BY. 20–40 min CPU.

**E15 V3 own directions.** `v3_own_directions.py` (`2eb8401a88aa`): PCA on train, PR, k* = round(PR);
salience = CAV energy in the top-k* subspace vs 10,000 shuffled-label CAVs, BY; principal angles between
concept/nuisance spans and that subspace; held-out single-PC test; discovery grids (PNG). Reads
`r3_orientation.npz`. 30–60 min CPU.

**E16 W2 naming PC1/PC2.** `w2_name_pcs.py --planted/--measure/--run` (`abf5cefe4340`): stamp pass
measuring padding, sky, noise, crowding, colour; 10,000-perm BY tests. ~30 min CPU.

**E17 X1 handedness / pose symmetry.** `x1_handedness.py --planted/--bank/--run` (`cfd43eb4f64d`): the X1c
sample of 2,000 under 4 transforms through encoder + 3 untrained (~5 min MPS); GZ1 table 2; 1,000
bootstrap, BY. `x1_vector_posthoc.py` is exploratory and M-specific.

**E18 Y/Y3 pitch angle.** `y_pitch.py` (`12e787a006a0`), `y3_science.py --bank/--planted3/--y3`
(`f25bfeb8deca`): banks Hart and Yu & Ho galaxies outside the union through encoder + 3 untrained
(~10 min MPS); T1–T5 vs Hayes SpArcFiRe pitch, BY, 10,000 perms, a 200-draw shared null. `--y1/--y2` are
encoder-free.

**E19 Z.** `z_checks.py` (Z1) encoder-free. `z3_bt.py` (Z3, `8129c9a3d979`): B/T 2×2 with Y3's
`two_by_two` and a shared null. 10–20 min.

**E20 AA1 label efficiency.** `aa1_label_efficiency.py` (`04d0eb19b04a`): n ∈ {100, 300, 1k, 3k, 10k, 30k,
full} × 5 subsets × 37 answers × (encoder + 3 untrained); ~5,200 fits, 1–2 h CPU. Supervised arm not
implemented.

**E21 AA2 DECaLS coverage.** `aa2_decals.py`: encoder-free.

**E22 AA3a pose code.** `aa3a_pose.py` (`95904aef99b4`): per-band centroids of the X1c stamps on the
active cache; causal arm: 200 g-shifted stamps re-embedded; 1,000 perms, BY. 15–30 min.

**E23 AA3b dihedral pose averaging.** `aa3b_pose_average.py --bank/--planted/--aa3b` (`74c2cdd0a292`):
the union under 8 transforms (~600,000 forwards, 90–100 min MPS, ~460 MB fp16 memmaps); existence,
salience and winding re-read-outs (winding needs Y3/U3).

**E24 DD interpretability (per encoder).** Chain as `runs/aligned_score/chain_resume3.sh`, env `DD_TAG`,
`DD_ENCODER`, `DD_OUT=runs/dd_<e>`, `DD_BANK=artifacts/out/<e>_embeddings.npz`, `DD_LOCAL=runs/dd_local_v2`:
(1) `dd_sae_score.py plants` (`ce7f728fba05`); (2) `dd_sae.py extract` (`a0436325f756`): block 11 and 6
tokens for the sae (20,000) and sae_eval (4,999) sets, fp16, ~9.8 GB per trained encoder; (3) `dd_sae.py
train <enc> <layer> <mult>`: TopK SAEs at {11, 6} × {8×, 16×}, k = 32, AuxK fixed-shape, 8 epochs;
(4) `evaluate` × 8 (S3 faithfulness, scored by powered-UNEVEN); (5) `dd_sae_score.py score`: S1 offset
latent (|ρ| ≥ 0.5, untrained match < 0.2), width 8× unless 8× fails and 16× does not; (6) `dd_part4b.py
flags|stats|pools|plants|run` (`c4fec662cc3a`): 50 energy-matched random sets, 95th-percentile states,
INAPPLICABLE branch, 8× only; (7) `dd_part1.py v3` (`9981cc744ff2`): V3′ token-drop maps on 794 GZ3D
galaxies with v2-reprojected labels (`runs/dd_local_v2/v3_patch_labels.npz`); (8) `dd_part1b.py
direct|plants|v3` (`c7dca4bad08b`): ring-stratified AUC, 10,000 bootstrap. Untrained SAEs and tokens are
shared (in `runs/dd_a1`). Not in the chain: V1, V2, V4, S2, cards, Part 2, `dd_s2_probe`. M-specific:
`dd_part4.py` (latent 328), `dd_l2336_scan.py`. ~5–7 h (extract 5–10 min MPS; 4 SAE trainings 16–35 min
each; 8 evaluations ~1 h; scoring + 4b 0.5–1 h; V3′ 1–2 h MPS). SAE training ~1.4–1.6 GiB driver.

**E25 Aligned criteria.** `aligned_score.py powered|embed-m1v2|c1a|c2|c3`, `aligned_c13.py`
(`14361e7ac657`): the comparison with M — excluded from ladder v2.

**E26 Plate export (Almagest).** `plate_export.py subset|pcasweep|basiseval|export|testfile|sanity`
(`e37b1954f71f`): `plate.bin` v2 (`artifacts/plate_data_format.md`) from the E1 bank, 37 `_fit` probes on
the full train split (as E2), a MiniBatchKMeans(400) subset with rare-class oversampling, a basis of 37
QR-orthogonalised probe directions + residual PCs, UMAP and 6-NN, raw GZ2 fractions/counts,
`modelMag_r`, `expAB_r`, ra/dec. Depends on E1 (and E2's protocol) only; M-only by hard-coding.
20–40 min CPU, 2–3 GB RAM.

**E27 Package battery.** `harness.probe_frozen_checkpoint` (`harness.py:926`) → `probing/run.run_probing`
(`4ac720c4c179`): no CLI; all ~230k probe galaxies, 70/15/15, real + untrained ×3 + noise, ladder on full
and conditional, gated uncertainty geometry, figures, `ladder_summary.json`. Never produced the
catalogue. ~3 h MPS + 12–24 h CPU, 6–8 GB RAM. Excluded (decision 3).

Not per-encoder: N1/N2 (floor frozen); J, K, L, O3, `m_band_axis` (trajectory/recipe diagnostics;
`m_band_axis` reads v1 `aa3a_offsets.npz`); leakage audit; BB/BB2; distractor sensitivity (v1 by
design); unimplemented plan items (8×8 ablation, β-sweep, cross-objective ladder).

## 4. Dependency graph
```
v2 cache + freeze
 ├─ E1 bank ─┬─ E2 ── (E25, excluded)
 │           └─ E26 Plate
 ├─ E3 o1_matched → union bank (real + untrained s0)
 │    ├─ E24 DD chain
 │    └─ E7 bank (untrained s1, s2; shared) ─┐
 ├─ E4 untrained bank K = 30 (shared) ───────┤
 └───────────────────────────────────────────┴─ E5 ladder [+ E6]
        ├─ E7 r0, r3 → r2 → r1
        │     └─ E8 S1 ── E11 U2 (also r_r2.json) ── E13 V1
        ├─ E11 U2 (R1/R2 from t1_ladder.json)
        ├─ E12 U3 ── E18 Y3 T4 ── E19 Z3
        ├─ E10 U1, E14 V2, E15 V3 (r3_orientation.npz) ── E16 W2 ── E17 X1 ── E22 AA3a ── E23 AA3b
        └─ E20 AA1 (union bank + E7 seeds)
```

## 5. Flags
**Hard-coded to M/O2.** `R.prepare("runs/m/encoder.pt", …)`: `r2_reach_split.py:23`,
`s1_spiral_inclination.py:99`, `s2_verify.py:65,130`, `t1_ladder.py:49`, `t2_other.py:116`,
`u1_bar_winding.py:70`, `u2_uncertainty.py:232,310`, `u3_graded.py:131`, `v1_loose_ends.py:239`,
`v2_independent.py:233`, `v3_own_directions.py:103`, `w2_name_pcs.py:334`, `x1_handedness.py:663`,
`x1_vector_posthoc.py:83`, `y_pitch.py:432`, `y3_science.py:707`, `z3_bt.py:149`,
`aa1_label_efficiency.py:207`, `aa2_decals.py:320`, `aa3a_pose.py:342`, `aa3b_pose_average.py:399`,
`readme_figures.py:814`. `--checkpoint runs/m/encoder.pt` defaults: `p2_ladder.py:105`,
`p1_untrained_bank.py:76` (harmless), `r_nonlinear.py:654`. M's headline string in the record:
`p2_ladder.py:63`. Fixed bank names: `r_nonlinear.py:62` (`o1_embeddings.npz`), `:63`
(`r_untrained_seeds.npz`) — load-time SHA-1/id checks refuse a mismatch. Untagged records:
`r_nonlinear.py:153` (stages skipped if a record exists); every Brief R–AA driver writes
`OUT/<name>.json`. Reads M's records: `r_nonlinear.py:209` (`p2_ladder.json`), `t1_ladder.py:145ff`
(`compare()` vs `s2_ladder.json`, `t1_old_pairs.json`), `u2_uncertainty.py:55-56`, `u3_graded`, `v1`,
`v3` (`t1_ladder.json`, `r_r2.json`, `r3_orientation.npz`). `s2_verify.py:41-44`: the noise bank is
reused whenever the ids match, with no encoder check (A2/A3 would get A1's noise control). DD defaults:
`dd_core.py:38,39,41` (env-overridable). A3 missing from `aligned_score.py:47,49`; `:121` uses M's
checkpoint. Plate M-only: `plate_export.py:38, 108-111, 639`. `o1_matched.py:155` defaults to
`out_dir/encoder.pt` (A1 under the v2 env). Feature lists chosen from M's outcomes:
`u2_uncertainty.py:53-54` (`FLAGGED`), `v1_loose_ends.py:34` (`SUPPRESSION`), `t2_other.py:33`,
`x1_vector_posthoc.py`, `dd_part4.py`, `dd_l2336_scan.py`. v1 defaults when env unset:
`f0_preconditions.py:25,26,31`.

**v1-only inputs.** `configs/probe.yaml untrained_bank_path` is the v1 bank; `ladder._load_untrained_bank`
(`ladder.py:113-145`) never checks the model or split key. The 40,000 stride over v2's train ids differs
from M's union (52,151 shared of ~74,800); the identical-set rule is implemented only in
`aligned_score._identical_prepare`. The effect floor was frozen on the v1 40,000 / 34,829 split. v1
metadata hard-coded: `v2_independent.py:56`, `dd_sae_cards.py:43`, `dd_v3.py:75`, `z_checks.py:31`,
`readme_figures.py:32`. v1 by design: distractor sensitivity; c1/S1/4b offsets (v1's recorded offsets
from probe_v2's `cut_log`); `aa3a_offsets.npz`.

**Never built.** 8×8 ablation (2E / D11); the untrained-MLP bar (D25), so R3 is unreachable (ceiling
FLAGGED, `ladder.py:587-614`); the v1-vs-v2 comparison and v1 Figs 18–19 (4C); calibration in Fig 1;
Tracy–Widom (`NotImplementedError`); the vote-floor sweep [1, 5, 11, 21, 37]; free probe replication
(1B); confidence as a target (D9); GalaxyMNIST; the supervised label-efficiency arm; the DECaLS paired
referee; the β-sweep; the cross-objective ladder.

**Built differently from the plan's wording.** Existence: spec "beat the max of four nulls at p < 0.05";
production D23 `untrained_z` with K = 30; `n_null_draws` 50; `assert_null_resolution` bites only under
`empirical`. Matching: plan "triggered"; code always matches the worst single nuisance (joint
magnitude×size only in O1). Scheme 2: the package raises `GradedExistenceTestUndecided`
(`schemes.py:152-160`); D26 exists only in `u3_graded.py`; the 10-family Scheme 2 ladder never run;
`consensus_gate` 0.5 still OPEN. Uncertainty geometry: package version off; U2 adds a reach floor 21 and
off-axis / visibility arms. Inclination conditioning "recoverability and entanglement vs b/a" (D13,
Epic R): not written (proxies E8, E14, R3's gate, the Plate's b/a column). Confound taxonomy: confounds 1
and 3 covered only via E6, E10, E14; confound 2 (recoverability vs imaging depth) has no test.
Must change before a new-encoder run: `u2_uncertainty.py:51 REVERSAL_STATE = False` (D27: REVERSES).

## 6. Contaminated sources
Seen and not used: `docs/probing-harness-design.md:161-168, 196-221`; `docs/galaxy-jepa-spec.tex`
~736-744, 760-788, 896-898; `TODO.md` ~410-414, 434-462, 484-492, 908-910, 937-959;
`configs/effect_floor.json` (all but `value`); `configs/pretrain_v2.yaml` comment blocks ~1-7, 75-175;
`artifacts/aligned_comparison.md` ~101-109, 133-141, 199, 214-228, 245-262, 281-290, 336-340, 528-543,
583-589, 614-647, 825-827 (Results, Checks, A3 report and Plant results sections not opened);
`artifacts/interp_tooling.md` 125-136, 159-160, 197-213, 378-381, 405-410, 672-681, 725-747, 776-777,
802-820; `artifacts/distractor_sensitivity.md:20-23`; `artifacts/framing_a_claims.md:1-31`. Docstrings
with result remarks: `probing/nulls.py:64-67, 224-225, 366-371, 398-399`; `ladder.py:197-199, 446-456`;
`controls.py:36`; `matching.py:190-194`; `p2_ladder.py:63`; `j4_spread_controls.py:66, 85-90`;
`p1_untrained_bank.py:7-8, 18`; `o1_matched.py:5-8`; `n1_controls.py:6-9, 20-22`; `l1b_mlp_ladder.py:3-4`;
`o3_overfit_gate.py:5-7, 16-17`; `k2_trajectory_probe.py:3-4`; `r_nonlinear.py:8-9`; `s2_verify.py`;
`t1_ladder.py:104`; `t2_other.py`; `x1_vector_posthoc.py`; `z3_bt.py:4`; `aa3a_pose.py:3-5`;
`plate_export.py:19-23`. Avoided entirely: every `*_findings.md`, result JSON/NPZ, run logs, A3's report,
`~/.claude`.
