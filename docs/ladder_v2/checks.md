# Ladder v2 — pre-reading checks (blinded)

- **Date:** 2026-10-17
- **HEAD:** `7aed76108d2abac38a5a7cd27b83952505ff824c` on branch `ladder-v2` (working tree: only the untracked `docs/ladder_v2/runner_log`; this file is the only addition)
- **Checker:** a fresh, blinded sub-agent.
- **Method:** code and metadata only. Drivers, `artifacts/ladder_env.py`, `artifacts/probe_bank.py`, `artifacts/ladder_v2_run.py`, `configs/probe.yaml` and the probing package were read; every code file cited was first confirmed to hash to its manifest pin (so the code read is the code that ran). The manifest, `state.json` and `audit.md` were read. Outputs were opened only programmatically, extracting named fields: `smoke` / `escape_hatches` / budget fields (`K`, `max_train`, `n_boot`, `n_strata`, `limit`, `k`, SAE epoch indices), identity fields (`checkpoint`, `checkpoint_sha1`, `arm`, `encoder`), key **names** (never values) for the "M" search, the E19/E18 *planted* `T2_*` state strings, A1's `chosen` SAE width and whether A1's Part 4b records carry the `INAPPLICABLE` state. Entry logs were read only for their runner-written `# cmd:` / `# env:` header lines and for lines matching SHA/checkpoint patterns. The three encoder checkpoints were hashed (`shasum`), never loaded.
- **Blinding statement:** no result value (AUC, statistic, p, CI, real-experiment state, pass count) is recorded in this file. `report.md`, `*_findings.md`, `artifacts/aligned_comparison.md`, `artifacts/out/`, runs outside `runs/ladder_v2` (bar hashing the checkpoints), user memory and `.env` were not opened. **Disclosure:** three tool outputs during the check displayed real values the checker did not intend to see — one SHA/checkpoint grep over the E0 logs printed their `PROBE` result lines, one budget-key extraction printed the untrained-seed fields of `a1/r_r1.json`, and the Part 4b extraction for (d) also evaluated the A2/A3 records' state. None of those values or states is recorded here, and nothing below depends on them.

Encoders (SHA-1 computed today = manifest pin, all three):

| tag | checkpoint | SHA-1 (computed) | manifest pin |
|---|---|---|---|
| A1 | `runs/m_v2/encoder.pt` | `34bb30a74d38b1fc4622861f43959a92fc01520c` | same |
| A2 | `runs/m_v2_s1/encoder.pt` | `380cdb7b67a97873c7b2c3b0458a638966eff9b1` | same |
| A3 | `runs/m_v2_s2/encoder.pt` | `4c3e1c7e88f662a40cae07141b859db264ddac0a` | same |

Every `run` entry except `E4.p1_untrained_bank.shared` (untrained bank; uses the checkpoint only for its architecture config) pins its own `LADDER_ENCODER` checkpoint, and the runner re-checks every pin before each entry and stops on a moved pin (`artifacts/ladder_v2_run.py:206-215`, `:291-293`).

---

## (a) `"smoke": true` and full settings

### Where `smoke` appears in outputs

A key-name search of every `.json` / `.jsonl` under `runs/ladder_v2` (and the member names of every `.npz`) found `smoke` in exactly these records, for each of A1/A2/A3:

| record | value | producing entry |
|---|---|---|
| `aX/h5_probes.jsonl` (one line each) | `true` | `E0.h5_probe.aX` |
| `aX/o1_matched.json` (top level and each of 6 `features[*]`) | `true` | `E3.o1_matched.aX` |
| `aX/o1_matched.partial.json` (6 rows) | `true` | `E3.o1_matched.aX` |
| `aX/p2_ladder.json` | `false` (with `escape_hatches: []`) | `E5.p2_ladder.aX` |

No other output carries `smoke`, `escape_hatches`, `quick`, `fast` or `reduced` keys.

### Is the flag behavioural?

- **`artifacts/h5_probe_lean.py`** — `"smoke": True` is a literal in the record dict at `:106`. It is not a variable, is not read anywhere in the driver, and no ladder driver reads `h5_probes.jsonl` (the only readers, `h5_read.py` and `readme_figures.py`, are not ladder entries). **Label only.** It is a legacy stamp from H5's era ("the effect floor is open"); it does not change the split, the cap or the bootstrap.
- **`artifacts/o1_matched.py`** — `"smoke": True` literals at `:207` (per feature) and `:265` (top level). Never read by the driver or by any ladder driver (`grep` over `artifacts/*.py`: only `o1_matched.py` itself and the manifest generator name the file). **Label only.**
- **`artifacts/p2_ladder.py`** — records `bool(pc.smoke)` at `:163` from `ProbingConfig` (`configs/probe.yaml`: `smoke: false`). In the package `ProbingConfig.smoke` (`src/galaxy_jepa/probing/config.py:108`) is behavioural in one place only: it lets a run score without the frozen effect-floor record (`floor.py:243`, `:284`, `:295`), and it is stamped into the ledger (`run.py:231`). It changes no null-draw, permutation, seed or sample budget. In ladder v2 it is `false` in every driver that builds a `ProbingConfig` (all of them through `j4_spread_controls.prepare`, which loads `configs/probe.yaml` at `:161` and overrides only `untrained_bank_path`, `:163-165`). **Not active.**
- A related flag, `r_nonlinear.Ctx(..., dry=True)`, is passed by most downstream drivers. It only suffixes `Ctx.record()` paths with `_dry` (`r_nonlinear.py:147`, `:153`); none of those drivers calls `record()`, and `n_features=0` keeps all 37 answers (`:141`). In `r_nonlinear.py` itself `dry` is set only by `--features`/`--only` (`:678`), which no manifest entry passes. **Not a reduction.**

### Manifest and runtime overrides

- Manifest env keys, over all 205 entries: `LADDER_ENCODER`, `LADDER_OUT`, `LADDER_TAG`, `DD_TAG`, `DD_ENCODER`, `DD_OUT`, `DD_BANK`, `DD_LOCAL`; `common_env`: `GJ_CONFIG`, `F0_CACHE_BASE`, `F0_NORM_PREFIX`, `LADDER_SPLIT=identical`, `LADDER_UNTRAINED_BANK`, `PYTHONUNBUFFERED`, `OMP_NUM_THREADS`. No `SMOKE`, `QUICK`, `N_*`, `*_FAST`, `LIMIT` or similar. The runner deletes any inherited `LADDER_*`, `DD_*`, `F0_*`, `GJ_CONFIG` before applying the manifest's (`ladder_v2_run.py:239-245`), and the drivers read no other environment variable (`grep os.environ` over the ladder drivers: only `dd_core.py:32-41` and `f0_preconditions.py:25-31`, plus `ladder_env`).
- Manifest arguments: the only CLI arguments passed are mode switches (`--bank`, `--planted`, `--run`, `--z3`, …, `plants`, `train A1 11 8`, …), `--checkpoint <encoder>`, `--tag o1`, `--stages <stage>`, `--counts`, `--arm aX`, `--seeds 30`. No entry passes `--features`, `--only`, `--plan`, `--limit`, `--max-train`, `--force`, `--dco`, `--agree`, `--z2`, `smoke`, or any budget argument.
- The A2/A3 entries are the A1 entries with the encoder substituted: after normalising the encoder path and tag, `cmd` and `env` are identical across A1/A2/A3 for all 69 per-encoder entry bases (0 differences). Five bases exist only for A1 (`E7.r_bank`, the four `E24.dd_train_untrained_*`); A2/A3 link those encoder-independent outputs instead (`E7.r_bank_link.*`, `E24.dd_untrained_link.*`).
- **What actually ran:** each of the 198 `run`-entry logs has exactly one runner-written `# cmd:` and `# env:` header (`ladder_v2_run.py:314`), and all 198 equal the manifest's `cmd` and `env` exactly. `state.json`: 205/205 `done`, exit 0.

### Budget per driver

All values are constants or argparse defaults in pinned code, or `configs/probe.yaml` (pinned `c0ddb4e3…`); none is overridden by any entry. Where an output records its own budget, the extracted field is given.

| driver | smoke source | behavioural? | budget settings (value, source) | entries | full settings? |
|---|---|---|---|---|---|
| `h5_probe_lean.py` | literal `:106` | no (label) | `--max-train` 40,000 (`:126`, not passed); identical split (`:80-82`); `n_boot` 2,000 (`probe_auc_ci` default, `logistic.py:269`). Recorded: `max_train` 40000 | E0.h5_probe ×3 | yes |
| `o1_matched.py` | literals `:207`, `:265` | no (label) | `--max-train` 40,000 (`:156`); `N_STRATA` 5 (`:73`); `n_boot` = probe.yaml 2,000 (`:204`, `:240`). Recorded: `n_strata` 5 | E3.o1_matched ×3 | yes |
| `p2_ladder.py` | `pc.smoke` (`:163`) | only if true (floor bypass); is `false` | `existence_method` untrained_z, K = 30 (probe.yaml `n_untrained_seeds`; bank resolution guard `:129`); `n_null_draws` 50, `n_boot` 2,000, `mlp_epochs` 200 (probe.yaml); `MAX_TRAIN` 40,000 (`:55`); `--features` 0 = all (`:106`). Recorded: `K` 30, `smoke` false, `escape_hatches` [], 37 features | E5.p2_ladder ×3 | yes |
| `p1_untrained_bank.py` | none | — | `--seeds 30` (manifest; default `:83`); `MAX_TRAIN` 40,000 (`:52`). Recorded: 30 seeds, `max_train` 40000 | E4 (shared) | yes |
| `ladder_v2_c2.py` | none | — | `prepare(None, 0)` = no train cap (`:35`); reuses a bank only after SHA-1 + id check (`:40-47`) | E1 ×3 | yes |
| `ladder_v2_e2.py` | none | — | `N_BOOT` 2,000, `SEED` (`aligned_c2.py:28`). Recorded: `n_boot` 2000 | E2 ×3 | yes |
| `r_nonlinear.py` | none | — | `N_BOOT` 2,000 (`:68`); `EXTRA_SEEDS` (1, 2) (`:64`); `CLUSTER_COUNTS` (100, 1,000) (`:67`); MLP from probe.yaml (`:386`); all features (`:656`, `:678`) | E7.r_bank (A1), E7.r0/r1/r2/r3 ×3 | yes |
| `s2_verify.py`, `t1_ladder.py` | none | — | `run_ladder` with probe.yaml as for P2; untrained bar from the shared K=30 bank | E5.s2_verify, E5.t1_ladder ×3 | yes |
| `r2_reach_split.py` | none | — | `r_nonlinear` constants (`:23-24`) | E7.r2_reach_split ×3 | yes |
| `s1_spiral_inclination.py` | none | — | `B_PERM` 10,000, `N_BOOT` 2,000 (`:30-31`) | E8 ×3 | yes |
| `u1_bar_winding.py` | none | — | `N_SHUFFLE` 30 (`:47`) | E10 ×3 | yes |
| `u2_uncertainty.py` | none | — | `N_PERM` 10,000 (`:44`); `FLOORS` (10, 21, 37) (`:43`); `perm=True` default (`:171`, call `:257`) | E11.u2_counts, E11.u2_uncertainty ×3 | yes |
| `u3_graded.py` | none | — | `N_PERM` 10,000, `N_NULL_C` 200, `N_BOOT` 1,000 (`:45-47`) | E12 ×3 | yes |
| `v2_independent.py` | none | — | `N_BOOT` 2,000 (`:37`); `U2.N_PERM` 10,000 (BY guard `:299`) | E14 ×3 | yes |
| `v3_own_directions.py` | none | — | `N_PERM` 10,000, `N_GROUP_NULL` 200 (`:34`) | E15 ×3 | yes |
| `w2_name_pcs.py` | none | — | `N_PERM` 10,000 (`:30`) | E16 measure/planted/run ×3 | yes |
| `x1_handedness.py` | none | — | `N_GROUP` 1,000, `N_BOOT` 1,000 (`:37-39`); `W.N_PERM` 10,000 (BY guards `:461`, `:541`) | E17 bank/planted/run ×3 | yes |
| `y3_science.py` | none | — | `N_SHARED` 200 (`:155`); `W.N_PERM` 10,000; `--z2`/`--dco`/`--agree` not passed (`:703-706`). Recorded: planted `shared_null.k` 200 | E18 bank/planted3/y3 ×3 | yes |
| `z3_bt.py` | none | — | `N_SHARED` 200 via `y3_science`; `W.N_PERM` 10,000 (BY guard `:127`). Recorded: `shared_null.k` 200 | E19 planted/z3 ×3 | yes |
| `aa1_label_efficiency.py` | none | — | `NS` (100 … 30,000) + full, `K` 5 (`:28-29`) | E20 planted/aa1 ×3 | yes |
| `aa3a_pose.py` | none | — | `N_PERM` 1,000 (`:46`; BY resolution guard `:173`); `N_CAUSAL` 200 (`:49`) | E22 planted/aa3a ×3 | yes |
| `aa3b_pose_average.py` | none | — | 8 D4 poses (`:41`); `n_boot` 2,000 (`:164`) | E23 bank/planted/aa3b ×3 | yes |
| `dd_sae.py` | none (has a separate `smoke` self-test command, `:257`, `:288`, never invoked by the manifest) | — | `EPOCHS` 8, `K` 32, `BATCH` 4,096, `LR` 2e-4 (`:40-41`); `LAYERS` (11, 6), `MULTS` (8, 16) (`:38-39`). Recorded: every `*_x*.log.json` (24) spans epochs 0–7 = 8 epochs | E24 extract/train/evaluate ×3 (untrained trains A1 only) | yes |
| `dd_sae_score.py` | none | — | constants `:24-28` | E24 sae_plants/score ×3 | yes |
| `dd_part4b.py` | none | — | `LAYER` 11, `MULT` 8, `N_SETS` 50 (`:32`); `MAX_PROPOSALS` 200,000 (`:33`) | E24 4b flags/stats/pools/plants/run ×3 | yes (A1: see (d)) |
| `dd_part1.py` | none | — | `N_SECONDARY` 400 (`:43`, a declared design subset in code, not an override) | E24.dd_v3_maps ×3 | yes |
| `dd_part1b.py` | none | — | `N_BOOT` 10,000 (`:34`) | E24 v3b direct/plants/v3 ×3 | yes |
| `plate_export.py` | none | — | `--limit` 0 (`:859`), `--size` 40,000, `--clusters` 400 (`:860-861`). Recorded: `plate_subset.npz` `limit` 0 | E26 subset/export/sanity ×3 | yes |

Note, not a reduction: `aa3a_pose.N_PERM` (1,000) and `u1_bar_winding.N_SHUFFLE` (30) are below probe.yaml's `n_perm` 10,000, but they are the drivers' own declared constants (the probe.yaml floor governs the package's uncertainty-geometry test, not these drivers), aa3a guards its family resolution at `:173`, and they are identical to what the pinned code declares. They are what the ladder was configured to run.

### REDUCED ENTRIES

**None.** No entry ran below its configured settings. The two `smoke: true` stamps (`h5_probes.jsonl`, `o1_matched*.json`) are hard-coded labels with no behavioural effect; the only behavioural smoke flag (`ProbingConfig.smoke`) is `false` everywhere.

---

## (b) Encoder identity for records that label the encoder under test "M"

**How the encoder is resolved.** Every ladder driver obtains its checkpoint through `j4_spread_controls.prepare` → `ladder_env.encoder` (`j4_spread_controls.py:174`; `ladder_env.py:40-43`), which returns `LADDER_ENCODER` whenever the `LADDER_ENCODER`/`LADDER_OUT` pair is set, whatever literal (`"runs/m/encoder.pt"`) the driver passes. The DD drivers load the encoder from `DD_ENCODER` (`dd_core.py:38`, `:58-61`). The manifest sets both to the same path for every entry (checked: no mismatch).

**Runtime SHA-1 guards.** Most of these drivers build `r_nonlinear.Ctx`, whose `load_matrices` calls `probe_bank.load_bank(o1_embeddings.npz, expected_checkpoint=setup.ckpt)` (`r_nonlinear.py:89`), which raises unless the checkpoint at `setup.ckpt` hashes to the bank's recorded `checkpoint_sha1` (`probe_bank.py:134-141`). Each encoder's `o1_embeddings.npz` records its own path and full SHA-1 (`34bb30a7…` / `380cdb7b…` / `4c3e1c7e…`, provenance `built`), so an entry that built `Ctx` could not have exited 0 with any other checkpoint. The DD maps entries have the same guard through `dd_core._bank()` (`dd_core.py:49-53`, called from `probe_readout`, `:134`, in `dd_part1.Ctx`, `dd_part1.py:61`).

**Search.** Key names across all outputs, plus string values equal to `"M"`, plus `.npz` member names and output file names. No record has an `encoder`/`model` field equal to `"M"`. No M-labelled record carries a checkpoint path or SHA-1 itself; every one was therefore checked against its entry log's runner-written `# env:` header (which records the `LADDER_ENCODER`/`DD_ENCODER` the process was started with), compared with the expected checkpoint and its SHA-1.

Evidence class in the last column: **G** = the entry's code also passed the SHA-1 guard above at run time; **D** = the DD guard (`dd_core._bank`); **U** = the entry loads the encoder from `LADDER_ENCODER` but performs no SHA-1 comparison and the output carries no identity (`x1_handedness.bank`, `:313-330`; `y3_science.bank`, `:392-404`); **N** = the entry loads no encoder at all — it scores the maps written by `E24.dd_v3_maps` / `E24.dd_v3b_direct` in the same per-encoder `DD_OUT` directory (class D).

| record (rel. to `runs/ladder_v2`) | entry | label used | recorded checkpoint / SHA-1 (12 hex), where found | expected | result |
|---|---|---|---|---|---|
| a1/aa1_label_efficiency.json | E20.aa1_aa1.a1 | `M`, `auc_M` | none in record; log `# env:` header LADDER_ENCODER=`runs/m_v2/encoder.pt` → SHA-1 `34bb30a74d38` | A1 `runs/m_v2/encoder.pt` `34bb30a74d38` (pin) | PASS (G) |
| a1/aa1_planted.json | E20.aa1_planted.a1 | `auc_M` | none in record; log `# env:` header LADDER_ENCODER=`runs/m_v2/encoder.pt` → SHA-1 `34bb30a74d38` | A1 `runs/m_v2/encoder.pt` `34bb30a74d38` (pin) | PASS (G) |
| a1/aa3a_pose.json | E22.aa3a_aa3a.a1 | `M` | none in record; log `# env:` header LADDER_ENCODER=`runs/m_v2/encoder.pt` → SHA-1 `34bb30a74d38` | A1 `runs/m_v2/encoder.pt` `34bb30a74d38` (pin) | PASS (G) |
| a1/w2_name_pcs.json | E16.w2_run.a1 | `share_M` | none in record; log `# env:` header LADDER_ENCODER=`runs/m_v2/encoder.pt` → SHA-1 `34bb30a74d38` | A1 `runs/m_v2/encoder.pt` `34bb30a74d38` (pin) | PASS (G) |
| a1/x1_transform_bank.npz | E17.x1_bank.a1 | arrays `M:orig`, `M:rot90`, `M:rot180`, `M:mirror` | none in record; log `# env:` header LADDER_ENCODER=`runs/m_v2/encoder.pt` → SHA-1 `34bb30a74d38` | A1 `runs/m_v2/encoder.pt` `34bb30a74d38` (pin) | PASS (U) |
| a1/x1_planted.json | E17.x1_planted.a1 | `M` | none in record; log `# env:` header LADDER_ENCODER=`runs/m_v2/encoder.pt` → SHA-1 `34bb30a74d38` | A1 `runs/m_v2/encoder.pt` `34bb30a74d38` (pin) | PASS (G) |
| a1/x1_handedness.json | E17.x1_run.a1 | `M` | none in record; log `# env:` header LADDER_ENCODER=`runs/m_v2/encoder.pt` → SHA-1 `34bb30a74d38` | A1 `runs/m_v2/encoder.pt` `34bb30a74d38` (pin) | PASS (G) |
| a1/y3_extra_bank.npz | E18.y3_bank.a1 | array `M` | none in record; log `# env:` header LADDER_ENCODER=`runs/m_v2/encoder.pt` → SHA-1 `34bb30a74d38` | A1 `runs/m_v2/encoder.pt` `34bb30a74d38` (pin) | PASS (U) |
| a1/y3_pitch.json | E18.y3_y3.a1 | `M` | none in record; log `# env:` header LADDER_ENCODER=`runs/m_v2/encoder.pt` → SHA-1 `34bb30a74d38` | A1 `runs/m_v2/encoder.pt` `34bb30a74d38` (pin) | PASS (G) |
| a1/dd/part1/v3_m_drop1.npy, v3_m_drop2.npy, v3_m_noise1.npy | E24.dd_v3_maps.a1 | `m` in file name | none in record; log `# env:` header LADDER_ENCODER + DD_ENCODER=`runs/m_v2/encoder.pt` → SHA-1 `34bb30a74d38` | A1 `runs/m_v2/encoder.pt` `34bb30a74d38` (pin) | PASS (D) |
| a1/dd/part1/direct_m_occl.npy, direct_m_v3.npy | E24.dd_v3b_direct.a1 | `m` in file name | none in record; log `# env:` header LADDER_ENCODER + DD_ENCODER=`runs/m_v2/encoder.pt` → SHA-1 `34bb30a74d38` | A1 `runs/m_v2/encoder.pt` `34bb30a74d38` (pin) | PASS (D) |
| a1/dd/part1/v3b_plants.json | E24.dd_v3b_plants.a1 | `brightness_as_M` | none in record; log `# env:` header LADDER_ENCODER + DD_ENCODER=`runs/m_v2/encoder.pt` → SHA-1 `34bb30a74d38` | A1 `runs/m_v2/encoder.pt` `34bb30a74d38` (pin) | PASS (N) |
| a1/dd/part1/v3b_score.json | E24.dd_v3b_v3.a1 | `auc_m`, `m_drop1_*`, `m_drop2_v3`, `m_noise1_v3` | none in record; log `# env:` header LADDER_ENCODER + DD_ENCODER=`runs/m_v2/encoder.pt` → SHA-1 `34bb30a74d38` | A1 `runs/m_v2/encoder.pt` `34bb30a74d38` (pin) | PASS (N) |
| a2/aa1_label_efficiency.json | E20.aa1_aa1.a2 | `M`, `auc_M` | none in record; log `# env:` header LADDER_ENCODER=`runs/m_v2_s1/encoder.pt` → SHA-1 `380cdb7b67a9` | A2 `runs/m_v2_s1/encoder.pt` `380cdb7b67a9` (pin) | PASS (G) |
| a2/aa1_planted.json | E20.aa1_planted.a2 | `auc_M` | none in record; log `# env:` header LADDER_ENCODER=`runs/m_v2_s1/encoder.pt` → SHA-1 `380cdb7b67a9` | A2 `runs/m_v2_s1/encoder.pt` `380cdb7b67a9` (pin) | PASS (G) |
| a2/aa3a_pose.json | E22.aa3a_aa3a.a2 | `M` | none in record; log `# env:` header LADDER_ENCODER=`runs/m_v2_s1/encoder.pt` → SHA-1 `380cdb7b67a9` | A2 `runs/m_v2_s1/encoder.pt` `380cdb7b67a9` (pin) | PASS (G) |
| a2/w2_name_pcs.json | E16.w2_run.a2 | `share_M` | none in record; log `# env:` header LADDER_ENCODER=`runs/m_v2_s1/encoder.pt` → SHA-1 `380cdb7b67a9` | A2 `runs/m_v2_s1/encoder.pt` `380cdb7b67a9` (pin) | PASS (G) |
| a2/x1_transform_bank.npz | E17.x1_bank.a2 | arrays `M:orig`, `M:rot90`, `M:rot180`, `M:mirror` | none in record; log `# env:` header LADDER_ENCODER=`runs/m_v2_s1/encoder.pt` → SHA-1 `380cdb7b67a9` | A2 `runs/m_v2_s1/encoder.pt` `380cdb7b67a9` (pin) | PASS (U) |
| a2/x1_planted.json | E17.x1_planted.a2 | `M` | none in record; log `# env:` header LADDER_ENCODER=`runs/m_v2_s1/encoder.pt` → SHA-1 `380cdb7b67a9` | A2 `runs/m_v2_s1/encoder.pt` `380cdb7b67a9` (pin) | PASS (G) |
| a2/x1_handedness.json | E17.x1_run.a2 | `M` | none in record; log `# env:` header LADDER_ENCODER=`runs/m_v2_s1/encoder.pt` → SHA-1 `380cdb7b67a9` | A2 `runs/m_v2_s1/encoder.pt` `380cdb7b67a9` (pin) | PASS (G) |
| a2/y3_extra_bank.npz | E18.y3_bank.a2 | array `M` | none in record; log `# env:` header LADDER_ENCODER=`runs/m_v2_s1/encoder.pt` → SHA-1 `380cdb7b67a9` | A2 `runs/m_v2_s1/encoder.pt` `380cdb7b67a9` (pin) | PASS (U) |
| a2/y3_pitch.json | E18.y3_y3.a2 | `M` | none in record; log `# env:` header LADDER_ENCODER=`runs/m_v2_s1/encoder.pt` → SHA-1 `380cdb7b67a9` | A2 `runs/m_v2_s1/encoder.pt` `380cdb7b67a9` (pin) | PASS (G) |
| a2/dd/part1/v3_m_drop1.npy, v3_m_drop2.npy, v3_m_noise1.npy | E24.dd_v3_maps.a2 | `m` in file name | none in record; log `# env:` header LADDER_ENCODER + DD_ENCODER=`runs/m_v2_s1/encoder.pt` → SHA-1 `380cdb7b67a9` | A2 `runs/m_v2_s1/encoder.pt` `380cdb7b67a9` (pin) | PASS (D) |
| a2/dd/part1/direct_m_occl.npy, direct_m_v3.npy | E24.dd_v3b_direct.a2 | `m` in file name | none in record; log `# env:` header LADDER_ENCODER + DD_ENCODER=`runs/m_v2_s1/encoder.pt` → SHA-1 `380cdb7b67a9` | A2 `runs/m_v2_s1/encoder.pt` `380cdb7b67a9` (pin) | PASS (D) |
| a2/dd/part1/v3b_plants.json | E24.dd_v3b_plants.a2 | `brightness_as_M` | none in record; log `# env:` header LADDER_ENCODER + DD_ENCODER=`runs/m_v2_s1/encoder.pt` → SHA-1 `380cdb7b67a9` | A2 `runs/m_v2_s1/encoder.pt` `380cdb7b67a9` (pin) | PASS (N) |
| a2/dd/part1/v3b_score.json | E24.dd_v3b_v3.a2 | `auc_m`, `m_drop1_*`, `m_drop2_v3`, `m_noise1_v3` | none in record; log `# env:` header LADDER_ENCODER + DD_ENCODER=`runs/m_v2_s1/encoder.pt` → SHA-1 `380cdb7b67a9` | A2 `runs/m_v2_s1/encoder.pt` `380cdb7b67a9` (pin) | PASS (N) |
| a3/aa1_label_efficiency.json | E20.aa1_aa1.a3 | `M`, `auc_M` | none in record; log `# env:` header LADDER_ENCODER=`runs/m_v2_s2/encoder.pt` → SHA-1 `4c3e1c7e88f6` | A3 `runs/m_v2_s2/encoder.pt` `4c3e1c7e88f6` (pin) | PASS (G) |
| a3/aa1_planted.json | E20.aa1_planted.a3 | `auc_M` | none in record; log `# env:` header LADDER_ENCODER=`runs/m_v2_s2/encoder.pt` → SHA-1 `4c3e1c7e88f6` | A3 `runs/m_v2_s2/encoder.pt` `4c3e1c7e88f6` (pin) | PASS (G) |
| a3/aa3a_pose.json | E22.aa3a_aa3a.a3 | `M` | none in record; log `# env:` header LADDER_ENCODER=`runs/m_v2_s2/encoder.pt` → SHA-1 `4c3e1c7e88f6` | A3 `runs/m_v2_s2/encoder.pt` `4c3e1c7e88f6` (pin) | PASS (G) |
| a3/w2_name_pcs.json | E16.w2_run.a3 | `share_M` | none in record; log `# env:` header LADDER_ENCODER=`runs/m_v2_s2/encoder.pt` → SHA-1 `4c3e1c7e88f6` | A3 `runs/m_v2_s2/encoder.pt` `4c3e1c7e88f6` (pin) | PASS (G) |
| a3/x1_transform_bank.npz | E17.x1_bank.a3 | arrays `M:orig`, `M:rot90`, `M:rot180`, `M:mirror` | none in record; log `# env:` header LADDER_ENCODER=`runs/m_v2_s2/encoder.pt` → SHA-1 `4c3e1c7e88f6` | A3 `runs/m_v2_s2/encoder.pt` `4c3e1c7e88f6` (pin) | PASS (U) |
| a3/x1_planted.json | E17.x1_planted.a3 | `M` | none in record; log `# env:` header LADDER_ENCODER=`runs/m_v2_s2/encoder.pt` → SHA-1 `4c3e1c7e88f6` | A3 `runs/m_v2_s2/encoder.pt` `4c3e1c7e88f6` (pin) | PASS (G) |
| a3/x1_handedness.json | E17.x1_run.a3 | `M` | none in record; log `# env:` header LADDER_ENCODER=`runs/m_v2_s2/encoder.pt` → SHA-1 `4c3e1c7e88f6` | A3 `runs/m_v2_s2/encoder.pt` `4c3e1c7e88f6` (pin) | PASS (G) |
| a3/y3_extra_bank.npz | E18.y3_bank.a3 | array `M` | none in record; log `# env:` header LADDER_ENCODER=`runs/m_v2_s2/encoder.pt` → SHA-1 `4c3e1c7e88f6` | A3 `runs/m_v2_s2/encoder.pt` `4c3e1c7e88f6` (pin) | PASS (U) |
| a3/y3_pitch.json | E18.y3_y3.a3 | `M` | none in record; log `# env:` header LADDER_ENCODER=`runs/m_v2_s2/encoder.pt` → SHA-1 `4c3e1c7e88f6` | A3 `runs/m_v2_s2/encoder.pt` `4c3e1c7e88f6` (pin) | PASS (G) |
| a3/dd/part1/v3_m_drop1.npy, v3_m_drop2.npy, v3_m_noise1.npy | E24.dd_v3_maps.a3 | `m` in file name | none in record; log `# env:` header LADDER_ENCODER + DD_ENCODER=`runs/m_v2_s2/encoder.pt` → SHA-1 `4c3e1c7e88f6` | A3 `runs/m_v2_s2/encoder.pt` `4c3e1c7e88f6` (pin) | PASS (D) |
| a3/dd/part1/direct_m_occl.npy, direct_m_v3.npy | E24.dd_v3b_direct.a3 | `m` in file name | none in record; log `# env:` header LADDER_ENCODER + DD_ENCODER=`runs/m_v2_s2/encoder.pt` → SHA-1 `4c3e1c7e88f6` | A3 `runs/m_v2_s2/encoder.pt` `4c3e1c7e88f6` (pin) | PASS (D) |
| a3/dd/part1/v3b_plants.json | E24.dd_v3b_plants.a3 | `brightness_as_M` | none in record; log `# env:` header LADDER_ENCODER + DD_ENCODER=`runs/m_v2_s2/encoder.pt` → SHA-1 `4c3e1c7e88f6` | A3 `runs/m_v2_s2/encoder.pt` `4c3e1c7e88f6` (pin) | PASS (N) |
| a3/dd/part1/v3b_score.json | E24.dd_v3b_v3.a3 | `auc_m`, `m_drop1_*`, `m_drop2_v3`, `m_noise1_v3` | none in record; log `# env:` header LADDER_ENCODER + DD_ENCODER=`runs/m_v2_s2/encoder.pt` → SHA-1 `4c3e1c7e88f6` | A3 `runs/m_v2_s2/encoder.pt` `4c3e1c7e88f6` (pin) | PASS (N) |

**Count: 39 PASS, 0 FAIL, 0 NO EVIDENCE.** All 39 rest on the log header; 21 (G) + 6 (D) also passed a run-time SHA-1 guard; 6 (U) and 6 (N) rest on the header and the code path alone.

Gaps worth knowing (not failures): `x1_transform_bank.npz` and `y3_extra_bank.npz` carry no checkpoint identity, and their readers (`x1_handedness.py:581`, `y3_science.py:77`) `np.load` them without a check — so a bank from another encoder would not be caught by the run itself. Here each was produced in the same per-encoder directory under the same `LADDER_ENCODER` (headers above), so the gap did not bite.

**Records whose `M`-like keys are not encoder labels** (identity nevertheless consistent): `r_r0.json` (`M`, `M_lo`, `M_hi`, `C_m` = the *matched* AUC and its bar, `r_nonlinear.py:248`), `r_r1.json` (`M_mlp`, matched, `:542`), `s2_ladder.json` / `t1_ladder.json` (`M_lo`, `C_m`, matched, `s2_verify.py:67`), `v2_independent.json`, `y3_pitch.json`, `y3_planted.json`, `z3_bt.json`, `z3_planted.json` (`a_m` = the measurement leg "A_m", `v2_independent.py:195`, `y3_science.py:140`, `z3_bt.py:39`, `:93`). The R/Y3/Z3/V2 entries build `Ctx` (guard G) and their log headers name the expected checkpoint for all three encoders.

**Records that carry identity themselves** (not M-labelled; all PASS against the expected path and SHA-1): `aX/h5_probes.jsonl` (`arm` aX, `checkpoint` path); `aX/o1_matched.json` and `.partial.json` (`checkpoint` path); `aX/p2_ladder.json`, `s2_ladder.json`, `t1_ladder.json` (`checkpoint` path); `aX/e2_answers.json` (`encoder` aX); `aX/o1_embeddings.npz`, `aX/c2_aX_embeddings.npz`, `aX/s2_noise.npz` (path + full SHA-1 equal to the pin, provenance `built`). The `E1.c2_bank.aX` logs each state the linked bank's checkpoint SHA-1 and ids were verified.

---

## (c) E19 (Z3, B/T) planted check

**What the planted check is.** `z3_bt.py --planted` (`:136-144`) runs `y3_science.planted_2x2` (`y3_science.py:593-629`) on V2's primary B/T rows. It plants three synthetic target pairs built from random directions of the encoder's own embedding and passes each through the identical `two_by_two` → `shared_null` → `state_2x2` path the real test uses (BY over the same 4-test family). Each output key names the state its plant must reach (`z3_bt.py:12`; `planted_2x2` docstring):

- `T2_both` — two independent latents, one per target → must read **BOTH**;
- `T2_shared` — both targets are noisy copies of **one** latent → must read **SHARED ONLY** (neither leg may clear the shared-quantity null);
- `T2_neither` — pure noise → must read **NEITHER**.

Under D28 a planted check that does not reach its named state shows the test cannot be relied on to separate those states, so a real result in that region is not trustworthy.

**Planted fields only** (`aX/z3_planted.json`, keys `T2_both`/`T2_shared`/`T2_neither`, fields `state` and `state_y3_rule`; the real record `z3_bt.json` was not opened):

| encoder | `T2_both` | `T2_shared` | `T2_neither` |
|---|---|---|---|
| A1 | BOTH (as named) | **VOTES BEYOND MEASUREMENT** (named: SHARED ONLY) — both under the primary rule (random-latent null, 97.5th) and `state_y3_rule` (ridge-latent null, 95th) | NEITHER (as named) |
| A2 | BOTH | SHARED ONLY | NEITHER |
| A3 | BOTH | SHARED ONLY | NEITHER |

**Finding.** For **A1**, the **`T2_shared`** plant reached **VOTES BEYOND MEASUREMENT**, while its key names **SHARED ONLY**. The votes leg of the test cleared the shared-quantity null on a plant in which no votes-specific signal exists. A one-sided verdict was therefore produced where the correct reading was "shared only".

**Consequence (D28).** E19's real result for **A1** is **not to be relied on**. The planted check that must fire for the SHARED ONLY state did not do so on A1's embedding through the identical code path. The real E19 result was not opened for any encoder, and none is reported here.

Context (planted only): the same `planted_2x2` path in E18 (`aX/y3_planted.json`, `T2_*`) reached all three named states for A1, A2 and A3. The failure is specific to E19's B/T rows on A1.

---

## (d) E24 Part 4b for A1 — NOT RUN

**Record: E24 Part 4b for A1 was NOT RUN. Its records are not a result.**

- **Width requirement.** Part 4b and its flags read the 8× dictionary at block 11: `LAYER, MULT = 11, 8` (`artifacts/dd_part4b.py:32`), loaded at `:48` (flags), `:77` (stats) and `:205` (the bench). `_offset_set()` refuses if the scored dictionary is not 8× (`:97-100`: `if score.get("chosen", MULT) != MULT: raise SystemExit(...)`).
- **The chosen width.** `dd_sae_score.chosen_size` (`dd_sae_score.py:259-262`) picks 8× unless 8× fails S3 and 16× does not. `score()` records it (`:276-277`) and computes S1 on that chosen width (`:284`). **A1's `dd/sae/score.json` `chosen` = 16** (the only field extracted). For reference, A2 and A3 record `chosen` = 8.
- **What the A1 4b entries did.** `pools()`, `plants()` and `run()` each call `_inapplicable()` first (`:178`, `:277`, `:305`). It returns `{"state": "INAPPLICABLE", …}` when `score.json["S1"]["b11"]` has no candidates (`:104-110`). For A1 that S1 was scored on the **16×** dictionary. A1's `part4b.json` and `part4b_plants.json` both carry `state = INAPPLICABLE` (`part4b_plants.json` with `fire_as_expected: null`). The entries exited 0. The 8× guard at `:99-100`, which would have refused, is reached only after `_inapplicable()` and so never ran. `E24.dd_4b_flags.a1` and `E24.dd_4b_stats.a1` did compute `flags.npz` / `part4b_stats.npz`, but on the 8× dictionary, which is not A1's chosen one, and nothing consumed them.
- **Reading.** A1's Part 4b `INAPPLICABLE` label comes from the order of the checks. It is not a finding: the test was never run on the dictionary its pre-registration requires. Record it as **NOT RUN (A1 chose 16×; Part 4b requires 8×)** and read nothing from A1's `part4b*.json`, `flags.npz` or `part4b_stats.npz`.
- **Problem to note.** The code labels a width mismatch as INAPPLICABLE instead of refusing. If `_inapplicable()` ran after the width check, or checked `chosen` itself, A1's 4b entries would have failed loudly as the house rule requires.

---

## (e) E5 — what the records hold

- E5 is `p2_ladder.py`, `s2_verify.py` and `t1_ladder.py`. The per-answer existence p-value is computed in the package: `nulls.existence_verdicts` builds `pvals` with `untrained_z_pvalue(fc.real_auc, real_se[feat], untrained_bank[feat])` (`src/galaxy_jepa/probing/nulls.py:503-505`; the function is defined at `:208`), applies BY at `:517`, and keeps both in `ExistenceVerdict(pvalue=…, exceeds_null=…)` (`:443`, `:522-523`).
- No E5 driver writes the p-value. `t1_ladder.py:119-120` writes only `exceeds_null` and `clean` per answer (pass/fail against the family-corrected null, and pass/fail plus the effect floor). `p2_ladder._rung_row` (`p2_ladder.py:85-100`) and `s2_verify.row` (`s2_verify.py:62-69`) write neither. A key-name check of A1's three records confirms it: `exceeds_null` is present only in `t1_ladder.json`, and no record has a `p`, `pvalue` or `p_*` key.
- **The p-values can be recomputed later from stored inputs.** Under `existence_method: untrained_z` the null is the K-seed untrained bar, not shuffled draws. It is stored in `runs/ladder_v2/shared/p1_untrained_bank.json` under `seeds`: a dict of 30 seeds, each a dict of 37 answer → float. The real side is stored per answer in each E5 record's rows as `auc` (= `fc.real_auc`, `ladder.py:525`) and `real_se` (the bootstrap SE, `ladder.py:695`, passed at `:729`). Recompute with `nulls.untrained_z_pvalue(auc, real_se, [seeds[s][answer] for s in seeds])`, then BY over the family (`multiplicity: benjamini_yekutieli`, family = the scheme's 37). The `n_null_draws: 50` control draws (shuffled / random embedding) are not stored in these records, but under `untrained_z` they are not the existence test's denominator (probe.yaml comment on `n_null_draws`).

---

## Summary

| check | outcome |
|---|---|
| (a) smoke / full settings | No entry ran reduced. `smoke: true` in `h5_probes.jsonl` and `o1_matched*.json` is a hard-coded label with no behavioural effect. `ProbingConfig.smoke` is behavioural (effect-floor bypass only) but `false`. 198/198 executed command lines and envs equal the manifest. |
| (b) encoder identity ("M" records) | 39 PASS / 0 FAIL / 0 NO EVIDENCE. Identity comes from the entry log env headers, and 27 of the 39 also passed a run-time SHA-1 guard. The x1/y3 banks have no identity check. |
| (c) E19 planted | A1, `T2_shared`: reached VOTES BEYOND MEASUREMENT, named SHARED ONLY. Under D28 the real E19 result for A1 is not to be relied on. |
| (d) E24 Part 4b, A1 | NOT RUN, not a result. A1's `chosen` = 16, but 4b requires 8× (`dd_part4b.py:32`, `:97-100`). Its records read INAPPLICABLE because of the order of the checks (`:104-110`, `:277`, `:305`). |
| (e) E5 | The records hold `exceeds_null` (and `clean`) without p-values. The p-values can be recomputed: the 30×37 untrained bar is stored, and so are the per-answer `auc` and `real_se`. |
