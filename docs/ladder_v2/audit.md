# Ladder v2 — completeness and integrity audit (blinded)

- **Audit run:** 2026-10-17 04:09 (Europe/London, BST)
- **HEAD:** `4bc25a60bfec13ba6d5ba25fd2dd894b812d56fd` on branch `ladder-v2` (working tree: only the four untracked runner records `outputs.sha1`, `run_log.md`, `runner_log`, `state.json`; no tracked file modified)
- **Auditor:** a fresh, blinded sub-agent; no result value was read, printed or recorded.

## Verdict

**COMPLETE** — 205/205 entries `done` (exit 0), 0 failed, 0 skipped; 432/432 recorded outputs re-hashed and matching both records; 53/53 distinct pins (2958 pin references) unmoved.

Two observations that are by design, not defects (detailed below): the 7 `link` entries have no `logs/<id>.log` (the runner writes one for a link entry only when it fails), and the 3 `E24.dd_4b_pools.*` entries record no outputs (that mode only reports pool sizes to its log).

## Method

- Read in full: `docs/ladder_v2/manifest.yaml`, `state.json`, `outputs.sha1`, `run_log.md`, and the runner `artifacts/ladder_v2_run.py` (for the state/pin/output schema and hashing). `build_notes.md`, `plan.md` and `inventory.md` were only grepped for one entry's mode name; the mode's writes were confirmed by grepping the driver source `artifacts/dd_part4b.py` for write calls.
- SHA-1s of the records as audited: `manifest.yaml` `6c2f462009ce7f60a81b85e9eca02b8b2662732c`, `state.json` `16b521906ec9c85a058a75ee4f730c8f26ea0847`, `outputs.sha1` `26989f55eb505f209ba96e1e7c55bb4ea770247b`, `run_log.md` `c2ab5e19a6d85eb9eb509b30904daae10faec06b`, `artifacts/ladder_v2_run.py` `af94ea57031545a8c8c52e26d08146d46a38245f`.
- **Outputs** were re-hashed with the runner's own method (`_file_sha`): a regular file → SHA-1 of its bytes (4 MiB blocks); a symlink → the SHA-1 of its resolved target, and a target directory → `_sha1_tree` (sorted `rglob('*')`, each file as `relpath\0sha1\n`). Bytes were fed to `hashlib` and never printed or parsed. Each recomputed SHA-1 was compared with `state.json["files"]` and with `outputs.sha1`; the two records were compared with each other; each entry's `outputs` list was reconciled with `files` (and owner); every manifest `expect` file was checked to exist and to be recorded; and the four output directories (`a1`, `a2`, `a3`, `shared`) were walked (as the runner's `_snapshot` does) for files on disk the state does not record.
- The manifest has no `outputs` key: an entry's outputs are, by the runner's design, every file its run created or changed in its `out_dir` (snapshot before/after), plus the `expect` files that must exist. Both were checked.
- **Pins** were recomputed with the runner's `_pin_sha`: `…/**.py` → `_sha1_tree_pkg` (sorted `rglob('*.py')`, `relpath\0sha1\n`); anything else → file SHA-1. Encoder checkpoints were hashed as bytes, never loaded. Every entry's pin map was also checked for internal consistency (no path pinned at two SHA-1s).
- **Inputs**: every `inputs` edge checked to point at an entry earlier in manifest order whose status is `done`.
- **Logs**: existence and size of `runs/ladder_v2/logs/<id>.log` only (`stat`), and that `state.json` names that path.
- **Interruptions**: `state.json` `pauses`; `run_log.md` rows not `done`; runner logs grepped only for stage lines (`=== k/205 id`) and the permitted status words (`PAUSED`, `RESUMED`, `interrupted`, `finished`, `done`/`failed` lines reduced to status, wall-clock, rss and `outputs N`); `last reboot` for host reboot times; `stat` mtimes of the rerun entries' outputs.

## 1. Manifest vs state

| check | result |
|---|---|
| manifest SHA-1 = `state.json` `manifest_sha1` | yes (`6c2f462009ce7f60a81b85e9eca02b8b2662732c`) |
| manifest entries / unique ids / state entries | 205 / 205 / 205 |
| ids missing from state / extra in state | 0 / 0 |
| status counts | `done` 205 (failed 0, skipped(dep) 0) |
| `pending` / `running` remaining | 0 |
| `done` with non-zero exit | 0 |
| `skipped(dep)` not a transitive dependant of a failure | 0 (there are no failures and no skips) |
| stray state flags (`interrupted`, `retried`, `missing_expected`, `rewrote_files_of`, `escaped`) | 0 |
| top-level `paused` key (an unresolved pause) | absent |

## 2. Outputs

| check | result |
|---|---|
| recorded outputs in `state.json` `files` / lines in `outputs.sha1` | 432 / 432 |
| paths or SHA-1s on which the two records disagree | 0 |
| sum of per-entry `outputs` lists | 432 (no path claimed by two entries: 0 multi-owner) |
| entry outputs absent from `files` / `files` paths in no entry / owner mismatches | 0 / 0 / 0 |
| recorded outputs missing on disk | 0 |
| outputs re-hashed | 432 (400 regular files, 43,849,929,725 bytes; 32 symlinks hashed through their targets, 3 of them to directories) |
| SHA-1 mismatches (recomputed vs `state.json` and vs `outputs.sha1`) | 0 |
| manifest `expect` files / missing / not recorded | 150 / 0 / 0 |
| files on disk under `a1`, `a2`, `a3`, `shared` / not recorded in the state | 432 / 0 |
| `done` entries recording zero outputs | 3: `E24.dd_4b_pools.a1`, `E24.dd_4b_pools.a2`, `E24.dd_4b_pools.a3` |

The three zero-output entries run `artifacts/dd_part4b.py pools`, whose `pool_report` writes no file (it reports pool sizes and sampler acceptance to the entry log); none has an `expect`, and each exited 0. Their dependants (`E24.dd_4b_plants.*`) are `done`. By design, not a defect.

Symlinked outputs (32): `setup.ext.a1/a2/a3` (→ the `artifacts/out/ext` directory), `E1.c2_bank.*` (1 each, → a file in `artifacts/out`), `E7.r_bank_link.a2/a3` (1 each, → `a1`), `E24.dd_untrained_link.a2/a3` (12 each, → `a1/dd/sae*`). As in the runner, each was hashed through its target; none is dangling. Hashing the `artifacts/out` targets read their bytes into `hashlib` only.

## 3. Pins

2958 pin references over 205 entries, 53 distinct paths/trees; 0 paths pinned at two SHA-1s within the manifest; **0 moved**.

| pinned path | manifest SHA-1 | recomputed |
|---|---|---|
| `artifacts/aa1_label_efficiency.py` | `04d0eb19b04ac0fcbc6be6b2dfc8282347cca754` | match |
| `artifacts/aa3a_pose.py` | `95904aef99b459a39dea8f7f475238aeda7d0b62` | match |
| `artifacts/aa3b_pose_average.py` | `74c2cdd0a2927c62a0a557dfa8e4e178d5466c02` | match |
| `artifacts/aligned_c2.py` | `029adeb522a3026603e36f7bf9187e3e9ae39c8a` | match |
| `artifacts/dd_circuits.py` | `b2f6d5f0f89d56490c2ebe7d5f4e2fd963526445` | match |
| `artifacts/dd_core.py` | `1694ba80f78168f15d7d83fbff925a8fafea328b` | match |
| `artifacts/dd_part1.py` | `9981cc744ff2115f5abb1bf95bc85837e74c8027` | match |
| `artifacts/dd_part1b.py` | `c7dca4bad08b34b0e96c2ecc88f8ed336adc1633` | match |
| `artifacts/dd_part2.py` | `c89dcc858b7d4d5838cb6bacc767c5edcddfc40f` | match |
| `artifacts/dd_part4.py` | `e7a769c4df7b5bfe9c38d3d226237f9d7e27539c` | match |
| `artifacts/dd_part4b.py` | `c4fec662cc3af7426e554850f9045785ed15c25b` | match |
| `artifacts/dd_sae.py` | `a0436325f756e2652f82b0a18ce9421157ad9a69` | match |
| `artifacts/dd_sae_cards.py` | `ca26fa40625705a9c591c9fc0158bbf51879fa60` | match |
| `artifacts/dd_sae_score.py` | `ce7f728fba05e92c682c05a55c5b3e3625886d5c` | match |
| `artifacts/dd_step0.py` | `e97b3d79b5e0941860a460863d1f194eda0c5248` | match |
| `artifacts/dd_v3.py` | `a517d796b76a010556446a699048b126f1342858` | match |
| `artifacts/f0_preconditions.py` | `b5491b4f50f4630258928fdf0a32ec1b36a0cdf5` | match |
| `artifacts/f1_loader_bench.py` | `75376602ff0b50093ed2af4edaf59053541f3cbb` | match |
| `artifacts/h5_probe_lean.py` | `9572f62a19a63b6fb7581cdf8b06218c025b83b4` | match |
| `artifacts/j4_spread_controls.py` | `9598302bdba7e3376c35cdfff4dfa0ddbe4e69b9` | match |
| `artifacts/ladder_env.py` | `7ffe0f430ef55c67a744bd19aa873cd71872eb6c` | match |
| `artifacts/ladder_v2_c2.py` | `a19f711f4885b41dbcb43a32ff5de8127b1c9fef` | match |
| `artifacts/ladder_v2_e2.py` | `7006f2822810e4f330ece88f3c0baf2dcf32a40e` | match |
| `artifacts/o1_matched.py` | `6ffd24bbb7e2c815754be24ae7b2bcc15968d051` | match |
| `artifacts/p1_untrained_bank.py` | `ba48d0fc01d0b94cec74941f077dadeceba15e34` | match |
| `artifacts/p2_ladder.py` | `9bf8ff5bb88856ad0b686fba77d2e263e6ebac4e` | match |
| `artifacts/plate_export.py` | `c3853acb22b770927e5d0f04c2c0bfca9e24560a` | match |
| `artifacts/probe_bank.py` | `b5b854f9011b0b9965e562ef8adeb3c4bddbbf57` | match |
| `artifacts/r2_reach_split.py` | `c8e3633d4245134e8f84fb243874c7f6ea1ca2aa` | match |
| `artifacts/r_nonlinear.py` | `9e7aeaa276a43bf219997e9d4a474036c4df57ca` | match |
| `artifacts/s1_spiral_inclination.py` | `f2063bf02b89978e5c290a3c23228b140586371b` | match |
| `artifacts/s2_verify.py` | `ca631c8f6411a0c6dbea93bdf823b63b885007bd` | match |
| `artifacts/t1_ladder.py` | `2540bde70589f0070a846f0128ebf6beed5c22a5` | match |
| `artifacts/t2_other.py` | `c1cdb613303e4cb41dbcd8b40f86d701cfda62da` | match |
| `artifacts/u1_bar_winding.py` | `1d651c5e698817479b6c0d441f8bf4a735a37e23` | match |
| `artifacts/u2_uncertainty.py` | `e5a36d7d0727f551f5bb6ec4ae1fff4afe62d3e7` | match |
| `artifacts/u3_graded.py` | `808687d15983842b7ef5e647f75145f05f9c0392` | match |
| `artifacts/v1_loose_ends.py` | `467112f52a66f177901842e7adec112d87da6b4a` | match |
| `artifacts/v2_independent.py` | `e99d4cb29ed98a9607aaf38ab05797e7d7d1fb2b` | match |
| `artifacts/v3_own_directions.py` | `2eb8401a88aae25f99afacd6452152c4cfe80326` | match |
| `artifacts/w2_name_pcs.py` | `abf5cefe434037123217826c9fb3c5f14116b43d` | match |
| `artifacts/x1_handedness.py` | `cfd43eb4f64d49d7da7efee64442da07d46340f8` | match |
| `artifacts/y3_science.py` | `f25bfeb8deca993761b2d8732fc20d95ff141395` | match |
| `artifacts/y_pitch.py` | `12e787a006a0e0c8acfd996b49c24d1c5b95f405` | match |
| `artifacts/z3_bt.py` | `8129c9a3d97998fed0cba219dbcedab89ee06402` | match |
| `configs/effect_floor.json` | `e6b8b8ab7f0f0a9fea733a5d7d68135098b43cd1` | match |
| `configs/pretrain.yaml` | `13c483b63a3e07b19a3a8f5a1ac7fc7a9b2ab1d0` | match |
| `configs/pretrain_v2.yaml` | `c232b054f42114c9ad6e5e76dbc8c8c342ebf838` | match |
| `configs/probe.yaml` | `c0ddb4e322b70e90adfaf6b2d7760ba42003d28b` | match |
| `runs/m_v2/encoder.pt` | `34bb30a74d38b1fc4622861f43959a92fc01520c` | match |
| `runs/m_v2_s1/encoder.pt` | `380cdb7b67a97873c7b2c3b0458a638966eff9b1` | match |
| `runs/m_v2_s2/encoder.pt` | `4c3e1c7e88f662a40cae07141b859db264ddac0a` | match |
| `src/galaxy_jepa/**.py` | `9400ddf417ab8690d3148c7500125f3eb7bbf4cc` | match |

Not a pin (by the runner's design): `artifacts/ladder_v2_run.py` itself. It changed once during the ladder (commit `4bc25a6`, 14 Oct 15:56, adding pause/resume) between the first and second runner sessions; the current file (`af94ea57031545a8c8c52e26d08146d46a38245f`) equals HEAD. The manifest SHA-1 in the state guarded every restart.

## 4. Inputs

192 entries declare inputs (773 edges). Edges whose input is not earlier in manifest order or not `done`: **0**.

## 5. Logs

- Entry logs present: **198/205**; empty: 0; `state.json` `log` path differing from `logs/<id>.log`: 0.
- Absent (7), all `kind: link` entries, which the runner (`run_entry`) logs only on failure — the state still names the path: `setup.ext.a1`, `setup.ext.a2`, `E7.r_bank_link.a2`, `E24.dd_untrained_link.a2`, `setup.ext.a3`, `E7.r_bank_link.a3`, `E24.dd_untrained_link.a3`.
- Other files in `logs/` (not entry logs; stat only): `memguard.log`, `runner_20261013_1456.log`, `runner_20261014_1723.log`, `runner_20261015_2136.log`, `runner_20261015_2327.log`, `watchdog.log`.

## 6. Interruptions and pauses

`state.json` `pauses`: **1** record(s):

| paused at | before (k/205) | entry | how | resumed |
|---|---|---|---|---|
| 2026-10-15 22:18:44 | 114 | `E24.dd_train_trained_11_8.a2` | interrupted | 2026-10-15 23:27:01 |

Runner sessions (from the stage lines and permitted status lines of the four runner logs):

| runner log | first → last stage | how it ended | entry cut short | rerun in |
|---|---|---|---|---|
| `runner_20261013_1456.log` | 1 → 23 (13 Oct 14:56 → 14 Oct 14:45 start of 23) | no terminal line (the stop time itself is not logged); consistent with the reported manual pause of 14 Oct, made before the pause mechanism existed (commit `4bc25a6`, 15:56), hence no `pauses` record | `E16.w2_measure.a1` (23) | session 2: 17:23:54 start, done 17:57:45, 33.8 min, outputs 1 |
| `runner_20261014_1723.log` | 1 → 110 (22 earlier; 14 Oct 17:23 → 15 Oct 21:07 start of 110) | no terminal line; host reboot at 15 Oct 21:15 (`last reboot`), consistent with the first reported kernel panic | `E23.aa3b_aa3b.a2` (110) | session 3: 21:36:51 start, done 21:58:37, 21.7 min, outputs 1 |
| `runner_20261015_2136.log` | 1 → 114 (109 earlier) | `signal 15 — pausing now` 22:18:41; `interrupted by the pause after 15.4 min`; `PAUSED` 22:18:44 (memory-guard pause; `memguard.log` mtime 22:18) | `E24.dd_train_trained_11_8.a2` (114) | session 4: 23:27:50 start, done 23:42:15, 14.4 min, outputs 2 |
| `runner_20261015_2327.log` | 1 → 205 (113 earlier); `RESUMED` 23:27:01 | `finished — {'done': 205}` 17 Oct 04:03:49 | — | — |

A second host reboot at 15 Oct 22:22 (`last reboot`; consistent with the second reported kernel panic) fell during the memory-guard pause (runner exited 75 at 22:18:44, resumed 23:27:01). No entry was running, so it left nothing to rerun and no trace in the state.

Each interrupted entry was rerun to `done`:

| entry | state `started` (rerun) | exit | wall | output | size | mtime | SHA-1 (recomputed = both records) |
|---|---|---|---|---|---|---|---|
| `E16.w2_measure.a1` | 2026-10-14 17:23:55 | 0 | 33.8 min | `a1/w2_stamp_axes.npz` | 8,982,788 | 2026-10-14 17:57:44 | `d15920e8d083fe927e8d870588b251abbbd0a946` |
| `E23.aa3b_aa3b.a2` | 2026-10-15 21:36:55 | 0 | 21.7 min | `a2/aa3b_pose_average.json` | 28,492 | 2026-10-15 21:58:31 | `cdc88e0de18f2d0cf3071578bc1ecad1592bc7b4` |
| `E24.dd_train_trained_11_8.a2` | 2026-10-15 23:27:52 | 0 | 14.4 min | `a2/dd/sae/A2_b11_x8.log.json` | 76,847 | 2026-10-15 23:42:12 | `1709fc9329d622747e6b6db6795fb285d3cd568d` |
| `E24.dd_train_trained_11_8.a2` | 2026-10-15 23:27:52 | 0 | 14.4 min | `a2/dd/sae/A2_b11_x8.pt` | 9,478,573 | 2026-10-15 23:42:12 | `fcfb9a47267eee2d115284231c238043c1ff0565` |

Every rerun output's mtime falls after its rerun began, so each SHA-1 is of the rerun's bytes. The runner records no SHA-1 for an interrupted attempt (it clears the attempt's files and resets the record), so there is no earlier SHA-1 to compare against; no `cleared interrupted output` line appears in any runner log, i.e. none of the three interrupted attempts had left a file in its directory. `run_log.md` carries 206 rows: 205 final rows plus one `interrupted (paused)` row for `E24.dd_train_trained_11_8.a2`; no row is `failed` or `skipped(dep)`.

## Blinding

- No content of any output under `runs/ladder_v2/{a1,a2,a3,shared}` was opened, parsed or printed: outputs were `stat`ed and their bytes streamed into SHA-1 only. The same holds for the symlink targets in `artifacts/out` and for the encoder checkpoints (`runs/m_v2*/encoder.pt`), which were hashed, never loaded.
- No entry log in `runs/ladder_v2/logs/` was read (existence, size, mtime only). `memguard.log`, `watchdog.log` and `docs/ladder_v2/runner_log` were `stat`ed only. Runner logs were read only through greps for stage lines and the permitted status words, keeping only ids, statuses, wall-clock, rss and output counts.
- Not opened: any `*_findings.md`, `artifacts/aligned_comparison.md`, `artifacts/out/` (other than hashing the symlink targets above), user memory, `runs/` outside `runs/ladder_v2` (bar hashing the pinned encoders), any A1/A2/A3/M/O2 result report, or `.env`.
- This file contains only ids, statuses, paths, sizes, SHA-1s, counts and timestamps.
