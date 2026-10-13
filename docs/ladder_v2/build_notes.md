# Ladder v2 — build notes (13 October 2026)

What the build did differently from `plan.md`, and why. No results are cited; none were read.

## Deviations from the plan
1. **Per-encoder directories, not per-experiment ones.** The drivers read each other's records by
   fixed name (`t1_ladder.json`, `r_r2.json`, `o1_embeddings.npz`, …). So each encoder's records share
   one directory, `runs/ladder_v2/<a1|a2|a3>/`, and the manifest maps entries to outputs. The shared
   untrained bank is in `runs/ladder_v2/shared/`.
2. **E2 is excluded** (decision 4). Its only implementation, `aligned_score.py c2`, scores the family
   in `powered.json`, which was fixed from M1/M2's scores.
3. **`p2_ladder` runs as its own E5 entry.** `r0`, `s2_verify` and `t1_ladder` read its record.
   **R0 runs before S2/T1, not last** (as `r3→r2→r1→r0` in the plan), because S2 reads `r_r0.json`.
   R3 → R2 → R1 then follow.
4. **`dd_v3` is unchanged** (plan Task 1b.5). The chain calls only its `_stamps_for`. V3′'s v2
   method pairs v1 positions with the reprojected labels in `runs/dd_local_v2`, by design.
   `dd_sae_cards` is not in the chain.
5. **E0 (`h5_probe_lean`) went through the contract too.** It appended to M's
   `artifacts/out/h5_probes.jsonl` and built its own split. It now writes to the encoder's directory
   and takes the identical split. Unset, it is unchanged.
6. **E1 is reused only when provably right** (`artifacts/ladder_v2_c2.py`). The c2 bank is linked only
   if its checkpoint SHA-1 matches the encoder and its ids equal the identical split's uncapped union.
   Otherwise it re-embeds.
7. **Shared, architecture-only items are built once on A1 and linked for A2/A3**, as in the aligned
   chain: the untrained bank (E4), the untrained seeds 1–2 (E7 bank), and the untrained DD tokens and
   SAEs.
8. **D28 planted runs are their own entries**, before each driver's main run (W2, X1, Y3, Z3, AA1,
   AA3a, AA3b).

## The identical set (decision 1), as resolved
`j4_spread_controls._identical_split` takes v1's split and 40,000 stride, then drops galaxies the
active corpus lacks. Anything outside `ladder_env.NINE` refuses. Under A1 it resolves to
39,997 train + 34,828 test. 4 of the 9 fall in v1's split.

## Runner guarantees (`artifacts/ladder_v2_run.py`)
- **Pins:** checked before every entry; a moved pin stops the runner. The pins cover the driver,
  its artifacts import closure, the package tree, the configs and the encoder.
- **Outputs:** every created or changed file is SHA-1'd (`outputs.sha1`) and re-verified before every
  later entry.
- **Guards:** `artifacts/out` and `runs/dd_local_v2` are snapshotted around each entry, and any change
  stops the runner. **So, while the ladder runs, nothing else may write there, and no branch may be
  checked out in this working tree.** Baselines work goes in a separate `git worktree`.
- **Failure:** a failed entry is recorded, its dependants become `skipped(dep)`, and the run
  continues. A retry happens only on `--retry-failed`, which clears the failed attempt first.
- **Interruption:** a SIGTERM takes the entry's process group down with the runner. On restart the
  interrupted entry's files are cleared (bar its `keep` globs) and it reruns. The runner refuses to
  start if the old attempt still holds the heavy lock.
- **Blind:** entry output goes only to `runs/ladder_v2/logs/<id>.log`.

## Verified before the run
- **E10 inertness:** with the contract unset, `u1_bar_winding` reproduces M's record byte-for-byte
  (SHA-1). `artifacts/out` was unchanged afterwards.
- **Split key:** `untrained_split_key` reproduces M's existing bank key.
- **Plumbing run** (scratch manifest, 2 entries): resume after SIGTERM, failure → `skipped(dep)`,
  `--retry-failed`, tamper detection, pin-move stop, and the guard all behaved as specified.
