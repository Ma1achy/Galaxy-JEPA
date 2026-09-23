# Brief T — findings

M's 4-epoch checkpoint, P2's union and split (40,000 train / 34,829 test), both populations.

## T1 — every consumer of the effect floor (D25)

### Pre-registration (written 2026-09-23, before the D25 ladder was run)

**The fix.** Two consumers tested something relative:

- **2A's conditional leg** (`_entangled_map`) judged "A survives matching on B" as *A's matched AUC
  ≥ 0.7267*. It now applies D24's retention rule to A matched on B's vote fraction. The mapping:
  SURVIVES → survived; COLLAPSES → vanished (world correlation); PARTIAL and UNRESOLVED → the leg
  did not attribute (`survived_matching=None`, so `adjudicate_pair` returns *inconclusive*, which
  marks nothing). A's margin is established by construction, since only existence-passing
  directions reach a pair, and the ladder asserts this.
- **The MLP decode** (`rung_from_sweep(..., decode_threshold=effect_floor)`) is no longer
  assigned. There is no untrained-MLP bar at K ≥ 20, so every existence-failing answer reads R4,
  *MLP decode unadjudicated*. The construction and its cost are below; it is not run under the
  resolution gate.

**Inputs the prediction uses** (no number under the new rule):
- P2's recorded 53 full pairs.
- The 41 conditional pairs rebuilt under the old leg (`t1_ladder.py --old-pairs`). The rebuild
  reproduces P2's 53 full verdicts exactly.
- S2's A and C per answer.

A pair's retention needs roughly *M − C_m ≥ ½(A − C)*. The old leg needed only *M ≥ 0.7267*.

**Predicted:**

1. **Nuisance clearance reproduces S2 exactly.** Its code was refactored into the helper it now
   shares with 2A, and the number is unchanged. Every retention verdict, and A, C, M, M_lo and
   C_m, are equal to S2's. So is every AUC.
2. **Old world-correlation verdicts that were arithmetic turn representational.** These are the
   pairs where A was below the floor *before* matching. In full: `no bar × 2 arms`,
   `just noticeable × loose`, `odd yes × disturbed`, `odd no × disturbed`, and the within-question
   pairs whose A sat below. The conditional population has the matching set. I expect most to
   SURVIVE, because matching on another answer's fraction moved AUC by a median of 0.021 in R0.
   Within-question pairs between a binary question's two answers (smooth/features, edge-on
   yes/no, bar yes/no, spiral yes/no, odd yes/no) match A on its own complement. I expect those to
   come back COLLAPSES or UNRESOLVED and stay non-representational.
3. **Some old representational pairs lose SURVIVES.** Where C is high, retention asks more of M
   than 0.7267 did: smooth and features need M ≈ 0.84, no-spiral and no-bulge ≈ 0.80. Those pairs
   can go PARTIAL, and that no longer marks either answer.
4. **Rung changes, full population: none.**
   - The one R1, `irregular`, is in no pair.
   - The two answers that could rise to R1 are `bulge: dominant` and `cigar-shaped`. Both are
     clean and cleared, and both are marked only through pairs with a high retention bar. Each
     also sits in an arithmetic pair I expect to turn representational (`just noticeable ×
     dominant`, `in between × cigar-shaped`, where A was 0.602), which keeps it entangled.
   - Counts stay {R1 1, R2 32, R4 4}.
5. **Rung changes, conditional population: `t08 odd: other` R1 → R2.** Its only pair is
   `other × merger`, within question, cosine −0.31, with 934 matched test galaxies, so it can
   resolve. The old leg called it world-correlation because other's matched AUC fell below
   0.7267. With A − C = 0.119, retention needs M − C_m ≳ 0.06. Under magnitude matching other kept
   110% of its margin, and a within-question non-complement partner should move it less. I expect
   SURVIVES, which makes the pair representational and both answers entangled.
   - **`t09 bulge shape: no bulge` stays R1.** Its only pair (`rounded × no bulge`) has **272**
     matched test galaxies, below 500, so it is UNRESOLVED, then inconclusive, and marks nothing.
   - **`irregular` (conditional) is a candidate to rise, and I expect it stays R2.** It is marked
     only through `no bulge × irregular` (800 matched; M needs ≈ 0.83). I expect SURVIVES.
   - Counts {R1 2, R2 27, R4 8} → **{R1 1, R2 28, R4 8}**.
6. **Mechanisms.** Below-floor answers that enter a newly representational pair are relabelled
   from *present, below the effect floor* to *entangled linear*, and stay R2. Full candidates:
   `disturbed`, `2 arms`, `tight`. Every R4 changes its mechanism text to *MLP decode
   unadjudicated*. That is a label change, not a rung change: no R3 existed, and none can be
   assigned now.
7. **Anything else that moves is unpredicted** and is reported as such.

**The MLP decode, verified rather than assumed** (from R1's record, before any rerun):
- **Full population.** R1 fitted 3 untrained-MLP draws per answer. Against that K = 3 bar, the
  four failing answers' MLP margins are:

  | answer | MLP margin | z |
  |---|---|---|
  | `star/artifact` | +0.026 | ≈ 0.4 |
  | `winding: medium` | +0.014 | ≈ 2.3 |
  | `arms: 3` | +0.032 | ≈ 2.1 |
  | `arms: 4` | +0.038 | ≈ 1.3 |

  None approaches a family-corrected bar: BY rank-1 at m = 4 needs p ≈ 0.006, and at m = 37,
  3.2e-4. So in full, a correct MLP existence test would assign no R3.
- **Conditional.** Not yet measured: R1's untrained MLP was fitted on the full population only.
  I report it after the rerun: a K = 3 bound, and a list of which of the 8 conditional R4s have a
  competitive nuisance. Under D24 a failing answer's clearance is UNRESOLVED, so for those answers
  R3 is structurally impossible whatever the decode test says.

**The construction proposed for the MLP decode (not run).**
- *R3 ⇔* the MLP clears an existence test against an **untrained-MLP bar**.
  - The statistic is D23's:
    `z = (MLP_real − mean_K MLP_untrained) / √(sd_K² + se²)`,
    Student-t with df = K − 1, K ≥ 20 (`nulls.K_MIN`).
  - BY runs across the answers that reach the MLP rung in that population.
  - Width is chosen on an inner split of train, as in R1, and never on test. The same protocol
    runs on every untrained draw.
- The permuted-label selectivity ceiling stays.
- Nuisance clearance for an R3 candidate uses the MLP's own retention: matched MLP, with
  untrained-MLP bars on the matched rows (K = 3, per D24), and the MLP existence test as its
  margin floor.
- **Cost:**
  - 17 more untrained matrices (K = 20 with R's 3): ≈ 3 h to extract, ≈ 1 GB banked as fp16.
  - 12 failing answers × 20 draws of the inner sweep plus refit: ≈ 4 h.
  - Matched MLP on any survivors: ≲ 0.5 h.
  - **≈ 7–8 h.**

*Pre-registration ends here. SHA-1 of the file up to this line:
`aff39c7d65ff4b42d7025cae87800584be404e52`, taken before the rerun started (20:52 BST).*

### Result (`artifacts/t1_ladder.py` → `artifacts/out/t1_ladder.json`, 3 h 01 m)

**Rungs: as predicted, in both populations.**

| population | before (S2) | after (D25) | predicted |
|---|---|---|---|
| full | R1 1, R2 32, R4 4 | R1 1, R2 32, R4 4 | unchanged ✓ |
| conditional | R1 2, R2 27, R4 8 | **R1 1, R2 28, R4 8** | `other` R1 → R2 ✓ |

- **`t08 odd: other` (conditional) moves from R1 to R2, entangled.** `other × merger` SURVIVES, with
  1.09 of its margin retained on 934 matched test galaxies. So the pair is representational. Its
  old world-correlation verdict came from the floor alone.
- **`t09 bulge shape: no bulge` stays R1** (conditional). Its only pair is UNRESOLVED on 272 matched
  test galaxies, below the 500 needed.
- **`irregular` (conditional) stays R2.** `no bulge × irregular` SURVIVES (1.08).
- **In full, `bulge: dominant` and `cigar-shaped` stay R2.** The arithmetic pairs I named keep
  them entangled: `just noticeable × dominant` 1.03 and `in between × cigar-shaped` 1.23, both
  SURVIVES.
- **Nuisance clearance reproduces S2 exactly** wherever S2 recorded it. Every AUC, C, C_m, M_lo and
  verdict is equal. The failing answers now carry their clearance on the record; S2's did not.
  Every one is UNRESOLVED, as D24 requires.

**Mechanisms: relabelled from *below the effect floor* to *entangled*, all staying R2.**
- Full: `disturbed` and `2 arms`, as predicted.
- `tight` did not move: `tight × loose` came back PARTIAL.
- Conditional: `bar`, `no bar`, `in between` and `can't tell`.
- All 12 R4s now read *MLP decode unadjudicated*.

**The MLP decode is verified: it changes no rung in either population, by construction.** Every one
of the 12 existence-failing answers has **all five** nuisances competitive. Their real AUCs are low,
so each nuisance decodes at least as well. Their clearance is therefore UNRESOLVED (D24), and R3 is
impossible whatever the decode test says. The old absolute rule would have fired once, before
demotion: conditional `disturbed` at width 16. Demotion took it to R4 anyway. The K = 3 conditional
bound the pre-registration promised is not needed.

**Unpredicted, reported rather than suppressed.**

1. **World correlation all but disappears.**

   | population | before (old leg) | after (D25) |
   |---|---|---|
   | full | 35 representational, 18 world | **48 representational, 0 world, 5 inconclusive** |
   | conditional | 23 representational, 18 world | **37 representational, 1 world, 3 inconclusive** |

   - The one survivor is conditional `bar × no bar`, which COLLAPSES.
   - Every old world-correlation verdict was a matched AUC under 0.7267. Retention says the
     direction did not vanish in any of them but that one.
2. **Binary-question complements behaved worse than I predicted.**
   - I expected COLLAPSES or UNRESOLVED. That held for `edge-on yes × no` and `bar × no bar`
     (UNRESOLVED; under 10% of the test set survives matching) and for `spiral × no spiral`
     (PARTIAL).
   - `smooth × features` (retained 0.50) and `odd yes × no` (0.56) SURVIVE. They count as
     representational, but that means little: A matched on its own near-complement.
3. **P's "geometric necessity" reading is withdrawn.** P called `edge-on yes/no × cigar-shaped`
   (|cos| ≈ 0.96) world correlation and "the verdict function working". Under retention both
   SURVIVE: edge-on stays decodable with cigar-shaped fraction held fixed.
   - The within-question pairs, which P said do not belong in an entanglement count, are
     unaffected in principle.
   - P §(f)'s Hart reading needs a separate note. It says stage 2 was licensed because bar ×
     winding "survived as a world-correlation candidate". But bar × tight and bar × loose
     (cosines −0.24 and +0.04) never cleared the 0.30 pair floor, so stage 1 never adjudicated
     them in either run. Flagged in `p_findings.md`; not re-run here.

