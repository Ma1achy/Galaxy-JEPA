# Brief P — findings

The catalogue. All 37 Scheme 1 answers on M's 4-epoch checkpoint, rungs emitted by the hard gate,
both populations, power travelling with every verdict.

- **Encoder:** `runs/m/encoder.pt`, M's 4-epoch checkpoint — the pre-registered product. O2's
  2-epoch checkpoint reached 0.9678, the best number either run produced, and was **not**
  substituted: choosing it would be AUC-based selection across training draws, 1C by another route.
- **Headline, with its range:** M 4-epoch consensus AUC **0.9646**; range across two training draws
  **[0.9609, 0.9646]**, splits held fixed. That is training-draw variance with n=2 — a range, not a
  variance, and not an interval on the headline.
- **Split:** the frozen 40,000 train / 34,829 test. The effect floor was frozen against it and the
  untrained bank is keyed to it; moving it would leave the floor no longer corresponding to the
  evidence that justified it. The power cost is reported, not avoided.
- **Existence:** D23's untrained-z, K=30 (`artifacts/out/p1_untrained_bank.json`).
- **Effect floor:** 0.7267, frozen (D22).
- **Not a smoke, not stamped.** `probe.yaml` carries `smoke: false` and an empty
  `escape_hatches`, so every grounded default ran at full strength. But the driver calls
  `run_ladder` directly to keep the frozen split, so the record has no `RunStamp` — real evidence,
  at artifacts level. Record: `artifacts/out/p2_ladder.json`.
- **Cost:** 1,900 s extraction (three sources) + 6,081 s full + 2,800 s conditional ≈ **2.8 h**,
  against a 5.1 h projection from the 3-feature dry run. The untrained bank cost 5.4 h separately
  and is reusable across every future encoder of this architecture.

---

Figures: `assets/ladder_catalogue.png` (§a, §b), `assets/ladder_power.png` (§c, §d),
`assets/concept_structure.png` (§e, §f, §g). All three regenerate from the records above via
`uv run python artifacts/readme_figures.py`.

---

## (a) The headline finding: almost everything exists, almost nothing is clean

| rung | full | conditional |
|---|---|---|
| R1 — clean linear direction | **1** | 1 |
| R2 — present, not clean | 32 | 28 |
| R3 — non-linear only | 0 | 0 |
| R4 — not recoverable | 4 | 8 |
| underpowered | 5/37 | 10/37 |

One feature of thirty-seven is a clean linear direction: **`t08_odd_feature_a22_irregular`**, AUC
0.7462, surviving matching at 0.7428 on 29% of the test set. Everything else that exists is
entangled or confounded. *(Corrected in §b: "confounded" was the gate's arithmetic. 21 of those 22
sit below the effect floor unmatched, and none loses its effect under matching.)*

This is not a negative result about the encoder. Thirty-three of thirty-seven answers clear their
own untrained bar — the representation *contains* the tree. What it does not contain is thirty-seven
separable axes. The standing question was *is a human concept a direction in the representation*,
and the catalogue's answer is: it is a direction, but almost never an independent one.

**R3 is empty, and that number cannot be read.** `ladder.py:248` uses the frozen `effect_floor`
(0.7267) as the MLP decode threshold, and a feature reaches that branch only *because* it failed
existence — so the MLP is asked to clear 0.7267 on buckets sitting near 0.55. Unreachable by
arithmetic. R3 = 0 means "the gate could not have fired", not "no feature is non-linearly
decodable". Changing it is a mechanism change needing its own D-entry; reported as a limitation.

---

## (b) The mechanism — corrected by Brief R0: the effect floor, not apparent size

> **Correction (2026-09-22, Brief R0 — `artifacts/r_findings.md`, `artifacts/out/r_r0.json`).**
> This section originally read *"apparent size, doing more work than everything else combined:
> nineteen of thirty-seven answers lose their effect when apparent size is matched."* That was an
> artefact of the gate, and my error in writing it up: D22 and O1 had both recorded the arithmetic.

The ladder judges "survived matching" as *matched AUC ≥ the effect floor* (`ladder.py`,
`survive_threshold=config.effect_floor`). **21 of the 22** answers labelled "confounded" had an
*unmatched* AUC already below 0.7267, so they failed that test whatever matching did. Matching moved
their AUC by a median of **0.021**.

Re-judged under **O1's pre-registered retention rule** — does the margin over the untrained bar
survive, measured on the same matched rows with the untrained bar re-measured there (three seeds)?
Full population:

| P's mechanism | n | R0: SURVIVES | UNRESOLVED | PARTIAL | COLLAPSES |
|---|---|---|---|---|---|
| confounded by size | 19 | **17** | 2 | 0 | 0 |
| confounded by magnitude | 3 | **3** | 0 | 0 | 0 |
| entangled linear | 10 | 10 | 0 | 0 | 0 |
| not recoverable | 4 | 0 | 3 | 0 | 1 |
| clean linear direction | 1 | 1 | 0 | 0 | 0 |

**None of the 22 “confounded” answers loses its effect when matched on size, magnitude or
redshift.** The median margin retained is 1.00. The one collapse is `t10 arms winding: medium` (AUC 0.523 unmatched — barely off
chance, and the feature D23 found a single untrained seed could decide). The UNRESOLVED are matched
sets too thin to judge (`star_or_artifact`: 52 matched test galaxies).

**What this changes, and what it does not.** The rung table in §a stands: R1 separately requires
AUC ≥ 0.7267 (`existence.clean`), so the 21 below-floor answers could not have been R1 on effect
size alone. What was wrong is the *reason* given. The mechanism label checks nuisance clearance
before entanglement, so an answer below the floor is labelled "confounded" before anything else is
asked. The accurate reading is **present, but below the effect floor**. Whether these answers are
also entangled is hidden by that label order; P2's record does not keep the metric.

The "size axis" story in the first draft — Galaxy Zoo votes depend on how large a galaxy appears —
may still be true of the representation. This catalogue does not show it. The
population comparison that followed ("conditioning replaces size with magnitude") compared the same
artefact across two populations and is withdrawn with it. Under retention the conditional
population reads 26 SURVIVES / 10 UNRESOLVED / 1 COLLAPSES (`t10 medium` again).

The ladder's `survive_threshold` is a defect on the verdict path, flagged in `TODO.md` and owed a
D-entry. Production is unchanged, since this brief changes no rung.

---

## (c) Power: five features cannot resolve, and one of them looks like a result

The pre-registered rule is a **margin over the feature's own untrained bar**, not an absolute MDE.
The first form of this rule was wrong and testing it against real numbers caught it: `MDE = C_f +
(z_.80 + z_BY)·se_real` compared to `effect_floor` declared `t01_featured-or-disk` — the
best-powered feature in the catalogue — underpowered, because its bar (0.7905) already exceeds the
floor (0.7267) and the inequality then holds by arithmetic at any N.

Underpowered in the full population:

| feature | AUC | bar | margin | positives |
|---|---|---|---|---|
| `t01_star_or_artifact` | 0.7738 | 0.7247 | 0.2243 | **28** |
| `t11_arms_number_4` | 0.5805 | 0.5351 | 0.1003 | 161 |
| `t08_dust_lane` | 0.7124 | 0.5905 | 0.0604 | 471 |
| `t11_arms_number_3` | 0.5568 | 0.5206 | 0.0591 | 448 |
| `t11_more_than_4` | 0.6941 | 0.5666 | 0.0573 | 415 |

**`t01_star_or_artifact` is the case the rule exists for.** AUC 0.7738 sits *above* its untrained
bar of 0.7247 and would read as a finding. Twenty-eight positives in 34,829 give se 0.0626 and a
resolvable margin of 0.2243 — nothing this bucket could have shown would have survived BY. Its R4
means **cannot resolve at this N**, and must never be read as a scientific null.

**The margin orders by the size of the *rarer* class, not by the bucket.** Plotting it against
positives alone gives Spearman −0.71 and a misleading upturn at the high end; against
`min(positives, negatives)` it is −0.91 and monotone. The majority-class answers are the reason:
`t02_edgeon_no` has 29,641 positives but only 4,710 negatives, and it is the 4,710 that bind. An
earlier draft of this document said "orders exactly by bucket size" — that was wrong, and the
figure is what caught it.

---

## (d) The population comparison is mostly a power collapse, not a second opinion

Twelve of thirty-seven features change rung between populations. The positives tell the story:

| feature | change | AUC | positives |
|---|---|---|---|
| `t08_lens_or_arc` | R2→R4 | 0.7039 → 0.3760 | 2004 → **5** |
| `t08_disturbed` | R2→R4 | 0.6504 → 0.7087 | 4515 → **15** |
| `t09_boxy` | R2→R4 | 0.5847 → 0.4815 | 1198 → **13** |
| `t05_bulge_dominant` | R2→R4 | 0.7517 → 0.7215 | 2288 → **20** |
| `t11_arms_number_4` | R4→R2 | 0.5805 → 0.7310 | 161 → **30** |
| `t09_no_bulge` | R2→**R1** | 0.7066 → 0.7944 | 4025 → 433 |
| `t04_no_spiral` | R2→R4 | 0.8457 → 0.6979 | 23524 → 1306 |

The consensus gate removes 99%+ of the positives from the deep `t08`/`t09` answers. A verdict on
five galaxies is not a verdict. Where the rung moves *up* — `t11_arms_number_4` climbing from 0.5805
to 0.7310 on thirty positives — the AUC is rising as the sample collapses, which is small-sample
instability, not a signal emerging.

**This is the strongest argument for the pre-registered choice to read verdicts from `full`.** The
conditional column was expected to say where a signal lives; on this corpus it mostly says how few
galaxies survive a majority gate. `t09_no_bulge` reaching R1 on 433 positives is the one change
that might be substantive, and it needs its own look.

**The caveat that travels with reading `full`:** a full-population verdict answers *"can the encoder
read this among galaxies where the question was ever asked"*, not *"among galaxies that genuinely
have the parent property"*. For deep features the full population includes minority-route galaxies
and may dilute the signal.

---

## (e) Entanglement geometry

Thirty-three existence-passing directions, from the ladder's own `W`.

- **Concept span:** effective rank **25.24** of the embedding's **83.13** — span ratio **0.304**.
  Thirty-three concepts occupy about twenty-five independent directions, inside a representation
  with eighty-three. The tree is neither collapsed onto a handful of axes nor spread across the
  whole space.
- **Marchenko–Pastur:** significant. The eigenspectrum's top edge exceeds what random directions of
  this count and dimension would produce, so the structure is real rather than an artefact of
  stacking thirty-three vectors.
- **First component loadings:** `smooth` −0.298, `features_or_disk` +0.297, `cigar_shaped` +0.285,
  `edgeon_no` −0.263. The dominant axis is the smooth/featured split with elongation loading onto
  it — the tree's root question, which is what should dominate.

### Pair verdicts, and a correction to how they should be counted

Fifty-three pairs cleared the cosine floor. **Thirteen of them are within-question** — two answers
to the same tree node, cosines from −0.99 to −0.49, negative *by construction* because the answers
are near-complements. They are label-scheme structure, not representational structure, and they do
not belong in an entanglement count. The record now carries `pair_verdicts_across_question`
alongside the raw list.

| | within-question (13) | across-question (**40**) |
|---|---|---|
| representational entanglement | 1 | **34** |
| world correlation | 12 | 6 |

The across-question read is the one that carries weight: **34 representational, 6 world
correlation**.

The six world-correlation pairs include the two strongest associations in the matrix —
`edgeon_no × cigar_shaped` at **−0.965** and `edgeon_yes × cigar_shaped` at **+0.957**. Both carry
`survived_matching = False` — stage 1 saying the association is a real correlation in the data
rather than a representational artefact. All six world-correlation pairs do. It plainly is: an edge-on disc *is* cigar-shaped in projection. The apparatus recovering a
geometric necessity and labelling it world-correlation rather than entanglement is the verdict
function working.

> **Correction (Brief T1, D25).** This was the effect floor again. The conditional leg judged "A
> survives matching on B" as *matched AUC ≥ 0.7267*. Under D24's retention rule, `edge-on yes/no ×
> cigar-shaped` both SURVIVE: edge-on stays decodable with cigar-shaped held fixed. Full-population
> world correlation goes from 18 pairs to **0** (48 representational, 5 inconclusive). The
> "geometric necessity" reading is withdrawn. `artifacts/t_findings.md` §T1.

**Both of v1's readable constants reproduced.** v1's Figs 18–19 survive only as three prose
numbers, measured on the PyPI `galaxy-datasets` release. Against this pull and this encoder:

| v1 | v1's value | here |
|---|---|---|
| edge-on × cigar-shaped | +0.83 | **+0.957** |
| bar × 2-arms | +0.56 | **+0.419** |
| 3-arms ↔ 4-arms | 0.16 | — both failed existence |

Two of three recovered at the same sign and comparable magnitude through a different dataset and a
different representation, which is all a continuity reference can be asked to do. The third cannot
be checked: 3-arms and 4-arms are the two starved buckets (448 and 161 positives) and neither
reached the entanglement set.

---

## (f) D13's hard case: physics, not classification bleed

Stage 1 (the 2A conditional cross-check) decides whether entanglement is representational or
reflects a real correlation in the data. Stage 2 (invariance plus literature) decides whether a real
correlation is astrophysics or a labelling artefact. Both recorded, in that order.

Stage 2's anchor is directional rather than a judgement call. Hart et al. measured spiral arms in
strongly barred galaxies as roughly **4–6° looser** than in unbarred ones. So if the encoder's bar
direction is tracking physics it should lean towards *loose* winding specifically; if it is tracking
confident classification it should lean towards every spiral answer about equally.

**Verdict: `physics_consistent`.**

| bar × | cosine |
|---|---|
| loose winding | **+0.042** |
| tight winding | **−0.239** |

Spread 0.282. The bar direction is not merely nearer loose — it is *anti-aligned* with tight. That
is Hart's ordering, and it is the opposite of uniform bleed.

**One caveat, stated because it changed the code.** Medium winding failed existence (R4, AUC 0.5226)
and the test originally refused the whole reading for want of the middle point. That was refusing on
the wrong ground: Hart's prediction is about the tight-to-loose axis, and both endpoints reached the
set. The test now needs the bar and both endpoints, with medium required to sit between them only
when it is present. This verdict therefore reads the tight-to-loose axis alone, and was
re-adjudicated from the serialised cosine matrix — a deterministic function of the recorded numbers,
not a second run.

---

## (g) The encoder against human vote structure, same corpus

v1's Figs 18–19 were computed on the PyPI `galaxy-datasets` release, not this pull. Laying that
matrix against an embedding matrix built here would mix **two** differences at once — dataset and
representation — and neither could be read off the result. So the comparison is same-corpus: the
human vote-correlation matrix computed on the 230k galaxies already in hand, pairwise-complete over
each pair's intersecting eligible sets, with the overlap count carried per cell.

**Spearman +0.643 over 528 pairs.** The encoder's concept geometry substantially tracks the
structure of human voting on the same galaxies, without ever having seen a vote.

The disagreements are the output, not the agreement:

| pair | cosine | votes | |
|---|---|---|---|
| `t07_completely_round × t07_in_between` | −0.018 | −0.574 | encoder separates less |
| `t05_obvious × t05_dominant` | +0.314 | −0.141 | encoder ties them |
| `t01_features_or_disk × t08_irregular` | −0.342 | +0.108 | encoder separates |
| `t02_edgeon_no × t09_boxy` | +0.400 | −0.034 | encoder ties them |
| `t01_smooth × t08_disturbed` | +0.389 | −0.041 | encoder ties them |

Two patterns. Where the encoder ties things the voters separate (`edgeon_no × boxy`,
`smooth × disturbed`) the candidate explanation is the size axis of §(b) — both members read along
it. Where the encoder separates things the voters tie (`features_or_disk × irregular`) it may be
doing better than the labels, since vote correlation carries the voters' own confusions.

---

## (h) Normality: 36 of 37, and the one failure is the starved bucket

D23 buys BY a usable p-value at the price of a distributional assumption, and that assumption is
the weakest joint in the construction. Tested per feature, reported whatever it says.

**36 of 37 pass Shapiro–Wilk at α=0.05.** One fails:

> `t09_bulge_shape_a26_boxy` — p=0.0176, skew +0.98, excess kurtosis +0.23, K=30. Its existence
> p-value is **model-based in a region the data cannot validate**: BY's bar lives ~3.4 standard
> deviations out and 30 samples do not describe that tail.

Boxy is also the bucket the power rule flagged as starved (1,198 test positives, 89.8% on ≤2 votes)
and the one whose conditional verdict rests on thirteen galaxies. Three independent diagnostics
pointing at the same feature is the apparatus being coherent.

One failure in thirty-seven is about what α=0.05 predicts, so the bank as a whole holds. Boxy
specifically does not, and that travels with its verdict rather than being averaged away.

**How far to trust the passes.** At K=30 Shapiro–Wilk is noisy enough to reject genuinely normal
data — a normal draw in the test suite lands at p=0.047. Thirty-six passes mean "no detectable
departure at this K", not "normal in the far tail". BY's rank-1 bar is a ~3.4σ statement and thirty
samples do not reach there. This remains the weakest joint in D23 and is not resolved by having
tested it.

---

## (i) The untrained bar, measured

`artifacts/out/p1_untrained_bank.json` — K=30 × 37, keyed to model `c44bdef1dbdf5acc` and split
`1b5e59c63271592d`. Architecture-determined: it never touches a trained checkpoint, so it is
reusable across every future encoder of this architecture.

- sd of the bar: median **0.0043**, range 0.0027–0.0296
- widest: `t01_star_or_artifact` **0.0296** (bar 0.7247) — four to five times any other, and the one
  feature whose z-denominator is governed by the bar's seed-dependence rather than the real AUC's
  sampling error
- highest bars: `features_or_disk` **0.7905**, `smooth` **0.7873**, `no_spiral` 0.7566

That last line is the recalibration Q0 made for mean-cosine showing up again in AUC. An untrained
ViT reads the smooth/featured split at 0.79 before it has learned anything. **The existence bar is
not 0.5**, and existence means clearing *that*.

N2's three-seed estimate put the median spread at 0.0073; at K=30 it is 0.0043. The earlier figure
was inflated by estimating a standard deviation from three points.

---

## Limitations, carried forward

1. **The effect floor governs 37 features having been calibrated on 6.** J5's fourth objection,
   unresolved, travels with every verdict in this catalogue.
2. **R3 is unreachable for the features that reach it** (§a). A mechanism change needing its own
   D-entry; not made here.
3. **Single objective.** Fig 3 is meant to be comparative across JEPA/MAE/MoCo for Framing B. This
   is what one objective can say and no more.
4. **A rung is not a property of a feature alone.** `t01_features_or_disk` came back R1 in the
   3-feature dry run and R2 in the full catalogue: the entanglement test consults the other
   directions, so adding features can move a verdict. The 37-feature reading is the pre-registered
   one; the dry run's is not a second opinion.
5. **Uncertainty geometry deferred** (`uncertainty_geometry: false`) — the high-beta headline, with
   an open decision of its own about a vote floor local to the uncertainty test.
6. **Stage 1 licenses stage 2.** The Hart reading in §(f) is only meaningful because the
   bar+winding association survived as a world-correlation candidate. Reported in that order.
   *(Flag, Brief T1: `bar × tight` and `bar × loose` (cosines −0.24, +0.04) never cleared the 0.30
   pair floor, so stage 1 never adjudicated them. `bar_winding_alignment` ran unconditionally. The
   premise of this item needs checking before §(f) is cited.)*
