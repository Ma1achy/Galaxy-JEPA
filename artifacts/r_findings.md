# Brief R — findings

Nonlinear structure and concept geometry on M's 4-epoch checkpoint. **Descriptive, not gating:** no
rung changes, and no selection across checkpoints or draws. The null outcome counts as a result,
not a failure: MLP headroom near zero and paths straight within noise would be positive evidence
for the *feature = direction* working hypothesis, tested across all 37 answers rather than assumed.

## Bottom line

1. **No nonlinear headroom anywhere (R1).** 37 of 37 answers read NO EVIDENCE in both populations,
   and 33 of 37 headroom CIs lie wholly *below* zero. The same MLP decodes random cluster labels at
   AUC 0.89–0.99, so the null has teeth. *Feature = direction* holds for readout, tested across all
   37 rather than assumed, under the ladder's fixed recipe.
2. **P's "size dominates" was the gate's arithmetic (R0).** Under O1's retention rule, none of P's
   22 "confounded" answers loses its effect to matching. What keeps them off R1 is effect size, not
   a nuisance. P's rung counts stand; its stated mechanism does not.
3. **Directions are read linearly, but the conditional means are not all lines (R2, validated by
   R3).** 20 of 36 paths bend past the pre-registered bar, far above the 1.8 expected by chance, yet
   the probe still orders them (median readout 0.93). Most large bends are a **mixture of vote-reach
   groups**, not the concept's shape. Three paths bend robustly (merger, spiral, anything odd).
   Spiral's arc puts uncertain-vote galaxies off the line, which is the uncertainty-geometry brief's
   question, and that brief needs a local vote-count floor first.
4. **D10's missing augmentation leaves an orientation nuisance that is mostly architectural (R3).**
   Ridge R² is 0.505 trained against 0.42–0.44 for random weights.

---

- **Encoder:** `runs/m/encoder.pt`. **Split:** the frozen 40,000 train / 34,829 test.
- **Untrained reference:** three seeds (O1's seed 0, plus 1 and 2 banked here in
  `artifacts/out/r_untrained_seeds.npz`). With K=3, the untrained numbers are a **range**, never a
  spread estimate.
- **Driver:** `artifacts/r_nonlinear.py`. Stages run in a fixed order (r0 → r3 → r1 → r2), because
  each reads what the one before decided. Records are in `artifacts/out/r_{r0,r3,r1,r2}.json`.
- **Package instruments** (tested): `probing/geometry.py`, `probing/orientation.py`,
  `matching.retention_verdict`, `logistic.paired_auc_bootstrap`, `mlp.select_width`,
  `controls.cluster_control_labels`.

---

## R0 — Brief P's "size dominates" was the gate's arithmetic

The ladder judges "survived matching" as *matched AUC ≥ the effect floor*. **21 of the 22** answers
P labelled "confounded" had an unmatched AUC already below 0.7267, so they failed whatever matching
did. Matching moved their AUC by a median of **0.021**. R0 re-judges all 37 under **O1's
pre-registered retention rule, unchanged**, on the ladder's own matched rows. `C` and `C_m` are
3-seed means.

**Faithfulness:** every real matched AUC reproduces P2's `matched_auc` exactly. R0 re-judges the
re-probe P ran, not a different one.

| population | SURVIVES | PARTIAL | COLLAPSES | UNRESOLVED |
|---|---|---|---|---|
| full | **31** | 0 | 1 | 5 |
| conditional | 26 | 0 | 1 | 10 |

Of P's 22 "confounded" answers (full population): **20 SURVIVE, 2 UNRESOLVED, none collapses.** The
median margin retained is 1.00.

- The one collapse is **`t10 arms winding: medium`**, in both populations: AUC 0.523 unmatched,
  barely off chance, and the feature D23 found one untrained seed could decide.
- The UNRESOLVED answers are matched sets too thin to judge, e.g. `star_or_artifact` with 52 matched
  test galaxies. Their verdict is *cannot tell*, not *confounded*.
- **`bulge: dominant`** was the one answer P showed losing an above-floor effect (0.752 → 0.636).
  It SURVIVES: its untrained bar falls with it (0.662 → 0.550), and it keeps 96% of its margin.

**What this changes:** the *reason* P gave, not P's rungs. R1 separately requires AUC ≥ 0.7267, so
P's single clean direction stands. The other answers are **present but below the effect floor**,
not confounded. Corrected in `p_findings.md` §b, the README Catalogue section, and the right panel
of `assets/ladder_catalogue.png`. The ladder's `survive_threshold` is owed a D-entry (`TODO.md`);
production is unchanged.

**Consequence for amendments 5 and 6:** R0 defines "confounded" (PARTIAL or COLLAPSES) for the
matched MLP and the matched-row R2. That set is one feature, `t10 medium`, not P's 19.

---

## R3 — the instrument finds a circle that is certainly there (run first)

Orientation is 180°-periodic, so the target is the doubled position angle, in 12 bins over
[0°, 180°). The angle comes from each cached stamp's own intensity-weighted second moments within
1.5 × `petroRad_r`, which is exactly what the encoder sees, with no SDSS frame convention involved.

**Gate:** `expAB_r ≤ 0.6` and moment axis ratio ≤ 0.8, leaving **26,751 of 74,829** galaxies. The
compact probe-column sidecar `prepare` reads does not carry `expAB_r`, so it is streamed from
`data/probe/metadata.csv` (finite on 100% of the union). The sidecar gap is logged in `TODO.md`
against D13.

| encoder | circle | turns | order violations | eff. dim | straightness | path variance | ridge R² (cos 2θ, sin 2θ) |
|---|---|---|---|---|---|---|---|
| **M** | **recovered** | +1 | 1 | 3.54 | 0.58 | 0.28 | **0.505** |
| untrained s0 | not recovered | 0 | 12 | 2.40 | 0.73 | 0.43 | 0.441 |
| untrained s1 | recovered | +1 | 1 | 2.20 | 0.79 | 0.46 | 0.424 |
| untrained s2 | recovered | +1 | 1 | 2.35 | 0.78 | 0.39 | 0.425 |

**The instrument is validated on M.** The same statistics R2 uses (cross-fitted spectrum,
shuffled-value null, curvature) call a known loop a loop. They separate M's loop from untrained seed
0's, where orientation is linearly readable but the centroids do not wind. So R2's "straight"
verdicts can be read as evidence rather than as the instrument's blindness.

**The D10 nuisance (divergence recorded under D10):** pretraining applied no rotation or reflection
augmentation, so orientation was free to be encoded. It is, but **mostly for architectural
reasons**:

- A random-weight encoder already reads the doubled angle at ridge R² 0.42–0.44. M reads it at
  0.505, so training adds about 0.07.
- M's orientation path is *weaker* in cross-fitted variance than the untrained paths (0.28 against
  0.39–0.46) but more isotropic: effective dimension 3.5 against 2.2–2.4, straightness 0.58
  against 0.73–0.79. The untrained loops are flattened ellipses; M's is rounder.
- Whether an untrained encoder's loop winds depends on the draw: two seeds of three. That is D23's
  lesson again.

Implementing D10 would have to remove what the patch embedding sees for free, not only what M
learned. Whether to do so before the headline run is a decision for its own brief.

---

## R1 — MLP headroom: none, on any of the 37

**37 of 37 answers read NO EVIDENCE, in both populations.** The null outcome is the result: no
answer carries structure a one-hidden-layer MLP finds and the linear probe misses.

**The recipe.** The ladder's own recipe, for parity with L1: depth 1, full-batch Adam, 200 epochs,
weight decay 1e-4, widths 16–512. Width is chosen on an inner 80/20 split of *train* and refit on
full train; test is never seen. L1 took the best of six widths on test, an upward bias the size
of the effect. Headroom = MLP_test − linear_test, with a paired bootstrap on the same resampled test
galaxies.

| | full | conditional |
|---|---|---|
| headroom CI wholly **below** 0 | **33** | 28 |
| straddles 0 | 4 | 9 |
| wholly **above** 0 | **0** | **0** |
| median headroom | −0.011 | −0.022 |
| largest point value | +0.0017 (`winding: medium`) | +0.069 (`disturbed`, CI [−0.07, +0.25]) |

The four full-population straddles are thin buckets or near-chance answers: `star_or_artifact`,
`winding: medium`, `arms: 3`, `arms: more than 4`. Those four read *cannot tell*; the other 33
read *no*.

**The null has teeth: the cluster control task.** k-means is fitted on the 40,000 train embeddings
only. Each cluster gets a random label at the answer's own base rate, so the control labels
*recur* across train and test (permuted labels do not). On those labels the same MLP reaches
AUC **0.99** (C=100) and **0.89** (C=1,000), beating the linear probe by **+0.06** and **+0.18**.
The MLP can exploit nonlinear structure when it is there, and the concepts offer none. Selective
headroom, (MLP−lin)_real − (MLP−lin)_ctrl, has a median of −0.07 and −0.19.

**Memorisation, measured directly.** The permuted-label MLP reaches a train AUC of **0.655** (full,
median) and a test AUC of **0.497**. It memorises a little, and the Hewitt–Liang permuted gap
(median 0.19, reported because asked) cannot see it. This is the L1 correction, measured.

**Untrained headroom (descriptive, three seeds, a range).** Negative too: the median per-feature
range is [−0.013, −0.005], overall [−0.038, +0.021]. M's headroom falls *below* the untrained range
on 22 of 37 answers and above it on 7. On random features as well, the fixed-recipe MLP trails
the linear probe. Training widens that gap rather than creating it: M's representation is, if
anything, *more* linearly readable than an architecture's default.

**Scope.** This holds under the ladder's bounded recipe. A deeper or longer-trained MLP was not
tested, and negative headroom partly reflects the bound. The cluster control shows the bound is not
so tight that the MLP can only fit linear structure.

### Matched MLP (amendment 5): one feature, and its flag is noise

R0's PARTIAL/COLLAPSES set is one answer, `t10 winding: medium`. The same matched rows, the same test
and the same retention rule give **linear COLLAPSES → MLP SURVIVES**, which by the pre-registered
reading is *"separable from the nuisance nonlinearly"*. The numbers do not support that reading:

| | M | untrained bar (3 seeds) | margin |
|---|---|---|---|
| linear, full rows | 0.5226 | 0.519 (0.516–0.522) | **0.003** |
| MLP, full rows | 0.5243 | 0.510 (0.505–0.515) | 0.014 |
| linear, matched rows | 0.5105 | 0.511 | −0.000 |
| MLP, matched rows | 0.5176 | 0.504 | 0.014 |

- M's MLP does not beat M's linear probe: 0.5243 against 0.5226, and R1's headroom CI straddles 0.
- The MLP's margin is larger only because *untrained* MLPs score lower than untrained linear probes.
- The linear margin (0.003) is smaller than the spread of the linear bar across seeds (0.006), so
  the COLLAPSES verdict is itself a verdict on noise.

Reported as **unresolvable at this margin**.

**A gap in O1's retention rule, found here and recorded rather than patched.** The rule is a ratio
of margins with no floor under the unmatched margin. When A − C sits inside the untrained seed
range, "retained" divides noise by noise and can return any verdict. It does not affect R0's
survivors, whose margins are well clear of the range, but the rule needs a margin floor (e.g.
A − C above the resolvable margin) before it is reused. Logged in `TODO.md`.

## R2 — concept geometry: 20 curved, and most of the curvature is a mixture

**The object is E[z | vote fraction].** It carries everything that co-varies with the fraction, so
on the full population it is the shape of the conditional mean, not of the concept alone.

**Construction:** embeddings are z-scored per dimension over the union, label-free. Bins are 10
equal-width, with at least 100 galaxies each and at least 5 bins surviving. Signal covariance is
split-half cross-fitted (averaged over 2 splits), so noise does not manufacture curvature. The null
shuffles values with occupancy preserved. The draw count is sequential: 200 for every feature, topped
up to 4,000 when p ≤ α/H₃₇ = 0.0119. 28 of 36 features were topped up, and the null resolution was
asserted before BY.

### The pre-registered verdicts

**Instrument validated first (R3).** A *straight* verdict here is evidence, not blindness.

| verdict (M, full population) | n |
|---|---|
| **curved** (BY-significant **and** bend = 1 − straightness ≥ 0.10) | **20** |
| straight within noise | 16 |
| uncharacterised (occupancy) | 1 (`star_or_artifact`) |

**Multiplicity (amendment 7).** At 5% uncorrected, a wholly straight world would produce **1.8**
false *curved* verdicts from 36 tests. BY was applied across the family, and 20 is far above that.
The bend magnitude, not the p-value, decides these verdicts: 28 features hit the null's floor
(p = 1/4,001), because at n ≈ 75,000 any detectable bend is significant.

**Readout ceiling.** A bend does not break the linear readout. Along the ladder's logistic direction
the centroids read out at a median Pearson of **0.93** for curved paths and 0.90 for straight ones.
The bend lives *off* the probe's axis, so a direction still orders the concept. This agrees with
R1: curvature in the conditional mean is not nonlinear separability.

**Matched rows (amendment 6):** one feature, `t10 winding: medium`, reads straight within noise
(bend 0.054).

### Three things that change how the 20 read

**1. Mirror pairs measure the instrument's own wobble: about ±0.04 in bend.** Binary questions
(t02, t03, t04, t06) have answer fractions that sum to exactly 1, on the same galaxies. Each pair
is one path measured from both ends, so it should give one bend. It gives two that differ by
0.002 (spiral), 0.018 (odd), 0.041 (bar) and 0.040 (edge-on), from bin-edge assignment of GZ2's
quantised fractions and from the random cross-fit splits. Four of the 20 are the second half of a
pair, so **16 distinct paths**, not 20. Verdicts within ~0.04 of the 0.10 bar are knife-edge.
`smooth` (0.096, straight) and `features or disk` (0.110, curved) are the clearest case.

**2. Most of the large bends are a mixture of vote-reach groups, not the concept's shape**
(`artifacts/r2_reach_split.py` → `artifacts/out/r2_reach_split.json`; descriptive: no null, no BY).
GZ2's conditional questions reach a **median of 5–8 volunteers**. Half their vote fractions rest on
a handful of votes, which pile onto simple fractions (1/4, 1/3, 1/2), and reach itself tracks the
parent answer. Recomputed on the well-voted half (reach ≥ its median) and on the rest:

| answer | bend, all | well-voted half | rest | median reach |
|---|---|---|---|---|
| edge-on: yes | 0.301 | **0.022** | −0.053 | 8 |
| bar | 0.300 | **0.058** | noise | 6 |
| ring | 0.274 | **0.051** | −0.059 | 5 |
| merger | 0.338 | **0.303** | −0.095 | 5 |
| spiral | 0.109 | **0.197** | 0.031 | 6 |
| anything odd: yes | 0.129 | **0.175** | 0.050 | 41 |

For edge-on, **both halves are straight and only their mixture bends.** The share of low-reach
galaxies changes from bin to bin, and the two groups differ along a strong axis, so the bin
centroids swing off the line as the mix shifts. This is the open `TODO.md` item — *vote count is
not merely noise for the uncertainty geometry* — now measured. Across the 20:

| class | n | answers |
|---|---|---|
| **holds on the well-voted half, clear of the ±0.04 wobble** | **3 paths** | merger (0.30); spiral / no spiral (0.20); anything odd: yes / no (0.15–0.18) |
| holds, within ±0.04 of the bar | 4 | cigar-shaped (0.13), completely round (0.12), in between (0.11), features or disk (0.11) |
| **low-reach mixture** (below 0.10 on the well-voted half) | 11 | edge-on ×2, bar ×2, ring, lens or arc, other, disturbed, bulge shape: rounded / no bulge, bulge: no bulge |

Splitting by reach also shifts the population (reach tracks featuredness), so the well-voted column
is a sub-population's shape, not a corrected one.

**3. Learned or architectural?** The untrained references ran without the sequential top-up, so
their p-values cannot fall below 1/201. Their verdict counts (7–10 curved) are **not comparable**
with M's. Compared on bend magnitude instead:

- **Learned:** merger (M 0.34 against untrained 0.02), ring (0.27 / 0.02), odd yes/no (0.13 / 0.00),
  bar (0.30 / 0.09), edge-on: no (0.26 / ≤0), features or disk (0.11 / 0.04).
- **Architectural, or straightened by training:** cigar-shaped (0.27 / 0.32–0.37), in between
  (0.12 / 0.29–0.33), bulge shape: rounded (0.17 / 0.24), bulge: no bulge (0.11 / 0.28–0.31), and
  the spiral pair (0.11 / 0.14–0.16).

### Forward to the uncertainty-geometry brief

Candidate *mechanism* for a weak result there, in order of confidence:

1. **merger**, **spiral**, **anything odd**: bends that survive the reach split and sit clear of
   the instrument's wobble. Spiral's path is a clean arc: PC1 rises monotonically, and PC2 is an
   inverted U peaking at the undecided middle (f ≈ 0.55). This is read from plain bin centroids,
   which is safe only because spiral's bins hold 2,700–28,000 galaxies; thin bins' plain centroids
   carry noise of norm ≈ √(384/n). **Uncertain-vote galaxies sit off the
   line, in a common direction.** That is the geometry the uncertainty brief projects onto a line.
   Merger is learned (untrained bend ≈ 0.02) but not ordered along its principal axis
   (monotonicity 0.02); its readout along the probe is 0.83.
2. **cigar-shaped, completely round, in between, features or disk**: bends near the bar.
3. **The low-reach mixture is a warning, not a candidate.** For any conditional question, a
   full-population E[z|f] mixes vote-reach groups. The uncertainty-geometry brief must decide a
   **local vote-count floor** (the open TODO) before projecting anything. Otherwise it will
   measure the mixture.

---

## Records corrected alongside

- **`l_findings.md`:** the permuted-label control scores against *real* test labels. A galaxy does
  not recur between train and test, so the control cannot see memorisation. "The control measures
  that it did not [memorise]" and "λ=0 is memorisable" are both unsupported as stated. A correction
  note has been added; L1's headline does not rest on either claim.
- **O3 did not measure MLP memorisation.** O3 is the encoder's overfit-one-batch gate. L1 is the
  nearest measurement.
- **D10:** recorded as a spec/code divergence in `DECISIONS.md` and `TODO.md`.
