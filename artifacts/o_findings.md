# Brief O — the floor frozen, and the confound objection answered

Three phases reported here. **O0** froze the last open statistical gate. **O1** is the decisive
measurement: it answers the standing objection that the probe detects observing conditions rather
than morphology. **O3** is the overfit-one-batch gate, reported against the existing record of real
runs. **O2** (second training seed to 4 epochs) is built and awaiting the machine; its section is
marked as such rather than left blank.

No rung verdicts are assigned anywhere in this document. That was the condition on reading O1, and
it holds.

---

## O0 — the effect floor is frozen at 0.7267

`effect_floor: 0.7267`, form **(a)**, absolute, recorded as `EffectFloorFreeze` in
`configs/probe.yaml` with `derived_from` / `frozen_at` / `frozen_by` / `rationale` all populated.
Spec register item 4 is closed; **D22** records it. `refit`/override is refused by
`_freezes_agree_with_the_live_values`, and `headline=True` is permitted only under the freeze.

**Chosen by structure, not by which features it admits.** Every value in **(0.6538, 0.7995]** gives
an identical partition of the six probed features — a band 0.146 wide. 0.7267 is its centre,
0.0729 from either flip point. The old placeholder 0.6500 sat 0.0038 from a flip; the null ceiling
0.7908 sat 0.0087 from one.

**The margin form was measured and rejected.** The decisive ground: the per-feature null ceiling is
*exactly* the supremum of `existence_null_samples`, verified against the production function over
200 random cases. A margin floor is therefore a **tightening of existence**, not a second gate —
and "it no longer does the existence work" is the one property 3B relies on when it grounds the
floor. A margin form would have needed its own D-series entry arguing a mechanism change; it did
not earn one.

**The existence bar is architecture-determined** — stated as the bar's definition, not left
implicit. `untrained_encoder_matrix` (`controls.py:140-168`) never sees the trained checkpoint, so
the bar is a property of (architecture, seed, galaxies) alone. M's untrained nulls came back
identical to J's to four decimals. The untrained singleton dominates every resampled draw, so
`existence_null_samples` has zero variance and the existence test reduces exactly to
`real_auc > untrained_encoder_auc`.

**Two limitations travel with the value, in the freeze record itself:**

1. **J5's fourth objection is unresolved.** n = 6 cannot locate a threshold for a catalogue of 37.
   The band argument softens this; it does not remove it.
2. **t10-medium's verdict is seed-dependent** (+0.0065 at seed 0, +0.0006 at seed 1). No floor
   fixes that. It is recorded to be **reported, not hidden** — and O1 below has now independently
   found the same feature to be the one that does not survive matching.

---

## O1 — matched evaluation: the objection does not hold

### The objection, stated in its own terms

> *You are not detecting spiral arms. You are detecting that nearby, bright, well-resolved galaxies
> look different.*

This is licensed by the nuisance panel on M's encoder, which is **worse** than J's: magnitude
0.9033, size 0.9061, SNR 0.8687, redshift 0.8371, PSF 0.8186 — against featured-ness at 0.8845 and
every other morphology feature below 0.66. PSF moved 0.5813 → 0.8186 between J and M, so the recipe
fix improved the confounds *faster* than it improved morphology. Every nuisance beats every
morphology feature except featured-ness. That is the serious finding, not a footnote, and it is
what O1 had to answer.

### Method, stated before the numbers

M's 4-epoch checkpoint (`runs/m/encoder.pt`, step 101,308, λ=0 under D21, D17 schedule, stamped
`v2:61330a0012234374`). One split: 40,000 train / 34,829 test, the same union N1 used. Real and
untrained embeddings extracted once (571 s and 1,147 s) and banked to
`artifacts/out/o1_embeddings.npz`; everything after is numpy and logistic fits. Total 1,422 s.

**Matching method.** `matching.stratified_match` is **quantile-binned stratification with
class-balanced downsampling** — not propensity scoring, not caliper nearest-neighbour. The nuisance
is cut into `n_strata = 5` quantile strata; within each stratum the per-stratum minority count of
each class is kept. So the **"tolerance" is the stratum width**, a property of the data rather than
a parameter, and it is reported as measured quantile edges per feature per nuisance in
`artifacts/out/o1_matched.json`. Train matches on `seed`, test on `seed + 1`.

**Five single nuisances** — magnitude, size, SNR, redshift, PSF — **and one joint
magnitude × size match**, implemented as 5 × 5 quantile cells with class balancing inside each
cell. The objection is a *joint* one; matching one nuisance at a time does not answer it.

**Alignment** uses the production contract throughout: every match vector comes from
`labels.nuisance_value(name, feature_ids(...))`, never the raw id list, and real and untrained
embeddings are asserted co-indexed by `np.array_equal` on their object ids before anything is fit.

**The driver applies `labels.nuisance_valid`; production does not.** See the P0 in `TODO.md`.

### The pre-registered read, fixed before any number was seen

For feature *f* with unmatched AUC `A`, matched AUC `M`, and untrained-encoder null `C` and `C_m`
measured on the unmatched and matched subsets respectively:

```
SURVIVES  : (M - C_m) >= 0.5 * (A - C)  AND  M's CI lower bound > C_m
COLLAPSES : (M - C_m) <= 0              OR  M's CI contains C_m
PARTIAL   : between the two — real, but substantially confounded
UNRESOLVED: < 500 matched test galaxies, or < 10% of the unmatched test set surviving
            (a statement about the SAMPLE, not the signal — never folded into COLLAPSES)
```

The floor is **not** used as the survival bar. Production passes
`survive_threshold=config.effect_floor` = 0.7267, and four of the six features sit below that
*unmatched* — they would fail by arithmetic whatever matching did. The 0.5 retention figure is a
declared choice, not a derived one.

### Results

`retained` = (M − C_m) / (A − C). The SURVIVES bar is 0.50.

| feature | unmatched A [CI] | C | nuisance | matched M [CI] | C_m | n test (share) | retained | verdict |
|---|---|---|---|---|---|---|---|---|
| **t01** featured-or-disk | 0.8845 [.8809,.8882] | 0.7908 | magnitude | 0.8751 [.8702,.8800] | 0.7744 | 18,038 (51.8%) | **1.08** | SURVIVES |
| | | | size | 0.8021 [.7952,.8088] | 0.7137 | 14,470 (41.5%) | 0.94 | SURVIVES |
| | | | snr | 0.8730 [.8679,.8778] | 0.7769 | 18,038 (51.8%) | 1.03 | SURVIVES |
| | | | redshift | 0.8539 [.8485,.8594] | 0.7496 | 18,038 (51.8%) | 1.11 | SURVIVES |
| | | | psf | 0.8829 [.8782,.8877] | 0.7899 | 18,038 (51.8%) | 0.99 | SURVIVES |
| | | | **mag × size** | 0.7968 [.7899,.8038] | 0.7062 | 14,406 (41.4%) | **0.97** | **SURVIVES** |
| **t02** edge-on | 0.7995 [.7933,.8059] | 0.6368 | magnitude | 0.7871 [.7792,.7955] | 0.6186 | 11,008 (32.0%) | 1.04 | SURVIVES |
| | | | size | 0.7959 [.7878,.8041] | 0.6224 | 10,878 (31.7%) | 1.07 | SURVIVES |
| | | | snr | 0.7894 [.7810,.7979] | 0.6240 | 11,008 (32.0%) | 1.02 | SURVIVES |
| | | | redshift | 0.8076 [.7998,.8156] | 0.6492 | 11,008 (32.0%) | 0.97 | SURVIVES |
| | | | psf | 0.7927 [.7846,.8004] | 0.6285 | 11,008 (32.0%) | 1.01 | SURVIVES |
| | | | **mag × size** | 0.8003 [.7922,.8080] | 0.6376 | 10,878 (31.7%) | **1.00** | **SURVIVES** |
| **t10** arms tight | 0.5889 [.5814,.5964] | 0.5474 | magnitude | 0.5875 [.5794,.5961] | 0.5474 | 18,224 (90.0%) | 0.97 | SURVIVES |
| | | | size | 0.5819 [.5738,.5898] | 0.5410 | 18,002 (88.9%) | 0.99 | SURVIVES |
| | | | snr | 0.5818 [.5735,.5901] | 0.5411 | 18,224 (90.0%) | 0.98 | SURVIVES |
| | | | redshift | 0.5910 [.5828,.5988] | 0.5467 | 18,224 (90.0%) | 1.07 | SURVIVES |
| | | | psf | 0.5879 [.5800,.5961] | 0.5468 | 18,224 (90.0%) | 0.99 | SURVIVES |
| | | | **mag × size** | 0.5695 [.5612,.5779] | 0.5284 | 17,932 (88.6%) | **0.99** | **SURVIVES** |
| **t10** arms medium | 0.5226 [.5144,.5307] | 0.5160 | magnitude | 0.5105 [.5011,.5192] | 0.5070 | 14,864 (73.4%) | 0.53 | **COLLAPSES** |
| | | | size | 0.5105 [.5015,.5198] | 0.5069 | 14,738 (72.8%) | 0.55 | **COLLAPSES** |
| | | | snr | 0.5024 [.4930,.5119] | 0.5005 | 14,864 (73.4%) | 0.28 | **COLLAPSES** |
| | | | redshift | 0.5150 [.5056,.5239] | 0.5089 | 14,864 (73.4%) | 0.93 | **COLLAPSES** |
| | | | psf | 0.5191 [.5106,.5288] | 0.5165 | 14,864 (73.4%) | 0.40 | **COLLAPSES** |
| | | | **mag × size** | 0.5030 [.4939,.5118] | 0.5024 | 14,738 (72.8%) | **0.10** | **COLLAPSES** |
| **t10** arms loose | 0.6538 [.6445,.6633] | 0.5645 | magnitude | 0.6377 [.6245,.6504] | 0.5529 | 6,842 (33.8%) | 0.95 | SURVIVES |
| | | | size | 0.6264 [.6132,.6402] | 0.5485 | 6,756 (33.4%) | 0.87 | SURVIVES |
| | | | snr | 0.6460 [.6329,.6592] | 0.5457 | 6,842 (33.8%) | 1.12 | SURVIVES |
| | | | redshift | 0.6543 [.6419,.6674] | 0.5636 | 6,842 (33.8%) | 1.02 | SURVIVES |
| | | | psf | 0.6425 [.6293,.6549] | 0.5668 | 6,842 (33.8%) | 0.85 | SURVIVES |
| | | | **mag × size** | 0.6228 [.6097,.6359] | 0.5426 | 6,756 (33.4%) | **0.90** | **SURVIVES** |
| **t09** bulge boxy | 0.5847 [.5687,.6008] | 0.5359 | magnitude | 0.5678 [.5455,.5899] | 0.5316 | 2,396 (14.4%) | 0.74 | SURVIVES |
| | | | size | 0.5559 [.5320,.5786] | 0.5289 | 2,386 (14.3%) | **0.55** | SURVIVES (narrow) |
| | | | snr | 0.5678 [.5455,.5910] | 0.5158 | 2,396 (14.4%) | 1.07 | SURVIVES |
| | | | redshift | 0.5633 [.5397,.5857] | 0.5161 | 2,396 (14.4%) | 0.97 | SURVIVES |
| | | | psf | 0.5666 [.5440,.5900] | 0.5171 | 2,396 (14.4%) | 1.01 | SURVIVES |
| | | | **mag × size** | 0.5727 [.5491,.5951] | 0.5243 | 2,386 (14.3%) | **0.99** | **SURVIVES** |

**30 of 36 SURVIVES. 6 COLLAPSES, all of them t10-medium. No PARTIAL. No UNRESOLVED.**

### Reading it

**The objection does not hold, and it fails in its own strongest form.** The joint
magnitude × size match is the version of the objection that actually names "nearby, bright,
well-resolved", and it is the one that matters. On five of six features the joint match retains
**0.90–1.00** of the unmatched margin. Holding brightness and apparent size jointly constant costs
the signal essentially nothing. That is a stronger result than surviving the singles.

**Retention near or above 1.0 is not an artefact and deserves stating.** Many cells retain more
than the unmatched margin. This is because the null is **re-measured on the same matched subset**,
so the comparison stays like-for-like: matching removes nuisance variance from the untrained
baseline as well as from the trained one, and where a nuisance was carrying the untrained
encoder more than the trained one, the margin widens. Had the null been held at its unmatched value
this column would have read as a spurious gain. Re-measuring `C_m` is what makes the read honest.

**The sample cost is real and is reported as cost, not as signal.** Matching discards between 10%
and 86% of the test set depending on feature and nuisance. t09-boxy falls to 14.3–14.4% — above the
10% UNRESOLVED floor, but the closest anything came, and its intervals are correspondingly wide
(±0.023 against ±0.004 on t01). The plan named t09's joint match in advance as the likeliest
UNRESOLVED; it did not materialise, but the margin was thin and it is recorded that way.

**t09-boxy's size match is the narrowest SURVIVES in the battery** — retained 0.553 against a bar
of 0.500, with the CI lower bound at 0.5320 against `C_m` = 0.5289, a separation of 0.0031. It sits
in a column of verdicts that look alike and it is not like them. On a feature where 89.8% of
positives carry ≤ 2 votes, this should be read as "not refuted by matching" rather than as
"survives matching".

**t10-medium collapses on all six, and this is the second independent time that feature has failed.**
Its unmatched margin was +0.0065 over its own binding null. Matched, it retains 0.10–0.93 of that,
and every one of the six matched AUCs has a CI containing its matched null — including the joint,
at retained 0.10. Critically, this lands on the **sample-adequate** branch, not UNRESOLVED: 14,864
test galaxies survived, 73.4% of the unmatched set. So it is a statement about the **signal**, not
about what matching cost. The freeze record already flagged this feature as seed-dependent
(+0.0065 at seed 0, +0.0006 at seed 1). Two independent lines of evidence — seed sensitivity and
matched collapse — now say the same thing: **t10-medium sits below what this apparatus resolves.**

**The graded axis is not monotone, and this must not be smoothed.** Across t10's three arms the
matched result reads tight ✓ / medium ✗ / loose ✓. If arm-winding were a single clean direction
with a difficulty gradient, the middle rung would not be the one that vanishes. Two readings are
available and O1 does not choose between them: either "medium" is the genuinely ambiguous category
where human labellers disagree most (its unmatched AUC of 0.5226 is consistent with that), or the
ordered question is not being represented as an ordered axis at all. Distinguishing them needs the
ordinal machinery, not more matching.

### What O1 does not establish

- It is stamped `smoke: True`, as the J/N-series batteries are. The spread runs 50–200 null draws
  against the ≥10,000 permutation floor, so these numbers **cannot be read back as a headline
  result** — by design, and the stamp enforces it. O1 is investigation-grade evidence answering a
  specific objection, which is what was asked of it.
- One checkpoint, one split, one matching seed. The CIs are bootstrap intervals on the matched AUC,
  not intervals over matching draws.
- Six features of a catalogue of 37.
- Nothing here licenses a rung. No rung verdicts are assigned.

---

## O3 — the overfit-one-batch gate

**Pre-registered pass condition, fixed before running:** *prediction loss falls below 0.05 within
2,000 steps on a batch of 32.* Deliberately conservative — 0.05 sits below **0.0721**, the deepest
prediction-loss minimum any real run in this project reached (I2/H5's D17 arm). The gate asks the
path to do on 32 images something no run managed on 810,491.

One fixed batch of 32 (production `batch_size`, so the recipe is unchanged), through the
production path: real masker, real ViT, real EMA target, real objective, D17 schedule with D21's
λ=0. No dataloader — the batch is drawn once and reused. Recipe hash `61330a0012234374…`, M's.

### Verdict: PASS

```
O3 LOSS FLOOR : 0.046759 at step 1,913   (started 1.2856)
```

1,180 s. The gate now writes `artifacts/out/o3_overfit_gate.json` keyed on recipe hash and
`code_sha`, and `j1_preflight` (check 10, so `m1_preflight` and therefore M2/O2 too) refuses to
start a long job without a passing record. Verified by running it before the record existed:
exit 1, with the reason named. A check that can be skipped silently is not a check.

### 1. The loss floor, against the existing record

This number had never been measured in this project. Every previous statement about whether a
prediction loss was "low" was made without a reference point.

| run | prediction-loss minimum | × the overfit floor |
|---|---|---|
| **O3 overfit floor (32 images)** | **0.0468** | 1.0 |
| I2/H5 D17 arm — deepest real minimum | 0.0721 | 1.5× |
| M at its 4-epoch plateau (step 101,300) | 0.1155 | 2.5× |
| M at 2 epochs (step 50,654) | 0.1357 | 2.9× |
| M at 0.5 epochs (step 12,700) | 0.2984 | 6.4× |
| J under SIGReg, minimum ~step 2,000 | 0.3244 | 6.9× |
| J under SIGReg, endpoint step 50,000 | 0.4138 | 8.8× |

**The gap is smaller than expected, and that is the finding.** M's 4-epoch plateau sits at 2.5× the
floor achievable when the model is allowed to memorise 32 images outright. On 810,491 galaxies that
is not a large distance. The reading is that M's loss is not sitting far above what this
architecture can reach — the plateau at 0.1155 is a *real* plateau of the objective, not a symptom
of a path that cannot descend. J's SIGReg runs at 6.9–8.8× are, by contrast, a long way from the
floor, which is consistent with the regulariser dominating rather than the predictor failing.

This does not say M is near-optimal. Memorising 32 images and generalising over 810,491 are
different problems, and the achievable floor on the full corpus is unmeasured and certainly higher.
What it does close off is the hypothesis that the training path was structurally unable to descend.

### 2. The EMA/collapse signature at genuine memorisation — the substantive result

The I-JEPA loss is between the online encoder and its own EMA, so it can fall because the model
learned **or** because the two collapsed together. Until now there was no reference for which
looks like which. There is now.

| | std | effective rank | mean cosine |
|---|---|---|---|
| at initialisation | 0.109 | 19.8 | **+0.988** |
| step ~400 (loss 0.159) | 0.317 | **9.37** | +0.950 |
| step ~1,000 (loss 0.064) | 0.976 | 13.6 | +0.721 |
| **at the floor (loss 0.047)** | **1.556** | **19.72** | **+0.413** |

Online-vs-EMA cosine at the floor: **+0.8309** — the two towers stay aligned, as they must for the
loss to be low, while the *embeddings across galaxies* spread apart.

Three things follow, and one of them corrects a reading made earlier in this brief.

**(a) Mean cosine at initialisation is +0.988.** An untrained encoder already maps every galaxy to
nearly the same direction. This reframes the SIGReg diagnosis directly: **J's +0.984 under SIGReg
is not a collapse the regulariser caused — it is approximately the untrained value, i.e. the
representation barely moved off its initialisation.** That is a different and more specific failure
than "SIGReg collapsed the representation", and it should be reported as the former.

**(b) Genuine learning ends at high rank and low cosine, not low rank.** Effective rank dips to
9.37 around step 400 and then recovers monotonically to 19.72, while cosine falls the whole way.
The mid-run dip is a transient of early fitting and **not** the signature — reading it as one
(as "32 images need fewer directions") would have been wrong, and the endpoint reverses it.
The signature of real memorisation is std ×14 and cosine +0.988 → +0.413 at *undiminished* rank.

**(c) The collapse halt thresholds are reading the right quantity.** M1's soft effective-rank floor
of 2.5 is nowhere near either trajectory — genuine learning here never went below 9.37. The floor
is not at risk of firing on a healthy run, which is worth knowing before a 46-hour job.

### 3. Masking integrity — what was checked

Checked before the first optimiser step, over five mask seeds, and recorded in the JSON rather
than asserted in prose:

- **Context and target blocks are disjoint.** Maximum overlap 0 tokens, total overlap 0 tokens,
  across all five seeds. The predictor is never shown the tokens it is asked to predict.
- **Block sizes are as intended.** Context 72–91 tokens, target 174–202 tokens, of a 256-token
  grid. Neither degenerate-empty nor near-total.
- **The bbox bias is live but mild.** Target-block galaxy-weight ratio against the grid:
  1.1222, 1.1438, 1.1431, 1.1625, 1.1576 — every seed above 1.0, so targets do land preferentially
  on the galaxy as designed, by ~12–16% and not by a factor that would make prediction trivial.

The floor arrived at step 1,913 of a 2,000 budget — slowly, not implausibly fast. Had it arrived in
a few dozen steps, masking is where the fault would have been, and this is the check that would
have found it.

### What O3 does not establish

- The record was taken with a dirty working tree (`code_dirty: true`, `code_sha b288e6f`), and the
  repo has moved since. `assert_gate_passed` reports code drift **loudly** rather than raising on
  it, by design: the recipe hash determines the training path's shape, the suite covers code
  changes, and a gate made mandatory per commit gets disabled rather than run. Naming the drift is
  not the same as skipping the check — but it does mean this record should be re-taken before a
  headline run.
- One batch, one seed, one recipe. The floor is a property of (architecture, batch, schedule);
  it is not a bound on anything measured over the full corpus.

---

## O2 — a second training seed on the plateau

*Built (`artifacts/o2_second_seed.py`), not launched — the machine is occupied by O1 and O3.
Reported when it lands, and its interval will be labelled for exactly what it is: training-draw
variance with the splits held fixed, not an interval on the headline AUC.*
