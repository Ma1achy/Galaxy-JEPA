# Brief K — findings

Three items, in order. No training run was launched. K2's reading is a **proposal**; nothing is
merged and no gate is re-frozen on its evidence.

Repo state at the start: branch `sigreg-ablation`, HEAD `8753b49`, tree clean.

---

## K1 — The sky-noise control is out of the existence bar

**Landed.** The diagnosis holds and it is a category error, not a bug.

### What was wrong

`nulls.five_null_samples` took the elementwise maximum over all five 3C controls. A null has to
be **chance-calibrated** — it must answer "what AUC does this machinery reach when the thing being
measured is absent?". Four of the five break something and therefore do:

| control | what it breaks | what it kills |
|---|---|---|
| 3C-1 shuffled vote fractions | the image–label correspondence | "the probe exploits label marginals" |
| 3C-2 random embeddings | the representation | "any high-D vector predicts this" |
| 3C-3 noise through the real encoder | the images | "the encoder imposes structure on anything" |
| 3C-4 untrained encoder | the **pretraining** | "the probe, not the pretraining, did the work" |

3C-5 breaks nothing. Real images, real frozen encoder, real probe, and a *different real label*.
Its AUC is not what chance looks like; it is **how much image-quality content the representation
holds**. Entering it in a maximum asks a morphology probe to beat a nuisance probe before the
morphology feature is allowed to exist. No value of it is evidence about whether morphology is a
direction in the representation.

**The proof that it is one measurement entered twice.** 3C-5 and the `snr` nuisance probe came out
**bit-identical** on all six features probed at J4 — 0.8373, 0.8381, 0.8416, 0.8416, 0.8416,
0.8355. Both read `snr_r` through a median split over the same eligible ids. Same code path, two
names, one as a bar and one as a diagnostic.

### State it plainly

**Every feature failed existence under the broken bar, featured-ness included.**

| feature | real AUC | 3C-5 | strongest chance-calibrated null | old bar | corrected bar |
|---|---|---|---|---|---|
| t01 featured-or-disk | 0.8365 | 0.8373 | 0.7908 (untrained) | **fails** | clears by +0.0457 |
| t02 edge-on yes | 0.7320 | 0.8381 | 0.6368 (untrained) | **fails** | clears by +0.0952 |
| t10 arms tight | 0.5740 | 0.8416 | 0.5474 (untrained) | **fails** | clears by +0.0266 |
| t10 arms medium | 0.5161 | 0.8416 | 0.5160 (untrained) | **fails** | +0.0001 — a tie |
| t10 arms loose | 0.6098 | 0.8416 | 0.5645 (untrained) | **fails** | clears by +0.0453 |
| t09 bulge boxy | 0.5534 | 0.8355 | 0.5359 (untrained) | **fails** | clears by +0.0175 |

The designed ladder would have returned an all-R3/R4 catalogue on an encoder whose headline
feature probes at 0.9278 consensus — a catalogue that reads like a scientific null and is nothing
of the kind. **That near-miss is why this is documented rather than quietly applied.** A gate no
real effect can pass is not conservative; it is broken, and its output is indistinguishable from
an honest negative result.

### What changed

A pre-registered gate moved *after* a measurement existed, which is exactly when a reader is
entitled to ask whether it moved because it was wrong or because it was inconvenient. The
reasoning above does not depend on which features pass, and the record says so in three places.

- **Code.** `nulls.five_null_samples` → **`existence_null_samples`**, taking the max over the four
  chance-calibrated controls. A function that combines four must not be named for five; the count
  belonged in the name only while the count was the claim. `controls.py`, `ladder.py` prose
  corrected. In `run.py` the artefact key is now `sky_noise_diagnostic`, so `ladder_summary.json`
  cannot read a diagnostic back as a bar — an artefact that misdescribes its own gate is the same
  defect class as a stamp that misdescribes its own code.
- **Spec.** `docs/galaxy-jepa-spec.tex` §3C (title, calibration sentence, and control 5's entry
  rewritten with the reasoning), open-questions register item 8, and the controls deliverable
  line; `docs/probing-harness-design.md` §3C; `docs/spec/gates.md`'s `negative_control` row.
  **The PDF is stale** — there is no LaTeX toolchain in this environment. The `.tex` is the
  corrected source and the PDF needs a rebuild.
- **Decision.** `DECISIONS.md` **D19**, with the table above and the reasoning written out.
- **Test.** `test_the_sky_noise_control_is_a_diagnostic_and_never_sets_the_bar` pins that moving
  3C-5 from 0.51 to 0.99 cannot move the null, the p-value or the verdict, and that it is still
  carried on `FeatureControls`.

`uv run pytest` green, `ruff check` / `ruff format --check` clean, `mypy src/galaxy_jepa` clean.

### What K1 does not fix

- **The nuisance problem is relocated, not removed.** It now belongs to 3D-ii's matched
  evaluation, which is the machinery built to ask whether a morphology axis is really a nuisance
  axis. See K3-i: on this evidence that machinery is load-bearing for Paper 1.
- **The register's degeneracy item (8) stands.** The untrained-encoder singleton still exceeds
  every resampled draw on every feature — untrained 0.5160–0.7908 against shuffled-max
  0.5136–0.5552 — so the combined null still has zero variance and the existence *p* can still
  only be 1/(n+1) or 1. D19 narrowed that item from three singletons to two; it did not close it.

---

## K2 — The trajectory probe

### The reading rule, written before the numbers

Recorded here in advance so the reading is not fitted to the curve, the way
`artifacts/i_decision_rule.md` was committed before Brief I's arms ran.

| shape of the morphology curve | reads as |
|---|---|
| monotonic decline from early (already falling by ~6,000) | implicates **SIGReg** — the penalty keeps pushing after isotropy is reached and can then only cost structure |
| flat through the middle, dropping late (after ~27,000) | implicates the **cosine decay** — this run is its first real exercise; at 3,000 the LR had never left 99.7% of peak |
| a rise then a fall with a peak in the middle | **neither** cleanly — say so, and report where the peak is |
| noisy, no shape resolvable at n=8 | **neither** — report the curve and the noise |

**A third explanation is possible and inventing an attribution would be worse than reporting an
ambiguous curve.** n = 8, one trajectory, one seed. This is not a controlled comparison: nothing
here holds the decay fixed while varying SIGReg, or vice versa. Separating them properly needs two
arms, which is a training run, which this brief does not launch.

The nuisance curve carries its own reading, independent of the morphology one: **if nuisance AUC
climbs while morphology falls**, the representation is not simply degrading — it is re-allocating
onto image properties, and that is a mechanism story rather than a curve.

Two checkpoints are free replication checks rather than new information: **step 3,000** should
reproduce I2's `sigreg_050` at 0.9470, and **step 50,000** should reproduce J4(A) at 0.9278. If
either misses, the protocol is not what it claims and the rest of the table is void.

### Protocol

Eight of the 34 checkpoints — 1,500 / 3,000 / 6,000 / 10,500 / 18,000 / 27,000 / 37,500 / 50,000 —
declared in `artifacts/k2_trajectory_probe.py` before the run, not chosen after seeing a curve.
1,500 is the earliest that exists and the nearest to where the prediction term bottoms (~2,000).

Each checkpoint is re-exported through `i3_loss_usability.frozen_from` so `load_frozen_encoder`
stays the only route to a probe (the freeze boundary runs through disk), then embedded **once**
over the union of J4(A)'s split — 40,000 train on H5's deterministic stride plus the uncapped
34,829 held-out. The morphology numbers come off that matrix through `_extremes` + `probe_auc_ci`
with `configs/pretrain.yaml`'s probe block, so they are H5-protocol and land directly beside
0.9470 and 0.9278. Asserted before the first encode: consensus train **25,305** and held-out
**34,829**, both matching J4(A) exactly — if they had not, the comparison would be void and the
driver refuses rather than reporting a different measurement.

The five nuisances are probed off the **same** embeddings, so a nuisance AUC and the morphology
AUC at a checkpoint are read from one representation rather than two. Their validity masks and
median splits are drawn **once** from the metadata, before the loop, so the labels are identical
at every checkpoint — re-deriving them per encoder would let the median move and quietly make the
curve incomparable. They are built over the whole split rather than a feature's eligible subset
(as `build_feature_controls` does), because a nuisance is a property of the representation, not of
a morphology question, and holding n fixed is what makes a trajectory readable. **Stated limit:**
these are therefore not comparable to J4's per-feature `nuisance_aucs`, which carry that feature's
eligibility. 374 of the 40,000 training galaxies are dropped from the `size` nuisance only, on
`petrorad_suspect`.

Loss terms are read from `runs/full/traces.json` as a trailing 100-step mean (a single step is
noise at batch 32), with the point value alongside. Collapse readings are the monitor's, at the
last reading **at or before** the checkpoint — 50,000's is the 49,900 reading, and the table says
so. Reading a step-49,900 value against a step-50,000 value without saying so is the like-for-like
error Brief I already caught once with i2's `erank_final`.

Results are banked after every checkpoint.

### The trajectory

74,829 stamps per checkpoint, ~620 s each, 8 checkpoints, 1.4 h. Nothing halted; `size` drops 374
of 40,000 training galaxies on `petrorad_suspect`, every other nuisance uses all 40,000.

**Both replication checks pass exactly.**

| check | K2 | reference |
|---|---|---|
| step 3,000 vs I2 `sigreg_050` | 0.9470 [0.9435, 0.9506] | 0.94697 [0.94345, 0.95057] |
| step 50,000 vs J4(A) | 0.9278 [0.9234, 0.9320] | 0.92782 [0.92345, 0.93204] |

Point and both interval ends agree to four decimals in both cases. A checkpoint re-exported through
`frozen_from` and probed by this driver reproduces `h5_probe_lean.py` exactly, so the rest of the
table sits legitimately beside 0.9470 and 0.9278.

| step | consensus AUC [95% CI] | all held-out | ambiguous | pred loss | SIGReg | erank | std | cos | lr (% peak) |
|---|---|---|---|---|---|---|---|---|---|
| 1,500 | **0.9477** [0.9440, 0.9515] | 0.8581 | 0.6770 | 0.2901 | 1.9880 | 24.28 | 0.9565 | +0.2257 | 100.0% |
| 3,000 | 0.9470 [0.9435, 0.9506] | 0.8558 | 0.6734 | 0.3299 | 1.5179 | 34.56 | 0.9826 | +0.1358 | 99.7% |
| 6,000 | 0.9448 [0.9410, 0.9485] | 0.8535 | 0.6711 | 0.3871 | 1.3546 | 44.39 | 0.9624 | +0.0849 | 97.7% |
| 10,500 | 0.9412 [0.9372, 0.9451] | 0.8493 | 0.6677 | 0.4107 | 1.2738 | 49.66 | 0.9734 | +0.0689 | 91.4% |
| 18,000 | 0.9376 [0.9334, 0.9417] | 0.8454 | 0.6669 | 0.4280 | 1.1913 | 53.43 | 0.9613 | +0.0526 | 73.6% |
| 27,000 | 0.9295 [0.9249, 0.9338] | 0.8364 | 0.6592 | 0.4276 | 1.1411 | 55.39 | 0.9290 | +0.0353 | 45.6% |
| 37,500 | 0.9284 [0.9240, 0.9327] | 0.8343 | 0.6580 | 0.4165 | 1.1020 | 57.38 | 0.8914 | +0.0251 | 15.4% |
| 50,000 | 0.9278 [0.9234, 0.9320] | 0.8336 | 0.6582 | 0.4151 | 1.0798 | 57.61\* | 0.8851\* | +0.0250\* | 0.1% |

\* the 49,900 collapse reading — the monitor's last. Named rather than silently aligned.

**Monotonic, and it starts at the first checkpoint.** Every step is down. All three framings —
consensus, all-held-out, ambiguous middle — fall together, which is what makes it a property of
the representation rather than of one eligibility convention.

### Where the damage happens, and where it does not

| window | ΔAUC | steps | lr across the window |
|---|---|---|---|
| 1,500 → 27,000 | **−0.0182** | 25,500 | 100% → 46% of peak |
| 27,000 → 50,000 | **−0.0017** | 23,000 | 46% → 0.1% of peak |

**91% of the total loss happens in the first half, while the LR is still above half of peak. The
second half — the entire window in which the cosine actually collapses — costs −0.0017.**

That is the reverse of the decay's signature. The pre-registered rule said *flat, then dropping
late → implicates the cosine decay*. What the run shows is dropping early, flat late. If the decay
were doing the damage, the damage would concentrate where the LR is falling fastest; instead it is
almost complete before the LR has dropped below half, and it stops as the LR collapses. On this
trajectory the decay looks like what **arrests** the decline, not what causes it.

The SIGReg side moves the other way and saturates on the same schedule: the penalty falls
monotonically 1.988 → 1.080 against a measured isotropic floor of ≈1.065, effective rank climbs
24.3 → 57.6, mean cosine +0.226 → +0.025 — with the large moves early (rank is already 49.7 by
10,500) and near-saturation late. The AUC decline flattens as the constraint stops having anywhere
left to push.

**The AUC peak is not where the prediction term bottoms.** The prediction loss minimum is 0.2804
at step **1,954** (100-step mean), confirming J's "~2,000". But AUC at 1,500 already exceeds 3,000,
so the two are not co-located — and 1,500 is the **earliest checkpoint that exists**. The true
peak may be earlier and this design cannot bound it from below. Stated as a limit, not a result.

### The nuisances fall too — nothing climbs

| step | morphology | redshift | magnitude | size | SNR | PSF |
|---|---|---|---|---|---|---|
| 1,500 | 0.9477 | 0.8196 | 0.9191 | 0.9058 | 0.8818 | 0.5852 |
| 3,000 | 0.9470 | 0.8166 | 0.9247 | 0.9038 | 0.8898 | 0.5634 |
| 6,000 | 0.9448 | 0.8022 | 0.9210 | 0.8937 | 0.8779 | 0.5544 |
| 10,500 | 0.9412 | 0.7938 | 0.9192 | 0.8812 | 0.8731 | 0.5570 |
| 18,000 | 0.9376 | 0.7960 | 0.9005 | 0.8678 | 0.8484 | 0.5659 |
| 27,000 | 0.9295 | 0.7918 | 0.8743 | 0.8500 | 0.8287 | 0.5708 |
| 37,500 | 0.9284 | 0.7936 | 0.8735 | 0.8503 | 0.8361 | 0.5807 |
| 50,000 | 0.9278 | 0.7918 | 0.8733 | 0.8501 | 0.8373 | 0.5813 |

**The re-allocation story is not what happened.** Nothing climbs while morphology falls. In
excess-over-chance terms (AUC − 0.5), across 1,500 → 50,000:

| axis | retained |
|---|---|
| **morphology** | **−4.4%** |
| PSF | −4.6% (from a base of 0.085 — near-noise throughout) |
| redshift | −8.7% |
| magnitude | −10.9% |
| SNR | −11.7% |
| size | −13.7% |

Morphology is the **best-preserved informative axis on the trajectory.** Every nuisance that
carries real signal loses two to three times as much of it.

A mechanism candidate, offered as a reading and not a finding: isotropisation strips the dominant
low-dimensional, high-variance directions first, and observing conditions — apparent brightness,
apparent size, depth — are exactly the kind of global structure that occupies the top principal
directions of an image representation. On that reading SIGReg is preferentially removing nuisance
axes and morphology is collateral damage. One trajectory, no control arm, no test of the claim.

**Sanity check on the caveat I wrote in advance.** K2's nuisances are built over the whole split
rather than a feature's eligible subset, so I recorded that they would not be comparable to J4's
per-feature panel. For t01 at `vote_count_min: 1` the eligibility filter turns out to keep
everything, and K2's step-50,000 row is *identical* to J4's t01 panel to four decimals. The
caveat is correct in general and moot for this feature; both are stated rather than the
convenient one.

### The reading

Against the rule written before the numbers: **the shape matches the SIGReg arm and is
inconsistent with the decay arm.** Monotonic decline from the earliest checkpoint, 91% of it
complete before the LR falls below half of peak.

**What this does not establish.** It is not a controlled separation, and the limits are not small:

- **Everything here is monotone in step.** AUC, the SIGReg penalty, effective rank, mean cosine,
  the LR and the number of samples seen all move one way across the whole run, so any rank
  correlation among them is ±1 by construction and carries no information. n = 8, one trajectory,
  one seed. Nothing here holds the decay fixed while varying SIGReg or the reverse.
- **A third cause is not excluded: data seen.** 1,500 steps is 0.058 epochs over 826,968 stamps;
  50,000 steps is 1.94 epochs. Plain over-training is confounded with both suspects, and this
  design cannot see past it.
- **What the curve does do is falsify the decay's temporal signature**, which is a narrower claim
  than "SIGReg did it" and is the strongest thing the evidence supports. It weakens one suspect;
  it does not convict the other.
- **The peak is unbounded below.** 1,500 is the earliest checkpoint on disk.

**This does not license checkpoint selection.** 1C stands: the checkpoint rule is label-blind, and
a curve that peaks early is precisely the temptation 1C exists to refuse. Brief I already measured
that question and found no usable label-free signal.

### Candidates for an experiment that would actually separate them — recorded, not proposed for launch

Each is a training run, which this brief does not launch, and each is listed so the choice is
argued rather than defaulted:

1. **A λ = 0 arm at full length.** If an unconstrained arm degrades the same way, SIGReg is
   exonerated and the cause is the decay or the epochs. This is the cleanest single experiment and
   it is D12's business as well as D18's. ~9.4 h.
2. **λ = 0.05 with a constant LR at 50,000 steps.** Holds SIGReg fixed and removes the decay.
   Answers the narrower question directly, but changes the recipe D17 and D18 were adopted with.
   ~9.4 h.
3. **Neither, yet.** The nuisance panel (K3-i) is a P0 blocker on the headline run's budget and
   arguably outranks this. Knowing *why* the representation degrades is worth less right now than
   knowing whether matched evaluation is affordable at all.

**Propose, do not merge.** No config changed, no gate re-frozen, nothing in `configs/` touched by
K2.

---

## K3 — Recorded, not acted on

Three things logged, none acted on. Each names the fix; none applies it.

### K3-i — The nuisance panel is the serious finding, and it moves matched evaluation into Paper 1

The J4 panel, measured on the frozen 50,000-step embedding:

| nuisance | AUC |
|---|---|
| magnitude (`modelMag_r`) | **0.8733** |
| size (`petroRad_r`) | **0.8501** |
| SNR (`snr_r`) | **0.8373** |
| redshift (`specz`) | **0.7918** |
| PSF (`psfWidth_r`) | 0.5813 |

Against featured-ness at 0.8365 and **every other probed morphology feature below 0.74**. The
representation encodes observing conditions at least as strongly as it encodes morphology.

The spec (3D-ii) specifies matched evaluation as *targeted* — "fires only for flagged features —
bounded cost". That bound is what let it be promoted from "Paper-2/if-feasible" to "Paper-1,
targeted" in the first place. **On this evidence it fires for every feature on three or four
nuisances each.** So it is not a targeted contingency but a load-bearing component of Paper 1, and
its cost is a headline-run cost rather than a tail. `matching.py` moves from formality to critical
path, and the headline run cannot be budgeted without it.

Physically this is unsurprising — morphology genuinely correlates with depth, size and brightness,
which is 3D's own premise. What was not anticipated is the magnitude, and what was mis-sized is
the machinery.

Recorded in `TODO.md` (P0), `docs/galaxy-jepa-spec.tex` §3D (as an `OPEN` item against the
"targeted, bounded cost" claim) and `docs/probing-harness-design.md` §3D-ii.

### K3-ii — `code_dirty` can be wrong about the code it describes

`RunStamp.create` shells out to `git status --porcelain` at `_make_stamp` time. `run_harness`
reaches `_make_stamp` *after* `_prepare` — 72 minutes into the J3 run. An `artifacts/` file edited
at 21:30, three minutes after the run started, was recorded as dirtiness of the code the run
executed, stamped at 22:39.

Two distinct failings:

- **Timing.** Dirtiness is captured 72 minutes after the process starts, so anything edited during
  the bake lands in the stamp. The fix is to capture `(sha, dirty)` **once at process start** and
  pass it down, rather than moving the `_make_stamp` call — the call's position is load-bearing
  for a different reason (the checkpointer records `config_hash` so a resume cannot continue a
  different experiment).
- **Scope.** `--porcelain` covers the whole tree, including `artifacts/`, which is excluded from
  lint/CI and never imported by the package. A dirty `artifacts/` file cannot change a number the
  run produces. The status check should be narrowed to the package + `configs/`.

A stamp that can be wrong about its own code is the same class of defect as the three
reproducibility holes already closed. Recorded in `TODO.md` (P1). Not fixed.

### K3-iii — The fourth full-table build: there isn't one, and that is checked rather than assumed

`grep` over `rows_by_id`, `DirectorySource(...).rows` and `_probe_rows`. Every probe-path consumer
now goes through the sidecar:

| call site | path |
|---|---|
| `run_harness`'s post-train probe (`harness.py:636`) | `_probe_rows` ✓ (the Brief I fix, landed in J) |
| `evaluate_probe` (`harness.py:723`) | `_probe_rows` ✓ |
| `run_probing` / `probe_frozen_checkpoint` (`harness.py:838`) | `_probe_rows` ✓ |

Two full-table builds remain, both bounded and both deliberate:

1. **`_prepare` (`harness.py:463-464`)** builds `DirectorySource` over *both* corpora. It is the
   **producer** of the scalar and probe-column sidecars, so it cannot read them — and it is
   explicitly `del`'d before training starts, with the comment saying why. Bake-time cost, not a
   training-phase resident.
2. **`_probe_rows`' own fallback (`harness.py:703-704`)** fires only when no sidecar exists, and
   logs a warning naming the size difference when it does.

`artifacts/f1_loader_bench.py:223` is a deliberate benchmark of the old path and says so.

**No fourth site.** Recorded in `TODO.md` alongside the `_prepare` cost item.

### K3-iv — found while building K2: `EmbeddingMatrix.index` is quadratic at every call site

Not asked for, found by measurement, and worth more than it looks.

`EmbeddingMatrix.index` is a plain `@property` that rebuilds a full `{object_id: row}` dict on
every read — 5.6 ms over 74,829 entries. `probing.extract.feature_ids` reads it **inside a
comprehension's condition**:

```python
return labels.eligible(feature, [int(o) for o in ids if int(o) in matrix.index])
```

so the dict is rebuilt once per element. Measured: filtering 40,000 ids costs **8.6 ms** with the
property hoisted and **225 s** without — a factor of **26,000**.

`build_feature_controls` performs eight such filters per feature (six `feature_embeddings` — real,
noise and untrained, train and test — plus two explicit `feature_ids`), four over the 40,000-id
train list and four over the 34,829-id test list: **1,680 s per feature**. Over J4's six features
that is **≈10,080 s of its 21,598 s total — about 2.8 hours of a 6.0-hour run**, which I had
attributed at the time to memory-compressor thrashing. Thrashing was real and additional; this was
underneath it.

The fix is one line at each of two sites (bind `index` once in `feature_ids`, or make it a cached
attribute on the frozen dataclass). It is production code on the critical path of every ladder
run, so it is **recorded, not applied** — a K3 item, not a K2 edit. K2's own driver hoists it
locally and says so in a comment.

