# Spec — the data stack (pretrain–probe parity)

*Status: design proposal for sign-off. Expands `docs/architecture.md` → "The data stack
— pretrain-probe parity". Built in `src/galaxy_jepa/data/` (next phase). British
English.*

Three concerns (Prism's `DataSource` / `Transform` / `Sink`, scaled down):

- **`DataSource`** — SDSS cutouts + the CasJobs/SkyServer metadata join.
- **Transform pipeline** — a composable, ordered list (decode → stretch → crop →
  augment).
- **`StatefulTransform`** — a transform that **must be fitted** before use (normalisation
  is the one that matters), with a frozen-state it carries everywhere.

**Masking is not a data transform** — it is part of the JEPA objective and lives in
`objectives/`. Keep that boundary clean.

**Setup.** The networked dependencies (astropy + astroquery + pillow) are the `data`
optional extra: `uv sync --extra data`. They are imported **lazily** — `import
galaxy_jepa.data` and the offline pipeline work without them; only `FitsFrameSource` /
the live pull require them.

---

## 1. The parity rule — format + stretch + normalisation, byte-identical everywhere

The correctness trap: the preprocessing used in pretraining must be applied
**identically** when computing probe embeddings, or the probe sees a different
distribution than the encoder was trained on — and the D6 corpus decoupling (pretrain on
large unlabelled SDSS, probe on GZ2) silently breaks, because it *assumes the
representation transfers*.

So the parity rule extends upstream beyond the normalisation statistic to the **whole
front of the pipeline**:

> **Format + stretch + normalisation are byte-identical across the pretraining corpus,
> the probing corpus, and every baseline.**

Enforced by a **same-pipeline requirement**: both corpora are pulled the *same way*; you
do not mix a FITS+asinh pipeline for one with a pre-stretched-cutout pipeline for the
other.

### 1.1 The fp16 pre-bake cache — parity locked, run once, on disk

The training dataloader must **never** read FITS and run asinh+normalise per batch (the
per-step CPU cost starves the device). So the frozen `Pipeline` runs **once** over the
pulled corpus and the result is written as **fp16** to a memory-mapped array
(`data/cache.py`), which the dataloader reads with zero per-batch preprocessing (fp16
halves the working set vs fp32 — decisive on an 18 GB unified-memory Mac). The cache is
the parity rule made physical: one baked tensor set, shared by pretraining and probing.

Two contract points make it correct across the staged pilot → full run:

- **Hash-keyed, auto-invalidating.** The cache lives under `<base>/<pipeline_hash>/`, where
  `pipeline_hash = config_hash(pipeline)`. A different `Q` / flux-scale / normalisation
  statistic ⇒ a different directory ⇒ stale stats can never silently mix with fresh ones.
- **Normalisation fitted once and frozen to disk; incremental top-up.** The per-channel mean/std
  are fitted **once**, by `artifacts/e5_fit_normalisation.py`, and pinned into the config as a
  `NormalisationFreeze`; a run **loads** them and has no fitting path at all (§1.3, D16). Because
  every run shares that one frozen pipeline they share the `pipeline_hash`, so topping a corpus up
  **appends** new stamps and **reuses** every existing one — never a re-bake.

  > The earlier version of this paragraph said the statistic was fitted on "a seeded ~5–10k
  > subsample" because "fitting on the full ≫100k×256² set will not fit in RAM". The second half is
  > true and the first does not follow: the fit streams per-channel sums, so nothing is ever
  > stacked and the whole corpus costs one pass. That conflation is what produced the defect §1.3
  > describes — and the claim about never re-baking was itself false in practice, because a refit
  > moved the statistic, which moved the hash.

### 1.2 Valid pixels — the constant-region detector

A cutout that runs off an SDSS frame boundary is padded with a constant, and the pad value is
**exactly 0.0** while the sky sits at median 0.0013 with σ = 0.114 (measured over 250 pretrain
stamps). The pad is therefore **0.0σ from sky**: no value threshold can separate them, and any
that appeared to work would be cutting sky. What *does* separate them is exact constancy — the padding is bit-identical over a region,
and sky is not.

> **The rule** (`data/validity.py`, one site, consumed by normalisation, the exposure survey and
> — if it is ever needed — the sampler): a pixel is a constant-region candidate iff it is
> **bit-identical to at least one 4-neighbour in every channel**. Candidates are grouped into
> connected components by row-run labelling + union-find (**no scipy** — it is an optional
> dependency here), and components smaller than **`MIN_REGION_PX` = 256** are dropped.

Two details are load-bearing and were each arrived at by correcting a wrong first attempt:

- **"At least one neighbour", not "all four."** An erode-style "equals all four neighbours" rule
  has no interior at all on a 2-px-wide dead column, and loses every rectangle corner. It silently
  under-detects exactly the shapes the detector exists for.
- **The 256-px floor is measured, not chosen.** SDSS stamps are heavily quantised (~569 distinct
  values per 65,536 px, uniform step 2⁻¹⁹), so bit-identical neighbours arise **by chance** about
  17 times a stamp. Over 800 stamps the detector found 14,486 regions: 89% are exactly 2 px, the
  chance tail ends at 27 px, **nothing at all falls between 33 and 255 px**, and real regions
  resume at 256. Any floor in [33, 256] gives the same answer; 256 is chosen because it is one
  16×16 ViT token. (A synthetic pure-noise null cannot catch this — generating the null yourself
  reproduces your own assumption about the pixel distribution rather than the corpus's.)

Edge-touching and interior components are carried as **two separate bitplanes**
(`invalid_planes`), because they would not be equally safe to avoid when masking: padding is
off-galaxy by construction, while a saturated core sits at the centre where bulge structure
lives. **On these corpora the distinction is moot** — surveyed over 3,000 stamps, every
qualifying region in both the probe and the pretraining corpus is edge padding, and the interior
count is **exactly zero**. The earlier "7.2% interior" reading was quantisation noise under the
bare bit-identity rule, not dead pixels; the size floor removed it. The classification is kept as
the guard that says so, not because this data needs it.

**Exposure, measured against the real sampler.** `MultiBlockMasker.sample` and `Jepa.weight_maps`
were run unmodified over pretrain stamps at β ∈ {0, 0.5, 1.0}, projecting the pixel validity mask
onto the 16×16 token grid. The pre-registered threshold quantity — **% of target blocks ≥50%
invalid** — came out at **0.22% in the worst case (β = 0**, the pure-I-JEPA control, which has no
bbox bias pulling masks onto the galaxy and so is the most exposed). That is inside the
pre-registered "< 1% ⇒ negligible" branch, so **the sampler is unchanged**: `MaskConfig` gains no
avoidance flag, no validity sidecar is baked, and **β = 0 keeps its meaning as the published
control**. The padding matters for the *statistic*, not for the masking.

### 1.3 The normalisation statistic — fitted once, over valid pixels, frozen

**The defect this closes.** `harness._build_pipeline` used to call `fit_normalise` on every run
and persist nothing. It was seeded, so it looked reproducible — but the subsample is
`rng.choice(len(source), n_sample)`, and `len(source)` went from 10,000 to 826,968. The same seed
then draws an entirely different sample. Nothing on disk recorded the constants: the stamped
`config.json` carried `norm_sample: 8000` — the *instruction to fit* — not the numbers fitting
produced. Two runs could differ in their input transform with identical `config_hash`es and no
way to tell afterwards. The statistic is the parity lock; a parity lock that re-derives itself per
run is not one.

So the statistic is now an **artefact**. `NormalisationFreeze` is a `FrozenChoice`: per-channel
mean/std, the corpus and sample it came from, the stretch `Q` it sits downstream of, the
valid-pixel flag, the detector rule, a content hash over all of those, plus `frozen_at` /
`frozen_by` / `rationale`. Being a `RunConfig` it is hashed into `config_hash` and written to
every artefact's `config.json`, so a result carries where its normalisation came from.

Three refusals make it hold rather than merely document:

- **A missing record stops the run.** `_build_pipeline` takes the freeze and has no fitting path
  at all; `harness.py` no longer imports `fit_normalise`. There is no escape hatch here, unlike
  `effect_floor`: a run that fitted its own statistic and stamped the forfeit would still have
  broken parity with every other run, so the forfeit would be unpayable.
- **A `Q` mismatch stops the run.** These are *post-stretch* statistics; under another `Q` they
  are not stale, they are meaningless.
- **A hand-edited record stops the run.** `assert_intact()` recomputes the content hash over the
  determining fields, so editing a mean without re-fitting makes the provenance a lie and the lie
  is caught at load. Provenance prose (`rationale`, `frozen_by`, …) is deliberately outside the
  hash: it explains the number, it does not determine it.

**Fitted on pretrain, applied to both corpora.** A separate probe fit would absorb the D6
magnitude/SNR shift uncontrolled *and* hand the frozen encoder a different input transform at
probe time than it saw in training — the exact failure the parity rule exists to prevent.

**Valid pixels only, and what that bought.** Over the whole pretraining corpus **98.317% of
pixels count as valid**, and **114,370 stamps (13.8%) carry padding** — among those, a median
11.8% of the frame, p90 21.9%, max 79.8%. Because the pad sits at exactly 0 it drags every mean
down and deflates every σ, uniformly across channels:

| ch | naive mean | valid mean | Δ | naive std | valid std | Δ |
|---|---|---|---|---|---|---|
| 0 | 0.009296 | 0.009455 | **+1.71%** | 0.078388 | 0.079046 | **+0.84%** |
| 1 | 0.016034 | 0.016308 | **+1.71%** | 0.107311 | 0.108204 | **+0.83%** |
| 2 | 0.022794 | 0.023184 | **+1.71%** | 0.139794 | 0.140953 | **+0.83%** |

**The whole corpus, not a subsample — and why the subsample had to go.** The shipped default was
`n_sample=8000`. Two disjoint halves of it disagreed by **4.54%**, and the measured curve is a
clean 1/√n that nothing reachable clears:

| n_sample | 2,000 | 8,000 | 32,000 | 100,000 | 400,000 | all |
|---|---|---|---|---|---|---|
| median half-split Δ | 6.09% | 3.37% | 1.48% | 0.96% | 0.43% | **0.32%** |

So every stamp is counted, and `n_sample` is **removed from the config** — no run can ask for a
draw at all. `seed` survives in the record only to state that it was inert: no draw was made.

**The trim — a degree of freedom, pinned like one.** The census statistic still failed the
stability gate: **11.0% of 200 disjoint halves** disagreed past the 1% tolerance, because the
variance is carried by a minority of stamps. The heaviest single stamp holds **0.090% of the whole
corpus's ch0 sum-of-squares** — 745× its uniform share — and the top 1% hold 13.4 / 17.9 / 10.5%
by channel.

> **The rule.** Rank every stamp by **one** scalar — total valid-pixel sum-of-squares across all
> three channels, post-asinh — and exclude from the **fit** any stamp above the 99.9th percentile
> (threshold **36779.561450**). That is **827 stamps, 0.100%**; the excluded set is pinned by
> `sha256 = bf2a8d0b…` over the sorted object IDs.

One scalar, not a per-channel cut: ranking each channel independently would exclude different
stamps from different channel statistics and the channels would stop being comparable. **The trim
applies to the fit only** — every stamp stays in the corpus, in the cache and in training.

It is not gate-passing. A σ inflated by a handful of outliers compresses the typical galaxy's
post-normalisation range; the trimmed σ reflects the typical stamp and bright stamps correctly
extend past ±1. Normalisation constants are a preprocessing transform, not a population parameter
needing an unbiased estimate. With the trim, the gate passes cleanly: over 200 disjoint halves of
413,070, **median 0.314%, p95 0.616%, worst 0.861%, 0.0% breaching 1%**.

**What the trimmed stamps turned out to be — a data-quality finding, reported not acted on.** Not
bright galaxies and not saturated stars: they are **low-SNR stamps concentrated in a few bad
imaging runs**. Median `snr_r` 50.4 against the corpus's 93.9, with `modelMagErr_r` nearly double
— and that column is SDSS's own photometric error, so it corroborates independently of our pixels.
**67.4% of the 827 come from a single SDSS run, run 1000**, which is trimmed at 11.07% against the
corpus-wide 0.100% — a **111× enrichment**; runs 2194 (79×) and 5181 (70×) follow, and the three
together account for 86.7%. 28.3% carry `petroRadErr_r = −1000`, a failed radius fit, against
6.80% corpus-wide. **No GZ2 probe galaxy comes from any of those runs**, so the probing corpus is
untouched — GZ2's own selection had already excluded them.

---

## 2. Format + stretch — **decided: FITS + asinh for both corpora**

Both the large unlabelled SDSS pretraining set and the GZ2 probing set are pulled as
**raw FITS** and put through an **identical asinh stretch**.

**Why FITS, not pre-stretched cutouts.** 8-bit display-stretched cutouts (what
`galaxy-datasets` serves) irreversibly compress the **low-surface-brightness range** —
exactly where the confused features live (winding tightness, arm count, tidal
structure). Probing on display-stretched data would confound a genuine **Rung-4** result
(*absent from the pixels*) with **display-stretch loss** (*destroyed by the 8-bit
quantisation*) — fatal to the measurement this project exists to make.

**Why not resampled FITS cutouts either (no-rebin is empirically proven).** A fast
cutout service (`hips2fits`, CDS `P/SDSS9`) returns calibrated FITS (flux ratio to native
**1.005**, so *not* display-scaled) ~40× faster than downloading native frames — tempting
as the corpus path. It was **tested and rejected** (`artifacts/fidelity_test.py`, 8 faint
GZ2 spirals, native stamp vs hips2fits cutout): the bright signal survives (absolute
feature flux ratio **1.00**) — *which is the trap* — but the resampling **attenuates
high-frequency power to 0.115** of native (≈89% lost) and **correlates the pixel noise
from lag-1 autocorrelation 0.017 (independent) to 0.443** while collapsing the sky MAD to
43%. A 0.44-correlated noise field is **learnable as fake structure** by the encoder, and
the high-freq loss sits exactly in the faint regime Rung-4 lives in — so a resampled
corpus makes "absent from the pixels" *uninterpretable* (can't separate genuine
non-resolution from resampling smoothing). This is the **same fidelity-over-convenience
call as FITS-vs-cutouts**: the `native 0.396″/px, no rebin` rule is therefore a
**measured Rung-4 protection, not a preference**.

The contract, concretely:

1. **asinh parameters are config.** The softening scale `Q` + per-channel flux scale (or
   the chosen asinh parameterisation) are chosen **once on the pretraining corpus**,
   **frozen**, and **parity-locked** across all corpora and baselines. Being config, they
   **enter the config hash + run-stamp** (`docs/spec/config.md`) — a stamped,
   reproducible decision, never a notebook constant.
2. **normalisation is fitted *after* the stretch.** The normalisation statistic
   (per-channel mean/std) is a `StatefulTransform` **fitted once on the pretraining
   corpus, post-stretch**, then frozen and applied everywhere — pretraining, probing,
   every baseline.
3. **stretch-sanity check before any pretraining.** A cheap check confirms faint arms
   **survive** stretch+normalise while the **sky-noise floor stays controlled**. It is
   Tier-2 (`docs/spec/validation.md` `T2.stretch-sanity`) and **pairs with the collapse
   monitor**: if the encoder starts modelling noise, the stretch is too aggressive.

   The measurement is **two spatially-separated numbers per galaxy**, post
   stretch+normalise (`data/sanity.py::galaxy_zone_metrics`), because faint outer arms and
   sky noise occupy the *same* brightness range (the reason for FITS over cutouts) — a
   whole-image score cannot tell them apart:
   - **faint-retention** = median pixel value in the **annulus** `R_petro ≤ r <
     min(k·R_petro, stamp/2)` (real outskirts; same `R_petro`→px scaling as the masking
     box, `k = 2.5`);
   - **sky-noise floor** = MAD of the blank-sky **corner patches**.
   The honest signal quantity is the **gap** `faint-retention − sky-floor`: the two
   *diverging* is retained signal, the two *tracking together* is amplified noise (the
   suspect case). A missing/oversized `R_petro` excludes that galaxy from faint-retention
   with a loud log — never a fabricated annulus.

   **Choosing `Q`.** `Q` is a constrained trade-off (both numbers rise with `Q`), so it
   is set from a **sweep** (`data/q_sweep.py`) over a `Q` grid on a few-thousand-galaxy
   probe sample, with **normalisation re-fit per `Q`** (the fit is post-stretch and
   interacts with `Q`) and the flux scale held fixed. The sweep produces a curve
   (faint-retention, sky-floor, gap) + multi-`Q` contact sheets; a human sets the
   sky-noise ceiling and picks the `Q`. **asinh params stay unfrozen until then** (§4).

This is reflected in the scratchpad's preprocessing section (proposed edit).

---

## 3. The CasJobs join + declared metadata columns

The GZ2 → SDSS join (CasJobs/SkyServer) supplies the nuisance battery on the **probing**
corpus. Declared columns (existence checked at Tier-1 `T1.metadata-columns-real`):

| Column | Source | Used by |
|---|---|---|
| `z` (redshift) | `SpecObj.z`, joined on `specObjID = zoo2MainSpecz.specobjid` | nuisance probe |
| apparent magnitude (`modelMag_r`) | SDSS `PhotoObjAll` | nuisance probe |
| `petroRad_r` (Petrosian radius) | SDSS `PhotoObjAll` | nuisance probe **and** per-galaxy masking box |
| **`SNR_r`** (image-domain) | **derived: `1.0857 / modelMagErr_r`** | nuisance probe |
| PSF width (`psfWidth_r`) | SDSS **`Field`** table, joined on `fieldID` | nuisance probe |
| **`expAB_r` / `deVAB_r`** (axis ratio b/a) | SDSS `PhotoObjAll`, joined on `objID` | **inclination conditioning (D13) — NOT a nuisance regressor** |
| **`petrorad_suspect`** | **derived: `petroRad_r` > `PETRORAD_SUSPECT_ARCSEC` (25″)** | **excludes a row from the radius nuisance probe only** |

**The over-stamp flag — flag, never drop.** 2,143 of the 230,358 probe galaxies (0.93%) have a
`petroRad_r` wider than the 256 px (101″) stamp can hold, running to 258″ = 651 px. For those
rows the column describes a galaxy the encoder only ever saw a fragment of, so it cannot serve as
an honest size nuisance — but the *galaxy* is fine, and most of these are not failures at all:
25–100″ is 98.7% of the flagged set and behaves exactly like real large nearby disks (redshift
falling monotonically with radius, r ≈ 13.5–14.5, featured fraction ~0.66); only past 100″ (27
objects) does the deblending signature appear. They also skew hard to featured (0.66 against
0.32), so leaving them in would load the top half of the size median-split with disc galaxies and
make "size" read as morphology — a false nuisance-competitive trigger.

So they stay in the corpus, in the feature probes and in every *other* nuisance; the exclusion
applies to the Petrosian-radius control alone (`probing.extract.NUISANCE_FLAG_COLS`, part of the
nuisance schema so a caller with its own columns declares its own failures). A corpus without the
flag column **refuses** that control rather than running it uncorrected — a skipped control is an
error, not a default. The flag is derived at the same single site as `snr_r`, and an unreadable
`petroRad_r` flags `True`: unmeasurable is no more usable than absurd.

The derived SNR column is written as **`snr_r`**, matching the band suffix every other
photometric column carries. It has one derivation site, `data.pull.with_derived_columns`, which
both pull paths route through (the HTTP pull inline, the SciServer driver on its target rows
before the server-side cut) — so a corpus cannot end up without it depending on which driver
pulled it. `pull.backfill_derived` applies the same function to a corpus already on disk.

**Axis ratio is a conditioning axis, not a nuisance.** `expAB_r` / `deVAB_r` are the
non-circular inclination proxy (an independent photometric measurement, so conditioning on it to
study *vote* confusion is legitimate where the t01/t07 votes would be circular). They are
deliberately **absent** from `probing.extract.DEFAULT_NUISANCE_COLS`: regressing inclination out
as a nuisance would remove exactly the variation the confound taxonomy exists to study. An
invariant test pins that separation. Both variants are pulled — which applies per population
(disk vs elliptical) is an open item. Catalogue-only: `metadata.AXIS_RATIO_SQL` mirrors the probe
query's FROM/JOIN/ORDER BY, so `TOP n` selects the same deterministic object set, and the top-up
joins on `objID` with no image re-cut.
| **t01 debiased vote fractions** (`..._a01_smooth`, `..._a02_features_or_disk`, `..._a03_star_or_artifact`) | `zoo2MainSpecz` | **probe label** + the uncertainty firewall |

*(Schema verified live against DR17: `zoo2MainSpecz` carries ids/coords + the GZ2 vote
fractions — no redshift column — so `z` comes from `SpecObj`; `psfWidth_r` is in `Field`,
not `PhotoObjAll`.)*

**The probe label (smooth-vs-featured).** The headline read-out's target is derived
downstream (`data/metadata.featured_label`, **not** in SQL) from the t01 **featured/disk
debiased** fraction `v = a02_debiased`: `featured = v ≥ 0.5`. The headline AUC is reported
on the **confident extremes** only (`v ≥ 0.8` or `v ≤ 0.2`; `is_confident_extreme`), the
same high-consensus set the uncertainty firewall (`data/splits.py`) keeps in the axis-fit
set — so the number reflects the clean signal, not the genuinely ambiguous middle. The
`a03` (star/artifact) fraction lets a caller drop non-galaxies. **asinh `Q` stays frozen at
4; only the label columns are added to the probe pull.**

**SNR is photometric, not spectroscopic.** The image-quality nuisance probe asks "does
the concept axis read off image *depth*", so the SNR must be an **image-domain** quantity:
the r-band SNR derived from the photometry, `SNR_r ≈ 1.0857 / modelMagErr_r` (computed in
`data/metadata.py`, **not** in SQL). `SpecObj.snMedian` measures the *spectrum*
(fibre/exposure), the wrong domain — it is **not** joined.

**Distinct pretraining pull (D6).** The per-galaxy Petrosian masking box (`docs/masking.md`
§3.1) needs `petroRad` + the cutout's **arcsec/pixel** scale for the **pretraining**
corpus — the large unlabelled SDSS set, *not* the GZ2 probing set the nuisance join
covers. `petroRad` is in `PhotoObjAll` for every photometrically-detected galaxy, so it
is available, but it is a **distinct pull** the pretraining `DataSource` must fetch.

**The two pulls, concretely** (`data/metadata.py`):

- **Probing** — `zoo2MainSpecz` joined to `PhotoObjAll` on **`dr8objid = objID`** (the
  DR7 objID does *not* match DR17 `PhotoObjAll` — it returns zero rows), to `Field` on
  `fieldID` for `psfWidth_r`, and to `SpecObj` on `specObjID` for redshift. The join key
  is **verified before it is trusted**: a 10-row check confirms ra/dec agree between the
  GZ2 row and the matched `PhotoObjAll` row (a silent key mismatch returns wrong-galaxy
  metadata with *no* error), and the bulk join does not run until it passes. Queries run
  via the `SkyServerWS/SearchTools/SqlSearch` REST endpoint (`astroquery.query_sql`
  returns an HTML error page on these multi-table joins).
- **Pretraining** — `PhotoPrimary` with `type = 3 AND clean = 1` and
  `modelMag_r ∈ [14.0, 19.0]`. This reaches ≫250k but goes ~1.2 mag **fainter** than the
  GZ2 spectroscopic limit (`r < 17.77`), so the corpus skews fainter / smaller in apparent
  size (a mild, acceptable pretrain–probe shift) and `petroRad` gets **noisier on the
  faint end** → the global-box fallback rate (`data/bbox.py`) is a quantity to **watch at
  the eyeball gate**.

Both pulls carry **`ORDER BY objID`** so a `TOP n` slice is deterministic — without it the
slice (and so the manifest hash / `data_snapshot`) is non-reproducible in T-SQL. Stamps
are cut at the **native 0.396″/px**, **no rebin** (rebinning interacts with the Rung-4
resolution question; it is kept out of the data layer).

> **⚠ Stamp size — decided: 256 px (the train-compute vs clip-rate balance point).**
> Measured on a **5 000-galaxy probe sample** (`petroRad_r` percentiles, fraction whose
> `2.5·R_petro` box is clipped):
>
> | percentile | `petroRad_r` | px to hold `2.5·R` | | stamp | clipped (`R` past half-box) |
> |---|---|---|---|---|---|
> | p50 | 6.3″ | 79 px | | **64 px** | **6.96%** |
> | p90 | 11.3″ | 142 px | | **256 px** | **1.38%** |
> | p95 | 14.0″ | 177 px | | 288 px | 0.96% |
> | p99 | 22.3″ | 282 px | | 320 px | 0.64% |
> | p99.9 | 47.0″ | 594 px | | — | — |
> | p100 | 106.7″ | 1347 px | | — | — |
>
> **Why 256 — a real trade-off, not a forced move.** The `native 0.396″/px, no rebin`
> rule forbids resizing, so the **stamp size *is* the encoder input dim** (a 64 px stamp
> into a 256² encoder would need up-sampling — a rebin — so the prototype's 64 px was a
> placeholder). But that constraint does *not* by itself pick 256: any multiple of 16 is
> admissible, and bigger stamps are **not free even with SciServer cutouts** — the cost
> simply **moves from pull-time to per-step training compute**. 256² is **256 tokens**;
> 320² is **400 tokens** ≈ **1.5× the ViT self-attention cost on *every* step** of a
> from-scratch pretraining run (the dominant cost for a from-scratch encoder). Weighed
> against that: 288/320 px recovers only **0.4–0.7 pp** of galaxies (clip 1.38% → 0.96% →
> 0.64%), and the residual clipped tail is **genuinely giant nearby galaxies** (p99.9 =
> 47″, p100 = 107″) that *no* sane stamp holds — they take the documented fallback
> regardless. So **256 px is the balance point** between clip rate and train-step compute;
> that it also keeps the **D2 ViT-S/16 @ 256²** anchor is a **bonus, not the reason**. It
> is a multiple of 16 (16×16 tokens), spans 101.4″ (half-box 50.7″), and holds the full
> `2.5·R` box for any `R_petro ≤ 20.3″`.
>
> **The cost being accepted.** 256 px clips **1.38%** of galaxies (down 5× from 64 px's
> 6.96%), but that 1.4% is **not random — it is enriched in the large, extended,
> morphology-rich population the nameability/uncertainty probes most want**. This is a
> **characterised systematic limitation of the Paper-1 corpus** (recorded in the
> scratchpad's Risks), accepted deliberately, not a footnote: the paper must not claim
> coverage of the most extended morphologies.
>
> **Policy for the clipped tail (~1.4%).** A galaxy with `R_petro > 20.3″` falls back to
> the **global average-image masking box** (`data/bbox.py`, already the missing-radius
> fallback) and is **excluded from the `T2` faint-retention metric** (already the
> oversized-`R_petro` behaviour, `galaxy_zone_metrics`). No new mechanism.
>
> **Decided** (sets the D2 encoder input dim @ 256²). It does **not** affect the asinh-`Q`
> choice (the median/IQR is over the in-frame majority).

**The pull runs server-side on SciServer Compute (confirmed native-fidelity path).** A
direct SDSS frame download is infeasible at corpus scale — a hard per-IP HTTP throttle
(~1 MB/s, parallelism-proof: 8 connections aggregate the same as 1; a process pool is
*slower*) means ~10 MB/galaxy → ~11.5 days for 250k. So the cutout is done **next to the
data**: the SDSS **SAS** volume mounts the native frames inside a SciServer compute
container (`/home/idies/workspace/sdss_sas/dr17/eboss/photoObj/frames/...`); `Cutout2D`
runs server-side and only the ~50 KB stamps cross the link, never the 10 MB frames. This
preserves fidelity exactly — a server-side stamp is **byte-identical** to the HTTP-pulled
native stamp (verified `max|Δ| = 0.0`), so it passes the Rung-4 test (sky lag-1 noise
autocorr 0.026 white, high-k power fraction 0.683 full) where `hips2fits` failed. Measured
throughput (32-core container) parallelises **17.6×** (no per-IP throttle on the mounted
volume) to **3.79 gal/s → ~18 h for 250k, ~73 h for 1M** (one-time, chunked across jobs).
Driven from the repo via the SciServer Jobs API (`artifacts/sciserver_*.py`).

**Artifacts-vs-package split (token-only-in-artifacts).** The SciServer pull is deliberately
split so the **auth token never enters the importable package**:

* **`artifacts/sciserver_pull.py`** — the live driver: loads the token from `.env`
  (`_sciserver_auth.authenticate`), picks a compute domain, submits the cut jobs, polls, and
  downloads the per-chunk `corpus.tar.gz`. *All* token handling and *all* SciServer Jobs/Files
  API calls live here. `artifacts/` is excluded from lint/CI (it is investigation/ops code,
  not package code).
* **`galaxy_jepa.data.sciserver`** — the *pure*, importable, token-free helpers the driver
  reuses: `chunk_target_ids(ids, max_per_job)` (the **chunking contract**: split the ordered
  target list so each SciServer job stays under the Small-domain ~1 h timeout cap — ≈12k
  galaxies at 3.79 gal/s) and `merge_corpora(chunk_dirs, out_dir)` (stitch the per-chunk
  `DirectorySource` outputs into one corpus + a combined `manifest.json`). No network, no SDK,
  no secret.
* **`galaxy_jepa.data.pull --source {http,sciserver}`** — `http` is the in-package
  frame-download-and-cut path (small slices); `--source sciserver` fails loudly with a pointer
  to the `artifacts/` driver rather than calling the Jobs API from the package (the token rule).
  The package default stamp size is now **256 px** (the frozen spec), matching the server-side
  cutter.

---

## 4. Forks

| Fork | Options | Recommendation | Status |
|---|---|---|---|
| Single shared pipeline | FITS+asinh for both / pre-stretched cutouts for both | **FITS + asinh for both** | **decided** |
| Normalisation statistic | per-channel mean/std / robust percentiles | **per-channel mean/std, fitted post-stretch** | proposed (recommendation stands) |
| asinh parameterisation | per-channel `Q` + flux scale / single global | per-channel, tuned on pretraining corpus | open (science) — `Q` chosen via `data/q_sweep.py`; **unfrozen** pending the curve |
| Stretch-sanity galaxy set | curated faint-arm exemplars / random faint sample | **per-galaxy annulus vs corner sky on a few-thousand random probe sample** (`galaxy_zone_metrics`) | decided |
| Stamp size (= encoder input dim, no-rebin) | 256 / 288 / 320 px (+ clipped-tail policy) | **256 px** — balance of clip rate (1.38%) vs per-step train compute (320 px ≈ 1.5× attention cost/step for only 0.4–0.7 pp fewer clips); keeps the D2 ViT-S/16@256² anchor as a bonus. Clipped ~1.4% (giants, enriched in extended morphologies) take the global-box + `T2`-exclusion fallback — a characterised corpus limitation | **decided** (sets D2 encoder dim @ 256²) |
