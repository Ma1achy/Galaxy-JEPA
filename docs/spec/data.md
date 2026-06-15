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

*(Schema verified live against DR17: `zoo2MainSpecz` carries only ids/coords + the GZ2
vote fractions — no redshift column — so `z` comes from `SpecObj`; `psfWidth_r` is in
`Field`, not `PhotoObjAll`.)*

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

> **⚠ Stamp size vs the largest galaxies (open — revisit before the corpus slice).** The
> current stamp is **64 px** = 25.3″ across (half-width 12.67″). A galaxy with
> `petroRad_r > ~12.7″` has its whole faint-outskirt annulus `[R_petro, 2.5·R_petro]`
> **off-frame** — so it is excluded from the `T2` faint-retention metric, *and* the
> `k = 2.5` masking box clips it. On a 100-galaxy probe slice this is ~5% (`petroRad_r`
> ranges to ~23″), and it bites **disproportionately the extended, morphology-rich
> galaxies the nameability/uncertainty probes most want**. Holding `2.5·R_petro` for a
> 23″ galaxy needs ~290 px. The stamp size is therefore a **fork to settle before the
> ≥250k slice** (larger stamps ⇒ more bytes/pixels per galaxy; trade-off against the
> encoder input size). It does **not** affect the asinh-`Q` choice (the median/IQR is
> over the in-frame majority).

---

## 4. Forks

| Fork | Options | Recommendation | Status |
|---|---|---|---|
| Single shared pipeline | FITS+asinh for both / pre-stretched cutouts for both | **FITS + asinh for both** | **decided** |
| Normalisation statistic | per-channel mean/std / robust percentiles | **per-channel mean/std, fitted post-stretch** | proposed (recommendation stands) |
| asinh parameterisation | per-channel `Q` + flux scale / single global | per-channel, tuned on pretraining corpus | open (science) — `Q` chosen via `data/q_sweep.py`; **unfrozen** pending the curve |
| Stretch-sanity galaxy set | curated faint-arm exemplars / random faint sample | **per-galaxy annulus vs corner sky on a few-thousand random probe sample** (`galaxy_zone_metrics`) | decided |
