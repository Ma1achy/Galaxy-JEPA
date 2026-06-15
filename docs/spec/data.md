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
   **survive** stretch+normalise on a few known faint-arm galaxies while the **sky-noise
   floor stays controlled**. It is Tier-2 (`docs/spec/validation.md` `T2.stretch-sanity`)
   and **pairs with the collapse monitor**: if the encoder starts modelling noise, the
   stretch is too aggressive.

This is reflected in the scratchpad's preprocessing section (proposed edit).

---

## 3. The CasJobs join + declared metadata columns

The GZ2 → SDSS join (CasJobs/SkyServer) supplies the nuisance battery on the **probing**
corpus. Declared columns (existence checked at Tier-1 `T1.metadata-columns-real`):

| Column | Source | Used by |
|---|---|---|
| `z` (redshift) | spec/photo-z | nuisance probe |
| apparent magnitude (`mag_r`, …) | SDSS photometry | nuisance probe |
| `petroRad` (Petrosian radius) | SDSS `PhotoObjAll` | nuisance probe **and** per-galaxy masking box |
| SNR | photometric/field tables | nuisance probe |
| PSF width | field tables | nuisance probe |

**Distinct pretraining pull (D6).** The per-galaxy Petrosian masking box (`docs/masking.md`
§3.1) needs `petroRad` + the cutout's **arcsec/pixel** scale for the **pretraining**
corpus — the large unlabelled SDSS set, *not* the GZ2 probing set the nuisance join
covers. `petroRad` is in `PhotoObjAll` for every photometrically-detected galaxy, so it
is available, but it is a **distinct pull** the pretraining `DataSource` must fetch.

---

## 4. Forks

| Fork | Options | Recommendation | Status |
|---|---|---|---|
| Single shared pipeline | FITS+asinh for both / pre-stretched cutouts for both | **FITS + asinh for both** | **decided** |
| Normalisation statistic | per-channel mean/std / robust percentiles | **per-channel mean/std, fitted post-stretch** | proposed (recommendation stands) |
| asinh parameterisation | per-channel `Q` + flux scale / single global | per-channel, tuned on pretraining corpus | open (science) |
| Stretch-sanity galaxy set | curated faint-arm exemplars / random faint sample | curated exemplars + a sky-noise control | open (minor) |
