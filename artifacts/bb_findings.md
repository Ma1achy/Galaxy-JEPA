# Brief BB — which pitch-angle method works at SDSS quality?

Status: in progress. Each stage's section is hashed (SHA-1 over the section text, from its heading to
the line before its footer, newline-joined with a trailing newline) before its run.

Tools, kept outside this repository (not vendored):
- **PyArcFiRe** — `Ma1achy/pyarcfire` (private; a clone of `pavyamsiri/pyarcfire` at 629c810, not a
  GitHub fork), at `~/Documents/pitch-methods/pyarcfire`. Run through `scripts/bb_measure.py` in its
  own uv environment. Brief BB additions, commit adb3e05:
  - **arc-length fix.** Upstream computed `|r_end − r_start / sin(pitch)|`; SpArcFiRe's
    `calcLgspArcLengths.m` is `|r_end − r_start| / sin(pitch)`. Arc length drives both the chirality
    vote and the DCO weights.
  - `get_dco_pitch_angle` = SpArcFiRe's `pa_alenWtd_avg_domChiralityOnly` (`getGalaxyParams.m`).
    Upstream's `get_overall_pitch_angle` is an unweighted mean, not SpArcFiRe's estimator.
  - `get_all_arcs_pitch_angle` = `pa_alenWtd_avg_abs` (sensitivity only).
  - `deproject(image, q, pa, centre)`: a minor-axis stretch from a supplied ellipse.
- **SpArcFiRe** reference: `waynebhayes/SpArcFiRe` (shallow clone), regression-test outputs only.
  MATLAB is not available here, so SpArcFiRe itself is never run.
- **P2DFFT 6.2** (`treuthardt/P2DFFT`, `p2dfft-6.2.tgz`), built locally with Apple clang + libomp.
  Changes from the release: the vestigial `#include <magic.h>` removed (libmagic is not called;
  CHANGES 6.2 says it was dropped), and the Homebrew-LLVM `if(APPLE)` block replaced by explicit
  flags. No algorithmic change.

## BB0a — is PyArcFiRe faithful to SpArcFiRe?

### Pre-registration

**Inputs.** SpArcFiRe's regression test `convert-FITS_no_arguments`: 25 SDSS galaxies, default
settings, all `fit_state` OK, none using a bar (`bar_used` false for all 25). For each galaxy,
PyArcFiRe is given **SpArcFiRe's own standardised image**, `-B_autoCrop.png` (256 × 256, already
star-masked, centred, deprojected and resized by SpArcFiRe). This isolates the arc-finding and fitting
stage, which is the part PyArcFiRe ports; the standardisation is not ported and is not being tested.

**Settings.** PyArcFiRe's defaults equal the test's `-S_settings.txt` on every mapped parameter:
3 orientation levels, similarity cutoff (`stopThres`) 0.15, error-ratio 2.5, merge-check size 25,
minimum cluster 150, centre cluster removed, unsharp mask 25 / 6. Declared differences, not patched:
no median filter (`medFiltRad` 1), no bar detection (moot: no galaxy uses one), and fits use the
processed intensities where SpArcFiRe sets `fitUsingNonUsmIVals` 1.

**Quantities.**
- Primary: Δ = |pitch_py| − |pitch_ref| on the DCO estimator, per galaxy.
- A galaxy where PyArcFiRe raises, finds no arcs, or ties the chirality vote is a **failure**.
- **Chirality agreement**: the sign of PyArcFiRe's DCO pitch against the sign of SpArcFiRe's.
  - Both codes use θ = atan2(−(row − c_r), col − c_c) and ρ = r₀·exp(−a·θ)
    (`fitLogSpiral.m:87`, `pyarcfire/arc/utils.py:45–49`), so the signs are compared directly.
  - The mapping is fixed by the code, not fitted to these galaxies.
- Descriptive: Spearman ρ of |pitch| between the two, arc counts, and seconds per galaxy.

**States. The first match applies (D27):**

| # | condition | state |
|---|---|---|
| 1 | failures > 5 of 25, **or** chirality agreement < 18 of the evaluable, **or** median \|Δ\| > 6°, **or** Spearman ρ of \|pitch\| < 0.5 | **BROKEN** |
| 2 | failures ≤ 2, chirality agreement ≥ 90% of the evaluable, median \|Δ\| ≤ 2°, **and** ≥ 80% of evaluable galaxies within 5° | **FAITHFUL** |
| 3 | otherwise | **DIVERGENT** |

- **The rank criterion was added before hashing, after the D28 plant showed it was needed.** A
  permuted "port" had the right distribution and chirality but no per-galaxy relation to the
  reference, and without it read DIVERGENT, not BROKEN. Real pitches cluster tightly enough that a
  6° median bound alone doesn't catch that.
- 6° is about the spread of pitch between real galaxies (Z1: sd 6.0–6.6°). A port whose typical
  error matches that spread can't rank galaxies, and that counts as broken, not divergent.
- **If DIVERGENT, every PyArcFiRe result below is reported as PyArcFiRe's, not SpArcFiRe's**, as
  the brief states.
- **If BROKEN, the SpArcFiRe family's arm of BB2 is not run on PyArcFiRe.** Stage 0 is still run and
  reported, because it shows whether the break matters on clean toys.
- n = 25 is the whole reference set. The state is about these galaxies, not a population.

**D28 (run before hashing; `bb_pitch.py --plant-fidelity`).**
- State logic on constructed inputs: the reference against itself; the reference with ~3° noise and
  two failures; permuted pitches; flipped chirality.
- One end-to-end plant through PyArcFiRe: every input **mirrored left–right**. A mirror flips
  chirality and keeps |pitch|, so chirality agreement must collapse and the state must read BROKEN.
  Only the chirality count and the state are printed for this plant. Its |Δ| is not looked at before
  hashing, because it would preview the real result.

`out/bb/bb0a_planted.json`. Every state is reachable, and the end-to-end plant fires:

| plant | state | required |
|---|---|---|
| reference against itself | FAITHFUL | FAITHFUL |
| reference + N(0, 3°), two failures (Spearman 0.83) | FAITHFUL | not BROKEN |
| reference + 3° bias | DIVERGENT | DIVERGENT |
| permuted pitches, chirality kept | BROKEN (via the rank criterion) | BROKEN |
| chirality flipped | BROKEN | BROKEN |
| 6 failures | BROKEN | BROKEN |
| **end to end: inputs mirrored left–right** | **BROKEN**, chirality agreement 6 of 25 | BROKEN |

- The mirrored plant agrees on 6 of 25, not 0. Upstream warns that PyArcFiRe is not
  transpose-invariant, so a mirror does not flip every arc's fit.
- PyArcFiRe takes 21 s per 256² galaxy (median; range 5–81 s), single process.

*BB0a pre-registration ends here: the BB0a section above (68 lines from "## BB0a"), SHA-1 `3160fe357d74116abe344f1ec39ab17d1d1a0b6d`.*

### Result (`out/bb/bb0a_fidelity.json`)

**BROKEN**, by the rank criterion. PyArcFiRe runs cleanly on SpArcFiRe's own standardised images
but does not reproduce SpArcFiRe's per-galaxy pitch.

| | value | bar |
|---|---|---|
| failures | 0 of 25 | ≤ 2 FAITHFUL; > 5 BROKEN |
| chirality agreement | 19 of 25 | ≥ 90% FAITHFUL; < 18 BROKEN |
| median \|Δ\| (DCO) | 5.1° (signed: PyArcFiRe reads 3.2° tighter at the median) | ≤ 2° FAITHFUL; > 6° BROKEN |
| within 5° | 48% | ≥ 80% FAITHFUL |
| **Spearman ρ of \|pitch\|** | **0.09** | **< 0.5 BROKEN** |
| all-arcs estimator, median \|Δ\| | 4.7° | (sensitivity) |
| seconds per galaxy | 20 | |

- Arc counts are similar (e.g. 25 vs 18, 22 vs 21, 4 vs 4). The two codes find a comparable number
  of arcs but measure different pitches from them. Six galaxies flip chirality.
- **Pre-registered consequence: the SpArcFiRe family's arm of BB2 is not run on PyArcFiRe.**
  Stage 0 is still run and reported.
- Candidate causes, **not tested and not a re-adjudication**:
  - the declared differences: no median filter, and fits on unsharp-masked rather than raw
    intensities (`fitUsingNonUsmIVals` 1);
  - 8-bit PNG input;
  - the upstream port's own status ("pre-alpha"; not transpose-invariant).
  - Finding which of these, if any, is responsible would be exploratory. It could not turn this
    state into FAITHFUL for BB.

## BB0b — does the Fourier method (P2DFFT) recover known pitch?

### Pre-registration

**Implementation.** No usable public 2DFFT beyond P2DFFT itself.
- `bendavis007/2DFFT` needs Numerical Recipes' `fourn.c`, which is not distributed, and a g77
  Fortran step.
- P2DFFT 6.2 is the maintained successor, and its PA_Notes report agreement with the original
  2DFFT within ±0.25° per radius on the authors' test images.
- So P2DFFT is the Fourier-family method. Pitch is read by the authors' own `p2pa`, run in an
  isolated uv environment with `numpy<2` (p2pa's compiled dependencies fail to import under
  NumPy 2).

**Validation set.** The authors' own idealised generator, `p2spiral`, at 255² with default core
(radius 20) and no noise:
- pitch {10, 15, …, 45}° × arms {1, 2, 3, 4} × feather {2, 5} (arm widths 5 and 11 px), sweep 180°;
- 64 images, with pitch written into the FITS headers by `p2spiral -e`.

**Protocol.** `p2dfft <files>` with defaults (inner radius 1, outer radius from image size).
- **Primary:** `p2pa -m`, the mode chosen by maximum Fourier amplitude. This is fully automated, as
  SpArcFiRe is.
- **Sensitivity:** `p2pa -a <true arms>`, the mode supplied. This is the oracle a tracing or a human
  provides.
- Error = |PA| − true pitch. A file with no PA, or a non-finite one, is a failure.

**States (automated mode). The first match applies (D27):**

| # | condition | state |
|---|---|---|
| 1 | failures > 16 of 64, **or** median \|error\| > 5° | **FAILS** |
| 2 | failures ≤ 3, median \|error\| ≤ 2°, **and** ≥ 90% within 5° | **VALID** |
| 3 | otherwise | **BIASED** (reported with its bias by pitch and arm count) |

- Mode choice is reported separately as the share of images where amplitude picks the true arm
  count.

**What this can and cannot check.**
- It checks that the build, the I/O and the pitch read-out recover the pitch of the model the
  method's authors built it around.
- It cannot check the method against a different model family, against real galaxies, or against
  the original 2DFFT's outputs: the `p2dfft-test` package is not public.
- **A VALID here is a floor, not independent evidence.** Stage 0's H&T toys are the external check,
  and even they come from the same group (Hewitt & Treuthardt).

**D28 (run before hashing; `bb_pitch.py --plant-fourier`).**
- State logic on constructed error vectors.
- End to end:
  - the true pitches shuffled against the images must read FAILS;
  - 8 pure-noise images (Gaussian, no spiral) must fail or return a PA unrelated to anything,
    reported as their |PA| spread.

`out/bb/bb0b_planted.json`. Every state is reachable:

| plant | state | required |
|---|---|---|
| exact | VALID | VALID |
| N(0, 1°) | VALID | VALID |
| +3° bias | BIASED | BIASED |
| 20 failures | FAILS | FAILS |
| N(0, 10°) | FAILS | FAILS |
| **end to end: true pitches shuffled** | **FAILS** | FAILS |
| **end to end: 8 pure-noise images** | **no failures; \|PA\| 1.7–11.8°** | reported |

- **P2DFFT has no failure mode.** On pure noise it returns a pitch every time, spread over 2–12°.
  That carries into BB2: at low SNR the Fourier method will produce a number, not a failure, so its
  failure rate cannot be compared with SpArcFiRe's, only its error. This is recorded before BB2 is
  designed.

*BB0b pre-registration ends here: the BB0b section above (67 lines from "## BB0b"), SHA-1 `7501a8cc07c8fa3886c290cb754d8d65a650825e`.*

### Result (`out/bb/bb0b_fourier.json`)

**VALID.** On its own generator, P2DFFT recovers pitch.

| | automated (`-m`) | oracle arms (`-a`) |
|---|---|---|
| failures | 0 of 64 | 0 |
| median \|error\| | **0.62°** | 0.62° |
| within 5° | 92% | 92% |
| median signed error | +0.21° | +0.21° |

- **Mode choice is perfect here.** Amplitude picks the true arm count on all 64, so the two columns
  are identical. Real images will not be this kind.
- **Bias by pitch** (median signed error): 10° **+4.2°**, 15° +1.5°, 20° +0.7°, 25–45° within ±0.6°.
  Tight spirals read looser at 255 px with a 180° sweep. This is the regime where BB2 will matter
  most.
- **Median \|error\| by arm count:** 1 arm 1.3°, 2 arms 0.6°, 3 arms 0.6°, 4 arms 0.3°.
- As pre-registered, this is a floor: the build and read-out work on the authors' model family.
  Nothing more.

## Decision after BB0 (user, 2026-09-24)

- **PyArcFiRe is recorded as BROKEN for BB and parked.** It is not debugged and not used in any BB
  result.
- The SpArcFiRe arm uses the **official compiled SpArcFiRe on the free MATLAB Runtime** (BB0c):
  `matlab/findClusterArcsServer`, built against Runtime 9.2 (R2017a), in an x86-64 Linux container
  under emulation.
- **BB2 addition, both methods.** Every stage includes noise-only and armless (smooth-disc)
  synthetics. Each method's output distribution on them defines a null range, and a pitch inside it
  counts as **NO DETECTION**. Detection rates are reported alongside accuracy.
