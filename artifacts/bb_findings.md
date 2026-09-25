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
- **SpArcFiRe** reference: `waynebhayes/SpArcFiRe` (shallow clone). In BB0a it supplied only
  regression-test outputs. From BB0c the official compiled build runs in its own container
  (`~/Documents/pitch-methods/sparcfire-docker/`; see BB0c).
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

## BB1 Stage 0 — replicating the published toy-set accuracies

### Pre-registration

**The set.** The Portman et al. archive, `SpArcFiRe-HT-Response.zip`
(sparcfire.ics.uci.edu/archive/SpArcFiRe-HT-Response/, accessed 2026-09-24). It holds 200 H&T toy
inputs as 256² RGB JPEGs (103 barred, 97 unbarred), plus the `run.sh` Portman et al. executed.
- Filenames give the truth: `{TOY|BAR}_{pitch}_a{arms}_f{feather}_{c|b}{…}_{sweep}L`.
- All toys are S-wise. Only \|pitch\| is compared.
- **The published series** (H&T 2020 Figs 3–7), baseline pitch 25°, 2 arms, f5, 180°:
  - **pitch:** 10–60° × {TOY, BAR}, 22 images;
  - **arms:** a1–a6;
  - **width:** f1–f7;
  - **sweep:** 90–540°;
  - **bar length:** b50–b100, barred only.
- The union, with the baseline counted once, is **60 galaxies**: Portman et al.'s evaluation set.
- The other 140 (a1/f1/f3 pitch variants and `c` variants) are run and reported descriptively. They
  enter no replication state.

**P2DFFT.**
- Each JPEG becomes a single-plane FITS (the RGB mean) at 256², with default radii.
- **Primary:** `p2pa -a <true arms>`.
  - `p2pa` uses a FITS `ARMS` keyword when present, and H&T's generator-made FITS most plausibly
    carried one. So supplied arms is the protocol closest to their "p2dfft:auto".
  - This is stated before any toy is run.
- **Sensitivity:** `p2pa -m`, amplitude-chosen and fully automated.
- **Declared difference:** H&T ran on 835² FITS. Only the 256² JPEGs are public, so resolution is
  3.3× lower, with JPEG compression.
- **Published targets**, the mean \|error\| per series (H&T Table; p2dfft:auto):

  | series | non-barred | barred |
  |---|---|---|
  | pitch | 1.44° | 1.80° |
  | arms | 1.44° | 0.89° |
  | width | 1.89° | 2.00° |
  | sweep | 0.91° | 1.84° |
  | bar length | — | 5.41° |

- A cell **replicates** iff ours ≤ published + max(1.0°, 0.3 × published). The test is one-sided.
  A cell below published − the same tolerance is also counted as **better than published** and
  reported: a protocol difference, not a harness failure.
- **Amended before hashing, after D28.** The first draft was two-sided, and a zero-error method read
  NOT REPRODUCED. That contradicted this section's own reading of "better".

**SpArcFiRe** (the official build, BB0c; run later, pre-registered here).
- **Primary settings:** `run.sh` as executed — `-stopThres 0.1 -useImageStandardization 0
  -allowArcBeyond2pi 0 -errRatioThres 5 -unsharpMaskSigma 10`.
- **Sensitivity:** the same plus `-unsharpMaskAmt 15`.
  - Portman et al.'s Table 1 lists `unsharpMaskAmt` 15, but `run.sh` does not set it.
  - This discrepancy is recorded. Neither is chosen by outcome.
- Estimator: \|`pa_alenWtd_avg_domChiralityOnly`\|. A galaxy with no arcs, or no DCO pitch, is a
  failure.
- **Published target** (Portman et al. 2023, on the 60): 5 failures; 54 within 2°; the remaining one
  at 3.2°.

**States, per method. The first match applies (D27):**

| method | REPRODUCED | NOT REPRODUCED | PARTIAL |
|---|---|---|---|
| P2DFFT (primary protocol) | ≥ 7 of the 9 cells replicate | ≤ 4 cells replicate | otherwise |
| SpArcFiRe (primary settings) | failures ≤ 8, **and** ≥ 85% of the rest within 2°, **and** max error ≤ 5° | failures > 15, **or** < 60% of the rest within 2° | otherwise |

- The direction is always reported, as are the cells better than published.
- **Reading:** "if BB's harness can't reproduce known results, nothing downstream is read." Under
  NOT REPRODUCED for a method, that method's BB2 results are not read until the cause is found.

**D28 (run before hashing; `bb_pitch.py --plant-stage0`).**
- State logic on constructed values for both methods.
- End to end on P2DFFT: the true pitches shuffled within the 60 must read NOT REPRODUCED.

D28 (`out/bb/stage0_planted.json`). Every state is reachable for both methods:

| plant | state |
|---|---|
| P2DFFT: exact | REPRODUCED |
| P2DFFT: errors at the published size | REPRODUCED |
| P2DFFT: 6° on the bar, sweep and width series | PARTIAL |
| P2DFFT: 6° everywhere | NOT REPRODUCED |
| **P2DFFT end to end, truth shuffled within the 60** (both protocols) | **NOT REPRODUCED** |
| SpArcFiRe: as published (5 fail, 54 < 2°, one 3.2°) | REPRODUCED |
| SpArcFiRe: 10 failures, the rest at 1° | PARTIAL |
| SpArcFiRe: 20 failures | NOT REPRODUCED |
| SpArcFiRe: half within 2° | NOT REPRODUCED |

*BB1 Stage 0 pre-registration ends here: the BB1 Stage 0 section above (83 lines from "## BB1 Stage 0"), SHA-1 `29260b7e7eac99d22111af5fbc9cb89f9a435f1b`.*

### Result — P2DFFT (`out/bb/stage0_p2dfft.json`)

**REPRODUCED**, under both protocols. The harness recovers H&T's published p2dfft:auto accuracies
from the public 256² JPEGs.

| series | non-barred: ours / published | barred: ours / published |
|---|---|---|
| pitch | 1.27° / 1.44° | 2.12° / 1.80° |
| arms | 1.33° / 1.44° | 0.55° / 0.89° (automated: 0.64°) |
| width | 1.05° / 1.89° | 1.37° / 2.00° |
| sweep | 0.82° / 0.91° | **3.17° / 1.84°**: does not replicate (bar 2.84°) |
| bar length | — | 4.81° / 5.41° |

- **8 of 9 cells replicate, and none is better than published by more than the tolerance.**
- The automated (amplitude) protocol gives the same state and the same cells except barred arms.
  It picks the true arm count on 84% of the 200 toys.
- **Barred sweep misses on one toy.** `BAR_25_a2_f5_c25_090L` reads +16.6°, and the other five
  barred sweeps are within 1.6°. The unbarred 90° arm is also the worst of its series (+3.1°). H&T
  flag the same systematic error for the shortest (90°) arms. Why it is larger here than in H&T is
  not tested.
- Median \|error\| over all 200 toys: 1.57° (supplied arms) and 1.61° (automated).

## BB0c — the official SpArcFiRe, on the free MATLAB Runtime

**The image.** Tag `sparcfire-official:r2017a`, image ID
`sha256:bd682e088bf4ca27931950dec4e7d53aa2a313183662edb337ddfebd3251ffa4`.
- Its recipe is committed at `tools/sparcfire-docker/`: the Dockerfile, `run-in-image.sh` and a
  README with the installer's URL and SHA-256. It holds no SpArcFiRe code and no Runtime.
- The copy that built the image lives in `~/Documents/pitch-methods/sparcfire-docker/`. Its build
  instructions are identical to the committed ones; only the comments differ.
- **Regression verdict: PARTIAL.** GALFIT is untested, because the image has Python 3.6; see below.
- Base: `ubuntu:18.04`, linux/amd64, run under emulation on the M3 (Docker Desktop, aarch64 host).
- Follows `UbuntuSetup.sh`: apt python2.7 with numpy, scipy, Pillow and astropy; MATLAB Runtime
  R2017a (9.2) installed silently and moved to `/pkg/matlab/R2017a`.
- Added for the repo's own scripts: SExtractor (symlinked as `sex`), ImageMagick, gawk, and
  build-essential (the suite compiles `delete-commas-inside-quotes`; the driver runs `make all`).
- Runtime installer: `MCR_R2017a_glnxa64_installer.zip`, 1,334,959,803 bytes, SHA-256
  `a54f04d360e540986c83dec87bb940c6d37a1843574328729f44019f5fcfac1c` (MathWorks, accessed
  2026-09-24).
- The SpArcFiRe clone is mounted at `/sparcfire`. The binary is `matlab/findClusterArcsServer`
  (ELF x86-64); `ldd` resolves every library against the Runtime.

**Regression verdict: PARTIAL.** `regression-test-all.sh` exits with 1 failure.
- All four SpArcFiRe regression tests **pass**, with 0 failures, against the repo's reference outputs
  (`regression-diff.sh`: 2% relative tolerance, time columns excluded).
  - `convert-FITS_+_generate_fit_quality`: 25 SDSS FITS galaxies.
  - `convert-FITS_no_arguments`: the same 25, default path.
  - `disparate-sides`: post-processing statistics on stored TSVs, no MATLAB.
  - `image_guiding`: 9 galaxies × 5 guiding thresholds.
- The 1 failure is the suite's **GALFIT section, which did not run.** It needs Python ≥ 3.7 and the
  18.04 image has 3.6, so the script tallies a failure without testing anything. GALFIT is a separate
  module (`GalfitModule/`), and BB never calls it.
- Reading: the arc-finding path BB uses reproduces its reference outputs under emulation. The install
  is not a full PASS, because one component was not exercised.
- Outputs are archived beside the image (`sparcfire-docker/regress_outputs/`), and the clone is
  cleaned.

**Timing under emulation** (wall clock, one process, ~124% CPU, peak RSS 1.0 GB):

| run | images | wall | per image |
|---|---|---|---|
| regression: SDSS FITS + fit quality (incl. FITS conversion and SExtractor) | 25 | 199 s | 8.0 s |
| regression: SDSS FITS, default | 25 | 178 s | 7.1 s |
| regression: image guiding | 45 runs | 620 s | 13.8 s |
| Stage 0 toys, 256² JPEG (primary) | 200 | 673 s | 3.4 s |
| Stage 0 toys (sensitivity) | 200 | 681 s | 3.4 s |

Stage 0 takes 11 minutes per setting, far inside the ~12 h gate, so no cloud VM is needed. For the
BB2 budget, SDSS-sized FITS run at ~7–8 s each here: ~2 h per 1,000 images.

**Reading `galaxy.tsv`.**
- `galaxy.csv` holds unquoted commas and does not parse. SpArcFiRe's own `galaxy.tsv` is read
  instead, by header name.
- A zero-arc galaxy's row carries 4 extra padding fields and is shifted from the header from the
  arc-length block onward. A rejected input is written as a 2-field row.
- Such rows are counted as failures. `bb_pitch.sf_read` first checks that none carries a finite value
  anywhere in the pitch block, and raises if one does.
- Checked here: all 32 ragged rows have 0 arcs in `galaxy_arcs.tsv`, and all 167 regular rows have
  ≥ 1.

### Result — SpArcFiRe (`out/bb/stage0_sparcfire_{primary,amt15}.json`)

Estimator \|`pa_alenWtd_avg_domChiralityOnly`\|. A galaxy with no arcs is a failure. States are read
on the 60 under the hashed rule.

| settings | state | failures (60) | within 2° | max \|error\| |
|---|---|---|---|---|
| published (Portman et al. 2023) | — | 5 | 54 / 55 | 3.2° |
| **primary: `run.sh` as executed** | **PARTIAL** | **11** | 48 / 49 (98%) | 2.85° |
| sensitivity: + `-unsharpMaskAmt 15` | REPRODUCED | 8 | 51 / 52 (98%) | 3.31° |

- **The pre-registered state is PARTIAL.** It fails only on the count: 11 failures against a bound of
  ≤ 8. Among the galaxies it measures, it is at least as accurate as published.
- **`run.sh` does not run at Table 1's settings.** SpArcFiRe's settings dump shows
  `unsharpMaskAmt: 6` (the build default) on all 200 primary runs, and 15 on all 200 sensitivity
  runs.
  - The sensitivity run matches the published profile closely: 8 against 5 failures, and max 3.31°
    against 3.2°.
  - That is evidence Portman et al.'s numbers came from `unsharpMaskAmt` 15, not from `run.sh` as
    archived.
  - Per the pre-registration, neither setting is chosen by outcome. The primary stays PARTIAL, and
    the sensitivity result stands beside it.
- **The primary run's 11 failures fall along a grain:**
  - narrow arms: `a2_f1` and `a2_f2` at 25°, barred and unbarred;
  - 90° sweep: barred and unbarred;
  - low pitch, unbarred: 10° and 15°;
  - many arms, unbarred: 3, 5 and 6.
  Every one is a no-arcs output, not a wrong pitch. The sensitivity setting recovers the unbarred
  arm-count failures (3 → 0) and loses one width toy (2 → 3).
- **When it returns a pitch, it is near-exact.** Primary over all 200: median \|error\| 0.19°,
  median signed error +0.04°, and 98% within 2°. The two large misses are outside the 60:
  `TOY_25_a2_f5_c15` at +22.2° and `TOY_40_a2_f3` at 3.8°.
- **Per series**, mean \|error\| over the measured toys, then failures (primary):

  | series | non-barred | barred |
  |---|---|---|
  | pitch | 0.39°, 2 fail | 0.46°, 0 fail |
  | arms | 0.58°, 3 fail | 0.22°, 0 fail |
  | width | 0.92°, 2 fail | 0.17°, 2 fail |
  | sweep | 0.17°, 1 fail | 0.11°, 1 fail |
  | bar length | — | 0.07°, 0 fail |

- Over all 200: 33 failures under primary and 26 under sensitivity.
- **Both methods, side by side.** P2DFFT never fails, but errs by ~1–2° on average (median 1.57° over
  the 200). SpArcFiRe errs by ~0.2°, but returns nothing on 17% of the toys at `run.sh` settings
  (13% at 15).
  - The two failure modes differ in kind. BB2's NO DETECTION null range is what lets them be
    compared on one scale.
- **Not tested:** whether emulation, the ImageMagick version used for the JPEG → PNG step, or the
  Runtime build shifts which faint arcs survive clustering. The regression pass bounds the numerics
  on the SDSS path, not arc survival on toys.

## BB2 — synthetic SDSS-matched spirals: does either method measure pitch at SDSS quality, and where does it stop?

**Question.** Both methods are shown the same synthetic spirals of known pitch. Each spiral borrows
its degradation (PSF, size, surface brightness, sky noise) from a real Galaxy Zoo 2 spiral, the
"donor", and so inherits that donor's V1 visibility.
- Where does each method detect arms?
- How accurately does it then measure pitch?
- Does it rank galaxies by pitch faithfully?
- At what visibility does it stop?

Code `artifacts/bb2.py` (sha256 `882cc6801b92b05e…` at this hash). Outputs go to `out/bb/bb2/`.

**Generator** (known in advance; D28 below).
- **Model.** Exponential disc × (1 + A·window·((1 + cos mψ)/2)^4), with
  ψ = θ − χ·ln(r/r_in)/tan φ − θ0, plus a round de Vaucouleurs bulge (B/T ~ U(0.05, 0.3)).
- **PSF and pixels.** Gaussian PSF with FWHM = the donor's `psfWidth_r`, rendered at 4×
  oversampling and integrated to native 0.396″ pixels on a 256² stamp.
- **Scale length.** h is solved so the PSF-convolved model's SDSS Petrosian radius (η = 0.2) equals
  the donor's `petroRad_r`. If that is unreachable in a 256 px stamp, h is NaN: dropped and counted.
- **Flux.** The donor's `modelMag_r`.
- **Noise model (declared).** The donor's own r-band sky σ (robust σ of its v1 stamp's 16 px
  border) plus Poisson noise from the galaxy at 4.7 e⁻/DN, 0.005 nMgy/DN. This gives synthetic
  SNR 1.2–2.1× the donor's `snr_r`. A uniform σ matched to `snr_r` was rejected: it put 2–3.6× the
  real sky noise in.
- **Donors.** Y's spiral definition (features ≥ 0.5, not edge-on ≥ 0.5, spiral ≥ 0.5): 42,564,
  of which 41,476 are eligible (finite PSF and `expAB_r`, `petroRad_r` > 0, not
  `petrorad_suspect`).
- **V1** = `v1_loose_ends.composite` over the pool: loadings mag 0.566, snr −0.499,
  size −0.463, z 0.465. Quintile edges [−1.673, −0.434, 0.643, 1.670]; Q1 is the most visible.
- **FITS files** are float32 in nanomaggies with a blind header (`BUNIT`, `PIXSCALE` only; p2pa
  reads an `ARMS` keyword, so the truth is never written).
- **Grid axes.** Pitch {5, 10, 15, 20, 25, 30, 40}°; arms {1, 2, 3, 4}; contrast A ∈
  {0.3, 0.6, 1.2}; chirality ±1 at random.

**Stages.**
1. Clean and face-on.
2. Inclined, q = max(`expAB_r`, 0.3) of the donor, with a random PA.
3. As 2, but flocculent (lognormal σ 0.7, smoothed 0.3h), broken (2–3 gaps of ±0.125 in ln r per
   arm) or barred (a Gaussian bar, 10% of the flux, arms starting at the bar end).

Every stage includes nulls: noise only, and armless discs with the same donors.

**Methods and settings.**
- **SpArcFiRe**, official compiled build (BB0c image `sha256:bd682e08…`).
  - Primary: its documented defaults via `-convert-FITS`.
  - Inclined stages: the true ellipse supplied through `-elps_dir`, in SpArcFiRe's own convention
    (calibrated; D28 (b)).
  - Sensitivities: `run.sh`'s settings, with and without `-unsharpMaskAmt 15`. These set
    `useImageStandardization 0`, which returns before the ellipse branch. **The sensitivities
    therefore see un-deprojected inclined images (declared).**
  - Estimator: |`pa_alenWtd_avg_domChiralityOnly`| (DCO). All-arcs `pa_alenWtd_avg` is reported
    for Peng P5 only.
- **P2DFFT 6.2**, on the image (Stage 1) or on the image deprojected with the true ellipse
  (Stages 2–3; cubic `map_coordinates`).
  - Inner radius 1 px; outer radius min(round(1.5 × `petroRad_r`/0.396), 127) px.
  - The p2dfft 6.2 parameter-file parser needs a 4th field (a stale getline token; worked around,
    no algorithmic change).
  - Primary: its own mode choice (`p2pa` auto). Sensitivity: the oracle arm count.

**Detection (user, 2026-09-25; replaces the null-range rule of the 2026-09-24 decision for both
methods).**
- **SpArcFiRe** detects when it returns any arc (a finite DCO).
- **P2DFFT detects 100% by definition (option 1).** Every finite answer is a detection.
  - Its answers on null images are reported as a false-answer rate (100% expected) with their
    |pitch| distribution.
  - Why: the null-range rule failed D28. P2DFFT's |pitch| on nulls spans [2.2°, 90°], which
    saturates the range, so 0/10 strong spirals were "detected" although all 10 were within 2°.
    Neither its error, its amplitude nor its FFT SNR separated nulls from spirals.
- **Finding, recorded now.** P2DFFT never abstains. On armless discs and noise it returns pitches
  spanning 2–90° (D28 (d): 60/60 nulls answered, median 41°). So in any published P2DFFT catalogue,
  values for galaxies without clear arms are indistinguishable from measurements.
- **Consequence, stated before the hash.** Detection is 100% for P2DFFT, so wherever SpArcFiRe
  detects ≤ 90%, the detection route to FOURIER BETTER opens (logic plant: both methods good,
  SpArcFiRe at 90% → FOURIER BETTER, 9/10 draws). The headline verdict must be read with **common
  support**, which says whether P2DFFT's extra answers are measurements.

**Per method, per stage × V1 quintile cell. Two co-primary ladders, reported side by side.**
- **Accurate (WORKS)** iff detection ≥ 50% AND RMS of (measured − true) over detected spirals
  ≤ 7°. 7° is the ~7° spread between real galaxies (Z1).
- **Rank-faithful (WORKS)** iff detection ≥ 50% AND Spearman(measured, true) ≥ **0.70**, over
  detected spirals with true pitch 10–30° (sd 7.1°, matching the real ~7° spread).
  - Justification, fixed before hashing: with measurement error e on a truth of spread σ,
    ρ ≈ σ/√(σ²+e²). That is 0.707 at e = σ = 7°, the rank equivalent of the 7° RMS bar.
  - Two methods each at 0.70 with truth agree with each other at ≈ 0.5, which is Z1's bar.
- **Cell states** for both ladders:
  - INSUFFICIENT if n < 40;
  - FAILS if there are fewer than 20 detected spirals (or 20 in the 10–30° set for rank) and
    detection < 50%;
  - INSUFFICIENT if there are fewer than 20 but detection ≥ 50%.
- **Borderline** means a bootstrap CI (2,000 draws) straddles a bar. It feeds FRAGILE.
- **Also reported per cell:** detection, RMS, bias (median error), scatter (1.4826·MAD), ρ.

**Quintile calls and the stage verdict** (unchanged from the logic D28 below; applied per ladder).
- **Quintile call:**
  - INSUFFICIENT if either cell is;
  - NEITHER if both fail;
  - S or F if only one works;
  - if both work, a paired bootstrap decides.
- **Paired bootstrap:**
  - Accuracy ladder: better via detection (Δdet ≥ 0.10 with CI > 0, RMS not worse beyond 1°) or
    via RMS (ΔRMS ≥ 1° with CI < 0, detection not worse beyond 0.10).
  - Rank ladder: the same with Δρ ≥ 0.05 replacing ΔRMS.
  - Below the margins it is a TIE, noted if significant.
- **Stage verdict** (first match applies):
  1. INSUFFICIENT (≥ 2 insufficient quintiles);
  2. BOTH FAIL AT SDSS QUALITY (each method works in ≤ 2 quintiles);
  3. DEPENDS ON VISIBILITY (at least one S call and one F call);
  4. SPARCFIRE BETTER (no F call, and ≥ 2 S calls or S works in ≥ 3 while F works in ≤ 2);
  5. FOURIER BETTER (mirror of 4);
  6. BOTH ADEQUATE (both work in ≥ 3);
  7. an unmatched case raises.
- **FRAGILE:** the alternative verdicts reached by pushing every borderline cell to WORKS, then
  to FAILS. The crossover quintiles are reported.
- **Stop-visibility** per method and ladder: the first quintile, from most to least visible,
  where it stops working (with the V1 edge), flagged if non-monotone.
- **BB3 uses a method as a reference only if it is rank-faithful in ≥ 3 of 5 Stage-3 quintiles
  (primary settings).**

**Common support** (per stage, pooled over quintiles, and per quintile as description).
- **(a)** Both methods on the spirals SpArcFiRe detected: RMS, bias, scatter and ρ for each, and
  paired ΔRMS and Δρ with CIs.
- **(b)** P2DFFT's median |error| on SpArcFiRe's abstentions minus on its detections (bootstrap):
  - **ABSTAINS WHERE HARD** if the CI is > 0 and the difference ≥ 1°;
  - **NOT WHERE HARD** if the CI lies below 1°;
  - **UNRESOLVED** otherwise;
  - **INSUFFICIENT** under 20 in either group.

**Diagnostic, nulls and Peng.**
- **Diagnostic:** do the methods agree with each other on the co-detected spirals? Z1's bar:
  - AGREE if the outer envelope of the Fisher (Bonett–Wright) and bootstrap CIs is ≥ 0.5;
  - DISAGREE if it is < 0.5;
  - UNRESOLVED otherwise;
  - INSUFFICIENT under 30.
- **Nulls, every stage:** SpArcFiRe's any-arc rate (a false detection), and P2DFFT's false-answer
  rate and |pitch| quantiles. Everything is also broken down by pitch, arms, contrast and stage.
- **Peng et al. 2018's SpArcFiRe trends** on synthetics with truth. Each reads MATCHES / OPPOSITE
  / NOT RESOLVED / NOT TESTABLE:
  - P1: convergence;
  - P2: error grows with faintness;
  - P3: loose arms err more in degrees AND tight arms more as a fraction (read as a pair, because
    P3b alone saturates on the null);
  - P4: tight arms are lost first;
  - P5: DCO is less affected than all arcs.
  - Over 40 null draws, P2 read MATCHES twice (the nominal ~2.5% one-sided rate). A single MATCHES
    is read with that rate.

**Known in advance (declared, not findings).**
- **SpArcFiRe compresses loose arms**:
  - 40° → 31.8° face-on;
  - 30° → ~19.5° at q = 0.5, with the true ellipse and with its own;
  - D28 (c): median |error| 6.1° on 15°/30° at q = 0.5, while supplied-vs-own ellipse agree to 0.52°.
  So the accuracy ladder will penalise SpArcFiRe at 30–40°, and the rank ladder is where
  compression is not a failure.
- **The `run.sh` sensitivities see un-deprojected inclined images.**
- The noise model above.

**D28, before this hash** (`plant_generator.json`, `plant_ellipse.json`, `plant_nulls.json`,
`plant_logic.json`).
- **(a) Generator validity** on 11 clean spirals. SpArcFiRe: median |err| 1.78°, ρ 0.92.
  P2DFFT auto and oracle: 0.26°, ρ 0.92. All VALID. The chirality convention is consistent
  (S-wise ↔ P2DFFT negative).
- **(b) Ellipse convention** calibrated on 9 armless discs (angle sign −1, offset 0; σ and length
  per h linear in q).
- **(c) Ellipse plumbing: VALID.** The supplied ellipse is echoed exactly. SpArcFiRe's pitch with
  the supplied ellipse matches that with its own fit (median 0.52°). P2DFFT on our deprojection:
  median 1.37°.
- **(d) Nulls under option 1: VALID.** P2DFFT answered 60/60 nulls (quantiles 2.1°, 4.2°, 14.1°,
  41.2°, 79.7°, 90°, 90°) and was right on the strong spirals (median 0.61°). The null-range rule
  it replaces is recorded as FAILED.
- **(e) Logic.** Every verdict is reached from fakes at the planned cell size:
  - BOTH ADEQUATE; SPARCFIRE BETTER; FOURIER BETTER; BOTH FAIL; DEPENDS (crossover Q2–Q3);
  - detection-driven and accuracy-driven calls; a below-margin TIE; FRAGILE; INSUFFICIENT
    (30 per quintile).
  - Rank ladder (modal verdict over 10 draws):
    - compressed 0.4·truth+7 → accuracy BOTH FAIL 10/10, rank BOTH ADEQUATE 9/10;
    - shuffled S → rank FOURIER BETTER 10/10;
    - ρ on the 0.70 bar → FRAGILE 10/10;
    - option 1 → FOURIER BETTER via detection 9/10.
  - Common support (b): ABSTAINS WHERE HARD, NOT WHERE HARD and INSUFFICIENT are all reached.
  - The BB3 rule is reached.
  - Diagnostic: AGREE, DISAGREE, UNRESOLVED and INSUFFICIENT are all reached.
  - Peng: every test MATCHES on a planted pattern.

**Pilot, then stop** (budget ≤ 8 h of SpArcFiRe in total; one heavy process at a time).
- 300 Stage 1 images: per V1 quintile, 52 spirals at random grid points and 8 nulls
  (4 noise, 4 armless).
- SpArcFiRe default; P2DFFT auto and oracle.
- **The pilot reports timing, detection rates and null answers only. No accuracy is computed.**
  Accuracy is the grid's test.
- Then the full grid for Stages 1–3 plus nulls is proposed. It was provisionally approved for
  the pilot: per stage, 7 pitches × 4 arms × 3 contrasts = 84 per quintile × 5 quintiles, plus
  nulls; SpArcFiRe sensitivities on half.
- **Stop for the go-ahead before the full run.**

*BB2 pre-registration ends here: the BB2 section above (193 lines from "## BB2"), SHA-1 `49f8772d4b1e9772189fc9d13bc912ba97997723`.*

### Pilot result — Stage 1, 300 images (`out/bb/bb2/pilot.json`; detection and timing only, no accuracy computed)

**Timing.** Render 0.31 s/image. SpArcFiRe 5.28 s/image (1,583 s for 300, emulated x86-64).
P2DFFT 0.32 s/image. No h was unreachable (0 dropped).

**Detection by V1 quintile (52 spirals + 8 nulls each).**

| Quintile | SpArcFiRe any-arc, spirals | P2DFFT answers, spirals | SpArcFiRe any-arc, nulls | P2DFFT answers, nulls |
|---|---|---|---|---|
| Q1 | 0.92 | 1.00 | 0.13 | 1.00 |
| Q2 | 0.85 | 1.00 | 0.50 | 1.00 |
| Q3 | 0.94 | 1.00 | 0.38 | 1.00 |
| Q4 | 0.87 | 1.00 | 0.38 | 1.00 |
| Q5 | 0.92 | 1.00 | 0.50 | 1.00 |

- **Nulls by kind.** SpArcFiRe returns arcs on **0/20 noise-only images but 15/20 armless discs
  (75%)**, a median of 2 arcs (up to 15) against 5 on spirals. P2DFFT answers 40/40, with |pitch|
  quantiles 1.3°, 6.3°, 18.0°, 48.2°, 76.0°, 90°, 90°.
- **SpArcFiRe detection on spirals** by pitch: 5° 0.77, 10° 0.89, 15° 0.90, 20° 0.95, 25° 0.97,
  30° 0.95, 40° 0.88. By arms: 1: 0.88, 2: 0.94, 3: 0.93, 4: 0.85. By contrast: 0.3: 0.84,
  0.6: 0.91, 1.2: 0.94.
- **Detection is flat across V1.** Synthetic matched-filter SNR falls from a median of 631 (Q1) to
  218 (Q5), 10th percentile 175. The scale length falls from 11.4 to 5.2 px. So at Stage 1 the V1
  axis mostly shrinks the galaxy and never makes it faint enough to lose the disc.

**Read (descriptive; no state is assigned by the pilot).**
- SpArcFiRe's "any arc" is a weak detector on discs. It fires on 75% of armless discs and on ~90%
  of spirals.
- So, like P2DFFT, it rarely abstains on a real disc. Its abstentions come mostly from tight (5°)
  and low-contrast arms.
- Common support (b) will therefore have few abstentions to work with (INSUFFICIENT is likely in
  Q1–Q5 separately; pooled per stage it has ~40–60).

### Recorded before the grid (user, 2026-09-25)

- **Finding: neither method reliably abstains on a disc.** SpArcFiRe's "any arc" fires on 15/20
  armless discs (75%), against ~90% of spirals and 0/20 noise images. P2DFFT answers every image.
  So pitch values for galaxies without arms are mostly measurements of nothing, in either method's
  catalogue. Published catalogues depend on pre-selecting spirals; the Hayes table's P_CS cut does
  this.
- **Exploratory, reported beside the hashed rule and labelled as such:** detection =
  `top2_chirality_agreement == 'agree'`, the reliability filter declared in Y4. Reported: its
  detection rate on spirals against armless discs, per quintile and stage. It enters no verdict.
  On the pilot: 'agree' on 94/260 spirals (36%), 0/20 armless and 0/20 noise. The rest were
  all-short 67, one-long 33, <2 arcs 24, disagree 16, no arcs 26.
- **If common support (b) reads INSUFFICIENT, that is an acceptable outcome.** The grid is not
  enlarged to rescue it.
- **Stage 1's flat detection across visibility is a property of the stage, not the harness.**
  At Stage 1, visibility is mostly size (disc scale 11 → 5 px); SNR stays ≥ ~175.
- **Grid, as proposed** (~5.5 h SpArcFiRe, ~20 min P2DFFT).
  - Per stage, 84 spirals per quintile (7 pitches × 4 arms × 3 contrasts; Stage 3: 28 each of
    flocculent, broken and barred) plus 20 noise + 20 armless per quintile: 620 images per stage.
  - SpArcFiRe default on all 1,860. The `run.sh` sensitivities (with and without amt 15) on a fixed
    half: every other grid point in each stage-quintile, plus half the nulls.
  - P2DFFT auto and oracle on all (deprojected for Stages 2–3).

### Grid run notes (2026-09-25)

- **The first grid run failed after Stage 1's SpArcFiRe runs had finished.** `run.sh` does no disk
  or bulge fit, so those columns hold `[]`, which the parser did not read as missing. Those columns
  are only read from the ellipse runs of Stages 2–3, so no measured value changes. Log:
  `out/bb/bb2/grid.failed_parse.log`.
- **One Stage-1 null was dropped and counted**, per the pilot's `dropped_h_nan` rule, which the grid
  path lacked: `s1_q5_armless12`.
  - Its donor, 1237667448343429137 (r = 20.18, petroRad_r 1.25″), is smaller than its PSF
    (FWHM 1.09″), so no disc matches its Petrosian radius, and the image rendered all-NaN.
  - SpArcFiRe could not convert it, and it was missing from all three runs.
  - Left in, it would have read as a SpArcFiRe abstention on a null. It is now kept out of both
    methods' inputs and listed in `grid.json` → `build.dropped_h_nan`.

### BB2 grid result (`out/bb/bb2/grid.json`, `grid_s{1,2,3}_rows.csv`)

**BOTH FAIL AT SDSS QUALITY at every stage, on both ladders.** **`bb3_reference`: neither method is
rank-faithful in any Stage-3 quintile**, so neither is usable in BB3.

| stage | SpArcFiRe det | SpArcFiRe RMS (°) | SpArcFiRe ρ | P2DFFT RMS (°) | P2DFFT ρ |
|---|---|---|---|---|---|
| 1 (face-on, clean) | 0.85–0.92 | **6.5 (Q1, WORKS)** → 13.9 | **0.78 (Q1, WORKS)** → 0.15 | 37–52 | 0.50 → −0.17 |
| 2 (inclined) | 0.82–0.99 | 9.6 → 15.6 | 0.61 → 0.29 | 47–54 | ≤ 0 |
| 3 (full) | 0.94–0.98 | 10.2 → 15.3 | 0.62 → 0.13 | 48–55 | ≤ 0.09 |

Each range runs Q1 → Q5. Q1 at Stage 1 is the only WORKS cell (SpArcFiRe, both ladders).

- **SpArcFiRe compresses loose arms, as declared.** The Stage-3 median at truth 40° is 20.9°.
  - Its rank fidelity falls with the quintile, and the fall is not flat: the Q5 synthetic SNR runs
    218 against Q1's 631.
  - The `run.sh` sensitivities detect too little to rank beyond Stage-1 Q1.
- **Nulls.**
  - SpArcFiRe's false detection on armless discs: 0.62 (Stage 1) and 0.66 (Stage 3).
  - P2DFFT answers 100% of nulls, as defined, with pitches of 1–90°.
- **Common support (a).** On SpArcFiRe's own detections, SpArcFiRe beats P2DFFT:
  - ΔRMS −32.6° [−35.2, −30.0] at Stage 1 and −37.7° at Stage 3;
  - Δρ +0.39 [0.24, 0.54].
- **EXPLORATORY `top2 == agree` rule** (fires on spirals / armless / noise):

  | quintile | Stage 1 | Stage 3 |
  |---|---|---|
  | Q1 | 0.49 / 0.05 / 0.00 | 0.35 / 0.00 / 0.15 |
  | Q5 | 0.17 / 0.00 / 0.00 | 0.20 / 0.05 / **0.70** |

  It fires on noise increasingly at Stage 3, so it is not a detection rule there.
- **Peng predictions.** All MATCH at Stage 3 except P5, which is NOT RESOLVED.
- **Dropped (h undefined):** 1 at Stage 1, 2 at Stage 2, 1 at Stage 3.

**OPEN — P2DFFT's failure is quantised, and I missed it at the pilot stop.**
- Its answers pile onto arctan(m/p) at the lowest radial frequencies: 90.0°, 75.96° (arctan 4),
  63.43° (arctan 2), 82.87° (arctan 8). The median answer for truth 5–15° is 76°.
- The pilot showed the same (median 53°); I reported only detection then.
- The clean-image generator plant passed P2DFFT (ρ ≥ 0.9, median error ≤ 3°), and the oracle-m
  sensitivity recovers ρ 0.84 at Stage-1 Q1. So under noise, **auto mode's pick of m and p lands on
  the low-frequency end**.
- Whether that is the method or the declared configuration (the inner radius does not exclude the
  bulge; the outer radius is 1.5 × Petrosian) is unresolved.
- The verdicts above stand as pre-registered. Separating method from configuration would be an
  exploratory rerun of P2DFFT on the Stage-1 images, with the bulge excluded and on noise-free
  copies.

**Conclusion (user, 2026-09-25).**
- The verdicts stand as pre-registered.
- **At SDSS depth, neither method ranks pitch well enough for BB3.**
- The winding referee stays DECaLS (AA2) plus votes. **BB3 as planned does not run.**
- Nothing is written about P2DFFT's performance until the exploratory Stage-1 rerun is in:
  (a) the bulge excluded by the inner radius; (b) noise-free copies; (c) both.

### P2DFFT exploratory rerun (Stage 1) — EXPLORATORY

(`artifacts/bb2_p2_explore.py`; `out/bb/bb2/p2_explore.{json,png,log}`.) **The pre-registered
verdicts stand. Nothing here changes them.**

**Set-up**
- The 619 Stage-1 images; `s1_q5_armless12` stays dropped.
- Scored on the 420 spirals. ρ is Spearman on truth 10–30°. "Pile" is the share of answers within
  ±0.05° of 90 / 82.87 / 75.96 / 63.43 / 45°.

**Starting radius**
- Set per image through P2DFFT's own mechanism: the BAR FITS keyword, which p2pa reads as the
  starting inner radius. The same `.h5` files are reused.
- *bulge*: the first radius, moving out, where the PSF-convolved disc reaches the PSF-convolved
  bulge, from truth. Spirals: median 4 px, 5–95% 1–6 px.
- *arm* (supplementary, added because the bulge crossing lies inside the arms' start): ceil(0.75 h),
  from truth. Median 6 px, 5–95% 4–11 px.
- Both are best cases, not blind recipes. 1 image is capped at end − 5.

**Other conditions**
- *clean*: the same spec and solved h with the noise draw removed. The 100 noise-only nulls drop out,
  since they would be blank.
- *base* reproduces `grid_s1_rows.csv` exactly in both modes (max |Δ| 0.0, no NaN mismatch).

| condition | mode | pile | 90° | 75.96° | 63.43° | median | RMS | ρ pooled | ρ Q1…Q5 |
|---|---|---|---|---|---|---|---|---|---|
| base | auto | 0.53 | 0.24 | 0.20 | 0.06 | 63.4 | 46.8 | −0.03 | 0.50 0.02 −0.10 −0.18 −0.17 |
| bulge | auto | 0.41 | 0.15 | 0.15 | 0.08 | 38.7 | 41.3 | 0.02 | 0.46 0.13 0.06 −0.21 −0.14 |
| clean | auto | 0.61 | 0.51 | 0.06 | 0.02 | 90.0 | 54.8 | −0.05 | 0.48 0.03 −0.05 −0.30 −0.20 |
| clean + bulge | auto | 0.50 | 0.42 | 0.04 | 0.03 | 53.1 | 50.3 | 0.02 | 0.51 0.17 0.07 −0.25 −0.18 |
| arm (supp.) | auto | 0.35 | 0.10 | 0.14 | 0.08 | 33.7 | 36.9 | 0.11 | 0.56 0.15 0.09 −0.10 0.03 |
| clean + arm (supp.) | auto | 0.46 | 0.39 | 0.03 | 0.01 | 38.7 | 48.4 | 0.07 | 0.51 0.15 0.16 −0.21 −0.10 |
| base | oracle | 0.17 | 0.06 | 0.05 | 0.03 | 23.2 | 26.3 | 0.53 | 0.84 0.50 0.56 0.57 0.26 |
| bulge | oracle | 0.12 | 0.03 | 0.03 | 0.03 | 21.8 | 22.2 | 0.54 | 0.83 0.59 0.55 0.50 0.29 |
| clean | oracle | 0.20 | 0.10 | 0.06 | 0.03 | 25.2 | 30.7 | 0.61 | 0.89 0.71 0.60 0.49 0.39 |
| clean + bulge | oracle | 0.15 | 0.06 | 0.04 | 0.03 | 24.8 | 25.9 | 0.67 | 0.89 0.82 0.66 0.52 0.49 |
| arm (supp.) | oracle | 0.10 | 0.01 | 0.03 | 0.03 | 21.8 | 20.2 | 0.55 | 0.82 0.57 0.53 0.58 0.31 |
| clean + arm (supp.) | oracle | 0.13 | 0.04 | 0.03 | 0.01 | 24.6 | 22.7 | 0.69 | 0.87 0.78 0.68 0.56 0.57 |

(82.87° ≤ 0.007 and 45° ≤ 0.04 everywhere.)

**Median answer by true pitch 5 / 10 / 15 / 20 / 25 / 30 / 40°**

| condition | mode | 5° | 10° | 15° | 20° | 25° | 30° | 40° |
|---|---|---|---|---|---|---|---|---|
| base | auto | 76 | 76 | 76 | 22 | 51 | 32 | 45 |
| clean + bulge | auto | 90 | 90 | 33 | 21 | 26 | 30 | 41 |
| every condition | oracle | 5–12 | 10 | 15 | 20 | 25 | 30 | 39–41 |

**Where the pile-up sits: it is m = 1**
- In auto mode, p2pa picks m = 1 for 227–294 of the 420 spirals, against 105 true one-armed spirals.
- 142–254 of those m = 1 picks land on a pile value.
- Of the picks at m ≥ 2, at most 5 do, in any condition.
- Removing noise makes it **worse**, not better: 90° goes from 24% to 51%. Every noise-free armless
  disc answers 90° (99/99).
- So a deterministic m = 1, zero-frequency component in the smooth disc outweighs the arms whenever
  their contrast is modest.
- Its origin is not isolated here. Candidates are the centre convention or the log-polar m = 1 term
  of an off-centre profile. Noise dilutes it rather than causing it.

**Conclusion (exploratory)**
- **The pile-up does not go away under any condition in auto mode.** Excluding the bulge or the
  inner disc trims it: 53% → 41% (bulge) and → 35% (arm start). Noise-free copies raise it to
  46–61%. Auto-mode ρ stays ≤ 0.11 everywhere.
- With the arm count supplied, the pile-up is 10–20%, the median tracks truth at 10–40°, and ρ rises
  from 0.53 to 0.67–0.69 once noise is removed. Even then it stays under 0.70 pooled; only Q1–Q2
  clear it.
- **This points to the configuration's mode selection, not the Fourier pitch measurement.** p2pa's
  amplitude-chosen m is captured by a spurious m = 1 low-frequency term. Given the right m, the method
  measures pitch; the inner radius and noise are second-order.
- The pre-registered result is unchanged. Auto mode is what the declared configuration ran, and it
  fails.

**Recorded (user, 2026-09-25; EXPLORATORY).**
- In automatic mode P2DFFT picks one arm for most spirals, from a low-frequency signal of the smooth
  disc. Neither noise nor the bulge causes the pile-up.
- With the true arm count supplied, it tracks pitch. It still stays below ρ 0.70, even noise-free
  with the bulge excluded using true parameters (the best case).
- The pre-registered BB2 verdicts are unchanged.
