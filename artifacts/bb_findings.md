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
