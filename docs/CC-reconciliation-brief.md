# CC brief — reconcile today's design work into the repo

**Context for CC:** A long design session (browser-Claude) produced genuine new design
work — a *confound taxonomy*, an *inclination conditioning axis*, *conditional-population
probing*, and a *two-scheme feature experiment* — plus a consolidated LaTeX design spec.
None of it is in the repo yet, and the planning docs (`DECISIONS.md`, `PROJECT_PLAN.md`,
`TODO.md`) predate it. This brief lands that work: new decision records, plan updates, and
the spec as the canonical design doc. **The science spine is unchanged** — this is *additive*
(a new conditioning dimension + interpretive layer), not a redesign. British English.

**Malachy has signed off the open decisions and the two new ones (D13, D14) below.** Nothing
here needs a fresh call from him — this is execution. Where a step edits an existing doc,
exact anchors are given.

---

## Part 0 — the ONE file Malachy uploads (do not skip)

The LaTeX design spec was authored outside the repo (in the browser session's sandbox), so CC
cannot generate it — **Malachy will place two files in the repo root before you run this brief:**
- `galaxy-jepa-spec.tex`
- `galaxy-jepa-spec.pdf`

If they are **not** present when you start, stop and ask him to add them. Everything else in this
brief you do yourself.

---

## Part 1 — tick the decisions Malachy signed off (`DECISIONS.md`)

He accepted every open recommendation. Tick these boxes (change `[ ]` → `[x]`, and update the
header status from *needs your call* / *proposed* to *decided (signed off)*):

- **D1** (line ~16) — **PyTorch**. Tick.
- **D3** (line ~57) — **uv + devcontainer + pytest + pre-commit (ruff), Python 3.11**. Tick.
- **D4** (line ~68) — **From-scratch**. Tick.
- **D5** (line ~81) — **Bounding-box-biased multi-block** (β=0 = I-JEPA control). Tick.
- **D6** (line ~93) — **Decouple corpora** (large unlabelled SDSS pretrain; probe on GZ2). Tick.
  This greenlights the large pretraining pull as a committed critical-path item.
- **D8** (line ~139) — **Reuse v1 mean+2σ filter** (kept separate from the uncertainty-geometry
  consensus-extremes split). Tick.
- **D12 sub** — confirm **MoCo** as the contrastive baseline and **MAE = Wu & Walmsley recipe
  reproduced on the SDSS corpus** (Euclid MAE reference-only). Already written in D12; just mark
  the sub-decision resolved.

(D2, D7, D9, D10, D11, D12-main were already ticked. After this, D1–D12 are all resolved.)

---

## Part 2 — add TWO new decision records (`DECISIONS.md`, append after D12)

These are the genuinely-new design decisions from today, of the same weight as D1–D12. Append
them verbatim (adjust cross-reference doc names if they differ):

### D13 — Confound taxonomy + inclination conditioning — *decided (signed off; Framing-B mechanism)*

- [x] **Human confusion has distinct *physical* causes, diagnosed with the label-free encoder.**
- [x] **Inclination is a first-class conditioning axis; proxy = axis ratio (b/a).**

The label-free encoder never sees votes, so per confused feature we can ask whether the confusion
is in the **data** (encoder also confused → genuine information limit) or the **humans** (info in
the pixels; encoder separates what people cannot). Grounded in v1's own correlation analysis,
confusion splits three ways, each with a distinct fingerprint across the
**(inclination × imaging-depth)** plane:

1. **Projection** (viewing-angle information loss) — e.g. edge-on disk ↔ cigar elliptical
   (v1: Edge-on × Cigar = +0.83). Angle-dependent, imaging-depth-**invariant**.
2. **Resolution *or* semantic** — arm-count, winding, bulge-shape (near-zero v1 off-diagonals).
   The imaging-depth axis distinguishes them: resolution **improves** with deeper imaging;
   semantic does not.
3. **Genuine co-occurrence vs artefactual correlation** *(HYPOTHESIS — unconfirmed)* — bar +
   spiral structure (v1: Bar × 2-arms = +0.56). The method's hard case: entanglement here may be
   **correct physics**, not a representation limit. Adjudicated by the eigen-triangulation's
   causal cross-check (conditional-recoverability under matching).

**Inclination proxy = axis ratio (b/a)** — an *independent photometric* measurement (SDSS
pipeline, from the pixels), so conditioning on it to study *vote*-confusion is **not circular**
(using the T01/T07 votes as the proxy *would* be). This is a **new capability on the existing
probe** (probe within inclination bins / with b/a as covariate) — it does **not** revise the
locked probing sub-systems.

**Status of the taxonomy for the paper:** it is the *mechanism for Framing-B's earned payoff*,
held as interpretive lens **pending results** — NOT the paper's spine (which stays Framing-A:
method + ladder + controls). See the design spec (`docs/galaxy-jepa-spec.pdf`, §Framing,
§Confound).

**Data-layer consequence (see Part 3):** b/a (`expAB_r`, `deVAB_r`) is SDSS photometry, **not** a
GZ2 vote column and **not** in the current metadata — it is a **new pull requirement** for the
probe corpus, distinct from both the masking pull (petroRad + arcsec/pixel) and the nuisance join
(z/mag/radius/SNR/PSF). Cheap: a `PhotoObj` join on `objID`, no image re-cut.

### D14 — Feature-set = a two-scheme experiment, conditional-population probing — *decided (signed off)*

- [x] **The feature set is an *experiment over schemes*, not a fixed choice.**
- [x] **Each feature probed within its conditional population — as a *comparison*, not a hard mask.**

**Conditional-population probing.** The GZ2 tree is conditional: a feature is only well-defined
within the population that reaches its question (boxy-bulge is meaningless for a no-bulge galaxy —
v1's "Q4 can't be yes and Q7 can't be no for the same galaxy"). **But do not hard-mask the
"incoherent" galaxies away** — a no-bulge galaxy carrying boxy-bulge votes is a *measurement of
human disagreement*, and masking it pre-imposes the tree's logic before testing whether it holds
(circular). Instead: probe each feature across **different** population definitions (full vs
consensus-conditional) and **compare**; study the off-population galaxies as their own object
(concentrated = systematic confusion = finding; scattered = noise). Reuse `data/splits.py`
firewall/masking machinery. Uncertain-upstream gating (bulge-presence is itself a contested vote)
→ consensus-extreme gating where conditioning is applied; the gate threshold is a **per-run knob**.

**The two schemes (the experiment).**
- **Scheme 1 — full tree (37 answers, per-bucket)**, each in its conditional population. Honest
  baseline; expected weak on the v1-confused features (echoes v1 = a finding). BY family ≈ 37.
  **Power confound:** per-bucket deep-feature weakness is confounded between genuine-absence and
  split-sample (~9,870 spirals ÷ 6 arm-buckets ≈ 1,600 each) — Scheme 1 alone can't distinguish;
  do **not** read per-bucket weakness as "absent."
- **Scheme 2 — reduced/smart (diss-style)**: graded questions → one graded axis each; binary
  well-posed → one binary feature; odd-subtypes exploratory. BY family ≈ 10–13. Also a **power
  diagnostic** (arm-count-as-axis on full ~9,870 recoverable but per-bucket isn't ⇒ the failure
  was power).
- **The comparison is a result.** Same ladder both ways ⇒ reduction cosmetic; differ ⇒ reducing
  changes what's expressible ⇒ a real taxonomy result. **Order: full first** (transparent).

Implementation: **schemes are configs**, one harness (reconfigure, don't rebuild); BY family
count is per-config. See `docs/galaxy-jepa-spec.pdf`, §Feature-scheme experiment.

**Open sub-question (flag, do not resolve): graded-axis existence test — AUC vs correlation.**
A graded axis may get a *correlation* existence test (Spearman/permutation) rather than AUC — but
that is the *same measurement* as the uncertainty geometry for that feature, so they may collapse.
Binary features keep AUC. Decide before wiring Scheme 2.

---

## Part 3 — update the plan to match (`PROJECT_PLAN.md` + `TODO.md`)

### 3a. `TODO.md` — Epic B (Data layer, P2): add the axis-ratio pull

After the existing nuisance-join task (line ~30-ish, the "CasJobs / SkyServer metadata join" item),
add a **new task**:

```
- [ ] (P1) **Axis-ratio pull for inclination conditioning** — `expAB_r` + `deVAB_r` from SDSS
  `PhotoObj`, join on `objID` for the GZ2 **probe** corpus. Independent photometric inclination
  proxy (non-circular). Distinct from both the masking pull (petroRad + arcsec/pixel) and the
  nuisance join (z/mag/radius/SNR/PSF). Cheap: catalogue-only, no image re-cut. *(D13)*
```

### 3b. `TODO.md` — Epic F (Probing harness, P6): add the conditional-population + two-scheme tasks

Add two new tasks to Epic F:

```
- [ ] (P1) **Conditional-population probing (as a comparison)** — probe each feature across
  population definitions (full vs consensus-conditional); do NOT hard-mask; study off-population
  galaxies as a finding. Reuse `data/splits.py`. *(D14)*
- [ ] (P1) **Two-scheme feature experiment** — Scheme 1 (full-37 per-bucket) and Scheme 2
  (reduced graded-axis) as configs on one harness; per-config BY family; full-first order; the
  comparison is a result. Watch the Scheme-1 power confound. *(D14)*
```

And add to **Epic R (Rung controls / ablations)** the new axis:

```
- [ ] (P1) **Inclination conditioning** — recoverability/entanglement as a function of axis ratio
  (b/a); the confound-fingerprint deliverable. New axis on the existing probe, not a redesign.
  *(D13)*
```

### 3c. `PROJECT_PLAN.md` — phase table

- **P2 row** (line ~53): append to the milestone cell — `; axis-ratio (expAB_r/deVAB_r) join for
  inclination conditioning (D13)`.
- **P6 row** (line ~57): append to the milestone cell — `; conditional-population probing
  (comparison, not mask); two-scheme feature experiment (Scheme 1 full-37 / Scheme 2 reduced);
  inclination conditioning (D13, D14)`.
- Add a short paragraph under the phase table noting the **confound taxonomy** as the Framing-B
  interpretive layer (held pending results, not the spine), pointing to `docs/galaxy-jepa-spec.pdf`.

### 3d. Statistics — note that the five flagged decisions are now GROUNDED

The five `# FLAGGED` statistical decisions in code (`config.py` effect-floor, `nulls.py`
existence-p + family-significance, `mlp.py` selectivity, `entanglement.py` MP-null,
`uncertainty.py` permutation-p) were **owned in the design session**. Record in `TODO.md` Epic F
(or a stats note) that the decisions are settled — **multiplicity = Benjamini–Yekutieli (FDR)**,
effect-floor = a single pre-registered minimum AUC (Malachy's value TBD), permutation ≥10,000
two-tailed, MP-edge for the actual matrix shape, existence = real-vs-null at the floor. The
*structure* was already in `docs/probing-harness-design.md`; this just marks the choices locked.
(Porting the grounded values into the flagged code is a **separate** task — see Part 5 — not this
brief.)

---

## Part 4 — install the design spec as canonical (`docs/`)

1. `git mv galaxy-jepa-spec.tex galaxy-jepa-spec.pdf docs/` (from repo root, after Malachy adds
   them). If `git mv` complains they're untracked, `mv` then `git add docs/galaxy-jepa-spec.*`.
2. Add a line to `docs/`'s index / the README doc list: *"`galaxy-jepa-spec.pdf` — consolidated
   design spec (science + paper skeleton); the reconciled design source of truth as of this
   commit."*
3. In `galaxy-jepa-scratchpad.md` **and** `PROJECT_PLAN.md`, add a one-line pointer near the top:
   *"The consolidated design spec (`docs/galaxy-jepa-spec.pdf`) is the current design source of
   truth; this doc predates the confound/inclination/feature-scheme work (see D13, D14)."* Do NOT
   rewrite the scratchpad — just point to the spec.

---

## Part 5 — do NOT do these here (they are separate, later tasks)

- **Porting the grounded stats values into the flagged code** — separate task/session.
- **The large pretraining pull** (D6) — separate compute task; the throttle driver is ready.
- **The 40k corpus verification/ingest** — separate (blocker-free) task.
- **Wiring the conditional-population / two-scheme code** — separate build task, after the design
  is committed here.

This brief is **documentation reconciliation only**: decisions, plan, spec-as-canonical. It leaves
the repo's *plan* current with today's design, so subsequent build tasks execute against an
up-to-date plan.

---

## Commit

One commit (or two: "decisions + plan" then "spec"), message e.g.:
`docs: reconcile confound taxonomy, inclination axis, feature-scheme experiment (D13/D14) + land design spec`
Tree clean, branch in sync. Report: which files changed, the D13/D14 records, and confirm the spec
is in `docs/`.
