# Spec — train / val / test split policy (leak-impossible by construction)

*Status: design proposal for sign-off. Expands `docs/architecture.md` → D6 (corpus
decoupling) and the uncertainty-geometry protocol. The structural guards land in
`src/galaxy_jepa/data/splits.py`; the orchestration (the actual pulls, ratios,
stratified assignment) is **not** built yet. British English.*

The split policy exists to protect one thing: **a probe number you can believe**. Every
way a split can lie is a way the headline result becomes uninterpretable —
the encoder having seen a test galaxy, the uncertainty axis having been trained on the
gradient it then "recovers", a non-reproducible partition that drifts between runs. So the
governing principle, the same one the testing and masking specs carry:

> **A guarantee that depends on remembering to do the right thing is not a guarantee.**
> Every leak this document names is made **impossible in code** — enforced by a guard with
> a loud failure and a merge-blocking invariant test — not left to keyboard discipline.

The **forks** (§7) are now **settled** (collision → probing; 70/15/15; fixed monitor; the
uncertainty fit/test partition composes *inside* the three-way split, not as a fourth
top-level hold-out). The structural guards are built; what remains is the orchestration
(the actual pulls, stratified assignment, per-split manifests), deferred to the corpus pull.

---

## 1. The shape of the problem — two corpora, split differently (D6)

D6 decouples the corpora: pretraining runs on a **large unlabelled SDSS** sample
(≫250k), probing on the **GZ2-labelled** set (~250k). They are not one dataset sliced —
they are two pulls with two jobs, and they are split **separately**:

- **Pretraining corpus** → `pretrain-train` + a small **`pretrain-monitor`** held-out
  slice. The monitor is *only* for the collapse / loss-curve watch (`docs/spec/validation.md`);
  it never carries labels and is never probed. No val/test, because pretraining is
  label-free — there is no score to tune against, only the SSL objective.
- **Probing corpus** → `probe-train` / `probe-val` / `probe-test`. This is where the
  science is measured, so it carries the real three-way split, the stratification, and
  the uncertainty-geometry partition (§3).

The two never share a split axis, but they **do** share galaxies — the same physical
object can be pulled into both (the unlabelled pull is not GZ2-aware). That overlap is the
central leak this spec closes (§2).

---

## 2. The leak guard — cross-corpus dedup by `objID`

**The leak.** If a galaxy is in both corpora, the frozen encoder saw it during
pretraining and is then asked to *generalise* to it at probe time. The probe number stops
measuring "does the representation transfer" and starts measuring "did the encoder
memorise this galaxy" — and D6's whole premise (the representation transfers rather than
memorises) is silently void. It is silent because nothing errors: you simply get an
optimistic probe score.

**The guard (structural).** Dedup on **SDSS `objID`** — the stable 64-bit cross-catalogue
identity, never ra/dec floats (a 0.1″ astrometric jitter would defeat a coordinate
match). Two operations, in `data/splits.py`:

- `exclude_probe_from_pretrain(pretrain_ids, probe_ids)` → the pretraining manifest with
  every probing galaxy removed. This **is** the collision resolution (§7 fork): a shared
  galaxy belongs to probing and is dropped from pretraining.
- `assert_no_cross_corpus_leak(pretrain_ids, probe_ids)` → raises `LeakError` if the
  intersection is non-empty. This is the **post-condition**, run on the manifests a run
  actually trains on, and mirrored as a **merge-blocking invariant test**
  (`tests/test_data_splits.py`, `@pytest.mark.invariant`). A PR that lets a probe galaxy
  into pretraining cannot merge.

`objID`s are coerced through `to_object_ids` (int/str/float → `int`, `bool` and
non-integral floats rejected) so the comparison is always identity-on-`int`, never a
silent `"123" != 123` type mismatch that would pass a real overlap through.

The guard feeds the existing `manifest_hash` (`data/manifest.py`): the deduped pretraining
ID set is what gets hashed into `data_snapshot`, so the snapshot records the *leak-free*
corpus, structurally.

---

## 3. The uncertainty-geometry partition — extremes fit, the middle is tested

Within the **probe** corpus, the headline result (uncertainty geometry) imposes a second
partition that is **not** the train/val/test split — it is orthogonal to it and protects a
*different* leak: circularity.

- **Fit set = high-consensus extremes only** (`v >= 0.8` or `v <= 0.2`, binary). The
  concept axis is estimated here.
- **Test set = the ambiguous middle** (`0.2 < v < 0.8`), held out of estimation entirely,
  projected afterwards to ask whether margin distance ranks the human vote fraction.

The leak it closes: if any middle galaxy enters the fit, the axis is trained on the very
gradient it is later asked to "recover" — a tautology, and the result is worthless. So the
firewall is structural:

- `partition_uncertainty(vote_fractions, low=0.2, high=0.8)` → `(fit_extremes, test_middle)`.
- `assert_uncertainty_firewall(fit_vote_fractions, low=0.2, high=0.8)` → raises
  `LeakError` if **any** fit-set vote fraction lies in the open middle. Run before fitting;
  also an `@pytest.mark.invariant` test.

This is the code form of the scratchpad's "non-circular protocol". The `0.2`/`0.8` bounds
match the scratchpad throughout; they are a fork only in the sense that the threshold is a
science choice (§7), but the *firewall itself* is not negotiable.

> **Interaction with the three-way split.** The fit/test partition lives **inside**
> `probe-train`+`probe-val` (extremes) and `probe-test` (the middle is a test-time
> projection). The extremes still obey the train/val/test boundaries — i.e. the axis is
> fitted on `probe-train` extremes, tuned on `probe-val` extremes, and the
> uncertainty-geometry read-out is the `probe-test` middle. The firewall and the
> three-way split compose; neither overrides the other.

---

## 4. Stratification — the split must be representative where it is thin

A uniform random split is fine for the bulk but starves the **rare, decisive** cells —
and those are exactly where the science lives. Stratify the probe `train/val/test` draw so
each split holds a representative share of:

- **Hard / confused features** — the bulge-shape, winding-tightness, arm-count questions
  the project exists to diagnose (the v1 confusion set). A split that puts most barred or
  tight-winding galaxies in `train` leaves `test` unable to measure them.
- **Large extended galaxies** — the morphology-rich, `petroRad`-large objects that are
  also the ones the 64 px stamp clips (`data/data.md` §3 stamp-size warning). They are
  rare and high-value; an unstratified split can leave a handful in `test`.
- **The uncertainty extremes vs middle** (§3) — so each split has enough confident
  examples to fit a stable axis *and* enough ambiguous ones to test it.

Stratification is applied as a constraint on the deterministic assignment (§5): galaxies
are bucketed by stratum, then each bucket is split by the same hashed coordinate, so every
stratum hits the target ratio. The exact variable list is a fork (§7).

---

## 5. Determinism — a split reproducible from the seed, with nothing stored to drift

The partition must be **reconstructible from the seed alone** — no split file to commit,
lose, or let drift out of sync with the data. The primitive (`data/splits.py`):

- `assignment_unit(objID, seed, salt="")` → a stable `[0, 1)` coordinate from a
  **SHA-256** of `(seed, salt, objID)` — *not* Python's `hash()`, which is salted per
  process and would give a different split every run. A galaxy's home is a pure function
  of its `objID` and the seed.
- `salt` namespaces independent partitions (`"probe"` vs `"pretrain-monitor"`) so they do
  not correlate — the monitor slice is not accidentally a shifted copy of the probe split.

The seed and ratios are **config** (`docs/spec/config.md`): they enter the config hash and
run-stamp, so the split is a stamped, reproducible decision. Combined with `manifest_hash`
over the deduped ID set, a run's exact split is fully determined by `(manifest, seed,
ratios, strata)` — all stamped, none remembered.

> **Scope.** `data/splits.py` ships the **guards and the deterministic primitive** only
> (dedup, firewall, `assignment_unit`). The orchestrator that pulls both corpora, applies
> the stratified ratios, and writes the per-split manifests is **deliberately not built
> here** — it lands with the corpus pull, on top of these guarantees.

---

## 6. What is enforced in code today vs proposed

| Guarantee | Mechanism | Status |
|---|---|---|
| Cross-corpus dedup (no probe galaxy in pretraining) | `assert_no_cross_corpus_leak` + `exclude_probe_from_pretrain`, invariant test | **built now** |
| Uncertainty-geometry firewall (no middle in fit) | `assert_uncertainty_firewall` + `partition_uncertainty`, invariant test | **built now** |
| Deterministic, seed-reproducible assignment | `assignment_unit` (SHA-256), invariant test | **built now** |
| Two-corpus split structure (D6) | §1 — orchestrator over the guards | proposed (design) |
| Stratified ratios | §4 — constraint on `assignment_unit` | proposed (design) |

---

## 7. Forks — flagged for sign-off

| Fork | Options | Recommendation | Status |
|---|---|---|---|
| **Collision resolution** | shared galaxy → probing (exclude from pretrain) / → pretraining (exclude from probe) / drop from both | **→ probing, excluded from pretraining.** The probe set is the scarce, labelled, science-critical corpus; the unlabelled pretraining corpus is ≫250k and losing a few thousand galaxies to dedup costs nothing, whereas shrinking the labelled probe set costs measurement power. Dropping from both needlessly wastes a labelled galaxy. | **decided** |
| **train/val/test ratios** | 70/15/15 / 80/10/10 / 60/20/20 | **70/15/15.** ~250k probe galaxies → ~37.5k each for val and test, ample for stratified per-feature AUC with controls; 80/10/10 thins the rare hard-feature cells in test, 60/20/20 needlessly starves the fit. | **decided** |
| **Stratification variables** | hard/confused features only / + large extended galaxies / + redshift–size–brightness nuisance cells | **hard features + large extended galaxies + the uncertainty extremes/middle balance** (§4). Adding the full nuisance grid (z × size × brightness) multiplies cells until each is too thin to stratify — defer it to matched-evaluation re-tests. | proposed (recommendation stands) |
| **Uncertainty thresholds** | 0.8 / 0.2 / tighter (0.85 / 0.15) / looser | **0.8 / 0.2** (scratchpad-wide). Tighter buys axis purity at the cost of fit-set size; revisit only if the extremes turn out noisy. | proposed (recommendation stands) |
| **Pretrain-monitor split** | fixed held-out slice / rotating per epoch | **Fixed held-out slice.** A stable monitor makes the collapse / loss curve comparable across checkpoints and runs; a rotating monitor reshuffles the baseline every epoch and muddies exactly the trend the monitor exists to read. Costs a small fixed slice of pretraining data — negligible at ≫250k. | **decided** |
