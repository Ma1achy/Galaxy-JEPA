# Spec — testing strategy

*Status: design proposal for sign-off. Expands `docs/architecture.md` → "Validation
tiers" (cross-cutting). The `core/` tests already in Track B
(`tests/test_config.py`, `test_gates.py`, `test_encoder.py`) are the **first
instance** of this strategy, not separate from it. British English.*

This project needs **two kinds of test**, and conflating them is the failure mode:

1. **Is the code correct?** — the conventional pyramid (unit → integration → e2e).
2. **Is the science valid?** — invariant / property tests + the validity gates
   (`docs/spec/gates.md`). This is the **higher-value tier** and gets its own section
   below; it is *never* folded into "unit tests".

The whole design philosophy is that the experimental guardrails are **structural**, not
discipline-at-the-keyboard. A test is how a structural guarantee stays true as the code
changes. The corollary, and the rule this doc exists to enforce:

> **An untested invariant is just a comment.** Every hard invariant in
> `docs/architecture.md` must have a live test (or a registered, skipped stub naming the
> TODO) — the frozen encoder, the β=0⇒I-JEPA generalisation, the
> format+stretch+normalisation parity, the uncertainty-geometry label firewall.

---

## 1. The tiers

### 1.1 Unit — fast, pure, no GPU, no network (the bulk)

The `core/` framework tests are the live examples (Track B): config round-trip,
`ClassRef` / nested-config serialisation, config-hash stability, `*args`/`**kwargs`
rejection at class-definition time, run-stamp git SHA; gate one-comparator enforcement,
empty-composite rejection, the verdict tree, verdict-independent-of-run-status;
`assert_frozen`.

As components land, their **pure logic** joins this tier — each tested in isolation, no
model, no GPU: bbox computation from `petroRad` + pixel scale, Petrosian→pixel scaling,
the L2 logistic fit, `selectivity = real − control`, the consensus-extremes split
(train v>0.8 / v<0.2, test on the held-out 0.2–0.8 middle).

### 1.2 Invariant / property — the science-protecting tier

Its own tier because a passing unit test says *the function does what its author
intended*; an invariant test says *the methodology cannot be silently violated*. These
use **Hypothesis** to generate random inputs and assert the invariant holds **across
all** of them — not on a hand-picked example that happens to pass.

| Invariant | What the test asserts | Status |
|---|---|---|
| β=0 ⇒ I-JEPA | Over **random** masking configs, the β=0 block-sampling statistics are identical to standard I-JEPA (strict generalisation). | stub: `test_masking_beta_zero_is_ijepa` |
| Preprocessing parity | Format + stretch + normalisation are byte-identical across the pretraining corpus, the probing corpus, and every baseline. | stub: `test_normalisation_parity` |
| Uncertainty-geometry label firewall | The graded vote fraction **cannot** enter the fit set — asserted **two ways**: (a) by signature (the fit path has no parameter that accepts it), **and** (b) by feeding **poisoned** graded labels and asserting they are provably unused. | lands with `probing/` |
| Frozen encoder on the probing path | Run a probe end-to-end, then assert **no encoder parameter accumulated a gradient** (not merely `requires_grad=False` — that it stayed zero). | lands with `probing/` |
| Negative control must fail | A shuffled-label / random-label probe scores **at chance** — if a negative control "passes", the whole measurement is suspect. | lands with `probing/` |

Hypothesis is the tool of record here: generate random masking configs (β, τ, φ), random
gate trees, random metric dicts, and assert the property over the generated space.
`bbox_degradation` is a **property test, not a gate** (`docs/spec/gates.md` §3) — recorded
here so the invariant is not double-booked across two mechanisms. See
`docs/spec/validation.md` for the run-time validity rules that complement these.

### 1.3 Integration — component seams, tiny fixtures, CPU

Exercises the joins between components on the committed fixture (§3), no GPU:

- **Data pipeline end-to-end** on a fixture: FITS stamps → stretch → normalise →
  tensors, asserting the stretch-sanity property (faint arms survive; the sky-noise
  floor stays controlled).
- **The disk freeze boundary**: an objective writes a provenance-stamped checkpoint →
  `probing/` reloads it as a **frozen** `Encoder` and `assert_frozen` passes
  (`docs/spec/objectives.md` §3).
- **A dummy `Encoder` through the full probe ladder** — proves the cross-objective
  machinery is genuinely encoder-agnostic **before** three real encoders exist (the
  cheapest possible guard on the D12 contract).
- **A Tier-1 validator rejecting a controls-missing config** (`T1.controls-required`,
  `docs/spec/validation.md`).

### 1.4 E2E — one tiny real run

A miniature of the vertical slice: tiny synthetic corpus → a few hundred JEPA steps →
freeze → one linear probe → assert a **stamped figure artefact** is produced and the
reported number is **finite**. This tests that **the pipeline runs and the provenance
chain is intact end-to-end** — *not* that the science works; the run is far too small to
learn anything. Slow / GPU, so it is **scheduled / merge-to-main**, never on every push.

---

## 2. CI pipeline — staged by cost, fail-fast

| Stage | Trigger | Runner | Selection | Budget |
|---|---|---|---|---|
| Lint + type-check | every push / PR | CPU | `ruff check` + `ruff format --check` + `mypy` | seconds |
| **Unit + invariant** | every push | CPU | default `pytest` (unmarked) | **~2 min wall** on the test step (install excluded via cache) — *the gate* |
| Integration | PR | CPU | `pytest -m integration` (fixture-backed) | a few minutes |
| E2E | nightly / merge-to-main | GPU (deferred) | `pytest -m e2e` — or **`make e2e`** locally | minutes |

- **Lint first** (cheapest, fails in seconds) so a formatting or type error never burns
  a test run.
- The **unit + invariant** stage is the gate and must stay fast, or it stops being run
  locally. Install time is kept out of the budget by caching `uv sync`; torch is pulled
  from the **CPU wheel index** in CI.
- **Merge-blocking rule:** the **invariant / property tests are a merge-blocking gate**.
  A PR that breaks the frozen-encoder guarantee, the preprocessing parity, or the
  label firewall **cannot merge** — the CI mirror of the structural guardrails that make
  those mistakes impossible to commit by hand. This is the single most important line in
  this document.
- **Coverage is reported, not chased as a number.** A coverage target rewards trivial
  getters and is blind to whether the *invariants* are covered. Instead, invariant
  coverage is tracked **qualitatively**: a checklist of which structural guarantee has a
  live test (the table in §1.2 is that checklist).

GPU CI is **deferred** (fork below): the e2e job exists in the workflow but is
`workflow_dispatch`-only until a GPU runner is wired; until then e2e is run locally via
`make e2e`. The `Makefile` targets mirror the CI selection one-to-one, so "green
locally" means "green in CI".

---

## 3. ML-specific policies

- **Determinism.** Seed everything; any test asserting a numerical output uses a **fixed
  seed + an explicit tolerance**, never bitwise equality where it isn't warranted.
  Document the torch **GPU non-determinism** policy: which ops are not bit-reproducible
  (e.g. atomics in some scatter/reduction and conv backward kernels) and the tolerance
  used for them. This shares the seeding discipline with the config-hash / run-stamp
  reproducibility story (`docs/spec/config.md`) — the same seed that makes a run
  reproducible makes its test reproducible.
- **No network in CI.** A test that hits CasJobs / SkyServer is flaky by definition.
  Network pulls are mocked or fixture-backed through the `DataSource` abstraction
  (`docs/spec/data.md`) — a **fixture source** stands in for the live one.
- **Tiny committed fixtures.** A **dozen-galaxy** dataset that exercises the whole
  pipeline in seconds, so everything integration-and-up has something to run against.
  Build it **early** — it is a dependency of the integration tier, not an afterthought.
  - *Where:* `tests/fixtures/`.
  - *How:* a **committed seeded generator script** (synthetic FITS stamps with planted
    faint structure for the stretch-sanity test) **plus a dozen cached real cutouts**
    (so the pipeline is also exercised against genuine survey data, offline). The
    generator is seeded and deterministic so the fixture is reproducible.
- **Markers, ruthlessly.** GPU and slow tests are guarded behind pytest markers
  (`integration`, `e2e`, `gpu`, `slow`) registered in `pyproject.toml`. The default
  selection (unmarked) is the fast suite, and it **must stay fast** — the moment the
  local fast suite takes minutes, it stops being run, and the gate's value evaporates.

---

## 4. Forks

| Fork | Options | Recommendation | Status |
|---|---|---|---|
| Hypothesis as a dependency | yes / hand-picked cases only | **Yes** — the property tier needs generated configs / gate trees / metric dicts; an invariant proven on one example is not proven. Declared now, used as the invariant tests land. | proposed (recommendation stands) |
| Synthetic fixture | generator script / cached real cutouts / both | **Both** — a committed **seeded generator** (synthetic, with planted faint structure) **+ a dozen cached real cutouts**, under `tests/fixtures/`. | proposed (recommendation stands) |
| GPU CI now vs defer | wire GPU CI now / defer, e2e local | **Defer GPU CI**; keep e2e runnable via `make e2e`; the nightly GPU workflow is documented but not wired. | proposed (recommendation stands) |
| Fast-suite time budget | strict number / soft target | **~2 min wall** on the unit+invariant step (install excluded via cache). | proposed (recommendation stands) |
| Type checker | mypy / ty / pyright / none | **mypy** — established and CPU-cheap for the lint stage. `ty` (Astral, matches the uv + ruff toolchain) is the forward option; not switched now. | **decided** (mypy) |
