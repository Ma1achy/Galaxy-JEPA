# Spec — controls as gates

*Status: design proposal for sign-off. Expands `docs/architecture.md` → "Controls as
gates". Primitives implemented (Track B) in `src/galaxy_jepa/core/gates.py`; the
concrete control gates are built in `probing/` (next phase). British English.*

A `Gate` is a first-class, composable object. Each feature's place on the nameability
ladder is a gate tree carried **with** its result. The load-bearing rule:

> **The verdict is separate from run status.** A feature failing its controls is a
> *finding* (`passed=False`), recorded — not an error that aborts the run. Evaluation
> only raises on a *structural* mistake (a metric the run never produced), because that
> is a bug, not a finding.

---

## 1. The primitives (core)

- **`MetricGate(metric, <comparator>=threshold)`** — exactly one of
  `gt`/`gte`/`lt`/`lte` (enforced in `__post_init__`; zero or two raise). No float `eq`
  — exact equality on a float metric is a latent bug.
- **`AllGate` / `AnyGate`** — composites that **reject empty children** (a degenerate
  logic error, not a silent no-op). Built with the `gate.all(...)` / `gate.any(...)`
  factories.
- **`GateResult`** — the verdict tree: `label`, `passed`, optional `detail`, and
  `children`. `render()` produces an indented `[PASS]/[FAIL]` tree.
- **`evaluate(metrics)`** — recursive; returns the result tree, never raising on a
  failed bar.

Custom gates: subclass `Gate` and implement `evaluate`.

---

## 2. The concrete control gates (built in `probing/`)

Compositions of the primitives, each consuming a named metric the probing run produces:

| Gate | Composition | Metric consumed | Pass means |
|---|---|---|---|
| `selectivity` | `MetricGate("selectivity", gte=τ_sel)` | real-label AUC − control-label AUC (Hewitt–Liang) | the axis beats a control task |
| `negative_control` | `MetricGate("<ctrl>_auc", lte=τ_neg)` | AUC on shuffled votes / random labels / random embeddings / metadata bins / sky-noise | the control axis **does not** decode (it must fail to be credible) |
| `nuisance_clearance` | `MetricGate("nuisance_auc_max", lte=τ_nui)` | max AUC across z / magnitude / Petrosian size / SNR / PSF | the morphology axis is **not** reading off a nuisance |

A linearly-nameable feature is then, e.g.:

```python
linearly_nameable = gate.all(
    MetricGate("auc",              gte=0.80),
    MetricGate("selectivity",      gte=0.10),
    MetricGate("nuisance_auc_max", lte=0.65),
)
```

The thresholds (`τ_*`, `auc` bar) are **science-tunable** and owned by
`galaxy-jepa-scratchpad.md`, not hard-coded here.

---

## 3. What is **not** a gate

**`bbox_degradation` (β=0 ⇒ I-JEPA) is not a control gate.** It is deterministic
code-correctness — a **property test** (`tests/test_invariants_stubs.py
::test_masking_beta_zero_is_ijepa`), not a metric threshold. Modelling it as a gate
would double-book the invariant across two mechanisms; it is named here only to record
that it lives in the test suite, not the gate tree.

---

## 4. Forks

| Fork | Options | Recommendation | Status |
|---|---|---|---|
| Comparator set | `gt/gte/lt/lte` / add float `eq` / add `approx` | **`gt/gte/lt/lte`, no float `eq`** | proposed (recommendation stands) |
| Default thresholds (`τ_sel`, `τ_nui`, `auc` bar) | per-feature / global | flagged **science-tunable**, set in scratchpad | open (science) |
| `GateResult` serialisation into the artefact stamp | JSON tree / flat | JSON tree (mirrors `render()`) | open (minor) |
