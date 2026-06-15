# Galaxy-JEPA — Architecture build-out brief (for Claude Code)

> Drop in the repo next to `docs/architecture.md`. Then: *"read `docs/architecture-buildout.md` and begin."*

## Where we are

The codebase architecture is designed (`docs/architecture.md`). This phase does two things: **(A)** flesh the architecture from a design doc into implementation-ready component contracts, and **(B)** build the **minimal `core/`** — and nothing past it.

Read first, in this order: `docs/architecture.md` (engineering source of truth), `galaxy-jepa-scratchpad.md` (science source of truth), `DECISIONS.md`, `docs/masking.md`. The whole point of the design is that the **experimental guardrails are structural, not discipline-at-the-keyboard** — preserve that in everything you write.

Non-negotiable discipline (from the architecture doc — do not erode):
- **Take the DNA, not the organs.** No reconciler, CRD, registry-DB, serving, or orchestration. This is a research library.
- **Hard invariants stay hard.** Frozen encoder in probing (no unfreeze path); uncertainty-geometry axis structurally cannot see graded vote fractions; a probing run carries its controls or it does not run; β=0 ⇒ I-JEPA as a property test; fail loudly, never silently default.
- **The "second consumer" test.** Build an abstraction when the second case arrives, not in anticipation. The registry and the full Tier-1 validator suite are **not** in this phase.
- British English. uv + pytest + ruff. Every new framework file gets a docstring naming the `architecture.md` section it implements.

---

## Track A — flesh out the component contracts (propose; flag the forks)

Expand the high-level architecture into implementation-ready detail. Prefer **splitting into focused docs under `docs/spec/`** (one per concern) over one giant file. Where a contract isn't pinned, **propose a recommendation and mark it a fork for sign-off** — don't silently choose. Cover:

1. **`Encoder` Protocol (`docs/spec/encoder.md`).** Finalise the contract: `encode()` (pooled, default) and `encode_tokens()` return shapes; `embed_dim`; a frozen-state check; a stable `name`. Resolve the pooled-vs-tokens default (recommendation in `architecture.md`: pooled default, tokens exposed). State exactly what "satisfies `Encoder`" means for JEPA, MAE, and contrastive so the cross-objective ladder (D12) is encoder-agnostic by contract.
2. **Config + provenance (`docs/spec/config.md`).** The pydantic config base; whether config derives from component `__init__` (Prism-style auto-capture) or explicit config classes — **pick one, justify it**. The run-stamp format: `(config hash, code SHA, data-snapshot version, seed, escape_hatches_used)`. The local artefact-directory layout. How a figure regenerates from its stamp.
3. **Gates (`docs/spec/gates.md`).** `MetricGate` (exactly-one-comparator, enforced), `all`/`any` composites (reject empty children), `GateResult` tree, the **verdict-separate-from-run-status** rule, the custom-gate mechanism. Define the concrete control gates — `selectivity`, `negative_control`, `nuisance_clearance`, `bbox_degradation` — and the metric each consumes.
4. **Power-path ledger (`docs/spec/escape-hatches.md`).** The `escape_hatches_used` mechanism: how a deviation (MLP probe, custom masking, warm-start) names the guarantee it forfeits and gets stamped onto the artefact.
5. **Validation tiers (`docs/spec/validation.md`).** Enumerate Tier-1 rules with IDs (Prism-style), starting with the must-haves: *a probing run must carry its controls*; *β in range*; *the uncertainty-geometry target is the binary extremes label, not the graded fraction*; *declared metadata columns are real*. Then Tier-2 (run-start: `petroRad` + pixel scale present, checkpoint loads, normalisation stats present) and Tier-3 (collapse/NaN/shape).
6. **Callbacks (`docs/spec/callbacks.md`).** The hook surface, the field-availability contract (what exists at which hook), and the executor-owned (config: seed, EMA schedule, masking ratio, grad clip) vs lifecycle-owned (callbacks: collapse monitor, sky-fraction logger, EMA updater, checkpointer) split.
7. **Data stack (`docs/spec/data.md`).** `DataSource` / transform pipeline / `StatefulTransform` contracts; the **pretrain-probe normalisation parity** rule (fit once on the pretraining corpus, freeze, apply everywhere); the CasJobs join — and note the dependency surfaced earlier: **the per-galaxy Petrosian box needs `petroRad` for the *pretraining* corpus (the large unlabelled SDSS set), not just the GZ2 probing set**, so the unlabelled pull must fetch `petroRad` + pixel scale.
8. **Objective interface (`docs/spec/objectives.md`).** The common shape of a training procedure (JEPA / MAE / contrastive) that *produces a frozen `Encoder`* — what they share, where JEPA's masking+EMA+predictor+latent-MSE lives.

Keep each spec doc tied back to its `architecture.md` section. Update `architecture.md` to link them. If anything in the flesh-out changes the **science**, propose a `galaxy-jepa-scratchpad.md` edit, don't diverge.

---

## Track B — build the minimal `core/` (do it; with tests)

This is the agreed minimal scope from the sequencing — build it directly, with property tests. **Do not build encoders, objectives, masking, probing, or data yet** — those are the next phase and need sign-off.

- `core/encoder.py` — the `Encoder` Protocol + a `is_frozen(encoder) -> bool` / `assert_frozen(encoder)` helper. No concrete encoder.
- `core/config.py` — the pydantic config base; the run-stamp (config hash; code SHA via `git`; seed; `escape_hatches_used`); serialise/deserialise; the artefact-stamp writer.
- `core/gates.py` — `Gate`, `MetricGate`, `AllGate`, `AnyGate`, `GateResult`; `__post_init__` enforcement (exactly one comparator; non-empty composites raise); the `gate.all(...)` / `gate.any(...)` factories; recursive `evaluate()` producing the result tree.
- `tests/` — property tests that pin the invariants:
  - gate composition (nested all/any), exactly-one-comparator raises, empty-composite raises;
  - a `GateResult` verdict tree renders and **run-status is independent of gate pass/fail**;
  - `assert_frozen` catches a non-frozen module;
  - **stubs** (xfail/skip with a TODO) for the β=0⇒I-JEPA masking property test and the normalisation-parity test, so the invariants are registered as tests before their code exists.

Verification: `uv run pytest` green; `python -c "import galaxy_jepa.core"` works; `pre-commit run --all-files` clean.

---

## Stop here

Hold at the `core/` boundary. The **registry** waits for the second encoder; the **full Tier-1 validator suite** grows one rule at a time (first rule: controls-required); **encoders / objectives / masking / probing / data** are the next phase and need sign-off on the Track-A specs first.

## First response I want

1. The Track-A spec docs under `docs/spec/` — as proposals, with every unpinned contract marked a fork + your recommendation.
2. The Track-B `core/` implementation + tests, built.
3. A short note listing the forks that need my call (at minimum: config-derivation approach, the callback hook surface, anything in the validation-rule list you're unsure of).

No encoders/objectives/probing/data code until the Track-A specs are signed off.
