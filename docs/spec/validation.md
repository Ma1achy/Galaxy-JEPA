# Spec — validation tiers

*Status: design proposal for sign-off. Expands `docs/architecture.md` → "Validation
tiers". The suite grows **one rule at a time** (the second-consumer discipline); this
doc enumerates the rules, their IDs, and which lands first. British English.*

Three tiers catch different error classes at the earliest possible point. The entire
value is **never burning a multi-hour pretraining run on a config typo or a forgotten
control.** Each rule has a stable ID so a failure cites the rule, not a stack trace.

---

## 1. Tier 1 — pre-run (cheap, before any GPU)

| ID | Rule | Fails when |
|---|---|---|
| `T1.controls-required` | A probing run carries its control battery (selectivity, negative controls, nuisance probes). | any are absent |
| `T1.beta-in-range` | Masking `β ∈ [0, 1]`. | out of range |
| `T1.uncertainty-target-binary` | The uncertainty-geometry **fit** target is the **binary** consensus-extremes label, never the graded vote fraction. | the graded fraction is passed as the fit target |
| `T1.metadata-columns-real` | Every declared nuisance/metadata column exists in the loaded catalogue. | a declared column is missing |

`T1.controls-required` **lands first in code** — it protects the headline result (a
morphology rung means nothing without its controls). The rest are added one at a time as
the guardrails are implemented; there is **no full Tier-1 validator suite this phase**.

`T1.uncertainty-target-binary` is belt-and-braces: the circularity is already
*impossible by construction* in `uncertainty_geometry` (the graded values enter only as
the test target — `docs/architecture.md` hard invariant 2). The Tier-1 rule catches a
*caller* who tries to pass the wrong thing, before compute.

---

## 2. Tier 2 — run start

| ID | Rule |
|---|---|
| `T2.petrorad-present` | The SDSS pretraining pull has `petroRad` + pixel scale (needed for the per-galaxy masking box). |
| `T2.checkpoint-loads` | A probing run's encoder checkpoint loads and is frozen (`assert_frozen`). |
| `T2.norm-stats-present` | The fitted normalisation statistics exist and match the declared pipeline. |
| `T2.stretch-sanity` | The stretch-sanity check passes — faint arms survive stretch+normalise on known faint-arm galaxies while the sky-noise floor stays controlled (`docs/spec/data.md`). |

`T2.stretch-sanity` **pairs with the collapse monitor** (Tier 3): if the encoder starts
modelling noise during pretraining, the stretch was too aggressive.

---

## 3. Tier 3 — runtime

| ID | Rule |
|---|---|
| `T3.no-nan-loss` | A NaN/Inf loss halts the run (fail loud, never silently continue). |
| `T3.collapse-monitor` | Representation variance / rank / std stays above the collapse floor (callback; may raise `StopRun`). |
| `T3.shape-on-first-forward` | Tensor shapes match the declared contract on the first forward pass. |

---

## 4. Forks

| Fork | Options | Recommendation | Status |
|---|---|---|---|
| First rule implemented | `controls-required` / `beta-in-range` / other | **`controls-required`** | proposed (recommendation stands) |
| Rule engine | Prism-style declarative validator suite / plain functions | **plain functions, grown one at a time** (no suite this phase) | proposed |
| Where Tier-1 runs | a `validate(run_config)` entry point / inline at run assembly | single `validate()` called before any GPU work | open (minor) |
