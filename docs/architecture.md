# Galaxy-JEPA — Codebase Architecture

*Status: design proposal. The engineering counterpart to `galaxy-jepa-scratchpad.md` — the scratchpad is the **science** source of truth, this is the **codebase** source of truth. If a design choice here changes the science, it's proposed as a scratchpad edit, not made silently.*

*Informed by the M1 (Developer SDK) design of Prism — its invariants, its normal/power/hard-invariant taxonomy, its gates, and its "the model is just an `nn.Module`" stance. It borrows those **patterns**, not Prism's platform machinery (see "What we deliberately do not build").*

---

## Guiding principle — a research library, not a platform

Prism is a multi-tenant platform with a Kubernetes reconciler, a CRD, a four-namespace governance model, a Postgres/MinIO registry, Argo orchestration, and Triton serving. **None of that belongs here.** Galaxy-JEPA is a single-researcher codebase whose job is to produce three defensible figures from a reproducible pipeline. Rebuilding platform machinery would be exactly the premature-MLOps trap the scratchpad warns against.

What transfers is the **design discipline** — and most of it is nearly free, because it's how you structure contracts and handle failure, not infrastructure you stand up. The rule throughout: **take the DNA, not the organs.** Where a pattern costs real code (a registry, validators), it is sequenced to arrive only when a second consumer makes it pay (see "Sequencing").

The discipline matters here for a specific reason: the entire thesis rests on the probe being a *trustworthy instrument*. A codebase that can silently skip a control, quietly unfreeze an encoder, or let the uncertainty-geometry axis see the labels it's meant to predict is a codebase that cannot support the claims. So the experimental guardrails are encoded **structurally**, the way Prism encodes artefact integrity — not left to discipline-at-the-keyboard.

---

## The keystone — the `Encoder` Protocol

The single most important architectural decision. The cross-objective control (DECISIONS D12) requires the *same* probe ladder to run identically over JEPA, MAE, and contrastive encoders — that is the only way a rung difference attributes to the objective rather than to the images. So the seam between "how an encoder was trained" and "how it is probed" is a **structural `Protocol`**, satisfied without inheritance:

```python
from typing import Protocol
import torch

class Encoder(Protocol):
    """Images -> embeddings. The only contract the probing/eval layer knows about.
    JEPA, MAE, and contrastive encoders all satisfy this — no shared base class."""
    embed_dim: int
    def encode(self, images: torch.Tensor) -> torch.Tensor: ...   # (B, embed_dim), pooled
    def encode_tokens(self, images: torch.Tensor) -> torch.Tensor: ...  # (B, N, embed_dim)
```

Borrowed directly from Prism's "the model is just an `nn.Module` — there is no `prism.BaseModel`": the encoder is plain PyTorch architecture, and **all training machinery lives outside it**. JEPA's masking, EMA target, predictor, and latent-MSE loss are *not* on the encoder — they live in `objectives/`. The consequence is the one that matters: the object you pretrain is the same object you freeze and probe, and the probing layer consumes `Encoder` and *nothing else*. Adding the MAE baseline means producing another `Encoder`; the ladder doesn't change a line.

*(Decision to pin: whether the probe operates on the pooled `encode()` or on `encode_tokens()` aggregated. Default to pooled for concept directions; expose tokens for the attention/Jacobian explorations in the parking lot.)*

---

## Hard invariants — the experimental discipline, in code

Prism's sharpest idea: some constraints have **no power path**, because an escape hatch would undermine the correctness model ("a platform that lets you skip the checksum check no longer guarantees integrity"). Your methodology *is* a correctness model. These are enforced structurally, with no override flag:

1. **The encoder is frozen during probing.** The probing API only ever receives a frozen encoder; there is no `unfreeze=` parameter on that path. Fine-tuning is a *separate*, clearly-named procedure in `objectives/`, never reachable from `probing/`.
2. **The uncertainty-geometry axis cannot see graded vote fractions.** The circularity you fixed is made *impossible by construction* — the function fits on consensus extremes and the graded values enter only as the test target:
   ```python
   def uncertainty_geometry(enc: Encoder, imgs, votes):
       ext = (votes > 0.8) | (votes < 0.2)            # fit set: extremes only
       axis = fit_logistic(enc.encode(imgs[ext]), (votes[ext] > 0.5))
       mid = (votes >= 0.2) & (votes <= 0.8)          # held out of fitting entirely
       return spearman(project(enc.encode(imgs[mid]), axis), votes[mid])
       # graded `votes` are a TEST target here and nowhere a fit target — structurally.
   ```
3. **A probing run carries its control battery or it does not run.** Selectivity, negative controls, and the core nuisance probes are required arguments to a probing run, validated at Tier 1. A morphology result without its controls is rejected before compute, not produced and quietly trusted.
4. **β = 0 reproduces standard I-JEPA.** The masking module's strict-generalisation property (`docs/masking.md`) is a **property test**, not a comment — `test_masking_beta_zero_is_ijepa` asserts identical block statistics at β=0. The invariant is executable.
5. **Pretraining and probing corpora are decoupled but the preprocessing is shared** (D6). The encoder is pretrained on the large unlabelled SDSS set and probed on GZ2; the image preprocessing (esp. normalisation stats) is *fitted once on the pretraining corpus and applied identically everywhere* (see "The data stack").
6. **Fail loudly, never silently corrupt** (Prism Invariant 1). A missing `petroRad` is *either* a loud error *or* an explicit, logged fallback to the global box — never a silent default. A control that errored is a gate that *errored*, never a silent pass. A NaN loss halts the run.

---

## Power paths — deviations that record what they forfeit

Between "plain config" and "hard invariant" sits Prism's middle tier: a custom implementation is allowed, but it **names the guarantee it forfeits** and records it in `escape_hatches_used`. Ported, this makes every deviation from the canonical method *auditable in the run metadata*:

- Swap the canonical L2-logistic probe for an MLP → records *"forfeits the clean-linear-direction claim; result supports at most Rung-2/3 with controls."*
- Supply a custom masking scheme → records *"forfeits the β=0 I-JEPA-equivalence guarantee unless property-tested."*
- Warm-start the encoder from ImageNet → records *"forfeits the 'directions present before any label' attribution; Paper-2 ablation only."*

The run still proceeds; the deviation is stamped onto the artefact so no figure's provenance can hide a methodological shortcut.

---

## Controls as gates

Prism's `Gate` is a first-class composable object — `MetricGate` with exactly one comparator, `all`/`any` composites that reject empty children ("a degenerate logic error, not a no-op"), and a result tree. Crucially, **the verdict is recorded separately from run status**: the job *succeeds* (the work ran) even if every gate *fails* (the bar wasn't met). That distinction is exactly right for science — a feature failing its controls is a *finding*, not an error.

```python
linearly_nameable = gate.all(
    MetricGate("auc",             gte=0.80),
    MetricGate("selectivity",     gte=0.10),   # real-label AUC minus control-label AUC
    MetricGate("nuisance_auc_max", lte=0.65),  # the axis must NOT read off z / size / brightness
)
```

Each feature's place on the nameability ladder is a gate tree carried *with* its result. The probing run produces the AUC, the cosine matrix, and the uncertainty-geometry scatter regardless; the gate tree records the *verdict* per feature. Controls become structurally impossible to skip, because the verdict is a required output, not an optional afterthought.

---

## Config is the experiment record

Prism captures `__init__` kwargs into auto-generated `to_config()`/`from_config()`, forbids `*args/**kwargs` so components are reproducible by construction, and never fetches config at runtime ("configuration over runtime cleverness"). Here:

- Typed config (pydantic), validated at load. A run is **fully determined by** `(config hash, code SHA, data-snapshot version, seed)`.
- Each component's config derives from its constructor signature — no separate, driftable config schema to keep in sync.
- Every artefact (checkpoint, embeddings, probe, figure) is **stamped** with that tuple plus `escape_hatches_used`. The three headline figures must each **regenerate from their stamp** — this is the scientific analogue of Prism's three-file artefact atomicity. No figure exists without the config that produced it.
- No runtime config fetching, no dynamic reloading mid-run. The config *is* the experiment.

---

## Validation tiers

Three tiers, catching different error classes at the earliest possible point (Prism's tiering, scaled to one researcher):

- **Tier 1 — pre-run (cheap, before any GPU).** Is the control battery present? Is β in range? Is the uncertainty-geometry target the *binary* extremes label rather than the graded fraction? Are the declared metadata columns real? Catches the missing-control and out-of-range mistakes in a second.
- **Tier 2 — run start.** The SDSS pull actually has `petroRad` and pixel scale; the checkpoint loads; the fitted normalisation stats are present.
- **Tier 3 — runtime.** Collapse monitor, NaN guard, shape mismatch on first forward.

The entire value is **never burning a multi-hour pretraining run on a config typo or a forgotten control.**

---

## The data stack — pretrain-probe parity

Prism's three-concern split (`DataSource` / `Transform` / `Sink`), with a `StatefulTransform` that *must be fitted* and a serving path enforcing train-serve parity, maps to a real correctness trap: the image preprocessing — normalisation stats above all — used in pretraining must be applied *identically* when computing probe embeddings, or the probe sees a different distribution than the encoder was trained on.

So: a `DataSource` for SDSS cutouts + the CasJobs metadata join; a composable transform pipeline (crop, augment); normalisation as a `StatefulTransform` **fitted once on the pretraining corpus, frozen, and applied everywhere** — pretraining, probing, and every baseline. Masking is *not* a data transform (it's part of the JEPA objective, applied in `objectives/`); keep that boundary clean.

---

## Package layout

```
src/galaxy_jepa/
  core/         # the small framework — Encoder Protocol, config base + provenance
                #   stamping, Gate types. (Registry + Tier-1 validators arrive later.)
  models/       # pure encoders (ViT now, CCT later) — plain nn.Modules satisfying
                #   Encoder. No training logic, no loss, no masking.
  objectives/   # JEPA (masking+EMA+predictor+latent-MSE), MAE, contrastive — each a
                #   config-driven procedure that PRODUCES a frozen Encoder.
  masking/      # bbox-biased multi-block sampler; β=0⇒I-JEPA as a property test.
  data/         # DataSource (SDSS cutouts + CasJobs join), transform pipeline,
                #   fitted normalisation (StatefulTransform).
  probing/      # encoder-agnostic: logistic concept directions, the ladder,
                #   controls-as-gates, the non-circular uncertainty protocol.
  eval/         # the three figures + metrics, each stamped and regenerable.
  callbacks/    # training-loop hooks: collapse monitor, sky-fraction logger,
                #   EMA updater, checkpointer.
configs/        # pretrain.yaml, probe.yaml — typed, validated, the experiment record.
scripts/        # thin CLI entry points: pull_data, run_pretrain, run_probe, make_figure.
tests/          # property tests for the hard invariants (β=0, frozen-encoder, parity).
```

**Control-flow ownership** (Prism's executor- vs lifecycle-owned split): things the training loop needs *before/during* `loss.backward()` — `seed`, EMA schedule, masking ratio, grad clip — live in the run config; behaviour that's *inserted around* the loop — collapse monitoring, the sky-fraction diagnostic, logging, checkpointing — lives in `callbacks/`. The encoder, the objective, the data pipeline, and the probe are independently importable and testable with no orchestration present.

---

## Component specs

The high-level contracts above are fleshed into implementation-ready detail under
`docs/spec/` (one focused doc per concern). Each names the section here that it expands
and ends with a flagged forks list.

| Spec | Expands | Built |
|---|---|---|
| [`spec/encoder.md`](spec/encoder.md) | The keystone — the `Encoder` Protocol | `core/encoder.py` |
| [`spec/config.md`](spec/config.md) | Config is the experiment record | `core/config.py` |
| [`spec/gates.md`](spec/gates.md) | Controls as gates | `core/gates.py` (+ `probing/`) |
| [`spec/escape-hatches.md`](spec/escape-hatches.md) | Power paths | with the deviating subsystem |
| [`spec/validation.md`](spec/validation.md) | Validation tiers | grown one rule at a time |
| [`spec/callbacks.md`](spec/callbacks.md) | Control-flow ownership | `callbacks/` |
| [`spec/data.md`](spec/data.md) | The data stack — pretrain-probe parity | `data/` |
| [`spec/objectives.md`](spec/objectives.md) | Package layout (`objectives/`) | `objectives/` |

## What we deliberately do not build

The boundary, held firmly — none of this earns its place in Paper 1:

- **No reconciler, no CRD, no Kubernetes operator.**
- **No registry-as-database** (Postgres/MinIO model registry). A stamped local artefact directory is enough.
- **No serving stack** (Triton/KServe). Nothing here is served.
- **No orchestration substrate** (Argo). A sweep is a config + a loop.
- **No multi-namespace governance, no secrets injection, no team packaging.** Single user.

When the Paper-2 sweeps get large — the data-degradation × hyperparameter cross-product for the survey-leakage ablation — *that* is where declarative orchestration earns its place, and it is the seam where the real Prism could actually run the workload. Not before.

---

## Sequencing — start minimal, grow on demand

`core/` is deliberately tiny on day one — **the `Encoder` Protocol, the pydantic config base + provenance stamping, and the `Gate` types only.** Everything else is plain modules.

- The **registry** (`@register.encoder("vit_s16")`, config-string → implementation) arrives the moment the **second encoder** lands — i.e. when the MAE baseline or the CCT backbone makes "resolve a class from a config string" pay for itself. Until then, direct construction is simpler and YAGNI applies.
- The **Tier-1 validator suite** grows one rule at a time as the guardrails are implemented — the first rule is "a probing run must carry its controls," because that one protects the headline result.

The test for adding framework code is always: *is there a second consumer yet?* Build the abstraction when the second case arrives, not in anticipation of it — the discipline (invariants, gates, parity, fail-loud) is what's bought upfront; the machinery is grown.
