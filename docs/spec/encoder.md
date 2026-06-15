# Spec — the `Encoder` Protocol

*Status: design proposal for sign-off. Expands `docs/architecture.md` → "The keystone
— the `Encoder` Protocol". Implemented (Track B) in `src/galaxy_jepa/core/encoder.py`.
British English.*

The `Encoder` is the seam between **how an encoder was trained** and **how it is
probed**. The cross-objective control (`DECISIONS.md` D12) runs the *same* probe ladder
identically over JEPA, MAE, and contrastive encoders; that only attributes a rung
difference to the objective if every encoder presents an identical surface. So the
contract is a structural `typing.Protocol`, satisfied by duck typing — no shared base
class, no `BaseModel`. The thing you pretrain is the thing you freeze and probe.

---

## 1. The contract

```python
@runtime_checkable
class Encoder(Protocol):
    embed_dim: int
    name: str
    def encode(self, images: torch.Tensor) -> torch.Tensor: ...        # (B, embed_dim)
    def encode_tokens(self, images: torch.Tensor) -> torch.Tensor: ... # (B, N, embed_dim)
    def encode_at(self, images: torch.Tensor, layer: int) -> torch.Tensor: ...  # (B, embed_dim)
```

| Member | Shape / type | Meaning |
|---|---|---|
| `embed_dim` | `int` | Width of the pooled embedding from `encode`. |
| `name` | `str` | Stable slug (e.g. `vit_s16_jepa`); appears in artefact stamps + the comparison table. |
| `encode(images)` | `(B, embed_dim)` | Pooled embedding at the **pinned layer** (the headline read-out). |
| `encode_tokens(images)` | `(B, N, embed_dim)` | Per-token embeddings at the pinned layer (attention/Jacobian explorations). |
| `encode_at(images, layer)` | `(B, embed_dim)` | Pooled embedding at an **explicit** block — for the supplementary layer profile only. |

`encode`/`encode_tokens`/`encode_at` all return the **pre-projection backbone**
representation — projection heads (contrastive) and decoders (MAE) are excluded.
`embed_dim` is the backbone width, identical in meaning across objectives.

**What "satisfies `Encoder`" means per objective:**
- **JEPA** — the context encoder (the EMA target encoder, predictor, and latent-MSE
  loss stay in `objectives/`, not on the encoder).
- **MAE** — the encoder only; the decoder is discarded after pretraining.
- **Contrastive** — the backbone only; the projection/prediction MLP is discarded.

Each is exported frozen from `objectives/` (see `docs/spec/objectives.md`), so the
probing layer consumes `Encoder` and nothing else.

---

## 2. Pooling — **decided: mean over the layer's tokens**

`encode` mean-pools the chosen layer's token embeddings. This is objective-agnostic:
JEPA and MAE do not train a CLS token the way some supervised ViTs do, so relying on a
CLS embedding would inject a pooling artefact into the D12 comparison. Mean pooling is
defined identically for all three.

---

## 3. Layer policy — **decided: fixed penultimate block (`DEFAULT_LAYER = -2`)**

A single pooled vector is implicitly the *last* layer. But MAE's final block
**specialises for its decoder** and probes markedly worse than its intermediate layers
(a well-replicated MAE result), whereas JEPA and contrastive probe fine at the last
layer. A last-layer-only ladder would confound a genuine rung difference with a
layer-choice artefact and make MAE look artificially weak.

So the headline comparison reads a **fixed penultimate block, matched across all three
objectives** — `encode` uses `core.encoder.DEFAULT_LAYER`. The rule, stated in code and
here: **layer is not a free per-result parameter.**

- The **full layer profile** (probe AUC vs depth) is produced as a **supplementary
  analysis** via `encode_at`, for transparency — never to pick a winning layer.
- A best-layer **sweep is rejected** for the headline: selecting the layer that
  maximises probe AUC makes depth a researcher degree of freedom and contaminates the
  controls (selectivity / nuisance-clearance evaluated at a layer chosen for high
  decodability are not honest). *If* a sweep is ever reintroduced, it must be a
  pre-registered policy chosen on a **validation** split and frozen before probe +
  controls run on the **test** split, applied identically across all three objectives.

This is reflected in the scratchpad's cross-objective section (proposed edit).

---

## 4. Frozen-state check

`is_frozen(encoder) -> bool` and `assert_frozen(encoder)` are **free functions**, not
Protocol members, keeping the surface minimal. `assert_frozen` is called on entry to
every probing run (`docs/architecture.md` hard invariant 1): a still-trainable encoder
fails loudly; there is no `unfreeze=` path on the probing API. Duck-typed on
`parameters()`, so any `nn.Module`-shaped object works; an object exposing no
parameters is vacuously frozen.

---

## 5. Forks

| Fork | Options | Recommendation | Status |
|---|---|---|---|
| Pooling | mean-over-tokens / CLS | **mean** | **decided** |
| Layer policy | fixed penultimate / pre-registered validation-split sweep | **fixed penultimate** | **decided** |
| `name` shape | free string / structured `(arch, objective, …)` | free-form **stable slug** | proposed |
| Frozen check | free function / Protocol method | **free function** | proposed |
| Token aggregation for `encode_tokens` consumers | raw tokens exposed / forced pooling | **expose raw tokens** (parking-lot Jacobian/attention) | proposed |
