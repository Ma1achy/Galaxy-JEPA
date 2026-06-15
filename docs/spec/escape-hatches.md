# Spec — the power-path ledger (`escape_hatches_used`)

*Status: design proposal for sign-off. Expands `docs/architecture.md` → "Power paths —
deviations that record what they forfeit". Built with the relevant subsystem (probing /
objectives), recorded via `core/config.py`'s `RunStamp.escape_hatches_used`. British
English.*

Between "plain config" and "hard invariant" sits the middle tier: a custom
implementation is **allowed**, but it **names the guarantee it forfeits** and that name
is stamped onto the artefact. The run still proceeds; the deviation is auditable in the
run metadata, so no figure's provenance can hide a methodological shortcut.

---

## 1. Mechanism

An escape hatch is an **identifier string** appended to `RunStamp.escape_hatches_used`
when a deviation is taken. It travels into `stamp.json` with the rest of the provenance
tuple. A hatch identifier names *what guarantee is lost*, not merely *what was done*.

```
escape_hatches_used = ["mlp_probe", "imagenet_warmstart"]
```

---

## 2. The known-hatch registry (seeded)

From `docs/architecture.md`:

| Identifier | Deviation | Guarantee forfeited |
|---|---|---|
| `mlp_probe` | non-linear (MLP/k-NN) probe instead of L2-logistic | the clean-linear-direction claim; result supports at most Rung-2/3 **with controls** |
| `custom_masking` | a masking scheme other than the signed-off bbox-biased sampler | the β=0 ⇒ I-JEPA equivalence, **unless** property-tested |
| `imagenet_warmstart` | encoder warm-started from ImageNet | the "directions present *before any label*" attribution; Paper-2 ablation only |

The registry is extended as new sanctioned deviations appear (e.g. `unfrozen_finetune`
in the Paper-2 fine-tuning contrast).

---

## 3. Closed enum vs free-form — **recommendation: registry + warned free-form**

A **known-hatch registry** (the table above) is the sanctioned set: passing a known
identifier is silent and expected. A **free-form** identifier is *also* accepted but
emits a **loud warning** ("undeclared escape hatch — add it to the registry"). This is
the fail-loud-but-don't-block stance: exploration is never blocked, but an
undocumented deviation is impossible to take *quietly*.

A hatch must never be a hard invariant in disguise — the frozen-encoder rule, the
non-circular uncertainty axis, and "a probing run carries its controls" have **no**
hatch (`docs/architecture.md` hard invariants); they are structural and cannot be
forfeited.

---

## 4. Forks

| Fork | Options | Recommendation | Status |
|---|---|---|---|
| Enum vs free-form | closed enum / registry + warned free-form / open | **registry + warned free-form** | proposed (recommendation stands) |
| Where a hatch is declared | at component construction / at run assembly | at run assembly (so the stamp sees the whole set) | open (minor) |
