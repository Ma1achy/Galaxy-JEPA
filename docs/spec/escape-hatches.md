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
| `effect_floor_open` | scoring without an intact effect-floor record (`ProbingConfig.effect_floor_file`, written once by `probing/floor.freeze_effect_floor`) — **smoke only**, see §2.1 | the pre-registered clean-vs-marginal line (D22); the run's `clean` flags, and so its R1/R2 split, were judged at an inline placeholder |

The registry is extended as new sanctioned deviations appear (e.g. `unfrozen_finetune`
in the Paper-2 fine-tuning contrast).

### 2.1 `effect_floor_open` — stamped, never declared; a smoke is the only bypass

Unlike the other entries this one is **not declarable**: `ProbingConfig` refuses
`effect_floor_open` in `escape_hatches` at load. It used to be a declarable hatch that let
any run fall back past a missing or edited floor file to the inline `effect_floor`; that
fallback is withdrawn, because the floor is the one number in the battery that is a call
rather than a derivation, and a hatch any run can take is not a pre-registration.

- A **non-smoke** run scores against an intact record or not at all. No
  `effect_floor_file`, a missing one, or one whose hash or rule no longer holds refuses the
  run at `floor.resolve_effect_floor`, before any fit. An inline `effect_floor_freeze` alone
  is not the gate, and `headline=True` requires `effect_floor_file`.
- A **smoke** (`smoke: true`) without an intact record scores at its inline `effect_floor`
  under a `floor.FloorBypass`. Its outputs carry the marker **`FLOOR BYPASSED`**: in
  `stamp.json`'s `escape_hatches_used` (as `["smoke", "effect_floor_open", "FLOOR BYPASSED"]`),
  in `ladder_summary.json`'s `effect_floor.status`, and on every existence entry's `floor`.
- **Every verdict read-out refuses a bypassed result**: `LadderResult.assert_reportable`,
  `ProbingReport.rung_table` / `population_comparison`, and the figures (`run_probing` draws
  none and records why). A result that names no floor at all is refused the same way.
- `nulls.existence_verdicts` takes the floor as a required object — the intact
  `EffectFloorRecord` (re-checked on entry) or a `FloorBypass`, which marks every verdict —
  never a bare number, so a direct caller cannot score past the gate.

Artefacts stamped before the gate carry `effect_floor_open` without the marker; a reader
treats either string as a bypass.

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
