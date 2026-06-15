# Spec — the objective interface

*Status: design proposal for sign-off. Expands `docs/architecture.md` → "Package layout"
(`objectives/`) and "The keystone". Built in `src/galaxy_jepa/objectives/` (next phase).
British English.*

An **objective** is a config-driven training procedure that **produces a frozen
`Encoder`**. JEPA, MAE, and contrastive are three objectives that share a shape; the
D12 ladder then runs identically over their outputs (`docs/spec/encoder.md`).

---

## 1. What every objective shares

- It is `Configurable` (auto-captured config; `docs/spec/config.md`).
- It runs a config-driven loop with the callback hook surface
  (`docs/spec/callbacks.md`) — executor-owned `seed`/EMA-schedule/masking-ratio/grad-clip
  in config, lifecycle behaviour in `callbacks/`.
- It **produces an `Encoder`** — a plain `nn.Module` from `models/` satisfying the
  Protocol. Training machinery (losses, predictors, EMA targets, decoders, projection
  heads) lives in the objective and is discarded or detached on export.

## 2. Where JEPA-specific pieces live

All inside `objectives/jepa.py`, **not** on the encoder:

- the **bbox-biased multi-block masking** (`masking/`, applied here — `docs/masking.md`);
- the **EMA target encoder** (updated by the EMA-updater callback);
- the **predictor**;
- the **latent-MSE loss**.

MAE adds a decoder + pixel-reconstruction loss; contrastive adds a projection head +
the contrastive loss (and its augmentation policy). Each reproduced **on the SDSS
corpus** (D12) — the MAE per the Wu & Walmsley recipe (ViT ~30M, 3-layer decoder, 8×8
patches), the released Euclid MAE used only to validate the reimplementation.

---

## 3. Handoff to probing — **decided: stamped checkpoint, reloaded frozen**

An objective **writes a provenance-stamped checkpoint** (`docs/spec/config.md`
`write_stamp`); `probing/` **reloads it as a frozen `Encoder`** and calls `assert_frozen`
on entry. This forces the freeze boundary **through disk** — there is no live handle to a
still-trainable module that probing could accidentally update — and guarantees every
probe run is reproducible from a stamped artefact. (It also means probing never imports
`objectives/`: it consumes a checkpoint + `models/`.)

---

## 4. Forks

| Fork | Options | Recommendation | Status |
|---|---|---|---|
| Handoff | stamped checkpoint reloaded frozen / in-memory handle | **stamped checkpoint** | **decided** |
| Objective base | shared `Objective` ABC / duck-typed `run()` | duck-typed `run(config) -> checkpoint path` (no premature ABC) | proposed |
| Contrastive variant | MoCo / BYOL | **open — your call** (both SDSS-trained; `DECISIONS.md` D12 sub) | open (science) |
