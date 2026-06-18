# Embedding explorer — build plan

*Status: **build plan, not yet built.** The design rationale is `docs/embedding-explorer.md`;
this is the actionable sequencing. **Do not build during the consolidation pass.** Gated on:
(1) the probing harness existing (the explorer is richest over its full set of concept
directions, not the slice's single `featured` axis), and (2) the pilot showing life — which it
has (AUC ≈ 0.91, healthy trace). British English.*

The explorer is a **lens over artefacts the harness already produces**, not new science. It
turns the representation into something you can poke: drop a galaxy in, see its neighbours,
read off a morphology score, walk a concept direction. Build it *after* the probing harness so
each concept direction the ladder fits plugs straight into the "X-ness score" (capability 3).

## What already exists (the seam is in place)

The consolidation pass made the harness persist the explorer's inputs as **web-ready static
blobs** under `runs/<run>/explorer/` (`galaxy_jepa.harness._write_explorer_blobs`):

- `embeddings.npz` — `object_ids`, `x` (N×384 frozen `probe-test` embeddings), `y`, `fraction`.
- `concept_directions.json` — the fitted L2-logistic direction(s): `w_unit` (the projection
  axis), `w_raw`/`bias` (the affine logit weights in raw embedding space), AUC + CI, and the
  encoder stamp. Today: `featured`. The full ladder adds bar-ness, spiral-ness, arm-count, …
- `umap_coords.json` — saved 2D coordinates + objIDs (a stable map).
- `index.json` — manifest (n, embed_dim, encoder stamp, file pointers).

Everything carries the **encoder stamp** (config_hash + git SHA): neighbours/directions are
only meaningful relative to a specific encoder's geometry, and a collapsed encoder's index is
visibly garbage — itself a diagnostic.

## Build sequence (CLI-first; don't gold-plate)

1. **`eval/explore.py` — load the index.** Read `embeddings.npz` + the JSONs into a small
   `EmbeddingIndex` (embeddings, objIDs, paths via `DirectorySource`, labels, stamp). One
   precomputed artefact; no live model behind it.
2. **Capability 1 — nearest neighbours.** Cosine top-k over the normalised embedding matrix
   (plain torch/numpy or sklearn `NearestNeighbors`). A new query image is embedded through the
   *same frozen encoder + frozen preprocessing pipeline* (parity matters — Q=4 asinh + the
   frozen normalisation, or it lands wrong). Report **k-NN label agreement** as a cheap number
   complementary to the probe AUC. CLI: `explore neighbours --galaxy <objID> --k 12`.
3. **Capability 2 — "where it lands".** Reuse `eval/embed.umap_2d` / the saved
   `umap_coords.json`; project a new point onto the saved map. CLI: `explore locate --galaxy
   <objID>`. Honesty: UMAP is a gut-check, never a measurement.
4. **Capability 3 — the X-ness score (the demo that matters).** Project an embedding onto a
   concept direction `w_unit` from `concept_directions.json` (the *canonical* probe direction,
   not a one-off): the scalar **is** the score. The **direction walk** — sort the reference set
   by projection and montage galaxies along the axis (most-smooth → most-featured) — is the
   qualitative companion to the controls battery. CLI: `explore score --feature <name>
   --galaxy <objID>` and `--walk`. A direction is only calibrated where uncertainty geometry
   holds; for a Rung-4 feature there is no meaningful direction, and that absence is a finding.
5. **Notebook** for interactive poking. A small interactive web app is a *later* north star
   (`docs/embedding-explorer.md` "Future phase") — a static site over these same blobs, no live
   GPU — explicitly deferred; it presents results that don't exist yet.

## Reuses (take the DNA, not the organs)

`core.encoder` (frozen, `assert_frozen`); `eval/embed.py` (UMAP, already persists coords);
`probing/logistic.py` (`probe_direction` → the canonical directions); `DirectorySource`
(cutout loading for montages). New dependency: none beyond the slice's `umap-learn`.

## Disciplines

Frozen-only, stamped provenance, honest visualisation (UMAP = gut-check, projection/AUC = the
claim), canonical directions (not ad-hoc), same preprocessing for dropped-in images, and
**gated on life** — meaningless over a collapsed encoder.

## When to build

After the probing harness fits its first few concept directions. At that point the
index + neighbours + direction-walk are ~an afternoon on top of existing artefacts, and the
explorer becomes the standing interpretability instrument for the rest of the project.
