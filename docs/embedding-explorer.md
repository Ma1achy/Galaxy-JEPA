# Galaxy-JEPA — embedding explorer (post-slice design sketch)

*Status: design sketch — **do not build yet**. Gated on the vertical slice showing life: an explorer over a collapsed encoder is just nearest-neighbours-in-noise. Build the moment the pilot's UMAP has a shape and the featured-ness AUC clears chance. British English.*

This is the layer that makes the representation **tangible** — drop a galaxy in, see where it lands, find its lookalikes, read off a morphology score. None of it is new science; it's a *lens* on the science the probing harness already produces, and most of it is a thin layer over artifacts the slice already creates (the frozen encoder, the embeddings, the concept direction).

---

## The intuition, made concrete

Three things, each a real capability with a real diagnostic use:

1. **"See where an image lands amongst the training set"** → project a new galaxy into the saved 2D map.
2. **"Find galaxies that look like this one"** → nearest neighbours in the 384-D embedding space.
3. **"See the spiral-ness of a galaxy"** → project its embedding onto a concept direction; the scalar *is* the score.

The third needs precision: a readable "X-ness" number requires X to be a **direction** in the space — and whether it is, and whether the score tracks human judgement, is *exactly* the nameability-ladder / uncertainty-geometry result. So the explorer doesn't *add* spiral-ness; it *displays* whatever the probing harness has established is a real direction. Tonight that's **featured-ness** (the one easy axis the slice fits). Bar-ness, spiral-ness, arm-count come online one at a time as the full ladder fits each direction — and the score is *calibrated* (tracks the human vote fraction) only for features where uncertainty geometry holds.

---

## The shared primitive — a stamped embedding index

Everything below reads from one precomputed artifact: the **embedding index** — the (N, 384) frozen embeddings of a reference set (probe-test, or a larger reference pull), with each galaxy's objID, cutout path, and labels.

```python
@torch.no_grad()
def build_index(encoder, reference_loader, stamp) -> EmbeddingIndex:
    assert_frozen(encoder)                          # never an unfrozen encoder
    embs, oids, paths, labels = [], [], [], []
    for b in reference_loader:
        embs.append(encoder.encode(b.images).cpu()) # (B, 384), mean-pool DEFAULT_LAYER=-2
        oids += b.object_ids; paths += b.paths; labels += b.labels
    return EmbeddingIndex(torch.cat(embs), oids, paths, labels, stamp=stamp)
```

It carries the **encoder stamp** (checkpoint config-hash + git SHA). Neighbours are only meaningful relative to a *specific* encoder's geometry — the stamp tells you which encoder's space you're in, and an index from a collapsed encoder is visibly garbage (incoherent neighbours), which is itself a diagnostic.

---

## Capability 1 — nearest neighbours ("lookalikes")

```python
def neighbours(index, query_emb, k=12, metric="cosine"):
    X = F.normalize(index.embeddings, dim=1) if metric == "cosine" else index.embeddings
    q = F.normalize(query_emb, dim=0)        if metric == "cosine" else query_emb
    sims = X @ q                              # cosine similarity
    idx = sims.topk(k).indices
    return [(index.object_ids[i], index.paths[i], sims[i].item()) for i in idx]
```

- **Cosine by default** (direction matters more than magnitude in these spaces); Euclidean optional.
- Query is either an existing galaxy (by objID, already in the index) or a **new image** — embed it fresh through the *same frozen encoder and the same frozen preprocessing pipeline* (parity matters here too: a new galaxy must get Q=4 asinh + the frozen normalisation, or it lands in the wrong place).
- **Diagnostic value:** drop in a barred spiral — if its neighbours are barred spirals, the representation captures real morphology; if they're a grab-bag, it doesn't. A representation-quality check you can *see*, and quantifiable as **k-NN label agreement** (a cheap number complementary to the probe AUC).

---

## Capability 2 — "where does it land"

```python
reducer = umap.UMAP(metric="cosine").fit(index.embeddings.numpy())  # fit ONCE, save the reducer
pt2d = reducer.transform(query_emb[None].numpy())                   # new galaxy → its 2D point
# plot: reference cloud coloured by label, query point marked
```

- The fitted reducer is **saved** so the map is stable — the same galaxy always lands in the same place.
- **Honesty:** UMAP is a 2D shadow of 384 dimensions, and transforming new points is approximate. It can invent separations that aren't real and hide ones that are. Treat "where it lands" as a **gut-check, never a measurement** — the quantitative claims are the AUC and the concept-direction projection, not the picture.

---

## Capability 3 — the "X-ness score" (the concept-direction projection)

The one that makes morphology a readable number — and the one wired straight to the science.

```python
def x_ness(embedding, w):          # w = unit concept direction, from the canonical logistic probe
    return float(embedding @ w)     # signed margin distance = the score

# the direction walk (the killer demo):
order = (index.embeddings @ w).argsort()              # least → most "X"
montage([index.paths[i] for i in order[::stride]])    # galaxies along the axis
```

- The direction is **loaded from the canonical L2-logistic probe artifact**, not recomputed ad hoc — so the score uses the *same* direction the science uses, plus its histogram position in the reference distribution (query marked).
- **The direction walk is the demo that matters:** sort the reference galaxies by projection onto `w` and show the gradient — most-smooth → most-featured. When you later claim "bar-ness is a clean direction," walking `w` and showing galaxies getting *progressively more barred* is the most convincing possible evidence the direction is real and not a nuisance correlate. This is the **qualitative companion to the controls battery** — it makes every probe result legible.
- **Honesty (again):** the score is only "spiral-ness" for features that *are* readable directions (the ladder decides), and only *calibrated* where uncertainty geometry holds. For an entangled feature the score varies but isn't independent; for a Rung-4 feature there's no meaningful direction to project onto — and that absence is a finding, not a bug.

---

## Where it lives / what it reuses

A small `eval/explore.py` (or an `explore/` package), built **on existing parts, not reinventing**:

- `core.encoder` — frozen, stamped (`assert_frozen`).
- `eval/embed.py` — the slice already computes embeddings; extend it to *persist* the index.
- `probing/logistic.py` — the fitted concept directions.
- `DirectorySource` — cutout loading for the montages.

New dependency: none beyond the slice's `umap-learn`. Nearest-neighbours is plain torch/numpy (or sklearn `NearestNeighbors`).

---

## Disciplines (the same ones as everywhere)

- **Frozen only** — never an unfrozen encoder, always the *stamped* checkpoint that produced the index.
- **Provenance** — the index carries the encoder stamp; you always know whose geometry you're exploring.
- **Honest visualisation** — UMAP is the gut-check; the projection / AUC is the claim.
- **Canonical directions** — X-ness uses the science's L2-logistic direction, not a one-off.
- **Same preprocessing for new images** — a dropped-in galaxy gets the frozen Q=4 asinh + normalisation, or it lands wrong.
- **Web-ready artifacts (cheap now, saves the demo later).** Persist the index, the UMAP 2D coordinates, the concept directions, and the cutout thumbnails as *static, serializable* files (JSON + image assets), not Python-runtime-only objects. Costs almost nothing at explorer-build time and means the eventual web demo is a reskin over existing files, not a from-scratch data re-extraction.
- **Gated on life** — meaningless over a collapsed encoder; build only once the slice says the science is alive.

---

## Interface staging (don't gold-plate)

1. **CLI first** — covers everything above:
   - `explore neighbours --galaxy <objID> --k 12` → montage of lookalikes.
   - `explore locate --galaxy <objID>` → its point on the saved UMAP.
   - `explore score --feature featured --galaxy <objID>` → the X-ness + histogram, and `--walk` for the direction montage.
2. **Notebook** — for interactive poking (drop an image, see neighbours, walk a direction).
3. **A little interactive app** — genuinely lovely *eventually* (hover the UMAP, click a galaxy, neighbours live), but it's a frontend project, not the first build. Defer until the science is carrying its weight; the CLI + notebook deliver the value now.

---

## When to build it

**The moment the pilot shows life** — a UMAP with a shape, featured-ness AUC above chance. At that point the index + nearest-neighbours + the featured-ness direction-walk are ~an afternoon on top of the slice's artifacts, and they turn the representation from an abstraction into something you can poke. Then **each new concept direction the full ladder fits plugs straight into capability 3** — bar-ness, spiral-ness, arm-count, scored and walkable as they come online. The explorer becomes the standing interpretability instrument for the whole rest of the project, and the most persuasive way to *show* (not just tabulate) that a concept direction is real.

If the pilot collapses, this waits — there's nothing to explore until the encoder learns structure.

---

## Future phase — the interactive paper (north star, dependency-chained)

The explorer and a web demo are **the same artifact at two fidelities**: the three capabilities above *are* the widgets an interactive write-up would embed. So the demo is a frontend over the explorer's precomputed **static** artifacts — frozen embeddings, UMAP coordinates, and concept directions as fixed blobs, **no live GPU model behind it** (the whole point of frozen representations is the geometry is precomputed). That's a static site with client-side interactivity: cheap to host, can't fall over.

**Why this project wants it more than most.** The claims here are geometric and visual — "this human concept is a *direction*," "distance along it reproduces human uncertainty," "galaxies cluster by a morphology the model was never told about." Those land far harder when a reader *walks the direction themselves* and watches galaxies get more barred, or hovers the embedding map and sees the structure, than as "ρ = 0.4, see Figure 7." For representational / interpretability work the interactive form isn't decoration — it's arguably the most honest *presentation* of a result that genuinely is a shape in a space. (Precedent: Distill-style interactive explainers.)

**Placement — the discipline.** This is the **end** of the dependency chain: pilot shows life → full slice → the *full probing harness* (the ladder, controls, uncertainty geometry — the actual results) → the explorer over those real results → **then** the web demo presenting them. The demo presents results that don't exist yet, so it stays a north star, not a this-week build. It's the same gold-plating trap as ever, wearing its most seductive costume — a beautiful demo is *much* more tempting to build than another control gate, which is exactly why it's the one to be most disciplined about deferring.

**Publication shape.** Both, not blog-alone: a **rigorous paper** (arXiv / a venue — the citability, peer review, and credential you want a v2-of-the-dissertation to carry) *and* an **interactive blog post** as its public face (the reach and legibility), with the explorer as the shared engine under both. The post multiplies the paper rather than replacing it — don't trade the academic weight for the reach.
