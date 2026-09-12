"""Pixel validity — which pixels of a stamp carry real measurement (Brief E).

SDSS cutouts near a frame boundary run off the edge and are padded with a constant; the
frames also carry masked/dead pixels and saturated cores. All of them are *exactly constant*
regions, and all of them are degenerate the same way: they bias a normalisation statistic
toward the constant and deflate its variance, and a JEPA target block that lands on one is
trivially predictable — free loss reduction that teaches nothing.

**The rule, and why it is this one.** A pixel is invalid iff it is bit-identical, in every
channel, to at least one of its in-bounds 4-neighbours, **and** its connected region spans at
least ``MIN_REGION_PX``. The size floor is not tidying: these stamps are quantised finely
enough that adjacent pixels collide by chance about seventeen times per stamp, and without it
the detector calls every stamp in the corpus contaminated. See ``MIN_REGION_PX`` for the
measured distribution that places it.

Exact identity, not a tolerance, and not a value threshold — measured on 250 pretrain stamps,
the pad value is ``0`` while sky sits at median ``0.0013`` with ``sigma = 0.114``, i.e. **0.0
sigma apart**. Nothing separates them by value at all. What separates them is extent: sky
varies pixel to pixel, so a *large* exactly-constant region is only ever an artefact.

Per-channel comparison rather than a shared constant, deliberately: ``AsinhStretch`` and
``Normalise`` are per-channel affine, so "constant within a channel" survives the pipeline
while "the same value across channels" does not. The detector therefore holds on raw and
normalised stamps alike.

``RegionKind`` separates edge-touching padding from interior regions (thin dead columns,
compact saturated cores) because the two would not be equally safe to avoid when masking:
padding is off-galaxy by construction, while a saturated core sits at the centre where bulge
structure lives. **On these corpora that distinction is currently moot** — surveyed over 3,000
stamps, every qualifying region in both the probe and pretrain corpora is edge padding and the
interior count is exactly zero. The classification is kept as the guard that says so, not
because this data needs it; the earlier "7.2% interior" reading was quantisation noise under a
weaker rule, not dead pixels.

No scipy: ``scipy`` is an optional dependency here, so the connected-component pass is
row-run labelling with union-find, written out here.
"""

from __future__ import annotations

import dataclasses

import numpy as np

__all__ = [
    "MIN_REGION_PX",
    "RegionKind",
    "ConstantRegion",
    "StampValidity",
    "validity_mask",
    "invalid_planes",
    "edge_connected",
    "analyse_validity",
    "token_invalid_fraction",
    "LINEAR_ASPECT",
    "LINEAR_WIDTH_PX",
]

Array = np.ndarray

#: Smallest constant region that counts. **This is load-bearing, and it is measured.** SDSS
#: stamps are heavily quantised — ~569 distinct values across 65,536 pixels, in uniform steps
#: of 2^-19 — so adjacent pixels are bit-identical by chance constantly, and the bare
#: bit-identity rule fires ~17 times per stamp on nothing at all.
#:
#: Measured over **14,486 raw regions across 800 stamps of both corpora**, the size
#: distribution is not merely skewed, it is *disjoint*: 89% are exactly 2 px, a decaying tail
#: of chance clusters ends at 27 px, and then there is **nothing whatsoever between 33 and 255
#: px** before the real regions resume at 256 and run past 14,000. Any floor in [33, 256]
#: gives an identical answer on this data, so the choice is not fitted to the gap — it is
#: taken at the one value in the gap that means something. 256 px is a single 16x16 ViT token:
#: below it a region cannot degrade even one token of the mask, and weighs under 0.4% of a
#: normalisation statistic. Below the granularity anything downstream acts at, in other words.
MIN_REGION_PX = 256

#: A component is "linear" (a dead row/column) at or past this bounding-box aspect ratio, or
#: at or below this short-side width. Dead columns are one or a few pixels wide and run the
#: height of the frame; a saturated core is blobby. Both tests, because a short dead segment
#: is thin without being long.
LINEAR_ASPECT = 4.0
LINEAR_WIDTH_PX = 3


class RegionKind:
    """What a constant region is, which decides who is allowed to ignore it."""

    EDGE = "edge"  # touches the frame border — cutout padding
    LINEAR = "linear"  # thin and long, interior — a masked/dead row or column
    COMPACT = "compact"  # blobby, interior — a saturated core or masked blob


@dataclasses.dataclass(frozen=True)
class ConstantRegion:
    """One exactly-constant connected region, and the shape facts that classify it."""

    n_pixels: int
    touches_edge: bool
    height: int
    width: int
    centroid_radius_px: float  # distance of the centroid from the stamp centre

    @property
    def aspect(self) -> float:
        """Bounding-box long side over short side (``1.0`` for a square)."""
        lo, hi = sorted((self.height, self.width))
        return float(hi) / float(max(lo, 1))

    @property
    def kind(self) -> str:
        if self.touches_edge:
            return RegionKind.EDGE
        if self.aspect >= LINEAR_ASPECT or min(self.height, self.width) <= LINEAR_WIDTH_PX:
            return RegionKind.LINEAR
        return RegionKind.COMPACT


@dataclasses.dataclass(frozen=True)
class StampValidity:
    """A stamp's validity mask plus the regions that produced it."""

    valid: Array  # (H, W) bool — True is a real pixel
    regions: tuple[ConstantRegion, ...]

    @property
    def invalid_fraction(self) -> float:
        return float(1.0 - self.valid.mean())

    def fraction_of_kind(self, kind: str) -> float:
        """Share of the frame taken by regions of one :class:`RegionKind`."""
        n = sum(r.n_pixels for r in self.regions if r.kind == kind)
        return float(n) / float(self.valid.size)


def _like_neighbour(image: Array) -> Array:
    """``(H, W)`` bool: pixels bit-identical, in every channel, to **at least one** 4-neighbour.

    "At least one", not "all". Requiring all four would erode the region and need dilating
    back, and that fails on precisely the shapes that matter: a two-pixel-wide dead column has
    no pixel with four like neighbours at all, so it would vanish entirely, and a rectangle's
    corner sits two steps from the nearest such pixel. One like neighbour catches every run of
    two or more contiguous constant pixels, which is every region that occurs.
    """
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"expected a (C, H, W) stamp, got shape {arr.shape}")
    like = np.zeros(arr.shape[1:], dtype=bool)
    vertical = (arr[:, :-1, :] == arr[:, 1:, :]).all(axis=0)
    horizontal = (arr[:, :, :-1] == arr[:, :, 1:]).all(axis=0)
    like[:-1, :] |= vertical
    like[1:, :] |= vertical
    like[:, :-1] |= horizontal
    like[:, 1:] |= horizontal
    return like


def _invalid_mask(image: Array) -> Array:
    """``(H, W)`` bool: True where the pixel carries no real measurement."""
    labels, n = _label(_like_neighbour(image))
    if n == 0:
        return np.zeros(labels.shape, dtype=bool)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    keep = sizes >= MIN_REGION_PX
    keep[0] = False
    return keep[labels]


def edge_connected(invalid: Array) -> Array:
    """The subset of ``invalid`` reachable from the frame border — i.e. cutout padding.

    Morphological reconstruction rather than labelling: seed with the border's invalid pixels
    and dilate within the mask until it stops growing. Each pass is a handful of whole-array
    boolean ops, which is what makes this affordable on the bake's 827k stamps — walking
    components there would cost hours of Python. :func:`analyse_validity` does the full
    component pass on the sample-sized reporting path instead.
    """
    reached = np.zeros_like(invalid)
    reached[0, :] = invalid[0, :]
    reached[-1, :] = invalid[-1, :]
    reached[:, 0] = invalid[:, 0]
    reached[:, -1] = invalid[:, -1]
    while True:
        grown = reached.copy()
        grown[:-1, :] |= reached[1:, :]
        grown[1:, :] |= reached[:-1, :]
        grown[:, :-1] |= reached[:, 1:]
        grown[:, 1:] |= reached[:, :-1]
        grown &= invalid
        if np.array_equal(grown, reached):
            return reached
        reached = grown


def validity_mask(image: Array) -> Array:
    """``(H, W)`` bool for a ``(C, H, W)`` stamp — ``True`` where the pixel is real.

    The hot path: :func:`galaxy_jepa.data.cache.fit_normalise` and the cache bake call this
    per stamp, so it stays fully vectorised and never walks components.
    """
    return ~_invalid_mask(image)


def invalid_planes(image: Array) -> tuple[Array, Array]:
    """``(edge, interior)`` invalid masks — the two populations, kept apart.

    The cache bake stores these as separate bitplanes on purpose. Edge padding may be avoided
    by the mask sampler freely; interior regions include saturated cores, which sit at the
    galaxy centre where bulge structure lives, so avoiding *those* carries a morphological
    cost that edge padding does not. Keeping the planes apart means that call can be made
    later without re-baking the corpus.
    """
    invalid = _invalid_mask(image)
    edge = edge_connected(invalid)
    return edge, invalid & ~edge


def _label(invalid: Array) -> tuple[Array, int]:
    """4-connected component labels for ``invalid`` → ``(labels, n)``; label 0 is background.

    Row-run labelling with union-find, not a per-pixel flood fill and not
    ``scipy.ndimage.label`` (scipy is an optional dependency here). The runs matter: a padded
    strip is ~256 runs rather than ~25,000 pixels, which is the difference between this being
    affordable on the bake's 827k stamps and not.
    """
    height = invalid.shape[0]
    parent: list[int] = [0]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    labels = np.zeros(invalid.shape, dtype=np.int32)
    prev_runs: list[tuple[int, int, int]] = []  # (start, stop, label) on the row above
    for r in range(height):
        row = invalid[r]
        if not row.any():
            prev_runs = []
            continue
        edges = np.flatnonzero(np.diff(np.concatenate(([False], row, [False])).astype(np.int8)))
        runs: list[tuple[int, int, int]] = []
        for start, stop in zip(edges[::2], edges[1::2], strict=True):
            overlapping = [lab for (s, e, lab) in prev_runs if s < stop and start < e]
            if overlapping:
                lab = min(find(x) for x in overlapping)
                for other in overlapping:
                    union(lab, other)
            else:
                lab = len(parent)
                parent.append(lab)
            labels[r, start:stop] = lab
            runs.append((int(start), int(stop), lab))
        prev_runs = runs

    if len(parent) == 1:
        return labels, 0
    roots = np.array([find(i) for i in range(len(parent))], dtype=np.int32)
    order = {root: i + 1 for i, root in enumerate(sorted(set(roots[1:].tolist())))}
    remap = np.zeros(len(parent), dtype=np.int32)
    for i in range(1, len(parent)):
        remap[i] = order[int(roots[i])]
    return remap[labels], len(order)


def _components(invalid: Array) -> list[Array]:
    """Connected components of ``invalid``, each as a boolean mask."""
    labels, n = _label(invalid)
    return [labels == i for i in range(1, n + 1)]


def analyse_validity(image: Array) -> StampValidity:
    """Full analysis: the validity mask plus a classified record per constant region.

    Walks components, so it is the reporting/bake path rather than the per-epoch one — use
    :func:`validity_mask` where only the mask is needed.
    """
    invalid = _invalid_mask(image)
    height, width = invalid.shape
    centre = np.array([(height - 1) / 2.0, (width - 1) / 2.0])
    regions: list[ConstantRegion] = []
    for comp in _components(invalid):
        rows, cols = np.nonzero(comp)
        touches = bool(
            rows.min() == 0
            or cols.min() == 0
            or rows.max() == height - 1
            or cols.max() == width - 1
        )
        centroid = np.array([rows.mean(), cols.mean()])
        regions.append(
            ConstantRegion(
                n_pixels=int(comp.sum()),
                touches_edge=touches,
                height=int(rows.max() - rows.min() + 1),
                width=int(cols.max() - cols.min() + 1),
                centroid_radius_px=float(np.hypot(*(centroid - centre))),
            )
        )
    return StampValidity(valid=~invalid, regions=tuple(regions))


def token_invalid_fraction(invalid: Array, grid_size: int) -> Array:
    """Mean-pool an **invalid** pixel mask onto the ``(G, G)`` token grid.

    The sampler thresholds on invalid fraction, so that is the sense carried here; pass
    ``~validity_mask(image)``, or one of :func:`invalid_planes`' two planes.
    """
    mask = np.asarray(invalid, dtype=np.float64)
    height, width = mask.shape
    if height % grid_size or width % grid_size:
        raise ValueError(f"stamp {height}x{width} does not divide into a {grid_size} token grid")
    patch_h, patch_w = height // grid_size, width // grid_size
    return mask.reshape(grid_size, patch_h, grid_size, patch_w).mean(axis=(1, 3))
