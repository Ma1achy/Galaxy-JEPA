"""The cost spine — extract the frozen embeddings **once**, slice them per feature.

The whole probing battery (every feature's linear probe, the five negative controls, the
five nuisances, the entanglement Gram, the uncertainty axis) reads off *one* embedding
matrix per encoder. Re-encoding per feature would be the dominant cost; the design's "cheap
once the axis exists" only holds if the encode happens once. So:

* :func:`extract_matrix` runs the frozen encoder over a split **once** → an
  :class:`EmbeddingMatrix` keyed by ``object_id`` (label-free, unlike the slice's
  ``logistic.extract_embeddings`` which is welded to the featured-ness label).
* :class:`LabelProvider` maps ``object_id`` → per-feature vote fractions / binary labels and
  → per-nuisance values, from the metadata rows. It is the one place the GZ2 vote-fraction
  columns and the nuisance columns are read, so adding the full fixed-tree feature set later
  (the 2C build flag) is a config change here, not new plumbing.
* :func:`feature_embeddings` slices the matrix + a label vector into the established
  :class:`~galaxy_jepa.probing.logistic.Embeddings` the linear-probe machinery already
  consumes — so the ladder reuses ``probe_auc_ci`` / ``probe_direction`` unchanged.

The encoder is asserted **frozen** on entry (the probing freeze boundary). This module never
imports ``objectives`` — it consumes a frozen ``Encoder`` + the metadata rows.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from galaxy_jepa.core.encoder import Encoder, assert_frozen
from galaxy_jepa.data.metadata import FEATURED_FRACTION_COL
from galaxy_jepa.probing.logistic import Embeddings
from galaxy_jepa.probing.schemes import (
    DEFAULT_CONSENSUS_GATE,
    FeatureScheme,
    eligible_ids,
)

__all__ = [
    "EmbeddingMatrix",
    "extract_matrix",
    "LabelProvider",
    "feature_embeddings",
    "feature_ids",
    "DEFAULT_FEATURE_COLS",
    "DEFAULT_NUISANCE_COLS",
    "NUISANCE_FLAG_COLS",
]

# The slice's single feature. The full dissertation-fixed GZ2 tree (design 2C) extends this
# mapping — feature name → RAW vote-fraction column — over the columns now in the metadata
# pull (data/metadata.gz2_vote_columns: every t01–t11 answer's `_fraction`). The default now
# points at the RAW t01 a02 fraction via FEATURED_FRACTION_COL (the debiased column is no
# longer pulled — it injects z into the target).
DEFAULT_FEATURE_COLS: dict[str, str] = {"featured": FEATURED_FRACTION_COL}

# The five nuisances (design 3D-i), all from the existing metadata join. Keys are the *logical*
# nuisance names (what a report says); values are the *probe-corpus metadata column* names. SNR
# is `snr_r`, not `snr`: every photometric column in the corpus carries its band suffix
# (`petroRad_r`, `modelMag_r`, `psfWidth_r`) and `data.pull.with_derived_columns` — the single
# derivation site — writes `snr_r`. A caller with a different schema passes its own mapping.
DEFAULT_NUISANCE_COLS: dict[str, str] = {
    "redshift": "specz",
    "magnitude": "modelMag_r",
    "size": "petroRad_r",
    "snr": "snr_r",
    "psf": "psfWidth_r",
}

# Logical nuisance name → the corpus column that *invalidates* its measurement. A nuisance is only
# a fair control where the quantity describes the image the encoder actually saw: the probe corpus
# holds 2,143 galaxies whose `petroRad_r` is larger than the 256 px stamp (`size`), so for those
# rows the column describes a galaxy the encoder only ever saw a fragment of. Worse for the
# control, they are systematically bright, nearby and featured (mean featured fraction 0.66 against
# 0.32) — leaving them in loads the top half of the median split with disc galaxies and makes
# "size" read as morphology, firing a false nuisance-competitive trigger. The rows stay in the
# corpus and in every other probe — flagged, never dropped (`docs/spec/data.md`).
NUISANCE_FLAG_COLS: dict[str, str] = {"size": "petrorad_suspect"}


@dataclasses.dataclass(frozen=True)
class EmbeddingMatrix:
    """Frozen-encoder embeddings for a split, keyed by ``object_id`` — extracted once."""

    object_ids: np.ndarray  # (N,) int64, the row order
    x: np.ndarray  # (N, D)
    encoder_name: str

    def __post_init__(self) -> None:
        if self.object_ids.shape[0] != self.x.shape[0]:
            raise ValueError("object_ids and x must have the same length")

    @property
    def index(self) -> dict[int, int]:
        """``object_id`` → row index, for O(1) per-feature slicing."""
        return {int(o): i for i, o in enumerate(self.object_ids)}

    def rows_for(self, ids: Sequence[int]) -> np.ndarray:
        """Row indices for the present subset of ``ids``, in ``ids`` order (skips absent)."""
        idx = self.index
        return np.asarray([idx[int(o)] for o in ids if int(o) in idx], dtype=np.int64)


@torch.no_grad()
def extract_matrix(
    encoder: Encoder,
    dataset: Dataset,
    *,
    device: str = "cpu",
    batch_size: int = 128,
) -> EmbeddingMatrix:
    """Run the frozen encoder over ``dataset`` **once** → an ``EmbeddingMatrix`` (no labels).

    The dataset must yield ``image`` and ``object_id`` per item (``StampDataset`` does);
    labels are *not* required here — they are attached later, per feature, by
    :func:`feature_embeddings`. ``assert_frozen`` is the freeze boundary, as in the probe.
    """
    assert_frozen(encoder)
    module = encoder
    if isinstance(module, torch.nn.Module):
        module.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    xs: list[np.ndarray] = []
    oids: list[int] = []
    for batch in loader:
        if "object_id" not in batch:
            raise ValueError("extract_matrix needs 'object_id' per item (use StampDataset)")
        emb = encoder.encode(batch["image"].float().to(device))
        xs.append(emb.cpu().numpy())
        oids.extend(int(o) for o in batch["object_id"])
    if not xs:
        raise ValueError("dataset produced no items — nothing to extract")
    return EmbeddingMatrix(
        object_ids=np.asarray(oids, dtype=np.int64),
        x=np.concatenate(xs),
        encoder_name=getattr(encoder, "name", "encoder"),
    )


class LabelProvider:
    """Maps ``object_id`` → per-feature labels / vote fractions and → per-nuisance values.

    Built once from the metadata rows. The single reader of the GZ2 vote-fraction columns
    and the nuisance columns, so the full fixed-tree feature set (2C) and any nuisance-schema
    difference are configured here, not threaded through the ladder.
    """

    def __init__(
        self,
        rows: Mapping[int, Mapping[str, Any]],
        *,
        feature_cols: Mapping[str, str] | None = None,
        nuisance_cols: Mapping[str, str] | None = None,
        nuisance_flag_cols: Mapping[str, str] | None = None,
        threshold: float = 0.5,
        scheme: FeatureScheme | None = None,
        population: str = "full",
        vote_count_min: float,
        consensus_gate: float = DEFAULT_CONSENSUS_GATE,
    ):
        if scheme is not None and feature_cols is None:
            feature_cols = scheme.feature_cols()
        if population not in ("full", "conditional"):
            raise ValueError(f"population must be 'full' or 'conditional', got {population!r}")
        self.rows = {int(k): dict(v) for k, v in rows.items()}
        self.feature_cols = dict(feature_cols or DEFAULT_FEATURE_COLS)
        self.nuisance_cols = dict(nuisance_cols or DEFAULT_NUISANCE_COLS)
        # Part of the schema, so it travels with `nuisance_cols`: a caller mapping the nuisances
        # onto its own columns is also the one who knows which of them can fail to be measured.
        # `None` takes the corpus default; `{}` is an explicit statement that none can.
        self.nuisance_flag_cols = dict(
            NUISANCE_FLAG_COLS if nuisance_flag_cols is None else nuisance_flag_cols
        )
        self.threshold = float(threshold)
        self.scheme = scheme
        self.population = population
        self.vote_count_min = float(vote_count_min)
        self.consensus_gate = float(consensus_gate)

    @property
    def features(self) -> list[str]:
        return list(self.feature_cols)

    def eligible(self, feature: str, ids: Sequence[int]) -> list[int]:
        """The subset of ``ids`` that enters ``feature``'s probe set under this population.

        Without a scheme this is the identity — the plumbing path is unchanged. With one, the
        per-feature vote-count floor always applies, and the conditional-population gate applies
        only when this provider was built with ``population="conditional"``. Running one provider
        each way is the D14 comparison; a single permanently-conditional provider would be the
        hard mask the design forbids.
        """
        if self.scheme is None:
            return [int(o) for o in ids]
        spec = self.scheme.by_name.get(feature)
        if spec is None:
            return [int(o) for o in ids]
        return eligible_ids(
            self.rows,
            spec,
            ids,
            vote_count_min=self.vote_count_min,
            conditional=self.population == "conditional",
            consensus_gate=self.consensus_gate,
        )

    @property
    def nuisances(self) -> list[str]:
        return list(self.nuisance_cols)

    def with_population(self, population: str) -> LabelProvider:
        """A sibling provider over the same rows but the other population definition.

        The D14 comparison is *two runs of the same ladder*, not one masked run, so the two
        populations must differ in exactly one field and nothing else — hence deriving the
        sibling here rather than rebuilding it at the call site.
        """
        return LabelProvider(
            self.rows,
            feature_cols=self.feature_cols,
            nuisance_cols=self.nuisance_cols,
            nuisance_flag_cols=self.nuisance_flag_cols,
            threshold=self.threshold,
            scheme=self.scheme,
            population=population,
            vote_count_min=self.vote_count_min,
            consensus_gate=self.consensus_gate,
        )

    def _column(self, ids: Sequence[int], col: str) -> np.ndarray:
        return np.asarray(
            [float(self.rows.get(int(o), {}).get(col, np.nan)) for o in ids], dtype=np.float64
        )

    def vote_fraction(self, feature: str, ids: Sequence[int]) -> np.ndarray:
        """The GZ2 vote fraction for ``feature`` over ``ids`` (NaN where the row is missing)."""
        return self._column(ids, self.feature_cols[feature])

    def binary_label(self, feature: str, ids: Sequence[int]) -> np.ndarray:
        """Binary label for ``feature``: ``1`` iff the vote fraction ≥ the threshold."""
        return (self.vote_fraction(feature, ids) >= self.threshold).astype(np.int64)

    def nuisance_value(self, name: str, ids: Sequence[int]) -> np.ndarray:
        """The raw continuous nuisance value for ``name`` over ``ids``."""
        return self._column(ids, self.nuisance_cols[name])

    def nuisance_valid(self, name: str, ids: Sequence[int]) -> np.ndarray:
        """Boolean mask over ``ids``: rows whose ``name`` measurement is usable.

        All-true for a nuisance with no flag column in :data:`NUISANCE_FLAG_COLS`. Where there
        is one, a missing column **raises** rather than waving the rows through: silently
        running an uncorrected radius probe is precisely the skipped control the invariants
        forbid. ``data.pull.backfill_derived`` re-derives the flag on a corpus already pulled.
        """
        col = self.nuisance_flag_cols.get(name)
        if col is None:
            return np.ones(len(ids), dtype=bool)
        if not any(col in row for row in self.rows.values()):
            raise KeyError(
                f"nuisance {name!r} needs the {col!r} flag column and the corpus has none, so "
                f"its control cannot exclude unusable rows. Re-derive it with "
                f"galaxy_jepa.data.pull.backfill_derived(corpus_dir)."
            )
        return np.asarray([not self._flagged(o, col) for o in ids], dtype=bool)

    def _flagged(self, oid: int, col: str) -> bool:
        """Unusable unless the row says plainly that it is not — an absent row, an empty cell
        or an unreadable value all count as flagged, which is the conservative direction."""
        raw = self.rows.get(int(oid), {}).get(col, 1)
        try:
            return bool(int(float(raw)))
        except (TypeError, ValueError):
            return True

    def nuisance_label(self, name: str, ids: Sequence[int]) -> np.ndarray:
        """Binarised nuisance for the parallel-probe AUC — **median split** (placeholder).

        UNDER-SPECIFIED (surfaced for the stats grounding): the design reports a
        "nuisance-AUC", but z / magnitude / size / SNR / PSF are continuous. The
        binarisation (median split vs tertiles vs a regression-R² variant) is not yet
        finalised; the median split is a defensible, scale-free default that keeps the
        nuisance probe using the *same* AUC machinery as the morphology probe.
        """
        v = self.nuisance_value(name, ids)
        med = float(np.nanmedian(v))
        return (v >= med).astype(np.int64)


def feature_ids(
    matrix: EmbeddingMatrix,
    labels: LabelProvider,
    feature: str,
    ids: Sequence[int],
) -> list[int]:
    """The exact object ids :func:`feature_embeddings` will keep for ``feature``, in order.

    **Every vector that is co-indexed with a feature's embedding rows must come from here.**
    The eligibility filter (vote-count floor, conditional population) shortens the embedding
    matrix per feature, so a label vector built from the raw id list is silently *longer* and
    misaligned — which surfaces as an sklearn length error at best, and as a label vector
    quietly paired with the wrong galaxies at worst.
    """
    return labels.eligible(feature, [int(o) for o in ids if int(o) in matrix.index])


def feature_embeddings(
    matrix: EmbeddingMatrix,
    labels: LabelProvider,
    feature: str,
    ids: Sequence[int],
) -> Embeddings:
    """Slice ``matrix`` + a per-feature label vector into an ``Embeddings`` (the probe input).

    Selects the rows of ``matrix`` whose ``object_id`` is in ``ids`` (and present), attaches
    the binary label and the vote fraction for ``feature`` — so the linear-probe machinery
    (``probe_auc_ci`` / ``probe_direction``) runs unchanged on any feature.
    """
    present = feature_ids(matrix, labels, feature, ids)
    rows = matrix.rows_for(present)
    x = matrix.x[rows]
    y = labels.binary_label(feature, present)
    fraction = labels.vote_fraction(feature, present)
    return Embeddings(x=x, y=y, fraction=fraction)
