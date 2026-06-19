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

__all__ = [
    "EmbeddingMatrix",
    "extract_matrix",
    "LabelProvider",
    "feature_embeddings",
    "DEFAULT_FEATURE_COLS",
    "DEFAULT_NUISANCE_COLS",
]

# The slice's single feature. The full dissertation-fixed GZ2 tree (design 2C) extends this
# mapping — feature name → RAW vote-fraction column — over the columns now in the metadata
# pull (data/metadata.gz2_vote_columns: every t01–t11 answer's `_fraction`). The default now
# points at the RAW t01 a02 fraction via FEATURED_FRACTION_COL (the debiased column is no
# longer pulled — it injects z into the target).
DEFAULT_FEATURE_COLS: dict[str, str] = {"featured": FEATURED_FRACTION_COL}

# The five nuisances (design 3D-i), all from the existing metadata join. The column names are
# the *probe-corpus* metadata keys; a caller with a different schema (e.g. the fixture) passes
# its own mapping.
DEFAULT_NUISANCE_COLS: dict[str, str] = {
    "redshift": "specz",
    "magnitude": "modelMag_r",
    "size": "petroRad_r",
    "snr": "snr",
    "psf": "psfWidth_r",
}


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
        threshold: float = 0.5,
    ):
        self.rows = {int(k): dict(v) for k, v in rows.items()}
        self.feature_cols = dict(feature_cols or DEFAULT_FEATURE_COLS)
        self.nuisance_cols = dict(nuisance_cols or DEFAULT_NUISANCE_COLS)
        self.threshold = float(threshold)

    @property
    def features(self) -> list[str]:
        return list(self.feature_cols)

    @property
    def nuisances(self) -> list[str]:
        return list(self.nuisance_cols)

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
    present = [int(o) for o in ids if int(o) in matrix.index]
    rows = matrix.rows_for(present)
    x = matrix.x[rows]
    y = labels.binary_label(feature, present)
    fraction = labels.vote_fraction(feature, present)
    return Embeddings(x=x, y=y, fraction=fraction)
