"""The bounded-capacity MLP ladder — the capacity-trap-proof R3/R4 boundary (design 2D/2F).

Fires **only on linear-failures** (a cleanly-R1/R2 feature never needs it). The MLP detects
nonlinear-but-present structure, but capacity is a trap: too weak misses real nonlinearity
(false R4), too strong decodes *anything* including the nulls (false R3). The resolution is a
**bounded capacity ladder** — sweep hidden-layer **width** (the only knob; depth / reg /
training-time fixed), but trust the verdict only *below the selectivity ceiling*, the width at
which the negative control itself starts to decode.

* **R3** iff the feature becomes decodable at some width **below** the ceiling.
* **R4** iff it only decodes at/above the ceiling (where the controls decode too) — or never.

This kills the capacity trap by construction: you cannot manufacture R3 by cranking capacity,
because past the ceiling the verdict is invalid.

**FLAGGED (2): the selectivity-ceiling "exceeds its own null" mechanics** — the only flagged
piece. The sweep, the per-width real and control AUCs, and the ceiling-relative R3/R4 logic
are all built; :func:`selectivity_ceiling` carries the placeholder predicate.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
from torch import nn

from galaxy_jepa.probing.logistic import Embeddings

__all__ = [
    "MLPProbe",
    "mlp_auc",
    "SweepRow",
    "capacity_sweep",
    "selectivity_ceiling",
    "rung_from_sweep",
]


class MLPProbe(nn.Module):
    """A small MLP probe: ``depth`` hidden layers of ``width``, GELU, single-logit head.

    Width is the swept capacity knob; depth is held fixed so the sweep is a clean, one-
    dimensional, interpretable capacity axis (design 2D).
    """

    def __init__(self, input_dim: int, *, width: int, depth: int = 1):
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for _ in range(max(depth, 1)):
            layers += [nn.Linear(d, width), nn.GELU()]
            d = width
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _standardise(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0) + 1e-8
    return (train_x - mean) / std, (test_x - mean) / std


def mlp_auc(
    train: Embeddings,
    test: Embeddings,
    *,
    width: int,
    depth: int = 1,
    weight_decay: float = 1e-4,
    epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 0,
) -> float:
    """Fit an MLP probe (standardised features, fixed recipe) and return ROC-AUC on ``test``."""
    from sklearn.metrics import roc_auc_score

    if len(np.unique(train.y)) < 2 or len(np.unique(test.y)) < 2:
        return 0.5
    torch.manual_seed(seed)
    xtr, xte = _standardise(train.x, test.x)
    xt = torch.as_tensor(xtr, dtype=torch.float32)
    yt = torch.as_tensor(train.y, dtype=torch.float32)
    model = MLPProbe(train.x.shape[1], width=width, depth=depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.as_tensor(xte, dtype=torch.float32))).numpy()
    return float(roc_auc_score(test.y, scores))


@dataclasses.dataclass(frozen=True)
class SweepRow:
    """One rung of the capacity sweep: width, the real-label AUC, and the control-label AUC."""

    width: int
    real_auc: float
    control_auc: float


def capacity_sweep(
    train: Embeddings,
    test: Embeddings,
    control_train: Embeddings,
    control_test: Embeddings,
    *,
    widths: tuple[int, ...],
    depth: int = 1,
    weight_decay: float = 1e-4,
    epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 0,
) -> list[SweepRow]:
    """Sweep hidden width, recording the real-label AUC and the negative-control AUC per width.

    ``control_*`` carries the same embeddings with a *control* label (e.g. shuffled), so the
    control AUC traces where the MLP starts decoding noise — the input to the ceiling.
    """
    rows: list[SweepRow] = []
    for width in widths:
        real = mlp_auc(
            train,
            test,
            width=width,
            depth=depth,
            weight_decay=weight_decay,
            epochs=epochs,
            lr=lr,
            seed=seed,
        )
        ctrl = mlp_auc(
            control_train,
            control_test,
            width=width,
            depth=depth,
            weight_decay=weight_decay,
            epochs=epochs,
            lr=lr,
            seed=seed,
        )
        rows.append(SweepRow(width=width, real_auc=real, control_auc=ctrl))
    return rows


def selectivity_ceiling(rows: list[SweepRow], *, null_threshold: float) -> int | None:
    """The width at which the negative control **first exceeds its own null** (design 2D).

    FLAGGED: pending stats grounding — do not finalise. Placeholder predicate: the control
    AUC exceeds ``null_threshold`` (e.g. the 95th percentile of a chance null for this probe,
    supplied by the caller from the control battery). The grounding session owns the precise
    null-distribution machinery for the breakdown point; only this predicate changes. Returns
    the first such width, or ``None`` if the control never decodes across the swept range
    (the whole range is then valid).
    """
    for row in rows:
        if row.control_auc > null_threshold:
            return row.width
    return None


def rung_from_sweep(
    rows: list[SweepRow],
    ceiling: int | None,
    *,
    decode_threshold: float,
) -> str:
    """R3 iff the feature decodes below the ceiling; else R4 (design 2D verdict logic).

    "Decodes" = real-label AUC ≥ ``decode_threshold``. Widths at or above the ceiling are
    invalid (the controls decode there too), so a feature that only crosses the bar there is
    R4 — the capacity trap closed by construction.
    """
    for row in rows:
        if ceiling is not None and row.width >= ceiling:
            break  # past the ceiling the verdict is invalid
        if row.real_auc >= decode_threshold:
            return "R3"
    return "R4"
