"""Exploratory information-theoretic estimators (strictly distinct from AUC / transfer ratios).

Prior to November 2026, Galaxy-JEPA contained **no** mutual-information estimator — existing
``almagest_axes.py`` cross-concept ratios ``T_{X->Y} = (AUC(s_X, Y) - 0.5) / (AUC(s_Y, Y) - 0.5)``
are predictive transfer ratios, never mutual information. This module provides standalone,
independently benchmarked **exploratory** MI / conditional-MI estimators with explicit units
(``"nats"`` or ``"bits"``):

* :func:`discrete_entropy`, :func:`discrete_mi`, :func:`discrete_cmi` — exact plug-in discrete
  entropy, mutual information ``I(X; Y)``, and conditional mutual information ``I(X; Y | Z)``.
* :func:`variational_mi_lower_bound` — held-out variational lower bound
  ``I(Y; Z) >= H(Y) + E[log q_theta(Y | Z)]`` fitted strictly on ``train`` and evaluated on
  held-out ``test`` (unclipped, because finite-sample lower bounds can fluctuate negative under
  the null and clipping breaks null calibration).
* :func:`ksg_mi` — Kraskov–Stögbauer–Grassberger (KSG-1) k-nearest-neighbour continuous MI
  estimator using Chebyshev (``L_inf``) distances via ``scipy.spatial.cKDTree``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

Units = Literal["nats", "bits"]

__all__ = [
    "Units",
    "convert_units",
    "discrete_entropy",
    "discrete_mi",
    "discrete_cmi",
    "VariationalMIBound",
    "variational_mi_lower_bound",
    "ksg_mi",
]


def convert_units(value_nats: float, units: Units = "nats") -> float:
    """Convert an information quantity from natural logs (nats) to ``units``."""
    if units == "nats":
        return float(value_nats)
    if units == "bits":
        return float(value_nats / math.log(2.0))
    raise ValueError(f"unknown units {units!r}; expected 'nats' or 'bits'")


def _as_1d_discrete(arr: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D discrete labels, got shape {a.shape}")
    if a.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return a


def discrete_entropy(x: np.ndarray, *, units: Units = "nats") -> float:
    """Plug-in Shannon entropy ``H(X)`` for a 1-D discrete array."""
    a = _as_1d_discrete(x, "x")
    _, counts = np.unique(a, return_counts=True)
    probs = counts.astype(np.float64) / float(a.size)
    h_nats = float(-np.sum(probs * np.log(probs)))
    return convert_units(h_nats, units)


def discrete_mi(x: np.ndarray, y: np.ndarray, *, units: Units = "nats") -> float:
    """Exact plug-in discrete mutual information ``I(X; Y) = H(X) + H(Y) - H(X, Y)``."""
    ax = _as_1d_discrete(x, "x")
    ay = _as_1d_discrete(y, "y")
    if ax.size != ay.size:
        raise ValueError(f"x ({ax.size}) and y ({ay.size}) must have equal length")
    _, ix = np.unique(ax, return_inverse=True)
    _, iy = np.unique(ay, return_inverse=True)
    ny = int(iy.max()) + 1
    joint = ix.astype(np.int64) * ny + iy.astype(np.int64)
    mi_nats = (
        discrete_entropy(ix, units="nats")
        + discrete_entropy(iy, units="nats")
        - discrete_entropy(joint, units="nats")
    )
    return convert_units(max(mi_nats, 0.0) if abs(mi_nats) < 1e-14 else mi_nats, units)


def discrete_cmi(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    units: Units = "nats",
) -> float:
    """Exact plug-in discrete CMI ``I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z)``."""
    ax = _as_1d_discrete(x, "x")
    ay = _as_1d_discrete(y, "y")
    az = _as_1d_discrete(z, "z")
    if not (ax.size == ay.size == az.size):
        raise ValueError("x, y, and z must have equal length")
    _, ix = np.unique(ax, return_inverse=True)
    _, iy = np.unique(ay, return_inverse=True)
    _, iz = np.unique(az, return_inverse=True)
    ny = int(iy.max()) + 1
    nz = int(iz.max()) + 1
    xz = ix.astype(np.int64) * nz + iz.astype(np.int64)
    yz = iy.astype(np.int64) * nz + iz.astype(np.int64)
    xyz = (ix.astype(np.int64) * ny + iy.astype(np.int64)) * nz + iz.astype(np.int64)
    cmi_nats = (
        discrete_entropy(xz, units="nats")
        + discrete_entropy(yz, units="nats")
        - discrete_entropy(iz, units="nats")
        - discrete_entropy(xyz, units="nats")
    )
    return convert_units(max(cmi_nats, 0.0) if abs(cmi_nats) < 1e-14 else cmi_nats, units)


@dataclasses.dataclass(frozen=True)
class VariationalMIBound:
    """Held-out variational MI lower bound ``I(Y; Z) >= H(Y) + E[log q_theta(Y | Z)]``."""

    bound: float
    entropy_y: float
    mean_log_q: float
    units: Units
    n_train: int
    n_test: int


def variational_mi_lower_bound(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_test: np.ndarray,
    y_test: np.ndarray,
    *,
    c: float = 1.0,
    units: Units = "nats",
    eps: float = 1e-12,
) -> VariationalMIBound:
    """Estimate the Barber–Agakov variational lower bound on ``I(Y; Z)`` for discrete ``Y``.

    Fits a standardised L2 logistic decoder ``q_theta(Y | Z)`` strictly on ``(z_train, y_train)``
    and evaluates ``H_test(Y) + mean_test log q_theta(y_i | z_i)`` on held-out ``(z_test, y_test)``.
    The bound is intentionally **unclipped** so finite-sample fluctuations under the null remain
    honest and calibrated.
    """
    ztr = np.asarray(z_train, dtype=np.float64)
    zte = np.asarray(z_test, dtype=np.float64)
    if ztr.ndim == 1:
        ztr = ztr[:, None]
    if zte.ndim == 1:
        zte = zte[:, None]
    ytr = _as_1d_discrete(y_train, "y_train")
    yte = _as_1d_discrete(y_test, "y_test")
    if ztr.shape[0] != ytr.size or zte.shape[0] != yte.size or ztr.shape[1] != zte.shape[1]:
        raise ValueError("mismatched shapes between train/test features and labels")
    if not (np.isfinite(ztr).all() and np.isfinite(zte).all()):
        raise ValueError("variational_mi_lower_bound requires finite features")

    classes = np.unique(ytr)
    if classes.size < 2:
        raise ValueError("y_train must contain at least 2 classes")

    scaler = StandardScaler().fit(ztr)
    clf = LogisticRegression(C=c, max_iter=2000).fit(scaler.transform(ztr), ytr)
    probs = np.clip(clf.predict_proba(scaler.transform(zte)), eps, 1.0)
    class_to_col = {cls: idx for idx, cls in enumerate(clf.classes_)}
    if any(lbl not in class_to_col for lbl in np.unique(yte)):
        raise ValueError("y_test contains class label absent from y_train")
    col_idx = np.array([class_to_col[lbl] for lbl in yte], dtype=np.int64)
    mean_log_q_nats = float(np.mean(np.log(probs[np.arange(yte.size), col_idx])))
    h_y_nats = discrete_entropy(yte, units="nats")
    bound_nats = h_y_nats + mean_log_q_nats
    return VariationalMIBound(
        bound=convert_units(bound_nats, units),
        entropy_y=convert_units(h_y_nats, units),
        mean_log_q=convert_units(mean_log_q_nats, units),
        units=units,
        n_train=int(ytr.size),
        n_test=int(yte.size),
    )


def ksg_mi(
    x: np.ndarray,
    y: np.ndarray,
    *,
    k: int = 5,
    units: Units = "nats",
    jitter: float = 0.0,
    seed: int = 0,
) -> float:
    """Kraskov–Stögbauer–Grassberger (KSG-1) continuous mutual information estimator.

    Uses exact Chebyshev (``L_inf``) k-NN radii in the joint ``(X, Y)`` space and counts
    strictly-closer marginal neighbours ``n_x, n_y`` via ``scipy.spatial.cKDTree``:
        I_hat(X; Y) = digamma(k) + digamma(N) - mean(digamma(n_x + 1) + digamma(n_y + 1)).

    Unclipped (finite-sample null estimates fluctuate around 0). Raises :class:`ValueError` on
    invalid inputs or ``N <= k``.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.ndim == 1:
        xa = xa[:, None]
    if ya.ndim == 1:
        ya = ya[:, None]
    if xa.ndim != 2 or ya.ndim != 2 or xa.shape[0] != ya.shape[0]:
        raise ValueError(f"incompatible shapes x={xa.shape}, y={ya.shape}")
    n = xa.shape[0]
    if k < 1 or n <= k:
        raise ValueError(f"require 1 <= k < N, got k={k}, N={n}")
    if not (np.isfinite(xa).all() and np.isfinite(ya).all()):
        raise ValueError("ksg_mi requires finite x and y")

    # Standardise each coordinate by its empirical spread so Chebyshev radii are scale-balanced
    # (invariant to positive coordinate rescaling; constant columns stay zero).
    sx = xa.std(axis=0)
    sy = ya.std(axis=0)
    x_work = np.where(sx > 0.0, (xa - xa.mean(axis=0)) / np.where(sx > 0.0, sx, 1.0), 0.0)
    y_work = np.where(sy > 0.0, (ya - ya.mean(axis=0)) / np.where(sy > 0.0, sy, 1.0), 0.0)

    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        x_work = x_work + rng.normal(scale=jitter, size=x_work.shape)
        y_work = y_work + rng.normal(scale=jitter, size=y_work.shape)

    # Constant variable carries zero MI regardless of k
    if np.all(sx == 0.0) or np.all(sy == 0.0):
        return 0.0

    joint = np.concatenate([x_work, y_work], axis=1)
    tree_xy = cKDTree(joint)
    tree_x = cKDTree(x_work)
    tree_y = cKDTree(y_work)

    # k+1 because index 0 is the point itself at distance 0
    dists, _ = tree_xy.query(joint, k=k + 1, p=np.inf)
    eps = np.nextafter(dists[:, k], 0.0)

    # Count points strictly within eps in each marginal (excluding self -> subtract 1)
    nx = np.empty(n, dtype=np.int64)
    ny = np.empty(n, dtype=np.int64)
    for i in range(n):
        r = float(eps[i])
        if r <= 0.0:
            nx[i] = 0
            ny[i] = 0
        else:
            nx[i] = max(len(tree_x.query_ball_point(x_work[i], r=r, p=np.inf)) - 1, 0)
            ny[i] = max(len(tree_y.query_ball_point(y_work[i], r=r, p=np.inf)) - 1, 0)

    mi_nats = float(digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1)))
    return convert_units(mi_nats, units)
