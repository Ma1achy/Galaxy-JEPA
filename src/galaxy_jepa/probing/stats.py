"""Shared statistical primitives for probing and downstream analysis scripts.

Centralises numerical routines that were previously duplicated across one-off analysis scripts
(and where bugs such as undamped Newton divergence on rare binary targets or mismatched
bootstrap estimators appeared):

* :func:`damped_logistic` — L2-penalised logistic regression via Armijo-backtracking damped
  Newton with clipped logits and explicit convergence checks.
* :func:`residualise_ranks` and :func:`partial_spearman` — rank-based partial correlation
  controlling for one or more continuous nuisances.
* :func:`paired_pearson_bootstrap` — percentile confidence interval for Pearson correlation on
  pre-residualised arrays (matched to :func:`partial_spearman` rather than re-ranking residuals).
* :func:`write_json_atomic` — RFC-8259 atomic JSON persistence (sanitises non-finite floats to
  ``null`` and writes with ``allow_nan=False`` before ``os.replace``).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

__all__ = [
    "damped_logistic",
    "residualise_ranks",
    "partial_spearman",
    "paired_pearson_bootstrap",
    "sanitise_json",
    "write_json_atomic",
]


def _sigmoid_clipped(z: np.ndarray) -> np.ndarray:
    zc = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-zc))


def _logistic_loss(x_aug: np.ndarray, y: np.ndarray, beta: np.ndarray, reg: float) -> float:
    z = np.clip(x_aug @ beta, -35.0, 35.0)
    # log(1 + exp(z)) - y * z computed stably
    nll = float(np.sum(np.logaddexp(0.0, z) - y * z))
    pen = 0.5 * reg * float(np.dot(beta[:-1], beta[:-1]))
    return nll + pen


def damped_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    c: float = 1.0,
    max_iter: int = 80,
    tol: float = 1e-7,
) -> tuple[np.ndarray, float]:
    """Fit L2-penalised binary logistic regression via Armijo-backtracking damped Newton.

    Minimises ``sum_i log(1 + exp(-y_signed_i (x_i @ w + b))) + (1 / (2 C)) ||w||_2^2`` (matching
    ``sklearn.linear_model.LogisticRegression(C=c, penalty="l2", fit_intercept=True)``).

    Returns ``(w, b)`` in float64. Raises :class:`ValueError` on invalid inputs or non-convergence.
    """
    if c <= 0.0 or not math.isfinite(c):
        raise ValueError(f"c must be finite and > 0, got {c!r}")
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64).ravel()
    if x64.ndim != 2 or x64.shape[0] != y64.size:
        raise ValueError(f"x shape {x64.shape} does not match y length {y64.size}")
    if not (np.isfinite(x64).all() and np.isfinite(y64).all()):
        raise ValueError("damped_logistic requires finite x and y")
    uniq = np.unique(y64)
    if uniq.size < 2 or not np.all((uniq == 0.0) | (uniq == 1.0)):
        raise ValueError(f"y must contain both binary classes {{0, 1}}, got {uniq.tolist()}")

    n, d = x64.shape
    reg = 1.0 / float(c)
    x_aug = np.empty((n, d + 1), dtype=np.float64)
    x_aug[:, :d] = x64
    x_aug[:, d] = 1.0

    # Initialise intercept at log-odds so rare classes start near the optimum.
    p0 = float(np.clip(y64.mean(), 1e-6, 1.0 - 1e-6))
    beta = np.zeros(d + 1, dtype=np.float64)
    beta[d] = math.log(p0 / (1.0 - p0))
    reg_diag = np.full(d + 1, reg, dtype=np.float64)
    reg_diag[d] = 0.0

    loss = _logistic_loss(x_aug, y64, beta, reg)
    converged = False
    for _ in range(max_iter):
        p = _sigmoid_clipped(x_aug @ beta)
        grad = x_aug.T @ (p - y64) + reg_diag * beta
        if float(np.linalg.norm(grad, ord=np.inf)) <= tol:
            converged = True
            break
        w_diag = np.maximum(p * (1.0 - p), 1e-12)
        hess = (x_aug.T * w_diag) @ x_aug
        hess[np.diag_indices(d + 1)] += reg_diag + 1e-12
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        slope = float(np.dot(grad, step))
        if slope <= 0.0:
            step = grad
            slope = float(np.dot(grad, step))

        alpha = 1.0
        accepted = False
        while alpha >= 1e-10:
            cand = beta - alpha * step
            cand_loss = _logistic_loss(x_aug, y64, cand, reg)
            if cand_loss <= loss - 1e-4 * alpha * slope:
                beta = cand
                loss = cand_loss
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
        if float(np.linalg.norm(alpha * step, ord=np.inf)) <= tol:
            converged = True
            break

    if not (converged and np.isfinite(beta).all()):
        p = _sigmoid_clipped(x_aug @ beta)
        grad_inf = float(np.linalg.norm(x_aug.T @ (p - y64) + reg_diag * beta, ord=np.inf))
        if not (np.isfinite(beta).all() and grad_inf <= 1e-4):
            raise ValueError(
                f"damped_logistic failed to converge after {max_iter} iterations "
                f"(||grad||_inf={grad_inf:.3e})"
            )
    return beta[:d].copy(), float(beta[d])


def _design_from_nuisance(nuisance: np.ndarray) -> np.ndarray:
    arr = np.asarray(nuisance, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"nuisance must be 1-D or 2-D, got shape {arr.shape}")
    ranked = np.column_stack([rankdata(arr[:, j]) for j in range(arr.shape[1])])
    return np.column_stack([np.ones(arr.shape[0], dtype=np.float64), ranked])


def residualise_ranks(values: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
    """Rank-transform ``values`` and ``nuisance``, then OLS-residualise ``rank(values)`` on
    ``[1, rank(nuisance)]``. Returns float64 residuals of the same length."""
    v = np.asarray(values, dtype=np.float64).ravel()
    z = np.asarray(nuisance, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    if v.size != z.shape[0]:
        raise ValueError(f"values length {v.size} does not match nuisance rows {z.shape[0]}")
    if not (np.isfinite(v).all() and np.isfinite(z).all()):
        raise ValueError("residualise_ranks requires finite values and nuisance")
    r = rankdata(v)
    design = _design_from_nuisance(z)
    coef, *_ = np.linalg.lstsq(design, r, rcond=None)
    return r - design @ coef


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    ac = a - a.mean()
    bc = b - b.mean()
    denom = float(np.linalg.norm(ac) * np.linalg.norm(bc))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(ac, bc) / denom)


def partial_spearman(x: np.ndarray, y: np.ndarray, nuisance: np.ndarray) -> float:
    """Partial Spearman rank correlation between ``x`` and ``y`` controlling for ``nuisance``.

    Computed as the Pearson correlation of OLS-residualised ranks of ``x`` and ``y`` given
    ``rank(nuisance)`` over jointly finite rows. Returns ``NaN`` if fewer than 3 finite rows
    remain or either residual is constant.
    """
    xv = np.asarray(x, dtype=np.float64).ravel()
    yv = np.asarray(y, dtype=np.float64).ravel()
    zv = np.asarray(nuisance, dtype=np.float64)
    if zv.ndim == 1:
        zv = zv[:, None]
    if not (xv.size == yv.size == zv.shape[0]):
        raise ValueError("x, y, and nuisance must have identical sample counts")
    ok = np.isfinite(xv) & np.isfinite(yv) & np.isfinite(zv).all(axis=1)
    if int(ok.sum()) < 3:
        return float("nan")
    rx = residualise_ranks(xv[ok], zv[ok])
    ry = residualise_ranks(yv[ok], zv[ok])
    return _pearson(rx, ry)


def paired_pearson_bootstrap(
    rx: np.ndarray,
    ry: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Paired percentile bootstrap CI for Pearson correlation on pre-residualised arrays.

    Matches :func:`partial_spearman` (which computes Pearson on rank-residuals) instead of
    re-ranking residuals on each bootstrap draw.
    """
    a = np.asarray(rx, dtype=np.float64).ravel()
    b = np.asarray(ry, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError(f"rx ({a.size}) and ry ({b.size}) must have equal length")
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = a.size
    if n < 3 or n_boot <= 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    asamp = a[idx]
    bsamp = b[idx]
    ac = asamp - asamp.mean(axis=1, keepdims=True)
    bc = bsamp - bsamp.mean(axis=1, keepdims=True)
    denom = np.sqrt(np.sum(ac * ac, axis=1) * np.sum(bc * bc, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        corrs = np.where(denom > 0.0, np.sum(ac * bc, axis=1) / denom, np.nan)
    finite = corrs[np.isfinite(corrs)]
    if finite.size == 0:
        return (float("nan"), float("nan"))
    lo, hi = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return (float(lo), float(hi))


def sanitise_json(obj: Any) -> Any:
    """Recursively replace non-finite floats (``NaN``, ``±inf``) with ``None`` for RFC-8259 JSON."""
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return sanitise_json(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): sanitise_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitise_json(v) for v in obj]
    return obj


def write_json_atomic(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    """Write ``payload`` as strict RFC-8259 JSON (``allow_nan=False``) via atomic ``os.replace``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    text = json.dumps(sanitise_json(payload), indent=indent, allow_nan=False) + "\n"
    tmp.write_text(text)
    os.replace(tmp, target)
    return target
