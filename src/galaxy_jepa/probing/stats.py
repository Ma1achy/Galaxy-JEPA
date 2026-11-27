"""Shared statistical primitives for probing and downstream analysis scripts.

Centralises numerical routines that were previously duplicated across one-off analysis scripts
(and where bugs such as undamped Newton divergence on rare binary targets, clipped-objective
derivative mismatch, or mismatched bootstrap estimators appeared):

* :func:`damped_logistic` — L2-penalised logistic regression via Armijo-backtracking damped
  Newton on the exact **unclipped** softplus objective with ``scipy.special.expit`` and strict
  gradient-residual convergence.
* :func:`residualise_ranks` and :func:`partial_spearman` — rank-based partial correlation
  controlling for one or more continuous nuisances.
* :func:`partial_spearman_bootstrap` — full paired-row bootstrap that re-ranks and
  re-residualises inside each replicate (chunked to bound memory, with relative residual-norm
  guard and explicit invalid-replicate reporting).
* :func:`paired_pearson_bootstrap` — conditional fixed-residual percentile CI for Pearson
  correlation on pre-residualised arrays.
* :func:`write_json_atomic` — RFC-8259 atomic JSON persistence (sanitises non-finite floats to
  ``null`` and writes with ``allow_nan=False`` before ``os.replace``).
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata

__all__ = [
    "logistic_objective_and_derivatives",
    "damped_logistic",
    "residualise_ranks",
    "partial_spearman",
    "BootstrapResult",
    "partial_spearman_bootstrap",
    "paired_pearson_bootstrap",
    "sanitise_json",
    "write_json_atomic",
]


def logistic_objective_and_derivatives(
    x_aug: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    reg: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Exact unclipped L2-penalised logistic loss, gradient, and Hessian.

    Minimises:
        L(w, b) = sum_i log(1 + exp(-s_i z_i)) + (reg / 2) ||w||_2^2,   s_i = 2 y_i - 1.

    Uses ``np.logaddexp(0.0, -s * z)`` and ``scipy.special.expit`` without clipping ``z``, so the
    gradient and Hessian are exact analytical derivatives of ``L`` across the entire real line
    (including extreme logits ``|z| > 35``).
    """
    z = x_aug @ beta
    signed = 2.0 * y - 1.0
    nll = float(np.sum(np.logaddexp(0.0, -signed * z)))
    w = beta[:-1]
    loss = nll + 0.5 * reg * float(np.dot(w, w))

    p = expit(z)
    reg_vec = np.empty_like(beta)
    reg_vec[:-1] = reg
    reg_vec[-1] = 0.0
    grad = x_aug.T @ (p - y) + reg_vec * beta

    # expit(z) * expit(-z) avoids catastrophic cancellation in p * (1 - p) for large +z
    w_diag = expit(z) * expit(-z)
    hess = (x_aug.T * w_diag) @ x_aug
    hess[np.diag_indices(beta.size)] += reg_vec
    return loss, grad, hess


def _logistic_loss_only(x_aug: np.ndarray, y: np.ndarray, beta: np.ndarray, reg: float) -> float:
    z = x_aug @ beta
    signed = 2.0 * y - 1.0
    nll = float(np.sum(np.logaddexp(0.0, -signed * z)))
    w = beta[:-1]
    return nll + 0.5 * reg * float(np.dot(w, w))


def damped_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    c: float = 1.0,
    max_iter: int = 80,
    tol: float = 1e-7,
) -> tuple[np.ndarray, float]:
    """Fit L2-penalised binary logistic regression via Armijo-backtracking damped Newton.

    Minimises ``sum_i softplus(-s_i (x_i @ w + b)) + ||w||_2^2 / (2 C)`` (matching
    ``sklearn.linear_model.LogisticRegression(C=c, penalty="l2", fit_intercept=True)``).

    Returns ``(w, b)`` in float64. Convergence requires ``||grad||_inf <= tol`` (never merely a
    tiny backtracked step). Raises :class:`ValueError` on invalid inputs or non-convergence.
    """
    if c <= 0.0 or not math.isfinite(c):
        raise ValueError(f"c must be finite and > 0, got {c!r}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter!r}")
    if tol <= 0.0 or not math.isfinite(tol):
        raise ValueError(f"tol must be finite and > 0, got {tol!r}")

    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64).ravel()
    if x64.ndim != 2 or x64.shape[0] != y64.size:
        raise ValueError(f"x shape {x64.shape} does not match y length {y64.size}")
    if x64.shape[0] < 2:
        raise ValueError(f"need at least 2 observations, got {x64.shape[0]}")
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

    inv_n = 1.0 / float(n)
    loss, grad, hess = logistic_objective_and_derivatives(x_aug, y64, beta, reg)
    converged = False
    for _ in range(max_iter):
        grad_inf = float(np.linalg.norm(grad, ord=np.inf))
        if not math.isfinite(grad_inf):
            raise ValueError("damped_logistic encountered non-finite gradient")
        if grad_inf <= tol:
            converged = True
            break

        # Scale Newton system and Armijo test by 1/N so loss ~ O(1) and float64 ULP noise
        # does not stall line search at large N (e.g. N = 40,000 GZ2 samples).
        loss_m = loss * inv_n
        grad_m = grad * inv_n
        hess_m = hess * inv_n
        h_reg = hess_m.copy()
        h_reg[np.diag_indices(d + 1)] += 1e-12
        try:
            step = np.linalg.solve(h_reg, grad_m)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(h_reg, grad_m, rcond=None)[0]

        slope_m = float(np.dot(grad_m, step))
        if not math.isfinite(slope_m) or slope_m <= 0.0:
            step = grad_m
            slope_m = float(np.dot(grad_m, step))

        alpha = 1.0
        accepted = False
        ulp_slack = 1e-12 * max(1.0, abs(loss_m))
        while alpha >= 1e-12:
            cand = beta - alpha * step
            cand_loss_m = _logistic_loss_only(x_aug, y64, cand, reg) * inv_n
            if math.isfinite(cand_loss_m) and (
                cand_loss_m <= loss_m - 1e-4 * alpha * slope_m + ulp_slack
            ):
                beta = cand
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            break
        loss, grad, hess = logistic_objective_and_derivatives(x_aug, y64, beta, reg)

    final_grad_inf = float(np.linalg.norm(grad, ord=np.inf))
    if not converged and final_grad_inf <= tol:
        converged = True

    if not (converged and np.isfinite(beta).all()):
        raise ValueError(
            f"damped_logistic failed to converge after {max_iter} iterations "
            f"(||grad||_inf={final_grad_inf:.3e}, tol={tol:.3e})"
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


def _pearson(
    a: np.ndarray,
    b: np.ndarray,
    *,
    scale_a: float = 1.0,
    scale_b: float = 1.0,
    rel_tol: float = 1e-10,
) -> float:
    ac = a - a.mean()
    bc = b - b.mean()
    na = float(np.linalg.norm(ac))
    nb = float(np.linalg.norm(bc))
    if na <= rel_tol * max(scale_a, 1.0) or nb <= rel_tol * max(scale_b, 1.0):
        return float("nan")
    return float(np.clip(np.dot(ac, bc) / (na * nb), -1.0, 1.0))


def partial_spearman(
    x: np.ndarray,
    y: np.ndarray,
    nuisance: np.ndarray,
    *,
    resid_rel_tol: float = 1e-10,
) -> float:
    """Partial Spearman rank correlation between ``x`` and ``y`` controlling for ``nuisance``.

    Computed as the Pearson correlation of OLS-residualised ranks of ``x`` and ``y`` given
    ``[1, rank(nuisance)]`` over jointly finite rows. Returns ``NaN`` if fewer than 3 finite rows
    remain or either residual is constant within ``resid_rel_tol`` of the rank norm.
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
    x_ok, y_ok, z_ok = xv[ok], yv[ok], zv[ok]
    rx_raw = rankdata(x_ok)
    ry_raw = rankdata(y_ok)
    design = _design_from_nuisance(z_ok)
    cx, *_ = np.linalg.lstsq(design, rx_raw, rcond=None)
    cy, *_ = np.linalg.lstsq(design, ry_raw, rcond=None)
    rx = rx_raw - design @ cx
    ry = ry_raw - design @ cy
    return _pearson(
        rx,
        ry,
        scale_a=float(np.linalg.norm(rx_raw - rx_raw.mean())),
        scale_b=float(np.linalg.norm(ry_raw - ry_raw.mean())),
        rel_tol=resid_rel_tol,
    )


@dataclasses.dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap point estimate, percentile CI, and replicate-validity accounting."""

    estimate: float
    lo: float
    hi: float
    n_valid: int
    n_invalid: int
    n_boot: int
    kind: str


def partial_spearman_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    nuisance: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    chunk_size: int = 256,
    resid_rel_tol: float = 1e-10,
    max_invalid_frac: float = 0.25,
) -> BootstrapResult:
    """Full paired-row bootstrap for :func:`partial_spearman`.

    Resamples rows ``(x_i, y_i, nuisance_i)`` with replacement and **re-computes ranks, nuisance
    design, and OLS residualisation** inside every replicate, capturing full estimation
    uncertainty (unlike :func:`paired_pearson_bootstrap`, which conditions on fixed ranks and
    residuals). Uses chunked index draws of size ``chunk_size`` to bound peak memory and returns
    explicit valid/invalid replicate counts. If the invalid replicate fraction exceeds
    ``max_invalid_frac``, ``lo`` and ``hi`` are returned as ``NaN`` rather than hiding degeneracy.
    """
    xv = np.asarray(x, dtype=np.float64).ravel()
    yv = np.asarray(y, dtype=np.float64).ravel()
    zv = np.asarray(nuisance, dtype=np.float64)
    if zv.ndim == 1:
        zv = zv[:, None]
    if not (xv.size == yv.size == zv.shape[0]):
        raise ValueError("x, y, and nuisance must have identical sample counts")
    ok = np.isfinite(xv) & np.isfinite(yv) & np.isfinite(zv).all(axis=1)
    x_ok, y_ok, z_ok = xv[ok], yv[ok], zv[ok]
    n = x_ok.size
    est = partial_spearman(x_ok, y_ok, z_ok, resid_rel_tol=resid_rel_tol)
    if n < 3 or n_boot <= 0:
        return BootstrapResult(
            est, float("nan"), float("nan"), 0, max(n_boot, 0), max(n_boot, 0), "full_paired_row"
        )

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    done = 0
    step = max(int(chunk_size), 1)
    while done < n_boot:
        m = min(step, n_boot - done)
        idx_chunk = rng.integers(0, n, size=(m, n))
        for j in range(m):
            idx = idx_chunk[j]
            draws[done + j] = partial_spearman(
                x_ok[idx], y_ok[idx], z_ok[idx], resid_rel_tol=resid_rel_tol
            )
        done += m

    finite = draws[np.isfinite(draws)]
    n_valid = int(finite.size)
    n_invalid = n_boot - n_valid
    if n_valid == 0 or (n_invalid / n_boot) > max_invalid_frac:
        return BootstrapResult(
            est, float("nan"), float("nan"), n_valid, n_invalid, n_boot, "full_paired_row"
        )
    lo, hi = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapResult(est, float(lo), float(hi), n_valid, n_invalid, n_boot, "full_paired_row")


def paired_pearson_bootstrap(
    rx: np.ndarray,
    ry: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    chunk_size: int = 512,
) -> tuple[float, float]:
    """Conditional fixed-residual paired percentile bootstrap CI for Pearson correlation.

    Resamples pre-residualised arrays ``(rx, ry)`` in chunks (conditional on fitted ranks and
    nuisance regression). For full unconditional uncertainty over the rank + residualisation
    pipeline, use :func:`partial_spearman_bootstrap`.
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
    corrs = np.empty(n_boot, dtype=np.float64)
    done = 0
    step = max(int(chunk_size), 1)
    while done < n_boot:
        m = min(step, n_boot - done)
        idx = rng.integers(0, n, size=(m, n))
        asamp = a[idx]
        bsamp = b[idx]
        ac = asamp - asamp.mean(axis=1, keepdims=True)
        bc = bsamp - bsamp.mean(axis=1, keepdims=True)
        denom = np.sqrt(np.sum(ac * ac, axis=1) * np.sum(bc * bc, axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            corrs[done : done + m] = np.where(denom > 0.0, np.sum(ac * bc, axis=1) / denom, np.nan)
        done += m

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
