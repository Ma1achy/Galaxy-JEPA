"""Erratum for aligned_comparison.md Criterion 3: 1-df plug-in floor calibration (2026-11-26).

Why this exists:
    ``artifacts/aligned_c3_floor.py`` calibrated the Criterion 3 seed-range multiplier ``k = 1.110``
    under a **known-sigma** null (``z[:, :2]`` vs ``z[:, 2:]`` with fixed ``sigma = 1``). At
    scoring time, however, ``F_q = k * s_M`` plugs in the 1-df sample spread
    ``s_M = |M1_q - M2_q| / sqrt(2)`` estimated from the two M seeds themselves. Because the two M
    seeds sit in both the gap numerator and the plug-in denominator, ``k = 1.110`` actually
    false-calls ~16.9% of exchangeable null draws (not 5%).

    The exact 1-df plug-in 95th percentile of ``max(min(A) - max(M), min(M) - max(A)) / s_M``
    under ``A1, A2, M1, M2 ~ N(0, 1)`` i.i.d. is ``k_plugin ≈ 5.864``.

This script leaves ``artifacts/out/c3_floor.json`` and the hashed pre-registration section of
``artifacts/aligned_comparison.md`` untouched. It reproduces the 1-df plug-in Monte Carlo
calibration on CPU and re-scores the observed Criterion 3 table from ``aligned_comparison.md``
under both ``k = 1.110`` (original) and ``k_plugin`` (calibrated 1-df plug-in).

    .venv/bin/python artifacts/erratum_aligned_c3.py
    .venv/bin/python artifacts/erratum_aligned_c3.py --draws 1000000 --out /tmp/erratum_c3.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# Observed Criterion 3 numbers from artifacts/aligned_comparison.md ("Results — Numbers — 3").
OBSERVED: dict[str, dict[str, tuple[float, float]]] = {
    "N": {"A": (0.261, 0.365), "M": (0.541, 0.480)},
    "PR": {"A": (16.53, 11.53), "M": (9.63, 13.19)},
    "offset_flagged": {"A": (0.0028, 0.0011), "M": (0.2757, 0.5036)},
    "Mo": {"A": (0.0, 0.085), "M": (0.0, 0.0)},
}

LABELS: dict[str, tuple[str, str, str]] = {
    "N": ("DIRTIER", "SAME", "CLEANER"),
    "PR": ("HIGHER", "SAME", "LOWER"),
    "offset_flagged": ("HIGHER", "SAME", "LOWER"),
    "Mo": ("GAINED", "SAME", "LOST"),
}

K_ORIGINAL = 1.1096


def seed_range(
    a: tuple[float, float],
    m: tuple[float, float],
    hi: str,
    same: str,
    lo: str,
    floor: float,
) -> str:
    if min(a) - max(m) > floor:
        return hi
    if min(m) - max(a) > floor:
        return lo
    return same


def calibrate_plugin(draws: int = 1_000_000, seed: int = 20261126, alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((draws, 4))
    a, m = z[:, :2], z[:, 2:]
    up = a.min(axis=1) - m.max(axis=1)
    down = m.min(axis=1) - a.max(axis=1)
    gap = np.maximum(up, down)
    s_m = np.abs(m[:, 0] - m[:, 1]) / math.sqrt(2.0)

    k_known = float(np.quantile(gap, 1.0 - alpha))
    ratio = gap / np.maximum(s_m, 1e-30)
    k_plugin = float(np.quantile(ratio, 1.0 - alpha))

    rate_orig_known = float((gap > K_ORIGINAL).mean())
    rate_orig_plugin = float((gap > K_ORIGINAL * s_m).mean())
    rate_cal_plugin = float((gap > k_plugin * s_m).mean())
    rate_unfloored = float((gap > 0.0).mean())

    scored: dict[str, dict[str, object]] = {}
    for q, pair in OBSERVED.items():
        a_pair, m_pair = pair["A"], pair["M"]
        hi, same, lo = LABELS[q]
        s_obs = abs(m_pair[0] - m_pair[1]) / math.sqrt(2.0)
        f_orig = K_ORIGINAL * s_obs
        f_plug = k_plugin * s_obs
        up_gap = min(a_pair) - max(m_pair)
        down_gap = min(m_pair) - max(a_pair)
        scored[q] = {
            "A": list(a_pair),
            "M": list(m_pair),
            "s_M": round(s_obs, 6),
            "gap_up": round(up_gap, 6),
            "gap_down": round(down_gap, 6),
            "F_original": round(f_orig, 6),
            "state_original": seed_range(a_pair, m_pair, hi, same, lo, f_orig),
            "F_plugin": round(f_plug, 6),
            "state_plugin": seed_range(a_pair, m_pair, hi, same, lo, f_plug),
        }

    return {
        "draws": draws,
        "seed": seed,
        "alpha": alpha,
        "k_original": K_ORIGINAL,
        "k_known_sigma_repro": round(k_known, 4),
        "k_plugin_95": round(k_plugin, 4),
        "false_call_rate_unfloored": round(rate_unfloored, 5),
        "false_call_rate_original_under_known_sigma": round(rate_orig_known, 5),
        "false_call_rate_original_under_1df_plugin": round(rate_orig_plugin, 5),
        "false_call_rate_calibrated_under_1df_plugin": round(rate_cal_plugin, 5),
        "observed_rescoring": scored,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="1-df plug-in erratum check for aligned C3.")
    ap.add_argument("--draws", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=20261126)
    ap.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    args = ap.parse_args()

    result = calibrate_plugin(draws=args.draws, seed=args.seed)
    text = json.dumps(result, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
