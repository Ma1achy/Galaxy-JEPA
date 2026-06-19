"""Probing-harness configuration — every threshold and method selector in one place.

A single ``ProbingConfig`` fully determines a probing run (the ladder + controls +
uncertainty geometry), mirroring how ``HarnessConfig`` determines a pretraining run. It is a
:class:`~galaxy_jepa.core.config.RunConfig` (pydantic, ``extra='forbid'``), so an unknown key
is a loud load-time error and the whole config is stamped into the artefact provenance.

**The five FLAGGED statistical decisions live here as method-selector fields**, each with a
placeholder default and a ``# FLAGGED`` marker. The *structure* around each is fully built in
the modules that consume them (``nulls`` / ``mlp`` / ``entanglement`` / ``uncertainty``); the
stats-grounding session finalises a choice by editing one field here plus one clearly-marked
function body there — never by rebuilding. See ``docs/probing-harness-design.md`` (the
"FORMULA-FLAGGED" boxes).
"""

from __future__ import annotations

from typing import Literal

from galaxy_jepa.core.config import RunConfig

__all__ = ["ProbingConfig"]


class ProbingConfig(RunConfig):
    """A complete, stamped probing run: splits + probe + the gated ladder + the flagged stats."""

    # --- splits / extremes (reuse data/orchestrate + the firewall thresholds) -----------
    seed: int = 0
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    extreme_low: float = 0.2
    extreme_high: float = 0.8
    device: str | None = None

    # --- the canonical linear probe -----------------------------------------------------
    c: float = 1.0  # L2 inverse-strength for the logistic probe
    n_boot: int = 2000  # bootstrap resamples for the AUC CI

    # --- existence verdict + effect floor (3B) ----------------------------------------
    # The existence verdict is "exceeds the 5-null max at p < alpha"; the effect floor
    # separates clean-vs-marginal among the confirmed-real. The null distribution and the
    # gate wiring are built; only the significance machinery (`nulls.existence_pvalue`) and
    # the floor value are pending.
    alpha: float = 0.05
    effect_floor: float = 0.65  # FLAGGED: pending stats grounding — do not finalise.
    #   ^ a meaningful-AUC floor for "clean direction"; set once, with a defensible rationale.

    # --- multiplicity correction (2B) -------------------------------------------------
    # FLAGGED: pending stats grounding — do not finalise.
    # Bonferroni is the defensible placeholder; Benjamini–Yekutieli (dependence-robust FDR)
    # is the higher-power alternative. `nulls.family_significant` swaps on this.
    multiplicity: Literal["bonferroni", "benjamini_yekutieli"] = "bonferroni"
    n_primary_tests: int | None = None  # the ~12 existence tests; None ⇒ infer from feature count

    # --- selectivity / nuisance gate thresholds (3D / gates.md §2) ----------------------
    selectivity_floor: float = 0.10  # real-label AUC − control-label AUC (Hewitt–Liang)
    # FLAGGED trigger: a nuisance is "competitive" when its AUC is within this margin of (or
    # above) the morphology AUC, firing the targeted matched-evaluation (3D-ii).
    nuisance_competitive_margin: float = 0.0  # FLAGGED: pending stats grounding.

    # --- entanglement / Marchenko–Pastur (2A) -----------------------------------------
    # FLAGGED: pending stats grounding — do not finalise. The spectrum/erank/cosine/
    # eigenvectors are all computed; only the MP significance test is a placeholder.
    mp_method: Literal["upper_edge", "tracy_widom"] = "upper_edge"
    # FLAGGED trigger: which feature-pairs the surgical conditional cross-check fires on —
    # the top fraction by recovered cosine. Pending.
    entangled_pair_quantile: float = 0.90  # FLAGGED: pending stats grounding.

    # --- MLP bounded capacity ladder (2D) ---------------------------------------------
    # The sweep itself (width axis, depth/reg/training fixed) is fully built; only the
    # selectivity-ceiling "exceeds its own null" predicate is flagged (`mlp.selectivity_ceiling`).
    mlp_widths: tuple[int, ...] = (16, 32, 64, 128, 256, 512)
    mlp_depth: int = 1  # hidden layers (fixed; width is the only swept knob)
    mlp_weight_decay: float = 1e-4  # fixed regularisation
    mlp_epochs: int = 200  # fixed training time
    mlp_lr: float = 1e-3
    # FLAGGED: pending stats grounding — the ceiling null mechanics.
    ceiling_method: Literal["null_quantile"] = "null_quantile"
    ceiling_null_quantile: float = 0.95  # FLAGGED: control-AUC "exceeds its own null".

    # --- uncertainty geometry permutation test (4A) -----------------------------------
    # Spearman (primary) / Pearson (secondary) are built; only the permutation-p mechanics
    # are flagged (`uncertainty.permutation_p`).
    # FLAGGED: pending stats grounding — do not finalise.
    permutation_method: Literal["two_sided", "greater"] = "two_sided"
    n_perm: int = 10_000  # FLAGGED: permutation count for the null.
