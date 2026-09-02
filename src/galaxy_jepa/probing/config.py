"""Probing-harness configuration — every threshold and method selector in one place.

A single ``ProbingConfig`` fully determines a probing run (the ladder + controls +
uncertainty geometry), mirroring how ``HarnessConfig`` determines a pretraining run. It is a
:class:`~galaxy_jepa.core.config.RunConfig` (pydantic, ``extra='forbid'``), so an unknown key
is a loud load-time error and the whole config is stamped into the artefact provenance.

**The five statistical decisions are GROUNDED** (``docs/galaxy-jepa-spec.pdf`` §Statistics):
existence = real value vs a chance null at the effect floor; a separate effect-size floor on top
of significance (two gates, both must pass); multiplicity = **Benjamini–Yekutieli**; the
permutation test = ≥10,000 shuffles, two-tailed, shuffling the vote fractions; the
Marchenko–Pastur edge computed for the **actual** matrix shape. The fields below carry them.

Three things are still open and stay marked, because the spec's own register still lists them:

* the **effect-floor value** — the mechanism is settled, the number is a scientific call
  (spec open register, item 4);
* **tie-handling** in the existence p (AUC ≈ 1) and the permutation test (register item 9);
* the **selectivity-ceiling** predicate in ``mlp`` — not one of the five, and unmapped.

Everything else here is a trigger threshold, not one of the five, and is marked as such.
"""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from galaxy_jepa.core.config import RunConfig
from galaxy_jepa.probing.schemes import DEFAULT_CONSENSUS_GATE, DEFAULT_VOTE_COUNT_MIN

__all__ = ["EffectFloorFreeze", "ProbingConfig"]


class EffectFloorFreeze(RunConfig):
    """The record of the effect floor being pinned: what set it, when, and on whose call.

    The floor is the one number in the battery that is a *scientific judgement* rather than a
    derivation, and it has exactly one honest window. Choosing it before there is an AUC
    distribution to look at is choosing blind; choosing it after the headline run is p-hacking.
    The medium local run is that window — a real spread of per-feature AUCs, on an encoder whose
    verdicts nobody is going to publish.

    Making the freeze an object rather than a note means the provenance rides in the config,
    so it is hashed into ``config_hash`` and written into every artefact's ``config.json``: a
    result carries the story of where its floor came from, or it is not a headline run.
    """

    value: float
    derived_from: str  # the run the AUC distribution came from (out_dir, or its stamp hash)
    frozen_at: str  # ISO-8601 date the call was made
    frozen_by: str  # who made it — this is a judgement, so it has an author
    rationale: str


class ProbingConfig(RunConfig):
    """A complete, stamped probing run: splits + probe + the gated ladder + the flagged stats."""

    # --- run marking -------------------------------------------------------------------
    # A smoke exercises the load path; it is NOT a result. Setting this stamps ``smoke`` into
    # ``escape_hatches_used`` (the power-path ledger, docs/spec/escape-hatches.md) and changes
    # the config hash, so a smoke artefact can never be mistaken for — or collide with — a real
    # run's. The forfeited guarantee is the whole scientific claim, which is exactly what the
    # ledger exists to record.
    smoke: bool = False

    #: A headline run is one whose verdicts are meant to be read as the result. Marking it so
    #: turns on the pre-registration gates (currently: the effect floor must be frozen).
    headline: bool = False

    #: Declared deviations from a grounded default (``docs/spec/escape-hatches.md``). A run that
    #: wants to weaken a grounded floor must *name* what it forfeits here, and the name is stamped
    #: onto every artefact — so "we ran 200 shuffles" can never be read off a result as if it were
    #: the pre-registered ≥10,000. Known: ``reduced_permutations``.
    escape_hatches: tuple[str, ...] = ()

    # --- splits / extremes (reuse data/orchestrate + the firewall thresholds) -----------
    seed: int = 0
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    extreme_low: float = 0.2
    extreme_high: float = 0.8
    device: str | None = None

    # --- feature scheme + conditional population (D14) ----------------------------------
    # The GZ2 tree is conditional, but conditioning is run as a **comparison**, never as a mask:
    # the ladder runs once on the full population and once on the consensus-conditional one, and
    # the difference between them is the result. Hard-masking the off-population galaxies would
    # pre-impose the tree's logic before testing whether it holds.
    #: Name of the active feature scheme, recorded so the stamped config says which
    #: experiment arm produced the artefact (the scheme sets the BY family size).
    scheme_name: str | None = None
    compare_populations: bool = True
    # OPEN (spec register item 2): which upstream vote, which threshold, consensus vs weighted.
    consensus_gate: float = DEFAULT_CONSENSUS_GATE
    # OPEN (spec register item 3): v1's mean+2σ method lands near 21; the usable deep-feature N
    # at that threshold has not been re-counted, so this is a knob, not a settled number.
    vote_count_min: float = DEFAULT_VOTE_COUNT_MIN

    # --- the canonical linear probe -----------------------------------------------------
    c: float = 1.0  # L2 inverse-strength for the logistic probe
    n_boot: int = 2000  # bootstrap resamples for the AUC CI

    # --- existence verdict + effect floor (3B) — GROUNDED, decisions (1) and (2) --------
    # TWO gates, both must pass, and they do different jobs: significance-against-the-null
    # decides real/not-real (adapting to sample size and probe capacity automatically), and the
    # effect floor then separates clean from marginal among the real. The floor is acceptable as
    # a pre-registered constant precisely *because* it no longer does the existence work — it
    # cannot rescue a null result, only downgrade a significant-but-trivial one (AUC 0.54 at
    # large N).
    alpha: float = 0.05
    # Resamplable null draws per control. This sets the *resolution* of the existence test: the
    # smallest attainable p is 1/(n+1), so the family-corrected bar has to be reachable within it
    # (``nulls.assert_null_resolution`` enforces that rather than letting the catalogue come back
    # silently empty). Raising it costs one logistic fit per draw per feature.
    n_null_draws: int = 50
    # OPEN (spec register item 4): the mechanism is grounded, the *number* is a scientific call
    # still to be made. Until `effect_floor_freeze` is filled in this value is a placeholder,
    # and `headline=True` is refused — see `assert_effect_floor_frozen`.
    effect_floor: float = 0.65
    effect_floor_freeze: EffectFloorFreeze | None = None

    # --- multiplicity correction (2B) — GROUNDED, decision (3) -------------------------
    # Benjamini–Yekutieli, chosen knowingly over BH: the existence tests are correlated *by
    # construction* (bulge levels partition one variable; features co-occur), so an
    # independence-assuming correction is simply wrong here, and FDR — not FWER — is the right
    # target for a discovery catalogue. Bonferroni is retained only as a sensitivity check.
    multiplicity: Literal["bonferroni", "benjamini_yekutieli"] = "benjamini_yekutieli"
    # The BY family count is **per-scheme**, never a global constant (≈37 full tree, ≈10–13
    # reduced). ``None`` ⇒ take it from the active scheme, else the feature count.
    n_primary_tests: int | None = None

    # --- selectivity / nuisance gate thresholds (3D / gates.md §2) ----------------------
    selectivity_floor: float = 0.10  # real-label AUC − control-label AUC (Hewitt–Liang)
    # FLAGGED trigger: a nuisance is "competitive" when its AUC is within this margin of (or
    # above) the morphology AUC, firing the targeted matched-evaluation (3D-ii).
    nuisance_competitive_margin: float = 0.0  # FLAGGED: pending stats grounding.

    # --- entanglement / Marchenko–Pastur (2A) — GROUNDED, decision (5) ------------------
    # An eigenvalue counts as signal if it sits past the MP edge computed for the **actual**
    # matrix shape (k directions × D embedding dims), not a nominal one. Tracy–Widom stays
    # unimplemented: it is the "if a reviewer wants hard significance" upgrade, not the decision.
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

    # --- uncertainty geometry permutation test (4A) — GROUNDED, decision (4) -----------
    # Shuffle the **vote fractions**, recompute Spearman, locate the real value in that null.
    # ≥10,000 shuffles, two-tailed. The floor is enforced below rather than merely documented,
    # so a cheap run cannot quietly weaken the headline test.
    # OPEN (spec register item 9): tie-handling.
    permutation_method: Literal["two_sided", "greater"] = "two_sided"
    n_perm: int = 10_000

    @model_validator(mode="after")
    def _effect_floor_freeze_agrees(self) -> ProbingConfig:
        """A freeze record that disagrees with the live floor would make the provenance a lie."""
        freeze = self.effect_floor_freeze
        if freeze is not None and freeze.value != self.effect_floor:
            raise ValueError(
                f"effect_floor={self.effect_floor} contradicts its freeze record "
                f"({freeze.value}, frozen {freeze.frozen_at} by {freeze.frozen_by}). Change one "
                "or the other deliberately; a stamped floor must be the floor that ran."
            )
        return self

    @model_validator(mode="after")
    def _headline_requires_a_frozen_floor(self) -> ProbingConfig:
        if self.headline and self.effect_floor_freeze is None:
            raise ValueError(
                "headline=True but the effect floor is still OPEN. The floor separates clean "
                "from marginal among the real, so setting it after seeing headline verdicts is "
                "p-hacking; set it from the medium local run's AUC distribution and record the "
                "freeze in effect_floor_freeze (value / derived_from / frozen_at / frozen_by / "
                "rationale) before the headline run."
            )
        if self.headline and self.smoke:
            raise ValueError("a run cannot be both a smoke and the headline")
        return self

    @model_validator(mode="after")
    def _permutation_floor(self) -> ProbingConfig:
        if self.n_perm < 10_000 and "reduced_permutations" not in self.escape_hatches:
            raise ValueError(
                f"n_perm={self.n_perm} is below the grounded floor of 10,000 shuffles "
                "(spec §Statistics (4)). A run that genuinely needs fewer must declare "
                "escape_hatches=('reduced_permutations',), which stamps the forfeit onto the "
                "artefact rather than hiding it."
            )
        return self
