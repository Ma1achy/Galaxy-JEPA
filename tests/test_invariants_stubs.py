"""Registered-but-unimplemented invariant tests.

These two invariants are structural to the methodology, so the tests exist *now* —
skipped with a TODO — to reserve their names and intent before the code that satisfies
them is written. When ``masking/`` and ``data/`` land (next phase), drop the skip and
fill the body.

* ``test_masking_beta_zero_is_ijepa`` — the masking module is a strict generalisation
  of I-JEPA: at β=0 the block-sampling statistics must be identical to standard I-JEPA
  (``docs/masking.md`` §6, ``docs/architecture.md`` hard invariant 4). This is
  deterministic code-correctness, hence a property test rather than a control gate.
* ``test_normalisation_parity`` — format + stretch + normalisation are byte-identical
  across the pretraining corpus, the probing corpus, and every baseline
  (``docs/spec/data.md``; protects D6 and the Rung-4 result).
"""

import pytest


@pytest.mark.invariant
@pytest.mark.skip(reason="TODO(P3): build masking/ then assert β=0 block stats == I-JEPA")
def test_masking_beta_zero_is_ijepa():
    raise NotImplementedError


@pytest.mark.invariant
@pytest.mark.skip(
    reason="TODO(P2): build data/ then assert format+stretch+norm parity across corpora"
)
def test_normalisation_parity():
    raise NotImplementedError
