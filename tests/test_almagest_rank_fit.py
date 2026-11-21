"""almagest_spiral.rank_fit fails loudly on a non-finite value.

Almagest B2 found a NaN SNR silently read as the top rank.
"""

from pathlib import Path

import numpy as np
import pytest

# almagest_spiral.py sets up run directories at import; load only rank_fit's source
# so the test stays pure.
_SRC = Path(__file__).resolve().parents[1] / "artifacts" / "almagest_spiral.py"


def _rank_fit():
    text = _SRC.read_text()
    start = text.index("def rank_fit(")
    end = text.index("\ndef ", start + 1)
    ns: dict = {"np": np}
    exec(compile(text[start:end], str(_SRC), "exec"), ns)  # noqa: S102
    return ns["rank_fit"]


@pytest.mark.invariant
def test_rank_fit_ranks_finite_values() -> None:
    f = _rank_fit()(np.arange(10.0))
    np.testing.assert_allclose(f(np.array([-1.0, 4.0, 99.0])), [0.05, 0.5, 0.95])


@pytest.mark.invariant
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_rank_fit_raises_on_non_finite_apply(bad: float) -> None:
    f = _rank_fit()(np.arange(10.0))
    with pytest.raises(ValueError, match="non-finite"):
        f(np.array([1.0, bad, 3.0]))


@pytest.mark.invariant
def test_rank_fit_raises_on_non_finite_train() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        _rank_fit()(np.array([1.0, np.nan, 3.0]))
