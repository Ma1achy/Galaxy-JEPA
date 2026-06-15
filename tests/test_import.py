"""Trivial import test — confirms the package skeleton is importable."""

import galaxy_jepa


def test_package_imports():
    assert galaxy_jepa.__version__ == "0.0.0"


def test_subpackages_import():
    import galaxy_jepa.callbacks  # noqa: F401
    import galaxy_jepa.core  # noqa: F401
    import galaxy_jepa.data  # noqa: F401
    import galaxy_jepa.eval  # noqa: F401
    import galaxy_jepa.masking  # noqa: F401
    import galaxy_jepa.models  # noqa: F401
    import galaxy_jepa.objectives  # noqa: F401
    import galaxy_jepa.probing  # noqa: F401
