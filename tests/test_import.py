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


def test_data_imports_without_astropy(monkeypatch):
    """Offline-import guard: ``import galaxy_jepa.data`` must not need the ``data`` extra.

    astropy/astroquery are imported lazily inside FitsFrameSource / load_fits_stamp / the
    pull, so the module loads without them; only constructing/using those needs the extra.
    """
    import importlib
    import sys

    for name in [m for m in sys.modules if m.startswith("galaxy_jepa.data")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    # Poison the astronomy imports so any module-scope use would raise ImportError.
    for blocked in ("astropy", "astropy.io", "astroquery", "astroquery.sdss"):
        monkeypatch.setitem(sys.modules, blocked, None)

    module = importlib.import_module("galaxy_jepa.data")
    assert hasattr(module, "DirectorySource")
    assert hasattr(module, "FitsFrameSource")
