"""The data stack — pretrain–probe parity.

Implements ``docs/architecture.md`` "The data stack — pretrain-probe parity" and
``docs/spec/data.md``.

This package is built on a short empirical leash (the data layer collides with the real
world). The pieces present here are the **parity-locked front of the pipeline** —
``Transform`` / ``AsinhStretch`` / ``Normalise`` / ``Pipeline`` (`transforms`), the
provenance manifest (`manifest`), and the offline ``DataSource`` surface (`sources`).
The networked SDSS pull and the CasJobs metadata join arrive next, behind the
``DataSource`` boundary, so tests stay network-free.

The single load-bearing rule (``docs/spec/data.md`` §1): **format + stretch +
normalisation are byte-identical across the pretraining corpus, the probing corpus, and
every baseline** — enforced by sharing *one* fitted
:class:`~galaxy_jepa.data.transforms.Pipeline`, never building a second.
"""

from galaxy_jepa.data.manifest import manifest_hash
from galaxy_jepa.data.sources import DataSource, FixtureSource, load_fits_stamp
from galaxy_jepa.data.transforms import AsinhStretch, Normalise, Pipeline, Transform

__all__ = [
    "AsinhStretch",
    "DataSource",
    "FixtureSource",
    "Normalise",
    "Pipeline",
    "Transform",
    "load_fits_stamp",
    "manifest_hash",
]
