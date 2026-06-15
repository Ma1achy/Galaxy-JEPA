"""The data stack — pretrain–probe parity.

Implements ``docs/architecture.md`` "The data stack — pretrain-probe parity" and
``docs/spec/data.md``.

This package is built on a short empirical leash (the data layer collides with the real
world). The parity-locked front of the pipeline — ``Transform`` / ``AsinhStretch`` /
``Normalise`` / ``Pipeline`` (`transforms`) — feeds off a ``DataSource`` (`sources`): the
offline ``DirectorySource`` (fixtures and a written pull share it) and the networked
``FitsFrameSource``. The CasJobs join + derived image-domain SNR live in `metadata`, the
Petrosian sampling box in `bbox`, the Tier-2 stretch check in `sanity`, and the
eyeball-gate render in `contact_sheet`. Provenance: `manifest`.

The single load-bearing rule (``docs/spec/data.md`` §1): **format + stretch +
normalisation are byte-identical across the pretraining corpus, the probing corpus, and
every baseline** — enforced by sharing *one* fitted
:class:`~galaxy_jepa.data.transforms.Pipeline`, never building a second.
"""

from galaxy_jepa.data.bbox import Box, fallback_rate, petrosian_box
from galaxy_jepa.data.manifest import manifest_hash
from galaxy_jepa.data.sanity import StretchSanity, stretch_sanity
from galaxy_jepa.data.sources import (
    DataSource,
    DirectorySource,
    FitsFrameSource,
    FixtureSource,
    load_fits_stamp,
)
from galaxy_jepa.data.transforms import AsinhStretch, Normalise, Pipeline, Transform

__all__ = [
    "AsinhStretch",
    "Box",
    "DataSource",
    "DirectorySource",
    "FitsFrameSource",
    "FixtureSource",
    "Normalise",
    "Pipeline",
    "StretchSanity",
    "Transform",
    "fallback_rate",
    "load_fits_stamp",
    "manifest_hash",
    "petrosian_box",
    "stretch_sanity",
]
