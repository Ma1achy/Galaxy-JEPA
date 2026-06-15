"""Data-snapshot manifest — reproducibility without a hand-bumped version.

Implements ``docs/spec/config.md`` (the ``data_snapshot`` field of
:class:`~galaxy_jepa.core.config.RunStamp`) and ``docs/spec/data.md`` §3.

The snapshot identifier is a **manifest hash** over the exact object IDs pulled plus the
query that pulled them, so the data a run saw is structural — change the sample or the
query and the hash changes, with nothing to remember to bump.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

_PREFIX = "manifest:"


def manifest_hash(object_ids: Iterable[int], query: str) -> str:
    """Return ``"manifest:<sha256>"`` over the sorted object IDs and the pull query.

    The IDs are sorted so the hash is order-independent (the *set* of galaxies a run
    saw, not the order they arrived in). Feeds ``RunStamp.data_snapshot``.
    """
    payload = {
        "object_ids": sorted(int(o) for o in object_ids),
        "query": str(query),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return _PREFIX + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
