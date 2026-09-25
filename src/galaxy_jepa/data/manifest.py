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
from pathlib import Path

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


def corpus_identity(corpus_dir: str | Path) -> str:
    """The corpus's own snapshot when its pixels are part of its identity, else ``""``.

    ``manifest_hash`` covers IDs + query, so re-cutting the same IDs with a different cutter
    would reuse the old corpus's identity: different pixels, same ``data_snapshot``. A corpus
    whose manifest query names its cutter (``|cutter=``, written by the aligned re-pull) carries
    that into every run built on it. A corpus without one returns ``""``, so every identity
    recorded before the re-pull (M's included) is unchanged.
    """
    manifest = Path(corpus_dir) / "manifest.json"
    if not manifest.exists():
        return ""
    record = json.loads(manifest.read_text())
    if "|cutter=" not in str(record.get("query", "")):
        return ""
    return str(record["data_snapshot"])


def corpora_query(pretrain_dir: str | Path, probe_dir: str | Path) -> str:
    """The split plan's query: each corpus's pixel identity, where it has one."""
    parts = [
        f"{name}={ident}"
        for name, ident in (
            ("pretrain", corpus_identity(pretrain_dir)),
            ("probe", corpus_identity(probe_dir)),
        )
        if ident
    ]
    return "|".join(parts)
