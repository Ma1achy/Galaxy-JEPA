"""SciServer pull — the *pure*, importable helpers (no token, no Jobs API, no network).

The SciServer Compute pull (server-side native cutouts; ``docs/spec/data.md`` §3) is the
scalable corpus path, but its **auth and job orchestration live in ``artifacts/`` only** —
the SciServer token never enters the importable package (memory: token-only-in-artifacts).
This module holds the two pieces of that pull that are pure functions of data, so they can
be unit-tested offline and reused by the ``artifacts/`` driver:

* :func:`chunk_target_ids` — split a target object-ID list into job-sized chunks, so a large
  pull runs as several SciServer jobs each under the ~1 h Small-domain timeout cap;
* :func:`merge_corpora` — stitch the per-chunk ``DirectorySource`` outputs back into one
  corpus directory (one ``metadata.csv`` + the ``<object_id>.fits`` stamps + a combined
  ``manifest.json``), so the rest of the pipeline sees a single corpus;
* :func:`align_append_fieldnames` — decide the column order an *incremental* chunk append
  must write in, so a chunk can never be appended under a header it does not match.

Neither touches the network, the SciServer SDK, or any secret. The live submit/poll/fetch
driver — and all authentication — stays in ``artifacts/sciserver_pull.py``.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Sequence
from pathlib import Path

from galaxy_jepa.data.manifest import manifest_hash

__all__ = ["align_append_fieldnames", "chunk_target_ids", "merge_corpora"]


def chunk_target_ids(ids: Sequence[int], max_per_job: int) -> list[list[int]]:
    """Split ``ids`` into order-preserving chunks of at most ``max_per_job`` each.

    The chunking is deterministic (a contiguous slice of the ordered target list), so a
    chunked pull and a single pull cover exactly the same galaxies in the same order — the
    cut server-side runs ``ORDER BY objID``, and so does this. Used by the ``artifacts/``
    driver to keep each SciServer job under the Small-domain ~1 h timeout cap.
    """
    if max_per_job <= 0:
        raise ValueError(f"max_per_job must be a positive integer, got {max_per_job!r}")
    ids = list(ids)
    return [ids[i : i + max_per_job] for i in range(0, len(ids), max_per_job)]


def align_append_fieldnames(
    corpus_meta: str | Path, chunk_fieldnames: Sequence[str]
) -> tuple[list[str], bool]:
    """Column order for appending a chunk to ``corpus_meta``, plus whether to write a header.

    :func:`merge_corpora` is safe because it re-writes one file from the union of every
    chunk's columns. The *incremental* path is not: it appends to a corpus whose header is
    already fixed, and a chunk's own column order is **not** the corpus's. A chunk arrives in
    SQL order (``object_id`` then passthrough); the corpus is alphabetised by
    ``pull.write_metadata``. Handing ``csv.DictWriter`` the chunk's order writes every value
    under the wrong name, and nothing complains -- ``object_id`` occupies slot 0 in both
    orderings, so even an ID-keyed integrity check passes. That silently mislabelled 190,358
    galaxies of the probe corpus: vote counts landed in ``_fraction`` columns and a specobjid
    in ``deVAB_r``.

    So: the existing header wins. Columns it declares that the chunk lacks are the caller's
    to fill empty (the D13 axis-ratio top-up is exactly this case), matching
    ``pull.merge_columns``' "never silently shrink a corpus". A column the *chunk* carries
    and the header does not is real schema drift -- appending would discard it, so it raises.
    """
    path = Path(corpus_meta)
    if not path.exists():
        return list(chunk_fieldnames), True
    with path.open(newline="") as handle:
        header = next(csv.reader(handle), None)
    if not header:
        raise ValueError(f"{path} exists but has no header row")
    unknown = [c for c in chunk_fieldnames if c not in set(header)]
    if unknown:
        raise ValueError(
            f"chunk columns absent from {path}: {unknown}. Appending would silently drop "
            f"them -- the corpus schema has drifted."
        )
    return header, False


def merge_corpora(
    chunk_dirs: Sequence[str | Path], out_dir: str | Path, *, query: str | None = None
) -> Path:
    """Merge per-chunk ``DirectorySource`` corpora into one corpus directory.

    Each chunk dir is a ``metadata.csv`` + ``<object_id>.fits`` layout (what a single
    SciServer job unpacks to). The merged corpus unions the metadata rows by ``object_id``
    (first occurrence wins — chunks are disjoint by construction, so this only guards an
    accidental overlap), copies every stamp across, and writes a combined ``manifest.json``
    whose ``data_snapshot`` hash covers the full merged object-ID set.

    ``query`` records the SQL that produced the targets; if omitted it is taken from the
    first chunk's ``manifest.json`` (so the merged snapshot stays reproducible).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    rows_by_oid: dict[int, dict[str, str]] = {}
    resolved_query = query
    for chunk in chunk_dirs:
        chunk = Path(chunk)
        meta_path = chunk / "metadata.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"chunk dir {chunk} has no metadata.csv")
        with meta_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for name in reader.fieldnames or []:
                if name not in fieldnames:
                    fieldnames.append(name)
            for row in reader:
                oid = int(row["object_id"])
                if oid in rows_by_oid:
                    continue
                rows_by_oid[oid] = row
                src_fits = chunk / f"{oid}.fits"
                if not src_fits.exists():
                    raise FileNotFoundError(f"chunk {chunk} lists {oid} but {src_fits} is missing")
                shutil.copy2(src_fits, out / f"{oid}.fits")
        if resolved_query is None:
            manifest_path = chunk / "manifest.json"
            if manifest_path.exists():
                resolved_query = json.loads(manifest_path.read_text()).get("query")

    ordered = [rows_by_oid[oid] for oid in sorted(rows_by_oid)]
    with (out / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)

    snapshot = manifest_hash(rows_by_oid.keys(), resolved_query or "")
    (out / "manifest.json").write_text(
        json.dumps(
            {"data_snapshot": snapshot, "n": len(ordered), "query": resolved_query or ""}, indent=2
        )
        + "\n"
    )
    return out
