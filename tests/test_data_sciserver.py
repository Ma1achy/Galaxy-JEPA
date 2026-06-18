"""Tests for the pure SciServer pull helpers (network-free, token-free).

The token-only-in-artifacts rule means the package's SciServer contribution is just two pure
functions: chunking the target list under the job-timeout cap, and merging the per-chunk
corpora back into one. These run offline with no SDSS frames and no secret — ``merge_corpora``
only *copies* the ``.fits`` files (never reads them), so plain placeholder files suffice.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from galaxy_jepa.data.sciserver import chunk_target_ids, merge_corpora

pytestmark = pytest.mark.invariant


def test_chunk_target_ids_partitions_exactly():
    ids = list(range(10))
    chunks = chunk_target_ids(ids, 4)
    assert chunks == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    # exact partition: every id once, order preserved, no chunk over the cap
    assert [i for c in chunks for i in c] == ids
    assert all(len(c) <= 4 for c in chunks)


def test_chunk_target_ids_edges():
    assert chunk_target_ids([], 5) == []
    assert chunk_target_ids([1, 2, 3], 10) == [[1, 2, 3]]  # one chunk when cap exceeds n
    with pytest.raises(ValueError):
        chunk_target_ids([1, 2], 0)


def _write_chunk(root: Path, oids: list[int], *, query: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows = [{"object_id": o, "ra": 150.0 + o, "petroRad_r": 4.0} for o in oids]
    with (root / "metadata.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["object_id", "ra", "petroRad_r"])
        w.writeheader()
        w.writerows(rows)
    for o in oids:
        (root / f"{o}.fits").write_bytes(b"FITS-placeholder")  # merge copies, never reads
    (root / "manifest.json").write_text(json.dumps({"query": query, "n": len(oids)}))
    return root


def test_merge_corpora_unions_rows_and_copies_stamps(tmp_path):
    a = _write_chunk(tmp_path / "c0", [101, 102], query="SELECT ...")
    b = _write_chunk(tmp_path / "c1", [103, 104], query="SELECT ...")
    out = merge_corpora([a, b], tmp_path / "merged")

    with (out / "metadata.csv").open() as fh:
        merged = list(csv.DictReader(fh))
    assert sorted(int(r["object_id"]) for r in merged) == [101, 102, 103, 104]
    assert all((out / f"{o}.fits").exists() for o in (101, 102, 103, 104))

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n"] == 4
    assert manifest["data_snapshot"].startswith("manifest:")
    assert manifest["query"] == "SELECT ..."  # taken from the chunk manifests


def test_merge_corpora_dedups_overlap(tmp_path):
    a = _write_chunk(tmp_path / "c0", [201, 202], query="q")
    b = _write_chunk(tmp_path / "c1", [202, 203], query="q")  # 202 overlaps
    out = merge_corpora([a, b], tmp_path / "merged", query="q")
    with (out / "metadata.csv").open() as fh:
        merged = list(csv.DictReader(fh))
    assert sorted(int(r["object_id"]) for r in merged) == [201, 202, 203]


def test_merge_corpora_missing_stamp_fails_loud(tmp_path):
    chunk = _write_chunk(tmp_path / "c0", [301], query="q")
    (chunk / "301.fits").unlink()  # metadata lists it but the stamp is gone
    with pytest.raises(FileNotFoundError):
        merge_corpora([chunk], tmp_path / "merged", query="q")
