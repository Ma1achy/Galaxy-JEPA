"""Tests for the pure SciServer pull helpers (network-free, token-free).

The token-only-in-artifacts rule means the package's SciServer contribution is a handful of
pure functions: chunking the target list under the job-timeout cap, merging the per-chunk
corpora back into one, and deciding the column order an incremental append must use. These run
offline with no SDSS frames and no secret — ``merge_corpora`` only *copies* the ``.fits`` files
(never reads them), so plain placeholder files suffice.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from galaxy_jepa.data.pull import write_metadata
from galaxy_jepa.data.sciserver import (
    align_append_fieldnames,
    chunk_target_ids,
    merge_corpora,
)

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


# --- incremental append: the guard for the defect that mislabelled 190,358 galaxies -------
#
# `_accumulate` in the artifacts driver appends each fetched chunk to a corpus whose header is
# already fixed. It took the column order from the *chunk*, so once the corpus had been
# alphabetised by `write_metadata` (the D13 axis-ratio top-up did exactly that), every appended
# row was written under the wrong names — undetected, because `object_id` is slot 0 either way.


def _corpus_with_alphabetised_header(tmp_path: Path) -> Path:
    """A corpus written the way ``pull.write_metadata`` writes one: object_id, then sorted."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_metadata(corpus, [{"object_id": 1, "petroRad_r": 4.0, "deVAB_r": 0.7, "ra": 150.0}])
    return corpus


def test_append_uses_the_corpus_header_not_the_chunks_order(tmp_path):
    """A chunk in SQL order must be realigned to the corpus's order, not written blind."""
    corpus = _corpus_with_alphabetised_header(tmp_path)
    header = (corpus / "metadata.csv").read_text().splitlines()[0].split(",")
    assert header == ["object_id", "deVAB_r", "petroRad_r", "ra"]  # alphabetised

    chunk_order = ["object_id", "petroRad_r", "ra"]  # SQL order, as sciserver_cut writes it
    fieldnames, write_header = align_append_fieldnames(corpus / "metadata.csv", chunk_order)

    assert fieldnames == header, "append must follow the corpus header, never the chunk"
    assert write_header is False, "the header is already on disk; writing it again corrupts"


def test_append_round_trips_values_under_their_own_names(tmp_path):
    """The end-to-end property the bug violated: a value keeps its column across an append."""
    corpus = _corpus_with_alphabetised_header(tmp_path)
    meta = corpus / "metadata.csv"
    # Ordered as the chunk emits it. Every column below lands in a *different* slot under the
    # corpus's alphabetised header, so each assertion genuinely discriminates: appending blind
    # writes petroRad_r into deVAB_r, ra into petroRad_r, and leaves ra short.
    row = {"object_id": 2, "petroRad_r": 11.9, "ra": 200.0}  # no deVAB_r: not yet topped up

    fieldnames, write_header = align_append_fieldnames(meta, list(row))
    with meta.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    with meta.open(newline="") as handle:
        appended = list(csv.DictReader(handle))[-1]
    assert appended["petroRad_r"] == "11.9"  # appending blind puts ra's 200.0 here
    assert appended["ra"] == "200.0"  # appending blind leaves this short, i.e. None
    assert appended["deVAB_r"] == ""  # declared but absent: empty, never a shifted neighbour


def test_append_refuses_a_chunk_carrying_unknown_columns(tmp_path):
    """Real schema drift must raise — appending would silently discard the new column."""
    corpus = _corpus_with_alphabetised_header(tmp_path)
    with pytest.raises(ValueError, match="drifted"):
        align_append_fieldnames(corpus / "metadata.csv", ["object_id", "ra", "brand_new_column"])


def test_append_writes_a_header_only_for_a_fresh_corpus(tmp_path):
    fieldnames, write_header = align_append_fieldnames(tmp_path / "absent.csv", ["object_id", "ra"])
    assert (fieldnames, write_header) == (["object_id", "ra"], True)


def test_append_refuses_a_headerless_corpus(tmp_path):
    empty = tmp_path / "metadata.csv"
    empty.write_text("")
    with pytest.raises(ValueError, match="no header"):
        align_append_fieldnames(empty, ["object_id"])
