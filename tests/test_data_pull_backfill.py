"""Metadata top-ups on an already-pulled corpus: the derived SNR column and the D13 axis ratios.

Offline by construction — the networked half (``pull_axis_ratios``) is a thin ``run_sql`` call;
what is tested here is the pure half that decides *identity* and *column merge*, because that is
where a silent corruption lives: the probe corpus carries both ``object_id`` (DR8, the key the
FITS filenames use) and ``dr7objid``, and reading the wrong one rewrites the corpus's identity
with no error.
"""

from __future__ import annotations

import csv

import pytest

from galaxy_jepa.data.metadata import AXIS_RATIO_COLS
from galaxy_jepa.data.pull import (
    _object_id,
    backfill_derived,
    merge_columns,
    read_metadata,
    with_derived_columns,
)

pytestmark = pytest.mark.invariant

# The corpus schema that actually bit: object_id is DR8, dr7objid is a *different* number for
# the same galaxy, and both are present.
DR8, DR7 = 1237648702966268030, 587722981741363294


def _corpus(tmp_path, rows):
    out = tmp_path / "corpus"
    out.mkdir()
    with (out / "metadata.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return out


def test_object_id_prefers_written_identity_over_dr7():
    """An on-disk ``object_id`` outranks ``dr7objid`` — the two are different numbers."""
    assert _object_id({"object_id": DR8, "dr7objid": DR7}) == DR8
    assert _object_id({"objID": DR8, "dr7objid": DR7}) == DR8
    assert _object_id({"dr7objid": DR7}) == DR7  # last resort only


def test_object_id_raises_without_any_identity():
    with pytest.raises(KeyError):
        _object_id({"ra": 1.0, "dec": 2.0})


def test_backfill_derives_snr_without_touching_identity(tmp_path):
    corpus = _corpus(
        tmp_path,
        [{"object_id": DR8, "dr7objid": DR7, "modelMagErr_r": "0.10"}],
    )
    assert backfill_derived(corpus) == 1
    (row,) = read_metadata(corpus)
    assert int(row["object_id"]) == DR8  # NOT rewritten to dr7objid
    assert float(row["snr_r"]) == pytest.approx(10.857, rel=1e-3)


def test_backfill_is_idempotent(tmp_path):
    corpus = _corpus(tmp_path, [{"object_id": DR8, "dr7objid": DR7, "modelMagErr_r": "0.10"}])
    backfill_derived(corpus)
    first = read_metadata(corpus)
    backfill_derived(corpus)
    assert read_metadata(corpus) == first


def test_backfill_fails_loudly_without_the_source_column(tmp_path):
    """The unlabelled pretrain pull has no ``modelMagErr_r``; that must raise, not write NaNs."""
    corpus = _corpus(tmp_path, [{"object_id": DR8, "petroRad_r": "3.0"}])
    with pytest.raises(ValueError, match="modelMagErr_r"):
        backfill_derived(corpus)


def test_bad_mag_error_becomes_nan_not_an_exception():
    (row,) = with_derived_columns([{"object_id": DR8, "modelMagErr_r": "0"}])
    assert row["snr_r"] != row["snr_r"]  # NaN — one bad row must not kill a 40k pull


def test_merge_columns_joins_on_objid(tmp_path):
    corpus = _corpus(tmp_path, [{"object_id": DR8, "dr7objid": DR7}])
    matched, missing = merge_columns(
        corpus, [{"objID": str(DR8), "expAB_r": "0.81", "deVAB_r": "0.76"}], AXIS_RATIO_COLS
    )
    assert (matched, missing) == (1, 0)
    (row,) = read_metadata(corpus)
    assert float(row["expAB_r"]) == pytest.approx(0.81)
    assert int(row["object_id"]) == DR8


def test_merge_columns_keeps_unmatched_rows(tmp_path):
    """A partial catalogue top-up must never silently shrink the corpus."""
    corpus = _corpus(tmp_path, [{"object_id": DR8}, {"object_id": DR8 + 1}])
    matched, missing = merge_columns(
        corpus, [{"objID": str(DR8), "expAB_r": "0.81", "deVAB_r": "0.76"}], AXIS_RATIO_COLS
    )
    assert (matched, missing) == (1, 1)
    rows = read_metadata(corpus)
    assert len(rows) == 2
    assert rows[1]["expAB_r"] == ""


def test_axis_ratios_are_not_nuisance_regressors():
    """D13: inclination is a *conditioning* axis, so regressing it out as a nuisance is wrong."""
    from galaxy_jepa.probing.extract import DEFAULT_NUISANCE_COLS

    assert not set(AXIS_RATIO_COLS) & set(DEFAULT_NUISANCE_COLS.values())


def test_snr_nuisance_column_matches_what_the_pull_writes():
    """The nuisance mapping's value must be the column the single derivation site writes."""
    from galaxy_jepa.probing.extract import DEFAULT_NUISANCE_COLS

    (row,) = with_derived_columns([{"object_id": DR8, "modelMagErr_r": "0.10"}])
    assert DEFAULT_NUISANCE_COLS["snr"] in row
