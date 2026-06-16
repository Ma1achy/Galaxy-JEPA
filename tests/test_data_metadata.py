"""Unit tests for the metadata helpers (data/metadata.py) — all offline.

Pins: the image-domain photometric SNR derivation; the deterministic ``ORDER BY`` in the
SQL templates (manifest stability); and the GZ2↔PhotoObjAll join guard — agreement passes,
a silent key mismatch raises loudly.
"""

import pytest

from galaxy_jepa.data.metadata import (
    assert_radec_agree,
    photometric_snr,
    pretrain_sql,
    probe_sql,
)


def test_photometric_snr_from_mag_error():
    # SNR ≈ 1.0857 / modelMagErr_r
    assert photometric_snr(0.10) == pytest.approx(10.857, rel=1e-3)
    with pytest.raises(ValueError):
        photometric_snr(0.0)
    with pytest.raises(ValueError):
        photometric_snr(-0.1)


def test_sql_templates_are_deterministic():
    for sql in (pretrain_sql(2000), probe_sql(2000)):
        assert "ORDER BY" in sql  # without it TOP n is non-deterministic in T-SQL
        assert "TOP 2000" in sql
    # photometric SNR is derived in Python, not the spectrum: snMedian must not appear.
    assert "snMedian" not in probe_sql(10)
    assert "modelMagErr_r" in probe_sql(10)


def test_join_guard_passes_on_agreement():
    rows = [
        {"dr7objid": 1, "gz_ra": 150.0, "gz_dec": 2.0, "phot_ra": 150.00001, "phot_dec": 2.00001}
    ]
    assert_radec_agree(rows)  # sub-arcsec agreement -> no raise


def test_join_guard_raises_on_mismatch():
    rows = [{"dr7objid": 2, "gz_ra": 150.0, "gz_dec": 2.0, "phot_ra": 151.0, "phot_dec": 2.0}]
    with pytest.raises(ValueError, match="join"):
        assert_radec_agree(rows)
