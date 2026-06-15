"""CasJobs / SkyServer metadata — the queries and the derived nuisance columns.

Implements ``docs/spec/data.md`` §3. Two distinct pulls (``DECISIONS.md`` D6): the
large unlabelled **pretraining** corpus (the distinct ``petroRad`` pull) and the
GZ2-labelled **probing** corpus (the nuisance battery). The SQL is parameterised on the
row limit and **ordered by object ID** so a ``TOP n`` slice is deterministic — without
``ORDER BY`` it is non-deterministic in T-SQL and would break the manifest-hash
reproducibility (``docs/spec/config.md``).

Two corrections baked in from the CasJobs gate:

* **SNR is photometric, image-domain** — derived here as ``1.0857 / modelMagErr_r``, the
  r-band image SNR the encoder actually sees. ``SpecObj.snMedian`` measures the *spectrum*
  (fibre/exposure), the wrong domain for an image-quality nuisance probe, so it is not
  joined.
* **The GZ2↔PhotoObjAll join is verified before it is trusted** (:func:`assert_radec_agree`
  / :func:`join_check_sql`): a silent key mismatch returns wrong-galaxy metadata with no
  error, so a 10-row ra/dec agreement check runs first.

The networked call (:func:`run_sql`) imports ``astroquery`` lazily; the pure helpers are
unit-tested offline.
"""

from __future__ import annotations

from typing import Any

# --- SQL templates (deterministic via ORDER BY) ------------------------------------

# Pretraining is unlabelled: it needs petroRad (+ pixel scale) for the masking box and
# the frame coords to cut stamps (D6) — not the nuisance battery (that is on the probing
# corpus). psfWidth_r is not on PhotoPrimary anyway (it lives in Field); omitted here.
PRETRAIN_SQL = """\
SELECT TOP {limit}
    p.objID, p.ra, p.dec,
    p.petroRad_r, p.petroRadErr_r,
    p.modelMag_r,
    p.run, p.camcol, p.field, p.rerun
FROM PhotoPrimary AS p
WHERE p.type = 3 AND p.clean = 1
  AND p.modelMag_r BETWEEN {mag_min} AND {mag_max}
ORDER BY p.objID"""

# Join keys confirmed against the live DR17 schema:
#   * PhotoObjAll on dr8objid (dr7objid does not match DR17 objIDs → zero rows);
#   * psfWidth_r lives in the Field table, not PhotoObjAll;
#   * redshift is SpecObj.z joined on specObjID = zoo2MainSpecz.specobjid
#     (zoo2MainSpecz itself has no redshift column — only GZ2 vote fractions).
PROBE_SQL = """\
SELECT TOP {limit}
    g.dr8objid AS objID, g.ra, g.dec,
    s.z AS specz,
    p.petroRad_r, p.petroRadErr_r,
    p.modelMag_r, p.modelMagErr_r,
    f.psfWidth_r,
    p.run, p.camcol, p.field, p.rerun
FROM zoo2MainSpecz AS g
JOIN PhotoObjAll AS p ON p.objID = g.dr8objid
JOIN Field AS f ON f.fieldID = p.fieldID
JOIN SpecObj AS s ON s.specObjID = g.specobjid
ORDER BY g.dr8objid"""

# Selects BOTH sides' ra/dec so the join key can be validated before it is trusted.
JOIN_CHECK_SQL = """\
SELECT TOP {limit}
    g.dr8objid AS objID, g.ra AS gz_ra, g.dec AS gz_dec,
    p.ra AS phot_ra, p.dec AS phot_dec
FROM zoo2MainSpecz AS g
JOIN PhotoObjAll AS p ON p.objID = g.dr8objid
ORDER BY g.dr8objid"""


def pretrain_sql(limit: int, *, mag_min: float = 14.0, mag_max: float = 19.0) -> str:
    return PRETRAIN_SQL.format(limit=int(limit), mag_min=mag_min, mag_max=mag_max)


def probe_sql(limit: int) -> str:
    return PROBE_SQL.format(limit=int(limit))


def join_check_sql(limit: int = 10) -> str:
    return JOIN_CHECK_SQL.format(limit=int(limit))


# --- derived columns ----------------------------------------------------------------

# d(mag) = -2.5/ln(10) * d(flux)/flux  ⇒  SNR = flux/d(flux) ≈ 1.0857 / modelMagErr.
_MAG_SNR_CONST = 1.0857362


def photometric_snr(model_mag_err_r: float) -> float:
    """r-band image-domain SNR from the photometric magnitude error.

    This is the ``"SNR"`` nuisance column (``docs/spec/data.md`` §3) — an image-domain
    quantity, so the probe genuinely asks "does the concept axis read off image depth".
    """
    if not (model_mag_err_r > 0):
        raise ValueError(f"modelMagErr_r must be > 0 to derive SNR, got {model_mag_err_r!r}")
    return _MAG_SNR_CONST / model_mag_err_r


# --- join verification --------------------------------------------------------------


def _angular_sep_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Small-angle great-circle separation in arcsec (cheap, no astropy dependency)."""
    import math

    dec_mean = math.radians((dec1 + dec2) / 2.0)
    d_ra = (ra1 - ra2) * math.cos(dec_mean)
    d_dec = dec1 - dec2
    return math.hypot(d_ra, d_dec) * 3600.0


def assert_radec_agree(rows: list[dict[str, Any]], *, tol_arcsec: float = 1.0) -> None:
    """Raise loudly if any GZ2↔PhotoObjAll matched row disagrees on sky position.

    Guards the silent-mismatch failure mode: a wrong join key returns a real-but-wrong
    galaxy's metadata with no error.
    """
    for row in rows:
        sep = _angular_sep_arcsec(
            float(row["gz_ra"]), float(row["gz_dec"]),
            float(row["phot_ra"]), float(row["phot_dec"]),
        )
        if sep > tol_arcsec:
            raise ValueError(
                f"GZ2↔PhotoObjAll join mismatch for objID={row.get('objID', row.get('dr7objid'))}: "
                f"sky positions differ by {sep:.2f}″ (> {tol_arcsec}″). The join key is "
                "wrong — metadata would belong to a different galaxy."
            )


# --- networked execution (devcontainer) ---------------------------------------------


def run_sql(sql: str, *, data_release: int = 17, timeout: int = 300) -> list[dict[str, Any]]:
    """Execute a SkyServer SQL query via the SqlSearch REST endpoint, returning row dicts.

    Uses the ``SkyServerWS/SearchTools/SqlSearch`` endpoint rather than
    ``astroquery.sdss.query_sql`` — the latter returns an HTML error page on the heavier
    multi-table joins (zoo2 ⋈ PhotoObjAll ⋈ Field ⋈ SpecObj), whereas this endpoint
    returns clean CSV. Fails loudly on a SkyServer error body (rather than letting a CSV
    parser choke on it).
    """
    import csv
    import io

    import requests

    url = f"https://skyserver.sdss.org/dr{data_release}/SkyServerWS/SearchTools/SqlSearch"
    resp = requests.get(url, params={"cmd": sql, "format": "csv"}, timeout=timeout)
    resp.raise_for_status()
    text = resp.text.lstrip()
    if text.startswith("{") and "ErrorMessage" in text:
        raise RuntimeError(f"SkyServer SQL error: {text[:300]}")
    # The CSV body opens with a '#Table1' comment line, then the header, then rows.
    body = "\n".join(ln for ln in resp.text.splitlines() if not ln.startswith("#"))
    return [dict(row) for row in csv.DictReader(io.StringIO(body))]
