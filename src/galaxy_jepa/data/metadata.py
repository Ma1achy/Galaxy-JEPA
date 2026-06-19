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

# The full GZ2 decision tree (Willett+2013), answer stems grouped by question — verbatim
# from the zoo2MainSpecz schema. Single source of truth for the vote-fraction pull: the
# SELECT block is generated from it (:func:`gz2_vote_columns`), so an answer stem cannot
# silently drift from the catalogue and the raw-only guarantee is enforced in one place.
GZ2_TREE: dict[str, tuple[str, ...]] = {
    "t01_smooth_or_features": ("a01_smooth", "a02_features_or_disk", "a03_star_or_artifact"),
    "t02_edgeon": ("a04_yes", "a05_no"),
    "t03_bar": ("a06_bar", "a07_no_bar"),
    "t04_spiral": ("a08_spiral", "a09_no_spiral"),
    "t05_bulge_prominence": (
        "a10_no_bulge",
        "a11_just_noticeable",
        "a12_obvious",
        "a13_dominant",
    ),
    "t06_odd": ("a14_yes", "a15_no"),
    "t07_rounded": ("a16_completely_round", "a17_in_between", "a18_cigar_shaped"),
    "t08_odd_feature": (
        "a19_ring",
        "a20_lens_or_arc",
        "a21_disturbed",
        "a22_irregular",
        "a23_other",
        "a24_merger",
        "a38_dust_lane",
    ),
    "t09_bulge_shape": ("a25_rounded", "a26_boxy", "a27_no_bulge"),
    "t10_arms_winding": ("a28_tight", "a29_medium", "a30_loose"),
    "t11_arms_number": (
        "a31_1",
        "a32_2",
        "a33_3",
        "a34_4",
        "a36_more_than_4",
        "a37_cant_tell",
    ),
}

# RAW continuous vote variants only. `_debiased` and `_flag` are DELIBERATELY EXCLUDED and
# the exclusion is enforced below: `_debiased` applies the Willett+2013 redshift correction,
# which injects z into the target and would contaminate the uncertainty geometry (and fight
# the z-nuisance control). v2 reads the raw human votes — never debiased.
GZ2_VOTE_VARIANTS: tuple[str, ...] = ("fraction", "weighted_fraction", "count")

#: Variant suffixes that must NEVER be pulled or stored (redshift-debiased fraction; the
#: clean-sample flag). The guard in :func:`gz2_vote_columns` is the structural backstop.
_DISQUALIFIED_VOTE_VARIANTS: tuple[str, ...] = ("debiased", "flag")


def gz2_vote_columns() -> list[str]:
    """Raw GZ2 vote columns: every answer × {``_fraction``, ``_weighted_fraction``, ``_count``}.

    Generated from :data:`GZ2_TREE` so the list cannot drift from the schema. Raises if any
    generated name carries a disqualified suffix (``_debiased`` / ``_flag``) — the no-debiasing
    guarantee is structural here, not a convention, because a debiased target would silently
    corrupt the uncertainty geometry (``docs/spec/data.md``).
    """
    cols = [
        f"{question}_{answer}_{variant}"
        for question, answers in GZ2_TREE.items()
        for answer in answers
        for variant in GZ2_VOTE_VARIANTS
    ]
    bad = [c for c in cols if any(v in c for v in _DISQUALIFIED_VOTE_VARIANTS)]
    if bad:
        raise ValueError(
            f"disqualified GZ2 vote column(s) generated: {bad!r} — debiased/flag must never "
            "be pulled or stored (redshift contamination of the vote target)"
        )
    return cols


# The probe corpus query. The vote block is generated (raw fractions + weighted fractions +
# per-answer counts for the full t01–t11 tree); table-level dr7objid/specobjid and the
# total_classifications/total_votes denominators ride along for join provenance and
# per-question reach. The binary smooth-vs-featured label and the confident extremes
# (uncertainty firewall, data/splits.py) are derived downstream from the RAW a02 fraction —
# no debiasing anywhere in what is pulled or stored. a03 (star/artifact) lets a caller drop
# non-galaxies.
PROBE_SQL_TEMPLATE = """\
SELECT TOP {limit}
    g.dr8objid AS objID, g.dr7objid, g.specobjid, g.ra, g.dec,
    s.z AS specz,
    p.petroRad_r, p.petroRadErr_r,
    p.modelMag_r, p.modelMagErr_r,
    f.psfWidth_r,
{votes},
    g.total_classifications, g.total_votes,
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
    votes = ",\n".join(f"    g.{col}" for col in gz2_vote_columns())
    return PROBE_SQL_TEMPLATE.format(limit=int(limit), votes=votes)


def join_check_sql(limit: int = 10) -> str:
    return JOIN_CHECK_SQL.format(limit=int(limit))


# --- GZ2 t01 label derivation -------------------------------------------------------

#: The featured/disk RAW vote fraction — the probe's label source. v -> 1 means a confident
#: "featured/disk", v -> 0 a confident "smooth". The smooth fraction (a01_fraction) is very
#: nearly 1 - this for two-way responses, so a02 alone defines the binary axis without
#: double-counting.
#: PROVENANCE: the original slice/pilot AUC (~0.905) was measured on the *debiased* a02
#: fraction; the switch to the RAW fraction here is deliberate — debiasing applies the
#: Willett+2013 redshift correction, which injects z into the target (disqualified for v2's
#: uncertainty geometry and the z-nuisance control). See gz2_vote_columns / GZ2_VOTE_VARIANTS.
FEATURED_FRACTION_COL = "t01_smooth_or_features_a02_features_or_disk_fraction"


def featured_label(featured_fraction: float) -> int:
    """Binary smooth-vs-featured label: ``1`` = featured/disk, ``0`` = smooth.

    The headline split at 0.5 (``docs/spec/data.md``). Independent of the firewall: this
    is the *target*, :func:`is_confident_extreme` decides which galaxies enter the fit.
    """
    return int(float(featured_fraction) >= 0.5)


def is_confident_extreme(featured_fraction: float, *, low: float = 0.2, high: float = 0.8) -> bool:
    """True for a high-consensus extreme (``v >= high`` or ``v <= low``).

    These are the galaxies the uncertainty firewall (``data/splits.py``) keeps in the
    axis-fit set, and — for this slice — the ones the headline AUC is reported on, so the
    number is not drowned by the genuinely ambiguous middle.
    """
    v = float(featured_fraction)
    return v >= high or v <= low


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
            float(row["gz_ra"]),
            float(row["gz_dec"]),
            float(row["phot_ra"]),
            float(row["phot_dec"]),
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
