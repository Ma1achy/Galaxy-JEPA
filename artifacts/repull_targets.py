"""Target lists for the aligned re-pull (v2): the old corpora's written IDs, full-precision coords.

*   **probe_v2** — the 230,358 probe IDs with PhotoObjAll ra/dec (the v1 probe was centred on
    GZ2's 4-decimal ra/dec, up to ~0.45 px off). Paged public SkyServer SQL (no token), streamed a
    page at a time; every row's separation from GZ2's position recorded. Large separations (GZ2's dr8objid
    naming a different PhotoObj source than the galaxy at GZ2's position) are counted and
    resolved as a policy step, not here.
*   **pretrain_v2** — `pretrain_all_targets.csv` restricted to the 826,968 IDs that landed. Its
    ra/dec are already PhotoPrimary at full precision. The 16 targets that never landed are
    written separately (they were per-object `cut_one` failures, not chunk deaths).

IDs and coordinates stay strings end to end — csv module only, no float round trip.
"""

import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sciserver_pull import _net_retry  # noqa: E402

from galaxy_jepa.data.metadata import run_sql

WORK = Path(".sciserver_work")
PAGE = 20_000
PROBE_N = 230_358
PRETRAIN_N = 826_968
COLS = ("objID", "ra", "dec", "run", "camcol", "field", "rerun",
        "rowc_g", "colc_g", "rowc_r", "colc_r", "rowc_i", "colc_i", "gz2_ra", "gz2_dec", "sep_arcsec")
PARTIAL = WORK / "probe_v2_targets.partial.csv"
PROBE_OUT = WORK / "probe_v2_targets.csv"
PRETRAIN_OUT = WORK / "pretrain_v2_targets.csv"
FAILED16 = WORK / "pretrain_v1_failed.csv"


def _sql(after: str | None) -> str:
    where = f"WHERE g.dr8objid > {after}\n" if after else ""
    return (f"SELECT TOP {PAGE} CAST(g.dr8objid AS varchar(20)) AS objID,\n"
            "  LTRIM(STR(p.ra,25,15)) AS ra, LTRIM(STR(p.dec,25,15)) AS dec,\n"
            "  p.run, p.camcol, p.field, p.rerun,\n"
            "  p.rowc_g, p.colc_g, p.rowc_r, p.colc_r, p.rowc_i, p.colc_i,\n"
            "  g.ra AS gz2_ra, g.dec AS gz2_dec\n"
            "FROM zoo2MainSpecz g JOIN PhotoObjAll p ON p.objID = g.dr8objid\n"
            f"{where}ORDER BY g.dr8objid")


def _sep(r: dict) -> float:
    ra, dec, gra, gdec = (float(r[k]) for k in ("ra", "dec", "gz2_ra", "gz2_dec"))
    dra = (ra - gra) * math.cos(math.radians(dec))
    return 3600.0 * math.hypot(dra, dec - gdec)


def probe() -> None:
    n, after = 0, None
    if PARTIAL.exists():
        for r in csv.DictReader(PARTIAL.open(newline="")):
            n, after = n + 1, r["objID"]
        print(f"resuming probe at {n:,} (last {after})", flush=True)
    while True:
        got = _net_retry(run_sql, _sql(after), timeout=900, _tries=6)
        if not got:
            break
        with PARTIAL.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS)
            if not n:
                w.writeheader()
            for r in got:
                r["sep_arcsec"] = f"{_sep(r):.4f}"
                w.writerow({k: r[k] for k in COLS})
        n, after = n + len(got), got[-1]["objID"]
        print(f"  probe {n:>7,}  (last {after})", flush=True)
        if len(got) < PAGE:
            break
    have = {r["object_id"] for r in csv.DictReader(Path("data/probe/metadata.csv").open(newline=""))}
    rows = [r for r in csv.DictReader(PARTIAL.open(newline="")) if r["objID"] in have]
    if len(rows) != PROBE_N or len(have) != PROBE_N:
        sys.exit(f"probe: {len(rows)} matched of {len(have)} on disk, expected {PROBE_N}")
    with PROBE_OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    seps = sorted(float(r["sep_arcsec"]) for r in rows)
    print(f"probe_v2: {len(rows):,} targets; GZ2→PhotoObj sep median {seps[len(seps)//2]:.3f}″, "
          f"99% {seps[int(.99*len(seps))]:.3f}″, max {seps[-1]:.3f}″; "
          f"> 0.4″ (1 px): {sum(x > 0.396 for x in seps):,}; > 1″: {sum(x > 1 for x in seps):,}; "
          f"> 3″: {sum(x > 3 for x in seps):,}; > 10″: {sum(x > 10 for x in seps):,}")


def pretrain() -> None:
    have = {r["object_id"] for r in csv.DictReader(Path("data/pretrain/metadata.csv").open(newline=""))}
    n = 0
    with (WORK / "pretrain_all_targets.csv").open(newline="") as fi, \
            PRETRAIN_OUT.open("w", newline="") as fo, FAILED16.open("w", newline="") as ff:
        rd = csv.DictReader(fi)
        wo, wf = csv.DictWriter(fo, fieldnames=rd.fieldnames), csv.DictWriter(ff, fieldnames=rd.fieldnames)
        wo.writeheader()
        wf.writeheader()
        nf = 0
        for r in rd:
            if r["objID"] in have:
                wo.writerow(r)
                n += 1
            else:
                wf.writerow(r)
                nf += 1
    if n != PRETRAIN_N or nf != 16:
        sys.exit(f"pretrain: {n} landed + {nf} failed, expected {PRETRAIN_N} + 16")
    print(f"pretrain_v2: {n:,} targets; {nf} v1 failures → {FAILED16}")


EXCLUDE_SEP = 3.0  # ″ (user, 2026-09-25): beyond this, re-centring could put another object under the votes
FLAG_SEP = 1.0  # 1″–3″: kept, flagged in cut_log (gz2_sep_arcsec)
CUT_COLS = ("objID", "ra", "dec", "run", "camcol", "field", "rerun", "v1_ra", "v1_dec", "gz2_sep_arcsec")


def cutter_lists() -> None:
    """The driver's inputs: WORK/<corpus>_all_targets.csv, cutter columns only. probe_v2 drops the
    objects whose GZ2 position sits > 3″ from PhotoObj (listed in repull_findings.md); they are
    also excluded from every M-vs-v2 comparison (EXCLUDED_PROBE)."""
    far = []
    with PROBE_OUT.open(newline="") as fi, (WORK / "probe_v2_all_targets.csv").open("w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=CUT_COLS)
        w.writeheader()
        n = 0
        for r in csv.DictReader(fi):
            if float(r["sep_arcsec"]) > EXCLUDE_SEP:
                far.append(r)
                continue
            w.writerow({**{k: r[k] for k in CUT_COLS[:7]}, "v1_ra": r["gz2_ra"], "v1_dec": r["gz2_dec"],
                        "gz2_sep_arcsec": r["sep_arcsec"]})
            n += 1
    with EXCLUDED_PROBE.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["objID", "sep_arcsec", "ra", "dec", "gz2_ra", "gz2_dec"])
        w.writeheader()
        w.writerows({k: r[k] for k in ("objID", "sep_arcsec", "ra", "dec", "gz2_ra", "gz2_dec")} for r in far)
    with PRETRAIN_OUT.open(newline="") as fi, (WORK / "pretrain_v2_all_targets.csv").open("w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=CUT_COLS)
        w.writeheader()
        m = 0
        for r in csv.DictReader(fi):
            w.writerow({**{k: r[k] for k in CUT_COLS[:7]}, "v1_ra": r["ra"], "v1_dec": r["dec"], "gz2_sep_arcsec": ""})
            m += 1
    print(f"probe_v2: {n:,} targets ({len(far)} excluded > {EXCLUDE_SEP}″ → {EXCLUDED_PROBE}); pretrain_v2: {m:,}")


EXCLUDED_PROBE = WORK / "probe_v2_excluded_gz2_sep.csv"


if __name__ == "__main__":
    {"probe": probe, "pretrain": pretrain, "cutter": cutter_lists}[sys.argv[1]]()
