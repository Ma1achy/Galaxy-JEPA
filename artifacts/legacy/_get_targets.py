from galaxy_jepa.data.metadata import run_sql

SQL = """SELECT TOP 5 g.dr8objid AS objID, g.ra, g.dec, p.run, p.camcol, p.field, p.rerun,
 p.petroRad_r, p.modelMag_r, g.t04_spiral_a08_spiral_fraction AS f_spiral
FROM zoo2MainSpecz AS g JOIN PhotoObjAll AS p ON p.objID=g.dr8objid
WHERE p.modelMag_r BETWEEN 16.3 AND 17.6 AND p.petroRad_r BETWEEN 4 AND 9
 AND g.t04_spiral_a08_spiral_fraction > 0.5 ORDER BY g.dr8objid"""

for r in run_sql(SQL):
    vals = (int(r["objID"]), float(r["ra"]), float(r["dec"]), int(r["run"]),
            int(r["camcol"]), int(r["field"]), int(r["rerun"]))
    note = "r=%.1f Rp=%.1f fsp=%.2f" % (
        float(r["modelMag_r"]), float(r["petroRad_r"]), float(r["f_spiral"]))
    print("    (%d, %.5f, %.5f, %d, %d, %d, %d),  # %s" % (*vals, note))
