import numpy as np

from galaxy_jepa.data.metadata import run_sql

# Sample 20k probe galaxies' petroRad_r (lighter than a full-table aggregate) → percentiles.
SQL = """SELECT TOP 20000 p.petroRad_r
FROM zoo2MainSpecz AS g JOIN PhotoObjAll AS p ON p.objID = g.dr8objid
ORDER BY g.dr8objid"""

rp = np.array([float(r["petroRad_r"]) for r in run_sql(SQL) if r["petroRad_r"] not in ("", "nan")])
rp = rp[np.isfinite(rp) & (rp > 0)]
print("n =", rp.size)
for q in (50, 90, 95, 99, 99.5, 99.9, 100):
    v = np.percentile(rp, q)
    px = round(2 * 2.5 * v / 0.396)  # px to hold the k=2.5 box
    print("  p%-5s petroRad = %5.1f\"  -> 2.5xR box needs %4d px" % (q, v, px))
for thr, lab in [(12.7, "64px"), (20.3, "256px"), (22.8, "288px"), (25.3, "320px")]:
    frac = 100 * np.mean(rp > thr)
    print("  clipped by %-6s (R>%.1f\"): %.2f%%" % (lab, thr, frac))
