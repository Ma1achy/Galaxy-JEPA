"""Part B pull (brief, 24 Nov 2026): SDSS photometry and MPA-JHU (galSpecExtra) for the 74,825-galaxy union, public SkyServer, no token.
Batches of 100 kept as they land (jsonl), so a rerun resumes; the assembled CSV is pinned by SHA-1.   .venv/bin/python artifacts/axes_b_pull.py"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from axes_c_star import retry

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs/almagest/axes_phys/b"
B = 100
P = ("petroR50_r petroR90_r petroRad_r fracDeV_r expAB_r deVAB_r expRad_r deVRad_r clean flags "
     "petroMag_u petroMag_g petroMag_r petroMag_i petroMag_z petroMagErr_u petroMagErr_g petroMagErr_r petroMagErr_i petroMagErr_z "
     "modelMag_u modelMag_g modelMag_r modelMag_i modelMag_z modelMagErr_u modelMagErr_g modelMagErr_r modelMagErr_i modelMagErr_z "
     "extinction_u extinction_g extinction_r extinction_i extinction_z").split()
SQL = ("SELECT CAST(p.objID AS varchar(20)) AS objID, " + ", ".join(f"p.{c}" if c != "flags" else "CAST(p.flags AS varchar(24)) AS flags" for c in P) +
       ", CAST(s.specObjID AS varchar(24)) AS specObjID, s.z AS spec_z, s.zWarning, s.class AS spec_class, e.lgm_tot_p16, e.lgm_tot_p50, e.lgm_tot_p84, "
       "e.specsfr_tot_p16, e.specsfr_tot_p50, e.specsfr_tot_p84, e.bptclass, i.reliable AS mpa_reliable, i.targettype "
       "FROM PhotoObjAll p LEFT JOIN SpecObj s ON s.bestObjID = p.objID LEFT JOIN galSpecExtra e ON e.specObjID = s.specObjID "
       "LEFT JOIN galSpecInfo i ON i.specObjID = s.specObjID WHERE p.objID IN ({})")
if __name__ == "__main__":
    if (OUT / "sdss_record.json").exists():
        raise SystemExit("pulled already; pinned")
    ids = [str(i) for i in np.load(REPO / "runs/almagest/a1/a1_union_ids.npy")]
    part = OUT / "sdss_batches.jsonl"
    got = {}
    if part.exists():
        for ln in part.open():
            r = json.loads(ln)
            got[r["k"]] = r["rows"]
    t0 = time.time()
    with part.open("a") as fh:
        for k in range(0, len(ids), B):
            if k in got:
                continue
            got[k] = retry(SQL.format(",".join(ids[k:k + B])))
            fh.write(json.dumps({"k": k, "rows": got[k]}) + "\n"); fh.flush()
            if k // B % 25 == 0:
                print(k, f"{time.time() - t0:.0f}s", flush=True)
            time.sleep(0.5)
    df = pd.DataFrame([r for k in sorted(got) for r in got[k]])
    dup = int(df.objID.duplicated().sum())
    df = df.sort_values(["objID", "zWarning"]).drop_duplicates("objID")
    df.to_csv(OUT / "sdss.csv", index=False)
    rec = {"when": time.strftime("%Y-%m-%d %H:%M"), "service": "public SkyServer SQL, DR17, no token", "query": SQL, "batch": B, "requested": len(ids),
           "rows": len(df), "duplicate_spectrum_rows_dropped": dup, "missing": len(set(ids) - set(df.objID)), "with_spectrum": int((df.specObjID.fillna("") != "").sum()),
           "with_mpa": int(pd.to_numeric(df.lgm_tot_p50, errors="coerce").gt(-9000).sum()), "sha1": hashlib.sha1((OUT / "sdss.csv").read_bytes()).hexdigest()}
    (OUT / "sdss_record.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "query"}, indent=1))
