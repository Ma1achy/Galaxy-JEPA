"""Part B: the 20,000-galaxy test subsample for statmorph (fixed seed), with the two set-up columns it needs. One-off."""
import sys
sys.path.insert(0, "artifacts")
import numpy as np, pandas as pd
import almagest_axes as AXS
d = AXS.Data(untrained=False)
te = np.array(sorted(int(i) for i in d.setup.test_ids if int(i) in d.row))
pick = np.sort(np.random.default_rng(20261125).choice(te, 20_000, replace=False))
md = pd.read_csv("data/probe_v2/metadata.csv", usecols=["object_id", "petroRad_r", "psfWidth_r"], dtype={"object_id": str}).set_index("object_id")
o = md.loc[[str(i) for i in pick]].reset_index()
assert o.notna().all().all()
o.to_csv("runs/almagest/axes_phys/b/statmorph_ids.csv", index=False); print(len(te), len(o))
