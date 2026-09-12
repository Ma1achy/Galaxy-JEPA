import json, sys
from pathlib import Path
sys.path.insert(0, str(Path("artifacts").resolve()))
from _sciserver_auth import authenticate
authenticate(verbose=False)
from SciServer import Jobs

for d in Jobs.getDockerComputeDomains():
    if d.get("name") not in ("Small Jobs Domain", "Large Jobs Domain"):
        continue
    print(f"=== {d['name']} ===")
    for k, v in sorted(d.items()):
        if k in ("images", "volumes", "userVolumes"):
            print(f"  {k}: {len(v)} entries")
            continue
        print(f"  {k}: {json.dumps(v)[:300]}")
    print()
