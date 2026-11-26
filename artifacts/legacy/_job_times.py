import sys, time
from pathlib import Path
sys.path.insert(0, str(Path("artifacts").resolve()))
from _sciserver_auth import authenticate
authenticate(verbose=False)
from SciServer import Jobs

def ts(ms):
    return time.strftime("%H:%M:%S", time.localtime(ms / 1000)) if ms else "-"

# SciServer's import injects its own flags into argv; take only the job ids.
for jid in (int(a) for a in sys.argv[1:] if a.isdigit()):
    d = Jobs.getJobDescription(jid)
    st, en = d.get("startTime"), d.get("endTime")
    dur = f"{(en - st) / 60000:.1f} min" if (st and en) else "running"
    print(f"{jid}  status={d.get('status'):>3}  start={ts(st)} end={ts(en)}  run={dur}")
