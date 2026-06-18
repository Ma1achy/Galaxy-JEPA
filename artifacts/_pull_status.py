"""Print the current status of a SciServer pull (cut / download / extract phases).

Usage:  python artifacts/_pull_status.py [pretrain|probe]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sciserver_auth import authenticate  # noqa: E402

CODES = {
    1: "PENDING (queued)",
    2: "QUEUED",
    4: "ACCEPTED",
    8: "STARTED — cutting server-side",
    16: "FINISHED",
    32: "SUCCESS — cut done",
    64: "ERROR",
    128: "TIMED OUT / CANCELLED",
}


def main() -> None:
    corpus = sys.argv[1] if len(sys.argv) > 1 else "pretrain"
    work = Path(".sciserver_work")
    out_dir = Path("data") / corpus

    # [3] extracted? (terminal state)
    n_fits = len(list(out_dir.glob("*.fits"))) if out_dir.exists() else 0
    if n_fits:
        print(f"[{corpus}] DONE — {n_fits} stamps extracted under {out_dir}")
        return

    # [2] downloading?
    tar = work / f"{corpus}_corpus.tar.gz"
    if tar.exists():
        print(f"[{corpus}] DOWNLOADING — {tar.stat().st_size / 1e6:.0f} MB pulled so far")
        return

    # [1] cutting? (query the job)
    state_path = work / f"{corpus}.job.json"
    if not state_path.exists():
        print(f"[{corpus}] no active pull (no state file)")
        return
    st = json.loads(state_path.read_text())
    from SciServer import Jobs

    authenticate(verbose=False)
    status = Jobs.getJobDescription(st["jid"]).get("status")
    print(f"[{corpus}] job {st['jid']} ({st.get('n_targets', '?')} targets): {CODES.get(status, status)}")


if __name__ == "__main__":
    main()
