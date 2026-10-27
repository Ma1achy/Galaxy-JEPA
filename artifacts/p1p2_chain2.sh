#!/bin/bash
# P1/P2 continuation (25 Oct). The first chain's identity checks crashed in the comparator (NumPy RNG
# leaf compared with ==) after training had finished; the recompare (p1_resume.py --recompare) found
# every training-state leaf identical except rng/torch, which differs by exactly one DataLoader
# base-seed draw. Whether that passes P1 is the user's decision. These resumes and probes are made so
# the points exist if it does; nothing reads them, and they are deleted unread if it does not.
# A crash (rc not 0) is reported as a technical failure, never as an identity result.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
PY=.venv/bin/python; H=artifacts/_heavy.sh
export GJ_CHECKPOINT_EVERY=2000 PYTHONUNBUFFERED=1
LOGS=runs/_p1/logs; mkdir -p "$LOGS"
step() {
  local id=$1; shift; local t0=$(date +%s)
  echo "$(date '+%F %T') START $id"
  $H "p1p2 $id" "$@" > "$LOGS/$id.log" 2>&1; local rc=$?
  echo "$(date '+%F %T') END   $id rc=$rc $(( ($(date +%s) - t0) / 60 )) min$( [ $rc -ne 0 ] && echo '  TECHNICAL FAILURE')"
  return $rc
}
for A in A1 A2 A3; do
  a=$(echo $A | tr 'A' 'a')
  step "$A.e05" $PY artifacts/p1_resume.py $A --start 12000 --stop 12664 --out runs/_p1/${A}_e05 && \
    step "$A.traj1" $PY artifacts/baseline_trajectory_probe.py --run runs/_p1/${A}_e05 --steps 12664 --tag ${a}_traj1
  step "$A.e1" $PY artifacts/p1_resume.py $A --start 24000 --stop 25327 --out runs/_p1/${A}_e1 && \
    step "$A.traj2" $PY artifacts/baseline_trajectory_probe.py --run runs/_p1/${A}_e1 --steps 25327 --tag ${a}_traj2
  step "$A.e2" $PY artifacts/p1_resume.py $A --start 50000 --stop 50654 --out runs/_p1/${A}_e2 && \
    step "$A.traj3" $PY artifacts/baseline_trajectory_probe.py --run runs/_p1/${A}_e2 --steps 50654 --tag ${a}_traj3
done
echo "$(date '+%F %T') p1p2 chain2 finished"
