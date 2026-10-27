#!/bin/bash
# P1 end-to-end check under the amended rule (user, 25 Oct): a fresh resume 12,000 -> 14,000 per A run
# must reproduce the original ckpt_000014000 (training state + every generator training draws from).
# rc 0 PASS, 3 FAIL, anything else a TECHNICAL FAILURE. Never edit while running.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
PY=.venv/bin/python; H=artifacts/_heavy.sh
export GJ_CHECKPOINT_EVERY=2000 PYTHONUNBUFFERED=1
LOGS=runs/_p1/logs
src() { case $1 in A1) echo runs/m_v2;; A2) echo runs/m_v2_s1;; A3) echo runs/m_v2_s2;; esac; }
for A in A1 A2 A3; do
  t0=$(date +%s); echo "$(date '+%F %T') START $A.e2e"
  $H "p1 $A.e2e" $PY artifacts/p1_resume.py $A --start 12000 --stop 14000 --out runs/_p1/${A}_e2e \
    --compare "$(src $A)/checkpoints/ckpt_000014000.pt" > "$LOGS/$A.e2e.log" 2>&1; rc=$?
  v=$( [ $rc -eq 0 ] && echo PASS || ( [ $rc -eq 3 ] && echo FAIL || echo "TECHNICAL FAILURE" ) )
  echo "$(date '+%F %T') END   $A.e2e rc=$rc $v $(( ($(date +%s) - t0) / 60 )) min"
done
echo "$(date '+%F %T') e2e chain finished"
