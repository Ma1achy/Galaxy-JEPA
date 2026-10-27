#!/bin/bash
# P1 then P2 (baselines_comparison.md, Common ground), detached, one heavy step at a time.
#   P1: per A run, the identity check (12,000 -> 14,000 must equal its own ckpt_000014000 bit for bit),
#       then -- only if identical -- the epoch checkpoints 12,664 / 25,327 / 50,654 by resume.
#       A failed identity check leaves that run's early points VOID (no substitute).
#   P2: baseline_trajectory_probe.py on each A-side point, tags a{1,2,3}_traj{1..4}; step 101,308 is
#       the run's own final checkpoint. The probe prints no AUC; nothing here reads a probe file.
# Each step's output goes to runs/_p1/logs/<id>.log; this log carries id, exit code, wall-clock.
# Never edit this file while it runs (bash reads it incrementally).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
PY=.venv/bin/python; H=artifacts/_heavy.sh
export GJ_CHECKPOINT_EVERY=2000 PYTHONUNBUFFERED=1
LOGS=runs/_p1/logs; mkdir -p "$LOGS"
step() {  # id, command...
  local id=$1; shift; local t0=$(date +%s)
  echo "$(date '+%F %T') START $id"
  $H "p1p2 $id" "$@" > "$LOGS/$id.log" 2>&1; local rc=$?
  echo "$(date '+%F %T') END   $id rc=$rc $(( ($(date +%s) - t0) / 60 )) min"
  return $rc
}
src() { case $1 in A1) echo runs/m_v2;; A2) echo runs/m_v2_s1;; A3) echo runs/m_v2_s2;; esac; }
OK=" "
for A in A1 A2 A3; do
  if step "$A.identity" $PY artifacts/p1_resume.py $A --start 12000 --stop 14000 \
       --out runs/_p1/${A}_identity --compare "$(src $A)/checkpoints/ckpt_000014000.pt"; then
    OK="$OK$A "
    step "$A.e05" $PY artifacts/p1_resume.py $A --start 12000 --stop 12664 --out runs/_p1/${A}_e05
    step "$A.e1"  $PY artifacts/p1_resume.py $A --start 24000 --stop 25327 --out runs/_p1/${A}_e1
    step "$A.e2"  $PY artifacts/p1_resume.py $A --start 50000 --stop 50654 --out runs/_p1/${A}_e2
  else
    echo "$(date '+%F %T') $A identity NOT passed: early points VOID for $A (P1)"
  fi
done
for A in A1 A2 A3; do
  a=$(echo $A | tr 'A' 'a')
  if [[ "$OK" == *" $A "* ]]; then
    i=1
    for pair in e05:12664 e1:25327 e2:50654; do
      d=runs/_p1/${A}_${pair%%:*}; s=${pair##*:}
      [ -f "$d/checkpoints/ckpt_$(printf %09d $s).pt" ] && \
        step "$A.traj$i" $PY artifacts/baseline_trajectory_probe.py --run "$d" --steps $s --tag ${a}_traj$i
      i=$((i + 1))
    done
  fi
  step "$A.traj4" $PY artifacts/baseline_trajectory_probe.py --run "$(src $A)" --steps 101308 --tag ${a}_traj4
done
echo "$(date '+%F %T') p1p2 chain finished"
