#!/usr/bin/env bash
# Mac-side watcher for a preemptible Nebius VM running run.sh as a boot service (user, 25 Nov: "a watcher on the Mac is acceptable").
# Nebius never restarts a preempted VM (recovery policy FAIL only), so this does: every POLL seconds it asks the results bucket whether run.sh has
# written an outcome, and if not, whether the VM is stopped; a stopped VM with no outcome was preempted (or failed to boot), so it is started again.
# run.sh resumes on boot. An outcome in the bucket (ok or fail) ends the watch: run.sh stops the VM itself, and the watcher never restarts a finished
# or failed run.
#
#   VM=computeinstance-... STAGES=arms [PREFIX=grid] [POLL=300] [MAX_STARTS=40] bash artifacts/nebius_watch.sh   # your own nebius CLI profile
#
# Reads no key itself: the nebius CLI uses your profile, results.py the Mac's .env (R2_RES_*). Logs to runs/almagest/nebius_watch.log.
set -uo pipefail
cd "$(dirname "$0")/.."
: "${VM:?set VM to the instance id}" "${STAGES:?set STAGES as run.sh has it (smokes|arms|all)}"
case "$STAGES" in smokes|arms|all) ;; *) echo "STAGES must be smokes, arms or all (matching run.sh), not $STAGES" >&2; exit 1;; esac
PREFIX=${PREFIX:-grid} POLL=${POLL:-300} MAX_STARTS=${MAX_STARTS:-40}
LOG=runs/almagest/nebius_watch.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
state() {  # the status field's name is not documented: print the raw JSON once, then read the first of the likely paths
  nebius compute instance get --id "$VM" --format json 2>/dev/null | .venv/bin/python -c "
import json, sys
d = json.load(sys.stdin)
for path in (('status', 'state'), ('status', 'status'), ('state',)):
    x = d
    for k in path:
        x = x.get(k, {}) if isinstance(x, dict) else {}
    if isinstance(x, str) and x:
        print(x.upper()); break
else:
    print('UNKNOWN')"
}
nebius compute instance get --id "$VM" --format json > runs/almagest/nebius_vm_first.json 2>&1 || { say "the nebius CLI cannot read $VM"; exit 1; }
say "watching $VM (STAGES=$STAGES); first state $(state); raw JSON in runs/almagest/nebius_vm_first.json"
starts=0
while :; do
  for o in ok fail; do
    if .venv/bin/python artifacts/results.py exists "$PREFIX/OUTCOME-$o-stages-$STAGES"; then say "outcome $o in the bucket: watch ends ($starts restarts)"; exit 0; fi
  done
  s=$(state)
  case $s in
    *STOPPED*)
      [ "$starts" -ge "$MAX_STARTS" ] && { say "stopped with no outcome after $starts restarts: giving up, yours to look at"; exit 1; }
      starts=$((starts + 1)); say "state $s, no outcome: start #$starts"
      nebius compute instance start --id "$VM" >> "$LOG" 2>&1 || say "start refused (capacity or price?): retry in ${POLL}s";;
    UNKNOWN) say "state unreadable: check runs/almagest/nebius_vm_first.json for the field name";;
  esac
  sleep "$POLL"
done
