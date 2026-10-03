#!/bin/bash
# Training watchdog (user, 2026-10-03): every 5 min, check the run's process and its log. Notify if
# the process has gone, or the log has not advanced for 15 min, or the log cannot be read (the macOS
# access bug cuts reads before it kills anything). Never restarts: the user resumes by hand.
#   artifacts/train_watchdog.sh A1        launch detached via artifacts/_detach.py
# The notification is a macOS one (osascript), so it does not depend on any Claude session surviving.
# Its own record goes to ~/Library/Logs, outside the folders the access bug blocks.
run=${1:?A1 or A2}
LOG="/Users/malachy/Documents/Galaxy-JEPA/runs/aligned_$run.log"
REC="$HOME/Library/Logs/galaxy-jepa-watchdog-$run.log"
PERIOD=300 STALE=900

note() {
  echo "$(date '+%F %T') $1" >> "$REC"
  osascript -e "display notification \"$1\" with title \"Galaxy-JEPA $run\" sound name \"Basso\"" 2>/dev/null
}

echo "$(date '+%F %T') watching $run (log $LOG; every ${PERIOD}s, stale after ${STALE}s)" >> "$REC"
sleep 120  # let the launcher take the heavy-job lock and start the trainer
while true; do
  if ! pgrep -f "aligned_train.py $run" >/dev/null; then
    if grep -q "stopping after .* steps this invocation (at step 101308)" "$LOG" 2>/dev/null; then
      note "$run finished (process exited after the fixed stop)."
    else
      note "$run STOPPED: trainer process gone. Check the log; resume by hand."
    fi
    exit 0
  fi
  if ! m=$(stat -f %m "$LOG" 2>/dev/null); then
    note "$run: cannot read its log (macOS access?). Trainer still running; check it."
  elif [ $(( $(date +%s) - m )) -gt $STALE ]; then
    note "$run STALLED: log silent for $(( ($(date +%s) - m) / 60 )) min; trainer still running."
  fi
  sleep $PERIOD
done
