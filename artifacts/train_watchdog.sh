#!/bin/bash
# Training watchdog (user, 2026-10-03; memory alarm 2026-10-06): checks the run's process and its log
# every 5 min, and free memory every minute. Notifies if the process has gone, the log has not
# advanced for 15 min, the log cannot be read (the macOS access bug cuts reads before it kills
# anything), or free memory has stayed below 15% for 10 min. Never stops or restarts anything.
#   artifacts/train_watchdog.sh A1        launch detached via artifacts/_detach.py
# The notification is a macOS one (osascript), so it does not depend on any Claude session surviving.
# Its own record goes to ~/Library/Logs, outside the folders the access bug blocks.
run=${1:?A1 or A2}
LOG="/Users/malachy/Documents/Galaxy-JEPA/runs/aligned_$run.log"
REC="$HOME/Library/Logs/galaxy-jepa-watchdog-$run.log"
TICK=60 PERIOD=300 STALE=900 MEM_FLOOR=15 MEM_FOR=600

note() {
  echo "$(date '+%F %T') $1" >> "$REC"
  osascript -e "display notification \"$1\" with title \"Galaxy-JEPA $run\" sound name \"Basso\"" 2>/dev/null
}
free_pct() { memory_pressure 2>/dev/null | awk '/free percentage/ {gsub("%", "", $NF); print $NF}'; }

echo "$(date '+%F %T') watching $run (log $LOG; every ${PERIOD}s, stale after ${STALE}s; memory below ${MEM_FLOOR}% for ${MEM_FOR}s)" >> "$REC"
sleep 120  # let the launcher take the heavy-job lock and start the trainer
low_since="" warned="" last=0
while true; do
  now=$(date +%s)
  f=$(free_pct)
  if [ -n "$f" ] && [ "$f" -lt "$MEM_FLOOR" ]; then
    low_since=${low_since:-$now}
    if [ -z "$warned" ] && [ $(( now - low_since )) -ge "$MEM_FOR" ]; then
      note "A1 under memory pressure: consider pausing the Docker tests (free ${f}% for $(( (now - low_since) / 60 )) min)."
      warned=1
    fi
  elif [ -n "$f" ]; then
    [ -n "$warned" ] && echo "$(date '+%F %T') memory recovered: free ${f}%" >> "$REC"
    low_since="" warned=""
  fi
  if [ $(( now - last )) -ge "$PERIOD" ]; then
    last=$now
    if ! pgrep -f "aligned_train.py $run" >/dev/null; then
      if grep -q "stopping after .* steps this invocation (at step 101308)" "$LOG" 2>/dev/null; then
        note "$run finished (process exited after the fixed stop)."
      else
        note "$run STOPPED: trainer process gone. Check the log; resume from the last checkpoint."
      fi
      exit 0
    fi
    if ! m=$(stat -f %m "$LOG" 2>/dev/null); then
      note "$run: cannot read its log (macOS access?). Trainer still running; check it."
    elif [ $(( now - m )) -gt $STALE ]; then
      note "$run STALLED: log silent for $(( (now - m) / 60 )) min; trainer still running."
    fi
  fi
  sleep $TICK
done
