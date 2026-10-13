#!/bin/bash
# Ladder v2 watchdog + memory alarm (second approval, 13 October), after train_watchdog.sh: free memory
# every minute; every 5 min the runner's process, the current entry's log (by mtime only — entry logs
# carry results and are never read), and the runner log's status lines (ids and states, no values).
# Notifies when: the runner has gone without finishing; an entry's log has not moved for 45 min (some
# CPU fits are long and quiet — a prompt to look, not a verdict); an entry failed; free memory has
# stayed below 15% for 10 min; the ladder finished. Never stops or restarts anything.
#   .venv/bin/python artifacts/_detach.py runs/ladder_v2/logs/watchdog.log --name ladder_v2_watchdog -- \
#       artifacts/ladder_v2_watchdog.sh runs/ladder_v2/logs/runner_<ts>.log
RLOG=${1:?runner log}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$REPO/runs/ladder_v2/logs"
REC="$HOME/Library/Logs/galaxy-jepa-watchdog-ladder_v2.log"
TICK=60 PERIOD=300 STALE=2700 MEM_FLOOR=15 MEM_FOR=600

note() {
  echo "$(date '+%F %T') $1" >> "$REC"
  osascript -e "display notification \"$1\" with title \"Galaxy-JEPA ladder v2\" sound name \"Basso\"" 2>/dev/null
}
free_pct() { memory_pressure 2>/dev/null | awk '/free percentage/ {gsub("%", "", $NF); print $NF}'; }

echo "$(date '+%F %T') watching ladder v2 (runner log $RLOG)" >> "$REC"
sleep 120
low_since="" warned="" last=0 failed_seen=0 stale_warned=""
while true; do
  now=$(date +%s)
  f=$(free_pct)
  if [ -n "$f" ] && [ "$f" -lt "$MEM_FLOOR" ]; then
    low_since=${low_since:-$now}
    if [ -z "$warned" ] && [ $(( now - low_since )) -ge "$MEM_FOR" ]; then
      note "memory pressure: free ${f}% for $(( (now - low_since) / 60 )) min (current entry: $(grep -o '=== [0-9]*/[0-9]* [^ ]*' "$RLOG" | tail -1))"
      warned=1
    fi
  elif [ -n "$f" ]; then
    [ -n "$warned" ] && echo "$(date '+%F %T') memory recovered: free ${f}%" >> "$REC"
    low_since="" warned=""
  fi
  if [ $(( now - last )) -ge "$PERIOD" ]; then
    last=$now
    if grep -q "^.\{0,9\}ladder v2 runner: finished" "$RLOG" 2>/dev/null; then
      note "ladder finished: $(grep 'ladder v2 runner: finished' "$RLOG" | tail -1 | sed 's/.*finished — //')"
      exit 0
    fi
    if ! pgrep -f "artifacts/ladder_v2_run.py" >/dev/null; then
      note "the runner has gone without finishing (last: $(grep -o '=== [0-9]*/[0-9]* [^ ]*' "$RLOG" | tail -1))"
      exit 1
    fi
    n=$(grep -c "^.\{0,9\}    failed" "$RLOG" 2>/dev/null); n=${n:-0}
    if [ "$n" -gt "$failed_seen" ]; then
      note "$(( n - failed_seen )) entry failure(s) recorded; the runner continues"
      failed_seen=$n
    fi
    cur=$(grep -o '=== [0-9]*/[0-9]* [^ ]*' "$RLOG" | tail -1 | awk '{print $3}')
    elog="$LOGS/$cur.log"
    if [ -n "$cur" ] && [ -f "$elog" ]; then
      age=$(( now - $(stat -f %m "$elog") ))
      if [ "$age" -ge "$STALE" ] && [ "$stale_warned" != "$cur" ]; then
        note "$cur: its log has not moved for $(( age / 60 )) min"
        stale_warned=$cur
      fi
    fi
  fi
  sleep "$TICK"
done
