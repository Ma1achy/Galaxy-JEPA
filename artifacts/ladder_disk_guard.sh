#!/bin/bash
# Disk guard for a ladder run (user, 4 Nov): other sessions share the runs SSD and their build output swings
# by tens of GB. Every minute it reads the volume's free space and stops the ladder only through
# ladder_v2_ctl.py's own pause, so every stop is clean and resumable:
#   below SOFT GB  pause at the next entry boundary (nothing lost; the current entry finishes)
#   below HARD GB  pause --now (the current entry is cleared and reruns from scratch on resume)
# It reads no output or log. `ladder_v2_ctl.py resume` starts a fresh one.
#   artifacts/ladder_disk_guard.sh <manifest> <name> [SOFT] [HARD]
MANIFEST=${1:?manifest} NAME=${2:?name} SOFT=${3:-30} HARD=${4:-15}
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO" || exit 1
CTL=(.venv/bin/python artifacts/ladder_v2_ctl.py --manifest "$MANIFEST")
VOL="/Volumes/X10 Pro"
REC="$HOME/Library/Logs/galaxy-jepa-diskguard-$NAME.log"
note() {
  echo "$(date '+%F %T') $1" >> "$REC"; echo "$(date '+%F %T') $1"
  osascript -e "display notification \"$1\" with title \"Galaxy-JEPA $NAME disk guard\" sound name \"Basso\"" 2>/dev/null
}
free_gb() { df -g "$VOL" 2>/dev/null | awk 'NR==2{print $4}'; }
note "watching $VOL for $NAME: boundary pause below $SOFT GB, pause --now below $HARD GB"
sleep 30
soft_sent=""
while true; do
  f=$(free_gb)
  if [ -z "$f" ]; then note "$VOL is not mounted; the guard stops"; exit 1; fi
  if ! "${CTL[@]}" status 2>/dev/null | grep -q "runner: alive"; then
    echo "$(date '+%F %T') runner not alive; the guard stops" >> "$REC"; exit 0
  fi
  if [ "$f" -lt "$HARD" ]; then
    note "free ${f} GB < $HARD: pause --now (the entry reruns on resume)"; "${CTL[@]}" pause --now; exit 0
  elif [ "$f" -lt "$SOFT" ] && [ -z "$soft_sent" ]; then
    note "free ${f} GB < $SOFT: pause at the next entry boundary"; "${CTL[@]}" pause; soft_sent=1
  fi
  sleep 60
done
