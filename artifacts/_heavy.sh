#!/bin/bash
# One heavy local process at a time (18 GB machine). Wrap any heavy command:
#   artifacts/_heavy.sh <label> <command...>
# Waits for the lock (a directory: mkdir is atomic), runs, releases. A stale lock whose
# holder pid is gone is taken over.
LOCK=/private/tmp/claude-501/galaxy-jepa-heavy.lock
label=$1; shift
# Scratch on the external SSD: the internal disk is short of space (user, 2026-09-28).
export TMPDIR="/Volumes/X10 Pro/galaxy-jepa/tmp" MPLCONFIGDIR="/Volumes/X10 Pro/galaxy-jepa/tmp/mpl"
mkdir -p "$MPLCONFIGDIR"
while ! mkdir "$LOCK" 2>/dev/null; do
  holder=$(cat "$LOCK/pid" 2>/dev/null)
  if [ -n "$holder" ] && ! kill -0 "$holder" 2>/dev/null; then rm -rf "$LOCK"; continue; fi
  sleep 15
done
echo $$ > "$LOCK/pid"; echo "$label" > "$LOCK/label"
trap 'rm -rf "$LOCK"' EXIT
echo "[heavy] $label: lock held $(date +%T)" >&2
"$@"
