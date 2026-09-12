#!/bin/zsh
# Kill a benchmark and everything it spawned, if the machine starts paying for it in swap.
#
# Growth, not absolute use: this machine idles with >10 GB of swap already allocated, so the
# absolute figure says nothing. macOS grows the swap file on demand, and that growth is the honest
# signal that a measurement has stopped being free.
#
# Kills the whole DESCENDANT TREE, walked from the pid. Two earlier versions got this wrong and
# both cost measurements. Killing the pid alone orphaned a dataloader point that then read the
# drive for four more minutes, contaminating everything measured afterwards and driving the swap
# file to 47 GB. Killing the process *group* (`kill -9 -pid`) looks right and does nothing here:
# a non-interactive zsh has job control off, so a background job does not become a group leader
# and grandchildren — `uv` -> python -> spawn workers — are in the caller's group, not the job's.
# An orphaned child is worse than the pressure the watchdog was watching for.
pid="$1"; limit_mb="${2:-6000}"
base=$(sysctl -n vm.swapusage | awk '{gsub(/M/,"",$6); print $6}')

descendants() {  # breadth-first over pgrep -P, deepest last
  local frontier=("$1") all=()
  while (( ${#frontier} )); do
    local next=()
    for p in $frontier; do
      all+=("$p")
      for c in ${(f)"$(pgrep -P $p 2>/dev/null)"}; do [[ -n "$c" ]] && next+=("$c"); done
    done
    frontier=($next)
  done
  print -l $all
}

reap() {
  local tree=(${(f)"$(descendants $1)"})
  for p in ${(Oa)tree}; do kill -9 "$p" 2>/dev/null; done   # children before parents
}

while kill -0 "$pid" 2>/dev/null; do
  sw=$(sysctl -n vm.swapusage | awk '{gsub(/M/,"",$6); print $6}')
  if awk -v s="$sw" -v b="$base" -v l="$limit_mb" 'BEGIN{exit !(s-b > l)}'; then
    echo "WATCHDOG: swap grew $(awk -v s=$sw -v b=$base 'BEGIN{printf "%.1f", (s-b)/1024}') GB — killing the tree under $pid" >&2
    reap "$pid"; sleep 1; reap "$pid"; exit 1
  fi
  sleep 3
done
