#!/bin/zsh
# Kill a benchmark child if the machine starts paying for it in swap.
# Growth, not absolute use: this machine idles with >10 GB of swap already allocated, so the
# absolute figure says nothing. macOS grows the swap file on demand, and that growth is the
# honest signal that a measurement has stopped being free.
pid="$1"; limit_mb="${2:-6000}"
base=$(sysctl -n vm.swapusage | awk '{gsub(/M/,"",$6); print $6}')
while kill -0 "$pid" 2>/dev/null; do
  sw=$(sysctl -n vm.swapusage | awk '{gsub(/M/,"",$6); print $6}')
  if awk -v s="$sw" -v b="$base" -v l="$limit_mb" 'BEGIN{exit !(s-b > l)}'; then
    echo "WATCHDOG: swap grew $(awk -v s=$sw -v b=$base 'BEGIN{printf "%.1f", (s-b)/1024}') GB — killing $pid" >&2
    kill -9 "$pid"; exit 1
  fi
  sleep 5
done
