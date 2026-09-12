#!/bin/zsh
# Kill a benchmark and everything it spawned if THE PROCESS ITSELF runs away.
#
# Supersedes f_watchdog.sh's system-swap trigger for long runs, which measured the wrong thing.
# A 3,000-step arm at batch 32 streams 96,000 stamps = 37.7 GB of memmap pages through an 18 GB
# machine, so the page cache MUST grow past RAM and macOS MUST swap. That is the kernel doing its
# job, not a leak: across the 1,775 steps the swap guard eventually killed, the arm's own MPS
# driver never moved off ~4.97 GB. H2's 500-step arms touched only 6.3 GB of pages and so never
# tripped it, which is why the 6 GB swap limit looked reasonable until the run got longer.
#
# What actually indicates trouble is the arm's own resident set. That is what this watches.
# Walks the descendant tree for the kill, for the reason recorded in f_watchdog.sh: a
# non-interactive zsh has job control off, so `kill -9 -pid` leaves grandchildren running, and an
# orphaned dataloader reading the drive contaminates every measurement taken afterwards.
pid="$1"; limit_gb="${2:-12}"; poll="${3:-15}"

descendants() {
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

tree_rss_gb() {  # summed RSS of the whole tree, in GB
  local tree=(${(f)"$(descendants $1)"}) total=0
  for p in $tree; do
    local kb=$(ps -o rss= -p "$p" 2>/dev/null | tr -d ' ')
    [[ -n "$kb" ]] && (( total += kb ))
  done
  print -- $(( total / 1048576.0 ))
}

reap() { local tree=(${(f)"$(descendants $1)"}); for p in ${(Oa)tree}; do kill -9 "$p" 2>/dev/null; done }

while kill -0 "$pid" 2>/dev/null; do
  rss=$(tree_rss_gb "$pid")
  if awk -v r="$rss" -v l="$limit_gb" 'BEGIN{exit !(r > l)}'; then
    echo "WATCHDOG: tree RSS ${rss} GB exceeded ${limit_gb} GB — killing the tree under $pid" >&2
    reap "$pid"; sleep 1; reap "$pid"; exit 1
  fi
  sleep "$poll"
done
