#!/bin/zsh
# Sample the RSS of every process under a pid, plus system swap, every `interval` seconds.
# Brief G3 needs to say *where* a worker point's memory goes, not that it went somewhere: at
# num_workers=2 the swap file grew 7.9 GB while the dataset was 8.5 MB, and the only way to tell
# a dataset copy from the resident mapped pages of a 415.8 GB memmap is to watch each process.
pid="$1"; interval="${2:-2}"
base=$(sysctl -n vm.swapusage | awk '{gsub(/M/,"",$6); print $6}')
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
echo "t_s n_proc total_rss_gb per_proc_rss_gb swap_growth_gb free_pct"
t0=$(date +%s)
while kill -0 "$pid" 2>/dev/null; do
  tree=(${(f)"$(descendants $pid)"})
  rss=$(ps -o rss= -p ${(j:,:)tree} 2>/dev/null | awk '{s+=$1} END{printf "%.2f", s/1048576}')
  each=$(ps -o rss= -p ${(j:,:)tree} 2>/dev/null | awk '{printf "%.2f ", $1/1048576}')
  sw=$(sysctl -n vm.swapusage | awk '{gsub(/M/,"",$6); print $6}')
  free=$(memory_pressure | awk -F': ' '/free percentage/{gsub(/%/,"",$2); print $2}')
  printf "%s %s %s [%s] %.2f %s\n" "$(( $(date +%s) - t0 ))" "${#tree}" "$rss" "${each% }" \
    "$(awk -v s=$sw -v b=$base 'BEGIN{print (s-b)/1024}')" "$free"
  sleep "$interval"
done
