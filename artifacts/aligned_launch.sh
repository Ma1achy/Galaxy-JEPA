#!/bin/bash
# Kickoff E launch: A1 / A2 with every write on the SSD (the internal disk is ~16 GB free).
#   artifacts/aligned_launch.sh --resolve     print the resolved paths, launch nothing
#   artifacts/aligned_launch.sh A1 [--plan]   then A2; each runs under the one-heavy-job lock
set -euo pipefail
REPO=/Users/malachy/Documents/Galaxy-JEPA
SSD="/Volumes/X10 Pro/galaxy-jepa"
export GJ_CONFIG=configs/pretrain_v2.yaml
export TMPDIR="$SSD/tmp"               # torch / tempfile scratch
export MPLCONFIGDIR="$SSD/tmp/mpl"
export PYTHONUNBUFFERED=1              # a detached log is block-buffered otherwise
LOGDIR="$REPO/runs"                    # runs -> $SSD/runs
out_of() { case $1 in A1) echo runs/m_v2;; A2) echo runs/m_v2_s1;; *) echo "";; esac; }

resolve() {
  echo "TMPDIR       $(cd "$TMPDIR" && pwd -P)"
  echo "MPLCONFIGDIR $(cd "$MPLCONFIGDIR" && pwd -P)"
  for r in A1 A2; do
    echo "$r log       $(cd "$LOGDIR" && pwd -P)/aligned_$r.log"
    echo "$r out_dir   $(cd "$REPO/runs" && pwd -P)/$(basename "$(out_of $r)")  (checkpoints/, encoder.pt, traces.json)"
  done
  echo "curves/probe $(cd "$REPO/artifacts/out" && pwd -P)  (m2_long_run OUT; unused with --no-probe)"
  echo "GJ_CONFIG    $REPO/$GJ_CONFIG $( [ -f "$REPO/$GJ_CONFIG" ] && echo present || echo 'MISSING (written at the v2 freeze)')"
  for p in "$TMPDIR" "$LOGDIR" "$REPO/runs" "$REPO/artifacts/out"; do
    case "$(cd "$p" && pwd -P)" in "$SSD"*) ;; *) echo "NOT ON THE SSD: $p" >&2; exit 1;; esac
  done
  echo "all on $SSD"
}

mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
cd "$REPO"
if [ "${1:-}" = "--resolve" ]; then resolve; exit 0; fi
run=${1:?A1 or A2}; shift
[ -n "$(out_of "$run")" ] || { echo "run must be A1 or A2" >&2; exit 1; }
[ -f "$GJ_CONFIG" ] || { echo "$GJ_CONFIG missing: the v2 freeze comes first (Kickoff E step 3)" >&2; exit 1; }
resolve >/dev/null
# Kickoff E's disk precondition is on the external SSD, not the internal disk (user, 2026-09-28).
free_gb=$(df -g "$SSD" | awk 'NR==2 {print $4}')
[ "$free_gb" -ge 20 ] || { echo "external SSD has ${free_gb} GB free; Kickoff E needs >= 20" >&2; exit 1; }
artifacts/_heavy.sh "aligned_$run" .venv/bin/python artifacts/aligned_train.py "$run" "$@" \
  2>&1 | tee -a "$LOGDIR/aligned_$run.log"
