#!/bin/bash
# Kickoff E launch: A1 / A2 with every write on the SSD (the internal disk is ~16 GB free).
#   artifacts/aligned_launch.sh --resolve     print the resolved paths, launch nothing
#   artifacts/aligned_launch.sh --check       the cache-vs-freeze check alone, launch nothing
#   artifacts/aligned_launch.sh A1 [--plan]   then A2; each runs under the one-heavy-job lock
set -euo pipefail
REPO=/Users/malachy/Documents/Galaxy-JEPA
SSD="/Volumes/X10 Pro/galaxy-jepa"
export GJ_CONFIG=configs/pretrain_v2.yaml
# The v2 cache and freeze (user, 2026-10-01). Without these F0 reads M's v1 cache and refuses.
export F0_CACHE_BASE="$SSD/runs/m_v2/cache"
export F0_NORM_PREFIX=246d8de6ba9a
export TMPDIR="$SSD/tmp"               # torch / tempfile scratch
export MPLCONFIGDIR="$SSD/tmp/mpl"
export PYTHONUNBUFFERED=1              # a detached log is block-buffered otherwise
export GJ_CHECKPOINT_EVERY=2000        # run-time only, not hashed: a checkpoint every ~22 min (user, 2026-10-06)
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
  echo "F0_CACHE_BASE $F0_CACHE_BASE"
  echo "F0_NORM_PREFIX $F0_NORM_PREFIX"
  echo "all on $SSD"
}

# Refuse unless the cache under F0_CACHE_BASE was baked under GJ_CONFIG's freeze: its index's
# normalisation hash must equal the config's content_hash, F0_NORM_PREFIX must agree with both, and
# the cache must hold every stamp of both corpora.
check() {
  .venv/bin/python - <<'PY'
import os, sys, yaml
from galaxy_jepa.data.cache import TensorCache, pipeline_hash
from galaxy_jepa.data.sources import DirectorySource
from galaxy_jepa.harness import HarnessConfig, _build_pipeline
cfg = HarnessConfig(**yaml.safe_load(open(os.environ["GJ_CONFIG"])))
want = cfg.normalisation.content_hash
key = pipeline_hash(_build_pipeline(q=cfg.q, freeze=cfg.normalisation))
d = os.path.join(os.environ["F0_CACHE_BASE"], key)
print(f"config      {os.environ['GJ_CONFIG']}  freeze {want[:12]}  pipeline {key[:12]}")
if not os.path.isdir(d):
    sys.exit(f"REFUSED: no cache at {d}")
c = TensorCache(d)
got = c.index.normalisation_hash
n_want = len(DirectorySource(cfg.paths.pretrain_dir)) + len(DirectorySource(cfg.paths.probe_dir))
print(f"cache       {d}\n  normalisation {got[:12]}  stamps {c.index.n:,} of {n_want:,}")
if got != want:
    sys.exit(f"REFUSED: cache normalisation {got[:12]} != config freeze {want[:12]}")
if not want.startswith(os.environ["F0_NORM_PREFIX"]):
    sys.exit(f"REFUSED: F0_NORM_PREFIX {os.environ['F0_NORM_PREFIX']} does not prefix the freeze {want[:12]}")
if c.index.n != n_want:
    sys.exit(f"REFUSED: the cache holds {c.index.n:,} stamps, the corpora {n_want:,} (bake incomplete)")
print("OK: cache, freeze and prefix agree; cache complete")
PY
}

mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
cd "$REPO"
if [ "${1:-}" = "--resolve" ]; then resolve; exit 0; fi
if [ "${1:-}" = "--check" ]; then check; exit $?; fi
run=${1:?A1 or A2}; shift
[ -n "$(out_of "$run")" ] || { echo "run must be A1 or A2" >&2; exit 1; }
[ -f "$GJ_CONFIG" ] || { echo "$GJ_CONFIG missing: the v2 freeze comes first (Kickoff E step 3)" >&2; exit 1; }
resolve >/dev/null
check || { echo "aligned_launch: refusing to start $run" >&2; exit 1; }
# Kickoff E's disk precondition is on the external SSD, not the internal disk (user, 2026-09-28).
free_gb=$(df -g "$SSD" | awk 'NR==2 {print $4}')
[ "$free_gb" -ge 20 ] || { echo "external SSD has ${free_gb} GB free; Kickoff E needs >= 20" >&2; exit 1; }
artifacts/_heavy.sh "aligned_$run" .venv/bin/python artifacts/aligned_train.py "$run" "$@" \
  2>&1 | tee -a "$LOGDIR/aligned_$run.log"
