#!/bin/bash
# M's criterion-4 references on the 4,999-galaxy panel: dd_v4ref's chain, panel without …3595058
# (user, 2026-10-09). V3′ (part1b v3) is not rerun: its 794 galaxies contain none of the 9.
cd /Users/malachy/Documents/Galaxy-JEPA
export DD_OUT=runs/dd_v4ref_4999 DD_LOCAL=runs/dd_local_4999 OMP_NUM_THREADS=4
run() { echo "=== $* $(date +%T)"; /usr/bin/time -l .venv/bin/python "$@" > "runs/dd_v4ref_4999/log_$(basename $1 .py)_$2.txt" 2>&1; echo "exit $?"; }
run artifacts/dd_sae.py evaluate M 11 8
run artifacts/dd_sae_score.py score
run artifacts/dd_part4b.py flags
run artifacts/dd_part4b.py stats
run artifacts/dd_part4b.py pools
run artifacts/dd_part4b.py run
echo ALLDONE
