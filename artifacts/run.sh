#!/usr/bin/env bash
# Patch grid, the whole rental unattended (docs/patch_grid/runbook.md, third version; user, 25 Nov 2026):
#   preflight → environment → pull + verify cache → parity → stage 1 checklist (nine smokes; stop on failure) → masks →
#   stage 1 arms (patch 16, 8) → judge (stop on failure) → stage 2 arms (patch 4) → 36 embeddings → results up → shut the machine down.
# Every step is idempotent and asks the results bucket (gj-results) what is already done, so a preempted or restarted box reruns this same
# command and carries on: a finished arm is skipped, a running one resumes from its newest checkpoint (local disk, else the bucket).
#
#   NVME=/data MANIFEST_SHA1=<from the Mac> STAGES=smokes|arms|all bash artifacts/run.sh     # as a boot service (runbook), or in tmux
# Nebius plan (user, 26 Nov): STAGES=smokes on an on-demand VM (the stage 1 checklist: nine uninterrupted 2,000-step smokes, judged strictly), then
# STAGES=arms on one preemptible VM: all 27 arms with resume, and the 36 embeddings. The checklist judges only the smokes; the real arms' rows are
# recorded (state/judge_info.txt) and gate nothing, because a resumed arm's rate can't be measured.
#
# Inputs (environment): SPOT=1 shortens patch-4 checkpoint intervals (every_for); NVME (local disk), MANIFEST_SHA1 (the cache manifest's SHA-1, by hand), STAGES = smokes | arms | all (default all),
#   WORKERS (loader workers per arm, default 8), GPUS (default: every GPU nvidia-smi lists), PREFIX (bucket folder, default grid),
#   SHUTDOWN_OK / SHUTDOWN_FAIL: the commands that stop billing. On RunPod both default to `runpodctl pod stop $RUNPOD_POD_ID`; anywhere else they
#   must be given (an in-VM `shutdown` does not stop a Nebius bill), or set to `none` to leave the box running on purpose.
# Credentials: ~/Galaxy-JEPA/.env (chmod 600): R2_ENDPOINT, R2_RO_* (cache bucket, read only), R2_RES_* (gj-results only). Read by ship.py and
#   results.py; this script never reads or prints them.
# Never edits a pinned file; patch_grid_pins.py must say 101 files match or nothing runs.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
: "${NVME:?set NVME to the local disk}" "${MANIFEST_SHA1:?set MANIFEST_SHA1 to the cache manifest SHA-1 printed on the Mac}"
STAGES=${STAGES:-all} WORKERS=${WORKERS:-8} PREFIX=${PREFIX:-grid}
STOP=101308
LOG=$NVME/run.log
STATE=$NVME/state
mkdir -p "$STATE" "$NVME/masks"
# runs/ lives on the data disk: checkpoints are kept whole by the launcher (18 stage-1 arms ≈ 150–190 GB; nine patch-4 arms at the SPOT intervals ≈ 230 GB),
# and on Nebius the data disk is what carries from box 1 to box 2
mkdir -p "$NVME/runs"
if [ ! -L runs ]; then rmdir runs 2>/dev/null; [ -e runs ] && { echo "runs/ exists and is not empty: move it to $NVME/runs first"; exit 1; }; ln -s "$NVME/runs" runs; fi
exec > >(tee -a "$LOG") 2>&1
say() { echo "$(date -u +%FT%TZ) run.sh | $*"; }
PY=$REPO/.venv/bin/python
res() { "$PY" artifacts/results.py "$@"; }

# Nebius: an in-VM `shutdown` keeps billing, so the VM stops itself through the API, as its attached service account (role
# compute.instance-power-operator, bound to this VM only). Its id comes from the metadata service.
if [ -z "${RUNPOD_POD_ID:-}" ] && NEBIUS_VM=$(curl -sf -m 3 -H "Metadata: true" http://metadata.nebius.internal/v1/instance-data/id); then
  SHUTDOWN_OK=${SHUTDOWN_OK:-"nebius compute instance stop --id $NEBIUS_VM"} SHUTDOWN_FAIL=${SHUTDOWN_FAIL:-"nebius compute instance stop --id $NEBIUS_VM"}
fi
if [ -n "${RUNPOD_POD_ID:-}" ]; then
  SHUTDOWN_OK=${SHUTDOWN_OK:-"runpodctl pod stop $RUNPOD_POD_ID"} SHUTDOWN_FAIL=${SHUTDOWN_FAIL:-"runpodctl pod stop $RUNPOD_POD_ID"}
fi
: "${SHUTDOWN_OK:?set SHUTDOWN_OK (the command that stops billing here, or none)}" "${SHUTDOWN_FAIL:?set SHUTDOWN_FAIL (or none)}"

upload_log() { cp "$LOG" "$STATE/log.snap" && res put "$STATE/log.snap" "$PREFIX/logs/$(hostname)-$BOOT.log" >/dev/null 2>&1 || true; }  # a snapshot: a growing file fails the MD5 check
BOOT=$(date -u +%Y%m%dT%H%M%SZ)
( while sleep 300; do upload_log; done ) & LOGLOOP=$!

finish() {  # $1 = ok | fail, $2 = why
  trap - EXIT
  kill "$LOGLOOP" 2>/dev/null
  echo "$1 $(date -u +%FT%TZ) $2" > "$STATE/outcome"
  res put "$STATE/outcome" "$PREFIX/OUTCOME-$1-stages-$STAGES" >/dev/null 2>&1 || true
  say "END $1: $2"
  upload_log
  local cmd=$SHUTDOWN_OK; [ "$1" = ok ] || cmd=$SHUTDOWN_FAIL
  [ "$cmd" = none ] && { say "shutdown: none (left running on purpose; it bills)"; exit 0; }
  sync; say "shutdown: $cmd"; upload_log; eval "$cmd"; exit 0
}
fail() { finish fail "$*"; }
on_term() { trap - EXIT; say "SIGTERM (preemption?): pushing the newest checkpoints, then exiting; rerun resumes"
  for d in runs/patch_grid/*/ runs/patch_grid_rerun/*/; do [ -f "$d/checkpoints/checkpoints.json" ] && [ ! -f "$d/traces.json" ] && \
    timeout 45 flock "${d%/}.ckpt.lock" "$PY" artifacts/results.py ckpt-push "${d%/}" "$PREFIX/arms/$(basename "$(dirname "$d")")/$(basename "$d")" >/dev/null 2>&1 & done
  wait; upload_log; exit 143; }
trap on_term TERM
trap 'fail "run.sh exited without reaching an end (an unset variable, or a bug)"' EXIT   # subshells don't inherit it: a GPU worker never shuts the box

# --- 0. preflight: refuse before the expensive part, not after it -----------------------------------------------------------
say "preflight: STAGES=$STAGES WORKERS=$WORKERS PREFIX=$PREFIX NVME=$NVME"
command -v nvidia-smi >/dev/null || fail "no nvidia-smi"
NG=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
GPUS=${GPUS:-$(seq -s ' ' 0 $((NG - 1)))}
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
case $STAGES in smokes|arms|all) ;; *) fail "STAGES must be smokes, arms or all, not $STAGES";; esac
if :; then  # MAE at patch 4 (its smoke too) needs about 32 GB ± 30%
  for m in $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits); do [ "$m" -ge 70000 ] || fail "a GPU has ${m} MiB: stage 2 needs 80-GB cards"; done
fi
[ "$(stat -c %a .env 2>/dev/null)" = 600 ] || fail ".env missing or not chmod 600"
[ "$SHUTDOWN_OK" = none ] || command -v "${SHUTDOWN_OK%% *}" >/dev/null || fail "SHUTDOWN_OK's command is not installed: ${SHUTDOWN_OK%% *}"
[ -n "${NEBIUS_VM:-}" ] && [ "$SHUTDOWN_OK" != none ] && { nebius compute instance get --id "$NEBIUS_VM" >/dev/null 2>&1 || fail "the nebius CLI cannot see this VM ($NEBIUS_VM): it could not stop it at the end"; }
[ -n "${RUNPOD_POD_ID:-}" ] && { runpodctl pod get "$RUNPOD_POD_ID" >/dev/null 2>&1 || fail "runpodctl cannot see this pod: it could not stop it at the end"; }
for t in rclone zstd flock uv; do command -v $t >/dev/null || fail "missing $t (runbook §2)"; done
nproc; free -g | head -2; df -h "$NVME" | tail -1

# --- 1. environment ---------------------------------------------------------------------------------------------------------
uv sync --frozen --extra data --extra eval -q || fail "uv sync"
"$PY" -c "import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.device_count())" || fail "torch cannot use the GPU"
"$PY" artifacts/patch_grid_pins.py > "$STATE/pins.txt" 2>&1; grep -q "101 files match" "$STATE/pins.txt" || fail "pins: this checkout is not the code of record (an amendment is the user's)"
res selftest >/dev/null || fail "results.py selftest"
res writetest || fail "the results token cannot write to gj-results"
res exists "$PREFIX/OUTCOME-ok-stages-$STAGES" && { say "the bucket already says this grid finished"; finish ok "nothing left to do"; }
"$PY" -m pip freeze > "$STATE/pip_freeze.txt" 2>/dev/null || uv pip freeze > "$STATE/pip_freeze.txt"
nvidia-smi -q > "$STATE/nvidia_smi.txt"; git rev-parse HEAD > "$STATE/commit.txt"
res putdir "$STATE" "$PREFIX/box/$(hostname)-$BOOT" >/dev/null

# --- 2. pull + verify the cache (ship.py: manifest SHA-1, every compressed and raw SHA-1, the tiling; resumes) ----------------
PH=eab81d179579f2f083f11b2301a662db781bd395e664f5ba5d71041163d5abfd
if ! grep -q "\"$MANIFEST_SHA1\"" "$NVME/cache/$PH/cache_verified.json" 2>/dev/null; then
  say "pull"
  for i in 1 2 3; do "$PY" artifacts/ship.py pull --manifest-sha1 "$MANIFEST_SHA1" --dest "$NVME/cache" --workers 8 && break; say "pull attempt $i failed"; sleep 60; done
  grep -q "\"$MANIFEST_SHA1\"" "$NVME/cache/$PH/cache_verified.json" 2>/dev/null || fail "the cache did not verify"
fi
export GJ_CONFIG=configs/pretrain_v2.yaml F0_NORM_PREFIX=246d8de6ba9a F0_CACHE_BASE=$NVME/cache GJ_CACHE_MANIFEST_SHA1=$MANIFEST_SHA1

# --- 3. parity: F0 and the nine config hashes computed on the Mac for device cuda -------------------------------------------
"$PY" artifacts/f0_preconditions.py > "$STATE/f0.txt" 2>&1; tail -3 "$STATE/f0.txt"; grep -q "PASS" "$STATE/f0.txt" || fail "F0 did not pass"
declare -A HASH=([jepa_p16]=763cba6a9ce2ca0a [moco_p16]=67ef97b0d148d37e [mae_p16]=dad0a614a1c5c17a [jepa_p8]=7b3a08b6eedeb158 [moco_p8]=92f90272ff77f702
                 [mae_p8]=b4d7e90e2cd42a9a [jepa_p4]=f5a5e85e35a7def2 [moco_p4]=5d8d494750a41020 [mae_p4]=9ba716d915c82856)
for k in "${!HASH[@]}"; do
  p=$("$PY" artifacts/patch_grid_run.py "${k}_s0" --plan $([ "$k" = jepa_p4 ] && echo --masks none) 2>/dev/null)
  echo "$p" | grep -q "v2:${HASH[$k]}" || fail "parity: $k's config hash is not ${HASH[$k]}: a different experiment"
  echo "$p" | grep -q "train stamps 810,491" || fail "parity: $k's train split is not 810,491 stamps"
  echo "$p" | grep -q "device cuda" || fail "parity: $k does not resolve to cuda"
done
say "parity: F0 PASS, nine config hashes, 810,491 train stamps, cuda"

S1=runs/patch_grid/stage1
mkdir -p "$S1"
res getdir "$PREFIX/stage1" "$S1" >/dev/null || true   # records a previous box wrote (smokes, real arms); the judge reads them

# --- the GPU pool: one worker per GPU takes the next task from a queue file under flock ----------------------------------------
pool() {  # $1 = queue file (one task per line), $2 = task function; returns non-zero if any task failed
  local q=$1 fn=$2 pids=()
  : > "$q.failed"
  for g in $GPUS; do
    ( export CUDA_VISIBLE_DEVICES=$g
      while t=$(flock "$q.lock" sh -c "head -1 '$q'; sed -i 1d '$q'") && [ -n "$t" ]; do
        $fn $t || echo "$t" >> "$q.failed"
      done ) & pids+=($!)
  done
  wait "${pids[@]}"
  [ ! -s "$q.failed" ] || { say "failed in $q: $(tr '\n' ' ' < "$q.failed")"; return 1; }
}

# --- an arm: skip if done, restore, train under watch, keep the newest checkpoint in the bucket, VOID → one rerun --------------
arm_dir() { [ -d "runs/patch_grid_rerun/$1" ] && echo "runs/patch_grid_rerun/$1" || echo "runs/patch_grid/$1"; }
prune_local() {  # user, 26 Nov: the newest two committed checkpoints stay on the disk (the newest is the final one at the end); the launcher keeps all
  "$PY" - "$1/checkpoints" <<'EOF2'
import json, sys; from pathlib import Path
d = Path(sys.argv[1]); e = sorted(json.loads((d / "checkpoints.json").read_text())["entries"], key=lambda x: int(x["step"]))
keep = {x["file"] for x in e[-2:]}; floor = int(e[-2]["step"]) if len(e) > 1 else 0
for f in d.glob("ckpt_*.pt"):
    if f.name not in keep and int(f.stem.split("_")[1]) < floor:
        f.unlink()
EOF2
}
committed_step() { "$PY" -c "import json,sys; e=json.load(open(sys.argv[1]))['entries']; print(max(int(x['step']) for x in e) if e else 0)" "$1/checkpoints/checkpoints.json" 2>/dev/null || echo 0; }
is_void() { "$PY" - "$1" <<'EOF'
import json, sys; sys.path.insert(0, "artifacts")
from patch_grid_auc import diverged
r = json.load(open(sys.argv[1] + "/traces.json"))
sys.exit(0 if r.get("halted") or diverged(r["losses"])["diverged"] else 1)
EOF
}
masks_for() { case $1 in jepa_p4_s*) echo "--masks $NVME/masks/s${1##*_s}";; esac; }
# SPOT=1 (user, 25 Nov: stage 2 on spot): patch-4 arms checkpoint about every 5–10 min at the cost doc's hours (MAE 4.2–8.3 h → 5.0–9.8 min,
# JEPA 2.7–5.3 h → 5.0–9.9, MoCo 2.2–4.6 h → 4.8–9.9). GJ_CHECKPOINT_EVERY is outside the config hash; stage 1 keeps the recipe's 6,331.
every_for() { [ "${SPOT:-0}" = 1 ] || return 0; case $1 in mae_p4_*) echo 2000;; jepa_p4_*) echo 3150;; moco_p4_*) echo 3650;; esac; }

train_one() {  # $1 arm, $2 out dir, $3 = watch | rerun
  local a=$1 d=$2 key=$PREFIX/arms/$(basename "$(dirname "$2")")/$1
  if [ ! -f "$d/checkpoints/checkpoints.json" ]; then res ckpt-pull "$key" "$d" && say "$a: restored step $(committed_step "$d") from the bucket"; fi
  if [ "$(committed_step "$d")" -ge "$STOP" ] && [ ! -f "$d/traces.json" ]; then
    say "$a: HOLD: committed at step $STOP but no traces.json (killed between the last checkpoint and the trace); the launcher would train one step past STOP. Yours to decide"
    echo "$a" >> "$STATE/holds"; return 1
  fi
  [ -f "$S1/$a.log" ] && mv "$S1/$a.log" "$S1/$a.log.$(date -u +%s)"   # a resumed invocation keeps the earlier log beside it
  ( last=0; while sleep 60; do s=$(committed_step "$d"); [ "$s" -gt "$last" ] && flock "$d.ckpt.lock" "$PY" artifacts/results.py ckpt-push "$d" "$key" >/dev/null 2>&1 && last=$s && prune_local "$d"; done ) & local sync=$!
  local rc=0 ev; ev=$(every_for "$a")
  if [ -n "$ev" ]; then export GJ_CHECKPOINT_EVERY=$ev; say "$a: checkpoint every $ev steps (SPOT)"; else unset GJ_CHECKPOINT_EVERY; fi
  if [ "$3" = watch ]; then "$PY" artifacts/patch_grid_stage1.py watch "$a" --workers "$WORKERS" $(masks_for "$a") || rc=$?
  else "$PY" artifacts/patch_grid_run.py "$a" --workers "$WORKERS" --out "$d" $(masks_for "$a") > "$d.log" 2>&1 || rc=$?; fi
  kill $sync 2>/dev/null; wait $sync 2>/dev/null
  flock "$d.ckpt.lock" "$PY" artifacts/results.py ckpt-push "$d" "$key" >/dev/null || rc=1   # after any push in flight (the lock); the final full checkpoint stays
  prune_local "$d" 2>/dev/null
  [ $rc = 0 ] || return $rc
  res put "$d/traces.json" "$key/traces.json" && res put "$d/encoder.pt" "$key/encoder.pt" || return 1
  [ "$3" = watch ] && res putdir "$S1" "$PREFIX/stage1" >/dev/null
  return 0
}

run_arm() {
  local a=$1
  for base in patch_grid patch_grid_rerun; do  # finished on an earlier box: fetch what the embeddings need, nothing else
    if res exists "$PREFIX/arms/$base/$a/traces.json"; then
      mkdir -p "runs/$base/$a"
      [ -f "runs/$base/$a/encoder.pt" ] || res get "$PREFIX/arms/$base/$a/encoder.pt" "runs/$base/$a/encoder.pt"
      [ -f "runs/$base/$a/traces.json" ] || res get "$PREFIX/arms/$base/$a/traces.json" "runs/$base/$a/traces.json"
    fi
  done
  if [ ! -f "runs/patch_grid/$a/traces.json" ]; then
    say "$a: train (GPU ${CUDA_VISIBLE_DEVICES:-unset})"
    train_one "$a" "runs/patch_grid/$a" watch || { say "$a: launcher failed; one retry (resume)"; train_one "$a" "runs/patch_grid/$a" watch || return 1; }
  fi
  if is_void "runs/patch_grid/$a" && [ ! -f "runs/patch_grid_rerun/$a/traces.json" ]; then
    say "$a: VOID; the one rerun, same seed and recipe, fresh directory"
    mkdir -p runs/patch_grid_rerun
    train_one "$a" "runs/patch_grid_rerun/$a" rerun || return 1
  fi
  say "$a: done$(is_void "$(arm_dir "$a")" && echo ' (VOID)')"
}

# --- 4. stage 1 checklist: nine smokes, one at a time on GPU 0 (a shared GPU confounds rate) ---------------------------------
smoke_rows_ok() { "$PY" - <<'EOF'
import json, sys; sys.path.insert(0, "artifacts")
import patch_grid_stage1 as S
L = lambda f: json.loads((S.OUT / f).read_text()) if (S.OUT / f).exists() else None
bad = [a for o in S.OBJ for p in S.PATCHES for a in [f"{o}_p{p}_s0"] if not all(S.verdicts(a, L(f"{a}_smoke.json"), L(f"{a}_flash.json"), L(f"{a}_roundtrip.json")).values())]
print("smoke rows:", "all ok" if not bad else "FAIL " + " ".join(bad)); sys.exit(1 if bad else 0)
EOF
}
masks_build() {  # $1 seed; CPU only, niced beside the GPU work
  local s=$1 d=$NVME/masks/s$1
  res exists "$PREFIX/masks/s$s/CHECKED" && { [ -d "$d" ] || res getdir "$PREFIX/masks/s$s" "$d" >/dev/null; }
  if [ ! -f "$d/CHECKED" ]; then
    rm -rf "$d"
    nice -n 10 "$PY" artifacts/patch_grid_masks.py build --seed "$s" --grid 64 --workers $(( $(nproc) / 2 - 10 > 4 ? $(nproc) / 2 - 10 : 4 )) --out "$d" || return 1
    "$PY" artifacts/patch_grid_masks.py check --seed "$s" --grid 64 --out "$d" > "$STATE/masks_s$s.check" 2>&1; grep -q "0 batches differ" "$STATE/masks_s$s.check" || return 1
    date -u > "$d/CHECKED"
    res putdir "$d" "$PREFIX/masks/s$s" >/dev/null || return 1
  fi
}

if [ "$STAGES" != arms ]; then
  if ! smoke_rows_ok >/dev/null 2>&1; then
    say "stage 1 checklist: smokes"
    [ -d "$NVME/masks/smoke_s0" ] || "$PY" artifacts/patch_grid_masks.py build --seed 0 --grid 64 --steps 2000 --workers 16 --out "$NVME/masks/smoke_s0" || fail "smoke masks"
    for o in jepa moco mae; do for p in 16 8 4; do A=${o}_p${p}_s0
      [ -f "$S1/${A}_smoke.json" ] && [ -f "$S1/${A}_flash.json" ] && [ -f "$S1/${A}_roundtrip.json" ] && continue
      export CUDA_VISIBLE_DEVICES=${GPUS%% *}
      [ -f "runs/patch_grid/smoke_$A/traces.json" ] || rm -rf "runs/patch_grid/smoke_$A"   # a smoke can't resume (its checkpoint is its stop)
      "$PY" artifacts/patch_grid_stage1.py flash "$A" || fail "flash $A"
      "$PY" artifacts/patch_grid_stage1.py watch "$A" --smoke 2000 $([ $o$p = jepa4 ] && echo --masks "$NVME/masks/smoke_s0") || fail "smoke $A"
      "$PY" artifacts/patch_grid_stage1.py roundtrip "$A" --run "runs/patch_grid/smoke_$A" || fail "roundtrip $A"
      res putdir "$S1" "$PREFIX/stage1" >/dev/null
    done; done
    unset CUDA_VISIBLE_DEVICES
  fi
  smoke_rows_ok || fail "stage 1 checklist: a smoke row failed. Nothing is loosened on the box"
  res putdir "$S1" "$PREFIX/stage1" >/dev/null
  [ "$STAGES" = smokes ] && finish ok "stage 1 checklist PASS: nine smokes, flash and round trip"
fi
smoke_rows_ok || fail "the stage 1 checklist records (from the bucket) do not pass: no arm runs"

# --- 5. masks for jepa_p4, beside stage 1 -------------------------------------------------------------------------------------
( for s in 0 1 2; do masks_build $s || { echo "SEED $s" >> "$STATE/masks_refused"; }; done ) & MASKS=$!

# --- 6. stage 1: patch 16 and 8, 18 arms; then the judge --------------------------------------------------------------------
if :; then
  : > "$STATE/q1"; for s in 0 1 2; do for p in 16 8; do for o in jepa moco mae; do echo ${o}_p${p}_s${s} >> "$STATE/q1"; done; done; done
  pool "$STATE/q1" run_arm || fail "stage 1: an arm failed twice"
  "$PY" artifacts/patch_grid_auc.py valid --runs runs/patch_grid | tee "$STATE/valid_stage1.txt"
  "$PY" artifacts/patch_grid_stage1.py judge > "$STATE/judge_info.txt" 2>&1 || true   # information only (user, 26 Nov): a resumed arm's throughput row reads FAIL
  res putdir "$STATE" "$PREFIX/state" >/dev/null
fi
wait $MASKS
[ -s "$STATE/masks_refused" ] && fail "patch-4 masks refused: $(cat "$STATE/masks_refused")"

# --- 7. stage 2: patch 4, longest first so the GPUs finish together -----------------------------------------------------------
: > "$STATE/q2"; for o in mae jepa moco; do for s in 0 1 2; do echo ${o}_p4_s${s} >> "$STATE/q2"; done; done
pool "$STATE/q2" run_arm || fail "stage 2: an arm failed twice"
"$PY" artifacts/patch_grid_auc.py valid --runs runs/patch_grid | tee "$STATE/valid.txt"

# --- 8. embeddings: 27 arms + 9 untrained banks, fp32, the ladder's recipe -----------------------------------------------------
for a in $(for p in 16 8; do for s in 0 1 2; do for o in jepa moco mae; do echo ${o}_p${p}_s${s}; done; done; done); do run_arm "$a" >/dev/null; done  # stage 1's encoders on a stage-2 box
EMB=runs/patch_grid/emb
mkdir -p "$EMB"
embed() {  # $1 = an arm, or untrained_p<P>_s<S>
  res exists "$PREFIX/emb/$1.npz" && return 0
  if [ "${1%%_*}" = untrained ]; then local p=${1#untrained_p}; "$PY" artifacts/patch_grid_embed.py untrained --patch "${p%_s*}" --seed "${1##*_s}" --out "$EMB" || return 1
  else
    is_void "$(arm_dir "$1")" && { say "$1: VOID, not embedded"; return 0; }
    "$PY" artifacts/patch_grid_embed.py trained --arm "$1" --ckpt "$(arm_dir "$1")/encoder.pt" --out "$EMB" || return 1
  fi
  res put "$EMB/$1.npz" "$PREFIX/emb/$1.npz"
}
: > "$STATE/q3"; for p in 4 8 16; do for s in 0 1 2; do for o in jepa moco mae untrained; do echo ${o}_p${p}_s${s} >> "$STATE/q3"; done; done; done
pool "$STATE/q3" embed || fail "embeddings: one failed twice"

# --- 9. results: SHA-1 lists over what scoring needs, then the outcome ---------------------------------------------------------
( cd runs && find patch_grid patch_grid_rerun -type f \( -name encoder.pt -o -name traces.json -o -path '*/stage1/*' -o -path '*/emb/*.npz' \) -not -path '*/smoke_*' -print0 2>/dev/null | sort -z | xargs -0 sha1sum ) > "$STATE/results.sha1"
( cd "$NVME/masks" && find s0 s1 s2 -type f -print0 | sort -z | xargs -0 sha1sum ) > "$STATE/masks.sha1"
res putdir "$S1" "$PREFIX/stage1" >/dev/null && res putdir "$STATE" "$PREFIX/state" >/dev/null || fail "final upload"
n=$(ls "$EMB"/*.npz 2>/dev/null | wc -l)
finish ok "grid done: $n embeddings; results.sha1 in $PREFIX/state"
