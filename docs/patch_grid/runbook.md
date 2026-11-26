# Patch grid: box requirements and rental runbook (25 Nov 2026, third version: unattended)

Nothing is booked. The pre-registration is hashed (`docs/patch_grid/prereg.sha1`, commit `bfe2871`); its 101 pinned files still match at the branch head, so the box runs a new commit that adds `artifacts/run.sh` and `artifacts/results.py` (unpinned) beside them. Hours and prices are estimates; nothing here has run on a GPU, and `run.sh` has not run end to end anywhere.

**Decided (user, 25 Nov, third version):** one unattended script, `artifacts/run.sh`. Results go to a second bucket, `gj-results`, through a third token (Object Read & Write, `gj-results` only) on the box; the box keeps read-only access to the cache bucket. This replaces the Mac pulling over SSH. The cache still goes up over R2 with `ship.py` before booking.

## 0. run.sh

```
NVME=/data MANIFEST_SHA1=<from the Mac> STAGES=smokes|arms|all [SPOT=1] [GPUS="0 1 2 3"] [WORKERS=8] [SHUTDOWN_OK=... SHUTDOWN_FAIL=...] bash artifacts/run.sh
```
Preflight → environment and pins → pull and verify the cache (`ship.py`) → parity (F0, the nine config hashes, 810,491 stamps, cuda) → stage 1 checklist: nine smokes, one at a time on one GPU, **stop on a failed row** → patch-4 masks on the CPUs (built and checked, or fetched from `gj-results`) → stage 1 arms (patch 16, 8) over the GPU pool → `judge`, **stop on FAIL** → stage 2 arms (patch 4, longest first) → 36 embeddings → SHA-1 lists → `OUTCOME-ok|fail-stages-<S>` in the bucket → shutdown.

- **Resume.** Every step asks the bucket what is done. Rerunning the same command after a preemption skips finished arms and resumes a running one from its newest checkpoint: the local disk first, else the bucket (`results.py ckpt-pull`, sha256 checked against the checkpoint manifest). While an arm trains, a sync loop pushes each newly committed checkpoint within a minute and removes the older one from the bucket.
- **Stops.** Each stop calls `SHUTDOWN_FAIL` after uploading the log: a failed smoke row, parity, an arm whose launcher fails twice, refused masks. An arm committed at step 101,308 with no `traces.json` is held, not relaunched, because the launcher would train one step past STOP. A VOID arm gets its one rerun in `runs/patch_grid_rerun/` automatically.
- **Shutdown.** On RunPod both commands default to `runpodctl pod stop $RUNPOD_POD_ID`, and preflight refuses if `runpodctl` can't see the pod. Elsewhere they must be given. **On Nebius an in-VM `shutdown` does not stop the bill** (their docs); it takes their CLI with a credential, or you stopping it. `none` leaves the box running on purpose.
- **Log.** `$NVME/run.log` goes to `gj-results/grid/logs/` every 5 min and at the end.
- **Stages.** `STAGES=smokes`: the stage 1 checklist only (pull, parity, nine smokes), judged strictly. `STAGES=arms`: checks the smoke records from the bucket, then all 27 arms with resume and the 36 embeddings. The judge's real-arm rows are recorded, not gated (user, 26 Nov). `STAGES=all`: both on one box.

### What goes up (`gj-results/grid/`)

| path | what | size |
|---|---|---|
| `emb/*.npz` | 36 embeddings, fp32 (27 arms + 9 untrained banks; VOID arms skipped) | ≈ 4.1 GB |
| `arms/patch_grid[_rerun]/<arm>/encoder.pt`, `traces.json` | 27 encoders and training traces | ≈ 2.3 GB |
| `arms/.../<arm>/checkpoints/` | **the final full checkpoint only** (step 101,308) and a one-entry manifest. While an arm runs, its newest checkpoint is held here, then replaced | 27 × 0.38–0.57 ≈ 13 GB |
| `stage1/` | smoke, flash, round-trip and real-arm records and logs (a resumed invocation's earlier log kept beside it) | < 0.1 GB |
| `masks/s{0,1,2}/` | patch-4 masks with their check | ≈ 0.1 GB |
| `state/` | `results.sha1`, `masks.sha1`, judge and validity tables, F0, pins, queues | small |
| `box/<host>-<boot>/` | `pip freeze`, `nvidia-smi -q`, the commit | small |
| `logs/`, `OUTCOME-*` | the run log, the outcome | small |

About 20 GB in all, which is cents a month on R2. The scoring needs the first row, the traces, `stage1/` and `state/`, about 6 GB. Nothing else leaves the box: no cache, no intermediate checkpoints, no smoke runs.

**On the Mac afterwards:** `RES_LOCAL=` unset, `python artifacts/results.py getdir grid/emb runs/almagest/patch_grid/box/emb` (and likewise `state`, `stage1`, `arms`), then `shasum -c` against `state/results.sha1`. Every line must be OK.

### Credentials (box `.env`, `chmod 600`, written by you)

| name | Mac | box |
|---|---|---|
| `R2_ENDPOINT` | yes | yes |
| `R2_RW_*` (cache bucket, write) | yes | **never** |
| `R2_RO_*` (cache bucket, read only) | yes | yes |
| `R2_RES_*` (gj-results only, Object Read & Write) | for the `getdir` above | yes |

**Measured 25 Nov:** the token stored as `R2_RO_*` **accepted a write**: a 1-byte object, `_rotest/probe`, is in the cache bucket. Recreate it as Object Read only before any box gets it. The manifest SHA-1 still protects what a box trains on, but a write key on the box could delete the cache.

### Spot / preemptible

Checkpoints land every 6,331 steps: 2.6–4.9 min at patch 16 and 8, and 10–20 / 8–17 / 16–31 min for JEPA / MoCo / MAE at patch 4. With `SPOT=1`, `run.sh` sets `GJ_CHECKPOINT_EVERY` (outside the config hash) for the patch-4 arms to 3,150 / 3,650 / 2,000, about 5–10 min each (user, 26 Nov). A resumed arm's throughput row would read FAIL, so the strict checklist is the nine uninterrupted smokes on demand. The real arms' rows are recorded and gate nothing (user, 26 Nov).

### Recommended: Nebius, stage 1 on demand, stage 2 on a preemptible VM, one disk carried across (research of 25 Nov; nothing booked)

**RunPod is out for spot:** its v2 API lists spot pods under "Removed", with no alternative, and its pricing pages show on-demand rates only. Nebius has every piece the unattended plan needs:
- a role that can only view, start and stop VMs (`compute.instance-power-operator`), bindable to one VM through a group permit;
- the VM's own id from the metadata service;
- stop (not delete) on preemption, with SIGTERM 60 s before;
- no compute billing while stopped.

Its gap is disk: **no local NVMe for H100.** Network SSD Non-replicated (NRD) is listed at 2 GiB/s per disk and 75k IOPS. That is a ceiling, not a measurement, so step 4 measures it before anything trains.

Unverified until steps 3–4:
- whether that role can be bound to a single VM;
- whether the CLI on the VM can act as the attached service account;
- the name of the status field `instance get` returns;
- whether a restart can fail for capacity (the watcher retries).

**Approved (user, 26 Nov), with changes:**
- On demand only for the stage 1 checklist: nine separate, uninterrupted 2,000-step smokes, judged strictly.
- **All 27 full arms** (patch 16, 8 and 4), and the embeddings, on **one** preemptible VM with resume, price capped at **$2.00/h**. The checklist judges only the smokes. The real arms' rows are recorded in `state/judge_info.txt` and gate nothing.
- The newest two checkpoints per arm stay on the disk; the final one is kept and uploaded.
- In step 3, a failed per-VM permit or a failed CLI service-account auth **stops** the plan for the user. The role is never widened.

1. `[YOU]` In the console, project in eu-north1:
   - a service account `gj-vm-power`, and a group `gj-vm-power` containing it;
   - a **non-replicated network SSD of 558 GiB** (6 × 93 GiB). It holds 417 GB of cache, 27 final checkpoints (11–15 GB), two working checkpoints per running arm, the nine smoke runs (about 5 GB), 27 encoders, 36 embeddings and the masks: about 445 GB in all. `run.sh` links `runs/` onto it. About $30 a month, so a few dollars for the days it lives.
2. `[YOU]` Box 1, **regular (on-demand)** `gpu-h100-sxm`, `1gpu-16vcpu-200gb`:
   - the disk attached, mounted at `/data`;
   - `--service-account-id` of `gj-vm-power`.

   Bind the role to this VM only: `nebius iam access-permit create --parent-id <group id> --resource-id <box 1 id> --role compute.instance-power-operator`.
   **If the per-VM permit refuses that role: stop and tell me.** No wider role is tried.
3. `[YOU]` SSH in and install the toolchain (§2: apt packages, rclone, uv, `git clone`, `git checkout <the box commit>`). Write `.env` (R2_ENDPOINT, R2_RO_*, R2_RES_*; `chmod 600`). Install the nebius CLI, then check that **with no profile or key of yours** it can act as the attached service account:
   - `nebius compute instance get --id $(curl -sH 'Metadata: true' http://metadata.nebius.internal/v1/instance-data/id)` must print this VM;
   - `nebius compute instance get --id <any other VM id in the project>` must be refused (the permit is this VM only).

   **If either fails: stop and tell me.** No key file is made in its place.
4. Measure the disk: `fio --name=r --filename=/data/fio --size=20G --rw=randread --bs=384k --iodepth=16 --ioengine=libaio --direct=1 --runtime=30 --time_based; rm /data/fio`. **Below 1 GB/s: stop and report.** The smokes' throughput rows would be measuring the disk.
5. The boot service. Write `/etc/gj.env` (`NVME=/data MANIFEST_SHA1=<from the Mac> STAGES=smokes`) and a unit:
   ```
   [Service]
   Type=simple
   User=<you>
   EnvironmentFile=/etc/gj.env
   WorkingDirectory=/home/<you>/Galaxy-JEPA
   ExecStart=/bin/bash artifacts/run.sh
   Restart=no
   [Install]
   WantedBy=multi-user.target
   ```
   Then `sudo systemctl enable --now gj`. Box 1 pulls and verifies the cache onto the disk, checks parity, and runs the nine smokes, with flash and round trip, one at a time. It writes `OUTCOME-ok-stages-smokes` or `OUTCOME-fail-stages-smokes` and stops itself through the API. No watcher. The log is in `gj-results/grid/logs/`.
6. `[YOU]` After `OUTCOME-ok-stages-smokes`: **delete box 1** (the disk stays), then create box 2:
   - **preemptible** `gpu-h100-sxm` 1 GPU, `--preemptible-on-preemption stop`;
   - a spot pricing policy capped at **$2.00/h**;
   - the same disk attached at `/data`, and the same service account.

   Move the permit to box 2's id, and repeat step 3's two checks on box 2. The cache is already verified on the disk, so there is **no second pull**. In `/etc/gj.env`, set `STAGES=arms SPOT=1`; `systemctl enable --now gj`.
7. On the Mac: `VM=<box 2 id> STAGES=arms bash artifacts/nebius_watch.sh`, detached. It polls every 5 min. A stopped VM with no outcome in the bucket was preempted, so the watcher starts it again, and the boot service reruns `run.sh`, which resumes from the newest checkpoint. Once an outcome is in the bucket, the watcher never restarts the VM.
8. Box 2 re-reads the smoke records from the bucket and refuses to train unless they pass. Then:
   - it builds and checks the patch-4 masks on its CPUs, beside the first arms;
   - it runs the 18 patch-16 and patch-8 arms (checkpoints every 6,331 steps, 2.6–4.9 min);
   - then the nine patch-4 arms (with `SPOT=1`: every 2,000 / 3,150 / 3,650 steps for MAE / JEPA / MoCo, about 5–10 min);
   - then the 36 embeddings.

   43.6–84.9 GPU-h before preemption, so 2–4 days of wall-clock, plus waits for capacity or for the price to fall under $2. On SIGTERM, run.sh pushes each running arm's newest checkpoint before the VM stops.
9. `[YOU]` After `OUTCOME-ok-stages-arms`: on the Mac, run `results.py getdir` and `shasum -c` against `state/results.sha1`. Then delete box 2 and the disk, and revoke the R2_RO_* and R2_RES_* tokens.

**Cost (26 Nov plan):**

| part | GPU-h | rate | cost |
|---|---|---|---|
| box 1, on demand: boot, environment, pull, parity, nine smokes | 2.3–3.5 | $4.50 | $10–16 |
| box 2, preemptible: 27 arms + 36 embeddings + restarts, with 5–25% preemption overhead | 46–106 | $0.79 floor | $36–84 |
| the same | 46–106 | $2.00 cap | $92–212 |
| disk, 558 GiB NRD, 4–6 days | | $0.053/GiB-month | $4–6 |
| **total** | | | **$50–106 at $0.79; $106–234 if spot sits at the cap** |

The spot price moves between the $0.79 floor and the $2.00 cap, so the real figure lies between those two totals. R2 storage for the results is cents.

The by-hand steps below are what `run.sh` automates. They are kept for reading and for a manual fallback. Where they say results come home over SSH, `gj-results` replaces that.

## 1. Box requirements

| | box 1: stage 1 | box 2: stage 2 (4 arms at once) | why |
|---|---|---|---|
| GPU | 1 × H100 80 GB (SXM preferred) | **4 × H100 80 GB**, the same H100 variant as box 1 if you can | MAE at patch 4 needs about 32 GB ± 30% at batch 32: a 40-GB card is too close. One arm per GPU, independent runs, no NVLink needed |
| NVIDIA driver | ≥ 580 | ≥ 580 | `uv.lock` pins torch 2.12.0 with CUDA 13.0 |
| CPU | **32 cores (64 vCPU)**; 16 cores work | **48 cores (96 vCPU)** minimum, 64 recommended | Box 1: 8 loader workers, 1 trainer, and the masks: about 50 core-hours for three seeds (Mac measurement), 2–3 h on 20 cores. Box 2: 4 × (8 workers + 1 trainer) = 36 busy processes; no masks to build |
| RAM | 128 GB | **256 GB** minimum, 480 GB recommended | per arm: 8 workers and a trainer, about 16 GB; four arms 64 GB. The rest is page cache for random reads of a 417-GB memory-mapped file, which doesn't fit and needn't |
| NVMe, local | 1 TB | 1 TB (1.5 TB recommended) | cache 417 GB; pull staging under 20 GB (each compressed chunk goes as soon as it verifies); checkpoints 17 per arm × 0.38–0.57 GB: box 1 about 140 GB for 18 arms, box 2 about 70 GB for 9 |
| NVMe speed, random 384-KB reads | ≥ 1 GB/s | ≥ 1.5 GB/s | a patch-8/16 arm at the step floor reads 500 MB/s; a patch-4 arm up to 170 MB/s, four of them under 0.7 GB/s. Network storage can't be relied on for either |
| Network | ≥ 1 Gbit/s down | **≥ 5 Gbit/s down** | the pull is about 345 GB: 46 min at 1 Gbit/s, 9 min at 5. Box 2 bills four GPUs while it pulls |
| OS | Linux x86-64, glibc ≥ 2.28, Ubuntu 22.04/24.04 | same | the locked wheels are manylinux_2_28 |
| Access | SSH, root or Docker, on-demand | same | a preempted box loses its NVMe, though arms resume from checkpoints (every 6,331 steps) |

If one provider lets the same volume move from a 1-GPU to a 4-GPU instance, use that: box 2 then skips the second pull.

**GPU-hours** (estimates; billed hours count all four GPUs of box 2):

| | wall-clock | GPU-hours billed | cost at $2.00–3.50/h |
|---|---|---|---|
| box 1: boot, environment, pull, parity, nine smokes | 2.3–3.5 h | 2.3–3.5 | $5–12 |
| box 1: stage 1, 18 arms | 13–22 h | 13–22 | $25–76 |
| box 1: masks (CPU, beside stage 1), results off | 0.5 h | 0.5 | $1–2 |
| box 2: boot, environment, pull, parity | 0.5–1.5 h | 2–6 | $4–21 |
| box 2: stage 2, 9 arms on 4 GPUs | 6.9–13.8 h | 28–55 | $55–193 |
| box 2: 36 embeddings, fp32, on 4 GPUs | 1–2 h | 3–8 | $6–28 |
| box 2: last results off | 0.5 h | 2 | $4–7 |
| **total** | **25–44 h, plus the gap between boxes** | **51–97** | **$102–340** |

This is above the cost doc's 45–87 because of the second set-up on four billed GPUs and the idle tail. Set a spending alarm at your own ceiling before booting.

## 2. Credentials and who does what

`[YOU]` marks your steps. No token, password or key is typed by Claude, pasted into chat, or put in a command line. `.env` holds, by these names (the bucket's name is not in `.env`):

| name | Mac `.env` | box `.env` |
|---|---|---|
| `R2_ENDPOINT` (`https://<account id>.r2.cloudflarestorage.com`) | yes | yes |
| `R2_RW_ACCESS_KEY_ID`, `R2_RW_SECRET_ACCESS_KEY` | yes | **never** |
| `R2_RO_ACCESS_KEY_ID`, `R2_RO_SECRET_ACCESS_KEY` | yes (for the one-chunk pull test) | yes, until its pull is done |

**Standing rule.** If a pinned file has to change for anything to run, **stop**: it is an amendment to the pre-registration, and yours to make. Nothing is edited on a box.

`ARM` is `<jepa|moco|mae>_p<16|8|4>_s<0|1|2>`. Run in `tmux`, from `~/Galaxy-JEPA`. `$NVME` is the local disk's mount point.

## 3. Runbook

### 0. Before booking (Mac)
1. `[YOU]` Create the bucket `gj-cache-eab81d179579f2f0` and two bucket-scoped tokens (Object Read & Write; Object Read only), into the Mac's `.env`.
2. One chunk: `python artifacts/ship.py push --only 0 --no-cap`, then `ship.py testpull --only 0`. Report the rate. **Wait for the user's go.**
3. `python artifacts/_detach.py runs/almagest/ship/push.log --name "R2 upload" -- .venv/bin/python artifacts/ship.py push` (6 Mbit/s 08:00–23:00, uncapped overnight; stop and rerun at will). Note the printed **manifest SHA-1**.
4. `python artifacts/patch_grid_pins.py` in a checkout of the box commit says `101 files match`.

### 1. Boot box 1 (clock starts)
1. `[YOU]` Book and boot; add your SSH key. Note the hourly rate, the instance id and the start time.
2. Check it against section 1 and release it if it fails:
   ```
   nvidia-smi; nproc; lscpu | grep -E 'Core|Socket'; free -g; df -h; lsblk -d -o NAME,ROTA,SIZE,MODEL
   fio --name=r --filename=$NVME/fio --size=20G --rw=randread --bs=384k --iodepth=16 --ioengine=libaio --direct=1 --runtime=30 --time_based; rm $NVME/fio
   ```

### 2. Environment (both boxes)
```
sudo apt-get update && sudo apt-get install -y git tmux zstd fio rsync unzip
curl -fsSLO https://downloads.rclone.org/v1.76.0/rclone-v1.76.0-linux-amd64.zip && curl -fsSL https://downloads.rclone.org/v1.76.0/SHA256SUMS | grep rclone-v1.76.0-linux-amd64.zip | sha256sum -c   # Ubuntu's own rclone is too old
unzip -j rclone-v1.76.0-linux-amd64.zip '*/rclone' && sudo install rclone /usr/local/bin/
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
git clone https://github.com/Ma1achy/Galaxy-JEPA.git && cd Galaxy-JEPA && git checkout b1ab4c2
uv sync --frozen --extra data --extra eval && source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.device_count(), torch.cuda.is_bf16_supported())"
python artifacts/patch_grid_pins.py        # must say: 101 files match
```
The repository is public: no GitHub credential goes on a box.

### 3. R2 pull and parity (both boxes)
1. `[YOU]` Write `R2_ENDPOINT` and the **read-only** pair into `~/Galaxy-JEPA/.env` by hand; `chmod 600 .env`.
2. `python artifacts/ship.py pull --manifest-sha1 <the SHA-1 from step 0> --dest $NVME/cache --workers 8`
   It checks the manifest against that SHA-1, every compressed and raw SHA-1, and that the chunks tile the stamp file; writes `data/*/metadata.csv` (the split needs them); opens the cache under the pipeline hash this code resolves; and only then writes `cache_verified.json`. A chunk that fails twice stops the pull. Rerun to resume.
3. In `~/pg.env`, sourced in every shell from here:
   ```
   export GJ_CONFIG=configs/pretrain_v2.yaml F0_NORM_PREFIX=246d8de6ba9a F0_CACHE_BASE=$NVME/cache
   export GJ_CACHE_MANIFEST_SHA1=<the SHA-1 from step 0>
   ```
   `$F0_CACHE_BASE` is the directory *above* the pipeline-hash directory.
4. Parity, independent of the pull:
   ```
   python artifacts/f0_preconditions.py                         # F0 PASS; n = 1,057,317
   for o in jepa moco mae; do for p in 16 8 4; do python artifacts/patch_grid_run.py ${o}_p${p}_s0 --plan $([ $o$p = jepa4 ] && echo --masks none) | head -3; done; done
   ```
   Expect `train stamps 810,491` and device `cuda` nine times, and these config hashes (computed on the Mac for `device: cuda`; the same for every seed). A different hash is a different experiment: stop.

   | arm | config hash | arm | config hash | arm | config hash |
   |---|---|---|---|---|---|
   | jepa_p16 | `763cba6a9ce2ca0a` | moco_p16 | `67ef97b0d148d37e` | mae_p16 | `dad0a614a1c5c17a` |
   | jepa_p8 | `7b3a08b6eedeb158` | moco_p8 | `92f90272ff77f702` | mae_p8 | `b4d7e90e2cd42a9a` |
   | jepa_p4 | `f5a5e85e35a7def2` | moco_p4 | `5d8d494750a41020` | mae_p4 | `9ba716d915c82856` |
5. `[YOU]` `rm ~/Galaxy-JEPA/.env` on the box. Nothing later needs it. Revoke the read-only token after **box 2's** pull.

### 4. Smokes: the checklist on nine arm types (box 1)
One at a time, on an idle GPU (the first 200 steps of each are §7's timed smoke).
```
python artifacts/patch_grid_masks.py build --seed 0 --grid 64 --steps 2000 --workers 16 --out $NVME/masks/smoke_s0      # a minute or two
for o in jepa moco mae; do for p in 16 8 4; do A=${o}_p${p}_s0
  python artifacts/patch_grid_stage1.py flash $A
  python artifacts/patch_grid_stage1.py watch $A --smoke 2000 $([ $o$p = jepa4 ] && echo --masks $NVME/masks/smoke_s0)
  python artifacts/patch_grid_stage1.py roundtrip $A --run runs/patch_grid/smoke_$A
done; done
python artifacts/patch_grid_stage1.py judge      # the nine smoke rows must be all ok; the real-arm rows say "no record" until step 6
```
Bars: throughput ≥ 70% of the cost doc's lower estimate; peak memory ≤ 90% of the card; loss; flash; round trip. Utilisation is printed and gates nothing. **A failed row: stop, report, wait.**

### 5. Patch-4 masks, all three seeds (box 1's CPUs, beside stage 1)
```
for s in 0 1 2; do
  nice -n 10 python artifacts/patch_grid_masks.py build --seed $s --grid 64 --workers $(( $(nproc)/2 - 10 )) --out $NVME/masks/s$s
  python artifacts/patch_grid_masks.py check --seed $s --grid 64 --out $NVME/masks/s$s || echo "SEED $s REFUSED"
done
```
Each `check` must print `0 batches differ`; any difference refuses `jepa_p4` for that seed: stop. The workers formula leaves 10 cores to the running arm. **Stage 2 is not booked until all three pass.**

### 6. Stage 1: patch 16 and 8, 18 arms (box 1)
```
for s in 0 1 2; do for p in 16 8; do for o in jepa moco mae; do
  python artifacts/patch_grid_stage1.py watch ${o}_p${p}_s${s} --workers 8 || break 3
done; done; done
python artifacts/patch_grid_auc.py valid --runs runs/patch_grid     # VOID / incomplete per arm (ignore smoke_* rows); training traces only
python artifacts/patch_grid_stage1.py judge                         # must say PASS
```
- **An interrupted arm** restarts with the same command and resumes. Its throughput row then reads FAIL, because the rate covers only the last part: report it as such.
- **A finished or halted arm** refuses to start again.
- **A VOID arm** (halted by the collapse bar, or diverged) is rerun once, same seed and recipe, in a fresh directory, and the rerun is the arm: `python artifacts/patch_grid_run.py ARM --workers 8 --out runs/patch_grid_rerun/ARM`. VOID again: it stays VOID, and is named in `final --void`.
- `judge` FAIL → stop before stage 2 and report. Nothing here reads a probe, an AUC or a score.

### 7. Box 1's results home, then tear it down
```
cd runs && find patch_grid patch_grid_rerun -type f \( -name encoder.pt -o -name traces.json -o -path '*/stage1/*' \) -not -path '*/smoke_*' -print0 2>/dev/null | sort -z | xargs -0 sha1sum > box1.sha1
cp -r $NVME/masks/s? . && find s? -type f -print0 | sort -z | xargs -0 sha1sum >> box1.sha1
```
On the Mac, into `runs/almagest/patch_grid/box1/`: `rsync -a --partial --files-from=<the paths in box1.sha1> box1:Galaxy-JEPA/runs/ .` plus `box1.sha1`; then `shasum -c box1.sha1`, **every line OK**. 18 encoders (1.5 GB), traces, the checklist and the masks (0.1 GB): 10–15 min at 18–32 Mbit/s.
Also `pip freeze` and `nvidia-smi -q`. Then step 11 for box 1.

### 8. Boot box 2, steps 1–3 again
Same commit, pins, pull, parity. Then, from the Mac: the masks, and stage 1's 18 `encoder.pt` (1.5 GB, about 20 min at 10 Mbit/s up; **send them during the pull**), each checked with `sha1sum -c` on arrival. All 36 embeddings are made on box 2, so every arm and bank shares one path (§3).

### 9. Stage 2: patch 4, one arm per GPU (box 2)
Assigned by the cost doc's hours so the four GPUs finish together (6.9–13.8 h): GPUs 0–2 run one MAE then one JEPA; GPU 3 runs the three MoCo arms.
```
for s in 0 1 2; do tmux new -d -s g$s "source ~/pg.env; export CUDA_VISIBLE_DEVICES=$s; python artifacts/patch_grid_stage1.py watch mae_p4_s$s --workers 8; python artifacts/patch_grid_stage1.py watch jepa_p4_s$s --workers 8 --masks $NVME/masks/s$s"; done
tmux new -d -s g3 "source ~/pg.env; export CUDA_VISIBLE_DEVICES=3; for s in 0 1 2; do python artifacts/patch_grid_stage1.py watch moco_p4_s\$s --workers 8; done"
```
(the tmux shells need `cd ~/Galaxy-JEPA` and the venv active). After an hour, read the first steps per second and utilisation of all four: if four arms together run below 70% of the single-arm smoke, the disk or the CPUs are the limit; report it. It is not a stop.
Then `python artifacts/patch_grid_auc.py valid --runs runs/patch_grid`. VOID and interrupted arms as step 6. Stage 2 runs whatever stage 1's losses look like, once the checklist passes.

### 10. Embeddings (fp32, the ladder's recipe): 27 + 9 files, all on box 2, four at a time
```
emb(){ A=$1; D=runs/patch_grid/$A; [ -d runs/patch_grid_rerun/$A ] && D=runs/patch_grid_rerun/$A; python artifacts/patch_grid_embed.py trained --arm $A --ckpt $D/encoder.pt --out runs/patch_grid/emb; }
export -f emb
for g in 0 1 2 3; do ( export CUDA_VISIBLE_DEVICES=$g; i=0
  for p in 4 8 16; do for s in 0 1 2; do for o in jepa moco mae untrained; do i=$((i+1)); [ $((i % 4)) = $g ] || continue
    if [ $o = untrained ]; then python artifacts/patch_grid_embed.py untrained --patch $p --seed $s --out runs/patch_grid/emb; else emb ${o}_p${p}_s${s}; fi
  done; done; done ) & done; wait
ls runs/patch_grid/emb/*.npz | wc -l      # 36, less the VOID arms
```
Each finished file can go home at once, so the last pull is short. Then, as step 7, `box2.sha1` over `emb/*.npz`, stage 2's `encoder.pt`, `traces.json` and `stage1/`: about 5 GB, checked with `shasum -c` on the Mac. **Every line OK before step 11.**
Compare `untrained_p16_s1.npz` with the Mac's banked untrained seed 1 (same weights, another device) and report the largest difference. It decides nothing.

### 11. Tear-down (each box)
1. `shasum -c` passed on the Mac, and (box 2) the 36 embeddings open there.
2. On the box: `rm -f ~/Galaxy-JEPA/.env ~/pg.env; history -c`.
3. `[YOU]` Terminate (not stop) the instance and **delete its volume**. Check the console shows no instance, volume or reserved IP, and note the bill. After box 2's pull: revoke the read-only token; after the upload: the read-write one.
4. `[YOU]` Decide whether the R2 bucket stays (about $6.30 a month).

### 12. Scoring (Mac)
From a worktree at the hash commit, because later edits on the branch change pinned files: `git worktree add ../Galaxy-JEPA-patchgrid bfe2871`, then link `runs/` and `data/` into it.
1. §8's secondaries (ordinal τ, CI knee, relabel2) are written and pinned by SHA-1 **before** this opens.
2. `patch_grid_auc.py auc` per embedding file (36) → one records directory; `auc` prints no score.
3. `patch_grid_auc.py manifest --dir DIR`, once. Commit `outputs.sha1`.
4. `patch_grid_auc.py final --dir DIR [--void NAME ...] --out <outside DIR>`. This opens every reading. Not before all 27 arms and nine banks are in.
