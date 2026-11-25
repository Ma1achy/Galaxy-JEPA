# Patch grid: box requirements and rental runbook (25 Nov 2026)

Nothing is booked. The pre-registration is hashed (`docs/patch_grid/prereg.sha1`, commit `bfe2871`). The box runs commit `9af2e7a`: the same pinned files, plus a fix to the unpinned stage 1 checker. Hours and prices are `docs/patch_ablation_cost.md`'s estimates, not measurements, and the prices are unconfirmed.
Nothing in this runbook has run on a GPU. The shipping list was dry-run on the Mac in a bare clone of `bfe2871` (pins, launch plan, embedding smoke).

**Three things must exist before booking:**
1. **`artifacts/ship.py`** (pack, push, pull). Not written; it waits on your upload speed. Steps 0 and 4 are written against its plan in `docs/shipping_plan.md`.
2. **The upload finished,** and its manifest SHA-1 in your hands. Renting first would bill the upload's hours.
3. **A way back for the results** (decision A below).

## 1. Box requirements

| | required | recommended | why |
|---|---|---|---|
| GPU | 1 × H100 80 GB | H100 SXM; 2–4 on one box cuts the wall-clock at the same cost | MAE at patch 4 needs about 32 GB ± 30% at batch 32, so a 40-GB card is too close. H100 is 2.3–2.5× an A100 at patch 4. The arms are independent runs, one per GPU |
| NVIDIA driver | ≥ 580 | | `uv.lock` pins torch 2.12.0 with CUDA 13.0 |
| CPU | 16 cores (32 vCPU) | 32 cores (64 vCPU) for 1–2 GPUs; +10 cores per further GPU | JEPA's patch-4 masks: about 17 core-hours per seed on the Mac, 50 for three, an hour a seed on 16 cores. Loaders: 8 workers per running arm, beside the masks |
| NVMe, local | **1.5 TB**, ≥ 3 GB/s reads, ≥ 2 GB/s sustained at random 400-KB reads | 2 TB | Cache 417 GB; the compressed copy, 345–374 GB, during the pull; about 200 GB of checkpoints (17 per arm, 0.38–0.57 GB each, 27 arms). 1 TB works only if each compressed chunk is deleted once its raw SHA-1 passes. One floor-bound arm reads about 500 MB/s at random. Network storage can't serve that |
| RAM | 128 GB | 256 GB | 8 loader workers per arm, the mask pool, and page cache for random reads (the cache is memory-mapped; it doesn't fit and needn't) |
| Network | 1 Gbit/s down | 10 Gbit/s | The pull is 345–417 GB: about 50 min at 1 Gbit/s |
| OS | Linux x86-64, glibc ≥ 2.28 | Ubuntu 22.04/24.04 | the locked wheels are manylinux_2_28 |
| Access | SSH, root or Docker; **on-demand, not spot** | | A preempted box resumes from checkpoints (every 6,331 steps), but loses its NVMe |

Budget: the full grid is 45–87 H100-hours ($90–305 at the assumed $2.00–3.50/h), 47–90 with the renderer. Set a provider spending alarm at your own ceiling before booting.

## 2. Decisions to make first (yours)

- **A. How results return.** The shipping plan gives the box a read-only R2 token *and* uploads results to the same bucket. Both can't hold. **Recommended: the Mac pulls them over SSH (`rsync`), so the box holds no write token at all.** The other way is a second bucket with its own write token.
- **B. Upload staging.** The SSD has 263 GB free; the packed cache is 345–374 GB. So `ship.py` has to pack, push, verify and drop one chunk at a time, not stage the whole. I'll write it that way unless you object.
- **C. GPU count.** One H100 is 2–4 days of wall-clock; two halve that.
- **D. The checklist's literal bars** (`flags.md` G) may stop a sound run, above all utilisation > 85% at patch 16 and 8. As approved, they stop it.

## 3. Runbook

`ARM` is `<jepa|moco|mae>_p<16|8|4>_s<0|1|2>`. Run everything in `tmux`, from `~/Galaxy-JEPA`. `[YOU]` marks your steps: no token, password or key is ever typed by Claude or pasted into chat.

**Standing rule.** If a pinned file has to change for anything to run, **stop**. It is an amendment to the pre-registration, and yours to make. Nothing is edited on the box.

### 0. Before booking (Mac)
1. Upload speed measured → `ship.py` written and tested → `[YOU]` create the bucket and two tokens: read-write (Mac `.env`) and read-only (for the box).
2. `ship.py pack`/`push`; note the printed **manifest SHA-1**.
3. Clean state: `git status` is clean, `python artifacts/patch_grid_pins.py` in a checkout of `9af2e7a` says `101 files match`.

### 1. Boot (clock starts)
1. `[YOU]` Book and boot; add your SSH key. Note the hourly rate, the instance id and the start time.
2. Check the box against section 1, and release it if it fails:
   ```
   nvidia-smi; nproc; lscpu | grep -E 'Core|Socket'; free -g; df -h; lsblk -d -o NAME,ROTA,SIZE,MODEL
   fio --name=r --filename=$NVME/fio --size=20G --rw=randread --bs=384k --iodepth=16 --ioengine=libaio --direct=1 --runtime=30 --time_based; rm $NVME/fio
   ```
   `$NVME` is the local disk's mount point.

### 2. Environment
```
sudo apt-get install -y git tmux zstd fio rsync rclone
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/Ma1achy/Galaxy-JEPA.git && cd Galaxy-JEPA && git checkout 9af2e7a
uv sync --frozen --extra data --extra eval
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```
The repository is public, so no GitHub credential goes on the box. Expect 2.12.0 and bf16 True.

### 3. Pins
```
python artifacts/patch_grid_pins.py        # must say: 101 files match
```

### 4. R2 pull and parity
1. `[YOU]` Write the read-only R2 keys into `~/Galaxy-JEPA/.env` by hand (`chmod 600 .env`).
2. `python artifacts/ship.py pull --manifest-sha1 <the SHA-1 from step 0> --dest $NVME/cache` **[WAITS on ship.py]**. It checks the manifest against that SHA-1, every compressed and raw SHA-1, the row count, and the pipeline hash, then writes `cache_verified.json`.
   `ship.py` is not in `9af2e7a`. It is unpinned; copy it in by `scp` and check its SHA-1 against the Mac's.
3. The corpus metadata: `data/pretrain_v2/metadata.csv` (105 MB) and `data/probe_v2/metadata.csv` (125 MB), in those directories, from the same pull. The split needs them. No FITS and no v1 cache: the identical-split probes run on the Mac.
4. Environment, in every shell from here (put it in `~/pg.env`):
   ```
   export GJ_CONFIG=configs/pretrain_v2.yaml F0_NORM_PREFIX=246d8de6ba9a F0_CACHE_BASE=$NVME/cache
   export GJ_CACHE_MANIFEST_SHA1=<the SHA-1 from step 0>
   ```
   `$F0_CACHE_BASE` is the directory *above* `eab81d179579f2f083f11b2301a662db781bd395e664f5ba5d71041163d5abfd/`.
5. Parity, independent of the pull:
   ```
   python artifacts/f0_preconditions.py                         # F0 PASS; n = 1,057,317
   cat $F0_CACHE_BASE/eab8*/cache_verified.json
   for o in jepa moco mae; do for p in 16 8 4; do python artifacts/patch_grid_run.py ${o}_p${p}_s0 --plan $([ $o$p = jepa4 ] && echo --masks none) | head -3; done; done
   ```
   Expect `train stamps 810,491` and device `cuda` nine times, and these config hashes (computed on the Mac for `device: cuda`; the same for every seed). A different hash means a different experiment: stop.

   | arm | config hash | arm | config hash | arm | config hash |
   |---|---|---|---|---|---|
   | jepa_p16 | `763cba6a9ce2ca0a` | moco_p16 | `67ef97b0d148d37e` | mae_p16 | `dad0a614a1c5c17a` |
   | jepa_p8 | `7b3a08b6eedeb158` | moco_p8 | `92f90272ff77f702` | mae_p8 | `b4d7e90e2cd42a9a` |
   | jepa_p4 | `f5a5e85e35a7def2` | moco_p4 | `5d8d494750a41020` | mae_p4 | `9ba716d915c82856` |
6. `[YOU]` Delete the keys from the box's `.env` and revoke the read-only token. Nothing later needs it.

### 5. Smokes: the checklist on nine arm types
One at a time, on an idle GPU, nothing else running (the first 200 steps of each are §7's timed smoke).
```
python artifacts/patch_grid_masks.py build --seed 0 --grid 64 --steps 2000 --workers 16 --out $NVME/masks/smoke_s0      # a minute or two
for o in jepa moco mae; do for p in 16 8 4; do A=${o}_p${p}_s0
  python artifacts/patch_grid_stage1.py flash $A
  python artifacts/patch_grid_stage1.py watch $A --smoke 2000 $([ $o$p = jepa4 ] && echo --masks $NVME/masks/smoke_s0)
  python artifacts/patch_grid_stage1.py roundtrip $A --run runs/patch_grid/smoke_$A
done; done
python artifacts/patch_grid_stage1.py judge      # the nine smoke rows must be all ok; the real-arm rows say "no record" until step 7
```
**A failed smoke row: stop, report, and wait for the user.** Idle cost is $2–3.50 an hour, so a decision that will take long means tear-down (step 11) and a fresh pull later.

### 6. Masks for JEPA patch 4 (CPU, beside stage 1)
```
for s in 0 1 2; do
  nice -n 10 python artifacts/patch_grid_masks.py build --seed $s --grid 64 --workers $(( $(nproc)/2 - 10 )) --out $NVME/masks/s$s
  python artifacts/patch_grid_masks.py check --seed $s --grid 64 --out $NVME/masks/s$s || echo "SEED $s REFUSED"
done
```
Each `check` must print `0 batches differ`. Any difference refuses `jepa_p4` for that seed: stop. The workers formula leaves 10 cores to a running arm; on 32 vCPU that is 6 workers, about 3 h a seed if the cores match the Mac's. If a stage-1 arm's utilisation drops while masks build, pause the build (`kill -STOP`), don't move the bar.

### 7. Stage 1: patch 16 and 8, 18 arms
```
for s in 0 1 2; do for p in 16 8; do for o in jepa moco mae; do
  python artifacts/patch_grid_stage1.py watch ${o}_p${p}_s${s} --workers 8 || break 3
done; done; done
```
- One arm per GPU (`CUDA_VISIBLE_DEVICES=k`), because sharing a GPU confounds the checklist's rates. Expect 13–22 GPU-hours.
- **An interrupted arm** is restarted with the same command and resumes from its last checkpoint. Its checklist throughput then reads FAIL, because the rate covers only the last part: report it as such.
- **A finished or halted arm** refuses to start again.
- Nothing here reads a probe, an AUC or a score.

Then:
```
python artifacts/patch_grid_auc.py valid --runs runs/patch_grid     # VOID / incomplete per arm (ignore smoke_* rows)
python artifacts/patch_grid_stage1.py judge                         # must say PASS
```
**FAIL → stop before stage 2 and report.**
**A VOID arm** (halted by the collapse bar, or diverged) is rerun once, same seed and recipe, in a fresh directory, and the rerun is the arm:
`python artifacts/patch_grid_run.py ARM --workers 8 --out runs/patch_grid_rerun/ARM`. VOID again: it stays VOID, and is named in `final --void`.

### 8. Stage 2: patch 4, 9 arms
```
for s in 0 1 2; do for o in jepa moco mae; do
  python artifacts/patch_grid_stage1.py watch ${o}_p4_s${s} --workers 8 $([ $o = jepa ] && echo --masks $NVME/masks/s$s) || break 2
done; done
python artifacts/patch_grid_auc.py valid --runs runs/patch_grid
```
Expect 27–55 GPU-hours. Same rules for interruptions and VOID arms. Stage 2 runs whatever stage 1's losses look like, once the checklist passes.

### 9. Embeddings (fp32, the ladder's recipe): 27 + 9 files, 3–8 GPU-hours
```
for o in jepa moco mae; do for p in 16 8 4; do for s in 0 1 2; do A=${o}_p${p}_s${s}; D=runs/patch_grid/$A; [ -d runs/patch_grid_rerun/$A ] && D=runs/patch_grid_rerun/$A
  python artifacts/patch_grid_embed.py trained --arm $A --ckpt $D/encoder.pt --out runs/patch_grid/emb; done; done; done
for p in 16 8 4; do for s in 0 1 2; do python artifacts/patch_grid_embed.py untrained --patch $p --seed $s --out runs/patch_grid/emb; done; done
ls runs/patch_grid/emb/*.npz | wc -l      # 36
```
A VOID arm is not embedded: 36 less the VOID arms.

### 10. Results home
```
cd runs && find patch_grid patch_grid_rerun -type f \( -name '*.npz' -o -name encoder.pt -o -name traces.json -o -path '*stage1*' -o -name 'ckpt_000025324.pt' -o -name 'ckpt_000050648.pt' -o -name 'ckpt_000075972.pt' -o -name 'ckpt_000101308.pt' \) -print0 2>/dev/null | sort -z | xargs -0 sha1sum > results.sha1
```
1. `[YOU]` or Claude on the Mac (decision A): `rsync -a --partial --files-from=<the paths in results.sha1> box:Galaxy-JEPA/runs/ "/Volumes/X10 Pro/galaxy-jepa/runs/almagest/patch_grid/box/"`. About 4 GB of embeddings, 2 GB of encoders, and 49 GB if all four checkpoints per arm come (2 + 4 GB without them).
2. On the Mac, in that directory: `shasum -c results.sha1`. **Every line must say OK before step 11.**
3. Copy `~/pg.env` without the key lines, `pip freeze`, `nvidia-smi -q` and the `masks/*/meta.json` files too, into the same directory.
4. Compare `untrained_p16_s1.npz` with the Mac's banked untrained seed 1: the same weights through the same recipe on another device. Report the largest difference. It decides nothing.

### 11. Tear-down
1. `shasum -c` passed on the Mac, and the 36 embeddings open there.
2. On the box: `rm -f ~/Galaxy-JEPA/.env ~/pg.env`; `history -c`.
3. `[YOU]` Revoke every R2 token. Terminate (not stop) the instance, and **delete its volume**. Check the provider's console shows no running instance, no volume and no reserved IP, and note the final bill.
4. `[YOU]` Decide whether the R2 bucket stays (about $6.30 a month).

### 12. Scoring (Mac)
From a worktree at the hash commit, because later edits on the branch change pinned files:
```
git worktree add ../Galaxy-JEPA-patchgrid bfe2871      # then link runs/ and data/ into it
```
1. §8's secondaries (ordinal τ, CI knee, relabel2) are written and pinned by SHA-1 **before** this opens.
2. `patch_grid_auc.py auc` per embedding file (36) → one records directory; `auc` prints no score.
3. `patch_grid_auc.py manifest --dir DIR`, once. Commit `outputs.sha1`.
4. `patch_grid_auc.py final --dir DIR [--void NAME ...] --out <outside DIR>`. This opens every reading. Not before all 27 arms and nine banks are in.

## 4. Where the time goes (one H100)

| step | GPU-hours | note |
|---|---|---|
| 1–5 boot, environment, pull, parity, smokes | 2–3 in the cost doc | the nine 2,000-step smokes add about 0.3–0.5 h to it (sum of the hours per seed × 2%) |
| 7 stage 1 | 13–22 | 18 arms |
| 8 stage 2 | 27–55 | 9 arms |
| 9 embeddings | 3–8 | fp32 |
| 10–11 results, tear-down | under 1 | not in the cost doc's total: about 50 GB over your downlink; the box idles at the hourly rate while it copies |
