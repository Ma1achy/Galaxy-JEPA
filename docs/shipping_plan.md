# Shipping the cache to a rented GPU: plan (not built)

Plan only (21 Nov 2026). No bucket created, nothing uploaded, no code written. The route (user, 21 Nov): **the
user uploads from the Mac; the rented box never talks to SciServer**, so the SciServer token stays on the Mac.
Transport is rclone to Cloudflare R2. The R2 layer was designed earlier but never built.

## 1. What the grid needs

Measured by `artifacts/ship_measure.py ids` (`docs/ship_measure.json`). The cache is **one directory**,
`runs/m_v2/cache/eab81d179579f2f083f11b2301a662db781bd395e664f5ba5d71041163d5abfd/`, holding both corpora in one
stamp file. There is no separate probe cache.

| file | size |
|---|---|
| `stamps.f16` | 415.75 GB (1,057,317 stamps × 393,216 B) |
| `probe_columns.f64` | 685 MB |
| `index.json` | 26 MB |
| `petro_rad_arcsec.f64` | 8 MB |
| corpus metadata: `pretrain_v2/metadata.csv` 105 MB, `probe_v2/metadata.csv` 125 MB, two `manifest.json` | 230 MB |
| **total** | **≈ 416.7 GB** (+ about 0.84 GB of v1 split inputs; see below) |

What the grid reads, by stamp:

| set | stamps | GB |
|---|---|---|
| A1's training split (`split_pretrain`, seed 0) | 810,491 | 318.7 |
| the monitor slice (2%) | 16,477 | 6.5 |
| probe_v2, all rows | 230,349 | 90.6 |
| **union needed** | **1,057,317** | **415.75** |
| cache rows needed by nothing | 0 | 0 |

- **The minimum set is the whole cache.** The two corpora do not overlap (objID dedup), every pretraining
  stamp is in the split, and there are no stale rows.
- **Probe rows outside the ladder's union are read (checked 21 Nov), so the reduced set is not shipped by
  default.**

  | step | probe galaxies read | evidence |
  |---|---|---|
  | JEPA training (`aligned_train` → `m2_long_run`) | no stamps; probe `metadata.csv` ids for the dedup only | `aligned_train.py:57-58` forces `--no-probe` |
  | **MoCo / MAE training, in-run trajectory probe** (default; `--no-probe` skips it) | **195,677**: v1's full split, uncapped (`prepare(ckpt, 0)`) | `baseline_run.py:250-256`, `aligned_c2.py:39`; banks `c2_moco_s24_000101308` and `c2_a2_traj4_000101308` each hold 195,677 ids |
  | E4 untrained bank, E3, E7, p2, r0, s2, t1 | the 74,825 union (identical split, MAX_TRAIN 40,000 stride, minus the nine) | `p1_untrained_bank.py:52`, `o1_matched.py:156`, `r_nonlinear.py:61`, `j4_spread_controls.py:100-135` |

  - **Probe stamps needed:** the union alone is 29.4 GB; with the in-run probe it is 195,677 galaxies,
    76.9 GB; everything is 90.6 GB.
  - **Shipping the union only would also need two code changes.** Both launch gates refuse a partial cache:
    `aligned_launch.sh:65` (cache n must equal the metadata rows) and `j1_preflight.py:219` (cache n must
    equal pretrain + probe). The sidecars would also have to be rewritten for a subset index.
  - **The saving is 61 GB, 15% of the upload.**
  - **Decided (user, 24 Nov): ship the full probe corpus as planned.** No `--no-probe`, no launch-gate changes; the
    v1 cache pieces below stay in the plan.
- **Newly found box requirement: v1's cache, for the identical split.** `_identical_split` rebuilds v1's
  split from v1's cache at a hard-coded path (`j4_spread_controls.py:58`,
  `/Volumes/X10 Pro/galaxy-jepa/runs/full/cache/…`). On the box it needs:
  - v1's `index.json` (26 MB) and `probe_columns.f64` (685 MB);
  - v1's corpus `data/probe/metadata.csv` (125 MB);
  - a `stamps.f16` of the right size (415,757,500,416 B). `TensorCache` memmaps it but the split never
    reads a pixel, so a sparse file made with `truncate -s` serves. That is to be confirmed on the box
    smoke.

  The path must be adapted on the box (an environment override, not an edit to the frozen constant), and
  the manifest gets a v1 section.
- **The box does not need the FITS files** (`data/*/*.fits`): `bake_cache` skips every id already in the
  cache without opening its FITS.

## 2. Compression (zstd)

Measured by `artifacts/ship_measure.py zstd` (zstd 1.5.5, `-T0`, Mac M3 Pro; load 5–8 from other sessions).

- **Samples:** `stamps.f16` as 32 pieces of 64 MiB at random stamp-aligned offsets (2 GiB); every other
  component whole (all are under 2 GB).
- **Time:** wall-clock per raw GB, including the read.
  - The plain `stamps.f16` pass reads from the SSD.
  - The shuffled pass reused the same pieces from the page cache, so its time understates a cold read.

| component | raw | ratio, level 3 | s/GB, level 3 | ratio, level 19 | s/GB, level 19 |
|---|---|---|---|---|---|
| `stamps.f16` | 415.75 GB | **1.115** | 3.9 | **1.115** | 39 |
| `stamps.f16`, byte planes split (fp16 high bytes, then low) | — | **1.207** | 1.0 (warm) | — | — |
| `probe_columns.f64` | 685 MB | 29.5 | 1.2 | 40.1 | 10.6 |
| `index.json` | 26 MB | 5.2 | 1.6 | 10.3 | 299 |
| `petro_rad_arcsec.f64` | 8 MB | 1.31 | 5.5 | 2.12 | 221 |
| `pretrain_v2/metadata.csv` | 105 MB | 2.35 | 1.2 | 2.93 | 323 |
| `probe_v2/metadata.csv` | 125 MB | 3.47 | 1.0 | 5.31 | 182 |

**Reading:**

- **The stamps barely compress.** Normalised fp16 sky noise is close to incompressible: level 3 saves
  10%, and level 19 saves nothing more at 10× the time.
- **Splitting the fp16 bytes into planes before zstd** gives 1.21, saving 17%. It is a reversible
  reshape, checked by the raw SHA-1 after reassembly.
- **Upload volume:**
  - raw 416.7 GB;
  - level 3, 373.6 GB;
  - byte planes + level 3, **345 GB**.

  At 100 Mbit/s that is 9.3 h, 8.3 h and 7.7 h; at 40 Mbit/s, 23 h, 21 h and 19 h. Compression saves at
  most 17%. **The uplink, not the codec, sets the transfer.**
- **Mac time to pack** the full cache at byte planes + level 3: about 7–30 min. The floor is the SSD read
  (about 416 GB at about 1 GB/s ≈ 7 min); the ceiling is the cold level-3 rate (3.9 s/GB). SHA-1 over raw
  and compressed adds a few minutes of CPU.
- **Recommendation:** split the bytes into planes, then level 3, for `stamps.f16`; level 19 for the small
  files, where it is seconds.

## 3. The layer

### Credentials

- **R2 keys live only in `.env`**, beside the SciServer token, and are read only by an `artifacts/` script
  (`artifacts/ship.py`, to be written). They never enter `src/galaxy_jepa/`.
- **rclone takes its remote from environment variables** set by that script for the child process
  (`RCLONE_CONFIG_R2_TYPE=s3`, `…_PROVIDER=Cloudflare`, `…_ACCESS_KEY_ID`, `…_SECRET_ACCESS_KEY`,
  `…_ENDPOINT`). No `rclone.conf` holding a secret is written. rclone's logs are run with `--log-level INFO`
  and never `--dump headers`.
- **The credential check already covers these keys.** `tools/credential_check.py` matches every `.env`
  value of 8 characters or more, and R2's access key (32 hex) and secret (64 hex) qualify. It runs before
  every commit and push as now.
- **Two tokens, scoped per bucket:**
  - read-write on the Mac, for the upload;
  - read-only on the box, written into the box's own `.env` by hand;
  - both revoked when the rental ends.

### Buckets and layout

- **One bucket per `pipeline_hash`.** R2 bucket names allow at most 63 characters, so the name is
  `gj-cache-<first 16 hex>`, for example `gj-cache-eab81d179579f2f0`. The manifest carries the full
  64-hex hash, and the box checks it.
- **`stamps.f16` ships in chunks aligned to stamps.** One stamp is 3 × 256 × 256 fp16 = 393,216 bytes, so a
  chunk of 5,461 stamps is 2.147 GB. That gives about 194 chunks for the full file. Each chunk is
  compressed on its own (`chunk-00000.f16.zst`, …). This makes each chunk:
  - a resumable unit;
  - decompressible in parallel on the box;
  - re-fetchable alone if a check fails.

  A subset cache (section 1) is packed the same way.
- **Small files ship whole and compressed:** `index.json`, `probe_columns.f64`, `petro_rad_arcsec.f64`, and
  for both corpora `metadata.csv` and `manifest.json`. These are what `DirectorySource` and the
  run's `data_snapshot` read. FITS files are not shipped, because `bake_cache` skips every id already in
  the cache without opening its FITS.

### The manifest

`manifest.json`, written last, after every upload has been verified:

- **Identity:** `pipeline_hash`, `normalisation_hash`, `scalars_sha256` and `probe_columns_sha256` (copied
  from `index.json`); the number of stamps; the object-id list's SHA-1.
- **Per file:**
  - name, raw size, raw SHA-1, compressed size, compressed SHA-1;
  - for chunks, the first stamp row and the number of stamps;
  - the zstd level.
- **Provenance:** the code SHA of `ship.py`, the creation time, and rclone's version.

**The trust anchor.** The manifest's own SHA-1 is printed on the Mac at upload, and the user passes it to
the box on the command line. The box does not trust a manifest just because it is in the bucket.

### Upload (Mac)

`ship.py pack` → `ship.py push`:

1. Chunk and compress into a staging directory on the SSD, recording raw and compressed SHA-1 per chunk.
   Staging needs about the compressed size free on the SSD.
2. `rclone copy` with `--checksum` (resumable: files already present with matching hashes are skipped),
   `--transfers 4` and `--s3-upload-concurrency 4`. Multipart uploads resume per part on a retry.
3. Verify the bucket against the local list (`rclone check --one-way`), then upload `manifest.json`.

Resumption is per chunk. An interrupted upload reruns the same command and sends only what is missing.
Uplink time is the real cost: see the cost doc's transfer section.

### Pull and verify (box)

`ship.py pull --manifest-sha1 <sha1>`, which **refuses to train on any mismatch:**

1. Fetch `manifest.json` and check its SHA-1 against the argument; stop on a mismatch.
2. `rclone copy` the bucket to local NVMe (≥ 0.5 TB).
3. Check every compressed file's SHA-1, decompress, and check every raw SHA-1. A chunk that fails is
   re-fetched once, then the pull stops.
4. Reassemble `stamps.f16` by writing each chunk at its row offset into a preallocated file. Check its size
   against the stamp count × 393,216.
5. **Pipeline parity:**
   - `TensorCache(cache_dir)` opens the cache, and its `index.json` `pipeline_hash` must equal both the
     manifest's and the directory name;
   - the run's config, resolved on the box, must give the same `data.cache.pipeline_hash`. That is
     `config_hash` over the pixel pipeline's own config (stretch and normalisation), not over the run
     config, so `runtime.device` never enters it and CUDA does not move it (`data/cache.py:60`);
   - `load_scalars` and the probe-column loader recheck their own SHA-256 digests (existing code).
6. Write `cache_verified.json`: the manifest SHA-1, the pipeline hash and the time.
7. **Gate:** the training launcher, an artifacts-side copy of `aligned_launch.sh` with its config-hash
   `EXPECT`, refuses to start unless `cache_verified.json` exists and its manifest SHA-1 and pipeline
   hash match the expected values. A cache it would have to top up (a missing id) fails loudly.

### Back to the Mac

Checkpoints and embeddings go up to a separate prefix in the same bucket (`results/<run>/`) with their own
SHA-1 list, and the Mac pulls and verifies them the same way. This is ≤ 30 GB for the full grid.

## Open choices (not decided here)

- Byte planes + level 3 (recommended) or plain level 3 for the stamps (section 2).
- Full cache or subset (section 1). A subset is a new cache file with its own `index.json`. It has the same
  `pipeline_hash`, since the pixels and normalisation are identical, but a different id list, so the
  manifest must record which ids it holds.
