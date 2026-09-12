# F3 — the identical benchmark on the M5 Max

**Not run.** This session has no access to that machine; the peers visible to it are other
projects, offline Remote Control sessions, and cloud sessions with neither the SSD nor MPS. So
the M3/M5 comparison the brief asks for is prepared, not delivered — the numbers below are the
M3 Pro's alone.

The comparison *will* be meaningful when it is run: the M3 came out **compute-bound, 83.8%
compute to 16.2% data-wait**, drawing 41.5 stamps/s against a 479 stamps/s loader ceiling. Both
machines read the same external SSD, so if the M5 also lands well under that ceiling the two are
comparing compute. If the M5 gets fast enough to approach ~400 stamps/s the comparison becomes
drive-bound and void, and the note in F1 about `num_workers` and moving the cache off USB starts
to matter.

## Run it

```sh
git clone <this repo> && cd Galaxy-JEPA
git checkout <the branch carrying Brief F>
uv sync --extra dev --extra data --extra eval        # D3: report anything that fails to resolve,
                                                    #     especially torch/MPS
# the corpora are symlinks into the SSD; recreate them for the mount point on that machine
mkdir -p data
ln -s "/Volumes/X10 Pro/galaxy-jepa/raw/pretrain" data/pretrain
ln -s "/Volumes/X10 Pro/galaxy-jepa/raw/probe"    data/probe

# F0 first — it is a gate, and both later scripts import it
uv run python artifacts/f0_preconditions.py

# F1-A device layer (~3 min + a 9.4 GB stage to the internal SSD as the ceiling reference)
uv run python artifacts/f1_loader_bench.py device

# F1-B pipeline layer. Single-process points only; see the TODO note on why workers are refused.
for v in production lean; do
  for a in shuffled sequential; do
    uv run python artifacts/f1_loader_bench.py point --variant $v --access $a --workers 0 \
      --seconds 130 | grep '^RESULT '
  done
done

# F2 — three phases, each its own process. Run under the watchdog on any machine with <32 GB.
uv run python artifacts/f2_model_smoke.py placement
uv run python artifacts/f2_model_smoke.py throughput
uv run python artifacts/f2_model_smoke.py batch
```

`f2_model_smoke.py all` drives the three phases itself and writes
`artifacts/out/f2_<cpu>.json` plus a stamped `smoke_<cpu>/` directory. Every output file is
tagged with `machdep.cpu.brand_string`, so the two machines' JSONs sit side by side without
overwriting each other.

## Safety rail — use it

`artifacts/f_watchdog.sh <child-pid> <swap-growth-MB>` kills the child if the swap file grows
past the limit. It exists because the first version of this benchmark **OOM'd the M3**: it held
the 4.07 GB production dataset resident, built a second full table beside it, and then sized both
with `pickle.dumps`, which materialises the whole serialisation in RAM. Growth is the metric, not
absolute use — this machine idles with >10 GB of swap already allocated.

```sh
uv run python artifacts/f2_model_smoke.py throughput > out.json 2> out.log &
child=$!; ./artifacts/f_watchdog.sh $child 6000 & wd=$!
wait $child; kill $wd
```

## What to compare

| quantity | M3 Pro (measured) |
|---|---|
| device, external, shuffled | 1,383 stamps/s · 544 MB/s |
| device, internal, shuffled | 3,837 stamps/s · 1,509 MB/s |
| loader, batch 32, `w=0`, shuffled | 479 stamps/s |
| model, batch 32, real data | 1.297 steps/s · 41.5 stamps/s |
| compute / data-wait | 83.8% / 16.2% |
| MPS driver peak @ batch 32 | 8.07 GB (recommended max 14.3 GB) |
| largest batch that fits | 128 (OOM at 192) |
| fastest batch | 64 · 52.8 stamps/s |
| ops without an MPS kernel | 1 — `aten::_linalg_svd.U`, now explicitly on the CPU |

The M5 Max has more unified memory, so expect the largest-batch figure to move most. Report
whether `PYTORCH_ENABLE_MPS_FALLBACK` is unset there too — if the throughput phase completes with
it unset, nothing fell back silently, and that is the only way to know.
