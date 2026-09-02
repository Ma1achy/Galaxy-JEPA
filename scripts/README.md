# Scripts

There are no standalone entry-point scripts: the runnable surfaces are module entry points and
the harness API, so a run is determined by one validated, hashed config rather than by argv.

| What | How |
|---|---|
| Pull a corpus (HTTP) | `uv run python -m galaxy_jepa.data.pull --corpus probe --limit 2000 --out data/probe` |
| Verify the join first | `uv run python -m galaxy_jepa.data.pull --corpus probe --check-join` |
| Backfill derived columns | `uv run python -m galaxy_jepa.data.pull --backfill-derived data/probe-40k` |
| Axis-ratio top-up (D13) | `uv run python -m galaxy_jepa.data.pull --axis-ratios data/probe-40k --limit 40000` |
| Pull at native fidelity, at scale | `python artifacts/sciserver_pull.py --corpus probe --limit 40000 --out data/probe-40k` |
| Pull status | `python artifacts/_pull_status.py probe` |
| Train → freeze → probe → figures | `galaxy_jepa.harness.run_harness(HarnessConfig(...))` — see `configs/pretrain.yaml` |
| Probe-only re-evaluation | `galaxy_jepa.harness.evaluate_probe(config, checkpoint=...)` |
| The full probing battery | `galaxy_jepa.harness.probe_frozen_checkpoint(config, checkpoint=..., scheme=...)` |
| Throughput calibration | `galaxy_jepa.harness.calibrate(...)` — classifies compute- vs data-bound |

**`artifacts/` is not part of the package.** It holds the networked, credential-touching pull
glue and is excluded from lint/CI. The SciServer token lives only in the gitignored `.env` and
is read only there; `pull.py --source sciserver` fails loudly with a pointer rather than
touching the Jobs API.
