# CLAUDE.md — Galaxy-JEPA working conventions

Guidance for working in this repo. The design source of truth is `docs/architecture.md` +
`docs/architecture-buildout.md` + `docs/spec/`; this file is the quick reference for the
hard-won conventions that aren't obvious from the code.

## What this is

A from-scratch JEPA that learns galaxy morphology **label-free**, then reads concepts off the
**frozen** representation with linear probes. The premise is proven at pilot scale
(frozen-probe AUC ≈ 0.91). The standing question the codebase serves: *is a human concept a
direction in the representation, and does distance along it track human judgement?* The probe
is only trustworthy if the experimental guardrails are structural — hence the invariants below.

## Toolchain

- **uv + pytest + ruff** (`docs/architecture-buildout.md:15`). `uv sync --extra dev --extra data
  --extra eval`. Lint/format: `ruff check` + `ruff format`; pin is kept in lock-step between
  `.pre-commit-config.yaml` and `uv.lock` (currently 0.15.17). Types: `mypy src/galaxy_jepa`.
- Python ≥ 3.11. Tests are tiered (`@pytest.mark.invariant`, `@pytest.mark.integration`);
  integration needs the `data`/`eval` extras. **No test touches the network or a GPU.**

## Hard invariants — discipline encoded in code, not at the keyboard

(`docs/architecture.md:41` "Hard invariants"; `architecture-buildout.md:13` "stay hard".)

- **Frozen-encoder discipline.** The probing path only ever receives a frozen encoder —
  there is no `unfreeze=` parameter on it (`docs/architecture.md:45`). The freeze boundary
  runs through disk: `objectives/` writes a checkpoint, `probing/` reads it via
  `load_frozen_encoder` (which `eval()`s + `requires_grad_(False)`); probing never imports
  objectives. `extract_embeddings` asserts `assert_frozen` on entry.
- **Splits leak guards.** Cross-corpus dedup by objID and the uncertainty-geometry partition
  are made *impossible in code*, with a loud failure and a merge-blocking invariant test
  (`docs/spec/splits.md:15`). Use `data/orchestrate.py` (`resolve_corpora`, `split_pretrain`,
  `assign_three_way`) — don't hand-roll splits.
- **Fail loudly, never silently default** (`architecture-buildout.md:13`,
  `architecture.md:15`). A skipped control / silent unfreeze / label leak must raise, not
  shrug. Config uses `extra='forbid'`; an unknown key is a load-time error.
- **Format + stretch + normalise parity** (`docs/spec/data.md:25`). Preprocessing is
  byte-identical across the pretraining corpus, the probing corpus, and every baseline; the
  fp16 pre-bake cache is the parity lock (keyed on the pipeline `config_hash`). Fit
  `Normalise` once on a subsample and **freeze before any training** so the cache tops up
  incrementally (never re-bakes) — see `harness._build_pipeline`.
- **No-rebin** (`docs/spec/data.md:80`, empirically proven in `artifacts/fidelity_test.py`).
  Native 0.396″/px, no resampling — resampling attenuates high-frequency power to ~0.11 of
  native. Stamp size is 256 px (the frozen spec); `bbox.py` geometry is consumed, untouched.
- **Provenance is structural.** A run is determined by `(config_hash, code_sha, data_snapshot,
  seed)` — `core.config.RunStamp`; every artefact is stamped via `write_stamp`. `data_snapshot`
  is the manifest hash over the object IDs + the pull query, not a hand-bumped string.

## Token-only-in-artifacts

The SciServer auth token lives **only** in the gitignored `.env` and is read **only** inside
`artifacts/` (`artifacts/_sciserver_auth.py`, `artifacts/sciserver_pull.py`). It must never
enter the importable package (`src/galaxy_jepa/`). The package contributes only *pure*,
token-free pull helpers (`data/sciserver.py`: `chunk_target_ids`, `merge_corpora`);
`pull.py --source sciserver` fails loudly with a pointer to the artifacts driver rather than
calling the Jobs API. The account is federated Microsoft SSO — **no password login**; the
token is refreshed manually and goes stale (see the user-memory note). Never print/log/commit
the token, and never ask the user to paste it into chat. `artifacts/` is excluded from
lint/CI (`pyproject.toml:45`) — investigation/ops code, deliberately terse, not package code.

## House style

- **British English** throughout (docs, comments, identifiers) — `docs/spec/*` status lines.
- **Second-consumer rule** (`architecture.md:13`, `architecture-buildout.md:14`): build an
  abstraction when the second case actually arrives, not in anticipation. (E.g. the harness
  has a single `build_objective` switch point, not an objective registry, until a baseline
  lands.) "Take the DNA, not the organs."
- Match the surrounding code's comment density and idiom; comments explain *why*, not *what*.

## Workflow

Designs are drafted with browser-Claude against `galaxy-jepa-scratchpad.md` + `docs/`, then
executed here by Claude Code. Keep the scratchpad and derived docs in sync with what was
built. **Commit only when the user asks**; show plans and forks first.
