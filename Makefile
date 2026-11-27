# Galaxy-JEPA local test targets — mirror the CI selection one-to-one
# (.github/workflows/ci.yml) so "green locally" means "green in CI".
# See docs/spec/testing.md and docs/validation/validation_report.md for tiers.

.PHONY: lint unit invariant fast integration e2e calibration mutation research-validate test

# Lint + format check + type-check (the cheapest stage; fails in seconds).
lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src

# The fast gate: unit + invariant + regression + differential + scientific + reduced calibration + mutation.
fast:
	uv run pytest -m "not integration and not e2e and not gpu and not slow and not external_data" --durations=10

# Unit tier only (pure, fast, no GPU/network).
unit:
	uv run pytest -m "not integration and not e2e and not gpu and not invariant and not slow and not external_data"

# Invariant / property tier only (the science-protecting tests).
invariant:
	uv run pytest -m "invariant or property"

# Statistical calibration tier (Type-I error, CI coverage, BY FDR, matching).
calibration:
	uv run pytest -m calibration

# Isolated §14 scientific mutation audit (must kill all 21 critical mutants).
mutation:
	uv run python tools/mutation_audit.py

# Integration tier (component seams, tiny fixtures, CPU).
integration:
	uv run pytest -m integration

# E2E tier — CPU miniature pipeline runs locally without GPU; GPU tests stay marked `gpu`.
e2e:
	uv run pytest -m "e2e and not gpu"

# Full provenance-bound research validation manifest (before GPU rental / publication run).
research-validate:
	uv run python tools/research_validate.py --out runs/validation_manifest.json

# Everything (what a full local check covers before pushing).
test: lint fast integration e2e
