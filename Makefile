# Galaxy-JEPA local test targets — mirror the CI selection one-to-one
# (.github/workflows/ci.yml) so "green locally" means "green in CI".
# See docs/spec/testing.md for the tiers and the staged pipeline.

.PHONY: lint unit invariant fast integration e2e test

# Lint + format check + type-check (the cheapest stage; fails in seconds).
lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src

# The fast gate: unit + invariant tiers (the unmarked default selection).
fast:
	uv run pytest -m "not integration and not e2e and not gpu" --durations=10

# Unit tier only (pure, fast, no GPU/network).
unit:
	uv run pytest -m "not integration and not e2e and not gpu and not invariant"

# Invariant / property tier only (the science-protecting tests).
invariant:
	uv run pytest -m invariant

# Integration tier (component seams, tiny fixtures, CPU).
integration:
	uv run pytest -m integration

# E2E tier — the GPU path CI defers; run it here locally.
e2e:
	uv run pytest -m e2e

# Everything (what a full local check covers before pushing).
test: lint fast integration
