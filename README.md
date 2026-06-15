# Galaxy-JEPA

*v2 of Galaxy-Zoo-Classifier. Status: planning & scaffolding — **not training**.*

Pretrain a galaxy-image representation **unsupervised** with an I-JEPA, **freeze**
it, then use Galaxy Zoo labels **only as a read-out key** — to *name and test*
directions the representation already learned, never to train it. The science is
the **nameability ladder** + **uncertainty geometry**, not classification
accuracy. **Paper 1 is single-survey (SDSS / GZ2).**

## Documents

- **`galaxy-jepa-scratchpad.md`** — design source of truth.
- **`PROJECT_PLAN.md`** — phased plan; Paper 1 critical path → the three headline
  figures.
- **`TODO.md`** — prioritised backlog (controls are first-class tasks).
- **`DECISIONS.md`** — open forks with recommendations.
- **`docs/masking.md`** — bounding-box-biased masking design note.
- **`docs/related-work.md`** — arXiv sweep (no firstness claims).

## Guardrails

Single-survey for Paper 1 · encoder frozen for all probing · controls mandatory ·
non-circular uncertainty protocol · canonical L2 logistic probe · SSL baselines
(MAE, contrastive) as controls · no firstness claims.

## Development

```bash
uv sync --extra dev          # create env + install deps (Python 3.11)
uv run pytest                # run tests
uv run pre-commit install    # enable lint/format hooks
```

A devcontainer (`.devcontainer/`) provisions Python 3.11 + uv. **No model or
training code yet** — `src/galaxy_jepa/` subpackages are placeholders until the
plan and the masking approach are signed off.
