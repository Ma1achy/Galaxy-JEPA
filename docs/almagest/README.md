# Almagest — branch `almagest-geometry`

Descriptive work for the Almagest Plate: geometry of A1's answer axes and a diffusion renderer for
visualising points in A1's embedding space. **Nothing here is in the science record**; no rung, claim or
finding depends on it.

| piece | script | outputs |
|---|---|---|
| Spiral axis (steps 1–5) | `artifacts/almagest_spiral.py` | `a1_spiral_report.md`, `a1_spiral_*.json`, figures; `runs/almagest/a1/plate.bin` (v3, spiral) |
| All 37 answer axes | `artifacts/almagest_axes.py` | `a1_axes_report.md`, `a1_axes_table.md`, `a1_axes.json`, `a1_axes_sigma_w_cos.png`, `a1_axes_contact_sheet.jpg`; `runs/almagest/a1/plate_axes.bin` |
| Diffusion feasibility pilot | `artifacts/almagest_diffusion.py` | `diffusion_pilot.md`, `diffusion_pilot_*.json`, reconstruction sheets; `runs/almagest/diffusion/` |
| Diffusion checks: normalisation, native 128-px crop ceiling | `artifacts/almagest_diffusion_checks.py` | `diffusion_checks.md`, `diffusion_checks.json`, `diffusion_checks_padded_crops.png`; `runs/almagest/diffusion/checks/` |
| Diffusion smoke: three input fixes at 64 px (stats only) | `artifacts/almagest_diffusion_smoke.py` | `diffusion_smoke.md`, `diffusion_smoke.json`; `runs/almagest/diffusion/smoke_{full,nosky}/` (checkpoints, ε-bias and clamp diagnostics) |
| Visibility tests B0–B2 (exploratory; feeds the 8×8 pre-registration) | `artifacts/almagest_visibility.py` | `a1_visibility_report.md`, `a1_visibility_b1.{json,png}`, `a1_visibility_b2.json`, `a1_visibility_b2_examples.png`; `runs/almagest/a1/visibility/` (embeddings, post-hoc diagnostics) |

## Downsampling in the diffusion pilot is display-only

The diffusion pilot trains on 64×64 images made by 4×4 area-averaging the native 256-px stamps. That
downsampling exists **only** for this rendering model. It does not touch the science pipeline's no-rebin
rule (`docs/spec/data.md`, proven in `artifacts/fidelity_test.py`): no probe, ladder rung or baseline reads
these images, and A1's conditioning embeddings are always computed from the native 256-px stamps.

The conditioning vector is **raw** A1: frozen `runs/m_v2/encoder.pt`, penultimate block, mean-pooled, 384-d,
the probes' own vector. The nuisance cleaning and γ are used for walk directions only, never for conditioning.

## Write rules

Nothing is written to `artifacts/out` or `runs/ladder_v2*`. Large arrays go to `runs/almagest/`
(on the SSD); reports, small JSON and figures go here.
