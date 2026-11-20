# Findings so far: a one-page summary

Plain-language consolidation of the four completed results (20 Nov 2026). Every number is quoted from
the committed report named beside it; nothing here is new analysis. All encoders are ViT-S/16 trained
label-free; morphology is read off the frozen representation with linear probes on Galaxy Zoo 2 answers.
M is the original JEPA (v1 corpus, whose imaging bands were slightly misregistered); A1–A3 are M's exact
recipe retrained on the band-aligned v2 corpus (seeds 0, 1, 2).

## 1. Aligned comparison: removing the band artefact (scored 11 Oct)

`artifacts/aligned_comparison.md`, "Results" (commit `439076a`; A3 report `f88b7d6`).

- **The artefact is gone (FIXED).** Probes for the four band offsets read at chance in both aligned
  seeds (AUC 0.499–0.507); M's strongest offset-tracking latent had |ρ| 0.910, the aligned encoders'
  0.059 and 0.044.
- **Morphology did not improve (SAME).** Mean difference over 33 powered answers +0.0039, inside the
  seed bar ±0.018; 30 answers SAME, 3 UNRESOLVED, none better or worse.
- **Where the capacity went.** The share of the top principal components tracking offsets or brightness
  fell (A1 0.26, A2 0.37 against M's 0.54 and 0.48), but the morphology share did not rise (SAME), and
  PC1 now tracks brightness (18% and 24% of variance). Latents flagged as tracking a nuisance moved from
  the offsets to other nuisances (0.92 and 0.86 against 0.65 and 0.48), so the total barely changed.

## 2. Ladder v2: the existence catalogue on A1–A3 (17 Oct)

`docs/ladder_v2/report.md` (commit `e0c6b0a`); audit COMPLETE, 205/205 entries.

- **Almost everything is present, almost nothing is clean.** Of 37 answers, full population: R2
  (present linearly but entangled or confounded) 33 / 34 / 32 for A1 / A2 / A3; R1 (a clean linear
  direction) 0 / 1 / 1; R4 (not recoverable linearly) 4 / 2 / 4. Conditional population: R1 2 / 2 / 0,
  R2 25 / 27 / 27, R4 10 / 8 / 10.
- **Nuisance picture.** Under matching on each answer's worst nuisance (D24), the signal SURVIVES on
  30 / 31 / 30 answers, is UNRESOLVED on 6 in every seed, and collapses on one (1 arm, A1 only).
- **Winding and arm count are where it thins out.** Medium winding is R4 in all three seeds. 3 arms and
  4 arms are R4 in A1 and A3 and R2 in A2 (both underpowered); star/artifact is R4 in all three
  (underpowered). In the full population the only R1 is irregular (A2 and A3).

## 3. Baselines: MoCo and MAE against the JEPA (15 Nov; exploratory)

`docs/baselines_comparison/report.md` (commit `89e94a0`). Every encoder is a smoke-stamped run, so the
comparison is exploratory.

- **Both UNRESOLVED (underpowered).** The seed-spread bar exceeds the 0.02 smallest effect of interest
  in both, so neither "better" nor "no different" was reachable.
- **MoCo is a near miss.** MoCo − JEPA +0.0419, 97.5% CI [+0.0402, +0.0434]; BETTER needed the CI wholly
  above the bar 0.0404, and it fell short by 0.0002. All three MoCo seed pairs are positive at every epoch.
- **MAE is level with the JEPA:** +0.0044, bar 0.0491.
- **Per answer:** MoCo 9 BETTER (against ≤ 0.9 by chance) and 26 UNRESOLVED; MAE 35 UNRESOLVED.
- **Nuisance readout:** MoCo carries less PSF and less sky information than the JEPA (two LESS calls,
  0.15 expected by chance).

## 4. Comparison with M: did the band artefact change any existence result? (19 Nov)

`docs/m_comparison/report.md` (commit `bd11367`; record `f9eceeb`).

- **No.** Seed-matched pairs (M with A1, O2 with A2), A3 as a check: AGREES on 32 of 32 scoreable
  answers in the full population, THIN 5, no directional state; overall INSUFFICIENT (no answer
  changed rung in both pairs, so there was nothing to count). M's ladder results stand.
- **Reproduction check PASS:** M re-laddered under today's code matches its 23 Sep record in every cell.
- **The limit:** this is present-versus-absent, not magnitude.
- **Run-level spread:** the five runs' mean AUCs are 0.706 (M), 0.687 (A1), 0.702 (O2), 0.724 (A2),
  0.690 (A3), a spread of about ±0.02 around their centre; the A seeds alone span 0.687–0.724.

## Cross-cutting: run-to-run noise is the binding constraint

The same thing limits every comparison: retraining the same recipe with another seed moves the
whole-encoder probe level by 0.02–0.03 AUC. The aligned seeds differ by 0.030 on average (M's two seeds
by 0.006); the JEPA's and MAE's seed spreads (0.017 and 0.020) put the baselines' bars above the
smallest effect of interest; and across the five M-comparison runs the means span about 0.04. Much of
it is one seed: training seed 1 runs about 0.03 high for both the JEPA (A2) and MAE on the v2 corpus,
but not for MoCo and not on v1 (O2, which shares A2's initial weights). With three seeds, any claim of
"better" or "no different" at the 0.02 scale is out of reach; existence (present or absent), which sits
far from the bar on most answers, is robust to it.

## Open items

- **8×8-patch ablation (D11, the Rung-4 control):** whether R4 answers (medium winding, star/artifact,
  3 and 4 arms) are under-resolved at 16-pixel patches or truly absent. Costed in
  `docs/patch8_cost.md`.
- **Masking β-sweep** {0, 0.5, 1.0}; β = 0 is the standard I-JEPA control.
- **Backbone sweep:** clean ViT → conv-stem hybrid → E(2)-equivariant ViT (D2).
- **The seed-1 puzzle** (TODO, P2 parked): why seed 1 runs high on v2 for the JEPA and MAE only; needs
  extra training runs.
