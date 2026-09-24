# Framing A — claims to evidence (Brief X2)

Outline and table only; no prose draft. Every headline number is stated as corrected.
Encoder: M's 4-epoch checkpoint (`runs/m/encoder.pt`). Split: the frozen 40,000 train /
34,829 test. Untrained references are K = 3 draws (a range) unless the ladder's K = 30 bank is
named.

**Status key:** SUPPORTED · SUPPORTED WITH CAVEAT · PENDING · UNSUPPORTED.

---

## Outline

1. **Premise.** A label-free JEPA learns galaxy morphology; human concepts can be read off the
   frozen representation (A1–A3).
2. **The instrument earns trust.** Guardrails are structural; the bar is the untrained
   architecture, not chance; power travels with every verdict (A4–A6).
3. **The catalogue.** A concept is a direction, but almost never an independent one; the
   nuisance does not do the damage (A7–A10).
4. **Shape.** Readout is linear; conditional means bend, mostly as a mixture (A11–A12).
5. **Human judgement.** The geometry tracks the vote structure; the axis reproduces vote
   uncertainty, stated as a margin over untrained (A13–A15).
6. **Beyond the votes.** Graded axes against vote-free measurements (A16–A19).
7. **Salience.** Decodable, not salient; the top components are learned but unnamed (A20–A22).
8. **Nuisances the design left in.** Orientation (D10), inclination (A23–A24).
9. **Withdrawn**, stated as such.
10. **Framing B** (cross-objective): the claims one encoder cannot make, with the rental run each
    needs.

---

## Framing A — one encoder

| # | claim | evidence (brief · findings · figure) | status | reviewer caveats | what would strengthen it |
|---|---|---|---|---|---|
| A1 | Label-free pretraining does not collapse at full scale, and the recipe was selected on the probe, not on the loss | Pilot + H5/I/J/K (D17) · `README.md` "Before the Full Run" · `assets/schedule_dose_response.png`, `assets/resolving_run.png` | SUPPORTED | Schedule chosen on one seed; the loss-vs-probe inversion is one pair of arms | A second seed of the resolving run |
| A2 | The frozen encoder reads smooth/featured at AUC **0.9646** on 34,829 held-out galaxies, range [0.9609, 0.9646] across two training draws | M/N · `m_findings.md`, `n_findings.md` · README "The Catalogue" | SUPPORTED WITH CAVEAT | **The untrained architecture already reaches ≈ 0.79** on this split; the claim is the margin, not the AUC. Two draws is a range, not a spread | State as margin over the K = 30 bar everywhere it appears |
| A3 | The stopping rule, pre-registered, ended training at 4 epochs (101,308 of 253,270 steps) | M · `m_findings.md` | SUPPORTED | Flattening at 4 epochs does not say later epochs add nothing to *other* answers | Probe a later checkpoint on the ladder (descriptive) |
| A4 | Existence is measured against each answer's **untrained-encoder bar** (K = 30, D23), not chance | P · `p_findings.md` §a · `assets/ladder_catalogue.png` (grey ticks) | SUPPORTED | Student-t on K = 30 assumes normality; the Shapiro–Wilk report is still open (TODO P1) | Report the normality check |
| A5 | **33 of 37** answers clear their own untrained bar: the representation contains the tree | P → T1 (D25) · `t_findings.md` §T1 · `ladder_catalogue.png` | SUPPORTED | Existence is not independence; see A7 | — |
| A6 | Power travels with every rung. **Full: 4 R4, of which 3 are underpowered** (star/artifact, 28 positives; 3 arms; 4 arms) and read *"cannot resolve at this N"*, never *"absent"*. The one adequately powered R4 is *winding: medium* (AUC 0.523, bar 0.519). Conditional: 8 R4, 6 underpowered | P §c, T1 · `p_findings.md` §c · `assets/ladder_power.png` | SUPPORTED | "R4" must never be read as absence without the power column; conditional R4s (spiral / no spiral) are powered | — |
| A7 | **One** of 37 answers is a clean independent direction (full: *irregular*; conditional: *no bulge*); the rest that exist are entangled | P → T1 (D25) · `t_findings.md` §T1 · `ladder_catalogue.png` | SUPPORTED WITH CAVEAT | "Entangled" is relative to the other 36 answers' directions and the 0.30 pair floor; binary complements (smooth × features) count as entangled trivially | Report entanglement excluding within-question complements |
| A8 | The nuisance is **not** doing the damage: under retention, none of P's 22 "confounded" answers loses its effect to size, magnitude or redshift; median margin retained 1.00 | R0 · `r_findings.md` §R0 · README "The Catalogue" | SUPPORTED | Matching on 5 quantile strata removes a nuisance only coarsely | Finer matching or a regression-adjusted retention |
| A9 | What keeps most answers off R1 is **effect size** (below the 0.7267 floor), not a nuisance | R0, T1 · `r_findings.md` §R0, `t_findings.md` | SUPPORTED | The floor itself is a design choice (D25 lists its consumers) | — |
| A10 | Old world-correlation verdicts turn **representational** under retention (full: 48 representational, 0 world, 5 inconclusive) | T1 · `t_findings.md` §T1 "Unpredicted" | SUPPORTED WITH CAVEAT | Retention on near-complements (smooth × features) means little | — |
| A11 | **Feature = direction for readout:** no nonlinear headroom on any of 37, both populations; the same MLP decodes random cluster labels at AUC 0.89–0.99 | R1 · `r_findings.md` §R1 | SUPPORTED WITH CAVEAT | Depth-1 MLP, ladder recipe; untrained-MLP bar not measured (see B5) | The untrained-MLP bar (B5) |
| A12 | Conditional means bend for 20 of 36 answers, but mostly as a **mixture of vote-reach groups**; three bend robustly (merger, spiral, anything odd). Instrument validated on a known loop (R3) | R2, R3, S5 · `r_findings.md` §R2–R3, `s_findings.md` §S5 | SUPPORTED WITH CAVEAT | M's loop is recovered only under the amended ≥ 1.5 dimension criterion (non-planar, eff. dim 3.54) | — |
| A13 | Encoder concept geometry tracks human vote structure: **Spearman +0.643** over 528 answer pairs, same galaxies | P · `p_findings.md` · `assets/concept_structure.png` | SUPPORTED WITH CAVEAT | No untrained-encoder RSA reported beside it | RSA on the three untrained draws, stated as a margin |
| A14 | Bar × 2 arms is **shared** with the votes (encoder +0.42, humans +0.38); across eight bar pairs encoder cosine tracks human r at Pearson 0.98 (exploratory) | U1 · `u_findings.md` §U1 | SUPPORTED WITH CAVEAT | Verdict rests on a single pair; the 0.98 is exploratory; untrained mean 0.47 | — |
| A15 | **Fig 2, as a margin:** all 27 answers with a linear direction rank the ambiguous middle by vote fraction, each above all three untrained draws; the **learned margin is +0.04 to +0.23** (smooth/features +0.12 to +0.16; bulge up to +0.2). Untrained already reaches ρ = +0.32 on smooth | U2 → V1.1 · `u_findings.md` "Restated (Brief V1)" · README "The uncertainty geometry" | SUPPORTED WITH CAVEAT | Most of "reproduces human uncertainty" is architectural; K = 3 is a range | Larger untrained K; a second trained draw |
| A16 | Off-axis: ambiguity goes with bend beyond visibility for 13 answers — read by retention: **spiral mostly visibility** (0.27–0.29); **merger reverses** under a single visibility index; **bulge "obvious" was suppression** | U2 → V1.2–V1.3 · `u_findings.md`, `v_findings.md` §V1 | SUPPORTED WITH CAVEAT | Reversal verdicts are post hoc (U2 had no reversal state; D27); V1.3 is exploratory | Pre-registered re-test on a fresh draw with the reversal state |
| A17 | **U3 roundness:** ordered, and tracks measured axis ratio (DECODED 0.54 vs untrained 0.27; vote and measurement directions r_d ≈ 0.98) | U3 → V2 → W1 · `u_findings.md` "Restated (Brief W1)", `v_findings.md` §V2 | SUPPORTED | `expAB_r` is itself a photometric model | — |
| A18 | **U3 bulge prominence:** mostly ordered (the top end, obvious → dominant on 38 test galaxies, is underpowered), and tracks measured B/T both ways (agreement 0.72; corrected 2×2 **BOTH**: A_m 0.27, A_v 0.39 beyond the other decoder and visibility, each ≥ 3× the shared-quantity null) | U3 → V2 → Z3 · `z_findings.md` §Z3 | SUPPORTED WITH CAVEAT | Simard 2011 B/T is a model decomposition with its own degeneracies. **The vote leg is mostly architectural:** untrained encoders reach 0.25–0.28 of A_v's 0.39; the measurement leg's learned margin (+0.15) carries the claim. V2's original form (0.36 / 0.47) overstated both legs | — | — |
| A19a | **An algorithmic handedness label exists on our stamps:** log-polar chirality χ agrees with GZ1 CW − ACW at ρ = −0.64 (n 3,093); the sign fixes the GZ1 ↔ array convention | X1b · `x_findings.md` §X1 | SUPPORTED WITH CAVEAT | Sign convention established empirically after a pre-registered prediction failed (MIRROR CONTRADICTED on the stated convention) | PyArcFiRe spin on the same galaxies (Brief Y) |
| A19 | **U3 winding:** ordered along an axis fitted on the extremes. Against machine-measured pitch (Brief Y): the votes' ordering tracks SpArcFiRe pitch **beyond visibility** (τ 0.25 → 0.22 Hayes; 0.23 → 0.23 Hart), and w_avg agrees with it (ρ −0.35 / −0.32). The encoder's winding is vote-like: VOTES BEYOND MEASUREMENT (A_v 0.27; A_m 0.097), pitch decode margin over untrained only +0.02 to +0.06; its axis meets pitch mostly through visibility (retention 0.42) | U3 → W1 → Y · `u_findings.md`, `y_findings.md` §Y2–Y3 | SUPPORTED WITH CAVEAT (ordering is physical within SpArcFiRe) | **Pitch angle is not a usable independent reference (Z1: DISAGREE).** Hart (SpArcFiRe) vs Yu & Ho (2DFFT): ρ −0.25, 95% upper bound +0.10, n 34 (the whole overlap), against the ≥ 0.5 two published methods should reach. So "physical" here means "within SpArcFiRe". Yu & Ho 2DFFT also agrees with neither the votes nor SpArcFiRe (n 550, 376); Hayes vs Hart only ρ 0.38; Hayes is selected on P_CS > 0.5 and disc axis ratio > 0.5 (fainter, smaller than our spirals) | A larger 2DFFT sample; PyArcFiRe on our own stamps (parked) |
| A20 | **U3 arm count:** ordered | U3 → V1 → W1 · same | SUPPORTED WITH CAVEAT — **not established as morphology** | Visibility falls monotonically from "1 arm" to "4+"; can't-tell galaxies are *more* visible (AUC 0.40) | Exploratory (Y3 T5): SpArcFiRe arc counts track the voted arm count beyond visibility (partial +0.24 to +0.29) — some content beyond the gradient; a pre-registered test would decide it. **Limitation (Z1):** those arcs come from the one algorithm whose pitch no second method supports (DISAGREE, ρ −0.25, n 34), so arc counts are not an independent reference either |
| A21 | **Salience (held-out):** no human concept is one of the encoder's own components; best frozen component ≤ 0.63 of the probe's margin; 15 of 33 no better than the same protocol untrained. *Decodable, not salient* | V3 → W1 · `v_findings.md` "Corrected (Brief W1)" | SUPPORTED | PCA-only discovery; ICA/sparse/SAE not run | A second discovery method through the same held-out protocol |
| A22 | The top two components (PC1 24.9%, PC2 11.9%) are **learned** and are **not morphology**: they transform as the x/y components of an image-plane vector (rot90 maps PC1 → PC2; mirror flips PC1 only), equally in smooth and spiral galaxies; **not handedness** (X1: ORIENTATION-LIKE; GZ1 and χ both n.s.). 41% of M's variance is mirror-odd vs 2–7% untrained | W2, X1 · `w_findings.md` §W2, `x_findings.md` §X1 · `out/w2_pc12_plane.png` | SUPPORTED WITH CAVEAT | What the vector *is* stays open: not patch-grid or sub-pixel position (interventions), not frame-level PSF, not neighbours/padding/sky | Per-object PSF asymmetry; attribution maps; D10-augmented retrain (B4) |
| A23 | **Orientation is carried, and mostly architecturally** (ridge R² 0.505 trained vs 0.42–0.44 untrained); a consequence of D10's unimplemented augmentation | R3, S4 · `r_findings.md` §R3–R1 bottom line, `s_findings.md` §S4 | SUPPORTED | A spec/code divergence, disclosed | Augmented retrain (Framing B, B4) |
| A24 | Spiral's bend is **not inclination** (share explained 0.087 full, 0.027 well-voted); training grew the bend ~3× without adding inclination | S1 → T3 · `s_findings.md` §S1, `t_findings.md` | SUPPORTED | Carried forward as visibility (T3) | — |
| A25 | `t08 other`'s direction is a mix (close companions and …), not a category | T2 · `t_findings.md` §T2 | SUPPORTED | Grid inspection is qualitative | — |

## WITHDRAWN

| claim as once made | where it was made | why withdrawn | where recorded |
|---|---|---|---|
| **Size dominates the catalogue** — "nineteen of thirty-seven lose their effect when apparent size is matched" | P write-up, README Catalogue (old), `ladder_catalogue.png` right panel (old), two commit messages | The gate's arithmetic: 18 of 19 (21 of 22 overall) were below the 0.7267 floor *before* matching; matching moved AUC by a median 0.027 | `r_findings.md` §R0; README corrected |
| **World-correlation adjudication** (18 world-correlation pairs per population; edge-on × cigar as "geometric necessity") | P §(f), P2 | Every old world verdict was a matched AUC under the floor; under retention 0 remain in full, 1 in conditional | `t_findings.md` §T1 "Unpredicted"; D25 |
| **Bar × winding as a resolution of the D13 hard case** (Hart 4–6° looser arms) | README inset, P §(f) | +0.042 to loose is inside the noise of two random directions; neither pair cleared the 0.30 floor, so stage 1 never adjudicated them; volunteers do not tie bar to winding (|r| ≤ 0.19) | U0/U1 · `u_findings.md` §U1; README "Restated (Brief U0)" |
| **Arm count as morphology** | U3 result as first read | Ordering is substantially a visibility gradient | W1 · `u_findings.md` "Restated (Brief W1)" |
| **The V3 energy test** (salience by CAV projection energy) | V3 result | VOID: the shuffled-CAV null median sat at 0.755 of the range, so no concept could exceed it (D28) | `v_findings.md` "Corrected (Brief W1)" |

## Framing B — cross-objective (needs more than one encoder)

Each claim below is **not made** under Framing A. It names the rental run it needs.

| # | claim Framing B would make | why one encoder cannot | rental run needed | status |
|---|---|---|---|---|
| B1 | The learned margins (A5, A15, A21) are **the JEPA objective's doing**, not any self-supervised objective's | One objective is not a comparison | **MAE baseline** and **MoCo baseline**, same ViT-S/16, corpus, split and ladder | PENDING |
| B2 | "Decodable, not salient" (A21) is a property of **latent prediction** specifically; a contrastive objective would put concepts in its top components | Salience is objective-dependent in principle | **MoCo baseline** (and MAE), same held-out recovery protocol | PENDING |
| B3 | Fine-scale answers (winding, arm count, A19–A20) are limited by **16-px patches** | Patch size and objective are confounded in M | **8×8 patch ablation** | PENDING |
| B4 | The effective-rank dynamics and the D17 schedule result generalise beyond **batch 32**; the orientation nuisance (A23) is removed by D10's augmentation | M was trained at one batch size, no augmentation | **Batch-size run** (towards the I-JEPA reference); augmented retrain | PENDING |
| B5 | Nonlinear headroom (A11) is null **relative to the architecture**, not just absolutely | The MLP has no untrained bar | **Untrained-MLP bar** (MLP on untrained-encoder embeddings, K ≥ 3) | PENDING |
| B6 | The image-plane vector code in PC1/PC2 (A22) is an objective-specific signature, and augmentation removes it | One encoder's PCs cannot be compared across objectives | MAE / MoCo baselines through W2 + X1 | PENDING |
