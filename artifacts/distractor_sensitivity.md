# Distractor sensitivity across objectives — pre-registration (DRAFT, not hashed)

**Question.** How much of each encoder's representation is spent on instrumental nuisance rather than
morphology?

**Status.** Draft v3, 2026-10-02 (Kickoff B task 3; Follow-up item 6; user's 28 September item 2).
Scope as the user settled it: every arm is pre-registered now, including MAE and MoCo, which are
**pending baselines (Kickoff F)** and scored only once they exist. The plants run on M, O2 and the
untrained encoder (D28), and the user approves before hashing. Once settled, the section from
"## Pre-registration" through its last content line is hashed (SHA-1, trailing newline), **before any
MAE or MoCo embedding is read**. Changes from v2: (1) UNRESOLVED is now the CI **straddling** the
detectable threshold (lo < MDD < hi, or lo < −MDD < hi), replacing v2's half-width rule, in the index,
the secondary index, the per-variable states and the one-seed read; (2) `sky_r` is pulled (record
pinned) and is variable 7; `snr_r` is a declared secondary, reported with no state; (3) all 17 plants
are rerun once with `sky_r` under the new rule; v2's output on record is kept as
`distractor_plants_v2.json`. (A stray duplicate of the v1 status paragraph, which opened with a
second "## Pre-registration" line, is removed.) **v3 rerun done (2026-10-02 19:28):** every plant
fires but two, the thin-n plant now reads UNRESOLVED, and MDD = 0.014257, MDD′ = 0.005435 (see "Plant
results — v3"). **Not hashable yet:** two plants fail as written: the null (one draw's NI CI misses 0)
and the secondary index's Δ = 0 plant (SEEDS DISAGREE 2 of 20). Both are diagnosed and their fixes
proposed, not applied (D28: kept on record beside the fix); the user decides, then one rerun.

## Pre-registration

### Common ground

- **Encoders.**
  - **J1, J2 (the JEPA pair)**: J1 = M (`runs/m`, training seed 0, 101,308 steps); J2 = O2 (`runs/o2`,
    training seed 1, M's `config_hash`, splits and v1 data, stopped at 101,308 steps). Banks
    `artifacts/out/c2_m{1,2}_embeddings.npz`. **The J1–J2 spread is the noise term** throughout.
  - **MAE1, MAE2** and **MoCo1, MoCo2**: **pending baselines (Kickoff F)**. Not yet trained
    (`TODO.md:468-469`; D12, `DECISIONS.md:414-433`). Before either is scored (settled, user
    2026-09-27):
    - two training seeds each, on the same SDSS pretraining corpus as M (D12);
    - the same corpus **version** as M: **v1 pretrain**. The four offsets below are v1's recorded
      misregistration, so a baseline trained on pretrain_v2 would be compared on a nuisance it never
      saw, against a JEPA pair that did;
    - a flip-free recipe: MoCo without horizontal flips (Brief S4's forward constraint; M has none),
      and the MAE recipe checked for flips the same way;
    - **the same backbone and patch size as M: ViT-S/16.** D12's Wu & Walmsley MAE uses 8×8 patches;
      at 16×16 the MAE row compares objective alone, not objective and tokeniser at once. The
      deviation from D12's recipe is declared on the MAE row;
    - **the same training length as M: 101,308 steps** (matched compute), not each recipe's own
      schedule.
  - **U (the untrained floor)**: M's architecture, random initialisation, frozen
    (`probing.controls.untrained_encoder_matrix`), **three seeds**: seed 0 is O1's bank
    (`o1_embeddings.npz`), seeds 1 and 2 are Brief R's (`r_untrained_seeds.npz`), all co-indexed over
    the capped train + test galaxies below (checked in every run). Read on the same v1 stamps as J1
    and J2.
- **Read-out equivalence.** Every encoder is read by M's convention (`models/vit.py:6-7,172`,
  `galaxy-jepa-scratchpad.md:239`): the **penultimate block** (block 11 of 12 for M, O2 and U;
  `DEFAULT_LAYER = −2`), **mean-pooled over all patch tokens**, **no CLS token** and no final norm.
  - For a baseline of different depth, the read-out is its own penultimate block (index −2).
  - A CLS or register token, if the recipe has one, is excluded from the mean.
  - With 8×8 patches the mean runs over 1,024 tokens instead of 256; the pooled vector is still one
    `embed_dim` vector per galaxy. The probes standardise it, and the PC shares are scale-free, so a
    different width does not enter either measure. PR is width-dependent and carries no state.
  - The layer is not optimised per encoder (the scratchpad's rejection of a best-layer sweep).
  - Every encoder reaches the probe through `load_frozen_encoder` (`eval()` + `requires_grad_(False)`).
- **Galaxies.** All on the v1 probe corpus and its splits (`assign_three_way`, seed 0).
  - Criteria 1 and 2: the DD cap, 40,000 train galaxies (H5's deterministic stride), and the whole
    probe-test split, 34,829 galaxies. These are exactly N1's galaxies (`n_findings.md`, t01's panel)
    and criteria 1 and 3 of `aligned_comparison.md`.
  - Criterion 3: the headline protocol, the full probe-train split (160,857) and the whole test split.
- **Nuisance variables — the seven instrumental ones.** Each probe trains on the train galaxies with
  the value recorded and is scored on the test galaxies with the value recorded.
  1. **Band offsets, four:** g−r x, g−r y, i−r x, i−r y, v1's recorded in-stamp offsets from
     `data/probe_v2/cut_log.csv` (`dd_sae_cards._panel`). The target is sign(offset), exactly
     criterion 1a's (`aligned_c13.c1_arrays`).
  2. **PSF:** `psfWidth_r` (FWHM, arcsec).
  3. **Brightness:** `modelMag_r`.
  4. **Sky: `sky_r`** (PhotoObjAll, the sky flux at the object's centre after deblending,
     nanomaggies/arcsec²), median split as the other continuous variables. Settled (user, 2026-09-27:
     `sky_r` if cheap to add to a PhotoObj query; 2026-09-28: pull it now).
     - **The pull, pinned** (`distractor_plants.py pull-sky`, 2026-10-02; record
       `artifacts/out/distractor_sky_r.json`): public SkyServer SQL (`metadata.run_sql`, DR17, no
       token), `SELECT CAST(p.objID AS varchar(20)) AS objID, p.sky_r FROM PhotoObjAll p WHERE p.objID
       IN ({})`, 100 ids per call (400-id lists failed to connect that week), each batch kept as it
       landed. The galaxies are criteria 1 and 2's: the capped train + test, 74,829 ids (O1's bank ids,
       which every plant run checks equal to them; ids SHA-1 `3ff38816…`). Requested 74,829, returned
       74,829, **missing 0**; sentinels (≤ −9000) 0, zero or negative 0. Median 3.97, 1–99% 2.73–7.34,
       maximum 24.7 nanomaggies/arcsec². Output `artifacts/out/distractor_sky_r.csv`, SHA-1
       `6fa39dbb9a36e254b160c937a76f4012132dd24d`; the plant script refuses a file that does not
       match its record.
     - On these galaxies `sky_r` is nearly independent of the other two continuous variables:
       Spearman ρ +0.04 with `modelMag_r`, +0.00 with `psfWidth_r` (and −0.15 with `snr_r`).
  - **Declared secondary, reported with no state: `snr_r`** (the image-domain SNR,
    `metadata.photometric_snr`, const / `modelMagErr_r`). It held slot 7 through v2. It measures depth,
    which mixes source brightness with sky noise, so it is largely brightness (Spearman ρ −0.84 with
    `modelMag_r` on the test split); as variable 7 it entered brightness into the index about twice.
    Its AUC and CI are reported per encoder beside the index; it is in neither index and carries no
    state.
  - **Reported, no state:** redshift (`specz`) and size (`petroRad_r`, without the
    `petrorad_suspect` rows), N1's two physical nuisances, for continuity with N1's panel.
- **Continuous variables: the median-split AUC, not ridge R².** PSF, magnitude and sky are binarised
  at each split's own median, as N1's panel does (`LabelProvider.nuisance_label`); settled (user,
  2026-09-27) as the protocol throughout. The probe is the one fit path (`probing.logistic._fit`: standardised L2 logistic, C = 1), which is also criterion 1a's
  probe. Why:
  - **Comparable with M's existing numbers.** N1's panel (`n_findings.md:71-86`: magnitude 0.903, size
    0.906, SNR 0.869, redshift 0.837, PSF 0.819) is this probe on these galaxies. t01's eligible rows
    are all 40,000 train and 34,829 test galaxies, so the protocol reproduces N1 rather than
    approximating it. The full run checks this (precondition below).
  - **One scale.** The offsets are AUCs. A median-split AUC puts all seven readabilities on one scale,
    so the index can average them.
  - **Robust.** It is scale-free and insensitive to the heavy tails of PSF and sky.
  - The cost, declared: the split discards ranking within each half, and ridge R² would keep it. R² is
    not computed.
- **Bootstrap.** Criterion 2's paired Poisson galaxy bootstrap of the test split (`aligned_c2.weights`,
  2,000 draws), with weights shared across encoders and variables.
  - The point AUCs are criterion 1a's; the CI differs from 1a's 10,000-draw multinomial loop, because
    a comparison needs draws paired across encoders.
- **Preconditions** (full run; if either fails, nothing below is scored until it is explained):
  - J1's PSF, magnitude, SNR, redshift and size AUCs reproduce N1's to within 0.001;
  - the pread r-flux equals `aligned_c13._r_flux` (rtol 1e-12) on the first 512 test galaxies.

### 1. Readability — how readable is the nuisance?

- **Per encoder e:** AUC_ev for the seven variables, with a 95% bootstrap CI.
  - The **nuisance index** NI_e = mean over the seven of (AUC_ev − 0.5), with equal weights.
  - It is signed: an AUC below 0.5 lowers the index. A variable whose CI lies wholly below 0.5 is
    flagged BELOW CHANCE and reported (a probe that anti-generalises), not clipped.
  - Range: 0 is nothing readable; 0.5 is every variable perfectly readable.
- **Per baseline B ∈ {MAE, MoCo} against the JEPA pair:** criterion 2's statistic unchanged
  (`aligned_c2.statistic`), with the seven variables in place of the answers:
  - D_v = mean(B1, B2)_v − mean(J1, J2)_v, and S_v = (|J1_v − J2_v| + |B1_v − B2_v|) / 2.
  - D̄ = mean_v D_v = NI_B − NI_J, and S̄ = mean_v S_v, the pooled seed spread of both arms.
  - The 95% CI of D̄ comes from the shared bootstrap.
- **The smallest detectable difference, MDD** (the power check, criterion 2's realistic width, fixed
  from M and O2 alone before any baseline is read): J1 and J2 are two genuinely independent draws, so
  sd(D̄) ≈ sd_boot(NI_J1 − NI_J2) / √2 over the 2,000 shared draws, and
  **MDD = S̄_J + 1.96 sd(D̄)**, with S̄_J = mean_v |J1_v − J2_v|. On v3's full run (`sky_r` as
  variable 7) **MDD = 0.014257** (S̄_J 0.013582, sd(D̄) 0.000344; `distractor_plants.json` 'power'), and
  the secondary index's **MDD′ = 0.005435** (S̄′_J 0.004834, sd 0.000307). v2's run (`snr_r`) gave
  0.012765; it is superseded. It is a constant of this pre-registration: it is not recomputed on a baseline's rows, seeds or galaxies.
- **States of the index, in precedence** (the first that applies is the state):
  1. **INSUFFICIENT:** any of the seven variables, in any of the four encoders, has fewer than 500 test
     galaxies or a smaller class below 100. The index is a fixed set of seven, so a variable is not
     dropped to rescue it.
  2. **UNRESOLVED (straddles the MDD):** the CI of D̄ straddles the detectable threshold,
     lo < MDD < hi or lo < −MDD < hi, whatever the bar S̄. The measurement cannot say whether the
     difference is above or below the smallest one this design is powered to detect, so no direction or
     equivalence is read from it. Settled (user, 2026-09-28, replacing v2's half-width rule) after the
     thin-n plant read SAME: its independent score noise inflated S̄ from 0.012 to 0.019, so a CI too
     wide to settle anything still sat inside ±S̄, and its half-width (0.0092) was below the MDD while
     the CI ran across it. It stands above SEEDS DISAGREE because that state reads the two B seeds'
     point NIs, which such a CI does not resolve either.
  3. **SEEDS DISAGREE:** the two B seeds lie on opposite sides of the JEPA mean NI, each by more than
     2 s_J, where s_J = |NI_J1 − NI_J2|. The baseline's two seeds disagree in direction beyond JEPA's
     own seed noise, so the objective does not decide it at two seeds. (The margin is 2 s_J, not
     s_J: at 1 s_J the Δ = 0 plant false-called, see the plant results.)
  4. **MORE:** CI(D̄) lies wholly above +S̄.
  5. **LESS:** CI(D̄) lies wholly below −S̄.
  6. **SAME:** CI(D̄) lies inside [−S̄, +S̄].
  7. **LEANS MORE / LEANS LESS:** the CI excludes 0 but is not beyond ±S̄. The difference is
     significant against galaxy sampling but not larger than seed noise; significance and magnitude
     disagree, and it is named as such, not read as MORE or LESS.
  8. **UNRESOLVED:** otherwise. The CI includes 0 and reaches beyond a bar, so it is not settled.
- **Per variable** (criterion 2's per-answer rule): the shrunk bar B_v = (S_v + S̄) / 2, with the
  bootstrap p-values of the MORE and LESS directions BY-corrected across the seven at q = 0.05.
  - States, in precedence: **INSUFFICIENT**; **UNRESOLVED (straddles the MDD)**, when D_v's 95%
    bootstrap CI straddles the same MDD (lo < MDD < hi or lo < −MDD < hi), whatever B_v; **SEEDS DISAGREE** (the two B
    seeds on opposite sides of the JEPA mean of that variable, each by more than 2 × |J1_v − J2_v|);
    then the shrunk-bar read **MORE / LESS / SAME / UNRESOLVED**. The MDD is the index's; the
    per-variable rule uses it unchanged, because a per-variable power check is not part of the draft.
  - Variables whose state runs opposite to the index are listed as **counter-direction**. For example,
    B reads MORE overall but LESS on PSF. They are reported beside the index and do not change it.
- **One seed** (a baseline with a single trained seed): D = NI_B1 − mean(NI_J1, NI_J2), against the
  JEPA spread alone, S̄_J = mean_v |J1_v − J2_v|. The baseline's own seed noise is unknown.
  - The same thresholds and precedence apply (INSUFFICIENT, then UNRESOLVED when the CI straddles ±
    the same MDD, then MORE / LESS / SAME / UNRESOLVED), and the state is suffixed **(PROVISIONAL, one seed)**.
  - SEEDS DISAGREE cannot be read, and per-variable states are not given (D_v is reported).
  - When the second seed lands, the two-seed read replaces this one; the two are never combined.
- **Two indices** (settled, user 2026-09-27). The **primary** index is NI above, equal-weighted over the
  seven; it governs the reading. The **secondary** index NI′ is the same mean over the **six without
  PSF** (g−r x, g−r y, i−r x, i−r y, magnitude, sky). Why: PSF carries almost all of the JEPA
  seed noise (J1 and J2 differ by 0.066 on PSF and by at most 0.007 elsewhere), so it sets most of
  the primary bar, and NI′ asks the question at the resolution the other six allow.
  - NI′ has its own states, by the same rules and the same precedence, with every quantity taken over
    the six: D̄′, S̄′ = mean over the six of S_v, s_J′ = |NI′_J1 − NI′_J2| for SEEDS DISAGREE, INSUFFICIENT
    over the six, and its own MDD′ = S̄′_J + 1.96 sd_boot(NI′_J1 − NI′_J2)/√2, fixed from M and O2 in
    the same run (value *(from the rerun; not yet run)*).
  - NI′ carries index states only. The per-variable states are the primary's (BY across the seven)
    and are not recomputed over six.
  - It is reported beside the primary and qualifies it; it never replaces it. A primary SAME with a
    secondary MORE, for example, is read as "no difference at the primary's resolution, which PSF's
    seed noise sets; more on the six others", not as MORE.
- **Reachability, declared:**
  - The index's headroom above the JEPA pair is 0.5 − max(NI_J); MORE is reachable if the headroom
    exceeds S̄ + the CI half-width.
  - Per variable, the headroom is 1 − max(AUC_J). On the i−r offsets J already reads about 0.97, so a
    per-variable MORE there has about 0.03 of room. The per-variable headroom is reported, and a
    variable whose headroom is below its B_v cannot read MORE.
  - The realistic-width check is the MDD above; it also carries a state (UNRESOLVED when the CI
    straddles ±MDD).
- **Reported, no state:**
  - U's AUCs and NI_U (and NI′_U), the floor, for three untrained seeds, as a **range** (min–max), never a
    spread.
  - The excess NI_e − NI_U for every encoder, against each end of the untrained range: the readability
    that training added over a random ViT on the same pixels.
  - The redshift and size AUCs, and `snr_r`'s (the declared secondary, see variable 7).
- **Why U gets no state.** U is not a candidate objective, and "J reads more nuisance than a random
  network" is not the question. A state would also need U's own seed noise, and three untrained draws
  give a range, not a spread to test against, as Brief R did.

### 2. Representation — how much of the leading variance tracks nuisance?

Criterion 3's machinery (`aligned_c13.representation`) on the test split's pooled embeddings, per
encoder.

- **Top 10 PCs.** Each PC's variance share, and its galaxy-level |Spearman| with four families:
  - offsets: the four above;
  - brightness: `modelMag_r` and the total r flux (criterion 3's);
  - observing: `psfWidth_r` and `sky_r` (`snr_r` until v3, with variable 7);
  - morphology: the 37 raw vote fractions (not debiased), each on its eligible galaxies.

  A PC **tracks** the family with its largest |ρ| if that |ρ| ≥ 0.3 (criterion 3's rule); otherwise
  it is untracked. Each PC's top family is always reported.
  - **N** = the top-10 variance share tracking offsets, brightness or observing. This is criterion
    3's N with the observing family added.
  - **Mo** = the share tracking morphology.
  - The participation ratio PR is reported, with no state.
- **States, per baseline, for N and for Mo separately:** criterion 3's seed-range rule **with a
  materiality floor of 0.10**. Settled (user, 2026-09-27): the floor stays, justified by the plants —
  on the shares null (Δ = 0, 2,000 draws) the false-call rate is **0.25%** (required ≤ 5%), and a
  +0.20 shift reads MORE **1,999 of 2,000** (full run, v1 plants).
  - **MORE:** both B seeds above both J seeds **and** D = mean(B) − mean(J) ≥ +0.10.
  - **LESS:** both B seeds below both J seeds **and** D ≤ −0.10.
  - **SPLIT:** one B seed above both J seeds and the other below both, each at least 0.10 from the J
    mean.
  - **UNRESOLVED:** both B seeds beyond both J seeds in one direction, but |D| < 0.10.
  - **SAME:** otherwise.
  - One seed: B1 beyond both J seeds with |D| ≥ 0.10, suffixed (PROVISIONAL, one seed).
  - **Why the floor.** Under an exchangeable null of four seeds, "both B beyond both J" happens with
    probability 2!·2!/4! = 1/6 in each direction, so the bare seed-range rule false-calls 1 time in
    3. A margin scaled by the one observed J spread barely helps, because that spread is itself one
    draw. A fixed floor of 0.10, about 2.3 τ̂ for τ̂ = |N_J1 − N_J2|/√2 ≈ 0.043, holds the null to
    about 2% and detects a 0.20 shift about 99% of the time (simulated before the plants; the
    shares-null plant below measures it). A null's test is its false-call rate (MORE, LESS or SPLIT at
    most 5%), because UNRESOLVED is a named non-call.
- **Reachability, declared:**
  - Both J encoders have **Mo = 0**: no top-10 PC tracks morphology at |ρ| ≥ 0.3. So **Mo LESS and Mo
    SPLIT are unreachable**, and "B spends less of its leading variance on morphology than JEPA" cannot
    be observed. Mo MORE needs a morphology-tracking share above 0 in both B seeds and a mean of at
    least 0.10.
  - N MORE needs both B seeds above max(N_J) and mean(N_B) ≥ mean(N_J) + 0.10. N LESS needs the
    mirror.
- **Declared: U is not a floor here.** A random ViT's leading variance is almost all total flux (in
  the full plant run, U's PC1 holds 82% of the variance and tracks brightness at |ρ| 0.95, so
  N_U ≈ 0.875).
  N_U is reported with no state. The floor reading applies to criterion 1 only.

### 3. Morphology — context only, no state

- The mean AUC over criterion 2's **33 powered answers** (`artifacts/out/c2_plants.json` 'powered'),
  per encoder. It uses the headline protocol: `_fit` with C from `probe.yaml` on the full train split
  (`aligned_c2.fit_scores`), and AUC on the whole test split.
- Reported for J1, J2, U, MAE1–2 and MoCo1–2 beside criteria 1 and 2, so a reader can see whether an
  encoder that reads less nuisance also reads less morphology.
- It carries no state and does not change a reading. It needs full-split banks: J1 and J2 have them.
  U does not yet (O1's bank is capped-train + test), so `distractor_plants.py embed-untrained` is run
  before scoring. Each baseline is embedded on the full split (`aligned_c2.py embed <tag> <ckpt>`).

### Reading — the answer to the question, per baseline

The "how much" is answered by the reported quantities: NI_e, the excess over U, and N_e. The states
answer the comparative question, whether B spends more or less of its representation on
instrumental nuisance than JEPA does.

| 1 (index) | 2 (N) | reading |
|---|---|---|
| MORE | MORE | B spends more on instrumental nuisance than JEPA: it is both more readable and more of the leading variance |
| MORE | SAME / UNRESOLVED | B encodes the nuisance more readably, but not in more of its leading variance |
| MORE | LESS | **MIXED**: more readable but less dominant. Named, with no summary direction |
| SAME | SAME | The same as JEPA within seed noise |
| SAME | MORE / LESS | The same readability, with the leading variance allotted differently; the per-PC table names where |
| LESS | LESS | B spends less on instrumental nuisance than JEPA |
| LESS | SAME / UNRESOLVED | B encodes the nuisance less readably, and it is not less of its leading variance |
| LESS | MORE | **MIXED**, the mirror of MORE / LESS |
| LEANS MORE / LEANS LESS | any | A direction significant against galaxy sampling but inside seed noise; not settled at two seeds |
| UNRESOLVED | any | Not settled at two seeds |
| SEEDS DISAGREE | any | B's seeds disagree in direction; unanswered at two seeds. N SPLIT is read the same way |
| INSUFFICIENT | any | Unanswered |
| any, PROVISIONAL (one seed) | any | As above, provisional; replaced by the two-seed read |

Precedence: criterion 1 governs the reading and criterion 2 qualifies it. Criterion 3 is context and
never changes a reading. MAE and MoCo are each read against JEPA. A direct MAE-against-MoCo statement
is not pre-registered and would be exploratory. No criterion's state is revised after another's is
seen.

### Plant evidence (pinned)

SHA-1 of each file as it stands after the full plant run. The hash of this section covers these pins.
*(Provisional until the two v3 failures are settled and the plants rerun: the first three pins are then
replaced.)*

| file | role | SHA-1 |
|---|---|---|
| `artifacts/out/distractor_plants.json` | criteria 1 and 2 plants, v3 full run | `84464a45d305d8ac3b495ecd23b53b682bea02f5` |
| `artifacts/out/distractor_v3_diag.json` | diagnosis of v3's two failures (20 null draws, 200 secondary Δ = 0) | `ed399a8840f399247820b8f8e52be3a45d37ae7c` |
| `artifacts/distractor_v3_diag.py` | its script | `9c0af22003864b2bb16691cd5795eca1ef9864c3` |
| `artifacts/out/distractor_plants_v2.json` | v2's output on record (v1's run, `snr_r`, half-width rule not yet in it) | `b042f29fa8a0ff96a112a1e7046fadb645e71e9a` |
| `artifacts/distractor_plants.py` | the plant script, as run for v3 | `b4d1041a347b6121195fa6114236f63e653e425a` |
| `artifacts/out/distractor_sky_r.csv` | variable 7, the `sky_r` pull | `6fa39dbb9a36e254b160c937a76f4012132dd24d` |
| `artifacts/out/distractor_sky_r.json` | the pull's record (query, batch, n, output SHA-1) | `ecc144debb3b4c533fb62789875bc75419362803` |
| `artifacts/aligned_c2.py` | criterion 2's statistic, imported | `5428d679770696f9a4ddfbc037961a11c1864b9f` (as pinned in `aligned_comparison.md`) |
| `artifacts/aligned_c13.py` | criterion 3's representation, 1a's arrays, imported | `2cc7cffdf1b8bc0bfcfa8abd38787e7b87ce8e89` (as pinned in `aligned_comparison.md`) |

## Settled (user, 2026-09-27)

- "Draft now, M/O2 + untrained": every arm is pre-registered now, MAE and MoCo included; the plants run
  now on M, O2 and U; MAE and MoCo are scored only once they exist, with two seeds each.
- Follow-up item 6 (the v1 open questions):
  - **The rule, not the plant.** The index and every variable are UNRESOLVED whenever the CI
    half-width exceeds the MDD (0.012765, the power check's value fixed from M and O2), whatever the
    bar; precedence stated in criterion 1. The thin-n plant is not changed. *(Superseded on
    2026-09-28: the CI straddling the MDD replaces the half-width; see below.)*
  - **Baselines** train on **v1 pretrain**, with **M's backbone and patch size (ViT-S/16)**, for
    **101,308 steps** (matched compute).
  - **Materiality floor 0.10** stays, justified by the shares plants (0.25% false calls at Δ = 0;
    MORE 1,999 / 2,000 at +0.20).
  - **Sky:** `sky_r` if cheap to add to a PhotoObj query, otherwise `snr_r` as the declared
    stand-in. Measured cheap (3.9 s per 10,000-row page, about 1.5 min for the corpus). *(Pulled
    2026-10-02; see below.)*
  - **Median-split AUC** throughout.
  - **Three untrained seeds**: seed 0 is O1's bank, seeds 1 and 2 from `r_untrained_seeds.npz`, whose
    ids are checked equal to the capped train + test in every run.
  - **Primary index equal-weighted; secondary index without PSF**, with the same rules and its own
    MDD′, read in the plants.
  - **MAE and MoCo**: pending baselines (Kickoff F).

## Settled (user, 2026-09-28, item 2)

- **UNRESOLVED iff the CI straddles the detectable threshold:** lo < MDD < hi, or lo < −MDD < hi. It
  replaces the half-width wording everywhere (index, secondary index with its own MDD′, per variable,
  one seed), at the same place in the precedence.
- **Pull `sky_r` now**, recording the query and the output's hash; rerun all 17 plants once with
  `sky_r`; keep `snr_r` as a declared secondary.

## Plant results — v3 full run (2026-10-02, before the hash)

`sky_r` as variable 7, the straddle rule, the secondary index (without PSF), three untrained seeds. 40,000
capped train and 34,829 test galaxies, 2,000 bootstrap draws, 20 realisations per score plant;
`artifacts/out/distractor_plants.json`, 1,469 s, max RSS 4.1 GB. `sky_r`: 0 missing in train or test.
Precondition met: the median split reproduces N1's panel on J1 to within 0.0001 (PSF, magnitude, SNR,
redshift, size).

| | g−r x | g−r y | i−r x | i−r y | PSF | mag | sky | **NI** (95% CI) | NI′ | N | PR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| J1 (M) | 0.882 | 0.872 | 0.975 | 0.966 | 0.819 | 0.903 | 0.923 | **0.406** (0.404–0.407) | 0.420 | 0.640 | 9.6 |
| J2 (O2) | 0.883 | 0.875 | 0.974 | 0.965 | 0.753 | 0.910 | 0.906 | **0.395** (0.394–0.397) | 0.419 | 0.572 | 13.2 |
| U (seeds 0/1/2) | 0.61–0.62 | 0.62 | 0.60–0.62 | 0.61 | 0.58–0.59 | 0.86–0.87 | 0.84 | **0.177–0.182** | 0.192–0.197 | 0.87–0.89 | 1.5 |

- Seed noise is PSF's (J1 − J2 = 0.066) and now sky's (0.016); every other variable is within 0.007.
  S̄_J = 0.0136; S̄′_J = 0.0048. SNR is reported, no state: 0.869 / 0.874 / 0.839.
- Headroom above the JEPA pair: NI 0.094 (sky 0.077).

| plant | required | got | fires |
|---|---|---|---|
| inject, 50% | MORE | MORE (D̄ +0.100), all 7 MORE; NI′ MORE | yes |
| erase (LEACE) | LESS | LESS (−0.401), all 7 LESS; NI′ LESS | yes |
| swap | SAME | SAME; NI′ SAME | yes |
| split | SEEDS DISAGREE | SEEDS DISAGREE, all 7; NI′ SEEDS DISAGREE | yes |
| one seed | MORE (PROVISIONAL) | MORE (PROVISIONAL); NI′ likewise | yes |
| morphology direction, 20% | Mo MORE | Mo MORE (0.225 / 0.209) | yes |
| **null: J1 permuted** | every variable NEAR CHANCE; NI CI ∋ 0 | all NEAR CHANCE; **NI 0.0025 (0.00016–0.0049)** | **no** |
| score Δ = +0.03 (20) | MORE ≥ 95% | MORE 20; NI′ MORE 20 | yes |
| score Δ = −0.03 (20) | LESS ≥ 95% | LESS 20; NI′ LESS 20 | yes |
| score Δ = 0 (20) | SAME ≥ 95% | SAME 20; **NI′ SAME 18, SEEDS DISAGREE 2** | primary yes; **NI′ no** |
| score Δ = +0.02 (power) | — | MORE 18, LEANS MORE 1, SAME 1 | — |
| shares Δ = 0 (2,000) | false calls ≤ 5% | 1.25% (τ 0.048) | yes |
| shares Δ = +0.20 (2,000) | MORE | MORE 1,995 | yes |
| lean up / down, Δ = ±S̄_J | LEANS MORE / LESS | LEANS MORE / LESS | yes |
| thin: 600 galaxies, Δ = S̄/2 | UNRESOLVED | **UNRESOLVED**: CI 0.0017–0.0178 straddles MDD 0.0143 | **yes** (v2's failure, fixed by the straddle rule, as predicted) |
| insufficient: 400 galaxies | INSUFFICIENT | INSUFFICIENT, all 7; NI′ likewise | yes |
| NI′ lean up / down, Δ = ±S̄′ | LEANS MORE / LESS | LEANS MORE / LESS | yes |
| NI′ thin | UNRESOLVED | UNRESOLVED: CI −0.0030–0.0128 straddles MDD′ 0.0054 | yes |

### Open (D28): the two failures, diagnosed, fixes proposed and not applied

`artifacts/distractor_v3_diag.py` measured both (independent RNG stream; 2,819 s, 2.4 GB).

1. **The null plant is one draw judged on a 95% CI.** The CI is the test-split bootstrap with the
   probe held fixed. v2's draw passed because SNR (0.494) sat below chance and pulled NI in; with sky
   (0.501) in its place the same draw sits 2.05 standard errors out. Over **20 independent permutation
   draws**: all 20 read every variable NEAR CHANCE; the NI CI contains 0 in **18 of 20**, the two misses
   on opposite sides (+0.0027, −0.0030); mean NI +0.0002, sd 0.0013, against a bootstrap se of
   about 0.0012. No bias; the miss is the CI's coverage, slightly short of nominal because the bootstrap
   omits the probe's own fitting noise.
   **Proposed:** the null plant runs 20 permutation draws and requires every variable NEAR CHANCE in
   every draw, and the NI CI to contain 0 in ≥ 16 of 20. At the measured coverage (90%) it fails with
   probability 0.043, at nominal (95%) 0.003; ≥ 17 would fail 13% of the time at 90%.
2. **The secondary index's SEEDS DISAGREE bar is a single difference that cancels.** The bar is
   2 s′_J, s′_J = |NI′_J1 − NI′_J2| = 0.00097: J1 − J2 per variable is −0.0009, −0.0037, +0.0011,
   +0.0001, −0.0071 and +0.0162 over the six, and they nearly cancel. The plant's seed noise on each B
   seed's d_e has sd 0.0022, so opposite signs both beyond 0.0019 is common. Over **200** Δ = 0
   realisations: **9.0%** false SEEDS DISAGREE at 2 s′_J, **0%** at 2 S̄′_J (0.0097). The primary index
   is not exposed (2 s_J = 0.0205 against PSF-dominated noise); it read SAME 20 of 20.
   **Proposed:** SEEDS DISAGREE's margin becomes 2 max(s_J, S̄_J) (the pooled per-variable spread, which
   cannot cancel), for both indices. The split plant (d_e ≈ ±0.25) still reaches it.

## Plant results — v2 full run (2026-09-27), superseded, kept on record

> **v2's run, kept on record; the v3 rerun is queued.** Everything in this section is v1's full run under
> the v1 rules (`snr_r` as variable 7, no straddle rule, no secondary index, one untrained seed); its
> output is `distractor_plants_v2.json`. The v3 rerun of all 17 plants (with the three secondary-only
> plants), with `sky_r` as variable 7 and the straddle rule, is queued on the heavy-job lock
> (2026-10-02 13:17; log `/Volumes/X10 Pro/galaxy-jepa/tmp/distractor_plants_v3.log`). v2's blocker,
> `f0_preconditions.check` step 4, is gone: the check now reads `effect_floor_file` and passes
> (2026-10-02).
>
> **Prediction, stated before the rerun:** on v2's own numbers the thin-n plant's CI [0.0006, 0.0190]
> straddles MDD 0.0128, so it would read UNRESOLVED under the straddle rule. The rerun's RNG stream is
> unchanged, but `sky_r` changes variable 7's scores and the MDD, so the prediction is not a result.

**Full: 40,000 capped train and 34,829 test galaxies, 2,000 bootstrap draws, 20 realisations per score
plant.** `artifacts/out/distractor_plants.json`, 947 s. The DEV subsample run (10,000 + 10,000) that
preceded it is superseded; its rule corrections are recorded below.

J1 = `runs/m/encoder.pt` and J2 = `runs/o2/encoder.pt` (c2 banks). U = O1's untrained bank (seed 0).

**Precondition met.** The median-split protocol reproduces N1's panel on J1 to within 0.0001: PSF 0.8186
(N1 0.8186), magnitude 0.9033 (0.9033), SNR 0.8687 (0.8687), redshift 0.8371 (0.8371), size 0.9061
(0.9061).

### The encoders

| | g−r x | g−r y | i−r x | i−r y | PSF | mag | SNR | **NI** (95% CI) | N | Mo | PR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| J1 (M) | 0.882 | 0.872 | 0.975 | 0.966 | 0.819 | 0.903 | 0.869 | **0.398** (0.397–0.399) | 0.541 | 0 | 9.6 |
| J2 (O2) | 0.883 | 0.875 | 0.974 | 0.965 | 0.753 | 0.910 | 0.874 | **0.391** (0.389–0.392) | 0.480 | 0 | 13.2 |
| U | 0.625 | 0.620 | 0.622 | 0.607 | 0.591 | 0.868 | 0.839 | **0.182** (0.180–0.184) | 0.875 | 0 | 1.5 |

- **PSF carries the JEPA seed noise.** J1 and J2 differ by 0.066 on PSF and by at most 0.007 on every
  other variable. S̄_J = 0.0120.
- The excess over U is 0.216 for J1 and 0.209 for J2.
- U reads the v1 offsets at 0.61–0.63, against 0.50 for the untrained encoder on v2 pixels (criterion 1's
  confound floor in `aligned_comparison.md`). On v1 the misregistration is in the input pixels, so a
  random ViT partly passes it through.
- Reported, no state: redshift 0.837 / 0.839 / 0.769 and size 0.906 / 0.903 / 0.845 (J1 / J2 / U).
- J1's and J2's top-10 tables reproduce criterion 3's (`c13_plants.json`). U's PC1 holds 82% of the
  variance and tracks brightness at |ρ| 0.95.
- Headroom above the JEPA pair: NI 0.102; per variable, i−r x 0.025, i−r y 0.034, magnitude 0.090,
  g−r x 0.117, g−r y 0.125, SNR 0.126, PSF 0.181.
- Realistic power: the bootstrap sd of NI_J1 − NI_J2 is 0.0005, so sd(D̄) ≈ 0.0004, and the minimum
  detectable D̄ ≈ S̄_J + 1.96 sd ≈ **0.013**. The bar is seed noise (mostly PSF's), not galaxy sampling.

### Criterion 1 plants — 16 of 17 fire; thin-n UNRESOLVED does not

Embedding plants go through the whole fit path (`_fit` probes, bootstrap, `aligned_c2.statistic`).

| plant | planted effect | index state (D̄, S̄) | per variable | required |
|---|---|---|---|---|
| inject: seven directions, one per nuisance, 50% of variance | NI → 0.4999 | MORE (+0.106, 0.006) | all 7 MORE | MORE |
| erase: LEACE on the seven binary targets | NI → ~0 | LESS (−0.394, 0.009) | all 7 LESS | LESS |
| swap: B = (J2, J1) | none | SAME (0, 0.012) | all 7 SAME | SAME |
| split: B = (inject J1, erase J2) | opposite directions | SEEDS DISAGREE | all 7 SEEDS DISAGREE | SEEDS DISAGREE |
| one seed: B = (inject J1) | as inject | MORE (PROVISIONAL, one seed) | no state | MORE (PROVISIONAL) |
| null: J1 embeddings permuted against the targets | none | NI 0.002 (−0.001 to 0.004); every variable 0.494–0.507 | — | NEAR CHANCE, CI ∋ 0 |

Score plants use criterion 2's construction (`aligned_c2.shifted`): each variable's AUC moved to
J_e + Δ + ε, ε ~ N(0, σ_v), σ_v = |J1_v − J2_v| / √2 (PSF 0.047; the others 0.0001–0.0050).

| plant | index states (of 20) | per variable: MORE / LESS / SAME / UNRESOLVED | required (index) |
|---|---|---|---|
| Δ = +0.03 | MORE 20 | 90.0% / 0% / 9.3% / 0.7% | MORE |
| Δ = −0.03 | LESS 20 | 0% / 92.1% / 7.9% / 0% | LESS |
| Δ = 0 | SAME 20 | **1.4% / 2.1%** / 96.4% / 0% | SAME |
| Δ = +0.02 (power, no requirement) | MORE 19, SAME 1 | 88.6% / 0% / 10.7% / 0.7% | — |

| deterministic plant | D̄ | S̄ | 95% CI | state | required |
|---|---|---|---|---|---|
| Δ = +S̄_J (0.0120), no ε | +0.0120 | 0.0120 | 0.0119–0.0121 | LEANS MORE | LEANS MORE |
| Δ = −S̄_J, no ε | −0.0120 | 0.0120 | −0.0121 to −0.0119 | LEANS LESS | LEANS LESS |
| 600 test galaxies, Δ = S̄/2, independent score noise | +0.0095 | 0.0191 | 0.0006–0.0190 | **SAME** | UNRESOLVED — **did not fire** |
| 400 test galaxies, Δ = +0.03 | — | — | — | INSUFFICIENT (all 7) | INSUFFICIENT |

- **Open (D28): UNRESOLVED is not shown reachable on the full run.** The thin-n plant's independent score
  noise widens the CI as intended, but it also inflates each planted seed's AUC spread, so the pooled
  bar S̄ rose from 0.012 to 0.019 and the CI fell inside it. The DEV run reached UNRESOLVED only because
  its CI happened to cross −S̄. A fix is a plant change after seeing its result, so it waits for the
  user; nothing is changed here.
- The null false-calls 0 of 20 at family level, and 3.5% per variable (MORE or LESS).
- **Rules revised on the DEV plants, before any baseline exists** (carried over). SEEDS DISAGREE first used
  a 1 s_J margin and the index bar was the index's own seed spread; the DEV Δ = 0 plant then false-called
  (SEEDS DISAGREE 1 of 5, LESS 3 of 20). The corrections are a 2 s_J margin and criterion 2's pooled
  S̄ = mean_v S_v. At that bar +0.02 is detected 19 of 20 on the full run, but the required plant stays
  ±0.03 as drafted.
- PSF alone rarely clears its own bar: its seed spread is about 10 times the others'. A per-variable PSF
  state is weak at two seeds, and this is declared.
- Declared limitation (criterion 2's): score-planted encoders share J's per-galaxy errors, so their CIs
  are narrower than a real baseline's. The embedding plants are deliberately gross, and test the logic,
  not the power.

### Criterion 2 plants — all fire

| plant | N_B | Mo_B | N state | Mo state | required |
|---|---|---|---|---|---|
| inject (as above, 50%) | 0.726 / 0.685 | 0.047 / 0 | MORE | SAME | N MORE |
| erase (LEACE) | 0.251 / 0.128 | 0 / 0 | LESS | SAME | N LESS |
| swap | 0.480 / 0.541 | 0 / 0 | SAME | SAME | both SAME |
| split (inject J1, erase J2) | 0.726 / 0.128 | 0.047 / 0 | SPLIT | SAME | N SPLIT |
| one seed (inject J1) | 0.726 | 0.047 | MORE (PROVISIONAL) | UNRESOLVED (PROVISIONAL) | N MORE (PROVISIONAL) |
| featured-or-disk direction, 20% of variance | 0.361 / 0.382 | 0.225 / 0.209 | LESS | **MORE** | Mo MORE |

- Shares null, through `share_state`: N_B = N_Je + ε, ε ~ N(0, τ), τ = 0.043, 2,000 draws.
  - Δ = 0 reads SAME 1,825, UNRESOLVED 170, SPLIT 4, LESS 1; the false-call rate is **0.25%**
    (required ≤ 5%).
  - Δ = +0.20 reads MORE 1,999 of 2,000.
- Correction, declared (DEV): this plant was first written to require SAME ≥ 95%. The shortfall is
  UNRESOLVED, a named non-call, so the pass criterion is false calls ≤ 5%.
- The PR unit checks and the flag-rule check are criterion 3's (`c13_plants.json`), unchanged.
