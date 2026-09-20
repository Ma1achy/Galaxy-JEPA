# Brief N — the existence bar on a good encoder, and the effect floor's evidence

**Status: N1 and N2 complete. Nothing here is frozen.** `effect_floor_freeze` stays `None`,
`configs/probe.yaml` is untouched, `headline=True` is still refused at load. Every artefact carries
`smoke`. No rungs were assigned and no existence p-values were computed.

Source: `runs/m/encoder.pt` — M's 4-epoch checkpoint (step 101,308, λ=0, D17 schedule, consensus
0.9646), confirmed by device+inode to be the file M wrote. Split identical to J4's: 40,000 train,
34,829 test, `full_tree`, `vote_count_min=1`, C=1.0.

---

## N1 — the controls battery, re-run on M's encoder

### The headline: the bar never moved, so every gain is the encoder

The untrained-encoder null came back **identical to J's on all six features, to four decimals**.
That is not luck and it is the most useful structural fact in this brief: `untrained_encoder_matrix`
builds a fresh random-init ViT from the *model config* and a seed. It never sees the trained
checkpoint. Same architecture, same seed, same galaxies gives the same number whatever encoder is
under test.

**The existence bar is a property of the architecture, not of the run.** J's bar and M's bar are the
same bar. So the margins below moved only because the encoder improved.

| feature | role | J real | **M real** | Δ | binding null | J margin | **M margin** |
|---|---|---|---|---|---|---|---|
| t01 featured-or-disk | anchor, clean binary | 0.8365 | **0.8845** [0.8809, 0.8882] | +0.0480 | 0.7908 | +0.0457 | **+0.0937** |
| t02 edge-on yes | clean binary | 0.7320 | **0.7995** [0.7933, 0.8059] | +0.0676 | 0.6368 | +0.0952 | **+0.1627** |
| t10 arms loose | graded (3/3) | 0.6098 | **0.6538** [0.6445, 0.6633] | +0.0440 | 0.5645 | +0.0453 | **+0.0893** |
| t10 arms tight | graded (1/3) | 0.5740 | **0.5889** [0.5814, 0.5964] | +0.0149 | 0.5474 | +0.0267 | **+0.0415** |
| t09 bulge boxy | deep + confused | 0.5534 | **0.5847** [0.5687, 0.6008] | +0.0313 | 0.5359 | +0.0176 | **+0.0489** |
| t10 arms medium | graded (2/3) | 0.5161 | **0.5226** [0.5144, 0.5307] | +0.0065 | 0.5160 | +0.0000 | **+0.0065** |

Six for six, and the gains are **largest at the easy end**: the two clean binaries gained +0.048 and
+0.068, the hardest feature +0.0065 with intervals that overlap almost entirely. The recipe fix
bought discrimination where there was already discrimination to buy.

The graded axis reads **0.5889 → 0.5226 → 0.6538** (tight → medium → loose). The middle category is
still the minimum, which is what a binary probe reading a graded quantity should do — predicted
before J's numbers were read, and now held on two independent encoders.

### The controls behave

| control | min | median | max |
|---|---|---|---|
| shuffled labels (max/feature) | 0.5143 | 0.5272 | 0.5717 |
| random embeddings (max/feature) | 0.5091 | 0.5101 | 0.5244 |
| noise images | 0.4906 | 0.5000 | 0.5048 |
| **untrained encoder** | 0.5160 | 0.5559 | **0.7908** |
| sky-noise label (DIAGNOSTIC, D19) | 0.8662 | 0.8687 | 0.8692 |

Shuffled-label mean across features **0.5005** against a construction chance of 0.5 — the
calibration is honest. **D19's fix works as intended on a good encoder:** the sky-noise diagnostic
sits at 0.8662–0.8692 and would, under the pre-K1 bar, have failed featured-ness at 0.8845 by a hair
and every other feature comfortably. It is out of the bar, and the bar is now the untrained encoder.
The diagnostic is still bit-identical to `nuisance_aucs["snr"]`, as K1 found — one measurement, two
entries.

**The null is still a point mass.** The untrained singleton exceeds the shuffled maximum on all six
features (0.5160 vs 0.5143 at the narrowest), so `existence_null_samples` — a per-draw max — has zero
variance and `existence_pvalue` can return only `1/(n+1)` or `1.0`. The existence test reduces, on
this apparatus, to exactly `real_auc > untrained_encoder_auc`. Register item 8 stands.

### Selectivity

+0.3852 (featured), +0.2988 (edge-on), +0.1537 (loose), +0.0871 (tight), +0.0840 (boxy), +0.0220
(medium). All above `selectivity_floor = 0.10` except tight, boxy and medium. Up on J throughout.

### The nuisance panel — the serious finding, and it has NOT gone away

| | magnitude | size | SNR | redshift | PSF |
|---|---|---|---|---|---|
| J's encoder | 0.8733 | 0.8501 | 0.8373 | 0.7918 | 0.5813 |
| **M's encoder** | **0.9033** | **0.9061** | **0.8687** | **0.8371** | **0.8186** |

**Training longer at λ=0 made the representation encode observing conditions *more* strongly, not
less.** Every nuisance rose, and by more than most morphology features did. **PSF is the striking
case: 0.5813 → 0.8186.** On J's encoder PSF was the one nuisance carrying almost nothing; it now
carries as much as redshift.

The premise being tested was 3D-ii's, that matched evaluation is *targeted* — "fires only for flagged
features". State plainly: **it has changed, but it has not been rescued.**

* Featured-ness (0.8845) now beats SNR, redshift and PSF. On J's encoder it beat none of them.
* But **magnitude (0.9033) and size (0.9061) still beat featured-ness**, and *all five* nuisances
  still beat *all five* of the other morphology features.

So matched evaluation still fires for every feature on two to five nuisances each. It remains
load-bearing for Paper 1, not a bounded contingency, and `matching.py` stays on the critical path.
The direction of travel is the worrying part: the fix that improved morphology improved the
confounds faster.

### What N1 did not do

No rungs, no verdicts, no existence p-values, no floor applied. `assert_null_resolution` would refuse
at this draw budget anyway — Scheme 1's BY family of 37 needs ≥3,109 draws and this ran at 50–200.
These draws are characterisation for a floor *proposal*, which is what they are honest for.

---

## N2 — the effect floor

### The stability measurement, and the rule that was fixed before it

J5's third objection — that an absolute AUC cannot encode a per-feature null — points toward a
**margin** form. But the quantity a margin form would consume is one scalar from **one draw of a
random network**, and its variability had never been measured. So N1 measured it at three seeds
(`cfg.seed = 0` primary, plus 1 and 2), and N2's decision rule was written down before the numbers
existed:

> **STABLE** — `range_f < margin_f` for every feature AND `max range_f ≤ 0.010` → choose on the merits.
> **MARGINAL** — `max range_f` in 0.010–0.030, or a near-zero margin is seed-determined → recommend the **absolute** form; it does not inherit the instability.
> **UNSTABLE** — `max range_f ≥ 0.030`, or a substantial margin is seed-determined → recommend **no floor**; the control is broken and averaging seeds would hide it.

| feature | seed 0 | seed 1 | seed 2 | range | margin | range < margin? |
|---|---|---|---|---|---|---|
| t01 featured | 0.7908 | 0.7933 | 0.7916 | **0.0026** | +0.0937 | yes |
| t02 edge-on | 0.6368 | 0.6334 | 0.6241 | 0.0127 | +0.1627 | yes |
| t10 tight | 0.5474 | 0.5406 | 0.5453 | 0.0067 | +0.0415 | yes |
| t10 medium | 0.5160 | 0.5220 | 0.5197 | 0.0059 | +0.0065 | **yes, by 0.0006** |
| t10 loose | 0.5645 | 0.5616 | 0.5694 | 0.0078 | +0.0893 | yes |
| t09 boxy | 0.5359 | 0.5253 | 0.5149 | **0.0209** | +0.0489 | yes |

**PRE-REGISTERED VERDICT: MARGINAL** — `max range = 0.0209` at t09 boxy, inside the 0.010–0.030 band.

No feature is strictly seed-determined, but **t10 medium clears its own bar by 0.0006**. Under seed 1
its margin is +0.0006 rather than +0.0065; its existence verdict is decided by which random network
was drawn. That feature is a coin flip whichever form is chosen, and no floor fixes it.

### Both forms, and why the margin form fails on this evidence

The per-feature ceiling is **exactly** the supremum of `existence_null_samples` — verified against the
production function over 200 random cases, not asserted. So `margin_f` is precisely the quantity the
existence gate already thresholds at zero, and **form (b) is a tightening of the existence test, not
a second and independent question.** That is the property design 3B explicitly disclaims: the floor
"is a pre-registered constant, acceptable *because it no longer does the existence work*".

Three further findings against form (b):

1. **It reorders the catalogue by the performance of a random network.** Bulge-boxy has a *lower*
   real AUC than arms-tight (0.5847 vs 0.5889) but a *higher* margin (+0.0489 vs +0.0415). Any
   margin between those two values admits boxy and excludes tight — ranking one concept above
   another because a random ViT happened to do worse on it. At margin 0.10 the same effect excludes
   **featured-ness (0.8845)**, the anchor and the strongest signal in the catalogue, while admitting
   edge-on (0.7995).
2. **Its verdicts move with the seed.** Re-classifying under each seed's null, the admitted set
   changes at margin 0.05 (boxy flips) and at 0.09 (loose flips). Form (a)'s verdicts cannot move
   with the seed at all, because it never reads the null.
3. **It is not a one-line change.** `effect_floor` has five consumers. A margin is definable at
   `nulls.py:264` and `gates.py:62`, but at `ladder.py:121` (entangled pair), `:161` (competitive
   nuisance) and `:248` (MLP decode) there is no defined "the feature's own binding null" to take a
   margin over. Form (b) would either need three further definitions or would silently leave three
   thresholds meaning something other than the one the spec names.

### RECOMMENDATION — form (a), absolute, value 0.7267

Following the pre-registered rule. The value is the **midpoint of the widest gap in the real spread**
(0.6538 → 0.7995), and it is recommended for a robustness reason rather than an aesthetic one: it is
the point **furthest from any feature's flip point** (0.0729 either way) of anything on the sweep. For contrast,
the current placeholder 0.6500 sits 0.0038 from a flip and 0.7908 sits 0.0087 from one — both are
coincidences rather than thresholds.

**The band matters more than the point.** Every value in **(0.6538, 0.7995]** produces the
*identical* partition of these six features — a band 0.146 wide, bounded by arms-loose below and
edge-on above. So the recommendation is really "somewhere in that band", and the exact number is not
load-bearing among the features probed. 0.7267 is simply its centre.

**What it admits:** featured-ness (0.8845), edge-on (0.7995).
**What it excludes:** arms-loose (0.6538), arms-tight (0.5889), bulge-boxy (0.5847), arms-medium
(0.5226).

Note what that partition is *not*: it is not "the features that exist". All six clear their own
binding null, so all six would pass existence. The floor separates two features whose directions are
clean from four that are real but marginal — which is the job 3B gives it.

**Candidates, for the proposal to be argued against rather than asserted:**

| candidate | value | admits | nearest flip | comment |
|---|---|---|---|---|
| **widest-gap midpoint** | **0.7267** | featured, edge-on | **0.0729** | most robust point on the sweep; location is a fact about these six (J5's objection 4) |
| pooled null ceiling | 0.7908 | featured, edge-on | 0.0087 | J5's preferred candidate; on M's encoder it is fragile, and it is a null quantity doing floor duty |
| median of real spread | 0.6214 | featured, edge-on, loose | 0.0324 | descriptive, not principled |
| current placeholder | 0.6500 | featured, edge-on, loose | 0.0038 | ungrounded, and a coincidence |

### What is NOT resolved, and must travel with the value

**J5's fourth objection stands: n = 6 cannot locate a threshold in a spread.** The widest gap is a
fact about which six features were chosen, not about the catalogue of 37. The band argument softens
this — a wrong point inside (0.6538, 0.7995] costs nothing here — but it does not remove it. The
recommendation is a proposal against six features, and the catalogue it will govern has 37.

**No D-series entry is drafted.** One would have been required only if form (b) were recommended,
since that changes a mechanism the spec records as settled. Form (a) is the mechanism already in the
code, so this is a value proposal and nothing more.

**PROPOSE, DO NOT FREEZE.** `effect_floor_freeze` stays `None`. The value is Malachy's.

---

## Provenance

`artifacts/out/n1_spread_controls.json` (checkpoint `runs/m/encoder.pt`, `untrained_seeds: [0,1,2]`),
`artifacts/out/n1_controls.log`, `artifacts/out/n2_floor_evidence.txt`.
Drivers: `artifacts/n1_controls.py` → `artifacts/j4_spread_controls.py` (parameterised, J's defaults
unchanged), `artifacts/n2_floor_evidence.py` → `artifacts/j5_floor_evidence.py` (input parameterised,
tables unchanged). J's record at `artifacts/out/j4_spread_controls.json` is untouched.
