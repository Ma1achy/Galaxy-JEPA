"""M and O2 (v1 corpus) against A1–A3 (v2, band-aligned), seed-matched pairs: the states, the plants, the
calibration, and (after the hash) the scoring (`docs/m_comparison/prereg.md`, draft v1).

One decision path. `decide()` is the decision layer, vectorised over a leading draws axis; `score()` builds
its inputs from loaded t1 record dicts and banks, and is what the plants and the real scoring call. The
pairs (user, 17 Nov): (M, A1) on train seed 0 and (O2, A2) on train seed 1, identical init within a pair,
splits on cfg.seed 0 for all; A3 (v2, train seed 2) is unpaired and estimates v2 seed noise.

    calibrate   decision-layer Monte Carlo on synthetic z: k by the fixed rule, null rates with and
                without u_seed -> docs/m_comparison/calibration.json
    plants      synthetic t1 records and banks (no real value), written to a temp dir, through
                load_inputs() -> score() -> docs/m_comparison/plants.json
    score --go  (after the hash only) the real records; refuses without the hash

Plants and calibration use **no real result**: every AUC, SE, bank value and count is a synthetic draw.

Investigation code: terse, excluded from lint/CI.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.stats import binom
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))

# --- the pre-registered constants (prereg.md §C–§F) -------------------------------------------
RUNS = ("M", "A1", "O2", "A2", "A3")
SIDE = {"M": "v1", "O2": "v1", "A1": "v2", "A2": "v2", "A3": "v2"}
PAIRS = (("M", "A1"), ("O2", "A2"))  # (v1, v2) per train seed 0, 1; identical init within a pair
UNPAIRED = "A3"
POPS = ("full", "conditional")
PRIMARY_POP = "full"  # the only confirmatory population (user, 17 Nov)
N_FEATURES = 37
K_BANK = 30  # D23's untrained bank, per side
K_BAR_D24 = 3  # the record's k_bar: matching.RETENTION_SEEDS, the D24 draws, NOT the bank's K
FLOOR = 0.7267
ALPHA_BY = 0.05  # the ladder's own existence family (probe.yaml)
ALPHA_AGG = 0.025  # per direction, one-sided
VOID_ALLOW = 3  # of 37 cells per population; more -> that population's aggregate is VOID
GUARD_BOUND = 0.025  # the k rule: "both pairs beyond F_c, same direction" <= this per cell per direction
K_GRID = np.round(np.arange(0.05, 5.0001, 0.05), 2)
K_GUARD: float | None = 1.1  # from `calibrate` (calibration.json "k"); plants assert they agree

ROW_KEYS = ("feature", "rung", "mechanism", "auc", "n_test", "positives_test", "real_se",
            "resolvable_margin", "underpowered", "exceeds_null", "clean", "competitive")
RUNGS = ("R1", "R2", "R4")  # R3 is unreachable (no untrained-MLP bar, D25): a recorded R3 is VOID

PREREG = REPO / "docs" / "m_comparison" / "prereg.md"
PREREG_SHA1: str | None = None  # set at the hash; until then load_inputs refuses every real record

LADDER = REPO / "runs" / "ladder_v2"
RECORDS = {r: LADDER / d / "t1_ladder.json" for r, d in
           (("M", "m"), ("O2", "o2"), ("A1", "a1"), ("A2", "a2"), ("A3", "a3"))}
BANKS = {"v1": LADDER / "shared_v1" / "p1_untrained_bank.json",
         "v2": LADDER / "shared" / "p1_untrained_bank.json"}
# P1 pins. A1–A3 and the v2 bank from docs/ladder_v2/outputs.sha1 (hash lines only, cross-checked at
# scoring); M, O2 and the v1 bank from docs/ladder_v2_m/outputs.sha1, read at scoring only after its own
# SHA-1 equals V1_OUTPUTS_SHA1 (recorded outside the hash when the runner finishes; fingerprint only).
PINS: dict[str, str | None] = {
    "A1": "705154a1605b82afb6833bf340e68b102f3a9b5c",
    "A2": "aee7d218d8c10ec3f0344cf41575924b711d65c4",
    "A3": "2c30f8f282035e8d3baf600b7fadf539d720248b",
    "bank_v2": "7a8efc748b4a65391abd4fc98c35214e5dc53c8f",
    "M": None, "O2": None, "bank_v1": None,  # filled from V1_OUTPUTS by v1_pins()
}
A_OUTPUTS = REPO / "docs" / "ladder_v2" / "outputs.sha1"
V1_OUTPUTS = REPO / "docs" / "ladder_v2_m" / "outputs.sha1"
V1_OUTPUTS_SHA1: str | None = None  # outside the hash, when the M/O2 runner finishes
CHECKPOINTS = {"M": "runs/m/encoder.pt", "O2": "runs/o2/encoder.pt", "A1": "runs/m_v2/encoder.pt",
               "A2": "runs/m_v2_s1/encoder.pt", "A3": "runs/m_v2_s2/encoder.pt"}
# P4: the code each ladder ran at is what its manifest pins per entry (the runner checks the pins at
# launch). The manifests are pinned here by SHA-1 at their committing commits.
MANIFESTS = {"A": ("fcea3b2", "docs/ladder_v2/manifest.yaml", "6c2f462009ce7f60a81b85e9eca02b8b2662732c"),
             "V1": ("17945d9", "docs/ladder_v2_m/manifest.yaml", "b4dbd5171c8c96cbe88e83cfebeedb921d7c4ae4")}
P4_ENTRIES = ("E5.t1_ladder", "E5.s2_verify", "E5.p2_ladder", "E4.p1_untrained_bank", "E7.r_bank",
              "E7.r0", "E3.o1_matched")
P4_ENCODERS = {"A": {"a1", "a2", "a3", "shared"}, "V1": {"m", "o2", "shared_v1"}}
# The only code pins allowed to differ, each as its exact (A, V1) SHA-1 pair, classified before the hash:
P4_DECLARED = {
    # 17945d9 `_minus_nine`: v1's split minus NINE whatever the corpus holds; on v2 the nine are absent,
    # so A's set is unchanged (its test); on v1 it makes the split identical to A's
    "artifacts/j4_spread_controls.py": ("9598302bdba7e3376c35cdfff4dfa0ddbe4e69b9",
                                        "de65785469401de2a480dfdea227a5f3f3ddd14a"),
    # fcea3b2..17945d9 touches src/galaxy_jepa only in harness.py and objectives/moco.py: MoCo's
    # sky_sigma default and docstrings. Training-side; not on the probing path of a JEPA encoder: inert
    "src/galaxy_jepa/**.py": ("9400ddf417ab8690d3148c7500125f3eb7bbf4cc",
                              "afc8ce9bc026f99470c3da0f8f02035a356046e8"),
}
# §I: M's original record, reproduction check only (does not gate); pinned by `shasum` (digest only)
ORIGINAL_M = REPO / "artifacts" / "out" / "t1_ladder.json"
ORIGINAL_M_SHA1 = "d71ce4ee858cc35f1531107a1db62b6f6133228f"

REAL_ROOTS = tuple({p.resolve() for p in (REPO / "runs", REPO / "artifacts" / "out",
                                          Path("/Volumes/X10 Pro/galaxy-jepa/runs"),
                                          Path("/Volumes/X10 Pro/galaxy-jepa/artifacts_out"))})
OUT = REPO / "docs" / "m_comparison"
CALIBRATION = OUT / "calibration.json"

STATES = ("VOID", "THIN", "PAIRS-OPPOSE", "ONE-PAIR", "A-SPLIT", "A-LOWER", "A-HIGHER", "NEAR-BAR",
          "SHIFTED", "SEED-SPLIT", "AGREES")
S = {s: i for i, s in enumerate(STATES)}
AGG_STATES = ("VOID", "INSUFFICIENT", "BOTH WAYS", "CARRIED", "MASKED", "WITHIN-CHANCE")
G_STATES = ("G-NA", "G-OPPOSE", "G-ONE-PAIR", "G-SPLIT", "G-LOWER", "G-HIGHER", "G-SEED-SPLIT", "G-AGREES")


# --- the hash ---------------------------------------------------------------------------------
def prereg_section(path: Path = PREREG) -> str:
    """From '## Pre-registration' to the next level-2 heading, trailing newline (as the house rule)."""
    lines = Path(path).read_text().splitlines()
    a = lines.index("## Pre-registration")
    b = next((k for k in range(a + 1, len(lines)) if lines[k].startswith("## ")), len(lines))
    return "\n".join(lines[a:b]).rstrip() + "\n"


def assert_hashed() -> None:
    if PREREG_SHA1 is None:
        raise SystemExit("m_comparison: the pre-registration is not hashed; no real record is opened")
    got = hashlib.sha1(prereg_section().encode()).hexdigest()
    if got != PREREG_SHA1:
        raise SystemExit(f"m_comparison: pre-registration changed since the hash ({got} != {PREREG_SHA1})")


def _sha1(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_real(p: Path) -> bool:
    rp = Path(p).resolve()
    return any(rp == r or r in rp.parents for r in REAL_ROOTS)


def load_inputs(records: dict, banks: dict, pins: dict) -> tuple[dict, dict, list[str]]:
    """P1 and the load. A path under a real root (runs/, artifacts/out, the SSD's runs and artifacts_out)
    is refused before a byte of it is read, unless the pre-registration is hashed. A missing or
    mismatched pin is a P1 failure (everything VOID), returned rather than raised."""
    paths = [Path(p) for p in (*records.values(), *banks.values())]
    if any(is_real(p) for p in paths):
        assert_hashed()
    named = [*records.items(), *((f"bank_{s}", p) for s, p in banks.items())]
    p1 = [n for n, p in named if pins.get(n) is None or _sha1(Path(p)) != pins[n]]
    recs = {r: json.loads(Path(p).read_text()) for r, p in records.items()}
    bk = {s: json.loads(Path(p).read_text()) for s, p in banks.items()}
    return recs, bk, p1


def load_pinned(path: Path, pin: str) -> dict | None:
    """One pinned JSON (M's original record), under the same refusal as load_inputs; None on a pin mismatch."""
    if is_real(Path(path)):
        assert_hashed()
    return json.loads(Path(path).read_text()) if _sha1(Path(path)) == pin else None


def outputs_pins(path: Path, want: dict[str, Path]) -> dict[str, str | None]:
    """The runner's outputs.sha1 (`<sha1>  <path>` lines) -> the pin of each wanted file, matched by path."""
    got: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            got[str(Path(parts[1].strip()))] = parts[0]
    full = lambda p: str(Path(p) if Path(p).is_absolute() else REPO / p)  # noqa: E731
    return {n: got.get(full(p)) or got.get(str(Path(p).relative_to(REPO)) if Path(p).is_absolute() else str(p))
            for n, p in want.items()}


def reproduction(original: dict | None, relad: dict) -> dict:
    """§I: M's original t1 record against the re-laddered M, rung by rung. Reported after scoring; no gate."""
    if original is None:
        return {"result": "NOT RUN", "why": "original record does not match its pin"}
    diff, why = 0, None
    for pop in POPS:
        a = {r["feature"]: r.get("rung") for r in original.get(pop, [])}
        b = {r["feature"]: r.get("rung") for r in relad.get(pop, [])}
        if set(a) != set(b):
            why = f"{pop}: answer sets differ"
        diff += sum(a[f] != b[f] for f in set(a) & set(b))
    if why:
        return {"result": "FAIL", "cells_rung_differs": diff, "why": why}
    return {"result": "PASS"} if diff == 0 else {"result": "FAIL", "cells_rung_differs": diff}


# --- the statistics ---------------------------------------------------------------------------
def drift(d1: np.ndarray, d2: np.ndarray, ok: np.ndarray, s2: np.ndarray) -> dict:
    """§E: the measured run-specific drift, mean over non-VOID cells of Δ1 − Δ2 with its SE, and the same
    mean in run-noise units (|mean| / √(σ̂²/2)), the scale the ω lookup reads. Vectorised like decide()."""
    x = np.where(ok, d1 - d2, 0.0)
    n = np.maximum(ok.sum(-1), 1)
    m = x.sum(-1) / n
    var = np.where(ok, (d1 - d2 - m[..., None]) ** 2, 0.0).sum(-1) / np.maximum(n - 1, 1)
    return {"mean": m, "se": np.sqrt(var / n), "scaled": np.abs(m) / np.sqrt(np.maximum(s2 / 2.0, 1e-12))}



def by_reject(p: np.ndarray, alpha: float = ALPHA_BY, m: int = N_FEATURES) -> np.ndarray:
    """BY step-up along the last axis: `nulls.family_significant` vectorised (a test asserts identity)."""
    p = np.asarray(p, dtype=np.float64)
    n = p.shape[-1]
    order = np.argsort(p, axis=-1, kind="stable")
    ps = np.take_along_axis(p, order, -1)
    thr = np.arange(1, n + 1) / (m * float(np.sum(1.0 / np.arange(1, m + 1)))) * alpha
    ok = ps <= thr
    kmax = np.where(ok.any(-1), n - np.argmax(ok[..., ::-1], -1), 0)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(n), order.shape).copy(), -1)
    return rank < kmax[..., None]


def guard(d1: np.ndarray, d2: np.ndarray, nu: np.ndarray, b2: np.ndarray, ok: np.ndarray):
    """σ̂²: the pooled within-pair noise, from the two pairs' disagreement Δ1 − Δ2, in which u_seed, the
    corpus effect and the bank errors all cancel. Uncentred, so a pair-level drift inflates it. The guard's
    unit adds b_c², the two banks' mean error, which is common to both pairs and does not cancel."""
    q = np.where(ok, (d1 - d2) ** 2 / (2.0 * nu), 0.0)
    s2 = np.asarray(q.sum(-1) / np.maximum(ok.sum(-1), 1))
    return s2, np.sqrt(nu * s2[..., None] + b2)


def classify(E: dict, d1: np.ndarray, d2: np.ndarray, unit: np.ndarray, k: float, void, thin):
    """The per-cell existence states, first match wins (§D). Returns (code, direction ∈ {-1, 0, +1})."""
    e = {r: np.asarray(E[r]).astype(np.int8) for r in RUNS}
    c1, c2 = e["A1"] - e["M"], e["A2"] - e["O2"]
    F = k * unit
    down, up = (d1 < -F) & (d2 < -F), (d1 > F) & (d2 > F)
    same = (c1 == c2) & (c1 != 0)
    a3ok = e["A3"] == e["A1"]
    still = (c1 == 0) & (c2 == 0)
    conds = [void, thin, c1 * c2 == -1, (c1 != 0) ^ (c2 != 0), same & ~a3ok,
             same & (c1 < 0) & down, same & (c1 > 0) & up, same, still & (down | up),
             still & (e["M"] != e["O2"])]
    code = np.select(conds, list(range(len(conds))), default=S["AGREES"]).astype(np.int8)
    dirn = np.select([code == S["ONE-PAIR"], np.isin(code, (S["A-SPLIT"], S["NEAR-BAR"])),
                      code == S["A-LOWER"], code == S["A-HIGHER"], code == S["SHIFTED"]],
                     [np.sign(c1 + c2), np.sign(c1), -1, 1, np.where(up, 1, -1)], 0).astype(np.int8)
    return code, dirn


def aggregate(E: dict, void, thin):
    """§E: the paired sign-flip binomial. O = scoreable cells where both pairs flip E; under within-pair
    exchangeability the four flip patterns are equally likely, so #both-down ~ Binomial(|O|, 1/4)."""
    e = {r: np.asarray(E[r]).astype(np.int8) for r in RUNS}
    c1, c2 = e["A1"] - e["M"], e["A2"] - e["O2"]
    o = ~void & ~thin & (c1 != 0) & (c2 != 0)
    n = o.sum(-1)
    md, mu = (o & (c1 < 0) & (c2 < 0)).sum(-1), (o & (c1 > 0) & (c2 > 0)).sum(-1)
    pd, pu = binom.sf(md - 1, n, 0.25), binom.sf(mu - 1, n, 0.25)
    nv = np.asarray(void).sum(-1)
    hd, hu = pd <= ALPHA_AGG, pu <= ALPHA_AGG
    st = np.select([nv > VOID_ALLOW, n < 3, hd & hu, hd, hu], [0, 1, 2, 3, 4], 5)
    return {"n": n, "m_down": md, "m_up": mu, "p_down": pd, "p_up": pu, "n_void": nv, "state": st,
            "n_opposite": (o & (c1 * c2 == -1)).sum(-1)}


def reach(n: int) -> int | None:
    """The smallest both-same-direction count that would fire at this n (None: unreachable)."""
    return next((m for m in range(n + 1) if binom.sf(m - 1, n, 0.25) <= ALPHA_AGG), None)


def decide(E: dict, z: dict, nu, b2, void, thin, k: float) -> dict:
    """The decision layer, shared by score() (shape (n,)) and calibrate() (shape (R, n))."""
    d1, d2 = z["A1"] - z["M"], z["A2"] - z["O2"]
    s2, unit = guard(d1, d2, nu, b2, ~void)
    code, dirn = classify(E, d1, d2, unit, k, void, thin)
    return {"d1": d1, "d2": d2, "s2": s2, "unit": unit, "code": code, "dir": dirn,
            "agg": aggregate(E, void, thin)}


def grade(G: dict, eligible: np.ndarray) -> np.ndarray:
    """§D grade tier (R1 = 1, R2 = 0), unguarded, exploratory: the same pattern logic with no magnitude."""
    g = {r: np.asarray(G[r]).astype(np.int8) for r in RUNS}
    c1, c2 = g["A1"] - g["M"], g["A2"] - g["O2"]
    same = (c1 == c2) & (c1 != 0)
    still = (c1 == 0) & (c2 == 0)
    return np.select([~eligible, c1 * c2 == -1, (c1 != 0) ^ (c2 != 0), same & (g["A3"] != g["A1"]),
                      same & (c1 < 0), same & (c1 > 0), still & (g["M"] != g["O2"])],
                     list(range(7)), 7).astype(np.int8)


# --- preconditions ----------------------------------------------------------------------------
def bank_matrix(bank: dict, feats: list[str]) -> np.ndarray:
    return np.array([[float(s[f]) for f in feats] for s in bank["seeds"].values()])


def p2_record(records: dict, banks: dict) -> list[str]:
    """Record-level P2: populations, the feature set, k_bar, checkpoints, the banks' K, keys and cover."""
    bad: list[str] = []
    feats = None
    for r in RUNS:
        rec = records.get(r)
        if not isinstance(rec, dict) or any(p not in rec for p in POPS):
            bad.append(f"{r}: populations missing")
            continue
        if rec.get("k_bar") != K_BAR_D24:
            bad.append(f"{r}: k_bar {rec.get('k_bar')!r} != {K_BAR_D24}")
        if not str(rec.get("checkpoint", "")).endswith(CHECKPOINTS[r]):
            bad.append(f"{r}: checkpoint does not name {CHECKPOINTS[r]}")
        for pop in POPS:
            fs = [row.get("feature") for row in rec[pop]]
            if len(fs) != N_FEATURES or len(set(fs)) != N_FEATURES:
                bad.append(f"{r}/{pop}: not {N_FEATURES} distinct answers")
            feats = feats or set(fs)
            if set(fs) != feats:
                bad.append(f"{r}/{pop}: answer set differs")
    for s in ("v1", "v2"):
        b = banks.get(s, {})
        if len(b.get("seeds", {})) != K_BANK:
            bad.append(f"bank {s}: K != {K_BANK}")
        elif feats and any(not feats <= set(v) for v in b["seeds"].values()):
            bad.append(f"bank {s}: a seed lacks an answer")
    b1, b2 = banks.get("v1", {}), banks.get("v2", {})
    if b1.get("model_key") != b2.get("model_key") or b1.get("split_key") != b2.get("split_key"):
        bad.append("banks: model or split key differs (the identical split and architecture are required)")
    if b1.get("pixels_key") is None or b1.get("pixels_key") == b2.get("pixels_key"):
        bad.append("banks: the v1 bank's pixels key is missing or equals v2's")
    return bad


def p2_row(row: dict) -> bool:
    if any(k not in row for k in ROW_KEYS) or row["rung"] not in RUNGS:
        return False
    if row.get("k_bar") is not None and row["k_bar"] != K_BAR_D24:
        return False
    a, se = row["auc"], row["real_se"]
    return a is not None and se is not None and math.isfinite(a) and math.isfinite(se) and se > 0


def p5_record(rec: dict, bank: dict) -> list[str]:
    """P5: D23 recomputed through the package (nulls.untrained_z_pvalue + family_significant), the rung's
    consistency with it, and the power annotation (resolvable_margin, is_underpowered)."""
    from galaxy_jepa.probing import nulls

    bad = []
    for pop in POPS:
        rows = rec[pop]
        feats = [r["feature"] for r in rows]
        mat = bank_matrix(bank, feats)
        p = {f: nulls.untrained_z_pvalue(float("nan") if r["auc"] is None else r["auc"],
                                         r["real_se"] or 0.0, mat[:, j])
             for j, (f, r) in enumerate(zip(feats, rows, strict=True))}
        sig = nulls.family_significant(p, alpha=ALPHA_BY, method="benjamini_yekutieli", n_tests=N_FEATURES)
        for j, r in enumerate(rows):
            if not p2_row(r):
                continue  # a malformed row is that cell's VOID (P2), not the record's; its p stays in the family
            f, ex = r["feature"], bool(r["exceeds_null"])
            margin = nulls.resolvable_margin(r["real_se"] or 0.0, sd_bar=float(np.std(mat[:, j], ddof=1)),
                                             alpha=ALPHA_BY, method="benjamini_yekutieli",
                                             n_tests=N_FEATURES, df=K_BANK - 1)
            clean_ok = bool(r["clean"]) == (ex and r["auc"] is not None and r["auc"] >= FLOOR)
            if (ex != sig[f] or (r["rung"] == "R4") == ex or (r["rung"] == "R1" and not r["clean"])
                    or not clean_ok or abs(margin - float(r["resolvable_margin"])) > 1e-9
                    or nulls.is_underpowered(margin) != bool(r["underpowered"])):
                bad.append(f"{pop}/{f}")
    return bad


def manifest_code_pins(text: str, encoders: set[str]) -> dict[str, set[str]]:
    """Every code pin (artifacts/, src/, configs/) of the comparison's entries (P4_ENTRIES) in a manifest."""
    import yaml

    out: dict[str, set[str]] = {}
    for e in yaml.safe_load(text)["entries"]:
        if e["id"].startswith(P4_ENTRIES) and e.get("encoder") in encoders:
            for k, v in (e.get("pins") or {}).items():
                if k.startswith(("artifacts/", "src/", "configs/")):
                    out.setdefault(k, set()).add(v)
    return out


def p4_compare(a: dict[str, set[str]], v: dict[str, set[str]]) -> list[str]:
    """P4 on the two manifests' code pins: one SHA-1 per file per side, the same file set, and equal
    SHA-1s except the declared pairs (P4_DECLARED), which must match exactly."""
    bad = [f"P4: {f} pinned only on {s}" for s, x, y in (("A", a, v), ("V1", v, a)) for f in sorted(set(x) - set(y))]
    for f in sorted(set(a) & set(v)):
        if len(a[f]) != 1 or len(v[f]) != 1:
            bad.append(f"P4: {f} has more than one pin within a side")
        elif a[f] != v[f] and (next(iter(a[f])), next(iter(v[f]))) != P4_DECLARED.get(f):
            bad.append(f"P4: {f} differs and is not a declared, classified difference")
    return bad


def p4_status() -> list[str]:
    """P4 from the manifests at their pinned commits (git show), each checked against its SHA-1."""
    texts = {}
    for side, (commit, path, sha) in MANIFESTS.items():
        t = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=REPO, capture_output=True, text=True,
                           check=True).stdout
        if hashlib.sha1(t.encode()).hexdigest() != sha:
            return [f"P4: {path} at {commit} does not match its pin"]
        texts[side] = t
    return p4_compare(manifest_code_pins(texts["A"], P4_ENCODERS["A"]),
                      manifest_code_pins(texts["V1"], P4_ENCODERS["V1"]))


# --- score ------------------------------------------------------------------------------------
def all_void(why: list[str]) -> dict:
    return {"void": why, "populations": {p: {"aggregate": {"state": "VOID"}, "cells": "all VOID"}
                                         for p in POPS}}


def observed_like(rows: list[dict], s2: float, tau_ratio: float, flip: float) -> dict:
    """The calibration row nearest the observed noise (§F), chosen mechanically after P1–P6 pass:
    nearest σ (log scale), then nearest τ/σ, then nearest flip rate; a tie takes the higher false-call rate."""
    base = [r for r in rows if r["kind"] == "base"]
    sig = math.sqrt(max(s2, 1e-12) / 2.0)
    best_s = min({r["sigma"] for r in base}, key=lambda v: abs(math.log(v / sig)))
    cand = [r for r in base if r["sigma"] == best_s]
    best_t = min({r["tau_ratio"] for r in cand}, key=lambda v: abs(v - tau_ratio))
    cand = [r for r in cand if r["tau_ratio"] == best_t]
    return min(cand, key=lambda r: (round(abs(r["flip_rate"] - flip), 3),
                                    -(r["rate"]["A-LOWER"] + r["rate"]["A-HIGHER"])))


THRESHOLDS = (0.025, 0.05)
A_BOUND_LEVEL = 0.95  # one-sided
A_BOUND_DF = 2  # three v2 runs
CURVE_OF = {"CARRIED": ("CARRIED",), "MASKED": ("MASKED",), "BOTH WAYS": ("BOTH WAYS",),
            "WITHIN-CHANCE": (), "INSUFFICIENT": (), "VOID": ()}


def omega_curve(rows: list[dict]) -> list[dict]:
    """§E (b): the ω grid's false-call rates, all rows, in ω order."""
    om = sorted((r for r in rows if r["kind"] == "omega"), key=lambda r: r["omega_ratio"])
    return [{"omega": r["omega_ratio"], **{s_: r["agg_rate"][s_] for s_ in ("CARRIED", "MASKED", "BOTH WAYS")}}
            for r in om]


def tipping(curve: list[dict], state: str, thr: float) -> dict:
    """§E (a): the ω (× per-answer run noise (σ)) at which `state`'s false-call rate first reaches `thr`, by linear
    interpolation between the bracketing grid rows. 'above at 0' if the first row already reaches it;
    '> 1.5×' (the grid's end) if no row does."""
    w = [c["omega"] for c in curve]
    r = [c[state] for c in curve]
    if r[0] >= thr:
        return {"omega": 0.0, "text": f"≤ 0× (already {r[0]:.2%} at 0)", "bracket": [w[0], w[0]]}
    for i in range(1, len(r)):
        if r[i] >= thr:
            x = w[i - 1] + (thr - r[i - 1]) / (r[i] - r[i - 1]) * (w[i] - w[i - 1])
            return {"omega": x, "text": f"{x:.2f}×", "bracket": [w[i - 1], w[i]]}
    return {"omega": None, "text": f"> {w[-1]}×", "bracket": [w[-1], None]}


def rate_at(curve: list[dict], state: str, omega: float) -> float:
    """The grid rate at ω by linear interpolation; at or beyond the grid's end, the end row's rate."""
    w = [c["omega"] for c in curve]
    return float(np.interp(min(omega, w[-1]), w, [c[state] for c in curve]))


def seed_noise_centred(d1: np.ndarray, d2: np.ndarray, nu: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """The grid's unit for ω: per-run cell-level noise σ², from Δ1 − Δ2 **centred** across cells, so that
    a run-level offset (ω, constant across cells) does not inflate it. Σ (x_c − x̄)² / (2 ν_c) / (2(n − 1))."""
    x = d1 - d2
    n = np.maximum(ok.sum(-1), 2)
    xbar = np.where(ok, x, 0.0).sum(-1) / n
    q = np.where(ok, (x - xbar[..., None]) ** 2 / (2.0 * nu), 0.0)
    return q.sum(-1) / (2.0 * (n - 1))


def a_bound(zA: np.ndarray, ok: np.ndarray, sig2: np.ndarray) -> dict:
    """§E (c): a one-sided 95% upper bound on run-specific noise ω, from A1–A3 alone.

    zA is (3, ..., n): the three v2 runs' z. With m_r the mean over non-VOID cells of run r:
      s_m² = Σ_r (m_r − m̄)² / 2                       (2 df)
      s_int² = Σ_rc (z_rc − m_r − z̄_c + m̄)² / (2(n − 1))  (run × cell residual)
    E[s_m²] = ω² + τ_run² + E[s_int²]/n, and 2 s_m² / E[s_m²] ~ χ²₂, so
      ω_up² = max(2 s_m² / χ²₂(0.05) − s_int²/n, 0)
    bounds ω² + τ_run² ≥ ω², hence ω, at 95% one-sided (if the v2 runs' ω is that of the v1 runs).
    Returned as ω_up / σ̂_c, the grid's unit (× per-answer run noise (σ)), σ̂_c² from seed_noise_centred()."""
    from scipy.stats import chi2

    okb = np.broadcast_to(ok, zA.shape)
    n = np.maximum(ok.sum(-1), 2)
    m = np.where(okb, zA, 0.0).sum(-1) / n
    mbar = m.mean(0)
    s_m2 = ((m - mbar) ** 2).sum(0) / A_BOUND_DF
    zc = zA.mean(0)
    res = np.where(okb, zA - m[..., None] - zc + mbar[..., None], 0.0)
    s_int2 = (res ** 2).sum((0, -1)) / (A_BOUND_DF * (n - 1))
    up2 = np.maximum(2 * s_m2 / chi2.ppf(1 - A_BOUND_LEVEL, A_BOUND_DF) - s_int2 / n, 0.0)
    return {"omega_up": np.sqrt(up2), "ratio": np.sqrt(up2) / np.sqrt(np.maximum(sig2, 1e-12)),
            "s_m2": s_m2, "s_int2": s_int2}


def run_noise_report(state: str, curve: list[dict], bound_ratio: float, drift_: dict) -> dict:
    """§E's block beside an overall state: tipping points, the curve, the A-bound and its rate, the drift."""
    tips = {s_: {f"{t:.1%}": tipping(curve, s_, t) for t in THRESHOLDS} for s_ in ("CARRIED", "MASKED", "BOTH WAYS")}
    applies = CURVE_OF.get(state, ())
    out = {"applies_to": list(applies) or "no false call to bound (CARRIED/MASKED shown for reference)",
           "tipping": tips, "curve": curve,
           "a_bound": {"omega_up_x_seed_noise": bound_ratio, "df": A_BOUND_DF, "level": A_BOUND_LEVEL,
                       "rate_at_bound": {s_: rate_at(curve, s_, bound_ratio) for s_ in ("CARRIED", "MASKED", "BOTH WAYS")},
                       "beyond_grid": bound_ratio > curve[-1]["omega"]},
           "drift": {"mean": drift_["mean"], "se": drift_["se"],
                     "label": "one draw of the run-level noise; not an estimate of its size"}}
    if state in ("CARRIED", "MASKED"):
        t = tips[state]["2.5%"]
        out["headline"] = (f"below 2.5% false calls if run-specific noise < {t['text']} per-answer run noise (σ); "
                           f"up to {curve[-1][state]:.1%} at {curve[-1]['omega']}×")
    return out


def score(records: dict, banks: dict, *, k: float | None = None, p1_fail=(), calibration: dict | None = None,
          p4_fail=()) -> dict:
    """The scoring, pure: loaded t1 records (run -> dict) and banks ("v1"/"v2" -> dict) in, states out."""
    k = K_GUARD if k is None else k
    if k is None:
        raise SystemExit("score: K_GUARD is not calibrated")
    bad = [f"P1: {n}" for n in p1_fail] + list(p4_fail) + p2_record(records, banks)
    if bad:
        return {"k": k, **all_void(bad)}
    p5 = {r: p5_record(records[r], banks[SIDE[r]]) for r in RUNS}
    if any(p5.values()):
        return {"k": k, **all_void([f"P5: {r} {v[:5]}" for r, v in p5.items() if v])}
    out: dict = {"k": k, "void": [], "populations": {}}
    for pop in POPS:
        feats = [row["feature"] for row in records["M"][pop]]
        rows = {r: {row["feature"]: row for row in records[r][pop]} for r in RUNS}
        mats = {s: bank_matrix(banks[s], feats) for s in ("v1", "v2")}
        mC = {s: m.mean(0) for s, m in mats.items()}
        sdC = {s: m.std(0, ddof=1) for s, m in mats.items()}
        void = np.array([not all(p2_row(rows[r][f]) for r in RUNS)
                         or len({(rows[r][f]["n_test"], rows[r][f]["positives_test"]) for r in RUNS}) != 1
                         for f in feats])
        num = lambda r, key: np.array([float(rows[r][f][key]) if p2_row(rows[r][f]) else np.nan for f in feats])  # noqa: E731
        se = {r: num(r, "real_se") for r in RUNS}
        dd = {r: np.sqrt(sdC[SIDE[r]] ** 2 + se[r] ** 2) for r in RUNS}
        z = {r: np.nan_to_num((num(r, "auc") - mC[SIDE[r]]) / dd[r]) for r in RUNS}
        E = {r: np.array([bool(rows[r][f]["exceeds_null"]) for f in feats]) for r in RUNS}
        up = {r: np.array([bool(rows[r][f]["underpowered"]) for f in feats]) for r in RUNS}
        thin = up["M"] | up["O2"] | ((up["A1"].astype(int) + up["A2"] + up["A3"]) >= 2)
        r2 = np.nan_to_num(np.mean([(se[r] / dd[r]) ** 2 for r in RUNS], 0), nan=1.0)
        nu = r2 / max(r2[~void].mean(), 1e-12) if (~void).any() else np.ones(len(feats))
        nu = np.where(void, 1.0, nu)
        dbar = {s: np.sqrt(sdC[s] ** 2 + np.nanmean([se[r] ** 2 for r in RUNS if SIDE[r] == s], 0))
                for s in ("v1", "v2")}
        b2 = np.nan_to_num(sum(sdC[s] ** 2 / (K_BANK * dbar[s] ** 2) for s in ("v1", "v2")))
        d = decide(E, z, nu, b2, void, thin, k)
        G = {r: np.array([rows[r][f]["rung"] == "R1" for f in feats]) for r in RUNS}
        allE = np.all([E[r] for r in RUNS], 0)
        g = grade(G, allE & np.isin(d["code"], (S["AGREES"], S["SHIFTED"])))
        F = k * d["unit"]
        elig = ~void & ~thin
        # chance counts (§E): the paired symmetry (each opposite-flip order is as likely as each
        # same-direction pattern) and, if given, the observed-like calibration row's rates
        c1 = E["A1"].astype(int) - E["M"]
        c2 = E["A2"].astype(int) - E["O2"]
        opp = elig & (c1 * c2 == -1)
        opp_g = opp & (np.abs(d["d1"]) > F) & (np.abs(d["d2"]) > F)
        flip = float((elig & ((c1 != 0) | (c2 != 0))).sum() / max(elig.sum(), 1))
        zA = np.stack([z["A1"], z["A2"], z["A3"]])
        s2v2 = float(np.mean((zA.var(0, ddof=1) / nu)[elig])) if elig.any() else float("nan")
        tau_ratio = math.sqrt(max(s2v2 / max(float(d["s2"]) / 2, 1e-12) - 1.0, 0.0))
        dr_ = drift(d["d1"], d["d2"], ~void, d["s2"])
        a = d["agg"]
        agg = {"state": AGG_STATES[int(a["state"])], "n": int(a["n"]), "m_down": int(a["m_down"]),
               "m_up": int(a["m_up"]), "p_down": float(a["p_down"]), "p_up": float(a["p_up"]),
               "n_void": int(a["n_void"]), "reach": reach(int(a["n"])), "chance_each_pattern": int(a["n"]) / 4,
               "drift": {"mean": float(dr_["mean"]), "se": float(dr_["se"]),
                         "label": "one draw of the run-level noise; not an estimate of its size"}}
        counts = {f"{STATES[c]}{'' if dr == 0 else ('↓' if dr < 0 else '↑')}": int(((d["code"] == c) & (d["dir"] == dr)).sum())
                  for c in range(len(STATES)) for dr in (-1, 0, 1) if ((d["code"] == c) & (d["dir"] == dr)).any()}
        chance = {"symmetric_unguarded_per_direction": float(opp.sum()) / 2,
                  "symmetric_guarded_per_direction_before_A3": float(opp_g.sum()) / 2,
                  "n_scoreable": int(elig.sum())}
        if calibration is not None:
            row = observed_like(calibration["rows"], float(d["s2"]), tau_ratio, flip)
            chance["observed_like_row"] = {kk: row[kk] for kk in ("name", "sigma", "tau_ratio", "lam", "beta")}
            chance["expected_per_state"] = {s_: row["rate"][s_] * int(elig.sum()) for s_ in row["rate"]}
            chance["A-SPLIT"] = {"per_answer": row["a_split_per_answer"],
                                 "expected_of_37": row["a_split_expected_37"],
                                 "expected_scoreable": row["a_split_per_answer"] * int(elig.sum())}
            chance["aggregate_false_rate"] = row["agg_rate"]
            # §E: run-specific noise beside the overall state; no single "observed" rate (user, 17 Nov)
            ab = a_bound(np.stack([z["A1"], z["A2"], z["A3"]]), ~void,
                         seed_noise_centred(d["d1"], d["d2"], nu, ~void))
            agg["run_noise"] = run_noise_report(agg["state"], omega_curve(calibration["rows"]),
                                                float(ab["ratio"]), {"mean": float(dr_["mean"]),
                                                                     "se": float(dr_["se"])})
        cells = []
        for j, f in enumerate(feats):
            cells.append({
                "feature": f, "state": STATES[d["code"][j]], "dir": int(d["dir"][j]), "grade": G_STATES[g[j]],
                "F": float(F[j]), "d1": float(d["d1"][j]), "d2": float(d["d2"][j]),
                "z": {r: float(z[r][j]) for r in RUNS}, "E": {r: bool(E[r][j]) for r in RUNS},
                "rung": {r: rows[r][f].get("rung") for r in RUNS},
                "mechanism": {r: rows[r][f].get("mechanism") for r in RUNS},
                "underpowered": {r: bool(up[r][j]) for r in RUNS},
                # the untrained bars, reported beside the state with no state of their own (§D):
                # D23's K-seed bank (the bar existence is judged against; one per answer, both populations)
                # and the record's D24 C (three untrained draws on this population's rows), where recorded
                "bar_d23": {s: {"mean": float(mC[s][j]), "sd": float(sdC[s][j]), "K": K_BANK} for s in ("v1", "v2")},
                "bar_d24_C": {r: rows[r][f].get("C") for r in RUNS},
                "retention": {r: {kk: rows[r][f].get(kk) for kk in ("retention", "retained",
                                                                     "margin_established", "competitive")}
                              for r in RUNS},
                "retention_pairs_agree": [rows[v][f].get("retention") == rows[w][f].get("retention")
                                          for v, w in PAIRS],
            })
        out["populations"][pop] = {"confirmatory": pop == PRIMARY_POP, "aggregate": agg, "counts": counts,
                                   "chance": chance, "sigma2_pair": float(d["s2"]), "tau_ratio_A": tau_ratio,
                                   "flip_rate": flip, "cells": cells}
    return out


def score_real(go: bool) -> dict:
    """The real scoring. Refuses without the hash, without --go, or with any pin, P4 SHA or K unset."""
    assert_hashed()
    if not go:
        raise SystemExit("m_comparison score: refuses without --go (the user's word). Nothing is read.")
    if V1_OUTPUTS_SHA1 is None or K_GUARD is None:
        raise SystemExit("m_comparison score: V1_OUTPUTS_SHA1 or K_GUARD is unset")
    if _sha1(V1_OUTPUTS) != V1_OUTPUTS_SHA1:
        raise SystemExit("m_comparison score: docs/ladder_v2_m/outputs.sha1 does not match its recorded SHA-1")
    pins = dict(PINS)
    pins.update(outputs_pins(V1_OUTPUTS, {"M": RECORDS["M"], "O2": RECORDS["O2"], "bank_v1": BANKS["v1"]}))
    a_list = outputs_pins(A_OUTPUTS, {"A1": RECORDS["A1"], "A2": RECORDS["A2"], "A3": RECORDS["A3"],
                                      "bank_v2": BANKS["v2"]})
    p4 = p4_status() + [f"P1: {n} pin differs from docs/ladder_v2/outputs.sha1" for n, v in a_list.items()
                        if v != PINS[n]]
    recs, banks, p1 = load_inputs(RECORDS, BANKS, pins)
    res = score(recs, banks, p1_fail=p1, p4_fail=p4, calibration=json.loads(CALIBRATION.read_text()))
    res["prereg_sha1"] = PREREG_SHA1
    # §I, after the scoring and whatever it says: M's original record, rung by rung (no gate)
    res["reproduction_M"] = reproduction(load_pinned(ORIGINAL_M, ORIGINAL_M_SHA1), recs["M"])
    (HERE / "out" / "m_comparison_score.json").write_text(json.dumps(res, indent=1, default=float))
    return res


# --- calibrate: decision-layer Monte Carlo on synthetic z -------------------------------------
H37 = float(np.sum(1.0 / np.arange(1, N_FEATURES + 1)))


OMEGA_GRID = np.round(np.arange(0.0, 1.5001, 0.1), 1)  # ω/σ, 0 to 1.5 in steps of 0.1


def _regimes() -> list[dict]:
    rows = []
    for sig in (0.5, 1.0, 2.0):
        for tr, lam in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (2.0, 0.0), (2.0, 1.0)):
            for beta in (0.1, 0.3):
                rows.append({"kind": "base", "sigma": sig, "tau_ratio": tr, "lam": lam, "beta": beta,
                             "ratio": 1.0, "omega_ratio": 0.0, "q": 1.0, "het": "se"})
    stress = [dict(ratio=2.0, tau_ratio=0.0), dict(ratio=2.0, tau_ratio=1.0),
              dict(omega_ratio=0.5, tau_ratio=1.0), dict(het="mis", tau_ratio=0.0),
              dict(het="mis", tau_ratio=1.0), dict(q=0.0, tau_ratio=0.0)]
    for st in stress:
        r = {"kind": "stress", "sigma": 1.0, "tau_ratio": 0.0, "lam": 0.0, "beta": 0.3, "ratio": 1.0,
             "omega_ratio": 0.0, "q": 1.0, "het": "se"}
        r.update(st)
        rows.append(r)
    for om in OMEGA_GRID:  # §E's lookup grid: σ 1, no u_seed (the worst case), β 0.3
        rows.append({"kind": "omega", "sigma": 1.0, "tau_ratio": 0.0, "lam": 0.0, "beta": 0.3, "ratio": 1.0,
                     "omega_ratio": float(om), "q": 1.0, "het": "se"})
    for om, tr, lam in ((0.5, 1.0, 1.0), (1.0, 1.0, 1.0), (0.5, 2.0, 0.0)):  # A-bound coverage with u_seed
        rows.append({"kind": "stress", "sigma": 1.0, "tau_ratio": tr, "lam": lam, "beta": 0.3, "ratio": 1.0,
                     "omega_ratio": om, "q": 1.0, "het": "se"})
    for r in rows:
        r["name"] = (f"{r['kind']} σ{r['sigma']} τ/σ{r['tau_ratio']} λ{r['lam']} β{r['beta']} "
                     f"v2/v1 {r['ratio']} ω/σ{r['omega_ratio']} bank{r['q']} {r['het']}")
    return rows


def draw(rng, R: int, reg: dict, lost: int = 0, gained: int = 0, delta: float = 0.0):
    """Synthetic z for the five runs. z_rc = μ_c + δ_c[v2] + u_{seed(r),c} + ω_r + β_{side(r),c}
    + σ_side √ν_c ε_rc. u_seed is shared by both members of a pair (λ: its run-level share), ω is
    corpus-specific run-level noise, β is each side's bank-mean error (common to that side's runs)."""
    n = N_FEATURES
    nb = int(round(reg["beta"] * n))
    ns = (n - nb) // 2
    nw = n - nb - ns
    zstar = float(student_t.isf((ns + nb / 2) / (n * H37) * ALPHA_BY, K_BANK - 1))
    mu = np.concatenate([rng.uniform(8, 15, (R, ns)), zstar + rng.normal(0, 0.3, (R, nb)),
                         rng.uniform(-2, 1, (R, nw))], 1)
    border = np.zeros(n, bool)
    border[ns:ns + nb] = True
    dlt = np.zeros(n)
    if lost:  # planted: v1 above the bar, v2 moved down by delta
        mu[:, :lost] = zstar + 2.0
        dlt[:lost] = -delta
    if gained:
        mu[:, ns + nb:ns + nb + gained] = zstar - 2.0
        dlt[ns + nb:ns + nb + gained] = delta
    nu_e = rng.lognormal(0, 0.5, (R, n))
    nu_e /= nu_e.mean(1, keepdims=True)
    nu_t = nu_e if reg["het"] == "se" else (lambda v: v / v.mean(1, keepdims=True))(rng.lognormal(0, 0.5, (R, n)))
    b2 = np.full((R, n), reg["q"] * 2 * 0.5 / K_BANK)  # (sd/d)² ≈ 0.5 on each side
    sig = reg["sigma"]
    tau, om = reg["tau_ratio"] * sig, reg["omega_ratio"] * sig
    u = {s: tau * (math.sqrt(reg["lam"]) * rng.normal(0, 1, (R, 1))
                   + math.sqrt(1 - reg["lam"]) * rng.normal(0, 1, (R, n))) for s in (0, 1, 2)}
    bank_err = {s: rng.normal(0, 1, (R, n)) * np.sqrt(b2 / 2) for s in ("v1", "v2")}
    seed = {"M": 0, "A1": 0, "O2": 1, "A2": 1, "A3": 2}
    z, E = {}, {}
    for r in RUNS:
        sr = sig * (reg["ratio"] if SIDE[r] == "v2" else 1.0)
        z[r] = (mu + (dlt if SIDE[r] == "v2" else 0.0) + u[seed[r]] + om * rng.normal(0, 1, (R, 1))
                + bank_err[SIDE[r]] + sr * np.sqrt(nu_t) * rng.normal(0, 1, (R, n)))
        E[r] = by_reject(student_t.sf(z[r], K_BANK - 1))
    return E, z, nu_e, b2, border


def _rates(d: dict, border: np.ndarray) -> dict:
    code, dirn = d["code"], d["dir"]
    rate = {}
    for s_ in STATES[2:]:
        for dr, tag in ((-1, "↓"), (1, "↑"), (0, "")):
            m = (code == S[s_]) & (dirn == dr)
            if m.any():
                rate[s_ + tag] = float(m.mean())
    rate["A-LOWER"] = float((code == S["A-LOWER"]).mean())
    rate["A-HIGHER"] = float((code == S["A-HIGHER"]).mean())
    a = d["agg"]["state"]
    asplit = float((code == S["A-SPLIT"]).mean())
    return {"rate": rate, "a_split_per_answer": asplit, "a_split_expected_37": asplit * N_FEATURES,
            "drift_median": float(np.median(drift(d["d1"], d["d2"], np.ones(code.shape, bool), d["s2"])["scaled"])),
            "a_lower_per_borderline": float((code[:, border] == S["A-LOWER"]).mean()) if border.any() else 0.0,
            "a_higher_per_borderline": float((code[:, border] == S["A-HIGHER"]).mean()) if border.any() else 0.0,
            "agg_rate": {AGG_STATES[i]: float((a == i).mean()) for i in range(len(AGG_STATES))}}


def calibrate(R: int = 20_000, seed: int = 20261117) -> dict:
    """Two passes over the same draws (one generator per regime, so memory holds one regime at a time):
    the first fixes k by the rule, the second scores every regime at that k."""
    regs = _regimes()
    rule = (f"k = the smallest value on a 0.05 grid such that, in every base regime (with and without "
            f"u_seed; equal corpus noise, no ω), the per-cell rate of 'both pairs beyond F_c in the same "
            f"direction' is <= {GUARD_BOUND} for each direction")
    mags_all, kreq = [], []
    for i, reg in enumerate(regs):
        E, z, nu, b2, border = draw(np.random.default_rng([seed, i]), R, reg)
        void = np.zeros(z["M"].shape, bool)
        d = decide(E, z, nu, b2, void, void, 1.0)
        mags = []
        for sgn in (-1, 1):
            ms = np.sort((np.minimum(sgn * d["d1"], sgn * d["d2"]) / d["unit"]).ravel())
            mags.append(1.0 - np.searchsorted(ms, K_GRID, "right") / ms.size)
        mags_all.append(mags)
        if reg["kind"] == "base":
            kreq.append(np.maximum(*mags))
        del E, z, d
    worst = np.max(kreq, 0)
    if not (worst <= GUARD_BOUND).any():
        raise SystemExit("calibrate: no k on the grid meets the rule")
    ki = int(np.argmax(worst <= GUARD_BOUND))
    k = float(K_GRID[ki])
    rows = []
    for i, reg in enumerate(regs):
        E, z, nu, b2, border = draw(np.random.default_rng([seed, i]), R, reg)
        void = np.zeros(z["M"].shape, bool)
        d = decide(E, z, nu, b2, void, void, k)
        c1 = E["A1"].astype(int) - E["M"]
        c2 = E["A2"].astype(int) - E["O2"]
        row = {**reg, "mag_down": float(mags_all[i][0][ki]), "mag_up": float(mags_all[i][1][ki]),
               "flip_rate": float(((c1 != 0) | (c2 != 0)).mean()),
               "median_unit": float(np.median(d["unit"])), **_rates(d, border)}
        ok1 = np.ones(z["M"].shape, bool)
        ab = a_bound(np.stack([z["A1"], z["A2"], z["A3"]]), ok1, seed_noise_centred(d["d1"], d["d2"], nu, ok1))
        row["a_bound_coverage"] = float((ab["ratio"] >= reg["omega_ratio"]).mean())
        row["a_bound_median"] = float(np.median(ab["ratio"]))
        rows.append(row)
        del E, z, d
        print(json.dumps({"k": k, "name": row["name"], "mag": [round(row["mag_down"], 4), round(row["mag_up"], 4)],
                          "A-LOWER/border": round(row["a_lower_per_borderline"], 4),
                          "CARRIED": round(row["agg_rate"]["CARRIED"], 4),
                          "MASKED": round(row["agg_rate"]["MASKED"], 4)}), flush=True)
    rng = np.random.default_rng([seed, 999])
    power = []
    for tr in (0.0, 1.0):
        reg = {**_regimes()[0], "sigma": 1.0, "tau_ratio": tr, "beta": 0.3, "kind": "power"}
        for j in (2, 4, 8):
            for delta in (2.0, 3.0, 4.0, 6.0):
                E, z, nu, b2, border = draw(rng, 4000, reg, lost=j, delta=delta)
                void = np.zeros(z["M"].shape, bool)
                d = decide(E, z, nu, b2, void, void, k)
                power.append({"tau_ratio": tr, "lost": j, "delta_z": delta,
                              "a_lower_on_planted": float((d["code"][:, :j] == S["A-LOWER"]).mean()),
                              "a_split_on_planted": float((d["code"][:, :j] == S["A-SPLIT"]).mean()),
                              "carried": float((d["agg"]["state"] == 3).mean()),
                              "both_ways": float((d["agg"]["state"] == 2).mean())})
    return {"R": R, "seed": seed, "rule": rule, "bound": GUARD_BOUND, "k": k, "rows": rows, "power": power}


# --- plants: synthetic t1 records through load_inputs -> score() ------------------------------
FEATS = [f"t{j // 3 + 1:02d}_a{j % 3:02d}" for j in range(N_FEATURES)]
N_STRONG = 22  # background: cells 0..21 strong (z 8–14, E = 1), 22..36 weak (z −1.5–0.5, E = 0)


def synth_banks(rng) -> dict:
    """Two K-seed banks on one synthetic centre per answer (so a v1/v2 rung difference is the plant's),
    the same model and split keys, different pixels keys."""
    centre = rng.uniform(0.74, 0.80, N_FEATURES)
    sd = rng.uniform(0.003, 0.006, N_FEATURES)
    return {s: {"model_key": "synthetic-model", "split_key": "synthetic-split", "pixels_key": f"synthetic-{s}",
                "max_train": 40_000,
                "seeds": {str(i): {f: float(rng.normal(centre[j], sd[j])) for j, f in enumerate(FEATS)}
                          for i in range(K_BANK)}} for s in ("v1", "v2")}


def synth_record(run: str, z: dict, se: dict, bank: dict, *, r2_cells=(), n_test=None) -> dict:
    """A t1 record with the full row schema, D23 computed through the package exactly as the ladder does."""
    from galaxy_jepa.probing import nulls

    mat = bank_matrix(bank, FEATS)
    mC, sdC = mat.mean(0), mat.std(0, ddof=1)
    out: dict = {"checkpoint": CHECKPOINTS[run], "k_bar": K_BAR_D24}
    for pop in POPS:
        auc = mC + z[pop] * np.sqrt(sdC ** 2 + se[pop] ** 2)
        p = {f: nulls.untrained_z_pvalue(auc[j], se[pop][j], mat[:, j]) for j, f in enumerate(FEATS)}
        sig = nulls.family_significant(p, alpha=ALPHA_BY, method="benjamini_yekutieli", n_tests=N_FEATURES)
        rows = []
        for j, f in enumerate(FEATS):
            ex = bool(sig[f])
            clean = ex and bool(auc[j] >= FLOOR)
            rung = "R4" if not ex else ("R2" if (not clean or (pop, j) in r2_cells) else "R1")
            margin = nulls.resolvable_margin(float(se[pop][j]), sd_bar=float(sdC[j]), alpha=ALPHA_BY,
                                             method="benjamini_yekutieli", n_tests=N_FEATURES, df=K_BANK - 1)
            nt = 1000 + 37 * j if n_test is None else n_test(pop, j)
            rows.append({"feature": f, "rung": rung, "mechanism": "synthetic", "auc": float(auc[j]),
                         "n_test": nt, "positives_test": 300 + j, "real_se": float(se[pop][j]),
                         "resolvable_margin": margin, "underpowered": nulls.is_underpowered(margin),
                         "matched_auc": None, "matched_n_test": None, "matched_share": None,
                         "matched_degenerate": None, "retention": "SURVIVES" if ex else "UNRESOLVED",
                         "retained": 1.0 if ex else None, "M_lo": None, "C": float(mC[j]), "C_m": None,
                         "k_bar": K_BAR_D24, "margin_established": ex, "competitive": [],
                         "exceeds_null": ex, "clean": clean})
        out[pop] = rows
    return out


def base_world(rng, sigma: float = 0.15) -> dict:
    base = {pop: np.concatenate([rng.uniform(8, 14, N_STRONG), rng.uniform(-1.5, 0.5, N_FEATURES - N_STRONG)])
            for pop in POPS}
    return {r: {pop: base[pop] + rng.normal(0, sigma, N_FEATURES) for pop in POPS} for r in RUNS}


def _se(val: float = 0.005) -> dict:
    return {pop: np.full(N_FEATURES, val) for pop in POPS}


def build(rng, z: dict, *, se: dict | None = None, banks: dict | None = None, r2=None, n_test=None,
          tamper=None) -> tuple[dict, dict]:
    banks = banks or synth_banks(rng)
    se = se or {r: _se() for r in RUNS}
    recs = {r: synth_record(r, z[r], se[r], banks[SIDE[r]], r2_cells=(r2 or {}).get(r, ()),
                            n_test=(n_test or {}).get(r)) for r in RUNS}
    if tamper:
        tamper(recs, banks)
    return recs, banks


def run_files(recs: dict, banks: dict, pins_override=None, k=None, calibration=None) -> dict:
    """Write the synthetic records to a temp dir, pin them, and score through load_inputs, as the real run."""
    with tempfile.TemporaryDirectory(prefix="m_comparison_plant_") as td:
        rp = {r: Path(td) / f"{r}.json" for r in RUNS}
        bp = {s: Path(td) / f"bank_{s}.json" for s in ("v1", "v2")}
        for r in RUNS:
            rp[r].write_text(json.dumps(recs[r]))
        for s in ("v1", "v2"):
            bp[s].write_text(json.dumps(banks[s]))
        pins = {**{r: _sha1(rp[r]) for r in RUNS}, **{f"bank_{s}": _sha1(bp[s]) for s in ("v1", "v2")}}
        pins.update(pins_override or {})
        lr, lb, p1 = load_inputs(rp, bp, pins)
    return score(lr, lb, p1_fail=p1, k=k, calibration=calibration)


def _cells(res: dict, pop: str = "full") -> list[dict]:
    c = res["populations"][pop]["cells"]
    return c if isinstance(c, list) else []


def plants() -> dict:
    cal = json.loads(CALIBRATION.read_text())
    if cal["k"] != K_GUARD:
        raise SystemExit(f"plants: K_GUARD {K_GUARD} is not the calibration's k {cal['k']}")
    rng = np.random.default_rng(20261118)
    out: dict = {"k": K_GUARD, "calibration_sha1": _sha1(CALIBRATION), "structural": {}, "void": {}, "null": {}}
    strong, weak = list(range(N_STRONG)), list(range(N_STRONG, N_FEATURES))

    def put(z, runs, cells, val, pop="full"):
        for r in runs:
            for c in cells:
                z[r][pop][c] = val

    def check(name, z, expect_cells, expect_agg, *, every=None, **kw):
        recs, banks = build(rng, z, **kw)
        res = run_files(recs, banks, calibration=cal)
        cells = _cells(res)
        got = {c: (cells[c]["state"], cells[c]["dir"]) for c in expect_cells}
        ok_cells = all(got[c] == want for c, want in expect_cells.items())
        if every:
            ok_cells = ok_cells and all((cl["state"], cl["dir"]) == every
                                        for p in POPS for cl in _cells(res, p))
        agg = res["populations"]["full"]["aggregate"]["state"]
        fired = ok_cells and (expect_agg is None or agg == expect_agg)
        out["structural"][name] = {"expect_cells": {FEATS[c]: list(v) for c, v in expect_cells.items()},
                                   "expect_aggregate": expect_agg, "got_aggregate": agg,
                                   "got_cells": {FEATS[c]: list(v) for c, v in got.items()},
                                   "counts_full": res["populations"]["full"]["counts"], "fired": fired}
        print(f"{name:14s} fired {fired}  agg {agg}  {res['populations']['full']['counts']}", flush=True)
        return res

    # identical: five copies of one world (no run noise)
    w = base_world(rng, 0.0)
    w = {r: {p: w["M"][p].copy() for p in POPS} for r in RUNS}
    check("identical", w, {}, "INSUFFICIENT", every=("AGREES", 0))
    v2, a12 = ("A1", "A2", "A3"), ("A1", "A2")
    # lost-8 / gained-8
    w = base_world(rng)
    put(w, v2, strong[:8], 0.0)
    check("lost-8", w, {c: ("A-LOWER", -1) for c in strong[:8]}, "CARRIED")
    w = base_world(rng)
    put(w, v2, weak[:8], 10.0)
    check("gained-8", w, {c: ("A-HIGHER", 1) for c in weak[:8]}, "MASKED")
    w = base_world(rng)
    put(w, v2, strong[:10], 0.0)
    put(w, v2, weak[:10], 10.0)
    check("both", w, {**{c: ("A-LOWER", -1) for c in strong[:10]}, **{c: ("A-HIGHER", 1) for c in weak[:10]}},
          "BOTH WAYS")
    w = base_world(rng)
    put(w, v2, strong[:2], 0.0)
    check("below-reach", w, {c: ("A-LOWER", -1) for c in strong[:2]}, "INSUFFICIENT")
    # sign flip across pairs; one pair only; A3 dissents
    w = base_world(rng)
    put(w, ("M", "A2"), weak[:5], 10.0)
    check("pairs-oppose", w, {c: ("PAIRS-OPPOSE", 0) for c in weak[:5]}, None)
    w = base_world(rng)
    put(w, ("A1",), strong[:5], 0.0)
    check("one-pair", w, {c: ("ONE-PAIR", -1) for c in strong[:5]}, None)
    w = base_world(rng)
    put(w, a12, strong[:5], 0.0)
    check("a-split", w, {c: ("A-SPLIT", -1) for c in strong[:5]}, None)
    # near-bar: the five cells straddle the BY bar at rank N_STRONG + 5 by ±0.05 in z
    zc = float(student_t.isf((N_STRONG + 5) / (N_FEATURES * H37) * ALPHA_BY, K_BANK - 1))
    w = base_world(rng)
    put(w, ("M", "O2"), weak[:5], zc + 0.05)
    put(w, v2, weak[:5], zc - 0.05)
    check("near-bar", w, {c: ("NEAR-BAR", -1) for c in weak[:5]}, None)
    # magnitude without a flip; a seed-driven reading
    w = base_world(rng)
    for r in v2:
        for c in strong[:5]:
            w[r]["full"][c] = w[r]["full"][c] + 4.0
    check("shifted", w, {c: ("SHIFTED", 1) for c in strong[:5]}, None)
    w = base_world(rng)
    put(w, ("M", "A1"), weak[:5], 10.0)
    check("seed-split", w, {c: ("SEED-SPLIT", 0) for c in weak[:5]}, None)
    # thin: M underpowered in three cells, A1 and A3 in three others
    w = base_world(rng)
    se = {r: _se() for r in RUNS}
    for c in weak[:3]:
        se["M"]["full"][c] = 0.03
    for c in weak[3:6]:
        se["A1"]["full"][c] = se["A3"]["full"][c] = 0.03
    check("thin", w, {c: ("THIN", 0) for c in weak[:6]}, None, se=se)
    # grade: five strong cells R1 on v1, R2 on all three v2 seeds
    w = base_world(rng)
    res = check("grade", w, {}, None, r2={r: {("full", c) for c in strong[:5]} for r in v2})
    gc = [_cells(res)[c]["grade"] for c in strong[:5]]
    out["structural"]["grade"].update({"expect_grade": "G-LOWER", "got_grade": gc,
                                       "fired": all(g == "G-LOWER" for g in gc)})
    print(f"{'grade':14s} grade {gc}", flush=True)

    # VOID set: each separately
    def void_plant(name, level, *, pins=None, tamper=None, n_test=None, want_agg: str | None = "VOID"):
        recs, banks = build(rng, base_world(rng), tamper=tamper, n_test=n_test)
        res = run_files(recs, banks, pins_override=pins, calibration=cal)
        if level == "all":
            fired = bool(res.get("void")) and isinstance(res["populations"]["full"]["cells"], str)
        else:
            cells = _cells(res)
            fired = cells[level]["state"] == "VOID" and sum(c["state"] == "VOID" for c in cells) == 1
        agg = res["populations"]["full"]["aggregate"]["state"]
        if want_agg is not None:
            fired = fired and agg == want_agg
        out["void"][name] = {"level": "everything" if level == "all" else f"cell {FEATS[level]}",
                             "void": res.get("void"), "aggregate": agg, "fired": fired}
        print(f"void:{name:22s} fired {fired}  {res.get('void')}", flush=True)

    def drop(recs, banks):
        recs["A2"]["full"] = recs["A2"]["full"][:-1]

    def r3(recs, banks):
        recs["O2"]["full"][5]["rung"] = "R3"

    def kbar(recs, banks):
        recs["A3"]["k_bar"] = 2

    def flip(recs, banks):
        recs["A1"]["conditional"][3]["exceeds_null"] = not recs["A1"]["conditional"][3]["exceeds_null"]

    def bank_k(recs, banks):
        banks["v1"]["seeds"].pop("29")

    def split_key(recs, banks):
        banks["v1"]["split_key"] = "another-split"

    def same_pixels(recs, banks):
        banks["v1"]["pixels_key"] = banks["v2"]["pixels_key"]

    def ckpt(recs, banks):
        recs["O2"]["checkpoint"] = "runs/m/encoder.pt"

    def four_cells(recs, banks):
        for c in range(4):
            recs["M"]["full"][c]["rung"] = "R3"

    void_plant("wrong_sha1", "all", pins={"A2": "0" * 40})
    void_plant("missing_answer", "all", tamper=drop)
    void_plant("recorded_R3", 5, tamper=r3, want_agg=None)
    void_plant("n_test_off_by_10", 7, n_test={"O2": lambda pop, j: 1000 + 37 * j + (10 if (pop, j) == ("full", 7) else 0)},
               want_agg=None)
    void_plant("k_bar_2", "all", tamper=kbar)
    void_plant("exceeds_null_flipped", "all", tamper=flip)
    void_plant("bank_K_29", "all", tamper=bank_k)
    void_plant("bank_split_key", "all", tamper=split_key)
    void_plant("bank_same_pixels", "all", tamper=same_pixels)
    void_plant("wrong_checkpoint", "all", tamper=ckpt)
    recs, banks = build(rng, base_world(rng), tamper=four_cells)
    res = run_files(recs, banks, calibration=cal)
    agg = res["populations"]["full"]["aggregate"]["state"]
    out["void"]["four_void_cells"] = {"level": "full aggregate", "aggregate": agg,
                                      "conditional_aggregate": res["populations"]["conditional"]["aggregate"]["state"],
                                      "fired": agg == "VOID"}
    print(f"void:four_void_cells        fired {agg == 'VOID'}", flush=True)

    # §E printing: tipping points against hand-computed values, the never-crosses and above-at-0 cases
    grid = [round(0.1 * i, 1) for i in range(16)]
    toy = [{"omega": w, "CARRIED": 0.01 * i, "MASKED": 0.01, "BOTH WAYS": 0.03} for i, w in enumerate(grid)]
    real = omega_curve(cal["rows"])
    hand = {}
    for st in ("CARRIED", "MASKED"):
        for t in THRESHOLDS:
            r = [c[st] for c in real]
            i = next((i for i in range(len(r)) if r[i] >= t), None)
            hand[(st, t)] = None if i is None or i == 0 else grid[i - 1] + (t - r[i - 1]) * 0.1 / (r[i] - r[i - 1])
    checks = {
        "toy CARRIED 2.5% = 0.25": abs(tipping(toy, "CARRIED", 0.025)["omega"] - 0.25) < 1e-12,
        "toy CARRIED 5% = 0.50": abs(tipping(toy, "CARRIED", 0.05)["omega"] - 0.50) < 1e-12,
        "toy never crosses -> > 1.5x": tipping(toy, "MASKED", 0.025)["text"] == "> 1.5×",
        "toy above at 0 -> <= 0x": tipping(toy, "BOTH WAYS", 0.025)["omega"] == 0.0,
        "real BOTH WAYS never crosses": tipping(real, "BOTH WAYS", 0.025)["text"] == "> 1.5×",
        **{f"real {st} {t:.1%} = hand": (tipping(real, st, t)["omega"] is None and hand[(st, t)] is None)
           or abs(tipping(real, st, t)["omega"] - hand[(st, t)]) < 1e-12 for (st, t) in hand},
    }
    out["printing"] = {"checks": checks, "fired": all(checks.values()),
                       "real_tipping": {f"{st} {t:.1%}": {**tipping(real, st, t),
                                                          "rows": [c for c in real if c["omega"] in tipping(real, st, t)["bracket"]]}
                                        for st in ("CARRIED", "MASKED") for t in THRESHOLDS}}
    print(f"printing         fired {out['printing']['fired']}  {checks}", flush=True)
    # and through score(): the block is printed beside every overall state, with the headline on CARRIED
    w = base_world(rng)
    put(w, v2, strong[:8], 0.0)
    rn = run_files(*build(rng, w), calibration=cal)["populations"]["full"]["aggregate"]["run_noise"]
    out["printing"]["score_block_on_CARRIED"] = {"headline": rn.get("headline"), "a_bound": rn["a_bound"],
                                                 "drift_label": rn["drift"]["label"]}
    out["printing"]["fired"] &= bool(rn.get("headline")) and len(rn["curve"]) == 16

    # null and power through the full path: z from the calibration's draw(), records from z
    def through(name, reg, n_real, lost=0, delta=0.0):
        t0 = time.perf_counter()
        banks = synth_banks(rng)
        tally = {"agg": {s_: 0 for s_ in AGG_STATES}, "cells": 0, "SHIFTED": 0, "A-LOWER": 0, "A-HIGHER": 0,
                 "A-SPLIT": 0, "A-LOWER_on_planted": 0, "A-SPLIT_on_planted": 0, "planted": 0}
        for _ in range(n_real):
            E_, zz, *_ = draw(rng, 2, reg, lost=lost, delta=delta)
            z = {r: {pop: zz[r][i] for i, pop in enumerate(POPS)} for r in RUNS}
            recs, _b = build(rng, z, banks=banks)
            res = score(recs, banks, calibration=cal)
            bd = res["populations"]["full"]["aggregate"]["run_noise"]["a_bound"]["omega_up_x_seed_noise"]
            tally["bounds"] = tally.get("bounds", []) + [bd]
            for pop in POPS:
                tally["agg"][res["populations"][pop]["aggregate"]["state"]] += 1
                for j, c in enumerate(res["populations"][pop]["cells"]):
                    tally["cells"] += 1
                    for s_ in ("SHIFTED", "A-LOWER", "A-HIGHER", "A-SPLIT"):
                        tally[s_] += c["state"] == s_
                    if j < lost:
                        tally["planted"] += 1
                        tally["A-LOWER_on_planted"] += c["state"] == "A-LOWER"
                        tally["A-SPLIT_on_planted"] += c["state"] == "A-SPLIT"
        npop = 2 * n_real
        r_ = {"regime": reg["name"], "populations": npop, "lost": lost, "delta_z": delta,
              "aggregate_rate": {s_: v / npop for s_, v in tally["agg"].items()},
              "per_cell": {s_: tally[s_] / tally["cells"] for s_ in ("SHIFTED", "A-LOWER", "A-HIGHER", "A-SPLIT")},
              "a_bound_coverage": float(np.mean(np.array(tally["bounds"]) >= reg["omega_ratio"])),
              "a_bound_median": float(np.median(tally["bounds"])),
              "seconds": round(time.perf_counter() - t0, 1)}
        if lost:
            r_["a_lower_on_planted"] = tally["A-LOWER_on_planted"] / tally["planted"]
            r_["a_split_on_planted"] = tally["A-SPLIT_on_planted"] / tally["planted"]
        se_agg = math.sqrt(ALPHA_AGG * (1 - ALPHA_AGG) / npop)
        se_cell = math.sqrt(GUARD_BOUND * (1 - GUARD_BOUND) / tally["cells"])
        if not lost:
            r_["bound_aggregate_per_direction"] = ALPHA_AGG + 2 * se_agg
            r_["bound_shifted_per_cell"] = 2 * (GUARD_BOUND + 2 * se_cell)
        out["null" if not lost else "power"] = out.get("null" if not lost else "power", {})
        out["null" if not lost else "power"][name] = r_
        print(f"{name:22s} {json.dumps({k_: r_[k_] for k_ in ('aggregate_rate', 'per_cell')})} "
              f"{r_.get('a_lower_on_planted', '')} {r_['seconds']}s", flush=True)
        return r_

    base = {r["name"]: r for r in _regimes()}
    pick = lambda **kw: next(r for r in base.values() if all(r[k_] == v for k_, v in kw.items()))  # noqa: E731
    cell = pick(kind="base", sigma=1.0, tau_ratio=0.0, lam=0.0, beta=0.3)
    seedrow = pick(kind="base", sigma=1.0, tau_ratio=1.0, lam=0.0, beta=0.3)
    seedrun = pick(kind="base", sigma=1.0, tau_ratio=1.0, lam=1.0, beta=0.3)
    omega = pick(kind="omega", omega_ratio=0.5)
    n0 = through("null_no_useed", cell, 300)
    n1 = through("null_useed_cell", seedrow, 300)
    n2 = through("null_useed_run", seedrun, 300)
    through("null_omega_0.5", omega, 300)
    through("null_omega_1.0", pick(kind="omega", omega_ratio=1.0), 300)
    through("null_omega_0.5_useed_run", pick(kind="stress", omega_ratio=0.5, tau_ratio=1.0, lam=1.0), 300)
    for name in ("null_omega_0.5", "null_omega_1.0", "null_omega_0.5_useed_run"):
        r_ = out["null"][name]
        r_["a_bound_fired"] = r_["a_bound_coverage"] >= A_BOUND_LEVEL - 2 * math.sqrt(0.05 * 0.95 / 300)
    for j in (2, 4, 8):
        through(f"power_lost{j}_dz4", cell, 60, lost=j, delta=4.0)
    for name, r_ in (("null_no_useed", n0), ("null_useed_cell", n1), ("null_useed_run", n2)):
        ar = r_["aggregate_rate"]
        r_["fired"] = (ar["CARRIED"] + ar["BOTH WAYS"] <= r_["bound_aggregate_per_direction"]
                       and ar["MASKED"] + ar["BOTH WAYS"] <= r_["bound_aggregate_per_direction"]
                       and r_["per_cell"]["SHIFTED"] <= r_["bound_shifted_per_cell"])
    out["all_fired"] = (all(v["fired"] for v in out["structural"].values())
                        and all(v["fired"] for v in out["void"].values())
                        and all(out["null"][n]["fired"] for n in ("null_no_useed", "null_useed_cell", "null_useed_run"))
                        and out["printing"]["fired"]
                        and all(out["null"][n]["a_bound_fired"]
                                for n in ("null_omega_0.5", "null_omega_1.0", "null_omega_0.5_useed_run")))
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "calibrate":
        CALIBRATION.write_text(json.dumps(calibrate(), indent=1))
    elif mode == "plants":
        (OUT / "plants.json").write_text(json.dumps(plants(), indent=1, default=float))
    elif mode == "score":
        score_real(go="--go" in sys.argv[2:])
    else:
        raise SystemExit("usage: m_comparison.py calibrate | plants | score --go")
