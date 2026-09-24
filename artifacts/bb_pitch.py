"""Brief BB — which pitch-angle method works at SDSS quality? Synthetic spirals with known truth.

Both methods are external tools, run as subprocesses; neither is vendored here.
  PyArcFiRe  ~/Documents/pitch-methods/pyarcfire   (private Ma1achy/pyarcfire; scripts/bb_measure.py)
  P2DFFT     ~/Documents/pitch-methods/p2dfft-6.2-src/build  (p2dfft, p2spiral; p2pa is python)
States and pre-registrations: `bb_findings.md`.

  --plant-fidelity   BB0a D28 (state logic + the mirrored end-to-end plant)
  --fidelity         BB0a: PyArcFiRe against SpArcFiRe's regression outputs
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

TOOLS = Path.home() / "Documents" / "pitch-methods"
PYARC = TOOLS / "pyarcfire"
FID = PYARC / "bb_local" / "fidelity"
OUT = Path(__file__).parent / "out" / "bb"
OUT.mkdir(parents=True, exist_ok=True)


def pyarcfire(batch: Path, out: Path, settings: dict | None = None) -> list[dict]:
    """Run PyArcFiRe on a batch npz in its own environment; resumable through `out`."""
    cmd = ["uv", "run", "--project", str(PYARC), "python", str(PYARC / "scripts" / "bb_measure.py"),
           str(batch), str(out)]
    if settings:
        s = out.with_suffix(".settings.json")
        s.write_text(json.dumps(settings))
        cmd += ["--settings", str(s)]
    subprocess.run(cmd, check=True, cwd=PYARC)
    return [json.loads(line) for line in out.read_text().splitlines() if line.strip()]


# ── BB0a: fidelity ───────────────────────────────────────────────────────────────────────────────

def signed_dco(rec: dict) -> float:
    """PyArcFiRe's DCO pitch with the dominant sign restored (the estimator returns |pitch|)."""
    if "error" in rec or not rec.get("arcs") or not np.isfinite(rec.get("dco", np.nan)):
        return np.nan
    p = np.array([a[0] for a in rec["arcs"]])
    w = np.array([a[1] for a in rec["arcs"]])
    vote = np.sign(w[p > 0].sum() - w[p < 0].sum())
    return float(vote * rec["dco"]) if vote else np.nan


def fidelity_state(ref: np.ndarray, py: np.ndarray) -> dict:
    """First match applies (bb_findings.md §BB0a). ref, py: signed DCO pitch; NaN in py = failure."""
    fail = ~np.isfinite(py)
    ev = ~fail & np.isfinite(ref)
    d = np.abs(py[ev]) - np.abs(ref[ev])
    agree = int(np.sum(np.sign(py[ev]) == np.sign(ref[ev])))
    n_ev = int(ev.sum())
    med = float(np.median(np.abs(d))) if n_ev else np.inf
    within5 = float(np.mean(np.abs(d) <= 5)) if n_ev else 0.0
    rho = float(spearmanr(np.abs(py[ev]), np.abs(ref[ev]))[0]) if n_ev > 2 else np.nan
    if fail.sum() > 5 or agree < 18 or med > 6 or not rho >= 0.5:
        s = "BROKEN"
    elif fail.sum() <= 2 and agree >= 0.9 * n_ev and med <= 2 and within5 >= 0.8:
        s = "FAITHFUL"
    else:
        s = "DIVERGENT"
    return {"state": s, "n": int(ref.size), "failures": int(fail.sum()), "evaluable": n_ev,
            "chirality_agree": agree, "median_abs_delta": med, "within_5deg": within5, "spearman_abs": rho,
            "median_delta_signed": float(np.median(d)) if n_ev else np.nan}


def fidelity_inputs() -> tuple[list[str], np.ndarray]:
    ref = json.loads((FID / "reference.json").read_text())
    names = [str(n) for n in np.load(FID / "batch.npz")["names"]]
    return names, np.array([float(ref[n]["pa_alenWtd_avg_domChiralityOnly"]) for n in names])


def plant_fidelity() -> dict:
    names, ref = fidelity_inputs()
    rng = np.random.default_rng(0)
    noisy = ref + rng.normal(0, 3, ref.size)
    noisy[:2] = np.nan
    out = {"state_logic": {
        "self": fidelity_state(ref, ref)["state"],
        "noise 3deg + 2 failures": fidelity_state(ref, noisy)["state"],
        "permuted": fidelity_state(ref, np.sign(ref) * np.abs(rng.permutation(ref)))["state"],
        "chirality flipped": fidelity_state(ref, -ref)["state"],
        "bias +3deg": fidelity_state(ref, np.sign(ref) * (np.abs(ref) + 3))["state"],
        "6 failures": fidelity_state(ref, np.where(np.arange(ref.size) < 6, np.nan, ref))["state"]}}
    # End-to-end: mirrored inputs through PyArcFiRe. Only chirality and the state are printed.
    b = np.load(FID / "batch.npz", allow_pickle=True)
    mb = FID / "batch_mirrored.npz"
    np.savez(mb, names=b["names"], images=np.stack([np.fliplr(x) for x in b["images"]]))
    recs = {r["name"]: r for r in pyarcfire(mb, FID / "py_mirrored.jsonl")}
    py = np.array([signed_dco(recs[n]) for n in names])
    st = fidelity_state(ref, py)
    out["mirrored_end_to_end"] = {k: st[k] for k in ("state", "chirality_agree", "evaluable", "failures")}
    noisy_rho = fidelity_state(ref, noisy)["spearman_abs"]
    out["noise_plant_spearman"] = noisy_rho
    out["seconds_per_galaxy"] = float(np.median([r["seconds"] for r in recs.values()]))
    return out


def fidelity() -> dict:
    names, ref = fidelity_inputs()
    recs = {r["name"]: r for r in pyarcfire(FID / "batch.npz", FID / "py.jsonl")}
    py = np.array([signed_dco(recs[n]) for n in names])
    refj = json.loads((FID / "reference.json").read_text())
    out = fidelity_state(ref, py)
    allpy = np.array([recs[n].get("all_arcs", np.nan) for n in names])
    allref = np.array([float(refj[n]["pa_alenWtd_avg_abs"]) for n in names])
    ok2 = np.isfinite(allpy)
    out["all_arcs_median_abs_delta"] = float(np.median(np.abs(allpy[ok2] - allref[ok2])))
    out["per_galaxy"] = {n: {"ref": float(r), "py": float(p), "n_arcs_ref": len(refj[n]["arcs"]),
                             "n_arcs_py": recs[n].get("n_arcs"), "error": recs[n].get("error")}
                         for n, r, p in zip(names, ref, py, strict=True)}
    out["seconds_per_galaxy"] = float(np.median([r["seconds"] for r in recs.values()]))
    return out


# ── BB0b: the Fourier method ─────────────────────────────────────────────────────────────────────

P2 = TOOLS / "p2dfft-6.2-src"
P2PA_ENV = ["uv", "run", "--no-project", "-q", "--with",
            "h5py,numpy<2,pandas,scipy,astropy,matplotlib,tqdm", "python", str(P2 / "p2pa")]


def p2dfft(work: Path, fits_names: list[str]) -> None:
    """Run p2dfft on FITS files in `work` (skips those already with an .h5)."""
    todo = [f for f in fits_names if not (work / (Path(f).stem + ".h5")).exists()]
    for i in range(0, len(todo), 50):
        subprocess.run([str(P2 / "build" / "p2dfft"), *todo[i:i + 50]], cwd=work, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def p2pa(work: Path, stems: list[str], arms: dict[str, int] | None = None) -> dict[str, dict]:
    """PA per stem: amplitude-chosen mode (`-m`), or the supplied arm count per stem (`-a`)."""
    import csv

    out: dict[str, dict] = {}
    groups = {None: stems} if arms is None else {}
    if arms is not None:
        for st in stems:
            groups.setdefault(arms[st], []).append(st)
    for a, sts in groups.items():
        csvp = work / f"p2pa_{'m' if a is None else f'a{a}'}.csv"
        flag = ["-m"] if a is None else ["-a", str(a)]
        for i in range(0, len(sts), 200):
            part = csvp.with_suffix(f".{i}.csv")
            subprocess.run([*P2PA_ENV, *flag, "-o", str(part), *sts[i:i + 200]], cwd=work,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if part.exists():
                for r in csv.DictReader(part.open()):
                    out[r["Name"]] = {"pa": float(r["PA"]) if r["PA"] not in ("", "nan") else np.nan,
                                      "err": float(r["Err"]) if r["Err"] else np.nan,
                                      "mode": int(r["Mode"]) if r["Mode"] else -1}
    return out


def fourier_state(err: np.ndarray) -> dict:
    """First match applies (bb_findings.md §BB0b). err = |PA| − truth; NaN = failure."""
    fail = ~np.isfinite(err)
    e = np.abs(err[~fail])
    med = float(np.median(e)) if e.size else np.inf
    w5 = float(np.mean(e <= 5)) if e.size else 0.0
    if fail.sum() > 16 or med > 5:
        s = "FAILS"
    elif fail.sum() <= 3 and med <= 2 and w5 >= 0.9:
        s = "VALID"
    else:
        s = "BIASED"
    return {"state": s, "n": int(err.size), "failures": int(fail.sum()), "median_abs_err": med,
            "within_5deg": w5, "median_signed_err": float(np.median(err[~fail])) if e.size else np.nan}


def fourier_set() -> tuple[Path, list[dict]]:
    work = OUT / "bb0b_p2spiral"
    work.mkdir(exist_ok=True)
    models = [{"name": f"p{pa}_a{a}_f{f}", "pa": pa, "arms": a, "feather": f}
              for pa in range(10, 50, 5) for a in (1, 2, 3, 4) for f in (2, 5)]
    if not all((work / f"{m['name']}.fits").exists() for m in models):
        lines = ["DEFAULTS:", "vsize=255", "hsize=255", "sweep=180", "MODELS:"]
        for m in models:
            lines += [f"name={m['name']}", f"pa={m['pa']}.0", f"arms={m['arms']}", f"feather={m['feather']}"]
        (work / "sp.txt").write_text("\n".join(lines) + "\n")
        subprocess.run([str(P2 / "build" / "p2spiral"), "-e", "sp.txt"], cwd=work, check=True,
                       stdout=subprocess.DEVNULL)
    p2dfft(work, [f"{m['name']}.fits" for m in models])
    return work, models


def plant_fourier() -> dict:
    rng = np.random.default_rng(0)
    out: dict = {"state_logic": {
        "exact": fourier_state(np.zeros(64))["state"],
        "noise 1deg": fourier_state(rng.normal(0, 1, 64))["state"],
        "bias 3deg": fourier_state(np.full(64, 3.0))["state"],
        "20 failures": fourier_state(np.where(np.arange(64) < 20, np.nan, 0.0))["state"],
        "noise 10deg": fourier_state(rng.normal(0, 10, 64))["state"]}}
    work, models = fourier_set()
    got = p2pa(work, [m["name"] for m in models])
    pa = np.array([abs(got.get(m["name"], {}).get("pa", np.nan)) for m in models])
    truth = np.array([m["pa"] for m in models], float)
    out["shuffled_truth"] = fourier_state(pa - rng.permutation(truth))["state"]
    from astropy.io import fits

    nd = OUT / "bb0b_noise"
    nd.mkdir(exist_ok=True)
    for i in range(8):
        fits.writeto(nd / f"noise{i}.fits", rng.normal(100, 10, (255, 255)).astype(np.float32),
                     overwrite=True)
    p2dfft(nd, [f"noise{i}.fits" for i in range(8)])
    g = p2pa(nd, [f"noise{i}" for i in range(8)])
    v = [abs(g.get(f"noise{i}", {}).get("pa", np.nan)) for i in range(8)]
    out["noise_images"] = {"abs_pa": v, "failures": int(np.sum(~np.isfinite(v)))}
    return out


def fourier() -> dict:
    work, models = fourier_set()
    stems = [m["name"] for m in models]
    auto = p2pa(work, stems)
    orac = p2pa(work, stems, arms={m["name"]: m["arms"] for m in models})
    truth = np.array([m["pa"] for m in models], float)
    out: dict = {}
    for k, g in (("auto", auto), ("oracle_arms", orac)):
        pa = np.array([abs(g.get(st, {}).get("pa", np.nan)) for st in stems])
        err = pa - truth
        out[k] = fourier_state(err)
        out[k]["by_pitch"] = {int(t): float(np.nanmedian(err[truth == t])) for t in np.unique(truth)}
        out[k]["by_arms"] = {a: float(np.nanmedian(np.abs(err[[m["arms"] == a for m in models]])))
                             for a in (1, 2, 3, 4)}
        out[k]["per_image"] = {st: float(e) for st, e in zip(stems, err, strict=True)}
    out["mode_correct_share"] = float(np.mean([auto.get(m["name"], {}).get("mode") == m["arms"]
                                               for m in models]))
    out["mode_by_arms"] = {a: float(np.mean([auto.get(m["name"], {}).get("mode") == a
                                             for m in models if m["arms"] == a])) for a in (1, 2, 3, 4)}
    return out


# ── BB1 Stage 0: the published toy set ──────────────────────────────────────────────────────────

TOYS = TOOLS / "ht_toys" / "SpArcFiRe-HT-Response" / "input"
PUB_P2 = {("pitch", "TOY"): 1.44, ("pitch", "BAR"): 1.80, ("arms", "TOY"): 1.44, ("arms", "BAR"): 0.89,
          ("width", "TOY"): 1.89, ("width", "BAR"): 2.00, ("sweep", "TOY"): 0.91, ("sweep", "BAR"): 1.84,
          ("bar", "BAR"): 5.41}


def toys() -> list[dict]:
    """Truth from each filename: {TOY|BAR}_{pitch}_a{arms}_f{feather}_{c|b}{n}_{sweep}L."""
    out = []
    for f in sorted(TOYS.glob("*.jpg")):
        kind, pa, a, fe, cb, sw = f.stem.split("_")
        t = {"name": f.stem, "kind": kind, "pitch": float(pa), "arms": int(a[1:]), "feather": int(fe[1:]),
             "cb": cb, "sweep": int(sw[:-1])}
        base = t["pitch"] == 25 and t["arms"] == 2 and t["feather"] == 5 and cb == "c25" and t["sweep"] == 180
        t["series"] = sorted(s for s, ok in (
            ("pitch", t["arms"] == 2 and t["feather"] == 5 and cb == "c25" and t["sweep"] == 180),
            ("arms", t["pitch"] == 25 and t["feather"] == 5 and cb == "c25" and t["sweep"] == 180),
            ("width", t["pitch"] == 25 and t["arms"] == 2 and cb == "c25" and t["sweep"] == 180),
            ("sweep", t["pitch"] == 25 and t["arms"] == 2 and t["feather"] == 5 and cb == "c25"),
            ("bar", kind == "BAR" and cb.startswith("b"))) if ok)
        t["baseline"] = base
        out.append(t)
    return out


def p2_cells(err: dict[str, float], ts: list[dict]) -> dict:
    cells = {}
    for (series, kind), pub in PUB_P2.items():
        e = [abs(err[t["name"]]) for t in ts if series in t["series"] and t["kind"] == kind
             and np.isfinite(err[t["name"]])]
        ours = float(np.mean(e)) if e else np.nan
        tol = max(1.0, 0.3 * pub)
        cells[f"{series}:{kind}"] = {"n": len(e), "ours": ours, "published": pub,
                                     "replicates": bool(ours <= pub + tol),
                                     "better_than_published": bool(ours < pub - tol)}
    return cells


def p2_state(cells: dict) -> str:
    k = sum(c["replicates"] for c in cells.values())
    return "REPRODUCED" if k >= 7 else "NOT REPRODUCED" if k <= 4 else "PARTIAL"


def sf_state(err: np.ndarray) -> dict:
    """SpArcFiRe on the 60: err = |pitch| − truth, NaN = failure."""
    fail = int(np.sum(~np.isfinite(err)))
    e = np.abs(err[np.isfinite(err)])
    w2 = float(np.mean(e <= 2)) if e.size else 0.0
    mx = float(e.max()) if e.size else np.inf
    if fail <= 8 and w2 >= 0.85 and mx <= 5:
        s = "REPRODUCED"
    elif fail > 15 or w2 < 0.6:
        s = "NOT REPRODUCED"
    else:
        s = "PARTIAL"
    return {"state": s, "failures": fail, "within_2deg": w2, "max_err": mx, "n": int(err.size)}


def toy_fits() -> Path:
    from astropy.io import fits
    from PIL import Image

    work = OUT / "stage0_p2dfft"
    work.mkdir(exist_ok=True)
    for f in TOYS.glob("*.jpg"):
        out = work / f"{f.stem}.fits"
        if not out.exists():
            fits.writeto(out, np.asarray(Image.open(f).convert("RGB"), np.float32).mean(2)[::-1],
                         overwrite=True)  # FITS row 0 is the bottom; keep the displayed orientation
    return work


def stage0_p2dfft(shuffle: bool = False) -> dict:
    ts = toys()
    work = toy_fits()
    p2dfft(work, [f"{t['name']}.fits" for t in ts])
    stems = [t["name"] for t in ts]
    orac = p2pa(work, stems, arms={t["name"]: t["arms"] for t in ts})
    auto = p2pa(work, stems)
    truth = {t["name"]: t["pitch"] for t in ts}
    if shuffle:
        sixty = [t["name"] for t in ts if t["series"]]
        perm = np.random.default_rng(0).permutation(sixty)
        truth.update({a: truth[b] for a, b in zip(sixty, perm, strict=True)})
    out: dict = {"n_sixty": sum(bool(t["series"]) for t in ts)}
    for k, g in (("oracle_arms", orac), ("auto", auto)):
        err = {st: abs(g.get(st, {}).get("pa", np.nan)) - truth[st] for st in stems}
        cells = p2_cells(err, ts)
        out[k] = {"state": p2_state(cells), "cells": cells,
                  "all200_median_abs_err": float(np.nanmedian(np.abs(list(err.values())))),
                  "per_toy": err}
    out["auto_mode_correct"] = float(np.mean([auto.get(t["name"], {}).get("mode") == t["arms"] for t in ts]))
    return out


def plant_stage0() -> dict:
    ts = [t for t in toys() if t["series"]]
    z = {t["name"]: 0.0 for t in ts}
    far = {t["name"]: 6.0 for t in ts}
    rng = np.random.default_rng(1)
    out = {"n_sixty": len(ts), "p2_logic": {
        "exact": p2_state(p2_cells(z, ts)),
        "published-size errors": p2_state(p2_cells({t["name"]: PUB_P2.get((t["series"][0], t["kind"]), 1.5)
                                                    for t in ts}, ts)),
        "6deg everywhere": p2_state(p2_cells(far, ts)),
        "6deg on the bar, sweep and width series": p2_state(p2_cells(
            {t["name"]: 6.0 if set(t["series"]) & {"bar", "sweep", "width"} else 0.0 for t in ts}, ts))},
        "sf_logic": {
        "published (5 fail, 54 <2, one 3.2)": sf_state(np.r_[[np.nan] * 5, rng.uniform(0, 2, 54), 3.2])["state"],
        "20 failures": sf_state(np.r_[[np.nan] * 20, np.zeros(40)])["state"],
        "half within 2": sf_state(np.r_[np.full(30, 1.0), np.full(30, 4.0)])["state"],
        "10 fail, rest fine": sf_state(np.r_[[np.nan] * 10, np.full(50, 1.0)])["state"]}}
    sh = stage0_p2dfft(shuffle=True)
    out["p2_shuffled_truth"] = {k: sh[k]["state"] for k in ("oracle_arms", "auto")}
    return out


SF_RUNS = TOOLS / "sparcfire-docker" / "out"  # official SpArcFiRe (BB0c image): <tag>/output/galaxy.tsv
SF_DCO = "pa_alenWtd_avg_domChiralityOnly"


def sf_read(path: Path) -> tuple[dict[str, float], dict[str, str]]:
    """DCO pitch per galaxy from SpArcFiRe's galaxy.tsv, read by header name.

    SpArcFiRe writes a zero-arc galaxy's row with extra padding fields, shifted from the header, and a
    rejected input as a 2-field row. Neither is read positionally: such a row is a failure, and only
    after checking it carries no finite value anywhere past the fixed-width head (so a shifted row
    holding a real pitch would stop the run rather than be dropped)."""
    lines = path.read_text().splitlines()
    head = lines[0].split("\t")
    col = head.index(SF_DCO)
    pa, why = {}, {}
    for ln in lines[1:]:
        f = ln.split("\t")
        if len(f) == len(head):
            pa[f[0]] = float(f[col]) if f[col] not in ("", "NaN") else np.nan
            if not np.isfinite(pa[f[0]]):
                why[f[0]] = "no DCO pitch"
            continue
        tail = f[head.index("pa_longest"):head.index("numArcs_largest_length_gap") + 4]
        if any(re.fullmatch(r"-?\d+\.\d+", v) for v in tail):
            raise ValueError(f"{f[0]}: ragged row ({len(f)} fields) carries a pitch — cannot read it safely")
        pa[f[0]] = np.nan
        why[f[0]] = f[1] if len(f) == 2 else "no arcs (ragged row)"
    return pa, why


def stage0_sparcfire(tag: str) -> dict:
    ts = toys()
    pa, why = sf_read(SF_RUNS / tag / "output" / "galaxy.tsv")
    # no arcs / no DCO pitch / missing row = failure (NaN), as pre-registered
    err = {t["name"]: abs(pa.get(t["name"], np.nan)) - t["pitch"] for t in ts}
    sixty = [t for t in ts if t["series"]]
    per_series = {}
    for series in ("pitch", "arms", "width", "sweep", "bar"):
        for kind in ("TOY", "BAR"):
            e = np.array([err[t["name"]] for t in sixty if series in t["series"] and t["kind"] == kind])
            if e.size:
                per_series[f"{series}:{kind}"] = {"n": int(e.size), "failures": int(np.sum(~np.isfinite(e))),
                                                  "mean_abs_err": float(np.nanmean(np.abs(e)))
                                                  if np.isfinite(e).any() else np.nan}
    a = np.array(list(err.values()))
    return {"tag": tag, "n_rows": len(pa), "failure_reasons": why, "n_sixty": len(sixty),
            "state": sf_state(np.array([err[t["name"]] for t in sixty])),
            "per_series": per_series,
            "all200": {"failures": int(np.sum(~np.isfinite(a))),
                       "median_abs_err": float(np.nanmedian(np.abs(a))),
                       "within_2deg": float(np.mean(np.abs(a[np.isfinite(a)]) <= 2))},
            "per_toy": err}


def main() -> None:
    mode = sys.argv[1]
    if mode == "--stage0-sparcfire":
        tag = sys.argv[2]
        out = stage0_sparcfire(tag)
        (OUT / f"stage0_sparcfire_{tag}.json").write_text(json.dumps(out, indent=1, default=float))
        print(json.dumps({k: v for k, v in out.items() if k != "per_toy"}, indent=1, default=float))
        return
    fn, path = {"--plant-fidelity": (plant_fidelity, "bb0a_planted.json"),
                "--fidelity": (fidelity, "bb0a_fidelity.json"),
                "--plant-fourier": (plant_fourier, "bb0b_planted.json"),
                "--fourier": (fourier, "bb0b_fourier.json"),
                "--plant-stage0": (plant_stage0, "stage0_planted.json"),
                "--stage0-p2dfft": (stage0_p2dfft, "stage0_p2dfft.json")}[mode]
    out = fn()
    (OUT / path).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "per_galaxy"}, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
