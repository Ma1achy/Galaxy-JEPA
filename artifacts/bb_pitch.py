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


def main() -> None:
    mode = sys.argv[1]
    fn, path = {"--plant-fidelity": (plant_fidelity, "bb0a_planted.json"),
                "--fidelity": (fidelity, "bb0a_fidelity.json"),
                "--plant-fourier": (plant_fourier, "bb0b_planted.json"),
                "--fourier": (fourier, "bb0b_fourier.json")}[mode]
    out = fn()
    (OUT / path).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "per_galaxy"}, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
