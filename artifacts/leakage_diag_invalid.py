"""(2026-10-06) Diagnose the leakage audit's INVALID: data only, nothing fitted beyond a phase-slope shift measurement.
Reads the audit's own memmaps (original and re-injected) and recomputes its intended offsets."""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/Users/malachy/Documents/Galaxy-JEPA")
sys.path.insert(0, str(REPO / "artifacts"))
import leakage_audit as la  # noqa: E402

N = 200
rep = json.load(open(la.OUT / "audit.json"))
S = 256
yy, xx = np.meshgrid(np.fft.fftfreq(S), np.fft.fftfreq(S), indexing="ij")
w = np.outer(np.hanning(S), np.hanning(S))
band = (np.hypot(yy, xx) > 0) & (np.hypot(yy, xx) < 0.12)


def measure(a, b):
    """(dy, dx) such that b ≈ a moved by (dy, dx): weighted LS on the cross-power phase."""
    A, B = np.fft.fft2(w * (a - a.mean())), np.fft.fft2(w * (b - b.mean()))
    C = B * np.conj(A)
    ph, wt = -np.angle(C[band]) / (2 * np.pi), np.abs(C[band])
    M = np.stack([yy[band], xx[band]], 1) * np.sqrt(wt)[:, None]
    return np.linalg.lstsq(M, ph * np.sqrt(wt), rcond=None)[0]


t0 = time.time()
logs = {c: pd.read_csv(REPO / "data" / c / "cut_log.csv", dtype={"object_id": str}, low_memory=False) for c in la.CORPORA}
out = {"cut_log": {}}
for c in la.CORPORA:
    p = REPO / "data" / c / "cut_log.csv"
    out["cut_log"][c] = {"rows": len(logs[c]), "mtime": time.strftime("%F %T", time.localtime(p.stat().st_mtime)),
                         "complete_now": la._complete(c)}
samples = la._samples(logs)
md = {c: pd.read_csv(REPO / "data" / c / "metadata.csv", usecols=["object_id"], dtype={"object_id": str}) for c in la.CORPORA}
phys = la._physics([o for c in la.CORPORA for o in samples[c].object_id])

# calibrate the measurement on real stamps with known shifts made by reinject itself
orig0 = np.load(la.OUT / "audit_probe_v2.f16", mmap_mode="r")
rng = np.random.default_rng(1)
cal = []
for k in range(10):
    x = np.asarray(orig0[k], np.float64)
    off = rng.uniform(-0.5, 0.5, (3, 2))
    y = la.reinject(x, off)
    cal.append([(off[b], measure(x[b], y[b])) for b in range(3)])
cal_err = np.array([[m - o for o, m in r] for r in cal])
out["calibration"] = {"n": 10, "max_abs_err_px": float(np.abs(cal_err).max()), "rms_err_px": float(np.sqrt((cal_err ** 2).mean()))}
# the fp16 store's own effect (the memmaps are fp16)
cal16 = []
for k in range(10):
    x = np.asarray(orig0[k], np.float64)
    off = rng.uniform(-0.5, 0.5, (3, 2))
    y = la.reinject(x, off).astype(np.float16).astype(np.float64)
    cal16 += [measure(x[b], y[b]) - off[b] for b in range(3)]
out["calibration"]["fp16_rms_err_px"] = float(np.sqrt((np.array(cal16) ** 2).mean()))

out["corpora"] = {}
for c in la.CORPORA:
    lg = samples[c].merge(phys, on="object_id", how="left").merge(md[c], on="object_id", how="left")
    ids = lg.object_id.tolist()
    sha_ok = hashlib.sha1(",".join(ids).encode()).hexdigest() == rep["sample_sha1"][c]
    tb = la._targets(lg)
    off = tb["_v1_off"]  # (n, 3, 2) = (dy, dx) per band, the check run's intended shift
    # row alignment: re-derive the offsets for the first N by object ID straight from the cut log
    cl = logs[c].set_index("object_id").loc[ids[:N]]
    by_id = np.stack([np.stack([cl[f"{b}_v1_rely"].to_numpy(float) - 127.5, cl[f"{b}_v1_relx"].to_numpy(float) - 127.5], 1)
                      for b in la.BANDS], 1)
    o = np.load(la.OUT / f"audit_{c}.f16", mmap_mode="r")
    r = np.load(la.OUT / f"audit_{c}_reinjected.f16", mmap_mode="r")
    meas = np.array([[measure(np.asarray(o[k, b], np.float64), np.asarray(r[k, b], np.float64)) for b in range(3)]
                     for k in range(N)])  # (N, 3, 2)
    res = {"sample_order_matches_audit_record": sha_ok, "offsets_by_id_equal_audit_rows": bool(np.array_equal(by_id, off[:N])),
           "per_band_axis": {}}
    for bi, b in enumerate(la.BANDS):
        for ai, a in enumerate(("y", "x")):
            i_, m_ = off[:N, bi, ai], meas[:, bi, ai]
            s_ = tb[f"{b}_s{a}"]
            v1 = off[:, bi, ai]
            res["per_band_axis"][f"{b}_{a}"] = {
                "pixels_intended_vs_measured": {"slope": float(np.polyfit(i_, m_, 1)[0]), "r": float(np.corrcoef(i_, m_)[0, 1]),
                                                "rms_resid_px": float(np.sqrt(((m_ - i_) ** 2).mean())),
                                                "frac_measured_abs_lt_0.02px": float((np.abs(m_) < 0.02).mean()),
                                                "intended_sd": float(i_.std()), "measured_sd": float(m_.std())},
                "full_sample_v1_offset_vs_s_b": {"r": float(np.corrcoef(v1, s_)[0, 1]),
                                                 "slope_on_minus_s": float(np.polyfit(-s_, v1, 1)[0]),
                                                 "v1_off_sd": float(v1.std()), "s_sd": float(s_.std()),
                                                 "frac_v1_off_abs_lt_0.02": float((np.abs(v1) < 0.02).mean())}}
    out["corpora"][c] = res
    print(c, "done", round(time.time() - t0), "s", flush=True)
out["seconds"] = round(time.time() - t0)
(la.OUT / "diag_invalid.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
