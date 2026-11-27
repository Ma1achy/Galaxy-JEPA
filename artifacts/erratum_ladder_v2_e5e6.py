"""Erratum re-run driver for Ladder v2 E5 (conditional bank) and E6 (2A pair decorrelation).

Why this exists (2026-11-26 audit):
    1. **E5 conditional population:** ``docs/ladder_v2`` ran ``population="conditional"`` against
       the full-population K=30 untrained bank ``p1_untrained_bank.json``. Under the corrected
       ``probing.ladder`` package, each population requires a bank keyed on its own split and
       population (``untrained_split_key(..., population="conditional")``).
    2. **E6 (2A pair cross-check):** ``probing.ladder._entangled_map`` previously checked whether
       feature A's AUC remained detectable when matched on B (testing A's retention rather than
       whether A and B's directions stayed aligned after decorrelation), and did not restrict to
       jointly eligible rows or isolate zero point-masses in ``stratified_match``.

This script leaves ``runs/ladder_v2/`` and ``docs/ladder_v2/`` completely untouched. It writes
exclusively to ``--out-dir`` (default ``docs/ladder_v2_erratum``):

    # 1. Inspect plan and keys (fast, no GPU/MPS compute):
    .venv/bin/python artifacts/erratum_ladder_v2_e5e6.py --plan

    # 2. Build the K=30 conditional-population untrained bank (resumable per seed):
    .venv/bin/python artifacts/erratum_ladder_v2_e5e6.py --stage bank --seeds 30

    # 3. Re-run E5 + E6 for A1, A2, A3 against the fixed package + conditional bank:
    .venv/bin/python artifacts/erratum_ladder_v2_e5e6.py --stage ladder --encoders a1,a2,a3
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "artifacts"))

ENCODERS: dict[str, str] = {
    "a1": "runs/m_v2/encoder.pt",
    "a2": "runs/m_v2_s1/encoder.pt",
    "a3": "runs/m_v2_s2/encoder.pt",
}
DEFAULT_OUT = REPO / "docs" / "ladder_v2_erratum"
DEFAULT_FULL_BANK = REPO / "runs" / "ladder_v2" / "shared" / "p1_untrained_bank.json"
MAX_TRAIN = 40_000


def _key(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def _sanitise(obj: object) -> object:
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v) for v in obj]
    return obj


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _load_or_init_bank(
    bank_path: Path, *, model_key: str, split_key: str, pixels_key: str, population: str
) -> dict:
    if not bank_path.exists():
        return {
            "model_key": model_key,
            "split_key": split_key,
            "pixels_key": pixels_key,
            "population": population,
            "max_train": MAX_TRAIN,
            "seeds": {},
        }
    rec = json.loads(bank_path.read_text())
    for name, want, got in (
        ("model", model_key, rec.get("model_key")),
        ("split", split_key, rec.get("split_key")),
        ("pixels", pixels_key, rec.get("pixels_key")),
        ("population", population, rec.get("population")),
    ):
        if got != want:
            raise SystemExit(
                f"erratum bank {bank_path} mismatch on {name}: found {got!r}, expected {want!r}"
            )
    return rec


def run_bank(args: argparse.Namespace, *, population: str = "conditional") -> None:
    from j4_spread_controls import _release, prepare

    from galaxy_jepa.models.vit import load_frozen_encoder
    from galaxy_jepa.probing import controls as ctl
    from galaxy_jepa.probing.extract import feature_embeddings
    from galaxy_jepa.probing.ladder import untrained_split_key
    from galaxy_jepa.probing.nulls import K_MIN

    ckpt = ENCODERS["a1"]
    setup = prepare(ckpt, MAX_TRAIN, label=f"ERRATUM-BANK-{population.upper()}", sources=1)
    pc, device = setup.pc, setup.device
    pop_labels = (
        setup.labels if population == "full" else setup.labels.with_population(population)
    )
    train_ids, test_ids, ds = setup.train_ids, setup.test_ids, setup.ds
    features = list(pop_labels.features)

    frozen = load_frozen_encoder(setup.ckpt)
    model_key = _key(dict(frozen.config))
    split_key = untrained_split_key(train_ids, test_ids, population=population)
    pixels_key = setup.cfg.normalisation.content_hash

    bank_path = Path(args.out_dir) / f"p1_untrained_bank_{population}.json"
    rec = _load_or_init_bank(
        bank_path,
        model_key=model_key,
        split_key=split_key,
        pixels_key=pixels_key,
        population=population,
    )
    have = sorted(int(s) for s in rec["seeds"])
    todo = [s for s in range(args.seeds) if s not in have]

    print(f"{population.capitalize()} bank: {bank_path}", file=sys.stderr)
    print(f"  features={len(features)} split={len(train_ids):,}/{len(test_ids):,}", file=sys.stderr)
    print(f"  model_key={model_key} split_key={split_key} K_MIN={K_MIN}", file=sys.stderr)
    print(f"  have={len(have)} todo={len(todo)} -> target K={args.seeds}", file=sys.stderr)
    if args.plan:
        return

    for idx, seed in enumerate(todo, start=1):
        t0 = time.perf_counter()
        _release(device)
        mat = ctl.untrained_encoder_matrix(frozen.config, ds, device=device, seed=seed)
        aucs: dict[str, float] = {}
        for feat in features:
            tr = feature_embeddings(mat, pop_labels, feat, train_ids)
            te = feature_embeddings(mat, pop_labels, feat, test_ids)
            aucs[feat] = float(ctl._safe_auc(tr, te, c=pc.c))
        del mat
        _release(device)
        rec["seeds"][str(seed)] = aucs
        _atomic_write_json(bank_path, _sanitise(rec))
        dt = time.perf_counter() - t0
        print(f"  seed {seed:2d}: {dt:.0f}s ({idx}/{len(todo)})", file=sys.stderr)


def _legacy_split_key(train_ids: list[int], test_ids: list[int]) -> str:
    """Historical prefix-only split_key used when runs/ladder_v2/shared/p1_untrained_bank.json was built."""
    union = sorted({*(int(i) for i in train_ids), *(int(i) for i in test_ids)})
    key = [len(train_ids), len(test_ids), union[:1], union[-1:]]
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _resolve_full_bank(
    full_bank: Path,
    out_dir: Path,
    *,
    train_ids: list[int],
    test_ids: list[int],
    new_split_key: str,
    split_provenance: Path | None = None,
) -> Path:
    """Return a full-population bank matching ``new_split_key`` without mutating ``full_bank``.

    If ``full_bank`` already carries ``new_split_key`` (full-membership hash) and
    ``population == 'full'``, it is used directly.

    If ``full_bank`` carries only the historical legacy fingerprint
    (``[len(train), len(test), union[:1], union[-1:]]``), matching that fingerprint does **not**
    establish train/test partition membership. Upgrading therefore requires a trusted historical
    ``split_provenance`` JSON bound to ``full_bank`` via ``bank_sha256``, ``population == 'full'``,
    ``legacy_split_key``, and exact verified train/test membership (or verified full-membership
    ``full_split_key``). Without verified provenance, fails loudly with a rebuild-required error.
    """
    from galaxy_jepa.probing.ladder import untrained_split_key

    rec = json.loads(full_bank.read_text())
    if rec.get("population", "full") != "full":
        raise SystemExit(f"full bank {full_bank} has non-full population {rec.get('population')!r}")
    if rec.get("split_key") == new_split_key:
        return full_bank

    legacy_key = _legacy_split_key(train_ids, test_ids)
    if rec.get("split_key") != legacy_key:
        raise SystemExit(
            f"rebuild required: full bank {full_bank} split_key {rec.get('split_key')!r} matches "
            f"neither full-membership key {new_split_key!r} nor legacy fingerprint {legacy_key!r}."
        )

    if split_provenance is None or not split_provenance.is_file():
        raise SystemExit(
            f"rebuild required: legacy full bank {full_bank} carries only prefix fingerprint "
            f"{legacy_key!r} and lacks verified full-membership split provenance. Supply "
            f"--full-bank-split-provenance <trusted_split_json> or rebuild cleanly via "
            f"--stage full-bank."
        )

    prov = json.loads(split_provenance.read_text())
    if not isinstance(prov, dict):
        raise SystemExit(f"rebuild required: split provenance {split_provenance} must be a JSON object")
    if prov.get("population") != "full":
        raise SystemExit(
            f"rebuild required: split provenance {split_provenance} has population "
            f"{prov.get('population')!r}, expected 'full'"
        )
    actual_sha = _sha256_file(full_bank)
    if prov.get("bank_sha256") != actual_sha:
        raise SystemExit(
            f"rebuild required: split provenance {split_provenance} bank_sha256 "
            f"{prov.get('bank_sha256')!r} does not match {full_bank} ({actual_sha!r})"
        )
    if prov.get("legacy_split_key") != rec.get("split_key"):
        raise SystemExit(
            f"rebuild required: split provenance {split_provenance} legacy_split_key "
            f"{prov.get('legacy_split_key')!r} does not match bank key {rec.get('split_key')!r}"
        )

    has_ids = "train_ids" in prov and "test_ids" in prov
    has_full_key = isinstance(prov.get("full_split_key"), str) and bool(prov.get("full_split_key"))
    if not has_ids and not has_full_key:
        raise SystemExit(
            f"rebuild required: split provenance {split_provenance} must contain verified "
            f"('train_ids', 'test_ids') or 'full_split_key'"
        )
    if has_ids:
        hist_tr = [int(i) for i in prov["train_ids"]]
        hist_te = [int(i) for i in prov["test_ids"]]
        if _legacy_split_key(hist_tr, hist_te) != rec.get("split_key"):
            raise SystemExit(
                f"rebuild required: historical IDs in {split_provenance} do not match legacy key "
                f"{rec.get('split_key')!r}"
            )
        hist_full_key = untrained_split_key(hist_tr, hist_te, population="full")
        if has_full_key and prov["full_split_key"] != hist_full_key:
            raise SystemExit(
                f"rebuild required: split provenance {split_provenance} full_split_key "
                f"contradicts its embedded train_ids/test_ids"
            )
        if sorted(hist_tr) != sorted(int(i) for i in train_ids) or sorted(hist_te) != sorted(
            int(i) for i in test_ids
        ):
            raise SystemExit(
                f"rebuild required: historical split membership in {split_provenance} does not "
                f"match current train/test partition despite sharing legacy fingerprint "
                f"{legacy_key!r}"
            )
        if hist_full_key != new_split_key:
            raise SystemExit(
                f"rebuild required: historical full_split_key {hist_full_key!r} != current "
                f"{new_split_key!r}"
            )
    elif prov["full_split_key"] != new_split_key:
        raise SystemExit(
            f"rebuild required: split provenance full_split_key {prov['full_split_key']!r} "
            f"does not match current {new_split_key!r}"
        )

    rekeyed = dict(rec)
    rekeyed["split_key"] = new_split_key
    rekeyed["legacy_split_key"] = legacy_key
    rekeyed["population"] = "full"
    rekeyed["rekeyed_from"] = str(full_bank)
    rekeyed["source_bank_sha256"] = actual_sha
    rekeyed["split_provenance"] = str(split_provenance)
    dst = out_dir / "p1_untrained_bank_full_rekeyed.json"
    _atomic_write_json(dst, _sanitise(rekeyed))
    return dst


def run_ladder_erratum(args: argparse.Namespace) -> None:
    from j4_spread_controls import _release, prepare

    from galaxy_jepa.models.vit import load_frozen_encoder
    from galaxy_jepa.probing import controls as ctl
    from galaxy_jepa.probing import ladder as ladder_mod
    from galaxy_jepa.probing.extract import extract_matrix

    out_dir = Path(args.out_dir)
    cond_bank = out_dir / "p1_untrained_bank_conditional.json"
    full_bank = Path(args.full_bank)
    enc_list = [e.strip() for e in args.encoders.split(",") if e.strip()]

    print(f"Erratum ladder re-run -> {out_dir}", file=sys.stderr)
    print(f"  encoders  : {enc_list}", file=sys.stderr)
    print(f"  full bank : {full_bank}", file=sys.stderr)
    print(f"  cond bank : {cond_bank}", file=sys.stderr)
    if args.plan:
        return

    if not full_bank.is_file():
        raise SystemExit(f"missing full-population bank: {full_bank}")
    if not cond_bank.is_file():
        raise SystemExit(f"missing conditional bank: {cond_bank} (run --stage bank first)")

    diffs: dict[str, object] = {}
    for enc in enc_list:
        if enc not in ENCODERS:
            raise SystemExit(f"unknown encoder {enc!r}; expected one of {sorted(ENCODERS)}")
        setup = prepare(ENCODERS[enc], MAX_TRAIN, label=f"ERRATUM-{enc.upper()}", sources=3)
        cfg, device = setup.cfg, setup.device
        labels, train_ids, test_ids, ds = setup.labels, setup.train_ids, setup.test_ids, setup.ds
        full_split_key = ladder_mod.untrained_split_key(train_ids, test_ids, population="full")
        prov_path = (
            Path(args.full_bank_split_provenance)
            if getattr(args, "full_bank_split_provenance", None)
            else None
        )
        resolved_full_bank = _resolve_full_bank(
            full_bank,
            out_dir,
            train_ids=list(train_ids),
            test_ids=list(test_ids),
            new_split_key=full_split_key,
            split_provenance=prov_path,
        )
        pc = setup.pc.model_copy(
            update={
                "untrained_bank_path": str(resolved_full_bank),
                "conditional_untrained_bank_path": str(cond_bank),
            }
        )
        frozen = load_frozen_encoder(setup.ckpt)

        real = extract_matrix(frozen, ds, device=device)
        _release(device)
        untrained = ctl.untrained_encoder_matrix(frozen.config, ds, device=device, seed=cfg.seed)
        _release(device)
        noise = ctl.noise_through_encoder_matrix(frozen, ds, device=device, seed=cfg.seed)
        _release(device)
        extra = tuple(
            ctl.untrained_encoder_matrix(frozen.config, ds, device=device, seed=cfg.seed + k)
            for k in range(1, 3)
        )
        controls = ctl.ControlEmbeddings(
            real=real, untrained=untrained, noise=noise, untrained_extra=extra
        )

        rec: dict[str, object] = {
            "encoder": enc,
            "checkpoint": str(setup.ckpt),
            "full_bank": str(resolved_full_bank),
            "conditional_bank": str(cond_bank),
        }
        for pop in ("full", "conditional"):
            lab = labels if pop == "full" else labels.with_population("conditional")
            res = ladder_mod.run_ladder(
                controls, lab, train_ids, test_ids, config=pc, sky_label_col="snr_r"
            )
            rec[pop] = [
                {
                    "feature": f,
                    "rung": v.rung,
                    "mechanism": v.mechanism,
                    "auc": v.metrics.get("auc"),
                    "exceeds_null": res.existence[f].exceeds_null,
                    "clean": res.existence[f].clean,
                    "n_test": v.n_test,
                    "positives_test": v.positives_test,
                    "real_se": v.real_se,
                    "resolvable_margin": v.resolvable_margin,
                    "underpowered": v.underpowered,
                }
                for f, v in res.verdicts.items()
            ]
            rec[f"{pop}_pair_verdicts"] = [dataclasses.asdict(pv) for pv in res.pair_verdicts]

        out_path = out_dir / f"{enc}_p2_ladder.json"
        _atomic_write_json(out_path, _sanitise(rec))

        orig_path = REPO / "runs" / "ladder_v2" / enc / "p2_ladder.json"
        if orig_path.is_file():
            orig = json.loads(orig_path.read_text())
            enc_diff: dict[str, list[dict[str, object]]] = {}
            for pop in ("full", "conditional"):
                old_by_f = {r["feature"]: r["rung"] for r in orig.get(pop, [])}
                new_by_f = {r["feature"]: r["rung"] for r in rec[pop]}  # type: ignore[union-attr]
                enc_diff[pop] = [
                    {"feature": f, "original": old_by_f.get(f), "erratum": new_by_f[f]}
                    for f in new_by_f
                    if old_by_f.get(f) != new_by_f[f]
                ]
            diffs[enc] = enc_diff

    if diffs:
        _atomic_write_json(out_dir / "diff_summary.json", _sanitise(diffs))


def main() -> None:
    ap = argparse.ArgumentParser(description="Ladder v2 E5/E6 erratum re-run driver.")
    ap.add_argument("--stage", choices=("bank", "full-bank", "ladder", "all"), default="all")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--encoders", default="a1,a2,a3")
    ap.add_argument("--full-bank", default=str(DEFAULT_FULL_BANK))
    ap.add_argument(
        "--full-bank-split-provenance",
        default=None,
        help=(
            "Optional path to a trusted historical split-provenance JSON (binding bank_sha256, "
            "population='full', legacy_split_key, and exact train_ids/test_ids or full_split_key) "
            "required to upgrade a legacy prefix-keyed full bank without rebuilding."
        ),
    )
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--plan", action="store_true", help="print plan and exit without running")
    args = ap.parse_args()

    if args.stage == "full-bank":
        run_bank(args, population="full")
    if args.stage in ("bank", "all"):
        run_bank(args, population="conditional")
    if args.stage in ("ladder", "all"):
        run_ladder_erratum(args)


if __name__ == "__main__":
    main()
