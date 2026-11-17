"""Ladder subset (a) on M and O2 under the current code → docs/ladder_v2_m/manifest.yaml.

    .venv/bin/python artifacts/ladder_v2_m_manifest.py
    .venv/bin/python artifacts/ladder_v2_run.py --manifest docs/ladder_v2_m/manifest.yaml --plan

The comparison with M (docs/m_comparison/prereg.md; user, 17 Nov, decisions 1–2) reads M's and O2's ladders
beside A1–A3's. M's own records in `artifacts/out` predate three changes to the ladder path, so both v1
encoders are re-laddered here with ladder v2's own commands (`ladder_v2_manifest.Builder`), the same
existence subset as the baselines (E3, E7 bank, E5 p2, E7 r0, E5 s2_verify, E5 t1), on the identical
split. M's original `t1_ladder.json` is kept for a pass/fail reproduction check only.

**What differs from ladder v2, and why.** M and O2 read v1 pixels (`configs/pretrain.yaml`, the v1 cache
and freeze), so the untrained bars cannot be ladder v2's: the K = 30 bank (E4) and the untrained seeds
1–2 bank (E7) are rebuilt on v1 pixels and the identical split, once, into `shared_v1` and M's directory
(O2 links M's, as A2 linked A1's: architecture + pixels + split only). The identical split drops
`ladder_env.NINE` on v1 too (`j4_spread_controls._minus_nine`), so all five encoders read the same
160,849 + 34,828 galaxies.

Investigation code: terse, excluded from lint/CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ladder_v2_manifest as M  # noqa: E402

ROOT = "runs/ladder_v2"
V1_ENV = {
    "GJ_CONFIG": "configs/pretrain.yaml",
    "F0_CACHE_BASE": "/Volumes/X10 Pro/galaxy-jepa/runs/full/cache",
    "F0_NORM_PREFIX": "75100066b3e0",
    "LADDER_SPLIT": "identical",
    "PYTHONUNBUFFERED": "1",
    "OMP_NUM_THREADS": "4",
    "LADDER_UNTRAINED_BANK": f"{ROOT}/shared_v1/p1_untrained_bank.json",
}
EXCLUDED = {
    "E0–E2, E7 r1–r3, E8–E24, E26": "outside subset (a), as the baselines",
    "M on probe_v2": "deferred (user, 17 Nov, decision 3)",
}
OUT = M.REPO / "docs/ladder_v2_m/manifest.yaml"


class V1(M.Builder):
    def bank(self) -> str:
        return self.add("E4.p1_untrained_bank.shared_v1", exp="E4", enc="shared_v1", script="p1_untrained_bank.py",
                        args=["--seeds", "30", "--checkpoint", M.ENCODERS["m"]], heavy="mps", est_h=6.0,
                        env={"LADDER_ENCODER": M.ENCODERS["m"], "LADDER_OUT": f"{self.root}/shared_v1",
                             "LADDER_TAG": "shared_v1"},
                        expect=["p1_untrained_bank.json"], keep=["p1_untrained_bank.json"])

    def encoder(self, enc: str, first: str, e4: str) -> None:
        a, ckpt = self.add, M.ENCODERS[enc]
        r = ["--checkpoint", ckpt]
        e3 = a(f"E3.o1_matched.{enc}", exp="E3", enc=enc, script="o1_matched.py", args=[*r, "--tag", "o1"],
               heavy="mps", est_h=0.6, expect=["o1_embeddings.npz", "o1_matched.json"])
        if enc == first:
            e7 = a(f"E7.r_bank.{enc}", exp="E7", enc=enc, script="r_nonlinear.py", args=[*r, "--stages", "bank"],
                   inputs=[e3], heavy="mps", est_h=0.4, expect=["r_untrained_seeds.npz"])
        else:
            e7 = a(f"E7.r_bank_link.{enc}", exp="E7", enc=enc, inputs=[f"E7.r_bank.{first}", e3],
                   link={"src": f"{self.out(first)}/r_untrained_seeds.npz", "dst": "r_untrained_seeds.npz"})
        base = [e3, e7, e4]
        p2 = a(f"E5.p2_ladder.{enc}", exp="E5", enc=enc, script="p2_ladder.py", args=r, inputs=base,
               heavy="mps", est_h=3.0, expect=["p2_ladder.json"])
        r0 = a(f"E7.r0.{enc}", exp="E7", enc=enc, script="r_nonlinear.py", args=[*r, "--stages", "r0"],
               inputs=[*base, p2], est_h=0.5, expect=["r_r0.json"])
        s2 = a(f"E5.s2_verify.{enc}", exp="E5", enc=enc, script="s2_verify.py", inputs=[*base, p2, r0],
               heavy="mps", est_h=3.0, expect=["s2_ladder.json", "s2_noise.npz"])
        a(f"E5.t1_ladder.{enc}", exp="E5", enc=enc, script="t1_ladder.py", inputs=[*base, p2, s2],
          est_h=4.0, expect=["t1_ladder.json"])


def main() -> None:
    M.ENCODERS.clear()
    M.ENCODERS.update({"m": "runs/m/encoder.pt", "o2": "runs/o2/encoder.pt"})
    M.COMMON_ENV.clear()
    M.COMMON_ENV.update(V1_ENV)
    b = V1(ROOT)
    for d in ("shared_v1", "m", "o2"):
        if (M.REPO / b.out(d)).exists():
            raise SystemExit(f"m manifest: {b.out(d)} exists; each starts from an empty directory")
    e4 = b.bank()
    for enc in ("m", "o2"):
        b.encoder(enc, "m", e4)
    ids = [e["id"] for e in b.entries]
    assert len(set(ids)) == len(ids)
    for e in b.entries:
        assert all(i in ids[: ids.index(e["id"])] for i in e["inputs"]), e["id"]
    doc = {"version": 1, "root": ROOT, "common_env": M.COMMON_ENV,
           "scope": "comparison with M (17 Nov): existence subset (a) on M and O2, v1 pixels, identical split",
           "excluded": EXCLUDED, "not_implemented": M.NOT_IMPLEMENTED, "entries": b.entries}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("# Generated by artifacts/ladder_v2_m_manifest.py — do not edit by hand.\n"
                   + yaml.safe_dump(doc, sort_keys=False, width=110, allow_unicode=True))
    print(f"m manifest: {len(b.entries)} entries → {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
