"""Ladder v2 manifest generator (docs/ladder_v2/plan.md, Task 2) → docs/ladder_v2/manifest.yaml.

    .venv/bin/python artifacts/ladder_v2_manifest.py [--out PATH] [--root DIR] [--encoders a1,a2,a3]

One entry per (experiment × encoder [× mode]), shared entries once, in the order the runner executes:
the shared untrained bank (E4), then A1, A2, A3 each in dependency order. The order is fixed here, in
code, and does not branch on any output. Every entry pins the SHA-1 of its driver, of the artifacts
modules it imports (closure), of the package source tree, of the configs and of its encoder; the
runner refuses an entry whose pins have moved.

What is in and out is decided in the approved plan (docs/ladder_v2/plan.md, decisions 1–6), restated
in EXCLUDED so the dry run prints it beside the order.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts"
PY = ".venv/bin/python"

ENCODERS = {"a1": "runs/m_v2/encoder.pt", "a2": "runs/m_v2_s1/encoder.pt", "a3": "runs/m_v2_s2/encoder.pt"}
COMMON_ENV = {
    "GJ_CONFIG": "configs/pretrain_v2.yaml",
    "F0_CACHE_BASE": "/Volumes/X10 Pro/galaxy-jepa/runs/m_v2/cache",
    "F0_NORM_PREFIX": "246d8de6ba9a",
    "LADDER_SPLIT": "identical",
    "PYTHONUNBUFFERED": "1",
    "OMP_NUM_THREADS": "4",
}
CONFIGS = ["configs/pretrain_v2.yaml", "configs/pretrain.yaml", "configs/probe.yaml", "configs/effect_floor.json"]

EXCLUDED = {
    "E2 per-answer AUC (aligned_score c2)": "decision 4 — its family is powered.json, fixed from M1/M2",
    "E9 T2 (t2_other)": "decision 4 — target chosen from M's ladder",
    "E13 V1 (v1_loose_ends)": "decision 4 — SUPPRESSION list chosen from M's U2; escape is encoder-free",
    "x1_vector_posthoc": "decision 4 — M-specific, exploratory",
    "dd_part4 / dd_l2336_scan": "decision 4 — M's latent 328 / 2336",
    "E25 aligned criteria (c1a/c2/c3)": "the comparison with M — a separate, later step",
    "E27 package battery": "decision 3 — not implemented as a ladder of record",
    "E21 AA2, Y1/Y2, Z1, N1/N2, J/K/L/O3, leakage, BB": "encoder-free or one-off (inventory §3)",
    "aa3b --ensemble, dd_sae_cards, DD V1/V2/V4/S2/Part 2": "not in the inventory's chain of record",
}
NOT_IMPLEMENTED = [
    "8×8 ablation (D11)", "untrained-MLP bar (D25; R3 unreachable)", "Tracy–Widom edge",
    "vote-floor sweep [1, 5, 11, 21, 37]", "inclination conditioning vs b/a (only S1/V2 proxies)",
    "confound 2: recoverability vs imaging depth", "Scheme 2 as a 10-family ladder (only U3's 4 axes)",
    "free probe replication (1B)", "confidence-as-target (D9)", "GalaxyMNIST", "supervised label-efficiency arm",
]


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_sha1(root: Path, pattern: str = "*.py") -> str:
    """One SHA-1 over a source tree: sorted (relative path, file SHA-1)."""
    h = hashlib.sha1()
    for p in sorted(root.rglob(pattern)):
        h.update(f"{p.relative_to(root)}\0{sha1(p)}\n".encode())
    return h.hexdigest()


_IMPORT = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)


def closure(script: str) -> list[str]:
    """The artifacts modules a driver imports, transitively (the package is pinned as a tree)."""
    seen: set[str] = set()
    todo = [ART / script]
    while todo:
        p = todo.pop()
        rel = str(p.relative_to(REPO))
        if rel in seen:
            continue
        seen.add(rel)
        for mod in _IMPORT.findall(p.read_text()):
            q = ART / f"{mod}.py"
            if q.exists():
                todo.append(q)
    return sorted(seen)


class Builder:
    def __init__(self, root: str) -> None:
        self.root = root.rstrip("/")
        self.entries: list[dict] = []
        self._sha: dict[str, str] = {}
        self.pkg = tree_sha1(REPO / "src" / "galaxy_jepa")

    def _pin(self, rel: str) -> str:
        if rel not in self._sha:
            self._sha[rel] = sha1(REPO / rel)
        return self._sha[rel]

    def out(self, enc: str) -> str:
        return f"{self.root}/{enc}"

    def add(self, eid: str, *, exp: str, enc: str, args: list[str] | None = None, script: str | None = None,
            env: dict | None = None, inputs: Sequence[str] = (), heavy: str = "cpu", est_h: float = 0.1,
            expect: Sequence[str] = (), keep: Sequence[str] = (), link: dict | None = None) -> str:
        e: dict = {"id": eid, "exp": exp, "encoder": enc, "heavy": heavy, "est_h": est_h,
                   "inputs": [i for i in inputs if i], "out_dir": self.out(enc)}
        pins = {rel: self._pin(rel) for rel in CONFIGS}
        pins["src/galaxy_jepa/**.py"] = self.pkg
        if enc in ENCODERS:
            pins[ENCODERS[enc]] = self._pin(ENCODERS[enc])
        if link is not None:
            e["kind"], e["est_h"] = "link", 0.0
            e["link"] = link
        else:
            e["kind"] = "run"
            assert script is not None, eid
            e["cmd"] = [PY, f"artifacts/{script}", *(args or [])]
            for rel in closure(script):
                pins[rel] = self._pin(rel)
            if enc in ENCODERS:
                env = {"LADDER_ENCODER": ENCODERS[enc], "LADDER_OUT": self.out(enc), "LADDER_TAG": enc, **(env or {})}
            e["env"] = env or {}
            if expect:
                e["expect"] = list(expect)
            if keep:
                e["keep"] = list(keep)  # survive an interrupted run: the driver resumes from them
        e["pins"] = pins
        self.entries.append(e)
        return eid

    def build(self, encoders: list[str]) -> list[dict]:
        bank = f"{self.root}/shared/p1_untrained_bank.json"
        COMMON_ENV["LADDER_UNTRAINED_BANK"] = bank
        # E4: architecture + pixels + split only, so one bank serves A1–A3 (decision 2). A1's checkpoint
        # supplies the constructor record; no weights reach the bar.
        e4 = self.add("E4.p1_untrained_bank.shared", exp="E4", enc="shared", script="p1_untrained_bank.py",
                      args=["--seeds", "30", "--checkpoint", ENCODERS["a1"]], heavy="mps", est_h=6.0,
                      env={"LADDER_ENCODER": ENCODERS["a1"], "LADDER_OUT": f"{self.root}/shared",
                           "LADDER_TAG": "shared"},
                      expect=["p1_untrained_bank.json"], keep=["p1_untrained_bank.json"])
        first = None
        for enc in encoders:
            first = first or enc
            self._encoder(enc, first, e4)
        return self.entries

    def _encoder(self, enc: str, first: str, e4: str) -> None:
        a, E = self.add, enc.upper()
        ext = a(f"setup.ext.{enc}", exp="setup", enc=enc,
                link={"src": "artifacts/out/ext", "dst": "ext"})  # external catalogues, encoder-free
        a(f"E0.h5_probe.{enc}", exp="E0", enc=enc, script="h5_probe_lean.py", args=["--arm", enc],
               heavy="mps", est_h=0.5, expect=["h5_probes.jsonl"])
        e1 = a(f"E1.c2_bank.{enc}", exp="E1", enc=enc, script="ladder_v2_c2.py", args=[enc],
               heavy="mps", est_h=0.5, expect=[f"c2_{enc}_embeddings.npz"])
        e3 = a(f"E3.o1_matched.{enc}", exp="E3", enc=enc, script="o1_matched.py",
               args=["--checkpoint", ENCODERS[enc], "--tag", "o1"], heavy="mps", est_h=0.6,
               expect=["o1_embeddings.npz", "o1_matched.json"])
        if enc == first:  # untrained seeds 1–2 over the union: architecture-only, built once, linked after
            e7b = a(f"E7.r_bank.{enc}", exp="E7", enc=enc, script="r_nonlinear.py",
                    args=["--checkpoint", ENCODERS[enc], "--stages", "bank"], inputs=[e3], heavy="mps",
                    est_h=0.4, expect=["r_untrained_seeds.npz"])
        else:
            e7b = a(f"E7.r_bank_link.{enc}", exp="E7", enc=enc, inputs=[f"E7.r_bank.{first}", e3],
                    link={"src": f"{self.out(first)}/r_untrained_seeds.npz", "dst": "r_untrained_seeds.npz"})
        base = [e3, e7b, e4]
        r = ["--checkpoint", ENCODERS[enc]]
        p2 = a(f"E5.p2_ladder.{enc}", exp="E5", enc=enc, script="p2_ladder.py", args=r, inputs=base,
               heavy="mps", est_h=3.0, expect=["p2_ladder.json"])
        r0 = a(f"E7.r0.{enc}", exp="E7", enc=enc, script="r_nonlinear.py", args=[*r, "--stages", "r0"],
               inputs=[*base, p2], est_h=0.5, expect=["r_r0.json"])
        s2 = a(f"E5.s2_verify.{enc}", exp="E5", enc=enc, script="s2_verify.py", inputs=[*base, p2, r0],
               heavy="mps", est_h=3.0, expect=["s2_ladder.json", "s2_noise.npz"])
        t1 = a(f"E5.t1_ladder.{enc}", exp="E5", enc=enc, script="t1_ladder.py", inputs=[*base, p2, s2],
               est_h=4.0, expect=["t1_ladder.json"])
        r3 = a(f"E7.r3.{enc}", exp="E7", enc=enc, script="r_nonlinear.py", args=[*r, "--stages", "r3"],
               inputs=[*base, r0], est_h=0.8, expect=["r_r3.json", "r3_orientation.npz"])
        r2 = a(f"E7.r2.{enc}", exp="E7", enc=enc, script="r_nonlinear.py", args=[*r, "--stages", "r2"],
               inputs=[*base, r0, r3], est_h=1.5, expect=["r_r2.json"])
        a(f"E7.r1.{enc}", exp="E7", enc=enc, script="r_nonlinear.py", args=[*r, "--stages", "r1"],
          inputs=[*base, r0], est_h=3.0, expect=["r_r1.json"])
        a(f"E7.r2_reach_split.{enc}", exp="E7", enc=enc, script="r2_reach_split.py", inputs=[*base, r2],
          est_h=0.3, expect=["r2_reach_split.json"])
        s1 = a(f"E8.s1_spiral_inclination.{enc}", exp="E8", enc=enc, script="s1_spiral_inclination.py",
               inputs=[*base, r2], est_h=0.3, expect=["s1_spiral_inclination.json"])
        a(f"E10.u1_bar_winding.{enc}", exp="E10", enc=enc, script="u1_bar_winding.py", inputs=[*base, t1],
          est_h=0.1, expect=["u1_bar_winding.json"])
        u2c = a(f"E11.u2_counts.{enc}", exp="E11", enc=enc, script="u2_uncertainty.py", args=["--counts"],
                inputs=[*base, t1], est_h=0.1, expect=["u2_counts.json"])
        u2 = a(f"E11.u2_uncertainty.{enc}", exp="E11", enc=enc, script="u2_uncertainty.py",
               inputs=[*base, t1, r2, s1, u2c], est_h=2.0, expect=["u2_uncertainty.json"])
        u3 = a(f"E12.u3_graded.{enc}", exp="E12", enc=enc, script="u3_graded.py", inputs=[*base, t1],
               est_h=0.5, expect=["u3_graded.json"])
        v2 = a(f"E14.v2_independent.{enc}", exp="E14", enc=enc, script="v2_independent.py",
               inputs=[*base, ext, t1, u2], est_h=0.5, expect=["v2_independent.json"])
        v3 = a(f"E15.v3_own_directions.{enc}", exp="E15", enc=enc, script="v3_own_directions.py",
               inputs=[*base, t1, r3], est_h=1.0, expect=["v3_own_directions.json"])
        w2 = [a(f"E16.w2_{m}.{enc}", exp="E16", enc=enc, script="w2_name_pcs.py", args=[f"--{m}"],
                inputs=[*base, ext, v3], heavy="cpu", est_h=0.2, expect=[x])
              for m, x in (("measure", "w2_stamp_axes.npz"), ("planted", "w2_planted.json"),
                           ("run", "w2_name_pcs.json"))]
        x1 = [a(f"E17.x1_{m}.{enc}", exp="E17", enc=enc, script="x1_handedness.py", args=[f"--{m}"],
                inputs=[*base, ext, *w2], heavy=h, est_h=t, expect=[x])
              for m, h, t, x in (("bank", "mps", 0.2, "x1_transform_bank.npz"),
                                 ("planted", "cpu", 0.1, "x1_planted.json"),
                                 ("run", "cpu", 0.2, "x1_handedness.json"))]
        y3 = [a(f"E18.y3_{m}.{enc}", exp="E18", enc=enc, script="y3_science.py", args=[f"--{m}"],
                inputs=[*base, ext, u3, v2, *x1], heavy=h, est_h=t, expect=[x])
              for m, h, t, x in (("bank", "mps", 0.2, "y3_extra_bank.npz"),
                                 ("planted3", "cpu", 0.2, "y3_planted.json"),
                                 ("y3", "cpu", 0.3, "y3_pitch.json"))]
        for m, x in (("planted", "z3_planted.json"), ("z3", "z3_bt.json")):
            a(f"E19.z3_{m}.{enc}", exp="E19", enc=enc, script="z3_bt.py", args=[f"--{m}"],
              inputs=[*base, ext, *y3], est_h=0.2, expect=[x])
        for m, x in (("planted", "aa1_planted.json"), ("aa1", "aa1_label_efficiency.json")):
            a(f"E20.aa1_{m}.{enc}", exp="E20", enc=enc, script="aa1_label_efficiency.py", args=[f"--{m}"],
              inputs=base, est_h=0.3 if m == "planted" else 1.5, expect=[x])
        aa3a = [a(f"E22.aa3a_{m}.{enc}", exp="E22", enc=enc, script="aa3a_pose.py", args=[f"--{m}"],
                  inputs=[*base, *x1], heavy="mps", est_h=0.3, expect=[x])
                for m, x in (("planted", "aa3a_planted.json"), ("aa3a", "aa3a_pose.json"))]
        for m, h, t, x in (("bank", "mps", 1.6, "aa3b_d4_progress.json"), ("planted", "cpu", 0.3, "aa3b_planted.json"),
                           ("aa3b", "cpu", 0.5, "aa3b_pose_average.json")):
            a(f"E23.aa3b_{m}.{enc}", exp="E23", enc=enc, script="aa3b_pose_average.py", args=[f"--{m}"],
              inputs=[*base, v3, *x1, *y3, *aa3a], heavy=h, est_h=t, expect=[x],
              keep=["aa3b_d4_*"] if m == "bank" else ())
        self._dd(enc, first, E, [e3])
        [a(f"E26.plate_{m}.{enc}", exp="E26", enc=enc, script="plate_export.py", args=[m],
                inputs=[e1], est_h=t, expect=[x])
              for m, t, x in (("subset", 0.3, "plate/plate_subset.npz"), ("export", 0.4, "plate/plate.bin"),
                              ("sanity", 0.05, "plate/plate.bin"))]

    def _dd(self, enc: str, first: str, E: str, base: list[str]) -> None:
        a = self.add
        dd = f"{self.out(enc)}/dd"
        env = {"DD_TAG": E, "DD_ENCODER": ENCODERS[enc], "DD_OUT": dd, "DD_BANK": f"{self.out(enc)}/o1_embeddings.npz",
               "DD_LOCAL": "runs/dd_local_v2"}
        prev = list(base)
        if enc != first:  # untrained tokens and SAEs are architecture-only: A1's, linked (as the aligned chain)
            src = f"{self.out(first)}/dd"
            prev.append(a(f"E24.dd_untrained_link.{enc}", exp="E24", enc=enc,
                          inputs=[f"E24.dd_train_untrained_{la}_{mu}.{first}" for la in (11, 6) for mu in (8, 16)],
                          link={"glob": [f"{src}/sae_tokens/untrained_*", f"{src}/sae/untrained_*.pt",
                                         f"{src}/sae/untrained_*.log.json"],
                                "dst_dir": {"sae_tokens/": "dd/sae_tokens", "sae/": "dd/sae"}}))

        def step(name, args, script, heavy="cpu", est=0.2, expect=()):
            prev.append(a(f"E24.dd_{name}.{enc}", exp="E24", enc=enc, script=script, args=args, env=env,
                          inputs=[*base, prev[-1]] if len(prev) > len(base) else list(base), heavy=heavy, est_h=est, expect=[f"dd/{x}" for x in expect]))

        step("sae_plants", ["plants"], "dd_sae_score.py", est=0.2)
        step("extract", ["extract"], "dd_sae.py", heavy="mps", est=0.2)
        encs = [E, "untrained"] if enc == first else [E]
        for e in encs:
            for la in (11, 6):
                for mu in (8, 16):
                    tag = "untrained" if e == "untrained" else "trained"
                    step(f"train_{tag}_{la}_{mu}", ["train", e, str(la), str(mu)], "dd_sae.py", heavy="mps",
                         est=0.45, expect=[f"sae/{e}_b{la}_x{mu}.pt"])
        for e in (E, "untrained"):
            for la in (11, 6):
                for mu in (8, 16):
                    tag = "untrained" if e == "untrained" else "trained"
                    step(f"eval_{tag}_{la}_{mu}", ["evaluate", e, str(la), str(mu)], "dd_sae.py", heavy="mps",
                         est=0.12)
        step("score", ["score"], "dd_sae_score.py", est=0.3, expect=["sae/score.json"])
        for s in ("flags", "stats", "pools", "plants", "run"):
            step(f"4b_{s}", [s], "dd_part4b.py", est=0.15)
        step("v3_maps", ["v3"], "dd_part1.py", heavy="mps", est=1.5)
        for s in ("direct", "plants", "v3"):
            step(f"v3b_{s}", [s], "dd_part1b.py", est=0.2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "docs/ladder_v2/manifest.yaml"))
    ap.add_argument("--root", default="runs/ladder_v2", help="per-encoder output directories live under here")
    ap.add_argument("--encoders", default="a1,a2,a3")
    args = ap.parse_args()
    b = Builder(args.root)
    entries = b.build(args.encoders.split(","))
    ids = [e["id"] for e in entries]
    if len(set(ids)) != len(ids):
        raise SystemExit("manifest: duplicate entry ids")
    for e in entries:
        missing = [i for i in e["inputs"] if i not in ids[: ids.index(e["id"])]]
        if missing:
            raise SystemExit(f"manifest: {e['id']} reads {missing}, which do not run before it")
    doc = {"version": 1, "root": args.root, "common_env": COMMON_ENV, "excluded": EXCLUDED,
           "not_implemented": NOT_IMPLEMENTED, "entries": entries}
    Path(args.out).write_text("# Generated by artifacts/ladder_v2_manifest.py — do not edit by hand.\n"
                              + yaml.safe_dump(doc, sort_keys=False, width=110, allow_unicode=True))
    print(f"manifest: {len(entries)} entries → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
