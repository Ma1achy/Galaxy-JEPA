"""The fixed-shape AuxK against the original, on recorded A1 batches (user, 2026-10-10).

`dd_sae.losses` now takes topk(K_AUX) over the dead-masked pre-activations; the original took
topk(min(K_AUX, dead)). `old_losses` below is the original, verbatim. For dead counts 0, 1, 50, 193
and 300 (> K_AUX), on real A1 block-11 token batches and an A1 SAE after 300 training steps:

- CPU: old and new loss and every parameter gradient bit-identical (torch.equal).
- MPS: identical, or within TOL_ABS / TOL_REL (fixed before the run); the max differences reported.
- No NaN or inf in the aux reconstruction, the loss or any gradient, either path, every count.
- Dead count 0: aux is exactly 0 and carries no graph, in both.

    uv run python artifacts/sae_auxk_identity.py   # -> artifacts/out/aligned/sae_auxk_identity.json
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import dd_sae as S  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "out" / "aligned" / "sae_auxk_identity.json"
TOKENS = REPO / "runs/dd_a1/sae_tokens/A1_b11_sae.npy"
DEADS, WARM, SEEN = (0, 1, 50, 193, 300), 300, 10 ** 8
TOL_ABS, TOL_REL = 1e-6, 1e-5


def old_losses(sae, x, seen):
    """dd_sae.losses before 2026-10-10, verbatim."""
    xs = x * sae.scale
    h = sae.pre(x)
    v, i = h.topk(sae.k, dim=-1)
    z = torch.zeros_like(h).scatter_(-1, i, torch.relu(v))
    rec = z @ sae.W_dec.T + sae.b_pre
    err = xs - rec
    mse = err.pow(2).sum(-1).mean()
    var = (xs - xs.mean(0)).pow(2).sum(-1).mean()
    fired = (z > 0).any(0)
    sae.last_fired[fired] = seen
    dead = (seen - sae.last_fired) > S.DEAD_TOKENS
    aux = torch.zeros((), device=x.device)
    if dead.any():
        ka = int(min(S.K_AUX, int(dead.sum())))
        hd = h.masked_fill(~dead, float("-inf"))
        va, ia = hd.topk(ka, dim=-1)
        za = torch.zeros_like(h).scatter_(-1, ia, torch.relu(va))
        aux = (err.detach() - za @ sae.W_dec.T).pow(2).sum(-1).mean() / var
    loss = mse / var + S.AUX_COEF * aux
    return loss, {"nmse": (mse / var).item(), "aux": aux.item(), "dead": dead.float().mean().item()}, aux


def new_losses(sae, x, seen):
    loss, st = S.losses(sae, x, seen)
    return loss, st, None


def _za_finite(sae, x, seen):
    """The new path's aux reconstruction, recomputed, for the finiteness assertion."""
    h = sae.pre(x)
    dead = (seen - sae.last_fired) > S.DEAD_TOKENS
    hd = h.masked_fill(~dead, float("-inf"))
    va, ia = hd.topk(min(S.K_AUX, h.shape[-1]), dim=-1)
    za = torch.zeros_like(h).scatter_(-1, ia, torch.relu(va))
    return bool(torch.isfinite(za).all()), bool(torch.isfinite(za @ sae.W_dec.T).all())


def warm_sae(tokens) -> S.TopKSAE:
    """A1's SAE as train() builds it, then WARM steps on MPS: a realistic, non-initial state."""
    rng = np.random.default_rng(S.SEED)
    probe = torch.from_numpy(np.asarray(tokens[np.sort(rng.choice(len(tokens), 65_536, replace=False))], np.float32))
    scale = float(np.sqrt(S.WIDTH / probe.pow(2).sum(-1).mean()))
    torch.manual_seed(S.SEED)
    sae = S.TopKSAE(S.WIDTH, S.WIDTH * 8, scale=scale)
    sae.b_pre.data = (probe * scale).mean(0)
    sae = sae.to("mps")
    opt = torch.optim.Adam(sae.parameters(), lr=S.lr_for(8), betas=(0.9, 0.999), eps=6.25e-10)
    for s in range(WARM):
        x = torch.from_numpy(np.asarray(tokens[s * S.BATCH:(s + 1) * S.BATCH], np.float32)).to("mps")
        loss, _ = S.losses(sae, x, (s + 1) * S.BATCH)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        sae.drop_parallel_grad()
        opt.step()
        sae.renorm()
    return sae.cpu()


def case(base: S.TopKSAE, x: torch.Tensor, n_dead: int, device: str) -> dict:
    """Old and new on identical state: exactly n_dead latents dead after this batch."""
    with torch.no_grad():
        z = base.encode(x)
        silent = torch.where(~(z > 0).any(0))[0]
    if len(silent) < n_dead:
        raise SystemExit(f"only {len(silent)} latents silent on this batch; cannot make {n_dead} dead")
    lf = torch.full_like(base.last_fired, SEEN)
    lf[silent[:n_dead]] = 0  # dead: last fired long ago and silent now
    out = {}
    res = {}
    for name, fn in (("old", old_losses), ("new", new_losses)):
        sae = copy.deepcopy(base).to(device)
        sae.last_fired.copy_(lf.to(device))
        xd = x.to(device)
        sae.zero_grad(set_to_none=True)
        loss, st, aux_old = fn(sae, xd, SEEN)
        loss.backward()
        grads = {k: (p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(p).cpu())
                 for k, p in sae.named_parameters()}
        res[name] = {"loss": loss.detach().cpu(), "st": st, "grads": grads,
                     "aux_requires_grad": None if aux_old is None else bool(aux_old.requires_grad)}
        if name == "new":
            sae2 = copy.deepcopy(base).to(device)
            sae2.last_fired.copy_(lf.to(device))
            out["za_finite"], out["za_recon_finite"] = _za_finite(sae2, xd, SEEN)
        out[f"{name}_finite"] = bool(torch.isfinite(loss).item()) and all(bool(torch.isfinite(g).all()) for g in grads.values())
    o, n = res["old"], res["new"]
    d_loss = float((o["loss"] - n["loss"]).abs())
    d_abs = max(float((o["grads"][k] - n["grads"][k]).abs().max()) for k in o["grads"])
    d_rel = max(float(((o["grads"][k] - n["grads"][k]).abs() / o["grads"][k].abs().clamp_min(1e-30)).max())
                for k in o["grads"])
    out |= {"n_dead": n_dead, "dead_frac_old": o["st"]["dead"], "dead_frac_new": n["st"]["dead"],
            "aux_old": o["st"]["aux"], "aux_new": n["st"]["aux"],
            "loss_bit_identical": bool(torch.equal(o["loss"], n["loss"])),
            "grads_bit_identical": all(torch.equal(o["grads"][k], n["grads"][k]) for k in o["grads"]),
            "max_abs_diff_loss": d_loss, "max_abs_diff_grad": d_abs, "max_rel_diff_grad": d_rel}
    if n_dead == 0:
        out["dead0_aux_exactly_zero"] = o["st"]["aux"] == 0.0 and n["st"]["aux"] == 0.0
        out["dead0_aux_no_graph_old"] = o["aux_requires_grad"] is False
    return out


def main() -> dict:
    tokens = np.load(TOKENS, mmap_mode="r")
    base = warm_sae(tokens)
    rng = np.random.default_rng(20261010)
    rows = sorted(int(r) for r in rng.choice(len(tokens) // S.BATCH, len(DEADS), replace=False))
    res = {"tolerance (fixed before the run)": {"abs": TOL_ABS, "rel": TOL_REL}, "warm_steps": WARM,
           "batches": rows, "cpu": [], "mps": []}
    for r, nd in zip(rows, DEADS, strict=True):
        x = torch.from_numpy(np.asarray(tokens[r * S.BATCH:(r + 1) * S.BATCH], np.float32))
        res["cpu"].append(case(base, x, nd, "cpu") | {"batch": r})
        res["mps"].append(case(base, x, nd, "mps") | {"batch": r})
    cpu_ok = all(c["loss_bit_identical"] and c["grads_bit_identical"] for c in res["cpu"])
    mps_ok = all(c["max_abs_diff_loss"] <= TOL_ABS and (c["max_abs_diff_grad"] <= TOL_ABS or c["max_rel_diff_grad"] <= TOL_REL)
                 for c in res["mps"])
    finite = all(c["old_finite"] and c["new_finite"] and c["za_finite"] and c["za_recon_finite"]
                 for c in res["cpu"] + res["mps"])
    zero = all(c["dead0_aux_exactly_zero"] and c["dead0_aux_no_graph_old"] for c in res["cpu"] + res["mps"] if c["n_dead"] == 0)
    res["pass"] = {"cpu_bit_identical": cpu_ok, "mps_within_tolerance": mps_ok, "finite": finite,
                   "dead0_exact_zero": zero, "all": cpu_ok and mps_ok and finite and zero}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    r = main()
    for dev in ("cpu", "mps"):
        for c in r[dev]:
            print(dev, {k: c[k] for k in ("n_dead", "loss_bit_identical", "grads_bit_identical", "max_abs_diff_loss",
                                          "max_abs_diff_grad", "max_rel_diff_grad", "aux_old", "aux_new")})
    print(json.dumps(r["pass"]))
