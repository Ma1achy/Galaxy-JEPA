"""Brief DD, Part 3: TopK sparse autoencoders on M's patch tokens. TOOL VALIDATION ONLY.

TopK SAE after Gao et al. 2024 ("Scaling and evaluating sparse autoencoders"):
  z = TopK(W_enc (x − b_pre) + b_enc),   x̂ = W_dec z + b_pre
- b_pre is the pre-encoder bias, shared with the decoder, initialised at the token mean.
- W_dec has unit-norm columns: renormalised after every step, with the gradient component parallel
  to each column removed first.
- W_enc is initialised as W_decᵀ.
- AuxK loss: dead latents (no firing in the last DEAD_TOKENS tokens) reconstruct the residual
  x − x̂ with their own top-k_aux, at coefficient 1/32.
- Inputs are scaled by one scalar so that E‖x‖² = d; the scalar is stored with the SAE.

Layers: block 11 (the probe layer) and block 6. Encoders: M and the untrained baseline (seed 0).
Tokens are stored fp16 as memmaps on runs/dd/sae_tokens.

  uv run python artifacts/dd_sae.py extract            # tokens: sae (train) + sae_eval, per encoder/layer
  uv run python artifacts/dd_sae.py train <enc> <layer> <mult>
  uv run python artifacts/dd_sae.py evaluate <enc> <layer> <mult>
  uv run python artifacts/dd_sae.py smoke              # synthetic end-to-end check (no real tokens)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
import dd_core as D  # noqa: E402

TOK = D.LOCAL / "sae_tokens"
SAE_DIR = D.LOCAL / "sae"
LAYERS = (11, 6)
MULTS = (8, 16)
K, K_AUX, AUX_COEF = 32, 256, 1 / 32
BATCH, LR, EPOCHS = 4096, 2e-4, 4
DEAD_TOKENS = 1_000_000
WIDTH = 384
SEED = 20260925


class TopKSAE(nn.Module):
    def __init__(self, d: int, n: int, k: int = K, scale: float = 1.0) -> None:
        super().__init__()
        self.k = k
        self.b_pre = nn.Parameter(torch.zeros(d))
        w = torch.randn(d, n)
        w = w / w.norm(dim=0, keepdim=True)
        self.W_dec = nn.Parameter(w)
        self.W_enc = nn.Parameter(w.T.clone())
        self.b_enc = nn.Parameter(torch.zeros(n))
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.register_buffer("last_fired", torch.zeros(n, dtype=torch.long))

    def pre(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.scale - self.b_pre) @ self.W_enc.T + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse codes (…, n): the top-k pre-activations, ReLU'd, everything else 0."""
        h = self.pre(x)
        v, i = h.topk(self.k, dim=-1)
        return torch.zeros_like(h).scatter_(-1, i, torch.relu(v))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Back in the encoder's own units."""
        return (z @ self.W_dec.T + self.b_pre) / self.scale

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    @torch.no_grad()
    def renorm(self) -> None:
        self.W_dec.data /= self.W_dec.data.norm(dim=0, keepdim=True)

    @torch.no_grad()
    def drop_parallel_grad(self) -> None:
        if self.W_dec.grad is not None:
            w = self.W_dec.data
            self.W_dec.grad -= w * (w * self.W_dec.grad).sum(0, keepdim=True)


def losses(sae: TopKSAE, x: torch.Tensor, seen: int) -> tuple[torch.Tensor, dict]:
    """Normalised MSE (in scaled units) + AuxK on the dead latents' own top-k_aux."""
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
    dead = (seen - sae.last_fired) > DEAD_TOKENS
    aux = torch.zeros((), device=x.device)
    if dead.any():
        ka = int(min(K_AUX, int(dead.sum())))
        hd = h.masked_fill(~dead, float("-inf"))
        va, ia = hd.topk(ka, dim=-1)
        za = torch.zeros_like(h).scatter_(-1, ia, torch.relu(va))
        aux = (err.detach() - za @ sae.W_dec.T).pow(2).sum(-1).mean() / var
    loss = mse / var + AUX_COEF * aux
    return loss, {"nmse": (mse / var).item(), "aux": aux.item(), "dead": dead.float().mean().item()}


class TokenStore:
    """fp16 token memmaps: TOK/<enc>_b<layer>_<sample>.npy, shape (N·256, 384)."""

    @staticmethod
    def path(enc: str, layer: int, sample: str) -> Path:
        return TOK / f"{enc}_b{layer}_{sample}.npy"

    @staticmethod
    def extract(model, enc: str, sample: str, batch: int = 64) -> None:
        ids, st = D.stamps(sample)
        outs = {L: np.lib.format.open_memmap(TokenStore.path(enc, L, sample), mode="w+", dtype=np.float16,
                                             shape=(len(ids) * 256, WIDTH)) for L in LAYERS}
        t0 = time.time()
        for a in range(0, len(ids), batch):
            x = torch.from_numpy(np.asarray(st[a:a + batch], np.float32)).to(D.DEVICE)
            got = D.block_tokens(model, x, LAYERS)
            for L, t in got.items():
                outs[L][a * 256:(a + len(x)) * 256] = t.reshape(-1, WIDTH).cpu().numpy().astype(np.float16)
            if (a // batch) % 50 == 0:
                print(f"  {enc} {sample} {a + len(x):,}/{len(ids):,}  {time.time() - t0:.0f}s", flush=True)
        for m in outs.values():
            m.flush()


def train(enc: str, layer: int, mult: int, tokens: np.ndarray | None = None, epochs: int = EPOCHS,
          lr: float = LR) -> TopKSAE:
    data = tokens if tokens is not None else np.load(TokenStore.path(enc, layer, "sae"), mmap_mode="r")
    n_tok = len(data)
    rng = np.random.default_rng(SEED)
    probe = torch.from_numpy(np.asarray(data[np.sort(rng.choice(n_tok, min(n_tok, 65_536), replace=False))],
                                        np.float32))
    scale = float(np.sqrt(WIDTH / probe.pow(2).sum(-1).mean()))
    torch.manual_seed(SEED)
    sae = TopKSAE(WIDTH, WIDTH * mult, scale=scale)
    sae.b_pre.data = (probe * scale).mean(0)
    sae = sae.to(D.DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999), eps=6.25e-10)
    seen, log = 0, []
    t0 = time.time()
    for ep in range(epochs):
        # contiguous chunks in random order: memmap-friendly, still shuffled within each batch
        chunk = BATCH * 16
        for c in rng.permutation(range(0, n_tok, chunk)):
            block = torch.from_numpy(np.asarray(data[c:c + chunk], np.float32))
            block = block[torch.randperm(len(block))].to(D.DEVICE)
            for b in range(0, len(block), BATCH):
                x = block[b:b + BATCH]
                seen += len(x)
                loss, st = losses(sae, x, seen)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                sae.drop_parallel_grad()
                opt.step()
                sae.renorm()
            log.append({"epoch": ep, "seen": seen, **st, "t": round(time.time() - t0)})
        print(f"  {enc} b{layer} ×{mult} epoch {ep}: {log[-1]}", flush=True)
    sae = sae.cpu()
    if tokens is None:
        SAE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"state": sae.state_dict(), "k": sae.k, "n": WIDTH * mult}, SAE_DIR / f"{enc}_b{layer}_x{mult}.pt")
        (SAE_DIR / f"{enc}_b{layer}_x{mult}.log.json").write_text(json.dumps(log))
    return sae


def load(enc: str, layer: int, mult: int) -> TopKSAE:
    ck = torch.load(SAE_DIR / f"{enc}_b{layer}_x{mult}.pt", map_location="cpu")
    sae = TopKSAE(WIDTH, ck["n"], ck["k"])
    sae.load_state_dict(ck["state"])
    return sae.eval()


@torch.no_grad()
def evaluate(enc: str, layer: int, mult: int, readout=None) -> dict:
    """On sae_eval tokens: variance explained, dead fraction, and (given the encoder's 37-probe
    read-out) task faithfulness — replace block-`layer` tokens with reconstructions, run the
    remaining blocks up to 11, re-pool, re-probe; AUC drop per answer."""
    sae = load(enc, layer, mult).to(D.DEVICE)
    data = np.load(TokenStore.path(enc, layer, "sae_eval"), mmap_mode="r")
    n_gal = len(data) // 256
    sse = sst = 0.0
    mu = torch.from_numpy(np.asarray(data[:: max(1, len(data) // 200_000)], np.float32)).mean(0).to(D.DEVICE)
    fired = torch.zeros(sae.W_enc.shape[0], dtype=torch.bool, device=D.DEVICE)
    model = D.m_encoder() if enc == "M" else D.untrained_encoder(D.m_encoder().config, seed=0)
    pooled_rec, pooled_orig = [], []
    for g in range(0, n_gal, 64):
        x = torch.from_numpy(np.asarray(data[g * 256:(g + 64) * 256], np.float32)).to(D.DEVICE)
        xr, z = sae(x)
        sse += float((x - xr).pow(2).sum())
        sst += float((x - mu).pow(2).sum())
        fired |= (z > 0).any(0)
        seq_r, seq_o = xr.reshape(-1, 256, WIDTH), x.reshape(-1, 256, WIDTH)
        for blk in model.blocks[layer:D.READ_BLOCK]:
            seq_r, seq_o = blk(seq_r), blk(seq_o)
        pooled_rec.append(seq_r.mean(1).cpu().numpy())
        pooled_orig.append(seq_o.mean(1).cpu().numpy())
    out = {"variance_explained": 1 - sse / sst, "dead_frac": float(1 - fired.float().mean()), "n_tokens": len(data)}
    if readout is not None:
        out["faithfulness"] = faithfulness(np.concatenate(pooled_orig), np.concatenate(pooled_rec), readout)
    (SAE_DIR / f"{enc}_b{layer}_x{mult}.eval.json").write_text(json.dumps(out, indent=1))
    return out


def faithfulness(orig: np.ndarray, rec: np.ndarray, readout) -> dict:
    """Per-answer AUC of the fixed probes on sae_eval galaxies, original vs reconstructed pooled."""
    import r_nonlinear as R
    from j4_spread_controls import prepare
    from sklearn.metrics import roc_auc_score
    setup = prepare(None, R.MAX_TRAIN, label="DD3", sources=1)
    ids = [int(i) for i in np.load(D.LOCAL / "sae_eval_ids.npy")]
    row = {o: k for k, o in enumerate(ids)}
    so, sr = readout.scores(orig.astype(np.float64)), readout.scores(rec.astype(np.float64))
    per = {}
    for j, f in enumerate(readout.names):
        el = [o for o in setup.labels.eligible(f, ids)]
        y = np.asarray(setup.labels.binary_label(f, el))
        if len(np.unique(y)) < 2:
            continue
        r = [row[o] for o in el]
        a0, a1 = roc_auc_score(y, so[r, j]), roc_auc_score(y, sr[r, j])
        per[f] = {"auc": float(a0), "auc_rec": float(a1), "drop": float(a0 - a1), "n": len(el)}
    return {"per_answer": per, "mean_drop": float(np.mean([v["drop"] for v in per.values()]))}


def smoke() -> None:
    """Synthetic end-to-end: tokens drawn from a sparse dictionary; the SAE must recover most of the
    variance, renorm must hold, and a dead latent must be revived by AuxK or reported dead."""
    rng = np.random.default_rng(0)
    true = rng.standard_normal((WIDTH, 512))
    true /= np.linalg.norm(true, axis=0)
    codes = np.zeros((200_000, 512))
    for r in range(len(codes)):
        idx = rng.choice(512, 8, replace=False)
        codes[r, idx] = rng.exponential(1.0, 8)
    x = (codes @ true.T + 0.01 * rng.standard_normal((len(codes), WIDTH))).astype(np.float16)
    sae = train("smoke", 0, 2, tokens=x, epochs=10, lr=1e-3)  # ~490 steps; 3 epochs at 2e-4 reach only 0.46
    with torch.no_grad():
        xt = torch.from_numpy(x[:20_000].astype(np.float32))
        xr, z = sae(xt)
        ve = 1 - float((xt - xr).pow(2).sum() / (xt - xt.mean(0)).pow(2).sum())
    norms = sae.W_dec.norm(dim=0)
    print({"variance_explained": round(ve, 3), "dec_norm_range": [float(norms.min()), float(norms.max())],
           "l0": float((z > 0).sum(-1).float().mean())})
    assert ve > 0.8, ve
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "extract":
        TOK.mkdir(parents=True, exist_ok=True)
        m = D.m_encoder()
        for enc, model in (("M", m), ("untrained", D.untrained_encoder(m.config, seed=0))):
            for sample in ("sae", "sae_eval"):
                if not TokenStore.path(enc, LAYERS[-1], sample).exists():
                    TokenStore.extract(model, enc, sample)
    elif cmd == "train":
        train(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "evaluate":
        enc, layer, mult = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
        import r_nonlinear as R
        from j4_spread_controls import prepare
        ro = D.probe_readout(prepare(None, R.MAX_TRAIN, label="DD3", sources=1),
                             "real" if enc == "M" else "untrained")
        print(json.dumps(evaluate(enc, layer, mult, ro), indent=1))
    elif cmd == "smoke":
        smoke()
