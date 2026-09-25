"""Brief DD core: encoders, probe directions, and occlusion maps on M. TOOL VALIDATION ONLY.

The read-out is the probes' own: block 11 of 12 (`DEFAULT_LAYER = -2`, before the final norm),
mean-pooled over the tokens present. Removing a patch = dropping its token (the context encoder's
training path — `patch_embed_tokens` → gather → blocks), or replacing it with the token of a patch of
sky noise matched to the stamp's own noise (never zeros, never the mean).

Encoders: M; the untrained baseline (seed 0, O1's bank — with its own probes); the cascade (M with
blocks randomised 12 → 1 from seed 1, each block's norms with it, then the patch embedding; M's
probes, pooling unchanged).

Maps are in units of the answer score's SD over the occlusion sample; positive = the patch supports
the concept (removing it lowers the score).
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO / "runs" / "dd"
OUT = REPO / "artifacts" / "out" / "dd"
O1_BANK = REPO / "artifacts" / "out" / "o1_embeddings.npz"
READ_BLOCK = 11  # 1-based; DEFAULT_LAYER = -2 of 12
GRID = 16
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ── encoders ─────────────────────────────────────────────────────────────────────────────────────

def m_encoder():
    from galaxy_jepa.models.vit import load_frozen_encoder
    return load_frozen_encoder(REPO / "runs" / "m" / "encoder.pt").to(DEVICE)


def untrained_encoder(config: dict, seed: int = 0):
    """O1's untrained baseline: the same constructor, seeded exactly as `untrained_encoder_matrix`."""
    from galaxy_jepa.models.vit import VisionTransformer
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = VisionTransformer(**dict(config))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(DEVICE)


def cascade(m, level: int, seed: int = 1):
    """M with its top `level` blocks re-initialised (block 12 first), each with its own norms;
    level 13 also re-initialises the patch embedding. Level 0 is M. The positional embedding is the
    fixed sin-cos table, not a weight, and stays."""
    rnd = untrained_encoder(m.config, seed=seed)
    out = copy.deepcopy(m)
    depth = len(out.blocks)
    for k in range(min(level, depth)):
        out.blocks[depth - 1 - k].load_state_dict(rnd.blocks[depth - 1 - k].state_dict())
    if level > depth:
        out.patch_embed.load_state_dict(rnd.patch_embed.state_dict())
    return out.eval()


@torch.no_grad()
def pooled(model, tokens: torch.Tensor, block: int = READ_BLOCK) -> torch.Tensor:
    """Run blocks 1..block over (B, n, D) tokens (positions already added); mean over tokens."""
    x = tokens
    for b in model.blocks[:block]:
        x = b(x)
    return x.mean(dim=1)


@torch.no_grad()
def block_tokens(model, images: torch.Tensor, blocks: tuple[int, ...]) -> dict[int, torch.Tensor]:
    """Per-token outputs of the named 1-based blocks for full images."""
    x = model.patch_embed_tokens(images)
    out = {}
    for i, b in enumerate(model.blocks[:max(blocks)], 1):
        x = b(x)
        if i in blocks:
            out[i] = x
    return out


# ── read-out directions ──────────────────────────────────────────────────────────────────────────

@dataclass
class Readout:
    """Linear read-outs in raw embedding coordinates: score = x @ w + b, one column per name."""

    names: list[str]
    w: np.ndarray  # (D, K)
    b: np.ndarray  # (K,)
    sd: np.ndarray | None = None  # per-name score SD over the occlusion sample (set by `calibrate`)

    def scores(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w + self.b

    def calibrate(self, x_sample: np.ndarray) -> None:
        self.sd = self.scores(x_sample).std(axis=0)


def probe_readout(setup, source: str = "real") -> Readout:
    """The 37 ladder probes (`_fit`, C from probe.yaml) on O1's bank train rows, in raw
    coordinates. `source`: 'real' (M) or 'untrained' (seed 0, its own probes)."""
    from galaxy_jepa.probing.extract import EmbeddingMatrix, feature_embeddings
    from galaxy_jepa.probing.logistic import _fit
    bank = np.load(O1_BANK, allow_pickle=False)
    mat = EmbeddingMatrix(bank["ids"].astype(np.int64), bank[source].astype(np.float64), source)
    names, ws, bs = [], [], []
    for f in setup.labels.features:
        tr = feature_embeddings(mat, setup.labels, f, setup.train_ids)
        if len(np.unique(tr.y)) < 2:
            continue
        scaler, clf = _fit(tr, c=setup.pc.c)
        w = clf.coef_[0] / scaler.scale_
        names.append(f)
        ws.append(w)
        bs.append(float(np.asarray(clf.intercept_)[0] - (clf.coef_[0] * scaler.mean_ / scaler.scale_).sum()))
    return Readout(names, np.stack(ws, 1), np.asarray(bs))


def band_offset_readout(setup) -> Readout:
    """AA3a's PC1 — the band-misregistration axis: PC1 of M's pooled embeddings over the bank's
    train rows (`m_band_axis` reproduces AA3a with a 5k stride of the same rows)."""
    bank = np.load(O1_BANK, allow_pickle=False)
    pos = {int(o): i for i, o in enumerate(bank["ids"])}
    x = bank["real"][[pos[i] for i in setup.train_ids if i in pos]].astype(np.float64)
    mu = x.mean(0)
    _, v = np.linalg.eigh(np.cov(x, rowvar=False))
    pc1 = v[:, -1]
    return Readout(["band_offset_pc1"], pc1[:, None], np.array([-(mu @ pc1)]))


# ── stamps and sky ───────────────────────────────────────────────────────────────────────────────

def stamps(name: str) -> tuple[np.ndarray, np.ndarray]:
    return np.load(LOCAL / f"{name}_ids.npy"), np.load(LOCAL / f"{name}_stamps.npy", mmap_mode="r")


def sky_stats(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per band: median and robust σ of the sky outside r = 64 px, 3σ-clipped twice."""
    yy, xx = np.mgrid[:img.shape[1], :img.shape[2]]
    outer = np.hypot(yy - 127.5, xx - 127.5) > 64
    med, sig = [], []
    for b in range(img.shape[0]):
        v = img[b][outer].astype(np.float64)
        for _ in range(2):
            m, s = np.median(v), 1.4826 * np.median(np.abs(v - np.median(v)))
            v = v[np.abs(v - m) < 3 * s]
        med.append(np.median(v))
        sig.append(1.4826 * np.median(np.abs(v - np.median(v))))
    return np.asarray(med), np.asarray(sig)


def noise_image(img: np.ndarray, seed: int) -> np.ndarray:
    med, sig = sky_stats(img)
    rng = np.random.default_rng(seed)
    return (med[:, None, None] + sig[:, None, None] * rng.standard_normal(img.shape)).astype(np.float32)


# ── occlusion ────────────────────────────────────────────────────────────────────────────────────

def _removals(scale: int) -> list[np.ndarray]:
    """Token indices removed per variant: single patches (256), or 2×2 blocks at stride 1 (225)."""
    if scale == 1:
        return [np.array([j]) for j in range(GRID * GRID)]
    return [np.array([(r + dr) * GRID + (c + dc) for dr in (0, 1) for dc in (0, 1)])
            for r in range(GRID - 1) for c in range(GRID - 1)]


class Occluder:
    """Occlusion maps for one encoder and a set of read-outs, both removal modes, both scales."""

    def __init__(self, model, readouts: list[Readout], batch: int = 256) -> None:
        self.model = model
        self.readouts = readouts
        self.batch = batch

    @torch.no_grad()
    def pooled_variants(self, img: np.ndarray, mode: str, scale: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """(base pooled (D,), pooled per removal (V, D))."""
        x = torch.from_numpy(np.asarray(img, np.float32))[None].to(DEVICE)
        tok = self.model.patch_embed_tokens(x)[0]  # (256, D)
        base = pooled(self.model, tok[None])[0]
        rem = _removals(scale)
        ntok = tok
        if mode == "noise":
            nz = torch.from_numpy(noise_image(np.asarray(img, np.float32), seed))[None].to(DEVICE)
            ntok = self.model.patch_embed_tokens(nz)[0]
        outs = []
        for a in range(0, len(rem), self.batch):
            chunk = rem[a:a + self.batch]
            if mode == "drop":
                keep = torch.stack([torch.from_numpy(np.setdiff1d(np.arange(GRID * GRID), r)) for r in chunk]).to(DEVICE)
                seq = tok[keep]  # (b, 256 − k, D)
            else:
                seq = tok[None].repeat(len(chunk), 1, 1)
                for i, r in enumerate(chunk):
                    seq[i, torch.from_numpy(r).to(DEVICE)] = ntok[torch.from_numpy(r).to(DEVICE)]
            outs.append(pooled(self.model, seq))
        return base.cpu().numpy().astype(np.float64), torch.cat(outs).cpu().numpy().astype(np.float64)

    def maps(self, img: np.ndarray, mode: str = "drop", scale: int = 1, seed: int = 0) -> dict[str, np.ndarray]:
        """{name: map} over every read-out, plus 'norm' (the concept-free map: Δ‖pooled‖, in units
        of ‖pooled‖). Single-patch maps are (16, 16); 2×2 maps are spread back onto patches (each
        patch = the mean over the blocks covering it)."""
        base, var = self.pooled_variants(img, mode, scale, seed)
        out: dict[str, np.ndarray] = {}
        for ro in self.readouts:
            assert ro.sd is not None, "calibrate each read-out on the occlusion sample first"
            s0, sv = ro.scores(base[None])[0], ro.scores(var)
            d = (s0[None] - sv) / ro.sd[None]  # positive = the patch supports it
            for k, n in enumerate(ro.names):
                out[n] = _to_patches(d[:, k], scale)
        n0 = np.linalg.norm(base)
        out["norm"] = _to_patches((n0 - np.linalg.norm(var, axis=1)) / n0, scale)
        return out


def _to_patches(v: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return v.reshape(GRID, GRID)
    acc, cnt = np.zeros((GRID, GRID)), np.zeros((GRID, GRID))
    for i, (r, c) in enumerate((r, c) for r in range(GRID - 1) for c in range(GRID - 1)):
        acc[r:r + 2, c:c + 2] += v[i]
        cnt[r:r + 2, c:c + 2] += 1
    return acc / cnt


def base_pooled(model, imgs: np.ndarray, batch: int = 64) -> np.ndarray:
    out = []
    with torch.no_grad():
        for a in range(0, len(imgs), batch):
            x = torch.from_numpy(np.asarray(imgs[a:a + batch], np.float32)).to(DEVICE)
            out.append(pooled(model, model.patch_embed_tokens(x)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)
