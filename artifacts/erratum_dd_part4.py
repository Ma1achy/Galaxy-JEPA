"""Erratum helper for Brief DD Part 4 density chunk over-count (REG-018, 2026-11-27).

Why this exists:
    ``artifacts/dd_part4.py`` (pinned by SHA-1 ``e7a769c4df7b5bfe9c38d3d226237f9d7e27539c`` in
    ``docs/ladder_v2/manifest.yaml`` and ``docs/ladder_v2/audit.md``) computes latent firing
    densities via::

        for a in range(0, min(len(tok), 256 * 1000), 65536):
            dens += (sae.encode(tok[a:a + 65536]) > 0).float().sum(0)
        dens /= min(len(tok), 256 * 1000)

    Because ``256 * 1000 = 256_000`` is not a multiple of ``65536``, when ``len(tok) >= 262_144``
    the final slice ``tok[196608:262144]`` sums **262,144** tokens while dividing by **256,000**.
    For an always-firing latent this inflates density to ``262144 / 256000 = 1.024`` (+2.4%),
    and when firing rates differ between the 256,000-token prefix and the extra 6,144-token tail,
    old ``(a_j + b_j) / 256000`` vs corrected ``a_j / 256000`` can shift absolute densities by up
    to ``0.024`` (2.4 percentage points), potentially changing latent rankings and ±25%-density
    control pools.

This module keeps ``artifacts/dd_part4.py`` byte-identical so every historical manifest SHA-1
pin remains intact, while providing the corrected :func:`capped_token_density` implementation
and a CLI comparison helper.

    .venv/bin/python artifacts/erratum_dd_part4.py --synthetic-check
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch


@torch.no_grad()
def capped_token_density(
    sae: torch.nn.Module,
    tok: np.ndarray,
    *,
    max_tokens: int = 256 * 1000,
    chunk: int = 65536,
    device: str = "cpu",
) -> np.ndarray:
    """Exact per-latent firing density over the first ``min(len(tok), max_tokens)`` tokens.

    Caps every slice at ``min(a + chunk, lim)`` so numerator and denominator always span the
    identical token count, including non-multiple chunk sizes and empty/short token stores.
    """
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk!r}")
    if max_tokens < 0:
        raise ValueError(f"max_tokens must be >= 0, got {max_tokens!r}")
    n_latents = int(sae.W_enc.shape[0])
    lim = min(len(tok), max_tokens)
    if lim == 0:
        return np.zeros(n_latents, dtype=np.float64)

    counts = np.zeros(n_latents, dtype=np.float64)
    seen = 0
    for a in range(0, lim, chunk):
        b = min(a + chunk, lim)
        batch = torch.from_numpy(np.asarray(tok[a:b], dtype=np.float32)).to(device)
        counts += (sae.encode(batch) > 0).float().sum(0).cpu().numpy().astype(np.float64)
        seen += b - a
    assert seen == lim
    return counts / float(lim)


@torch.no_grad()
def legacy_uncapped_token_density(
    sae: torch.nn.Module,
    tok: np.ndarray,
    *,
    max_tokens: int = 256 * 1000,
    chunk: int = 65536,
    device: str = "cpu",
) -> np.ndarray:
    """Isolated reproduction of the historical ``dd_part4.py:75-77`` uncapped chunk bug."""
    n_latents = int(sae.W_enc.shape[0])
    lim = min(len(tok), max_tokens)
    if lim == 0:
        return np.zeros(n_latents, dtype=np.float64)
    counts = np.zeros(n_latents, dtype=np.float64)
    for a in range(0, lim, chunk):
        batch = torch.from_numpy(np.asarray(tok[a : a + chunk], dtype=np.float32)).to(device)
        counts += (sae.encode(batch) > 0).float().sum(0).cpu().numpy().astype(np.float64)
    return counts / float(lim)


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantify REG-018 density bias in dd_part4.py.")
    ap.add_argument("--synthetic-check", action="store_true")
    args = ap.parse_args()
    if args.synthetic_check:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from dd_sae import TopKSAE

        sae = TopKSAE(d=8, n=4, k=4)
        with torch.no_grad():
            sae.b_enc.fill_(10.0)  # always fires -> true density is 1.0
        tok = np.zeros((300, 8), dtype=np.float32)
        buggy = legacy_uncapped_token_density(sae, tok, max_tokens=250, chunk=64)
        fixed = capped_token_density(sae, tok, max_tokens=250, chunk=64)
        print(
            json.dumps(
                {
                    "lim": 250,
                    "chunk": 64,
                    "true_density": 1.0,
                    "legacy_uncapped_density": buggy.tolist(),
                    "capped_density": fixed.tolist(),
                    "legacy_inflation_ratio": float(buggy[0] / fixed[0]),
                    "dd_part4_scale_inflation_ratio": 262144.0 / 256000.0,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
