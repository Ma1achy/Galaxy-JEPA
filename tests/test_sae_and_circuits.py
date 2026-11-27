"""SAE, hook/ablation, REG-018 token density, energy oracle & planted circuit suite (§9).

Covers:
- TopKSAE invariants: Top-K sparsity, scale roundtrip, unit-norm decoder columns, parallel-grad
  projection, dead-latent tracking + AuxK gradients, latent permutation invariance, planted sparse
  dictionary recovery (§9.1)
- Hooks & latent_patch: identity/dead no-op, exception-safe cleanup, causal layer isolation,
  single & multi-latent residual-preserving subtraction (§9.2)
- REG-018: pinned dd_part4.py SHA-1 preservation + erratum_dd_part4.py exact density (§9.3)
- Removed energy E(S) = 1_S^T C 1_S exact parity including cross-term cancellation & last-layer
  readout score delta oracle (§9.4)
- Planted computational-graph circuit benchmark: independent, shared, correlated distractor,
  redundant OR-like, synergistic AND/XOR-like, cancellation, bypass, and negative control (§9.5)
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from galaxy_jepa.models.vit import VisionTransformer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "artifacts"))


# ── §9.1 TopKSAE unit & planted sparse dictionary recovery ────────────────────────────────────────


@pytest.mark.scientific
def test_topk_sae_invariants_auxk_permutation_and_planted_dictionary() -> None:
    """Verify TopKSAE structural invariants, AuxK dead-latent gradients, and dictionary recovery."""
    import dd_sae as sae_mod

    torch.manual_seed(91)
    d, n, k = 16, 48, 4
    scale = 2.5
    sae = sae_mod.TopKSAE(d=d, n=n, k=k, scale=scale)
    x = torch.randn(32, d)

    # 1. Sparsity <= k, non-negativity, unit-norm columns
    x_hat, z = sae(x)
    assert x_hat.shape == x.shape and z.shape == (32, n)
    assert (z >= 0).all()
    assert int((z > 0).sum(dim=-1).max()) <= k
    col_norms = sae.W_dec.data.norm(dim=0)
    assert torch.allclose(col_norms, torch.ones(n), atol=1e-6)

    # 2. Scale invariance: multiplying x by alpha and dividing scale by alpha preserves z
    sae_scaled = sae_mod.TopKSAE(d=d, n=n, k=k, scale=scale / 3.0)
    sae_scaled.load_state_dict({**sae.state_dict(), "scale": torch.tensor(scale / 3.0)})
    x_hat_s, z_s = sae_scaled(x * 3.0)
    assert torch.allclose(z_s, z, atol=1e-5)
    assert torch.allclose(x_hat_s, x_hat * 3.0, atol=1e-5)

    # 3. Parallel-gradient removal makes W_dec.grad orthogonal to every W_dec column
    loss, stats = sae_mod.losses(sae, x, seen=100)
    assert np.isfinite(float(loss.detach())) and np.isfinite(stats["nmse"])
    loss.backward()
    sae.drop_parallel_grad()
    assert sae.W_dec.grad is not None
    dots = (sae.W_dec.data * sae.W_dec.grad).sum(dim=0)
    assert torch.allclose(dots, torch.zeros(n), atol=1e-6)

    # 4. AuxK loss produces non-zero gradient on forced-dead latents
    sae.zero_grad(set_to_none=True)
    # Mark all latents as long-unfired except the first k
    sae.last_fired.zero_()
    sae.last_fired[:k] = sae_mod.DEAD_TOKENS + 500
    # Push b_enc down slightly on dead latents? Keep positive pre-activations so AuxK fires them.
    sae.b_enc.data.fill_(0.5)
    loss_aux, stats_aux = sae_mod.losses(sae, x, seen=sae_mod.DEAD_TOKENS + 500)
    assert stats_aux["dead"] > 0.0 and stats_aux["aux"] > 0.0
    loss_aux.backward()
    assert sae.W_enc.grad is not None and float(sae.W_enc.grad[k:].abs().sum()) > 0.0

    # 5. Latent permutation invariance: permuting latent indices leaves reconstruction identical
    perm = torch.randperm(n)
    sae_p = sae_mod.TopKSAE(d=d, n=n, k=k, scale=scale)
    with torch.no_grad():
        sae_p.b_pre.copy_(sae.b_pre)
        sae_p.W_dec.copy_(sae.W_dec[:, perm])
        sae_p.W_enc.copy_(sae.W_enc[perm, :])
        sae_p.b_enc.copy_(sae.b_enc[perm])
    x_hat_1, z_1 = sae(x)
    x_hat_p, z_p = sae_p(x)
    assert torch.allclose(x_hat_1, x_hat_p, atol=1e-5)
    assert torch.allclose(z_1[:, perm], z_p, atol=1e-5)

    # 6. Planted incoherent sparse dictionary recovery (fast CPU smoke)
    rng = np.random.default_rng(910)
    d_p, m_true, k_true = 12, 16, 2
    true_dict = rng.normal(size=(d_p, m_true))
    true_dict /= np.linalg.norm(true_dict, axis=0, keepdims=True)
    n_tok = 1500
    codes = np.zeros((n_tok, m_true), dtype=np.float32)
    for i in range(n_tok):
        idx = rng.choice(m_true, size=k_true, replace=False)
        codes[i, idx] = rng.uniform(0.8, 2.0, size=k_true)
    tokens = torch.from_numpy((codes @ true_dict.T).astype(np.float32))
    scale_p = float(np.sqrt(d_p / float(tokens.pow(2).sum(-1).mean())))
    torch.manual_seed(911)
    sae_rec = sae_mod.TopKSAE(d=d_p, n=m_true * 2, k=k_true, scale=scale_p)
    opt = torch.optim.Adam(sae_rec.parameters(), lr=1e-2)
    seen = 0
    for _step in range(180):
        idx_b = torch.randint(0, n_tok, (256,))
        xb = tokens[idx_b]
        seen += len(xb)
        l_step, _ = sae_mod.losses(sae_rec, xb, seen)
        opt.zero_grad(set_to_none=True)
        l_step.backward()
        sae_rec.drop_parallel_grad()
        opt.step()
        sae_rec.renorm()
    with torch.no_grad():
        rec_all, _ = sae_rec(tokens)
        var_exp = 1.0 - float((tokens - rec_all).pow(2).sum() / tokens.pow(2).sum())
        cos_mat = np.abs(true_dict.T @ sae_rec.W_dec.detach().numpy())
        median_best_cos = float(np.median(cos_mat.max(axis=1)))
    assert var_exp > 0.85, f"planted dict variance explained too low: {var_exp:.3f}"
    assert median_best_cos > 0.85, f"planted dict atom recovery too low: {median_best_cos:.3f}"


# ── §9.2 Hooks, exception safety, layer isolation & latent_patch arithmetic ──────────────────────


@pytest.mark.scientific
def test_hooks_exception_cleanup_causal_isolation_and_residual_preserving_patch() -> None:
    """Verify Hooks cleanup on exception, causal layer isolation, and exact latent_patch."""
    import dd_circuits as circ
    import dd_sae as sae_mod

    torch.manual_seed(92)
    vit = VisionTransformer(img_size=32, patch_size=8, embed_dim=16, depth=3, heads=2).eval()
    x_img = torch.randn(2, 3, 32, 32)
    tok0 = vit.patch_embed_tokens(x_img)

    # 1. Identity patch leaves all block captures bit-for-bit identical
    with torch.no_grad():
        with circ.Hooks(vit, capture=(1, 2, 3)) as h_base:
            y_base = vit.run_tokens(tok0, norm=False)
        with circ.Hooks(vit, capture=(1, 2, 3), patch={2: lambda t: t}) as h_id:
            y_id = vit.run_tokens(tok0, norm=False)
    assert torch.equal(y_base, y_id)
    for layer in (1, 2, 3):
        assert torch.equal(h_base.acts[layer], h_id.acts[layer])

    # 2. Exception during forward MUST remove all forward hooks (zero hook leak)
    def _boom(_t: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("injected hook failure")

    with pytest.raises(RuntimeError, match="injected hook failure"):
        with circ.Hooks(vit, capture=(1, 2, 3), patch={2: _boom}):
            vit.run_tokens(tok0, norm=False)
    for blk in vit.blocks:
        assert len(blk._forward_hooks) == 0

    # 3. Causal layer isolation: patching block 3 leaves blocks 1 & 2 identical;
    #    patching block 1 alters blocks 2 & 3.
    with torch.no_grad():
        with circ.Hooks(vit, capture=(1, 2, 3), patch={3: lambda t: t + 5.0}) as h_late:
            vit.run_tokens(tok0, norm=False)
        with circ.Hooks(vit, capture=(1, 2, 3), patch={1: lambda t: t + 5.0}) as h_early:
            vit.run_tokens(tok0, norm=False)
    assert torch.equal(h_late.acts[1], h_base.acts[1])
    assert torch.equal(h_late.acts[2], h_base.acts[2])
    assert not torch.allclose(h_late.acts[3], h_base.acts[3])
    assert not torch.allclose(h_early.acts[2], h_base.acts[2])
    assert not torch.allclose(h_early.acts[3], h_base.acts[3])

    # 4. Residual-preserving latent_patch: x' = x - z_j * W_dec[:, j] / scale (keeps error)
    sae = sae_mod.TopKSAE(d=16, n=32, k=4, scale=1.75).eval()
    tokens = h_base.acts[2]
    z = sae.encode(tokens)
    firing_latents = torch.where((z > 0).any(dim=(0, 1)))[0]
    dead_latents = torch.where((z == 0).all(dim=(0, 1)))[0]
    assert len(firing_latents) >= 2 and len(dead_latents) >= 1

    # Dead latent patch is exact no-op
    j_dead = int(dead_latents[0])
    assert torch.equal(circ.latent_patch(sae, j_dead)(tokens), tokens)

    # Single-latent patch equals exact subtraction AND preserves SAE reconstruction error verbatim
    j0 = int(firing_latents[0])
    patched = circ.latent_patch(sae, j0)(tokens)
    x_hat = sae.decode(z)
    err = tokens - x_hat
    z_zeroed = z.clone()
    z_zeroed[..., j0] = 0.0
    expected = sae.decode(z_zeroed) + err
    assert torch.allclose(patched, expected, rtol=1e-5, atol=1e-6)


# ── §9.3 REG-018: Pinned dd_part4.py SHA-1 & erratum_dd_part4 exact density ──────────────────────


@pytest.mark.regression
def test_reg018_dd_part4_sha1_preserved_and_erratum_capped_density_exact() -> None:
    """REG-018: Pinned dd_part4.py stays byte-identical while erratum helper fixes +2.4%."""
    import erratum_dd_part4 as err4

    pinned_sha1 = "e7a769c4df7b5bfe9c38d3d226237f9d7e27539c"
    actual_sha1 = hashlib.sha1((REPO / "artifacts" / "dd_part4.py").read_bytes()).hexdigest()
    assert actual_sha1 == pinned_sha1

    # Exact reproduction of the +2.4% (262,144 / 256,000) overcount on always-firing latents
    # Use proportional scale 250 / 64 -> 256/250 = 1.024 (identical ratio 262144/256000)
    import dd_sae as sae_mod

    sae_all = sae_mod.TopKSAE(d=8, n=4, k=4)
    with torch.no_grad():
        sae_all.b_enc.fill_(10.0)
    tok_all = np.zeros((300, 8), dtype=np.float32)
    buggy = err4.legacy_uncapped_token_density(sae_all, tok_all, max_tokens=250, chunk=64)
    fixed = err4.capped_token_density(sae_all, tok_all, max_tokens=250, chunk=64)
    assert np.allclose(buggy, 256.0 / 250.0, rtol=1e-12)
    assert np.allclose(fixed, 1.0, rtol=1e-12)
    assert float(buggy[0] / fixed[0]) == pytest.approx(262_144.0 / 256_000.0, rel=1e-12)

    # Exact fractional density on non-multiple store lengths and empty store
    class _IdentitySAE(torch.nn.Module):
        def __init__(self, n_lat: int) -> None:
            super().__init__()
            self.W_enc = torch.nn.Parameter(torch.zeros(n_lat, n_lat))

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            return x

    id_sae = _IdentitySAE(3)
    tok_mix = np.zeros((10, 3), dtype=np.float32)
    tok_mix[:7, 0] = 1.0  # true density 0.7
    tok_mix[:3, 1] = 2.0  # true density 0.3
    dens = err4.capped_token_density(id_sae, tok_mix, chunk=4, device="cpu")
    assert np.allclose(dens, [0.7, 0.3, 0.0], atol=1e-12)

    empty = np.zeros((0, 3), dtype=np.float32)
    assert np.allclose(
        err4.capped_token_density(id_sae, empty, chunk=4, device="cpu"), [0.0, 0.0, 0.0]
    )


# ── §9.4 Removed Energy E(S) = 1_S^T C 1_S & Last-Layer Linear Readout Oracle ────────────────────


@pytest.mark.scientific
@pytest.mark.differential
def test_removed_energy_cross_terms_and_last_layer_ablation_score_oracle() -> None:
    """Verify E(S) = 1_S^T C 1_S exactness (with off-diagonal cancellation) & readout delta."""
    import dd_circuits as circ
    import dd_part4b as p4b
    import dd_sae as sae_mod

    rng = np.random.default_rng(94)
    n_tok, d, n_lat = 200, 12, 24
    scale = 1.8
    # Construct correlated non-orthogonal decoder columns so off-diagonal C_ij != 0!
    w_dec = rng.normal(size=(d, n_lat))
    w_dec[:, 1] = -0.8 * w_dec[:, 0] + 0.6 * w_dec[:, 1]  # negative inner product (cancellation)
    w_dec[:, 2] = 0.8 * w_dec[:, 0] + 0.6 * w_dec[:, 2]  # positive inner product (reinforcement)
    w_dec /= np.linalg.norm(w_dec, axis=0, keepdims=True)

    # Sparse positive codes where latents 0, 1, 2 co-fire
    z = np.zeros((n_tok, n_lat), dtype=np.float64)
    z[:, 0] = rng.uniform(0.5, 2.0, size=n_tok)
    z[:, 1] = rng.uniform(0.5, 2.0, size=n_tok)
    z[:, 2] = rng.uniform(0.5, 2.0, size=n_tok)
    z[:, 5] = rng.uniform(0.5, 2.0, size=n_tok)

    # Energy matrix C = (Z^T Z / n) * (W_dec^T W_dec / scale^2)
    c_mat = (z.T @ z / n_tok) * ((w_dec.T @ w_dec) / (scale**2))

    for subset in ([0], [0, 1], [0, 2], [0, 1, 2, 5]):
        e_formula = p4b._energy(c_mat, subset)
        delta_x = (z[:, subset] @ w_dec[:, subset].T) / scale
        e_direct = float(np.mean(np.sum(delta_x**2, axis=1)))
        assert e_formula == pytest.approx(e_direct, rel=1e-12, abs=1e-12)

    # Verify off-diagonal cancellation / reinforcement makes E(S) strictly != sum_j E({j})
    assert p4b._energy(c_mat, [0, 1]) < p4b._energy(c_mat, [0]) + p4b._energy(c_mat, [1]) - 0.1
    assert p4b._energy(c_mat, [0, 2]) > p4b._energy(c_mat, [0]) + p4b._energy(c_mat, [2]) + 0.1

    # Last-block linear readout oracle:
    # For a linear probe s_k(pooled) = pooled @ w_k + b_k reading the mean-pooled last block,
    # ablating latent j at the last block produces exact delta:
    #   Delta s_k = - mean_t(z_{t,j}) * (w_k^T W_dec[:, j]) / scale
    torch.manual_seed(941)
    vit = VisionTransformer(img_size=32, patch_size=8, embed_dim=16, depth=2, heads=2).eval()
    sae = sae_mod.TopKSAE(d=16, n=32, k=4, scale=2.0).eval()
    imgs = torch.randn(4, 3, 32, 32)

    # Capture un-normed block-2 tokens (exact last block before mean pooling)
    with torch.no_grad():
        with circ.Hooks(vit, capture=(2,)) as h:
            tok_in = vit.patch_embed_tokens(imgs)
            out_base = vit.run_tokens(tok_in, norm=False)
        b2_tokens = h.acts[2]
        assert torch.allclose(b2_tokens, out_base, atol=1e-6)
        z_b2 = sae.encode(b2_tokens)
        j_hit = int(torch.where((z_b2 > 0).any(dim=(0, 1)))[0][0])

        with circ.Hooks(vit, patch={2: circ.latent_patch(sae, j_hit)}):
            out_abl = vit.run_tokens(tok_in, norm=False)

    w_probe = rng.normal(size=(16, 3))
    pooled_base = out_base.mean(dim=1).double().numpy()
    pooled_abl = out_abl.mean(dim=1).double().numpy()
    delta_hook = (pooled_abl @ w_probe) - (pooled_base @ w_probe)

    z_bar_j = z_b2[..., j_hit].mean(dim=1).double().numpy()  # (B,)
    w_dec_j = sae.W_dec[:, j_hit].detach().double().numpy()  # (D,)
    delta_oracle = -np.outer(z_bar_j, w_dec_j @ w_probe) / float(sae.scale)
    assert np.allclose(delta_hook, delta_oracle, rtol=1e-5, atol=1e-6)


# ── §9.5 Planted Computational-Graph Circuit Benchmark ───────────────────────────────────────────


class _PlantedCircuit:
    """Explicit 8-node structural causal model exercising every circuit motif in §9.5.

    Latent sources:
      u_a, u_b, u_shared, u_r, u_s1, u_s2, u_c ~ N(0, 1) independent
    Nodes (hidden activations, all intervenable via do(node = 0)):
      h_a      = u_a                            (direct causal path to y_a)
      h_dist   = 0.9 * u_a + sqrt(0.19) * eps   (correlated NON-CAUSAL distractor!)
      h_b      = u_b                            (direct causal path to y_b)
      h_shared = u_shared                       (shared bottleneck -> y_a AND y_b)
      h_r1     = relu(u_r)                      (redundant OR-like branch 1 -> y_red)
      h_r2     = relu(u_r)                      (redundant OR-like branch 2 -> y_red)
      h_s1     = u_s1                           (synergistic branch 1 -> y_syn = h_s1 * h_s2)
      h_s2     = u_s2                           (synergistic branch 2 -> y_syn = h_s1 * h_s2)
      h_c_pos  = u_c                            (cancellation + branch -> y_cancel)
      h_c_neg  = -u_c                           (cancellation - branch -> y_cancel)
      h_med    = 0.7 * h_a                      (mediated path; h_a also has direct skip bypass)
    Readouts:
      y_a      = 1.5 * h_a + 1.0 * h_med + 1.2 * h_shared
      y_b      = 1.4 * h_b + 1.2 * h_shared
      y_red    = max(h_r1, h_r2)
      y_syn    = h_s1 * h_s2
      y_cancel = h_c_pos + h_c_neg + 0.5 * h_b
      y_null   = independent noise (negative control — zero causal parents)
    """

    NODES = (
        "h_a",
        "h_dist",
        "h_b",
        "h_shared",
        "h_r1",
        "h_r2",
        "h_s1",
        "h_s2",
        "h_c_pos",
        "h_c_neg",
        "h_med",
    )

    def __init__(self, n: int = 3000, seed: int = 95) -> None:
        rng = np.random.default_rng(seed)
        self.u = {
            k: rng.normal(size=n)
            for k in ("a", "b", "shared", "r", "s1", "s2", "c", "dist_eps", "null")
        }

    def forward(
        self, ablate: set[str] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        ab = ablate or set()
        u = self.u
        h: dict[str, np.ndarray] = {}
        h["h_a"] = np.zeros_like(u["a"]) if "h_a" in ab else u["a"]
        h["h_dist"] = (
            np.zeros_like(u["a"])
            if "h_dist" in ab
            else 0.9 * u["a"] + math.sqrt(0.19) * u["dist_eps"]
        )
        h["h_b"] = np.zeros_like(u["b"]) if "h_b" in ab else u["b"]
        h["h_shared"] = np.zeros_like(u["shared"]) if "h_shared" in ab else u["shared"]
        r_pos = np.maximum(u["r"], 0.0)
        h["h_r1"] = np.zeros_like(r_pos) if "h_r1" in ab else r_pos
        h["h_r2"] = np.zeros_like(r_pos) if "h_r2" in ab else r_pos
        h["h_s1"] = np.zeros_like(u["s1"]) if "h_s1" in ab else u["s1"]
        h["h_s2"] = np.zeros_like(u["s2"]) if "h_s2" in ab else u["s2"]
        h["h_c_pos"] = np.zeros_like(u["c"]) if "h_c_pos" in ab else u["c"]
        h["h_c_neg"] = np.zeros_like(u["c"]) if "h_c_neg" in ab else -u["c"]
        h["h_med"] = np.zeros_like(u["a"]) if "h_med" in ab else 0.7 * h["h_a"]

        y = {
            "y_a": 1.5 * h["h_a"] + 1.0 * h["h_med"] + 1.2 * h["h_shared"],
            "y_b": 1.4 * h["h_b"] + 1.2 * h["h_shared"],
            "y_red": np.maximum(h["h_r1"], h["h_r2"]),
            "y_syn": h["h_s1"] * h["h_s2"],
            "y_cancel": h["h_c_pos"] + h["h_c_neg"] + 0.5 * h["h_b"],
            "y_null": u["null"],
        }
        return h, y


@pytest.mark.scientific
def test_planted_circuit_motifs_blind_discovery_and_redundancy_pitfall() -> None:
    """§9.5: Validate causal discovery across all 8 circuit motifs + negative-control network."""
    scm = _PlantedCircuit(n=3000, seed=95)
    h_base, y_base = scm.forward()

    def rms_effect(node_set: set[str], target: str) -> float:
        _, y_ab = scm.forward(ablate=node_set)
        return float(np.sqrt(np.mean((y_ab[target] - y_base[target]) ** 2)))

    # 1. Correlated distractor: strong observational correlation with y_a, ZERO causal effect!
    corr_dist = abs(float(np.corrcoef(h_base["h_dist"], y_base["y_a"])[0, 1]))
    assert corr_dist > 0.70, "distractor must be strongly correlated observationally"
    assert rms_effect({"h_dist"}, "y_a") == pytest.approx(0.0, abs=1e-12)

    # 2. Independent & shared pathways: blind single-ablation recovers exact direct parents
    discovered_a = {n for n in scm.NODES if rms_effect({n}, "y_a") > 1e-3}
    discovered_b = {n for n in scm.NODES if rms_effect({n}, "y_b") > 1e-3}
    assert discovered_a == {"h_a", "h_med", "h_shared"}
    assert discovered_b == {"h_b", "h_shared"}
    assert discovered_a & discovered_b == {"h_shared"}

    # 3. Bypass / skip path: ablating h_med removes only the mediated share; bypass remains
    eff_med = rms_effect({"h_med"}, "y_a")
    eff_a_total = rms_effect({"h_a"}, "y_a")
    assert 0.5 < eff_med < eff_a_total - 1.0

    # 4. Redundant OR-like pathways: single ablation yields 0 on EACH branch,
    #    whereas joint ablation {h_r1, h_r2} drops y_red completely!
    #    Proves single-feature ablation CANNOT conclude non-causality for redundant circuits.
    assert rms_effect({"h_r1"}, "y_red") == pytest.approx(0.0, abs=1e-12)
    assert rms_effect({"h_r2"}, "y_red") == pytest.approx(0.0, abs=1e-12)
    assert rms_effect({"h_r1", "h_r2"}, "y_red") > 0.5

    # 5. Synergistic AND-like pathway: ablating either h_s1 or h_s2 zeroes y_syn;
    #    joint effect equals single effect (strictly sub-additive: joint != sum of singles!)
    e_s1 = rms_effect({"h_s1"}, "y_syn")
    e_s2 = rms_effect({"h_s2"}, "y_syn")
    e_s12 = rms_effect({"h_s1", "h_s2"}, "y_syn")
    assert e_s1 > 0.8 and e_s2 > 0.8
    assert abs(e_s12 - e_s1) < 1e-12 and e_s12 < e_s1 + e_s2 - 0.5

    # 6. Cancellation pair: ablating h_c_pos alone UNMASKS -h_c_neg, raising output variance!
    var_base = float(np.var(y_base["y_cancel"]))
    _, y_c_ab = scm.forward(ablate={"h_c_pos"})
    assert float(np.var(y_c_ab["y_cancel"])) > var_base + 0.8
    # Jointly ablating both cancelling nodes leaves y_cancel untouched
    assert rms_effect({"h_c_pos", "h_c_neg"}, "y_cancel") == pytest.approx(0.0, abs=1e-12)

    # 7. Negative-control readout: ZERO causal parents discovered (0 false positives)
    discovered_null = {n for n in scm.NODES if rms_effect({n}, "y_null") > 1e-6}
    assert discovered_null == set()
