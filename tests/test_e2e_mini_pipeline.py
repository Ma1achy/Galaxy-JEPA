"""Self-contained CPU miniature end-to-end scientific pipeline test (§12).

Exercises the complete Galaxy-JEPA workflow end-to-end on deterministic procedural galaxies:
  1. Disjoint pretrain / probe-train / probe-test splits with 64-bit IDs > 2^53
  2. Tiny ViT pretraining for I-JEPA (beta=0 and beta=0.5 with checkpoint/resume), MoCo, MAE
  3. Freeze boundary: save_encoder -> load_frozen_encoder -> assert zero grads / requires_grad=False
  4. Embedding extraction + ID/hash integrity -> linear probes, shuffled/untrained controls
  5. Conditional matching, covariance-aware alignment c_Sigma, exploratory MI (KSG & variational)
  6. Tiny TopKSAE + Hooks/latent_patch intervention on block-2 tokens
  7. Machine-readable JSON summary emitted and cross-checked against live values
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from galaxy_jepa.callbacks.checkpoint import TrainCheckpointer
from galaxy_jepa.data.dataset import ResumableShuffle
from galaxy_jepa.masking.blocks import MaskConfig
from galaxy_jepa.models.vit import VisionTransformer, load_frozen_encoder, save_encoder
from galaxy_jepa.objectives.jepa import Jepa, JepaConfig
from galaxy_jepa.objectives.loop import train_objective
from galaxy_jepa.objectives.mae import Mae, MaeConfig
from galaxy_jepa.objectives.moco import Moco, MocoConfig
from galaxy_jepa.probing import entanglement as ent
from galaxy_jepa.probing import information as info
from galaxy_jepa.probing import matching as match
from galaxy_jepa.probing.extract import EmbeddingMatrix, LabelProvider, feature_embeddings
from galaxy_jepa.probing.logistic import probe_auc_ci, probe_direction

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "artifacts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures_scientific import generate_procedural_galaxies  # noqa: E402


class _StampDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images: np.ndarray,
        ids: np.ndarray,
        petro: np.ndarray,
        pix_scale: np.ndarray,
    ) -> None:
        self.images = images
        self.ids = ids
        self.petro = petro
        self.pix_scale = pix_scale

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> dict[str, object]:
        return {
            "image": torch.from_numpy(self.images[i]),
            "object_id": int(self.ids[i]),
            "petro_rad_arcsec": float(self.petro[i]),
            "pixel_scale": float(self.pix_scale[i]),
        }


def _extract_frozen(enc: VisionTransformer, images: np.ndarray) -> np.ndarray:
    assert all(not p.requires_grad and p.grad is None for p in enc.parameters())
    with torch.no_grad():
        x = torch.from_numpy(images)
        return enc.encode(x).cpu().numpy().astype(np.float64)


@pytest.mark.e2e
def test_miniature_end_to_end_pipeline_cpu(tmp_path: Path) -> None:
    """Full synthetic CPU pipeline: train -> resume -> freeze -> probe -> SAE -> JSON report."""
    import dd_circuits as circ
    import dd_sae as sae_mod

    # 1. Procedural galaxies + strictly disjoint splits (64-bit IDs > 2^53)
    gal = generate_procedural_galaxies(n=120, img_size=32, seed=2026)
    pre_idx = np.arange(0, 40)
    tr_idx = np.arange(40, 80)
    te_idx = np.arange(80, 120)
    pre_ids = [int(i) for i in gal.ids[pre_idx]]
    tr_ids = [int(i) for i in gal.ids[tr_idx]]
    te_ids = [int(i) for i in gal.ids[te_idx]]
    assert set(pre_ids).isdisjoint(tr_ids)
    assert set(pre_ids).isdisjoint(te_ids)
    assert set(tr_ids).isdisjoint(te_ids)
    assert all(oid > (1 << 53) for oid in pre_ids + tr_ids + te_ids)

    pre_ds = _StampDataset(
        gal.images[pre_idx],
        gal.ids[pre_idx],
        gal.petro_rad_arcsec[pre_idx],
        gal.pixel_scale[pre_idx],
    )

    def _tiny_vit(seed: int) -> VisionTransformer:
        torch.manual_seed(seed)
        return VisionTransformer(img_size=32, patch_size=8, embed_dim=16, depth=2, heads=2)

    # 2a. I-JEPA beta=0.0 (uninterrupted)
    cfg_b0 = JepaConfig(
        steps=4,
        batch_size=8,
        lr=1e-3,
        warmup_steps=1,
        pred_dim=12,
        pred_depth=1,
        pred_heads=2,
        mask=MaskConfig(n_target_blocks=2, beta=0.0),
        monitor_every=2,
        checkpoint_every=2,
        seed=10,
    )
    jepa_b0 = Jepa(_tiny_vit(10), cfg_b0)
    samp_b0 = ResumableShuffle(len(pre_ds), seed=10)
    res_b0 = train_objective(
        jepa_b0,
        torch.utils.data.DataLoader(pre_ds, batch_size=8, sampler=samp_b0),
        device="cpu",
        sampler=samp_b0,
    )
    assert res_b0.steps_completed == 4 and all(np.isfinite(res_b0.losses))

    # 2b. I-JEPA beta=0.5 WITH checkpoint + resume parity check
    cfg_b5 = JepaConfig(
        steps=4,
        batch_size=8,
        lr=1e-3,
        warmup_steps=1,
        pred_dim=12,
        pred_depth=1,
        pred_heads=2,
        mask=MaskConfig(n_target_blocks=2, beta=0.5),
        monitor_every=2,
        checkpoint_every=2,
        seed=11,
    )
    ck = TrainCheckpointer(
        tmp_path / "jepa_b5_ckpts",
        every=2,
        keep=4,
        config_hash="e2e_cfg",
        normalisation_hash="e2e_norm",
        schedule={"steps": cfg_b5.steps, "warmup_steps": cfg_b5.warmup_steps, "lr": cfg_b5.lr},
    )
    jepa_part1 = Jepa(_tiny_vit(11), cfg_b5)
    samp_1 = ResumableShuffle(len(pre_ds), seed=11)
    r1 = train_objective(
        jepa_part1,
        torch.utils.data.DataLoader(pre_ds, batch_size=8, sampler=samp_1),
        device="cpu",
        checkpointer=ck,
        sampler=samp_1,
        stop_after=2,
    )
    assert r1.steps_completed == 2

    jepa_resumed = Jepa(_tiny_vit(11), cfg_b5)
    samp_2 = ResumableShuffle(len(pre_ds), seed=11)
    res_b5 = train_objective(
        jepa_resumed,
        torch.utils.data.DataLoader(pre_ds, batch_size=8, sampler=samp_2),
        device="cpu",
        checkpointer=ck,
        sampler=samp_2,
        resume=True,
    )
    assert res_b5.steps_completed == 4 and all(np.isfinite(res_b5.losses))

    # Compare resumed beta=0.5 against uninterrupted beta=0.5 run (strict resume parity)
    jepa_unint = Jepa(_tiny_vit(11), cfg_b5)
    samp_u = ResumableShuffle(len(pre_ds), seed=11)
    res_b5_unint = train_objective(
        jepa_unint,
        torch.utils.data.DataLoader(pre_ds, batch_size=8, sampler=samp_u),
        device="cpu",
        sampler=samp_u,
    )
    assert np.allclose(res_b5.losses, res_b5_unint.losses, rtol=1e-5, atol=1e-5)
    for (k1, v1), (k2, v2) in zip(
        jepa_resumed.encoder.state_dict().items(),
        jepa_unint.encoder.state_dict().items(),
        strict=True,
    ):
        assert k1 == k2 and torch.allclose(v1, v2, rtol=1e-5, atol=1e-5)

    # 2c. MoCo and MAE baseline pretrain smoke
    moco_cfg = MocoConfig(
        steps=3,
        batch_size=8,
        lr=1e-3,
        warmup_steps=1,
        queue=16,
        proj_hidden=16,
        proj_dim=8,
        window_min=2,
        window_max=3,
        monitor_every=2,
        seed=12,
    )
    moco_obj = Moco(_tiny_vit(12), moco_cfg)
    samp_moco = ResumableShuffle(len(pre_ds), seed=12)
    res_moco = train_objective(
        moco_obj,
        torch.utils.data.DataLoader(pre_ds, batch_size=8, sampler=samp_moco),
        device="cpu",
        sampler=samp_moco,
    )
    assert res_moco.steps_completed == 3 and all(np.isfinite(res_moco.losses))

    mae_cfg = MaeConfig(
        steps=3,
        batch_size=8,
        lr=1e-3,
        warmup_steps=1,
        dec_dim=16,
        dec_depth=1,
        dec_heads=2,
        monitor_every=2,
        seed=13,
    )
    mae_obj = Mae(_tiny_vit(13), mae_cfg)
    samp_mae = ResumableShuffle(len(pre_ds), seed=13)
    res_mae = train_objective(
        mae_obj,
        torch.utils.data.DataLoader(pre_ds, batch_size=8, sampler=samp_mae),
        device="cpu",
        sampler=samp_mae,
    )
    assert res_mae.steps_completed == 3 and all(np.isfinite(res_mae.losses))

    # 3. Freeze boundary: save & reload frozen encoder, verify zero trainable params / zero grads
    enc_path = tmp_path / "jepa_b5_encoder.pt"
    save_encoder(jepa_resumed.encoder, enc_path)
    frozen_real = load_frozen_encoder(enc_path)
    untrained_path = tmp_path / "untrained_encoder.pt"
    save_encoder(_tiny_vit(99), untrained_path)
    frozen_untrained = load_frozen_encoder(untrained_path)

    # 4. Frozen extraction on probe splits + ID integrity
    probe_idx = np.concatenate([tr_idx, te_idx])
    probe_ids = gal.ids[probe_idx]
    x_real = _extract_frozen(frozen_real, gal.images[probe_idx])
    x_untr = _extract_frozen(frozen_untrained, gal.images[probe_idx])
    assert x_real.shape == (80, 16) and np.isfinite(x_real).all()

    mat_real = EmbeddingMatrix(probe_ids, x_real, encoder_name="jepa_b5")
    mat_untr = EmbeddingMatrix(probe_ids, x_untr, encoder_name="untrained")
    labels = LabelProvider(
        gal.rows,
        feature_cols={"edge": "edge_frac", "spiral": "spiral_frac", "bar": "bar_frac"},
        nuisance_cols={"brightness": "brightness", "band_offset": "band_offset"},
        vote_count_min=1,
    )

    # 5. Probes, shuffled-label & untrained controls, matching, c_Sigma, and exploratory MI
    from galaxy_jepa.probing.logistic import Embeddings

    shuf_rng = np.random.default_rng(2027)
    dirs = []
    probe_metrics: dict[str, dict[str, float]] = {}
    for feat in ("edge", "spiral", "bar"):
        tr_e = feature_embeddings(mat_real, labels, feat, tr_ids)
        te_e = feature_embeddings(mat_real, labels, feat, te_ids)
        te_u = feature_embeddings(mat_untr, labels, feat, te_ids)
        tr_u = feature_embeddings(mat_untr, labels, feat, tr_ids)
        d_feat = probe_direction(tr_e, name=feat, c=1.0)
        dirs.append(d_feat)
        auc_r, lo_r, hi_r = probe_auc_ci(tr_e, te_e, n_boot=80, seed=0)
        auc_u, _, _ = probe_auc_ci(tr_u, te_u, n_boot=40, seed=0)
        tr_shuf = Embeddings(tr_e.x, shuf_rng.permutation(tr_e.y), tr_e.fraction)
        auc_shuf, _, _ = probe_auc_ci(tr_shuf, te_e, n_boot=40, seed=0)
        vb = info.variational_mi_lower_bound(tr_e.x, tr_e.y, te_e.x, te_e.y, c=1.0, units="nats")
        probe_metrics[feat] = {
            "auc_real": auc_r,
            "auc_lo": lo_r,
            "auc_hi": hi_r,
            "auc_untrained": auc_u,
            "auc_shuffled": auc_shuf,
            "mi_lower_bound_nats": vb.bound,
        }

    # Conditional matching on brightness nuisance for edge-on test split
    te_edge = feature_embeddings(mat_real, labels, "edge", te_ids)
    te_bright = labels.nuisance_value("brightness", te_ids)
    k_match = match.stratified_match(te_bright, te_edge.y, n_strata=4, seed=0)
    assert k_match.size > 0 and k_match.size % 2 == 0

    # Covariance-aware alignment & KSG MI on probe scores
    names_c, c_sig = ent.covariance_alignment(dirs, x_real, use_raw=True)
    assert names_c == ["edge", "spiral", "bar"] and np.isfinite(c_sig).all()
    score_edge = x_real @ dirs[0].w_raw + dirs[0].bias
    score_spiral = x_real @ dirs[1].w_raw + dirs[1].bias
    ksg_est = info.ksg_mi(score_edge, score_spiral, k=5, units="nats")
    assert np.isfinite(ksg_est)

    # 6. Tiny TopKSAE + Hooks/latent_patch intervention on block-2 tokens
    with torch.no_grad():
        tok_in = frozen_real.patch_embed_tokens(torch.from_numpy(gal.images[te_idx[:8]]))
        with circ.Hooks(frozen_real, capture=(2,)) as h_cap:
            base_out = frozen_real.run_tokens(tok_in, norm=False)
        b2_tok = h_cap.acts[2]
        sae = sae_mod.TopKSAE(d=16, n=32, k=4, scale=1.0).eval()
        z_sae = sae.encode(b2_tok)
        j_hit = int(torch.where((z_sae > 0).any(dim=(0, 1)))[0][0])
        with circ.Hooks(frozen_real, patch={2: circ.latent_patch(sae, j_hit)}):
            abl_out = frozen_real.run_tokens(tok_in, norm=False)
        delta_norm = float((abl_out - base_out).pow(2).mean().sqrt())
    assert delta_norm > 0.0

    # 7. Emit machine-readable JSON summary and cross-check schema/values
    split_hash = hashlib.sha256(
        json.dumps({"pre": pre_ids, "tr": tr_ids, "te": te_ids}).encode()
    ).hexdigest()
    summary = {
        "split_hash": split_hash,
        "n_splits": {
            "pretrain": len(pre_ids),
            "probe_train": len(tr_ids),
            "probe_test": len(te_ids),
        },
        "train_final_loss": {
            "jepa_b0": float(res_b0.losses[-1]),
            "jepa_b5_resumed": float(res_b5.losses[-1]),
            "moco": float(res_moco.losses[-1]),
            "mae": float(res_mae.losses[-1]),
        },
        "probes": probe_metrics,
        "c_sigma_diag": [float(c_sig[i, i]) for i in range(3)],
        "ksg_mi_edge_spiral_nats": float(ksg_est),
        "sae_intervention_rms_delta": delta_norm,
    }
    report_path = tmp_path / "mini_pipeline_summary.json"
    report_path.write_text(json.dumps(summary, indent=2) + "\n")
    reloaded = json.loads(report_path.read_text())
    assert reloaded["split_hash"] == split_hash
    assert all(v == pytest.approx(1.0, abs=1e-12) for v in reloaded["c_sigma_diag"])
    assert reloaded["sae_intervention_rms_delta"] == pytest.approx(delta_norm, rel=1e-12)
    assert all(np.isfinite(v) for v in reloaded["train_final_loss"].values())
    for feat_metrics in reloaded["probes"].values():
        assert all(np.isfinite(v) for v in feat_metrics.values())
        for auc_key in ("auc_real", "auc_lo", "auc_hi", "auc_untrained", "auc_shuffled"):
            assert 0.0 <= feat_metrics[auc_key] <= 1.0
