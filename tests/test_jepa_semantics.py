"""Semantic / anti-regression tests for Vessel-JEPA."""
from __future__ import annotations

import torch

from angio_flux.jepa import (
    JEPAPredictor,
    VesselJEPA,
    ViT3DEncoder,
    build_grid_positions,
)
from angio_flux.jepa.masking import batched_masks, sample_context_target_masks
from angio_flux.jepa.sampling import rpca_inflow_times
from angio_flux.jepa.tubelet import TubeletPatchifier
from angio_flux.losses.jepa import VesselJEPALoss, bat_rank_loss, vicreg_terms


def _tiny_cfg():
    return dict(
        in_channels=1, patch_size=8, tubelet_t=4, embed_dim=24,
        encoder_depth=2, encoder_heads=4,
        predictor_dim=12, predictor_depth=2, predictor_heads=4,
    )


def test_tubelet_roundtrip_shape():
    patcher = TubeletPatchifier(in_channels=1, patch_size=8, tubelet_t=4, embed_dim=24)
    video = torch.randn(2, 1, 16, 32, 32)
    tokens, grid = patcher(video)
    assert grid == (4, 4, 4)
    assert tokens.shape == (2, 64, 24)


def test_grid_positions_match_token_order():
    grid = (2, 3, 4)
    pos = build_grid_positions(grid)
    assert pos.shape == (24, 3)
    # Order is (t, y, x) with x fastest, then y, then t.
    assert torch.equal(pos[0], torch.tensor([0, 0, 0]))
    assert torch.equal(pos[1], torch.tensor([0, 0, 1]))
    assert torch.equal(pos[4], torch.tensor([0, 1, 0]))
    assert torch.equal(pos[12], torch.tensor([1, 0, 0]))


def test_ema_target_no_grad():
    torch.manual_seed(0)
    model = VesselJEPA(**_tiny_cfg())
    video = torch.randn(1, 1, 8, 16, 16)
    tgt_idx = torch.tensor([[0, 1, 2]])
    out = model.target_embeddings(video, tgt_idx)
    assert out.requires_grad is False
    for p in model.target.target.parameters():
        assert p.requires_grad is False


def test_predictor_uses_position():
    torch.manual_seed(0)
    pred = JEPAPredictor(encoder_dim=24, predictor_dim=12, depth=2, num_heads=4)
    ctx = torch.randn(1, 4, 24)
    ctx_pos = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
    out_a = pred(ctx, ctx_pos, torch.tensor([[0, 1, 1]]))
    out_b = pred(ctx, ctx_pos, torch.tensor([[1, 0, 0]]))
    assert not torch.allclose(out_a, out_b, atol=1.0e-4)


def test_vicreg_prevents_collapse_signal():
    collapsed = torch.zeros(64, 16)
    var, cov = vicreg_terms(collapsed)
    assert var.item() > 0.5  # large hinge when all features collapse
    diverse = torch.randn(64, 16) * 2.0
    var2, _ = vicreg_terms(diverse)
    assert var2.item() < var.item()


def test_bat_rank_orders_upstream_before_downstream():
    # Synthetic 1D "bolus": positions with smaller inflow_time should have smaller bat.
    bat = torch.nn.Parameter(torch.zeros(1, 8))
    inflow = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    optim = torch.optim.SGD([bat], lr=0.5)
    for _ in range(200):
        optim.zero_grad()
        loss = bat_rank_loss(bat, inflow, num_pairs=64, margin=0.1)
        loss.backward()
        optim.step()
    # After training, bat should be monotonically non-decreasing in time.
    diffs = bat.detach()[0, 1:] - bat.detach()[0, :-1]
    assert (diffs >= -1.0e-3).float().mean().item() > 0.8


def test_vessel_score_higher_on_flow_than_static():
    # Construct a synthetic video where one quadrant has a moving dark "vessel"
    # propagating across frames (predictable structure), and the rest is noise.
    torch.manual_seed(0)
    T, H, W = 16, 32, 32
    video = 0.5 + 0.05 * torch.randn(1, 1, T, H, W)
    for t in range(T):
        x = (t * (W // T)) % W
        video[0, 0, t, 4:12, x : x + 4] -= 0.4  # dark moving bar
    video = video.clamp(0.0, 1.0)

    # Patchify and check that tubelets covering the bar have higher RPCA presence
    # than tubelets in the noise region.
    from angio_flux.preprocess import rpca_sparse_component
    with torch.no_grad():
        rpca_dark = rpca_sparse_component(video, max_iter=10)
    grid = (T // 4, H // 8, W // 8)
    inflow, presence = rpca_inflow_times(rpca_dark, grid)
    flow_tokens = presence[0].reshape(*grid)
    bar_region = flow_tokens[:, 0:2, :].mean()
    noise_region = flow_tokens[:, 2:, :].mean()
    assert bar_region.item() > noise_region.item()


def test_mask_sampling_disjoint():
    grid = (4, 6, 6)
    ctx, tgt = sample_context_target_masks(grid, num_targets=4, target_block=(1, 2, 2))
    assert len(set(ctx.tolist()) & set(tgt.tolist())) == 0
    assert tgt.numel() > 0
    assert ctx.numel() > 0


def test_smoke_jepa_forward_backward():
    torch.manual_seed(0)
    model = VesselJEPA(**_tiny_cfg())
    loss_fn = VesselJEPALoss()
    video = torch.randn(1, 1, 8, 16, 16)
    tokens_all, positions_all, grid = model.patch_and_positions(video)
    ctx_idx, tgt_idx = batched_masks(1, grid, num_targets=2,
                                      target_block=(1, 1, 1), context_fraction=0.5)
    ctx_enc, ctx_pos = model.encode_context(tokens_all, positions_all, ctx_idx)
    pred, _ = model.predict_targets(ctx_enc, ctx_pos, positions_all, tgt_idx)
    target_emb = model.target_embeddings(video, tgt_idx)
    loss, parts = loss_fn(pred, target_emb)
    loss.backward()
    grads = [p.grad for p in model.student.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert torch.isfinite(loss)
    # Predictor must also receive gradient.
    pred_grads = [p.grad for p in model.predictor.parameters() if p.grad is not None]
    assert len(pred_grads) > 0


def test_ema_update_moves_target_toward_student():
    torch.manual_seed(0)
    model = VesselJEPA(**_tiny_cfg())
    # Perturb student.
    with torch.no_grad():
        for p in model.student.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    before = next(model.target.target.parameters()).clone()
    model.ema_momentum = 0.5
    model.update_ema()
    after = next(model.target.target.parameters())
    assert not torch.allclose(before, after)
