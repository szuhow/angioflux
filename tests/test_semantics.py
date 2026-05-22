import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from angio_flux.data import AngioSequenceDataset
from angio_flux.encoding import HemodynamicEventEncoder, append_contrast_channels, contrast_flow_prior_map
from angio_flux.losses.ssl import SSLPretrainLoss
from angio_flux.losses.multitask import soft_skeleton
from angio_flux.losses.ssl import build_targets
from angio_flux.modules.vqfr import VQFRHead
from angio_flux.preprocess import (
    bolus_peak_frame,
    frangi_vesselness,
    rpca_sparse_component,
    rpca_vessel_enhanced_video,
)
from angio_flux.training.pretrain import AngioFluxSSL


def _run_pipeline_module():
    path = Path("scripts/run_pipeline.py")
    spec = importlib.util.spec_from_file_location("run_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _erode_mask(mask: torch.Tensor, px: int) -> torch.Tensor:
    out = mask.float()
    kernel = torch.ones(1, 1, 3, 3, device=mask.device, dtype=mask.dtype)
    for _ in range(px):
        out = (torch.nn.functional.conv2d(out, kernel, padding=1) >= 9).float()
    return out


def test_vqfr_zero_stenosis_is_high_and_severity_lowers_output():
    head = VQFRHead(embed_dim=4)
    with torch.no_grad():
        for param in head.delta.parameters():
            param.zero_()

    graph_embed = torch.zeros(1, 4)
    venturi_zero = torch.zeros(1, 5)
    no_stenosis = torch.full((1, 5, 4), -20.0)
    no_stenosis[..., 0] = 20.0

    severe = torch.full((1, 5, 4), -20.0)
    severe[..., -1] = 20.0

    qfr_normal = head(graph_embed, venturi_zero, no_stenosis)
    qfr_severe = head(graph_embed, torch.ones(1, 5) * 2.0, severe)

    assert qfr_normal.item() > 0.8
    assert qfr_severe.item() < qfr_normal.item()


def test_soft_skeleton_preserves_thin_line_inside_mask():
    mask = torch.zeros(1, 1, 9, 9)
    mask[:, :, 4, 2:7] = 1.0

    skel = soft_skeleton(mask, iters=3)

    assert skel.sum().item() > 0
    assert ((skel > 0) & (mask == 0)).sum().item() == 0


def test_iou_in_roi_uses_true_roi_restricted_union():
    run_pipeline = _run_pipeline_module()
    pred = np.array([[1, 1, 0], [0, 1, 0], [1, 0, 1]], dtype=float)
    gt = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]], dtype=float)
    roi = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=float)

    assert run_pipeline.iou_in_roi(pred, gt, roi) == pytest.approx(0.5)


def test_frangi_responds_on_centerline_for_dark_and_bright_tubes():
    dark_tube = torch.ones(1, 1, 64, 64) * 0.8
    dark_tube[:, :, 30:34, 10:54] = 0.2
    dark_response = frangi_vesselness(dark_tube, dark_on_bright=True)

    bright_tube = torch.zeros(1, 1, 64, 64)
    bright_tube[:, :, 30:34, 10:54] = 1.0
    bright_response = frangi_vesselness(bright_tube, dark_on_bright=False)

    assert dark_response[0, 0, 32, 32].item() > 0.5 * dark_response.max().item()
    assert bright_response[0, 0, 32, 32].item() > 0.5 * bright_response.max().item()


def test_bolus_peak_frame_uses_mid_sequence_darkening_not_tail():
    video = torch.ones(1, 1, 8, 48, 48) * 0.8
    roi = torch.ones(1, 1, 48, 48)
    video[:, :, 2, 22:25, 10:38] = 0.45
    video[:, :, 3, 22:25, 10:38] = 0.10
    video[:, :, 4, 22:25, 10:38] = 0.35

    peak, peak_idx = bolus_peak_frame(
        video,
        roi=roi,
        peak_window=1,
        top_fraction=0.02,
        highpass_sigma=1.0,
    )

    assert peak_idx.item() == 3
    assert peak[0, 0, 23, 20].item() == pytest.approx(0.10)


def test_rpca_sparse_component_highlights_transient_dark_structure():
    background = torch.linspace(0.35, 0.85, 16).view(1, 1, 1, 1, 16).expand(1, 1, 5, 16, 16).clone()
    video = background.clone()
    video[:, :, 2, 7:10, 4:12] -= 0.35

    sparse = rpca_sparse_component(video, max_iter=8, tol=1.0e-4)

    vessel_signal = (-sparse[0, 0, 2, 7:10, 4:12]).clamp_min(0).mean()
    background_signal = sparse[0, 0, :, :3, :3].abs().mean()
    assert vessel_signal > background_signal


def test_rpca_vessel_enhanced_video_keeps_vessels_as_dark_inflow_signal():
    video = torch.ones(1, 1, 5, 24, 24) * 0.8
    sparse = torch.zeros_like(video)
    sparse[:, :, 2, 10:14, 6:18] = -0.4

    enhanced, dark_sparse = rpca_vessel_enhanced_video(video, rpca_sparse=sparse)

    assert dark_sparse[0, 0, 2, 11:13, 8:16].mean().item() > 0.9
    assert enhanced[0, 0, 2, 11:13, 8:16].mean().item() < 0.1
    assert enhanced[0, 0, 0].mean().item() > 0.95


def test_append_contrast_channels_adds_presence_and_flow_state():
    events = torch.zeros(1, 4, 3, 8, 8)
    rpca_dark = torch.zeros(1, 1, 4, 8, 8)
    rpca_dark[:, :, 1, 2:4, 2:4] = 0.3
    rpca_dark[:, :, 2, 2:4, 2:4] = 0.8
    rpca_dark[:, :, 3, 2:4, 2:4] = 0.2

    augmented = append_contrast_channels(events, rpca_dark, ["presence", "inflow", "washout"])

    assert augmented.shape == (1, 7, 3, 8, 8)
    assert augmented[:, 4, 1, 2:4, 2:4].mean().item() == pytest.approx(0.8)
    assert augmented[:, 5, 1, 2:4, 2:4].mean().item() == pytest.approx(0.5)
    assert augmented[:, 6, 2, 2:4, 2:4].mean().item() == pytest.approx(0.6)


def test_contrast_flow_prior_prefers_transient_injection_over_static_signal():
    rpca_dark = torch.zeros(1, 1, 5, 16, 16)
    rpca_dark[:, :, :, 2:5, 2:5] = 0.9
    rpca_dark[:, :, 2:4, 9:12, 4:12] = 0.9

    prior = contrast_flow_prior_map(rpca_dark, min_flow_weight=0.05)

    transient = prior[0, 0, 10:11, 6:10].mean()
    static = prior[0, 0, 2:5, 2:5].mean()
    assert transient > static * 3.0


def test_real_pseudo_gt_does_not_include_roi_frame():
    cfg_path = Path("configs/pretrain.yaml")
    if not cfg_path.exists():
        pytest.skip("pretrain config not available")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["data"]["root"])
    if not root.exists():
        pytest.skip("real angio data not available")

    dataset = AngioSequenceDataset(
        root=root,
        target_size=(128, 128),
        max_frames=cfg["data"].get("max_frames", 12),
        min_frames=cfg["data"].get("min_frames", 6),
    )
    if len(dataset) == 0:
        pytest.skip("real angio dataset is empty")

    video = dataset[0]["video"].unsqueeze(0)
    events = HemodynamicEventEncoder()(video)
    targets = build_targets(video, events)

    roi = targets["roi"]
    hard = targets["hard_mask"]
    roi_frame = (roi - _erode_mask(roi, 5)).clamp_min(0)

    assert (hard * roi_frame).sum().item() == 0


def test_build_targets_exposes_soft_training_support_maps():
    video = torch.ones(1, 1, 8, 48, 48) * 0.8
    video[:, :, 2:5, 22:25, 10:38] = 0.15
    events = HemodynamicEventEncoder(theta0=0.02, drift_window=9)(video)

    targets = build_targets(
        video,
        events,
        pseudo_gt_cfg={
            "threshold": 0.03,
            "positive_quantile": 0.94,
            "edge_margin_px": 1,
            "scale_reference_size": 48,
            "baseline_quantile": 0.85,
            "peak_window": 1,
        },
    )

    assert targets["mask_target"].shape == targets["soft_mask"].shape
    assert targets["regression_mask"].sum().item() > 0
    assert targets["bat_time"].amax().item() <= 1.0
    assert targets["amp_full"].amax().item() <= 1.0
    assert targets["peak_scores"].shape == (1, 8)


def test_motion_events_without_contrast_do_not_create_flow_support():
    video = torch.ones(1, 1, 8, 48, 48) * 0.7
    events = torch.zeros(1, 4, 7, 48, 48)
    events[:, 2, :, 10:38, 20:24] = 1.0
    events[:, 3, :, 10:38, 24:28] = 1.0

    targets = build_targets(
        video,
        events,
        pseudo_gt_cfg={
            "threshold": 0.05,
            "positive_quantile": 0.94,
            "edge_margin_px": 1,
            "scale_reference_size": 48,
            "baseline_quantile": 0.85,
            "peak_window": 1,
        },
    )

    assert targets["contrast_support"].sum().item() == pytest.approx(0.0)
    assert targets["dynamic_support"].sum().item() == pytest.approx(0.0)
    assert targets["mask_target"].sum().item() == pytest.approx(0.0)


def test_ssl_regression_loss_penalizes_background_bat_amp_predictions():
    loss_fn = SSLPretrainLoss(regression_bg_weight=0.5)
    roi = torch.ones(1, 1, 8, 8)
    regression_mask = torch.zeros(1, 1, 8, 8)
    regression_mask[:, :, 3:5, 3:5] = 1.0
    out = {
        "recon": torch.zeros(1, 1, 8, 8),
        "mask_logits": torch.zeros(1, 1, 8, 8),
        "mask": torch.zeros(1, 1, 8, 8),
        "bat": torch.ones(1, 1, 8, 8),
        "amp": torch.ones(1, 1, 8, 8),
        "events": torch.zeros(1, 4, 3, 8, 8),
        "spike_rates": [torch.zeros(1, 1, 8, 8)],
    }
    targets = {
        "roi": roi,
        "peak_frame": torch.zeros(1, 1, 8, 8),
        "soft_mask": regression_mask,
        "hard_mask": regression_mask,
        "mask_target": regression_mask,
        "regression_mask": regression_mask,
        "bat_time": torch.ones(1, 1, 8, 8),
        "amp_full": torch.ones(1, 1, 8, 8),
        "bat": regression_mask,
        "amp": regression_mask,
    }

    _, parts = loss_fn(out, torch.zeros(1, 1, 3, 8, 8), targets=targets)

    assert parts["bat"].item() > 0.0
    assert parts["amp"].item() > 0.0


def test_ssl_flow_prior_loss_penalizes_raw_predictions_without_flow():
    loss_fn = SSLPretrainLoss(lambda_flow_prior=1.0)
    roi = torch.ones(1, 1, 8, 8)
    out = {
        "recon": torch.zeros(1, 1, 8, 8),
        "mask_logits": torch.zeros(1, 1, 8, 8),
        "mask_raw": torch.ones(1, 1, 8, 8),
        "mask": torch.zeros(1, 1, 8, 8),
        "flow_gate": torch.zeros(1, 1, 8, 8),
        "bat": torch.zeros(1, 1, 8, 8),
        "amp": torch.zeros(1, 1, 8, 8),
        "events": torch.zeros(1, 4, 3, 8, 8),
        "spike_rates": [torch.zeros(1, 1, 8, 8)],
    }
    targets = {
        "roi": roi,
        "target_roi": roi,
        "peak_frame": torch.zeros(1, 1, 8, 8),
        "soft_mask": torch.zeros(1, 1, 8, 8),
        "hard_mask": torch.zeros(1, 1, 8, 8),
        "mask_target": torch.zeros(1, 1, 8, 8),
        "regression_mask": torch.zeros(1, 1, 8, 8),
        "bat_time": torch.zeros(1, 1, 8, 8),
        "amp_full": torch.zeros(1, 1, 8, 8),
    }

    _, parts = loss_fn(out, torch.zeros(1, 1, 3, 8, 8), targets=targets)

    assert parts["flow_prior"].item() == pytest.approx(1.0)


def test_ssl_wrapper_uses_rpca_contrast_channels_and_gates_regression_heads():
    with open("configs/pretrain.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["sunet"]["base_channels"] = 8
    cfg["flux_gnn"]["hidden_dim"] = 16
    cfg["flux_gnn"]["knn_k"] = 4
    cfg["pseudo_gt"]["rpca_max_iter"] = 2

    video = torch.ones(1, 1, 6, 32, 32) * 0.8
    video[:, :, 2:4, 14:17, 8:24] = 0.2
    sparse = rpca_sparse_component(video, max_iter=2)
    model = AngioFluxSSL(cfg)
    out = model(video, rpca_sparse=sparse)

    assert out["events"].shape[1] == 4
    assert out["seg_events"].shape[1] == 7
    assert out["flow_gate"] is not None
    assert torch.all(out["bat"] <= out["bat_raw"] + 1e-6)
    assert torch.all(out["amp"] <= out["amp_raw"] + 1e-6)
