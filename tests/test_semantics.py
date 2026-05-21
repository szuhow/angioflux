import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from angio_flux.data import AngioSequenceDataset
from angio_flux.encoding import HemodynamicEventEncoder
from angio_flux.losses.multitask import soft_skeleton
from angio_flux.losses.ssl import build_targets
from angio_flux.modules.vqfr import VQFRHead
from angio_flux.preprocess import frangi_vesselness


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