"""Self-supervised targets and losses for Angio-FLUX pretraining.

Targets (all derived from input video, no human GT):
    * vessel mask: combined Frangi + event density, restricted to ROI
    * bolus arrival time (BAT): per-pixel argmax of inflow events
    * peak amplitude: per-pixel max |Δlog I|
    * future frame reconstruction (E2VID-style)
    * polarity consistency (inflow vs washout shouldn't overlap)
    * spike rate energy budget
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..preprocess import temporal_std_roi, vessel_pseudo_gt


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).abs() * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def _masked_zero_l1(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (pred.abs() * mask).sum() / mask.sum().clamp_min(1.0)


def _masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: float = 5.0,
) -> torch.Tensor:
    pw = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    per_pixel = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pw, reduction="none"
    )
    return (per_pixel * mask).sum() / mask.sum().clamp_min(1.0)


def _masked_soft_dice(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = (pred * mask).flatten(1)
    target = (target * mask).flatten(1)
    inter = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    return (1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)).mean()


class FrameReconstructor(nn.Module):
    """Predicts peak-frame intensity from S-UNet decoder features."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features))


class DenseRegressionHead(nn.Module):
    """Dense per-pixel regression head (used for BAT and amplitude)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features))


@torch.no_grad()
def build_targets(
    video: torch.Tensor,
    events: torch.Tensor,
    pseudo_gt_cfg: dict | None = None,
    rpca_sparse: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute ROI mask + pseudo-GT for mask, BAT, amplitude."""
    pseudo_gt_cfg = pseudo_gt_cfg or {}
    roi = temporal_std_roi(
        video,
        mean_quantile=pseudo_gt_cfg.get("roi_mean_quantile", 0.55),
        shrink_frac=pseudo_gt_cfg.get("roi_shrink_frac", 0.10),
    )
    frangi_sigmas = pseudo_gt_cfg.get("frangi_sigmas")
    if frangi_sigmas is not None:
        frangi_sigmas = tuple(float(sigma) for sigma in frangi_sigmas)
    pgt = vessel_pseudo_gt(
        video,
        events,
        roi=roi,
        threshold=pseudo_gt_cfg.get("threshold", 0.18),
        positive_quantile=pseudo_gt_cfg.get("positive_quantile", 0.78),
        edge_margin_px=pseudo_gt_cfg.get("edge_margin_px", 7),
        scale_reference_size=pseudo_gt_cfg.get("scale_reference_size", 192.0),
        frangi_sigmas=frangi_sigmas,
        baseline_quantile=pseudo_gt_cfg.get("baseline_quantile", 0.85),
        highpass_sigma=pseudo_gt_cfg.get("highpass_sigma", 4.0),
        fusion_blur_sigma=pseudo_gt_cfg.get("fusion_blur_sigma", 0.7),
        peak_window=pseudo_gt_cfg.get("peak_window", 3),
        peak_top_fraction=pseudo_gt_cfg.get("peak_top_fraction", 0.02),
        use_rpca=pseudo_gt_cfg.get("use_rpca", False),
        rpca_sparse=rpca_sparse,
        rpca_lam=pseudo_gt_cfg.get("rpca_lam"),
        rpca_max_iter=pseudo_gt_cfg.get("rpca_max_iter", 20),
        rpca_tol=pseudo_gt_cfg.get("rpca_tol", 1.0e-5),
    )
    pgt["roi"] = roi
    return pgt


class SSLPretrainLoss(nn.Module):
    """Bundle of self-supervised losses with ROI masking and pos-weighted BCE."""

    def __init__(
        self,
        lambda_recon: float = 1.0,
        lambda_mask: float = 1.5,
        lambda_bat: float = 0.5,
        lambda_amp: float = 0.5,
        lambda_consist: float = 0.1,
        lambda_spike: float = 1.0e-3,
        lambda_flow_prior: float = 0.0,
        spike_target_rate: float = 0.05,
        pos_weight: float = 5.0,
        mask_bg_weight: float = 1.0,
        regression_bg_weight: float = 0.15,
    ) -> None:
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_mask = lambda_mask
        self.lambda_bat = lambda_bat
        self.lambda_amp = lambda_amp
        self.lambda_consist = lambda_consist
        self.lambda_spike = lambda_spike
        self.lambda_flow_prior = lambda_flow_prior
        self.spike_target_rate = spike_target_rate
        self.pos_weight = pos_weight
        self.mask_bg_weight = mask_bg_weight
        self.regression_bg_weight = regression_bg_weight

    def forward(
        self, out: dict, video: torch.Tensor, targets: dict | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if targets is None:
            targets = build_targets(video, out["events"])
        roi = targets["roi"]

        target_frame = targets.get("peak_frame")
        if target_frame is None:
            target_frame = video[:, :, -3:].mean(dim=2)  # (B, 1, H, W)
        l_recon = _masked_l1(out["recon"], target_frame, roi)

        # BCE on the full image so the model is forced to predict 0 outside
        # the ROI as well. Inside ROI uses pos_weight to balance vessel/bg;
        # outside ROI target is 0 → pulls predictions to background.
        full = torch.ones_like(roi)
        target_roi = targets.get("target_roi", roi)
        mask_loss_weight = target_roi + self.mask_bg_weight * (full - target_roi)
        mask_target = targets.get("mask_target")
        if mask_target is None:
            mask_target = torch.maximum(targets["soft_mask"], targets["hard_mask"])
        l_mask_bce = _masked_bce_with_logits(
            out["mask_logits"], mask_target, mask_loss_weight, pos_weight=self.pos_weight
        )
        l_mask_dice = _masked_soft_dice(out["mask"], mask_target, target_roi)
        l_mask = 0.5 * l_mask_bce + 0.5 * l_mask_dice

        l_flow_prior = torch.zeros((), device=video.device, dtype=video.dtype)
        flow_gate = out.get("flow_gate")
        if flow_gate is not None:
            raw_mask = out.get("mask_raw", out["mask"])
            non_flow = (1.0 - flow_gate).clamp(0, 1) * target_roi
            l_flow_prior = _masked_zero_l1(raw_mask, non_flow)

        # BAT/amplitude are meaningful only where the pseudo vessel support is
        # non-zero. Supervising them over the whole background makes all-zero
        # maps an easy local optimum.
        regression_mask = targets.get("regression_mask")
        if regression_mask is None:
            regression_mask = torch.maximum(targets["soft_mask"], targets["hard_mask"])
        bat_target = targets["bat_time"] if "bat_time" in targets else targets["bat"]
        amp_target = targets["amp_full"] if "amp_full" in targets else targets["amp"]
        l_bat_fg = _masked_l1(out["bat"], bat_target, regression_mask)
        l_amp_fg = _masked_l1(out["amp"], amp_target, regression_mask)
        regression_bg = (full - regression_mask).clamp(0, 1)
        l_bat = l_bat_fg + self.regression_bg_weight * _masked_zero_l1(out["bat"], regression_bg)
        l_amp = l_amp_fg + self.regression_bg_weight * _masked_zero_l1(out["amp"], regression_bg)

        ev = out["events"]
        l_consist = (ev[:, 0] * ev[:, 1]).mean()

        spike_terms = [(sr.mean() - self.spike_target_rate).abs() for sr in out["spike_rates"]]
        l_spike = torch.stack(spike_terms).mean()

        total = (
            self.lambda_recon * l_recon
            + self.lambda_mask * l_mask
            + self.lambda_bat * l_bat
            + self.lambda_amp * l_amp
            + self.lambda_consist * l_consist
            + self.lambda_spike * l_spike
            + self.lambda_flow_prior * l_flow_prior
        )
        return total, {
            "recon": l_recon.detach(),
            "mask": l_mask.detach(),
            "flow_prior": l_flow_prior.detach(),
            "bat": l_bat.detach(),
            "amp": l_amp.detach(),
            "consist": l_consist.detach(),
            "spike": l_spike.detach(),
            "total": total.detach(),
        }


# Back-compat shim (older code calls pseudo_vessel_mask).
def pseudo_vessel_mask(events: torch.Tensor) -> torch.Tensor:
    """Legacy helper — density-only pseudo mask, kept for compatibility."""
    density = events.sum(dim=(1, 2), keepdim=False).unsqueeze(1)
    dmax = density.amax(dim=(2, 3), keepdim=True).clamp_min(1.0)
    density = density / dmax
    return density
