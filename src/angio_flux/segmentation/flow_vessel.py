"""Self-supervised vessel segmentation from contrast-flow physics.

Fuses RPCA transient-dark signal, hemodynamic event causality (inflow→washout),
and tubular Frangi support from ``vessel_pseudo_gt``. No human labels required.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoding import HemodynamicEventEncoder, contrast_flow_prior_map
from ..losses.ssl import build_targets
from ..preprocess import rpca_sparse_component, rpca_vessel_enhanced_video, temporal_std_roi


@torch.no_grad()
def event_flow_saliency(
    events: torch.Tensor,
    coh_sigma: float = 1.5,
    coh_k: int = 5,
    transit_min: int = 2,
    transit_max: int = 40,
) -> dict[str, torch.Tensor]:
    """Causal bolus saliency from HEE polarity stream (B,4,T,H,W)."""
    assert events.dim() == 5 and events.shape[1] >= 2
    p_in = events[:, 0]
    p_out = events[:, 1]
    b, tm, h, w = p_in.shape

    fired_in = p_in.amax(dim=1) > 0
    t_idx = torch.arange(tm, device=events.device).view(1, tm, 1, 1).float()
    t_in_masked = torch.where(p_in > 0, t_idx, torch.full_like(t_idx, float(tm)))
    t_in = t_in_masked.amin(dim=1)
    t_in_norm = (t_in / max(1.0, float(tm - 1))).clamp(0, 1)

    t_after = torch.where(
        (p_out > 0) & (t_idx > t_in.unsqueeze(1)),
        t_idx,
        torch.full_like(t_idx, float(tm)),
    )
    t_out = t_after.amin(dim=1)
    transit = t_out - t_in
    washout_ok = fired_in & (transit >= transit_min) & (transit <= transit_max)

    fired_f = fired_in.float()
    t_safe = torch.where(fired_in, t_in, torch.zeros_like(t_in))
    pad = coh_k // 2
    num = F.avg_pool2d(
        (t_safe * fired_f).unsqueeze(1),
        kernel_size=coh_k,
        stride=1,
        padding=pad,
        count_include_pad=False,
    ).squeeze(1)
    den = F.avg_pool2d(
        fired_f.unsqueeze(1),
        kernel_size=coh_k,
        stride=1,
        padding=pad,
        count_include_pad=False,
    ).squeeze().clamp_min(1e-6)
    t_local = num / den
    resid = (t_in - t_local).abs()
    coherence = torch.exp(-(resid ** 2) / (2.0 * coh_sigma ** 2))
    coherence = torch.where(fired_in, coherence, torch.zeros_like(coherence))

    saliency = fired_in.float() * washout_ok.float() * coherence
    return {
        "saliency": saliency.unsqueeze(1),
        "arrival_time": t_in_norm.unsqueeze(1),
        "fired_in": fired_in.unsqueeze(1),
        "washout_ok": washout_ok.unsqueeze(1),
        "coherence": coherence.unsqueeze(1),
    }


@torch.no_grad()
def flow_vessel_segment(
    video: torch.Tensor,
    *,
    hee: HemodynamicEventEncoder | None = None,
    pseudo_gt_cfg: dict | None = None,
    rpca_sparse: torch.Tensor | None = None,
    use_rpca: bool = True,
    fuse_weights: tuple[float, float, float, float] = (0.40, 0.30, 0.20, 0.10),
    positive_quantile: float | None = None,
    hard_threshold: float | None = None,
    coh_sigma: float = 1.5,
    coh_k: int = 5,
    transit_min: int = 2,
    transit_max: int = 40,
) -> dict[str, torch.Tensor]:
    """Segment vessels from contrast flow (no learned weights).

    Returns soft score, hard mask, ROI, pseudo-GT maps, and diagnostics.
    """
    if video.dim() == 4:
        video = video.unsqueeze(1)
    assert video.dim() == 5 and video.shape[1] == 1
    pseudo_gt_cfg = dict(pseudo_gt_cfg or {})
    if positive_quantile is not None:
        pseudo_gt_cfg["positive_quantile"] = positive_quantile

    if hee is None:
        hee_cfg = pseudo_gt_cfg.get("hee", {})
        hee = HemodynamicEventEncoder(
            theta0=hee_cfg.get("theta0", 0.05),
            alpha=hee_cfg.get("alpha", 2.0),
            window=tuple(hee_cfg.get("window", (7, 7, 5))),
            drift_window=hee_cfg.get("drift_window", 31),
            theta_min_frac=hee_cfg.get("theta_min_frac", 0.6),
        )

    roi = temporal_std_roi(
        video,
        mean_quantile=pseudo_gt_cfg.get("roi_mean_quantile", 0.55),
        shrink_frac=pseudo_gt_cfg.get("roi_shrink_frac", 0.10),
    )

    sparse = rpca_sparse
    if sparse is None and use_rpca:
        sparse = rpca_sparse_component(
            video,
            lam=pseudo_gt_cfg.get("rpca_lam"),
            max_iter=pseudo_gt_cfg.get("rpca_max_iter", 20),
            tol=pseudo_gt_cfg.get("rpca_tol", 1.0e-5),
        )

    event_video = video
    rpca_dark = None
    if use_rpca and sparse is not None:
        event_video, rpca_dark = rpca_vessel_enhanced_video(
            video,
            rpca_sparse=sparse,
            lam=pseudo_gt_cfg.get("rpca_lam"),
            max_iter=pseudo_gt_cfg.get("rpca_max_iter", 20),
            tol=pseudo_gt_cfg.get("rpca_tol", 1.0e-5),
            quantile=pseudo_gt_cfg.get("rpca_event_quantile", 0.995),
            floor_quantile=pseudo_gt_cfg.get("rpca_event_floor_quantile", 0.0),
            blend=pseudo_gt_cfg.get("rpca_event_blend", 1.0),
        )

    events = hee(event_video)
    pgt = build_targets(video, events, pseudo_gt_cfg=pseudo_gt_cfg, rpca_sparse=sparse)
    evt = event_flow_saliency(
        events,
        coh_sigma=coh_sigma,
        coh_k=coh_k,
        transit_min=transit_min,
        transit_max=transit_max,
    )

    flow_prior = contrast_flow_prior_map(
        rpca_dark,
        events,
        gamma=pseudo_gt_cfg.get("contrast_flow_gamma", 1.0),
        min_flow_weight=pseudo_gt_cfg.get("contrast_flow_min_weight", 0.08),
    )
    if flow_prior is None:
        flow_prior = pgt["contrast_support"]

    w_pgt, w_evt, w_flow, w_rpca = fuse_weights
    soft = (
        w_pgt * pgt["soft_mask"]
        + w_evt * evt["saliency"]
        + w_flow * flow_prior
        + w_rpca * pgt.get("rpca_dark_peak", pgt["contrast_support"])
    )
    soft = soft.clamp(0, 1) * pgt["target_roi"]

    b = soft.shape[0]
    hard_thr = []
    for i in range(b):
        vals = soft[i, 0][pgt["target_roi"][i, 0] > 0.5]
        if vals.numel() == 0:
            hard_thr.append(torch.tensor(hard_threshold or 0.18, device=soft.device))
        elif hard_threshold is not None:
            hard_thr.append(torch.tensor(hard_threshold, device=soft.device))
        else:
            q = pseudo_gt_cfg.get("positive_quantile", 0.88)
            hard_thr.append(torch.quantile(vals, q))
    thr_t = torch.stack(hard_thr).view(b, 1, 1, 1)
    floor = pseudo_gt_cfg.get("threshold", 0.12)
    thr_t = torch.maximum(thr_t, torch.full_like(thr_t, floor))
    hard = (soft > thr_t).float() * pgt["target_roi"]

    return {
        "soft_mask": soft,
        "hard_mask": hard,
        "mask_logits": torch.logit(soft.clamp(1e-4, 1 - 1e-4)),
        "mask": soft,
        "roi": roi,
        "target_roi": pgt["target_roi"],
        "events": events,
        "rpca_dark": rpca_dark,
        "rpca_sparse": sparse,
        "flow_prior": flow_prior,
        "pseudo_gt": pgt,
        "event_saliency": evt["saliency"],
        "arrival_time": evt["arrival_time"],
    }


class FlowVesselSegmenter(nn.Module):
    """Optional light refinement on top of physics-based flow segmentation."""

    def __init__(self, cfg: dict | None = None) -> None:
        super().__init__()
        self.cfg = cfg or {}
        hee_cfg = self.cfg.get("hee", {})
        self.hee = HemodynamicEventEncoder(
            theta0=hee_cfg.get("theta0", 0.05),
            alpha=hee_cfg.get("alpha", 2.0),
            window=tuple(hee_cfg.get("window", (7, 7, 5))),
            drift_window=hee_cfg.get("drift_window", 31),
            theta_min_frac=hee_cfg.get("theta_min_frac", 0.6),
        )
        refine = self.cfg.get("refine", False)
        self.refine_head = None
        if refine:
            ch = int(self.cfg.get("refine_channels", 16))
            self.refine_head = nn.Sequential(
                nn.Conv2d(5, ch, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(ch, 1, 1),
            )

    def forward(
        self,
        video: torch.Tensor,
        rpca_sparse: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        base = flow_vessel_segment(
            video,
            hee=self.hee,
            pseudo_gt_cfg=self.cfg.get("pseudo_gt", {}),
            rpca_sparse=rpca_sparse,
            use_rpca=self.cfg.get("use_rpca", True),
            fuse_weights=tuple(self.cfg.get("fuse_weights", (0.40, 0.30, 0.20, 0.10))),
            coh_sigma=self.cfg.get("coh_sigma", 1.5),
            coh_k=self.cfg.get("coh_k", 5),
            transit_min=self.cfg.get("transit_min", 2),
            transit_max=self.cfg.get("transit_max", 40),
        )
        if self.refine_head is None:
            return base

        stack = torch.cat(
            [
                base["soft_mask"],
                base["flow_prior"],
                base["event_saliency"],
                base["pseudo_gt"]["soft_mask"],
                base["pseudo_gt"]["contrast_support"],
            ],
            dim=1,
        )
        delta = torch.sigmoid(self.refine_head(stack))
        refined = (base["soft_mask"] * (0.65 + 0.35 * delta)).clamp(0, 1) * base["target_roi"]
        out = dict(base)
        out["soft_mask"] = refined
        out["mask"] = refined
        out["mask_logits"] = torch.logit(refined.clamp(1e-4, 1 - 1e-4))
        out["hard_mask"] = (refined > base["hard_mask"]).float() * base["target_roi"]
        return out
