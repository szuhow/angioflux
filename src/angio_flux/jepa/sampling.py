"""Flow-aware sampling helpers.

We use the RPCA-dark sparse component (vessel-enhanced) to estimate, for
each tubelet, the approximate time at which contrast first arrives in that
spatial location. The argmax over time of the spatial-mean dark signal
within a tubelet is used as a proxy for bolus arrival time (BAT-proxy).

These proxies are used both for biased target sampling and for the
auxiliary flow-ordering rank loss.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F


@torch.no_grad()
def rpca_inflow_times(
    rpca_dark: torch.Tensor,
    grid: tuple[int, int, int],
    presence_quantile: float = 0.85,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-tubelet inflow time and presence score from RPCA-dark video.

    Args:
        rpca_dark: (B, 1, T, H, W) sparse component, dark-positive in [0, 1+]
        grid: (Gt, Gh, Gw) target token grid
        presence_quantile: per-batch quantile used to threshold "has inflow"

    Returns:
        inflow_time: (B, N) absolute frame index (0..T-1) of presumed bolus
            arrival inside the tubelet. -1 if the tubelet has no contrast.
        presence: (B, N) float in [0, 1] (mean dark signal inside tubelet).
    """
    b, _, t, h, w = rpca_dark.shape
    gt, gh, gw = grid
    # Accept either signed RPCA sparse (dark = negative) or pre-normalised
    # dark-positive signal. Convert to a dark-positive map.
    dark_pos = (-rpca_dark).clamp_min(0.0) if rpca_dark.min() < 0 else rpca_dark.clamp_min(0.0)
    # Pool spatially to (B, 1, T, gh, gw)
    pooled = F.adaptive_avg_pool3d(dark_pos, (t, gh, gw))  # (B, 1, T, gh, gw)
    # Split T into gt buckets for the argmax-of-time-per-bucket
    t_per = max(1, t // gt)
    # Per-tubelet mean and per-frame max-of-time
    pooled = pooled[:, 0]  # (B, T, gh, gw)
    # Reshape time into (gt, t_per) — drop overflow frames
    usable = gt * t_per
    pooled_usable = pooled[:, :usable].reshape(b, gt, t_per, gh, gw)
    presence_grid = pooled_usable.mean(dim=2)  # (B, gt, gh, gw)
    presence = presence_grid.reshape(b, -1)
    thr = torch.quantile(presence, presence_quantile, dim=1, keepdim=True)
    has_inflow = presence >= thr
    # inflow time = argmax within tubelet's temporal window + offset
    local_argmax = pooled_usable.argmax(dim=2)  # (B, gt, gh, gw)
    base = torch.arange(gt, device=rpca_dark.device).view(1, gt, 1, 1) * t_per
    inflow = (local_argmax + base).reshape(b, -1).long()
    inflow = torch.where(has_inflow, inflow, torch.full_like(inflow, -1))
    return inflow, presence


def flow_aware_target_indices(
    presence: torch.Tensor,
    inflow_time: torch.Tensor,
    num_targets: int = 8,
    bias_strength: float = 0.7,
    rng: random.Random | None = None,
) -> torch.Tensor:
    """Sample target token indices biased toward tubelets with contrast.

    Args:
        presence: (N,) per-tubelet contrast presence
        inflow_time: (N,) inflow time, -1 if no inflow
        num_targets: number of target tokens to sample
        bias_strength: 0 = uniform, 1 = sample only inflow tubelets
    Returns:
        idx: (num_targets,) long
    """
    rng = rng or random.Random()
    device = presence.device
    n = presence.numel()
    p = presence.reshape(-1).float().clone().clamp(min=0)
    if p.sum() < 1e-8:
        weights = torch.ones(n, device=device) / n
    else:
        p = p / (p.sum() + 1e-8)
        uniform = torch.ones(n, device=device) / n
        weights = bias_strength * p + (1.0 - bias_strength) * uniform
    weights = weights / weights.sum()
    idx = torch.multinomial(
        weights, num_samples=min(num_targets, n), replacement=False
    )
    return idx
