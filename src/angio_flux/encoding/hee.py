"""Hemodynamic Event Encoder (HEE).

Converts a DICOM cine sequence (B, 1, T, H, W) into a 4-channel event stream:
    p_in  : contrast inflow (pixel darkens for the first time in the window)
    p_out : contrast washout (pixel brightens after a prior inflow)
    p_LE  : leading edge of vessel motion (spatial gradient grows)
    p_TE  : trailing edge of vessel motion (spatial gradient decays)

Threshold is *adaptive* per pixel/time, driven by local coefficient of
variation in a 3D window. High-CV regions get a lower threshold than
uniform drift regions, while low-CV drift is assigned a higher threshold so
it tends to fall silent.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _local_stats(x: torch.Tensor, k_h: int, k_w: int, k_t: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute local mean and std over (k_t, k_h, k_w) window via 3D avg pooling.

    x: (B, 1, T, H, W)
    Returns (mean, std) of same shape.
    """
    pad = (k_w // 2, k_w // 2, k_h // 2, k_h // 2, k_t // 2, k_t // 2)
    xp = F.pad(x, pad, mode="replicate")
    mean = F.avg_pool3d(xp, kernel_size=(k_t, k_h, k_w), stride=1)
    mean_sq = F.avg_pool3d(xp * xp, kernel_size=(k_t, k_h, k_w), stride=1)
    var = (mean_sq - mean * mean).clamp_min(0.0)
    return mean, var.sqrt()


def _adaptive_threshold(cv: torch.Tensor, theta0: float, alpha: float) -> torch.Tensor:
    """Map local CV to an event threshold.

    Low CV / uniform drift receives up to ``theta0 * (1 + alpha)``. High CV
    lowers the threshold toward ``theta0`` or below, making localized contrast
    changes easier to emit as events.
    """
    return theta0 * (1.0 + alpha) / (1.0 + alpha * cv)


def _remove_spatial_drift(delta_log: torch.Tensor, window: int) -> torch.Tensor:
    """Subtract low-frequency frame-to-frame brightness drift.

    C-arm exposure changes, breathing, and table/body motion can darken large
    regions at once. Vessels are spatially sparse, so removing a broad local
    average keeps narrow bolus changes while suppressing anatomy-wide drift.
    """
    if window <= 1 or min(delta_log.shape[-2:]) < 8:
        return delta_log
    if window % 2 == 0:
        window += 1
    pad = (window // 2, window // 2, window // 2, window // 2, 0, 0)
    drift = F.avg_pool3d(
        F.pad(delta_log, pad, mode="replicate"),
        kernel_size=(1, window, window),
        stride=1,
    )
    return delta_log - drift


class HemodynamicEventEncoder(nn.Module):
    """4-channel adaptive event encoder.

    Args:
        theta0:   base log-intensity threshold.
        alpha:    adaptive scaling factor for local CV.
        window:   (k_h, k_w, k_t) local stats window.
        drift_window: spatial window used to remove low-frequency brightness drift.
        theta_min_frac: lower bound for adaptive threshold as a fraction of theta0.
        eps:      numerical floor for log/divisions.
    """

    def __init__(
        self,
        theta0: float = 0.05,
        alpha: float = 2.0,
        window: tuple[int, int, int] = (7, 7, 5),
        drift_window: int = 31,
        theta_min_frac: float = 0.6,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.theta0 = theta0
        self.alpha = alpha
        self.k_h, self.k_w, self.k_t = window
        self.drift_window = drift_window
        self.theta_min_frac = theta_min_frac
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, T, H, W) in [0, 1].
        Returns:
            events: (B, 4, T-1, H, W) — channels (p_in, p_out, p_LE, p_TE),
                    each in {0, 1}.
        """
        if x.dim() != 5 or x.size(1) != 1:
            raise ValueError(f"expected (B,1,T,H,W), got {tuple(x.shape)}")

        x = x.clamp_min(self.eps)
        x_log = torch.log(x)
        delta_log = x_log[:, :, 1:] - x_log[:, :, :-1]  # (B,1,T-1,H,W)
        delta_log = _remove_spatial_drift(delta_log, self.drift_window)

        # ---- adaptive threshold field ----
        mean, std = _local_stats(x, self.k_h, self.k_w, self.k_t)
        cv = std / (mean + self.eps)
        # align cv with delta (drop first time slice)
        cv = cv[:, :, 1:]
        theta = _adaptive_threshold(cv, self.theta0, self.alpha)  # (B,1,T-1,H,W)
        theta = theta.clamp_min(self.theta0 * self.theta_min_frac)

        # ---- inflow / washout (polarity-aware, with hysteresis) ----
        darkening = delta_log <= -theta
        brightening = delta_log >= theta

        # Hysteresis: p_in is the first darkening event per pixel; p_out only
        # fires after a prior inflow has occurred earlier in the sequence.
        inflow_history = torch.cummax(darkening.float(), dim=2).values
        prior_inflow = F.pad(inflow_history[:, :, :-1], (0, 0, 0, 0, 1, 0))
        p_in = (darkening & (prior_inflow <= 0)).float()
        p_out = (brightening & (prior_inflow > 0)).float()

        # ---- leading / trailing spatial-gradient edges ----
        # |grad I| over space, finite-diff.
        gx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        gy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        grad_mag = (gx * gx + gy * gy).sqrt()  # (B,1,T,H,W)
        d_grad = grad_mag[:, :, 1:] - grad_mag[:, :, :-1]  # (B,1,T-1,H,W)

        p_LE = (d_grad >= theta).float()
        p_TE = (d_grad <= -theta).float()

        events = torch.cat([p_in, p_out, p_LE, p_TE], dim=1)  # (B,4,T-1,H,W)
        return events


def events_to_voxel(events: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Aggregate 4-channel events into a fixed-size voxel grid for dense modules.

    Args:
        events:   (B, 4, T, H, W)
        num_bins: temporal bins per polarity channel.
    Returns:
        voxel:    (B, 4 * num_bins, H, W)
    """
    b, c, t, h, w = events.shape
    if c != 4:
        raise ValueError(f"expected 4 polarity channels, got {c}")
    bin_size = max(1, t // num_bins)
    bins = []
    for i in range(num_bins):
        t0 = i * bin_size
        t1 = (i + 1) * bin_size if i < num_bins - 1 else t
        bins.append(events[:, :, t0:t1].sum(dim=2))  # (B,4,H,W)
    voxel = torch.cat(bins, dim=1)  # (B, 4*num_bins, H, W)
    return voxel


def contrast_channel_count(cfg: dict) -> int:
    """Number of continuous RPCA contrast channels appended to S-UNet input."""
    hee_cfg = cfg.get("hee", {})
    if hee_cfg.get("event_input", "raw") != "rpca":
        return 0
    return len(hee_cfg.get("contrast_channels", []))


def append_contrast_channels(
    events: torch.Tensor,
    rpca_dark: torch.Tensor | None,
    channels: list[str] | tuple[str, ...] | None,
) -> torch.Tensor:
    """Append continuous contrast-state channels aligned to event timesteps.

    HEE events say that a threshold crossing happened. These channels preserve
    whether that crossing occurred inside RPCA-highlighted contrast, which is
    what separates vessel bolus motion from moving anatomy edges.
    """
    channels = list(channels or [])
    if rpca_dark is None or not channels:
        return events
    if rpca_dark.dim() != 5 or rpca_dark.shape[1] != 1:
        raise ValueError(f"expected rpca_dark (B,1,T,H,W), got {tuple(rpca_dark.shape)}")

    event_steps = events.shape[2]
    presence = rpca_dark[:, :, 1:]
    delta = rpca_dark[:, :, 1:] - rpca_dark[:, :, :-1]
    candidates = {
        "presence": presence,
        "inflow": delta.clamp_min(0),
        "washout": (-delta).clamp_min(0),
    }

    extra = []
    for name in channels:
        if name not in candidates:
            raise ValueError(f"unknown contrast channel {name!r}")
        channel = candidates[name]
        if channel.shape[2] > event_steps:
            channel = channel[:, :, :event_steps]
        elif channel.shape[2] < event_steps:
            pad = channel[:, :, -1:].expand(-1, -1, event_steps - channel.shape[2], -1, -1)
            channel = torch.cat([channel, pad], dim=2)
        extra.append(channel.to(device=events.device, dtype=events.dtype))
    return torch.cat([events, *extra], dim=1)


def contrast_prior_map(
    rpca_dark: torch.Tensor | None,
    gamma: float = 1.0,
) -> torch.Tensor | None:
    """Dense vessel-contrast prior from RPCA dark signal.

    This is not a vessel mask by itself; it gates learned predictions away from
    anatomy that may move but never carries contrast.
    """
    if rpca_dark is None:
        return None
    if rpca_dark.dim() != 5 or rpca_dark.shape[1] != 1:
        raise ValueError(f"expected rpca_dark (B,1,T,H,W), got {tuple(rpca_dark.shape)}")
    prior = rpca_dark[:, :, 1:].amax(dim=2).clamp(0, 1)
    gamma = max(0.25, float(gamma))
    if gamma != 1.0:
        prior = prior.pow(gamma)
    return prior


def _normalize_batched_map(x: torch.Tensor) -> torch.Tensor:
    """Normalize each sample's spatial map to [0, 1] without cross-sample leakage."""
    scale = x.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1.0e-6)
    return (x / scale).clamp(0, 1)


def contrast_flow_prior_map(
    rpca_dark: torch.Tensor | None,
    events: torch.Tensor | None = None,
    gamma: float = 1.0,
    min_flow_weight: float = 0.10,
) -> torch.Tensor | None:
    """Temporal contrast-injection prior for vessel-carrying pixels.

    ``contrast_prior_map`` marks any pixel that was RPCA-dark at least once.
    This stricter prior additionally requires transient inflow/range behavior,
    so static anatomy or persistent motion artifacts do not dominate the mask.
    """
    if rpca_dark is None:
        return None
    if rpca_dark.dim() != 5 or rpca_dark.shape[1] != 1:
        raise ValueError(f"expected rpca_dark (B,1,T,H,W), got {tuple(rpca_dark.shape)}")
    if rpca_dark.shape[2] < 2:
        return contrast_prior_map(rpca_dark, gamma=gamma)

    presence = rpca_dark[:, :, 1:].amax(dim=2).clamp(0, 1)
    delta = rpca_dark[:, :, 1:] - rpca_dark[:, :, :-1]
    inflow = _normalize_batched_map(delta.clamp_min(0).sum(dim=2))
    washout = _normalize_batched_map((-delta).clamp_min(0).sum(dim=2))
    temporal_range = _normalize_batched_map(
        (rpca_dark[:, :, 1:].amax(dim=2) - rpca_dark[:, :, 1:].amin(dim=2)).clamp_min(0)
    )

    flow_state = (0.55 * inflow + 0.30 * temporal_range + 0.15 * washout).clamp(0, 1)
    if events is not None:
        if events.dim() != 5 or events.shape[1] < 2:
            raise ValueError(f"expected events (B,C,T,H,W) with C>=2, got {tuple(events.shape)}")
        hemo = events[:, :2].sum(dim=(1, 2), keepdim=False).unsqueeze(1)
        if events.shape[1] > 2:
            motion = events[:, 2:].sum(dim=(1, 2), keepdim=False).unsqueeze(1)
        else:
            motion = torch.zeros_like(hemo)
        hemo_ratio = (hemo / (hemo + motion + 1.0)).clamp(0, 1)
        flow_state = flow_state * (0.35 + 0.65 * hemo_ratio)

    min_flow_weight = min(0.95, max(0.0, float(min_flow_weight)))
    prior = presence * (min_flow_weight + (1.0 - min_flow_weight) * flow_state)
    gamma = max(0.25, float(gamma))
    if gamma != 1.0:
        prior = prior.pow(gamma)
    return prior.clamp(0, 1)
