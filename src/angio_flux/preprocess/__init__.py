"""Preprocessing utilities: collimator/ROI detection and Frangi-like vesselness.

All functions are torch-native and differentiable-safe (used only as pseudo-GT
producers — wrap calls in ``torch.no_grad()`` where appropriate).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Collimator / ROI mask                                                       #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def temporal_std_roi(
    video: torch.Tensor,
    mean_quantile: float = 0.55,
    shrink_frac: float = 0.10,
) -> torch.Tensor:
    """Rectangular ROI = bounding box of bright pixels, shrunk inward.

    The angio panel is rectangular and brighter than the collimator gray ring.
    We take the bounding box of pixels above ``mean_quantile`` of the temporal
    mean and shrink it by ``shrink_frac`` of (H, W) on each side. This
    guarantees a clean rectangle that excludes the bright-square edge ring —
    that ring is the dominant false-positive for Frangi/event-density GT.

    Args:
        video: (B, 1, T, H, W) in [0, 1]
        mean_quantile: keep pixels above this per-sample quantile of mean
                       intensity (0.55 = top 45%).
        shrink_frac: inward shrink on each side, as a fraction of (H, W).
    Returns:
        roi: (B, 1, H, W) float mask in {0, 1}
    """
    assert video.dim() == 5 and video.shape[1] == 1, "expected (B, 1, T, H, W)"
    mean_img = video.mean(dim=2)  # (B, 1, H, W)
    b, _, h, w = mean_img.shape
    thr = torch.quantile(
        mean_img.view(b, -1), mean_quantile, dim=1
    ).view(b, 1, 1, 1)
    bright = mean_img > thr  # (B, 1, H, W) bool

    roi = torch.zeros_like(mean_img)
    sh = int(h * shrink_frac)
    sw = int(w * shrink_frac)
    for i in range(b):
        bi = bright[i, 0]
        rows = bi.any(dim=1).nonzero(as_tuple=True)[0]
        cols = bi.any(dim=0).nonzero(as_tuple=True)[0]
        if rows.numel() == 0 or cols.numel() == 0:
            continue
        r0 = int(rows.min().item()) + sh
        r1 = int(rows.max().item()) - sh
        c0 = int(cols.min().item()) + sw
        c1 = int(cols.max().item()) - sw
        if r1 > r0 and c1 > c0:
            roi[i, 0, r0:r1, c0:c1] = 1.0
    return roi


# --------------------------------------------------------------------------- #
# Gaussian smoothing & Hessian                                                #
# --------------------------------------------------------------------------- #
def _gaussian_kernel1d(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur for (B, 1, H, W) tensors."""
    k1d = _gaussian_kernel1d(sigma, x.device, x.dtype)
    r = k1d.numel() // 2
    kx = k1d.view(1, 1, 1, -1)
    ky = k1d.view(1, 1, -1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), kx)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), ky)
    return x


def _spatial_gradient_mag(x: torch.Tensor) -> torch.Tensor:
    """Finite-difference gradient magnitude for (B, 1, H, W)."""
    gx = x[:, :, :, 1:] - x[:, :, :, :-1]
    gy = x[:, :, 1:, :] - x[:, :, :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0), mode="replicate")
    gy = F.pad(gy, (0, 0, 0, 1), mode="replicate")
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def _erode_mask(mask: torch.Tensor, px: int) -> torch.Tensor:
    """Binary erosion for (B, 1, H, W) masks."""
    out = mask.float()
    if px <= 0:
        return out
    k = torch.ones(1, 1, 3, 3, device=mask.device, dtype=mask.dtype)
    for _ in range(px):
        out = (F.conv2d(out, k, padding=1) >= 9).float()
    return out


def _masked_quantile(x: torch.Tensor, mask: torch.Tensor, q: float, fallback: float) -> torch.Tensor:
    """Per-sample quantile over masked pixels, returned as (B,1,1,1)."""
    vals = []
    for i in range(x.shape[0]):
        xi = x[i, 0][mask[i, 0] > 0.5]
        if xi.numel() == 0:
            vals.append(torch.tensor(fallback, device=x.device, dtype=x.dtype))
        else:
            vals.append(torch.quantile(xi, q))
    return torch.stack(vals).view(-1, 1, 1, 1)


@torch.no_grad()
def frangi_vesselness(
    image: torch.Tensor,
    sigmas: tuple[float, ...] = (1.0, 1.8, 2.6, 3.6),
    beta: float = 0.5,
    c: float | None = None,
    dark_on_bright: bool = True,
) -> torch.Tensor:
    """Multi-scale 2D Frangi vesselness filter.

    Args:
        image: (B, 1, H, W) in [0, 1]
        sigmas: list of scales (in pixels) for the Hessian.
        beta: blob-ness sensitivity (typical 0.5).
        c: structureness sensitivity. If None, auto-set per sample to half of
           the maximum Hessian Frobenius norm.
        dark_on_bright: True if vessels are darker than background (XA angio).
    Returns:
        v: (B, 1, H, W) vesselness response in [0, 1].
    """
    assert image.dim() == 4 and image.shape[1] == 1

    # Vessels darker than bg → invert so vessels become "ridges" (bright lines).
    if dark_on_bright:
        image = 1.0 - image

    responses = []
    for sigma in sigmas:
        smoothed = _gaussian_blur(image, sigma)
        # Finite-diff Hessian (5-point).
        gx = 0.5 * (smoothed[..., 2:] - smoothed[..., :-2])
        gx = F.pad(gx, (1, 1, 0, 0), mode="replicate")
        gy = 0.5 * (smoothed[..., 2:, :] - smoothed[..., :-2, :])
        gy = F.pad(gy, (0, 0, 1, 1), mode="replicate")
        gxx = 0.5 * (gx[..., 2:] - gx[..., :-2])
        gxx = F.pad(gxx, (1, 1, 0, 0), mode="replicate")
        gyy = 0.5 * (gy[..., 2:, :] - gy[..., :-2, :])
        gyy = F.pad(gyy, (0, 0, 1, 1), mode="replicate")
        gxy = 0.5 * (gx[..., 2:, :] - gx[..., :-2, :])
        gxy = F.pad(gxy, (0, 0, 1, 1), mode="replicate")

        # Scale-normalize.
        gxx = sigma * sigma * gxx
        gyy = sigma * sigma * gyy
        gxy = sigma * sigma * gxy

        # Eigenvalues of 2x2 symmetric Hessian.
        tr = gxx + gyy
        det = gxx * gyy - gxy * gxy
        disc = torch.clamp(tr * tr / 4 - det, min=0.0).sqrt()
        l1 = tr / 2 + disc
        l2 = tr / 2 - disc
        # Ensure |λ1| ≤ |λ2|
        swap = l1.abs() > l2.abs()
        l1n = torch.where(swap, l2, l1)
        l2n = torch.where(swap, l1, l2)

        rb = (l1n / (l2n + 1e-8)).abs()
        s = torch.sqrt(l1n * l1n + l2n * l2n)

        if c is None:
            b = s.shape[0]
            c_b = s.view(b, -1).max(dim=1).values * 0.5
            c_b = c_b.view(b, 1, 1, 1).clamp_min(1e-3)
        else:
            c_b = c

        # Frangi vessel response. After the optional inversion above, vessels
        # are bright ridges; the largest-curvature eigenvalue is therefore
        # negative near the centerline.
        valid = (l2n < 0).float()
        v_sigma = valid * torch.exp(-(rb ** 2) / (2 * beta ** 2)) * (
            1 - torch.exp(-(s ** 2) / (2 * c_b ** 2))
        )
        responses.append(v_sigma)

    # Take maximum across scales.
    out = torch.stack(responses, dim=0).max(dim=0).values
    return out


@torch.no_grad()
def vessel_pseudo_gt(
    video: torch.Tensor,
    events: torch.Tensor,
    roi: torch.Tensor,
    threshold: float = 0.18,
    positive_quantile: float = 0.78,
    edge_margin_px: int = 7,
) -> dict[str, torch.Tensor]:
    """Combined pseudo ground-truth from Frangi + event density, restricted to ROI.

    Args:
        video:  (B, 1, T, H, W)
        events: (B, 4, T-1, H, W)
        roi:    (B, 1, H, W)
    Returns:
        dict with keys:
            soft_mask:  (B, 1, H, W) in [0, 1] — combined response
            hard_mask:  (B, 1, H, W) in {0, 1}
            bat:        (B, 1, H, W) bolus arrival time, normalized to [0, 1] within sequence
            amp:        (B, 1, H, W) peak amplitude, normalized per-sample
    """
    b = video.shape[0]
    peak = video[:, :, -3:].mean(dim=2)              # (B, 1, H, W) late-bolus frame
    baseline = video[:, :, :2].mean(dim=2)           # (B, 1, H, W) pre-contrast
    mean_img = video.mean(dim=2)
    temporal_std = video.std(dim=2)

    target_roi = _erode_mask(roi, edge_margin_px)

    # Differential signal: vessels darken when contrast arrives.
    dark_signal = (baseline - peak).clamp(min=0.0)   # > 0 where pixel got darker
    # Suppress broad anatomy-wide darkening and keep local line-like changes.
    dark_local = (dark_signal - _gaussian_blur(dark_signal, sigma=4.0)).clamp(min=0.0)

    # Inflow event density (where contrast arrived).
    inflow = events[:, 0].amax(dim=1, keepdim=True)  # first-inflow support
    event_density = events[:, :2].sum(dim=(1, 2), keepdim=False).unsqueeze(1)

    # Frangi on high-pass darkening: a pseudo-vessel needs tubular support,
    # not just a large region that changed intensity.
    fr = frangi_vesselness(
        dark_local,
        sigmas=(0.8, 1.2, 1.8, 2.4),
        dark_on_bright=False,
    )
    static_edges = _spatial_gradient_mag(mean_img)

    # Normalize each cue within ROI (avoids global outliers like ECG bar).
    def _norm_in_roi(x: torch.Tensor) -> torch.Tensor:
        xr = x * target_roi
        m = xr.view(b, -1).amax(dim=1).view(b, 1, 1, 1).clamp_min(1e-6)
        return (xr / m).clamp(0, 1)

    fr_n = _norm_in_roi(fr)
    inflow_n = _norm_in_roi(inflow)
    event_n = _norm_in_roi(event_density)
    dark_n = _norm_in_roi(dark_local)
    temporal_n = _norm_in_roi(temporal_std)
    static_n = _norm_in_roi(static_edges)
    dynamic_gate = (temporal_n / (temporal_n + static_n + 0.15)).clamp(0, 1)

    # Multiplicative fusion is deliberate: spine/ribs can have darkening or
    # event activity, but they should not become pseudo-vessels unless they are
    # also locally tubular and dynamically supported.
    dynamic_support = (0.50 * inflow_n + 0.30 * event_n + 0.20 * dark_n).clamp(0, 1)
    combined = fr_n * dynamic_support * (0.35 + 0.65 * dynamic_gate) * target_roi
    combined = _gaussian_blur(combined, sigma=0.7) * target_roi
    combined = _norm_in_roi(combined)

    q_thr = _masked_quantile(combined, target_roi, positive_quantile, fallback=threshold)
    hard_thr = torch.maximum(q_thr, torch.full_like(q_thr, threshold))
    hard = (combined > hard_thr).float() * target_roi

    # Bolus arrival time: argmax over T of inflow polarity (events[:, 0]).
    inflow_t = events[:, 0]  # (B, T-1, H, W)
    bat_idx = inflow_t.argmax(dim=1).float()  # (B, H, W)
    bat = (bat_idx / max(1, inflow_t.shape[1] - 1)).unsqueeze(1) * hard

    # Peak amplitude: max over T of summed polarity magnitude per pixel.
    amp = events.abs().sum(dim=1).amax(dim=1, keepdim=True)  # (B, 1, H, W)
    amp = _norm_in_roi(amp) * hard

    return {
        "soft_mask": combined,
        "hard_mask": hard,
        "bat": bat,
        "amp": amp,
        "target_roi": target_roi,
    }
