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


def _scale_for_resolution(height: int, width: int, reference_size: float) -> float:
    return max(0.25, min(height, width) / max(1.0, float(reference_size)))


def _scaled_px(px: int, height: int, width: int, reference_size: float) -> int:
    if px <= 0:
        return 0
    return max(1, int(round(px * _scale_for_resolution(height, width, reference_size))))


def _scaled_sigmas(
    sigmas: tuple[float, ...],
    height: int,
    width: int,
    reference_size: float,
    min_sigma: float = 0.6,
) -> tuple[float, ...]:
    scale = _scale_for_resolution(height, width, reference_size)
    return tuple(max(min_sigma, float(sigma) * scale) for sigma in sigmas)


def _window_mean_at_indices(sequence: torch.Tensor, indices: torch.Tensor, window: int) -> torch.Tensor:
    """Mean frame around one temporal index per batch item."""
    window = max(1, int(window))
    left = window // 2
    right = window - left
    frame_count = sequence.shape[2]
    frames = []
    for sample_idx in range(sequence.shape[0]):
        center = int(indices[sample_idx].item())
        start = max(0, center - left)
        end = min(frame_count, center + right)
        frames.append(sequence[sample_idx : sample_idx + 1, :, start:end].mean(dim=2))
    return torch.cat(frames, dim=0)


def _soft_threshold_tensor(x: torch.Tensor, tau: torch.Tensor | float) -> torch.Tensor:
    return torch.sign(x) * (x.abs() - tau).clamp_min(0)


def _svd_shrink_tensor(matrix: torch.Tensor, tau: torch.Tensor | float) -> torch.Tensor:
    left_vectors, singular_values, right_vectors_t = torch.linalg.svd(matrix, full_matrices=False)
    shrunk = (singular_values - tau).clamp_min(0)
    return (left_vectors * shrunk.view(1, -1)) @ right_vectors_t


@torch.no_grad()
def rpca_sparse_component(
    video: torch.Tensor,
    lam: float | None = None,
    max_iter: int = 20,
    tol: float = 1.0e-5,
) -> torch.Tensor:
    """Signed RPCA sparse component for (B, 1, T, H, W) angiography clips.

    Negative sparse values usually correspond to transient dark contrast inside
    vessels; the low-rank component absorbs mostly stationary anatomy.
    """
    assert video.dim() == 5 and video.shape[1] == 1, "expected (B, 1, T, H, W)"
    if max_iter <= 0 or video.shape[2] < 2:
        return torch.zeros_like(video)

    original_dtype = video.dtype
    batch_size, _, frame_count, height, width = video.shape
    sparse_frames = []
    for sample_idx in range(batch_size):
        matrix = video[sample_idx, 0].reshape(frame_count, height * width).float()
        lambda_sparse = float(lam) if lam is not None else 1.0 / math.sqrt(max(frame_count, height * width))

        spectral_norm = torch.linalg.svdvals(matrix).amax().clamp_min(1.0e-6)
        mu = 1.25 / spectral_norm
        mu_max = mu * 1.0e7
        rho = 1.5

        low_rank = torch.zeros_like(matrix)
        sparse = torch.zeros_like(matrix)
        dual = torch.zeros_like(matrix)
        norm_matrix = torch.linalg.norm(matrix, ord="fro").clamp_min(1.0e-6)

        for _ in range(max_iter):
            low_rank = _svd_shrink_tensor(matrix - sparse + dual / mu, 1.0 / mu)
            sparse = _soft_threshold_tensor(matrix - low_rank + dual / mu, lambda_sparse / mu)
            residual = matrix - low_rank - sparse
            dual = dual + mu * residual
            mu = torch.minimum(mu * rho, mu_max)
            err = torch.linalg.norm(residual, ord="fro") / norm_matrix
            if float(err) < tol:
                break

        sparse_frames.append(sparse.reshape(1, 1, frame_count, height, width).to(original_dtype))
    return torch.cat(sparse_frames, dim=0)


@torch.no_grad()
def bolus_peak_frame(
    video: torch.Tensor,
    roi: torch.Tensor | None = None,
    baseline_frames: int = 2,
    peak_window: int = 3,
    top_fraction: float = 0.02,
    highpass_sigma: float = 4.0,
    min_frame: int | None = None,
    rpca_sparse: torch.Tensor | None = None,
    use_rpca: bool = False,
    rpca_lam: float | None = None,
    rpca_max_iter: int = 20,
    rpca_tol: float = 1.0e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the frame/window with strongest localized dark bolus signal."""
    assert video.dim() == 5 and video.shape[1] == 1, "expected (B, 1, T, H, W)"
    batch_size, _, frame_count, height, width = video.shape
    baseline_count = min(max(1, int(baseline_frames)), frame_count)
    baseline = video[:, :, :baseline_count].mean(dim=2)
    dark_signal = (baseline.unsqueeze(2) - video).clamp_min(0)

    if rpca_sparse is None and use_rpca:
        rpca_sparse = rpca_sparse_component(video, lam=rpca_lam, max_iter=rpca_max_iter, tol=rpca_tol)
    if rpca_sparse is not None:
        dark_signal = torch.maximum(dark_signal, (-rpca_sparse).clamp_min(0))

    flat_dark = dark_signal.reshape(batch_size * frame_count, 1, height, width)
    local_dark = (flat_dark - _gaussian_blur(flat_dark, sigma=max(0.3, highpass_sigma))).clamp_min(0)
    local_dark = local_dark.reshape(batch_size, 1, frame_count, height, width)

    if roi is None:
        score_roi = torch.ones(batch_size, 1, height, width, device=video.device, dtype=video.dtype)
    else:
        score_roi = roi.to(device=video.device, dtype=video.dtype)

    top_fraction = min(1.0, max(1.0e-5, float(top_fraction)))
    blocked_frames = baseline_count if min_frame is None else max(0, int(min_frame))
    blocked_frames = min(blocked_frames, max(0, frame_count - 1))

    peak_indices = []
    for sample_idx in range(batch_size):
        mask_flat = (score_roi[sample_idx, 0] > 0.5).reshape(-1)
        if not bool(mask_flat.any()):
            mask_flat = torch.ones(height * width, device=video.device, dtype=torch.bool)
        values = local_dark[sample_idx, 0].reshape(frame_count, height * width)[:, mask_flat]
        top_count = max(1, min(values.shape[1], int(round(values.shape[1] * top_fraction))))
        frame_scores = values.topk(top_count, dim=1).values.mean(dim=1)

        if frame_count >= 3:
            kernel = torch.tensor([0.25, 0.5, 0.25], device=video.device, dtype=frame_scores.dtype).view(1, 1, 3)
            padded = F.pad(frame_scores.view(1, 1, -1), (1, 1), mode="replicate")
            frame_scores = F.conv1d(padded, kernel).view(-1)
        if blocked_frames > 0:
            frame_scores[:blocked_frames] = torch.finfo(frame_scores.dtype).min
        peak_indices.append(frame_scores.argmax())

    peak_idx = torch.stack(peak_indices).long()
    peak = _window_mean_at_indices(video, peak_idx, peak_window)
    return peak, peak_idx


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
    scale_reference_size: float = 192.0,
    frangi_sigmas: tuple[float, ...] | None = None,
    highpass_sigma: float = 4.0,
    fusion_blur_sigma: float = 0.7,
    peak_window: int = 3,
    peak_top_fraction: float = 0.02,
    use_rpca: bool = False,
    rpca_lam: float | None = None,
    rpca_max_iter: int = 20,
    rpca_tol: float = 1.0e-5,
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
    _, _, _, height, width = video.shape
    scale = _scale_for_resolution(height, width, scale_reference_size)
    scaled_edge_margin = _scaled_px(edge_margin_px, height, width, scale_reference_size)
    scaled_highpass_sigma = max(0.3, highpass_sigma * scale)
    scaled_fusion_blur_sigma = max(0.3, fusion_blur_sigma * scale)
    vessel_sigmas = _scaled_sigmas(
        frangi_sigmas or (0.8, 1.2, 1.8, 2.4),
        height,
        width,
        scale_reference_size,
    )

    baseline = video[:, :, :2].mean(dim=2)           # (B, 1, H, W) pre-contrast
    mean_img = video.mean(dim=2)
    temporal_std = video.std(dim=2)

    target_roi = _erode_mask(roi, scaled_edge_margin)

    sparse = None
    if use_rpca:
        sparse = rpca_sparse_component(video, lam=rpca_lam, max_iter=rpca_max_iter, tol=rpca_tol)
    peak, peak_idx = bolus_peak_frame(
        video,
        roi=target_roi,
        peak_window=peak_window,
        top_fraction=peak_top_fraction,
        highpass_sigma=scaled_highpass_sigma,
        rpca_sparse=sparse,
    )

    # Differential signal: vessels darken when contrast arrives.
    dark_signal = (baseline - peak).clamp(min=0.0)   # > 0 where pixel got darker
    if sparse is not None:
        sparse_peak = _window_mean_at_indices((-sparse).clamp_min(0), peak_idx, peak_window)
        dark_signal = torch.maximum(dark_signal, sparse_peak)
    # Suppress broad anatomy-wide darkening and keep local line-like changes.
    dark_local = (dark_signal - _gaussian_blur(dark_signal, sigma=scaled_highpass_sigma)).clamp(min=0.0)

    # Inflow event density (where contrast arrived).
    inflow = events[:, 0].amax(dim=1, keepdim=True)  # first-inflow support
    event_density = events[:, :2].sum(dim=(1, 2), keepdim=False).unsqueeze(1)

    # Frangi on high-pass darkening: a pseudo-vessel needs tubular support,
    # not just a large region that changed intensity.
    fr = frangi_vesselness(
        dark_local,
        sigmas=vessel_sigmas,
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
    combined = _gaussian_blur(combined, sigma=scaled_fusion_blur_sigma) * target_roi
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
        "peak_frame": peak,
        "peak_idx": peak_idx,
    }
