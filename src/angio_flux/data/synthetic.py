"""Synthetic coronary-like cine generator (no real DICOM dependency).

Procedurally renders a branching tree of vessels and animates a contrast
bolus propagating along the branches. Used for smoke tests and ablation
prototyping before any real-data integration.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch


@dataclass
class SyntheticSample:
    video: torch.Tensor       # (1, T, H, W)
    mask: torch.Tensor        # (1, H, W) vessel ground truth
    aha: torch.Tensor         # (N,) per-node labels (filled at collate time)
    stenosis: torch.Tensor    # (N,) per-node bin labels
    vqfr: torch.Tensor        # scalar


def _draw_branch(
    canvas: torch.Tensor,
    x0: float,
    y0: float,
    angle: float,
    length: float,
    width: float,
    depth: int,
) -> list[tuple[float, float]]:
    """Recursively rasterize a vessel branch onto a 2D mask; returns centerline."""
    h, w = canvas.shape
    pts: list[tuple[float, float]] = []
    steps = int(length)
    for i in range(steps):
        x = x0 + i * math.cos(angle)
        y = y0 + i * math.sin(angle)
        if not (0 <= x < w and 0 <= y < h):
            break
        r = max(1, int(width))
        xi, yi = int(x), int(y)
        x_lo, x_hi = max(0, xi - r), min(w, xi + r + 1)
        y_lo, y_hi = max(0, yi - r), min(h, yi + r + 1)
        canvas[y_lo:y_hi, x_lo:x_hi] = 1.0
        pts.append((x, y))
    if depth > 0 and pts:
        ex, ey = pts[-1]
        for branch in range(2):
            new_angle = angle + (0.4 if branch == 0 else -0.5) + random.uniform(-0.2, 0.2)
            new_len = length * random.uniform(0.5, 0.7)
            new_w = max(1.0, width * 0.7)
            pts += _draw_branch(canvas, ex, ey, new_angle, new_len, new_w, depth - 1)
    return pts


def make_sample(
    height: int = 128,
    width: int = 128,
    num_frames: int = 32,
    seed: int | None = None,
) -> SyntheticSample:
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    mask = torch.zeros(height, width)
    cx, cy = width * 0.2, height * 0.4
    centerline = _draw_branch(mask, cx, cy, angle=0.6, length=60, width=2.5, depth=3)

    # Build video: static tissue background + travelling bolus that darkens
    # pixels along the centerline as a function of (time, arclength).
    bg = 0.55 + 0.05 * torch.randn(height, width)
    # Faint static "ribs" — sinusoidal stripes (the kind of clutter HEE should kill).
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    bg = bg + 0.06 * torch.sin(0.4 * xx.float() + 1.7)
    bg = bg.clamp(0.05, 0.95)

    # Pre-compute arclength along centerline.
    arclen = []
    s = 0.0
    last = centerline[0] if centerline else (0.0, 0.0)
    for p in centerline:
        s += math.hypot(p[0] - last[0], p[1] - last[1])
        arclen.append(s)
        last = p
    s_max = max(arclen[-1], 1.0) if arclen else 1.0

    video = torch.zeros(1, num_frames, height, width)
    bolus_speed = s_max / (num_frames * 0.55)
    sigma = max(2.0, s_max / 12.0)
    for t in range(num_frames):
        frame = bg.clone()
        bolus_pos = bolus_speed * t
        for (px, py), s_pt in zip(centerline, arclen):
            # Bolus intensity profile (Gaussian travelling pulse).
            atten = math.exp(-((s_pt - bolus_pos) ** 2) / (2 * sigma * sigma))
            xi, yi = int(px), int(py)
            # darken in a small disk
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    xx2, yy2 = xi + dx, yi + dy
                    if 0 <= xx2 < width and 0 <= yy2 < height:
                        frame[yy2, xx2] = frame[yy2, xx2] * (1.0 - 0.7 * atten)
        # add small respiratory drift (uniform brightness shift)
        frame = (frame + 0.02 * math.sin(0.5 * t)).clamp(0.02, 0.99)
        video[0, t] = frame

    return SyntheticSample(
        video=video,
        mask=mask.unsqueeze(0),
        aha=torch.zeros(0, dtype=torch.long),
        stenosis=torch.zeros(0, dtype=torch.long),
        vqfr=torch.tensor(0.85),
    )


def make_batch(batch_size: int, height: int, width: int, num_frames: int, seed: int = 0) -> dict:
    samples = [make_sample(height, width, num_frames, seed=seed + i) for i in range(batch_size)]
    videos = torch.stack([s.video for s in samples]).unsqueeze(1).squeeze(2)
    # Above gives (B,1,T,H,W) -- recompute cleanly:
    videos = torch.stack([s.video for s in samples])  # (B,1,T,H,W)
    masks = torch.stack([s.mask for s in samples])    # (B,1,H,W)
    vqfr = torch.stack([s.vqfr for s in samples])     # (B,)
    return {"video": videos, "mask": masks, "vqfr": vqfr}
