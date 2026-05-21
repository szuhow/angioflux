"""Spiking U-Net++ for vessel segmentation from event voxel grids.

Two-level encoder/decoder; each encoder block produces spikes + V + ISI
which are consumed by both the decoder (via MembraneCarrySkip) and the
PLSR router. A lightweight non-spiking refinement head produces final logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .plsr import PLSR, SpikingConvBlock
from .skip import MembraneCarrySkip


class SpikingUNetPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        out_channels: int = 1,
        adlif_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        adlif_kwargs = adlif_kwargs or {}
        c1, c2 = base_channels, base_channels * 2

        self.enc1 = SpikingConvBlock(in_channels, c1, **adlif_kwargs)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = SpikingConvBlock(c1, c2, **adlif_kwargs)

        self.plsr1 = PLSR(c1)
        self.plsr2 = PLSR(c2)

        self.skip1 = MembraneCarrySkip(c1)
        self.skip2 = MembraneCarrySkip(c2)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec_conv = nn.Sequential(
            nn.Conv2d(c2 + c1, c1, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    @staticmethod
    def _pool_spikes(spikes: torch.Tensor) -> torch.Tensor:
        """Max-pool spike volume spatially (preserve time)."""
        b, c, t, h, w = spikes.shape
        x = spikes.reshape(b * t, c, h, w)
        x = F.max_pool2d(x, 2)
        _, _, h2, w2 = x.shape
        return x.reshape(b, c, t, h2, w2)

    def forward(self, voxel: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            voxel: (B, C_in, T, H, W) — event stream (sub-frame temporal axis kept).
        Returns dict with:
            mask_logits: (B, out, H, W)
            mask:        (B, out, H, W) sigmoid output
            features:    (B, c1, H, W)  — pre-head features (for downstream tasks)
            spike_rates: list of (B, C, H, W) per encoder level (for spike-rate loss)
            gates:       list of (g_struct, g_hemo) per level (for diagnostics)
        """
        e1 = self.enc1(voxel)
        g1s, g1h = self.plsr1(e1["mem"], e1["isi"])
        skip1 = self.skip1(e1["mem"], e1["spike_rate"], e1["isi"]) * g1s

        pooled = self._pool_spikes(e1["spikes"])
        e2 = self.enc2(pooled)
        g2s, g2h = self.plsr2(e2["mem"], e2["isi"])
        bottleneck = self.skip2(e2["mem"], e2["spike_rate"], e2["isi"]) * g2s

        up = self.up(bottleneck)
        # Align spatial size in case of odd inputs.
        if up.shape[-2:] != skip1.shape[-2:]:
            up = F.interpolate(up, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        dec = self.dec_conv(torch.cat([up, skip1], dim=1))
        logits = self.head(dec)
        mask = torch.sigmoid(logits)

        return {
            "mask_logits": logits,
            "mask": mask,
            "features": dec,
            "spike_rates": [e1["spike_rate"], e2["spike_rate"]],
            "gates": [(g1s, g1h), (g2s, g2h)],
            "hemo_streams": [e1["spike_rate"] * g1h, e2["spike_rate"] * g2h],
        }
