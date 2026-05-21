"""Membrane-Carry skip connection.

Carries three modalities from encoder to decoder:
    1. Membrane potential V (sub-threshold structural cue)
    2. Polarity sign (inflow vs washout) via spike-rate asymmetry
    3. ISI^{-1} (local firing-frequency proxy, ~ flow velocity)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MembraneCarrySkip(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        # 3 modalities -> single fused tensor with same channel count.
        self.fuse = nn.Conv2d(channels * 3, channels, kernel_size=1)

    def forward(
        self,
        mem: torch.Tensor,
        spike_rate: torch.Tensor,
        isi: torch.Tensor,
    ) -> torch.Tensor:
        # polarity proxy: spike_rate centered around its mean per-sample.
        sr_mean = spike_rate.mean(dim=(2, 3), keepdim=True)
        polarity = torch.tanh(spike_rate - sr_mean)
        freq = 1.0 / (isi + 1e-3)
        freq = freq / (freq.amax(dim=(2, 3), keepdim=True) + 1e-6)
        merged = torch.cat([mem, polarity, freq], dim=1)
        return self.fuse(merged)
