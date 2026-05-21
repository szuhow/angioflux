"""Convolutional AdLIF block and Phase-Locked Subthreshold Router (PLSR).

`SpikingConvBlock` wraps a Conv2d + AdLIFCell unrolled over T timesteps,
returning aggregated spike volume + membrane potential + ISI map (which the
Membrane-Carry skip connection consumes).

`PLSR` splits the spike stream into two routes:
    * structural — tonic spikes (long ISI)
    * hemodynamic — bursting spikes (short ISI)
The gate is computed from the membrane potential so it stays differentiable
without leaning on the spike surrogate.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..neurons import AdLIFCell


class SpikingConvBlock(nn.Module):
    """Conv2d + AdLIF, unrolled over T frames of an event voxel stack.

    Input:  (B, C_in, T, H, W)
    Output: dict with
        spikes : (B, C_out, T, H, W)
        spike_rate : (B, C_out, H, W)
        mem    : (B, C_out, H, W)    -- final membrane potential
        isi    : (B, C_out, H, W)    -- inter-spike interval estimate
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        beta: float = 0.9,
        rho: float = 0.95,
        a: float = 0.05,
        b: float = 0.1,
        v_th: float = 1.0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.cell = AdLIFCell(beta=beta, rho=rho, a=a, b=b, v_th=v_th)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b, _, t, h, w = x.shape
        device, dtype = x.device, x.dtype
        state = self.cell.init_state(
            torch.Size((b, self.out_channels, h, w)), device=device, dtype=dtype
        )
        spikes_t: list[torch.Tensor] = []
        for ti in range(t):
            I = self.conv(x[:, :, ti])
            sp, _, state = self.cell(I, state, t_index=ti)
            spikes_t.append(sp)
        spikes = torch.stack(spikes_t, dim=2)  # (B,C,T,H,W)
        spike_rate = spikes.mean(dim=2)
        # ISI may be zero where the neuron never spiked twice; use spike_rate fallback.
        isi = state.isi.clone()
        isi[isi <= 0] = float(t)  # large ISI → low frequency
        return {
            "spikes": spikes,
            "spike_rate": spike_rate,
            "mem": state.V,
            "isi": isi,
        }


class PLSR(nn.Module):
    """Phase-Locked Subthreshold Router (EKG-free variant).

    Gate computed from membrane potential statistics (mean of |V|) and ISI;
    differentiable end-to-end without spike surrogate.
    """

    def __init__(self, channels: int, hidden: int = 16) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, mem: torch.Tensor, isi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (g_structural, g_hemodynamic), each (B, C, H, W) in [0, 1]."""
        # Normalize ISI to roughly [0, 1].
        isi_n = isi / (isi.amax(dim=(2, 3), keepdim=True) + 1e-6)
        g_struct = self.gate(torch.cat([mem, isi_n], dim=1))
        g_hemo = 1.0 - g_struct
        return g_struct, g_hemo
