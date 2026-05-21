"""Differentiable virtual Quantitative Flow Ratio (vQFR) head.

Given graph embedding + per-node Venturi scores and stenosis logits, predicts
a single vQFR ∈ (0, 1] per case. The closed-form Bernoulli/Poiseuille
expression from the design doc is wrapped in a small MLP residual to remain
trainable without ground-truth pressure data.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VQFRHead(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.delta = nn.Sequential(
            nn.Linear(embed_dim + 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        graph_embed: torch.Tensor,
        venturi: torch.Tensor,
        stenosis_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Returns (B,) vQFR in (0, 1]."""
        sten_prob = stenosis_logits.softmax(-1)
        # severity weight: bin index * prob
        bins = torch.arange(stenosis_logits.shape[-1], device=stenosis_logits.device).float()
        sev = (sten_prob * bins).sum(-1)  # (B,N)
        sev = sev / bins[-1].clamp_min(1.0)
        sev_mean = sev.mean(-1, keepdim=True)
        vent_max = venturi.amax(-1, keepdim=True)
        # Base physics-inspired estimate: no stenosis / no Venturi starts near
        # 1.0, then decays toward severe flow loss as either term rises.
        pressure_drop = 1.4 * sev_mean + 0.8 * vent_max.clamp(min=0)
        base = 1.0 - 0.95 * (1.0 - torch.exp(-pressure_drop))
        # learnable residual
        delta = torch.tanh(self.delta(torch.cat([graph_embed, sev_mean, vent_max], dim=-1)))
        qfr = (base + 0.1 * delta).squeeze(-1).clamp(0.05, 1.0)
        return qfr
