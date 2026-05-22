"""Narrow JEPA predictor: latent-space target-token predictor."""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import TransformerBlock
from .tubelet import SinusoidalPosEmbed3D


class JEPAPredictor(nn.Module):
    """Predicts target-token embeddings from context tokens + target positions.

    Architecture:
      - linear from encoder_dim -> predictor_dim (narrow)
      - learnable [MASK] token added at target positions with sinusoidal pos emb
      - several transformer blocks; outputs at target positions are read out
      - linear projects predictor_dim back to encoder_dim
    """

    def __init__(
        self,
        encoder_dim: int = 384,
        predictor_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(encoder_dim, predictor_dim)
        self.out_proj = nn.Linear(predictor_dim, encoder_dim)
        self.pos_embed = SinusoidalPosEmbed3D(predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(predictor_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(predictor_dim)
        self.predictor_dim = predictor_dim

    def forward(
        self,
        ctx_tokens: torch.Tensor,
        ctx_positions: torch.Tensor,
        tgt_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            ctx_tokens: (B, Mc, D_enc)
            ctx_positions: (B, Mc, 3) or (Mc, 3)
            tgt_positions: (B, Mt, 3) or (Mt, 3)
        Returns:
            predicted target embeddings: (B, Mt, D_enc)
        """
        b = ctx_tokens.shape[0]
        ctx = self.in_proj(ctx_tokens)  # (B, Mc, D_pred)

        if ctx_positions.dim() == 2:
            ctx_pe = self.pos_embed(ctx_positions).unsqueeze(0).expand(b, -1, -1)
        else:
            mc = ctx_positions.shape[1]
            ctx_pe = self.pos_embed(ctx_positions.reshape(b * mc, 3)).reshape(b, mc, -1)
        ctx = ctx + ctx_pe

        if tgt_positions.dim() == 2:
            tgt_pe = self.pos_embed(tgt_positions).unsqueeze(0).expand(b, -1, -1)
        else:
            mt = tgt_positions.shape[1]
            tgt_pe = self.pos_embed(tgt_positions.reshape(b * mt, 3)).reshape(b, mt, -1)
        mt = tgt_pe.shape[1]
        tgt = self.mask_token.expand(b, mt, -1) + tgt_pe

        mc = ctx.shape[1]
        x = torch.cat([ctx, tgt], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        pred = x[:, mc:, :]  # (B, Mt, D_pred)
        return self.out_proj(pred)
