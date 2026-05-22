"""ViT-3D student encoder + EMA target encoder for V-JEPA."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .tubelet import SinusoidalPosEmbed3D, TubeletPatchifier, build_grid_positions


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class ViT3DEncoder(nn.Module):
    """Tubelet ViT for V-JEPA.

    Operates on a subset of tokens (selected by external mask) — positional
    embeddings are added per token according to its 3D position so the
    encoder is permutation- and subset-invariant.
    """

    def __init__(
        self,
        in_channels: int = 1,
        patch_size: int = 16,
        tubelet_t: int = 8,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.patchify = TubeletPatchifier(in_channels, patch_size, tubelet_t, embed_dim)
        self.pos_embed = SinusoidalPosEmbed3D(embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def patch_tokens(self, video: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        return self.patchify(video)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tokens: (B, M, D) — already patchified (subset of all tokens)
            positions: (M, 3) or (B, M, 3) integer 3D positions for those tokens
        Returns:
            (B, M, D)
        """
        if positions.dim() == 2:
            pe = self.pos_embed(positions)  # (M, D)
            x = tokens + pe.unsqueeze(0)
        else:
            b, m, _ = positions.shape
            pe = self.pos_embed(positions.reshape(b * m, 3)).reshape(b, m, -1)
            x = tokens + pe
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class EMATargetEncoder(nn.Module):
    """Frozen EMA copy of a student encoder. Updated via .update(student, m)."""

    def __init__(self, student: ViT3DEncoder) -> None:
        super().__init__()
        self.target = copy.deepcopy(student)
        for p in self.target.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: ViT3DEncoder, momentum: float) -> None:
        for p_t, p_s in zip(self.target.parameters(), student.parameters()):
            p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)
        for b_t, b_s in zip(self.target.buffers(), student.buffers()):
            b_t.data.copy_(b_s.data)

    @torch.no_grad()
    def forward(
        self,
        video: torch.Tensor,
        positions_filter: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        tokens, grid = self.target.patch_tokens(video)
        all_positions = build_grid_positions(grid, device=tokens.device)
        out = self.target(tokens, all_positions)
        if positions_filter is None:
            return out, grid
        # Filter to selected indices per batch element.
        # positions_filter: (B, M) long indices into N tokens.
        b, m = positions_filter.shape
        gather_idx = positions_filter.unsqueeze(-1).expand(-1, -1, out.shape[-1])
        return out.gather(1, gather_idx), grid
