"""Tubelet patchification and grid positions for V-JEPA.

Video shape convention: (B, C=1, T, H, W).
Tubelet = (T_patch, P, P) non-overlapping cuboid.

We use a single Conv3d projection as the patch embed (standard ViT trick).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TubeletPatchifier(nn.Module):
    """Linear projection of (T_patch x P x P) tubelets to D-dim tokens."""

    def __init__(
        self,
        in_channels: int = 1,
        patch_size: int = 16,
        tubelet_t: int = 8,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_t = tubelet_t
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(tubelet_t, patch_size, patch_size),
            stride=(tubelet_t, patch_size, patch_size),
        )

    def forward(self, video: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """
        Args:
            video: (B, C, T, H, W)
        Returns:
            tokens: (B, N, D) where N = Gt * Gh * Gw
            grid: (Gt, Gh, Gw)
        """
        assert video.dim() == 5, "expected (B, C, T, H, W)"
        # Pad spatial/temporal so dims are exact multiples (edge replication).
        b, c, t, h, w = video.shape
        pad_t = (self.tubelet_t - t % self.tubelet_t) % self.tubelet_t
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_t or pad_h or pad_w:
            video = torch.nn.functional.pad(
                video, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate"
            )
        x = self.proj(video)  # (B, D, Gt, Gh, Gw)
        gt, gh, gw = x.shape[-3:]
        tokens = x.flatten(2).transpose(1, 2).contiguous()  # (B, N, D)
        return tokens, (gt, gh, gw)


def build_grid_positions(grid: tuple[int, int, int], device=None) -> torch.Tensor:
    """Return integer positions (N, 3) ordered (t, y, x) matching token order."""
    gt, gh, gw = grid
    t = torch.arange(gt, device=device)
    y = torch.arange(gh, device=device)
    x = torch.arange(gw, device=device)
    tt, yy, xx = torch.meshgrid(t, y, x, indexing="ij")
    return torch.stack([tt.flatten(), yy.flatten(), xx.flatten()], dim=-1)


class SinusoidalPosEmbed3D(nn.Module):
    """Fixed 3D sinusoidal positional embeddings (t, y, x)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 6 == 0, f"dim must be divisible by 6, got {dim}"
        self.dim = dim

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (N, 3) integer or float positions (t, y, x)
        Returns:
            emb: (N, dim)
        """
        d_axis = self.dim // 3
        half = d_axis // 2
        device = positions.device
        freqs = torch.exp(
            -torch.arange(half, device=device, dtype=torch.float32)
            * (torch.log(torch.tensor(10000.0)) / max(1, half))
        )
        outs = []
        for axis in range(3):
            p = positions[:, axis].float().unsqueeze(-1)  # (N, 1)
            ang = p * freqs.unsqueeze(0)  # (N, half)
            outs.append(torch.sin(ang))
            outs.append(torch.cos(ang))
        emb = torch.cat(outs, dim=-1)  # (N, 3 * 2 * half) = (N, dim)
        return emb
