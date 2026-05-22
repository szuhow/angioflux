"""V-JEPA combined module: encoder + EMA target + predictor + BAT head."""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import EMATargetEncoder, ViT3DEncoder
from .predictor import JEPAPredictor
from .tubelet import build_grid_positions


class VesselJEPA(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        patch_size: int = 16,
        tubelet_t: int = 8,
        embed_dim: int = 384,
        encoder_depth: int = 12,
        encoder_heads: int = 6,
        predictor_dim: int = 192,
        predictor_depth: int = 6,
        predictor_heads: int = 6,
        ema_momentum: float = 0.998,
    ) -> None:
        super().__init__()
        self.student = ViT3DEncoder(
            in_channels=in_channels,
            patch_size=patch_size,
            tubelet_t=tubelet_t,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
        )
        self.target = EMATargetEncoder(self.student)
        self.predictor = JEPAPredictor(
            encoder_dim=embed_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
        )
        # Tiny BAT head over student embeddings (scalar per token).
        self.bat_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(), nn.Linear(embed_dim // 4, 1),
        )
        self.ema_momentum = ema_momentum
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.tubelet_t = tubelet_t

    def patch_and_positions(
        self, video: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int]]:
        tokens, grid = self.student.patch_tokens(video)
        positions = build_grid_positions(grid, device=tokens.device)
        return tokens, positions, grid

    def encode_context(
        self,
        tokens_all: torch.Tensor,
        positions_all: torch.Tensor,
        ctx_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode full spacetime grid, then gather context token embeddings."""
        b = tokens_all.shape[0]
        d = tokens_all.shape[-1]
        valid = ctx_idx >= 0
        all_enc = self.student(tokens_all, positions_all)
        safe = ctx_idx.clamp(min=0)
        ctx_enc = all_enc.gather(1, safe.unsqueeze(-1).expand(-1, -1, d))
        pos = positions_all[safe.reshape(-1)].reshape(b, ctx_idx.shape[1], 3)
        if not valid.all():
            ctx_enc = ctx_enc * valid.unsqueeze(-1).float()
        return ctx_enc, pos

    def predict_targets(
        self,
        ctx_encoded: torch.Tensor,
        ctx_pos: torch.Tensor,
        positions_all: torch.Tensor,
        tgt_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = ctx_encoded.shape[0]
        safe_t = tgt_idx.clamp(min=0)
        tgt_pos = positions_all[safe_t.reshape(-1)].reshape(b, tgt_idx.shape[1], 3)
        pred = self.predictor(ctx_encoded, ctx_pos, tgt_pos)
        return pred, tgt_pos

    @torch.no_grad()
    def target_embeddings(
        self,
        video: torch.Tensor,
        tgt_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Run EMA encoder over the FULL token set, then gather target indices."""
        b = video.shape[0]
        tokens, grid = self.target.target.patch_tokens(video)
        positions = build_grid_positions(grid, device=tokens.device)
        all_out = self.target.target(tokens, positions)
        d = all_out.shape[-1]
        safe = tgt_idx.clamp(min=0)
        return all_out.gather(1, safe.unsqueeze(-1).expand(-1, -1, d))

    def update_ema(self) -> None:
        self.target.update(self.student, self.ema_momentum)
