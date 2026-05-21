"""Top-level Angio-FLUX model (EKG-free)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoding import HemodynamicEventEncoder, events_to_voxel
from .modules import FluxGNN, SpikingUNetPP, VQFRHead


class AngioFlux(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.hee = HemodynamicEventEncoder(
            theta0=cfg["hee"]["theta0"],
            alpha=cfg["hee"]["alpha"],
            window=tuple(cfg["hee"]["window"]),
            drift_window=cfg["hee"].get("drift_window", 31),
            theta_min_frac=cfg["hee"].get("theta_min_frac", 0.6),
            eps=cfg["hee"]["eps"],
        )
        self.num_bins = cfg["voxel"]["num_bins"]
        in_channels = 4  # polarity channels
        self.sunet = SpikingUNetPP(
            in_channels=in_channels,
            base_channels=cfg["sunet"]["base_channels"],
            out_channels=cfg["sunet"]["out_channels"],
            adlif_kwargs={
                "beta": cfg["adlif"]["beta"],
                "rho": cfg["adlif"]["rho"],
                "a": cfg["adlif"]["a"],
                "b": cfg["adlif"]["b"],
                "v_th": cfg["adlif"]["v_th"],
            },
        )
        feat_dim = cfg["sunet"]["base_channels"]
        self.gnn = FluxGNN(
            in_channels=feat_dim,
            hidden_dim=cfg["flux_gnn"]["hidden_dim"],
            num_aha_classes=cfg["flux_gnn"]["num_classes_aha"],
            num_stenosis_bins=cfg["flux_gnn"]["num_stenosis_bins"],
            knn_k=cfg["flux_gnn"]["knn_k"],
        )
        self.vqfr_head = VQFRHead(embed_dim=cfg["flux_gnn"]["hidden_dim"])

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            video: (B, 1, T, H, W) cine sequence in [0, 1].
        Returns dict with all task outputs.
        """
        events = self.hee(video)  # (B, 4, T-1, H, W)
        # Optional: also build a voxel summary (for diagnostics / lightweight modes)
        voxel = events_to_voxel(events, num_bins=self.num_bins)

        seg = self.sunet(events)  # operates over T-1 timesteps
        hemo = seg["hemo_streams"][0]  # first-level hemodynamic stream
        gnn_out = self.gnn(seg["features"], hemo, seg["mask"].detach())
        qfr = self.vqfr_head(gnn_out["graph_embed"], gnn_out["venturi"], gnn_out["stenosis_logits"])

        return {
            "events": events,
            "voxel": voxel,
            "mask_logits": seg["mask_logits"],
            "mask": seg["mask"],
            "spike_rates": seg["spike_rates"],
            "aha_logits": gnn_out["aha_logits"],
            "stenosis_logits": gnn_out["stenosis_logits"],
            "venturi": gnn_out["venturi"],
            "vqfr": qfr,
            "node_coords": gnn_out["node_coords"],
        }
