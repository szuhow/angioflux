"""Top-level Angio-FLUX model (EKG-free)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoding import (
    HemodynamicEventEncoder,
    append_contrast_channels,
    contrast_channel_count,
    contrast_flow_prior_map,
    contrast_prior_map,
    events_to_voxel,
)
from .modules import FluxGNN, SpikingUNetPP, VQFRHead
from .preprocess import rpca_vessel_enhanced_video


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
        in_channels = 4 + contrast_channel_count(cfg)
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

    def forward(
        self,
        video: torch.Tensor,
        rpca_sparse: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            video: (B, 1, T, H, W) cine sequence in [0, 1].
        Returns dict with all task outputs.
        """
        event_video = video
        rpca_dark = None
        if self.cfg.get("hee", {}).get("event_input", "raw") == "rpca":
            pseudo_cfg = self.cfg.get("pseudo_gt", {})
            event_video, rpca_dark = rpca_vessel_enhanced_video(
                video,
                rpca_sparse=rpca_sparse,
                lam=pseudo_cfg.get("rpca_lam"),
                max_iter=pseudo_cfg.get("rpca_max_iter", 20),
                tol=pseudo_cfg.get("rpca_tol", 1.0e-5),
                quantile=self.cfg["hee"].get("rpca_event_quantile", 0.995),
                floor_quantile=self.cfg["hee"].get("rpca_event_floor_quantile", 0.0),
                blend=self.cfg["hee"].get("rpca_event_blend", 1.0),
            )
        events = self.hee(event_video)  # (B, 4, T-1, H, W)
        # Optional: also build a voxel summary (for diagnostics / lightweight modes)
        voxel = events_to_voxel(events, num_bins=self.num_bins)

        seg_input = append_contrast_channels(
            events,
            rpca_dark,
            self.cfg.get("hee", {}).get("contrast_channels", []),
        )
        seg = self.sunet(seg_input)  # operates over T-1 timesteps
        contrast_gate = contrast_prior_map(
            rpca_dark,
            gamma=self.cfg.get("hee", {}).get("contrast_gate_gamma", 1.0),
        )
        flow_gate = contrast_flow_prior_map(
            rpca_dark,
            events,
            gamma=self.cfg.get("hee", {}).get("contrast_flow_gamma", 1.0),
            min_flow_weight=self.cfg.get("hee", {}).get("contrast_flow_min_weight", 0.10),
        )
        mask_raw = seg["mask"]
        mask_gate = flow_gate if flow_gate is not None else contrast_gate
        mask = mask_raw if mask_gate is None else mask_raw * mask_gate
        hemo = seg["hemo_streams"][0]  # first-level hemodynamic stream
        gnn_out = self.gnn(seg["features"], hemo, mask.detach())
        qfr = self.vqfr_head(gnn_out["graph_embed"], gnn_out["venturi"], gnn_out["stenosis_logits"])

        return {
            "events": events,
            "seg_events": seg_input,
            "event_video": event_video,
            "rpca_dark": rpca_dark,
            "contrast_gate": contrast_gate,
            "flow_gate": flow_gate,
            "voxel": voxel,
            "mask_logits": seg["mask_logits"],
            "mask_raw": mask_raw,
            "mask": mask,
            "spike_rates": seg["spike_rates"],
            "aha_logits": gnn_out["aha_logits"],
            "stenosis_logits": gnn_out["stenosis_logits"],
            "venturi": gnn_out["venturi"],
            "vqfr": qfr,
            "node_coords": gnn_out["node_coords"],
        }
