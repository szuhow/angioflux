"""Self-supervised pretraining loop on real angiography sequences."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from ..data import AngioSequenceDataset, collate_pad
from ..encoding import append_contrast_channels, contrast_flow_prior_map, contrast_prior_map
from ..losses.ssl import (
    DenseRegressionHead,
    FrameReconstructor,
    SSLPretrainLoss,
    build_targets,
)
from ..model import AngioFlux
from ..preprocess import rpca_sparse_component, rpca_vessel_enhanced_video


class AngioFluxSSL(nn.Module):
    """Pretraining wrapper: HEE + S-UNet + recon + BAT + amplitude heads."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        base = AngioFlux(cfg)
        self.hee = base.hee
        self.sunet = base.sunet
        feat_dim = cfg["sunet"]["base_channels"]
        self.recon_head = FrameReconstructor(in_channels=feat_dim)
        self.bat_head = DenseRegressionHead(in_channels=feat_dim)
        self.amp_head = DenseRegressionHead(in_channels=feat_dim)

    def _event_video(
        self,
        video: torch.Tensor,
        rpca_sparse: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hee_cfg = self.cfg.get("hee", {})
        if hee_cfg.get("event_input", "raw") != "rpca":
            return video, None
        pseudo_cfg = self.cfg.get("pseudo_gt", {})
        enhanced, dark_sparse = rpca_vessel_enhanced_video(
            video,
            rpca_sparse=rpca_sparse,
            lam=pseudo_cfg.get("rpca_lam"),
            max_iter=pseudo_cfg.get("rpca_max_iter", 20),
            tol=pseudo_cfg.get("rpca_tol", 1.0e-5),
            quantile=hee_cfg.get("rpca_event_quantile", 0.995),
            floor_quantile=hee_cfg.get("rpca_event_floor_quantile", 0.0),
            blend=hee_cfg.get("rpca_event_blend", 1.0),
        )
        return enhanced, dark_sparse

    def forward(
        self,
        video: torch.Tensor,
        rpca_sparse: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        event_video, rpca_dark = self._event_video(video, rpca_sparse=rpca_sparse)
        events = self.hee(event_video)
        seg_input = append_contrast_channels(
            events,
            rpca_dark,
            self.cfg.get("hee", {}).get("contrast_channels", []),
        )
        seg = self.sunet(seg_input)
        feats = seg["features"]
        bat_raw = self.bat_head(feats)
        amp_raw = self.amp_head(feats)
        contrast_gate = contrast_prior_map(
            rpca_dark,
            gamma=self.cfg.get("hee", {}).get("contrast_gate_gamma", 1.0),
        )
        mask_raw = seg["mask"]
        flow_gate = contrast_flow_prior_map(
            rpca_dark,
            events,
            gamma=self.cfg.get("hee", {}).get("contrast_flow_gamma", 1.0),
            min_flow_weight=self.cfg.get("hee", {}).get("contrast_flow_min_weight", 0.10),
        )
        mask_gate = flow_gate if flow_gate is not None else contrast_gate
        vessel_gate = mask_raw if mask_gate is None else mask_raw * mask_gate
        return {
            "events": events,
            "seg_events": seg_input,
            "event_video": event_video,
            "rpca_dark": rpca_dark,
            "contrast_gate": contrast_gate,
            "flow_gate": flow_gate,
            "mask_logits": seg["mask_logits"],
            "mask_raw": mask_raw,
            "mask": vessel_gate,
            "spike_rates": seg["spike_rates"],
            "features": feats,
            "recon": self.recon_head(feats),
            "bat_raw": bat_raw,
            "amp_raw": amp_raw,
            "bat": bat_raw * vessel_gate,
            "amp": amp_raw * vessel_gate,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg.get("seed", 0))
    device = torch.device(cfg["train"].get("device", "cpu"))

    ds_cfg = cfg["data"]
    dataset = AngioSequenceDataset(
        root=ds_cfg["root"],
        target_size=tuple(ds_cfg.get("target_size", [256, 256])),
        max_frames=ds_cfg.get("max_frames", 12),
        min_frames=ds_cfg.get("min_frames", 6),
    )
    if args.limit is not None:
        dataset.studies = dataset.studies[: args.limit]
    print(f"[data] {len(dataset)} studies under {ds_cfg['root']}")

    loader = DataLoader(
        dataset,
        batch_size=cfg["train"].get("batch_size", 1),
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 0),
        collate_fn=collate_pad,
    )

    model = AngioFluxSSL(cfg).to(device)
    loss_fn = SSLPretrainLoss(
        lambda_recon=cfg["ssl"]["lambda_recon"],
        lambda_mask=cfg["ssl"]["lambda_mask"],
        lambda_bat=cfg["ssl"].get("lambda_bat", 0.5),
        lambda_amp=cfg["ssl"].get("lambda_amp", 0.5),
        lambda_consist=cfg["ssl"]["lambda_consist"],
        lambda_spike=cfg["ssl"]["lambda_spike"],
        lambda_flow_prior=cfg["ssl"].get("lambda_flow_prior", 0.0),
        spike_target_rate=cfg["loss"]["spike_target_rate"],
        pos_weight=cfg["ssl"].get("pos_weight", 5.0),
        mask_bg_weight=cfg["ssl"].get("mask_bg_weight", 1.0),
        regression_bg_weight=cfg["ssl"].get("regression_bg_weight", 0.15),
    )

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params: {n_params/1e6:.3f}M")

    ckpt_dir = Path(cfg["train"].get("ckpt_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["train"]["epochs"]):
        running = 0.0
        n = 0
        t0 = time.time()
        for batch in loader:
            video = batch["video"].to(device)
            optim.zero_grad()
            pseudo_cfg = cfg.get("pseudo_gt", {})
            use_shared_rpca = bool(
                pseudo_cfg.get("use_rpca", False)
                or cfg.get("hee", {}).get("event_input") == "rpca"
            )
            rpca_sparse = None
            if use_shared_rpca:
                rpca_sparse = rpca_sparse_component(
                    video,
                    lam=pseudo_cfg.get("rpca_lam"),
                    max_iter=pseudo_cfg.get("rpca_max_iter", 20),
                    tol=pseudo_cfg.get("rpca_tol", 1.0e-5),
                )
            out = model(video, rpca_sparse=rpca_sparse)
            targets = build_targets(
                video,
                out["events"],
                pseudo_gt_cfg=cfg.get("pseudo_gt"),
                rpca_sparse=rpca_sparse,
            )
            loss, parts = loss_fn(out, video, targets=targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running += float(loss.detach())
            n += 1
            if n % 5 == 0:
                msg = " ".join(f"{k}={v.item():.4f}" for k, v in parts.items())
                print(f"  step {n} {msg}")
        dt = time.time() - t0
        print(f"[epoch {epoch}] mean_loss={running / max(1, n):.4f}  time={dt:.1f}s  steps={n}")
        torch.save(
            {"model": model.state_dict(), "cfg": cfg, "epoch": epoch},
            ckpt_dir / f"ssl_epoch{epoch:02d}.pt",
        )


if __name__ == "__main__":
    main()
