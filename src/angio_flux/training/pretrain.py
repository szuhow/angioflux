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
from ..losses.ssl import (
    DenseRegressionHead,
    FrameReconstructor,
    SSLPretrainLoss,
    build_targets,
)
from ..model import AngioFlux


class AngioFluxSSL(nn.Module):
    """Pretraining wrapper: HEE + S-UNet + recon + BAT + amplitude heads."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        base = AngioFlux(cfg)
        self.hee = base.hee
        self.sunet = base.sunet
        feat_dim = cfg["sunet"]["base_channels"]
        self.recon_head = FrameReconstructor(in_channels=feat_dim)
        self.bat_head = DenseRegressionHead(in_channels=feat_dim)
        self.amp_head = DenseRegressionHead(in_channels=feat_dim)

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        events = self.hee(video)
        seg = self.sunet(events)
        feats = seg["features"]
        return {
            "events": events,
            "mask_logits": seg["mask_logits"],
            "mask": seg["mask"],
            "spike_rates": seg["spike_rates"],
            "features": feats,
            "recon": self.recon_head(feats),
            "bat": self.bat_head(feats),
            "amp": self.amp_head(feats),
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
        spike_target_rate=cfg["loss"]["spike_target_rate"],
        pos_weight=cfg["ssl"].get("pos_weight", 5.0),
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
            out = model(video)
            targets = build_targets(video, out["events"], pseudo_gt_cfg=cfg.get("pseudo_gt"))
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
