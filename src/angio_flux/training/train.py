"""Smoke training loop for Angio-FLUX on synthetic data.

Usage:
    python -m angio_flux.training.train --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml

from ..data import make_batch
from ..losses import AngioFluxLoss
from ..model import AngioFlux


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(cfg_path: str) -> None:
    cfg = load_cfg(cfg_path)
    torch.manual_seed(cfg["seed"])
    device = torch.device(cfg["train"]["device"])

    model = AngioFlux(cfg).to(device)
    loss_fn = AngioFluxLoss(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    steps_per_epoch = cfg["train"]["steps_per_epoch"]
    h, w, t = cfg["data"]["height"], cfg["data"]["width"], cfg["data"]["num_frames"]
    bs = cfg["data"]["batch_size"]

    print(f"[angio-flux] model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[angio-flux] device={device}  H={h} W={w} T={t} B={bs}")

    model.train()
    for epoch in range(cfg["train"]["epochs"]):
        t_ep = time.time()
        for step in range(steps_per_epoch):
            batch = make_batch(bs, h, w, t, seed=epoch * 1000 + step)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["video"])
            loss, parts = loss_fn(out, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % max(1, steps_per_epoch // 4) == 0:
                msg = " ".join(f"{k}={v.item():.4f}" for k, v in parts.items())
                print(f"  epoch {epoch} step {step}: {msg}")
        print(f"[epoch {epoch}] {time.time() - t_ep:.1f}s")

    print("[angio-flux] smoke training done.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path("configs") / "default.yaml"))
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
