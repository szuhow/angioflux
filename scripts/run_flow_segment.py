#!/usr/bin/env python
"""Self-supervised vessel segmentation from contrast flow (no labels).

    PYTHONPATH=src python scripts/run_flow_segment.py \\
        --config configs/pretrain.yaml --out outputs/flow_seg --num 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from angio_flux.data import AngioSequenceDataset  # noqa: E402
from angio_flux.segmentation import FlowVesselSegmenter, flow_vessel_segment  # noqa: E402


def _norm01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-8:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--full-res", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.80)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device or cfg.get("train", {}).get("device", "cpu"))
    ds_cfg = cfg["data"]
    size = (args.full_res, args.full_res) if args.full_res else tuple(ds_cfg.get("target_size", [256, 256]))
    dataset = AngioSequenceDataset(
        root=ds_cfg["root"],
        target_size=size,
        max_frames=ds_cfg.get("max_frames", 32),
        min_frames=ds_cfg.get("min_frames", 6),
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    segmenter = FlowVesselSegmenter(cfg).to(device).eval()
    summary = []
    n = min(args.num, len(dataset))

    try:
        import matplotlib.cm as cm
    except ImportError:
        cm = None

    for i in range(n):
        item = dataset[i]
        video = item["video"].unsqueeze(0).to(device)
        with torch.no_grad():
            out = segmenter(video)
        soft = out["soft_mask"][0, 0].cpu().numpy()
        hard = out["hard_mask"][0, 0].cpu().numpy()
        bg = video[0, 0].mean(dim=0).cpu().numpy()
        bg_n = _norm01(bg)
        viz = _norm01(soft)

        np.save(out_dir / f"vessel_soft_{i:03d}.npy", soft)
        np.save(out_dir / f"vessel_hard_{i:03d}.npy", hard)
        Image.fromarray((bg_n * 255).astype(np.uint8)).save(out_dir / f"bg_{i:03d}.png")
        Image.fromarray((viz * 255).astype(np.uint8)).save(out_dir / f"score_{i:03d}.png")
        Image.fromarray((hard * 255).astype(np.uint8)).save(out_dir / f"mask_{i:03d}.png")

        if cm is not None:
            bg_rgb = np.stack([bg_n, bg_n, bg_n], axis=-1)
            score_rgb = cm.get_cmap("hot")(viz)[..., :3]
            a = (viz ** 0.7) * args.alpha
            overlay = bg_rgb * (1.0 - a[..., None]) + score_rgb * a[..., None]
            Image.fromarray((np.clip(overlay, 0, 1) * 255).astype(np.uint8)).save(
                out_dir / f"overlay_{i:03d}.png"
            )

        summary.append({
            "index": i,
            "path": item["path"],
            "soft_mean": float(soft.mean()),
            "hard_pixels": int(hard.sum()),
        })
        print(f"[{i}] {Path(item['path']).name}  hard_px={int(hard.sum())}  soft_mean={soft.mean():.3f}")

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
