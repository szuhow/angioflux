"""End-to-end Angio-FLUX pipeline: train (SSL) → infer → visualize.

Usage:
    # Full pipeline:
    PYTHONPATH=src python scripts/run_pipeline.py --config configs/pretrain.yaml \\
        --epochs 3 --limit 20 --out outputs/run01

    # Inference only:
    PYTHONPATH=src python scripts/run_pipeline.py --config configs/pretrain.yaml \\
        --ckpt outputs/run01/model.pt --no-train --out outputs/run01
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from angio_flux.data import AngioSequenceDataset, collate_pad
from angio_flux.losses.ssl import SSLPretrainLoss, build_targets
from angio_flux.training.pretrain import AngioFluxSSL


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _norm01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-8:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def iou_in_roi(mask_pred: np.ndarray, pseudo_hard: np.ndarray, roi: np.ndarray) -> float:
    """IoU between binary prediction and pseudo-GT, restricted to ROI."""
    pred = mask_pred > 0.5
    gt = pseudo_hard > 0.5
    roi_m = roi > 0.5
    inter = (pred & gt & roi_m).sum()
    union = ((pred | gt) & roi_m).sum()
    return float(inter / max(1, union))


# --------------------------------------------------------------------------- #
# training                                                                    #
# --------------------------------------------------------------------------- #
def train(cfg: dict, dataset: AngioSequenceDataset, ckpt_path: Path,
          epochs: int, device: torch.device, log_every: int = 5) -> AngioFluxSSL:
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"].get("batch_size", 1),
        shuffle=True,
        num_workers=0,
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
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 1e-5),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] params: {n_params/1e6:.3f}M  studies: {len(dataset)}  epochs: {epochs}")

    history: list[dict] = []
    for epoch in range(epochs):
        model.train()
        running = 0.0
        t0 = time.time()
        for step, batch in enumerate(loader, start=1):
            video = batch["video"].to(device)
            opt.zero_grad()
            out = model(video)
            targets = build_targets(video, out["events"])
            loss, parts = loss_fn(out, video, targets=targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.detach())
            if step % log_every == 0:
                msg = " ".join(f"{k}={v.item():.4f}" for k, v in parts.items())
                print(f"  ep{epoch} step {step:>4d} {msg}")
            history.append({k: float(v) for k, v in parts.items()} | {"epoch": epoch, "step": step})
        dt = time.time() - t0
        print(f"[epoch {epoch}] mean_loss={running / max(1, step):.4f}  time={dt:.1f}s  steps={step}")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg, "history": history}, ckpt_path)
    print(f"[train] checkpoint → {ckpt_path}")
    return model


# --------------------------------------------------------------------------- #
# inference + visualization                                                   #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def visualize_sample(model: AngioFluxSSL, batch: dict, out_path: Path,
                     device: torch.device, title: str = "") -> dict:
    model.eval()
    video = batch["video"].to(device)
    out = model(video)
    targets = build_targets(video, out["events"])

    vid = _to_np(video[0, 0])
    events = _to_np(out["events"][0])
    peak = vid[-3:].mean(0)

    roi = _to_np(targets["roi"][0, 0])
    target_roi = _to_np(targets.get("target_roi", targets["roi"])[0, 0])
    pseudo_soft = _to_np(targets["soft_mask"][0, 0])
    pseudo_hard = _to_np(targets["hard_mask"][0, 0])
    bat_gt = _to_np(targets["bat"][0, 0])
    amp_gt = _to_np(targets["amp"][0, 0])

    mask_pred_raw = _to_np(out["mask"][0, 0])
    mask_pred = mask_pred_raw * target_roi
    bat_pred = _to_np(out["bat"][0, 0]) * target_roi
    amp_pred = _to_np(out["amp"][0, 0]) * target_roi
    recon = _to_np(out["recon"][0, 0])

    inflow = events[0].sum(0)
    washout = events[1].sum(0)

    # Vessels overlay: peak frame in grayscale + thresholded mask in red.
    overlay = np.stack([peak, peak, peak], axis=-1)
    overlay = _norm01(overlay)
    m_thr = (mask_pred > 0.5).astype(np.float32)
    overlay[..., 0] = np.clip(overlay[..., 0] + 0.7 * m_thr, 0, 1)
    overlay[..., 1] = overlay[..., 1] * (1 - 0.4 * m_thr)
    overlay[..., 2] = overlay[..., 2] * (1 - 0.4 * m_thr)

    # BAT colormap overlay
    bat_overlay = plt.cm.turbo(bat_pred)[..., :3]
    bat_overlay = bat_overlay * (mask_pred > 0.3)[..., None] + (1 - (mask_pred > 0.3)[..., None]) * 0.1

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    panels = [
        (peak, "peak frame", "gray"),
        (roi, "ROI mask (collimator removed)", "gray"),
        (_norm01(inflow), "inflow events Σ", "Reds"),
        (_norm01(washout), "washout events Σ", "Blues"),
        (pseudo_soft, "pseudo-GT mask (soft)", "viridis"),
        (pseudo_hard, "pseudo-GT mask (hard)", "viridis"),
        (mask_pred, "predicted mask", "viridis"),
        (overlay, "vessel overlay (pred)", None),
        (bat_gt, "BAT pseudo-GT", "turbo"),
        (bat_pred, "BAT predicted", "turbo"),
        (amp_gt, "amplitude pseudo-GT", "magma"),
        (amp_pred, "amplitude predicted", "magma"),
    ]
    for ax, (img, name, cmap) in zip(axes.flat, panels):
        if cmap is None:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    return {
        "recon_mae_roi": float(np.mean(np.abs(recon - peak) * roi) / max(1e-6, roi.mean())),
        "pred_mean_in_roi": float((mask_pred * target_roi).sum() / max(1.0, target_roi.sum())),
        "pred_mean_outside_roi": float((mask_pred_raw * (1.0 - target_roi)).sum() / max(1.0, (1.0 - target_roi).sum())),
        "pseudo_mean_in_roi": float((pseudo_hard * target_roi).sum() / max(1.0, target_roi.sum())),
        "iou_vs_pseudo": iou_in_roi(mask_pred, pseudo_hard, target_roi),
        "roi_coverage": float(roi.mean()),
        "target_roi_coverage": float(target_roi.mean()),
        "path": batch["path"][0],
    }


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", default="outputs/run01")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num-viz", type=int, default=5)
    p.add_argument("--no-train", action="store_true")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["train"].get("device", "cpu"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_cfg = cfg["data"]
    dataset = AngioSequenceDataset(
        root=ds_cfg["root"],
        target_size=tuple(ds_cfg.get("target_size", [192, 192])),
        max_frames=ds_cfg.get("max_frames", 12),
        min_frames=ds_cfg.get("min_frames", 6),
    )
    if args.limit is not None:
        dataset.studies = dataset.studies[: args.limit]
    print(f"[data] {len(dataset)} studies from {ds_cfg['root']}")

    ckpt_path = out_dir / "model.pt" if args.ckpt is None else Path(args.ckpt)

    if args.no_train:
        if not ckpt_path.exists():
            raise SystemExit(f"--no-train but no checkpoint at {ckpt_path}")
        model = AngioFluxSSL(cfg).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        print(f"[train] loaded {ckpt_path}")
    else:
        epochs = args.epochs if args.epochs is not None else cfg["train"]["epochs"]
        model = train(cfg, dataset, ckpt_path, epochs=epochs, device=device)

    viz_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_pad)
    metrics: list[dict] = []
    for i, batch in enumerate(viz_loader):
        if i >= args.num_viz:
            break
        study_name = Path(batch["path"][0]).name
        meta = batch["meta"][0]
        title = (f"{study_name}  |  {meta['projection_label']}  |  "
                 f"ft={meta['frame_time_ms']:.1f}ms  |  "
                 f"px={meta['pixel_spacing_mm']:.3f}mm")
        viz_path = out_dir / "viz" / f"{i:02d}_{study_name}.png"
        m = visualize_sample(model, batch, viz_path, device, title=title)
        metrics.append(m)
        print(f"[viz] {i:02d} {study_name}  IoU={m['iou_vs_pseudo']:.3f}  "
              f"ROI={m['roi_coverage']:.2f}  recon_MAE={m['recon_mae_roi']:.4f}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if metrics:
        print(f"\n[summary] over {len(metrics)} samples:")
        print(f"  mean IoU vs pseudo-GT : {np.mean([m['iou_vs_pseudo'] for m in metrics]):.3f}")
        print(f"  mean ROI coverage     : {np.mean([m['roi_coverage'] for m in metrics]):.3f}")
        print(f"  mean recon MAE (ROI)  : {np.mean([m['recon_mae_roi'] for m in metrics]):.4f}")
    print(f"[done] visualizations → {out_dir/'viz'}  metrics → {out_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
