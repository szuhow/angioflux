#!/usr/bin/env python
"""Entry-point script for V-JEPA pretraining and vessel-score extraction.

Usage:
    python scripts/run_jepa.py train --config configs/pretrain_jepa.yaml
    python scripts/run_jepa.py score --ckpt checkpoints/jepa/vjepa_epoch00.pt \\
        --config configs/pretrain_jepa.yaml --out outputs/jepa_vessel/

The "score" subcommand computes per-tubelet predictability + flow-prior to
produce an unsupervised pixel-level vessel score map for diagnostic viz.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from angio_flux.data import AngioSequenceDataset  # noqa: E402
from angio_flux.encoding import HemodynamicEventEncoder  # noqa: E402
from angio_flux.jepa import VesselJEPA  # noqa: E402
from angio_flux.jepa.masking import batched_masks  # noqa: E402
from angio_flux.jepa.sampling import rpca_inflow_times  # noqa: E402
from angio_flux.preprocess import rpca_sparse_component  # noqa: E402
from angio_flux.segmentation import flow_vessel_segment  # noqa: E402


def cmd_train(args: argparse.Namespace) -> None:
    from angio_flux.training.pretrain_jepa import main as train_main

    sys.argv = ["pretrain_jepa", "--config", args.config]
    if args.limit is not None:
        sys.argv += ["--limit", str(args.limit)]
    train_main()


def _resolve_ckpt(ckpt: str | None, flow_weight: float) -> Path | None:
    if flow_weight >= 1.0:
        return None
    if not ckpt or ckpt.strip() in ("...", "none", ""):
        raise SystemExit(
            "score: podaj --ckpt (ścieżka do .pt). "
            "Bez modelu użyj: python scripts/run_jepa.py score-flow --config ... --out ..."
        )
    path = Path(ckpt)
    if not path.is_file():
        raise SystemExit(f"score: brak pliku checkpoint: {path.resolve()}")
    return path


def cmd_score(args: argparse.Namespace) -> None:
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    j_cfg = cfg.get("jepa", {})
    device = torch.device(cfg["train"].get("device", "cpu"))
    flow_weight = float(getattr(args, "flow_weight", 0.85))
    ckpt_path = _resolve_ckpt(getattr(args, "ckpt", None), flow_weight)

    model = None
    if ckpt_path is not None:
        model = VesselJEPA(
            patch_size=j_cfg.get("patch_size", 16),
            tubelet_t=j_cfg.get("tubelet_t", 8),
            embed_dim=j_cfg.get("embed_dim", 384),
            encoder_depth=j_cfg.get("encoder_depth", 12),
            encoder_heads=j_cfg.get("encoder_heads", 6),
            predictor_dim=j_cfg.get("predictor_dim", 192),
            predictor_depth=j_cfg.get("predictor_depth", 6),
            predictor_heads=j_cfg.get("predictor_heads", 6),
        ).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        model.eval()

    ds = AngioSequenceDataset(
        root=cfg["data"]["root"],
        target_size=(args.full_res, args.full_res) if args.full_res
                    else tuple(cfg["data"].get("target_size", [256, 256])),
        max_frames=cfg["data"].get("max_frames", 64),
        min_frames=cfg["data"].get("min_frames", 16),
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = min(args.num, len(ds))
    n_passes = int(args.passes)
    alpha_blend = float(args.alpha)
    summary = []
    import numpy as np
    from PIL import Image
    import matplotlib.cm as cm

    flow_cfg = {
        "pseudo_gt": cfg.get("pseudo_gt", {}),
        "hee": cfg.get("hee", {}),
        "use_rpca": cfg.get("use_rpca_prior", cfg.get("use_rpca", True)),
        "coh_sigma": cfg.get("coh_sigma", 1.5),
        "coh_k": cfg.get("coh_k", 5),
        "fuse_weights": tuple(cfg.get("fuse_weights", (0.40, 0.30, 0.20, 0.10))),
        "transit_min": cfg.get("transit_min", 2),
        "transit_max": cfg.get("transit_max", 40),
    }
    for i in range(n):
        item = ds[i]
        video = item["video"].unsqueeze(0).to(device)  # (1,1,T,H,W)
        H, W = video.shape[-2], video.shape[-1]
        with torch.no_grad():
            flow_out = flow_vessel_segment(
                video,
                pseudo_gt_cfg=flow_cfg.get("pseudo_gt", {}),
                use_rpca=flow_cfg.get("use_rpca", True),
                fuse_weights=flow_cfg.get("fuse_weights", (0.40, 0.30, 0.20, 0.10)),
                coh_sigma=flow_cfg.get("coh_sigma", 1.5),
                coh_k=flow_cfg.get("coh_k", 5),
                transit_min=flow_cfg.get("transit_min", 2),
                transit_max=flow_cfg.get("transit_max", 40),
            )
            score_2d = flow_out["soft_mask"][0, 0]

            if flow_weight < 1.0 and model is not None:
                rpca_dark = rpca_sparse_component(video, max_iter=20)
                tokens_all, positions_all, grid = model.patch_and_positions(video)
                inflow_time, presence = rpca_inflow_times(rpca_dark, grid)
                gt, gh, gw = grid
                N = gt * gh * gw
                pred_sum = torch.zeros(1, N, device=device)
                pred_cnt = torch.zeros(1, N, device=device)
                for k in range(n_passes):
                    ctx_idx, tgt_idx = batched_masks(
                        1,
                        grid,
                        num_targets=max(8, N // (4 * n_passes)),
                        target_block=(1, 1, 1),
                        context_fraction=0.4,
                        seed=k + 1,
                        presence=presence,
                        inflow_time=inflow_time,
                        flow_bias=j_cfg.get("flow_target_bias", 0.7),
                    )
                    ctx_idx = ctx_idx.to(device)
                    tgt_idx = tgt_idx.to(device)
                    ctx_enc, ctx_pos = model.encode_context(tokens_all, positions_all, ctx_idx)
                    pred, _ = model.predict_targets(ctx_enc, ctx_pos, positions_all, tgt_idx)
                    target_emb = model.target_embeddings(video, tgt_idx)
                    err = (pred - target_emb).pow(2).mean(dim=-1)
                    predictability = torch.exp(-err)
                    pred_sum.scatter_add_(1, tgt_idx, predictability)
                    pred_cnt.scatter_add_(1, tgt_idx, torch.ones_like(predictability))

                pred_mean = pred_sum / pred_cnt.clamp_min(1.0)
                pmin = pred_mean[pred_cnt > 0].min() if (pred_cnt > 0).any() else pred_mean.min()
                pred_norm = ((pred_mean - pmin) / (pred_mean.max() - pmin + 1.0e-6)).clamp(0, 1)
                flow_norm = presence / (presence.amax(dim=1, keepdim=True) + 1.0e-6)
                jepa_tok = 0.5 * pred_norm + 0.5 * flow_norm
                jepa_2d = F.interpolate(
                    jepa_tok.reshape(1, 1, gt, gh, gw).max(dim=2).values,
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
                jw = 1.0 - flow_weight
                score_2d = (flow_weight * score_2d + jw * jepa_2d).clamp(0, 1)

        np_score = score_2d.cpu().numpy()
        # Per-clip min-max normalization for visualization
        vmin = float(np_score.min())
        vmax = float(np_score.max())
        viz = (np_score - vmin) / max(1.0e-6, vmax - vmin)

        # Background: temporal-mean frame (more vessel-like than first frame)
        bg = video[0, 0].mean(dim=0).cpu().numpy()
        bg = (bg - bg.min()) / max(1.0e-6, bg.max() - bg.min())
        bg_rgb = np.stack([bg, bg, bg], axis=-1)

        # Color the score with a hot colormap
        cmap = cm.get_cmap("hot")
        score_rgba = cmap(viz)[..., :3]  # (H, W, 3)
        # Modulate alpha by score so background still visible where low
        alpha = (viz ** 0.7) * alpha_blend
        alpha = alpha[..., None]
        overlay = bg_rgb * (1.0 - alpha) + score_rgba * alpha
        overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

        np.save(out_dir / f"vessel_score_{i:03d}.npy", np_score)
        Image.fromarray(overlay).save(out_dir / f"overlay_{i:03d}.png")
        Image.fromarray((bg * 255).astype(np.uint8)).save(out_dir / f"bg_{i:03d}.png")
        Image.fromarray((viz * 255).astype(np.uint8)).save(out_dir / f"score_{i:03d}.png")

        entry = {
            "index": i,
            "path": item["path"],
            "score_mean": float(np_score.mean()),
            "score_max": float(np_score.max()),
            "flow_weight": flow_weight,
        }
        if model is not None:
            entry["grid"] = list(grid)
        summary.append(entry)
        grid_msg = f"  grid={grid}" if model is not None else ""
        print(f"[{i}] {item['path']}{grid_msg}  "
              f"score range [{vmin:.3f}, {vmax:.3f}]")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def cmd_score_events(args: argparse.Namespace) -> None:
    """Event-camera vessel saliency.

    Treats the cine as an asynchronous event stream (HEE 4-channel polarities):
        p_in  : pixel darkens — contrast bolus arrives
        p_out : pixel brightens — contrast washes out
        p_LE  : leading spatial-gradient edge
        p_TE  : trailing spatial-gradient edge

    A vessel pixel emits a *causal, ordered* sequence: p_in fires once, then
    later p_out fires; neighboring vessel pixels fire p_in at consecutive
    times (the bolus front sweeps along the lumen). Noise and rigid motion
    violate one or both properties.

    Vessel score per pixel:
        score = fired_inflow
              * has_washout_after_inflow
              * exp( -|t_in - local_avg(t_in)|^2 / 2 sigma^2 )

    The arrival-time field t_in itself is the bolus phase map and is rendered
    as the overlay color (so a vessel branch shows up as a color-gradient
    streak following blood flow direction).
    """
    import numpy as np
    from PIL import Image
    import matplotlib.cm as cm

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(cfg["train"].get("device", "cpu"))

    ds = AngioSequenceDataset(
        root=cfg["data"]["root"],
        target_size=(args.full_res, args.full_res) if args.full_res
                    else tuple(cfg["data"].get("target_size", [256, 256])),
        max_frames=cfg["data"].get("max_frames", 64),
        min_frames=cfg["data"].get("min_frames", 16),
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    hee = HemodynamicEventEncoder(
        theta0=args.theta0,
        alpha=args.adapt_alpha,
        window=(args.win_h, args.win_w, args.win_t),
        drift_window=args.drift_window,
    ).to(device).eval()

    n = min(args.num, len(ds))
    summary = []
    alpha_blend = float(args.alpha)
    coh_sigma = float(args.coh_sigma)
    coh_k = int(args.coh_k)
    transit_min = int(args.transit_min)
    transit_max = int(args.transit_max)

    for i in range(n):
        item = ds[i]
        video = item["video"].unsqueeze(0).to(device)  # (1,1,T,H,W)
        T, H, W = video.shape[2], video.shape[3], video.shape[4]
        with torch.no_grad():
            events = hee(video)  # (1, 4, T-1, H, W)
            p_in = events[0, 0]    # (T-1, H, W)
            p_out = events[0, 1]
            Tm = p_in.shape[0]

            # ---- per-pixel first inflow time ----------------------------
            fired_in = p_in.amax(dim=0) > 0                  # (H, W)
            t_idx = torch.arange(Tm, device=device).view(Tm, 1, 1).float()
            # Set times where no inflow occurred to +inf so argmin picks first
            t_in_masked = torch.where(p_in > 0, t_idx,
                                       torch.full_like(t_idx, float(Tm)))
            t_in = t_in_masked.amin(dim=0)                   # (H, W) in [0, Tm]
            t_in_norm = t_in / max(1.0, float(Tm - 1))
            t_in_norm = t_in_norm.clamp(0.0, 1.0)

            # ---- washout-after-inflow ----------------------------------
            # For each pixel, find the first p_out time strictly after t_in.
            t_after = torch.where(
                (p_out > 0) & (t_idx > t_in.unsqueeze(0)),
                t_idx,
                torch.full_like(t_idx, float(Tm)),
            )
            t_out = t_after.amin(dim=0)                       # (H, W)
            transit = t_out - t_in
            washout_ok = (
                fired_in
                & (transit >= transit_min)
                & (transit <= transit_max)
            )

            # ---- local arrival-time coherence (wavefront test) ---------
            # For each fired pixel, compare t_in with the local average over
            # other fired pixels. Vessels have smooth t_in(x,y), noise does
            # not. Implement as masked avg-pool.
            fired_f = fired_in.float()
            t_safe = torch.where(fired_in, t_in, torch.zeros_like(t_in))
            pad = coh_k // 2
            num = F.avg_pool2d(
                (t_safe * fired_f).unsqueeze(0).unsqueeze(0),
                kernel_size=coh_k, stride=1, padding=pad,
                count_include_pad=False,
            ).squeeze()
            den = F.avg_pool2d(
                fired_f.unsqueeze(0).unsqueeze(0),
                kernel_size=coh_k, stride=1, padding=pad,
                count_include_pad=False,
            ).squeeze().clamp_min(1e-6)
            t_local = num / den
            resid = (t_in - t_local).abs()
            coherence = torch.exp(-(resid ** 2) / (2.0 * coh_sigma ** 2))
            coherence = torch.where(fired_in, coherence,
                                    torch.zeros_like(coherence))

            # ---- final saliency ----------------------------------------
            vessel = fired_in.float() * washout_ok.float() * coherence
            # Tiny per-image normalization
            vmin = float(vessel.min())
            vmax = float(vessel.max())
            v_norm = (vessel - vmin) / max(1.0e-6, vmax - vmin)

        # ---- background = temporal-mean frame --------------------------
        bg = video[0, 0].mean(dim=0).cpu().numpy()
        bg = (bg - bg.min()) / max(1.0e-6, bg.max() - bg.min())
        bg_rgb = np.stack([bg, bg, bg], axis=-1)

        # ---- color = arrival time (bolus phase), brightness = saliency --
        cmap = cm.get_cmap(args.cmap)
        phase = t_in_norm.cpu().numpy()
        sal = v_norm.cpu().numpy()
        color_rgba = cmap(phase)[..., :3]                # (H,W,3)
        a = (sal ** args.gamma) * alpha_blend
        a = a[..., None]
        overlay = bg_rgb * (1.0 - a) + color_rgba * a
        overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

        np.save(out_dir / f"vessel_score_{i:03d}.npy", sal)
        np.save(out_dir / f"arrival_time_{i:03d}.npy", phase)
        Image.fromarray(overlay).save(out_dir / f"overlay_{i:03d}.png")
        Image.fromarray((bg * 255).astype(np.uint8)).save(
            out_dir / f"bg_{i:03d}.png")
        Image.fromarray((sal * 255).astype(np.uint8)).save(
            out_dir / f"saliency_{i:03d}.png")
        # Visualize phase map masked by saliency for diagnostic
        phase_rgb = (color_rgba * (sal > 0.05)[..., None] * 255).astype(np.uint8)
        Image.fromarray(phase_rgb).save(out_dir / f"phase_{i:03d}.png")
        # Raw fired mask
        Image.fromarray((fired_in.cpu().numpy().astype(np.uint8) * 255)).save(
            out_dir / f"fired_{i:03d}.png")

        summary.append({
            "index": i, "path": item["path"],
            "shape": [int(H), int(W)],
            "fired_pixels": int(fired_in.sum().item()),
            "washout_ok_pixels": int(washout_ok.sum().item()),
            "vessel_score_p99": float(np.percentile(sal, 99)),
        })
        print(f"[{i}] {Path(item['path']).name}  {H}x{W}  "
              f"fired={int(fired_in.sum().item())}  "
              f"washout_ok={int(washout_ok.sum().item())}  "
              f"sal_p99={float(np.percentile(sal, 99)):.3f}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def cmd_score_rpca(args: argparse.Namespace) -> None:
    """Pure-RPCA full-resolution vessel overlay (no learned model).

    Diagnostic baseline: shows whether the RPCA sparse component alone
    localizes vessels at full resolution. Used to demonstrate that the
    signal in the data is strong even when the JEPA model collapses.
    """
    import numpy as np
    from PIL import Image
    import matplotlib.cm as cm
    from skimage.filters import frangi
    from skimage.morphology import (
        skeletonize, remove_small_objects, binary_closing,
        binary_dilation, disk,
    )
    from scipy import ndimage as ndi

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(cfg["train"].get("device", "cpu"))

    ds = AngioSequenceDataset(
        root=cfg["data"]["root"],
        target_size=(args.full_res, args.full_res) if args.full_res
                    else tuple(cfg["data"].get("target_size", [256, 256])),
        max_frames=cfg["data"].get("max_frames", 64),
        min_frames=cfg["data"].get("min_frames", 16),
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha_blend = float(args.alpha)
    mode = args.mode  # "dense" | "skeleton" | "raw"

    n = min(args.num, len(ds))
    summary = []
    for i in range(n):
        item = ds[i]
        video = item["video"].unsqueeze(0).to(device)  # (1,1,T,H,W)
        T, H, W = video.shape[2], video.shape[3], video.shape[4]
        with torch.no_grad():
            sparse = rpca_sparse_component(video, max_iter=args.rpca_iter)
            dark = (-sparse).clamp_min(0.0) if sparse.min() < 0 else sparse.clamp_min(0.0)
            dark = dark[0, 0]  # (T,H,W)
            k = max(2, T // 6)
            topk, _ = torch.topk(dark, k=k, dim=0)
            coherent = topk.mean(dim=0)
            rpca_score = coherent.cpu().numpy()

        bg = video[0, 0].mean(dim=0).cpu().numpy()
        bg_norm = (bg - bg.min()) / max(1.0e-6, bg.max() - bg.min())

        # ---------- Frangi vesselness on the DSA-like dark-channel image ----
        # Build a "DSA" image: temporal-min minus temporal-mean gives dark
        # vessel intrusion against background. Then run multiscale Frangi.
        tmin = video[0, 0].amin(dim=0).cpu().numpy()
        dsa = np.clip(bg - tmin, 0.0, None)
        dsa = (dsa - dsa.min()) / max(1.0e-6, dsa.max() - dsa.min())
        # Frangi expects bright ridges; dsa is already bright on vessels.
        sigmas = np.linspace(args.sigma_min, args.sigma_max, args.n_scales)
        frangi_map = frangi(dsa, sigmas=sigmas, black_ridges=False,
                            alpha=0.5, beta=0.5)
        frangi_norm = frangi_map / (frangi_map.max() + 1.0e-8)

        # ---------- Fuse RPCA temporal score with Frangi tubular response --
        # Both must agree (geometric mean) to keep a pixel.
        rs = rpca_score
        rs_norm = (rs - rs.min()) / max(1.0e-6, rs.max() - rs.min())
        fused = np.sqrt(rs_norm * frangi_norm)

        # ---------- Threshold + morphological cleanup -----------------------
        thr = float(np.percentile(fused, args.gate_pct))
        mask = fused > thr
        # Drop tiny speckle
        mask = remove_small_objects(mask, min_size=args.min_size)
        # Close small gaps along the vessel
        mask = binary_closing(mask, footprint=disk(args.close_radius))
        # Keep only K largest connected components (removes scattered artifacts)
        lbl, nlab = ndi.label(mask)
        if nlab > 0 and args.keep_components > 0:
            sizes = ndi.sum(mask, lbl, range(1, nlab + 1))
            order = np.argsort(sizes)[::-1][: args.keep_components]
            keep = np.zeros_like(mask)
            for idx in order:
                keep |= (lbl == (idx + 1))
            mask = keep

        # ---------- Build viz field -----------------------------------------
        if mode == "skeleton":
            skel = skeletonize(mask)
            # Make 1-px lines visible at 512x512
            skel_vis = binary_dilation(skel, footprint=disk(args.skel_thickness))
            viz_field = (skel_vis.astype(np.float32) *
                         (fused / (fused.max() + 1e-8)))
            paint = skel_vis
        elif mode == "dense":
            viz_field = fused * mask
            viz_field = viz_field / (viz_field.max() + 1e-8)
            paint = mask
        else:  # raw
            viz_field = rs_norm
            paint = np.ones_like(mask, dtype=bool)

        # ---------- Overlay --------------------------------------------------
        bg_rgb = np.stack([bg_norm, bg_norm, bg_norm], axis=-1)
        cmap = cm.get_cmap(args.cmap)
        score_rgba = cmap(viz_field)[..., :3]
        a = np.where(paint, (viz_field ** args.gamma) * alpha_blend, 0.0)
        a = a[..., None]
        overlay = bg_rgb * (1.0 - a) + score_rgba * a
        overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

        np.save(out_dir / f"vessel_score_{i:03d}.npy", fused)
        Image.fromarray(overlay).save(out_dir / f"overlay_{i:03d}.png")
        Image.fromarray((bg_norm * 255).astype(np.uint8)).save(
            out_dir / f"bg_{i:03d}.png")
        Image.fromarray((np.clip(viz_field, 0, 1) * 255).astype(np.uint8)).save(
            out_dir / f"score_{i:03d}.png")
        Image.fromarray((mask.astype(np.uint8) * 255)).save(
            out_dir / f"mask_{i:03d}.png")
        Image.fromarray((np.clip(frangi_norm, 0, 1) * 255).astype(np.uint8)).save(
            out_dir / f"frangi_{i:03d}.png")
        Image.fromarray((np.clip(dsa, 0, 1) * 255).astype(np.uint8)).save(
            out_dir / f"dsa_{i:03d}.png")

        summary.append({
            "index": i, "path": item["path"],
            "shape": [int(H), int(W)], "mode": mode,
            "mask_pixels": int(mask.sum()),
            "frangi_max": float(frangi_map.max()),
        })
        print(f"[{i}] {Path(item['path']).name}  {H}x{W}  "
              f"mask_px={int(mask.sum())}  "
              f"frangi_max={frangi_map.max():.3f}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--limit", type=int, default=None)
    p_train.set_defaults(func=cmd_train)

    p_evt = sub.add_parser("score-events",
        help="Event-camera vessel saliency from HEE polarity stream")
    p_evt.add_argument("--config", required=True)
    p_evt.add_argument("--out", required=True)
    p_evt.add_argument("--num", type=int, default=5)
    p_evt.add_argument("--full-res", type=int, default=None)
    p_evt.add_argument("--alpha", type=float, default=0.85)
    p_evt.add_argument("--gamma", type=float, default=1.0)
    p_evt.add_argument("--cmap", default="turbo",
                       help="colormap for arrival-time phase (turbo/viridis/jet)")
    # HEE params
    p_evt.add_argument("--theta0", type=float, default=0.05)
    p_evt.add_argument("--adapt-alpha", type=float, default=2.0)
    p_evt.add_argument("--win-h", type=int, default=7)
    p_evt.add_argument("--win-w", type=int, default=7)
    p_evt.add_argument("--win-t", type=int, default=5)
    p_evt.add_argument("--drift-window", type=int, default=31)
    # Coherence / transit params
    p_evt.add_argument("--coh-k", type=int, default=5,
                       help="local window for arrival-time smoothing")
    p_evt.add_argument("--coh-sigma", type=float, default=1.5,
                       help="tolerated arrival-time residual (frames)")
    p_evt.add_argument("--transit-min", type=int, default=2,
                       help="minimum frames between inflow and washout")
    p_evt.add_argument("--transit-max", type=int, default=40,
                       help="maximum frames between inflow and washout")
    p_evt.set_defaults(func=cmd_score_events)

    p_rpca = sub.add_parser("score-rpca",
        help="RPCA-only baseline overlay (no model needed)")
    p_rpca.add_argument("--config", required=True)
    p_rpca.add_argument("--out", required=True)
    p_rpca.add_argument("--num", type=int, default=5)
    p_rpca.add_argument("--alpha", type=float, default=0.75)
    p_rpca.add_argument("--full-res", type=int, default=None)
    p_rpca.add_argument("--rpca-iter", type=int, default=40)
    p_rpca.add_argument("--gamma", type=float, default=1.0,
                        help="alpha gamma; >1 = sharper")
    p_rpca.add_argument("--gate-pct", type=float, default=92.0,
                        help="percentile used as fused-score threshold")
    p_rpca.add_argument("--mode", choices=["dense", "skeleton", "raw"],
                        default="dense",
                        help="dense = Frangi-cleaned mask overlay; "
                             "skeleton = centerline; raw = unprocessed RPCA")
    p_rpca.add_argument("--sigma-min", type=float, default=1.0)
    p_rpca.add_argument("--sigma-max", type=float, default=4.0)
    p_rpca.add_argument("--n-scales", type=int, default=5)
    p_rpca.add_argument("--min-size", type=int, default=120,
                        help="remove connected components smaller than this (px)")
    p_rpca.add_argument("--close-radius", type=int, default=2,
                        help="binary closing disk radius (bridges gaps)")
    p_rpca.add_argument("--keep-components", type=int, default=6,
                        help="keep K largest components (0 = keep all)")
    p_rpca.add_argument("--skel-thickness", type=int, default=1,
                        help="dilation radius for skeleton visualization")
    p_rpca.add_argument("--cmap", default="hot")
    p_rpca.set_defaults(func=cmd_score_rpca)

    p_flow = sub.add_parser(
        "score-flow",
        help="segmentacja z przepływu kontrastu (bez checkpointu JEPA)",
    )
    p_flow.add_argument("--config", required=True)
    p_flow.add_argument("--out", required=True)
    p_flow.add_argument("--num", type=int, default=5)
    p_flow.add_argument("--alpha", type=float, default=0.75)
    p_flow.add_argument("--full-res", type=int, default=None)
    p_flow.set_defaults(func=cmd_score, flow_weight=1.0, ckpt=None, passes=0)

    p_score = sub.add_parser("score")
    p_score.add_argument("--ckpt", default=None,
                          help="wymagany gdy --flow-weight < 1.0")
    p_score.add_argument("--config", required=True)
    p_score.add_argument("--out", required=True)
    p_score.add_argument("--num", type=int, default=5)
    p_score.add_argument("--passes", type=int, default=8,
                          help="random mask passes averaged per clip")
    p_score.add_argument("--alpha", type=float, default=0.75,
                          help="overlay max alpha (0..1)")
    p_score.add_argument("--full-res", type=int, default=None,
                          help="override target_size with this NxN (e.g. 512)")
    p_score.add_argument("--flow-weight", type=float, default=0.85,
                          help="weight for physics flow segmenter (rest = JEPA)")
    p_score.set_defaults(func=cmd_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
