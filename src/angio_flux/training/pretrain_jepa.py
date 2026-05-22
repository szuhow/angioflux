"""V-JEPA pretraining loop on real angiography clips."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from ..data import AngioSequenceDataset, collate_pad
from ..jepa import VesselJEPA
from ..jepa.masking import batched_masks
from ..jepa.sampling import rpca_inflow_times
from ..losses.jepa import VesselJEPALoss
from ..preprocess import rpca_sparse_component


def cosine_momentum(step: int, total_steps: int, m_start: float, m_end: float = 1.0) -> float:
    if total_steps <= 0:
        return m_start
    frac = min(1.0, step / total_steps)
    return m_end - (m_end - m_start) * 0.5 * (1.0 + math.cos(math.pi * frac))


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
        max_frames=ds_cfg.get("max_frames", 64),
        min_frames=ds_cfg.get("min_frames", 16),
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

    j_cfg = cfg["jepa"]
    model = VesselJEPA(
        in_channels=1,
        patch_size=j_cfg["patch_size"],
        tubelet_t=j_cfg["tubelet_t"],
        embed_dim=j_cfg["embed_dim"],
        encoder_depth=j_cfg["encoder_depth"],
        encoder_heads=j_cfg["encoder_heads"],
        predictor_dim=j_cfg["predictor_dim"],
        predictor_depth=j_cfg["predictor_depth"],
        predictor_heads=j_cfg["predictor_heads"],
        ema_momentum=j_cfg.get("ema_momentum_start", 0.998),
    ).to(device)

    loss_fn = VesselJEPALoss(
        lambda_jepa=cfg["loss"]["lambda_jepa"],
        lambda_var=cfg["loss"]["lambda_vicreg_var"],
        lambda_cov=cfg["loss"]["lambda_vicreg_cov"],
        lambda_bat=cfg["loss"]["lambda_bat_rank"],
        var_target=cfg["loss"].get("vicreg_var_target", 1.0),
        bat_margin=cfg["loss"].get("bat_margin", 0.05),
    )

    optim = torch.optim.AdamW(
        list(model.student.parameters())
        + list(model.predictor.parameters())
        + list(model.bat_head.parameters()),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.05),
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] V-JEPA params: {n_params/1e6:.3f}M")

    ckpt_dir = Path(cfg["train"].get("ckpt_dir", "checkpoints/jepa"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    epochs = cfg["train"]["epochs"]
    total_steps = max(1, epochs * len(loader))
    m_start = j_cfg.get("ema_momentum_start", 0.998)
    m_end = j_cfg.get("ema_momentum_end", 1.0)

    global_step = 0
    for epoch in range(epochs):
        running = 0.0
        n = 0
        t0 = time.time()
        for batch in loader:
            video = batch["video"].to(device)
            optim.zero_grad()

            # RPCA-dark prior for flow-aware sampling + BAT-rank
            rpca_dark = None
            inflow_time_all = None
            presence_all = None
            if cfg.get("use_rpca_prior", True):
                with torch.no_grad():
                    rpca_dark = rpca_sparse_component(
                        video,
                        lam=cfg.get("rpca", {}).get("lam"),
                        max_iter=cfg.get("rpca", {}).get("max_iter", 20),
                        tol=cfg.get("rpca", {}).get("tol", 1.0e-5),
                    )

            # Patch all tokens once with the student patch embedder; the target
            # EMA encoder will patchify again internally (its conv has the
            # same weights via EMA copy, modulo a small lag).
            tokens_all, positions_all, grid = model.patch_and_positions(video)
            if rpca_dark is not None:
                inflow_time_all, presence_all = rpca_inflow_times(rpca_dark, grid)

            ctx_idx, tgt_idx = batched_masks(
                video.shape[0],
                grid,
                num_targets=j_cfg.get("num_target_blocks", 4),
                target_block=tuple(j_cfg.get("target_block", [1, 2, 2])),
                context_fraction=j_cfg.get("context_fraction", 0.6),
                seed=global_step,
                presence=presence_all,
                inflow_time=inflow_time_all,
                flow_bias=j_cfg.get("flow_target_bias", 0.7)
                if presence_all is not None
                else 0.0,
            )
            ctx_idx = ctx_idx.to(device)
            tgt_idx = tgt_idx.to(device)

            ctx_enc, ctx_pos = model.encode_context(tokens_all, positions_all, ctx_idx)
            pred, _ = model.predict_targets(ctx_enc, ctx_pos, positions_all, tgt_idx)
            target_emb = model.target_embeddings(video, tgt_idx)

            # Full-grid student pass for BAT head (cheap because we reuse tokens_all)
            full_student = model.student(tokens_all, positions_all)
            bat_scores = model.bat_head(full_student).squeeze(-1)  # (B, N)

            loss, parts = loss_fn(
                pred,
                target_emb,
                bat_scores=bat_scores,
                inflow_time=inflow_time_all,
            )
            loss.backward()
            trainable = (
                list(model.student.parameters())
                + list(model.predictor.parameters())
                + list(model.bat_head.parameters())
            )
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step()

            # EMA update with cosine momentum schedule
            m = cosine_momentum(global_step, total_steps, m_start, m_end)
            model.ema_momentum = m
            model.update_ema()

            running += float(loss.detach())
            n += 1
            global_step += 1
            if n % 5 == 0:
                msg = " ".join(f"{k}={v.item():.4f}" for k, v in parts.items())
                print(f"  step {n} ema_m={m:.4f} {msg}")
        dt = time.time() - t0
        print(f"[epoch {epoch}] mean_loss={running / max(1, n):.4f}  time={dt:.1f}s  steps={n}")
        torch.save(
            {"model": model.state_dict(), "cfg": cfg, "epoch": epoch},
            ckpt_dir / f"vjepa_epoch{epoch:02d}.pt",
        )


if __name__ == "__main__":
    main()
