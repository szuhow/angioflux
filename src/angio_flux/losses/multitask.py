"""Loss functions for Angio-FLUX."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = pred.flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    return 1.0 - (2.0 * inter + eps) / (p.sum(1) + t.sum(1) + eps)


def soft_skeleton(x: torch.Tensor, iters: int = 3) -> torch.Tensor:
    """Differentiable soft skeletonization used by clDice.

    This follows the standard soft-skeletonize recipe: skeleton mass is the
    residual between the mask and its soft opening, then the same residual is
    accumulated after iterative soft erosions. One-pixel-wide lines therefore
    remain non-empty instead of being erased by the first erosion.
    """
    def soft_erode(img: torch.Tensor) -> torch.Tensor:
        erode_y = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
        erode_x = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
        return torch.minimum(erode_x, erode_y)

    def soft_dilate(img: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)

    def soft_open(img: torch.Tensor) -> torch.Tensor:
        return soft_dilate(soft_erode(img))

    img = x.clamp(0, 1)
    opened = soft_open(img)
    skel = F.relu(img - opened)
    for _ in range(iters):
        img = soft_erode(img)
        opened = soft_open(img)
        delta = F.relu(img - opened)
        skel = skel + F.relu(delta - skel * delta)
    return skel.clamp(0, 1)


def cl_dice_loss(pred: torch.Tensor, target: torch.Tensor, iters: int = 3, eps: float = 1e-6) -> torch.Tensor:
    sk_pred = soft_skeleton(pred, iters)
    sk_targ = soft_skeleton(target, iters)
    tprec = (sk_pred * target).sum((1, 2, 3)) / (sk_pred.sum((1, 2, 3)) + eps)
    tsens = (sk_targ * pred).sum((1, 2, 3)) / (sk_targ.sum((1, 2, 3)) + eps)
    return 1.0 - 2.0 * tprec * tsens / (tprec + tsens + eps)


def spike_rate_penalty(spike_rates: list[torch.Tensor], target: float) -> torch.Tensor:
    losses = []
    for sr in spike_rates:
        losses.append((sr.mean() - target).abs())
    return torch.stack(losses).mean()


def focal_ce_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """logits: (..., C), targets: (...) long. Reduces to scalar."""
    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    tgt = targets.unsqueeze(-1)
    log_pt = log_p.gather(-1, tgt).squeeze(-1)
    pt = p.gather(-1, tgt).squeeze(-1)
    return -((1.0 - pt) ** gamma * log_pt).mean()


class AngioFluxLoss(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg["loss"]

    def forward(self, out: dict, batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mask = out["mask"]
        target = batch["mask"]
        l_dice = dice_loss(mask, target).mean()
        l_cl = cl_dice_loss(mask, target).mean()
        l_seg = l_dice + self.cfg["lambda_cldice"] * l_cl

        l_aha = torch.tensor(0.0, device=mask.device)
        l_sten = torch.tensor(0.0, device=mask.device)
        if "aha" in batch:
            l_aha = focal_ce_loss(out["aha_logits"], batch["aha"])
        if "stenosis" in batch:
            l_sten = focal_ce_loss(out["stenosis_logits"], batch["stenosis"])

        l_vqfr = torch.tensor(0.0, device=mask.device)
        if "vqfr" in batch:
            l_vqfr = F.smooth_l1_loss(out["vqfr"], batch["vqfr"])

        l_spike = spike_rate_penalty(out["spike_rates"], self.cfg["spike_target_rate"])

        total = (
            l_seg
            + self.cfg["lambda_aha"] * l_aha
            + self.cfg["lambda_stenosis"] * l_sten
            + self.cfg["lambda_vqfr"] * l_vqfr
            + self.cfg["lambda_spike"] * l_spike
        )
        return total, {
            "dice": l_dice.detach(),
            "cldice": l_cl.detach(),
            "aha": l_aha.detach(),
            "stenosis": l_sten.detach(),
            "vqfr": l_vqfr.detach(),
            "spike": l_spike.detach(),
            "total": total.detach(),
        }
