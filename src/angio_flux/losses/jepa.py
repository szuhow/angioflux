"""JEPA losses: latent SmoothL1 + VICReg + BAT-rank flow ordering."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def jepa_latent_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """SmoothL1 over (B, M, D) latent predictions vs (sg) target embeddings."""
    return F.smooth_l1_loss(pred, target.detach(), beta=1.0)


def vicreg_terms(
    embeddings: torch.Tensor,
    var_target: float = 1.0,
    eps: float = 1.0e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg variance & covariance terms.

    Args:
        embeddings: (B*M, D) or (B, M, D)
    Returns:
        var_loss: hinge so each feature has std >= var_target
        cov_loss: off-diagonal covariance penalty
    """
    if embeddings.dim() == 3:
        x = embeddings.reshape(-1, embeddings.shape[-1])
    else:
        x = embeddings
    x = x - x.mean(dim=0, keepdim=True)
    n, d = x.shape
    std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
    var_loss = F.relu(var_target - std).mean()
    cov = (x.T @ x) / max(1, n - 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    cov_loss = (off_diag.pow(2).sum()) / d
    return var_loss, cov_loss


def bat_rank_loss(
    bat_scores: torch.Tensor,
    inflow_time: torch.Tensor,
    num_pairs: int = 256,
    margin: float = 0.05,
) -> torch.Tensor:
    """Hinge ranking loss: upstream (earlier inflow) should have smaller bat.

    Args:
        bat_scores: (B, N) predicted BAT score per tubelet
        inflow_time: (B, N) integer inflow frame, -1 if no inflow
        num_pairs: number of random valid pairs per batch element
        margin: hinge margin
    """
    b, n = bat_scores.shape
    device = bat_scores.device
    total = bat_scores.new_zeros(())
    count = 0
    for i in range(b):
        valid_idx = (inflow_time[i] >= 0).nonzero(as_tuple=True)[0]
        if valid_idx.numel() < 2:
            continue
        m = valid_idx.numel()
        k = min(num_pairs, m * (m - 1))
        ai = valid_idx[torch.randint(0, m, (k,), device=device)]
        bi = valid_idx[torch.randint(0, m, (k,), device=device)]
        for j in range(k):
            if ai[j] == bi[j]:
                others = valid_idx[valid_idx != ai[j]]
                if others.numel() > 0:
                    bi[j] = others[torch.randint(0, others.numel(), (1,), device=device)]
        diff_t = inflow_time[i, bi].float() - inflow_time[i, ai].float()  # +ve: bi later
        sign = torch.sign(diff_t)
        # Skip equal-time pairs.
        mask = sign != 0
        if mask.sum() == 0:
            continue
        ai = ai[mask]; bi = bi[mask]; sign = sign[mask]
        # We want bat[upstream] < bat[downstream].
        # If sign > 0: bi is downstream -> bat[bi] > bat[ai] -> loss = relu(margin - (bat[bi]-bat[ai]))
        # If sign < 0: ai is downstream
        d_score = bat_scores[i, bi] - bat_scores[i, ai]
        # Multiply by sign so positive means "in correct direction".
        signed = d_score * sign
        loss = F.relu(margin - signed).mean()
        total = total + loss
        count += 1
    if count == 0:
        return bat_scores.new_zeros(())
    return total / count


class VesselJEPALoss(nn.Module):
    def __init__(
        self,
        lambda_jepa: float = 1.0,
        lambda_var: float = 1.0,
        lambda_cov: float = 0.04,
        lambda_bat: float = 0.5,
        var_target: float = 1.0,
        bat_margin: float = 0.05,
    ) -> None:
        super().__init__()
        self.lambda_jepa = lambda_jepa
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.lambda_bat = lambda_bat
        self.var_target = var_target
        self.bat_margin = bat_margin

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        bat_scores: torch.Tensor | None = None,
        inflow_time: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        l_jepa = jepa_latent_loss(pred, target)
        var, cov = vicreg_terms(pred, var_target=self.var_target)
        total = self.lambda_jepa * l_jepa + self.lambda_var * var + self.lambda_cov * cov
        parts = {"jepa": l_jepa.detach(), "var": var.detach(), "cov": cov.detach()}
        if bat_scores is not None and inflow_time is not None and self.lambda_bat > 0:
            l_bat = bat_rank_loss(bat_scores, inflow_time, margin=self.bat_margin)
            total = total + self.lambda_bat * l_bat
            parts["bat_rank"] = l_bat.detach()
        parts["total"] = total.detach()
        return total, parts
