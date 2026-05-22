"""Spacetime block-masking strategy for V-JEPA.

We sample disjoint context and target index sets per batch element.
Targets are sampled as small spacetime blocks (so the predictor must
reason about local propagation, not single isolated voxels).
"""
from __future__ import annotations

import random

import torch

from .sampling import flow_aware_target_indices


def sample_context_target_masks(
    grid: tuple[int, int, int],
    num_targets: int = 4,
    target_block: tuple[int, int, int] = (1, 2, 2),
    context_fraction: float = 0.6,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (context_indices, target_indices) into the flat token list.

    Args:
        grid: (Gt, Gh, Gw) tubelet grid
        num_targets: number of target blocks
        target_block: (dt, dh, dw) size of each block
        context_fraction: fraction of remaining tokens used as context
    Returns:
        ctx_idx: (Mc,) long
        tgt_idx: (Mt,) long
    """
    rng = rng or random.Random()
    gt, gh, gw = grid
    n = gt * gh * gw

    def lin(t: int, y: int, x: int) -> int:
        return (t * gh + y) * gw + x

    target_set: set[int] = set()
    dt, dh, dw = target_block
    for _ in range(num_targets):
        t0 = rng.randint(0, max(0, gt - dt))
        y0 = rng.randint(0, max(0, gh - dh))
        x0 = rng.randint(0, max(0, gw - dw))
        for tt in range(dt):
            for yy in range(dh):
                for xx in range(dw):
                    target_set.add(lin(t0 + tt, y0 + yy, x0 + xx))

    candidate_ctx = [i for i in range(n) if i not in target_set]
    rng.shuffle(candidate_ctx)
    n_ctx = max(1, int(len(candidate_ctx) * context_fraction))
    ctx_list = candidate_ctx[:n_ctx]
    tgt_list = sorted(target_set)

    return (
        torch.tensor(ctx_list, dtype=torch.long),
        torch.tensor(tgt_list, dtype=torch.long),
    )


def _flow_biased_target_set(
    grid: tuple[int, int, int],
    num_targets: int,
    target_block: tuple[int, int, int],
    presence: torch.Tensor,
    inflow_time: torch.Tensor,
    bias_strength: float,
    rng: random.Random,
) -> set[int]:
    gt, gh, gw = grid
    dt, dh, dw = target_block

    def lin(t: int, y: int, x: int) -> int:
        return (t * gh + y) * gw + x

    target_set: set[int] = set()
    n = gt * gh * gw
    for _ in range(num_targets):
        center = int(flow_aware_target_indices(
            presence[0] if presence.dim() == 2 else presence,
            inflow_time[0] if inflow_time.dim() == 2 else inflow_time,
            num_targets=1,
            bias_strength=bias_strength,
            rng=rng,
        ).item())
        t0 = (center // (gh * gw)) % gt
        y0 = (center // gw) % gh
        x0 = center % gw
        t0 = min(t0, max(0, gt - dt))
        y0 = min(y0, max(0, gh - dh))
        x0 = min(x0, max(0, gw - dw))
        for tt in range(dt):
            for yy in range(dh):
                for xx in range(dw):
                    target_set.add(lin(t0 + tt, y0 + yy, x0 + xx))
    if len(target_set) < max(1, num_targets // 2):
        for _ in range(num_targets):
            t0 = rng.randint(0, max(0, gt - dt))
            y0 = rng.randint(0, max(0, gh - dh))
            x0 = rng.randint(0, max(0, gw - dw))
            for tt in range(dt):
                for yy in range(dh):
                    for xx in range(dw):
                        target_set.add(lin(t0 + tt, y0 + yy, x0 + xx))
    return target_set


def batched_masks(
    batch_size: int,
    grid: tuple[int, int, int],
    num_targets: int = 4,
    target_block: tuple[int, int, int] = (1, 2, 2),
    context_fraction: float = 0.6,
    seed: int | None = None,
    presence: torch.Tensor | None = None,
    inflow_time: torch.Tensor | None = None,
    flow_bias: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-batch masks padded to common Mc, Mt with -1 padding sentinel.

    Returns:
        ctx_idx: (B, Mc) long, padded with -1
        tgt_idx: (B, Mt) long, padded with -1
    """
    rng = random.Random(seed)
    ctxs, tgts = [], []
    for b in range(batch_size):
        if (
            flow_bias > 0
            and presence is not None
            and inflow_time is not None
            and presence.shape[0] > b
        ):
            gt, gh, gw = grid
            n = gt * gh * gw
            target_set = _flow_biased_target_set(
                grid,
                num_targets,
                target_block,
                presence[b],
                inflow_time[b],
                flow_bias,
                rng,
            )
            candidate_ctx = [i for i in range(n) if i not in target_set]
            rng.shuffle(candidate_ctx)
            n_ctx = max(1, int(len(candidate_ctx) * context_fraction))
            c = torch.tensor(candidate_ctx[:n_ctx], dtype=torch.long)
            t = torch.tensor(sorted(target_set), dtype=torch.long)
        else:
            c, t = sample_context_target_masks(
                grid, num_targets, target_block, context_fraction, rng
            )
        ctxs.append(c)
        tgts.append(t)
    mc = max(c.numel() for c in ctxs)
    mt = max(t.numel() for t in tgts)
    ctx_pad = torch.full((batch_size, mc), -1, dtype=torch.long)
    tgt_pad = torch.full((batch_size, mt), -1, dtype=torch.long)
    for i, (c, t) in enumerate(zip(ctxs, tgts)):
        ctx_pad[i, : c.numel()] = c
        tgt_pad[i, : t.numel()] = t
    return ctx_pad, tgt_pad
