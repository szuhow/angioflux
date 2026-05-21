"""Adaptive Leaky Integrate-and-Fire neuron (AdLIF), EKG-free variant.

State:
    V[t+1] = beta * V[t] + I[t] - w[t] - S[t] * v_th
    w[t+1] = rho  * w[t] + a * V[t]  + b * S[t]
    S[t]   = 1{ V[t] >= v_th }

Backprop uses a fast-sigmoid surrogate for the Heaviside spike.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class _FastSigmoidSurrogate(torch.autograd.Function):
    """Heaviside in forward, fast-sigmoid derivative in backward."""

    @staticmethod
    def forward(ctx, v_minus_th: torch.Tensor, slope: float) -> torch.Tensor:
        ctx.save_for_backward(v_minus_th)
        ctx.slope = slope
        return (v_minus_th >= 0).float()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        (v_minus_th,) = ctx.saved_tensors
        denom = 1.0 + ctx.slope * v_minus_th.abs()
        grad = grad_out / (denom * denom)
        return grad, None


def spike_fn(v_minus_th: torch.Tensor, slope: float = 10.0) -> torch.Tensor:
    return _FastSigmoidSurrogate.apply(v_minus_th, slope)


@dataclass
class AdLIFState:
    V: torch.Tensor
    w: torch.Tensor
    last_spike_t: torch.Tensor   # for ISI tracking; -1 = never
    isi: torch.Tensor            # most recent inter-spike interval (in steps)


class AdLIFCell(nn.Module):
    """Stateless module that advances one AdLIF step.

    Inputs/outputs are arbitrary-shape tensors (the cell is element-wise);
    callers are responsible for the spatial convolution / linear layer that
    produces the synaptic current `I`.
    """

    def __init__(
        self,
        beta: float = 0.9,
        rho: float = 0.95,
        a: float = 0.05,
        b: float = 0.1,
        v_th: float = 1.0,
        surrogate_slope: float = 10.0,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.rho = rho
        self.a = a
        self.b = b
        self.v_th = v_th
        self.surrogate_slope = surrogate_slope

    def init_state(self, shape: torch.Size, device: torch.device, dtype: torch.dtype) -> AdLIFState:
        zeros = torch.zeros(shape, device=device, dtype=dtype)
        return AdLIFState(
            V=zeros.clone(),
            w=zeros.clone(),
            last_spike_t=torch.full(shape, -1.0, device=device, dtype=dtype),
            isi=torch.zeros(shape, device=device, dtype=dtype),
        )

    def forward(
        self,
        I: torch.Tensor,
        state: AdLIFState,
        t_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, AdLIFState]:
        """Advance one timestep.

        Returns:
            spike: same shape as I
            V_new: membrane potential (post-update, pre-reset by spike)
            state: updated AdLIFState
        """
        V_new = self.beta * state.V + I - state.w
        spike = spike_fn(V_new - self.v_th, slope=self.surrogate_slope)
        # Soft reset (subtractive)
        V_post = V_new - spike * self.v_th
        w_new = self.rho * state.w + self.a * state.V + self.b * spike

        # ISI bookkeeping (no-grad)
        with torch.no_grad():
            t_now = torch.full_like(state.last_spike_t, float(t_index))
            new_isi = torch.where(
                (spike > 0.5) & (state.last_spike_t >= 0),
                t_now - state.last_spike_t,
                state.isi,
            )
            new_last = torch.where(spike > 0.5, t_now, state.last_spike_t)

        new_state = AdLIFState(V=V_post, w=w_new, last_spike_t=new_last, isi=new_isi)
        return spike, V_post, new_state
