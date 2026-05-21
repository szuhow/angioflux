import torch

from angio_flux.neurons import AdLIFCell


def test_adlif_step_and_grad():
    cell = AdLIFCell(beta=0.9, rho=0.95, a=0.05, b=0.1, v_th=1.0)
    I = (torch.randn(2, 4) * 0.5).requires_grad_(True)
    state = cell.init_state(torch.Size((2, 4)), device=I.device, dtype=I.dtype)
    spikes = []
    s = state
    for ti in range(10):
        sp, _, s = cell(I, s, t_index=ti)
        spikes.append(sp)
    out = torch.stack(spikes).sum()
    out.backward()
    assert I.grad is not None
    assert torch.isfinite(I.grad).all()
