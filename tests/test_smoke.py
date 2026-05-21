import torch
import yaml

from angio_flux.data import make_batch
from angio_flux.losses import AngioFluxLoss
from angio_flux.model import AngioFlux


def _cfg():
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_forward_backward_smoke():
    cfg = _cfg()
    cfg["data"]["height"] = 48
    cfg["data"]["width"] = 48
    cfg["data"]["num_frames"] = 8
    cfg["data"]["batch_size"] = 1
    cfg["sunet"]["base_channels"] = 8
    cfg["flux_gnn"]["hidden_dim"] = 16
    cfg["voxel"]["num_bins"] = 2

    torch.manual_seed(0)
    model = AngioFlux(cfg)
    loss_fn = AngioFluxLoss(cfg)
    batch = make_batch(1, 48, 48, 8, seed=0)
    out = model(batch["video"])

    assert out["mask"].shape == (1, 1, 48, 48)
    assert out["vqfr"].shape == (1,)
    assert torch.all((out["vqfr"] >= 0.0) & (out["vqfr"] <= 1.0))

    loss, parts = loss_fn(out, batch)
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad
    assert torch.isfinite(loss)
