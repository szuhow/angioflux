import torch

from angio_flux.encoding.hee import HemodynamicEventEncoder, _adaptive_threshold, events_to_voxel


def test_hee_shapes_and_sparsity():
    enc = HemodynamicEventEncoder(theta0=0.05, alpha=2.0, window=(5, 5, 3))
    x = torch.rand(2, 1, 8, 32, 32)
    ev = enc(x)
    assert ev.shape == (2, 4, 7, 32, 32)
    # values are binary
    assert torch.all((ev == 0) | (ev == 1))


def test_voxel_grid_shape():
    enc = HemodynamicEventEncoder()
    x = torch.rand(1, 1, 16, 32, 32)
    ev = enc(x)
    vox = events_to_voxel(ev, num_bins=4)
    assert vox.shape == (1, 16, 32, 32)  # 4 polarities * 4 bins


def test_adaptive_threshold_is_higher_for_low_cv_drift():
    cv_low = torch.zeros(1, 1, 1, 1, 1)
    cv_high = torch.ones(1, 1, 1, 1, 1) * 3.0

    theta_low = _adaptive_threshold(cv_low, theta0=0.05, alpha=2.0)
    theta_high = _adaptive_threshold(cv_high, theta0=0.05, alpha=2.0)

    assert theta_low.item() > theta_high.item()


def test_p_in_is_first_darkening_only_and_p_out_needs_prior_inflow():
    enc = HemodynamicEventEncoder(theta0=0.05, alpha=0.0, window=(1, 1, 1))
    # darken, brighten, darken again at the same pixel
    x = torch.tensor([1.0, 0.8, 1.0, 0.6]).view(1, 1, 4, 1, 1)

    ev = enc(x)

    assert ev[0, 0, :, 0, 0].tolist() == [1.0, 0.0, 0.0]
    assert ev[0, 1, :, 0, 0].tolist() == [0.0, 1.0, 0.0]
