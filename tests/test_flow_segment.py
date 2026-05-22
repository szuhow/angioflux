import torch

from angio_flux.data.synthetic import make_sample
from angio_flux.preprocess import rpca_sparse_component
from angio_flux.segmentation import flow_vessel_segment


def test_flow_segment_finds_synthetic_vessel_centerline():
    sample = make_sample(height=96, width=96, num_frames=24, seed=7)
    video = sample.video.unsqueeze(0)
    if video.dim() == 4:
        video = video.unsqueeze(1)
    sparse = rpca_sparse_component(video, max_iter=6)

    out = flow_vessel_segment(
        video,
        rpca_sparse=sparse,
        pseudo_gt_cfg={
            "threshold": 0.04,
            "positive_quantile": 0.82,
            "edge_margin_px": 2,
            "scale_reference_size": 96,
            "use_rpca": True,
            "rpca_max_iter": 6,
        },
        coh_sigma=2.0,
        transit_min=1,
        transit_max=20,
    )

    soft = out["soft_mask"][0, 0]
    vessel_pts = sample.mask[0] > 0.5
    assert soft[vessel_pts].mean().item() > 0.15
    assert soft[vessel_pts].mean().item() > soft[~vessel_pts].mean().item() * 2.0


def test_flow_segment_hard_mask_nonempty_on_bolus():
    sample = make_sample(height=64, width=64, num_frames=20, seed=3)
    video = sample.video.unsqueeze(0)
    if video.dim() == 4:
        video = video.unsqueeze(1)
    out = flow_vessel_segment(video, use_rpca=True, pseudo_gt_cfg={"rpca_max_iter": 4})
    assert out["hard_mask"].sum().item() > 0
