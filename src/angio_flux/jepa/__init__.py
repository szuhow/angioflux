"""Vessel-JEPA: latent self-prediction with flow-ordering prior.

Self-supervised pretraining where a small ViT-3D encoder embeds spacetime
tubelets and a narrow predictor predicts target tubelet embeddings from
context tubelets conditioned on relative (dx, dy, dt) position. Target
embeddings come from an EMA-updated teacher (BYOL/JEPA style). An auxiliary
flow-ordering rank loss pushes upstream tubelets (earlier RPCA-dark inflow)
to have smaller bolus-arrival predictions than downstream ones.
"""
from .tubelet import TubeletPatchifier, build_grid_positions
from .encoder import ViT3DEncoder, EMATargetEncoder
from .predictor import JEPAPredictor
from .masking import sample_context_target_masks
from .sampling import rpca_inflow_times, flow_aware_target_indices
from .vjepa import VesselJEPA

__all__ = [
    "TubeletPatchifier",
    "build_grid_positions",
    "ViT3DEncoder",
    "EMATargetEncoder",
    "JEPAPredictor",
    "sample_context_target_masks",
    "rpca_inflow_times",
    "flow_aware_target_indices",
    "VesselJEPA",
]
