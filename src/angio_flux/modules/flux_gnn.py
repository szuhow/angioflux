"""FLUX-GNN: sparse spatio-temporal graph net over burst centroids.

Pipeline:
    1. Take encoder hemodynamic stream (B, C, H, W) and segmentation mask.
    2. Extract top-K bursting centroids on the vessel skeleton.
    3. Build a kNN graph in (x, y, mean-time) space; compute edge features
       including a Venturi-score (local firing-rate ratio).
    4. Run L message-passing rounds with edge-conditioned attention.
    5. Heads:
        - AHA-17 node classification
        - Stenosis-grade per node (4 bins) + Venturi-score regression
        - Graph-level pooled embedding (for downstream vQFR head)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _topk_centroids(score_map: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """Pick top-k pixel coordinates per batch where score_map * mask is highest.

    Returns: (B, k, 3) with [x, y, score] (x in [-1,1], y in [-1,1]).
    """
    b, _, h, w = score_map.shape
    sc = (score_map * mask).reshape(b, -1)
    k = min(k, sc.shape[1])
    top = sc.topk(k, dim=1)
    idx = top.indices  # (B, k)
    ys = (idx // w).float()
    xs = (idx % w).float()
    xs_n = 2.0 * xs / max(1, w - 1) - 1.0
    ys_n = 2.0 * ys / max(1, h - 1) - 1.0
    return torch.stack([xs_n, ys_n, top.values], dim=-1)


def _knn_edges(coords: torch.Tensor, k: int) -> torch.Tensor:
    """coords: (B, N, D) → (B, N, k) neighbor indices (excluding self)."""
    # squared distance
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B,N,N,D)
    d2 = (diff * diff).sum(-1)
    d2.diagonal(dim1=1, dim2=2).fill_(float("inf"))
    return d2.topk(k, dim=-1, largest=False).indices


class FluxGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_aha_classes: int = 17,
        num_stenosis_bins: int = 4,
        knn_k: int = 8,
        num_layers: int = 2,
        num_nodes: int = 128,
    ) -> None:
        super().__init__()
        self.k = knn_k
        self.num_nodes = num_nodes
        self.num_layers = num_layers

        # Node feature dimension: [channel_feature_at_pixel ... + (x,y,score,isi_inv)]
        node_in = in_channels + 4
        self.node_embed = nn.Linear(node_in, hidden_dim)

        # Edge features: [dx, dy, dist, venturi, score_j-score_i]
        edge_in = 5
        self.edge_embed = nn.Linear(edge_in, hidden_dim)

        self.msg_layers = nn.ModuleList(
            [nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(num_layers)]
        )
        self.upd_layers = nn.ModuleList(
            [nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(num_layers)]
        )
        self.attn_layers = nn.ModuleList(
            [nn.Linear(hidden_dim * 2, 1) for _ in range(num_layers)]
        )

        self.aha_head = nn.Linear(hidden_dim, num_aha_classes)
        self.sten_head = nn.Linear(hidden_dim, num_stenosis_bins)
        self.venturi_head = nn.Linear(hidden_dim, 1)
        self.graph_head = nn.Linear(hidden_dim, hidden_dim)

    def _sample_features(self, features: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Bilinear-sample feature map at normalized (x, y) ∈ [-1, 1].
        features: (B, C, H, W); coords (B, N, 2). Returns (B, N, C).
        """
        grid = coords.unsqueeze(2)  # (B, N, 1, 2)
        sampled = F.grid_sample(features, grid, mode="bilinear", align_corners=True)
        # (B, C, N, 1) -> (B, N, C)
        return sampled.squeeze(-1).permute(0, 2, 1).contiguous()

    def forward(
        self,
        features: torch.Tensor,
        hemo_rate: torch.Tensor,
        vessel_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            features:   (B, C, H, W) decoder features from S-UNet++
            hemo_rate:  (B, C', H, W) hemodynamic spike-rate stream (PLSR output)
            vessel_mask:(B, 1, H, W) vessel probability (segmentation)
        Returns:
            dict of head outputs + node embeddings.
        """
        # Combine hemo channels into a scalar 'firing intensity' map.
        firing = hemo_rate.mean(dim=1, keepdim=True)  # (B,1,H,W)

        nodes_xyzs = _topk_centroids(firing, vessel_mask, self.num_nodes)  # (B,N,3)
        coords = nodes_xyzs[..., :2]
        scores = nodes_xyzs[..., 2:3]

        # Sample firing density & feature vec at each node.
        fmap = self._sample_features(features, coords)        # (B,N,C)
        fire_at_node = self._sample_features(firing, coords)  # (B,N,1)
        # ISI^{-1} proxy = firing intensity.
        isi_inv = fire_at_node

        node_in = torch.cat([fmap, coords, scores, isi_inv], dim=-1)
        h = self.node_embed(node_in)  # (B,N,H)

        # ---- build graph ----
        nbrs = _knn_edges(coords, k=self.k)  # (B,N,k)
        b, n, kk = nbrs.shape

        # Edge features.
        gather = lambda x, idx: torch.gather(
            x.unsqueeze(2).expand(-1, -1, kk, -1), 1, idx.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1])
        )
        coords_j = gather(coords, nbrs)                     # (B,N,k,2)
        coords_i = coords.unsqueeze(2).expand_as(coords_j)
        d = coords_j - coords_i
        dist = d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        fire_i = fire_at_node.unsqueeze(2).expand(-1, -1, kk, -1)
        fire_j = gather(fire_at_node, nbrs)
        venturi = (fire_i + 1e-6) / (fire_j + 1e-6) + (fire_j + 1e-6) / (fire_i + 1e-6) - 2.0
        score_diff = gather(scores, nbrs) - scores.unsqueeze(2).expand(-1, -1, kk, -1)
        edge_feat = torch.cat([d, dist, venturi, score_diff], dim=-1)
        e = self.edge_embed(edge_feat)  # (B,N,k,H)

        for li in range(self.num_layers):
            h_j = gather(h, nbrs)                         # (B,N,k,H)
            h_i = h.unsqueeze(2).expand_as(h_j)
            msg_in = torch.cat([h_j + e, h_i], dim=-1)
            msg = self.msg_layers[li](msg_in)             # (B,N,k,H)
            attn = self.attn_layers[li](torch.cat([h_i, h_j + e], dim=-1)).squeeze(-1)
            attn = F.softmax(attn, dim=-1).unsqueeze(-1)
            agg = (attn * msg).sum(dim=2)                 # (B,N,H)
            h = F.gelu(self.upd_layers[li](torch.cat([h, agg], dim=-1))) + h

        aha = self.aha_head(h)
        sten = self.sten_head(h)
        venturi_score = self.venturi_head(h).squeeze(-1)  # per-node Venturi prediction
        graph_emb = self.graph_head(h.mean(dim=1))

        return {
            "node_embed": h,
            "node_coords": coords,
            "aha_logits": aha,            # (B,N,17)
            "stenosis_logits": sten,      # (B,N,bins)
            "venturi": venturi_score,     # (B,N)
            "graph_embed": graph_emb,     # (B,H)
        }
