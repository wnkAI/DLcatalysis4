"""
GVP-GNN pocket encoder for DLcatalysis 4.0.

Lightweight GVP implementation (scalar + vector channels) — no external
torch-drug dependency. Consumes the pocket tensors produced by
scripts/14_extract_pockets.py:

  node_s     (K, 26)   — AA 1hot + pLDDT + 4 flags + min_dist_to_substrate
  node_v     (K, 2, 3) — N→CA, C→CA unit vectors
  edge_index (2, E)    — k-NN CA graph
  edge_s     (E, 16)   — RBF CA-CA distance
  edge_v     (E, 1, 3) — src→dst CA unit vector

Output:
  pocket_tokens (B, K, D)        — per-residue embeddings for cross-attention
  pocket_pool   (B, D)           — attention-pooled pocket embedding (for y_struct)
  pocket_mask   (B, K) bool      — True = valid residue
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_batch


def _pairwise_norm(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Norm along last axis; clamp small values to avoid NaN."""
    return torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=eps)


class GVP(nn.Module):
    """Single Geometric Vector Perceptron layer.

    Inputs/outputs are tuples (s, v):
      s: (..., s_in)  scalar
      v: (..., v_in, 3) vector
    """
    def __init__(self, s_in, s_out, v_in, v_out, activations=(F.relu, torch.sigmoid)):
        super().__init__()
        self.s_in, self.s_out = s_in, s_out
        self.v_in, self.v_out = v_in, v_out
        self.h_v = max(v_in, v_out)

        self.vec_W_h = nn.Linear(v_in, self.h_v, bias=False)
        self.vec_W_mu = nn.Linear(self.h_v, v_out, bias=False)
        self.vec_W_scalar = nn.Linear(self.h_v, s_out, bias=False)

        self.scalar_W = nn.Linear(s_in + self.h_v, s_out)

        self.s_act, self.v_act = activations

    def forward(self, s: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # v: (..., v_in, 3)
        Vh = self.vec_W_h(v.transpose(-1, -2)).transpose(-1, -2)    # (..., h_v, 3)
        Vmu = self.vec_W_mu(Vh.transpose(-1, -2)).transpose(-1, -2) # (..., v_out, 3)
        vn_h = torch.linalg.norm(Vh, dim=-1)                        # (..., h_v)
        s_cat = torch.cat([s, vn_h], dim=-1)
        s_out = self.scalar_W(s_cat)
        s_out = self.s_act(s_out)

        # Vector gating by scalar projection
        gate = self.vec_W_scalar(vn_h)                              # (..., s_out)  (note: uses s_out dim)
        # Apply v_act to magnitudes; keep direction
        vn_mu = _pairwise_norm(Vmu)                                 # (..., v_out, 1)
        direction = Vmu / vn_mu
        # Gate with sigmoid of magnitudes themselves (simpler than full GVP paper; works well in practice)
        v_out = direction * self.v_act(vn_mu)
        return s_out, v_out


class GVPConv(MessagePassing):
    """GVP message-passing layer on residue graph."""
    def __init__(self, s_dim, v_dim, edge_s_dim=16, edge_v_dim=1):
        super().__init__(aggr="mean")
        self.s_dim = s_dim
        self.v_dim = v_dim

        # Message: combine (src scalar, edge scalar, src vec, edge vec) → (message_s, message_v)
        self.msg_gvp = GVP(
            s_in=s_dim + edge_s_dim, s_out=s_dim,
            v_in=v_dim + edge_v_dim, v_out=v_dim,
        )
        # Update: self + aggregated message
        self.upd_gvp = GVP(
            s_in=2 * s_dim, s_out=s_dim,
            v_in=2 * v_dim, v_out=v_dim,
        )
        self.ln_s = nn.LayerNorm(s_dim)

    def forward(self, s, v, edge_index, edge_s, edge_v):
        # PyG's propagate will call self.message + aggregate + self.update
        out_s, out_v = self.propagate(
            edge_index, s=s, v=v, edge_s=edge_s, edge_v=edge_v,
        )
        # Residual update
        s_cat = torch.cat([s, out_s], dim=-1)
        v_cat = torch.cat([v, out_v], dim=-2)  # concat along vector channel axis
        s_new, v_new = self.upd_gvp(s_cat, v_cat)
        s_new = self.ln_s(s + s_new)
        v_new = v + v_new
        return s_new, v_new

    def message(self, s_j, v_j, edge_s, edge_v):
        # s_j: (E, s_dim),  v_j: (E, v_dim, 3)
        # edge_s: (E, edge_s_dim), edge_v: (E, edge_v_dim, 3)
        msg_s = torch.cat([s_j, edge_s], dim=-1)
        msg_v = torch.cat([v_j, edge_v], dim=-2)
        m_s, m_v = self.msg_gvp(msg_s, msg_v)
        return m_s, m_v

    def aggregate(self, inputs, index, dim_size=None):
        m_s, m_v = inputs
        from torch_scatter import scatter
        agg_s = scatter(m_s, index, dim=0, dim_size=dim_size, reduce="mean")
        agg_v = scatter(m_v, index, dim=0, dim_size=dim_size, reduce="mean")
        return agg_s, agg_v

    def update(self, aggr_out):
        return aggr_out  # pass through; final combine in forward()


class PocketGVP(nn.Module):
    """Stack of GVP layers + attention pooling.

    Args:
        in_s_dim: pocket node_s dim (26 default from v2 features)
        in_v_dim: pocket node_v dim (2 default: N→CA, C→CA)
        hidden_dim: final scalar embedding dim (align to model.hidden_dim)
        v_hidden: vector hidden channels
        n_layers: GVP layers
        edge_s_dim: pocket edge_s dim (16 RBF)
        edge_v_dim: pocket edge_v dim (1)
    """
    def __init__(self, in_s_dim=26, in_v_dim=2,
                 hidden_dim=128, v_hidden=16,
                 n_layers=3,
                 edge_s_dim=16, edge_v_dim=1,
                 dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Lift raw scalar/vector features to hidden
        self.in_gvp = GVP(
            s_in=in_s_dim, s_out=hidden_dim,
            v_in=in_v_dim, v_out=v_hidden,
        )

        self.convs = nn.ModuleList([
            GVPConv(s_dim=hidden_dim, v_dim=v_hidden,
                    edge_s_dim=edge_s_dim, edge_v_dim=edge_v_dim)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        # Attention pool over residues (scalar-only)
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_s, node_v, edge_index, edge_s, edge_v, batch):
        """
        node_s: (N_total, in_s_dim)
        node_v: (N_total, in_v_dim, 3)
        edge_index: (2, E_total)
        edge_s: (E_total, edge_s_dim)
        edge_v: (E_total, edge_v_dim, 3)
        batch:  (N_total,) batch assignment

        Returns:
          pocket_tokens: (B, K_max, hidden_dim)  — padded dense
          pocket_pool:   (B, hidden_dim)        — attention-pooled
          pocket_mask:   (B, K_max) bool
        """
        s, v = self.in_gvp(node_s, node_v)
        for conv in self.convs:
            s_new, v_new = conv(s, v, edge_index, edge_s, edge_v)
            s = self.dropout(s_new)
            v = v_new

        # Dense batch
        pocket_tokens, pocket_mask = to_dense_batch(s, batch)  # (B, K, D), (B, K)

        # Attention pool
        scores = self.attn_pool(pocket_tokens)                 # (B, K, 1)
        scores = scores.masked_fill(~pocket_mask.unsqueeze(-1),
                                    torch.finfo(scores.dtype).min)
        has_valid = pocket_mask.any(dim=1, keepdim=True).unsqueeze(-1).float()
        weights = torch.softmax(scores, dim=1) * has_valid
        pocket_pool = (pocket_tokens * weights).sum(dim=1)     # (B, D)

        return pocket_tokens, pocket_pool, pocket_mask
