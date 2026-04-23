"""
3D-aware cross-attention between pocket residues and substrate atoms.

Usage in model forward:
    y_int3d = Int3DCrossAttn(pocket_tokens, atom_tokens,
                             pocket_mask, atom_mask,
                             pocket_xyz, atom_xyz)

The distance RBF bias injects 3D geometry into attention scores:
    attn_logits = Q K^T / sqrt(d) + W_dist * rbf(||pocket_i - atom_j||)

When no atom 3D coordinates are available (e.g., substrate only has 2D
RDKit graph, no docked pose), pass atom_xyz=None → distance bias is
disabled automatically (acts as plain 1D residue-atom cross-attention).
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_rbf(dist: torch.Tensor, n_bins: int = 32,
                  min_d: float = 0.0, max_d: float = 20.0,
                  sigma: float = 1.5) -> torch.Tensor:
    """Gaussian RBF expansion. dist: (...,) → (..., n_bins)."""
    centers = torch.linspace(min_d, max_d, n_bins, device=dist.device, dtype=dist.dtype)
    d = (dist.unsqueeze(-1) - centers) ** 2
    return torch.exp(-d / (2 * sigma ** 2))


class Int3DCrossAttnLayer(nn.Module):
    """One layer of bidirectional residue-atom cross-attention.

    Updates both pocket and substrate tokens symmetrically.
    """
    def __init__(self, hidden_dim=128, n_heads=4, n_rbf=32,
                 max_dist=20.0, dropout=0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.h = n_heads
        self.d = hidden_dim
        self.d_head = hidden_dim // n_heads
        self.max_dist = max_dist
        self.n_rbf = n_rbf

        # pocket ↔ atom projections (Q, K, V for both directions)
        self.p_q = nn.Linear(hidden_dim, hidden_dim)
        self.p_k = nn.Linear(hidden_dim, hidden_dim)
        self.p_v = nn.Linear(hidden_dim, hidden_dim)
        self.a_q = nn.Linear(hidden_dim, hidden_dim)
        self.a_k = nn.Linear(hidden_dim, hidden_dim)
        self.a_v = nn.Linear(hidden_dim, hidden_dim)

        # Distance bias projections (shared across directions)
        self.dist_proj = nn.Linear(n_rbf, n_heads)

        # Near-attack conformation (NAC) bias: per-head learnable gain
        # applied to the outer product of (catalytic residue indicator) ×
        # (reaction-center atom indicator). Init at zero so behavior matches
        # prior checkpoints unless the model learns to turn it on.
        self.nac_gain = nn.Parameter(torch.zeros(n_heads))

        self.p_out = nn.Linear(hidden_dim, hidden_dim)
        self.a_out = nn.Linear(hidden_dim, hidden_dim)
        self.ln_p = nn.LayerNorm(hidden_dim)
        self.ln_a = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        # (B, L, D) → (B, H, L, d_head)
        B, L, _ = x.shape
        return x.reshape(B, L, self.h, self.d_head).transpose(1, 2)

    def _combine_heads(self, x):
        # (B, H, L, d_head) → (B, L, D)
        B, H, L, d = x.shape
        return x.transpose(1, 2).reshape(B, L, H * d)

    def _dist_bias(self, xyz_p, xyz_a):
        """xyz_p (B, K, 3), xyz_a (B, A, 3) → (B, H, K, A) bias."""
        # pairwise distances
        dists = torch.cdist(xyz_p, xyz_a)  # (B, K, A)
        dists = dists.clamp(max=self.max_dist)
        rbf = _gaussian_rbf(dists, n_bins=self.n_rbf, max_d=self.max_dist)  # (B, K, A, R)
        bias = self.dist_proj(rbf)  # (B, K, A, H)
        bias = bias.permute(0, 3, 1, 2)  # (B, H, K, A)
        return bias

    def forward(self, p_tokens, a_tokens,
                p_mask: torch.Tensor, a_mask: torch.Tensor,
                xyz_p: Optional[torch.Tensor] = None,
                xyz_a: Optional[torch.Tensor] = None,
                xyz_valid_per_sample: Optional[torch.Tensor] = None,
                p_nac: Optional[torch.Tensor] = None,
                a_nac: Optional[torch.Tensor] = None):
        """
        p_tokens (B, K, D), a_tokens (B, A, D)
        p_mask   (B, K) bool, a_mask (B, A) bool  (True = valid)
        xyz_p    (B, K, 3) float — pocket Cα coords
        xyz_a    (B, A, 3) float — substrate atom coords (optional)
        xyz_valid_per_sample (B,) bool — per-sample flag; if provided, the
                                         distance bias is zeroed for samples
                                         where xyz_a is an invalid (zero)
                                         fallback. This prevents one valid
                                         sample enabling fake distance bias
                                         for invalid zero-coord samples.
        p_nac    (B, K) float in [0,1] — catalytic-residue indicator
        a_nac    (B, A) float in [0,1] — reaction-center atom indicator
                 Their outer product is added as a learnable per-head bias
                 to the attention logits, nudging attention toward
                 catalytic residue ↔ reacting atom pairs (NAC proxy).
        """
        B, K, _ = p_tokens.shape
        _, A, _ = a_tokens.shape

        # Distance bias (disabled if no substrate xyz)
        if xyz_p is not None and xyz_a is not None:
            bias = self._dist_bias(xyz_p, xyz_a)  # (B, H, K, A)
            # Per-sample gating: zero out bias for samples with invalid xyz
            if xyz_valid_per_sample is not None:
                v = xyz_valid_per_sample.to(bias.dtype).to(bias.device)
                v = v.view(B, 1, 1, 1)                      # (B,1,1,1)
                bias = bias * v
        else:
            bias = None

        # NAC bias: catalytic × reaction-center outer product, per-head gain.
        # Kept as a separate additive term so distance and NAC signals are
        # independent (and either can be learned off).
        if p_nac is not None and a_nac is not None:
            nac_outer = (p_nac.float().unsqueeze(-1)
                         * a_nac.float().unsqueeze(-2))          # (B, K, A)
            nac_bias = (nac_outer.unsqueeze(1)
                        * self.nac_gain.view(1, -1, 1, 1))       # (B, H, K, A)
            bias = nac_bias if bias is None else (bias + nac_bias)

        # Pocket ← atom (Q from pocket, K/V from atom)
        Qp = self._split_heads(self.p_q(p_tokens))   # (B, H, K, d_head)
        Ka = self._split_heads(self.a_k(a_tokens))   # (B, H, A, d_head)
        Va = self._split_heads(self.a_v(a_tokens))
        logits_p = (Qp @ Ka.transpose(-1, -2)) / math.sqrt(self.d_head)  # (B, H, K, A)
        if bias is not None:
            logits_p = logits_p + bias
        # Mask: a_mask broadcast over K
        logits_p = logits_p.masked_fill(~a_mask.unsqueeze(1).unsqueeze(2),
                                        torch.finfo(logits_p.dtype).min)
        w_p = F.softmax(logits_p, dim=-1)
        p_att = w_p @ Va  # (B, H, K, d_head)
        p_att = self._combine_heads(p_att)
        p_new = self.ln_p(p_tokens + self.dropout(self.p_out(p_att)))

        # Atom ← pocket (symmetric)
        Qa = self._split_heads(self.a_q(a_tokens))
        Kp = self._split_heads(self.p_k(p_tokens))
        Vp = self._split_heads(self.p_v(p_tokens))
        logits_a = (Qa @ Kp.transpose(-1, -2)) / math.sqrt(self.d_head)
        if bias is not None:
            # transpose bias for atom→pocket direction
            logits_a = logits_a + bias.transpose(-1, -2)
        logits_a = logits_a.masked_fill(~p_mask.unsqueeze(1).unsqueeze(2),
                                        torch.finfo(logits_a.dtype).min)
        w_a = F.softmax(logits_a, dim=-1)
        a_att = w_a @ Vp
        a_att = self._combine_heads(a_att)
        a_new = self.ln_a(a_tokens + self.dropout(self.a_out(a_att)))

        return p_new, a_new


class Int3DCrossAttn(nn.Module):
    """Stack of bidirectional 3D cross-attention layers + pooled output."""
    def __init__(self, hidden_dim=128, n_heads=4, n_layers=2,
                 n_rbf=32, max_dist=20.0, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            Int3DCrossAttnLayer(hidden_dim, n_heads, n_rbf, max_dist, dropout)
            for _ in range(n_layers)
        ])
        self.pool_p = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.pool_a = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def _attn_pool(self, tokens, mask, pool_head):
        scores = pool_head(tokens)
        scores = scores.masked_fill(~mask.unsqueeze(-1),
                                    torch.finfo(scores.dtype).min)
        has_valid = mask.any(dim=1, keepdim=True).unsqueeze(-1).float()
        weights = F.softmax(scores, dim=1) * has_valid
        return (tokens * weights).sum(dim=1)

    def forward(self, p_tokens, a_tokens, p_mask, a_mask,
                xyz_p=None, xyz_a=None,
                xyz_valid_per_sample=None,
                p_nac=None, a_nac=None):
        p, a = p_tokens, a_tokens
        for layer in self.layers:
            p, a = layer(p, a, p_mask, a_mask, xyz_p, xyz_a,
                         xyz_valid_per_sample=xyz_valid_per_sample,
                         p_nac=p_nac, a_nac=a_nac)
        p_pool = self._attn_pool(p, p_mask, self.pool_p)  # (B, D)
        a_pool = self._attn_pool(a, a_mask, self.pool_a)  # (B, D)
        return p, a, p_pool, a_pool
