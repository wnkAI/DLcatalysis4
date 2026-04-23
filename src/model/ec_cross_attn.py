"""
EnzymeCompoundCrossAttention + interaction_weight adapted from EnzymeCAGE
(Liu et al., Nature Catalysis 2026, arXiv 2412.09621; MIT licensed).

We directly port the two cross-attention primitives and the biology-prior
interaction weight (reacting-center × pocket-geometric-center outer product),
then wire them into our 4.0 regression model.

Difference from original:
  1. Regression (not classification): no sigmoid readout at the end.
  2. Enzyme pocket nodes use ProtT5-per-residue embedding concatenated with
     GVP geometric features (original paper used ESM2).
  3. Optional product arm — on by default if product_smiles available, else
     skipped without retraining.
"""
import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """Single-direction cross-attention with optional additive attn_bias.
    Directly adapted from EnzymeCAGE/enzymecage/interaction.py.
    """

    def __init__(self, query_input_dim: int, key_input_dim: int, output_dim: int):
        super().__init__()
        self.out_dim = output_dim
        self.W_Q = nn.Linear(query_input_dim, output_dim)
        self.W_K = nn.Linear(key_input_dim, output_dim)
        self.W_V = nn.Linear(key_input_dim, output_dim)
        self.scale_val = output_dim ** 0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query_input, key_input, value_input,
                query_input_mask, key_input_mask, attn_bias=None):
        """
        query_input: (B, Nq, Dq)
        key_input:   (B, Nk, Dk)
        *_mask:      (B, N*) bool
        attn_bias:   (B, Nq, Nk) additive prior (interaction_weight)
        """
        q = self.W_Q(query_input)
        k = self.W_K(key_input)
        v = self.W_V(value_input)

        logits = torch.matmul(q, k.transpose(1, 2)) / self.scale_val
        if attn_bias is not None:
            logits = logits + attn_bias

        attn_mask = query_input_mask.unsqueeze(-1) * key_input_mask.unsqueeze(-1).transpose(1, 2)
        logits = logits.masked_fill(attn_mask == 0, -1e9)
        weights = self.softmax(logits)
        return torch.matmul(weights, v), weights


class EnzymeCompoundCrossAttention(nn.Module):
    """Bidirectional enzyme↔substrate and (optional) enzyme↔product cross-attn.
    Returns concatenated pooled output across 2 or 4 directions.
    """

    def __init__(self, enz_node_dim: int, cpd_node_dim: int, output_dim: int,
                 use_prods_info: bool = True):
        super().__init__()
        self.cross_enz_sub = CrossAttention(enz_node_dim, cpd_node_dim, output_dim)
        self.cross_sub_enz = CrossAttention(cpd_node_dim, enz_node_dim, output_dim)
        self.use_prods_info = use_prods_info
        if use_prods_info:
            self.cross_enz_prod = CrossAttention(enz_node_dim, cpd_node_dim, output_dim)
            self.cross_prod_enz = CrossAttention(cpd_node_dim, enz_node_dim, output_dim)

    def forward(self, enz, sub, prod,
                enz_mask, sub_mask, prod_mask=None,
                interaction_weight=None, return_weights=False):
        """
        interaction_weight: (B, N_sub, N_enz). We use its transpose as attn_bias
        when enzyme is the query (bias shape (B, N_enz, N_sub)).
        """
        if interaction_weight is not None:
            sub_enz_bias = interaction_weight
            enz_sub_bias = interaction_weight.transpose(1, 2)
        else:
            sub_enz_bias = None
            enz_sub_bias = None

        enz_sub_out, enz_sub_w = self.cross_enz_sub(
            enz, sub, sub, enz_mask, sub_mask, attn_bias=enz_sub_bias)
        sub_enz_out, _ = self.cross_sub_enz(
            sub, enz, enz, sub_mask, enz_mask, attn_bias=sub_enz_bias)

        pooled = [enz_sub_out.mean(1), sub_enz_out.mean(1)]

        if self.use_prods_info and prod is not None and prod_mask is not None:
            enz_prod_out, _ = self.cross_enz_prod(
                enz, prod, prod, enz_mask, prod_mask, attn_bias=None)
            prod_enz_out, _ = self.cross_prod_enz(
                prod, enz, enz, prod_mask, enz_mask, attn_bias=None)
            pooled += [enz_prod_out.mean(1), prod_enz_out.mean(1)]

        cat = torch.cat(pooled, dim=-1)  # (B, 2D or 4D)
        if return_weights:
            return cat, enz_sub_w
        return cat


# ──────────────────────────────────────────────────────────────────────
# EnzymeCAGE's biology prior: interaction_weight
# ──────────────────────────────────────────────────────────────────────
def calculate_pocket_weights(pocket_xyz: torch.Tensor, pocket_mask: torch.Tensor,
                             floor: float = 0.15, ceiling: float = 0.20) -> torch.Tensor:
    """Pocket residues closer to the pocket's geometric center get higher weight.
    Returns (B, K) float in approximately [floor, ceiling] (matches EnzymeCAGE's / 5).
    """
    valid_counts = pocket_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
    center = (pocket_xyz * pocket_mask.unsqueeze(-1)).sum(dim=1) / valid_counts  # (B, 3)
    dists = torch.norm(pocket_xyz - center.unsqueeze(1), dim=2)  # (B, K)
    dmax, _ = dists.max(dim=1, keepdim=True)
    dmin, _ = dists.min(dim=1, keepdim=True)
    rng = (dmax - dmin).clamp(min=1e-6)
    norm_d = (dists - dmin) / rng
    # EnzymeCAGE form: (1 - normalized) / 5  → roughly [0, 0.2]
    w = (1.0 - norm_d) / 5.0
    return w * pocket_mask.float()


def calc_interaction_weight(pocket_xyz: torch.Tensor,
                            pocket_mask: torch.Tensor,
                            substrate_reacting_center: torch.Tensor,
                            substrate_mask: torch.Tensor,
                            rxn_weight_high: float = 0.5,
                            rxn_weight_base: float = 0.1) -> torch.Tensor:
    """Outer product of substrate reacting-center weight and pocket
    geometric-center weight. Shape (B, N_sub, K_pocket).

    substrate_reacting_center: (B, N_sub) 0/1 (1 = reaction-center atom)
    """
    pocket_w = calculate_pocket_weights(pocket_xyz, pocket_mask)  # (B, K)
    sub_w = (substrate_reacting_center.float() * rxn_weight_high + rxn_weight_base) \
        * substrate_mask.float()                                  # (B, N_sub)
    # (B, N_sub, K)
    return torch.einsum('bi,bj->bij', sub_w, pocket_w)
