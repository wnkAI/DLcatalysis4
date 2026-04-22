"""
Substrate GNN encoder using GINE (GIN with Edge features).
Encodes molecular graphs into atom-level tokens for cross-attention with enzyme residues.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.utils import to_dense_batch


class AtomEncoder(nn.Module):
    """Encode atom features into hidden_dim via learned embeddings."""

    def __init__(self, hidden_dim):
        super().__init__()
        # Atom type (atomic number): 0-119
        self.atom_type_emb = nn.Embedding(120, 64)
        # Degree: 0-6
        self.degree_emb = nn.Embedding(7, 16)
        # Formal charge: -3 to +3 (offset by 3 -> 0-6)
        self.charge_emb = nn.Embedding(7, 16)
        # Hybridization: sp, sp2, sp3, sp3d, sp3d2
        self.hybrid_emb = nn.Embedding(5, 16)
        # Num H: 0-5
        self.num_h_emb = nn.Embedding(6, 8)
        # Chirality: unspecified, R, S, other
        self.chirality_emb = nn.Embedding(4, 8)
        # Boolean features: aromatic, in_ring, donor, acceptor = 4 dims
        # Total raw dim: 64 + 16 + 16 + 16 + 8 + 8 + 4 = 132
        self.proj = nn.Linear(132, hidden_dim)

    def forward(self, x):
        """
        x: (N, 10) long tensor
           [atomic_num, degree, charge+3, hybrid, num_h, chirality, aromatic, in_ring, donor, acceptor]
        """
        embs = torch.cat([
            self.atom_type_emb(x[:, 0]),       # 64
            self.degree_emb(x[:, 1]),           # 16
            self.charge_emb(x[:, 2]),           # 16
            self.hybrid_emb(x[:, 3]),           # 16
            self.num_h_emb(x[:, 4]),            # 8
            self.chirality_emb(x[:, 5]),        # 8
            x[:, 6:10].float(),                 # 4 (aromatic, in_ring, donor, acceptor)
        ], dim=-1)                              # total: 132
        return self.proj(embs)                  # (N, hidden_dim)


class BondEncoder(nn.Module):
    """Encode bond features into hidden_dim."""

    def __init__(self, hidden_dim):
        super().__init__()
        # Bond type: single, double, triple, aromatic, other
        self.bond_type_emb = nn.Embedding(5, 16)
        # Stereo: none, Z, E, other
        self.stereo_emb = nn.Embedding(4, 8)
        # Boolean: conjugated, in_ring = 2 dims
        # Total raw: 16 + 8 + 2 = 26
        self.proj = nn.Linear(26, hidden_dim)

    def forward(self, edge_attr):
        """
        edge_attr: (E, 4) long tensor
                   [bond_type, stereo, conjugated, in_ring]
        """
        embs = torch.cat([
            self.bond_type_emb(edge_attr[:, 0]),  # 16
            self.stereo_emb(edge_attr[:, 1]),      # 8
            edge_attr[:, 2:4].float(),             # 2
        ], dim=-1)                                 # total: 26
        return self.proj(embs)                     # (E, hidden_dim)


class SubstrateGINE(nn.Module):
    """
    4-layer GINE encoder with JumpingKnowledge for substrate molecular graphs.

    Outputs:
        atom_tokens: (B, max_atoms, hidden_dim) - for cross-attention with enzyme
        graph_emb:   (B, hidden_dim) - global graph embedding
        atom_mask:   (B, max_atoms) - True for valid atoms, False for padding
    """

    def __init__(self, hidden_dim=256, n_layers=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)

        # GINE layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # JumpingKnowledge: learned weighted sum of all layer outputs
        self.jk_weights = nn.Parameter(torch.ones(n_layers) / n_layers)

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Args:
            x:          (N_total, 10) atom features
            edge_index: (2, E_total) bond indices
            edge_attr:  (E_total, 4) bond features
            batch:      (N_total,) batch assignment

        Returns:
            atom_tokens: (B, max_atoms, hidden_dim) padded atom embeddings
            graph_emb:   (B, hidden_dim) mean-pooled graph embedding
            atom_mask:   (B, max_atoms) bool mask, True = valid atom
        """
        # Encode raw features
        h = self.atom_encoder(x)
        e = self.bond_encoder(edge_attr)

        # Message passing with JumpingKnowledge
        layer_outputs = []
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index, e)
            h = bn(h)
            h = torch.relu(h)
            h = self.dropout(h)
            layer_outputs.append(h)

        # JK weighted sum: learned combination of all layer outputs
        weights = torch.softmax(self.jk_weights, dim=0)
        h_final = sum(w * h_l for w, h_l in zip(weights, layer_outputs))  # (N_total, hidden_dim)

        # Graph-level embedding
        graph_emb = global_mean_pool(h_final, batch)  # (B, hidden_dim)

        # Convert to dense padded batch for cross-attention
        atom_tokens, atom_mask = to_dense_batch(h_final, batch)  # (B, max_atoms, hidden_dim), (B, max_atoms)

        return atom_tokens, graph_emb, atom_mask
