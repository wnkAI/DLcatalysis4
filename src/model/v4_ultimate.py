"""
DLcatalysis 4.0 — v4-ultimate (集大成版).

    pred = y_seq + g_pair * (y_sub + y_rxn + y_int3d) + g_struct * y_struct + g_annot * y_annot

    Branches:
      y_seq    : ProtT5 (frozen) -> seq_mlp -> attn pool -> MLP       (enzyme baseline)
      y_sub    : GINE on substrate SMILES + RXNMapper center flag     (substrate effect)
      y_rxn    : DRFP reaction fingerprint (2048d) -> MLP             (reaction delta, TurNuP-style)
      y_int3d  : pocket residue × substrate atom cross-attn + 3D RBF  (interaction)
      y_struct : GVP-GNN pocket encoder -> attn pool -> MLP           (catalytic pocket)
      y_annot  : InterPro family + Pfam family + GO terms embedding   (functional prior)

    g_pair  : sigmoid over (enz_pool, graph_emb, rxn_emb?, ec?)
              NOTE: annot_emb is NOT fed into g_pair — the annotation branch
              has its own dedicated g_annot (see below). Mixing annot into
              g_pair would couple the pair-specific residual strength to
              enzyme annotation availability.
    g_struct: sigmoid over (enz_pool, graph_emb, pocket_pool, quality[3], ec?)
              quality = [n_res/K, has_active_site∨has_binding_site, has_cofactor]
              piped from dataloader ANNOT_has_* fields (commit 2a0f337).
              init bias = -2.0 (conservative).
    g_annot : sigmoid over (enz_pool, annot_emb). Separate gate for the
              annotation branch so the pair/structure branches stay clean.

Optional pH/temperature conditioning: scalar features concatenated into
enz_pool branch (EF-UniKP style).

All branches are optional via config flags — model degrades gracefully to
v4-minimal / v4-pocket depending on what modalities are enabled.
"""
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
from model.module import MLP
from model.substrate_gnn import SubstrateGINE
from model.gvp_pocket import PocketGVP
from model.int3d_cross_attn import Int3DCrossAttn
from util.featurize.seq_prot5 import prot5_embedding


class V4Ultimate(pl.LightningModule):
    """6-branch kcat/Km predictor combining sequence, substrate, reaction,
    3D pocket, cross-attention, and annotation priors."""

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters({"config": config})
        self.config       = config
        self.hidden_dim   = config["model"]["hidden_dim"]
        self.dropout_rate = config["model"].get("dropout", 0.2)
        self.automatic_optimization = True
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.train_preds, self.train_targets = [], []
        self.val_preds, self.val_targets     = [], []
        self.test_preds, self.test_targets   = [], []
        self._last_test_metrics = None

        self.log_dir = config["train"]["log_path"]
        os.makedirs(self.log_dir, exist_ok=True)
        model_name = config["model"].get("model_name", "v4_ultimate")
        self.results_file = os.path.join(self.log_dir, f"{model_name}_{self.run_id}.csv")

        # Feature flags
        self.use_rxn_drfp   = config["model"].get("use_rxn_drfp", True)
        self.use_rxn_center = config["model"].get("use_rxn_center", True)
        self.use_pocket     = config["model"].get("use_pocket", True)
        self.use_int3d      = config["model"].get("use_int3d", True)
        self.use_annot      = config["model"].get("use_annot", True)
        self.use_condition  = config["model"].get("use_condition", True)  # pH, temp
        self.use_ec         = config["model"].get("use_ec", True)
        # Heteroscedastic aleatoric uncertainty: predict mu + log_var per sample.
        # Loss becomes 0.5 * exp(-log_var) * (pred-y)^2 + 0.5 * log_var (Kendall & Gal 2017)
        self.use_uncertainty = config["model"].get("use_uncertainty", False)

        # ── ProtT5 (frozen, precomputed) ───────────────────────────────
        init_device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")
        enc_cfg = dict(config["model"])
        enc_cfg["precomputed_only"] = True
        self.prot5_encoder = prot5_embedding(device=init_device, config=enc_cfg)
        for p in self.prot5_encoder.parameters():
            p.requires_grad = False

        seq_cfg = config["model"]["seq_module"]
        self.seq_mlp = nn.Sequential(
            nn.Linear(self.prot5_encoder.embedding_dim, seq_cfg["seq_hidden_dim"]),
            nn.LayerNorm(seq_cfg["seq_hidden_dim"]),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(seq_cfg["seq_hidden_dim"], self.hidden_dim),
        )
        self.seq_attn_pool = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

        # ── Substrate GINE (optionally adds RXNMapper center flag as extra atom feat) ──
        gnn_cfg = config["model"].get("substrate_gnn", {})
        self.substrate_gnn = SubstrateGINE(
            hidden_dim=self.hidden_dim,
            n_layers=gnn_cfg.get("n_layers", 4),
            dropout=gnn_cfg.get("dropout", 0.1),
        )
        # RXNMapper center flag: if enabled, add a projection + add to GINE atom embeds
        if self.use_rxn_center:
            # Input: 1-dim boolean mask → small embedding added to atom tokens post-GINE
            self.rxn_center_emb = nn.Embedding(2, self.hidden_dim)

        # ── DRFP reaction fingerprint branch ──────────────────────────
        if self.use_rxn_drfp:
            drfp_dim = config["model"].get("drfp_dim", 2048)
            self.rxn_proj = nn.Sequential(
                nn.Linear(drfp_dim, 2 * self.hidden_dim),
                nn.LayerNorm(2 * self.hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            )

        # ── GVP pocket encoder ───────────────────────────────────────
        if self.use_pocket:
            struct_cfg = config["model"].get("struct", {})
            self.pocket_gvp = PocketGVP(
                in_s_dim=struct_cfg.get("in_s_dim", 26),
                in_v_dim=struct_cfg.get("in_v_dim", 2),
                hidden_dim=self.hidden_dim,
                v_hidden=struct_cfg.get("v_hidden", 16),
                n_layers=struct_cfg.get("gvp_layers", 3),
                edge_s_dim=struct_cfg.get("edge_s_dim", 16),
                edge_v_dim=struct_cfg.get("edge_v_dim", 1),
                dropout=self.dropout_rate,
            )

        # ── 3D cross-attention (pocket × substrate atom) ─────────────
        if self.use_int3d:
            int3d_cfg = config["model"].get("int3d", {})
            self.int3d = Int3DCrossAttn(
                hidden_dim=self.hidden_dim,
                n_heads=int3d_cfg.get("n_head", 4),
                n_layers=int3d_cfg.get("n_layers", 2),
                n_rbf=int3d_cfg.get("rbf_bins", 32),
                max_dist=int3d_cfg.get("max_dist", 20.0),
                dropout=self.dropout_rate,
            )

        # ── Annotation embedding (InterPro family + Pfam + GO) ───────
        if self.use_annot:
            vocab = config["model"].get("annot_vocab", {})
            ipr_n = vocab.get("interpro_family_vocab", 6000)
            pf_n  = vocab.get("pfam_family_vocab", 3000)
            go_n  = vocab.get("go_term_vocab", 2000)
            embed_dim = vocab.get("embed_dim", 32)
            # padding_idx=0 reserved for "unknown"
            self.ipr_emb = nn.Embedding(ipr_n + 1, embed_dim, padding_idx=0)
            self.pf_emb  = nn.Embedding(pf_n  + 1, embed_dim, padding_idx=0)
            self.go_emb  = nn.Embedding(go_n  + 1, embed_dim, padding_idx=0)
            self.annot_proj = nn.Sequential(
                nn.Linear(3 * embed_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout_rate),
            )

        # ── Condition-aware (pH, temp) ──────────────────────────────
        if self.use_condition:
            self.cond_proj = nn.Sequential(
                nn.Linear(4, self.hidden_dim),   # [pH, pH_missing, temp, temp_missing]
                nn.LayerNorm(self.hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout_rate),
            )

        # ── EC embedding ────────────────────────────────────────────
        if self.use_ec:
            ec_cfg = config["model"].get("ec", {})
            ec_dim = ec_cfg.get("ec_embed_dim", 16)
            self.ec1_emb = nn.Embedding(ec_cfg.get("ec1_vocab", 8), ec_dim, padding_idx=0)
            self.ec2_emb = nn.Embedding(ec_cfg.get("ec2_vocab", 200), ec_dim, padding_idx=0)
            self.ec3_emb = nn.Embedding(ec_cfg.get("ec3_vocab", 200), ec_dim, padding_idx=0)
            self.ec4_emb = nn.Embedding(ec_cfg.get("ec4_vocab", 1200), ec_dim, padding_idx=0)
            self.ec_proj = nn.Sequential(
                nn.Linear(4 * ec_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout_rate),
            )

        # ── Heads ───────────────────────────────────────────────────
        hdr = config["model"]["output_header"]
        def _mkhead(in_dim):
            return MLP(in_dim=in_dim, out_dim=1,
                       hidden_dim=hdr["hidden_dim"], num_layer=hdr["num_layers"],
                       norm=hdr["norm_fn"], act_fn=hdr["act_fn"], dropout=self.dropout_rate)
        self.head_seq    = _mkhead(self.hidden_dim)
        self.head_sub    = _mkhead(self.hidden_dim)
        if self.use_rxn_drfp:
            self.head_rxn    = _mkhead(self.hidden_dim)
        if self.use_int3d:
            self.head_int3d  = _mkhead(self.hidden_dim * 2)
        if self.use_pocket:
            self.head_struct = _mkhead(self.hidden_dim)
        if self.use_annot:
            self.head_annot  = _mkhead(self.hidden_dim)
        # Aleatoric uncertainty head: log-variance from fused enzyme pool
        if self.use_uncertainty:
            self.head_logvar = _mkhead(self.hidden_dim)
            self._last_log_var = None  # stash for loss access

        # ── Gates ──────────────────────────────────────────────────
        pair_in = self.hidden_dim * 2   # enz_pool + graph_emb
        if self.use_rxn_drfp:
            pair_in += self.hidden_dim
        if self.use_ec:
            pair_in += self.hidden_dim
        self.gate_pair = nn.Sequential(
            nn.Linear(pair_in, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )

        if self.use_pocket:
            struct_in = self.hidden_dim * 3 + 3   # enz + graph + pocket + (n_res, has_annot, has_cof)
            if self.use_ec:
                struct_in += self.hidden_dim
            self.gate_struct = nn.Sequential(
                nn.Linear(struct_in, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim, 1),
            )
            # Read gate bias from config for consistency with v4_pocket
            self._gate_struct_bias = float(
                config["model"].get("struct", {}).get("gate_init_bias", -2.0)
            )
            nn.init.constant_(self.gate_struct[-1].bias, self._gate_struct_bias)

        if self.use_annot:
            # Gate annot branch by how "known" the enzyme is
            annot_gate_in = self.hidden_dim * 2   # enz_pool + annot_emb
            self.gate_annot = nn.Sequential(
                nn.Linear(annot_gate_in, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim, 1),
            )
            nn.init.constant_(self.gate_annot[-1].bias, -1.0)  # start modest

        self._init_weights()
        if self.use_pocket:
            nn.init.constant_(self.gate_struct[-1].bias, self._gate_struct_bias)
        if self.use_annot:
            nn.init.constant_(self.gate_annot[-1].bias, -1.0)

    def _init_weights(self):
        prot5_ids = set(id(m) for m in self.prot5_encoder.modules())
        for m in self.modules():
            if isinstance(m, nn.Linear) and id(m) not in prot5_ids:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ──────────────────────────────────────────────────────────────
    # Encoders
    # ──────────────────────────────────────────────────────────────
    def _attn_pool_seq(self, feat, mask, pool_head):
        scores = pool_head(feat)
        scores = scores.masked_fill(mask < 0.5, torch.finfo(scores.dtype).min)
        has_valid = (mask > 0.5).any(dim=1, keepdim=True)
        weights = torch.softmax(scores, dim=1) * has_valid.float()
        return (feat * weights).sum(dim=1)

    def _encode_enzyme(self, G, B):
        if not hasattr(G, "SEQ_seq_padding_mask"):
            z = torch.zeros(B, self.hidden_dim, device=self.device)
            return None, z, None
        max_len = G.SEQ_seq_padding_mask.shape[1]
        seq_mask = (~G.SEQ_seq_padding_mask).unsqueeze(-1).float().to(self.device)
        x = G.SEQ_embedding.view(B, max_len, -1).to(self.device).float()
        feat = self.seq_mlp(x)
        pooled = self._attn_pool_seq(feat, seq_mask, self.seq_attn_pool)
        return feat, pooled, seq_mask

    def _encode_substrate(self, G, B):
        if not hasattr(G, "MOL_graph_x") or G.MOL_graph_x is None:
            z = torch.zeros(B, self.hidden_dim, device=self.device)
            return None, z, None, None
        num_nodes = G.MOL_graph_num_nodes.to(self.device).view(-1)
        graph_batch = torch.repeat_interleave(
            torch.arange(num_nodes.size(0), device=self.device), num_nodes
        )
        atom_tokens, graph_emb, atom_mask = self.substrate_gnn(
            x=G.MOL_graph_x.to(self.device),
            edge_index=G.MOL_graph_edge_index.to(self.device),
            edge_attr=G.MOL_graph_edge_attr.to(self.device),
            batch=graph_batch,
        )
        # Reaction-center dense flag, kept around for the NAC bias in int3d
        # even when `use_rxn_center` (the atom-token injection) is disabled.
        rxn_center_dense = None
        if hasattr(G, "MOL_rxn_center") and atom_mask is not None:
            from torch_geometric.utils import to_dense_batch
            center_flag_long = G.MOL_rxn_center.to(self.device).long()
            rxn_center_dense, _ = to_dense_batch(center_flag_long.float(),
                                                 graph_batch)            # (B, A)
        # Add RXNMapper center flag to atom tokens (if enabled)
        if self.use_rxn_center and hasattr(G, "MOL_rxn_center"):
            from torch_geometric.utils import to_dense_batch
            center_flag = G.MOL_rxn_center.to(self.device).long()
            center_dense, _ = to_dense_batch(center_flag, graph_batch)   # (B, A)
            center_emb = self.rxn_center_emb(center_dense)               # (B, A, D)
            m = atom_mask.unsqueeze(-1).float()                          # (B, A, 1)
            # Atom-level: add only where atom is valid
            atom_tokens = atom_tokens + center_emb * m
            # Graph-level: mask-aware mean over REAL atoms only (padding
            # rxn_center_emb(0) is non-zero and would bias by ligand size)
            masked_sum = (center_emb * m).sum(dim=1)                     # (B, D)
            n_atoms = m.sum(dim=1).clamp(min=1.0)                        # (B, 1)
            graph_emb = graph_emb + masked_sum / n_atoms
        return atom_tokens, graph_emb, atom_mask, rxn_center_dense

    def _encode_rxn(self, G, B):
        if not self.use_rxn_drfp or not hasattr(G, "RXN_drfp"):
            return torch.zeros(B, self.hidden_dim, device=self.device)
        drfp = G.RXN_drfp.to(self.device).float().view(B, -1)
        return self.rxn_proj(drfp)

    # Catalytic-residue prior used as a soft NAC bias in int3d.
    # Indices correspond to STANDARD_AA order in scripts/14_extract_pockets.py:
    # 1 ARG, 3 ASP, 4 CYS, 6 GLU, 8 HIS, 11 LYS, 15 SER, 18 TYR. These eight
    # residues account for the vast majority of catalytic activity in MCSA
    # (acid/base, nucleophile, metal-coordinating side chains).
    _CATALYTIC_AA_IDX = (1, 3, 4, 6, 8, 11, 15, 18)

    def _encode_pocket(self, G, B):
        if not self.use_pocket or not hasattr(G, "POCKET_node_s"):
            return (None,
                    torch.zeros(B, self.hidden_dim, device=self.device),
                    None, None, None)
        node_s = G.POCKET_node_s.to(self.device).float()
        node_v = G.POCKET_node_v.to(self.device).float()
        edge_index = G.POCKET_edge_index.to(self.device).long()
        edge_s = G.POCKET_edge_s.to(self.device).float()
        edge_v = G.POCKET_edge_v.to(self.device).float()
        num_nodes = G.POCKET_num_nodes.to(self.device).view(-1)
        batch = torch.repeat_interleave(
            torch.arange(num_nodes.size(0), device=self.device), num_nodes
        )
        p_tokens, p_pool, p_mask = self.pocket_gvp(
            node_s, node_v, edge_index, edge_s, edge_v, batch
        )
        ca_xyz = G.POCKET_ca_xyz.to(self.device).float()
        from torch_geometric.utils import to_dense_batch
        pocket_xyz, _ = to_dense_batch(ca_xyz, batch)

        # Catalytic-residue soft indicator for NAC bias. UniProt annotation
        # (is_active_site / is_binding_site) is treated as strong evidence;
        # the amino-acid-type prior is a weaker (0.5) fallback when the
        # enzyme has no catalytic-site annotation.
        is_annot = (node_s[:, 21] + node_s[:, 22]).clamp(max=1.0)  # (N_total,)
        cat_idx = torch.tensor(self._CATALYTIC_AA_IDX,
                               device=self.device, dtype=torch.long)
        is_aa = node_s[:, cat_idx].sum(dim=-1).clamp(max=1.0)     # (N_total,)
        cat_score = torch.maximum(is_annot, 0.5 * is_aa)
        cat_dense, _ = to_dense_batch(cat_score, batch)           # (B, K)

        return p_tokens, p_pool, p_mask, pocket_xyz, cat_dense

    def _encode_annot(self, G, B):
        if not self.use_annot or not hasattr(G, "ANNOT_ipr_ids"):
            return torch.zeros(B, self.hidden_dim, device=self.device)
        # IDs already padded to fixed max length with padding_idx=0 upstream
        ipr_ids = G.ANNOT_ipr_ids.to(self.device).view(B, -1)  # (B, max_fam)
        pf_ids  = G.ANNOT_pf_ids.to(self.device).view(B, -1)
        go_ids  = G.ANNOT_go_ids.to(self.device).view(B, -1)
        # Mean pool over non-zero entries
        def _bag_pool(ids, emb_layer):
            e = emb_layer(ids)                     # (B, L, D)
            m = (ids > 0).float().unsqueeze(-1)    # (B, L, 1)
            s = (e * m).sum(dim=1)
            n = m.sum(dim=1).clamp(min=1.0)
            return s / n
        ipr_pool = _bag_pool(ipr_ids, self.ipr_emb)
        pf_pool  = _bag_pool(pf_ids,  self.pf_emb)
        go_pool  = _bag_pool(go_ids,  self.go_emb)
        cat = torch.cat([ipr_pool, pf_pool, go_pool], dim=-1)
        return self.annot_proj(cat)

    def _encode_condition(self, G, B):
        if not self.use_condition:
            return torch.zeros(B, self.hidden_dim, device=self.device)
        ph = G.COND_ph.to(self.device).float().view(B, 1) if hasattr(G, "COND_ph") else torch.full((B, 1), 7.0, device=self.device)
        ph_missing = (torch.isnan(ph)).float()
        ph = torch.where(torch.isnan(ph), torch.full_like(ph, 7.0), ph) / 14.0
        temp = G.COND_temp.to(self.device).float().view(B, 1) if hasattr(G, "COND_temp") else torch.full((B, 1), 25.0, device=self.device)
        temp_missing = (torch.isnan(temp)).float()
        temp = torch.where(torch.isnan(temp), torch.full_like(temp, 25.0), temp) / 100.0
        cond = torch.cat([ph, ph_missing, temp, temp_missing], dim=-1)
        return self.cond_proj(cond)

    def _encode_ec(self, G, B):
        if not (self.use_ec and hasattr(G, "EC_ids")):
            return torch.zeros(B, self.hidden_dim, device=self.device)
        ec = G.EC_ids.to(self.device)
        if ec.dim() == 1:
            ec = ec.view(B, 4)
        cat = torch.cat([
            self.ec1_emb(ec[:, 0]), self.ec2_emb(ec[:, 1]),
            self.ec3_emb(ec[:, 2]), self.ec4_emb(ec[:, 3]),
        ], dim=-1)
        return self.ec_proj(cat)

    # ──────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────
    def forward(self, G):
        B = G.num_graphs if hasattr(G, "num_graphs") else 1
        if hasattr(G, "y") and G.y.device != self.device:
            G.y = G.y.to(self.device)

        # Encode all modalities
        _, enz_pool, _ = self._encode_enzyme(G, B)
        atom_tokens, graph_emb, atom_mask, rxn_center_dense = \
            self._encode_substrate(G, B)
        rxn_emb = self._encode_rxn(G, B)
        pocket_tokens, pocket_pool, pocket_mask, pocket_xyz, pocket_cat = \
            self._encode_pocket(G, B)
        annot_emb = self._encode_annot(G, B)
        cond_emb = self._encode_condition(G, B)
        ec_emb = self._encode_ec(G, B)

        # Fuse enzyme: baseline enzyme embedding mixes seq + condition (+ maybe more)
        enz_fused = enz_pool + cond_emb

        # ── Modality dropout ──
        # During training, drop each auxiliary modality with a config-level
        # probability per sample. The same survival mask is applied to (a) the
        # pooled embedding used by gates and (b) the final y_* contribution,
        # so "dropped" truly means the branch contributes zero for that
        # sample. Inference always keeps all modalities — dropout only
        # regularizes training. This lets us ablate modalities at test time
        # (e.g. inference without pocket) without retraining from scratch.
        md = self.config["model"].get("modality_dropout", {}) or {}
        def _keep(p):
            if (not self.training) or p <= 0.0:
                return torch.ones(B, 1, device=self.device)
            return (torch.rand(B, 1, device=self.device) >= float(p)).float()
        keep_rxn    = _keep(md.get("rxn",    0.0))
        keep_pocket = _keep(md.get("pocket", 0.0))
        keep_annot  = _keep(md.get("annot",  0.0))
        rxn_emb    = rxn_emb    * keep_rxn
        pocket_pool = pocket_pool * keep_pocket
        annot_emb  = annot_emb  * keep_annot

        # ── Branches ──
        y_seq = self.head_seq(enz_fused)
        y_sub = self.head_sub(graph_emb)
        y_rxn = self.head_rxn(rxn_emb) if self.use_rxn_drfp else torch.zeros(B, 1, device=self.device)

        # int3d
        if self.use_int3d and pocket_tokens is not None and atom_tokens is not None and pocket_mask is not None:
            from torch_geometric.utils import to_dense_batch
            atom_xyz = None
            xyz_valid_per_sample = None
            if hasattr(G, "MOL_graph_xyz") and hasattr(G, "MOL_graph_xyz_valid") \
                    and bool(G.MOL_graph_xyz_valid.any().item()):
                num_nodes = G.MOL_graph_num_nodes.to(self.device).view(-1)
                atom_batch = torch.repeat_interleave(
                    torch.arange(num_nodes.size(0), device=self.device), num_nodes
                )
                atom_xyz, _ = to_dense_batch(G.MOL_graph_xyz.to(self.device).float(), atom_batch)
                xyz_valid_per_sample = G.MOL_graph_xyz_valid.view(-1).to(self.device)
            _, _, p_pool_int, a_pool_int = self.int3d(
                pocket_tokens, atom_tokens, pocket_mask, atom_mask,
                xyz_p=pocket_xyz, xyz_a=atom_xyz,
                xyz_valid_per_sample=xyz_valid_per_sample,
                p_nac=pocket_cat, a_nac=rxn_center_dense,
            )
            y_int3d = self.head_int3d(torch.cat([p_pool_int, a_pool_int], dim=-1))
        else:
            y_int3d = torch.zeros(B, 1, device=self.device)

        y_struct = self.head_struct(pocket_pool) if self.use_pocket and pocket_tokens is not None else torch.zeros(B, 1, device=self.device)
        y_annot  = self.head_annot(annot_emb)  if self.use_annot  else torch.zeros(B, 1, device=self.device)

        # Modality dropout at output level. Heads on zeroed embeddings still
        # emit the head bias term, so we explicitly zero the y_* contribution
        # for dropped samples. keep_* is all-ones during inference (see
        # _keep), so eval metrics are unaffected.
        y_rxn    = y_rxn    * keep_rxn
        y_struct = y_struct * keep_pocket
        y_int3d  = y_int3d  * keep_pocket
        y_annot  = y_annot  * keep_annot

        # ── Gates ──
        pair_gate_in = [enz_fused, graph_emb]
        if self.use_rxn_drfp:
            pair_gate_in.append(rxn_emb)
        if self.use_ec:
            pair_gate_in.append(ec_emb)
        g_pair = self.gate_pair(torch.cat(pair_gate_in, dim=-1))

        if self.use_pocket:
            n_res = (pocket_mask.sum(dim=1, keepdim=True).float() / 32.0) \
                if pocket_mask is not None else torch.zeros(B, 1, device=self.device)
            # Real signals from dataloader (no longer stubbed zeros)
            has_annot_flag = (G.ANNOT_has_any.to(self.device).view(B, 1)
                              if hasattr(G, "ANNOT_has_any")
                              else torch.zeros(B, 1, device=self.device))
            has_cof = (G.ANNOT_has_cof.to(self.device).view(B, 1)
                       if hasattr(G, "ANNOT_has_cof")
                       else torch.zeros(B, 1, device=self.device))
            quality = torch.cat([n_res, has_annot_flag, has_cof], dim=-1)
            struct_in = [enz_fused, graph_emb, pocket_pool, quality]
            if self.use_ec:
                struct_in.append(ec_emb)
            g_struct = torch.sigmoid(self.gate_struct(torch.cat(struct_in, dim=-1)))
        else:
            g_struct = torch.zeros(B, 1, device=self.device)

        if self.use_annot:
            g_annot = torch.sigmoid(self.gate_annot(torch.cat([enz_fused, annot_emb], dim=-1)))
        else:
            g_annot = torch.zeros(B, 1, device=self.device)

        # ── Residual composition ──
        pair_delta = y_sub + y_rxn + y_int3d
        pred = (y_seq
                + g_pair * pair_delta
                + g_struct * y_struct
                + g_annot * y_annot)

        # Aleatoric uncertainty: log-variance output (stashed for get_loss)
        if self.use_uncertainty:
            # Clip raw log_var to reasonable range to avoid explosion
            raw_logvar = self.head_logvar(enz_fused)
            self._last_log_var = raw_logvar.clamp(min=-6.0, max=6.0)
        else:
            self._last_log_var = None

        # ── Diagnostics ──
        # Emits branch magnitudes, gate means, and modality coverage once a
        # trainer is attached. Guarded so the model still runs standalone
        # (unit tests / scripted inference) without a Lightning trainer.
        if getattr(self, "_trainer", None) is not None:
            self._log_diag(
                B=B, G=G,
                y_seq=y_seq, y_sub=y_sub, y_rxn=y_rxn,
                y_int3d=y_int3d, y_struct=y_struct, y_annot=y_annot,
                g_pair=g_pair, g_struct=g_struct, g_annot=g_annot,
                atom_mask=atom_mask, pocket_mask=pocket_mask,
            )

        return pred, G.y

    # ──────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────
    def _log_diag(self, *, B, G,
                  y_seq, y_sub, y_rxn, y_int3d, y_struct, y_annot,
                  g_pair, g_struct, g_annot,
                  atom_mask, pocket_mask):
        def _log(name, value):
            self.log(f"diag/{name}", value,
                     on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=B)

        # Branch magnitudes (mean |y_*|) — tells you which branch dominates pred
        _log("y_seq_abs",    y_seq.detach().abs().mean())
        _log("y_sub_abs",    y_sub.detach().abs().mean())
        _log("y_rxn_abs",    y_rxn.detach().abs().mean())
        _log("y_int3d_abs",  y_int3d.detach().abs().mean())
        _log("y_struct_abs", y_struct.detach().abs().mean())
        _log("y_annot_abs",  y_annot.detach().abs().mean())

        # Gate means — tells you how much each branch is actually mixed in
        _log("g_pair_mean",   g_pair.detach().mean())
        _log("g_struct_mean", g_struct.detach().mean())
        _log("g_annot_mean",  g_annot.detach().mean())

        # Modality coverage (fraction of the batch with real signal)
        has_rxn = float(hasattr(G, "RXN_drfp"))
        has_atoms = float(atom_mask.any().item()) if atom_mask is not None else 0.0
        has_pocket = float(pocket_mask.any().item()) if pocket_mask is not None else 0.0
        _log("cov_rxn_drfp", torch.tensor(has_rxn, device=self.device))
        _log("cov_atoms",    torch.tensor(has_atoms, device=self.device))
        _log("cov_pocket",   torch.tensor(has_pocket, device=self.device))

        if hasattr(G, "ANNOT_has_any"):
            _log("cov_annot", G.ANNOT_has_any.float().mean())
        if hasattr(G, "ANNOT_has_cof"):
            _log("cov_cof", G.ANNOT_has_cof.float().mean())
        if hasattr(G, "MOL_graph_xyz_valid"):
            _log("cov_mol_xyz", G.MOL_graph_xyz_valid.float().mean())

    # ──────────────────────────────────────────────────────────────
    # Loss / metrics / steps — identical machinery as v4_pocket
    # ──────────────────────────────────────────────────────────────
    def _prepare_target(self, y_true, stage):
        y = y_true.float()
        if stage == "train":
            noise = self.config["model"].get("label_noise", 0.0)
            if noise > 0:
                y = y + torch.randn_like(y) * noise
        return y

    def _pairwise_ranking(self, pred, target, ids, margin, min_diff=0.1, max_pairs=16):
        pred = pred.squeeze(-1)
        target = target.squeeze() if target.dim() > 1 else target
        loss = torch.tensor(0.0, device=self.device)
        n_pairs = n_correct = 0
        from collections import defaultdict
        groups = defaultdict(list)
        for i, gid in enumerate(ids):
            groups[gid].append(i)
        for _, idxs in groups.items():
            if len(idxs) < 2:
                continue
            pairs = []
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    if (target[idxs[i]] - target[idxs[j]]).abs() >= min_diff:
                        pairs.append((idxs[i], idxs[j]))
            if not pairs:
                continue
            if len(pairs) > max_pairs:
                sel = torch.randperm(len(pairs))[:max_pairs].tolist()
                pairs = [pairs[k] for k in sel]
            for a, b in pairs:
                diff_true = target[a] - target[b]
                diff_pred = pred[a] - pred[b]
                sign = torch.sign(diff_true)
                loss = loss + F.relu(margin - sign * diff_pred)
                n_pairs += 1
                if diff_pred.sign() == sign:
                    n_correct += 1
        return loss / max(n_pairs, 1), n_pairs, n_correct / max(n_pairs, 1)

    def get_loss(self, y_pred, y_true, stage, G=None):
        pred = y_pred.squeeze(-1).float()
        y_target = self._prepare_target(y_true, stage).squeeze(-1).float()

        loss_type = self.config["model"].get("loss_type", "logcosh")
        # Heteroscedastic aleatoric loss (Kendall & Gal 2017) weights the
        # per-sample error by the predicted variance. The weighted error is
        # built on top of the configured `loss_type` (logcosh / huber / mse)
        # so `loss_type` is respected whether uncertainty is on or off.
        if self.use_uncertainty and self._last_log_var is not None:
            log_var = self._last_log_var.squeeze(-1).float()
            if loss_type == "huber":
                delta = self.config["model"].get("huber_delta", 1.0)
                per_sample = F.huber_loss(pred, y_target, delta=delta, reduction="none")
            elif loss_type == "logcosh":
                d = pred - y_target
                per_sample = d + F.softplus(-2.0 * d) - math.log(2.0)
            else:
                per_sample = (pred - y_target) ** 2
            # Aleatoric weighting
            base = (0.5 * torch.exp(-log_var) * per_sample + 0.5 * log_var).mean()
            self.log(f"{stage}_logvar_mean", log_var.mean(),
                     prog_bar=False, logger=True, sync_dist=True, batch_size=y_true.size(0))
        else:
            if loss_type == "huber":
                base = F.huber_loss(pred, y_target,
                                    delta=self.config["model"].get("huber_delta", 1.0))
            elif loss_type == "logcosh":
                d = pred - y_target
                base = (d + F.softplus(-2.0 * d) - math.log(2.0)).mean()
            else:
                base = F.mse_loss(pred, y_target)
        loss = base

        pcc_w = self.config["model"].get("pcc_loss_weight", 0.0)
        if pcc_w > 0.0 and pred.shape[0] >= 16:
            vx = pred - pred.mean()
            vy = y_target - y_target.mean()
            pcc = (vx * vy).sum() / (vx.norm().clamp(min=1e-4) * vy.norm().clamp(min=1e-4))
            loss = loss + pcc_w * (1.0 - pcc.clamp(-1.0, 1.0))

        if stage == "train" and G is not None:
            y_clean = y_true.float().squeeze(-1)
            rw_enz = self.config["model"].get("rank_loss_weight", 0.0)
            if rw_enz > 0.0 and hasattr(G, "SEQ_seq_id"):
                rl, _, _ = self._pairwise_ranking(
                    y_pred, y_clean, G.SEQ_seq_id,
                    margin=self.config["model"].get("rank_margin_enzyme", 0.1),
                    min_diff=self.config["model"].get("rank_min_diff_enzyme", 0.1),
                    max_pairs=self.config["model"].get("rank_max_pairs_enzyme", 16),
                )
                loss = loss + rw_enz * rl
            rw_sub = self.config["model"].get("rank_loss_substrate_weight", 0.0)
            if rw_sub > 0.0 and hasattr(G, "MOL_smi_id"):
                rl, _, _ = self._pairwise_ranking(
                    y_pred, y_clean, G.MOL_smi_id,
                    margin=self.config["model"].get("rank_margin_substrate", 0.1),
                    min_diff=self.config["model"].get("rank_min_diff_substrate", 0.1),
                    max_pairs=self.config["model"].get("rank_max_pairs_substrate", 16),
                )
                loss = loss + rw_sub * rl

        self.log(f"{stage}_MSE", F.mse_loss(pred, y_target),
                 prog_bar=True, logger=True, sync_dist=True, batch_size=y_true.size(0))
        return loss

    def calculate_metrics(self, y_pred, y_true):
        yp = y_pred.cpu().numpy().flatten()
        yt = y_true.cpu().numpy().flatten()
        r2 = r2_score(yt, yp)
        pcc = np.corrcoef(yt, yp)[0, 1]
        if np.isnan(pcc): pcc = 0.0
        scc, _ = spearmanr(yt, yp)
        if np.isnan(scc): scc = 0.0
        mse = mean_squared_error(yt, yp)
        mae = mean_absolute_error(yt, yp)
        return {"R2": r2, "PCC": pcc, "SCC": scc,
                "MSE": mse, "MAE": mae, "RMSE": float(np.sqrt(mse))}

    def training_step(self, batch, batch_idx):
        y_pred, y_true = self(batch)
        loss = self.get_loss(y_pred, y_true, "train", G=batch)
        self.train_preds.append(y_pred.detach().cpu().float())
        self.train_targets.append(y_true.detach().cpu().float())
        return loss

    def validation_step(self, batch, batch_idx):
        y_pred, y_true = self(batch)
        loss = self.get_loss(y_pred, y_true, "val", G=batch)
        self.val_preds.append(y_pred.detach().cpu().float())
        self.val_targets.append(y_true.detach().cpu().float())
        return loss

    def test_step(self, batch, batch_idx):
        y_pred, y_true = self(batch)
        self.test_preds.append(y_pred.detach().cpu().float())
        self.test_targets.append(y_true.detach().cpu().float())
        return y_pred, y_true

    def _flush_epoch(self, split):
        preds = getattr(self, f"{split}_preds")
        targets = getattr(self, f"{split}_targets")
        if not preds:
            return
        m = self.calculate_metrics(torch.cat(preds), torch.cat(targets))
        self._save_results(split, m, len(torch.cat(targets)))
        setattr(self, f"{split}_preds", [])
        setattr(self, f"{split}_targets", [])

    def on_train_epoch_end(self):      self._flush_epoch("train")
    def on_validation_epoch_end(self): self._flush_epoch("val")

    def on_test_epoch_end(self):
        if not self.test_preds:
            return
        all_p = torch.cat(self.test_preds)
        all_t = torch.cat(self.test_targets)
        if self.trainer is not None and self.trainer.world_size > 1:
            all_p = self.all_gather(all_p).reshape(-1)
            all_t = self.all_gather(all_t).reshape(-1)
        m = self.calculate_metrics(all_p, all_t)
        self._last_test_metrics = dict(m)
        self._save_results("test", m, len(all_t))
        print(f"\nTEST RESULTS -- {self.run_id}")
        for k in ["R2", "PCC", "SCC", "MSE", "MAE", "RMSE"]:
            print(f"  {k}: {m[k]:.6f}")
        self.test_preds, self.test_targets = [], []

    def _save_results(self, split, metrics, n):
        row = pd.DataFrame([{"epoch": self.current_epoch, "split": split, **metrics, "samples": n}])
        row.to_csv(self.results_file, mode="a",
                   header=not os.path.exists(self.results_file), index=False)
        for k, v in metrics.items():
            self.log(f"{split}/{k}", v, prog_bar=k in ["R2", "PCC", "MSE"],
                     logger=True, sync_dist=True)

    def configure_optimizers(self):
        lr = float(self.config["train"]["optimizer"]["lr"])
        wd = float(self.config["train"]["optimizer"].get("weight_decay", 0.01))
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([{"params": trainable, "lr": lr, "weight_decay": wd}])
        warm = int(self.config["train"]["optimizer"].get("warm_epoch", 3))
        max_ep = int(self.config["train"]["max_epochs"])
        min_ratio = float(self.config["train"]["optimizer"]["min_lr"]) / max(lr, 1e-12)
        def lr_lambda(epoch):
            if epoch < warm:
                return (epoch + 1) / max(warm, 1)
            progress = (epoch - warm) / max(max_ep - warm, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_ratio, cosine)
        sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch", "frequency": 1}}
