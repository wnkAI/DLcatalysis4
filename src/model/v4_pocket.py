"""
DLcatalysis 4.0 — v4-pocket (full 4-branch residual with structure).

    pred = y_seq + g_pair * (y_sub + y_int3d) + g_struct * y_struct

    y_seq   : ProtT5 (frozen) -> attn-pool -> MLP head      (enzyme baseline)
    y_sub   : GINE(4) on substrate SMILES -> graph_pool      (substrate effect)
    y_int3d : residue tokens × atom tokens with 3D distance  (interaction)
             cross-attention bias → pooled → MLP
    y_struct: GVP-GNN pocket encoder -> attn-pool -> MLP      (pocket structure)
    g_pair  : sigmoid(MLP([enz_pool, graph_emb]))
    g_struct: sigmoid(MLP([enz_pool, graph_emb, pocket_pool, struct_quality]))
              initial bias = -2.0 so structure starts as conservative residual

Compared to v4_minimal:
  - Adds GVP pocket branch
  - Replaces attention-only interaction with 3D-distance-aware cross-attention
  - Two independent gates (g_pair stays, g_struct is new)
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


class V4Pocket(pl.LightningModule):
    """4-branch residual kcat/Km predictor with pocket structure."""

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
        model_name = config["model"].get("model_name", "v4_pocket")
        self.results_file = os.path.join(self.log_dir, f"{model_name}_{self.run_id}.csv")

        # ── ProtT5 (frozen, precomputed LMDB) ────────────────────────
        init_device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")
        enc_cfg = dict(config["model"])
        enc_cfg["precomputed_only"] = True
        self.prot5_encoder = prot5_embedding(device=init_device, config=enc_cfg)
        for p in self.prot5_encoder.parameters():
            p.requires_grad = False

        # ── Sequence projection + attn pool ──────────────────────────
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

        # ── Substrate GINE ───────────────────────────────────────────
        gnn_cfg = config["model"].get("substrate_gnn", {})
        self.substrate_gnn = SubstrateGINE(
            hidden_dim=self.hidden_dim,
            n_layers=gnn_cfg.get("n_layers", 4),
            dropout=gnn_cfg.get("dropout", 0.1),
        )

        # ── GVP pocket encoder ───────────────────────────────────────
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

        # ── 3D cross-attention (pocket residue × substrate atom) ────
        int3d_cfg = config["model"].get("int3d", {})
        self.int3d = Int3DCrossAttn(
            hidden_dim=self.hidden_dim,
            n_heads=int3d_cfg.get("n_head", 4),
            n_layers=int3d_cfg.get("n_layers", 2),
            n_rbf=int3d_cfg.get("rbf_bins", 32),
            max_dist=int3d_cfg.get("max_dist", 20.0),
            dropout=self.dropout_rate,
        )

        # ── EC embedding ────────────────────────────────────────────
        self.use_ec = config["model"].get("use_ec", False)
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

        # ── Four prediction heads ───────────────────────────────────
        hdr = config["model"]["output_header"]
        self.head_seq = MLP(in_dim=self.hidden_dim, out_dim=1,
                            hidden_dim=hdr["hidden_dim"], num_layer=hdr["num_layers"],
                            norm=hdr["norm_fn"], act_fn=hdr["act_fn"], dropout=self.dropout_rate)
        self.head_sub = MLP(in_dim=self.hidden_dim, out_dim=1,
                            hidden_dim=hdr["hidden_dim"], num_layer=hdr["num_layers"],
                            norm=hdr["norm_fn"], act_fn=hdr["act_fn"], dropout=self.dropout_rate)
        # int3d input: pool of pocket + pool of substrate (cross-attended)
        self.head_int3d = MLP(in_dim=self.hidden_dim * 2, out_dim=1,
                              hidden_dim=hdr["hidden_dim"], num_layer=hdr["num_layers"],
                              norm=hdr["norm_fn"], act_fn=hdr["act_fn"], dropout=self.dropout_rate)
        self.head_struct = MLP(in_dim=self.hidden_dim, out_dim=1,
                               hidden_dim=hdr["hidden_dim"], num_layer=hdr["num_layers"],
                               norm=hdr["norm_fn"], act_fn=hdr["act_fn"], dropout=self.dropout_rate)

        # ── Dual gates ──────────────────────────────────────────────
        pair_in = self.hidden_dim * 2
        if self.use_ec:
            pair_in += self.hidden_dim
        self.gate_pair = nn.Sequential(
            nn.Linear(pair_in, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )

        # g_struct input: enzyme_pool + graph_emb + pocket_pool + quality scalars
        # Quality scalars: n_pocket_residues / K, has_annotation, has_cofactor
        struct_quality_dim = 3
        struct_in = self.hidden_dim * 3 + struct_quality_dim
        if self.use_ec:
            struct_in += self.hidden_dim
        self.gate_struct = nn.Sequential(
            nn.Linear(struct_in, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 1),
        )  # sigmoid applied in forward (so we can set bias explicitly)
        nn.init.constant_(self.gate_struct[-1].bias, -2.0)  # conservative init

        self._init_weights()
        nn.init.constant_(self.gate_struct[-1].bias, -2.0)

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
            zeros = torch.zeros(B, self.hidden_dim, device=self.device)
            return None, zeros, None
        max_len = G.SEQ_seq_padding_mask.shape[1]
        seq_mask = (~G.SEQ_seq_padding_mask).unsqueeze(-1).float().to(self.device)
        x = G.SEQ_embedding.view(B, max_len, -1).to(self.device).float()
        feat = self.seq_mlp(x)
        pooled = self._attn_pool_seq(feat, seq_mask, self.seq_attn_pool)
        return feat, pooled, seq_mask

    def _encode_substrate(self, G, B):
        if not hasattr(G, "MOL_graph_x") or G.MOL_graph_x is None:
            z = torch.zeros(B, self.hidden_dim, device=self.device)
            return None, z, None
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
        return atom_tokens, graph_emb, atom_mask

    def _encode_pocket(self, G, B):
        """Pocket tensors packed as PyG sub-graph under POCKET_* prefix."""
        if not hasattr(G, "POCKET_node_s"):
            return None, torch.zeros(B, self.hidden_dim, device=self.device), None, None

        node_s = G.POCKET_node_s.to(self.device).float()
        node_v = G.POCKET_node_v.to(self.device).float()
        edge_index = G.POCKET_edge_index.to(self.device).long()
        edge_s = G.POCKET_edge_s.to(self.device).float()
        edge_v = G.POCKET_edge_v.to(self.device).float()

        # Batch vector reconstructed from num_nodes per pocket
        num_nodes = G.POCKET_num_nodes.to(self.device).view(-1)
        batch = torch.repeat_interleave(
            torch.arange(num_nodes.size(0), device=self.device), num_nodes
        )
        p_tokens, p_pool, p_mask = self.pocket_gvp(
            node_s, node_v, edge_index, edge_s, edge_v, batch
        )
        # Cα coords for 3D cross-attn
        ca_xyz = G.POCKET_ca_xyz.to(self.device).float()  # flat (N_total, 3)
        from torch_geometric.utils import to_dense_batch
        pocket_xyz, _ = to_dense_batch(ca_xyz, batch)  # (B, K_max, 3)
        return p_tokens, p_pool, p_mask, pocket_xyz

    def _encode_ec(self, G, B):
        if not (self.use_ec and hasattr(G, "EC_ids")):
            return None
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

        enzyme_feat, enz_pool, seq_mask = self._encode_enzyme(G, B)
        atom_tokens, graph_emb, atom_mask = self._encode_substrate(G, B)
        pocket_tokens, pocket_pool, pocket_mask, pocket_xyz = self._encode_pocket(G, B)
        ec_feat = self._encode_ec(G, B)
        _ec = ec_feat if ec_feat is not None else torch.zeros(B, self.hidden_dim, device=self.device)

        # ── Four branches ──
        y_seq = self.head_seq(enz_pool)
        y_sub = self.head_sub(graph_emb)

        # int3d: substrate atoms vs pocket residues with 3D distance bias
        # Substrate atom 3D coords from docked pose (if available); otherwise disable bias
        if pocket_tokens is not None and atom_tokens is not None and pocket_mask is not None:
            if hasattr(G, "MOL_graph_xyz"):
                # Substrate atom coords (N_total, 3) packed; to_dense_batch
                from torch_geometric.utils import to_dense_batch
                num_nodes = G.MOL_graph_num_nodes.to(self.device).view(-1)
                atom_batch = torch.repeat_interleave(
                    torch.arange(num_nodes.size(0), device=self.device), num_nodes
                )
                atom_xyz, _ = to_dense_batch(G.MOL_graph_xyz.to(self.device).float(), atom_batch)
            else:
                atom_xyz = None
            _, _, p_pool_int, a_pool_int = self.int3d(
                pocket_tokens, atom_tokens, pocket_mask, atom_mask,
                xyz_p=pocket_xyz, xyz_a=atom_xyz,
            )
            int3d_input = torch.cat([p_pool_int, a_pool_int], dim=-1)
            y_int3d = self.head_int3d(int3d_input)
        else:
            y_int3d = torch.zeros(B, 1, device=self.device)

        y_struct = self.head_struct(pocket_pool) if pocket_tokens is not None else torch.zeros(B, 1, device=self.device)

        # ── Two independent gates ──
        gate_pair_in = torch.cat([enz_pool, graph_emb], dim=-1)
        if self.use_ec:
            gate_pair_in = torch.cat([gate_pair_in, _ec], dim=-1)
        g_pair = self.gate_pair(gate_pair_in)  # (B, 1) sigmoid in sequential

        # Structure quality scalars
        if pocket_mask is not None:
            n_res = pocket_mask.sum(dim=1, keepdim=True).float() / 32.0
        else:
            n_res = torch.zeros(B, 1, device=self.device)
        has_annot = torch.zeros(B, 1, device=self.device)  # TODO: pass from dataloader
        has_cof = torch.zeros(B, 1, device=self.device)    # TODO: pass from dataloader
        struct_quality = torch.cat([n_res, has_annot, has_cof], dim=-1)  # (B, 3)

        gate_struct_in = torch.cat([enz_pool, graph_emb, pocket_pool, struct_quality], dim=-1)
        if self.use_ec:
            gate_struct_in = torch.cat([gate_struct_in, _ec], dim=-1)
        g_struct = torch.sigmoid(self.gate_struct(gate_struct_in))

        # ── Final prediction ──
        pair_delta = y_sub + y_int3d
        pred = y_seq + g_pair * pair_delta + g_struct * y_struct
        return pred, G.y

    # ──────────────────────────────────────────────────────────────
    # Target / loss / metrics / steps — identical to v4_minimal
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
