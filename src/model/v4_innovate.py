"""
DLcatalysis 4.0 — v4-innovate (novel-contribution focused)

THREE novel components w.r.t. 3.0 / CatPred / TurNuP / EnzymeCAGE:

  (1) Multi-task heads with physical-consistency regularization.
      Predict (log10 kcat, log10 Km, log10 kcat/Km) jointly.
      Enforce: ŷ_kcat - ŷ_km ≈ ŷ_kcatkm  (stoichiometry of rate constants).
      This decouples turnover vs binding contributions — no prior kinetics
      predictor trains all three with a consistency constraint.

  (2) Pair-specific local complex pocket.
      Pocket is keyed by (uniprot_id, smi_id), not uniprot_id alone.
      Different substrates for the same enzyme get different docked
      complexes → y_struct and y_int3d genuinely differ across substrates
      (EnzymeCAGE's AlphaFill-derived pocket is enzyme-level only).
      Docking handled upstream by scripts/20_run_vina_pair_docking.py.

  (3) Within-enzyme discrimination as primary metric.
      We report per-enzyme Spearman, top-1 hit, pair-accuracy in addition
      to global PCC/R². The model is graded on its ability to rank
      substrates WITHIN an enzyme, matching real-world panel screening.

Borrowed primitives (cited):
  - EnzymeCompoundCrossAttention + interaction_weight (EnzymeCAGE,
    Nat Catal 2026). Port in src/model/ec_cross_attn.py.
  - DRFP reaction fingerprint (TurNuP, Nat Commun 2023).
  - GVP pocket encoder primitive (LigandMPNN / EnzymeCAGE).
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from model.module import MLP
from model.substrate_gnn import SubstrateGINE
from model.gvp_pocket import PocketGVP
from model.ec_cross_attn import EnzymeCompoundCrossAttention, calc_interaction_weight
from util.featurize.seq_prot5 import prot5_embedding
from util.metrics import report_all


class V4Innovate(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters({"config": config})
        self.config = config
        self.hidden_dim = config["model"]["hidden_dim"]
        self.dropout_rate = config["model"].get("dropout", 0.2)
        self.automatic_optimization = True
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Feature flags
        self.use_rxn_drfp = config["model"].get("use_rxn_drfp", True)
        self.use_pocket = config["model"].get("use_pocket", True)
        self.use_int3d = config["model"].get("use_int3d", True)
        self.use_prods_info = config["model"].get("use_prods_info", False)
        self.use_interaction_weight = config["model"].get("use_interaction_weight", True)

        # Multi-task consistency weight
        self.lambda_cons = float(config["model"].get("lambda_cons", 0.3))
        self.lambda_reg = float(config["model"].get("lambda_reg", 1.0))

        # Log dir
        self.log_dir = config["train"]["log_path"]
        os.makedirs(self.log_dir, exist_ok=True)
        model_name = config["model"].get("model_name", "v4_innovate")
        self.results_file = os.path.join(self.log_dir, f"{model_name}_{self.run_id}.csv")

        # Storage for per-epoch flush (preds + targets + enzyme ids)
        self._buf = {s: {"pred_kcat": [], "pred_km": [], "pred_kkm": [],
                         "true_kcat": [], "true_km": [], "true_kkm": [],
                         "seq_id": []} for s in ("train", "val", "test")}
        self._last_test_metrics = None

        # ── ProtT5 (frozen, precomputed LMDB) ──────────────────────────
        init_device = torch.device(
            config["train"]["device"] if torch.cuda.is_available() else "cpu"
        )
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

        # ── Substrate GINE (atom features may include reacting_center flag) ──
        gnn_cfg = config["model"].get("substrate_gnn", {})
        self.substrate_gnn = SubstrateGINE(
            hidden_dim=self.hidden_dim,
            n_layers=gnn_cfg.get("n_layers", 4),
            dropout=gnn_cfg.get("dropout", 0.1),
        )

        # ── DRFP reaction branch ──────────────────────────────────────
        if self.use_rxn_drfp:
            drfp_dim = config["model"].get("drfp_dim", 2048)
            self.rxn_proj = nn.Sequential(
                nn.Linear(drfp_dim, 2 * self.hidden_dim),
                nn.LayerNorm(2 * self.hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            )

        # ── GVP pocket encoder ────────────────────────────────────────
        if self.use_pocket:
            sc = config["model"].get("struct", {})
            self.pocket_gvp = PocketGVP(
                in_s_dim=sc.get("in_s_dim", 21),     # 20 AA one-hot + 1 active-site flag
                in_v_dim=sc.get("in_v_dim", 2),
                hidden_dim=self.hidden_dim,
                v_hidden=sc.get("v_hidden", 16),
                n_layers=sc.get("gvp_layers", 3),
                edge_s_dim=sc.get("edge_s_dim", 16),
                edge_v_dim=sc.get("edge_v_dim", 1),
                dropout=self.dropout_rate,
            )

        # ── EnzymeCAGE cross-attention (bidirectional, optional prods) ─
        if self.use_int3d:
            ia = config["model"].get("int3d", {})
            self.interaction = EnzymeCompoundCrossAttention(
                enz_node_dim=self.hidden_dim,
                cpd_node_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                use_prods_info=self.use_prods_info,
            )
            n_directions = 4 if self.use_prods_info else 2
            self.int3d_proj = nn.Sequential(
                nn.Linear(self.hidden_dim * n_directions, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout_rate),
            )

        # ── Shared trunk MLP ──────────────────────────────────────────
        n_trunk_feats = 1  # enz_pool
        n_trunk_feats += 1  # graph_emb (substrate)
        if self.use_rxn_drfp:
            n_trunk_feats += 1
        if self.use_int3d:
            n_trunk_feats += 1  # int3d pooled
        if self.use_pocket:
            n_trunk_feats += 1  # pocket pool
        trunk_in_dim = self.hidden_dim * n_trunk_feats

        hdr = config["model"]["output_header"]
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in_dim, hdr["hidden_dim"]),
            nn.LayerNorm(hdr["hidden_dim"]),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hdr["hidden_dim"], hdr["hidden_dim"]),
            nn.LayerNorm(hdr["hidden_dim"]),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
        )

        # ── Three task heads: kcat / Km / kcat/Km ──────────────────────
        def _mkhead():
            return MLP(in_dim=hdr["hidden_dim"], out_dim=1,
                       hidden_dim=hdr["hidden_dim"], num_layer=2,
                       norm=hdr["norm_fn"], act_fn=hdr["act_fn"],
                       dropout=self.dropout_rate)
        self.head_kcat = _mkhead()
        self.head_km = _mkhead()
        self.head_kcatkm = _mkhead()

        self._init_weights()

    def _init_weights(self):
        prot5_ids = set(id(m) for m in self.prot5_encoder.modules())
        for m in self.modules():
            if isinstance(m, nn.Linear) and id(m) not in prot5_ids:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ──────────────────────────────────────────────────────────────
    # Encoders (same pattern as v4_pocket, minor cleanups)
    # ──────────────────────────────────────────────────────────────
    def _attn_pool_seq(self, feat, mask):
        scores = self.seq_attn_pool(feat)
        scores = scores.masked_fill(mask < 0.5, torch.finfo(scores.dtype).min)
        has_valid = (mask > 0.5).any(dim=1, keepdim=True)
        weights = torch.softmax(scores, dim=1) * has_valid.float()
        return (feat * weights).sum(dim=1)

    def _encode_enzyme(self, G, B):
        max_len = G.SEQ_seq_padding_mask.shape[1]
        seq_mask = (~G.SEQ_seq_padding_mask).unsqueeze(-1).float().to(self.device)
        x = G.SEQ_embedding.view(B, max_len, -1).to(self.device).float()
        feat = self.seq_mlp(x)
        pooled = self._attn_pool_seq(feat, seq_mask)
        return feat, pooled, seq_mask

    def _encode_substrate(self, G, B):
        if not hasattr(G, "MOL_graph_x") or G.MOL_graph_x is None:
            z = torch.zeros(B, self.hidden_dim, device=self.device)
            return None, z, None, None
        num_nodes = G.MOL_graph_num_nodes.to(self.device).view(-1)
        batch = torch.repeat_interleave(
            torch.arange(num_nodes.size(0), device=self.device), num_nodes
        )
        atom_tokens, graph_emb, atom_mask = self.substrate_gnn(
            x=G.MOL_graph_x.to(self.device),
            edge_index=G.MOL_graph_edge_index.to(self.device),
            edge_attr=G.MOL_graph_edge_attr.to(self.device),
            batch=batch,
        )
        return atom_tokens, graph_emb, atom_mask, batch

    def _encode_rxn(self, G, B):
        if not self.use_rxn_drfp or not hasattr(G, "RXN_drfp"):
            return torch.zeros(B, self.hidden_dim, device=self.device)
        drfp = G.RXN_drfp.to(self.device).float().view(B, -1)
        return self.rxn_proj(drfp)

    def _encode_pocket(self, G, B):
        if not self.use_pocket or not hasattr(G, "POCKET_node_s"):
            return None, torch.zeros(B, self.hidden_dim, device=self.device), None, None
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
        return p_tokens, p_pool, p_mask, pocket_xyz

    def _get_reacting_center(self, G, atom_batch, atom_mask):
        """Return (B, A) reacting-center mask aligned to atom_mask."""
        if atom_batch is None or not hasattr(G, "MOL_rxn_center"):
            return None
        from torch_geometric.utils import to_dense_batch
        rc = G.MOL_rxn_center.to(self.device).long()
        rc_dense, _ = to_dense_batch(rc, atom_batch)  # (B, A)
        return rc_dense

    # ──────────────────────────────────────────────────────────────
    # Forward: returns dict with 3 predictions
    # ──────────────────────────────────────────────────────────────
    def forward(self, G):
        B = G.num_graphs if hasattr(G, "num_graphs") else 1

        _, enz_pool, _ = self._encode_enzyme(G, B)
        atom_tokens, graph_emb, atom_mask, atom_batch = self._encode_substrate(G, B)
        rxn_emb = self._encode_rxn(G, B)
        pocket_tokens, pocket_pool, pocket_mask, pocket_xyz = self._encode_pocket(G, B)

        # Interaction (EnzymeCAGE-style bidirectional cross-attn)
        int3d_emb = torch.zeros(B, self.hidden_dim, device=self.device)
        if (self.use_int3d and pocket_tokens is not None and
                atom_tokens is not None and pocket_mask is not None and atom_mask is not None):
            # Compute biology-prior interaction weight if reacting center available
            interaction_weight = None
            if self.use_interaction_weight:
                rc = self._get_reacting_center(G, atom_batch, atom_mask)
                if rc is not None and pocket_xyz is not None:
                    interaction_weight = calc_interaction_weight(
                        pocket_xyz, pocket_mask, rc, atom_mask,
                    )  # (B, A, K)
            cat = self.interaction(
                enz=pocket_tokens, sub=atom_tokens, prod=None,
                enz_mask=pocket_mask, sub_mask=atom_mask, prod_mask=None,
                interaction_weight=interaction_weight,
            )
            int3d_emb = self.int3d_proj(cat)

        # Build trunk input
        trunk_input = [enz_pool, graph_emb]
        if self.use_rxn_drfp:
            trunk_input.append(rxn_emb)
        if self.use_int3d:
            trunk_input.append(int3d_emb)
        if self.use_pocket:
            trunk_input.append(pocket_pool)
        h = self.trunk(torch.cat(trunk_input, dim=-1))

        y_kcat = self.head_kcat(h).squeeze(-1)      # (B,)
        y_km = self.head_km(h).squeeze(-1)          # (B,)
        y_kcatkm = self.head_kcatkm(h).squeeze(-1)  # (B,)

        return {"pred_kcat": y_kcat, "pred_km": y_km, "pred_kcatkm": y_kcatkm}

    # ──────────────────────────────────────────────────────────────
    # Loss: multi-task + consistency
    # ──────────────────────────────────────────────────────────────
    def _extract_targets(self, G):
        """Return three tensors on device, each (B,), with NaN for missing."""
        device = self.device
        # G.y is kept for backward compat (log10(kcat/Km))
        # v4-innovate expects the richer targets through G.Y_KCAT etc.
        def _get(name, fallback=None):
            if hasattr(G, name):
                return getattr(G, name).to(device).float().view(-1)
            return fallback

        yk = _get("Y_KCAT", None)
        ym = _get("Y_KM", None)
        ykm = _get("Y_KCATKM", None)

        # Minimum contract: at least y (= log10 kcat/Km) must be present
        if ykm is None:
            if hasattr(G, "y"):
                ykm = G.y.to(device).float().view(-1)
            else:
                raise ValueError("Neither Y_KCATKM nor y present on batch")

        # Fill missing with NaN so masked loss can skip
        B = ykm.shape[0]
        if yk is None:
            yk = torch.full((B,), float("nan"), device=device)
        if ym is None:
            ym = torch.full((B,), float("nan"), device=device)
        return yk, ym, ykm

    def _masked_logcosh(self, pred, target):
        """LogCosh over the non-nan entries; returns scalar mean or 0 if all NaN."""
        valid = torch.isfinite(target)
        if not valid.any():
            return torch.tensor(0.0, device=pred.device)
        d = pred[valid] - target[valid]
        return (d + F.softplus(-2.0 * d) - math.log(2.0)).mean()

    def get_loss(self, preds: dict, G, stage: str):
        yk, ym, ykm = self._extract_targets(G)
        pk, pm, pkm = preds["pred_kcat"], preds["pred_km"], preds["pred_kcatkm"]

        L_kcat = self._masked_logcosh(pk, yk)
        L_km = self._masked_logcosh(pm, ym)
        L_kkm = self._masked_logcosh(pkm, ykm)
        L_reg = self.lambda_reg * (L_kcat + L_km + L_kkm)

        # Physical consistency: pred_kcat - pred_km ≈ pred_kcatkm
        #   (log10 space: log kcat/Km = log kcat - log Km)
        # Only compute where both kcat AND km targets valid
        both_valid = torch.isfinite(yk) & torch.isfinite(ym)
        if both_valid.any():
            diff = pk[both_valid] - pm[both_valid] - pkm[both_valid]
            d = diff
            L_cons = (d + F.softplus(-2.0 * d) - math.log(2.0)).mean()
        else:
            L_cons = torch.tensor(0.0, device=pk.device)

        total = L_reg + self.lambda_cons * L_cons

        # Log components
        self.log(f"{stage}_L_kcat", L_kcat, prog_bar=False, logger=True, sync_dist=True, batch_size=yk.size(0))
        self.log(f"{stage}_L_km", L_km, prog_bar=False, logger=True, sync_dist=True, batch_size=yk.size(0))
        self.log(f"{stage}_L_kkm", L_kkm, prog_bar=True, logger=True, sync_dist=True, batch_size=yk.size(0))
        self.log(f"{stage}_L_cons", L_cons, prog_bar=True, logger=True, sync_dist=True, batch_size=yk.size(0))
        self.log(f"{stage}_loss", total, prog_bar=True, logger=True, sync_dist=True, batch_size=yk.size(0))
        return total

    # ──────────────────────────────────────────────────────────────
    # Lightning steps
    # ──────────────────────────────────────────────────────────────
    def _buffer(self, stage, preds, G):
        yk, ym, ykm = self._extract_targets(G)
        self._buf[stage]["pred_kcat"].append(preds["pred_kcat"].detach().cpu())
        self._buf[stage]["pred_km"].append(preds["pred_km"].detach().cpu())
        self._buf[stage]["pred_kkm"].append(preds["pred_kcatkm"].detach().cpu())
        self._buf[stage]["true_kcat"].append(yk.detach().cpu())
        self._buf[stage]["true_km"].append(ym.detach().cpu())
        self._buf[stage]["true_kkm"].append(ykm.detach().cpu())
        if hasattr(G, "SEQ_seq_id"):
            self._buf[stage]["seq_id"].extend(list(G.SEQ_seq_id))
        else:
            self._buf[stage]["seq_id"].extend([""] * yk.shape[0])

    def training_step(self, batch, batch_idx):
        preds = self(batch)
        loss = self.get_loss(preds, batch, "train")
        self._buffer("train", preds, batch)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(batch)
        _ = self.get_loss(preds, batch, "val")
        self._buffer("val", preds, batch)

    def test_step(self, batch, batch_idx):
        preds = self(batch)
        _ = self.get_loss(preds, batch, "test")
        self._buffer("test", preds, batch)

    # ──────────────────────────────────────────────────────────────
    # Epoch flush → compute metrics (global + within-enzyme)
    # ──────────────────────────────────────────────────────────────
    def _flush(self, stage: str):
        buf = self._buf[stage]
        if not buf["pred_kkm"]:
            return
        pk = torch.cat(buf["pred_kcat"]).numpy()
        pm = torch.cat(buf["pred_km"]).numpy()
        pkm = torch.cat(buf["pred_kkm"]).numpy()
        tk = torch.cat(buf["true_kcat"]).numpy()
        tm = torch.cat(buf["true_km"]).numpy()
        tkm = torch.cat(buf["true_kkm"]).numpy()
        seq = buf["seq_id"]

        # Metrics for each target, with within-enzyme stats
        all_metrics = {}
        all_metrics.update(report_all(pkm, tkm, seq, tag="kcatkm_"))
        all_metrics.update(report_all(pk, tk, seq, tag="kcat_"))
        all_metrics.update(report_all(pm, tm, seq, tag="km_"))

        # Save row
        row = pd.DataFrame([{"epoch": self.current_epoch, "split": stage, **all_metrics}])
        row.to_csv(self.results_file, mode="a",
                   header=not os.path.exists(self.results_file), index=False)

        # Log the star metrics
        for key in ("kcatkm_global_PCC", "kcatkm_global_SCC", "kcatkm_global_MSE",
                    "kcatkm_within_SCC_mean", "kcatkm_within_top1_hit",
                    "kcatkm_within_pair_acc"):
            if key in all_metrics and np.isfinite(all_metrics[key]):
                self.log(f"{stage}/{key}", all_metrics[key],
                         prog_bar=key.endswith("within_SCC_mean"),
                         logger=True, sync_dist=True)

        # Clear
        for k in buf:
            buf[k].clear() if isinstance(buf[k], list) else None
        buf["seq_id"] = []

        if stage == "test":
            self._last_test_metrics = dict(all_metrics)
            print(f"\n=== TEST METRICS ({self.run_id}) ===")
            for k in ("kcatkm_global_PCC", "kcatkm_global_SCC",
                      "kcatkm_within_SCC_mean", "kcatkm_within_top1_hit",
                      "kcatkm_within_pair_acc", "kcat_global_PCC",
                      "km_global_PCC"):
                if k in all_metrics:
                    print(f"  {k}: {all_metrics[k]:.4f}")

    def on_train_epoch_end(self):      self._flush("train")
    def on_validation_epoch_end(self): self._flush("val")
    def on_test_epoch_end(self):       self._flush("test")

    # ──────────────────────────────────────────────────────────────
    # Optimizer
    # ──────────────────────────────────────────────────────────────
    def configure_optimizers(self):
        lr = float(self.config["train"]["optimizer"]["lr"])
        wd = float(self.config["train"]["optimizer"].get("weight_decay", 0.01))
        trainable = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)
        warm = int(self.config["train"]["optimizer"].get("warm_epoch", 3))
        max_ep = int(self.config["train"]["max_epochs"])
        min_ratio = float(self.config["train"]["optimizer"]["min_lr"]) / max(lr, 1e-12)

        def lr_lambda(epoch):
            if epoch < warm:
                return (epoch + 1) / max(warm, 1)
            progress = (epoch - warm) / max(max_ep - warm, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_ratio, cosine)

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
