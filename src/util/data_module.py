"""
DLcatalysis 4.0 data module — minimal version.

Strips MSA/Morgan/MolT5/Grover/UniMol from 3.0. Keeps:
  - ProtT5 precomputed LMDB
  - Substrate molecular graph (PyG Data) cache
  - Optional EC embedding
  - GroupedSampler for same-enzyme / same-substrate ranking

Data contract (CSV):
  DATA_ID, SEQ_ID, SMI_ID, EC_NUMBER, Y_VALUE
"""
import sys
import pickle
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import torch_geometric
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from util.tools import set_seed
from util.data_load import SequenceData, padding_seq_embedding
from util.seq_process import SEQ_LMDB_CONFIG, connect_lmdb, read_seq_data


# ──────────────────────────────────────────────────────────────────────
# EC parsing
# ──────────────────────────────────────────────────────────────────────
def _parse_ec_number(ec_str, caps=(8, 200, 200, 1200)):
    """EC '1.2.3.4' -> [ec1, ec2, ec3, ec4], clamped to embedding vocab."""
    try:
        parts = str(ec_str).replace(" ", "").split(".")
    except Exception:
        return [0, 0, 0, 0]
    result = []
    for i in range(4):
        if i < len(parts):
            try:
                val = int(parts[i])
                val = min(max(1, val), caps[i] - 1)
            except (ValueError, TypeError):
                val = 0
        else:
            val = 0
        result.append(val)
    return result


# ──────────────────────────────────────────────────────────────────────
# PyG data wrapper
# ──────────────────────────────────────────────────────────────────────
class SeqMolComplexData(torch_geometric.data.Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_dict(sequence=None, molecular=None, **kwargs):
        inst = SeqMolComplexData(**kwargs)
        if sequence is not None:
            for k, v in sequence.items():
                inst["SEQ_" + k] = v
        if molecular is not None:
            for k, v in molecular.items():
                inst["MOL_" + k] = v

        if hasattr(inst, "SEQ_seq_padding_mask"):
            m = inst.SEQ_seq_padding_mask
            inst.num_nodes = m.shape[0] if m.dim() == 1 else m.shape[1]
        else:
            inst.num_nodes = 1
        return inst

    def __inc__(self, key, value, *args, **kwargs):
        # PyG batching: edge_index shifts by num atoms per graph; everything
        # else is per-sample feature that must NOT be offset.
        if key == "MOL_graph_edge_index":
            return self.MOL_graph_x.size(0) if hasattr(self, "MOL_graph_x") else 0
        if key == "POCKET_edge_index":
            return self.POCKET_node_s.size(0) if hasattr(self, "POCKET_node_s") else 0
        if key in ("MOL_graph_x", "MOL_graph_edge_attr", "MOL_graph_num_nodes"):
            return 0
        if key.startswith("POCKET_"):
            return 0
        if key == "EC_ids":
            return 0
        if key.startswith("SEQ_"):
            return 0
        return super().__inc__(key, value, *args, **kwargs)


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────
class SeqMolDataset(torch.utils.data.Dataset):
    """Sequence (ProtT5) + Molecular graph (GINE). No structure, no MSA."""

    def __init__(self, config: dict, df: pd.DataFrame, is_train: bool = False):
        super().__init__()
        self.config = config
        self.df = df.reset_index(drop=True)
        self.is_train = is_train

        req = ["DATA_ID", "SEQ_ID", "SMI_ID", "EC_NUMBER", "Y_VALUE"]
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise ValueError(f"Dataframe missing columns: {missing}")

        self.data_ids = self.df["DATA_ID"].tolist()
        self.seq_ids  = self.df["SEQ_ID"].tolist()
        self.smi_ids  = self.df["SMI_ID"].tolist()
        self.ec_s     = self.df["EC_NUMBER"].tolist()
        self.y_s      = self.df["Y_VALUE"].tolist()

        self.use_ec = config["model"].get("use_ec", False)

        # ── Sequence LMDB (lazy, per-worker) ──────────────────────────
        self.seq_lmdb_config = SEQ_LMDB_CONFIG(
            seq_fp=config["data"]["seq_lmdb"]["seq_fp"],
            lmdb_fp=config["data"]["seq_lmdb"]["lmdb"],
            map_size=config["data"]["seq_lmdb"]["map_size"],
            max_seq_len=config["data"]["seq_lmdb"]["max_seq_len"],
        )
        self.seq_db = None

        # ── Molecular graphs (in-memory dict) ────────────────────────
        self.use_gnn = config["model"].get("substrate_gnn", {}).get("enabled", True)
        self.mol_graphs = None
        if self.use_gnn:
            gp = config["data"].get("mol_graph_path")
            if gp and Path(gp).exists():
                self.mol_graphs = torch.load(gp, weights_only=False)
                print(f"[data] Loaded {len(self.mol_graphs)} molecular graphs")
            else:
                raise FileNotFoundError(f"mol_graph_path not found: {gp}")

        # ── Pocket structures (in-memory dict, optional) ────────────
        self.pockets = None
        pocket_path = config["data"].get("pocket_path")
        if pocket_path and Path(pocket_path).exists():
            self.pockets = torch.load(pocket_path, weights_only=False)
            print(f"[data] Loaded {len(self.pockets)} pocket graphs")

    # ── LMDB helpers ───────────────────────────────────────────────
    def _get_seq_db(self):
        if self.seq_db is None:
            self.seq_db = connect_lmdb(self.seq_lmdb_config)
        return self.seq_db

    def __del__(self):
        if getattr(self, "seq_db", None) is not None:
            try:
                self.seq_db.close()
            except Exception:
                pass

    # ── Loaders ────────────────────────────────────────────────────
    def _load_seq(self, seq_id: str):
        db = self._get_seq_db()
        try:
            seq_data = read_seq_data(db, seq_id)
        except KeyError:
            print(f"[warn] seq {seq_id} not in LMDB, zero-embedding fallback")
            max_len = self.seq_lmdb_config.max_seq_len
            dat = {"embedding": np.zeros((1, 1024), dtype=np.float32), "sequence": "X"}
            dat = padding_seq_embedding(dat, max_len)
            seq_data = SequenceData.from_dict(dat)

        emb = seq_data.embedding
        if isinstance(emb, torch.Tensor):
            seq_data.embedding = emb.float()
        else:
            seq_data.embedding = torch.as_tensor(emb, dtype=torch.float32)
        seq_data.seq_padding_mask = torch.tensor(seq_data.seq_padding_mask, dtype=torch.bool)
        return seq_data

    def _load_mol_graph(self, smi_id: str) -> dict:
        if not self.use_gnn or self.mol_graphs is None:
            return {}
        g = self.mol_graphs.get(smi_id)
        if g is None:
            return {
                "graph_x": torch.zeros((1, 10), dtype=torch.long),
                "graph_edge_index": torch.zeros((2, 0), dtype=torch.long),
                "graph_edge_attr": torch.zeros((0, 4), dtype=torch.long),
                "graph_num_nodes": torch.tensor([1], dtype=torch.long),
            }
        return {
            "graph_x": g.x,
            "graph_edge_index": g.edge_index,
            "graph_edge_attr": g.edge_attr,
            "graph_num_nodes": torch.tensor([g.x.size(0)], dtype=torch.long),
        }

    def _load_pocket(self, seq_id: str) -> dict:
        """Return pocket tensors for this enzyme, or empty dict if missing."""
        if self.pockets is None:
            return {}
        p = self.pockets.get(seq_id)
        if p is None:
            # Empty fallback — single dummy residue, all zero features
            return {
                "node_s": torch.zeros((1, 26), dtype=torch.float32),
                "node_v": torch.zeros((1, 2, 3), dtype=torch.float32),
                "edge_index": torch.zeros((2, 0), dtype=torch.long),
                "edge_s": torch.zeros((0, 16), dtype=torch.float32),
                "edge_v": torch.zeros((0, 1, 3), dtype=torch.float32),
                "ca_xyz": torch.zeros((1, 3), dtype=torch.float32),
                "num_nodes": torch.tensor([1], dtype=torch.long),
            }
        return {
            "node_s": p["node_s"],
            "node_v": p["node_v"],
            "edge_index": p["edge_index"],
            "edge_s": p["edge_s"],
            "edge_v": p["edge_v"],
            "ca_xyz": p["ca_xyz"],
            "num_nodes": torch.tensor([p["node_s"].size(0)], dtype=torch.long),
        }

    # ── Item ───────────────────────────────────────────────────────
    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx: int):
        seq_id = self.seq_ids[idx]
        smi_id = self.smi_ids[idx]

        seq_data = self._load_seq(seq_id)
        mol_feat = self._load_mol_graph(smi_id)

        data = SeqMolComplexData.from_dict(sequence=seq_data, molecular=mol_feat)

        # Pocket fields (optional)
        pocket_feat = self._load_pocket(seq_id)
        for k, v in pocket_feat.items():
            data["POCKET_" + k] = v

        if self.use_ec:
            ec_ids = _parse_ec_number(self.ec_s[idx])
            data.EC_ids = torch.tensor([ec_ids], dtype=torch.long)

        data.y = torch.tensor([self.y_s[idx]], dtype=torch.float)

        # Store IDs for ranking losses (string — persists through batching)
        data.SEQ_seq_id = seq_id
        data.MOL_smi_id = smi_id

        return data


# ──────────────────────────────────────────────────────────────────────
# Grouped sampler for ranking losses
# ──────────────────────────────────────────────────────────────────────
class GroupedSampler(torch.utils.data.Sampler):
    """Group same-enzyme (or same-substrate) rows into same batch."""

    def __init__(self, df: pd.DataFrame, group_col: str,
                 batch_size: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.groups = defaultdict(list)
        for idx, gid in enumerate(df[group_col]):
            self.groups[gid].append(idx)
        self.group_keys = list(self.groups.keys())

    def __iter__(self):
        import random
        if self.shuffle:
            random.shuffle(self.group_keys)

        batch = []
        for key in self.group_keys:
            indices = self.groups[key]
            if self.shuffle:
                random.shuffle(indices)
            batch.extend(indices)
            while len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
        if batch:
            yield batch

    def __len__(self):
        total = sum(len(v) for v in self.groups.values())
        return (total + self.batch_size - 1) // self.batch_size


# ──────────────────────────────────────────────────────────────────────
# Lightning DataModule
# ──────────────────────────────────────────────────────────────────────
class Singledataset(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        set_seed(config["train"]["seed"])
        self.config = config
        self.batch_size = config["train"]["batch_size"]
        self.n_worker = config["train"]["n_cpus"]

        self.train_df = pd.read_csv(config["data"]["train_data_df"])
        self.val_df   = pd.read_csv(config["data"]["valid_data_df"])
        self.test_df  = pd.read_csv(config["data"]["test_data_df"])

        print(f"\n[data] Train: {len(self.train_df)}  Val: {len(self.val_df)}  Test: {len(self.test_df)}")

        self._train_data = SeqMolDataset(config, self.train_df, is_train=True)
        self._val_data   = SeqMolDataset(config, self.val_df,   is_train=False)
        self._test_data  = SeqMolDataset(config, self.test_df,  is_train=False)

    def train_dataloader(self):
        use_enz_rank = self.config["model"].get("rank_loss_weight", 0.0) > 0
        use_sub_rank = self.config["model"].get("rank_loss_substrate_weight", 0.0) > 0
        if use_enz_rank or use_sub_rank:
            epoch = self.trainer.current_epoch if self.trainer else 0
            if use_enz_rank and use_sub_rank:
                group_col = "SEQ_ID" if epoch % 2 == 0 else "SMI_ID"
            elif use_enz_rank:
                group_col = "SEQ_ID"
            else:
                group_col = "SMI_ID"
            sampler = GroupedSampler(self.train_df, group_col, self.batch_size, shuffle=True)
            return DataLoader(
                self._train_data, batch_sampler=sampler,
                num_workers=self.n_worker, persistent_workers=False, pin_memory=False,
                prefetch_factor=4 if self.n_worker > 0 else None,
            )
        return DataLoader(
            self._train_data, batch_size=self.batch_size, shuffle=True,
            num_workers=self.n_worker, persistent_workers=self.n_worker > 0,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=4 if self.n_worker > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_data, batch_size=self.batch_size, shuffle=False,
            num_workers=self.n_worker, persistent_workers=False, pin_memory=False,
            prefetch_factor=4 if self.n_worker > 0 else None,
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_data, batch_size=self.batch_size, shuffle=False,
            num_workers=self.n_worker, persistent_workers=False, pin_memory=False,
            prefetch_factor=4 if self.n_worker > 0 else None,
        )
