"""Synthetic smoke test for v4-ultimate.

No precomputed artifacts required — builds a fake PyG-like batch in
memory and calls `model(G)` end-to-end. Verifies that:
  - Model instantiates with the shipped config
  - forward() runs without crash for all branches
  - Modality dropout (train mode) and diagnostics (trainer attached)
    both execute without error
  - backward() computes gradients for at least one parameter

Usage:
    python scripts/27_smoke_forward.py --config config/v4_ultimate.yml
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def build_fake_batch(B: int, hidden_dim: int, device: str = "cpu") -> SimpleNamespace:
    """Create a namespace mimicking the PyG batch fields v4_ultimate consumes."""
    max_seq = 128
    # Substrate: 5 atoms per mol, 5 edges
    atoms_per = 5
    n_edges = 5
    K_pocket = 32
    max_fam = 16

    g = SimpleNamespace()
    g.num_graphs = B
    g.y = torch.randn(B, device=device)

    # Sequence: ProtT5 precomputed-style
    g.SEQ_embedding       = torch.randn(B, max_seq, 1024, device=device)
    g.SEQ_seq_padding_mask = torch.zeros(B, max_seq, dtype=torch.bool, device=device)
    g.SEQ_seq_padding_mask[:, max_seq // 2:] = True  # pad second half
    g.SEQ_seq_id           = list(range(B))

    # Substrate GINE: 9-dim atom features (SubstrateGINE default)
    total_atoms = B * atoms_per
    g.MOL_graph_x = torch.randn(total_atoms, 9, device=device)
    src_dst = torch.randint(0, atoms_per, (2, n_edges), device=device)
    edge_idx_list = []
    for b in range(B):
        edge_idx_list.append(src_dst + b * atoms_per)
    g.MOL_graph_edge_index = torch.cat(edge_idx_list, dim=1)
    g.MOL_graph_edge_attr  = torch.randn(B * n_edges, 3, device=device)
    g.MOL_graph_num_nodes  = torch.full((B,), atoms_per, dtype=torch.long, device=device)
    g.MOL_rxn_center       = torch.randint(0, 2, (total_atoms,), dtype=torch.long, device=device)
    g.MOL_graph_xyz        = torch.randn(total_atoms, 3, device=device) * 3
    g.MOL_graph_xyz_valid  = torch.ones(B, dtype=torch.bool, device=device)
    g.MOL_smi_id           = list(range(B))

    # Pocket: K=32 residues
    total_res = B * K_pocket
    g.POCKET_node_s    = torch.zeros(total_res, 26, device=device)
    g.POCKET_node_s[:, :20] = torch.eye(20).repeat((total_res + 19) // 20, 1)[:total_res]
    g.POCKET_node_s[:, 21] = (torch.rand(total_res) > 0.7).float()  # is_active_site
    g.POCKET_node_s[:, 22] = (torch.rand(total_res) > 0.7).float()  # is_binding_site
    g.POCKET_node_v    = torch.randn(total_res, 2, 3, device=device)
    n_edges_p = 16  # k-NN
    pe_list = []
    for b in range(B):
        src = torch.randint(0, K_pocket, (n_edges_p,), device=device) + b * K_pocket
        dst = torch.randint(0, K_pocket, (n_edges_p,), device=device) + b * K_pocket
        pe_list.append(torch.stack([src, dst]))
    g.POCKET_edge_index = torch.cat(pe_list, dim=1)
    g.POCKET_edge_s     = torch.randn(B * n_edges_p, 16, device=device)
    g.POCKET_edge_v     = torch.randn(B * n_edges_p, 1, 3, device=device)
    g.POCKET_num_nodes  = torch.full((B,), K_pocket, dtype=torch.long, device=device)
    g.POCKET_ca_xyz     = torch.randn(total_res, 3, device=device) * 5

    # RXN DRFP: 2048-dim
    g.RXN_drfp = (torch.rand(B, 2048, device=device) > 0.5).float()

    # Annotations
    g.ANNOT_ipr_ids  = torch.randint(1, 6000, (B, max_fam), device=device)
    g.ANNOT_pf_ids   = torch.randint(1, 3000, (B, max_fam), device=device)
    g.ANNOT_go_ids   = torch.randint(1, 2000, (B, max_fam), device=device)
    g.ANNOT_has_any  = torch.rand(B, device=device) > 0.3
    g.ANNOT_has_cof  = torch.rand(B, device=device) > 0.5

    # Condition
    g.COND_ph   = torch.rand(B, device=device) * 14.0
    g.COND_temp = torch.rand(B, device=device) * 100.0

    # EC
    g.EC_ids = torch.stack([
        torch.randint(1, 7, (B,), device=device),
        torch.randint(1, 199, (B,), device=device),
        torch.randint(1, 199, (B,), device=device),
        torch.randint(1, 1199, (B,), device=device),
    ], dim=1)

    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/v4_ultimate.yml")
    ap.add_argument("-B", type=int, default=4, help="batch size")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["train"]["device"] = "cpu"
    cfg["train"]["log_path"] = str(REPO / ".smoke_logs")
    (REPO / ".smoke_logs").mkdir(exist_ok=True)

    # Stub the ProtT5 encoder (LMDB not present locally)
    import model.v4_ultimate as mod
    class _StubProt5(torch.nn.Module):
        embedding_dim = 1024
        def __init__(self): super().__init__()
    def _fake_prot5(device, config):
        return _StubProt5().to(device)
    mod.prot5_embedding = _fake_prot5

    print("[instantiate] V4Ultimate ...")
    model = mod.V4Ultimate(cfg).cpu()
    print(f"[ok] model parameters: {sum(p.numel() for p in model.parameters()):,}")

    G = build_fake_batch(B=args.B, hidden_dim=cfg["model"]["hidden_dim"])

    # Eval mode first (no dropout, no trainer)
    print("\n[eval forward] ...")
    model.eval()
    with torch.no_grad():
        pred, y = model(G)
    print(f"  pred shape: {tuple(pred.shape)}, finite: {torch.isfinite(pred).all().item()}")

    # Train mode, no trainer (diagnostics skipped)
    print("\n[train forward, no trainer] ...")
    model.train()
    pred, y = model(G)
    print(f"  pred.sum: {pred.sum().item():.4f}")

    # Backward: verify gradient flow
    print("\n[backward] ...")
    loss = (pred - y.view(-1, 1)).pow(2).mean()
    loss.backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_with_grad = sum(1 for p in trainable if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  loss: {loss.item():.4f}")
    print(f"  params with nonzero grad: {n_with_grad} / {len(trainable)}")
    assert n_with_grad > 0, "no gradients flowed"

    # Now with a mock trainer attached so _log_diag runs
    print("\n[train forward, mock trainer attached] ...")
    import pytorch_lightning as pl
    class _StubLogger:
        def log_metrics(self, *a, **kw): pass
        def agg_and_log_metrics(self, *a, **kw): pass
        @property
        def save_dir(self): return "."
        @property
        def name(self): return "stub"
        @property
        def version(self): return 0
    # Attach a bare trainer reference so diag path is exercised. We don't
    # actually run trainer.fit(); just need `model._trainer is not None`.
    trainer = pl.Trainer(max_epochs=1, limit_train_batches=1, logger=False,
                         enable_progress_bar=False, enable_checkpointing=False,
                         accelerator="cpu", devices=1)
    model._trainer = trainer
    model.zero_grad()
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    # Patch self.log to avoid needing full Lightning context
    log_calls = []
    def _fake_log(name, value, **kw):
        log_calls.append((name, float(value) if torch.is_tensor(value) else value))
    model.log = _fake_log
    pred, y = model(G)
    print(f"  pred.sum: {pred.sum().item():.4f}")
    diag_keys = [k for k, _ in log_calls if k.startswith("diag/")]
    print(f"  diag keys logged: {len(diag_keys)}")
    for k in diag_keys[:20]:
        v = dict(log_calls)[k]
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    print("\n[done] all smoke checks passed.")


if __name__ == "__main__":
    main()
