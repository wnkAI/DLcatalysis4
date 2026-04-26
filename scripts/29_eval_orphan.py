"""
Evaluate a trained v4 checkpoint on the orphan-reaction held-out split.

Produces two complementary numbers:

  (A) REGRESSION on orphan-reaction pairs.
      The standard PCC / SCC / R² / MSE / MAE on `orphan_test.csv`,
      where every reaction template is unseen in training.

  (B) TOP-K ENZYME RETRIEVAL per orphan reaction.
      For each unique reaction template R in `orphan_test.csv`, all
      rows with that template form a candidate-enzyme list. We sort the
      candidates by the model's predicted log10(kcat/Km) and ask whether
      any "active" enzyme for R lies in the top-K.

      "Active" is defined by `--active_threshold` on the measured
      log10(kcat/Km): an (R, E) pair counts as active if its measured
      Y_VALUE is at or above the threshold. Default 0.0 (i.e., kcat/Km
      ≥ 1.0).

This is a *within-test* retrieval, not a full library retrieval against
all train enzymes. It directly answers:

  "Given an unseen reaction and a panel of candidate enzymes, can the
  model rank the actual catalysts to the top?"

A full-library retrieval (≈1k test reactions × ≈4k train enzymes ≈ 4M
pair predictions) is the natural follow-up and is gated separately by
`--full_library` (more expensive to run; needs train-enzyme features
loaded into memory).

Usage
-----
  # On the training machine (after a v4_ultimate run produces a ckpt):
  python scripts/29_eval_orphan.py \\
      --config config/v4_ultimate.yml \\
      --ckpt train/ckpt_ultimate/fold_0/last.ckpt \\
      --test_csv data/processed/folds/orphan_test.csv \\
      --active_threshold 0.0 \\
      --out REVIEWS/orphan_eval/fold_0_metrics.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def _load_checkpoint(ckpt_path: str, config_path: str):
    """Lazy import of training stack (PL / PyG); avoids forcing import on
    machines that just want to inspect the script."""
    import yaml
    import pytorch_lightning as pl  # noqa: F401
    from util.data_module import Singledataset
    from train.run_train import get_model_class

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    ModelCls = get_model_class(config)
    model = ModelCls.load_from_checkpoint(ckpt_path, config=config)
    model.eval()
    dm = Singledataset(config)
    return model, dm, config


def _predict_csv(model, dm, csv_path: str, B: int = 32) -> pd.DataFrame:
    """Run forward over every row of a CSV and attach a `pred` column."""
    import torch
    from torch_geometric.loader import DataLoader

    dm.test_data_df = pd.read_csv(csv_path)
    dm.setup("test")
    loader = DataLoader(dm.test_set, batch_size=B, shuffle=False,
                        num_workers=0, follow_batch=dm.follow_batch)

    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(model.device)
            y_pred, _ = model(batch)
            preds.append(y_pred.detach().cpu().numpy().ravel())
    preds = np.concatenate(preds, axis=0)

    out = dm.test_data_df.copy()
    out["pred"] = preds[: len(out)]
    return out


def _make_template_key_column(df: pd.DataFrame) -> pd.Series:
    """Reuse `28_make_orphan_split.py`'s reaction template hash."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_orphan", str(REPO / "scripts" / "28_make_orphan_split.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    keys = []
    for _, row in df.iterrows():
        rxn = row.get("RXN_SMILES", "")
        k = mod.reaction_template_key(rxn) if isinstance(rxn, str) else None
        if k is None:
            k = mod.fallback_template_key(row)
        keys.append(k)
    return pd.Series(keys, index=df.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_csv",
                    default="data/processed/folds/orphan_test.csv")
    ap.add_argument("--active_threshold", type=float, default=0.0,
                    help="rows with Y_VALUE >= this are 'active' for top-K recall")
    ap.add_argument("--k_values", type=int, nargs="+", default=[1, 5, 10, 50])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out", default="REVIEWS/orphan_eval/metrics.json")
    args = ap.parse_args()

    print(f"[load] config={args.config}  ckpt={args.ckpt}")
    model, dm, _ = _load_checkpoint(args.ckpt, args.config)

    print(f"[predict] {args.test_csv}")
    df = _predict_csv(model, dm, args.test_csv, B=args.batch_size)
    print(f"  predictions: {len(df)} rows")

    df["TEMPLATE"] = _make_template_key_column(df)

    from util.metrics import global_metrics, top_k_recall_per_reaction
    reg = global_metrics(df["pred"].to_numpy(), df["Y_VALUE"].to_numpy())

    truth = set(zip(df.loc[df["Y_VALUE"] >= args.active_threshold, "TEMPLATE"],
                    df.loc[df["Y_VALUE"] >= args.active_threshold, "SEQ_ID"]))
    retr = top_k_recall_per_reaction(
        df["pred"].to_numpy(),
        df["TEMPLATE"].tolist(),
        df["SEQ_ID"].tolist(),
        truth,
        k_values=tuple(args.k_values),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config":     args.config,
        "ckpt":       args.ckpt,
        "test_csv":   args.test_csv,
        "active_threshold": args.active_threshold,
        "regression": reg,
        "retrieval":  retr,
        "n_test_rows":          int(len(df)),
        "n_unique_templates":   int(df["TEMPLATE"].nunique()),
        "n_unique_enzymes":     int(df["SEQ_ID"].nunique()),
        "n_active_pairs":       int(len(truth)),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n=== ORPHAN EVAL SUMMARY ===")
    print(f"  rows / templates / enzymes: "
          f"{payload['n_test_rows']} / {payload['n_unique_templates']} / "
          f"{payload['n_unique_enzymes']}")
    print(f"  regression: PCC={reg['PCC']:.4f}  SCC={reg['SCC']:.4f}  "
          f"R²={reg['R2']:.4f}  MAE={reg['MAE']:.4f}")
    print("  retrieval:", {k: f"{v:.4f}" if isinstance(v, float) else v
                            for k, v in retr.items()})
    print(f"\n[wrote] {out_path}")


if __name__ == "__main__":
    main()
