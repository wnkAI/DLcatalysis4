"""
DLcatalysis 4.0 — single-fold or CV training entry.

Usage:
  # single fold
  python train/run_train.py --config config/v4_minimal.yml --fold 0

  # 10-fold CV
  python train/run_train.py --config config/v4_minimal.yml --cv

The config specifies train/valid/test CSV paths via {fold} placeholder or
hard-coded paths. When --fold is given we overwrite those paths to:
  data/processed/folds/train_fold{k}.csv
  data/processed/folds/valid_fold{k}.csv
  data/processed/folds/test_fold{k}.csv
"""
import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from util.data_module import Singledataset
from model.v4_minimal import V4Minimal
from model.v4_pocket import V4Pocket
from model.v4_ultimate import V4Ultimate


def get_model_class(config):
    name = config["model"].get("model_name", "v4_minimal").lower()
    if "ultimate" in name:
        return V4Ultimate
    if "pocket" in name:
        return V4Pocket
    return V4Minimal


def apply_fold_paths(config: dict, fold: int):
    base = Path(config["data"]["train_data_df"]).parent
    config["data"]["train_data_df"] = str(base / f"train_fold{fold}.csv")
    config["data"]["valid_data_df"] = str(base / f"valid_fold{fold}.csv")
    config["data"]["test_data_df"]  = str(base / f"test_fold{fold}.csv")
    return config


def train_one_fold(config: dict, fold: int, args):
    config = deepcopy(config)
    config = apply_fold_paths(config, fold)

    ckpt_dir = Path(config["train"]["checkpoint_path"]) / f"fold_{fold}"
    log_dir = Path(config["train"]["log_path"]) / f"fold_{fold}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    dm = Singledataset(config)
    ModelCls = get_model_class(config)
    model = ModelCls(config)
    print(f"[model] Using {ModelCls.__name__}")

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best-{epoch:02d}-{val_MSE:.4f}",
            monitor="val_MSE", mode="min", save_top_k=1, save_last=True,
        ),
        EarlyStopping(monitor="val_MSE", mode="min",
                      patience=config["train"].get("earlystrop_patience", 15)),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    logger = CSVLogger(save_dir=str(log_dir), name="csv_logs")

    trainer = pl.Trainer(
        accelerator="gpu" if config["train"].get("device") == "cuda" else "cpu",
        devices=config["train"].get("n_gpu", 1),
        max_epochs=config["train"]["max_epochs"],
        min_epochs=config["train"]["min_epochs"],
        precision=config["train"].get("precision", "32"),
        gradient_clip_val=1.0,
        accumulate_grad_batches=config["train"].get("grad_accum", 1),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=20,
        check_val_every_n_epoch=1,
        deterministic=False,
    )

    print(f"\n=== Fold {fold} training ===")
    trainer.fit(model, datamodule=dm)
    print(f"\n=== Fold {fold} testing ===")
    trainer.test(model, datamodule=dm, ckpt_path="best")
    return model._last_test_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=0, help="single-fold index (ignored if --cv)")
    ap.add_argument("--cv", action="store_true", help="run full 10-fold CV")
    ap.add_argument("--n_folds", type=int, default=10)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.cv:
        results = []
        for k in range(args.n_folds):
            m = train_one_fold(config, k, args)
            if m is not None:
                m = dict(m); m["fold"] = k
                results.append(m)
        if results:
            import pandas as pd
            df = pd.DataFrame(results)
            print("\n" + "=" * 60)
            print("10-fold CV summary:")
            print(df.to_string(index=False))
            for k in ["PCC", "SCC", "R2", "MSE", "MAE"]:
                if k in df.columns:
                    print(f"  {k}: {df[k].mean():.4f} ± {df[k].std():.4f}")
    else:
        train_one_fold(config, args.fold, args)


if __name__ == "__main__":
    main()
