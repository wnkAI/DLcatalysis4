"""
Make train/valid splits for each CV fold.

For fold k:
  test_fold{k}.csv  = fold_{k}.csv
  valid_fold{k}.csv = fold_{(k+1) % 10}.csv
  train_fold{k}.csv = concat of the remaining 8 folds

Usage:
  python scripts/13_make_cv_splits.py
"""
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds_dir", default="E:/AImodel/DLcatalysis4.0/data/processed/folds")
    ap.add_argument("--n_folds", type=int, default=10)
    args = ap.parse_args()

    folds_dir = Path(args.folds_dir)
    fold_dfs = []
    for k in range(args.n_folds):
        p = folds_dir / f"fold_{k}.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        fold_dfs.append(pd.read_csv(p))

    total = sum(len(d) for d in fold_dfs)
    print(f"[load] {args.n_folds} folds, {total} total rows")

    for k in range(args.n_folds):
        test_df = fold_dfs[k]
        val_df = fold_dfs[(k + 1) % args.n_folds]
        train_idxs = [i for i in range(args.n_folds) if i != k and i != (k + 1) % args.n_folds]
        train_df = pd.concat([fold_dfs[i] for i in train_idxs], ignore_index=True)
        test_df.to_csv(folds_dir / f"test_fold{k}.csv", index=False)
        val_df.to_csv(folds_dir / f"valid_fold{k}.csv", index=False)
        train_df.to_csv(folds_dir / f"train_fold{k}.csv", index=False)
        print(f"  fold {k}: train={len(train_df):5d}  val={len(val_df):5d}  test={len(test_df):5d}")


if __name__ == "__main__":
    main()
