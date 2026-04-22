"""
Precompute DRFP reaction fingerprints for unique RXN_SMILES.

DRFP (Differential Reaction Fingerprint) — Probst et al., J. Cheminform. 2022.
Captures atom/bond environment differences between reactants and products as
a binary fingerprint. Used by TurNuP (Nat Commun 2023) for kcat prediction.

Install:
  pip install drfp

Usage:
  python scripts/17_precompute_drfp.py \
    --final_data data/processed/final_data.csv \
    --out data/processed/rxn_drfp.npy \
    --keys_out data/processed/rxn_drfp_keys.csv \
    --n_folded 2048
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def rxn_hash(rxn_smi: str) -> str:
    """Stable 10-char ID for a reaction SMILES."""
    return "RXN_" + hashlib.sha1(rxn_smi.encode("utf-8")).hexdigest()[:10].upper()


def compute_drfp_batch(rxns, n_folded=2048):
    """Compute DRFP fingerprints for a list of reaction SMILES."""
    from drfp import DrfpEncoder
    fps = DrfpEncoder.encode(rxns, n_folded_length=n_folded)
    return np.asarray(fps, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_data", default="E:/AImodel/DLcatalysis4.0/data/processed/final_data.csv")
    ap.add_argument("--out", default="E:/AImodel/DLcatalysis4.0/data/processed/rxn_drfp.npy")
    ap.add_argument("--keys_out", default="E:/AImodel/DLcatalysis4.0/data/processed/rxn_drfp_keys.csv")
    ap.add_argument("--n_folded", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    df = pd.read_csv(args.final_data, low_memory=False)
    if "RXN_SMILES" not in df.columns:
        raise ValueError(f"'RXN_SMILES' column missing in {args.final_data}")

    # Deduplicate — many rows share the same reaction
    unique_rxns = df["RXN_SMILES"].dropna().drop_duplicates().tolist()
    print(f"[load] {len(df)} total rows, {len(unique_rxns)} unique reaction SMILES")

    # Compute in batches
    all_fps = []
    all_keys = []
    failed = []
    for i in tqdm(range(0, len(unique_rxns), args.batch_size), desc="drfp"):
        batch = unique_rxns[i : i + args.batch_size]
        try:
            fps = compute_drfp_batch(batch, n_folded=args.n_folded)
            for rxn, fp in zip(batch, fps):
                all_keys.append(rxn_hash(rxn))
                all_fps.append(fp)
        except Exception as e:
            # fallback: single encode to keep what works
            print(f"  batch {i} failed ({e}), trying one by one ...")
            from drfp import DrfpEncoder
            for r in batch:
                try:
                    fp = DrfpEncoder.encode([r], n_folded_length=args.n_folded)[0]
                    all_keys.append(rxn_hash(r))
                    all_fps.append(np.asarray(fp, dtype=np.float32))
                except Exception:
                    failed.append(r)

    fps_arr = np.stack(all_fps, axis=0)                    # (N, n_folded)
    keys_df = pd.DataFrame({
        "RXN_ID": all_keys,
        "RXN_SMILES": [r for r in unique_rxns if rxn_hash(r) in set(all_keys)],
    })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, fps_arr)
    keys_df.to_csv(args.keys_out, index=False)

    print()
    print(f"[done] {fps_arr.shape[0]} reactions encoded  (dim={args.n_folded})")
    print(f"[save] {args.out}  + {args.keys_out}")
    if failed:
        print(f"[warn] {len(failed)} reactions failed encoding")


if __name__ == "__main__":
    main()
