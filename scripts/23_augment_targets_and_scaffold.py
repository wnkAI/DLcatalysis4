"""
Augment final_data.csv and smi.csv with:
  (1) LOG_KCAT / LOG_KM / LOG_KCAT_KM — three independent regression targets
      (reviewer recommends decomposing catalytic efficiency into
       kcat (turnover) + km (binding-like) + kcat/km (efficiency) for
       future multi-task consistency loss).
  (2) SCAFFOLD_ID in smi.csv — RDKit Murcko scaffold hash per substrate,
      joined back into final_data via SMI_ID. Enables substrate OOD
      scaffold split and substrate-family analysis.

Input:
  data/processed/final_data.csv   (must have KCAT, KM, KCAT_KM, SMI_ID)
  data/processed/smi.csv          (SMI_ID, SMILES)

Output:
  data/processed/final_data_augmented.csv  (adds 3 log + SCAFFOLD_ID)
  data/processed/smi_augmented.csv         (adds SCAFFOLD_ID + SCAFFOLD_SMILES)

Usage:
  python scripts/23_augment_targets_and_scaffold.py
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")


def _safe_log10(x):
    if x is None:
        return np.nan
    try:
        v = float(x)
    except (ValueError, TypeError):
        return np.nan
    if not np.isfinite(v) or v <= 0:
        return np.nan
    return float(np.log10(v))


def murcko_scaffold_id(smi: str) -> tuple:
    """Return (scaffold_smiles, scaffold_hash_id). Empty-scaffold molecules
    (e.g., acyclic) collapse to 'NOSCAFFOLD' to keep cluster definition sane."""
    if not isinstance(smi, str) or not smi.strip():
        return ("", "SCF_NONE")
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ("", "SCF_INVALID")
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return ("", "SCF_ACYCLIC")
        scaffold_smi = Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return ("", "SCF_FAIL")
    h = "SCF_" + hashlib.sha1(scaffold_smi.encode("utf-8")).hexdigest()[:10].upper()
    return (scaffold_smi, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_data", default="E:/AImodel/DLcatalysis4.0/data/processed/final_data.csv")
    ap.add_argument("--smi_csv", default="E:/AImodel/DLcatalysis4.0/data/processed/smi.csv")
    ap.add_argument("--out_final", default="E:/AImodel/DLcatalysis4.0/data/processed/final_data_augmented.csv")
    ap.add_argument("--out_smi", default="E:/AImodel/DLcatalysis4.0/data/processed/smi_augmented.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.final_data, low_memory=False)
    smi_df = pd.read_csv(args.smi_csv)
    print(f"[load] final_data: {len(df)} rows  |  smi: {len(smi_df)} unique substrates")

    # ── 1. Multi-task targets ──────────────────────────────────────────
    print("[task] computing LOG_KCAT / LOG_KM / LOG_KCAT_KM ...")
    # KCAT_KM field is raw kcat/Km (M^-1 s^-1) if present; otherwise kcat/km
    df["LOG_KCAT"]    = df["KCAT"].map(_safe_log10)
    df["LOG_KM"]      = df["KM"].map(_safe_log10)
    df["LOG_KCAT_KM"] = df["KCAT_KM"].map(_safe_log10)

    # Fallback: if LOG_KCAT_KM missing but kcat AND km both log10-able,
    # compute it via subtraction.
    both_have = df["LOG_KCAT"].notna() & df["LOG_KM"].notna()
    missing_kkm = df["LOG_KCAT_KM"].isna()
    df.loc[both_have & missing_kkm, "LOG_KCAT_KM"] = (
        df.loc[both_have & missing_kkm, "LOG_KCAT"] - df.loc[both_have & missing_kkm, "LOG_KM"]
    )

    # Report coverage
    print(f"  LOG_KCAT    non-null: {df.LOG_KCAT.notna().sum():6d} / {len(df)} ({100*df.LOG_KCAT.notna().mean():.1f}%)")
    print(f"  LOG_KM      non-null: {df.LOG_KM.notna().sum():6d} / {len(df)} ({100*df.LOG_KM.notna().mean():.1f}%)")
    print(f"  LOG_KCAT_KM non-null: {df.LOG_KCAT_KM.notna().sum():6d} / {len(df)} ({100*df.LOG_KCAT_KM.notna().mean():.1f}%)")
    print(f"  all three non-null:   {(df.LOG_KCAT.notna() & df.LOG_KM.notna() & df.LOG_KCAT_KM.notna()).sum()}")

    # ── 2. Murcko scaffold ─────────────────────────────────────────────
    print("[task] Murcko scaffold per unique SMILES ...")
    tqdm.pandas(desc="scaffold")
    scaffold_info = smi_df["SMILES"].progress_apply(murcko_scaffold_id)
    smi_df["SCAFFOLD_SMILES"] = [t[0] for t in scaffold_info]
    smi_df["SCAFFOLD_ID"] = [t[1] for t in scaffold_info]

    n_unique_scaffolds = smi_df[smi_df["SCAFFOLD_ID"].str.startswith("SCF_")]["SCAFFOLD_ID"].nunique()
    print(f"  {len(smi_df)} SMILES → {n_unique_scaffolds} unique scaffolds")
    special = smi_df.loc[smi_df["SCAFFOLD_ID"].isin(
        ["SCF_NONE", "SCF_INVALID", "SCF_ACYCLIC", "SCF_FAIL"]), "SCAFFOLD_ID"].value_counts()
    if len(special):
        print("  special categories:")
        for cat, cnt in special.items():
            print(f"    {cat}: {cnt}")

    # Scaffold size distribution (for how useful the scaffold split will be)
    reg = smi_df[smi_df["SCAFFOLD_ID"].str.startswith("SCF_") & ~smi_df["SCAFFOLD_ID"].isin(
        ["SCF_NONE", "SCF_INVALID", "SCF_ACYCLIC", "SCF_FAIL"])]
    sc = reg["SCAFFOLD_ID"].value_counts()
    if len(sc):
        print(f"  regular scaffold size: min={sc.min()}, median={sc.median():.0f}, "
              f"max={sc.max()}, mean={sc.mean():.1f}")
        print(f"  singleton scaffolds (freq=1): {(sc == 1).sum()} ({100*(sc == 1).mean():.1f}%)")

    # ── 3. Join scaffold into final_data ───────────────────────────────
    sm_id_to_scaffold = dict(zip(smi_df["SMI_ID"], smi_df["SCAFFOLD_ID"]))
    df["SCAFFOLD_ID"] = df["SMI_ID"].map(sm_id_to_scaffold).fillna("SCF_UNK")

    # ── Save ───────────────────────────────────────────────────────────
    Path(args.out_final).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_final, index=False)
    smi_df.to_csv(args.out_smi, index=False)
    print()
    print(f"[save] {args.out_final}  ({len(df)} rows, +3 log targets +SCAFFOLD_ID)")
    print(f"[save] {args.out_smi}    ({len(smi_df)} substrates, +SCAFFOLD_ID / SCAFFOLD_SMILES)")


if __name__ == "__main__":
    main()
