"""
Prepare model input tables from dlcat4_v1_full_rxn.csv (25,518 rows).

Produces:
  data/processed/seq.csv            — unique (SEQ_ID, SEQUENCE, UNIPROT_ID)
  data/processed/smi.csv            — unique (SMI_ID, SMILES) — canonical
  data/processed/prod_smi.csv       — unique (SMI_ID, SMILES) for products
  data/processed/final_data.csv     — per-row target table:
                                      DATA_ID, SEQ_ID, SMI_ID, PROD_SMI_ID,
                                      EC_NUMBER, EC_IDS(4-tuple), KCAT, KM, KCAT_KM,
                                      Y_VALUE=log10(kcat_km), PH, TEMP
  data/processed/folds/fold_{0..9}.csv  — 10-fold random split (CD-HIT upgrade later)

Target convention:
  Y_VALUE = log10(kcat/Km) in M^-1 s^-1
  If kcat_km field present → use directly
  Else kcat[s^-1] / km[M] → kcat/Km[M^-1 s^-1] → log10

Usage:
  python scripts/10_prepare_model_inputs.py
"""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")


def smi_id(smi: str) -> str:
    """Stable 10-char ID from canonical SMILES."""
    return "SMI_" + hashlib.sha1(smi.encode("utf-8")).hexdigest()[:10].upper()


def canonicalize(smi: str) -> str | None:
    if not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def parse_ec(ec_str: str) -> tuple[int, int, int, int]:
    """Return 4-tuple of EC levels. Unknown levels → 0."""
    parts = (ec_str or "").split(".")
    out = []
    for i in range(4):
        if i < len(parts):
            try:
                out.append(int(parts[i]))
            except ValueError:
                out.append(0)
        else:
            out.append(0)
    return tuple(out)


def compute_y(row) -> float | None:
    """Return log10(kcat/Km) in M^-1 s^-1.
    Preference order:
      1. kcat_km field (if unit == M^-1*s^-1)
      2. kcat/km computed (kcat s^-1 / km M → M^-1 s^-1)
    """
    kkm = row.get("kcat_km")
    kkm_unit = str(row.get("kcat_km_unit", "")).strip()
    if pd.notna(kkm) and kkm > 0 and kkm_unit == "M^-1*s^-1":
        return float(np.log10(kkm))
    kcat = row.get("kcat")
    km = row.get("km")
    kcat_unit = str(row.get("kcat_unit", "")).strip()
    km_unit = str(row.get("km_unit", "")).strip()
    if (pd.notna(kcat) and pd.notna(km) and km > 0
            and kcat_unit == "s^-1" and km_unit == "M"):
        return float(np.log10(float(kcat) / float(km)))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="E:/AImodel/DLcatalysis4.0/data/processed/dlcat4_v1_full_rxn.csv")
    ap.add_argument("--outdir", default="E:/AImodel/DLcatalysis4.0/data/processed")
    ap.add_argument("--y_clip_low", type=float, default=-2.0,
                    help="Clip log10(kcat/Km) below this value")
    ap.add_argument("--y_clip_high", type=float, default=10.0,
                    help="Clip above")
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "folds").mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"  {len(df)} rows")

    # ── Canonicalize SMILES ──────────────────────────────────────────────
    print("[canonicalize] substrate + product SMILES via RDKit ...")
    df["substrate_canon"] = df["substrate_smiles"].astype(str).map(canonicalize)
    df["product_canon"] = df["product_smiles"].astype(str).map(canonicalize)
    # Drop rows where canonicalization failed on substrate (shouldn't happen, but safe)
    n0 = len(df)
    df = df[df["substrate_canon"].notna()].copy()
    print(f"  after canonicalize drop: {n0} → {len(df)}")

    # ── Compute Y_VALUE ──────────────────────────────────────────────────
    print("[target] compute Y_VALUE = log10(kcat/Km) ...")
    tqdm.pandas(desc="target")
    df["Y_VALUE"] = df.progress_apply(compute_y, axis=1)
    n_before = len(df)
    df = df[df["Y_VALUE"].notna()].copy()
    print(f"  with valid Y_VALUE: {n_before} → {len(df)}")

    # Clip Y_VALUE range
    n_extreme = ((df["Y_VALUE"] < args.y_clip_low) | (df["Y_VALUE"] > args.y_clip_high)).sum()
    print(f"  log10(kcat/Km) outside [{args.y_clip_low}, {args.y_clip_high}]: {n_extreme} → clipped")
    df["Y_VALUE"] = df["Y_VALUE"].clip(lower=args.y_clip_low, upper=args.y_clip_high)

    # ── Build unique sequence table ─────────────────────────────────────
    print("[seq] building unique sequence table ...")
    seq_df = df[["uniprot_id", "sequence"]].dropna().drop_duplicates("uniprot_id").reset_index(drop=True)
    seq_df = seq_df.rename(columns={"uniprot_id": "UNIPROT_ID", "sequence": "SEQUENCE"})
    seq_df["SEQ_ID"] = seq_df["UNIPROT_ID"]  # use UniProt ID as SEQ_ID
    seq_df = seq_df[["SEQ_ID", "UNIPROT_ID", "SEQUENCE"]]
    print(f"  unique enzymes: {len(seq_df)}")

    # ── Build unique substrate SMILES table ─────────────────────────────
    print("[smi] building unique substrate SMILES table ...")
    smi_tab = (df[["substrate_canon"]]
               .dropna().drop_duplicates()
               .rename(columns={"substrate_canon": "SMILES"})
               .reset_index(drop=True))
    smi_tab["SMI_ID"] = smi_tab["SMILES"].map(smi_id)
    smi_tab = smi_tab[["SMI_ID", "SMILES"]]
    # Deduplicate SMI_ID (hash collision: should be 0)
    assert smi_tab["SMI_ID"].is_unique, "SMI_ID hash collision"
    print(f"  unique substrate SMILES: {len(smi_tab)}")

    # ── Build unique product SMILES table ────────────────────────────────
    print("[smi] building unique product SMILES table ...")
    prod_tab = (df[["product_canon"]]
                .dropna().drop_duplicates()
                .rename(columns={"product_canon": "SMILES"})
                .reset_index(drop=True))
    prod_tab["SMI_ID"] = prod_tab["SMILES"].map(smi_id)
    prod_tab = prod_tab[["SMI_ID", "SMILES"]]
    assert prod_tab["SMI_ID"].is_unique, "product SMI_ID hash collision"
    print(f"  unique product SMILES: {len(prod_tab)}")

    # ── Build final target table ────────────────────────────────────────
    print("[final] building target table ...")
    smi_id_map = dict(zip(smi_tab["SMILES"], smi_tab["SMI_ID"]))
    prod_id_map = dict(zip(prod_tab["SMILES"], prod_tab["SMI_ID"]))

    df["SEQ_ID"] = df["uniprot_id"]
    df["SMI_ID"] = df["substrate_canon"].map(smi_id_map)
    df["PROD_SMI_ID"] = df["product_canon"].map(prod_id_map)

    # EC parsing
    ec_tuples = df["ec_number"].astype(str).apply(parse_ec)
    df["EC_1"] = ec_tuples.str[0]
    df["EC_2"] = ec_tuples.str[1]
    df["EC_3"] = ec_tuples.str[2]
    df["EC_4"] = ec_tuples.str[3]

    final = df[[
        "data_id", "SEQ_ID", "SMI_ID", "PROD_SMI_ID",
        "ec_number", "EC_1", "EC_2", "EC_3", "EC_4",
        "kcat", "km", "kcat_km", "Y_VALUE",
        "ph", "temperature",
        "reaction_smiles_merged", "smi_source",
    ]].rename(columns={
        "data_id": "DATA_ID",
        "ec_number": "EC_NUMBER",
        "kcat": "KCAT",
        "km": "KM",
        "kcat_km": "KCAT_KM",
        "ph": "PH",
        "temperature": "TEMP",
        "reaction_smiles_merged": "RXN_SMILES",
        "smi_source": "SMI_SOURCE",
    })

    # Drop rows without matched IDs (shouldn't happen after earlier filter)
    n_before = len(final)
    final = final[final["SEQ_ID"].notna() & final["SMI_ID"].notna()].copy()
    print(f"  target rows: {n_before} → {len(final)}")
    print(f"  Y_VALUE stats: mean={final.Y_VALUE.mean():.2f} std={final.Y_VALUE.std():.2f} "
          f"min={final.Y_VALUE.min():.2f} max={final.Y_VALUE.max():.2f}")

    # ── 10-fold random split (enzyme-grouped) ────────────────────────────
    # Keep same enzyme in same fold to approximate CD-HIT behavior until real CD-HIT is set up
    print(f"[split] enzyme-grouped random {args.n_folds}-fold ...")
    rng = np.random.RandomState(args.seed)
    unique_seqs = final["SEQ_ID"].unique()
    rng.shuffle(unique_seqs)
    seq_to_fold = {s: i % args.n_folds for i, s in enumerate(unique_seqs)}
    final["FOLD"] = final["SEQ_ID"].map(seq_to_fold)

    # ── Write outputs ────────────────────────────────────────────────────
    seq_df.to_csv(outdir / "seq.csv", index=False)
    smi_tab.to_csv(outdir / "smi.csv", index=False)
    prod_tab.to_csv(outdir / "prod_smi.csv", index=False)
    final.to_csv(outdir / "final_data.csv", index=False)

    for fold in range(args.n_folds):
        fold_df = final[final["FOLD"] == fold]
        fold_df.to_csv(outdir / "folds" / f"fold_{fold}.csv", index=False)
        print(f"  fold_{fold}: {len(fold_df)} rows, {fold_df.SEQ_ID.nunique()} enzymes")

    print()
    print("[done]")
    print(f"  seq.csv         : {len(seq_df)}")
    print(f"  smi.csv         : {len(smi_tab)}")
    print(f"  prod_smi.csv    : {len(prod_tab)}")
    print(f"  final_data.csv  : {len(final)}")
    print(f"  folds/fold_*    : {args.n_folds} files in {outdir / 'folds'}")


if __name__ == "__main__":
    main()
