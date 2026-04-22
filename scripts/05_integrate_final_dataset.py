"""
Integrate BRENDA + Rhea backfill + bucket classification into the final
DLcatalysis 4.0 training CSV.

Priority rules for each row:
  1. If BRENDA has substrate_smiles -> keep it (BRENDA is the direct experimental record)
  2. Else if Rhea (UniProt, EC) matches exactly one master reaction -> use Rhea's LHS as substrate
  3. Else if Rhea matches multiple, try to disambiguate via BRENDA substrate name token
     match to Rhea LHS canonical
  4. Else if substrate name is protein / polymer / generic -> drop
  5. Else mark as unresolved (keep for now but flag; decide at training time)

product_smiles: fill from BRENDA if available, else from Rhea RHS if (UniProt, EC)
matched, else leave empty.

reaction_smiles: build as substrate>>product when both sides present.

Output: data/processed/dlcat4_v1.csv with columns:
  data_id, ec_number, uniprot_id, organism, sequence, seq_length,
  kcat, kcat_unit, km, km_unit, kcat_km, kcat_km_unit,
  substrate, product,
  substrate_smiles, product_smiles, reaction_smiles,
  cofactor, active_site, binding_site, ph, temperature,
  pdb_id, alphafold_file, alphafill_file,
  is_natural,
  sub_class, bucket,
  smi_source            # 'brenda' / 'rhea_tier1' / 'rhea_tier2' / 'unresolved'
"""
import argparse
import gzip
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

RHEA_TSV_DIR = "F:/data/enzyme_mechanism_project/data/rhea/140/tsv"


def load_rhea_lookup(rhea_dir: str, target_ups: set[str]):
    p = Path(rhea_dir)

    print("[rhea] rhea-reaction-smiles.tsv ...")
    df_rxn = pd.read_csv(p / "rhea-reaction-smiles.tsv", sep="\t", names=["RHEA_ID", "SMILES"])
    rhea_id_to_smi = dict(zip(df_rxn.RHEA_ID.astype(int), df_rxn.SMILES))

    print("[rhea] rhea-directions.tsv ...")
    df_dir = pd.read_csv(p / "rhea-directions.tsv", sep="\t")
    for _, r in df_dir.iterrows():
        master, lr, rl, bi = int(r.RHEA_ID_MASTER), int(r.RHEA_ID_LR), int(r.RHEA_ID_RL), int(r.RHEA_ID_BI)
        base = (rhea_id_to_smi.get(lr) or rhea_id_to_smi.get(master)
                or rhea_id_to_smi.get(bi) or rhea_id_to_smi.get(rl))
        if base:
            for rid in (master, lr, bi):
                rhea_id_to_smi.setdefault(rid, base)
            if rl not in rhea_id_to_smi and ">>" in base:
                l, r_ = base.split(">>", 1)
                rhea_id_to_smi[rl] = f"{r_}>>{l}"

    print("[rhea] rhea2uniprot_sprot.tsv ...")
    up_to_master = defaultdict(set)
    df_up = pd.read_csv(p / "rhea2uniprot_sprot.tsv", sep="\t")
    for _, r in df_up.iterrows():
        up_to_master[str(r.ID).strip().upper()].add(int(r.MASTER_ID))
    print(f"  Swiss-Prot: {len(up_to_master)} entries")

    trembl = p / "rhea2uniprot_trembl.tsv.gz"
    if trembl.exists() and target_ups:
        print(f"[rhea] rhea2uniprot_trembl.tsv.gz (stream filter {len(target_ups)} targets)")
        with gzip.open(trembl, "rt") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                up = parts[3].strip().upper()
                if up not in target_ups:
                    continue
                try:
                    up_to_master[up].add(int(parts[2]))
                except ValueError:
                    pass
        print(f"  after TrEMBL merge: {len(up_to_master)} entries")

    print("[rhea] rhea2ec.tsv ...")
    master_to_ec = defaultdict(set)
    df_ec = pd.read_csv(p / "rhea2ec.tsv", sep="\t")
    for _, r in df_ec.iterrows():
        master_to_ec[int(r.MASTER_ID)].add(str(r.ID).strip())

    return rhea_id_to_smi, up_to_master, master_to_ec


def match_rhea(uniprot, ec, up_to_master, master_to_ec, rhea_id_to_smi):
    """Return list of (master_id, rxn_smi) for this (uniprot, ec)."""
    masters = up_to_master.get(uniprot, set())
    if not masters:
        return []
    matched = [m for m in masters if ec in master_to_ec.get(m, set())]
    if not matched:
        return []
    out = []
    for m in matched:
        smi = rhea_id_to_smi.get(m)
        if smi:
            out.append((m, smi))
    return out


def pick_best_rxn(rxn_list, brenda_sub_name):
    """Among multiple Rhea reactions, pick one.
    Currently: prefer one whose LHS contains a SMILES matching brenda_sub_name tokens
    (very loose); fall back to first.
    """
    if not rxn_list:
        return None
    if len(rxn_list) == 1:
        return rxn_list[0]
    # TODO: smarter disambiguation. For now just return first.
    return rxn_list[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="E:/AImodel/DLcatalysis4.0/data/processed/enzyme_data_seqge100.csv")
    ap.add_argument("--bucket", default="E:/AImodel/DLcatalysis4.0/data/processed/bucket_report.csv")
    ap.add_argument("--out", default="E:/AImodel/DLcatalysis4.0/data/processed/dlcat4_v1.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    bucket = pd.read_csv(args.bucket, low_memory=False)[["data_id", "sub_class", "prod_class", "bucket"]]
    df = df.merge(bucket, on="data_id", how="left")
    df["uniprot_id"] = df["uniprot_id"].astype(str).str.strip().str.upper()
    df["ec_number"] = df["ec_number"].astype(str).str.strip()
    print(f"[data] {len(df)} rows, {df.uniprot_id.nunique()} unique UniProt")

    target_ups = set(df.uniprot_id.unique())
    rhea_id_to_smi, up_to_master, master_to_ec = load_rhea_lookup(RHEA_TSV_DIR, target_ups)

    # Per-row processing
    sub_smi = df["substrate_smiles"].fillna("").astype(str).tolist()
    prod_smi = df["product_smiles"].fillna("").astype(str).tolist()
    rxn_smi = [""] * len(df)
    smi_source = [""] * len(df)

    for i, row in tqdm(df.iterrows(), total=len(df), desc="integrate"):
        has_sub = bool(sub_smi[i])
        has_prod = bool(prod_smi[i])
        sub_class = str(row.sub_class) if pd.notna(row.sub_class) else ""

        # Try to get Rhea for this row
        rxns = match_rhea(row.uniprot_id, row.ec_number, up_to_master, master_to_ec, rhea_id_to_smi)
        rhea_rec = pick_best_rxn(rxns, row.get("substrate", ""))

        if has_sub and has_prod:
            smi_source[i] = "brenda"
            rxn_smi[i] = f"{sub_smi[i]}>>{prod_smi[i]}"
        elif has_sub and not has_prod:
            smi_source[i] = "brenda_sub_only"
            if rhea_rec:
                # Fill product from Rhea if possible
                _, rxn = rhea_rec
                if ">>" in rxn:
                    _, rhs = rxn.split(">>", 1)
                    prod_smi[i] = rhs
                    rxn_smi[i] = f"{sub_smi[i]}>>{rhs}"
                    smi_source[i] = "brenda_sub_plus_rhea_prod"
        elif not has_sub and has_prod:
            # Missing substrate; try Rhea
            if rhea_rec and sub_class not in ("protein_or_polymer", "generic_entity"):
                _, rxn = rhea_rec
                if ">>" in rxn:
                    lhs, _ = rxn.split(">>", 1)
                    sub_smi[i] = lhs
                    rxn_smi[i] = f"{lhs}>>{prod_smi[i]}"
                    smi_source[i] = "rhea_sub_plus_brenda_prod"
                else:
                    smi_source[i] = "unresolved"
            else:
                smi_source[i] = "unresolved"
        else:  # neither
            if sub_class in ("protein_or_polymer", "generic_entity"):
                smi_source[i] = "drop_non_small_molecule"
            elif rhea_rec:
                _, rxn = rhea_rec
                if ">>" in rxn:
                    lhs, rhs = rxn.split(">>", 1)
                    sub_smi[i] = lhs
                    prod_smi[i] = rhs
                    rxn_smi[i] = rxn
                    smi_source[i] = "rhea_full"
                else:
                    smi_source[i] = "unresolved"
            else:
                smi_source[i] = "unresolved"

    df["substrate_smiles"] = [s if s else None for s in sub_smi]
    df["product_smiles"] = [s if s else None for s in prod_smi]
    df["reaction_smiles_merged"] = [s if s else None for s in rxn_smi]
    df["smi_source"] = smi_source

    # Summary
    print()
    print("=== Source distribution ===")
    for k, v in df["smi_source"].value_counts().items():
        print(f"  {k:35s}: {v:6d} ({100*v/len(df):.1f}%)")

    # Modelable = has substrate_smiles
    modelable = df["substrate_smiles"].notna() & (df["smi_source"] != "drop_non_small_molecule")
    print()
    print(f"Rows with substrate_smiles (modelable): {modelable.sum()}")
    print(f"  unique UniProt: {df.loc[modelable, 'uniprot_id'].nunique()}")
    print(f"  unique EC:      {df.loc[modelable, 'ec_number'].nunique()}")

    # Save modelable subset and the full (with source tags)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n[done] full record saved to {out_path}")
    df[modelable].to_csv(out_path.with_name(out_path.stem + "_modelable.csv"), index=False)
    print(f"[done] modelable subset saved to {out_path.with_name(out_path.stem + '_modelable.csv')}")


if __name__ == "__main__":
    main()
