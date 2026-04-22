"""
Rhea coverage probe: how many BRENDA rows can be backfilled with
substrate_smiles / product_smiles / reaction_smiles from Rhea?

Matching rule (tier cascade):
  Tier 1: (uniprot_id, ec_number) has exactly 1 Rhea master reaction
          -> direct assignment
  Tier 2: (uniprot_id, ec_number) has multiple Rhea reactions
          -> if BRENDA row has substrate_smiles, match by InChIKey of any reactant
          -> else ambiguous; leave for later
  Tier 3: (uniprot_id) has Rhea but EC mismatches
          -> flag as ec_mismatch

Output: coverage_report.csv with columns:
  row_index, uniprot_id, ec_number, source=brenda/rhea_tier1/rhea_tier2/rhea_mult/no_rhea,
  substrate_smiles, product_smiles, reaction_smiles
"""
import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm


TARGET_UPS = None  # set by main() before calling load_rhea


def load_rhea(rhea_tsv_dir: str):
    """Build three lookups:
      rhea_id_to_smiles: int -> rxn_smi
      uniprot_to_rhea:  str -> set(master_rhea_id)
      master_to_ec:     int -> set(ec_str)
    """
    tsv = Path(rhea_tsv_dir)

    # 1. reaction smiles (LR direction usually)
    print("[load] rhea-reaction-smiles.tsv ...")
    df_rxn = pd.read_csv(
        tsv / "rhea-reaction-smiles.tsv", sep="\t", names=["RHEA_ID", "SMILES"]
    )
    rhea_id_to_smi = dict(zip(df_rxn["RHEA_ID"].astype(int), df_rxn["SMILES"]))
    print(f"  {len(rhea_id_to_smi)} reactions with SMILES")

    # 2. rhea directions: MASTER <-> LR/RL/BI; SMILES stored for one direction
    # Propagate SMILES to all related IDs.
    print("[load] rhea-directions.tsv ...")
    df_dir = pd.read_csv(tsv / "rhea-directions.tsv", sep="\t")
    for _, row in df_dir.iterrows():
        master = int(row["RHEA_ID_MASTER"])
        lr = int(row["RHEA_ID_LR"])
        rl = int(row["RHEA_ID_RL"])
        bi = int(row["RHEA_ID_BI"])
        # prefer LR smiles, else any of them
        base_smi = (
            rhea_id_to_smi.get(lr)
            or rhea_id_to_smi.get(master)
            or rhea_id_to_smi.get(bi)
            or rhea_id_to_smi.get(rl)
        )
        if base_smi:
            for rid in (master, lr, bi):
                rhea_id_to_smi.setdefault(rid, base_smi)
            # RL is reverse — swap
            if rl not in rhea_id_to_smi and ">>" in base_smi:
                left, right = base_smi.split(">>", 1)
                rhea_id_to_smi[rl] = f"{right}>>{left}"
    print(f"  after direction expansion: {len(rhea_id_to_smi)} reactions with SMILES")

    # 3. UniProt -> Rhea master ids (merge Swiss-Prot + TrEMBL, filtered by target set)
    print("[load] rhea2uniprot_sprot.tsv ...")
    df_up = pd.read_csv(tsv / "rhea2uniprot_sprot.tsv", sep="\t")
    up_to_master = defaultdict(set)
    for _, row in df_up.iterrows():
        up = str(row["ID"]).strip().upper()
        master = int(row["MASTER_ID"])
        up_to_master[up].add(master)
    print(f"  Swiss-Prot: {len(up_to_master)} UniProt IDs")

    # TrEMBL is huge (40M rows); filter on the fly by target set if provided
    trembl_path = tsv / "rhea2uniprot_trembl.tsv.gz"
    if trembl_path.exists() and TARGET_UPS is not None:
        print(f"[load] rhea2uniprot_trembl.tsv.gz (streaming, filter to {len(TARGET_UPS)} targets) ...")
        import gzip
        n_trembl_kept = 0
        with gzip.open(trembl_path, "rt") as f:
            header = f.readline()  # skip header
            # columns: RHEA_ID, DIRECTION, MASTER_ID, ID
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                up = parts[3].strip().upper()
                if up not in TARGET_UPS:
                    continue
                try:
                    master = int(parts[2])
                except ValueError:
                    continue
                up_to_master[up].add(master)
                n_trembl_kept += 1
        print(f"  TrEMBL: kept {n_trembl_kept} mappings for target UniProts")
        print(f"  combined: {len(up_to_master)} UniProt IDs with Rhea mapping")

    # 4. master -> EC
    print("[load] rhea2ec.tsv ...")
    df_ec = pd.read_csv(tsv / "rhea2ec.tsv", sep="\t")
    master_to_ec = defaultdict(set)
    for _, row in df_ec.iterrows():
        master = int(row["MASTER_ID"])
        ec = str(row["ID"]).strip()
        master_to_ec[master].add(ec)
    print(f"  {len(master_to_ec)} master reactions with EC")

    return rhea_id_to_smi, up_to_master, master_to_ec


def find_rhea_for_row(uniprot, ec, up_to_master, master_to_ec):
    """Return list of matching master rhea ids for (uniprot, ec)."""
    masters = up_to_master.get(uniprot, set())
    if not masters:
        return []
    matched = [m for m in masters if ec in master_to_ec.get(m, set())]
    if matched:
        return matched
    # fallback: no EC match, return all masters for this uniprot
    return list(masters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--rhea", default="F:/data/enzyme_mechanism_project/data/rhea/140/tsv")
    ap.add_argument("--out", default="E:/AImodel/DLcatalysis4.0/data/processed/rhea_coverage.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    df["uniprot_id"] = df["uniprot_id"].astype(str).str.strip().str.upper()
    df["ec_number"] = df["ec_number"].astype(str).str.strip()

    # Build target set for TrEMBL filtering (avoid loading 40M rows)
    global TARGET_UPS
    TARGET_UPS = set(df["uniprot_id"].unique())
    print(f"[data] {len(df)} BRENDA rows, {len(TARGET_UPS)} unique UniProt IDs")

    rhea_id_to_smi, up_to_master, master_to_ec = load_rhea(args.rhea)

    results = []
    n_no_up_in_rhea = 0
    n_ec_match_one = 0
    n_ec_match_mult = 0
    n_up_but_no_ec = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="coverage"):
        up = row["uniprot_id"]
        ec = row["ec_number"]
        if up not in up_to_master:
            n_no_up_in_rhea += 1
            results.append({"idx": i, "tier": "no_rhea", "n_matches": 0})
            continue
        masters = [m for m in up_to_master[up] if ec in master_to_ec.get(m, set())]
        if len(masters) == 0:
            n_up_but_no_ec += 1
            results.append({"idx": i, "tier": "up_but_no_ec", "n_matches": 0})
            continue
        # Get all SMILES for matched masters
        rxn_smis = []
        for m in masters:
            smi = rhea_id_to_smi.get(m)
            if smi:
                rxn_smis.append(smi)
        if len(rxn_smis) == 0:
            results.append({"idx": i, "tier": "ec_match_no_smi", "n_matches": 0})
            continue
        if len(rxn_smis) == 1:
            n_ec_match_one += 1
            tier = "tier1_unique"
        else:
            n_ec_match_mult += 1
            tier = "tier2_multi"
        # Use first (will match to BRENDA substrate name later for disambiguation)
        rxn = rxn_smis[0]
        lhs, rhs = rxn.split(">>", 1) if ">>" in rxn else (rxn, "")
        results.append({
            "idx": i,
            "tier": tier,
            "n_matches": len(rxn_smis),
            "rxn_smi": rxn,
            "substrate_smi_from_rhea": lhs,
            "product_smi_from_rhea": rhs,
        })

    out = pd.DataFrame(results)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print()
    print("=== Rhea coverage on BRENDA ===")
    print(f"  total BRENDA rows:        {len(df)}")
    print(f"  uniprot NOT in Rhea:      {n_no_up_in_rhea}  ({100*n_no_up_in_rhea/len(df):.1f}%)")
    print(f"  uniprot in Rhea, EC mismatch: {n_up_but_no_ec}  ({100*n_up_but_no_ec/len(df):.1f}%)")
    print(f"  (uniprot, ec) -> 1 reaction:  {n_ec_match_one}  ({100*n_ec_match_one/len(df):.1f}%)  [Tier 1, direct use]")
    print(f"  (uniprot, ec) -> multi rxns:  {n_ec_match_mult}  ({100*n_ec_match_mult/len(df):.1f}%)  [Tier 2, disambig needed]")
    print()
    print(f"  total recoverable via Rhea:   {n_ec_match_one + n_ec_match_mult}  ({100*(n_ec_match_one+n_ec_match_mult)/len(df):.1f}%)")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
