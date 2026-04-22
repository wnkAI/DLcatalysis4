"""
Build structure-source manifest for Phase 1 (docking + pocket extraction).

Input: data/processed/final_data.csv  (merged with dlcat4_v1_full_rxn.csv
       to get structure file columns)

Output: data/processed/structure_manifest.csv
  One row per unique UNIPROT_ID with:
    UNIPROT_ID
    source           : alphafill_cofactor | pdb_real | af_only | missing
    structure_path   : absolute or relative path on server
    alphafill_ligands: csv str (if AlphaFill)
    has_cofactor     : bool
    cofactor_names   : text
    active_site      : UniProt annotation string (raw)
    binding_site     : UniProt annotation string (raw)
    needs_docking    : bool (True for af_only and pdb_real w/o cognate ligand)
    n_reactions      : how many (SMI_ID, Y_VALUE) rows this enzyme has

Routing rule (from PLAN_v3_addendum.md):
  1. has_cofactor AND has AlphaFill  -> source=alphafill_cofactor
                                         (already has cofactor pose, may
                                          also need docking for actual substrate)
  2. has experimental PDB            -> source=pdb_real
  3. has AlphaFold only              -> source=af_only
  4. nothing                         -> source=missing

Usage:
  python scripts/15_build_structure_manifest.py \
    --full_rxn data/processed/dlcat4_v1_full_rxn.csv \
    --final_data data/processed/final_data.csv \
    --out data/processed/structure_manifest.csv \
    --af_dir /data/structures/alphafold \
    --alphafill_dir /data/structures/alphafill \
    --pdb_dir /data/structures/pdb
"""
import argparse
from pathlib import Path

import pandas as pd


def has_nonempty(x) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip()
    return len(s) > 0 and s.lower() not in ("nan", "none", "null")


def route_structure(row) -> str:
    """Priority: alphafill_cofactor > pdb_real > af_only > missing."""
    has_cof = has_nonempty(row.get("cofactor"))
    has_af_fill = has_nonempty(row.get("alphafill_file"))
    has_pdb = has_nonempty(row.get("pdb_file"))
    has_af = has_nonempty(row.get("alphafold_file"))

    if has_cof and has_af_fill:
        return "alphafill_cofactor"
    if has_pdb:
        return "pdb_real"
    if has_af:
        return "af_only"
    return "missing"


def build_structure_path(source: str, row, af_dir: str, alphafill_dir: str, pdb_dir: str) -> str:
    """Return absolute path for this source choice."""
    if source == "alphafill_cofactor":
        fn = row["alphafill_file"]
        return str(Path(alphafill_dir) / fn) if fn else ""
    if source == "pdb_real":
        fn = row["pdb_file"]
        return str(Path(pdb_dir) / fn) if fn else ""
    if source == "af_only":
        fn = row["alphafold_file"]
        return str(Path(af_dir) / fn) if fn else ""
    return ""


def needs_docking(source: str, row) -> bool:
    """AlphaFill has cofactor but NOT the substrate → still need to dock substrate.
    PDB might have co-crystal ligand that IS the substrate; but we can't tell
    cheaply, so default to needing dock.
    AF-only always needs docking.
    """
    if source in ("alphafill_cofactor", "pdb_real", "af_only"):
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_rxn", default="E:/AImodel/DLcatalysis4.0/data/processed/dlcat4_v1_full_rxn.csv")
    ap.add_argument("--final_data", default="E:/AImodel/DLcatalysis4.0/data/processed/final_data.csv")
    ap.add_argument("--out", default="E:/AImodel/DLcatalysis4.0/data/processed/structure_manifest.csv")
    ap.add_argument("--af_dir", default="/data/structures/alphafold",
                    help="server path where *.pdb/*.cif AlphaFold structures live")
    ap.add_argument("--alphafill_dir", default="/data/structures/alphafill")
    ap.add_argument("--pdb_dir", default="/data/structures/pdb")
    args = ap.parse_args()

    rxn = pd.read_csv(args.full_rxn, low_memory=False)
    final = pd.read_csv(args.final_data, low_memory=False)

    print(f"[load] full_rxn={len(rxn)}, final_data={len(final)}")

    # Deduplicate to per-enzyme level, keep first row (they all share the
    # enzyme-level metadata: pdb_file, alphafold_file, etc. don't vary per row)
    rxn["uniprot_id"] = rxn["uniprot_id"].astype(str).str.strip().str.upper()
    enzyme_meta = rxn.drop_duplicates("uniprot_id").set_index("uniprot_id")
    print(f"[unique] {len(enzyme_meta)} unique UniProt IDs")

    # Count reactions per enzyme from final_data (training targets)
    final["SEQ_ID"] = final["SEQ_ID"].astype(str).str.strip().str.upper()
    n_rxn_per_enz = final.groupby("SEQ_ID").size().to_dict()

    rows = []
    for uid, row in enzyme_meta.iterrows():
        if uid not in n_rxn_per_enz:
            continue  # enzyme has no usable kinetic label → skip
        source = route_structure(row)
        structure_path = build_structure_path(source, row,
                                              args.af_dir, args.alphafill_dir, args.pdb_dir)
        rows.append({
            "UNIPROT_ID": uid,
            "source": source,
            "structure_path": structure_path,
            "alphafill_ligands": row.get("alphafill_ligands") if pd.notna(row.get("alphafill_ligands")) else "",
            "has_cofactor": has_nonempty(row.get("cofactor")),
            "cofactor_names": row.get("cofactor") if pd.notna(row.get("cofactor")) else "",
            "active_site": row.get("active_site") if pd.notna(row.get("active_site")) else "",
            "binding_site": row.get("binding_site") if pd.notna(row.get("binding_site")) else "",
            "needs_docking": needs_docking(source, row),
            "n_reactions": n_rxn_per_enz.get(uid, 0),
        })

    mf = pd.DataFrame(rows)
    print()
    print("=== Structure source distribution ===")
    print(mf["source"].value_counts().to_string())
    print()
    print(f"has_cofactor: {mf['has_cofactor'].sum()} / {len(mf)} ({100*mf['has_cofactor'].mean():.1f}%)")
    print(f"has active_site annot:  {(mf['active_site'].str.len() > 0).sum()}")
    print(f"has binding_site annot: {(mf['binding_site'].str.len() > 0).sum()}")
    print(f"needs_docking: {mf['needs_docking'].sum()}")
    print(f"total reactions covered: {mf['n_reactions'].sum()}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    mf.to_csv(args.out, index=False)
    print(f"\n[done] {args.out}  ({len(mf)} enzymes)")


if __name__ == "__main__":
    main()
