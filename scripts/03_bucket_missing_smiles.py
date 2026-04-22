"""
Bucket analysis of BRENDA rows by substrate/product SMILES availability,
following consultation advice:

Step 1: classify each row into:
  - has_sub_smi / no_sub_smi
  - has_prod_smi / no_prod_smi

Step 2: for rows missing substrate_smiles, classify the substrate name into:
  - small_molecule_candidate  (looks like a real compound name)
  - protein_or_polymer         (cytochrome c, starch, tRNA, ...)
  - generic_entity             (donor, acceptor, substrate, reduced ...)
  - assay_shorthand            (Z-Phe-Arg-AMC, pNP-X, DCPIP, ABTS, ...)
  - unclassified

Output: bucket_report.csv + on-screen summary.
"""
import re
from pathlib import Path

import pandas as pd

IN_CSV = "E:/AImodel/DLcatalysis4.0/data/processed/enzyme_data_seqge100.csv"
OUT_CSV = "E:/AImodel/DLcatalysis4.0/data/processed/bucket_report.csv"

# --- Non-small-molecule keyword rules ---

PROTEIN_POLYMER_KEYWORDS = [
    "cytochrome", "ferredoxin", "ferricytochrome", "ferrocytochrome",
    "hemoglobin", "myoglobin", "apoenzyme", "holoenzyme",
    "tRNA", "mRNA", "rRNA", "siRNA", "RNA",
    "ubiquitin", "thioredoxin",
    "starch", "inulin", "glycogen", "cellulose", "chitosan", "chitin",
    "pectin", "xylan", "arabinan", "galactan", "dextran",
    "lipopolysaccharide", "peptidoglycan",
    "-oligosaccharide", "oligosaccharide",
    "casein", "collagen", "albumin", "lysozyme",
    "protein ", "nucleic acid",
    "histone",
]

GENERIC_ENTITY_KEYWORDS = [
    r"\bdonor\b", r"\bacceptor\b",
    r"\breduced acceptor\b", r"\boxidized acceptor\b",
    r"\balcohol\b$", r"\baldehyde\b$",
    r"^a ", r"^an ",
    r"\bpolymer\b", r"\bmixture\b",
]

ASSAY_SHORTHAND_PATTERNS = [
    r"amido-4-methylcoumarin",
    r"-AMC$",
    r"-4-nitroanilide",
    r"-pNA$",
    r"-p-nitrophenol",
    r"-pNP",
    r"benzyloxycarbonyl",
    r"^Z-[A-Z]",
    r"^Bz-",
    r"^Ac-[A-Z]",
    r"^formyl-",
    r"indophenol",
    r"azino-bis",
    r"ferrocenium",
    r"Methylumbelliferyl",
    r"4-methylumbelliferyl",
    r"MUF-",
    r"\bDCPIP\b",
    r"\bABTS\b",
    r"\bBSA\b",
]


def classify_name(name):
    if not isinstance(name, str) or not name.strip():
        return "empty"
    n = name.strip()
    nl = n.lower()

    for kw in PROTEIN_POLYMER_KEYWORDS:
        if kw.lower() in nl:
            return "protein_or_polymer"

    for pat in GENERIC_ENTITY_KEYWORDS:
        if re.search(pat, nl):
            return "generic_entity"

    for pat in ASSAY_SHORTHAND_PATTERNS:
        if re.search(pat, n):
            return "assay_shorthand"

    return "small_molecule_candidate"


def main():
    df = pd.read_csv(IN_CSV, low_memory=False)
    print(f"Total rows: {len(df)}")

    has_sub = df["substrate_smiles"].notna()
    has_prod = df["product_smiles"].notna()
    has_sub_name = df["substrate"].notna()
    has_prod_name = df["product"].notna()

    print()
    print("=== SMILES × 文本覆盖分桶 ===")
    buckets = pd.crosstab(has_sub, has_prod, rownames=["has_sub_smi"], colnames=["has_prod_smi"])
    print(buckets)
    print()

    # Classify substrate names
    df["sub_class"] = df["substrate"].apply(classify_name)
    df["prod_class"] = df["product"].apply(classify_name)

    # Rows missing substrate_smiles
    miss_sub = df[~has_sub].copy()
    print(f"缺 substrate_smiles 的行: {len(miss_sub)}")
    print()
    print("按 substrate 名字分类:")
    sub_cls = miss_sub["sub_class"].value_counts()
    for k, v in sub_cls.items():
        print(f"  {k:30s}: {v:5d}  ({100*v/len(miss_sub):.1f}%)")
    print()

    # Rows missing product_smiles (but have substrate_smiles)
    miss_prod_only = df[has_sub & ~has_prod].copy()
    print(f"有 substrate_smiles 但缺 product_smiles 的行: {len(miss_prod_only)}")
    print()
    print("按 product 名字分类:")
    prod_cls = miss_prod_only["prod_class"].value_counts()
    for k, v in prod_cls.items():
        print(f"  {k:30s}: {v:5d}  ({100*v/len(miss_prod_only):.1f}%)")
    print()

    # Combined buckets: final decision
    def final_bucket(row):
        hs = pd.notna(row["substrate_smiles"])
        hp = pd.notna(row["product_smiles"])
        sc = row["sub_class"]
        if hs and hp:
            return "A_both_smiles_present"
        if hs and not hp:
            return "B_sub_only"
        if not hs and hp:
            return "C_prod_only"
        # neither
        return f"D_neither_{sc}"

    df["bucket"] = df.apply(final_bucket, axis=1)
    print("=== 最终分桶（用于后续 resolver 决策）===")
    for k, v in df["bucket"].value_counts().items():
        print(f"  {k:45s}: {v:5d}  ({100*v/len(df):.1f}%)")

    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df[["data_id","ec_number","uniprot_id","substrate","product",
        "substrate_smiles","product_smiles","sub_class","prod_class","bucket"]].to_csv(
        OUT_CSV, index=False
    )
    print(f"\nSaved to {OUT_CSV}")


if __name__ == "__main__":
    main()
