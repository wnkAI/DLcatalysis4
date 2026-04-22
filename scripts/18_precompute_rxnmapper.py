"""
Precompute RXNMapper atom-atom mapping → substrate reaction-center mask.

For each reaction SMILES in final_data.csv:
  - Run RXNMapper to get atom-atom map
  - Identify "reaction center" atoms: substrate atoms whose bonds change
    (present in reactant but not product, or vice versa)
  - Produce per-atom binary mask aligned with substrate RDKit atom ordering

Output: dict[SMI_ID -> torch.tensor shape (n_atoms,) bool]

Install:
  pip install rxnmapper

Usage:
  python scripts/18_precompute_rxnmapper.py \
    --final_data data/processed/final_data.csv \
    --smi_csv data/processed/smi.csv \
    --out data/processed/rxn_center_mask.pt \
    --device cuda:0
"""
import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")


def get_reaction_center_atoms(mapped_rxn: str) -> dict:
    """Parse mapped rxn SMILES 'R1.R2>>P1.P2' → {atom_map_num: bool is_center}."""
    try:
        reactants, products = mapped_rxn.split(">>", 1)
    except ValueError:
        return {}

    def _bond_set(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return set(), {}
        # Build map: atom_map_num -> atom_idx
        map_to_idx = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() > 0}
        bonds = set()
        for b in mol.GetBonds():
            m1 = b.GetBeginAtom().GetAtomMapNum()
            m2 = b.GetEndAtom().GetAtomMapNum()
            if m1 > 0 and m2 > 0:
                bonds.add(tuple(sorted((m1, m2))))
        return bonds, map_to_idx

    r_bonds, _ = _bond_set(reactants)
    p_bonds, _ = _bond_set(products)
    diff = r_bonds.symmetric_difference(p_bonds)
    center_map_nums = set()
    for m1, m2 in diff:
        center_map_nums.add(m1)
        center_map_nums.add(m2)
    return center_map_nums


def map_substrate_center(rxn_smi: str, sub_smi: str, mapper) -> torch.Tensor:
    """Return bool tensor of length n_atoms marking reaction-center atoms on substrate.

    Strategy:
      1. Run RXNMapper on rxn_smi → mapped rxn
      2. Identify atom-map numbers of reaction center atoms (bonds that change)
      3. Match canonical substrate atom indices to the mapped-reactant atom_map_nums
         via substructure match (both should canonicalize to same SMILES)
    """
    try:
        result = mapper.get_attention_guided_atom_maps([rxn_smi])
        mapped = result[0]["mapped_rxn"]
    except Exception:
        # Fallback — whole substrate, no reaction center info
        mol = Chem.MolFromSmiles(sub_smi)
        return torch.zeros(mol.GetNumAtoms() if mol else 0, dtype=torch.bool)

    center_map_nums = get_reaction_center_atoms(mapped)
    if not center_map_nums:
        mol = Chem.MolFromSmiles(sub_smi)
        return torch.zeros(mol.GetNumAtoms() if mol else 0, dtype=torch.bool)

    # For substrate side: split reactants, find which reactant corresponds to substrate
    reactants = mapped.split(">>", 1)[0]
    sub_mol_canon = Chem.MolFromSmiles(sub_smi)
    if sub_mol_canon is None:
        return torch.zeros(0, dtype=torch.bool)
    sub_canon = Chem.MolToSmiles(sub_mol_canon)
    n_atoms = sub_mol_canon.GetNumAtoms()

    mask = torch.zeros(n_atoms, dtype=torch.bool)
    # Try each reactant component
    for comp in reactants.split("."):
        comp_mol = Chem.MolFromSmiles(comp)
        if comp_mol is None:
            continue
        comp_canon = Chem.MolToSmiles(
            Chem.MolFromSmiles(Chem.MolToSmiles(comp_mol, canonical=True))
        )
        if comp_canon != sub_canon:
            continue
        # Match this mapped component to canonical substrate atoms via substructure
        match = sub_mol_canon.GetSubstructMatch(comp_mol)
        if not match:
            continue
        # For each atom in mapped component, check if map num is in center
        for i, atom in enumerate(comp_mol.GetAtoms()):
            if atom.GetAtomMapNum() in center_map_nums and i < len(match):
                mask[match[i]] = True
        break
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_data", default="E:/AImodel/DLcatalysis4.0/data/processed/final_data.csv")
    ap.add_argument("--smi_csv", default="E:/AImodel/DLcatalysis4.0/data/processed/smi.csv")
    ap.add_argument("--out", default="E:/AImodel/DLcatalysis4.0/data/processed/rxn_center_mask.pt")
    args = ap.parse_args()

    print("[load] RXNMapper ...")
    from rxnmapper import RXNMapper
    mapper = RXNMapper()

    df = pd.read_csv(args.final_data, low_memory=False)
    smi_df = pd.read_csv(args.smi_csv)
    smi_map = dict(zip(smi_df["SMI_ID"], smi_df["SMILES"]))

    # For each unique SMI_ID, pick one representative reaction
    rep = df.drop_duplicates("SMI_ID")[["SMI_ID", "RXN_SMILES"]]
    print(f"[plan] {len(rep)} unique substrates with reactions to process")

    out = {}
    for _, row in tqdm(rep.iterrows(), total=len(rep), desc="rxnmapper"):
        smi_id = row["SMI_ID"]
        sub_smi = smi_map.get(smi_id)
        rxn_smi = row["RXN_SMILES"]
        if not isinstance(sub_smi, str) or not isinstance(rxn_smi, str):
            continue
        mask = map_substrate_center(rxn_smi, sub_smi, mapper)
        out[smi_id] = mask

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"[done] {len(out)} masks saved to {args.out}")

    # Stats
    n_center = sum(m.sum().item() for m in out.values() if len(m) > 0)
    n_total = sum(len(m) for m in out.values() if len(m) > 0)
    print(f"  total atoms: {n_total}, reaction-center atoms: {n_center} ({100*n_center/max(n_total,1):.1f}%)")


if __name__ == "__main__":
    main()
