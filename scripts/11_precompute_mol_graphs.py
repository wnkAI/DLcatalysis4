"""
Precompute PyG molecular graphs for substrate SMILES.

Schema (matches src/model/substrate_gnn.py):
  x         (N, 10) long: [atomic_num, degree, charge+3, hybrid, num_h,
                           chirality, aromatic, in_ring, donor, acceptor]
  edge_index(2, E) long
  edge_attr (E, 4)  long: [bond_type, stereo, conjugated, in_ring]

Output: torch.save(dict[SMI_ID -> torch_geometric.data.Data], mol_graphs.pt)

Usage:
  python scripts/11_precompute_mol_graphs.py \
    --smi_csv data/processed/smi.csv \
    --out     data/processed/mol_graphs.pt
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdchem import BondType, HybridizationType, ChiralType, BondStereo
from torch_geometric.data import Data
from tqdm import tqdm
import os

RDLogger.DisableLog("rdApp.*")


HYBRID_TO_IDX = {
    HybridizationType.SP: 0,
    HybridizationType.SP2: 1,
    HybridizationType.SP3: 2,
    HybridizationType.SP3D: 3,
    HybridizationType.SP3D2: 4,
    HybridizationType.UNSPECIFIED: 2,  # fallback to SP3
    HybridizationType.S: 2,
    HybridizationType.OTHER: 2,
}

CHIRAL_TO_IDX = {
    ChiralType.CHI_UNSPECIFIED: 0,
    ChiralType.CHI_TETRAHEDRAL_CW: 1,    # R (assigned CW → R approx)
    ChiralType.CHI_TETRAHEDRAL_CCW: 2,   # S
    ChiralType.CHI_OTHER: 3,
}

BOND_TYPE_TO_IDX = {
    BondType.SINGLE: 0,
    BondType.DOUBLE: 1,
    BondType.TRIPLE: 2,
    BondType.AROMATIC: 3,
}

STEREO_TO_IDX = {
    BondStereo.STEREONONE: 0,
    BondStereo.STEREOZ: 1,
    BondStereo.STEREOE: 2,
    BondStereo.STEREOCIS: 1,
    BondStereo.STEREOTRANS: 2,
    BondStereo.STEREOANY: 3,
}


def _safe_idx(mapping, key, default=0):
    return mapping.get(key, default)


def build_chem_feature_factory():
    """Factory for donor/acceptor features."""
    fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    return ChemicalFeatures.BuildFeatureFactory(fdef)


def atom_features(mol, donors, acceptors):
    """Return (N, 10) long tensor."""
    feats = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        a = [
            min(atom.GetAtomicNum(), 119),             # 0: atomic_num ∈ [0, 119]
            min(atom.GetDegree(), 6),                  # 1: degree ∈ [0, 6]
            min(max(atom.GetFormalCharge() + 3, 0), 6),# 2: charge+3 ∈ [0, 6]
            _safe_idx(HYBRID_TO_IDX, atom.GetHybridization(), 2),
            min(atom.GetTotalNumHs(), 5),              # 4: num_h ∈ [0, 5]
            _safe_idx(CHIRAL_TO_IDX, atom.GetChiralTag(), 0),
            int(atom.GetIsAromatic()),                 # 6: aromatic
            int(atom.IsInRing()),                      # 7: in_ring
            int(idx in donors),                        # 8: donor
            int(idx in acceptors),                     # 9: acceptor
        ]
        feats.append(a)
    return torch.tensor(feats, dtype=torch.long)


def bond_features(mol):
    """Return (edge_index[2,E], edge_attr[E,4])."""
    src, dst, attrs = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = _safe_idx(BOND_TYPE_TO_IDX, bond.GetBondType(), 4)  # 4 = other
        st = _safe_idx(STEREO_TO_IDX, bond.GetStereo(), 0)
        conj = int(bond.GetIsConjugated())
        ring = int(bond.IsInRing())
        # Add both directions for undirected graph
        for a, b in [(i, j), (j, i)]:
            src.append(a); dst.append(b)
            attrs.append([bt, st, conj, ring])
    if not src:  # isolated atom (rare: single-atom SMILES)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 4), dtype=torch.long)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(attrs, dtype=torch.long)
    return edge_index, edge_attr


def _generate_3d_coords(mol, max_attempts: int = 3, seed: int = 42):
    """Generate 3D conformer via RDKit ETKDG → MMFF optimize. Return (N, 3) or None."""
    try:
        from rdkit.Chem import AllChem
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        for _ in range(max_attempts):
            cid = AllChem.EmbedMolecule(mol_h, params)
            if cid >= 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol_h, confId=cid)
                except Exception:
                    pass
                conf = mol_h.GetConformer(cid)
                # Strip Hs back, align to original heavy-atom ordering
                mol_no_h = Chem.RemoveHs(mol_h)
                n_heavy = mol.GetNumAtoms()
                if mol_no_h.GetNumAtoms() != n_heavy:
                    return None
                coords = np.array([
                    list(mol_no_h.GetConformer().GetAtomPosition(i))
                    for i in range(n_heavy)
                ], dtype=np.float32)
                return coords
            params.randomSeed += 1
    except Exception:
        pass
    return None


def smiles_to_graph(smi, factory, compute_3d: bool = True):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    donors, acceptors = set(), set()
    try:
        for feat in factory.GetFeaturesForMol(mol):
            if feat.GetFamily() == "Donor":
                donors.update(feat.GetAtomIds())
            elif feat.GetFamily() == "Acceptor":
                acceptors.update(feat.GetAtomIds())
    except Exception:
        pass
    x = atom_features(mol, donors, acceptors)
    edge_index, edge_attr = bond_features(mol)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if compute_3d:
        import numpy as np  # local import to keep function self-contained
        coords = _generate_3d_coords(mol)
        if coords is not None and coords.shape[0] == x.size(0):
            data.xyz = torch.tensor(coords, dtype=torch.float32)
        else:
            # Fallback: zero coords (mark invalid via xyz_valid flag)
            data.xyz = torch.zeros((x.size(0), 3), dtype=torch.float32)
            data.xyz_valid = False
        if not hasattr(data, "xyz_valid"):
            data.xyz_valid = True
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smi_csv", default="E:/AImodel/DLcatalysis4.0/data/processed/smi.csv")
    ap.add_argument("--out", default="E:/AImodel/DLcatalysis4.0/data/processed/mol_graphs.pt")
    ap.add_argument("--id_col", default="SMI_ID")
    ap.add_argument("--smi_col", default="SMILES")
    ap.add_argument("--no_3d", action="store_true",
                    help="skip RDKit 3D conformer generation (2D-only graphs)")
    args = ap.parse_args()

    df = pd.read_csv(args.smi_csv)
    print(f"[load] {len(df)} SMILES from {args.smi_csv}")
    factory = build_chem_feature_factory()
    compute_3d = not args.no_3d

    out = {}
    n_ok = 0
    n_fail = 0
    n_3d_ok = 0
    n_3d_fail = 0
    fail_list = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="graphs"):
        sid = row[args.id_col]
        smi = row[args.smi_col]
        try:
            g = smiles_to_graph(smi, factory, compute_3d=compute_3d)
            if g is None or g.x.size(0) == 0:
                n_fail += 1
                fail_list.append(sid)
                continue
            out[sid] = g
            n_ok += 1
            if compute_3d:
                if getattr(g, "xyz_valid", False):
                    n_3d_ok += 1
                else:
                    n_3d_fail += 1
        except Exception as e:
            n_fail += 1
            fail_list.append(sid)

    print(f"[done] {n_ok} ok, {n_fail} failed")
    if compute_3d:
        print(f"       3D conformer ok={n_3d_ok}, fallback (zeros)={n_3d_fail}")
    if fail_list[:10]:
        print(f"  sample failures: {fail_list[:10]}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"[save] {args.out}  ({n_ok} entries)")

    if out:
        k = next(iter(out))
        g = out[k]
        extra = ""
        if hasattr(g, "xyz"):
            extra = f", xyz={tuple(g.xyz.shape)}, xyz_valid={getattr(g, 'xyz_valid', 'n/a')}"
        print(f"[sanity] {k}: x={tuple(g.x.shape)}, edge_index={tuple(g.edge_index.shape)}, edge_attr={tuple(g.edge_attr.shape)}{extra}")


if __name__ == "__main__":
    main()
