"""
DLcatalysis 4.0 — Pocket extraction (draft v1, aligned with EnzymeCAGE).

Strategy (tier cascade, driven by structure source routing):
  Tier 1 (ligand-available): AlphaFill transplanted ligand OR docked
                              substrate pose → fixed K=32 residues by min
                              distance (CA) to any ligand heavy atom.
  Tier 2 (AF only + annotations): UniProt active/binding site residues
                                   + 1-hop CA-CA ≤ 8Å expansion → K=32.
  Tier 3 (AF only, no annotations): fpocket top-1 pocket centroid →
                                     K=32 nearest CA residues.

Output: dict[UNIPROT_ID -> torch_geometric.data.Data] saved as pockets.pt
with fields:
    residue_idx : (K,) long   — 0-based position in UniProt sequence
    aa_id       : (K,) long   — standard AA index 0..19
    ca_xyz      : (K, 3) float
    bb_xyz      : (K, 4, 3) float — N/CA/C/O
    node_s      : (K, S) float — GVP scalar features (dihedrals + ...)
    node_v      : (K, V, 3) float — GVP vector features
    edge_index  : (2, E) long
    edge_s      : (E, Es) float — RBF distance + sequence separation
    edge_v      : (E, 1, 3) float — edge orientation
    mask        : (K,) bool
    source      : str  — "ligand" / "annot" / "fpocket"
    n_residues  : int  — actual K (may be < target if protein is tiny)

This script requires:
  - AlphaFold PDB path per UniProt   (provided via CSV)
  - AlphaFill CIF + JSON per UniProt (optional)
  - Docked pose PDB per (UniProt, SMI_ID) (optional, from Vina stage)
  - fpocket binary in PATH (for Tier 3)

NOTE: fpocket Tier-3 is STUB below until fpocket-python wrapper is finalized.

Usage (server):
  python scripts/14_extract_pockets.py \
    --manifest data/processed/structure_manifest.csv \
    --out data/processed/pockets.pt \
    --k 32
"""
import argparse
from pathlib import Path
from typing import Optional, Tuple, List
import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# BioPython lives in requirements; import lazily to allow syntax-check on
# machines without biopython installed.
def _import_bio():
    from Bio.PDB import MMCIFParser, PDBParser, NeighborSearch
    return MMCIFParser, PDBParser, NeighborSearch


# ──────────────────────────────────────────────────────────────────────
# Amino acid mapping
# ──────────────────────────────────────────────────────────────────────
STANDARD_AA = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
               "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]
AA3_TO_IDX = {a: i for i, a in enumerate(STANDARD_AA)}
AA3_TO_1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q",
            "GLU":"E","GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K",
            "MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W",
            "TYR":"Y","VAL":"V"}


def load_structure(path: str):
    """Parse PDB or mmCIF into a BioPython Structure."""
    MMCIFParser, PDBParser, _ = _import_bio()
    suffix = Path(path).suffix.lower()
    if suffix == ".cif":
        parser = MMCIFParser(QUIET=True)
    elif suffix == ".pdb":
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported structure format: {path}")
    return parser.get_structure("prot", path)


def get_protein_residues(structure):
    """Return list of standard-AA residues with N/CA/C/O present."""
    out = []
    for res in structure.get_residues():
        if res.id[0] != " ":
            continue  # hetero
        if res.resname not in AA3_TO_IDX:
            continue
        if not all(atom in res for atom in ("N", "CA", "C", "O")):
            continue
        out.append(res)
    return out


def get_ligand_atoms(structure, keep_hoh=False):
    """All non-standard residue atoms (treat as ligand heavy atoms)."""
    atoms = []
    for res in structure.get_residues():
        if res.id[0] == " ":
            continue
        if not keep_hoh and res.resname in ("HOH", "H2O"):
            continue
        for atom in res.get_atoms():
            if atom.element != "H":
                atoms.append(atom)
    return atoms


# ──────────────────────────────────────────────────────────────────────
# Tier 1: ligand-based pocket (fixed K)
# ──────────────────────────────────────────────────────────────────────
def pocket_tier1_ligand(structure, k=32, probe_radius=10.0) -> List:
    """Return top-K protein residues ranked by min CA distance to any ligand atom."""
    _, _, NeighborSearch = _import_bio()
    prot_res = get_protein_residues(structure)
    if not prot_res:
        return []
    ligand_atoms = get_ligand_atoms(structure)
    if not ligand_atoms:
        return []

    # Broad neighbor search to get candidates
    all_atoms = [a for r in prot_res for a in r.get_atoms() if a.element != "H"]
    ns = NeighborSearch(all_atoms)
    candidates = set()
    for la in ligand_atoms:
        for res in ns.search(la.coord, probe_radius, level="R"):
            if res in prot_res and "CA" in res:
                candidates.add(res)

    # Rank by min CA-to-ligand-atom distance
    def min_dist(res):
        ca = np.array(res["CA"].coord)
        return min(np.linalg.norm(ca - np.array(la.coord)) for la in ligand_atoms)

    ranked = sorted(candidates, key=min_dist)
    return ranked[:k]


# ──────────────────────────────────────────────────────────────────────
# Tier 2: annotation-seeded pocket (active/binding site + 1-hop expand)
# ──────────────────────────────────────────────────────────────────────
def parse_annotation_sites(site_string: str) -> List[int]:
    """Parse UniProt binding_site / active_site string.
    Example: '47:Zn(2+)/catalytic; 68:Zn(2+)/catalytic; 201-206:NAD(+)'
    Return 1-based residue indices.
    """
    if not isinstance(site_string, str) or not site_string.strip():
        return []
    out = []
    for seg in site_string.split(";"):
        seg = seg.strip()
        if not seg or ":" not in seg:
            continue
        rng = seg.split(":", 1)[0].strip()
        if "-" in rng:
            try:
                s, e = rng.split("-", 1)
                for i in range(int(s), int(e) + 1):
                    out.append(i)
            except ValueError:
                continue
        else:
            try:
                out.append(int(rng))
            except ValueError:
                continue
    return sorted(set(out))


def pocket_tier2_annot(structure, annot_resids_1based: List[int],
                       k: int = 32, expand_radius: float = 8.0) -> List:
    """Start from annotation residues, expand via CA-CA ≤ radius, take top-K
    by "sum of distances to each seed" (ascending)."""
    _, _, NeighborSearch = _import_bio()
    prot_res = get_protein_residues(structure)
    if not prot_res or not annot_resids_1based:
        return []

    # Map 1-based author seqid -> residue
    resid_map = {res.id[1]: res for res in prot_res}
    seeds = [resid_map[i] for i in annot_resids_1based if i in resid_map and "CA" in resid_map[i]]
    if not seeds:
        return []

    # All CA atoms for neighbor search
    ca_atoms = [res["CA"] for res in prot_res]
    ns = NeighborSearch(ca_atoms)

    candidates = set(seeds)
    for seed in seeds:
        for atom in ns.search(seed["CA"].coord, expand_radius, level="A"):
            # find residue containing this CA atom
            res = atom.get_parent()
            if res in prot_res:
                candidates.add(res)

    # Rank by min CA-to-seed distance, then seed residues go first
    seed_set = set(seeds)
    def score(res):
        is_seed = 0 if res in seed_set else 1
        ca = np.array(res["CA"].coord)
        min_d = min(np.linalg.norm(ca - np.array(s["CA"].coord)) for s in seeds)
        return (is_seed, min_d)
    ranked = sorted(candidates, key=score)
    return ranked[:k]


# ──────────────────────────────────────────────────────────────────────
# Tier 3: fpocket stub (server-only; placeholder here)
# ──────────────────────────────────────────────────────────────────────
def pocket_tier3_fpocket(pdb_path: str, k: int = 32, fpocket_bin: str = "fpocket") -> List:
    """Run fpocket, take top-1 pocket center, then K nearest CA residues.
    STUB: caller must provide a pocket center via external command.
    """
    raise NotImplementedError(
        "fpocket integration not implemented in v1 script — "
        "will be added after CLI model consensus."
    )


# ──────────────────────────────────────────────────────────────────────
# GVP feature extraction (lightweight; full GVP ProteinGraphDataset comes later)
# ──────────────────────────────────────────────────────────────────────
def backbone_features_v2(residues: List,
                         annot_resids_1based: Optional[set] = None,
                         active_resids_1based: Optional[set] = None,
                         cofactor_atom_coords: Optional[np.ndarray] = None,
                         metal_atom_coords: Optional[np.ndarray] = None,
                         substrate_atom_coords: Optional[np.ndarray] = None,
                         k_nbr: int = 16) -> dict:
    """Per-residue feature bundle (v2).

    node_s (K, 26):
      20 — AA one-hot
       1 — pLDDT / 100 (AlphaFold B-factor; 1.0 if experimental PDB)
       1 — is_active_site    (from UniProt annotation)
       1 — is_binding_site   (from UniProt annotation)
       1 — is_cofactor_contact (heavy atom ≤4.0 Å to cofactor atom)
       1 — is_metal_contact    (CA ≤3.5 Å to metal atom)
       1 — min_dist_to_substrate / 10.0 (clipped; 0 + flag if no substrate pose)

    node_v (K, 2, 3):  N→CA, C→CA unit vectors

    edge_index (2, E): k-NN CA
    edge_s (E, 16): RBF distance
    edge_v (E, 1, 3): CA→CA unit vector

    Future (v3): add SASA (freesasa), secondary structure (DSSP).
    """
    K = len(residues)
    ca = np.stack([np.array(r["CA"].coord) for r in residues], axis=0)  # (K, 3)
    n  = np.stack([np.array(r["N"].coord)  for r in residues], axis=0)
    c  = np.stack([np.array(r["C"].coord)  for r in residues], axis=0)
    o  = np.stack([np.array(r["O"].coord)  for r in residues], axis=0)
    bb = np.stack([n, ca, c, o], axis=1)  # (K, 4, 3)

    aa_id = np.array([AA3_TO_IDX.get(r.resname, 0) for r in residues], dtype=np.int64)
    aa_onehot = np.eye(20, dtype=np.float32)[aa_id]

    # pLDDT from CA B-factor (AlphaFold convention); clamp to [0, 100]
    plddt = np.array([float(r["CA"].bfactor) for r in residues], dtype=np.float32)
    plddt = np.clip(plddt, 0.0, 100.0) / 100.0  # normalize

    # Annotation flags
    def _flag(ids_set):
        if not ids_set:
            return np.zeros(K, dtype=np.float32)
        return np.array([float(r.id[1] in ids_set) for r in residues], dtype=np.float32)
    is_active = _flag(active_resids_1based)
    is_binding = _flag(annot_resids_1based)

    # Cofactor / metal contact (distance-based)
    def _contact_flag(coords, cutoff):
        if coords is None or len(coords) == 0:
            return np.zeros(K, dtype=np.float32)
        flag = np.zeros(K, dtype=np.float32)
        for i, res in enumerate(residues):
            heavy = [np.array(a.coord) for a in res.get_atoms() if a.element != "H"]
            if not heavy:
                continue
            res_coords = np.stack(heavy, axis=0)  # (A_res, 3)
            dists = np.linalg.norm(res_coords[:, None, :] - coords[None, :, :], axis=-1)
            if dists.min() <= cutoff:
                flag[i] = 1.0
        return flag
    is_cof = _contact_flag(cofactor_atom_coords, 4.0)
    is_metal = _contact_flag(metal_atom_coords, 3.5)

    # min_dist_to_substrate (Cα to any substrate atom)
    if substrate_atom_coords is not None and len(substrate_atom_coords) > 0:
        dists = np.linalg.norm(ca[:, None, :] - substrate_atom_coords[None, :, :], axis=-1)
        min_dist = dists.min(axis=-1)  # (K,)
        min_dist_scaled = np.minimum(min_dist / 10.0, 2.0).astype(np.float32)  # clip at 20Å
    else:
        min_dist_scaled = np.full(K, -1.0, dtype=np.float32)  # -1 = no substrate info

    # Stack scalar features
    node_s = np.concatenate([
        aa_onehot,                          # 20
        plddt[:, None],                     # 1
        is_active[:, None],                 # 1
        is_binding[:, None],                # 1
        is_cof[:, None],                    # 1
        is_metal[:, None],                  # 1
        min_dist_scaled[:, None],           # 1
    ], axis=1).astype(np.float32)           # (K, 26)

    # Vector node: N→CA and C→CA unit vectors
    def _unit(v):
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.maximum(norm, 1e-8)
    n_ca = _unit(n - ca)
    c_ca = _unit(c - ca)
    node_v = np.stack([n_ca, c_ca], axis=1).astype(np.float32)  # (K, 2, 3)

    # Edges: k-nearest CA neighbors
    dist_mat = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    np.fill_diagonal(dist_mat, np.inf)
    k_eff = min(k_nbr, K - 1)
    nbr_idx = np.argpartition(dist_mat, kth=k_eff - 1, axis=-1)[:, :k_eff]
    src = np.repeat(np.arange(K), k_eff)
    dst = nbr_idx.flatten()
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)

    d_edge = dist_mat[src, dst]
    rbf_centers = np.linspace(2.0, 22.0, 16).astype(np.float32)
    rbf_sigma = 1.5
    edge_s = np.exp(-((d_edge[:, None] - rbf_centers[None, :]) ** 2) / (2 * rbf_sigma ** 2)).astype(np.float32)

    diff = ca[dst] - ca[src]
    edge_v = _unit(diff)[:, None, :].astype(np.float32)

    return {
        "aa_id": torch.tensor(aa_id),
        "ca_xyz": torch.tensor(ca, dtype=torch.float32),
        "bb_xyz": torch.tensor(bb, dtype=torch.float32),
        "node_s": torch.tensor(node_s),
        "node_v": torch.tensor(node_v),
        "edge_index": torch.tensor(edge_index),
        "edge_s": torch.tensor(edge_s),
        "edge_v": torch.tensor(edge_v),
    }


# Backward-compat alias
simple_backbone_features = backbone_features_v2


# ──────────────────────────────────────────────────────────────────────
# Driver (stub — full version wired to structure_manifest after server)
# ──────────────────────────────────────────────────────────────────────
def _collect_hetero_coords(structure, is_metal_fn):
    """Return (cofactor_atoms, metal_atoms) both as (A, 3) numpy arrays."""
    cof, metal = [], []
    for res in structure.get_residues():
        if res.id[0] == " ":
            continue  # protein residue
        if res.resname in ("HOH", "H2O"):
            continue
        for atom in res.get_atoms():
            if atom.element == "H":
                continue
            c = np.array(atom.coord)
            if is_metal_fn(atom.element):
                metal.append(c)
            else:
                cof.append(c)
    cof = np.stack(cof, axis=0) if cof else None
    metal = np.stack(metal, axis=0) if metal else None
    return cof, metal


METAL_ELEMENTS = {"ZN", "FE", "MG", "MN", "CA", "CU", "NI", "CO", "NA", "K", "CD", "HG"}


def extract_one(uniprot_id: str, structure_path: str, source: str,
                annot_binding: Optional[str] = None,
                annot_active: Optional[str] = None,
                k: int = 32,
                probe_radius: float = 10.0,
                annot_expand_radius: float = 8.0,
                substrate_pose_path: Optional[str] = None) -> Optional[dict]:
    """Extract pocket for one enzyme.

    Args:
        probe_radius: wide search radius for Tier-1 ligand candidates
                      (EnzymeCAGE uses 10.0 then top-K by CA distance;
                       LigandMPNN uses strict 5.0 direct cutoff).
        annot_expand_radius: CA-CA expansion radius for Tier-2 annotation seeds.
        substrate_pose_path: optional PDB/SDF of docked substrate pose.
    """
    structure = load_structure(structure_path)

    if source in ("alphafill_cofactor", "pdb_real", "docked"):
        residues = pocket_tier1_ligand(structure, k=k, probe_radius=probe_radius)
        tier = "ligand"
    elif source == "af_annotated":
        annot_ids = parse_annotation_sites(annot_active) + parse_annotation_sites(annot_binding)
        annot_ids = sorted(set(annot_ids))
        residues = pocket_tier2_annot(structure, annot_ids, k=k, expand_radius=annot_expand_radius)
        tier = "annot"
    elif source == "af_unannotated":
        residues = []
        tier = "fpocket"
    else:
        residues = []
        tier = "unknown"

    if not residues:
        return None

    # Collect cofactor / metal atoms for per-residue contact flags
    cofactor_coords, metal_coords = _collect_hetero_coords(
        structure, is_metal_fn=lambda e: e.upper() in METAL_ELEMENTS
    )

    # Annotation index sets (1-based)
    active_ids = set(parse_annotation_sites(annot_active)) if annot_active else set()
    binding_ids = set(parse_annotation_sites(annot_binding)) if annot_binding else set()

    # Substrate pose atoms (optional, for min_dist_to_substrate)
    substrate_coords = None
    if substrate_pose_path and Path(substrate_pose_path).exists():
        try:
            subs = load_structure(substrate_pose_path)
            substrate_coords = np.stack([
                np.array(a.coord) for a in subs.get_atoms() if a.element != "H"
            ], axis=0)
        except Exception:
            substrate_coords = None

    feat = backbone_features_v2(
        residues,
        annot_resids_1based=binding_ids,
        active_resids_1based=active_ids,
        cofactor_atom_coords=cofactor_coords,
        metal_atom_coords=metal_coords,
        substrate_atom_coords=substrate_coords,
        k_nbr=16,
    )
    feat["residue_idx"] = torch.tensor([r.id[1] for r in residues], dtype=torch.long)
    feat["mask"] = torch.ones(len(residues), dtype=torch.bool)
    feat["source"] = tier
    feat["n_residues"] = len(residues)
    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="CSV with UNIPROT_ID,structure_path,source,active_site,binding_site")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=32,
                    help="Fixed pocket size (EnzymeCAGE uses 32; Codex/Gemini propose 128)")
    ap.add_argument("--probe_radius", type=float, default=10.0,
                    help="Tier-1 wide-search radius for ligand-pocket candidates (Å)")
    ap.add_argument("--annot_expand_radius", type=float, default=8.0,
                    help="Tier-2 CA-CA expansion from annotation seeds (Å)")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    print(f"[load] manifest: {len(df)} structures")
    out = {}
    n_ok = 0
    n_fail = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="extract"):
        uid = row["UNIPROT_ID"]
        try:
            feat = extract_one(
                uid, row["structure_path"], row["source"],
                annot_binding=row.get("binding_site"),
                annot_active=row.get("active_site"),
                k=args.k,
                probe_radius=args.probe_radius,
                annot_expand_radius=args.annot_expand_radius,
                substrate_pose_path=row.get("substrate_pose_path"),
            )
            if feat is not None:
                out[uid] = feat
                n_ok += 1
            else:
                n_fail += 1
        except Exception as e:
            print(f"[fail] {uid}: {e}")
            n_fail += 1

    print(f"[done] ok={n_ok} fail={n_fail}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
