"""
Batch substrate docking with AutoDock-Vina-GPU-2.1.

Prerequisites (server-side):
  - autodock-vina-gpu-2.1 compiled with CUDA (binary on PATH as vina-gpu)
  - openbabel (obabel) for pdbqt conversion
  - meeko (Python) for ligand prep: pip install meeko
  - Reduce for adding hydrogens (optional; openbabel can do it)
  - fpocket (conda install -c bioconda fpocket) for box detection fallback

Box definition priority (per enzyme):
  1. AlphaFill transplanted ligand centroid (if source=alphafill_cofactor)
  2. PDB hetatm ligand centroid (if source=pdb_real and hetatm present)
  3. fpocket top-1 pocket centroid (all other cases)

Box size: fixed 22³ Å. Exhaustiveness=16, num_modes=5. Keep top-1 pose.

Output per (uniprot_id, smi_id):
  {dock_dir}/{uid}_{smi_id}.pdbqt  — docked pose
  {dock_dir}/{uid}_{smi_id}.log    — Vina log (score, affinity)
  manifest CSV updated with {uid}_{smi_id} -> pose_path, vina_score

Usage:
  python scripts/20_run_vina_gpu.py \
    --final_data data/processed/final_data.csv \
    --manifest data/processed/structure_manifest.csv \
    --smi_csv data/processed/smi.csv \
    --out_dir /data/wnk/docking_poses \
    --vina_gpu /usr/local/bin/vina-gpu \
    --workers 4
"""
import argparse
import json
import os
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_pdb_hetatm_centroid(pdb_path: str) -> np.ndarray:
    """Return centroid of first HETATM group in the PDB; None if none."""
    coords = []
    resname = None
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("HETATM"):
                rn = line[17:20].strip()
                if rn in ("HOH", "H2O"):
                    continue
                if resname is None:
                    resname = rn
                if rn != resname:
                    break  # first ligand only
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                coords.append([x, y, z])
    if not coords:
        return None
    return np.array(coords).mean(axis=0)


def run_fpocket(pdb_path: str, fpocket_bin: str = "fpocket") -> np.ndarray:
    """Run fpocket, return top-1 pocket centroid; None on failure."""
    try:
        subprocess.run([fpocket_bin, "-f", pdb_path],
                       capture_output=True, check=True, timeout=300)
        base = Path(pdb_path).with_suffix("")
        out_dir = Path(f"{base}_out")
        pockets_file = out_dir / "pockets" / "pocket1_atm.pdb"
        if not pockets_file.exists():
            return None
        coords = []
        with open(pockets_file) as f:
            for line in f:
                if line.startswith("ATOM"):
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                    coords.append([x, y, z])
        return np.array(coords).mean(axis=0) if coords else None
    except Exception:
        return None


def prep_ligand_pdbqt(smi: str, out_path: str) -> bool:
    """SMILES → 3D conformer → pdbqt via RDKit + Meeko."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from meeko import MoleculePreparation

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            return False
        AllChem.MMFFOptimizeMolecule(mol)
        prep = MoleculePreparation()
        prep.prepare(mol)
        pdbqt = prep.write_pdbqt_string()
        with open(out_path, "w") as f:
            f.write(pdbqt)
        return True
    except Exception as e:
        print(f"[ligand prep fail] {smi[:40]}: {e}")
        return False


def prep_receptor_pdbqt(pdb_path: str, out_path: str) -> bool:
    """Protein PDB → pdbqt via obabel (strip hetatm first)."""
    try:
        # Strip hetatm to apo form for docking
        apo_path = str(Path(out_path).with_suffix(".apo.pdb"))
        with open(pdb_path, "r") as fin, open(apo_path, "w") as fout:
            for line in fin:
                if line.startswith("ATOM") or line.startswith("TER") or line.startswith("END"):
                    fout.write(line)
        # obabel conversion (handles H addition and charge assignment)
        subprocess.run(
            ["obabel", apo_path, "-O", out_path, "-xr", "-h"],
            capture_output=True, check=True, timeout=120,
        )
        return Path(out_path).exists()
    except Exception as e:
        print(f"[receptor prep fail] {pdb_path}: {e}")
        return False


def run_vina_gpu(vina_gpu: str, receptor: str, ligand: str,
                 box_center: np.ndarray, box_size: float,
                 out_pdbqt: str, log_path: str,
                 exhaustiveness: int = 16, num_modes: int = 5) -> dict:
    """Invoke vina-gpu binary; return {"score": float, "ok": bool}."""
    cmd = [
        vina_gpu,
        "--receptor", receptor,
        "--ligand", ligand,
        "--center_x", f"{box_center[0]:.2f}",
        "--center_y", f"{box_center[1]:.2f}",
        "--center_z", f"{box_center[2]:.2f}",
        "--size_x", f"{box_size:.1f}",
        "--size_y", f"{box_size:.1f}",
        "--size_z", f"{box_size:.1f}",
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
        "--out", out_pdbqt,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        with open(log_path, "w") as f:
            f.write(proc.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(proc.stderr)
        # Parse score from output (first mode affinity)
        score = None
        for line in proc.stdout.splitlines():
            if line.strip().startswith("1 "):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        score = float(parts[1])
                        break
                    except ValueError:
                        continue
        return {"score": score, "ok": proc.returncode == 0 and Path(out_pdbqt).exists()}
    except Exception as e:
        return {"score": None, "ok": False, "error": str(e)}


def dock_one(task):
    """Worker: dock one (uniprot, smi_id) pair."""
    uid = task["uid"]
    smi_id = task["smi_id"]
    smi = task["smi"]
    receptor_pdb = task["receptor_pdb"]
    box_center = np.array(task["box_center"])
    out_dir = Path(task["out_dir"])
    vina_gpu = task["vina_gpu"]

    pair_prefix = f"{uid}_{smi_id}"
    lig_pdbqt = out_dir / f"{pair_prefix}.lig.pdbqt"
    rec_pdbqt = out_dir / f"{uid}.rec.pdbqt"  # cached per receptor
    out_pdbqt = out_dir / f"{pair_prefix}.pdbqt"
    log_path = out_dir / f"{pair_prefix}.log"

    # Skip if already done
    if out_pdbqt.exists():
        return {"pair": pair_prefix, "score": None, "ok": True, "cached": True}

    # Prep ligand (per pair)
    if not prep_ligand_pdbqt(smi, str(lig_pdbqt)):
        return {"pair": pair_prefix, "ok": False, "error": "ligand_prep"}

    # Prep receptor (cache per enzyme)
    if not rec_pdbqt.exists():
        if not prep_receptor_pdbqt(receptor_pdb, str(rec_pdbqt)):
            return {"pair": pair_prefix, "ok": False, "error": "receptor_prep"}

    # Run Vina-GPU
    result = run_vina_gpu(
        vina_gpu, str(rec_pdbqt), str(lig_pdbqt),
        box_center, box_size=22.0,
        out_pdbqt=str(out_pdbqt), log_path=str(log_path),
    )
    result["pair"] = pair_prefix
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_data", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--smi_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--vina_gpu", default="vina-gpu")
    ap.add_argument("--fpocket_bin", default="fpocket")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel dock jobs (careful with GPU; Vina-GPU uses 1 GPU)")
    ap.add_argument("--limit", type=int, default=None, help="debug: limit N pairs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tables
    final = pd.read_csv(args.final_data, low_memory=False)
    mani = pd.read_csv(args.manifest)
    smi_df = pd.read_csv(args.smi_csv)
    smi_map = dict(zip(smi_df["SMI_ID"], smi_df["SMILES"]))
    mani_map = dict(zip(mani["UNIPROT_ID"], mani.to_dict("records")))

    # Build unique (uniprot, smi_id) task list
    pairs = final[["SEQ_ID", "SMI_ID"]].drop_duplicates()
    if args.limit:
        pairs = pairs.head(args.limit)
    print(f"[dock] {len(pairs)} unique (enzyme, substrate) pairs")

    # Compute box center per enzyme (cached)
    print("[box] resolving box centers per enzyme ...")
    box_centers = {}
    for uid, mrow in tqdm(mani_map.items(), desc="box"):
        source = mrow["source"]
        rec_path = mrow["structure_path"]
        if not Path(rec_path).exists():
            continue
        center = None
        if source == "alphafill_cofactor":
            # centroid of any HETATM group that isn't water
            center = parse_pdb_hetatm_centroid(rec_path)
        elif source == "pdb_real":
            center = parse_pdb_hetatm_centroid(rec_path)
            if center is None:
                center = run_fpocket(rec_path, args.fpocket_bin)
        else:  # af_only
            center = run_fpocket(rec_path, args.fpocket_bin)
        if center is not None:
            box_centers[uid] = center.tolist()
    print(f"[box] {len(box_centers)} enzymes have valid box centers")

    # Build task list
    tasks = []
    for _, row in pairs.iterrows():
        uid = row["SEQ_ID"]
        smi_id = row["SMI_ID"]
        if uid not in box_centers or uid not in mani_map:
            continue
        if smi_id not in smi_map:
            continue
        tasks.append({
            "uid": uid,
            "smi_id": smi_id,
            "smi": smi_map[smi_id],
            "receptor_pdb": mani_map[uid]["structure_path"],
            "box_center": box_centers[uid],
            "out_dir": str(out_dir),
            "vina_gpu": args.vina_gpu,
        })
    print(f"[dock] {len(tasks)} tasks after filter")

    # Execute (serial if --workers=1, else parallel)
    results = []
    if args.workers == 1:
        for t in tqdm(tasks, desc="dock"):
            results.append(dock_one(t))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(dock_one, t) for t in tasks]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="dock"):
                results.append(fut.result())

    # Save dock manifest
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / "docking_manifest.csv", index=False)
    print()
    print(f"[done] total={len(res_df)}, ok={res_df['ok'].sum()}, "
          f"failed={(~res_df['ok'].astype(bool)).sum()}")
    print(f"[save] {out_dir}/docking_manifest.csv")


if __name__ == "__main__":
    main()
