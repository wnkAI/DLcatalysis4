"""
DLcatalysis 4.0 — Pair-specific docking + pocket extraction driver.

Three phases:
  Phase 1 (serial): prep receptor PDBQT + box center for each unique enzyme.
  Phase 2 (serial): prep ligand PDBQT for each unique SMILES.
  Phase 3 (parallel, N workers): dock each (uniprot_id, smi_id) pair with
                                  QVina2; extract 8Å pocket from docked pose.
  Phase 4 (serial): aggregate per-pair .pt files into pockets_pair.pt.

Output file:
  {out_dir}/receptors/{uniprot_id}.pdbqt
  {out_dir}/receptors/{uniprot_id}.box.json
  {out_dir}/ligands/{smi_id}.pdbqt
  {out_dir}/poses/{uniprot_id}_{smi_id}.pdbqt
  {out_dir}/logs/{uniprot_id}_{smi_id}.log
  {out_dir}/pockets/{uniprot_id}_{smi_id}.pt
  {out_dir}/status/{uniprot_id}_{smi_id}.json  (per-pair resume state)
  {out_dir}/run_status.csv
  pockets_pair.pt  (final aggregated dict)

Usage (server):
  python scripts/24_pair_pipeline.py \\
    --pairs data/processed/final_data.csv \\
    --smi data/processed/smi.csv \\
    --seq data/processed/seq.csv \\
    --manifest data/processed/structure_manifest.csv \\
    --annotations data/processed/enzyme_annotations.pt \\
    --out-dir data/processed/pair_docking \\
    --pockets-out data/processed/pockets_pair.pt \\
    --tool qvina2 \\
    --binary /usr/local/bin/qvina2.1_linux_x86_64 \\
    --workers 16

Pilot run (1000 pairs, ~6h):
  add --limit 1000 --pockets-out data/processed/pockets_pair_pilot.pt

EnzymeCAGE-style extraction:
  radius = 8.0 Å
  K = 32 (pad with mask if fewer)
  active_site + binding_site flags in node_s (via backbone_features_v2)
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Make scripts/14 importable as a module
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from importlib.util import spec_from_file_location, module_from_spec


def _import_sibling(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_pock = _import_sibling("pocket_extract", _HERE / "14_extract_pockets.py")
_dock = _import_sibling("docking", _HERE / "20_run_vina_gpu.py")


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════
def pair_seed(global_seed: int, uid: str, smi_id: str) -> int:
    h = hashlib.sha1(f"{global_seed}:{uid}:{smi_id}".encode()).hexdigest()[:8]
    return int(h, 16) % (2**31)


def atomic_write_bytes(path: Path, data: bytes):
    """Write via tmpfile + rename (crash-safe)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch_save(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def parse_annotation_sites_safe(site_string) -> set:
    """Parse annotation_sites via scripts/14 helper; return empty set on None."""
    if site_string is None:
        return set()
    try:
        return set(_pock.parse_annotation_sites(site_string))
    except Exception:
        return set()


# ══════════════════════════════════════════════════════════════════════
# Phase 1 & 2: serial prep
# ══════════════════════════════════════════════════════════════════════
def build_box_center(manifest_row: dict, pdb_path: str) -> np.ndarray:
    """Box center cascade (EnzymeCAGE-aligned, simplified):
      1. UniProt active_site centroid
      2. UniProt binding_site centroid
      3. protein geometric center (CA mean)
    """
    from Bio.PDB import PDBParser, MMCIFParser
    active = parse_annotation_sites_safe(manifest_row.get("active_site"))
    binding = parse_annotation_sites_safe(manifest_row.get("binding_site"))

    parser = PDBParser(QUIET=True) if pdb_path.endswith(".pdb") else MMCIFParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)

    # Get CA atoms per residue
    res_by_id = {}
    for res in structure.get_residues():
        if res.id[0] != " ":
            continue
        if "CA" not in res:
            continue
        res_by_id[res.id[1]] = np.array(res["CA"].coord)

    # Tier 1: active_site
    if active:
        coords = [res_by_id[i] for i in active if i in res_by_id]
        if coords:
            return np.mean(np.stack(coords), axis=0)

    # Tier 2: binding_site
    if binding:
        coords = [res_by_id[i] for i in binding if i in res_by_id]
        if coords:
            return np.mean(np.stack(coords), axis=0)

    # Tier 3: geometric center
    if res_by_id:
        return np.mean(np.stack(list(res_by_id.values())), axis=0)
    raise ValueError(f"No residues parseable in {pdb_path}")


def phase1_prep_receptors(mani_df: pd.DataFrame, out_dir: Path) -> dict:
    """Serial: prep receptor PDBQT + box.json for each unique enzyme.
    Returns dict[uid -> {"pdbqt": str, "box_center": [x,y,z]}].
    """
    rec_dir = out_dir / "receptors"
    rec_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for _, row in tqdm(mani_df.iterrows(), total=len(mani_df), desc="phase1 receptor"):
        uid = str(row["UNIPROT_ID"]).strip().upper()
        pdb_path = row["structure_path"]
        if not isinstance(pdb_path, str) or not Path(pdb_path).exists():
            continue
        pdbqt_path = rec_dir / f"{uid}.pdbqt"
        box_path = rec_dir / f"{uid}.box.json"

        if not pdbqt_path.exists():
            ok = _dock.prep_receptor_pdbqt(pdb_path, str(pdbqt_path))
            if not ok:
                continue
        if not box_path.exists():
            try:
                box_center = build_box_center(row.to_dict(), pdb_path)
                with open(box_path, "w") as f:
                    json.dump({"center": box_center.tolist()}, f)
            except Exception as e:
                print(f"[phase1] box fail {uid}: {e}")
                continue

        with open(box_path) as f:
            box_center = json.load(f)["center"]
        out[uid] = {"pdbqt": str(pdbqt_path), "box_center": box_center,
                    "ref_pdb": pdb_path,
                    "active_site": row.get("active_site"),
                    "binding_site": row.get("binding_site")}
    return out


def phase2_prep_ligands(smi_df: pd.DataFrame, out_dir: Path) -> dict:
    """Serial: prep ligand PDBQT for each unique SMILES."""
    lig_dir = out_dir / "ligands"
    lig_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for _, row in tqdm(smi_df.iterrows(), total=len(smi_df), desc="phase2 ligand"):
        sid = str(row["SMI_ID"])
        smi = str(row["SMILES"])
        lig_path = lig_dir / f"{sid}.pdbqt"
        if not lig_path.exists():
            if not _dock.prep_ligand_pdbqt(smi, str(lig_path)):
                continue
        out[sid] = str(lig_path)
    return out


# ══════════════════════════════════════════════════════════════════════
# Phase 3: per-pair dock + pocket extract worker
# ══════════════════════════════════════════════════════════════════════
def process_pair(task: dict) -> dict:
    """Worker function for ProcessPoolExecutor.

    task keys: uid, smi_id, rec_info, lig_path, out_dir, tool, binary,
               exhaustiveness, box_size, global_seed, radius, k, k_nbr.
    """
    uid = task["uid"]
    smi_id = task["smi_id"]
    rec = task["rec_info"]
    lig_path = task["lig_path"]
    out_dir = Path(task["out_dir"])
    radius = task.get("radius", 8.0)
    k = task.get("k", 32)
    k_nbr = task.get("k_nbr", 16)

    pose_path = out_dir / "poses" / f"{uid}_{smi_id}.pdbqt"
    log_path = out_dir / "logs" / f"{uid}_{smi_id}.log"
    pocket_path = out_dir / "pockets" / f"{uid}_{smi_id}.pt"
    status_path = out_dir / "status" / f"{uid}_{smi_id}.json"

    # Resume: if pocket already saved + validates, skip
    if pocket_path.exists():
        try:
            _ = torch.load(pocket_path, weights_only=False)
            return {"uid": uid, "smi_id": smi_id, "status": "cached"}
        except Exception:
            pocket_path.unlink()  # corrupt, re-run

    # Step 1: QVina2 dock
    seed = pair_seed(task["global_seed"], uid, smi_id)
    box_center = np.array(rec["box_center"])
    if task["tool"] == "qvina2":
        dock_result = _dock.run_qvina2(
            task["binary"], rec["pdbqt"], lig_path,
            box_center, box_size=task["box_size"],
            out_pdbqt=str(pose_path), log_path=str(log_path),
            exhaustiveness=task["exhaustiveness"], cpu=1, seed=seed,
            timeout=task.get("timeout", 60),
        )
    elif task["tool"] == "vina_gpu":
        dock_result = _dock.run_vina_gpu(
            task["binary"], rec["pdbqt"], lig_path,
            box_center, box_size=task["box_size"],
            out_pdbqt=str(pose_path), log_path=str(log_path),
            exhaustiveness=task["exhaustiveness"],
        )
    else:
        return {"uid": uid, "smi_id": smi_id, "status": "unknown_tool",
                "error": task["tool"]}

    if not dock_result.get("ok"):
        status = {"uid": uid, "smi_id": smi_id, "status": "dock_fail",
                  "error": dock_result.get("error", "qvina2 returned non-ok")}
        atomic_write_bytes(status_path, json.dumps(status).encode())
        return status

    # Step 2: extract pocket from pose
    active_set = parse_annotation_sites_safe(rec.get("active_site"))
    binding_set = parse_annotation_sites_safe(rec.get("binding_site"))

    try:
        pocket = _pock.extract_pocket_from_docked_pose(
            protein_path=rec["ref_pdb"],
            pose_pdbqt_path=str(pose_path),
            uniprot_active_sites_1based=active_set,
            uniprot_binding_sites_1based=binding_set,
            k=k,
            radius=radius,
            k_nbr=k_nbr,
        )
    except Exception as e:
        status = {"uid": uid, "smi_id": smi_id, "status": "extract_fail",
                  "error": f"{type(e).__name__}: {e}"}
        atomic_write_bytes(status_path, json.dumps(status).encode())
        return status

    if pocket is None:
        status = {"uid": uid, "smi_id": smi_id, "status": "pocket_invalid",
                  "dock_score": dock_result.get("score")}
        atomic_write_bytes(status_path, json.dumps(status).encode())
        return status

    # Enrich and save
    pocket["dock_score"] = dock_result.get("score")
    pocket["pose_path"] = str(pose_path)
    atomic_torch_save(pocket_path, pocket)

    status = {"uid": uid, "smi_id": smi_id, "status": "ok",
              "dock_score": dock_result.get("score"),
              "n_residues": pocket["n_residues"]}
    atomic_write_bytes(status_path, json.dumps(status).encode())
    return status


# ══════════════════════════════════════════════════════════════════════
# Phase 4: aggregate
# ══════════════════════════════════════════════════════════════════════
def phase4_aggregate(out_dir: Path, pockets_out: Path):
    """Load all per-pair .pt files → single dict keyed by (uid, smi_id)."""
    pocket_dir = out_dir / "pockets"
    files = sorted(pocket_dir.glob("*.pt"))
    print(f"[phase4] aggregating {len(files)} pocket files")
    agg = {}
    corrupt = 0
    for p in tqdm(files, desc="aggregate"):
        try:
            d = torch.load(p, weights_only=False)
        except Exception:
            corrupt += 1
            continue
        stem = p.stem  # uid_smiId
        try:
            uid, smi_id = stem.split("_", 1)
        except ValueError:
            continue
        agg[(uid, smi_id)] = d
    pockets_out.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(pockets_out, agg)
    print(f"[phase4] wrote {len(agg)} pair pockets → {pockets_out}  "
          f"({corrupt} corrupt files skipped)")
    return agg


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="final_data.csv")
    ap.add_argument("--smi", required=True, help="smi.csv")
    ap.add_argument("--seq", required=True, help="seq.csv")
    ap.add_argument("--manifest", required=True, help="structure_manifest.csv")
    ap.add_argument("--annotations", default=None, help="enzyme_annotations.pt (optional)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pockets-out", required=True, help="pockets_pair.pt output path")
    ap.add_argument("--tool", choices=["qvina2", "vina_gpu"], default="qvina2")
    ap.add_argument("--binary", default=None)
    ap.add_argument("--exhaustiveness", type=int, default=8)
    ap.add_argument("--box-size", type=float, default=22.0)
    ap.add_argument("--radius", type=float, default=8.0, help="pocket cutoff Å")
    ap.add_argument("--k", type=int, default=32, help="fixed pocket size")
    ap.add_argument("--k-nbr", type=int, default=16, help="k-NN edges")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=60, help="per-pair QVina2 timeout (s)")
    ap.add_argument("--global-seed", type=int, default=20260423)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N pairs")
    ap.add_argument("--skip-phase-1-2", action="store_true",
                    help="assume receptors/ligands already prepped")
    ap.add_argument("--only-aggregate", action="store_true",
                    help="skip docking; just aggregate existing pocket .pt files")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    for sub in ("receptors", "ligands", "poses", "logs", "pockets", "status"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    if args.binary is None:
        args.binary = "qvina2.1" if args.tool == "qvina2" else "vina-gpu"

    if args.only_aggregate:
        phase4_aggregate(out_dir, Path(args.pockets_out))
        return

    # Load data
    print(f"[load] pairs: {args.pairs}")
    pairs_df = pd.read_csv(args.pairs, low_memory=False)
    pairs_df["SEQ_ID"] = pairs_df["SEQ_ID"].astype(str).str.strip().str.upper()
    pairs_df["SMI_ID"] = pairs_df["SMI_ID"].astype(str)
    unique_pairs = pairs_df[["SEQ_ID", "SMI_ID"]].drop_duplicates().reset_index(drop=True)
    if args.limit:
        unique_pairs = unique_pairs.head(args.limit)
    print(f"[load] {len(unique_pairs)} unique (enzyme, substrate) pairs")

    print(f"[load] smi: {args.smi}")
    smi_df = pd.read_csv(args.smi)

    print(f"[load] manifest: {args.manifest}")
    mani_df = pd.read_csv(args.manifest)
    mani_df["UNIPROT_ID"] = mani_df["UNIPROT_ID"].astype(str).str.strip().str.upper()

    # Restrict to enzymes used in the (limited) pair list
    needed_uids = set(unique_pairs["SEQ_ID"])
    mani_df_sub = mani_df[mani_df["UNIPROT_ID"].isin(needed_uids)].copy()
    needed_smi = set(unique_pairs["SMI_ID"])
    smi_df_sub = smi_df[smi_df["SMI_ID"].isin(needed_smi)].copy()
    print(f"[load] {len(mani_df_sub)} unique enzymes, {len(smi_df_sub)} unique SMILES to prep")

    # ── Phase 1 + 2 (serial) ──────────────────────────────────────────
    if not args.skip_phase_1_2:
        t0 = time.time()
        rec_info = phase1_prep_receptors(mani_df_sub, out_dir)
        print(f"[phase1] {len(rec_info)}/{len(mani_df_sub)} receptors prepped "
              f"in {time.time()-t0:.0f}s")

        t0 = time.time()
        lig_info = phase2_prep_ligands(smi_df_sub, out_dir)
        print(f"[phase2] {len(lig_info)}/{len(smi_df_sub)} ligands prepped "
              f"in {time.time()-t0:.0f}s")
    else:
        # Reload from cache
        rec_info = {}
        for _, row in mani_df_sub.iterrows():
            uid = row["UNIPROT_ID"]
            pdbqt = out_dir / "receptors" / f"{uid}.pdbqt"
            box_path = out_dir / "receptors" / f"{uid}.box.json"
            if pdbqt.exists() and box_path.exists():
                with open(box_path) as f:
                    box_center = json.load(f)["center"]
                rec_info[uid] = {
                    "pdbqt": str(pdbqt),
                    "box_center": box_center,
                    "ref_pdb": row["structure_path"],
                    "active_site": row.get("active_site"),
                    "binding_site": row.get("binding_site"),
                }
        lig_info = {}
        for _, row in smi_df_sub.iterrows():
            sid = str(row["SMI_ID"])
            lp = out_dir / "ligands" / f"{sid}.pdbqt"
            if lp.exists():
                lig_info[sid] = str(lp)
        print(f"[cache] {len(rec_info)} receptors / {len(lig_info)} ligands cached")

    # ── Phase 3 (parallel) ────────────────────────────────────────────
    tasks = []
    for _, row in unique_pairs.iterrows():
        uid = row["SEQ_ID"]
        smi_id = row["SMI_ID"]
        if uid not in rec_info or smi_id not in lig_info:
            continue
        tasks.append({
            "uid": uid,
            "smi_id": smi_id,
            "rec_info": rec_info[uid],
            "lig_path": lig_info[smi_id],
            "out_dir": str(out_dir),
            "tool": args.tool,
            "binary": args.binary,
            "exhaustiveness": args.exhaustiveness,
            "box_size": args.box_size,
            "radius": args.radius,
            "k": args.k,
            "k_nbr": args.k_nbr,
            "global_seed": args.global_seed,
            "timeout": args.timeout,
        })
    print(f"[phase3] {len(tasks)} tasks to run with {args.workers} workers")

    # Run
    results = []
    if args.workers == 1:
        for t in tqdm(tasks, desc="dock+extract"):
            results.append(process_pair(t))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process_pair, t) for t in tasks]
            for f in tqdm(as_completed(futures), total=len(futures), desc="dock+extract"):
                results.append(f.result())

    # Run-status CSV
    status_df = pd.DataFrame(results)
    status_csv = out_dir / "run_status.csv"
    status_df.to_csv(status_csv, index=False)
    print(f"[phase3] wrote {status_csv}")
    print(status_df["status"].value_counts().to_string())

    # ── Phase 4: aggregate ───────────────────────────────────────────
    phase4_aggregate(out_dir, Path(args.pockets_out))


if __name__ == "__main__":
    main()
