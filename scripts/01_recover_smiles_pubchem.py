"""
Minimal PubChem SMILES recovery probe.

Given a list of substrate names missing SMILES, query PubChem REST API:
  name -> CID -> IsomericSMILES / CanonicalSMILES / InChIKey

Validate via RDKit. Report per-name result.

Usage:
  python scripts/01_recover_smiles_pubchem.py --csv <dedup.csv> --limit 200
"""
import argparse
import json
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
RATE_LIMIT_SECS = 0.21  # PubChem allows 5 req/s


def pubchem_name_to_cid(name: str, session: requests.Session) -> list[int]:
    """Return list of CIDs for a substrate name. Empty list if not found."""
    url = f"{PUBCHEM}/compound/name/{quote(name)}/cids/JSON"
    try:
        r = session.get(url, timeout=15)
    except Exception:
        return []
    if r.status_code == 404:
        return []
    if r.status_code != 200:
        return []
    try:
        data = r.json()
        return data.get("IdentifierList", {}).get("CID", []) or []
    except Exception:
        return []


def pubchem_cid_to_smiles(cid: int, session: requests.Session) -> dict:
    """Return {isomeric, canonical, inchikey, mw} for CID.
    PubChem API (2024+) renamed properties:
      IsomericSMILES -> SMILES
      CanonicalSMILES -> ConnectivitySMILES
    """
    url = (
        f"{PUBCHEM}/compound/cid/{cid}/property/"
        f"SMILES,ConnectivitySMILES,InChIKey,MolecularWeight/JSON"
    )
    try:
        r = session.get(url, timeout=15)
    except Exception:
        return {}
    if r.status_code != 200:
        return {}
    try:
        props = r.json()["PropertyTable"]["Properties"][0]
        return {
            "isomeric_smiles": props.get("SMILES"),
            "canonical_smiles": props.get("ConnectivitySMILES"),
            "inchikey": props.get("InChIKey"),
            "mw": float(props.get("MolecularWeight")) if props.get("MolecularWeight") else None,
        }
    except Exception:
        return {}


def validate_smiles(smi: str) -> dict:
    """Return dict of validation checks."""
    if not smi:
        return {"valid": False, "reason": "empty"}
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {"valid": False, "reason": "rdkit_parse_fail"}
    canon = Chem.MolToSmiles(mol)
    n_heavy = mol.GetNumHeavyAtoms()
    n_c = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
    mw = Descriptors.MolWt(mol)
    ok = (
        n_heavy >= 2
        and n_c >= 1
        and 20 <= mw <= 3000
    )
    return {
        "valid": ok,
        "canonical": canon,
        "mw": mw,
        "n_heavy": n_heavy,
        "n_c": n_c,
        "reason": "ok" if ok else f"mw={mw:.1f} n_heavy={n_heavy} n_c={n_c}",
    }


def recover_one(name: str, session: requests.Session) -> dict:
    cids = pubchem_name_to_cid(name, session)
    time.sleep(RATE_LIMIT_SECS)
    if not cids:
        return {"name": name, "status": "not_found", "n_cids": 0}
    # Take first CID (PubChem's canonical match)
    cid = cids[0]
    props = pubchem_cid_to_smiles(cid, session)
    time.sleep(RATE_LIMIT_SECS)
    if not props:
        return {"name": name, "status": "cid_no_props", "cid": cid, "n_cids": len(cids)}
    smi = props.get("isomeric_smiles") or props.get("canonical_smiles")
    val = validate_smiles(smi)
    rec = {
        "name": name,
        "status": "ok" if val["valid"] else "validation_fail",
        "cid": cid,
        "n_cids": len(cids),
        "smiles": smi,
        "canonical": val.get("canonical"),
        "inchikey": props.get("inchikey"),
        "mw": props.get("mw"),
        "n_heavy": val.get("n_heavy"),
        "validation_reason": val["reason"],
    }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with substrate+substrate_smiles columns")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--mode", choices=["top", "random"], default="top",
                    help="top: most frequent names; random: uniform sample")
    ap.add_argument("--out", default="E:/AImodel/DLcatalysis4.0/data/processed/smiles_probe.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    has_kcat = df.kcat.notna()
    has_km = df.km.notna() & (df.km > 0)
    has_kkm = df.kcat_km.notna() & (df.kcat_km > 0)
    label_ok = (has_kcat & has_km) | has_kkm
    missing = df[label_ok & df.sequence.notna() & df.substrate_smiles.isna()]
    missing = missing[missing["substrate"].notna()].copy()
    missing["sub_norm"] = missing["substrate"].astype(str).str.strip()
    unique_names = missing["sub_norm"].drop_duplicates().tolist()
    print(f"Total rows missing SMILES: {len(missing)}")
    print(f"Unique substrate names: {len(unique_names)}")
    print(f"Probing first {min(args.limit, len(unique_names))} names ...")

    freq = missing["sub_norm"].value_counts()
    if args.mode == "top":
        names_to_probe = freq.head(args.limit).index.tolist()
    elif args.mode == "random":
        import random
        random.seed(42)
        all_names = freq.index.tolist()
        names_to_probe = random.sample(all_names, min(args.limit, len(all_names)))
    else:
        raise ValueError(f"unknown mode: {args.mode}")
    print(f"Sampling mode: {args.mode}")

    session = requests.Session()
    session.headers.update({"User-Agent": "DLcatalysis4.0/SMILES-probe (research)"})

    results = []
    for name in tqdm(names_to_probe, desc="probing"):
        rec = recover_one(name, session)
        rec["row_count"] = int(freq[name])
        results.append(rec)

    out_df = pd.DataFrame(results)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    # Summary
    total_probed = len(out_df)
    n_ok = (out_df.status == "ok").sum()
    n_notfound = (out_df.status == "not_found").sum()
    n_fail = len(out_df) - n_ok - n_notfound
    rows_recoverable = out_df.loc[out_df.status == "ok", "row_count"].sum()
    rows_probed = out_df["row_count"].sum()
    print()
    print(f"=== Probe result ({total_probed} unique names) ===")
    print(f"  recovered (ok):      {n_ok:4d} / {total_probed}  ({100*n_ok/total_probed:.1f}%)")
    print(f"  not_found in PubChem: {n_notfound:4d}")
    print(f"  validation_fail:      {n_fail:4d}")
    print(f"  rows covered by probed names:     {rows_probed}")
    print(f"  rows recoverable under this pipe: {rows_recoverable} ({100*rows_recoverable/rows_probed:.1f}%)")
    print()
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
