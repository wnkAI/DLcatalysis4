"""
Layered substrate-name → SMILES resolver (replacement for `bioservices`,
which doesn't build on Python 3.14).

Cascade (TurNuP-style + consultation advice):
  Tier 0: Normalize name (lowercase, α/β, D/L, strip 'racemic ', etc.)
  Tier 1: ChEBI REST   (the biochem authority, name/synonym -> ChEBI ID -> SMILES)
  Tier 2: PubChem name search -> CID -> IsomericSMILES
  Tier 3: PubChem synonym match -> CID -> IsomericSMILES  (catches shorthands)
  Validation: RDKit parse + MW [20, 3000] + heavy_atoms [2, 250] + stereo check

Output columns per name:
  name, normalized, smiles, inchikey, mw, source, tier, confidence, reason
"""
import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CHEBI_WS = "https://www.ebi.ac.uk/webservices/chebi/2.0/test/getLiteEntity"
RATE_SLEEP = 0.21  # PubChem 5 req/s

STEREO_MARKERS = re.compile(r"\b(R|S|D|L|E|Z|cis|trans|alpha|beta)[\s-]", re.IGNORECASE)

STOP_PREFIXES = ["racemic ", "rac-", "(±)-", "(+/-)-", "(+)-", "(-)-"]


def normalize(name: str) -> str:
    if not isinstance(name, str):
        return ""
    n = name.strip()
    # Greek -> ascii
    n = n.replace("α", "alpha").replace("β", "beta").replace("γ", "gamma").replace("δ", "delta")
    n = n.replace("Α", "alpha").replace("Β", "beta")
    # Strip racemic markers (we track stereo separately)
    low = n.lower()
    for p in STOP_PREFIXES:
        if low.startswith(p):
            n = n[len(p):]
            break
    # Collapse spaces
    n = re.sub(r"\s+", " ", n).strip()
    return n


def has_stereo_hint(name: str) -> bool:
    return bool(STEREO_MARKERS.search(name or ""))


def mol_ok(smi: str) -> dict:
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return {"ok": False, "reason": "rdkit_parse_fail"}
    n_heavy = mol.GetNumHeavyAtoms()
    n_c = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
    mw = Descriptors.MolWt(mol)
    charge = Chem.GetFormalCharge(mol)
    has_stereo = any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms()) or \
                 any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in mol.GetBonds())
    ok = (2 <= n_heavy <= 250) and (20 <= mw <= 3000) and n_c >= 1 and abs(charge) <= 6
    return {
        "ok": ok,
        "canonical": Chem.MolToSmiles(mol),
        "inchikey": Chem.MolToInchiKey(mol) or "",
        "mw": mw,
        "n_heavy": n_heavy,
        "charge": charge,
        "has_stereo": has_stereo,
        "reason": "ok" if ok else f"mw={mw:.1f}/heavy={n_heavy}/C={n_c}/ch={charge}",
    }


def pubchem_name_cid(name: str, session) -> list[int]:
    url = f"{PUBCHEM}/compound/name/{quote(name, safe='')}/cids/JSON"
    try:
        r = session.get(url, timeout=20)
        if r.status_code == 200:
            return r.json().get("IdentifierList", {}).get("CID", []) or []
    except Exception:
        pass
    return []


def pubchem_cid_props(cid: int, session) -> dict:
    url = f"{PUBCHEM}/compound/cid/{cid}/property/SMILES,ConnectivitySMILES,InChIKey,MolecularWeight/JSON"
    try:
        r = session.get(url, timeout=20)
        if r.status_code == 200:
            p = r.json()["PropertyTable"]["Properties"][0]
            return {
                "smiles": p.get("SMILES") or p.get("ConnectivitySMILES"),
                "inchikey": p.get("InChIKey"),
                "mw": float(p.get("MolecularWeight")) if p.get("MolecularWeight") else None,
            }
    except Exception:
        pass
    return {}


def pubchem_synonym_to_cid(name: str, session) -> list[int]:
    """Search PubChem synonym table, returns CIDs whose synonyms include name."""
    url = f"{PUBCHEM}/compound/name/{quote(name, safe='')}/xrefs/RegistryID/JSON"
    # This endpoint actually just gets CIDs again; synonyms require a POST search.
    # Use autocomplete/synonym endpoint instead:
    url = f"{PUBCHEM}/compound/name/{quote(name, safe='')}/synonyms/JSON"
    try:
        r = session.get(url, timeout=20)
        if r.status_code == 200:
            info = r.json().get("InformationList", {}).get("Information", [])
            return [int(x["CID"]) for x in info if "CID" in x]
    except Exception:
        pass
    return []


def chebi_name_to_id(name: str, session) -> list[int]:
    """ChEBI REST: getLiteEntity?search=<name>&maximumResults=5&stars=ALL"""
    url = f"{CHEBI_WS}?search={quote(name)}&maximumResults=5&stars=ALL"
    try:
        r = session.get(url, timeout=20)
        if r.status_code != 200:
            return []
        # Response is SOAP/XML; quick regex for chebiId
        ids = re.findall(r"<chebiId>(CHEBI:\d+)</chebiId>", r.text)
        return ids
    except Exception:
        return []


def chebi_id_to_smiles(chebi_id: str, session) -> dict:
    """ChEBI getCompleteEntity -> look for smiles in response."""
    url = f"https://www.ebi.ac.uk/webservices/chebi/2.0/test/getCompleteEntity?chebiId={quote(chebi_id)}"
    try:
        r = session.get(url, timeout=20)
        if r.status_code != 200:
            return {}
        # Pull out <smiles> block
        m = re.search(r"<smiles>(.*?)</smiles>", r.text, re.DOTALL)
        if m:
            return {"smiles": m.group(1).strip()}
    except Exception:
        pass
    return {}


def resolve_one(raw_name: str, session) -> dict:
    """Layered cascade; returns record dict."""
    rec = {
        "name": raw_name,
        "normalized": normalize(raw_name),
        "smiles": None,
        "canonical": None,
        "inchikey": None,
        "mw": None,
        "tier": None,
        "source": None,
        "confidence": "D_unresolved",
        "stereo_hint": has_stereo_hint(raw_name),
        "reason": "",
    }
    norm = rec["normalized"]
    if not norm:
        rec["reason"] = "empty_after_norm"
        return rec

    # Tier 1: ChEBI
    chebi_ids = chebi_name_to_id(norm, session)
    time.sleep(RATE_SLEEP)
    for cid in chebi_ids[:2]:
        smi_data = chebi_id_to_smiles(cid, session)
        time.sleep(RATE_SLEEP)
        smi = smi_data.get("smiles")
        if not smi:
            continue
        val = mol_ok(smi)
        if val["ok"]:
            rec.update({
                "smiles": smi,
                "canonical": val["canonical"],
                "inchikey": val["inchikey"],
                "mw": val["mw"],
                "tier": 1,
                "source": f"chebi:{cid}",
                "confidence": "A_high" if (not rec["stereo_hint"] or val["has_stereo"]) else "B_stereo_missing",
                "reason": val["reason"],
            })
            return rec

    # Tier 2: PubChem name
    cids = pubchem_name_cid(norm, session)
    time.sleep(RATE_SLEEP)
    for cid in cids[:2]:
        props = pubchem_cid_props(cid, session)
        time.sleep(RATE_SLEEP)
        smi = props.get("smiles")
        if not smi:
            continue
        val = mol_ok(smi)
        if val["ok"]:
            rec.update({
                "smiles": smi,
                "canonical": val["canonical"],
                "inchikey": val["inchikey"] or props.get("inchikey"),
                "mw": val["mw"],
                "tier": 2,
                "source": f"pubchem:{cid}",
                "confidence": "A_high" if (not rec["stereo_hint"] or val["has_stereo"]) else "B_stereo_missing",
                "reason": val["reason"],
            })
            return rec

    # Tier 3: PubChem synonym search
    syn_cids = pubchem_synonym_to_cid(norm, session)
    time.sleep(RATE_SLEEP)
    for cid in syn_cids[:2]:
        props = pubchem_cid_props(cid, session)
        time.sleep(RATE_SLEEP)
        smi = props.get("smiles")
        if not smi:
            continue
        val = mol_ok(smi)
        if val["ok"]:
            rec.update({
                "smiles": smi,
                "canonical": val["canonical"],
                "inchikey": val["inchikey"] or props.get("inchikey"),
                "mw": val["mw"],
                "tier": 3,
                "source": f"pubchem_syn:{cid}",
                "confidence": "B_synonym" if (not rec["stereo_hint"] or val["has_stereo"]) else "C_synonym_stereo_missing",
                "reason": val["reason"],
            })
            return rec

    rec["reason"] = "all_tiers_failed"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket_csv", default="E:/AImodel/DLcatalysis4.0/data/processed/bucket_report.csv")
    ap.add_argument("--limit", type=int, default=200, help="probe first N small_molecule_candidate unique names")
    ap.add_argument("--out", default="E:/AImodel/DLcatalysis4.0/data/processed/resolver_probe.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.bucket_csv, low_memory=False)
    # Focus on rows missing substrate_smiles, small_molecule_candidate class
    mask = df["substrate_smiles"].isna() & (df["sub_class"] == "small_molecule_candidate")
    miss = df[mask].copy()
    miss["sub_norm"] = miss["substrate"].fillna("").astype(str).str.strip()
    freq = miss["sub_norm"].value_counts()
    names = freq.head(args.limit).index.tolist()
    print(f"Probing {len(names)} unique small-molecule names (freq-sorted)")

    session = requests.Session()
    session.headers.update({"User-Agent": "DLcatalysis4.0 SMILES resolver"})

    results = []
    for n in tqdm(names, desc="resolving"):
        r = resolve_one(n, session)
        r["row_count"] = int(freq[n])
        results.append(r)

    out = pd.DataFrame(results)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    total = len(out)
    n_resolved = (out.smiles.notna()).sum()
    n_a = (out.confidence == "A_high").sum()
    n_b = out.confidence.str.startswith("B").sum()
    n_c = out.confidence.str.startswith("C").sum()
    n_d = (out.confidence == "D_unresolved").sum()
    rows_a = out.loc[out.confidence == "A_high", "row_count"].sum()
    rows_b = out.loc[out.confidence.str.startswith("B"), "row_count"].sum()
    rows_total = out["row_count"].sum()

    print()
    print(f"=== Cascade result ({total} names, {rows_total} rows) ===")
    print(f"  A_high   : {n_a:4d} names / {rows_a:5d} rows")
    print(f"  B_synonym/stereo-missing: {n_b:4d} names / {rows_b:5d} rows")
    print(f"  C_weak   : {n_c:4d} names")
    print(f"  D_unresolved : {n_d:4d} names")
    print()
    print("By tier:")
    print(out["tier"].value_counts(dropna=False))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
