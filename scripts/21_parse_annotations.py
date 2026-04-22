"""
Parse InterPro / Pfam / GO annotations from dlcat4_v1_full_rxn.csv.

Produces two files:
  1. data/processed/enzyme_annotations.pt  — per-uniprot dict:
       {
         "interpro_domain_ranges": [(ipr_id, start, end), ...],   # 1-based residue spans
         "pfam_domain_ranges":     [(pf_id,  start, end), ...],
         "interpro_family_ids":    set(["IPR011032", ...]),
         "pfam_family_ids":        set(["PF00107", ...]),
         "go_term_ids":            set(["GO:0008270", ...]),
       }
  2. data/processed/annotation_vocabs.json  — vocabularies for embedding:
       {
         "interpro_family": [sorted IDs],
         "pfam_family":     [sorted IDs],
         "go_term":         [sorted IDs],
       }

Usage:
  python scripts/21_parse_annotations.py \
    --full_rxn data/processed/dlcat4_v1_full_rxn.csv \
    --out_pt   data/processed/enzyme_annotations.pt \
    --out_vocab data/processed/annotation_vocabs.json
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


def parse_domain_field(s: str, id_prefix: str):
    """Parse 'IPR013149(name, 204-336); IPR013154(name, 34-162); ...'
    into [(id, start, end), ...].
    Also handles simple 'ID(name)' without residue range → (id, None, None).
    """
    if not isinstance(s, str) or not s.strip():
        return []
    out = []
    for seg in s.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        # Match ID followed by (...) content
        m = re.match(rf"({id_prefix}\S+)\s*\((.+)\)", seg)
        if not m:
            # Maybe just "ID" alone
            if seg.startswith(id_prefix):
                out.append((seg.strip(), None, None))
            continue
        domain_id = m.group(1).strip()
        content = m.group(2)
        # Find trailing range like "204-336" or start-end at end
        range_match = re.search(r"(\d+)\s*-\s*(\d+)\s*$", content)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            out.append((domain_id, start, end))
        else:
            out.append((domain_id, None, None))
    return out


def parse_id_set(s: str, id_prefix: str):
    """Parse set of IDs from 'IPR011032(name); IPR036291(name)' etc."""
    if not isinstance(s, str) or not s.strip():
        return set()
    ids = set()
    for seg in s.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        m = re.match(rf"({id_prefix}\S+)", seg)
        if m:
            ids.add(m.group(1))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_rxn", default="E:/AImodel/DLcatalysis4.0/data/processed/dlcat4_v1_full_rxn.csv")
    ap.add_argument("--out_pt", default="E:/AImodel/DLcatalysis4.0/data/processed/enzyme_annotations.pt")
    ap.add_argument("--out_vocab", default="E:/AImodel/DLcatalysis4.0/data/processed/annotation_vocabs.json")
    args = ap.parse_args()

    df = pd.read_csv(args.full_rxn, low_memory=False)
    enz = df.drop_duplicates("uniprot_id").copy()
    enz["uniprot_id"] = enz["uniprot_id"].astype(str).str.strip().str.upper()
    print(f"[load] {len(enz)} unique UniProt IDs")

    annotations = {}
    interpro_family_all = set()
    pfam_family_all = set()
    go_all = set()

    for _, row in tqdm(enz.iterrows(), total=len(enz), desc="parse"):
        uid = row["uniprot_id"]
        ipr_domain = parse_domain_field(row.get("interpro_domains"), "IPR")
        pf_domain  = parse_domain_field(row.get("pfam_domains"), "PF")
        ipr_family = parse_id_set(row.get("interpro_family"), "IPR")
        pf_family  = parse_id_set(row.get("pfam_domains"), "PF")  # Pfam IDs reused as "family" signal
        go         = parse_id_set(row.get("go_terms"), "GO:")

        # Derived quality signals for gate_struct consumption
        cofactor_raw = row.get("cofactor")
        has_cofactor = bool(isinstance(cofactor_raw, str) and cofactor_raw.strip())
        active_raw = row.get("active_site")
        binding_raw = row.get("binding_site")
        has_active_site = bool(isinstance(active_raw, str) and active_raw.strip())
        has_binding_site = bool(isinstance(binding_raw, str) and binding_raw.strip())
        has_any_annot = bool(ipr_family or pf_family or go
                             or ipr_domain or pf_domain
                             or has_active_site or has_binding_site)

        annotations[uid] = {
            "interpro_domain_ranges": ipr_domain,
            "pfam_domain_ranges": pf_domain,
            "interpro_family_ids": ipr_family,
            "pfam_family_ids": pf_family,
            "go_term_ids": go,
            # Quality flags piped to gate_struct + logged for Tier-wise reports
            "has_cofactor": has_cofactor,
            "has_active_site": has_active_site,
            "has_binding_site": has_binding_site,
            "has_any_annot": has_any_annot,
        }
        interpro_family_all.update(ipr_family)
        pfam_family_all.update(pf_family)
        go_all.update(go)

    # Also include family IDs that appear in domain (for extra coverage)
    for uid, a in annotations.items():
        interpro_family_all.update([d[0] for d in a["interpro_domain_ranges"]])

    vocab = {
        "interpro_family": sorted(interpro_family_all),
        "pfam_family":     sorted(pfam_family_all),
        "go_term":         sorted(go_all),
    }

    Path(args.out_pt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(annotations, args.out_pt)
    with open(args.out_vocab, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    print()
    print(f"[vocab] InterPro family: {len(vocab['interpro_family'])}")
    print(f"        Pfam family:     {len(vocab['pfam_family'])}")
    print(f"        GO terms:        {len(vocab['go_term'])}")
    print()
    n_ipr_dom = sum(len(a['interpro_domain_ranges']) for a in annotations.values())
    n_pf_dom = sum(len(a['pfam_domain_ranges']) for a in annotations.values())
    print(f"[stats] total InterPro domain instances: {n_ipr_dom}")
    print(f"        total Pfam domain instances:     {n_pf_dom}")
    print(f"        enzymes with ≥1 GO term:         {sum(1 for a in annotations.values() if a['go_term_ids'])}")
    print()
    print(f"[save] {args.out_pt}  +  {args.out_vocab}")


if __name__ == "__main__":
    main()
