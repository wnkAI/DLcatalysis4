"""
Tag each row in enzyme_data_kcat_km_ion_cleaned.csv with is_natural flag
using BRENDA NATURAL_SUBSTRATE_PRODUCT records.

Logic:
  1. Parse BRENDA JSON dump
  2. Build (ec, uniprot) -> set of substrate-tokens-sets for natural reactions
  3. For each CSV row, normalize its reaction_equation, check if its substrate
     token set matches any NSP substrate set for that (ec, uniprot)
  4. Write new column `is_natural` (bool), keep all original rows

Usage:
  python scripts/00_tag_natural_substrate.py \
    --brenda F:/data/enzyme_scraper/brenda_extracted/brenda_2026_1.json \
    --csv F:/data/enzyme_scraper/enzyme_data_kcat_km_ion_cleaned.csv \
    --out E:/AImodel/DLcatalysis4.0/data/processed/enzyme_data_nsp_tagged.csv
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# Cofactors / common ions to ignore during substrate-set matching.
COMMON_COFACTORS = frozenset({
    "nad+", "nadh", "nad(p)+", "nadp+", "nadph", "nad(p)h",
    "atp", "adp", "amp", "gtp", "gdp", "gmp",
    "h+", "h2o", "h2o2", "o2", "co2", "pi", "ppi", "phosphate", "diphosphate",
    "fad", "fadh2", "fmn", "fmnh2",
    "coa", "coash", "acetyl-coa",
    "acceptor", "reduced acceptor", "reduced electron acceptor",
    "pyridoxal 5'-phosphate", "plp",
    "?",
})


def normalize_token(s: str) -> str:
    """Lowercase, strip whitespace, collapse spaces, strip stereo/parenthetical noise."""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_reaction_lhs(eq: str) -> frozenset:
    """Parse a reaction equation string; return frozenset of LHS substrate tokens
    excluding common cofactors. Supports '=' or ' -> ' separators."""
    if not eq or not isinstance(eq, str):
        return frozenset()
    # Normalize separator
    eq = eq.replace("-->", "=").replace("->", "=").replace("⇌", "=").replace("→", "=")
    if "=" not in eq:
        return frozenset()
    lhs = eq.split("=", 1)[0]
    # Split by '+'
    toks = [normalize_token(t) for t in lhs.split("+")]
    toks = [t for t in toks if t and t not in COMMON_COFACTORS]
    return frozenset(toks)


def build_nsp_lookup(brenda_json_path: str):
    """Return dict: (ec, uniprot) -> list of frozenset(substrate_tokens)"""
    print(f"[load] reading {brenda_json_path} ...")
    with open(brenda_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    data = raw["data"]
    print(f"[load] {len(data)} EC entries")

    lookup = defaultdict(list)
    n_ec_parsed = 0
    n_nsp_entries = 0
    n_mapped = 0

    for ec, ent in data.items():
        if not re.match(r"^\d+\.\d+\.\d+\.\d+$", ec):
            continue
        n_ec_parsed += 1
        proteins = ent.get("protein", {}) or {}
        # Map BRENDA internal protein id -> list of UniProt accessions
        pid_to_up = {}
        for pid, pdata in proteins.items():
            accs = pdata.get("accessions", []) or []
            if accs:
                pid_to_up[pid] = [a.strip().upper() for a in accs if a]

        nsp_list = ent.get("natural_substrates_products", []) or []
        for nsp in nsp_list:
            n_nsp_entries += 1
            sub_set = normalize_reaction_lhs(nsp.get("value", ""))
            if not sub_set:
                continue
            pids = nsp.get("proteins", []) or []
            for pid in pids:
                for up in pid_to_up.get(pid, []):
                    lookup[(ec, up)].append(sub_set)
                    n_mapped += 1
    lookup = {k: list(set(v)) for k, v in lookup.items()}
    print(
        f"[load] parsed {n_ec_parsed} EC, {n_nsp_entries} NSP entries, "
        f"built {len(lookup)} (ec, uniprot) keys, {n_mapped} (ec, uniprot, nsp) mappings"
    )
    return lookup


def row_substrate_set(row) -> frozenset:
    """Return normalized substrate token set from CSV row.
    Prefer `reaction_equation`; fall back to `substrate` column (semicolon-separated)."""
    eq = row.get("reaction_equation", "")
    s = normalize_reaction_lhs(eq) if isinstance(eq, str) else frozenset()
    if s:
        return s
    sub = row.get("substrate", "")
    if not isinstance(sub, str) or not sub.strip():
        return frozenset()
    toks = [normalize_token(t) for t in sub.split(";")]
    toks = [t for t in toks if t and t not in COMMON_COFACTORS]
    return frozenset(toks)


def is_natural_match(csv_set: frozenset, nsp_sets) -> bool:
    """Return True if csv_set shares meaningful overlap with any NSP set.
    Matching rule: non-empty intersection with csv_set that covers at least one
    non-cofactor substrate from csv_set (or NSP set is a subset of csv_set / vice versa).
    """
    if not csv_set:
        return False
    for nsp in nsp_sets:
        if not nsp:
            continue
        if csv_set == nsp:
            return True
        if csv_set.issubset(nsp) or nsp.issubset(csv_set):
            return True
        # token-level fuzzy: at least one main substrate matches exactly
        inter = csv_set & nsp
        if inter:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brenda", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lookup = build_nsp_lookup(args.brenda)

    print(f"[load] reading CSV {args.csv} ...")
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"[load] {len(df)} rows")

    df["uniprot_id"] = df["uniprot_id"].astype(str).str.strip().str.upper()
    df["ec_number"] = df["ec_number"].astype(str).str.strip()

    is_nat = [False] * len(df)
    n_keylookup_hit = 0
    n_nsp_match = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="tag NSP"):
        ec = row["ec_number"]
        up = row["uniprot_id"]
        if up == "NAN" or not up:
            continue
        key = (ec, up)
        nsp_sets = lookup.get(key)
        if nsp_sets is None:
            continue
        n_keylookup_hit += 1
        csv_set = row_substrate_set(row)
        if is_natural_match(csv_set, nsp_sets):
            is_nat[i] = True
            n_nsp_match += 1

    df["is_natural"] = is_nat
    print(
        f"[tag ] rows with (ec, uniprot) hit in BRENDA NSP lookup: {n_keylookup_hit}"
    )
    print(f"[tag ] rows labeled is_natural=True: {n_nsp_match} ({100*n_nsp_match/len(df):.1f}%)")

    # Per-uniprot summary
    per_up = df.groupby("uniprot_id")["is_natural"].agg(["sum", "count"])
    per_up["ratio"] = per_up["sum"] / per_up["count"]
    n_up_total = len(per_up)
    n_up_with_any_natural = (per_up["sum"] > 0).sum()
    print(
        f"[stats] unique uniprot: {n_up_total}; with >=1 natural row: {n_up_with_any_natural} "
        f"({100*n_up_with_any_natural/n_up_total:.1f}%)"
    )

    df.to_csv(out_path, index=False)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
