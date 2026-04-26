"""
Build an orphan-reaction-template held-out split.

Motivation
----------
Random / sequence-cluster splits leak: the test set may share reactions
with the train set, so a model that memorizes "this reaction class
runs at ~this k_cat/K_M" wins without learning the underlying chemistry.
Horizyn-1's strongest claim (PNAS 2026) is on orphan reactions —
reactions whose template never appeared in training.

This script produces a stricter split:
  - Group rows by REACTION TEMPLATE (atom-map-stripped canonical RXN_SMILES).
  - Whole templates go to test; no template appears in both train and test.
  - The result is `orphan_train.csv` / `orphan_valid.csv` / `orphan_test.csv`
    living next to the existing fold splits.

The hash is stable to atom-map permutations and reactant-order permutations.
If RDKit fails to parse a row's RXN_SMILES, we fall back to a deterministic
key built from `(canonical_substrate_smiles, canonical_product_smiles, EC_3)`
so no row is silently dropped.

Outputs
-------
  data/processed/folds/orphan_train.csv
  data/processed/folds/orphan_valid.csv
  data/processed/folds/orphan_test.csv
  data/processed/folds/orphan_split_stats.json     # reproducibility audit

Usage
-----
  python scripts/28_make_orphan_split.py \\
      --folds_dir data/processed/folds \\
      --test_template_pct 0.10 --valid_template_pct 0.05 --seed 42
"""
import argparse
import hashlib
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    HAVE_RDKIT = True
except ImportError:
    HAVE_RDKIT = False
    log.warning("rdkit not available — falling back to weak template key for ALL rows")


_ATOMMAP_RE = re.compile(r":\d+")


def _strip_atom_map_smiles(smi: str) -> str:
    """Remove `:N` atom-map labels from a SMILES string lexically. Cheap
    fallback when the full RDKit canonicalization isn't available."""
    return _ATOMMAP_RE.sub("", smi)


def _canonical_side(side_smiles: str) -> str:
    """Canonicalize one side of a reaction (`A.B.C`).
    Returns sorted dot-joined canonical SMILES. Returns "" on failure."""
    if not side_smiles:
        return ""
    parts = []
    for s in side_smiles.split("."):
        s = _strip_atom_map_smiles(s)
        if not HAVE_RDKIT:
            parts.append(s)
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            parts.append(s)
        else:
            parts.append(Chem.MolToSmiles(m, canonical=True))
    parts.sort()
    return ".".join(parts)


def reaction_template_key(rxn_smiles: str) -> str:
    """Stable hash for a reaction template, invariant to atom mapping
    and reactant ordering on each side. Returns None on parse failure."""
    if not isinstance(rxn_smiles, str) or ">>" not in rxn_smiles:
        return None
    try:
        left, right = rxn_smiles.split(">>", 1)
    except Exception:
        return None
    canon_left = _canonical_side(left)
    canon_right = _canonical_side(right)
    if not canon_left or not canon_right:
        return None
    blob = f"{canon_left}>>{canon_right}".encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def fallback_template_key(row) -> str:
    """Used when rxn_smiles can't be parsed. Substrate × product × EC3
    is a coarser proxy but still stable across rows that describe the
    "same" reaction with different SMILES decorations."""
    parts = (
        str(row.get("SMI_ID", "")),
        str(row.get("PROD_SMI_ID", "")),
        str(row.get("EC_3", "")),
    )
    blob = "|".join(parts).encode("utf-8")
    return "fb_" + hashlib.sha256(blob).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds_dir", default="data/processed/folds")
    ap.add_argument("--n_folds", type=int, default=10,
                    help="number of fold_{k}.csv files to glue together")
    ap.add_argument("--test_template_pct", type=float, default=0.10,
                    help="fraction of REACTION TEMPLATES held out as test")
    ap.add_argument("--valid_template_pct", type=float, default=0.05,
                    help="fraction held out as valid (also orphan w.r.t train)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_rows_per_template", type=int, default=1,
                    help="discard templates appearing fewer than this many rows")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    folds_dir = Path(args.folds_dir)
    if not folds_dir.exists():
        log.error(f"folds dir not found: {folds_dir}")
        sys.exit(2)

    parts = []
    for k in range(args.n_folds):
        p = folds_dir / f"fold_{k}.csv"
        if not p.exists():
            log.warning(f"missing fold file: {p}")
            continue
        parts.append(pd.read_csv(p))
    if not parts:
        log.error("no fold files loaded")
        sys.exit(2)
    df = pd.concat(parts, ignore_index=True)
    n_total = len(df)
    log.info(f"loaded {n_total} rows from {len(parts)} fold files")

    # ── Compute template keys ────────────────────────────────────────
    keys = []
    fallback_count = 0
    parse_failures = 0
    for _, row in df.iterrows():
        rxn = row.get("RXN_SMILES", "")
        k = reaction_template_key(rxn) if isinstance(rxn, str) else None
        if k is None:
            parse_failures += 1
            k = fallback_template_key(row)
            fallback_count += 1
        keys.append(k)
    df["_TEMPLATE"] = keys
    log.info(
        f"templates: rdkit_canonical={n_total - fallback_count}, "
        f"fallback={fallback_count} (parse failures: {parse_failures})"
    )

    # ── Drop tiny templates ─────────────────────────────────────────
    counts = Counter(keys)
    if args.min_rows_per_template > 1:
        keep = {t for t, c in counts.items() if c >= args.min_rows_per_template}
        before = len(df)
        df = df[df["_TEMPLATE"].isin(keep)].reset_index(drop=True)
        log.info(f"dropped {before - len(df)} rows from tiny templates")
        counts = Counter(df["_TEMPLATE"])

    unique_templates = list(counts.keys())
    rng.shuffle(unique_templates)
    n_t = len(unique_templates)
    n_test = int(round(n_t * args.test_template_pct))
    n_val = int(round(n_t * args.valid_template_pct))
    test_t = set(unique_templates[:n_test])
    val_t = set(unique_templates[n_test:n_test + n_val])
    train_t = set(unique_templates[n_test + n_val:])
    assert not (test_t & val_t)
    assert not (test_t & train_t)
    assert not (val_t & train_t)

    test_df  = df[df["_TEMPLATE"].isin(test_t)].drop(columns=["_TEMPLATE"]).reset_index(drop=True)
    val_df   = df[df["_TEMPLATE"].isin(val_t)].drop(columns=["_TEMPLATE"]).reset_index(drop=True)
    train_df = df[df["_TEMPLATE"].isin(train_t)].drop(columns=["_TEMPLATE"]).reset_index(drop=True)

    # ── Sanity: zero overlap on (SEQ_ID, SMI_ID) is NOT enforced (an enzyme
    # may legitimately appear in both train and test on different reactions).
    # What we DO enforce is zero overlap on reaction TEMPLATE, the actual
    # quantity that defines an "orphan reaction".
    train_templates = set(train_df["RXN_SMILES"].apply(reaction_template_key).dropna())
    test_templates  = set(test_df["RXN_SMILES"].apply(reaction_template_key).dropna())
    leak = train_templates & test_templates
    if leak:
        log.error(f"template leak between train and test: {len(leak)} templates")
        sys.exit(3)

    train_df.to_csv(folds_dir / "orphan_train.csv", index=False)
    val_df.to_csv(folds_dir / "orphan_valid.csv", index=False)
    test_df.to_csv(folds_dir / "orphan_test.csv", index=False)

    stats = {
        "seed": args.seed,
        "n_total_rows": n_total,
        "n_after_filter": len(df),
        "n_templates_total": n_t,
        "n_templates_train": len(train_t),
        "n_templates_valid": len(val_t),
        "n_templates_test":  len(test_t),
        "rows_train": len(train_df),
        "rows_valid": len(val_df),
        "rows_test":  len(test_df),
        "fallback_template_keys": fallback_count,
        "rdkit_available": HAVE_RDKIT,
    }
    with open(folds_dir / "orphan_split_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    log.info(f"wrote orphan_train ({len(train_df)} rows, {len(train_t)} templates)")
    log.info(f"wrote orphan_valid ({len(val_df)} rows, {len(val_t)} templates)")
    log.info(f"wrote orphan_test  ({len(test_df)} rows, {len(test_t)} templates)")
    log.info(f"wrote orphan_split_stats.json")


if __name__ == "__main__":
    main()
