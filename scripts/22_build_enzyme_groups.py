"""
Reorganize final_data.csv into per-enzyme groups for v5 preference modeling.

v5 main task: given an enzyme E and a substrate set S={s1..sk}, predict
catalytic preference (absolute y_i, ranking, pairwise Δ). This requires
a group-view dataset where each "sample" is one enzyme + its substrate panel.

Input:
  data/processed/final_data.csv       — per-row target table (DATA_ID, SEQ_ID, SMI_ID, Y_VALUE, ...)
  data/processed/smi.csv              — unique substrate SMILES

Output:
  data/processed/enzyme_groups.jsonl  — one JSON line per enzyme group
    {
      "seq_id": "P06757",
      "ec_number": "1.1.1.1",
      "n_substrates": 12,
      "substrates": [
        {"smi_id": "SMI_XX", "smiles": "CCO", "data_id": "DATA_000001",
         "y_value": 2.31, "ph": 7.5, "temp": 25.0},
        ...
      ]
    }
  data/processed/enzyme_groups_stats.csv  — panel size distribution

Split view CSVs (filters on enzyme_groups.jsonl):
  data/processed/groups_setA_strict.jsonl   — same enzyme + same assay conditions
  data/processed/groups_setB_loose.jsonl    — same enzyme, conditions may vary slightly
  data/processed/groups_setC_full.jsonl     — all groups (exploratory)

Min panel size filter: n_sub >= 3 (pairwise ranking needs >=2 valid pairs).

Usage:
  python scripts/22_build_enzyme_groups.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def _safe_float(x):
    """Parse scalar to float; return None on NaN / empty / non-numeric."""
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def build_groups(df: pd.DataFrame, smi_map: dict, min_panel: int):
    """Return list of group dicts keyed by seq_id."""
    groups = []
    n_drop_small = 0
    for seq_id, g in tqdm(df.groupby("SEQ_ID"), desc="group"):
        if len(g) < min_panel:
            n_drop_small += 1
            continue
        ec = g["EC_NUMBER"].mode().iloc[0] if "EC_NUMBER" in g.columns else ""
        substrates = []
        for _, row in g.iterrows():
            smi_id = row["SMI_ID"]
            substrates.append({
                "smi_id": smi_id,
                "smiles": smi_map.get(smi_id, ""),
                "data_id": row["DATA_ID"],
                "y_value": _safe_float(row["Y_VALUE"]),
                "ph": _safe_float(row.get("PH")),
                "temp": _safe_float(row.get("TEMP")),
                "rxn_smiles": row.get("RXN_SMILES") if isinstance(row.get("RXN_SMILES"), str) else None,
            })
        # Drop rows where y_value parse failed (shouldn't happen but safe)
        substrates = [s for s in substrates if s["y_value"] is not None]
        if len(substrates) < min_panel:
            n_drop_small += 1
            continue
        groups.append({
            "seq_id": seq_id,
            "ec_number": ec,
            "n_substrates": len(substrates),
            "substrates": substrates,
        })
    print(f"[build] {len(groups)} groups (dropped {n_drop_small} with < {min_panel} substrates)")
    return groups


def stratify_sets(groups: list, ph_tol: float = 0.5, temp_tol: float = 5.0):
    """Stratify groups into Set A (strict same-condition), B (loose), C (full).

    Set A: all substrates share pH ±ph_tol and temp ±temp_tol of each other
    Set B: condition spread within ±3*ph_tol, ±3*temp_tol
    Set C: all groups
    """
    set_a, set_b = [], []
    for g in groups:
        phs   = [s["ph"]   for s in g["substrates"] if s["ph"]   is not None]
        temps = [s["temp"] for s in g["substrates"] if s["temp"] is not None]
        # Missing conditions → treat as exploratory only
        if not phs or not temps:
            continue
        ph_spread = max(phs) - min(phs)
        t_spread  = max(temps) - min(temps)
        if ph_spread <= ph_tol and t_spread <= temp_tol:
            set_a.append(g)
        if ph_spread <= 3 * ph_tol and t_spread <= 3 * temp_tol:
            set_b.append(g)
    return set_a, set_b, groups  # C = all


def write_jsonl(path: Path, groups: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for g in groups:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")


def panel_size_stats(groups: list, name: str):
    sizes = np.array([g["n_substrates"] for g in groups], dtype=int) if groups else np.array([0])
    return {
        "set": name,
        "n_groups": len(groups),
        "n_samples": int(sizes.sum()),
        "size_min": int(sizes.min()) if groups else 0,
        "size_max": int(sizes.max()) if groups else 0,
        "size_mean": float(sizes.mean()) if groups else 0,
        "size_median": float(np.median(sizes)) if groups else 0,
        "pct_ge_5":  float((sizes >= 5).mean())  if groups else 0,
        "pct_ge_10": float((sizes >= 10).mean()) if groups else 0,
        "pct_ge_20": float((sizes >= 20).mean()) if groups else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_data", default="E:/AImodel/DLcatalysis4.0/data/processed/final_data.csv")
    ap.add_argument("--smi_csv", default="E:/AImodel/DLcatalysis4.0/data/processed/smi.csv")
    ap.add_argument("--out_dir", default="E:/AImodel/DLcatalysis4.0/data/processed")
    ap.add_argument("--min_panel", type=int, default=3)
    ap.add_argument("--ph_tol", type=float, default=0.5)
    ap.add_argument("--temp_tol", type=float, default=5.0)
    args = ap.parse_args()

    df = pd.read_csv(args.final_data, low_memory=False)
    smi_df = pd.read_csv(args.smi_csv)
    smi_map = dict(zip(smi_df["SMI_ID"], smi_df["SMILES"]))
    print(f"[load] {len(df)} rows, {df.SEQ_ID.nunique()} unique enzymes")

    groups = build_groups(df, smi_map, args.min_panel)
    set_a, set_b, set_c = stratify_sets(groups, ph_tol=args.ph_tol, temp_tol=args.temp_tol)

    out = Path(args.out_dir)
    write_jsonl(out / "enzyme_groups.jsonl", groups)
    write_jsonl(out / "groups_setA_strict.jsonl", set_a)
    write_jsonl(out / "groups_setB_loose.jsonl",  set_b)
    write_jsonl(out / "groups_setC_full.jsonl",   set_c)

    stats = pd.DataFrame([
        panel_size_stats(set_a, "A_strict"),
        panel_size_stats(set_b, "B_loose"),
        panel_size_stats(set_c, "C_full"),
    ])
    stats.to_csv(out / "enzyme_groups_stats.csv", index=False)

    print()
    print("=== Panel size distribution ===")
    print(stats.to_string(index=False))
    print()
    # Per-size histogram
    sizes = [g["n_substrates"] for g in groups]
    if sizes:
        print("histogram (full):")
        for threshold in [3, 4, 5, 6, 8, 10, 15, 20, 30, 50]:
            n = sum(1 for s in sizes if s >= threshold)
            print(f"  ≥{threshold:3d} substrates: {n:5d} enzymes")
    print()
    print(f"[save] {out}/enzyme_groups.jsonl                          ({len(groups)} groups)")
    print(f"[save] {out}/groups_setA_strict.jsonl                     ({len(set_a)} groups)")
    print(f"[save] {out}/groups_setB_loose.jsonl                      ({len(set_b)} groups)")
    print(f"[save] {out}/groups_setC_full.jsonl                       ({len(set_c)} groups)")
    print(f"[save] {out}/enzyme_groups_stats.csv")


if __name__ == "__main__":
    main()
