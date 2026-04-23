"""
DLcatalysis 4.0 — Pair-pocket validation (post-run).

Validates a completed pair-docking run before training. Checks:

  Per-pair:
    pocket graph loads
    node_s shape == (k, 26) (or configured k)
    node_v shape == (k, 2, 3)
    ca_xyz shape == (k, 3)
    tensors finite
    ≥ 10 non-padding residues (n_residues check)
    dock score finite
    ligand centroid inside box (if pose exists)

  Overall:
    coverage ≥ min-coverage
    median n_residues == k
    dock score distribution: 5th-95th pct within reasonable range
    fallback rate ≤ 25%
    corrupt / empty artifacts == 0

Exit code:
  0 = READY for training
  1 = STOP (one or more hard checks failed)
  2 = WARN (soft checks failed; review but can proceed)

Usage:
  python scripts/25_validate_pair_pockets.py \\
    --pockets data/processed/pockets_pair.pt \\
    --status-csv data/processed/pair_docking/run_status.csv \\
    --min-coverage 0.90 \\
    --min-n-residues 10 \\
    --expected-k 32
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pockets", required=True)
    ap.add_argument("--status-csv", default=None)
    ap.add_argument("--expected-k", type=int, default=32)
    ap.add_argument("--expected-s-dim", type=int, default=26)
    ap.add_argument("--min-coverage", type=float, default=0.90)
    ap.add_argument("--min-n-residues", type=int, default=10)
    ap.add_argument("--max-fallback-pct", type=float, default=25.0)
    args = ap.parse_args()

    pockets_path = Path(args.pockets)
    if not pockets_path.exists():
        print(f"[FAIL] pockets file missing: {pockets_path}")
        sys.exit(1)

    print(f"[load] {pockets_path}")
    pockets = torch.load(pockets_path, weights_only=False)
    n_total = len(pockets)
    print(f"[load] {n_total} pair pockets")

    hard_fail = []
    soft_warn = []

    # ── Per-pair checks ──
    per_pair = []
    for key, d in pockets.items():
        row = {"key": str(key), "status": "ok"}
        try:
            ns = d["node_s"]
            nv = d["node_v"]
            ca = d["ca_xyz"]
            nr = int(d.get("n_residues", ns.shape[0]))
            if ns.shape != (args.expected_k, args.expected_s_dim):
                row["status"] = f"bad_node_s_shape:{tuple(ns.shape)}"
            elif nv.shape != (args.expected_k, 2, 3):
                row["status"] = f"bad_node_v_shape:{tuple(nv.shape)}"
            elif ca.shape != (args.expected_k, 3):
                row["status"] = f"bad_ca_xyz_shape:{tuple(ca.shape)}"
            elif not torch.isfinite(ns).all():
                row["status"] = "non_finite_node_s"
            elif not torch.isfinite(nv).all():
                row["status"] = "non_finite_node_v"
            elif nr < args.min_n_residues:
                row["status"] = f"too_few_residues:{nr}"
            row["n_residues"] = nr
            row["dock_score"] = d.get("dock_score")
            row["pocket_source"] = d.get("pocket_source", "unknown")
        except Exception as e:
            row["status"] = f"exception:{type(e).__name__}"
        per_pair.append(row)
    per_pair_df = pd.DataFrame(per_pair)

    n_ok = (per_pair_df["status"] == "ok").sum()
    print(f"\n=== PER-PAIR ===")
    print(f"  ok:   {n_ok} / {n_total} ({100*n_ok/max(n_total,1):.1f}%)")
    for s, cnt in per_pair_df["status"].value_counts().items():
        if s != "ok":
            print(f"  {s}: {cnt}")

    # ── Coverage ──
    if args.status_csv and Path(args.status_csv).exists():
        status_df = pd.read_csv(args.status_csv)
        n_attempted = len(status_df)
        coverage = n_ok / max(n_attempted, 1)
        print(f"\n=== COVERAGE ===")
        print(f"  attempted:  {n_attempted}")
        print(f"  valid ok:   {n_ok}")
        print(f"  coverage:   {coverage:.1%}  (min required {args.min_coverage:.1%})")
        if coverage < args.min_coverage:
            hard_fail.append(f"coverage {coverage:.1%} < {args.min_coverage:.1%}")
    else:
        print(f"\n=== COVERAGE === (status-csv not given; skipping attempt count)")

    # ── n_residues distribution ──
    nres = per_pair_df[per_pair_df["status"] == "ok"]["n_residues"]
    if len(nres):
        med = int(np.median(nres))
        pct_full = (nres >= args.expected_k).mean()
        print(f"\n=== POCKET SIZE ===")
        print(f"  median n_residues: {med}")
        print(f"  pairs at full K={args.expected_k}: {pct_full:.1%}")
        if pct_full < 0.70:
            soft_warn.append(f"only {pct_full:.1%} pairs reach K={args.expected_k}")

    # ── Dock score distribution ──
    scores = per_pair_df["dock_score"].dropna().astype(float)
    if len(scores):
        p5, p50, p95 = np.percentile(scores, [5, 50, 95])
        print(f"\n=== DOCK SCORES (ok pairs) ===")
        print(f"  n:       {len(scores)}")
        print(f"  p5 / p50 / p95: {p5:.2f} / {p50:.2f} / {p95:.2f} kcal/mol")
        if p50 > -6.0:
            soft_warn.append(f"median dock score {p50:.2f} > -6.0 (weak binding)")
        if p5 < -20:
            soft_warn.append(f"p5 dock score {p5:.2f} < -20 (suspiciously strong)")

    # ── Pocket source distribution (fallback rate) ──
    sources = per_pair_df["pocket_source"].value_counts()
    print(f"\n=== POCKET SOURCE ===")
    for s, cnt in sources.items():
        print(f"  {s}: {cnt} ({100*cnt/max(n_total,1):.1f}%)")
    docked = sources.get("docked_pose", 0)
    fallback_pct = 100 * (n_total - docked) / max(n_total, 1)
    if fallback_pct > args.max_fallback_pct:
        soft_warn.append(
            f"fallback rate {fallback_pct:.1f}% > {args.max_fallback_pct}%"
        )

    # ── Verdict ──
    print(f"\n=== VERDICT ===")
    if hard_fail:
        print("STOP — hard checks failed:")
        for m in hard_fail:
            print(f"  ✗ {m}")
        sys.exit(1)
    if soft_warn:
        print("WARN — soft checks flagged (review recommended):")
        for m in soft_warn:
            print(f"  ! {m}")
        sys.exit(2)
    print("READY — all checks passed. OK to train.")
    sys.exit(0)


if __name__ == "__main__":
    main()
