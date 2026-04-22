"""
Download AlphaFold v4 monomer PDBs from EBI for each UniProt ID in seq.csv.

URL format:
  https://alphafold.ebi.ac.uk/files/AF-{UNIPROT}-F1-model_v4.pdb

Usage:
  python scripts/16_fetch_alphafold.py \
    --seq_csv data/processed/seq.csv \
    --out_dir /data/structures/alphafold \
    --workers 20

Skips files that already exist; safe to re-run.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


AFDB_URL = "https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v4.pdb"


def fetch_one(uid: str, out_dir: Path, session: requests.Session) -> dict:
    out_path = out_dir / f"AF-{uid}-F1-model_v4.pdb"
    if out_path.exists() and out_path.stat().st_size > 0:
        return {"uid": uid, "ok": True, "cached": True}
    url = AFDB_URL.format(uid=uid)
    try:
        r = session.get(url, timeout=30)
        if r.status_code == 200 and len(r.content) > 100:
            out_path.write_bytes(r.content)
            return {"uid": uid, "ok": True, "bytes": len(r.content)}
        return {"uid": uid, "ok": False, "status": r.status_code}
    except Exception as e:
        return {"uid": uid, "ok": False, "error": str(e)[:60]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_csv", default="data/processed/seq.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_csv(args.seq_csv)
    uids = df["UNIPROT_ID"].astype(str).str.upper().unique().tolist()
    print(f"[fetch] {len(uids)} UniProt IDs from {args.seq_csv}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "DLcatalysis4.0/AFfetch"})

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, uid, out_dir, session): uid for uid in uids}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="AF"):
            results.append(fut.result())

    df_res = pd.DataFrame(results)
    df_res.to_csv(out_dir / "fetch_log.csv", index=False)
    n_ok = df_res["ok"].sum()
    n_cached = df_res.get("cached", pd.Series([False]*len(df_res))).sum()
    print(f"[done] {n_ok}/{len(df_res)} ok  ({n_cached} cached)")
    print(f"[save] structures in {out_dir}")


if __name__ == "__main__":
    main()
