"""
DLcatalysis 4.0 — v4-ultimate ablation matrix generator.

Emits a family of YAML configs that progressively enable each modality,
so we can measure the incremental contribution of every branch of the
v4-ultimate residual:

    pred = y_seq + g_pair*(y_sub + y_rxn + y_int3d)
                + g_struct * y_struct
                + g_annot  * y_annot

Ablation ladder (each row keeps everything from the row above):

    A0  seq          seq only (ProtT5 + attn pool + MLP)
    A1  +sub         +substrate GINE, no rxn_center
    A2  +rxn_center  +RXNMapper reacting-center atom flag
    A3  +drfp        +DRFP reaction fingerprint
    A4  +pocket      +GVP pocket encoder
    A5  +int3d       +pocket × atom 3D cross-attention
    A6  +annot       +InterPro / Pfam / GO bag-of-words
    A7  +condition   +pH / temperature conditioning
    A8  full         +EC 4-level embedding   (== v4_ultimate)

Run:
    python scripts/26_generate_ablation_configs.py \\
        --base  config/v4_ultimate.yml \\
        --out   config/ablation/

Output: config/ablation/v4_ultimate_A{0..8}_{tag}.yml
Each config gets a disambiguated log_path / checkpoint_path / model_name
so Lightning will not overwrite runs from sibling ablations.
"""
import argparse
import copy
from pathlib import Path

import yaml


# (id, tag, flag overrides) — applied in order on top of the previous row
LADDER = [
    ("A0", "seq",         dict(use_rxn_drfp=False, use_rxn_center=False,
                               use_pocket=False,   use_int3d=False,
                               use_annot=False,    use_condition=False,
                               use_ec=False)),
    ("A1", "sub",         dict()),                         # GINE always on
    ("A2", "rxn_center",  dict(use_rxn_center=True)),
    ("A3", "drfp",        dict(use_rxn_drfp=True)),
    ("A4", "pocket",      dict(use_pocket=True)),
    ("A5", "int3d",       dict(use_int3d=True)),
    ("A6", "annot",       dict(use_annot=True)),
    ("A7", "condition",   dict(use_condition=True)),
    ("A8", "full",        dict(use_ec=True)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="config/v4_ultimate.yml")
    ap.add_argument("--out",  default="config/ablation")
    args = ap.parse_args()

    base_path = Path(args.base)
    out_dir   = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(base_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    cur = copy.deepcopy(base)
    # Start from the most-restrictive flags and layer back on
    flags = copy.deepcopy(LADDER[0][2])
    for aid, tag, delta in LADDER:
        flags.update(delta)
        cfg = copy.deepcopy(cur)

        # Override feature flags
        for k, v in flags.items():
            cfg["model"][k] = v

        # Disambiguate artifacts
        name = f"v4_ultimate_{aid}_{tag}"
        cfg["model"]["model_name"] = name
        cfg["train"]["checkpoint_path"] = f"train/ckpt_ablation/{name}"
        cfg["train"]["log_path"]        = f"train/log_ablation/{name}"

        out_path = out_dir / f"{name}.yml"
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        enabled = [k.replace("use_", "") for k, v in flags.items() if v]
        print(f"[write] {out_path}")
        print(f"        enabled: {enabled}")

    print(f"\n[done] {len(LADDER)} ablation configs in {out_dir}")


if __name__ == "__main__":
    main()
