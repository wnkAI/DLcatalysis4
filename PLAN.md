# DLcatalysis 4.0 — Framework Design

## Goal
Predict enzyme kinetic parameters (kcat, km, kcat/km) using **sequence + 3D pocket + reaction + cofactor** signals.

## Dataset (we have 32,429 records)
- Sequence (≤1000 aa, 20 standard AAs)
- UniProt ID → AlphaFold/AlphaFill structure files
- Active site / binding site residue positions (from UniProt)
- Reaction SMILES (substrate>>product) with ion components cleaned
- Cofactor list (from BRENDA, e.g. NAD+, FAD, Zn2+)
- Oligomeric state (Monomer/Dimer/...)
- InterPro domains, Pfam, GO terms
- Kinetic labels: kcat (s⁻¹), km (M), kcat/km (M⁻¹·s⁻¹)

## Architecture

### 1. Protein Encoder (multi-view)
- **Sequence view**: ProtT5-XL (frozen) → token embeddings [L, 1024]
- **Structure view**: GVP on AlphaFill pocket (extract 10Å around active_site residues)
  - Node features: AA type + ESM-C 600M embedding
  - Edge features: distance + vector (equivariant)
  - Output: pocket residue embeddings [P, D]
- **Active-site mask**: tokens at active_site positions get attention boost

### 2. Substrate/Reaction Encoder
- **Substrate graph**: GIN-E on molecular graph
- **Reaction fingerprint**: DRFP (differential reaction fingerprint)
- **Reaction center**: RXNMapper → atom-level changes
- **Global descriptors**: Morgan, MolT5, Uni-Mol2 (512-dim, frozen)

### 3. Cofactor / Condition Embedding
- Vocabulary of ~30 common cofactors (NAD+, NADP+, FAD, ATP, Zn2+, Mg2+, ...)
- Multi-hot encoding → FiLM layer to condition protein+reaction reps
- Unknown cofactor → special [UNK] token

### 4. Cross-Attention Fusion
- Query: substrate atom embeddings
- Key/Value: pocket residue embeddings
- 2 layers, 4 heads
- Produces substrate-atom-level attention maps (interpretable)

### 5. Output Heads (multi-task)
- 3 shared MLP heads: `head_kcat`, `head_km`, `head_kcat_km`
- Log-transformed targets (log10)
- LogCosh loss + auxiliary:
  - PCC loss (Pearson correlation)
  - Pair ranking loss (enzyme-pair, substrate-pair)
  - Domain classification auxiliary (predict Pfam from embedding)

## Training Strategy
- 10-fold CV (split by EC number, not random)
- Bayesian HPO on fold 0-2 for hyperparameters
- Stage 1: Train with frozen ProtT5 + frozen ESM (only cross-attn + heads)
- Stage 2: LoRA on ProtT5 last 4 layers + fine-tune everything
- Mixed precision (16-mixed), batch_size=16, AdamW
- Target normalization: log10 + z-score per parameter

## Evaluation Metrics
- PCC, R², MSE, MAE, RMSE (per parameter)
- Out-of-distribution: held-out EC classes
- Comparison baselines: DLkcat, UniKP, CatPred, DLcatalysis3.0

## Expected Improvements (vs v3.0)
- kcat/km PCC: +0.05 (from 3D pocket)
- km PCC: +0.03 (from cofactor conditioning)
- Better OOD generalization (from multi-view + domain features)
- Interpretability: attention maps on pocket residues

## Risks
- AlphaFill missing for ~10% of UniProts → fallback to AlphaFold only
- Small dataset for rare cofactors (Fe4S4, PLP) → may overfit FiLM
- Training time: 2x v3.0 due to GVP

## File Structure
```
DLcatalysis4.0/
├── config/
│   └── config.yml
├── src/
│   ├── model/
│   │   ├── v4_model.py         # main model
│   │   ├── gvp_pocket.py       # equivariant pocket encoder
│   │   ├── rxn_encoder.py      # reaction encoder
│   │   ├── cofactor.py         # cofactor FiLM
│   │   └── losses.py
│   ├── util/
│   │   ├── data_module.py
│   │   ├── pocket_extract.py   # extract pocket from PDB
│   │   ├── featurize.py        # pre-compute embeddings
│   │   └── rxn_features.py     # DRFP + reacting center
│   └── tools/
│       ├── train.py
│       └── infer.py
├── DataSet/
│   └── final_data/
│       ├── protein_gvp.pt       # pre-computed GVP features
│       ├── esm_node.pt          # ESM-C node features
│       ├── drfp.pkl             # reaction fingerprints
│       └── train/val/test.csv
└── train/
    └── run_train.py
```
