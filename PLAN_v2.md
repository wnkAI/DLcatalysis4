# DLcatalysis 4.0 — Framework Design (v2)

## Changes from v1 (based on five-model review)
1. Removed ESM-C (keep only ProtT5 to avoid feature redundancy)
2. Predict only kcat and km; kcat/km derived analytically (math consistency)
3. Added MMseqs2 40% sequence-identity split (not just EC-based)
4. **Replaced AlphaFill pocket with docking-based pocket extraction**
5. Staged ablation strategy: build v4-minimal first, add components one at a time
6. Cofactor: learned embedding (not FiLM) to handle power-law distribution
7. Missing-structure mask for AlphaFold-only fallback

## Goal
Predict enzyme kinetic parameters (kcat, km) from sequence + 3D pocket + reaction + cofactor.
kcat/km is derived: `log10(kcat/km) = log10(kcat) - log10(km)`

## Dataset
- 32,429 records after cleaning (kcat_km + kcat+km rows, seq ≤1000aa, 20 standard AAs, ions cleaned from SMILES)
- Split strategy (hierarchical):
  - Primary: MMseqs2 40% identity cluster, group by (cluster_id, reaction_SMILES)
  - OOD-1: held-out EC families (first 3 levels)
  - OOD-2: held-out MMseqs2 clusters (truly novel sequences)
- Deduplicate on (UniProt, canonical_SMILES, rounded pH, rounded temperature)

## Data Pipeline

### Stage A: Pocket Extraction (offline, GPU)
```
For each (UniProt, substrate_SMILES) pair:
  1. Load AlphaFold structure for UniProt (32K unique)
  2. Run P2Rank → predict top-3 cavities (no ligand needed)
  3. Uni-Dock into best cavity → top-5 poses (~2s/pair on GPU)
  4. Select best pose (lowest energy / highest Vina score)
  5. Extract pocket: residues within 6 Å of ligand heavy atoms
  6. Save: {uniprot_id}_{smi_hash}.pt with pocket residue indices + coords
```

Estimated cost: 32K pairs × 2s ≈ 20 h on one GPU.

### Stage B: Feature Pre-compute (offline)
- ProtT5-XL token embeddings per unique sequence (~10K unique)
- DRFP reaction fingerprint per unique reaction (~20K unique)
- RXNMapper reaction center atoms
- Uni-Mol2 embedding per unique substrate

### Stage C: Training-time
- Load pre-computed embeddings from lmdb
- Construct pocket graph from saved residue indices + AlphaFold coords
- No on-the-fly docking

## Architecture

### 1. Protein Encoder (single backbone)
- ProtT5-XL (frozen, LoRA in stage 2) → [L, 1024] token embeddings
- Active-site mask from UniProt annotations (if available)
- **No ESM-C** (removed)

### 2. Pocket Encoder (docking-derived)
- Input: pocket residue indices + 3D coords from Stage A
- Node features: AA type one-hot + ProtT5 token embedding at that residue
- Edge features: distance + unit vector (GVP-equivariant)
- Output: pocket residue embeddings [P, D]
- **Missing-structure mask**: if docking failed / no pocket, use learned [MASK] token + flag

### 3. Reaction Encoder
- Substrate graph (GIN-E) → atom embeddings [N, d_atom]
- DRFP fingerprint → [2048-dim global]
- Reaction center mask (RXNMapper) → atom-level binary flag
- Uni-Mol2 embedding → [512-dim global]
- Concatenate global + project → d_model

### 4. Cofactor Embedding
- Vocab: ~30 cofactors with freq > 50; rest → [OTHER]
- Simple lookup embedding (not FiLM)
- Multi-hot cofactor → sum of embeddings → add to global rep

### 5. Cross-Attention Fusion
- Q: substrate atom embeddings
- K/V: pocket residue embeddings (concatenated with ProtT5 at active-site positions)
- 2 layers, 4 heads
- Interpretable attention maps

### 6. Output Heads (consistency-constrained)
- Two MLP heads: `head_kcat` → log10(kcat), `head_km` → log10(km)
- Derived: `log10(kcat/km) = head_kcat - head_km` (no separate head)
- Loss: LogCosh per target, weighted

## Loss
```
L = λ_kcat · LogCosh(kcat_pred, kcat_true, mask_kcat)
  + λ_km   · LogCosh(km_pred, km_true, mask_km)
  + λ_kkm  · LogCosh(kcat_pred - km_pred, kcat_km_true, mask_kkm)   # consistency via shared heads
  + λ_pcc  · (1 - PCC(kcat_pred, kcat_true))    # rank quality
```
- λ_kcat = λ_km = 1.0, λ_kkm = 0.5 (already implicit via kcat-km), λ_pcc = 0.3
- Missing-label masking per sample

## Training Strategy (staged, ablation-first)

### v4-minimal (baseline, day 1)
- Frozen ProtT5 global + DRFP + cofactor multi-hot → MLP heads
- Proves: sequence-only baseline, no structure

### v4-pocket (day 3)
- + Pocket GVP
- Ablate: does structure help on MMseqs2 40% split?

### v4-fusion (day 5)
- + Cross-attention between pocket and substrate
- Ablate: is cross-attn better than concat?

### v4-full (if all ablations positive)
- + Active-site mask on ProtT5
- + Reaction center feature
- + LoRA fine-tuning

## Evaluation
- Metrics per target: PCC, R², MSE, MAE, RMSE
- Per-split: primary / OOD-EC / OOD-seqcluster
- Report ablation table for each component
- Baselines: DLkcat, UniKP, CatPred, DLcatalysis 3.0

## File Structure
```
DLcatalysis4.0/
├── config/
│   ├── v4_minimal.yml
│   ├── v4_pocket.yml
│   └── v4_full.yml
├── src/
│   ├── model/
│   │   ├── v4_model.py
│   │   ├── gvp_pocket.py
│   │   ├── rxn_encoder.py
│   │   └── losses.py
│   ├── util/
│   │   ├── data_module.py
│   │   ├── featurize.py
│   │   └── splits.py                 # MMseqs2 + EC split
│   └── tools/
│       ├── train.py
│       └── infer.py
├── preprocessing/
│   ├── run_p2rank.py                 # cavity prediction
│   ├── run_unidock.py                # docking
│   ├── extract_pocket.py             # from docked pose
│   ├── compute_drfp.py
│   ├── compute_rxnmapper.py
│   └── mmseqs_cluster.py             # 40% identity split
├── DataSet/
│   └── final_data/
│       ├── pockets/                  # {uniprot}_{smi_hash}.pt
│       ├── seq_prot5.lmdb
│       ├── drfp.pkl
│       ├── rxn_center.pkl
│       ├── unimol.npy
│       └── splits/
│           ├── primary_cv10.json
│           ├── ood_ec.json
│           └── ood_seqcluster.json
└── train/
    └── run_train.py
```

## Open Questions
1. Do we use DiffDock or Uni-Dock for docking? (Uni-Dock is 10x faster)
2. Should cofactor be input feature or auxiliary prediction?
3. How to handle kcat-only or km-only rows (single-label training)?
4. Pretraining on larger unlabeled enzyme-substrate pairs?
