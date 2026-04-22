# DLcatalysis 4.0 — A100 服务器运行指南（完整版）

本地 (Windows) 负责 **代码 + 数据清洗**；GPU 训练 + 对接全部在 A100 40G 服务器跑。

## 0. 仓库状态

```
data/processed/
  dlcat4_v1_full_rxn.csv  — 25,518 行（完整 substrate+product+reaction SMILES）
  final_data.csv          — 24,961 行（去掉单位不规范的 557 条）
  seq.csv                 — 7,100 unique UniProt
  smi.csv                 — 5,990 unique substrate SMILES
  prod_smi.csv            — 3,614 unique products
  folds/{train,valid,test}_fold{0..9}.csv — 10-fold enzyme-grouped CV

src/model/
  v4_minimal.py   — 2 分支 baseline (y_seq + g·y_sub)
  v4_pocket.py    — 4 分支 (y_seq + g_pair·(y_sub + y_int3d) + g_struct·y_struct)
  gvp_pocket.py   — GVP-GNN pocket encoder (K=32)
  int3d_cross_attn.py — 3D 距离 RBF bias 的 residue×atom cross-attn

config/
  v4_minimal.yml  — 无结构 baseline
  v4_pocket.yml   — 结构增强版

scripts/
  10-13  — 数据准备（本地已跑）
  11,12  — 分子图 / ProtT5 预计算（服务器跑）
  14     — Pocket v2 提取（K=32, 26d 特征：AA + pLDDT + 4 flags + dist）
  15     — 结构源路由（AlphaFill → PDB → AF）
  16     — AlphaFold 批量下载
  20     — AutoDock-Vina-GPU 批量对接
```

---

## 1. 上传代码到服务器

```bash
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.log' \
  E:/AImodel/DLcatalysis4.0/ user@a100:/data/wnk/DLcatalysis4.0/
```

---

## 2. 环境（一次性）

```bash
# 主环境
conda create -n dlcat4 python=3.10 -y
conda activate dlcat4

# PyTorch + PyG (CUDA 12.1)
pip install torch==2.2.0
pip install torch-scatter torch-cluster torch-sparse torch-geometric \
  -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# 项目依赖
pip install -r requirements.txt
pip install biopython meeko requests  # pocket / docking 需要

# 对接工具链
conda install -c bioconda openbabel fpocket -y

# AutoDock-Vina-GPU-2.1 (编译 ~30 min)
cd /tmp
git clone https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1.git
cd Vina-GPU-2.1/AutoDock-Vina-GPU-2-1
# 改 makefile 里的 CUDA 路径，然后：
make -j8
sudo cp AutoDock-Vina-GPU-2-1 /usr/local/bin/vina-gpu
```

---

## 3. 数据预计算（服务器跑一次）

```bash
cd /data/wnk/DLcatalysis4.0

# 3a. 分子图 (CPU, <1 min for 5,990 SMILES)
python scripts/11_precompute_mol_graphs.py \
  --smi_csv data/processed/smi.csv \
  --out data/processed/mol_graphs.pt

# 3b. ProtT5 embedding LMDB (GPU, 30-60 min on A100 for 7,100 seqs)
python scripts/12_precompute_prot5_lmdb.py \
  --seq_csv data/processed/seq.csv \
  --output data/processed/seq_prot5.lmdb \
  --model Rostlab/prot_t5_xl_uniref50 \
  --device cuda:0 \
  --max_seq_len 1000 \
  --batch_size 8
```

---

## 4. v4-minimal baseline（先跑对标 3.0）

```bash
# Single fold 快速 sanity
python train/run_train.py --config config/v4_minimal.yml --fold 0

# Full 10-fold CV (~8h on A100)
python train/run_train.py --config config/v4_minimal.yml --cv
```

输出：
- ckpts: `train/ckpt_minimal/fold_{k}/best-*.ckpt`
- metrics: `train/log_minimal/dlcat4_minimal_*.csv`
- 预期 PCC 0.35-0.40（2 分支无结构）

---

## 5. 对接数据准备（v4-pocket 前置）

### 5a. 批量拉 AlphaFold (EBI 下载, ~10-20 min 并发)

```bash
python scripts/16_fetch_alphafold.py \
  --seq_csv data/processed/seq.csv \
  --out_dir /data/structures/alphafold \
  --workers 20
```

（若有 AlphaFill / PDB，分别放到 `/data/structures/alphafill` 和 `/data/structures/pdb`）

### 5b. 结构源路由 manifest

```bash
python scripts/15_build_structure_manifest.py \
  --full_rxn data/processed/dlcat4_v1_full_rxn.csv \
  --final_data data/processed/final_data.csv \
  --out data/processed/structure_manifest.csv \
  --af_dir /data/structures/alphafold \
  --alphafill_dir /data/structures/alphafill \
  --pdb_dir /data/structures/pdb
```

期望输出：~7,100 enzymes 分布
- alphafill_cofactor (优先)：~1,500-2,000
- pdb_real：~2,800-3,000
- af_only：~2,100-2,500
- missing：<100

### 5c. 批量对接（核心步骤）

```bash
# 全量 25k pair 对接 (~20-40 GPU-h on A100)
python scripts/20_run_vina_gpu.py \
  --final_data data/processed/final_data.csv \
  --manifest data/processed/structure_manifest.csv \
  --smi_csv data/processed/smi.csv \
  --out_dir /data/wnk/docking_poses \
  --vina_gpu /usr/local/bin/vina-gpu \
  --workers 1

# 省算力模式：每酶只对接一次（~6-10 GPU-h）
python scripts/20_run_vina_gpu.py \
  --final_data data/processed/final_data.csv \
  --manifest data/processed/structure_manifest.csv \
  --smi_csv data/processed/smi.csv \
  --out_dir /data/wnk/docking_poses \
  --vina_gpu /usr/local/bin/vina-gpu \
  --limit 7200   # 和 unique enzyme 数一致
```

每个对接对产出：
- `{uid}_{smi_id}.pdbqt` — top-1 pose
- `{uid}_{smi_id}.log` — Vina 打分
- `docking_manifest.csv` — 全部状态 + 打分

### 5d. Pocket 提取

```bash
# 把 docked pose 路径加到 manifest 里（update structure_manifest.csv）
# 然后：
python scripts/14_extract_pockets.py \
  --manifest data/processed/structure_manifest.csv \
  --out data/processed/pockets.pt \
  --k 32 \
  --probe_radius 10.0 \
  --annot_expand_radius 8.0
```

输出：per-enzyme pocket tensor（K=32, 26d scalar + 2×3 vector + k-NN graph）。

---

## 6. v4-pocket 完整训练

```bash
# Single fold
python train/run_train.py --config config/v4_pocket.yml --fold 0

# Full 10-fold CV (~12h on A100)
python train/run_train.py --config config/v4_pocket.yml --cv
```

**预期 PCC**：相对 v4-minimal 至少提升 0.05-0.10（结构分支真贡献）。如果没提升，说明 pocket 信号被 sequence branch 已覆盖 —— 走 CatPred 的警示。

---

## 7. Ablation 三件套（证伪 pocket 信号）

在 fold 0 上跑：

```bash
# 1. Pocket scrambling 控制（随机 32 个非 pocket 残基）
# → 需要手改 scripts/14_extract_pockets.py 添加 --mode random
python scripts/14_extract_pockets.py --mode random --out data/processed/pockets_random.csv
python train/run_train.py --config config/v4_pocket_random.yml --fold 0

# 2. Radius sweep (5Å LigandMPNN / 8Å EnzymeCAGE / 10Å)
for r in 5.0 8.0 10.0; do
  python scripts/14_extract_pockets.py --probe_radius $r --out data/processed/pockets_r${r}.pt
done

# 3. AlphaFill-masked 推理（结构分支在 75% 无 AlphaFill 样本上也要有贡献）
# → 通过 dropout 或推理时清零 y_struct 跑
```

报告时把 **Tier-1 vs Tier-2 vs Tier-3** 子集的性能分开报告，展示 pocket 在各层的真实贡献。

---

## 8. 已知 TODO

- [ ] fold split 目前是 enzyme-grouped random，改 **CD-HIT 40% identity**（服务器装 cd-hit 后用）
- [ ] Pocket Tier-3 fpocket 代码是 stub，落地时需补
- [ ] `substrate_pose_path` 列要在对接完成后 update 到 structure_manifest.csv
- [ ] has_cofactor / has_annotation 信号要 pipe 到 gate_struct（目前是 stub 零值）

---

## 9. 结果对比预期

| 版本 | 分支 | 数据 | 预期 PCC | 说明 |
|---|---|---|---|---|
| DLcatalysis 3.0 | 3-branch seq-only | 22k | 0.43 | 现有 baseline |
| DLKcat | seq + substrate GNN | 17k | 0.35 | 领域老 SOTA |
| CatPred | seq + whole-prot E-GNN | 23k | 0.42 | 他们说 E-GNN 没提升 |
| TurNuP | ProtT5 + DRFP | 13k | 0.57 (SCC) | 只做 kcat/Km |
| EnzymeCAGE | AlphaFill pocket GNN | 1.5M (reaction task) | 不是回归，不可比 |
| **v4-minimal** | 2-branch ProtT5+GINE | 25k | **0.38-0.42** | 目标对标 3.0 |
| **v4-pocket** | 4-branch + GVP + 3D cross-attn | 25k | **>0.50** | 目标超 3.0 |

如果 v4-pocket PCC > 0.50，就可以作为 Nature 投稿的核心结果。
