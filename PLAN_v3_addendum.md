# PLAN_v3 Addendum — 2026-04-21

增量变更，不重写 PLAN_v2。仅覆盖与 v2 冲突的三个点。

## 1. 数据源切换（原始数据）

从 `F:/data/enzyme_scraper/brenda_2026_1.json`（BRENDA 2026.1 原始 dump，677 MB）+ `F:/data/enzyme_scraper/enzyme_data_kcat_km_ion_cleaned.csv`（32,429 条）联合产出最终训练 CSV。

## 2. 主底物标注（新增 Phase 0a）

每条 kinetic 记录打 `is_natural` 标签：

```
(EC, UniProt_ID) → BRENDA['natural_substrates_products'][*]
                 → cross-ref via BRENDA['protein'][pid]['accessions']
                 → match CSV row's reaction_equation 到 NSP reaction set
                 → is_natural = True/False
```

只保留 `is_natural=True` 子集做 v4 主训练。对比 baseline 可用 `is_natural=True` vs 全量。

覆盖率预估：BRENDA 只有 42% 的 protein 有 UniProt accession，NSP 覆盖会受此限制；实际可用子集预估 5,000–8,000 条（低于 PLAN_v2 的 32k）。

## 3. 结构源路由（替换 PLAN_v2 Stage A step 1）

```python
def route_structure(row):
    has_cof = pd.notna(row['cofactor']) and row['cofactor'].strip() != ''
    if has_cof and pd.notna(row['alphafill_file']):
        return 'alphafill'        # 最优：已有 cofactor pose
    if pd.notna(row['pdb_file']):
        return 'pdb_real'          # 次优：真实实验结构
    if pd.notna(row['alphafold_file']):
        return 'af_only'           # 兜底：AlphaFold 预测
    return 'missing'
```

覆盖预估（在 `is_natural=True` 子集上）：
- alphafill: ~15–20%
- pdb_real:  ~40–45%
- af_only:   ~35–40%
- missing:    <5%

## 4. Docking 工具切换

**从 Uni-Dock 换成 AutoDock-Vina-GPU-2.1**（A100 40G 上 1-3h 跑完 5-8k 对）。
理由：审稿人更熟悉经典 Vina 打分，A100 上 GPU 加速后速度已对标 Uni-Dock。

Box 定义：
- 若 `route=alphafill`：box center = AlphaFill 移植 cofactor 的 centroid
- 若 `route=pdb_real` 且 PDB 含 hetatm ligand：用 ligand centroid
- 否则：fpocket top-1 pocket centroid
- box size 统一 22³ Å

Ligand prep：RDKit ETKDG 3D → Meeko → pdbqt  
Receptor prep：Reduce 加氢 → pdbqt  
Vina 参数：`exhaustiveness=16, num_modes=5`，保 top-1 pose，Vina score < -5 kcal/mol 标低置信度

## 5. 更新后的 Phase 0-2

| Phase | 任务 | 工时（A100 服务器）|
|---|---|---|
| 0a | NSP tagging from BRENDA JSON | 0.5 d CPU |
| 0b | clip log10(kcat/km) + unit check + dedup + CD-HIT 40% fold split | 0.5 d CPU |
| 1  | 结构源路由 + 文件索引 | 0.25 d CPU |
| 2a | fpocket 批量（AF/PDB 无配体时用）| 0.25 d CPU |
| 2b | Ligand pdbqt 预处理（每独立 SMILES 一次）| 0.25 d CPU |
| 2c | AutoDock-Vina-GPU-2.1 批量对接 | **1–3 h GPU** |
| 3  | Docked-pose-based pocket extraction → `pocket_{uniprot}_{smi}.pt` | 0.25 d CPU |

Stage B/C（特征预计算、模型训练）沿用 PLAN_v2，不改。
