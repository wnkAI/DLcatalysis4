# DLcatalysis 4.0 数据构建：当前卡住的问题（求教）

**日期**：2026-04-21
**目标**：构建酶动力学回归模型 DLcatalysis 4.0（预测 log10(kcat/Km)），在 DLcatalysis 3.0 基础上加入结构信息（AlphaFold/AlphaFill + pocket GVP + residue-substrate cross-attention）。

---

## 1. 项目背景

- **数据源**：BRENDA 2026.1 kinetic 数据（从原始 dump 爬取，F:/data/enzyme_scraper/）
- **建模输入**：enzyme sequence + substrate SMILES + (可选) product SMILES + 结构
- **建模输出**：log10(kcat/Km) 的回归
- **现有规模**：32k–33k 行（经过 kcat/Km 标签过滤和 dedup，seq ≤1000 aa），包含 8,453 个独立 UniProt
- **对标基线**：DLKcat (17k)、UniKP (17k)、CatPred (23k)、DLcatalysis 3.0 (22k)

---

## 2. 核心问题：底物 SMILES 覆盖率

在 33,407 可用标签行中，**只有 23,929 行（72%）有 substrate_smiles**。剩下 **~9,500 行有 substrate 文本名但缺 SMILES**。

### 这些缺失底物大致分三类：

| 类别 | 占比估算 | 例子 |
|---|---|---|
| 蛋白 / RNA / 多糖（本身不是小分子）| ~20% | cytochrome c, ferricytochrome c, tRNALeu, starch, inulin, reduced ferredoxin |
| 实验室习惯命名 / 缩写 | ~40% | benzyloxycarbonyl-Phe-Arg-7-amido-4-methylcoumarin (即 Z-Phe-Arg-AMC), 2,2'-azino-bis-(3-ethylbenzthiazoline-6-sulfonic acid) (即 ABTS) |
| 真实小分子但命名不规范 | ~40% | 3,4-dihydro-retinol, D-talonate, (R)-alpha-tetralol, 12-oxolauric acid methyl ester |

### 进一步观察

水解酶（EC 3.x.x.x）的 6,310 行中 **94.6% 没有 product**（BRENDA 惯例，水解产物默认已知），与缺 SMILES 问题部分重叠。

---

## 3. 我们尝试过的方案和结果

### 方案 A：PubChem REST name 搜索

```python
name → PubChem /compound/name/{name}/CID → CID → SMILES
```

**结果**：
- Top 200 最常见底物：回收 5.0%（10/200），rows 覆盖 8%
- 随机 200 底物：回收 3.5%（7/200）
- 失败原因：BRENDA 命名不规范，PubChem name 搜索要求精确匹配同义词

### 方案 B：用 Rhea 反补（参考 EnzymeCAGE 管道）

```
(UniProt, EC) → Rhea master reaction → 拿 reaction SMILES → 反应物是 substrate，产物是 product
```

已实现的 lookup：
- `rhea-reaction-smiles.tsv`: 36,014 reactions（Rhea 140, 2025 版）
- `rhea2uniprot_sprot.tsv`: 236k Swiss-Prot 映射
- `rhea2uniprot_trembl.tsv.gz`: 40.9M TrEMBL 映射（流式过滤到我们的 8,453 个目标 UniProt 后只剩 3,892 条新命中）
- `rhea2ec.tsv`: 7,613 EC 映射

**覆盖结果**：

| 场景 | 行数 | 占比 |
|---|---|---|
| UniProt 在 Rhea 且 (UniProt, EC) 唯一匹配一个 Rhea 反应 | 17,878 | 53.5% |
| UniProt 在 Rhea 但 (UniProt, EC) 匹配多个反应（需消歧）| 3,409 | 10.2% |
| UniProt 在 Rhea 但 EC 不匹配 | 1,524 | 4.6% |
| UniProt 不在 Rhea | 9,821 | 29.4% |
| **总可通过 Rhea 覆盖** | **21,287** | **63.7%** |

### BRENDA + Rhea 合并

| 类别 | 行数 | % |
|---|---|---|
| 两边都有 SMILES（可交叉验证） | 17,536 | 52.5% |
| 只有 BRENDA SMILES（Rhea 无此反应）| 6,393 | 19.1% |
| 只有 Rhea SMILES（BRENDA 缺，从 Rhea 反补）| 3,751 | 11.2% |
| 两边都没（完全空缺）| **5,727** | **17.1%** |
| **合并总可建模** | **27,680** | **82.9%** |

---

## 4. 我们的具体疑问

### Q1：Rhea + BRENDA 的合并方法是否可靠？

做法是：若 `(BRENDA uniprot, BRENDA ec)` 命中 Rhea 唯一一个 master reaction，就把该反应的反应物作为 substrate SMILES、产物作为 product SMILES。

- **假设**：BRENDA 记录的 kcat/Km 实验是在这个标准反应上测的
- **风险**：BRENDA 可能在同一条 (UniProt, EC) 下记录多个**非天然底物** kinetic 值（promiscuity 测试），而 Rhea 只给 canonical 反应 → 把非天然底物的 kinetic 值强行对到 canonical 反应的 SMILES 上会错
- **想问**：这个风险实际多大？有没有公开工作用类似管道？具体怎么验证这种 mapping 的正确性？

### Q2：(UniProt, EC) 多反应时怎么消歧？

10.2%（3,409 行）匹配到多个 Rhea 反应（比如一个多功能酶能催化好几个反应）。我们目前的思路是：
- 如果 BRENDA 行已有 substrate_smiles → 用它和 Rhea 每个反应的反应物做 InChIKey 匹配选一个
- 如果 BRENDA 行只有 substrate **名字** → 用名字和 Rhea 反应的 ChEBI 名字做 fuzzy 匹配

**想问**：有没有更稳的消歧方法？EnzymeCAGE 他们怎么处理多反应情形？

### Q3：剩下 17.1%（5,727 行）两边都没 SMILES 的处理

这些行 UniProt 不在 Rhea，BRENDA 也没给 SMILES。选项：
- **丢掉** → 数据量从 27,680 → 27,680（不动）
- **走 PubChem + OPSIN + LLM cascade 兜底** → 实测 PubChem 命中率很低（3-8%），OPSIN/LLM 会有幻觉风险
- **硬写特殊 SMILES（蛋白 → 肽 SMILES、多糖 → 聚合物记号）** → 审稿抗打性问号

**想问**：这批数据在公开模型里是怎么处理的？是否有"放弃 17% 是合理代价"的共识？

### Q4：是否应该反过来 —— 以 Rhea 为主干？

EnzymeCAGE 从 Rhea 出发（15k reactions × 几十万 UniProt → 1.5M 对），BRENDA 只给标签。
我们现在是从 BRENDA 出发（33k 标签行 → 反查 Rhea 补结构）。

**两种思路差别**：
- 从 BRENDA 出发：保证每行都有 kcat/Km 标签，但底物不一定是 canonical
- 从 Rhea 出发：每行都是 canonical 反应，但很多反应没有 kinetic 标签（BRENDA 里可能没测过）

**想问**：对 kcat/Km 回归任务，这两种数据组织方式哪个更对？

### Q5：BRENDA `substrate` 字段的命名规范

BRENDA 里大量底物用实验室缩写（Z-Phe-Arg-AMC 的全名、ABTS 的全名、DCPIP 等），这些在 PubChem name search 里完全查不到。

**想问**：社区里有没有"BRENDA substrate name → canonical name/CAS/SMILES"的字典或工具？比如 biochem entity resolver 之类？

---

## 5. 当前状态

- **已有代码**：
  - `scripts/00_tag_natural_substrate.py`（用 BRENDA JSON 打 NATURAL_SUBSTRATE_PRODUCT 标签）
  - `scripts/01_recover_smiles_pubchem.py`（PubChem REST 探针）
  - `scripts/02_rhea_coverage_probe.py`（Rhea 覆盖统计）
- **待定**：把 Rhea 反补的 SMILES 回写 CSV 的脚本 + 合并策略
- **不确定的决策**：是否保留 17% 完全缺失的那部分数据

---

## 6. 具体请教对象可参考的开源项目

- **EnzymeCAGE**（Nature Catalysis 2026）：完全从 Rhea 出发，数据清洗逻辑在 `scripts/rhea_data_cleaning.py`
  - 他们用：`rhea-reaction-smiles.tsv` + `rhea2uniprot_sprot.tsv` + `rhea2ec.tsv` + `rhea-directions.tsv`
  - 过滤：原子 >150 丢、反应物/产物含碳分子 >3 丢（保留干净酶催化反应）
- **CatPred / DLKcat / UniKP**：kcat/Km 回归，都没说清他们怎么从 BRENDA 拿到 substrate SMILES
- **Nature Commun 2023 论文** (DOI 10.1038/s41467-023-39840-4) — 需要别人帮我们看一下他们的数据 curation 怎么做的，我们这边 WebFetch 被 Nature 挡（需登录）

---

## 7. 补充：对标现有 SOTA 数据 curation 的发现

读完 UniKP (Nature Commun 2023, DOI 10.1038/s41467-023-39840-4) 和 DLKcat (Nature Catalysis 2022) 的 Methods：

- **DLKcat**：从 BRENDA + SABIO-RK（API）取 `substrate name → PubChem → SMILES`，用**自定义同义词预处理脚本**保证同一底物不同写法归并到同一 canonical SMILES。最终 **17,010 条**。脚本没公开。
- **UniKP**：
  - kcat 数据 = **直接用 DLKcat 的 16,838 条**
  - Km 数据 = **借前序 SOTA 论文的 11,722 条**
  - kcat/Km 数据 = 自己建，**只 910 条**，方法还是 `name → PubChem`
  - pH/temp 数据 = 自己建，各 ~600 条，方法一样
- **CatPred / 其他**：估计类似套路

**结论**：整个领域都用 `name → PubChem`，真正的数据量上限就是 ~17-23k。
**我们的 Rhea 反补（82.9%, 27,680 行）是 novel 贡献**，超过所有现有 SOTA 数据规模。

## 8. 最想得到的具体答复

1. **BRENDA + Rhea 反补**这条路在行业里常见吗？有没有公开脚本？或者曾经被尝试但被审稿人拒过？
2. **多反应消歧**（同一 (UniProt, EC) 在 Rhea 里对应多个 reaction）的标准做法是什么？EnzymeCAGE 怎么做的？
3. 剩下 **17% 缺失**是丢还是补，业界共识？
4. **DLKcat 2022 Supplementary Fig. 2** 里的 data cleaning 详细步骤（以及那个**同义词预处理 Python 脚本**）有途径拿到吗？这是他们论文里的 curation 细节，但不在主文。
5. 有没有现成的 **"BRENDA substrate name → SMILES / CAS / ChEBI ID" 字典**？（ChEBI Ontology Lookup + 同义词库之类）
6. 如果老师/同行做过类似数据 curation 的项目代码或 CSV，能否分享一份参考？
