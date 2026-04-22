# /five review 综合报告 (2026-04-22)

## 审查状态

| Reviewer | 状态 | Verdict |
|---|---|---|
| Codex | ✅ 完成 | BLOCK → 修 2 后进入 NEEDS_WORK |
| DeepSeek | ✅ 完成 | NEEDS_WORK（但 3 项指控是幻觉）|
| Gemini (2.5-pro) | ✅ 完成（多次 429 重试）| NEEDS_WORK |
| Kimi | ⏳ 还在跑 | — |

---

## 已修复的 Bug（commit f6de404）

### BUG #1: xyz_valid 批样本级泄漏（Codex + DeepSeek 一致命中）
**问题**：`v4_pocket.py` 和 `v4_ultimate.py` 用 `G.MOL_graph_xyz_valid.any()` 批全局开关。一个样本有效就给**整个 batch** 喂 xyz；无效样本的零坐标进 `cdist` 产生假距离 bias。

**修复**：改成 per-sample 掩码。`Int3DCrossAttnLayer` 接受 `xyz_valid_per_sample: (B,) bool`，在无效样本上把 bias 置零（`bias = bias * v.view(B, 1, 1, 1)`）。

### BUG #5: v4_ultimate 没读 `struct.gate_init_bias`（Codex）
**问题**：v4_pocket 修了（79af90a），v4_ultimate 仍硬编码 `-2.0`。

**修复**：从 config 读，和 v4_pocket 保持一致。

---

## Codex 点出但未修的 Bug（待下次 commit）

### BUG #2: feature flag 错位
- dataset 自动 disable 缺失模态（DRFP、rxn_center、annot 找不到文件）
- model 按原 config flag 构建分支（head + MLP）
- 运行时 `_encode_*` 返回零 → head 输出 bias → 静默污染
- **修法**: dataset disable 后回写 config，model 读 dataset 的"实际可用 flag"

### BUG #3: v4_pocket 没有 `use_pocket` 开关
- `pocket_path` 缺失时 dataset 默默 `self.pockets = None`
- `_encode_pocket` 返回全零 + mask=None → y_struct 永远 0
- **修法**: 给 v4_pocket 加 `use_pocket` 配置项

### BUG #4: Int3DCrossAttn 全 mask 边界
- pocket_mask 或 atom_mask 全 False 时 softmax 在 `finfo.min` 上均匀分布
- **修法**: softmax 前加 `has_valid` 短路

---

## Gemini 架构层面的批评

### 1. 代码重复 / hidden coupling（VALID）
> "Three separate, copy-pasted pl.LightningModule classes is a significant flaw. A single, unified, configurable V4Model should have been the objective."

**评估**：公正。v4_minimal / v4_pocket / v4_ultimate 有大量复制粘贴。
**计划**：v4-minimal / v4-pocket baseline 跑通后，重构到单一 `V4Model` 类。**不阻塞训练**，作 v1.1 技术债。

### 2. 数据加载 scalability（VALID at 10x）
> "All precomputed features loaded entirely into memory from .pt files. 10x data → 10x RAM per worker."

**评估**：在当前 25k 规模 RAM ~1-2 GB，**不是瓶颈**；扩到 250k 会 OOM。
**计划**：当前不修；扩数据时 pockets.pt / mol_graphs.pt / rxn_center_mask.pt 全迁到 LMDB。

### 3. Gate 设计过度（DISAGREE）
> "A simpler architecture would concatenate final embeddings and feed to a single MLP."

**评估**：这是有意选择而非 bug。残差分解 `pred = y_seq + g * delta` 是原始人类审稿人**明确肯定**的方向（enzyme baseline + pair-specific deviation 物理直觉明确）。concat+MLP 会让所有 branch 混杂，失去 ablation 可解释性。
**决定**：保持当前设计。论文里对比 "concat + MLP" 作 ablation control。

### 4. README 文档过时（VALID，本次修）
> "README_SERVER.md omits scripts/17, 18, 21."

**评估**：正确。README 停留在 v4-pocket 阶段，缺 v4-ultimate 的 3 个预计算脚本。
**修复**：本次 commit 更新 README_SERVER.md。

### 5. Kitchen sink / 参数过多（VALID 但有 mitigation）
> "6 branches risk parameter inflation and information redundancy."

**评估**：同意。v4-ultimate 作"upper bound"放论文附录；主文 ablation 用 v4-minimal → v4-pocket。

---

## DeepSeek 的幻觉（与实际代码不符，已核验）

| DeepSeek 指控 | 实际代码 |
|---|---|
| RBF bias shape `(B, K, A, R)` 没 permute 到 `(B, H, K, A)` | `int3d_cross_attn.py:67` 已 `bias.permute(0, 3, 1, 2)` |
| Annotation mean pool 没 mask | `v4_ultimate.py` `_bag_pool` 用 `m = (ids > 0).float()` 做 masked mean |
| PCC loss 没 div-zero 保护 | `vx.norm().clamp(min=1e-4) * vy.norm().clamp(min=1e-4)` 已在 |

---

## 修复优先级

| # | 问题 | 来源 | 状态 |
|---|---|---|---|
| 1 | xyz_valid 批泄漏 | Codex + DeepSeek | ✅ 修（f6de404） |
| 5 | v4_ultimate gate bias | Codex | ✅ 修（f6de404）|
| D | README 过时（3 脚本缺失）| Gemini | 🔨 本次修 |
| 2 | feature flag 错位 | Codex | ⏳ 下次 |
| 3 | v4_pocket 缺 use_pocket | Codex | ⏳ 下次 |
| 4 | 全 mask softmax | Codex | ⏳ 下次 |
| A1 | 代码重复 | Gemini | 📝 v1.1 重构 |
| A2 | 数据 scalability | Gemini | 📝 扩数据再改 |
| A3 | Gate 过度设计 | Gemini | ❌ 有意不改 |

---

## 待 Kimi 全仓审查报告

Kimi 返回后会补跨文件一致性检查（script DAG 完整性、config 被消费度、dispatcher 正确性等）。
