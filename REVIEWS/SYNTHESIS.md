# /five review 综合报告 (2026-04-22)

## Verdict: BLOCK → fixed → 进入 NEEDS_WORK

## 两家命中的共同 bug（已修复，commit pending）

### BUG #1: xyz_valid 批样本级泄漏（Codex + DeepSeek 一致）
- **问题**: `v4_pocket.py` 和 `v4_ultimate.py` 用 `G.MOL_graph_xyz_valid.any()` 批全局开关
- **后果**: 一个样本有效就给**整个 batch** 喂 xyz；无效样本的零坐标进 `cdist` 产生假距离 bias
- **修复**: 改成 per-sample 掩码，`Int3DCrossAttn` 接受 `xyz_valid_per_sample: (B,) bool`，在无效样本上把 bias 置零

### BUG #5: v4_ultimate 没读 `struct.gate_init_bias` (Codex)
- **问题**: v4_pocket 修了（commit 79af90a），v4_ultimate 仍硬编码 -2.0
- **修复**: 同步 v4_pocket 做法，从 `config["model"]["struct"]["gate_init_bias"]` 读

## 其它 Codex 点出的 bug（暂未修，待所有 4 家回完一起处理）

### BUG #2: feature flag 错位
- dataset 自动 disable 缺失模态（DRFP、rxn_center、annot 找不到文件）
- model 还按原 config flag 构建分支（head + MLP）
- 运行时 `_encode_*` 返回零 → head 输出 bias → 静默污染
- **待修**: dataset 应在 disable 后回写 config，model 读 dataset 的"实际可用 flag"

### BUG #3: v4_pocket 没有 use_pocket 开关
- `pocket_path` 缺失时 dataset 默默 `self.pockets = None`
- `_encode_pocket` 返回全零 + mask=None → y_struct 永远 0
- **待修**: 给 v4_pocket 加 `use_pocket` 配置项，和 v4_ultimate 保持一致

### BUG #4: Int3DCrossAttn 全 mask 边界情况
- 如果 pocket_mask 或 atom_mask 全 False，softmax 在 `finfo.min` 上均匀分布 → 输出变成"所有无效 token 的平均"
- **待修**: 在 softmax 前加 `has_valid` 短路

## DeepSeek 的幻觉（与实际代码不符，已核验）

| DeepSeek 指控 | 实际代码 |
|---|---|
| RBF bias shape `(B, K, A, R)` 没 permute 到 `(B, H, K, A)` | `int3d_cross_attn.py:67` 已 `bias.permute(0, 3, 1, 2)` |
| Annotation mean pool 没 mask | `v4_ultimate.py` `_bag_pool` 用 `m = (ids > 0).float()` 做 masked mean |
| PCC loss 没 div-zero 保护 | `vx.norm().clamp(min=1e-4) * vy.norm().clamp(min=1e-4)` 已在 |

## Gemini / Kimi: 仍在跑，回来后补到本文件

## 下一步优先级

1. ✅ 修 BUG #1 + #5（本次 commit）
2. ⏳ Gemini / Kimi 回来 → 更新本文件
3. 🔨 BUG #2-4 统一修：dataset ↔ model flag 同步机制
4. 🔨 CD-HIT 40% split（审稿人 #1 优先级，独立任务）
