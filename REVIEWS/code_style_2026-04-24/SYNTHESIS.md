# /five code-style + redundancy synthesis
Date: 2026-04-24
Reviewers: Codex, DeepSeek, Gemini, internal Claude. (Kimi stalled on
its own session-log PermissionError; output unusable.)

## Consensus findings — what 2+ reviewers flagged independently

### Tier A — Trivial deletes, zero risk (BATCH 1)

**A1.** `src/model/module.py` carries ~280 lines of v3.0 dead code that
no file under `src/` imports. Keep only `MLP` + `NONLINEARITIES`.
Dead: `split_tensor_by_batch`, `concat_tensors_to_batch`,
`split_tensor_to_segments`, `split_tensor_by_lengths`,
`batch_intersection_mask`, `MeanReadout`, `SumReadout`,
`MultiLayerPerceptron`, `SmoothCrossEntropyLoss`, `GaussianSmearing`,
`SinRBF`, `ShiftedSoftplus`, `compose_context`, `compose_batch_context`,
`get_complete_graph`, `compose_context_stable`, `Swish`, `_WeightedLoss`
import, `scatter_mean`/`scatter_add` imports, `PI` import, `__main__`
test block. (internal Claude DC1, Codex, DeepSeek).

**A2.** `src/util/data_load.py` contains a second orphaned structure
pipeline: `StructureComplexData`, `padding_struct_lig_embedding`,
`PDBProtein`, `get_ligand_atom_features`, `parse_sdf_file_mol` — zero
callers in `src/`. Only `SequenceData` + `padding_seq_embedding` are
still used (by `data_module.py` and `seq_process.py`). Keep the two
live classes, move them to `seq_process.py`, delete the rest (removes
~300 lines and rdkit import from hot path).
(internal Claude DC2, Codex).

**A3.** `src/util/constants.py` — once A2 lands, has zero importers.
Delete. (internal Claude DC3).

**A4.** `src/util/tools.py` — `init_config` and `df_split` have zero
callers. Keep only `set_seed`. File shrinks 70 → 21 lines.
(internal Claude DC4).

**A5.** Stale config keys in all 13 YAML files — never read by any
Python code: `embedding: {morgan, molt5, grover, unimol, pro_seq}`,
`use_msa`, `feature_norm`, `sched_factor`, `sched_patience`,
`valid_freq`. Delete from every config. (internal Claude DC5+DC6).

**A6.** Dead imports across `data_load.py`, `data_module.py`,
`metrics.py`: `set_seed`, `pl`, `Compose`, `DataLoader`, `Tuple`,
`pickle`, `List`, `Dict`, `Optional`, `kendalltau`, `ChemicalFeatures`,
`RDConfig`. (Codex).

**A7.** Lazy module-level imports inside hot-path methods in
`v4_ultimate.py` (line 343, 405, 559: `from torch_geometric.utils import
to_dense_batch` three times inside methods; line 761: `from collections
import defaultdict` inside `_pairwise_ranking`). Hoist to top of file.
(internal Claude OE1).

**A8.** Double bias init in `v4_ultimate.__init__` (lines 265, 275) and
`v4_pocket.__init__` (lines 179, 182) — first init is zeroed by
`_init_weights()` immediately after. Remove the first init.
(internal Claude OE2).

### Tier B — Low-risk polish (BATCH 2)

**B1.** `run_train.py` missing `--smoke` flag. Add:
`--smoke` → `limit_train_batches=5, limit_val_batches=5, max_epochs=2`.
(internal Claude RT2).

**B2.** `run_train.py:70` — `args` parameter passed but never used.
Remove or wire for `--smoke`. (internal Claude RT3).

**B3.** `scripts/20_run_vina_gpu.py` — `run_qvina2` and `run_vina_gpu`
are 90% identical (CLI build, subprocess.run, log parse). Consolidate
into `run_docking(binary, ...)`. (Gemini).

**B4.** `scripts/24_pair_pipeline.py` — `parse_annotation_sites_safe`
re-implements guards that `scripts/14_extract_pockets.py:parse_annotation_sites`
already does. Drop and call through. (Gemini).

**B5.** `scripts/11_precompute_mol_graphs.py` — `_safe_idx(mapping, key,
default=0): return mapping.get(key, default)` is a wrapper around
`dict.get`. Inline. (Gemini).

**B6.** `scripts/10_prepare_model_inputs.py::parse_ec` — 10-line
try/except/loop replaceable by 2-line comprehension. (Gemini).

**B7.** `scripts/14_extract_pockets.py` — `_import_bio()` lazy-load
pattern clutters each function. Hoist BioPython imports to top;
fail-fast is preferable. (Gemini).

**B8.** `_pairwise_ranking` — `from collections import defaultdict`
inside method body. Hoist. (Codex).

**B9.** `mask < 0.5` check in `_attn_pool_seq` is redundant — mask is
already a 0/1 float, convert to bool directly. (DeepSeek).

### Tier C — Structural refactor (BATCH 3, NEEDS USER APPROVAL)

**C1.** Extract `KcatTrainingMixin` or `BaseKcatLightningModule`.
Duplicated across v4_minimal / v4_pocket / v4_ultimate (identical):
`_prepare_target`, `_pairwise_ranking`, `get_loss`, `calculate_metrics`,
`training_step`, `validation_step`, `test_step`, `_flush_epoch`,
`on_*_epoch_end`, `_save_results`, `configure_optimizers`,
`_init_weights`, `_encode_ec`. Total savings: ~600 lines across 3 files.
v4_innovate stays separate (multi-task). (Codex, internal Claude D1-D5,
DeepSeek).

**C2.** Extract shared `masked_attn_pool(tokens, mask, head)` util —
implemented 5 times (v4_minimal, v4_pocket, v4_ultimate, v4_innovate,
int3d_cross_attn, gvp_pocket). (Codex, internal Claude D6).

**C3.** Extract `encode_substrate_graph` / `encode_pocket_core` helpers
— triplicated stems across v4_pocket/v4_innovate/v4_ultimate, with
model-specific extensions layered on top. (Codex).

**C4.** `v4_minimal` and `v4_pocket` have inconsistent signatures for
`_attn_pool_seq` vs `v4_ultimate` (with/without `pool_head`). Unify.
(internal Claude N1, DeepSeek).

**C5.** `get_model_class` routing fragile — silently falls through to
V4Minimal if `model_name` doesn't contain "ultimate" and all ultimate
flags are off. Add explicit `model_class` config key. (internal Claude RT1).

## Top priorities

From my reading of the 4 reviews:

1. **A1–A6** (dead code sweep) — trivial, zero risk, biggest
   readability payoff for the effort. DO NOW.
2. **B1–B2** (`--smoke` flag + drop unused arg) — saves hours of
   debug time on server runs. DO NOW.
3. **A7, A8, B8, B9** (import hoisting + init redundancy) — zero risk.
   DO NOW.
4. **B3–B7** (pipeline script polish) — one file each, low risk.
   DO NOW or as follow-up.
5. **C1** (KcatTrainingMixin) — highest code-mass payoff, ~medium risk
   because it changes the inheritance chain of 3 training models.
   Should be its own commit with focused smoke-test after. ASK USER.
6. **C2–C5** — defer until C1 is in.

## Execution plan for this session

- Apply A1–A8 and B1–B9 in one commit each group (dead-code, polish).
- Leave Tier C for a follow-up commit with user approval.

Run the smoke (`scripts/27_smoke_forward.py`) after each batch to verify
nothing broke.
