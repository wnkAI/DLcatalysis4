# DLcatalysis 4.0 — Per-file audit

Date: 2026-04-23
Commit at audit: `3309c17` + follow-up B1/B2 patches.
Scope: every file under `src/` and `scripts/` touched by the v4 work.
Each entry: **purpose / invariants / known risks / status**.

Status legend:
- `OK`       — read, cross-checked, no open issues
- `WATCH`    — works but fragile or under-tested, listed risks stand
- `TODO`     — logic gap acknowledged; tracked below
- `STUB`     — not yet used in training; safe to ignore for v4 run

---

## src/model/

### v4_ultimate.py (status: WATCH)
- **Purpose.** 6-branch residual: seq + gate_pair·(sub+rxn+int3d) + gate_struct·struct + gate_annot·annot.
- **Invariants.**
  - `pair_delta = y_sub + y_rxn + y_int3d`; branches disabled via flags return zero-tensors, so the sum is safe.
  - `graph_emb` picks up rxn-center contribution via mask-weighted mean (no padding bias).
  - `gate_struct` bias is read from `struct.gate_init_bias` and applied twice (in `__init__` body + after `_init_weights`) so Xavier re-init cannot wipe it.
  - `use_uncertainty` path now respects `loss_type`: per-sample loss is built from the configured loss first, then aleatoric-weighted.
  - `_log_diag` is guarded by `self._trainer is not None` so forward() still runs in scripted inference.
- **Risks.**
  - `y_int3d` head has input dim `hidden_dim * 2`; if int3d is off the head is never built — check training scripts don't index `self.head_int3d`.
  - `g_annot` bias init is `-1.0`; if annotation coverage is low, the gate may saturate closed and annotation branch won't train — diagnostics (`diag/g_annot_mean`, `diag/cov_annot`) are there precisely to catch this.
- **Status.** B1 diagnostics added. Needs a smoke run before claiming green.

### v4_pocket.py (status: OK)
- **Purpose.** 4-branch (seq + sub + rxn + pocket) model — predecessor to v4_ultimate, still used as fallback config.
- **Invariants.** `has_annot`/`has_cof` now read real fields via `_pull()` (commit 3309c17); no longer stubbed zeros.
- **Risks.** None known. Shares pocket/rxn encoders with v4_ultimate.
- **Status.** OK after fix.

### v4_innovate.py (status: STUB)
- **Purpose.** Novelty-focused 3-head multi-task model (kcat / Km / kcat/Km) with physical consistency loss.
- **Risks.** Not yet wired into training driver; no experiment has run end-to-end.
- **Status.** Kept for paper's "novelty" path; not on the critical run for the first submission.

### v4_minimal.py (status: OK)
- **Purpose.** 3.0-equivalent seq + sub baseline, kept for reproducing 3.0 numbers.
- **Status.** Unchanged by 4.0 work.

### gvp_pocket.py (status: OK)
- **Purpose.** GVP-GNN pocket encoder; returns `(tokens, pool, mask)`.
- **Invariants.** `in_s_dim=26` matches `backbone_features_v2`; `in_v_dim=2` covers (CA→N, CA→C) unit vectors.
- **Status.** Unchanged; consumed by v4_pocket and v4_ultimate.

### int3d_cross_attn.py (status: WATCH)
- **Purpose.** Cross-attention between pocket residues and substrate atoms with RBF-distance bias.
- **Risks.** Silently falls back when 3D coords are invalid per-sample (`xyz_valid_per_sample`). The `diag/cov_mol_xyz` stream (B1) exposes coverage so we can see if the 3D channel is in fact running.
- **Status.** WATCH until first run confirms coverage > 0.

### substrate_gnn.py (status: OK)
- **Purpose.** GINE substrate encoder; outputs per-atom tokens + graph pool + atom mask.
- **Status.** Unchanged.

### ec_cross_attn.py (status: STUB)
- **Purpose.** EnzymeCAGE-style cross-attention + pocket-weight module, imported for v4_innovate.
- **Status.** Not on v4_ultimate path.

### module.py (status: OK)
- **Purpose.** Small MLP / pool utility.
- **Status.** Unchanged.

---

## scripts/ (data & training)

### 00..05 (SMILES recovery + dataset integration) (status: OK)
Run-once ETL. `05_integrate_final_dataset.py` produces the master CSV that downstream scripts key on. Not exercised in v4 work besides dataset consumption.

### 10_prepare_model_inputs.py / 11_precompute_mol_graphs.py (status: OK)
- Produces `mol_graphs.pt` keyed by `smi_id`; graph dicts include `x`, `edge_index`, `edge_attr`, `num_nodes`, optional `xyz` + `xyz_valid`.
- **Risk.** Downstream code assumes `xyz_valid` is a bool per-sample; verified by v4_ultimate before use.

### 12_precompute_prot5_lmdb.py (status: OK)
- Writes ProtT5 embeddings into LMDB keyed by `seq_id`.

### 13_make_cv_splits.py (status: OK)
- Produces `train/valid/test_fold{k}.csv`.

### 14_extract_pockets.py (status: WATCH)
- **Purpose.** Extract K=32 pocket residues within 8 Å of docked pose (EnzymeCAGE-exact).
- **Invariants.** `backbone_features_v2` produces `node_s.shape == (K, 26)`; pipeline pads to K when fewer residues found.
- **Risks.** Pocket source tagging (`docked_pose` vs fallback) is used by 25's fallback-rate check — any new fallback path must set `pocket_source` explicitly.
- **Status.** WATCH until 25 validates.

### 15_build_structure_manifest.py / 16_fetch_alphafold.py (status: OK)
Structure prep; unchanged by v4 review.

### 17_precompute_drfp.py (status: OK)
- Writes `rxn_drfp.npy` aligned to `rxn_drfp_keys.csv`.

### 18_precompute_rxnmapper.py (status: OK)
- Writes per-substrate reacting-center atom mask `rxn_center_mask.pt`.

### 20_run_vina_gpu.py (status: WATCH)
- **Purpose.** QVina2 / Vina-GPU driver per pair.
- **Invariants.** Per-pair seed = `hash(global_seed:uid:smi_id)`, exhaustiveness=8.
- **Risks.** QVina2 timeout fallback; any silent failure must emit `run_status` row so 25 catches the coverage hit.
- **Status.** WATCH pending full pair-run results.

### 21_parse_annotations.py (status: OK)
- Ingests InterPro / Pfam / GO into `enzyme_annotations.pt` + vocab JSON.

### 22_build_enzyme_groups.py (status: OK)
- Produces enzyme-level grouping used by `rank_loss` hook.

### 23_augment_targets_and_scaffold.py (status: OK)
- Adds LOG_KCAT / LOG_KM / LOG_KCAT_KM and Murcko `SCAFFOLD_ID`.

### 24_pair_pipeline.py (status: WATCH)
- **Purpose.** 3-phase docking driver: prep receptors → prep ligands → parallel dock+extract → aggregate.
- **Invariants.** Uses `importlib.util.spec_from_file_location` to import sibling `14_*` / `20_*` scripts whose filenames start with digits.
- **Risks.**
  - Phase 3 is parallel; `run_status.csv` appends are not locked — if two workers collide, rows may interleave. For small N this is fine; for a full run, either serialize writes or switch to per-worker status files.
  - Phase 4 aggregator reads per-pair `.pkl` blobs and builds `pockets_pair.pt`; must match the dict shape 25 expects (`node_s`, `node_v`, `ca_xyz`, `n_residues`, `dock_score`, `pocket_source`).
- **Status.** WATCH until a real pair run completes and 25 signs off.

### 25_validate_pair_pockets.py (status: OK)
- **Purpose.** Post-run hard/soft validation of `pockets_pair.pt`.
- **Invariants.** Exit codes 0 (READY) / 1 (STOP) / 2 (WARN) used by CI wrappers.
- **Status.** OK — read-only, exits early on missing artifact.

### 26_generate_ablation_configs.py (status: OK)
- **Purpose.** Emits A0..A8 ablation YAMLs with disambiguated ckpt/log paths.
- **Invariants.** Each row inherits from the one above; A8 == `v4_ultimate.yml` semantically.
- **Status.** OK — generated 9 configs, spot-checked A0/A4/A8 flags.

---

## src/util/

### data_module.py (status: WATCH)
- **Purpose.** PyG data representer; attaches all modality tensors under namespaced keys (`SEQ_*`, `MOL_*`, `POCKET_*`, `RXN_*`, `ANNOT_*`, `COND_*`, `EC_*`, `Y_*`).
- **Invariants.** `__inc__` returns 0 for all scalar-per-graph keys so batching doesn't offset them.
- **Risks.** Any new key added to the model must also be declared in `__inc__` / `follow_batch` lists, or PyG will try to shift it as an edge-index-like tensor.
- **Status.** WATCH — future contributors must remember the __inc__ contract.

### featurize/seq_prot5.py (status: OK)
- Reads precomputed ProtT5 from LMDB; `precomputed_only=True` is enforced upstream.

### metrics.py (status: OK)
- Within-enzyme Spearman / top-1 hit / pair accuracy; unit-tested against synthetic data (within-Spearman 0.78 at noise=0.5).

---

## Known open items

1. **v4_ultimate smoke run** — no end-to-end run with all 6 branches on + diagnostics yet. Until then the branch-magnitude / gate-mean expectations in this doc are hypotheses.
2. **run_status.csv write contention (24)** — benign for current N; revisit if we scale past ~4 workers.
3. **v4_innovate training driver** — model exists, trainer entry does not; novelty path is paper-facing, not on the first-submission critical path.
