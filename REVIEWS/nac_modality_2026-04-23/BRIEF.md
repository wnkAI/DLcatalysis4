# /five review brief — v4-ultimate NAC bias + modality dropout + diagnostics + ablation

You are one of five independent reviewers. Do NOT consult the others. Read the
diff (`diff.patch`), form your own judgment, report what you find.

## Context (kept minimal on purpose)
- DLcatalysis 4.0, enzyme `log10(kcat/Km)` predictor. Repo at
  `E:/AImodel/DLcatalysis4.0/`.
- Core model: `src/model/v4_ultimate.py` (PyTorch Lightning).
  ```
  pred = y_seq
       + g_pair  * (y_sub + y_rxn + y_int3d)
       + g_struct * y_struct
       + g_annot * y_annot
  ```
- 3D cross-attention module: `src/model/int3d_cross_attn.py`
  (pocket residues × substrate atoms, bidirectional).
- Pocket features node_s shape is `(K=32, 26)`: cols 0..19 = AA one-hot
  (STANDARD_AA order in `scripts/14_extract_pockets.py`), col 20 = pLDDT/100,
  col 21 = is_active_site, col 22 = is_binding_site, col 23 = is_cofactor_contact,
  col 24 = is_metal_contact, col 25 = min_dist_to_substrate/10.

## What the two commits changed
1. **Diagnostics + ablation scaffolding** (commit 4f88914).
   - `_log_diag` added to `V4Ultimate.forward` (TensorBoard: branch magnitudes,
     gate means, modality coverage).
   - `scripts/26_generate_ablation_configs.py` emits A0..A8 ablation YAMLs.
   - `REVIEWS/AUDIT_PERFILE.md` per-file risk audit.
2. **NAC bias + modality dropout** (commit 18345ef).
   - `int3d_cross_attn.Int3DCrossAttnLayer`: new `nac_gain` parameter
     (per-head, init 0) and optional `p_nac (B, K)` / `a_nac (B, A)` masks.
     The outer product `p_nac ⊗ a_nac` times `nac_gain` is added to attention
     logits alongside the existing RBF distance bias.
   - `V4Ultimate._encode_pocket` now also returns `pocket_cat (B, K)` — a
     soft catalytic-residue indicator: `max(is_active_site ∨ is_binding_site,
     0.5 * is_AA ∈ {R,D,C,E,H,K,S,Y})`.
   - `V4Ultimate._encode_substrate` now also returns `rxn_center_dense (B, A)`
     unconditionally (no longer gated by `use_rxn_center`).
   - New training-time per-sample Bernoulli dropout of rxn / pocket / annot
     branches via `model.modality_dropout.{rxn,pocket,annot}`. Applied to
     both the pooled embedding (so gates can't cheat) AND the final y_*
     (so head biases don't leak).

## What to audit (ranked)
1. **Correctness bugs.** Tensor shape mismatches, broadcast errors, masked
   ops that silently zero-divide or propagate NaN, wrong dtype / device.
2. **Logic errors.** Does the NAC bias actually do what it claims? Does
   modality dropout behave correctly when multiple modalities are dropped
   together? Does the `keep_*` mask interact correctly with `_log_diag`?
3. **Gradient flow.** Can `nac_gain` learn? Does the `0.5 * is_AA` clamp
   saturate gradients? Does modality dropout at output level vs embed level
   cause gradient inconsistencies?
4. **Data / leakage.** Does anything about the diagnostics or dropout paths
   leak information between train/val/test? Can diagnostics accidentally
   reach the loss?
5. **Numerical.** Softmax over attention logits with added NAC bias — any
   risk of blowup at larger `nac_gain` magnitudes or extreme RBF outputs?
6. **Style / small issues** (report only if you also found a real bug, so
   the signal-to-noise stays high).

## Output format
Report as a bulleted list ordered by severity. For each finding:
- File + line (rough is fine)
- 1-sentence description
- Why it matters
- Suggested fix

If you find no bugs, say so plainly.

Do not rewrite the whole code. Do not praise the diff. Do not produce
executive summaries. We want bug findings only.
