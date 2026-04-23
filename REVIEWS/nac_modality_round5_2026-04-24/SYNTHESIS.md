# /five synthesis — round-5 (final)

Date: 2026-04-24
Reviewers: Codex, DeepSeek, Gemini, Kimi, internal Claude.

## Verdict: **APPROVED (5/5 reviewers, 0 bugs)**

## Convergence history

| Round | Commit   | Bugs flagged   | Reviewers |
|-------|----------|----------------|-----------|
| 1     | 18345ef  | 10 bugs        | 4 (Gemini Windows stdin bug) |
| 2     | 69d1f94  | 3 regressions  | 5 |
| 3     | 702fedc  | 2 remaining    | 5 (Kimi session log bug) |
| 4     | b651165  | 2 regressions  | 5 |
| 5     | e344130  | 0 bugs         | 5 all APPROVED |

## Round-5 findings

- **Codex**: No bugs found. APPROVED.
- **DeepSeek**: No bugs found. APPROVED.
- **Gemini**: No bugs found. Explicit verification of Fix F dim match,
  Fix G num_graphs=0 preservation, zero-pad mask behavior, and empty
  batch (B=0) propagation. APPROVED.
- **Kimi**: No bugs found. APPROVED.
- **Internal Claude code-reviewer**: No bugs found. Explicit confirmation
  that `_bag_pool` returns `(B, embed_dim)` for both None and present
  ids, `annot_proj` input width 3*embed_dim, `getattr` returns 0 when
  attr explicitly 0. APPROVED.

## What's in the codebase now (end of loop)

Module: `src/model/v4_ultimate.py`, `src/model/int3d_cross_attn.py`

- **Dropout**: two masks per modality (`mask01` for zeroing input/token,
  `keep` for inverted scaling at output), single scaling factor in the
  predictor path, train/eval consistent under standard inverted-dropout
  semantics.
- **NAC bias**: catalytic-residue × reaction-center outer product added
  to int3d attention logits, per-head gain bounded by
  `tanh(nac_gain) * nac_max` with `nac_max=3.0`, init at zero.
- **Modality ablation flags**: `use_substrate`, `use_rxn_drfp`,
  `use_rxn_center`, `use_pocket`, `use_int3d`, `use_annot`,
  `use_condition`, `use_ec`. All exercised by `config/ablation/A0..A8`.
  Gate input dims match forward list construction for every
  combination.
- **None safety**: zero `hasattr(G, "...")` patterns left; every
  attribute access uses `getattr(..., None)`. `_encode_annot` degrades
  per-field (one of {ipr, pfam, go} missing → zero pool for that slot,
  pool the present fields normally).
- **Diagnostics**: `_log_diag` logs branch magnitudes on the kept
  subset (no head-bias leak from dropped samples), gate means, and
  per-sample modality coverage (mask.any(dim=1).float().mean()).
- **Ablation**: `config/ablation/v4_ultimate_A{0..8}_*.yml` regenerated
  from the current base config, including the modality_dropout block.

## All fixes, chronologically

Round 1 (H1/M1-M4/L1/L3/L4/L5, commit 69d1f94):
- H1 use_substrate flag + LADDER A0 fix
- M1 per-sample cov_* diagnostics
- M2 inverted dropout scaling (first attempt)
- M3 pocket_tokens gated under pocket dropout
- M4 _log_diag on pre-output y_*
- L1 getattr None guards in diagnostics
- L3 single to_dense_batch in _encode_substrate
- L4 nac_gain bounded via tanh
- L5 node_s shape assert

Round 2 (split masks + complete None + gate dims, commit 702fedc):
- Split `keep_*` into `mask01` (no scaling) + `keep` (inverted), applied
  only once each (A)
- Full forward-path None guards for the 4 sites flagged by reviewers (B)
- `pair_in` / `struct_in` / forward list conditional on use_substrate (C)

Round 3 (None sweep + kept-subset mean, commit b651165):
- Every remaining `hasattr(G, "...")` replaced with `getattr(...) is not
  None` (D)
- `y_pre` switched to (tensor, mask01) tuples; `_log_diag` computes
  kept-subset mean (E)

Round 4 (annot per-field + num_graphs=0, commit e344130):
- `_encode_annot` per-field fallback (F)
- `B = getattr(G, "num_graphs", 1)` preserves explicit 0 (G)

Round 5 (this doc): no bugs.

## Closing

The iteration showed the typical /five pattern: round-N tends to find
round-(N-1) fix over-corrections. The code converged in 5 rounds with
no reviewer producing a new finding on the final pass.
