# /five round-4 brief — verify round-3 fixes

Round-3 review found 2 remaining issues on commit 702fedc (round-2 fixes).
Commit `b651165 "round-3 fixes"` claims to fix them. Verify.

Repo: `E:/AImodel/DLcatalysis4.0`
Diff: `REVIEWS/nac_modality_round4_2026-04-24/diff.patch` (230 lines).

## Round-3 fix list

| ID | Round-3 claim |
|----|---------------|
| D  | Every remaining `hasattr(G, "...")` in v4_ultimate.py replaced by `getattr(G, "...", None) is not None`, so present-but-None attrs no longer crash the forward path. `grep "hasattr(G, "` should return 0 matches. |
| E  | `y_pre` now stores `(tensor, mask01)` tuples. `_log_diag` computes kept-subset weighted mean so dropped samples' head(0)=head_bias doesn't underestimate branch magnitude. |

Also: stale "see _keep" comment in the output-dropout block was updated (function had been renamed to `_masks` in round-2).

## What to check

1. **Fix D — zero hasattr remaining.** Run a mental grep: is there ANY
   `hasattr(G, "...")` still reachable in `v4_ultimate.py`? The module
   should be clean. If a previously missed site remains, flag it.

2. **Fix E — kept-subset mean.** `(|y| * w).sum() / w.sum().clamp(1)`:
   - For y_seq/y_sub: w is all-ones tensor → reduces to `|y|.sum() / B`
     = plain mean. OK.
   - For y_rxn/y_struct/y_int3d/y_annot: w = mask01 ∈ {0,1}. Sum of w
     is the kept-count. Numerator is sum-of-abs over kept samples.
     Result is kept-subset mean. Correct semantics.
   - Edge case: entire batch dropped (w.sum()=0). Denominator clamped
     to 1 avoids div-by-zero; result is 0. Acceptable.

3. **Regressions.** Any new bug introduced by:
   - Changing encoder-entry early returns from `hasattr` to
     `getattr is not None`? Particularly for `_encode_condition`
     and `_encode_annot` which now bail if *any* of the required
     tensors is None (e.g., ANNOT_ipr_ids present but pf_ids None).
     Is this the right semantics, or should each tensor be checked
     independently with default zeros?
   - Changing `B = G.num_graphs if hasattr(G, "num_graphs") else 1`
     to `B = getattr(G, "num_graphs", None) or 1` — the `or` trick
     means `num_graphs=0` would fall back to 1. Is that correct or
     should it be `getattr(...) if getattr(...) is not None else 1`?
   - New `(tensor, mask)` tuple in y_pre. Any risk of double-detach,
     gradient leak, or memory explosion?

4. **Full-forward smoke.** Trace one kept sample at p_pocket=0.15 for
   pocket branch:
   - mask01_pocket=1, keep_pocket=1/(1-0.15)≈1.176.
   - pocket_pool unchanged by mask01_pocket=1.
   - pocket_tokens unchanged by mask01_pocket.unsqueeze(-1)=1.
   - y_struct = head_struct(pocket_pool) = y_struct_eval.
   - After output scaling: y_struct * 1.176.
   - In residual: g_struct * y_struct * 1.176.
   - At eval same sample: mask01=1, keep=1, so y_struct * 1 = y_struct_eval.
   - Expected contribution during training = (1-p) * y_struct_eval * (1/(1-p)) + p * 0 = y_struct_eval. ✓

## Output format

Bullets, severity-ordered. File + line, 1-sentence description, impact,
fix. End with `APPROVED` / `NEEDS_WORK` / `BLOCK`.

"No bugs found" if clean. No preamble, no praise.
