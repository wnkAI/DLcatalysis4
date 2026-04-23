# /five round-3 brief — verify round-2 fixes

Round-2 review of commit `69d1f94` (round-1 fixes) found 3 issues on
commit 69d1f94. Commit `702fedc "round-2 fixes"` attempts to fix them.
Your job: verify the round-2 fixes are correct and didn't introduce
new bugs.

Repo: `E:/AImodel/DLcatalysis4.0`
Diff: `REVIEWS/nac_modality_round3_2026-04-24/diff.patch` (185 lines,
code-only).

## Round-2 fix list

| ID | Round-2 finding | Round-2 fix |
|----|-----------------|-------------|
| A  | Inverted dropout mask `keep_*` applied at embed + token + output levels → compounded scaling, softmax temperature shift in int3d | Split into `mask01` (0/1, used at embed/token) and `keep` (inverted, used only at output) |
| B  | L1 incomplete — `hasattr` still used in forward path for `MOL_rxn_center`, `MOL_graph_xyz`, `MOL_graph_xyz_valid`, `ANNOT_has_any`, `ANNOT_has_cof` | Replaced with `getattr(..., None) is not None` at all 4 call sites |
| C  | `graph_emb` still fed to pair_gate / gate_struct when `use_substrate=False`, wastes parameters | Made `pair_in` / `struct_in` dim computation + forward list construction conditional on `use_substrate` |

## What to check

1. **Fix A (split mask).** For kept samples at training with p=0.15:
   - Input path: `x * mask01_rxn = x * 1 = x` (not amplified)
   - Head: `y = head(x)` (no shift)
   - Output: `y * keep_rxn = y * (1 / 0.85) ≈ y * 1.176` (inverted-scaled once)
   - Eval (keep_*, mask01_*) both all-ones, so `y_* * keep_* = y_*`.
   Trace the math and confirm E[output_contribution_train] == output_contribution_eval.

2. **Fix A interaction with diagnostics.** `y_pre` is now populated
   from `y_*` BEFORE the output-level dropout multiplication. Verify
   `y_pre` values match the *eval-time* magnitudes (kept samples,
   since mask01 didn't scale them).

3. **Fix B.** Every forward-path attribute access that used
   `hasattr(G, "X")` and then `.to(self.device)` should now use
   `_x = getattr(G, "X", None); if _x is not None: _x.to(...)`.
   Look for *remaining* hasattr calls in the forward path that still
   have the old pattern (the fix was only claimed for 4 sites; were
   any missed?).

4. **Fix C.** The conditional pair_in / struct_in dims and the
   conditional forward list construction must stay perfectly in
   sync — if they diverge, torch.cat will throw at runtime.
   Specifically check: when `use_substrate=False`, `use_pocket=True`,
   `use_rxn_drfp=False`, `use_ec=False`, is `pair_in` correctly
   `hidden_dim` (enz only)? And for struct_in: is it
   `2*hidden_dim + 3` (enz + pocket + quality)? Then in forward,
   does the list concat produce a matching tensor?

5. **Regressions.** Any new bug introduced? Watch for:
   - Broadcasting errors in `pocket_tokens * mask01_pocket.unsqueeze(-1)`.
   - `mask01` is (B, 1); `pocket_tokens` is (B, K, D); `.unsqueeze(-1)`
     makes (B, 1, 1). OK.
   - `keep_rxn` vs `mask01_rxn` are both (B, 1); all `y_*` are (B, 1).
     OK.

## Output format

Bulleted list, ordered by severity. Each item: file + line,
1-sentence description, why it matters, suggested fix.

End with `APPROVED` / `NEEDS_WORK` / `BLOCK`.

If no bugs, say "no bugs found" plainly. No preamble, no executive
summaries.
