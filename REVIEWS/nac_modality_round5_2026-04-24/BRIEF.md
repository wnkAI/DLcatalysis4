# /five round-5 brief — verify round-4 fixes

Round-4 review on commit `b651165` flagged 2 issues. Commit `e344130
"round-4 fixes"` addresses them. Verify.

Diff: `REVIEWS/nac_modality_round5_2026-04-24/diff.patch` (60 lines,
src/ only).

## Round-4 fix list

| ID | Round-4 finding | Round-4 fix |
|----|-----------------|-------------|
| F  | `_encode_annot` zeroed the whole branch when *any* one of ANNOT_ipr_ids / ANNOT_pf_ids / ANNOT_go_ids was None | Per-field degrade: the branch is dark only when all three are None; missing fields contribute a zero (B, embed_dim) pool while present fields are pooled normally |
| G  | `B = getattr(G, "num_graphs", None) or 1` silently coerced `num_graphs=0` to 1 | Changed to `B = getattr(G, "num_graphs", 1)` so explicit 0 is preserved |

## What to check

1. **Fix F (annot per-field).** For partial annotations:
   - ipr present, pf & go None → pooled ipr + zeros + zeros → annot_proj
     gets `(B, embed_dim + 0 + 0)` ? — check that the concatenated input
     to `annot_proj` is still `(B, 3 * embed_dim)` because each `_bag_pool`
     returns `(B, embed_dim)` whether or not the ids tensor is None.
   - All three None → early return zeros. OK.

2. **Fix G.** `getattr(G, "num_graphs", 1)` returns 1 when attr is missing,
   and preserves 0 when it's explicitly set. Correct.

3. **Regressions.** Any path where a present `ipr_ids` but `padding_idx=0`
   for all entries (fully-padded) causes issues? `_bag_pool` uses
   `(ids > 0).float()` so fully-zero ids produce mask=0 → denominator
   clamped to 1 → returns zero pool. OK.

4. **Anything else you see broken** after 5 rounds of iterative fixes.

## Output format

Bulleted list, severity-ordered. End with `APPROVED` /
`NEEDS_WORK` / `BLOCK`. If clean, say "no bugs found."

No preamble.
