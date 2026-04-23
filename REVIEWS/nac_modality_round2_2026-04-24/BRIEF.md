# /five round-2 brief — verify round-1 fixes landed cleanly

You are one of five reviewers. Do NOT read other reviewers' output. Form
your own judgment.

## Context

Round-1 review (see `REVIEWS/nac_modality_2026-04-23/SYNTHESIS.md`) found
10 bugs. The commit `69d1f94 "/five round-1 fixes"` attempts to fix all
of them. Your job is to verify, on the ACTUAL CODE, that each fix is
correct AND did not introduce new bugs.

Repo root: `E:/AImodel/DLcatalysis4.0`. The diff being reviewed is in
`REVIEWS/nac_modality_round2_2026-04-24/diff.patch`.

## Fix list to verify

| ID | What round-1 claimed to fix | Files touched |
|----|-----------------------------|---------------|
| H1 | A0_seq ≡ A1_sub → added `use_substrate` flag, gated substrate_gnn/head_sub/_encode_substrate/y_sub, updated LADDER, regenerated 9 ablation YAMLs | `v4_ultimate.py`, `26_generate_ablation_configs.py`, `config/v4_ultimate.yml`, `config/ablation/*.yml` |
| M1 | `cov_*` diagnostics were batch-level booleans → now per-sample via `mask.any(dim=1).float().mean()`, RXN validity via `drfp.abs().sum > 0` fallback | `v4_ultimate.py` |
| M2 | Modality dropout was not inverted-scaled → `mask / (1-p)` added so `E[keep*x] == x` | `v4_ultimate.py` |
| M3 | Pocket dropout leaked gradient to pocket GVP via `pocket_tokens` → now `pocket_tokens = pocket_tokens * keep_pocket.unsqueeze(-1)` before int3d call | `v4_ultimate.py` |
| M4 | `_log_diag` logged post-dropout `y_*` → now snapshots `y_pre = {...}` dict before dropout and logs pre-dropout magnitudes | `v4_ultimate.py` |
| L1 | `hasattr` didn't guard None-valued attrs → replaced with `getattr(..., None) is not None` | `v4_ultimate.py` |
| L3 | `to_dense_batch` was called twice for rxn_center → now once as float, cast to long for Embedding | `v4_ultimate.py` |
| L4 | `nac_gain` unbounded → now `tanh(nac_gain) * nac_max` with `nac_max=3.0`, init still zero | `int3d_cross_attn.py` |
| L5 | No schema assertion on `node_s` → `assert node_s.shape[-1] >= 23` added | `v4_ultimate.py` |

## What to check (ordered)

1. **Each fix actually landed in the code** — don't trust the commit
   message, read the function bodies.
2. **Each fix is correct** — does the math/logic achieve the claim?
   In particular:
   - **M2 inverted scaling** at p=0 should be a no-op (mask/1.0 ≡ mask).
     Does `max(q, 1e-6)` ever produce weird values when p=1.0?
   - **M3 pocket_tokens gate**: `pocket_tokens * keep_pocket.unsqueeze(-1)`
     broadcasts `(B, 1, 1)` against `(B, K, D)`. Correct.
     But with inverted scaling enabled, `keep_pocket` has magnitude
     `1/(1-p)` for kept samples — so `pocket_tokens` is now *amplified*
     by `1/(1-p)` in the kept subset. Is this the intended interaction?
   - **M4 y_pre** uses `.detach()` everywhere; no gradient leaks through
     diagnostics. Verify.
   - **H1 use_substrate**: with `use_substrate=False`, `graph_emb` is a
     zero tensor. The `pair_gate` still receives it as input via
     `pair_gate_in = [enz_fused, graph_emb, ...]`. The gate's linear
     layer still has a `W_graph` column; is this an acceptable design
     (just learns zero contribution) or should `graph_emb` be dropped
     from the gate when `use_substrate=False`?
   - **L4 nac_max=3.0**: bias in [−3, 3] added to attention logits.
     Softmax with logit range ~3 gives attention weights in ratio
     exp(3)/exp(-3) ≈ 403. Reasonable or still too strong?
3. **Did the fixes introduce regressions?** Especially:
   - `use_substrate=False` path — does the model still run when
     `_encode_substrate` returns `(None, zeros, None, None)`? The int3d
     block checks `atom_tokens is not None` — so int3d is skipped. OK.
     But `rxn_center_dense` returned from `_encode_substrate` is also
     None in that path — passing `a_nac=None` to int3d skips NAC bias.
     The int3d block is already gated by `atom_tokens is not None`, so
     NAC being skipped inside is irrelevant. Double-check.
   - `use_substrate=False` but `use_rxn_center=True` in config — the
     `if self.use_rxn_center and self.use_substrate` guard now controls
     `rxn_center_emb` construction. If a config has one true and the
     other false, what happens? (A0 has both false, A1+ both true.)
4. **Anything NOT in the fix list that's newly broken** — modality
   dropout's new `/ (1-p)` path; new pocket_tokens gate broadcast;
   the y_pre dict allocation; the int3d nac_max attribute.

## Output format

Bulleted list, ordered by severity. Each item:
- File + line (rough is fine)
- 1-sentence description
- Why it matters
- Suggested fix

Verdict at the end: **APPROVED** / **NEEDS_WORK** / **BLOCK**.
Say "no bugs found" plainly if that's the case.

No preamble, no executive summaries, no praise.
