# /five synthesis — v4-ultimate NAC + modality dropout + diagnostics + ablation
Date: 2026-04-23
Reviewers: Codex, DeepSeek, Kimi, internal Claude code-reviewer. Gemini
hung on stdin pipe (Windows arg-list / credential issue), no output.

All findings below are reproduced against the on-disk code, not taken on
faith from the reviewer.

## HIGH — blocks the ablation matrix

### H1 — A0_seq is functionally identical to A1_sub
- Reporters: Codex (and internal Claude post-verification).
- Location: `scripts/26_generate_ablation_configs.py:41-49`, `src/model/v4_ultimate.py:115`, `src/model/v4_ultimate.py:471`.
- The ablation ladder uses flag deltas like `("A0","seq", {...use_rxn_drfp=False, use_pocket=False, ...})`. But V4Ultimate has **no** `use_substrate` flag — `SubstrateGINE` is always constructed and `y_sub = self.head_sub(graph_emb)` always runs. Verified: `grep use_substrate` returns 0 hits in the model and the generator.
- Impact: A0 ≠ "seq only", it's "seq + substrate GINE + rxn_center_emb (since rxn_center atom injection is independent of rxn_drfp)". Any paper claim about "incremental contribution of substrate signal" is wrong as-is.
- Fix: add a `use_substrate: bool` flag, gate `_encode_substrate` output and `y_sub` on it, set `use_substrate=False` in A0, regenerate configs.

## MEDIUM — correct-ish behavior but bad signal

### M1 — `cov_*` diagnostics are per-batch constants, not per-sample fractions
- Reporters: Codex, internal Claude.
- Location: `src/model/v4_ultimate.py:595-601`.
- `cov_rxn_drfp = float(hasattr(G, "RXN_drfp"))` → 0 or 1 for the whole epoch (dataloader either attaches it on all samples or none).
- `cov_atoms = float(atom_mask.any().item())` → 1 whenever *any* atom in the batch is valid, which is always.
- Same for `cov_pocket`.
- Impact: The stated goal was "fraction of the batch with real signal". These metrics don't measure that; they're pegged at ~1.0 forever. Useless for catching missing-modality bugs.
- Fix: `atom_mask.any(dim=1).float().mean()`, `pocket_mask.any(dim=1).float().mean()`. For RXN, add a per-sample `RXN_drfp_valid` bool at dataloader level; here log `G.RXN_drfp_valid.float().mean()`.

### M2 — Modality dropout not inverted-scaled
- Reporter: internal Claude.
- Location: `src/model/v4_ultimate.py:459-467`.
- `keep = (rand >= p).float()`, no `/ (1-p)`. Standard dropout scales by `1/(1-p)` so `E[output]` matches between train and eval. Here: expected embed magnitude at train is `(1-p) * raw`; at eval (always keep) is `raw`. 15% systematic shift.
- Impact: Gates and heads see one distribution at train, another at eval. Effect is small at p=0.15 but it's a latent bias.
- Fix: `return (torch.rand(B, 1, device=self.device) >= p).float() / (1.0 - p)`. Or deliberately keep non-scaled and rename to "modality cutoff regularization" (not dropout) in the paper.

### M3 — Pocket dropout leaves pocket GVP gradient-exposed via int3d
- Reporter: internal Claude.
- Location: `src/model/v4_ultimate.py:466, 487-494`.
- `pocket_pool = pocket_pool * keep_pocket` zeroes the pool used by `g_struct` / `head_struct`, and later `y_int3d = y_int3d * keep_pocket` zeroes the int3d prediction. But the **unmodified** `pocket_tokens` still flows through `self.int3d(...)` — the cross-attn forward runs, returns `p_pool_int` / `a_pool_int`, which goes through `head_int3d` to produce `y_int3d`. That means the pocket GVP still receives gradient through the int3d path for "dropped" samples (via chain rule through `y_int3d = head(p_pool_int)`, then zero-mult by `keep_pocket`).
- Impact: Modality dropout fails to fully decouple the pocket encoder from the int3d path. The branch "contributes zero" to the prediction, but the encoder is still being trained on it. Weakens the robustness claim.
- Fix: either `pocket_tokens = pocket_tokens * keep_pocket.unsqueeze(-1)` before the int3d call, or skip the int3d forward whenever the entire batch has `keep_pocket=0` (rare, probably not worth it). Former is the clean fix.

### M4 — `_log_diag` logs post-dropout `y_*`
- Reporter: internal Claude.
- Location: `src/model/v4_ultimate.py:504-507` (dropout) then `559-566` (log call).
- The diagnostic call happens after `y_rxn *= keep_rxn`, `y_struct *= keep_pocket`, etc. So `diag/y_rxn_abs` is ~`(1-p) * true_magnitude`. When comparing which branch dominates, the undropped branches (`y_seq`, `y_sub`) look systematically larger by ~15%.
- Impact: Makes the most useful diagnostic (which branch carries the signal?) harder to interpret.
- Fix: move `_log_diag` above the output dropout block, or log both pre- and post-dropout.

## LOW — fragility / polish

### L1 — `hasattr(G, "X")` does not guard against `G.X is None`
- Reporter: DeepSeek.
- Location: `src/model/v4_ultimate.py:603-608`, same pattern at `521-522`.
- `hasattr` returns True for None attributes. `G.ANNOT_has_any.float()` crashes if the attribute is set but None. Same for `ANNOT_has_cof`, `MOL_graph_xyz_valid`, `RXN_drfp`.
- Impact: Probably not triggered by the current dataloader (values are set or key is absent), but fragile.
- Fix: `if getattr(G, "ANNOT_has_any", None) is not None:`.

### L2 — Ablation configs missing `modality_dropout` block
- Reporter: internal Claude (verified by grep).
- Location: `config/ablation/v4_ultimate_A*_*.yml`, all 9 files.
- The ablation configs were generated from `config/v4_ultimate.yml` *before* the `modality_dropout` block was added. `grep -l modality_dropout config/ablation/*.yml` returns empty. A8 therefore ≠ current `v4_ultimate.yml`.
- Impact: The ablation matrix measures a different model than the "full" config.
- Fix: rerun `python scripts/26_generate_ablation_configs.py` after adding `use_substrate` (so A0 is also fixed).

### L3 — Redundant `to_dense_batch` in `_encode_substrate`
- Reporter: Kimi, internal Claude.
- Location: `src/model/v4_ultimate.py:315-325`.
- When both the NAC path and `use_rxn_center=True` are active, `to_dense_batch` is called twice on the same rxn-center flag (once as float for NAC, once as long for Embedding).
- Impact: minor wasted compute, no correctness issue.
- Fix: compute once, cast: `center_dense = rxn_center_dense.long()`.

### L4 — `nac_gain` has no upper bound
- Reporter: internal Claude.
- Location: `src/model/int3d_cross_attn.py:63`.
- Init at 0, only weight-decay pulls it down. If NAC signal is strongly correlated with the target, `nac_gain` can grow arbitrarily, and softmax over `nac_outer * nac_gain` will collapse attention onto the catalytic↔rxn-center pairs, potentially destabilizing.
- Impact: unlikely at p=15% `wd=0.01`, but no safety net.
- Fix: `nac_gain.tanh() * max_gain` (e.g. `max_gain=3.0`), or explicit clamp in the bias computation.

### L5 — No shape assertion on `node_s` width
- Reporter: internal Claude.
- Location: `src/model/v4_ultimate.py:374-378`.
- Code indexes `node_s[:, 21]`, `node_s[:, 22]`, and `node_s[:, (1,3,4,6,8,11,15,18)]` without asserting width. If pocket featurization ever changes (e.g., drops the active-site / binding-site flags), this silently computes on wrong data.
- Fix: `assert node_s.shape[-1] >= 23, f"node_s width {node_s.shape[-1]} < 23 — column layout changed?"`.

## What's NOT broken (cross-checked by Kimi)
- `p_nac` / `a_nac` kwargs in `int3d_cross_attn.py` match the v4_ultimate call site.
- The new tuple arity of `_encode_substrate` (4-tuple) / `_encode_pocket` (5-tuple) does not break any downstream caller — v4_pocket, v4_innovate, v4_minimal each define their own encoders.
- NAC gradient reaches `nac_gain` (smoke-tested during the commit).
- `self.training` correctly disables modality dropout at val/test.
- `cat_score` / `pocket_cat` dtype and device are correct.
- Diagnostics use `.detach()` throughout; no loss leakage.

## Fix priority
Must fix before ablation matrix has any meaning:
- **H1** (use_substrate flag + regenerate configs) — otherwise A0 vs A1 delta is 0 by construction.

Should fix before training run has useful diagnostics:
- **M1** (real per-sample cov_*)
- **M4** (log pre-dropout y_*)

Should fix for the paper's dropout story to hold up:
- **M2** (inverted scaling or rename)
- **M3** (zero pocket_tokens under dropout)

Can ship without fixing, but track as debt:
- L1–L5.
