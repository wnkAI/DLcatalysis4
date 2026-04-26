# /five brief — project-level code style + redundancy audit

Repo: `E:/AImodel/DLcatalysis4.0` (~10k lines Python, 41 files).

File-by-file line counts in `file_list.txt`.

This is not a bug-review pass (we just finished 5 rounds of that). The
goal here is **code hygiene**: find duplication, dead code, over-engineered
abstractions, verbose non-idiomatic Python, and comments that have rotted.

## What to look for

1. **Cross-file duplication.** Same function reimplemented in multiple
   places (e.g. attention pooling, masked mean, to_dense_batch wrappers).
   Flag pairs/triples that could be consolidated into a shared util.

2. **Dead code.** Imports never used, functions never called, parameters
   wired but never exercised, feature flags left over from v3.0 that no
   longer matter.

3. **Over-engineered abstractions.** A 3-line helper that's called from
   one site, a class that only has one instance, config options that
   silently fall through to a single hardcoded default.

4. **Verbose / non-idiomatic Python.** Places where:
   - A dict comprehension would replace a loop.
   - `if hasattr(x, "y") else default` should be `getattr(x, "y", default)`.
     (Note: forward-path hasattr-then-.to() was already swept to
     `getattr(..., None) is not None` — don't re-flag that.)
   - String formatting inconsistent (% / .format() / f-string mix).
   - Defensive type casts that the type system already guarantees.

5. **Stale comments / docstrings.** Comments that reference removed
   functionality, TODOs dated pre-3.0, docstrings describing old
   signatures.

6. **Inconsistent naming.** Same concept named differently across files
   (`atom_mask` vs `a_mask` vs `mask_a`), snake_case vs camelCase mix
   in a single module.

7. **Redundant "safety" logic.** Defensive None checks for values the
   caller cannot actually send None. `.clamp(min=1e-6)` where division
   is on a value that can't be 0.

## What NOT to flag

- Correctness bugs — different review loop.
- Docstring formatting preferences — only flag if a docstring is
  actually misleading or describes a different function than the one
  it sits on.
- Minor stylistic preference (single vs double quotes etc.). We use
  whatever the file already uses.
- Comments explaining **why** a non-obvious thing is the way it is
  (those are load-bearing).

## Output format

Grouped by theme, e.g.:

```
## Duplication
- `src/util/tools.py::_attn_pool` and `src/model/v4_ultimate.py:_attn_pool_seq`
  are both weighted softmax over a masked sequence. Consolidate to util.

## Dead code
- `src/model/v4_innovate.py` imports `from module import MLP2` — MLP2
  doesn't exist in module.py; import is stale.

## Verbose
- ...

## Stale comments
- ...
```

Each item: file + rough line(s), 1-sentence description, why it matters,
**suggested replacement** (actual code or at least the shape).

End with a priority-ordered list of "top 5 refactors worth doing"
from your perspective.

Be honest about trade-offs — if a duplication has a real reason
(e.g. v4_innovate wants a different _encode_substrate than v4_ultimate),
say so rather than demanding consolidation.
