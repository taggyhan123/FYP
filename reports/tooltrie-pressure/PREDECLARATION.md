# ToolTrie-v0 under a limited cache budget — predeclaration

**Written before any code change, workload build, or measurement.**

## Question

Brief §2 Q3 asks whether frequently occurring tool sequences can be represented
as a weighted trie or prefix memory **under a limited cache budget**. The
capacity-constrained clause has never been tested for the project's own
prototype: the controlled-pressure matrix at `20260807-005414` contains six
fixed orderings and no ToolTrie condition.

This run adds exactly that one condition. It does not rerun, overwrite or
reinterpret the accepted 24 regime-runs.

## Why a code change is needed, and its exact scope

`scripts/locality_replay.py` accepts six orderings and applies them through
`order_tool_ids(...)`, a **stateless** function called per task. Its own help
text declares the ordering "held constant across replay conditions". ToolTrie is
**causal and adaptive** — `plan()` reads paths observed from strictly earlier
requests and `observe()` updates them — so it cannot be expressed as a fixed
permutation and cannot be passed to the existing code path.

The change is opt-in and additive:

- add `tooltrie` to the `--ordering` choices;
- inside `build_sequence`, when and only when `--ordering tooltrie`, construct a
  fresh `ToolTrie` and call `plan()` then `observe()` per task;
- every other ordering keeps a byte-identical code path.

**Acceptance requires that the six existing orderings are untouched**: the
diff must not alter any statement reachable when `--ordering != "tooltrie"`.
The accepted matrix must remain reproducible from the modified script.

## Declared configuration

| parameter | value | source |
| --- | --- | --- |
| model | `Qwen/Qwen3-0.6B` | matches the accepted pressure run |
| `num_gpu_blocks_override` | 480 | predeclared in the original protocol |
| live capacity | 7,680 tokens | asserted before replaying |
| `max_model_len` | 7,168 | as in the original protocol |
| partition / limit / menu | bfcl / 200 / 64 | unchanged |
| `--random-seed` / `--replay-seed` | 42 / 2026 | unchanged |
| `--support-mode` | disjoint | unchanged |
| regimes | empirical, uniform, skewed, session_bursty | unchanged |
| `--require-peak-kv-usage` | 0.90 | unchanged, not to be lowered |
| ToolTrie `recency_window` | 128 | the value used in every prior ToolTrie run |
| ToolTrie `capacity_tokens` | 7,680 | the live capacity, so its budget binds |
| ToolTrie `fallback` | alphabetical | the value used in every prior ToolTrie run |
| runs per cell | 1 | matches the existing matrix |

A fresh `ToolTrie` is built per regime, because each regime resets the prefix
cache and a warm planner against a cold cache would misrepresent both.

## Predictions, recorded before measurement

1. **ToolTrie's own eviction path will fire for the first time.** At the
   188,912-token capacity used in every previous ToolTrie run, a ~6,900-token
   menu never approaches the budget, so `_over_budget` / `_evict_one_leaf` have
   effectively never executed. At 7,680 they must. A crash or a zero eviction
   count is a reportable defect, not a result to discard.
2. **ToolTrie will not reach its uncapped 87.19%.** Every ordering loses heavily
   under this capacity; alphabetical falls from 38.13% to 29.21%.
3. **The decisive comparison is against `random`, seed 42, which leads every
   regime at 32.16 / 31.27 / 32.67 / 30.19%.** ToolTrie beating random in all
   four regimes would be the first evidence that trie structure earns its keep;
   losing in all four extends the report's most uncomfortable finding to the
   project's own prototype.

## What this run may not claim

- No equivalence or superiority test is declared, so results are estimation
  only.
- One run per cell at one menu seed and one replay seed.
- Latency must not be compared with any 188,912-token-capacity run, and this is
  a controlled stress test rather than natural production pressure.
- A ToolTrie result here says nothing about the **weighted** trie in §2 Q3.
  `visit_count` remains unread; this run tests the prefix-memory clause under a
  limited budget, not the weighting clause.

## Stop conditions

Halt and report rather than improvise if: the selected GPU becomes occupied,
live capacity is not exactly 7,680, the peak-KV gate fails, the six existing
orderings change behaviour, pytest fails, or the summarizer rejects the matrix.
Preserve every failed output.
