# ToolTrie-v0 under a limited cache budget — GPU executor handover

**Stamp** `20260811-001032`
**Executed FYP commit** `6f0ba2f3d97d8f9bd01dbc373596eb208f35bed9`
**Engine** vLLM 0.26.0, unmodified. Qwen3-0.6B, GPU 2, single server, single
sequential client under `flock`.

Executes `PREDECLARATION.md` in this directory, written and committed **before**
any workload was built or replayed. It adds one condition to the controlled-
pressure design and does not rerun, overwrite or reinterpret the accepted 24
regime-runs at `20260807-005414`.

## Raw archive

```
/home/taghan/tooltrie-pressure-20260811-001032.tar.gz
sha256 f6e282cb35a4752cf83da303b41a77b3ad8545e2ff15c3f15779fe109721af93
14 entries, 88 KB
```

## Acceptance

```
Accepted pressure regimes: 4/4
Validation checks: 64/64
```

Live capacity asserted at exactly **7,680 tokens** (block size 16, 480 GPU
blocks) before replaying. Peak KV occupancy **0.90397–0.90814**, above the
predeclared 0.90 in every regime; the threshold was not altered. Zero
preemptions, zero metric scrape errors, cache reset before every regime,
`counter_validation` clean throughout. `total_prompt_tokens` is 1,380,694 per
regime, identical to every other ordering measured on these menus — the
ordering is a genuine permutation.

## Result

Reuse %, 200 requests per regime, one run per cell, matching the existing
matrix. The first four rows are the accepted `20260807-005414` values, quoted
for comparison; only the last row is new.

| ordering | empirical | uniform | skewed | session_bursty |
| --- | ---: | ---: | ---: | ---: |
| original | 1.18% | 0.69% | 1.24% | 0.69% |
| alphabetical | 29.21% | 29.35% | 28.06% | 27.76% |
| random, seed 42 | 32.16% | 31.27% | 32.67% | 30.19% |
| frequency = schema-cost = FP-tree | 9.44% | 8.99% | 9.51% | 9.00% |
| **ToolTrie-v0** | **87.18%** | **88.54%** | **94.73%** | **91.62%** |

ToolTrie-v0 leads random by **+55.02, +57.27, +62.06 and +61.43 points**.

## Predeclared predictions: two held, one failed

1. **Held.** ToolTrie's own metadata-budget eviction fired for the first time —
   506 to 1,494 planner evictions per regime. At the 188,912-token capacity used
   in every previous ToolTrie run that path was effectively dead code. It ran
   without error.
2. **Failed.** The prediction was that ToolTrie would not reach its uncapped
   87.19%. It reached **87.18%** on the empirical regime — the same value to
   within 0.01 points — and **exceeded** it in the other three.
3. **Held, decisively.** ToolTrie beats `random` in all four regimes.

## Why prediction 2 failed

ToolTrie converges on one consistent path, so the ~63-tool shared core lands in
the same prefix every request. Estimated peak resident tokens are 6,942–6,975
against a 7,680-token capacity: that single prefix **fits, once**, and LRU keeps
it hot because every request touches it.

Alphabetical places the one varying tool at mean position 24.1 of 64, so only
its first ~24 schemas are shared and the remainder thrashes — which is why it
falls from 38.13% uncapped to 29.21% here. ToolTrie places that tool at 57.3,
so almost the whole menu is shared prefix and shrinking the cache 25× costs it
nothing.

The ordering that wins under abundance and the ordering that wins under scarcity
are different, and the reason is prefix *concentration*, not reuse magnitude.

## What this does and does not establish

Closes the "under a limited cache budget" clause of brief §2 Q3 for the
project's own prototype, which had never been tested.

It does **not** test the "weighted trie" clause. `visit_count` remains unread;
`plan()` still tie-breaks on reachable cached cost then frozen training support.

It does **not** show ToolTrie beats ContextPilot or `frequency_online` under
pressure. **Neither was run in this matrix.** Both concentrate the shared core
even harder than ToolTrie does at full capacity (96.16% and 96.27% against
87.19%), so the mechanism above predicts they would do at least as well. That is
a prediction, not a measurement, and the obvious next run.

One run per cell, one menu seed, one replay seed, single-tenant server. No
equivalence margin was declared, so this is estimation only. Latency must not be
compared with any 188,912-token-capacity run.

## Method deviations

Two scripts were changed, both opt-in, both committed before measurement in
`6f0ba2f` (the first) and alongside this handover (the second):

- `scripts/locality_replay.py` — added `tooltrie` to `--ordering`. When the
  value is anything else, `planner is None` and the identical `order_tool_ids`
  call runs, so the accepted matrix stays reproducible. A fresh planner is
  constructed per regime, matching the cache reset that precedes each.
- `scripts/summarize_pressure_replays.py` — added `--expected-ordering`. The
  library function already accepted `expected_orderings`; only the CLI hard-
  coded the six. Default behaviour is unchanged and no per-regime check was
  modified. Without it the summarizer refuses any matrix that is not exactly
  the predeclared six.

`src/tatm/` was not modified. 129 tests pass.
