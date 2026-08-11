# Weighted ToolTrie and online pair/triple — predeclaration

**Written before any policy was implemented, any workload built, or any replay
executed.**

Closes the two clauses of the initial brief that remain genuinely unanswered:

- **§2 Q3** — *"a **weighted** trie or prefix memory under a limited cache
  budget"*. The budget clause was measured on 2026-08-11
  (`reports/tooltrie-pressure/20260811-001032/`). The weighting clause was never
  built: `ToolTrie.plan()` tie-breaks on reachable cached cost then *frozen
  training* support, and `visit_count` is incremented on every observation and
  read nowhere, in selection or in eviction.
- **§7 Q5** — *"How much additional benefit comes from pair/triple workflow
  structure?"* Never posed: `conditional_pair` and `conditional_pair_triple`
  emit byte-identical orderings to `frequency_fitted` on **both** workloads
  (`reports/tooltrie-phase2/fitted-policy-equivalence.json`, 0 differing records,
  shared sequence hash).

## Structural result established first, on CPU

`scripts/audit_pair_triple_information.py` shows why Q5 was never posed, and it
is a property of the workloads rather than of any planner:

| workload | tools in every request | pair support == min(presence) | violations |
| --- | ---: | ---: | ---: |
| bfcl-padded64 | 63 | **100.00%** | **0** |
| toolret-padded64 | 60 | 99.39% | 46 |
| toolret-bm25-k16 | 0 | 85.16% | 2,651 |
| toolret-bm25-k64 | 0 | 76.10% | 50,491 |
| toolret-bm25-k128 | 0 | 67.34% | 208,615 |

On `bfcl-padded64` pair support is a **deterministic function of presence
counts** — zero violations over 14,490 pairs. An ordering keyed on pair
co-occurrence therefore *cannot* differ from one keyed on frequency, whatever
planner consumes it. **Q5 is unanswerable on padded menus by construction**, and
the five-way tie reported in earlier work is a theorem rather than a
coincidence.

Structure exists only on retrieved menus, so that is where Q5 must be measured.

## The two policies

Both are **proposals alongside** the existing policies, in the pattern of
`frequency_online` beside `frequency_fitted`. Neither modifies ToolTrie-v0's
behaviour; v0 remains the Task E deliverable that every published figure
describes.

**`tooltrie_v1`** — identical to v0 except that it reads `visit_count`:

1. selection tie-break becomes `-child.visit_count` in place of the frozen
   training support term;
2. eviction removes the **least-visited** leaf, breaking ties by least-recently
   used, in place of pure LRU.

Enabling this requires one behaviour-preserving refactor of v0: the selection
key, currently an inline lambda, is extracted to `_selection_key`. Acceptance
requires the existing 129 tests to pass unchanged, and v0's emitted orderings to
be unaffected.

**`pair_triple_online`** — the causal counterpart of `conditional_pair_triple`.
Pair and triple support are accumulated from **already-served** menus; each menu
is ordered by a greedy chain keyed on `(-triple, -pair, -presence, name, id)`.
`plan()` reads only counts from strictly earlier requests; `observe()` runs
after.

## Declared runs

**Run A — §2 Q3, limited cache budget.** Qwen3-0.6B,
`--num-gpu-blocks-override 480`, live capacity asserted at 7,680 tokens,
`--require-peak-kv-usage 0.90`, empirical regime only. The four-regime result of
2026-08-11 showed no row moving more than a few points across regimes, so one
regime is sufficient and is declared here rather than chosen afterwards.

Conditions: `tooltrie_v1`, `frequency_online`. Compared against the accepted
`tooltrie` (87.18%) and the six static orderings already in the matrix.

**Run B — §7 Q5, where the structure exists.** Qwen3-0.6B at native capacity,
`toolret-bm25-k128` and `toolret-bm25-k16` base workloads from the accepted
dual-model run. Conditions: `pair_triple_online`, and `frequency_online` rebuilt
from the same base as an in-session control.

## Predictions, recorded before measurement

1. **`tooltrie_v1` will not beat `tooltrie` by more than 2 points under
   pressure.** v0 already reaches 87.18% where the shared core occupies almost
   the whole 7,680-token cache, so little headroom remains.
2. **`frequency_online` will beat `tooltrie` under pressure**, because it
   concentrates the shared core harder — position 63.7 of 64 against 57.3.
   If it does, finding 8's claim must be narrowed from "the trie wins under
   scarcity" to "adaptive policies win under scarcity".
3. **`pair_triple_online` will differ from `frequency_online` on retrieved menus
   and not on padded ones.** The CPU result above makes the padded half a
   theorem; the retrieved half is the genuine test.
4. **Any Q5 effect will be small in absolute terms**, because retrieved-menu
   reuse is 0.3–19%. A positive result would mean pair/triple structure helps
   precisely where there is almost nothing to gain.

## What these runs may not claim

- No equivalence margin is declared; all comparisons are estimation only.
- One run per cell, one menu seed, one replay seed, single-tenant server.
- Run A's values must not be compared numerically with any native-capacity run.
- `tooltrie_v1` is a proposal. Adopting it in place of v0 is the analysis
  session's decision, and would require rerunning the figures that describe v0.

## Stop conditions

Halt and report rather than improvise if: the selected GPU becomes occupied,
Run A's live capacity is not exactly 7,680, the peak-KV gate fails, the 129
existing tests fail, v0's emitted orderings change, or a summarizer rejects a
matrix. Preserve every failed output.
