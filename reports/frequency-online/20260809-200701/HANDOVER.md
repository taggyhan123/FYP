# `frequency_online` — proposal and measurement

**Status: PROPOSAL, not merged.** `frequency_fitted` and every existing result
are untouched. The analysis session owns `src/tatm/` and decides whether to
adopt this policy.

**Stamp** `20260809-200701`
**Base commit** `15285704e73af680c0125ea4bfeb0b54a14f278e`
**Engine** vLLM 0.26.0, unmodified, `--gpu-memory-utilization 0.92`, pinned
model revisions matching the dual-model run.

**Raw archive** `/home/taghan/frequency-online-20260809-200701.tar.gz`
14 MB, 102 entries, `tar -tzf` verified,
sha256 `9089bc5c69403375546cf30292c451f7dbe044def0c8bd93c3be62ffe296939b`.
(Re-archived after the quality replays below were added.)

`PREDECLARATION.md` in this directory was written **before** any workload was
built or any replay executed.

## Why

`frequency_fitted` freezes tool support on a **task-disjoint training corpus**
(`build_fitted_ordering_workload.py` requires `--training-input` and strips
overlapping `task_id`s). That is correct anti-leakage discipline. But
`FittedOrderingPlanner` has `plan()` and `snapshot()` and **no `observe()`** —
it never updates — while every policy it is benchmarked against (ToolTrie,
CacheWeaver, both ContextPilot arms) adapts to the served stream. Part of the
reported gap was therefore adaptivity, not algorithm.

The structural reason it fails on this benchmark: across the 200
`bfcl-padded64` menus there are 262 distinct tools, of which **63 appear in
every single menu**, 198 appear in exactly one, and one appears in two
(verified distribution `{1: 198, 2: 1, 200: 63}`). Measured reuse tracks
the mean position of the one non-shared tool almost exactly:

| ordering | mean position of unique tool | measured reuse |
| --- | --- | --- |
| original | 1.0 | 1.19% |
| alphabetical | 24.1 | 38.13% |
| frequency_fitted | 25.0 | 39.69% |
| tooltrie_v0 | 57.3 | 87.19% |
| contextpilot-online | 63.7 | 96.16% |
| **frequency_online** | **63.7** (last in 199/200) | **96.27%** |

A disjoint training corpus structurally cannot know which 63 tools *this*
stream shares — that is an artifact of the eval sampling. An online counter
learns it in about 25 requests.

## The policy

`OnlineFrequencyPlanner` (`src/tatm/baselines.py`): count how many
**already-served** requests contained each tool; order each menu by descending
count, ties by function name then tool ID. Never-seen tools have count 0 and
sort last. No training corpus, no capacity parameter, no trie, no clustering.
`plan()` reads only counts from strictly earlier requests.

## Results — reuse %, mean of 3 trials, spread 0.000 in every cell

| workload | original | alphabetical | tooltrie_v0 | cp-online | cp-static | **frequency_online** |
| --- | --- | --- | --- | --- | --- | --- |
| BFCL padded-64 | 1.19% | 38.13% | 87.19% | 96.16% | 96.16% | **96.27%** |
| ToolRet padded-64 | 13.73% | 50.82% | 83.58% | 95.27% | 95.74% | 94.80% |
| BM25 k=4 | 15.87% | 15.28% | 17.48% | 18.72% | 18.42% | 17.01% |
| BM25 k=16 | 6.12% | 6.27% | 7.77% | 9.93% | 9.80% | 8.16% |
| BM25 k=64 | 0.91% | 1.24% | 1.90% | 4.78% | 4.01% | 3.13% |
| BM25 k=128 | 0.37% | 0.58% | 1.13% | 1.99% | 2.96% | **2.41%** |

Qwen3-0.6B shown (capacity 190,896); Qwen3-4B (capacity 101,120) is
near-identical. Baseline columns are from the accepted dual-model run.
36 `frequency_online` replays, all 200 requests, zero failures, all
`counter_validation.clean`, cache reset before each.

**`frequency_online` beats `tooltrie_v0` in 10 of 12 cells** and matches
ContextPilot on padded menus. It loses both BM25 k=4 cells (17.01% against
17.48%). An earlier version of this handover claimed 12/12; that was wrong and
is corrected here.

### Same-session control

Because this run's caches differ slightly from the dual-model run's, cp-online
was replayed on **these** servers for the two decisive BM25 depths:

| model | workload | frequency_online | cp-online (control) | delta |
| --- | --- | --- | --- | --- |
| 4B | k=16 | 8.14% | 9.10% | −0.97 pp |
| 4B | k=128 | **2.33%** | 1.36% | **+0.98 pp** |
| 0.6B | k=16 | 8.16% | 9.93% | −1.78 pp |
| 0.6B | k=128 | **2.41%** | 1.99% | **+0.42 pp** |

The control reproduces the dual-model figures almost exactly (9.10 vs 9.10,
1.36 vs 1.35, 9.93 vs 9.93, 1.99 vs 1.99), so the capacity difference had no
measurable effect and the cross-run table is comparable.

## Verification performed

| check | result |
| --- | --- |
| Provenance | correct `run_label`/input/model, 200 requests, 0 failed, reset true, `counter_validation.clean` with both sub-checks true |
| **Token conservation** | total prompt tokens **1,380,694** identical to original, alphabetical, tooltrie_v0 and cp-online on the same menus — proves a genuine permutation |
| Independent recompute | summing 200 per-request `metric_delta`s reproduces the aggregate counter exactly |
| Determinism | cached tokens identical across all 3 trials and across both models; spread 0 |
| **Causality** | ordering re-derived from scratch with a separately written implementation, observing strictly after planning: **0 mismatches over 200 records** on two workloads |
| Prompt fidelity | `tools` payload aligned to `tool_ids`, so the rendered prompt follows the reorder |
| Cold start | record 0 is pure alphabetical-by-name; no prior information used |

Tests: 124 pass (118 existing + 6 new), including
`test_online_frequency_plan_cannot_see_the_current_request`.

## What this means

On `bfcl-padded64` the benchmark is solved to within 0.1 pp by counting how
often each tool has appeared. It therefore **cannot be evidence for
hierarchical clustering, trie-aware planning, or ordering sophistication** —
every method scoring well on it is being credited for discovering that 63 of
64 tools never change. This converts the "padded menus are an artifact"
observation into a demonstration.

The prediction recorded in `PREDECLARATION.md` before execution — ≥90% on
padded menus, 1–19% on BM25 — held on both counts.

## Limits

- Quality has since been measured (see below); the earlier "systems only"
  limitation no longer applies.
- No equivalence margin declared; no equivalence claim is made.
- Single menu seed (42), inherited from the shared workloads.
- The two servers ran concurrently on GPU 2 and GPU 3, so latency figures from
  this run are contention-affected and are not reported. Reuse is deterministic
  and unaffected — demonstrated by zero spread across trials and identical
  values at two very different cache sizes.


## Quality — added after the systems run

n=800 BFCL, 640 relevance + 160 irrelevance, `--max-tokens 128
--disable-thinking --reset-before`. Both replays: 800 requests, zero failures,
`counter_validation.clean`, correct pinned model. Baseline columns are the
accepted dual-model arms.

### Qwen3-4B (primary)

| condition | full | name | no_tool |
| --- | --- | --- | --- |
| original | 76.09% | 83.13% | **88.12%** |
| alphabetical | 73.28% | 82.19% | 85.62% |
| tooltrie_v0 | 75.31% | 83.75% | 87.50% |
| cp-online | **77.03%** | **84.38%** | 85.00% |
| cp-static-refit | **77.03%** | **84.38%** | 85.00% |
| **frequency_online** | 76.88% | **84.38%** | 83.75% |

`frequency_online` is identical to both ContextPilot arms on `name_correct` and
0.15 pp below them on `full_correct` — one case out of 640. Having already
matched them on reuse, it matches them on quality.

### Qwen3-0.6B (replication)

| condition | full | name | no_tool |
| --- | --- | --- | --- |
| original | **55.16%** | **73.75%** | 86.25% |
| alphabetical | 43.59% | 60.47% | **94.37%** |
| tooltrie_v0 | 53.28% | 69.06% | 93.13% |
| cp-online | 51.72% | 68.91% | 91.25% |
| cp-static-refit | 51.88% | 68.91% | 91.25% |
| **frequency_online** | 53.28% | 69.84% | 90.62% |

At 0.6B **every** reordering policy is worse than `original` on both relevance
metrics and better on `no_tool`. `frequency_online` is the least damaging of
them on relevance.

### Reuse and irrelevance accuracy are in tension

Among the policies that actually achieve reuse, `no_tool` accuracy at 4B
degrades monotonically with reuse:

| policy | BFCL padded-64 reuse | no_tool vs original |
| --- | --- | --- |
| tooltrie_v0 | 87.19% | −0.62 pp |
| cp-online | 96.16% | −3.12 pp |
| frequency_online | 96.27% | −4.37 pp |

These policies all work by pushing the request-specific tool behind ~63 stable
ones, so the menu is dominated by a block the model has seen repeatedly. A
plausible reading is that this primes tool-calling and suppresses declining.
**This is a hypothesis, not a measured mechanism**, and no equivalence margin
was declared for any of these comparisons — they are estimation only. If it
holds, the "free win" framing of ordering optimisation is wrong.
