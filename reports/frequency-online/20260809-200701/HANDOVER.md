# `frequency_online` — proposal and measurement

**Status: PROPOSAL, not merged.** `frequency_fitted` and every existing result
are untouched. The analysis session owns `src/tatm/` and decides whether to
adopt this policy.

**Stamp** `20260809-200701`
**Base commit** `15285704e73af680c0125ea4bfeb0b54a14f278e`
**Engine** vLLM 0.26.0, unmodified, `--gpu-memory-utilization 0.92`, pinned
model revisions matching the dual-model run.

**Raw archive** `/home/taghan/frequency-online-20260809-200701.tar.gz`
9.6 MB, 87 entries, `tar -tzf` verified,
sha256 `e6a774b158069a3742adb66584d35dc7053c443da8d5ef500f077f51e308caef`.

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
every single menu** and 199 appear in exactly one each. Measured reuse tracks
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

**`frequency_online` beats `tooltrie_v0` in all 12 cells** and matches
ContextPilot on padded menus.

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

- **Systems only. No quality replay was run for `frequency_online`.** Ordering
  moved `full_accuracy` by up to 11.6 pp at 0.6B, so nothing should be assumed
  about its quality effect in either direction.
- No equivalence margin declared; no equivalence claim is made.
- Single menu seed (42), inherited from the shared workloads.
- The two servers ran concurrently on GPU 2 and GPU 3, so latency figures from
  this run are contention-affected and are not reported. Reuse is deterministic
  and unaffected — demonstrated by zero spread across trials and identical
  values at two very different cache sizes.
