# `frequency_online` — proposal and measurement

**Status: adopted as a separately named experimental baseline after local
review.** `frequency_fitted` and every existing result remain untouched. See
`reports/frequency-online/findings.md` for the analysis-session corrections and
evidence limits.

**Stamp** `20260809-200701`
**Base commit** `15285704e73af680c0125ea4bfeb0b54a14f278e`
**Engine** vLLM 0.26.0, unmodified, `--gpu-memory-utilization 0.92`, pinned
model revisions matching the dual-model run.

**Raw archive** `/home/taghan/frequency-online-20260809-200701.tar.gz`
9.6 MB, 87 entries, `tar -tzf` verified,
sha256 `e6a774b158069a3742adb66584d35dc7053c443da8d5ef500f077f51e308caef`.

The executor states that `PREDECLARATION.md` was written **before** any workload
was built or replay executed. It was pushed in the same final commit as the
implementation and results, so Git does not provide an independent pre-run
timestamp for that statement.

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
every single menu**. There are 199 other distinct tools: 198 appear once and
one appears twice. Every menu still contains exactly one member outside the
universal 63-tool set. Measured reuse tracks the mean position of that
non-universal tool almost exactly:

| ordering | mean position of unique tool | measured reuse |
| --- | --- | --- |
| original | 1.0 | 1.19% |
| alphabetical | 24.1 | 38.13% |
| frequency_fitted | 25.0 | 39.69% |
| tooltrie_v0 | 57.3 | 87.19% |
| contextpilot-online | 63.7 | 96.16% |
| **frequency_online** | **63.7** (last in 199/200) | **96.27%** |

A disjoint training corpus structurally cannot know which 63 tools *this*
stream shares — that is an artifact of the eval sampling. After observing the
first request, the online counter already gives all 63 universal tools a higher
count than the next unseen non-universal tool.

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

**Correction:** `frequency_online` beats `tooltrie_v0` in **10 of 12** cells. It
loses both BM25 k=4 cells: 17.01% versus 17.48%. It is close to ContextPilot on
padded menus, but no equivalence margin was declared, so “matches” is not an
equivalence claim.

### Same-session control

Because this run's caches differ slightly from the dual-model run's, cp-online
was replayed on **these** servers for the two decisive BM25 depths:

| model | workload | frequency_online | cp-online (control) | delta |
| --- | --- | --- | --- | --- |
| 4B | k=16 | 8.14% | 9.10% | −0.97 pp |
| 4B | k=128 | **2.33%** | 1.36% | **+0.98 pp** |
| 0.6B | k=16 | 8.16% | 9.93% | −1.78 pp |
| 0.6B | k=128 | **2.41%** | 1.99% | **+0.42 pp** |

The control reproduces the dual-model figures to within 0.01 percentage point
(9.10 vs 9.10, 1.36 vs 1.35, 9.93 vs 9.93, 1.99 vs 1.99). This supports the two
controlled BM25 depths; it does not erase the capacity mismatch for the other
four cross-run rows.

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

The executor reported 124 passing tests at its branch. The merged suite retains
`test_online_frequency_plan_cannot_see_the_current_request` and adds compact
and structural audits.

## What this means

On the `bfcl-padded64` positive-control workload, the high-reuse result is
reproduced to within 0.1 pp of ContextPilot by counting how
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
- The predeclaration specified one server at a time, but the two model servers
  actually ran concurrently on GPU 2 and GPU 3. Latency figures are therefore
  excluded. Aggregate reuse is deterministic with zero trial spread, but this
  remains a recorded protocol deviation.
- The `frequency_online` capacities (101,120 at 4B; 190,896 at 0.6B) differ from
  the accepted dual-model capacities (96,832; 188,912). Same-session ContextPilot
  controls cover only BM25 k=16 and k=128.
- Git contains aggregate reuse values and builder summaries, not the 36 raw
  replay files. The executor's clean-counter claim depends on the server-only
  archive until it is copied and re-audited.
