# Predeclaration — `frequency_online` ordering policy

Written **before** any workload was built or any replay executed. Recorded so
the prediction below cannot be adjusted after seeing results.

**Stamp** `20260809-200701`
**Base commit** `15285704e73af680c0125ea4bfeb0b54a14f278e`

## Motivation

The existing `frequency_fitted` baseline estimates tool support on a
**task-disjoint held-out corpus** (`build_fitted_ordering_workload.py` requires
`--training-input` and strips overlapping `task_id`s). That is correct
anti-leakage discipline and implements the brief's rule *never interpret
benchmark frequency as production popularity*.

However it makes the comparison against `tooltrie_v0`, `cacheweaver` and both
ContextPilot arms **unfair in one specific way**: `FittedOrderingPlanner` has
`plan()` and `snapshot()` but **no `observe()`** — its statistics never update
after construction, while every policy it is benchmarked against adapts to the
served stream. Part of the observed gap is therefore adaptivity, not algorithm.

Structural finding that motivates the specific prediction: across the 200
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

A disjoint training corpus structurally cannot identify which 63 tools *this*
eval stream happens to share, because that is an artifact of the eval sampling.
An online counter can, within a few requests.

## Policy under test

`frequency_online` — causal online presence-frequency ordering.

Maintain a count of how many **already-served** requests contained each tool.
Order each new menu by descending observed count, ties broken by function name
then canonical tool ID. Never-seen tools have count 0 and therefore sort last.
`plan()` uses only counts from strictly earlier requests; `observe()` is called
only after the request has been served. No training corpus is required.

This is *not* an edit to `frequency_fitted`, which is left untouched.

## Predictions (recorded before execution)

1. On `bfcl-padded64` and `toolret-padded64`, `frequency_online` reaches
   **≥ 90% reuse**, placing the unique tool at mean position **≥ 60 of 64** —
   i.e. matching or beating `tooltrie_v0` (87.19% / 82.18%) and approaching
   both ContextPilot arms (95–96%).
2. On the four BM25-retrieved workloads it stays in the **1–19% band** with the
   other policies, because those menus have no always-present set for an online
   counter to discover.
3. Prediction 1 holding while prediction 2 also holds is the intended result:
   it would show that on padded menus a ~10-line online counter matches
   hierarchical clustering, so the 95–96% figure reflects menu construction
   rather than algorithmic sophistication.

**Falsification:** if `frequency_online` lands near `frequency_fitted` (~40%) on
padded menus, the position analysis above is wrong and adaptivity is not the
explanation. That outcome is equally publishable and must be reported.

No equivalence margin is declared. No quality claim is made — this run measures
prefix-cache reuse only.

## Method

Same six systems workloads, three reset trials per cell, one idle physical GPU,
one server at a time, one sequential client, unmodified vLLM 0.26.0 APC. Server
configuration identical to the accepted dual-model arms so the new column is
directly comparable: pinned model revisions and
`--gpu-memory-utilization 0.92`. Every replay resets the prefix cache first and
must report `counter_validation.clean`. `--allow-counter-mismatch` will not be
used. Results go to a new directory; no existing result directory is modified.
