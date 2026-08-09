# `frequency_online` — reconciled findings

## Bottom line

`frequency_online` is a valid causal baseline and should remain in the
codebase under that exact name. It uses only tool presence counts from strictly
earlier requests; it is not the frozen, task-disjoint `frequency_fitted`
baseline from the initial brief.

Its most important result is diagnostic rather than competitive: a trivial
online counter reaches 96.27% reuse on the BFCL padded-64 positive control,
compared with 96.16% for both ContextPilot adaptations and 87.19% for
ToolTrie-v0. That shows the padded workload mainly rewards discovering its
nearly fixed 63-tool core. It is not evidence that frequency counting is the
best policy on genuinely retrieved menus.

The GPU handover contained two factual errors that are corrected here:

- `frequency_online` beats ToolTrie-v0 in **10 of 12**, not 12 of 12,
  model/workload cells. It loses BM25 k=4 in both models, 17.01% versus 17.48%.
- BFCL padded-64 has 198 tools seen once and one seen twice, not 199 singleton
  tools. There are 199 distinct non-universal tools in total.

## What was independently checked

The local analysis regenerated all six original workloads. Their SHA-256
values match the accepted dual-model copies byte-for-byte. The structural
audit then found:

| BFCL padded-64 property | Value |
| --- | ---: |
| Requests / tools per request | 200 / 64 |
| Distinct tools | 262 |
| Present in all 200 requests | 63 |
| Present once | 198 |
| Present twice | 1 |
| Non-universal tools per request | exactly 1 |
| Non-universal tool last under causal online frequency | 199/200 requests |
| Mean non-universal position | 63.745 / 64 |

The first request is a true cold start and places its non-universal tool at
position 13 by the declared alphabetical tie-break. Once that request has been
observed, the 63 universal tools have positive counts and the next unseen tool
has count zero, so it moves to the tail immediately. All four BM25-retrieved
workloads have zero tools present in every request.

The implementation also passes direct causality and permutation tests:
`plan()` reads the current counter without changing it, the workload builder
records the plan, and only then calls `observe()`.

Sources: `structure-audit.json`, `local-audit.json`,
`scripts/audit_frequency_online_structure.py`, and
`scripts/audit_frequency_online_compact.py`.

## Reuse results

Percent cached prompt tokens, three reset trials per cell. The ContextPilot
column is the better of its persistent and static-refit ordering-only arms in
the accepted dual-model matrix.

### Qwen3-4B primary

| Workload | `frequency_online` | ToolTrie-v0 | Best ContextPilot arm |
| --- | ---: | ---: | ---: |
| BFCL padded-64 | **96.27%** | 87.19% | 96.16% |
| ToolRet padded-64 | 94.80% | 82.18% | **95.74%** |
| BM25 k=4 | 17.01% | 17.48% | **18.72%** |
| BM25 k=16 | 8.14% | 6.73% | **9.36%** |
| BM25 k=64 | 2.96% | 1.63% | **3.48%** |
| BM25 k=128 | **2.33%** | 0.89% | 2.23% |

### Qwen3-0.6B replication

| Workload | `frequency_online` | ToolTrie-v0 | Best ContextPilot arm |
| --- | ---: | ---: | ---: |
| BFCL padded-64 | **96.27%** | 87.19% | 96.16% |
| ToolRet padded-64 | 94.80% | 83.58% | **95.74%** |
| BM25 k=4 | 17.01% | 17.48% | **18.72%** |
| BM25 k=16 | 8.16% | 7.77% | **9.93%** |
| BM25 k=64 | 3.13% | 1.90% | **4.78%** |
| BM25 k=128 | 2.41% | 1.13% | **2.96%** |

These are systems results, not a universal ranking. `frequency_online` is
stronger than ToolTrie-v0 in 10 cells and weaker in two. ContextPilot is higher
in nine cells, `frequency_online` is numerically higher in three, and no
equivalence or superiority test was declared.

## Evidence limits

1. The compact package is internally consistent, but Git contains aggregate
   reuse values rather than the 36 raw replay files. Clean counter windows,
   zero failures, and cache resets remain executor assertions backed by a
   server-only 9.6 MB archive.
2. The predeclaration and final results were pushed in one commit. Its text says
   it was written first, but Git does not independently timestamp that ordering.
3. The predeclaration required one server at a time. In execution, the 4B and
   0.6B servers ran concurrently on separate GPUs. Cache-token counts had zero
   trial spread; latency is correctly excluded, but the method deviation must
   remain visible.
4. Cache capacities differ from the accepted comparison run: 101,120 versus
   96,832 tokens at 4B, and 190,896 versus 188,912 at 0.6B. Same-session
   ContextPilot controls reproduce the accepted result within 0.01 percentage
   point at k=16 and k=128 only. Tiny cross-run differences elsewhere should
   not be treated as controlled wins.
5. There is one request sequence and one menu seed. No `frequency_online`
   quality replay was run, so no function-call or no-tool claim follows.

## Research consequence

The 95–96% padded-menu headline cannot distinguish tries, clustering, or a
counter: after one request, all can identify the fixed core. The retrieved-menu
matrix remains the deployment-relevant test, where reuse is 2–17% and no one
policy dominates every cell.

ToolTrie-v0's `visit_count` is incremented but never read by its planner. That
makes popularity-aware ToolTrie-v1 a legitimate hypothesis, but
`frequency_online` is now the mandatory ablation: a v1 gain must exceed what a
plain causal counter already provides and must retain the ordinary selected-tool
text path as fallback. It also needs quality and multiple-sequence evaluation
before it can replace v0.
