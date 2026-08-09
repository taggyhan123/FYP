# Key findings — trie-aware tool ordering for prefix-cached LLM serving

Measured on an isolated RTX 3090 with vLLM 0.26.0 automatic prefix caching.
Primary model Qwen/Qwen3-4B, replicated on Qwen/Qwen3-0.6B. 190 accepted GPU
replays, all 33 audit checks passed. Full evidence and provenance in
[`consolidated-report.md`](consolidated-report.md).

```
reuse = prompt_tokens_cached / (prompt_tokens_cached + request_prefill_kv_computed_tokens_sum)
```

Quality is one n=800 BFCL replay per condition: 640 relevance cases
(function-name, full call) and 160 irrelevance cases (no-tool). Differences are
quoted in percentage points.

## Headline

**Reordering tool schemas produces large prefix-cache reuse only when the tool
menu barely changes between requests. Under realistic retrieval it produces
almost none.**

| Qwen3-4B | padded menu | retrieved menu (k=128) | full call | no-tool |
| --- | ---: | ---: | ---: | ---: |
| Original text prefill | 1.19% | 0.34% | 76.09% | **88.12%** |
| Alphabetical | 37.99% | 0.44% | 73.28% | 85.62% |
| ToolTrie-v0 (ours) | 87.19% | 0.89% | 75.31% | 87.50% |
| ContextPilot, persistent API | **96.16%** | 1.35% | **77.03%** | 85.00% |
| ContextPilot, static refit | **96.16%** | **2.23%** | **77.03%** | 85.00% |
| Online frequency counter | 96.27% | 2.33% | 76.88% | 83.75% |

Reuse falls from 82–96% to 0.3–19% between the two menu types, for every
policy, at both model sizes.

## 1. The widely-quoted 95–96% figure is a property of the workload

Padded evaluation menus share 60–63 of their 64 tools between requests; menus
built by BM25 retrieval do not. We reproduce the published ContextPilot figure
exactly (96.16%) and show it collapses to 1.35% on retrieved menus of the same
size. Any deployment estimate taken from padded menus will be optimistic by
roughly two orders of magnitude.

## 2. The standard padded benchmark cannot distinguish between methods

The BFCL padded workload contains 63 tools present in every request and 198
present in one. Measured reuse is almost exactly the position of the single
non-shared tool:

| ordering | position of that tool | reuse |
| --- | ---: | ---: |
| original | 1.0 | 1.19% |
| alphabetical | 24.1 | 38.13% |
| ToolTrie-v0 | 57.3 | 87.19% |
| ContextPilot | 63.7 | 96.16% |
| online frequency counter | 63.7 | 96.27% |

A thirty-line counter with no clustering, no trie and no training corpus reaches
the same result as hierarchical clustering — both saturate a structural ceiling,
separated by 95 cache blocks across 200 requests. **Results on this workload
therefore cannot support claims about ordering sophistication.** This is a
finding about how tool-serving benchmarks should be constructed.

## 3. ToolTrie-v0 is outperformed

Both ContextPilot adaptations beat it in all twelve systems cells; the online
counter beats it in ten of twelve. Its advantage over the static baselines is
real but concentrated entirely in the padded regime.

## 4. No ordering policy is a quality win

Against ordinary text prefill at 4B, the best relevance gain is **+0.94 points**
of full-call accuracy, and every policy loses irrelevance accuracy. Alphabetical
— the baseline most published comparisons use — is itself worse than doing
nothing (−2.81 points at 4B, −11.57 at 0.6B), so much of the reported benefit in
the literature is recovering ground that baseline gave away.

Per-case at 4B, of 160 irrelevance tasks, the high-reuse policies never repair a
case and only break them: ContextPilot loses 5 and recovers 0, the counter loses
7 and recovers 0, while ToolTrie-v0 is bidirectional at 5 and 4. The counts are
small, so this indicates a direction rather than a resolved magnitude.

## 5. How frequency is estimated decides whether it works

Fitting tool frequency on a held-out corpus reaches 39.69%, and on ToolRet is
worse than alphabetical. Estimating the same signal online from the served
stream reaches 96.27%. A held-out corpus structurally cannot know which tools a
given evaluation stream shares. This reverses our earlier conclusion that
frequency-based ordering does not help, and it means published comparisons
should state which estimator they used.

## 6. Ordering effects on quality are a small-model artifact

Ordering moves full-call accuracy by 11.6 points at 0.6B and 3.8 points at 4B.
An irrelevance regression visible at 8B (−5.00 points) does not reproduce at 4B
(+0.00, p = 1.0). Ordering-quality conclusions drawn at small scale measure the
model rather than the method — which is why the primary result uses 4B.

## Scope and limitations

- **Latency is not established.** Reuse measures prefill work avoided, not
  time-to-first-token. Planning cost is excluded from every figure and is not
  negligible: static refit costs 0.75 s/request at n=800 against roughly 2–7 ms
  for the persistent API.
- One menu seed and one request sequence per condition; single-tenant server
  with no competing traffic. All figures are therefore best cases.
- Quality uses a reduced BFCL-style AST checker, so values are comparable within
  this work but not against published BFCL leaderboard scores.
- No equivalence margins were declared, so quality comparisons are estimation
  only.
- On the padded workload the five fitted co-occurrence policies emit identical
  orderings, so that workload cannot speak to the value of pair/triple structure.

## Suggested next steps

1. **Measure end-to-end latency including planning cost.** This is the gap
   between "prefill work avoided" and any deployment claim.
2. **Ablate the ToolTrie retention parameters.** Its recency window has been
   fixed at 128 throughout with no sensitivity analysis, and its `visit_count`
   field is maintained but never read — the popularity signal its objective
   currently lacks.
3. **Add menu seeds to the irrelevance measurement.** 160 cases at one seed
   cannot settle whether high-reuse orderings genuinely harm refusal behaviour.
4. **Build a retrieval-realistic benchmark.** The central negative result argues
   that padded menus should not be used to evaluate ordering policies at all.
