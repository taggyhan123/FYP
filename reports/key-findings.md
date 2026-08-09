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

**Table 1 — prefix-cache reuse.** 200 requests per workload, 3 trials, zero
spread. Reuse only; no accuracy is involved.

| Qwen3-4B | padded menu (64 tools) | retrieved menu, BM25 k=128 |
| --- | ---: | ---: |
| Original text prefill | 1.19% | 0.34% |
| Alphabetical | 37.99% | 0.44% |
| ToolTrie-v0 (ours) | 87.19% | 0.89% |
| ContextPilot-derived, persistent API † | **96.16%** | 1.35% |
| ContextPilot-derived, static refit † | **96.16%** | **2.23%** |
| Online frequency counter | 96.27% | 2.33% |

The two workloads differ structurally, and that difference is the finding:
padded menus hold 64 tools drawn from a 262-tool pool with **63 present in every
request** (98.4% overlap between consecutive requests), while BM25 retrieval
draws 128 tools from a 12,353-tool pool with **none universal** (7.7% overlap).
Reuse falls from 82–96% to 0.3–19% between them, for every policy, at both
model sizes.

**Table 2 — function-calling accuracy.** A separate experiment: one n=800 BFCL
replay per condition, 640 relevance cases and 160 irrelevance cases. No reuse is
involved.

| Qwen3-4B | full call (640 cases) | no-tool (160 cases) |
| --- | ---: | ---: |
| Original text prefill | 76.09% | **88.12%** |
| Alphabetical | 73.28% | 85.62% |
| ToolTrie-v0 (ours) | 75.31% | 87.50% |
| ContextPilot-derived, persistent API † | **77.03%** | 85.00% |
| ContextPilot-derived, static refit † | **77.03%** | 85.00% |
| Online frequency counter | 76.88% | 83.75% |

† Both ContextPilot rows are **ordering-only adaptations at alpha=0.001** on the
pinned upstream commit. Neither is the full ContextPilot system: both omit
relevance annotations, eviction feedback and de-duplication. They should not be
read as an evaluation of that system.

The two tables move almost independently — the ordering that wins Table 1 is not
the one that wins Table 2, which is why no policy is best across both.

## 1. The 95–96% padded-menu figure is a property of the workload

Padded evaluation menus share 60–63 of their 64 tools between requests; menus
built by BM25 retrieval do not. The strongest ordering we measured reaches
96.16% on padded menus and **1.35% on retrieved menus of the same size**. Any
deployment estimate taken from padded menus will be optimistic by roughly two
orders of magnitude.

The 96.16% also reproduces our own earlier internal measurement of the same
ordering under a different alpha, which is why we treat it as a stable property
of the workload rather than a tuning artifact. We make no claim about numbers
published elsewhere; nothing here was compared against a figure from the
ContextPilot paper.

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
the same result as hierarchical clustering, separated by 0.11 points — 95 cache
blocks across 200 requests. **Results on this workload therefore cannot support
claims about ordering sophistication.** This is a finding about how tool-serving
benchmarks should be constructed.

## 3. ToolTrie-v0 is outperformed

Both ContextPilot adaptations beat it in all twelve systems cells; the online
counter beats it in ten of twelve. Its advantage over the static baselines is
real but concentrated entirely in the padded regime.

## 4. No ordering policy is a quality win

Against ordinary text prefill at 4B, the best relevance gain is **+0.94 points**
of full-call accuracy, and every policy loses irrelevance accuracy. Alphabetical
— the baseline our own earlier comparisons used, and the natural one to reach
for — is itself worse than doing nothing (−2.81 points at 4B, −11.57 at 0.6B).
Much of the apparent benefit in any comparison drawn against it is therefore
recovering ground that baseline gave away, which is why the table above reports
against original text prefill instead.

Per-case at 4B, of 160 irrelevance tasks, the high-reuse policies never repair a
case and only break them: ContextPilot loses 5 and recovers 0, the counter loses
7 and recovers 0, while ToolTrie-v0 is bidirectional at 5 and 4. The counts are
small, so this indicates a direction rather than a resolved magnitude.

## 5. Tool-schema prefill dominates TTFT, and retrieval depth sets the cost

The brief's first question. Prompt size and time-to-first-token both scale with
menu depth, on the retrieved workloads:

| k | prompt tokens/query | TTFT (fallback) | TTFT (ToolTrie-v0) |
| ---: | ---: | ---: | ---: |
| 4 | 640 | 32.3 ms | 31.2 ms |
| 16 | 2,139 | 70.7 ms | 70.3 ms |
| 64 | 8,361 | 349.7 ms | 350.0 ms |
| 128 | 16,267 | 937.0 ms | 933.5 ms |

Prompt cost grows about 25× from k=4 to k=128 while retrieval macro-recall rises
only 25.8 points, so deeper menus buy little and cost a great deal. **Ordering
does not change this**: the three-trial TTFT intervals overlap at every depth and
no paired-difference test was predeclared, so nothing here establishes that
reordering makes retrieved-menu serving faster.

## 6. Under cache pressure, a fixed random ordering wins

The brief asks for the empirical order plus uniform, skewed and session-bursty
replays. With the cache capped at 480 blocks and all 24 regime-runs accepted at
91.0–91.9% peak occupancy:

| Ordering | empirical | uniform | skewed | session-bursty |
| --- | ---: | ---: | ---: | ---: |
| Original | 1.18% | 0.69% | 1.24% | 0.69% |
| Alphabetical | 29.21% | 29.35% | 28.06% | 27.76% |
| **Random, seed 42** | **32.16%** | **31.27%** | **32.67%** | **30.19%** |
| Frequency = schema-cost = FP-tree | 9.44% | 8.99% | 9.51% | 9.00% |

A fixed random permutation leads in **every** regime, and the fitted policies
trail it by more than 20 points. Locality regime barely moves any row — the
ordering matters far more than the request distribution. This is one run per
cell at one seed with no repeated-trial interval, so it motivates a seed sweep
rather than a recommendation; but it is hard to reconcile with the premise that
engineered orderings beat arbitrary ones.

## 7. How frequency is estimated decides whether it works

Fitting tool frequency on a held-out corpus reaches 39.69%, and on ToolRet is
worse than alphabetical. Estimating the same signal online from the served
stream reaches 96.27%. A held-out corpus structurally cannot know which tools a
given evaluation stream shares. This reverses our earlier conclusion that
frequency-based ordering does not help, and it means any comparison of
frequency ordering must state which estimator it used.

## 8. Ordering effects on quality are a small-model artifact

Ordering moves full-call accuracy by 11.6 points at 0.6B and 3.8 points at 4B.
An irrelevance regression visible at 8B (−5.00 points) does not reproduce at 4B
(+0.00, p = 1.0). Ordering-quality conclusions drawn at small scale measure the
model rather than the method — which is why the primary result uses 4B.

## Scope and limitations

- **No ordering policy is shown to be faster.** Finding 5 establishes that
  tool-schema prefill dominates TTFT, but the three-trial intervals between
  orderings overlap at every depth. Planning cost is also excluded from every
  figure and is not negligible: static refit costs 0.75 s/request at n=800
  against roughly 2–7 ms for the persistent API.
- One menu seed and one request sequence per condition; single-tenant server
  with no competing traffic. All figures are therefore best cases.
- The cache-pressure matrix (finding 6) is one run per cell at a single random
  seed, on a deliberately capped 480-block cache. Its values must not be
  compared numerically against the 190,896-token results above.
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
3. **Sweep the random seed under cache pressure.** Finding 6 shows one random
   permutation beating every engineered ordering by 20+ points in all four
   locality regimes. Either that generalises, which undercuts the premise of
   ordering optimisation, or it does not, which is equally worth knowing.
4. **Add menu seeds to the irrelevance measurement.** 160 cases at one seed
   cannot settle whether high-reuse orderings genuinely harm refusal behaviour.
5. **Build a retrieval-realistic benchmark.** The central negative result argues
   that padded menus should not be used to evaluate ordering policies at all.
