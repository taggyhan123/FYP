# Key findings — trie-aware tool ordering for prefix-cached LLM serving

Measured on an isolated RTX 3090 with vLLM 0.26.0 automatic prefix caching.
Primary model Qwen/Qwen3-4B, replicated on Qwen/Qwen3-0.6B. 190 accepted GPU
replays, all 33 audit checks passed. Full evidence and provenance in
[`consolidated-report.md`](consolidated-report.md); every brief question with its
status in [`brief-questions-and-answers.md`](brief-questions-and-answers.md).

```
reuse = prompt_tokens_cached / (prompt_tokens_cached + request_prefill_kv_computed_tokens_sum)
```

Findings are ordered by the brief's own experimental questions (§7), and each
states which question it answers. Differences are quoted in percentage points.

## Headline

**Reordering tool schemas produces large prefix-cache reuse only when the tool
menu barely changes between requests. Under realistic retrieval it produces
almost none, and no ordering policy is a quality win.**

**Table 1 — prefix-cache reuse.** 200 requests per workload, 3 trials, zero
spread across trials. Reuse only; no accuracy is involved.

| Qwen3-4B | padded menu (64 tools) | retrieved menu, BM25 k=128 |
| --- | ---: | ---: |
| Original text prefill | 1.19% | 0.34% |
| Alphabetical | 37.99% | 0.44% |
| ToolTrie-v0 (ours) | 87.19% | 0.89% |
| ContextPilot-derived, persistent API † | **96.16%** | 1.35% |
| ContextPilot-derived, static refit † | **96.16%** | **2.23%** |
| Online frequency counter | 96.27% | 2.33% |

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
the one that wins Table 2.

---

## 1. The problem is real: tool-schema prefill dominates TTFT
> **Brief §7 Q1** — *How much of TTFT is caused by selected tool-schema prefill?*

| k | prompt tokens/query | TTFT (fallback) | TTFT (ToolTrie-v0) |
| ---: | ---: | ---: | ---: |
| 4 | 640 | 32.3 ms | 31.2 ms |
| 16 | 2,139 | 70.7 ms | 70.3 ms |
| 64 | 8,361 | 349.7 ms | 350.0 ms |
| 128 | 16,267 | 937.0 ms | 933.5 ms |

Prompt cost grows about 25× from k=4 to k=128 and TTFT tracks it, from 32 ms to
937 ms, while retrieval macro-recall rises only 25.8 points. Deeper menus buy
little and cost a great deal, which is what makes the optimisation worth
attempting at all.

**But ordering does not fix it.** The three-trial TTFT intervals overlap at every
depth and no paired-difference test was predeclared, so nothing here shows that
reordering makes retrieved-menu serving faster.

## 2. Available reuse depends almost entirely on the workload
> **Brief §7 Q2** — *How much exact prefix reuse is available without changing the tool set?*

The project's central result. The two workloads in Table 1 differ structurally:

| | padded menu | retrieved k=128 |
| --- | ---: | ---: |
| tools per request | 64 | 128 |
| drawn from a pool of | 262 | 12,353 |
| tools present in **every** request | **63** | **0** |
| overlap with previous request | **98.4%** | **7.7%** |

Reuse falls from 82–96% to 0.3–19% between them, for every policy, at both model
sizes. Membership never changed — only order. **Any deployment estimate taken
from padded menus will be optimistic by roughly two orders of magnitude.**

## 3. Frequency ordering works or fails depending on the estimator
> **Brief §7 Q3** — *Does frequency-based ordering improve reusable prefix length?*
> **Brief §7 Q4** — *Does weighting by schema length or prefill time perform better?*

Fitting tool frequency on a held-out corpus reaches 39.69% on padded menus, and
on ToolRet is *worse* than alphabetical. Estimating the same signal online from
the served stream reaches 96.27%. A held-out corpus structurally cannot know
which tools a given evaluation stream happens to share.

This reverses the earlier conclusion that frequency ordering does not help, and
means any comparison of frequency ordering must state which estimator it used.
Schema-cost weighting (Q4) adds nothing over plain frequency: on padded menus it
emits an identical ordering.

## 4. Pair/triple structure was not actually tested on the padded workload
> **Brief §7 Q5** — *How much additional benefit comes from pair/triple workflow structure?*

On BFCL the five fitted policies — frequency, schema-cost, FP-tree, pair and
triple — emit **byte-identical `tool_ids` on all 200 records**, differing only in
the label string. Their extra statistics never discriminate, so each falls
through to the same frequency key. An experiment in which the pair/triple policy
produced the same file as the frequency policy did not test pair/triple structure
and find it unhelpful; it never tested it.

On ToolRet the policies genuinely differ, and there the answer is no benefit: all
sit near 41.6% against alphabetical's 51.05%.

## 5. Under cache pressure, a fixed random ordering wins
> **Brief §7 Q6** — *How sensitive are results to request ordering and session locality?*
> **Brief §4.5** — requires empirical, uniform, skewed and session-bursty replays.

Cache capped at 480 blocks, all 24 regime-runs accepted at 91.0–91.9% peak
occupancy:

| Ordering | empirical | uniform | skewed | session-bursty |
| --- | ---: | ---: | ---: | ---: |
| Original | 1.18% | 0.69% | 1.24% | 0.69% |
| Alphabetical | 29.21% | 29.35% | 28.06% | 27.76% |
| **Random, seed 42** | **32.16%** | **31.27%** | **32.67%** | **30.19%** |
| Frequency = schema-cost = FP-tree | 9.44% | 8.99% | 9.51% | 9.00% |

A fixed random permutation leads in **every** regime, with the fitted policies
more than 20 points behind. The locality regime itself barely moves any row —
ordering matters far more than the request distribution. One run per cell at one
seed, so this motivates a seed sweep rather than a recommendation; but it is hard
to reconcile with the premise that engineered orderings beat arbitrary ones.

## 6. No ordering policy is a quality win
> **Brief §7 Q7** — *Does tool reordering change function-call accuracy?*

Yes, and mostly for the worse. Against ordinary text prefill at 4B (Table 2), the
best relevance gain is **+0.94 points** of full-call accuracy, and every policy
loses irrelevance accuracy. Alphabetical — the baseline our earlier comparisons
used, and the natural one to reach for — is itself worse than doing nothing
(−2.81 points at 4B, −11.57 at 0.6B), so much of the apparent benefit in any
comparison drawn against it is recovering ground that baseline gave away.

Per-case at 4B, of 160 irrelevance tasks, the high-reuse policies never repair a
case and only break them: ContextPilot loses 5 and recovers 0, the counter loses
7 and recovers 0, while ToolTrie-v0 is bidirectional at 5 and 4. Small counts, so
this is a direction rather than a resolved magnitude.

**Model scale matters more than policy here.** Ordering moves full-call accuracy
by 11.6 points at 0.6B and 3.8 points at 4B, and an irrelevance regression
visible at 8B (−5.00 points) does not reproduce at 4B (+0.00, p = 1.0).
Ordering-quality conclusions drawn at small scale measure the model, not the
method — which is why the primary result uses 4B.

## 7. Where trie-aware ordering provides little or no benefit
> **Brief §7 Q8** — *Under which workloads does trie-aware ordering provide little or no benefit?*

Three regimes, in increasing order of how badly it does:

- **Retrieved menus.** ToolTrie-v0 reaches 0.89% at k=128 against a 0.34%
  fallback. The gain is real but negligible in absolute terms.
- **Against stronger comparators.** Both ContextPilot adaptations beat it in all
  twelve systems cells, and the online counter beats it in ten of twelve. Its
  advantage over the static baselines is concentrated entirely in the padded
  regime.
- **On the padded benchmark, the workload itself is the problem.** Reuse there is
  almost exactly the position of the single non-shared tool:

  | ordering | position of that tool | reuse |
  | --- | ---: | ---: |
  | original | 1.0 | 1.19% |
  | alphabetical | 24.1 | 38.13% |
  | ToolTrie-v0 | 57.3 | 87.19% |
  | ContextPilot | 63.7 | 96.16% |
  | online frequency counter | 63.7 | 96.27% |

  A thirty-line counter with no clustering, no trie and no training corpus
  matches hierarchical clustering, separated by 0.11 points — 95 cache blocks
  across 200 requests. **Results on this workload cannot support claims about
  ordering sophistication.** That is a finding about how tool-serving benchmarks
  should be constructed, and it is arguably the most transferable result here.

## Scope and limitations

- **No ordering policy is shown to be faster.** Finding 1 establishes that
  tool-schema prefill dominates TTFT, but the three-trial intervals between
  orderings overlap at every depth. Planning cost is also excluded from every
  figure and is not negligible: static refit costs 0.75 s/request at n=800
  against roughly 2–7 ms for the persistent API.
- One menu seed and one request sequence per condition; single-tenant server with
  no competing traffic. All figures are therefore best cases.
- The cache-pressure matrix (finding 5) is one run per cell at a single random
  seed on a deliberately capped 480-block cache. Its values must not be compared
  numerically against the 190,896-token results above.
- Quality uses a reduced BFCL-style AST checker, so values are comparable within
  this work but not against published BFCL leaderboard scores.
- No equivalence margins were declared, so quality comparisons are estimation
  only.
- **How much reuse is theoretically available on retrieved menus has not been
  established.** Finding 2 reports what policies achieved, not the ceiling.

## Suggested next steps

1. **Measure end-to-end latency including planning cost.** Finding 1 shows the
   cost is real; nothing yet shows an ordering recovers it.
2. **Sweep the random seed under cache pressure.** Finding 5 shows one random
   permutation beating every engineered ordering by 20+ points in all four
   locality regimes. Either that generalises, which undercuts the premise of
   ordering optimisation, or it does not — both are worth knowing.
3. **Establish the retrieved-menu ceiling.** Finding 2 reports what was captured;
   the gap to what is available would decide whether better ordering is worth
   pursuing at all.
4. **Ablate the ToolTrie retention parameters.** Its recency window has been fixed
   at 128 throughout with no sensitivity analysis, and its `visit_count` field is
   maintained but never read — the popularity signal its objective currently
   lacks.
5. **Add menu seeds to the irrelevance measurement.** 160 cases at one seed cannot
   settle whether high-reuse orderings genuinely harm refusal behaviour.
6. **Build a retrieval-realistic benchmark.** Finding 7 argues that padded menus
   should not be used to evaluate ordering policies at all.
