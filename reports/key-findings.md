# Key findings — trie-aware tool ordering for prefix-cached LLM serving

Measured on an isolated RTX 3090 with vLLM 0.26.0 automatic prefix caching.
Primary model Qwen/Qwen3-4B, replicated on Qwen/Qwen3-0.6B. 190 accepted GPU
replays, all 33 audit checks passed. Full evidence and provenance in
[`consolidated-report.md`](consolidated-report.md); every brief question with its
status in [`brief-questions-and-answers.md`](brief-questions-and-answers.md).

```
reuse = prompt_tokens_cached / (prompt_tokens_cached + request_prefill_kv_computed_tokens_sum)
```

Findings 1–7 are ordered by the brief's own experimental questions (§7); finding
8 answers the high-level research question the project is named after (§2 Q3).
Each states which question it answers. Differences are in percentage points.
How the project's own planner works, with the code, is in
[`notes/tooltrie-v0-design.md`](../notes/tooltrie-v0-design.md).

## Headline

**Reordering tool schemas produces large prefix-cache reuse only when the tool
menu barely changes between requests. Under realistic retrieval it produces
almost none, and the small relevance gains it buys are paid for in no-tool
decisions.**

**Table 1 — prefix-cache reuse.** 200 requests per workload, 3 trials, zero
spread across trials. Reuse only; no accuracy is involved. Qwen3-4B at its
native 96,832-token capacity. **Bold marks the best value in each column**, the
convention used in every policy-comparison table below.

| Qwen3-4B | padded menu (64 tools) | retrieved menu, BM25 k=128 |
| --- | ---: | ---: |
| Original text prefill | 1.19% | 0.34% |
| Alphabetical | 37.99% | 0.44% |
| [ToolTrie-v0 (ours)](../notes/tooltrie-v0-design.md) | 87.19% | 0.89% |
| ContextPilot-derived, persistent API † | 96.16% | 1.35% |
| ContextPilot-derived, static refit † | 96.16% | 2.23% |
| Online frequency counter | **96.27%** | **2.33%** |

> **Later correction (concurrent-latency phase).** Table 1's padded column is a
> 200-request measurement, which is inside ToolTrie-v0's learning phase — it
> reaches an optimal ordering only at request 222, where ContextPilot reaches one
> at request 2. Extended to 600 requests, the padded gap closes: 97.05% against
> 97.09%, and every timing metric ties within noise. **ContextPilot's advantage on
> padded menus is cold-start speed, not ordering quality.** The retrieved column
> is unaffected — there is no common core to converge to, and ContextPilot leads
> at every depth. The "zero spread across trials" above measures determinism
> (the orderings are frozen files), not robustness; arrival order is the axis that
> actually moves these numbers. See
> [`concurrent-latency/findings.md`](concurrent-latency/findings.md) §1.5 and A.6–A.7.

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

Qwen3-4B, the primary model, from the accepted dual-model matrix; Qwen3-0.6B
alongside for scale. TTFT is per request, mean of 3 trials × 200 requests.

| k | prompt tokens/query | 4B fallback | 4B ToolTrie-v0 | 0.6B fallback |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 640 | 89.7 ms | 88.0 ms | 31.8 ms |
| 16 | 2,139 | 289.0 ms | 284.8 ms | 71.2 ms |
| 64 | 8,361 | 1,360.4 ms | 1,351.8 ms | 353.4 ms |
| 128 | 16,267 | **3,291.1 ms** | 3,280.1 ms | 940.4 ms |

Prompt cost grows about 25× from k=4 to k=128 and TTFT tracks it — to **3.3
seconds** before the first token on the primary model — while retrieval
macro-recall rises only 25.8 points. Deeper menus buy little and cost a great
deal, which is what makes the optimisation worth attempting at all. The cost is
roughly 3.5× larger at 4B than at 0.6B, so results quoted on the small model
understate it.

**But ordering does not fix it.** The three-trial TTFT intervals overlap at every
depth and no paired-difference test was predeclared, so nothing here shows that
reordering makes retrieved-menu serving faster.

## 2. Available reuse depends almost entirely on the workload
> **Brief §7 Q2** — *How much exact prefix reuse is available without changing the tool set?*

The project's central result. The table below is menu composition, so it is
model-independent; the reuse figures it explains are Qwen3-4B from Table 1,
and Qwen3-0.6B shows the same collapse.

| | padded menu | retrieved k=128 |
| --- | ---: | ---: |
| tools per request | 64 | 128 |
| drawn from a pool of | 262 | 12,353 |
| tools present in **every** request | **63** | **0** |
| overlap with previous request | **98.4%** | **7.7%** |

Reuse falls from 82–96% to 0.3–19% between them, for every policy, at both model
sizes. Membership never changed — only order. **Any deployment estimate taken
from padded menus will be optimistic by one to two orders of magnitude** —
between 41× and 98×, depending on the policy.

## 3. Frequency ordering works or fails depending on the estimator
> **Brief §7 Q3** — *Does frequency-based ordering improve reusable prefix length?*
> **Brief §7 Q4** — *Does weighting by schema length or prefill time perform better?*

The same frequency signal under two estimators, measured at both model sizes.
Qwen3-4B at 101,120-token capacity is primary; Qwen3-0.6B at 190,896 alongside,
so the effect of halving the cache is visible. Menus byte-identical to Table 1,
but these are **same-session controls at a different capacity** from Table 1's
96,832 — read the columns against each other, not against Table 1.

| Ordering | BFCL 4B | BFCL 0.6B | ToolRet 4B | ToolRet 0.6B |
| --- | ---: | ---: | ---: | ---: |
| Original text prefill | 1.19% | 1.19% | 7.15% | 13.87% |
| Alphabetical | 37.99% | 38.13% | 44.31% | 51.05% |
| `frequency_fitted` — frozen on a held-out corpus | 39.55% | 39.69% | 34.55% | 41.58% |
| `schema_cost_fitted` — frequency × schema length | 39.55% | 39.69% | 34.84% | 41.87% |
| **`frequency_online` — from the served stream** | **96.27%** | **96.27%** | **94.80%** | **94.80%** |

**The relationship is capacity-robust.** The frozen estimator beats alphabetical
on BFCL by **+1.56 points at both capacities** — 39.55 against 37.99, and 39.69
against 38.13, the same margin to the digit. On ToolRet it trails alphabetical by
**9.76 points at 4B** and 9.47 at 0.6B: halving the cache costs alphabetical
about 7 points and the frozen fit about 7 points, so the gap survives.

The online estimator reaches 96.27% at both sizes. A held-out corpus structurally
cannot know which tools a given evaluation stream happens to share — that is a
property of the eval sample, not of the tool population.

This reverses the earlier conclusion that frequency ordering does not help, and
means any comparison of frequency ordering must state which estimator it used.
**Schema-cost weighting (Q4) adds nothing over plain frequency** — 39.69% on
BFCL, identical to the digit, because it emits an identical ordering there.

## 4. Pair/triple structure is worth +0.35 points, and only where menus genuinely vary
> **Brief §7 Q5** — *How much additional benefit comes from pair/triple workflow structure?*

**The answer, on the only workloads where the question can be asked:** an online
pair/triple estimator against an online frequency counter, Qwen3-0.6B at 190,896
tokens, 200 requests each, identical prompt-token totals.

| workload | `frequency_online` | `pair_triple_online` | delta |
| --- | ---: | ---: | ---: |
| BM25 k=16 | 8.155% | **8.514%** | **+0.359** |
| BM25 k=128 | 2.412% | **2.746%** | **+0.334** |

Orderings differ on 69/200 and 194/200 records. Real, consistent in sign, and
small — pair/triple structure helps precisely where there is almost nothing left
to gain, and neither figure reaches ContextPilot at the same depth. The k=128
row is pair-only; triples are cubic in menu width and were capped.

**On padded menus the question cannot be asked at all**, which is why every
earlier attempt returned a tie. Both model sizes, 3 trials each, zero spread:

| Fitted policy | BFCL 4B | BFCL 0.6B | ToolRet 4B | ToolRet 0.6B |
| --- | ---: | ---: | ---: | ---: |
| `frequency_fitted` | **39.55%** | **39.69%** | 34.55% | 41.58% |
| `schema_cost_fitted` | **39.55%** | **39.69%** | 34.84% | 41.87% |
| `fp_tree_conditional` | **39.55%** | **39.69%** | 34.53% | 41.56% |
| `conditional_pair` | **39.55%** | **39.69%** | 34.55% | 41.58% |
| `conditional_pair_triple` | **39.55%** | **39.69%** | 34.55% | 41.58% |
| *alphabetical, for reference* | *37.99%* | *38.13%* | *44.31%* | *51.05%* |

**The five-way tie is a separate fact and pair redundancy does not explain it.**
`schema_cost_fitted` differs by schema length and `fp_tree_conditional` by tree
traversal; neither is a pair key. Their agreement with `frequency_fitted` on
BFCL is a verified property of the emitted sequences in this fitted setup, not a
consequence of the table above.

On BFCL all five land on the same value **to the digit at both capacities** —
39.55 at 4B and 39.69 at 0.6B — which is not merely close. They emit
**byte-identical `tool_ids` on all 200 records**, differing only in the label
string, because their extra statistics never discriminate and each key falls
through to the same frequency ordering. An experiment in which the pair/triple
policy produced the same file as the frequency policy did not test pair/triple
structure and find it unhelpful; it never tested it.

On ToolRet padded menus `schema_cost_fitted` differs on all 200 records and
`fp_tree_conditional` on 10 — but **both pair/triple policies are byte-identical
to `frequency_fitted` there too**, so Q5 was not posed on either padded workload.

**Why: on padded menus the pair signal is redundant with frequency.** When a
menu is a fixed core plus one varying tool, observed pair support tracks
`min(presence(a), presence(b))`, so a pair key adds little a frequency key does
not already carry:

| workload | tools in every request | `pair == min(presence)` | violations |
| --- | ---: | ---: | ---: |
| bfcl-padded64 | 63 | **100.00%** | **0** |
| toolret-padded64 | 60 | 99.39% | 46 |
| toolret-bm25-k16 | 0 | 85.16% | 2,651 |
| toolret-bm25-k128 | 0 | 67.34% | 208,615 |

**Read this as a measured property of these workloads, not a proof.** The audit
examines only pairs actually observed together, and skips triples on menus wider
than 16 — which includes `bfcl-padded64`. What is established is narrower and
sufficient: replaying both online planners over the real workloads, they emit
**0 of 200 differing orderings on `bfcl-padded64`** and **4 of 200 on
`toolret-padded64`**, which is why the fitted pair and triple policies produced
one file. It does not follow that no pair-keyed policy could ever differ, and
padded ToolRet shows the redundancy is not total even there — its 46 violations
are real.


## 5. Under cache pressure, locality barely matters and random beats every *static* ordering
> **Brief §7 Q6** — *How sensitive are results to request ordering and session locality?*
> **Brief §4.5** — requires empirical, uniform, skewed and session-bursty replays.

Qwen3-0.6B with the cache deliberately capped at 480 blocks — capacity is the
controlled variable here, so it overrides the model's native size. 24 regime-runs
accepted at 91.0–91.9% peak occupancy, plus ToolTrie-v0 added later at
90.4–90.8% (finding 8):

| Ordering | empirical | uniform | skewed | session-bursty |
| --- | ---: | ---: | ---: | ---: |
| Original | 1.18% | 0.69% | 1.24% | 0.69% |
| Alphabetical | 29.21% | 29.35% | 28.06% | 27.76% |
| Random, seed 42 | 32.16% | 31.27% | 32.67% | 30.19% |
| Frequency = schema-cost = FP-tree | 9.44% | 8.99% | 9.51% | 9.00% |
| **ToolTrie-v0** *(adaptive)* | **87.18%** | **88.54%** | **94.73%** | **91.62%** |

**Locality barely matters.** No row moves more than a few points across the four
regimes: ordering dominates the request distribution.

Among the *static* orderings a fixed random permutation leads every regime, with
the fitted policies more than 20 points behind — hard to reconcile with the
premise that engineered orderings beat arbitrary ones. One run per cell at one
seed, so this motivates a seed sweep rather than a recommendation.

But the only *adaptive* policy in the matrix beats all of them by 55–62 points.
The static-ordering result above is therefore a statement about static
orderings, not about ordering optimisation in general.

## 6. No ordering policy is a quality win
> **Brief §7 Q7** — *Does tool reordering change function-call accuracy?*

Yes, and mostly for the worse. Against ordinary text prefill at 4B (Table 2), the
best relevance gain is **+0.94 points** of full-call accuracy, and every policy
loses irrelevance accuracy. Alphabetical — the baseline our earlier comparisons
used, and the natural one to reach for — is itself worse than doing nothing
(−2.81 points at 4B, −11.57 at 0.6B), so much of the apparent benefit in any
comparison drawn against it is recovering ground that baseline gave away.

Per-case at 4B against original order, of 160 irrelevance tasks (original
correct on 141):

| Policy | cases broken | cases repaired | net |
| --- | ---: | ---: | ---: |
| ToolTrie-v0 | 5 | **4** | −1 |
| ContextPilot persistent | 5 | 0 | −5 |
| Online frequency counter | 7 | 0 | −7 |

ToolTrie-v0 is bidirectional, which is what noise looks like. The two high-reuse
policies repair **zero** cases — they only break them. Exact McNemar gives
p ≈ 0.016 for the counter and p ≈ 0.063 for ContextPilot. Small counts, so this
is a direction rather than a resolved magnitude.

**Model scale matters more than policy here.** Ordering moves full-call accuracy
by 11.6 points at 0.6B and 3.8 points at 4B, and an irrelevance regression
visible at 8B (−5.00 points) does not reproduce at 4B (+0.00, p = 1.0).
Ordering-quality conclusions drawn at small scale measure the model, not the
method — which is why the primary result uses 4B.

## 7. Where trie-aware ordering provides little or no benefit
> **Brief §7 Q8** — *Under which workloads does trie-aware ordering provide little or no benefit?*

Three regimes, in increasing order of how badly it does:

- **Retrieved menus.** On Qwen3-4B, ToolTrie-v0 reaches 0.89% at k=128 against a
  0.34% fallback. The gain is real but negligible in absolute terms.
- **Against stronger comparators.** Across both models, both ContextPilot
  adaptations beat it in all twelve systems cells, and the online counter beats it in ten of twelve. Its
  advantage over the static baselines is concentrated entirely in the padded
  regime.
- **On the padded benchmark, the workload itself is the problem.** Position is a
  property of the emitted ordering and so model-independent; the reuse column is
  Qwen3-0.6B. Reuse is almost exactly the position of the single non-shared tool:

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
  ordering sophistication.** It is a finding about how tool-serving benchmarks
  should be constructed.

## 8. The trie does not do the work — adaptivity does
> **Brief §2 Q3** — *Can frequently occurring tool sequences be represented as a
> weighted trie or prefix memory under a limited cache budget?*

Three tries were measured. None beats a method without one, and adding weights
to ours changed nothing. 

**A trie-based ordering and a frequency sort produce the same file.**
`fp_tree_conditional` builds an FP-tree from training transactions and descends
it greedily. Offline it is indistinguishable from sorting by frequency:

| Ordering | Trie nodes | Node compression | Est. token reuse |
| --- | ---: | ---: | ---: |
| original | 9,952 | 29.45% | 26.91% |
| alphabetical | 9,766 | 30.77% | 26.86% |
| frequency | **8,904** | **36.88%** | 31.35% |
| FP-tree global | **8,904** | **36.88%** | 31.35% |

Identical to the digit, and on GPU the two emit **byte-identical `tool_ids`** on
all 200 BFCL records (10 of 200 differ on ToolRet). The tree changes nothing
about what is served.

**ToolTrie-v0 loses to a thirty-line counter at both capacities.** It beats every
static ordering everywhere — by 55–62 points under the 480-block cache, the
largest margin in this report — but a policy with no tree at all beats it:

| Qwen3-0.6B | ToolTrie-v0 | online counter |
| --- | ---: | ---: |
| padded, native 190,896 | 87.19% | **96.27%** |
| padded, 480-block cache | 87.18–94.73% | **96.16–96.40%** |
| retrieved k=128 | 1.13% | **2.41%** |

Both ContextPilot arms also beat it in all twelve systems cells. **So what wins
under scarcity is adaptivity, not trie structure** — the trie's large lead in
finding 5 is over *static* orderings only.

Two qualifications on the counter's pressure column. It reached peak occupancy
0.89979 against the predeclared 0.90 gate on two of four regimes, so that matrix
validates 2/4 and is not accepted as complete pressure evidence; the threshold
was not lowered. The two regimes that pass still exceed the trie. And
ContextPilot has never been run under pressure.

That near-miss is worth recording on its own: peak occupancy *falls* as a policy
concentrates reuse better, because fewer distinct blocks stay resident. **A gate
certifying cache pressure penalises the policies it exists to reward.**

**Weighting the trie changes nothing.** `tooltrie_v1` reads the `visit_count`
that v0 never consults, in selection and in eviction. It matches v0 to five
decimal places in all four regimes, and re-deriving both planners offline on
identical menus shows **0 of 200 records differ**. The weighting does act — v1
evicts 1,497 nodes against 1,494 on empirical — but `_reachable_cached_cost`
decides almost every choice, so the tie-break holding the weight is rarely
reached. Only the empirical regime was predeclared: **1/1 predeclared run
accepted, three further regimes added afterwards and also accepted.**

**Why the trie has so little to exploit here.** A trie rewards shared prefixes
across repeated sequences. Padded menus hold 63 of 64 tools fixed, so anything
that finds that block wins without needing sequence structure; retrieved menus
share 7.7% of membership in no consistent order, so there are barely any
prefixes to share. Offline node compression never exceeds 36.88% on any
ordering, which bounds the structure available.

Sources: `reports/tooltrie-pressure/20260811-001032/`,
`reports/tooltrie-weighted/20260811-144741/`.
