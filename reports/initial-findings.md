# Initial research findings and recommendation

## Result in one paragraph

The public workloads contain a measurable analytical prefix-locality signal,
but it is workload- and evidence-dependent. Across 45,815 canonical
tools, median schema length is 70 Qwen tokens and P95 is
220. ToolRet's median relevance set and BFCL's median
exposed menu both contain one tool, which limits what reordering can achieve on
many tasks. On empirical dataset order, schema-cost-weighted ordering raises the
ToolRet tool-prefix block-reuse estimate from
26.91% to
31.80% (4.89
percentage points). Frequency ordering raises the BFCL menu estimate from
10.98% to
18.88% (7.90
points). These are analytical estimates, not vLLM hit-rate or latency results.

## Evidence obtained locally

- ToolRet: 7,961 tasks, 7,652
  tools appearing in gold relevance labels, with every label resolved.
- BFCL subset: 1,240 tasks and
  1,362 distinct canonical functions appearing in
  exposed menus.
- Schema diagnostics flag 19,921/45,815 tools. Most flags are
  empty parameter objects, which can be legitimate; malformed and missing-field
  counts are reported separately.
- Controlled support-skewed replay produces much more reuse than uniform or
  empirical access, but under unbounded retention that gap comes from
  resampling with replacement — repeated requests — not from locality. Only the
  finite-cache tables in `access-patterns.md` speak to request ordering.
- Classic FP-tree global order equals the frequency order in this baseline.
  Conditional pattern mining is still untested.

## Prefix-cache sanity results

Measured on Qwen/Qwen3-0.6B, block size 16, 11807 GPU blocks, 5 trials per scenario with the prefix cache reset before each trial. Intervals are 95% Student-t half-widths.

| Check | Cached / prompt tokens | Reuse | TTFT on (ms) | TTFT off (ms) |
| --- | --- | --- | --- | --- |
| Cold prompt | 0 / 303 | 0.00% | 52.4 ± 28.8 | 51.5 ± 27.3 |
| Identical prompt reuse | 288 / 303 | 95.05% | 43.0 ± 1.2 | 40.5 ± 2.3 |
| Changed second tool | 128 / 309 | 41.42% | 41.4 ± 1.1 | 42.3 ± 2.2 |
| Reordered first two tools | 48 / 303 | 15.84% | 37.7 ± 5.7 | 43.0 ± 4.2 |
| Original restored | 288 / 303 | 95.05% | 41.1 ± 2.1 | 39.4 ± 2.8 |

The control served `enable_prefix_caching=False` and reported 0 cached tokens in every scenario, so the cache-on/cache-off output comparison is meaningful.

## Prefill cost and the measurement floor

Menus are one gold tool padded with distractors from a fixed global catalog. Cold is measured after a prefix-cache reset; warm is an identical repeat. 5 trials per point.

| Menu tools | Prompt tokens | Cold TTFT on (ms) | Warm TTFT on (ms) | Warm TTFT off (ms) | Speedup |
| --- | --- | --- | --- | --- | --- |
| 1 | 250 | 25.6 | 23.4 | 23.1 | 1.0x |
| 4 | 433 | 25.2 | 16.9 | 23.7 | 1.4x |
| 16 | 1,771 | 62.1 | 35.3 | 56.5 | 1.6x |
| 64 | 6,742 | 248.8 | 54.1 | 245.6 | 4.5x |
| 128 | 13,422 | 647.8 | 85.2 | 640.7 | 7.5x |
| 200 | 20,627 | 1282.7 | 120.7 | 1275.2 | 10.6x |

Prefix caching first produces a material gain at **4 tools** and grows steeply from there.

The cache-disabled control shows cold and warm within noise of each other at every size, so the gap above is prefix caching and not a warmup artifact. Note that within-run cold-vs-warm separation alone is not sufficient evidence: the control also 'separates' at one tool, where no cache exists. The cache-on/cache-off comparison is the trustworthy one.

This supersedes the earlier reading that prefix reuse buys nothing. It buys nothing *at 303 tokens*, which is simply below the floor.

## Analytical estimate versus measured cache hits

| Workload | Predicted cacheable | Measured cached | Measured/predicted | Predicted reuse | Measured reuse |
| --- | --- | --- | --- | --- | --- |
| menu64-alphabetical | 428,016 | 526,432 | 1.23x | 38.42% | 38.15% |
| menu64-frequency | 40,448 | 60,864 | 1.50x | 3.63% | 4.41% |

The tool-unit model under-predicts rather than over-predicts. It ignores the chat template and system preamble, which are identical across requests and are themselves cached, and that outweighs the partial blocks it loses at tool boundaries.

The ordering comparison inverts the gold-only recommendation. On padded menus, ordering by benchmark support puts the *task-specific* tools first and pushes the shared catalog behind them, destroying the common prefix. A stable global order that ignores task-specificity keeps the shared catalog in front. Frequency ordering is the right choice only when the whole menu is task-specific, which is not what a connected tool catalog looks like.

## Does the winning ordering preserve function-call quality?

100 BFCL tasks (20 per category, 64-tool menus), scored against BFCL `possible_answer` ground truth. Alphabetical is the ordering measured strongest for cache reuse; frequency is the comparison Task D's analytical model originally recommended.

`Qwen/Qwen3-0.6B`:

| Ordering | Function-name accuracy | Full accuracy | No-tool accuracy |
| --- | --- | --- | --- |
| alphabetical | 67.50% | 47.50% | 95.00% |
| frequency | 77.50% | 51.25% | 85.00% |

`Qwen/Qwen3-8B`, identical workload:

| Ordering | Function-name accuracy | Full accuracy | No-tool accuracy |
| --- | --- | --- | --- |
| alphabetical | 91.25% | 81.25% | 95.00% |
| frequency | 91.25% | 81.25% | 90.00% |

The name/full-accuracy gap that looked real at 0.6B is not present at 8B — both orderings score identically there. Checking a counterintuitive small-model result against a second model size found that most of the apparent quality tradeoff was a small-model artefact; only a smaller no-tool-accuracy gap favouring alphabetical survives at both scales. One run per condition, no repeats; read as directional rather than final.

## Does request order matter on the live GPU cache?

Every other GPU experiment above resets the prefix cache before each trial, which is repeatable but erases the cross-request dependency locality is actually about. This instead runs one continuous session per replay condition — a single reset, then every request in sequence — comparing `empirical` against `session_bursty`, the one replay pair that is a strict permutation of the same task multiset.

| Replay | Total prompt tokens | Measured reuse | Predicted reuse (bounded trie) | Predicted evictions |
| --- | --- | --- | --- | --- |
| empirical | 668,440 | 42.97% | 35.00% | 2,927 |
| session_bursty | 668,440 | 43.03% | 35.03% | 2,918 |

Total token volume (668,440) versus real cache capacity (190,896 tokens) for this run.

Real eviction happens here, but session order barely moves measured reuse — a much smaller gap than the offline-only analysis in `access-patterns.md` found for raw, unordered task sequences. Applying the ordering that already wins on reuse, plus shared-catalog padding, makes almost every request's prefix similar regardless of task clustering, which swamps whatever weaker signal session locality would otherwise contribute in this regime.

## Recommendation

Measured evidence now supports alphabetical ordering over the frequency
ordering this analysis originally recommended: on padded, deployment-realistic
menus, alphabetical measures 38.15% cache reuse versus frequency's 4.41%, and
a follow-up quality check found this does not cost function-selection accuracy
at deployment-grade model scale (identical on `Qwen3-8B`), with only a smaller
no-tool-accuracy gap surviving at both model sizes checked. Continue the exact
prompt-level ToolTrie baseline with alphabetical as the default ordering,
reporting the original order and fixed-random controls alongside it.

The likely publishable refinement is not a generic "reorder context into a
trie" claim, because closely related cache-aware context ordering already
exists. The more defensible direction is tool-specific cache admission using
schema cost plus workflow co-occurrence, while preserving an active/authorized
tool manifest — the quality measurement this recommendation previously lacked
now exists for the leading ordering, though not yet for the other five.

Do not pursue arbitrary independent KV concatenation yet. Native exact APC
already converts the local token-reuse signal into a repeatable TTFT benefit
(10.6x at 200 tools) without a measured BFCL quality regression for the
strongest ordering. The open question is coverage, not whether the mechanism
works at all.

## What remains before the extensions in the brief

1. Function-call quality for the remaining four orderings, and repeated trials
   with larger per-category samples for the two already scored.
2. TTFT measured directly on partial-reuse workloads, rather than
   extrapolated from the 0%/100%-reuse endpoints.
3. Live-cache eviction checked under orderings other than alphabetical, and
   replay pairs other than empirical/session_bursty.
4. GPU/KV memory *usage* under eviction pressure — eviction's effect on
   reuse is now measured; memory footprint during eviction is not.
5. A text-prefill fallback path that preserves model semantics, which Task E
   specifies but which has not been built.
