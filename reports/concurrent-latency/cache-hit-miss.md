# Cache hits and misses, in absolute terms

**Have we investigated this? Partly, until now.** Every "reuse" number in the
concurrent-latency work *is* the absolute cache hit rate: the share of prompt
tokens vLLM served from its prefix cache instead of recomputing. It equals
vLLM's own counter (`prefix_cache_hits / prefix_cache_queries`) to the token,
and the miss rate is 100 minus it. But it was reported mostly as ratios ("v1 caches
2.1x ContextPilot") and never broken down.

This document breaks it down:

- the absolute hit and miss rates for every policy and menu size (§1);
- what the hits actually are (§2);
- why the misses happen, and the most any ordering could ever reach (§3);
- how much the size of the cache matters (§4).

**Setup.** One RTX 3090, vLLM 0.26.0 with prefix caching unmodified, dense
retriever, 200 single-turn requests per run, Qwen3-0.6B unless stated.

---

## How this was measured

No new GPU runs. `scripts/analyze_prefix_cache.py` does three things with the
existing replays:

1. **Rebuilds every prompt** exactly as vLLM tokenised it. The token counts
   match the served prompts in every request of every run.
2. **Replays them through a simulated prefix cache**
   (`src/tatm/prefix_cache_sim.py`). It models 16-token blocks, the server's
   real cache size, and vLLM's eviction order. The simulation matches vLLM's
   *measured* cached tokens exactly in 198-200 of 200 requests in all 48 runs
   analysed, so it can be trusted to answer "what if" questions.
3. **Asks what else was possible**: the best order for each request given what
   was in the cache, and an optimistic upper bound for any ordering at all.

The runs under load give **the same hit counts as the one-at-a-time runs**:
identical to the token in 23 of 24 runs, and within 32 tokens in the last. So
everything below also applies to the latency and throughput experiments.

---

## 1. Absolute hit and miss rates

Share of prompt tokens served from the cache (hit %). The miss rate is 100
minus the hit rate.

| policy | 4 tools | 10 tools | 16 tools | 64 → 10 | 64 tools | 128 tools |
|---|---|---|---|---|---|---|
| no reordering | 14.35 | 6.38 | 4.11 | 6.38 | 0.82 | 0.36 |
| alphabetical | 13.94 | 7.97 | 5.34 | 11.67 | 1.52 | 0.43 |
| frequency | 15.84 | 8.68 | 6.29 | 9.39 | 1.39 | 0.47 |
| ToolTrie-v0 | 15.67 | 9.94 | 7.32 | 16.12 | 2.40 | 0.91 |
| ContextPilot | **18.16** | 12.46 | 11.58 | 20.16 | 2.98 | 1.21 |
| **ToolTrie-v1** | 17.97 | **13.71** | **13.18** | **27.64** | **6.22** | **3.73** |
| *ToolTrie-v1 miss rate* | *82.0* | *86.3* | *86.8* | *72.4* | *93.8* | *96.3* |

"64 → 10" means retrieve 64, reorder, show the first 10. Mean prompt length is
594 tokens at 4 tools, 1,335 at 10, 2,071 at 16, 7,931 at 64 and 15,667 at
128.

**Even the best policy recomputes most of every prompt.** At 64 tools, v1 still
recomputes 93.8%. In absolute tokens, 200 requests send 1.59 million prompt
tokens: v1 serves 98,736 of them from the cache, ContextPilot 47,264, and no
reordering 13,040.

---

## 2. What the hits are

**48 tokens of every prompt are free for every policy.** The first 50 tokens of
every prompt are identical: the chat template's tool instructions, up to the
first tool's name. The cache works in whole 16-token blocks, so 48 of those
tokens hit on every request after the first, whatever the ordering. That fixed
part is large when prompts are short, and it says nothing about the ordering:

| | 4 tools | 64 tools | 128 tools | 64 → 10 |
|---|---|---|---|---|
| share of every prompt that is the fixed template | 8.04% | 0.60% | 0.30% | 3.57% |

Splitting each policy's hits into that fixed part and **real tool reuse**:

| policy | tool reuse, 4 tools | tool reuse, 64 tools | tool reuse, 128 tools | tool reuse, 64 → 10 | requests with any tool reuse, 64 tools (of 199) |
|---|---|---|---|---|---|
| no reordering | 6.30% | 0.22% | 0.06% | 2.80% | 17 |
| alphabetical | 5.90% | 0.92% | 0.13% | 8.12% | 34 |
| frequency | 7.80% | 0.79% | 0.17% | 6.16% | 37 |
| ToolTrie-v0 | 7.62% | 1.80% | 0.61% | 12.58% | 54 |
| ContextPilot | **10.12%** | 2.38% | 0.90% | 16.60% | 50 |
| **ToolTrie-v1** | 9.93% | **5.62%** | **3.43%** | **24.06%** | **64** |

- **In tool terms v1's lead is larger than the headline ratios.** At 64 tools,
  v1's tool reuse is 2.4x ContextPilot's (5.62 vs 2.38) and about 26x no
  reordering's (5.62 vs 0.22). The headline "2.1x" and "7.6x" include the fixed
  template, which every policy gets.
- **At 4 tools, about half or more of every policy's hits are the template**
  (44% for v1 and ContextPilot, 56-58% for no reordering and alphabetical).
  ContextPilot's tool reuse is slightly higher than v1's there, consistent with
  it winning that one cell.
- **Most requests get no tool reuse at all.** Every request after the first
  hits the template, so counting "requests that hit the cache" gives 100% for
  every policy and means nothing. Counting only tool reuse, at 64 tools:

  | policy | requests with any tool reuse |
  |---|---|
  | ToolTrie-v1 | 64 of 199 (32%) |
  | ContextPilot | 50 (25%) |
  | no reordering | 17 (9%) |

  With the list cut to 10, v1 reaches 76 (38%). At 64 tools, the other
  two-thirds of requests share no reusable prefix with anything still in the
  cache.

---

## 3. Why the misses happen, and the most any ordering could reach

Every prompt token is either a hit or a miss. The misses split into four
causes, each measured by asking what a better ordering could have done:

| cause | what it means |
|---|---|
| **this request's own order** | it could have reused more by reordering only itself, given what was in the cache |
| **coordination** | it needed an *earlier* request to have been ordered with it in mind. This is an optimistic upper bound: it lets every request pick its best earlier partner, ignoring that one order must serve many requests |
| **cache size** | the reusable prefix had already been evicted |
| **unavoidable** | no ordering could reuse it: tools never seen together earlier, the user's question, the end of the template |

At 64 tools, share of all prompt tokens:

| | ToolTrie-v1 | ContextPilot | no reordering |
|---|---|---|---|
| **hit** | **6.22** | 2.98 | 0.82 |
| lost to this request's own order | **0.00** | 1.68 | 4.68 |
| lost to lack of coordination (at most) | 13.93 | 15.33 | 14.49 |
| lost to cache size (at most) | 8.30 | 8.47 | 8.47 |
| **unavoidable** (at least) | **71.54** | **71.54** | **71.54** |

**Four findings:**

1. **v1 already picks the best order for every request, given what is in the
   cache.** Its own-order loss is 0.00 at 4, 16 and 64 tools, and 0.01 at 128.
   For each request, its trie finds the longest reusable prefix the cache
   actually holds; no reordering of that request would have reused more.
   ContextPilot leaves 1.7 points on the table, and no reordering 4.7. The
   remaining reuse needs coordination: ordering requests with the requests
   that come later in mind.
2. **At 64 tools, at least 71.5% of all prompt tokens can never be reused by
   any ordering.** So even a perfect, all-knowing ordering could serve at most
   about 20% of prompt tokens from this cache, or 28.5% from an unlimited one.
   v1 reaches 6.22%: about a third of what is possible in this cache.
3. **This explains why ordering's latency effects on retrieved menus are
   modest.** On whole menus, even in principle, no ordering can serve more
   than about 16-25% of prompt tokens from this cache, because most of each
   prompt is new to it.
4. **The ceiling stays around 16-25% at every menu size, but v1 gets more of it
   when the list is short or cut:**

   | | 4 tools | 16 tools | 64 tools | 128 tools | 64 → 10 |
   |---|---|---|---|---|---|
   | most any ordering could reach in this cache | 23.8% | 25.3% | 20.2% | 16.2% | 32.0%* |
   | ToolTrie-v1's hit rate | 18.0% | 13.2% | 6.2% | 3.7% | 27.6% |
   | share of the ceiling v1 gets | 76% | 52% | 31% | 23% | 86% |

   \*For "64 → 10" each policy shows different tools, so each has its own
   ceiling. 32.0% is the ceiling for the tools v1 chose to show.
   ContextPilot's shown tools have a higher ceiling, 35.5%, but it reaches
   only 20.2% (57%).

---

## 4. The size of the cache matters, and explains part of v1's lead

Bigger models leave less GPU memory for the cache, so fewer earlier prompts
stay available for reuse. Hit rate at 64 tools, with the cache each model
actually had:

| model | cache holds | ToolTrie-v1 | ContextPilot | no reordering | v1 ÷ ContextPilot |
|---|---|---|---|---|---|
| 0.6B | 189,712 tokens (~24 prompts) | 6.22% | 2.98% | 0.82% | 2.09x |
| 4B | 97,904 tokens (~12 prompts) | 4.35% | 2.10% | 0.72% | 2.07x |
| 8B | 36,704 tokens (~5 prompts) | 2.41% | 1.42% | 0.64% | 1.70x |
| unlimited (simulated) | everything | 7.31% | 5.10% | 1.07% | 1.43x |

At 128 tools the pattern repeats:

| | 0.6B | 4B | 8B | unlimited |
|---|---|---|---|---|
| ToolTrie-v1 | 3.73% | 2.19% | 1.09% | 3.91% |
| ContextPilot | 1.21% | 1.02% | 0.65% | 2.96% |
| v1 ÷ ContextPilot | 3.08x | 2.15x | 1.68x | 1.32x |

- **v1 beats ContextPilot at every cache size.**
- **Part of v1's lead is cache locality.** v1 matches the previous requests'
  prefixes, so its reuse comes from recent requests that are still in the
  cache. ContextPilot's clustering often reuses prefixes from older requests,
  which a finite cache has already evicted. At 0.6B and 64 tools, eviction
  costs ContextPilot 2.1 points (it would get 5.10% with an unlimited cache,
  not 2.98%), but costs v1 only 1.1.
- **With an unlimited cache the lead would be 1.4x at 64 tools and 1.3x at 128,
  not 2.1x and 3.1x.** Real caches are always finite, and smaller than this
  for bigger models, so the measured lead is the one a deployment sees.
- **The honest description is therefore:** v1 finds somewhat more reuse
  (1.3-1.4x with an unlimited cache), and it finds reuse the cache can
  actually serve.

---

## What this adds to the conclusions

- **Report tool reuse, not raw reuse, when comparing policies on short
  prompts.** At 4 tools almost half of every policy's hit rate is the fixed
  template.
- **v1 is already the best possible request by request.** Given what is in the
  cache, it leaves no reuse on the table for any request. Further gains need
  one of three things:
  - **coordination across requests.** Batching or scheduling requests with
    shared tools together is what ContextPilot's full system, not tested here,
    adds;
  - **a larger cache**;
  - **fewer tools per prompt.**
- **Retrieved menus cap what ordering can ever do.** On whole menus, at least
  71-76% of every prompt can never be reused under any ordering in these
  workloads; with the list cut to 10 it is 64-76%, depending on which tools
  are shown. That is why the latency effects of ordering are modest.

## Caveats

- Dense retriever only, Qwen3's chat template. The 48-token template figure is
  specific to that template and to 16-token blocks.
- The coordination and cache-size figures are **upper bounds**, built on
  optimistic pairwise matching. The truth lies between the "own order" and
  "coordination" lines.
- Which earlier prompts count as "in the cache" is approximated by the most
  recent requests that fit in its capacity. vLLM actually keeps the beginnings
  of older prompts a little longer than their ends.
- Single-turn only. In multi-turn traffic the conversation history is a large
  shared prefix, so hit rates would be far higher.

## Reproduce

```
conda run -n tatm python scripts/analyze_prefix_cache.py \
    --out-dir cluster/results/prefix-cache-accounting-<timestamp>
```

This needs `transformers` and the Qwen3 tokenizer, both in the `tatm`
environment. CPU only, a few minutes on 48 workers. The results in this
document are in `cluster/results/prefix-cache-accounting-20260911-222207/`.
The simulator's own tests are in `tests/test_prefix_cache_sim.py`.
