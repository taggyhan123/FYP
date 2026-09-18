# Cache hit and miss rate, with F1

How much of each prompt vLLM served from its prefix cache (hit) or had to
compute (miss) under each tool-ordering policy, and the F1 those same requests
scored. Hit/miss and F1 always come from the same replay.

**Data.** Single-turn replays of 200 ToolRet tasks, one request at a time, on
unmodified vLLM 0.26.0 with automatic prefix caching, cache reset before each
run. Tools are retrieved from the full 44,453-tool ToolRet catalogue. The full
table (96 replays, every menu size and model) is
`cluster/results/hit-miss-f1-20260916-220748/summary.json`, built by
`scripts/summarize_hit_miss_f1.py`.

## Summary

- **ToolTrie-v1 has the highest hit rate in 13 of 17 settings.** ContextPilot
  is higher only on 4-tool menus (by 0.2 points) and on BM25 retrieve-64 /
  show-10.
- **Miss rates stay high.** The best policy still computes 72-86% of prompt
  tokens when 10 tools are shown, and 94-99% when 64 or 128 are shown.
- **ToolTrie-v1 is never significantly worse on end-to-end F1 than any
  rival.** It is significantly better than ContextPilot in 7 of 17 settings and
  than frequency in 2 of 14, and **tied with no reordering in all 17**.
- **So v1 raises the hit rate without costing F1.** ContextPilot and frequency
  raise it too, but in some settings pay for it in F1.

## Where ToolTrie-v1 wins

| where | what v1 wins | by how much |
|---|---|---|
| **Retrieve 64, show 10, dense** (realistic setting, 4B) | highest hit rate | **27.6%** vs ContextPilot 20.0%, frequency 9.0%, no reordering 6.2% (1.4x / 3.1x / 4.4x) |
| | keeps the right tool in the 10 shown | **61.5%** vs ContextPilot 47.0%, frequency 38.5% (ties no reordering, 62.0%) |
| | end-to-end F1 | **+4.6 points over ContextPilot** (significant) |
| **Retrieve 10, show all 10, dense** (the prior-work setup, 4B) | highest hit rate | **13.7%** vs ContextPilot 12.4%, frequency 8.6%, no reordering 6.2% (1.1x / 1.6x / 2.2x), at unchanged F1 |
| **Retrieve 64, show 10, BM25** (4B and 0.6B) | end-to-end F1 | **+5.8 and +5.4 points over ContextPilot** (both significant), because v1 keeps the right tool in view for 59.5% of requests vs 39.5% |
| **64 and 128 tools shown** (4B) | highest hit rate | **2.1x ContextPilot** (4.35% vs 2.10%; 2.19% vs 1.02%), **4-6x frequency and no reordering** |
| **64 and 128 tools shown** (0.6B) | end-to-end F1 | **+4.5 and +7.9 points over ContextPilot, +5.8 and +7.1 over frequency** (all significant) |
| **16 tools shown** (4B and 0.6B) | end-to-end F1 | **+2.6 and +2.2 points over ContextPilot** (both significant) |
| **Qwen3-8B, 16 / 64 / 128 tools** | highest hit rate | 8.15 / 2.41 / 1.09% vs ContextPilot 7.53 / 1.42 / 0.65% |
| **Serving under load** (4B, from the frontier sweeps) | latency and throughput | fastest policy at all six offered rates on 64-tool menus (1.5x faster than no reordering at 0.8 req/s); on retrieve 64 / show 10 with 32 requests in flight, **+8% throughput over ContextPilot and +23% over no reordering** |

**Where v1 does not win:** ContextPilot caches slightly more on 4-tool menus
(by 0.2 points) and on BM25 retrieve-64 / show-10 (22.8% vs 21.8%, while losing
F1), and v1 ties no reordering on F1 everywhere.

## Retrieval: dense vs sparse, and why dense is the headline

Each request starts with a retrieval step that picks 64 candidate tools out of
the 44,453 in ToolRet. **The retriever is not part of ToolTrie or ContextPilot**:
it runs first and gives every policy the same 64 tools. The policies only
reorder them, and in the show-10 design the first 10 after reordering are what
the model sees.

Two retrievers are used. Both read the same parts of each tool (name,
description, parameter names and values); they differ only in how they match.

| | **dense** (`BAAI/bge-small-en-v1.5`) | **sparse** (BM25) |
|---|---|---|
| how it matches | turns the query and each tool into 384-number embeddings and ranks tools by cosine similarity: **similar meaning** matches, even with different words | scores tools by **shared words**, weighting rare words more and repeated words less (k1 = 1.5, b = 0.75); no shared word, no match |
| right tool among the 64 retrieved | **81.5%** | 75.5% |
| right tool in the retriever's top 10 | **62.0%** | 59.0% |
| code | `src/tatm/dense_retrieval.py` | `src/tatm/retrieval.py` |

- **Dense is the realistic setting.** Tool routers mostly match by embedding
  similarity, because users describe what they want in their own words rather
  than in the tool's vocabulary. It also finds the right tool more often here,
  and it is the retriever type ContextPilot's own paper evaluates with.
- **BM25 is kept as a robustness check, and it matters.** The two retrievers
  share only 17 of their 64 tools per request on average (range 1-47), so the
  policies reorder very different candidate lists. That is why the cache winner
  flips under BM25 while v1's accuracy advantage over ContextPilot holds
  under both.
- **Not tested:** hybrid retrieval (keyword and embedding scores combined, common
  in production) and larger embedding models (bge-small is small; ContextPilot's
  paper used a 7B embedding model). Either would change which 64 tools come
  back.

## 1. Retrieve 64, show 10 (dense): the realistic setting

Every policy reorders the same 64 retrieved tools and the model sees the first
10, so ordering decides both what is cached and whether the right tool
survives the cut. This is how routed tool-selection systems work. Qwen3-4B:

| policy | hit | miss | right tool in the 10 shown | end-to-end F1 | vs v1 |
|---|---|---|---|---|---|
| **ToolTrie-v1** | **27.6%** | **72.4%** | 61.5% | 26.0 | |
| ContextPilot | 20.0% | 80.0% | 47.0% | 21.4 | **−4.6 ▼** |
| frequency | 9.0% | 91.0% | 38.5% | 24.4 | −1.7 |
| no reordering | 6.2% | 93.8% | 62.0% | 26.7 | +0.7 |

- **v1 caches the most**: 4.4x no reordering and 1.4x ContextPilot.
- **It keeps the right tool in view as often as no reordering** (61.5% vs
  62.0%). ContextPilot and frequency push it out of the 10 for about 1 request
  in 7 and 1 in 4.
- **End-to-end F1: v1 ties no reordering and frequency, and beats ContextPilot
  by 4.6 points.**

## 2. Retrieve 10, show all 10 (dense): the setup prior papers use

The retriever returns only its top 10 and all 10 are shown, so every policy
gives the model the same tools and ordering can only change their order. This
matches how prompt-reordering papers such as ContextPilot evaluate (retrieve k,
show all k). Qwen3-4B:

| policy | hit | miss | requests hitting the tool list | right tool in the 10 shown | end-to-end F1 | vs v1 |
|---|---|---|---|---|---|---|
| **ToolTrie-v1** | **13.7%** | **86.3%** | **63 / 200** | 62.0% | 27.1 | |
| ContextPilot | 12.4% | 87.6% | 58 / 200 | 62.0% | 26.5 | −0.5 |
| frequency | 8.6% | 91.4% | 40 / 200 | 62.0% | 27.8 | +0.7 |
| no reordering | 6.2% | 93.8% | 33 / 200 | 62.0% | 26.7 | −0.3 |

- **v1 has the highest hit rate**: 2.2x no reordering, 1.6x frequency and
  1.1x ContextPilot, with more requests reusing part of the tool list (63 vs
  58, 40 and 33).
- **F1 is unchanged by ordering**: every policy shows the same 10 tools, and no
  difference is significant.
- **The margin over ContextPilot is small here.** With only 10 tools both find
  most of the same shared prefix; v1's lead grows when the pool is larger
  (§1: 1.4x) or the whole menu is long (§3: 2.1x). At light load (0.6B,
  10 req/s) the policies' time to first token is the same.

## 3. Every retrieved tool shown: hit rate across menu sizes

With the whole menu shown, the right tool is present equally often under every
policy, so ordering can only change where it sits. Qwen3-4B:

| tools shown | hit: v1 / ContextPilot / frequency / none | end-to-end F1: v1 / ContextPilot / frequency / none |
|---|---|---|
| 4 | 18.0 / **18.2** / 15.8 / 14.3% | 26.2 / 25.8 / 26.0 / 26.4 |
| 16 | **12.5** / 11.1 / 5.9 / 3.9% | 27.2 / **24.6 ▼** / 27.2 / 27.0 |
| 64 | **4.3** / 2.1 / 1.1 / 0.7% | 28.0 / 26.4 / 29.8 / 28.0 |
| 128 | **2.2** / 1.0 / 0.4 / 0.3% | 29.8 / 26.6 / 29.2 / 29.6 |

- **Hit rate falls fast as the menu grows**: at 64 and 128 tools every policy
  misses 94-99.7% of tokens. v1 still leads from 16 tools up, narrowly at 16
  and by about 2x over ContextPilot at 64 and 128.
- **At 4B, F1 barely depends on ordering.** The only significant gap is
  ContextPilot at 16 tools.
- **The small 0.6B model is more sensitive.** On long menus ContextPilot and
  frequency lose significantly (4.5-7.9 points at 64 and 128 tools) because
  they move the right tool deep into the list; v1 does not.

## How to read the numbers

- **hit / miss**: prompt tokens served from cache / computed, as a share of all
  prompt tokens (they sum to 100%). Every prompt starts with a 48-token
  template header that is cached under any ordering, so small menus have a
  floor of shared hits.
- **end-to-end F1**: F1 over the tools called vs the gold tools, counting a
  request as 0 when its menu has no gold tool. This is the fair accuracy
  metric when policies show different tools. Values ×100.
- **vs v1**: paired difference in end-to-end F1 over the same tasks (rival
  minus v1). **▼** = at least two standard errors below v1. With 48 such
  comparisons in the full table, about two would cross that line by chance, so
  the dependable result is the pattern: no rival is ever significantly above
  v1.

## Limits

- One run per policy and setting; 200 tasks from 2 of ToolRet's 35 sources.
- Multi-turn coding sessions are not here: their replays generate one token
  and have no F1. Their hit rates (about 93% for every policy with fixed tools)
  are in `multi-turn-sessions.md`.
- Hit rate does not change with load (identical at every request rate and
  concurrency tested); latency does, see the frontier figures.

## Reproduce

```bash
<tatm venv>/python scripts/summarize_hit_miss_f1.py \
    --output cluster/results/hit-miss-f1-<timestamp>/summary.json
```
