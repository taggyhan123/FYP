# Cache hit and miss rate, with F1, across evaluation settings

How much of each prompt vLLM served from its prefix cache (hit) or had to
compute (miss), under each tool-ordering policy, and what the same requests
scored on F1. Every number below comes from the same replay: hit/miss and F1
are never taken from different runs.

**Data.** 96 single-turn replays of 200 ToolRet tasks each, run one
request at a time on unmodified vLLM 0.26.0 with automatic prefix caching, the
cache reset before every run. Menus come from the full 44,453-tool ToolRet
catalogue; only the number of tools retrieved and shown changes between
settings. Source table: `cluster/results/hit-miss-f1-20260916-220748/summary.json`, built by `scripts/summarize_hit_miss_f1.py`.

## Summary

- **ToolTrie-v1 has the highest hit rate in 13 of 17 settings.**
  ContextPilot is higher in the other four: 4-tool menus (18.16% vs 17.97%, both
  model sizes) and BM25 retrieve-64 / show-10 (22.78% vs 21.75% at 4B,
  23.13% vs 21.75% at 0.6B).
- **Miss rates are high in every setting.** Even the best policy computes
  72-86% of prompt tokens when 10 tools are shown, and 94-99% when 64 or
  128 are shown. Ordering changes which few percent are cached; it does not
  make prompts mostly cached.
- **ToolTrie-v1 is never significantly worse on end-to-end F1 than any rival.**
  In 48 paired comparisons (same tasks, v1 minus rival):
  - vs **ContextPilot**: v1 significantly higher in **7 of 17**
    settings (by 2.3-7.9 points), lower in 0;
  - vs **frequency**: v1 significantly higher in **2 of 14**
    (0.6B at 64 and 128 tools), lower in 0;
  - vs **no reordering**: **0 of 17** either way. **On F1, v1 and no
    reordering are tied everywhere.**
- **So the policies split three ways.** No reordering keeps F1 and gets the
  lowest hit rate. ContextPilot and frequency raise the hit rate but, in some
  settings, pay for it in F1. ToolTrie-v1 raises the hit rate, usually the most,
  without a measurable F1 cost.
- **Where ContextPilot out-caches v1 (BM25, show 10), it loses 5.4-5.8 F1
  points.** Its extra hits come from moving shared tools into the 10 slots,
  which pushes the right tool out: it is in the menu for 39.5% of requests,
  against 59.5% for v1.

## 1. Headline, Qwen3-4B

| setting (Qwen3-4B) | hit rate: v1 / CP / freq / none | miss rate, v1 | end-to-end F1: v1 / CP / freq / none | highest hit | F1 verdict |
|---|---|---|---|---|---|
| Retrieve 64, show 10 — dense | 27.6% / 20.0% / 9.0% / 6.2% | 72.4% | 26.0 / 21.4 / 24.4 / 26.7 | ToolTrie-v1 | v1 ahead of ContextPilot |
| Retrieve 64, show 10 — BM25 | 21.8% / 22.8% / 7.3% / 8.4% | 78.2% | 28.1 / 22.3 / 27.9 / 28.9 | ContextPilot | v1 ahead of ContextPilot |
| Retrieve 10, show 10 — dense | 13.7% / 12.4% / 8.6% / 6.2% | 86.3% | 27.1 / 26.5 / 27.8 / 26.7 | ToolTrie-v1 | no significant difference |
| 4 tools, all shown | 18.0% / 18.2% / 15.8% / 14.3% | 82.0% | 26.2 / 25.8 / 26.0 / 26.4 | ContextPilot | no significant difference |
| 16 tools, all shown | 12.5% / 11.1% / 5.9% / 3.9% | 87.5% | 27.2 / 24.6 / 27.2 / 27.0 | ToolTrie-v1 | v1 ahead of ContextPilot |
| 64 tools, all shown | 4.3% / 2.1% / 1.1% / 0.7% | 95.7% | 28.0 / 26.4 / 29.8 / 28.0 | ToolTrie-v1 | no significant difference |
| 128 tools, all shown | 2.2% / 1.0% / 0.4% / 0.3% | 97.8% | 29.8 / 26.6 / 29.2 / 29.6 | ToolTrie-v1 | no significant difference |

"Highest hit" is the policy with the largest hit rate. "F1 verdict" names the
rivals v1 beats on end-to-end F1 by at least two standard errors on the
paired test (§6); no rival beats v1 anywhere.

## 2. Definitions

| term | meaning |
|---|---|
| **hit** | prompt tokens served from the prefix cache ÷ all prompt tokens, summed over the 200 requests (= vLLM's `prompt_tokens_cached` share) |
| **miss** | prompt tokens computed ÷ all prompt tokens; hit + miss = 100% |
| **hit inside the tool list** | the hit minus the first 48 tokens of each request. Those tokens are the chat-template header every prompt starts with, so they are cached under **any** ordering; only hits beyond them are the ordering's doing |
| **requests hitting the tool list** | requests whose hit reaches past the header, out of 200 |
| **right tool in menu** | share of requests whose shown menu contains at least one gold tool (the *ceiling*) |
| **F1** | per request, 2PR/(P+R) over the tools called vs the gold tools, averaged over requests whose menu contains a gold tool. Comparable across policies only when their ceilings match |
| **end-to-end F1** | ceiling × F1: a request with no gold tool in its menu scores 0. **This is the primary accuracy metric**, and the only one that is fair when policies show different tools |
| **rival − v1** | paired difference in end-to-end F1, in points. **▼** marks a rival at least two standard errors below v1 |

All F1 values are ×100.

## 3. Retrieve 64, show 10: the ordering chooses which tools the model sees

Every policy reorders the same 64 retrieved tools and the model is shown the
first 10, so ordering changes both what is cached and whether the right tool
survives the cut. This is the design closest to routed tool-selection systems.

### Dense retriever

#### Qwen3-4B

| policy | hit | miss | hit inside the tool list | requests hitting the tool list | right tool in menu | F1 | **end-to-end F1** | rival − v1 |
|---|---|---|---|---|---|---|---|---|
| **ToolTrie-v1** | 27.64% | 72.36% | 24.06% | 76 / 200 | 61.5% | 42.3 | **26.0** |  |
| ContextPilot | 19.96% | 80.04% | 16.40% | 67 / 200 | 47.0% | 45.6 | **21.4** | -4.6 ▼ |
| frequency | 9.00% | 91.00% | 5.77% | 53 / 200 | 38.5% | 63.3 | **24.4** | -1.7 |
| no reordering | 6.22% | 93.78% | 2.65% | 33 / 200 | 62.0% | 43.1 | **26.7** | +0.7 |

#### Qwen3-0.6B

| policy | hit | miss | hit inside the tool list | requests hitting the tool list | right tool in menu | F1 | **end-to-end F1** | rival − v1 |
|---|---|---|---|---|---|---|---|---|
| **ToolTrie-v1** | 27.64% | 72.36% | 24.06% | 76 / 200 | 61.5% | 32.4 | **19.9** |  |
| ContextPilot | 20.16% | 79.84% | 16.60% | 69 / 200 | 47.0% | 34.6 | **16.2** | -3.7 |
| frequency | 9.39% | 90.61% | 6.16% | 58 / 200 | 38.5% | 52.2 | **20.1** | +0.2 |
| no reordering | 6.38% | 93.62% | 2.80% | 35 / 200 | 62.0% | 32.0 | **19.8** | -0.1 |

- **v1 caches the most: 27.64% hit, 4.4x no reordering and 1.4x ContextPilot.**
  Its hits reach into the tool list for 76 of 200 requests, against 33 with no
  reordering.
- **v1 keeps the right tool in view as often as no reordering** (61.5% vs
  62.0%). ContextPilot keeps it for 47.0% and frequency for 38.5%.
- **End-to-end F1: v1 ties no reordering and frequency, and beats ContextPilot
  by 4.6 points at 4B** (significant) and 3.7 at 0.6B (not significant).
- **Frequency's plain F1 is the highest in the table (63.3) and misleading.** It
  is averaged only over the 38.5% of requests where the right tool survived,
  which are the easy ones. End-to-end, frequency is not ahead.

### BM25 retriever

#### Qwen3-4B

| policy | hit | miss | hit inside the tool list | requests hitting the tool list | right tool in menu | F1 | **end-to-end F1** | rival − v1 |
|---|---|---|---|---|---|---|---|---|
| **ToolTrie-v1** | 21.75% | 78.25% | 18.35% | 79 / 200 | 59.5% | 47.2 | **28.1** |  |
| ContextPilot | 22.78% | 77.22% | 19.46% | 88 / 200 | 39.5% | 56.5 | **22.3** | -5.8 ▼ |
| frequency | 7.31% | 92.69% | 4.00% | 60 / 200 | 41.5% | 67.3 | **27.9** | -0.2 |
| no reordering | 8.38% | 91.62% | 4.89% | 38 / 200 | 59.0% | 49.0 | **28.9** | +0.8 |

#### Qwen3-0.6B

| policy | hit | miss | hit inside the tool list | requests hitting the tool list | right tool in menu | F1 | **end-to-end F1** | rival − v1 |
|---|---|---|---|---|---|---|---|---|
| **ToolTrie-v1** | 21.75% | 78.25% | 18.35% | 79 / 200 | 59.5% | 37.0 | **22.0** |  |
| ContextPilot | 23.13% | 76.87% | 19.81% | 92 / 200 | 39.5% | 42.0 | **16.6** | -5.4 ▼ |
| frequency | 7.86% | 92.14% | 4.55% | 66 / 200 | 41.5% | 54.6 | **22.7** | +0.7 |
| no reordering | 8.76% | 91.24% | 5.28% | 39 / 200 | 59.0% | 39.0 | **23.0** | +1.0 |

- **This is one of the two settings where ContextPilot has the higher hit
  rate**: 22.78% vs 21.75% at 4B, with more requests hitting the tool list
  (88 vs 79).
- **It is paid for in accuracy.** ContextPilot shows the right tool for 39.5% of
  requests against v1's 59.5%, and its end-to-end F1 is **5.8 points lower at
  4B and 5.4 lower at 0.6B**, both significant.
- v1, frequency and no reordering are within 1 point of each other on
  end-to-end F1.

## 4. Retrieve 10, show 10: ordering can only move tools within the menu

All policies show the same 10 tools, so the right tool is in the menu equally
often (62.0%) and ordering only changes positions.

#### Qwen3-4B

| policy | hit | miss | hit inside the tool list | requests hitting the tool list | right tool in menu | F1 | **end-to-end F1** | rival − v1 |
|---|---|---|---|---|---|---|---|---|
| **ToolTrie-v1** | 13.69% | 86.31% | 10.11% | 63 / 200 | 62.0% | 43.7 | **27.1** |  |
| ContextPilot | 12.38% | 87.62% | 8.81% | 58 / 200 | 62.0% | 42.8 | **26.5** | -0.5 |
| frequency | 8.60% | 91.40% | 5.02% | 40 / 200 | 62.0% | 44.8 | **27.8** | +0.7 |
| no reordering | 6.22% | 93.78% | 2.65% | 33 / 200 | 62.0% | 43.1 | **26.7** | -0.3 |

#### Qwen3-0.6B

| policy | hit | miss | hit inside the tool list | requests hitting the tool list | right tool in menu | F1 | **end-to-end F1** | rival − v1 |
|---|---|---|---|---|---|---|---|---|
| **ToolTrie-v1** | 13.71% | 86.29% | 10.14% | 63 / 200 | 62.0% | 31.6 | **19.6** |  |
| ContextPilot | 12.46% | 87.54% | 8.88% | 58 / 200 | 62.0% | 32.4 | **20.1** | +0.5 |
| frequency | 8.68% | 91.32% | 5.10% | 41 / 200 | 62.0% | 30.9 | **19.2** | -0.4 |
| no reordering | 6.38% | 93.62% | 2.80% | 35 / 200 | 62.0% | 32.0 | **19.8** | +0.2 |

- **v1 has the highest hit rate (13.7%)**, narrowly ahead of ContextPilot
  (12.4%) and about double no reordering (6.2%).
- **F1 is flat**: every policy is within 1.3 points, and no difference is
  significant. With the same 10 tools shown, order within the menu barely
  changes what the model calls.

## 5. Whole menus: every retrieved tool is shown

The right tool is in the menu equally often under every policy (52.5% at 4
tools, 67.5% at 16, 81.5% at 64, 85.5% at 128), so plain F1 and end-to-end F1
rank the policies identically.

### 4 tools

#### Qwen3-4B

| policy | hit | miss | hit inside the tool list | requests hitting the tool list | right tool in menu | F1 | **end-to-end F1** | rival − v1 |
|---|---|---|---|---|---|---|---|---|
| **ToolTrie-v1** | 17.97% | 82.03% | 9.93% | 49 / 200 | 52.5% | 49.8 | **26.2** |  |
| ContextPilot | 18.16% | 81.84% | 10.12% | 46 / 200 | 52.5% | 49.2 | **25.8** | -0.3 |
| frequency | 15.84% | 84.16% | 7.80% | 41 / 200 | 52.5% | 49.6 | **26.0** | -0.1 |
| no reordering | 14.35% | 85.65% | 6.30% | 35 / 200 | 52.5% | 50.2 | **26.4** | +0.2 |

#### Qwen3-0.6B

| policy | hit | miss | hit inside the tool list | requests hitting the tool list | right tool in menu | F1 | **end-to-end F1** | rival − v1 |
|---|---|---|---|---|---|---|---|---|
| **ToolTrie-v1** | 17.97% | 82.03% | 9.93% | 49 / 200 | 52.5% | 39.1 | **20.5** |  |
| ContextPilot | 18.16% | 81.84% | 10.12% | 46 / 200 | 52.5% | 38.4 | **20.2** | -0.3 |
| frequency | 15.84% | 84.16% | 7.80% | 41 / 200 | 52.5% | 39.5 | **20.8** | +0.2 |
| no reordering | 14.35% | 85.65% | 6.30% | 35 / 200 | 52.5% | 39.7 | **20.8** | +0.3 |

### 16 tools

#### Qwen3-4B

| policy | hit | miss | hit inside the tool list | requests hitting the tool list | right tool in menu | F1 | **end-to-end F1** | rival − v1 |
|---|---|---|---|---|---|---|---|---|
| **ToolTrie-v1** | 12.53% | 87.47% | 10.23% | 66 / 200 | 67.5% | 40.2 | **27.2** |  |
| ContextPilot | 11.12% | 88.88% | 8.81% | 62 / 200 | 67.5% | 36.4 | **24.6** | -2.6 ▼ |
| frequency | 5.87% | 94.13% | 3.57% | 40 / 200 | 67.5% | 40.3 | **27.2** | +0.0 |
| no reordering | 3.89% | 96.11% | 1.58% | 29 / 200 | 67.5% | 40.1 | **27.0** | -0.1 |

#### Qwen3-0.6B

| policy | hit | miss | hit inside the tool list | requests hitting the tool list | right tool in menu | F1 | **end-to-end F1** | rival − v1 |
|---|---|---|---|---|---|---|---|---|
| **ToolTrie-v1** | 13.18% | 86.82% | 10.87% | 70 / 200 | 67.5% | 33.2 | **22.4** |  |
| ContextPilot | 11.58% | 88.42% | 9.28% | 69 / 200 | 67.5% | 29.9 | **20.2** | -2.2 ▼ |
| frequency | 6.29% | 93.71% | 3.98% | 44 / 200 | 67.5% | 28.5 | **19.2** | -3.2 |
| no reordering | 4.11% | 95.89% | 1.81% | 35 / 200 | 67.5% | 32.7 | **22.1** | -0.3 |

#### Qwen3-8B (no frequency run)

| policy | hit | miss | hit inside the tool list | requests hitting the tool list | right tool in menu | F1 | **end-to-end F1** | rival − v1 |
|---|---|---|---|---|---|---|---|---|
| **ToolTrie-v1** | 8.15% | 91.85% | 5.84% | 45 / 200 | 67.5% | 43.9 | **29.6** |  |
| ContextPilot | 7.53% | 92.47% | 5.22% | 37 / 200 | 67.5% | 41.6 | **28.1** | -1.5 |
| no reordering | 3.05% | 96.95% | 0.75% | 14 / 200 | 67.5% | 46.1 | **31.1** | +1.5 |

### 64 tools

#### Qwen3-4B

| policy | hit | miss | hit inside the tool list | requests hitting the tool list | right tool in menu | F1 | **end-to-end F1** | rival − v1 |
|---|---|---|---|---|---|---|---|---|
| **ToolTrie-v1** | 4.35% | 95.65% | 3.75% | 51 / 200 | 81.5% | 34.3 | **28.0** |  |
| ContextPilot | 2.10% | 97.90% | 1.49% | 34 / 200 | 81.5% | 32.4 | **26.4** | -1.6 |
| frequency | 1.06% | 98.94% | 0.45% | 25 / 200 | 81.5% | 36.6 | **29.8** | +1.9 |
| no reordering | 0.72% | 99.28% | 0.11% | 9 / 200 | 81.5% | 34.3 | **28.0** | -0.0 |

#### Qwen3-0.6B

| policy | hit | miss | hit inside the tool list | requests hitting the tool list | right tool in menu | F1 | **end-to-end F1** | rival − v1 |
|---|---|---|---|---|---|---|---|---|
| **ToolTrie-v1** | 6.22% | 93.78% | 5.62% | 64 / 200 | 81.5% | 27.0 | **22.0** |  |
| ContextPilot | 2.98% | 97.02% | 2.38% | 50 / 200 | 81.5% | 21.5 | **17.5** | -4.5 ▼ |
| frequency | 1.39% | 98.61% | 0.79% | 37 / 200 | 81.5% | 19.8 | **16.2** | -5.8 ▼ |
| no reordering | 0.82% | 99.18% | 0.22% | 17 / 200 | 81.5% | 25.7 | **20.9** | -1.1 |

#### Qwen3-8B (no frequency run)

| policy | hit | miss | hit inside the tool list | requests hitting the tool list | right tool in menu | F1 | **end-to-end F1** | rival − v1 |
|---|---|---|---|---|---|---|---|---|
| **ToolTrie-v1** | 2.41% | 97.59% | 1.81% | 29 / 200 | 81.5% | 35.3 | **28.8** |  |
| ContextPilot | 1.42% | 98.58% | 0.82% | 20 / 200 | 81.5% | 35.8 | **29.1** | +0.4 |
| no reordering | 0.64% | 99.36% | 0.04% | 5 / 200 | 81.5% | 35.6 | **29.0** | +0.2 |

### 128 tools

#### Qwen3-4B

| policy | hit | miss | hit inside the tool list | requests hitting the tool list | right tool in menu | F1 | **end-to-end F1** | rival − v1 |
|---|---|---|---|---|---|---|---|---|
| **ToolTrie-v1** | 2.19% | 97.81% | 1.88% | 41 / 200 | 85.5% | 34.9 | **29.8** |  |
| ContextPilot | 1.02% | 98.98% | 0.71% | 32 / 200 | 85.5% | 31.1 | **26.6** | -3.2 |
| frequency | 0.40% | 99.60% | 0.09% | 16 / 200 | 85.5% | 34.1 | **29.2** | -0.6 |
| no reordering | 0.34% | 99.66% | 0.03% | 6 / 200 | 85.5% | 34.6 | **29.6** | -0.2 |

#### Qwen3-0.6B

| policy | hit | miss | hit inside the tool list | requests hitting the tool list | right tool in menu | F1 | **end-to-end F1** | rival − v1 |
|---|---|---|---|---|---|---|---|---|
| **ToolTrie-v1** | 3.73% | 96.27% | 3.43% | 55 / 200 | 85.5% | 25.4 | **21.7** |  |
| ContextPilot | 1.21% | 98.79% | 0.90% | 42 / 200 | 85.5% | 16.2 | **13.8** | -7.9 ▼ |
| frequency | 0.47% | 99.53% | 0.17% | 29 / 200 | 85.5% | 17.1 | **14.6** | -7.1 ▼ |
| no reordering | 0.36% | 99.64% | 0.06% | 9 / 200 | 85.5% | 26.0 | **22.2** | +0.6 |

#### Qwen3-8B (no frequency run)

| policy | hit | miss | hit inside the tool list | requests hitting the tool list | right tool in menu | F1 | **end-to-end F1** | rival − v1 |
|---|---|---|---|---|---|---|---|---|
| **ToolTrie-v1** | 1.09% | 98.91% | 0.79% | 22 / 200 | 85.5% | 35.1 | **30.0** |  |
| ContextPilot | 0.65% | 99.35% | 0.35% | 19 / 200 | 85.5% | 35.3 | **30.2** | +0.1 |
| no reordering | 0.31% | 99.69% | 0.01% | 2 / 200 | 85.5% | 34.6 | **29.6** | -0.4 |

- **Hit rate falls as the menu grows.** At 4 tools the prompt is short, so the
  shared 48-token header alone is a large share of it: 8 of v1's 18
  hit points are header. At 64 and 128 tools every policy misses 94-99.7% of
  tokens.
- **ContextPilot is ahead only at 4 tools**, by 0.2 points. v1 leads at 16, 64
  and 128 tools, at every model size.
- **Hit rate also falls with model size, for a reason unrelated to the
  model.** The prompts are identical token for token across 0.6B, 4B and 8B.
  A bigger model leaves less GPU memory for the cache, which holds about 24
  prompts at 0.6B, 12 at 4B and 5 at 8B (`cache-hit-miss.md`), so fewer
  earlier prompts are still there to reuse.
- **F1 at 4B and 8B barely moves with ordering.** Every policy is within
  3.2 points of v1; the only significant gap is ContextPilot at 16 tools
  (−2.6 at 4B). At 4B and 64 tools frequency has the highest end-to-end F1
  (29.8 vs v1's 28.0), but the gap is within noise.
- **At 0.6B, ContextPilot and frequency cost F1 on long menus.** At 64 tools
  they are 4.5 and 5.8 points below v1, at 128 tools 7.9 and 7.1, all
  significant. The small model misses tools placed deep in a long list;
  v1 leaves the retriever's top picks near the top, as no reordering does.

## 6. How the comparisons were tested

For each setting and model, the end-to-end F1 contribution of every task was
paired between v1 and a rival (same task, same menu size, same model). The
table columns report the mean difference; **▼** means it is at least two
standard errors (z ≥ 2). Across the 48 comparisons, about two
would reach |z| ≥ 2 by chance alone, so a single ▼ near z = 2 is weak
evidence. The robust results are the pattern ones: **no rival is ever
significantly above v1**, and ContextPilot's losses repeat across settings,
retrievers and model sizes.

Per-request recomputation of end-to-end F1 was checked against
`scripts/score_end_to_end.py` for every replay; they agree to four decimals.

## 7. Settings not covered by F1 here

- **Multi-turn coding sessions.** Hit rates are in `multi-turn-sessions.md`:
  with tools that stay fixed through the session, no reordering, ContextPilot
  and v1 all hit 92.95-92.97%; with tools re-retrieved and placed at the front
  on every turn they collapse to 7.9 / 9.0 / 11.0% (re-measured with all three
  on one GPU). Those replays generate one token to time prefill, so they have
  no tool calls and no F1.
- **Load.** Hit rate does not depend on request rate or concurrency, because it
  depends only on the request order and cache contents. On the single-GPU 4B
  rate sweep v1's hit rate was 4.33-4.34% at all six rates, and the concurrency
  sweep reproduces 4.33 / 2.10 / 1.06 / 0.72% at N = 1. Latency does depend on
  load; see the frontier figures.
- **Padded menus** (almost all tools shared by every request), where ContextPilot
  is known to lead, have no scored ToolTrie-v1 or frequency accuracy runs.

## 8. Limits

- **One run per policy and setting.** The paired test accounts for per-task
  noise, not run-to-run variation in generation.
- **200 tasks from 2 of ToolRet's 35 sources** (`apibank`, `apigen`). Cache hit
  rates depend on how much requests overlap, which may differ on other sources.
- **The 48-token header split** is specific to the Qwen3 chat template
  and vLLM's 16-token cache blocks.
- **8B has no frequency run.**

## Reproduce

```bash
<tatm venv>/python scripts/summarize_hit_miss_f1.py \
    --output cluster/results/hit-miss-f1-<timestamp>/summary.json
```
