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

## 2. Retrieve 64, show 10 (BM25): where ContextPilot out-caches v1

The same design with a keyword retriever. Qwen3-4B:

| policy | hit | miss | right tool in the 10 shown | end-to-end F1 | vs v1 |
|---|---|---|---|---|---|
| **ToolTrie-v1** | 21.8% | 78.2% | 59.5% | 28.1 | |
| ContextPilot | **22.8%** | **77.2%** | 39.5% | 22.3 | **−5.8 ▼** |
| frequency | 7.3% | 92.7% | 41.5% | 27.9 | −0.2 |
| no reordering | 8.4% | 91.6% | 59.0% | 28.9 | +0.8 |

- **ContextPilot's extra hit rate costs accuracy.** It fills the 10 slots with
  widely shared tools, which pushes the right tool out: it is shown for 39.5%
  of requests against v1's 59.5%, and end-to-end F1 falls 5.8 points
  (5.4 at 0.6B). A higher hit rate is not a win if the model sees the wrong
  tools.

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
