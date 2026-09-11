# Key findings — ToolTrie: trie-aware tool ordering for prefix-cached LLM serving

ToolTrie orders the tool definitions in an LLM prompt so that each request
shares as long a prefix as possible with the ones served before it. vLLM can
then reuse the cached computation for that prefix instead of recomputing it.
This page is the project's results summary.

- **The top half is the current result: ToolTrie-v1 against every other
  ordering policy.**
- The bottom half is the August phase (ToolTrie-v0), kept as the record of how
  the project got there.

**Setup.** Qwen3 (0.6B, 4B, 8B) on one RTX 3090, vLLM 0.26.0 with automatic
prefix caching unmodified. Single-turn requests over the ToolRet catalogue
(44,453 tools), 200 tasks per cell. 534 GPU runs in the current phase, 190 in
the August phase.

```
reuse = prompt_tokens_cached / (prompt_tokens_cached + request_prefill_kv_computed_tokens_sum)
```

## Headline: ToolTrie-v1 beats every other ordering on real retrieved tool menus

On the menus a retriever actually produces, ToolTrie-v1 is the best of the six
ordering policies tested:

- **It caches the most.** At 64 tools on a dense retriever it caches 2.1x
  what ContextPilot caches and 7.6x what the unreordered list gets. It is
  first in 7 of 8 menu-size/retriever combinations.
- **It is the fastest under load and serves the most traffic within a
  1-second limit.** Its median latency is about 20% below the traditional
  orderings and 10% below ContextPilot. When requests arrive in bursts it
  sustains 5.8% more traffic than ContextPilot.
- **It keeps the right tool in view.** Take 64 retrieved tools and cut them
  to the 10 the model sees. v1 keeps the right tool among them as often as
  not reordering does. ContextPilot pushes it out in one request in five to
  seven.
- **It costs no accuracy.** At 0.6B and 4B its F1 matches the unreordered
  list and beats ContextPilot in all 8 cells.

**What made the difference was one line.** ToolTrie-v0 sorted every tool its
trie could not match alphabetically, which left ~93% of each menu to an order
that carries no information. ToolTrie-v1 leaves those tools in the order
retrieval returned them.

The six policies compared:

| policy | what it does to the retrieved list |
|---|---|
| no reordering | nothing: keeps the retriever's ranking |
| alphabetical | sorts by tool name |
| frequency | puts tools that appear in the most menus first |
| ToolTrie-v0 | trie-matched tools first, the rest alphabetical |
| ContextPilot | moves the tools that requests share to the front (published state of the art; its ordering only) |
| **ToolTrie-v1** | trie-matched tools first, the rest **left in retrieval order** |

## ToolTrie-v1 against the rest

v1's advantage over each policy, at 64 retrieved tools unless stated.
**Bold** marks a clear win.

| measure | vs no reordering | vs alphabetical | vs frequency | vs ToolTrie-v0 | vs **ContextPilot** |
|---|---|---|---|---|---|
| cache reuse, dense retriever | **7.6x** | **4.1x** | **4.5x** | **2.6x** | **2.1x** |
| cache reuse, BM25 retriever | **5.5x** | **4.1x** | **5.3x** | **2.6x** | 1.04x |
| p50 latency under load, BM25 | **20% lower** | **20% lower** | **21% lower** | **19% lower** | **10% lower** |
| traffic within a 1 s p50 limit, dense | +1.8% | +3.8% | +3.9% | +3.3% | +1.2% |
| same, requests arriving in bursts | **+8.0%** | — | — | — | **+5.8%** |
| right tool survives a cut to 10, dense | tie | **+54pp** | **+23pp** | **+53.5pp** | **+14.5pp** |
| right tool survives a cut to 10, BM25 | tie | **+48pp** | **+18pp** | **+48pp** | **+20pp** |
| end-to-end F1 after the cut | tie | **+19 to +22** | tie | **+19 to +22** | **+3.7 to +5.8** |
| whole-menu F1, cells where v1 is ahead | tie (all within 1.33) | 7 of 8 | 5 of 8 | 7 of 8 | **8 of 8** |

Against the unreordered list, v1 ties on accuracy by design and wins on
everything the cache decides. Against ContextPilot it wins every row. The one
regime where it loses is covered under "Where ToolTrie does not win" below.
The full scoreboard, with every measure, is in
[`concurrent-latency/top-findings.md`](concurrent-latency/top-findings.md).

### It caches the most

Reuse, dense retriever (BM25 in parentheses), %:

| policy | 4 tools | 16 tools | 64 tools | 128 tools |
|---|---|---|---|---|
| no reordering | 14.35 (15.87) | 4.11 (6.12) | 0.82 (0.91) | 0.36 (0.37) |
| alphabetical | 13.94 (15.28) | 5.34 (6.27) | 1.52 (1.22) | 0.43 (0.58) |
| frequency | 15.84 (14.62) | 6.29 (5.59) | 1.39 (0.94) | 0.47 (0.54) |
| ToolTrie-v0 | 15.67 (17.48) | 7.32 (7.77) | 2.40 (1.90) | 0.91 (1.13) |
| ContextPilot | **18.16** (18.72) | 11.58 (9.93) | 2.98 (4.78) | 1.21 (1.99) |
| **ToolTrie-v1** | 17.97 (**19.47**) | **13.18** (**11.31**) | **6.22** (**4.96**) | **3.74** (**2.21**) |

v1 is first in 7 of 8 cells. The exception is 4 tools on dense, where it
trails ContextPilot by 0.19 points. Its lead over ContextPilot on dense grows
with menu size (0.99x → 1.14x → 2.09x → 3.09x). It also holds across arrival
patterns: v1 caches more in 11 of 12 arrival-pattern × retriever cells.

**In absolute terms these hit rates are small, and part of v1's lead is cache
locality.**
- Even v1 recomputes 93.8% of every 64-tool prompt.
- 48 tokens of each hit are fixed template text that every policy gets.
- For every request, v1 already reuses the most the cache allows.
- v1 reuses recent requests, which a finite cache still holds. ContextPilot's
  reuse often comes from older requests that have already been evicted. With
  an unlimited cache, v1's lead at 64 tools would be 1.4x rather than 2.1x.

See [`concurrent-latency/cache-hit-miss.md`](concurrent-latency/cache-hit-miss.md).

### It is the fastest under load

Time to first token, 64 tools, BM25, 4 requests/s offered (above every
policy's capacity, so requests queue), ms:

| policy | p50 | p95 | p99 | max |
|---|---|---|---|---|
| no reordering | 9,667.9 | 24,699.7 | 26,490.8 | 26,820.4 |
| alphabetical | 9,727.5 | 24,854.3 | 26,781.7 | 27,104.6 |
| frequency | 9,760.9 | 25,009.1 | 26,842.3 | 27,161.5 |
| ToolTrie-v0 | 9,559.8 | 24,825.6 | 26,691.0 | 27,059.1 |
| ContextPilot | 8,565.4 | 22,452.4 | 25,163.5 | 25,432.4 |
| **ToolTrie-v1** | **7,739.7** | **21,219.8** | **23,824.2** | **24,177.4** |

**Fastest on every statistic.** Across four menu sizes, v1 beats ContextPilot
in 21 of 24 reuse, latency and throughput cells. When popular requests repeat
(the skewed arrival pattern), it is up to 43% faster at p50.

**Most traffic within a latency limit.** The highest request rate each policy
sustains with p50 time to first token under 1 second (64 tools, dense, 0.6B):

| policy | natural arrival | bursty arrival | F1 there |
|---|---|---|---|
| frequency | 2.570 req/s | — | 19.84 |
| alphabetical | 2.574 | — | 16.36 |
| ToolTrie-v0 | 2.585 | — | 16.36 |
| no reordering | 2.623 | 2.608 | 25.66 |
| ContextPilot | 2.638 | 2.662 | 21.47 |
| **ToolTrie-v1** | **2.671** | **2.817** | **26.99** |

v1 is the only policy best on both throughput and accuracy at once. At 2.75
req/s with bursty arrival, v1 answers in 821 ms and meets the budget;
ContextPilot takes 1,184 ms and misses it.

### It keeps the right tool in view

This is the design benchmarks and routed systems use: retrieve a wide pool,
reorder it, show the model the top 10. Every policy reorders the same 64
tools, so what matters is which 10 end up in front of the model:

| policy | right tool among the 10, dense | same, BM25 | reuse (dense) | p50 ms (dense) |
|---|---|---|---|---|
| no reordering | **62.0%** | 59.0% | 6.38% | 69.0 |
| **ToolTrie-v1** | 61.5% | **59.5%** | **27.64%** | **60.1** |
| ContextPilot | 47.0% | 39.5% | 20.16% | 64.1 |
| frequency | 38.5% | 41.5% | 9.39% | 72.8 |
| ToolTrie-v0 | 8.0% | 11.5% | 16.12% | 67.1 |
| alphabetical | 7.5% | 11.5% | 11.67% | 68.5 |

**v1 keeps the right tool as often as not reordering, with 4.3x the reuse and
13% lower latency.** It moves only the tools the trie has evidence for, so the
retriever's top picks stay on top when the list is cut. The other policies:

- **ContextPilot pushes the right tool out in one request in seven on dense
  (−15pp, 3.0 sigma) and one in five on BM25 (−19.5pp, 4.0 sigma).**
- **frequency pushes it out in about one request in four on dense and one in
  six on BM25.**
- **ToolTrie-v0 and alphabetical show it in only 7.5-11.5% of requests.**

**Bursty arrival widens the gap.** Under the most bursty arrival order the
data allows, v1 loses at most 2.1 end-to-end F1 points against not reordering.
ContextPilot loses 7.4-7.9. Measured without the model, v1 keeps the right
tool in view more often than ContextPilot in all six arrival patterns, by 8.5
to 22.0 points.

### It costs no accuracy

**Whole menu** (every policy shows the same tools; only the order differs).
v1 minus ContextPilot, F1:

| menu | 0.6B | 4B | 8B |
|---|---|---|---|
| 4 tools | +0.63 | +0.63 | — |
| 16 tools | +3.33 | +3.83 | +2.23 |
| 64 tools | +5.52 | +1.94 | −0.45 (tie) |
| 128 tools | **+9.20** | +3.78 | −0.16 (tie) |

- v1 is ahead in 9 of 11 cells; the other two are ties under the 0.6-point
  noise floor.
- v1 stays within 1.33 points of the unreordered list everywhere. ContextPilot
  falls up to 9.84 points below it. The mechanism is position: at 64 tools
  (dense), v1 leaves the right tool at position 7.3, where the retriever put
  it (7.6). ContextPilot moves it to 13.7, and deeper tools get missed more.

**Cut to 10** (retrieve 64, show 10). End-to-end F1 = share of requests with
the right tool shown × F1. It is the only fair measure when policies show
different tools:

| policy | dense, 0.6B | dense, 4B | BM25, 0.6B | BM25, 4B |
|---|---|---|---|---|
| no reordering | 19.83 | **26.73** | **23.00** | **28.92** |
| **ToolTrie-v1** | 19.92 | 26.03 | 22.00 | 28.08 |
| frequency | **20.08** | 24.37 | 22.67 | 27.92 |
| ContextPilot | 16.25 | 21.45 | 16.58 | 22.33 |
| ToolTrie-v0 | 0.75 | 3.58 | 3.45 | 5.67 |
| alphabetical | 0.75 | 3.58 | 2.45 | 5.67 |

v1 beats ContextPilot in every cell, by 3.7 to 5.8 points (6.7 under bursty
arrival). It stays within 1 point of not reordering. It ties frequency, which
gets about a third of v1's reuse (9.39% against 27.64%, dense).

### Why ToolTrie wins

**A trie matches the previous request's prefix directly. ContextPilot looks
for tools that every request shares.** Which one wins depends on how much
requests have in common:

| workload | tools in **every** request | tools shared by adjacent requests | winner |
|---|---|---|---|
| padded menus | **63 of 64** | 63.0 | ContextPilot |
| constructed 50% overlap | **32 of 64** | 32.0 | ContextPilot |
| retrieved, natural arrival | **0** | 1.8 | **ToolTrie-v1** |
| retrieved, most bursty arrival | **0** | 17.5 | **ToolTrie-v1** |

Real retrieved traffic never has a core shared by every request, even when
arrivals are as bursty as the data allows. Its overlap is local: each request
resembles its neighbours. That is exactly what a trie over served sequences
exploits, which is why v1's lead grows as arrivals get more bursty (BM25 reuse:
1.04x → 1.35x ContextPilot) instead of shrinking.

**And v1 does not undo the retriever.** It places only the tools it has
evidence for: at 128 tools, 2.55 of them on average. It leaves 120 of 200
requests byte-identical to what the retriever returned (BM25). ToolTrie-v0
permuted 99.1% of each menu and moved the right tool to position 62.8 of 128,
to gain 1.13% reuse.

### Where ToolTrie does not win, and what is untested

- **Menus where every request shares most of its tools** (padded menus, 63 of
  64). ContextPilot and ToolTrie-v0 are 48x faster than v1 there.
- **Larger models.** v1's accuracy lead over ContextPilot becomes a tie at 8B
  for 64 and 128 tools. At 8B all orderings converge on accuracy; the
  unreordered list edges v1 by 2.2 F1 at 16 tools, inside noise. v1's cache
  lead does not shrink (still about 1.7x).
- **ContextPilot's full system was not run**, only its ordering. The paper
  credits its annotations, de-duplication and scheduling with about half its
  cache gain; they might close some or all of the gap. This is the most
  important limit.
- **Multi-turn: no reversal, but no meaningful win either.** Simulated on
  300 real coding-agent sessions, v1 is never worse than ContextPilot. But the
  lead is at most 0.08 points of hit rate, because the conversation history
  dominates the cache. What matters there is where new tools are placed and
  how big the cache is
  ([`concurrent-latency/multi-turn-sessions.md`](concurrent-latency/multi-turn-sessions.md)).
  This is a simulation of cache hits with a synthetic tool layer; it has not
  been measured on the GPU.
- **The ordering effect is small in absolute terms on retrieved menus:**
  - between the best and worst policy it is 1.26x at p50, against 68x on
    padded menus;
  - at 64 tools, 94-99% of each prompt is recomputed whatever the policy;
  - over six arrival orders (BM25, 64 tools), v1 caches more than ContextPilot
    in 5 (sign test p = 0.109): suggestive, not significant;
  - at the 10-tool menus benchmarks show, with nothing cut, no policy moves
    latency.
- **The 200-task slice is easier than average.** On a random slice:
  - scores fall by up to a third;
  - v1 still beats ContextPilot, by about half the margin;
  - v1, the unreordered list and frequency end up within about one request
    of each other.
- **v1 does not improve accuracy over the unreordered list.** It avoids the
  damage ContextPilot does.

**Evidence:**

| document | covers |
|---|---|
| [`concurrent-latency/top-findings.md`](concurrent-latency/top-findings.md) | full scoreboard |
| [`concurrent-latency/answers.md`](concurrent-latency/answers.md) | direct answers to the research questions |
| [`concurrent-latency/f1-latency-tools-answers.md`](concurrent-latency/f1-latency-tools-answers.md) | F1 and recompute, latency–precision tradeoff, the 1-second limit, throughput under a limit, how many tools |
| [`concurrent-latency/cache-hit-miss.md`](concurrent-latency/cache-hit-miss.md) | absolute cache hit and miss rates, why misses happen, the ceiling for any ordering, cache size |
| [`concurrent-latency/multi-turn-sessions.md`](concurrent-latency/multi-turn-sessions.md) | multi-turn coding-agent sessions: ordering vs placement vs cache size |
| [`concurrent-latency/README.md`](concurrent-latency/README.md) | key results under load |
| [`concurrent-latency/findings.md`](concurrent-latency/findings.md) | full record; §4.4 dense retrieval, §4.5 arrival patterns, §5 ToolTrie-v1 |
| [`concurrent-latency/how-many-tools.md`](concurrent-latency/how-many-tools.md) | menu size, cut to 10 |
| [`concurrent-latency/metrics-and-latency-tradeoffs.md`](concurrent-latency/metrics-and-latency-tradeoffs.md) | F1, model size, throughput under a latency limit |
| [`../notes/tooltrie-v1-design.md`](../notes/tooltrie-v1-design.md) | how v1 works |

---

## The August phase (ToolTrie-v0): how the project got here

Everything below was measured in August with **ToolTrie-v0**, one request at a
time: 190 accepted GPU replays, primary model Qwen/Qwen3-4B, replicated on
Qwen/Qwen3-0.6B, all 33 audit checks passed. Full evidence and provenance are
in [`consolidated-report.md`](consolidated-report.md); every brief question with
its status is in [`brief-questions-and-answers.md`](brief-questions-and-answers.md).

Findings 1–7 follow the brief's own experimental questions (§7); finding 8
answers the high-level research question the project is named after (§2 Q3).
Each states which question it answers. Differences are in percentage points.
How ToolTrie-v0 works, with the code, is in
[`notes/tooltrie-v0-design.md`](../notes/tooltrie-v0-design.md).

**Findings 1–6 were not overturned**, with one qualification. Finding 1's
"ordering does not fix it" was measured one request at a time; under concurrent
load, ToolTrie-v1 cuts median latency by 15-20% against the unreordered list
at 64 tools (top half). Findings 7 and 8 are where ToolTrie-v0 lost to
ContextPilot and to a frequency counter. That loss is what led to ToolTrie-v1:
v0's trie was not failing, it was being undone by its alphabetical fallback.
Both are marked where they have been superseded.

## August headline (ToolTrie-v0)

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
> is unaffected for ToolTrie-v0 — there is no common core to converge to, and
> ContextPilot leads it at every depth. ToolTrie-v1, which keeps unmatched tools
> in retrieval order, reverses that (see the ToolTrie-v1 results at the top). The "zero spread across trials" above measures determinism
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

> **Superseded in part (2026-09-11).** This finding is about ToolTrie-v0. The
> "stronger comparators" bullet does not hold for ToolTrie-v1, which beats
> ContextPilot on retrieved menus. See the ToolTrie-v1 results at the top.

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

> **Revised (2026-09-11).** Measured for ToolTrie-v0, and true for it. But v0's
> trie was being undone by its alphabetical fallback rather than doing no work:
> with the fallback fixed (ToolTrie-v1), the trie beats ContextPilot on
> retrieved menus, and its lead grows with arrival locality — because a trie
> matches the *previous* request's prefix, which is exactly what local overlap
> rewards and what a set intersection cannot use. The claim that adaptivity
> alone wins holds where requests share a global core (padded menus), not on
> retrieved traffic. See the ToolTrie-v1 results at the top.

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

**Weighting the trie changes nothing.** `WeightedToolTrie` (then named
`tooltrie_v1`, now `src/tatm/tooltrie_weighted.py`; the name `tooltrie_v1` was
later reused for an unrelated variant) reads the `visit_count`
that v0 never consults, in selection and in eviction. It matches v0 to five
decimal places in all four regimes, and re-deriving both planners offline on
identical menus shows **0 of 200 records differ**. The weighting does act — v1
evicts 1,497 nodes against 1,494 on empirical — but `_reachable_cached_cost`
decides almost every choice, so the tie-break holding the weight is rarely
reached. Only the empirical regime was predeclared: **1/1 predeclared run
accepted, three further regimes added afterwards and also accepted.**


Sources: `reports/tooltrie-pressure/20260811-001032/`,
`reports/tooltrie-weighted/20260811-144741/`.
