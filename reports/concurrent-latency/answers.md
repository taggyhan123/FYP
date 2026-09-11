# ToolTrie-v1 vs ContextPilot, and the latency questions — answers

Direct answers to five questions, each with its verdict first and the evidence
under it. Every number is a run under `cluster/results/`; full derivations are
in [`findings.md`](findings.md), [`how-many-tools.md`](how-many-tools.md) and
[`metrics-and-latency-tradeoffs.md`](metrics-and-latency-tradeoffs.md).

| # | Question | Answer |
|---|---|---|
| 0 | Is ToolTrie-v1 better than ContextPilot? | **Yes, on single-turn retrieved menus — and it is the safer of the two** |
| 1 | Does using F1 → better accuracy → less KV-cache recompute? | **No.** Accuracy and reuse are independent |
| 2 | Is there a latency–precision tradeoff through configuration? | **Yes, through model size**, with sharply diminishing returns. Quantization untested |
| 3 | If every request must answer within 1 s, what accuracy? | **On average: 0.6B fits 64 tools (F1 27), 4B fits 16 (F1 40), 8B neither. For *every* request, only 0.6B fits (≤16 tools)** |
| 4 | Optimise throughput under a latency limit, scored on F1? | **ToolTrie-v1 is the only policy best on both axes at once** |

All results are Qwen3 on one RTX 3090, vLLM with prefix caching unmodified,
single-turn requests over the ToolRet catalogue (44,453 tools).

---

## 0. Is ToolTrie-v1 better than ContextPilot?

**Yes, on the problem this project studies — but the accurate claim is narrower
than "better":**

> **On single-turn retrieved tool menus, ToolTrie-v1's ordering beats
> ContextPilot's ordering — and it is the safer of the two when the menu is cut
> short, because it does not push the right tool out of view.**

### Where v1 wins

| measure | result |
|---|---|
| cache reuse | wins 4 of 4 menu sizes on BM25 and 3 of 4 on dense (loses 4 tools by 0.19 points); 11 of 12 arrival patterns |
| latency | faster in 11 of 12 cells, by up to 43% |
| keeping the right tool in the shown 10 | **v1 loses almost nothing; ContextPilot drops it in 1 request in 5–7** |
| throughput under a 1-second limit | +1.2% on natural arrival order, **+5.8%** on bursty arrival |
| accuracy (F1) | ties or beats ContextPilot at 0.6B and 4B; see below |

### Why it wins — one rule that explains every result

| workload | tools present in **every** request | winner |
|---|---|---|
| padded menus | **63 of 64** | ContextPilot |
| constructed 50% overlap | **32 of 64** | ContextPilot |
| retrieved, natural order | **0** | **ToolTrie-v1** |
| retrieved, most bursty order possible | **0** | **ToolTrie-v1** |

- **ContextPilot wins where requests share a global core** — tools present in
  every request — because it computes a set intersection, and the intersection
  *is* the core.
- **ToolTrie-v1 wins where overlap is only local** — each request resembling
  its neighbours, with nothing common to all — because a trie matches the
  previous request's prefix directly.
- **Real retrieved traffic never has a global core**, even at its most bursty.
  That is why v1's lead over ContextPilot *grows* as arrivals get more bursty
  (BM25: 1.04× → 1.35×) instead of shrinking.

### Where the claim stops

1. **We beat ContextPilot's *ordering*, not ContextPilot the system.** It ran
   with no order annotations, de-duplication or scheduling — which its own
   paper credits with roughly half its cache gain. The full system might close
   some or all of the gap. **This is the most important limit.**
2. **With a global core, ContextPilot wins decisively.** On padded menus v1 is
   near the bottom.
3. **Margins are small and the statistics are weak.** Reuse on retrieved menus
   is 1–8% in absolute terms. Across arrival orders v1 wins 5 of 6, sign test
   **p = 0.109** — suggestive, not significant. At 8B the accuracy advantage
   becomes a tie.
4. **Multi-turn is untested.** When conversation history stays in the prompt,
   requests do share a core — ContextPilot's actual design regime, and the one
   condition that could reverse this.

---

## F1: v1 doesn't *improve* accuracy — ContextPilot *damages* it

### Whole menu (every policy shows the same tools, only the order differs)

v1 minus ContextPilot, F1 points:

| menu | 0.6B | 4B | 8B |
|---|---|---|---|
| 4 tools | +0.63 | +0.63 | — |
| 16 tools | +3.33 | +3.83 | +2.23 |
| 64 tools | +5.52 | +1.94 | **−0.45** |
| 128 tools | **+9.20** | +3.78 | **−0.16** |

**v1 is ahead in 9 of 11 cells; the two exceptions are ties** (both 8B deltas
sit under the 0.6-point run-to-run noise floor). The gap **grows with menu
size** and **shrinks with model size** — a larger model copes better with
ContextPilot's reordering.

### Truncated to 10 — the realistic design, where they separate most

Scored on end-to-end F1 (share of requests with the right tool in view × F1),
because the tools shown differ by policy:

| setting | v1 minus ContextPilot |
|---|---|
| dense retriever, natural arrival | +3.67 |
| dense retriever, bursty arrival | **+6.66** |
| BM25 retriever, natural arrival | +5.42 |
| BM25 retriever, bursty arrival | +5.84 |
| 4B model, dense / BM25 | +4.58 / +5.75 |

**v1 wins every truncation cell, by 3.7 to 6.7 points** — no ties, no exceptions.

### Why: compare both against not reordering at all

| F1 | no reordering | **ToolTrie-v1** | ContextPilot |
|---|---|---|---|
| 64 tools, 0.6B | 25.66 | 26.99 | 21.47 |
| 128 tools, 0.6B | 26.02 | 25.38 | **16.18** |
| 64 tools, 4B | 34.29 | 34.29 | 32.35 |
| random task sample | 20.39 | 20.27 | 17.31 |

**v1 stays within ±1.3 points of doing nothing; ContextPilot falls up to 10
points below it.** ContextPilot moves the right tool deeper into the list
(position 13.7 against v1's 7.3, at 64 tools), and deeper tools get missed more.

### Where F1 does not separate them

- **At 10 tools with nothing cut off**, it is a tie (−0.80 at 0.6B, +0.86 at 4B).
- **At 8B**, the advantage disappears.
- **Individual cells are mostly not significant alone** — about 5 points of
  standard error each at 200 tasks. The case rests on consistency across ~20
  cells plus the position mechanism that predicts the direction.
- **On a random, harder task sample** the whole-menu margin roughly halves
  (+5.52 → +2.96).

---

## 1. Does using F1 → better accuracy → reduce KV-cache recompute?

**No. The chain fails at both links.**

**Choosing a metric cannot cause anything.** Switching to F1 changes how
precisely accuracy is *reported*. It does not change what the model does, and
it cannot change the cache.

**And accuracy does not drive cache reuse — the two are independent.** Reuse is
decided when the request arrives: vLLM matches the prompt's tokens against
prefixes already computed. That finishes **before the model generates
anything**. Accuracy is a property of what the model generates *afterwards*. A
later step cannot change an earlier one. Nor can one request's accuracy affect
the next request's reuse: requests here are independent, so no output ever
enters another request's prompt.

**The data shows them moving independently, in both directions:**

| case | reuse | accuracy |
|---|---|---|
| ToolTrie-v0, truncated to 10 | **16.12%** — 2.5× no reordering | **0.75** end-to-end F1 — the worst |
| v1 vs ContextPilot at 0.6B | v1 caches ~2× more | v1 ahead by 5.5–9.2 F1 |
| v1 vs ContextPilot at 8B | v1 **still** caches ~1.7× more | **tied** |

High reuse with terrible accuracy; reuse held constant while accuracy moved. If
one caused the other, neither could happen.

**What is actually true.** The *ordering policy* affects both, separately: it
decides how much of the prompt matches earlier prompts (reuse) *and* where the
right tool lands in the list (accuracy). Sometimes those move together,
sometimes they trade off hard. **F1 is still the right metric** — it is the
field's standard for tasks needing several tools — but it measures the
accuracy side only.

*One boundary:* in multi-turn traffic the model's own output enters the next
turn's prompt, so a channel from accuracy to reuse would exist. Untested here.

---

## 2. Is there a latency–precision tradeoff through configuration?

**Yes, through model size — with sharply diminishing returns. Quantization, the
other obvious lever, was never tested.**

**Model size**, ToolTrie-v1, per-request latency with no queue:

| menu | 0.6B | 4B | 8B |
|---|---|---|---|
| 64 tools | 510 ms / F1 26.99 | 1,950 ms / F1 34.29 | 3,062 ms / F1 35.32 |
| 128 tools | 1,090 ms / F1 25.38 | 3,884 ms / F1 34.87 | 5,677 ms / F1 35.13 |

- **0.6B → 4B**: about **3.8× the latency** for **+7 to +9.5 F1**. A real trade.
- **4B → 8B**: another **46–57% latency** for only **+0.3 to +1.0 F1**. On this
  task, 8B is close to pure cost.

**Numeric precision (quantization)** — fp8, int8, int4, AWQ, GPTQ — was checked
across every server launch in the project: **never used**. Everything ran at
native bf16. It typically cuts latency and memory substantially at some
accuracy cost, and it remains the most obvious untested configuration.

**Menu size is also a configuration, and a bigger lever than ordering.** At the
10-tool menus published benchmarks use, no ordering policy moves latency at all
(66–68 ms, all six). The cheapest latency win is showing fewer tools — see
[`how-many-tools.md`](how-many-tools.md).

---

## 3. If every request must answer within 1 second, what accuracy?

**It depends on the model — and on whether requests queue.**

**With no queue** (one request at a time — the hard floor), the largest menu
each model can serve in under 1 second **on average**:

| model | largest menu under 1 s, average | ToolTrie-v1 F1 | ContextPilot F1 | no reordering F1 |
|---|---|---|---|---|
| **0.6B** | **64 tools** (~510 ms) | **26.99** | 21.47 | 25.66 |
| **4B** | **16 tools** (~900 ms) | **40.25** | 36.42 | 40.05 |
| 8B | **none** — 16 tools already takes ~1,470 ms | — | — | — |

- **4B gives the highest accuracy that fits on average** (F1 40), on a 16-tool
  menu.
- **If *every* request must finish its reply within 1 s, only 0.6B qualifies,
  at up to 16 tools.** At 4B/16 tools, 54 of 200 replies take longer. If the
  limit is on the *first token* instead, 8B fits at 16 tools. All three
  readings, with end-to-end F1 per policy, are in
  [`f1-latency-tools-answers.md`](f1-latency-tools-answers.md) §3.
- **8B cannot finish replies within 1 second** at any menu size tested.
- **With no queue, ordering barely changes latency** — every policy lands within
  1–2% at a fixed menu size, because the time is almost all spent reading the
  prompt, which depends on its length, not its order. Ordering changes the
  *accuracy* you get inside the budget. Its effect on *latency* needs load —
  question 4.

---

## 4. Optimise throughput under a latency limit, scored on F1

**ToolTrie-v1 is the only policy that is best on both axes at once.**

The right way to frame it: **fix a latency limit, find the highest traffic each
policy sustains within it, and report the accuracy delivered there.** Here:
p50 time-to-first-token under 1,000 ms, 64-tool menus, 0.6B, all six policies:

| policy | highest traffic within 1 s | F1 there |
|---|---|---|
| frequency sort | 2.570 req/s | 19.84 |
| alphabetical | 2.574 | 16.36 |
| ToolTrie-v0 | 2.585 | 16.36 |
| no reordering | 2.623 | 25.66 |
| ContextPilot | 2.638 | 21.47 |
| **ToolTrie-v1** | **2.671** | **26.99** |

**Every other policy trades one axis for the other** — frequency sort sustains
the least traffic; no reordering matches v1's accuracy but sustains less;
ContextPilot is middling on both.

**Under bursty arrival the margin grows** — this was the result most expected to
flip, and it strengthened:

| policy | natural arrival | bursty arrival |
|---|---|---|
| no reordering | 2.623 req/s | 2.608 |
| ContextPilot | 2.638 | 2.662 |
| **ToolTrie-v1** | 2.671 | **2.817** |
| v1 over ContextPilot | +1.2% | **+5.8%** |

At 2.75 req/s v1 answers in **821 ms** and meets the budget; ContextPilot takes
**1,184 ms** and misses it.

**Caveats.** The spread is small (about 4% between best and worst policy on
natural arrival) — ordering barely matters on retrieved menus in general. And
this is **p50**: SLAs are usually written on **p95**, which binds far earlier
(at 2.5 req/s every policy's p95 is already 2,000–2,600 ms). One menu size, one
model, two arrival patterns — a first measurement of this framing, not a curve.

---

## Standing limits on everything above

| limit | effect |
|---|---|
| single-turn only | multi-turn is ContextPilot's design regime; could reverse the headline |
| ContextPilot run as ordering only | its full system is untested |
| 200-task sample is easier than average | absolute accuracy flattered by up to ~⅓; v1 still beats ContextPilot on a random sample but by half as much, and the middle of the ranking reorders (ContextPilot falls behind frequency sort) |
| decoding 199/200 reproducible | gaps under ~0.6 F1 are noise |
| 200 tasks per cell | ~5 points standard error; most single cells not significant alone |
