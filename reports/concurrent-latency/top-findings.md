# Top findings: ToolTrie-v1 against ContextPilot and the traditional orderings

This document collects the project's strongest results so far. It has three
emphases:

- how ToolTrie-v1 compares with the traditional baselines and with ContextPilot;
- whether the right tool ends up among the k the model sees;
- what accuracy and F1 show.

Every number is a run under `cluster/results/`. Each part ends with where its
numbers are derived.

**Setup.** Qwen3 (0.6B, 4B, 8B) on one RTX 3090, vLLM 0.26.0 with automatic
prefix caching unmodified. Single-turn requests over the ToolRet catalogue
(44,453 tools), 200 tasks per cell.

---

## In one paragraph

**On real retrieved tool menus, ToolTrie-v1 is the only ordering policy that
gets the cache benefit without an accuracy cost.** At 64 tools on a dense
retriever it caches 2.1x what ContextPilot caches and 7.6x what the unreordered
list gets, and it sustains the most traffic under a 1-second latency limit.
Suppose a system retrieves 64 tools but shows the model only 10. v1 keeps the
right tool in view as often as not reordering at all. ContextPilot pushes it
out in one request in five to seven, and the alphabetical-style orderings show
it in only about one request in ten. On accuracy (F1), v1 matches not
reordering, beats ContextPilot in every 0.6B and 4B cell, and ties it at 8B on
the larger menus. ContextPilot wins only where every request shares a common
core of tools. Real retrieved traffic never has one.

---

## The six policies

| policy | what it does to the retrieved list | role |
|---|---|---|
| no reordering (`original`) | nothing: keeps the retriever's ranking | traditional baseline |
| `alphabetical` | sorts by tool name | traditional baseline |
| `frequency` | puts tools that appear in the most menus first | traditional baseline |
| ToolTrie-v0 | trie-matched tools first, the rest sorted **alphabetically** | this project, first version |
| ContextPilot | moves the tools that requests share to the front | published state of the art (its ordering only) |
| **ToolTrie-v1** | trie-matched tools first, the rest **left in retrieval order** | this project |

The only difference between v0 and v1 is that one line: what happens to the
tools the trie cannot match. In v0 the alphabetical fallback decides about 93%
of every menu. At 128 tools, v0 moved the right tool to position 62.8 on
average, to gain 1.13% reuse. v1 places only the matched tools (2.55 of 128)
and leaves 120 of 200 requests byte-identical to what the retriever returned.
These figures are from the BM25 retriever.

---

## 1. The scoreboard: v1 against every rival

The table shows v1's advantage over each rival. Menus are 64 retrieved tools
unless stated. **Bold** marks a clear v1 win.

| measure | vs no reordering | vs alphabetical | vs frequency | vs ToolTrie-v0 | vs **ContextPilot** |
|---|---|---|---|---|---|
| ***Cache and speed*** | | | | | |
| cache reuse, dense retriever | **7.6x** | **4.1x** | **4.5x** | **2.6x** | **2.1x** |
| cache reuse, BM25 retriever | **5.5x** | **4.1x** | **5.3x** | **2.6x** | 1.04x |
| p50 latency, BM25, overloaded (4 req/s) | **20% lower** | **20% lower** | **21% lower** | **19% lower** | **10% lower** |
| p99 latency, same run | **10% lower** | **11% lower** | **11% lower** | **11% lower** | **5% lower** |
| traffic sustained at p50 < 1 s, dense | +1.8% | +3.8% | +3.9% | +3.3% | +1.2% |
| same, most bursty arrival possible | **+8.0%** | — | — | — | **+5.8%** |
| ***Right tool among the 10 shown*** (retrieve 64, show 10) | | | | | |
| right tool survives the cut, dense | tie (−0.5pp) | **+54pp** | **+23pp** | **+53.5pp** | **+14.5pp** |
| right tool survives the cut, BM25 | tie (+0.5pp) | **+48pp** | **+18pp** | **+48pp** | **+20pp** |
| end-to-end F1 (2 retrievers × 2 models) | tie (−1.0 to +0.1) | **+19.2 to +22.5** | tie (−0.7 to +1.7) | **+18.6 to +22.5** | **+3.7 to +5.8** |
| cache reuse in this design | **4.3x** | **2.4x** | **2.9x** | **1.7x** | **1.4x** |
| ***Accuracy, whole menu*** (F1; 4 menu sizes × 0.6B, 4B) | | | | | |
| cells where v1 is ahead | 4 of 8, 1 tie; every gap ≤ 1.33 | 7 of 8 | 5 of 8 | 7 of 8 | **8 of 8** |
| F1 margin at 64 tools, 0.6B | +1.33 | **+10.63** | **+7.15** | **+10.63** | **+5.52** |
| ***Where v1 loses*** | | | | | |
| p50 latency on padded menus (63 of 64 tools shared) | 30% lower | 16x higher | — | 48x higher | 48x higher |

**How to read it:**

- **Against the traditional orderings, v1 wins everything the cache
  decides:** 4-8x the reuse, about 20% lower median latency under load, and
  the most traffic under the 1-second limit.
- **Against no reordering, accuracy is a tie by design.** v1 moves only the
  tools it has evidence for and leaves the retriever's ranking alone
  otherwise. It therefore keeps the unreordered list's accuracy and adds the
  cache benefit on top. That is the whole idea of v1.
- **`frequency` is the strongest traditional baseline on accuracy.** It ties
  v1 end-to-end when the list is cut, and has the best whole-menu F1 of any
  policy at 64 tools on 4B (36.58 against v1's 34.29). But it has the least
  reuse of any reordering policy. When the list is cut, the right tool is
  missing from view in 18-23 more requests out of every 100 than with v1.
- **Against ContextPilot, v1 is ahead on every row except the padded one.**
  The thinnest margins are BM25 reuse at 64 tools (1.04x) and throughput on
  natural arrival (+1.2%). The widest are keeping the right tool in view (+14.5
  to +20pp) and reuse on the dense retriever (2.1x).
- **The last row is ContextPilot's regime** (section 4). When requests share
  most of their tools, the orderings that group shared tools win by 48x, and v1
  is second-worst.

*Derived in:* README §1, §3 · `findings.md` §4.4, §4.5 ·
`how-many-tools.md` §5.2-5.3 · `metrics-and-latency-tradeoffs.md` §1.2, §4.2-4.3.

---

## 2. The right tool among the k shown

Two separate questions are easy to mix up here. **Is the right tool in the
menu at all?** The menu size k decides that, not the ordering. **Does it stay in
the part of the list the model actually sees?** That depends on the ordering
policy.

### 2.1 Menu size k decides whether the right tool is there at all

Dense retriever, ToolTrie-v1, Qwen3-0.6B:

| k (tools in the menu) | right tool in the menu | tools the model calls per request | end-to-end F1 |
|---|---|---|---|
| 4 | 52.5% | 0.70 | 20.50 |
| 16 | 67.5% | 0.80 | 22.42 |
| 64 | 81.5% | 0.88 | 22.00 |
| 128 | 85.5% | 0.90 | 21.70 |

- **Tasks need 1.54 tools on average** (1.77 across the whole corpus), and
  **the model calls about one, whatever k is**. Showing 128 tools does not
  make it call more.
- **A bigger menu is a bigger bet that the right tool is in it.** At 4 tools it
  is missing from almost half of all requests, and nothing downstream can
  recover a tool that was never shown.
- **End-to-end F1 is flat across a 32x range of k.** Small menus miss the tool;
  big menus contain it but distract the model. Among the requests where the
  model called nothing, 62% at k=4 had no right tool to call. At k=64 and k=128,
  90% had the right tool in front of it. **The failure moves from retrieval to
  the model as k grows.**
- **At the 10 tools benchmarks show, with nothing cut, ordering changes nothing
  you can measure.** All six policies have a p50 of 66-68 ms, and accuracy is
  within one standard error. v1 still caches the most (13.71%, against
  ContextPilot's 12.46% and no reordering's 6.38%). So the mechanism survives
  at k=10, but the effect needs either a long prompt or a cut list.

### 2.2 Retrieve 64, show 10: the ordering decides whether the right tool survives

This design is how routed systems and benchmarks work: retrieve a wide pool,
reorder it, show the model the top 10. It is comparable to the field (10
shown) and realistic (a 64-tool pool behind the cut). It is also the design
that most clearly separates the policies. Every policy reorders the same 64
tools, so the question is which 10 end up in front of the model.

| policy | right tool among the 10, dense | same, BM25 | reuse (dense) | p50 ms (dense) |
|---|---|---|---|---|
| no reordering | 62.0% | 59.0% | 6.38% | 69.0 |
| **ToolTrie-v1** | **61.5%** | **59.5%** | **27.64%** | **60.1** |
| ContextPilot | 47.0% | 39.5% | 20.16% | 64.1 |
| `frequency` | 38.5% | 41.5% | 9.39% | 72.8 |
| ToolTrie-v0 | 8.0% | 11.5% | 16.12% | 67.1 |
| `alphabetical` | 7.5% | 11.5% | 11.67% | 68.5 |

- **v1 keeps the right tool in view as often as not reordering does** (−0.5pp
  dense, +0.5pp BM25). It also gets **4.3x the reuse and 13% lower p50**. It
  moves only the tools the trie has evidence for, so the retriever's top
  picks stay on top when the list is cut.
- **ContextPilot pushes the right tool out of view in one request in seven on
  dense (−15pp, 3.0 sigma) and one in five on BM25 (−19.5pp, 4.0 sigma).**
  It moves shared, cache-friendly tools to the front, and when only 10
  survive, those tools take the right one's place.
- **ToolTrie-v0 and `alphabetical` show the right tool in only 7.5-11.5% of
  requests.** Sorting by name puts arbitrary tools first. v0's 16% reuse is
  bought by showing the wrong tools.
- **`frequency` pushes the right tool out in about one request in four on
  dense (−23.5pp) and one in six on BM25 (−17.5pp).** It puts the common tools
  first, and the right tool is often not one of them.

**The gap widens when requests arrive in bursts.** The next table uses the
most bursty arrival order the data allows. It is an upper bound: adjacent
requests share 17.5 of 64 tools rather than 1.8. Scores are end-to-end F1, 0.6B:

| retriever / arrival | no reordering | **ToolTrie-v1** | ContextPilot | v1 − ContextPilot |
|---|---|---|---|---|
| dense, natural | 19.83 | 19.92 | 16.25 | +3.67 |
| dense, **most bursty** | 19.83 | 19.08 | **12.42** | **+6.66** |
| BM25, natural | 23.00 | 22.00 | 16.58 | +5.42 |
| BM25, **most bursty** | 23.00 | 20.92 | **15.08** | **+5.84** |

On dense, bursty traffic roughly doubles the damage ContextPilot does: its loss
against no reordering grows from 3.6 to 7.4 points. v1 now loses a little too
(0.8 dense, 2.1 BM25), but far less. Measured without the model at all, v1
keeps the right tool in view more often than ContextPilot by **+8.5 to +22.0pp
in all six arrival patterns tested, on both retrievers**.

### 2.3 Whole menus: where the right tool lands in the list

With nothing cut, every policy shows the same tools, but the ordering still
decides **how deep the right tool sits**, and deeper tools get missed more.
The first right tool's average position on the dense retriever (lower is
better):

| policy | 64 tools | 128 tools |
|---|---|---|
| no reordering | 7.6 | 11.7 |
| **ToolTrie-v1** | **7.3** | **11.6** |
| ContextPilot | 13.7 | 22.3 |
| ToolTrie-v0 | 32.5 | 66.3 |

v1 leaves the right tool where the retriever put it. ContextPilot pushes it
about twice as deep, and v0 pushes it to the middle of the list. On BM25,
ContextPilot keeps it shallower than v1 (12.5 against 18.0 at 64 tools), and
whole-menu accuracy against ContextPilot there is a wash (section 3.2).

*Derived in:* `how-many-tools.md` §3, §5.1-5.4 · `findings.md` §4.4, §4.5.

---

## 3. Accuracy and F1

### 3.1 Five measures, and when each one is fair

| measure | what it asks | computed as | when it is fair |
|---|---|---|---|
| **accuracy** (`gold_hit_ceil`) | did the model call at least one right tool? | only over requests whose menu contains a right tool | legacy; fair only when every policy shows the same tools |
| **precision** | of the tools it called, how many were right? | correct calls ÷ calls made, per request | same condition |
| **recall** | of the tools the task needs, how many did it call? | correct calls ÷ tools needed, per request | same condition |
| **F1** | both at once | 2PR/(P+R) per request, averaged over requests with a right tool in the menu | the field's standard for tasks needing several tools; same condition |
| **end-to-end F1** | what the user actually gets | (share of requests with a right tool in the menu) × F1 | **always; primary metric** |

**Whole menu:** every policy shows the same tools, so every policy has the
same chance of the right tool being present. F1 and end-to-end F1 then differ
only by a constant factor and rank the policies identically.

**Retrieve 64, show 10:** the policy decides which tools are shown, so only
end-to-end F1 is fair (section 3.3 shows what goes wrong otherwise).

**None of these divides by menu size.** A request that calls its 2 needed
tools correctly scores F1 = 100% from a 3-tool menu or from a 64-tool menu. So
when accuracy falls with k, that is the model genuinely doing worse, not a
bigger denominator.

### 3.2 Whole menu: F1 for all six policies (dense retriever)

| menu | model | **v1** | ContextPilot | v0 | alphabetical | frequency | no reordering |
|---|---|---|---|---|---|---|---|
| 4 | 0.6B | 39.05 | 38.41 | **40.63** | **40.63** | 39.52 | 39.68 |
| 16 | 0.6B | **33.21** | 29.88 | 26.10 | 26.10 | 28.52 | 32.72 |
| 64 | 0.6B | **26.99** | 21.47 | 16.36 | 16.36 | 19.84 | 25.66 |
| 128 | 0.6B | 25.38 | 16.18 | 12.18 | 12.18 | 17.06 | **26.02** |
| 4 | 4B | 49.84 | 49.21 | 46.51 | 46.51 | 49.59 | **50.22** |
| 16 | 4B | 40.25 | 36.42 | 36.54 | 37.04 | **40.30** | 40.05 |
| 64 | 4B | 34.29 | 32.35 | 28.81 | 28.10 | **36.58** | 34.29 |
| 128 | 4B | **34.87** | 31.09 | 27.88 | 27.19 | 34.11 | 34.58 |

**v1 minus ContextPilot, by model size**, including Qwen3-8B:

| menu | 0.6B | 4B | 8B |
|---|---|---|---|
| 4 | +0.63 | +0.63 | — |
| 16 | +3.33 | +3.83 | +2.23 |
| 64 | +5.52 | +1.94 | −0.45 (tie) |
| 128 | **+9.20** | +3.78 | −0.16 (tie) |

- **v1 beats ContextPilot in all 8 cells at 0.6B and 4B.** At 0.6B the margin
  grows steadily with menu size, to +9.20 at 128 tools; at 4B it is smaller
  and uneven (+0.63 to +3.83). At 8B, v1 is still ahead at 16 tools and tied
  at 64 and 128, where both gaps are under the 0.6-point run-to-run noise
  floor.
- **v1 stays within 1.33 points of no reordering in every cell. ContextPilot
  falls up to 9.84 points below it** (128 tools, 0.6B). So v1 does not
  improve accuracy. What it avoids is the damage ContextPilot does.
- **Traditional baselines:** `alphabetical` and v0 fall 3-13 points below v1
  once the menu has 16 or more tools. `frequency` beats ContextPilot in 6 of 8
  cells. It edges v1 in 3 of 8, but by a clear margin only at 64 tools on 4B
  (+2.29).
- **At 4 tools there is almost nothing to reorder**, and every gap is within
  noise.

**The legacy accuracy metric agrees.** On dense, v1 beats ContextPilot in all
8 cells measured (4 menu sizes × 0.6B, 4B), by 0.95 to 10.52 points. Only 128
tools on 0.6B clears 2 standard errors on its own. On BM25, whole-menu accuracy
against ContextPilot is a wash: 2 wins, 2 losses, 2 ties, all under 1.6
standard errors. The picture also holds across three definitions of accuracy
(any right tool, all right tools, fraction of right tools). In every cell
checked (64 and 128 tools, 0.6B and 4B), {v1, no reordering} > ContextPilot >
v0.

### 3.3 Retrieve 64, show 10: end-to-end F1, and the trap in plain F1

| policy | dense, 0.6B | dense, 4B | BM25, 0.6B | BM25, 4B |
|---|---|---|---|---|
| no reordering | 19.83 | **26.73** | **23.00** | **28.92** |
| **ToolTrie-v1** | 19.92 | 26.03 | 22.00 | 28.08 |
| `frequency` | **20.08** | 24.37 | 22.67 | 27.92 |
| ContextPilot | 16.25 | 21.45 | 16.58 | 22.33 |
| ToolTrie-v0 | 0.75 | 3.58 | 3.45 | 5.67 |
| `alphabetical` | 0.75 | 3.58 | 2.45 | 5.67 |

**v1 beats ContextPilot in every cell, by 3.7 to 5.8 points** (6.7 under bursty
arrival), and stays within 1 point of no reordering.

**Plain (conditional) F1 would have recommended the wrong policies.** It only
scores requests whose right tool survived the cut. `frequency` loses the hard
cases to truncation, so it gets scored on the easy ones. Its plain F1 is the
highest of any policy on either retriever: 52.16 and 63.29 on dense, 54.62 and
67.27 on BM25. Yet its end-to-end F1 is ordinary. ContextPilot shows the same
pattern more mildly. **Ranked by plain F1, the four usable policies would
put first the two that lose the most requests.** That is why end-to-end F1 is
the primary metric.

### 3.4 Accuracy does not buy cache reuse

Reuse is settled when the request arrives: the prompt is matched against
cached prefixes before the model generates anything. Accuracy is measured on
what the model generates afterwards, so neither causes the other. The data
shows them moving independently:

- ToolTrie-v0 on the cut list has 16.12% reuse (2.5x no reordering) and the
  worst end-to-end F1 (0.75).
- At 8B, v1 still caches about 1.7x what ContextPilot caches, while their
  accuracy is tied.

The ordering policy affects both, but separately. It decides how much of the
prompt matches earlier prompts, and it decides where the right tool lands.

*Derived in:* `metrics-and-latency-tradeoffs.md` §1-§3 · README ·
`how-many-tools.md` §5.2-5.4 · `answers.md` §0-§1.

---

## 4. Why v1 wins here, and where ContextPilot wins

One rule explains every v1-vs-ContextPilot result in the project:

| workload | tools present in **every** request | tools shared by adjacent requests | winner |
|---|---|---|---|
| padded menus | **63 of 64** | 63.0 | ContextPilot |
| constructed 50% overlap | **32 of 64** | 32.0 | ContextPilot |
| retrieved, natural arrival | **0** | 1.8 | **ToolTrie-v1** |
| retrieved, most bursty arrival | **0** | 17.5 | **ToolTrie-v1** |

- **ContextPilot wins where requests share a global core**, meaning tools
  present in every request. It computes what requests have in common, and
  there that shared set *is* the core. On padded menus its p50 is 91.7 ms
  against v1's 4,389.5 ms.
- **v1 wins where overlap is only local:** each request resembles its
  neighbours, but nothing is common to all. A trie matches the previous
  request's prefix directly.
- **Real retrieved traffic never has a global core**, even in the most bursty
  order possible. That is why v1's lead grows as arrivals get more bursty
  (BM25 reuse: 1.04x → 1.35x ContextPilot) instead of shrinking.
- Across the six arrival patterns tested, v1 caches more in 11 of 12 cells
  and is faster in 11 of 12, by up to 43%.

**On the dense retriever the two move in opposite directions.** Between BM25
and dense, ContextPilot's reuse at 64 tools falls (4.78% → 2.98%) while v1's
rises (4.96% → 6.22%). Dense ranks similar queries similarly: when two menus
share leading tools, they share them in the same order. v1 keeps that
agreement; ContextPilot overrides it.

*Derived in:* `findings.md` §4.3-4.5 · README.

---

## 5. Where the claims stop

1. **This compares ContextPilot's ordering, not its full system.** Its order
   annotations, de-duplication and scheduling were not run, and its paper
   credits them with about half its cache gain. The full system might close
   some or all of the gap. **This is the most important limit.**
2. **Multi-turn was tested afterwards, in simulation: no reversal.** On 300
   real coding-agent sessions, v1 is never worse than ContextPilot, but by at
   most 0.08 points of hit rate, because the conversation history dominates.
   The one exception is re-retrieving tools every turn at the front of the
   prompt: there v1 leads 20.3% to 16.2%, and appending new tools gives 93%
   instead ([`multi-turn-sessions.md`](multi-turn-sessions.md)). It was
   confirmed on the GPU, where the policies' latencies are equal. The tool
   layer is synthetic.
3. **The cache margins are small in absolute terms.** At 64 tools, 94-99% of
   each prompt is recomputed whatever the policy. Over six random arrival
   orders (64 tools, BM25), v1 caches more than ContextPilot in 5 (sign test
   p = 0.109): suggestive, not significant.
4. **The accuracy advantage over ContextPilot is bounded by model size.** It
   becomes a tie at 8B for 64 and 128 tools. The cache advantage does not
   shrink (still about 1.7x).
5. **Single cells are weak alone.** Each has about 5 points of standard error
   at 200 tasks, and decoding is 199/200 reproducible, so gaps under ~0.6 F1
   are noise. The case rests on consistency across about 20 cells plus the
   position mechanism that predicts the direction.
6. **The 200-task slice is easier than average.** On a random, harder slice:
   - scores fall by up to a third;
   - v1 still beats ContextPilot, by about half the margin (F1 +5.52 → +2.96);
   - v1, no reordering and `frequency` end up within about one request of each
     other;
   - ContextPilot falls behind `frequency`.

   Trust the v1-vs-ContextPilot direction, not the absolute scores or the
   finer ordering among similar policies.
7. **At the 10-tool menus benchmarks use, with nothing cut, ordering has no
   latency effect.** Its value there shows up only in whether the right tool
   survives the cut (section 2.2).
8. **Untested:**
   - quantization;
   - models other than Qwen3;
   - GPUs other than the RTX 3090;
   - a ~150-tool menu (the commonly cited real deployment size);
   - SLAs on p95 rather than p50. p95 binds far earlier: at 2.5 req/s every
     policy's p95 is already 2,000-2,600 ms.

---

## Where each result comes from

| result | derivation | runs |
|---|---|---|
| reuse and latency, six policies, BM25 | README §1, §3 · `findings.md` §4.1, §5 | several; listed in `findings.md` §A.5 |
| reuse and latency, dense | `findings.md` §4.4 | `dense-retrieval-20260902-234630/` |
| arrival patterns, global-core rule | `findings.md` §4.5 | `eval-locality-20260910-231437/` |
| k=10 and retrieve-64/show-10, dense | `how-many-tools.md` §5.1-5.2 | `eval-gaps-20260907-224433/` |
| retrieve-64/show-10, BM25 | `how-many-tools.md` §5.3 | `eval-bm25trunc-20260908-000327/` |
| whole-menu accuracy and F1, dense, six policies, three models | `metrics-and-latency-tradeoffs.md` §1-§3 · `how-many-tools.md` §4 · `findings.md` §4.4 | `dense-accuracy-20260903-002742/`, `eval-validity-20260906-165826/` |
| throughput under 1 s | `metrics-and-latency-tradeoffs.md` §4 | `eval-validity-20260906-165826/`, `eval-locality-20260910-231437/` |
| menu size and tools called | `how-many-tools.md` §3 | listed in its Runs table |
| random task slice | `findings.md` §4.5 | `eval-locality-20260910-231437/` |

Direct answers to the research questions are in [`answers.md`](answers.md).
The menu-size argument in full is in [`how-many-tools.md`](how-many-tools.md),
with a plain-language version in
[`how-many-tools-summary.md`](how-many-tools-summary.md).
