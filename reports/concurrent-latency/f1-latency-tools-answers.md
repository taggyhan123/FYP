# Answers: F1, latency limits, and how many tools

Five questions, each with its answer first and the evidence under it:

1. F1 → better accuracy → less KV-cache recompute?
2. The latency–precision tradeoff, and which configurations move it.
3. What accuracy do we get if every request must respond within 1 second?
4. How to get higher throughput under a latency limit, scored on precision and
   F1.
5. How many tools we use.

ToolTrie-v1's results are highlighted throughout.

**Setup.** Qwen3 on one RTX 3090, vLLM 0.26.0 with automatic prefix caching
unmodified, single-turn requests over the ToolRet catalogue (44,453 tools), 200
tasks per cell. Every number is a run under `cluster/results/`. Derivations are
in [`answers.md`](answers.md),
[`metrics-and-latency-tradeoffs.md`](metrics-and-latency-tradeoffs.md) and
[`how-many-tools.md`](how-many-tools.md). The response-time grid in §2-§3 was
computed for this document from the per-request timings in those same runs.

| # | Question | Answer |
|---|---|---|
| 1 | Does F1 → better accuracy → less KV-cache recompute? | **No.** F1 only measures accuracy, and accuracy does not change cache reuse. What cuts recompute is the ordering: ToolTrie-v1 recomputes the least, and F1 shows that costs no accuracy. |
| 2 | Is there a latency–precision tradeoff through configuration? | **Yes.** Model size and menu size trade latency for accuracy. Ordering is the one setting that improves latency without costing accuracy, and ToolTrie-v1 is the best choice of ordering. |
| 3 | If every request must respond within 1 s, what accuracy? | **It depends on what "respond" means.** Whole reply, every request: only 0.6B fits, best end-to-end F1 22.4 (16 tools, ToolTrie-v1 top). On average: 4B, 27.2. First token only: 8B, 29.6. |
| 4 | Optimise for higher throughput under a latency limit, scored on precision/F1? | **ToolTrie-v1** sustains the most traffic within the limit and has the highest precision and F1 there. It is the only policy best on both. |
| 5 | How many tools do we use? | Menus of 4, 16, 64 and 128 tools, plus 10 and "retrieve 64, show 10". Tasks need about 1.5 tools, and the model calls about 1 whatever the menu size. **Recommended evaluation: retrieve 64, show 10.** |

---

## 1. Does F1 → better accuracy → less KV-cache recompute?

**No. The chain breaks at both links.**

- **F1 → better accuracy:** F1 is a way of *measuring* accuracy. Switching to
  it changes how precisely accuracy is reported. It does not change what the
  model does.
- **Better accuracy → less recompute:** reuse is decided when the request
  arrives. vLLM matches the prompt against prefixes already cached, and that
  happens **before the model generates anything**. Accuracy is a property of
  what the model generates afterwards, so it cannot reach back. Requests here
  are single-turn and independent, so one request's answer never enters
  another request's prompt either.

**The data shows reuse and accuracy moving independently:**

| case | reuse | accuracy |
|---|---|---|
| ToolTrie-v0, list cut to 10 | **16.12%**, 2.5x no reordering | **0.75** end-to-end F1, the worst |
| v1 vs ContextPilot at 0.6B | v1 caches ~2x more | v1 ahead by 5.5-9.2 F1 |
| v1 vs ContextPilot at 8B | v1 **still** caches ~1.7x more | **tied** |

**What does hold: the ordering drives both, separately.** The ordering decides
how much of the prompt matches earlier prompts, which is less recompute. It
also decides where the right tool lands, which is accuracy. ToolTrie-v1 is the
ordering that wins on the first without losing on the second. At 64 tools,
dense retriever, Qwen3-0.6B:

| policy | reuse (share of the prompt not recomputed) | precision | recall | F1 | end-to-end F1 |
|---|---|---|---|---|---|
| **ToolTrie-v1** | **6.22%** | **28.99** | **26.18** | **26.99** | **22.00** |
| no reordering | 0.82% | 27.45 | 24.95 | 25.66 | 20.92 |
| ContextPilot | 2.98% | 21.98 | 21.27 | 21.47 | 17.50 |
| frequency | 1.39% | 21.47 | 19.12 | 19.84 | 16.17 |
| ToolTrie-v0 | 2.40% | 17.79 | 15.64 | 16.36 | 13.33 |
| alphabetical | 1.52% | 17.79 | 15.64 | 16.36 | 13.33 |

**v1 is top in every column.** It recomputes the least of any policy and has
the highest precision, recall and F1. So F1's real job here is to **prove the
reuse was not bought with accuracy**. ContextPilot's reuse cost about 4 points
of F1 against no reordering (21.47 vs 25.66).

**Why F1 rather than plain accuracy.** 44% of our tasks (87 of 200) need two or
more tools. The old metric (did the model call *any* right tool) ignores that,
and ignores wrong extra calls. F1 counts both: precision is correct calls ÷ calls made,
recall is correct calls ÷ tools needed. It is the field's standard for
multi-tool tasks. **End-to-end F1** additionally scores a request as zero when
the right tool was never shown, so it is the primary metric wherever policies
show different tools.

*One boundary:* in multi-turn traffic the model's own output enters the next
turn's prompt, so accuracy *could* affect reuse there. Untested.

*Derived in:* `answers.md` §1 · `metrics-and-latency-tradeoffs.md` §1-§2 ·
`findings.md` §4.4.

---

## 2. The latency–precision tradeoff, and which configurations move it

**Yes, there is a tradeoff. Four configuration knobs:**

- two measured tradeoffs: model size and menu size;
- one knob with no tradeoff: the ordering policy;
- one untested: quantization.

The full grid, ToolTrie-v1 on the dense retriever, one request at a time. Time
is the whole reply, up to 128 output tokens:

| model | tools shown | median reply | 95th percentile | slowest | **end-to-end F1** |
|---|---|---|---|---|---|
| 0.6B | 4 | 134 ms | 260 ms | 363 ms | 20.50 |
| 0.6B | 10 | 155 ms | 322 ms | 456 ms | 19.58 |
| 0.6B | 16 | 178 ms | 346 ms | 501 ms | **22.42** |
| 0.6B | 64 → 10 | 149 ms | 314 ms | 445 ms | 19.92 |
| 0.6B | 64 | 464 ms | 795 ms | 1,028 ms | 22.00 |
| 0.6B | 128 | 991 ms | 1,856 ms | 3,252 ms | 21.70 |
| 4B | 4 | 604 ms | 1,389 ms | 1,696 ms | 26.17 |
| 4B | 10 | 729 ms | 1,283 ms | 1,884 ms | 27.07 |
| 4B | 16 | 817 ms | 1,507 ms | 2,043 ms | 27.17 |
| 4B | 64 → 10 | 697 ms | 1,342 ms | 1,885 ms | 26.03 |
| 4B | 64 | 1,889 ms | 3,087 ms | 4,119 ms | 27.95 |
| 4B | 128 | 3,729 ms | 6,227 ms | 9,267 ms | **29.82** |
| 8B | 16 | 1,386 ms | 2,301 ms | 3,379 ms | 29.62 |
| 8B | 64 | 2,976 ms | 4,498 ms | 6,051 ms | 28.78 |
| 8B | 128 | 5,571 ms | 8,667 ms | 11,679 ms | **30.03** |

"64 → 10" means retrieve 64 tools, reorder, and show the model the first 10.

**Model size is the big accuracy lever, with sharply diminishing returns:**

| step | extra end-to-end F1 | cost in median reply time |
|---|---|---|
| 0.6B → 4B | **+4.8 to +8.1** | 3.8-4.6x |
| 4B → 8B | +0.2 to +2.5 | +49-70% |

On this task, 8B is close to pure cost.

**Menu size is the big latency lever, and on small models it is almost free.**
At 0.6B, end-to-end F1 stays between 19.6 and 22.4 from 4 to 128 tools, while
the median reply grows 7x. Showing fewer tools buys speed at no accuracy cost.
At 4B, a bigger menu does buy a little accuracy: 26.2 at 4 tools rises to 29.8
at 128, at 6x the time.

**Ordering is the knob with no tradeoff, and ToolTrie-v1 is its best
setting.** With no queue, every policy's time is within 1-2% at a fixed menu
size, because the time is spent reading the prompt and that depends on its
length, not its order. So ordering changes the *accuracy* inside a given time
(§3). Under load it also changes the time itself:

- v1's median time to first token is 10-20% lower than the other policies at
  64 tools;
- with the list cut to 10, it is 13% lower than no reordering and 6% lower
  than ContextPilot, with the right tool kept in view as often as no
  reordering;
- v1 costs no accuracy in any of these (§1, §4).

**Quantization** (fp8, int8, int4) was never used; everything ran at native
bf16. It is the most obvious untested knob. It would be the natural way to get
4B's accuracy inside a 1-second limit (§3).

*Derived in:* `metrics-and-latency-tradeoffs.md` §3-§4 · `how-many-tools.md`
§3, §5 · `findings.md` §4.4, §5.

---

## 3. If every request must respond within 1 second, what accuracy?

**It depends on what "respond" means, and the strict reading changes the
answer.** All measured one request at a time, with no queue. Under load, see
§4.

| the 1-second limit applies to | settings that fit | best accuracy that fits |
|---|---|---|
| **the whole reply, for every request** | only 0.6B, at 4, 10 or 16 tools, or 64 → 10 (64 tools misses by one request: 1,028 ms) | **22.42** end-to-end F1 (0.6B, 16 tools) |
| the whole reply, for 95% of requests | 0.6B up to 64 tools; no 4B setting | 22.42 (0.6B, 16 tools) |
| the whole reply, **on average** | 0.6B up to 64 tools; 4B up to 16 tools | **27.17** (4B, 16 tools) |
| **the first token, for every request** | 0.6B up to 64 tools; 4B up to 16 tools; 8B at 16 tools | **29.62** (8B, 16 tools) |

- **4B cannot keep every reply under 1 s at any menu size tested.** At 16
  tools, 54 of 200 replies take longer (the slowest 2.0 s). Even at 4 tools,
  21 do. The slow ones are replies that generate more output: 77 tokens on
  average, against 39 for the replies under 1 s (16 tools). Earlier documents'
  "4B fits 16 tools" is the *average* reading.
- **8B fits only if the limit is on the first token.** At 16 tools every
  request starts answering within 963 ms, but its whole replies average 1.4 s.

**Which ordering gives the best accuracy inside the limit.** Ordering does not
change time here, so each policy fits the same settings, and the difference is
accuracy. End-to-end F1 at each reading's best setting:

| policy | whole reply, every request (0.6B, 16 tools) | whole reply, average (4B, 16 tools) | first token, every request (8B, 16 tools) |
|---|---|---|---|
| **ToolTrie-v1** | **22.42** | 27.17 | 29.62 |
| no reordering | 22.08 | 27.03 | **31.12** |
| frequency | 19.25 | **27.20** | not run |
| ContextPilot | 20.17 | 24.58 | 28.12 |
| ToolTrie-v0 | 17.62 | 24.67 | 28.48 |
| alphabetical | 17.62 | 25.00 | not run |

- **Under the strict limit, ToolTrie-v1 gives the highest accuracy of any
  policy**: +2.25 over ContextPilot, +0.34 over no reordering.
- On the average reading it ties the top (27.17 against frequency's 27.20),
  +2.59 over ContextPilot.
- At 8B the policies converge. No reordering leads v1 by 1.5 (inside noise at
  200 tasks), and v1 is still ahead of ContextPilot.
- **v1 beats ContextPilot under every reading.**

**Retrieve 64, show 10** fits the strictest reading too: every 0.6B reply is
under 445 ms. v1 keeps the right tool in view in 61.5% of requests there,
against ContextPilot's 47.0% (end-to-end F1 19.92 vs 16.25).

*Derived in:* per-request timings of the accuracy runs
(`dense-accuracy-20260903-002742/`, `eval-validity-20260906-165826/`,
`eval-gaps-20260907-224433/`), scored with `scripts/score_end_to_end.py`.
Earlier documents quote per-request time as total run time ÷ 200 (~510 ms for
0.6B at 64 tools), which also counts overhead between requests. The medians
above are per request.

---

## 4. Higher throughput under a latency limit, scored on precision and F1

**Fix the latency limit, find the most traffic each policy sustains within it,
and report the accuracy delivered there.** Here the limit is median time to
first token under 1,000 ms, 64 tools, dense retriever, Qwen3-0.6B:

| policy | most traffic within the limit | precision | F1 | end-to-end F1 |
|---|---|---|---|---|
| frequency | 2.570 req/s | 21.47 | 19.84 | 16.17 |
| alphabetical | 2.574 | 17.79 | 16.36 | 13.33 |
| ToolTrie-v0 | 2.585 | 17.79 | 16.36 | 13.33 |
| no reordering | 2.623 | 27.45 | 25.66 | 20.92 |
| ContextPilot | 2.638 | 21.98 | 21.47 | 17.50 |
| **ToolTrie-v1** | **2.671** | **28.99** | **26.99** | **22.00** |

**ToolTrie-v1 is the only policy best on both axes at once.** Every other
policy trades one for the other:

- frequency sustains the least traffic;
- no reordering nearly matches v1's accuracy but sustains less;
- ContextPilot is middling on both.

**When requests arrive in bursts, the margin grows:**

| policy | natural arrival | bursty arrival |
|---|---|---|
| no reordering | 2.623 req/s | 2.608 |
| ContextPilot | 2.638 | 2.662 |
| **ToolTrie-v1** | 2.671 | **2.817** |
| v1 over ContextPilot | +1.2% | **+5.8%** |

At 2.75 req/s with bursty arrival, v1 answers in 821 ms and meets the limit;
ContextPilot takes 1,184 ms and misses it.

**Why v1 wins.** It caches the most (6.22% of each prompt against
ContextPilot's 2.98%), so each request's prefill is shorter. The queue then
drains faster and more requests fit under the limit. It keeps the retriever's
order, so precision and F1 stay at the top.

**How to optimise the system for throughput under a latency limit, in order of
impact:**

1. **Menu size first.** It is the biggest lever: fewer tools shorten every
   prompt, and at 0.6B this costs no end-to-end accuracy (§2). At 10 tools
   every policy answers in 66-68 ms under light load.
2. **Then ToolTrie-v1 ordering.** It gives 1.2-5.8% more traffic than
   ContextPilot at the highest precision and F1. With the list cut to 10, its
   median is 60 ms against 64 for ContextPilot and 69 for no reordering, with
   the right tool kept as often as no reordering.
3. **Not shortest-job-first queuing if the limit is on the tail.** It cuts the
   median by 85% but makes the slowest requests 127% slower (with aging 250:
   −47% median, +46% slowest). It helps a median limit and hurts a p95 or
   worst-case one, and no setting improves both.

**Caveats:**

- This limit is on the median. Limits are usually written on p95, which binds
  far earlier: at 2.5 req/s every policy's p95 is already 2,000-2,600 ms.
- One menu size, one model.
- The spread between policies is small, about 4% best to worst.

*Derived in:* `metrics-and-latency-tradeoffs.md` §4 · `findings.md` §4.2,
§4.5 · `how-many-tools.md` §5.

---

## 5. How many tools do we use?

This is already answered in full in [`how-many-tools.md`](how-many-tools.md),
with a plain-language version in
[`how-many-tools-summary.md`](how-many-tools-summary.md). The short version:

**Four numbers that are easy to confuse:**

| | what it means | value |
|---|---|---|
| catalog | every tool that could be retrieved | 44,453 |
| **menu (k)** | the tools put in the prompt | **4, 16, 64, 128**, plus 10 and "retrieve 64, show 10" |
| needed | the tools a task actually requires | 1.54 on our tasks (1.77 across the corpus) |
| called | the tools the model actually calls | **about 1** (0.70-1.10), whatever the menu size |

**Why the menu is bigger than what a task needs.** The system does not know
*which* 1-2 tools a task needs; that is retrieval's job, and it is imperfect.
The menu is a bet that the right tool is in it. ToolTrie-v1, 0.6B, dense:

| tools in the menu | right tool is in the menu | tools called per request | end-to-end F1 |
|---|---|---|---|
| 4 | 52.5% | 0.70 | 20.50 |
| 16 | 67.5% | 0.80 | 22.42 |
| 64 | 81.5% | 0.88 | 22.00 |
| 128 | 85.5% | 0.90 | 21.70 |

- At 4 tools the right tool is missing from almost half of all requests.
  Nothing downstream can call a tool that was never shown.
- Bigger menus contain it more often but distract the model, so **end-to-end
  F1 is flat across a 32x range of menu size**.
- The model under-calls: it needs about 1.5 tools and calls about 1. When a
  menu is small, the requests where it called nothing mostly had no right tool
  to call (62% at 4 tools). When a menu is large, they mostly had it in front
  of the model (90% at 64 and 128 tools). **The failure moves from retrieval
  to the model as the menu grows.**

**What we decided.** Three questions, three answers:

| purpose | menu size | why |
|---|---|---|
| comparable to published benchmarks | 10 | BFCL, ToolBench, ToolRet and LiveMCPBench all show 5-11 |
| realistic for deployments with no router | ~150 | several ordinary MCP integrations together; sits between our 64 and 128 |
| able to see ordering effects at all | large, or cut | at 10 tools with nothing cut, every policy is identical (66-68 ms) |

**Recommended evaluation: retrieve 64, show 10.** It is the only design that
satisfies all three purposes at once:

- **comparable** to the field: 10 tools shown;
- **realistic**: a router picking a shortlist from a larger pool;
- **discriminating**: a 15-20pp gap between policies in how often the right
  tool survives;
- **fast**: every 0.6B reply under 445 ms, so it fits even the strictest
  1-second reading (§3);
- **where ToolTrie-v1 shows its main advantage**: it keeps the right tool in
  view as often as no reordering, with 4.3x the reuse and 13% lower latency,
  while ContextPilot loses the right tool in one request in five to seven.

*Derived in:* `how-many-tools.md` §1-§6.
