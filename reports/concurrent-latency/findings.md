# Tool ordering under concurrent load

Qwen3-0.6B on one RTX 3090 (plus a Qwen3-4B check), vLLM 0.26.0, prefix caching
unmodified. 187 GPU runs. Raw outputs are in the git-ignored `cluster/results/`
directories listed in the Appendix.

---

## Summary

Four questions, none previously run in this project.

| # | Question | Answer | Where |
|---|---|---|---|
| 1 | Tail latency under parallel load | Ordering matters far more than it did serially — but only on padded menus | §1 |
| 2 | An adaptor trading mean against max | Works; not worth deploying. Ordering is worth 6.7x more | §2 |
| 3 | How the trie reduces parallel latency | By raising cache reuse, which compounds under load into higher capacity | §3 |
| 4 | Cutting tail latency by smart queuing | No. It cuts the median and *raises* the tail | §4 |

### What we found

**Parallel load separates the orderings; serial testing hid this.** At 4 req/s
ToolTrie cuts both p50 and p95 by 96.5% against unordered menus (a padded
workload — see §1.5 and §4.1 for how much of that generalises), and
sustains 11.24 req/s against 3.56 — 3.2x the capacity. Serially the same
statistics were flat within 1.25x.

**Most of that is the test workload, not the method.** The padded menus give
every request 63 of the same 64 tools. On real BM25-retrieved menus the gap
between arms collapses: reuse spread 94.97pp → 0.98pp, p50 spread 36x → 1.02x.

**Reordering costs accuracy.** On retrieved menus ToolTrie loses 6.8 points of
tool-selection accuracy at k128 and 11.3 at k64. ContextPilot at k128 is the one
policy that gains reuse at zero accuracy cost.

**The reason is position.** Accuracy tracks how far down the menu the correct
tool is pushed. The cache wants the *common* tools first; the model wants the
*relevant* tools first; prefix caching only reuses a leading prefix. Both
compete for the front of the prompt.

**ToolTrie places second on retrieved menus and ties on padded ones.** It beats
every simple heuristic — unordered, alphabetical, frequency — everywhere.
ContextPilot beats it at all four retrieval depths, by 1.07x to 2.52x — but the
k64 figure is the top of its range: over five arrival orders it spans 1.09x to
2.57x, the other four at 1.09-1.39x (A.6). On padded menus its 9-point deficit is
warm-up: by request 200 the two are 0.04pp apart and ToolTrie is marginally
faster (§1.5).

### What to be careful with

- Parts 1–3 describe a padded workload. §4 is the check on that, and it is the
  most important limit in this report.
- One model, one trial per cell. Eight cells were re-run under 3–5 arrival
  permutations (A.6); every other cell rests on one.
- `max` is a single sample at n=200 and moved 269 → 710 ms between two runs of
  the same configuration. Use p50 and p95.

---

## Setup

**Why tool order affects latency.** An agent request carries a menu of tool
schemas — often thousands of tokens — before the user's question. vLLM caches
the work done on a prompt and reuses it for any later prompt starting with the
same tokens. That reuse only applies to a *leading* prefix, so if two requests
share tools but list them in different orders, nothing is reused. Reordering the
menu so shared tools come first is therefore worth latency, and that is what
every arm below tries to do.

**Two workloads, and the difference matters more than any policy.**

| | how the menu is built | tools shared between any two requests |
|---|---|---|
| **padded-64** | each task's real tools, padded out to 64 from a fixed filler set | **63 of 64** |
| **retrieved (k4–k128)** | BM25 retrieves the top *k* tools for that query | 0.06 (k4) to 8.3 (k128) |

Padded menus are nearly identical to each other, so there is a large shared
prefix waiting to be exploited. Retrieved menus genuinely differ per query, so
there is little to exploit. Parts 1–3 use padded menus; §4 repeats the
comparison on retrieved ones, and that contrast is the report's main limit.

**The arms** — five ways of ordering the same menu:

| arm | what it does |
|---|---|
| `original` | leaves the retriever's relevance ranking alone — the do-nothing baseline |
| `alphabetical` | sorts by tool name |
| `frequency` | most-used tools first |
| `tooltrie_v0` | this project's method: a trie that grows a shared prefix across requests |
| ContextPilot | clusters requests by which tools they share, then hoists the shared set |

Fourteen frozen orderings exist; four were replayed under load, chosen to span
the distinct reuse levels — five of the unused arms sit at an identical 39.69%
and two more at 1.19%, so replaying them would duplicate curves.

**What is measured.**

- **Reuse** — the share of prompt tokens served from cache instead of recomputed.
  This is what ordering controls directly.
- **Latency** — time to first token, timed client-side from the first streamed
  chunk. Where a policy holds requests in a queue before sending them (§2, §4),
  the figure reported is **arrival to first token**, which includes that wait;
  measuring from dispatch would hide the delay the policy itself causes.
- **Accuracy** (§1.4) — whether the model still calls a correct tool afterwards.

**How runs were done.** 200 requests per run, replaying frozen orderings so every
arm is a permutation of the same menus. Open-loop Poisson arrivals at a fixed
rate, seed 42, cache cleared before each run, decode pinned to 48 tokens so
decode-length variance cannot pollute the tails. 187 runs total.

**One naming note.** ContextPilot throughout is the official reordering API at
the paper's `alpha=0.001`, ordering only — no annotations, de-duplication or
scheduling, which its paper credits for roughly half its cache gain. Earlier
drafts of this report used a project-built variant at `alpha=0.5`; those runs
have been replaced. Two appendix tables (capacity, order control) still rest on
it, noted where they appear. Appendix A.4 has the detail.

---

## 1. Latency under parallel load

### 1.1 Results

Time to first token, ms. Both workloads at 4 req/s, and they are size-matched —
6,903 tokens per request padded against 6,896 retrieved — so the only difference
is whether the menus genuinely overlap.

**Padded menus** (63 of 64 tools shared)

| arm | achieved | reuse | p50 | p95 | max |
|---|---|---|---|---|---|
| original | 3.5645 | 1.19% | 3310.4 | 6498.6 | 7476.0 |
| alphabetical | 4.0081 | 38.13% | 331.8 | 1058.1 | 1724.9 |
| frequency | 4.0076 | 39.69% | 317.9 | 1028.9 | 1521.4 |
| tooltrie_v0 | 4.0139 | 87.19% | 116.3 | 225.7 | 558.4 |
| **ContextPilot** | 4.0143 | **96.16%** | **92.7** | **129.0** | **300.2** |

**Retrieved menus, same size** (2.8 of 64 tools shared)

| arm | achieved | reuse | p50 | p95 | max |
|---|---|---|---|---|---|
| original | 2.6346 | 0.91% | 9667.9 | 24699.7 | 26820.4 |
| alphabetical | 2.6252 | 1.22% | 9727.5 | 24854.3 | 27104.6 |
| frequency | 2.6230 | 0.94% | 9760.9 | 25009.1 | 27161.5 |
| tooltrie_v0 | 2.6286 | 1.90% | 9559.9 | 24825.6 | 27059.1 |
| **ContextPilot** | 2.6934 | **4.78%** | **8565.4** | **22452.4** | **25432.4** |

**Ordering is decisive on padded menus and nearly irrelevant on retrieved ones.**
p50 spans **36x** across the padded arms and **1.14x** across the retrieved ones,
at the same request size and the same offered rate. Every retrieved arm also
saturates at ~2.6 req/s regardless of policy. §4 pursues this; it is the most
important qualification in the report, so it appears here rather than only there.

The rest of Part 1 uses padded menus, where the arms separate enough to study.

**Padded menus across load** — the same table at three rates:

| rate | arm | achieved | reuse | p50 | p95 | max |
|---|---|---|---|---|---|---|
| 1 | original | 1.0066 | 1.19% | 266.4 | 431.5 | 622.4 |
| 1 | alphabetical | 1.0071 | 38.13% | 230.7 | 364.4 | 493.5 |
| 1 | tooltrie_v0 | 1.0075 | 87.19% | 110.1 | 162.1 | 343.8 |
| 1 | ContextPilot | 1.0075 | 96.16% | **92.0** | **115.3** | 307.0 |
| 2 | original | 2.0087 | 1.19% | 284.9 | 710.8 | 1208.2 |
| 2 | alphabetical | 2.0106 | 38.13% | 246.5 | 486.7 | 896.0 |
| 2 | tooltrie_v0 | 2.0124 | 87.19% | 114.6 | 177.5 | 363.7 |
| 2 | ContextPilot | 2.0124 | 96.16% | **89.5** | **119.0** | **274.6** |
| 4 | original | **3.5645** | 1.19% | 3310.4 | 6498.6 | 7476.0 |
| 4 | alphabetical | 4.0081 | 38.13% | 331.8 | 1058.1 | 1724.9 |
| 4 | tooltrie_v0 | 4.0139 | 87.19% | 116.3 | 225.7 | 558.4 |
| 4 | ContextPilot | 4.0143 | 96.16% | **92.7** | **129.0** | **300.2** |

ToolTrie against the arms it replaces, padded:

| rate | vs original | vs alphabetical |
|---|---|---|
| 1 | p50 −58.7%, p95 −62.4% | p50 −52.3%, p95 −55.5% |
| 2 | p50 −59.8%, p95 −75.0% | p50 −53.5%, p95 −63.5% |
| 4 | **p50 −96.5%, p95 −96.5%** | p50 −64.9%, p95 −78.7% |

ContextPilot leads ToolTrie at every rate here, and ToolTrie's median penalty is
18-25 ms. That lead is warm-up, not ordering quality: over a 600-request run the
two are 0.04pp apart in reuse once ToolTrie has converged, and the latency
penalty reverses to −0.5 ms (§1.5).

### 1.2 Three things serial testing could not show

**`max` reverses.** Serially it was flat across arms (~270 ms, 1.25x spread).
At 4 req/s it spans 308 to 7476 ms — 24.3x. Max is dominated by queueing, which
concurrency of 1 cannot produce.

**Reuse becomes capacity.** Sustained throughput at 64 req/s offered:

| arm | reuse | ceiling |
|---|---|---|
| original | 1.19% | **3.57 req/s** |
| alphabetical | 38.13% | 4.74 |
| tooltrie_v0 | 87.19% | 11.24 |
| **ContextPilot** | 96.16% | **16.83** |

Ordering alone is worth **4.7x the admission capacity** end to end, and ToolTrie
3.1x over no reordering. Reuse was identical at every rate from 1 to 64, so
ordering sets reuse and load does not change it.

**The slow requests are different requests.** At rate 4 the five slowest are:

| arm | slowest indices (of 200) | meaning |
|---|---|---|
| original | 198, 193, 197, 192, 196 | the *last* arrivals — the backlog grows without bound |
| alphabetical | 53, 52, 50, 47, 51 | a bounded mid-run spike |
| tooltrie_v0 | 3, 4, 2, 5, 46 | warm-up |
| ContextPilot | 2, 1, 0, 3, 4 | cold start only |

ContextPilot's tail is a fixed startup cost, so its p95 moves 115.3 → 129.0 ms
across a 4x load increase while `original` goes 431.5 → 6498.6.

### 1.3 Larger model

Qwen3-4B, same frozen orderings, native cache 96,400 tokens.

Reuse is unchanged by model size — 1.19 / 37.99 / 87.21 / 96.16% against
1.19 / 38.13 / 87.19 / 96.16% at 0.6B. The two small differences have
non-model explanations: `alphabetical` is the one capacity-sensitive arm and 4B
has a smaller cache; the ToolTrie run lost one request to a client socket error.

**The gap grows with model size**, so the 0.6B figures understate it. Against
`alphabetical`, ContextPilot's lead goes from **2.51x to 12.86x**, because the
fixed ~16 ms client overhead is 18% of its latency at 0.6B but 3% at 4B.

Under load at 4B the ranking holds and widens sharply:

| rate | ContextPilot p50 | ToolTrie p50 | ratio |
|---|---|---|---|
| 1 | 121.7 | 298.8 | 2.45x |
| 2 | 135.6 | 304.7 | 2.25x |
| 4 | **174.3** | **5301.8** | **30.41x** |

That 30x is a capacity effect, not a prefill effect: at 4 req/s ToolTrie's 12.81%
uncached prefill pushes it just past its service ceiling (3.8405 of an offered 4)
while ContextPilot's 3.84% keeps it just under (3.9757). A 9-point reuse gap
becomes the difference between coping and collapsing.

One claim does not transfer: ContextPilot's flat latency *shape* is specific to
0.6B. At 4B the last 1% of requests break away — p99/p50 is 8.62 against ~2.9 at
0.6B — while p95/p50 stays at 1.34. The blow-up is a handful of requests, which
is why the tables above stop at p95.

### 1.4 What the reordering costs

Latency and reuse say nothing about whether the model still picks the right
tool. On retrieved menus, often it does not.

**Accuracy here means: did the model call the right tool.** Each task has
labelled correct tools. The model gets the menu and the question, and either
emits a tool call or does not. Accuracy is the share that called a correct one,
counting only requests where a correct tool was actually in the menu — otherwise
the score would include the retriever's failures rather than the ordering's.

Worked through for `k64` with no reordering: of 200 requests, 134 emitted a tool
call at all, 151 had a correct tool present in the menu, and 56 of those 151
called it — accuracy 37.09%. Every arm holds identical menus, so that 151 is the
same for all of them and the comparison isolates ordering. That ceiling was
checked across arms before the results were used, and it matched.

So "ToolTrie −11.3pp at k64" means that of the same 151 answerable requests, no
reordering got 56 right and ToolTrie's ordering got 39.

ContextPilot here is the official-API arm at the paper's alpha (§Setup).

| depth | arm | reuse | accuracy | vs original | mean position of correct tool |
|---|---|---|---|---|---|
| k128 | original | 0.37% | **22.98%** | — | **11.5** |
| k128 | ContextPilot | 1.99% | **22.98%** | **0.0pp** | 27.8 |
| k128 | tooltrie_v0 | 1.13% | 16.15% | **−6.8pp** | 62.8 |
| k64 | original | 0.91% | **37.09%** | — | **6.1** |
| k64 | ContextPilot | 4.78% | 27.15% | −10.0pp | 12.5 |
| k64 | tooltrie_v0 | 1.90% | 25.83% | **−11.3pp** | 31.5 |

**ToolTrie's reuse is paid for in accuracy** — 6.8 points at k128, 11.3 at k64.
The project's earlier quality work is all on padded menus; this appears to be
the first measurement on retrieved ones.

**ContextPilot at k128 is the exception**: 5.4x the reuse of no reordering, at
zero accuracy cost, and without the order annotations its paper adds for this
purpose.

**Accuracy tracks position, not policy.** The further down the menu the correct
tool is pushed, the worse the model does — the lost-in-the-middle effect
ContextPilot's paper cites as the reason reordering costs accuracy. ContextPilot
keeps that tool shallowest of any reordering arm.

This is the central tension in the problem: **the cache wants the common tools
first, the model wants the relevant tools first, and prefix caching only reuses a
leading prefix.** Padded menus hide it, because there "common" is 63 of 64 tools.

Caveat: a 0.6B model may be unusually sensitive to ordering. ContextPilot reports
newer, larger models being nearly insensitive to it.

---

### 1.5 The padded gap between ToolTrie and ContextPilot is warm-up

Every padded figure above comes from a 200-request run, and ToolTrie is still
learning for most of it. Extending the workload to 600 requests by its own
construction rule — the same 63-tool core in the same order, one fresh singleton
per request — and differencing the 600-run against the 200-run of the same file
(their first 200 records are byte-identical, both planners being causal):

| requests | ToolTrie | ContextPilot | gap |
|---|---|---|---|
| 1–200 (every other table here) | 87.19% | 96.16% | **8.97pp** (1.103x) |
| 1–600 | 93.76% | 96.78% | 3.02pp |
| **201–600** | **97.05%** | **97.09%** | **0.04pp** (1.000x) |

Uncached prefill over 201–600 is 2.95% against 2.91% — 1.01x the work, where
over the first 200 it was 3.33x. Latency follows, and the residual penalty
reverses:

| padded @4, TTFT ms | ToolTrie | ContextPilot |
|---|---|---|
| p50, requests 1–200 | 116.0 | **93.9** |
| p50, requests 201–600 | **91.2** | 91.7 |
| p95, requests 201–600 | **122.5** | 123.7 |
| max, requests 201–600 | **141.9** | 151.1 |

Measured over all 600 requests ToolTrie's p50 is 93.2 ms, still carrying its own
warm-up; the 201–600 window is the one that answers the question.

The ordering explains it exactly. Measured as the position of the singleton,
where 63 is optimal:

| requests | ToolTrie | ContextPilot |
|---|---|---|
| 1–200 | 56.31 | 62.69 |
| 201–400 | 62.90 | 63.00 |
| 401–600 | **63.00** | **63.00** |

ContextPilot is optimal from request 51; ToolTrie takes about 150. Both finish
at the same place. **So ContextPilot's advantage on padded menus is cold-start
speed, not ordering quality** — a real advantage for short or bursty workloads,
but not the one §1.1 appears to report.

This is forced once the orderings converge: on padded menus reuse *is* the
singleton's position (§4.1), so two policies that both place it at 63 must tie.

Two limits. The 400 added requests pair real tool schemas with borrowed queries,
so they are valid for reuse and ordering but carry no accuracy signal. And this
says nothing about retrieved menus, where there is no common core to converge to
and the §4.1 ranking stands.

---

## 2. The mean-versus-max adaptor

Both policies sit behind a shared in-flight cap, so the only difference is which
waiting request goes next. `fifo` takes them in arrival order. `affinity` takes
the one sharing the longest leading tool prefix with the last request sent,
unless something has waited longer than `D` seconds.

### 2.1 It works, and `D` is the knob

`alphabetical`, rate 4, cap 4. Arrival to first token, ms.

| config | p50 | max | change vs fifo |
|---|---|---|---|
| `fifo` | 1046.0 | 3120.2 | — |
| `D=0.5` | 1051.8 | 3084.7 | no gain, no damage |
| **`D=2.0`** | **827.2** | **3302.6** | **p50 −20.9%, max +5.8%** |
| `D=0` | **715.9** | **12136.4** | p50 −31.6%, max **+289%** |

`D=2.0` is the operating point. `W`, the window size, is not a knob — 8 and 32
are indistinguishable.

The gain is not a caching effect: reuse is unchanged (36.65% → 36.64%). Grouping
similar requests drains the queue faster for most, while the ones repeatedly
skipped accumulate the 12-second tail.

### 2.2 Not worth deploying

All at 4 req/s behind an in-flight cap of 4. Arrival to first token, ms.

| policy | p50 | max | reordered |
|---|---|---|---|
| `frequency` | 825.2 | 2640.3 | — |
| `frequency` + adaptor | **685.2** (−17.0%) | **2961.4** (+12.2%) | 105/200 |
| `alphabetical` | 1046.0 | 3120.2 | — |
| `alphabetical` + adaptor | **827.2** (−20.9%) | **3302.6** (+5.8%) | 104/200 |
| `tooltrie_v0` | **122.8** | **915.6** | — |
| `ContextPilot` | **94.2** | **452.6** | — |

The adaptor works on both orderings where it can act, and with the same shape:
median down, worst case up. It reordered around half the requests in each.

**ToolTrie and ContextPilot are shown without it because it cannot act on them** —
0 of 200 reordered in both cases, so running it issues the identical dispatch
sequence FIFO does. Adding it to ContextPilot gave 94.4 ms against 94.2, two
samples of the same thing.

**And the comparison that settles it: the best the adaptor achieves anywhere is
685.2 ms, against 122.8 for ToolTrie and 94.2 for ContextPilot, neither of which
uses it.** Fixing the ordering is worth 5.6x to 7.3x more than fixing the
dispatch order.

**That is inertness, not a broken policy** — checked by reconstructing the
pending queue at every dispatch and recomputing what the policy was choosing
between:

| | ToolTrie @ 16 req/s | alphabetical @ 4 req/s |
|---|---|---|
| dispatches with 2+ candidates | 195 | 131 |
| **all candidates tied** | **194** | 12 |
| scores actually varied | **1** | **119** |
| picked a top-scoring candidate | 1 of 1 | **118 of 119** |

On ToolTrie the policy faced a genuine decision once in 195 opportunities and got
it right. On `alphabetical` it faced 119 and got 118 right. It selects correctly
whenever there is anything to select; on ToolTrie there almost never is.

**A tradeoff is still achievable there — just not an intelligent one.** The
`random` control on the same ToolTrie runs reordered 193 of 200 and produced the
shape an adaptor is supposed to produce:

| ToolTrie @ 16 req/s | p50 | p95 |
|---|---|---|
| fifo | 8814 | 13457 |
| random | **6427 (−27.1%)** | **19665 (+46.1%)** |

So a dispatcher can move latency around on ToolTrie. What it cannot do is move it
around for a *reason*, because every signal it might use is flat:

| signal | on padded ToolTrie | usable |
|---|---|---|
| prefix affinity | every candidate ~56 of 64 tools, spread 0.08 | no |
| job size | every request ~5,570 tokens, CV 0.0038 | no |
| none (random) | — | acts, but the gain is the saturation artifact of §4.3, not information |

**ToolTrie has already made every request look alike, which is exactly why it is
fast and exactly why no dispatcher can improve on it.** Where a real signal does
exist — retrieved menus, where job sizes vary 5.3x — size-aware queuing does work,
and §4.3 measures it: median −85.5%, of which 29.2 points survive subtracting the
random baseline. But it buys the median at the tail's expense, and no setting
improves both.

The structural problem is worse than the arithmetic. **The adaptor needs
variance in affinity, not affinity**, and it is inert at both ends of the range.
It picks the first candidate with a strictly longer shared prefix, so when
candidates tie it falls back to arrival order:

| arm | reuse | mean prefix score | spread among candidates | all-tied windows | usable |
|---|---|---|---|---|---|
| original | 1.19% | 0.3 | 0.33 | **99.5%** | no |
| alphabetical | 38.13% | 15.5 | **13.23** | 27.2% | **yes** |
| frequency | 39.69% | 16.7 | **12.77** | 27.7% | **yes** |
| tooltrie_v0 | 87.19% | 56.4 | 0.08 | **99.5%** | no |
| ContextPilot | 96.16% | 62.7 | **0.01** | **99.5%** | no |

**Usability is an inverted U in ordering quality.** On `original` every candidate
scores 0. On ToolTrie and ContextPilot every candidate scores 56 and 63
respectively, with spreads of 0.08 and 0.01 — the shared prefix is so uniformly
long there is nothing to choose between. Only the two mid-table arms have prefix
lengths that vary, and they sit at ~39% reuse, less than half of what ToolTrie
reaches. **The adaptor is usable only on orderings that would not be deployed.**

The reason generalises beyond prefix affinity. A good ordering makes every request
share a long prefix, which makes requests interchangeable from the cache's point
of view, and interchangeable requests give a dispatcher nothing to decide. On
padded ToolTrie every dispatcher signal is flat: affinity spread 0.08, job-size
CV 0.0038, and shortest-uncached-suffix reduces to tokens x (1 - 56/64), again
near-constant. **Ordering and dispatch scheduling are substitutes rather than
complements, and ordering wins by 6.7x.**

This was checked directly rather than inferred. ToolTrie was re-run at 16 req/s,
above its 11.24 ceiling, forcing an 8.8-second client queue — and the adaptor
still reordered **0 of 200**, while a `random` control on the same runs reordered
193 of 200. The queue was real; the policy simply had no basis to act on.

**So it is inert where it is needed and inert where it would appear to fit.**

This null is also weak evidence. The design is a poor version of a standard
algorithm (longest prefix match), and it was tested on a single replica with a
cache large enough to hold everything — the one setting where the mechanism
cannot pay. Published gains come from routing across replicas. Details and
references in Appendix A.3.

---

## 3. How the trie reduces parallel latency

**Prefix caching is sequential.** A request can only reuse a prefix that an
earlier request has already computed and stored. Requests overlapping in time
cannot help each other — they all miss the same cold prefix and each recomputes
it.

Requests already in flight when the first one finishes:

| offered req/s | original | tooltrie_v0 | ContextPilot |
|---|---|---|---|
| 4 | 29 | 7 | 6 |
| 8 | 63 | **37** | **16** |
| 16 | 113 | **109** | **55** |
| 32 | **200** | **200** | 121 |

At 32 req/s **every one of ToolTrie's 200 requests starts before any has
finished** — the whole run is cold, as does the unordered baseline. ContextPilot
never does, even at 64.

This compounds: lower reuse means slower requests, which means more arrive
before the first finishes, which means more redundant work. ToolTrie's pile-up
is consistently about twice ContextPilot's, so a 9-point reuse gap becomes a 2x
difference in how many requests get no cache benefit at all.

The same effect shows in warm-up. Mean latency by arrival position at rate 4:

| arm | reqs 0–3 | 3–6 | 6–10 | 50–200 |
|---|---|---|---|---|
| tooltrie_v0 | 303 | **451** | 135 | 113 |
| ContextPilot | 279 | 173 | 100 | 95 |

ToolTrie's requests 3–5 are *slower* than requests 0–2 — they were sent before
0–2 had stored anything, so three copies of the same prefix were computed at
once. ContextPilot settles after ~3 requests; ToolTrie needs 6–10.

**So the trie helps through exactly one mechanism**: raising reuse from 38% to
87%, which shortens prefill and compounds under load into higher capacity. Its
parallel-specific weakness is the longer warm-up, which explains its tail but
not its median — the first 10 of 200 requests are only 9–11% of total latency.

---

## 4. Larger tool sets and smart queuing

### 4.1 On real retrieved menus, ordering barely matters

Parts 1–3 use padded menus. Repeating the comparison on BM25-retrieved menus at
four depths, with rates scaled to hold the offered token rate roughly constant:

**Reuse**

| workload | original | alphabetical | frequency | tooltrie_v0 | **ContextPilot** | spread |
|---|---|---|---|---|---|---|
| k4 | 15.87% | 15.28% | 14.62% | 17.48% | **18.72%** | 4.10pp |
| k16 | 6.12% | 6.27% | 5.59% | 7.77% | **9.93%** | 4.34pp |
| k64 | 0.91% | 1.22% | 0.94% | 1.90% | **4.78%** | 3.87pp |
| k128 | 0.37% | 0.58% | 0.54% | 1.13% | **1.99%** | 1.62pp |
| **padded-64** | 1.19% | 38.13% | 39.69% | 87.19% | **96.16%** | **94.97pp** |

A reimplementation of ContextPilot's clustering without its persistent index
(Appendix A.4) tracks it closely except at k128, where it reaches 2.96%. It is
not a separate method and is excluded from the comparison.

**The same cells under the two fairness controls.** Every figure above is one
arrival permutation of a 200-request run, and both of those turn out to matter
(§1.5, A.6). Values are ToolTrie / ContextPilot:

| workload | as measured above | under the control | control applied |
|---|---|---|---|
| k64 | 1.90% / 4.78% — **2.52x** | 1.82% / 2.81% — **1.54x** | mean of 5 arrival orders |
| k128 | 1.13% / 1.99% — **1.76x** | 1.03% / 1.85% — **1.80x** | mean of 3 arrival orders |
| padded-64 | 87.19% / 96.16% — **1.10x** | 97.05% / 97.09% — **1.00x** | steady state, req 201–600 |

ContextPilot still wins every retrieved cell. k128's margin is unchanged and k64's
falls by a third; the padded margin disappears entirely. k4 and k16 remain
single-permutation and are not corrected.

**Latency spread across arms**

| workload | achieved (range) | p50 range (ms) | spread |
|---|---|---|---|
| k4 | 15.907 – 15.924 | 47.6 – 52.3 | 1.10x |
| k64 | 2.623 – 2.635 | 9559.9 – 9760.9 | **1.02x** |
| k128 | 1.004 – 1.007 | 35192.4 – 35852.7 | **1.02x** |
| **padded-64** | 3.564 – 4.014 | 91.8 – 3310.4 | **36.06x** |

**The 95-point reuse spread that drives Parts 1–3 becomes under 3 points, and
the 36x latency spread becomes 1.02x.** Padding gives every request the same
63-tool core, which ordering can pull to the front. Real retrieval returns a
different set per query, so there is little shared structure for any ordering to
exploit. On retrieved menus 95–98% of prefill is uncacheable whatever policy is
used.

All three variants score an identical 96.16% on padded menus despite emitting
almost entirely different orderings (1 of 200 records shared). Padded menus
cannot distinguish them: any consistent hoisting of the 63-tool core scores the
same.

**What padded menus actually measure.** 63 of the 64 tools are common to all 200
requests, and in the `original` arm they appear in one fixed order with the
remaining tool first. The shared prefix therefore runs until that one tool, which
makes reuse arithmetically its position:

| arm | the differing tool sits at | reuse |
|---|---|---|
| original | **0%** of the menu | 1.19% |
| alphabetical | 36% | 38.13% |
| tooltrie_v0 | 88% | 87.19% |
| ContextPilot | 98% | 96.16% |

r² = 0.9996, mean gap 1.47 points. So the 95-point reuse spread and 36x latency
spread in Part 1 are a single quantity — how far down each policy pushes one
tool — measured from a baseline that puts it first in every request. This is why
§1.5's steady-state tie is forced rather than surprising: once two policies both
place that tool last, they must score the same.

**Is this a fair comparison?** Both are causal — neither sees future requests —
and both run on identical menus with identical measurement. ContextPilot is run
at the paper's alpha but **ordering-only**, without the annotations,
de-duplication and scheduling its paper attributes roughly half its cache gain
to. Its results here are therefore a floor rather than a ceiling.

**ToolTrie is second.** It beats unordered, alphabetical and frequency at every
depth — its cleanest sweep anywhere — but loses to ContextPilot at all four:

| depth | tooltrie_v0 | ContextPilot | ratio |
|---|---|---|---|
| k4 | 17.48% | 18.72% | 1.07x |
| k16 | 7.77% | 9.93% | 1.28x |
| k64 | 1.90% | 4.78% | **2.52x** |
| k128 | 1.13% | 1.99% | 1.76x |

The gap widens with menu size up to k64 and then narrows. An earlier draft
reported it growing monotonically to 2.63x; that used the static-refit
reimplementation, which beats ContextPilot only at k128.

The ratios overstate the practical difference twice over. At k64 a 2.52x ratio
is worth 2.88 points of prefill, and every arm lands within 2% on p50. It is
also the top of its own range: across five arrival permutations the k64 ratio
spans 1.09-2.57x, with the other four at 1.09-1.39x. k128 is stable by contrast
(1.65-2.15x over three, with 1.76x mid-range). ContextPilot wins every
permutation at both depths, so the ranking holds and the k64 magnitude does not
(A.6).

**Putting frequently used tools first does not work.** The `frequency` arm is
the worst at k4 and k16 and near-worst at k64 and k128.

### 4.2 Smart queuing trades the tail for the median

Retrieved menus have variable request sizes (k64 spans 3,407–18,049 tokens,
5.3x), unlike padded menus where every request is the same size. That is the
setting where shortest-job-first should help, and job size is known exactly on
arrival because prefill dominates. Three policies were added: `sjf` by job size,
`suffix` by estimated uncached size, and `random` as a control.

**It does not reduce the tail.** k64 at 4 req/s, arrival to first token, ms:

| policy | mean | p50 | p95 | max |
|---|---|---|---|---|
| `fifo` | 13156 | 11227 | 26647 | 29251 |
| `sjf` | 8944 (−32%) | **1628 (−85.5%)** | **41271 (+54.9%)** | **66490 (+127%)** |
| `sjf` + aging 2000 | 12735 (−3.2%) | 10766 | 26724 (+0.3%) | 31387 (+7.3%) |

k128 behaves the same: p50 −87.6%, p95 +25.8%, max +67.4%.

The fairness knob traces a frontier, and **no point on it improves both**:

| aging | mean | p50 | p95 | max |
|---|---|---|---|---|
| 0 | −32.0% | −85.5% | **+54.9%** | **+127.3%** |
| **250** | **−24.6%** | **−47.3%** | **+17.6%** | **+45.6%** |
| 1000 | −7.9% | −6.4% | −3.4% | +17.8% |
| 2000 | −3.2% | −4.1% | +0.3% | +7.3% |

At aging 250 you keep most of the median gain for a contained +17.6% tail. At
2000 the tail is safe but the policy has become fifo. Aging 1000 is the one
point where p95 improves (−3.4%) while `max` still degrades 17.8% — the jobs SJF
starves are too few to reach p95, which is why `max` is shown here.

**A control was needed, and it changed the number.** On padded menus — where
every request is the same size, so the sort key carries no information — `sjf`
still cut the median 37%. Under a deep queue, *any* order unrelated to arrival
reshapes the distribution, because fifo makes waiting time grow with arrival
position. A `random` policy measures that baseline:

| workload | from leaving fifo alone | from actual size information |
|---|---|---|
| k64 (variable sizes) | median −27.4% | **mean −29.2%** |
| padded-64 (uniform sizes) | median −27.5% | mean −3.7% |

The artifact is nearly identical in both (−27.4% vs −27.5%), and reproduced a
third time on ToolTrie at 16 req/s (−27.1%), confirming the mechanism. Subtracting it, the real size-aware gain is 29.2 points against 3.7 —
an 8x separation tracking the 87x difference in size variance. **Without this
control the headline would have been "SJF cuts median latency 6.9x", crediting
the policy for something a coin flip partly reproduces.**

`suffix` was indistinguishable from `sjf`, as predicted: reuse on these
workloads is 0.4–2%, so the uncached portion is ~99% of every request and the
cache-aware term has nothing to work with.

---

## 5. Explored and rejected

Two ordering methods outside the four questions, recorded so they are not retried.

**A single global tool order.** Sort every request's tools by one global
frequency ranking — no index, no clustering, no parameters, one sort per
request, against ContextPilot's O(N²) index build.

| depth | tooltrie | canonical (deployable) | canonical (oracle) | ContextPilot |
|---|---|---|---|---|
| k64 | 1.90% | 2.75% | **4.28%** | 4.78% |
| k128 | 1.13% | 1.99% | **2.87%** | 1.99% |

It reaches 90% of ContextPilot's reuse at k64 and exceeds it at k128, with none
of its machinery. The deployable version, counting only earlier requests, reaches
58-67%.

**A trie inside ContextPilot.** ContextPilot leaves everything after its matched
prefix in original order, so that portion can never be shared. Keeping its head
and reordering only the tail:

| depth | ContextPilot | hybrid | vs ContextPilot |
|---|---|---|---|
| k64 | 4.78% | 4.17% | −12.7% |
| k128 | 1.99% | **3.11%** | **+56.2%** |

At k128 this was the best policy in the study on reuse, latency and throughput.

**It cannot work on padded menus, and needs no run to show it.** There,
ContextPilot's causal head already covers 62.69 of 64 tools, leaving a 1.31-tool
tail that sits *behind* the shared prefix, so reordering it cannot change what
the cache matches. Constructing the hybrid on padded menus produces orderings
**identical to ContextPilot in 199 of 200 records** — the one exception being the
first request, which has no predecessor and so no head. The idea has room to act
only where ContextPilot's head is short, which is precisely the retrieved menus
measured above (head 0.36 tools at k128).

**Both were rejected on accuracy.**

| depth | arm | reuse | accuracy | mean position of correct tool |
|---|---|---|---|---|
| k128 | original | 0.37% | **22.98%** | **11.5** |
| k128 | ContextPilot | 1.99% | **22.98%** | 27.8 |
| k128 | canonical | 2.87% | 14.29% | 56.5 |
| k128 | hybrid | **3.11%** | **13.66%** | 57.7 |

The hybrid's k128 win costs **9.3 points of accuracy for 1.12 points of reuse** —
8.3 points of accuracy per point of reuse. Sorting by global frequency can send
a request's own top-ranked tool to position 56 of 128; ContextPilot moves only
tools inside a matched cluster and leaves the rest in relevance order.

**What would revive them.** No arm here used ContextPilot's order annotations,
which exist to decouple relevance from position and which its paper reports can
lift accuracy above the unordered baseline. Canonical and hybrid displace the
correct tool most, so they would gain most. If annotations recover those 9.3
points the exchange rate changes; if not, the line is closed.

---

## Limitations

1. **Parts 1–3 describe a padded workload, not tool ordering in general.** §4.1
   is the check: reuse spread falls 94.97pp → 0.98pp, latency spread 36x → 1.02x.
   The most important limit here. Two further properties of that workload were
   measured late: reuse on it is arithmetically the position of the single tool
   that differs between requests (r² = 0.9996), and the ToolTrie–ContextPilot gap
   on it is warm-up that vanishes by request 200 (§1.5).
2. **Accuracy is one model, one seed, two depths.** Absolute accuracy is low
   (11–28%), a 0.6B model may be unusually order-sensitive, and no arm used
   order annotations. The direction is consistent; the magnitudes are not
   transferable.
3. **Model size is only partly controlled.** Reuse was confirmed unchanged at
   4B and the ordering gap grows there. 8B untested. One claim was falsified by
   the 4B run — ContextPilot's flat latency shape is 0.6B-specific.
4. **Queuing was tested only under saturation at one in-flight cap.** Every run
   was deeply backlogged. Preemptive policies, and any policy inside the engine
   rather than in front of it, are untested.
5. **`max` is a single sample** and moved 269 → 710 ms between two runs of the
   same configuration. Use p50 and p95.
6. **Two runs are n=199**, each losing one request to a client-side socket
   error, not a server fault.
7. **One trial per cell, and one arrival order.** Reuse reproduced to the
   digit across the order control, seven rates, three cache sizes and same-day
   re-runs, which is the evidence for run-to-run stability. Arrival permutation
   is a separate and larger source of variation: eight cells were re-run under
   3–5 permutations (A.6) and k64 moved 1.09–2.57x. Every other cell rests on
   one permutation, so point estimates here should be read as draws, not
   constants.
8. **The §5 methods are single-configuration**, rejected on the accuracy
    exchange rate rather than an exhaustive sweep.

---

## Appendix

### A.1 Validity checks

All 12 runs in §1.1 were audited: one server process throughout; identical
model, decode settings, arrival seed and request sequence; **prompt token counts
identical per request across arms** (1,380,694 total, 0 mismatches); menus
identical in membership, so arms are pure permutations; no foreign traffic on
the server; zero preemptions.

Cache reset was verified rather than assumed: `original` cached exactly 16,384
tokens both on a cold server and immediately after ContextPilot had filled the
cache to 96%.

**Instrument check.** The client-side timer runs 15–22 ms above the server's own
counter, constant across all runs and arms. That offset is 20% of ContextPilot's
latency but 0.6% of `original`'s, so it *understates* the winner's advantage —
the bias runs against the conclusion.

**Order control.** Arms always ran in the same sequence, so the whole set was
re-run reversed. Reuse reproduced to the digit in all eight cells and the
ranking held; run order was not a confound.

**One benign anomaly.** `original` at rate 4 logged 2.5x the expected cache
lookups. It is the only arm that saturated, so its prefills were split across
scheduler steps and re-queried the same blocks. Actual computed tokens were
unchanged and preemptions were zero, so no reported figure is affected.

**Capacity.** Cutting the cache 4.2x left three of four arms completely
unchanged (0.00pp reuse change). Only `alphabetical` degraded. There was no
eviction pressure at this size, which is why §2's caching mechanism had nothing
to exploit.

### A.2 Two corrections made during the work

**Time to first token was the wrong metric for §2 and §4.** It starts at
dispatch, so it cannot see delay a queuing policy imposes. Measured that way the
adaptor looked like a flat null and it is not. All queuing results here report
arrival to first token. Parts 1 and 3 are unaffected — they ran uncapped, where
the two agree within 1.8 ms.

**Capping in-flight requests was claimed to cut the tail 8x. It does not.** That
was the same metric error. Capping moves the queue from the server to the
client: user-visible latency is 1.8x *worse* at p50 and 1.5x worse at max, and
throughput drops. The correct sign is negative.

Two policy configurations were also flagged as having reordered nothing, so
their nulls are mechanical rather than real. The check that catches this —
comparing dispatch order against the baseline — is built into the summariser,
because the same failure had already produced two false nulls earlier.

### A.3 How this is normally done

The adaptor is a weak variant of **longest prefix match**, which SGLang
implements and vLLM does not. It differs in four ways, each weakening it: it
compares against the last dispatched request rather than the actual cache
contents; it picks one winner rather than sorting the queue; it bounds fairness
with a wall-clock cutoff rather than deficit counters; and it sits in a client
rather than in the engine or a router. Scheduling with prefix reuse under
latency constraints is NP-hard, and both FIFO and greedy LPM have documented
failure modes, so a greedy heuristic was never going to be near-optimal.

Ordering and scheduling are complementary, not competing: ordering decides what
the shared prefix is, scheduling decides who runs when.

References: [ContextPilot](https://arxiv.org/abs/2511.03475) ·
[DLPM](https://arxiv.org/abs/2501.14312) ·
[k-LPM, NP-hardness](https://arxiv.org/pdf/2502.04677) ·
[SGLang RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) ·
[llm-d routing](https://developers.redhat.com/articles/2026/01/13/accelerate-multi-turn-workloads-llm-d) ·
[Sarathi-Serve chunked prefill](https://arxiv.org/html/2403.02310) ·
[PARS](https://arxiv.org/html/2510.03243) ·
[ReCache](https://arxiv.org/html/2608.19662)

### A.4 The ContextPilot naming

Three arms have carried the name. Only one is ContextPilot:

| arm in this report | alpha | path | ContextPilot? |
|---|---|---|---|
| **ContextPilot (ordering)** | 0.001 | official `ContextPilot.reorder`, persistent index | **yes** |
| static-refit (a=0.001) | 0.001 | `ContextIndex.fit_transform`, no persistent index | no |
| static-refit (a=0.5) | **0.5** | `ContextIndex.fit_transform` | no |

`alpha` weights positional alignment against tool overlap in ContextPilot's
clustering distance. The paper sets it to **0.001 across all experiments**,
inside a declared `[0.001, 0.01]`, so that "overlap count remains the dominant
factor" — a tie-breaker, not a driver. The 0.5 arm is 500x that, which inverts
the intent.

"Static refit" means no persistent index: for request *n* the whole index is
rebuilt from scratch over requests 0..*n*, used once, and discarded. ContextPilot
instead maintains one index incrementally. Both are causal — neither sees future
requests — but only the official path is ContextPilot's.

The 0.5 value was an error rather than a choice; the builder that produced it now
defaults to 0.001 and requires an explicit override to reproduce it. **It changes
no result.** alpha does change the ordering substantially — only 1 of 200 padded
records is identical between the two settings — but the outcome is the same,
because on padded menus any ordering that hoists the 63-tool core into a
consistent leading block scores alike: at rates 1, 2 and 4 both give 96.16% reuse
with p50 within 4 ms.

Verified at 0.001: the padded rates 1-4. Not re-run at 0.001: the saturation
sweep, capacity squeeze, 4B legs, §3 and the order control. Those are driven by
reuse, which is identical, so they are expected to reproduce — inference, not
measurement.

Even the official arm runs ordering only, with no annotations, de-duplication or
ContextPilot scheduling, so nothing here measures the full system.

### A.5 Runs

| stage | directory | runs |
|---|---|---|
| Rate sweep | `concurrent-latency-20260820-212500/` | 12 |
| Order-reversal control | `concurrent-order-control-20260829-142216/` | 8 |
| Adaptor sweep | `adaptor-sweep-20260829-143543/` | 15 |
| Capacity 44,656 | `capacity-44656-20260829-143821/` | 14 |
| Saturation sweep | `saturation-20260829-160137/` | 8 |
| Qwen3-4B transfer | `qwen3-4b-transfer-20260829-162208/` | 4 |
| BM25 k sweep | `bm25-k-sweep-20260829-203836/` | 16 |
| Qwen3-4B under load | `qwen3-4b-load-20260829-203933/` | 5 |
| Size-aware queuing | `sjf-queuing-20260829-212639/` | 24 |
| ContextPilot at alpha=0.001 | `alpha001-comparison-20260829-224719/` | 17 |
| Canonical and hybrid ordering | `canonical-order-20260830-003615/` | 8 |
| Accuracy | `accuracy-gate-20260830-011416/` | 10 |
| Adaptor on ToolTrie under a forced queue | `tooltrie-adaptor-20260830-121234/` | 4 |
| Adaptor on frequency and ContextPilot | `adaptor-table-20260830-141733/` | 4 |
| ContextPilot + baselines on padded | `cp-online-padded-20260830-115456/` | 15 |
| Fairness audit: planner budget, arrival seeds | `tooltrie-uncapped-20260830-212728/` | 9 |
| Steady state (600 req) and more arrival seeds | `steady-and-seeds-20260830-221902/` | 14 |

**187 runs.** All under the git-ignored `cluster/results/`.

Driver `scripts/replay_vllm_concurrent.py`; also added
`summarize_queuing_runs.py`, `build_canonical_ordering.py`,
`score_tool_selection.py`. Server: Qwen3-0.6B on GPU 2 port 8300, prefix caching
on, 64 max sequences, 8192 max batched tokens; 4B on GPU 3 port 8301. Each stage
holds a lock so only one driver ever runs against a server.

**Known script defect.** A server started from inside a lock-holding script
inherits the lock and keeps it after the script exits, deadlocking the next
stage. This happened once and is fixed by closing the descriptor in the child.
No results were affected. Scripts written before the fix still contain it.

A quarantine directory in the first run holds six contaminated files from an
aborted attempt where two drivers ran at once. They are excluded from everything
above and must not be cited.

---

### A.6 Is the ToolTrie-vs-ContextPilot comparison fair?

Audited rather than asserted. Both planners are **causal**: ToolTrie calls
`plan()` before `observe()` on every request in arrival order; ContextPilot uses
its persistent online `reorder` API with an index that grows as requests arrive.
Neither sees the future, neither gets training data, and the shipped ToolTrie
orderings reproduce **200/200** from their recorded parameters, with eviction
and node counts matching exactly.

| | ToolTrie v0 | ContextPilot |
|---|---|---|
| information regime | causal | causal |
| training data | none | none |
| menu membership | unchanged | unchanged |
| tuning | `v0` defaults | authors' defaults (α=0.001) |
| planner memory | 190,896-token budget, LRU | unbounded |
| planning cost, padded | 3.62 ms/req | 1.97 ms/req |
| planning cost, k128 | 0.20 ms/req | 1.94 ms/req |

**Planning cost is not a differentiator.** It is at most 3.1% of the fastest
arm's p50 and 0.006% at k128. ToolTrie's apparent 9.5x advantage at k128 is a
symptom, not a saving: its cost tracks how deep the trie walk gets, so a cheap
plan means it matched nothing (56.08 tools matched and 3.62 ms on padded menus
against 1.50 tools and 0.20 ms at k128). ContextPilot runs the same clustering
either way. The O(N²) distance matrix belongs to its offline `fit_transform`,
not the online API used here.

**The memory asymmetry is real but immaterial.** ToolTrie's budget bound hard on
retrieved menus — 10,886 evictions at k64, 23,648 at k128 — while ContextPilot's
index had no bound. Lifting it to 8M tokens removes every eviction and raises
the mean shared leading prefix 35–46%. Reuse does not follow:

| | evictions | shared prefix | reuse |
|---|---|---|---|
| k64 capped | 10,886 | 1.402 | **1.90%** |
| k64 uncapped | 0 | 1.899 (+35%) | 1.86% |
| k128 capped | 23,648 | 2.523 | **1.13%** |
| k128 uncapped | 0 | 3.673 (+46%) | 1.16% |

The budget is not a handicap; it is a model of the cache. At 189,728 KV tokens
the server holds ~27 requests at k64 and ~14 at k128, and ToolTrie's best matches
already sit a median 25–28 requests back. Extra planner memory buys hints the
cache cannot honour, and it trades away ones it can: matches *inside* the
residency horizon fall from 42 to 36 at k64 and 24 to 22 at k128 as the greedy
walk chases older, longer paths.

That horizon is also where ContextPilot's lead comes from. Its matches are both
recent and long:

| k64, cache holds ~27.5 requests | matches | within horizon | mean prefix within |
|---|---|---|---|
| tooltrie capped | 86 | 42 | 3.48 |
| tooltrie uncapped | 91 | 36 | 3.56 |
| ContextPilot | 92 | **55** | **9.31** |

**Arrival order is the weaker control, and its effect depends on depth.** Every
other result here uses one arrival permutation. Re-running k64 under five and
k128 under three permutations of the same 200 records:

| k64 | ToolTrie | ContextPilot | ratio |
|---|---|---|---|
| seed 0 (used throughout this report) | 1.86% | **4.78%** | **2.57x** |
| seed 1 | 1.80% | 2.51% | 1.39x |
| seed 2 | 1.98% | 2.15% | 1.09x |
| seed 3 | 1.77% | 2.44% | 1.38x |
| seed 4 | 1.68% | 2.17% | 1.29x |
| mean | 1.82% | 2.81% | 1.54x |
| spread | 0.30pp | **2.63pp** | |

| k128 | ToolTrie | ContextPilot | ratio |
|---|---|---|---|
| seed 0 (used throughout this report) | 1.16% | 1.99% | 1.72x |
| seed 1 | 1.10% | 1.81% | 1.65x |
| seed 2 | 0.82% | 1.76% | 2.15x |
| mean | 1.03% | 1.85% | 1.84x |
| spread | 0.34pp | 0.23pp | |

The `mean` ratios are the mean of the per-seed ratios. Taking the ratio of the
column means instead gives 1.55x at k64 (identical) and 1.80x at k128, which is
the figure §4.1 quotes beside those means.

The ToolTrie column uses the budget-lifted planner, legitimate because capped and
uncapped are a measured null above; seed 0 against the capped arm is the 2.52x
and 1.76x reported in §4.1.

**k128 is stable and k64 is not.** At k128 the ratio stays in 1.65-2.15x and the
published 1.76x sits mid-range, so that cell needs no qualification. At k64 the
ratio spans 1.09-2.57x, and the published 2.52x is the top of it — the other four
permutations cluster at 1.09-1.39x. ContextPilot wins all eight cells, so the
ranking is now better evidenced than before; the k64 *magnitude* is not.

**Why seed 0 is the outlier, and why it is not a fluke.** The benchmark's
natural order is blocked by source: 101 `toolret:apibank` requests followed by 99
`toolret:apigen`, with 198 of 199 consecutive pairs sharing a source. Adjacent
requests overlap 3.40 tools in that order against 1.43 shuffled — 2.4x the
locality. ContextPilot's set-overlap clustering converts that into reuse;
ToolTrie's exact-prefix walk cannot. Seed 0 is therefore the *ordered* case, not
a lucky draw, and which column applies depends on whether the served traffic
arrives in blocks of similar requests. Shuffled is the conservative assumption.

The sensitivity does not belong to one method. At k64 ContextPilot varies 2.63pp
against ToolTrie's 0.30pp; at k128 it is the other way round, 0.23pp against
0.34pp. Every cell outside these eight still rests on seed 0 alone.

**Two asymmetries stand.** ToolTrie is `v0` against a tuned published system,
though sweeping its one ordering knob found the shipped value already optimal
(`recency_window` 32/128/512/2048/none → 1.43/1.90/1.90/1.90/1.90 at k64) and
`tooltrie_v1.py` has never entered any comparison. And the workload favours
ContextPilot by construction: 63 of 64 padded tools are common to all 200
requests, so its root cluster alone is the answer, while a trie must discover the
same block through prefix agreement.

The comparison's best guarantee is structural: ToolTrie is this project's method,
ContextPilot is the baseline, and ContextPilot wins.
