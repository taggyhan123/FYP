# Tool ordering under concurrent load

Four questions, none of which had been run in this project before. Each maps to
one part of this document:

| question | part |
|---|---|
| Parallel requests → latency distribution at p95, p99, max | **Part 1** |
| An adaptor with a mean-vs-max tradeoff | **Part 2** |
| How the trie reduces parallel request latency | **Part 3** |
| Reducing tail latency by smart queuing, for larger tool sets | **Part 4** |
| (outside the brief) methods explored and rejected | **Part 5** |

All numbers are Qwen3-0.6B on one RTX 3090 (with a Qwen3-4B transfer test in
§1.8), vLLM 0.26.0 with unmodified automatic prefix caching. Raw outputs live
under the git-ignored `cluster/results/` directories named in **Provenance**;
this file is the compact summary.

**Naming.** The arm stored as `contextpilot_causal` is **not ContextPilot**. In
the paper (Jiang, Huang et al., MLSys 2026, arXiv:2511.03475) `alpha` weights the
positional term of the context distance function against the overlap term:

`d_ij = 1 - |S_ij|/max(|C_i|,|C_j|) + alpha * mean_k |p_i(k) - p_j(k)|`

The paper sets **`alpha = 0.001`**, inside a declared range of `[0.001, 0.01]`,
explicitly so that "overlap count remains the dominant factor while
incorporating positional alignment". **`alpha` is intended as a tie-breaker on
position, not a driver of the clustering.**

This arm uses **`alpha = 0.5`** — 500x that value — which inverts the intent: the
overlap term is bounded in [0,1] while the positional term reaches ~5-12 at 64
tools, so position decides the clustering instead of shared content. It is also
ordering-only, with no annotations, de-duplication or scheduling. Attribution to
online ContextPilot is **withdrawn** by `reports/tooltrie-phase2/findings.md`;
read it as **ContextPilot static-refit causal adaptation (alpha=0.5; ordering
only)**. Correctly configured `alpha=0.001` arms — both the static refit and the
official online API — are measured against it in §4.1.

**Headline — ToolTrie-v0 under parallel load.** Against the baselines it
replaces on padded-64, and the margin *grows* with load:

| rate | vs `original` | vs `alphabetical` |
|---|---|---|
| 1 | p50 -58.7%, p99 -48.7%, max -44.8% | p50 -52.3%, p99 -41.0% |
| 2 | p50 -59.8%, p99 -65.0%, max -69.9% | p50 -53.5%, p99 -58.7% |
| 4 | **p50 -96.5%, p99 -94.7%, max -92.5%** | p50 -65.0%, p99 -72.8% |

It also sustains **11.24 req/s against `original`'s 3.56 — 3.2x the admission
capacity** (§1.7), and at Qwen3-4B it is **5.2x faster than `alphabetical` at
rate 1 and 118x at rate 2**, where `alphabetical` collapses (§1.8). Serially
none of this was visible: max was flat within 1.25x and the project's earlier
latency claims were deliberately weak.

**On genuinely retrieved menus the separation almost vanishes** — the between-arm
reuse spread falls from 94.97pp to 0.98pp and the p50 spread from 36x to 1.02x
(§4.1). ToolTrie beats every simple heuristic there but loses to
correctly-configured ContextPilot at all four depths.

**Where ToolTrie loses.** On padded-64 the static-refit arm beats it in 17 of 18
distribution cells (p50 1.25-1.27x). On retrieved menus, correctly-configured
ContextPilot beats it at all four depths, by a margin that grows with menu size
(1.07x at k4 to 2.63x at k128, §4.1).

**And the reordering is not free.** Tool-selection accuracy on retrieved menus
falls **6.8 points at k128 and 11.3 at k64** relative to not reordering at all
(§1.9). ContextPilot at k128 is the one policy that gains reuse at *zero*
accuracy cost. Every latency and reuse figure below should be read with that
alongside it: **cache reuse and answer accuracy compete for the front of the
prompt.**

**The adaptor is a negative result.** It works and traces a real mean-vs-max
frontier, but it is a weak variant of a standard algorithm (LPM), was evaluated
in the one regime where its mechanism cannot pay, and is beaten 6.7x by simply
choosing a better ordering. Two claims made from time-to-first-token alone were
wrong and are corrected in §2.3 and §2.5.

---

## Part 1 — Parallel request latency distribution

### 1.1 Why this needed new machinery

Every prior replay in this project was strictly serial. The existing driver
derives per-request TTFT by differencing the cumulative Prometheus counter
`vllm:time_to_first_token_seconds_sum` around each request, which is only valid
with exactly one request in flight. It could not be reused, and no latency
percentile of any kind existed anywhere in `reports/` or `cluster/results/`
before this work — the only percentile figures in the repository describe
*dataset* properties (schema tokens per tool, tools per task), not latency.

`scripts/replay_vllm_concurrent.py` was written as a separate driver. It issues
requests with `stream=True` and timestamps the first streamed chunk on the
client, which is valid at any concurrency, and it generates open-loop Poisson
arrivals so the offered rate is independent of completions. Closed-loop load
would have self-throttled and bounded p99 by the concurrency limit, making tail
analysis meaningless.

### 1.2 Method

- 200 BFCL requests, padded 64-tool menus, mean 6,903 prompt tokens/request.
- Orderings are the **frozen phase-1 emissions** from
  `cluster/results/tooltrie-phase2-20260803-181133/`. Phase 2 only replays them.
- Decode pinned with `ignore_eos` and `max_tokens=48`, so all 2,400 requests in
  Stage A emitted exactly 48 tokens. Without this, `completion_tokens` ranged
  8-48 (sd ~14) and decode-length variance polluted the tails.
- Open-loop Poisson arrivals, `arrival_seed=42`, uncapped in-flight.
- Prefix cache reset before every run.

**Arm selection.** Fourteen frozen BFCL orderings exist in the phase-2
directory; four were replayed under load. That is a deliberate choice, not an
oversight: on the serial phase-2 measurements the twelve comparable arms
collapse into a small number of distinct reuse tiers, and the four chosen span
all of them.

| reuse (0.6B, serial) | arms | replayed under load |
|---|---|---|
| 96.67% | `contextpilot_intra`, `contextpilot_intra_schedule` | no — **offline**, both use future information |
| 96.18% | `contextpilot_causal` | **yes** — best deployable |
| 87.11% | `tooltrie_v0` | **yes** |
| 39.69% | `conditional_pair`, `conditional_pair_triple`, `fp_tree_conditional`, `frequency_fitted`, `schema_cost_fitted` | no — five arms degenerate at one value |
| 38.13% | `alphabetical` | **yes** |
| 1.19% | `cacheweaver`, `original` | **yes** (`original`; `cacheweaver` is degenerate with it) |

`frequency_fitted` in particular sits in the `alphabetical` tier, 1.56pp away,
and would trace a curve on top of it. The only arms above `contextpilot_causal`
are the two offline ContextPilot variants, which are not deployable.

**One competitive arm is absent and cannot be replayed.** `frequency_online`
reaches **96.27%** reuse, above `contextpilot_causal`'s 96.16%. It is an online
policy requiring a persistent counter over strictly earlier requests, so no
frozen ordering exists for it; only the task-disjoint `frequency_fitted` is
replayable. Its own report treats that result as diagnostic rather than
competitive — `reports/frequency-online/findings.md` states that a trivial
counter reaching parity "shows the padded workload mainly rewards discovering
its nearly fixed 63-tool core" and "is not evidence that frequency counting is
the best policy on genuinely retrieved menus." That is a caveat on **all** of
Part 1, since every concurrent run here uses padded-64 menus (Limitation 5).

### 1.3 Results — TTFT (ms), capacity 188,688 tokens

| rate | arm | achieved | reuse | p50 | p90 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| 1 | original | 1.0066 | 1.19% | 266.4 | 403.8 | 431.5 | 477.9 | 622.4 | 293.2 |
| 1 | alphabetical | 1.0071 | 38.13% | 230.7 | 288.6 | 364.4 | 414.9 | 493.5 | 226.6 |
| 1 | tooltrie_v0 | 1.0075 | 87.19% | 110.1 | 135.7 | 162.1 | **244.9** | 343.8 | 112.0 |
| 1 | contextpilot_causal | 1.0075 | 96.16% | **88.0** | **94.6** | **107.6** | 257.9 | **269.1** | **89.9** |
| 2 | original | 2.0087 | 1.19% | 284.9 | 563.7 | 710.8 | 934.0 | 1208.2 | 373.1 |
| 2 | alphabetical | 2.0106 | 38.13% | 246.5 | 425.0 | 486.7 | 792.5 | 896.0 | 271.8 |
| 2 | tooltrie_v0 | 2.0124 | 87.19% | 114.6 | 147.9 | 177.5 | 327.1 | 363.7 | 116.6 |
| 2 | contextpilot_causal | 2.0124 | 96.16% | **91.4** | **116.4** | **122.1** | **265.5** | **281.5** | **95.8** |
| 4 | original | **3.5645** | 1.19% | 3310.4 | 6111.4 | 6498.6 | 7205.0 | 7476.0 | 3479.0 |
| 4 | alphabetical | 4.0081 | 38.13% | 331.8 | 932.5 | 1058.1 | 1406.8 | 1724.9 | 447.2 |
| 4 | tooltrie_v0 | 4.0139 | 87.19% | 116.3 | 198.5 | 225.7 | 382.7 | 558.4 | 131.2 |
| 4 | contextpilot_causal | 4.0142 | 96.16% | **91.8** | **124.8** | **132.5** | **266.7** | **308.2** | **99.7** |

Head-to-head across all 18 distribution cells, `contextpilot_causal` beats
`tooltrie_v0` in **17**. The single exception is p99 at rate 1 (244.9 vs 257.9
ms, a 5% edge) and it disappears by rate 2. On the median specifically,
ToolTrie is 1.25x / 1.25x / 1.27x ContextPilot — a stable +22 to +25 ms penalty
at every load.

### 1.4 Three findings the serial data could not have produced

**(a) `max` reverses.** Serially, max was flat across arms (~270 ms, 1.25x
spread) and the honest conclusion was "nobody wins on max". Under load at 4
req/s it spans 308.2 to 7476.0 ms — **24.3x**. Max is not an ordering-invariant
property; it is dominated by queueing, which concurrency-1 cannot exhibit.

**(b) Reuse converts into admission capacity.** `original` is the only arm that
failed to keep up, achieving 3.5645 against a 4 req/s offer while every other
arm tracked the offered rate. It saturates between 2 and 4 req/s.

**(c) The tails are structurally different populations.** Index of the five
slowest requests out of 200 arrivals, rate 4:

| arm | slowest request indices | what this is |
|---|---|---|
| original | **#198, #193, #197, #192, #196** | the *last* arrivals — monotonically growing backlog, i.e. genuine queue instability |
| alphabetical | #53, #52, #50, #47, #51 | a bounded mid-run excursion |
| tooltrie_v0 | #3, #4, #2, #5, #46 | warm-up ramp, slightly long |
| contextpilot_causal | #2, #1, #0, #3, #4 | cold start only |

ContextPilot's tail is a **fixed startup cost**, not a load-induced queue. That
is why its p99 moves only 257.9 -> 265.5 -> 266.7 ms across a 4x load increase
(1.03x) while `original` goes 477.9 -> 934.0 -> 7205.0 (15.08x). Normalised:

| arm | p99/p50 @1 | @2 | @4 |
|---|---|---|---|
| original | 1.79 | 3.28 | 2.18 |
| alphabetical | 1.80 | 3.22 | 4.24 |
| tooltrie_v0 | 2.22 | 2.85 | 3.29 |
| **contextpilot_causal** | **2.93** | **2.90** | **2.91** |

ContextPilot is the only arm whose latency shape is load-invariant.

### 1.5 Validity

Every controllable axis was audited across the 12 Stage A runs:

| check | result |
|---|---|
| Server process | one PID (2816760) for all 12 runs, started before the first replay |
| Model / endpoint / request_count / format_version | identical 12/12 |
| `decode` | identical; **all 2,400 requests emitted exactly 48 tokens** |
| `dispatch`, `arrival_seed`, `max_inflight` | identical |
| Arrival offsets | **bit-identical across the 4 arms** within each rate |
| Task sequence | one distinct `task_id` sequence across all 12 runs |
| **Prompt length** | `prompt_tokens` identical per request, 1,380,694 total, **0 mismatches** |
| Menu membership | same tool *set* per task, all menus size 64 — arms are pure permutations |
| Foreign traffic | `inter_token_latency_count = 9400 = 200x47` in 12/12 |
| Preemptions | 0 in all 12 |
| Cache reset | `original` after a cold start and after ContextPilot filled the cache to 96% both cached **16,384 tokens exactly** |

**Instrument cross-check.** Client-side streaming timer vs the server's own
`time_to_first_token_seconds_sum`: a constant **15.2-22.2 ms** offset across all
12 runs, independent of arm and rate (loopback HTTP + SSE parse). That offset is
20.3% of ContextPilot's TTFT but 0.64% of `original`'s, so the instrument
**understates** the winner's advantage. The bias runs against the conclusion.

**One benign anomaly.** `original` at rate 4 logged `prefix_cache_queries` =
3,478,803 against 1,380,694 elsewhere. It is the only arm that saturated, so its
prefills were chunked across scheduler steps and each chunk re-queried the same
blocks. It is a lookup counter, not work: `request_prefill_kv_computed_tokens_sum`
is unchanged at 1,364,310 and `num_preemptions` is 0. Reuse derives from the two
counters that are identical, and latency is client-timed, so no reported figure
is affected. It also sits in the baseline, so it cannot flatter the winner.

**Order-reversal control.** In Stage A the arms always ran
`original -> alphabetical -> tooltrie_v0 -> contextpilot_causal`, so ContextPilot
was always last. The full configuration was re-run with the order **reversed**
(`cluster/results/concurrent-order-control-20260829-142216/`):

| rate | arm | reuse A / rev | p50 A / rev | p99 A / rev |
|---|---|---|---|---|
| 1 | original | 1.19 / 1.19 | 266.4 / 270.6 | 477.9 / 485.7 |
| 1 | alphabetical | 38.13 / 38.13 | 230.7 / 235.6 | 414.9 / 416.9 |
| 1 | tooltrie_v0 | 87.19 / 87.19 | 110.1 / 115.5 | 244.9 / 245.5 |
| 1 | contextpilot_causal | 96.16 / 96.16 | 88.0 / 93.2 | 257.9 / 263.5 |
| 4 | original | 1.19 / 1.19 | 3310.4 / 3356.5 | 7205.0 / 7215.6 |
| 4 | alphabetical | 38.13 / 38.13 | 331.8 / 328.2 | 1406.8 / 1392.3 |
| 4 | tooltrie_v0 | 87.19 / 87.19 | 116.3 / 119.0 | 382.7 / 381.7 |
| 4 | contextpilot_causal | 96.16 / 96.16 | 91.8 / 95.1 | 266.7 / 261.2 |

Reuse reproduces **to the digit in all eight cells** and the ranking is fully
preserved. Run order was not a confound.

Two caveats on this control. The `original` rate-1 reversed run is **n=199**: one
request died on a stale aiohttp keep-alive socket. It was a client-side pool
issue, not a server fault — the server PID and 8-day uptime were unbroken, its
692 KB log contains no error, and reuse still matched at exactly 1.19%. Separately,
`contextpilot_causal` rate 1 reversed recorded max = 710.2 ms against 269.1 in
Stage A. Every other statistic in that run matches closely. **`max` is a single
order statistic at n=200 and is by far the least stable figure in this report**;
p95 and p99 should carry the weight.

### 1.6 Capacity robustness

Stage A ran at 188,688 tokens — the most generous KV capacity anywhere in the
project, and therefore the most favourable setting for the winner.
`consolidated-report.md:792` reported the ordering ranking **inverting** at
7,680 tokens — but `:796` revises that: "**Both later additions revise that
claim.** `frequency_online` ... beats ToolTrie in every regime by up to 9.1
points, so the correct statement is that *adaptive* policies win under scarcity,
not that the trie does." The revised claim is what is tested here. The comparison was repeated at **44,656 tokens** (8B-equivalent, per
`consolidated-report.md:702`) using `--num-gpu-blocks-override 2791`, so model
size, prefill cost and tokenisation are all held fixed and capacity is isolated.
7,680 was rejected: it fits only ~1.1 of these requests, so under load the cache
itself serialises the run and the concurrency premise is destroyed.

Rate 4, effect of a 4.2x capacity cut:

| arm | reuse 188,688 -> 44,656 | p50 | p99 | max |
|---|---|---|---|---|
| contextpilot_causal | 96.16 -> **96.16** (0.00) | +0.9% | **-0.8%** | -1.7% |
| tooltrie_v0 | 87.19 -> **87.19** (0.00) | +0.8% | +1.6% | +0.7% |
| alphabetical | 38.13 -> 36.65 (**-1.48pp**) | **+31.8%** | **+34.3%** | +18.8% |
| original | 1.19 -> **1.19** (0.00) | -5.3% | -18.8% | -18.0% |

**No inversion.** Three of four arms are completely unaffected by removing 77% of
the cache. The mechanism is prefix concentration: ContextPilot and ToolTrie
collapse the shared menu into one contiguous ~6,950-token run that stays resident
in 44,656 tokens, while alphabetical scatters it and is the only arm that pays.
The report's own 7,680-token matrix (`consolidated-report.md:796`) shows
`frequency_online` at ~96% reuse beating ToolTrie in all four regimes, so
scarcity favours *adaptive* policies rather than the trie. Nothing here
contradicts that: at 44,656 no arm faced eviction pressure at all.

`original` improving slightly when squeezed is **not** claimed as a finding —
n=1, and max is unstable as noted above.

### 1.7 Saturation point and throughput capacity

Stage A stopped at 4 req/s, where every arm except `original` still tracked the
offered rate. Pushing further locates the ceiling (native capacity; the server
re-profiled to 11,858 blocks / 189,728 tokens against Stage A's 11,793, a 0.55%
difference that is immaterial given §1.6):

| offered req/s | contextpilot_causal achieved | tooltrie_v0 achieved |
|---|---|---|
| 1 | 1.0075 | 1.0075 |
| 2 | 2.0124 | 2.0124 |
| 4 | 4.0142 | 4.0139 |
| 8 | 7.9890 | 7.9849 |
| 16 | **12.8818** | **11.3241** |
| 32 | 14.6634 | 11.2891 |
| 64 | **16.3907** | **11.2450** |

Both saturate between 8 and 16 req/s. **Ceiling: ~16.4 req/s for
`contextpilot_causal` against ~11.3 for `tooltrie_v0` — 45% more throughput
capacity from ordering alone.**

The ranking survives saturation and widens. TTFT (ms):

| rate | cp p50 | trie p50 | cp p99 | trie p99 |
|---|---|---|---|---|
| 8 | 103.1 | 147.9 | 395.4 | 628.3 |
| 16 | 380.7 | 3388.7 | 2107.2 | 5181.4 |
| 32 | 2456.9 | 6543.8 | 6954.9 | 11115.8 |
| 64 | 3303.4 | 8164.4 | 8715.3 | 14131.3 |

**Reuse is invariant across the entire 64x load range** — 96.16% and 87.19% at
every one of the seven rates, to the digit, including deep in saturation.
Ordering determines reuse; load does not touch it.

---

### 1.8 Transfer to Qwen3-4B

Everything above is Qwen3-0.6B. The transfer argument was that reuse depends on
the token sequence and cache capacity rather than parameter count, and that the
between-arm comparisons are ratios which survive uniform cost scaling. That was
an argument, not a measurement. It was tested directly: Qwen3-4B on an idle GPU,
its own port, native capacity read back as 6,025 blocks = **96,400 tokens**
(within 0.45% of the 96,832 recorded for 4B in `consolidated-report.md:57`),
replaying the identical frozen orderings at the identical arrival seed.

**Reuse is model-size invariant.**

| arm | 0.6B @188,688 | **4B @96,400** | 0.6B @44,656 |
|---|---|---|---|
| original | 1.19% | **1.19%** | 1.19% |
| alphabetical | 38.13% | **37.99%** | 36.65% |
| tooltrie_v0 | 87.19% | **87.21%** | 87.19% |
| contextpilot_causal | 96.16% | **96.16%** | 96.16% |

Two arms reproduce exactly. Neither deviation requires a model-size explanation:
`alphabetical` is the one arm §1.6 identified as capacity-sensitive, and 37.99%
lies monotonically between the 188,688 and 44,656 values for a cache 49%
smaller; `tooltrie_v0`'s run is **n=199** (one request lost to the same stale
keep-alive socket described in §1.5), which shifts the denominator.

**The ordering advantage grows substantially with model size.** p50 TTFT
relative to `contextpilot_causal`, rate 1:

| arm | 0.6B | **4B** |
|---|---|---|
| original | 3.03x | **154x** (saturated, see below) |
| alphabetical | 2.62x | **12.99x** |
| tooltrie_v0 | 1.25x | **2.48x** |

Against `alphabetical`, ContextPilot's lead goes from 2.62x to **12.99x** — the
advantage amplifies roughly 5x purely from a realistic model size. The mechanism
is the fixed-overhead effect noted in §1.5: the constant ~16 ms client overhead
is 18% of ContextPilot's TTFT at 0.6B but only ~3% at 4B, so the small model
compresses every ratio toward 1. **All 0.6B ratios in this report are therefore
conservative.**

**`original` cannot sustain 1 req/s at 4B.** It achieved 0.8715 against an offer
of 1.0, with p50 = 18,590 ms. The 154x figure is queue collapse, not prefill
cost, and must be reported as a capacity failure rather than a latency ratio.
For reference it sustained 4 req/s at 0.6B.

**One claim does not transfer.** §1.4 reported ContextPilot as having a
load-invariant latency *shape* (p99/p50 = 2.93 / 2.90 / 2.91). At 4B its
p99/p50 is **8.82**, because its cold-start requests now carry ~6.7x the prefill
cost while its warm requests stay cheap, so the ramp stands out far more. Full
4B distribution at rate 1 (TTFT ms):

| arm | p50 | p95 | p99 | max | p99/p50 |
|---|---|---|---|---|---|
| original | 18589.9 | 34067.2 | 36044.6 | 36821.9 | 1.94 |
| alphabetical | 1565.9 | 5641.1 | 6852.8 | 7555.3 | 4.38 |
| tooltrie_v0 | 298.8 | 737.0 | 1467.0 | 2240.9 | 4.91 |
| contextpilot_causal | **120.6** | **167.6** | **1063.9** | **1090.2** | 8.82 |

ContextPilot still wins every absolute statistic at 4B, but **the flat-shape
result is a 0.6B property and should not be generalised.**

**Under concurrent load at 4B the ranking holds and the gap explodes.**

| rate | arm | achieved | reuse | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| 1 | contextpilot_causal | 1.0051 | 96.16% | **120.6** | 167.6 | 1063.9 | 1090.2 |
| 1 | tooltrie_v0 | 1.0001 | 87.21% | 298.8 | 737.0 | 1467.0 | 2240.9 |
| 1 | alphabetical | 1.0030 | 37.99% | 1565.9 | 5641.1 | 6852.8 | 7555.3 |
| 2 | contextpilot_causal | 2.0026 | 96.16% | **134.1** | 211.3 | 1487.5 | 1657.9 |
| 2 | tooltrie_v0 | 2.0027 | 87.19% | 304.7 | 2271.7 | 2813.7 | 2973.0 |
| 2 | alphabetical | **1.2464** | 37.99% | 36059.8 | 60110.1 | 61274.4 | 63312.0 |
| 4 | contextpilot_causal | **3.9755** | 96.16% | **172.9** | 912.2 | 1910.0 | 1995.0 |
| 4 | tooltrie_v0 | **3.8405** | 87.19% | 5301.8 | 9579.1 | 10118.2 | 10214.8 |

`tooltrie_v0` p50 as a multiple of `contextpilot_causal` p50:

| rate | 0.6B | 4B | amplification |
|---|---|---|---|
| 1 | 1.25x | 2.48x | 2.0x |
| 2 | 1.25x | 2.27x | 1.8x |
| 4 | **1.27x** | **30.67x** | **24.2x** |

The amplification is a **capacity** effect, not a prefill-cost effect. At 4B and
4 req/s, `tooltrie_v0`'s 12.81% uncached prefill at ~6.7x the per-token cost
pushes it just past its service ceiling (achieved 3.8405 against an offered 4)
while `contextpilot_causal`'s 3.84% keeps it just under (3.9755). **The 8.97
point reuse gap becomes the difference between sustaining the load and
collapsing**, so a 1.27x median gap at 0.6B becomes 30.67x. `alphabetical` fails
far earlier, sustaining only 1.2464 of an offered 2 req/s with a 36-second
median.

Reuse is again invariant across load at 4B — 96.16% / 87.19% / 37.99% at every
rate, matching the 0.6B figures. 8B remains untested.

---

### 1.9 What the reordering costs: tool-selection accuracy

Parts 1-3 measure latency and cache reuse. Neither says whether the model still
picks the right tool after its menu has been rearranged. On retrieved menus it
often does not.

Method: serial replay (the concurrent driver does not capture `tool_calls`, and
accuracy depends on the prompt, not on load), `max_tokens=128`, no `ignore_eos`,
thinking disabled. The ToolRet workloads carry `gold_tool_ids`, so the metric is
whether a called tool is a gold tool. `gold_hit_ceil` restricts to requests whose
menu actually contains a gold tool, isolating the ordering's contribution from
retrieval's own misses. Every arm is a permutation of the same retrieval, so the
ceiling is identical across arms — a built-in comparability check, which passed.

| depth | arm | reuse | **accuracy** | vs `original` | mean gold-tool position |
|---|---|---|---|---|---|
| k128 | `original` | 0.37% | **22.98%** | — | **11.5** |
| k128 | `cp_online_a001` | 1.99% | **22.98%** | **0.00pp** | 27.8 |
| k128 | `tooltrie_v0` | 1.13% | 16.15% | **-6.8pp** | 62.8 |
| k64 | `original` | 0.91% | **37.09%** | — | **6.1** |
| k64 | `cp_online_a001` | 4.78% | 27.15% | -10.0pp | 12.5 |
| k64 | `tooltrie_v0` | 1.90% | 25.83% | **-11.3pp** | 31.5 |

**Two findings.**

**Reordering for cache reuse costs accuracy, and ToolTrie's cost is
substantial** — 6.8 points at k128, 11.3 at k64. The project's earlier quality
work is all on BFCL padded menus; this appears to be the first measurement on
retrieved ones. Every latency and reuse figure in Parts 1-3 should be read
alongside it.

**ContextPilot at k128 is the exception: 5.4x the reuse of `original` at exactly
zero accuracy cost** (22.98% against 22.98%). No other policy achieves that, and
this arm has `annotations_enabled: False`, so it does so without the order
annotations its paper introduces for the purpose.

**The mechanism is position, not policy.** Accuracy tracks how deep the gold
tool ends up — the lost-in-the-middle effect ContextPilot's paper cites (Liu et
al. 2023) as the reason alignment costs accuracy. ContextPilot preserves
accuracy because it keeps the gold tool shallowest of any reordering arm (12.5
against 24-32 at k64), not because of anything specific to clustering.

This exposes a structural tension that governs the whole problem. **The cache
wants the *common* tools first; the model wants the *relevant* tools first; and
automatic prefix caching only reuses a *leading* prefix, so both compete for the
front of the prompt.** Padded-64 hides it, because there "common" is 63 of 64
tools.

A caveat that cuts against generalising: Qwen3-0.6B may be unusually
order-sensitive. ContextPilot's Table 1 reports newer, larger models showing
near-zero sensitivity to input ordering. No arm here used order annotations.

---

## Part 2 — The mean-vs-max dispatch adaptor

### 2.1 What it is

Both dispatch policies sit behind a shared in-flight cap, so the only difference
is which pending request is chosen next:

- `fifo` — arrival order.
- `affinity` — among the first `W` pending requests, dispatch the one sharing the
  longest leading `tool_ids` prefix with the last dispatched request, subject to a
  fairness bound `D`: any request that has waited >= `D` seconds goes first.

Grouping prefix-affine requests is intended to lower mean latency; it delays the
requests it steps over, raising max. `D` is the tradeoff knob. Both policies are
purely client-side — the vLLM server and its prefix cache are unmodified.

**This had never been run.** All 22 prior runs were `"policy": "fifo"`.

### 2.2 Two mechanical traps found before drawing conclusions

**Trap 1 — the cap.** With `--max-inflight 0` (used by 21 of 22 prior runs) the
dispatch loop drains the pending list every tick, so `choose_index` never sees
more than one candidate and affinity **silently degenerates to FIFO**. Running
the sweep as previously configured would have produced a confident null for a
purely mechanical reason. Fixed by pinning `--max-inflight 4`, which is also
where the measured `kv_cache_max_concurrency` (4.61) sits.

**Trap 2 — no signal.** Even with a queue, the adaptor is inert if candidates
share no leading prefix. Shared leading `tool_ids` prefix length over all request
pairs within a 32-request window:

| arm | pairs with shared prefix > 0 | mean prefix length |
|---|---|---|
| original | **1 / 5,872 (0.0%)** | 0.01 |
| alphabetical | 5,744 / 5,872 (97.8%) | 13.77 |
| tooltrie_v0 | 5,872 / 5,872 (100%) | 55.81 |
| contextpilot_causal | 5,840 / 5,872 (99.5%) | 62.66 |

`choose_index` selects the first index with a *strictly* greater score, so when
every candidate scores 0 the tie resolves to index 0. On `original` the adaptor
is therefore **FIFO by construction** — confirmed empirically at 0/200 positions
reordered across all four configs. This is not a bug; it is the correct
statement that BM25-ranked orderings expose no cross-request prefix to exploit.

Activation, measured directly:

| arm | queue formed (qd p50) | reordered vs FIFO | verdict |
|---|---|---|---|
| original | yes, deep (5.25 s) | 0/200 | inert — no affinity signal |
| alphabetical | yes (654 ms) | **110-122/200**, max displacement 41 | **exercised** |
| tooltrie_v0 | **no (1.2 ms)** | 0/200 | **untested** — too fast to backlog |

### 2.3 Correction: TTFT is the wrong metric for this experiment

TTFT is timed from **dispatch**, so it is structurally blind to the delay the
adaptor imposes — the wait lands entirely in queue time, before that clock
starts. The user-visible metric is **arrival -> first token** (`queue_delay +
ttft`). On TTFT the adaptor looks like a flat null; it is not.

This does not affect Part 1. Stage A, the order control and the capacity
replication all ran uncapped, where queue delay p50 was 0.8 ms and TTFT matches
arrival-to-first-token to within 1.8 ms.

### 2.4 The frontier

`alphabetical`, rate 4, in-flight cap 4, capacity 188,688 — arrival-to-first-token (ms):

| config | mean | p50 | p95 | p99 | max | reordered | max displacement |
|---|---|---|---|---|---|---|---|
| `fifo` | 1172.0 | 1046.0 | 2521.6 | 2881.7 | 3120.2 | — | — |
| `aff W=32 D=0.5` | 1147.8 | 1051.8 | 2491.0 | 2849.7 | 3084.7 | 21/200 | 3 |
| **`aff W=32 D=2.0`** | 1122.2 | **827.2** | 2814.8 | 3182.3 | **3302.6** | 104/200 | 8 |
| `aff W=32 D=0` | 1085.3 | **715.9** | 2985.2 | 6855.9 | **12136.4** | 110/200 | 41 |
| `aff W=8 D=0` | 1078.7 | 724.3 | 2859.5 | 6851.7 | 12147.0 | 115/200 | 41 |

**The adaptor works and `D` traces the frontier:**

| D | p50 change | max change |
|---|---|---|
| 0.5 | +0.6% (no gain) | -1.1% (no damage) |
| **2.0** | **-20.9%** | **+5.8%** |
| 0 | **-31.6%** | **+289%** (3.1 s -> 12.1 s) |

`D=2.0` is the operating point: two-thirds of the median gain for almost no tail
damage. `D=0.5` is too tight to permit useful reordering and reverts to FIFO.
`D=0` maximises the mean gain and destroys the tail. **`W` is not the knob** —
`W=8` and `W=32` are within noise of each other; `D` is.

The same frontier appears at 44,656-token capacity (`D=0`: p50 1073.3 -> 773.7,
max 3083.5 -> 12401.4), so the tradeoff is **capacity-independent**.

**The gain is not a cache effect.** Reuse is unchanged (36.65% -> 36.64%).
vLLM's APC is content-addressed by prefix hash, not ordered by recency of
dispatch, so clustering affine requests does not produce more hits. The median
gain comes from queueing discipline — affine requests in flight together drain
the bulk faster, while the requests repeatedly stepped over accumulate the
12-second tail. A prediction that capacity pressure would activate a *cache*
benefit was tested at 44,656 and **falsified** on the one arm where it could be
tested.

### 2.5 Correction: admission control is not the lever it appeared to be

An earlier reading of this data claimed that capping in-flight requests bought an
8x tail reduction for 6% throughput. That was the same TTFT error and is
**retracted**. `original` at rate 4:

| | TTFT p50 | TTFT max | arrival->first-token p50 | arrival->first-token max | achieved |
|---|---|---|---|---|---|
| uncapped | 3310.4 | 7476.0 | **3312.2** | **7478.0** | 3.5645/s |
| in-flight cap 4 | 514.3 | 909.3 | **5927.9** | **11475.5** | 3.3537/s |

Capping did not shrink the queue, it **moved it from the server to the client**,
and cost throughput doing so. User-visible latency is **1.8x worse at p50 and
1.5x worse at max**. The correct sign of this effect is negative.

### 2.6 How this is normally done, and where the design diverges

The adaptor is a weak variant of a standard algorithm. **Longest Prefix Match
(LPM)** is the canonical prefix-aware scheduling policy; SGLang implements it
alongside `DFS_WEIGHT`, while vLLM's waiting queue remains FCFS and
prefix-agnostic. The design here diverges from LPM in four ways, each of which
makes it weaker:

| | this adaptor | standard practice |
|---|---|---|
| Affinity reference | the **last dispatched request** only — a myopic greedy chain | each waiting request matched against the **global radix tree**, i.e. actual cache contents |
| Operation | pick one winner per dispatch | **sort the whole waiting queue** by matched prefix length (O(W·T + W log W), falling back to FCFS above W>128) — or compose the batch directly |
| Fairness | absolute wall-clock cutoff `D` | **deficit counters** (DLPM / D²LPM, k-LPM), giving a fairness *guarantee* and work conservation |
| Layer | a **client**, ahead of one replica, creating a second queue in front of the engine's own | inside the **engine scheduler**, or in a **router across replicas** |

The wall-clock bound is visibly the wrong primitive even in this data: `D=0.5`
was tight enough to forbid nearly all reordering (21/200) while `D=0` allowed a
12.1-second wait, and there was no principled way to select `D` between them. A
deficit counter requires no such choice. Relevant also, the scheduling problem
with prefix reuse under TTFT constraints is **NP-hard**, and both FCFS and greedy
LPM have documented failure modes under latency constraints — a greedy heuristic
was never going to be near-optimal.

**The null result is regime-specific, not a refutation of prefix-aware
scheduling.** The mechanism pays when a request might land somewhere its prefix
is *not* cached. With one replica and a content-addressed cache large enough to
hold the working set, every request hits the same cache regardless of dispatch
order, so reordering has nothing to buy — which is precisely what was measured
(reuse 36.65% -> 36.64% after reordering 110/200 requests). §1.6 independently
confirms there was no eviction pressure to exploit: three of four arms showed
**0.00pp** reuse change under a 4.2x capacity cut. **These experiments were run
in the one regime where the mechanism cannot pay.** Reported gains in the
literature come from cross-replica routing — llm-d measured 57x faster TTFT and
2x throughput against round-robin on 8 pods / 16 H100s — or from genuine
cache-pressure regimes, neither of which a single uncontended GPU exhibits.

**Consequence for this project.** Tool ordering and LPM are orthogonal and
composable: ordering determines *what the prefix is*, LPM determines *who is
scheduled when*. The ordering contribution sits upstream of the scheduling
literature and feeds it rather than competing with it. Building a scheduler was
not necessary and, on this testbed, could not have been evaluated fairly.

References: [DLPM/D²LPM](https://arxiv.org/abs/2501.14312),
[k-LPM and NP-hardness](https://arxiv.org/pdf/2502.04677),
[batch size vs prefix homogeneity](https://arxiv.org/abs/2605.06046),
[SGLang RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/),
[Ray PrefixCacheAffinityRouter](https://docs.ray.io/en/latest/serve/llm/prefix-aware-request-router.html),
[vLLM Router](https://vllm.ai/blog/2025-12-13-vllm-router-release),
[llm-d routing](https://developers.redhat.com/articles/2026/01/13/accelerate-multi-turn-workloads-llm-d),
[vLLM priority-scheduling RFC](https://github.com/vllm-project/vllm/issues/6077).

---

### 2.7 Verdict: the adaptor is not worth deploying

The frontier in §2.4 is real, but it is a second-order optimisation on top of a
first-order decision worth roughly an order of magnitude more. Like-for-like at
rate 4, in-flight cap 4, arrival-to-first-token (ms):

| | p50 | p95 | max |
|---|---|---|---|
| `alphabetical`, plain FIFO | 1046.0 | 2521.6 | 3120.2 |
| `alphabetical` + **best** adaptor setting (D=2.0) | **827.2** | 2814.8 | 3302.6 |
| `tooltrie_v0`, **no adaptor at all** | **122.8** | 620.8 | 915.6 |

**The best the adaptor achieves on a mediocre ordering is 6.7x worse than doing
nothing on a good one.** Its ceiling (-20.9% median) is far below the gap that
ordering alone closes.

The structural objection is stronger than the arithmetic. The adaptor requires
candidates that share leading tool prefixes — exactly the property good ordering
policies create and bad ones lack (§2.2). On `original`, the arm with 3.3-second
medians that most needs help, it is FIFO by construction and cannot act at all.
On `tooltrie_v0` and `contextpilot_causal` the affinity is present but they never
backlog, so there is nothing to reorder. **It is inert where it is needed and
unnecessary where it would work.**

Nor is the measured gain the mechanism it was designed around: reuse is unchanged
(36.65% -> 36.64%), so the median improvement is a queueing artifact, not a cache
effect.

**This null is weak evidence, not a clean negative result.** It comes from a
non-standard implementation (§2.6) evaluated in a regime where the mechanism
cannot pay (§2.6, §1.6). It should not be cited as evidence about prefix-aware
scheduling in general.

---

## Part 3 — How the trie reduces parallel request latency

### 3.1 Mechanism: prefix caching is a sequential dependency

Sections 1.3-1.7 measure the outcome. This section isolates the mechanism,
which turns out to be parallel-specific rather than a scaled-up version of the
serial one.

**Prefix caching is a sequential dependency.** Request N can only reuse a prefix
that some earlier request has already computed *and committed*. Requests that
overlap in time therefore cannot help each other — they all miss the same cold
prefix and each computes it redundantly. The size of that "thundering herd" is
the number of requests dispatched before the first completion:

| offered req/s | tooltrie_v0 | contextpilot_causal |
|---|---|---|
| 1 | 1 | 1 |
| 2 | 1 | 1 |
| 4 | 7 | 6 |
| 8 | **37** | **17** |
| 16 | **109** | **48** |
| 32 | **200** | 124 |
| 64 | 200 | 200 |

At 32 req/s **every one of ToolTrie's 200 requests is in flight before the first
one finishes** — the entire workload runs cold. ContextPilot reaches that point
only at 64.

**This compounds.** Lower reuse means longer service time, which means more
requests arrive before the first completes, which means a larger herd, which
means more redundant prefill. ToolTrie's herd is consistently ~2x ContextPilot's
(37 vs 17, 109 vs 48). The reuse gap of 8.97 points at concurrency 1 becomes a
2x difference in how many requests get no cache benefit at all.

**Warm-up length differs too.** Mean TTFT (ms) by arrival-index bucket:

| arm | rate | [0-3) | [3-6) | [6-10) | [10-20) | [20-50) | [50-200) |
|---|---|---|---|---|---|---|---|
| tooltrie_v0 | 1 | 284 | 196 | 124 | 144 | 125 | 102 |
| tooltrie_v0 | 4 | 303 | **451** | 135 | 183 | 157 | 113 |
| contextpilot_causal | 1 | 262 | 87 | 92 | 95 | 89 | 86 |
| contextpilot_causal | 4 | 286 | 177 | 103 | 107 | 98 | 94 |

ContextPilot reaches steady state after roughly **3** requests; ToolTrie needs
**6-10**. And ToolTrie at rate 4 shows an inversion — bucket [3-6) is *worse*
than [0-3) at 451 vs 303 ms — which is the herd signature: requests 3-5 are
dispatched before 0-2 have committed anything to the cache, so they miss a
prefix that is already being computed three times over.

**So ToolTrie reduces parallel latency by exactly one mechanism** — raising reuse
38.13% -> 87.19%, shortening prefill, which under concurrency compounds into a
higher admission ceiling (11.3 vs `original` never exceeding ~3.6 req/s). Its
parallel-specific *weakness* is the slower ramp: the trie needs more requests to
build its shared prefix, and concurrency multiplies the cost of that ramp because
overlapping requests cannot warm each other.

The ramp is nonetheless a small share of total cost at these scales: the first 10
of 200 requests account for 8.6% of all TTFT at rate 1 and 10.7% at rate 4
(ContextPilot: 7.9% and 9.0%). The ramp explains the *tail*, not the median.

---

## Part 4 — Larger tool sets and smart queuing

### 4.1 On genuinely retrieved menus, ordering barely matters

Parts 1-3 all use **padded-64** menus. `reports/frequency-online/findings.md`
warns that this workload "mainly rewards discovering its nearly fixed 63-tool
core" and that results on it are "not evidence that frequency counting is the
best policy on genuinely retrieved menus". That warning was tested directly by
replaying the BM25-retrieved workloads under load, at four retrieval depths.

`k64` is the controlled comparison: 6,896 canonical tool tokens per request
against padded-64's 6,903 — the same prompt volume, real retrieval instead of
padding. Rates were scaled to hold the offered *token* rate roughly constant
(k4@16, k16@8, k64@4, k128@2) so load is comparable across depths rather than
confounded by menu size. ContextPilot was added to these workloads at the
paper's `alpha=0.001`, in both the static-refit and official online-API forms.

**Reuse:**

| workload | original | alphabetical | frequency | tooltrie_v0 | CP refit a=.001 | CP online a=.001 | spread |
|---|---|---|---|---|---|---|---|
| k4 | 15.87% | 15.28% | 14.62% | 17.48% | 18.42% | **18.72%** | 4.10pp |
| k16 | 6.12% | 6.27% | 5.59% | 7.77% | 9.80% | **9.93%** | 4.34pp |
| k64 | 0.91% | 1.22% | 0.94% | 1.90% | 4.01% | **4.78%** | 3.87pp |
| k128 | 0.37% | 0.58% | 0.54% | 1.13% | **2.96%** | 1.99% | 2.59pp |
| **padded-64** | 1.19% | 38.13% | 87.19%¹ | **96.16%**² | 96.16% | 96.16% | **94.97pp** |

¹`tooltrie_v0`  ²`contextpilot_causal` (a=0.5). All three ContextPilot variants
reach an identical 96.16% on padded-64: with a near-constant tool core the
overlap term of Eq. 1 does not vary between pairs, so the clustering is
invariant to a. Padded-64 cannot discriminate `alpha` at all.

**Latency spread across arms, and achieved rate:**

| workload | offered | achieved (range) | p50 range (ms) | p50 spread |
|---|---|---|---|---|
| k4 | 16 | 15.907 - 15.924 | 47.6 - 52.3 | 1.10x |
| k16 | 8 | 7.998 - 8.001 | 99.3 - 101.7 | 1.02x |
| k64 | 4 | 2.623 - 2.635 | 9559.9 - 9760.9 | **1.02x** |
| k128 | 2 | 1.004 - 1.007 | 35192.4 - 35852.7 | **1.02x** |
| **padded-64** | 4 | 3.564 - 4.014 | 91.8 - 3310.4 | **36.06x** |

**The 94.97-point reuse spread that drives every result in Parts 1-3 collapses
to under 3 points on real retrieval, and the 36x latency spread collapses to
1.02x.** At k64 all four arms saturate identically at ~2.63 req/s against an
offered 4 and land within 2% of each other on every percentile.

The mechanism is the one the frequency-online report predicted. Padding fills
each request up to 64 tools from a nearly fixed filler core, so most of every
menu is shared across requests and ordering can align it into one long prefix.
BM25 returns a genuinely different set per query, so **there is no shared prefix
for any ordering to exploit.** `contextpilot_causal`'s 96.16% is substantially a
property of the padded workload rather than of tool ordering in general.

**ToolTrie-v0 beats every simple heuristic but loses to ContextPilot at all
four depths.** It is ahead of `original`, `alphabetical` and `frequency`
everywhere — its cleanest sweep in the study, since on padded-64 it loses to
both ContextPilot arms. But correctly-configured ContextPilot beats it at every
depth, and the margin *grows* with menu size:

| depth | tooltrie_v0 | best ContextPilot | CP / ToolTrie |
|---|---|---|---|
| k4 | 17.48% | 18.72% | 1.07x |
| k16 | 7.77% | 9.93% | 1.28x |
| k64 | 1.90% | 4.78% | **2.52x** |
| k128 | 1.13% | 2.96% | **2.63x** |

An earlier draft of this section reported the growing-margin trend as ToolTrie's,
measured against a field that excluded ContextPilot. With ContextPilot included
the trend runs the other way. ToolTrie's honest position on retrieved menus is
**second, ahead of every naive heuristic**.

The reuse ratios overstate the practical gap. What matters is *points of prefill
removed*: at k64 ContextPilot's 2.52x ratio is worth only **+2.88pp**, and the
p50 spread across all six arms is **1.02x**. On retrieved menus 95-98% of prefill
is uncacheable whatever policy is used.

**`frequency` — "put frequently used tools in front" — does not work here.** It
is the worst arm at k4 and k16 and near-worst at k64/k128, and on padded-64 it
reaches 39.69% against `alphabetical`'s 38.13%. As a standalone heuristic it is
not competitive.

### 4.2 Why queuing may pay here when it does not on padded menus

The verdict in §2.7 is scoped to this workload, and the scoping matters more
than it first appears.

A scheduler can beat FIFO on mean flow time only when jobs differ in size —
with identical jobs, FIFO is already mean-optimal and every size-based
discipline (SJF, SRPT) degenerates to it. Padded-64 makes jobs identical **by
construction**:

| workload | CV of prompt tokens | range |
|---|---|---|
| **padded-64** (all runs in this report) | **0.0038** | 5,514 - 5,668 |
| k4 retrieved | 0.4884 | 160 - 1,318 |
| k16 retrieved | 0.4036 | 766 - 5,437 |
| k64 retrieved | 0.3314 | 3,407 - 18,049 (5.3x) |
| k128 retrieved | 0.3012 | 7,117 - 27,646 |
| mixed k4-k128 | **1.0079** | 160 - 27,646 (173x) |

Job-size variance is **87x higher** on retrieved menus and **265x higher** on a
mixed tool-set workload. Head-of-line blocking — a long request occupying the
server while short ones queue behind it — is exactly the regime where SJF wins,
and it is exactly the regime that larger and more varied tool sets produce.
**The null in §2.4 and the verdict in §2.7 therefore apply to padded menus and
must not be generalised to realistic retrieval workloads.**

**Two things make this domain unusually favourable for SJF.** The published work
on SJF for LLM serving (PARS, EWSJF, Clairvoyant, ELIS, learning-to-rank
scheduling) spends most of its effort *predicting* job size, because output
length is unknown at arrival. Here prefill dominates completely — 6,896 to
27,646 prompt tokens against 48 output tokens — and **prompt length is known
exactly at arrival**, so SJF is implementable with zero prediction error.

Further, under prefix caching the real job size is not the prompt but the
**uncached suffix**. A policy that sorts by shortest uncached suffix is
simultaneously SJF *and* prefix-aware: in this domain SJF and LPM collapse into
one policy. That, rather than the affinity adaptor of §2.1, is the design worth
building if scheduling is pursued.

**Caveat on the available upside.** Chunked prefill (Sarathi-Serve) is the
standard mitigation for long-prompt head-of-line blocking and is **already the
default in vLLM**. Every run in this report used
`--max-num-batched-tokens 8192`, so the baseline already includes it. Any SJF
gain would be additional to chunked prefill, not a replacement for it.

**Status: untested.** No scheduling policy has been evaluated on a
heterogeneous-job workload. This is the one place where the queuing question
remains genuinely open.

References: [PARS / prompt-aware scheduling](https://arxiv.org/html/2510.03243),
[EWSJF](https://arxiv.org/html/2601.21758v1),
[Clairvoyant predictive SJF](https://arxiv.org/pdf/2606.07248),
[ELIS response-length predictor](https://arxiv.org/pdf/2505.09142),
[Efficient LLM Scheduling by Learning to Rank](https://arxiv.org/pdf/2408.15792),
[Sarathi-Serve chunked prefill](https://arxiv.org/html/2403.02310).

---

### 4.3 Smart queuing: it trades the tail for the median, and cannot reduce both

The question was whether size-aware queuing reduces **tail** latency for larger
tool sets. §4.2 argued the regime was favourable: retrieved menus are
heterogeneous (k64 CV 0.3314), prefill dominates so job size is known exactly at
arrival from `canonical_tool_tokens`, and no scheduling policy had been tested
there. Three policies were added to `scripts/replay_vllm_concurrent.py`:
`sjf` (by job size), `suffix` (size discounted by the fraction of the menu
already covered by a dispatched prefix — SJF ∩ prefix-aware, matched against a
trie of *all* dispatched sequences rather than only the last), and `random` (a
control, explained below). All run behind a fixed `--max-inflight 4`; comparisons
are policy-vs-policy at that fixed cap, never against uncapped (§2.5).

**Answer: no.** Every configuration that improved the median made the tail
worse, and the only configuration that did not hurt the tail also gave up the
median gain. Arrival-to-first-token, k64 at 4 req/s:

| policy | mean | p50 | p99 | max |
|---|---|---|---|---|
| `fifo` | 13156 | 11227 | 28842 | 29251 |
| `sjf` | 8944 (**-32.0%**) | **1628 (-85.5%)** | **60932 (+111.3%)** | 66490 |
| `sjf` + aging 2000 | 12735 (-3.2%) | 10766 (-4.1%) | 28522 (-1.1%) | 31387 |

k128 has the same shape: p50 36128 → **4470 (-87.6%)**, p99 98224 →
**157021 (+59.9%)**.

**The aging frontier** (k64, tokens/s of score forgiven per second waited):

| aging | mean | p50 | p99 |
|---|---|---|---|
| 0 | -32.0% | -85.5% | **+111.3%** |
| 100 | -34.7% | -84.0% | +56.2% |
| **250** | **-24.6%** | **-47.3%** | **+19.7%** |
| 500 | -14.0% | -7.8% | +19.5% |
| 1000 | -7.9% | -6.4% | +12.0% |
| 2000 | -3.2% | -4.1% | -1.1% |

**No point on this frontier improves the mean and the tail together.** The knee
is around `aging=250`: most of the mean gain for a contained +19.7% p99. At
`aging=2000` the tail is finally protected, but the policy has degenerated to
approximately FIFO.

#### The saturation artifact, and why a `random` control was necessary

The negative control initially appeared to fail. On **padded-64** — where jobs
are near-identical by construction, CV 0.0038, a 154-token range — `sjf` still
produced the same qualitative shape (p50 -37.0%, p99 +55%) once the workload was
driven hard enough to form a real queue. A sort key carrying almost no
information should do nothing.

The explanation is that under a deep queue FIFO makes waiting time grow roughly
with arrival position, so **any** order uncorrelated with arrival reshapes the
distribution — many requests served earlier, a few starved — lowering the median
and raising the tail *regardless of whether the priority key is meaningful*. A
`random` dispatch policy measures that baseline, and a size-aware policy may
only be credited with the margin it wins **above random**:

| workload | leaving FIFO alone (`random`) | attributable to size information (`sjf` − `random`) |
|---|---|---|
| k64 (CV 0.3314) | mean -2.8%, p50 -27.4% | **mean -29.2%** |
| padded-64 (CV 0.0038) | mean -5.8%, p50 -27.5% | mean -3.7% |

The artifact is real and almost identical in both workloads (p50 -27.4% vs
-27.5%), which confirms the mechanism. But it is small on the mean, and once it
is subtracted **the size-aware gain is still 29.2 points on k64 against 3.7 on
padded-64 — an 8x separation tracking the 87x difference in job-size variance.**
SJF genuinely works here; roughly a quarter of its apparent *median* improvement
is artifact and must not be claimed.

Without the `random` control the headline would have been "SJF cuts median
latency 6.9x", crediting a policy for an effect that a coin flip reproduces in
part.

#### `suffix` is indistinguishable from `sjf`, as predicted

Mean 9074 vs 8944, p50 1604 vs 1628 — inside run-to-run noise. Predicted in the
plan: reuse on k64/k128 is 0.37-1.98%, so the uncached suffix is ~99% of every
prompt and the prefix-aware term has nothing to discount. This is a property of
the available workloads, not a failure of the policy — no workload in this
project has high reuse *and* high size variance simultaneously, so the two
mechanisms can be validated separately but never in combination.

#### Verification

All 23 runs passed the integrity checks: reuse within 0.1pp across policies on
each workload, `inter_token_latency_count` exactly 200x47 (no foreign traffic),
and every policy confirmed to have acted (191/200 reordered at k64, max
displacement 174). Two Stage B configurations were flagged **inert** by the
diagnostic (`padded64 sjf-age2000` and `suffix-age2000`, 0/200 reordered) and
their nulls are therefore mechanical, not scientific; the rate-16 rerun replaces
them. Reported on arrival-to-first-token throughout — TTFT understated the
effect by 13-17x, since it is timed from dispatch and cannot see queueing.

## Part 5 — Also investigated

Three lines outside the four research questions, recorded so they are not
repeated. Two were rejected on measurement; the third is the reason why.

### 5.1 A canonical global tool order

**Idea.** Sort every request's tools by one global frequency ranking. No index,
no clustering, no distance function, no `alpha` — one sort per request, against
ContextPilot's O(N^2) index build (8 s per 2,000 contexts in its paper).

| depth | tooltrie | canonical *causal* | canonical *oracle* | best CP |
|---|---|---|---|---|
| k64 | 1.90% | 2.75% | **4.28%** | 4.78% |
| k128 | 1.13% | 1.99% | **2.87%** | 2.96% |

It reaches **90-97% of ContextPilot's reuse with none of its machinery**, and
beats ToolTrie by 2.3-2.5x. The deployable *causal* variant, which counts only
strictly earlier requests, reaches just 58-67% of ContextPilot — the oracle's
advantage comes from knowing global frequency in advance.

**Verdict: rejected on accuracy (§5.3), not on reuse.**

### 5.2 Trie inside ContextPilot: clustered head, canonical tail

**Idea.** ContextPilot emits "matched prefix + remaining documents in their
original order", so its tail keeps each request's BM25 ranking and can never be
shared. This keeps its head untouched and canonicalises only the tail, layering
the two mechanisms rather than comparing them. Head defined causally as the
longest prefix shared with any strictly earlier request.

| depth | CP online | **hybrid** | vs CP online | vs best CP |
|---|---|---|---|---|
| k64 | 4.78% | 4.17% | **-12.7%** | 0.87x |
| k128 | 1.99% | **3.11%** | **+56.2%** | **1.05x** |

At k128 the hybrid is the best policy in the study on reuse, p50 and throughput
simultaneously — the only arm anywhere that beats the strongest ContextPilot
variant. It helps exactly where ContextPilot's tail ordering is doing no work
(k128 has 8.32 shared tools per pair against k64's 2.78) and hurts where it is.

**Verdict: rejected.** The k128 win is +1.12pp of reuse for **-9.3pp of
accuracy** (§1.9): an exchange rate of **8.3 points of accuracy per point of
reuse**. The winning variant is also oracle-ranked; the deployable causal hybrid
reaches 2.15%.

### 5.3 Why both were rejected, and what it revealed

Accuracy was measured before either was pursued further, and it killed both:

| depth | arm | reuse | accuracy | mean gold position |
|---|---|---|---|---|
| k128 | `original` | 0.37% | **22.98%** | **11.5** |
| k128 | `cp_online_a001` | 1.99% | **22.98%** | 27.8 |
| k128 | `canonical_oracle` | 2.87% | 14.29% | 56.5 |
| k128 | `hybrid_oracle` | **3.11%** | **13.66%** | 57.7 |

Accuracy falls monotonically with how deep the gold tool is pushed. Canonical
ordering sorts by *global* frequency, so a request's own top-ranked tool can be
sent to position 56 of 128; ContextPilot moves only tools inside a matched
cluster and leaves the rest in relevance order, keeping the gold tool at 27.8.

That is the finding worth keeping from this line, and it generalises beyond it:
**cache reuse and answer accuracy compete for the front of the prompt**, and an
ordering policy is only viable to the extent that it buys prefix sharing without
displacing the relevant tool. It is stated in full in §1.9.

**What would revive these methods.** No arm tested here used ContextPilot's
**order annotations**, which exist precisely to decouple relevance from position
and which its paper reports can lift accuracy above the unordered baseline.
Canonical and hybrid displace the gold tool most, so they would gain most. If
annotations recover the 9.3 points, the exchange rate changes and this line
reopens; if they do not, it is closed.

---

## Limitations

1. **Parts 1-3 characterise a padded workload, not tool ordering in general
   (§4.1).** On BM25-retrieved menus of the same size the between-arm reuse
   spread falls from 94.97pp to 0.98pp and the p50 spread from 36.06x to 1.02x.
   Padding supplies a nearly fixed tool core that ordering can align; real
   retrieval does not. This is the single most important scope bound in the
   report.
2. **Model size is only partly controlled (§1.8).** Reuse was confirmed
   model-size invariant at Qwen3-4B and the ordering advantage was shown to
   *grow* (2.62x -> 12.99x against `alphabetical`), so the 0.6B ratios are
   conservative, and the ranking was confirmed under concurrent load at 4B where
   the gap amplifies 24.2x. Untested: 8B entirely. One claim was falsified by the 4B run —
   ContextPilot's load-invariant p99/p50 shape is 0.6B-specific.
3. **Queuing was tested only under saturation, and only at one in-flight cap.**
   §4.3 answers the scheduling question, but every run there is deeply
   backlogged (queue delay p50 10-34 s) at `--max-inflight 4`. The saturation
   artifact it uncovered — any non-FIFO order lowers the median ~27% under a
   deep queue — may behave differently at lighter load or a different cap.
   Untested: preemptive disciplines (SRPT), and any policy inside the engine
   rather than in front of it.
4. **The `tooltrie_v0` adaptor cell is untested, not null.** It never backlogged
   at rate 4 / cap 4 (queue delay p50 = 1.2 ms, 0/200 reordered) at either
   capacity. It requires a tighter cap or a higher rate.
5. **`max` is a single order statistic at n=200** and varied 269.1 -> 710.2 ms
   between two runs of the same configuration. Weight p95 and p99.
6. **`original`'s saturation ceiling was never measured** — the sweep above 4
   req/s covered only `tooltrie_v0` and `contextpilot_causal`. Its ceiling is
   known only to be below 3.6 req/s on padded-64.
7. **`n=199`** in the `original` rate-1 order-control run, and in the
   `tooltrie_v0` 4B transfer run; both lost one request to a stale client-side
   keep-alive socket, not a server fault (§1.5).
8. **No ToolRet workload under load.** BFCL only, in both padded and retrieved
   form.
9. **Accuracy is measured on one model, one seed, at two depths (§1.9).**
   Absolute `gold_hit` is low (11-28%), Qwen3-0.6B may be unusually
   order-sensitive, and no arm used ContextPilot's order annotations — which
   exist precisely to mitigate the effect measured. The direction is consistent
   across both depths and all arms; the magnitudes are not transferable.
10. **The Part 5 methods are single-configuration.** Canonical and hybrid were
   each run at two depths, one arrival seed, with an oracle and a causal variant.
   They were rejected on the accuracy exchange rate, not on an exhaustive sweep.
11. **Three trials were not run per cell.** Cross-run agreement — the order
   control, and reuse identical to the digit across seven rates and three
   capacities — is the evidence for stability.

## Provenance

| stage | directory | runs |
|---|---|---|
| Stage A rate sweep | `cluster/results/concurrent-latency-20260820-212500/` | 12 |
| Order-reversal control | `cluster/results/concurrent-order-control-20260829-142216/` | 8 |
| Adaptor sweep, native capacity | `cluster/results/adaptor-sweep-20260829-143543/` | 15 |
| Capacity 44,656 + adaptor re-test | `cluster/results/capacity-44656-20260829-143821/` | 14 |
| Saturation sweep (8/16/32/64 req/s) | `cluster/results/saturation-20260829-160137/` | 8 |
| Qwen3-4B transfer test | `cluster/results/qwen3-4b-transfer-20260829-162208/` | 4 |
| BM25 k sweep under load (k4/16/64/128) | `cluster/results/bm25-k-sweep-20260829-203836/` | 16 |
| Qwen3-4B under load (rates 2, 4) | `cluster/results/qwen3-4b-load-20260829-203933/` | 5 |
| Size-aware queuing (Part 4.3) | `cluster/results/sjf-queuing-20260829-212639/` | 24 |
| ContextPilot at a=0.001 + frequency | `cluster/results/alpha001-comparison-20260829-224719/` | 17 |
| Canonical ordering (Part 5.1) | `cluster/results/canonical-order-20260830-003615/` | 8 |
| Accuracy gate (1.9, 5.3) | `cluster/results/accuracy-gate-20260830-011416/` | 10 |

**137 runs total.** Supporting scripts added: `summarize_queuing_runs.py`,
`build_canonical_ordering.py`, `score_tool_selection.py`.

Driver `scripts/replay_vllm_concurrent.py`. Server: Qwen3-0.6B, GPU 2, port
8300, `--enable-prefix-caching --max-num-seqs 64 --max-num-batched-tokens 8192`.
Capacity stage adds `--num-gpu-blocks-override 2791`, asserted against
`/metrics` before any run. The 4B transfer test runs Qwen3-4B on GPU 3, port
8301, native capacity 6,025 blocks = 96,400 tokens. Each stage holds a `flock`
(`/tmp/fyp-replay-8300.lock`, or `-8301` for 4B) so exactly one driver ever runs
against a given server.

**Known defect in the run scripts.** A server launched from inside a
lock-holding script inherits the lock file descriptor and continues to hold the
`flock` after its parent exits, deadlocking the next stage against a server it
intends to replace. This occurred once, was diagnosed via `fuser`, and is fixed
by closing the descriptor in the child (`9>&-`). No result data was affected —
no driver ran during the deadlock. Scripts written before the fix
(`run_capacity.sh`) still contain it.

A quarantine directory inside the Stage A run holds six contaminated files from
an aborted first attempt in which two drivers ran concurrently against one
server. They are excluded from every figure above and must not be cited.
