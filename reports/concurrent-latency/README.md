# Tool ordering under concurrent load — key findings

Qwen3-0.6B on one RTX 3090 (plus a Qwen3-4B check), vLLM 0.26.0, prefix caching
unmodified. 310 GPU runs.

The short version. Every number links to the section of
[`findings.md`](findings.md) that derives it, with its runs and controls.

**Two workloads, and which one a number comes from decides what it means.**

| | padded-64 | retrieved (BM25, k4–k128) |
|---|---|---|
| tools shared between two requests | **63 of 64** | 0.03 (k4) to 4.95 (k128) |
| what it is for | the **instrument** — a 36–68x p50 spread makes the mechanism visible | the **deployment case** the brief describes |
| input ordering | adversarial by construction: the one differing tool is first in every request | a relevance ranking worth preserving |
| best arm | ContextPilot / `tooltrie_v0` | **`tooltrie_v1`** |

Arms swap places between them, and that is the point rather than a
contradiction: padded menus isolate how ordering becomes reuse becomes capacity,
and retrieved menus say how much of that survives real retrieval. Questions 1–4
below are answered on whichever workload can answer them; where only padded can,
it says so.

| # | Question | Answer | on |
|---|---|---|---|
| 1 | Latency distribution at p95 / p99 / max under parallel load | Ordering is worth 68x at p50 on padded menus and 1.26x on retrieved ones | both |
| 2 | An adaptor trading mean against max | Works, not worth deploying. Ordering is worth 5.6–7.3x more, and it cannot act on the good orderings at all | padded |
| 3 | How the trie reduces parallel latency | One mechanism: it raises reuse, which compounds under load into admission capacity — worth 4.8x padded, 1.05x retrieved | both |
| 4 | Cutting tail latency by smart queuing | **No.** It cuts the median ~85% and raises the tail ~125%, on every ordering | retrieved |

---

## The headline: one line of ToolTrie was worth 1.11–2.61x

v0 sorts the tools its trie cannot match **alphabetically**. That fallback decides
~93% of every menu, and tool name correlates with neither relevance nor
commonality — so v0 permuted 99.1% of each menu on no information, displacing the
correct tool to position 62.8 of 128 to buy 1.13% reuse. **37 positions of
displacement per point of reuse, against ContextPilot's 8.**

**v1 keeps unmatched tools in the order they arrived.** It places 2.55 tools of
128 and leaves 120 of 200 requests byte-identical to their input.

| reuse | k4 | k16 | k64 | k128 |
|---|---|---|---|---|
| `original` | 15.87% | 6.12% | 0.91% | 0.37% |
| `alphabetical` | 15.28% | 6.27% | 1.22% | 0.58% |
| `frequency` | 14.62% | 5.59% | 0.94% | 0.54% |
| `tooltrie_v0` | 17.48% | 7.77% | 1.90% | 1.13% |
| ContextPilot | 18.72% | 9.93% | 4.78% | 1.99% |
| **`tooltrie_v1`** | **19.47%** | **11.31%** | **4.96%** | **2.21%** |

Wins **4 of 4 retrieved depths** and **5 of 5 arrival permutations** at k64
(p = 0.031; reuse is deterministic, so each paired comparison is exact), and 21
of 24 latency cells.

**And it holds on the retriever most deployments actually use.** Every result
above is BM25, a lexical matcher; the brief never specified one. Repeating the
whole comparison with a dense retriever (§4.4) *widens* the margin at the deep
menus rather than closing it:

| depth | v1 vs ContextPilot, dense | on BM25 |
|---|---|---|
| k4 | 0.99x (loses by 0.19pp) | 1.04x |
| k16 | 1.14x | 1.14x |
| k64 | **2.09x** | 1.04x |
| k128 | **3.09x** | 1.11x |

ContextPilot gets *worse* on dense at k64/k128 while v1 gets *better*, even
though dense menus share half as many tools. Dense also retrieves better
(hit 0.815 vs 0.755 at k64) and overlaps less, because BM25's overlap comes
substantially from hub tools — its top tool lands in 66 of 200 menus against
dense's 21. **Some of the reuse measured on BM25 menus is retrieval error that
happens to be cache-friendly.**

**On dense the accuracy stops being a wash and the reuse becomes free.**
`gold_hit_ceil`, ceiling identical across arms within each depth:

| cell | `original` | ContextPilot | **`tooltrie_v1`** | v1 − CP |
|---|---|---|---|---|
| k64 @ 0.6B | 29.45 | 24.54 | **31.29** | +6.75 |
| k128 @ 0.6B | 29.24 | 18.13 | **28.65** | **+10.52** (2.3 SE) |
| k64 @ 4B | 39.26 | 38.04 | **39.88** | +1.84 |
| k128 @ 4B | 39.18 | 36.26 | **40.35** | +4.09 |

v1 beats ContextPilot in **all four cells** (BM25 was 2 wins, 2 losses, 2 ties)
and is within noise of *not reordering at all* in every cell — so it buys 2.09x
ContextPilot's reuse at k64 and 3.09x at k128 at no accuracy cost. Only one cell
clears 2 SE alone; the result rests on four-of-four consistency plus the gold
position that predicted it (7.3 against ContextPilot's 13.7 at k64).

Accuracy against ContextPilot is a **wash** — the six cells measured are four
depths at 0.6B plus k64 and k128 repeated at 4B:

| cell | `tooltrie_v1` | ContextPilot | delta | SE | |
|---|---|---|---|---|---|
| k4 @ 0.6B | 47.57% | 47.57% | +0.00 | 5.75 | tie |
| k16 @ 0.6B | 40.46% | 41.22% | −0.76 | 6.07 | loss |
| k64 @ 0.6B | **33.11%** | 27.15% | **+5.96** | 5.27 | win |
| k128 @ 0.6B | 22.36% | 22.98% | −0.62 | 4.67 | loss |
| k64 @ 4B | 42.38% | 42.38% | +0.00 | 5.69 | tie |
| k128 @ 4B | **44.72%** | 37.27% | **+7.45** | 5.47 | win |

2 wins, 2 losses, 2 ties, and **every margin is under 1.6 SE** — the largest is
1.35, the two losses are 0.13 and 0.16. So the claim is *more reuse at no
measurable accuracy cost*, not better accuracy.

**Read the margins by comparator.** Against ContextPilot it is 1.04–1.14x, which
at k64 is +0.18pp of reuse on a workload where 95% of prefill is uncacheable
either way. Against `tooltrie_v0` it is 1.11–2.61x. **The v0 column is the
finding**: ToolTrie was leaving most of its own value unclaimed through one line.
The ContextPilot comparison shows v1 has *caught up* with the published baseline,
not passed it meaningfully.
([§5](findings.md#5-tooltrie-v1-reorder-only-what-the-trie-matched) ·
[how it works](../../notes/tooltrie-v1-design.md))

**It is regime-dependent, and this is measured:**

| input ordering | vs ContextPilot | vs `tooltrie_v0` |
|---|---|---|
| **dense relevance (k4–k128)** | **wins 3/4, by 1.14–3.09x** | wins 4/4 |
| BM25 relevance (k4–k128) | wins 4/4, by 1.04–1.14x | wins 4/4 |
| shuffled (§4.3, 25–75% shared) | loses, 0.80–0.95x | wins, ~3x |
| adversarial (`padded-64`) | loses, 0.01x | loses, 0.01x |

ContextPilot overrides its input when its clustering disagrees; v1 essentially
never does. So v1 turns a good input ordering into more reuse than ContextPilot
can, and cannot rescue a bad one — which is why it is the best arm on retrieved
menus and near the bottom on padded ones. Both appear below.

---

## 1. Latency distribution at p95 / p99 / max

This question has two answers, because it has two workloads. **Padded menus are
the instrument; retrieved menus are the deployment.** Both at 4 req/s and
size-matched — 6,903 tokens per request against 6,896 — so the only difference is
whether the menus genuinely overlap. All figures are time to first token in ms;
these runs are uncapped, so no client-side queue forms and arrival-to-first-token
is within 2.2 ms of every number below.

**Padded menus** (63 of 64 tools shared), converged requests 201–600. This is the
control that makes the mechanism visible:

| arm | reuse | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| `original` | 0.70% | 6239.7 | 15175.0 | 15659.5 | 16108.6 |
| `tooltrie_v1` | 0.70% | 4389.5 | 10493.9 | 11045.5 | 11564.3 |
| `alphabetical` | 46.29% | 271.3 | 586.6 | 820.6 | 960.4 |
| `tooltrie_v0` | 97.05% | **91.2** | **122.5** | **133.1** | **141.9** |
| ContextPilot | 97.09% | 91.7 | 123.7 | 134.3 | 151.1 |

**Retrieved menus** (k64, 1.61 of 64 shared), the regime the brief describes and
the one v1 is built for:

| arm | reuse | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| `original` | 0.91% | 9667.9 | 24699.7 | 26490.8 | 26820.4 |
| `alphabetical` | 1.22% | 9727.5 | 24854.3 | 26781.7 | 27104.6 |
| `frequency` | 0.94% | 9760.9 | 25009.1 | 26842.3 | 27161.5 |
| `tooltrie_v0` | 1.90% | 9559.8 | 24825.6 | 26691.0 | 27059.1 |
| ContextPilot | 4.78% | 8565.4 | 22452.4 | 25163.5 | 25432.4 |
| **`tooltrie_v1`** | **4.96%** | **7739.7** | **21219.8** | **23824.2** | **24177.4** |

**Ordering is worth 68x at p50 on padded menus and 1.26x on retrieved ones.**
That collapse is the single most important qualification in this report — 95–98%
of prefill on retrieved menus is uncacheable whatever policy is used, so there is
far less for any ordering to win.

**The two tables answer different halves of the question, and the arms swap
places between them.** On padded menus `tooltrie_v1` is second-worst: its rule is
to preserve the input ordering, and padded's input puts the differing tool at
position 0 of every request, so it preserves the worst possible layout. On
retrieved menus, where the input is a BM25 relevance ranking worth preserving,
the same rule makes it the **fastest arm on every statistic**. That is not a
contradiction — it is the boundary condition measured from both sides, and it is
why padded is the mechanism control rather than the deployment case.

**Serial testing hid all of this.** At concurrency 1 every arm sat within 1.25x
on every statistic and `max` was flat at ~270 ms. Under load `max` is dominated
by queueing, which one-at-a-time testing cannot produce.

**And a badly ordered arm has no fixed distribution.** `original` sustains only
3.51 of an offered 4 req/s on padded menus, so its backlog grows and its latency
grows with run length — p50 3310 ms over 200 requests, 6240 over 600. The ordered
arms are stable. So the ratio depends on how long you run (36x at 200, 68x at
600), and that divergence is the finding, not any single number.

On padded menus the bottom two rows are a **tie** — every gap is inside
measurement noise, and from request 222 both arms place the odd tool identically.
Over the first 200 requests ContextPilot leads v0 by 8.97pp of reuse and 22 ms of
p50; that gap is **warm-up only**. `frequency` is absent from the converged table
because it is a *fitted* baseline whose training corpus is not recorded
recoverably; over the first 200 it scores 39.69% reuse and 317.9 ms p50, tracking
`alphabetical` within 1.6pp and 14 ms.
([§1.1](findings.md#11-results) · [§1.5](findings.md#15-why-the-two-windows-differ-warm-up) ·
[§4.1](findings.md#41-on-real-retrieved-menus-ordering-barely-matters))

---

## 2. An adaptor trading mean against max

**Measured on padded menus only.** The adaptor dispatches to maximise shared
prefix with what is already resident, so it needs a workload with prefix to be
affine to; at 1–5% reuse on retrieved menus it has almost nothing to act on.
Question 4 covers dispatch on retrieved menus with a stronger policy.

It works, and `D` — how long a request may be held back — is the knob. Arrival to
first token, ms, at 4 req/s behind an in-flight cap of 4:

| policy | p50 | max | reordered |
|---|---|---|---|
| `frequency` | 825.2 | 2640.3 | — |
| `frequency` + adaptor | **685.2** (−17.0%) | 2961.4 (+12.2%) | 105/200 |
| `alphabetical` | 1046.0 | 3120.2 | — |
| `alphabetical` + adaptor | **827.2** (−20.9%) | 3302.6 (+5.8%) | 104/200 |
| `tooltrie_v0` | **122.8** | **915.6** | 0/200 |
| **ContextPilot** | **94.2** | **452.6** | 0/200 |
| `tooltrie_v1` | 5052.9 | 10159.5 | 0/200 |

The shape is right — median down, worst case up, about half the requests moved.
**But the best it achieves anywhere is 685 ms, against 123 for ToolTrie and 94
for ContextPilot, neither of which uses it.** Fixing the ordering is worth
5.6–7.3x more than fixing the dispatch order — **on padded menus.** That exchange
rate does not transfer: ordering is worth 68x here and 1.26x on retrieved menus,
where §4 finds SJF cutting the median 85%. The resolution is in §4 — most of that
median gain is a deep-queue artifact any non-arrival order reproduces, and the
tail rises 127%. So ordering dominates dispatch where menus overlap; where they
do not, neither lever is worth much and the one that moves the median pays for it
in the tail.

**It cannot act on three of six arms at all.** Usability is an inverted U and
those three sit at its ends: `tooltrie_v0` and ContextPilot because every
candidate scores 56–63 and ties, `tooltrie_v1` because on padded menus every
candidate scores near 0 and ties. It works only on mid-quality orderings.
([§2](findings.md#2-the-mean-versus-max-adaptor))

---

## 3. How the trie reduces parallel latency

**One mechanism: it raises reuse, which shortens prefill, which compounds under
load into admission capacity.** Sustained throughput at 64 req/s offered:

| arm | reuse (converged) | ceiling, first 200 | ceiling, converged |
|---|---|---|---|
| `original` | 0.69% | 3.57 req/s | 3.49 |
| `tooltrie_v1` | 1.19% | 3.53 | — |
| `alphabetical` | 46.08% | 4.74 | 5.18 |
| `tooltrie_v0` | 96.81% | 11.24 | 16.79 |
| ContextPilot | 96.85% | **16.83** | 16.16 |

Ordering alone is worth **4.7x the admission capacity**, 4.8x once converged. The
converged column's top two rows are a tie within a 0.96 within-arm spread.

**On retrieved menus the same mechanism is worth 4.6% instead of 380%.**
Capacity ceiling at k64, 4 req/s offered, same six arms — every one saturates
below the offered rate, so these are real ceilings:

| arm | reuse | ceiling |
|---|---|---|
| `frequency` | 0.94% | 2.623 req/s |
| `alphabetical` | 1.22% | 2.625 |
| `tooltrie_v0` | 1.90% | 2.629 |
| `original` | 0.91% | 2.635 |
| ContextPilot | 4.78% | 2.693 |
| **`tooltrie_v1`** | **4.96%** | **2.736** |

**1.046x spread against padded's 4.8x.** The two tables are at different offered
rates (64 vs 4 req/s), so the absolute ceilings are not comparable — only the
spread within each, which is what this claim rests on. The mechanism survives in
direction —
the two arms above 4.7% reuse are the two fastest — but the four below 2% sit
inside 0.4% of each other, which is within run-to-run noise. Ordering still buys
admission capacity on real retrieved menus; it buys 4.6% of it.

The parallel-specific part is that **prefix caching is sequential** — a request
can only reuse what an earlier one already stored, so requests in flight together
all miss the same cold prefix. Measured as how many are already dispatched when
the first completes, on converged records at 64 req/s: `original` and
`alphabetical` **400/400**, `tooltrie_v0` 218/400, ContextPilot 216/400.
**Pile-up tracks reuse, not policy.**
([§3](findings.md#3-how-the-trie-reduces-parallel-latency))

---

## 4. Cutting tail latency by smart queuing

**No. It cuts the median and raises the tail, and no setting improves both.**
k64 at 4 req/s, arrival to first token, against FIFO:

| aging | mean | p50 | p95 | max |
|---|---|---|---|---|
| 0 | −32.0% | −85.5% | **+54.9%** | **+127.3%** |
| **250** | **−24.6%** | **−47.3%** | **+17.6%** | **+45.6%** |
| 1000 | −7.9% | −6.4% | −3.4% | +17.8% |
| 2000 | −3.2% | −4.1% | +0.3% | +7.3% |

Aging 250 keeps most of the median gain for a contained tail; by 2000 the policy
has become FIFO again.

**A control was essential and changed the number.** Under a deep queue *any*
order unrelated to arrival cuts the median ~27% — reproduced three times,
including on padded menus where every request is the same size and the sort key
carries no information. Subtracting it, the real size-aware gain is 29.2 points
against 3.7. Without the control the headline would have been "SJF cuts median
latency 6.9x", crediting the policy for what a coin flip reproduces.

**And the verdict does not depend on the ordering underneath it** — measured on
all five. p50 falls 84–86% and `max` rises 121–130% whether SJF dispatches
`tooltrie_v0`, `alphabetical`, `frequency`, ContextPilot or `tooltrie_v1`,
because SJF sorts by job size and job size is invariant under any permutation of
a menu. ([§4.2](findings.md#42-smart-queuing-trades-the-tail-for-the-median))

---

## What ordering costs

Reuse is bought with accuracy, and the mechanism is **position**:

| depth | arm | gold position | reuse | accuracy 0.6B | accuracy 4B |
|---|---|---|---|---|---|
| k64 | `original` | 6.1 | 0.91% | **37.09%** | **44.37%** |
| k64 | ContextPilot | 12.5 | 4.78% | 27.15% | 42.38% |
| k64 | `tooltrie_v1` | 18.0 | 4.96% | 33.11% | 42.38% |
| k64 | `tooltrie_v0` | 31.5 | 1.90% | 25.83% | 39.74% |
| k128 | `original` | 11.5 | 0.37% | **22.98%** | **44.72%** |
| k128 | ContextPilot | 27.8 | 1.99% | 22.98% | 37.27% |
| k128 | `tooltrie_v1` | 46.6 | 2.21% | 22.36% | **44.72%** |
| k128 | `tooltrie_v0` | 62.8 | 1.13% | 16.15% | 32.30% |

The cache wants the *common* tools first; the model wants the *relevant* tools
first; prefix caching only reuses a leading prefix. Both compete for the front.

**A larger model is about half as depth-sensitive** — the slope is 0.191 pp lost
per position at 4B, at both depths, against 0.359 at 0.6B/k64. And **ContextPilot's
"free lunch" at k128 was a floor effect**: 0.00pp at 0.6B, **−7.45pp** at 4B.
`tooltrie_v1` is the one arm that gains reuse without paying — at 4B/k128 it
matches the unordered baseline exactly while carrying 6x its reuse.
([§1.4](findings.md#14-what-the-reordering-costs))


---

## Two things that changed how the numbers read

**ContextPilot's padded lead is warm-up.** Over 600 requests the 8.97pp reuse gap
becomes 0.04pp and the 22 ms median penalty falls to 0.5 ms. Both converge to
placing the odd tool last, so the tie is forced. ContextPilot's real advantage
there is **cold-start speed** — optimal from request 2 against request 222.

**Arrival order matters more than policy choice.** The benchmark's natural order
is blocked by source (101 `apibank` then 99 `apigen`), so adjacent requests
overlap 3.40 tools against 1.43 when shuffled. That locality alone moved
ContextPilot from 2.15% to 4.78% at k64 — larger than the gap between any two
policies.

---
