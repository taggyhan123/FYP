# Tool ordering under concurrent load — key findings

Qwen3-0.6B on one RTX 3090 (plus a Qwen3-4B check), vLLM 0.26.0, prefix caching
unmodified. 270 GPU runs.

The short version. Every number links to the section of
[`findings.md`](findings.md) that derives it, with its runs and controls.

| # | Question | Answer |
|---|---|---|
| 1 | Latency distribution at p95 / p99 / max under parallel load | Ordering is worth 68x at p50 and 113–124x across the tail — but mostly on padded menus |
| 2 | An adaptor trading mean against max | Works, not worth deploying. Ordering is worth 5.6–7.3x more, and it cannot act on the good orderings at all |
| 3 | How the trie reduces parallel latency | One mechanism: it raises reuse, which compounds under load into admission capacity |
| 4 | Cutting tail latency by smart queuing | **No.** It cuts the median ~85% and raises the tail ~125%, on every ordering |

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
| BM25 relevance (k4–k128) | **wins 4/4** | wins 4/4 |
| shuffled (§4.3, 25–75% shared) | loses, 0.80–0.95x | wins, ~3x |
| adversarial (`padded-64`) | loses, 0.01x | loses, 0.01x |

ContextPilot overrides its input when its clustering disagrees; v1 essentially
never does. So v1 turns a good input ordering into more reuse than ContextPilot
can, and cannot rescue a bad one. **Questions 1–3 below use padded menus — the
one workload where v1 is the wrong choice** — so it appears there at `original`'s
level, and that is the documented boundary rather than its general performance.

---

## 1. Latency distribution at p95 / p99 / max

Padded menus at 4 req/s, time to first token in ms, over **converged** requests
(201–600) so no arm is still learning:

| arm | reuse | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| `original` | 0.70% | 6239.7 | 15175.0 | 15659.5 | 16108.6 |
| `tooltrie_v1` | 0.70% | 4389.5 | 10493.9 | 11045.5 | 11564.3 |
| `alphabetical` | 46.29% | 271.3 | 586.6 | 820.6 | 960.4 |
| `tooltrie_v0` | 97.05% | **91.2** | **122.5** | **133.1** | **141.9** |
| ContextPilot | 97.09% | 91.7 | 123.7 | 134.3 | 151.1 |

`frequency` cannot join: it is a *fitted* baseline whose training corpus is not
recorded recoverably, so extending it to 600 requests would mean guessing. Over
the first 200 it scores 39.69% reuse and 317.9 ms p50 — within 1.6pp and 14 ms of
`alphabetical`, which it tracks throughout.

**Serial testing hid all of this.** At concurrency 1 every arm sat within 1.25x
on every statistic and `max` was flat at ~270 ms. Under load `max` is dominated
by queueing, which one-at-a-time testing cannot produce.

**The real answer is that a badly ordered arm has no fixed distribution.**
`original` sustains only 3.51 of an offered 4 req/s, so its backlog grows and its
latency grows with run length — p50 3310 ms over 200 requests, 6240 over 600. The
ordered arms are stable. So the ratio depends on how long you run (36x at 200,
68x at 600), and that divergence is the finding, not any single number.

The bottom two rows are a **tie** — every gap is inside measurement noise, and
from request 222 both arms place the odd tool identically. Over the first 200
requests ContextPilot leads v0 by 8.97pp of reuse and 22 ms of p50; that gap is
**warm-up only** and is gone by the window above.
([§1.1](findings.md#11-results) · [§1.5](findings.md#15-why-the-two-windows-differ-warm-up))

---

## 2. An adaptor trading mean against max

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
5.6–7.3x more than fixing the dispatch order.

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

## Four caveats on the ContextPilot comparison

1. **It is measured far outside its own evaluation regime.** Its paper uses a
   dense retriever on 3 of 4 datasets at k=3–20 (primarily 15), over multi-turn
   workloads with **~40% overlap** between a turn and earlier ones. Here it sees
   BM25 only, k=4–128, independent requests, **1.6–7.7% adjacent overlap**. v1's
   wins are where ContextPilot is furthest off-design; its losses are on the §4.3
   menus, closest to ContextPilot's own regime.
2. **It runs ordering-only** — no annotations, de-duplication or scheduling,
   which its paper credits for roughly half its cache gain. This beats its
   *ordering component*, not the system.
3. **v1 was designed while looking at k64 and k128**, with no held-out set. k4,
   k16, four of five arrival seeds and both 4B accuracy cells came after the rule
   was fixed and it leads on all of them — but it was not preregistered.
4. **The case for discounting padded menus was assembled after v1 failed on
   them.** The brief does sanction padded as a *control* (§4.2 designates BFCL
   for "constructing controlled tool-menu workloads"), so this is v1 measured
   outside its regime, not an unfair test.
   ([A.6](findings.md#a6-is-the-tooltrie-vs-contextpilot-comparison-fair))

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

## What is not settled

- **No arm used ContextPilot's order annotations**, which exist to decouple
  relevance from position. Three methods have now died on the accuracy gate, and
  this is the only untried mechanism that could move it.
- **The hybrid ordering rejected in [§6](findings.md#6-explored-and-rejected) is
  reopened, and probably not settleable here.** Its accuracy penalty against
  ContextPilot is 9.32pp at 0.6B (2.2 SE) but 1.25pp at 4B (0.2 SE). Decoding is
  greedy, so re-running reproduces the same answers — confirming a 1.25pp effect
  at 2 SE needs ~75x the evaluation data, about 15,000 tasks against 200.
- **No representative chronological trace.** The brief's own §4.5 warns that no
  listed dataset provides one, and arrival locality turned out to move reuse more
  than any policy does. Every number here inherits that.
- **Coverage is uneven.** Accuracy has two models at k64/k128; converged capacity
  has three replicates per arm; eight reuse cells have 3–5 arrival permutations.
  Everything else is a single draw at one model.
