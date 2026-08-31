# Tool ordering under concurrent load — key findings

Qwen3-0.6B on one RTX 3090 (plus a Qwen3-4B check), vLLM 0.26.0, prefix caching
unmodified. 233 GPU runs.

This is the short version. Every number links to the section of
[`findings.md`](findings.md) that derives it, with its runs and controls.

---

## 1. Parallel requests → latency distribution at p95 / p99 / max

Padded menus at 4 req/s, time to first token in ms, on **converged** requests
(201–600) so no arm is still learning:

| arm | p50 | p95 | p99 | max |
|---|---|---|---|---|
| `original` | 6239.7 | 15175.0 | 15659.5 | 16108.6 |
| `alphabetical` | 271.3 | 586.6 | 820.6 | 960.4 |
| `tooltrie_v0` | 91.2 | 122.5 | 133.1 | 141.9 |
| ContextPilot | 91.7 | 123.7 | 134.3 | 151.1 |

The bottom two rows are a **tie**, not a ToolTrie win — every gap there is inside
measurement noise, and from request 222 both arms place the odd tool identically.
See the ToolTrie-vs-ContextPilot section below.

**Ordering is worth 68x at p50 and 113–124x across the tail.** Serial testing hid
all of it — at concurrency 1 every arm sat within 1.25x on every statistic and
`max` was flat at ~270 ms. Under load `max` is dominated by queueing, which
one-at-a-time testing cannot produce.

**The real answer is that a badly ordered arm has no fixed distribution.**
`original` cannot keep up at 4 req/s (it sustains 3.51), so its backlog grows and
its latency grows with run length: p50 3310 ms over 200 requests, 6240 over 600.
The ordered arms are stable and converge to a fixed distribution. So the ratio
depends on how long you run — 36x at 200 requests, 68x at 600 — and that
divergence, not any single number, is the finding.

For reference, the same table over the first 200 requests, which is the window
every other section uses and the only one with a `frequency` arm:

| arm | reuse | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| `original` | 1.19% | 3310.4 | 6498.6 | 7205.0 | 7476.0 |
| `alphabetical` | 38.13% | 331.8 | 1058.1 | 1406.8 | 1724.9 |
| `frequency` | 39.69% | 317.9 | 1028.9 | 1286.2 | 1521.4 |
| `tooltrie_v0` | 87.19% | 116.3 | 225.7 | 382.7 | 558.4 |
| **ContextPilot** | **96.16%** | **92.7** | **129.0** | **260.9** | **300.2** |

**Two qualifications, both important.** This is a padded workload where 63 of 64
tools are shared; on genuinely retrieved menus the spread collapses to at most
1.14x and 95–98% of prefill is uncacheable whatever policy is used. And the
ToolTrie-vs-ContextPilot difference in the second table is **warm-up only** — it
vanishes in the first, and in the converged table above.
([§1.1](findings.md#11-results), [§4.1](findings.md#41-on-real-retrieved-menus-ordering-barely-matters))

---

## 2. An adaptor trading mean against max

It works, and `D` — how long a request may be held back — is the knob. But it is
not worth deploying. Arrival to first token, ms:

| policy | p50 | max | reordered |
|---|---|---|---|
| `frequency` | 825.2 | 2640.3 | — |
| `frequency` + adaptor | **685.2** (−17.0%) | 2961.4 (+12.2%) | 105/200 |
| `alphabetical` | 1046.0 | 3120.2 | — |
| `alphabetical` + adaptor | **827.2** (−20.9%) | 3302.6 (+5.8%) | 104/200 |
| `tooltrie_v0` | **122.8** | **915.6** | — |
| **ContextPilot** | **94.2** | **452.6** | — |

The shape is right — median down, worst case up, about half the requests moved.
**The problem is that the best the adaptor achieves anywhere is 685 ms, against
123 for ToolTrie and 94 for ContextPilot, neither of which uses it.** Fixing the
ordering is worth 5.6–7.3x more than fixing the dispatch order.

It also **cannot act on the good orderings at all**: 0 of 200 reordered on both
ToolTrie and ContextPilot, because the adaptor needs *variance* in prefix
affinity and those arms have made every request look alike (spread 0.08 and
0.01). Usability is an inverted U — it only works on mid-quality orderings.
([§2](findings.md#2-the-mean-versus-max-adaptor))

---

## 3. How the trie reduces parallel latency

**Through exactly one mechanism: it raises cache reuse, which shortens prefill,
which compounds under load into admission capacity.** Sustained throughput at
64 req/s offered:

| arm | ceiling, first 200 | ceiling, converged |
|---|---|---|
| `original` | 3.57 req/s | 3.49 |
| `alphabetical` | 4.74 | 5.18 |
| `tooltrie_v0` | 11.24 | 16.79 |
| ContextPilot | **16.83** | 16.16 |

Ordering alone is worth **4.7x the admission capacity**, and 4.8x once converged.
The converged column's top two rows are a tie (within a 0.96 within-arm spread);
ContextPilot's first-200 lead is warm-up.

The parallel-specific part is that **prefix caching is sequential**: a request
can only reuse what an earlier one already stored, so requests in flight together
all miss the same cold prefix. Measured as how many requests are already
dispatched when the first one completes, on converged records at 64 req/s:

| arm | reuse | pile-up |
|---|---|---|
| `original` | 0.69% | **400 / 400** |
| `alphabetical` | 46.08% | **400 / 400** |
| `tooltrie_v0` | 96.81% | 218 / 400 |
| ContextPilot | 96.85% | 216 / 400 |

**Pile-up tracks reuse, not policy.** Below ~50% reuse every request in the run
starts cold; at ~97% barely half do, and the two high-reuse arms are
indistinguishable. An earlier version of this reported ToolTrie's pile-up as
twice ContextPilot's — that compared a still-learning ToolTrie against a
converged ContextPilot.
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
order unrelated to arrival cuts the median about 27% — reproduced three times,
including on padded menus where every request is the same size and the sort key
carries no information. Subtracting that, the real size-aware gain is 29.2 points
against 3.7. Without the control the headline would have been "SJF cuts median
latency 6.9x", crediting the policy for something a coin flip reproduces.
([§4.2](findings.md#42-smart-queuing-trades-the-tail-for-the-median))

---

## What ordering costs, across all four answers

Reuse is bought with accuracy, and the mechanism is position:

| depth | arm | gold position | reuse | 0.6B | 4B |
|---|---|---|---|---|---|
| k64 | `original` | 6.1 | 0.91% | **37.09%** | **44.37%** |
| k64 | ContextPilot | 12.5 | 4.78% | 27.15% | 42.38% |
| k64 | `tooltrie_v0` | 31.5 | 1.90% | 25.83% | 39.74% |
| k128 | `original` | 11.5 | 0.37% | **22.98%** | **44.72%** |
| k128 | ContextPilot | 27.8 | 1.99% | **22.98%** | 37.27% |
| k128 | `tooltrie_v0` | 62.8 | 1.13% | 16.15% | 32.30% |

The cache wants the *common* tools first; the model wants the *relevant* tools
first; prefix caching only reuses a leading prefix. Both compete for the front of
the prompt.

**A larger model is about half as depth-sensitive.** Fitting accuracy against
gold position across all arms in each cell gives a slope of **0.191 pp lost per
position at 4B, at both depths** — against 0.359 at 0.6B/k64. The 0.6B k128 cell
matches 4B only because it is floor-limited: a 22.98% baseline leaves little room
to fall, which compresses the slope.

**ContextPilot's "free lunch" at k128 was that same floor effect.** It cost
0.00pp at 0.6B and **−7.45pp at 4B**. It is still the cheapest reordering at both
depths — 1.99pp at k64 — but it is not free.
([§1.4](findings.md#14-what-the-reordering-costs))

---

## Where ToolTrie stands against ContextPilot

**Where it ties.** On padded menus after it has converged, every timing metric is
inside measurement noise — three replicates per arm, byte-identical orderings:

| | ToolTrie | ContextPilot | within-arm spread | |
|---|---|---|---|---|
| throughput (req/s) | 16.14 | 15.91 | 0.96 | tie |
| p50 (ms) | 8794 | 8960 | 632 | tie |
| p95 (ms) | 14827 | 15158 | 1417 | tie |
| reuse | 97.05% | 97.09% | **0.00** | ContextPilot, by 0.04pp |

Reuse is the only resolvable difference, because reuse is exactly reproducible
and timing is not ([A.7](findings.md#a7-what-counts-as-a-difference)).

**Where it wins.** One thing, secondary: planning cost on retrieved menus,
**0.20 ms/request against 1.94 ms** — about 10x cheaper. It did not matter here
(1.94 ms against a 9-second p50 is 0.02%). It also bounds its own memory where
ContextPilot's index grows without limit, also measured immaterial at this scale.

**Where it loses.**

| regime | margin |
|---|---|
| padded, before convergence | optimal at request **2** vs **222** |
| retrieved k4 / k16 | 1.07x / 1.28x reuse |
| retrieved k64 / k128 | **1.54x** / **1.80x** (5- and 3-seed means) |
| accuracy @ k128 | −7.45pp cost vs ToolTrie's −12.42pp (4B) |
| accuracy @ k64 | −1.99pp vs −4.63pp (4B) |

**ToolTrie beats every simple heuristic — unordered, alphabetical, frequency —
everywhere. It never beats ContextPilot on a primary metric anywhere.**

The one tie needs reading carefully, because two things make it uninformative
rather than encouraging:

- **The measurement has a ceiling.** On padded menus reuse is set by where the
  one odd tool sits, and position 63 is the maximum. From request 222 onward both
  arms place it at 63 on *every* request — the orderings are identical in the only
  respect that affects reuse, so the same score is forced. Two students both
  scoring 100% tells you the test was too easy, not that they are equally able.
- **The window is a favourable selection.** Requests 201–600 exclude 1–221, which
  is the whole of ToolTrie's disadvantage on padded menus. Even the residual
  0.04pp reuse gap comes entirely from requests 201–221, where ToolTrie is still
  below 63 in all 21 of them.

So the tie is where the workload stops measuring, not where the methods converge
in quality.

---

## Two things that changed how the numbers should be read

**ContextPilot's padded lead is warm-up.** Over a 600-request run the 8.97pp
reuse gap becomes 0.04pp and ToolTrie's 22 ms median penalty falls to 0.5 ms —
a tie, not a reversal. Both converge to placing the odd tool last, so the tie is
forced. ContextPilot's real advantage there is **cold-start speed**, and it is
large: optimal from request 2 against request 222.
([§1.5](findings.md#15-why-the-two-windows-differ-warm-up))

**Arrival order matters more than policy choice.** The benchmark's natural order
is blocked by source (101 `apibank` then 99 `apigen`), so adjacent requests
overlap 3.40 tools against 1.43 when shuffled. That locality alone moved
ContextPilot from 2.15% to 4.78% reuse at k64 — a larger effect than the gap
between any two policies. ([A.6](findings.md#a6-is-the-tooltrie-vs-contextpilot-comparison-fair))

---

## What is not settled

- ~~The interesting middle is unmeasured.~~ **Measured, and it went the other
  way.** At 25/50/75% shared tools ContextPilot reaches the ceiling every time
  (28.8 / 49.9 / 74.0%) and ToolTrie reaches about 30% of it. Its advantage is an
  inverted U in overlap — 1.5–1.8x on retrieved menus, **3.2–4.1x in the middle**,
  a tie on padded ones. This band was the trie's best prospect and is where it
  does worst ([§4.3](findings.md#43-the-middle-of-the-overlap-range)).
- **The hybrid ordering rejected in [§5](findings.md#5-explored-and-rejected) is
  reopened, and probably not settleable on this benchmark.** Its accuracy penalty
  against ContextPilot is 9.32pp at 0.6B (2.2 SE) but 1.25pp at 4B (0.2 SE),
  while it carries 56% more reuse. Decoding is greedy (`temperature 0, seed 0`),
  so re-running reproduces the same answers — more samples means more *tasks*,
  and confirming a 1.25pp effect at 2 SE needs roughly **75x the evaluation data,
  about 15,000 tasks against the 200 these workloads carry**. What is resolvable
  is that the 0.6B penalty was real and the 4B one is not detectable; that is the
  finding, and a bigger benchmark is the only way past it.
- **No arm used ContextPilot's order annotations**, which exist to decouple
  relevance from position and would help the displaced orderings most. This is now
  the only untried idea that could change the accuracy gate, and three separate
  methods have died on that gate.
- **`canonical` and `hybrid` were measured one line short.** Their tie-break falls
  through to each request's own position index, so identical-frequency tools come
  out in a different order per request. Fixing it lifts the k128 prefix 24%. Their
  §5 numbers understate them.
- **Coverage is uneven.** Accuracy has two models at both depths; converged
  capacity has three replicates per arm; eight reuse cells were re-run under 3–5
  arrival permutations. Everything else is a single draw at one model.
