# Tool ordering under concurrent load — key findings

Qwen3-0.6B on one RTX 3090 (plus a Qwen3-4B check), vLLM 0.26.0, prefix caching
unmodified. 268 GPU runs.

This is the short version. Every number links to the section of
[`findings.md`](findings.md) that derives it, with its runs and controls.

**How the method works:** [`ToolTrie-v0`](../../notes/tooltrie-v0-design.md) ·
[`ToolTrie-v1`](../../notes/tooltrie-v1-design.md) — v1 changes exactly one line
of v0, the order given to tools the trie could not match.

| | v0 | **v1** |
|---|---|---|
| trie, walk, selection, budget, causality | — | identical |
| tools the trie **matched** | placed at the front | placed at the front |
| tools it **could not** match | sorted **alphabetically** | **left in arrival order** |
| menu displaced | 99.1% | **13%** |
| mean tool movement | 41.9 positions | **1.1** |
| requests left untouched | 0 / 200 | **120 / 200** |

The fallback decides ~93% of every ordering, because the trie matches only 1.19
of 64 tools at k64. Tool name correlates with neither relevance nor commonality,
so v0 permuted almost the whole menu on no information — displacing the correct
tool to position 62.8 of 128 and returning 1.13% reuse for it. v1 keeps what the
retriever produced and reorders only what the trie has evidence for.

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
1.26x and 95–98% of prefill is uncacheable whatever policy is used. And the
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
| `tooltrie_v0` | **122.8** | **915.6** | 0/200 |
| **ContextPilot** | **94.2** | **452.6** | 0/200 |
| `tooltrie_v1` | 5052.9 | 10159.5 | 0/200 |

The shape is right — median down, worst case up, about half the requests moved.
**The problem is that the best the adaptor achieves anywhere is 685 ms, against
123 for ToolTrie and 94 for ContextPilot, neither of which uses it.** Fixing the
ordering is worth 5.6–7.3x more than fixing the dispatch order.

It also **cannot act on three of the six arms at all**: 0 of 200 reordered on
`tooltrie_v0`, ContextPilot and `tooltrie_v1`, because the adaptor needs
*variance* in prefix affinity. Usability is an inverted U and those three sit at
its two ends — v0 and ContextPilot because every candidate scores 56–63 and ties,
v1 because on padded menus every candidate scores near 0 and ties. It works only
on mid-quality orderings.
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

**And the verdict does not depend on the ordering underneath it** — measured on
all five. p50 falls 84–86% and `max` rises 121–130% whether SJF dispatches
`tooltrie_v0`, `alphabetical`, `frequency`, ContextPilot or `tooltrie_v1`,
because SJF sorts by job size and job size is invariant under any permutation of
a menu. ([§4.2](findings.md#42-smart-queuing-trades-the-tail-for-the-median))

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

**ToolTrie-v0 beats every simple heuristic — unordered, alphabetical, frequency —
everywhere, and never beats ContextPilot on a primary metric anywhere.
ToolTrie-v1 does, on both.**

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

### ToolTrie-v1 changes this

v0 sorts the tools its trie cannot match **alphabetically**, and that fallback
orders ~93% of every menu. Tool name correlates with neither relevance nor
commonality, so the permutation is information-free: at k128 it displaces 99.1%
of the menu by a mean of 41.9 positions for 1.13% reuse — **37 positions per
point of reuse against ContextPilot's 8**.

v1 keeps unmatched tools in the order they arrived. At k128 it places **2.55
tools of 128** and leaves **120 of 200 requests byte-identical** to their input.

| depth | original | tooltrie_v0 | ContextPilot | **tooltrie_v1** |
|---|---|---|---|---|
| k4 | 15.87% | 17.48% | 18.72% | **19.47%** |
| k16 | 6.12% | 7.77% | 9.93% | **11.31%** |
| k64 | 0.91% | 1.90% | 4.78% | **4.96%** |
| k128 | 0.37% | 1.13% | 1.99% | **2.21%** |

**4 of 4 retrieved depths, and 5 of 5 arrival permutations at k64** (p = 0.031;
reuse is deterministic, so each paired comparison is exact). **But read the
margins by comparator** — against ContextPilot it is 1.04–1.14x, which at k64 is
+0.18pp of reuse where 95% of prefill is uncacheable either way; against
`tooltrie_v0` it is 1.11–2.61x, and against no reordering up to 5.97x. The
finding that matters is the v0 column: one line of fallback ordering was costing
ToolTrie most of its own reuse. The ContextPilot comparison establishes that v1
has *caught up* with the published baseline, not that it has passed it
meaningfully. Accuracy against
ContextPilot is a **wash** — across all six cells measured it is 2 wins, 2 losses
and 2 ties, every margin under 1.6 SE — so the claim is *more reuse at no
measurable accuracy cost*, not better accuracy. At 4B/k128 it matches the
*unordered baseline exactly* (44.72%) while carrying 6x its reuse. It is also
faster than ContextPilot on 21 of 24 metric-cells across the four retrieved
depths — every cell at k64 and k128 — which widens the retrieved spread quoted in
question 1 from 1.14x to **1.26x** at k64. The three it loses (`max` at k4, `p95`
at k16, achieved rate at k4) are millisecond gaps at 46–210 ms.

**Audited for fairness, not assumed.** Causal (trie updated only after planning);
menu membership unchanged 200/200; matched prefix at the front 200/200; tail
exactly the input order 200/200; same rates, seed, cap and decode settings as the
arms it is compared against. On whether it uses privileged information: measuring
how much of the retriever's pairwise ordering each arm keeps (1.00 = identical,
0.50 = uncorrelated) gives alphabetical 0.510, tooltrie_v0 0.511, **ContextPilot
0.860**, tooltrie_v1 0.991. ContextPilot already preserves the input ordering —
v1 is the same family of design, more conservatively applied. v0 is the outlier
that throws it away.

**Four caveats the mechanical checks miss.** ContextPilot is measured far outside
its own evaluation regime: its paper uses a dense retriever on 3 of 4 datasets at
k=3–20 (primarily 15), over multi-turn workloads with **~40% overlap between a
turn and earlier ones**. Here it sees BM25 only, k=4–128, independent requests,
and **1.6–7.7% adjacent overlap**. v1's wins are on those low-overlap menus where
ContextPilot is furthest off-design; its losses are on the §4.3 menus, closest to
ContextPilot's own regime. ContextPilot also runs *ordering only*
here, without the annotations, de-duplication and scheduling its paper credits
for roughly half its cache gain — so this beats its ordering component, not the
system. v1 was **designed while looking at k64 and k128**, with no held-out set;
k4, k16, four of five arrival seeds and both 4B accuracy cells came after the
rule was fixed and it leads on all of them, but it was not preregistered. And the
case for discounting padded menus was assembled *after* v1 failed on them, though
its evidence is only partly objective: the brief never uses the word, but §4.2
does designate BFCL for "constructing controlled tool-menu workloads", which is
what padded-64 is. It is a sanctioned control, not an unsanctioned workload.

**It loses on `padded-64`, badly and for a knowable reason.** That workload's
`original` arm puts the one differing tool at position 0 of every request; v1
never overrides its input, so it preserves the worst possible layout and scores
1.19% against v0's 87.19%. That is the construction, not the overlap level — at
25/50/75% shared tools with a *shuffled* input order v1 reaches 23.03 / 42.55 /
70.62% reuse against ContextPilot's 28.80 / 49.87 / 73.98% — it loses there too,
while still beating v0 by ~3x. So v1's advantage is a joint property of the
method and the retriever:

| input ordering | vs ContextPilot | vs tooltrie_v0 |
|---|---|---|
| BM25 relevance | **wins 4/4** | wins 4/4 |
| shuffled | loses, 0.80–0.95x | wins, ~3x |
| adversarial (padded-64) | loses, 0.01x | loses, 0.01x |

ContextPilot overrides its input when its clustering disagrees; v1 essentially
never does. Use v1 where the retriever's ranking is meaningful — the deployed
case — and not as a general-purpose ordering.
([§5](findings.md#5-tooltrie-v1-reorder-only-what-the-trie-matched))

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
- **The hybrid ordering rejected in [§6](findings.md#6-explored-and-rejected) is
  reopened, and probably not settleable on this benchmark.** Its accuracy penalty
  against ContextPilot is 9.32pp at 0.6B (2.2 SE) but 1.25pp at 4B (0.2 SE),
  while it carries 56% more reuse. Decoding is greedy (`temperature 0, seed 0`),
  so re-running reproduces the same answers — more samples means more *tasks*,
  and confirming a 1.25pp effect at 2 SE needs roughly **75x the evaluation data,
  about 15,000 tasks against the 200 these workloads carry**. What is resolvable
  is that the 0.6B penalty was real and the 4B one is not detectable; that is the
  finding, and a bigger benchmark is the only way past it.
- **No arm used ContextPilot's order annotations**, which exist to decouple
  relevance from position and would help the displaced orderings most.
- **ToolTrie-v1 has no accuracy baseline at k4/k16**, and no latency-under-load
  numbers. Its reuse and accuracy at k64/k128 are measured; the rest is not.
- **`canonical` and `hybrid` were measured one line short.** Their tie-break falls
  through to each request's own position index, so identical-frequency tools come
  out in a different order per request. Fixing it lifts the k128 prefix 24%. Their
  §6 numbers understate them.
- **Coverage is uneven.** Accuracy has two models at both depths; converged
  capacity has three replicates per arm; eight reuse cells were re-run under 3–5
  arrival permutations. Everything else is a single draw at one model.
