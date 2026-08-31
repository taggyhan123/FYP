# Tool ordering under concurrent load — key findings

Qwen3-0.6B on one RTX 3090 (plus a Qwen3-4B check), vLLM 0.26.0, prefix caching
unmodified. 195 GPU runs.

This is the short version. Every number links to the section of
[`findings.md`](findings.md) that derives it, with its runs and controls.

| # | Question | Answer |
|---|---|---|
| 1 | Tail latency under parallel load | Ordering matters far more than it did serially — but mostly on padded menus |
| 2 | An adaptor trading mean against max | Works; not worth deploying. Ordering is worth 6.7x more |
| 3 | How the trie reduces parallel latency | By raising cache reuse, which compounds under load into capacity |
| 4 | Cutting tail latency by smart queuing | No. It cuts the median and *raises* the tail |

---

## The five findings worth knowing

### 1. Parallel load separates orderings that serial testing could not

Serially, every arm sat within 1.25x on every statistic. Under load they span
36x at p50. Reuse turns directly into admission capacity — sustained throughput
at 64 req/s offered, for the first 200 requests and for the 400 converged
records replayed as their own workload:

| arm | reuse 1–200 | ceiling, first 200 | reuse converged | ceiling converged |
|---|---|---|---|---|
| `original` | 1.19% | **3.57 req/s** | 0.69% | **3.49** |
| `alphabetical` | 38.13% | 4.74 | 46.08% | 5.18 |
| `tooltrie_v0` | 87.19% | 11.24 | 96.81% | **16.79** |
| **ContextPilot** | 96.16% | **16.83** | 96.85% | 16.16 |

Ordering alone is worth **4.7x the admission capacity**, and **4.8x** once
converged. The ToolTrie–ContextPilot capacity gap is warm-up like the rest of the
padded story: **1.50x** over the first 200 requests, and a tie afterwards. ([§1.2](findings.md#12-three-things-serial-testing-could-not-show))

### 2. Most of that is the test workload, not the method

Padded menus give every request 63 of the same 64 tools. On genuinely retrieved
menus the effect nearly vanishes — reuse spread 94.97pp → 0.98pp, p50 spread
36x → 1.02x, and 95–98% of prefill is uncacheable whatever policy is used.

**This is the report's most important limit.** ([§4.1](findings.md#41-on-real-retrieved-menus-ordering-barely-matters))

### 3. Reordering costs accuracy, and the reason is position

| depth | arm | reuse | accuracy | mean position of correct tool |
|---|---|---|---|---|
| k64 | `original` | 0.91% | **37.09%** | **6.1** |
| k64 | ContextPilot | 4.78% | 27.15% | 12.5 |
| k64 | `tooltrie_v0` | 1.90% | 25.83% | 31.5 |
| k128 | `original` | 0.37% | **22.98%** | **11.5** |
| k128 | ContextPilot | 1.99% | **22.98%** | 27.8 |
| k128 | `tooltrie_v0` | 1.13% | 16.15% | 62.8 |

Accuracy tracks how far down the menu the correct tool is pushed. The cache wants
the *common* tools first; the model wants the *relevant* tools first; prefix
caching only reuses a leading prefix. Both compete for the front of the prompt.
ContextPilot at k128 is the one policy that gains reuse at zero accuracy cost.
([§1.4](findings.md#14-what-the-reordering-costs))

### 4. ContextPilot's padded lead is warm-up, not ordering quality

Two windows of the same 600-request run:

| arm | adapts | reuse 1–200 | reuse 201–600 | p50 1–200 | p50 201–600 |
|---|---|---|---|---|---|
| `original` | no | 1.19% | 0.70% | 3622.8 | 6239.7 |
| `alphabetical` | no | 38.13% | 46.29% | 335.2 | 271.3 |
| `tooltrie_v0` | **yes** | 87.19% | **97.05%** | 116.0 | **91.2** |
| **ContextPilot** | **yes** | **96.16%** | **97.09%** | **93.9** | 91.7 |

An 8.97pp gap becomes 0.04pp and ToolTrie's 22 ms median penalty falls to 0.5 ms.
Read that as a tie, not a reversal: at p95/p99/max the remaining gaps are smaller
than ToolTrie's own run-to-run spread. Both planners converge to placing the odd
tool last, so the tie is forced by construction. ContextPilot's real advantage on
padded menus is **cold-start speed**. Once ToolTrie converges the two are
indistinguishable at every load tested — but the tie is a ceiling effect, since
both then place the odd tool optimally and the workload cannot separate them.

Ordering-vs-no-ordering meanwhile *strengthens*: `original` has nothing to learn
and its backlog keeps growing, so its p50 rises while every ordered arm's falls —
the gap widens from 31x to **68x**. ([§1.5](findings.md#15-why-the-two-windows-differ-warm-up))

### 5. Smart queuing buys the median by selling the tail

k64 at 4 req/s, arrival to first token. No point on the fairness frontier
improves both ends:

| aging | mean | p50 | p95 | max |
|---|---|---|---|---|
| 0 | −32.0% | −85.5% | **+54.9%** | **+127.3%** |
| **250** | **−24.6%** | **−47.3%** | **+17.6%** | **+45.6%** |
| 1000 | −7.9% | −6.4% | −3.4% | +17.8% |
| 2000 | −3.2% | −4.1% | +0.3% | +7.3% |

A `random` control was essential: under a deep queue *any* non-arrival order cuts
the median ~27%, reproduced three times. Without subtracting it the headline
would have been "SJF cuts median latency 6.9x". ([§4.2](findings.md#42-smart-queuing-trades-the-tail-for-the-median))

---

## Two things that changed how the numbers should be read

**Arrival order matters more than policy choice.** The benchmark's natural order
is blocked by source (101 `apibank` then 99 `apigen`), so adjacent requests
overlap 3.40 tools against 1.43 when shuffled. That locality alone moved
ContextPilot from 2.15% to 4.78% reuse at k64 — a larger effect than the gap
between any two policies. The k64 margin is 2.52x on the natural order and 1.54x
averaged over five permutations. k128 is stable either way. ([A.6](findings.md#a6-is-the-tooltrie-vs-contextpilot-comparison-fair))

**The comparison is tilted against this project's own method, not for it.**
ToolTrie is `v0` against a tuned published system, ran under a planner memory
budget ContextPilot did not have, and still ContextPilot wins on retrieved menus.
The budget turned out to be immaterial — it models the KV cache almost exactly
(190,896 tokens against 189,728 actual), so lifting it buys hints the cache
cannot honour. ([A.6](findings.md#a6-is-the-tooltrie-vs-contextpilot-comparison-fair))

---

## What is not settled

- **The interesting middle is unmeasured.** Padded menus share 98.4% of tools;
  retrieved ones share 0% and overlap 21–33% with their best partner. Nothing
  exists between. A workload at 30–70% overlap is where a trie could earn its
  keep rather than tie at a ceiling or lose in the noise.
- **No arm used ContextPilot's order annotations**, which exist to decouple
  relevance from position and are the one thing that could reopen the rejected
  methods in [§5](findings.md#5-explored-and-rejected).
- **One model, one trial per cell.** Eight cells were re-run under 3–5 arrival
  permutations; every other cell is a single draw.
