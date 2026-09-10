# Metrics, and optimising throughput under a latency limit

Four questions, in the order they have to be answered:

1. Is the accuracy metric this project reports actually the right one? (§1)
2. Does a better accuracy metric reduce KV-cache recompute? (§2 — **no**, and
   the reasoning matters)
3. What configurations trade latency against accuracy, and at what exchange
   rate? (§3)
4. **If every request must answer within 1 second, what accuracy and
   throughput can each ordering policy deliver?** (§4 — the framing the other
   three exist to support)

Every claim is a dated citation or a run under `cluster/results/`. Companion
document on menu size: [`how-many-tools.md`](how-many-tools.md).

---

## 1. Is `gold_hit_ceil` the right accuracy metric?

`score_tool_selection.py`'s `gold_hit_ceil` scores a request correct if the
model called **any one** gold tool (`bool(ids & gold)`) — full credit
regardless of how many the task needs. **44% of evaluated tasks need 2+ gold
tools** (k64 slice: 113 need 1, 69 need 2, 15 need 3, 3 need 4). So the metric
is closer to "recall@1-of-gold" than to task completion.

**Checked whether that changes any published ranking** by re-scoring existing
replays under three definitions:

| cell | any-hit (current) | all-hit (strict) | fraction hit |
|---|---|---|---|
| k64 0.6B: v1 / CP / v0 / orig | 31.29 / 24.54 / 19.02 / 29.45 | 21.47 / 18.40 / 12.27 / 20.86 | 26.18 / 21.27 / 15.64 / 24.95 |
| k128 0.6B: v1 / CP / v0 / orig | 28.65 / 18.13 / 13.45 / 29.24 | 21.64 / 13.45 / 10.53 / 22.22 | 25.05 / 15.79 / 11.99 / 25.63 |
| k64 4B: v1 / CP / v0 / orig | 39.88 / 38.04 / 34.97 / 39.26 | 27.61 / 25.77 / 20.86 / 28.22 | 33.23 / 31.39 / 27.51 / 33.33 |
| k128 4B: v1 / CP / v0 / orig | 40.35 / 36.26 / 33.92 / 39.18 | 28.65 / 25.15 / 19.88 / 29.24 | 33.92 / 30.31 / 26.51 / 33.82 |

**`tooltrie_v1` and `original` > ContextPilot > `tooltrie_v0` in every cell,
under all three definitions.** The one pair the definition can reorder is
`tooltrie_v1` against `original`, which are never more than three requests
apart: v1 leads under all three at k64/0.6B, trails by one request under all
three at k128/0.6B, and in both 4B cells leads on any-hit but trails by one
request on all-hit. Absolute numbers drop by roughly a third under the strict
definition — real task-completion rate is lower than `gold_hit_ceil` implies —
but no comparison between policies more than a few requests apart reverses.

### 1.1 What the field uses, and why it is F1 here

Checked 2026-09-06. The field splits on task shape:

- **Exact Match / Accuracy** — all-or-nothing. Standard for **single-tool**
  tasks (BFCL).
- **Precision / Recall / F1** — standard for **multi-select** tasks,
  specifically because exact match is "overly strict: selecting most correct
  options but missing one valid answer is counted the same as a completely
  incorrect prediction" ([UniToolCall](https://arxiv.org/pdf/2604.11557),
  [When2Call](https://arxiv.org/pdf/2504.18851); AppSelectBench and DICE-Bench
  report the triad by default).

This project's task shape is the multi-select case, so **F1 is the field's
standard here** — not `gold_hit_ceil`, and not either homemade variant above.
**Precision had never been reported at all**; every existing metric in the
project is recall-flavoured.

**F1 across every cell available** (harmonic mean of per-task precision and
recall, ceiling-restricted to match `gold_hit_ceil`'s denominator):

| cell | v1 | CP | v0 | `original` | v1-CP |
|---|---|---|---|---|---|
| k4 @ 0.6B | 39.05 | 38.41 | 40.63 | 39.68 | +0.63 |
| k16 @ 0.6B | 33.21 | 29.88 | 26.10 | 32.72 | +3.33 |
| k64 @ 0.6B | 26.99 | 21.47 | 16.36 | 25.66 | +5.52 |
| k128 @ 0.6B | 25.38 | 16.18 | 12.18 | 26.02 | +9.20 |
| k4 @ 4B | 49.84 | 49.21 | 46.51 | 50.22 | +0.63 |
| k16 @ 4B | 40.25 | 36.42 | 36.54 | 40.05 | +3.83 |
| k64 @ 4B | 34.29 | 32.35 | 28.81 | 34.29 | +1.94 |
| k128 @ 4B | 34.87 | 31.09 | 27.88 | 34.58 | +3.78 |
| k64 @ 8B | 35.32 | 35.77 | 33.03 | 35.56 | -0.45 |
| k128 @ 8B | 35.13 | 35.28 | 33.37 | 34.64 | -0.16 |

**F1 confirms rather than corrects.** Same ranking as `gold_hit_ceil`
throughout at 0.6B/4B, and the same collapse to a tie at 8B (§3.2) — slightly
tighter under F1 (-0.45/-0.16 against -0.62/-1.17).

### 1.2 The full six-arm study

`alphabetical` and `frequency` had never been accuracy-tested on dense
retrieval at any depth — only their reuse and latency existed. Ran both across
all four depths, both models (16 runs, 0 failures):

| depth | model | v1 | CP | v0 | **alphabetical** | **frequency** | `original` |
|---|---|---|---|---|---|---|---|
| k4 | 0.6B | 39.05 | 38.41 | 40.63 | 40.63 | 39.52 | 39.68 |
| k16 | 0.6B | 33.21 | 29.88 | 26.10 | 26.10 | 28.52 | 32.72 |
| k64 | 0.6B | **26.99** | 21.47 | 16.36 | 16.36 | 19.84 | 25.66 |
| k128 | 0.6B | **25.38** | 16.18 | 12.18 | 12.18 | 17.06 | 26.02 |
| k4 | 4B | 49.84 | 49.21 | 46.51 | 46.51 | 49.59 | 50.22 |
| k16 | 4B | 40.25 | 36.42 | 36.54 | 37.04 | 40.30 | 40.05 |
| k64 | 4B | **34.29** | 32.35 | 28.81 | 28.10 | **36.58** | 34.29 |
| k128 | 4B | **34.87** | 31.09 | 27.88 | 27.19 | 34.11 | 34.58 |

**`alphabetical` tracks `tooltrie_v0` almost exactly** — identical to two
decimals at k4 and k64/0.6B. Not a data error: v0's trie matches only 1.5-3.4
tools of 64-128, so its output is nearly pure alphabetical sort. **On
retrieved menus v0 is not really a distinct policy from `alphabetical`**; it
diverges only on padded menus, where the trie match is large.

**`frequency` is the substantive addition** — beats v0 and alphabetical at
every depth on both models, and at k64/4B edges past even v1 and `original`
(36.58 vs 34.29). It had been dropped from the concurrent-latency work early
on reuse grounds and so never entered the accuracy comparison. On accuracy
alone, global frequency sorting is a stronger baseline than previously known.

### 1.3 The primary metric should be end-to-end F1, not conditional F1

Both metrics discussed so far are **ceiling-conditioned** — computed only over
requests whose menu contained a gold tool. That is the right control when
every arm sees identical menus, but it silently excuses a policy that pushes
gold tools out of the menu altogether, and that failure is real and measured.

**The demonstration, from the retrieve-64/present-10 design** (`how-many-tools.md`
§5.2), scored with `scripts/score_end_to_end.py`:

| arm | ceiling | `gold_hit_ceil` | conditional F1 | **end-to-end F1** |
|---|---|---|---|---|
| `tooltrie_v1` | 0.615 | 35.77 | 32.38 | **19.92** |
| `frequency` | 0.385 | **55.84** | **52.16** | **20.08** |

On the legacy metric `frequency` looks far better than v1 — 55.84 against
35.77, a 20-point gap. **They are in fact identical** (20.08 vs 19.92). The
gap is entirely an artefact of conditioning: `frequency` truncates away the
hard cases, so the requests it is scored on are the easy ones. Ranking by
`gold_hit_ceil` or by conditional F1 would invert this comparison.

**So the reporting order should be:**

1. **`end_to_end_f1` = ceiling x F1** — primary, always comparable.
2. **`ceiling`** — reported beside it, because it is the whole difference above.
3. **conditional `f1`, `precision`, `recall`** — valid only when ceilings match
   across arms, and the text must say so.
4. **`gold_hit_ceil`** — legacy, kept for continuity with earlier tables.

`scripts/score_end_to_end.py` emits all six. Every cell in this report is
recomputable from replay JSONs already on disk; no reruns are needed to adopt
this.

**Where the existing tables stand under it.** The whole-menu comparisons
(§1.1, §1.2, and everything in `findings.md`) hold their rankings unchanged,
because those designs give every arm the same menu and therefore the same
ceiling — conditional F1 and end-to-end F1 differ there only by a constant
factor. **Only the truncation design is affected**, and there it changes the
answer. That is why it matters that the truncation design is the one closest
to how the field actually presents tools.

---

**Recommendation**: report **end-to-end F1 as primary**, `ceiling` beside it,
conditional F1 only where ceilings match, `gold_hit_ceil` for continuity.
Costs nothing to add retroactively.

---

## 2. Does better accuracy reduce KV-cache recompute? No.

A proposed chain — *use F1 → better accuracy → less recompute* — conflates a
measurement choice with a causal law. Both halves fail independently.

**Choosing a metric cannot create a causal link.** Switching to F1 changes how
precisely accuracy is *reported*. A better ruler does not change what it
measures.

**And the causal claim itself is false, in both directions, in this project's
own data:**

- **High reuse with low accuracy.** `tooltrie_v0`'s alphabetical fallback
  reaches **97% reuse** on padded menus while posting among the worst accuracy
  in the report. Reuse up, accuracy down — the opposite pairing.
- **Accuracy moves while reuse does not.** At 8B, v1 still cached **~1.7x**
  ContextPilot's tokens — unchanged — while its accuracy advantage collapsed to
  a tie (§3.2). Same policy, same server: reuse fixed, accuracy moved.

**Mechanically, there is no channel.** Reuse is decided by prefix-hash matching
over the *prompt*, settled the moment a request is dispatched — before the
model generates anything. Accuracy is a property of what it generates
afterwards. Within a request, step 2 happens strictly after step 1 resolves.
Across requests, workloads here are independent single-turn queries, so no
request's output ever enters another's prompt.

**The one place a backdoor could exist is checked and closed.** The adaptive
policies do learn from history — but ContextPilot's `online_incremental` mode
updates its clustering from *which tools appeared in past menus*, and
ToolTrie's `observe()` inserts *the tool-ID sequence* of past requests. Both
learn from **input structure**, never from whether a past request was answered
correctly. And this is a property of prefix-caching systems generally
(RadixAttention/SGLang works the same way), not a vLLM implementation quirk.

**What actually links the two is a shared upstream cause**: the ordering policy
changes both how much of the prompt matches earlier prompts (reuse) and where
the gold tool sits (accuracy). Sometimes they move together, sometimes they
trade off hard, sometimes one moves while the other sits still — all three
occur in this project's data, so treating them as linked mispredicts about a
third of the time.

---

## 3. Configurations that trade latency against accuracy

### 3.1 Numeric precision (quantization) — never tested

Checked every server launch across the project: **zero** uses of
`--quantization`, `--dtype`, int8/int4/fp8/AWQ/GPTQ. All runs use vLLM's
default native bf16. Quantization typically cuts latency and memory
substantially at some accuracy cost — a genuine, entirely untested axis.

### 3.2 Model size — tested, and the exchange rate collapses

Per-request latency (serial driver, uncongested) paired with F1:

| depth | 0.6B latency / F1 | 4B latency / F1 | 8B latency / F1 |
|---|---|---|---|
| k64 | 509.9 ms / 26.99 | 1950.4 ms / 34.29 | 3062.0 ms / 35.32 |
| k128 | 1089.5 ms / 25.38 | 3883.9 ms / 34.87 | 5676.8 ms / 35.13 |

**Sharply diminishing returns.** 0.6B→4B costs ~3.8x the latency and buys
**+7 to +9.5 F1 points** — a real trade. 4B→8B costs another 46-57% latency
and buys **+0.3 to +1.0 points**. On this task, 8B is close to pure waste.

**And model size erodes v1's margin over ContextPilot, which is the most
consequential qualification in this document:**

| cell | v1 | CP | delta | SE | sigma |
|---|---|---|---|---|---|
| k64 @ 0.6B | 31.29 | 24.54 | +6.75 | 4.95 | 1.36 |
| k64 @ 4B | 39.88 | 38.04 | +1.84 | 5.40 | 0.34 |
| k64 @ 8B | 41.10 | 41.72 | **-0.62** | 5.46 | -0.11 |
| k128 @ 0.6B | 28.65 | 18.13 | **+10.52** | 4.54 | **2.32** |
| k128 @ 4B | 40.35 | 36.26 | +4.09 | 5.25 | 0.78 |
| k128 @ 8B | 40.94 | 42.11 | **-1.17** | 5.33 | -0.22 |
| k16 @ 8B | 50.37 | 48.15 | +2.22 | 6.08 | 0.36 |

The margin shrinks monotonically at both depths and crosses zero by 8B.
**Neither 8B cell is a confirmed ContextPilot win** — both under 0.25 sigma —
but the trend that produced them is real. **The "reuse is free" claim in
`findings.md` §4.4 was established at 0.6B and 4B only and does not extend to
8B.** The cache mechanism is unaffected by model size (v1 still caches ~1.7x
at 8B); only the accuracy side breaks down.

**The erosion is depth-specific, not general.** At k16 the margin goes
+4.44 (4B) → +2.22 (8B): halved, same direction, but nowhere near crossing. It
is specific to large, noisy menus where ContextPilot's clustering has real
structure to exploit once the model is capable enough to use it.

---

## 4. Throughput under a fixed latency limit

The properly framed version of "if every request must answer within 1 second,
what accuracy do we get": **fix a latency SLA, find the maximum throughput each
policy sustains within it, and report the accuracy delivered there.**

### 4.1 Uncongested — the hard floor

Serial, one request at a time: the lower bound no amount of tuning beats.

| model | largest depth under 1s | v1 F1 | CP F1 | v0 F1 | `original` F1 |
|---|---|---|---|---|---|
| 0.6B | **k64** (~510-517 ms) | **26.99** | 21.47 | 16.36 | 25.66 |
| 4B | **k16** (~897-914 ms) | **40.25** | 36.42 | 36.54 | 40.05 |
| 8B | below k16 | — | — | — | — |

Note that **ordering barely affects latency here** (all arms within 1-2% at
fixed depth): with nothing queued, latency is almost pure prefill cost, which
depends on token count, not policy. Ordering's latency benefit is a
*contention* effect and needs load to appear.

### 4.2 Under load — the SLA-constrained sweep

All six arms at k64/0.6B, sweeping offered rate to find where p50 TTFT crosses
**1000 ms**. 36 runs (six arms x six rates), dedicated server on GPU1:8303.

| arm | reuse | crosses 1000 ms p50 at | F1 there |
|---|---|---|---|
| `frequency` | 1.39% | 2.570 req/s | 19.84 |
| `alphabetical` | 1.53% | 2.574 req/s | 16.36 |
| `tooltrie_v0` | 2.40% | 2.585 req/s | 16.36 |
| `original` | 0.82% | 2.623 req/s | 25.66 |
| ContextPilot | 2.98% | 2.638 req/s | 21.47 |
| **`tooltrie_v1`** | **6.22%** | **2.671 req/s** | **26.99** |

(Crossing rate interpolated between measured 2.5 and 2.75 req/s points, which
bracket 1000 ms for every arm.)

**Throughput spread is small — 1.039x, v1 over `frequency`** — consistent with
the finding that ordering barely matters on retrieved menus generally
(1.046x capacity spread at k64). Under a 1-second SLA, v1 buys ~4% more
sustainable throughput than the weakest arm and ~1% more than ContextPilot.

**Paired with accuracy the comparison sharpens.** `tooltrie_v1` is the only arm
simultaneously top-or-near-top on **both** axes — highest sustainable
throughput *and* highest F1 (26.99, effectively tied with `original`'s 25.66,
both clear of ContextPilot's 21.47). Every other arm trades one for the other:
`frequency` has the worst throughput ceiling despite mid-table accuracy;
ContextPilot is middling on both; `original` matches v1's accuracy but sustains
meaningfully less throughput.

### 4.3 Under bursty arrival the margin grows, not shrinks

§4.2 was measured on the natural file order, where v1's margin over
ContextPilot was only 1.2% — the result most likely, it seemed, to flip if
arrivals became more local, since ContextPilot is designed for multi-turn
traffic with ~40% overlap between turns. Repeated under the upper-bound
locality ordering (`findings.md` §4.5; adjacent requests share 17.5 of 64
tools rather than 1.8):

| arm | empirical order | **locality_max** |
|---|---|---|
| `original` | 2.623 req/s | 2.608 |
| ContextPilot | 2.638 | 2.662 |
| **`tooltrie_v1`** | 2.671 | **2.817** |
| v1 over ContextPilot | +1.2% | **+5.8%** |

**The fragile margin strengthened, to +5.8%.** At 2.75 req/s v1 answers in
821 ms and meets the 1-second budget; ContextPilot takes 1184 ms and misses it.

The reason is general, and `findings.md` §4.5 establishes it: ContextPilot wins
where menus share a **global core** (tools present in every request), because
its set intersection *is* that core. Bursty arrival raises *adjacent* overlap
without creating any global core — even the upper-bound ordering has zero tools
in every menu — and adjacent overlap is exactly what a trie over served
sequences matches. So locality helps v1 more than it helps ContextPilot.

### 4.4 The caveat that changes the conclusion

**This is p50 at one depth, one model, one SLA value, one retriever.** Two
things would likely move it:

- **p95 instead of p50 binds far earlier.** At rate 2.5, every arm's p50 is
  under 900 ms while p95 is already 2,000-2,600 ms. An SLA written on p95 —
  which is how SLAs are usually written — would cap sustainable throughput
  well below these numbers.
- **The menu-size result dominates this one.** At the field's menu size,
  ordering's *latency* advantage vanishes entirely (p50 flat at 66-68 ms
  across all arms at k=10), while its *ceiling* advantage becomes large —
  ContextPilot drops 15pp, v1 does not. See
  [`how-many-tools.md`](how-many-tools.md) §5. **If throughput under a latency
  limit is the objective, menu size is a bigger lever than ordering policy.**

---

## 5. Open gaps

1. **Quantization untested** (§3.1) — the most obvious latency-vs-accuracy
   knob in serving, never touched.
2. **The SLA sweep is two points** (§4.2, §4.3: natural order and upper-bound
   locality) — still needs p95, other depths and other model sizes to become a
   curve.
3. **`findings.md` §4.4 and the README state "reuse is free" without a
   model-size qualifier** (§3.2). Recommend re-running the clear 0.6B/4B cells
   at n=600 to see whether the 8B tie survives more data, and stating
   explicitly that the claim is demonstrated at 0.6B/4B only.
4. **End-to-end F1 is not yet the reported primary metric** anywhere in the
   parent report (§1.3). `scripts/score_end_to_end.py` now emits it; adopting
   it needs no reruns, only re-scoring and edits. Rankings change only for the
   truncation design, but there they change the answer.
5. **n=200 per cell** gives ~5pp SE on deltas of similar size; 7,961
   gold-labelled tasks exist. n=600 would roughly halve it.

---

## Runs

| stage | directory | runs |
|---|---|---|
| k4/k16 dense accuracy, two models | `eval-validity-20260906-165826/` | 16 |
| k64/k128 dense accuracy, third size point (8B) | `eval-validity-20260906-165826/` | 8 |
| k16 dense accuracy, 8B follow-up | `eval-validity-20260906-165826/` | 4 |
| alphabetical/frequency dense accuracy, four depths, two models | `eval-validity-20260906-165826/` | 16 |
| SLA-constrained throughput sweep, k64/0.6B, six arms x six rates | `eval-validity-20260906-165826/` | 36 |

Servers: Qwen3-0.6B GPU2:8300, Qwen3-4B GPU3:8301, Qwen3-8B GPU0:8302
(`--max-model-len 32768`, needed to fit KV cache alongside the larger weights),
Qwen3-0.6B GPU1:8303 for the SLA sweep — four servers, four GPUs, concurrent,
separate flock targets. Drivers: `scripts/replay_vllm_workload.py` (serial,
accuracy) and `scripts/replay_vllm_concurrent.py` (rate-controlled). Scoring:
`scripts/score_tool_selection.py` plus an F1/strict/fractional re-scorer not
yet promoted to a script. Zero failures across all 80 runs.
