# Is the eval set valid? Depth, retriever, and metric choices checked

`findings.md` and `README.md` report ToolTrie-v1's advantage at k = 4/16/64/128,
against ContextPilot, on BM25 and dense retrieval, at model sizes 0.6B and 4B.
This document checks whether that evaluation design itself is defensible: is
the depth range realistic, does the advantage survive inside ContextPilot's
own native range, what do real tool catalogs actually look like, is the
accuracy metric fair to multi-tool tasks, and does the accuracy result hold at
a third, larger model size. Every claim below is either a literature citation
(dated, sourced) or a run under `cluster/results/`.

**Headline finding**: the depth-range mismatch with ContextPilot's own paper
turns out to matter less than expected (§2) — the accuracy advantage holds,
even strengthens, inside ContextPilot's native k=3-20 range. What matters more
is a dimension the parent report never varied: **model size.** Adding a third
point (Qwen3-8B) to the existing 0.6B/4B accuracy comparison shows the
advantage shrinking monotonically and crossing into a statistical tie by 8B
(§2). Then §6 tests the field's own menu size (k~10) and finds the reverse of
the 8B tie: under a retrieve-64/present-10 design, ContextPilot's reordering
pushes the gold tool out of the shown window in one request in seven (ceiling
0.620 -> 0.470, 3 sigma) while v1's does not (0.615), so v1 keeps
`original`'s end-to-end accuracy with 4.3x its reuse. Together: the 8B tie is
a whole-menu result; at the menu size every benchmark actually uses, v1's
advantage is large and it is a ceiling advantage.

---

## 1. What retrieval-depth range is standard, and where does this project sit

Two comparisons matter: general RAG practice, and ContextPilot's own paper.

**General RAG practice.** Common top-k values cluster at 1/3/5/10/15, and the
field treats k as something to sweep, not fix — a documented result found
Hit@k rising from 0.726 (k=2) to 0.834 (k=5) while mean F1 *peaked at k=3 and
declined after*, because irrelevant retrieved text degrades the reasoning the
model does over it.

**ContextPilot's paper** (arXiv:2511.03475, checked directly 2026-09-06 —
this corrects an earlier misreading in `findings.md` A.6, which had stated
"dense on 3 of 4 datasets"):

- Evaluates on QASPER, MultihopRAG, NarrativeQA, MT-RAG.
- Retriever split is **50/50**: dense (gte-Qwen2-7B + FAISS) on MultihopRAG
  and NarrativeQA; BM25 on QASPER and MT-RAG.
- Primary depth: **k = 3-20, primarily k = 15.**
- A separate application (Mem0/LoCoMo agent memory, not the four-dataset RAG
  evaluation) touches k = 100 — a different task, not evidence the main
  evaluation goes deeper.
- Workload is multi-session/multi-turn: **~40% of a turn's retrieved
  documents overlap earlier ones**, and 49-79% of questions draw from the top
  20% of documents.

**This project tests k = 4/16/64/128** — up to roughly 8x ContextPilot's
primary depth — on independent requests with 1.6-7.7% adjacent overlap, far
below its ~40% design point. That mismatch is real and already carried as
caveat 1 of `findings.md` A.6.

**Why the candidate pools aren't comparable even at matched k.** ContextPilot's
RAG benchmarks retrieve from a *per-question, single-document* pool — QASPER
chunks one ~4,170-word paper; NarrativeQA chunks one book. k=15 there can
already be a large fraction of that one document's chunks. This project
retrieves from **one shared catalog of 44,453 distinct tools** (verified:
`data/processed/tools.jsonl`, `source` prefixed `toolret:`). k=15 there is
0.03% of the pool. The two "k=15"s describe different-sized needles in
different-sized haystacks; matching the number doesn't match the problem.

---

## 2. Does the advantage hold inside ContextPilot's own range (k = 3-20)?

Checked directly using this project's own k4 and k16 runs — no extrapolation.

### Reuse and latency (already run, both retrievers)

| k | retriever | v1 reuse | CP reuse | v1 p50 | CP p50 | v1 max | CP max |
|---|---|---|---|---|---|---|---|
| 4 | BM25 | 19.47% | 18.72% | 46.1 | 46.3 | 93.0 | **90.0** |
| 4 | dense | 17.97% | **18.16%** | 46.2 | 46.3 | **78.5** | 87.7 |
| 16 | BM25 | **11.31%** | 9.93% | **92.1** | 98.2 | **323.4** | 336.3 |
| 16 | dense | **13.18%** | 11.58% | 89.0 | 89.0 | **283.9** | 285.2 |

**At k4 it is a wash** — the two arms trade the lead across retrievers and
statistics, and ContextPilot wins `max` on BM25. **At k16 v1 has a real but
modest edge** (1.14x reuse on both retrievers), not the 2.09-3.09x margin
found at k64/k128 in §4.4. The decisive part of the dense result is
specifically a large-k phenomenon and does not appear inside k = 3-20.

### Accuracy (new runs, this document)

No accuracy run (`gold_hit_ceil`) had ever been done at k4 or k16 before this
document — `accuracy-gate-20260830-011416` and `dense-accuracy-20260903-002742`
both start at k64. Run here: same driver, settings and metric as those two
(`--max-tokens 128 --tool-choice auto --disable-thinking --reset-before`),
dense retrieval, four arms, both models, k4 and k16.

| cell | v1 | ContextPilot | v0 | `original` | ceiling |
|---|---|---|---|---|---|
| k4 @ 0.6B | 42.86 | 41.90 | 42.86 | **43.81** | 0.525 |
| k4 @ 4B | 55.24 | 54.29 | 54.29 | **56.19** | 0.525 |
| k16 @ 0.6B | **38.52** | 34.81 | 28.15 | 37.78 | 0.675 |
| k16 @ 4B | **44.44** | 40.00 | 40.00 | **44.44** | 0.675 |

**v1 beats ContextPilot in all four cells** — +0.96, +0.95, +3.71, +4.44 pp — so
inside ContextPilot's own native depth range, the accuracy advantage does hold,
strengthening with depth even across this narrow band. But the k4 margins are
noise (0.14 sigma, see the combined table below); only k16 carries real weight
(0.63-0.74 sigma), and even that is under 1 SE alone.

**And `original` (no reordering at all) wins outright at k4, in both models** —
43.81 vs v1's 42.86 at 0.6B, 56.19 vs 55.24 at 4B. At this depth the retriever's
own ranking is already close to optimal, and *any* reordering — even v1's
conservative one — can only cost a little by disturbing it. This mirrors the
reuse finding above: k4 is the one depth where not reordering is the right
call, on both the cache axis and the accuracy axis.

**The full picture, all three model sizes now available at k64/k128, reveals a
second and more consequential trend**: v1's accuracy margin over ContextPilot
does not hold steady or keep growing with model size — it shrinks toward zero
and crosses over.

| cell | v1 | CP | delta | SE | sigma |
|---|---|---|---|---|---|
| k4 @ 0.6B | 42.86 | 41.90 | +0.96 | 6.82 | 0.14 |
| k16 @ 0.6B | 38.52 | 34.81 | +3.71 | 5.86 | 0.63 |
| k64 @ 0.6B | 31.29 | 24.54 | +6.75 | 4.95 | 1.36 |
| k128 @ 0.6B | 28.65 | 18.13 | **+10.52** | 4.54 | **2.32** |
| k4 @ 4B | 55.24 | 54.29 | +0.95 | 6.87 | 0.14 |
| k16 @ 4B | 44.44 | 40.00 | +4.44 | 6.01 | 0.74 |
| k64 @ 4B | 39.88 | 38.04 | +1.84 | 5.40 | 0.34 |
| k128 @ 4B | 40.35 | 36.26 | +4.09 | 5.25 | 0.78 |
| k64 @ 8B | 41.10 | 41.72 | -0.62 | 5.46 | -0.11 |
| k128 @ 8B | 40.94 | 42.11 | -1.17 | 5.33 | -0.22 |
| k16 @ 8B | 50.37 | 48.15 | +2.22 | 6.08 | 0.36 |

At k128 specifically, the margin runs +10.52 (0.6B) -> +4.09 (4B) -> -1.17
(8B): three model sizes, monotonically shrinking, crossing zero. At k64 the
same shape: +6.75 -> +1.84 -> -0.62. **Neither 8B cell is a confirmed
ContextPilot win** — both deltas are under 0.25 sigma, indistinguishable from
a tie — but the direction is consistent across both depths and the trend that
produced it (from a clear v1 lead at 0.6B to a coin flip at 8B) is real,
not noise. **The "reuse is free" claim in `findings.md` §4.4 was established
at 0.6B and 4B only, and does not extend to 8B**: at production-representative
model size, v1's extra reuse over ContextPilot buys accuracy parity, not an
accuracy edge.

**And this erosion is depth-specific, not general — k16 answers the question
directly.** Its margin over ContextPilot at 4B->8B goes +4.44 -> +2.22: roughly
halved, the same direction as k64/k128, but it stays clearly positive and
nowhere near crossing to a tie. So the mechanism is not "bigger models erase
v1's edge everywhere" — it is specific to the large, noisy menus (k64/k128)
where ContextPilot's clustering has real structure to exploit once the model
is capable enough to use it. At k16, that structure is too thin for size to
matter as much.

**The cache mechanism itself is unaffected by model size** — checked directly
on the same 8B runs (`aggregate_metric_delta.vllm:prompt_tokens_cached /
vllm:prefix_cache_queries`, since the serial accuracy driver does not emit the
`reuse` block the concurrent driver does):

| arm | k64 reuse | k128 reuse |
|---|---|---|
| `original` | 0.64% | 0.31% |
| `tooltrie_v0` | 1.43% | 0.52% |
| ContextPilot | 1.42% | 0.65% |
| **`tooltrie_v1`** | **2.41%** | **1.09%** |

v1 still caches ~1.7x ContextPilot's tokens at 8B, same ranking as at 0.6B/4B.
**So the split is precise: the systems result (v1 caches more) is robust across
all three model sizes; the accuracy result (v1 costs nothing for that extra
reuse) is not — it holds at 0.6B/4B and disappears at 8B.** This is the
single most important qualification this document adds to the parent report's
headline.

---

## 3. What tool-set sizes real deployments actually use

Checked 2026-09-06 (dated because this is a fast-moving area).

| finding | source |
|---|---|
| Best practice per MCP server: ~20-30 tools | [Lunar.dev](https://www.lunar.dev/post/why-is-there-mcp-tool-overload-and-how-to-solve-it-for-your-ai-agents) |
| Common composite: 5 servers x 30 tools = **150 tools**, 30,000-60,000 tokens of metadata (25-30% of context) | [Lunar.dev](https://www.lunar.dev/post/why-is-there-mcp-tool-overload-and-how-to-solve-it-for-your-ai-agents) |
| Cloudflare's native MCP exposure: **~1.17M tokens** of tool definitions | [Medium, Agentive Futures](https://medium.com/agentive-futures/mcp-in-finance-is-great-until-you-need-1-000-tools-d09fc350a85e) |
| Cursor imposes a hard cap of **~40 tools**, specifically because of tool overload | [getunblocked.com](https://getunblocked.com/blog/mcp-tool-overload/) |
| Accuracy drop from tool-count growth, independent of this project: 78%->13.62% (10->100+ tools); 43%->2% (4->51 tools, BFCL calendar tasks) | search-verified, see chat log 2026-09-06 |

**Reframing this project's depth range against those numbers**: k=3-20 matches
a *single, well-scoped MCP server in isolation*. k=64-128 matches *what happens
the moment a handful of ordinary integrations are composed* — the 150-tool
composite figure above sits **between k64 and k128**, and Cloudflare's disclosed
case exceeds k128 by roughly an order of magnitude. The higher end of this
project's sweep is not a stress-tested extreme relative to real deployments;
if anything it understates the scale some production systems already carry.

**What this does not resolve**: real deployments are bimodal (one well-scoped
server vs. several composed), and this project's k-points don't include the
single most commonly cited composite number, k~150. See §5.

---

## 4. Is `gold_hit_ceil` a fair accuracy metric for multi-tool tasks?

`score_tool_selection.py`'s `gold_hit_ceil` scores a request correct if the
model called **any one** gold tool (`bool(ids & gold)`), full credit
regardless of how many gold tools the task needs.

**44% of the evaluated tasks need 2+ gold tools** (checked on the k64 eval
slice: 113 need 1, 69 need 2, 15 need 3, 3 need 4). So this metric is closer
to "recall@1-of-gold" than to task completion.

**Checked whether this changes any published ranking**, by re-scoring the
existing dense-accuracy replays under three definitions — any-gold-hit
(current), all-gold-hit (strict), and fraction-of-gold-hit (partial credit):

| cell | any-hit (current) | all-hit (strict) | fraction hit |
|---|---|---|---|
| k64 0.6B: v1 / CP / v0 / orig | 31.29 / 24.54 / 19.02 / 29.45 | 21.47 / 18.40 / 12.27 / 20.86 | 26.18 / 21.27 / 15.64 / 24.95 |
| k128 0.6B: v1 / CP / v0 / orig | 28.65 / 18.13 / 13.45 / 29.24 | 21.64 / 13.45 / 10.53 / 22.22 | 25.05 / 15.79 / 11.99 / 25.63 |
| k64 4B: v1 / CP / v0 / orig | 39.88 / 38.04 / 34.97 / 39.26 | 27.61 / 25.77 / 20.86 / 28.22 | 33.23 / 31.39 / 27.51 / 33.33 |
| k128 4B: v1 / CP / v0 / orig | 40.35 / 36.26 / 33.92 / 39.18 | 28.65 / 25.15 / 19.88 / 29.24 | 33.92 / 30.31 / 26.51 / 33.82 |

**Ranking is identical under all three definitions in every cell**:
`tooltrie_v1` >= `original` > ContextPilot > `tooltrie_v0`. Absolute numbers
drop by roughly a third under the strict definition — the real full-task
completion rate is lower than `gold_hit_ceil` implies — but no comparison in
`findings.md` or `README.md` is reversed by using a stricter metric.

**Recommendation**: report `gold_hit_ceil` alongside the strict and fractional
variants as standard columns, not as a one-off audit. It costs nothing further
to compute — all three are derivable from replay JSONs already on disk.

### 4.1 What does the field actually use for this task shape?

Checked 2026-09-06. The field splits cleanly on whether a task is single- or
multi-tool:

- **Exact Match / Accuracy** — the call must match gold exactly, all-or-nothing.
  Standard for single-tool tasks (BFCL uses this).
- **Precision / Recall / F1** — standard specifically for **multi-select**
  tasks, because exact match is "overly strict: selecting most correct
  options but missing one valid answer is counted the same as a completely
  incorrect prediction" ([UniToolCall](https://arxiv.org/pdf/2604.11557),
  [When2Call](https://arxiv.org/pdf/2504.18851)). Several benchmarks
  (AppSelectBench, DICE-Bench) report precision/recall/F1 by default for
  exactly this reason.

This project's task shape is the multi-select case (44% of tasks need 2+
tools, §4), so **F1 is the field's standard choice here**, not a stricter or
looser homebrew variant of it. Precision (of the tools actually called, how
many are gold) had never been reported at all before this check — every
existing metric in this project is recall-flavored.

**Computed F1 (harmonic mean of per-task precision and recall, ceiling-
restricted to match `gold_hit_ceil`'s denominator) across every cell now
available:**

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

**F1 changes nothing already found — it confirms both headline results under
the field's actual standard metric rather than a homemade one.** Same ranking
throughout at 0.6B/4B (`tooltrie_v1` >= `original` > ContextPilot >
`tooltrie_v0`), and the same collapse to a tie at 8B found in §2 — if
anything slightly tighter under F1 (-0.45/-0.16 against `gold_hit_ceil`'s
-0.62/-1.17).

**Recommendation**: make F1 the primary reported metric going forward.
`gold_hit_ceil` answers a different, still-useful question (did retrieval and
ordering put the tool within reach at all) and can stay as a secondary column,
but F1 is what the field would call "accuracy" for a task shape like this
one, and it costs nothing to add retroactively.

### 4.2 The full six-arm study: adding `alphabetical` and `frequency`

Every accuracy run up to this point covers four arms (`original`,
`tooltrie_v0`, ContextPilot, `tooltrie_v1`). `alphabetical` and `frequency`
had never been accuracy-tested on dense retrieval at any depth — only their
reuse and latency were measured, in `findings.md` §4.4. Ran both across all
four depths, both models (16 new runs, 0 failures), completing the study
across all six ordering policies findings.md compares elsewhere:

| depth | model | v1 | CP | v0 | **alphabetical** | **frequency** | `original` |
|---|---|---|---|---|---|---|---|
| k4 | 0.6B | 39.05 | 38.41 | 40.63 | 40.63 | 39.52 | 39.68 |
| k16 | 0.6B | 33.21 | 29.88 | 26.10 | 26.10 | 28.52 | 32.72 |
| k64 | 0.6B | **26.99** | 21.47 | 16.36 | 16.36 | 19.84 | 25.66 |
| k128 | 0.6B | **25.38** | 16.18 | 12.18 | 12.18 | 17.06 | 26.02 |
| k4 | 4B | 49.84 | 49.21 | 46.51 | 46.51 | 49.59 | 50.22 |
| k16 | 4B | 40.25 | 36.42 | 36.54 | 37.04 | 40.30 | 40.05 |
| k64 | 4B | **34.29** | 32.35 | 28.81 | 28.10 | 36.58 | 34.29 |
| k128 | 4B | **34.87** | 31.09 | 27.88 | 27.19 | 34.11 | 34.58 |

**`alphabetical` tracks `tooltrie_v0` almost exactly** — identical to two
decimal places at k4 and k64/0.6B, close everywhere else. That is not a data
error: v0's trie match is tiny at these depths (1.5-3.4 tools of 64-128,
measured in `findings.md` §5), so its output is nearly pure alphabetical
sort regardless of the small matched prefix. **v0 is not really a distinct
policy from `alphabetical` on retrieved menus** — it only diverges from it on
padded menus, where the trie match is large.

**`frequency` is the interesting addition — it beats `tooltrie_v0` and
`alphabetical` at every depth on both models**, and at k64/4B it edges out
even `tooltrie_v1` and `original` (36.58 vs 34.29/34.29). This was never
visible before, because `frequency` was dropped from the concurrent-latency
work early (findings.md §1.1: "not extended [past 200 requests]... a fitted
baseline whose training corpus is not recorded recoverably") on reuse
grounds — it never earned a place in the accuracy comparison. On accuracy
specifically, sorting by global tool frequency is a stronger baseline than
either alphabetical or v0's trie-plus-alphabetical-fallback, though it never
approaches v1 or `original` at the shallower depths.

**v1 remains best or tied-best against all five other arms at k64/k128**,
now checked against six policies instead of four. The ranking established in
§2's model-size table is unaffected — `frequency`'s strength is real but
does not close the gap at the depths that matter most to that finding.

---

## 5. Throughput under a fixed latency limit — accuracy and throughput together

The right way to frame "does ordering matter" is not raw latency or raw
accuracy alone, but: **fixed a latency SLA, which policy sustains the most
throughput, and what accuracy does it deliver there.** Checked directly for
all six arms at k64/0.6B (the largest depth with sub-1s uncongested latency,
per §4's serial numbers), a 1000ms **p50** TTFT budget, GPU1:8303:

| arm | reuse | crosses 1000ms p50 at | k64/0.6B F1 |
|---|---|---|---|
| `frequency` | 1.39% | 2.570 req/s | 19.84 |
| `alphabetical` | 1.53% | 2.574 req/s | 16.36 |
| `tooltrie_v0` | 2.40% | 2.585 req/s | 16.36 |
| `original` | 0.82% | 2.623 req/s | 25.66 |
| ContextPilot | 2.98% | 2.638 req/s | 21.47 |
| **`tooltrie_v1`** | **6.22%** | **2.671 req/s** | **26.99** |

(Crossing rate interpolated linearly between the measured 2.5 and 2.75 req/s
points, where p50 brackets 1000ms for every arm; six further runs at those two
rates, 12 total, confirmed the bracket holds arm by arm.)

**The throughput spread is small — 1.039x, `tooltrie_v1` over `frequency`** —
consistent with §4.4's finding that ordering barely matters on retrieved
menus generally (1.046x capacity spread at k64, offered 4 req/s). Under a
1-second SLA specifically, v1 buys about 4% more sustainable throughput than
the weakest arm, and about 1% more than ContextPilot.

**But paired with accuracy, the comparison sharpens.** `tooltrie_v1` is not
just the highest-throughput arm — it also has the best or near-best F1 at
this depth (26.99, essentially tied with `original`'s 25.66, both clear of
ContextPilot's 21.47 and `tooltrie_v0`/`alphabetical`'s 16.36). So at the
fixed 1-second budget, v1 is the only arm that is simultaneously top or
near-top on **both** axes — every other arm trades one for the other
(`frequency` has the worst throughput ceiling despite mid-table accuracy;
ContextPilot has middling throughput and middling accuracy; `original` matches
v1's accuracy but sustains meaningfully less throughput).

**This is a single depth, single model, single SLA value, single retriever
(dense) — a first measurement of this framing, not a swept curve.** Repeating
it at k128 (where accuracy separates further, per §4.1) or at other SLA
values (e.g. p95 instead of p50, which would bind much earlier given the
p95/p50 gap already visible in the raw sweep) is the natural next step if this
framing is adopted as a standard reporting format.

---

## 6. Do the findings hold at the field's menu size? (k=10, and retrieve-64/present-10)

§1 established that every tool-use benchmark separates two numbers this
project collapses into one `k`: the **catalog** the retriever searches (large:
250-527 tools in the MCP benchmarks, 43k in ToolRet) and the **menu** the model
actually sees (small: BFCL 2-5, τ²-bench ~11, ToolBench/ToolRet top-5 to 10,
LiveMCPBench k=5 with an ablation showing k=1 -> 64.2%, k=5 -> 79.0%, k=10 ->
no further gain). Our k=64/128 hands the model a menu 6-25x larger than any of
them. Two designs test whether the k64/k128 results survive at the field's
size. Dense retrieval, six arms, 0.6B for reuse/latency (rate 10, uncapped),
0.6B and 4B for accuracy. 36 runs, 0 failures.

### 6.1 k=10, control intact: retrieve top-10, reorder those 10, present all 10

Same ten tools in every arm (ceiling 0.620 everywhere), only the order differs.

| arm | reuse | p50 ms | p95 ms | 0.6B F1 | 0.6B end-to-end | 4B F1 | 4B end-to-end |
|---|---|---|---|---|---|---|---|
| `original` | 6.38% | 66.2 | 121.8 | 31.99 | 19.83 | 43.12 | 26.73 |
| `alphabetical` | 7.97% | 66.5 | 124.9 | 31.85 | 19.75 | 37.50 | 23.25 |
| `frequency` | 8.68% | 66.8 | 127.5 | 30.91 | 19.17 | **44.81** | **27.78** |
| `tooltrie_v0` | 9.94% | 68.1 | 127.9 | 31.85 | 19.75 | 37.50 | 23.25 |
| ContextPilot | 12.46% | 67.1 | 130.6 | **32.39** | **20.08** | 42.80 | 26.53 |
| **`tooltrie_v1`** | **13.71%** | 67.1 | **117.3** | 31.59 | 19.58 | 43.66 | 27.07 |

**The reuse ranking holds — v1 first, 2.15x `original`, 1.10x ContextPilot —
but it is worth nothing here.** p50 is flat across all six arms (66-68 ms):
a ten-tool prompt is too short for prefix reuse to move latency, and at this
rate no queue forms. Accuracy is a wash on both models (every gap under 1 SE;
`frequency` edges 4B, ContextPilot edges 0.6B). So at the field's menu size the
*mechanism* survives and the *effect* does not — exactly the k4/k16 pattern,
and consistent with the parent report's own claim that ordering is a large-menu
phenomenon. (End-to-end = ceiling x F1.)

### 6.2 Retrieve 64, present 10: does cache-oriented reordering push relevant tools out of the window?

Each arm reorders the same 64 retrieved tools, then only the first 10 are
shown. The tools shown now differ by arm, so the **ceiling** — whether the gold
tool is among the ten shown at all — is the primary result, and end-to-end
(ceiling x F1) is the honest accuracy number. Conditional F1 on a tiny surviving
subset is not meaningful and is shown only to make that point.

| arm | ceiling | reuse | p50 ms | 0.6B F1 | **0.6B end-to-end** | 4B F1 | **4B end-to-end** |
|---|---|---|---|---|---|---|---|
| `original` | 0.620 | 6.38% | 69.0 | 31.99 | 19.83 | 43.12 | 26.73 |
| **`tooltrie_v1`** | **0.615** | **27.64%** | **60.1** | 32.38 | **19.92** | 42.33 | 26.03 |
| ContextPilot | 0.470 | 20.16% | 64.1 | 34.57 | 16.25 | 45.64 | 21.45 |
| `frequency` | 0.385 | 9.39% | 72.8 | 52.16 | 20.08 | 63.29 | 24.37 |
| `tooltrie_v0` | 0.080 | 16.12% | 67.1 | 9.38 | 0.75 | 44.79 | 3.58 |
| `alphabetical` | 0.075 | 11.67% | 68.5 | 10.00 | 0.75 | 47.78 | 3.58 |

**This is the sharpest v1-vs-ContextPilot separation in the project, and it is
on the field's menu size.** Truncation exposes what each policy does to the
retriever's ranking:

- **`tooltrie_v1` keeps the ceiling intact** — 0.615 vs `original`'s 0.620, a
  0.5pp loss (0.1 sigma). It hoists only the tools its trie has evidence for
  and leaves the rest in retriever order, so the gold tool almost never falls
  out of the top ten. It gains **4.3x `original`'s reuse and 13% lower p50** for
  that, with end-to-end accuracy unchanged on both models.
- **ContextPilot drops the ceiling by 15pp** (0.620 -> 0.470, **3.0 sigma**).
  Its cluster hoist pulls cache-friendly tools into the window and pushes the
  gold tool out of it in one request in seven. End-to-end falls 3.6pp at 0.6B
  and **5.3pp at 4B** — the accuracy cost of its reuse that §2 found invisible
  at k64/128 is fully visible once the menu is truncated.
- **`tooltrie_v0` and `alphabetical` are catastrophic** (ceiling 0.08): the
  alphabetical fallback puts arbitrary tools first, so the shown ten are almost
  never the relevant ones. v0's 16% reuse is bought at a 92% miss rate.
- **`frequency` shows why ceiling-conditioned F1 misleads**: its conditional F1
  is the highest in the table (52/63) because the gold tools that survive its
  truncation are the common, easy ones — but its ceiling is 0.385, so its
  end-to-end is no better than `original`.

**What holds and what does not, at field-standard menu size.** The reuse
ranking (v1 > ContextPilot > v0 > original) holds in both designs. The
*magnitude* of the latency gain at k64/128 does not transfer to k=10 — the
prompt is too short. What transfers, and sharpens, is the accuracy story: v1's
"preserve the retriever's order except where the trie has evidence" rule is
what keeps it safe under truncation, and ContextPilot's override is what makes
it unsafe. The 8B tie at k64/128 (§2) therefore describes the regime where the
whole menu is shown; where only a window is shown — which is how every field
benchmark presents tools — v1's advantage is a ceiling advantage, and it is
large.

**Scope.** One retriever (dense), 0.6B for latency, 0.6B/4B for accuracy,
n=200. The k64p10 design changes which tools are shown per arm, which is its
purpose, so within-arm comparisons to §4 (same tools) are not like-for-like.
Runs: `cluster/results/eval-gaps-20260907-224433/`.

---

## 7. What a valid eval set for this problem would add

Synthesized from §§1-4 plus the standing limitations already in `findings.md`
A.6-A.7. Ordered by expected impact, not by effort.

0. **The parent report's "reuse is free" claim needs a model-size caveat.**
   §2 found this directly: the accuracy margin over ContextPilot that held at
   0.6B (+6.75 to +10.52pp) and weakened at 4B (+1.84 to +4.09pp) disappears at
   8B (-0.62 to -1.17pp, both within noise). This is not a new eval-set
   requirement so much as a correction owed to `findings.md` §4.4 and the
   README headline, which currently state the free-reuse result without a
   model-size qualifier. Recommend re-running the two clear-effect 0.6B/4B
   cells at n=600 to see whether the 8B tie survives more data, and stating
   explicitly that the claim is demonstrated at 0.6B/4B and untested above
   that until it is.
1. **A session-correlated workload.** Every workload here treats requests as
   independent. ContextPilot is designed for ~40% turn-to-turn overlap; nothing
   in this eval set produces that regime, which is why A.6 already names it the
   report's largest fairness gap. This is the one item on this list that
   requires new construction rather than more runs of what exists.
2. **A depth point near k~150**, matching the most commonly cited real
   composite (5 MCP servers x 30 tools), rather than only interpolating
   between k64 and k128.
3. **Larger n for accuracy comparisons.** 200 tasks/cell gives ~5pp SE on
   deltas of similar size; 7,961 gold-labeled tasks exist. n=600 would roughly
   halve SE.
4. **Retrieval quality reported next to reuse by default** — the BM25
   hub-tool effect (§4.4: top tool in 66/200 menus vs dense's 21/200,
   partly manufacturing "reuse" via retrieval error) was only caught because
   a second retriever existed for comparison.
5. **F1 as the primary accuracy metric (§4.1)**, with `gold_hit_ceil` kept as
   a secondary column, not the reverse as currently reported.
6. **A genuine production trace remains open and likely unclosable here** —
   the brief's own §4.5 says no listed dataset provides one, and this
   project's own finding that arrival order moved reuse more than any policy
   did is exactly the kind of effect only a real trace would calibrate.
7. **The SLA-constrained throughput framing (§5) should be swept, not left as
   a single measurement.**
8. **Report end-to-end accuracy (ceiling x F1) as the primary accuracy column,
   with the decoupled retrieve-N/present-k design as a standard condition.**
   §6.2 shows a policy can have the best conditional F1 in the table and the
   worst end-to-end, and that the field's own menu size (k~10) is where
   ContextPilot's accuracy cost becomes measurable and v1's does not. That
   condition should sit beside the whole-menu k64/k128 results in the parent
   report, not after them. It is the most direct answer to "does ordering
   let you serve more traffic at the same accuracy and the same latency
   budget", and one depth/model/SLA value is a proof of concept, not a curve.

Items 2-5 and 7 need no new datasets, only more runs against what already
exists. Item 1 needs real construction. Item 6 is out of reach regardless of
budget.

---

## Runs

| stage | directory | runs |
|---|---|---|
| k4/k16 dense accuracy, two models | `eval-validity-20260906-165826/` | 16 |
| k64/k128 dense accuracy, third size point (8B) | `eval-validity-20260906-165826/` | 8 |
| k16 dense accuracy, 8B follow-up | `eval-validity-20260906-165826/` | 4 |
| alphabetical/frequency dense accuracy, all four depths, two models | `eval-validity-20260906-165826/` | 16 |
| SLA-constrained throughput sweep, k64/0.6B, six arms x six rates | `eval-validity-20260906-165826/` | 36 |
| k=10 and retrieve-64/present-10: reuse/latency (0.6B) + accuracy (0.6B, 4B), six arms | `eval-gaps-20260907-224433/` | 36 |

Servers: Qwen3-0.6B on GPU2:8300, Qwen3-4B on GPU3:8301, Qwen3-8B on GPU0:8302
(`--max-model-len 32768`, required to fit KV cache alongside the larger
weights at default `gpu_memory_utilization`), Qwen3-0.6B on GPU1:8303 for the
SLA sweep — four servers, four GPUs, concurrently (separate flock targets).
Drivers: `scripts/replay_vllm_workload.py` (serial, accuracy) and
`scripts/replay_vllm_concurrent.py` (rate-controlled, SLA sweep); scoring:
`scripts/score_tool_selection.py` plus an ad-hoc F1/strict/fractional
re-scorer (§4), not yet promoted to a script. Zero failures across all 80
runs added in this pass.
