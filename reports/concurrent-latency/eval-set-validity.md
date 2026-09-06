# Is the eval set valid? Depth, retriever, and metric choices checked

`findings.md` and `README.md` report ToolTrie-v1's advantage at k = 4/16/64/128,
against ContextPilot, on BM25 and dense retrieval. This document checks whether
that evaluation design itself is defensible: is the depth range realistic, does
the advantage survive inside ContextPilot's own native range, what do real tool
catalogs actually look like, and is the accuracy metric fair to multi-tool
tasks. Every claim below is either a literature citation (dated, sourced) or a
run under `cluster/results/`.

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

<!-- ACCURACY_TABLE -->

<!-- ACCURACY_DISCUSSION -->

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

---

## 5. What a valid eval set for this problem would add

Synthesized from §§1-4 plus the standing limitations already in `findings.md`
A.6-A.7. Ordered by expected impact, not by effort.

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
5. **The strict/fractional accuracy columns from §4**, as standard reporting,
   not an audit.
6. **A genuine production trace remains open and likely unclosable here** —
   the brief's own §4.5 says no listed dataset provides one, and this
   project's own finding that arrival order moved reuse more than any policy
   did is exactly the kind of effect only a real trace would calibrate.

Items 2-5 need no new datasets, only more runs against what already exists.
Item 1 needs real construction. Item 6 is out of reach regardless of budget.

---

## Runs

| stage | directory | runs |
|---|---|---|
| k4/k16 dense accuracy, two models | `eval-validity-20260906-165826/` | 16 |

Server: Qwen3-0.6B on GPU2:8300, Qwen3-4B on GPU3:8301, both native capacity,
run concurrently (separate flock targets, 2 GPUs). Driver:
`scripts/replay_vllm_workload.py`; scoring: `scripts/score_tool_selection.py`
plus an ad-hoc strict/fractional re-scorer (§4), not yet promoted to a script.
