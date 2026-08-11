# Trie-Aware Tool Ordering for Prefix-Cached LLM Serving

Consolidated report: mechanism, evidence, external comparison, and findings.

All cache and latency figures come from the serving engine's own counters. The
planner's analytical estimates are never used to support a cache claim. Every
numeric claim carries a **Source** pointer to the tracked report or artifact it
came from; see Appendix A for the full traceability map.

**Status of this document.** This is a hand-written synthesis of the tracked
reports, written to put the mechanism and the external comparison in one place.
The generated reports remain authoritative: where any figure here disagrees with
`reports/initial-findings.md`, `reports/initial-brief-closure/findings.md`,
`reports/tooltrie-phase2/findings.md`, or
`reports/contextpilot-dual-model/findings.md`, those win.

---

## 1. Summary

Large tool catalogs make prompts expensive. When an agent exposes 64 or 128 tool
schemas per request, tool definitions dominate the prompt, and every request
re-prefills them. Automatic prefix caching (APC) can eliminate that cost, but
only for an **exact** shared token prefix — so the order in which tools are
serialized determines how much is reusable.

This project built a causal trie-aware ordering planner (ToolTrie-v0), measured
it against six static and fitted orderings plus two external systems on two
serving engines, and audited the result at the level of server-rendered tokens
and cache blocks.

**The headline is a positive mechanism with a negative magnitude.** Ordering
does create exact prefix reuse. In the initial-brief matrix, ToolTrie-v0 led the
six static/fitted orderings at every BM25 menu size, but only by **+0.76 to
+1.65 percentage points** over ordinary text prefill and with no established
latency gain, against **+49 points** on padded menus. Later causal ContextPilot
and online-frequency arms show that ToolTrie is not the best measured ordering
in general. The large padded figure is a property of the workload, not of the
planner.

Two further results constrain any claim made from this work. The historical
**ContextPilot-derived static-refit causal adaptation leads ToolTrie by about 9
points on padded-menu reuse**, and a corrected persistent-API adaptation at
`alpha=0.001` also leads ToolTrie on every BM25-retrieved menu size. Neither is
the full ContextPilot system: both omit annotations and eviction feedback. The
persistent arm's reuse falls from 95–96% on padded menus to 1.99–18.72% on
retrieved menus, confirming that workload overlap—not the planner label—drives
the large headline.

**The primary model is Qwen3-4B.** The supervisor-requested dual-model protocol
designates Qwen3-4B primary and Qwen3-0.6B a replication; 190 GPU replays were
accepted with all 33 audit checks passing. The headline numbers, all from that
accepted matrix except where noted:

Prefix-cache reuse, 200 requests x 3 trials per cell:

| Qwen3-4B (native capacity 96,832) | BFCL padded-64 | BM25 k=128 |
| --- | ---: | ---: |
| Original text prefill | 1.19% | 0.34% |
| Alphabetical | 37.99% | 0.44% |
| ToolTrie-v0 | 87.19% | 0.89% |
| ContextPilot persistent API | **96.16%** | 1.35% |
| ContextPilot static refit | **96.16%** | **2.23%** |
| `frequency_online` | 96.27% † | 2.33% † |

Function-calling accuracy, a separate n=800 BFCL replay per condition:

| Qwen3-4B | full call (640 cases) | no-tool (160 cases) |
| --- | ---: | ---: |
| Original text prefill | 76.09% | **88.12%** |
| Alphabetical | 73.28% | 85.62% |
| ToolTrie-v0 | 75.31% | 87.50% |
| ContextPilot persistent API | **77.03%** | 85.00% |
| ContextPilot static refit | **77.03%** | 85.00% |
| `frequency_online` | 76.88% | 83.75% |

† `frequency_online` systems figures come from a separate run at capacity
101,120; a same-session ContextPilot control reproduced the accepted values to
within 0.01 points, so the columns are comparable.

### What the percentages mean

**Reuse %** — of all the prompt tokens the server had to process during one
replay, the fraction it served from the KV cache instead of recomputing:

```
reuse = prompt_tokens_cached / (prompt_tokens_cached + request_prefill_kv_computed_tokens_sum)
```

Both terms are Prometheus counter deltas read from vLLM across the replay, so
this is the engine's own accounting, not an estimate. 0% means every token was
prefilled from scratch; 96% means only 4% of the work was actually done. Three
things it is **not**:

- **Not a latency saving.** It is prefill work avoided. Whether that converts
  into faster time-to-first-token is a separate question this report does not
  claim to have settled.
- **Not token-exact.** vLLM matches whole 16-token blocks, so a prefix shorter
  than a block boundary earns nothing.
- **Not a per-request average.** It is one ratio over the whole 200-request
  replay, which is why the first request — necessarily cold — is included.

Because it depends only on the token sequence and the cache capacity, it is
deterministic: every cell in this report has zero spread across its three
trials.

Where SGLang figures appear, the denominator is built differently — SGLang
reports cached tokens directly and the computed term is derived as
`prompt_tokens - cached_tokens` (`src/tatm/replay_summary.py`). SGLang also
matches at token rather than block granularity, so its reuse reads slightly
higher than vLLM's for the same ordering. Cross-engine values are therefore
directionally comparable but not interchangeable.

**Quality %** — one n=800 BFCL replay per condition, split 640 relevance cases
and 160 irrelevance cases:

- **function-name** — of the 640 relevance cases, the fraction where the model
  called the correct function name(s). Ignores arguments.
- **full call** — of the same 640, the fraction where name *and* arguments both
  matched ground truth. Strictly harder than function-name, so always lower.
- **no-tool** — of the 160 irrelevance cases, the fraction where the model
  correctly declined to call any tool. A model that calls something when it
  should stay silent loses points here, so this column is a safety measure
  rather than a capability one.

These use the repository's reduced BFCL-style AST checker, not the official BFCL
leaderboard evaluator, so they are comparable **within** this report and not
against published BFCL scores.

**Points versus percent.** Differences are quoted in *percentage points*: 77.03%
against 76.09% is "+0.94 points", not "+1.2%".

Three things follow. Reuse collapses from 96% to under 3% between a padded menu
and a realistically retrieved one, for **every** policy. On the padded control
`frequency_online` and both ContextPilot arms are tied at the structural
ceiling, so that benchmark cannot separate a counter from a clusterer. And no
reordering improves on ordinary text prefill by more than about one point of
full-call accuracy, while all of them lose no-tool accuracy.

**Audit note (updated 2026-08-09).** Sequence-dependent planner intervals in the
historical reports describe their fixed request order; they do not include
uncertainty over alternative request sequences. The replacement SGLang audit
now accepts all 72 historical raw runs against the independent aggregate
cached-token counter; this validates counter cleanliness, not condition
distinctness. Tables retain the cross-engine comparability limits below.
The generalized ContextPilot static-refit adapter's integer-ID restoration was
fixed as well; the historical Phase 2 builder already used an equivalent inverse
mapping, so that repair does not alter the old emitted workload. A fresh,
predeclared 190-replay experiment has now measured both corrected ContextPilot
adaptations on Qwen3-4B and Qwen3-0.6B. Quality intervals for all stateful
planners remain fixed-sequence descriptions. In particular, the persistent
arm's historical 8B no-tool difference is −5.00 points, while its fresh 4B
difference from alphabetical is −0.62 points with an interval spanning zero; a
universal safety penalty is not supported.

The historical 8B static-refit cell and SGLang 72/72 raw-counter audit are now
integrated as compact evidence. The 8B static and persistent APIs have identical
aggregate and paired statistics, but that historical quality matrix still lacks
an ordinary `original` fallback arm. Their raw archives remain server-only.

---

## 2. How ToolTrie-v0 works

**Source:** `src/tatm/tooltrie.py` (283 lines). This section describes code, not
measurements.

ToolTrie-v0 is a **prompt-layer ordering planner**. Given the tool set that
retrieval has already selected for a request, it decides the order in which
those tools are serialized into the prompt, so that the request reuses as much
of the engine's existing prefix cache as possible. It never adds, removes, or
substitutes a tool, and it never touches KV tensors, attention, or any engine
internal.

### 2.1 Data structure

The planner maintains a trie whose **edges are tool IDs** and whose root-to-node
paths are tool sequences that have actually been served. Each node carries the
tool's rendered `schema_tokens`, the `cumulative_schema_tokens` of the path
reaching it, a `visit_count`, and `last_seen` (the index of the most recent
request that traversed it).

In v0, `visit_count` is incremented during `observe()` but never read by
`plan()`. Popularity-aware path convergence is therefore a ToolTrie-v1
hypothesis, not a mechanism present in the reported v0 results.

Nodes are deliberately **planner metadata, not physical KV blocks**. The trie
models which tool sequences the engine has recently seen; the engine's measured
APC counters remain the sole authority on whether rendered token blocks were
actually reused.

### 2.2 Causality: the `plan` / `observe` split

Two methods with a strict ordering contract:

- `plan(selected_ids)` returns an ordering using **only** paths recorded by
  earlier requests. It never mutates state.
- `observe(ordered_ids)` inserts a sequence into the trie and is called **only
  after** that request has been served.

This makes the planner deployable: a live server cannot see future requests, so
an ordering policy that requires them is not a candidate.

It is also load-bearing rather than a handicap. A tested offline variant, which
grants the planner the whole evaluation batch and iterates to a fixpoint,
**collapses from 87.19% to 29.96% reuse** — below plain alphabetical ordering.
Early requests establish a path and later ones follow it; treating the batch
atemporally destroys exactly that self-reinforcement. "Offline" is therefore not
an upper bound for this planner, it is a different and worse algorithm.

**Source:** `reports/tooltrie-phase2/findings.md` §2a.

### 2.3 The planning algorithm

`plan` performs a greedy descent through the existing trie:

```python
while remaining:
    candidates = [child for tool_id, child in node.children.items()
                  if tool_id in remaining and self._is_resident_hint(child)]
    if not candidates:
        break
    chosen = min(candidates, key=lambda c: (
        -self._reachable_cached_cost(c, frozen_remaining),  # longest reusable path
        -self.support.get(c.tool_id, 0),                    # training support
        c.tool_id))                                         # lexicographic tie-break
    matched.append(chosen.tool_id)
    remaining.remove(chosen.tool_id)
    node = chosen

ordered_ids = (*matched, *self._fallback_order(remaining))
```

It walks as deep into the trie as it can, but only through nodes that are both
present in this request's selected set and plausibly still cached. Two details
carry the method:

**Path-level lookahead.** `_reachable_cached_cost` recursively returns the
largest total schema-token path reachable *through* a candidate child using only
tools still remaining for this request. The planner selects the branch that
maximises reuse over the whole remaining path, not the locally cheapest next
tool.

**Conservative residency hint.** `_is_resident_hint` accepts a node only if it is
live and `request_index - last_seen < recency_window` (default 128). This is an
intentionally cheap approximation of whether the engine still holds the
corresponding blocks. The planner does not query the cache and does not claim to
know; the measured counters adjudicate afterwards.

### 2.4 Fallback

The default fallback is alphabetical — ordinary deterministic text prefill with
no special machinery, and the conservative text path required by the brief. It
is not guaranteed to preserve model behavior relative to another ordering. A
`frequency` fallback exists but is **refused at construction time** unless given
support statistics fitted on a separate training workload, encoding the rule
that benchmark frequency must never be read as production popularity.

### 2.5 Permutation guarantee

```python
assert set(ordered_ids) == set(ids)
assert len(ordered_ids) == len(ids)
```

The output is strictly a permutation of the retrieved set. Tool membership and
schema content are unchanged, so ordering is the only intended intervention;
model behavior can still change because tool position changes. This is verified
independently: within each menu size, all seven conditions carry byte-identical
sorted tool sets per case.

**Source:** `reports/initial-brief-closure/20260805-222246-gpu-executor/HANDOVER.md`.

### 2.6 Metadata budget

`capacity_tokens` and `max_nodes` (default 100,000) bound the trie, with
leaf-first LRU eviction. This budgets **the planner's own retained metadata** —
how much served history it remembers — and is **not** KV-cache retention.
Keeping a cached tool the current request does not require is a separate,
unbuilt extension.

### 2.7 Mechanism, and where it stops working

The planner re-derives the shared catalog order from requests it has already
served and pushes each request's novel tools to the tail through the fallback,
so the varying part of a menu stops breaking the common prefix. This is the
inverse of the frequency-ordering failure, where placing request-specific tools
early destroys the shared prefix.

The mechanism depends entirely on consecutive requests sharing tools. When they
do not, `candidates` is empty on the first iteration and `plan` degenerates to
plain alphabetical ordering. Section 4.2 shows this is the normal case on
independently retrieved menus.

---

## 3. Evidence base

**Source:** `reports/initial-brief-closure/findings.md`; GPU handovers in
`reports/initial-brief-closure/20260805-222246-gpu-executor/` and
`reports/initial-brief-pressure-rerun/20260807-005414/`.

Measured on an isolated RTX 3090, vLLM 0.26.0, temperature 0, seed 0, thinking
disabled. Live APC capacity was read back from the server (190,896 tokens =
16 tokens/block x 11,931 blocks) rather than assumed.

**Multiple models, by design.** The original systems series — reuse, prefill,
TTFT, and cache pressure — uses `Qwen/Qwen3-0.6B` for within-series
comparability, while the original correctness and safety series uses
`Qwen/Qwen3-8B`. The fresh replication instead makes Qwen3-4B primary and
reports Qwen3-0.6B separately. Ordering-quality effects are model-sensitive;
the evidence does not establish a general model-size law. Section 4.8 documents
the sign changes and their inference limits.

| Stage | Declared | Produced | Status |
| --- | ---: | ---: | --- |
| Retrieved-menu workloads | 4 menu sizes x 7 conditions = 28 | 28 | complete |
| Primary replays | 4 x 7 x 3 trials = 84 | 84 | complete, all counters clean |
| Paired menu-size summaries | 4 | 4 | complete |
| Rendered-token/block audits (k=64) | 7 | 7 | all validations pass |
| Controlled memory-pressure runs | 6 orderings x 4 regimes = 24 | 24 | 24/24 accepted, 384/384 checks |

Every replay carries 200 requests, a verified cache reset, and the per-request
identity `cached + computed == prompt_tokens`. That identity is also a
concurrency detector: it fails whenever a second client shares the server, which
is how contaminated runs were identified and quarantined rather than reported.

Raw per-request records and token IDs are preserved in checksummed archives
outside version control (Appendix A).

---

## 4. Findings

### 4.1 Retrieval is a separate bottleneck

**Source:** `reports/retrieval-bm25-sweep.json`;
`reports/initial-brief-closure/findings.md` §1.

Tools were selected by a deterministic BM25 baseline over canonical tool
definitions, using only fields available before evaluation. Gold labels were
used afterwards solely to score retrieval.

| Retrieved tools | Macro recall | Any-gold hit rate | MRR |
| ---: | ---: | ---: | ---: |
| 4 | 41.71% | 51.50% | 0.3858 |
| 16 | 55.04% | 65.50% | 0.4022 |
| 64 | 64.21% | 75.50% | 0.4061 |
| 128 | 67.54% | 80.50% | 0.4066 |

Even at k=128, **19.5% of queries retrieve none of their gold tools**. No
ordering or caching method can recover a tool that was never retrieved. This is
a reproducible lexical baseline, not the official ToolRet retriever; they are
not a proven lower or upper bound for stronger retrieval systems.

**Why sparse retrieval.** BM25 is deterministic, has no fitted parameters, and
reads only fields available before evaluation, so menu membership is
byte-reproducible and cannot leak gold labels into selection. The retriever is
not the object of study; it supplies a deterministic, reproducible menu.

### 4.2 Reuse on BM25-retrieved menus — the central result

**Source:** `reports/initial-brief-closure/findings.md` §2;
`retrieved-k{4,16,64,128}-summary.json` in the closure handover.

Every condition received the same selected tool set per query; only order
changed. `original` is the ordinary text-prefill fallback in BM25 rank order.

| k | Original | Alphabetical | Random(42) | Frequency | Schema-cost | FP-tree | **ToolTrie-v0** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 15.87% | 15.28% | 14.17% | 14.62% | 14.84% | 14.62% | **17.48%** |
| 16 | 6.12% | 6.27% | 5.39% | 5.59% | 6.24% | 5.59% | **7.77%** |
| 64 | 0.91% | 1.24% | 0.96% | 0.96% | 1.09% | 0.96% | **1.90%** |
| 128 | 0.37% | 0.58% | 0.59% | 0.54% | 0.57% | 0.54% | **1.13%** |

ToolTrie-v0 leads at all four sizes — a positive ordering result — but the gain
over ordinary text prefill is **+1.61, +1.65, +0.99, and +0.76 percentage
points**. Independently retrieved menus do not share enough exact rendered
prefix to reproduce the much larger reuse seen on padded shared-catalog menus.

**This is the finding that most constrains the work.** The same planner, the
same code, and the same engine produce 87.19% on padded menus and 1.90% at
k=64 on retrieved menus. The difference is the workload.

### 4.3 Prompt cost and TTFT

**Source:** `reports/initial-brief-closure/findings.md` §3.

| k | Prompt tokens/query | Fallback reuse | ToolTrie reuse | Fallback TTFT | ToolTrie TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 640 | 15.87% | 17.48% | 32.3 ms | 31.2 ms |
| 16 | 2,139 | 6.12% | 7.77% | 70.7 ms | 70.3 ms |
| 64 | 8,361 | 0.91% | 1.90% | 349.7 ms | 350.0 ms |
| 128 | 16,267 | 0.37% | 1.13% | 937.0 ms | 933.5 ms |

Prompt cost grows about 25x from k=4 to k=128 while macro recall rises only
25.83 points. The three-trial TTFT intervals overlap and no paired-difference
test was predeclared, so **these results do not establish that ToolTrie makes
retrieved-menu serving faster.**

At k=64 both direct partial-reuse conditions contained 199 genuinely partial
requests and one cold request per trial (alphabetical 1.37% partial cached ratio
at 351.7 ms; ToolTrie 2.12% at 350.5 ms). The stratum exists and was measured
directly; the latency difference is too small to support a speed claim.

### 4.4 Exact rendered-token and block audit

**Source:** `audit-k64-validation-summary.json` in the closure handover;
`reports/initial-brief-closure/findings.md` §4.

All seven k=64 audits passed every declared validation: cache reset before
measurement; server `/tokenize` count equals completion prompt usage; cached
plus computed tokens equal rendered prompt tokens per request; block size is the
live server value of 16. This closes the rendered-token evidence gap without
treating analytical schema-token estimates as server tokenization.

### 4.5 External comparison — what this was measured against

**Source:** `reports/tooltrie-phase2/findings.md` §2, §2a, §5, §7;
`reports/contextpilot-confirmation/findings.md`;
`reports/contextpilot-dual-model/findings.md`.

Measured on vLLM, with the same ranking observed provisionally on a second
engine. **Padded shared-catalog menus**, Qwen3-0.6B, 3 trials.

| Condition | vLLM cached (BFCL) | SGLang cached (BFCL) |
| --- | ---: | ---: |
| ContextPilot, offline | 96.67% | 96.86% |
| **ContextPilot static-refit causal adaptation (alpha=0.5)** | **96.18%** | **96.38%** |
| ToolTrie-v0 | 87.11% | 87.29% |
| Fitted policies | 39.69% | 39.58% |
| Alphabetical | 38.13% | 38.02% |
| CacheWeaver / original | 1.19% | 1.19% |

| ToolRet | vLLM cached | SGLang cached |
| --- | ---: | ---: |
| ContextPilot static-refit causal adaptation (alpha=0.5) | 94.86% | 95.09% |
| ToolTrie-v0 | 83.55% | 83.74% |
| Alphabetical | 51.06% | 49.94% |
| CacheWeaver | 22.46% | 17.75% |
| Original | 13.88% | 11.29% |

The later persistent-API confirmation used stock vLLM, the same first 200
requests and selected sets, upstream commit `1fa0a143`, and `alpha=0.001`:

| Workload | Original | Alphabetical | ToolTrie-v0 | ContextPilot persistent API |
| --- | ---: | ---: | ---: | ---: |
| BFCL padded 64 | 1.19% | 38.13% | 87.19% | **96.16%** |
| ToolRet padded 64 | 13.87% | 51.05% | 83.58% | **95.27%** |
| ToolRet BM25 k=4 | 15.87% | 15.28% | 17.48% | **18.72%** |
| ToolRet BM25 k=16 | 6.12% | 6.27% | 7.77% | **9.93%** |
| ToolRet BM25 k=64 | 0.91% | 1.24% | 1.90% | **4.78%** |
| ToolRet BM25 k=128 | 0.37% | 0.58% | 1.13% | **1.99%** |

The fresh dual-model replication then ran both corrected ContextPilot
adaptations in one predeclared matrix. Qwen3-4B is primary:

| Qwen3-4B workload | Original | Alphabetical | ToolTrie-v0 | CP persistent API | CP static refit |
| --- | ---: | ---: | ---: | ---: | ---: |
| BFCL padded 64 | 1.19% | 37.99% | 87.19% | **96.16%** | **96.16%** |
| ToolRet padded 64 | 6.56% | 43.81% | 82.18% | 95.27% | **95.74%** |
| ToolRet BM25 k=4 | 15.87% | 15.28% | 17.48% | **18.72%** | 18.42% |
| ToolRet BM25 k=16 | 5.14% | 5.40% | 6.73% | 9.10% | **9.36%** |
| ToolRet BM25 k=64 | 0.72% | 0.98% | 1.63% | 3.07% | **3.48%** |
| ToolRet BM25 k=128 | 0.34% | 0.44% | 0.89% | 1.35% | **2.23%** |

Qwen3-0.6B separately replicates the ordering result: both ContextPilot arms
beat ToolTrie-v0 in all six cells. Across the two model arms, that is 12/12
systems cells. The persistent and static-refit variants are close and neither
dominates the other.

Three conclusions:

1. **Both ContextPilot-derived policies are stronger measured ordering
   competitors.** Restricting the historical static refit to already-served
   requests costs only 0.48 points on BFCL and 1.48 on ToolRet, so future-batch
   visibility is not the explanation. The persistent-API adaptation also leads
   ToolTrie at every retrieved menu size. It uses the official `reorder` API but
   remains ordering-only, without annotations or eviction feedback, so it is
   not a measurement of the full ContextPilot system.
2. **CacheWeaver did not lose — it did not apply.** Its 1.19% is identical to
   `original` because Algorithm 1 returned the unmodified input ordering on
   **200/200 BFCL requests** (180/200 on ToolRet), verified against the input
   files. It targets overlapping retrieved text evidence; tool menus drawn from a
   shared catalog do not present that structure. Any write-up must say this
   rather than claiming a win.
3. **The ordering ranking appears across engines and the corrected counter
   audit passes.** Cached ratios agree within about 0.2 points under vLLM's
   paged APC and SGLang's radix tree. The independent aggregate-counter audit
   accepts 72/72 historical raw runs. This establishes clean counter windows;
   it does not establish that every policy label is behaviorally distinct.

**Comparability limits.** The two engines use different chat templates
(`hermes` vs `qwen`), and SGLang's rendering costs exactly +640 tokens on every
request, so totals differ by about 9%. Cached *ratios* are the closest
cross-engine summary, but even they have different rendered denominators;
absolute tokens, prefill, and TTFT are not comparable. SGLang's higher absolute
TTFT must **not** be attributed to RadixAttention — the scheduler, attention
backend, and memory configuration all differ.

The retrieved-menu gap is now measured for both corrected `alpha=0.001`
adaptations in the fresh dual-model matrix. The historical static-refit arm at
`alpha=0.5` was not retroactively mixed into it. See §7.2.

### 4.6 Statistical ordering policies collapse onto one order

**Source:** `reports/tooltrie-phase2/findings.md` §2;
`reports/tooltrie-phase2/fitted-policy-equivalence.json`;
`reports/initial-brief-pressure-rerun/ordering-equivalence.json`.

Frequency, schema-cost weighting, FP-tree-derived, pair, and triple labels emit
one **byte-identical 200-request sequence** on BFCL, while alphabetical differs.
They are one tested behavior there, not five independent policies. On ToolRet,
frequency/pair/triple are identical while schema-cost and FP-tree each differ,
giving three fitted sequences. Under controlled pressure, frequency,
schema-cost weighted, and FP-tree global likewise emit byte-identical sequences
in all four regimes.

The interpretation is that these measured cache conditions reward **identical**
prefixes, not importance scores by themselves. In this workload, the fitted
importance/co-occurrence policies added little unless they caused requests to
emit the same prefix repeatedly.

The causal `frequency_online` ablation then isolates adaptivity. It reaches
96.27% on BFCL padded-64 because 63 of 64 tools occur in every request and an
unseen non-universal tool moves to the tail after the first observation. This
is a valid demonstration that the padded positive control cannot distinguish a
trie or clusterer from a simple counter. It is not a universal winner: it beats
ToolTrie-v0 in 10/12 cells and loses both BM25 k=4 cells (17.01% versus
17.48%). See `reports/frequency-online/findings.md`.

**The 0.11-point difference is not a meaningful separation.** A per-request
audit shows `frequency_online` caching 6,704 tokens — 419 blocks — on **195 of
198** warm requests, the remaining three caching slightly more. That plateau is
the block-aligned shared prefix: preamble plus the 63 universal schemas. Its
only two losses are request 0, cold for any causal policy, and request 1, where
the zero-information cold ordering diverges early. The gap to the persistent
ContextPilot arm is 1,536 tokens — 95 blocks across 200 requests.

An independent check supports this being close to what the workload allows: an
exact-token prefix trie over server-rendered prompts, simulated without
eviction, predicts 96.16% for the ContextPilot ordering against 96.16%
measured. Capacity is therefore not binding here, so each policy is achieving
what its own ordering permits. **Whether some better ordering exists on this
workload has not been established** — computing a true maximum over orderings
remains open.

One variable predicts the whole table: **how many tools per request are not
shared by all**. BFCL padded-64 has 63 universal tools and exactly one
non-shared tool per request, and counting ties clustering there. ToolRet
padded-64 has 60 universal and four non-shared, and clustering pulls ahead
(95.27% versus 94.80%) because the order *among* those four then matters. The
BM25 workloads have no universal set at all.

The frozen `frequency_fitted` result was re-audited to the same standard and is
sound: counters clean, request 0 cold in every trial, zero trial spread,
aggregate equal to the sum of per-request deltas, and total prompt tokens
identical across all conditions on the same menus. A leakage check is decisive
in the other direction — a fit contaminated by evaluation data would rank the
universal tools first and reach ~96%, whereas the fitted ranking places them
mid-menu (mean position 31.6 of 64 on BFCL) and on ToolRet promotes a
training-frequent tool to near-first, destroying the prefix immediately. Its
failure is itself evidence of its integrity.

### 4.7 Quality and safety — a real frontier

**Source:** `reports/tooltrie-phase2/findings.md` §1 and §4;
`reports/contextpilot-confirmation/findings.md`;
`reports/contextpilot-dual-model/findings.md`;
`reports/contextpilot-static-refit-resume/findings.md`.

Historical BFCL correctness, n=800 (160 per domain), Qwen3-8B:

| Condition | function-name | full call | no-tool |
| --- | ---: | ---: | ---: |
| Alphabetical | 82.81% | 76.41% | **89.38%** |
| ToolTrie-v0 | 83.75% | 77.66% | 87.50% |
| ContextPilot static-refit causal adaptation (alpha=0.5) | **84.84%** | **79.06%** | 81.88% |

The historical persistent-API confirmation and the fresh dual-model run use
the same 800-case quality design:

| Model | Condition | function-name | full call | no-tool |
| --- | --- | ---: | ---: | ---: |
| Qwen3-8B | Alphabetical | 82.81% | 76.41% | **89.38%** |
| Qwen3-8B | ToolTrie-v0 | 83.75% | 77.66% | 87.50% |
| Qwen3-8B | ContextPilot persistent API | **84.84%** | **78.44%** | 84.38% |
| Qwen3-8B | ContextPilot static refit | **84.84%** | **78.44%** | 84.38% |
| Qwen3-4B | Original | 83.13% | 76.09% | **88.12%** |
| Qwen3-4B | Alphabetical | 82.19% | 73.28% | 85.62% |
| Qwen3-4B | ToolTrie-v0 | 83.75% | 75.31% | 87.50% |
| Qwen3-4B | ContextPilot persistent API | **84.38%** | **77.03%** | 85.00% |
| Qwen3-4B | ContextPilot static refit | **84.38%** | **77.03%** | 85.00% |
| Qwen3-4B | `frequency_online` | **84.38%** | 76.88% | 83.75% |

The Qwen3-0.6B replication, same 800-case design:

| Model | Condition | function-name | full call | no-tool |
| --- | --- | ---: | ---: | ---: |
| Qwen3-0.6B | Original | **73.75%** | **55.16%** | 86.25% |
| Qwen3-0.6B | Alphabetical | 60.47% | 43.59% | **94.37%** |
| Qwen3-0.6B | ToolTrie-v0 | 69.06% | 53.28% | 93.13% |
| Qwen3-0.6B | ContextPilot persistent API | 68.91% | 51.72% | 91.25% |
| Qwen3-0.6B | ContextPilot static refit | 68.91% | 51.88% | 91.25% |
| Qwen3-0.6B | `frequency_online` | 69.84% | 53.28% | 90.62% |

`frequency_online` matches both ContextPilot arms on 4B function-name accuracy
exactly and sits 0.15 points below them on full-call accuracy, a difference of
one scored case in 640. Having matched them on reuse it also matches them on
quality, with no clustering, no trie and no training corpus. At 0.6B it is the
least damaging reordering on relevance, though every reordering there is worse
than original order.

The fresh 4B relevance-side gain over alphabetical reproduces the earlier 4B
addendum, but no-tool accuracy is highest under original order. ContextPilot
minus alphabetical is −5.00 no-tool points in the historical 8B run and −0.62
points in the fresh 4B run, whose fixed-sequence interval spans zero. Ranking
by function-name accuracy and no-tool accuracy is negatively associated in the
historical 8B matrix, but the current evidence does **not** show a universal
causal trade-off.

A per-case audit against `original` at 4B sharpens what the no-tool column is
showing. Of 160 irrelevance cases, `original` is correct on 141, and the
reordering policies differ from it as follows:

| Policy | cases lost | cases recovered | net |
| --- | ---: | ---: | ---: |
| ToolTrie-v0 | 5 | **4** | −1 |
| ContextPilot persistent API | 5 | **0** | −5 |
| `frequency_online` | 7 | **0** | −7 |

ToolTrie-v0 is bidirectional, which is what noise looks like. The two
high-reuse policies recover **zero** cases: exact McNemar gives p ≈ 0.016 for
`frequency_online` and p ≈ 0.063 for the persistent API. So the defensible
statement is that the high-reuse policies never repair an irrelevance case and
only break them — **not** that degradation scales with reuse, which 1, 5 and 7
cases out of 160 cannot establish. No equivalence margin was declared for any of
these comparisons, so they remain estimation only.

The two 8B ContextPilot rows have identical aggregate and paired statistics,
but the 8B matrix has no `original` row. Because alphabetical is a harmful
quality baseline at 4B and 0.6B, these 8B gains cannot be interpreted as gains
over ordinary selected-tool text prefill.

The 0.6B replication is qualitatively different: original order has the best
relevance accuracy, while alphabetical has the best no-tool accuracy.
Cross-model differences are therefore model-sensitive observations, not a pure
model-size effect. Native capacities differ, ToolTrie is capacity-dependent,
and only one checkpoint was tested at each size.

All these quality values use the repository's reduced BFCL-style AST checker,
not the official BFCL leaderboard evaluator. They also come from one complete
replay per condition. The earlier and fresh 4B runs differ by at most one
scored case per shared condition and metric, so complete sequence replicates or
multiple inference seeds remain necessary for publication-grade uncertainty.

The targeted analysis is better powered and more damaging. All 240 BFCL
irrelevance tasks under 5 fixed menu seeds, bootstrap clustered on tasks:

| | value |
| --- | --- |
| Alphabetical | 87.50% |
| ToolTrie-v0 | 83.75% |
| Difference | **-3.75 points** |
| Task-clustered 95% CI | **[-5.67, -2.00]**, excludes zero |
| Discordant pairs | alphabetical-only correct **54**, ToolTrie-only correct **9** |

**This is a resolved difference for the fixed emitted sequence, not a null
result.** The earlier n=100 no-tool difference was 0.00 points, while the nested
n=1000 run produced a -4.00-point estimate. These are not three independent
measurements. Because ToolTrie ordering depends on preceding requests, the task-
clustered interval above does not cover alternative request sequences. Plain
alphabetical remains the strongest condition in this targeted Qwen3-8B
ToolTrie-versus-alphabetical workload; that statement does not generalize to
the separate Qwen3-4B addendum.

Counting the historical padded-sequence axes honestly: the static-refit adapter beats
ToolTrie-v0 on reuse, function-name accuracy, and full-call accuracy. ToolTrie-v0
wins one axis, no-tool safety, by 5.62 points. These quality scores apply to the
emitted ordering, not full annotated ContextPilot. **No combined utility
weighting has been declared, so no overall winner is claimed** — and any such
weighting must be declared before looking at these numbers.

For the persistent-API adaptation, ToolTrie likewise loses every measured reuse
cell and the relevance-side point estimates. Its no-tool advantage is 3.12
points at 8B and 2.50 points in the fresh 4B run, with the 4B fixed-sequence interval spanning
zero. This reinforces the need for a predeclared utility rather than a single
aggregate “score.”

### 4.8 Model scale: why two models, and what changed with size

**Source:** `PROJECT_STATUS.md`, "ToolTrie-v0" section;
`reports/tooltrie-phase2/findings.md` §4;
`reports/contextpilot-dual-model/findings.md`.

On the same 100 stratified BFCL tasks, ToolTrie minus alphabetical was **+7.50
points** on function-name accuracy at 0.6B but **-3.75 points** at 8B — the
effect inverted with model size. Reporting the 0.6B quality numbers alone would
have produced a confident and wrong conclusion.

The ordering/capacity explanation was ruled out for that specific comparison:
the 8B planner ran at 44,656-token capacity versus 190,896 at 0.6B and evicted
424 trie nodes, yet produced **byte-identical orderings on all 100 records**.
That makes the observation model-dependent, but not a general scaling law: only
one checkpoint was tested at each size.

**The larger nested sample reversed the 8B point estimates.** Repeated at 200 per domain
(n=1,000), a nested superset of the 100 above:

| ToolTrie - alphabetical, Qwen3-8B | n=100 | n=1,000 | 95% CI at n=1,000 |
| --- | ---: | ---: | --- |
| function-name | -3.75 | **+0.75** | -3.03 … +4.53 |
| full call | -6.25 | **+1.50** | -2.66 … +5.66 |
| no-tool | +0.00 | -4.00 | -10.89 … +2.89 |

Both headline metrics reverse sign and zero sits inside every interval, so **no
quality cost is resolved in this fixed 8B replay**. Two methodological lessons
follow:

1. **A 100-task BFCL sample got the *sign* wrong, not merely the magnitude.**
   Results at that size are pilots, not evidence.
2. **The small sample was optimistic as well as noisy.** Absolute accuracy fell
   for every condition on the larger set (alphabetical full call 81.25% to
   75.62%), so the first 20 tasks per domain were easier than the remaining 180.

Note this does **not** contradict §4.7: that regression comes from the targeted
240-task irrelevance protocol, which is better powered on the no-tool metric
specifically than this mixed-set sweep.

A consequence for the predeclared gate: the 95% CI on the full-accuracy
difference is about +/-4.2 points at n=1,000, so resolving a 1-point threshold
is much wider than a 1-point equivalence margin. A crude independent-proportions
calculation suggests a much larger sample, but paired precision depends on
discordance and ToolTrie treatment depends on request history. **This design and
budget do not practically resolve the 1-point gate**; it should be restated as
an equivalence test with a declared margin and predeclared sequence replicates.

The fresh 4B-primary/0.6B-replication matrix reinforces the same caution. The
quality spread across orderings is much larger at 0.6B, but native capacities
differ, ToolTrie is rebuilt per model, and quality has one complete replay per
condition. It is accurate to call the result model-sensitive; it is too strong
to call every 0.6B difference a model-size artifact.

### 4.9 Behaviour under controlled cache pressure

**Source:** `reports/initial-brief-closure/findings.md` §5;
`reports/initial-brief-pressure-rerun/20260807-005414/HANDOVER.md`;
`cluster/initial-brief-pressure-rerun-manifest.json`.

An initial pressure attempt failed its own acceptance criterion: peak KV
occupancy reached only 3.64-3.69% against a predeclared 90% threshold, because a
strictly sequential client holds at most one roughly 7,000-token request
resident in a 190,896-token cache. Cumulative prompt volume of 7.23x capacity is
throughput, not simultaneous residency. Those runs were preserved and
quarantined from pressure claims rather than reinterpreted, and the threshold
was not lowered after inspection.

A separately predeclared protocol changed **one** controlled variable — capping
the cache at 480 blocks (7,680 tokens) — while holding concurrency at one and
the threshold at 90%. All 24 regime-runs passed: peak occupancy 91.02-91.86%,
sampled evictions 57,696-85,340 at 100% metric sampling, zero preemptions, and
peak running/waiting requests of 1/0 confirming the sequential client. 384/384
validation checks passed.

| Ordering | Empirical | Uniform | Skewed | Session-bursty |
| --- | ---: | ---: | ---: | ---: |
| Original | 1.18% | 0.69% | 1.24% | 0.69% |
| Alphabetical | 29.21% | 29.35% | 28.06% | 27.76% |
| Random, seed 42 | 32.16% | 31.27% | 32.67% | 30.19% |
| Frequency = schema-cost = FP-tree | 9.44% | 8.99% | 9.51% | 9.00% |
| ToolTrie-v0 *(added 2026-08-11)* | 87.18% | 88.54% | 94.73% | 91.62% |
| ToolTrie-v1, visit-weighted *(added 2026-08-11)* | 87.18% | 88.54% | 94.73% | 91.62% |
| **Online frequency counter** *(added 2026-08-11)* | **96.27%** | **96.16%** | **96.40%** | **96.33%** |

All six original orderings are **static** permutations — `locality_replay.py`
held the intra-menu order constant across regimes by design, so the adaptive
ToolTrie-v0 could not be expressed in that matrix and was absent. It was added
later under the same capacity, regimes, seeds and gates: 4/4 accepted, 64/64
checks, peak occupancy 0.904–0.908, 506–1,494 planner evictions per regime —
the first time its own metadata-budget path has executed
(`reports/tooltrie-pressure/20260811-001032/`).

Among the static orderings, fixed random seed 42 leads every regime. This is
**one run per cell and one seed with no repeated-trial interval** — a
sensitivity result motivating a random-seed sweep, not a deployment
recommendation.

**ToolTrie-v0 leads all of them by 55–62 points, and loses nothing to the
smaller cache** — 87.18% here against 87.19% at 190,896 tokens. The mechanism is
prefix concentration: it places the single non-shared tool at mean position 57.3
of 64, so the shared core forms one ~6,950-token prefix that fits inside the
7,680-token capacity and stays resident, whereas alphabetical places it at 24.1
and thrashes the remainder. **This inverts the full-capacity ranking**, where
both ContextPilot arms and the online counter beat ToolTrie-v0.

**Both later additions revise that claim.** `frequency_online` was run in this
matrix and beats ToolTrie in every regime by up to 9.1 points, so the correct
statement is that *adaptive* policies win under scarcity, not that the trie
does. Its matrix validates only **2/4 regimes**: peak occupancy reached 0.89979
against the predeclared 0.90 gate on empirical and session-bursty. The threshold
was not lowered and the failing output is preserved. This is mechanically
informative rather than a fluke — peak occupancy *falls* as a policy
concentrates reuse better, since fewer distinct blocks stay resident, so a gate
that certifies pressure penalises the policies it exists to reward. At this
capacity nothing can hold 96% reuse and 90% occupancy simultaneously.

`tooltrie_v1`, which reads the `visit_count` v0 never consults in both selection
and eviction, matches v0 **to five decimal places in all four regimes**, with 0
of 200 emitted orderings differing. The weighting acts — v1 evicts 1,497 nodes
against v0's 1,494 on empirical — but `_reachable_cached_cost` decides almost
every choice, so the tie-break holding the weight is rarely reached.

ContextPilot remains the one policy never run under pressure. Reuse and latency here must not be
compared numerically with the 190,896-token cache, and this is not evidence of
natural production pressure.

---

## 5. Provenance and required labelling

**Source:** `reports/tooltrie-phase2/findings.md` §7.

Three comparison conditions are **not** upstream implementations and must be
labelled accordingly in any write-up:

| Condition | Implementation | Required label |
| --- | --- | --- |
| ToolTrie-v0 | this project's causal recent-path planner | `ToolTrie-v0` |
| CacheWeaver | faithful tool-ID transcription of the paper's Algorithm 1; no public implementation existed as of 2026-08-03 | **`CacheWeaver Algorithm-1 reimplementation`** |
| FP-tree conditional | training-only adaptation of FP-tree traversal | **`FP-tree-derived adaptation`**, not an FP-Growth result |
| Pair/triple conditional | training-only unordered co-occurrence statistics | **`pair/triple adaptation`** |
| ContextPilot offline | upstream code at commit `1fa0a143fdeda344585666648ab2b30cb7fea77f`, whole-batch `fit_transform` | `ContextPilot offline/transductive` |
| ContextPilot causal follow-up | our repeated-prefix `fit_transform` harness, historical `alpha=0.5`, no annotations or eviction feedback | `ContextPilot static-refit causal adaptation (alpha=0.5; ordering only)`, never official online ContextPilot |
| ContextPilot persistent confirmation | upstream persistent `ContextPilot.reorder` API at the same commit, `alpha=0.001`, unique conversation per benchmark case, shared causal index; no annotations or eviction feedback | `ContextPilot persistent-API adaptation (alpha=0.001; no eviction feedback or annotations)`, not the full system |
| SGLang | official `v0.5.15.post1`, commit `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd` | `SGLang/RadixAttention engine` |

---

## 6. Conclusion and fallback policy

The initial hypothesis is **refined, not rejected**. Exact tool ordering can
create prefix reuse, and a causal trie planner exploits it better than any
static or fitted ordering tested. But reuse alone is not sufficient: retrieval
coverage, menu overlap, prompt cost, no-tool safety, and cache capacity jointly
determine whether the optimization is worth applying.

**Operating rule.** Send the ordinary retrieved tool set through normal text
prefill whenever expected savings do not exceed context, decode, retrieval, and
safety costs. No inactive tools are retained and no KV tensors are composed in
that fallback.

Tasks A-F and the main planned experiment matrix are substantively complete;
experimental question 4 remains partial because measured per-schema prefill-time
weighting was not tested. The local analysis and traceability repairs are now in
place, and the 190-replay dual-model matrix is accepted for its declared scope.
Operational closure still requires off-machine backup of the raw archives. The
historical Qwen3-8B table lacks `original`, but its server command also omitted
a model revision: one appended row is controlled only if the exact historical
snapshot can be proved. Otherwise use a fresh pinned five-condition matrix or
omit that optional 8B comparison; the accepted 4B-primary table already has the
ordinary fallback. The static-refit and SGLang handovers are integrated.
Section 9 extensions — retained inactive tools, active-tool manifests,
admission policy, and KV composition — remain future work and are unbuilt.

---

## 7. Limitations and open questions

### 7.1 The retriever is a lexical baseline

BM25 is a reproducible baseline, not the official ToolRet retriever. A dense
retriever returns more semantically clustered menus, which could mean *more*
exact prefix overlap. Since the central negative result rests on retrieved menus
sharing little prefix, this is the highest-value robustness check available.

### 7.2 The external retrieved-menu result is ordering-only

Both corrected ContextPilot adaptations have now been run on BM25-retrieved
menus in the fresh dual-model matrix, and both lead ToolTrie in every model/menu
cell. This closes the earlier head-to-head gap while exposing a stronger
limitation: their absolute reuse is only 1.35–18.72% on 4B, far below the
95–96% padded positive control. Neither is the full ContextPilot system because
eviction feedback, annotations, de-duplication, and interleaved
planning/serving are absent. The historical 8B static-refit
closure is now tracked, but its matrix has no ordinary-original fallback arm.

### 7.3 The no-tool comparisons are model- and sequence-limited

The historical static-refit adapter's −7.50-point no-tool penalty and the
persistent arm's −5.00-point 8B penalty each come from 160 irrelevance cases on
one planner sequence. The fresh persistent difference is −0.62 points versus
alphabetical at 4B, with an interval spanning zero. The better-powered 240-task
protocol was run only for ToolTrie against alphabetical. Complete predeclared
request-sequence replicates, rather than resampling outcomes from one stateful
sequence, are needed for a population claim. The historical 8B ContextPilot
quality comparison also needs `original` before it can support a claim against
ordinary selected-tool text prefill.

### 7.4 No overall winner is claimed

No combined utility or cost weighting has been declared between reuse, call
accuracy, and decline safety. Any weighting must be declared before looking at
the measured numbers.

### 7.5 Latency claims are deliberately weak

No paired-difference interval was predeclared for TTFT on the retrieved arm, and
the predeclared 1-point quality gate is not practically resolved by the current
design and budget (§4.8). Latency across the two cache capacities in §4.9 is not
comparable at all.

### 7.6 Recommended next experiment

The active-tool manifest with constrained generation, because it targets the
fixed-sequence decline regression: separating the tools present in cached
context from the tools that may actually be called could allow reuse without the
same decline penalty.
It must be evaluated against the historical static-refit adapter, the corrected
persistent-API arm, `frequency_online`, and the ordinary text-prefill fallback.
It should include a random-seed sensitivity sweep rather than promoting seed 42
as an algorithm.

---

## Appendix A — Traceability

### Tracked reports

| Document | Contents |
| --- | --- |
| `reports/initial-findings.md` | Task F deliverable, generated by `src/tatm/reporting.py` |
| `reports/initial-brief-closure/findings.md` | Closure verdict, retrieved-menu arm, audits, pressure |
| `reports/tooltrie-phase2/findings.md` | External comparison, quality/safety, provenance table |
| `PROJECT_STATUS.md` | Hand-maintained status index |
| `reports/retrieval-bm25-sweep.json` | Retrieval curve (§4.1) |
| `reports/initial-brief-pressure-rerun/ordering-equivalence.json` | Sequence-equivalence audit (§4.6) |
| `reports/tooltrie-phase2/fitted-policy-equivalence.json` | Exact fitted-label equivalence audit (§4.6) |
| `reports/contextpilot-confirmation/findings.md` | Reconciled persistent-API result and limitations |
| `reports/contextpilot-confirmation/local-audit.json` | Compact-artifact invariant audit and explicit missing cells |
| `reports/contextpilot-dual-model/findings.md` | Reconciled 4B-primary/0.6B-replication result and limitations |
| `scripts/audit_contextpilot_dual_model_compact.py` | Reproducible 15-check audit of the tracked dual-model evidence |
| `reports/contextpilot-static-refit-resume/findings.md` | Static-refit closure and SGLang counter-audit interpretation |
| `reports/contextpilot-static-refit-resume/local-audit.json` | Reproducible six-check compact audit |
| `reports/frequency-online/findings.md` | Reconciled online-frequency ablation |
| `reports/frequency-online/local-audit.json` | Aggregate comparison audit, including the corrected 10/12 count |
| `reports/frequency-online/structure-audit.json` | Exact workload hashes and 63-tool-core reconstruction |

### GPU execution records

| Handover | Covers |
| --- | --- |
| `reports/initial-brief-closure/20260805-222246-gpu-executor/HANDOVER.md` | 28 workloads, 84 replays, 7 audits, failed pressure attempt |
| `reports/initial-brief-pressure-rerun/20260807-005414/HANDOVER.md` | 24/24 accepted controlled-pressure runs |
| `reports/contextpilot-confirmation/20260807-222212/HANDOVER.md` | 72 systems replays, three Qwen3-8B quality replays, persistent-API adaptation |
| `reports/contextpilot-quality-4b/20260808-105833/HANDOVER.md` | three-run Qwen3-4B quality addendum |
| `reports/contextpilot-dual-model/20260809-004603/HANDOVER.md` | accepted 190-replay dual-model matrix |
| `reports/contextpilot-static-refit-resume/20260808-234909/HANDOVER.md` | 18 static-refit systems replays, one 8B quality replay, 72-run SGLang audit |
| `reports/frequency-online/20260809-200701/HANDOVER.md` | causal online-frequency systems proposal |

### Predeclared protocols

- `cluster/initial-brief-closure-manifest.json`
- `cluster/initial-brief-pressure-rerun-manifest.json`
- `cluster/contextpilot-dual-model-manifest.json`

All were written before their corresponding execution. The pressure threshold was never lowered
after inspection; a separate protocol was declared instead.

The online-frequency `PREDECLARATION.md` is preserved separately. The executor
states that it was written before execution, but it was pushed in the same
commit as the implementation and results, so Git does not independently prove
that timing.

### Raw archives (outside version control)

| Archive | Size | SHA-256 |
| --- | ---: | --- |
| `initial-brief-closure-20260805-222246.tar.gz` | 127,635,469 B, 221 entries | `568dedaea859056a4c3d2cb4773a810364b40b03858efc83092c5d0804d4a7d5` |
| `initial-brief-pressure-rerun-20260807-005414.tar.gz` | 833,218 B, 36 entries | `d18c613958dbeadbb9114a7ec6c1418d1c062803075974940b470791e86a78e0` |
| `contextpilot-confirmation-20260807-222212.tar.gz` | 32 MB, 171 entries | `aa4e196d140dc8d478ea8fd8b9da069e460779b4c888647ba228a7a60ec4a6bd` |
| `contextpilot-quality-4b-20260808-105833.tar.gz` | 6.2 MB, 40 entries | `c8883e5a268765e0d00904b1936417094979999294a5f1e4af14a0e579c3dbc5` |
| `contextpilot-dual-model-20260809-004603.tar.gz` | 58 MB, 666 entries | `3fc5f5ec08580c22dcee65ed015fcb96ae6b36598aeaaaf3eeb8aeb8aac62012` |
| `contextpilot-static-refit-20260808-234909.tar.gz` | 12 MB, 79 entries | `8473faaee1d74dbb1559ad3e035a54b146193eb7f7dc961547adbe5f501f9b12` |
| `sglang-counter-audit-20260808-234909.tar.gz` | 2.8 MB, 74 entries | `2a8dfb9063d4306d52390fefbbaf506a5aa16e741349629f3205e7a53c7a2e14` |
| `frequency-online-20260809-200701.tar.gz` | 9.6 MB, 87 entries | `e6a774b158069a3742adb66584d35dc7053c443da8d5ef500f077f51e308caef` |

All eight listed archives currently have only one known copy on the GPU
server's physical disk and require an off-machine backup. The recent GPU
handoff called its list “five archives” but enumerated six; the table above
records each archive separately.

### Source code

- `src/tatm/tooltrie.py` — the planner (§2)
- `src/tatm/retrieval.py` — BM25 baseline (§4.1)
- `src/tatm/reporting.py` — report generation
