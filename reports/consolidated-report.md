# Trie-Aware Tool Ordering for Prefix-Cached LLM Serving

Consolidated report: mechanism, evidence, external comparison, and findings.

All cache and latency figures come from the serving engine's own counters. The
planner's analytical estimates are never used to support a cache claim. Every
numeric claim carries a **Source** pointer to the tracked report or artifact it
came from; see Appendix A for the full traceability map.

**Status of this document.** This is a hand-written synthesis of the tracked
reports, written to put the mechanism and the external comparison in one place.
The generated reports remain authoritative: where any figure here disagrees with
`reports/initial-findings.md`, `reports/initial-brief-closure/findings.md`, or
`reports/tooltrie-phase2/findings.md`, those win.

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
does create exact prefix reuse, and the trie planner leads every ordering
baseline at every menu size. But on realistically retrieved menus the margin is
**+0.76 to +1.65 percentage points** with no established latency gain, against
**+49 points** on padded menus drawn from a shared catalog. The large figure is a
property of the workload, not of the planner.

Two further results constrain any claim made from this work. **Causal
ContextPilot, an existing system, leads ToolTrie by about 9 points on reuse**,
and that lead survives placing it under an equal information regime. And
reuse-optimizing orders carry a **confirmed regression on correctly declining
irrelevant requests**.

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
no special machinery, and the quality-preserving path required by the brief. A
`frequency` fallback exists but is **refused at construction time** unless given
support statistics fitted on a separate training workload, encoding the rule
that benchmark frequency must never be read as production popularity.

### 2.5 Permutation guarantee

```python
assert set(ordered_ids) == set(ids)
assert len(ordered_ids) == len(ids)
```

The output is strictly a permutation of the retrieved set. Model semantics are
unchanged, so any measured difference between conditions is attributable to
ordering alone. This is verified independently: within each menu size, all seven
conditions carry byte-identical sorted tool sets per case.

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

**Two models, by design.** All systems measurements — reuse, prefill, TTFT, and
cache pressure — use `Qwen/Qwen3-0.6B`, keeping them comparable across the whole
measurement series; cache behaviour depends on the serving stack and prompt
tokens, not on model quality. All correctness and safety measurements use
`Qwen/Qwen3-8B`, because ordering-quality effects at 0.6B proved to be
small-model artefacts. Section 4.8 documents the difference, which inverts sign
between the two scales.

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
a reproducible lexical baseline, not the official ToolRet retriever, so these
are floor numbers.

**Why sparse retrieval.** BM25 is deterministic, has no fitted parameters, and
reads only fields available before evaluation, so menu membership is
byte-reproducible and cannot leak gold labels into selection. The retriever is
not the object of study; it only has to produce a realistic, reproducible menu.

### 4.2 Reuse on true retrieved menus — the central result

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

**Source:** `reports/tooltrie-phase2/findings.md` §2, §2a, §5, §7.

Measured on vLLM and replicated on a second engine. **Padded shared-catalog
menus**, Qwen3-0.6B, 3 trials.

| Condition | vLLM cached (BFCL) | SGLang cached (BFCL) |
| --- | ---: | ---: |
| ContextPilot, offline | 96.67% | 96.86% |
| **ContextPilot, causal** | **96.18%** | **96.38%** |
| ToolTrie-v0 | 87.11% | 87.29% |
| Fitted policies | 39.69% | 39.58% |
| Alphabetical | 38.13% | 38.02% |
| CacheWeaver / original | 1.19% | 1.19% |

| ToolRet | vLLM cached | SGLang cached |
| --- | ---: | ---: |
| ContextPilot, causal | 94.86% | 95.09% |
| ToolTrie-v0 | 83.55% | 83.74% |
| Alphabetical | 51.06% | 49.94% |
| CacheWeaver | 22.46% | 17.75% |
| Original | 13.88% | 11.29% |

Three conclusions:

1. **ContextPilot is the genuine competitor and it leads by about 9 points.** An
   earlier draft of the Phase 2 report dismissed this as an artifact of its
   offline regime. That was tested and is wrong: restricting it to already-served
   requests costs only 0.48 points on BFCL and 1.48 on ToolRet. The gap survives
   under an equal information regime.
2. **CacheWeaver did not lose — it did not apply.** Its 1.19% is identical to
   `original` because Algorithm 1 returned the unmodified input ordering on
   **200/200 BFCL requests** (180/200 on ToolRet), verified against the input
   files. It targets overlapping retrieved text evidence; tool menus drawn from a
   shared catalog do not present that structure. Any write-up must say this
   rather than claiming a win.
3. **The ordering effect replicates across engines.** Cached ratios agree within
   about 0.2 points under vLLM's paged APC and SGLang's radix tree, with
   identical condition ranking — evidence the mechanism belongs to the ordering,
   not to one cache implementation.

**Comparability limits.** The two engines use different chat templates
(`hermes` vs `qwen`), and SGLang's rendering costs exactly +640 tokens on every
request, so totals differ by about 9%. Only cached *ratios* are cross-engine
comparable; absolute tokens, prefill, and TTFT are not. SGLang's higher absolute
TTFT must **not** be attributed to RadixAttention — the scheduler, attention
backend, and memory configuration all differ.

**Not yet measured:** no external system has been run on the retrieved-menu arm.
See §7.2.

### 4.6 Statistical ordering policies collapse onto one order

**Source:** `reports/tooltrie-phase2/findings.md` §2;
`reports/initial-brief-pressure-rerun/ordering-equivalence.json`.

Frequency, schema-cost weighting, FP-tree-derived, pair, and triple policies
land within 0.01 points of one another (39.69% on BFCL). Under controlled
pressure, three of them were shown to emit **byte-identical tool sequences** for
all 200 requests in every regime, verified by SHA-256 sequence digests.

The interpretation is that the cache rewards **identical**, not **important**.
Modelling which tools matter adds nothing if the resulting order still varies per
request; what pays is emitting the same prefix repeatedly.

### 4.7 Quality and safety — a real frontier

**Source:** `reports/tooltrie-phase2/findings.md` §1 and §4.

BFCL correctness, n=800 (160 per domain), Qwen3-8B:

| Condition | function-name | full call | no-tool |
| --- | ---: | ---: | ---: |
| Alphabetical | 82.81% | 76.41% | **89.38%** |
| ToolTrie-v0 | 83.75% | 77.66% | 87.50% |
| ContextPilot, causal | **84.84%** | **79.06%** | 81.88% |

Ranking by function-name accuracy produces almost exactly the reverse ranking on
no-tool accuracy. **An ordering that makes the right tool easier to find also
makes the model likelier to call something when it should decline.** The effect
is systematic across all conditions, not specific to any one method.

The targeted analysis is better powered and more damaging. All 240 BFCL
irrelevance tasks under 5 fixed menu seeds, bootstrap clustered on tasks:

| | value |
| --- | --- |
| Alphabetical | 87.50% |
| ToolTrie-v0 | 83.75% |
| Difference | **-3.75 points** |
| Task-clustered 95% CI | **[-5.67, -2.00]**, excludes zero |
| Discordant pairs | alphabetical-only correct **54**, ToolTrie-only correct **9** |

**This is a confirmed regression, not a null result.** The point estimate has
been stable near -4 points across three independent measurements; only the
interval changed. Plain alphabetical ordering remains the safest condition
measured at declining irrelevant requests.

Counting axes honestly: causal ContextPilot beats ToolTrie-v0 on reuse, on
function-name accuracy, and on full-call accuracy. ToolTrie-v0 wins one axis,
no-tool safety, by 5.62 points. **No combined utility weighting has been
declared, so no overall winner is claimed** — and any such weighting must be
declared before looking at these numbers.

### 4.8 Model scale: why two models, and what changed with size

**Source:** `PROJECT_STATUS.md`, "ToolTrie-v0" section;
`reports/tooltrie-phase2/findings.md` §4.

On the same 100 stratified BFCL tasks, ToolTrie minus alphabetical was **+7.50
points** on function-name accuracy at 0.6B but **-3.75 points** at 8B — the
effect inverted with model size. Reporting the 0.6B quality numbers alone would
have produced a confident and wrong conclusion.

This was verified to be a genuine scale effect rather than a capacity confound:
the 8B planner ran at 44,656-token capacity versus 190,896 at 0.6B and evicted
424 trie nodes, yet produced **byte-identical orderings on all 100 records**.

**The 8B regression was itself sampling noise.** Repeated at 200 per domain
(n=1,000), a nested superset of the 100 above:

| ToolTrie - alphabetical, Qwen3-8B | n=100 | n=1,000 | 95% CI at n=1,000 |
| --- | ---: | ---: | --- |
| function-name | -3.75 | **+0.75** | -3.03 … +4.53 |
| full call | -6.25 | **+1.50** | -2.66 … +5.66 |
| no-tool | +0.00 | -4.00 | -10.89 … +2.89 |

Both headline metrics reverse sign and zero sits inside every interval, so **no
quality cost is detectable at 8B on this sample**. Two methodological lessons
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
needs roughly n=15,000 — about 12 GPU-hours per condition at 8B. **The 1-point
gate is unfalsifiable at any affordable sample size** and should be restated as
an equivalence test with a declared margin.

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
| **Random, seed 42** | **32.16%** | **31.27%** | **32.67%** | **30.19%** |
| Frequency = schema-cost = FP-tree | 9.44% | 8.99% | 9.51% | 9.00% |

Fixed random seed 42 leads reuse in every regime. This is **one run per cell and
one seed with no repeated-trial interval** — a sensitivity result motivating a
random-seed sweep, not a deployment recommendation. Reuse and latency here must
not be compared numerically with the 190,896-token cache, and this is not
evidence of natural production pressure.

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
| ContextPilot | actual upstream code, v0.4.1, commit `1fa0a143fdeda344585666648ab2b30cb7fea77f`; causal variant driven by our harness with the algorithm unmodified | `ContextPilot offline` / `ContextPilot causal (our online harness)` |
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

The initial brief is formally closed. Section 9 extensions — retained inactive
tools, active-tool manifests, admission policy, KV composition — remain future
work and are unbuilt.

---

## 7. Limitations and open questions

### 7.1 The retriever is a lexical floor

BM25 is a reproducible baseline, not the official ToolRet retriever. A dense
retriever returns more semantically clustered menus, which could mean *more*
exact prefix overlap. Since the central negative result rests on retrieved menus
sharing little prefix, this is the highest-value robustness check available.

### 7.2 No external system has been run on retrieved menus

ContextPilot, CacheWeaver, and the fitted policies were measured only on padded
menus. Whether ContextPilot's ~9-point lead survives the move that cost ToolTrie
48 points is unknown, and it directly decides how general the negative result is.
This is a cheap experiment on existing harness code.

### 7.3 The no-tool comparison against ContextPilot is under-powered

ContextPilot's -7.50-point no-tool penalty comes from a 160-case arm. The
well-powered 240-task protocol was run only for ToolTrie against alphabetical.
Since no-tool safety is the single axis ToolTrie wins, this is the measurement
that decides the multi-objective comparison.

### 7.4 No overall winner is claimed

No combined utility or cost weighting has been declared between reuse, call
accuracy, and decline safety. Any weighting must be declared before looking at
the measured numbers.

### 7.5 Latency claims are deliberately weak

No paired-difference interval was predeclared for TTFT on the retrieved arm, and
the predeclared 1-point quality gate is unfalsifiable at affordable sample sizes
(§4.8). Latency across the two cache capacities in §4.9 is not comparable at all.

### 7.6 Recommended next experiment

The active-tool manifest with constrained generation, because it targets the one
confirmed regression: separating the tools present in cached context from the
tools that may actually be called would allow reuse without the decline penalty.
It must be evaluated against causal ContextPilot and against the ordinary
text-prefill fallback, and should include a random-seed sensitivity sweep rather
than promoting seed 42 as an algorithm.

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

### GPU execution records

| Handover | Covers |
| --- | --- |
| `reports/initial-brief-closure/20260805-222246-gpu-executor/HANDOVER.md` | 28 workloads, 84 replays, 7 audits, failed pressure attempt |
| `reports/initial-brief-pressure-rerun/20260807-005414/HANDOVER.md` | 24/24 accepted controlled-pressure runs |

### Predeclared protocols

- `cluster/initial-brief-closure-manifest.json`
- `cluster/initial-brief-pressure-rerun-manifest.json`

Both were written before execution. The pressure threshold was never lowered
after inspection; a separate protocol was declared instead.

### Raw archives (outside version control)

| Archive | Size | SHA-256 |
| --- | ---: | --- |
| `initial-brief-closure-20260805-222246.tar.gz` | 127,635,469 B, 221 entries | `568dedaea859056a4c3d2cb4773a810364b40b03858efc83092c5d0804d4a7d5` |
| `initial-brief-pressure-rerun-20260807-005414.tar.gz` | 833,218 B, 36 entries | `d18c613958dbeadbb9114a7ec6c1418d1c062803075974940b470791e86a78e0` |

Both currently exist as a single copy on the GPU server's physical disk and
require an off-machine backup.

### Source code

- `src/tatm/tooltrie.py` — the planner (§2)
- `src/tatm/retrieval.py` — BM25 baseline (§4.1)
- `src/tatm/reporting.py` — report generation
