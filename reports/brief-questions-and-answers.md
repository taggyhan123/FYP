# Initial Research Brief — Questions and Answers

Every question posed by `initial-research-brief.md`, with the answer the
measured work supports.

Questions are quoted verbatim from the brief. Answers cite the tracked report or
artifact behind them. Where a question is unanswered, that is stated rather than
softened — four of the brief's questions were deliberately deferred to the §9
extension stage and one sub-question was never measured.

**Answer status legend**

| Mark | Meaning |
| --- | --- |
| **Answered** | Measured evidence supports a definite answer |
| **Partly answered** | Measured, but with a material gap named in the answer |
| **Deferred** | Deliberately out of scope for the initial stage |

---

## Part 1 — High-level research questions (brief §2)

**Read the scope note before the answers.** The brief states, immediately after
listing these six:

> *"The first three questions define the initial implementation scope. The later
> questions are research extensions that may be pursued once the basic
> measurements and baselines are reliable."*

**Q1–Q3 are the questions the initial stage was required to answer, and all
three are now answered — Q3's "weighted" clause was closed on 2026-08-11 and
its answer is negative: weighting the trie changes no emitted ordering.** Q4–Q6
were designated extensions from the outset; two were nevertheless answered in
substantial part.

| Question | Required by the initial stage? | Status |
| --- | --- | --- |
| Q1 Tool locality | **Yes** | **Answered** |
| Q2 Prefix organization | **Yes** | **Answered** |
| Q3 Trie-aware memory | **Yes** | **Answered** — weighted variant measured; it changes no ordering |
| Q4 Retention trade-off | No — extension (§9.1) | Deferred |
| Q5 Long-tail tools | No — extension (§9.4) | Partly answered anyway |
| Q6 Quality and safety | No — extension (§9.2) | Mostly answered anyway |

A "Deferred" or "Partly answered" mark below therefore records **correct
sequencing, not an unmet requirement**. Each remaining gap is blocked on
machinery that would compromise the trustworthiness of the current baseline if
added early; the reason is given per question.

### Q1. Tool locality — **Answered**

> *Do tool requirements and successful calls exhibit hotspots, co-occurrence
> patterns, repeated workflows, or session locality?*

**Yes, but the locality is modest and it is not where it was expected.**

ToolRet's natural file order is **more** local than a synthetic session-bursty
replay — 31.35% versus 27.10% reuse at 25% capacity — because that file order is
already **99.86% same-domain adjacent**. Constructing an artificial bursty
workload made locality worse, not better.

Under controlled cache pressure, `empirical` and `session_bursty` do produce
different reuse despite being permutations of one task multiset, which confirms
request order matters once the cache is genuinely constrained.

Two honesty notes carried from the record:

- An earlier locality claim was **withdrawn**. The first `trie_metrics`
  implementation retained every node forever, which made reuse depend only on
  the multiset of requests and not their order; `session_bursty` was therefore
  identical to `empirical` *by construction*. `bounded_trie_metrics` added
  capacity with leaf-first LRU eviction, and only then did the replays diverge.
- The practical ceiling is low: ToolRet's median relevance set and BFCL's median
  exposed menu **both contain one tool**, which limits what any reordering can
  achieve on many tasks.

**Source:** `PROJECT_STATUS.md` "Task D"; `reports/access-patterns.md`;
`reports/initial-brief-closure/findings.md` §5.

### Q2. Prefix organization — **Answered**

> *Can deterministic, cost-aware tool ordering create longer reusable prefixes
> than the original, alphabetical, or frequency-only order?*

**Yes — but the winning method is not the cost-aware one.**

On padded shared-catalog menus, ToolTrie-v0 reaches **87.19%** measured prefix
reuse against 38.13% for alphabetical and 1.19% for the original order, with
non-overlapping 95% intervals and a 2.3x aggregate TTFT reduction.

Crucially, the *cost-aware* orderings the question anticipates do **not** win.
On BFCL, exact reconstruction shows that frequency, schema-cost,
FP-tree-derived, pair, and triple fitted labels emit the **same byte-identical
tool sequence** on all 200 requests; the five labels therefore represent one
tested policy behavior there. On ToolRet there are three distinct fitted
sequences: frequency/pair/triple are identical, while schema-cost and FP-tree
each differ. Alphabetical is substantially higher than all three on that arm.

The lesson in these measured workloads is that the cache rewards **identical**
prefixes, not importance scores by themselves. Modelling which tools matter
adds little unless it makes requests emit the same prefix.

The magnitude is workload-dependent — see Part 2, Q2.

**Source:** `reports/tooltrie-phase2/findings.md` §2;
`reports/initial-brief-pressure-rerun/ordering-equivalence.json`.

### Q3. Trie-aware memory — **Answered**: built, bounded, budget-tested and weighted; the weighting changes nothing

> *Can frequently occurring tool sequences be represented as a weighted trie or
> prefix memory under a limited cache budget?*

**Yes. Built, bounded, and measured.**

`src/tatm/tooltrie.py` maintains a trie whose edges are tool IDs and whose
root-to-node paths are sequences that were actually served. Nodes carry
`schema_tokens`, `cumulative_schema_tokens`, `visit_count`, and `last_seen`. The
budget is enforced by `capacity_tokens` and `max_nodes` with leaf-first LRU
eviction.

The budget was exercised, not merely implemented: at Qwen3-8B the planner ran at
a **44,656-token capacity** versus 190,896 at 0.6B and **evicted 424 nodes**, yet
produced **byte-identical orderings on all 100 records**. The bounded structure
therefore works without distorting the ordering at that scale.

**Measured under a genuinely constrained cache on 2026-08-11.** The clause "under
a limited cache budget" was untested until then. ToolTrie-v0 replayed into the
480-block harness — 7,680 tokens, a 25× reduction — across all four locality
regimes: 4/4 accepted, 64/64 checks, peak occupancy 0.904–0.908.

| regime | empirical | uniform | skewed | session-bursty |
| --- | ---: | ---: | ---: | ---: |
| ToolTrie-v0 | **87.18%** | **88.54%** | **94.73%** | **91.62%** |
| best of the six static orderings | 32.16% | 31.27% | 32.67% | 30.19% |

It leads by **55–62 points** and loses nothing to the smaller cache — 87.18%
against 87.19% uncapped — because it concentrates the shared core into a single
~6,950-token prefix that fits inside the 7,680-token capacity. Its own metadata
eviction fired 506–1,494 times per regime, the first time that path has run.
**Source:** `reports/tooltrie-pressure/20260811-001032/`.

**The "weighted" half was answered on 2026-08-11, and the answer is that it
makes no difference.** `WeightedToolTrie` (`src/tatm/tooltrie_v1.py`) reads the
`visit_count` v0 never consults — in the selection tie-break, and in eviction,
where it drops the least-visited leaf instead of the least recent. Run into the
same 480-block harness: **4/4 accepted, 64/64 checks**.

| regime | empirical | uniform | skewed | session-bursty |
| --- | ---: | ---: | ---: | ---: |
| ToolTrie-v0 | 87.184% | 88.535% | 94.734% | 91.619% |
| ToolTrie-v1 weighted | 87.184% | 88.535% | 94.734% | 91.619% |

Identical to five decimal places. Re-deriving both planners offline on the same
menus shows **0 of 200 records differ**, in every regime. The weighting does act
— v1 evicts differently, 1,497 nodes against 1,494 on empirical — but
`_reachable_cached_cost` decides almost every choice, so the tie-break holding
the weight is rarely reached.

So the trie is now bounded, budget-tested **and** weighted, and the weighting
changes nothing on this workload. The caveat is scope: one workload, one
capacity, one seed. Weighting could matter where the leading term ties more
often — narrower menus, or a workload without a fixed 63-tool core.
**Source:** `reports/tooltrie-weighted/20260811-144741/`.

One clarification for readers of the code: this budget governs the **planner's
own metadata** — how much served history it remembers — and is *not* KV-cache
retention. That is Q4, and it is unbuilt.

**Source:** `src/tatm/tooltrie.py`; `PROJECT_STATUS.md` "ToolTrie-v0".

### Q4. Retention trade-off — **Deferred** *(extension question; not required by the initial stage)*

> *Is it sometimes beneficial to retain a small number of cached but currently
> inactive tools in order to reuse a longer prefix?*

**Not answered. Deliberately out of scope.**

This is brief §9.1, and it was explicitly listed under `excludes` in the closure
manifest as "inactive-tool retention". The brief sequences extensions after a
stable baseline, and no inactive tool is retained anywhere in the current system.

**Why it cannot simply be added.** Retaining an inactive tool means putting a
tool in the prompt that the request does not need. That breaks the permutation
guarantee `assert set(ordered_ids) == set(ids)`, which is exactly what makes
every current measurement attributable to ordering alone. Answering Q4 changes
the experimental design rather than extending it, and the brief additionally
requires evaluating context overhead, decode KV I/O, tool confusion, and
permissions alongside it.

The initial stage did produce what the brief asked for here — an experimental
basis for deciding whether to pursue it. See Part 3.

**Source:** `cluster/initial-brief-closure-manifest.json`, `scope.excludes`.

### Q5. Long-tail tools — **Partly answered** *(extension question; not required by the initial stage)*

> *For tools not covered by the exact prefix memory, when is text prefill
> preferable to loading or composing a precomputed KV representation?*

**The text-prefill side is answered; the KV-composition side was never built.**

What is established is a crossover and a decision rule. Prefix caching pays only
above roughly **4 tools / 433 tokens**; below that there is no resolvable
benefit. Native exact APC already converts the token-reuse signal into a
repeatable TTFT benefit — **10.6x at 200 tools**, 4.5x at 64.

The measured recommendation is therefore: **do not pursue arbitrary independent
KV concatenation yet**, because native APC already captures the available
signal, and text prefill remains the conservative default whenever
expected savings do not exceed context, decode, retrieval, and safety costs.

What is *not* answered is the comparison the question literally poses, because
no precomputed KV representation was ever composed. "KV tensor composition" was
in the closure manifest's `excludes`; it is brief §9.4 and is flagged there as
an advanced extension.

**Why it cannot simply be added.** Composing a precomputed KV representation
requires computing a tool's KV state independently and relocating or linking it
after a cached prefix — that is, modifying KV tensors. The project's own working
rules forbid patching vLLM, CUDA, attention, or KV-cache internals, precisely so
that measured results remain attributable to prompt-layer changes.

**Source:** `reports/initial-findings.md` "Prefill cost and the measurement
floor" and "Recommendation".

### Q6. Quality and safety — **Partly answered** *(extension question; not required by the initial stage)*

> *How do additional or reordered tools affect tool selection, argument
> generation, no-tool decisions, and unauthorized-tool prevention?*

**Three of the four sub-questions have fixed-workload evidence. The fourth was
never measured.** The Qwen3-8B runs show a relevance/no-tool frontier on their
fixed sequence, but a Qwen3-4B confirmation does not reproduce the no-tool
penalty. The evidence therefore does not support a universal causal trade-off.

BFCL correctness, n=800, Qwen3-8B:

| Condition | function-name | full call | no-tool |
| --- | ---: | ---: | ---: |
| Alphabetical | 82.81% | 76.41% | **89.38%** |
| ToolTrie-v0 | 83.75% | 77.66% | 87.50% |
| ContextPilot static-refit causal adaptation (alpha=0.5) | **84.84%** | **79.06%** | 81.88% |

The later ContextPilot persistent-API adaptation (`alpha=0.001`, ordering only)
scores 84.84% / 78.44% / 84.38% at 8B. Against alphabetical that is +2.03
points on function name, +2.03 on full call, and −5.00 on no-tool accuracy. At
4B, its relevance gain reproduces (84.53% / 77.19%), while the no-tool
difference against alphabetical is exactly 0.00 points (85.62% each).

The targeted no-tool analysis is better powered and more damaging. Across all
240 BFCL irrelevance tasks under 5 fixed menu seeds, task-clustered bootstrap:

| | value |
| --- | --- |
| Alphabetical | 87.50% |
| ToolTrie-v0 | 83.75% |
| Difference | **-3.75 points** |
| 95% CI | **[-5.67, -2.00]**, excludes zero |
| Discordant pairs | alphabetical-only correct **54**, ToolTrie-only correct **9** |

This is a **resolved difference for the fixed emitted sequence, not a null
result**. The n=100 no-tool difference was 0.00 points and the nested n=1000
estimate was -4.00 points, so these are not three independent replications.
Because ToolTrie ordering depends on prior requests, uncertainty over alternative
request sequences requires complete planner replay replicates.

**Unauthorized-tool prevention was not measured at all.** It requires separating
the tools present in cached context from the tools that may actually be called —
an active-tool manifest plus a constrained decoder. That is brief §9.2, and it is
unbuilt: `grep -i constrain` over `src/` and `scripts/` returns zero hits.

The other three sub-questions were answered despite this being an extension
question, so Q6 is substantially over-delivered relative to what the initial
stage required.

**Source:** `reports/tooltrie-phase2/findings.md` §1 and §4;
`reports/contextpilot-confirmation/findings.md`.

---

## Part 2 — Initial experimental questions (brief §7)

### Q1. How much of TTFT is caused by selected tool-schema prefill? — **Answered**

Material, but only above a floor. A controlled sweep found **no resolvable
benefit at 303 tokens**, a **4.5x** warm-cache TTFT speedup at 64 padded tools,
and **10.6x** at 200 tools. On the retrieved arm, prompt cost per query runs from
640 tokens at k=4 to 16,267 at k=128, with TTFT from 32.3 ms to 937.0 ms.

### Q2. How much exact prefix reuse is available without changing the tool set? — **Answered**

Reuse exists without altering membership, but it depends on set overlap and is
small on the deterministic BM25 menus. Ordinary retrieved-order reuse falls from **15.87% at
k=4 to 0.37% at k=128**. Deterministic reordering improves it without adding or
removing any tool:

| k | Original (fallback) | Best static | **ToolTrie-v0** | ToolTrie gain |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 15.87% | 15.87% | **17.48%** | +1.61 |
| 16 | 6.12% | 6.27% | **7.77%** | +1.65 |
| 64 | 0.91% | 1.24% | **1.90%** | +0.99 |
| 128 | 0.37% | 0.59% | **1.13%** | +0.76 |

**This is the project's central negative result.** The same planner and code
yield 87.19% on padded shared-catalog menus and 1.90% at k=64 on independently
retrieved menus. The large figure is a property of the workload, not the planner.

### Q3. Does frequency-based ordering improve reusable prefix length? — **Answered: it depends on the estimator**

**Corrected 2026-08-10.** This was previously recorded as "no". That answer
holds only for the *frozen* variant, and the distinction turns out to be the
whole question.

- **Frozen, fitted on a task-disjoint corpus (`frequency_fitted`): no.** 39.69%
  on BFCL against alphabetical's 38.13%, and 41.58% on ToolRet against
  alphabetical's 51.05% — actively worse. It can destroy the shared prefix by
  placing request-specific tools early. A held-out corpus structurally cannot
  know which tools *this* evaluation stream happens to share, so it places the
  odd-tool-out at mean position 25.0 of 64, no better than alphabetical's 24.1.
- **Estimated online from the served stream (`frequency_online`): yes.** 96.27%
  on BFCL padded-64, tied with ContextPilot at the structural ceiling, and it
  beats ToolTrie-v0 in 10 of 12 systems cells. Same signal, causal, no training
  corpus, roughly thirty lines.

The frozen fit was re-audited and is uncontaminated: a leaked fit would rank the
universal tools first and reach ~96%, whereas it places them at mean position
31.6 of 64. Its failure is evidence of its integrity, not of the signal being
weak.

Note also that the frozen family has no `observe()` at all, while every policy
it was benchmarked against adapts to the served stream. Part of the original gap
was adaptivity rather than algorithm. See `reports/frequency-online/findings.md`
and `reports/key-findings.md` §6.

### Q4. Does weighting frequency by schema length or measured prefill time perform better? — **Partly answered**

Schema-cost weighting is **not reliably better either**. It narrowly beats
several static controls in some rows but never beats ToolTrie, and it collapses
onto the same order as frequency and FP-tree under measurement. This tests
tokenized schema length as a cost proxy; it does **not** test a policy weighted
by separately measured per-schema prefill time, so that half of the question
remains open.

### Q5. How much additional benefit comes from pair/triple workflow structure? — **Answered 2026-08-11: +0.33 to +0.36 points, and only on retrieved menus**

Downgraded 2026-08-10 from "Answered: essentially none" to "never tested"; answered
properly on 2026-08-11, first with a proof and then with a measurement.

On BFCL the five fitted policies do not merely land within 0.01 percentage
points of one another — they emit **byte-identical `tool_ids` on all 200
records**, differing only in the `ordering` label string. Their extra statistics
never discriminate, so each key falls through to the same frequency ordering.
An experiment in which the pair/triple policy produced the same file as the
frequency policy did not test pair/triple structure and find it unhelpful; it
**never tested it**. That is absence of evidence, not evidence of absence, and
the BFCL row of that table reports one policy five times.

On ToolRet padded menus `schema_cost_fitted` differs on 200/200 records and
`fp_tree_conditional` on 10/200 — but `conditional_pair` and
`conditional_pair_triple` remain identical to `frequency_fitted` **there too**,
so Q5 was not posed on either padded workload. An earlier version of this entry
said the question "was actually posed" on ToolRet and the answer was no benefit.
That was wrong: the two policies carrying pair/triple structure never
discriminated, and the ones that did differ test Q4 and FP-tree order instead.

**Why no planner could have posed it: pair support is redundant with frequency.**
When a menu is a fixed core plus one varying tool, `support(a, b)` equals
`min(presence(a), presence(b))` exactly, so a pair key carries no signal a
frequency key does not
(`scripts/audit_pair_triple_information.py`):

| workload | tools in every request | `pair == min(presence)` | violations |
| --- | ---: | ---: | ---: |
| bfcl-padded64 | 63 | **100.00%** | **0** |
| toolret-padded64 | 60 | 99.39% | 46 |
| toolret-bm25-k16 | 0 | 85.16% | 2,651 |
| toolret-bm25-k128 | 0 | 67.34% | 208,615 |

Zero violations over 14,490 BFCL pairs. The five-way tie is a theorem, not a
coincidence, and structure exists only where menus are genuinely retrieved.

**Measured where it exists.** `OnlinePairTriplePlanner` against an online
frequency counter, Qwen3-0.6B at 190,896 tokens, 200 requests each, identical
prompt-token totals, orderings differing on 69/200 and 194/200 records:

| workload | `frequency_online` | `pair_triple_online` | delta |
| --- | ---: | ---: | ---: |
| toolret-bm25-k16 | 8.155% | **8.514%** | **+0.359 pp** |
| toolret-bm25-k128 | 2.412% | **2.746%** | **+0.334 pp** |

Real, consistent in sign across two depths, and small. Pair/triple structure
helps precisely where there is almost nothing left to gain, and neither figure
reaches ContextPilot at the same depth. The k=128 row is pair-only — triples are
cubic in menu width and were capped.

**Source:** `reports/tooltrie-weighted/20260811-144741/`.
The ContextPilot-derived static-refit ordering is the important measured
competitor here, not evidence that the current pair/triple adaptation is
sufficient. A later persistent-API adaptation also leads ToolTrie on every
retrieved menu size; neither adaptation is the full annotated/eviction-aware
ContextPilot system.

### Q6. How sensitive are results to request ordering and session locality? — **Answered: highly**

Causality is **load-bearing**. Granting ToolTrie the whole evaluation batch and
iterating to a fixpoint collapses it from **87.19% to 29.96%** — below plain
alphabetical. Early requests establish a path and later ones follow it; treating
the batch atemporally destroys that self-reinforcement. "Offline" is not an upper
bound for this planner, it is a different and worse algorithm.

Under controlled pressure, `empirical` and `session_bursty` produce different
reuse despite being permutations of one task multiset.

**But the locality regime is the weaker of the two factors.** Across the 28
regime-runs in the pressure matrix no ordering moves more than a few points
between empirical, uniform, skewed and session-bursty, while the spread *between*
orderings within one regime is 1.18% to 87.18%. Ordering dominates the request
distribution. The strongest sensitivity is to whether the policy is adaptive at
all: ToolTrie-v0 leads the best static ordering by 55–62 points in every regime
(`reports/tooltrie-pressure/20260811-001032/`).

### Q7. Does tool reordering change function-call accuracy? — **Answered: yes**

See Part 1, Q6. On the fixed Qwen3-8B sequence, a reuse/selection/no-tool
frontier exists: both ContextPilot-derived policies lead ToolTrie on reuse and
relevance-side point estimates while ToolTrie has higher no-tool accuracy.
These are ordering-only results, not full-system ContextPilot quality.
ToolTrie's own no-tool regression against alphabetical is confirmed at −3.75
points with an interval excluding zero for its targeted fixed sequence. The
persistent arm's no-tool loss does not reproduce at Qwen3-4B, so the broader
trade-off is model-sensitive.

### Q8. Under which workloads does trie-aware ordering provide little or no benefit? — **Answered**

Three regimes, all demonstrated by the retrieved-menu arm:

1. **Menus sharing little exact prefix.** Independently retrieved menus reduce
   the gain to +0.76–1.65 points.
2. **Prompts below the latency floor.** Below roughly 4 tools / 433 tokens there
   is no resolvable TTFT benefit.
3. **Retrieval missing the needed tool.** At k=128, **19.5% of queries retrieve
   none of their gold tools**. No ordering can recover a tool never retrieved.

A fourth, from the external comparison: the ContextPilot persistent-API
adaptation leads ToolTrie even on retrieved menus, but its own reuse falls to
1.99–18.72%. ToolTrie is not the best measured policy, and no ordering solves
the low-overlap regime.

---

## Part 3 — Desired initial outcomes (brief §8)

| Outcome required | Status |
| --- | --- |
| A clean and reproducible tool-schema processing pipeline | **Delivered** — 45,815 schemas tokenized; 44,453 ToolRet tools, 1,362 canonical BFCL functions |
| A reliable vLLM prefix-cache benchmark | **Delivered** — 84 replays with clean counters; the `cached + computed == prompt_tokens` identity doubles as a contamination detector |
| An evidence-based characterization of tool locality and cacheability | **Delivered** — Part 1 Q1; includes a withdrawn earlier claim |
| A set of exact, quality-preserving ordering baselines | **Delivered with a qualification** — the orderings are exact permutations, but "quality-preserving" does not hold unconditionally given the no-tool regression |
| A first ToolTrie prototype using native prefix caching | **Delivered** — causal, bounded, measured on vLLM; the SGLang arm awaits corrected raw-counter revalidation |
| An experimental basis for deciding whether to pursue retention, long-tail KV memory, or another direction | **Delivered** — see below |

The brief states the primary success criterion is *"obtaining a trustworthy
answer to whether public tool workloads contain exploitable prefix locality and
identifying the conditions under which the optimization is beneficial."*

**That criterion is met, and the answer is qualified rather than affirmative.**
Exploitable locality exists but is small on deterministic BM25-retrieved menus, the
optimization is beneficial mainly on padded shared-catalog workloads above the
latency floor, and it carries a measured safety cost.

**The direction the evidence supports** is brief §9.2 — an active-tool manifest
with constrained generation — because it targets the fixed-sequence decline regression:
separating the tools present in cached context from the tools that may actually
be called would allow reuse without the decline penalty. §9.3 admission by
measured reuse value is well motivated by the "identical, not important" result
but optimizes something that already works. §9.4 KV composition is explicitly
advised against for now.

---

## Part 4 — First discussion deliverables (brief §11)

| Deliverable | Status |
| --- | --- |
| 1. Reading note from Task A | **Delivered** — `notes/reading-note.md` |
| 2. Working vLLM prefix-cache sanity script | **Delivered** — Task B, five checks measured on GPU |
| 3. Dataset inventory for ToolRet and BFCL | **Delivered** — `reports/dataset-inventory.md` |
| 4. Preliminary schema-length and frequency analysis | **Delivered** — median schema 70 Qwen tokens, P95 220 |
| 5. One-page proposal for the first exact ToolTrie baseline | **Superseded** — the baseline was built and measured rather than proposed |
| 6. List of unresolved questions or implementation risks | **Delivered** — Part 5 |

---

## Part 5 — Unresolved questions and risks

1. **The retriever is a lexical baseline.** BM25 is deterministic and leak-free but
   is not the official ToolRet retriever. A dense retriever returns more
   semantically clustered menus, which could mean *more* exact prefix overlap.
   Since the central negative result rests on retrieved menus sharing little
   prefix, this is the highest-value robustness check available.
2. **The external retrieved-menu comparison is ordering-only.** The
   ContextPilot persistent-API adaptation is now measured and leads ToolTrie at
   k=4/16/64/128, but it omits annotations, de-duplication, eviction feedback,
   and interleaved serving. The corrected `alpha=0.001` static-refit cells are
   still pending.
3. **The no-tool comparisons are model- and sequence-limited.** The historical
   static-refit penalty (−7.50 points) and persistent 8B penalty (−5.00) each
   use 160 irrelevance cases on one planner sequence; the persistent difference
   is 0.00 at 4B. The larger targeted 240-task protocol was run only for
   ToolTrie against alphabetical.
4. **No combined utility weighting has been declared,** so no overall winner is
   claimed between ToolTrie and either ContextPilot-derived adaptation. Any
   weighting must be declared before looking at the numbers.
5. **The predeclared 1-point quality gate is not practically resolved by the
   current design and budget.** The 95% CI on the full-accuracy difference is
   about +/-4.2 points at n=1,000. A defensible power calculation must account
   for paired discordance and sequence-dependent treatment, rather than simply
   scaling an independent-sample estimate. It should be restated as an
   equivalence test with a declared margin and predeclared sequence replicates.
6. **A 100-task BFCL sample got the *sign* wrong.** At n=100 the 8B quality gap
   read -3.75 points; at n=1,000 it read +0.75 with zero inside every interval.
   Small BFCL samples are pilots, not evidence.
7. **Seed 42's pressure win is one run per cell with one seed.** It leads reuse
   in all four regimes under the 7,680-token cache, but with no repeated-trial
   interval it is a sensitivity result motivating a seed sweep, not a policy.
8. **All four recorded raw archives have only one known copy** on the GPU
   server's physical disk and need an off-machine backup.
9. **The historical SGLang cleanliness check was not independent.** Its
   reconciliation compared two values derived from the same responses. The raw
   runs must be revalidated against aggregate `sglang:cached_tokens_total`
   before the second-engine arm is called contamination-validated.
10. **The accepted ContextPilot quality JSON predates the inference-metadata
    fix.** Its point estimates remain valid, but GPU-side raw score files must
    be re-analysed with `--sequence-state-dependent` before those files are
    used as inferential artifacts.

---

## Appendix — Sources

| Document | Contents |
| --- | --- |
| `initial-research-brief.md` | The questions quoted above |
| `reports/initial-findings.md` | Task F deliverable, generated |
| `reports/initial-brief-closure/findings.md` | Closure verdict, retrieved-menu arm, audits, pressure |
| `reports/tooltrie-phase2/findings.md` | External comparison, quality/safety, provenance |
| `reports/consolidated-report.md` | Mechanism and comparison in one place |
| `PROJECT_STATUS.md` | Hand-maintained status index |
| `src/tatm/tooltrie.py` | The planner |

GPU execution records:
`reports/initial-brief-closure/20260805-222246-gpu-executor/HANDOVER.md` and
`reports/initial-brief-pressure-rerun/20260807-005414/HANDOVER.md`.

Predeclared protocols: `cluster/initial-brief-closure-manifest.json` and
`cluster/initial-brief-pressure-rerun-manifest.json`.
