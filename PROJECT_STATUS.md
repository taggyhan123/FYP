# Project status

Status after the first local research pass:

| Brief task | Status | Artifact / remaining work |
| --- | --- | --- |
| A — reading note | Complete | `notes/reading-note.md` |
| B — exact prefix caching | Complete; all five checks measured on GPU | See "Task B detail" below |
| C — normalize datasets | Complete for ToolRet and five BFCL V4 static subsets | `scripts/download_datasets.py`, `scripts/run_pipeline.py`, `reports/dataset-inventory.md` |
| D — access patterns | Complete for benchmark evidence, four controlled replays, and the retrieved-menu ordering matrix | `reports/access-patterns.md`, `reports/tables/`, `reports/initial-brief-closure/findings.md` |
| E — exact ToolTrie baseline | Measured on GPU on shared padded catalogs and true BM25-retrieved menus, with ordinary text prefill retained as the explicit fallback | `src/tatm/tooltrie.py`, "Task E" and "ToolTrie-v0" below, `reports/tooltrie-v0/findings.md`, `reports/initial-brief-closure/findings.md` |
| F — initial report | Substantively complete; dual-model, static-refit, and SGLang compact evidence are integrated. Raw-archive backup remains; the historical 8B fallback comparison is optional and uncontrolled unless its unpinned model snapshot can be recovered | `reports/initial-findings.md`, `reports/initial-brief-closure/findings.md`, `reports/contextpilot-dual-model/findings.md` |

The external comparison from `NUS_GPU_PHASE2_INSTRUCTIONS.md` has now been
**measured on GPU** — targeted no-tool evaluation, CacheWeaver, fitted
FP-tree/co-occurrence baselines, ContextPilot-derived offline/static-refit
orderings, and a separate stock
SGLang/RadixAttention engine comparison. See "Phase 2" below.

The historical §5 engine tables contain all planned replays:
`contextpilot_causal` was replayed 3×
on all three systems arms and replicates to within 0.27pp (BFCL 96.16 / 96.18 /
96.38% on unsanitized vLLM, sanitized vLLM, SGLang). A post-run audit found that
this row is a **static-refit causal adaptation** (`alpha=0.5`, ordering only),
not official persistent online ContextPilot. The second engine validates the
same precomputed ordering but does not validate planner/API parity. The old
SGLang reconciliation was circular, but the later fail-closed audit now checks
all 72 raw runs against the independently recorded aggregate cached-token
counter and accepts 72/72 with no refusals.
The generalized static-refit adapter's integer-ID restoration was also fixed;
the historical Phase 2 builder already performed the equivalent inverse
mapping, so that correction does not invalidate its emitted workload.

The historical §4 quality table is also complete. The static-refit arm scores
+2.03pp on function-name accuracy but **−7.50pp on no-tool** (CI
[−11.88, −3.75], discordant 12:0 against it). These are valid scores for the
emitted ordering, not for full ContextPilot, which uses the persistent API and
can add relevance annotations.

**Counted honestly, the historical static-refit adaptation wins three of four
axes on the padded trace**: reuse
(+8.97pp BFCL, +11.24pp ToolRet), function-name accuracy (84.84% vs 83.75%) and
full-call accuracy (79.06% vs 77.66%). ToolTrie-v0 wins one — no-tool safety, by
5.62pp — and is the only high-reuse ordering whose no-tool interval does not
exclude zero. "Neither dominates" holds only as a multi-objective statement and
must not be read as parity: **no combined utility or cost weighting has been
defined**, so this report declares no overall winner. Any such weighting has to
be declared before looking at these numbers.

**No historical Phase 2 arms remain in flight.** A later confirmation has now
measured a causal **ContextPilot persistent-API adaptation** at the pinned
upstream commit and paper/default `alpha=0.001`. It is ordering-only: no
eviction feedback, relevance annotations, or de-duplication, so it is not the
full ContextPilot system.

That persistent arm leads ToolTrie-v0 on all four BM25-retrieved menu sizes:
18.72 vs 17.48% at k=4, 9.93 vs 7.77% at k=16, 4.78 vs 1.90% at k=64, and
1.99 vs 1.13% at k=128. It also reproduces the 96.16% BFCL padded-menu result.
This resolves the earlier attribution and retrieved-menu gaps, but changes the
interpretation: the 95–96% headline is a property of highly overlapping padded
menus, while realistic retrieved-menu reuse remains small.

The confirmation's Qwen3-8B quality replay finds +2.03pp function-name,
+2.03pp full-call, and −5.00pp no-tool accuracy against alphabetical on the
fixed sequence. An earlier Qwen3-4B run found 0.00pp on no-tool, and the fresh
dual-model 4B run finds −0.62pp with an interval spanning zero. The relevance
gain reproduces, but the decline penalty is model-sensitive on current
evidence, not a universal effect.

The corrected `alpha=0.001` static-refit and persistent-API adaptations are now
both measured in a fresh dual-model experiment. Qwen3-4B is the primary model
and Qwen3-0.6B is a separately reported replication. All 190 predeclared GPU
replays are accepted, all 33 GPU-side audit checks pass, and the copied
protocol manifest is byte-identical to the predeclared manifest. Both
ContextPilot adaptations lead ToolTrie-v0 in all 12 systems cells. The more
important result is the magnitude collapse: adaptive ordering reaches 82–96%
reuse on padded shared catalogs but only 0.9–18.7% for ToolTrie and 1.3–18.7%
for the best ContextPilot arm on 4B BM25-retrieved menus.

The quality result does not identify one overall winner. On 4B, ContextPilot
has the highest relevance-side point estimates while original order has the
highest no-tool accuracy. On 0.6B, original order has the highest relevance
accuracy and alphabetical has the highest no-tool accuracy. These are
fixed-sequence descriptions from this repository's reduced BFCL-style checker,
not official leaderboard scores or uncertainty over alternative planner
sequences. Cross-model differences are model-sensitive observations, not a
pure model-size effect, because native capacities differ and ToolTrie is
rebuilt per model.

See `reports/contextpilot-dual-model/findings.md`,
`cluster/contextpilot-dual-model-manifest.json`, and
`NUS_GPU_CONTEXTPILOT_DUAL_MODEL_INSTRUCTIONS.md`. Do not rerun the accepted
dual-model matrix. Its 58 MB raw archive still has no verified off-machine
copy.

The separately pushed static-refit/SGLang handover is now integrated. A local
compact audit verifies 18/18 static-refit systems trials, the Qwen3-8B aggregate
quality cell, six corrected paired comparisons, and 72/72 independent SGLang
counter decisions. The two raw archives remain server-only. The historical 8B
quality matrix still has no `original` condition, so its ContextPilot and
ToolTrie comparisons against alphabetical do not establish a gain over the
ordinary selected-tool text fallback.

A separately named `frequency_online` baseline was also reviewed. It is causal
and uses only presence counts from earlier requests. It reaches 96.27% on BFCL
padded-64, revealing that this positive control can be solved after one request
by identifying its 63-tool universal core. The executor's “12/12” claim was
wrong: it beats ToolTrie-v0 in 10/12 model/workload cells and loses both BM25
k=4 cells, 17.01% versus 17.48%. It has no quality replay, its capacities differ
slightly from the accepted dual-model run, and its raw replays remain
server-only. See `reports/frequency-online/findings.md`.

## Notable findings

The counterintuitive results, in order of how surprising they were:

1. **Frozen benchmark frequency and causal stream frequency answer different
   questions.** Sorting by gold-support frequency is nearly 9x worse than
   alphabetical on the original padded test (4.41% vs 38.15%) because it moves
   the changing gold tool to the front. In contrast, `frequency_online` reaches
   96.27% by counting menu presence from earlier requests and moving unseen
   tools to the tail. The latter is not production popularity; it exposes a
   workload whose 63 of 64 tools are universal.
2. **A quality tradeoff that looked real turned out to be mostly a small-model
   artefact.** On `Qwen3-0.6B`, alphabetical won reuse by 9x but frequency
   scored higher on function-name and full accuracy (77.5%/51.25% vs
   67.5%/47.5%). Re-run on `Qwen3-8B` with the same workload: that gap is
   exactly zero (91.25%/81.25% for both). What survives at both scales is
   narrower — alphabetical is modestly better at correctly declining when no
   tool applies (95.0% vs 90.0% at 8B, 95.0% vs 85.0% at 0.6B). Checking a
   counterintuitive result against a second model size was what caught this;
   the 0.6B number alone would have overstated the tradeoff. See "Ordering does
   not win on both axes" below.
3. **A metric we trusted couldn't answer the question it existed for — and
   once it could, the answer changed again on real hardware.** The original
   locality trie retained every request forever, so a "bursty" replay and a
   shuffled replay of the *same* requests always scored identically, by
   construction. `bounded_trie_metrics` with capacity eviction fixed that
   offline and found a real gap (31.35% vs 27.10%). The follow-up GPU run found
   almost identical measured cached-token totals (287,200 vs 287,616), but its
   old 42.97%/43.03% display divided by canonical tool tokens rather than
   rendered prompt tokens, and it did not sample occupancy. The ordering result
   survives; the absolute reuse and "genuine eviction" wording does not. The
   corrected harness now records the proper denominator and requires measured
   pressure before making that claim.
   See "Task D — locality is now measurable" below.
4. **The prediction model's error runs the "wrong" way.** A simplified
   tool-unit reuse estimate would be expected to overestimate, since it ignores
   messy block-alignment realities. It actually *under*-predicts by 1.2-1.5x,
   because it ignores the chat template and system preamble — boring, fixed
   text that is cached every time and outweighs everything the model gets
   wrong about tool boundaries.
5. **The framing result underneath all of the above: benchmark tasks, as
   distributed, sit below the size where any of this matters.** Median tools
   per task is 1, well under the ~433-token crossover. Every finding above only
   exists because menus were padded to deployment-realistic sizes — the
   benchmarks alone would have produced a false negative.

## ToolTrie-v0 — first causal run (measured on GPU)

Full report: `reports/tooltrie-v0/findings.md`. Raw run:
`cluster/results/tooltrie-v0-20260803-011418/` (git-ignored). RTX 3090,
vLLM 0.26.0, commit `558e923`, 3 trials per systems condition, APC reset once
before each replay, 200 requests on 64-tool menus.

| Ordering | BFCL cached ratio | BFCL TTFT (s) | ToolRet cached ratio | ToolRet TTFT (s) |
| --- | --- | --- | --- | --- |
| original | 1.19% | 53.140 ± 1.753 | 13.87% | 48.688 ± 0.149 |
| alphabetical | 38.13% | 40.324 ± 0.636 | 51.05% | 34.647 ± 0.366 |
| ToolTrie-v0 | **87.19%** | **17.378 ± 0.741** | **83.58%** | **19.762 ± 0.846** |

Every ToolTrie-vs-baseline difference has non-overlapping 95% intervals.
Alphabetical reproduced at 38.13% against the 38.15% recorded earlier in this
file, which is the cross-check that the setup had not drifted. Cached-token
counts were bit-identical across the three trials.

The mechanism is the inverse of the frequency-ordering failure above: the
planner re-derives the shared catalog order from already-served requests and
pushes each request's novel tools to the tail through its alphabetical
fallback, so the varying part of the menu stops breaking the common prefix.

### The quality gate fails at 8B, and only checking both scales revealed it

100 stratified BFCL tasks, 20 per domain, scored against `possible_answer`.

| Ordering | 0.6B name / full / no-tool | 8B name / full / no-tool |
| --- | --- | --- |
| original | 78.8% / 52.5% / 90.0% | 87.5% / 78.8% / 90.0% |
| alphabetical | 67.5% / 47.5% / 95.0% | **91.2% / 81.2% / 95.0%** |
| ToolTrie-v0 | 75.0% / 53.8% / 95.0% | 87.5% / 75.0% / 95.0% |

ToolTrie minus alphabetical is **+7.50pp** name and **+6.25pp** full at 0.6B but
**−3.75pp** and **−6.25pp** at 8B. The 8B planner ran at a 44,656-token capacity
versus 190,896 at 0.6B and evicted 424 nodes, yet produced byte-identical
orderings on all 100 records, so this is a clean model-size comparison with no
capacity confound.

### …and the larger nested sample reversed the 8B point estimates

Repeated at the maximum balanced sample, **200 per domain (n=1000)**, a nested
superset of the 100 above:

| ToolTrie − alphabetical, Qwen3-8B | n=100 | n=1000 | 95% CI at n=1000 |
| --- | --- | --- | --- |
| function-name | −3.75pp | **+0.75pp** | −3.03 … +4.53pp |
| full | −6.25pp | **+1.50pp** | −2.66 … +5.66pp |
| no-tool | +0.00pp | −4.00pp | −10.89 … +2.89pp |

Both headline metrics reverse sign and zero sits inside every interval: **no
quality cost is resolved in this fixed n=1000 replay**. Two further lessons. The n=100 sample was
optimistic as well as noisy — absolute accuracy fell for every condition on the
larger set (alphabetical full 81.25%→75.62%), so the first 20 tasks per domain
were easier than the remaining 180. And a 100-task BFCL sample got the *sign*
wrong, not merely the magnitude, so results at that size are pilots.

**The predeclared ≤1pp gate is not practically resolved by this design and
budget.**
The 95% CI on the full-accuracy difference is ±4.2pp at n=1000; resolving 1pp
needs roughly n=15,000 (~12 GPU-hours per condition at 8B). It should be
restated as an equivalence test with a declared margin. The defensible claim is
*no detectable quality cost at n=1000*, not *passes the gate*.

Open: no-tool accuracy is the one metric whose point estimate favours
alphabetical (−4.00pp, not significant), and it is the same direction
alphabetical won at 0.6B. Two appearances is worth a targeted irrelevance-only
run, since declining irrelevant requests is safety-relevant.

### Unrelated traffic during the first attempt

Two replay drivers issued requests against the same server between 01:27 and
01:43. Because `vllm:prompt_tokens_cached` is a global counter, each run's
metric window also captured the other's hits, producing impossible cached ratios
above 100%. A trial is accepted only when `vllm:prefix_cache_queries` for its
window equals its own prompt-token total; the 10 failing replays are quarantined
and excluded, and their conditions were re-run single-driver. All 18 trials
behind the table above pass that check. See
`reports/tooltrie-v0/contamination-incident.txt`.

## Phase 2 — external comparison (measured on GPU)

Full report: `reports/tooltrie-phase2/findings.md`.

**The no-tool regression is resolved for the fixed replay sequence.** All 240
BFCL irrelevance tasks x 5 menu seeds on Qwen3-8B, bootstrap clustered on the
240 tasks: ToolTrie-v0 is
**-3.75pp** vs alphabetical, 95% CI **[-5.67, -2.00]**, discordant pairs 54 vs 9.
The interval excludes zero for that emitted sequence. Because ToolTrie orderings
depend on preceding requests, this task bootstrap does not cover alternative
request sequences; full planner replays over predeclared sequence replicates are
needed for population-level uncertainty. The n=100 no-tool difference was
0.00pp; the n=1000 point estimate was -4.00pp on a nested sample.

**ToolTrie beats the simple and fitted causal baselines, but not either
measured ContextPilot-derived policy.** The historical padded-menu comparison
used the static-refit adaptation at `alpha=0.5`; vLLM Qwen3-0.6B, 200 requests,
3 trials:

| condition | regime | BFCL cached | ToolRet cached |
| --- | --- | --- | --- |
| contextpilot_causal *(static-refit, alpha=0.5)* | causal | **96.16%** | **94.82%** |
| tooltrie_v0 | causal | 87.19% | 83.58% |
| alphabetical | causal | 38.13% | 51.05% |
| tooltrie_offline | offline | 29.96% | 43.20% |
| cacheweaver | causal | 1.19% | 22.46% |

Two results that overturned earlier drafts:

1. **The adapter's lead is not future-batch visibility.** Restricting static
   refitting to observed requests costs only 0.48pp (BFCL), and that emitted
   order beats ToolTrie-v0 by ~9pp. It is a causally valid ordering adaptation,
   not an evaluation of the official persistent online API.
2. **ToolTrie's causality is the source of its benefit.** Given the whole batch
   it collapses to 29.96%, below alphabetical. Early requests establish a path
   that later ones follow; treating the batch atemporally destroys that
   self-reinforcement.

The later confirmation used the official persistent ordering API at
`alpha=0.001` and measured the same selected sets under realistic BM25 menus:

| k | original | alphabetical | ToolTrie-v0 | ContextPilot persistent API |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 15.87% | 15.28% | 17.48% | **18.72%** |
| 16 | 6.12% | 6.27% | 7.77% | **9.93%** |
| 64 | 0.91% | 1.24% | 1.90% | **4.78%** |
| 128 | 0.37% | 0.58% | 1.13% | **1.99%** |

This persistent arm is still an ordering-only adaptation, not full ContextPilot.
It leads ToolTrie in every retrieved cell, but the absolute ratios show that
ordering cannot manufacture overlap among independently retrieved menus.

**CacheWeaver is a no-op on BFCL, but not on ToolRet.** The Algorithm-1
reimplementation returned the unmodified input order on 200/200 BFCL requests
and 180/200 ToolRet requests. Exact deterministic reconstruction now shows that
the five fitted labels (frequency, schema-cost, FP-tree, pair, triple) emit one
byte-identical 200-request sequence on BFCL, not five independent policies. On
ToolRet there are three distinct fitted sequences: frequency/pair/triple are
identical, while schema-cost and FP-tree each differ. See
`reports/tooltrie-phase2/fitted-policy-equivalence.json`.

**A systematic quality trade-off.** At n=800 on 8B, ranking orderings by
function-name accuracy reverses their ranking on no-tool accuracy. Alphabetical
is worst at selection (82.81%) and best at declining (89.38%); the
ContextPilot-derived adapters show the reverse. ToolTrie sits mid-curve with a
smaller no-tool cost (-1.88pp) than CacheWeaver (-6.88pp), offline ContextPilot
(-6.88pp) or the **causal static-refit adaptation
(-7.50pp, the largest penalty measured, discordant 12:0)** — all three CIs
exclude zero in the fixed-sequence analysis. This is an association among the
measured orderings, not evidence that every reuse-optimizing ordering must incur
the same trade-off.

**The historical SGLang arm shows the same ordering ranking** (87.11% vLLM vs
87.29% SGLang for ToolTrie on BFCL). Its replacement audit now accepts 72/72
raw runs against the independent aggregate server counter. This validates
counter cleanliness, not condition distinctness. Cached ratios remain only the
closest cross-engine summary: the engines render different prompts, with
SGLang adding 640 tokens per request in this workload.

**Both benchmarks ship invalid JSON Schema** - 74 BFCL and 187 ToolRet tools use
Python type names, plus one ToolRet tool colliding with the reserved `title`
annotation. vLLM tolerates it; SGLang rejects every request. See
`reports/tooltrie-phase2/schema_sanitizer.py`.

## Task B detail

RTX 3090 (23.56 GiB), vLLM 0.26.0, `Qwen/Qwen3-0.6B`, block size 16, 11,807 GPU
blocks. Five trials per scenario, prefix cache reset before each trial. Results
in `cluster/results/`.

| Brief check | Status | Evidence |
| --- | --- | --- |
| 1. Identical prompts reuse prefix | Pass | 288/303 prompt tokens cached (95.0%) |
| 2. Changed tool misses after change point | Pass | drops to 128/309 (41.4%) |
| 3. Reorder changes reusable prefix | Pass | drops to 48/303 (15.8%) |
| 4. Cache on/off output equality | Pass | 288 vs 0 cached tokens, identical output |
| 5. Records length/cached/prefill/TTFT/GPU memory | Pass | plus inter-token latency and peak KV usage |

Check 4 required a rerun. The first attempt started its control without
`--enable-prefix-caching`, which does not disable the feature in vLLM V1, so
both servers ran cache-enabled and the "equality" was vacuous. The control now
serves `enable_prefix_caching=False` and reports 0 cached tokens everywhere;
`scripts/compare_probe_runs.py` refuses to report equality unless it can verify
both conditions from the results file.

### Prefix reuse yields no TTFT gain *at this prompt size*

Superseded in scope by the prefill sweep below: the null result here is a
measurement floor, not a property of prefix caching. At 200-tool menus the same
mechanism gives a 10.6x TTFT reduction.

| Scenario | Reuse | TTFT cache-on (ms) | TTFT cache-off (ms) |
| --- | --- | --- | --- |
| original_cold | 0.0% | 52.4 ± 28.8 | 51.5 ± 27.3 |
| original_identical | 95.0% | 43.0 ± 1.2 | 40.5 ± 2.3 |
| changed_second_tool | 41.4% | 41.4 ± 1.1 | 42.3 ± 2.2 |
| reordered_first_two | 15.8% | 37.7 ± 5.7 | 43.0 ± 4.2 |
| original_restored | 95.0% | 41.1 ± 2.1 | 39.4 ± 2.8 |

Intervals are 95% Student-t half-widths over five trials. No scenario separates
cache-on from cache-off, and reuse fraction does not order TTFT: the 15.8%-reuse
case has the lowest mean. The wide `original_cold` interval is one slow
first-ever request (93.9 ms, then 41.8/42.7/42.5/41.3 ms); after that warmup,
cold and 95%-reuse requests cost the same.

This is a measurement-floor result, not evidence against the hypothesis. A
303-token prompt prefills in roughly 40 ms end-to-end, which is dominated by
fixed per-request overhead, so removing 288 tokens of prefill work is not
resolvable.

The earlier single-trial run appeared to show a 3x TTFT gain (69 ms cold vs
23 ms warm). That was the first-request warmup artifact, and it is the concrete
reason repeated trials are mandatory here.

## Task E — measurements on padded menus

Benchmark tasks expose a median of one tool, which is both unreorderable and
below the floor above. Menus are therefore one gold tool plus distractors drawn
from a fixed global catalog, matching a deployment that keeps the same connected
tools loaded across requests.

### Prefill sweep: prefix caching pays once catalogs are realistic

| Menu tools | Prompt tokens | Cold TTFT (ms) | Warm TTFT (ms) | Warm, cache off (ms) | Speedup |
| --- | --- | --- | --- | --- | --- |
| 1 | 250 | 25.6 | 23.4 | 23.1 | 1.0x |
| 4 | 433 | 25.2 | 16.9 | 23.7 | 1.4x |
| 16 | 1,771 | 62.1 | 35.3 | 56.5 | 1.6x |
| 64 | 6,742 | 248.8 | 54.1 | 245.6 | 4.5x |
| 128 | 13,422 | 647.8 | 85.2 | 640.7 | 7.5x |
| 200 | 20,627 | 1282.7 | 120.7 | 1275.2 | 10.6x |

The cache-disabled control stays flat (cold within noise of warm at every size),
so the gap is caching. Crossover is around 4 tools / 433 prompt tokens. Every
ordering experiment must run above it.

Caveat on method: within-run cold-vs-warm separation is not sufficient evidence.
The control "separates" at one tool where no cache exists, so the cache-on
versus cache-off comparison is the one to trust.

### Ordering: the gold-only recommendation inverts

| Ordering, 64-tool menus | Measured cached tokens | Measured reuse |
| --- | --- | --- |
| alphabetical | 526,432 | 38.15% |
| frequency | 60,864 | 4.41% |

Task D recommended frequency ordering from an analytical model over gold-only
menus. On padded menus it is nearly 9x worse. Ordering by benchmark support
ranks the task-specific gold tools first and pushes the shared catalog behind
them, so the common prefix is destroyed. A stable global order that ignores
task-specificity keeps the shared catalog in front. Frequency ordering only wins
when the entire menu is task-specific, which is not what a connected catalog
looks like.

### Trie model calibration

The analytical estimate **under**-predicts measured cache hits: 1.23x for
alphabetical, 1.50x for frequency. It ignores the chat template and system
preamble, which are identical across requests and cached, and that outweighs the
partial blocks it loses at tool boundaries. Calibration is tighter where reuse
is high, so the model is most trustworthy exactly where it matters.

### Ordering does not win on both axes

100 BFCL tasks (20 per category, 64-tool menus), scored against BFCL
`possible_answer` ground truth (`src/tatm/bfcl_score.py`):

| Ordering | Function-name accuracy | Full accuracy | No-tool accuracy |
| --- | --- | --- | --- |
| alphabetical | 67.5% | 47.5% | 95.0% |
| frequency | 77.5% | 51.25% | 85.0% |

Frequency ordering — nearly 9x worse for cache reuse — scores higher on
function-selection accuracy; alphabetical is markedly better at correctly
declining when no tool applies. Neither ordering dominates once quality is
included. One run, no repeats, small per-category samples (n=20); read as
directional, not final. This is what closes brief question 7 for the first
time, at least partially.

**Update, same workload replayed on `Qwen/Qwen3-8B`:** the name/full-accuracy
gap above is model- or sample-sensitive. At 8B both orderings score identically
on function-name (91.25%) and full accuracy (81.25%) — the gap observed at
0.6B vanishes for this checkpoint and workload. The no-tool-accuracy gap
survives, smaller: 95.0% vs 90.0% at 8B (was 95.0% vs 85.0% at 0.6B), same
direction both times. Revised reading: alphabetical does not show a resolved
selection-accuracy cost across models; the remaining caution is narrower and
concerns correctly declining irrelevant requests.

## Task D — locality is now measurable

`trie_metrics` retained every node forever, making reuse depend only on the
multiset of requests and not their order; `session_bursty` was therefore
identical to `empirical` by construction, and the skewed-replay gap came from
resampling with replacement rather than locality. `bounded_trie_metrics` adds
capacity with leaf-first LRU eviction, and the replays now diverge. ToolRet's
natural file order turns out to be *more* local than the synthetic bursty replay
(31.35% vs 27.10% reuse at 25% capacity), because its file order is already
99.86% same-domain adjacent.

**Checked on the live GPU cache** (`scripts/locality_replay.py`, one continuous
session per replay condition): the old run measured 287,200 cached tokens for
`empirical` and 287,616 for `session_bursty`, a 0.14% difference. That supports
near-parity between these two request orders after shared-catalog padding and
alphabetical ordering. However, its 668,440 denominator was canonical tool
tokens, not rendered prompt tokens, and occupancy was not sampled. Therefore
the previously reported 42.97%/43.03% values are diagnostic ratios, not vLLM
cache-hit rates, and the claim of directly observed eviction is withdrawn. The
new format-v2 harness fixes the denominator, validates it against vLLM's query
counter, samples KV occupancy, and can enforce a predeclared pressure threshold.

## Verified locally

- 44,453 ToolRet tools and 7,961 ToolRet tasks;
- 1,362 canonical BFCL functions and 1,240 BFCL tasks;
- 45,815 total schemas tokenized with `Qwen/Qwen3-0.6B`;
- all ToolRet label references resolve to the downloaded corpus;
- The full local test suite passes; run `uv run pytest` to reproduce it.

## Reports are generated, not hand-written

Top-level `reports/*.md` are produced by `src/tatm/reporting.py`. Measured GPU
results are normally read from `cluster/results/` by `load_probe_results`.
Because those raw files are intentionally git-ignored and may exist only on the
cluster, CPU-only regeneration now preserves an already committed measured GPU
section instead of replacing it with "not yet measured." Other generated
sections are still overwritten. `PROJECT_STATUS.md` and `cluster/README.md` are
hand-maintained.

## Initial brief substantively executed

The explicit Tasks A-F in `initial-research-brief.md` now have reported
evidence, and the later stricter gap-closure execution manifest is accepted:
retrieved-menu replay, ordinary fallback, direct partial reuse, exact
rendered-prefix auditing, and controlled memory pressure are all measured. This
completes the planned initial experiment stage; it does not retroactively make
a §9 retained-tool or KV-composition extension part of the result. Formal
publication-grade closure still requires corrected analysis paths and an
artifact-to-report audit. Initial experimental question 4 is also partial:
schema-token weighting was tested, but separately measured per-schema prefill-
time weighting was not.

The corrected GPU handover is commit `6378d78`, with measurements executed at
the pinned `65b86ba`. The only mid-run worktree change was the documentation-only
addition of `AGENTS.md`; the measured code, data, scripts, model, and server
configuration did not change. Validation found:

- 84/84 primary retrieved-menu replays clean (4 menu sizes x 7 conditions x 3
  trials, 200 requests each);
- identical selected membership and request sequence across all orderings;
- 7/7 exact k64 rendered-token/block audits clean;
- direct partial-reuse strata present for both predeclared k64 conditions;
- a readable, checksummed 221-entry raw archive, with invalid and contended
  attempts retained separately in quarantine;
- 24/24 controlled-pressure regime-runs accepted with 384/384 checks passing,
  positive sampled evictions, zero preemptions, and sequential execution.

The BM25 retrieval artifact is `reports/retrieval-bm25-sweep.json`. Selection
never reads evaluation gold IDs; this remains a lexical baseline, not the
official ToolRet retriever or a production trace.

| BM25 cutoff | Macro recall | Hit rate | MRR |
| --- | --- | --- | --- |
| 4 | 41.71% | 51.50% | 0.3858 |
| 16 | 55.04% | 65.50% | 0.4022 |
| 64 | 64.21% | 75.50% | 0.4061 |
| 128 | 67.54% | 80.50% | 0.4066 |

These values show why the gold/exposed and retrieved arms cannot be mixed: even
at 128 tools, 19.5% of queries retrieve none of their gold tools and macro
recall is only 67.5%. Ordering cannot repair a missing tool.

Measured rendered-prompt cache reuse on the same menus is:

| k | Original fallback | Alphabetical | Random 42 | Frequency | Schema-cost | FP-tree global | ToolTrie-v0 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 15.87% | 15.28% | 14.17% | 14.62% | 14.84% | 14.62% | **17.48%** |
| 16 | 6.12% | 6.27% | 5.39% | 5.59% | 6.24% | 5.59% | **7.77%** |
| 64 | 0.91% | 1.24% | 0.96% | 0.96% | 1.09% | 0.96% | **1.90%** |
| 128 | 0.37% | 0.58% | 0.59% | 0.54% | 0.57% | 0.54% | **1.13%** |

ToolTrie is best at every retrieved menu size, but only by 0.76-1.65 percentage
points over ordinary text prefill. No resolved TTFT improvement follows: the
three-trial intervals overlap and no paired-difference interval was declared.
The BM25-retrieved-set overlap, rather than the padded shared-catalog result,
is therefore the limiting regime for deployment claims.

The first pressure matrix completed cleanly, but 0/24 regime-runs met the
predeclared 90% occupancy threshold. Peak occupancy was only 3.6379-3.6882%:
one sequential approximately 7k-token request in a 190,896-token cache. Those
runs are valid low-occupancy evidence but cannot support eviction or
memory-under-pressure claims.

The separately predeclared controlled-cache rerun fixes capacity at 480
blocks/7,680 tokens while preserving the six labels, four regimes, sequential
request order, and 90% threshold. It passes 24/24 regime-runs and 384/384
checks. Peak occupancy is 91.02-91.86%, sampled evictions are
57,696-85,340, preemptions are zero, and peak running/waiting requests are 1/0.
Cross-capacity latency comparison remains forbidden.

Within that controlled stress condition, fixed random seed 42 ranks above
alphabetical on reuse in all four regimes, while original retrieval order is
worst. This is a single-run, single-seed sensitivity finding without an
uncertainty interval, not a recommendation to use random ordering. A
deterministic reconstruction also confirms that frequency,
schema-cost weighted, and FP-tree global emit identical 200-request sequences
in every regime, so the 24 executions represent four distinct orderings rather
than six independent policies. See
`reports/initial-brief-pressure-rerun/ordering-equivalence.json`.

The local audit has repaired metric-specific quality scoping, explicit
sequence-state metadata, fail-closed cache reset checks, and independent SGLang
counter validation. The 18 static-refit systems replays and one 8B quality
replay are now tracked as compact evidence. Publication-grade closure still
needs verified off-machine copies of the server-only archives and raw per-case
verification of the executor's ContextPilot API-identity claim. The historical
Qwen3-8B server was not pinned to a model revision, so adding one new
`original` row is controlled only if the exact cached snapshot used by the old
rows can be proved. Otherwise run a fresh pinned five-condition 8B matrix or
omit that optional 8B fallback comparison. The accepted 4B-primary matrix
already contains the fallback, and no accepted dual-model replay needs to be
repeated.

The next extension may evaluate safe inactive-tool retention, but it must retain
ordinary selected-tool text prefill as fallback and compare against both
clearly labelled ContextPilot adaptations, the no-retention baselines, and
multiple random seeds.

Analytical reuse estimates must still be labelled as estimates, but they are no
longer unvalidated: they under-predict measured hits by 1.2-1.5x on the two
workloads checked.
