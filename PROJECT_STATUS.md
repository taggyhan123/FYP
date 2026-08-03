# Project status

Status after the first local research pass:

| Brief task | Status | Artifact / remaining work |
| --- | --- | --- |
| A — reading note | Complete | `notes/reading-note.md` |
| B — exact prefix caching | Complete; all five checks measured on GPU | See "Task B detail" below |
| C — normalize datasets | Complete for ToolRet and five BFCL V4 static subsets | `scripts/download_datasets.py`, `scripts/run_pipeline.py`, `reports/dataset-inventory.md` |
| D — access patterns | Complete for benchmark evidence and four controlled replays | `reports/access-patterns.md` and `reports/tables/` |
| E — exact ToolTrie baseline | Measured on GPU: prefill sweep, crossover, and two orderings validated against real cache hits. Causal ToolTrie-v0 now measured end to end against both static orderings | `scripts/prefill_sweep.py`, `scripts/validate_reuse_estimate.py`, `src/tatm/tooltrie.py`, "Task E" and "ToolTrie-v0" below, `reports/tooltrie-v0/findings.md` |
| F — initial report | Complete and regenerated from measured results | `reports/initial-findings.md` |

## Notable findings

The counterintuitive results, in order of how surprising they were:

1. **The obvious ordering choice is close to the worst one.** Sorting tools by
   how often they're needed (frequency) is nearly 9x worse for cache reuse than
   plain alphabetical order (4.41% vs 38.15% measured). Frequency ranks each
   request's own gold tool near the front — but the gold tool is the one thing
   that changes request to request. It optimizes for "important" and
   accidentally destroys "identical," which is the only thing the cache
   rewards.
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
   offline and found a real gap (31.35% vs 27.10%). Checked on the live GPU
   cache under genuine eviction (668k tokens vs 191k capacity, ~2,920
   evictions): the gap nearly disappears (42.97% vs 43.03%), because applying
   the winning ordering plus shared-catalog padding makes almost every
   request's prefix identical regardless of task clustering, swamping the
   locality signal the offline-only result found. Both offline predictions
   were right for the setup they described; the setups just weren't the same.
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
**−3.75pp** and **−6.25pp** at 8B. The sign flips with model scale. This is the
third time in this project that a 0.6B quality reading did not survive to 8B,
but the first time the direction reverses rather than merely shrinking, so the
0.6B run alone would have reported ToolTrie as quality-free.

Against the predeclared one-percentage-point regression threshold, ToolTrie-v0
**does not pass** at 8B despite winning decisively on every systems metric. Two
cautions: the gate is below its own resolution (n=100 means 1pp is one task, and
the observed gap is 6 tasks), and the quality runs are single-pass. The 8B
planner ran at a 44,656-token capacity versus 190,896 at 0.6B and evicted 424
nodes, yet produced byte-identical orderings on all 100 records, so this is a
clean model-size comparison with no capacity confound.

### Unrelated traffic during the first attempt

Two replay drivers issued requests against the same server between 01:27 and
01:43. Because `vllm:prompt_tokens_cached` is a global counter, each run's
metric window also captured the other's hits, producing impossible cached ratios
above 100%. A trial is accepted only when `vllm:prefix_cache_queries` for its
window equals its own prompt-token total; the 10 failing replays are quarantined
and excluded, and their conditions were re-run single-driver. All 18 trials
behind the table above pass that check. See
`reports/tooltrie-v0/contamination-incident.txt`.

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
gap above was mostly a small-model artefact. At 8B both orderings score
identically on function-name (91.25%) and full accuracy (81.25%) — the gap that
looked real at 0.6B vanishes at deployment-grade model capability. The
no-tool-accuracy gap survives, smaller: 95.0% vs 90.0% at 8B (was 95.0% vs
85.0% at 0.6B), same direction both times. Revised reading: alphabetical does
not cost selection accuracy the way the 0.6B run suggested; the caution that
remains is narrower and only about correctly declining irrelevant requests.

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
session per replay condition, no resets between requests, so real eviction can
happen): 120 padded/alphabetically-ordered BFCL tasks, 668,440 total tokens vs
190,896 real cache capacity — genuine eviction occurred (~2,920 evictions per
condition). Measured reuse for `empirical` vs `session_bursty` was 42.97% vs
43.03%, essentially no gap, matching the offline model's own prediction of
35.00% vs 35.03% for this exact setup. The offline finding above still holds
for raw, unordered task sequences; it does not survive once menus are padded
from a shared catalog and forced into the ordering that already wins on
reuse — that ordering dominates request-to-request similarity and swamps
whatever weaker signal session clustering would otherwise contribute. Measured
reuse also exceeded prediction by 1.228x in both conditions, independently
matching the 1.23x calibration factor from the static Task E validation.

## Verified locally

- 44,453 ToolRet tools and 7,961 ToolRet tasks;
- 1,362 canonical BFCL functions and 1,240 BFCL tasks;
- 45,815 total schemas tokenized with `Qwen/Qwen3-0.6B`;
- all ToolRet label references resolve to the downloaded corpus;
- 47 unit tests pass.

## Reports are generated, not hand-written

`reports/*.md` are produced by `src/tatm/reporting.py` and overwritten on every
`scripts/run_pipeline.py` run. Anything typed directly into them is destroyed.
Measured GPU results are read back from `cluster/results/` by
`load_probe_results` and rendered into the findings report, so they survive
regeneration. `PROJECT_STATUS.md` and `cluster/README.md` are the only
hand-maintained documents.

## Evidence still required

- rendered full-prompt token IDs and exact vLLM block boundaries;
- BFCL quality under the remaining four orderings, and repeated trials with
  larger per-category samples for the three already scored — the current
  quality numbers are a single, small-sample run. **This is now the highest
  priority item:** it is the only thing standing between the ToolTrie-v0
  systems result and a defensible claim, since the 4-6pp regression measured
  against alphabetical at 8B rests on one pass of 100 tasks and a threshold
  finer than the sample can resolve;
- ordering comparisons across all six orderings above the crossover, not just
  the two validated so far;
- live-cache eviction under other orderings and other replay pairs (`uniform`
  and `skewed` were excluded above since they resample with replacement);
  actual GPU/KV memory *usage* under eviction pressure, as opposed to
  eviction's effect on reuse, which is now measured.

Analytical reuse estimates must still be labelled as estimates, but they are no
longer unvalidated: they under-predict measured hits by 1.2-1.5x on the two
workloads checked.
