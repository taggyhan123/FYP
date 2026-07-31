# Project status

Status after the first local research pass:

| Brief task | Status | Artifact / remaining work |
| --- | --- | --- |
| A — reading note | Complete | `notes/reading-note.md` |
| B — exact prefix caching | Complete; all five checks measured on GPU | See "Task B detail" below |
| C — normalize datasets | Complete for ToolRet and five BFCL V4 static subsets | `scripts/download_datasets.py`, `scripts/run_pipeline.py`, `reports/dataset-inventory.md` |
| D — access patterns | Complete for benchmark evidence and four controlled replays | `reports/access-patterns.md` and `reports/tables/` |
| E — exact ToolTrie baseline | Local planner/workload builder and replay client complete; serving run pending | `src/tatm/prompting.py`, `scripts/build_cluster_workload.py`, `scripts/replay_vllm_workload.py`, `cluster/README.md` |
| F — initial report | Local findings complete; GPU section explicitly pending | `reports/initial-findings.md` |

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

### Prefix reuse does not yield a TTFT gain at this prompt size

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
resolvable. Task E must use tool catalogs large enough that schema prefill
dominates TTFT before any ordering claim can be tested. Establishing that
threshold is now the first thing Task E should measure.

The earlier single-trial run appeared to show a 3x TTFT gain (69 ms cold vs
23 ms warm). That was the first-request warmup artifact, and it is the concrete
reason repeated trials are mandatory here.

## Verified locally

- 44,453 ToolRet tools and 7,961 ToolRet tasks;
- 1,362 canonical BFCL functions and 1,240 BFCL tasks;
- 45,815 total schemas tokenized with `Qwen/Qwen3-0.6B`;
- all ToolRet label references resolve to the downloaded corpus;
- eight unit tests pass;
- a ten-request OpenAI-compatible BFCL workload smoke test succeeds.

## Cluster-only evidence still required

- rendered full-prompt token IDs and exact vLLM block boundaries;
- prefix cache hits/queries and cached/computed prompt tokens;
- prefill time, TTFT, repeated-trial latency, and confidence intervals;
- GPU/KV memory and eviction behavior;
- cache-on/off output-token equality;
- BFCL tool-name, argument, and no-tool quality under every ordering.

The local analytical reuse estimates must not be described as vLLM cache-hit
rates or latency speedups.
