# Project status

Status after the first local research pass:

| Brief task | Status | Artifact / remaining work |
| --- | --- | --- |
| A — reading note | Complete | `notes/reading-note.md` |
| B — exact prefix caching | Harness complete; measurement blocked locally | Run `scripts/vllm_prefix_cache_probe.py` on the CUDA SoC cluster |
| C — normalize datasets | Complete for ToolRet and five BFCL V4 static subsets | `scripts/download_datasets.py`, `scripts/run_pipeline.py`, `reports/dataset-inventory.md` |
| D — access patterns | Complete for benchmark evidence and four controlled replays | `reports/access-patterns.md` and `reports/tables/` |
| E — exact ToolTrie baseline | Local planner/workload builder and replay client complete; serving run pending | `src/tatm/prompting.py`, `scripts/build_cluster_workload.py`, `scripts/replay_vllm_workload.py`, `cluster/README.md` |
| F — initial report | Local findings complete; GPU section explicitly pending | `reports/initial-findings.md` |

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
