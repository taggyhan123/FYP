# First exact ToolTrie baseline — one-page proposal

## Objective

Test whether deterministic ordering of an already-selected tool set increases
native vLLM exact-prefix reuse without changing the tool text, active tool set,
or KV tensors.

## Inputs and invariant

For every task, freeze the selected tool IDs before the cache experiment. Use
gold ToolRet relevance sets for retrieval-isolated analysis and BFCL exposed
menus for function-calling quality. Serialize every tool with the same canonical
function shape, then let one fixed model chat template render the request.

The invariant is:

```text
same task + same selected tools + same schema content
```

Only the tool order changes. Ordinary selected-tool text prefill remains the
fallback.

## Baselines

Compare original, alphabetical, three fixed global-random orders, descending
benchmark support, support × schema-token cost, and FP-tree global support
order. The last two frequency-named baselines are expected to be identical
until conditional pattern mining is introduced.

## Workloads

Run empirical benchmark order, uniform replay with replacement, controlled
support-skewed replay, and domain-grouped session-bursty replay. Keep ToolRet and
BFCL conclusions separate because their evidence types differ.

## System design

The local planner maps selected IDs to one deterministic order and writes
OpenAI-compatible request JSONL. A stock vLLM server with APC enabled receives
the requests. No attention kernel, cache block, or KV tensor is modified.
Prometheus and API metrics are collected before/after each request.

## Measurements

Report:

- complete rendered prompt tokens and selected schema tokens;
- prefix cache hits/queries;
- cached prompt tokens and computed prefill tokens;
- prefill time, TTFT, and end-to-end latency;
- KV/GPU memory and eviction settings;
- valid tool name, argument correctness, no-tool correctness, and exact output
  equality for cache-on/off requests with the same token sequence.

Use repeated trials, documented warm-up/cache-reset policy, confidence intervals,
and the same model revision/server flags.

## Decision criterion

Continue toward weighted trie admission only if at least one realistic
multi-tool regime shows repeatable cached-token and TTFT improvement without a
quality regression. If gains appear only in synthetic bursty replay, narrow the
claim to workload-aware scheduling. If one-tool tasks dominate, focus on menu
construction or repeated workflow traces rather than ordering.

## Cluster boundary

Local work produces canonical schemas, workload JSONL, analytical estimates,
and tests. CUDA vLLM is required for cache/latency/memory evidence. Commands and
metrics are listed in `cluster/README.md`.
