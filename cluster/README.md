# CUDA cluster hand-off: exact prefix-cache experiments

Tasks B and E require a CUDA-capable vLLM server. The local pipeline prepares
schemas and workloads without claiming GPU measurements.

## 1. Copy or clone the repository on the SoC cluster

Use the cluster's approved storage and environment. Do not commit credentials,
tokens, model weights, raw datasets, or `cluster/results/`.

## 2. Start the cache-enabled server

Use the model and vLLM module/version approved for the cluster. A representative
command is:

```bash
vllm serve Qwen/Qwen3-0.6B \
  --enable-prefix-caching \
  --host 127.0.0.1 \
  --port 8000
```

Record the exact model revision, vLLM version, GPU type, block size, dtype,
maximum model length, tensor parallelism, and every server flag.

## 3. Run the black-box sanity probe

```bash
python scripts/vllm_prefix_cache_probe.py \
  --run-label cache-enabled \
  --output cluster/results/cache-enabled.json
```

The fixed sequence checks:

1. an initially cold prompt;
2. the identical prompt;
3. one changed definition in the second position;
4. the first two tools reordered;
5. the original prompt restored.

The probe records API usage, wall time, tool-call output, and deltas for the
vLLM Prometheus metrics that exist in the installed version, including prefix
cache hits/queries, cached prompt tokens, computed prefill KV tokens, prefill
time, TTFT, and KV-cache usage.

## 4. Run the cache-disabled control

Stop the server, restart it with identical settings but without
`--enable-prefix-caching`, and run:

```bash
python scripts/vllm_prefix_cache_probe.py \
  --run-label cache-disabled \
  --output cluster/results/cache-disabled.json
```

Compare the response token sequence and structured tool calls for each scenario.
If exact token equality is required, save token IDs from an endpoint/version
that exposes them. Text equality alone is a weaker check.

The included comparison performs the weaker projected-output check:

```bash
python scripts/compare_probe_runs.py \
  --enabled cluster/results/cache-enabled.json \
  --disabled cluster/results/cache-disabled.json
```

## 5. Generate a benchmark workload

After the local normalization pipeline has run:

```bash
python scripts/build_cluster_workload.py \
  --partition bfcl \
  --ordering frequency \
  --limit 500 \
  --output data/processed/bfcl-frequency.jsonl
```

Repeat with `original`, `alphabetical`, `random`,
`schema_cost_weighted`, and `fp_tree_global`. Use a fresh server/cache state or a
documented warm-up policy for each comparison. The workload records use the
OpenAI `tools` shape and preserve a deterministic canonical order.

Replay one generated file with per-request and aggregate metric collection:

```bash
python scripts/replay_vllm_workload.py \
  --input data/processed/bfcl-frequency.jsonl \
  --run-label bfcl-frequency-cache-enabled \
  --output cluster/results/bfcl-frequency-cache-enabled.json
```

Repeat the complete workload under each ordering and cache condition. Generate
all compared files from the same task partition and limit.

## 6. Capture the cluster environment

```bash
python -c 'import platform, torch, vllm; print(platform.platform()); print(torch.__version__); print(vllm.__version__)'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
curl -s http://127.0.0.1:8000/metrics > cluster/results/metrics-snapshot.txt
```

Also save the scheduler/server command, model commit, tokenizer files, chat
template, tool-call parser, CUDA version, and job allocation.

## Measurement rules

- Report prompt tokens, cached prompt tokens, computed prefill tokens, prefill
  time, TTFT, end-to-end latency, output quality, and GPU/KV memory together.
- Separate cold, warm, and steady-state observations.
- Run enough repetitions for confidence intervals; do not compare one-off wall
  times.
- Use the same request token sequence for cache-on/off semantic equivalence.
- Treat the local trie numbers as hypotheses, not vLLM measurements.
- Preserve normal selected-tool text prefill as the semantics-preserving
  fallback.
