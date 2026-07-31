# GPU procedure: exact prefix-cache experiments

Tasks B and E require a CUDA-capable vLLM server. The local pipeline prepares
schemas and workloads without claiming GPU measurements.

## 1. Environment

The workstation itself has four RTX 3090s and a vLLM 0.26.0 environment at
`/home/taghan/tatm/.venv`, so these steps run locally; no separate cluster is
needed. Pick an idle GPU with `nvidia-smi` and pin it — other users share this
machine.

Two environment prerequisites, both discovered the hard way:

- `CPATH` must point at Python 3.12 headers. Triton JIT-compiles a small
  `cuda_utils.c` and the system `python3.12-dev` package is not installed, so
  startup dies with `fatal error: Python.h: No such file or directory` whenever
  the torch compile cache misses. Headers are provided by the `hdr312` conda
  env.
- `VLLM_SERVER_DEV_MODE=1` is required to expose `POST /reset_prefix_cache`,
  which the probe uses to make each trial genuinely cold.
- The venv's `bin` must be on `PATH`, not just used to resolve the `vllm`
  executable. FlashInfer JIT-builds its sampling kernel by shelling out to
  `ninja`, which only exists inside the venv; without it startup dies with
  `FileNotFoundError: 'ninja'` during KV-cache initialization.

Record the exact model revision, vLLM version, GPU type, block size, dtype,
maximum model length, tensor parallelism, and every server flag.

## 2. Start the cache-enabled server

```bash
PATH=/home/taghan/tatm/.venv/bin:$PATH \
VLLM_SERVER_DEV_MODE=1 \
CUDA_VISIBLE_DEVICES=2 \
CPATH=/home/taghan/miniconda3/envs/hdr312/include/python3.12 \
vllm serve Qwen/Qwen3-0.6B \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --host 127.0.0.1 \
  --port 8000
```

## 3. Run the black-box sanity probe

```bash
python scripts/vllm_prefix_cache_probe.py \
  --run-label cache-enabled \
  --repeats 5 \
  --output cluster/results/cache-enabled.json
```

Each trial resets the prefix cache and then walks the fixed sequence:

1. an initially cold prompt;
2. the identical prompt;
3. one changed definition in the second position;
4. the first two tools reordered;
5. the original prompt restored.

`--repeats` controls how many trials run. One trial produces point estimates
with no error bars, which is not enough to compare latencies; the summary block
reports mean, standard deviation, and a 95% Student-t half-width per scenario.

The probe records API usage, wall time, tool-call output, and deltas for the
vLLM Prometheus metrics present in the installed version: prefix cache
hits/queries, cached prompt tokens, prefill time, TTFT, inter-token latency, and
decode time. It also reads back `vllm:cache_config_info` so the served
`enable_prefix_caching` value is stored with the results rather than assumed.

KV-cache usage is sampled by a background thread *while each request is in
flight*. `vllm:kv_cache_usage_perc` is an instantaneous gauge, so scraping it
only after a request returns always reads ~0 — the blocks have already been
freed. Peak values land in `peak_gauges`.

## 4. Run the cache-disabled control

Stop the server, restart it with identical settings but with prefix caching
explicitly turned off, and run:

```bash
PATH=/home/taghan/tatm/.venv/bin:$PATH \
VLLM_SERVER_DEV_MODE=1 \
CUDA_VISIBLE_DEVICES=2 \
CPATH=/home/taghan/miniconda3/envs/hdr312/include/python3.12 \
vllm serve Qwen/Qwen3-0.6B \
  --no-enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --host 127.0.0.1 \
  --port 8000
```

`--no-enable-prefix-caching` is required. Merely omitting `--enable-prefix-caching`
does **not** disable it: vLLM V1 defaults `enable_prefix_caching=True`, so the
control silently runs with caching on. This exact mistake invalidated the first
attempt at check 4.

Then run:

```bash
python scripts/vllm_prefix_cache_probe.py \
  --run-label cache-disabled \
  --repeats 5 \
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

## 5. Locate the measurement floor before any ordering experiment

```bash
python scripts/prefill_sweep.py \
  --run-label sweep-cache-enabled \
  --repeats 5 \
  --output cluster/results/prefill-sweep-enabled.json
```

Repeat against a `--no-enable-prefix-caching` server with
`--run-label sweep-cache-disabled`. Trust the cache-on versus cache-off
comparison, not within-run cold-versus-warm separation: the control "separates"
at one tool, where no cache exists.

Ordering results below the crossover are not interpretable.

## 6. Validate the analytical estimate against real cache hits

```bash
python scripts/build_cluster_workload.py \
  --partition bfcl --ordering alphabetical \
  --menu-size 64 --limit 200 \
  --output data/processed/bfcl-alpha-menu64.jsonl

python scripts/validate_reuse_estimate.py \
  --input data/processed/bfcl-alpha-menu64.jsonl \
  --run-label validate-menu64-alphabetical \
  --output cluster/results/validate-menu64-alphabetical.json
```

`--menu-size` pads each task's gold tools with distractors from a fixed global
catalog. Without it the median task exposes one tool, which cannot be reordered
and sits below the crossover.

## 7. Generate a benchmark workload

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
