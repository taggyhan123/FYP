# GPU procedure: exact prefix-cache experiments

Tasks B and E require a CUDA-capable vLLM server. The local pipeline prepares
schemas and workloads without claiming GPU measurements.

Tasks B/E and ToolTrie-v0 have now been measured. For the next targeted
no-tool, CacheWeaver, FP-tree/ContextPilot, and SGLang comparison sequence, use
[`../NUS_GPU_PHASE2_INSTRUCTIONS.md`](../NUS_GPU_PHASE2_INSTRUCTIONS.md).

The supervisor-requested model update was executed with
[`../NUS_GPU_CONTEXTPILOT_DUAL_MODEL_INSTRUCTIONS.md`](../NUS_GPU_CONTEXTPILOT_DUAL_MODEL_INSTRUCTIONS.md)
with [`contextpilot-dual-model-manifest.json`](contextpilot-dual-model-manifest.json).
That protocol makes Qwen3-4B primary and Qwen3-0.6B a separately reported
replication; it must not be folded into the older Phase 2 procedure. Its 190
accepted replays and interpretation are in
[`../reports/contextpilot-dual-model/findings.md`](../reports/contextpilot-dual-model/findings.md).
Do not rerun that matrix unless a new protocol is declared.

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

## 6a. Score function-call quality under an ordering

`build_cluster_workload.py --partition bfcl` samples tasks in dataset order,
which is alphabetically `irrelevance` first — the first 200 tasks are entirely
no-tool cases and cannot test name/argument accuracy. Use the stratified
builder instead:

```bash
python scripts/build_bfcl_quality_workload.py \
  --ordering alphabetical --per-domain 20 --menu-size 64 \
  --output data/processed/bfcl-quality-alphabetical.jsonl

python scripts/replay_vllm_workload.py \
  --input data/processed/bfcl-quality-alphabetical.jsonl \
  --run-label bfcl-quality-alphabetical --max-tokens 128 --disable-thinking \
  --output cluster/results/bfcl-quality-alphabetical.json

python scripts/score_bfcl_quality.py \
  --replay-result cluster/results/bfcl-quality-alphabetical.json \
  --output cluster/results/bfcl-quality-alphabetical-score.json
```

`--disable-thinking` is not optional here: Qwen3 emits a `<think>` block by
default, which consumes the whole `--max-tokens` budget before any tool call is
produced, silently zeroing every score. Scoring requires
`data/raw/bfcl/possible_answer/*.json`, fetched by `download_datasets.py`; the
`irrelevance` config has no such file since its ground truth is "call nothing."

`scripts/score_bfcl_quality.py` is a reduced reimplementation of the official
Gorilla/BFCL AST checker, not a vendored copy — it does not execute code and
treats any predicted argument absent from ground truth as a mismatch.

## 6b. Measure whether request order matters on the live cache

Every experiment above resets the prefix cache before each trial, which is
what makes them repeatable, but it also erases the cross-request dependency
that a locality question is actually about. This runs one continuous session
per replay condition instead — a single reset at the start, then every
request in sequence, no resets in between:

```bash
python scripts/locality_replay.py \
  --run-label locality-replay-bfcl-alpha64 \
  --limit 120 --menu-size 64 --ordering alphabetical \
  --output cluster/results/locality-replay-bfcl-alpha64.json
```

Compares `empirical` against `session_bursty` from `replay_workloads` — the one
pair that is a strict permutation of the same task multiset (`uniform` and
`skewed` resample with replacement, so they differ in content, not just
order). `--limit`/`--menu-size` should produce enough rendered token volume to
challenge the server's real cache capacity (`block_size * num_gpu_blocks`, read
back from `vllm:cache_config_info`). That is necessary but not sufficient
evidence of pressure; only a run that samples and reaches the declared
occupancy threshold supports an eviction-pressure claim.

Before the generic gold/exposed-menu builder below, close the brief's required
retrieval-vs-serving separation with the following three checks.

### 6c. Build a BM25-retrieved-tool workload (CPU; no GPU required)

`build_cluster_workload.py --partition toolret` uses ToolRet gold relevance IDs
as the selected set. That is the correct oracle/gold arm, but it is **not** a
retrieval experiment. Build an independent BM25 top-k menu instead:

```bash
uv run python scripts/build_retrieved_tool_workload.py \
  --menu-size 64 --offset 0 --limit 200 \
  --ordering original \
  --output data/processed/toolret-bm25-k64-original.jsonl
```

The retriever sees only the query and canonical tool corpus. The workload keeps
gold IDs as evaluation metadata after selection, and the companion
`*-retrieval-metrics.json` reports recall, precision, hit rate, MRR, retrieved
frequency, zero-score fallback counts, and ordering-fit provenance. `original`
preserves BM25 rank. The other five brief orderings permute exactly the same
retrieved set; they do not change membership.

Use a disjoint support split (the default) for frequency-based orderings. Never
describe ToolRet benchmark frequency as production popularity.

### 6d. Capture exact rendered tokens and block evidence (GPU server)

```bash
uv run python scripts/audit_rendered_prefix.py \
  --input data/processed/toolret-bm25-k64-original.jsonl \
  --run-label toolret-bm25-k64-rendered-audit \
  --measure --disable-thinking \
  --output cluster/results/toolret-bm25-k64-rendered-audit.json
```

This calls vLLM's server-side `/tokenize` route with the same `messages` and
`tools` sent to chat completion. It stores the exact rendered token IDs,
server-reported block size, every full/partial block boundary, immediate and
best-prior common prefixes, measured cache hits, prefill, and TTFT. The
completion prompt-token count must equal `/tokenize`'s count; a mismatch marks
the run unclean. This is the evidence to use for block claims, not canonical
schema-token estimates.

### 6e. Measure partial reuse and memory under demonstrated pressure

Every normal replay now emits direct cold/partial/full reuse buckets:

```bash
uv run python scripts/replay_vllm_workload.py \
  --input data/processed/toolret-bm25-k64-original.jsonl \
  --run-label toolret-bm25-k64-original \
  --reset-before --disable-thinking \
  --output cluster/results/toolret-bm25-k64-original.json
```

`direct_measurements_by_reuse_bucket` contains measured cached ratio, prefill,
TTFT, and wall time for requests that actually achieved partial reuse; it does
not interpolate between the cold/warm endpoints. Repeat each condition at least
three times and pass all trial files to `summarize_ordering_replays.py`; its
`direct_reuse_buckets` output reports trial-level intervals for the partial
reuse strata as well as the aggregate condition.

Run memory-pressure experiments separately so the background metrics scrape
does not silently become part of the primary latency comparison:

```bash
uv run python scripts/locality_replay.py \
  --run-label bfcl-alpha64-all-regimes-pressure \
  --limit 200 --menu-size 64 --ordering alphabetical \
  --condition empirical \
  --condition uniform \
  --condition skewed \
  --condition session_bursty \
  --require-peak-kv-usage 0.90 \
  --output cluster/results/bfcl-alpha64-all-regimes-pressure.json
```

The output records cache capacity, rendered prompt-token volume, sampled peak/
mean/final occupancy, estimated resident tokens, preemptions, and optional
sampled eviction-residency counters. `--require-peak-kv-usage` fails only after
preserving the output if the workload never reaches the predeclared pressure
level. `empirical` and `session_bursty` are matched permutations;
`uniform`/`skewed` are distribution stress tests that resample with replacement
and are labelled accordingly.

The first initial-brief pressure matrix completed cleanly but reached only
3.64-3.69% occupancy because one sequential approximately 7k-token request ran
against a 190,896-token cache. Those 24 runs remain low-occupancy evidence and
must not be relabelled as pressure. The replacement is separately predeclared in
`initial-brief-pressure-rerun-manifest.json` and executed by
`../NUS_GPU_PRESSURE_RERUN_INSTRUCTIONS.md`: it preserves sequential request
order and the 0.90 threshold while fixing the controlled cache at 7,680 tokens.
The rerun is now accepted 24/24, with 384/384 checks, 91.02-91.86% occupancy,
positive sampled evictions, and zero preemptions. Cross-capacity latency
comparison remains forbidden. See
`../reports/initial-brief-pressure-rerun/20260807-005414/HANDOVER.md`.

### 6f. Record the ordinary text-prefill fallback explicitly

The predeclared fallback for the retrieved arm is the BM25-selected menu in its
original retrieval-rank order, sent through the ordinary OpenAI `tools` field.
It retains no inactive tools and performs no KV-tensor composition or mutation:

```bash
uv run python scripts/replay_vllm_workload.py \
  --input data/processed/toolret-bm25-k64-original.jsonl \
  --run-label toolret-bm25-k64-ordinary-text-prefill \
  --condition-role ordinary_text_prefill_fallback \
  --reset-before --disable-thinking --max-tokens 1 \
  --output cluster/results/toolret-bm25-k64-ordinary-text-prefill.json
```

The replay output records this contract under `execution_condition` and rejects
the fallback label if the input workload has been reordered. The full pinned
matrix, stop conditions, and GPU handover procedure are in
`../NUS_GPU_BRIEF_CLOSURE_INSTRUCTIONS.md` and
`initial-brief-closure-manifest.json`.

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

## 8. Run the causal ToolTrie-v0 workflow

The ToolTrie is prompt-layer CPU metadata. It reorders selected tool IDs and
does not patch vLLM, CUDA, attention, or KV tensors. Build one base workload so
the static and ToolTrie conditions receive exactly the same selected tool sets
and request order:

```bash
python scripts/build_cluster_workload.py \
  --partition bfcl --ordering original --menu-size 64 --limit 200 \
  --output data/processed/bfcl-base-menu64.jsonl

python scripts/build_tooltrie_workload.py \
  --input data/processed/bfcl-base-menu64.jsonl \
  --fallback alphabetical --recency-window 128 \
  --capacity-tokens 188912 \
  --output data/processed/bfcl-tooltrie-menu64.jsonl
```

`188912 = 16 * 11807` matches the measured block size and GPU-block count of
the current Qwen3-0.6B RTX 3090 setup. Recalculate it from
`vllm:cache_config_info` if the model, GPU allocation, or server configuration
changes. It is only a planner hint; measured vLLM counters remain authoritative.

Planning for record *n* occurs before that record is inserted, so the generated
ordering uses only records `0..n-1`. Frequency fallback is intentionally refused
unless a distinct training workload is supplied with `--training-input`.

Start each compared replay from the same documented cache state. With
`VLLM_SERVER_DEV_MODE=1`, reset once immediately before each complete replay:

```bash
curl -fsS -X POST -H 'Content-Type: application/json' \
  -d '{}' http://127.0.0.1:8000/reset_prefix_cache

python scripts/replay_vllm_workload.py \
  --input data/processed/bfcl-tooltrie-menu64.jsonl \
  --run-label bfcl-tooltrie-v0-menu64 --disable-thinking \
  --output cluster/results/bfcl-tooltrie-v0-menu64.json
```

For the static control, reset again and replay the base workload, or generate an
alphabetical workload from the same partition and limit. Do not compare runs
that began from different warm-cache states.

For the ToolRet systems workload, use the same commands with `--partition
toolret`; score only APC/TTFT outcomes because ToolRet provides retrieval
relevance labels rather than BFCL function-call outputs.

For BFCL quality, substitute `build_bfcl_quality_workload.py --ordering
original` for the base builder, pass its output through
`build_tooltrie_workload.py`, replay with `--max-tokens 128
--disable-thinking`, then run `score_bfcl_quality.py` exactly as in section 6a.

## 9. Capture the cluster environment

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
