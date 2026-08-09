# NUS GPU agent instructions — run ToolTrie-v0

Read this file completely before running commands. The objective is to execute
the first causal ToolTrie-v0 experiment on the NUS RTX 3090 server using
**unmodified vLLM automatic prefix caching (APC)**, then report the measured
systems and BFCL quality results.

## Scope and safety

- Repository: `https://github.com/taggyhan123/FYP.git`
- Required branch: `tooltrie-v0-workflow`
- The branch must contain commit `9866367` or a descendant.
- Preserve any existing modified or untracked files. Never run `git reset
  --hard`, `git clean`, force-push, or overwrite an existing result directory.
- Do not modify vLLM, CUDA kernels, attention code, or KV tensors.
- Do not install anything with `sudo`; the working environment is user-owned.
- Do not use a GPU occupied by another user. Inspect `nvidia-smi` first.
- Raw/processed datasets and `cluster/results/` are intentionally ignored by
  Git. Do not add large datasets or raw result dumps to Git.
- Run the supplied baseline before changing the planner. If something fails,
  preserve the failure output and diagnose it instead of silently changing the
  experimental method.

## 1. Synchronize and verify the repository

The expected server checkout used by earlier experiments is
`/home/taghan/tatm`. If the repository is elsewhere, use its real path.

```bash
cd /home/taghan/tatm
git status --short --branch
git fetch origin
git switch tooltrie-v0-workflow 2>/dev/null || \
  git switch --track origin/tooltrie-v0-workflow
git pull --ff-only
git log -1 --oneline
```

If `git status` shows existing work that conflicts with this branch, do not
discard it. Report the paths and use a separate clean clone or worktree.

Activate the existing project environment. Do not run a normal exact `uv sync`
against a venv containing vLLM because it may remove packages that are not in
`pyproject.toml`.

```bash
cd /home/taghan/tatm
source .venv/bin/activate
python --version
python -m pytest -q
```

Expected local verification: all tests pass (55 tests at commit `9866367`).

## 2. Ensure datasets are available

The workflow needs `data/processed/tools.jsonl` and `tasks.jsonl`; BFCL scoring
also needs `data/raw/bfcl/possible_answer/`.

```bash
test -s data/processed/tools.jsonl && test -s data/processed/tasks.jsonl
```

If either processed file is absent, reproduce the data:

```bash
python scripts/download_datasets.py
python scripts/run_pipeline.py
```

Do not commit the resulting `data/` files.

## 3. Select an idle GPU and start stock vLLM

Inspect all GPUs and choose one with enough free memory and no conflicting
process. Never terminate another user's process.

```bash
nvidia-smi
```

In a dedicated terminal or tmux pane, replace `GPU_ID` below with the selected
physical GPU number:

```bash
cd /home/taghan/tatm
source .venv/bin/activate

PATH=/home/taghan/tatm/.venv/bin:$PATH \
VLLM_SERVER_DEV_MODE=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
CUDA_VISIBLE_DEVICES=GPU_ID \
CPATH=/home/taghan/miniconda3/envs/hdr312/include/python3.12 \
vllm serve Qwen/Qwen3-0.6B \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --host 127.0.0.1 \
  --port 8000
```

Wait until both endpoints respond:

```bash
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8000/metrics | \
  grep 'vllm:cache_config_info' | head -n 1
```

Confirm `enable_prefix_caching="True"` in `vllm:cache_config_info`. Record its
`block_size` and `num_gpu_blocks`. Set ToolTrie's `--capacity-tokens` to their
product. The earlier setup reported `16 * 11807 = 188912`; use `188912` only if
the current server reports those same values.

## 4. Create a unique result directory

Create one new directory for this run and use it for every result. Never reuse
or overwrite an earlier directory.

```bash
cd /home/taghan/tatm
RUN_STAMP=$(date +%Y%m%d-%H%M%S)
RESULT_DIR="cluster/results/tooltrie-v0-${RUN_STAMP}"
mkdir -p "$RESULT_DIR"
echo "$RESULT_DIR"
```

Keep this shell open so `RESULT_DIR` remains defined. Replace `188912` in the
commands below if the current server reports a different cache capacity.

## 5. Build matched BFCL systems workloads on CPU

Generate original, alphabetical, and ToolTrie conditions from the same first
200 BFCL requests and the same 64-tool menus:

```bash
python scripts/build_cluster_workload.py \
  --partition bfcl --ordering original --menu-size 64 --limit 200 \
  --output "$RESULT_DIR/bfcl-original-menu64.jsonl"

python scripts/build_cluster_workload.py \
  --partition bfcl --ordering alphabetical --menu-size 64 --limit 200 \
  --output "$RESULT_DIR/bfcl-alphabetical-menu64.jsonl"

python scripts/build_tooltrie_workload.py \
  --input "$RESULT_DIR/bfcl-original-menu64.jsonl" \
  --fallback alphabetical --recency-window 128 \
  --capacity-tokens 188912 \
  --output "$RESULT_DIR/bfcl-tooltrie-menu64.jsonl"
```

The ToolTrie builder must report 200 requests. Request 0 should have no matched
prefix; later requests may use only paths observed in earlier records.

## 6. Replay BFCL systems conditions with identical cache policy

Reset APC exactly once immediately before each complete replay. A reset requires
`VLLM_SERVER_DEV_MODE=1`; stop if the reset request fails.

```bash
curl -fsS -X POST -H 'Content-Type: application/json' -d '{}' \
  http://127.0.0.1:8000/reset_prefix_cache
python scripts/replay_vllm_workload.py \
  --input "$RESULT_DIR/bfcl-original-menu64.jsonl" \
  --run-label bfcl-original-menu64 --disable-thinking \
  --output "$RESULT_DIR/bfcl-original-replay.json"

curl -fsS -X POST -H 'Content-Type: application/json' -d '{}' \
  http://127.0.0.1:8000/reset_prefix_cache
python scripts/replay_vllm_workload.py \
  --input "$RESULT_DIR/bfcl-alphabetical-menu64.jsonl" \
  --run-label bfcl-alphabetical-menu64 --disable-thinking \
  --output "$RESULT_DIR/bfcl-alphabetical-replay.json"

curl -fsS -X POST -H 'Content-Type: application/json' -d '{}' \
  http://127.0.0.1:8000/reset_prefix_cache
python scripts/replay_vllm_workload.py \
  --input "$RESULT_DIR/bfcl-tooltrie-menu64.jsonl" \
  --run-label bfcl-tooltrie-v0-menu64 --disable-thinking \
  --output "$RESULT_DIR/bfcl-tooltrie-replay.json"
```

Do not issue unrelated model requests between a reset and the end of its replay.

## 7. Run ToolRet as the systems-only workload

Repeat the matched original/alphabetical/ToolTrie construction with ToolRet:

```bash
python scripts/build_cluster_workload.py \
  --partition toolret --ordering original --menu-size 64 --limit 200 \
  --output "$RESULT_DIR/toolret-original-menu64.jsonl"

python scripts/build_cluster_workload.py \
  --partition toolret --ordering alphabetical --menu-size 64 --limit 200 \
  --output "$RESULT_DIR/toolret-alphabetical-menu64.jsonl"

python scripts/build_tooltrie_workload.py \
  --input "$RESULT_DIR/toolret-original-menu64.jsonl" \
  --fallback alphabetical --recency-window 128 \
  --capacity-tokens 188912 \
  --output "$RESULT_DIR/toolret-tooltrie-menu64.jsonl"
```

Replay all three files exactly as in section 6, resetting once before each run
and writing three distinct JSON result files. ToolRet is used for APC, TTFT, and
systems measurements; do not interpret its retrieval labels as BFCL call
correctness.

## 8. Run the stratified BFCL quality comparison

The systems subset may be dominated by irrelevance tasks, so build the separate
stratified quality workload:

```bash
python scripts/build_bfcl_quality_workload.py \
  --ordering original --per-domain 20 --menu-size 64 \
  --output "$RESULT_DIR/bfcl-quality-original.jsonl"

python scripts/build_bfcl_quality_workload.py \
  --ordering alphabetical --per-domain 20 --menu-size 64 \
  --output "$RESULT_DIR/bfcl-quality-alphabetical.jsonl"

python scripts/build_tooltrie_workload.py \
  --input "$RESULT_DIR/bfcl-quality-original.jsonl" \
  --fallback alphabetical --recency-window 128 \
  --capacity-tokens 188912 \
  --output "$RESULT_DIR/bfcl-quality-tooltrie.jsonl"
```

For each of the three files, reset once, then replay with `--max-tokens 128
--disable-thinking` into a distinct result file. Example for ToolTrie:

```bash
curl -fsS -X POST -H 'Content-Type: application/json' -d '{}' \
  http://127.0.0.1:8000/reset_prefix_cache
python scripts/replay_vllm_workload.py \
  --input "$RESULT_DIR/bfcl-quality-tooltrie.jsonl" \
  --run-label bfcl-quality-tooltrie-v0 \
  --max-tokens 128 --disable-thinking \
  --output "$RESULT_DIR/bfcl-quality-tooltrie-replay.json"

python scripts/score_bfcl_quality.py \
  --replay-result "$RESULT_DIR/bfcl-quality-tooltrie-replay.json" \
  --output "$RESULT_DIR/bfcl-quality-tooltrie-score.json"
```

Score the original and alphabetical replay files in the same way.

## 9. Capture environment and evidence

```bash
python -c 'import platform, torch, vllm; print(platform.platform()); print(torch.__version__); print(vllm.__version__)' \
  > "$RESULT_DIR/environment.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv >> "$RESULT_DIR/environment.txt"
curl -fsS http://127.0.0.1:8000/metrics \
  > "$RESULT_DIR/metrics-after-runs.txt"
git rev-parse HEAD > "$RESULT_DIR/git-commit.txt"
```

Also save the exact vLLM launch command and selected physical GPU ID in
`$RESULT_DIR/server-command.txt`.

## 10. Required result report

Do not say the task is complete merely because the scripts exited. Report:

1. Repository commit, model, vLLM version, GPU, block size, GPU blocks, and exact
   server command.
2. For original, alphabetical, and ToolTrie on both BFCL and ToolRet:
   request count, total prompt tokens, measured cached prompt tokens, cached
   ratio, computed prefill tokens, aggregate prefill time, aggregate TTFT, and
   wall time.
3. ToolTrie planner metadata: requests with a hinted prefix, hinted schema
   tokens, final node count, retained schema tokens, and planner evictions.
4. BFCL quality for all three orderings: function-name accuracy, full
   name-plus-arguments accuracy, and no-tool accuracy, overall and by domain.
5. Any failed or retried requests and whether any unrelated traffic reached the
   server during a replay.
6. A cautious comparison: whether ToolTrie improves measured APC reuse and TTFT
   over both original and alphabetical without more than a predeclared
   one-percentage-point BFCL quality regression.

The analytical `hinted_schema_tokens` value is not a vLLM cache hit. Only the
server's measured counters support APC or latency claims. Do not commit or push
results unless the user explicitly asks; return the result-directory path and a
concise findings summary first.
