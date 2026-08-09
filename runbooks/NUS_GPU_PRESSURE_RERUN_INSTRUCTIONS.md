# NUS GPU instructions — controlled-cache pressure rerun

Read this file and
`cluster/initial-brief-pressure-rerun-manifest.json` completely before running
commands. This procedure replaces only the failed memory-pressure acceptance
criterion from the initial-brief closure run. It does not rerun or overwrite
the 84 retrieved-menu replays, seven rendered-prefix audits, or their archive.

## Experimental contract

- Run the exact commit supplied by the local analysis session in a separate
  detached worktree. Do not advance or clean another session's worktree.
- Use one idle GPU, one vLLM server, and one sequential replay client. Do not
  use concurrency, terminate another user's process, or change the workload.
- Keep the original 0.90 occupancy threshold. The only controlled intervention
  is the predeclared 480-block KV cache (7,680 tokens).
- This is a controlled stress test. Never compare its latency with the earlier
  190,896-token-capacity runs or describe it as natural production pressure.
- Preserve every failed output. Do not lower the threshold, alter capacity, or
  substitute a condition after seeing results.
- Raw outputs stay outside Git in a unique, checksummed archive. Commit only a
  compact summary, environment record, and handover.

## 1. Create an isolated worktree at the pinned commit

The handoff sender must provide the full commit in place of
`REPLACE_WITH_PINNED_COMMIT`.

```bash
export FYP_SOURCE=/home/taghan/FYP
export EXPECTED_FYP_COMMIT=REPLACE_WITH_PINNED_COMMIT
export PRESSURE_STAMP=$(date +%Y%m%d-%H%M%S)
export PRESSURE_WORKTREE=/home/taghan/FYP-pressure-rerun-$PRESSURE_STAMP

cd "$FYP_SOURCE"
git fetch origin
test ! -e "$PRESSURE_WORKTREE"
git worktree add --detach "$PRESSURE_WORKTREE" "$EXPECTED_FYP_COMMIT"
cd "$PRESSURE_WORKTREE"
test "$(git rev-parse HEAD)" = "$EXPECTED_FYP_COMMIT"
git status --short --branch
git log -1 --oneline
```

Use the existing project environment without modifying it:

```bash
export FYP_PYTHON=/home/taghan/FYP/.venv/bin/python
test -x "$FYP_PYTHON"
"$FYP_PYTHON" --version
"$FYP_PYTHON" -m pytest -q
"$FYP_PYTHON" -m json.tool cluster/initial-brief-pressure-rerun-manifest.json >/dev/null
"$FYP_PYTHON" scripts/locality_replay.py --help >/dev/null
"$FYP_PYTHON" scripts/summarize_pressure_replays.py --help >/dev/null
test -s "$FYP_SOURCE/data/processed/tools.jsonl"
test -s "$FYP_SOURCE/data/processed/tasks.jsonl"
```

## 2. Select an idle GPU and create a unique result directory

```bash
nvidia-smi
```

Set `GPU_ID` manually to an idle physical GPU. Do not leave the placeholder.

```bash
export GPU_ID=REPLACE_WITH_IDLE_GPU_NUMBER
export VLLM_BIN=/home/taghan/tatm/.venv/bin/vllm
test -x "$VLLM_BIN"
export PRESSURE_RESULTS="$PRESSURE_WORKTREE/cluster/results/initial-brief-pressure-rerun-$PRESSURE_STAMP"
test ! -e "$PRESSURE_RESULTS"
mkdir -p "$PRESSURE_RESULTS"/{runs,summaries}
```

Record provenance before starting the server:

```bash
cd "$PRESSURE_WORKTREE"
{
  date --iso-8601=seconds
  git rev-parse HEAD
  git status --short --branch
  uname -a
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv
  "$FYP_PYTHON" -c 'import torch, vllm; print("torch", torch.__version__); print("vllm", vllm.__version__)'
} | tee "$PRESSURE_RESULTS/environment.txt"

"$VLLM_BIN" serve --help | rg -- '--num-gpu-blocks-override|--kv-cache-metrics|--kv-cache-metrics-sample|--max-model-len'
```

Stop if the selected GPU is occupied, any declared flag is unavailable, or the
reported vLLM version is not 0.26.0.

## 3. Start the controlled-cache server

Start this command in a dedicated tmux pane. The cache capacity and metric
sampling flags are part of the method and must not be changed.

```bash
cd "$PRESSURE_WORKTREE"
PATH=/home/taghan/tatm/.venv/bin:$PATH \
VLLM_SERVER_DEV_MODE=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
CPATH=/home/taghan/miniconda3/envs/hdr312/include/python3.12 \
"$VLLM_BIN" serve Qwen/Qwen3-0.6B \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 7168 \
  --num-gpu-blocks-override 480 \
  --kv-cache-metrics \
  --kv-cache-metrics-sample 1.0 \
  --host 127.0.0.1 \
  --port 8000 2>&1 | tee "$PRESSURE_RESULTS/vllm-controlled-cache.log"
```

In the client pane, read back the live configuration. Do not rely on the
command line alone.

```bash
cd "$PRESSURE_WORKTREE"
curl -fsS http://127.0.0.1:8000/v1/models | tee "$PRESSURE_RESULTS/models.json"
"$FYP_PYTHON" scripts/inspect_vllm_server.py | tee "$PRESSURE_RESULTS/cache-config.json"
export PRESSURE_CAPACITY=$("$FYP_PYTHON" scripts/inspect_vllm_server.py --capacity-only)
test "$PRESSURE_CAPACITY" = 7680
curl -fsS http://127.0.0.1:8000/metrics > "$PRESSURE_RESULTS/metrics-before.txt"
rg 'vllm:kv_block_idle_before_evict_seconds_count|vllm:cache_config_info' "$PRESSURE_RESULTS/metrics-before.txt"
```

Stop unless prefix caching is true, block size is 16, GPU blocks are 480, live
capacity is exactly 7,680 tokens, and the eviction histogram is present.

## 4. Run all six orderings and four regimes

Each ordering output contains all four regimes and resets APC once before each
regime. The client remains sequential, preserving the original request-order
experiment.

```bash
cd "$PRESSURE_WORKTREE"
pressure_failures=0
for ordering in original alphabetical random frequency schema_cost_weighted fp_tree_global; do
  output="$PRESSURE_RESULTS/runs/$ordering.json"
  test ! -e "$output"
  "$FYP_PYTHON" scripts/locality_replay.py \
    --processed-dir "$FYP_SOURCE/data/processed" \
    --run-label "controlled-pressure-bfcl-k64-$ordering" \
    --partition bfcl --offset 0 --limit 200 --menu-size 64 \
    --ordering "$ordering" --random-seed 42 --replay-seed 2026 \
    --support-mode disjoint --kv-sample-interval 0.01 \
    --condition empirical \
    --condition uniform \
    --condition skewed \
    --condition session_bursty \
    --require-peak-kv-usage 0.90 \
    --output "$output" || pressure_failures=$((pressure_failures + 1))
done
printf 'ordering commands with failed pressure checks: %s\n' "$pressure_failures"
```

Do not rerun an individual ordering into the same path. A failure is evidence
and must remain in the archive.

## 5. Validate the complete matrix

The summarizer verifies the declared cache, workload, support split, resets,
counters, occupancy, scrape errors, sequential execution, absence of
preemption, and positive sampled eviction count in every regime-run.

```bash
cd "$PRESSURE_WORKTREE"
run_args=()
for ordering in original alphabetical random frequency schema_cost_weighted fp_tree_global; do
  run_args+=(--run "$ordering=$PRESSURE_RESULTS/runs/$ordering.json")
done
"$FYP_PYTHON" scripts/summarize_pressure_replays.py \
  "${run_args[@]}" \
  --expected-capacity-tokens 7680 \
  --required-peak-fraction 0.90 \
  --output "$PRESSURE_RESULTS/summaries/pressure-summary.json"
```

Acceptance requires `24/24` regime-runs and every validation check to pass.
If the summarizer exits non-zero, preserve everything and report each failed
check. Do not change the manifest or launch a second protocol.

## 6. Archive and hand back

Create a unique tracked handover directory and archive the complete raw result.

```bash
export TRACKED_HANDOVER="$PRESSURE_WORKTREE/reports/initial-brief-pressure-rerun/$PRESSURE_STAMP"
test ! -e "$TRACKED_HANDOVER"
mkdir -p "$TRACKED_HANDOVER"
cp "$PRESSURE_RESULTS/environment.txt" "$TRACKED_HANDOVER/"
cp "$PRESSURE_RESULTS/cache-config.json" "$TRACKED_HANDOVER/"
cp "$PRESSURE_RESULTS/models.json" "$TRACKED_HANDOVER/"
cp "$PRESSURE_RESULTS/summaries/pressure-summary.json" "$TRACKED_HANDOVER/"

export PRESSURE_ARCHIVE="/home/taghan/initial-brief-pressure-rerun-$PRESSURE_STAMP.tar.gz"
test ! -e "$PRESSURE_ARCHIVE"
tar -C "$PRESSURE_WORKTREE/cluster/results" -czf "$PRESSURE_ARCHIVE" "initial-brief-pressure-rerun-$PRESSURE_STAMP"
tar -tzf "$PRESSURE_ARCHIVE" >/dev/null
sha256sum "$PRESSURE_ARCHIVE" | tee "$TRACKED_HANDOVER/raw-archive.sha256"
```

Write `HANDOVER.md` in the tracked directory with the exact commit, GPU and
server provenance, 24-run acceptance table, any failed check, archive path,
archive size and entry count, and SHA-256. State explicitly that the run is a
controlled-cache stress test and that cross-capacity latency comparisons are
invalid.

Commit the handover on a separate branch:

```bash
cd "$PRESSURE_WORKTREE"
git switch -c "gpu/initial-brief-pressure-rerun-$PRESSURE_STAMP"
git add "$TRACKED_HANDOVER"
git commit -m "Record controlled-cache pressure rerun"
git push -u origin HEAD
git log -1 --oneline
```

Return the branch, commit, archive path, archive size and SHA-256, accepted
regime count, and every failed condition to the local analysis session. Do not
edit `PROJECT_STATUS.md` or scientific reports in the GPU executor session.
