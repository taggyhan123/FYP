# NUS GPU instructions — ContextPilot confirmation

This run closes the implementation-attribution gap found after Phase 2. Read the
whole file before executing it. Use one idle physical GPU, one vLLM server, and
one sequential replay driver. Never reuse a result directory.

## What this run can and cannot establish

It compares two explicitly different causal order generators from the pinned
ContextPilot source:

| Condition | API | Required label |
| --- | --- | --- |
| `contextpilot_static_refit_causal` | new `ContextIndex.fit_transform` over every observed request prefix | `ContextPilot static-refit causal adaptation (alpha=0.001; ordering only)` |
| `contextpilot_online_incremental` | one persistent `contextpilot.server.live_index.ContextPilot.reorder` instance | `ContextPilot persistent-API adaptation (alpha=0.001; no eviction feedback or annotations)` |

Both preserve request order and selected-tool membership. The second arm uses
the official persistent ordering API, but its workload is still built before
serving. It therefore does **not** synchronize engine evictions, insert relevance
annotations, de-duplicate content, or measure planning interleaved with serving.
Do not call either arm the “full ContextPilot system.” The builder records these
limitations and its CPU planning time in every summary.

The historical `contextpilot_causal` result (96.16% BFCL / 94.82% ToolRet) used
`alpha=0.5`; it is not an input to this run and must not be overwritten.

## Acceptance conditions

- exact ContextPilot commit `1fa0a143fdeda344585666648ab2b30cb7fea77f`;
- paper/default `alpha=0.001`;
- first 200 requests, fixed sequence, identical selected sets per comparison;
- padded 64-tool positive control and BM25-retrieved menus at k=4/16/64/128;
- three clean vLLM trials per systems condition;
- n=800 BFCL quality pass for both corrected ContextPilot modes;
- all replay counter validations clean, with cache reset before every trial;
- builder summaries, dependency versions, input/output SHA-256 values, raw
  replay files, server commands, and archive SHA-256 preserved.

## 1. Prepare a pinned worktree and external environment

The handoff message must provide `CONFIRM_COMMIT`. Do not infer it from a moving
branch.

```bash
FYP_REPO=/home/taghan/FYP
CONFIRM_COMMIT=REPLACE_WITH_HANDOFF_COMMIT
CONFIRM_STAMP=$(date +%Y%m%d-%H%M%S)
CONFIRM_RESULTS="/home/taghan/contextpilot-confirmation-$CONFIRM_STAMP"
CONFIRM_WORKTREE="/home/taghan/FYP-contextpilot-confirmation-$CONFIRM_STAMP"

git -C "$FYP_REPO" fetch origin
git -C "$FYP_REPO" worktree add --detach "$CONFIRM_WORKTREE" "$CONFIRM_COMMIT"
cd "$CONFIRM_WORKTREE"
test "$(git rev-parse HEAD)" = "$CONFIRM_COMMIT"
test -z "$(git status --porcelain)"
mkdir -p "$CONFIRM_RESULTS"

uv sync
uv run pytest -q
```

Install the exact upstream checkout outside either serving environment:

```bash
REFERENCE_ROOT=/home/taghan/external-references
CONTEXTPILOT_REPO="$REFERENCE_ROOT/ContextPilot-1fa0a143"
CONTEXTPILOT_VENV="/home/taghan/venvs/contextpilot-1fa0a143-$CONFIRM_STAMP"

if [ ! -d "$CONTEXTPILOT_REPO/.git" ]; then
  git clone https://github.com/EfficientContext/ContextPilot.git "$CONTEXTPILOT_REPO"
fi
test -z "$(git -C "$CONTEXTPILOT_REPO" status --porcelain)"
git -C "$CONTEXTPILOT_REPO" fetch origin
git -C "$CONTEXTPILOT_REPO" switch --detach 1fa0a143fdeda344585666648ab2b30cb7fea77f
test "$(git -C "$CONTEXTPILOT_REPO" rev-parse HEAD)" = 1fa0a143fdeda344585666648ab2b30cb7fea77f

python3.12 -m venv "$CONTEXTPILOT_VENV"
"$CONTEXTPILOT_VENV/bin/python" -m pip install --upgrade pip
"$CONTEXTPILOT_VENV/bin/python" -m pip install -e "$CONTEXTPILOT_REPO"
"$CONTEXTPILOT_VENV/bin/python" scripts/build_contextpilot_workload.py --help >/dev/null
"$CONTEXTPILOT_VENV/bin/python" -m pip freeze > "$CONFIRM_RESULTS/contextpilot-pip-freeze.txt"
```

If any directory already contains unrelated changes, stop and use a new path.

## 2. Build the corrected CPU workloads

Build one true BM25-selected menu per k. Gold labels are scored only after
selection and never determine membership.

```bash
for K in 4 16 64 128; do
  uv run python scripts/build_retrieved_tool_workload.py \
    --menu-size "$K" --offset 0 --limit 200 --ordering original \
    --output "$CONFIRM_RESULTS/toolret-bm25-k$K-original.jsonl"
  uv run python scripts/build_retrieved_tool_workload.py \
    --menu-size "$K" --offset 0 --limit 200 --ordering alphabetical \
    --output "$CONFIRM_RESULTS/toolret-bm25-k$K-alphabetical.jsonl"

  for MODE in static_refit_causal online_incremental; do
    "$CONTEXTPILOT_VENV/bin/python" scripts/build_contextpilot_workload.py \
      --input "$CONFIRM_RESULTS/toolret-bm25-k$K-original.jsonl" \
      --contextpilot-repo "$CONTEXTPILOT_REPO" \
      --mode "$MODE" --alpha 0.001 \
      --output "$CONFIRM_RESULTS/toolret-bm25-k$K-contextpilot-$MODE.jsonl" \
      --summary-output "$CONFIRM_RESULTS/toolret-bm25-k$K-contextpilot-$MODE-summary.json"
  done
done
```

Build the padded positive controls and n=800 quality inputs:

```bash
for DATASET in bfcl toolret; do
  uv run python scripts/build_cluster_workload.py \
    --partition "$DATASET" --ordering original --offset 0 --limit 200 \
    --menu-size 64 --random-seed 42 \
    --output "$CONFIRM_RESULTS/$DATASET-padded64-original.jsonl"
  uv run python scripts/build_cluster_workload.py \
    --partition "$DATASET" --ordering alphabetical --offset 0 --limit 200 \
    --menu-size 64 --random-seed 42 \
    --output "$CONFIRM_RESULTS/$DATASET-padded64-alphabetical.jsonl"
  for MODE in static_refit_causal online_incremental; do
    "$CONTEXTPILOT_VENV/bin/python" scripts/build_contextpilot_workload.py \
      --input "$CONFIRM_RESULTS/$DATASET-padded64-original.jsonl" \
      --contextpilot-repo "$CONTEXTPILOT_REPO" \
      --mode "$MODE" --alpha 0.001 \
      --output "$CONFIRM_RESULTS/$DATASET-padded64-contextpilot-$MODE.jsonl"
  done
done

uv run python scripts/build_bfcl_quality_workload.py \
  --ordering original --per-domain 160 --menu-size 64 --random-seed 42 \
  --output "$CONFIRM_RESULTS/quality-original.jsonl"
uv run python scripts/build_bfcl_quality_workload.py \
  --ordering alphabetical --per-domain 160 --menu-size 64 --random-seed 42 \
  --output "$CONFIRM_RESULTS/quality-alphabetical.jsonl"
for MODE in static_refit_causal online_incremental; do
  "$CONTEXTPILOT_VENV/bin/python" scripts/build_contextpilot_workload.py \
    --input "$CONFIRM_RESULTS/quality-original.jsonl" \
    --contextpilot-repo "$CONTEXTPILOT_REPO" \
    --mode "$MODE" --alpha 0.001 \
    --output "$CONFIRM_RESULTS/quality-contextpilot-$MODE.jsonl"
done
```

Do not continue if a summary reports `alpha != 0.001`,
`full_contextpilot_system: true`, changed request order, or missing hashes.

## 3. Run the systems matrix on Qwen3-0.6B

Start the same stock vLLM configuration used by Phase 2 on one idle GPU. Record
the exact command and read back live cache capacity. Do not install ContextPilot
into the vLLM environment for these ordering-only arms. Export the §1 paths in
each new server/client pane; shell variables are not shared between panes.

```bash
GPU_ID=REPLACE_WITH_IDLE_GPU
VLLM_BIN=/home/taghan/tatm/.venv/bin/vllm
PATH=/home/taghan/tatm/.venv/bin:$PATH \
VLLM_SERVER_DEV_MODE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
CPATH=/home/taghan/miniconda3/envs/hdr312/include/python3.12 \
"$VLLM_BIN" serve Qwen/Qwen3-0.6B \
  --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser hermes \
  --host 127.0.0.1 --port 8000 \
  2>&1 | tee "$CONFIRM_RESULTS/vllm-qwen3-0.6b.log"
```

In the single client pane:

```bash
cd "$CONFIRM_WORKTREE"
source .venv/bin/activate
curl -fsS http://127.0.0.1:8000/v1/models >/dev/null
python scripts/inspect_vllm_server.py --base-url http://127.0.0.1:8000 \
  > "$CONFIRM_RESULTS/vllm-cache-config.json"
VLLM_CAPACITY_06B=$(python scripts/inspect_vllm_server.py \
  --base-url http://127.0.0.1:8000 --capacity-only)

for K in 4 16 64 128; do
  python scripts/build_tooltrie_workload.py \
    --input "$CONFIRM_RESULTS/toolret-bm25-k$K-original.jsonl" \
    --policy tooltrie_v0 --fallback alphabetical --recency-window 128 \
    --capacity-tokens "$VLLM_CAPACITY_06B" \
    --output "$CONFIRM_RESULTS/toolret-bm25-k$K-tooltrie_v0.jsonl"
done
for DATASET in bfcl toolret; do
  python scripts/build_tooltrie_workload.py \
    --input "$CONFIRM_RESULTS/$DATASET-padded64-original.jsonl" \
    --policy tooltrie_v0 --fallback alphabetical --recency-window 128 \
    --capacity-tokens "$VLLM_CAPACITY_06B" \
    --output "$CONFIRM_RESULTS/$DATASET-padded64-tooltrie_v0.jsonl"
done

for K in 4 16 64 128; do
  for CONDITION in original alphabetical tooltrie_v0 \
    contextpilot-static_refit_causal contextpilot-online_incremental; do
    for TRIAL in 1 2 3; do
      python scripts/replay_vllm_workload.py \
        --input "$CONFIRM_RESULTS/toolret-bm25-k$K-$CONDITION.jsonl" \
        --run-label "toolret-bm25-k$K-$CONDITION-trial-$TRIAL" \
        --max-tokens 1 --disable-thinking --reset-before \
        --output "$CONFIRM_RESULTS/toolret-bm25-k$K-$CONDITION-trial-$TRIAL.json"
    done
  done
done

for DATASET in bfcl toolret; do
  for CONDITION in original alphabetical tooltrie_v0 \
    contextpilot-static_refit_causal contextpilot-online_incremental; do
    for TRIAL in 1 2 3; do
      python scripts/replay_vllm_workload.py \
        --input "$CONFIRM_RESULTS/$DATASET-padded64-$CONDITION.jsonl" \
        --run-label "$DATASET-padded64-$CONDITION-trial-$TRIAL" \
        --max-tokens 48 --disable-thinking --reset-before \
        --output "$CONFIRM_RESULTS/$DATASET-padded64-$CONDITION-trial-$TRIAL.json"
    done
  done
done

for K in 4 16 64 128; do
  SUMMARY_ARGS=()
  for CONDITION in original alphabetical tooltrie_v0 \
    contextpilot-static_refit_causal contextpilot-online_incremental; do
    for TRIAL in 1 2 3; do
      SUMMARY_ARGS+=(--run "$CONDITION=$CONFIRM_RESULTS/toolret-bm25-k$K-$CONDITION-trial-$TRIAL.json")
    done
  done
  python scripts/summarize_ordering_replays.py "${SUMMARY_ARGS[@]}" \
    --output "$CONFIRM_RESULTS/toolret-bm25-k$K-summary.json"
done

for DATASET in bfcl toolret; do
  SUMMARY_ARGS=()
  for CONDITION in original alphabetical tooltrie_v0 \
    contextpilot-static_refit_causal contextpilot-online_incremental; do
    for TRIAL in 1 2 3; do
      SUMMARY_ARGS+=(--run "$CONDITION=$CONFIRM_RESULTS/$DATASET-padded64-$CONDITION-trial-$TRIAL.json")
    done
  done
  python scripts/summarize_ordering_replays.py "${SUMMARY_ARGS[@]}" \
    --output "$CONFIRM_RESULTS/$DATASET-padded64-summary.json"
done
```

Every replay must report `counter_validation.clean: true`. Preserve and
quarantine failures; never pass `--allow-counter-mismatch` into an accepted run.
Every summary must report the same case set, request sequence, and selected tool
sets across conditions.

## 4. Run the n=800 quality confirmation on Qwen3-8B

Stop the 0.6B server completely. In the server pane, export the same
`CONFIRM_RESULTS` and selected `GPU_ID`, then start the exact Phase 2 8B server:

```bash
PATH=/home/taghan/tatm/.venv/bin:$PATH \
VLLM_SERVER_DEV_MODE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
CPATH=/home/taghan/miniconda3/envs/hdr312/include/python3.12 \
/home/taghan/tatm/.venv/bin/vllm serve Qwen/Qwen3-8B \
  --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser hermes \
  --host 127.0.0.1 --port 8000 \
  2>&1 | tee "$CONFIRM_RESULTS/vllm-qwen3-8b.log"
```

In the single client pane, export the §1 paths, activate the confirmation
worktree environment, wait for `/v1/models`, and run:

```bash
cd "$CONFIRM_WORKTREE"
source .venv/bin/activate
curl -fsS http://127.0.0.1:8000/v1/models >/dev/null
VLLM_CAPACITY_8B=$(python scripts/inspect_vllm_server.py \
  --base-url http://127.0.0.1:8000 --capacity-only)
python scripts/build_tooltrie_workload.py \
  --input "$CONFIRM_RESULTS/quality-original.jsonl" \
  --policy tooltrie_v0 --fallback alphabetical --recency-window 128 \
  --capacity-tokens "$VLLM_CAPACITY_8B" \
  --output "$CONFIRM_RESULTS/quality-tooltrie_v0.jsonl"

for CONDITION in alphabetical tooltrie_v0 \
  contextpilot-static_refit_causal contextpilot-online_incremental; do
  python scripts/replay_vllm_workload.py \
    --input "$CONFIRM_RESULTS/quality-$CONDITION.jsonl" \
    --run-label "quality-$CONDITION-qwen3-8b" \
    --max-tokens 128 --disable-thinking --reset-before \
    --output "$CONFIRM_RESULTS/quality-$CONDITION-replay.json"
  python scripts/score_bfcl_quality.py \
    --replay-result "$CONFIRM_RESULTS/quality-$CONDITION-replay.json" \
    --output "$CONFIRM_RESULTS/quality-$CONDITION-score.json"
done

for CONDITION in contextpilot-static_refit_causal contextpilot-online_incremental; do
  for METRIC in name_correct full_correct no_tool_correct; do
    python scripts/compare_bfcl_quality.py \
      --baseline "$CONFIRM_RESULTS/quality-alphabetical-score.json" \
      --candidate "$CONFIRM_RESULTS/quality-$CONDITION-score.json" \
      --metric "$METRIC" --bootstrap-samples 50000 --bootstrap-seed 42 \
      --output "$CONFIRM_RESULTS/quality-$CONDITION-vs-alphabetical-$METRIC.json"
    python scripts/compare_bfcl_quality.py \
      --baseline "$CONFIRM_RESULTS/quality-tooltrie_v0-score.json" \
      --candidate "$CONFIRM_RESULTS/quality-$CONDITION-score.json" \
      --metric "$METRIC" --bootstrap-samples 50000 --bootstrap-seed 42 \
      --output "$CONFIRM_RESULTS/quality-$CONDITION-vs-tooltrie_v0-$METRIC.json"
  done
done
```

Function/full metrics apply to 640 relevance cases; no-tool applies to 160
irrelevance cases. These remain ordering-only quality results because no
ContextPilot relevance annotation is inserted.

## 5. Preserve and hand over

```bash
git rev-parse HEAD > "$CONFIRM_RESULTS/fyp-git-commit.txt"
git -C "$CONTEXTPILOT_REPO" rev-parse HEAD \
  > "$CONFIRM_RESULTS/contextpilot-git-commit.txt"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv \
  > "$CONFIRM_RESULTS/gpu-environment.csv"

CONFIRM_ARCHIVE="/home/taghan/contextpilot-confirmation-$CONFIRM_STAMP.tar.gz"
tar -czf "$CONFIRM_ARCHIVE" -C /home/taghan \
  "$(basename "$CONFIRM_RESULTS")"
tar -tzf "$CONFIRM_ARCHIVE" >/dev/null
sha256sum "$CONFIRM_ARCHIVE" | tee "$CONFIRM_ARCHIVE.sha256"
```

The handover must give the branch/commit containing compact summaries, archive
path and SHA-256, accepted/quarantined counts, exact server commands, and a
clear statement that no eviction-feedback or annotation arm was executed.
