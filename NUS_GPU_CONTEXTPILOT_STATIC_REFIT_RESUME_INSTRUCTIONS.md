# NUS GPU instructions — ContextPilot static-refit-only resume

This protocol completes only the cells quarantined in the accepted
`20260807-222212` ContextPilot run. Read the whole file before running it.

Do **not** rerun:

- the 72 accepted Qwen3-0.6B systems replays;
- the three accepted Qwen3-8B persistent/ToolTrie/alphabetical quality replays;
- the separate three-run Qwen3-4B addendum.

Those runs have clean counters and successful cache resets. This resume adds 18
Qwen3-0.6B static-refit systems trials and one Qwen3-8B static-refit quality
replay, then performs CPU-only re-analysis of the existing quality outputs.

## 1. Inputs and immutable pins

The analysis-session handoff must replace `REPLACE_WITH_HANDOFF_COMMIT` with an
exact commit. Do not infer it from a moving branch.

```bash
FYP_REPO=/home/taghan/FYP
RESUME_COMMIT=REPLACE_WITH_HANDOFF_COMMIT
ACCEPTED_RESULTS=/home/taghan/contextpilot-confirmation-20260807-222212
QUALITY_4B_RESULTS=/home/taghan/contextpilot-quality-4b-20260808-105833
RESUME_STAMP=$(date +%Y%m%d-%H%M%S)
RESUME_WORKTREE="/home/taghan/FYP-contextpilot-static-refit-$RESUME_STAMP"
RESUME_RESULTS="/home/taghan/contextpilot-static-refit-$RESUME_STAMP"
REFERENCE_ROOT=/home/taghan/external-references
CONTEXTPILOT_REPO="$REFERENCE_ROOT/ContextPilot-1fa0a143"
CONTEXTPILOT_VENV="/home/taghan/venvs/contextpilot-1fa0a143-$RESUME_STAMP"

git -C "$FYP_REPO" fetch origin
git -C "$FYP_REPO" worktree add --detach "$RESUME_WORKTREE" "$RESUME_COMMIT"
cd "$RESUME_WORKTREE"
test "$(git rev-parse HEAD)" = "$RESUME_COMMIT"
test -z "$(git status --porcelain)"
mkdir -p "$RESUME_RESULTS"
uv sync
uv run pytest -q
test -d "$ACCEPTED_RESULTS"
test -f "$ACCEPTED_RESULTS/quality-alphabetical-score.json"
test -f "$ACCEPTED_RESULTS/quality-tooltrie_v0-score.json"
test -f "$ACCEPTED_RESULTS/quality-contextpilot-online_incremental-score.json"
```

If the accepted result directory is absent, restore the verified archive
`/home/taghan/contextpilot-confirmation-20260807-222212.tar.gz` to a new path
before continuing. Never overwrite the archive or an existing result directory.

Prepare the exact upstream checkout. The server previously lacked
`ensurepip`, so use `uv venv`; this is the already-recorded environment
deviation, not a new method change.

```bash
if [ ! -d "$CONTEXTPILOT_REPO/.git" ]; then
  git clone https://github.com/EfficientContext/ContextPilot.git "$CONTEXTPILOT_REPO"
fi
test -z "$(git -C "$CONTEXTPILOT_REPO" status --porcelain)"
git -C "$CONTEXTPILOT_REPO" fetch origin
git -C "$CONTEXTPILOT_REPO" switch --detach 1fa0a143fdeda344585666648ab2b30cb7fea77f
test "$(git -C "$CONTEXTPILOT_REPO" rev-parse HEAD)" = 1fa0a143fdeda344585666648ab2b30cb7fea77f
uv venv --python 3.12 "$CONTEXTPILOT_VENV"
uv pip install --python "$CONTEXTPILOT_VENV/bin/python" -e "$CONTEXTPILOT_REPO"
"$CONTEXTPILOT_VENV/bin/python" scripts/build_contextpilot_workload.py --help >/dev/null
uv pip freeze --python "$CONTEXTPILOT_VENV/bin/python" \
  > "$RESUME_RESULTS/contextpilot-pip-freeze.txt"
```

## 2. Build only the corrected static-refit workloads

Use the accepted run's exact input workloads. This removes dataset regeneration
as a source of variation.

```bash
for K in 4 16 64 128; do
  "$CONTEXTPILOT_VENV/bin/python" scripts/build_contextpilot_workload.py \
    --input "$ACCEPTED_RESULTS/toolret-bm25-k$K-original.jsonl" \
    --contextpilot-repo "$CONTEXTPILOT_REPO" \
    --mode static_refit_causal --alpha 0.001 \
    --output "$RESUME_RESULTS/toolret-bm25-k$K-contextpilot-static_refit_causal.jsonl" \
    --summary-output "$RESUME_RESULTS/toolret-bm25-k$K-contextpilot-static_refit_causal-summary.json"
done

for DATASET in bfcl toolret; do
  "$CONTEXTPILOT_VENV/bin/python" scripts/build_contextpilot_workload.py \
    --input "$ACCEPTED_RESULTS/$DATASET-padded64-original.jsonl" \
    --contextpilot-repo "$CONTEXTPILOT_REPO" \
    --mode static_refit_causal --alpha 0.001 \
    --output "$RESUME_RESULTS/$DATASET-padded64-contextpilot-static_refit_causal.jsonl" \
    --summary-output "$RESUME_RESULTS/$DATASET-padded64-contextpilot-static_refit_causal-summary.json"
done

"$CONTEXTPILOT_VENV/bin/python" scripts/build_contextpilot_workload.py \
  --input "$ACCEPTED_RESULTS/quality-original.jsonl" \
  --contextpilot-repo "$CONTEXTPILOT_REPO" \
  --mode static_refit_causal --alpha 0.001 \
  --output "$RESUME_RESULTS/quality-contextpilot-static_refit_causal.jsonl" \
  --summary-output "$RESUME_RESULTS/quality-contextpilot-static_refit_causal-summary.json"
```

Stop if any builder summary reports the wrong commit or alpha, future-request
visibility, changed request order, changed selected membership,
`full_contextpilot_system: true`, or missing hashes.

## 3. Run the 18 missing Qwen3-0.6B trials

Use one idle physical GPU, one stock vLLM server, and one sequential client.
Record `GPU_ID` and the exact server command. Start this in the server pane:

```bash
GPU_ID=REPLACE_WITH_IDLE_GPU
PATH=/home/taghan/tatm/.venv/bin:$PATH \
VLLM_SERVER_DEV_MODE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
CPATH=/home/taghan/miniconda3/envs/hdr312/include/python3.12 \
/home/taghan/tatm/.venv/bin/vllm serve Qwen/Qwen3-0.6B \
  --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser hermes \
  --host 127.0.0.1 --port 8000 \
  2>&1 | tee "$RESUME_RESULTS/vllm-qwen3-0.6b.log"
```

In the client pane, export the variables from §1, then run only the static arm:

```bash
cd "$RESUME_WORKTREE"
source .venv/bin/activate
curl -fsS http://127.0.0.1:8000/v1/models >/dev/null
python scripts/inspect_vllm_server.py --base-url http://127.0.0.1:8000 \
  > "$RESUME_RESULTS/vllm-cache-config-0.6b.json"

for K in 4 16 64 128; do
  for TRIAL in 1 2 3; do
    python scripts/replay_vllm_workload.py \
      --input "$RESUME_RESULTS/toolret-bm25-k$K-contextpilot-static_refit_causal.jsonl" \
      --run-label "toolret-bm25-k$K-contextpilot-static_refit_causal-trial-$TRIAL" \
      --max-tokens 1 --disable-thinking --reset-before \
      --output "$RESUME_RESULTS/toolret-bm25-k$K-contextpilot-static_refit_causal-trial-$TRIAL.json"
  done
done

for DATASET in bfcl toolret; do
  for TRIAL in 1 2 3; do
    python scripts/replay_vllm_workload.py \
      --input "$RESUME_RESULTS/$DATASET-padded64-contextpilot-static_refit_causal.jsonl" \
      --run-label "$DATASET-padded64-contextpilot-static_refit_causal-trial-$TRIAL" \
      --max-tokens 48 --disable-thinking --reset-before \
      --output "$RESUME_RESULTS/$DATASET-padded64-contextpilot-static_refit_causal-trial-$TRIAL.json"
  done
done
```

Every replay must have 200 requests, zero failures,
`counter_validation.clean: true`, and a successful pre-run cache reset. Never
use `--allow-counter-mismatch` for an accepted run.

Combine old and new arms only in summaries:

```bash
for K in 4 16 64 128; do
  SUMMARY_ARGS=()
  for CONDITION in original alphabetical tooltrie_v0 contextpilot-online_incremental; do
    for TRIAL in 1 2 3; do
      SUMMARY_ARGS+=(--run "$CONDITION=$ACCEPTED_RESULTS/toolret-bm25-k$K-$CONDITION-trial-$TRIAL.json")
    done
  done
  for TRIAL in 1 2 3; do
    SUMMARY_ARGS+=(--run "contextpilot-static_refit_causal=$RESUME_RESULTS/toolret-bm25-k$K-contextpilot-static_refit_causal-trial-$TRIAL.json")
  done
  python scripts/summarize_ordering_replays.py "${SUMMARY_ARGS[@]}" \
    --output "$RESUME_RESULTS/toolret-bm25-k$K-combined-summary.json"
done

for DATASET in bfcl toolret; do
  SUMMARY_ARGS=()
  for CONDITION in original alphabetical tooltrie_v0 contextpilot-online_incremental; do
    for TRIAL in 1 2 3; do
      SUMMARY_ARGS+=(--run "$CONDITION=$ACCEPTED_RESULTS/$DATASET-padded64-$CONDITION-trial-$TRIAL.json")
    done
  done
  for TRIAL in 1 2 3; do
    SUMMARY_ARGS+=(--run "contextpilot-static_refit_causal=$RESUME_RESULTS/$DATASET-padded64-contextpilot-static_refit_causal-trial-$TRIAL.json")
  done
  python scripts/summarize_ordering_replays.py "${SUMMARY_ARGS[@]}" \
    --output "$RESUME_RESULTS/$DATASET-padded64-combined-summary.json"
done
```

All six combined summaries must report identical case sets, request sequences,
and selected tool sets across all five conditions.

## 4. Run the one missing Qwen3-8B quality replay

Stop the 0.6B server completely and confirm the GPU is released. Start the same
8B server used by the accepted run:

```bash
PATH=/home/taghan/tatm/.venv/bin:$PATH \
VLLM_SERVER_DEV_MODE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
CPATH=/home/taghan/miniconda3/envs/hdr312/include/python3.12 \
/home/taghan/tatm/.venv/bin/vllm serve Qwen/Qwen3-8B \
  --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser hermes \
  --host 127.0.0.1 --port 8000 \
  2>&1 | tee "$RESUME_RESULTS/vllm-qwen3-8b.log"
```

In the client pane:

```bash
cd "$RESUME_WORKTREE"
source .venv/bin/activate
curl -fsS http://127.0.0.1:8000/v1/models >/dev/null
python scripts/inspect_vllm_server.py --base-url http://127.0.0.1:8000 \
  > "$RESUME_RESULTS/vllm-cache-config-8b.json"
python scripts/replay_vllm_workload.py \
  --input "$RESUME_RESULTS/quality-contextpilot-static_refit_causal.jsonl" \
  --run-label quality-contextpilot-static_refit_causal-qwen3-8b \
  --max-tokens 128 --disable-thinking --reset-before \
  --output "$RESUME_RESULTS/quality-contextpilot-static_refit_causal-replay.json"
python scripts/score_bfcl_quality.py \
  --replay-result "$RESUME_RESULTS/quality-contextpilot-static_refit_causal-replay.json" \
  --output "$RESUME_RESULTS/quality-contextpilot-static_refit_causal-score.json"
```

Require 800 requests, zero failures, a clean counter window, the recorded model
`Qwen/Qwen3-8B`, and a successful pre-run reset.

## 5. Regenerate sequence-state comparison metadata on CPU

The current comparison code scopes mixed BFCL metrics automatically. Every
comparison involving ToolTrie or either ContextPilot planner must carry
`--sequence-state-dependent`. This changes metadata and framing, not model
outputs or point estimates.

Generate the six new static-refit comparisons:

```bash
for BASELINE in alphabetical tooltrie_v0; do
  for METRIC in name_correct full_correct no_tool_correct; do
    python scripts/compare_bfcl_quality.py \
      --baseline "$ACCEPTED_RESULTS/quality-$BASELINE-score.json" \
      --candidate "$RESUME_RESULTS/quality-contextpilot-static_refit_causal-score.json" \
      --metric "$METRIC" --bootstrap-samples 50000 --bootstrap-seed 42 \
      --sequence-state-dependent \
      --output "$RESUME_RESULTS/quality-contextpilot-static_refit_causal-vs-$BASELINE-$METRIC.json"
  done
done
```

Regenerate the nine accepted 8B comparisons from their unchanged raw scores:

```bash
for METRIC in name_correct full_correct no_tool_correct; do
  python scripts/compare_bfcl_quality.py \
    --baseline "$ACCEPTED_RESULTS/quality-alphabetical-score.json" \
    --candidate "$ACCEPTED_RESULTS/quality-tooltrie_v0-score.json" \
    --metric "$METRIC" --bootstrap-samples 50000 --bootstrap-seed 42 \
    --sequence-state-dependent \
    --output "$RESUME_RESULTS/quality-8b-tooltrie_v0-vs-alphabetical-$METRIC.json"
  python scripts/compare_bfcl_quality.py \
    --baseline "$ACCEPTED_RESULTS/quality-alphabetical-score.json" \
    --candidate "$ACCEPTED_RESULTS/quality-contextpilot-online_incremental-score.json" \
    --metric "$METRIC" --bootstrap-samples 50000 --bootstrap-seed 42 \
    --sequence-state-dependent \
    --output "$RESUME_RESULTS/quality-8b-contextpilot-online_incremental-vs-alphabetical-$METRIC.json"
  python scripts/compare_bfcl_quality.py \
    --baseline "$ACCEPTED_RESULTS/quality-tooltrie_v0-score.json" \
    --candidate "$ACCEPTED_RESULTS/quality-contextpilot-online_incremental-score.json" \
    --metric "$METRIC" --bootstrap-samples 50000 --bootstrap-seed 42 \
    --sequence-state-dependent \
    --output "$RESUME_RESULTS/quality-8b-contextpilot-online_incremental-vs-tooltrie_v0-$METRIC.json"
done
```

If the 4B raw result directory is present, regenerate its nine comparisons. Do
not rerun the 4B model:

```bash
if [ -d "$QUALITY_4B_RESULTS" ]; then
  for METRIC in name_correct full_correct no_tool_correct; do
    python scripts/compare_bfcl_quality.py \
      --baseline "$QUALITY_4B_RESULTS/quality-alphabetical-score.json" \
      --candidate "$QUALITY_4B_RESULTS/quality-tooltrie_v0-score.json" \
      --metric "$METRIC" --bootstrap-samples 50000 --bootstrap-seed 42 \
      --sequence-state-dependent \
      --output "$RESUME_RESULTS/quality-4b-tooltrie_v0-vs-alphabetical-$METRIC.json"
    python scripts/compare_bfcl_quality.py \
      --baseline "$QUALITY_4B_RESULTS/quality-alphabetical-score.json" \
      --candidate "$QUALITY_4B_RESULTS/quality-contextpilot-online_incremental-score.json" \
      --metric "$METRIC" --bootstrap-samples 50000 --bootstrap-seed 42 \
      --sequence-state-dependent \
      --output "$RESUME_RESULTS/quality-4b-contextpilot-online_incremental-vs-alphabetical-$METRIC.json"
    python scripts/compare_bfcl_quality.py \
      --baseline "$QUALITY_4B_RESULTS/quality-tooltrie_v0-score.json" \
      --candidate "$QUALITY_4B_RESULTS/quality-contextpilot-online_incremental-score.json" \
      --metric "$METRIC" --bootstrap-samples 50000 --bootstrap-seed 42 \
      --sequence-state-dependent \
      --output "$RESUME_RESULTS/quality-4b-contextpilot-online_incremental-vs-tooltrie_v0-$METRIC.json"
  done
fi
```

Every regenerated comparison must report:

- `sequence_state_dependent: true`;
- `cluster_bootstrap_generalizes_across_request_sequences: false`;
- `mcnemar_independence_assumption_met: false`;
- 640 paired cases for name/full or 160 for no-tool.

## 6. Preserve, commit compact evidence, and hand over

Stop the 8B server and confirm the GPU is released. Then record provenance and
archive the new result directory without changing either accepted directory:

```bash
git rev-parse HEAD > "$RESUME_RESULTS/fyp-git-commit.txt"
git -C "$CONTEXTPILOT_REPO" rev-parse HEAD \
  > "$RESUME_RESULTS/contextpilot-git-commit.txt"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv \
  > "$RESUME_RESULTS/gpu-environment.csv"
RESUME_ARCHIVE="/home/taghan/contextpilot-static-refit-$RESUME_STAMP.tar.gz"
tar -czf "$RESUME_ARCHIVE" -C /home/taghan "$(basename "$RESUME_RESULTS")"
tar -tzf "$RESUME_ARCHIVE" >/dev/null
sha256sum "$RESUME_ARCHIVE" | tee "$RESUME_ARCHIVE.sha256"
```

Create a new branch in the detached worktree. Commit only a concise handover,
the six combined systems summaries, the compact static quality summary, the 15
corrected 8B comparison JSON files, any corrected 4B comparison JSON files, and
small provenance/configuration files. Do not commit replay JSONL/JSON, logs,
model files, raw score rows, `.venv`, or ignored dataset directories.

The handover must state:

- branch and commit;
- executed FYP and ContextPilot commits;
- 18/18 systems and 1/1 quality acceptance counts;
- all six equivalence guards;
- static-refit results beside the four previously accepted conditions;
- exact server commands, GPU, model, vLLM version, and live cache capacity;
- archive path, entry count, size, and SHA-256;
- confirmation that the original 75 accepted replays and 4B addendum were not
  rerun;
- confirmation that no annotation or eviction-feedback arm was executed;
- any deviation or quarantine, without silently weakening acceptance criteria.
