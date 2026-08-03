# NUS GPU instructions — ToolTrie external-comparison phase

Read this file completely before running commands. Execute it section by
section; do not paste the whole document into one shell. This phase has four
ordered objectives:

1. close the targeted BFCL irrelevance/no-tool question;
2. compare ToolTrie-v0 with CacheWeaver on the same stock vLLM server;
3. compare frozen FP-tree/co-occurrence and ContextPilot-derived orderings on
   that same server;
4. replay the identical workloads on stock SGLang/RadixAttention as a separate
   engine-level comparison.

The actual GPU measurements have **not** been run by the laptop session that
prepared this file. A successful local test run is not an experimental result.

## Experimental contract

- Repository: `https://github.com/taggyhan123/FYP.git`
- Branch: `tooltrie-v0-workflow`
- Preserve existing modified/untracked files. Never use `git reset --hard`,
  `git clean`, force-push, or reuse an existing result directory.
- Never use a GPU occupied by another user and never terminate another user's
  process.
- Use no `sudo`. All environments and external checkouts are user-owned.
- Only reorder an already-selected tool set. The builders and summarizer abort
  if a condition adds, removes, or duplicates a tool.
- Use **unmodified vLLM APC** for every prompt-layer comparison. Do not patch
  vLLM, CUDA, attention, or KV-cache internals.
- Run one replay driver per server. The replay scripts now reject global metric
  windows contaminated by other inference traffic.
- Keep raw workloads/replays under ignored `cluster/results/`. Write durable
  prose only under `reports/tooltrie-phase2/`; `reports/*.md` is generated and
  can be overwritten by `run_pipeline.py`.
- Do not call a confidence interval containing zero a pass or a regression.
  The no-tool analysis is estimation-first; an equivalence claim requires a
  supervisor-approved margin declared **before** looking at its result.

## What is upstream code and what is an adaptation?

| Condition | Implementation used here | Required label in report |
| --- | --- | --- |
| ToolTrie-v0 | This repository's causal recent-path planner | `ToolTrie-v0` |
| CacheWeaver | Faithful tool-ID transcription of paper Algorithm 1; the paper says code will be released upon acceptance, and no public implementation was found as checked on 2026-08-03 | `CacheWeaver Algorithm-1 reimplementation` |
| FP-tree conditional | This repository's training-only tool-order adaptation of FP-tree traversal | `FP-tree-derived adaptation`, not “FP-Growth implementation result” |
| Pair/triple conditional | Training-only unordered co-occurrence statistics | `pair/triple adaptation` |
| ContextPilot | Actual upstream code at commit `1fa0a143fdeda344585666648ab2b30cb7fea77f` | `ContextPilot offline/transductive` |
| SGLang | Official `v0.5.15.post1` package/tag, commit `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd` | `SGLang/RadixAttention engine` |

ContextPilot has two outputs. `contextpilot_intra` preserves the empirical
request sequence and belongs in the fixed-order table. Its
`contextpilot_intra_schedule` variant enables the inter-context scheduler and
may change request order, so it gets a separate scheduling table. The builder
records whether the mapping actually changed; an identity mapping is a valid
negative result. It is invalid to attribute any scheduling gain to
within-request tool ordering.

SGLang results must also be reported separately. Compare relative behavior
within each engine; do not present a vLLM-versus-SGLang latency difference as a
pure radix-tree effect because the entire serving engine changes.

## 1. Synchronize and verify

The current project checkout is expected at `/home/taghan/FYP`. If it is
elsewhere, set `FYP_REPO` to its real path.

```bash
FYP_REPO=/home/taghan/FYP
cd "$FYP_REPO"
git status --short --branch
git fetch origin
git switch tooltrie-v0-workflow 2>/dev/null || git switch --track origin/tooltrie-v0-workflow
git pull --ff-only
git log -1 --oneline
```

If tracked local changes conflict with the pull, stop and preserve them; use a
separate clone/worktree rather than discarding them.

Use the project/client environment for builders, replay clients, and scoring.
Do not run a normal exact `uv sync` inside the separate vLLM server environment.

```bash
cd "$FYP_REPO"
source .venv/bin/activate
python --version
python -m pytest -q
python scripts/build_tooltrie_workload.py --help >/dev/null
python scripts/build_contextpilot_workload.py --help >/dev/null
python scripts/replay_sglang_workload.py --help >/dev/null
```

Verify data and create one unique result directory:

```bash
test -s data/processed/tools.jsonl
test -s data/processed/tasks.jsonl
test -d data/raw/bfcl/possible_answer
PHASE2_STAMP=$(date +%Y%m%d-%H%M%S)
PHASE2_RESULTS="$FYP_REPO/cluster/results/tooltrie-phase2-$PHASE2_STAMP"
mkdir -p "$PHASE2_RESULTS"
printf '%s\n' "$PHASE2_RESULTS"
```

Keep `FYP_REPO` and `PHASE2_RESULTS` exported or re-create them exactly in every
new tmux pane. Never point them at a previous result directory.

## 2. Select a GPU and locate the vLLM executable

```bash
nvidia-smi
```

Set `GPU_ID` to an idle physical GPU only. The previous run used vLLM from
`/home/taghan/tatm/.venv`; verify rather than assume it still exists.

```bash
GPU_ID=REPLACE_WITH_IDLE_GPU_NUMBER
VLLM_BIN=/home/taghan/tatm/.venv/bin/vllm
test -x "$VLLM_BIN"
"$VLLM_BIN" --version
```

Record the selected GPU and both Python environments in
`$PHASE2_RESULTS/environment.txt`.

## 3. Targeted no-tool evaluation — run this first

### 3.1 Start stock vLLM with Qwen3-8B

Use a dedicated tmux pane. The command blocks while the server runs.

```bash
cd "$FYP_REPO"
PATH=/home/taghan/tatm/.venv/bin:$PATH VLLM_SERVER_DEV_MODE=1 VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES="$GPU_ID" CPATH=/home/taghan/miniconda3/envs/hdr312/include/python3.12 "$VLLM_BIN" serve Qwen/Qwen3-8B --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser hermes --host 127.0.0.1 --port 8000 2>&1 | tee "$PHASE2_RESULTS/vllm-qwen3-8b.log"
```

In the client pane, wait for readiness and read back the real cache capacity:

```bash
cd "$FYP_REPO"
source .venv/bin/activate
curl -fsS http://127.0.0.1:8000/v1/models >/dev/null
python scripts/inspect_vllm_server.py --base-url http://127.0.0.1:8000 | tee "$PHASE2_RESULTS/vllm-qwen3-8b-cache.json"
VLLM_CAPACITY_8B=$(python scripts/inspect_vllm_server.py --base-url http://127.0.0.1:8000 --capacity-only)
printf '8B capacity tokens: %s\n' "$VLLM_CAPACITY_8B"
```

Do not reuse the historical value `44656` unless the live server reports it.

### 3.2 Build every available irrelevance task under five fixed menus

BFCL contains only 240 unique irrelevance tasks. Five menu seeds produce 1,200
paired menu realizations, but the analysis correctly treats the 240 tasks—not
the 1,200 rows—as independent bootstrap clusters. Each seed starts a fresh
causal planner.

```bash
NO_TOOL_SEEDS=(7 42 101 202 303)
for SEED in "${NO_TOOL_SEEDS[@]}"; do
  python scripts/build_bfcl_quality_workload.py --ordering original --domain irrelevance --per-domain 240 --menu-size 64 --random-seed "$SEED" --output "$PHASE2_RESULTS/no-tool-original-seed-$SEED.jsonl"
  python scripts/build_bfcl_quality_workload.py --ordering alphabetical --domain irrelevance --per-domain 240 --menu-size 64 --random-seed "$SEED" --output "$PHASE2_RESULTS/no-tool-alphabetical-seed-$SEED.jsonl"
  python scripts/build_tooltrie_workload.py --input "$PHASE2_RESULTS/no-tool-original-seed-$SEED.jsonl" --policy tooltrie_v0 --fallback alphabetical --recency-window 128 --capacity-tokens "$VLLM_CAPACITY_8B" --output "$PHASE2_RESULTS/no-tool-tooltrie-seed-$SEED.jsonl"
done
```

### 3.3 Replay sequentially and score

`--reset-before` performs exactly one APC reset immediately before each full
condition. Do not send any other model request until that replay finishes.

```bash
for SEED in "${NO_TOOL_SEEDS[@]}"; do
  python scripts/replay_vllm_workload.py --input "$PHASE2_RESULTS/no-tool-alphabetical-seed-$SEED.jsonl" --run-label "no-tool-alphabetical-seed-$SEED" --max-tokens 128 --disable-thinking --reset-before --output "$PHASE2_RESULTS/no-tool-alphabetical-seed-$SEED-replay.json"
  python scripts/score_bfcl_quality.py --replay-result "$PHASE2_RESULTS/no-tool-alphabetical-seed-$SEED-replay.json" --output "$PHASE2_RESULTS/no-tool-alphabetical-seed-$SEED-score.json"
  python scripts/replay_vllm_workload.py --input "$PHASE2_RESULTS/no-tool-tooltrie-seed-$SEED.jsonl" --run-label "no-tool-tooltrie-seed-$SEED" --max-tokens 128 --disable-thinking --reset-before --output "$PHASE2_RESULTS/no-tool-tooltrie-seed-$SEED-replay.json"
  python scripts/score_bfcl_quality.py --replay-result "$PHASE2_RESULTS/no-tool-tooltrie-seed-$SEED-replay.json" --output "$PHASE2_RESULTS/no-tool-tooltrie-seed-$SEED-score.json"
done
```

If a replay exits on a counter mismatch, preserve it under a `quarantine/`
subdirectory, inspect whether other traffic reached the server, and rerun it.
Never bypass the check for a result used in a table.

Compute the paired estimate without an equivalence margin:

```bash
ALPHA_NO_TOOL_SCORES=()
TOOLTRIE_NO_TOOL_SCORES=()
for SEED in "${NO_TOOL_SEEDS[@]}"; do
  ALPHA_NO_TOOL_SCORES+=("$PHASE2_RESULTS/no-tool-alphabetical-seed-$SEED-score.json")
  TOOLTRIE_NO_TOOL_SCORES+=("$PHASE2_RESULTS/no-tool-tooltrie-seed-$SEED-score.json")
done
python scripts/compare_bfcl_quality.py --baseline "${ALPHA_NO_TOOL_SCORES[@]}" --candidate "${TOOLTRIE_NO_TOOL_SCORES[@]}" --metric no_tool_correct --bootstrap-samples 50000 --bootstrap-seed 42 --output "$PHASE2_RESULTS/no-tool-tooltrie-vs-alphabetical.json"
```

Required wording: report the point difference, task-clustered 95% CI, 240
unique tasks, 1,200 paired menu cases, and discordant counts. Do not multiply
the effective sample size by five and do not add a post-hoc margin.

Stop the 8B server cleanly after these files are complete.

## 4. Build the external ordering workloads

### 4.1 Start the same stock vLLM setup with Qwen3-0.6B

Use the same idle physical GPU and the same vLLM environment:

```bash
cd "$FYP_REPO"
PATH=/home/taghan/tatm/.venv/bin:$PATH VLLM_SERVER_DEV_MODE=1 VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES="$GPU_ID" CPATH=/home/taghan/miniconda3/envs/hdr312/include/python3.12 "$VLLM_BIN" serve Qwen/Qwen3-0.6B --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser hermes --host 127.0.0.1 --port 8000 2>&1 | tee "$PHASE2_RESULTS/vllm-qwen3-0.6b.log"
```

Read the live capacity in the client pane:

```bash
cd "$FYP_REPO"
source .venv/bin/activate
curl -fsS http://127.0.0.1:8000/v1/models >/dev/null
python scripts/inspect_vllm_server.py --base-url http://127.0.0.1:8000 | tee "$PHASE2_RESULTS/vllm-qwen3-0.6b-cache.json"
VLLM_CAPACITY_06B=$(python scripts/inspect_vllm_server.py --base-url http://127.0.0.1:8000 --capacity-only)
printf '0.6B capacity tokens: %s\n' "$VLLM_CAPACITY_06B"
```

### 4.2 Pull and pin the actual ContextPilot source

Use a separate CPU environment so ContextPilot's dependencies and hook do not
modify the vLLM server environment.

```bash
REFERENCE_ROOT=/home/taghan/external-references
CONTEXTPILOT_REPO="$REFERENCE_ROOT/ContextPilot"
CONTEXTPILOT_VENV=/home/taghan/venvs/contextpilot-1fa0a143
mkdir -p "$REFERENCE_ROOT" /home/taghan/venvs
if [ ! -d "$CONTEXTPILOT_REPO/.git" ]; then git clone https://github.com/EfficientContext/ContextPilot.git "$CONTEXTPILOT_REPO"; fi
test -z "$(git -C "$CONTEXTPILOT_REPO" status --porcelain)"
git -C "$CONTEXTPILOT_REPO" fetch origin
git -C "$CONTEXTPILOT_REPO" switch --detach 1fa0a143fdeda344585666648ab2b30cb7fea77f
test "$(git -C "$CONTEXTPILOT_REPO" rev-parse HEAD)" = 1fa0a143fdeda344585666648ab2b30cb7fea77f
python3.12 -m venv "$CONTEXTPILOT_VENV"
"$CONTEXTPILOT_VENV/bin/python" -m pip install --upgrade pip
"$CONTEXTPILOT_VENV/bin/python" -m pip install -e "$CONTEXTPILOT_REPO"
"$CONTEXTPILOT_VENV/bin/python" -c 'import contextpilot; print(contextpilot.__file__)'
```

If the reference checkout already contains changes, do not overwrite them;
make a second clean clone with a different explicit path.

### 4.3 Build matched BFCL and ToolRet conditions

Evaluation is the same first 200 tasks and same fixed 64-tool catalog used by
ToolTrie-v0. Fitted policies use only later records in their native benchmark
menus; evaluation task IDs are therefore disjoint and shared-catalog
distractors cannot manufacture pair/triple support. Native ToolRet IDs are gold
requirements, while native BFCL IDs are exposed menus—not observed calls.

```bash
for DATASET in bfcl toolret; do
  python scripts/build_cluster_workload.py --partition "$DATASET" --ordering original --offset 0 --limit 200 --menu-size 64 --random-seed 42 --output "$PHASE2_RESULTS/$DATASET-original.jsonl"
  python scripts/build_cluster_workload.py --partition "$DATASET" --ordering alphabetical --offset 0 --limit 200 --menu-size 64 --random-seed 42 --output "$PHASE2_RESULTS/$DATASET-alphabetical.jsonl"
  python scripts/build_cluster_workload.py --partition "$DATASET" --ordering original --offset 200 --limit 100000 --menu-size 0 --random-seed 42 --output "$PHASE2_RESULTS/$DATASET-training-native.jsonl"
  python scripts/build_tooltrie_workload.py --input "$PHASE2_RESULTS/$DATASET-original.jsonl" --policy tooltrie_v0 --fallback alphabetical --recency-window 128 --capacity-tokens "$VLLM_CAPACITY_06B" --output "$PHASE2_RESULTS/$DATASET-tooltrie_v0.jsonl"
  python scripts/build_tooltrie_workload.py --input "$PHASE2_RESULTS/$DATASET-original.jsonl" --policy cacheweaver --recency-window 128 --output "$PHASE2_RESULTS/$DATASET-cacheweaver.jsonl"
  for POLICY in frequency_fitted schema_cost_fitted fp_tree_conditional conditional_pair conditional_pair_triple; do
    python scripts/build_fitted_ordering_workload.py --input "$PHASE2_RESULTS/$DATASET-original.jsonl" --training-input "$PHASE2_RESULTS/$DATASET-training-native.jsonl" --policy "$POLICY" --output "$PHASE2_RESULTS/$DATASET-$POLICY.jsonl"
  done
  "$CONTEXTPILOT_VENV/bin/python" scripts/build_contextpilot_workload.py --input "$PHASE2_RESULTS/$DATASET-original.jsonl" --contextpilot-repo "$CONTEXTPILOT_REPO" --mode intra --output "$PHASE2_RESULTS/$DATASET-contextpilot_intra.jsonl"
  "$CONTEXTPILOT_VENV/bin/python" scripts/build_contextpilot_workload.py --input "$PHASE2_RESULTS/$DATASET-original.jsonl" --contextpilot-repo "$CONTEXTPILOT_REPO" --mode intra_schedule --output "$PHASE2_RESULTS/$DATASET-contextpilot_intra_schedule.jsonl"
done
```

Interpretation rules:

- `frequency_fitted` and `schema_cost_fitted` answer whether frozen training
  frequency or frequency × schema length is better.
- `conditional_pair` / `conditional_pair_triple` test additional co-occurrence
  structure. ToolRet supports a gold co-requirement interpretation; BFCL only
  supports menu co-exposure. Neither is a production execution trace.
- `fp_tree_conditional` is the explicit FP-tree-derived baseline.
- ContextPilot is an offline batch upper/reference baseline. It sees the full
  evaluation batch and is not a causal online deployment policy.

## 5. Same-server vLLM systems comparison

Run three trials per fixed-sequence condition. The cache resets inside the
client, so do not add separate warmup requests between reset and replay.

```bash
FIXED_CONDITIONS=(original alphabetical tooltrie_v0 cacheweaver frequency_fitted schema_cost_fitted fp_tree_conditional conditional_pair conditional_pair_triple contextpilot_intra)
for DATASET in bfcl toolret; do
  for CONDITION in "${FIXED_CONDITIONS[@]}"; do
    for TRIAL in 1 2 3; do
      python scripts/replay_vllm_workload.py --input "$PHASE2_RESULTS/$DATASET-$CONDITION.jsonl" --run-label "$DATASET-$CONDITION-vllm-trial-$TRIAL" --max-tokens 48 --disable-thinking --reset-before --output "$PHASE2_RESULTS/$DATASET-$CONDITION-vllm-trial-$TRIAL.json"
    done
  done
done
```

Run the scheduler-enabled ContextPilot variant separately:

```bash
for DATASET in bfcl toolret; do
  for TRIAL in 1 2 3; do
    python scripts/replay_vllm_workload.py --input "$PHASE2_RESULTS/$DATASET-contextpilot_intra_schedule.jsonl" --run-label "$DATASET-contextpilot_intra_schedule-vllm-trial-$TRIAL" --max-tokens 48 --disable-thinking --reset-before --output "$PHASE2_RESULTS/$DATASET-contextpilot_intra_schedule-vllm-trial-$TRIAL.json"
  done
done
```

Generate clean summaries with Student-t 95% intervals:

```bash
for DATASET in bfcl toolret; do
  SUMMARY_ARGS=()
  for CONDITION in "${FIXED_CONDITIONS[@]}"; do
    for TRIAL in 1 2 3; do
      SUMMARY_ARGS+=(--run "$CONDITION=$PHASE2_RESULTS/$DATASET-$CONDITION-vllm-trial-$TRIAL.json")
    done
  done
  python scripts/summarize_ordering_replays.py "${SUMMARY_ARGS[@]}" --output "$PHASE2_RESULTS/$DATASET-vllm-fixed-summary.json"
  SCHEDULE_ARGS=()
  for CONDITION in contextpilot_intra contextpilot_intra_schedule; do
    for TRIAL in 1 2 3; do
      SCHEDULE_ARGS+=(--run "$CONDITION=$PHASE2_RESULTS/$DATASET-$CONDITION-vllm-trial-$TRIAL.json")
    done
  done
  python scripts/summarize_ordering_replays.py "${SCHEDULE_ARGS[@]}" --output "$PHASE2_RESULTS/$DATASET-vllm-contextpilot-schedule-summary.json"
done
```

For the fixed table, emphasize measured cached ratio, computed prefill tokens,
prefill time, TTFT, and wall time. CacheWeaver's main comparison is against
ToolTrie-v0 under this identical engine/model/workload. Do not compare the
paper's published absolute percentage directly to this different workload.

## 6. BFCL correctness for the external orderings

Stop the 0.6B server and restart the exact 8B server from section 3. Re-read
`VLLM_CAPACITY_8B`; do not assume the earlier shell variable survived.

In the server pane:

```bash
cd "$FYP_REPO"
PATH=/home/taghan/tatm/.venv/bin:$PATH VLLM_SERVER_DEV_MODE=1 VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES="$GPU_ID" CPATH=/home/taghan/miniconda3/envs/hdr312/include/python3.12 "$VLLM_BIN" serve Qwen/Qwen3-8B --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser hermes --host 127.0.0.1 --port 8000 2>&1 | tee "$PHASE2_RESULTS/vllm-qwen3-8b-quality.log"
```

In the client pane:

```bash
curl -fsS http://127.0.0.1:8000/v1/models >/dev/null
VLLM_CAPACITY_8B=$(python scripts/inspect_vllm_server.py --base-url http://127.0.0.1:8000 --capacity-only)
printf '8B capacity tokens: %s\n' "$VLLM_CAPACITY_8B"
```

Use 160 tasks from each domain (n=800). This leaves 440 disjoint BFCL tasks for
training the fitted policies. A larger transductive n=1000 test would leave
almost no balanced training data, while n=100 is already known to be an
unreliable pilot.

```bash
python scripts/build_bfcl_quality_workload.py --ordering original --per-domain 160 --menu-size 64 --random-seed 42 --output "$PHASE2_RESULTS/quality-original.jsonl"
python scripts/build_bfcl_quality_workload.py --ordering alphabetical --per-domain 160 --menu-size 64 --random-seed 42 --output "$PHASE2_RESULTS/quality-alphabetical.jsonl"
python scripts/build_bfcl_quality_workload.py --ordering original --per-domain 100000 --menu-size 0 --random-seed 42 --output "$PHASE2_RESULTS/quality-training-all-native.jsonl"
python scripts/build_tooltrie_workload.py --input "$PHASE2_RESULTS/quality-original.jsonl" --policy tooltrie_v0 --fallback alphabetical --recency-window 128 --capacity-tokens "$VLLM_CAPACITY_8B" --output "$PHASE2_RESULTS/quality-tooltrie_v0.jsonl"
python scripts/build_tooltrie_workload.py --input "$PHASE2_RESULTS/quality-original.jsonl" --policy cacheweaver --recency-window 128 --output "$PHASE2_RESULTS/quality-cacheweaver.jsonl"
for POLICY in fp_tree_conditional conditional_pair_triple; do
  python scripts/build_fitted_ordering_workload.py --input "$PHASE2_RESULTS/quality-original.jsonl" --training-input "$PHASE2_RESULTS/quality-training-all-native.jsonl" --exclude-evaluation-task-ids --policy "$POLICY" --output "$PHASE2_RESULTS/quality-$POLICY.jsonl"
done
"$CONTEXTPILOT_VENV/bin/python" scripts/build_contextpilot_workload.py --input "$PHASE2_RESULTS/quality-original.jsonl" --contextpilot-repo "$CONTEXTPILOT_REPO" --mode intra --output "$PHASE2_RESULTS/quality-contextpilot_intra.jsonl"
```

The required quality set is alphabetical, ToolTrie-v0, CacheWeaver,
FP-tree-conditional, pair/triple-conditional, and ContextPilot-intra. One
deterministic pass per condition spends compute on task coverage rather than
repeating identical `temperature=0, seed=0` generations.

```bash
QUALITY_CONDITIONS=(alphabetical tooltrie_v0 cacheweaver fp_tree_conditional conditional_pair_triple contextpilot_intra)
for CONDITION in "${QUALITY_CONDITIONS[@]}"; do
  python scripts/replay_vllm_workload.py --input "$PHASE2_RESULTS/quality-$CONDITION.jsonl" --run-label "quality-$CONDITION-qwen3-8b" --max-tokens 128 --disable-thinking --reset-before --output "$PHASE2_RESULTS/quality-$CONDITION-replay.json"
  python scripts/score_bfcl_quality.py --replay-result "$PHASE2_RESULTS/quality-$CONDITION-replay.json" --output "$PHASE2_RESULTS/quality-$CONDITION-score.json"
done
```

Compute candidate-minus-alphabetical paired intervals for all three metrics:

```bash
for CONDITION in tooltrie_v0 cacheweaver fp_tree_conditional conditional_pair_triple contextpilot_intra; do
  for METRIC in name_correct full_correct no_tool_correct; do
    python scripts/compare_bfcl_quality.py --baseline "$PHASE2_RESULTS/quality-alphabetical-score.json" --candidate "$PHASE2_RESULTS/quality-$CONDITION-score.json" --metric "$METRIC" --bootstrap-samples 50000 --bootstrap-seed 42 --output "$PHASE2_RESULTS/quality-$CONDITION-vs-alphabetical-$METRIC.json"
  done
done
```

These intervals estimate detectable differences; they do not establish the old
±1 percentage-point gate. The targeted 240-task/five-menu result from section 3
remains the primary no-tool analysis.

Stop vLLM completely before starting SGLang.

## 7. Separate SGLang/RadixAttention engine comparison

### 7.1 Pull the tagged source and install the official release in isolation

```bash
SGLANG_REPO=/home/taghan/external-references/sglang-v0.5.15.post1
SGLANG_VENV=/home/taghan/venvs/sglang-0.5.15.post1
if [ ! -d "$SGLANG_REPO/.git" ]; then git clone --branch v0.5.15.post1 --depth 1 https://github.com/sgl-project/sglang.git "$SGLANG_REPO"; fi
test -z "$(git -C "$SGLANG_REPO" status --porcelain)"
test "$(git -C "$SGLANG_REPO" rev-parse 'v0.5.15.post1^{}')" = 0b3bb0cbe31873994c9f989fddfe2f87ca839fdd
python3.12 -m venv "$SGLANG_VENV"
"$SGLANG_VENV/bin/python" -m pip install --upgrade pip
"$SGLANG_VENV/bin/python" -m pip install 'sglang[all]==0.5.15.post1'
"$SGLANG_VENV/bin/python" -c 'import sglang; print(sglang.__version__, sglang.__file__)'
```

The source checkout records the exact official tag; the official wheel is used
for reproducible CUDA/runtime installation. Do not install SGLang into either
the project venv or the vLLM server venv.

### 7.2 Start SGLang with Qwen3-0.6B

In a dedicated server pane:

```bash
cd "$FYP_REPO"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$SGLANG_VENV/bin/python" -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --host 127.0.0.1 --port 30000 --enable-metrics --enable-cache-report --tool-call-parser qwen --mem-fraction-static 0.85 2>&1 | tee "$PHASE2_RESULTS/sglang-qwen3-0.6b.log"
```

Verify all required surfaces before a measurement:

```bash
curl -fsS http://127.0.0.1:30000/v1/models >/dev/null
curl -fsS http://127.0.0.1:30000/metrics | grep 'sglang:prompt_tokens_total' | head -n 1
curl -fsS -X POST http://127.0.0.1:30000/flush_cache
```

If `--tool-call-parser qwen` is rejected by the installed release, inspect
`python -m sglang.launch_server --help` and record the supported Qwen parser;
do not silently change model, chat template, or tool semantics.

### 7.3 Replay the identical prebuilt files

The inputs below are byte-for-byte the vLLM comparison workloads from section
4. Only the engine and measured tokenization/cache behavior change.

```bash
cd "$FYP_REPO"
source .venv/bin/activate
FIXED_CONDITIONS=(original alphabetical tooltrie_v0 cacheweaver frequency_fitted schema_cost_fitted fp_tree_conditional conditional_pair conditional_pair_triple contextpilot_intra)
for DATASET in bfcl toolret; do
  for CONDITION in "${FIXED_CONDITIONS[@]}"; do
    for TRIAL in 1 2 3; do
      python scripts/replay_sglang_workload.py --input "$PHASE2_RESULTS/$DATASET-$CONDITION.jsonl" --run-label "$DATASET-$CONDITION-sglang-trial-$TRIAL" --max-tokens 48 --disable-thinking --flush-before --output "$PHASE2_RESULTS/$DATASET-$CONDITION-sglang-trial-$TRIAL.json"
    done
  done
  for TRIAL in 1 2 3; do
    python scripts/replay_sglang_workload.py --input "$PHASE2_RESULTS/$DATASET-contextpilot_intra_schedule.jsonl" --run-label "$DATASET-contextpilot_intra_schedule-sglang-trial-$TRIAL" --max-tokens 48 --disable-thinking --flush-before --output "$PHASE2_RESULTS/$DATASET-contextpilot_intra_schedule-sglang-trial-$TRIAL.json"
  done
done
```

Summarize SGLang separately:

```bash
for DATASET in bfcl toolret; do
  SUMMARY_ARGS=()
  for CONDITION in "${FIXED_CONDITIONS[@]}"; do
    for TRIAL in 1 2 3; do
      SUMMARY_ARGS+=(--run "$CONDITION=$PHASE2_RESULTS/$DATASET-$CONDITION-sglang-trial-$TRIAL.json")
    done
  done
  python scripts/summarize_ordering_replays.py "${SUMMARY_ARGS[@]}" --output "$PHASE2_RESULTS/$DATASET-sglang-fixed-summary.json"
  SCHEDULE_ARGS=()
  for CONDITION in contextpilot_intra contextpilot_intra_schedule; do
    for TRIAL in 1 2 3; do
      SCHEDULE_ARGS+=(--run "$CONDITION=$PHASE2_RESULTS/$DATASET-$CONDITION-sglang-trial-$TRIAL.json")
    done
  done
  python scripts/summarize_ordering_replays.py "${SCHEDULE_ARGS[@]}" --output "$PHASE2_RESULTS/$DATASET-sglang-contextpilot-schedule-summary.json"
done
```

The report should compare each ordering's relative cache reuse and TTFT within
SGLang, then contrast those relative patterns with vLLM. It must not claim that
SGLang itself is faster/slower solely because of RadixAttention.

## 8. Capture provenance, report, and preserve raw results

Capture at minimum:

```bash
git rev-parse HEAD > "$PHASE2_RESULTS/fyp-git-commit.txt"
git -C "$CONTEXTPILOT_REPO" rev-parse HEAD > "$PHASE2_RESULTS/contextpilot-git-commit.txt"
git -C "$SGLANG_REPO" rev-parse 'v0.5.15.post1^{}' > "$PHASE2_RESULTS/sglang-git-commit.txt"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv > "$PHASE2_RESULTS/gpu-environment.csv"
python --version > "$PHASE2_RESULTS/client-python.txt"
```

Also save the exact server commands, Python/package versions, model IDs, cache
capacities, and whether any retry/quarantine occurred.

Write durable findings to
`reports/tooltrie-phase2/findings.md`. Required sections:

1. targeted no-tool result with clustered uncertainty;
2. vLLM fixed-request-order table (three trials and 95% intervals);
3. separate ContextPilot scheduling table;
4. n=800 BFCL correctness table and paired intervals;
5. separate SGLang/RadixAttention table;
6. positive and negative regimes, limitations, failures, and contamination;
7. exact upstream/adaptation provenance from the table at the start of this
   file.

Do not put raw replays or generated workloads in Git. Preserve them outside the
repository with a verified archive:

```bash
RAW_ARCHIVE="/home/taghan/tooltrie-phase2-$PHASE2_STAMP-raw.tar.gz"
tar -czf "$RAW_ARCHIVE" -C "$(dirname "$PHASE2_RESULTS")" "$(basename "$PHASE2_RESULTS")"
sha256sum "$RAW_ARCHIVE" | tee "$RAW_ARCHIVE.sha256"
ls -lh "$RAW_ARCHIVE" "$RAW_ARCHIVE.sha256"
```

Before committing, run tests again and inspect `git status`. Commit only the
small findings/status updates; preserve all pre-existing files and keep the raw
archive on the GPU server.
