# NUS GPU instructions — close the initial research brief

Read this file and `cluster/initial-brief-closure-manifest.json` completely
before running commands. This procedure closes the remaining evidence gaps in
Tasks B, D, E, and F. It does **not** begin the §9 retention extensions.

## Experimental contract

- Pull and verify the exact commit supplied in the handover. Do not run from an
  older commit, including `80bf2e3`.
- Never use a GPU occupied by another user, terminate another user's process,
  use `sudo`, patch vLLM/CUDA, or alter the declared method after seeing results.
- Use one isolated replay client per server. Other inference traffic invalidates
  the global counter windows.
- Use a new result directory. Never overwrite an earlier result or hide a failed
  condition.
- ToolRet gold IDs are evaluation metadata only. BM25 selects menu membership;
  ordering may only permute that selected set.
- `original` is the predeclared fallback: BM25-ranked selected tools sent through
  the ordinary OpenAI `tools` text-prefill path, with no inactive tools and no
  KV-tensor modification.
- Raw outputs stay under ignored `cluster/results/` and in a checksummed archive.
  Only compact summaries and a handover are committed.

## 1. Synchronize and verify the pinned commit

The person handing off this run must replace `REPLACE_WITH_PINNED_COMMIT` with
the full commit printed by the local Codex session.

```bash
export FYP_REPO=/home/taghan/FYP
export EXPECTED_FYP_COMMIT=REPLACE_WITH_PINNED_COMMIT
cd "$FYP_REPO"
git status --short --branch
git fetch origin
git switch tooltrie-v0-workflow 2>/dev/null || git switch --track origin/tooltrie-v0-workflow
git pull --ff-only origin tooltrie-v0-workflow
test "$(git rev-parse HEAD)" = "$EXPECTED_FYP_COMMIT"
git log -1 --oneline
```

If tracked changes prevent a fast-forward, stop and use a separate clone or
worktree. Do not reset, clean, stash, or discard another session's work.

Use the project client environment and verify the code before allocating a GPU:

```bash
cd "$FYP_REPO"
source .venv/bin/activate
python --version
python -m pytest -q
python -m json.tool cluster/initial-brief-closure-manifest.json >/dev/null
python scripts/replay_vllm_workload.py --help >/dev/null
python scripts/audit_rendered_prefix.py --help >/dev/null
python scripts/locality_replay.py --help >/dev/null
```

Verify the processed datasets. If either file is missing, follow `README.md` to
download and regenerate them before continuing.

```bash
test -s data/processed/tools.jsonl
test -s data/processed/tasks.jsonl
```

## 2. Select an idle GPU and create a unique result directory

```bash
nvidia-smi
```

Set `GPU_ID` manually to an idle physical GPU. The following command must not be
run with the placeholder unchanged.

```bash
export GPU_ID=REPLACE_WITH_IDLE_GPU_NUMBER
export VLLM_BIN=/home/taghan/tatm/.venv/bin/vllm
test -x "$VLLM_BIN"
export BRIEF_STAMP=$(date +%Y%m%d-%H%M%S)
export BRIEF_RESULTS="$FYP_REPO/cluster/results/initial-brief-closure-$BRIEF_STAMP"
test ! -e "$BRIEF_RESULTS"
mkdir -p "$BRIEF_RESULTS"/{workloads,replays,audits,pressure,summaries}
printf '%s\n' "$BRIEF_RESULTS"
```

Record provenance before starting the server:

```bash
{
  date --iso-8601=seconds
  git rev-parse HEAD
  git status --short --branch
  uname -a
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv
  /home/taghan/tatm/.venv/bin/python -c 'import torch, vllm; print("torch", torch.__version__); print("vllm", vllm.__version__)'
} | tee "$BRIEF_RESULTS/environment.txt"
```

## 3. Start the pinned systems server

Start this in a dedicated tmux pane. It blocks while serving. Use exactly the
same Qwen3-0.6B/vLLM arm as the existing systems measurements.

```bash
cd "$FYP_REPO"
PATH=/home/taghan/tatm/.venv/bin:$PATH \
VLLM_SERVER_DEV_MODE=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
CPATH=/home/taghan/miniconda3/envs/hdr312/include/python3.12 \
"$VLLM_BIN" serve Qwen/Qwen3-0.6B \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --host 127.0.0.1 \
  --port 8000 2>&1 | tee "$BRIEF_RESULTS/vllm-qwen3-0.6b.log"
```

In the client pane, wait for readiness and read back the live configuration.
Do not reuse a historical cache-capacity number.

```bash
cd "$FYP_REPO"
source .venv/bin/activate
curl -fsS http://127.0.0.1:8000/v1/models | tee "$BRIEF_RESULTS/models.json"
python scripts/inspect_vllm_server.py | tee "$BRIEF_RESULTS/cache-config.json"
export VLLM_CAPACITY=$(python scripts/inspect_vllm_server.py --capacity-only)
printf 'live capacity tokens: %s\n' "$VLLM_CAPACITY"
curl -fsS http://127.0.0.1:8000/metrics > "$BRIEF_RESULTS/metrics-before.txt"
```

Stop if `enable_prefix_caching` is not reported as true or the real block size
and GPU-block count are absent.

## 4. Build the retrieved-menu matrix

Generate all four menu sizes and the six brief ordering baselines. The builder
recomputes deterministic BM25 membership for each output; the replay summarizer
later verifies that every ordering kept the same selected set.

```bash
cd "$FYP_REPO"
source .venv/bin/activate
for menu_size in 4 16 64 128; do
  mkdir -p "$BRIEF_RESULTS/workloads/k$menu_size"
  for ordering in original alphabetical random frequency schema_cost_weighted fp_tree_global; do
    python scripts/build_retrieved_tool_workload.py \
      --menu-size "$menu_size" --offset 0 --limit 200 \
      --ordering "$ordering" --random-seed 42 --support-mode disjoint \
      --output "$BRIEF_RESULTS/workloads/k$menu_size/$ordering.jsonl"
  done
  python scripts/build_tooltrie_workload.py \
    --input "$BRIEF_RESULTS/workloads/k$menu_size/original.jsonl" \
    --fallback alphabetical --recency-window 128 \
    --capacity-tokens "$VLLM_CAPACITY" \
    --output "$BRIEF_RESULTS/workloads/k$menu_size/tooltrie_v0.jsonl"
done
```

Do not call ToolRet benchmark frequency production popularity. The frequency
and schema-cost support here comes from a disjoint benchmark split.

## 5. Primary replay: three clean trials per condition

These runs collect latency without background KV scraping. Each complete
condition starts from one clean APC reset. `original` is explicitly tagged as
the ordinary text-prefill fallback.

```bash
cd "$FYP_REPO"
source .venv/bin/activate
for menu_size in 4 16 64 128; do
  mkdir -p "$BRIEF_RESULTS/replays/k$menu_size"
  for condition in original alphabetical random frequency schema_cost_weighted fp_tree_global tooltrie_v0; do
    role=ordering_candidate
    if [ "$condition" = original ]; then
      role=ordinary_text_prefill_fallback
    fi
    for trial in 1 2 3; do
      output="$BRIEF_RESULTS/replays/k$menu_size/$condition-trial-$trial.json"
      test ! -e "$output"
      python scripts/replay_vllm_workload.py \
        --input "$BRIEF_RESULTS/workloads/k$menu_size/$condition.jsonl" \
        --run-label "retrieved-k$menu_size-$condition-trial-$trial" \
        --condition-role "$role" \
        --max-tokens 1 --disable-thinking --reset-before \
        --output "$output"
    done
  done
done
```

Summarize each paired menu-size matrix. The summarizer rejects dirty counters,
missing resets, changed case sets, changed selected-tool membership, and drift
between repeated trials.

```bash
for menu_size in 4 16 64 128; do
  run_args=()
  for condition in original alphabetical random frequency schema_cost_weighted fp_tree_global tooltrie_v0; do
    for trial in 1 2 3; do
      run_args+=(--run "$condition=$BRIEF_RESULTS/replays/k$menu_size/$condition-trial-$trial.json")
    done
  done
  python scripts/summarize_ordering_replays.py \
    "${run_args[@]}" \
    --output "$BRIEF_RESULTS/summaries/retrieved-k$menu_size-summary.json"
done
```

For the direct-partial requirement, inspect `direct_reuse_buckets.partial_all`
for the predeclared k64 `alphabetical` and `tooltrie_v0` conditions. If either
condition has no actual partial-reuse requests, report the missing stratum; do
not interpolate between cold and warm endpoints or silently select a different
condition after seeing results.

## 6. Exact rendered-token and cache-block audit

Audit the k64 workloads used above. This stores exact server-rendered token IDs,
block boundaries, best prior full-block prefixes, actual cached tokens, prefill,
and TTFT.

```bash
mkdir -p "$BRIEF_RESULTS/audits/k64"
for condition in original alphabetical random frequency schema_cost_weighted fp_tree_global tooltrie_v0; do
  output="$BRIEF_RESULTS/audits/k64/$condition.json"
  test ! -e "$output"
  python scripts/audit_rendered_prefix.py \
    --input "$BRIEF_RESULTS/workloads/k64/$condition.jsonl" \
    --run-label "rendered-audit-k64-$condition" \
    --measure --max-tokens 1 --disable-thinking \
    --output "$output"
done
```

Every audit must record `validation.clean=true`. In particular, `/tokenize`
token counts must match completion prompt usage and cached plus computed tokens
must equal the rendered prompt count.

## 7. Separate demonstrated-pressure runs

KV scraping can perturb latency, so these runs are not used for the primary
latency table. Each static ordering is tested across all four declared replay
regimes. The script preserves the output and exits non-zero if any regime fails
to reach 90% sampled occupancy.

```bash
mkdir -p "$BRIEF_RESULTS/pressure"
for ordering in original alphabetical random frequency schema_cost_weighted fp_tree_global; do
  output="$BRIEF_RESULTS/pressure/$ordering.json"
  test ! -e "$output"
  python scripts/locality_replay.py \
    --run-label "pressure-bfcl-k64-$ordering" \
    --partition bfcl --offset 0 --limit 200 --menu-size 64 \
    --ordering "$ordering" --random-seed 42 --support-mode disjoint \
    --condition empirical \
    --condition uniform \
    --condition skewed \
    --condition session_bursty \
    --require-peak-kv-usage 0.90 \
    --output "$output"
done
```

Only `empirical` and `session_bursty` are matched permutations. `uniform` and
`skewed` resample with replacement and are distribution stress tests, not a
paired locality comparison. If pressure is not reached, quarantine that run and
ask for a newly predeclared rerun; do not lower the threshold after inspection.

## 8. Validate, archive, and hand back

Before summarizing, confirm all replay counter validations and audit validations
are clean, all declared files exist, all pressure conditions reached the 0.90
threshold, and the k64 direct-partial strata are present. Any exception must be
listed in the handover.

Copy only compact artifacts into the tracked report directory:

```bash
export TRACKED_HANDOVER="$FYP_REPO/reports/initial-brief-closure/$BRIEF_STAMP"
test ! -e "$TRACKED_HANDOVER"
mkdir -p "$TRACKED_HANDOVER"
cp "$BRIEF_RESULTS/environment.txt" "$TRACKED_HANDOVER/"
cp "$BRIEF_RESULTS/cache-config.json" "$TRACKED_HANDOVER/"
cp "$BRIEF_RESULTS/models.json" "$TRACKED_HANDOVER/"
cp "$BRIEF_RESULTS/summaries/"*.json "$TRACKED_HANDOVER/"
```

Raw token IDs and per-request records are intentionally not committed. Archive
the complete unique result directory and record its checksum:

```bash
export BRIEF_ARCHIVE="/home/taghan/initial-brief-closure-$BRIEF_STAMP.tar.gz"
test ! -e "$BRIEF_ARCHIVE"
tar -C "$FYP_REPO/cluster/results" -czf "$BRIEF_ARCHIVE" "initial-brief-closure-$BRIEF_STAMP"
sha256sum "$BRIEF_ARCHIVE" | tee "$TRACKED_HANDOVER/raw-archive.sha256"
```

Write `HANDOVER.md` in `$TRACKED_HANDOVER` with the exact commit, result count,
validation outcome, missing/quarantined runs, raw archive path, and checksum.
Do not write scientific conclusions in that handover.

Create a separate result branch so the local analysis session can review it:

```bash
cd "$FYP_REPO"
git switch -c "gpu/initial-brief-closure-$BRIEF_STAMP"
git add "$TRACKED_HANDOVER"
git commit -m "Record initial-brief GPU closure handover"
git push -u origin HEAD
git log -1 --oneline
```

Return the result branch name, commit hash, archive path, checksum, and every
failed condition to the local Codex session. The local session—not the executor—
will validate the evidence, update the reports, and decide whether the brief is
closed.
