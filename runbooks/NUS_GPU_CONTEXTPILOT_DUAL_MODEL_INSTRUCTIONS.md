# NUS GPU instructions — Qwen3-4B primary and Qwen3-0.6B replication

This is the supervisor-requested dual-model replication. **Qwen3-4B is the
primary result.** Qwen3-0.6B is retained as the small-model systems replication
requested by the initial brief. Read this file and
`cluster/contextpilot-dual-model-manifest.json` completely before execution.

This is a new, self-contained experiment. Do not overwrite or silently merge
the earlier confirmation directories. The accepted tables contain:

- two models, run one at a time on the same idle physical GPU;
- six systems workloads;
- five ordering conditions;
- three reset trials per systems cell: 90 systems replays per model;
- one n=800 BFCL quality replay per condition: five quality replays per model;
- 190 accepted GPU replays in total, plus one excluded startup warmup per model.

This supervisor-requested run replaces neither the raw historical artifacts nor
the old runbook's one missing Qwen3-8B static-refit quality cell. The latter can
remain documented as an unfinished historical extension; it is not part of the
new 4B-primary result.

The two models use their native live KV-cache capacities. ToolTrie is rebuilt
from each live capacity. Therefore report two separate within-model tables:
do not pool the models, compare absolute TTFT across models, or describe their
difference as a pure model-size effect.

## 1. Immutable inputs and clean worktree

Replace `REPLACE_WITH_HANDOFF_COMMIT` with the exact commit supplied by the
analysis session. Never infer it from a moving branch.

```bash
FYP_REPO=/home/taghan/FYP
DUAL_COMMIT=REPLACE_WITH_HANDOFF_COMMIT
SOURCE_RESULTS=/home/taghan/contextpilot-confirmation-20260807-222212
DUAL_STAMP=$(date +%Y%m%d-%H%M%S)
DUAL_WORKTREE="/home/taghan/FYP-contextpilot-dual-model-$DUAL_STAMP"
DUAL_ROOT="/home/taghan/contextpilot-dual-model-$DUAL_STAMP"
SHARED_WORKLOADS="$DUAL_ROOT/shared-workloads"
Q4B_RESULTS="$DUAL_ROOT/qwen3-4b"
Q06B_RESULTS="$DUAL_ROOT/qwen3-0.6b"
REFERENCE_ROOT=/home/taghan/external-references
CONTEXTPILOT_REPO="$REFERENCE_ROOT/ContextPilot-1fa0a143"
CONTEXTPILOT_VENV="/home/taghan/venvs/contextpilot-dual-$DUAL_STAMP"
Q4B_REVISION=1cfa9a7208912126459214e8b04321603b3df60c
Q06B_REVISION=c1899de289a04d12100db370d81485cdf75e47ca

git -C "$FYP_REPO" fetch origin
git -C "$FYP_REPO" worktree add --detach "$DUAL_WORKTREE" "$DUAL_COMMIT"
cd "$DUAL_WORKTREE"
test "$(git rev-parse HEAD)" = "$DUAL_COMMIT"
test -z "$(git status --porcelain)"
test ! -e "$DUAL_ROOT"
mkdir -p "$SHARED_WORKLOADS"
cp cluster/contextpilot-dual-model-manifest.json "$DUAL_ROOT/protocol-manifest.json"
uv sync
uv run pytest -q
test -d "$SOURCE_RESULTS"
```

If the source directory is absent, restore the verified archive
`/home/taghan/contextpilot-confirmation-20260807-222212.tar.gz` to a new path.
Never modify the original archive or result directory.

## 2. Pin ContextPilot and build shared workloads

Use the same upstream ContextPilot revision and paper/default alpha as the
accepted confirmation. The server previously lacked `ensurepip`, so use
`uv venv`.

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
uv pip freeze --python "$CONTEXTPILOT_VENV/bin/python" \
  > "$DUAL_ROOT/contextpilot-pip-freeze.txt"

/home/taghan/tatm/.venv/bin/python \
  scripts/audit_qwen_tokenizer_compatibility.py \
  --output "$DUAL_ROOT/tokenizer-compatibility.json"
```

The tokenizer audit must report both `tokenizer_json_identical` and
`chat_template_identical` as true. ToolTrie capacity is expressed in schema
tokens originally counted with the 0.6B tokenizer; stop and retokenize per
model if this compatibility check fails.

Copy the exact accepted original and alphabetical workloads. This prevents
dataset regeneration from becoming a hidden difference.

```bash
for STEM in bfcl-padded64 toolret-padded64 \
  toolret-bm25-k4 toolret-bm25-k16 toolret-bm25-k64 toolret-bm25-k128; do
  test -f "$SOURCE_RESULTS/$STEM-original.jsonl"
  test -f "$SOURCE_RESULTS/$STEM-alphabetical.jsonl"
  cp "$SOURCE_RESULTS/$STEM-original.jsonl" "$SHARED_WORKLOADS/$STEM-original.jsonl"
  cp "$SOURCE_RESULTS/$STEM-alphabetical.jsonl" "$SHARED_WORKLOADS/$STEM-alphabetical.jsonl"
done
test -f "$SOURCE_RESULTS/quality-original.jsonl"
test -f "$SOURCE_RESULTS/quality-alphabetical.jsonl"
cp "$SOURCE_RESULTS/quality-original.jsonl" "$SHARED_WORKLOADS/quality-original.jsonl"
cp "$SOURCE_RESULTS/quality-alphabetical.jsonl" "$SHARED_WORKLOADS/quality-alphabetical.jsonl"
```

Build both ContextPilot ordering-only adaptations once. Their emitted workloads
are shared byte-for-byte by the two model arms.

```bash
for STEM in bfcl-padded64 toolret-padded64 \
  toolret-bm25-k4 toolret-bm25-k16 toolret-bm25-k64 toolret-bm25-k128 quality; do
  for MODE in static_refit_causal online_incremental; do
    "$CONTEXTPILOT_VENV/bin/python" scripts/build_contextpilot_workload.py \
      --input "$SHARED_WORKLOADS/$STEM-original.jsonl" \
      --contextpilot-repo "$CONTEXTPILOT_REPO" \
      --mode "$MODE" --alpha 0.001 \
      --output "$SHARED_WORKLOADS/$STEM-contextpilot-$MODE.jsonl" \
      --summary-output "$SHARED_WORKLOADS/$STEM-contextpilot-$MODE-summary.json"
  done
done

uv run python scripts/run_contextpilot_dual_model.py \
  --shared-workloads "$SHARED_WORKLOADS" \
  --model Qwen/Qwen3-4B --model-revision "$Q4B_REVISION" \
  --preflight-only
find "$SHARED_WORKLOADS" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$DUAL_ROOT/shared-workload-sha256.txt"
```

The preflight must pass. It verifies request counts, request sequence, selected
membership, ordering labels, builder hashes, ContextPilot commit, alpha, API
mode, and the ordering-only scope.

## 3. Run the primary Qwen3-4B arm first

Use one idle physical GPU, one server, and one sequential client. Do not run any
other replay driver against port 8000. In the server pane:

```bash
GPU_ID=REPLACE_WITH_IDLE_GPU
PATH=/home/taghan/tatm/.venv/bin:$PATH \
VLLM_SERVER_DEV_MODE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
CPATH=/home/taghan/miniconda3/envs/hdr312/include/python3.12 \
/home/taghan/tatm/.venv/bin/vllm serve Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser hermes \
  --host 127.0.0.1 --port 8000 \
  2>&1 | tee "$DUAL_ROOT/vllm-qwen3-4b.log"
```

In the client pane, export the paths from sections 1–2 and run exactly one
driver:

```bash
cd "$DUAL_WORKTREE"
curl -fsS http://127.0.0.1:8000/v1/models >/dev/null
uv run python scripts/run_contextpilot_dual_model.py \
  --shared-workloads "$SHARED_WORKLOADS" \
  --output-dir "$Q4B_RESULTS" \
  --model Qwen/Qwen3-4B \
  --model-revision "$Q4B_REVISION"
```

The driver performs one excluded warmup followed by 90 systems and five
quality replays. It refuses an existing output directory, resets APC before
every replay, rejects counter mismatches, rotates condition execution order
across the three systems trials, and scores all five quality conditions. The
expensive bootstrap comparisons are deliberately deferred until after both GPU
servers stop.

If it exits nonzero, preserve the entire 4B directory under a uniquely named
`$DUAL_ROOT/quarantine/` path. Do not edit or resume it. After diagnosing the
failure, rerun that model into a newly absent `$Q4B_RESULTS` path; a clean 4B
arm never has to be repeated merely because the later 0.6B arm failed.

## 4. Run the Qwen3-0.6B replication

Stop the 4B server completely and confirm the GPU memory is released. Start the
0.6B server on the **same physical GPU**:

```bash
PATH=/home/taghan/tatm/.venv/bin:$PATH \
VLLM_SERVER_DEV_MODE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
CPATH=/home/taghan/miniconda3/envs/hdr312/include/python3.12 \
/home/taghan/tatm/.venv/bin/vllm serve Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser hermes \
  --host 127.0.0.1 --port 8000 \
  2>&1 | tee "$DUAL_ROOT/vllm-qwen3-0.6b.log"
```

In the client pane:

```bash
cd "$DUAL_WORKTREE"
curl -fsS http://127.0.0.1:8000/v1/models >/dev/null
uv run python scripts/run_contextpilot_dual_model.py \
  --shared-workloads "$SHARED_WORKLOADS" \
  --output-dir "$Q06B_RESULTS" \
  --model Qwen/Qwen3-0.6B \
  --model-revision "$Q06B_REVISION"
```

Apply the same per-model quarantine rule to any failure. Do not substitute a
different model revision, lower an acceptance check, or pass
`--allow-counter-mismatch`.

## 5. CPU-only analysis and fail-closed audit

Stop the 0.6B server. Generate both sets of predeclared comparisons without
holding GPU memory. Four CPU workers run independent deterministic comparisons;
this does not change the bootstrap samples or seed.

```bash
cd "$DUAL_WORKTREE"
uv run python scripts/analyze_contextpilot_dual_model.py \
  --model-results "$Q4B_RESULTS" \
  --output-dir "$DUAL_ROOT/analysis/qwen3-4b" \
  --model Qwen/Qwen3-4B --workers 4
uv run python scripts/analyze_contextpilot_dual_model.py \
  --model-results "$Q06B_RESULTS" \
  --output-dir "$DUAL_ROOT/analysis/qwen3-0.6b" \
  --model Qwen/Qwen3-0.6B --workers 4
```

Each command must report 27 accepted comparisons. If CPU analysis fails,
preserve that analysis directory as quarantine; do not weaken the metadata or
bootstrap checks.

Record provenance and run the aggregate audit:

```bash
cd "$DUAL_WORKTREE"
git rev-parse HEAD > "$DUAL_ROOT/fyp-git-commit.txt"
git -C "$CONTEXTPILOT_REPO" rev-parse HEAD \
  > "$DUAL_ROOT/contextpilot-git-commit.txt"
/home/taghan/tatm/.venv/bin/python -c \
  'import importlib.metadata; print(importlib.metadata.version("vllm"))' \
  > "$DUAL_ROOT/vllm-version.txt"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv \
  > "$DUAL_ROOT/gpu-environment.csv"

uv run python scripts/audit_contextpilot_dual_model.py \
  --root "$DUAL_ROOT" \
  --expected-fyp-commit "$DUAL_COMMIT" \
  --output "$DUAL_ROOT/dual-model-audit.json"
```

Accept the run only if the audit reports:

```text
status = accepted
accepted_gpu_replays = 190
all_checks_passed = true
```

The audit independently reloads all 190 replay files, verifies model IDs,
request counts, resets, clean counters, six within-model equivalence summaries,
ten quality score files, 54 comparison files, and byte-identical
capacity-independent tool sequences across models.

## 6. Archive and hand over

```bash
DUAL_ARCHIVE="/home/taghan/contextpilot-dual-model-$DUAL_STAMP.tar.gz"
tar -czf "$DUAL_ARCHIVE" -C /home/taghan "$(basename "$DUAL_ROOT")"
tar -tzf "$DUAL_ARCHIVE" >/dev/null
sha256sum "$DUAL_ARCHIVE" | tee "$DUAL_ARCHIVE.sha256"
```

Create a new branch in the detached worktree. Commit only:

- a concise handover Markdown file;
- `dual-model-audit.json`;
- both `model-run-summary.json` files;
- the six systems summaries and compact quality summary for each model;
- the 54 comparison JSON files;
- small configuration/provenance files and the archive SHA-256.

Do not commit raw replay JSON/JSONL, logs, model weights, environments, or the
raw score rows. The handover must report the executed FYP and ContextPilot
commits, both model revisions, exact server commands, physical GPU, vLLM
version, native capacities, 190/190 acceptance count, archive path/entry
count/size/SHA-256, and every deviation or quarantined attempt.

The interpretation is fixed in advance: Qwen3-4B is primary; Qwen3-0.6B is a
replication. Compare policies within each model. Do not call either
ContextPilot arm the full system, and retain original-order selected-tool text
prefill as the fallback condition.
