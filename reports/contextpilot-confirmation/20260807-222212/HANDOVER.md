# ContextPilot confirmation run — GPU executor handover

**Run stamp** `20260807-222212`
**Branch** `gpu/contextpilot-confirmation-20260807-222212`
**FYP commit executed** `ece2b267fe9411c861bc6cbcb7f52a04ad0ad656` (`tooltrie-v0-workflow`)
**Runbook** `NUS_GPU_CONTEXTPILOT_CONFIRMATION_INSTRUCTIONS.md` at that commit
**ContextPilot upstream** `1fa0a143fdeda344585666648ab2b30cb7fea77f`, `alpha=0.001`

## Raw archive

```
/home/taghan/contextpilot-confirmation-20260807-222212.tar.gz
sha256 aa4e196d140dc8d478ea8fd8b9da069e460779b4c888647ba228a7a60ec4a6bd
```

32 MB compressed, 325 MB expanded, 171 entries, verified with `tar -tzf`. Also
recorded in `raw-archive.sha256`. The archive holds every replay JSON, every
workload JSONL, both server logs, and the full 800-row score files. Only compact
summaries are committed here. **The archive is on the GPU server only and is not
backed up off-machine.**

## Counts

| | |
| --- | --- |
| Accepted replays | **75** — 72 systems (Qwen3-0.6B) + 3 quality (Qwen3-8B) |
| Failed requests | 0 |
| Counter validation | `clean` on all 75 |
| Cache reset before every replay | yes, all 75 |
| `--allow-counter-mismatch` | never passed to any run |
| Quarantined arms | **1** — `contextpilot_static_refit_causal`, 0 replays |

## What was NOT executed

**`contextpilot_static_refit_causal` never ran.** It is blocked against pinned
upstream: `ContextIndex.fit_transform` returns contexts in ContextPilot's
internal integer-ID space and never calls its own `_convert_to_str` inverse, so
the adapter's set-membership guard fails on every record. Full diagnosis,
reproduction, scope, and an unapplied proposed fix are in
`STATIC_REFIT_DEFECT.md`. The defect was preserved, not worked around; nothing
was patched in `src/tatm/contextpilot_adapter.py`.

Note for whoever owns that adapter: `tests/test_contextpilot_adapter.py:70` uses
a `FakeIndex` that echoes back input strings, so "104 tests pass" is not
evidence that this arm works against real upstream.

**No eviction-feedback arm and no annotation arm were executed.** Neither was in
the runbook and neither exists in the harness.

## Naming — read before quoting these numbers

`contextpilot-online_incremental` is **ContextPilot persistent-API adaptation
(alpha=0.001; no eviction feedback or annotations)**. It drives one persistent
`contextpilot.server.live_index.ContextPilot.reorder` instance across the
request stream. It is **ordering only**. It does not use cache-eviction
feedback and inserts no relevance annotations. **It is not the full ContextPilot
system and must not be described as such in any report.**

## Section 3 — systems matrix (Qwen3-0.6B, 200 requests × 3 trials per cell)

Reuse = `vllm:prompt_tokens_cached / (vllm:prompt_tokens_cached +
vllm:request_prefill_kv_computed_tokens_sum)`, from counter deltas. All six
summaries pass the three equivalence guards (same case set, same request
sequence, same selected tool sets across conditions).

### Padded 64-tool positive controls

| Workload | original | alphabetical | tooltrie_v0 | contextpilot online |
| --- | --- | --- | --- | --- |
| BFCL | 1.19% | 38.13% | 87.19% | **96.16%** |
| ToolRet | 13.87% | 51.05% | 83.58% | **95.27%** |

Zero variance across all three trials in every cell.

### BM25 retrieved menus (ToolRet)

| k | original | alphabetical | tooltrie_v0 | contextpilot online |
| --- | --- | --- | --- | --- |
| 4 | 15.87% | 15.28% | 17.48% | **18.72%** |
| 16 | 6.12% | 6.27% | 7.77% | **9.93%** |
| 64 | 0.91% | 1.24% | 1.90% | **4.78%** |
| 128 | 0.37% | 0.58% | 1.13% | **1.99%** |

**The gap between the two blocks is the finding.** With corrected `alpha=0.001`
and the persistent `reorder()` API, BFCL padded-64 reproduces the historical
96.16% *exactly*, and ToolRet gives 95.27% against a historical 94.82%. On
genuinely varying BM25-retrieved menus the same ordering yields 1.99–18.72%.
The 95–96% figure is a property of padded menus that share 60–63 of 64 tools,
not of ContextPilot ordering under retrieval.

Per-condition TTFT with 95% Student-t half-widths is in `acceptance-table.txt`;
full distributions are in the committed `*-summary.json` files.

## Section 4 — BFCL quality (Qwen3-8B, n=800, single pass)

640 relevance + 160 irrelevance cases, `--max-tokens 128 --disable-thinking
--reset-before`. All three replays: `request_count` 800, 0 failures,
`counter_validation.clean` true, model `Qwen/Qwen3-8B`.

| Condition | full | name | no_tool |
| --- | --- | --- | --- |
| alphabetical | 76.41% | 82.81% | 89.38% |
| tooltrie_v0 | 77.66% | 83.75% | 87.50% |
| contextpilot online | 78.44% | 84.84% | 84.38% |

Paired task-clustered bootstrap, 50 000 samples, seed 42:

| Comparison | metric | Δ pp | 95% CI | n |
| --- | --- | --- | --- | --- |
| cp-online vs alphabetical | name | +2.03 | [+0.47, +3.59] | 640 |
| cp-online vs alphabetical | full | +2.03 | [+0.16, +3.91] | 640 |
| cp-online vs alphabetical | no_tool | **−5.00** | [−8.75, −1.88] | 160 |
| cp-online vs tooltrie_v0 | name | +1.09 | [+0.00, +2.19] | 640 |
| cp-online vs tooltrie_v0 | full | +0.78 | [−0.31, +1.88] | 640 |
| cp-online vs tooltrie_v0 | no_tool | **−3.12** | [−6.25, −0.62] | 160 |
| tooltrie_v0 vs alphabetical | name | +0.94 | [−0.47, +2.34] | 640 |
| tooltrie_v0 vs alphabetical | full | +1.25 | [−0.47, +3.12] | 640 |
| tooltrie_v0 vs alphabetical | no_tool | −1.88 | [−4.38, +0.00] | 160 |

**No `--equivalence-margin-pp` was declared, so none of these is an equivalence
result.** They are estimation only. Any equivalence claim needs a
supervisor-approved margin declared before the numbers are looked at — which has
now been violated for this data, so a fresh run would be required to support one.

The one result worth flagging: `contextpilot-online_incremental` gains on
relevance metrics but its **irrelevance (no-tool) accuracy drops 5.00 pp against
alphabetical**, CI excluding zero, with 8 discordant pairs all in one direction
(0 cases recovered). Reordering appears to make the 8B model more willing to
call a tool when it should decline. `tooltrie_v0` shows the same sign at −1.88 pp
with a CI touching zero. This is ordering-only quality — no ContextPilot
relevance annotation was inserted in any arm.

## Exact server commands

Both servers ran alone on **GPU 2** (RTX 3090, driver 580.173.02), one server at
a time, one isolated replay client. GPUs 0/1/3 belonged to other users and were
never touched; no `sudo` was used.

Qwen3-0.6B — `capacity_tokens` 190 896 (`block_size` 16 × `num_gpu_blocks` 11 931):

```
PATH=/home/taghan/tatm/.venv/bin:$PATH \
VLLM_SERVER_DEV_MODE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
CUDA_VISIBLE_DEVICES=2 \
CPATH=/home/taghan/miniconda3/envs/hdr312/include/python3.12 \
/home/taghan/tatm/.venv/bin/vllm serve Qwen/Qwen3-0.6B \
  --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser hermes \
  --host 127.0.0.1 --port 8000
```

Qwen3-8B — `capacity_tokens` 44 656 (`block_size` 16 × `num_gpu_blocks` 2 791),
`max_model_len` 40 960:

```
PATH=/home/taghan/tatm/.venv/bin:$PATH \
VLLM_SERVER_DEV_MODE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
CUDA_VISIBLE_DEVICES=2 \
CPATH=/home/taghan/miniconda3/envs/hdr312/include/python3.12 \
/home/taghan/tatm/.venv/bin/vllm serve Qwen/Qwen3-8B \
  --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser hermes \
  --host 127.0.0.1 --port 8000
```

vLLM 0.26.0, unmodified. No patch to vLLM, CUDA, attention, or KV-cache
internals. `enable_prefix_caching` confirmed true on both
(`vllm-cache-config.json`, `vllm-cache-config-8b.json`). Both servers were shut
down by explicit PID after their sections; GPU 2 returned to 1 MiB.

## Deviations, all recorded before execution

1. **venv bootstrapper** — `uv venv` instead of `python3.12 -m venv`; this
   machine's system python3.12 has no `ensurepip` and the documented fix needs
   `sudo`, which project rules forbid. Same interpreter, same pinned commit,
   same editable target. `environment-deviations.txt`, `contextpilot-pip-freeze.txt`.
2. **Quality comparison inputs** — the runbook's §4 comparison block fails as
   written, because `compare_bfcl_quality.py` has no domain filter while
   irrelevance and relevance rows carry different metric keys. Metric-scoped
   *views* of the unmodified score files were written into the results directory
   and passed to the unmodified comparison script. Case counts match the
   runbook's declared 640/160 exactly. `QUALITY_COMPARISON_SCOPING.md`.
3. **Dataset symlinks** — `data/{processed,raw,tokenizers}` in the confirmation
   worktree symlink to `/home/taghan/FYP/data/*`. Read-only inputs; nothing
   modified, nothing committed.

Declared alpha, workloads, condition labels, single-GPU/single-server policy and
acceptance conditions were **not** changed.

## Files in this directory

- `acceptance-table.txt` — every accepted number in one plain-text table
- `HANDOVER.md` — this file
- `STATIC_REFIT_DEFECT.md` — the quarantined arm
- `QUALITY_COMPARISON_SCOPING.md` — deviation 2
- `environment-deviations.txt` — deviation 1
- `*-summary.json` — 6 systems summaries (3 trials aggregated, with guards)
- `*-contextpilot-online_incremental-summary.json` — 7 ordering-build summaries
- `*-retrieval-metrics.json` — BM25 retrieval quality at k=4/16/64/128
- `quality-scores-compact.json` — overall + by-domain for the 3 quality conditions
- `quality-*-vs-*.json` — 9 paired bootstrap comparisons
- `vllm-cache-config*.json`, `server-command-*.txt`, `gpu-environment.csv`,
  `fyp-git-commit.txt`, `contextpilot-git-commit.txt`, `contextpilot-pip-freeze.txt`
- `raw-archive.sha256`

Per project rules this session did not edit `PROJECT_STATUS.md` or any
scientific report.
