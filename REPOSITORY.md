# Repository map

What every tracked file is for, and which ones must not be moved or deleted.

`README.md` covers setup and commands; `PROJECT_STATUS.md` covers task-by-task
progress. This file covers **layout and purpose** — one line per file.

Audited 2026-08-10: every tracked file below has at least one inbound
reference, except where explicitly noted.

## Root

| File | Purpose |
| --- | --- |
| `initial-research-brief.md` | the anchor — every question the project must answer |
| `AGENTS.md` | working rules for agent sessions |
| `README.md` | setup, commands, main outputs |
| `PROJECT_STATUS.md` | task-by-task completion state |
| `REPOSITORY.md` | this file |
| `pyproject.toml` | package metadata and dependencies (`uv sync`) |
| `.gitignore` | excludes datasets, raw results, rendered HTML |
| `task_b_e.md` / `task_b_e.pdf` | Task B/E deliverable and its print version |
| `task_c_d.md` / `task_c_d.pdf` | Task C/D deliverable and its print version |

## `runbooks/` — GPU runbooks

Predeclared protocols executed on the NUS server; each fixes conditions, models
and acceptance criteria before measurement. **All eight are executed; none is
pending.** See `runbooks/README.md` for the status table, including the one
protocol that was only partially executed before being superseded.

Handover and findings documents name these in prose rather than linking to them,
so they were moved out of the repository root on 2026-08-10 with only one link
to update.

| Runbook | Purpose |
| --- | --- |
| `runbooks/NUS_GPU_AGENT_INSTRUCTIONS.md` | run ToolTrie-v0 |
| `runbooks/NUS_GPU_PHASE2_INSTRUCTIONS.md` | external comparison: CacheWeaver, fitted policies, SGLang |
| `runbooks/NUS_GPU_BRIEF_CLOSURE_INSTRUCTIONS.md` | close the initial research brief |
| `runbooks/NUS_GPU_PRESSURE_RERUN_INSTRUCTIONS.md` | controlled-cache pressure rerun |
| `runbooks/NUS_GPU_CONTEXTPILOT_CONFIRMATION_INSTRUCTIONS.md` | ContextPilot confirmation at alpha=0.001 |
| `runbooks/NUS_GPU_CONTEXTPILOT_STATIC_REFIT_RESUME_INSTRUCTIONS.md` | resume the quarantined static-refit cells |
| `runbooks/NUS_GPU_SGLANG_COUNTER_AUDIT_INSTRUCTIONS.md` | CPU-only audit of historical SGLang counters |
| `runbooks/NUS_GPU_CONTEXTPILOT_DUAL_MODEL_INSTRUCTIONS.md` | Qwen3-4B primary + Qwen3-0.6B replication |

## `src/tatm/` — the library

| Module | Purpose |
| --- | --- |
| `models.py` | `CanonicalTool` and `TaskRecord` — the core records |
| `io.py` | JSON/JSONL read and write helpers (most-imported module) |
| `download.py` | fetch public ToolRet and BFCL inputs |
| `datasets.py` | parse downloaded parquet/JSON into task records |
| `normalize.py` | canonicalize heterogeneous tool schemas |
| `serialization.py` | deterministic schema serialization and token counting |
| `prompting.py` | build OpenAI-format tool menus; apply an ordering |
| `analysis.py` | offline corpus analysis: locality, co-occurrence, analytical trie |
| `retrieval.py` | BM25 retrieval over tool documents |
| `tooltrie.py` | **ToolTrie-v0 planner** — causal recent-path ordering |
| `tooltrie_v1.py` | reorders only what the trie matched; unmatched tools keep their incoming order |
| `tooltrie_weighted.py` | visit-weighted variant — a proposal, never run |
| `baselines.py` | ordering baselines: CacheWeaver, fitted policies, online frequency |
| `contextpilot_adapter.py` | validation and materialization for ContextPilot orderings |
| `vllm_client.py` | vLLM HTTP client, Prometheus counter parsing, cache reset |
| `sglang_client.py` | equivalent helpers for an SGLang server |
| `measurement.py` | per-request measurement projection and reuse bucketing |
| `prefix_evidence.py` | exact common-prefix and token-block evidence |
| `replay_summary.py` | engine-aware summaries for repeated ordering replays |
| `pressure_summary.py` | validate and compact the controlled-pressure matrix |
| `bfcl_score.py` | BFCL-style AST scoring of tool calls |
| `paired_quality.py` | paired task-clustered bootstrap for quality comparisons |
| `stats.py` | small-sample statistics (mean, 95% Student-t half-width) |
| `reporting.py` | render Markdown/CSV analysis reports |

## `scripts/` — CLI entry points

**Data and pipeline**

| Script | Purpose |
| --- | --- |
| `download_datasets.py` | download the public ToolRet and BFCL inputs |
| `run_pipeline.py` | normalize datasets and generate the local analysis reports |
| `render_task_report.py` | render task Markdown to standalone HTML |

**Workload builders** — each emits a JSONL workload with one ordering applied

| Script | Purpose |
| --- | --- |
| `build_cluster_workload.py` | deterministic OpenAI-compatible workloads (original/alphabetical) |
| `build_retrieved_tool_workload.py` | menus whose membership comes from BM25 retrieval |
| `build_bfcl_quality_workload.py` | BFCL workload stratified across all five categories |
| `build_tooltrie_workload.py` | causal ToolTrie-v0 reordering |
| `build_fitted_ordering_workload.py` | frozen training-only frequency / co-occurrence baselines |
| `build_frequency_online_workload.py` | causal online presence-frequency ordering |
| `build_pair_triple_online_workload.py` | causal online pair/triple co-occurrence ordering |
| `build_contextpilot_workload.py` | labelled ContextPilot-derived orderings |

**Replay and measurement**

| Script | Purpose |
| --- | --- |
| `replay_vllm_workload.py` | replay a workload against vLLM (most-used script) |
| `replay_sglang_workload.py` | replay against stock SGLang/RadixAttention |
| `inspect_vllm_server.py` | read back vLLM's live prefix-cache configuration |
| `run_contextpilot_dual_model.py` | drive one model arm of the dual-model protocol |
| `locality_replay.py` | does request ordering change reuse on the live cache |
| `prefill_sweep.py` | catalog size at which prefix caching starts to pay |
| `vllm_prefix_cache_probe.py` | black-box exact-prefix sanity probe |
| `compare_probe_runs.py` | cache-enabled probe against a cache-disabled control |

**Scoring, summarizing, comparing**

| Script | Purpose |
| --- | --- |
| `score_bfcl_quality.py` | score a replayed BFCL workload against ground truth |
| `compare_bfcl_quality.py` | paired bootstrap comparison of two score files |
| `summarize_ordering_replays.py` | summarize repeated replays with 95% intervals |
| `summarize_pressure_replays.py` | validate and compact the pressure matrix |
| `analyze_contextpilot_dual_model.py` | CPU-only quality comparisons after servers stop |
| `validate_reuse_estimate.py` | analytical trie estimate against measured cache hits |

**Audits** — fail-closed checks; each verifies evidence rather than producing it

| Script | Purpose |
| --- | --- |
| `audit_contextpilot_dual_model.py` | fail-closed audit of the 190-replay matrix |
| `audit_contextpilot_dual_model_compact.py` | audit the git-tracked dual-model evidence |
| `audit_contextpilot_confirmation.py` | audit the tracked confirmation evidence |
| `audit_contextpilot_static_refit_compact.py` | audit static-refit and SGLang handover artifacts |
| `audit_frequency_online_compact.py` | audit the tracked `frequency_online` proposal |
| `audit_frequency_online_structure.py` | reconstruct the menu structure behind that result |
| `audit_phase2_fitted_equivalence.py` | reconstruct Phase-2 fitted-policy ordering equivalence |
| `audit_pressure_ordering_equivalence.py` | reconstruct and hash pressure ordering sequences |
| `audit_qwen_tokenizer_compatibility.py` | verify pinned models share tokenizer and chat template |
| `audit_rendered_prefix.py` | capture server-rendered tokens, block boundaries, measured reuse |
| `audit_pair_triple_information.py` | how much room a pair key has beyond frequency (CPU only) |

## `reports/` — the evidence chain

Three kinds of document, and the distinctions matter:

- **`findings.md`** — the analysis session's interpretation of a run.
- **`<stamp>/HANDOVER.md`** — the GPU executor's record of what was executed:
  exact commits, model revisions, server commands, live capacities, acceptance
  counts, deviations, and the raw archive SHA-256.
- **`PREDECLARATION.md`** — conditions, predictions and stop conditions fixed
  and committed *before* measurement, so a result can be checked against what
  was promised rather than what was found. Present in `frequency-online/` and
  `tooltrie-pressure/`.

**Handovers are load-bearing and must not be deleted.** `consolidated-report.md`
cites them in a seven-row provenance table, `brief-questions-and-answers.md`
cites two, and several `findings.md` files link to them directly. They are also
the only in-repo record of server commands and archive hashes for runs whose raw
data lives outside Git.

| Directory | Run |
| --- | --- |
| `tooltrie-v0/` | first ToolTrie-v0 GPU measurement |
| `tooltrie-phase2/` | external comparison: CacheWeaver, fitted policies, SGLang |
| `initial-brief-closure/` | brief-closure run |
| `initial-brief-pressure-rerun/` | controlled-cache pressure rerun |
| `contextpilot-confirmation/` | ContextPilot confirmation at alpha=0.001 |
| `contextpilot-quality-4b/` | Qwen3-4B quality addendum |
| `contextpilot-static-refit-resume/` | static-refit cells + SGLang counter audit |
| `contextpilot-dual-model/` | **accepted 190-replay 4B/0.6B matrix** |
| `frequency-online/` | causal online-frequency proposal |
| `tooltrie-pressure/` | ToolTrie-v0 under the 480-block cache budget |
| `tooltrie-weighted/` | weighted trie and online pair/triple |
| `pair-triple-information/` | CPU audit behind the §7 Q5 answer |

Top level of `reports/`: **`key-findings.md`** (start here — one-page summary),
`consolidated-report.md` (full synthesis, evidence and caveats),
`brief-questions-and-answers.md` (every brief question, its answer, its status),
`initial-findings.md`, `dataset-inventory.md` (Task C), `access-patterns.md`
(Task D), `analysis-summary.json`, `retrieval-bm25-sweep.json`, and `tables/`
(five CSVs: ordering results, schema issues, top pairs, tool frequency, workload
locality).

## `cluster/`, `notes/`, `tests/`

| Path | Purpose |
| --- | --- |
| `cluster/README.md` | exact steps for running Task B/E on a CUDA vLLM server |
| `cluster/*-manifest.json` | predeclared experiment manifests (conditions, models, acceptance) |
| `notes/reading-note.md` | Task A reading note and request-flow diagram |
| `notes/tooltrie-v0-design.md` | how ToolTrie-v0 works, with the code — linked from `reports/key-findings.md` |
| `notes/exact-tooltrie-proposal.md` | early design proposal, 2026-07-31, superseded by `tooltrie-v0-design.md` |
| `notes/open-questions.md` | early risk list, 2026-07-31 — **no inbound references**, and several entries have since been answered; read `brief-questions-and-answers.md` for current status |
| `tests/` | pytest suite, 129 tests |

## Not in Git, and how to get it back

- `data/raw/`, `data/processed/`, `data/tokenizers/` — run
  `scripts/download_datasets.py` then `scripts/run_pipeline.py`.
- `cluster/results/` — raw replay JSON. Preserved as `tar.gz` archives outside
  the repository, each with a recorded SHA-256. **These archives are the only
  copy of the per-request data**; committed reports carry compact summaries only.
- Rendered HTML (`task_b_e.html`, `task_c_d.html`,
  `reports/initial-findings.html`) — run `scripts/render_task_report.py`. The
  `.pdf` versions are tracked because they are exported by hand from the browser
  and no script reproduces them.

## Conventions

- One directory per run, `<experiment>/<timestamp>/`, never reused.
- Raw outputs stay out of Git; compact summaries are committed.
- Every accepted run records its FYP commit, model revision, server command,
  live cache capacity, and archive SHA-256.
- Policy names are declared in a manifest before execution. Two label/behaviour
  mismatches have been found historically, so treat a policy name as a claim to
  check against the emitted ordering, not as a guarantee.
