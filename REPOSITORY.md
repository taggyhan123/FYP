# Repository map

What every tracked item is for, and which ones must not be moved or deleted.

`README.md` covers setup and outputs; `PROJECT_STATUS.md` covers task-by-task
progress. This file covers **layout** — where things live and why.

## Layout at a glance

```
initial-research-brief.md     the anchor: every question the project must answer
AGENTS.md                     working rules for agent sessions
README.md                     setup, commands, main outputs
PROJECT_STATUS.md             task-by-task completion state
REPOSITORY.md                 this file

NUS_GPU_*_INSTRUCTIONS.md     8 GPU runbooks (see below) — pinned by path, do not move
task_b_e.{md,pdf}             Task B/E deliverable + print version
task_c_d.{md,pdf}             Task C/D deliverable + print version

src/tatm/                     library: planners, clients, scoring, statistics
scripts/                      CLI entry points: build workloads, replay, score, audit
tests/                        pytest suite
notes/                        reading notes, open questions, design proposals
cluster/                      cluster README + predeclared experiment manifests
reports/                      committed evidence and findings (see below)
data/                         datasets — git-ignored, regenerate with scripts
cluster/results/              raw run outputs — git-ignored, archived to tar.gz
```

## The GPU runbooks

Each is a predeclared protocol executed on the NUS server. They are referenced
**by path** from `PROJECT_STATUS.md`, `README.md`, `cluster/README.md`, and from
handover records that state which runbook they executed. **Moving them would
either break those links or require editing evidence documents, so they stay at
the repository root.**

| Runbook | Purpose |
| --- | --- |
| `NUS_GPU_AGENT_INSTRUCTIONS.md` | run ToolTrie-v0 |
| `NUS_GPU_PHASE2_INSTRUCTIONS.md` | external-comparison phase (CacheWeaver, fitted policies, SGLang) |
| `NUS_GPU_BRIEF_CLOSURE_INSTRUCTIONS.md` | close the initial research brief |
| `NUS_GPU_PRESSURE_RERUN_INSTRUCTIONS.md` | controlled-cache pressure rerun |
| `NUS_GPU_CONTEXTPILOT_CONFIRMATION_INSTRUCTIONS.md` | ContextPilot confirmation at alpha=0.001 |
| `NUS_GPU_CONTEXTPILOT_STATIC_REFIT_RESUME_INSTRUCTIONS.md` | resume the quarantined static-refit cells |
| `NUS_GPU_SGLANG_COUNTER_AUDIT_INSTRUCTIONS.md` | CPU-only audit of historical SGLang counters |
| `NUS_GPU_CONTEXTPILOT_DUAL_MODEL_INSTRUCTIONS.md` | Qwen3-4B primary + Qwen3-0.6B replication |

## `reports/` — the evidence chain

Two kinds of document, and the distinction matters:

- **`findings.md`** — the analysis session's interpretation of a run.
- **`<stamp>/HANDOVER.md`** — the GPU executor's record of what was actually
  executed: exact commits, model revisions, server commands, capacities,
  acceptance counts, deviations, and the raw archive SHA-256.

**Handovers are load-bearing and must not be deleted.** They are cited from
`consolidated-report.md` (which carries a seven-row provenance table mapping
each handover to the evidence it certifies), from `brief-questions-and-answers.md`,
and by markdown link from several `findings.md` files. They are also the only
in-repo record of the server commands and archive hashes for runs whose raw data
lives outside Git.

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

Top-level in `reports/`: `consolidated-report.md` (main synthesis),
`brief-questions-and-answers.md` (every brief question with its answer and
status), `initial-findings.md`, `dataset-inventory.md`, `access-patterns.md`,
`analysis-summary.json`, `tables/`, `retrieval-bm25-sweep.json`.

## What is deliberately not in Git

- `data/raw/`, `data/processed/`, `data/tokenizers/` — regenerate with
  `scripts/download_datasets.py` then `scripts/run_pipeline.py`.
- `cluster/results/` — raw replay JSON. Preserved as `tar.gz` archives outside
  the repository, each with a recorded SHA-256. **These archives are the only
  copy of the per-request data**; the committed reports carry compact summaries
  only.
- Rendered report HTML (`task_b_e.html`, `task_c_d.html`,
  `reports/initial-findings.html`) — regenerate with
  `scripts/render_task_report.py`. The `.pdf` versions are kept because they are
  produced by hand from the browser and no script reproduces them.

## Conventions

- One directory per run, named `<experiment>/<timestamp>/`, never reused.
- Raw outputs stay out of Git; compact summaries are committed.
- Every accepted run records its FYP commit, model revision, server command,
  live cache capacity, and archive SHA-256.
- Policy names are declared in a manifest before execution. Two label/behaviour
  mismatches have been found historically, so treat a policy name as a claim to
  be checked against the emitted ordering, not as a guarantee.
