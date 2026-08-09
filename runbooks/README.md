# GPU runbooks

Predeclared protocols executed on the NUS server. Each one fixes the conditions,
models, workloads and acceptance criteria **before** any measurement, so that a
result can be checked against what was actually promised. They are the method
record for the corresponding report directory.

**All eight have been executed. None is pending.** They are kept because
"executed the runbook" is only a meaningful claim if the runbook still exists to
be read — and because they hold the predeclared acceptance criteria that make
the results auditable rather than merely reported.

| Runbook | Purpose | Results | Status |
| --- | --- | --- | --- |
| `NUS_GPU_AGENT_INSTRUCTIONS.md` | run ToolTrie-v0 | `reports/tooltrie-v0/` | executed |
| `NUS_GPU_PHASE2_INSTRUCTIONS.md` | external comparison: CacheWeaver, fitted policies, SGLang | `reports/tooltrie-phase2/` | executed |
| `NUS_GPU_BRIEF_CLOSURE_INSTRUCTIONS.md` | close the initial research brief | `reports/initial-brief-closure/` | executed |
| `NUS_GPU_PRESSURE_RERUN_INSTRUCTIONS.md` | controlled-cache pressure rerun | `reports/initial-brief-pressure-rerun/` | executed |
| `NUS_GPU_CONTEXTPILOT_CONFIRMATION_INSTRUCTIONS.md` | ContextPilot confirmation at alpha=0.001 | `reports/contextpilot-confirmation/` | executed |
| `NUS_GPU_CONTEXTPILOT_STATIC_REFIT_RESUME_INSTRUCTIONS.md` | resume the quarantined static-refit cells | `reports/contextpilot-static-refit-resume/` | **partially executed — superseded** |
| `NUS_GPU_SGLANG_COUNTER_AUDIT_INSTRUCTIONS.md` | CPU-only audit of historical SGLang counters | `reports/contextpilot-static-refit-resume/` | executed, 72/72 accepted |
| `NUS_GPU_CONTEXTPILOT_DUAL_MODEL_INSTRUCTIONS.md` | Qwen3-4B primary + Qwen3-0.6B replication | `reports/contextpilot-dual-model/` | executed, 190/190 accepted |

## The one partial execution

`NUS_GPU_CONTEXTPILOT_STATIC_REFIT_RESUME_INSTRUCTIONS.md` was superseded
mid-flight by the dual-model protocol, which required that the older runbooks
not run alongside it. Sections 1–4 completed — 18 Qwen3-0.6B static-refit
systems replays and one Qwen3-8B quality replay, all accepted. Section 5's
regeneration of the nine pre-existing 8B comparisons with
`--sequence-state-dependent`, and section 6's combined archive and handover,
were **not** run. See that run's `HANDOVER.md` for exactly what was and was not
completed.

## Reading order

For the current state of the project, start with the most recent:
`NUS_GPU_CONTEXTPILOT_DUAL_MODEL_INSTRUCTIONS.md` defines the accepted
190-replay result. The earlier runbooks are the path that led there and remain
the method record for their own results.

## Note on references

Handover and findings documents mention these runbooks by name in prose rather
than by link, so relocating them here did not invalidate any record: a document
stating that it executed `NUS_GPU_X_INSTRUCTIONS.md` still names the same
protocol. Prior to 2026-08-10 these files lived at the repository root, which is
where Git history and older handovers will show them.
