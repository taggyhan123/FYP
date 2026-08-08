# ContextPilot confirmation — reconciled findings

## Status

The **persistent-API adaptation** is measured and accepted. The corrected
**static-refit causal adaptation** has not yet been replayed and remains the
only missing cell in the predeclared confirmation matrix.

These labels are mandatory:

| Short name | Exact meaning |
| --- | --- |
| ContextPilot persistent-API adaptation | `ContextPilot.reorder`, `alpha=0.001`, one persistent index, causal, ordering only, no eviction feedback or relevance annotations |
| ContextPilot static-refit causal adaptation | a new `ContextIndex.fit_transform` over each observed request prefix, `alpha=0.001`, causal, ordering only |

Neither condition is the full ContextPilot system. The persistent arm uses the
upstream online ordering API, but workloads were planned before serving and no
vLLM eviction event was fed back into its index. It also omits ContextPilot's
relevance annotations and de-duplication path.

## Accepted systems result

All values below are server-counter cache ratios from stock vLLM 0.26.0 on
`Qwen/Qwen3-0.6B`, 200 requests and three reset trials per cell. All 72 systems
replays had clean counter validation; all comparison summaries confirmed the
same case set, request sequence, and selected tool membership.

### Padded 64-tool positive control

| Workload | Original | Alphabetical | ToolTrie-v0 | ContextPilot persistent API |
| --- | ---: | ---: | ---: | ---: |
| BFCL | 1.19% | 38.13% | 87.19% | **96.16%** |
| ToolRet | 13.87% | 51.05% | 83.58% | **95.27%** |

### Deterministic BM25-retrieved ToolRet menus

| Menu size | Original | Alphabetical | ToolTrie-v0 | ContextPilot persistent API |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 15.87% | 15.28% | 17.48% | **18.72%** |
| 16 | 6.12% | 6.27% | 7.77% | **9.93%** |
| 64 | 0.91% | 1.24% | 1.90% | **4.78%** |
| 128 | 0.37% | 0.58% | 1.13% | **1.99%** |

The persistent-API adaptation leads ToolTrie-v0 in every measured retrieved
cell, by 0.86–2.88 percentage points. However, both methods lose almost all of
their padded-menu reuse as independently retrieved menus become larger. The
95–96% result is therefore a **shared-menu positive-control result**, not a
realistic retrieval result. No paired TTFT contrast was predeclared, so the
systems run does not establish a retrieved-menu latency win.

## Accepted quality result

The quality arm contains 640 relevance and 160 irrelevance cases per condition.
These are single fixed request-sequence replays. Because both ToolTrie and the
ContextPilot planner depend on earlier requests, task bootstrap intervals are
descriptive for that emitted sequence; they do not quantify variation over
other request orders.

### Qwen3-8B

| Condition | Function name | Full call | No tool |
| --- | ---: | ---: | ---: |
| Alphabetical | 82.81% | 76.41% | **89.38%** |
| ToolTrie-v0 | 83.75% | 77.66% | 87.50% |
| ContextPilot persistent API | **84.84%** | **78.44%** | 84.38% |

Against alphabetical, the persistent arm is +2.03 points on function name,
+2.03 on full call, and −5.00 on no-tool accuracy. The fixed-sequence 95%
interval for the no-tool difference is [−8.75, −1.88]. This is evidence of an
8B trade-off on this sequence, not evidence that reordering universally causes
extra calls.

### Qwen3-4B addendum

| Condition | Function name | Full call | No tool |
| --- | ---: | ---: | ---: |
| Alphabetical | 82.03% | 73.28% | 85.62% |
| ToolTrie-v0 | 83.75% | 75.31% | **86.88%** |
| ContextPilot persistent API | **84.53%** | **77.19%** | 85.62% |

The relevance-side gain reproduces at 4B, but the 8B no-tool drop does not:
ContextPilot minus alphabetical is exactly 0.00 points, with a fixed-sequence
interval of [−4.38, +4.38]. The defensible conclusion is model-sensitive
evidence, not a universal safety penalty. More irrelevance cases and complete
planner-sequence replicates are needed to distinguish model dependence from
sampling noise.

## Legitimacy audit

The accepted persistent arm has the following safeguards:

- exact upstream commit `1fa0a143fdeda344585666648ab2b30cb7fea77f` and
  paper/default `alpha=0.001`;
- the official persistent `ContextPilot.reorder` API, with no future-request
  visibility;
- unchanged request order and selected tool sets;
- 75 accepted replays in total: 72 systems and three Qwen3-8B quality runs,
  with zero failed requests, clean server-counter validation, and a successful
  cache reset before every replay;
- one idle RTX 3090, one stock vLLM server, and one sequential client;
- builder summaries recording input/output hashes, planning time, API,
  information regime, and disabled full-system features.

The machine-readable local audit passes every invariant available from the
tracked compact artifacts: provenance, cache configuration, all six
equivalence guards, three-trial shapes, cached-plus-computed token identities,
and agreement between compact quality scores and all 18 comparison point
estimates. It deliberately reports `partial_requires_cluster_followup` because
compact artifacts cannot prove the raw archive and the missing cells below are
real.

Three limitations remain material:

1. The raw 32 MB archive has only one known copy on the GPU server.
2. The original comparison JSON predates the sequence-state metadata repair and
   incorrectly marks McNemar independence as satisfied. Point estimates and
   emitted model outputs are unchanged; the GPU-side raw score files must be
   re-analysed with `--sequence-state-dependent`.
3. The static-refit arm was quarantined before serving because pinned
   ContextPilot returned internal integer IDs. The adapter now restores caller
   IDs through the public, positionally aligned `IndexResult.original_contexts`
   field and has a real-upstream integration test, but its 18 systems trials and
   one Qwen3-8B quality replay still require GPU execution.

## Research conclusion

ToolTrie-v0 is a legitimate exact-prefix baseline, but it is **not the best
measured ordering policy**. The ContextPilot persistent-API adaptation produces
more reuse on both padded and BM25-retrieved menus and usually higher
relevance-side quality. ToolTrie's useful contribution is narrower: a simple,
bounded recent-path mechanism, a trustworthy exact-prefix measurement harness,
and evidence showing when cache-aware ordering ceases to matter because
retrieved menus have too little exact overlap.

The next design should not be framed as “ToolTrie beats ContextPilot.” A viable
extension must target a gap the stronger ordering does not solve, such as safe
inactive-tool retention with an active-tool manifest, while retaining ordinary
selected-tool text prefill as the predeclared fallback.

## Provenance

- Accepted 0.6B/8B handover:
  [`20260807-222212/HANDOVER.md`](20260807-222212/HANDOVER.md)
- Machine-readable local audit:
  [`local-audit.json`](local-audit.json)
- Qwen3-4B addendum:
  [`20260808-105833/HANDOVER.md`](../contextpilot-quality-4b/20260808-105833/HANDOVER.md)
- Static-refit defect record:
  [`STATIC_REFIT_DEFECT.md`](20260807-222212/STATIC_REFIT_DEFECT.md)
- Metric-scoping record:
  [`QUALITY_COMPARISON_SCOPING.md`](20260807-222212/QUALITY_COMPARISON_SCOPING.md)
- Static-refit resume protocol:
  [`NUS_GPU_CONTEXTPILOT_CONFIRMATION_INSTRUCTIONS.md`](../../NUS_GPU_CONTEXTPILOT_CONFIRMATION_INSTRUCTIONS.md)
