# Initial-brief controlled-cache pressure rerun

The predeclared replacement for the failed pressure criterion is
[`cluster/initial-brief-pressure-rerun-manifest.json`](../../cluster/initial-brief-pressure-rerun-manifest.json).
The executable GPU procedure is
[`NUS_GPU_PRESSURE_RERUN_INSTRUCTIONS.md`](../../NUS_GPU_PRESSURE_RERUN_INSTRUCTIONS.md).

This directory is reserved for a compact, independently validated GPU
handover. The original 24 low-occupancy runs remain preserved under
`reports/initial-brief-closure/20260805-222246-gpu-executor/`; they are not
overwritten or relabelled.

A result subdirectory is accepted only when
`scripts/summarize_pressure_replays.py` confirms all 24 regime-runs, all
predeclared cache and counter checks, at least 90% sampled occupancy, and a
positive sampled eviction count in every regime-run. Raw per-request outputs,
server logs, and metrics remain outside Git in a checksummed archive.

## Accepted result

`20260807-005414/` passes 24/24 regime-runs and 384/384 checks. Its peak
occupancy is 91.02-91.86%, every run records positive sampled evictions, and
all runs preserve one running and zero waiting requests. The execution record
is [`20260807-005414/HANDOVER.md`](20260807-005414/HANDOVER.md).

The GPU files did not retain ordered tool IDs. A deterministic local
reconstruction under the pinned manifest parameters is recorded in
[`ordering-equivalence.json`](ordering-equivalence.json): frequency,
schema-cost weighted, and FP-tree global emit identical sequences for every
request and regime, leaving four distinct orderings among six labels.

Regenerate it after the normalized datasets are available:

```bash
uv run python scripts/audit_pressure_ordering_equivalence.py \
  --output reports/initial-brief-pressure-rerun/ordering-equivalence.json
```

The result closes the pressure acceptance criterion. It remains a controlled
7,680-token stress test; cross-capacity latency comparison and claims of natural
production pressure are still forbidden.
